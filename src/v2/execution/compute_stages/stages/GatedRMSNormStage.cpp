/**
 * @file GatedRMSNormStage.cpp
 * @brief Implementation of gated RMS normalization stage
 */

#include "GatedRMSNormStage.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"

#include <cmath>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

// GPU kernel declarations
#ifdef HAVE_CUDA
extern "C" bool cudaGDN_gated_rmsnorm(
    const float *input, const float *gate, const float *gamma,
    float *output,
    int seq_len, int d_model, int norm_dim, int gamma_period,
    float eps, bool subtract_one, bool gate_silu,
    int device_idx, void *stream);
#endif
#ifdef HAVE_ROCM
extern "C" bool rocmGDN_gated_rmsnorm(
    const float *input, const float *gate, const float *gamma,
    float *output,
    int seq_len, int d_model, int norm_dim, int gamma_period,
    float eps, bool subtract_one, bool gate_silu,
    int device_idx, void *stream);
#endif

namespace llaminar2
{

    GatedRMSNormStage::GatedRMSNormStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool GatedRMSNormStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "GatedRMSNormStage"))
            return false;

        if (!ensureRequiredPointers("GatedRMSNormStage",
                                    {{"input", params_.input},
                                     {"gate", params_.gate},
                                     {"output", params_.output},
                                     {"gamma", const_cast<ITensor *>(params_.gamma)}}))
            return false;

        auto *input_base = requireTensorBasePtr(params_.input, "input");
        auto *gate_base = requireTensorBasePtr(params_.gate, "gate");
        auto *output_base = requireTensorBasePtr(params_.output, "output");
        auto *gamma_base = dynamic_cast<const TensorBase *>(params_.gamma);

        if (!input_base || !gate_base || !output_base || !gamma_base)
        {
            LOG_ERROR("[GatedRMSNormStage] Tensor cast failed");
            return false;
        }

        const int seq_len = (params_.seq_len > 0)
                                ? params_.seq_len
                                : static_cast<int>(input_base->rows());

        const size_t d_model = input_base->shape().size() > 1 ? input_base->shape()[1] : input_base->numel();

        // Determine normalization dimension. When norm_dim > 0, normalize
        // over chunks of norm_dim (per-head normalization). Otherwise, full d_model.
        const size_t norm_dim = (params_.norm_dim > 0)
                                    ? static_cast<size_t>(params_.norm_dim)
                                    : d_model;
        const float eps = params_.eps;
        const bool subtract_one = params_.subtract_one;
        const bool gate_silu = params_.gate_silu;

        // Gamma weight may be smaller than norm_dim (e.g., [d_v=128] with full-row
        // norm over value_dim=2048). In this case, gamma cycles every gamma_period elements.
        const size_t gamma_numel = gamma_base->numel();
        const size_t gamma_period = (gamma_numel < norm_dim) ? gamma_numel : norm_dim;

        // GPU dispatch path — do NOT call data()/mutable_data() before this check!
        // Those calls trigger D2H transfers when tensors are GPU-authoritative,
        // then the GPU path re-uploads them (wasteful round-trip).
        if (params_.device_id.is_gpu())
        {
            // Ensure all tensors are on device (gamma is a weight, not arena-managed)
            auto *input_mut = const_cast<TensorBase *>(input_base);
            auto *gate_mut = const_cast<TensorBase *>(gate_base);
            auto *gamma_mut = const_cast<TensorBase *>(gamma_base);
            input_mut->ensureOnDevice(params_.device_id);
            gate_mut->ensureOnDevice(params_.device_id);
            gamma_mut->ensureOnDevice(params_.device_id);
            output_base->allocateOnDevice(params_.device_id);

            const float *inp_gpu = static_cast<const float *>(input_base->active_data_ptr());
            const float *gate_gpu = static_cast<const float *>(gate_base->active_data_ptr());
            const float *gamma_gpu = static_cast<const float *>(gamma_base->active_data_ptr());
            float *out_gpu = static_cast<float *>(output_base->active_mutable_data_ptr());
            int dev_idx = params_.device_id.toKernelDeviceIndex();
            void *stream = gpuStream();

#ifdef HAVE_CUDA
            if (params_.device_id.is_cuda())
            {
                bool ok = cudaGDN_gated_rmsnorm(
                    inp_gpu, gate_gpu, gamma_gpu, out_gpu,
                    seq_len, static_cast<int>(d_model),
                    static_cast<int>(norm_dim), static_cast<int>(gamma_period),
                    eps, subtract_one, gate_silu,
                    dev_idx, stream);
                if (!ok)
                {
                    LOG_ERROR("[GatedRMSNormStage] CUDA kernel failed");
                    return false;
                }
                return true;
            }
#endif
#ifdef HAVE_ROCM
            if (params_.device_id.is_rocm())
            {
                bool ok = rocmGDN_gated_rmsnorm(
                    inp_gpu, gate_gpu, gamma_gpu, out_gpu,
                    seq_len, static_cast<int>(d_model),
                    static_cast<int>(norm_dim), static_cast<int>(gamma_period),
                    eps, subtract_one, gate_silu,
                    dev_idx, stream);
                if (!ok)
                {
                    LOG_ERROR("[GatedRMSNormStage] ROCm kernel failed");
                    return false;
                }
                return true;
            }
#endif
        }

        // CPU path: extract host pointers (only reached when NOT on GPU)
        const float *input_data = input_base->data();
        const float *gate_data = gate_base->data();
        const float *gamma_data = gamma_base->data();
        float *output_data = output_base->mutable_data();

        if (!input_data || !gate_data || !gamma_data || !output_data)
        {
            LOG_ERROR("[GatedRMSNormStage] Null data pointer");
            return false;
        }

        const size_t n_groups = d_model / norm_dim;

        auto do_work = [&]()
        {
            // Parallelize over rows (timesteps) for prefill, or groups for single token
            const int total_work = seq_len * static_cast<int>(n_groups);
#pragma omp for schedule(static)
            for (int work_idx = 0; work_idx < total_work; ++work_idx)
            {
                const int t = work_idx / static_cast<int>(n_groups);
                const size_t g = work_idx % n_groups;
                const size_t offset = static_cast<size_t>(t) * d_model + g * norm_dim;

                // Compute RMS over norm_dim elements
#if defined(__AVX512F__)
                {
                    __m512 vsum_sq = _mm512_setzero_ps();
                    const size_t nd_vec = norm_dim & ~static_cast<size_t>(15);
                    size_t j = 0;
                    for (; j < nd_vec; j += 16)
                    {
                        __m512 vv = _mm512_loadu_ps(input_data + offset + j);
                        vsum_sq = _mm512_fmadd_ps(vv, vv, vsum_sq);
                    }
                    float sum_sq = _mm512_reduce_add_ps(vsum_sq);
                    for (; j < norm_dim; ++j)
                    {
                        const float v = input_data[offset + j];
                        sum_sq += v * v;
                    }
                    const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(norm_dim) + eps);

                    // Normalize, apply gamma, multiply by gate
                    const __m512 vinv_rms = _mm512_set1_ps(inv_rms);
                    const __m512 vone = _mm512_set1_ps(1.0f);
                    const __m512 vneg1 = _mm512_set1_ps(-1.0f);
                    j = 0;
                    for (; j < nd_vec; j += 16)
                    {
                        __m512 vnorm = _mm512_mul_ps(_mm512_loadu_ps(input_data + offset + j), vinv_rms);
                        __m512 vgamma = _mm512_loadu_ps(gamma_data + (j % gamma_period));
                        if (subtract_one)
                            vgamma = _mm512_add_ps(vone, vgamma);
                        __m512 vgate = _mm512_loadu_ps(gate_data + offset + j);
                        __m512 vgate_act;
                        if (gate_silu)
                        {
                            // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
                            // Fast exp via range reduction: exp(x) = 2^(x*log2e) = 2^n * 2^f
                            // where n = round(x*log2e) and f = x*log2e - n ∈ [-0.5, 0.5].
                            // Polynomial approximates 2^f (Taylor of exp(f*ln2)).
                            __m512 neg_g = _mm512_mul_ps(vgate, vneg1);
                            neg_g = _mm512_max_ps(_mm512_set1_ps(-88.0f), _mm512_min_ps(_mm512_set1_ps(88.0f), neg_g));
                            const __m512 vlog2e = _mm512_set1_ps(1.4426950408889634f);
                            const __m512 vln2 = _mm512_set1_ps(0.6931471805599453f);
                            __m512 neg_g_scaled = _mm512_mul_ps(neg_g, vlog2e);
                            __m512 vn = _mm512_roundscale_ps(neg_g_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                            __m512 vf = _mm512_sub_ps(neg_g_scaled, vn);
                            __m512 vp = _mm512_fmadd_ps(_mm512_set1_ps(0.0013333558146428f), vf, _mm512_set1_ps(0.0096181291076285f));
                            vp = _mm512_fmadd_ps(vp, vf, _mm512_set1_ps(0.0555041086648216f));
                            vp = _mm512_fmadd_ps(vp, vf, _mm512_set1_ps(0.2402265069591007f));
                            vp = _mm512_fmadd_ps(vp, vf, vln2);
                            vp = _mm512_fmadd_ps(vp, vf, vone);
                            __m512i vi_n = _mm512_add_epi32(_mm512_cvtps_epi32(vn), _mm512_set1_epi32(127));
                            __m512 v2n = _mm512_castsi512_ps(_mm512_slli_epi32(vi_n, 23));
                            __m512 vexp = _mm512_mul_ps(vp, v2n);
                            __m512 vsig = _mm512_div_ps(vone, _mm512_add_ps(vone, vexp));
                            vgate_act = _mm512_mul_ps(vgate, vsig);
                        }
                        else
                        {
                            vgate_act = vgate;
                        }
                        _mm512_storeu_ps(output_data + offset + j, _mm512_mul_ps(_mm512_mul_ps(vnorm, vgamma), vgate_act));
                    }
                    for (; j < norm_dim; ++j)
                    {
                        const float normalized = input_data[offset + j] * inv_rms;
                        const float gamma_eff = subtract_one ? (1.0f + gamma_data[j % gamma_period]) : gamma_data[j % gamma_period];
                        const float gate_val = gate_data[offset + j];
                        const float gate_act = gate_silu ? gate_val / (1.0f + std::exp(-gate_val)) : gate_val;
                        output_data[offset + j] = normalized * gamma_eff * gate_act;
                    }
                }
#else
                float sum_sq = 0.0f;
                for (size_t j = 0; j < norm_dim; ++j)
                {
                    const float v = input_data[offset + j];
                    sum_sq += v * v;
                }
                const float rms = std::sqrt(sum_sq / static_cast<float>(norm_dim) + eps);
                const float inv_rms = 1.0f / rms;

                // Normalize, apply gamma (with optional subtract_one), multiply by gate
                for (size_t j = 0; j < norm_dim; ++j)
                {
                    const float normalized = input_data[offset + j] * inv_rms;
                    const float gamma_eff = subtract_one
                                                ? (1.0f + gamma_data[j % gamma_period])
                                                : gamma_data[j % gamma_period];
                    const float gate_val = gate_data[offset + j];
                    const float gate_act = gate_silu
                                               ? gate_val / (1.0f + std::exp(-gate_val))
                                               : gate_val;
                    output_data[offset + j] = normalized * gamma_eff * gate_act;
                }
#endif
            }
        };
        OMP_WORKSHARE_REGION(do_work);

        LOG_DEBUG("[GatedRMSNormStage] Executed: seq_len=" << seq_len
                                                           << " d_model=" << d_model
                                                           << " norm_dim=" << norm_dim
                                                           << " n_groups=" << n_groups
                                                           << " gate_silu=" << params_.gate_silu
                                                           << " subtract_one=" << params_.subtract_one);
        return true;
    }

    size_t GatedRMSNormStage::estimatedFlops() const
    {
        auto *input_base = dynamic_cast<TensorBase *>(params_.input);
        if (!input_base)
            return 0;
        // RMSNorm (~3 ops/element) + gate multiply (1 op/element)
        return input_base->numel() * 4;
    }

    size_t GatedRMSNormStage::estimatedMemoryBytes() const
    {
        auto *input_base = dynamic_cast<TensorBase *>(params_.input);
        if (!input_base)
            return 0;
        // Read input + gate + gamma, write output
        return input_base->numel() * 4 * sizeof(float);
    }

    bool GatedRMSNormStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#ifdef HAVE_CUDA
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#ifdef HAVE_ROCM
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    StageDumpInfo GatedRMSNormStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Use actual seq_len dimensions, not the buffer capacity.
        // Output cols = same as input cols (norm doesn't change dimension).
        auto *out_base = dynamic_cast<const TensorBase *>(params_.output);
        if (out_base)
        {
            const size_t rows = params_.seq_len > 0
                                    ? static_cast<size_t>(params_.seq_len)
                                    : out_base->rows();
            info.addOutput("output", params_.output, rows, out_base->cols());
        }

        return info;
    }

    StageBufferRequirements GatedRMSNormStage::getBufferRequirements() const
    {
        return {};
    }

    StageBufferContract GatedRMSNormStage::bufferContract() const
    {
        StageBufferContract contract;
        if (params_.input_buffer_id)
            contract.addInput(*params_.input_buffer_id);
        if (params_.gate_buffer_id)
            contract.addInput(*params_.gate_buffer_id);
        if (params_.output_buffer_id)
            contract.addOutput(*params_.output_buffer_id);
        return contract;
    }

} // namespace llaminar2
