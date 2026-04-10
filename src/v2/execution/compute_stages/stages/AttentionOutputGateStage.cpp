/**
 * @file AttentionOutputGateStage.cpp
 * @brief Implementation of sigmoid-gated attention output stage
 */

#include "AttentionOutputGateStage.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"

#include <cmath>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#include "../../../kernels/cpu/simd/AVX2Helpers.h"
#endif

// GPU kernel declarations
#ifdef HAVE_CUDA
extern "C" bool cudaGDN_attention_output_gate(
    const float *input, const float *gate, float *output,
    int size, int device_idx, void *stream);
#endif
#ifdef HAVE_ROCM
extern "C" bool rocmGDN_attention_output_gate(
    const float *input, const float *gate, float *output,
    int size, int device_idx, void *stream);
#endif

namespace llaminar2
{

    // ========================================================================
    // Named ISA implementations: attention_gate_row
    // out[i] = sigmoid(gate[i]) * input[i]
    // ========================================================================

    void attention_gate_row_scalar(const float *gate_row, const float *inp_row,
                                   float *out_row, size_t cols)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            const float sig = 1.0f / (1.0f + std::exp(-gate_row[j]));
            out_row[j] = sig * inp_row[j];
        }
    }

#if defined(__AVX2__)
    void attention_gate_row_avx2(const float *gate_row, const float *inp_row,
                                 float *out_row, size_t cols)
    {
        size_t j = 0;
        const size_t vec_end = cols & ~static_cast<size_t>(7);
        for (; j < vec_end; j += 8)
        {
            __m256 vg = _mm256_loadu_ps(gate_row + j);
            __m256 vsig = avx2::fast_sigmoid(vg);
            __m256 vinp = _mm256_loadu_ps(inp_row + j);
            _mm256_storeu_ps(out_row + j, _mm256_mul_ps(vsig, vinp));
        }
        for (; j < cols; ++j)
        {
            const float sig = 1.0f / (1.0f + std::exp(-gate_row[j]));
            out_row[j] = sig * inp_row[j];
        }
    }
#endif

#if defined(__AVX512F__)
    void attention_gate_row_avx512(const float *gate_row, const float *inp_row,
                                   float *out_row, size_t cols)
    {
        const __m512 vone = _mm512_set1_ps(1.0f);
        const __m512 vneg = _mm512_set1_ps(-1.0f);
        const __m512 vmin = _mm512_set1_ps(-88.0f);
        const __m512 vmax = _mm512_set1_ps(88.0f);
        const __m512 vlog2e = _mm512_set1_ps(1.4426950408889634f);
        const __m512 vc0 = _mm512_set1_ps(1.0f);
        const __m512 vc1 = _mm512_set1_ps(0.6931471805599453f);
        const __m512 vc2 = _mm512_set1_ps(0.2402265069591007f);
        const __m512 vc3 = _mm512_set1_ps(0.0555041086648216f);
        const __m512 vc4 = _mm512_set1_ps(0.0096181291076285f);
        const __m512 vc5 = _mm512_set1_ps(0.0013333558146428f);

        size_t j = 0;
        const size_t vec_end = cols & ~static_cast<size_t>(15);
        for (; j < vec_end; j += 16)
        {
            __m512 vg = _mm512_loadu_ps(gate_row + j);
            __m512 neg_g = _mm512_max_ps(vmin, _mm512_min_ps(vmax, _mm512_mul_ps(vg, vneg)));
            __m512 neg_g_scaled = _mm512_mul_ps(neg_g, vlog2e);
            __m512 vn = _mm512_roundscale_ps(neg_g_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m512 vf = _mm512_sub_ps(neg_g_scaled, vn);
            __m512 vp = _mm512_fmadd_ps(vc5, vf, vc4);
            vp = _mm512_fmadd_ps(vp, vf, vc3);
            vp = _mm512_fmadd_ps(vp, vf, vc2);
            vp = _mm512_fmadd_ps(vp, vf, vc1);
            vp = _mm512_fmadd_ps(vp, vf, vc0);
            __m512i vi_n = _mm512_cvtps_epi32(vn);
            vi_n = _mm512_add_epi32(vi_n, _mm512_set1_epi32(127));
            vi_n = _mm512_slli_epi32(vi_n, 23);
            __m512 v2n = _mm512_castsi512_ps(vi_n);
            __m512 vexp = _mm512_mul_ps(vp, v2n);
            __m512 vsig = _mm512_div_ps(vone, _mm512_add_ps(vone, vexp));
            __m512 vinp = _mm512_loadu_ps(inp_row + j);
            _mm512_storeu_ps(out_row + j, _mm512_mul_ps(vsig, vinp));
        }
        for (; j < cols; ++j)
        {
            const float sig = 1.0f / (1.0f + std::exp(-gate_row[j]));
            out_row[j] = sig * inp_row[j];
        }
    }
#endif

    inline void attention_gate_row(const float *gate_row, const float *inp_row,
                                   float *out_row, size_t cols)
    {
#if defined(__AVX512F__)
        attention_gate_row_avx512(gate_row, inp_row, out_row, cols);
#elif defined(__AVX2__)
        attention_gate_row_avx2(gate_row, inp_row, out_row, cols);
#else
        attention_gate_row_scalar(gate_row, inp_row, out_row, cols);
#endif
    }

    // ========================================================================
    // End named ISA implementations
    // ========================================================================

    AttentionOutputGateStage::AttentionOutputGateStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool AttentionOutputGateStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "AttentionOutputGateStage"))
            return false;

        if (!ensureRequiredPointers("AttentionOutputGateStage",
                                    {{"input", params_.input},
                                     {"gate", params_.gate},
                                     {"output", params_.output}}))
            return false;

        auto *input_base = requireTensorBasePtr(params_.input, "input");
        auto *gate_base = requireTensorBasePtr(params_.gate, "gate");
        auto *output_base = requireTensorBasePtr(params_.output, "output");

        if (!input_base || !gate_base || !output_base)
        {
            LOG_ERROR("[AttentionOutputGateStage] Tensor cast failed");
            return false;
        }

        const int seq_len = (params_.seq_len > 0)
                                ? params_.seq_len
                                : static_cast<int>(input_base->rows());

        const size_t cols = input_base->shape().size() > 1 ? input_base->shape()[1] : input_base->numel();
        const size_t total = static_cast<size_t>(seq_len) * cols;

        // GPU dispatch path
        if (params_.device_id.is_gpu())
        {
            // Ensure all tensors are on device
            const_cast<TensorBase *>(input_base)->ensureOnDevice(params_.device_id);
            const_cast<TensorBase *>(gate_base)->ensureOnDevice(params_.device_id);
            output_base->allocateOnDevice(params_.device_id);

            const float *inp_gpu = static_cast<const float *>(input_base->active_data_ptr());
            const float *gate_gpu = static_cast<const float *>(gate_base->active_data_ptr());
            float *out_gpu = static_cast<float *>(output_base->active_mutable_data_ptr());
            int dev_idx = params_.device_id.toKernelDeviceIndex();
            void *stream = gpuStream();

#ifdef HAVE_CUDA
            if (params_.device_id.is_cuda())
            {
                bool ok = cudaGDN_attention_output_gate(
                    inp_gpu, gate_gpu, out_gpu,
                    static_cast<int>(total), dev_idx, stream);
                if (!ok)
                {
                    LOG_ERROR("[AttentionOutputGateStage] CUDA kernel failed");
                    return false;
                }
                return true;
            }
#endif
#ifdef HAVE_ROCM
            if (params_.device_id.is_rocm())
            {
                bool ok = rocmGDN_attention_output_gate(
                    inp_gpu, gate_gpu, out_gpu,
                    static_cast<int>(total), dev_idx, stream);
                if (!ok)
                {
                    LOG_ERROR("[AttentionOutputGateStage] ROCm kernel failed");
                    return false;
                }
                return true;
            }
#endif
        }

        const float *input_data = input_base->data();
        const float *gate_data = gate_base->data();
        float *output_data = output_base->mutable_data();

        if (!input_data || !gate_data || !output_data)
        {
            LOG_ERROR("[AttentionOutputGateStage] Null data pointer");
            return false;
        }

        // output[i] = sigmoid(gate[i]) * input[i]
        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int t = 0; t < seq_len; ++t)
            {
                const size_t row_off = static_cast<size_t>(t) * cols;
                attention_gate_row(gate_data + row_off, input_data + row_off,
                                   output_data + row_off, cols);
            }
        };
        OMP_WORKSHARE_REGION(do_work);

        LOG_DEBUG("[AttentionOutputGateStage] Executed: seq_len=" << seq_len
                                                                  << " cols=" << cols
                                                                  << " total=" << total);
        return true;
    }

    size_t AttentionOutputGateStage::estimatedFlops() const
    {
        auto *input_base = dynamic_cast<TensorBase *>(params_.input);
        if (!input_base)
            return 0;
        // sigmoid (4 ops) + multiply (1 op) per element
        return input_base->numel() * 5;
    }

    size_t AttentionOutputGateStage::estimatedMemoryBytes() const
    {
        auto *input_base = dynamic_cast<TensorBase *>(params_.input);
        if (!input_base)
            return 0;
        // Read input + gate, write output (each float)
        return input_base->numel() * 3 * sizeof(float);
    }

    bool AttentionOutputGateStage::supportsBackend(ComputeBackendType backend) const
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

    StageDumpInfo AttentionOutputGateStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        auto *input_base = dynamic_cast<TensorBase *>(params_.input);
        auto *gate_base = dynamic_cast<TensorBase *>(params_.gate);
        auto *output_base = dynamic_cast<TensorBase *>(params_.output);

        if (input_base)
            info.inputs.push_back({"input", input_base});
        if (gate_base)
            info.inputs.push_back({"gate", gate_base});
        if (output_base)
            info.outputs.push_back({"output", output_base});

        return info;
    }

    StageBufferRequirements AttentionOutputGateStage::getBufferRequirements() const
    {
        return {};
    }

    StageBufferContract AttentionOutputGateStage::bufferContract() const
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
