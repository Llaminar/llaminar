/**
 * @file FusedResidualNormStage.cpp
 * @brief Fused Residual Add + RMSNorm stage (GPU optimization)
 *
 * On GPU: Single kernel reads input+residual, writes updated residual AND
 *         normalized output. Saves one global memory round-trip per fusion point.
 * On CPU: Falls back to sequential KernelFactory dispatch.
 */

#include "FusedResidualNormStage.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/cpu/primitives/RMSNormPrimitives.h"
#include "../../../utils/KernelProfiler.h"
#include <cmath>

#ifdef HAVE_CUDA
extern "C"
{
    bool cudaOps_fused_residual_rmsnorm_fp32(
        const float *input, const float *residual, const float *gamma,
        float *residual_output, float *norm_output,
        int rows, int cols, float eps,
        int device_idx, void *stream);
}
#endif

#ifdef HAVE_ROCM
extern "C"
{
    bool hipOps_fused_residual_rmsnorm_fp32(
        const float *input, const float *residual, const float *gamma,
        float *residual_output, float *norm_output,
        int rows, int cols, float eps,
        int device_idx, void *stream);
}
#endif

namespace llaminar2
{

    FusedResidualNormStage::FusedResidualNormStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool FusedResidualNormStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "FusedResidualNormStage"))
            return false;

        if (!ensureRequiredPointers("FusedResidualNormStage", {
                                                                  {"input", params_.input},
                                                                  {"residual", params_.residual},
                                                                  {"gamma", params_.gamma},
                                                                  {"norm_output", params_.norm_output},
                                                              }))
            return false;

        auto *input_base = requireTensorBasePtr(params_.input, "input");
        auto *residual_base = requireTensorBasePtr(params_.residual, "residual");
        auto *gamma_base = requireTensorBasePtr(params_.gamma, "gamma");
        auto *norm_output_base = requireTensorBasePtr(params_.norm_output, "norm_output");
        if (!input_base || !residual_base || !gamma_base || !norm_output_base)
            return false;

        const int seq_len = params_.seq_len > 0 ? params_.seq_len : static_cast<int>(input_base->rows());
        const int hidden_dim = params_.hidden_dim > 0 ? params_.hidden_dim : static_cast<int>(input_base->cols());
        const size_t num_elements = static_cast<size_t>(seq_len) * hidden_dim;

        LOG_DEBUG("[FusedResidualNormStage] seq_len=" << seq_len
                                                      << " hidden_dim=" << hidden_dim
                                                      << " eps=" << params_.eps);

#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
        if (params_.device_id.is_gpu())
        {
            // GPU fused path: single kernel for residual add + RMSNorm
            const float *d_input = static_cast<const float *>(input_base->gpu_data_ptr());
            const float *d_residual = static_cast<const float *>(residual_base->gpu_data_ptr());
            const float *d_gamma = static_cast<const float *>(gamma_base->gpu_data_ptr());

            // residual is updated in-place (output = residual buffer)
            float *d_residual_out = static_cast<float *>(residual_base->gpu_data_ptr());
            float *d_norm_out = static_cast<float *>(norm_output_base->gpu_data_ptr());

            if (!d_input || !d_residual || !d_gamma || !d_residual_out || !d_norm_out)
            {
                LOG_ERROR("[FusedResidualNormStage] Null GPU pointer");
                return false;
            }

            void *stream = gpuStream();
            bool ok = false;

#ifdef HAVE_CUDA
            if (params_.device_id.is_cuda())
            {
                ok = cudaOps_fused_residual_rmsnorm_fp32(
                    d_input, d_residual, d_gamma,
                    d_residual_out, d_norm_out,
                    seq_len, hidden_dim, params_.eps,
                    params_.device_id.toKernelDeviceIndex(), stream);
            }
#endif
#ifdef HAVE_ROCM
            if (params_.device_id.is_rocm())
            {
                ok = hipOps_fused_residual_rmsnorm_fp32(
                    d_input, d_residual, d_gamma,
                    d_residual_out, d_norm_out,
                    seq_len, hidden_dim, params_.eps,
                    params_.device_id.toKernelDeviceIndex(), stream);
            }
#endif

            if (!ok)
            {
                LOG_ERROR("[FusedResidualNormStage] GPU fused kernel failed");
                return false;
            }

            // Mark both outputs as device-dirty (GPU is authoritative)
            residual_base->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            norm_output_base->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

            traceOutput("residual", params_.residual);
            traceOutput("norm_output", params_.norm_output);
            return true;
        }
#endif

        // CPU path: fused residual add + RMSNorm
        // For FP32 with small seq_len (decode), use a single fused primitive
        // that avoids two separate kernel dispatches and their OMP overhead.
        if (input_base->native_type() == TensorType::FP32 && seq_len <= 4)
        {
            KERNEL_PROFILE_SCOPE(KernelType::RMS_NORM);

            const float *in_data = input_base->data();
            float *res_data = residual_base->mutable_data();
            const float *gamma_data = gamma_base->data();
            float *out_data = norm_output_base->mutable_data();

#if defined(__AVX512F__)
            for (int r = 0; r < seq_len; ++r)
            {
                primitives::fused_residual_rmsnorm_row_avx512(
                    in_data + r * hidden_dim,
                    res_data + r * hidden_dim,
                    gamma_data,
                    out_data + r * hidden_dim,
                    static_cast<std::size_t>(hidden_dim),
                    params_.eps);
            }
#else
            // Scalar fallback: separate operations
            for (int r = 0; r < seq_len; ++r)
            {
                float *res_row = res_data + r * hidden_dim;
                const float *in_row = in_data + r * hidden_dim;
                float *out_row = out_data + r * hidden_dim;

                // Residual add
                for (int i = 0; i < hidden_dim; ++i)
                    res_row[i] += in_row[i];

                // RMSNorm
                double sum_sq = 0.0;
                for (int i = 0; i < hidden_dim; ++i)
                    sum_sq += static_cast<double>(res_row[i]) * static_cast<double>(res_row[i]);

                float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / hidden_dim) + params_.eps);
                for (int i = 0; i < hidden_dim; ++i)
                    out_row[i] = gamma_data[i] * res_row[i] * inv_rms;
            }
#endif

            traceOutput("residual", params_.residual);
            traceOutput("norm_output", params_.norm_output);

            // Temporary layer trace: print post-residual hidden state (last row, first 6 elms)
            {
                static bool do_trace = std::getenv("LLAMINAR_LAYER_TRACE") != nullptr;
                if (do_trace)
                {
                    const int last = (seq_len - 1) * hidden_dim;
                    LOG_DEBUG("[LayerTrace] " << name() << " post-residual [" << seq_len << "x" << hidden_dim
                                             << "] last_row[:6]=" << res_data[last] << "," << res_data[last + 1] << "," << res_data[last + 2]
                                             << "," << res_data[last + 3] << "," << res_data[last + 4] << "," << res_data[last + 5]);
                }
            }

            return true;
        }

        // CPU general path: use KernelFactory for sequential residual add + RMSNorm
        auto *res_kernel = llaminar::v2::kernels::KernelFactory::getOrCreateResidualAdd(
            input_base, params_.device_id);
        if (!res_kernel)
        {
            LOG_ERROR("[FusedResidualNormStage] Failed to create ResidualAdd kernel");
            return false;
        }

        // residual = input + residual (in-place)
        bool ok = res_kernel->apply_tensor(
            input_base, residual_base,
            residual_base, // output = residual (in-place)
            num_elements,
            params_.mpi_ctx,
            params_.device_id.toKernelDeviceIndex());
        if (!ok)
        {
            LOG_ERROR("[FusedResidualNormStage] ResidualAdd failed");
            return false;
        }

        // Temporary layer trace: print post-residual hidden state (last row, first 6 elms)
        {
            static bool do_trace = std::getenv("LLAMINAR_LAYER_TRACE") != nullptr;
            if (do_trace)
            {
                const float *r = residual_base->data();
                const int last = (seq_len - 1) * hidden_dim;
                LOG_DEBUG("[LayerTrace] " << name() << " post-residual [" << seq_len << "x" << hidden_dim
                                         << "] last_row[:6]=" << r[last] << "," << r[last + 1] << "," << r[last + 2]
                                         << "," << r[last + 3] << "," << r[last + 4] << "," << r[last + 5]);
            }
        }

        auto *norm_kernel = llaminar::v2::kernels::KernelFactory::getOrCreateRMSNorm(
            residual_base, params_.device_id);
        if (!norm_kernel)
        {
            LOG_ERROR("[FusedResidualNormStage] Failed to create RMSNorm kernel");
            return false;
        }

        ok = norm_kernel->apply_tensor(
            residual_base, gamma_base, norm_output_base,
            seq_len, hidden_dim, params_.eps,
            params_.mpi_ctx,
            params_.device_id.toKernelDeviceIndex());

        if (ok)
        {
            traceOutput("residual", params_.residual);
            traceOutput("norm_output", params_.norm_output);
        }
        return ok;
    }

    size_t FusedResidualNormStage::estimatedFlops() const
    {
        // ResidualAdd: N adds + RMSNorm: N muls + N adds + sqrt + N muls + N muls
        const size_t n = static_cast<size_t>(params_.seq_len) * params_.hidden_dim;
        return n + 4 * n; // ~5N
    }

    size_t FusedResidualNormStage::estimatedMemoryBytes() const
    {
        // Reads: input + residual + gamma. Writes: residual_out + norm_out
        const size_t n = static_cast<size_t>(params_.seq_len) * params_.hidden_dim;
        return (3 * n + 2 * n) * sizeof(float);
    }

    StageDumpInfo FusedResidualNormStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        const int seq_len = params_.seq_len > 0
                                ? params_.seq_len
                                : (params_.input ? static_cast<int>(requireTensorBasePtr(params_.input, "")->rows()) : 0);
        const int hidden_dim = params_.hidden_dim > 0
                                   ? params_.hidden_dim
                                   : (params_.input ? static_cast<int>(requireTensorBasePtr(params_.input, "")->cols()) : 0);

        info.addInput("input", params_.input, seq_len, hidden_dim);
        info.addInput("residual", params_.residual, seq_len, hidden_dim);
        info.addInput("gamma", params_.gamma, 1, hidden_dim);
        info.addOutput("residual_out", params_.residual, seq_len, hidden_dim);
        info.addOutput("norm_output", params_.norm_output, seq_len, hidden_dim);
        info.addScalar("eps", params_.eps);
        return info;
    }

    StageBufferContract FusedResidualNormStage::bufferContract() const
    {
        StageBufferContract contract;
        if (params_.input_buffer_id)
            contract.addInput(*params_.input_buffer_id);
        if (params_.residual_buffer_id)
            contract.addInOut(*params_.residual_buffer_id);
        if (params_.norm_output_buffer_id)
            contract.addOutput(*params_.norm_output_buffer_id);
        // Gamma is a model weight, not arena-managed
        if (params_.gamma)
            contract.addWeight(const_cast<ITensor *>(params_.gamma));
        return contract;
    }

} // namespace llaminar2
