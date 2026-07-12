/**
 * @file FusedResidualNormStage.cpp
 * @brief Backend-native implementation of fused residual-add plus RMSNorm.
 *
 * GPU execution is fully device-owned: all four tensor bindings are resolved
 * through gpu_data_ptr(), launches use the graph executor's explicit stream, and
 * both mutable tensors become device-authoritative only after a successful
 * launch. FP32, BF16, and FP16 use matching one-block-per-row CUDA/HIP kernels.
 *
 * CPU execution retains the fused cache-resident FP32 decode primitive for
 * M=1..4. Other native floating formats execute one typed residual-add call and
 * one typed RMSNorm call under this stage. Neither backend implementation calls
 * the serial runtime or replays M=1 stage executions in production.
 */

#include "FusedResidualNormStage.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/cpu/primitives/RMSNormPrimitives.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../local_execution/graph/GraphCaptureGuard.h"
#include <cmath>
#include <string>

#ifdef HAVE_CUDA
extern "C"
{
    bool cudaOps_fused_residual_rmsnorm_fp32(
        const float *input, const float *residual, const float *gamma,
        float *residual_output, float *norm_output,
        int rows, int cols, float eps,
        int device_idx, void *stream);

    bool cudaOps_fused_residual_rmsnorm_bf16(
        const uint16_t *input, const uint16_t *residual, const float *gamma,
        uint16_t *residual_output, uint16_t *norm_output,
        int rows, int cols, float eps,
        int device_idx, void *stream);

    bool cudaOps_fused_residual_rmsnorm_fp16(
        const uint16_t *input, const uint16_t *residual, const float *gamma,
        uint16_t *residual_output, uint16_t *norm_output,
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

    bool hipOps_fused_residual_rmsnorm_bf16(
        const uint16_t *input, const uint16_t *residual, const float *gamma,
        uint16_t *residual_output, uint16_t *norm_output,
        int rows, int cols, float eps,
        int device_idx, void *stream);

    bool hipOps_fused_residual_rmsnorm_fp16(
        const uint16_t *input, const uint16_t *residual, const float *gamma,
        uint16_t *residual_output, uint16_t *norm_output,
        int rows, int cols, float eps,
        int device_idx, void *stream);
}
#endif

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Return whether a tensor format has a native fused stage route.
         *
         * Quantized weight codebooks are not activation formats for this stage.
         * The graph must provide one of the three floating activation formats so
         * the in-place residual and normalized output retain identical storage.
         */
        bool isSupportedActivationFormat(TensorType type)
        {
            return type == TensorType::FP32 ||
                   type == TensorType::BF16 ||
                   type == TensorType::FP16;
        }

        /**
         * @brief Validate the native-format and active-shape contract.
         *
         * Explicit active rows may be smaller than arena capacity during decode,
         * therefore capacity is checked with numel() rather than requiring exact
         * tensor shapes. Gamma remains FP32 for every activation format.
         */
        bool validateTensorContract(
            const TensorBase *input,
            const TensorBase *residual,
            const TensorBase *gamma,
            const TensorBase *norm_output,
            int rows,
            int cols)
        {
            if (rows <= 0 || cols <= 0)
            {
                LOG_ERROR("[FusedResidualNormStage] Invalid active shape rows="
                          << rows << " cols=" << cols);
                return false;
            }

            const TensorType activation_type = input->native_type();
            if (!isSupportedActivationFormat(activation_type))
            {
                LOG_ERROR("[FusedResidualNormStage] Unsupported activation format "
                          << tensorTypeName(activation_type));
                return false;
            }

            if (residual->native_type() != activation_type ||
                norm_output->native_type() != activation_type)
            {
                LOG_ERROR("[FusedResidualNormStage] input/residual/norm_output formats must match: input="
                          << tensorTypeName(activation_type)
                          << " residual=" << tensorTypeName(residual->native_type())
                          << " norm_output=" << tensorTypeName(norm_output->native_type()));
                return false;
            }

            if (gamma->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[FusedResidualNormStage] gamma must be FP32, got "
                          << tensorTypeName(gamma->native_type()));
                return false;
            }

            const size_t active_elements = static_cast<size_t>(rows) *
                                           static_cast<size_t>(cols);
            if (input->numel() < active_elements ||
                residual->numel() < active_elements ||
                norm_output->numel() < active_elements ||
                gamma->numel() < static_cast<size_t>(cols))
            {
                LOG_ERROR("[FusedResidualNormStage] Tensor capacity is smaller than the active shape");
                return false;
            }
            return true;
        }

        /**
         * @brief Publish one successful grouped production route to perfstats.
         *
         * M=1 serial witnesses intentionally do not emit this metric. Focused
         * integration tests can therefore reset immediately before any runtime
         * M>=2 invocation and require exactly one record, proving the grouped
         * stage actually ran instead of merely comparing precomputed bytes.
         */
        void recordGroupedRoute(
            const char *backend,
            const char *counter_name,
            TensorType activation_type,
            int rows,
            int cols,
            const char *implementation,
            const char *invocation_policy)
        {
            if (rows < 2)
                return;

            PerfStatsCollector::addCounter(
                "kernel",
                counter_name,
                1.0,
                "verifier",
                backend,
                {{"tensor_format", tensorTypeName(activation_type)},
                 {"verifier_rows", std::to_string(rows)},
                 {"cols", std::to_string(cols)},
                 {"implementation", implementation},
                 {"capture_mode", isGraphCaptureActive() ? "graph_capture" : "direct"},
                 {"row_mapping", "independent_rows"},
                 {"invocation_policy", invocation_policy}});
        }
    } // namespace

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

        if (!validateTensorContract(
                input_base, residual_base, gamma_base, norm_output_base,
                seq_len, hidden_dim))
            return false;

        LOG_DEBUG("[FusedResidualNormStage] seq_len=" << seq_len
                                                      << " hidden_dim=" << hidden_dim
                                                      << " eps=" << params_.eps);

#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
        if (params_.device_id.is_gpu())
        {
            // Device ownership is strict: no active_data_ptr() bridge is legal
            // here because graph capture and replay must never adopt host storage.
            const void *d_input = input_base->gpu_data_ptr();
            const void *d_residual = residual_base->gpu_data_ptr();
            const float *d_gamma = static_cast<const float *>(gamma_base->gpu_data_ptr());
            void *d_residual_out = residual_base->gpu_data_ptr();
            void *d_norm_out = norm_output_base->gpu_data_ptr();

            if (!d_input || !d_residual || !d_gamma || !d_residual_out || !d_norm_out)
            {
                LOG_ERROR("[FusedResidualNormStage] Null GPU pointer");
                return false;
            }

            void *stream = gpuStream();
            if (!stream)
            {
                LOG_ERROR("[FusedResidualNormStage] GPU execution requires an explicit non-default stream");
                return false;
            }

            bool ok = false;
            const TensorType activation_type = input_base->native_type();
            const int device_index = params_.device_id.toKernelDeviceIndex();

#ifdef HAVE_CUDA
            if (params_.device_id.is_cuda())
            {
                switch (activation_type)
                {
                case TensorType::FP32:
                    ok = cudaOps_fused_residual_rmsnorm_fp32(
                        static_cast<const float *>(d_input),
                        static_cast<const float *>(d_residual), d_gamma,
                        static_cast<float *>(d_residual_out),
                        static_cast<float *>(d_norm_out),
                        seq_len, hidden_dim, params_.eps, device_index, stream);
                    break;
                case TensorType::BF16:
                    ok = cudaOps_fused_residual_rmsnorm_bf16(
                        static_cast<const uint16_t *>(d_input),
                        static_cast<const uint16_t *>(d_residual), d_gamma,
                        static_cast<uint16_t *>(d_residual_out),
                        static_cast<uint16_t *>(d_norm_out),
                        seq_len, hidden_dim, params_.eps, device_index, stream);
                    break;
                case TensorType::FP16:
                    ok = cudaOps_fused_residual_rmsnorm_fp16(
                        static_cast<const uint16_t *>(d_input),
                        static_cast<const uint16_t *>(d_residual), d_gamma,
                        static_cast<uint16_t *>(d_residual_out),
                        static_cast<uint16_t *>(d_norm_out),
                        seq_len, hidden_dim, params_.eps, device_index, stream);
                    break;
                default:
                    break;
                }
            }
#endif
#ifdef HAVE_ROCM
            if (params_.device_id.is_rocm())
            {
                switch (activation_type)
                {
                case TensorType::FP32:
                    ok = hipOps_fused_residual_rmsnorm_fp32(
                        static_cast<const float *>(d_input),
                        static_cast<const float *>(d_residual), d_gamma,
                        static_cast<float *>(d_residual_out),
                        static_cast<float *>(d_norm_out),
                        seq_len, hidden_dim, params_.eps, device_index, stream);
                    break;
                case TensorType::BF16:
                    ok = hipOps_fused_residual_rmsnorm_bf16(
                        static_cast<const uint16_t *>(d_input),
                        static_cast<const uint16_t *>(d_residual), d_gamma,
                        static_cast<uint16_t *>(d_residual_out),
                        static_cast<uint16_t *>(d_norm_out),
                        seq_len, hidden_dim, params_.eps, device_index, stream);
                    break;
                case TensorType::FP16:
                    ok = hipOps_fused_residual_rmsnorm_fp16(
                        static_cast<const uint16_t *>(d_input),
                        static_cast<const uint16_t *>(d_residual), d_gamma,
                        static_cast<uint16_t *>(d_residual_out),
                        static_cast<uint16_t *>(d_norm_out),
                        seq_len, hidden_dim, params_.eps, device_index, stream);
                    break;
                default:
                    break;
                }
            }
#endif

            if (!ok)
            {
                LOG_ERROR("[FusedResidualNormStage] GPU fused kernel failed");
                return false;
            }

            if (params_.device_id.is_cuda())
            {
                recordGroupedRoute(
                    "cuda", "cuda_fused_residual_rmsnorm_grouped_verifier_rows_calls",
                    activation_type, seq_len, hidden_dim,
                    "native_fused_kernel", "single_grouped_launch");
            }
            else
            {
                recordGroupedRoute(
                    "rocm", "rocm_fused_residual_rmsnorm_grouped_verifier_rows_calls",
                    activation_type, seq_len, hidden_dim,
                    "native_fused_kernel", "single_grouped_launch");
            }

            // Mark both outputs as device-dirty (GPU is authoritative)
            residual_base->transitionToWithEvent(TensorCoherenceState::DEVICE_AUTHORITATIVE,
                                                 params_.device_id,
                                                 stream);
            norm_output_base->transitionToWithEvent(TensorCoherenceState::DEVICE_AUTHORITATIVE,
                                                    params_.device_id,
                                                    stream);

            traceOutput("residual", params_.residual);
            traceOutput("norm_output", params_.norm_output);
            return true;
        }
#endif

        // CPU path: fused residual add + RMSNorm
        // FP32 verifier and decode rows share one fused primitive for every M.
        // This avoids two separate kernel dispatches while preserving the same
        // per-row residual and RMS reduction order at every speculative depth.
        if (input_base->native_type() == TensorType::FP32)
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
            // Portable scalar implementation preserves the same two-pass row
            // arithmetic when this translation unit has no AVX-512 target.
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
                const bool do_trace = debugEnv().runtime_debug.layer_trace;
                if (do_trace)
                {
                    const int last = (seq_len - 1) * hidden_dim;
                    LOG_DEBUG("[LayerTrace] " << name() << " post-residual [" << seq_len << "x" << hidden_dim
                                              << "] last_row[:6]=" << res_data[last] << "," << res_data[last + 1] << "," << res_data[last + 2]
                                              << "," << res_data[last + 3] << "," << res_data[last + 4] << "," << res_data[last + 5]);
                }
            }

            recordGroupedRoute(
                "cpu", "cpu_fused_residual_rmsnorm_grouped_verifier_rows_calls",
                input_base->native_type(), seq_len, hidden_dim,
                "fused_cache_resident_rows", "single_grouped_stage_call");
            return true;
        }

        // The BF16/FP16 implementation uses one grouped typed residual kernel
        // followed by one grouped typed RMSNorm kernel. Both consume the full
        // active M-row span; neither invokes or re-enters the stage row by row.
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
            const bool do_trace = debugEnv().runtime_debug.layer_trace;
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

            recordGroupedRoute(
                "cpu", "cpu_fused_residual_rmsnorm_grouped_verifier_rows_calls",
                input_base->native_type(), seq_len, hidden_dim,
                "typed_residual_then_rmsnorm", "single_grouped_stage_call");
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
