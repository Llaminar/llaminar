/**
 * @file FusedResidualNormStage.h
 * @brief Production residual-add plus RMSNorm fusion for CPU, CUDA, and ROCm.
 *
 * The stage publishes two values from one logical operation:
 *
 * @code
 * residual[row, col] = input[row, col] + residual[row, col]
 * norm_output[row, col] = RMSNorm(residual[row, :], gamma[:], epsilon)[col]
 * @endcode
 *
 * CUDA and ROCm execute one device kernel with one block/workgroup per row. The
 * kernel keeps each row's unrounded residual sums in registers while reducing
 * the RMS value, then publishes both the native residual and normalized output.
 * CPU FP32 verifier rows use a fused cache-resident row primitive; the remaining
 * CPU native floating formats use the typed residual-add and RMSNorm kernels as
 * one stage-owned implementation.
 *
 * MTP verifier execution passes M=2..4 rows through this stage at once. Every
 * backend route must remain byte-identical to M independent production M=1
 * executions. In particular, callers must never substitute host row replay for
 * the grouped stage or permit a GPU tensor to be accessed through host-visible
 * storage.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"

#include <optional>

namespace llaminar2
{

    /**
     * @brief Add a projection into the residual stream and normalize the result.
     *
     * Input, residual, and normalized output tensors use one matching native
     * floating format (FP32, BF16, or FP16); gamma is always FP32. GPU execution
     * requires all tensors to be resident on the selected device and requires a
     * non-null stream assigned by the graph executor through setGPUStream().
     */
    class FusedResidualNormStage : public IComputeStage
    {
    public:
        /** @brief Immutable tensor bindings and active-shape policy for one stage. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            /// Projection output to add, for example Wo or FFN down-projection output.
            const ITensor *input = nullptr;

            /// Residual stream updated in place with input.
            ITensor *residual = nullptr;

            /// FP32 RMSNorm scale vector with at least hidden_dim values.
            const ITensor *gamma = nullptr;

            /// Native-format normalized output consumed by the next projection.
            ITensor *norm_output = nullptr;

            /// Numerical stability epsilon added to the row mean square.
            float eps = 1e-6f;

            /// Active row count; zero derives the count from input rows.
            int seq_len = 0;

            /// Active row width; zero derives the width from input columns.
            int hidden_dim = 0;

            /// Optional arena identifier for the projection input.
            std::optional<BufferId> input_buffer_id;

            /// Optional arena identifier for the in-place residual stream.
            std::optional<BufferId> residual_buffer_id;

            /// Optional arena identifier for the normalized output.
            std::optional<BufferId> norm_output_buffer_id;
        };

        /** @brief Construct a stage with fixed tensor bindings and active shape. */
        explicit FusedResidualNormStage(Params params);

        /**
         * @brief Execute the backend-native residual-add plus RMSNorm route.
         *
         * @param ctx Device context selected by the graph executor.
         * @return true after both outputs have been published, otherwise false.
         */
        bool execute(IDeviceContext *ctx) override;

        /** @return Stable stage kind used by graph scheduling and diagnostics. */
        ComputeStageType type() const override { return ComputeStageType::FUSED_RESIDUAL_NORM; }

        /** @return Approximate arithmetic operation count for the active shape. */
        size_t estimatedFlops() const override;

        /** @return Approximate bytes read and written for the active shape. */
        size_t estimatedMemoryBytes() const override;

        /** @return Whether this stage has a production implementation on backend. */
        bool supportsBackend(ComputeBackendType backend) const override
        {
            return backend == ComputeBackendType::CPU ||
                   backend == ComputeBackendType::GPU_CUDA ||
                   backend == ComputeBackendType::GPU_ROCM;
        }

        /** @return Tensor/scalar metadata used by stage snapshot diagnostics. */
        StageDumpInfo buildDumpInfoImpl() const override;

        /** @return Arena coherence contract for input, residual, output, and gamma. */
        StageBufferContract bufferContract() const override;

    private:
        Params params_;
    };

} // namespace llaminar2
