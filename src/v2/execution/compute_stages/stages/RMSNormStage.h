/**
 * @file RMSNormStage.h
 * @brief Typed RMS-normalization stage and its diagnostic publication policy.
 *
 * The stage owns no activation storage: callers bind stable input, output, and
 * gamma tensors, and execution dispatches the matching CPU/CUDA/ROCm kernel on
 * the stage's exact stream.  Its optional diagnostic publication policy makes
 * a read-only input visible to graph-stable snapshot capture without changing
 * the arithmetic or introducing a second host-owned copy of live model state.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"

#include <cstdint>
#include <optional>
#include <memory>

namespace llaminar2
{
    // Forward declarations
    class ITensorRMSNorm;
    class FP32Tensor;

    /**
     * @brief RMS normalization stage
     *
     * Type-safe implementation using TensorBase* instead of void*.
     * The tensor's native_type() determines precision dispatch.
     * Uses IActivationTensor::applyRMSNorm() for polymorphic device dispatch.
     *
     * Device-agnostic: IActivationTensor handles CPU/GPU dispatch internally.
     */
    class RMSNormStage : public IComputeStage
    {
    public:
        /**
         * @brief Select an additional read-only input view published for diagnostics.
         *
         * Snapshot capture ordinarily observes only values produced by a
         * stage.  The MTP predictor's first normalization is also the exact
         * transaction boundary at which its terminal-hidden input is consumed.
         * Publishing that already-resident input as a diagnostic output lets
         * parity prove the transaction without an identity kernel, host copy,
         * or a second authority for the value.
         */
        enum class DiagnosticInputPublication : std::uint8_t
        {
            None,              ///< Publish only the normalized output.
            MTPTerminalHidden, ///< Also publish the exact MTP terminal-hidden input.
        };

        /** @brief Complete immutable bindings and policies for one RMSNorm stage. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Type-safe tensor pointers (required)
            ITensor *input = nullptr;       ///< Input activation tensor (IActivationTensor*)
            ITensor *output = nullptr;      ///< Output tensor (can be same as input for in-place)
            const ITensor *gamma = nullptr; ///< Gamma weights tensor

            float eps = 1e-6f; ///< Epsilon for numerical stability

            /// Subtract-one weight mode: gamma_effective = 1.0 + gamma_stored.
            /// Used by DeepSeek and Qwen 3.5 variants where norm weights are stored
            /// as (gamma - 1.0) in the GGUF file.
            bool subtract_one = false;

            // Explicit seq_len override for pre-allocated buffers
            // If 0, derives from input tensor dimensions
            // CRITICAL: Must be set during decode when using pre-allocated buffers
            int seq_len = 0;

            // Optional BufferIds for contract-based coherence
            std::optional<BufferId> input_buffer_id;
            std::optional<BufferId> output_buffer_id;

            /// Optional read-only input view exposed to graph-stable diagnostics.
            DiagnosticInputPublication diagnostic_input_publication =
                DiagnosticInputPublication::None;
        };

        explicit RMSNormStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::RMS_NORM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;

        /// Target device for coherence management

        const Params &getParams() const { return params_; }

    private:
        Params params_;
        mutable llaminar2::ITensorRMSNorm *cached_kernel_ = nullptr;
        mutable int cached_kernel_tensor_type_ = -1;

        /// Lazily-allocated gamma with subtract_one transform applied.
        /// Only populated when params_.subtract_one is true.
        mutable std::shared_ptr<FP32Tensor> subtract_one_gamma_;
    };

} // namespace llaminar2
