/**
 * @file MoEOverlayEpochBoundaryStage.h
 * @brief Captured request-reader admission and release for ExpertOverlay epochs.
 *
 * Durable ExpertOverlay placement is an RCU-published family of immutable
 * per-layer banks. A complete inference transaction must acquire one published
 * family before its first MoE reader and release it only after the final state
 * publication. This stage makes either boundary a normal graph-capturable
 * operation with stable model-lifetime addresses and an exact non-null stream.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"

#include "../../../execution/moe/DeviceMoEOverlayEpochArena.h"
#include "../../../kernels/IMoEKernel.h"

#include <cstdint>
#include <memory>
#include <string>

namespace llaminar2
{
    /**
     * @brief One typed edge of an ExpertOverlay inference-reader transaction.
     *
     * Acquire snapshots the currently published selector into the request slot
     * and increments that bank's reader count. Release decrements the exact
     * reader named by the ticket and clears the slot. The two operations are
     * deliberately separate captured graphs so an orchestrator can place every
     * main, sidecar, verifier, and publication graph between them.
     */
    class MoEOverlayEpochBoundaryStage final : public IComputeStage
    {
    public:
        /** @brief Closed set of legal request-reader lifecycle transitions. */
        enum class Operation : std::uint8_t
        {
            Acquire = 0,
            Release = 1,
        };

        /**
         * @brief Immutable model-lifetime addresses embedded in capture.
         *
         * The shared arena owner keeps the control, ticket, and status addresses
         * alive for every executable that clones this stage. Request-varying
         * epoch and selector values live only behind those device addresses.
         */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            std::shared_ptr<DeviceMoEOverlayEpochArena> arena;
            std::uint32_t request_slot = 0u;
            Operation operation = Operation::Acquire;
            std::string stage_name = "moe_overlay_epoch_boundary";
        };

        static_assert(StageParamsRequired<Params>);

        /**
         * @brief Construct one capture-stable boundary and its backend kernel.
         * @param params Complete arena, request slot, device, and operation.
         * @throws std::invalid_argument when topology is incomplete or mismatched.
         * @throws std::runtime_error when the backend kernel cannot be created.
         */
        explicit MoEOverlayEpochBoundaryStage(Params params);

        /**
         * @brief Enqueue the selected boundary on the stage's exact GPU stream.
         * @param ctx Device context matching the immutable participant.
         * @return true when the backend operation was enqueued.
         */
        bool execute(IDeviceContext *ctx) override;

        /** @return Stable type identifier used by graph diagnostics. */
        ComputeStageType type() const override
        {
            return ComputeStageType::MOE_OVERLAY_EPOCH_BOUNDARY;
        }

        /** @return Human-readable capture role. */
        std::string name() const override { return params_.stage_name; }

        /** @return A small constant estimate for the bounded control kernel. */
        size_t estimatedFlops() const override { return 32u; }

        /** @return Control, ticket, and status traffic touched by the boundary. */
        size_t estimatedMemoryBytes() const override;

        /** @return true only for CUDA and ROCm execution. */
        bool supportsBackend(ComputeBackendType backend) const override;

        /** @return true because both backend operations are capture-safe kernels. */
        bool isGraphCapturable() const override { return true; }

        /** @return false; epoch admission is participant-local RCU bookkeeping. */
        bool isCollectiveStage() const override { return false; }

        /** @return NONE because the arena is not a Tensor coherence resource. */
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }

        /** @return Empty tensor contract; all storage belongs to the epoch arena. */
        StageBufferContract bufferContract() const override { return {}; }

        /** @return Scalar diagnostic fields for graph dumps. */
        StageDumpInfo buildDumpInfoImpl() const override;

        /**
         * @brief Compare every address and scalar embedded in captured nodes.
         * @param other Candidate immutable binding.
         * @return true only when the same executable may be reused safely.
         */
        [[nodiscard]] bool hasSameCaptureIdentity(
            const Params &other) const noexcept;

        /** @return Immutable stage parameters for lifecycle validation. */
        [[nodiscard]] const Params &getParams() const noexcept
        {
            return params_;
        }

    private:
        /** @brief Validate participant, arena identity, and request-slot bounds. */
        [[nodiscard]] bool validate() const noexcept;

        Params params_;
        std::unique_ptr<IMoEKernel> moe_kernel_;
    };
} // namespace llaminar2
