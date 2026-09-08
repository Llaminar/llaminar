/**
 * @file MoEOverlayDeviceControllerStage.h
 * @brief Captured graph stage for one sole-authority ExpertOverlay transition.
 *
 * A heterogeneous all-GPU overlay cannot let CUDA, ROCm, or rank-local
 * controllers make independent placement decisions. This stage embeds one
 * immutable participant binding and one typed transition into a participant's
 * production graph. Live lifecycle and policy values remain behind device
 * pointers; the host can neither select a transaction branch nor mirror an
 * epoch during replay.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"

#include "../../../execution/moe/MoEOverlayNodeLocalDeviceControllerFabric.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../kernels/IMoEKernel.h"

#include <memory>
#include <string>

namespace llaminar2
{
    /**
     * @brief One graph-capturable action in the mapped controller protocol.
     *
     * Role validity is checked at construction: global mutations require the
     * sole authority leader and group acknowledgements require a group root.
     * `PublishCommand` reads its device-authored policy result from persistent
     * workspace so capture embeds no request-varying host value.
     */
    class MoEOverlayDeviceControllerStage final
        : public IComputeStage,
          public IWorkspaceConsumer
    {
    public:
        /** Stable shared suffix for the leader policy output allocation. */
        static constexpr const char *kPolicyResultBuffer =
            "moe_overlay_device_controller_policy_result";

        /** Immutable graph identity for one participant transition. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;
            MoEOverlayDeviceControllerParticipantBinding binding;
            MoEOverlayDeviceControllerAction action =
                MoEOverlayDeviceControllerAction::Invalid;
            MoEOverlayDeviceControllerTransactionKind transaction_kind =
                MoEOverlayDeviceControllerTransactionKind::Invalid;
            /** Immutable phase identity for Dynamic policy authoring. */
            MoEOverlayDeviceDemandPhase demand_phase =
                MoEOverlayDeviceDemandPhase::Invalid;
            std::string workspace_name = "moe_overlay_device_controller";
            std::string stage_name = "moe_overlay_device_controller";
        };

        static_assert(StageParamsRequired<Params>);

        /**
         * @brief Freeze role, mapped aliases, action, and backend kernel.
         * @throws std::invalid_argument for a role/action mismatch.
         * @throws std::runtime_error when no backend implementation exists.
         */
        explicit MoEOverlayDeviceControllerStage(Params params);

        /** Enqueue exactly one controller action on the stage-owned stream. */
        bool execute(IDeviceContext *ctx) override;

        /** @return Stable diagnostic type for controller transitions. */
        ComputeStageType type() const override
        {
            return ComputeStageType::MOE_OVERLAY_DEVICE_CONTROLLER;
        }

        /** @return Caller-selected role name used in graph traces. */
        std::string name() const override { return params_.stage_name; }

        /** @return Constant control-kernel arithmetic estimate. */
        size_t estimatedFlops() const override { return 256u; }

        /** @return Maximum mapped control/snapshot bytes read by this action. */
        size_t estimatedMemoryBytes() const override;

        /** @return True for CUDA and ROCm participant graphs only. */
        bool supportsBackend(ComputeBackendType backend) const override;

        /** @return True; the action allocates, copies, and synchronizes nothing. */
        bool isGraphCapturable() const override { return true; }

        /** @return False; cross-group ordering is the mapped protocol itself. */
        bool isCollectiveStage() const override { return false; }

        /** @return NONE because mapped control pages are not Tensor resources. */
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }

        /** @return Empty tensor contract; storage is fabric/workspace-owned. */
        StageBufferContract bufferContract() const override { return {}; }

        /** @return Leader policy-result storage required by PublishCommand. */
        WorkspaceRequirements getWorkspaceRequirements(
            int m,
            int n = 0,
            int k = 0) const override;

        /** Bind the model-lifetime device workspace before capture. */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;

        /** Remove the non-owning workspace binding during graph invalidation. */
        void unbindWorkspace() override;

        /** @return Whether a persistent workspace is currently bound. */
        bool hasWorkspace() const override { return workspace_ != nullptr; }

        /** @return Non-owning persistent workspace binding. */
        DeviceWorkspaceManager *getWorkspace() const override
        {
            return workspace_;
        }

        /** @return Immutable stage parameters for graph validation. */
        const Params &getParams() const noexcept { return params_; }

        /**
         * @brief Compare every pointer/scalar embedded in captured launch args.
         * @return True only when an existing executable is safe to reuse.
         */
        [[nodiscard]] bool hasSameCaptureIdentity(
            const Params &other) const noexcept;

    protected:
        /** @return Scalar-only dump metadata; no live mapped bytes are copied. */
        StageDumpInfo buildDumpInfoImpl() const override;

    private:
        /** @return Whether action, role, device, and transaction kind agree. */
        [[nodiscard]] bool validate() const noexcept;

        /** @return Stable workspace name shared with the device policy stage. */
        [[nodiscard]] std::string policyResultBufferName() const;

        Params params_;
        std::unique_ptr<IMoEKernel> moe_kernel_;
        DeviceWorkspaceManager *workspace_ = nullptr;
    };
} // namespace llaminar2
