/**
 * @file MoEDeviceDecodeCommitBoundaryStage.h
 * @brief Captured ROCm serial-decode publication for device MoE cadence.
 *
 * HIP does not provide CUDA-style conditional graph nodes.  Homogeneous ROCm
 * ExpertOverlay therefore lets the host observe a compact authenticated ticket
 * only at conservatively scheduled boundaries. Ordinary decode and explicitly
 * committed scalar MTP conditions both own new serial rows. Serial decode still
 * owns its cadence value on device: this stage appends the publish/acknowledge
 * pair to the complete main decode graph so a non-due token never requires a
 * second graph launch, event fence, host callback, or host mirror.
 *
 * The persistent controller storage is allocated by the retained maintenance
 * graph and shared by exact workspace name.  This stage declares and consumes
 * that same graph-family buffer.  Request-varying counters remain behind the
 * stable device address embedded during capture.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"

#include "../../../interfaces/IWorkspaceConsumer.h"

#include <string>

namespace llaminar2
{
    class IBackend;
    class DeviceWorkspaceManager;

    /**
     * @brief Publish one serial round inside ordinary or committed-MTP HIP graphs.
     *
     * The first backend kernel advances the device-resident cadence exactly
     * once unless grouped MTP already published the same edge.  The second
     * kernel clears the edge immediately when maintenance is not due and leaves
     * a due edge intact for the authenticated host ticket dispatcher.  Both
     * kernels run on the exact stream selected for this graph stage.
     *
     * CUDA deliberately does not use this stage: its native conditional parent
     * owns `publish -> IF(due) maintenance -> acknowledge` as one transaction.
     */
    class MoEDeviceDecodeCommitBoundaryStage final
        : public IComputeStage,
          public IWorkspaceConsumer
    {
    public:
        /** @brief Immutable graph-family bindings captured by the stage. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            IBackend *backend = nullptr;
            std::string workspace_name;
            std::string stage_name =
                "moe_device_decode_commit_boundary";
        };

        static_assert(StageParamsRequired<Params>);

        /**
         * @brief Construct a serial-decode cadence boundary.
         * @param params ROCm participant, backend, and maintenance-workspace identity.
         * @throws std::invalid_argument when an immutable binding is incomplete.
         */
        explicit MoEDeviceDecodeCommitBoundaryStage(Params params);

        /**
         * @brief Enqueue publish then non-due acknowledgement on the graph stream.
         * @param ctx Exact ROCm device context selected by the executor.
         * @return true only when both graph-capturable kernels were enqueued.
         */
        bool execute(IDeviceContext *ctx) override;

        /** @return Stable operation type used by capture and diagnostics. */
        ComputeStageType type() const override
        {
            return ComputeStageType::MOE_DEVICE_DECODE_COMMIT_BOUNDARY;
        }

        /** @return Human-readable stage identity. */
        std::string name() const override { return params_.stage_name; }

        /** @return Constant control-flow work performed by the two kernels. */
        size_t estimatedFlops() const override { return 32u; }

        /** @return Approximate controller bytes read or written by both kernels. */
        size_t estimatedMemoryBytes() const override;

        /** @return true only for the ROCm backend that needs hosted ticket dispatch. */
        bool supportsBackend(ComputeBackendType backend) const override;

        /** @return true; execution contains only two allocation-free HIP kernels. */
        bool isGraphCapturable() const override { return true; }

        /** @return false; cadence publication is participant-local. */
        bool isCollectiveStage() const override { return false; }

        /** @return NONE because the controller is raw graph-family workspace. */
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }

        /** @return Empty tensor contract; no activation tensor is read or written. */
        StageBufferContract bufferContract() const override { return {}; }

        /**
         * @brief Declare the shared persistent controller buffer by exact name.
         * @param m Ignored activation-row hint.
         * @param n Ignored output-column hint.
         * @param k Ignored input-column hint.
         * @return One required controller-state workspace descriptor.
         */
        WorkspaceRequirements getWorkspaceRequirements(
            int m,
            int n = 0,
            int k = 0) const override;

        /** @brief Bind the graph-family workspace without allocating storage. */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;

        /** @brief Drop the non-owning workspace binding during hard invalidation. */
        void unbindWorkspace() override;

        /** @return true when a graph-family workspace has been bound. */
        bool hasWorkspace() const override
        {
            return bound_workspace_ != nullptr;
        }

        /** @return Non-owning workspace used to resolve the persistent controller. */
        DeviceWorkspaceManager *getWorkspace() const override
        {
            return bound_workspace_;
        }

        /** @return Scalar-only diagnostic description of the cadence boundary. */
        StageDumpInfo buildDumpInfoImpl() const override;

        /** @return Immutable captured parameters for topology tests. */
        [[nodiscard]] const Params &getParams() const noexcept
        {
            return params_;
        }

    private:
        /** @return Exact shared controller buffer name in the workspace family. */
        [[nodiscard]] std::string controllerBufferName() const;

        /** @return true when immutable ROCm and workspace bindings are complete. */
        [[nodiscard]] bool validate() const noexcept;

        Params params_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
    };
} // namespace llaminar2
