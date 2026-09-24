/**
 * @file MoEGPUCurrentBatchLLEPStage.h
 * @brief Explicit graph phases for GPU current-batch LLEP weight movement.
 *
 * Current-batch least-loaded expert placement consumes router output before it
 * can know which expert weights are missing. The selected foreign experts must
 * then arrive before routed FFN begins. This stage exposes the two local halves
 * of that transaction so the model graph can place a real collective between
 * them instead of hiding a standalone allgather inside expert compute.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../execution/moe/DeviceMoERebalanceController.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../memory/BufferId.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace llaminar2
{
    class DeviceWorkspaceManager;
    class ILocalTPContext;
    class IMoEKernel;
    class IMoERuntimeTable;
    class ITensor;

    /**
     * @brief Select one side of the graph-visible current-batch transaction.
     *
     * PlanAndPack consumes router output and publishes a fixed-size local byte
     * lane. UnpackApplyAndAssign consumes the all-gathered lane after an
     * explicit model collective and publishes final participant assignments.
     * There is deliberately no phase that launches its own collective.
     */
    enum class GPUCurrentBatchLLEPPhase : std::uint8_t
    {
        PlanAndPack = 0,
        UnpackApplyAndAssign = 1,
    };

    /**
     * @brief Graph-capturable local halves of one GPU current-batch LLEP move.
     *
     * Both instances bind the same persistent workspace names and request-local
     * child runtime table. A TP allreduce or rooted collective between them
     * carries the byte payload as an allgather sideband. This class owns only
     * local kernels: collective ordering remains visible in ComputeGraph and is
     * therefore symmetric across all LocalTP participants.
     */
    class MoEGPUCurrentBatchLLEPStage final
        : public IComputeStage,
          public IWorkspaceConsumer
    {
    public:
        /** @brief Immutable topology and storage bindings for one phase. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            GPUCurrentBatchLLEPPhase phase =
                GPUCurrentBatchLLEPPhase::PlanAndPack;
            ILocalTPContext *tp_ctx = nullptr;
            IMoERuntimeTable *moe_runtime_table = nullptr;
            ITensor *routing_indices = nullptr;
            ITensor *routing_weights = nullptr;
            std::optional<BufferId> routing_indices_buffer_id;
            std::optional<BufferId> routing_weights_buffer_id;

            int tp_device_idx = -1;
            int layer_idx = -1;
            int seq_len = 0;
            int num_experts = 0;
            int top_k = 0;

            DeviceMoERebalanceConfig config;
            DeviceMoEExpertDirectoryEntry *local_transfer_slots = nullptr;
            std::uint32_t local_transfer_slot_count = 0;
            std::uint64_t payload_slot_bytes = 0;
            std::uint32_t payload_slot_capacity = 0;
            DeviceMoERebalanceTransferMode transfer_mode =
                DeviceMoERebalanceTransferMode::ResidentOnly;

            std::string workspace_name;
            std::string stage_name;
        };

        static_assert(StageParamsRequired<Params>);

        /**
         * @brief Construct one phase without allocating backend resources.
         * @param params Complete graph-owned transaction identity.
         */
        explicit MoEGPUCurrentBatchLLEPStage(Params params);

        /** @brief Destroy the stage-owned backend kernel outside graph replay. */
        ~MoEGPUCurrentBatchLLEPStage() override;

        /**
         * @brief Execute only the local work belonging to the selected phase.
         * @param ctx Exact device context supplied by the graph executor.
         * @return true after every kernel is enqueued on the bound stream.
         */
        bool execute(IDeviceContext *ctx) override;

        /** @return Stable type used by graph topology tests and diagnostics. */
        ComputeStageType type() const override
        {
            return ComputeStageType::MOE_GPU_CURRENT_BATCH_LLEP;
        }

        /** @return Graph-provided semantic phase name. */
        std::string name() const override;

        /** @return false because the intervening TP stage owns the collective. */
        bool isCollectiveStage() const override { return false; }

        /** @return true for compiled CUDA or ROCm backends only. */
        bool supportsBackend(ComputeBackendType backend) const override;

        /**
         * @brief Report capture readiness after the backend kernel is prepared.
         * @return true only when stable workspace and kernel state are bound.
         */
        bool isGraphCapturable() const override;

        /** @return true because prepareGraphLaunch creates stable kernel state. */
        bool supportsGraphCaptureAfterLaunchPreparation() const override
        {
            return true;
        }

        /** @return true because preparation, not eager arithmetic, makes capture ready. */
        bool supportsLazyPrefillGraphCapturePreflight() const override
        {
            return true;
        }

        /** @return true because router weights mask padded suffix rows. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override
        {
            return true;
        }

        /**
         * @brief Create and bind the stage-owned kernel before native capture.
         * @param ctx Exact device context selected for capture.
         * @param stream Non-null capture stream.
         * @return true when execute can remain allocation-free.
         */
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;

        /** @return CaptureOnly because preparation installs model-lifetime state. */
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        {
            return GraphLaunchPreparationPolicy::CaptureOnly;
        }

        /** @return Routing tensor declarations for the producer phase. */
        StageBufferRequirements getBufferRequirements() const override;

        /** @return Arena read contract for router output in PlanAndPack. */
        StageBufferContract bufferContract() const override;

        /** @return NONE because runtime/workspace publication is explicitly owned. */
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }

        /**
         * @brief Declare persistent transaction buffers shared by both phases.
         * @param m Ignored; Params::seq_len fixes the captured geometry.
         * @param n Ignored.
         * @param k Ignored.
         * @return Complete fixed-address workspace bill of materials.
         */
        WorkspaceRequirements getWorkspaceRequirements(
            int m,
            int n = 0,
            int k = 0) const override;

        /** @brief Bind the graph family's persistent workspace. */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;

        /** @brief Release only the non-owning workspace binding. */
        void unbindWorkspace() override;

        /** @return Whether a persistent workspace is currently bound. */
        bool hasWorkspace() const override { return bound_workspace_ != nullptr; }

        /** @return Current non-owning workspace binding. */
        DeviceWorkspaceManager *getWorkspace() const override
        {
            return bound_workspace_;
        }

        /** @brief Drop backend dynamic state after graph invalidation. */
        void invalidateKernelDynamicState() override;

        /** @return Immutable graph parameters for topology regressions. */
        const Params &params() const noexcept { return params_; }

        /**
         * @brief Build a transaction workspace buffer name.
         * @param base_name Stable DeviceMoERebalanceStage buffer family.
         * @param workspace_name Graph-lifetime lane identity.
         * @return Fully qualified persistent-workspace name.
         */
        static std::string workspaceBufferName(
            const char *base_name,
            const std::string &workspace_name);

        /**
         * @brief Build a layer-owned status publication name.
         * @param base_name Status or apply-status buffer family.
         * @param workspace_name Rolling bulk-payload lane identity.
         * @param layer_idx Logical MoE layer that owns the status record.
         * @return Unique layer publication name.
         */
        static std::string layerPublicationBufferName(
            const char *base_name,
            const std::string &workspace_name,
            int layer_idx);

        /** @return Fixed merged command capacity for @p params. */
        static std::uint32_t planCapacity(const Params &params);

        /** @return Fixed compact payload slots for @p params. */
        static std::uint32_t payloadSlotCount(const Params &params);

        /** @return Local sideband bytes published by PlanAndPack. */
        static std::size_t localPayloadBytes(const Params &params);

    protected:
        /** @return Snapshot metadata for CSV/stage diagnostics. */
        StageDumpInfo buildDumpInfoImpl() const override;

    private:
        /** @return true when every topology and storage binding is coherent. */
        bool validateBindings(const char *context) const;

        /** @return Stage-owned kernel, creating it only outside capture. */
        IMoEKernel *ensureKernel();

        /** @return Immutable current-batch config with histogram reset enabled. */
        DeviceMoERebalanceConfig transactionConfig() const;

        /** @return Planner policy derived solely from graph-owned config. */
        least_loaded_ep::LeastLoadedExpertAssignmentConfig plannerConfig() const;

        /** @brief Execute route grouping, LLEP planning, and payload packing. */
        bool executePlanAndPack(IMoEKernel *kernel);

        /** @brief Execute payload unpack, arrival apply, and route assignment. */
        bool executeUnpackApplyAndAssign(IMoEKernel *kernel);

        Params params_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        std::unique_ptr<IMoEKernel> owned_moe_kernel_;
    };

} // namespace llaminar2
