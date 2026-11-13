/**
 * @file TPKVCacheStateAllGatherStage.h
 * @brief Exact CPU/native-GPU prefill handoff into replicated decode KV rows.
 *
 * Live projection prefixes alone enter the collective. CPU ownership is the
 * declared logical participant, including NUMA endpoints sharing DeviceId::cpu;
 * GPU ownership retains its exact producer stream and native captured launch.
 * All compact/gathered payload bytes belong to the existing admitted workspace.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../memory/BufferId.h"

#include <optional>
#include <string>

namespace llaminar2
{
    class ILocalTPContext;

    /**
     * @brief Gather TP-local K/V projection rows into full replicated K/V rows.
     *
     * Phase-split dense execution keeps prefill attention projections tensor
     * parallel, then switches decode to mirrored dense weights. The decode cache
     * therefore must be seeded with full K/V rows during prefill. This stage
     * allgathers compact local K/V rows using the exact CPU participant or
     * active GPU LocalTP stream, then deinterleaves the participant-contiguous
     * payload into per-token full-width rows. CPU empty work joins metadata
     * without touching payload; no backend changes the admitted cache layout.
     */
    class TPKVCacheStateAllGatherStage final : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        static constexpr const char *WS_GATHERED_K = "tp_kv_state_gathered_k";
        static constexpr const char *WS_GATHERED_V = "tp_kv_state_gathered_v";
        static constexpr const char *WS_COMPACT_K = "tp_kv_state_compact_k";
        static constexpr const char *WS_COMPACT_V = "tp_kv_state_compact_v";

        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ILocalTPContext *tp_ctx = nullptr;
            const ITensor *local_K = nullptr;
            const ITensor *local_V = nullptr;
            ITensor *full_K = nullptr;
            ITensor *full_V = nullptr;
            int layer_idx = -1;
            int tp_device_idx = -1;
            int tokens = 0;
            int local_kv_dim = 0;
            int full_kv_dim = 0;

            // Logical row strides for local_K/local_V. Phase-split prefill may
            // reserve full decode-width tensor storage while the projection GEMM
            // writes packed TP-local rows, so this must be explicit.
            int local_k_stride = 0;
            int local_v_stride = 0;
            std::string stage_name;
            std::optional<BufferId> local_k_buffer_id;
            std::optional<BufferId> local_v_buffer_id;
            std::optional<BufferId> full_k_buffer_id;
            std::optional<BufferId> full_v_buffer_id;
        };

        static_assert(StageParamsRequired<Params>);

        explicit TPKVCacheStateAllGatherStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::TP_KV_CACHE_STATE_ALLGATHER; }
        size_t estimatedFlops() const override { return 0; }
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }
        bool isGraphCapturable() const override;

        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override { return bound_workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return bound_workspace_; }

        const Params &getParams() const { return params_; }

    private:
        Params params_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        std::string host_k_stage_name_; ///< Retained protocol identity; no execution-time concatenation.
        std::string host_v_stage_name_; ///< Separate K/V rounds cannot cross-match.

        /**
         * @brief Execute the first-class CPU handoff on host-owned live spans.
         * @return True only after both exact gathers and row publications finish.
         *
         * Packing/deinterleaving uses byte copies in fixed participant order;
         * scratch is exclusively borrowed from the already admitted workspace.
         */
        bool executeHost();

        std::string gatheredKBufferName() const;
        std::string gatheredVBufferName() const;
        std::string compactKBufferName() const;
        std::string compactVBufferName() const;
    };

} // namespace llaminar2
