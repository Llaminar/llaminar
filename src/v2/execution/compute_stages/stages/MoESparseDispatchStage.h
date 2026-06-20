/**
 * @file MoESparseDispatchStage.h
 * @brief Graph-native sparse MoE dispatch payload stage.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"
#include "MoEExpertDispatchStage.h"
#include "../../moe/MoEOverlaySparseCollective.h"

#include <memory>
#include <optional>

namespace llaminar2
{
    class TensorBase;

    class MoESparseDispatchStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            IMoEOverlaySparseCollectiveContext *collective_context = nullptr;
            std::shared_ptr<IMoEOverlaySparseCollectiveContext> collective_context_lifetime;
            MoEOverlayCollectiveWorkspace *workspace = nullptr;
            std::shared_ptr<MoEOverlayCollectiveWorkspace> workspace_lifetime;
            MoEOverlayCollectiveKey key;
            int source_participant = -1;
            int target_participant = -1;

            const TensorBase *hidden = nullptr;
            const TensorBase *routing_indices = nullptr;
            const TensorBase *routing_weights = nullptr;
            std::optional<BufferId> hidden_buffer_id;
            std::optional<BufferId> routing_indices_buffer_id;
            std::optional<BufferId> routing_weights_buffer_id;
            int seq_len = 0;
            int top_k = 0;
            int d_model = 0;

            const MoEExpertTierDispatch *tier_dispatch = nullptr;
            const MoEExpertDispatchOutput *dispatch_output = nullptr;
            std::shared_ptr<MoEExpertDispatchOutput> dispatch_output_lifetime;
            int tier_index = -1;

            bool replicated_hidden_export = false;
            int logical_continuation_root_participant = -1;
            bool manual_boundary_requires_collective_completion = true;

            MoEOverlaySparseRows *inbound_rows = nullptr;
            std::shared_ptr<MoEOverlaySparseRows> inbound_rows_lifetime;
        };

        static_assert(StageParamsRequired<Params>);

        explicit MoESparseDispatchStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_SPARSE_DISPATCH; }
        std::string name() const override { return "moe_sparse_dispatch"; }
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return false; }
        bool isManualGraphBoundary() const override { return true; }
        bool manualGraphBoundaryComplete() const override
        {
            return last_collective_result_.ok &&
                   (!params_.manual_boundary_requires_collective_completion ||
                    last_collective_result_.collective_complete);
        }
        bool allowsZeroOutput() const override { return true; }
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        const Params &params() const { return params_; }

    private:
        Params params_;
        MoEOverlayCollectiveResult last_collective_result_{};
        uint64_t execution_count_ = 0;
    };

} // namespace llaminar2
