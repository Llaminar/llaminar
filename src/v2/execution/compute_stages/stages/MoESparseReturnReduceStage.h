/**
 * @file MoESparseReturnReduceStage.h
 * @brief Graph-native sparse MoE return/reduce payload stage.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../moe/MoEOverlaySparseCollective.h"

namespace llaminar2
{
    class ITPContext;
    class TensorBase;

    class MoESparseReturnReduceStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            IMoEOverlaySparseCollectiveContext *collective_context = nullptr;
            std::shared_ptr<IMoEOverlaySparseCollectiveContext> collective_context_lifetime;
            std::shared_ptr<MoEOverlayCollectiveWorkspace> workspace_lifetime;
            MoEOverlayCollectiveKey key;
            int source_participant = -1;
            int target_participant = -1;
            const MoEOverlayReturnRows *outbound_rows = nullptr;
            std::shared_ptr<const MoEOverlayReturnRows> outbound_rows_lifetime;
            MoEOverlayReturnRows *inbound_rows = nullptr;
            std::shared_ptr<MoEOverlayReturnRows> inbound_rows_lifetime;
            TensorBase *dense_output = nullptr;
            int seq_len = 0;
            int d_model = 0;
            bool clear_output_before_scatter = false;

            ITPContext *continuation_tp_context = nullptr;
            bool broadcast_after_scatter = false;
            int continuation_root_tp_index = 0;
        };

        static_assert(StageParamsRequired<Params>);

        explicit MoESparseReturnReduceStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_SPARSE_RETURN_REDUCE; }
        std::string name() const override { return "moe_sparse_return_reduce"; }
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return false; }
        bool isManualGraphBoundary() const override { return true; }
        bool manualGraphBoundaryComplete() const override
        {
            return last_collective_result_.ok && last_collective_result_.collective_complete;
        }
        bool allowsZeroOutput() const override { return true; }
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        const Params &params() const { return params_; }

    private:
        Params params_;
        MoEOverlayCollectiveResult last_collective_result_{};
        uint64_t execution_count_ = 0;
    };

} // namespace llaminar2
