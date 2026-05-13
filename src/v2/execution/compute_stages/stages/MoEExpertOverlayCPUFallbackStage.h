/**
 * @file MoEExpertOverlayCPUFallbackStage.h
 * @brief Graph-integrated NodeLocalTP CPU fallback stage for MoE expert overlays.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../moe/MoEExpertOverlayCPUFallback.h"

namespace llaminar2
{
    struct MoEExpertDispatchOutput;

    class MoEExpertOverlayCPUFallbackStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ExpertComputeDomain domain;
            int root_world_rank = 0;
            int domain_id = -1;
            std::string hostfile_path;

            TensorBase *input = nullptr;
            TensorBase *routing_indices = nullptr;
            TensorBase *routing_weights = nullptr;
            TensorBase *gate_exps = nullptr;
            TensorBase *up_exps = nullptr;
            TensorBase *down_exps = nullptr;
            TensorBase *output = nullptr;

            int seq_len = 0;
            int d_model = 0;
            int num_experts = 0;
            int top_k = 0;
            int expert_intermediate = 0;
            int layer_idx = -1;

            std::vector<bool> expert_mask;
            MoEExpertTransferMode transfer_mode = MoEExpertTransferMode::Auto;
            std::vector<int> transfer_token_rows;
            const MoEExpertDispatchOutput *dispatch_output = nullptr;
            std::shared_ptr<MoEExpertDispatchOutput> dispatch_output_lifetime;
            int dispatch_tier_index = -1;
            std::shared_ptr<const MoECPUFallbackDomainContext> domain_context;
            PreparedWeightStore *prepared_store = nullptr;
            MoECPUFallbackTensorParallelStats *tensor_parallel_stats = nullptr;
            MoECPUFallbackTransferStats *transfer_stats = nullptr;
        };

        static_assert(StageParamsRequired<Params>);

        explicit MoEExpertOverlayCPUFallbackStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_EXPERT_FFN; }
        std::string name() const override { return "moe_expert_overlay_cpu_fallback"; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return false; }
        bool allowsZeroOutput() const override { return true; }
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        const Params &params() const { return params_; }

    private:
        Params params_;
    };

} // namespace llaminar2
