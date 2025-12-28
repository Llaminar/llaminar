/**
 * @file MoEStages.h
 * @brief MoE (Mixture of Experts) stages: Router, Expert, Combine
 */

#pragma once

#include "../IComputeStage.h"

namespace llaminar2
{

    /**
     * @brief MoE router stage: compute expert selection
     */
    class MoERouterStage : public IComputeStage
    {
    public:
        struct Params
        {
            const TensorBase *hidden = nullptr;
            const TensorBase *gate_weights = nullptr;
            TensorBase *router_logits = nullptr;
            int seq_len = 0;
            int d_model = 0;
            int num_experts = 0;
        };

        explicit MoERouterStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_ROUTER; }
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo getDumpInfo() const override;

    private:
        Params params_;
    };

    /**
     * @brief Single expert FFN execution
     *
     * This stage handles tokens routed to one specific expert.
     * Multiple MoEExpertStages can execute in parallel on different devices.
     */
    class MoEExpertStage : public IComputeStage
    {
    public:
        struct Params
        {
            int expert_id = 0;
            const TensorBase *input = nullptr;
            TensorBase *output = nullptr;
            const TensorBase *gate_w = nullptr;
            const TensorBase *up_w = nullptr;
            const TensorBase *down_w = nullptr;
            const std::vector<int> *token_indices = nullptr;
            int d_model = 0;
            int intermediate_dim = 0;
        };

        explicit MoEExpertStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_EXPERT_FFN; }
        std::string name() const override;
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo getDumpInfo() const override;

    private:
        Params params_;
    };

    /**
     * @brief Combine expert outputs with router weights
     */
    class MoECombineStage : public IComputeStage
    {
    public:
        struct Params
        {
            const std::vector<const TensorBase *> *expert_outputs = nullptr;
            const std::vector<float> *expert_weights = nullptr;
            const std::vector<int> *token_expert_map = nullptr;
            TensorBase *output = nullptr;
            int seq_len = 0;
            int d_model = 0;
            int top_k = 0;
        };

        explicit MoECombineStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_COMBINE; }
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo getDumpInfo() const override;

    private:
        Params params_;
    };

} // namespace llaminar2
