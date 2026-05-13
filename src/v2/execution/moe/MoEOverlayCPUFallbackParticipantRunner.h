/**
 * @file MoEOverlayCPUFallbackParticipantRunner.h
 * @brief Non-root MoE overlay CPU fallback participant runner.
 */

#pragma once

#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/moe/MoEExpertParallelPlan.h"
#include "loaders/ModelContext.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    class IMPIContext;
    class IDeviceContext;
    class PreparedWeightStore;

    class MoEOverlayCPUFallbackParticipantRunner final : public IInferenceRunner
    {
    public:
        struct Config
        {
            std::shared_ptr<ModelContext> model_ctx;
            std::shared_ptr<IMPIContext> mpi_ctx;
            std::shared_ptr<const MoEExpertParallelPlan> overlay_plan;
            std::shared_ptr<const MoEExpertOverlayExecutionPlan> execution_plan;
            std::string hostfile_path;

            // Temporary compatibility bridge for direct/native fallback loops.
            // Graph-native endpoint ranks must not enter NodeLocalTP collectives
            // unless a dispatch backend has explicitly opted into that behavior.
            bool enable_native_compatibility_fallback = false;
        };

        explicit MoEOverlayCPUFallbackParticipantRunner(Config config);
        ~MoEOverlayCPUFallbackParticipantRunner() override;

        bool initialize();

        bool forward(const int *tokens, int seq_len) override;
        const float *logits() const override { return nullptr; }
        int vocab_size() const override;
        void clear_cache() override;
        int get_position() const override { return position_; }
        ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
        const char *architecture() const override;

    private:
        struct LayerFallbackWeights
        {
            std::shared_ptr<TensorBase> gate;
            std::shared_ptr<TensorBase> up;
            std::shared_ptr<TensorBase> down;
        };

        struct Scratch
        {
            int seq_len = 0;
            std::shared_ptr<TensorBase> input;
            std::shared_ptr<TensorBase> routing_indices;
            std::shared_ptr<TensorBase> routing_weights;
            std::shared_ptr<TensorBase> output;
        };

        const ExpertLayerPlacement *placementForLayer(int layer_idx) const;
        const ExpertComputeDomain *domainForName(const std::string &domain_name) const;
        int cpuFallbackTierIndex() const;
        std::vector<bool> expertMaskForTier(const ExpertLayerPlacement &placement, int tier_index) const;
        bool hasActiveExpertMask(const std::vector<bool> &mask) const;
        bool isMoELayer(int layer_idx) const;
        bool loadLayerWeights(int layer_idx, LayerFallbackWeights &weights);
        LayerFallbackWeights *cachedLayerWeights(int layer_idx);
        bool ensureScratch(int seq_len);
        int expertIntermediate(const LayerFallbackWeights &weights) const;
        bool executeLayerFallback(int layer_idx, int tier_index, const ExpertComputeDomain &domain,
                                  const std::vector<bool> &expert_mask, const LayerFallbackWeights &weights);

        Config config_;
        std::unique_ptr<IDeviceContext> cpu_ctx_;
        std::shared_ptr<PreparedWeightStore> prepared_store_;
        std::unordered_map<int, LayerFallbackWeights> layer_weights_;
        Scratch scratch_;
        std::string architecture_;
        int d_model_ = 0;
        int num_experts_ = 0;
        int top_k_ = 0;
        int expert_intermediate_ = 0;
        int position_ = 0;
        bool initialized_ = false;
    };

    std::unique_ptr<IInferenceRunner> createMoEOverlayCPUFallbackParticipantRunner(
        MoEOverlayCPUFallbackParticipantRunner::Config config);

} // namespace llaminar2