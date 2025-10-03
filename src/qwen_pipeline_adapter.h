/**
 * @file qwen_pipeline_adapter.h
 * @brief Adapter wrapping legacy MPITransformerPipeline behind AbstractPipeline interface.
 */
#pragma once

#include "abstract_pipeline.h"
#include "mpi_transformer_pipeline.h"
#include <memory>

namespace llaminar
{
    struct QwenModelWeights : public IModelWeights
    {
        MPITransformerPipeline::ModelWeights inner;
    };

    class QwenPipelineAdapter : public AbstractPipeline
    {
    public:
        explicit QwenPipelineAdapter(const TransformerLayerConfig &cfg);
        const TransformerLayerConfig &config() const override { return cfg_; }
        bool prefill(const std::vector<int> &tokens, const IModelWeights &weights, StageContext &ctx) override;
        bool decode(int next_token, const IModelWeights &weights, StageContext &ctx) override;
        bool logits(std::shared_ptr<TensorBase> &out_logits) override;
        std::string name() const override { return "QwenPipelineAdapter"; }
        const KVCacheState *kvCacheState() const override;
        bool ensureKVCapacity(int required_tokens) override;

    private:
        TransformerLayerConfig cfg_;
        std::unique_ptr<MPITransformerPipeline> legacy_;
        std::shared_ptr<TensorBase> last_logits_;
        std::vector<int> current_tokens_;
    };

    // Registration helper
    void registerQwenPipeline();
} // namespace llaminar
