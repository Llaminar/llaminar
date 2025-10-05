/**
 * @file llama_pipeline_adapter.h
 * @brief Adapter wrapping DistributedTransformerPipeline for Llama models.
 * @author David Sanftenberg
 *
 * This is a stub implementation that validates the multi-architecture interface.
 * Initially reuses the Qwen/DistributedTransformerPipeline implementation since
 * both architectures share similar transformer structures.
 *
 * Future enhancements can specialize Llama-specific features:
 * - Different attention mechanisms (e.g., GQA)
 * - Llama-specific normalization (RMSNorm variations)
 * - Rope frequency adjustments
 */
#pragma once

#include "abstract_pipeline.h"
#include "qwen_pipeline.h"
#include <memory>

namespace llaminar
{
    /**
     * @brief Llama-specific model weights implementing IModelWeights interface.
     *
     * Wraps DistributedTransformerPipeline::ModelWeights and provides typed accessors
     * for architecture-agnostic code. Currently identical to QwenModelWeights structure
     * since both use the same underlying weight layout.
     */
    struct LlamaModelWeights : public IModelWeights
    {
        QwenPipeline::ModelWeights inner;

        // Typed accessors for weight components
        const std::shared_ptr<TensorBase> &embedding() const { return inner.token_embedding; }
        const std::shared_ptr<TensorBase> &lm_head() const { return inner.lm_head; }
        int layer_count() const { return static_cast<int>(inner.wq.size()); }
        const std::shared_ptr<TensorBase> &output_norm() const { return inner.output_norm_weight; }

        // Layer-specific accessors
        const std::shared_ptr<TensorBase> &attn_norm(int layer) const { return inner.attn_norm_weight[layer]; }
        const std::shared_ptr<TensorBase> &wq(int layer) const { return inner.wq[layer]; }
        const std::shared_ptr<TensorBase> &wk(int layer) const { return inner.wk[layer]; }
        const std::shared_ptr<TensorBase> &wv(int layer) const { return inner.wv[layer]; }
        const std::shared_ptr<TensorBase> &wo(int layer) const { return inner.wo[layer]; }
        const std::shared_ptr<TensorBase> &ffn_norm(int layer) const { return inner.ffn_norm_weight[layer]; }
        const std::shared_ptr<TensorBase> &w_gate(int layer) const { return inner.w_gate[layer]; }
        const std::shared_ptr<TensorBase> &w_up(int layer) const { return inner.w_up[layer]; }
        const std::shared_ptr<TensorBase> &w_down(int layer) const { return inner.w_down[layer]; }
    };

    /**
     * @brief Llama pipeline adapter implementing AbstractPipeline interface.
     *
     * This stub implementation delegates to DistributedTransformerPipeline,
     * validating the multi-architecture factory pattern. Future versions can
     * specialize Llama-specific optimizations while maintaining the same interface.
     */
    class LlamaPipelineAdapter : public AbstractPipeline
    {
    public:
        explicit LlamaPipelineAdapter(const ModelConfig &cfg);
        const ModelConfig &config() const override { return cfg_; }
        bool prefill(const std::vector<int> &tokens, const IModelWeights &weights, StageContext &ctx) override;
        bool decode(int next_token, const IModelWeights &weights, StageContext &ctx) override;
        bool logits(std::shared_ptr<TensorBase> &out_logits) override;
        std::string name() const override { return "LlamaPipelineAdapter"; }
        const KVCacheState *kvCacheState() const override;
        bool ensureKVCapacity(int required_tokens) override;

        /**
         * @brief Load Llama model weights from file.
         *
         * @param path Path to GGUF model file
         * @return LlamaModelWeights wrapping the loaded weights
         */
        std::unique_ptr<IModelWeights> loadWeights(const std::string &path) override;

    private:
        ModelConfig cfg_;
        std::unique_ptr<QwenPipeline> legacy_;
        std::shared_ptr<TensorBase> last_logits_;
        std::vector<int> current_tokens_;
    };

    /**
     * @brief Register Llama pipeline with the factory.
     *
     * Call once during initialization to enable "llama" architecture support.
     */
    void registerLlamaPipeline();

} // namespace llaminar
