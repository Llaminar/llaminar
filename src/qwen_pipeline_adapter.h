/**
 * @file qwen_pipeline_adapter.h
 * @brief Adapter wrapping DistributedTransformerPipeline behind AbstractPipeline interface.
 */
#pragma once

#include "abstract_pipeline.h"
#include "qwen_pipeline.h"
#include "weight_contracts.h"
#include <memory>

namespace llaminar
{
    /**
     * @brief Qwen-specific model weights implementing IModelWeights interface.
     *
     * Wraps DistributedTransformerPipeline::ModelWeights and provides typed accessors
     * for architecture-agnostic code to access common weight components.
     *
     * IMPORTANT: All weights are guaranteed to match the canonical GGUF format:
     * - Q/K/V: [out_features, in_features] = [n_head*head_dim, d_model]
     * - Output: [in_features, out_features] = [d_model, n_head*head_dim]
     *
     * This contract is validated at load time via weight_contracts.h
     */
    struct QwenModelWeights : public IModelWeights
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

        /**
         * @brief Validate all weights against canonical contracts.
         *
         * Ensures loaded weights match expected GGUF format. Called automatically
         * during loading to fail fast with clear error messages.
         *
         * @param cfg Model configuration
         * @throws std::runtime_error if any weight violates its contract
         */
        void validate(const TransformerLayerConfig &cfg) const
        {
            auto contracts = getQwenWeightContracts();

            // Validate global weights
            contracts.validate_global(inner.token_embedding, inner.output_norm_weight,
                                      inner.lm_head, cfg);

            // Validate each layer
            for (int layer = 0; layer < layer_count(); ++layer)
            {
                contracts.validate_layer(layer,
                                         inner.attn_norm_weight[layer],
                                         inner.wq[layer],
                                         inner.wk[layer],
                                         inner.wv[layer],
                                         inner.wo[layer],
                                         inner.ffn_norm_weight[layer],
                                         inner.w_gate[layer],
                                         inner.w_up[layer],
                                         inner.w_down[layer],
                                         cfg);
            }
        }
    };

    class QwenPipelineAdapter : public AbstractPipeline
    {
    public:
        explicit QwenPipelineAdapter(const ModelConfig &cfg);
        const ModelConfig &config() const override { return cfg_; }
        bool prefill(const std::vector<int> &tokens, const IModelWeights &weights, StageContext &ctx) override;
        bool decode(int next_token, const IModelWeights &weights, StageContext &ctx) override;
        bool logits(std::shared_ptr<TensorBase> &out_logits) override;
        std::string name() const override { return "QwenPipelineAdapter"; }
        const KVCacheState *kvCacheState() const override;
        bool ensureKVCapacity(int required_tokens) override;

        /**
         * @brief Load Qwen model weights from file.
         *
         * @param path Path to GGUF model file
         * @return QwenModelWeights wrapping the loaded weights
         */
        std::unique_ptr<IModelWeights> loadWeights(const std::string &path) override;

    private:
        ModelConfig cfg_;
        std::unique_ptr<QwenPipeline> legacy_;
        std::shared_ptr<TensorBase> last_logits_;
        std::vector<int> current_tokens_;
    };

    // Registration helper
    void registerQwenPipeline();
} // namespace llaminar
