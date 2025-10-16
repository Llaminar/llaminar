/**
 * @file QwenPipelineAdapter.h
 * @brief Adapter wrapping DistributedTransformerPipeline behind AbstractPipeline interface.
 */
#pragma once

#include "AbstractPipeline.h"
#include "QwenPipeline.h"
#include "WeightContracts.h"
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
     * This contract is validated at load time via WeightContracts.h
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
            validate_with_mpi(cfg, 0, 1);
        }

        /**
         * @brief Validate all weights with MPI context (slicing-aware).
         *
         * @param cfg Model configuration
         * @param mpi_rank Current MPI rank
         * @param mpi_size Total MPI world size
         * @throws std::runtime_error if any weight violates its contract
         */
        void validate_with_mpi(const TransformerLayerConfig &cfg, int mpi_rank, int mpi_size) const
        {
            auto contracts = getQwenWeightContracts();

            // Validate global weights
            contracts.validate_global_with_mpi(inner.token_embedding, inner.output_norm_weight,
                                               inner.lm_head, cfg, mpi_rank, mpi_size);

            // Validate each layer
            int n_layers = layer_count();
            LOG_INFO("[WeightContract] Validating " << n_layers << " layers (rank "
                                                    << mpi_rank << "/" << mpi_size << ")");

            for (int layer = 0; layer < n_layers; ++layer)
            {
                contracts.validate_layer_with_mpi(layer,
                                                  inner.attn_norm_weight[layer],
                                                  inner.wq[layer],
                                                  inner.wk[layer],
                                                  inner.wv[layer],
                                                  inner.wo[layer],
                                                  inner.ffn_norm_weight[layer],
                                                  inner.w_gate[layer],
                                                  inner.w_up[layer],
                                                  inner.w_down[layer],
                                                  cfg,
                                                  mpi_rank,
                                                  mpi_size);

                // Log progress every 5 layers
                if ((layer + 1) % 5 == 0 || layer == n_layers - 1)
                {
                    LOG_INFO("[WeightContract] ✓ Validated layers 0-" << layer << " (" << (layer + 1)
                                                                      << "/" << n_layers << ")");
                }
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
        bool prefillBatch(const std::vector<std::vector<int>> &token_batches,
                          const IModelWeights &weights,
                          StageContext &ctx,
                          std::shared_ptr<TensorBase> &out_logits) override;
        bool decodeBatch(const std::vector<int> &next_tokens,
                         const IModelWeights &weights,
                         StageContext &ctx,
                         std::shared_ptr<TensorBase> &out_logits) override;
        bool logits(std::shared_ptr<TensorBase> &out_logits) override;
        std::string name() const override { return "QwenPipelineAdapter"; }
        const KVCacheState *kvCacheState() const override;
        bool ensureKVCapacity(int required_tokens) override;

        /**
         * @brief Forward snapshot capture to wrapped legacy pipeline
         *
         * @note This ensures snapshot_source_ from the wrapped QwenPipeline is used
         */
        void captureStageSnapshot(
            PipelineStage stage,
            int layer_index,
            const float *data,
            int seq_len,
            int feature_dim,
            const std::string &source = "llaminar") override
        {
            if (legacy_)
            {
                // Call through AbstractPipeline interface (legacy_ implements AbstractPipeline)
                static_cast<AbstractPipeline *>(legacy_.get())->captureStageSnapshot(stage, layer_index, data, seq_len, feature_dim, source);
            }
        }

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
