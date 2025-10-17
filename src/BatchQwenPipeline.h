/**
 * @file BatchQwenPipeline.h
 * @brief Parallel batched execution pipeline for Qwen models (foundation skeleton)
 *
 * This class provides a clean-room implementation focused on true batch-major
 * execution (all sequences flow through each layer together) rather than the
 * legacy per-sequence loop used by QwenPipeline::prefillBatch/decodeBatch.
 *
 * Phase Scope (Skeleton):
 *  - Defines public interface & state surfaces
 *  - Integrates with AbstractPipeline for selection & logits exposure
 *  - Establishes KV cache ownership via BatchedKVCache
 *  - Provides padding utilities hook (BatchPaddingUtils)
 *  - Stub implementations (return fast-fail) to allow incremental fill‑in
 *
 * Subsequent Phases (not yet implemented):
 *  1. Embedding batch path (token + position + optional RoPE staging)
 *  2. Transformer layer batched attention + FFN (shared matmuls)
 *  3. Incremental decode batch path (single-step append per sequence)
 *  4. Bucketing & dynamic scheduling
 *  5. Performance instrumentation & adaptive backend tuning
 *
 * Design Principles:
 *  - No mutation of single‑sequence QwenPipeline internals; isolation reduces risk
 *  - Minimal per-sequence branching inside performance hot loops
 *  - Clear separation between logical batch (variable lengths) and physical padded
 *    tensors (uniform [B, T_max, ...])
 *  - Reuse existing operator primitives (MPI*Operator) with added batch dims
 *
 * @author David Sanftenberg
 */
#pragma once

#include "AbstractPipeline.h"
#include "PipelineBase.h"
#include "TransformerConfig.h"
#include "ModelLoader.h"
#include "tensors/BatchedKVCache.h"
#include "BatchPaddingUtils.h"
#include "QwenPipeline.h"
#include "PipelineStages.h"
#include <memory>
#include <vector>

namespace llaminar
{

    /**
     * @brief Lightweight wrapper for Qwen weights in batch pipeline context
     *
     * Reuses QwenPipeline::ModelWeights internally but adapts to IModelWeights.
     * Provides const accessors for batch operations without copying large tensors.
     */
    struct BatchQwenWeights : public IModelWeights
    {
        QwenPipeline::ModelWeights inner;

        const std::shared_ptr<TensorBase> &embedding() const { return inner.token_embedding; }
        const std::shared_ptr<TensorBase> &lm_head() const { return inner.lm_head; }
        const std::shared_ptr<TensorBase> &output_norm() const { return inner.output_norm_weight; }
        int layer_count() const { return static_cast<int>(inner.wq.size()); }

        // Layer accessors
        const std::shared_ptr<TensorBase> &attn_norm(int layer) const { return inner.attn_norm_weight[layer]; }
        const std::shared_ptr<TensorBase> &wq(int layer) const { return inner.wq[layer]; }
        const std::shared_ptr<TensorBase> &wk(int layer) const { return inner.wk[layer]; }
        const std::shared_ptr<TensorBase> &wv(int layer) const { return inner.wv[layer]; }
        const std::shared_ptr<TensorBase> &wo(int layer) const { return inner.wo[layer]; }
        const std::shared_ptr<TensorBase> &bq(int layer) const { return inner.bq[layer]; }
        const std::shared_ptr<TensorBase> &bk(int layer) const { return inner.bk[layer]; }
        const std::shared_ptr<TensorBase> &bv(int layer) const { return inner.bv[layer]; }
        const std::shared_ptr<TensorBase> &ffn_norm(int layer) const { return inner.ffn_norm_weight[layer]; }
        const std::shared_ptr<TensorBase> &w_gate(int layer) const { return inner.w_gate[layer]; }
        const std::shared_ptr<TensorBase> &w_up(int layer) const { return inner.w_up[layer]; }
        const std::shared_ptr<TensorBase> &w_down(int layer) const { return inner.w_down[layer]; }
    };

    class BatchQwenPipeline : public PipelineBase, public AbstractPipeline
    {
    public:
        explicit BatchQwenPipeline(const ModelConfig &config);
        explicit BatchQwenPipeline(const ModelConfig &config, const MPIContext &ctx);
        ~BatchQwenPipeline() override;

        // AbstractPipeline overrides (single sequence intentionally unsupported here)
        bool prefill(const std::vector<int> &tokens, const IModelWeights &weights, StageContext &ctx) override;
        bool decode(int next_token, const IModelWeights &weights, StageContext &ctx) override;
        bool logits(std::shared_ptr<TensorBase> &out_logits) override;
        const KVCacheState *kvCacheState() const override;   // snapshot of aggregated state
        bool ensureKVCapacity(int required_tokens) override; // batched capacity mgmt (WIP)
        std::unique_ptr<IModelWeights> loadWeights(const std::string &path) override;

        // True batched interfaces
        bool prefillBatch(const std::vector<std::vector<int>> &token_batches,
                          const IModelWeights &weights,
                          StageContext &ctx,
                          std::shared_ptr<TensorBase> &out_logits) override;
        bool decodeBatch(const std::vector<int> &next_tokens,
                         const IModelWeights &weights,
                         StageContext &ctx,
                         std::shared_ptr<TensorBase> &out_logits) override;

        const ModelConfig &config() const override { return config_; }
        std::string name() const override { return "BatchQwenPipeline"; }
        std::string getOperatorType() const override { return "BatchQwenPipeline"; }
        size_t getExpectedInputCount() const override { return 1; }
        size_t getExpectedOutputCount() const override { return 1; }

        // OperatorBase execute (unused but required)
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override
        {
            LOG_ERROR("BatchQwenPipeline::execute(tensors) not implemented - use prefillBatch/decodeBatch");
            return false;
        }

        // OperatorBase validate
        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const override
        {
            return true; // Basic validation stub
        } // Batch state lifecycle
        void clearState();
        void setMaxBatchSize(size_t max_batch) { max_batch_size_ = max_batch; }

        // Public for testing/debugging
        bool prepareEmbedding(const std::vector<std::vector<int>> &token_batches,
                              const BatchQwenWeights &weights,
                              std::shared_ptr<TensorBase> &embedded); // [B, T, D]

    private:
        // Internal helpers
        bool runBatchedLayers(std::shared_ptr<TensorBase> &hidden,
                              const BatchQwenWeights &weights,
                              bool is_prefill);
        bool projectOutput(std::shared_ptr<TensorBase> &hidden,
                           const BatchQwenWeights &weights,
                           std::shared_ptr<TensorBase> &logits_out); // [B, T_last, V]
        bool appendDecodeTokens(const std::vector<int> &next_tokens,
                                std::shared_ptr<TensorBase> &hidden_step,
                                std::shared_ptr<TensorBase> &logits_out);

        // Model context / configuration
        ModelConfig config_;

        // Logical batch metadata
        size_t current_batch_size_{0};
        size_t max_batch_size_{0};
        size_t max_context_observed_{0};

        // Padding & length tracking
        std::vector<int> sequence_lengths_; // actual lengths per sequence
        size_t padded_length_{0};           // T_max for current batch

        // KV cache (per layer, per sequence)
        std::shared_ptr<BatchedKVCache> kv_cache_; // created on demand
        mutable KVCacheState kv_snapshot_{};       // aggregated view
        bool kv_initialized_{false};

        // Last computed logits (batched view). Shape: [B, vocab]
        std::shared_ptr<TensorBase> last_logits_{};

        // Reusable buffers for batched operations (allocated once, reused across layers)
        std::shared_ptr<TensorBase> attn_norm_buffer_;   // [B, T, D]
        std::shared_ptr<TensorBase> attn_out_buffer_;    // [B, T, D]
        std::shared_ptr<TensorBase> post_attn_buffer_;   // [B, T, D]
        std::shared_ptr<TensorBase> ffn_norm_buffer_;    // [B, T, D]
        std::shared_ptr<TensorBase> gate_buffer_;        // [B, T, d_ff]
        std::shared_ptr<TensorBase> up_buffer_;          // [B, T, d_ff]
        std::shared_ptr<TensorBase> swiglu_buffer_;      // [B, T, d_ff]
        std::shared_ptr<TensorBase> ffn_out_buffer_;     // [B, T, D]
    };

} // namespace llaminar
