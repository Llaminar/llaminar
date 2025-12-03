/**
 * @file Qwen2Pipeline.h
 * @brief Qwen 2.x transformer pipeline with direct kernel orchestration
 *
 * Clean greenfield implementation:
 * - No operator layer (direct kernel calls)
 * - No slab cache (streaming dequant in kernels)
 * - Per-tensor device placement
 * - Selective BF16 (bandwidth-bound ops only)
 *
 * Supports Qwen 2.0 and Qwen 2.5 model families:
 * - 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B parameter sizes
 * - Grouped-query attention (GQA)
 * - SwiGLU activation
 * - RoPE positional embeddings
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../PipelineBase.h"
#include "../TensorDimensions.h"
#include "../ops/Ops.h"
#include "../../tensors/BatchedKVCache.h"
#include "../../kernels/cpu/gemm_v4/FusedGEMM.h"

namespace llaminar2
{
    /**
     * @brief Ensure Qwen2 pipeline is registered with factory
     *
     * This function can be called to force registration if needed.
     * Registration also happens automatically via static constructor.
     */
    void ensureQwen2Registration();

    /**
     * @brief Qwen 2.x transformer pipeline
     *
     * Architecture-specific implementation for Qwen 2.0/2.5 models.
     */
    class Qwen2Pipeline : public PipelineBase
    {
    public:
        // Layer weights structure (public for lazy loading accessors)
        struct LayerWeights
        {
            std::shared_ptr<TensorBase> wq;        // Query projection [d_model, d_model]
            std::shared_ptr<TensorBase> wk;        // Key projection [d_model, n_kv_heads * head_dim]
            std::shared_ptr<TensorBase> wv;        // Value projection [d_model, n_kv_heads * head_dim]
            std::shared_ptr<TensorBase> wo;        // Output projection [d_model, d_model]
            std::shared_ptr<TensorBase> attn_norm; // Pre-attention norm gamma [d_model]
            std::shared_ptr<TensorBase> gate_proj; // FFN gate projection [d_model, d_ff]
            std::shared_ptr<TensorBase> up_proj;   // FFN up projection [d_model, d_ff]
            std::shared_ptr<TensorBase> down_proj; // FFN down projection [d_ff, d_model]
            std::shared_ptr<TensorBase> ffn_norm;  // Pre-FFN norm gamma [d_model]

            // Attention biases (Qwen2 uses Q/K/V biases)
            std::shared_ptr<TensorBase> q_bias; // Query bias [d_model]
            std::shared_ptr<TensorBase> k_bias; // Key bias [n_kv_heads * head_dim]
            std::shared_ptr<TensorBase> v_bias; // Value bias [n_kv_heads * head_dim]

            // Fused GEMM kernels (lazily initialized)
            // These quantize activations once and reuse for multiple projections
            mutable std::unique_ptr<FusedGEMM> qkv_fused;     // Q/K/V fused projection
            mutable std::unique_ptr<FusedGEMM> gate_up_fused; // Gate/Up fused projection
        };

        /**
         * @brief Construct Qwen 2.x pipeline (batch-first design)
         *
         * @param model_ctx Model context with GGUF metadata and weights
         * @param mpi_ctx MPI context (nullptr = single-rank)
         * @param device_idx Default device (-1 = CPU, ≥0 = GPU)
         * @param placement_map Weight placement strategy (nullptr = single device)
         * @param config Runtime configuration (max_seq_len, threading, etc.)
         * @param batch_size Number of sequences to process simultaneously (default=1)
         */
        Qwen2Pipeline(std::shared_ptr<ModelContext> model_ctx,
                      std::shared_ptr<MPIContext> mpi_ctx,
                      int device_idx,
                      std::shared_ptr<WeightPlacementMap> placement_map,
                      const PipelineConfig &config = PipelineConfig{},
                      int batch_size = 1);
        ~Qwen2Pipeline() override = default;

        // PipelineBase interface
        bool forward(const int *tokens, int seq_len) override; // Legacy single-sequence (wraps batch version)
        const char *architecture() const override { return "qwen2"; }
        const float *logits() const override; // Return logits for LAST token of first sequence (for sampling)

        /**
         * @brief Batch-first forward pass (primary interface)
         *
         * @param token_batches Vector of token sequences (batch_size sequences)
         * @return true if forward pass succeeded
         */
        bool forward_batch(const std::vector<std::vector<int>> &token_batches);

        /**
         * @brief Get output logits for E2E testing/validation (full sequence)
         *
         * Returns pointer to logits for the START of the requested sequence's logits.
         * Layout: [padded_seq_len, vocab_size] for seq_idx.
         * Use this for E2E parity tests that compare all positions.
         *
         * @param seq_idx Sequence index in batch (default=0)
         * @return Logits tensor pointer, or nullptr if forward() not called
         */
        const float *getLogits(int seq_idx = 0) const;

        /**
         * @brief Get logits for the last token of a sequence (for sampling)
         *
         * This is what you need for autoregressive generation - the logits
         * predicting the next token given all previous tokens.
         *
         * @param seq_idx Sequence index in batch (default=0)
         * @return Pointer to [vocab_size] logits for last token, or nullptr if forward() not called
         */
        const float *getLastTokenLogits(int seq_idx = 0) const;

        /**
         * @brief Get batch size
         */
        int batch_size() const { return batch_size_; }

        /**
         * @brief Get padded sequence length for current batch
         */
        int padded_seq_len() const { return padded_seq_len_; }

        /**
         * @brief Get sequence lengths for current batch
         */
        const std::vector<int> &sequence_lengths() const { return sequence_lengths_; }

        /**
         * @brief Get specific layer weight for device placement
         *
         * @param layer_idx Layer index (0-indexed)
         * @param weight_name Weight name ("wq", "wk", "wv", "wo", "gate", "up", "down")
         * @return Tensor pointer, or nullptr if not found
         */
        std::shared_ptr<TensorBase> get_layer_weight(int layer_idx, const std::string &weight_name);

        /**
         * @brief Get layer weights (lazy loaded)
         *
         * @param layer_idx Layer index (0-indexed)
         * @return Reference to layer weights structure
         */
        LayerWeights &getLayerWeights(int layer_idx);

        /**
         * @brief Get embedding table (lazy loaded)
         */
        std::shared_ptr<TensorBase> getEmbeddingTable();

        /**
         * @brief Get final norm (lazy loaded)
         */
        std::shared_ptr<TensorBase> getFinalNorm();

        /**
         * @brief Get LM head (lazy loaded)
         */
        std::shared_ptr<TensorBase> getLMHead();

    protected:
        // PipelineBase interface
        bool transformer_layer(int layer_idx, int seq_len) override;

        // Multi-device infrastructure (implements abstract methods from PipelineBase)
        std::vector<std::string> getAllWeightNames() const override;
        ActivationBuffers createBuffersForDevice(int device_idx, int max_seq_len) override;

        /**
         * @brief Initialize infrastructure with batched workspace buffers
         *
         * Calls PipelineBase::initializeDeviceInfrastructure with batch_size
         * to ensure workspace mask has correct dimensions for batched attention.
         */
        void initializeInfrastructureBatched();

    private:
        // Qwen2-specific architecture parameters
        int d_ff_ = 0;                     // Full FFN intermediate size
        int d_ff_local_ = 0;               // Local FFN size per rank (d_ff_ / world_size when column-parallel)
        bool ffn_column_parallel_ = false; // Whether FFN uses column-parallel sharding

        // Batch configuration
        int batch_size_ = 1;                // Number of sequences in batch
        int padded_seq_len_ = 0;            // Max sequence length in current batch
        std::vector<int> sequence_lengths_; // Actual length per sequence [batch_size]

        // Weights (quantized, stay on host for CPU, uploaded to GPU for GPU backends)
        std::shared_ptr<TensorBase> embedding_table_; // [vocab_size, d_model] FP32
        std::vector<LayerWeights> layers_;            // Per-layer weights
        std::shared_ptr<TensorBase> final_norm_;      // Final RMSNorm gamma [d_model]
        std::shared_ptr<TensorBase> lm_head_;         // [d_model, vocab_size] IQ4_NL

        // Activations (precision set by config_.activation_precision, sized for batch_size * max_seq_len)
        std::shared_ptr<TensorBase> current_hidden_; // [batch_size * padded_seq_len, d_model]
        std::shared_ptr<TensorBase> logits_buffer_;  // [batch_size * padded_seq_len, vocab_size]

        // Batched KV cache
        std::shared_ptr<BatchedKVCache> kv_cache_batched_;

        // NOTE: Ops are now in PipelineBase (rmsnorm_op_, gemm_op_, etc.)
        // Child pipelines use the declarative methods: rms_norm(), project(), add_residual(), etc.

        // Helper methods for dimension specifications (batch-aware)
        // All tensors treat first dimension as batch_size * padded_seq_len
        TensorSpec spec_hidden(int effective_seq_len) const
        {
            return TensorSpec({static_cast<size_t>(effective_seq_len), static_cast<size_t>(d_model_)},
                              "hidden[" + std::to_string(effective_seq_len) + "," + std::to_string(d_model_) + "]");
        }

        TensorSpec spec_q(int effective_seq_len) const
        {
            return TensorSpec({static_cast<size_t>(effective_seq_len), static_cast<size_t>(n_heads_ * head_dim_)},
                              "Q[" + std::to_string(effective_seq_len) + "," + std::to_string(n_heads_ * head_dim_) + "]");
        }

        TensorSpec spec_kv(int effective_seq_len) const
        {
            return TensorSpec({static_cast<size_t>(effective_seq_len), static_cast<size_t>(n_kv_heads_ * head_dim_)},
                              "KV[" + std::to_string(effective_seq_len) + "," + std::to_string(n_kv_heads_ * head_dim_) + "]");
        }

        TensorSpec spec_ffn_intermediate(int effective_seq_len) const
        {
            int dim = ffn_column_parallel_ ? d_ff_local_ : d_ff_;
            return TensorSpec({static_cast<size_t>(effective_seq_len), static_cast<size_t>(dim)},
                              "ffn_intermediate[" + std::to_string(effective_seq_len) + "," + std::to_string(dim) + "]");
        }

        TensorSpec spec_ffn_gate_up(int effective_seq_len) const
        {
            int dim = ffn_column_parallel_ ? d_ff_local_ : d_ff_;
            return TensorSpec({static_cast<size_t>(effective_seq_len), static_cast<size_t>(dim)},
                              "ffn_gate_up[" + std::to_string(effective_seq_len) + "," + std::to_string(dim) + "]");
        }

        TensorSpec spec_logits(int effective_seq_len) const
        {
            return TensorSpec({static_cast<size_t>(effective_seq_len), static_cast<size_t>(vocab_size_)},
                              "logits[" + std::to_string(effective_seq_len) + "," + std::to_string(vocab_size_) + "]");
        }

        TensorSpec spec_norm_gamma() const
        {
            return TensorSpec({static_cast<size_t>(d_model_)},
                              "norm_gamma[" + std::to_string(d_model_) + "]");
        }

        // Helper methods (batch-aware)
        bool attention_block(const LayerWeights &layer, int layer_idx, int effective_seq_len);
        bool ffn_block(const LayerWeights &layer, int layer_idx, int effective_seq_len);
        bool embedding_batch(const std::vector<std::vector<int>> &token_batches, TensorBase *output);
        bool lm_head_batch(TensorBase *hidden, int effective_seq_len);

        /**
         * @brief Ensure attention weights are on the target device (lazy GPU transfer)
         *
         * Triggers lazy transfer of Q/K/V/O weights to GPU if placement_map indicates GPU.
         * No-op if weights are already on target device or target is CPU.
         *
         * @param layer Layer weights
         * @param target_device Target device index (-1 = CPU)
         * @return true if all transfers succeeded
         */
        bool ensureAttentionWeightsOnDevice(const LayerWeights &layer, int target_device);

        /**
         * @brief Ensure FFN weights are on the target device (lazy GPU transfer)
         *
         * Triggers lazy transfer of gate/up/down weights to GPU if placement_map indicates GPU.
         * No-op if weights are already on target device or target is CPU.
         *
         * @param layer Layer weights
         * @param target_device Target device index (-1 = CPU)
         * @return true if all transfers succeeded
         */
        bool ensureFFNWeightsOnDevice(const LayerWeights &layer, int target_device);

        // NOTE: Composite operations (rms_norm, project, add_residual, etc.)
        // are now provided by PipelineBase. Child pipelines chain them to form
        // a declarative compute graph.

    public:
        /**
         * @brief Clear KV cache (reset for new sequence)
         */
        void clear_cache()
        {
            if (kv_cache_)
            {
                kv_cache_->clear();
            }
            current_positions_.assign(batch_size_, 0);
        }

        /**
         * @brief Get current cache position (for first sequence)
         */
        int get_position() const { return current_positions_.empty() ? 0 : current_positions_[0]; }
    };

} // namespace llaminar2
