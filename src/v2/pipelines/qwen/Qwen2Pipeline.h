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
#include "../../tensors/KVCache.h"

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
        };

        /**
         * @brief Construct Qwen 2.x pipeline
         *
         * @param model_ctx Model context with GGUF metadata and weights
         * @param mpi_ctx MPI context (nullptr = single-rank)
         * @param device_idx Default device (-1 = CPU, ≥0 = GPU)
         * @param placement_map Weight placement strategy (nullptr = single device)
         * @param config Runtime configuration (max_seq_len, threading, etc.)
         */
        Qwen2Pipeline(std::shared_ptr<ModelContext> model_ctx,
                      std::shared_ptr<MPIContext> mpi_ctx,
                      int device_idx,
                      std::shared_ptr<WeightPlacementMap> placement_map,
                      const PipelineConfig &config = PipelineConfig{});
        ~Qwen2Pipeline() override = default;

        // PipelineBase interface
        bool forward(const int *tokens, int seq_len) override;
        const char *architecture() const override { return "qwen2"; }

        /**
         * @brief Get output logits for E2E testing/validation
         *
         * @return Logits tensor [seq_len, vocab_size], or nullptr if forward() not called
         */
        const float *getLogits() const { return logits(); }

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

    private:
        // Qwen2-specific architecture parameters
        int d_ff_ = 0;

        // Weights (quantized, stay on host for CPU, uploaded to GPU for GPU backends)
        std::shared_ptr<TensorBase> embedding_table_; // [vocab_size, d_model] FP32
        std::vector<LayerWeights> layers_;            // Per-layer weights
        std::shared_ptr<TensorBase> final_norm_;      // Final RMSNorm gamma [d_model]
        std::shared_ptr<TensorBase> lm_head_;         // [d_model, vocab_size] IQ4_NL

        // Activations (FP32, on host or device depending on device_idx)
        std::shared_ptr<FP32Tensor> current_hidden_; // [seq_len, d_model]

        // Helper methods for dimension specifications (Qwen2-specific)
        TensorSpec spec_hidden(int seq_len) const
        {
            return TensorSpec({static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)},
                              "hidden[" + std::to_string(seq_len) + "," + std::to_string(d_model_) + "]");
        }

        TensorSpec spec_q(int seq_len) const
        {
            return TensorSpec({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads_ * head_dim_)},
                              "Q[" + std::to_string(seq_len) + "," + std::to_string(n_heads_ * head_dim_) + "]");
        }

        TensorSpec spec_kv(int seq_len) const
        {
            return TensorSpec({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads_ * head_dim_)},
                              "KV[" + std::to_string(seq_len) + "," + std::to_string(n_kv_heads_ * head_dim_) + "]");
        }

        TensorSpec spec_ffn_intermediate(int seq_len) const
        {
            return TensorSpec({static_cast<size_t>(seq_len), static_cast<size_t>(d_ff_)},
                              "ffn_intermediate[" + std::to_string(seq_len) + "," + std::to_string(d_ff_) + "]");
        }

        TensorSpec spec_ffn_gate_up(int seq_len) const
        {
            return TensorSpec({static_cast<size_t>(seq_len), static_cast<size_t>(d_ff_)},
                              "ffn_gate_up[" + std::to_string(seq_len) + "," + std::to_string(d_ff_) + "]");
        }

        TensorSpec spec_logits(int seq_len) const
        {
            return TensorSpec({static_cast<size_t>(seq_len), static_cast<size_t>(vocab_size_)},
                              "logits[" + std::to_string(seq_len) + "," + std::to_string(vocab_size_) + "]");
        }

        TensorSpec spec_norm_gamma() const
        {
            return TensorSpec({static_cast<size_t>(d_model_)},
                              "norm_gamma[" + std::to_string(d_model_) + "]");
        }

        // Helper methods
        bool attention_block(const LayerWeights &layer, int layer_idx, int seq_len);
        bool ffn_block(const LayerWeights &layer, int seq_len);

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
            current_position_ = 0;
        }

        /**
         * @brief Get current cache position
         */
        int get_position() const { return current_position_; }
    };

} // namespace llaminar2
