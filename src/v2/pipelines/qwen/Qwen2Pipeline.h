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
        /**
         * @brief Construct Qwen2 pipeline
         *
         * @param model_path Path to GGUF model file
         * @param mpi_ctx MPI context for distributed execution (nullptr = single node)
         * @param device_idx Default device for tensors (-1 = CPU, ≥0 = GPU device)
         */
        Qwen2Pipeline(const std::string &model_path,
                      std::shared_ptr<MPIContext> mpi_ctx = nullptr,
                      int device_idx = -1);

        ~Qwen2Pipeline() override = default;

        // PipelineBase interface
        bool forward(const int *tokens, int seq_len) override;
        const float *logits() const override;
        const char *architecture() const override { return "qwen2"; }

        /**
         * @brief Get specific layer weight for device placement
         *
         * @param layer_idx Layer index (0-indexed)
         * @param weight_name Weight name ("wq", "wk", "wv", "wo", "gate", "up", "down")
         * @return Tensor pointer, or nullptr if not found
         */
        std::shared_ptr<TensorBase> get_layer_weight(int layer_idx, const std::string &weight_name);

    protected:
        // PipelineBase interface
        bool load_weights(const std::string &model_path) override;
        bool transformer_layer(int layer_idx, int seq_len) override;

    private:
        // Qwen2-specific architecture parameters
        int n_heads_ = 0;
        int n_kv_heads_ = 0;
        int head_dim_ = 0;
        int d_ff_ = 0;

        // Layer weights structure
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

        // Weights (quantized, stay on host for CPU, uploaded to GPU for GPU backends)
        std::shared_ptr<TensorBase> embedding_table_; // [vocab_size, d_model] FP32
        std::vector<LayerWeights> layers_;            // Per-layer weights
        std::shared_ptr<TensorBase> final_norm_;      // Final RMSNorm gamma [d_model]
        std::shared_ptr<TensorBase> lm_head_;         // [d_model, vocab_size] IQ4_NL

        // Activations (FP32, on host or device depending on device_idx)
        std::shared_ptr<FP32Tensor> current_hidden_; // [seq_len, d_model]
        std::shared_ptr<FP32Tensor> logits_;         // [seq_len, vocab_size]

        // Helper methods
        bool attention_block(const LayerWeights &layer, int seq_len);
        bool ffn_block(const LayerWeights &layer, int seq_len);
    };

} // namespace llaminar2
