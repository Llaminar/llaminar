/**
 * @file QwenPipeline.h
 * @brief Qwen transformer pipeline with direct kernel orchestration
 *
 * Clean greenfield implementation:
 * - No operator layer (direct kernel calls)
 * - No slab cache (streaming dequant in kernels)
 * - Per-tensor device placement
 * - Selective BF16 (bandwidth-bound ops only)
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../utils/MPIContext.h"
#include "../backends/ComputeBackend.h"
#include "../tensors/Tensors.h"
#include "../tensors/TensorKernels.h"
#include <vector>
#include <memory>
#include <string>

namespace llaminar2
{

    /**
     * @brief Qwen transformer pipeline
     */
    class QwenPipeline
    {
    public:
        /**
         * @brief Construct Qwen pipeline
         *
         * @param model_path Path to GGUF model file
         * @param mpi_ctx MPI context for distributed execution (nullptr = single node)
         * @param device_idx Default device for tensors (-1 = CPU, ≥0 = GPU device)
         */
        QwenPipeline(const std::string &model_path,
                     std::shared_ptr<MPIContext> mpi_ctx = nullptr,
                     int device_idx = -1);

        /**
         * @brief Forward pass (prefill or decode)
         *
         * @param tokens Token IDs [seq_len]
         * @param seq_len Number of tokens
         * @return true on success, false on error
         */
        bool forward(const int *tokens, int seq_len);

        /**
         * @brief Get output logits (FP32)
         *
         * @return Logits tensor [seq_len, vocab_size]
         */
        const float *logits() const;

        /**
         * @brief Get specific layer weight for device placement
         *
         * @param layer_idx Layer index (0-indexed)
         * @param weight_name Weight name ("wq", "wk", "wv", "wo", "gate", "up", "down")
         * @return Tensor pointer, or nullptr if not found
         */
        std::shared_ptr<TensorBase> get_layer_weight(int layer_idx, const std::string &weight_name);

    private:
        // Context management
        std::shared_ptr<MPIContext> mpi_ctx_;
        int device_idx_; // Default device

        // Model architecture (Qwen 2.5 0.5B)
        int n_layers_ = 24;
        int n_heads_ = 14;
        int n_kv_heads_ = 2;
        int head_dim_ = 64;
        int d_model_ = 896;
        int d_ff_ = 4864;
        int vocab_size_ = 151936;

        // Layer weights structure
        struct LayerWeights
        {
            std::shared_ptr<TensorBase> wq;        // Query projection
            std::shared_ptr<TensorBase> wk;        // Key projection
            std::shared_ptr<TensorBase> wv;        // Value projection
            std::shared_ptr<TensorBase> wo;        // Output projection
            std::shared_ptr<TensorBase> attn_norm; // Pre-attention norm gamma
            std::shared_ptr<TensorBase> gate_proj; // FFN gate projection
            std::shared_ptr<TensorBase> up_proj;   // FFN up projection
            std::shared_ptr<TensorBase> down_proj; // FFN down projection
            std::shared_ptr<TensorBase> ffn_norm;  // Pre-FFN norm gamma
        };

        // Weights (quantized, stay on host for CPU, uploaded to GPU for GPU backends)
        std::shared_ptr<TensorBase> embedding_table_; // [vocab_size, d_model] FP32
        std::vector<LayerWeights> layers_;            // Per-layer weights
        std::shared_ptr<TensorBase> final_norm_;      // Final RMSNorm gamma
        std::shared_ptr<TensorBase> lm_head_;         // [vocab_size, d_model] IQ4_NL

        // Activations (FP32, on host or device depending on device_idx)
        std::shared_ptr<FP32Tensor> current_hidden_; // [seq_len, d_model]
        std::shared_ptr<FP32Tensor> logits_;         // [seq_len, vocab_size]

        // Helper methods
        bool load_weights(const std::string &model_path);
        bool transformer_layer(int layer_idx, int seq_len);
        bool attention_block(const LayerWeights &layer, int seq_len);
        bool ffn_block(const LayerWeights &layer, int seq_len);
    };

} // namespace llaminar2
