#pragma once

#include <vector>
#include <memory>
#include "../tensor.h"

namespace llaminar
{

    /**
     * @brief Multi-head attention kernel for transformer layers
     *
     * Implements scaled dot-product multi-head attention with KV caching:
     * Q = input @ q_weight
     * K = input @ k_weight
     * V = input @ v_weight
     * attention_output = softmax(Q @ K^T / sqrt(head_dim)) @ V
     * output = attention_output @ output_weight
     *
     * Expected inputs:
     * - input: [seq_len, hidden_size] - input tensor
     * - q_weight: [hidden_size, hidden_size] - query projection weight
     * - k_weight: [hidden_size, hidden_size] - key projection weight
     * - v_weight: [hidden_size, hidden_size] - value projection weight
     * - output_weight: [hidden_size, hidden_size] - output projection weight
     * - k_cache: [max_seq_len, n_head_kv, head_dim] - key cache tensor
     * - v_cache: [max_seq_len, n_head_kv, head_dim] - value cache tensor
     *
     * Expected outputs:
     * - output: [seq_len, hidden_size] - attention output tensor
     */
    class AttentionKernel
    {
    public:
        AttentionKernel();

        /**
         * @brief Execute multi-head attention
         * @param inputs Vector containing input and weight tensors, plus KV cache
         * @param outputs Vector containing output tensor
         * @return true if execution succeeded, false otherwise
         */
        bool execute(const std::vector<std::shared_ptr<llaminar::Tensor>> &inputs,
                     std::vector<std::shared_ptr<llaminar::Tensor>> &outputs);

        /**
         * @brief Validate input and output tensor shapes and types
         * @param inputs Input tensors to validate
         * @param outputs Output tensors to validate
         * @return true if tensors are valid, false otherwise
         */
        bool validate(const std::vector<std::shared_ptr<llaminar::Tensor>> &inputs,
                      const std::vector<std::shared_ptr<llaminar::Tensor>> &outputs) const;

        // Configuration
        void setHeadDimensions(int n_head, int n_head_kv, int head_dim);
        void setSequencePosition(int n_past) { n_past_ = n_past; }

    private:
        /**
         * @brief Compute Q, K, V projections
         * @param input Input data pointer
         * @param q_weight Q weight matrix pointer
         * @param k_weight K weight matrix pointer
         * @param v_weight V weight matrix pointer
         * @param q_bias Q bias vector pointer (can be nullptr)
         * @param k_bias K bias vector pointer (can be nullptr)
         * @param v_bias V bias vector pointer (can be nullptr)
         * @param q_proj Output Q projection pointer
         * @param k_proj Output K projection pointer
         * @param v_proj Output V projection pointer
         * @param seq_len Sequence length
         * @param hidden_size Hidden dimension size
         */
        void computeQKVProjections(const float *input,
                                   const float *q_weight, const float *k_weight, const float *v_weight,
                                   const float *q_bias, const float *k_bias, const float *v_bias,
                                   float *q_proj, float *k_proj, float *v_proj,
                                   int seq_len, int hidden_size);

        /**
         * @brief Compute scaled dot-product attention
         * @param q Query tensor pointer
         * @param k Key cache pointer
         * @param v Value cache pointer
         * @param attn_weights Attention weights output pointer
         * @param output Attention output pointer
         * @param seq_len Current sequence length
         * @param n_head Number of attention heads
         * @param head_dim Head dimension size
         */
        void computeScaledDotProductAttention(const float *q, const float *k, const float *v,
                                              float *attn_weights, float *output,
                                              int seq_len, int n_head, int head_dim);

        /**
         * @brief Update KV cache with new key and value tensors
         * @param k_new New key tensor pointer
         * @param v_new New value tensor pointer
         * @param k_cache Key cache pointer
         * @param v_cache Value cache pointer
         * @param seq_len Current sequence length
         * @param n_head_kv Number of key-value heads
         * @param head_dim Head dimension size
         * @param n_past Number of past tokens in cache
         */
        void updateKVCache(const float *k_new, const float *v_new,
                           float *k_cache, float *v_cache,
                           int seq_len, int n_head_kv, int head_dim, int n_past);

        int n_head_;    ///< Number of attention heads
        int n_head_kv_; ///< Number of key-value heads (for GQA)
        int head_dim_;  ///< Dimension per head
        int n_past_;    ///< Number of past tokens in KV cache
    };

} // namespace llaminar