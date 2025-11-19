/**
 * @file FP32AttentionKernel.cpp
 * @brief FP32 multi-head attention implementation
 * @author David Sanftenberg
 * @date 2025-11-06
 */

#include "FP32AttentionKernel.h"
#include "../../utils/Logger.h"

#include <cmath>
#include <algorithm>
#include <limits>

namespace llaminar2
{

    FP32AttentionKernel::FP32AttentionKernel(int n_heads, int d_head, int n_kv_heads, int device_idx)
        : n_heads_(n_heads), d_head_(d_head), device_idx_(device_idx), n_kv_heads_(n_kv_heads)
    {
    }

    bool FP32AttentionKernel::forward(
        const float *q,
        const float *k,
        const float *v,
        float *output,
        int batch,
        int seq_len,
        bool use_causal_mask,
        float eps)
    {
        // Validate device
        if (device_idx_ != -1)
        {
            LOG_ERROR("FP32AttentionKernel only supports CPU (device_idx=-1), got " << device_idx_);
            return false;
        }

        // Validate pointers
        if (!q || !k || !v || !output)
        {
            LOG_ERROR("FP32AttentionKernel: null pointer(s) detected");
            return false;
        }

        // Validate dimensions
        if (batch <= 0 || seq_len <= 0 || n_heads_ <= 0 || d_head_ <= 0)
        {
            LOG_ERROR("FP32AttentionKernel: invalid dimensions: batch=" << batch
                                                                        << " seq_len=" << seq_len
                                                                        << " n_heads=" << n_heads_
                                                                        << " d_head=" << d_head_);
            return false;
        }

        // Infer n_kv_heads from K shape (this is a simplification - in practice should be passed)
        // For now, assume n_kv_heads divides n_heads evenly (GQA)
        // We'll detect this from the caller or default to n_kv_heads = n_heads (standard MHA)
        // For this implementation, we'll assume standard MHA unless explicitly set
        if (n_kv_heads_ == 0)
        {
            n_kv_heads_ = n_heads_; // Default: standard multi-head attention
        }

        // Allocate temporary buffers
        size_t scores_size = batch * n_heads_ * seq_len * seq_len;
        size_t kv_expanded_size = batch * seq_len * n_heads_ * d_head_;

        scores_buffer_.resize(scores_size);
        attn_weights_buffer_.resize(scores_size);

        // Expand K/V if needed (GQA)
        const float *k_to_use = k;
        const float *v_to_use = v;

        if (n_kv_heads_ < n_heads_)
        {
            // Allocate expansion buffers
            k_expanded_buffer_.resize(kv_expanded_size);
            v_expanded_buffer_.resize(kv_expanded_size);

            LOG_DEBUG("FP32AttentionKernel: Expanding K/V from " << n_kv_heads_ << " to " << n_heads_ << " heads");
            LOG_DEBUG("  Input K size: " << batch * seq_len * n_kv_heads_ * d_head_ << " elements");
            LOG_DEBUG("  Expanded K size: " << kv_expanded_size << " elements");

            // Expand K/V heads: [batch, seq_len, n_kv_heads*d_head] -> [batch, seq_len, n_heads*d_head]
            expand_kv_heads(k, k_expanded_buffer_.data(), batch, seq_len);
            expand_kv_heads(v, v_expanded_buffer_.data(), batch, seq_len);

            LOG_DEBUG("  K first 5 values (before): " << k[0] << " " << k[1] << " " << k[2] << " " << k[3] << " " << k[4]);
            LOG_DEBUG("  K first 5 values (after): " << k_expanded_buffer_[0] << " " << k_expanded_buffer_[1] << " "
                                                     << k_expanded_buffer_[2] << " " << k_expanded_buffer_[3] << " " << k_expanded_buffer_[4]);

            k_to_use = k_expanded_buffer_.data();
            v_to_use = v_expanded_buffer_.data();
        }
        else
        {
            LOG_DEBUG("FP32AttentionKernel: No K/V expansion needed (n_kv_heads=" << n_kv_heads_ << " == n_heads=" << n_heads_ << ")");
        }

        // Step 1: Compute attention scores Q @ K^T (now standard MHA - K is expanded)
        compute_scores(q, k_to_use, batch, seq_len);

        // Step 2: Apply softmax (with optional causal mask)
        apply_softmax(batch, seq_len, use_causal_mask, eps);

        // Debug: Print first attention weights
        if (batch > 0 && seq_len > 0 && n_heads_ > 0)
        {
            LOG_DEBUG("FP32AttentionKernel: First attention weights[0, 0, 0, :]: "
                      << attn_weights_buffer_[0] << " " << attn_weights_buffer_[1]);
            LOG_DEBUG("FP32AttentionKernel: First attention weights[0, 0, 1, :]: "
                      << attn_weights_buffer_[seq_len] << " " << attn_weights_buffer_[seq_len + 1]);
        }

        // Step 3: Compute output attn_weights @ V (now standard MHA - V is expanded)
        compute_output(v_to_use, output, batch, seq_len);

        return true;
    }

    void FP32AttentionKernel::compute_scores(
        const float *q,
        const float *k,
        int batch,
        int seq_len)
    {
        // For each batch and head, compute: Q[seq_len, d_head] @ K^T[d_head, seq_len]
        // Result: scores[seq_len, seq_len] per head
        //
        // K may be either already expanded to n_heads_ (standard MHA), or still have
        // n_kv_heads_ heads when GQA is used. In forward() we only call compute_scores
        // with an expanded K, but the FP32 parity tests use this kernel directly with
        // full K heads (n_kv_heads_ == n_heads_). To keep the indexing correct and
        // avoid out-of-bounds, we derive d_model_k from the configured KV head count.

        // K is always expanded to n_heads_ before calling this function
        // (either naturally if n_kv_heads == n_heads, or via expand_kv_heads)
        int d_model_q = n_heads_ * d_head_;
        int d_model_k = n_heads_ * d_head_;

        for (int b = 0; b < batch; ++b)
        {
            for (int h = 0; h < n_heads_; ++h)
            {
                // Standard MHA - no GQA mapping (K/V already expanded)
                for (int i = 0; i < seq_len; ++i)
                {
                    for (int j = 0; j < seq_len; ++j)
                    {
                        // Compute dot product: Q[i, h] · K[j, h]
                        float dot = 0.0f;

                        for (int d = 0; d < d_head_; ++d)
                        {
                            // Q index: [b, i, h * d_head + d]
                            int q_idx = (b * seq_len + i) * d_model_q + h * d_head_ + d;
                            // K index: [b, j, h * d_head + d] (same head - K is expanded)
                            int k_idx = (b * seq_len + j) * d_model_k + h * d_head_ + d;

                            dot += q[q_idx] * k[k_idx];
                        }

                        // Store score (before scaling - will scale in softmax)
                        int score_idx = ((b * n_heads_ + h) * seq_len + i) * seq_len + j;
                        scores_buffer_[score_idx] = dot;
                    }
                }
            }
        }
    }

    void FP32AttentionKernel::apply_softmax(
        int batch,
        int seq_len,
        bool use_causal_mask,
        float eps)
    {
        // Softmax is applied per-row: softmax(scores[i, :] / sqrt(d_head)) for each query position i
        float sqrt_d_head = std::sqrt(static_cast<float>(d_head_));

        for (int b = 0; b < batch; ++b)
        {
            for (int h = 0; h < n_heads_; ++h)
            {
                for (int i = 0; i < seq_len; ++i)
                {
                    // Get row base index
                    int row_base = ((b * n_heads_ + h) * seq_len + i) * seq_len;

                    // Step 1: Scale by sqrt(d_head) and find max (for numerical stability)
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int j = 0; j < seq_len; ++j)
                    {
                        if (use_causal_mask && j > i)
                        {
                            continue; // Skip future positions
                        }
                        float score = scores_buffer_[row_base + j] / sqrt_d_head;
                        max_val = std::max(max_val, score);
                    }

                    // Step 2: Compute exp(score - max) and sum
                    float sum_exp = 0.0f;
                    for (int j = 0; j < seq_len; ++j)
                    {
                        float prob;
                        if (use_causal_mask && j > i)
                        {
                            prob = 0.0f; // Masked positions
                        }
                        else
                        {
                            float score = scores_buffer_[row_base + j] / sqrt_d_head;
                            prob = std::exp(score - max_val);
                            sum_exp += prob;
                        }
                        attn_weights_buffer_[row_base + j] = prob;
                    }

                    // Step 3: Normalize to get probabilities
                    sum_exp = std::max(sum_exp, eps); // Avoid division by zero

                    for (int j = 0; j < seq_len; ++j)
                    {
                        attn_weights_buffer_[row_base + j] /= sum_exp;
                    }
                }
            }
        }
    }

    void FP32AttentionKernel::compute_output(
        const float *v,
        float *output,
        int batch,
        int seq_len)
    {
        // Compute: output = attn_weights @ V
        // For each query position, compute weighted sum of values
        //
        // NOTE: V is already expanded to n_heads if GQA was used, so we treat this as standard MHA

        int d_model_v = n_heads_ * d_head_; // V is expanded to match Q
        int d_model_out = n_heads_ * d_head_;

        LOG_DEBUG("compute_output: d_model_v=" << d_model_v << ", d_model_out=" << d_model_out);
        LOG_DEBUG("compute_output: V first 5 values: " << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << " " << v[4]);

        // Initialize output to zero
        std::fill(output, output + batch * seq_len * d_model_out, 0.0f);

        for (int b = 0; b < batch; ++b)
        {
            for (int h = 0; h < n_heads_; ++h)
            {
                // Standard MHA - no GQA mapping (V already expanded)
                for (int i = 0; i < seq_len; ++i)
                {
                    for (int d = 0; d < d_head_; ++d)
                    {
                        float sum = 0.0f;

                        for (int j = 0; j < seq_len; ++j)
                        {
                            // Attention weight: [b, h, i, j]
                            int attn_idx = ((b * n_heads_ + h) * seq_len + i) * seq_len + j;
                            // V index: [b, j, h * d_head + d] (same head - V is expanded)
                            int v_idx = (b * seq_len + j) * d_model_v + h * d_head_ + d;

                            sum += attn_weights_buffer_[attn_idx] * v[v_idx];
                        }

                        // Output index: [b, i, h * d_head + d]
                        int out_idx = (b * seq_len + i) * d_model_out + h * d_head_ + d;
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }

    void FP32AttentionKernel::expand_kv_heads(
        const float *kv_heads,
        float *expanded_heads,
        int batch,
        int seq_len)
    {
        // Expand K/V heads for GQA
        // Each KV head is repeated (n_heads / n_kv_heads) times

        int num_repeats = n_heads_ / n_kv_heads_;

        for (int b = 0; b < batch; ++b)
        {
            for (int i = 0; i < seq_len; ++i)
            {
                for (int kv_h = 0; kv_h < n_kv_heads_; ++kv_h)
                {
                    for (int d = 0; d < d_head_; ++d)
                    {
                        // Source index
                        int src_idx = ((b * seq_len + i) * n_kv_heads_ + kv_h) * d_head_ + d;
                        float val = kv_heads[src_idx];

                        // Repeat this head num_repeats times
                        for (int r = 0; r < num_repeats; ++r)
                        {
                            int q_head = kv_h * num_repeats + r;
                            int dst_idx = ((b * seq_len + i) * n_heads_ + q_head) * d_head_ + d;
                            expanded_heads[dst_idx] = val;
                        }
                    }
                }
            }
        }
    }

} // namespace llaminar2
