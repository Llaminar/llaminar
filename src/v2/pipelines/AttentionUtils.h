/**
 * @file AttentionUtils.h
 * @brief Utilities for attention computation across different architectures
 *
 * Provides helper functions for GQA (Grouped Query Attention), MHA, and MQA:
 * - K/V head broadcasting for GQA
 * - Causal and sliding window mask generation
 * - Head reshaping utilities
 *
 * Used by PipelineBase::attention_gqa() default orchestration.
 *
 * @author David Sanftenberg
 */

#pragma once

#include <cstddef>
#include <cmath>
#include <cstring>
#include <limits>

namespace llaminar2
{
    namespace attention_utils
    {

        /**
         * @brief Broadcast K/V heads to match Q heads for GQA
         *
         * In GQA (Grouped Query Attention), each KV head serves multiple Q heads:
         *   - n_heads > n_kv_heads (e.g., 14 Q heads, 2 KV heads)
         *   - Each KV head is replicated to (n_heads / n_kv_heads) Q heads
         *
         * Special cases:
         *   - MHA: n_heads == n_kv_heads (no broadcasting, just copy)
         *   - MQA: n_kv_heads == 1 (broadcast single KV head to all Q heads)
         *
         * @param kv_in Input K or V tensor [seq_len, n_kv_heads, head_dim] (flattened)
         * @param kv_out Output K or V tensor [seq_len, n_heads, head_dim] (flattened)
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads
         * @param head_dim Dimension per head
         */
        inline void broadcast_kv_heads(
            const float *kv_in, float *kv_out,
            int seq_len, int n_heads, int n_kv_heads, int head_dim)
        {
            const int heads_per_kv = n_heads / n_kv_heads;

            // Parallelize over sequences (outer loop)
#pragma omp parallel for if (seq_len * n_kv_heads > 512)
            for (int s = 0; s < seq_len; ++s)
            {
                for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h)
                {
                    // Source: kv_in[s, kv_h, :] (one KV head)
                    const float *src = kv_in + s * n_kv_heads * head_dim + kv_h * head_dim;

                    // Broadcast to multiple Q heads: kv_out[s, kv_h*heads_per_kv + i, :]
                    for (int i = 0; i < heads_per_kv; ++i)
                    {
                        const int q_h = kv_h * heads_per_kv + i;

                        // Safety: guard against any mismatch where
                        // n_heads is not an exact multiple of n_kv_heads.
                        if (q_h >= n_heads)
                        {
                            continue;
                        }

                        float *dst = kv_out + s * n_heads * head_dim + q_h * head_dim;
                        std::memcpy(dst, src, head_dim * sizeof(float));
                    }
                }
            }
        }

        /**
         * @brief Create causal attention mask (lower triangular)
         *
         * For autoregressive generation: token i can only attend to tokens [0..i]
         *
         * mask[i][j] = 0.0    if i >= j (can attend)
         *            = -inf   if i < j  (cannot attend to future)
         *
         * Sliding window variant (window_size > 0):
         * mask[i][j] = 0.0    if i >= j AND i - j < window_size
         *            = -inf   otherwise
         *
         * @param mask Output mask [seq_len, seq_len]
         * @param seq_len Sequence length
         * @param window_size Sliding window size (-1 = full attention, 0+ = local window)
         */
        inline void create_causal_mask(
            float *mask, int seq_len, int window_size = -1)
        {
            const float neg_inf = -std::numeric_limits<float>::infinity();

// Parallelize over rows (each row independent)
#pragma omp parallel for if (seq_len * seq_len > 1024)
            for (int i = 0; i < seq_len; ++i)
            {
                for (int j = 0; j < seq_len; ++j)
                {
                    bool can_attend = (i >= j); // Causal: can't attend to future

                    // Sliding window: also check distance
                    // NOTE: window_size=0 treated same as -1 (no windowing)
                    if (window_size > 0 && can_attend)
                    {
                        can_attend = (i - j < window_size);
                    }

                    mask[i * seq_len + j] = can_attend ? 0.0f : neg_inf;
                }
            }
        }

        /**
         * @brief Create batch-aware causal attention mask with sequence boundaries
         *
         * For batched inputs: [batch_size, padded_seq_len, ...]
         * - Causal masking within each sequence (token i can only attend to tokens [0..i] in same sequence)
         * - Full masking across sequence boundaries (sequences cannot see each other)
         * - Optional padding masking for variable-length sequences
         *
         * Mask structure for batch_size=2, padded_seq_len=2:
         * ```
         *      s0t0  s0t1  s1t0  s1t1
         * s0t0   0    -∞    -∞    -∞     (seq0, tok0 can only see itself)
         * s0t1   0     0    -∞    -∞     (seq0, tok1 can see seq0's tok0-1)
         * s1t0  -∞    -∞     0    -∞     (seq1, tok0 can only see itself - isolated from seq0!)
         * s1t1  -∞    -∞     0     0     (seq1, tok1 can see seq1's tok0-1)
         * ```
         *
         * @param mask Output mask [total_len, total_len] where total_len = batch_size * padded_seq_len
         * @param batch_size Number of sequences in batch
         * @param padded_seq_len Maximum sequence length (after padding)
         * @param sequence_lengths Actual length per sequence (nullptr = all sequences use padded_seq_len)
         * @param window_size Sliding window size (-1 = full attention, 0+ = local window)
         */
        inline void create_batch_causal_mask(
            float *mask,
            int batch_size,
            int padded_seq_len,
            const int *sequence_lengths = nullptr,
            int window_size = -1)
        {
            const float neg_inf = -std::numeric_limits<float>::infinity();
            const int total_len = batch_size * padded_seq_len;

// Parallelize over rows - each row's mask is independent
// Use flattened 1D iteration for better load balancing
#pragma omp parallel for if (total_len * total_len > 4096) schedule(static)
            for (int i = 0; i < total_len; ++i)
            {
                const int batch_i = i / padded_seq_len; // Which sequence does token i belong to?
                const int pos_i = i % padded_seq_len;   // Position within that sequence

                // Get actual sequence length for this sequence
                const int actual_len_i = sequence_lengths ? sequence_lengths[batch_i] : padded_seq_len;

                // If token i is a padding token, it cannot attend to anything
                const bool i_is_padding = (pos_i >= actual_len_i);

                for (int j = 0; j < total_len; ++j)
                {
                    const int batch_j = j / padded_seq_len;
                    const int pos_j = j % padded_seq_len;

                    bool can_attend = false;

                    // Padding tokens cannot attend to anything
                    if (!i_is_padding)
                    {
                        // 1. Must be in same sequence (block-diagonal structure)
                        if (batch_i == batch_j)
                        {
                            // 2. Check if token j is within valid sequence (not padding)
                            const int actual_len_j = sequence_lengths ? sequence_lengths[batch_j] : padded_seq_len;
                            const bool j_is_padding = (pos_j >= actual_len_j);

                            if (!j_is_padding)
                            {
                                // 3. Causal constraint: can't attend to future tokens
                                if (pos_i >= pos_j)
                                {
                                    // 4. Sliding window constraint (if enabled)
                                    // NOTE: window_size=0 treated same as -1 (no windowing)
                                    if (window_size <= 0 || (pos_i - pos_j < window_size))
                                    {
                                        can_attend = true;
                                    }
                                }
                            }
                        }
                    }

                    mask[i * total_len + j] = can_attend ? 0.0f : neg_inf;
                }
            }
        }

        /**
         * @brief Create batch padding mask (non-causal, padding-only)
         *
         * Creates a combined mask for batched attention that ONLY masks padding tokens.
         * No causal masking applied - allows attending to all valid tokens in sequence.
         *
         * Layout: Combined mask [total_tokens, total_tokens] where total_tokens = batch_size * padded_seq_len
         * - Block-diagonal structure: Each sequence can only attend within itself
         * - Padding masking: Padding tokens (pos >= sequence_length) are masked with -inf
         * - No causal constraint: Can attend to past AND future tokens (bi-directional)
         *
         * @param mask Output mask buffer [total_tokens, total_tokens]
         * @param batch_size Number of sequences in batch
         * @param padded_seq_len Padded sequence length (uniform for all sequences)
         * @param sequence_lengths Actual lengths of each sequence (nullptr = all sequences are full length)
         * @param window_size Sliding window size (-1 = disabled, >0 = limit context window)
         */
        inline void create_batch_padding_mask(
            float *mask,
            int batch_size,
            int padded_seq_len,
            const int *sequence_lengths = nullptr,
            int window_size = -1)
        {
            const float neg_inf = -std::numeric_limits<float>::infinity();
            const int total_len = batch_size * padded_seq_len;

// Parallelize over rows - each row's mask is independent
// Use flattened 1D iteration for better load balancing
#pragma omp parallel for if (total_len * total_len > 4096) schedule(static)
            for (int i = 0; i < total_len; ++i)
            {
                const int batch_i = i / padded_seq_len; // Which sequence does token i belong to?
                const int pos_i = i % padded_seq_len;   // Position within that sequence

                // Get actual sequence length for this sequence
                const int actual_len_i = sequence_lengths ? sequence_lengths[batch_i] : padded_seq_len;

                // If token i is a padding token, it cannot attend to anything
                const bool i_is_padding = (pos_i >= actual_len_i);

                for (int j = 0; j < total_len; ++j)
                {
                    const int batch_j = j / padded_seq_len;
                    const int pos_j = j % padded_seq_len;

                    bool can_attend = false;

                    // Padding tokens cannot attend to anything
                    if (!i_is_padding)
                    {
                        // 1. Must be in same sequence (block-diagonal structure)
                        if (batch_i == batch_j)
                        {
                            // 2. Check if token j is within valid sequence (not padding)
                            const int actual_len_j = sequence_lengths ? sequence_lengths[batch_j] : padded_seq_len;
                            const bool j_is_padding = (pos_j >= actual_len_j);

                            if (!j_is_padding)
                            {
                                // 3. NO CAUSAL CONSTRAINT - can attend to any valid token (past or future)
                                // 4. Sliding window constraint (if enabled)
                                // NOTE: window_size=0 treated same as -1 (no windowing)
                                if (window_size <= 0 || (std::abs(pos_i - pos_j) < window_size))
                                {
                                    can_attend = true;
                                }
                            }
                        }
                    }

                    mask[i * total_len + j] = can_attend ? 0.0f : neg_inf;
                }
            }
        }

        /**
         * @brief Apply attention mask to scores (adds mask to scores)
         *
         * scores[i][j] += mask[i][j]
         *
         * This adds -inf to masked positions, which become 0 after softmax.
         *
         * @param scores Attention scores [rows, cols] (modified in-place)
         * @param mask Attention mask [rows, cols]
         * @param rows Number of rows (typically seq_len)
         * @param cols Number of columns (typically seq_len)
         */
        inline void apply_attention_mask(
            float *scores, const float *mask,
            int rows, int cols)
        {
            const int total = rows * cols;

// Parallelize mask application (vectorizable)
#pragma omp parallel for if (total > 4096)
            for (int i = 0; i < total; ++i)
            {
                scores[i] += mask[i];
            }
        }

        /**
         * @brief Reshape flat tensor to head dimensions
         *
         * Reshapes [seq_len, n_heads * head_dim] to [seq_len, n_heads, head_dim]
         * This is a no-op in terms of memory layout (just dimensional interpretation),
         * but helps clarify the head structure for attention computation.
         *
         * Note: This is conceptual - actual memory layout stays the same.
         * Use for documentation/validation purposes.
         *
         * @param total_size Expected total size (seq_len * n_heads * head_dim)
         * @param seq_len Sequence length
         * @param n_heads Number of heads
         * @param head_dim Dimension per head
         * @return true if dimensions are consistent
         */
        inline bool validate_head_reshape(
            size_t total_size, int seq_len, int n_heads, int head_dim)
        {
            return (total_size == static_cast<size_t>(seq_len * n_heads * head_dim));
        }

        /**
         * @brief Scale attention scores by 1/sqrt(head_dim)
         *
         * Applies the standard attention scaling factor to prevent softmax saturation.
         *
         * scores[i] *= 1.0 / sqrt(head_dim)
         *
         * @param scores Attention scores (modified in-place)
         * @param count Number of elements
         * @param head_dim Dimension per head
         */
        inline void scale_attention_scores(
            float *scores, int count, int head_dim)
        {
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            for (int i = 0; i < count; ++i)
            {
                scores[i] *= scale;
            }
        }

        /**
         * @brief Create combined causal + padding mask for batched attention
         *
         * Combines multiple masking constraints:
         * 1. Block-diagonal: Tokens from different batches never attend to each other
         * 2. Padding masking: Real tokens cannot attend to padding positions
         * 3. Causal masking: Token i cannot attend to tokens j > i (future) [optional]
         * 4. Padding tokens: Cannot attend to anything (all positions masked)
         *
         * Result: mask[total_len, total_len] where total_len = batch_size * seq_len
         * - mask[i, j] = 0.0    if token i can attend to token j
         * - mask[i, j] = -inf   otherwise
         *
         * Masking rules:
         * - Cross-batch masking: Always masked (different batches never attend)
         * - Padding masking: Masked if j >= actual_lengths[batch_j] (padding position)
         * - Causal masking: Masked if j > i within same batch (cannot attend to future)
         * - Padding tokens: Cannot attend to anything (i >= actual_lengths[batch_i])
         *
         * @param mask Output mask [batch_size * seq_len, batch_size * seq_len] (flattened 2D)
         * @param batch_size Number of sequences in batch
         * @param seq_len Maximum sequence length
         * @param actual_lengths Actual length of each sequence [batch_size]
         * @param causal Apply causal masking (default: true)
         * @param window_size Sliding window size (-1 = full attention)
         */
        inline void create_combined_batch_mask(
            float *mask,
            int batch_size,
            int seq_len,
            const int *actual_lengths,
            bool causal = true,
            int window_size = -1)
        {
            const float neg_inf = -std::numeric_limits<float>::infinity();
            const int total_len = batch_size * seq_len;

// Parallelize over rows - each row's mask is independent
#pragma omp parallel for if (total_len * total_len > 4096) schedule(static)
            for (int i = 0; i < total_len; ++i)
            {
                const int batch_i = i / seq_len; // Which batch does token i belong to?
                const int pos_i = i % seq_len;   // Position within that batch

                // Get actual sequence length for this batch
                const int actual_len_i = actual_lengths[batch_i];

                // If token i is a padding token, it cannot attend to anything
                const bool i_is_padding = (pos_i >= actual_len_i);

                for (int j = 0; j < total_len; ++j)
                {
                    const int batch_j = j / seq_len; // Which batch does token j belong to?
                    const int pos_j = j % seq_len;   // Position within that batch

                    bool can_attend = false;

                    // Padding tokens cannot attend to anything
                    if (!i_is_padding)
                    {
                        // 1. Must be in same batch (block-diagonal structure)
                        if (batch_i == batch_j)
                        {
                            // 2. Check if token j is within valid sequence (not padding)
                            const int actual_len_j = actual_lengths[batch_j];
                            const bool j_is_padding = (pos_j >= actual_len_j);

                            if (!j_is_padding)
                            {
                                // 3. Causal constraint: can't attend to future tokens
                                if (!causal || pos_j <= pos_i)
                                {
                                    // 4. Sliding window constraint (if enabled)
                                    // NOTE: window_size=0 treated same as -1 (no windowing)
                                    if (window_size <= 0 || (pos_i - pos_j < window_size))
                                    {
                                        can_attend = true;
                                    }
                                }
                            }
                        }
                    }

                    mask[i * total_len + j] = can_attend ? 0.0f : neg_inf;
                }
            }
        }

    } // namespace attention_utils
} // namespace llaminar2
