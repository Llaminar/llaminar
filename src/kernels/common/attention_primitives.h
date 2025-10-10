/**
 * @file attention_primitives.h
 * @brief Low-level scalar attention building blocks (layout-compatible with MPIAttentionKernel).
 *
 * Layout Assumptions:
 *  Q, K, V, and output are stored row-major over sequence. Each row packs all heads
 *  contiguously: [head0(d0..d_{d-1}), head1(...), ...]. Thus the stride between two
 *  consecutive heads within a token row is head_dim, and the stride between two token
 *  rows is heads * head_dim.
 *
 * Functions are intentionally side-effect free (except in-place RoPE) and have no MPI
 *  dependencies, enabling direct unit testing without distributed context.
 *
 * Numerical Notes:
 *  - Softmax uses standard max-subtraction for stability.
 *  - RoPE matches current MPIAttentionKernel (n_past added externally if needed).
 *  - All loops are simple scalar reference style prioritizing clarity over speed.
 *
 * These primitives allow refactoring MPIAttentionKernel into orchestrating distribution,
 *  projections, and gather while delegating math to easily testable, deterministic blocks.
 */
#pragma once

#include <vector>
#include <cstddef>

namespace llaminar::attn
{

    void apply_rope(float *q, float *k, int seq_len, int head_dim, int q_heads, int k_heads, int n_past, float freq_base);
    void compute_qk_scores(const float *q, const float *k, float *scores, int seq_len, int head_dim, int heads, bool causal, bool apply_softmax);
    void apply_scores_to_v(const float *scores, const float *v, float *out, int seq_len, int head_dim, int heads);
    void fused_attention(const float *q, const float *k, const float *v, float *out, int seq_len, int head_dim, int heads, bool causal);

    /**
     * @brief Expand KV heads for Grouped Query Attention (GQA)
     *
     * Maps n_kv_heads to n_heads by replicating each KV head group.
     * Parallelized with OpenMP over sequence positions for optimal performance.
     *
     * @param k_compact Input K tensor [seq_len, n_kv_heads * head_dim] row-major
     * @param v_compact Input V tensor [seq_len, n_kv_heads * head_dim] row-major
     * @param k_expanded Output K tensor [seq_len, n_heads * head_dim] row-major
     * @param v_expanded Output V tensor [seq_len, n_heads * head_dim] row-major
     * @param seq_len Sequence length
     * @param head_dim Dimension per head
     * @param n_heads Number of query heads for THIS rank (local count in distributed)
     * @param n_kv_heads Number of key/value heads (GLOBAL count, always)
     * @param head_offset Global head offset for this rank (0 for single-rank)
     * @param total_q_heads Total number of Q heads across ALL ranks (GLOBAL count)
     */
    void expand_kv_for_gqa(
        const float *k_compact,
        const float *v_compact,
        float *k_expanded,
        float *v_expanded,
        int seq_len,
        int head_dim,
        int n_heads,
        int n_kv_heads,
        int head_offset = 0,
        int total_q_heads = -1);

    /**
     * @brief Expand KV for Multi-Head Attention (MHA)
     *
     * Simple parallel copy when n_kv_heads == n_heads (no head replication needed).
     * Parallelized with OpenMP over sequence positions.
     *
     * @param k_compact Input K tensor [seq_len, kv_head_dim] row-major
     * @param v_compact Input V tensor [seq_len, kv_head_dim] row-major
     * @param k_expanded Output K tensor [seq_len, total_head_dim] row-major
     * @param v_expanded Output V tensor [seq_len, total_head_dim] row-major
     * @param seq_len Sequence length
     * @param kv_head_dim Input KV dimension (n_kv_heads * head_dim)
     * @param total_head_dim Output dimension (n_heads * head_dim)
     */
    void expand_kv_for_mha(
        const float *k_compact,
        const float *v_compact,
        float *k_expanded,
        float *v_expanded,
        int seq_len,
        int kv_head_dim,
        int total_head_dim);

    struct RowSoftmaxStats
    {
        float max_row_deviation;
        float max_negative;
        float max_prob;
    };
    RowSoftmaxStats validate_softmax_rows(const float *scores, int seq_len, int heads);

} // namespace llaminar::attn
