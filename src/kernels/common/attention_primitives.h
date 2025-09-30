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

    void apply_rope(float *q, float *k, int seq_len, int head_dim, int heads, int n_past, float freq_base);
    void compute_qk_scores(const float *q, const float *k, float *scores, int seq_len, int head_dim, int heads, bool causal, bool apply_softmax);
    void apply_scores_to_v(const float *scores, const float *v, float *out, int seq_len, int head_dim, int heads);
    void fused_attention(const float *q, const float *k, const float *v, float *out, int seq_len, int head_dim, int heads, bool causal);

    struct RowSoftmaxStats
    {
        float max_row_deviation;
        float max_negative;
        float max_prob;
    };
    RowSoftmaxStats validate_softmax_rows(const float *scores, int seq_len, int heads);

} // namespace llaminar::attn
