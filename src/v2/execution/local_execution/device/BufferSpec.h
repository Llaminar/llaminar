/**
 * @file BufferSpec.h
 * @brief Specification for buffer allocation sizes in PP+TP execution
 *
 * Part of the Unified PP Graph Architecture (Phase 1).
 * Captures buffer dimensions without knowing concrete tensor types.
 *
 * @see docs/v2/UNIFIED_PP_GRAPH_ARCHITECTURE_PLAN.md
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "../../config/RuntimeConfig.h"
#include <cstddef>

namespace llaminar2
{

/**
 * @brief Specification for PP stage buffer allocation sizes
 *
 * Captures the shapes needed for each buffer type, allowing
 * allocation without knowing the concrete tensor types.
 *
 * Note: This is distinct from the string-based BufferSpec in GraphSchema.h
 * which is used for declarative graph specification. This struct provides
 * concrete numeric dimensions for actual buffer allocation.
 *
 * ## Usage
 *
 * @code
 * PPStageBufferSpec spec;
 * spec.batch_size = 1;
 * spec.seq_len = 512;
 * spec.d_model = 896;
 * spec.n_heads = 14;
 * spec.n_kv_heads = 2;
 * spec.head_dim = 64;
 * spec.intermediate_size = 4864;
 * spec.vocab_size = 151936;
 * spec.precision = ActivationPrecision::FP32;
 *
 * // Use with PerStageBufferPool
 * PerStageBufferPool pool;
 * pool.initialize(pipeline_config, spec);
 * @endcode
 */
struct PPStageBufferSpec
{
    size_t batch_size = 1;
    size_t seq_len = 512;
    size_t d_model = 896;
    size_t n_heads = 14;
    size_t n_kv_heads = 2;
    size_t head_dim = 64;
    size_t intermediate_size = 4864;
    size_t vocab_size = 151936;

    /// Activation precision (FP32, HybridQ16, etc.)
    ActivationPrecision precision = ActivationPrecision::FP32;

    /// Whether to allocate snapshot buffers
    bool enable_snapshots = false;

    // =========================================================================
    // Derived dimensions
    // =========================================================================

    /// KV dimension (n_kv_heads * head_dim)
    [[nodiscard]] size_t kv_dim() const { return n_kv_heads * head_dim; }

    /// Hidden dimension (alias for d_model)
    [[nodiscard]] size_t hidden_dim() const { return d_model; }

    /// Total tokens in batch (batch_size * seq_len)
    [[nodiscard]] size_t total_tokens() const { return batch_size * seq_len; }

    // =========================================================================
    // Buffer size calculations (element counts)
    // =========================================================================

    /// Residual/normalized buffer: [total_tokens, d_model]
    [[nodiscard]] size_t residual_elements() const { return total_tokens() * d_model; }

    /// Q buffer: [total_tokens, n_heads * head_dim]
    [[nodiscard]] size_t qkv_elements() const { return total_tokens() * n_heads * head_dim; }

    /// K/V buffer: [total_tokens, n_kv_heads * head_dim]
    [[nodiscard]] size_t kv_elements() const { return total_tokens() * kv_dim(); }

    /// FFN intermediate buffer: [total_tokens, intermediate_size]
    [[nodiscard]] size_t ffn_elements() const { return total_tokens() * intermediate_size; }

    /// Logits buffer: [total_tokens, vocab_size]
    [[nodiscard]] size_t logits_elements() const { return total_tokens() * vocab_size; }

    /// Attention scores workspace: [total_tokens, total_tokens] (worst case)
    [[nodiscard]] size_t scores_elements() const { return total_tokens() * seq_len; }
};

} // namespace llaminar2
