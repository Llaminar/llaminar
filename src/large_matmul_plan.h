/**
 * @file large_matmul_plan.h
 * @brief Encapsulates COSMA prefill planning and validation.
 * @author David Sanftenberg
 *
 * This module moves COSMA environment checks and configuration into a validated
 * plan object, simplifying the executePrefillAttentionCosma signature and
 * centralizing COSMA prefill decision logic.
 */

#pragma once

#include "transformer_config.h"
#include "utils/debug_env.h"
#include "logger.h"
#include <string>
#include <cstddef>

namespace llaminar
{
    /**
     * @brief Plan for large-scale matrix multiplication using COSMA.
     *
     * Encapsulates the decision to use COSMA, memory requirements, and
     * configuration parameters. Created via plan_attention_prefill() which
     * performs all environment checks and validation.
     */
    struct LargeMatmulPlan
    {
        /// Whether to use COSMA for this operation
        bool use_cosma{false};

        /// Whether to use fused RMSNorm + QKV projection
        bool fused_qkv{true};

        /// Estimated memory requirement in bytes (for validation)
        size_t estimated_memory_bytes{0};

        /// Sequence length for this operation
        int seq_len{0};

        /// Model dimension (hidden size)
        int d_model{0};

        /// Number of attention heads
        int n_heads{0};

        /// Number of KV heads (for GQA)
        int n_kv_heads{0};

        /// Head dimension
        int head_dim{0};

        /// MPI world size
        int world_size{1};

        /// Human-readable rationale for the decision
        std::string rationale;

        /**
         * @brief Check if plan is valid and approved for execution.
         */
        bool is_valid() const { return use_cosma; }

        /**
         * @brief Get total head dimension (n_heads * head_dim).
         */
        int total_head_dim() const { return n_heads * head_dim; }

        /**
         * @brief Get KV head dimension (n_kv_heads * head_dim).
         */
        int kv_head_dim() const { return n_kv_heads * head_dim; }
    };

    /**
     * @brief Plan COSMA prefill for attention operation.
     *
     * Performs all environment checks and validation to determine whether
     * to use COSMA for the given prefill operation. Consolidates logic
     * previously scattered across shouldUseCosmaPrefill, executePrefillAttentionCosma,
     * and various environment variable checks.
     *
     * @param seq_len Sequence length
     * @param config Model configuration
     * @param world_size MPI world size
     * @param rank MPI rank (for logging)
     *
     * @return Validated plan with use_cosma flag and configuration
     *
     * Decision factors:
     * 1. Environment overrides (ADAPTIVE_DISABLE_COSMA forces false)
     * 2. Sequence length threshold (LLAMINAR_COSMA_PREFILL_THRESHOLD, default 4096)
     * 3. World size (requires distributed environment)
     * 4. Memory budget (LLAMINAR_COSMA_MAX_RESIDENT_MB check)
     * 5. Operation size (avoid COSMA for tiny operations)
     */
    LargeMatmulPlan plan_attention_prefill(
        int seq_len,
        const ModelConfig &config,
        int world_size,
        int rank = 0);

} // namespace llaminar
