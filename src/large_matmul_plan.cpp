/**
 * @file large_matmul_plan.cpp
 * @brief Implementation of COSMA prefill planning.
 * @author David Sanftenberg
 */

#include "large_matmul_plan.h"
#include "utils/debug_env.h"
#include "logger.h"
#include <cstdlib>

namespace llaminar
{
    LargeMatmulPlan plan_attention_prefill(
        int seq_len,
        const ModelConfig &config,
        int world_size,
        int rank)
    {
        LargeMatmulPlan plan;
        plan.seq_len = seq_len;
        plan.d_model = config.getLayerConfig().d_model;
        plan.n_heads = config.getLayerConfig().n_head;
        plan.n_kv_heads = config.getLayerConfig().n_head_kv;
        plan.head_dim = config.getLayerConfig().head_dim;
        plan.world_size = world_size;

        // Default to not using COSMA
        plan.use_cosma = false;

        // Factor 1: Environment override - ADAPTIVE_DISABLE_COSMA or LLAMINAR_COSMA_DISABLE
        const auto &env = debugEnv();
        if (env.adaptive.disable_cosma || env.cosma.disable)
        {
            plan.rationale = "COSMA disabled by environment (ADAPTIVE_DISABLE_COSMA or LLAMINAR_COSMA_DISABLE)";
            if (rank == 0)
            {
                LOG_DEBUG("[PrefillPlan] " << plan.rationale);
            }
            return plan;
        }

        // Factor 2: World size - COSMA requires distributed environment
        if (world_size <= 1)
        {
            plan.rationale = "Single-rank environment (world_size=" + std::to_string(world_size) + ")";
            if (rank == 0)
            {
                LOG_DEBUG("[PrefillPlan] COSMA skipped: " << plan.rationale);
            }
            return plan;
        }

        // Factor 3: Sequence length threshold
        const int threshold = env.cosma.prefill_threshold;
        if (seq_len < threshold)
        {
            plan.rationale = "Sequence length " + std::to_string(seq_len) +
                             " below threshold " + std::to_string(threshold);
            if (rank == 0)
            {
                LOG_DEBUG("[PrefillPlan] COSMA skipped: " << plan.rationale);
            }
            return plan;
        }

        // Factor 4: Estimate memory requirements
        // Q, K, V tensors + attention context output
        // Note: Attention scores are computed incrementally per head, not all at once
        size_t q_size = (size_t)seq_len * plan.total_head_dim() * sizeof(float);
        size_t k_size = (size_t)seq_len * plan.kv_head_dim() * sizeof(float);
        size_t v_size = (size_t)seq_len * plan.kv_head_dim() * sizeof(float);
        size_t context_size = (size_t)seq_len * plan.total_head_dim() * sizeof(float);
        size_t norm_size = (size_t)seq_len * plan.d_model * sizeof(float);

        // Conservative estimate: QKV + context + norm + workspace buffer
        plan.estimated_memory_bytes = q_size + k_size + v_size + context_size + norm_size;
        plan.estimated_memory_bytes = (plan.estimated_memory_bytes * 3) / 2; // Add 50% margin for workspace

        // Check against memory budget
        const size_t max_mb = env.cosma.max_resident_mb;
        const size_t max_bytes = max_mb * 1024ULL * 1024ULL;
        if (plan.estimated_memory_bytes > max_bytes)
        {
            plan.rationale = "Estimated memory " +
                             std::to_string(plan.estimated_memory_bytes / (1024 * 1024)) +
                             " MB exceeds budget " + std::to_string(max_mb) + " MB";
            if (rank == 0)
            {
                LOG_WARN("[PrefillPlan] COSMA skipped: " << plan.rationale);
            }
            return plan;
        }

        // Factor 5: Operation size sanity check
        size_t total_elements = (size_t)seq_len * plan.d_model * plan.total_head_dim();
        const size_t min_elements = 1024 * 1024; // 1M elements minimum
        if (total_elements < min_elements)
        {
            plan.rationale = "Operation too small (" +
                             std::to_string(total_elements) + " elements < " +
                             std::to_string(min_elements) + ")";
            if (rank == 0)
            {
                LOG_DEBUG("[PrefillPlan] COSMA skipped: " << plan.rationale);
            }
            return plan;
        }

        // All checks passed - approve COSMA usage
        plan.use_cosma = true;
        plan.fused_qkv = true; // Enable fused RMSNorm + QKV by default
        plan.rationale = "COSMA approved: seq_len=" + std::to_string(seq_len) +
                         ", world=" + std::to_string(world_size) +
                         ", est_mem=" + std::to_string(plan.estimated_memory_bytes / (1024 * 1024)) + "MB";

        if (rank == 0)
        {
            LOG_INFO("[PrefillPlan] " << plan.rationale);
        }

        return plan;
    }

} // namespace llaminar
