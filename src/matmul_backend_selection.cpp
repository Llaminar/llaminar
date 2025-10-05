/**
 * @file matmul_backend_selection.cpp
 * @brief Implementation of centralized backend selection logic.
 * @author David Sanftenberg
 */

#include "matmul_backend_selection.h"
#include "cosma_prefill_manager.h"
#include <sstream>

namespace llaminar
{

    MatMulBackendDecision MatMulBackendSelector::selectBackend(
        int m, int n, int k,
        const StageContext &ctx,
        const ModelConfig &model_config,
        bool distributed_partition,
        int mpi_rank,
        int mpi_size)
    {
        const size_t total_elements = static_cast<size_t>(m) * n * k;
        const auto &env = debugEnv();

        // === 1. Environment Overrides ===

        // Global COSMA disable
        if (env.adaptive.disable_cosma || env.cosma.disable)
        {
            if (total_elements < TINY_OP_THRESHOLD)
            {
                return MatMulBackendDecision(
                    MatMulBackend::SINGLE_THREADED_OPENBLAS,
                    "COSMA disabled by env + tiny op");
            }
            return MatMulBackendDecision(
                MatMulBackend::MULTI_THREADED_OPENBLAS,
                "COSMA disabled by environment");
        }

        // Force COSMA override (for testing/debugging)
        auto &prefill_mgr = CosmaPrefillManager::instance();
        if (prefill_mgr.force_cosma() && !distributed_partition)
        {
            return MatMulBackendDecision(
                MatMulBackend::COSMA,
                "Forced by LLAMINAR_COSMA_FORCE_DIRECT");
        }

        // === 2. Distributed Partition Safety ===

        // Never use COSMA when caller has already partitioned data across ranks
        // (e.g., MPILinearKernel splits output features). COSMA expects full global
        // matrices and would produce incorrect results on partial data.
        if (distributed_partition)
        {
            if (total_elements < TINY_OP_THRESHOLD)
            {
                return MatMulBackendDecision(
                    MatMulBackend::SINGLE_THREADED_OPENBLAS,
                    "Pre-partitioned data + tiny op");
            }
            return MatMulBackendDecision(
                MatMulBackend::MULTI_THREADED_OPENBLAS,
                "Pre-partitioned across MPI ranks (COSMA unsafe)");
        }

        // === 3. Single-Rank Fallback ===

        if (mpi_size <= 1)
        {
            if (total_elements < TINY_OP_THRESHOLD)
            {
                return MatMulBackendDecision(
                    MatMulBackend::SINGLE_THREADED_OPENBLAS,
                    "Single rank + tiny op");
            }
            if (total_elements < SMALL_OP_THRESHOLD)
            {
                return MatMulBackendDecision(
                    MatMulBackend::MULTI_THREADED_OPENBLAS,
                    "Single rank + medium op");
            }
            return MatMulBackendDecision(
                MatMulBackend::MULTI_THREADED_OPENBLAS,
                "Single rank (no distribution benefit)");
        }

        // === 4. Operation Size Thresholds ===

        // Tiny operations: single-threaded to avoid overhead
        if (total_elements < TINY_OP_THRESHOLD)
        {
            return MatMulBackendDecision(
                MatMulBackend::SINGLE_THREADED_OPENBLAS,
                "Tiny op (< 8K elements)");
        }

        // Avoid COSMA for massive vocabulary projections
        if (n > VOCAB_PROJECTION_THRESHOLD)
        {
            return MatMulBackendDecision(
                MatMulBackend::MULTI_THREADED_OPENBLAS,
                "Huge vocab projection (COSMA inefficient)");
        }

        // === 5. Stage-Specific Heuristics ===

        const bool is_prefill = (ctx.stage == InferenceStage::Prefill);

        if (is_prefill)
        {
            // Prefill: favor COSMA for large operations

            // Minimum sequence length check (policy: COSMA only beneficial >= 4K tokens)
            if (m < static_cast<int>(PREFILL_COSMA_SEQ_THRESHOLD))
            {
                if (total_elements < SMALL_OP_THRESHOLD)
                {
                    return MatMulBackendDecision(
                        MatMulBackend::MULTI_THREADED_OPENBLAS,
                        "Prefill below COSMA seq threshold (< 4K tokens)");
                }
                return MatMulBackendDecision(
                    MatMulBackend::MULTI_THREADED_OPENBLAS,
                    "Prefill moderate size (OpenBLAS competitive)");
            }

            // Check CosmaPrefillManager gating (env + world size)
            if (prefill_mgr.enabled_for(m))
            {
                // Large prefill: COSMA becomes beneficial
                if (total_elements >= LARGE_PREFILL_THRESHOLD)
                {
                    return MatMulBackendDecision(
                        MatMulBackend::COSMA,
                        "Large prefill (>= 8M elements, seq >= 4K)");
                }

                // Medium-large prefill: still use COSMA if enabled
                return MatMulBackendDecision(
                    MatMulBackend::COSMA,
                    "Prefill with COSMA enabled (seq >= threshold)");
            }
            else
            {
                // COSMA not enabled (env or size constraints)
                if (total_elements >= LARGE_PREFILL_THRESHOLD && mpi_size >= 2)
                {
                    return MatMulBackendDecision(
                        MatMulBackend::DISTRIBUTED_OPENBLAS,
                        "Large prefill, COSMA disabled, using distributed OpenBLAS");
                }
                return MatMulBackendDecision(
                    MatMulBackend::MULTI_THREADED_OPENBLAS,
                    "Prefill, COSMA not enabled");
            }
        }
        else
        {
            // Decode: strongly favor local OpenBLAS

            // Decode is typically single-token or small batch
            // Communication overhead dominates for COSMA
            if (total_elements < SMALL_OP_THRESHOLD)
            {
                return MatMulBackendDecision(
                    MatMulBackend::MULTI_THREADED_OPENBLAS,
                    "Decode (small batch, OpenBLAS dominates)");
            }

            // Even large decode batches: OpenBLAS better due to locality
            return MatMulBackendDecision(
                MatMulBackend::MULTI_THREADED_OPENBLAS,
                "Decode (large batch, avoid communication overhead)");
        }

        // === 6. Default Fallback ===

        // Should rarely reach here; fall back to multi-threaded OpenBLAS
        return MatMulBackendDecision(
            MatMulBackend::MULTI_THREADED_OPENBLAS,
            "Default fallback (no specific threshold matched)");
    }

    void MatMulBackendSelector::logDecision(
        const MatMulBackendDecision &decision,
        const std::string &operation_label,
        int log_level)
    {
        const auto &env = debugEnv();

        // Respect COSMA log level setting (int: 0=error, 1=warn, 2=info, 3=debug, 4=trace)
        int cosma_log_level = env.cosma.log_level;

        // Only log if COSMA log level is high enough
        if (log_level > cosma_log_level)
        {
            return;
        }

        std::ostringstream oss;
        oss << "[MatMulBackendSelector] " << operation_label << " → ";

        switch (decision.backend)
        {
        case MatMulBackend::SINGLE_THREADED_OPENBLAS:
            oss << "SINGLE_THREADED_OPENBLAS";
            break;
        case MatMulBackend::MULTI_THREADED_OPENBLAS:
            oss << "MULTI_THREADED_OPENBLAS";
            break;
        case MatMulBackend::DISTRIBUTED_OPENBLAS:
            oss << "DISTRIBUTED_OPENBLAS";
            break;
        case MatMulBackend::COSMA:
            oss << "COSMA";
            break;
        }

        oss << " (" << decision.rationale << ")";

        // Use log level: 0=error, 1=warn, 2=info, 3=debug, 4=trace
        if (log_level >= 4)
        {
            LOG_TRACE(oss.str());
        }
        else if (log_level >= 3)
        {
            LOG_DEBUG(oss.str());
        }
        else if (log_level >= 2)
        {
            LOG_INFO(oss.str());
        }
        else if (log_level >= 1)
        {
            LOG_WARN(oss.str());
        }
        else
        {
            LOG_ERROR(oss.str());
        }
    }

} // namespace llaminar
