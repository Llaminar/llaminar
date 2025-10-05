/**
 * @file matmul_backend_selection.h
 * @brief Centralized backend selection for matrix multiplication operations.
 * @author David Sanftenberg
 *
 * This module consolidates scattered COSMA/OpenBLAS heuristics into a single
 * decision point, replacing the deprecated shouldUseCosmaPrefill() and implicit
 * adaptiveMatMul backend selection logic.
 */

#pragma once

#include "abstract_pipeline.h"
#include "transformer_config.h"
#include "utils/debug_env.h"
#include "logger.h"
#include <string>

namespace llaminar
{
    /**
     * @brief Matrix multiplication backend options for matmul operations.
     */
    enum class MatMulBackend
    {
        SINGLE_THREADED_OPENBLAS, ///< Single-threaded OpenBLAS (minimal overhead for tiny ops)
        MULTI_THREADED_OPENBLAS,  ///< Multi-threaded OpenBLAS (good for medium ops)
        DISTRIBUTED_OPENBLAS,     ///< Distributed OpenBLAS with MPI collectives
        COSMA                     ///< COSMA distributed matrix multiplication
    };

    /**
     * @brief Backend selection decision with rationale for matmul operations.
     */
    struct MatMulBackendDecision
    {
        MatMulBackend backend;           ///< Selected backend
        std::string rationale;           ///< Human-readable reason for selection
        bool is_distributed;             ///< Whether backend uses MPI collectives
        bool skip_small_op_optimization; ///< Whether to skip single-threaded path

        MatMulBackendDecision(MatMulBackend b, const std::string &r)
            : backend(b), rationale(r),
              is_distributed(b == MatMulBackend::DISTRIBUTED_OPENBLAS || b == MatMulBackend::COSMA),
              skip_small_op_optimization(false)
        {
        }

        // Helper method for backward compatibility
        bool use_cosma() const { return backend == MatMulBackend::COSMA; }
    };

    /**
     * @brief Centralized backend selector for matrix multiplication operations.
     *
     * Replaces scattered heuristics (shouldUseCosmaPrefill, implicit adaptiveMatMul logic)
     * with a single decision function aware of:
     * - Operation size (m, n, k)
     * - Pipeline stage (prefill vs decode)
     * - Model features (from ModelConfig)
     * - Environment overrides (debugEnv)
     * - MPI topology (rank count, distributed partitioning)
     */
    class MatMulBackendSelector
    {
    public:
        /**
         * @brief Select optimal backend for a matrix multiplication.
         *
         * @param m Number of rows in result (typically sequence length)
         * @param n Number of columns in result (typically feature dimension)
         * @param k Inner dimension (typically input feature dimension)
         * @param ctx Stage context (prefill vs decode, sequence info)
         * @param model_config Model configuration (architecture, features)
         * @param distributed_partition Whether caller already partitions across MPI ranks
         * @param mpi_rank Current MPI rank (for logging)
         * @param mpi_size Total MPI ranks
         *
         * @return Backend decision with rationale
         *
         * Decision hierarchy:
         * 1. Environment overrides (ADAPTIVE_DISABLE_COSMA, etc.)
         * 2. Distributed partition safety (never use COSMA for pre-partitioned data)
         * 3. Operation size thresholds (tiny -> single-threaded, huge -> distributed)
         * 4. Stage-specific heuristics (prefill favors COSMA, decode favors OpenBLAS)
         * 5. Model feature awareness (future: GQA, MoE-specific tuning)
         */
        static MatMulBackendDecision selectBackend(
            int m, int n, int k,
            const StageContext &ctx,
            const ModelConfig &model_config,
            bool distributed_partition = false,
            int mpi_rank = 0,
            int mpi_size = 1);

        /**
         * @brief Log backend decision at appropriate verbosity.
         *
         * @param decision Backend decision to log
         * @param operation_label Human-readable operation name (e.g., "Q*K^T", "FFN_gate")
         * @param log_level Minimum log level for output (default: info)
         */
        static void logDecision(
            const MatMulBackendDecision &decision,
            const std::string &operation_label,
            int log_level = 2); // 2 = info level    private:
        // Empirical thresholds (tunable via environment in future)
        static constexpr size_t TINY_OP_THRESHOLD = 8192;            // Total elements below = single-threaded
        static constexpr size_t SMALL_OP_THRESHOLD = 1048576;        // Total elements below = multi-threaded local
        static constexpr size_t PREFILL_COSMA_SEQ_THRESHOLD = 4096;  // Minimum sequence length for COSMA prefill
        static constexpr size_t VOCAB_PROJECTION_THRESHOLD = 100000; // Skip COSMA for huge vocab projections
        static constexpr size_t LARGE_PREFILL_THRESHOLD = 8388608;   // Total elements for distributed consideration
    };

} // namespace llaminar
