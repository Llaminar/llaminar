#pragma once

#include <cstdlib>
#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstring>
#include <optional>

/**
 * @file DebugEnv.h
 * @brief Runtime configuration via environment variables for v2 architecture
 * @author David Sanftenberg
 *
 * Centralized configuration for debug/instrumentation features.
 * All environment variables are parsed once at startup to avoid hot-path overhead.
 */

// =============================================================================
// ASSERTIONS ACTIVE CHECK (duplicated from Assertions.h to avoid circular dep)
// =============================================================================
// Assertions are active when:
//   1. NDEBUG is NOT defined (Debug builds), OR
//   2. LLAMINAR_ENABLE_ASSERTIONS is defined (Integration builds)
// =============================================================================
#ifndef LLAMINAR_ASSERTIONS_ACTIVE
#if !defined(NDEBUG) || defined(LLAMINAR_ENABLE_ASSERTIONS)
#define LLAMINAR_ASSERTIONS_ACTIVE 1
#else
#define LLAMINAR_ASSERTIONS_ACTIVE 0
#endif
#endif

namespace llaminar2
{

    /**
     * @brief Dequantization configuration group
     */
    struct DequantConfig
    {
        // IQ4_NL tile sizes (tuned via LLAMINAR_IQ4_M_TILE and LLAMINAR_IQ4_N_TILE)
        size_t iq4_m_tile = 64; ///< Default: 64 (optimal from tile sweep)
        size_t iq4_n_tile = 32; ///< Default: 32 (optimal from tile sweep)

        // Experimental features
        bool iq4_microkernel = false;     ///< Multi-block vectorized decode (experimental)
        bool iq4_direct_decode = false;   ///< Direct decode path (vs cache-blocked, experimental)
        bool iq4_gemm_microkernel = true; ///< GEMM microkernel optimization (CANONICAL - enabled by default)

        // Override tile sizes (0 = use adaptive defaults)
        int iq4_override_m_tile = 0;      ///< Override M tile size (FP32)
        int iq4_override_n_tile = 0;      ///< Override N tile size (FP32)
        int iq4_override_m_tile_bf16 = 0; ///< Override M tile size (BF16)
        int iq4_override_n_tile_bf16 = 0; ///< Override N tile size (BF16)

        // SIMD path control for testing (LLAMINAR_DEQUANT_SIMD_PATH)
        // Values: "auto" (default), "scalar", "avx2", "avx512"
        std::string simd_path = "auto"; ///< Force specific SIMD path for testing

        DequantConfig()
        {
            // Parse environment variables
            const char *m_tile_env = std::getenv("LLAMINAR_IQ4_M_TILE");
            if (m_tile_env)
            {
                iq4_m_tile = static_cast<size_t>(std::atoi(m_tile_env));
            }

            const char *n_tile_env = std::getenv("LLAMINAR_IQ4_N_TILE");
            if (n_tile_env)
            {
                iq4_n_tile = static_cast<size_t>(std::atoi(n_tile_env));
            }

            const char *microkernel_env = std::getenv("LLAMINAR_IQ4_MICROKERNEL");
            if (microkernel_env)
            {
                iq4_microkernel = (std::atoi(microkernel_env) != 0);
            }

            const char *direct_decode_env = std::getenv("LLAMINAR_IQ4_DIRECT_DECODE");
            if (direct_decode_env)
            {
                iq4_direct_decode = (std::atoi(direct_decode_env) != 0);
            }

            const char *gemm_micro_env = std::getenv("LLAMINAR_IQ4_GEMM_MICROKERNEL");
            if (gemm_micro_env)
            {
                iq4_gemm_microkernel = (std::atoi(gemm_micro_env) != 0);
            }

            // SIMD path control for testing
            const char *simd_path_env = std::getenv("LLAMINAR_DEQUANT_SIMD_PATH");
            if (simd_path_env)
            {
                simd_path = simd_path_env;
            }
        }
    };

    /**
     * @brief Kernel profiling configuration group
     *
     * Controls per-operation timing instrumentation for performance analysis.
     * When disabled, profiling methods have zero overhead (compile-time elimination).
     */
    struct ProfileConfig
    {
        bool enabled = false;       ///< Enable profiling (LLAMINAR_PROFILING=1 or legacy LLAMINAR_PROFILE_KERNELS=1)
        bool per_layer = false;     ///< Breakdown by layer index (LLAMINAR_PROFILE_PER_LAYER=1)
        bool per_iteration = false; ///< Print stats per decode iteration (LLAMINAR_PROFILE_PER_ITER=1)
        int print_interval = 0;     ///< Print every N iterations (0=only at end)

        ProfileConfig()
        {
            reload();
        }

        void reload()
        {
            // New unified env var - enables all profiling
            const char *unified_env = std::getenv("LLAMINAR_PROFILING");
            if (unified_env)
            {
                enabled = (std::atoi(unified_env) != 0);
            }
            // Legacy env var - still supported for backward compatibility
            const char *enabled_env = std::getenv("LLAMINAR_PROFILE_KERNELS");
            if (enabled_env)
            {
                enabled = enabled || (std::atoi(enabled_env) != 0);
            }

            const char *per_layer_env = std::getenv("LLAMINAR_PROFILE_PER_LAYER");
            if (per_layer_env)
            {
                per_layer = (std::atoi(per_layer_env) != 0);
            }

            const char *per_iter_env = std::getenv("LLAMINAR_PROFILE_PER_ITER");
            if (per_iter_env)
            {
                per_iteration = (std::atoi(per_iter_env) != 0);
            }

            const char *interval_env = std::getenv("LLAMINAR_PROFILE_INTERVAL");
            if (interval_env)
            {
                print_interval = std::atoi(interval_env);
            }
        }
    };

    /**
     * @brief GEMM kernel configuration group
     */
    struct GemmConfig
    {
        // Q8_1 GEMM compensation strategy
        bool use_sa_compensation = false; ///< Use sA-based compensation (vs sum_qs-based, experimental)

        // Q8_1 GEMM microkernel variant selection
        bool use_dense_dpbusd = false; ///< Use dense dpbusd (accumulate across K-blocks, experimental)

        // Cache blocking tuning parameters
        float gemm_l2_limit_pct = 0.9f;          ///< L2 cache limit percentage (default: 0.9)
        float gemm_k_tile_threshold_pct = 0.75f; ///< K-tiling threshold percentage (default: 0.75)
        float gemm_target_b_size_pct = 0.5f;     ///< Target B-tile size percentage (default: 0.5)

        // Advanced tuning parameters
        int gemm_oversubscription_factor = 4;  ///< Task oversubscription factor (default: 4)
        int gemm_min_block_size = 65536;       ///< Minimum block size in bytes (default: 64KB)
        int gemm_k_tile_min_blocks = 32;       ///< Minimum K-tile blocks (default: 32)
        int gemm_k_tile_max_blocks = 256;      ///< Maximum K-tile blocks (default: 256)
        float gemm_l3_share_pct = 0.9f;        ///< L3 cache share percentage (default: 0.9)
        int gemm_m_task_granularity = 2;       ///< M-dimension task granularity (default: 2)
        int gemm_m_unroll_factor = 2;          ///< M-dimension loop unroll factor (default: 2)
        int gemm_quant_parallel_threshold = 0; ///< Threshold for parallel quantization (0=auto)

        // JIT tuning parameters (tuned empirically - see Perf__GemmSweep results)
        int gemm_jit_prefetch_distance = 0; ///< Prefetch distance in cache lines (0=disabled, hw prefetch is better)
        int gemm_jit_unroll_n = 8;          ///< JIT N-dimension unroll factor (optimal: 8 for batch prefill)
        int gemm_jit_unroll_k = 1;          ///< JIT K-dimension unroll factor (default: 1)
        int gemm_jit_m_blocking = 1;        ///< M-rows per JIT call (default: 1, try 2/4 for B-reuse)
        bool gemm_dynamic_schedule = false; ///< Use dynamic OMP scheduling (static is better)
        int gemm_n_tile = 0;                ///< N-dimension tile size (0=no tiling is optimal for large batches)

        GemmConfig()
        {
            reload();
        }

        void reload()
        {
            const char *sa_comp_env = std::getenv("LLAMINAR_USE_SA_COMPENSATION");
            if (sa_comp_env)
            {
                use_sa_compensation = (std::atoi(sa_comp_env) != 0);
            }

            const char *dense_env = std::getenv("LLAMINAR_USE_DENSE_DPBUSD");
            if (dense_env)
            {
                use_dense_dpbusd = (std::atoi(dense_env) != 0);
            }

            const char *l2_limit_env = std::getenv("LLAMINAR_GEMM_L2_LIMIT_PCT");
            if (l2_limit_env)
            {
                gemm_l2_limit_pct = std::strtof(l2_limit_env, nullptr);
            }

            const char *k_tile_env = std::getenv("LLAMINAR_GEMM_K_TILE_THRESHOLD_PCT");
            if (k_tile_env)
            {
                gemm_k_tile_threshold_pct = std::strtof(k_tile_env, nullptr);
            }

            const char *target_b_env = std::getenv("LLAMINAR_GEMM_TARGET_B_SIZE_PCT");
            if (target_b_env)
            {
                gemm_target_b_size_pct = std::strtof(target_b_env, nullptr);
            }

            const char *oversub_env = std::getenv("LLAMINAR_GEMM_OVERSUBSCRIPTION_FACTOR");
            if (oversub_env)
                gemm_oversubscription_factor = std::atoi(oversub_env);

            const char *min_block_env = std::getenv("LLAMINAR_GEMM_MIN_BLOCK_SIZE");
            if (min_block_env)
                gemm_min_block_size = std::atoi(min_block_env);

            const char *jit_prefetch_env = std::getenv("LLAMINAR_GEMM_JIT_PREFETCH_DISTANCE");
            if (jit_prefetch_env)
                gemm_jit_prefetch_distance = std::atoi(jit_prefetch_env);

            const char *jit_unroll_env = std::getenv("LLAMINAR_GEMM_JIT_UNROLL_N");
            if (jit_unroll_env)
                gemm_jit_unroll_n = std::atoi(jit_unroll_env);

            const char *jit_unroll_k_env = std::getenv("LLAMINAR_GEMM_JIT_UNROLL_K");
            if (jit_unroll_k_env)
                gemm_jit_unroll_k = std::atoi(jit_unroll_k_env);

            const char *jit_m_blocking_env = std::getenv("LLAMINAR_GEMM_JIT_M_BLOCKING");
            if (jit_m_blocking_env)
                gemm_jit_m_blocking = std::atoi(jit_m_blocking_env);

            const char *dynamic_sched_env = std::getenv("LLAMINAR_GEMM_DYNAMIC_SCHEDULE");
            if (dynamic_sched_env)
                gemm_dynamic_schedule = (std::atoi(dynamic_sched_env) != 0);

            const char *n_tile_env = std::getenv("LLAMINAR_GEMM_N_TILE");
            if (n_tile_env)
                gemm_n_tile = std::atoi(n_tile_env);

            const char *k_min_env = std::getenv("LLAMINAR_GEMM_K_TILE_MIN_BLOCKS");
            if (k_min_env)
                gemm_k_tile_min_blocks = std::atoi(k_min_env);

            const char *k_max_env = std::getenv("LLAMINAR_GEMM_K_TILE_MAX_BLOCKS");
            if (k_max_env)
                gemm_k_tile_max_blocks = std::atoi(k_max_env);

            const char *l3_share_env = std::getenv("LLAMINAR_GEMM_L3_SHARE_PCT");
            if (l3_share_env)
                gemm_l3_share_pct = std::strtof(l3_share_env, nullptr);

            const char *m_gran_env = std::getenv("LLAMINAR_GEMM_M_TASK_GRANULARITY");
            if (m_gran_env)
                gemm_m_task_granularity = std::atoi(m_gran_env);

            const char *m_unroll_env = std::getenv("LLAMINAR_GEMM_M_UNROLL_FACTOR");
            if (m_unroll_env)
                gemm_m_unroll_factor = std::atoi(m_unroll_env);

            const char *quant_thresh_env = std::getenv("LLAMINAR_GEMM_QUANT_PARALLEL_THRESHOLD");
            if (quant_thresh_env)
                gemm_quant_parallel_threshold = std::atoi(quant_thresh_env);

            const char *trace_q8_env = std::getenv("LLAMINAR_TRACE_Q8_1_DIRECT");
            if (trace_q8_env)
                trace_q8_1_direct = (std::atoi(trace_q8_env) != 0);

            const char *cuda_concurrent_env = std::getenv("LLAMINAR_CUDA_CONCURRENT_PREFILL");
            if (cuda_concurrent_env)
                cuda_concurrent_prefill = (std::atoi(cuda_concurrent_env) != 0);

            const char *fused_trace_env = std::getenv("LLAMINAR_CUDA_FUSED_GEMM_TRACE");
            if (fused_trace_env)
                cuda_fused_gemm_trace = (std::atoi(fused_trace_env) != 0);
        }

        bool trace_q8_1_direct = false;      ///< Enable detailed Q8_1 JIT kernel tracing
        bool cuda_concurrent_prefill = true; ///< Multi-stream concurrent fused GEMM projections during CUDA prefill (LLAMINAR_CUDA_CONCURRENT_PREFILL, default ON)
        bool cuda_fused_gemm_trace = false;  ///< Trace fused GEMM output checksums for debugging (LLAMINAR_CUDA_FUSED_GEMM_TRACE)
    };

    /**
     * @brief RMSNorm kernel configuration group
     *
     * Parallelization Tuning (Q8_1 Pure Integer RMSNorm):
     *   The Q8_1 pure integer path uses a 3-phase parallel structure:
     *   - Phase 0: Quantize gamma (sequential, once per call, ~1% of work)
     *   - Phase 1: Parallel compute sumsq for all rows (all threads active)
     *   - Phase 2: Compute inv_rms (sequential, tiny per-row)
     *   - Phase 3: Parallel apply normalization (all threads active)
     *
     *   This avoids the "omp single" bottleneck where threads waited idle during
     *   gamma quantization.
     *
     * Environment Variables:
     *   LLAMINAR_Q8_PURE_INTEGER_RMSNORM - Enable pure integer path (default: 1)
     *   LLAMINAR_RMSNORM_PARALLEL_MIN_ROWS - Min rows for parallelism (default: 64)
     *   LLAMINAR_RMSNORM_PARALLEL_MIN_ELEMS - Min elements for parallelism (default: 65536)
     *   LLAMINAR_RMSNORM_MIN_ELEMS_PER_THREAD - Min elements per thread (default: 8192)
     *   LLAMINAR_RMSNORM_MAX_THREADS - Max threads to use (0=unlimited, default: 0)
     *   LLAMINAR_RMSNORM_Q8_MIN_BYTES_PARALLEL - Min bytes for Q8_1 parallelization (default: 512KB)
     *   LLAMINAR_RMSNORM_Q8_SCALE_THRESHOLD - Bytes threshold for scaling threads (default: 8MB)
     *   LLAMINAR_RMSNORM_Q8_MIN_ROWS_PER_THREAD - Min rows per thread (default: 8)
     */
    struct RMSNormConfig
    {
        // Q8_1 pure integer path - default ON (faster, ~0.995 cosine similarity vs FP32)
        // Disable with LLAMINAR_Q8_PURE_INTEGER_RMSNORM=0 if needed
        bool q8_pure_integer = true; ///< Use pure integer RMSNorm (default: enabled)

        // Two-pass approach - default ON (enables block pipelining by separating y-compute from rescale)
        // Uses global max_abs for entire row instead of per-block max_abs
        // Slight precision tradeoff but enables better vectorization
        bool q8_twopass = true; ///< Use two-pass approach (default: enabled)

        // Parallelization tuning parameters (3-phase parallel structure)
        int parallel_min_rows = 64;      ///< Minimum rows before considering parallelism
        int parallel_min_elems = 65536;  ///< Minimum total elements (rows * cols) for parallelism (~64K)
        int min_elems_per_thread = 8192; ///< Min elements per thread to avoid false sharing (~8K)
        int max_threads = 0;             ///< Maximum threads to use (0 = unlimited/OMP_NUM_THREADS)

        // Q8_1-specific parallelization thresholds (empirically tuned Dec 2025)
        // Dynamic thread scaling based on workload size
        size_t q8_bytes_per_thread = 32 * 1024; ///< Target bytes per thread (32KB)
        int q8_min_rows_per_thread = 4;         ///< Min rows per thread for Q8_1

        RMSNormConfig()
        {
            reload();
        }

        void reload()
        {
            const char *pure_int_env = std::getenv("LLAMINAR_Q8_PURE_INTEGER_RMSNORM");
            if (pure_int_env)
            {
                q8_pure_integer = (std::atoi(pure_int_env) != 0);
            }

            const char *twopass_env = std::getenv("LLAMINAR_Q8_RMSNORM_TWOPASS");
            if (twopass_env)
            {
                q8_twopass = (std::atoi(twopass_env) != 0);
            }

            const char *min_rows_env = std::getenv("LLAMINAR_RMSNORM_PARALLEL_MIN_ROWS");
            if (min_rows_env)
            {
                parallel_min_rows = std::atoi(min_rows_env);
            }

            const char *min_elems_env = std::getenv("LLAMINAR_RMSNORM_PARALLEL_MIN_ELEMS");
            if (min_elems_env)
            {
                parallel_min_elems = std::atoi(min_elems_env);
            }

            const char *elems_per_thread_env = std::getenv("LLAMINAR_RMSNORM_MIN_ELEMS_PER_THREAD");
            if (elems_per_thread_env)
            {
                min_elems_per_thread = std::atoi(elems_per_thread_env);
            }

            const char *max_threads_env = std::getenv("LLAMINAR_RMSNORM_MAX_THREADS");
            if (max_threads_env)
            {
                max_threads = std::atoi(max_threads_env);
            }

            // Q8_1-specific thresholds
            const char *q8_bytes_per_thread_env = std::getenv("LLAMINAR_RMSNORM_Q8_BYTES_PER_THREAD");
            if (q8_bytes_per_thread_env)
            {
                q8_bytes_per_thread = static_cast<size_t>(std::atol(q8_bytes_per_thread_env));
            }

            const char *q8_min_rows_env = std::getenv("LLAMINAR_RMSNORM_Q8_MIN_ROWS_PER_THREAD");
            if (q8_min_rows_env)
            {
                q8_min_rows_per_thread = std::atoi(q8_min_rows_env);
            }
        }
    };

    /**
     * @brief Attention kernel configuration group
     *
     * Controls Q8_1 attention precision/performance tradeoffs.
     *
     * Environment Variables:
     *   LLAMINAR_Q8_ATTENTION_FP32_SCORES - Use FP32 for Q·K scores (default: 0)
     *       When enabled, dequantizes Q and K to FP32 before computing attention scores.
     *       This improves precision (softmax sensitivity to small differences) at the cost
     *       of additional computation. V accumulation remains in optimized Q8_1 path.
     *       Useful when high accuracy is required or for debugging precision issues.
     *
     * Performance Impact (approximate):
     *   - FP32 scores: ~10-20% slower attention, but significantly higher accuracy
     *   - Integer scores: Fastest, but accumulates quantization error through softmax
     *
     * Precision Impact:
     *   - FP32 scores: ~0.999 cosine similarity at ATTENTION_CONTEXT
     *   - Integer scores: ~0.89-0.98 cosine similarity at ATTENTION_CONTEXT
     */
    struct AttentionConfig
    {
        bool fp32_scores = false; ///< Use FP32 for Q·K score computation (default: integer)

        // Wo projection mode (JIT backend only)
        // When enabled, Wo weights are expected to be passed as packed QuantisedPackedWeights
        // and Wo projection is executed via gemm (AVX-512 VNNI) with on-the-fly activation quantization.
        bool wo_vnni_packed = false;

        // CPU flash attention KV tile overrides (0 or negative = disabled)
        int flash_kv_tile_decode = -1;  ///< Override decode kv tile (LLAMINAR_FLASH_ATTN_KV_TILE_DECODE)
        int flash_kv_tile_prefill = -1; ///< Override prefill kv tile (LLAMINAR_FLASH_ATTN_KV_TILE_PREFILL)

        // CPU flash attention prefill INT16 (12-bit effective) Q·K path
        bool flash_prefill_i16_i12 = true;            ///< Enable prefill INT16(i12) Q·K path (LLAMINAR_FLASH_PREFILL_I16_I12)
        int flash_prefill_i16_i12_min_seq = 128;      ///< Minimum seq_len for INT16(i12) path (LLAMINAR_FLASH_PREFILL_I16_I12_MIN_SEQ)
        int flash_prefill_i16_i12_min_kv = 128;       ///< Minimum kv_len for INT16(i12) path (LLAMINAR_FLASH_PREFILL_I16_I12_MIN_KV)
        int64_t flash_prefill_i16_i12_min_work = 0;   ///< Minimum seq_len*kv_len for INT16(i12) path (LLAMINAR_FLASH_PREFILL_I16_I12_MIN_WORK)
        int flash_prefill_i16_i12_qmax = 2047;        ///< Effective quant range cap (LLAMINAR_FLASH_PREFILL_I16_I12_QMAX)
        int flash_prefill_i16_i12_max_head_dim = 256; ///< Max head_dim for safe INT32 accumulation (LLAMINAR_FLASH_PREFILL_I16_I12_MAX_HEAD_DIM)

        AttentionConfig()
        {
            reload();
        }

        void reload()
        {
            const char *fp32_scores_env = std::getenv("LLAMINAR_Q8_ATTENTION_FP32_SCORES");
            if (fp32_scores_env)
            {
                fp32_scores = (std::atoi(fp32_scores_env) != 0);
            }

            const char *wo_vnni_env = std::getenv("LLAMINAR_Q8_WO_VNNI_PACKED");
            if (wo_vnni_env)
            {
                wo_vnni_packed = (std::atoi(wo_vnni_env) != 0);
            }

            const char *flash_decode_tile_env = std::getenv("LLAMINAR_FLASH_ATTN_KV_TILE_DECODE");
            if (flash_decode_tile_env)
            {
                const int parsed = std::atoi(flash_decode_tile_env);
                flash_kv_tile_decode = parsed > 0 ? parsed : -1;
            }

            const char *flash_prefill_tile_env = std::getenv("LLAMINAR_FLASH_ATTN_KV_TILE_PREFILL");
            if (flash_prefill_tile_env)
            {
                const int parsed = std::atoi(flash_prefill_tile_env);
                flash_kv_tile_prefill = parsed > 0 ? parsed : -1;
            }

            const char *flash_i16_i12_env = std::getenv("LLAMINAR_FLASH_PREFILL_I16_I12");
            if (flash_i16_i12_env)
            {
                flash_prefill_i16_i12 = (std::atoi(flash_i16_i12_env) != 0);
            }

            const char *flash_i16_min_seq_env = std::getenv("LLAMINAR_FLASH_PREFILL_I16_I12_MIN_SEQ");
            if (flash_i16_min_seq_env)
            {
                flash_prefill_i16_i12_min_seq = std::max(1, std::atoi(flash_i16_min_seq_env));
            }

            const char *flash_i16_min_kv_env = std::getenv("LLAMINAR_FLASH_PREFILL_I16_I12_MIN_KV");
            if (flash_i16_min_kv_env)
            {
                flash_prefill_i16_i12_min_kv = std::max(1, std::atoi(flash_i16_min_kv_env));
            }

            const char *flash_i16_min_work_env = std::getenv("LLAMINAR_FLASH_PREFILL_I16_I12_MIN_WORK");
            if (flash_i16_min_work_env)
            {
                flash_prefill_i16_i12_min_work = std::max<int64_t>(0, std::atoll(flash_i16_min_work_env));
            }

            const char *flash_i16_qmax_env = std::getenv("LLAMINAR_FLASH_PREFILL_I16_I12_QMAX");
            if (flash_i16_qmax_env)
            {
                const int parsed = std::atoi(flash_i16_qmax_env);
                flash_prefill_i16_i12_qmax = std::max(1, std::min(parsed, 32767));
            }

            const char *flash_i16_max_hd_env = std::getenv("LLAMINAR_FLASH_PREFILL_I16_I12_MAX_HEAD_DIM");
            if (flash_i16_max_hd_env)
            {
                flash_prefill_i16_i12_max_head_dim = std::max(1, std::atoi(flash_i16_max_hd_env));
            }
        }
    };

    /**
     * @brief Q16 attention debug dump configuration
     *
     * Environment Variables:
     *   LLAMINAR_Q16_ATTN_DUMP            - Enable targeted Q16 attention dumps (default: 0)
     *   LLAMINAR_Q16_ATTN_DUMP_ONCE       - Dump only once per process (default: 1)
     *   LLAMINAR_Q16_ATTN_DUMP_LAYER      - Layer index to dump (default: -1 = any)
     *   LLAMINAR_Q16_ATTN_DUMP_HEAD       - Query head index to dump (default: 0)
     *   LLAMINAR_Q16_ATTN_DUMP_ROW        - Query row to dump (default: -1 = last row)
     *   LLAMINAR_Q16_ATTN_DUMP_TOPK       - Top-k entries to print for scores/weights (default: 8)
     */
    struct Q16AttentionDumpConfig
    {
        bool enabled = false;
        bool once = true;
        int layer = -1;
        int head = 0;
        int row = -1;
        int topk = 8;

        Q16AttentionDumpConfig() { reload(); }

        void reload()
        {
            const char *enabled_env = std::getenv("LLAMINAR_Q16_ATTN_DUMP");
            if (enabled_env)
            {
                enabled = (std::atoi(enabled_env) != 0);
            }

            const char *once_env = std::getenv("LLAMINAR_Q16_ATTN_DUMP_ONCE");
            if (once_env)
            {
                once = (std::atoi(once_env) != 0);
            }

            const char *layer_env = std::getenv("LLAMINAR_Q16_ATTN_DUMP_LAYER");
            if (layer_env)
            {
                layer = std::atoi(layer_env);
            }

            const char *head_env = std::getenv("LLAMINAR_Q16_ATTN_DUMP_HEAD");
            if (head_env)
            {
                head = std::atoi(head_env);
            }

            const char *row_env = std::getenv("LLAMINAR_Q16_ATTN_DUMP_ROW");
            if (row_env)
            {
                row = std::atoi(row_env);
            }

            const char *topk_env = std::getenv("LLAMINAR_Q16_ATTN_DUMP_TOPK");
            if (topk_env)
            {
                topk = std::atoi(topk_env);
            }
        }
    };

    /**
     * @brief Execution framework configuration group
     *
     * Controls the Graph-based execution system (LayerExecutor / DeviceGraphOrchestrator).
     * As of December 2025, the Graph execution system is the PRIMARY path.
     *
     * Environment Variables:
     *   LLAMINAR_USE_LAYER_EXECUTOR        - Enable LayerExecutor (default: 1 - ON)
     *   LLAMINAR_EXECUTION_MODE            - Execution mode: "sequential", "parallel", "pipelined" (default: "sequential")
     *   LLAMINAR_EXECUTOR_PROFILING        - Enable per-stage profiling in LayerExecutor (default: 0)
     *   LLAMINAR_EXECUTOR_VALIDATION       - Enable output validation after each stage (default: 0)
     *   LLAMINAR_AUTO_WEIGHT_TRANSFER      - Auto-transfer weights to target device (default: 1)
     *   LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT - Use DeviceGraphBufferManager for buffers (default: 1 - ON)
     *   LLAMINAR_EXEC_FULL_FORWARD         - Use full forward graph execution (default: 1 - ON)
     *   LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED - Allow segmented GPU-graph replay for decode graphs
     *                                        containing collectives (default: 0 - OFF, experimental)
     *
     * Device Placement / Heterogeneous Execution:
     *   LLAMINAR_CPU_PREFILL_PARTICIPATE   - Enable CPU participation in PREFILL phase (default: 0 - OFF)
     *                                        When OFF (default), CPU only participates in DECODE phase
     *                                        (Option A: Selective Duplication - GPU prefill, CPU decode).
     *                                        When ON, CPU participates in BOTH prefill and decode
     *                                        (Option C: Full CPU Participation - for memory-constrained systems).
     *
     * Per-Operation Flags (all default to 1 - ON):
     *   These flags exist for debugging only - to selectively DISABLE operations.
     *   Set to 0 to disable the corresponding ComputeStage:
     *   LLAMINAR_EXEC_EMBEDDING            - Use ComputeStage for Embedding lookup
     *   LLAMINAR_EXEC_LM_HEAD              - Use ComputeStage for LM head projection
     *   LLAMINAR_EXEC_RMSNORM              - Use ComputeStage for RMSNorm
     *   LLAMINAR_EXEC_ROPE                 - Use ComputeStage for RoPE
     *   LLAMINAR_EXEC_ATTENTION            - Use ComputeStage for Attention
     *   LLAMINAR_EXEC_GEMM                 - Use ComputeStage for GEMM
     *   LLAMINAR_EXEC_SWIGLU               - Use ComputeStage for SwiGLU
     *   LLAMINAR_EXEC_RESIDUAL             - Use ComputeStage for residual add
     *
     * Example Usage:
     *   # Run with profiling enabled
     *   LLAMINAR_EXECUTOR_PROFILING=1 ./run_llaminar.sh ...
     *
     *   # Debug: disable only SwiGLU stage (all others remain enabled)
     *   LLAMINAR_EXEC_SWIGLU=0 ./run_llaminar.sh ...
     */
    struct ExecutionConfig
    {
        bool use_layer_executor = true;                                        ///< Master switch for LayerExecutor (default: ON)
        std::string execution_mode = "sequential";                             ///< Execution mode
        bool executor_profiling = false;                                       ///< Enable stage profiling
        bool executor_validation = false;                                      ///< Validate outputs after each stage
        bool auto_weight_transfer = true;                                      ///< Auto-transfer weights to device
        bool use_graph_buffer_management = true;                               ///< Use DeviceGraphBufferManager for buffer allocation (default: ON)
        bool exec_full_forward = true;                                         ///< Use orchestrator->executeForward() for complete inference (default: ON)
        bool fast_decode = true;                                               ///< Use fast decode path skipping coherence/debug overhead (default: ON, env: LLAMINAR_FAST_DECODE)
        bool gpu_graphs = true;                                                ///< Use GPU graph capture/replay for decode (default: ON, env: LLAMINAR_GPU_GRAPHS)
        bool gpu_graph_verify = false;                                         ///< Verify graph replay vs direct execution (default: OFF, env: LLAMINAR_GPU_GRAPH_VERIFY)
        bool gpu_graph_recapture = false;                                      ///< Re-capture each decode step instead of replaying cached graph (default: OFF, env: LLAMINAR_GPU_GRAPH_RECAPTURE)
        int gpu_graph_max_stages = 0;                                          ///< Max stages per capturable segment (0=unlimited, env: LLAMINAR_GPU_GRAPH_MAX_STAGES)
        bool gpu_graph_collective_segmented = false;                           ///< Enable segmented replay for collective decode graphs (default: OFF, env: LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED)
        std::vector<std::string> gpu_graph_collective_segmented_capture_allow; ///< Optional stage-name allowlist for segmented collective capture (env: LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED_CAPTURE_ALLOW)
        bool gpu_graph_stream_only = false;                                    ///< Execute segmented path on stream-only mode (env: LLAMINAR_GPU_GRAPH_STREAM_ONLY)
        bool gpu_graph_stream_only_default = false;                            ///< Stream-only mode uses default stream (env: LLAMINAR_GPU_GRAPH_STREAM_ONLY_DEFAULT)
        bool gpu_graph_trace_replay = false;                                   ///< Trace per-segment progress during graph replay (env: LLAMINAR_GPU_GRAPH_TRACE_REPLAY)
        bool force_mpi_collective_context = false;                             ///< Force MPI-backed CollectiveContext in GLOBAL TP (env: LLAMINAR_FORCE_MPI_COLLECTIVE_CONTEXT)

        // =================================================================
        // Device Placement / Heterogeneous Execution
        // =================================================================

        /**
         * @brief Enable CPU participation in PREFILL phase (LLAMINAR_CPU_PREFILL_PARTICIPATE)
         *
         * By default (false), CPU only participates in DECODE phase:
         *   - PREFILL: GPU only (compute-bound, benefits from GPU parallelism)
         *   - DECODE: CPU + GPU (memory-bound, CPU can help with bandwidth)
         *   This is "Option A: Selective Duplication" - optimal for most systems.
         *
         * When enabled (true), CPU participates in BOTH prefill and decode:
         *   - PREFILL: CPU + GPU (slower but uses less GPU memory)
         *   - DECODE: CPU + GPU
         *   This is "Option C: Full CPU Participation" - for memory-constrained systems
         *   where GPU cannot hold 100% of weights.
         *
         * @note This is an escape hatch for memory-constrained systems.
         *       The default (false) provides better prefill performance.
         */
        bool cpu_prefill_participate = false;

        // =================================================================
        // Stage Tracing Configuration (Task 3: Debugging Infrastructure)
        // =================================================================

        /**
         * @brief Enable stage execution tracing (LLAMINAR_TRACE_STAGES)
         *
         * When enabled, logs input/output tensor values for each compute stage.
         * Useful for debugging divergence between configurations (e.g., 1 rank vs 2 ranks).
         */
        bool trace_stages = false;

        /**
         * @brief Include tensor shapes in trace output (LLAMINAR_TRACE_SHAPES)
         */
        bool trace_shapes = false;

        /**
         * @brief Number of tensor elements to print (LLAMINAR_TRACE_SAMPLE_COUNT)
         */
        int trace_sample_count = 8;

        /**
         * @brief Only trace stages matching this substring (LLAMINAR_TRACE_FILTER)
         *
         * Examples: "layer0", "attention", "ffn_down"
         * Empty string = trace all stages
         */
        std::string trace_filter = "";

        /**
         * @brief Compute and log checksum of tensor data (LLAMINAR_TRACE_CHECKSUMS)
         *
         * Useful for detecting divergence without examining all values.
         */
        bool trace_checksums = false;

        // Per-operation feature flags - ALL ENABLED BY DEFAULT as of Dec 2025
        // These flags now exist only for debugging (to selectively disable operations)
        // Model-level operations (embedding, final norm, lm head)
        bool exec_embedding = true; ///< Use ComputeStage for Embedding lookup
        bool exec_lm_head = true;   ///< Use ComputeStage for LM head projection
        // Layer-level operations
        bool exec_rmsnorm = true;   ///< Use ComputeStage for RMSNorm
        bool exec_rope = true;      ///< Use ComputeStage for RoPE
        bool exec_attention = true; ///< Use ComputeStage for Attention
        bool exec_gemm = true;      ///< Use ComputeStage for GEMM
        bool exec_swiglu = true;    ///< Use ComputeStage for SwiGLU
        bool exec_residual = true;  ///< Use ComputeStage for residual add

        ExecutionConfig()
        {
            reload();
        }

        void reload()
        {
            const char *use_exec_env = std::getenv("LLAMINAR_USE_LAYER_EXECUTOR");
            if (use_exec_env)
            {
                use_layer_executor = (std::atoi(use_exec_env) != 0);
            }

            const char *mode_env = std::getenv("LLAMINAR_EXECUTION_MODE");
            if (mode_env)
            {
                execution_mode = mode_env;
            }

            // New unified env var - enables all profiling including executor profiling
            const char *unified_env = std::getenv("LLAMINAR_PROFILING");
            if (unified_env)
            {
                executor_profiling = (std::atoi(unified_env) != 0);
            }
            // Legacy env var - still supported for backward compatibility
            const char *prof_env = std::getenv("LLAMINAR_EXECUTOR_PROFILING");
            if (prof_env)
            {
                executor_profiling = executor_profiling || (std::atoi(prof_env) != 0);
            }

            const char *valid_env = std::getenv("LLAMINAR_EXECUTOR_VALIDATION");
            if (valid_env)
            {
                executor_validation = (std::atoi(valid_env) != 0);
            }

            const char *auto_xfer_env = std::getenv("LLAMINAR_AUTO_WEIGHT_TRANSFER");
            if (auto_xfer_env)
            {
                auto_weight_transfer = (std::atoi(auto_xfer_env) != 0);
            }

            const char *graph_buf_env = std::getenv("LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT");
            if (graph_buf_env)
            {
                use_graph_buffer_management = (std::atoi(graph_buf_env) != 0);
            }

            const char *full_forward_env = std::getenv("LLAMINAR_EXEC_FULL_FORWARD");
            if (full_forward_env)
            {
                exec_full_forward = (std::atoi(full_forward_env) != 0);
            }

            const char *fast_decode_env = std::getenv("LLAMINAR_FAST_DECODE");
            if (fast_decode_env)
            {
                fast_decode = (std::atoi(fast_decode_env) != 0);
            }

            const char *gpu_graphs_env = std::getenv("LLAMINAR_GPU_GRAPHS");
            if (gpu_graphs_env)
            {
                gpu_graphs = (std::atoi(gpu_graphs_env) != 0);
            }

            const char *gpu_graph_verify_env = std::getenv("LLAMINAR_GPU_GRAPH_VERIFY");
            if (gpu_graph_verify_env)
            {
                gpu_graph_verify = (std::atoi(gpu_graph_verify_env) != 0);
            }

            const char *gpu_graph_recapture_env = std::getenv("LLAMINAR_GPU_GRAPH_RECAPTURE");
            if (gpu_graph_recapture_env)
            {
                gpu_graph_recapture = (std::atoi(gpu_graph_recapture_env) != 0);
            }

            const char *gpu_graph_max_stages_env = std::getenv("LLAMINAR_GPU_GRAPH_MAX_STAGES");
            if (gpu_graph_max_stages_env)
            {
                gpu_graph_max_stages = std::atoi(gpu_graph_max_stages_env);
            }

            const char *gpu_graph_collective_segmented_env = std::getenv("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED");
            if (gpu_graph_collective_segmented_env)
            {
                gpu_graph_collective_segmented = (std::atoi(gpu_graph_collective_segmented_env) != 0);
            }

            gpu_graph_collective_segmented_capture_allow.clear();
            const char *gpu_graph_collective_segmented_allow_env = std::getenv("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED_CAPTURE_ALLOW");
            if (gpu_graph_collective_segmented_allow_env && *gpu_graph_collective_segmented_allow_env)
            {
                std::stringstream ss(gpu_graph_collective_segmented_allow_env);
                std::string token;
                while (std::getline(ss, token, ','))
                {
                    token.erase(0, token.find_first_not_of(" \t"));
                    token.erase(token.find_last_not_of(" \t") + 1);
                    if (!token.empty())
                    {
                        gpu_graph_collective_segmented_capture_allow.push_back(token);
                    }
                }
            }

            const char *gpu_graph_stream_only_env = std::getenv("LLAMINAR_GPU_GRAPH_STREAM_ONLY");
            if (gpu_graph_stream_only_env)
            {
                gpu_graph_stream_only = (std::atoi(gpu_graph_stream_only_env) != 0);
            }

            const char *gpu_graph_stream_only_default_env = std::getenv("LLAMINAR_GPU_GRAPH_STREAM_ONLY_DEFAULT");
            if (gpu_graph_stream_only_default_env)
            {
                gpu_graph_stream_only_default = (std::atoi(gpu_graph_stream_only_default_env) != 0);
            }

            const char *gpu_graph_trace_replay_env = std::getenv("LLAMINAR_GPU_GRAPH_TRACE_REPLAY");
            if (gpu_graph_trace_replay_env)
            {
                gpu_graph_trace_replay = (std::atoi(gpu_graph_trace_replay_env) != 0);
            }

            const char *force_mpi_collective_ctx_env = std::getenv("LLAMINAR_FORCE_MPI_COLLECTIVE_CONTEXT");
            if (force_mpi_collective_ctx_env)
            {
                force_mpi_collective_context = (std::atoi(force_mpi_collective_ctx_env) != 0);
            }

            // Model-level operation flags (embedding, lm_head)
            const char *embedding_env = std::getenv("LLAMINAR_EXEC_EMBEDDING");
            if (embedding_env)
                exec_embedding = (std::atoi(embedding_env) != 0);

            const char *lm_head_env = std::getenv("LLAMINAR_EXEC_LM_HEAD");
            if (lm_head_env)
                exec_lm_head = (std::atoi(lm_head_env) != 0);

            // Layer-level operation flags
            const char *rmsnorm_env = std::getenv("LLAMINAR_EXEC_RMSNORM");
            if (rmsnorm_env)
                exec_rmsnorm = (std::atoi(rmsnorm_env) != 0);

            const char *rope_env = std::getenv("LLAMINAR_EXEC_ROPE");
            if (rope_env)
                exec_rope = (std::atoi(rope_env) != 0);

            const char *attn_env = std::getenv("LLAMINAR_EXEC_ATTENTION");
            if (attn_env)
                exec_attention = (std::atoi(attn_env) != 0);

            const char *gemm_env = std::getenv("LLAMINAR_EXEC_GEMM");
            if (gemm_env)
                exec_gemm = (std::atoi(gemm_env) != 0);

            const char *swiglu_env = std::getenv("LLAMINAR_EXEC_SWIGLU");
            if (swiglu_env)
                exec_swiglu = (std::atoi(swiglu_env) != 0);

            const char *residual_env = std::getenv("LLAMINAR_EXEC_RESIDUAL");
            if (residual_env)
                exec_residual = (std::atoi(residual_env) != 0);

            // Device placement configuration
            const char *cpu_prefill_env = std::getenv("LLAMINAR_CPU_PREFILL_PARTICIPATE");
            if (cpu_prefill_env)
            {
                std::string val(cpu_prefill_env);
                cpu_prefill_participate = (val == "1" || val == "true");
            }

            // Stage tracing configuration
            const char *trace_stages_env = std::getenv("LLAMINAR_TRACE_STAGES");
            if (trace_stages_env)
                trace_stages = (std::atoi(trace_stages_env) != 0);

            const char *trace_shapes_env = std::getenv("LLAMINAR_TRACE_SHAPES");
            if (trace_shapes_env)
                trace_shapes = (std::atoi(trace_shapes_env) != 0);

            const char *trace_count_env = std::getenv("LLAMINAR_TRACE_SAMPLE_COUNT");
            if (trace_count_env)
                trace_sample_count = std::atoi(trace_count_env);

            const char *trace_filter_env = std::getenv("LLAMINAR_TRACE_FILTER");
            if (trace_filter_env)
                trace_filter = trace_filter_env;

            const char *trace_checksums_env = std::getenv("LLAMINAR_TRACE_CHECKSUMS");
            if (trace_checksums_env)
                trace_checksums = (std::atoi(trace_checksums_env) != 0);

            // NOTE: Legacy cascade logic removed (Dec 2025)
            // All exec_* flags now default to true. The cascade logic below was
            // for incremental migration - now that Graph is the primary path,
            // all stages are always enabled unless explicitly disabled for debugging.
        }
    };

    /**
     * @brief Snapshot and tensor dump configuration group
     *
     * Controls the snapshot framework for E2E parity testing and debugging.
     * When tensor_dump is enabled, raw tensor data (both FP32 dequantized and
     * native Q8_1 blocks) are saved to disk for detailed analysis.
     *
     * Environment Variables:
     *   LLAMINAR_SNAPSHOT_TENSOR_DUMP      - Enable tensor dump to disk (default: 0)
     *   LLAMINAR_SNAPSHOT_DUMP_DIR         - Output directory (default: "/tmp/llaminar_tensor_dumps")
     *   LLAMINAR_SNAPSHOT_DUMP_LAYERS      - Comma-separated layer indices to dump (default: "all")
     *                                        Examples: "21", "0,5,21", "all"
     *   LLAMINAR_SNAPSHOT_DUMP_STAGES      - Comma-separated stage names to dump (default: "all")
     *                                        Examples: "FFN_RESIDUAL", "FFN_DOWN,FFN_RESIDUAL", "all"
     *   LLAMINAR_SNAPSHOT_DUMP_RANK        - Only dump from this MPI rank (-1=all, default: 0)
     *
     * Stage Names (for LLAMINAR_SNAPSHOT_DUMP_STAGES):
     *   EMBEDDING, FINAL_NORM, LM_HEAD
     *   Per-layer: ATTENTION_NORM, Q_PROJECTION, K_PROJECTION, V_PROJECTION,
     *              Q_ROPE, K_ROPE, ATTENTION_CONTEXT, ATTENTION_OUTPUT,
     *              ATTENTION_RESIDUAL, FFN_NORM, FFN_GATE, FFN_UP,
     *              FFN_SWIGLU, FFN_DOWN, FFN_INPUT_RESIDUAL, FFN_RESIDUAL
     *
     * Example Usage:
     *   # Dump layer 21 FFN stages for debugging residual add issues
     *   LLAMINAR_SNAPSHOT_TENSOR_DUMP=1 \
     *   LLAMINAR_SNAPSHOT_DUMP_LAYERS=21 \
     *   LLAMINAR_SNAPSHOT_DUMP_STAGES=FFN_INPUT_RESIDUAL,FFN_DOWN,FFN_RESIDUAL \
     *   ./run_llaminar.sh -m model.gguf -p "test"
     *
     *   # Enable zero-copy mapped memory for snapshots
     *   LLAMINAR_SNAPSHOT_USE_MAPPED=1 ./run_llaminar.sh -m model.gguf -p "test"
     */
    struct SnapshotConfig
    {
        bool tensor_dump_enabled = false;                    ///< Enable raw tensor dump to disk
        std::string dump_dir = "/tmp/llaminar_tensor_dumps"; ///< Output directory for dumps
        std::set<int> dump_layers;                           ///< Layer indices to dump (empty=all)
        std::set<std::string> dump_stages;                   ///< Stage names to dump (empty=all)
        int dump_rank = 0;                                   ///< MPI rank to dump (-1=all)
        bool dump_all_layers = true;                         ///< Whether to dump all layers
        bool dump_all_stages = true;                         ///< Whether to dump all stages
        bool use_mapped_memory = false;                      ///< Use mapped memory for zero-copy snapshots

        SnapshotConfig()
        {
            reload();
        }

        void reload()
        {
            const char *enabled_env = std::getenv("LLAMINAR_SNAPSHOT_TENSOR_DUMP");
            if (enabled_env)
            {
                tensor_dump_enabled = (std::atoi(enabled_env) != 0);
            }

            // LLAMINAR_SNAPSHOT_USE_MAPPED - Enable mapped memory for zero-copy snapshots
            // When enabled, FP32 activation buffers are allocated using mapped memory
            // (hipHostMallocMapped/cudaHostAllocMapped) which enables zero-copy access
            // from both host and device without memcpy.
            const char *mapped_env = std::getenv("LLAMINAR_SNAPSHOT_USE_MAPPED");
            if (mapped_env)
            {
                use_mapped_memory = (std::atoi(mapped_env) != 0);
            }

            const char *dir_env = std::getenv("LLAMINAR_SNAPSHOT_DUMP_DIR");
            if (dir_env)
            {
                dump_dir = dir_env;
            }

            const char *layers_env = std::getenv("LLAMINAR_SNAPSHOT_DUMP_LAYERS");
            if (layers_env)
            {
                std::string layers_str(layers_env);
                if (layers_str == "all")
                {
                    dump_all_layers = true;
                    dump_layers.clear();
                }
                else
                {
                    dump_all_layers = false;
                    dump_layers.clear();
                    std::istringstream iss(layers_str);
                    std::string token;
                    while (std::getline(iss, token, ','))
                    {
                        // Trim whitespace
                        token.erase(0, token.find_first_not_of(" \t"));
                        token.erase(token.find_last_not_of(" \t") + 1);
                        if (!token.empty())
                        {
                            dump_layers.insert(std::atoi(token.c_str()));
                        }
                    }
                }
            }

            const char *stages_env = std::getenv("LLAMINAR_SNAPSHOT_DUMP_STAGES");
            if (stages_env)
            {
                std::string stages_str(stages_env);
                if (stages_str == "all")
                {
                    dump_all_stages = true;
                    dump_stages.clear();
                }
                else
                {
                    dump_all_stages = false;
                    dump_stages.clear();
                    std::istringstream iss(stages_str);
                    std::string token;
                    while (std::getline(iss, token, ','))
                    {
                        // Trim whitespace
                        token.erase(0, token.find_first_not_of(" \t"));
                        token.erase(token.find_last_not_of(" \t") + 1);
                        if (!token.empty())
                        {
                            dump_stages.insert(token);
                        }
                    }
                }
            }

            const char *rank_env = std::getenv("LLAMINAR_SNAPSHOT_DUMP_RANK");
            if (rank_env)
            {
                dump_rank = std::atoi(rank_env);
            }
        }

        /**
         * @brief Check if a specific layer should be dumped
         * @param layer_idx Layer index to check
         * @return true if this layer should be dumped
         */
        bool shouldDumpLayer(int layer_idx) const
        {
            return dump_all_layers || dump_layers.count(layer_idx) > 0;
        }

        /**
         * @brief Check if a specific stage should be dumped
         * @param stage_name Stage name to check (e.g., "FFN_RESIDUAL")
         * @return true if this stage should be dumped
         */
        bool shouldDumpStage(const std::string &stage_name) const
        {
            return dump_all_stages || dump_stages.count(stage_name) > 0;
        }

        /**
         * @brief Check if a specific layer/stage combination should be dumped
         * @param layer_idx Layer index (-1 for non-layer stages like EMBEDDING)
         * @param stage_name Stage name suffix (e.g., "FFN_RESIDUAL")
         * @return true if this combination should be dumped
         */
        bool shouldDump(int layer_idx, const std::string &stage_name) const
        {
            if (!tensor_dump_enabled)
                return false;
            if (layer_idx >= 0 && !shouldDumpLayer(layer_idx))
                return false;
            return shouldDumpStage(stage_name);
        }

        /**
         * @brief Parse a snapshot key to extract layer index and stage name
         *
         * Handles keys like "layer21_FFN_RESIDUAL" → (21, "FFN_RESIDUAL")
         * and "EMBEDDING" → (-1, "EMBEDDING")
         *
         * @param key Full snapshot key
         * @param layer_idx Output: layer index (-1 if not a layer-specific key)
         * @param stage_name Output: stage name suffix
         */
        static void parseSnapshotKey(const std::string &key, int &layer_idx, std::string &stage_name)
        {
            // Check for "layer<N>_" prefix
            if (key.substr(0, 5) == "layer")
            {
                size_t underscore = key.find('_');
                if (underscore != std::string::npos)
                {
                    layer_idx = std::atoi(key.substr(5, underscore - 5).c_str());
                    stage_name = key.substr(underscore + 1);
                    return;
                }
            }
            // No layer prefix - global stage
            layer_idx = -1;
            stage_name = key;
        }

        /**
         * @brief Check if a snapshot key should be dumped based on current config
         * @param key Full snapshot key (e.g., "layer21_FFN_RESIDUAL")
         * @return true if this snapshot should be dumped
         */
        bool shouldDumpKey(const std::string &key) const
        {
            if (!tensor_dump_enabled)
                return false;

            int layer_idx;
            std::string stage_name;
            parseSnapshotKey(key, layer_idx, stage_name);
            return shouldDump(layer_idx, stage_name);
        }

        /**
         * @brief Check if this MPI rank should perform dumps
         * @param rank Current MPI rank
         * @return true if this rank should dump
         */
        bool shouldDumpRank(int rank) const
        {
            return dump_rank == -1 || dump_rank == rank;
        }
    };

    /**
     * @brief Configuration for compute stage input/output dumping
     *
     * Enables first-class debugging of individual compute stages by dumping
     * all inputs, outputs, and parameters to disk for later analysis or replay.
     *
     * Unlike SnapshotConfig (which captures tensor values at pipeline stages),
     * StageDumpConfig captures the exact kernel-level inputs for any ComputeStage:
     * - GEMM: A matrix, B tensor (with dequant info), C output, dimensions
     * - RMSNorm: input, gamma weights, output, eps
     * - Attention: Q/K/V projections (Q8_1 blocks), mask, scale, output
     * - SwiGLU: gate input, up input, output
     * - etc.
     *
     * Environment Variables:
     * - LLAMINAR_STAGE_DUMP_ENABLED=1          Master enable for stage dumping
     * - LLAMINAR_STAGE_DUMP_DIR=/path          Output directory (default: /tmp/llaminar_stage_dumps)
     * - LLAMINAR_STAGE_DUMP_TYPES=GEMM,ATTENTION  Stage types to dump (default: all)
     * - LLAMINAR_STAGE_DUMP_NAMES=layer0_attention  Stage names to dump (default: all)
     * - LLAMINAR_STAGE_DUMP_LAYERS=0,1,5       Layer indices to dump (default: all)
     * - LLAMINAR_STAGE_DUMP_RANK=0             MPI rank to dump (-1=all, default: 0)
     * - LLAMINAR_STAGE_DUMP_MAX=100            Maximum dumps per type (prevent disk explosion)
     * - LLAMINAR_STAGE_DUMP_INPUTS=1           Dump input tensors (default: 1)
     * - LLAMINAR_STAGE_DUMP_OUTPUTS=1          Dump output tensors (default: 1)
     * - LLAMINAR_STAGE_DUMP_WEIGHTS=1          Dump weight tensors (default: 1)
     * - LLAMINAR_STAGE_DUMP_ITERATION=0,1,2    Specific decode iterations to dump (default: all)
     *
     * Stage Types (from ComputeStageType enum):
     *   GEMM, GEMM_BIAS, GEMM_FUSED_QKV, RMS_NORM, LAYER_NORM, SWIGLU, GELU, SILU,
     *   ROPE, ATTENTION, ATTENTION_QK, ATTENTION_SOFTMAX, ATTENTION_V,
     *   ADD_RESIDUAL, SCALE, MOE_ROUTER, MOE_EXPERT_FFN, MOE_COMBINE,
     *   ALLREDUCE, ALLGATHER, COPY, QUANTIZE, DEQUANTIZE
     *
     * Output Format:
     *   <dump_dir>/stage_<counter>_<type>_layer<N>/
     *     metadata.txt        - Human-readable parameters
     *     params.bin          - Binary struct for replay
     *     input_A.bin         - Input matrix A (FP32 or Q8_1 blocks)
     *     input_B_dequant.bin - Weight tensor dequantized to FP32
     *     input_B_raw.bin     - Raw quantized weight data
     *     output_C.bin        - Output tensor
     *     <stage-specific files>
     *
     * Example Usage:
     *   # Dump all GEMM stages in layer 0
     *   LLAMINAR_STAGE_DUMP_ENABLED=1 \
     *   LLAMINAR_STAGE_DUMP_TYPES=GEMM \
     *   LLAMINAR_STAGE_DUMP_LAYERS=0 \
     *   ./run_llaminar.sh -m model.gguf -p "test"
     *
     *   # Dump attention stage by name for HybridQ16 debugging
     *   LLAMINAR_STAGE_DUMP_ENABLED=1 \
     *   LLAMINAR_STAGE_DUMP_NAMES=layer0_attention \
     *   ./run_llaminar.sh -m model.gguf -p "test"
     *
     *   # Dump attention stages for first decode iteration only
     *   LLAMINAR_STAGE_DUMP_ENABLED=1 \
     *   LLAMINAR_STAGE_DUMP_TYPES=ATTENTION \
     *   LLAMINAR_STAGE_DUMP_ITERATION=0 \
     *   ./run_llaminar.sh -m model.gguf -p "test"
     */
    struct StageDumpConfig
    {
        bool enabled = false;                               ///< Master enable for stage dumping
        std::string dump_dir = "/tmp/llaminar_stage_dumps"; ///< Output directory
        std::set<std::string> dump_types;                   ///< Stage types to dump (empty=all)
        std::set<std::string> dump_names;                   ///< Stage names to dump (empty=all)
        std::set<int> dump_layers;                          ///< Layer indices to dump (empty=all)
        std::set<int> dump_iterations;                      ///< Decode iterations to dump (empty=all)
        int dump_rank = 0;                                  ///< MPI rank to dump (-1=all)
        int max_dumps_per_type = 100;                       ///< Max dumps per stage type (prevent disk explosion)
        bool dump_inputs = true;                            ///< Dump input tensors
        bool dump_outputs = true;                           ///< Dump output tensors
        bool dump_weights = true;                           ///< Dump weight tensors
        bool async_dump = true;                             ///< Use async I/O for dumps (default: true)
        int async_threads = 2;                              ///< Number of async I/O threads
        bool dump_all_types = true;                         ///< Whether to dump all stage types
        bool dump_all_names = true;                         ///< Whether to dump all stage names
        bool dump_all_layers = true;                        ///< Whether to dump all layers
        bool dump_all_iterations = true;                    ///< Whether to dump all iterations

        StageDumpConfig()
        {
            reload();
        }

        void reload()
        {
            const char *enabled_env = std::getenv("LLAMINAR_STAGE_DUMP_ENABLED");
            if (enabled_env)
            {
                enabled = (std::atoi(enabled_env) != 0);
            }

            const char *dir_env = std::getenv("LLAMINAR_STAGE_DUMP_DIR");
            if (dir_env)
            {
                dump_dir = dir_env;
            }

            const char *types_env = std::getenv("LLAMINAR_STAGE_DUMP_TYPES");
            if (types_env)
            {
                std::string types_str(types_env);
                if (types_str == "all")
                {
                    dump_all_types = true;
                    dump_types.clear();
                }
                else
                {
                    dump_all_types = false;
                    dump_types.clear();
                    std::istringstream iss(types_str);
                    std::string token;
                    while (std::getline(iss, token, ','))
                    {
                        token.erase(0, token.find_first_not_of(" \t"));
                        token.erase(token.find_last_not_of(" \t") + 1);
                        if (!token.empty())
                        {
                            dump_types.insert(token);
                        }
                    }
                }
            }

            const char *names_env = std::getenv("LLAMINAR_STAGE_DUMP_NAMES");
            if (names_env)
            {
                std::string names_str(names_env);
                if (names_str == "all")
                {
                    dump_all_names = true;
                    dump_names.clear();
                }
                else
                {
                    dump_all_names = false;
                    dump_names.clear();
                    std::istringstream iss(names_str);
                    std::string token;
                    while (std::getline(iss, token, ','))
                    {
                        token.erase(0, token.find_first_not_of(" \t"));
                        token.erase(token.find_last_not_of(" \t") + 1);
                        if (!token.empty())
                        {
                            dump_names.insert(token);
                        }
                    }
                }
            }

            const char *layers_env = std::getenv("LLAMINAR_STAGE_DUMP_LAYERS");
            if (layers_env)
            {
                std::string layers_str(layers_env);
                if (layers_str == "all")
                {
                    dump_all_layers = true;
                    dump_layers.clear();
                }
                else
                {
                    dump_all_layers = false;
                    dump_layers.clear();
                    std::istringstream iss(layers_str);
                    std::string token;
                    while (std::getline(iss, token, ','))
                    {
                        token.erase(0, token.find_first_not_of(" \t"));
                        token.erase(token.find_last_not_of(" \t") + 1);
                        if (!token.empty())
                        {
                            dump_layers.insert(std::atoi(token.c_str()));
                        }
                    }
                }
            }

            const char *iterations_env = std::getenv("LLAMINAR_STAGE_DUMP_ITERATION");
            if (iterations_env)
            {
                std::string iter_str(iterations_env);
                if (iter_str == "all")
                {
                    dump_all_iterations = true;
                    dump_iterations.clear();
                }
                else
                {
                    dump_all_iterations = false;
                    dump_iterations.clear();
                    std::istringstream iss(iter_str);
                    std::string token;
                    while (std::getline(iss, token, ','))
                    {
                        token.erase(0, token.find_first_not_of(" \t"));
                        token.erase(token.find_last_not_of(" \t") + 1);
                        if (!token.empty())
                        {
                            dump_iterations.insert(std::atoi(token.c_str()));
                        }
                    }
                }
            }

            const char *rank_env = std::getenv("LLAMINAR_STAGE_DUMP_RANK");
            if (rank_env)
            {
                dump_rank = std::atoi(rank_env);
            }

            const char *max_env = std::getenv("LLAMINAR_STAGE_DUMP_MAX");
            if (max_env)
            {
                max_dumps_per_type = std::atoi(max_env);
            }

            const char *inputs_env = std::getenv("LLAMINAR_STAGE_DUMP_INPUTS");
            if (inputs_env)
            {
                dump_inputs = (std::atoi(inputs_env) != 0);
            }

            const char *outputs_env = std::getenv("LLAMINAR_STAGE_DUMP_OUTPUTS");
            if (outputs_env)
            {
                dump_outputs = (std::atoi(outputs_env) != 0);
            }

            const char *weights_env = std::getenv("LLAMINAR_STAGE_DUMP_WEIGHTS");
            if (weights_env)
            {
                dump_weights = (std::atoi(weights_env) != 0);
            }

            const char *async_env = std::getenv("LLAMINAR_STAGE_DUMP_ASYNC");
            if (async_env)
            {
                async_dump = (std::atoi(async_env) != 0);
            }

            const char *async_threads_env = std::getenv("LLAMINAR_STAGE_DUMP_ASYNC_THREADS");
            if (async_threads_env)
            {
                async_threads = std::atoi(async_threads_env);
                if (async_threads < 1)
                    async_threads = 1;
                if (async_threads > 16)
                    async_threads = 16;
            }
        }

        /**
         * @brief Check if a stage type should be dumped
         * @param type_name Stage type name (e.g., "GEMM", "ATTENTION")
         * @return true if this stage type should be dumped
         */
        bool shouldDumpType(const std::string &type_name) const
        {
            return dump_all_types || dump_types.count(type_name) > 0;
        }

        /**
         * @brief Check if a specific layer should be dumped
         * @param layer_idx Layer index (-1 for non-layer stages)
         * @return true if this layer should be dumped
         */
        bool shouldDumpLayer(int layer_idx) const
        {
            if (layer_idx < 0)
                return true; // Non-layer stages always pass layer filter
            return dump_all_layers || dump_layers.count(layer_idx) > 0;
        }

        /**
         * @brief Check if a specific iteration should be dumped
         * @param iteration Decode iteration index (-1 for prefill)
         * @return true if this iteration should be dumped
         */
        bool shouldDumpIteration(int iteration) const
        {
            if (iteration < 0)
                return true; // Prefill always passes iteration filter
            return dump_all_iterations || dump_iterations.count(iteration) > 0;
        }

        /**
         * @brief Check if this MPI rank should perform dumps
         * @param rank Current MPI rank
         * @return true if this rank should dump
         */
        bool shouldDumpRank(int rank) const
        {
            return dump_rank == -1 || dump_rank == rank;
        }

        /**
         * @brief Check if a specific stage name should be dumped
         *
         * Uses **substring matching** for flexibility:
         * - "attn_norm" matches "layer0_attn_norm", "layer5_attn_norm", etc.
         * - "layer0" matches all layer0 stages
         * - Exact match also works: "layer0_attn_norm" matches "layer0_attn_norm"
         *
         * @param stage_name Stage node name (e.g., "layer0_attention", "prefill_attention")
         * @return true if this stage name should be dumped
         */
        bool shouldDumpName(const std::string &stage_name) const
        {
            if (dump_all_names)
                return true;

            // Substring matching: any filter string that appears in stage_name matches
            for (const auto &filter : dump_names)
            {
                if (stage_name.find(filter) != std::string::npos)
                    return true;
            }
            return false;
        }

        /**
         * @brief Full filter: check if a stage execution should be dumped
         * @param type_name Stage type name (e.g., "GEMM", "ATTENTION")
         * @param stage_name Stage node name (e.g., "layer0_attention")
         * @param layer_idx Layer index (-1 for non-layer stages)
         * @param iteration Decode iteration (-1 for prefill)
         * @param rank MPI rank
         * @return true if this stage execution should be dumped
         */
        bool shouldDump(const std::string &type_name, const std::string &stage_name,
                        int layer_idx, int iteration, int rank) const
        {
            if (!enabled)
                return false;
            return shouldDumpType(type_name) &&
                   shouldDumpName(stage_name) &&
                   shouldDumpLayer(layer_idx) &&
                   shouldDumpIteration(iteration) &&
                   shouldDumpRank(rank);
        }

        /**
         * @brief Legacy full filter (without stage name)
         * @deprecated Use the overload with stage_name parameter
         */
        bool shouldDump(const std::string &type_name, int layer_idx, int iteration, int rank) const
        {
            return shouldDump(type_name, "", layer_idx, iteration, rank);
        }
    };

    /**
     * @brief MPI collective operation logging configuration
     *
     * Controls diagnostic logging for MPI operations like AllReduce, AllGather, etc.
     * Useful for debugging tensor parallelism issues.
     *
     * **Environment Variables**:
     * - `LLAMINAR_MPI_LOG_COLLECTIVES`: Enable logging of MPI collective operations (0/1)
     * - `LLAMINAR_MPI_LOG_TIMING`: Enable timing of MPI operations (0/1)
     * - `LLAMINAR_MPI_VERIFY_CHECKSUMS`: Enable checksum verification before/after MPI ops (0/1)
     *
     * **Usage**:
     * @code
     *   # Log all MPI collectives with timing
     *   LLAMINAR_MPI_LOG_COLLECTIVES=1 LLAMINAR_MPI_LOG_TIMING=1 \
     *   mpirun -np 2 ./run_llaminar.sh -m model.gguf -p "test"
     *
     *   # Enable checksum verification (slow, for debugging)
     *   LLAMINAR_MPI_VERIFY_CHECKSUMS=1 mpirun -np 2 ...
     * @endcode
     */
    struct MPILoggingConfig
    {
        bool log_collectives = false;  ///< Log MPI collective operations (start/end)
        bool log_timing = false;       ///< Log timing of MPI operations
        bool verify_checksums = false; ///< Verify checksums before/after MPI ops (slow)
        int recv_timeout_ms = 0;       ///< MPI_Recv timeout warning threshold (0 = disabled)

        MPILoggingConfig()
        {
            reload();
        }

        void reload()
        {
            const char *log_collectives_env = std::getenv("LLAMINAR_MPI_LOG_COLLECTIVES");
            if (log_collectives_env)
            {
                log_collectives = (std::atoi(log_collectives_env) != 0);
            }

            const char *log_timing_env = std::getenv("LLAMINAR_MPI_LOG_TIMING");
            if (log_timing_env)
            {
                log_timing = (std::atoi(log_timing_env) != 0);
            }

            const char *verify_checksums_env = std::getenv("LLAMINAR_MPI_VERIFY_CHECKSUMS");
            if (verify_checksums_env)
            {
                verify_checksums = (std::atoi(verify_checksums_env) != 0);
            }

            const char *recv_timeout_env = std::getenv("LLAMINAR_MPI_RECV_TIMEOUT_MS");
            if (recv_timeout_env)
            {
                recv_timeout_ms = std::atoi(recv_timeout_env);
            }
        }
    };

    /**
     * @brief Stage output debug print configuration
     *
     * Enables printing first N elements of stage outputs AFTER coherence is handled,
     * ensuring proper GPU→host synchronization before reading data.
     *
     * Environment Variables:
     *   LLAMINAR_STAGE_OUTPUT_PRINT=1       - Enable output printing
     *   LLAMINAR_STAGE_OUTPUT_PRINT_N=8     - Number of elements to print per row
     *   LLAMINAR_STAGE_OUTPUT_PRINT_ROWS=2  - Number of rows to print (first + last)
     *   LLAMINAR_STAGE_OUTPUT_PRINT_STAGES  - Comma-separated stage names to print
     *
     * Usage:
     * @code
     *   # Print first 8 elements of all stage outputs
     *   LLAMINAR_STAGE_OUTPUT_PRINT=1 ./build_v2/llaminar2 ...
     *
     *   # Print only LM_HEAD stage output
     *   LLAMINAR_STAGE_OUTPUT_PRINT=1 LLAMINAR_STAGE_OUTPUT_PRINT_STAGES=lm_head ./build_v2/llaminar2 ...
     * @endcode
     */
    struct StageOutputPrintConfig
    {
        bool enabled = false;         ///< Enable stage output printing
        int num_elements = 8;         ///< Number of elements to print per row
        int num_rows = 2;             ///< Number of rows to print (first N and last)
        std::set<std::string> stages; ///< Specific stages to print (empty = all)

        StageOutputPrintConfig()
        {
            reload();
        }

        void reload()
        {
            const char *enabled_env = std::getenv("LLAMINAR_STAGE_OUTPUT_PRINT");
            if (enabled_env)
            {
                enabled = (std::atoi(enabled_env) != 0);
            }

            const char *num_elements_env = std::getenv("LLAMINAR_STAGE_OUTPUT_PRINT_N");
            if (num_elements_env)
            {
                num_elements = std::atoi(num_elements_env);
                if (num_elements < 1)
                    num_elements = 1;
            }

            const char *num_rows_env = std::getenv("LLAMINAR_STAGE_OUTPUT_PRINT_ROWS");
            if (num_rows_env)
            {
                num_rows = std::atoi(num_rows_env);
                if (num_rows < 1)
                    num_rows = 1;
            }

            const char *stages_env = std::getenv("LLAMINAR_STAGE_OUTPUT_PRINT_STAGES");
            if (stages_env)
            {
                std::string stages_str(stages_env);
                std::stringstream ss(stages_str);
                std::string stage;
                while (std::getline(ss, stage, ','))
                {
                    // Trim whitespace and convert to lowercase
                    stage.erase(0, stage.find_first_not_of(" \t"));
                    stage.erase(stage.find_last_not_of(" \t") + 1);
                    std::transform(stage.begin(), stage.end(), stage.begin(), ::tolower);
                    if (!stage.empty())
                    {
                        stages.insert(stage);
                    }
                }
            }
        }

        /// Check if a stage should have its output printed
        bool shouldPrint(const std::string &stage_name) const
        {
            if (!enabled)
                return false;
            if (stages.empty())
                return true; // Print all stages if no filter specified

            // Convert stage name to lowercase for comparison
            std::string lower_name = stage_name;
            std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

            // Check for substring match (e.g., "lm_head" matches "layer23_lm_head")
            for (const auto &pattern : stages)
            {
                if (lower_name.find(pattern) != std::string::npos)
                {
                    return true;
                }
            }
            return false;
        }
    };

    /**
     * @brief Buffer validation configuration
     *
     * Controls runtime validation of tensor buffers to catch uninitialized
     * or corrupted data early. Part of the Buffer Contract Validation System.
     *
     * **Environment Variables**:
     * - `LLAMINAR_VALIDATE_BUFFERS`: Enable buffer validation after stage execution (0/1)
     * - `LLAMINAR_FAIL_ON_ZERO`: Fail immediately when zero tensor detected (0/1)
     * - `LLAMINAR_FAIL_ON_NAN`: Fail immediately when NaN/Inf detected (0/1)
     *
     * **Automatic Enablement**:
     * When `LLAMINAR_ASSERTIONS_ACTIVE` is defined (Debug/Integration builds),
     * `validate_buffers` defaults to true. Set `LLAMINAR_VALIDATE_BUFFERS=0` to
     * explicitly disable in these builds.
     *
     * **Usage**:
     * @code
     *   # Buffer validation is automatic in Debug/Integration builds
     *   ./build_v2/llaminar2 -m model.gguf -p "test"
     *
     *   # Strict mode: fail on any zero or NaN tensor
     *   LLAMINAR_FAIL_ON_ZERO=1 LLAMINAR_FAIL_ON_NAN=1 ./build_v2/llaminar2 ...
     *
     *   # Disable validation even in debug builds
     *   LLAMINAR_VALIDATE_BUFFERS=0 ./build_v2/llaminar2 ...
     * @endcode
     *
     * @note Validation only compiles in when LLAMINAR_ASSERTIONS_ACTIVE is defined
     * @see Assertions.h for LLAMINAR_ASSERTIONS_ACTIVE definition
     */
    struct ValidationConfig
    {
        bool validate_buffers = false;                    ///< Enable buffer validation after stage execution
        bool validate_inputs = false;                     ///< Enable input validation BEFORE stage execution (Phase 1)
        bool validate_gpu_ptrs = false;                   ///< Validate GPU pointer device ownership before stage execution
        bool sync_each_stage = false;                     ///< Force hipDeviceSynchronize after each stage (ROCm debug)
        bool sync_after_embedding_stage = false;          ///< Force stream sync only after embedding kernel launch (targeted race debug)
        bool sync_local_tp_allreduce = false;             ///< Force device sync before/after LOCAL TP allreduce (targeted debug)
        bool serialize_embedding_stage = false;           ///< Serialize embedding stage execution across LOCAL TP device threads (targeted race debug)
        bool serialize_rocm_gemm_stage = false;           ///< Serialize ROCm CK GEMM dispatch across LOCAL TP device threads (targeted race debug)
        bool strict_local_tp_stage_barrier = false;       ///< Fail fast when LOCAL TP barrier receives mixed stage names in one generation
        bool serialize_local_tp_allreduce_launch = false; ///< Serialize LOCAL TP collective launch section across threads (targeted race debug)
        bool trace_local_tp_pointer = false;              ///< Enable targeted pointer watch for LOCAL TP allreduce buffers
        uint64_t trace_local_tp_pointer_address = 0;      ///< Watched address for LOCAL TP pointer tracing (env accepts hex/decimal)
        bool fail_on_zero = false;                        ///< Fail immediately when all-zero OUTPUT tensor detected (auto-enabled in Debug)
        bool fail_on_nan = false;                         ///< Fail immediately when NaN/Inf detected
        bool dump_on_failure = true;                      ///< Dump all buffers to disk when verification fails
        int sample_rows = 8;                              ///< Number of rows to sample for verification (efficiency)

        ValidationConfig()
        {
            reload();
        }

        void reload()
        {
            // Auto-enable buffer validation when assertions are active
            // This is the default for Debug and Integration builds
#if LLAMINAR_ASSERTIONS_ACTIVE
            validate_buffers = true;
            validate_inputs = true; // Enable input validation in debug builds
            fail_on_nan = true;     // NaN/Inf is always a bug, fail by default
            fail_on_zero = true;    // All-zero OUTPUTS are almost always bugs
#endif

            // Environment variables can override the defaults
            const char *validate_env = std::getenv("LLAMINAR_VALIDATE_BUFFERS");
            if (validate_env)
            {
                validate_buffers = (std::atoi(validate_env) != 0);
            }

            const char *validate_inputs_env = std::getenv("LLAMINAR_VALIDATE_INPUTS");
            if (validate_inputs_env)
            {
                validate_inputs = (std::atoi(validate_inputs_env) != 0);
            }

            const char *fail_zero_env = std::getenv("LLAMINAR_FAIL_ON_ZERO");
            if (fail_zero_env)
            {
                fail_on_zero = (std::atoi(fail_zero_env) != 0);
            }

            const char *fail_nan_env = std::getenv("LLAMINAR_FAIL_ON_NAN");
            if (fail_nan_env)
            {
                fail_on_nan = (std::atoi(fail_nan_env) != 0);
            }

            const char *dump_env = std::getenv("LLAMINAR_DUMP_ON_FAILURE");
            if (dump_env)
            {
                dump_on_failure = (std::atoi(dump_env) != 0);
            }

            const char *sample_rows_env = std::getenv("LLAMINAR_VALIDATION_SAMPLE_ROWS");
            if (sample_rows_env)
            {
                sample_rows = std::atoi(sample_rows_env);
                if (sample_rows < 1)
                    sample_rows = 1;
            }

            const char *gpu_ptrs_env = std::getenv("LLAMINAR_VALIDATE_GPU_PTRS");
            if (gpu_ptrs_env)
            {
                validate_gpu_ptrs = (std::atoi(gpu_ptrs_env) != 0);
            }

            const char *sync_stage_env = std::getenv("LLAMINAR_SYNC_EACH_STAGE");
            if (sync_stage_env)
            {
                sync_each_stage = (std::atoi(sync_stage_env) != 0);
            }

            const char *sync_embed_env = std::getenv("LLAMINAR_SYNC_AFTER_EMBEDDING_STAGE");
            if (sync_embed_env)
            {
                sync_after_embedding_stage = (std::atoi(sync_embed_env) != 0);
            }

            const char *sync_local_tp_env = std::getenv("LLAMINAR_SYNC_LOCAL_TP_ALLREDUCE");
            if (sync_local_tp_env)
            {
                sync_local_tp_allreduce = (std::atoi(sync_local_tp_env) != 0);
            }

            const char *serialize_embed_env = std::getenv("LLAMINAR_SERIALIZE_EMBEDDING_STAGE");
            if (serialize_embed_env)
            {
                serialize_embedding_stage = (std::atoi(serialize_embed_env) != 0);
            }

            const char *serialize_rocm_gemm_env = std::getenv("LLAMINAR_SERIALIZE_ROCM_GEMM_STAGE");
            if (serialize_rocm_gemm_env)
            {
                serialize_rocm_gemm_stage = (std::atoi(serialize_rocm_gemm_env) != 0);
            }

            const char *strict_local_tp_stage_env = std::getenv("LLAMINAR_STRICT_LOCAL_TP_STAGE_BARRIER");
            if (strict_local_tp_stage_env)
            {
                strict_local_tp_stage_barrier = (std::atoi(strict_local_tp_stage_env) != 0);
            }

            const char *serialize_local_tp_launch_env = std::getenv("LLAMINAR_SERIALIZE_LOCAL_TP_ALLREDUCE_LAUNCH");
            if (serialize_local_tp_launch_env)
            {
                serialize_local_tp_allreduce_launch = (std::atoi(serialize_local_tp_launch_env) != 0);
            }

            const char *trace_local_tp_ptr_env = std::getenv("LLAMINAR_TRACE_LOCAL_TP_PTR");
            if (trace_local_tp_ptr_env && trace_local_tp_ptr_env[0] != '\0')
            {
                char *end_ptr = nullptr;
                const unsigned long long parsed = std::strtoull(trace_local_tp_ptr_env, &end_ptr, 0);
                if (end_ptr != trace_local_tp_ptr_env)
                {
                    trace_local_tp_pointer = true;
                    trace_local_tp_pointer_address = static_cast<uint64_t>(parsed);
                }
            }
        }
    };

    /**
     * @brief Weight streaming configuration (Option B)
     *
     * Controls the weight streaming subsystem for running models that exceed
     * GPU memory capacity. When enabled, layer weights are streamed on-demand
     * from host memory with configurable prefetching and eviction policies.
     *
     * **Environment Variables**:
     * - `LLAMINAR_WEIGHT_STREAMING`: Enable weight streaming mode (0/1, default: 0)
     * - `LLAMINAR_STREAM_MEMORY_MB`: GPU memory budget for weights in MB (0 = auto)
     * - `LLAMINAR_STREAM_PREFETCH_DEPTH`: Number of layers to prefetch ahead (default: 1)
     * - `LLAMINAR_STREAM_EVICTION_POLICY`: Eviction policy: "lru", "fifo", "none" (default: "lru")
     * - `LLAMINAR_STREAM_VERBOSE`: Enable verbose streaming logs (0/1, default: 0)
     *
     * **Usage**:
     * @code
     *   # Enable weight streaming with 4GB GPU budget
     *   LLAMINAR_WEIGHT_STREAMING=1 LLAMINAR_STREAM_MEMORY_MB=4096 \
     *   ./build_v2_release/llaminar2 -m large_model.gguf -p "Hello"
     *
     *   # Enable with aggressive prefetching
     *   LLAMINAR_WEIGHT_STREAMING=1 LLAMINAR_STREAM_PREFETCH_DEPTH=2 \
     *   ./build_v2_release/llaminar2 -m large_model.gguf -p "Hello"
     *
     *   # Debug streaming with verbose logs
     *   LLAMINAR_WEIGHT_STREAMING=1 LLAMINAR_STREAM_VERBOSE=1 \
     *   ./build_v2/llaminar2 -m model.gguf -p "test"
     * @endcode
     *
     * @see IWeightStreamer.h for the streaming interface
     * @see LayerWeightStreamer.h for the implementation
     * @see docs/v2/OPTION_B_WEIGHT_STREAMING_DESIGN.md for design details
     */
    struct StreamingEnv
    {
        bool enabled = false;                ///< Enable weight streaming (LLAMINAR_WEIGHT_STREAMING)
        size_t memory_budget_mb = 0;         ///< GPU memory budget in MB (0 = auto-detect)
        int prefetch_depth = 1;              ///< Layers to prefetch ahead (default: 1)
        std::string eviction_policy = "lru"; ///< Eviction policy: lru, fifo, none
        bool verbose = false;                ///< Enable verbose streaming logs

        StreamingEnv()
        {
            reload();
        }

        void reload()
        {
            // Reset to defaults first (enables proper re-reading when env vars are unset)
            enabled = false;
            memory_budget_mb = 0;
            prefetch_depth = 1;
            eviction_policy = "lru";
            verbose = false;

            const char *enabled_env = std::getenv("LLAMINAR_WEIGHT_STREAMING");
            if (enabled_env)
            {
                enabled = (std::atoi(enabled_env) != 0);
            }

            const char *memory_env = std::getenv("LLAMINAR_STREAM_MEMORY_MB");
            if (memory_env)
            {
                memory_budget_mb = static_cast<size_t>(std::atol(memory_env));
            }

            const char *prefetch_env = std::getenv("LLAMINAR_STREAM_PREFETCH_DEPTH");
            if (prefetch_env)
            {
                prefetch_depth = std::atoi(prefetch_env);
                if (prefetch_depth < 0)
                    prefetch_depth = 0;
            }

            const char *policy_env = std::getenv("LLAMINAR_STREAM_EVICTION_POLICY");
            if (policy_env)
            {
                std::string policy(policy_env);
                // Normalize to lowercase
                std::transform(policy.begin(), policy.end(), policy.begin(), ::tolower);
                // Validate policy
                if (policy == "lru" || policy == "fifo" || policy == "none")
                {
                    eviction_policy = policy;
                }
                // Invalid values keep the default "lru"
            }

            const char *verbose_env = std::getenv("LLAMINAR_STREAM_VERBOSE");
            if (verbose_env)
            {
                verbose = (std::atoi(verbose_env) != 0);
            }
        }
    };

    /**
     * @brief Memory transfer tracing configuration
     *
     * Provides diagnostic instrumentation for GPU memory transfers (H2D and D2H).
     * When enabled, logs transfers with stack traces to identify wasteful transfers.
     *
     * Environment variables:
     * - `LLAMINAR_TRACE_TRANSFERS=1` - Enable transfer tracing (logs all H2D/D2H)
     * - `LLAMINAR_TRACE_TRANSFERS_STACKTRACE=1` - Include C++23 stacktrace in log
     * - `LLAMINAR_TRACE_TRANSFERS_THROW=1` - Throw exception with stacktrace for unexpected transfers
     * - `LLAMINAR_TRACE_TRANSFERS_MIN_BYTES=0` - Minimum transfer size to trace (default: 0 = all)
     * - `LLAMINAR_TRACE_TRANSFERS_ONLY_D2H=1` - Only trace D2H transfers (not H2D)
     *
     * @code
     *   # Log all D2H transfers larger than 1MB with stack traces
     *   LLAMINAR_TRACE_TRANSFERS=1 \
     *   LLAMINAR_TRACE_TRANSFERS_STACKTRACE=1 \
     *   LLAMINAR_TRACE_TRANSFERS_ONLY_D2H=1 \
     *   LLAMINAR_TRACE_TRANSFERS_MIN_BYTES=1048576 \
     *   ./build_v2_release/llaminar2 -m model.gguf -p "test"
     * @endcode
     */
    struct TransferTracingConfig
    {
        bool enabled = false;            ///< Master enable for transfer tracing
        bool include_stacktrace = false; ///< Include C++23 stacktrace in log output
        bool throw_on_transfer = false;  ///< Throw exception with stacktrace for unexpected transfers
        size_t min_bytes = 0;            ///< Minimum transfer size to trace (0 = all)
        bool only_d2h = false;           ///< Only trace D2H transfers (D2H is more suspicious)

        // Transfer counters for testing (mutable to allow incrementing from const context)
        mutable std::atomic<size_t> h2d_count{0}; ///< Count of H2D transfers matching filter
        mutable std::atomic<size_t> d2h_count{0}; ///< Count of D2H transfers matching filter
        mutable std::atomic<size_t> h2d_bytes{0}; ///< Total H2D bytes transferred
        mutable std::atomic<size_t> d2h_bytes{0}; ///< Total D2H bytes transferred

        /// Reset transfer counters (call before test runs)
        void resetCounters() const
        {
            h2d_count.store(0, std::memory_order_relaxed);
            d2h_count.store(0, std::memory_order_relaxed);
            h2d_bytes.store(0, std::memory_order_relaxed);
            d2h_bytes.store(0, std::memory_order_relaxed);
        }

        /// Record an H2D transfer
        void recordH2D(size_t bytes) const
        {
            h2d_count.fetch_add(1, std::memory_order_relaxed);
            h2d_bytes.fetch_add(bytes, std::memory_order_relaxed);
        }

        /// Record a D2H transfer
        void recordD2H(size_t bytes) const
        {
            d2h_count.fetch_add(1, std::memory_order_relaxed);
            d2h_bytes.fetch_add(bytes, std::memory_order_relaxed);
        }

        TransferTracingConfig()
        {
            reload();
        }

        void reload()
        {
            enabled = false;
            include_stacktrace = false;
            throw_on_transfer = false;
            min_bytes = 0;
            only_d2h = false;

            const char *trace_env = std::getenv("LLAMINAR_TRACE_TRANSFERS");
            if (trace_env)
            {
                enabled = (std::atoi(trace_env) != 0);
            }

            const char *stack_env = std::getenv("LLAMINAR_TRACE_TRANSFERS_STACKTRACE");
            if (stack_env)
            {
                include_stacktrace = (std::atoi(stack_env) != 0);
            }

            const char *throw_env = std::getenv("LLAMINAR_TRACE_TRANSFERS_THROW");
            if (throw_env)
            {
                throw_on_transfer = (std::atoi(throw_env) != 0);
            }

            const char *min_env = std::getenv("LLAMINAR_TRACE_TRANSFERS_MIN_BYTES");
            if (min_env)
            {
                min_bytes = static_cast<size_t>(std::atoll(min_env));
            }

            const char *only_d2h_env = std::getenv("LLAMINAR_TRACE_TRANSFERS_ONLY_D2H");
            if (only_d2h_env)
            {
                only_d2h = (std::atoi(only_d2h_env) != 0);
            }
        }
    };

    /**
     * @brief Logger environment configuration
     */
    struct LoggerConfig
    {
        std::string log_level; ///< LLAMINAR_LOG_LEVEL (empty if unset)
        int buffer_lines = 0;  ///< LLAMINAR_LOG_BUFFER_LINES (0 if unset/invalid)

        LoggerConfig()
        {
            reload();
        }

        void reload()
        {
            log_level.clear();
            buffer_lines = 0;

            const char *level_env = std::getenv("LLAMINAR_LOG_LEVEL");
            if (level_env && *level_env)
            {
                log_level = level_env;
            }

            const char *buffer_env = std::getenv("LLAMINAR_LOG_BUFFER_LINES");
            if (buffer_env)
            {
                int lines = std::atoi(buffer_env);
                if (lines > 0)
                {
                    buffer_lines = lines;
                }
            }
        }
    };

    /**
     * @brief Topology-related environment configuration
     */
    struct TopologyEnvConfig
    {
        std::string cuda_visible_devices; ///< CUDA_VISIBLE_DEVICES
        std::string hip_visible_devices;  ///< HIP_VISIBLE_DEVICES

        TopologyEnvConfig()
        {
            reload();
        }

        void reload()
        {
            cuda_visible_devices.clear();
            hip_visible_devices.clear();

            const char *cuda_visible = std::getenv("CUDA_VISIBLE_DEVICES");
            if (cuda_visible && *cuda_visible)
            {
                cuda_visible_devices = cuda_visible;
            }

            const char *hip_visible = std::getenv("HIP_VISIBLE_DEVICES");
            if (hip_visible && *hip_visible)
            {
                hip_visible_devices = hip_visible;
            }
        }
    };

    /**
     * @brief MPI bootstrap environment snapshot
     */
    struct MPIBootstrapEnvConfig
    {
        std::string ompi_comm_world_size;
        std::string ompi_comm_world_rank;
        std::string pmi_size;
        std::string pmi_rank;
        std::string slurm_ntasks;
        std::string slurm_procid;
        std::string mpi_localrankid;
        std::string ompi_comm_world_local_rank;
        std::string ompi_mca_btl_vader_single_copy_mechanism;

        MPIBootstrapEnvConfig()
        {
            reload();
        }

        void reload()
        {
            auto read_env = [](const char *name) -> std::string
            {
                const char *val = std::getenv(name);
                if (val && *val)
                {
                    return std::string(val);
                }
                return "";
            };

            ompi_comm_world_size = read_env("OMPI_COMM_WORLD_SIZE");
            ompi_comm_world_rank = read_env("OMPI_COMM_WORLD_RANK");
            pmi_size = read_env("PMI_SIZE");
            pmi_rank = read_env("PMI_RANK");
            slurm_ntasks = read_env("SLURM_NTASKS");
            slurm_procid = read_env("SLURM_PROCID");
            mpi_localrankid = read_env("MPI_LOCALRANKID");
            ompi_comm_world_local_rank = read_env("OMPI_COMM_WORLD_LOCAL_RANK");
            ompi_mca_btl_vader_single_copy_mechanism = read_env("OMPI_MCA_btl_vader_single_copy_mechanism");
        }

        std::optional<std::string> get(const char *name) const
        {
            const auto value = getRef(name);
            if (!value.empty())
            {
                return value;
            }
            return std::nullopt;
        }

    private:
        const std::string &getRef(const char *name) const
        {
            if (std::strcmp(name, "OMPI_COMM_WORLD_SIZE") == 0)
                return ompi_comm_world_size;
            if (std::strcmp(name, "OMPI_COMM_WORLD_RANK") == 0)
                return ompi_comm_world_rank;
            if (std::strcmp(name, "PMI_SIZE") == 0)
                return pmi_size;
            if (std::strcmp(name, "PMI_RANK") == 0)
                return pmi_rank;
            if (std::strcmp(name, "SLURM_NTASKS") == 0)
                return slurm_ntasks;
            if (std::strcmp(name, "SLURM_PROCID") == 0)
                return slurm_procid;
            if (std::strcmp(name, "MPI_LOCALRANKID") == 0)
                return mpi_localrankid;
            if (std::strcmp(name, "OMPI_COMM_WORLD_LOCAL_RANK") == 0)
                return ompi_comm_world_local_rank;
            if (std::strcmp(name, "OMPI_MCA_btl_vader_single_copy_mechanism") == 0)
                return ompi_mca_btl_vader_single_copy_mechanism;
            static const std::string empty;
            return empty;
        }
    };

    /**
     * @brief ROCm-specific debugging and instrumentation configuration
     *
     * Environment variables:
     * - `LLAMINAR_ROCM_TRACE_COHERENCE=1` - Enable detailed coherence timing logs
     * - `LLAMINAR_ROCM_TRACE_KERNELS=1` - Enable per-kernel timing breakdown
     * - `LLAMINAR_ROCM_SYNC_AFTER_KERNEL=1` - Force hipDeviceSynchronize after each kernel
     * - `LLAMINAR_ROCM_GEMV_LAYOUT=vnni` - Use VNNI-packed weights for GEMV when available
     * - `LLAMINAR_ROCM_PACK_VNNI_ONLY=1` - Prefer VNNI-only host packed buffers (drop row-major host copy when safe)
     * - `LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR=1` - Enable INT8 prefill grid-kpar split-K variant
     * - `LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR_SPLITS=<n>` - Split-K slice count for grid-kpar prefill variant (`0` = auto)
     * - `LLAMINAR_ROCM_VNNI_PREFILL_CPT=<n>` - Outputs-per-thread for INT8 prefill kernels (`1`, `2`, or `4`)
     * - `LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR_KB=<n>` - Grid-kpar K-block count override (`0` = auto policy)
     * - `LLAMINAR_ROCM_VNNI_PREFILL_VARIANT=<id>` - Force baseline INT8 prefill tile variant (`-1` auto, `0` 16x16, `1` 32x8, `2` 8x32, `3` 8x8)
     * - `LLAMINAR_ROCM_VNNI_PREFILL_GRID_VARIANT=<id>` - Force grid-kpar INT8 prefill tile variant (`-1` auto, `0` 16x16, `1` 32x8, `2` 8x32, `3` 8x8)
     * - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE=1` - Enable FFN-wide policy override for production dispatch path
     * - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_KPAR=<n>` - FFN override grid-kpar mode (`-1` policy default, `0` baseline, `1` grid-kpar)
     * - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_SPLITS=<n>` - FFN override split-K slices (`0` policy default)
     * - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_CPT=<n>` - FFN override outputs-per-thread (`0` policy default, `1`,`2`,`4` valid)
     * - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_VARIANT=<id>` - FFN override tile variant (`-1` policy default, `0` 16x16, `1` 32x8, `2` 8x32, `3` 8x8)
     * - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_KERNEL_BODY=<id>` - FFN override kernel-body variant (`0` baseline loop, `1` software-pipelined loop, `2` LDS B-tile + pipelined loop)
     * - `LLAMINAR_ROCM_VNNI_PREFILL_GRID_SWIZZLE=<id>` - Global grid-kpar swizzle variant (`-1` policy default, `0` unswizzled, `1` swizzled)
     * - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_SWIZZLE=<id>` - FFN override grid-kpar swizzle variant (`-1` policy default, `0` unswizzled, `1` swizzled)
     * - `LLAMINAR_ROCM_RATIO_PREFILL_VARIANT=<id>` - Force ratio-prefill tile variant (`-1` auto, `0` 16x16, `1` 32x8, `2` 8x32, `3` 8x8)
     * - `LLAMINAR_ROCM_RATIO_PREFILL_KB=<n>` - Ratio-prefill split-K blocks (`0` = auto)
     * - `LLAMINAR_ROCM_RATIO_PREFILL_LINEAR_VARIANT=<id>` - Force linear-codebook ratio prefill tile variant (`-1`=use global/auto)
     * - `LLAMINAR_ROCM_RATIO_PREFILL_LINEAR_KB=<n>` - Force linear-codebook ratio prefill split-K blocks (`0`=use global/auto)
     * - `LLAMINAR_ROCM_RATIO_PREFILL_IQ4_VARIANT=<id>` - Force IQ4-codebook ratio prefill tile variant (`-1`=use global/auto)
     * - `LLAMINAR_ROCM_RATIO_PREFILL_IQ4_KB=<n>` - Force IQ4-codebook ratio prefill split-K blocks (`0`=use global/auto)
     * - `LLAMINAR_ROCM_REPACK_SLOTS=<n>` - Ring-buffer slot count for startup GPU repack pipeline
     * - `LLAMINAR_ROCM_REPACK_BUDGET_MB=<mb>` - VRAM budget cap for startup GPU repack staging buffers
     * - `LLAMINAR_ROCM_REPACK_STREAMS=<n>` - Stream count hint for startup GPU repack pipeline
     *
     * @code
     *   LLAMINAR_ROCM_TRACE_COHERENCE=1 \
     *   ./build_v2_e2e_release/llaminar2 -m model.gguf -p "test"
     * @endcode
     */
    struct ROCmConfig
    {
        bool trace_coherence = false;           ///< Log detailed coherence timings (LLAMINAR_ROCM_TRACE_COHERENCE)
        bool trace_kernels = false;             ///< Log per-kernel timing breakdown (LLAMINAR_ROCM_TRACE_KERNELS)
        bool sync_after_kernel = false;         ///< Force sync after each kernel (LLAMINAR_ROCM_SYNC_AFTER_KERNEL)
        std::string gemv_mode = "fp32";         ///< GEMV mode: fp32 (default), fp16, int8
        std::string gemv_layout = "row";        ///< GEMV weight layout: row (default), vnni
        bool pack_vnni_only_host = false;       ///< Prefer VNNI-only host packed buffers (LLAMINAR_ROCM_PACK_VNNI_ONLY)
        bool vnni_prefill_grid_kpar = false;    ///< Enable INT8 prefill grid-kpar split-K variant (LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR)
        int vnni_prefill_grid_kpar_splits = 0;  ///< Split-K slices for grid-kpar variant (`0` = auto)
        int vnni_prefill_cpt = 1;               ///< Outputs-per-thread for INT8 prefill kernels (1,2,4)
        int vnni_prefill_grid_kpar_kb = 0;      ///< Grid-kpar K-block count override (0=auto)
        int vnni_prefill_variant = -1;          ///< Baseline prefill tile variant override (-1=auto,0=16x16,1=32x8,2=8x32,3=8x8)
        int vnni_prefill_grid_variant = -1;     ///< Grid-kpar prefill tile variant override (-1=auto,0=16x16,1=32x8,2=8x32,3=8x8)
        bool vnni_prefill_ffn_override = false; ///< Enable FFN-wide policy override for INT8 prefill
        int vnni_prefill_ffn_override_grid_kpar = -1;
        int vnni_prefill_ffn_override_splits = 0;
        int vnni_prefill_ffn_override_cpt = 0;
        int vnni_prefill_ffn_override_variant = -1;
        int vnni_prefill_ffn_override_kernel_body = 2;
        int vnni_prefill_grid_swizzle = -1;
        int vnni_prefill_ffn_override_grid_swizzle = 1;
        bool wide_tile_v3 = false;              ///< Use V3 wide-tile kernel (LDS double-buffered, N64) (LLAMINAR_ROCM_WIDE_TILE_V3)
        bool wide_tile_v7 = false;              ///< Use V7 wide-tile kernel (safe-tile split, N128) (LLAMINAR_ROCM_WIDE_TILE_V7)
        int wide_tile_kt = 16;                  ///< KT parameter for wide-tile kernels (4, 8, or 16) (LLAMINAR_ROCM_WIDE_TILE_KT)
        int blockwise_v3_mt = -1;               ///< V3 blockwise M_TILE override (-1=auto, 16/32) (LLAMINAR_ROCM_BLOCKWISE_V3_MT)
        int blockwise_v7_mt = -1;               ///< V7 blockwise M_TILE override (-1=auto, 16/32/64) (LLAMINAR_ROCM_BLOCKWISE_V7_MT)
        int blockwise_v3_unroll = -1;           ///< V3 blockwise UNROLL_KK override (-1=auto, 0=disable, 1=full, 2/4=partial) (LLAMINAR_ROCM_BLOCKWISE_V3_UNROLL)
        int blockwise_v7_unroll = -1;           ///< V7 blockwise UNROLL_KK override (-1=auto, 0=disable, 1=full, 2/4=partial) (LLAMINAR_ROCM_BLOCKWISE_V7_UNROLL)
        bool blockwise_force_v3 = false;        ///< Force V3 blockwise for all shapes (LLAMINAR_ROCM_BLOCKWISE_FORCE_V3)
        bool blockwise_force_v7 = false;        ///< Force V7 blockwise for all shapes (LLAMINAR_ROCM_BLOCKWISE_FORCE_V7)
        int blockwise_quant_variant = 0;        ///< Blockwise quant kernel variant (0=auto, 1-5=manual) (LLAMINAR_ROCM_BLOCKWISE_QUANT_VARIANT)
        int nvnni_mt = -1;                      ///< Native-VNNI M_TILE override (-1=auto, 16/32/64) (LLAMINAR_ROCM_NVNNI_MT)
        int nvnni_unroll = -1;                  ///< Native-VNNI UNROLL_G override (-1=auto, 0=none, 1=full, 2/4=partial) (LLAMINAR_ROCM_NVNNI_UNROLL)
        int nvnni_min_blocks = -1;              ///< Native-VNNI MIN_BLOCKS override (-1=auto, 1=bare, 2=2-wave, 3=3-wave) (LLAMINAR_ROCM_NVNNI_MIN_BLOCKS)
        bool nvnni_force_n64 = false;           ///< Force N64 for all native-VNNI shapes (LLAMINAR_ROCM_NVNNI_FORCE_N64)
        bool nvnni_force_n128 = false;          ///< Force N128 for all native-VNNI shapes (LLAMINAR_ROCM_NVNNI_FORCE_N128)
        int ratio_prefill_variant = -1;         ///< Ratio prefill tile variant override (-1=auto,0=16x16,1=32x8,2=8x32,3=8x8)
        int ratio_prefill_kb = 0;               ///< Ratio prefill split-K blocks override (0=auto)
        int ratio_prefill_linear_variant = -1;  ///< Linear codebook ratio prefill tile override (-1=use global/auto)
        int ratio_prefill_linear_kb = 0;        ///< Linear codebook ratio prefill split-K override (0=use global/auto)
        int ratio_prefill_iq4_variant = -1;     ///< IQ4 codebook ratio prefill tile override (-1=use global/auto)
        int ratio_prefill_iq4_kb = 0;           ///< IQ4 codebook ratio prefill split-K override (0=use global/auto)
        bool concurrent_prefill = true;         ///< Multi-stream concurrent fused GEMM projections during prefill (LLAMINAR_ROCM_CONCURRENT_PREFILL, default ON)
        bool concurrent_decode = false;         ///< Enable multi-stream concurrent fused GEMV projections during decode (LLAMINAR_ROCM_CONCURRENT_DECODE)
        bool moe_grouped_decode = true;         ///< Enable grouped MoE decode down path when supported (LLAMINAR_ROCM_MOE_GROUPED_DECODE)
        bool moe_grouped_decode_router = false; ///< Enable experimental grouped MoE decode router logits path (LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER)
        bool moe_device_routed_decode = true;   ///< Enable runtime-table device routed MoE decode (LLAMINAR_ROCM_MOE_DEVICE_ROUTED_DECODE)
        bool moe_grouped_prefill = true;        ///< Enable grouped MoE prefill path when supported (LLAMINAR_ROCM_MOE_GROUPED_PREFILL)

        // --- Startup GPU weight loading pipeline (LoadOrchestrator) ---
        int repack_slots = 3;     ///< Ring-buffer slot count for startup GPU repack pipeline (LLAMINAR_ROCM_REPACK_SLOTS)
        int repack_budget_mb = 0; ///< VRAM budget cap for startup GPU repack staging buffers, 0=unlimited (LLAMINAR_ROCM_REPACK_BUDGET_MB)
        int repack_streams = 3;   ///< H2D stream count for startup GPU repack pipeline (LLAMINAR_ROCM_REPACK_STREAMS)

        ROCmConfig()
        {
            reload();
        }

        void reload()
        {
            trace_coherence = false;
            trace_kernels = false;
            sync_after_kernel = false;
            gemv_mode = "fp32";
            gemv_layout = "row";
            pack_vnni_only_host = false;
            vnni_prefill_grid_kpar = false;
            vnni_prefill_grid_kpar_splits = 0;
            vnni_prefill_cpt = 1;
            vnni_prefill_grid_kpar_kb = 0;
            vnni_prefill_variant = -1;
            vnni_prefill_grid_variant = -1;
            vnni_prefill_ffn_override = false;
            vnni_prefill_ffn_override_grid_kpar = -1;
            vnni_prefill_ffn_override_splits = 0;
            vnni_prefill_ffn_override_cpt = 0;
            vnni_prefill_ffn_override_variant = -1;
            vnni_prefill_ffn_override_kernel_body = 2;
            vnni_prefill_grid_swizzle = -1;
            vnni_prefill_ffn_override_grid_swizzle = 1;
            wide_tile_v3 = false;
            wide_tile_v7 = false;
            wide_tile_kt = 16;
            blockwise_v3_mt = -1;
            blockwise_v7_mt = -1;
            blockwise_v3_unroll = -1;
            blockwise_v7_unroll = -1;
            blockwise_force_v3 = false;
            blockwise_force_v7 = false;
            blockwise_quant_variant = 0;
            nvnni_mt = -1;
            nvnni_unroll = -1;
            nvnni_min_blocks = -1;
            nvnni_force_n64 = false;
            nvnni_force_n128 = false;
            ratio_prefill_variant = -1;
            ratio_prefill_kb = 0;
            ratio_prefill_linear_variant = -1;
            ratio_prefill_linear_kb = 0;
            ratio_prefill_iq4_variant = -1;
            ratio_prefill_iq4_kb = 0;
            concurrent_prefill = true;
            concurrent_decode = false;
            moe_grouped_decode = true;
            moe_grouped_decode_router = false;
            moe_device_routed_decode = true;
            moe_grouped_prefill = true;
            repack_slots = 3;
            repack_budget_mb = 0;
            repack_streams = 3;

            const char *trace_coh_env = std::getenv("LLAMINAR_ROCM_TRACE_COHERENCE");
            if (trace_coh_env)
            {
                trace_coherence = (std::atoi(trace_coh_env) != 0);
            }

            const char *trace_kern_env = std::getenv("LLAMINAR_ROCM_TRACE_KERNELS");
            if (trace_kern_env)
            {
                trace_kernels = (std::atoi(trace_kern_env) != 0);
            }

            const char *sync_env = std::getenv("LLAMINAR_ROCM_SYNC_AFTER_KERNEL");
            if (sync_env)
            {
                sync_after_kernel = (std::atoi(sync_env) != 0);
            }

            const char *gemv_env = std::getenv("LLAMINAR_ROCM_GEMV_MODE");
            if (gemv_env)
            {
                gemv_mode = gemv_env;
                std::transform(gemv_mode.begin(), gemv_mode.end(), gemv_mode.begin(),
                               [](unsigned char c)
                               { return static_cast<char>(std::tolower(c)); });
            }

            const char *gemv_layout_env = std::getenv("LLAMINAR_ROCM_GEMV_LAYOUT");
            if (gemv_layout_env)
            {
                gemv_layout = gemv_layout_env;
                std::transform(gemv_layout.begin(), gemv_layout.end(), gemv_layout.begin(),
                               [](unsigned char c)
                               { return static_cast<char>(std::tolower(c)); });
            }

            const char *pack_vnni_only_env = std::getenv("LLAMINAR_ROCM_PACK_VNNI_ONLY");
            if (pack_vnni_only_env)
            {
                pack_vnni_only_host = (std::atoi(pack_vnni_only_env) != 0);
            }

            const char *vnni_prefill_grid_kpar_env = std::getenv("LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR");
            if (vnni_prefill_grid_kpar_env)
            {
                vnni_prefill_grid_kpar = (std::atoi(vnni_prefill_grid_kpar_env) != 0);
            }

            const char *vnni_prefill_grid_kpar_splits_env = std::getenv("LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR_SPLITS");
            if (vnni_prefill_grid_kpar_splits_env)
            {
                vnni_prefill_grid_kpar_splits = std::clamp(std::atoi(vnni_prefill_grid_kpar_splits_env), 0, 32);
            }

            const char *vnni_prefill_cpt_env = std::getenv("LLAMINAR_ROCM_VNNI_PREFILL_CPT");
            if (vnni_prefill_cpt_env)
            {
                const int requested_cpt = std::atoi(vnni_prefill_cpt_env);
                if (requested_cpt == 1 || requested_cpt == 2 || requested_cpt == 4)
                {
                    vnni_prefill_cpt = requested_cpt;
                }
            }

            const char *vnni_prefill_grid_kpar_kb_env = std::getenv("LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR_KB");
            if (vnni_prefill_grid_kpar_kb_env)
            {
                vnni_prefill_grid_kpar_kb = std::clamp(std::atoi(vnni_prefill_grid_kpar_kb_env), 0, 128);
            }

            const char *vnni_prefill_variant_env = std::getenv("LLAMINAR_ROCM_VNNI_PREFILL_VARIANT");
            if (vnni_prefill_variant_env)
            {
                vnni_prefill_variant = std::clamp(std::atoi(vnni_prefill_variant_env), -1, 3);
            }

            const char *vnni_prefill_grid_variant_env = std::getenv("LLAMINAR_ROCM_VNNI_PREFILL_GRID_VARIANT");
            if (vnni_prefill_grid_variant_env)
            {
                vnni_prefill_grid_variant = std::clamp(std::atoi(vnni_prefill_grid_variant_env), -1, 3);
            }

            const char *vnni_prefill_ffn_override_env = std::getenv("LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE");
            if (vnni_prefill_ffn_override_env)
            {
                vnni_prefill_ffn_override = (std::atoi(vnni_prefill_ffn_override_env) != 0);
            }

            const char *vnni_prefill_ffn_override_grid_kpar_env = std::getenv("LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_KPAR");
            if (vnni_prefill_ffn_override_grid_kpar_env)
            {
                vnni_prefill_ffn_override_grid_kpar = std::clamp(std::atoi(vnni_prefill_ffn_override_grid_kpar_env), -1, 1);
            }

            const char *vnni_prefill_ffn_override_splits_env = std::getenv("LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_SPLITS");
            if (vnni_prefill_ffn_override_splits_env)
            {
                vnni_prefill_ffn_override_splits = std::clamp(std::atoi(vnni_prefill_ffn_override_splits_env), 0, 32);
            }

            const char *vnni_prefill_ffn_override_cpt_env = std::getenv("LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_CPT");
            if (vnni_prefill_ffn_override_cpt_env)
            {
                const int requested_cpt = std::atoi(vnni_prefill_ffn_override_cpt_env);
                if (requested_cpt == 0 || requested_cpt == 1 || requested_cpt == 2 || requested_cpt == 4)
                {
                    vnni_prefill_ffn_override_cpt = requested_cpt;
                }
            }

            const char *vnni_prefill_ffn_override_variant_env = std::getenv("LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_VARIANT");
            if (vnni_prefill_ffn_override_variant_env)
            {
                vnni_prefill_ffn_override_variant = std::clamp(std::atoi(vnni_prefill_ffn_override_variant_env), -1, 3);
            }

            const char *vnni_prefill_ffn_override_kernel_body_env = std::getenv("LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_KERNEL_BODY");
            if (vnni_prefill_ffn_override_kernel_body_env)
            {
                vnni_prefill_ffn_override_kernel_body = std::clamp(std::atoi(vnni_prefill_ffn_override_kernel_body_env), 0, 2);
            }

            const char *vnni_prefill_grid_swizzle_env = std::getenv("LLAMINAR_ROCM_VNNI_PREFILL_GRID_SWIZZLE");
            if (vnni_prefill_grid_swizzle_env)
            {
                vnni_prefill_grid_swizzle = std::clamp(std::atoi(vnni_prefill_grid_swizzle_env), -1, 1);
            }

            const char *vnni_prefill_ffn_override_grid_swizzle_env = std::getenv("LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_SWIZZLE");
            if (vnni_prefill_ffn_override_grid_swizzle_env)
            {
                vnni_prefill_ffn_override_grid_swizzle = std::clamp(std::atoi(vnni_prefill_ffn_override_grid_swizzle_env), -1, 1);
            }

            const char *wide_tile_v3_env = std::getenv("LLAMINAR_ROCM_WIDE_TILE_V3");
            if (wide_tile_v3_env)
            {
                wide_tile_v3 = (std::atoi(wide_tile_v3_env) != 0);
            }

            const char *wide_tile_v7_env = std::getenv("LLAMINAR_ROCM_WIDE_TILE_V7");
            if (wide_tile_v7_env)
            {
                wide_tile_v7 = (std::atoi(wide_tile_v7_env) != 0);
            }

            const char *wide_tile_kt_env = std::getenv("LLAMINAR_ROCM_WIDE_TILE_KT");
            if (wide_tile_kt_env)
            {
                const int kt = std::atoi(wide_tile_kt_env);
                if (kt == 4 || kt == 8 || kt == 16)
                {
                    wide_tile_kt = kt;
                }
            }

            const char *bw_v3_mt_env = std::getenv("LLAMINAR_ROCM_BLOCKWISE_V3_MT");
            if (bw_v3_mt_env)
            {
                blockwise_v3_mt = std::atoi(bw_v3_mt_env);
            }

            const char *bw_v7_mt_env = std::getenv("LLAMINAR_ROCM_BLOCKWISE_V7_MT");
            if (bw_v7_mt_env)
            {
                blockwise_v7_mt = std::atoi(bw_v7_mt_env);
            }

            const char *bw_v3_unroll_env = std::getenv("LLAMINAR_ROCM_BLOCKWISE_V3_UNROLL");
            if (bw_v3_unroll_env)
            {
                blockwise_v3_unroll = std::clamp(std::atoi(bw_v3_unroll_env), -1, 4);
            }

            const char *bw_v7_unroll_env = std::getenv("LLAMINAR_ROCM_BLOCKWISE_V7_UNROLL");
            if (bw_v7_unroll_env)
            {
                blockwise_v7_unroll = std::clamp(std::atoi(bw_v7_unroll_env), -1, 4);
            }

            const char *bw_force_v3_env = std::getenv("LLAMINAR_ROCM_BLOCKWISE_FORCE_V3");
            if (bw_force_v3_env)
            {
                blockwise_force_v3 = (std::atoi(bw_force_v3_env) != 0);
            }

            const char *bw_force_v7_env = std::getenv("LLAMINAR_ROCM_BLOCKWISE_FORCE_V7");
            if (bw_force_v7_env)
            {
                blockwise_force_v7 = (std::atoi(bw_force_v7_env) != 0);
            }

            const char *bw_quant_var_env = std::getenv("LLAMINAR_ROCM_BLOCKWISE_QUANT_VARIANT");
            if (bw_quant_var_env)
            {
                blockwise_quant_variant = std::clamp(std::atoi(bw_quant_var_env), 0, 5);
            }

            const char *nvnni_mt_env = std::getenv("LLAMINAR_ROCM_NVNNI_MT");
            if (nvnni_mt_env)
            {
                nvnni_mt = std::atoi(nvnni_mt_env);
            }

            const char *nvnni_unroll_env = std::getenv("LLAMINAR_ROCM_NVNNI_UNROLL");
            if (nvnni_unroll_env)
            {
                nvnni_unroll = std::clamp(std::atoi(nvnni_unroll_env), -1, 8);
            }

            const char *nvnni_min_blocks_env = std::getenv("LLAMINAR_ROCM_NVNNI_MIN_BLOCKS");
            if (nvnni_min_blocks_env)
            {
                nvnni_min_blocks = std::clamp(std::atoi(nvnni_min_blocks_env), -1, 3);
            }

            const char *nvnni_force_n64_env = std::getenv("LLAMINAR_ROCM_NVNNI_FORCE_N64");
            if (nvnni_force_n64_env)
            {
                nvnni_force_n64 = (std::atoi(nvnni_force_n64_env) != 0);
            }

            const char *nvnni_force_n128_env = std::getenv("LLAMINAR_ROCM_NVNNI_FORCE_N128");
            if (nvnni_force_n128_env)
            {
                nvnni_force_n128 = (std::atoi(nvnni_force_n128_env) != 0);
            }

            const char *ratio_prefill_variant_env = std::getenv("LLAMINAR_ROCM_RATIO_PREFILL_VARIANT");
            if (ratio_prefill_variant_env)
            {
                ratio_prefill_variant = std::clamp(std::atoi(ratio_prefill_variant_env), -1, 3);
            }

            const char *ratio_prefill_kb_env = std::getenv("LLAMINAR_ROCM_RATIO_PREFILL_KB");
            if (ratio_prefill_kb_env)
            {
                ratio_prefill_kb = std::clamp(std::atoi(ratio_prefill_kb_env), 0, 32);
            }

            const char *ratio_prefill_linear_variant_env = std::getenv("LLAMINAR_ROCM_RATIO_PREFILL_LINEAR_VARIANT");
            if (ratio_prefill_linear_variant_env)
            {
                ratio_prefill_linear_variant = std::clamp(std::atoi(ratio_prefill_linear_variant_env), -1, 3);
            }

            const char *ratio_prefill_linear_kb_env = std::getenv("LLAMINAR_ROCM_RATIO_PREFILL_LINEAR_KB");
            if (ratio_prefill_linear_kb_env)
            {
                ratio_prefill_linear_kb = std::clamp(std::atoi(ratio_prefill_linear_kb_env), 0, 32);
            }

            const char *ratio_prefill_iq4_variant_env = std::getenv("LLAMINAR_ROCM_RATIO_PREFILL_IQ4_VARIANT");
            if (ratio_prefill_iq4_variant_env)
            {
                ratio_prefill_iq4_variant = std::clamp(std::atoi(ratio_prefill_iq4_variant_env), -1, 3);
            }

            const char *ratio_prefill_iq4_kb_env = std::getenv("LLAMINAR_ROCM_RATIO_PREFILL_IQ4_KB");
            if (ratio_prefill_iq4_kb_env)
            {
                ratio_prefill_iq4_kb = std::clamp(std::atoi(ratio_prefill_iq4_kb_env), 0, 32);
            }

            const char *concurrent_prefill_env = std::getenv("LLAMINAR_ROCM_CONCURRENT_PREFILL");
            if (concurrent_prefill_env)
            {
                concurrent_prefill = (std::atoi(concurrent_prefill_env) != 0);
            }

            const char *concurrent_decode_env = std::getenv("LLAMINAR_ROCM_CONCURRENT_DECODE");
            if (concurrent_decode_env)
            {
                concurrent_decode = (std::atoi(concurrent_decode_env) != 0);
            }

            const char *moe_grouped_decode_env = std::getenv("LLAMINAR_ROCM_MOE_GROUPED_DECODE");
            if (moe_grouped_decode_env)
            {
                moe_grouped_decode = (std::atoi(moe_grouped_decode_env) != 0);
            }

            const char *moe_grouped_decode_router_env = std::getenv("LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER");
            if (moe_grouped_decode_router_env)
            {
                moe_grouped_decode_router = (std::atoi(moe_grouped_decode_router_env) != 0);
            }

            const char *moe_device_routed_decode_env = std::getenv("LLAMINAR_ROCM_MOE_DEVICE_ROUTED_DECODE");
            if (moe_device_routed_decode_env)
            {
                moe_device_routed_decode = (std::atoi(moe_device_routed_decode_env) != 0);
            }

            const char *moe_grouped_prefill_env = std::getenv("LLAMINAR_ROCM_MOE_GROUPED_PREFILL");
            if (moe_grouped_prefill_env)
            {
                moe_grouped_prefill = (std::atoi(moe_grouped_prefill_env) != 0);
            }

            const char *repack_slots_env = std::getenv("LLAMINAR_ROCM_REPACK_SLOTS");
            if (repack_slots_env)
            {
                repack_slots = std::clamp(std::atoi(repack_slots_env), 1, 8);
            }

            const char *repack_budget_env = std::getenv("LLAMINAR_ROCM_REPACK_BUDGET_MB");
            if (repack_budget_env)
            {
                repack_budget_mb = std::max(0, std::atoi(repack_budget_env));
            }

            const char *repack_streams_env = std::getenv("LLAMINAR_ROCM_REPACK_STREAMS");
            if (repack_streams_env)
            {
                repack_streams = std::clamp(std::atoi(repack_streams_env), 1, 8);
            }
        }
    };

    /**
     * @brief HybridQ16 debug configuration
     */
    struct HybridQ16Config
    {
        bool trace = false; ///< Enable HybridQ16 dataflow trace logs (LLAMINAR_HYBRIDQ16_TRACE)

        HybridQ16Config()
        {
            reload();
        }

        void reload()
        {
            trace = false;
            const char *trace_env = std::getenv("LLAMINAR_HYBRIDQ16_TRACE");
            if (trace_env)
            {
                trace = (std::atoi(trace_env) != 0);
            }
        }
    };

    /**
     * @brief CPU NativeVNNI tile dispatch configuration
     *
     * Override the auto-tuned tile parameters for the CPU NativeVNNI GEMV/GEMM
     * kernel (CPUNativeVNNIGemv.h / CPUNativeVNNITileConfig.h).
     * All values default to 0 (= auto, use heuristic).
     *
     * Environment Variables:
     *   LLAMINAR_CPU_VNNI_N_BLOCK_CHUNKS   - N-chunks per parallel task (0=auto)
     *   LLAMINAR_CPU_VNNI_K_TILE_BLOCKS    - K-blocks per tile (0=auto/full K)
     *   LLAMINAR_CPU_VNNI_M_UNROLL         - M-loop unroll factor (0=auto, 1 or 2)
     *   LLAMINAR_CPU_VNNI_K_TILES          - K-parallel tile count for GEMV (0=auto)
     *   LLAMINAR_CPU_VNNI_MIN_BPR_K_PARALLEL - Min blocks-per-row for K-parallel (0=auto, default 256)
     */
    struct CPUVNNIConfig
    {
        int n_block_chunks = 0;     ///< Override n_block_chunks (0=auto)
        int k_tile_blocks = 0;      ///< Override k_tile_blocks (0=auto)
        int m_unroll = 0;           ///< Override m_unroll (0=auto)
        int k_tiles = 0;            ///< Override k_tiles (0=auto)
        int min_bpr_k_parallel = 0; ///< Override MIN_BPR_FOR_K_PARALLEL (0=auto → 256)

        CPUVNNIConfig()
        {
            reload();
        }

        void reload()
        {
            auto readInt = [](const char *name) -> int
            {
                const char *v = std::getenv(name);
                return v ? std::atoi(v) : 0;
            };
            n_block_chunks = readInt("LLAMINAR_CPU_VNNI_N_BLOCK_CHUNKS");
            k_tile_blocks = readInt("LLAMINAR_CPU_VNNI_K_TILE_BLOCKS");
            m_unroll = readInt("LLAMINAR_CPU_VNNI_M_UNROLL");
            k_tiles = readInt("LLAMINAR_CPU_VNNI_K_TILES");
            min_bpr_k_parallel = readInt("LLAMINAR_CPU_VNNI_MIN_BPR_K_PARALLEL");
        }
    };

    /**
     * @brief Global debug environment snapshot
     */
    struct DebugEnv
    {
        DequantConfig dequant;
        GemmConfig gemm;
        CPUVNNIConfig cpu_vnni; ///< CPU NativeVNNI tile dispatch overrides
        ProfileConfig profile;
        RMSNormConfig rmsnorm;
        AttentionConfig attention; ///< Q8_1 attention precision configuration
        Q16AttentionDumpConfig q16_attention_dump;
        ExecutionConfig execution;                 ///< LayerExecutor framework configuration
        SnapshotConfig snapshot;                   ///< Snapshot and tensor dump configuration
        StageDumpConfig stage_dump;                ///< Compute stage input/output dumping
        MPILoggingConfig mpi_logging;              ///< MPI collective operation logging
        StageOutputPrintConfig stage_output_print; ///< Stage output debug printing
        ValidationConfig validation;               ///< Buffer validation configuration
        StreamingEnv streaming;                    ///< Weight streaming configuration (Option B)
        HybridQ16Config hybrid_q16;                ///< HybridQ16 debug configuration
        ROCmConfig rocm;                           ///< ROCm-specific debugging configuration
        TransferTracingConfig transfer_tracing;    ///< Memory transfer tracing for H2D/D2H debugging
        LoggerConfig logger;                       ///< Logger environment configuration
        TopologyEnvConfig topology;                ///< Topology-related environment configuration
        MPIBootstrapEnvConfig mpi_bootstrap;       ///< MPI bootstrap environment snapshot

        /// MoE expert rebalancing configuration
        struct
        {
            /// Rebalance mode: "off", "observe", "dynamic" (from LLAMINAR_MOE_REBALANCE)
            std::string mode = "dynamic";
            /// Histogram window size in decode tokens (from LLAMINAR_MOE_REBALANCE_WINDOW)
            int window_size = 256;
            /// Max window size for adaptive growth (0 = no growth) (from LLAMINAR_MOE_REBALANCE_MAX_WINDOW)
            int max_window_size = 4096;
            /// Window growth factor after each rebalance (from LLAMINAR_MOE_REBALANCE_WINDOW_GROWTH)
            float window_growth_factor = 1.5f;
            /// Max experts to replicate per socket (0 = auto: 2×top_k) (from LLAMINAR_MOE_REBALANCE_REPLICAS)
            int max_replicas = 0;
            /// Routed experts per layer to cache on GPU in mixed CPU/GPU MoE domains.
            /// 0 disables cross-domain GPU-cache placement. (from LLAMINAR_MOE_GPU_EXPERT_CACHE)
            int gpu_cache_experts_per_layer = 0;
            /// Release raw expert weight data after eager packed-weight preparation.
            /// Enabled by default; set LLAMINAR_MOE_RELEASE_RAW_WEIGHTS=0 to opt out.
            bool release_raw_weights = true;
        } moe_rebalance;

        /// MoE expert overlay transfer/debug bridge configuration.
        struct
        {
            /// Force dense full-sequence routed-tier transfers for compatibility/debugging.
            /// Sparse token-row transfer remains the production default.
            /// (env: LLAMINAR_MOE_EP_DENSE_TRANSFER)
            bool dense_transfer = false;
            /// Emit human-readable overlay profiling and routing diagnostics.
            /// (env: LLAMINAR_MOE_EP_TRACE)
            bool trace = false;
            /// Dump resolved per-layer expert tier placement summaries.
            /// (env: LLAMINAR_MOE_EP_DUMP_PLACEMENT)
            bool dump_placement = false;
            /// Trace selected rows, routed entries, and transfer byte estimates by domain.
            /// (env: LLAMINAR_MOE_EP_TRANSFER_TRACE)
            bool transfer_trace = false;
            /// Write machine-readable overlay profiling rows at timeline flush.
            /// (env: LLAMINAR_MOE_EP_PROFILE_CSV)
            bool profile_csv_enabled = false;
            /// CSV output path. Truthy env values use this default; non-bool env values are paths.
            std::string profile_csv_path = "/tmp/llaminar_moe_ep_profile.csv";
        } moe_expert_overlay;

        bool tp_timing = false;                    ///< Enable TP forward timing breakdown (env: LLAMINAR_TP_TIMING)
        bool skip_allreduce = false;               ///< DIAGNOSTIC: Skip allreduce for profiling (env: LLAMINAR_SKIP_ALLREDUCE)
        bool tp_collective_contract_trace = false; ///< Trace LocalTP collective context identity and sequence contract (env: LLAMINAR_TP_COLLECTIVE_CONTRACT_TRACE)
        bool gpu_stage_timing = true;              ///< GPU event-based per-stage timing (env: LLAMINAR_GPU_STAGE_TIMING)
        bool gpu_stage_timing_detail = false;      ///< Print per-stage detail (env: LLAMINAR_GPU_STAGE_TIMING_DETAIL)
        bool coherence_audit = false;              ///< Per-tensor coherence audit log (env: LLAMINAR_COHERENCE_AUDIT)

        /// Block-diagonal activation rotation for Q8_1 kurtosis reduction.
        /// Reduces outlier sensitivity in int8 quantization by spreading energy across blocks.
        /// (env: LLAMINAR_ACTIVATION_ROTATION, default: "1" = enabled)
        /// Set to "0" to disable. Adds one-time cost at model load (~2s for 4B model).
        bool activation_rotation = true;

        /// Block dimension for activation rotation (env: LLAMINAR_ROTATION_BLOCK_DIM, default: 128)
        /// Must divide hidden_dim and ffn_dim evenly. Smaller = less beneficial, larger = more expensive.
        int rotation_block_dim = 128;

        /// Block-diagonal orthogonal rotation for Q16_1 KV cache quantization.
        /// Spreads outlier energy (from low-frequency RoPE dimensions) across all dims,
        /// reducing clipping at the fixed Q16_1 scale.
        /// (env: LLAMINAR_KV_ROTATION, default: "1" = enabled)
        /// Set to "0" to disable for benchmarking the rotation overhead.
        bool kv_rotation = true;

        /// Global allreduce precision fallback: "fp16", "bf16", or "fp32"
        /// (env: LLAMINAR_ALLREDUCE_PRECISION, default: "fp32")
        /// This is the ULTIMATE FALLBACK — per-layer precision from the model schema
        /// takes priority. Only used when no schema-level precision is configured.
        /// FP16/BF16 halves PCIe transfer bandwidth; FP32 is lossless but slower on bandwidth-limited links.
        std::string allreduce_precision = "fp32";

        /// Timeout in ms for tensor-parallel coordination waits and blocking MPI collectives
        /// (env: LLAMINAR_TP_COLLECT_TIMEOUT_MS).
        /// Debug/Integration builds default to a 30s safety net to avoid deadlocked tests;
        /// Release builds default to 0 (wait forever) for production runs.
#if LLAMINAR_ASSERTIONS_ACTIVE
        int tp_collect_timeout_ms = 30000;
#else
        int tp_collect_timeout_ms = 0;
#endif

        /// Enable model weight lifecycle trace events (env: LLAMINAR_WEIGHT_LIFECYCLE_TRACE=1)
        bool weight_lifecycle_trace = false;

        static std::string normalizedEnvValue(const char *value)
        {
            if (!value)
                return {};
            std::string normalized(value);
            normalized.erase(normalized.begin(), std::find_if(normalized.begin(), normalized.end(), [](unsigned char ch)
                                                              { return !std::isspace(ch); }));
            normalized.erase(std::find_if(normalized.rbegin(), normalized.rend(), [](unsigned char ch)
                                          { return !std::isspace(ch); })
                                 .base(),
                             normalized.end());
            std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch)
                           { return static_cast<char>(std::tolower(ch)); });
            return normalized;
        }

        static bool isTruthyEnvValue(const char *value)
        {
            const std::string normalized = normalizedEnvValue(value);
            if (!normalized.empty() &&
                std::all_of(normalized.begin(), normalized.end(), [](unsigned char ch)
                            { return std::isdigit(ch) != 0; }))
            {
                return std::atoi(normalized.c_str()) != 0;
            }
            return normalized == "1" || normalized == "true" || normalized == "on" || normalized == "yes";
        }

        static bool isFalseyEnvValue(const char *value)
        {
            const std::string normalized = normalizedEnvValue(value);
            return normalized.empty() || normalized == "0" || normalized == "false" || normalized == "off" || normalized == "no";
        }

        void reloadMoEExpertOverlayEnv()
        {
            moe_expert_overlay.dense_transfer = isTruthyEnvValue(std::getenv("LLAMINAR_MOE_EP_DENSE_TRANSFER"));
            moe_expert_overlay.trace = isTruthyEnvValue(std::getenv("LLAMINAR_MOE_EP_TRACE"));
            moe_expert_overlay.dump_placement = isTruthyEnvValue(std::getenv("LLAMINAR_MOE_EP_DUMP_PLACEMENT"));
            moe_expert_overlay.transfer_trace = isTruthyEnvValue(std::getenv("LLAMINAR_MOE_EP_TRANSFER_TRACE"));

            moe_expert_overlay.profile_csv_enabled = false;
            moe_expert_overlay.profile_csv_path = "/tmp/llaminar_moe_ep_profile.csv";
            const char *csv = std::getenv("LLAMINAR_MOE_EP_PROFILE_CSV");
            if (!csv || isFalseyEnvValue(csv))
                return;
            moe_expert_overlay.profile_csv_enabled = true;
            if (!isTruthyEnvValue(csv))
                moe_expert_overlay.profile_csv_path = csv;
        }

        DebugEnv()
        {
            const char *tp_env = std::getenv("LLAMINAR_TP_TIMING");
            tp_timing = tp_env && std::string(tp_env) == "1";
            const char *skip_ar = std::getenv("LLAMINAR_SKIP_ALLREDUCE");
            skip_allreduce = skip_ar && std::string(skip_ar) == "1";
            tp_collective_contract_trace = isTruthyEnvValue(std::getenv("LLAMINAR_TP_COLLECTIVE_CONTRACT_TRACE"));
            const char *stl_env = std::getenv("LLAMINAR_GPU_STAGE_TIMING");
            if (stl_env && std::string(stl_env) == "0")
                gpu_stage_timing = false;
            const char *stl_detail = std::getenv("LLAMINAR_GPU_STAGE_TIMING_DETAIL");
            gpu_stage_timing_detail = stl_detail && std::string(stl_detail) == "1";
            if (gpu_stage_timing_detail)
                gpu_stage_timing = true;
            const char *ar_prec = std::getenv("LLAMINAR_ALLREDUCE_PRECISION");
            if (ar_prec)
                allreduce_precision = ar_prec;
            const char *collect_timeout = std::getenv("LLAMINAR_TP_COLLECT_TIMEOUT_MS");
            if (collect_timeout)
                tp_collect_timeout_ms = std::atoi(collect_timeout);
            const char *weight_trace = std::getenv("LLAMINAR_WEIGHT_LIFECYCLE_TRACE");
            weight_lifecycle_trace = weight_trace && std::string(weight_trace) == "1";
            const char *coh_audit = std::getenv("LLAMINAR_COHERENCE_AUDIT");
            coherence_audit = coh_audit && std::string(coh_audit) == "1";
            const char *act_rot = std::getenv("LLAMINAR_ACTIVATION_ROTATION");
            if (act_rot && std::string(act_rot) == "0")
                activation_rotation = false;
            const char *rot_bdim = std::getenv("LLAMINAR_ROTATION_BLOCK_DIM");
            if (rot_bdim)
                rotation_block_dim = std::atoi(rot_bdim);
            const char *kv_rot = std::getenv("LLAMINAR_KV_ROTATION");
            if (kv_rot && std::string(kv_rot) == "0")
                kv_rotation = false;
            // MoE rebalancing
            const char *moe_reb = std::getenv("LLAMINAR_MOE_REBALANCE");
            if (moe_reb)
                moe_rebalance.mode = moe_reb;
            const char *moe_reb_win = std::getenv("LLAMINAR_MOE_REBALANCE_WINDOW");
            if (moe_reb_win)
                moe_rebalance.window_size = std::atoi(moe_reb_win);
            const char *moe_reb_max_win = std::getenv("LLAMINAR_MOE_REBALANCE_MAX_WINDOW");
            if (moe_reb_max_win)
                moe_rebalance.max_window_size = std::atoi(moe_reb_max_win);
            const char *moe_reb_growth = std::getenv("LLAMINAR_MOE_REBALANCE_WINDOW_GROWTH");
            if (moe_reb_growth)
                moe_rebalance.window_growth_factor = std::atof(moe_reb_growth);
            const char *moe_reb_replicas = std::getenv("LLAMINAR_MOE_REBALANCE_REPLICAS");
            if (moe_reb_replicas)
                moe_rebalance.max_replicas = std::atoi(moe_reb_replicas);
            const char *moe_gpu_cache = std::getenv("LLAMINAR_MOE_GPU_EXPERT_CACHE");
            if (!moe_gpu_cache)
                moe_gpu_cache = std::getenv("LLAMINAR_MOE_GPU_EXPERT_CACHE_PER_LAYER");
            if (moe_gpu_cache)
                moe_rebalance.gpu_cache_experts_per_layer = std::atoi(moe_gpu_cache);
            const char *moe_reb_release_ctor = std::getenv("LLAMINAR_MOE_RELEASE_RAW_WEIGHTS");
            if (moe_reb_release_ctor)
                moe_rebalance.release_raw_weights = (std::atoi(moe_reb_release_ctor) != 0);
            reloadMoEExpertOverlayEnv();
        }

        void reload()
        {
            const char *tp_env = std::getenv("LLAMINAR_TP_TIMING");
            tp_timing = tp_env && std::string(tp_env) == "1";
            const char *skip_ar = std::getenv("LLAMINAR_SKIP_ALLREDUCE");
            skip_allreduce = skip_ar && std::string(skip_ar) == "1";
            tp_collective_contract_trace = isTruthyEnvValue(std::getenv("LLAMINAR_TP_COLLECTIVE_CONTRACT_TRACE"));
            gpu_stage_timing = true; // default on
            const char *stl_env = std::getenv("LLAMINAR_GPU_STAGE_TIMING");
            if (stl_env && std::string(stl_env) == "0")
                gpu_stage_timing = false;
            const char *stl_detail = std::getenv("LLAMINAR_GPU_STAGE_TIMING_DETAIL");
            gpu_stage_timing_detail = stl_detail && std::string(stl_detail) == "1";
            if (gpu_stage_timing_detail)
                gpu_stage_timing = true;
            const char *ar_prec = std::getenv("LLAMINAR_ALLREDUCE_PRECISION");
            if (ar_prec)
                allreduce_precision = ar_prec;
            const char *collect_timeout = std::getenv("LLAMINAR_TP_COLLECT_TIMEOUT_MS");
            if (collect_timeout)
                tp_collect_timeout_ms = std::atoi(collect_timeout);
            const char *weight_trace = std::getenv("LLAMINAR_WEIGHT_LIFECYCLE_TRACE");
            weight_lifecycle_trace = weight_trace && std::string(weight_trace) == "1";
            const char *coh_audit = std::getenv("LLAMINAR_COHERENCE_AUDIT");
            coherence_audit = coh_audit && std::string(coh_audit) == "1";
            activation_rotation = true; // default on
            const char *act_rot = std::getenv("LLAMINAR_ACTIVATION_ROTATION");
            if (act_rot && std::string(act_rot) == "0")
                activation_rotation = false;
            const char *rot_bdim = std::getenv("LLAMINAR_ROTATION_BLOCK_DIM");
            if (rot_bdim)
                rotation_block_dim = std::atoi(rot_bdim);
            kv_rotation = true; // default on
            const char *kv_rot = std::getenv("LLAMINAR_KV_ROTATION");
            if (kv_rot && std::string(kv_rot) == "0")
                kv_rotation = false;
            // MoE rebalancing
            moe_rebalance.mode = "dynamic";
            const char *moe_reb = std::getenv("LLAMINAR_MOE_REBALANCE");
            if (moe_reb)
                moe_rebalance.mode = moe_reb;
            moe_rebalance.window_size = 256;
            const char *moe_reb_win = std::getenv("LLAMINAR_MOE_REBALANCE_WINDOW");
            if (moe_reb_win)
                moe_rebalance.window_size = std::atoi(moe_reb_win);
            moe_rebalance.max_window_size = 4096;
            const char *moe_reb_max_win = std::getenv("LLAMINAR_MOE_REBALANCE_MAX_WINDOW");
            if (moe_reb_max_win)
                moe_rebalance.max_window_size = std::atoi(moe_reb_max_win);
            moe_rebalance.window_growth_factor = 1.5f;
            const char *moe_reb_growth = std::getenv("LLAMINAR_MOE_REBALANCE_WINDOW_GROWTH");
            if (moe_reb_growth)
                moe_rebalance.window_growth_factor = std::atof(moe_reb_growth);
            moe_rebalance.max_replicas = 0;
            const char *moe_reb_replicas = std::getenv("LLAMINAR_MOE_REBALANCE_REPLICAS");
            if (moe_reb_replicas)
                moe_rebalance.max_replicas = std::atoi(moe_reb_replicas);
            moe_rebalance.gpu_cache_experts_per_layer = 0;
            const char *moe_gpu_cache = std::getenv("LLAMINAR_MOE_GPU_EXPERT_CACHE");
            if (!moe_gpu_cache)
                moe_gpu_cache = std::getenv("LLAMINAR_MOE_GPU_EXPERT_CACHE_PER_LAYER");
            if (moe_gpu_cache)
                moe_rebalance.gpu_cache_experts_per_layer = std::atoi(moe_gpu_cache);
            moe_rebalance.release_raw_weights = true;
            const char *moe_reb_release = std::getenv("LLAMINAR_MOE_RELEASE_RAW_WEIGHTS");
            if (moe_reb_release)
                moe_rebalance.release_raw_weights = (std::atoi(moe_reb_release) != 0);
            reloadMoEExpertOverlayEnv();
            gemm.reload();
            profile.reload();
            rmsnorm.reload();
            attention.reload();
            q16_attention_dump.reload();
            execution.reload();
            snapshot.reload();
            stage_dump.reload();
            mpi_logging.reload();
            stage_output_print.reload();
            validation.reload();
            streaming.reload();
            hybrid_q16.reload();
            rocm.reload();
            transfer_tracing.reload();
            logger.reload();
            topology.reload();
            mpi_bootstrap.reload();
        }
    };

    /**
     * @brief Access global debug environment (mutable for testing/reloading)
     */
    inline DebugEnv &mutableDebugEnv()
    {
        static DebugEnv env;
        return env;
    }

    /**
     * @brief Access global debug environment (read-only)
     */
    inline const DebugEnv &debugEnv()
    {
        return mutableDebugEnv();
    }

} // namespace llaminar2
