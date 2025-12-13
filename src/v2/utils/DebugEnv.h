#pragma once

#include <cstdlib>
#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <algorithm>

/**
 * @file DebugEnv.h
 * @brief Runtime configuration via environment variables for v2 architecture
 * @author David Sanftenberg
 *
 * Centralized configuration for debug/instrumentation features.
 * All environment variables are parsed once at startup to avoid hot-path overhead.
 */

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
        bool enabled = false;       ///< Enable kernel profiling (LLAMINAR_PROFILE_KERNELS=1)
        bool per_layer = false;     ///< Breakdown by layer index (LLAMINAR_PROFILE_PER_LAYER=1)
        bool per_iteration = false; ///< Print stats per decode iteration (LLAMINAR_PROFILE_PER_ITER=1)
        int print_interval = 0;     ///< Print every N iterations (0=only at end)

        ProfileConfig()
        {
            reload();
        }

        void reload()
        {
            const char *enabled_env = std::getenv("LLAMINAR_PROFILE_KERNELS");
            if (enabled_env)
            {
                enabled = (std::atoi(enabled_env) != 0);
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
        }

        bool trace_q8_1_direct = false; ///< Enable detailed Q8_1 JIT kernel tracing
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
     * @brief Global debug environment snapshot
     */
    struct DebugEnv
    {
        DequantConfig dequant;
        GemmConfig gemm;
        ProfileConfig profile;
        RMSNormConfig rmsnorm;
        AttentionConfig attention; ///< Q8_1 attention precision configuration
        SnapshotConfig snapshot;   ///< Snapshot and tensor dump configuration

        // Add more config groups as needed:
        // NumaConfig numa;
        // MPIConfig mpi;

        DebugEnv() = default;

        void reload()
        {
            gemm.reload();
            profile.reload();
            rmsnorm.reload();
            attention.reload();
            snapshot.reload();
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
