/**
 * @file debug_env.h
 * @brief Unified registry for LLAMINAR_* and related debug / diagnostic environment flags.
 *
 * This central facility avoids ad-hoc std::getenv() calls spread across the codebase.
 * It snapshots environment variables on first access (lazy, thread-safe) and exposes
 * structured groups. Extend incrementally: when you introduce a new flag, add a field
 * and parsing logic here instead of sprinkling more getenv calls.
 */
#pragma once
#include <string>
#include <cstdlib>
#include <cstdint>
#include <vector>

namespace llaminar
{

    struct ShardingEnv
    {
        bool debug_materialize_attention = false; // LLAMINAR_DEBUG_MATERIALIZE_ATTENTION
        bool dump_shards = false;                 // LLAMINAR_DUMP_SHARDS
        int dump_shards_n = 16;                   // LLAMINAR_DUMP_SHARDS_N (optional)
        bool shard_parity_check = false;          // LLAMINAR_SHARD_PARITY_CHECK
        bool assert_replicated_misuse = false;    // LLAMINAR_ASSERT_REPLICATED_MISUSE
        bool shard_load_diag = false;             // LLAMINAR_SHARD_LOAD_DIAG
    };

    struct CosmaEnv
    {
        int prefill_threshold = 4096;       // LLAMINAR_COSMA_PREFILL_THRESHOLD
        int fast_path_threshold = -1;       // LLAMINAR_COSMA_FAST_PATH_THRESHOLD
        int validate_tile = 0;              // LLAMINAR_COSMA_VALIDATE_TILE (>0 enables)
        int log_level = 2;                  // LLAMINAR_COSMA_LOG_LEVEL (0..4)
        long long max_resident_mb = 2048;   // LLAMINAR_COSMA_MAX_RESIDENT_MB
        bool disable = false;               // LLAMINAR_COSMA_DISABLE
        bool force = false;                 // LLAMINAR_COSMA_FORCE
        bool diag = false;                  // LLAMINAR_COSMA_DIAG
        bool diag_deep = false;             // LLAMINAR_COSMA_DIAG_DEEP
        bool diag_axis = false;             // LLAMINAR_COSMA_DIAG_AXIS
        bool compare_replicated = false;    // LLAMINAR_COSMA_COMPARE_REPLICATED
        bool debug_recon = false;           // LLAMINAR_COSMA_DEBUG_RECON
        bool force_direct = false;          // LLAMINAR_COSMA_FORCE_DIRECT
        bool force_replicated = false;      // LLAMINAR_COSMA_FORCE_REPLICATED
        bool force_replicated_diag = false; // LLAMINAR_COSMA_FORCE_REPLICATED_DIAG
        bool fast_unverified = false;       // LLAMINAR_COSMA_FAST_UNVERIFIED
        bool auto_fix_transpose = false;    // LLAMINAR_COSMA_AUTO_FIX_TRANSPOSE
        bool force_fallback = false;        // LLAMINAR_COSMA_FORCE_FALLBACK
        bool force_distributed_act = false; // LLAMINAR_COSMA_FORCE_DISTRIBUTED_ACT
        bool disable_fused_dequant = false; // LLAMINAR_COSMA_DISABLE_FUSED_DEQUANT
        // Extended diagnostics / reconstruction / overlap / preflight
        bool diag_coord_invert = false;           // LLAMINAR_COSMA_DIAG_COORD_INVERT
        bool diag_local_probe = false;            // LLAMINAR_COSMA_DIAG_LOCAL_PROBE
        bool diag_local_probe_deep = false;       // LLAMINAR_COSMA_DIAG_LOCAL_PROBE_DEEP
        bool diag_recon_bypass = false;           // LLAMINAR_COSMA_DIAG_RECON_BYPASS
        bool diag_recon_transpose = false;        // LLAMINAR_COSMA_DIAG_RECON_TRANSPOSE
        bool diag_recon_brute = false;            // LLAMINAR_COSMA_DIAG_RECON_BRUTE
        bool diag_recon_map = false;              // LLAMINAR_COSMA_DIAG_RECON_MAP
        bool recon_force_legacy = false;          // LLAMINAR_COSMA_RECON_FORCE_LEGACY
        bool diag_swaprc = false;                 // LLAMINAR_COSMA_DIAG_SWAPRC
        bool diag_try_transpose = false;          // LLAMINAR_COSMA_DIAG_TRY_TRANSPOSE
        bool diag_skip_norm = false;              // LLAMINAR_COSMA_DIAG_SKIP_NORM
        bool diag_dump_small = false;             // LLAMINAR_COSMA_DUMP_SMALL
        bool pop_forward_legacy = false;          // LLAMINAR_COSMA_POP_FORWARD_LEGACY
        bool replicate_B = false;                 // LLAMINAR_COSMA_REPLICATE_B
        bool force_unified_strategy = false;      // LLAMINAR_COSMA_FORCE_UNIFIED
        bool overlap_enabled = false;             // LLAMINAR_COSMA_OVERLAP_STREAM
        bool overlap_verbose = false;             // LLAMINAR_COSMA_OVERLAP_VERBOSE
        bool preflight_disable = false;           // LLAMINAR_COSMA_PREFLIGHT_DISABLE
        bool rmsnorm_validate = false;            // LLAMINAR_COSMA_RMSNORM_VALIDATE
        bool rmsnorm_trace = false;               // LLAMINAR_COSMA_RMSNORM_TRACE
        bool rmsnorm_trace_points_active = false; // derived from LLAMINAR_COSMA_RMSNORM_TRACE_POINTS
        std::string rmsnorm_trace_points_spec;    // spec string
        bool direct_threshold_override = false;   // LLAMINAR_COSMA_DIRECT_THRESHOLD_OPS present
        long long direct_threshold_ops = 0;       // parsed direct threshold override
        bool diag_tap_enabled = false;            // LLAMINAR_COSMA_DIAG_TAP present
        int diag_tap_value = 0;                   // tap value
        bool diag_perm_infer_active = false;      // LLAMINAR_COSMA_DIAG_PERM_INFER
        std::string diag_perm_spec;               // perm infer spec
        bool diag_samples_active = false;         // LLAMINAR_COSMA_DIAG_SAMPLES
        std::string diag_samples_spec;            // samples spec
        double preflight_safety = 1.2;            // LLAMINAR_COSMA_PREFLIGHT_SAFETY_FACTOR (clamped)
        bool preflight_safety_override = false;   // true if user provided override
        int forced_openblas_threads = 0;          // LLAMINAR_OPENBLAS_THREADS / OPENBLAS_NUM_THREADS
        int forced_replicated_threads = 0;        // LLAMINAR_COSMA_FORCE_REPLICATED_THREADS
    };

    struct PipelineEnv
    {
        bool capture_pre_lm = false;             // LLAMINAR_PIPELINE_CAPTURE_PRE_LM
        bool layerwise_stats = false;            // LLAMINAR_PIPELINE_LAYERWISE_STATS
        bool enable_abstract_pipeline = false;   // LLAMINAR_ENABLE_ABSTRACT_PIPELINE (feature flag for new AbstractPipeline scaffolding)
        bool disable_incremental_decode = false; // LLAMINAR_DISABLE_INCREMENTAL_DECODE (forces replay decode path)
        bool layer_token_diff = false;           // LLAMINAR_PIPELINE_LAYER_TOKEN_DIFF (capture last-token row per layer for diff diagnostics)
        bool layer_token_diff_verbose = false;   // LLAMINAR_PIPELINE_LAYER_TOKEN_DIFF_VERBOSE (log each capture row)
        bool attn_ref_compare = false;           // LLAMINAR_DEBUG_ATTENTION_REF (run reference full attention for incremental token)
        bool layer_replay_compare = false;       // LLAMINAR_DEBUG_LAYER_REPLAY_COMPARE (per-layer immediate replay compare during incremental decode)
        bool pre_lm_row_diff = false;            // LLAMINAR_PIPELINE_PRE_LM_ROW_DIFF (emit per-call pre/post LM head last-row diff logs)
    };

    // KV cache policy (dynamic capacity management for incremental decode)
    struct KVCacheEnv
    {
        bool dynamic_init = false; // LLAMINAR_KV_DYNAMIC_INIT (allocate exactly prefill length instead of max_seq_len)
        int growth_factor = 2;     // LLAMINAR_KV_GROWTH_FACTOR (capacity *= factor when expansion required)
    };

    struct DequantEnv
    {
        bool stats = false;     // LLAMINAR_DEQUANT_STATS
        bool anomalies = false; // LLAMINAR_DEQUANT_ANOMALIES
    };

    struct AdaptiveEnv
    {
        bool disable_cosma = false; // ADAPTIVE_DISABLE_COSMA
    };

    struct AttentionEnv
    {
        bool validate_primitives = false; // LLAMINAR_ATTN_PRIMITIVES_VALIDATE
        bool validate_output = false;     // LLAMINAR_ATTN_OUTPUT_VALIDATE
        bool use_primitives = true;       // LLAMINAR_ATTN_USE_PRIMITIVES (toggle new centralized primitive path)
        // Newly centralized flags
        std::string output_mode;         // LLAMINAR_ATTN_OUTPUT_MODE
        bool output_mode_forced = false; // presence of mode env
        int gather_threshold = -1;       // LLAMINAR_ATTN_GATHER_THRESHOLD
        bool force_scalar = false;       // LLAMINAR_ATTN_FORCE_SCALAR
        bool validate_proj = false;      // LLAMINAR_ATTN_VALIDATE_PROJ
        bool micro_trace = false;        // LLAMINAR_ATTN_MICRO_TRACE
        bool dump_attention = false;     // LLAMINAR_ATTN_DUMP_ATTENTION
        bool tp_disable = false;         // LLAMINAR_ATTN_TP_DISABLE
        int tp_partitions = 1;           // LLAMINAR_ATTN_TP_PARTITIONS
        bool tp_auto = false;            // LLAMINAR_ATTN_TP_AUTO
        bool tp_force_splitter = false;  // LLAMINAR_ATTN_TP_FORCE_SPLITTER
        // Primitive optimization knobs
        int prim_parallel_elems_threshold = 32768;   // LLAMINAR_ATTN_PRIM_PARALLEL_ELEMS (heads*seq_len*seq_len or heads*seq_len*D)
        bool prim_force_scalar = false;              // LLAMINAR_ATTN_PRIM_FORCE_SCALAR
        int prim_fused_recompute_threshold = 0;      // LLAMINAR_ATTN_PRIM_FUSED_RECOMPUTE_THRESHOLD (seq_len threshold to use fused two-pass path)
        bool prim_force_fused = false;               // LLAMINAR_ATTN_PRIM_FORCE_FUSED
        bool prim_disable_fused = false;             // LLAMINAR_ATTN_PRIM_DISABLE_FUSED
        bool prim_rope_vectorize = true;             // LLAMINAR_ATTN_PRIM_ROPE_VECTORIZE (disable for debugging)
        bool prim_rope_fused_sincos = true;          // LLAMINAR_ATTN_PRIM_ROPE_FUSED_SINCOS (gate libmvec sincosf usage)
        int prim_rope_recurrence_threshold = 300000; // LLAMINAR_ATTN_PRIM_ROPE_RECURRENCE_THRESHOLD (elements = heads*seq_len*head_dim). Tuned from empirical sweep: recurrence under-performs below ~300k elems, wins at 640x8x64 (327,680) but is noisy at 896.
        bool prim_rope_disable_recurrence = false;   // LLAMINAR_ATTN_PRIM_ROPE_DISABLE_RECURRENCE (force old per-pos trig path)
        bool prim_rope_trace = false;                // LLAMINAR_ATTN_PRIM_ROPE_TRACE (emit instrumentation for RoPE path selection)
    };

    struct EmbeddingEnv
    {
        bool trace = false;          // LLAMINAR_EMBED_TRACE
        bool fail_fast = false;      // LLAMINAR_EMBED_FAIL_FAST
        int trace_tokens = 2;        // LLAMINAR_EMBED_TRACE_TOKENS
        int trace_dims = 8;          // LLAMINAR_EMBED_TRACE_DIMS
        std::string trace_rows_spec; // LLAMINAR_EMBED_TRACE_ROWS (pipeline)
    };

    struct RMSNormEnv
    {
        bool validate_ref = false;        // LLAMINAR_RMSNORM_VALIDATE_REF
        bool dump_gamma = false;          // LLAMINAR_RMSNORM_DUMP_GAMMA
        bool force_unit_gamma = false;    // LLAMINAR_RMSNORM_FORCE_UNIT_GAMMA
        bool gamma_checksum = false;      // LLAMINAR_RMSNORM_GAMMA_CHECKSUM (kernel)
        std::string trace_rows_spec;      // LLAMINAR_RMSNORM_TRACE_ROWS
        bool verbose = false;             // LLAMINAR_RMSNORM_VERBOSE (extra row-sum and stats logging)
        bool force_scalar = false;        // LLAMINAR_RMSNORM_FORCE_SCALAR (override parallel heuristics)
        bool disable_tls_scratch = false; // LLAMINAR_RMSNORM_DISABLE_TLS_SCRATCH (revert to per-call allocs for A/B)
        int scratch_prealloc_rows = 0;    // LLAMINAR_RMSNORM_SCRATCH_PREALLOC_ROWS (>0 pre-reserves TLS scratch capacity)
        bool false_sharing_probe = false; // LLAMINAR_RMSNORM_FALSE_SHARING_PROBE (bench harness extra test)
        bool fast_accumulate = false;     // LLAMINAR_RMSNORM_FAST_ACC (accumulate in float then widen)
        int vec_impl = 0;                 // LLAMINAR_RMSNORM_VEC_IMPL (0=auto,1=scalar,2=avx2,3=avx512)
    };

    struct SoftmaxEnv
    {
        bool force_scalar = false;               // LLAMINAR_SOFTMAX_FORCE_SCALAR (disable OpenMP and SIMD hints)
        bool validate = false;                   // LLAMINAR_SOFTMAX_VALIDATE (optional reference compare hooks)
        int parallel_row_threshold = 0;          // LLAMINAR_SOFTMAX_PARALLEL_ROW_THRESHOLD (min rows to parallelize)
        int parallel_elems_threshold = 32768;    // LLAMINAR_SOFTMAX_PARALLEL_ELEMS (rows*cols threshold)
        int validate_scalar_row_threshold = 256; // LLAMINAR_SOFTMAX_VALIDATE_SCALAR_ROW_THRESHOLD (rows below this use scalar validation)
        bool causal_fuse = true;                 // LLAMINAR_SOFTMAX_CAUSAL_FUSE (allow masking during max/sum passes)
        bool fast_exp = false;                   // LLAMINAR_SOFTMAX_FAST_EXP (enable polynomial / approx exp)
        int fast_exp_mode = 0;                   // LLAMINAR_SOFTMAX_FAST_EXP_MODE (reserved for future variants)
        bool dist_recompute = false;             // LLAMINAR_SOFTMAX_DIST_RECOMPUTE (recompute exp in normalize pass to save memory traffic)
        int dist_recompute_threshold = 0;        // LLAMINAR_SOFTMAX_DIST_RECOMPUTE_THRESHOLD (auto-enable recompute if rows*cols >= this and dist_recompute not set)
        // Validation extended controls (active only if validate && fast_exp)
        int validate_sample_rows = 4;       // LLAMINAR_SOFTMAX_VALIDATE_SAMPLE_ROWS (<=0 => all rows)
        double validate_rel_l2_tol = 2e-5;  // LLAMINAR_SOFTMAX_VALIDATE_REL_L2
        double validate_max_abs_tol = 1e-6; // LLAMINAR_SOFTMAX_VALIDATE_MAX_ABS
        bool validate_abort = false;        // LLAMINAR_SOFTMAX_VALIDATE_ABORT (abort/fail if tolerance exceeded)
    };

    struct SwiGLUEnv
    {
        bool validate = false; // LLAMINAR_SWIGLU_VALIDATE
        std::string algo;      // LLAMINAR_SWIGLU_ALGO
    };

    struct LinearEnv
    {
        bool diag = false;
    };
    struct LoaderEnv
    {
        bool log_eps = false;               // LLAMINAR_LOG_EPS
        bool model_load_debug = false;      // LLAMINAR_MODEL_LOAD_DEBUG (non-zero => enable)
        bool model_compare_gguf = false;    // LLAMINAR_MODEL_COMPARE_GGUF
        bool enum_map_debug = false;        // LLAMINAR_ENUM_MAP_DEBUG
        long long shard_cache_max_mb = 512; // LLAMINAR_SHARD_CACHE_MAX_MB (0 disables)
    };

    // --- Phase 2 additional groups ---
    struct AblationEnv
    {
        bool ablate_attention = false;
        bool ablate_ffn = false;
    };
    struct LayerCaptureEnv
    {
        bool capture = false;
        std::string tokens_spec;
        std::vector<int> tokens;
    };
    struct RMSForensicsEnv
    {
        bool enabled = false;
        double warn_rel_l2 = 1e-5;
        bool trace_vectors = false;
        bool diff_only = false;
        std::string layers_spec;
        std::string rows_spec;
        std::vector<int> layers;
        std::vector<int> rows;
    };
    struct PrefillDebugEnv
    {
        bool trace_io = false;
        bool debug_compare = false;
        bool debug_attention = false;
        bool debug_output = false;
    };
    struct EmbeddingDiagEnv
    {
        bool parity = false;
    }; // parity rows reuse LayerCaptureEnv tokens
    struct LogitDiagEnv
    {
        bool dot_check = false;
        std::string dot_check_spec;
        bool dot_dump = false;
        bool dot_prenorm = false;
    };
    struct OutputNormEnv
    {
        bool bypass = false;
        bool force_unit = false;
        bool force_unit_all = false;
        bool clamp = false;
    };
    struct LMHeadEnv
    {
        bool raw_orientation = false;
        bool cosine_diag = false;
    };
    struct CosmaCaptureEnv
    {
        bool capture_last_gemm = false;
        int capture_sample_dim = 0;
        int capture_depth = 0;
        bool dump_stats = false;
        std::string dump_stats_path;
        bool dump_gemm_snapshots = false;
        std::string dump_gemm_snapshots_path;
    };

    // Baseline capture/compare for prefill reference snapshots
    struct BaselineEnv
    {
        bool capture = false;
        bool compare = false;
    };

    // FFN shard trace (prefill) consolidating multi-env configuration
    struct FFNShardTraceEnv
    {
        bool enabled = false;
        bool match_all = false;
        int limit = 1;
        std::string shards_spec;
        std::string rows_spec;
        std::string cols_spec;
        std::vector<int> rows;
        std::vector<int> cols;
    };

    // Fused RMSNorm / QKV fused path diagnostics
    struct RMSFusedEnv
    {
        bool forensics = false;
        int rows_preview = 2;
        int cols_preview = 16;
        bool dump_layer = false;
        std::string dump_layer_spec;
        bool eps_override_active = false;
        double eps_override = 0.0;
    };

    // General TP policy (separate from simulation executor flags)
    struct TPPolicyEnv
    {
        int force_blas_threads = 0;     // LLAMINAR_TP_FORCE_BLAS_THREADS (>0 overrides per partition threads)
        int max_blas_threads = 0;       // LLAMINAR_TP_MAX_BLAS_THREADS (cap)
        int seq_len_hint = 0;           // LLAMINAR_SEQ_LEN_HINT (guides small decode heuristics)
        bool outer_parallel = false;    // LLAMINAR_TP_OUTER_PARALLEL (request outer parallel loop)
        bool disable_outer_par = false; // LLAMINAR_TP_DISABLE_OUTER_PAR (force disable outer loop)
    };

    struct LoggerEnv
    {
        size_t buffer_lines_override = 0; // LLAMINAR_LOG_BUFFER_LINES (>0 => override ring size)
    };

    struct MLPTPEnv
    {
        bool enable = false;       // LLAMINAR_TP_MLP_ENABLE
        int partitions = 1;        // LLAMINAR_TP_MLP_SIZE
        bool validate = false;     // LLAMINAR_TP_MLP_VALIDATE (run parity check for small shapes)
        bool force_column = false; // LLAMINAR_TP_MLP_FORCE_COLUMN (override auto)
        bool force_row = false;    // LLAMINAR_TP_MLP_FORCE_ROW
    };

    // Embedding warnings
    struct EmbeddingWarnEnv
    {
        bool transpose_warn = false;
    };

    // Test / harness controls
    struct TestHarnessEnv
    {
        bool skip_mpi_in_single_test = false;
    };

    // Global logging controls
    struct LoggingEnv
    {
        bool log_level_active = false;
        std::string log_level;
    }; // LLAMINAR_LOG_LEVEL

    struct DistributionEnvConfig
    {
        bool force_replicated = false;        // LLAMINAR_FORCE_REPLICATED
        bool force_sharded = false;           // LLAMINAR_FORCE_SHARDED
        std::string distribution_mode;        // LLAMINAR_DISTRIBUTION_MODE ("replicated"|"sharded"|"")
        int param_threshold_billions = 32;    // LLAMINAR_SHARDING_PARAM_THRESHOLD (billions)
        double model_mem_fraction_max = 0.55; // LLAMINAR_MODEL_MEM_FRACTION_MAX
    };

    struct PerformanceEnv
    {
        bool enable = false;          // LLAMINAR_PERF_ENABLE
        bool log_each_matmul = false; // LLAMINAR_PERF_LOG_EACH_MATMUL
        int log_rank = 0;             // LLAMINAR_PERF_LOG_RANK (only rank prints per-op)
        bool layer_mlp = false;       // LLAMINAR_PERF_LAYER_MLP (emit per-MLP layer timing breakdown)
        bool layer_verbose = false;   // LLAMINAR_PERF_LAYER_VERBOSE (include gather/parity & extra fields)
        bool layer_attention = false; // LLAMINAR_PERF_LAYER_ATTENTION (emit per-Attention layer timing)
    };

    // ---------------------------------------------------------------------
    // Phase 0 Weight Slicing Infrastructure
    // ---------------------------------------------------------------------
    // Lightweight role registry for model weight matrices. This will evolve
    // as slicing phases add coverage (initial attention Q/K/V + MLP W1/W2/W3).
    // Keeping the enum here avoids scattering role string heuristics across
    // loader and slicing policy code. Future additions (e.g., RMSNormGamma,
    // RopeFreq, ClassifierHead) can extend this safely – only append new
    // values to preserve stable underlying integers if any serialization
    // later appears.
    enum class WeightRole : uint8_t
    {
        Unknown = 0,
        W_Q,      // Attention query projection
        W_K,      // Attention key projection
        W_V,      // Attention value projection
        W_O,      // Attention output projection
        W1,       // FFN first (gate) / up projection (model specific naming)
        W2,       // FFN down / output projection
        W3,       // FFN secondary (SwiGLU) projection (if present)
        Embedding // Token embedding matrix
    };

    // Convert role enum to canonical short string (stable identifiers).
    const char *weightRoleToString(WeightRole r);
    // Parse a (lower-cased) string into a WeightRole. Accepts a small set of
    // common aliases so that loader naming differences do not derail early
    // experimentation. Unrecognized => WeightRole::Unknown.
    WeightRole weightRoleFromString(const std::string &name);

    struct WeightSlicingEnv
    {
        bool disable = false;  // LLAMINAR_DISABLE_WEIGHT_SHARDING : hard off switch
        bool force = false;    // LLAMINAR_FORCE_WEIGHT_SHARDING   : force enable even if heuristics say no
        bool validate = false; // LLAMINAR_WEIGHT_SLICE_VALIDATE   : run post-slice parity checks
        int min_cols = 0;      // LLAMINAR_WEIGHT_SLICE_MIN_COLS   : skip slicing if global cols < this
        // Reserved / future:
        // bool verbose = false;  // Potential future detailed logging toggle
    };

    // Lightweight global counters & per-weight stats for slicing (Phase 1 observability)
    struct WeightSlicingCounters
    {
        uint64_t captured = 0;       // weights captured for parity
        uint64_t sliced = 0;         // weights actually sliced
        uint64_t validated_ok = 0;   // validations passed
        uint64_t validated_fail = 0; // validations failed
        struct PerWeightStat
        {
            std::string name;
            WeightRole role;
            int rows;
            int cols_global;
            int cols_local;
        };
        std::vector<PerWeightStat> per_weight; // only rank 0 required for summary; still filled for simplicity
    };

    // Accessor for mutable counters (not thread-safe; model load is single-threaded today)
    WeightSlicingCounters &weightSlicingCounters();

    struct DebugEnvSnapshot
    {
        ShardingEnv sharding;
        CosmaEnv cosma;
        PipelineEnv pipeline;
        DequantEnv dequant;
        AdaptiveEnv adaptive;
        AttentionEnv attention;
        EmbeddingEnv embedding;
        RMSNormEnv rmsnorm;
        SwiGLUEnv swiglu;
        LinearEnv linear;
        LoaderEnv loader;
        AblationEnv ablation;
        LayerCaptureEnv layer_capture;
        RMSForensicsEnv rms_forensics;
        PrefillDebugEnv prefill_debug;
        EmbeddingDiagEnv embedding_diag;
        LogitDiagEnv logit;
        OutputNormEnv output_norm;
        LMHeadEnv lm_head;
        CosmaCaptureEnv cosma_capture;
        BaselineEnv baseline;
        FFNShardTraceEnv ffn_shard_trace;
        RMSFusedEnv rms_fused;
        EmbeddingWarnEnv embedding_warn;
        TestHarnessEnv test_harness;
        LoggingEnv logging;
        DistributionEnvConfig distribution; // new group for high-level mode selection
        PerformanceEnv performance;         // performance instrumentation controls
        WeightSlicingEnv weight_slicing;    // phase 0: foundational weight slicing flags
        TPPolicyEnv tp_policy;              // tensor parallel policy settings
        LoggerEnv logger;                   // logger ring buffer sizing
        MLPTPEnv mlp_tp;                    // MLP tensor parallel execution controls
        KVCacheEnv kv_cache;                // KV cache dynamic capacity controls
        SoftmaxEnv softmax;                 // softmax core execution tuning
        struct ThreadingEnv
        {                                 // Global OpenMP / threading policy
            bool use_physical = false;    // LLAMINAR_OMP_USE_PHYSICAL (if set => restrict to physical cores per rank)
            int force_threads = 0;        // LLAMINAR_OMP_FORCE (if >0 overrides everything)
            bool configured = false;      // internal: whether we already applied global policy
            bool bind_per_socket = false; // LLAMINAR_BIND_PER_SOCKET (if set => bind each rank to one socket's primary cores)
        } threading;
        struct TPSimEnv
        {                        // Tensor Parallel simulation for local W/O executor
            bool enable = false; // LLAMINAR_TP_WO_SIM_ENABLE (gates simulation mode)
            int partitions = 0;  // LLAMINAR_TP_WO_SIM_PARTITIONS (>0 => shard count)
            enum class Mode
            {
                Auto = 0,
                Row,
                Col
            } mode = Mode::Auto; // LLAMINAR_TP_WO_SIM_MODE=row|col|auto
        } tp_sim;
    };

    // Accessor (lazy init, thread-safe via magic statics)
    const DebugEnvSnapshot &debugEnv();
    // Utility to format a concise multi-line summary of active debug env groups
    // Implemented in debug_env.cpp
    std::vector<std::string> formatDebugEnvSummary(const DebugEnvSnapshot &s);

    /**
     * @brief Refresh the cached debug environment snapshot.
     *
     * The project guidelines mandate avoiding repeated getenv calls in hot paths
     * by capturing a single immutable snapshot on first access. Certain tests
     * (e.g. weight slicing parity) need to mutate environment variables at
     * runtime before constructing model loader state. In CI these tests export
     * variables at process start, but when invoked programmatically via GTest
     * with setenv() calls, the snapshot may already have been materialized by
     * an earlier indirect debugEnv() access (e.g. through another component's
     * static initialization).
     *
     * This function re-parses the environment and replaces the cached snapshot.
     * It should ONLY be used in test code and never inside performance critical
     * inference loops.
     */
    void debugEnvRefresh();

    // Configure a global OpenMP thread policy based on the debug environment and
    // detected topology. Safe to call multiple times; only first successful
    // application per process has effect (idempotent via snapshot.threading.configured).
    // Heuristic:
    // 1. If LLAMINAR_OMP_FORCE>0 => omp_set_num_threads(force)
    // 2. Else if LLAMINAR_OMP_USE_PHYSICAL set => derive physical cores per MPI rank.
    //    a) Detect topology (hyperthreading disabled for core count) and compute
    //       per-rank = physical_cores / mpi_size (>=1).
    //    b) Clamp to at least 1.
    // 3. Else do nothing (respect external OMP_NUM_THREADS or runtime defaults).
    // Logs a single INFO line on rank 0 summarizing decision.
    void configureGlobalOpenMPThreads();

} // namespace llaminar
