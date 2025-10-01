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
        bool capture_pre_lm = false;  // LLAMINAR_PIPELINE_CAPTURE_PRE_LM
        bool layerwise_stats = false; // LLAMINAR_PIPELINE_LAYERWISE_STATS
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
        bool validate_ref = false;     // LLAMINAR_RMSNORM_VALIDATE_REF
        bool dump_gamma = false;       // LLAMINAR_RMSNORM_DUMP_GAMMA
        bool force_unit_gamma = false; // LLAMINAR_RMSNORM_FORCE_UNIT_GAMMA
        bool gamma_checksum = false;   // LLAMINAR_RMSNORM_GAMMA_CHECKSUM (kernel)
        std::string trace_rows_spec;   // LLAMINAR_RMSNORM_TRACE_ROWS
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
    };

    // Accessor (lazy init, thread-safe via magic statics)
    const DebugEnvSnapshot &debugEnv();

    // Produce human-readable summary lines describing enabled / configured flags.
    // Intended for rank 0 startup logging. Each string is a full line (already
    // prefixed with [DebugEnv]). Caller is responsible for choosing log level.
    std::vector<std::string> formatDebugEnvSummary(const DebugEnvSnapshot &snap);

} // namespace llaminar
