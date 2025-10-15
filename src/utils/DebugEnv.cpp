#include "DebugEnv.h"
#include "Logger.h"
#include "TopologyManager.h"
#include <omp.h>
#include <sched.h>
#include <unistd.h>
#ifdef LLAMINAR_HAVE_MPI
#include <mpi.h>
#endif
#include <algorithm>
#include <cstring>
#include <vector>
#include <sstream>

namespace llaminar
{

    static bool flag(const char *v) { return v && *v; }
    static int to_int(const char *v, int d) { return v ? std::atoi(v) : d; }
    static long long to_ll(const char *v, long long d) { return v ? std::atoll(v) : d; }

    // ---------------------------------------------------------------------
    // Weight role helpers (Phase 0 weight slicing infra)
    // ---------------------------------------------------------------------
    const char *weightRoleToString(WeightRole r)
    {
        switch (r)
        {
        case WeightRole::W_Q:
            return "w_q";
        case WeightRole::W_K:
            return "w_k";
        case WeightRole::W_V:
            return "w_v";
        case WeightRole::W_O:
            return "w_o";
        case WeightRole::W1:
            return "w1";
        case WeightRole::W2:
            return "w2";
        case WeightRole::W3:
            return "w3";
        case WeightRole::Embedding:
            return "embedding";
        case WeightRole::Unknown:
        default:
            return "unknown";
        }
    }

    WeightRole weightRoleFromString(const std::string &name)
    {
        if (name.empty())
            return WeightRole::Unknown;
        std::string norm = name;
        for (char &c : norm)
            c = (char)std::tolower(c);
        if (norm == "w_q" || norm == "q" || norm == "query")
            return WeightRole::W_Q;
        if (norm == "w_k" || norm == "k" || norm == "key")
            return WeightRole::W_K;
        if (norm == "w_v" || norm == "v" || norm == "value")
            return WeightRole::W_V;
        if (norm == "w_o" || norm == "o" || norm == "out" || norm == "output")
            return WeightRole::W_O;
        if (norm == "w1" || norm == "ffn_w1" || norm == "gate" || norm == "up")
            return WeightRole::W1;
        if (norm == "w2" || norm == "ffn_w2" || norm == "down" || norm == "proj")
            return WeightRole::W2;
        if (norm == "w3" || norm == "ffn_w3")
            return WeightRole::W3;
        if (norm == "embedding" || norm == "embed" || norm == "tok_embedding")
            return WeightRole::Embedding;
        return WeightRole::Unknown;
    }

    // Use indirection so we can refresh in tests by replacing the pointed snapshot.
    static DebugEnvSnapshot *g_snapshot = nullptr;

    const DebugEnvSnapshot &debugEnv()
    {
        if (!g_snapshot)
        {
            g_snapshot = new DebugEnvSnapshot([]()
                                              {
        DebugEnvSnapshot s;
        // Sharding
        s.sharding.debug_materialize_attention = flag(std::getenv("LLAMINAR_DEBUG_MATERIALIZE_ATTENTION"));
        s.sharding.dump_shards = flag(std::getenv("LLAMINAR_DUMP_SHARDS"));
        s.sharding.dump_shards_n = to_int(std::getenv("LLAMINAR_DUMP_SHARDS_N"), 16);
        if(s.sharding.dump_shards_n <= 0 || s.sharding.dump_shards_n > 8192) s.sharding.dump_shards_n = 16;
        s.sharding.shard_parity_check = flag(std::getenv("LLAMINAR_SHARD_PARITY_CHECK"));
        s.sharding.assert_replicated_misuse = flag(std::getenv("LLAMINAR_ASSERT_REPLICATED_MISUSE"));
    s.sharding.shard_load_diag = flag(std::getenv("LLAMINAR_SHARD_LOAD_DIAG"));
        // COSMA
        s.cosma.prefill_threshold = to_int(std::getenv("LLAMINAR_COSMA_PREFILL_THRESHOLD"), 4096);
        s.cosma.fast_path_threshold = to_int(std::getenv("LLAMINAR_COSMA_FAST_PATH_THRESHOLD"), -1);
        s.cosma.validate_tile = to_int(std::getenv("LLAMINAR_COSMA_VALIDATE_TILE"), 0);
        s.cosma.log_level = to_int(std::getenv("LLAMINAR_COSMA_LOG_LEVEL"), 2);
        // (weight role helpers defined at namespace scope above)
        s.cosma.max_resident_mb = to_ll(std::getenv("LLAMINAR_COSMA_MAX_RESIDENT_MB"), 2048);
        s.cosma.disable = flag(std::getenv("LLAMINAR_COSMA_DISABLE"));
        s.cosma.force = flag(std::getenv("LLAMINAR_COSMA_FORCE"));
        s.cosma.diag = flag(std::getenv("LLAMINAR_COSMA_DIAG"));
        s.cosma.diag_deep = flag(std::getenv("LLAMINAR_COSMA_DIAG_DEEP"));
        s.cosma.diag_axis = flag(std::getenv("LLAMINAR_COSMA_DIAG_AXIS"));
        s.cosma.compare_replicated = flag(std::getenv("LLAMINAR_COSMA_COMPARE_REPLICATED"));
        s.cosma.debug_recon = flag(std::getenv("LLAMINAR_COSMA_DEBUG_RECON"));
        s.cosma.force_direct = flag(std::getenv("LLAMINAR_COSMA_FORCE_DIRECT"));
        s.cosma.force_replicated = flag(std::getenv("LLAMINAR_COSMA_FORCE_REPLICATED"));
        s.cosma.force_replicated_diag = flag(std::getenv("LLAMINAR_COSMA_FORCE_REPLICATED_DIAG"));
        s.cosma.fast_unverified = flag(std::getenv("LLAMINAR_COSMA_FAST_UNVERIFIED"));
        s.cosma.auto_fix_transpose = flag(std::getenv("LLAMINAR_COSMA_AUTO_FIX_TRANSPOSE"));
    s.cosma.force_fallback = flag(std::getenv("LLAMINAR_COSMA_FORCE_FALLBACK"));
    s.cosma.force_distributed_act = flag(std::getenv("LLAMINAR_COSMA_FORCE_DISTRIBUTED_ACT"));
    s.cosma.disable_fused_dequant = flag(std::getenv("LLAMINAR_COSMA_DISABLE_FUSED_DEQUANT"));
        s.cosma.diag_coord_invert = flag(std::getenv("LLAMINAR_COSMA_DIAG_COORD_INVERT"));
        s.cosma.diag_local_probe = flag(std::getenv("LLAMINAR_COSMA_DIAG_LOCAL_PROBE"));
        s.cosma.diag_local_probe_deep = flag(std::getenv("LLAMINAR_COSMA_DIAG_LOCAL_PROBE_DEEP"));
        s.cosma.diag_recon_bypass = flag(std::getenv("LLAMINAR_COSMA_DIAG_RECON_BYPASS"));
        s.cosma.diag_recon_transpose = flag(std::getenv("LLAMINAR_COSMA_DIAG_RECON_TRANSPOSE"));
        s.cosma.diag_recon_brute = flag(std::getenv("LLAMINAR_COSMA_DIAG_RECON_BRUTE"));
        s.cosma.diag_recon_map = flag(std::getenv("LLAMINAR_COSMA_DIAG_RECON_MAP"));
        s.cosma.recon_force_legacy = flag(std::getenv("LLAMINAR_COSMA_RECON_FORCE_LEGACY"));
        s.cosma.diag_swaprc = flag(std::getenv("LLAMINAR_COSMA_DIAG_SWAPRC"));
        s.cosma.diag_try_transpose = flag(std::getenv("LLAMINAR_COSMA_DIAG_TRY_TRANSPOSE"));
        s.cosma.diag_skip_norm = flag(std::getenv("LLAMINAR_COSMA_DIAG_SKIP_NORM"));
        s.cosma.diag_dump_small = flag(std::getenv("LLAMINAR_COSMA_DUMP_SMALL"));
        s.cosma.pop_forward_legacy = flag(std::getenv("LLAMINAR_COSMA_POP_FORWARD_LEGACY"));
        s.cosma.replicate_B = flag(std::getenv("LLAMINAR_COSMA_REPLICATE_B"));
        s.cosma.force_unified_strategy = flag(std::getenv("LLAMINAR_COSMA_FORCE_UNIFIED"));
        s.cosma.overlap_enabled = flag(std::getenv("LLAMINAR_COSMA_OVERLAP_STREAM"));
        s.cosma.overlap_verbose = flag(std::getenv("LLAMINAR_COSMA_OVERLAP_VERBOSE"));
        s.cosma.preflight_disable = flag(std::getenv("LLAMINAR_COSMA_PREFLIGHT_DISABLE"));
    // Attention decode instrumentation (initial snapshot parse)
    s.attention_decode.decode_diag = flag(std::getenv("LLAMINAR_ATTENTION_DECODE_DIAG"));
    s.attention_decode.dump_full_qkv = flag(std::getenv("LLAMINAR_ATTENTION_DECODE_DUMP_QKV"));
    if(const char* dl0 = std::getenv("LLAMINAR_ATTENTION_DECODE_DUMP_LIMIT")) { int v=std::atoi(dl0); if(v>0 && v<=4096) s.attention_decode.dump_limit = v; }
        s.cosma.rmsnorm_validate = flag(std::getenv("LLAMINAR_COSMA_RMSNORM_VALIDATE"));
        s.cosma.rmsnorm_trace = flag(std::getenv("LLAMINAR_COSMA_RMSNORM_TRACE"));
        if(const char* pts = std::getenv("LLAMINAR_COSMA_RMSNORM_TRACE_POINTS")){
            if(*pts){ s.cosma.rmsnorm_trace_points_active = true; s.cosma.rmsnorm_trace_points_spec = pts; }
        }
        if(const char* direct = std::getenv("LLAMINAR_COSMA_DIRECT_THRESHOLD_OPS")){
            long long val = std::atoll(direct); if(val>0){ s.cosma.direct_threshold_override = true; s.cosma.direct_threshold_ops = val; }
        }
        if(const char* tap = std::getenv("LLAMINAR_COSMA_DIAG_TAP")){
            s.cosma.diag_tap_enabled = true; s.cosma.diag_tap_value = std::max(1, std::atoi(tap));
        }
        if(const char* perm = std::getenv("LLAMINAR_COSMA_DIAG_PERM_INFER")){
            if(*perm){ s.cosma.diag_perm_infer_active = true; s.cosma.diag_perm_spec = perm; }
        }
        if(const char* samples = std::getenv("LLAMINAR_COSMA_DIAG_SAMPLES")){
            if(*samples){ s.cosma.diag_samples_active = true; s.cosma.diag_samples_spec = samples; }
        }
        if(const char* safety = std::getenv("LLAMINAR_COSMA_PREFLIGHT_SAFETY_FACTOR")){
            try { s.cosma.preflight_safety = std::stod(safety); s.cosma.preflight_safety = std::max(0.5, std::min(4.0, s.cosma.preflight_safety)); s.cosma.preflight_safety_override = true; }
            catch(...) { s.cosma.preflight_safety = 1.2; }
        }
        if(const char* thr = std::getenv("LLAMINAR_OPENBLAS_THREADS")){
            s.cosma.forced_openblas_threads = std::max(1, std::atoi(thr));
        } else if(const char* thr2 = std::getenv("OPENBLAS_NUM_THREADS")){
            s.cosma.forced_openblas_threads = std::max(1, std::atoi(thr2));
        }
        if(const char* rthr = std::getenv("LLAMINAR_COSMA_FORCE_REPLICATED_THREADS")){
            s.cosma.forced_replicated_threads = std::max(1, std::atoi(rthr));
        }
        // Pipeline
        s.pipeline.capture_pre_lm = flag(std::getenv("LLAMINAR_PIPELINE_CAPTURE_PRE_LM"));
        s.pipeline.layerwise_stats = flag(std::getenv("LLAMINAR_PIPELINE_LAYERWISE_STATS"));
    s.pipeline.enable_abstract_pipeline = flag(std::getenv("LLAMINAR_ENABLE_ABSTRACT_PIPELINE"));
    s.pipeline.disable_incremental_decode = flag(std::getenv("LLAMINAR_DISABLE_INCREMENTAL_DECODE"));
    s.pipeline.layer_token_diff = flag(std::getenv("LLAMINAR_PIPELINE_LAYER_TOKEN_DIFF"));
    s.pipeline.layer_token_diff_verbose = flag(std::getenv("LLAMINAR_PIPELINE_LAYER_TOKEN_DIFF_VERBOSE"));
    s.pipeline.attn_ref_compare = flag(std::getenv("LLAMINAR_DEBUG_ATTENTION_REF"));
    s.pipeline.layer_replay_compare = flag(std::getenv("LLAMINAR_DEBUG_LAYER_REPLAY_COMPARE"));
    // Pre-LM row diff instrumentation (explicit opt-in; unset or "0" keeps it off)
    s.pipeline.pre_lm_row_diff = flag(std::getenv("LLAMINAR_PIPELINE_PRE_LM_ROW_DIFF"));
    // Incremental decode instrumentation controls
    s.pipeline.incr_trace = flag(std::getenv("LLAMINAR_PIPELINE_INCR_TRACE"));
    s.pipeline.incr_cache_trace = flag(std::getenv("LLAMINAR_PIPELINE_INCR_CACHE_TRACE"));
    s.pipeline.incr_hidden_trace = flag(std::getenv("LLAMINAR_PIPELINE_INCR_HIDDEN_TRACE"));
    s.pipeline.debug_decode_embed = flag(std::getenv("LLAMINAR_DEBUG_DECODE_EMBED"));
    // FFN fusion: default ON, can disable with LLAMINAR_FFN_FUSION=0
    {
        const char *ffn_fusion_env = std::getenv("LLAMINAR_FFN_FUSION");
        s.pipeline.ffn_fusion_enabled = ffn_fusion_env ? (std::atoi(ffn_fusion_env) != 0) : true;
    }
    // KV cache
    s.kv_cache.dynamic_init = flag(std::getenv("LLAMINAR_KV_DYNAMIC_INIT"));
    if(const char* gfac = std::getenv("LLAMINAR_KV_GROWTH_FACTOR")) { int v=std::atoi(gfac); if(v>=1 && v<=16) s.kv_cache.growth_factor = v; }
        // Dequant
        s.dequant.stats = flag(std::getenv("LLAMINAR_DEQUANT_STATS"));
        s.dequant.anomalies = flag(std::getenv("LLAMINAR_DEQUANT_ANOMALIES"));
    // Adaptive
    s.adaptive.disable_cosma = flag(std::getenv("ADAPTIVE_DISABLE_COSMA"));
    s.adaptive.log_threading = flag(std::getenv("LLAMINAR_LOG_THREADING"));
    // Attention
    s.attention.validate_primitives = flag(std::getenv("LLAMINAR_ATTN_PRIMITIVES_VALIDATE"));
    s.attention.validate_output = flag(std::getenv("LLAMINAR_ATTN_OUTPUT_VALIDATE"));
    if(const char* up = std::getenv("LLAMINAR_ATTN_USE_PRIMITIVES")) {
        // empty string => treat as true; explicit 0 disables
        if(*up == '0') s.attention.use_primitives = false; else s.attention.use_primitives = true;
    }
    if(const char* om = std::getenv("LLAMINAR_ATTN_OUTPUT_MODE")) { if(*om){ s.attention.output_mode_forced=true; s.attention.output_mode = om; }}
    if(const char* gt = std::getenv("LLAMINAR_ATTN_GATHER_THRESHOLD")) { s.attention.gather_threshold = std::atoi(gt); }
    s.attention.force_scalar = flag(std::getenv("LLAMINAR_ATTN_FORCE_SCALAR"));
    s.attention.validate_proj = flag(std::getenv("LLAMINAR_ATTN_VALIDATE_PROJ"));
    s.attention.verbose = flag(std::getenv("LLAMINAR_ATTN_VERBOSE"));
    s.attention.micro_trace = flag(std::getenv("LLAMINAR_ATTN_MICRO_TRACE"));
    s.attention.trace_weight_slicing = flag(std::getenv("LLAMINAR_ATTN_TRACE_WEIGHT_SLICE"));
    s.attention.trace_k_projection = flag(std::getenv("LLAMINAR_ATTN_TRACE_K_PROJECTION"));
    s.attention.dump_attention = flag(std::getenv("LLAMINAR_ATTN_DUMP_ATTENTION"));
    s.attention.tp_disable = flag(std::getenv("LLAMINAR_ATTN_TP_DISABLE"));
    if(const char* tpp = std::getenv("LLAMINAR_ATTN_TP_PARTITIONS")) { int v=std::atoi(tpp); if(v>0 && v<1024) s.attention.tp_partitions = v; }
    s.attention.tp_auto = flag(std::getenv("LLAMINAR_ATTN_TP_AUTO"));
    s.attention.tp_force_splitter = flag(std::getenv("LLAMINAR_ATTN_TP_FORCE_SPLITTER"));
    s.attention.internal_diff = flag(std::getenv("LLAMINAR_ATTN_INTERNAL_DIFF"));
    // Attention primitive optimization flags
    if(const char* ppe = std::getenv("LLAMINAR_ATTN_PRIM_PARALLEL_ELEMS")) { int v=std::atoi(ppe); if(v>0) s.attention.prim_parallel_elems_threshold = v; }
    s.attention.prim_force_scalar = flag(std::getenv("LLAMINAR_ATTN_PRIM_FORCE_SCALAR"));
    if(const char* frt = std::getenv("LLAMINAR_ATTN_PRIM_FUSED_RECOMPUTE_THRESHOLD")) { int v=std::atoi(frt); if(v>0) s.attention.prim_fused_recompute_threshold = v; }
    s.attention.prim_force_fused = flag(std::getenv("LLAMINAR_ATTN_PRIM_FORCE_FUSED"));
    s.attention.prim_disable_fused = flag(std::getenv("LLAMINAR_ATTN_PRIM_DISABLE_FUSED"));
    {
        const char* rv = std::getenv("LLAMINAR_ATTN_PRIM_ROPE_VECTORIZE");
        if(rv && *rv=='0') s.attention.prim_rope_vectorize = false; // default true
    }
    {
        const char* rfs = std::getenv("LLAMINAR_ATTN_PRIM_ROPE_FUSED_SINCOS");
        if(rfs && *rfs=='0') s.attention.prim_rope_fused_sincos = false; // default true
    }
    {
        if(const char* rt = std::getenv("LLAMINAR_ATTN_PRIM_ROPE_RECURRENCE_THRESHOLD")) { int v=std::atoi(rt); if(v>0) s.attention.prim_rope_recurrence_threshold = v; }
        if(flag(std::getenv("LLAMINAR_ATTN_PRIM_ROPE_DISABLE_RECURRENCE"))) s.attention.prim_rope_disable_recurrence = true;
        s.attention.prim_rope_trace = flag(std::getenv("LLAMINAR_ATTN_PRIM_ROPE_TRACE"));
    }
    // Embedding
    s.embedding.trace = flag(std::getenv("LLAMINAR_EMBED_TRACE"));
    s.embedding.fail_fast = flag(std::getenv("LLAMINAR_EMBED_FAIL_FAST"));
    if(const char* tt = std::getenv("LLAMINAR_EMBED_TRACE_TOKENS")){ int v=std::atoi(tt); if(v>0&&v<=4096) s.embedding.trace_tokens=v; }
    if(const char* td = std::getenv("LLAMINAR_EMBED_TRACE_DIMS")){ int v=std::atoi(td); if(v>0&&v<=65536) s.embedding.trace_dims=v; }
    if(const char* tr = std::getenv("LLAMINAR_EMBED_TRACE_ROWS")){ if(*tr) s.embedding.trace_rows_spec = tr; }
    // RMSNorm
    s.rmsnorm.validate_ref = flag(std::getenv("LLAMINAR_RMSNORM_VALIDATE_REF"));
    s.rmsnorm.dump_gamma = flag(std::getenv("LLAMINAR_RMSNORM_DUMP_GAMMA"));
    s.rmsnorm.force_unit_gamma = flag(std::getenv("LLAMINAR_RMSNORM_FORCE_UNIT_GAMMA"));
    s.rmsnorm.gamma_checksum = flag(std::getenv("LLAMINAR_RMSNORM_GAMMA_CHECKSUM"));
    if(const char* rr = std::getenv("LLAMINAR_RMSNORM_TRACE_ROWS")){ if(*rr) s.rmsnorm.trace_rows_spec = rr; }
    s.rmsnorm.verbose = flag(std::getenv("LLAMINAR_RMSNORM_VERBOSE"));
    s.rmsnorm.force_scalar = flag(std::getenv("LLAMINAR_RMSNORM_FORCE_SCALAR"));
    s.rmsnorm.disable_tls_scratch = flag(std::getenv("LLAMINAR_RMSNORM_DISABLE_TLS_SCRATCH"));
    if(const char* sp = std::getenv("LLAMINAR_RMSNORM_SCRATCH_PREALLOC_ROWS")) { int v=std::atoi(sp); if(v>0 && v<1000000) s.rmsnorm.scratch_prealloc_rows = v; }
    s.rmsnorm.false_sharing_probe = flag(std::getenv("LLAMINAR_RMSNORM_FALSE_SHARING_PROBE"));
    s.rmsnorm.fast_accumulate = flag(std::getenv("LLAMINAR_RMSNORM_FAST_ACC"));
    if(const char* rvi = std::getenv("LLAMINAR_RMSNORM_VEC_IMPL")) { int v=std::atoi(rvi); if(v>=0 && v<=3) s.rmsnorm.vec_impl = v; }
    {
        const char* lg = std::getenv("LLAMINAR_RMSNORM_LEGACY_GLOBAL_STATS");
        if(lg){
            // default true; explicit 0/false/off disables legacy global sequence-length dependent stats aggregation
            if(*lg=='0') s.rmsnorm.legacy_global_stats = false; else s.rmsnorm.legacy_global_stats = true;
        }
    }

        // Softmax core
        s.softmax.force_scalar = flag(std::getenv("LLAMINAR_SOFTMAX_FORCE_SCALAR"));
        s.softmax.validate = flag(std::getenv("LLAMINAR_SOFTMAX_VALIDATE"));
        s.softmax.parallel_row_threshold = to_int(std::getenv("LLAMINAR_SOFTMAX_PARALLEL_ROW_THRESHOLD"), 0);
        s.softmax.parallel_elems_threshold = to_int(std::getenv("LLAMINAR_SOFTMAX_PARALLEL_ELEMS"), 32768);
        s.softmax.validate_scalar_row_threshold = to_int(std::getenv("LLAMINAR_SOFTMAX_VALIDATE_SCALAR_ROW_THRESHOLD"), 256);
        {
            const char* cf = std::getenv("LLAMINAR_SOFTMAX_CAUSAL_FUSE");
            if(cf && *cf == '0') s.softmax.causal_fuse = false; // default true
        }
        s.softmax.fast_exp = flag(std::getenv("LLAMINAR_SOFTMAX_FAST_EXP"));
        if(const char* fem = std::getenv("LLAMINAR_SOFTMAX_FAST_EXP_MODE")) { int v=std::atoi(fem); if(v>=0 && v<=4) s.softmax.fast_exp_mode = v; }
        s.softmax.dist_recompute = flag(std::getenv("LLAMINAR_SOFTMAX_DIST_RECOMPUTE"));
    if(const char* drt = std::getenv("LLAMINAR_SOFTMAX_DIST_RECOMPUTE_THRESHOLD")) { int v=std::atoi(drt); if(v>0) s.softmax.dist_recompute_threshold = v; }
        if(const char* vsr = std::getenv("LLAMINAR_SOFTMAX_VALIDATE_SAMPLE_ROWS")) { int v=std::atoi(vsr); s.softmax.validate_sample_rows = v; }
        if(const char* vrl = std::getenv("LLAMINAR_SOFTMAX_VALIDATE_REL_L2")) { try { s.softmax.validate_rel_l2_tol = std::stod(vrl);} catch(...){} }
        if(const char* vma = std::getenv("LLAMINAR_SOFTMAX_VALIDATE_MAX_ABS")) { try { s.softmax.validate_max_abs_tol = std::stod(vma);} catch(...){} }
        s.softmax.validate_abort = flag(std::getenv("LLAMINAR_SOFTMAX_VALIDATE_ABORT"));
    // SwiGLU
    s.swiglu.validate = flag(std::getenv("LLAMINAR_SWIGLU_VALIDATE"));
    if(const char* algo = std::getenv("LLAMINAR_SWIGLU_ALGO")) s.swiglu.algo = algo;
    // Linear
    s.linear.diag = flag(std::getenv("LLAMINAR_LINEAR_DIAG"));
    // Loader
        s.loader.log_eps = flag(std::getenv("LLAMINAR_LOG_EPS"));
        if(const char* mld = std::getenv("LLAMINAR_MODEL_LOAD_DEBUG")) { if(*mld && std::string(mld) != "0") s.loader.model_load_debug = true; }
        s.loader.model_compare_gguf = flag(std::getenv("LLAMINAR_MODEL_COMPARE_GGUF"));
        s.loader.enum_map_debug = flag(std::getenv("LLAMINAR_ENUM_MAP_DEBUG"));
        if(const char* scm = std::getenv("LLAMINAR_SHARD_CACHE_MAX_MB")) {
            long long v = std::atoll(scm);
            if(v >= 0) s.loader.shard_cache_max_mb = v; // keep 0 sentinel (disabled)
        }
        // NUMA-aware allocation (default enabled for multi-socket performance)
        if(const char* nft = std::getenv("LLAMINAR_NUMA_FIRST_TOUCH")) {
            if(*nft == '0') s.loader.numa_first_touch = false; // explicitly disabled
        }
        s.loader.numa_verify_locality = flag(std::getenv("LLAMINAR_NUMA_VERIFY_LOCALITY"));

        // --- Phase 2 parsing additions ---
        // Ablation
        s.ablation.ablate_attention = flag(std::getenv("LLAMINAR_ABLATE_ATTENTION"));
        s.ablation.ablate_ffn       = flag(std::getenv("LLAMINAR_ABLATE_FFN"));
        // Layer capture
        s.layer_capture.capture = flag(std::getenv("LLAMINAR_CAPTURE_LAYER_ACT"));
        if(const char* ls = std::getenv("LLAMINAR_CAPTURE_LAYER_ACT_TOKENS")) if(*ls) s.layer_capture.tokens_spec = ls;
        auto parse_index_spec = [](const std::string &spec)->std::vector<int>{
            std::vector<int> out; if(spec.empty()) return out; size_t pos=0; while(pos<spec.size()){ size_t c=spec.find(',',pos); std::string tok=spec.substr(pos,c==std::string::npos?std::string::npos:c-pos); auto trim=[](std::string &x){ size_t a=x.find_first_not_of(" \t"); size_t b=x.find_last_not_of(" \t"); if(a==std::string::npos){ x.clear(); return;} x=x.substr(a,b-a+1);}; trim(tok); if(!tok.empty()){ size_t dash=tok.find('-'); if(dash!=std::string::npos){ std::string a=tok.substr(0,dash); std::string b=tok.substr(dash+1); trim(a); trim(b); try{ int ia=std::stoi(a); int ib=std::stoi(b); if(ia<=ib){ for(int v=ia; v<=ib; ++v) out.push_back(v);} }catch(...){} } else { try{ out.push_back(std::stoi(tok)); }catch(...){} } } if(c==std::string::npos) break; pos=c+1; } std::sort(out.begin(),out.end()); out.erase(std::unique(out.begin(),out.end()), out.end()); return out; };
        if(!s.layer_capture.tokens_spec.empty()) s.layer_capture.tokens = parse_index_spec(s.layer_capture.tokens_spec);
        // RMS Forensics
        s.rms_forensics.enabled = flag(std::getenv("LLAMINAR_RMS_FORENSICS"));
        if(const char* lspec = std::getenv("LLAMINAR_RMS_FORENSICS_LAYERS")) if(*lspec) s.rms_forensics.layers_spec = lspec;
        if(const char* rspec = std::getenv("LLAMINAR_RMS_FORENSICS_ROWS")) if(*rspec) s.rms_forensics.rows_spec = rspec;
        if(const char* warn  = std::getenv("LLAMINAR_RMS_FORENSICS_REL_L2_WARN")) { try { s.rms_forensics.warn_rel_l2 = std::stod(warn); } catch(...){} }
        s.rms_forensics.trace_vectors = flag(std::getenv("LLAMINAR_RMS_FORENSICS_TRACE"));
        s.rms_forensics.diff_only     = flag(std::getenv("LLAMINAR_RMS_FORENSICS_DIFF_ONLY"));
        if(!s.rms_forensics.layers_spec.empty()) s.rms_forensics.layers = parse_index_spec(s.rms_forensics.layers_spec);
        if(!s.rms_forensics.rows_spec.empty()) s.rms_forensics.rows = parse_index_spec(s.rms_forensics.rows_spec);
        // Prefill COSMA attention debugging (reference / tracing)
        s.prefill_debug.trace_io       = flag(std::getenv("LLAMINAR_COSMA_PREFILL_TRACE_IO"));
        s.prefill_debug.debug_compare  = flag(std::getenv("LLAMINAR_COSMA_PREFILL_DEBUG_COMPARE"));
        s.prefill_debug.debug_attention= flag(std::getenv("LLAMINAR_COSMA_PREFILL_DEBUG_ATTENTION"));
        s.prefill_debug.debug_output   = flag(std::getenv("LLAMINAR_COSMA_PREFILL_DEBUG_OUTPUT"));
        // Embedding parity
        s.embedding_diag.parity = flag(std::getenv("LLAMINAR_EMBEDDING_PARITY"));
        // Logit diagnostics
        if(const char* ds = std::getenv("LLAMINAR_LOGIT_DOT_CHECK")) if(*ds){ s.logit.dot_check=true; s.logit.dot_check_spec=ds; }
        s.logit.dot_dump   = flag(std::getenv("LLAMINAR_LOGIT_DOT_CHECK_DUMP"));
        s.logit.dot_prenorm= flag(std::getenv("LLAMINAR_LOGIT_DOT_CHECK_PRENORM"));
        // Output norm controls
        s.output_norm.bypass          = flag(std::getenv("LLAMINAR_BYPASS_OUTPUT_NORM"));
        s.output_norm.force_unit      = flag(std::getenv("LLAMINAR_OUTPUT_NORM_FORCE_UNIT"));
        s.output_norm.force_unit_all  = flag(std::getenv("LLAMINAR_ALL_RMSNORM_FORCE_UNIT"));
        s.output_norm.clamp           = flag(std::getenv("LLAMINAR_OUTPUT_NORM_CLAMP"));
        // LM head diagnostics
        s.lm_head.raw_orientation = flag(std::getenv("LLAMINAR_LM_HEAD_RAW_ORIENTATION"));
        s.lm_head.cosine_diag     = flag(std::getenv("LLAMINAR_LM_HEAD_COSINE_DIAG"));
        // COSMA capture / dump
        s.cosma_capture.capture_last_gemm = flag(std::getenv("LLAMINAR_COSMA_CAPTURE_LAST_GEMM"));
        if(const char* cap_samp = std::getenv("LLAMINAR_COSMA_CAPTURE_SAMPLE_DIM")) s.cosma_capture.capture_sample_dim = std::max(0, std::atoi(cap_samp));
        if(const char* cap_depth= std::getenv("LLAMINAR_COSMA_CAPTURE_DEPTH")) s.cosma_capture.capture_depth = std::max(0, std::atoi(cap_depth));
        s.cosma_capture.dump_stats = flag(std::getenv("LLAMINAR_COSMA_DUMP_STATS"));
        if(const char* dsp = std::getenv("LLAMINAR_COSMA_DUMP_STATS_PATH")) if(*dsp) s.cosma_capture.dump_stats_path = dsp;
        s.cosma_capture.dump_gemm_snapshots = flag(std::getenv("LLAMINAR_COSMA_DUMP_GEMM_SNAPSHOTS"));
        if(const char* gsp = std::getenv("LLAMINAR_COSMA_DUMP_GEMM_SNAPSHOTS_PATH")) if(*gsp) s.cosma_capture.dump_gemm_snapshots_path = gsp;
    // Baseline capture / compare
    s.baseline.capture = flag(std::getenv("LLAMINAR_PREFILL_CAPTURE_BASELINE"));
    s.baseline.compare = flag(std::getenv("LLAMINAR_PREFILL_COMPARE_BASELINE"));
    // FFN shard tracing
    if(const char* sh = std::getenv("LLAMINAR_PREFILL_TRACE_FFN_SHARDS")) if(*sh){ s.ffn_shard_trace.enabled=true; s.ffn_shard_trace.shards_spec=sh; if(s.ffn_shard_trace.shards_spec=="*"||s.ffn_shard_trace.shards_spec=="all") s.ffn_shard_trace.match_all=true; }
    if(const char* rr = std::getenv("LLAMINAR_PREFILL_TRACE_FFN_ROWS")) if(*rr){ s.ffn_shard_trace.enabled=true; s.ffn_shard_trace.rows_spec=rr; }
    if(const char* cc = std::getenv("LLAMINAR_PREFILL_TRACE_FFN_COLS")) if(*cc){ s.ffn_shard_trace.enabled=true; s.ffn_shard_trace.cols_spec=cc; }
    if(const char* ll = std::getenv("LLAMINAR_PREFILL_TRACE_FFN_LIMIT")) if(*ll){ try { int v=std::stoi(ll); if(v>0) { s.ffn_shard_trace.limit=v; s.ffn_shard_trace.enabled=true; } } catch(...){} }
    auto parse_simple_index_list = [](const std::string& spec){ std::vector<int> out; if(spec.empty()) return out; std::stringstream ss(spec); std::string tok; while(std::getline(ss,tok,',')){ if(tok.empty()) continue; try { out.push_back(std::stoi(tok)); } catch(...){} } std::sort(out.begin(), out.end()); out.erase(std::unique(out.begin(), out.end()), out.end()); return out; };
    if(!s.ffn_shard_trace.rows_spec.empty()) s.ffn_shard_trace.rows = parse_simple_index_list(s.ffn_shard_trace.rows_spec);
    if(!s.ffn_shard_trace.cols_spec.empty()) s.ffn_shard_trace.cols = parse_simple_index_list(s.ffn_shard_trace.cols_spec);
    // Fused RMS forensics
    // Performance instrumentation
    s.performance.enable = flag(std::getenv("LLAMINAR_PERF_ENABLE"));
    s.performance.log_each_matmul = flag(std::getenv("LLAMINAR_PERF_LOG_EACH_MATMUL"));
    if(const char* pr = std::getenv("LLAMINAR_PERF_LOG_RANK")) { int v=std::atoi(pr); if(v>=0) s.performance.log_rank = v; }
    s.performance.layer_mlp = flag(std::getenv("LLAMINAR_PERF_LAYER_MLP"));
    s.performance.layer_verbose = flag(std::getenv("LLAMINAR_PERF_LAYER_VERBOSE"));
    s.performance.layer_attention = flag(std::getenv("LLAMINAR_PERF_LAYER_ATTENTION"));
    if(const char* fr = std::getenv("LLAMINAR_RMS_FORENSICS_FUSED")) if(*fr) s.rms_fused.forensics = true;
    if(const char* rr2 = std::getenv("LLAMINAR_RMS_FUSED_ROWS")) if(*rr2){ int v=std::atoi(rr2); if(v>0) s.rms_fused.rows_preview = v; }
    if(const char* cc2 = std::getenv("LLAMINAR_RMS_FUSED_COLS")) if(*cc2){ int v=std::atoi(cc2); if(v>0) s.rms_fused.cols_preview = v; }
    if(const char* dl = std::getenv("LLAMINAR_FUSED_RMS_DUMP_LAYER")) if(*dl){ s.rms_fused.dump_layer=true; s.rms_fused.dump_layer_spec = dl; }
    if(const char* eps = std::getenv("LLAMINAR_RMS_EPS_OVERRIDE")) if(*eps){ try { s.rms_fused.eps_override = std::stod(eps); s.rms_fused.eps_override_active=true; } catch(...){} }
    // Embedding warnings
    s.embedding_warn.transpose_warn = flag(std::getenv("LLAMINAR_EMBEDDING_TRANSPOSE_WARN"));
    
    // Parity testing controls
    s.parity.save_per_token = flag(std::getenv("LLAMINAR_PARITY_SAVE_PER_TOKEN"));
    if (const char* od = std::getenv("LLAMINAR_PARITY_OUTPUT_DIR")) {
        if (*od) s.parity.output_dir = od;
    }
    
    // Test harness
    s.test_harness.skip_mpi_in_single_test = flag(std::getenv("LLAMINAR_SKIP_MPI_IN_SINGLE_TEST"));

    // Logging
    if(const char* lvl = std::getenv("LLAMINAR_LOG_LEVEL")) { if(*lvl){ s.logging.log_level_active=true; s.logging.log_level = lvl; } }
    // TP policy (general)
    if(const char* fbt = std::getenv("LLAMINAR_TP_FORCE_BLAS_THREADS")) { int v=std::atoi(fbt); if(v>0) s.tp_policy.force_blas_threads = v; }
    if(const char* mbt = std::getenv("LLAMINAR_TP_MAX_BLAS_THREADS")) { int v=std::atoi(mbt); if(v>0) s.tp_policy.max_blas_threads = v; }
    if(const char* slh = std::getenv("LLAMINAR_SEQ_LEN_HINT")) { int v=std::atoi(slh); if(v>0) s.tp_policy.seq_len_hint = v; }
    if(std::getenv("LLAMINAR_TP_OUTER_PARALLEL")) s.tp_policy.outer_parallel = true;
    if(std::getenv("LLAMINAR_TP_DISABLE_OUTER_PAR")) s.tp_policy.disable_outer_par = true;
    // Logger buffer override
    if(const char* lbl = std::getenv("LLAMINAR_LOG_BUFFER_LINES")) { try { long long v = std::stoll(lbl); if(v>0 && v < 2000000) s.logger.buffer_lines_override = (size_t)v; } catch(...){} }
        // KV cache (refresh)
        s.kv_cache.dynamic_init = flag(std::getenv("LLAMINAR_KV_DYNAMIC_INIT"));
        if(const char* gfac2 = std::getenv("LLAMINAR_KV_GROWTH_FACTOR")) { int v=std::atoi(gfac2); if(v>=1 && v<=16) s.kv_cache.growth_factor = v; }
        // Distribution Mode (two-tier policy)
        if(const char* dm = std::getenv("LLAMINAR_DISTRIBUTION_MODE")) { if(*dm) s.distribution.distribution_mode = dm; }
        s.distribution.force_replicated = flag(std::getenv("LLAMINAR_FORCE_REPLICATED"));
        s.distribution.force_sharded    = flag(std::getenv("LLAMINAR_FORCE_SHARDED"));
        if(const char* pt = std::getenv("LLAMINAR_SHARDING_PARAM_THRESHOLD")) { int v=std::atoi(pt); if(v>0) s.distribution.param_threshold_billions = v; }
        if(const char* mf = std::getenv("LLAMINAR_MODEL_MEM_FRACTION_MAX")) { try { double d=std::stod(mf); if(d>0.0 && d<0.99) s.distribution.model_mem_fraction_max = d; } catch(...){} }
        // Weight slicing (Phase 0 flags)
        s.weight_slicing.disable = flag(std::getenv("LLAMINAR_DISABLE_WEIGHT_SHARDING"));
        s.weight_slicing.force   = flag(std::getenv("LLAMINAR_FORCE_WEIGHT_SHARDING"));
        s.weight_slicing.validate= flag(std::getenv("LLAMINAR_WEIGHT_SLICE_VALIDATE"));
        if(const char* mc = std::getenv("LLAMINAR_WEIGHT_SLICE_MIN_COLS")) { int v=std::atoi(mc); if(v>0) s.weight_slicing.min_cols = v; }
        if(s.weight_slicing.force || s.weight_slicing.validate) {
            LOG_INFO("[DEBUG_ENV_INIT] weight_slicing disable=" << s.weight_slicing.disable
                     << " force=" << s.weight_slicing.force
                     << " validate=" << s.weight_slicing.validate
                     << " min_cols=" << s.weight_slicing.min_cols);
        }
        // Threading (global OpenMP policy)
    if(const char* tf = std::getenv("LLAMINAR_OMP_FORCE")) { int v = std::atoi(tf); if(v>0) s.threading.force_threads = v; }
    if(std::getenv("LLAMINAR_OMP_USE_PHYSICAL")) s.threading.use_physical = true;
    if(std::getenv("LLAMINAR_BIND_PER_SOCKET")) s.threading.bind_per_socket = true;
    // TP simulation (local W/O executor experimentation)
    if(std::getenv("LLAMINAR_TP_WO_SIM_ENABLE")) s.tp_sim.enable = true;
    if(const char* tpp = std::getenv("LLAMINAR_TP_WO_SIM_PARTITIONS")) { int v=std::atoi(tpp); if(v>0 && v < 4096) s.tp_sim.partitions = v; }
    if(const char* md = std::getenv("LLAMINAR_TP_WO_SIM_MODE")) {
        std::string m(md); for(char &c: m) c = (char)std::tolower(c);
        if(m=="row" || m=="rows") s.tp_sim.mode = DebugEnvSnapshot::TPSimEnv::Mode::Row;
        else if(m=="col" || m=="cols" || m=="column") s.tp_sim.mode = DebugEnvSnapshot::TPSimEnv::Mode::Col;
        else s.tp_sim.mode = DebugEnvSnapshot::TPSimEnv::Mode::Auto;
    }
    // MLP TP
    if(std::getenv("LLAMINAR_TP_MLP_ENABLE")) s.mlp_tp.enable = true;
    if(const char* ms = std::getenv("LLAMINAR_TP_MLP_SIZE")) { int v=std::atoi(ms); if(v>1 && v<4096) s.mlp_tp.partitions = v; }
    if(std::getenv("LLAMINAR_TP_MLP_VALIDATE")) s.mlp_tp.validate = true;
    if(std::getenv("LLAMINAR_TP_MLP_FORCE_COLUMN")) s.mlp_tp.force_column = true;
    if(std::getenv("LLAMINAR_TP_MLP_FORCE_ROW")) s.mlp_tp.force_row = true;
    
    // Performance tracing
    if(std::getenv("LLAMINAR_PERF_TRACE")) s.perf.trace_enabled = true;
    if(const char* td = std::getenv("LLAMINAR_PERF_TRACE_DETAIL")) {
        std::string detail(td);
        for(char &c: detail) c = (char)std::tolower(c);
        s.perf.trace_detail = detail;
    }
    if(const char* tf = std::getenv("LLAMINAR_PERF_TRACE_FILTER")) {
        if(*tf) s.perf.trace_filter = tf;
    }
    if(const char* to = std::getenv("LLAMINAR_PERF_TRACE_DUMP")) {
        if(*to) s.perf.trace_output_file = to;
    }
    if(const char* ad = std::getenv("LLAMINAR_PERF_TRACE_AUTO_DUMP")) {
        if(*ad == '0') s.perf.trace_dump_on_exit = false;
    }
    
    return s; }());
        }
        return *g_snapshot;
    }

    void debugEnvRefresh()
    {
        // Rebuild snapshot; safe in tests before weight loading. Not thread-safe.
        if (g_snapshot)
        {
            *g_snapshot = [&]()
            { DebugEnvSnapshot s; 
                // Minimal reparse: replicate logic from original construction (keep in sync!)
                // Sharding
                s.sharding.debug_materialize_attention = flag(std::getenv("LLAMINAR_DEBUG_MATERIALIZE_ATTENTION"));
                s.sharding.dump_shards = flag(std::getenv("LLAMINAR_DUMP_SHARDS"));
                s.sharding.dump_shards_n = to_int(std::getenv("LLAMINAR_DUMP_SHARDS_N"), 16);
                if(s.sharding.dump_shards_n <= 0 || s.sharding.dump_shards_n > 8192) s.sharding.dump_shards_n = 16;
                s.sharding.shard_parity_check = flag(std::getenv("LLAMINAR_SHARD_PARITY_CHECK"));
                s.sharding.assert_replicated_misuse = flag(std::getenv("LLAMINAR_ASSERT_REPLICATED_MISUSE"));
                s.sharding.shard_load_diag = flag(std::getenv("LLAMINAR_SHARD_LOAD_DIAG"));
                // Pipeline (new flags must be kept in sync with initial static snapshot parse)
                s.pipeline.layer_token_diff = flag(std::getenv("LLAMINAR_PIPELINE_LAYER_TOKEN_DIFF"));
                s.pipeline.layer_token_diff_verbose = flag(std::getenv("LLAMINAR_PIPELINE_LAYER_TOKEN_DIFF_VERBOSE"));
                s.pipeline.attn_ref_compare = flag(std::getenv("LLAMINAR_DEBUG_ATTENTION_REF"));
                s.pipeline.layer_replay_compare = flag(std::getenv("LLAMINAR_DEBUG_LAYER_REPLAY_COMPARE"));
                s.pipeline.capture_pre_lm = flag(std::getenv("LLAMINAR_PIPELINE_CAPTURE_PRE_LM"));
                // COSMA subset
                s.cosma.prefill_threshold = to_int(std::getenv("LLAMINAR_COSMA_PREFILL_THRESHOLD"), 4096);
                s.cosma.fast_path_threshold = to_int(std::getenv("LLAMINAR_COSMA_FAST_PATH_THRESHOLD"), -1);
                s.cosma.validate_tile = to_int(std::getenv("LLAMINAR_COSMA_VALIDATE_TILE"), 0);
                s.cosma.log_level = to_int(std::getenv("LLAMINAR_COSMA_LOG_LEVEL"), 2);
                s.cosma.max_resident_mb = to_ll(std::getenv("LLAMINAR_COSMA_MAX_RESIDENT_MB"), 2048);
                s.cosma.disable = flag(std::getenv("LLAMINAR_COSMA_DISABLE"));
                s.cosma.force = flag(std::getenv("LLAMINAR_COSMA_FORCE"));
                // Distribution
                // Softmax (refresh subset)
                s.softmax.force_scalar = flag(std::getenv("LLAMINAR_SOFTMAX_FORCE_SCALAR"));
                s.softmax.validate = flag(std::getenv("LLAMINAR_SOFTMAX_VALIDATE"));
                s.softmax.parallel_row_threshold = to_int(std::getenv("LLAMINAR_SOFTMAX_PARALLEL_ROW_THRESHOLD"), 0);
                s.softmax.parallel_elems_threshold = to_int(std::getenv("LLAMINAR_SOFTMAX_PARALLEL_ELEMS"), 32768);
                { const char* cf = std::getenv("LLAMINAR_SOFTMAX_CAUSAL_FUSE"); if(cf && *cf=='0') s.softmax.causal_fuse=false; else if(cf && *cf) s.softmax.causal_fuse=true; }
                s.softmax.fast_exp = flag(std::getenv("LLAMINAR_SOFTMAX_FAST_EXP"));
                if(const char* fem = std::getenv("LLAMINAR_SOFTMAX_FAST_EXP_MODE")) { int v=std::atoi(fem); if(v>=0 && v<=4) s.softmax.fast_exp_mode = v; }
                s.softmax.dist_recompute = flag(std::getenv("LLAMINAR_SOFTMAX_DIST_RECOMPUTE"));
                if(const char* drt = std::getenv("LLAMINAR_SOFTMAX_DIST_RECOMPUTE_THRESHOLD")) { int v=std::atoi(drt); if(v>0) s.softmax.dist_recompute_threshold = v; }
                if(const char* vsr = std::getenv("LLAMINAR_SOFTMAX_VALIDATE_SAMPLE_ROWS")) { int v=std::atoi(vsr); s.softmax.validate_sample_rows = v; }
                if(const char* vrl = std::getenv("LLAMINAR_SOFTMAX_VALIDATE_REL_L2")) { try { s.softmax.validate_rel_l2_tol = std::stod(vrl);} catch(...){} }
                if(const char* vma = std::getenv("LLAMINAR_SOFTMAX_VALIDATE_MAX_ABS")) { try { s.softmax.validate_max_abs_tol = std::stod(vma);} catch(...){} }
                s.softmax.validate_abort = flag(std::getenv("LLAMINAR_SOFTMAX_VALIDATE_ABORT"));
                if(const char* dm = std::getenv("LLAMINAR_DISTRIBUTION_MODE")) if(*dm) s.distribution.distribution_mode = dm;
                s.distribution.force_replicated = flag(std::getenv("LLAMINAR_FORCE_REPLICATED"));
                s.distribution.force_sharded    = flag(std::getenv("LLAMINAR_FORCE_SHARDED"));
                if(const char* pt = std::getenv("LLAMINAR_SHARDING_PARAM_THRESHOLD")) { int v=std::atoi(pt); if(v>0) s.distribution.param_threshold_billions = v; }
                if(const char* mf = std::getenv("LLAMINAR_MODEL_MEM_FRACTION_MAX")) { try { double d=std::stod(mf); if(d>0.0 && d<0.99) s.distribution.model_mem_fraction_max = d; } catch(...){} }
                // Weight slicing
                s.weight_slicing.disable = flag(std::getenv("LLAMINAR_DISABLE_WEIGHT_SHARDING"));
                s.weight_slicing.force   = flag(std::getenv("LLAMINAR_FORCE_WEIGHT_SHARDING"));
                s.weight_slicing.validate= flag(std::getenv("LLAMINAR_WEIGHT_SLICE_VALIDATE"));
                if(const char* mc = std::getenv("LLAMINAR_WEIGHT_SLICE_MIN_COLS")) { int v=std::atoi(mc); if(v>0) s.weight_slicing.min_cols = v; }
                if(s.weight_slicing.force || s.weight_slicing.validate) {
                    LOG_INFO("[DEBUG_ENV_REFRESH] weight_slicing disable=" << s.weight_slicing.disable
                             << " force=" << s.weight_slicing.force
                             << " validate=" << s.weight_slicing.validate
                             << " min_cols=" << s.weight_slicing.min_cols);
                }
                // Threading (refresh)
                if(const char* tf = std::getenv("LLAMINAR_OMP_FORCE")) { int v = std::atoi(tf); if(v>0) s.threading.force_threads = v; }
                if(std::getenv("LLAMINAR_OMP_USE_PHYSICAL")) s.threading.use_physical = true;
                if(std::getenv("LLAMINAR_BIND_PER_SOCKET")) s.threading.bind_per_socket = true;
                // TP simulation (refresh subset)
                if(std::getenv("LLAMINAR_TP_WO_SIM_ENABLE")) s.tp_sim.enable = true;
                if(const char* tpp2 = std::getenv("LLAMINAR_TP_WO_SIM_PARTITIONS")) { int v=std::atoi(tpp2); if(v>0 && v<4096) s.tp_sim.partitions = v; }
                if(const char* md2 = std::getenv("LLAMINAR_TP_WO_SIM_MODE")) {
                    std::string m(md2); for(char &c: m) c = (char)std::tolower(c);
                    if(m=="row" || m=="rows") s.tp_sim.mode = DebugEnvSnapshot::TPSimEnv::Mode::Row;
                    else if(m=="col" || m=="cols" || m=="column") s.tp_sim.mode = DebugEnvSnapshot::TPSimEnv::Mode::Col;
                    else s.tp_sim.mode = DebugEnvSnapshot::TPSimEnv::Mode::Auto;
                }
                // MLP TP refresh
                if(std::getenv("LLAMINAR_TP_MLP_ENABLE")) s.mlp_tp.enable = true;
                if(const char* ms2 = std::getenv("LLAMINAR_TP_MLP_SIZE")) { int v=std::atoi(ms2); if(v>1 && v<4096) s.mlp_tp.partitions = v; }
                if(std::getenv("LLAMINAR_TP_MLP_VALIDATE")) s.mlp_tp.validate = true;
                if(std::getenv("LLAMINAR_TP_MLP_FORCE_COLUMN")) s.mlp_tp.force_column = true;
                if(std::getenv("LLAMINAR_TP_MLP_FORCE_ROW")) s.mlp_tp.force_row = true;
                // Attention decode instrumentation (refresh subset)
                if(std::getenv("LLAMINAR_ATTENTION_DECODE_DIAG")) s.attention_decode.decode_diag = true;
                if(std::getenv("LLAMINAR_ATTENTION_DECODE_DUMP_QKV")) s.attention_decode.dump_full_qkv = true;
                if(const char* dl = std::getenv("LLAMINAR_ATTENTION_DECODE_DUMP_LIMIT")) { int v=std::atoi(dl); if(v>0 && v<=4096) s.attention_decode.dump_limit = v; }
                // RMSNorm tuning flags (subset needed for dynamic test reconfiguration)
                s.rmsnorm.force_scalar = flag(std::getenv("LLAMINAR_RMSNORM_FORCE_SCALAR"));
                s.rmsnorm.disable_tls_scratch = flag(std::getenv("LLAMINAR_RMSNORM_DISABLE_TLS_SCRATCH"));
                if(const char* sp2 = std::getenv("LLAMINAR_RMSNORM_SCRATCH_PREALLOC_ROWS")) { int v=std::atoi(sp2); if(v>0 && v<1000000) s.rmsnorm.scratch_prealloc_rows = v; }
                s.rmsnorm.false_sharing_probe = flag(std::getenv("LLAMINAR_RMSNORM_FALSE_SHARING_PROBE"));
                s.rmsnorm.fast_accumulate = flag(std::getenv("LLAMINAR_RMSNORM_FAST_ACC"));
                if(const char* rvi2 = std::getenv("LLAMINAR_RMSNORM_VEC_IMPL")) { int v=std::atoi(rvi2); if(v>=0 && v<=3) s.rmsnorm.vec_impl = v; }
                // TP policy refresh
                if(const char* fbt = std::getenv("LLAMINAR_TP_FORCE_BLAS_THREADS")) { int v=std::atoi(fbt); if(v>0) s.tp_policy.force_blas_threads = v; }
                if(const char* mbt = std::getenv("LLAMINAR_TP_MAX_BLAS_THREADS")) { int v=std::atoi(mbt); if(v>0) s.tp_policy.max_blas_threads = v; }
                if(const char* slh = std::getenv("LLAMINAR_SEQ_LEN_HINT")) { int v=std::atoi(slh); if(v>0) s.tp_policy.seq_len_hint = v; }
                if(std::getenv("LLAMINAR_TP_OUTER_PARALLEL")) s.tp_policy.outer_parallel = true;
                if(std::getenv("LLAMINAR_TP_DISABLE_OUTER_PAR")) s.tp_policy.disable_outer_par = true;
                // Logger buffer override
                if(const char* lbl = std::getenv("LLAMINAR_LOG_BUFFER_LINES")) { try { long long v = std::stoll(lbl); if(v>0 && v < 2000000) s.logger.buffer_lines_override = (size_t)v; } catch(...){} }
                // Logging level
                if(const char* lvl = std::getenv("LLAMINAR_LOG_LEVEL")) { if(*lvl){ s.logging.log_level_active=true; s.logging.log_level = lvl; } }
                return s; }();
        }
        else
        {
            // If not yet built, just access debugEnv() which will build with current env.
            (void)debugEnv();
        }
    }

    void configureGlobalOpenMPThreads()
    {
        auto &snap = const_cast<DebugEnvSnapshot &>(debugEnv());
        if (snap.threading.configured)
            return; // idempotent

        int rank = 0, size = 1;
#ifdef LLAMINAR_HAVE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
        int chosen = 0;
        std::string reason;
        if (snap.threading.force_threads > 0)
        {
            chosen = snap.threading.force_threads;
            reason = "force";
        }
        else if (snap.threading.use_physical)
        {
            try
            {
                // Lightweight topology query (hyperthreading disabled for baseline core count)
                TopologyManager tm; // include header
                auto cpuTopo = tm.detectCPUTopology(false);
                int per_rank = 0;
                if (snap.threading.bind_per_socket)
                {
                    if (cpuTopo.sockets >= size)
                    {
                        // Intent: one rank per socket; each rank gets that socket's physical cores
                        // Map rank -> socket by simple ordering
                        per_rank = cpuTopo.cores_per_socket;
                        reason = "physical_socket";
                    }
                    else
                    {
                        if (rank == 0)
                        {
                            LOG_WARN("[THREAD_CONFIG] bind_per_socket requested but sockets=" << cpuTopo.sockets << " < mpi_size=" << size << "; falling back to shared physical division");
                        }
                        per_rank = cpuTopo.physical_cores > 0 ? cpuTopo.physical_cores / std::max(1, size) : 0;
                    }
                }
                else
                {
                    per_rank = cpuTopo.physical_cores > 0 ? cpuTopo.physical_cores / std::max(1, size) : 0;
                }
                if (per_rank <= 0)
                    per_rank = 1;
                chosen = per_rank;
                if (reason.empty())
                    reason = "physical";
            }
            catch (...)
            {
                // Fallback: leave default
                chosen = 0;
                reason = "error";
            }
        }
        if (chosen > 0)
        {
            omp_set_num_threads(chosen);
            if (rank == 0)
            {
                LOG_INFO("[THREAD_CONFIG] global OpenMP threads=" << chosen << " reason=" << reason << (snap.threading.bind_per_socket ? " bind_per_socket=on" : ""));
            }
            // If binding per socket requested, attempt to export placement hints (best-effort)
            if (snap.threading.bind_per_socket)
            {
                // Set OMP env hints for downstream library phases (cannot unset existing)
                if (std::getenv("OMP_PLACES") == nullptr)
                    setenv("OMP_PLACES", "sockets", 1);
                if (std::getenv("OMP_PROC_BIND") == nullptr)
                    setenv("OMP_PROC_BIND", "close", 1);
                // Emit per-rank socket assignment + primary cores list (best-effort)
                if (reason == "physical_socket")
                {
                    try
                    {
                        TopologyManager tm_socket;
                        auto topo_socket = tm_socket.detectCPUTopology(false);
                        int socket_id = rank;
                        if (topo_socket.sockets > 0 && socket_id >= topo_socket.sockets)
                            socket_id = socket_id % topo_socket.sockets;
                        auto it = topo_socket.socket_to_primary_cpus.find(socket_id);
                        std::vector<int> primaries;
                        if (it != topo_socket.socket_to_primary_cpus.end())
                            primaries = it->second;
                        size_t limit = std::min<size_t>(32, primaries.size());
                        std::ostringstream oss;
                        oss << "[THREAD_CONFIG] rank=" << rank << " socket_id=" << socket_id << " primary_cores=";
                        for (size_t i = 0; i < limit; i++)
                        {
                            if (i)
                                oss << ',';
                            oss << primaries[i];
                        }
                        if (primaries.size() > limit)
                            oss << ",... (total=" << primaries.size() << ")";
                        else
                            oss << " (total=" << primaries.size() << ")";
                        LOG_INFO(oss.str());
                    }
                    catch (...)
                    {
                        if (rank == 0)
                            LOG_WARN("[THREAD_CONFIG] failed to enumerate primary cores for socket logging");
                    }
                }
            }
            // Emit actual CPU affinity list for observability (rank-specific)
            cpu_set_t mask;
            CPU_ZERO(&mask);
            if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
            {
                std::vector<int> cpus;
                cpus.reserve(256);
                int max_probe = 4096; // safety upper bound
                for (int cpu = 0; cpu < max_probe; ++cpu)
                {
                    if (CPU_ISSET(cpu, &mask))
                        cpus.push_back(cpu);
                    if (cpu > 4 && cpus.size() > 0 && cpu > 128 && cpu > (int)cpus.back() + 256)
                        break; // soft break condition
                }
                // Limit display to first 32 then ellipsis if longer
                std::ostringstream oss;
                oss << "[THREAD_CONFIG] rank=" << rank << " affinity_cpus=";
                size_t limit = std::min<size_t>(32, cpus.size());
                for (size_t i = 0; i < limit; i++)
                {
                    if (i)
                        oss << ',';
                    oss << cpus[i];
                }
                if (cpus.size() > limit)
                    oss << ",... (total=" << cpus.size() << ")";
                else
                    oss << " (total=" << cpus.size() << ")";
                LOG_INFO(oss.str());
            }
            else if (rank == 0)
            {
                LOG_WARN("[THREAD_CONFIG] unable to query sched_getaffinity");
            }
        }
        else
        {
            if (rank == 0 && (snap.threading.force_threads > 0 || snap.threading.use_physical))
            {
                LOG_INFO("[THREAD_CONFIG] no change (reason=" << reason << ")");
            }
        }
        snap.threading.configured = true;
    }

    std::vector<std::string> formatDebugEnvSummary(const DebugEnvSnapshot &s)
    {
        std::vector<std::string> lines;
        auto on = [&](bool v)
        { return v ? "on" : "off"; };
        // Sharding
        lines.push_back(std::string("[DebugEnv] sharding: materialize_attn=") + on(s.sharding.debug_materialize_attention) +
                        " dump_shards=" + on(s.sharding.dump_shards) +
                        (s.sharding.dump_shards ? (" n=" + std::to_string(s.sharding.dump_shards_n)) : "") +
                        " parity_check=" + on(s.sharding.shard_parity_check) +
                        " load_diag=" + on(s.sharding.shard_load_diag));
        // COSMA (core subset always printed)
        {
            std::ostringstream oss;
            oss << "[DebugEnv] cosma: disable=" << on(s.cosma.disable)
                << " force=" << on(s.cosma.force)
                << " prefill_thr=" << s.cosma.prefill_threshold
                << " fast_path_thr=" << s.cosma.fast_path_threshold
                << " validate_tile=" << s.cosma.validate_tile
                << " log_level=" << s.cosma.log_level
                << " compare_repl=" << on(s.cosma.compare_replicated)
                << " force_direct=" << on(s.cosma.force_direct)
                << " force_repl=" << on(s.cosma.force_replicated)
                << " repl_diag=" << on(s.cosma.force_replicated_diag)
                << " overlap=" << on(s.cosma.overlap_enabled)
                << " overlap_verbose=" << on(s.cosma.overlap_verbose);
            lines.push_back(oss.str());
            if (s.cosma.direct_threshold_override)
            {
                lines.push_back(std::string("[DebugEnv] cosma: direct_threshold_ops=") + std::to_string(s.cosma.direct_threshold_ops));
            }
        }
        lines.push_back(std::string("[DebugEnv] pipeline: capture_pre_lm=") + on(s.pipeline.capture_pre_lm) +
                        " layerwise_stats=" + on(s.pipeline.layerwise_stats) +
                        " abstract_base=" + on(s.pipeline.enable_abstract_pipeline) +
                        " incr_decode=" + on(!s.pipeline.disable_incremental_decode) +
                        " layer_token_diff=" + on(s.pipeline.layer_token_diff) +
                        " pre_lm_row_diff=" + on(s.pipeline.pre_lm_row_diff));
        // KV cache (print only if non-default policies active to limit noise)
        if (s.kv_cache.dynamic_init || s.kv_cache.growth_factor != 2)
        {
            lines.push_back(std::string("[DebugEnv] kv_cache: dynamic_init=") + on(s.kv_cache.dynamic_init) +
                            " growth_factor=" + std::to_string(s.kv_cache.growth_factor));
        }
        lines.push_back(std::string("[DebugEnv] embedding: trace=") + on(s.embedding.trace) +
                        " fail_fast=" + on(s.embedding.fail_fast) +
                        " trace_tokens=" + std::to_string(s.embedding.trace_tokens) +
                        " trace_dims=" + std::to_string(s.embedding.trace_dims) +
                        (s.embedding.trace_rows_spec.empty() ? "" : " rows_spec=" + s.embedding.trace_rows_spec));
        lines.push_back(std::string("[DebugEnv] rmsnorm: validate_ref=") + on(s.rmsnorm.validate_ref) +
                        " dump_gamma=" + on(s.rmsnorm.dump_gamma) +
                        " force_unit_gamma=" + on(s.rmsnorm.force_unit_gamma) +
                        " gamma_checksum=" + on(s.rmsnorm.gamma_checksum) +
                        (s.rmsnorm.trace_rows_spec.empty() ? "" : " trace_rows=" + s.rmsnorm.trace_rows_spec) +
                        (s.rmsnorm.force_scalar ? " force_scalar=on" : "") +
                        (s.rmsnorm.disable_tls_scratch ? " tls_scratch=off" : "") +
                        (s.rmsnorm.scratch_prealloc_rows > 0 ? (" prealloc_rows=" + std::to_string(s.rmsnorm.scratch_prealloc_rows)) : "") +
                        (s.rmsnorm.fast_accumulate ? " fast_acc=on" : "") +
                        (s.rmsnorm.vec_impl ? (" vec_impl=" + std::to_string(s.rmsnorm.vec_impl)) : ""));
        // Distribution summary (only if any override active)
        if (!s.distribution.distribution_mode.empty() || s.distribution.force_replicated || s.distribution.force_sharded)
        {
            lines.push_back(std::string("[DebugEnv] distribution: mode=") + (s.distribution.distribution_mode.empty() ? "<auto>" : s.distribution.distribution_mode) +
                            " force_repl=" + on(s.distribution.force_replicated) +
                            " force_sharded=" + on(s.distribution.force_sharded) +
                            " param_thr(B)=" + std::to_string(s.distribution.param_threshold_billions) +
                            " mem_frac_max=" + std::to_string(s.distribution.model_mem_fraction_max));
        }
        lines.push_back(std::string("[DebugEnv] ablation: attention=") + on(s.ablation.ablate_attention) +
                        " ffn=" + on(s.ablation.ablate_ffn));
        if (s.layer_capture.capture)
        {
            lines.push_back(std::string("[DebugEnv] layer_capture: tokens_spec=") +
                            (s.layer_capture.tokens_spec.empty() ? "<empty>" : s.layer_capture.tokens_spec) +
                            " parsed_tokens=" + std::to_string(s.layer_capture.tokens.size()));
        }
        if (s.rms_forensics.enabled)
        {
            std::ostringstream oss;
            oss << "[DebugEnv] rms_forensics: layers_spec=" << (s.rms_forensics.layers_spec.empty() ? "<all>" : s.rms_forensics.layers_spec)
                << " rows_spec=" << (s.rms_forensics.rows_spec.empty() ? "<none>" : s.rms_forensics.rows_spec)
                << " warn_rel_l2=" << s.rms_forensics.warn_rel_l2
                << " trace_vectors=" << on(s.rms_forensics.trace_vectors)
                << " diff_only=" << on(s.rms_forensics.diff_only);
            lines.push_back(oss.str());
        }
        if (s.prefill_debug.trace_io || s.prefill_debug.debug_compare || s.prefill_debug.debug_attention || s.prefill_debug.debug_output)
        {
            lines.push_back(std::string("[DebugEnv] prefill_debug: trace_io=") + on(s.prefill_debug.trace_io) +
                            " debug_compare=" + on(s.prefill_debug.debug_compare) +
                            " debug_attention=" + on(s.prefill_debug.debug_attention) +
                            " debug_output=" + on(s.prefill_debug.debug_output));
        }
        if (s.embedding_diag.parity)
        {
            lines.push_back("[DebugEnv] embedding_diag: parity=on");
        }
        if (s.logit.dot_check)
        {
            lines.push_back(std::string("[DebugEnv] logit: dot_check_spec=") + s.logit.dot_check_spec +
                            " dump=" + on(s.logit.dot_dump) +
                            " prenorm=" + on(s.logit.dot_prenorm));
        }
        if (s.output_norm.bypass || s.output_norm.force_unit || s.output_norm.force_unit_all || s.output_norm.clamp)
        {
            lines.push_back(std::string("[DebugEnv] output_norm: bypass=") + on(s.output_norm.bypass) +
                            " force_unit=" + on(s.output_norm.force_unit) +
                            " force_unit_all=" + on(s.output_norm.force_unit_all) +
                            " clamp=" + on(s.output_norm.clamp));
        }
        if (s.lm_head.raw_orientation || s.lm_head.cosine_diag)
        {
            lines.push_back(std::string("[DebugEnv] lm_head: raw_orientation=") + on(s.lm_head.raw_orientation) +
                            " cosine_diag=" + on(s.lm_head.cosine_diag));
        }
        // Threading summary (only if any override present)
        if (s.threading.force_threads > 0 || s.threading.use_physical || s.threading.bind_per_socket)
        {
            std::ostringstream oss;
            oss << "[DebugEnv] threading: force_threads=" << s.threading.force_threads
                << " use_physical=" << on(s.threading.use_physical)
                << " bind_per_socket=" << on(s.threading.bind_per_socket);
            lines.push_back(oss.str());
        }
        if (s.cosma_capture.capture_last_gemm || s.cosma_capture.dump_stats || s.cosma_capture.dump_gemm_snapshots)
        {
            std::ostringstream oss;
            oss << "[DebugEnv] cosma_capture: last_gemm=" << on(s.cosma_capture.capture_last_gemm)
                << " sample_dim=" << s.cosma_capture.capture_sample_dim
                << " depth=" << s.cosma_capture.capture_depth
                << " dump_stats=" << on(s.cosma_capture.dump_stats);
            if (!s.cosma_capture.dump_stats_path.empty())
                oss << " stats_path=" << s.cosma_capture.dump_stats_path;
            oss << " dump_gemm_snapshots=" << on(s.cosma_capture.dump_gemm_snapshots);
            if (!s.cosma_capture.dump_gemm_snapshots_path.empty())
                oss << " gemm_snapshots_path=" << s.cosma_capture.dump_gemm_snapshots_path;
            lines.push_back(oss.str());
        }
        if (s.baseline.capture || s.baseline.compare)
        {
            lines.push_back(std::string("[DebugEnv] baseline: capture=") + on(s.baseline.capture) +
                            " compare=" + on(s.baseline.compare));
        }
        if (s.ffn_shard_trace.enabled)
        {
            std::ostringstream oss;
            oss << "[DebugEnv] ffn_shard_trace: shards_spec=" << (s.ffn_shard_trace.shards_spec.empty() ? "<implicit>" : s.ffn_shard_trace.shards_spec)
                << " match_all=" << on(s.ffn_shard_trace.match_all)
                << " limit=" << s.ffn_shard_trace.limit;
            if (!s.ffn_shard_trace.rows_spec.empty())
                oss << " rows=" << s.ffn_shard_trace.rows_spec;
            if (!s.ffn_shard_trace.cols_spec.empty())
                oss << " cols=" << s.ffn_shard_trace.cols_spec;
            lines.push_back(oss.str());
        }
        if (s.rms_fused.forensics || s.rms_fused.dump_layer || s.rms_fused.eps_override_active)
        {
            std::ostringstream oss;
            oss << "[DebugEnv] rms_fused: forensics=" << on(s.rms_fused.forensics)
                << " rows_preview=" << s.rms_fused.rows_preview
                << " cols_preview=" << s.rms_fused.cols_preview
                << " dump_layer=" << on(s.rms_fused.dump_layer);
            if (s.rms_fused.dump_layer && !s.rms_fused.dump_layer_spec.empty())
                oss << " dump_layer_spec=" << s.rms_fused.dump_layer_spec;
            if (s.rms_fused.eps_override_active)
                oss << " eps_override=" << s.rms_fused.eps_override;
            lines.push_back(oss.str());
        }
        if (s.embedding_warn.transpose_warn)
        {
            lines.push_back("[DebugEnv] embedding_warn: transpose_warn=on");
        }
        if (s.parity.save_per_token)
        {
            std::ostringstream oss;
            oss << "[DebugEnv] parity: save_per_token=on output_dir=" << s.parity.output_dir;
            lines.push_back(oss.str());
        }
        if (s.test_harness.skip_mpi_in_single_test)
        {
            lines.push_back("[DebugEnv] test_harness: skip_mpi_in_single_test=on");
        }
        if (s.logging.log_level_active)
        {
            lines.push_back(std::string("[DebugEnv] logging: level=") + s.logging.log_level);
        }
        if (s.weight_slicing.disable || s.weight_slicing.force || s.weight_slicing.validate || s.weight_slicing.min_cols > 0)
        {
            std::ostringstream oss;
            oss << "[DebugEnv] weight_slicing: disable=" << on(s.weight_slicing.disable)
                << " force=" << on(s.weight_slicing.force)
                << " validate=" << on(s.weight_slicing.validate)
                << " min_cols=" << s.weight_slicing.min_cols;
            lines.push_back(oss.str());
        }
        if (s.tp_sim.enable || s.tp_sim.partitions > 0)
        {
            auto mode_to_str = [&](DebugEnvSnapshot::TPSimEnv::Mode m)
            {
                switch (m)
                {
                case DebugEnvSnapshot::TPSimEnv::Mode::Row:
                    return "row";
                case DebugEnvSnapshot::TPSimEnv::Mode::Col:
                    return "col";
                case DebugEnvSnapshot::TPSimEnv::Mode::Auto:
                default:
                    return "auto";
                }
            };
            std::ostringstream oss;
            oss << "[DebugEnv] tp_sim: enable=" << on(s.tp_sim.enable)
                << " mode=" << mode_to_str(s.tp_sim.mode)
                << " partitions=" << s.tp_sim.partitions;
            lines.push_back(oss.str());
        }
        return lines;
    }

} // namespace llaminar

// Global counters implementation (outside namespace block above due to closing brace)
llaminar::WeightSlicingCounters &llaminar::weightSlicingCounters()
{
    static WeightSlicingCounters ctrs;
    return ctrs;
}
