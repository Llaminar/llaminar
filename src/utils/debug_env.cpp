#include "debug_env.h"
#include <algorithm>
#include <cstring>
#include <vector>
#include <sstream>

namespace llaminar
{

    static bool flag(const char *v) { return v && *v; }
    static int to_int(const char *v, int d) { return v ? std::atoi(v) : d; }
    static long long to_ll(const char *v, long long d) { return v ? std::atoll(v) : d; }

    const DebugEnvSnapshot &debugEnv()
    {
        static DebugEnvSnapshot snap = []()
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
        // Dequant
        s.dequant.stats = flag(std::getenv("LLAMINAR_DEQUANT_STATS"));
        s.dequant.anomalies = flag(std::getenv("LLAMINAR_DEQUANT_ANOMALIES"));
    // Adaptive
    s.adaptive.disable_cosma = flag(std::getenv("ADAPTIVE_DISABLE_COSMA"));
    // Attention
    s.attention.validate_primitives = flag(std::getenv("LLAMINAR_ATTN_PRIMITIVES_VALIDATE"));
    s.attention.validate_output = flag(std::getenv("LLAMINAR_ATTN_OUTPUT_VALIDATE"));
    if(const char* om = std::getenv("LLAMINAR_ATTN_OUTPUT_MODE")) { if(*om){ s.attention.output_mode_forced=true; s.attention.output_mode = om; }}
    if(const char* gt = std::getenv("LLAMINAR_ATTN_GATHER_THRESHOLD")) { s.attention.gather_threshold = std::atoi(gt); }
    s.attention.force_scalar = flag(std::getenv("LLAMINAR_ATTN_FORCE_SCALAR"));
    s.attention.validate_proj = flag(std::getenv("LLAMINAR_ATTN_VALIDATE_PROJ"));
    s.attention.micro_trace = flag(std::getenv("LLAMINAR_ATTN_MICRO_TRACE"));
    s.attention.dump_attention = flag(std::getenv("LLAMINAR_ATTN_DUMP_ATTENTION"));
    s.attention.tp_disable = flag(std::getenv("LLAMINAR_ATTN_TP_DISABLE"));
    if(const char* tpp = std::getenv("LLAMINAR_ATTN_TP_PARTITIONS")) { int v=std::atoi(tpp); if(v>0 && v<1024) s.attention.tp_partitions = v; }
    s.attention.tp_auto = flag(std::getenv("LLAMINAR_ATTN_TP_AUTO"));
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
    if(const char* fr = std::getenv("LLAMINAR_RMS_FORENSICS_FUSED")) if(*fr) s.rms_fused.forensics = true;
    if(const char* rr2 = std::getenv("LLAMINAR_RMS_FUSED_ROWS")) if(*rr2){ int v=std::atoi(rr2); if(v>0) s.rms_fused.rows_preview = v; }
    if(const char* cc2 = std::getenv("LLAMINAR_RMS_FUSED_COLS")) if(*cc2){ int v=std::atoi(cc2); if(v>0) s.rms_fused.cols_preview = v; }
    if(const char* dl = std::getenv("LLAMINAR_FUSED_RMS_DUMP_LAYER")) if(*dl){ s.rms_fused.dump_layer=true; s.rms_fused.dump_layer_spec = dl; }
    if(const char* eps = std::getenv("LLAMINAR_RMS_EPS_OVERRIDE")) if(*eps){ try { s.rms_fused.eps_override = std::stod(eps); s.rms_fused.eps_override_active=true; } catch(...){} }
    // Embedding warnings
    s.embedding_warn.transpose_warn = flag(std::getenv("LLAMINAR_EMBEDDING_TRANSPOSE_WARN"));
    // Test harness
    s.test_harness.skip_mpi_in_single_test = flag(std::getenv("LLAMINAR_SKIP_MPI_IN_SINGLE_TEST"));

    // Logging
    if(const char* lvl = std::getenv("LLAMINAR_LOG_LEVEL")) { if(*lvl){ s.logging.log_level_active=true; s.logging.log_level = lvl; } }
        return s; }();
        return snap;
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
                        " layerwise_stats=" + on(s.pipeline.layerwise_stats));
        lines.push_back(std::string("[DebugEnv] embedding: trace=") + on(s.embedding.trace) +
                        " fail_fast=" + on(s.embedding.fail_fast) +
                        " trace_tokens=" + std::to_string(s.embedding.trace_tokens) +
                        " trace_dims=" + std::to_string(s.embedding.trace_dims) +
                        (s.embedding.trace_rows_spec.empty() ? "" : " rows_spec=" + s.embedding.trace_rows_spec));
        lines.push_back(std::string("[DebugEnv] rmsnorm: validate_ref=") + on(s.rmsnorm.validate_ref) +
                        " dump_gamma=" + on(s.rmsnorm.dump_gamma) +
                        " force_unit_gamma=" + on(s.rmsnorm.force_unit_gamma) +
                        " gamma_checksum=" + on(s.rmsnorm.gamma_checksum) +
                        (s.rmsnorm.trace_rows_spec.empty() ? "" : " trace_rows=" + s.rmsnorm.trace_rows_spec));
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
        if (s.test_harness.skip_mpi_in_single_test)
        {
            lines.push_back("[DebugEnv] test_harness: skip_mpi_in_single_test=on");
        }
        if (s.logging.log_level_active)
        {
            lines.push_back(std::string("[DebugEnv] logging: level=") + s.logging.log_level);
        }
        return lines;
    }

} // namespace llaminar
