#pragma once
// COSMA Prefill Manager (Phase 1 / 1b)
// --------------------------------------------------------------
// This component implements the Phase 1 + Phase 1b scope of the COSMA prefill
// integration plan documented in:
//   .github/instructions/cosma-prefill-plan.instructions.md
// Phase 1 goals:
//   - Enable COSMA only for large prefill matmuls (long sequence)
//   - Provide gating via env + sequence length
//   - Stream (float32) weights into distributed layout (no quant fusion yet)
//   - Provide instrumentation (bytes, timings, counters)
//   - Optional validation tile GEMM for early corruption detection
//   - Memory budget guard for single allocations
// Phase 1b adds: cumulative resident tracking, JSON stats export, env audit helpers.
// Non-goals (deferred): fused dequant, full in-layout elementwise kernels, stream overlap.
// --------------------------------------------------------------
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <optional>
#include <utility>
#include <cosma/matrix.hpp>
#include <cosma/strategy.hpp>
#include <mpi.h>
#include <atomic>
#include <mutex>

namespace llaminar
{

    // Forward declarations
    class CosmaPrefillManager;

    struct WeightDescriptor
    {
        std::string id{};              // unique layer+projection key
        int rows{0};                   // k (input dim)
        int cols{0};                   // n (output dim)
        int64_t row_stride{0};         // stride in source storage
        int64_t col_stride{0};         // stride for columns if non-contiguous
        int quant_type{0};             // 0=float32 baseline (future: enum >0 means quant format)
        const void *base_ptr{nullptr}; // pointer to start of original tensor (quant or float)
        size_t quant_block_size{0};    // bytes per quant block if quantized
    };

    struct CosmaView
    {
        // Retain prerequisite allocations (typically operand buffers rebuilt for COSMA layout)
        // so their destruction occurs after this view releases its own matrix, preserving the
        // COSMA memory pool's LIFO discipline.
        std::vector<std::shared_ptr<cosma::CosmaMatrix<float>>> release_chain;
        std::shared_ptr<cosma::CosmaMatrix<float>> mat; // shared so temporary results survive chaining
        int global_rows = 0;
        int global_cols = 0;
        char label = 'A';
        const cosma::Strategy *strategy = nullptr; // optional pointer to strategy used when allocated
        std::string strategy_key;                  // optional cache key / descriptor for diagnostics
        // Single-rank fallback support
        const float *original_row_major = nullptr;      // points to source data (A or B) if available
        std::shared_ptr<std::vector<float>> host_owned; // for outputs or copies when needed

        ~CosmaView()
        {
            if (mat)
            {
                mat.reset();
            }
            for (auto it = release_chain.rbegin(); it != release_chain.rend(); ++it)
            {
                if (*it)
                {
                    it->reset();
                }
            }
        }
    };

    struct CosmaWeightHandle
    {
        CosmaView view;
        WeightDescriptor desc;
    };

    struct MatrixDebugSummary
    {
        int rows{0};
        int cols{0};
        int sample_rows{0};
        int sample_cols{0};
        std::vector<float> sample;           // row-major sample (top-left sample_rows x sample_cols)
        std::vector<float> reference_sample; // optional reference sample (e.g., OpenBLAS)
        double rel_l2_vs_original{-1.0};
        double max_abs_vs_original{0.0};
        double rel_l2_vs_reference{-1.0};
        double max_abs_vs_reference{0.0};
    };

    struct MatmulDebugSnapshot
    {
        bool valid{false};
        int world{1};
        int rank{0};
        int m{0};
        int n{0};
        int k{0};
        bool transposeW{false};
        float alpha{1.f};
        float beta{0.f};
        long long volume{0};
        bool used_cosma{false};
        bool used_fast_path{false};
        bool cosma_disabled{false};
        bool force_direct{false};
        bool force_replicated{false};
        bool force_fallback{false};
        bool compare_replicated{false};
        bool direct_override{false};
        long long fast_path_threshold_ops{0};
        long long direct_threshold_ops{0};
        std::string path_desc;
        std::string strategy_A;
        std::string strategy_B;
        std::string strategy_C;
        std::vector<std::string> env_flags;
        MatrixDebugSummary A_summary;
        MatrixDebugSummary B_summary;
        MatrixDebugSummary C_summary;
        std::string notes;
    };

    struct FusedRmsnormQkvResult
    {
        // Retains the activation matrix until final destruction to satisfy COSMA memory pool LIFO semantics.
        CosmaView activation_guard;
        struct WeightGuard
        {
            CosmaPrefillManager *manager{nullptr};
            CosmaWeightHandle handle;

            WeightGuard() = default;
            WeightGuard(CosmaPrefillManager *mgr, CosmaWeightHandle &&h);
            WeightGuard(WeightGuard &&other) noexcept;
            WeightGuard &operator=(WeightGuard &&other) noexcept;
            WeightGuard(const WeightGuard &) = delete;
            WeightGuard &operator=(const WeightGuard &) = delete;
            ~WeightGuard();
        };

        CosmaView normalized;
        WeightGuard wq_guard;
        WeightGuard wk_guard;
        WeightGuard wv_guard;

        CosmaView q;
        CosmaView k;
        CosmaView v;
    };

    struct GemmStrategies
    {
        const cosma::Strategy *A; // strategy chosen for operand A (m x k)
        const cosma::Strategy *B; // strategy chosen for operand B (k x n)
        const cosma::Strategy *C; // unified strategy for result (m x n)
    };

    class StrategyCache
    {
    public:
        const cosma::Strategy &get(int m, int n, int k, int p);
        struct Stats
        {
            std::atomic<long long> hits{0};
            std::atomic<long long> misses{0};
        } stats;

    private:
        std::unordered_map<std::string, cosma::Strategy> cache_;
        std::string make_key(int m, int n, int k, int p) const;
    };

    class CosmaPrefillManager
    {
    public:
        static CosmaPrefillManager &instance();
        ~CosmaPrefillManager();

        bool enabled_for(int seq_len) const; // threshold + env flag + force override

        // Explicit force enable/disable (API knob) irrespective of sequence length threshold.
        // Still requires multi-rank and not ADAPTIVE_DISABLE_COSMA.
        void set_force_cosma(bool enable) { force_cosma_ = enable; }
        bool force_cosma() const { return force_cosma_; }

        // Unified strategy helper: always use the same strategy for A,B,C of an m x n x k GEMM.
        const cosma::Strategy &unified_strategy(int m, int n, int k) { return strategy_cache_.get(m, n, k, world_size_); }

        // Derive per-operand strategies (currently may map to different bucketed keys).
        GemmStrategies derive_strategies(int m, int n, int k)
        {
            auto &sA = strategy_cache_.get(m, k, k, world_size_);
            auto &sB = strategy_cache_.get(k, n, k, world_size_);
            auto &sC = strategy_cache_.get(m, n, k, world_size_);
            return GemmStrategies{&sA, &sB, &sC};
        }

        // Convert row-major activation into COSMA matrix using cached strategy for (m,k)
        CosmaView convert_activation_in(const float *row_major, int m, int k);
        // Explicit strategy-driven conversion (reuse strategy returned for gemm m x n x k)
        CosmaView convert_activation_in_with_strategy(const float *row_major, int m, int k, const cosma::Strategy &strat);
        // Public accessor for strategy caching to allow tests to coordinate A/W/C layouts
        const cosma::Strategy &strategy_for(int m, int n, int k) { return strategy_cache_.get(m, n, k, world_size_); }
        CosmaWeightHandle load_weight(const WeightDescriptor &desc);
        CosmaWeightHandle load_weight_with_strategy(const WeightDescriptor &desc, const cosma::Strategy &strat);
        CosmaView convert_activation_operand(const float *row_major, int m, int k, const cosma::Strategy &strat);
        CosmaWeightHandle load_weight_operand(const WeightDescriptor &desc, const cosma::Strategy &strat);

        CosmaView matmul(const CosmaView &A,
                         const CosmaWeightHandle &W,
                         int m, int k, int n,
                         bool transposeW = false,
                         float alpha = 1.f,
                         float beta = 0.f);

        // Phase 2: In-layout elementwise kernels operating directly on distributed layout.
        // Applies RMSNorm over the last dimension (feature axis) independently for each sequence position.
        // Input/Output: src may equal dst for in-place; weight is length hidden_size.
        // Returns false on validation failure.
        bool rmsnorm_in_layout(const CosmaView &src, CosmaView &dst, const float *weight, int seq_len, int hidden_size, float eps = 1e-5f);

        // Phase 2: SwiGLU activation: out = silu(up) * gate (both same shape [seq_len, hidden]) in layout.
        // gate and up are CosmaView of identical dims; out may alias one of them (in-place not recommended for clarity).
        bool swiglu_in_layout(const CosmaView &gate, const CosmaView &up, CosmaView &out, int seq_len, int hidden_size);

        // Phase 2: In-layout softmax over columns for each row of distributed matrix.
        // Optional scaling factor (commonly 1/sqrt(d_k)) applied before exponentiation.
        bool softmax_in_layout(const CosmaView &scores, CosmaView &dst, int rows, int cols, float scale = 1.0f);

        // Phase 2: Fused RMSNorm -> QKV matmul chain with optional weight streaming overlap.
        FusedRmsnormQkvResult fused_rmsnorm_qkv(const float *activation_row_major,
                                                const float *gamma,
                                                const WeightDescriptor &wq,
                                                const WeightDescriptor &wk,
                                                const WeightDescriptor &wv,
                                                int seq_len,
                                                int hidden_size,
                                                float eps = 1e-5f,
                                                float softmax_scale = 1.0f,
                                                bool transpose_k = false);

        void to_row_major(const CosmaView &src, float *dst, bool force_normalize = false) const;
        // Generic reconstruction helper (optionally normalization when overlapping ownership occurs)
        void reconstruct_matrix(const CosmaView &src, float *dst, bool normalize = true) const;
        // Debug: reconstruct and compare to original row-major (env LLAMINAR_COSMA_DEBUG_RECON)
        void debug_compare_original(const CosmaView &src, int rows, int cols, const float *original) const;
        // Diagnostics: global checksum & norm (activated via LLAMINAR_COSMA_DIAG)
        void diag_global_checksum(const CosmaView &src, const char *tag) const;
        // Diagnostics: sample a set of (row,col) pairs and log value vs original (env LLAMINAR_COSMA_DIAG_SAMPLES="r0,c0;r1,c1")
        void diag_sample_points(const CosmaView &src, const float *original, const char *tag) const;
        // Diagnostics: compare two row-major matrices (A,B) computing rel L2 & max abs; logs if env LLAMINAR_COSMA_DIAG_GEMM
        void diag_compare_row_major(const float *A, const float *B, int m, int n, const char *tagA, const char *tagB) const;
        void release_weight(CosmaWeightHandle &&handle);

        void set_threshold(int t) { threshold_ = t; }

        struct PrefillStats
        {
            std::atomic<long long> single_rank_calls{0};
            std::atomic<long long> fast_path_calls{0};
            std::atomic<long long> cosma_path_calls{0};
            // Instrumentation (aggregated)
            std::atomic<long long> bytes_streamed_weights{0};
            std::atomic<long long> bytes_converted_activations{0};
            std::atomic<long long> matmul_invocations{0};
            std::atomic<long long> validation_tile_checks{0};
            // Time in microseconds (accumulated) for coarse profiling
            std::atomic<long long> us_stream_weights{0};
            std::atomic<long long> us_convert_activation{0};
            std::atomic<long long> us_matmul{0};
            // Phase 1b: cumulative resident memory tracking
            std::atomic<long long> current_resident_bytes{0};
            std::atomic<long long> peak_resident_bytes{0};
            std::atomic<long long> allocations_tracked{0};
            std::atomic<long long> allocations_denied{0};
            // Phase 1b minor: preflight projection counters
            std::atomic<long long> preflight_invocations{0};
            std::atomic<long long> preflight_denied{0};
            std::atomic<long long> preflight_estimated_bytes_last{0};
            // Phase 2: fused dequant + direct layout
            std::atomic<long long> fused_dequant_invocations{0};
            std::atomic<long long> fused_dequant_elements{0};
            // Phase 2: in-layout softmax and fused chains
            std::atomic<long long> softmax_invocations{0};
            std::atomic<long long> softmax_rows_processed{0};
            std::atomic<long long> us_softmax{0};
            std::atomic<long long> fused_rmsnorm_qkv_invocations{0};
            std::atomic<long long> us_fused_rmsnorm_qkv{0};
            std::atomic<long long> mixed_zero_tile_fallbacks{0};
            std::atomic<long long> overlap_stream_invocations{0};
            std::atomic<long long> us_overlap_stream{0};
        } stats_;
        const PrefillStats &stats() const { return stats_; }
        const StrategyCache::Stats &strategy_stats() const { return strategy_cache_.stats; }

        // Phase 1b additions
        void dump_stats_json(const std::string &path) const;          // structured stats export
        void dump_stats_if_requested() const;                         // honour LLAMINAR_COSMA_DUMP_STATS
        void dump_gemm_snapshots_json(const std::string &path) const; // write captured GEMM snapshots (JSON)
        void dump_gemm_snapshots_if_requested() const;                // honour LLAMINAR_COSMA_DUMP_GEMM_SNAPSHOTS
        void reset_stats();                                           // test isolation helper
        static const std::vector<std::string> &recognized_env_vars(); // env audit support

        // Debug capture helpers
        void enable_last_gemm_capture(bool enable, int sample_dim = 8, size_t depth = 16);
        void set_capture_config(bool enable, int sample_dim, size_t depth);
        void clear_recent_gemm_snapshots();
        std::vector<MatmulDebugSnapshot> recent_gemm_snapshots(size_t limit = 0) const;

    private:
        CosmaPrefillManager();
        int world_size_ = 1;
        int rank_ = 0;
        int threshold_ = 4096;                                   // default; overridden by env
        long long fast_path_threshold_ops_ = 64ll * 64ll * 64ll; // default fast path volume (m*n*k) below which we bypass COSMA multi-rank
        int validate_tile_tokens_ = 0;                           // env LLAMINAR_COSMA_VALIDATE_TILE (>0 enables)
        int log_level_ = 2;                                      // 0=error 1=warn 2=info 3=debug 4=trace (mapped from LLAMINAR_COSMA_LOG_LEVEL)
        long long max_resident_mb_ = 2048;                       // LLAMINAR_COSMA_MAX_RESIDENT_MB
        bool force_cosma_ = false;                               // Env: LLAMINAR_COSMA_FORCE (or API set)

        bool capture_last_gemm_{false};
        int capture_sample_dim_{8};
        size_t capture_depth_{16};
        mutable std::mutex debug_mutex_;
        std::deque<MatmulDebugSnapshot> gemm_debug_ring_;
        mutable std::atomic<bool> snapshot_dirty_{false};

        StrategyCache strategy_cache_;

        struct EnvSnapshot
        {
            bool cosma_disabled = false;
            bool adaptive_disabled = false;
            bool diag_enabled = false;
            bool diag_deep = false;
            bool diag_axis = false;
            bool diag_coord_invert = false;
            bool diag_local_probe = false;
            bool diag_local_probe_deep = false;
            bool diag_recon_bypass = false;
            bool diag_recon_transpose = false;
            bool diag_recon_brute = false;
            bool diag_recon_map = false;
            bool recon_force_legacy = false;
            bool diag_swaprc = false;
            bool diag_try_transpose = false;
            bool diag_skip_norm = false;
            bool debug_recon = false;
            bool compare_replicated = false;
            bool diag_dump_small = false;
            bool pop_forward_legacy = false;
            bool force_fallback = false;
            bool force_distributed_act = false;
            bool fast_unverified = false;
            bool disable_fused_dequant = false;
            bool force_replicated_diag = false;
            bool force_replicated = false;
            bool force_direct = false;
            bool replicate_B = false;
            bool auto_fix_transpose = false;
            bool force_unified_strategy = false;
            bool overlap_enabled = false;
            bool overlap_verbose = false;
            bool preflight_disable = false;
            bool rmsnorm_validate = false;
            bool rmsnorm_trace = false;
            bool rmsnorm_trace_points_active = false;
            std::string rmsnorm_trace_points_spec;
            bool diag_perm_infer_active = false;
            bool diag_samples_active = false;
            bool preflight_safety_override = false;
            bool direct_threshold_override = false;
            bool diag_tap_enabled = false;
            int diag_tap_value = 0;
            long long direct_threshold_ops = 0;
            double preflight_safety = 1.2;
            int forced_openblas_threads = -1;
            int forced_replicated_threads = -1;
            std::string diag_perm_spec;
            std::string diag_samples_spec;
        };

        struct AllocationRecord
        {
            std::weak_ptr<cosma::CosmaMatrix<float>> ref;
            long long bytes{0};
        };
        std::vector<AllocationRecord> allocations_; // tracked allocations for resident memory accounting
        mutable std::mutex allocations_mutex_;

        void recalc_resident();
        EnvSnapshot capture_env_snapshot() const;
        const EnvSnapshot &resolve_env(const EnvSnapshot *override_env, EnvSnapshot &storage) const;
        bool snapshot_dump_requested(std::string &path) const;

        // Internal helpers
        CosmaView allocate_matrix(char label, int m, int n, const cosma::Strategy &strat, bool zero = false);
        void fill_activation(CosmaView &dst, const float *src_row_major, int m, int k, const EnvSnapshot &env);
        void scatter_row_major_dest_local(CosmaView &dst, const float *src_row_major, int rows, int cols);
        void stream_weight_blocks(CosmaView &dst, const WeightDescriptor &desc, const EnvSnapshot &env);
        void stream_weight_blocks_quantized(CosmaView &dst, const WeightDescriptor &desc, const EnvSnapshot &env);
        void maybe_validation_tile_gemm(const float *A, const float *B, const float *C_ref_full,
                                        int m, int k, int n, bool transposeB);
        bool should_log(int lvl) const { return lvl <= log_level_; }
        bool memory_budget_allows(size_t bytes_needed) const;
        bool preflight_allows(int seq_len, int d_model, int d_ff, int n_proj = 3) const; // attention: Q,K,V by default
        void debug_compare_original_impl(const CosmaView &src, int rows, int cols, const float *original, const EnvSnapshot &env) const;
        void diag_global_checksum_impl(const CosmaView &src, const char *tag, const EnvSnapshot &env) const;
        void diag_sample_points_impl(const CosmaView &src, const float *original, const char *tag, const EnvSnapshot &env) const;
        void diag_compare_row_major_impl(const float *A, const float *B, int m, int n, const char *tagA, const char *tagB, const EnvSnapshot &env) const;
        void reconstruct_matrix_impl(const CosmaView &src, float *dst, bool normalize, const EnvSnapshot &env) const;
        void capture_gemm_debug(MatmulDebugSnapshot &snapshot,
                                const CosmaView &A,
                                const CosmaView &B,
                                const WeightDescriptor &w_desc,
                                const CosmaView &C,
                                const EnvSnapshot &env) const;
        void record_snapshot(MatmulDebugSnapshot &&snapshot);
        std::vector<std::string> active_env_flags(const EnvSnapshot &env) const;
        static std::string describe_strategy(const cosma::Strategy *strat);
    };

} // namespace llaminar
