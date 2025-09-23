#pragma once
// COSMA Prefill Manager (Phase 1)
// --------------------------------------------------------------
// This component implements the Phase 1 scope of the COSMA prefill
// integration plan documented in:
//   .github/instructions/cosma-prefill-plan.instructions.md
// Phase 1 goals:
//   - Enable COSMA only for large prefill matmuls (long sequence)
//   - Provide gating via env + sequence length
//   - Stream (float32) weights into distributed layout (no quant fusion yet)
//   - Provide instrumentation (bytes, timings, counters)
//   - Optional validation tile GEMM for early corruption detection
//   - Memory budget guard for single allocations
// Non-goals (deferred): cumulative resident tracking, fused dequant, elementwise kernels in-place.
// --------------------------------------------------------------
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <optional>
#include <cosma/matrix.hpp>
#include <cosma/strategy.hpp>
#include <mpi.h>
#include <atomic>

namespace llaminar
{

    // Forward declarations
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
        std::shared_ptr<cosma::CosmaMatrix<float>> mat; // shared so temporary results survive chaining
        int global_rows = 0;
        int global_cols = 0;
        char label = 'A';
        // Single-rank fallback support
        const float *original_row_major = nullptr;      // points to source data (A or B) if available
        std::shared_ptr<std::vector<float>> host_owned; // for outputs or copies when needed
    };

    struct CosmaWeightHandle
    {
        CosmaView view;
        WeightDescriptor desc;
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

        void to_row_major(const CosmaView &src, float *dst) const;
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
        } stats_;
        const PrefillStats &stats() const { return stats_; }
        const StrategyCache::Stats &strategy_stats() const { return strategy_cache_.stats; }

        // Phase 1b additions
        void dump_stats_json(const std::string &path) const;          // structured stats export
        void reset_stats();                                           // test isolation helper
        static const std::vector<std::string> &recognized_env_vars(); // env audit support

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

        StrategyCache strategy_cache_;

        struct AllocationRecord
        {
            std::weak_ptr<cosma::CosmaMatrix<float>> ref;
            long long bytes{0};
        };
        std::vector<AllocationRecord> allocations_; // tracked allocations for resident memory accounting

        void recalc_resident();

        // Internal helpers
        CosmaView allocate_matrix(char label, int m, int n, const cosma::Strategy &strat, bool zero = false);
        void fill_activation(CosmaView &dst, const float *src_row_major, int m, int k);
        void stream_weight_blocks(CosmaView &dst, const WeightDescriptor &desc);
        void stream_weight_blocks_quantized(CosmaView &dst, const WeightDescriptor &desc);
        void maybe_validation_tile_gemm(const float *A, const float *B, const float *C_ref_full,
                                        int m, int k, int n);
        bool should_log(int lvl) const { return lvl <= log_level_; }
        bool memory_budget_allows(size_t bytes_needed) const;
        bool preflight_allows(int seq_len, int d_model, int d_ff, int n_proj = 3) const; // attention: Q,K,V by default
    };

} // namespace llaminar
