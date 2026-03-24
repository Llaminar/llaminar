/**
 * @file Perf__CpuFlashAttentionSweep.cpp
 * @brief Comprehensive CPU Flash Attention performance sweep across Qwen model
 *        sizes (0.5B–32B), TP degrees (1/2/4), KV context lengths, and both
 *        decode (seq_len=1) and prefill (seq_len=N) workloads.
 *
 * Reports for each configuration:
 *   - Latency (µs) with min/mean/stddev
 *   - Memory bandwidth (GB/s) — bytes touched / time
 *   - Roofline efficiency (%) — vs calibrated DRAM peak
 *   - GFLOP/s throughput
 *   - Per-phase breakdown (QK vs V) when LLAMINAR_PROFILING=1
 *
 * Model configurations: Qwen 2.5 family (0.5B, 1.5B, 3B, 7B, 14B, 32B).
 *
 * Decode bandwidth model (seq_len=1, dominant cost):
 *   - QK phase: kv_len × n_kv_heads × head_dim × 4B (K reads)
 *   - V phase:  kv_len × n_kv_heads × head_dim × 4B (V reads)
 *   - Total ≈ 2 × kv_len × n_kv_heads × head_dim × 4B
 *
 * System roofline: Calibrated at startup via STREAM-like DRAM read benchmark,
 * identical to the GEMV sweep methodology.
 *
 * @note Run with Release build:
 *   ctest -R V2_Perf_CPUFlashAttentionSweep --verbose
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

#include "fort.hpp"
#include "kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "utils/KernelProfiler.h"
#include "utils/Logger.h"

using namespace llaminar2;

namespace
{

    // =========================================================================
    // MPI environment
    // =========================================================================
    void mpi_abort_handler(int sig)
    {
        const char *msg = "\n[FATAL] Signal in attention perf test — MPI_Abort\n";
        [[maybe_unused]] auto _ = write(STDERR_FILENO, msg, strlen(msg));
        MPI_Abort(MPI_COMM_WORLD, sig);
        _exit(128 + sig);
    }

    class MPIEnvironment : public ::testing::Environment
    {
    public:
        void SetUp() override
        {
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (!initialized)
                MPI_Init(nullptr, nullptr);
            std::signal(SIGSEGV, mpi_abort_handler);
            std::signal(SIGABRT, mpi_abort_handler);
        }
        void TearDown() override
        {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized)
                MPI_Finalize();
        }
    };

    static auto *g_mpi_env [[maybe_unused]] =
        ::testing::AddGlobalTestEnvironment(new MPIEnvironment);

    // =========================================================================
    // Cache-flush utility
    // =========================================================================
    static inline void flush_cache_range(const void *ptr, size_t bytes)
    {
#if defined(__x86_64__) || defined(_M_X64)
        const char *p = static_cast<const char *>(ptr);
        for (size_t off = 0; off < bytes; off += 64)
        {
            _mm_clflushopt(const_cast<char *>(p + off));
        }
        _mm_mfence();
#else
        (void)ptr;
        (void)bytes;
#endif
    }

    // =========================================================================
    // DRAM bandwidth calibration (STREAM-like)
    // =========================================================================
    static double calibrateDramBandwidth()
    {
        constexpr size_t BUF_SIZE = 256 * 1024 * 1024; // 256 MB
        constexpr int CALIB_RUNS = 5;

        auto *buf = static_cast<char *>(std::aligned_alloc(4096, BUF_SIZE));
        if (!buf)
            return 60.0;

        // First-touch for NUMA locality
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < BUF_SIZE; i += 4096)
            buf[i] = static_cast<char>(i & 0xFF);

        double best_gbs = 0;
        for (int run = 0; run < CALIB_RUNS; ++run)
        {
            flush_cache_range(buf, BUF_SIZE);

            volatile uint64_t sink = 0;
            auto t0 = std::chrono::high_resolution_clock::now();

            uint64_t local_sum = 0;
#pragma omp parallel reduction(+ : local_sum)
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();
                size_t chunk = BUF_SIZE / nthreads;
                size_t start = tid * chunk;
                size_t end = (tid == nthreads - 1) ? BUF_SIZE : start + chunk;
                const uint64_t *p = reinterpret_cast<const uint64_t *>(buf + start);
                const uint64_t *pe = reinterpret_cast<const uint64_t *>(buf + end);
                uint64_t acc = 0;
                while (p < pe)
                {
                    acc += *p;
                    p += 8; // stride 64B (cache line)
                }
                local_sum += acc;
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            sink = local_sum;
            (void)sink;

            double secs = std::chrono::duration<double>(t1 - t0).count();
            double gbs = static_cast<double>(BUF_SIZE) / secs / 1e9;
            best_gbs = std::max(best_gbs, gbs);
        }

        std::free(buf);
        return best_gbs;
    }

    static double s_calibrated_bw_gbs = 0;

    static double systemBandwidthGB()
    {
        if (s_calibrated_bw_gbs <= 0)
        {
            s_calibrated_bw_gbs = calibrateDramBandwidth();
            std::cout << "\n=== DRAM Bandwidth Calibration ===\n"
                      << "Measured: " << std::fixed << std::setprecision(1)
                      << s_calibrated_bw_gbs << " GB/s"
                      << "  (threads=" << omp_get_max_threads() << ")\n"
                      << std::endl;
        }
        return s_calibrated_bw_gbs;
    }

    // =========================================================================
    // Model definitions — Qwen2.5 family (0.5B through 32B)
    // =========================================================================
    struct ModelConfig
    {
        const char *name;
        int d_model;
        int n_heads;
        int n_kv_heads;
        int head_dim;
    };

    static constexpr ModelConfig ALL_MODELS[] = {
        {"0.5B", 896, 14, 2, 64},
        {"1.5B", 1536, 12, 2, 128},
        {"3B", 2048, 16, 2, 128},
        {"7B", 3584, 28, 4, 128},
        {"14B", 5120, 40, 8, 128},
        {"32B", 5120, 40, 8, 128},
    };

    // =========================================================================
    // Attention workload configuration
    // =========================================================================
    struct AttnBenchConfig
    {
        ModelConfig model;
        int seq_len;   // 1 for decode, >1 for prefill
        int kv_len;    // context length (≥ seq_len)
        int tp_degree; // tensor parallelism degree (affects n_heads, n_kv_heads)
        bool causal;
        int warmup;
        int iters;
    };

    // =========================================================================
    // Bandwidth model
    // =========================================================================

    /// Bytes of memory traffic for the attention operation.
    /// Decode (seq_len=1): dominated by KV cache reads.
    /// Prefill: Q reads + KV reads + output writes.
    static double attention_bytes(int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim, bool causal)
    {
        // Causal halving only applies to prefill: for seq_len=1 (decode), the single query
        // at the end attends to ALL kv_len positions — no triangular mask effect.
        bool effective_causal = causal && (seq_len > 1);
        double effective_kv = effective_causal ? (static_cast<double>(kv_len) + 1.0) / 2.0 : kv_len;

        // Q: read once per head across all kv tiles — seq_len × n_heads × head_dim × 4B
        // But for decode (seq_len=1), Q fits in registers, so KV dominates.
        double q_bytes = static_cast<double>(seq_len) * n_heads * head_dim * sizeof(float);

        // K: each KV position read seq_len times (once per query position per tile).
        // But flash attention reads K in tiles, and for decode seq_len=1 → read each K once.
        // Total K reads: kv_len × n_kv_heads × head_dim × 4B (each K row read once per Q row, but
        // the dominant cost is streaming KV from DRAM).
        double k_bytes = effective_kv * n_kv_heads * head_dim * sizeof(float);
        double v_bytes = effective_kv * n_kv_heads * head_dim * sizeof(float);

        // Output: n_heads × head_dim × seq_len × 4B (write)
        double out_bytes = static_cast<double>(seq_len) * n_heads * head_dim * sizeof(float);

        return q_bytes + k_bytes + v_bytes + out_bytes;
    }

    /// FLOPs for the attention operation.
    static double attention_flops(int seq_len, int kv_len, int n_heads, int head_dim, bool causal)
    {
        bool effective_causal = causal && (seq_len > 1);
        double effective_kv = effective_causal ? (static_cast<double>(kv_len) + 1.0) / 2.0 : kv_len;
        // QK: 2 × seq_len × effective_kv × head_dim × n_heads (mul-add)
        double qk = 2.0 * seq_len * effective_kv * head_dim * n_heads;
        // Softmax: ~5 ops per position
        double sm = 5.0 * seq_len * effective_kv * n_heads;
        // AV: 2 × seq_len × effective_kv × head_dim × n_heads
        double av = 2.0 * seq_len * effective_kv * head_dim * n_heads;
        return qk + sm + av;
    }

    // =========================================================================
    // Benchmark result
    // =========================================================================
    struct BenchResult
    {
        double mean_us;
        double min_us;
        double max_us;
        double stddev_us;
        double gflops;
        double bw_gbs;       // effective memory bandwidth (GB/s)
        double roofline_pct; // % of calibrated DRAM peak
    };

    // =========================================================================
    // Random fill utility
    // =========================================================================
    static void fill_random(float *data, size_t n, unsigned seed)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (size_t i = 0; i < n; ++i)
            data[i] = dist(gen);
    }

    // =========================================================================
    // Core benchmark runner
    // =========================================================================
    static BenchResult run_attention_bench(const AttnBenchConfig &cfg)
    {
        // Apply TP sharding to head counts
        const int n_heads = cfg.model.n_heads / cfg.tp_degree;
        const int n_kv_heads = std::max(1, cfg.model.n_kv_heads / cfg.tp_degree);
        const int head_dim = cfg.model.head_dim;

        const int q_stride = n_heads * head_dim;
        const int kv_stride = n_kv_heads * head_dim;
        const size_t q_size = static_cast<size_t>(cfg.seq_len) * q_stride;
        const size_t kv_size = static_cast<size_t>(cfg.kv_len) * kv_stride;
        const size_t out_size = static_cast<size_t>(cfg.seq_len) * q_stride;

        std::vector<float> Q(q_size), K(kv_size), V(kv_size), output(out_size);
        fill_random(Q.data(), q_size, 42);
        fill_random(K.data(), kv_size, 43);
        fill_random(V.data(), kv_size, 44);

        const int position_offset = (cfg.seq_len == 1) ? cfg.kv_len - 1 : 0;
        const bool is_decode = (cfg.seq_len == 1 && cfg.kv_len > 1);

        CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;

        auto run_once = [&]()
        {
            if (is_decode)
            {
                kernel.compute_decode(
                    Q.data(), K.data(), V.data(), output.data(),
                    cfg.seq_len, cfg.kv_len,
                    n_heads, n_kv_heads, head_dim,
                    cfg.causal, position_offset);
            }
            else
            {
                kernel.compute(
                    Q.data(), K.data(), V.data(), output.data(),
                    cfg.seq_len, n_heads, n_kv_heads, head_dim,
                    cfg.causal);
            }
        };

        // Warmup
        for (int i = 0; i < cfg.warmup; ++i)
            run_once();

        // Timed iterations
        std::vector<double> times_us;
        times_us.reserve(cfg.iters);

        for (int i = 0; i < cfg.iters; ++i)
        {
            // For decode with large KV, flush KV cache from CPU cache to measure
            // true DRAM-bound performance.
            if (is_decode && kv_size * sizeof(float) > 256 * 1024)
            {
                flush_cache_range(K.data(), kv_size * sizeof(float));
                flush_cache_range(V.data(), kv_size * sizeof(float));
            }

            auto start = std::chrono::high_resolution_clock::now();
            run_once();
            auto end = std::chrono::high_resolution_clock::now();
            times_us.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }

        // Statistics
        double sum = std::accumulate(times_us.begin(), times_us.end(), 0.0);
        double mean = sum / times_us.size();
        double sq_sum = 0.0;
        for (auto t : times_us)
            sq_sum += (t - mean) * (t - mean);
        double stddev = std::sqrt(sq_sum / times_us.size());
        double mn = *std::min_element(times_us.begin(), times_us.end());
        double mx = *std::max_element(times_us.begin(), times_us.end());

        double flops = attention_flops(cfg.seq_len, cfg.kv_len, n_heads, head_dim, cfg.causal);
        double gflops = flops / (mean * 1e3);

        double bytes = attention_bytes(cfg.seq_len, cfg.kv_len, n_heads, n_kv_heads, head_dim, cfg.causal);
        double bw_gbs = bytes / (mean * 1e3);
        double roofline_pct = (bw_gbs / systemBandwidthGB()) * 100.0;

        return {mean, mn, mx, stddev, gflops, bw_gbs, roofline_pct};
    }

    // =========================================================================
    // Reference attention for accuracy checks
    // =========================================================================
    static void reference_attention(
        const float *Q, const float *K, const float *V, float *output,
        int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
        bool causal, int position_offset)
    {
        const int heads_per_kv = n_heads / n_kv_heads;
        const int q_stride = n_heads * head_dim;
        const int kv_stride = n_kv_heads * head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        for (int h = 0; h < n_heads; ++h)
        {
            const int kv_h = h / heads_per_kv;
            for (int q = 0; q < seq_len; ++q)
            {
                const int q_abs = position_offset + q;
                const float *q_ptr = Q + static_cast<size_t>(q) * q_stride + static_cast<size_t>(h) * head_dim;
                float *out = output + static_cast<size_t>(q) * q_stride + static_cast<size_t>(h) * head_dim;

                std::vector<float> scores(kv_len);
                float max_s = -std::numeric_limits<float>::infinity();
                for (int k = 0; k < kv_len; ++k)
                {
                    if (causal && k > q_abs)
                    {
                        scores[k] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    float dot = 0.0f;
                    const float *k_ptr = K + static_cast<size_t>(k) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
                    for (int d = 0; d < head_dim; ++d)
                        dot += q_ptr[d] * k_ptr[d];
                    scores[k] = dot * scale;
                    max_s = std::max(max_s, scores[k]);
                }

                float sum_exp = 0.0f;
                std::fill(out, out + head_dim, 0.0f);
                for (int k = 0; k < kv_len; ++k)
                {
                    if (!std::isfinite(scores[k]))
                        continue;
                    float p = std::exp(scores[k] - max_s);
                    sum_exp += p;
                    const float *v_ptr = V + static_cast<size_t>(k) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
                    for (int d = 0; d < head_dim; ++d)
                        out[d] += p * v_ptr[d];
                }
                for (int d = 0; d < head_dim; ++d)
                    out[d] /= sum_exp;
            }
        }
    }

    static float compute_max_abs_error(const float *a, const float *b, size_t n)
    {
        float max_err = 0.0f;
        for (size_t i = 0; i < n; ++i)
            max_err = std::max(max_err, std::abs(a[i] - b[i]));
        return max_err;
    }

    static float compute_cosine_sim(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
    }

} // namespace

// ============================================================================
// Test fixture
// ============================================================================

class Perf__CpuFlashAttentionSweep : public ::testing::Test
{
protected:
    static std::string fmt_us(double v)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(0) << v;
        return oss.str();
    }
    static std::string fmt_1(double v)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << v;
        return oss.str();
    }
    static std::string fmt_2(double v)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << v;
        return oss.str();
    }
    static std::string fmt_pct(double v)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(0) << v << "%";
        return oss.str();
    }
};

// ============================================================================
// DECODE: KV-length scaling with bandwidth analysis (all models, TP=1)
// ============================================================================

TEST_F(Perf__CpuFlashAttentionSweep, Decode_KVLengthScaling_AllModels)
{
    const int kv_lengths[] = {64, 128, 256, 512, 1024, 2048, 4096};
    const int warmup = 200;
    const int iters = 500;

    // Trigger calibration first
    systemBandwidthGB();

    for (const auto &model : ALL_MODELS)
    {
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "kv_len" << "Mean µs" << "Min µs" << "Stddev"
              << "BW GB/s" << "R%" << "GFLOP/s" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::right);
        for (int c = 1; c <= 6; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (int kv : kv_lengths)
        {
            AttnBenchConfig cfg{model, 1, kv, 1, true, warmup, iters};
            auto r = run_attention_bench(cfg);

            table << std::to_string(kv) << fmt_us(r.mean_us) << fmt_us(r.min_us)
                  << fmt_1(r.stddev_us) << fmt_1(r.bw_gbs) << fmt_pct(r.roofline_pct)
                  << fmt_2(r.gflops) << fort::endr;
        }

        std::cout << "\nDECODE KV-Length Scaling: " << model.name
                  << " (h=" << model.n_heads << " kv_h=" << model.n_kv_heads
                  << " d=" << model.head_dim << ")\n"
                  << table.to_string() << std::endl;
    }
}

// ============================================================================
// DECODE: TP degree sweep (7B model, representative kv_lengths)
// ============================================================================

TEST_F(Perf__CpuFlashAttentionSweep, Decode_TPDegreeSweep)
{
    const int tp_degrees[] = {1, 2, 4};
    const int kv_lengths[] = {128, 512, 1024, 2048};
    const int warmup = 200;
    const int iters = 500;

    systemBandwidthGB();

    for (const auto &model : ALL_MODELS)
    {
        // Skip models where TP>1 would give < 1 KV head
        if (model.n_kv_heads < 2)
            continue;

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Model" << "TP" << "kv_len"
              << "Heads" << "KV Heads" << "Mean µs" << "BW GB/s" << "R%" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 7; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (int tp : tp_degrees)
        {
            // Skip if TP degree doesn't divide evenly
            if (model.n_heads % tp != 0 || model.n_kv_heads % tp != 0)
                continue;
            if (model.n_kv_heads / tp < 1)
                continue;

            for (int kv : kv_lengths)
            {
                AttnBenchConfig cfg{model, 1, kv, tp, true, warmup, iters};
                auto r = run_attention_bench(cfg);

                int local_heads = model.n_heads / tp;
                int local_kv_heads = model.n_kv_heads / tp;

                table << model.name << std::to_string(tp) << std::to_string(kv)
                      << std::to_string(local_heads) << std::to_string(local_kv_heads)
                      << fmt_us(r.mean_us) << fmt_1(r.bw_gbs) << fmt_pct(r.roofline_pct)
                      << fort::endr;
            }
        }

        std::cout << "\nDECODE TP Sweep: " << model.name << "\n"
                  << table.to_string() << std::endl;
    }
}

// ============================================================================
// DECODE: 7B single-layer latency projection
// ============================================================================

TEST_F(Perf__CpuFlashAttentionSweep, Decode_7B_LayerProjection)
{
    // Focus on the 7B model to project per-layer and per-token attention cost.
    // The 7B model has 28 layers — total attention time = 28 × single-layer time.
    const auto &model = ALL_MODELS[3]; // 7B
    const int kv_lengths[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    const int warmup = 200;
    const int iters = 500;
    const int n_layers = 28;

    systemBandwidthGB();

    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);
    table << fort::header << "kv_len" << "1-Layer µs" << "28-Layer ms"
          << "Proj tok/s" << "BW GB/s" << "R%" << fort::endr;

    table.column(0).set_cell_text_align(fort::text_align::right);
    for (int c = 1; c <= 5; ++c)
        table.column(c).set_cell_text_align(fort::text_align::right);

    for (int kv : kv_lengths)
    {
        AttnBenchConfig cfg{model, 1, kv, 1, true, warmup, iters};
        auto r = run_attention_bench(cfg);

        double total_ms = r.mean_us * n_layers / 1000.0;
        // Inverse: "if attention alone were the bottleneck, N tok/s"
        double proj_toks = 1000.0 / total_ms;

        table << std::to_string(kv) << fmt_us(r.mean_us) << fmt_1(total_ms)
              << fmt_1(proj_toks) << fmt_1(r.bw_gbs) << fmt_pct(r.roofline_pct)
              << fort::endr;
    }

    std::cout << "\n7B DECODE: Attention-Only Layer Projection (28 layers)\n"
              << table.to_string() << std::endl;
}

// ============================================================================
// PREFILL: Seq-length scaling with bandwidth and GFLOP/s (all models)
// ============================================================================

TEST_F(Perf__CpuFlashAttentionSweep, Prefill_SeqLenScaling_AllModels)
{
    const int warmup = 3;
    const int iters = 10;

    systemBandwidthGB();

    struct PrefillSpec
    {
        int model_idx;
        int seq_len;
    };
    const PrefillSpec specs[] = {
        // Small models: longer sequences
        {0, 32},
        {0, 128},
        {0, 256},
        {0, 512},
        {1, 32},
        {1, 128},
        {1, 256},
        {1, 512},
        {2, 32},
        {2, 128},
        {2, 256},
        // Large models: shorter sequences (memory limited)
        {3, 32},
        {3, 128},
        {3, 256},
        {4, 32},
        {4, 128},
        {5, 32},
        {5, 128},
    };

    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);
    table << fort::header << "Model" << "seq_len" << "Mean µs" << "Min µs"
          << "BW GB/s" << "GFLOP/s" << "R%" << fort::endr;

    table.column(0).set_cell_text_align(fort::text_align::left);
    for (int c = 1; c <= 6; ++c)
        table.column(c).set_cell_text_align(fort::text_align::right);

    for (const auto &sp : specs)
    {
        const auto &model = ALL_MODELS[sp.model_idx];
        AttnBenchConfig cfg{model, sp.seq_len, sp.seq_len, 1, true, warmup, iters};
        auto r = run_attention_bench(cfg);

        table << model.name << std::to_string(sp.seq_len)
              << fmt_us(r.mean_us) << fmt_us(r.min_us) << fmt_1(r.bw_gbs)
              << fmt_2(r.gflops) << fmt_pct(r.roofline_pct) << fort::endr;
    }

    std::cout << "\nPREFILL: Seq-Length Scaling (causal, all models)\n"
              << table.to_string() << std::endl;
}

// ============================================================================
// PREFILL: TP degree sweep (7B, 14B)
// ============================================================================

TEST_F(Perf__CpuFlashAttentionSweep, Prefill_TPDegreeSweep)
{
    const int tp_degrees[] = {1, 2, 4};
    const int seq_lengths[] = {128, 256};
    const int warmup = 3;
    const int iters = 10;

    systemBandwidthGB();

    // Only test models with enough KV heads for TP
    const int model_indices[] = {3, 4, 5}; // 7B, 14B, 32B

    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);
    table << fort::header << "Model" << "TP" << "seq_len"
          << "Heads" << "Mean µs" << "GFLOP/s" << "Speedup" << fort::endr;

    table.column(0).set_cell_text_align(fort::text_align::left);
    for (int c = 1; c <= 6; ++c)
        table.column(c).set_cell_text_align(fort::text_align::right);

    for (int mi : model_indices)
    {
        const auto &model = ALL_MODELS[mi];
        for (int sl : seq_lengths)
        {
            double tp1_us = 0;
            for (int tp : tp_degrees)
            {
                if (model.n_heads % tp != 0 || model.n_kv_heads % tp != 0)
                    continue;
                if (model.n_kv_heads / tp < 1)
                    continue;

                AttnBenchConfig cfg{model, sl, sl, tp, true, warmup, iters};
                auto r = run_attention_bench(cfg);

                if (tp == 1)
                    tp1_us = r.mean_us;

                double speedup = (tp1_us > 0) ? tp1_us / r.mean_us : 1.0;
                int local_heads = model.n_heads / tp;

                table << model.name << std::to_string(tp) << std::to_string(sl)
                      << std::to_string(local_heads) << fmt_us(r.mean_us)
                      << fmt_2(r.gflops) << fmt_2(speedup) + "x" << fort::endr;
            }
        }
    }

    std::cout << "\nPREFILL: TP Degree Sweep\n"
              << table.to_string() << std::endl;
}

// ============================================================================
// DECODE: Model comparison at fixed kv_len (bandwidth-focused)
// ============================================================================

TEST_F(Perf__CpuFlashAttentionSweep, Decode_ModelComparison)
{
    const int kv_len = 512;
    const int warmup = 200;
    const int iters = 500;

    systemBandwidthGB();

    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);
    table << fort::header << "Model" << "Heads" << "KV Heads" << "head_dim"
          << "Mean µs" << "BW GB/s" << "R%" << "GFLOP/s" << fort::endr;

    table.column(0).set_cell_text_align(fort::text_align::left);
    for (int c = 1; c <= 7; ++c)
        table.column(c).set_cell_text_align(fort::text_align::right);

    for (const auto &model : ALL_MODELS)
    {
        AttnBenchConfig cfg{model, 1, kv_len, 1, true, warmup, iters};
        auto r = run_attention_bench(cfg);

        table << model.name << std::to_string(model.n_heads) << std::to_string(model.n_kv_heads)
              << std::to_string(model.head_dim) << fmt_us(r.mean_us) << fmt_1(r.bw_gbs)
              << fmt_pct(r.roofline_pct) << fmt_2(r.gflops) << fort::endr;
    }

    std::cout << "\nDECODE: Model Comparison (kv_len=" << kv_len << ", causal)\n"
              << table.to_string() << std::endl;
}

// ============================================================================
// ACCURACY: Flash attention vs scalar reference
// ============================================================================

TEST_F(Perf__CpuFlashAttentionSweep, Accuracy_DecodeAndPrefill)
{
    struct AccuracyConfig
    {
        int model_idx;
        int seq_len;
        int kv_len;
        const char *label;
    };

    const AccuracyConfig configs[] = {
        {0, 1, 64, "0.5B decode kv=64"},
        {0, 1, 256, "0.5B decode kv=256"},
        {0, 1, 1024, "0.5B decode kv=1024"},
        {0, 32, 32, "0.5B prefill sl=32"},
        {0, 128, 128, "0.5B prefill sl=128"},
        {1, 1, 128, "1.5B decode kv=128"},
        {1, 1, 512, "1.5B decode kv=512"},
        {1, 64, 64, "1.5B prefill sl=64"},
        {2, 1, 256, "3B decode kv=256"},
        {3, 1, 512, "7B decode kv=512"},
        {3, 1, 2048, "7B decode kv=2048"},
        {3, 32, 32, "7B prefill sl=32"},
        {4, 1, 512, "14B decode kv=512"},
        {5, 1, 512, "32B decode kv=512"},
    };

    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);
    table << fort::header << "Config" << "Max AbsErr" << "Cosine Sim" << "Status" << fort::endr;
    table.column(0).set_cell_text_align(fort::text_align::left);
    table.column(1).set_cell_text_align(fort::text_align::right);
    table.column(2).set_cell_text_align(fort::text_align::right);
    table.column(3).set_cell_text_align(fort::text_align::center);

    bool all_pass = true;
    for (const auto &c : configs)
    {
        const auto &model = ALL_MODELS[c.model_idx];
        const int q_stride = model.n_heads * model.head_dim;
        const int kv_stride = model.n_kv_heads * model.head_dim;
        const size_t q_size = static_cast<size_t>(c.seq_len) * q_stride;
        const size_t kv_size = static_cast<size_t>(c.kv_len) * kv_stride;
        const size_t out_size = static_cast<size_t>(c.seq_len) * q_stride;

        std::vector<float> Q(q_size), K(kv_size), V(kv_size);
        std::vector<float> flash_out(out_size, 0.0f), ref_out(out_size, 0.0f);
        fill_random(Q.data(), q_size, 100);
        fill_random(K.data(), kv_size, 101);
        fill_random(V.data(), kv_size, 102);

        const int position_offset = (c.seq_len == 1) ? c.kv_len - 1 : 0;
        const bool is_decode = (c.seq_len == 1 && c.kv_len > 1);

        CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
        if (is_decode)
        {
            kernel.compute_decode(
                Q.data(), K.data(), V.data(), flash_out.data(),
                c.seq_len, c.kv_len,
                model.n_heads, model.n_kv_heads, model.head_dim,
                true, position_offset);
        }
        else
        {
            kernel.compute(
                Q.data(), K.data(), V.data(), flash_out.data(),
                c.seq_len, model.n_heads, model.n_kv_heads, model.head_dim,
                true);
        }

        reference_attention(
            Q.data(), K.data(), V.data(), ref_out.data(),
            c.seq_len, c.kv_len,
            model.n_heads, model.n_kv_heads, model.head_dim,
            true, position_offset);

        float max_err = compute_max_abs_error(flash_out.data(), ref_out.data(), out_size);
        float cos_sim = compute_cosine_sim(flash_out.data(), ref_out.data(), out_size);

        bool pass = (max_err < 1e-4f) && (cos_sim > 0.99999f);
        all_pass &= pass;

        std::ostringstream err_str, cos_str;
        err_str << std::scientific << std::setprecision(2) << max_err;
        cos_str << std::fixed << std::setprecision(6) << cos_sim;

        table << c.label << err_str.str() << cos_str.str()
              << (pass ? "✓" : "✗") << fort::endr;
    }

    std::cout << "\nACCURACY: Flash Attention vs Scalar Reference\n"
              << table.to_string() << std::endl;

    EXPECT_TRUE(all_pass) << "One or more accuracy checks failed";
}

// ============================================================================
// DECODE: All models summary table (one-line-per-model at key kv_len)
// ============================================================================

TEST_F(Perf__CpuFlashAttentionSweep, Decode_Summary)
{
    // Produce a concise summary across all models at kv_len=512 (common inference scenario).
    // Reports both per-layer and projected 28-layer full-model attention time.
    const int kv_len = 512;
    const int warmup = 200;
    const int iters = 500;

    systemBandwidthGB();

    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);
    table << fort::header << "Model" << "Config" << "1-Layer µs" << "BW GB/s"
          << "R%" << "Layers" << "Full ms" << fort::endr;

    table.column(0).set_cell_text_align(fort::text_align::left);
    table.column(1).set_cell_text_align(fort::text_align::left);
    for (int c = 2; c <= 6; ++c)
        table.column(c).set_cell_text_align(fort::text_align::right);

    // Layer counts for Qwen 2.5 family
    const int layer_counts[] = {24, 28, 36, 28, 40, 64};

    for (size_t i = 0; i < std::size(ALL_MODELS); ++i)
    {
        const auto &model = ALL_MODELS[i];
        AttnBenchConfig cfg{model, 1, kv_len, 1, true, warmup, iters};
        auto r = run_attention_bench(cfg);

        std::ostringstream config_str;
        config_str << model.n_heads << "h/" << model.n_kv_heads << "kv d=" << model.head_dim;

        double full_ms = r.mean_us * layer_counts[i] / 1000.0;

        table << model.name << config_str.str() << fmt_us(r.mean_us)
              << fmt_1(r.bw_gbs) << fmt_pct(r.roofline_pct)
              << std::to_string(layer_counts[i]) << fmt_1(full_ms) << fort::endr;
    }

    std::cout << "\nDECODE SUMMARY: All Models @ kv_len=" << kv_len << " (cold-cache)\n"
              << table.to_string() << std::endl;
}
