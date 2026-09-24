/**
 * @file Perf__CpuFlashAttentionKernelT.cpp
 * @brief Performance benchmark for CPUFlashAttentionKernelT (FP32 flash attention)
 *
 * Benchmarks the CPU flash attention kernel across multiple model configurations
 * and decode/prefill workloads. Measures:
 *   - Latency (ms) with min/mean/max/stddev
 *   - GFLOP/s throughput
 *   - Per-phase breakdown (QK vs V) via LLAMINAR_PROFILE_KERNELS
 *   - KV-length scaling curves for decode
 *   - Accuracy validation against a reference (online softmax correctness)
 *
 * Model configurations:
 *   - Qwen 2.5 0.5B  (14 heads, 2 KV heads, head_dim=64)
 *   - Qwen 2.5 1.5B  (12 heads, 2 KV heads, head_dim=128)
 *   - Qwen 2.5 3B    (16 heads, 2 KV heads, head_dim=128)
 *   - Qwen 2.5 7B    (28 heads, 4 KV heads, head_dim=128)
 *
 * Tile override: Set LLAMINAR_ATTN_FLASH_KV_TILE_DECODE or
 *                LLAMINAR_ATTN_FLASH_KV_TILE_PREFILL to sweep tile sizes.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "fort.hpp"
#include "kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "utils/KernelProfiler.h"
#include "utils/Logger.h"

using namespace llaminar2;

// ============================================================================
// Configuration
// ============================================================================

struct ModelConfig
{
    int n_heads;
    int n_kv_heads;
    int head_dim;
    const char *name;
};

static constexpr ModelConfig kQwen05B = {14, 2, 64, "Qwen2.5-0.5B"};
static constexpr ModelConfig kQwen15B = {12, 2, 128, "Qwen2.5-1.5B"};
static constexpr ModelConfig kQwen3B = {16, 2, 128, "Qwen2.5-3B"};
static constexpr ModelConfig kQwen7B = {28, 4, 128, "Qwen2.5-7B"};

struct BenchConfig
{
    ModelConfig model;
    int seq_len;
    int kv_len;
    bool causal;
    int warmup;
    int iters;
    const char *label;
};

struct BenchResult
{
    double mean_us;
    double stddev_us;
    double min_us;
    double max_us;
    double gflops;
};

// ============================================================================
// FLOPs calculation
// ============================================================================

static double attention_flops(int seq_len, int kv_len, int n_heads, int head_dim, bool causal)
{
    // Q·K^T: 2 * seq_len * kv_len * head_dim * n_heads (mul-add per element)
    // softmax: ~5 ops per position (exp, sub, add, div, max)
    // attn*V:  2 * seq_len * kv_len * head_dim * n_heads
    // For causal prefill, effective kv_len per query averages (kv_len+1)/2
    double effective_kv = causal ? (static_cast<double>(kv_len) + 1.0) / 2.0 : kv_len;
    double qk = 2.0 * seq_len * effective_kv * head_dim * n_heads;
    double sm = 5.0 * seq_len * effective_kv * n_heads;
    double av = 2.0 * seq_len * effective_kv * head_dim * n_heads;
    return qk + sm + av;
}

// ============================================================================
// Benchmark helpers
// ============================================================================

static void fill_random(float *data, size_t n, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (size_t i = 0; i < n; ++i)
    {
        data[i] = dist(gen);
    }
}

static BenchResult run_bench(const BenchConfig &cfg)
{
    const int q_stride = cfg.model.n_heads * cfg.model.head_dim;
    const int kv_stride = cfg.model.n_kv_heads * cfg.model.head_dim;
    const size_t q_size = static_cast<size_t>(cfg.seq_len) * q_stride;
    const size_t kv_size = static_cast<size_t>(cfg.kv_len) * kv_stride;
    const size_t out_size = static_cast<size_t>(cfg.seq_len) * q_stride;

    std::vector<float> Q(q_size), K(kv_size), V(kv_size), output(out_size);
    fill_random(Q.data(), q_size, 42);
    fill_random(K.data(), kv_size, 43);
    fill_random(V.data(), kv_size, 44);

    // Position offset for decode: simulate generating after a long prefill
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
                cfg.model.n_heads, cfg.model.n_kv_heads, cfg.model.head_dim,
                cfg.causal, position_offset);
        }
        else
        {
            kernel.compute(
                Q.data(), K.data(), V.data(), output.data(),
                cfg.seq_len, cfg.model.n_heads, cfg.model.n_kv_heads, cfg.model.head_dim,
                cfg.causal);
        }
    };

    // Warmup
    for (int i = 0; i < cfg.warmup; ++i)
    {
        run_once();
    }

    // Timed iterations
    std::vector<double> times_us;
    times_us.reserve(cfg.iters);

    for (int i = 0; i < cfg.iters; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        run_once();
        auto end = std::chrono::high_resolution_clock::now();
        times_us.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    double sum = std::accumulate(times_us.begin(), times_us.end(), 0.0);
    double mean = sum / times_us.size();
    double sq_sum = 0.0;
    for (auto t : times_us)
    {
        sq_sum += (t - mean) * (t - mean);
    }
    double stddev = std::sqrt(sq_sum / times_us.size());
    double mn = *std::min_element(times_us.begin(), times_us.end());
    double mx = *std::max_element(times_us.begin(), times_us.end());

    double flops = attention_flops(cfg.seq_len, cfg.kv_len, cfg.model.n_heads, cfg.model.head_dim, cfg.causal);
    double gflops = flops / (mean * 1e3); // GFLOP/s = FLOPs / (μs * 1e3)

    return {mean, stddev, mn, mx, gflops};
}

// ============================================================================
// Reference softmax for accuracy checking
// ============================================================================

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

            // Compute scores
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
                {
                    dot += q_ptr[d] * k_ptr[d];
                }
                scores[k] = dot * scale;
                max_s = std::max(max_s, scores[k]);
            }

            // Softmax + V accumulation
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
                {
                    out[d] += p * v_ptr[d];
                }
            }
            for (int d = 0; d < head_dim; ++d)
            {
                out[d] /= sum_exp;
            }
        }
    }
}

static float compute_max_abs_error(const float *a, const float *b, size_t n)
{
    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    }
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

// ============================================================================
// Test fixture
// ============================================================================

class Perf__CpuFlashAttentionKernelT : public ::testing::Test
{
protected:
    void render_results_table(const std::string &title,
                              const std::vector<std::pair<BenchConfig, BenchResult>> &results)
    {
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Config" << "Mean (μs)" << "Min (μs)" << "Max (μs)"
              << "Stddev" << "GFLOP/s" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 5; ++c)
        {
            table.column(c).set_cell_text_align(fort::text_align::right);
        }

        for (const auto &[cfg, res] : results)
        {
            std::ostringstream desc;
            desc << cfg.model.name << " "
                 << (cfg.seq_len == 1 ? "decode" : "prefill")
                 << " sl=" << cfg.seq_len << " kv=" << cfg.kv_len;

            auto fmt = [](double v)
            {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(1) << v;
                return oss.str();
            };
            auto fmtG = [](double v)
            {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2) << v;
                return oss.str();
            };

            table << desc.str() << fmt(res.mean_us) << fmt(res.min_us)
                  << fmt(res.max_us) << fmt(res.stddev_us) << fmtG(res.gflops) << fort::endr;
        }

        std::cout << "\n"
                  << title << "\n"
                  << table.to_string() << std::endl;
    }
};

// ============================================================================
// Decode benchmarks: KV-length scaling across model sizes
// ============================================================================

TEST_F(Perf__CpuFlashAttentionKernelT, Decode_KVLengthScaling)
{
    const ModelConfig models[] = {kQwen05B, kQwen15B, kQwen3B, kQwen7B};
    const int kv_lengths[] = {64, 128, 256, 512, 1024, 2048};
    const int warmup = 200;
    const int iters = 500;

    std::vector<std::pair<BenchConfig, BenchResult>> results;
    for (const auto &model : models)
    {
        for (int kv : kv_lengths)
        {
            BenchConfig cfg{model, 1, kv, true, warmup, iters, "decode"};
            results.emplace_back(cfg, run_bench(cfg));
        }
    }
    render_results_table("DECODE: KV-Length Scaling (seq_len=1, causal)", results);
}

// ============================================================================
// Prefill benchmarks: seq_len scaling across model sizes
// ============================================================================

TEST_F(Perf__CpuFlashAttentionKernelT, Prefill_SeqLenScaling)
{
    const int warmup = 3;
    const int iters = 10;

    // Small models get longer sequences, large models get shorter
    struct PrefillSpec
    {
        ModelConfig model;
        int seq_len;
    };
    const PrefillSpec specs[] = {
        {kQwen05B, 32},
        {kQwen05B, 128},
        {kQwen05B, 256},
        {kQwen05B, 512},
        {kQwen15B, 32},
        {kQwen15B, 128},
        {kQwen15B, 256},
        {kQwen3B, 32},
        {kQwen3B, 128},
        {kQwen3B, 256},
        {kQwen7B, 32},
        {kQwen7B, 128},
    };

    std::vector<std::pair<BenchConfig, BenchResult>> results;
    for (const auto &sp : specs)
    {
        BenchConfig cfg{sp.model, sp.seq_len, sp.seq_len, true, warmup, iters, "prefill"};
        results.emplace_back(cfg, run_bench(cfg));
    }
    render_results_table("PREFILL: Seq-Length Scaling (causal)", results);
}

// ============================================================================
// Accuracy validation: flash attention vs scalar reference
// ============================================================================

TEST_F(Perf__CpuFlashAttentionKernelT, Accuracy_DecodeAndPrefill)
{
    struct AccuracyConfig
    {
        ModelConfig model;
        int seq_len;
        int kv_len;
        const char *label;
    };

    const AccuracyConfig configs[] = {
        {kQwen05B, 1, 64, "0.5B decode kv=64"},
        {kQwen05B, 1, 256, "0.5B decode kv=256"},
        {kQwen05B, 1, 1024, "0.5B decode kv=1024"},
        {kQwen05B, 32, 32, "0.5B prefill sl=32"},
        {kQwen05B, 128, 128, "0.5B prefill sl=128"},
        {kQwen15B, 1, 128, "1.5B decode kv=128"},
        {kQwen15B, 1, 512, "1.5B decode kv=512"},
        {kQwen15B, 64, 64, "1.5B prefill sl=64"},
        {kQwen3B, 1, 256, "3B decode kv=256"},
        {kQwen3B, 1, 1024, "3B decode kv=1024"},
        {kQwen7B, 1, 512, "7B decode kv=512"},
        {kQwen7B, 32, 32, "7B prefill sl=32"},
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
        const int q_stride = c.model.n_heads * c.model.head_dim;
        const int kv_stride = c.model.n_kv_heads * c.model.head_dim;
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
                c.model.n_heads, c.model.n_kv_heads, c.model.head_dim,
                true, position_offset);
        }
        else
        {
            kernel.compute(
                Q.data(), K.data(), V.data(), flash_out.data(),
                c.seq_len, c.model.n_heads, c.model.n_kv_heads, c.model.head_dim,
                true);
        }

        reference_attention(Q.data(), K.data(), V.data(), ref_out.data(),
                            c.seq_len, c.kv_len,
                            c.model.n_heads, c.model.n_kv_heads, c.model.head_dim,
                            true, position_offset);

        float max_err = compute_max_abs_error(flash_out.data(), ref_out.data(), out_size);
        float cos_sim = compute_cosine_sim(flash_out.data(), ref_out.data(), out_size);

        // Tolerances: max absolute error < 1e-4, cosine similarity > 0.99999
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
// Decode latency vs head_dim (comparing model architectures)
// ============================================================================

TEST_F(Perf__CpuFlashAttentionKernelT, Decode_ModelComparison)
{
    const int kv_len = 512;
    const int warmup = 200;
    const int iters = 500;

    const ModelConfig models[] = {kQwen05B, kQwen15B, kQwen3B, kQwen7B};
    std::vector<std::pair<BenchConfig, BenchResult>> results;

    for (const auto &model : models)
    {
        BenchConfig cfg{model, 1, kv_len, true, warmup, iters, "decode"};
        results.emplace_back(cfg, run_bench(cfg));
    }
    render_results_table("DECODE: Model Comparison (kv_len=512, causal)", results);
}

// ============================================================================
// Prefill throughput comparison across models
// ============================================================================

TEST_F(Perf__CpuFlashAttentionKernelT, Prefill_ModelComparison)
{
    const int seq_len = 256;
    const int warmup = 5;
    const int iters = 20;

    const ModelConfig models[] = {kQwen05B, kQwen15B, kQwen3B, kQwen7B};
    std::vector<std::pair<BenchConfig, BenchResult>> results;

    for (const auto &model : models)
    {
        BenchConfig cfg{model, seq_len, seq_len, true, warmup, iters, "prefill"};
        results.emplace_back(cfg, run_bench(cfg));
    }
    render_results_table("PREFILL: Model Comparison (seq_len=256, causal)", results);
}
