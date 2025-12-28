/**
 * @file Perf__FusedAttentionWo.cpp
 * @brief Performance benchmark comparing fused vs unfused attention paths
 *
 * This test benchmarks the fused attention + Wo kernel performance against
 * the separate attention + Wo GEMM path.
 *
 * Measurements:
 *   - Fused JIT kernel (attention + Wo in single kernel)
 *   - Unfused path (attention kernel + separate Wo GEMM)
 *   - Speedup ratio
 *
 * Configurations tested:
 *   - Single token decode (seq_len=1, variable kv_len)
 *   - Prefill (seq_len > 1, seq_len == kv_len)
 *   - Various model sizes (Qwen 0.5B, 1.5B, 7B dimensions)
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>

// V2 includes
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include "kernels/cpu/attention/q8_1/FusedAttentionWoKernel.h"
#include "kernels/cpu/attention/q8_1/FusedAttentionWoRef.h"
#include "kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.h"
#include "execution/RuntimeConfig.h"
#include "utils/Logger.h"

using namespace llaminar::v2::kernels;
using namespace llaminar::v2::kernels::jit;
using namespace llaminar::v2::kernels::microkernels;
using namespace llaminar2;

// Explicitly alias to resolve ambiguity between llaminar2::FusedAttentionWoParams and llaminar::v2::kernels::FusedAttentionWoParams
// This alias shadows both namespace imports when unqualified FusedAttentionWoParams is used
namespace
{
    using FusedAttentionWoParamsKernel = llaminar::v2::kernels::FusedAttentionWoParams;
}

// ============================================================================
// Benchmark Configuration
// ============================================================================

struct FusedAttentionBenchConfig
{
    int seq_len_q;    // Query sequence length (1 for decode, >1 for prefill)
    int seq_len_kv;   // KV cache length
    int num_heads;    // Query heads
    int num_kv_heads; // KV heads (GQA)
    int head_dim;     // Dimension per head
    int d_model;      // Model dimension (num_heads * head_dim)
    int warmup_iters;
    int bench_iters;
    std::string description;
};

struct BenchmarkResult
{
    double fused_jit_ms;
    double fused_jit_fa2_ms; // FA2 4x tiled path
    double fused_ref_ms;
    double unfused_ms;
    double speedup_vs_ref;
    double speedup_vs_unfused;
    double fa2_speedup_vs_fa1; // FA2 vs FA1 speedup
    double flops;              // Total FLOPs for the operation
    double fa1_gflops;         // FA1 GFLOP/s
    double fa2_gflops;         // FA2 GFLOP/s
};

/**
 * @brief Calculate FLOPs for fused attention + Wo projection
 *
 * Attention FLOPs:
 *   - Q·K^T scores: 2 * seq_q * seq_kv * head_dim * num_heads (mul-add)
 *   - Softmax: ~5 * seq_q * seq_kv * num_heads (exp, sub, div, add)
 *   - Attn * V: 2 * seq_q * seq_kv * head_dim * num_heads
 *
 * Wo projection FLOPs:
 *   - Context * Wo: 2 * seq_q * (num_heads * head_dim) * d_model
 *
 * Note: For GQA, Q·K and Attn*V use num_heads but K/V are shared across groups
 */
inline double calculate_attention_flops(int seq_q, int seq_kv, int num_heads, int num_kv_heads, int head_dim, int d_model)
{
    // For causal attention, average KV positions per query = (seq_kv + 1) / 2 for prefill
    // For decode (seq_q=1), all seq_kv positions are used
    double effective_kv = (seq_q == 1) ? seq_kv : (seq_kv + 1.0) / 2.0;

    // Q·K^T: Each head computes seq_q * effective_kv dot products of head_dim
    double qk_flops = 2.0 * seq_q * effective_kv * head_dim * num_heads;

    // Softmax: ~5 ops per score (max, sub, exp, sum, div)
    double softmax_flops = 5.0 * seq_q * effective_kv * num_heads;

    // Attention * V: Each head computes seq_q outputs, each is weighted sum of effective_kv V vectors
    double av_flops = 2.0 * seq_q * effective_kv * head_dim * num_heads;

    // Wo projection: seq_q * (num_heads * head_dim) × d_model
    double wo_flops = 2.0 * seq_q * num_heads * head_dim * d_model;

    return qk_flops + softmax_flops + av_flops + wo_flops;
}

// ============================================================================
// Test Fixture
// ============================================================================

class Perf__FusedAttentionWo : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

    void SetUp() override
    {
        // Seed RNG for reproducibility
        rng_.seed(42);
    }

    // Generate random Q8_1 blocks (matches Q8_1Block with uint16_t d (FP16) and int16_t sum_qs)
    void generate_q8_1_blocks(std::vector<Q8_1Block> &blocks, size_t num_elements)
    {
        size_t num_blocks = (num_elements + 31) / 32;
        blocks.resize(num_blocks);

        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> fp32_data(num_elements);
        for (auto &v : fp32_data)
            v = dist(rng_);

        // Quantize to Q8_1 format
        for (size_t b = 0; b < num_blocks; ++b)
        {
            float max_abs = 0.0f;
            size_t start = b * 32;
            size_t end = std::min(start + 32, num_elements);

            for (size_t i = start; i < end; ++i)
                max_abs = std::max(max_abs, std::abs(fp32_data[i]));

            float scale = max_abs / 127.0f;
            if (scale < 1e-10f)
                scale = 1e-10f;
            float inv_scale = 127.0f / max_abs;
            if (max_abs < 1e-10f)
                inv_scale = 0.0f;

            int32_t sum_qs = 0;
            for (size_t i = 0; i < 32; ++i)
            {
                if (start + i < end)
                {
                    float val = fp32_data[start + i];
                    int8_t q = static_cast<int8_t>(std::round(val * inv_scale));
                    q = std::max(int8_t(-127), std::min(int8_t(127), q));
                    blocks[b].qs[i] = q;
                    sum_qs += q;
                }
                else
                {
                    blocks[b].qs[i] = 0;
                }
            }

            // Convert scale to FP16 and store sum
            blocks[b].d = fp32_to_fp16(scale);
            blocks[b].sum_qs = static_cast<int16_t>(sum_qs);
        }
    }

    BenchmarkResult run_benchmark(const FusedAttentionBenchConfig &config)
    {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "Benchmark: " << config.description << std::endl;
        std::cout << "  seq_len_q=" << config.seq_len_q
                  << ", seq_len_kv=" << config.seq_len_kv
                  << ", heads=" << config.num_heads << "/" << config.num_kv_heads
                  << ", head_dim=" << config.head_dim
                  << ", d_model=" << config.d_model << std::endl;
        std::cout << "================================================================" << std::endl;

        const float scale = 1.0f / std::sqrt(static_cast<float>(config.head_dim));
        const int blocks_per_head = config.head_dim / 32;

        // Allocate Q8_1 tensors
        size_t Q_elements = config.seq_len_q * config.num_heads * config.head_dim;
        size_t K_elements = config.seq_len_kv * config.num_kv_heads * config.head_dim;
        size_t V_elements = config.seq_len_kv * config.num_kv_heads * config.head_dim;

        std::vector<Q8_1Block> Q_blocks, K_blocks, V_blocks;
        generate_q8_1_blocks(Q_blocks, Q_elements);
        generate_q8_1_blocks(K_blocks, K_elements);
        generate_q8_1_blocks(V_blocks, V_elements);

        // Allocate Wo weights (FP32 for simplicity)
        std::vector<float> Wo_fp32(config.d_model * config.num_heads * config.head_dim);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (auto &v : Wo_fp32)
            v = dist(rng_);

        // Output buffer
        std::vector<float> output(config.seq_len_q * config.d_model);

        BenchmarkResult result{};

        // ====================================================================
        // Benchmark 1: Fused JIT Kernel
        // ====================================================================
        {
            JitAttentionConfig jit_config;
            jit_config.head_dim = config.head_dim;
            jit_config.num_heads = config.num_heads;
            jit_config.num_kv_heads = config.num_kv_heads;
            // Use seq_len_q as batch_size to trigger prefill mode for multi-token
            jit_config.batch_size = config.seq_len_q;
            jit_config.wo_format = WoFormat::FP32;
            jit_config.causal = true;

            JitFusedAttentionWo jit_kernel(jit_config);

            // Warmup
            for (int i = 0; i < config.warmup_iters; ++i)
            {
                jit_kernel.compute(Q_blocks.data(), K_blocks.data(), V_blocks.data(),
                                   Wo_fp32.data(), output.data(),
                                   config.seq_len_q, config.seq_len_kv, scale,
                                   config.seq_len_kv - config.seq_len_q);
            }

            // Benchmark
            std::vector<double> times;
            times.reserve(config.bench_iters);

            for (int i = 0; i < config.bench_iters; ++i)
            {
                auto start = std::chrono::high_resolution_clock::now();
                jit_kernel.compute(Q_blocks.data(), K_blocks.data(), V_blocks.data(),
                                   Wo_fp32.data(), output.data(),
                                   config.seq_len_q, config.seq_len_kv, scale,
                                   config.seq_len_kv - config.seq_len_q);
                auto end = std::chrono::high_resolution_clock::now();
                times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            }

            std::sort(times.begin(), times.end());
            result.fused_jit_ms = times[times.size() / 2]; // Median
            std::cout << "  Fused JIT (FA1):" << std::fixed << std::setprecision(3)
                      << result.fused_jit_ms << " ms (median)" << std::endl;
        }

        // ====================================================================
        // Benchmark 1b: Fused JIT Kernel with FA2 Tiling (4x batched)
        // ====================================================================
        {
            JitAttentionConfig jit_config;
            jit_config.head_dim = config.head_dim;
            jit_config.num_heads = config.num_heads;
            jit_config.num_kv_heads = config.num_kv_heads;
            jit_config.batch_size = config.seq_len_q;
            jit_config.wo_format = WoFormat::FP32;
            jit_config.causal = true;
            jit_config.use_fa2_tiling = true; // Enable FA2 4x tiling

            JitFusedAttentionWo jit_kernel_fa2(jit_config);

            // Warmup
            for (int i = 0; i < config.warmup_iters; ++i)
            {
                jit_kernel_fa2.compute(Q_blocks.data(), K_blocks.data(), V_blocks.data(),
                                       Wo_fp32.data(), output.data(),
                                       config.seq_len_q, config.seq_len_kv, scale,
                                       config.seq_len_kv - config.seq_len_q);
            }

            // Benchmark
            std::vector<double> times;
            times.reserve(config.bench_iters);

            for (int i = 0; i < config.bench_iters; ++i)
            {
                auto start = std::chrono::high_resolution_clock::now();
                jit_kernel_fa2.compute(Q_blocks.data(), K_blocks.data(), V_blocks.data(),
                                       Wo_fp32.data(), output.data(),
                                       config.seq_len_q, config.seq_len_kv, scale,
                                       config.seq_len_kv - config.seq_len_q);
                auto end = std::chrono::high_resolution_clock::now();
                times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            }

            std::sort(times.begin(), times.end());
            result.fused_jit_fa2_ms = times[times.size() / 2]; // Median
            std::cout << "  Fused JIT (FA2):" << std::fixed << std::setprecision(3)
                      << result.fused_jit_fa2_ms << " ms (median)" << std::endl;
        }

        // ====================================================================
        // Benchmark 2: Fused Reference Kernel
        // ====================================================================
        {
            FusedAttentionWoParamsKernel params;
            params.Q = Q_blocks.data();
            params.K = K_blocks.data();
            params.V = V_blocks.data();
            params.Wo = Wo_fp32.data();
            params.wo_type = WoWeightType::FP32;
            params.output = output.data();
            params.batch_size = 1;
            params.seq_len = config.seq_len_q;
            params.kv_seq_len = config.seq_len_kv;
            params.num_heads = config.num_heads;
            params.num_kv_heads = config.num_kv_heads;
            params.head_dim = config.head_dim;
            params.d_model = config.d_model;
            params.scale = scale;
            params.causal = true;
            params.position_offset = config.seq_len_kv - config.seq_len_q;

            // Warmup
            for (int i = 0; i < config.warmup_iters; ++i)
            {
                FusedAttentionWoRef::execute(params);
            }

            // Benchmark
            std::vector<double> times;
            times.reserve(config.bench_iters);

            for (int i = 0; i < config.bench_iters; ++i)
            {
                auto start = std::chrono::high_resolution_clock::now();
                FusedAttentionWoRef::execute(params);
                auto end = std::chrono::high_resolution_clock::now();
                times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            }

            std::sort(times.begin(), times.end());
            result.fused_ref_ms = times[times.size() / 2]; // Median
            std::cout << "  Fused Reference:" << std::fixed << std::setprecision(3)
                      << result.fused_ref_ms << " ms (median)" << std::endl;
        }

        // Compute speedups and FLOPS
        result.speedup_vs_ref = result.fused_ref_ms / result.fused_jit_ms;
        result.fa2_speedup_vs_fa1 = result.fused_jit_ms / result.fused_jit_fa2_ms;
        result.unfused_ms = 0; // Not implemented yet
        result.speedup_vs_unfused = 0;

        // Calculate FLOPS
        result.flops = calculate_attention_flops(config.seq_len_q, config.seq_len_kv,
                                                 config.num_heads, config.num_kv_heads,
                                                 config.head_dim, config.d_model);
        result.fa1_gflops = (result.flops / 1e9) / (result.fused_jit_ms / 1000.0);
        result.fa2_gflops = (result.flops / 1e9) / (result.fused_jit_fa2_ms / 1000.0);

        std::cout << "\n  JIT (FA1) Speedup vs Reference: " << std::fixed << std::setprecision(2)
                  << result.speedup_vs_ref << "x" << std::endl;
        std::cout << "  JIT (FA2) Speedup vs FA1:       " << std::fixed << std::setprecision(2)
                  << result.fa2_speedup_vs_fa1 << "x" << std::endl;
        std::cout << "  JIT (FA2) Speedup vs Reference: " << std::fixed << std::setprecision(2)
                  << (result.fused_ref_ms / result.fused_jit_fa2_ms) << "x" << std::endl;
        std::cout << "\n  Total FLOPs:    " << std::scientific << std::setprecision(2)
                  << result.flops << std::endl;
        std::cout << "  FA1 Throughput: " << std::fixed << std::setprecision(1)
                  << result.fa1_gflops << " GFLOP/s" << std::endl;
        std::cout << "  FA2 Throughput: " << std::fixed << std::setprecision(1)
                  << result.fa2_gflops << " GFLOP/s" << std::endl;

        return result;
    }
};

// ============================================================================
// Benchmark Tests
// ============================================================================

TEST_F(Perf__FusedAttentionWo, Qwen05B_SingleToken_Decode)
{
    // Qwen 0.5B: 14 heads, 2 KV heads, head_dim=64, d_model=896
    FusedAttentionBenchConfig config;
    config.seq_len_q = 1;
    config.seq_len_kv = 128;
    config.num_heads = 14;
    config.num_kv_heads = 2;
    config.head_dim = 64;
    config.d_model = 896;
    config.warmup_iters = 10;
    config.bench_iters = 100;
    config.description = "Qwen 0.5B - Single Token Decode (kv=128)";

    auto result = run_benchmark(config);
    EXPECT_GT(result.speedup_vs_ref, 1.0) << "JIT should be faster than Reference";
}

TEST_F(Perf__FusedAttentionWo, Qwen05B_SingleToken_LongContext)
{
    FusedAttentionBenchConfig config;
    config.seq_len_q = 1;
    config.seq_len_kv = 512;
    config.num_heads = 14;
    config.num_kv_heads = 2;
    config.head_dim = 64;
    config.d_model = 896;
    config.warmup_iters = 5;
    config.bench_iters = 50;
    config.description = "Qwen 0.5B - Single Token Decode (kv=512)";

    auto result = run_benchmark(config);
    EXPECT_GT(result.speedup_vs_ref, 1.0) << "JIT should be faster than Reference";
}

TEST_F(Perf__FusedAttentionWo, Qwen05B_Prefill_Short)
{
    FusedAttentionBenchConfig config;
    config.seq_len_q = 32;
    config.seq_len_kv = 32;
    config.num_heads = 14;
    config.num_kv_heads = 2;
    config.head_dim = 64;
    config.d_model = 896;
    config.warmup_iters = 5;
    config.bench_iters = 50;
    config.description = "Qwen 0.5B - Prefill (seq=32)";

    auto result = run_benchmark(config);
    EXPECT_GT(result.speedup_vs_ref, 1.0) << "JIT should be faster than Reference";
}

TEST_F(Perf__FusedAttentionWo, Qwen05B_Prefill_Medium)
{
    FusedAttentionBenchConfig config;
    config.seq_len_q = 128;
    config.seq_len_kv = 128;
    config.num_heads = 14;
    config.num_kv_heads = 2;
    config.head_dim = 64;
    config.d_model = 896;
    config.warmup_iters = 3;
    config.bench_iters = 20;
    config.description = "Qwen 0.5B - Prefill (seq=128)";

    auto result = run_benchmark(config);
    EXPECT_GT(result.speedup_vs_ref, 1.0) << "JIT should be faster than Reference";
}

TEST_F(Perf__FusedAttentionWo, Qwen15B_SingleToken_Decode)
{
    // Qwen 1.5B: 16 heads, 2 KV heads, head_dim=96, d_model=1536
    FusedAttentionBenchConfig config;
    config.seq_len_q = 1;
    config.seq_len_kv = 128;
    config.num_heads = 16;
    config.num_kv_heads = 2;
    config.head_dim = 96;
    config.d_model = 1536;
    config.warmup_iters = 10;
    config.bench_iters = 100;
    config.description = "Qwen 1.5B - Single Token Decode (kv=128)";

    auto result = run_benchmark(config);
    // Note: head_dim=96 may not be JIT-optimized, check for graceful fallback
    std::cout << "  Note: head_dim=96 may use fallback path" << std::endl;
}

TEST_F(Perf__FusedAttentionWo, Qwen7B_SingleToken_Decode)
{
    // Qwen 7B: 28 heads, 4 KV heads, head_dim=128, d_model=3584
    FusedAttentionBenchConfig config;
    config.seq_len_q = 1;
    config.seq_len_kv = 128;
    config.num_heads = 28;
    config.num_kv_heads = 4;
    config.head_dim = 128;
    config.d_model = 3584;
    config.warmup_iters = 5;
    config.bench_iters = 50;
    config.description = "Qwen 7B - Single Token Decode (kv=128)";

    auto result = run_benchmark(config);
    EXPECT_GT(result.speedup_vs_ref, 1.0) << "JIT should be faster than Reference";
}

TEST_F(Perf__FusedAttentionWo, Qwen7B_Prefill_Medium)
{
    FusedAttentionBenchConfig config;
    config.seq_len_q = 128;
    config.seq_len_kv = 128;
    config.num_heads = 28;
    config.num_kv_heads = 4;
    config.head_dim = 128;
    config.d_model = 3584;
    config.warmup_iters = 3;
    config.bench_iters = 20;
    config.description = "Qwen 7B - Prefill (seq=128)";

    auto result = run_benchmark(config);
    EXPECT_GT(result.speedup_vs_ref, 1.0) << "JIT should be faster than Reference";
}

// ============================================================================
// Summary Test
// ============================================================================

TEST_F(Perf__FusedAttentionWo, Summary)
{
    std::cout << "\n\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                                    FUSED ATTENTION BENCHMARK SUMMARY                                                 ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Config             │ FA1 (ms)  │ FA2 (ms)  │ FA2/FA1  │ FA1 GFLOP/s │ FA2 GFLOP/s │ FLOPs                            ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣" << std::endl;

    // Run quick benchmarks for summary
    std::vector<std::pair<std::string, FusedAttentionBenchConfig>> configs = {
        {"Qwen0.5B decode", {1, 128, 14, 2, 64, 896, 5, 20, ""}},
        {"Qwen0.5B prefill", {32, 32, 14, 2, 64, 896, 5, 20, ""}},
        {"Qwen7B decode", {1, 128, 28, 4, 128, 3584, 3, 10, ""}},
        {"Qwen7B prefill", {128, 128, 28, 4, 128, 3584, 2, 5, ""}},
    };

    for (const auto &[name, config] : configs)
    {
        const float scale = 1.0f / std::sqrt(static_cast<float>(config.head_dim));

        std::vector<Q8_1Block> Q, K, V;
        generate_q8_1_blocks(Q, config.seq_len_q * config.num_heads * config.head_dim);
        generate_q8_1_blocks(K, config.seq_len_kv * config.num_kv_heads * config.head_dim);
        generate_q8_1_blocks(V, config.seq_len_kv * config.num_kv_heads * config.head_dim);

        std::vector<float> Wo(config.d_model * config.num_heads * config.head_dim, 0.01f);
        std::vector<float> output(config.seq_len_q * config.d_model);

        // ====== FA1 JIT ======
        JitAttentionConfig jit_config_fa1;
        jit_config_fa1.head_dim = config.head_dim;
        jit_config_fa1.num_heads = config.num_heads;
        jit_config_fa1.num_kv_heads = config.num_kv_heads;
        jit_config_fa1.batch_size = config.seq_len_q;
        jit_config_fa1.wo_format = WoFormat::FP32;
        jit_config_fa1.causal = true;
        jit_config_fa1.use_fa2_tiling = false;

        JitFusedAttentionWo jit_kernel_fa1(jit_config_fa1);

        // Warmup FA1
        for (int i = 0; i < config.warmup_iters; ++i)
            jit_kernel_fa1.compute(Q.data(), K.data(), V.data(), Wo.data(), output.data(),
                                   config.seq_len_q, config.seq_len_kv, scale, 0);

        // FA1 time
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < config.bench_iters; ++i)
            jit_kernel_fa1.compute(Q.data(), K.data(), V.data(), Wo.data(), output.data(),
                                   config.seq_len_q, config.seq_len_kv, scale, 0);
        auto t1 = std::chrono::high_resolution_clock::now();
        double fa1_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / config.bench_iters;

        // ====== FA2 JIT ======
        JitAttentionConfig jit_config_fa2;
        jit_config_fa2.head_dim = config.head_dim;
        jit_config_fa2.num_heads = config.num_heads;
        jit_config_fa2.num_kv_heads = config.num_kv_heads;
        jit_config_fa2.batch_size = config.seq_len_q;
        jit_config_fa2.wo_format = WoFormat::FP32;
        jit_config_fa2.causal = true;
        jit_config_fa2.use_fa2_tiling = true;

        JitFusedAttentionWo jit_kernel_fa2(jit_config_fa2);

        // Warmup FA2
        for (int i = 0; i < config.warmup_iters; ++i)
            jit_kernel_fa2.compute(Q.data(), K.data(), V.data(), Wo.data(), output.data(),
                                   config.seq_len_q, config.seq_len_kv, scale, 0);

        // FA2 time
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < config.bench_iters; ++i)
            jit_kernel_fa2.compute(Q.data(), K.data(), V.data(), Wo.data(), output.data(),
                                   config.seq_len_q, config.seq_len_kv, scale, 0);
        auto t3 = std::chrono::high_resolution_clock::now();
        double fa2_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / config.bench_iters;

        // Calculate FLOPS
        double flops = calculate_attention_flops(config.seq_len_q, config.seq_len_kv,
                                                 config.num_heads, config.num_kv_heads,
                                                 config.head_dim, config.d_model);
        double fa1_gflops = (flops / 1e9) / (fa1_ms / 1000.0);
        double fa2_gflops = (flops / 1e9) / (fa2_ms / 1000.0);
        double fa2_vs_fa1 = fa1_ms / fa2_ms;

        std::cout << "║  " << std::left << std::setw(18) << name
                  << " │ " << std::right << std::setw(8) << std::fixed << std::setprecision(3) << fa1_ms
                  << " │ " << std::setw(8) << fa2_ms
                  << " │ " << std::setw(7) << std::setprecision(2) << fa2_vs_fa1 << "x"
                  << " │ " << std::setw(10) << std::setprecision(1) << fa1_gflops
                  << " │ " << std::setw(10) << fa2_gflops
                  << " │ " << std::setw(10) << std::scientific << std::setprecision(2) << flops
                  << "                   ║" << std::endl;
    }

    std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nLegend: FA1 = single-score path, FA2 = 4x batched tiling" << std::endl;
    std::cout << "FLOPs = Q·K + Softmax + Attn·V + Wo projection (causal mask applied)" << std::endl;
}

// ============================================================================
// FA2 vs FA1 Scaling Test - How does FA2 perform as KV length increases?
// ============================================================================

TEST_F(Perf__FusedAttentionWo, FA2_vs_FA1_KV_Scaling)
{
    std::cout << "\n\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                                    FA2 vs FA1 - KV SEQUENCE LENGTH SCALING                                                 ║" << std::endl;
    std::cout << "║                                           (Qwen 0.5B decode mode)                                                          ║" << std::endl;
    std::cout << "╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  KV Length │ FA1 (ms) │ FA2 (ms) │ FA2/FA1 │ FA1 GFLOP/s │ FA2 GFLOP/s │ Tiles │      FLOPs                                ║" << std::endl;
    std::cout << "╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣" << std::endl;

    // Qwen 0.5B config
    const int num_heads = 14;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = 896;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int warmup = 5;
    const int iters = 30;

    std::vector<int> kv_lengths = {16, 32, 64, 128, 256, 512, 1024};

    for (int kv_len : kv_lengths)
    {
        std::vector<Q8_1Block> Q, K, V;
        generate_q8_1_blocks(Q, 1 * num_heads * head_dim); // seq_len_q = 1
        generate_q8_1_blocks(K, kv_len * num_kv_heads * head_dim);
        generate_q8_1_blocks(V, kv_len * num_kv_heads * head_dim);

        std::vector<float> Wo(d_model * num_heads * head_dim, 0.01f);
        std::vector<float> output(1 * d_model);

        // FA1
        JitAttentionConfig cfg_fa1;
        cfg_fa1.head_dim = head_dim;
        cfg_fa1.num_heads = num_heads;
        cfg_fa1.num_kv_heads = num_kv_heads;
        cfg_fa1.batch_size = 1;
        cfg_fa1.wo_format = WoFormat::FP32;
        cfg_fa1.causal = true;
        cfg_fa1.use_fa2_tiling = false;
        JitFusedAttentionWo kernel_fa1(cfg_fa1);

        for (int i = 0; i < warmup; ++i)
            kernel_fa1.compute(Q.data(), K.data(), V.data(), Wo.data(), output.data(), 1, kv_len, scale, 0);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
            kernel_fa1.compute(Q.data(), K.data(), V.data(), Wo.data(), output.data(), 1, kv_len, scale, 0);
        auto t1 = std::chrono::high_resolution_clock::now();
        double fa1_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        // FA2
        JitAttentionConfig cfg_fa2;
        cfg_fa2.head_dim = head_dim;
        cfg_fa2.num_heads = num_heads;
        cfg_fa2.num_kv_heads = num_kv_heads;
        cfg_fa2.batch_size = 1;
        cfg_fa2.wo_format = WoFormat::FP32;
        cfg_fa2.causal = true;
        cfg_fa2.use_fa2_tiling = true;
        JitFusedAttentionWo kernel_fa2(cfg_fa2);

        for (int i = 0; i < warmup; ++i)
            kernel_fa2.compute(Q.data(), K.data(), V.data(), Wo.data(), output.data(), 1, kv_len, scale, 0);
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
            kernel_fa2.compute(Q.data(), K.data(), V.data(), Wo.data(), output.data(), 1, kv_len, scale, 0);
        auto t3 = std::chrono::high_resolution_clock::now();
        double fa2_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / iters;

        double speedup = fa1_ms / fa2_ms;
        int tile_count = kv_len / 4;

        // Calculate FLOPS
        double flops = calculate_attention_flops(1, kv_len, num_heads, num_kv_heads, head_dim, d_model);
        double fa1_gflops = (flops / 1e9) / (fa1_ms / 1000.0);
        double fa2_gflops = (flops / 1e9) / (fa2_ms / 1000.0);

        std::cout << "║  kv=" << std::left << std::setw(6) << kv_len
                  << " │ " << std::right << std::setw(8) << std::fixed << std::setprecision(3) << fa1_ms
                  << " │ " << std::setw(8) << fa2_ms
                  << " │ " << std::setw(6) << std::setprecision(2) << speedup << "x"
                  << " │ " << std::setw(11) << std::setprecision(1) << fa1_gflops
                  << " │ " << std::setw(11) << fa2_gflops
                  << " │ " << std::setw(5) << tile_count
                  << " │ " << std::setw(10) << std::scientific << std::setprecision(2) << flops
                  << "                     ║" << std::endl;
    }

    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝" << std::endl;
}

// ============================================================================
// FA2 vs FA1 Scaling Test for Qwen 7B (head_dim=128)
// ============================================================================

TEST_F(Perf__FusedAttentionWo, FA2_vs_FA1_KV_Scaling_Qwen7B)
{
    std::cout << "\n\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                                    FA2 vs FA1 - KV SEQUENCE LENGTH SCALING                                                 ║" << std::endl;
    std::cout << "║                                           (Qwen 7B decode mode)                                                            ║" << std::endl;
    std::cout << "╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  KV Length │ FA1 (ms) │ FA2 (ms) │ FA2/FA1 │ FA1 GFLOP/s │ FA2 GFLOP/s │ Tiles │      FLOPs                                ║" << std::endl;
    std::cout << "╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣" << std::endl;

    // Qwen 7B config
    const int num_heads = 28;
    const int num_kv_heads = 4;
    const int head_dim = 128;
    const int d_model = 3584;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int warmup = 3;
    const int iters = 15;

    std::vector<int> kv_lengths = {16, 32, 64, 128, 256, 512};

    for (int kv_len : kv_lengths)
    {
        std::vector<Q8_1Block> Q, K, V;
        generate_q8_1_blocks(Q, 1 * num_heads * head_dim); // seq_len_q = 1
        generate_q8_1_blocks(K, kv_len * num_kv_heads * head_dim);
        generate_q8_1_blocks(V, kv_len * num_kv_heads * head_dim);

        std::vector<float> Wo(d_model * num_heads * head_dim, 0.01f);
        std::vector<float> output(1 * d_model);

        // FA1
        JitAttentionConfig cfg_fa1;
        cfg_fa1.head_dim = head_dim;
        cfg_fa1.num_heads = num_heads;
        cfg_fa1.num_kv_heads = num_kv_heads;
        cfg_fa1.batch_size = 1;
        cfg_fa1.wo_format = WoFormat::FP32;
        cfg_fa1.causal = true;
        cfg_fa1.use_fa2_tiling = false;
        JitFusedAttentionWo kernel_fa1(cfg_fa1);

        for (int i = 0; i < warmup; ++i)
            kernel_fa1.compute(Q.data(), K.data(), V.data(), Wo.data(), output.data(), 1, kv_len, scale, 0);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
            kernel_fa1.compute(Q.data(), K.data(), V.data(), Wo.data(), output.data(), 1, kv_len, scale, 0);
        auto t1 = std::chrono::high_resolution_clock::now();
        double fa1_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        // FA2
        JitAttentionConfig cfg_fa2;
        cfg_fa2.head_dim = head_dim;
        cfg_fa2.num_heads = num_heads;
        cfg_fa2.num_kv_heads = num_kv_heads;
        cfg_fa2.batch_size = 1;
        cfg_fa2.wo_format = WoFormat::FP32;
        cfg_fa2.causal = true;
        cfg_fa2.use_fa2_tiling = true;
        JitFusedAttentionWo kernel_fa2(cfg_fa2);

        for (int i = 0; i < warmup; ++i)
            kernel_fa2.compute(Q.data(), K.data(), V.data(), Wo.data(), output.data(), 1, kv_len, scale, 0);
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
            kernel_fa2.compute(Q.data(), K.data(), V.data(), Wo.data(), output.data(), 1, kv_len, scale, 0);
        auto t3 = std::chrono::high_resolution_clock::now();
        double fa2_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / iters;

        double speedup = fa1_ms / fa2_ms;
        int tile_count = kv_len / 4;

        // Calculate FLOPS
        double flops = calculate_attention_flops(1, kv_len, num_heads, num_kv_heads, head_dim, d_model);
        double fa1_gflops = (flops / 1e9) / (fa1_ms / 1000.0);
        double fa2_gflops = (flops / 1e9) / (fa2_ms / 1000.0);

        std::cout << "║  kv=" << std::left << std::setw(6) << kv_len
                  << " │ " << std::right << std::setw(8) << std::fixed << std::setprecision(3) << fa1_ms
                  << " │ " << std::setw(8) << fa2_ms
                  << " │ " << std::setw(6) << std::setprecision(2) << speedup << "x"
                  << " │ " << std::setw(11) << std::setprecision(1) << fa1_gflops
                  << " │ " << std::setw(11) << fa2_gflops
                  << " │ " << std::setw(5) << tile_count
                  << " │ " << std::setw(10) << std::scientific << std::setprecision(2) << flops
                  << "                     ║" << std::endl;
    }

    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝" << std::endl;
}
