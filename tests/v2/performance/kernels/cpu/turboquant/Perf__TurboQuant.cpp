/**
 * @file Perf__TurboQuant.cpp
 * @brief Performance benchmark for TQ4 and TQ8 quantize / dequantize / KV cache
 *
 * Measures throughput for the TurboQuant hot paths:
 *   1. TQ4 Quantize:   FP32 → TQ4
 *   2. TQ4 Dequantize: TQ4  → FP32
 *   3. TQ4 Roundtrip:  FP32 → TQ4 → FP32
 *   4. TQ8 Quantize:   FP32 → TQ8
 *   5. TQ8 Dequantize: TQ8  → FP32
 *   6. TQ8 Roundtrip:  FP32 → TQ8 → FP32
 *   7. KV Cache append + gather (TQ8-K / TQ4-V split precision)
 *
 * Each operation is benchmarked with warmup, then timed over many iterations.
 * Reports: ops/sec, throughput (GB/s for data-movement-bound ops), and
 * per-vector latency in nanoseconds.
 *
 * `ProductionFusedKVProfilePoint` provides the profiler-facing entry point.
 * It repeats exactly one fully configured production operation after all
 * allocation and TurboQuant context construction, so Linux perf samples are
 * never polluted by a different quantizer, materializer, or setup operation.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/cpu/turboquant/TurboQuantCodebook.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/cpu/turboquant/TurboQuantDequantizeSplitTQ.h"
#include "kernels/cpu/turboquant/TurboQuantDequantizeTQ4.h"
#include "kernels/cpu/turboquant/TurboQuantDequantizeTQ8.h"
#include "kernels/cpu/turboquant/TurboQuantQuantizeKV.h"
#include "kernels/cpu/turboquant/TurboQuantQuantizeTQ4.h"
#include "kernels/cpu/turboquant/TurboQuantQuantizeTQ8.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// ============================================================================
// Helpers
// ============================================================================

static std::vector<float> make_random_fp32(int count, unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(count);
    for (auto &x : v)
        x = dist(rng);
    return v;
}

struct BenchResult
{
    double total_sec;
    int64_t total_ops;
    double ops_per_sec() const { return total_ops / total_sec; }
    double ns_per_op() const { return total_sec * 1e9 / total_ops; }
};

// ============================================================================
// D = 128  (Qwen3 / Llama-3 head_dim)
// ============================================================================

static constexpr int D = 128;
using Block = TQ4Block<D>;

TEST(Perf__TurboQuant, Quantize_ScalarFull_D128)
{
    const int num_vectors = 1024;
    const int iterations = 200;
    const int warmup = 50;

    auto fp32_data = make_random_fp32(num_vectors * D, 42);
    TurboQuantContext ctx(D, 31, 131);

    std::vector<Block> blocks(num_vectors);
    alignas(64) float scratch0[D], scratch1[D];

    // Warmup
    for (int w = 0; w < warmup; ++w)
        for (int i = 0; i < num_vectors; ++i)
            turboquant_quantize_tq4<D>(
                fp32_data.data() + i * D, ctx, blocks[i], scratch0, scratch1);

    // Bench
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
        for (int i = 0; i < num_vectors; ++i)
            turboquant_quantize_tq4<D>(
                fp32_data.data() + i * D, ctx, blocks[i], scratch0, scratch1);
    auto t1 = std::chrono::high_resolution_clock::now();

    BenchResult r{std::chrono::duration<double>(t1 - t0).count(),
                  static_cast<int64_t>(iterations) * num_vectors};

    std::cout << "\n=== TQ4 Quantize (D=" << D << ") ===" << std::endl;
    std::cout << "  Vectors:     " << r.total_ops << std::endl;
    std::cout << "  Time:        " << std::fixed << std::setprecision(3) << r.total_sec * 1e3 << " ms" << std::endl;
    std::cout << "  Throughput:  " << std::fixed << std::setprecision(0) << r.ops_per_sec() << " vec/s" << std::endl;
    std::cout << "  Latency:     " << std::fixed << std::setprecision(0) << r.ns_per_op() << " ns/vec" << std::endl;

    double input_gb = static_cast<double>(r.total_ops) * D * 4 / 1e9;
    std::cout << "  Input B/W:   " << std::fixed << std::setprecision(2) << input_gb / r.total_sec << " GB/s" << std::endl;

    EXPECT_GT(r.ops_per_sec(), 0);
}

TEST(Perf__TurboQuant, Dequant_ScalarFull_D128)
{
    const int num_blocks = 1024;
    const int iterations = 500;
    const int warmup = 100;

    // Generate blocks by quantizing random data
    auto fp32_data = make_random_fp32(num_blocks * D, 42);
    TurboQuantContext ctx(D, 31, 131);

    std::vector<Block> blocks(num_blocks);
    alignas(64) float s0[D], s1[D];
    for (int i = 0; i < num_blocks; ++i)
        turboquant_quantize_tq4<D>(
            fp32_data.data() + i * D, ctx, blocks[i], s0, s1);

    std::vector<float> output(num_blocks * D);
    alignas(64) float scratch[D];

    // Warmup
    for (int w = 0; w < warmup; ++w)
        for (int i = 0; i < num_blocks; ++i)
            turboquant_dequantize_tq4<D>(
                blocks[i], ctx, output.data() + i * D, scratch);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
        for (int i = 0; i < num_blocks; ++i)
            turboquant_dequantize_tq4<D>(
                blocks[i], ctx, output.data() + i * D, scratch);
    auto t1 = std::chrono::high_resolution_clock::now();

    BenchResult r{std::chrono::duration<double>(t1 - t0).count(),
                  static_cast<int64_t>(iterations) * num_blocks};

    std::cout << "\n=== TQ4 Dequant (D=" << D << ") ===" << std::endl;
    std::cout << "  Blocks:      " << r.total_ops << std::endl;
    std::cout << "  Time:        " << std::fixed << std::setprecision(3) << r.total_sec * 1e3 << " ms" << std::endl;
    std::cout << "  Throughput:  " << std::fixed << std::setprecision(0) << r.ops_per_sec() << " vec/s" << std::endl;
    std::cout << "  Latency:     " << std::fixed << std::setprecision(0) << r.ns_per_op() << " ns/vec" << std::endl;

    // Block read bandwidth: 72 bytes per block
    double block_gb = static_cast<double>(r.total_ops) * sizeof(Block) / 1e9;
    std::cout << "  Block B/W:   " << std::fixed << std::setprecision(2) << block_gb / r.total_sec << " GB/s" << std::endl;

    EXPECT_GT(r.ops_per_sec(), 0);
}

TEST(Perf__TurboQuant, RoundTrip_QuantDequant_D128)
{
    // Full quantize + dequantize roundtrip (e.g., KV cache append + attention dequant)
    const int num_vectors = 1024;
    const int iterations = 200;
    const int warmup = 50;

    auto fp32_data = make_random_fp32(num_vectors * D, 42);
    TurboQuantContext ctx(D, 31, 131);

    std::vector<Block> blocks(num_vectors);
    std::vector<float> output(num_vectors * D);

    alignas(64) float s0[D], s1[D], scratch[D];

    // Warmup
    for (int w = 0; w < warmup; ++w)
        for (int i = 0; i < num_vectors; ++i)
        {
            turboquant_quantize_tq4<D>(
                fp32_data.data() + i * D, ctx, blocks[i], s0, s1);
            turboquant_dequantize_tq4<D>(
                blocks[i], ctx, output.data() + i * D, scratch);
        }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
        for (int i = 0; i < num_vectors; ++i)
        {
            turboquant_quantize_tq4<D>(
                fp32_data.data() + i * D, ctx, blocks[i], s0, s1);
            turboquant_dequantize_tq4<D>(
                blocks[i], ctx, output.data() + i * D, scratch);
        }
    auto t1 = std::chrono::high_resolution_clock::now();

    BenchResult r{std::chrono::duration<double>(t1 - t0).count(),
                  static_cast<int64_t>(iterations) * num_vectors};

    std::cout << "\n=== TQ4 Roundtrip (quant + dequant, D=" << D << ") ===" << std::endl;
    std::cout << "  Roundtrips:  " << r.total_ops << std::endl;
    std::cout << "  Time:        " << std::fixed << std::setprecision(3) << r.total_sec * 1e3 << " ms" << std::endl;
    std::cout << "  Throughput:  " << std::fixed << std::setprecision(0) << r.ops_per_sec() << " rt/s" << std::endl;
    std::cout << "  Latency:     " << std::fixed << std::setprecision(0) << r.ns_per_op() << " ns/rt" << std::endl;

    EXPECT_GT(r.ops_per_sec(), 0);
}

// ============================================================================
// TQ8 Benchmarks  (D = 128)
// ============================================================================

using TQ8Block128 = TQ8Block<D>;

TEST(Perf__TurboQuant, TQ8_Quantize_D128)
{
    const int num_vectors = 1024;
    const int iterations = 200;
    const int warmup = 50;

    auto fp32_data = make_random_fp32(num_vectors * D, 42);
    TurboQuantContext ctx(D, 31, 131);

    std::vector<TQ8Block128> blocks(num_vectors);
    alignas(64) float scratch0[D], scratch1[D];

    for (int w = 0; w < warmup; ++w)
        for (int i = 0; i < num_vectors; ++i)
            turboquant_quantize_tq8<D>(
                fp32_data.data() + i * D, ctx, blocks[i], scratch0, scratch1);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
        for (int i = 0; i < num_vectors; ++i)
            turboquant_quantize_tq8<D>(
                fp32_data.data() + i * D, ctx, blocks[i], scratch0, scratch1);
    auto t1 = std::chrono::high_resolution_clock::now();

    BenchResult r{std::chrono::duration<double>(t1 - t0).count(),
                  static_cast<int64_t>(iterations) * num_vectors};

    std::cout << "\n=== TQ8 Quantize (D=" << D << ") ===" << std::endl;
    std::cout << "  Vectors:     " << r.total_ops << std::endl;
    std::cout << "  Time:        " << std::fixed << std::setprecision(3) << r.total_sec * 1e3 << " ms" << std::endl;
    std::cout << "  Throughput:  " << std::fixed << std::setprecision(0) << r.ops_per_sec() << " vec/s" << std::endl;
    std::cout << "  Latency:     " << std::fixed << std::setprecision(0) << r.ns_per_op() << " ns/vec" << std::endl;

    double input_gb = static_cast<double>(r.total_ops) * D * 4 / 1e9;
    std::cout << "  Input B/W:   " << std::fixed << std::setprecision(2) << input_gb / r.total_sec << " GB/s" << std::endl;

    EXPECT_GT(r.ops_per_sec(), 0);
}

TEST(Perf__TurboQuant, TQ8_Dequant_D128)
{
    const int num_blocks = 1024;
    const int iterations = 500;
    const int warmup = 100;

    auto fp32_data = make_random_fp32(num_blocks * D, 42);
    TurboQuantContext ctx(D, 31, 131);

    std::vector<TQ8Block128> blocks(num_blocks);
    alignas(64) float s0[D], s1[D];
    for (int i = 0; i < num_blocks; ++i)
        turboquant_quantize_tq8<D>(
            fp32_data.data() + i * D, ctx, blocks[i], s0, s1);

    std::vector<float> output(num_blocks * D);
    alignas(64) float scratch[D];

    for (int w = 0; w < warmup; ++w)
        for (int i = 0; i < num_blocks; ++i)
            turboquant_dequantize_tq8<D>(
                blocks[i], ctx, output.data() + i * D, scratch);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
        for (int i = 0; i < num_blocks; ++i)
            turboquant_dequantize_tq8<D>(
                blocks[i], ctx, output.data() + i * D, scratch);
    auto t1 = std::chrono::high_resolution_clock::now();

    BenchResult r{std::chrono::duration<double>(t1 - t0).count(),
                  static_cast<int64_t>(iterations) * num_blocks};

    std::cout << "\n=== TQ8 Dequant (D=" << D << ") ===" << std::endl;
    std::cout << "  Blocks:      " << r.total_ops << std::endl;
    std::cout << "  Time:        " << std::fixed << std::setprecision(3) << r.total_sec * 1e3 << " ms" << std::endl;
    std::cout << "  Throughput:  " << std::fixed << std::setprecision(0) << r.ops_per_sec() << " vec/s" << std::endl;
    std::cout << "  Latency:     " << std::fixed << std::setprecision(0) << r.ns_per_op() << " ns/vec" << std::endl;

    double block_gb = static_cast<double>(r.total_ops) * sizeof(TQ8Block128) / 1e9;
    std::cout << "  Block B/W:   " << std::fixed << std::setprecision(2) << block_gb / r.total_sec << " GB/s" << std::endl;

    EXPECT_GT(r.ops_per_sec(), 0);
}

TEST(Perf__TurboQuant, TQ8_RoundTrip_QuantDequant_D128)
{
    const int num_vectors = 1024;
    const int iterations = 200;
    const int warmup = 50;

    auto fp32_data = make_random_fp32(num_vectors * D, 42);
    TurboQuantContext ctx(D, 31, 131);

    std::vector<TQ8Block128> blocks(num_vectors);
    std::vector<float> output(num_vectors * D);

    alignas(64) float s0[D], s1[D], scratch[D];

    for (int w = 0; w < warmup; ++w)
        for (int i = 0; i < num_vectors; ++i)
        {
            turboquant_quantize_tq8<D>(
                fp32_data.data() + i * D, ctx, blocks[i], s0, s1);
            turboquant_dequantize_tq8<D>(
                blocks[i], ctx, output.data() + i * D, scratch);
        }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
        for (int i = 0; i < num_vectors; ++i)
        {
            turboquant_quantize_tq8<D>(
                fp32_data.data() + i * D, ctx, blocks[i], s0, s1);
            turboquant_dequantize_tq8<D>(
                blocks[i], ctx, output.data() + i * D, scratch);
        }
    auto t1 = std::chrono::high_resolution_clock::now();

    BenchResult r{std::chrono::duration<double>(t1 - t0).count(),
                  static_cast<int64_t>(iterations) * num_vectors};

    std::cout << "\n=== TQ8 Roundtrip (quant + dequant, D=" << D << ") ===" << std::endl;
    std::cout << "  Roundtrips:  " << r.total_ops << std::endl;
    std::cout << "  Time:        " << std::fixed << std::setprecision(3) << r.total_sec * 1e3 << " ms" << std::endl;
    std::cout << "  Throughput:  " << std::fixed << std::setprecision(0) << r.ops_per_sec() << " rt/s" << std::endl;
    std::cout << "  Latency:     " << std::fixed << std::setprecision(0) << r.ns_per_op() << " ns/rt" << std::endl;

    EXPECT_GT(r.ops_per_sec(), 0);
}

// ============================================================================
// TQ4 vs TQ8 Head-to-Head Comparison
// ============================================================================

TEST(Perf__TurboQuant, TQ4_vs_TQ8_Quantize_HeadToHead)
{
    const int num_vectors = 1024;
    const int iterations = 200;
    const int warmup = 50;

    auto fp32_data = make_random_fp32(num_vectors * D, 42);
    TurboQuantContext ctx(D, 31, 131);

    std::vector<Block> blocks4(num_vectors);
    std::vector<TQ8Block128> blocks8(num_vectors);
    alignas(64) float s0[D], s1[D];

    // TQ4
    for (int w = 0; w < warmup; ++w)
        for (int i = 0; i < num_vectors; ++i)
            turboquant_quantize_tq4<D>(fp32_data.data() + i * D, ctx, blocks4[i], s0, s1);

    auto t0_4 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
        for (int i = 0; i < num_vectors; ++i)
            turboquant_quantize_tq4<D>(fp32_data.data() + i * D, ctx, blocks4[i], s0, s1);
    auto t1_4 = std::chrono::high_resolution_clock::now();
    double sec_tq4 = std::chrono::duration<double>(t1_4 - t0_4).count();

    // TQ8
    for (int w = 0; w < warmup; ++w)
        for (int i = 0; i < num_vectors; ++i)
            turboquant_quantize_tq8<D>(fp32_data.data() + i * D, ctx, blocks8[i], s0, s1);

    auto t0_8 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
        for (int i = 0; i < num_vectors; ++i)
            turboquant_quantize_tq8<D>(fp32_data.data() + i * D, ctx, blocks8[i], s0, s1);
    auto t1_8 = std::chrono::high_resolution_clock::now();
    double sec_tq8 = std::chrono::duration<double>(t1_8 - t0_8).count();

    int64_t total_ops = static_cast<int64_t>(iterations) * num_vectors;
    double tq4_ns = sec_tq4 * 1e9 / total_ops;
    double tq8_ns = sec_tq8 * 1e9 / total_ops;

    std::cout << "\n=== TQ4 vs TQ8 Quantize (D=" << D << ") ===" << std::endl;
    std::cout << "  TQ4:  " << std::fixed << std::setprecision(0) << tq4_ns << " ns/vec  ("
              << std::setprecision(0) << total_ops / sec_tq4 << " vec/s)" << std::endl;
    std::cout << "  TQ8:  " << std::fixed << std::setprecision(0) << tq8_ns << " ns/vec  ("
              << std::setprecision(0) << total_ops / sec_tq8 << " vec/s)" << std::endl;
    std::cout << "  TQ8/TQ4 ratio: " << std::fixed << std::setprecision(2) << tq8_ns / tq4_ns << "x" << std::endl;
    std::cout << "  TQ4 block size: " << sizeof(Block) << " bytes" << std::endl;
    std::cout << "  TQ8 block size: " << sizeof(TQ8Block128) << " bytes" << std::endl;

    EXPECT_GT(total_ops / sec_tq4, 0);
    EXPECT_GT(total_ops / sec_tq8, 0);
}

TEST(Perf__TurboQuant, TQ4_vs_TQ8_Dequant_HeadToHead)
{
    const int num_blocks = 1024;
    const int iterations = 500;
    const int warmup = 100;

    auto fp32_data = make_random_fp32(num_blocks * D, 42);
    TurboQuantContext ctx(D, 31, 131);

    std::vector<Block> blocks4(num_blocks);
    std::vector<TQ8Block128> blocks8(num_blocks);
    alignas(64) float s0[D], s1[D];
    for (int i = 0; i < num_blocks; ++i)
    {
        turboquant_quantize_tq4<D>(fp32_data.data() + i * D, ctx, blocks4[i], s0, s1);
        turboquant_quantize_tq8<D>(fp32_data.data() + i * D, ctx, blocks8[i], s0, s1);
    }

    std::vector<float> output(num_blocks * D);
    alignas(64) float scratch[D];

    // TQ4 dequant
    for (int w = 0; w < warmup; ++w)
        for (int i = 0; i < num_blocks; ++i)
            turboquant_dequantize_tq4<D>(blocks4[i], ctx, output.data() + i * D, scratch);

    auto t0_4 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
        for (int i = 0; i < num_blocks; ++i)
            turboquant_dequantize_tq4<D>(blocks4[i], ctx, output.data() + i * D, scratch);
    auto t1_4 = std::chrono::high_resolution_clock::now();
    double sec_tq4 = std::chrono::duration<double>(t1_4 - t0_4).count();

    // TQ8 dequant
    for (int w = 0; w < warmup; ++w)
        for (int i = 0; i < num_blocks; ++i)
            turboquant_dequantize_tq8<D>(blocks8[i], ctx, output.data() + i * D, scratch);

    auto t0_8 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
        for (int i = 0; i < num_blocks; ++i)
            turboquant_dequantize_tq8<D>(blocks8[i], ctx, output.data() + i * D, scratch);
    auto t1_8 = std::chrono::high_resolution_clock::now();
    double sec_tq8 = std::chrono::duration<double>(t1_8 - t0_8).count();

    int64_t total_ops = static_cast<int64_t>(iterations) * num_blocks;
    double tq4_ns = sec_tq4 * 1e9 / total_ops;
    double tq8_ns = sec_tq8 * 1e9 / total_ops;

    std::cout << "\n=== TQ4 vs TQ8 Dequant (D=" << D << ") ===" << std::endl;
    std::cout << "  TQ4:  " << std::fixed << std::setprecision(0) << tq4_ns << " ns/vec  ("
              << std::setprecision(0) << total_ops / sec_tq4 << " vec/s)" << std::endl;
    std::cout << "  TQ8:  " << std::fixed << std::setprecision(0) << tq8_ns << " ns/vec  ("
              << std::setprecision(0) << total_ops / sec_tq8 << " vec/s)" << std::endl;
    std::cout << "  TQ8/TQ4 ratio: " << std::fixed << std::setprecision(2) << tq8_ns / tq4_ns << "x" << std::endl;

    EXPECT_GT(total_ops / sec_tq4, 0);
    EXPECT_GT(total_ops / sec_tq8, 0);
}

// ============================================================================
// KV Cache: Append + Gather throughput (split TQ8-K / TQ4-V)
// ============================================================================

TEST(Perf__TurboQuant, KVCache_SplitTQ_AppendGather_D128)
{
    static constexpr int N_KV_HEADS = 8;
    static constexpr int KV_DIM = N_KV_HEADS * D;
    static constexpr int MAX_SEQ = 4096;
    static constexpr int N_PREFILL = 512;
    static constexpr int N_DECODE_STEPS = 200;
    static constexpr int WARMUP_STEPS = 20;

    MPIContext mpi_ctx{0, 1, MPI_COMM_WORLD};
    TurboQuantContext tq_ctx(D, 31, 131);

    CPURingKVCacheTQ cache(mpi_ctx, /*n_layers=*/1, /*batch_size=*/1, MAX_SEQ,
                           N_KV_HEADS, D, DeviceId::cpu());

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    auto make_fp32 = [&](int rows)
    {
        auto t = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(KV_DIM)});
        float *d = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
            d[i] = dist(rng);
        return t;
    };

    auto quantize_k = [&](const FP32Tensor &src)
    {
        return TQ8Tensor::quantize_from_fp32(src.data(), src.shape(), D, tq_ctx);
    };
    auto quantize_v = [&](const FP32Tensor &src)
    {
        return TQ4Tensor::quantize_from_fp32(src.data(), src.shape(), D, tq_ctx);
    };

    auto fp32_prefill_k = make_fp32(N_PREFILL);
    auto fp32_prefill_v = make_fp32(N_PREFILL);
    auto tq8_prefill_k = quantize_k(*fp32_prefill_k);
    auto tq4_prefill_v = quantize_v(*fp32_prefill_v);

    std::vector<std::shared_ptr<TQ8Tensor>> decode_k_tokens(N_DECODE_STEPS + WARMUP_STEPS);
    std::vector<std::shared_ptr<TQ4Tensor>> decode_v_tokens(N_DECODE_STEPS + WARMUP_STEPS);
    for (int i = 0; i < N_DECODE_STEPS + WARMUP_STEPS; ++i)
    {
        auto fk = make_fp32(1);
        auto fv = make_fp32(1);
        decode_k_tokens[i] = quantize_k(*fk);
        decode_v_tokens[i] = quantize_v(*fv);
    }

    auto out_k = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, D);
    auto out_v = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, D);
    std::vector<int> kv_lens;

    // Prefill
    auto t0_prefill = std::chrono::high_resolution_clock::now();
    cache.append_kv(0, 0, tq8_prefill_k.get(), tq4_prefill_v.get(), N_PREFILL);
    auto t1_prefill = std::chrono::high_resolution_clock::now();
    double prefill_sec = std::chrono::duration<double>(t1_prefill - t0_prefill).count();

    // Warmup decode
    for (int s = 0; s < WARMUP_STEPS; ++s)
    {
        cache.append_kv(0, 0, decode_k_tokens[s].get(), decode_v_tokens[s].get(), 1);
        cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    }

    // Timed decode
    double total_append_sec = 0.0;
    double total_gather_sec = 0.0;

    for (int s = 0; s < N_DECODE_STEPS; ++s)
    {
        int tok_idx = WARMUP_STEPS + s;

        auto ta0 = std::chrono::high_resolution_clock::now();
        cache.append_kv(0, 0, decode_k_tokens[tok_idx].get(), decode_v_tokens[tok_idx].get(), 1);
        auto ta1 = std::chrono::high_resolution_clock::now();
        total_append_sec += std::chrono::duration<double>(ta1 - ta0).count();

        auto tg0 = std::chrono::high_resolution_clock::now();
        cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
        auto tg1 = std::chrono::high_resolution_clock::now();
        total_gather_sec += std::chrono::duration<double>(tg1 - tg0).count();
    }

    double total_decode_sec = total_append_sec + total_gather_sec;
    double avg_kv_len = N_PREFILL + WARMUP_STEPS + N_DECODE_STEPS / 2.0;
    double append_ns = total_append_sec * 1e9 / N_DECODE_STEPS;
    double gather_ns = total_gather_sec * 1e9 / N_DECODE_STEPS;
    double step_ns = total_decode_sec * 1e9 / N_DECODE_STEPS;

    double k_bytes_per_row = static_cast<double>(N_KV_HEADS) * sizeof(TQ8Block128);
    double v_bytes_per_row = static_cast<double>(N_KV_HEADS) * sizeof(Block);
    double gather_bytes = avg_kv_len * (k_bytes_per_row + v_bytes_per_row);
    double gather_gb = gather_bytes * N_DECODE_STEPS / 1e9;
    double gather_bw = gather_gb / total_gather_sec;

    std::cout << "\n=== KV Cache Split TQ (TQ8-K / TQ4-V, D=" << D
              << ", N_KV_HEADS=" << N_KV_HEADS << ") ===" << std::endl;
    std::cout << "  Max seq:       " << MAX_SEQ << std::endl;
    std::cout << "  Prefill:       " << N_PREFILL << " tokens in " << std::fixed << std::setprecision(3)
              << prefill_sec * 1e3 << " ms ("
              << std::setprecision(0) << N_PREFILL / prefill_sec << " tok/s)" << std::endl;
    std::cout << "  Decode steps:  " << N_DECODE_STEPS << std::endl;
    std::cout << "  Avg KV len:    " << std::fixed << std::setprecision(0) << avg_kv_len << " tokens" << std::endl;
    std::cout << "  Append:        " << std::fixed << std::setprecision(0) << append_ns << " ns/step" << std::endl;
    std::cout << "  Gather:        " << std::fixed << std::setprecision(0) << gather_ns << " ns/step" << std::endl;
    std::cout << "  Total step:    " << std::fixed << std::setprecision(0) << step_ns << " ns/step" << std::endl;
    std::cout << "  Gather B/W:    " << std::fixed << std::setprecision(2) << gather_bw << " GB/s" << std::endl;
    std::cout << "  K block size:  " << sizeof(TQ8Block128) << " bytes (TQ8)" << std::endl;
    std::cout << "  V block size:  " << sizeof(Block) << " bytes (TQ4)" << std::endl;
    std::cout << "  Row bytes:     " << std::fixed << std::setprecision(0)
              << k_bytes_per_row + v_bytes_per_row << " (K=" << k_bytes_per_row
              << " + V=" << v_bytes_per_row << ")" << std::endl;

    EXPECT_GT(N_DECODE_STEPS / total_decode_sec, 0);
}

// ============================================================================
// Row-level quantize/dequant throughput (multi-head KV append hot path)
// ============================================================================

TEST(Perf__TurboQuant, TQ8_QuantizeRow_MultiHead_D128)
{
    static constexpr int N_KV_HEADS = 8;
    static constexpr int KV_DIM = N_KV_HEADS * D;
    const int num_rows = 256;
    const int iterations = 200;
    const int warmup = 50;

    auto fp32_data = make_random_fp32(num_rows * KV_DIM, 42);
    TurboQuantContext ctx(D, 31, 131);

    std::vector<TQ8Block128> blocks(num_rows * N_KV_HEADS);
    alignas(64) float s0[D], s1[D];

    for (int w = 0; w < warmup; ++w)
        for (int r = 0; r < num_rows; ++r)
            turboquant_quantize_row_tq8<D>(
                fp32_data.data() + r * KV_DIM, ctx,
                blocks.data() + r * N_KV_HEADS, N_KV_HEADS, s0, s1);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
        for (int r = 0; r < num_rows; ++r)
            turboquant_quantize_row_tq8<D>(
                fp32_data.data() + r * KV_DIM, ctx,
                blocks.data() + r * N_KV_HEADS, N_KV_HEADS, s0, s1);
    auto t1 = std::chrono::high_resolution_clock::now();

    int64_t total_rows = static_cast<int64_t>(iterations) * num_rows;
    int64_t total_vectors = total_rows * N_KV_HEADS;
    BenchResult r{std::chrono::duration<double>(t1 - t0).count(), total_vectors};

    double row_ns = r.total_sec * 1e9 / total_rows;

    std::cout << "\n=== TQ8 Row Quantize (D=" << D << ", " << N_KV_HEADS << " heads/row) ===" << std::endl;
    std::cout << "  Total vectors: " << r.total_ops << " (" << total_rows << " rows)" << std::endl;
    std::cout << "  Time:          " << std::fixed << std::setprecision(3) << r.total_sec * 1e3 << " ms" << std::endl;
    std::cout << "  Per-vector:    " << std::fixed << std::setprecision(0) << r.ns_per_op() << " ns/vec" << std::endl;
    std::cout << "  Per-row:       " << std::fixed << std::setprecision(0) << row_ns << " ns/row" << std::endl;
    std::cout << "  Row throughput:" << std::fixed << std::setprecision(0) << total_rows / r.total_sec << " rows/s" << std::endl;
    std::cout << "  Vec throughput:" << std::fixed << std::setprecision(0) << r.ops_per_sec() << " vec/s" << std::endl;

    double input_gb = static_cast<double>(total_rows) * KV_DIM * 4 / 1e9;
    std::cout << "  Input B/W:     " << std::fixed << std::setprecision(2) << input_gb / r.total_sec << " GB/s" << std::endl;

    EXPECT_GT(r.ops_per_sec(), 0);
}

TEST(Perf__TurboQuant, TQ8_DequantRow_MultiHead_D128)
{
    static constexpr int N_KV_HEADS = 8;
    static constexpr int KV_DIM = N_KV_HEADS * D;
    const int num_rows = 256;
    const int iterations = 500;
    const int warmup = 100;

    auto fp32_data = make_random_fp32(num_rows * KV_DIM, 42);
    TurboQuantContext ctx(D, 31, 131);

    std::vector<TQ8Block128> blocks(num_rows * N_KV_HEADS);
    alignas(64) float s0[D], s1[D];
    for (int r = 0; r < num_rows; ++r)
        turboquant_quantize_row_tq8<D>(
            fp32_data.data() + r * KV_DIM, ctx,
            blocks.data() + r * N_KV_HEADS, N_KV_HEADS, s0, s1);

    std::vector<float> output(num_rows * KV_DIM);
    alignas(64) float scratch[D];

    for (int w = 0; w < warmup; ++w)
        for (int r = 0; r < num_rows; ++r)
            turboquant_dequantize_row_tq8<D>(
                blocks.data() + r * N_KV_HEADS, ctx,
                output.data() + r * KV_DIM, N_KV_HEADS, scratch);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
        for (int r = 0; r < num_rows; ++r)
            turboquant_dequantize_row_tq8<D>(
                blocks.data() + r * N_KV_HEADS, ctx,
                output.data() + r * KV_DIM, N_KV_HEADS, scratch);
    auto t1 = std::chrono::high_resolution_clock::now();

    int64_t total_rows = static_cast<int64_t>(iterations) * num_rows;
    int64_t total_vectors = total_rows * N_KV_HEADS;
    BenchResult r_bench{std::chrono::duration<double>(t1 - t0).count(), total_vectors};

    double row_ns = r_bench.total_sec * 1e9 / total_rows;

    std::cout << "\n=== TQ8 Row Dequant (D=" << D << ", " << N_KV_HEADS << " heads/row) ===" << std::endl;
    std::cout << "  Total vectors: " << r_bench.total_ops << " (" << total_rows << " rows)" << std::endl;
    std::cout << "  Time:          " << std::fixed << std::setprecision(3) << r_bench.total_sec * 1e3 << " ms" << std::endl;
    std::cout << "  Per-vector:    " << std::fixed << std::setprecision(0) << r_bench.ns_per_op() << " ns/vec" << std::endl;
    std::cout << "  Per-row:       " << std::fixed << std::setprecision(0) << row_ns << " ns/row" << std::endl;
    std::cout << "  Row throughput:" << std::fixed << std::setprecision(0) << total_rows / r_bench.total_sec << " rows/s" << std::endl;
    std::cout << "  Vec throughput:" << std::fixed << std::setprecision(0) << r_bench.ops_per_sec() << " vec/s" << std::endl;

    double block_gb = static_cast<double>(total_rows) * N_KV_HEADS * sizeof(TQ8Block128) / 1e9;
    std::cout << "  Block B/W:     " << std::fixed << std::setprecision(2) << block_gb / r_bench.total_sec << " GB/s" << std::endl;

    EXPECT_GT(r_bench.ops_per_sec(), 0);
}

// ============================================================================
// Production fused KV publication/materialization matrix
// ============================================================================

namespace
{
    struct ProductionKVGeometry
    {
        size_t tq4_block_bytes = 0;
        size_t tq8_block_bytes = 0;
    };

    ProductionKVGeometry production_kv_geometry(int head_dim)
    {
        switch (head_dim)
        {
        case 64:
            return {sizeof(TQ4Block_64), sizeof(TQ8Block_64)};
        case 128:
            return {sizeof(TQ4Block_128), sizeof(TQ8Block_128)};
        case 256:
            return {sizeof(TQ4Block_256), sizeof(TQ8Block_256)};
        default:
            throw std::invalid_argument("unsupported TurboQuant head dimension");
        }
    }

    int production_kv_iterations(int rows)
    {
        if (rows <= 2)
            return 100;
        if (rows <= 8)
            return 40;
        if (rows <= 32)
            return 12;
        return 3;
    }

    /**
     * @brief Own every persistent buffer required by one production KV point.
     *
     * Construction deliberately performs allocation and deterministic input
     * generation once. The public operations below therefore contain exactly
     * the CPU KV-cache hot paths used by append and attention materialization.
     */
    class ProductionKVCase final
    {
    public:
        ProductionKVCase(
            int head_dim,
            int rows,
            int n_kv_heads,
            bool value_uses_tq8)
            : head_dim_(head_dim),
              rows_(rows),
              n_kv_heads_(n_kv_heads),
              value_uses_tq8_(value_uses_tq8),
              geometry_(production_kv_geometry(head_dim)),
              k_block_bytes_(geometry_.tq8_block_bytes),
              v_block_bytes_(
                  value_uses_tq8 ? geometry_.tq8_block_bytes
                                 : geometry_.tq4_block_bytes),
              k_row_bytes_(
                  static_cast<size_t>(n_kv_heads) * k_block_bytes_),
              v_row_bytes_(
                  static_cast<size_t>(n_kv_heads) * v_block_bytes_),
              elements_(
                  static_cast<size_t>(rows) * n_kv_heads * head_dim),
              k_fp32_(make_random_fp32(
                  static_cast<int>(elements_),
                  static_cast<unsigned>(1000 + head_dim + rows))),
              v_fp32_(make_random_fp32(
                  static_cast<int>(elements_),
                  static_cast<unsigned>(2000 + head_dim + rows))),
              k_output_(elements_),
              v_output_(elements_),
              k_blocks_(static_cast<size_t>(rows) * k_row_bytes_),
              v_blocks_(static_cast<size_t>(rows) * v_row_bytes_),
              context_(head_dim, 31, 131)
        {
        }

        void quantize()
        {
            turboquant_quantize_kv_rows(
                k_fp32_.data(),
                v_fp32_.data(),
                k_blocks_.data(),
                v_blocks_.data(),
                rows_,
                head_dim_,
                n_kv_heads_,
                k_row_bytes_,
                v_row_bytes_,
                k_block_bytes_,
                v_block_bytes_,
                context_,
                value_uses_tq8_);
        }

        void dequantize()
        {
            if (value_uses_tq8_)
            {
                turboquant_dequantize_tq8_kv_rows(
                    k_blocks_.data(),
                    v_blocks_.data(),
                    context_,
                    k_output_.data(),
                    v_output_.data(),
                    0,
                    rows_,
                    head_dim_,
                    n_kv_heads_,
                    k_row_bytes_,
                    v_row_bytes_,
                    k_block_bytes_,
                    v_block_bytes_);
                return;
            }

            turboquant_dequantize_split_kv_rows(
                k_blocks_.data(),
                v_blocks_.data(),
                context_,
                k_output_.data(),
                v_output_.data(),
                0,
                rows_,
                head_dim_,
                n_kv_heads_,
                k_row_bytes_,
                v_row_bytes_,
                k_block_bytes_,
                v_block_bytes_);
        }

        void dequantizeWithRoPE()
        {
            constexpr float kRoPETheta = 1000000.0f;
            constexpr int kPositionStart = 17;
            if (value_uses_tq8_)
            {
                turboquant_dequantize_tq8_kv_rows_with_rope(
                    k_blocks_.data(),
                    v_blocks_.data(),
                    context_,
                    k_output_.data(),
                    v_output_.data(),
                    0,
                    rows_,
                    head_dim_,
                    n_kv_heads_,
                    k_row_bytes_,
                    v_row_bytes_,
                    k_block_bytes_,
                    v_block_bytes_,
                    kRoPETheta,
                    kPositionStart);
                return;
            }

            turboquant_dequantize_split_kv_rows_with_rope(
                k_blocks_.data(),
                v_blocks_.data(),
                context_,
                k_output_.data(),
                v_output_.data(),
                0,
                rows_,
                head_dim_,
                n_kv_heads_,
                k_row_bytes_,
                v_row_bytes_,
                k_block_bytes_,
                v_block_bytes_,
                kRoPETheta,
                kPositionStart);
        }

        double vectorCount() const
        {
            return static_cast<double>(rows_) * n_kv_heads_ * 2.0;
        }

        float checksum() const
        {
            return k_output_.front() + v_output_.back();
        }

    private:
        int head_dim_;
        int rows_;
        int n_kv_heads_;
        bool value_uses_tq8_;
        ProductionKVGeometry geometry_;
        size_t k_block_bytes_;
        size_t v_block_bytes_;
        size_t k_row_bytes_;
        size_t v_row_bytes_;
        size_t elements_;
        std::vector<float> k_fp32_;
        std::vector<float> v_fp32_;
        std::vector<float> k_output_;
        std::vector<float> v_output_;
        std::vector<uint8_t> k_blocks_;
        std::vector<uint8_t> v_blocks_;
        TurboQuantContext context_;
    };

    /**
     * @brief Return the median duration of repeated invocation batches.
     *
     * Short CPU kernels are sensitive to worker wake-up and frequency changes.
     * Five independent batches reject isolated scheduling outliers while
     * retaining the full production call boundary inside every measurement.
     */
    template <typename Callable>
    double production_kv_median_us(Callable &&callable, int iterations)
    {
        constexpr int kSamples = 5;
        std::array<double, kSamples> samples{};
        for (double &sample : samples)
        {
            const auto start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < iterations; ++iteration)
                callable();
            const auto stop = std::chrono::steady_clock::now();
            sample =
                std::chrono::duration<double, std::micro>(stop - start).count() /
                iterations;
        }
        std::sort(samples.begin(), samples.end());
        return samples[kSamples / 2];
    }

    /**
     * @brief Measure the exact helpers called by CPU KV append and attention.
     *
     * Context derivation, OpenMP region creation, ISA dispatch, K/V fusion,
     * and output publication are all inside the timed call. Allocation and
     * random input generation remain outside, matching production ownership.
     */
    void benchmark_production_kv_case(
        int head_dim,
        int rows,
        int n_kv_heads,
        bool value_uses_tq8)
    {
        ProductionKVCase test_case(
            head_dim, rows, n_kv_heads, value_uses_tq8);

        const auto quantize = [&]()
        {
            test_case.quantize();
        };
        const auto dequantize = [&]()
        {
            test_case.dequantize();
        };
        const auto dequantize_with_rope = [&]()
        {
            test_case.dequantizeWithRoPE();
        };

        // Resolve and cache every derived head context before timing.
        quantize();
        dequantize();
        dequantize_with_rope();

        const int iterations = production_kv_iterations(rows);
        const double quant_us =
            production_kv_median_us(quantize, iterations);
        const double dequant_us =
            production_kv_median_us(dequantize, iterations);
        const double dequant_rope_us =
            production_kv_median_us(dequantize_with_rope, iterations);
        const double vectors = test_case.vectorCount();

        // A visible dependency prevents the compiler from proving that the
        // materialized output is dead while staying outside the timed region.
        volatile float checksum = test_case.checksum();
        (void)checksum;

        std::cout << "production_kv,"
                  << (value_uses_tq8 ? "tq8_k_tq8_v" : "tq8_k_tq4_v")
                  << ',' << head_dim
                  << ',' << rows
                  << ',' << n_kv_heads
                  << ',' << std::fixed << std::setprecision(3) << quant_us
                  << ',' << dequant_us
                  << ',' << dequant_rope_us
                  << ',' << (quant_us * 1000.0 / vectors)
                  << ',' << (dequant_us * 1000.0 / vectors)
                  << ',' << (dequant_rope_us * 1000.0 / vectors)
                  << '\n';

        EXPECT_GT(quant_us, 0.0);
        EXPECT_GT(dequant_us, 0.0);
        EXPECT_GT(dequant_rope_us, 0.0);
    }

    /**
     * @brief Read one positive integer profiler control from the environment.
     */
    int production_kv_profile_int(const char *name, int fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return fallback;

        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw || *end != '\0' || parsed <= 0 ||
            parsed > std::numeric_limits<int>::max())
        {
            throw std::invalid_argument(
                std::string(name) + " must be a positive integer");
        }
        return static_cast<int>(parsed);
    }

    /**
     * @brief Return a profiler string control without creating mutable state.
     */
    std::string production_kv_profile_string(
        const char *name,
        const char *fallback)
    {
        const char *raw = std::getenv(name);
        return raw && *raw ? std::string(raw) : std::string(fallback);
    }
} // namespace

/**
 * @brief Sweep decode, verifier, and moderate-prefill CPU TurboQuant regimes.
 *
 * Output columns are:
     * `name,mode,D,M,kv_heads,quant_us,dequant_us,dequant_rope_us,`
     * `quant_ns_per_vector,dequant_ns_per_vector,`
     * `dequant_rope_ns_per_vector`.
 */
TEST(Perf__TurboQuant, ProductionFusedKVMatrix)
{
    std::cout
        << "name,mode,D,M,kv_heads,quant_us,dequant_us,"
           "dequant_rope_us,quant_ns_per_vector,dequant_ns_per_vector,"
           "dequant_rope_ns_per_vector\n";
    constexpr std::array<int, 3> head_dims{64, 128, 256};
    constexpr std::array<int, 7> rows{1, 2, 4, 8, 16, 32, 128};
    constexpr int n_kv_heads = 4;

    for (const int head_dim : head_dims)
    {
        for (const bool value_uses_tq8 : {false, true})
        {
            for (const int row_count : rows)
            {
                benchmark_production_kv_case(
                    head_dim, row_count, n_kv_heads, value_uses_tq8);
            }
        }
    }
}

/**
 * @brief Repeat one production operation for isolated Linux perf collection.
 *
 * Environment controls:
 * - `LLAMINAR_TQ_PROFILE_OP`: `quantize`, `dequantize`, or `dequantize_rope`.
 * - `LLAMINAR_TQ_PROFILE_MODE`: `tq8_k_tq4_v` or `tq8_k_tq8_v`.
 * - `LLAMINAR_TQ_PROFILE_D`: head dimension 64, 128, or 256.
 * - `LLAMINAR_TQ_PROFILE_M`: row count.
 * - `LLAMINAR_TQ_PROFILE_KV_HEADS`: local KV head count.
 * - `LLAMINAR_TQ_PROFILE_ITERS`: measured invocation count.
 *
 * Persistent buffers and the rotation context are constructed before warmup.
 * The repeated loop invokes only the selected production helper, making each
 * perf sample attributable to one operation and one geometry.
 */
TEST(Perf__TurboQuant, ProductionFusedKVProfilePoint)
{
    const std::string operation =
        production_kv_profile_string("LLAMINAR_TQ_PROFILE_OP", "dequantize");
    const std::string mode =
        production_kv_profile_string(
            "LLAMINAR_TQ_PROFILE_MODE", "tq8_k_tq4_v");
    const int head_dim =
        production_kv_profile_int("LLAMINAR_TQ_PROFILE_D", 128);
    const int rows =
        production_kv_profile_int("LLAMINAR_TQ_PROFILE_M", 16);
    const int n_kv_heads =
        production_kv_profile_int("LLAMINAR_TQ_PROFILE_KV_HEADS", 4);
    const int iterations =
        production_kv_profile_int("LLAMINAR_TQ_PROFILE_ITERS", 10000);

    ASSERT_TRUE(head_dim == 64 || head_dim == 128 || head_dim == 256);
    ASSERT_TRUE(mode == "tq8_k_tq4_v" || mode == "tq8_k_tq8_v");
    ASSERT_TRUE(
        operation == "quantize" ||
        operation == "dequantize" ||
        operation == "dequantize_rope");

    ProductionKVCase test_case(
        head_dim, rows, n_kv_heads, mode == "tq8_k_tq8_v");

    /*
     * Materialization profiling must consume representative compressed data.
     * Populate K/V once before selecting the measured operation; otherwise
     * zero-initialized block scales exercise a degenerate fast path that never
     * represents a production cache populated from model activations.
     */
    test_case.quantize();

    const auto invoke = [&]()
    {
        if (operation == "quantize")
            test_case.quantize();
        else if (operation == "dequantize")
            test_case.dequantize();
        else
            test_case.dequantizeWithRoPE();
    };

    constexpr int kWarmupInvocations = 20;
    for (int warmup = 0; warmup < kWarmupInvocations; ++warmup)
        invoke();

    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration)
        invoke();
    const auto stop = std::chrono::steady_clock::now();

    const double latency_us =
        std::chrono::duration<double, std::micro>(stop - start).count() /
        iterations;
    volatile float checksum = test_case.checksum();
    EXPECT_TRUE(std::isfinite(checksum));

    std::cout
        << "production_kv_profile,"
        << operation << ','
        << mode << ','
        << head_dim << ','
        << rows << ','
        << n_kv_heads << ','
        << iterations << ','
        << std::fixed << std::setprecision(3) << latency_us
        << '\n';
}
