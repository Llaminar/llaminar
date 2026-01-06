/**
 * @file Perf__Q16_DynamicScaleRoPE.cpp
 * @brief Performance tests for dynamic-scale Q16→Q16 RoPE
 *
 * Benchmarks scalar, AVX2, and AVX512 implementations of the dynamic-scale
 * Q16→Q16 RoPE which preserves spiky K projection values by using
 * max(input_d) as the output scale.
 */

#include "kernels/cpu/primitives/RoPEPrimitives.h"
#include "tensors/BlockStructures.h"
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>

using namespace llaminar2;
using namespace llaminar2::primitives;

class Perf__Q16_DynamicScaleRoPE : public ::testing::Test
{
protected:
    // Test parameters - simulate realistic attention head processing
    static constexpr int NUM_HEADS = 14;       // Typical for Qwen2 0.5B
    static constexpr int HEAD_DIM = 64;        // Typical head dimension
    static constexpr int SEQ_LEN = 512;        // Moderate sequence length
    static constexpr int NUM_ITERATIONS = 100; // Iterations for stable timing

    // Derived constants
    static constexpr int BLOCKS_PER_HEAD = HEAD_DIM / 32;
    static constexpr int TOTAL_HEADS = NUM_HEADS * SEQ_LEN;

    std::mt19937 rng{42};
    std::uniform_real_distribution<float> dist{-100.0f, 100.0f};

    // Pre-computed sin/cos tables (Q15 format)
    std::vector<int16_t> cos_q15;
    std::vector<int16_t> sin_q15;

    void SetUp() override
    {
        // Generate RoPE sin/cos for head_dim=64
        cos_q15.resize(HEAD_DIM / 2);
        sin_q15.resize(HEAD_DIM / 2);

        const float rope_theta = 10000.0f;
        const int position = 100; // Arbitrary position

        for (int i = 0; i < HEAD_DIM / 2; ++i)
        {
            float freq = 1.0f / std::pow(rope_theta, 2.0f * i / HEAD_DIM);
            float angle = position * freq;
            cos_q15[i] = static_cast<int16_t>(std::round(std::cos(angle) * 32767.0f));
            sin_q15[i] = static_cast<int16_t>(std::round(std::sin(angle) * 32767.0f));
        }
    }

    // Generate random Q16 input with realistic spiky values
    void generate_spiky_q16_input(std::vector<Q16_1Block> &blocks)
    {
        blocks.resize(TOTAL_HEADS * BLOCKS_PER_HEAD);

        for (size_t h = 0; h < TOTAL_HEADS; ++h)
        {
            // Each head gets its own scale based on max value
            float max_val = 0.0f;
            std::vector<float> fp32_vals(HEAD_DIM);

            for (int i = 0; i < HEAD_DIM; ++i)
            {
                fp32_vals[i] = dist(rng);
                // Add occasional spikes (10% chance)
                if (rng() % 10 == 0)
                {
                    fp32_vals[i] *= 1.5f;
                }
                max_val = std::max(max_val, std::abs(fp32_vals[i]));
            }

            // Quantize to Q16 with per-head scale
            float d = max_val / 32767.0f;

            for (int b = 0; b < BLOCKS_PER_HEAD; ++b)
            {
                size_t blk_idx = h * BLOCKS_PER_HEAD + b;
                blocks[blk_idx].d = d;
                blocks[blk_idx].sum_qs = 0;

                for (int i = 0; i < 32; ++i)
                {
                    int idx = b * 32 + i;
                    int16_t qs = static_cast<int16_t>(std::round(fp32_vals[idx] / d));
                    blocks[blk_idx].qs[i] = qs;
                    blocks[blk_idx].sum_qs += qs;
                }
            }
        }
    }

    // Benchmark scalar implementation
    double benchmark_scalar(const std::vector<Q16_1Block> &input,
                            std::vector<Q16_1Block> &output,
                            std::vector<float> &unified_scales)
    {
        output.resize(input.size());
        unified_scales.resize(TOTAL_HEADS);

        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (size_t h = 0; h < TOTAL_HEADS; ++h)
            {
                const Q16_1Block *in_ptr = input.data() + h * BLOCKS_PER_HEAD;
                Q16_1Block *out_ptr = output.data() + h * BLOCKS_PER_HEAD;

                apply_rope_q16_to_q16_head_dynamic_scale_scalar<Q16_1Block>(
                    in_ptr, out_ptr, HEAD_DIM,
                    cos_q15.data(), sin_q15.data(),
                    &unified_scales[h]);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    }

#if defined(__AVX2__)
    // Benchmark AVX2 implementation
    double benchmark_avx2(const std::vector<Q16_1Block> &input,
                          std::vector<Q16_1Block> &output,
                          std::vector<float> &unified_scales)
    {
        output.resize(input.size());
        unified_scales.resize(TOTAL_HEADS);

        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (size_t h = 0; h < TOTAL_HEADS; ++h)
            {
                const Q16_1Block *in_ptr = input.data() + h * BLOCKS_PER_HEAD;
                Q16_1Block *out_ptr = output.data() + h * BLOCKS_PER_HEAD;

                apply_rope_q16_to_q16_head_dynamic_scale_avx2(
                    in_ptr, out_ptr, HEAD_DIM,
                    cos_q15.data(), sin_q15.data(),
                    &unified_scales[h]);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    }
#endif

#if defined(__AVX512F__)
    // Benchmark AVX512 implementation
    double benchmark_avx512(const std::vector<Q16_1Block> &input,
                            std::vector<Q16_1Block> &output,
                            std::vector<float> &unified_scales)
    {
        output.resize(input.size());
        unified_scales.resize(TOTAL_HEADS);

        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (size_t h = 0; h < TOTAL_HEADS; ++h)
            {
                const Q16_1Block *in_ptr = input.data() + h * BLOCKS_PER_HEAD;
                Q16_1Block *out_ptr = output.data() + h * BLOCKS_PER_HEAD;

                apply_rope_q16_to_q16_head_dynamic_scale_avx512(
                    in_ptr, out_ptr, HEAD_DIM,
                    cos_q15.data(), sin_q15.data(),
                    &unified_scales[h]);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    }
#endif

    // Benchmark auto-dispatch implementation
    double benchmark_auto_dispatch(const std::vector<Q16_1Block> &input,
                                   std::vector<Q16_1Block> &output,
                                   std::vector<float> &unified_scales)
    {
        output.resize(input.size());
        unified_scales.resize(TOTAL_HEADS);

        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (size_t h = 0; h < TOTAL_HEADS; ++h)
            {
                const Q16_1Block *in_ptr = input.data() + h * BLOCKS_PER_HEAD;
                Q16_1Block *out_ptr = output.data() + h * BLOCKS_PER_HEAD;

                apply_rope_q16_to_q16_head_dynamic_scale<Q16_1Block>(
                    in_ptr, out_ptr, HEAD_DIM,
                    cos_q15.data(), sin_q15.data(),
                    &unified_scales[h]);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    }

    void print_header()
    {
        const double data_mb = (TOTAL_HEADS * BLOCKS_PER_HEAD * sizeof(Q16_1Block)) / (1024.0 * 1024.0);
        const double heads_per_sec_unit = TOTAL_HEADS / 1000.0;

        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║           Dynamic-Scale Q16→Q16 RoPE Performance Benchmark                   ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Config: " << NUM_HEADS << " heads × " << SEQ_LEN << " positions = "
                  << std::setw(6) << TOTAL_HEADS << " total heads                            ║\n";
        std::cout << "║ Head dim: " << HEAD_DIM << " | Blocks/head: " << BLOCKS_PER_HEAD
                  << " | Data: " << std::fixed << std::setprecision(2) << data_mb << " MB"
                  << " | Iterations: " << NUM_ITERATIONS << "              ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Implementation     │    Time (ms)   │  Heads/ms  │  Speedup  │    Status     ║\n";
        std::cout << "╠════════════════════╪════════════════╪════════════╪═══════════╪═══════════════╣\n";
    }

    void print_row(const char *name, double time_ms, double baseline_ms, bool is_baseline = false)
    {
        double heads_per_ms = TOTAL_HEADS / time_ms;
        double speedup = baseline_ms / time_ms;
        const char *status = is_baseline ? "BASELINE" : (speedup > 1.0 ? "✓ FASTER" : "✗ SLOWER");

        std::cout << "║ " << std::left << std::setw(18) << name << " │ "
                  << std::right << std::fixed << std::setprecision(3) << std::setw(14) << time_ms << " │ "
                  << std::setw(10) << std::setprecision(1) << heads_per_ms << " │ "
                  << std::setw(9) << std::setprecision(2) << speedup << "x │ "
                  << std::setw(13) << status << " ║\n";
    }

    void print_footer()
    {
        std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
    }
};

// =============================================================================
// Performance Test: Compare Scalar vs SIMD implementations
// =============================================================================

TEST_F(Perf__Q16_DynamicScaleRoPE, ScalarVsAVX2VsAVX512)
{
    std::vector<Q16_1Block> input;
    std::vector<Q16_1Block> output;
    std::vector<float> unified_scales;

    // Generate realistic spiky input data
    generate_spiky_q16_input(input);

    print_header();

    // Benchmark scalar (baseline)
    double scalar_ms = benchmark_scalar(input, output, unified_scales);
    print_row("Scalar", scalar_ms, scalar_ms, true);

#if defined(__AVX2__)
    // Benchmark AVX2
    double avx2_ms = benchmark_avx2(input, output, unified_scales);
    print_row("AVX2", avx2_ms, scalar_ms);
    // Note: AVX2 may not be faster for small head_dim=64 due to setup overhead
    // We don't assert it's faster - just measure and report
#else
    std::cout << "║ AVX2               │           N/A  │        N/A │       N/A │  NOT AVAILABLE ║\n";
#endif

#if defined(__AVX512F__)
    // Benchmark AVX512
    double avx512_ms = benchmark_avx512(input, output, unified_scales);
    print_row("AVX512", avx512_ms, scalar_ms);
#else
    std::cout << "║ AVX512             │           N/A  │        N/A │       N/A │  NOT AVAILABLE ║\n";
#endif

    // Benchmark auto-dispatch (should pick best available)
    double auto_ms = benchmark_auto_dispatch(input, output, unified_scales);
    print_row("Auto-Dispatch", auto_ms, scalar_ms);

    print_footer();

    // AVX512 should always be faster than scalar for this workload
#if defined(__AVX512F__)
    EXPECT_LT(avx512_ms, scalar_ms) << "AVX512 should be faster than scalar";
    // Auto-dispatch should select AVX512 when available
    EXPECT_NEAR(auto_ms, avx512_ms, avx512_ms * 0.15)
        << "Auto-dispatch should use AVX512 when available";
#endif

    // Note: AVX2 may not be faster than scalar for small head_dim=64 due to setup overhead
    // The 256-bit registers don't provide enough benefit for 64-element heads
    // AVX2 speedup typically requires larger workloads (head_dim >= 128)
#if defined(__AVX2__) && !defined(__AVX512F__)
    // When only AVX2 is available (no AVX512), auto-dispatch should use it
    EXPECT_NEAR(auto_ms, avx2_ms, avx2_ms * 0.15)
        << "Auto-dispatch should use AVX2 when AVX512 unavailable";
    std::cout << "\n[INFO] AVX2 performance note: For small head_dim=64, AVX2 may have\n";
    std::cout << "       higher overhead than scalar. This is expected behavior.\n";
#endif
}

// =============================================================================
// Performance Test: Scaling with different head counts
// =============================================================================

TEST_F(Perf__Q16_DynamicScaleRoPE, ScalingWithHeadCount)
{
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           Scaling Test: Varying Total Head Count                             ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Heads   │ Scalar (ms) │ AVX2 (ms) │ AVX512 (ms) │ Auto (ms) │ Best Speedup  ║\n";
    std::cout << "╠══════════╪═════════════╪═══════════╪═════════════╪═══════════╪═══════════════╣\n";

    const std::vector<int> head_counts = {256, 1024, 4096, 8192, 16384};

    for (int total_heads : head_counts)
    {
        // Generate input for this head count
        std::vector<Q16_1Block> input(total_heads * BLOCKS_PER_HEAD);
        std::vector<Q16_1Block> output(total_heads * BLOCKS_PER_HEAD);
        std::vector<float> unified_scales(total_heads);

        // Fill with random data
        for (auto &blk : input)
        {
            blk.d = 0.01f;
            blk.sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                blk.qs[i] = static_cast<int16_t>(dist(rng) / blk.d);
                blk.sum_qs += blk.qs[i];
            }
        }

        // Benchmark each variant
        auto bench = [&](auto func) -> double
        {
            auto start = std::chrono::high_resolution_clock::now();
            for (int iter = 0; iter < 10; ++iter)
            {
                for (int h = 0; h < total_heads; ++h)
                {
                    func(input.data() + h * BLOCKS_PER_HEAD,
                         output.data() + h * BLOCKS_PER_HEAD,
                         HEAD_DIM, cos_q15.data(), sin_q15.data(),
                         &unified_scales[h]);
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end - start).count() / 10;
        };

        double scalar_ms = bench([](const Q16_1Block *in, Q16_1Block *out, int hd,
                                    const int16_t *c, const int16_t *s, float *us)
                                 { apply_rope_q16_to_q16_head_dynamic_scale_scalar<Q16_1Block>(in, out, hd, c, s, us); });

        double avx2_ms = -1.0, avx512_ms = -1.0;
#if defined(__AVX2__)
        avx2_ms = bench([](const Q16_1Block *in, Q16_1Block *out, int hd,
                           const int16_t *c, const int16_t *s, float *us)
                        { apply_rope_q16_to_q16_head_dynamic_scale_avx2(in, out, hd, c, s, us); });
#endif
#if defined(__AVX512F__)
        avx512_ms = bench([](const Q16_1Block *in, Q16_1Block *out, int hd,
                             const int16_t *c, const int16_t *s, float *us)
                          { apply_rope_q16_to_q16_head_dynamic_scale_avx512(in, out, hd, c, s, us); });
#endif

        double auto_ms = bench([](const Q16_1Block *in, Q16_1Block *out, int hd,
                                  const int16_t *c, const int16_t *s, float *us)
                               { apply_rope_q16_to_q16_head_dynamic_scale<Q16_1Block>(in, out, hd, c, s, us); });

        double best_simd = scalar_ms;
        if (avx2_ms > 0)
            best_simd = std::min(best_simd, avx2_ms);
        if (avx512_ms > 0)
            best_simd = std::min(best_simd, avx512_ms);
        double speedup = scalar_ms / best_simd;

        std::cout << "║ " << std::setw(8) << total_heads << " │ "
                  << std::fixed << std::setprecision(3)
                  << std::setw(11) << scalar_ms << " │ "
                  << std::setw(9) << (avx2_ms > 0 ? avx2_ms : 0.0) << " │ "
                  << std::setw(11) << (avx512_ms > 0 ? avx512_ms : 0.0) << " │ "
                  << std::setw(9) << auto_ms << " │ "
                  << std::setw(10) << std::setprecision(2) << speedup << "x   ║\n";
    }

    std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
}

// =============================================================================
// Performance Test: Throughput in heads/second
// =============================================================================

TEST_F(Perf__Q16_DynamicScaleRoPE, ThroughputMeasurement)
{
    std::vector<Q16_1Block> input;
    std::vector<Q16_1Block> output;
    std::vector<float> unified_scales;

    generate_spiky_q16_input(input);
    output.resize(input.size());
    unified_scales.resize(TOTAL_HEADS);

    // Warm-up
    for (int i = 0; i < 10; ++i)
    {
        for (size_t h = 0; h < TOTAL_HEADS; ++h)
        {
            apply_rope_q16_to_q16_head_dynamic_scale<Q16_1Block>(
                input.data() + h * BLOCKS_PER_HEAD,
                output.data() + h * BLOCKS_PER_HEAD,
                HEAD_DIM, cos_q15.data(), sin_q15.data(),
                &unified_scales[h]);
        }
    }

    // Measure sustained throughput
    const int DURATION_MS = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    int64_t heads_processed = 0;

    while (true)
    {
        for (size_t h = 0; h < TOTAL_HEADS; ++h)
        {
            apply_rope_q16_to_q16_head_dynamic_scale<Q16_1Block>(
                input.data() + h * BLOCKS_PER_HEAD,
                output.data() + h * BLOCKS_PER_HEAD,
                HEAD_DIM, cos_q15.data(), sin_q15.data(),
                &unified_scales[h]);
            heads_processed++;
        }

        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(now - start).count();
        if (elapsed >= DURATION_MS)
            break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double heads_per_sec = (heads_processed / elapsed_ms) * 1000.0;
    double elements_per_sec = heads_per_sec * HEAD_DIM;

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           Sustained Throughput (Auto-Dispatch, ~1 second run)                ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Heads processed: " << std::setw(12) << heads_processed
              << " in " << std::fixed << std::setprecision(1) << elapsed_ms << " ms"
              << "                            ║\n";
    std::cout << "║ Throughput:      " << std::setw(12) << std::setprecision(0) << heads_per_sec
              << " heads/sec"
              << "                                    ║\n";
    std::cout << "║ Elements:        " << std::setw(12) << std::setprecision(0) << elements_per_sec / 1e6
              << " M elements/sec"
              << "                               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";

    // Sanity check - should process at least 100k heads/sec on modern hardware
    EXPECT_GT(heads_per_sec, 100000.0) << "Throughput seems too low";
}
