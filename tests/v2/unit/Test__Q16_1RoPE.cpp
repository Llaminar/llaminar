/**
 * @file Test__Q16_1RoPE.cpp
 * @brief Unit tests for Q16_1 RoPE vectorized implementations
 * @author David Sanftenberg
 *
 * Tests:
 * 1. Correctness: Q16_1 RoPE (scalar/AVX2/AVX512) vs FP32 reference
 * 2. Implementation parity: AVX512 vs AVX2 vs Scalar produce identical Q16_1 results
 * 3. Performance: Expected speedups (AVX512 ~2x AVX2, AVX2 ~8x scalar)
 */

#include "kernels/cpu/primitives/RoPEPrimitives.h"
#include "tensors/BlockStructures.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>

using namespace llaminar2::primitives;
using namespace llaminar2;

namespace
{
    // ============================================================================
    // Test Utilities
    // ============================================================================

    /**
     * @brief Quantize FP32 data to Q16_1 blocks
     */
    std::vector<Q16_1Block> fp32_to_q16_1(const std::vector<float> &fp32)
    {
        const size_t n_blocks = (fp32.size() + 31) / 32;
        std::vector<Q16_1Block> blocks(n_blocks);

        std::vector<float> padded = fp32;
        padded.resize(n_blocks * 32, 0.0f);

        for (size_t b = 0; b < n_blocks; ++b)
        {
            const float *block_data = padded.data() + b * 32;
            Q16_1Block &blk = blocks[b];

            float max_abs = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                max_abs = std::max(max_abs, std::fabs(block_data[i]));
            }

            float scale = max_abs / 32767.0f;
            if (scale < 1e-20f)
                scale = 1e-20f;
            float inv_scale = 1.0f / scale;

            int32_t sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(block_data[i] * inv_scale));
                q = std::max(-32767, std::min(32767, q));
                blk.qs[i] = static_cast<int16_t>(q);
                sum_qs += q;
            }

            blk.d = scale;
            blk.sum_qs = sum_qs;
        }

        return blocks;
    }

    /**
     * @brief Dequantize Q16_1 blocks to FP32
     */
    std::vector<float> q16_1_to_fp32(const std::vector<Q16_1Block> &blocks)
    {
        std::vector<float> fp32(blocks.size() * 32);

        for (size_t b = 0; b < blocks.size(); ++b)
        {
            const Q16_1Block &blk = blocks[b];
            float *block_data = fp32.data() + b * 32;

            for (int i = 0; i < 32; ++i)
            {
                block_data[i] = blk.d * static_cast<float>(blk.qs[i]);
            }
        }

        return fp32;
    }

    /**
     * @brief Apply FP32 RoPE to head data (reference implementation)
     */
    void apply_rope_fp32_reference(
        float *head_data,
        int head_dim,
        const float *cos_fp32,
        const float *sin_fp32)
    {
        const int half_dim = head_dim / 2;
        for (int i = 0; i < half_dim; ++i)
        {
            float x = head_data[i];
            float y = head_data[i + half_dim];
            float c = cos_fp32[i];
            float s = sin_fp32[i];
            head_data[i] = x * c - y * s;
            head_data[i + half_dim] = x * s + y * c;
        }
    }

    /**
     * @brief Generate sin/cos tables in both Q15 and FP32 formats
     */
    void generate_rope_tables(
        int half_dim,
        int position,
        float rope_theta,
        std::vector<int16_t> &cos_q15,
        std::vector<int16_t> &sin_q15,
        std::vector<float> &cos_fp32,
        std::vector<float> &sin_fp32)
    {
        cos_q15.resize(half_dim);
        sin_q15.resize(half_dim);
        cos_fp32.resize(half_dim);
        sin_fp32.resize(half_dim);

        for (int i = 0; i < half_dim; ++i)
        {
            float freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / (half_dim * 2));
            float angle = static_cast<float>(position) * freq;
            float c = std::cos(angle);
            float s = std::sin(angle);

            cos_fp32[i] = c;
            sin_fp32[i] = s;
            cos_q15[i] = static_cast<int16_t>(std::round(c * 32767.0f));
            sin_q15[i] = static_cast<int16_t>(std::round(s * 32767.0f));
        }
    }

    /**
     * @brief Compare Q16_1 blocks for exact equality
     */
    bool q16_1_blocks_equal(const Q16_1Block &a, const Q16_1Block &b)
    {
        if (std::fabs(a.d - b.d) > 1e-10f)
            return false;
        if (a.sum_qs != b.sum_qs)
            return false;
        for (int i = 0; i < 32; ++i)
        {
            if (a.qs[i] != b.qs[i])
                return false;
        }
        return true;
    }

    /**
     * @brief Compute cosine similarity between two vectors
     */
    float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size() || a.empty())
            return 0.0f;

        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if (norm_a < 1e-10f || norm_b < 1e-10f)
            return 1.0f;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    /**
     * @brief Compute max absolute error
     */
    float max_abs_error(const std::vector<float> &a, const std::vector<float> &b)
    {
        float max_err = 0.0f;
        for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
        {
            max_err = std::max(max_err, std::fabs(a[i] - b[i]));
        }
        return max_err;
    }

} // anonymous namespace

// ============================================================================
// Test Fixture
// ============================================================================

class Test__Q16_1RoPE : public ::testing::Test
{
protected:
    void SetUp() override
    {
#if defined(__AVX512F__)
        has_avx512_ = true;
        has_avx2_ = true;
#elif defined(__AVX2__)
        has_avx2_ = true;
#endif
    }

    bool has_avx2_ = false;
    bool has_avx512_ = false;

    // Common test parameters
    static constexpr int HEAD_DIM_64 = 64;
    static constexpr int HEAD_DIM_128 = 128;
    static constexpr float ROPE_THETA = 10000.0f;
    static constexpr int POSITION = 42;
};

// ============================================================================
// Correctness Tests: Q16_1 RoPE vs FP32 Reference
// ============================================================================

TEST_F(Test__Q16_1RoPE, Scalar_Correctness_VsFP32Reference)
{
    const int head_dim = HEAD_DIM_64;
    const int blocks_per_head = head_dim / 32;
    const int half_dim = head_dim / 2;

    // Generate random FP32 head data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> fp32_data(head_dim);
    for (auto &v : fp32_data)
        v = dist(rng);

    // Generate sin/cos tables
    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(half_dim, POSITION, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // Path 1: FP32 reference RoPE
    std::vector<float> fp32_reference = fp32_data;
    apply_rope_fp32_reference(fp32_reference.data(), head_dim, cos_fp32.data(), sin_fp32.data());

    // Path 2: Q16_1 scalar RoPE
    auto q16_blocks = fp32_to_q16_1(fp32_data);
    apply_rope_q16_1_integer_head_scalar(q16_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());
    auto q16_result = q16_1_to_fp32(q16_blocks);
    q16_result.resize(head_dim);

    // Compare
    float sim = cosine_similarity(q16_result, fp32_reference);
    float max_err = max_abs_error(q16_result, fp32_reference);

    EXPECT_GT(sim, 0.9999f) << "Q16_1 scalar should match FP32 reference closely";
    EXPECT_LT(max_err, 1e-3f) << "Q16_1 scalar max error too high: " << max_err;
}

#if defined(__AVX2__)
TEST_F(Test__Q16_1RoPE, AVX2_Correctness_VsFP32Reference)
{
    const int head_dim = HEAD_DIM_64;
    const int blocks_per_head = head_dim / 32;
    const int half_dim = head_dim / 2;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> fp32_data(head_dim);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(half_dim, POSITION, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // FP32 reference
    std::vector<float> fp32_reference = fp32_data;
    apply_rope_fp32_reference(fp32_reference.data(), head_dim, cos_fp32.data(), sin_fp32.data());

    // Q16_1 AVX2
    auto q16_blocks = fp32_to_q16_1(fp32_data);
    apply_rope_q16_1_integer_head_avx2(q16_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());
    auto q16_result = q16_1_to_fp32(q16_blocks);
    q16_result.resize(head_dim);

    float sim = cosine_similarity(q16_result, fp32_reference);
    float max_err = max_abs_error(q16_result, fp32_reference);

    EXPECT_GT(sim, 0.9999f) << "Q16_1 AVX2 should match FP32 reference closely";
    EXPECT_LT(max_err, 1e-3f) << "Q16_1 AVX2 max error too high: " << max_err;
}
#endif

#if defined(__AVX512F__)
TEST_F(Test__Q16_1RoPE, AVX512_Correctness_VsFP32Reference)
{
    const int head_dim = HEAD_DIM_64;
    const int blocks_per_head = head_dim / 32;
    const int half_dim = head_dim / 2;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> fp32_data(head_dim);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(half_dim, POSITION, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // FP32 reference
    std::vector<float> fp32_reference = fp32_data;
    apply_rope_fp32_reference(fp32_reference.data(), head_dim, cos_fp32.data(), sin_fp32.data());

    // Q16_1 AVX512
    auto q16_blocks = fp32_to_q16_1(fp32_data);
    apply_rope_q16_1_integer_head_avx512(q16_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());
    auto q16_result = q16_1_to_fp32(q16_blocks);
    q16_result.resize(head_dim);

    float sim = cosine_similarity(q16_result, fp32_reference);
    float max_err = max_abs_error(q16_result, fp32_reference);

    EXPECT_GT(sim, 0.9999f) << "Q16_1 AVX512 should match FP32 reference closely";
    EXPECT_LT(max_err, 1e-3f) << "Q16_1 AVX512 max error too high: " << max_err;
}
#endif

// ============================================================================
// Implementation Parity: SIMD vs Scalar produce identical Q16_1 blocks
// ============================================================================

#if defined(__AVX2__)
TEST_F(Test__Q16_1RoPE, AVX2_Parity_VsScalar)
{
    const int head_dim = HEAD_DIM_128;
    const int blocks_per_head = head_dim / 32;
    const int half_dim = head_dim / 2;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> fp32_data(head_dim);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(half_dim, POSITION, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // Scalar path
    auto scalar_blocks = fp32_to_q16_1(fp32_data);
    apply_rope_q16_1_integer_head_scalar(scalar_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());

    // AVX2 path
    auto avx2_blocks = fp32_to_q16_1(fp32_data);
    apply_rope_q16_1_integer_head_avx2(avx2_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());

    // Compare block-by-block
    for (int b = 0; b < blocks_per_head; ++b)
    {
        EXPECT_TRUE(q16_1_blocks_equal(avx2_blocks[b], scalar_blocks[b]))
            << "AVX2 block " << b << " differs from scalar";
    }
}
#endif

#if defined(__AVX512F__)
TEST_F(Test__Q16_1RoPE, AVX512_Parity_VsScalar)
{
    const int head_dim = HEAD_DIM_128;
    const int blocks_per_head = head_dim / 32;
    const int half_dim = head_dim / 2;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> fp32_data(head_dim);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(half_dim, POSITION, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // Scalar path
    auto scalar_blocks = fp32_to_q16_1(fp32_data);
    apply_rope_q16_1_integer_head_scalar(scalar_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());

    // AVX512 path
    auto avx512_blocks = fp32_to_q16_1(fp32_data);
    apply_rope_q16_1_integer_head_avx512(avx512_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());

    // Compare block-by-block
    for (int b = 0; b < blocks_per_head; ++b)
    {
        EXPECT_TRUE(q16_1_blocks_equal(avx512_blocks[b], scalar_blocks[b]))
            << "AVX512 block " << b << " differs from scalar";
    }
}

TEST_F(Test__Q16_1RoPE, AVX512_Parity_VsAVX2)
{
    const int head_dim = HEAD_DIM_128;
    const int blocks_per_head = head_dim / 32;
    const int half_dim = head_dim / 2;

    std::mt19937 rng(456);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    std::vector<float> fp32_data(head_dim);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(half_dim, POSITION, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // AVX2 path
    auto avx2_blocks = fp32_to_q16_1(fp32_data);
    apply_rope_q16_1_integer_head_avx2(avx2_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());

    // AVX512 path
    auto avx512_blocks = fp32_to_q16_1(fp32_data);
    apply_rope_q16_1_integer_head_avx512(avx512_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());

    // Compare block-by-block
    for (int b = 0; b < blocks_per_head; ++b)
    {
        EXPECT_TRUE(q16_1_blocks_equal(avx512_blocks[b], avx2_blocks[b]))
            << "AVX512 block " << b << " differs from AVX2";
    }
}
#endif

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Test__Q16_1RoPE, Correctness_LargeHeadDim)
{
    // Test with head_dim=128 (common in larger models)
    const int head_dim = HEAD_DIM_128;
    const int blocks_per_head = head_dim / 32;
    const int half_dim = head_dim / 2;

    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> fp32_data(head_dim);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(half_dim, POSITION, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // FP32 reference
    std::vector<float> fp32_reference = fp32_data;
    apply_rope_fp32_reference(fp32_reference.data(), head_dim, cos_fp32.data(), sin_fp32.data());

    // Q16_1 (uses best SIMD automatically)
    auto q16_blocks = fp32_to_q16_1(fp32_data);
    apply_rope_q16_1_integer_head(q16_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());
    auto q16_result = q16_1_to_fp32(q16_blocks);
    q16_result.resize(head_dim);

    float sim = cosine_similarity(q16_result, fp32_reference);
    EXPECT_GT(sim, 0.9999f) << "Q16_1 RoPE with head_dim=128 should match FP32";
}

TEST_F(Test__Q16_1RoPE, Correctness_HighPosition)
{
    // Test with high position (typical in long context)
    const int head_dim = HEAD_DIM_64;
    const int blocks_per_head = head_dim / 32;
    const int half_dim = head_dim / 2;
    const int high_position = 4096;

    std::mt19937 rng(111);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> fp32_data(head_dim);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(half_dim, high_position, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // FP32 reference
    std::vector<float> fp32_reference = fp32_data;
    apply_rope_fp32_reference(fp32_reference.data(), head_dim, cos_fp32.data(), sin_fp32.data());

    // Q16_1
    auto q16_blocks = fp32_to_q16_1(fp32_data);
    apply_rope_q16_1_integer_head(q16_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());
    auto q16_result = q16_1_to_fp32(q16_blocks);
    q16_result.resize(head_dim);

    float sim = cosine_similarity(q16_result, fp32_reference);
    EXPECT_GT(sim, 0.9999f) << "Q16_1 RoPE at position " << high_position << " should match FP32";
}

TEST_F(Test__Q16_1RoPE, Correctness_SmallValues)
{
    // Test with small values (challenging for quantization)
    const int head_dim = HEAD_DIM_64;
    const int blocks_per_head = head_dim / 32;
    const int half_dim = head_dim / 2;

    std::mt19937 rng(222);
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
    std::vector<float> fp32_data(head_dim);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(half_dim, POSITION, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // FP32 reference
    std::vector<float> fp32_reference = fp32_data;
    apply_rope_fp32_reference(fp32_reference.data(), head_dim, cos_fp32.data(), sin_fp32.data());

    // Q16_1
    auto q16_blocks = fp32_to_q16_1(fp32_data);
    apply_rope_q16_1_integer_head(q16_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());
    auto q16_result = q16_1_to_fp32(q16_blocks);
    q16_result.resize(head_dim);

    float sim = cosine_similarity(q16_result, fp32_reference);
    // Slightly looser threshold for small values due to quantization noise
    EXPECT_GT(sim, 0.999f) << "Q16_1 RoPE with small values should match FP32";
}

// ============================================================================
// Performance Tests
// ============================================================================

class Test__Q16_1RoPE_Performance : public Test__Q16_1RoPE
{
protected:
    // Benchmark parameters
    static constexpr int WARMUP_ITERATIONS = 100;
    static constexpr int BENCHMARK_ITERATIONS = 1000;
    static constexpr int N_HEADS = 16;
    static constexpr int HEAD_DIM = 64;
    static constexpr int BLOCKS_PER_HEAD = HEAD_DIM / 32;

    /**
     * @brief Benchmark a RoPE implementation
     * @return Time per head in nanoseconds
     */
    template <typename Func>
    double benchmark_rope(Func func, int n_iterations)
    {
        const int half_dim = HEAD_DIM / 2;

        // Generate test data
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> fp32_data(N_HEADS * HEAD_DIM);
        for (auto &v : fp32_data)
            v = dist(rng);

        std::vector<Q16_1Block> q16_blocks;
        for (int h = 0; h < N_HEADS; ++h)
        {
            std::vector<float> head_data(fp32_data.begin() + h * HEAD_DIM,
                                         fp32_data.begin() + (h + 1) * HEAD_DIM);
            auto blocks = fp32_to_q16_1(head_data);
            q16_blocks.insert(q16_blocks.end(), blocks.begin(), blocks.end());
        }

        std::vector<int16_t> cos_q15, sin_q15;
        std::vector<float> cos_fp32, sin_fp32;
        generate_rope_tables(half_dim, POSITION, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; ++i)
        {
            for (int h = 0; h < N_HEADS; ++h)
            {
                func(q16_blocks.data() + h * BLOCKS_PER_HEAD, BLOCKS_PER_HEAD,
                     cos_q15.data(), sin_q15.data());
            }
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_iterations; ++i)
        {
            for (int h = 0; h < N_HEADS; ++h)
            {
                func(q16_blocks.data() + h * BLOCKS_PER_HEAD, BLOCKS_PER_HEAD,
                     cos_q15.data(), sin_q15.data());
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        return static_cast<double>(duration_ns) / (n_iterations * N_HEADS);
    }
};

TEST_F(Test__Q16_1RoPE_Performance, Scalar_Baseline)
{
    double ns_per_head = benchmark_rope(apply_rope_q16_1_integer_head_scalar, BENCHMARK_ITERATIONS);
    printf("[Performance] Scalar: %.1f ns/head (%.1f heads/us)\n",
           ns_per_head, 1000.0 / ns_per_head);

    // Informational only - timing can vary significantly under parallel test execution
    // Expected ~5000ns in isolation, but can be 20-50x slower under load
    if (ns_per_head >= 200000.0)
    {
        printf("[WARNING] Scalar RoPE slower than expected (%.1f ns > 200000 ns), "
               "likely due to parallel test execution\n",
               ns_per_head);
    }
}

#if defined(__AVX2__)
TEST_F(Test__Q16_1RoPE_Performance, AVX2_Speedup_VsScalar)
{
    double scalar_ns = benchmark_rope(apply_rope_q16_1_integer_head_scalar, BENCHMARK_ITERATIONS);
    double avx2_ns = benchmark_rope(apply_rope_q16_1_integer_head_avx2, BENCHMARK_ITERATIONS);

    double speedup = scalar_ns / avx2_ns;
    printf("[Performance] AVX2 vs Scalar: %.1fx speedup (scalar=%.1f ns, avx2=%.1f ns)\n",
           speedup, scalar_ns, avx2_ns);

    // AVX2 processes 8 elements/iteration vs 1 for scalar
    // However, scalar benefits from register renaming and ILP
    // The rotation involves: dequant → rotate pair → requant with scale update
    // With aggressive unrolling and interleaving, we achieve ~5-8x speedup in isolation
    // Under parallel test execution, resource contention can affect relative speedup
    if (speedup < 5.0 || speedup > 12.0)
    {
        printf("[WARNING] AVX2 vs Scalar speedup (%.1fx) outside expected range [5.0, 12.0], "
               "likely due to parallel test execution\n",
               speedup);
    }
}
#endif

#if defined(__AVX512F__)
TEST_F(Test__Q16_1RoPE_Performance, AVX512_Speedup_VsAVX2)
{
    double avx2_ns = benchmark_rope(apply_rope_q16_1_integer_head_avx2, BENCHMARK_ITERATIONS);
    double avx512_ns = benchmark_rope(apply_rope_q16_1_integer_head_avx512, BENCHMARK_ITERATIONS);

    double speedup = avx2_ns / avx512_ns;
    printf("[Performance] AVX512 vs AVX2: %.2fx speedup (avx2=%.1f ns, avx512=%.1f ns)\n",
           speedup, avx2_ns, avx512_ns);

    // AVX512 processes 16 elements/iteration vs 8 for AVX2
    // With interleaved processing of block pairs, we hide latency and achieve ~1.2x speedup
    // over the highly optimized AVX2 version.
    // Under parallel test execution, resource contention can affect relative speedup
    if (speedup < 1.15 || speedup > 3.0)
    {
        printf("[WARNING] AVX512 vs AVX2 speedup (%.2fx) outside expected range [1.15, 3.0], "
               "likely due to parallel test execution\n",
               speedup);
    }
}

TEST_F(Test__Q16_1RoPE_Performance, AVX512_Speedup_VsScalar)
{
    double scalar_ns = benchmark_rope(apply_rope_q16_1_integer_head_scalar, BENCHMARK_ITERATIONS);
    double avx512_ns = benchmark_rope(apply_rope_q16_1_integer_head_avx512, BENCHMARK_ITERATIONS);

    double speedup = scalar_ns / avx512_ns;
    printf("[Performance] AVX512 vs Scalar: %.1fx speedup (scalar=%.1f ns, avx512=%.1f ns)\n",
           speedup, scalar_ns, avx512_ns);

    // AVX512 processes 16 elements/iteration vs 1 for scalar
    // With optimizations, we achieve ~9-10x speedup in isolation
    // Under parallel test execution, resource contention can reduce observed speedup
    if (speedup < 8.0)
    {
        printf("[WARNING] AVX512 speedup (%.1fx) below expected 8x, likely due to parallel test execution\n", speedup);
    }
}
#endif

// ============================================================================
// Variable Block Size Tests (Q16_1Block_64, Q16_1Block_128)
// ============================================================================

namespace
{
    /**
     * @brief Quantize FP32 data to Q16_1Block_64
     */
    std::vector<Q16_1Block_64> fp32_to_q16_1_block64(const std::vector<float> &fp32)
    {
        constexpr size_t BLOCK_SIZE = Q16_1Block_64::BLOCK_SIZE;
        const size_t n_blocks = (fp32.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::vector<Q16_1Block_64> blocks(n_blocks);

        std::vector<float> padded = fp32;
        padded.resize(n_blocks * BLOCK_SIZE, 0.0f);

        for (size_t b = 0; b < n_blocks; ++b)
        {
            const float *block_data = padded.data() + b * BLOCK_SIZE;
            Q16_1Block_64 &blk = blocks[b];

            float max_abs = 0.0f;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                max_abs = std::max(max_abs, std::fabs(block_data[i]));
            }

            float scale = max_abs / 32767.0f;
            if (scale < 1e-20f)
                scale = 1e-20f;
            float inv_scale = 1.0f / scale;

            int32_t sum_qs = 0;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(block_data[i] * inv_scale));
                q = std::max(-32767, std::min(32767, q));
                blk.qs[i] = static_cast<int16_t>(q);
                sum_qs += q;
            }

            blk.d = scale;
            blk.sum_qs = sum_qs;
        }

        return blocks;
    }

    /**
     * @brief Dequantize Q16_1Block_64 blocks to FP32
     */
    std::vector<float> q16_1_block64_to_fp32(const std::vector<Q16_1Block_64> &blocks)
    {
        constexpr size_t BLOCK_SIZE = Q16_1Block_64::BLOCK_SIZE;
        std::vector<float> fp32(blocks.size() * BLOCK_SIZE);

        for (size_t b = 0; b < blocks.size(); ++b)
        {
            const Q16_1Block_64 &blk = blocks[b];
            float *block_data = fp32.data() + b * BLOCK_SIZE;

            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                block_data[i] = blk.d * static_cast<float>(blk.qs[i]);
            }
        }

        return fp32;
    }

    /**
     * @brief Quantize FP32 data to Q16_1Block_128
     */
    std::vector<Q16_1Block_128> fp32_to_q16_1_block128(const std::vector<float> &fp32)
    {
        constexpr size_t BLOCK_SIZE = Q16_1Block_128::BLOCK_SIZE;
        const size_t n_blocks = (fp32.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::vector<Q16_1Block_128> blocks(n_blocks);

        std::vector<float> padded = fp32;
        padded.resize(n_blocks * BLOCK_SIZE, 0.0f);

        for (size_t b = 0; b < n_blocks; ++b)
        {
            const float *block_data = padded.data() + b * BLOCK_SIZE;
            Q16_1Block_128 &blk = blocks[b];

            float max_abs = 0.0f;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                max_abs = std::max(max_abs, std::fabs(block_data[i]));
            }

            float scale = max_abs / 32767.0f;
            if (scale < 1e-20f)
                scale = 1e-20f;
            float inv_scale = 1.0f / scale;

            int32_t sum_qs = 0;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(block_data[i] * inv_scale));
                q = std::max(-32767, std::min(32767, q));
                blk.qs[i] = static_cast<int16_t>(q);
                sum_qs += q;
            }

            blk.d = scale;
            blk.sum_qs = sum_qs;
        }

        return blocks;
    }

    /**
     * @brief Dequantize Q16_1Block_128 blocks to FP32
     */
    std::vector<float> q16_1_block128_to_fp32(const std::vector<Q16_1Block_128> &blocks)
    {
        constexpr size_t BLOCK_SIZE = Q16_1Block_128::BLOCK_SIZE;
        std::vector<float> fp32(blocks.size() * BLOCK_SIZE);

        for (size_t b = 0; b < blocks.size(); ++b)
        {
            const Q16_1Block_128 &blk = blocks[b];
            float *block_data = fp32.data() + b * BLOCK_SIZE;

            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                block_data[i] = blk.d * static_cast<float>(blk.qs[i]);
            }
        }

        return fp32;
    }

    // Note: DeepSeek V3 MLA uses separate NOPE (128-dim) + ROPE (64-dim) tensors
    // with independent scales, not a combined 192-dim block.
    // See PROJECT_Q16_INTEGER_ATTENTION_V2.md MLA Architecture section.

} // anonymous namespace

class Test__Q16_VariableBlockRoPE : public ::testing::Test
{
protected:
    void SetUp() override
    {
#if defined(__AVX512F__)
        has_avx512_ = true;
        has_avx2_ = true;
#elif defined(__AVX2__)
        has_avx2_ = true;
#endif
    }

    bool has_avx2_ = false;
    bool has_avx512_ = false;

    static constexpr float ROPE_THETA = 10000.0f;
    static constexpr int POSITION = 42;
};

// ============================================================================
// Block Size 64 Tests (head_dim=64, 1 block per head, Qwen2.5-0.5B)
// ============================================================================

TEST_F(Test__Q16_VariableBlockRoPE, Block64_Scalar_Correctness)
{
    constexpr int HEAD_DIM = 64;
    constexpr int BLOCK_SIZE = 64;
    constexpr int BLOCKS_PER_HEAD = HEAD_DIM / BLOCK_SIZE; // 1 block
    const int half_dim = HEAD_DIM / 2;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> fp32_data(HEAD_DIM);
    for (auto &v : fp32_data)
        v = dist(rng);

    // Generate sin/cos tables
    std::vector<int16_t> cos_q15(half_dim), sin_q15(half_dim);
    std::vector<float> cos_fp32(half_dim), sin_fp32(half_dim);
    for (int i = 0; i < half_dim; ++i)
    {
        float freq = 1.0f / std::pow(ROPE_THETA, static_cast<float>(2 * i) / HEAD_DIM);
        float angle = static_cast<float>(POSITION) * freq;
        cos_fp32[i] = std::cos(angle);
        sin_fp32[i] = std::sin(angle);
        cos_q15[i] = static_cast<int16_t>(std::round(cos_fp32[i] * 32767.0f));
        sin_q15[i] = static_cast<int16_t>(std::round(sin_fp32[i] * 32767.0f));
    }

    // FP32 reference
    std::vector<float> fp32_reference = fp32_data;
    apply_rope_fp32_reference(fp32_reference.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    // Q16 Block64 scalar
    auto q16_blocks = fp32_to_q16_1_block64(fp32_data);
    apply_rope_q16_integer_head_scalar<Q16_1Block_64>(q16_blocks.data(), BLOCKS_PER_HEAD, cos_q15.data(), sin_q15.data());
    auto q16_result = q16_1_block64_to_fp32(q16_blocks);
    q16_result.resize(HEAD_DIM);

    float sim = cosine_similarity(q16_result, fp32_reference);
    float max_err = max_abs_error(q16_result, fp32_reference);

    EXPECT_GT(sim, 0.9999f) << "Block64 scalar should match FP32 reference closely";
    EXPECT_LT(max_err, 1e-3f) << "Block64 scalar max error too high: " << max_err;
}

#if defined(__AVX2__)
TEST_F(Test__Q16_VariableBlockRoPE, Block64_AVX2_Correctness)
{
    constexpr int HEAD_DIM = 64;
    constexpr int BLOCK_SIZE = 64;
    constexpr int BLOCKS_PER_HEAD = HEAD_DIM / BLOCK_SIZE;
    const int half_dim = HEAD_DIM / 2;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> fp32_data(HEAD_DIM);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15(half_dim), sin_q15(half_dim);
    std::vector<float> cos_fp32(half_dim), sin_fp32(half_dim);
    for (int i = 0; i < half_dim; ++i)
    {
        float freq = 1.0f / std::pow(ROPE_THETA, static_cast<float>(2 * i) / HEAD_DIM);
        float angle = static_cast<float>(POSITION) * freq;
        cos_fp32[i] = std::cos(angle);
        sin_fp32[i] = std::sin(angle);
        cos_q15[i] = static_cast<int16_t>(std::round(cos_fp32[i] * 32767.0f));
        sin_q15[i] = static_cast<int16_t>(std::round(sin_fp32[i] * 32767.0f));
    }

    std::vector<float> fp32_reference = fp32_data;
    apply_rope_fp32_reference(fp32_reference.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    auto q16_blocks = fp32_to_q16_1_block64(fp32_data);
    apply_rope_q16_integer_head_avx2<Q16_1Block_64>(q16_blocks.data(), BLOCKS_PER_HEAD, cos_q15.data(), sin_q15.data());
    auto q16_result = q16_1_block64_to_fp32(q16_blocks);
    q16_result.resize(HEAD_DIM);

    float sim = cosine_similarity(q16_result, fp32_reference);
    float max_err = max_abs_error(q16_result, fp32_reference);

    EXPECT_GT(sim, 0.9999f) << "Block64 AVX2 should match FP32 reference closely";
    EXPECT_LT(max_err, 1e-3f) << "Block64 AVX2 max error too high: " << max_err;
}

TEST_F(Test__Q16_VariableBlockRoPE, Block64_AVX2_Parity_VsScalar)
{
    constexpr int HEAD_DIM = 64;
    constexpr int BLOCKS_PER_HEAD = 1;
    const int half_dim = HEAD_DIM / 2;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> fp32_data(HEAD_DIM);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15(half_dim), sin_q15(half_dim);
    for (int i = 0; i < half_dim; ++i)
    {
        float freq = 1.0f / std::pow(ROPE_THETA, static_cast<float>(2 * i) / HEAD_DIM);
        float angle = static_cast<float>(POSITION) * freq;
        cos_q15[i] = static_cast<int16_t>(std::round(std::cos(angle) * 32767.0f));
        sin_q15[i] = static_cast<int16_t>(std::round(std::sin(angle) * 32767.0f));
    }

    auto scalar_blocks = fp32_to_q16_1_block64(fp32_data);
    auto avx2_blocks = fp32_to_q16_1_block64(fp32_data);

    apply_rope_q16_integer_head_scalar<Q16_1Block_64>(scalar_blocks.data(), BLOCKS_PER_HEAD, cos_q15.data(), sin_q15.data());
    apply_rope_q16_integer_head_avx2<Q16_1Block_64>(avx2_blocks.data(), BLOCKS_PER_HEAD, cos_q15.data(), sin_q15.data());

    // Compare blocks
    ASSERT_EQ(scalar_blocks.size(), avx2_blocks.size());
    for (size_t b = 0; b < scalar_blocks.size(); ++b)
    {
        EXPECT_FLOAT_EQ(scalar_blocks[b].d, avx2_blocks[b].d) << "Block " << b << " scale mismatch";
        EXPECT_EQ(scalar_blocks[b].sum_qs, avx2_blocks[b].sum_qs) << "Block " << b << " sum_qs mismatch";
        for (int i = 0; i < 64; ++i)
        {
            EXPECT_EQ(scalar_blocks[b].qs[i], avx2_blocks[b].qs[i])
                << "Block " << b << " element " << i << " mismatch";
        }
    }
}
#endif

// ============================================================================
// Block Size 128 Tests (head_dim=128, 1 block per head, Llama-3, Qwen3)
// ============================================================================

TEST_F(Test__Q16_VariableBlockRoPE, Block128_Scalar_Correctness)
{
    constexpr int HEAD_DIM = 128;
    constexpr int BLOCK_SIZE = 128;
    constexpr int BLOCKS_PER_HEAD = HEAD_DIM / BLOCK_SIZE;
    const int half_dim = HEAD_DIM / 2;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> fp32_data(HEAD_DIM);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15(half_dim), sin_q15(half_dim);
    std::vector<float> cos_fp32(half_dim), sin_fp32(half_dim);
    for (int i = 0; i < half_dim; ++i)
    {
        float freq = 1.0f / std::pow(ROPE_THETA, static_cast<float>(2 * i) / HEAD_DIM);
        float angle = static_cast<float>(POSITION) * freq;
        cos_fp32[i] = std::cos(angle);
        sin_fp32[i] = std::sin(angle);
        cos_q15[i] = static_cast<int16_t>(std::round(cos_fp32[i] * 32767.0f));
        sin_q15[i] = static_cast<int16_t>(std::round(sin_fp32[i] * 32767.0f));
    }

    std::vector<float> fp32_reference = fp32_data;
    apply_rope_fp32_reference(fp32_reference.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    auto q16_blocks = fp32_to_q16_1_block128(fp32_data);
    apply_rope_q16_integer_head_scalar<Q16_1Block_128>(q16_blocks.data(), BLOCKS_PER_HEAD, cos_q15.data(), sin_q15.data());
    auto q16_result = q16_1_block128_to_fp32(q16_blocks);
    q16_result.resize(HEAD_DIM);

    float sim = cosine_similarity(q16_result, fp32_reference);
    float max_err = max_abs_error(q16_result, fp32_reference);

    EXPECT_GT(sim, 0.9999f) << "Block128 scalar should match FP32 reference closely";
    EXPECT_LT(max_err, 1e-3f) << "Block128 scalar max error too high: " << max_err;
}

#if defined(__AVX2__)
TEST_F(Test__Q16_VariableBlockRoPE, Block128_AVX2_Correctness)
{
    constexpr int HEAD_DIM = 128;
    constexpr int BLOCKS_PER_HEAD = 1;
    const int half_dim = HEAD_DIM / 2;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> fp32_data(HEAD_DIM);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15(half_dim), sin_q15(half_dim);
    std::vector<float> cos_fp32(half_dim), sin_fp32(half_dim);
    for (int i = 0; i < half_dim; ++i)
    {
        float freq = 1.0f / std::pow(ROPE_THETA, static_cast<float>(2 * i) / HEAD_DIM);
        float angle = static_cast<float>(POSITION) * freq;
        cos_fp32[i] = std::cos(angle);
        sin_fp32[i] = std::sin(angle);
        cos_q15[i] = static_cast<int16_t>(std::round(cos_fp32[i] * 32767.0f));
        sin_q15[i] = static_cast<int16_t>(std::round(sin_fp32[i] * 32767.0f));
    }

    std::vector<float> fp32_reference = fp32_data;
    apply_rope_fp32_reference(fp32_reference.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    auto q16_blocks = fp32_to_q16_1_block128(fp32_data);
    apply_rope_q16_integer_head_avx2<Q16_1Block_128>(q16_blocks.data(), BLOCKS_PER_HEAD, cos_q15.data(), sin_q15.data());
    auto q16_result = q16_1_block128_to_fp32(q16_blocks);
    q16_result.resize(HEAD_DIM);

    float sim = cosine_similarity(q16_result, fp32_reference);
    float max_err = max_abs_error(q16_result, fp32_reference);

    EXPECT_GT(sim, 0.9999f) << "Block128 AVX2 should match FP32 reference closely";
    EXPECT_LT(max_err, 1e-3f) << "Block128 AVX2 max error too high: " << max_err;
}
#endif

#if defined(__AVX512F__)
TEST_F(Test__Q16_VariableBlockRoPE, Block128_AVX512_Correctness)
{
    constexpr int HEAD_DIM = 128;
    constexpr int BLOCKS_PER_HEAD = 1;
    const int half_dim = HEAD_DIM / 2;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> fp32_data(HEAD_DIM);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15(half_dim), sin_q15(half_dim);
    std::vector<float> cos_fp32(half_dim), sin_fp32(half_dim);
    for (int i = 0; i < half_dim; ++i)
    {
        float freq = 1.0f / std::pow(ROPE_THETA, static_cast<float>(2 * i) / HEAD_DIM);
        float angle = static_cast<float>(POSITION) * freq;
        cos_fp32[i] = std::cos(angle);
        sin_fp32[i] = std::sin(angle);
        cos_q15[i] = static_cast<int16_t>(std::round(cos_fp32[i] * 32767.0f));
        sin_q15[i] = static_cast<int16_t>(std::round(sin_fp32[i] * 32767.0f));
    }

    std::vector<float> fp32_reference = fp32_data;
    apply_rope_fp32_reference(fp32_reference.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    auto q16_blocks = fp32_to_q16_1_block128(fp32_data);
    apply_rope_q16_integer_head_avx512<Q16_1Block_128>(q16_blocks.data(), BLOCKS_PER_HEAD, cos_q15.data(), sin_q15.data());
    auto q16_result = q16_1_block128_to_fp32(q16_blocks);
    q16_result.resize(HEAD_DIM);

    float sim = cosine_similarity(q16_result, fp32_reference);
    float max_err = max_abs_error(q16_result, fp32_reference);

    EXPECT_GT(sim, 0.9999f) << "Block128 AVX512 should match FP32 reference closely";
    EXPECT_LT(max_err, 1e-3f) << "Block128 AVX512 max error too high: " << max_err;
}
#endif

// Note: DeepSeek V3 MLA uses separate NOPE (128-dim) + ROPE (64-dim) tensors
// with independent scales, not a combined 192-dim block.
// See PROJECT_Q16_INTEGER_ATTENTION_V2.md MLA Architecture section.

// ============================================================================
// Dispatch Function Tests
// ============================================================================

TEST_F(Test__Q16_VariableBlockRoPE, Dispatch_Block32_Matches_Legacy)
{
    constexpr int HEAD_DIM = 64;
    constexpr int BLOCKS_PER_HEAD = HEAD_DIM / 32; // 2 blocks
    const int half_dim = HEAD_DIM / 2;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> fp32_data(HEAD_DIM);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15(half_dim), sin_q15(half_dim);
    for (int i = 0; i < half_dim; ++i)
    {
        float freq = 1.0f / std::pow(ROPE_THETA, static_cast<float>(2 * i) / HEAD_DIM);
        float angle = static_cast<float>(POSITION) * freq;
        cos_q15[i] = static_cast<int16_t>(std::round(std::cos(angle) * 32767.0f));
        sin_q15[i] = static_cast<int16_t>(std::round(std::sin(angle) * 32767.0f));
    }

    // Legacy path
    auto legacy_blocks = fp32_to_q16_1(fp32_data);
    apply_rope_q16_1_integer_head(legacy_blocks.data(), BLOCKS_PER_HEAD, cos_q15.data(), sin_q15.data());
    auto legacy_result = q16_1_to_fp32(legacy_blocks);
    legacy_result.resize(HEAD_DIM);

    // Templated dispatch path
    auto dispatch_blocks = fp32_to_q16_1(fp32_data);
    apply_rope_q16_integer_head<Q16_1Block>(dispatch_blocks.data(), BLOCKS_PER_HEAD, cos_q15.data(), sin_q15.data());
    auto dispatch_result = q16_1_to_fp32(dispatch_blocks);
    dispatch_result.resize(HEAD_DIM);

    float sim = cosine_similarity(legacy_result, dispatch_result);
    EXPECT_GT(sim, 0.999999f) << "Templated dispatch should match legacy path exactly";
}

// ============================================================================
// Variable Block Size Performance Tests
// ============================================================================

TEST_F(Test__Q16_VariableBlockRoPE, Block64_AVX2_Speedup_VsScalar)
{
    constexpr int HEAD_DIM = 64;
    constexpr int BLOCK_SIZE = 64;
    constexpr int BLOCKS_PER_HEAD = HEAD_DIM / BLOCK_SIZE;
    const int half_dim = HEAD_DIM / 2;
    constexpr int WARMUP_ITERS = 100;
    constexpr int BENCH_ITERS = 10000;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> fp32_data(HEAD_DIM);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15(half_dim), sin_q15(half_dim);
    for (int i = 0; i < half_dim; ++i)
    {
        float freq = 1.0f / std::pow(ROPE_THETA, static_cast<float>(2 * i) / HEAD_DIM);
        float angle = static_cast<float>(POSITION) * freq;
        cos_q15[i] = static_cast<int16_t>(std::round(std::cos(angle) * 32767.0f));
        sin_q15[i] = static_cast<int16_t>(std::round(std::sin(angle) * 32767.0f));
    }

    // Pre-convert to Q16 blocks (don't include conversion in benchmark)
    auto blocks_scalar = fp32_to_q16_1_block64(fp32_data);
    auto blocks_avx2 = fp32_to_q16_1_block64(fp32_data);
    auto blocks_scalar_backup = blocks_scalar;
    auto blocks_avx2_backup = blocks_avx2;

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i)
    {
        blocks_scalar = blocks_scalar_backup;
        apply_rope_q16_integer_head_scalar<Q16_1Block_64>(blocks_scalar.data(), BLOCKS_PER_HEAD, cos_q15.data(), sin_q15.data());
        blocks_avx2 = blocks_avx2_backup;
        apply_rope_q16_integer_head_avx2<Q16_1Block_64>(blocks_avx2.data(), BLOCKS_PER_HEAD, cos_q15.data(), sin_q15.data());
    }

    // Benchmark scalar (RoPE only, no conversion)
    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < BENCH_ITERS; ++i)
    {
        blocks_scalar = blocks_scalar_backup;
        apply_rope_q16_integer_head_scalar<Q16_1Block_64>(blocks_scalar.data(), BLOCKS_PER_HEAD, cos_q15.data(), sin_q15.data());
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    double scalar_ns = std::chrono::duration<double, std::nano>(end_scalar - start_scalar).count() / BENCH_ITERS;

    // Benchmark AVX2 (RoPE only, no conversion)
    auto start_avx2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < BENCH_ITERS; ++i)
    {
        blocks_avx2 = blocks_avx2_backup;
        apply_rope_q16_integer_head_avx2<Q16_1Block_64>(blocks_avx2.data(), BLOCKS_PER_HEAD, cos_q15.data(), sin_q15.data());
    }
    auto end_avx2 = std::chrono::high_resolution_clock::now();
    double avx2_ns = std::chrono::duration<double, std::nano>(end_avx2 - start_avx2).count() / BENCH_ITERS;

    double speedup = scalar_ns / avx2_ns;
    std::cout << "[Performance] Block64 AVX2 vs Scalar: " << std::fixed << std::setprecision(1) << speedup
              << "x speedup (scalar=" << scalar_ns << " ns, avx2=" << avx2_ns << " ns)" << std::endl;

    // Note: Speedup varies significantly based on CPU load, virtualization, etc.
    // We just log the performance here for informational purposes.
}

#if defined(__AVX512F__)
TEST_F(Test__Q16_VariableBlockRoPE, Block128_AVX512_Speedup_VsScalar)
{
    constexpr int HEAD_DIM = 128;
    constexpr int BLOCK_SIZE = 128;
    constexpr int BLOCKS_PER_HEAD = HEAD_DIM / BLOCK_SIZE;
    const int half_dim = HEAD_DIM / 2;
    constexpr int WARMUP_ITERS = 100;
    constexpr int BENCH_ITERS = 10000;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> fp32_data(HEAD_DIM);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15(half_dim), sin_q15(half_dim);
    for (int i = 0; i < half_dim; ++i)
    {
        float freq = 1.0f / std::pow(ROPE_THETA, static_cast<float>(2 * i) / HEAD_DIM);
        float angle = static_cast<float>(POSITION) * freq;
        cos_q15[i] = static_cast<int16_t>(std::round(std::cos(angle) * 32767.0f));
        sin_q15[i] = static_cast<int16_t>(std::round(std::sin(angle) * 32767.0f));
    }

    // Pre-convert to Q16 blocks (don't include conversion in benchmark)
    auto blocks_scalar = fp32_to_q16_1_block128(fp32_data);
    auto blocks_avx512 = fp32_to_q16_1_block128(fp32_data);
    auto blocks_scalar_backup = blocks_scalar;
    auto blocks_avx512_backup = blocks_avx512;

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i)
    {
        blocks_scalar = blocks_scalar_backup;
        apply_rope_q16_integer_head_scalar<Q16_1Block_128>(blocks_scalar.data(), BLOCKS_PER_HEAD, cos_q15.data(), sin_q15.data());
        blocks_avx512 = blocks_avx512_backup;
        apply_rope_q16_integer_head_avx512<Q16_1Block_128>(blocks_avx512.data(), BLOCKS_PER_HEAD, cos_q15.data(), sin_q15.data());
    }

    // Benchmark scalar (RoPE only, no conversion)
    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < BENCH_ITERS; ++i)
    {
        blocks_scalar = blocks_scalar_backup;
        apply_rope_q16_integer_head_scalar<Q16_1Block_128>(blocks_scalar.data(), BLOCKS_PER_HEAD, cos_q15.data(), sin_q15.data());
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    double scalar_ns = std::chrono::duration<double, std::nano>(end_scalar - start_scalar).count() / BENCH_ITERS;

    // Benchmark AVX512 (RoPE only, no conversion)
    auto start_avx512 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < BENCH_ITERS; ++i)
    {
        blocks_avx512 = blocks_avx512_backup;
        apply_rope_q16_integer_head_avx512<Q16_1Block_128>(blocks_avx512.data(), BLOCKS_PER_HEAD, cos_q15.data(), sin_q15.data());
    }
    auto end_avx512 = std::chrono::high_resolution_clock::now();
    double avx512_ns = std::chrono::duration<double, std::nano>(end_avx512 - start_avx512).count() / BENCH_ITERS;

    double speedup = scalar_ns / avx512_ns;
    std::cout << "[Performance] Block128 AVX512 vs Scalar: " << std::fixed << std::setprecision(1) << speedup
              << "x speedup (scalar=" << scalar_ns << " ns, avx512=" << avx512_ns << " ns)" << std::endl;

    // For single-block (in-place rotation), speedup may be lower than multi-block case
    EXPECT_GT(speedup, 1.5) << "Block128 AVX512 should provide at least 1.5x speedup over scalar";
}
#endif
