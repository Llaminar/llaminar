/**
 * @file Test__Q8_1RoPE.cpp
 * @brief Unit tests for Q8_1 RoPE vectorized implementations
 * @author David Sanftenberg
 *
 * Tests:
 * 1. Correctness: Q8_1 RoPE (scalar/AVX2/AVX512) vs FP32 reference
 * 2. Implementation parity: AVX512 vs AVX2 vs Scalar produce identical Q8_1 results
 * 3. Performance: Expected speedups (AVX512 ~1.2x AVX2, AVX2 ~6x scalar)
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
    // FP32 to FP16 conversion helper
    inline uint16_t fp32_to_fp16(float value)
    {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(float));
        uint32_t sign = (bits >> 16) & 0x8000;
        int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t frac = (bits >> 13) & 0x3FF;
        if (exp <= 0)
            return static_cast<uint16_t>(sign);
        if (exp >= 31)
            return static_cast<uint16_t>(sign | 0x7C00);
        return static_cast<uint16_t>(sign | (exp << 10) | frac);
    }

    // FP16 to FP32 conversion helper
    inline float fp16_to_fp32(uint16_t value)
    {
        uint32_t sign = (value & 0x8000) << 16;
        uint32_t exp = (value >> 10) & 0x1F;
        uint32_t frac = value & 0x3FF;

        if (exp == 0)
        {
            if (frac == 0)
            {
                uint32_t result = sign;
                float f;
                std::memcpy(&f, &result, sizeof(float));
                return f;
            }
            // Denormal
            exp = 1;
            while (!(frac & 0x400))
            {
                frac <<= 1;
                exp--;
            }
            frac &= 0x3FF;
        }
        else if (exp == 31)
        {
            uint32_t result = sign | 0x7F800000 | (frac << 13);
            float f;
            std::memcpy(&f, &result, sizeof(float));
            return f;
        }

        uint32_t result = sign | ((exp + 127 - 15) << 23) | (frac << 13);
        float f;
        std::memcpy(&f, &result, sizeof(float));
        return f;
    }

    // ============================================================================
    // Test Utilities
    // ============================================================================

    /**
     * @brief Quantize FP32 data to Q8_1 blocks
     */
    std::vector<Q8_1Block> fp32_to_q8_1(const std::vector<float> &fp32)
    {
        const size_t n_blocks = (fp32.size() + 31) / 32;
        std::vector<Q8_1Block> blocks(n_blocks);

        std::vector<float> padded = fp32;
        padded.resize(n_blocks * 32, 0.0f);

        for (size_t b = 0; b < n_blocks; ++b)
        {
            const float *block_data = padded.data() + b * 32;
            Q8_1Block &blk = blocks[b];

            float max_abs = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                max_abs = std::max(max_abs, std::fabs(block_data[i]));
            }

            float scale = max_abs / 127.0f;
            if (scale < 1e-20f)
                scale = 1e-20f;
            float inv_scale = 1.0f / scale;

            int32_t sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(block_data[i] * inv_scale));
                q = std::max(-127, std::min(127, q));
                blk.qs[i] = static_cast<int8_t>(q);
                sum_qs += q;
            }

            blk.d = fp32_to_fp16(scale);
            blk.sum_qs = static_cast<int16_t>(sum_qs);
        }

        return blocks;
    }

    /**
     * @brief Dequantize Q8_1 blocks to FP32
     */
    std::vector<float> q8_1_to_fp32(const std::vector<Q8_1Block> &blocks)
    {
        std::vector<float> fp32(blocks.size() * 32);

        for (size_t b = 0; b < blocks.size(); ++b)
        {
            const Q8_1Block &blk = blocks[b];
            float *block_data = fp32.data() + b * 32;
            float scale = fp16_to_fp32(blk.d);

            for (int i = 0; i < 32; ++i)
            {
                block_data[i] = scale * static_cast<float>(blk.qs[i]);
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
        const float *cos_vals,
        const float *sin_vals)
    {
        const int half_dim = head_dim / 2;
        for (int i = 0; i < half_dim; ++i)
        {
            float x = head_data[i];
            float y = head_data[i + half_dim];
            float c = cos_vals[i];
            float s = sin_vals[i];

            head_data[i] = x * c - y * s;
            head_data[i + half_dim] = x * s + y * c;
        }
    }

    /**
     * @brief Generate RoPE sin/cos tables in Q15 and FP32 format
     */
    void generate_rope_tables(
        int half_dim,
        int position,
        float theta,
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
            float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / (2 * half_dim));
            float angle = position * freq;

            cos_fp32[i] = std::cos(angle);
            sin_fp32[i] = std::sin(angle);

            cos_q15[i] = static_cast<int16_t>(cos_fp32[i] * 32767.0f);
            sin_q15[i] = static_cast<int16_t>(sin_fp32[i] * 32767.0f);
        }
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

        if (norm_a < 1e-12f || norm_b < 1e-12f)
            return 0.0f;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    /**
     * @brief Compute max absolute error between two vectors
     */
    float max_abs_error(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size())
            return std::numeric_limits<float>::infinity();

        float max_err = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            max_err = std::max(max_err, std::fabs(a[i] - b[i]));
        }
        return max_err;
    }

    // Test constants
    constexpr int HEAD_DIM_64 = 64;
    constexpr int HEAD_DIM_128 = 128;
    constexpr int POSITION = 42;
    constexpr float ROPE_THETA = 10000.0f;

} // anonymous namespace

// ============================================================================
// Test Fixture
// ============================================================================

class Test__Q8_1RoPE : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// Correctness Tests: Q8_1 SIMD vs FP32 Reference
// ============================================================================

TEST_F(Test__Q8_1RoPE, Scalar_Correctness_VsFP32Reference)
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

    // Q8_1 scalar
    auto q8_blocks = fp32_to_q8_1(fp32_data);
    apply_rope_q8_1_integer_head_scalar(q8_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());
    auto q8_result = q8_1_to_fp32(q8_blocks);
    q8_result.resize(head_dim);

    float sim = cosine_similarity(q8_result, fp32_reference);
    float max_err = max_abs_error(q8_result, fp32_reference);

    // Q8_1 has lower precision than Q16_1, so we relax thresholds
    EXPECT_GT(sim, 0.999f) << "Q8_1 scalar should match FP32 reference closely";
    EXPECT_LT(max_err, 0.02f) << "Q8_1 scalar max error too high: " << max_err;
}

#if defined(__AVX2__)
TEST_F(Test__Q8_1RoPE, AVX2_Correctness_VsFP32Reference)
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

    // Q8_1 AVX2
    auto q8_blocks = fp32_to_q8_1(fp32_data);
    apply_rope_q8_1_integer_head_avx2(q8_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());
    auto q8_result = q8_1_to_fp32(q8_blocks);
    q8_result.resize(head_dim);

    float sim = cosine_similarity(q8_result, fp32_reference);
    float max_err = max_abs_error(q8_result, fp32_reference);

    EXPECT_GT(sim, 0.999f) << "Q8_1 AVX2 should match FP32 reference closely";
    EXPECT_LT(max_err, 0.02f) << "Q8_1 AVX2 max error too high: " << max_err;
}
#endif

#if defined(__AVX512F__)
TEST_F(Test__Q8_1RoPE, AVX512_Correctness_VsFP32Reference)
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

    // Q8_1 AVX512
    auto q8_blocks = fp32_to_q8_1(fp32_data);
    apply_rope_q8_1_integer_head_avx512(q8_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());
    auto q8_result = q8_1_to_fp32(q8_blocks);
    q8_result.resize(head_dim);

    float sim = cosine_similarity(q8_result, fp32_reference);
    float max_err = max_abs_error(q8_result, fp32_reference);

    EXPECT_GT(sim, 0.999f) << "Q8_1 AVX512 should match FP32 reference closely";
    EXPECT_LT(max_err, 0.02f) << "Q8_1 AVX512 max error too high: " << max_err;
}
#endif

// ============================================================================
// Implementation Parity: SIMD vs Scalar produce identical Q8_1 blocks
// ============================================================================

#if defined(__AVX2__)
TEST_F(Test__Q8_1RoPE, AVX2_Parity_VsScalar)
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

    // Scalar
    auto scalar_blocks = fp32_to_q8_1(fp32_data);
    apply_rope_q8_1_integer_head_scalar(scalar_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());

    // AVX2
    auto avx2_blocks = fp32_to_q8_1(fp32_data);
    apply_rope_q8_1_integer_head_avx2(avx2_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());

    // Compare dequantized results
    auto scalar_fp32 = q8_1_to_fp32(scalar_blocks);
    auto avx2_fp32 = q8_1_to_fp32(avx2_blocks);

    float sim = cosine_similarity(scalar_fp32, avx2_fp32);
    float max_err = max_abs_error(scalar_fp32, avx2_fp32);

    EXPECT_GT(sim, 0.9999f) << "AVX2 should produce nearly identical results to scalar";
    EXPECT_LT(max_err, 1e-3f) << "AVX2 vs Scalar max error: " << max_err;
}
#endif

#if defined(__AVX512F__)
TEST_F(Test__Q8_1RoPE, AVX512_Parity_VsScalar)
{
    const int head_dim = HEAD_DIM_128;
    const int blocks_per_head = head_dim / 32;
    const int half_dim = head_dim / 2;

    std::mt19937 rng(456);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> fp32_data(head_dim);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(half_dim, POSITION, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // Scalar
    auto scalar_blocks = fp32_to_q8_1(fp32_data);
    apply_rope_q8_1_integer_head_scalar(scalar_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());

    // AVX512
    auto avx512_blocks = fp32_to_q8_1(fp32_data);
    apply_rope_q8_1_integer_head_avx512(avx512_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());

    // Compare
    auto scalar_fp32 = q8_1_to_fp32(scalar_blocks);
    auto avx512_fp32 = q8_1_to_fp32(avx512_blocks);

    float sim = cosine_similarity(scalar_fp32, avx512_fp32);
    float max_err = max_abs_error(scalar_fp32, avx512_fp32);

    EXPECT_GT(sim, 0.9999f) << "AVX512 should produce nearly identical results to scalar";
    EXPECT_LT(max_err, 1e-3f) << "AVX512 vs Scalar max error: " << max_err;
}

TEST_F(Test__Q8_1RoPE, AVX512_Parity_VsAVX2)
{
    const int head_dim = HEAD_DIM_128;
    const int blocks_per_head = head_dim / 32;
    const int half_dim = head_dim / 2;

    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> fp32_data(head_dim);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(half_dim, POSITION, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // AVX2
    auto avx2_blocks = fp32_to_q8_1(fp32_data);
    apply_rope_q8_1_integer_head_avx2(avx2_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());

    // AVX512
    auto avx512_blocks = fp32_to_q8_1(fp32_data);
    apply_rope_q8_1_integer_head_avx512(avx512_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());

    // Compare
    auto avx2_fp32 = q8_1_to_fp32(avx2_blocks);
    auto avx512_fp32 = q8_1_to_fp32(avx512_blocks);

    float sim = cosine_similarity(avx2_fp32, avx512_fp32);
    float max_err = max_abs_error(avx2_fp32, avx512_fp32);

    EXPECT_GT(sim, 0.99999f) << "AVX512 and AVX2 should produce nearly identical results";
    EXPECT_LT(max_err, 1e-4f) << "AVX512 vs AVX2 max error: " << max_err;
}
#endif

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(Test__Q8_1RoPE, Correctness_LargeHeadDim)
{
    const int head_dim = 256;
    const int blocks_per_head = head_dim / 32;
    const int half_dim = head_dim / 2;

    std::mt19937 rng(999);
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

    // Q8_1 scalar
    auto q8_blocks = fp32_to_q8_1(fp32_data);
    apply_rope_q8_1_integer_head_scalar(q8_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());
    auto q8_result = q8_1_to_fp32(q8_blocks);
    q8_result.resize(head_dim);

    float sim = cosine_similarity(q8_result, fp32_reference);
    EXPECT_GT(sim, 0.999f) << "Large head_dim (256) should work correctly";
}

TEST_F(Test__Q8_1RoPE, Correctness_HighPosition)
{
    const int head_dim = HEAD_DIM_64;
    const int blocks_per_head = head_dim / 32;
    const int half_dim = head_dim / 2;
    const int high_position = 100000;

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

    // Q8_1 scalar
    auto q8_blocks = fp32_to_q8_1(fp32_data);
    apply_rope_q8_1_integer_head_scalar(q8_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());
    auto q8_result = q8_1_to_fp32(q8_blocks);
    q8_result.resize(head_dim);

    float sim = cosine_similarity(q8_result, fp32_reference);
    EXPECT_GT(sim, 0.999f) << "High position (100k) should work correctly";
}

TEST_F(Test__Q8_1RoPE, Correctness_SmallValues)
{
    const int head_dim = HEAD_DIM_64;
    const int blocks_per_head = head_dim / 32;
    const int half_dim = head_dim / 2;

    // Very small input values
    std::mt19937 rng(222);
    std::uniform_real_distribution<float> dist(-1e-4f, 1e-4f);
    std::vector<float> fp32_data(head_dim);
    for (auto &v : fp32_data)
        v = dist(rng);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(half_dim, POSITION, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // Q8_1 should handle small values without NaN/Inf
    auto q8_blocks = fp32_to_q8_1(fp32_data);
    apply_rope_q8_1_integer_head_scalar(q8_blocks.data(), blocks_per_head, cos_q15.data(), sin_q15.data());
    auto q8_result = q8_1_to_fp32(q8_blocks);

    bool has_nan_inf = false;
    for (float v : q8_result)
    {
        if (std::isnan(v) || std::isinf(v))
        {
            has_nan_inf = true;
            break;
        }
    }
    EXPECT_FALSE(has_nan_inf) << "Q8_1 RoPE should handle small values without NaN/Inf";
}

// ============================================================================
// Performance Tests
// ============================================================================

class Test__Q8_1RoPE_Performance : public ::testing::Test
{
protected:
    static constexpr int HEAD_DIM = 128;
    static constexpr int BLOCKS_PER_HEAD = HEAD_DIM / 32;
    static constexpr int HALF_DIM = HEAD_DIM / 2;
    static constexpr int N_HEADS = 32;
    static constexpr int BENCHMARK_ITERATIONS = 10000;

    std::vector<Q8_1Block> q8_blocks;
    std::vector<int16_t> cos_q15;
    std::vector<int16_t> sin_q15;

    void SetUp() override
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> fp32_data(HEAD_DIM * N_HEADS);
        for (auto &v : fp32_data)
            v = dist(rng);

        q8_blocks = fp32_to_q8_1(fp32_data);

        std::vector<float> cos_fp32, sin_fp32;
        generate_rope_tables(HALF_DIM, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);
    }

    using RopeFunc = void (*)(Q8_1Block *, int, const int16_t *, const int16_t *);

    double benchmark_rope(RopeFunc func, int n_iterations)
    {
        // Warmup
        for (int i = 0; i < 100; ++i)
        {
            for (int h = 0; h < N_HEADS; ++h)
            {
                func(q8_blocks.data() + h * BLOCKS_PER_HEAD, BLOCKS_PER_HEAD,
                     cos_q15.data(), sin_q15.data());
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < n_iterations; ++iter)
        {
            for (int h = 0; h < N_HEADS; ++h)
            {
                func(q8_blocks.data() + h * BLOCKS_PER_HEAD, BLOCKS_PER_HEAD,
                     cos_q15.data(), sin_q15.data());
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        return static_cast<double>(duration_ns) / (n_iterations * N_HEADS);
    }
};

TEST_F(Test__Q8_1RoPE_Performance, Scalar_Baseline)
{
    double ns_per_head = benchmark_rope(apply_rope_q8_1_integer_head_scalar, BENCHMARK_ITERATIONS);
    printf("[Performance] Scalar: %.1f ns/head (%.1f heads/us)\n",
           ns_per_head, 1000.0 / ns_per_head);

    EXPECT_LT(ns_per_head, 100000.0) << "Scalar RoPE too slow";
}

#if defined(__AVX2__)
TEST_F(Test__Q8_1RoPE_Performance, AVX2_Speedup_VsScalar)
{
    double scalar_ns = benchmark_rope(apply_rope_q8_1_integer_head_scalar, BENCHMARK_ITERATIONS);
    double avx2_ns = benchmark_rope(apply_rope_q8_1_integer_head_avx2, BENCHMARK_ITERATIONS);

    double speedup = scalar_ns / avx2_ns;
    printf("[Performance] AVX2 vs Scalar: %.1fx speedup (scalar=%.1f ns, avx2=%.1f ns)\n",
           speedup, scalar_ns, avx2_ns);

    // AVX2 processes 8 elements/iteration vs 1 for scalar
    // With aggressive unrolling and interleaving, we expect ~6-8x speedup
    EXPECT_GT(speedup, 4.0) << "AVX2 should be at least 4x faster than scalar";
    EXPECT_LT(speedup, 15.0) << "AVX2 speedup suspiciously high (>15x)";
}
#endif

#if defined(__AVX512F__)
TEST_F(Test__Q8_1RoPE_Performance, AVX512_Speedup_VsAVX2)
{
    double avx2_ns = benchmark_rope(apply_rope_q8_1_integer_head_avx2, BENCHMARK_ITERATIONS);
    double avx512_ns = benchmark_rope(apply_rope_q8_1_integer_head_avx512, BENCHMARK_ITERATIONS);

    double speedup = avx2_ns / avx512_ns;
    printf("[Performance] AVX512 vs AVX2: %.2fx speedup (avx2=%.1f ns, avx512=%.1f ns)\n",
           speedup, avx2_ns, avx512_ns);

    // Q8_1 blocks are smaller (36 bytes vs 72 bytes for Q16_1), so we hit
    // memory bandwidth limits at lower latency. AVX512 provides minimal
    // additional benefit over the highly optimized AVX2 at this data size.
    EXPECT_GT(speedup, 0.95) << "AVX512 should be at least as fast as AVX2";
    EXPECT_LT(speedup, 3.0) << "AVX512 speedup suspiciously high (>3x)";
}

TEST_F(Test__Q8_1RoPE_Performance, AVX512_Speedup_VsScalar)
{
    double scalar_ns = benchmark_rope(apply_rope_q8_1_integer_head_scalar, BENCHMARK_ITERATIONS);
    double avx512_ns = benchmark_rope(apply_rope_q8_1_integer_head_avx512, BENCHMARK_ITERATIONS);

    double speedup = scalar_ns / avx512_ns;
    printf("[Performance] AVX512 vs Scalar: %.1fx speedup (scalar=%.1f ns, avx512=%.1f ns)\n",
           speedup, scalar_ns, avx512_ns);

    // Combined effect of AVX2 and AVX512 optimizations
    EXPECT_GT(speedup, 6.0) << "AVX512 should be at least 6x faster than scalar";
}
#endif
