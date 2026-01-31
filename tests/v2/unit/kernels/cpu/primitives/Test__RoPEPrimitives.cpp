/**
 * @file Test__RoPEPrimitives.cpp
 * @brief Unit tests for RoPE primitive implementations (scalar, AVX2, AVX512)
 * @author David Sanftenberg
 *
 * Validates that:
 * 1. All implementations (scalar, AVX2, AVX512) produce identical results
 * 2. RoPE correctly handles various head dimensions (32, 64, 128)
 * 3. Position encoding is consistent across different positions
 * 4. Edge cases (odd head_dim, position=0) are handled correctly
 * 5. Q8_1 -> Q16_1 RoPE conversion maintains precision
 */

#include "kernels/cpu/primitives/RoPEPrimitives.h"
#include "tensors/BlockStructures.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>

using namespace llaminar2::primitives;
using namespace llaminar2;

namespace
{
    /**
     * @brief Generate random test data for a single head
     */
    std::vector<float> generate_test_head(int head_dim, uint32_t seed = 42)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> data(head_dim);
        for (auto &val : data)
        {
            val = dist(gen);
        }
        return data;
    }

    /**
     * @brief Compare two float vectors element-wise
     * @return Tuple of (max_abs_diff, rel_l2_norm, num_mismatches)
     */
    std::tuple<float, float, int> compare_vectors(
        const std::vector<float> &a,
        const std::vector<float> &b,
        float tolerance = 1e-6f)
    {
        EXPECT_EQ(a.size(), b.size());
        if (a.size() != b.size())
            return {INFINITY, INFINITY, static_cast<int>(std::max(a.size(), b.size()))};

        float max_abs_diff = 0.0f;
        float sum_sq_diff = 0.0f;
        float sum_sq_ref = 0.0f;
        int mismatches = 0;

        for (size_t i = 0; i < a.size(); ++i)
        {
            float diff = std::abs(a[i] - b[i]);
            max_abs_diff = std::max(max_abs_diff, diff);
            sum_sq_diff += diff * diff;
            sum_sq_ref += b[i] * b[i];

            if (diff > tolerance)
            {
                mismatches++;
            }
        }

        float rel_l2 = sum_sq_ref > 0.0f ? std::sqrt(sum_sq_diff / sum_sq_ref) : 0.0f;
        return {max_abs_diff, rel_l2, mismatches};
    }

    /**
     * @brief Print comparison metrics
     */
    void print_comparison(const char *impl1, const char *impl2, float max_abs, float rel_l2, int mismatches, int total)
    {
        printf("  %s vs %s: max_abs=%.2e, rel_l2=%.2e, mismatches=%d/%d\n",
               impl1, impl2, max_abs, rel_l2, mismatches, total);
    }
} // anonymous namespace

// ============================================================================
// Test Suite: RoPE Implementation Parity
// ============================================================================

class RoPEPrimitivesTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Detect available ISA
#if defined(__AVX512F__)
        has_avx512_ = true;
        has_avx2_ = true;
#elif defined(__AVX2__)
        has_avx2_ = true;
#else
        // Scalar only
#endif
    }

    bool has_avx2_ = false;
    bool has_avx512_ = false;
};

// ============================================================================
// Test 1: Scalar vs Vectorized Parity
// ============================================================================

TEST_F(RoPEPrimitivesTest, ScalarVsVectorizedParity)
{
    const int head_dim = 64;
    const int position = 10;
    const float freq_base = 10000.0f;

    const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

    // Generate test data
    auto original_data = generate_test_head(head_dim);

    // Apply scalar
    auto scalar_result = original_data;
    apply_rope_to_head_scalar(scalar_result.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX2__)
    if (has_avx2_)
    {
        auto avx2_result = original_data;
        int processed = apply_rope_to_head_avx2(avx2_result.data(), position, inv_freq, head_dim);
        apply_rope_to_head_scalar(avx2_result.data(), position, inv_freq, head_dim, processed);

        auto [max_abs, rel_l2, mismatches] = compare_vectors(avx2_result, scalar_result, 1e-6f);
        print_comparison("AVX2", "Scalar", max_abs, rel_l2, mismatches, head_dim);

        EXPECT_LT(max_abs, 1e-5f) << "AVX2 and scalar implementations differ significantly";
        EXPECT_LT(rel_l2, 1e-6f) << "AVX2 relative error too high";
        EXPECT_EQ(mismatches, 0) << "AVX2 has mismatches vs scalar";
    }
#endif

#if defined(__AVX512F__)
    if (has_avx512_)
    {
        auto avx512_result = original_data;
        int processed = apply_rope_to_head_avx512(avx512_result.data(), position, inv_freq, head_dim);
        apply_rope_to_head_scalar(avx512_result.data(), position, inv_freq, head_dim, processed);

        auto [max_abs, rel_l2, mismatches] = compare_vectors(avx512_result, scalar_result, 1e-6f);
        print_comparison("AVX512", "Scalar", max_abs, rel_l2, mismatches, head_dim);

        EXPECT_LT(max_abs, 1e-5f) << "AVX512 and scalar implementations differ significantly";
        EXPECT_LT(rel_l2, 1e-6f) << "AVX512 relative error too high";
        EXPECT_EQ(mismatches, 0) << "AVX512 has mismatches vs scalar";
    }
#endif
}

// ============================================================================
// Test 2: Multiple Head Dimensions
// ============================================================================

TEST_F(RoPEPrimitivesTest, VariousHeadDimensions)
{
    const std::vector<int> head_dims = {32, 64, 128};
    const int position = 5;
    const float freq_base = 10000.0f;

    for (int head_dim : head_dims)
    {
        SCOPED_TRACE("head_dim=" + std::to_string(head_dim));

        const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);
        auto original_data = generate_test_head(head_dim);

        // Scalar reference
        auto scalar_result = original_data;
        apply_rope_to_head_scalar(scalar_result.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX2__)
        if (has_avx2_)
        {
            auto avx2_result = original_data;
            int processed = apply_rope_to_head_avx2(avx2_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx2_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx2_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f) << "AVX2 failed for head_dim=" << head_dim;
            EXPECT_EQ(mismatches, 0);
        }
#endif

#if defined(__AVX512F__)
        if (has_avx512_)
        {
            auto avx512_result = original_data;
            int processed = apply_rope_to_head_avx512(avx512_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx512_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx512_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f) << "AVX512 failed for head_dim=" << head_dim;
            EXPECT_EQ(mismatches, 0);
        }
#endif
    }
}

// ============================================================================
// Test 3: Multiple Positions
// ============================================================================

TEST_F(RoPEPrimitivesTest, VariousPositions)
{
    const int head_dim = 64;
    const float freq_base = 10000.0f;
    const std::vector<int> positions = {0, 1, 10, 100, 1000};

    const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

    for (int position : positions)
    {
        SCOPED_TRACE("position=" + std::to_string(position));

        auto original_data = generate_test_head(head_dim);

        // Scalar reference
        auto scalar_result = original_data;
        apply_rope_to_head_scalar(scalar_result.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX2__)
        if (has_avx2_)
        {
            auto avx2_result = original_data;
            int processed = apply_rope_to_head_avx2(avx2_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx2_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx2_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f) << "AVX2 failed for position=" << position;
            EXPECT_EQ(mismatches, 0);
        }
#endif

#if defined(__AVX512F__)
        if (has_avx512_)
        {
            auto avx512_result = original_data;
            int processed = apply_rope_to_head_avx512(avx512_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx512_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx512_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f) << "AVX512 failed for position=" << position;
            EXPECT_EQ(mismatches, 0);
        }
#endif
    }
}

// ============================================================================
// Test 4: Frequency Base Variations
// ============================================================================

TEST_F(RoPEPrimitivesTest, FrequencyBaseVariations)
{
    const int head_dim = 64;
    const int position = 10;
    const std::vector<float> freq_bases = {10000.0f, 500000.0f, 1000000.0f};

    for (float freq_base : freq_bases)
    {
        SCOPED_TRACE("freq_base=" + std::to_string(freq_base));

        const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);
        auto original_data = generate_test_head(head_dim);

        // Scalar reference
        auto scalar_result = original_data;
        apply_rope_to_head_scalar(scalar_result.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX2__)
        if (has_avx2_)
        {
            auto avx2_result = original_data;
            int processed = apply_rope_to_head_avx2(avx2_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx2_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx2_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f);
            EXPECT_EQ(mismatches, 0);
        }
#endif

#if defined(__AVX512F__)
        if (has_avx512_)
        {
            auto avx512_result = original_data;
            int processed = apply_rope_to_head_avx512(avx512_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx512_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx512_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f);
            EXPECT_EQ(mismatches, 0);
        }
#endif
    }
}

// ============================================================================
// Test 5: Edge Cases
// ============================================================================

TEST_F(RoPEPrimitivesTest, EdgeCases)
{
    const int head_dim = 64;
    const float freq_base = 10000.0f;
    const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

    // Test 5a: Zero position
    {
        auto original_data = generate_test_head(head_dim);
        auto scalar_result = original_data;
        apply_rope_to_head_scalar(scalar_result.data(), 0, inv_freq, head_dim, 0);

        // At position 0, angles are all zero, so cos=1, sin=0
        // Rotation becomes: new_first = x_first * 1 - x_second * 0 = x_first
        //                   new_second = x_first * 0 + x_second * 1 = x_second
        // So result should equal input
        auto [max_abs, rel_l2, mismatches] = compare_vectors(scalar_result, original_data, 1e-6f);
        EXPECT_LT(max_abs, 1e-5f) << "Position 0 should be identity transform";
    }

    // Test 5b: All zeros input
    {
        std::vector<float> zeros(head_dim, 0.0f);
        auto result = zeros;
        apply_rope_to_head_scalar(result.data(), 10, inv_freq, head_dim, 0);

        auto [max_abs, rel_l2, mismatches] = compare_vectors(result, zeros, 1e-6f);
        EXPECT_LT(max_abs, 1e-5f) << "All-zeros input should remain all-zeros";
    }

    // Test 5c: Large position (test numerical stability)
    {
        auto original_data = generate_test_head(head_dim);
        auto scalar_result = original_data;
        apply_rope_to_head_scalar(scalar_result.data(), 1000000, inv_freq, head_dim, 0);

        // Check for NaN/Inf
        bool has_invalid = false;
        for (float val : scalar_result)
        {
            if (!std::isfinite(val))
            {
                has_invalid = true;
                break;
            }
        }
        EXPECT_FALSE(has_invalid) << "Large position should not produce NaN/Inf";
    }
}

// ============================================================================
// Test 6: Vectorized Tail Handling
// ============================================================================

TEST_F(RoPEPrimitivesTest, VectorizedTailHandling)
{
    // Test head dimensions that don't align perfectly with vector widths
    const std::vector<int> misaligned_dims = {36, 68, 100}; // Not multiples of 8 or 16
    const int position = 10;
    const float freq_base = 10000.0f;

    for (int head_dim : misaligned_dims)
    {
        SCOPED_TRACE("head_dim=" + std::to_string(head_dim));

        const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);
        auto original_data = generate_test_head(head_dim);

        // Scalar reference
        auto scalar_result = original_data;
        apply_rope_to_head_scalar(scalar_result.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX2__)
        if (has_avx2_)
        {
            auto avx2_result = original_data;
            int processed = apply_rope_to_head_avx2(avx2_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx2_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx2_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f) << "AVX2 tail handling failed for head_dim=" << head_dim;
            EXPECT_EQ(mismatches, 0);
        }
#endif

#if defined(__AVX512F__)
        if (has_avx512_)
        {
            auto avx512_result = original_data;
            int processed = apply_rope_to_head_avx512(avx512_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx512_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx512_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f) << "AVX512 tail handling failed for head_dim=" << head_dim;
            EXPECT_EQ(mismatches, 0);
        }
#endif
    }
}

// ============================================================================
// Test 7: Stress Test - Many Heads
// ============================================================================

TEST_F(RoPEPrimitivesTest, StressTestManyHeads)
{
    const int head_dim = 64;
    const int num_heads = 100;
    const int position = 42;
    const float freq_base = 10000.0f;

    const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

    for (int h = 0; h < num_heads; ++h)
    {
        auto original_data = generate_test_head(head_dim, 42 + h);

        // Scalar reference
        auto scalar_result = original_data;
        apply_rope_to_head_scalar(scalar_result.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX512F__)
        if (has_avx512_)
        {
            auto avx512_result = original_data;
            int processed = apply_rope_to_head_avx512(avx512_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx512_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx512_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f) << "Head " << h << " failed";
            EXPECT_EQ(mismatches, 0);
        }
#elif defined(__AVX2__)
        if (has_avx2_)
        {
            auto avx2_result = original_data;
            int processed = apply_rope_to_head_avx2(avx2_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx2_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx2_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f) << "Head " << h << " failed";
            EXPECT_EQ(mismatches, 0);
        }
#endif
    }
}
// ============================================================================
// Test Suite: Q8_1 -> Q16_1 RoPE Conversion
// ============================================================================

class RoPEQ8ToQ16Test : public ::testing::Test
{
protected:
    /**
     * @brief Create random Q8_1 blocks for a single head
     */
    std::vector<Q8_1Block> create_random_q8_1_head(int head_dim, uint32_t seed = 42)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        const int blocks_per_head = head_dim / 32;
        std::vector<Q8_1Block> blocks(blocks_per_head);

        for (int b = 0; b < blocks_per_head; ++b)
        {
            // Generate random values in [-1, 1]
            float max_val = 0.0f;
            std::array<float, 32> vals;
            for (int i = 0; i < 32; ++i)
            {
                vals[i] = dist(gen);
                max_val = std::max(max_val, std::abs(vals[i]));
            }

            // Quantize to Q8_1
            float scale = max_val / 127.0f;
            if (scale < 1e-10f)
                scale = 1e-10f;
            blocks[b].d = fp32_to_fp16(scale);
            int sum = 0;
            for (int i = 0; i < 32; ++i)
            {
                int8_t q = static_cast<int8_t>(std::round(vals[i] / scale));
                blocks[b].qs[i] = q;
                sum += q;
            }
            blocks[b].sum_qs = sum;
        }
        return blocks;
    }

    /**
     * @brief Dequantize Q8_1 blocks to FP32
     */
    std::vector<float> dequantize_q8_1(const std::vector<Q8_1Block> &blocks)
    {
        std::vector<float> result(blocks.size() * 32);
        for (size_t b = 0; b < blocks.size(); ++b)
        {
            float scale = fp16_to_fp32(blocks[b].d);
            for (int i = 0; i < 32; ++i)
            {
                result[b * 32 + i] = static_cast<float>(blocks[b].qs[i]) * scale;
            }
        }
        return result;
    }

    /**
     * @brief Dequantize Q16_1 blocks to FP32
     */
    std::vector<float> dequantize_q16_1(const std::vector<Q16_1Block> &blocks)
    {
        std::vector<float> result(blocks.size() * 32);
        for (size_t b = 0; b < blocks.size(); ++b)
        {
            float scale = blocks[b].d; // Q16_1 uses FP32 scale directly
            for (int i = 0; i < 32; ++i)
            {
                result[b * 32 + i] = static_cast<float>(blocks[b].qs[i]) * scale;
            }
        }
        return result;
    }

    /**
     * @brief Apply FP32 RoPE rotation (reference)
     */
    std::vector<float> apply_rope_fp32_reference(
        const std::vector<float> &input,
        int position,
        int head_dim,
        float rope_theta)
    {
        std::vector<float> result = input;
        const int half_dim = head_dim / 2;

        // Compute inv_freq
        std::vector<float> inv_freq(half_dim);
        for (int i = 0; i < half_dim; ++i)
        {
            float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
            inv_freq[i] = 1.0f / std::pow(rope_theta, exponent);
        }

        // Apply rotation
        for (int i = 0; i < half_dim; ++i)
        {
            float angle = static_cast<float>(position) * inv_freq[i];
            float c = std::cos(angle);
            float s = std::sin(angle);
            float x = input[i];
            float y = input[i + half_dim];
            result[i] = x * c - y * s;
            result[i + half_dim] = x * s + y * c;
        }
        return result;
    }

    /**
     * @brief Compute MSE between two float vectors
     */
    float compute_mse(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size())
            return INFINITY;
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum / static_cast<float>(a.size());
    }

    /**
     * @brief FP16 conversion helpers (matching tensors/FP16Utils.h)
     */
    static uint16_t fp32_to_fp16(float f)
    {
        uint32_t i;
        std::memcpy(&i, &f, sizeof(float));
        uint32_t sign = (i >> 16) & 0x8000;
        int32_t exp = ((i >> 23) & 0xFF) - 127 + 15;
        uint32_t frac = i & 0x7FFFFF;
        if (exp <= 0)
        {
            return static_cast<uint16_t>(sign);
        }
        else if (exp >= 31)
        {
            return static_cast<uint16_t>(sign | 0x7C00);
        }
        return static_cast<uint16_t>(sign | (exp << 10) | (frac >> 13));
    }

    static float fp16_to_fp32(uint16_t h)
    {
        uint32_t sign = (h & 0x8000) << 16;
        int32_t exp = (h >> 10) & 0x1F;
        uint32_t frac = h & 0x3FF;
        if (exp == 0)
        {
            return 0.0f;
        }
        else if (exp == 31)
        {
            return sign ? -INFINITY : INFINITY;
        }
        exp = exp - 15 + 127;
        uint32_t i = sign | (exp << 23) | (frac << 13);
        float f;
        std::memcpy(&f, &i, sizeof(float));
        return f;
    }
};

// ============================================================================
// Test 8: Q8_1 -> Q16_1 RoPE Basic Functionality
// ============================================================================

TEST_F(RoPEQ8ToQ16Test, BasicQ8ToQ16RoPE)
{
    const int head_dim = 128;
    const int blocks_per_head = head_dim / 32;
    const int seq_len = 1;
    const int n_heads = 1;
    const int n_kv_heads = 1;
    const int position = 10;
    const float rope_theta = 10000.0f;

    // Create Q8_1 input
    auto q8_blocks = create_random_q8_1_head(head_dim, 42);
    std::vector<Q16_1Block> q16_blocks(blocks_per_head);

    // Apply Q8_1 -> Q16_1 RoPE
    std::vector<int> position_ids = {position};
    apply_rope_q8_1_to_q16_1(
        q8_blocks.data(), nullptr,
        q16_blocks.data(), nullptr,
        position_ids.data(),
        seq_len, n_heads, n_kv_heads,
        head_dim, rope_theta);

    // Compute reference: Q8_1 -> FP32 -> RoPE -> FP32
    auto fp32_input = dequantize_q8_1(q8_blocks);
    auto fp32_reference = apply_rope_fp32_reference(fp32_input, position, head_dim, rope_theta);

    // Dequantize Q16_1 output
    auto fp32_output = dequantize_q16_1(q16_blocks);

    // Compare: Q16_1 output should be close to FP32 reference
    float mse = compute_mse(fp32_output, fp32_reference);

    // Integer-only Q8→Q16 RoPE has some precision loss compared to FP32.
    // Using the scalar implementation with int64 intermediates, we expect:
    // - MSE < 0.01 (quantization noise is inherent)
    // - This is still good enough for downstream attention computation
    EXPECT_LT(mse, 0.01f) << "Q8_1 -> Q16_1 RoPE error too high: MSE=" << mse;
}

// ============================================================================
// Test 9: Q8_1 -> Q16_1 RoPE vs Q8_1 -> FP32 Precision
// ============================================================================

TEST_F(RoPEQ8ToQ16Test, Q8ToQ16PrecisionVsFP32)
{
    const int head_dim = 128;
    const int blocks_per_head = head_dim / 32;
    const int position = 42;
    const float rope_theta = 10000.0f;

    // Create Q8_1 input
    auto q8_blocks = create_random_q8_1_head(head_dim, 123);

    // Path 1: Q8_1 -> Q16_1 -> dequant
    std::vector<Q16_1Block> q16_blocks(blocks_per_head);
    std::vector<int> position_ids = {position};
    apply_rope_q8_1_to_q16_1(
        q8_blocks.data(), nullptr,
        q16_blocks.data(), nullptr,
        position_ids.data(),
        1, 1, 1, head_dim, rope_theta);
    auto q16_dequant = dequantize_q16_1(q16_blocks);

    // Path 2: Q8_1 -> dequant -> FP32 RoPE (reference)
    auto fp32_input = dequantize_q8_1(q8_blocks);
    auto fp32_reference = apply_rope_fp32_reference(fp32_input, position, head_dim, rope_theta);

    // Q16_1 path should match FP32 reference with integer arithmetic tolerances.
    // The integer-only implementation has quantization error, but produces
    // valid rotations for attention computation.
    float mse = compute_mse(q16_dequant, fp32_reference);
    float max_abs = 0.0f;
    for (size_t i = 0; i < q16_dequant.size(); ++i)
    {
        max_abs = std::max(max_abs, std::abs(q16_dequant[i] - fp32_reference[i]));
    }

    // Integer RoPE tolerances (more lenient than FP32-based implementations)
    EXPECT_LT(mse, 0.01f) << "Q16_1 path MSE too high: " << mse;
    EXPECT_LT(max_abs, 0.5f) << "Q16_1 path max error too high: " << max_abs;
}

// ============================================================================
// Test 10: Q8_1 -> Q16_1 RoPE Multiple Heads
// ============================================================================

TEST_F(RoPEQ8ToQ16Test, MultiHeadQ8ToQ16RoPE)
{
    const int head_dim = 64;
    const int blocks_per_head = head_dim / 32;
    const int seq_len = 4;
    const int n_heads = 8;
    const int n_kv_heads = 2;
    const float rope_theta = 10000.0f;

    // Create Q8_1 input for all heads
    const int total_q_blocks = seq_len * n_heads * blocks_per_head;
    const int total_k_blocks = seq_len * n_kv_heads * blocks_per_head;

    std::vector<Q8_1Block> q_in(total_q_blocks);
    std::vector<Q8_1Block> k_in(total_k_blocks);
    std::vector<Q16_1Block> q_out(total_q_blocks);
    std::vector<Q16_1Block> k_out(total_k_blocks);

    // Fill with random data
    std::mt19937 gen(456);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto &block : q_in)
    {
        float max_val = 0.0f;
        std::array<float, 32> vals;
        for (int i = 0; i < 32; ++i)
        {
            vals[i] = dist(gen);
            max_val = std::max(max_val, std::abs(vals[i]));
        }
        float scale = max_val / 127.0f;
        if (scale < 1e-10f)
            scale = 1e-10f;
        block.d = fp32_to_fp16(scale);
        int sum = 0;
        for (int i = 0; i < 32; ++i)
        {
            int8_t q = static_cast<int8_t>(std::round(vals[i] / scale));
            block.qs[i] = q;
            sum += q;
        }
        block.sum_qs = sum;
    }

    for (auto &block : k_in)
    {
        float max_val = 0.0f;
        std::array<float, 32> vals;
        for (int i = 0; i < 32; ++i)
        {
            vals[i] = dist(gen);
            max_val = std::max(max_val, std::abs(vals[i]));
        }
        float scale = max_val / 127.0f;
        if (scale < 1e-10f)
            scale = 1e-10f;
        block.d = fp32_to_fp16(scale);
        int sum = 0;
        for (int i = 0; i < 32; ++i)
        {
            int8_t q = static_cast<int8_t>(std::round(vals[i] / scale));
            block.qs[i] = q;
            sum += q;
        }
        block.sum_qs = sum;
    }

    // Position IDs
    std::vector<int> position_ids = {0, 1, 2, 3};

    // Apply Q8_1 -> Q16_1 RoPE
    apply_rope_q8_1_to_q16_1(
        q_in.data(), k_in.data(),
        q_out.data(), k_out.data(),
        position_ids.data(),
        seq_len, n_heads, n_kv_heads,
        head_dim, rope_theta);

    // Verify Q output has valid values
    for (int t = 0; t < seq_len; ++t)
    {
        for (int h = 0; h < n_heads; ++h)
        {
            const int idx = t * n_heads * blocks_per_head + h * blocks_per_head;
            for (int b = 0; b < blocks_per_head; ++b)
            {
                const Q16_1Block &block = q_out[idx + b];
                EXPECT_GT(block.d, 0.0f) << "Invalid scale at t=" << t << " h=" << h << " b=" << b;
                // Check that values are in valid Q16_1 range
                for (int i = 0; i < 32; ++i)
                {
                    EXPECT_GE(block.qs[i], -32767);
                    EXPECT_LE(block.qs[i], 32767);
                }
            }
        }
    }

    // Verify K output has valid values
    for (int t = 0; t < seq_len; ++t)
    {
        for (int h = 0; h < n_kv_heads; ++h)
        {
            const int idx = t * n_kv_heads * blocks_per_head + h * blocks_per_head;
            for (int b = 0; b < blocks_per_head; ++b)
            {
                const Q16_1Block &block = k_out[idx + b];
                EXPECT_GT(block.d, 0.0f) << "Invalid K scale at t=" << t << " h=" << h << " b=" << b;
            }
        }
    }
}

// ============================================================================
// Test 11: Q8_1 -> Q16_1 RoPE Position Consistency
// ============================================================================

TEST_F(RoPEQ8ToQ16Test, PositionConsistency)
{
    const int head_dim = 128;
    const int blocks_per_head = head_dim / 32;
    const float rope_theta = 10000.0f;

    // Same input, different positions should give different outputs
    auto q8_blocks = create_random_q8_1_head(head_dim, 789);

    std::vector<Q16_1Block> q16_pos0(blocks_per_head);
    std::vector<Q16_1Block> q16_pos10(blocks_per_head);

    std::vector<int> pos0 = {0};
    std::vector<int> pos10 = {10};

    apply_rope_q8_1_to_q16_1(
        q8_blocks.data(), nullptr,
        q16_pos0.data(), nullptr,
        pos0.data(), 1, 1, 1, head_dim, rope_theta);

    apply_rope_q8_1_to_q16_1(
        q8_blocks.data(), nullptr,
        q16_pos10.data(), nullptr,
        pos10.data(), 1, 1, 1, head_dim, rope_theta);

    // Outputs should differ
    auto fp32_pos0 = dequantize_q16_1(q16_pos0);
    auto fp32_pos10 = dequantize_q16_1(q16_pos10);

    float sum_diff = 0.0f;
    for (size_t i = 0; i < fp32_pos0.size(); ++i)
    {
        sum_diff += std::abs(fp32_pos0[i] - fp32_pos10[i]);
    }

    // Position 0 and 10 should give noticeably different outputs
    EXPECT_GT(sum_diff, 0.1f) << "Different positions should give different outputs";
}