/**
 * @file Test__SoftmaxPrimitives.cpp
 * @brief Unit tests for softmax primitives (all precisions, all SIMD levels)
 * @author David Sanftenberg
 *
 * Tests:
 * 1. FP32 SIMD variant parity (scalar vs AVX2 vs AVX512)
 * 2. BF16 SIMD variant parity
 * 3. FP16 SIMD variant parity
 * 4. Cross-precision parity (BF16/FP16 vs FP32 with conversion tolerance)
 * 5. Causal masking correctness
 * 6. Numerical stability (large/small values)
 * 7. Multi-row batch correctness
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <numeric>
#include <random>

#include "v2/kernels/cpu/primitives/SoftmaxPrimitives_New.h"

using namespace llaminar2::primitives;

namespace
{
    // ============================================================================
    // Test Utilities
    // ============================================================================

    constexpr float FP32_TOLERANCE = 1e-6f;
    constexpr float BF16_TOLERANCE = 5e-3f; // BF16: 7-bit mantissa
    constexpr float FP16_TOLERANCE = 5e-4f; // FP16: 10-bit mantissa

    /**
     * @brief Compare two float arrays with tolerance
     */
    void expect_near_array(
        const float *expected,
        const float *actual,
        int count,
        float tolerance,
        const std::string &msg = "")
    {
        float max_diff = 0.0f;
        int mismatch_idx = -1;

        for (int i = 0; i < count; ++i)
        {
            float diff = std::abs(expected[i] - actual[i]);
            if (diff > max_diff)
            {
                max_diff = diff;
                mismatch_idx = i;
            }
        }

        EXPECT_LE(max_diff, tolerance)
            << msg << " - Max diff: " << max_diff
            << " at index " << mismatch_idx
            << " (expected: " << expected[mismatch_idx]
            << ", actual: " << actual[mismatch_idx] << ")";
    }

    /**
     * @brief Check that array sums to 1.0 (softmax property)
     */
    void expect_sums_to_one(const float *data, int count, float tolerance = 1e-5f)
    {
        float sum = 0.0f;
        for (int i = 0; i < count; ++i)
        {
            sum += data[i];
        }
        EXPECT_NEAR(sum, 1.0f, tolerance) << "Softmax output should sum to 1.0";
    }

    /**
     * @brief Check that all values are finite and non-negative
     */
    void expect_valid_probabilities(const float *data, int count)
    {
        for (int i = 0; i < count; ++i)
        {
            EXPECT_TRUE(std::isfinite(data[i]))
                << "Value at index " << i << " is not finite: " << data[i];
            EXPECT_GE(data[i], 0.0f)
                << "Value at index " << i << " is negative: " << data[i];
            EXPECT_LE(data[i], 1.0f)
                << "Value at index " << i << " exceeds 1.0: " << data[i];
        }
    }

    /**
     * @brief Convert BF16 buffer to FP32 for comparison
     */
    std::vector<float> bf16_to_fp32_buffer(const uint16_t *bf16, int count)
    {
        std::vector<float> fp32(count);
        for (int i = 0; i < count; ++i)
        {
            uint32_t fp32_bits = static_cast<uint32_t>(bf16[i]) << 16;
            std::memcpy(&fp32[i], &fp32_bits, sizeof(float));
        }
        return fp32;
    }

    /**
     * @brief Convert FP16 buffer to FP32 for comparison
     */
    std::vector<float> fp16_to_fp32_buffer(const uint16_t *fp16, int count)
    {
        std::vector<float> fp32(count);
        for (int i = 0; i < count; ++i)
        {
#if defined(__F16C__)
            __m128i vec = _mm_cvtsi32_si128(fp16[i]);
            __m128 fp32_vec = _mm_cvtph_ps(vec);
            fp32[i] = _mm_cvtss_f32(fp32_vec);
#else
            // Manual conversion
            uint16_t h = fp16[i];
            uint32_t sign = (h & 0x8000) << 16;
            uint32_t exp = (h & 0x7C00) >> 10;
            uint32_t mant = (h & 0x03FF);

            uint32_t fp32_bits;
            if (exp == 0)
            {
                if (mant == 0)
                {
                    fp32_bits = sign;
                }
                else
                {
                    exp = 1;
                    while ((mant & 0x0400) == 0)
                    {
                        mant <<= 1;
                        exp--;
                    }
                    mant &= 0x03FF;
                    fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
                }
            }
            else if (exp == 0x1F)
            {
                fp32_bits = sign | 0x7F800000 | (mant << 13);
            }
            else
            {
                fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
            }

            std::memcpy(&fp32[i], &fp32_bits, sizeof(float));
#endif
        }
        return fp32;
    }

} // anonymous namespace

// ============================================================================
// FP32 Softmax Tests
// ============================================================================

class SoftmaxPrimitives_FP32 : public ::testing::Test
{
protected:
    static constexpr int COLS = 128; // Typical attention dimension

    std::vector<float> create_test_data()
    {
        std::vector<float> data(COLS);
        for (int i = 0; i < COLS; ++i)
        {
            data[i] = static_cast<float>(i) / 10.0f; // 0.0, 0.1, 0.2, ...
        }
        return data;
    }
};

TEST_F(SoftmaxPrimitives_FP32, ScalarBasicCorrectness)
{
    auto data = create_test_data();
    softmax_row_fp32_scalar(data.data(), COLS, false, 1.0f, 0);

    expect_valid_probabilities(data.data(), COLS);
    expect_sums_to_one(data.data(), COLS);
}

TEST_F(SoftmaxPrimitives_FP32, ScalarVsAVX2)
{
    auto data_scalar = create_test_data();
    auto data_avx2 = data_scalar;

    softmax_row_fp32_scalar(data_scalar.data(), COLS, false, 1.0f, 0);
    softmax_row_fp32_avx2(data_avx2.data(), COLS, false, 1.0f, 0);

    expect_near_array(data_scalar.data(), data_avx2.data(), COLS, FP32_TOLERANCE,
                      "FP32 Scalar vs AVX2");
}

TEST_F(SoftmaxPrimitives_FP32, ScalarVsAVX512)
{
    auto data_scalar = create_test_data();
    auto data_avx512 = data_scalar;

    softmax_row_fp32_scalar(data_scalar.data(), COLS, false, 1.0f, 0);
    softmax_row_fp32_avx512(data_avx512.data(), COLS, false, 1.0f, 0);

    expect_near_array(data_scalar.data(), data_avx512.data(), COLS, FP32_TOLERANCE,
                      "FP32 Scalar vs AVX512");
}

TEST_F(SoftmaxPrimitives_FP32, AVX2VsAVX512)
{
    auto data_avx2 = create_test_data();
    auto data_avx512 = data_avx2;

    softmax_row_fp32_avx2(data_avx2.data(), COLS, false, 1.0f, 0);
    softmax_row_fp32_avx512(data_avx512.data(), COLS, false, 1.0f, 0);

    expect_near_array(data_avx2.data(), data_avx512.data(), COLS, FP32_TOLERANCE,
                      "FP32 AVX2 vs AVX512");
}

TEST_F(SoftmaxPrimitives_FP32, DispatchMatchesScalar)
{
    auto data_scalar = create_test_data();
    auto data_dispatch = data_scalar;

    softmax_row_fp32_scalar(data_scalar.data(), COLS, false, 1.0f, 0);
    softmax_row_fp32(data_dispatch.data(), COLS, false, 1.0f, 0);

    expect_near_array(data_scalar.data(), data_dispatch.data(), COLS, FP32_TOLERANCE,
                      "FP32 Dispatch vs Scalar");
}

TEST_F(SoftmaxPrimitives_FP32, CausalMasking)
{
    auto data = create_test_data();
    const int row_idx = 64; // Midpoint

    softmax_row_fp32(data.data(), COLS, true, 1.0f, row_idx);

    // Elements after row_idx should be zero
    for (int i = row_idx + 1; i < COLS; ++i)
    {
        EXPECT_EQ(data[i], 0.0f) << "Causal mask failed at index " << i;
    }

    // Elements up to row_idx should sum to 1.0
    float sum = 0.0f;
    for (int i = 0; i <= row_idx; ++i)
    {
        sum += data[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST_F(SoftmaxPrimitives_FP32, NumericalStability_LargeValues)
{
    std::vector<float> data = {100.0f, 200.0f, 300.0f, 400.0f, 500.0f};
    softmax_row_fp32(data.data(), 5, false, 1.0f, 0);

    expect_valid_probabilities(data.data(), 5);
    expect_sums_to_one(data.data(), 5);

    // Largest value should have probability close to 1.0
    EXPECT_GT(data[4], 0.999f) << "Largest value should dominate";
}

TEST_F(SoftmaxPrimitives_FP32, NumericalStability_SmallValues)
{
    std::vector<float> data = {-500.0f, -400.0f, -300.0f, -200.0f, -100.0f};
    softmax_row_fp32(data.data(), 5, false, 1.0f, 0);

    expect_valid_probabilities(data.data(), 5);
    expect_sums_to_one(data.data(), 5);

    // Largest value (least negative) should dominate
    EXPECT_GT(data[4], 0.999f);
}

TEST_F(SoftmaxPrimitives_FP32, ScaleParameter)
{
    auto data_scale1 = create_test_data();
    auto data_scale2 = data_scale1;

    softmax_row_fp32(data_scale1.data(), COLS, false, 1.0f, 0);
    softmax_row_fp32(data_scale2.data(), COLS, false, 2.0f, 0); // 2× scale

    // With larger scale, distribution should be sharper (higher entropy difference)
    // Not equal, but both should be valid probability distributions
    expect_valid_probabilities(data_scale1.data(), COLS);
    expect_valid_probabilities(data_scale2.data(), COLS);
    expect_sums_to_one(data_scale1.data(), COLS);
    expect_sums_to_one(data_scale2.data(), COLS);
}

TEST_F(SoftmaxPrimitives_FP32, UnalignedSize)
{
    // Test with size not divisible by SIMD width (17 elements)
    std::vector<float> data_scalar(17);
    std::vector<float> data_avx2(17);
    std::vector<float> data_avx512(17);

    for (int i = 0; i < 17; ++i)
    {
        data_scalar[i] = static_cast<float>(i);
        data_avx2[i] = static_cast<float>(i);
        data_avx512[i] = static_cast<float>(i);
    }

    softmax_row_fp32_scalar(data_scalar.data(), 17, false, 1.0f, 0);
    softmax_row_fp32_avx2(data_avx2.data(), 17, false, 1.0f, 0);
    softmax_row_fp32_avx512(data_avx512.data(), 17, false, 1.0f, 0);

    expect_near_array(data_scalar.data(), data_avx2.data(), 17, FP32_TOLERANCE,
                      "Unaligned AVX2 vs Scalar");
    expect_near_array(data_scalar.data(), data_avx512.data(), 17, FP32_TOLERANCE,
                      "Unaligned AVX512 vs Scalar");
}

// ============================================================================
// BF16 Softmax Tests
// ============================================================================

class SoftmaxPrimitives_BF16 : public ::testing::Test
{
protected:
    static constexpr int COLS = 128;

    std::vector<uint16_t> create_test_data_bf16()
    {
        std::vector<uint16_t> data(COLS);
        for (int i = 0; i < COLS; ++i)
        {
            float fp32_val = static_cast<float>(i) / 10.0f;
            uint32_t fp32_bits;
            std::memcpy(&fp32_bits, &fp32_val, sizeof(float));
            // Round to nearest even
            uint32_t rounding_bias = 0x7FFF + ((fp32_bits >> 16) & 1);
            uint32_t rounded = fp32_bits + rounding_bias;
            data[i] = static_cast<uint16_t>(rounded >> 16);
        }
        return data;
    }
};

TEST_F(SoftmaxPrimitives_BF16, ScalarBasicCorrectness)
{
    auto data = create_test_data_bf16();
    softmax_row_bf16_scalar(data.data(), COLS, false, 1.0f, 0);

    auto fp32_result = bf16_to_fp32_buffer(data.data(), COLS);
    expect_valid_probabilities(fp32_result.data(), COLS);
    expect_sums_to_one(fp32_result.data(), COLS, BF16_TOLERANCE);
}

TEST_F(SoftmaxPrimitives_BF16, ScalarVsAVX2)
{
    auto data_scalar = create_test_data_bf16();
    auto data_avx2 = data_scalar;

    softmax_row_bf16_scalar(data_scalar.data(), COLS, false, 1.0f, 0);
    softmax_row_bf16_avx2(data_avx2.data(), COLS, false, 1.0f, 0);

    auto fp32_scalar = bf16_to_fp32_buffer(data_scalar.data(), COLS);
    auto fp32_avx2 = bf16_to_fp32_buffer(data_avx2.data(), COLS);

    expect_near_array(fp32_scalar.data(), fp32_avx2.data(), COLS, BF16_TOLERANCE,
                      "BF16 Scalar vs AVX2");
}

TEST_F(SoftmaxPrimitives_BF16, ScalarVsAVX512)
{
    auto data_scalar = create_test_data_bf16();
    auto data_avx512 = data_scalar;

    softmax_row_bf16_scalar(data_scalar.data(), COLS, false, 1.0f, 0);
    softmax_row_bf16_avx512(data_avx512.data(), COLS, false, 1.0f, 0);

    auto fp32_scalar = bf16_to_fp32_buffer(data_scalar.data(), COLS);
    auto fp32_avx512 = bf16_to_fp32_buffer(data_avx512.data(), COLS);

    expect_near_array(fp32_scalar.data(), fp32_avx512.data(), COLS, BF16_TOLERANCE,
                      "BF16 Scalar vs AVX512");
}

TEST_F(SoftmaxPrimitives_BF16, CausalMasking)
{
    auto data = create_test_data_bf16();
    const int row_idx = 64;

    softmax_row_bf16(data.data(), COLS, true, 1.0f, row_idx);

    auto fp32_result = bf16_to_fp32_buffer(data.data(), COLS);

    // Elements after row_idx should be zero
    for (int i = row_idx + 1; i < COLS; ++i)
    {
        EXPECT_NEAR(fp32_result[i], 0.0f, BF16_TOLERANCE)
            << "BF16 causal mask failed at index " << i;
    }

    // Elements up to row_idx should sum to 1.0
    float sum = 0.0f;
    for (int i = 0; i <= row_idx; ++i)
    {
        sum += fp32_result[i];
    }
    EXPECT_NEAR(sum, 1.0f, BF16_TOLERANCE);
}

// ============================================================================
// FP16 Softmax Tests
// ============================================================================

class SoftmaxPrimitives_FP16 : public ::testing::Test
{
protected:
    static constexpr int COLS = 128;

    std::vector<uint16_t> create_test_data_fp16()
    {
        std::vector<uint16_t> data(COLS);
        for (int i = 0; i < COLS; ++i)
        {
            float fp32_val = static_cast<float>(i) / 10.0f;

#if defined(__F16C__)
            __m128 fp32_vec = _mm_set_ss(fp32_val);
            __m128i fp16_vec = _mm_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
            data[i] = static_cast<uint16_t>(_mm_cvtsi128_si32(fp16_vec));
#else
            // Manual conversion
            uint32_t fp32_bits;
            std::memcpy(&fp32_bits, &fp32_val, sizeof(float));

            uint32_t sign = (fp32_bits & 0x80000000) >> 16;
            int32_t exp = ((fp32_bits & 0x7F800000) >> 23) - 127 + 15;
            uint32_t mant = (fp32_bits & 0x007FFFFF);

            uint16_t fp16;
            if (exp <= 0)
            {
                if (exp < -10)
                {
                    fp16 = static_cast<uint16_t>(sign);
                }
                else
                {
                    mant |= 0x00800000;
                    mant >>= (1 - exp);
                    fp16 = static_cast<uint16_t>(sign | (mant >> 13));
                }
            }
            else if (exp >= 0x1F)
            {
                fp16 = static_cast<uint16_t>(sign | 0x7C00);
            }
            else
            {
                fp16 = static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
            }

            data[i] = fp16;
#endif
        }
        return data;
    }
};

TEST_F(SoftmaxPrimitives_FP16, ScalarBasicCorrectness)
{
    auto data = create_test_data_fp16();
    softmax_row_fp16_scalar(data.data(), COLS, false, 1.0f, 0);

    auto fp32_result = fp16_to_fp32_buffer(data.data(), COLS);
    expect_valid_probabilities(fp32_result.data(), COLS);
    expect_sums_to_one(fp32_result.data(), COLS, FP16_TOLERANCE);
}

TEST_F(SoftmaxPrimitives_FP16, ScalarVsAVX2)
{
    auto data_scalar = create_test_data_fp16();
    auto data_avx2 = data_scalar;

    softmax_row_fp16_scalar(data_scalar.data(), COLS, false, 1.0f, 0);
    softmax_row_fp16_avx2(data_avx2.data(), COLS, false, 1.0f, 0);

    auto fp32_scalar = fp16_to_fp32_buffer(data_scalar.data(), COLS);
    auto fp32_avx2 = fp16_to_fp32_buffer(data_avx2.data(), COLS);

    expect_near_array(fp32_scalar.data(), fp32_avx2.data(), COLS, FP16_TOLERANCE,
                      "FP16 Scalar vs AVX2");
}

TEST_F(SoftmaxPrimitives_FP16, ScalarVsAVX512)
{
    auto data_scalar = create_test_data_fp16();
    auto data_avx512 = data_scalar;

    softmax_row_fp16_scalar(data_scalar.data(), COLS, false, 1.0f, 0);
    softmax_row_fp16_avx512(data_avx512.data(), COLS, false, 1.0f, 0);

    auto fp32_scalar = fp16_to_fp32_buffer(data_scalar.data(), COLS);
    auto fp32_avx512 = fp16_to_fp32_buffer(data_avx512.data(), COLS);

    expect_near_array(fp32_scalar.data(), fp32_avx512.data(), COLS, FP16_TOLERANCE,
                      "FP16 Scalar vs AVX512");
}

TEST_F(SoftmaxPrimitives_FP16, CausalMasking)
{
    auto data = create_test_data_fp16();
    const int row_idx = 64;

    softmax_row_fp16(data.data(), COLS, true, 1.0f, row_idx);

    auto fp32_result = fp16_to_fp32_buffer(data.data(), COLS);

    // Elements after row_idx should be zero
    for (int i = row_idx + 1; i < COLS; ++i)
    {
        EXPECT_NEAR(fp32_result[i], 0.0f, FP16_TOLERANCE)
            << "FP16 causal mask failed at index " << i;
    }

    // Elements up to row_idx should sum to 1.0
    float sum = 0.0f;
    for (int i = 0; i <= row_idx; ++i)
    {
        sum += fp32_result[i];
    }
    EXPECT_NEAR(sum, 1.0f, FP16_TOLERANCE);
}

// ============================================================================
// Cross-Precision Parity Tests
// ============================================================================

class SoftmaxPrimitives_CrossPrecision : public ::testing::Test
{
protected:
    static constexpr int COLS = 128;

    std::vector<float> create_test_data_fp32()
    {
        std::vector<float> data(COLS);
        for (int i = 0; i < COLS; ++i)
        {
            data[i] = static_cast<float>(i) / 10.0f;
        }
        return data;
    }

    std::vector<uint16_t> fp32_to_bf16(const std::vector<float> &fp32)
    {
        std::vector<uint16_t> bf16(fp32.size());
        for (size_t i = 0; i < fp32.size(); ++i)
        {
            uint32_t fp32_bits;
            std::memcpy(&fp32_bits, &fp32[i], sizeof(float));
            uint32_t rounding_bias = 0x7FFF + ((fp32_bits >> 16) & 1);
            uint32_t rounded = fp32_bits + rounding_bias;
            bf16[i] = static_cast<uint16_t>(rounded >> 16);
        }
        return bf16;
    }

    std::vector<uint16_t> fp32_to_fp16(const std::vector<float> &fp32)
    {
        std::vector<uint16_t> fp16(fp32.size());
        for (size_t i = 0; i < fp32.size(); ++i)
        {
#if defined(__F16C__)
            __m128 fp32_vec = _mm_set_ss(fp32[i]);
            __m128i fp16_vec = _mm_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
            fp16[i] = static_cast<uint16_t>(_mm_cvtsi128_si32(fp16_vec));
#else
            // Manual conversion (same as create_test_data_fp16)
            uint32_t fp32_bits;
            std::memcpy(&fp32_bits, &fp32[i], sizeof(float));
            uint32_t sign = (fp32_bits & 0x80000000) >> 16;
            int32_t exp = ((fp32_bits & 0x7F800000) >> 23) - 127 + 15;
            uint32_t mant = (fp32_bits & 0x007FFFFF);

            if (exp <= 0)
            {
                if (exp < -10)
                {
                    fp16[i] = static_cast<uint16_t>(sign);
                }
                else
                {
                    mant |= 0x00800000;
                    mant >>= (1 - exp);
                    fp16[i] = static_cast<uint16_t>(sign | (mant >> 13));
                }
            }
            else if (exp >= 0x1F)
            {
                fp16[i] = static_cast<uint16_t>(sign | 0x7C00);
            }
            else
            {
                fp16[i] = static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
            }
#endif
        }
        return fp16;
    }
};

TEST_F(SoftmaxPrimitives_CrossPrecision, FP32VsBF16)
{
    auto data_fp32 = create_test_data_fp32();
    auto data_bf16 = fp32_to_bf16(data_fp32);

    softmax_row_fp32(data_fp32.data(), COLS, false, 1.0f, 0);
    softmax_row_bf16(data_bf16.data(), COLS, false, 1.0f, 0);

    auto bf16_as_fp32 = bf16_to_fp32_buffer(data_bf16.data(), COLS);

    expect_near_array(data_fp32.data(), bf16_as_fp32.data(), COLS, BF16_TOLERANCE,
                      "FP32 vs BF16 (with conversion tolerance)");
}

TEST_F(SoftmaxPrimitives_CrossPrecision, FP32VsFP16)
{
    auto data_fp32 = create_test_data_fp32();
    auto data_fp16 = fp32_to_fp16(data_fp32);

    softmax_row_fp32(data_fp32.data(), COLS, false, 1.0f, 0);
    softmax_row_fp16(data_fp16.data(), COLS, false, 1.0f, 0);

    auto fp16_as_fp32 = fp16_to_fp32_buffer(data_fp16.data(), COLS);

    expect_near_array(data_fp32.data(), fp16_as_fp32.data(), COLS, FP16_TOLERANCE,
                      "FP32 vs FP16 (with conversion tolerance)");
}

TEST_F(SoftmaxPrimitives_CrossPrecision, BF16VsFP16)
{
    auto data_fp32_src = create_test_data_fp32();
    auto data_bf16 = fp32_to_bf16(data_fp32_src);
    auto data_fp16 = fp32_to_fp16(data_fp32_src);

    softmax_row_bf16(data_bf16.data(), COLS, false, 1.0f, 0);
    softmax_row_fp16(data_fp16.data(), COLS, false, 1.0f, 0);

    auto bf16_as_fp32 = bf16_to_fp32_buffer(data_bf16.data(), COLS);
    auto fp16_as_fp32 = fp16_to_fp32_buffer(data_fp16.data(), COLS);

    // BF16 vs FP16: Both have precision loss, so use larger tolerance
    expect_near_array(bf16_as_fp32.data(), fp16_as_fp32.data(), COLS, BF16_TOLERANCE,
                      "BF16 vs FP16 (both reduced precision)");
}

// ============================================================================
// Multi-Row Batch Tests
// ============================================================================

class SoftmaxPrimitives_MultiRow : public ::testing::Test
{
protected:
    static constexpr int ROWS = 32;
    static constexpr int COLS = 128;

    std::vector<float> create_batch_data()
    {
        std::vector<float> data(ROWS * COLS);
        for (int r = 0; r < ROWS; ++r)
        {
            for (int c = 0; c < COLS; ++c)
            {
                data[r * COLS + c] = static_cast<float>(r * COLS + c) / 100.0f;
            }
        }
        return data;
    }
};

TEST_F(SoftmaxPrimitives_MultiRow, FP32_BatchCorrectness)
{
    auto data = create_batch_data();
    softmax_row_major_fp32(data.data(), ROWS, COLS, false, 1.0f, true);

    // Each row should sum to 1.0
    for (int r = 0; r < ROWS; ++r)
    {
        float *row = data.data() + r * COLS;
        expect_valid_probabilities(row, COLS);
        expect_sums_to_one(row, COLS);
    }
}

TEST_F(SoftmaxPrimitives_MultiRow, FP32_BatchVsSingleRow)
{
    auto data_batch = create_batch_data();
    auto data_single = data_batch;

    // Batch processing
    softmax_row_major_fp32(data_batch.data(), ROWS, COLS, false, 1.0f, true);

    // Single-row processing
    for (int r = 0; r < ROWS; ++r)
    {
        softmax_row_fp32(data_single.data() + r * COLS, COLS, false, 1.0f, r);
    }

    expect_near_array(data_batch.data(), data_single.data(), ROWS * COLS, FP32_TOLERANCE,
                      "Batch vs single-row processing");
}

TEST_F(SoftmaxPrimitives_MultiRow, FP32_CausalBatch)
{
    auto data = create_batch_data();
    softmax_row_major_fp32(data.data(), ROWS, COLS, true, 1.0f, true);

    // Each row r should have zeros after column r
    for (int r = 0; r < ROWS; ++r)
    {
        for (int c = r + 1; c < COLS; ++c)
        {
            EXPECT_EQ(data[r * COLS + c], 0.0f)
                << "Causal masking failed at row " << r << ", col " << c;
        }
    }
}

TEST_F(SoftmaxPrimitives_MultiRow, BF16_BatchCorrectness)
{
    auto fp32_data = create_batch_data();
    std::vector<uint16_t> bf16_data(ROWS * COLS);

    // Convert to BF16
    for (int i = 0; i < ROWS * COLS; ++i)
    {
        uint32_t fp32_bits;
        std::memcpy(&fp32_bits, &fp32_data[i], sizeof(float));
        uint32_t rounding_bias = 0x7FFF + ((fp32_bits >> 16) & 1);
        uint32_t rounded = fp32_bits + rounding_bias;
        bf16_data[i] = static_cast<uint16_t>(rounded >> 16);
    }

    softmax_row_major_bf16(bf16_data.data(), ROWS, COLS, false, 1.0f, true);

    // Each row should sum to ~1.0
    for (int r = 0; r < ROWS; ++r)
    {
        auto row_fp32 = bf16_to_fp32_buffer(bf16_data.data() + r * COLS, COLS);
        expect_valid_probabilities(row_fp32.data(), COLS);
        expect_sums_to_one(row_fp32.data(), COLS, BF16_TOLERANCE);
    }
}

TEST_F(SoftmaxPrimitives_MultiRow, FP16_BatchCorrectness)
{
    auto fp32_data = create_batch_data();
    std::vector<uint16_t> fp16_data(ROWS * COLS);

    // Convert to FP16
    for (int i = 0; i < ROWS * COLS; ++i)
    {
#if defined(__F16C__)
        __m128 fp32_vec = _mm_set_ss(fp32_data[i]);
        __m128i fp16_vec = _mm_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
        fp16_data[i] = static_cast<uint16_t>(_mm_cvtsi128_si32(fp16_vec));
#else
        // Manual conversion
        uint32_t fp32_bits;
        std::memcpy(&fp32_bits, &fp32_data[i], sizeof(float));
        uint32_t sign = (fp32_bits & 0x80000000) >> 16;
        int32_t exp = ((fp32_bits & 0x7F800000) >> 23) - 127 + 15;
        uint32_t mant = (fp32_bits & 0x007FFFFF);

        if (exp <= 0)
        {
            if (exp < -10)
                fp16_data[i] = static_cast<uint16_t>(sign);
            else
            {
                mant |= 0x00800000;
                mant >>= (1 - exp);
                fp16_data[i] = static_cast<uint16_t>(sign | (mant >> 13));
            }
        }
        else if (exp >= 0x1F)
        {
            fp16_data[i] = static_cast<uint16_t>(sign | 0x7C00);
        }
        else
        {
            fp16_data[i] = static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
        }
#endif
    }

    softmax_row_major_fp16(fp16_data.data(), ROWS, COLS, false, 1.0f, true);

    // Each row should sum to ~1.0
    for (int r = 0; r < ROWS; ++r)
    {
        auto row_fp32 = fp16_to_fp32_buffer(fp16_data.data() + r * COLS, COLS);
        expect_valid_probabilities(row_fp32.data(), COLS);
        expect_sums_to_one(row_fp32.data(), COLS, FP16_TOLERANCE);
    }
}

// ============================================================================
// ISA Parity Tests — detail:: softmax_row variants (scalar vs avx2 vs avx512)
// ============================================================================

#include "kernels/cpu/primitives/SoftmaxPrimitives_detail.h"

using namespace llaminar2::primitives::detail;

TEST(SoftmaxPrimitives_ISAParity, SoftmaxRow_ScalarVsAVX512)
{
    constexpr int COLS = 256;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    // aligned buffers
    alignas(64) float scalar_buf[COLS];
    alignas(64) float simd_buf[COLS];

    for (int i = 0; i < COLS; ++i)
    {
        float v = dist(rng);
        scalar_buf[i] = v;
        simd_buf[i] = v;
    }

    // use_fast_exp=false for exact scalar exp() parity
    softmax_row_scalar(scalar_buf, COLS, /*r=*/0, /*causal=*/false,
                       /*scale=*/1.0f, /*use_fast_exp=*/false);

#if defined(__AVX512F__)
    softmax_row_avx512(simd_buf, COLS, /*r=*/0, /*causal=*/false,
                       /*scale=*/1.0f, /*use_fast_exp=*/false);

    float max_diff = 0.0f;
    for (int i = 0; i < COLS; ++i)
    {
        max_diff = std::max(max_diff, std::abs(scalar_buf[i] - simd_buf[i]));
    }
    EXPECT_LE(max_diff, 1e-5f) << "Scalar vs AVX512 softmax_row mismatch";
#elif defined(__AVX2__)
    softmax_row_avx2(simd_buf, COLS, /*r=*/0, /*causal=*/false,
                     /*scale=*/1.0f, /*use_fast_exp=*/false);

    float max_diff = 0.0f;
    for (int i = 0; i < COLS; ++i)
    {
        max_diff = std::max(max_diff, std::abs(scalar_buf[i] - simd_buf[i]));
    }
    EXPECT_LE(max_diff, 1e-5f) << "Scalar vs AVX2 softmax_row mismatch";
#else
    // Scalar-only — nothing extra to compare
#endif
}

TEST(SoftmaxPrimitives_ISAParity, SoftmaxRow_CausalAndScale)
{
    constexpr int COLS = 128;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    alignas(64) float scalar_buf[COLS];
    alignas(64) float simd_buf[COLS];

    for (int i = 0; i < COLS; ++i)
    {
        float v = dist(rng);
        scalar_buf[i] = v;
        simd_buf[i] = v;
    }

    // Row 64 with causal — positions 0..64 active, 65..127 masked
    const float scale = 0.125f;
    softmax_row_scalar(scalar_buf, COLS, /*r=*/64, /*causal=*/true,
                       scale, /*use_fast_exp=*/false);

#if defined(__AVX512F__)
    softmax_row_avx512(simd_buf, COLS, /*r=*/64, /*causal=*/true,
                       scale, /*use_fast_exp=*/false);

    float max_diff = 0.0f;
    for (int i = 0; i < COLS; ++i)
    {
        max_diff = std::max(max_diff, std::abs(scalar_buf[i] - simd_buf[i]));
    }
    EXPECT_LE(max_diff, 1e-5f) << "Scalar vs AVX512 causal softmax mismatch";
#elif defined(__AVX2__)
    softmax_row_avx2(simd_buf, COLS, /*r=*/64, /*causal=*/true,
                     scale, /*use_fast_exp=*/false);

    float max_diff = 0.0f;
    for (int i = 0; i < COLS; ++i)
    {
        max_diff = std::max(max_diff, std::abs(scalar_buf[i] - simd_buf[i]));
    }
    EXPECT_LE(max_diff, 1e-5f) << "Scalar vs AVX2 causal softmax mismatch";
#endif
}
