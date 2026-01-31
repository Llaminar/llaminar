/**
 * @file Test__BF16_SIMD_Accuracy.cpp
 * @brief Accuracy tests for SIMD-optimized BF16 conversion
 *
 * @author David Sanftenberg
 * @date November 8, 2025
 *
 * Tests verify that AVX-512, AVX2, and scalar BF16 conversion methods
 * produce identical results across various input patterns.
 */

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include "utils/BFloat16.h"
#include "tensors/SIMDHelpers.h"

using namespace llaminar2;
using namespace llaminar2::simd;

namespace
{

    /**
     * @brief Initialize random test data with various patterns
     */
    void init_test_data(std::vector<float> &data, const std::string &pattern)
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> uniform(-10.0f, 10.0f);
        std::normal_distribution<float> normal(0.0f, 1.0f);

        if (pattern == "uniform")
        {
            for (auto &val : data)
            {
                val = uniform(rng);
            }
        }
        else if (pattern == "normal")
        {
            for (auto &val : data)
            {
                val = normal(rng);
            }
        }
        else if (pattern == "small")
        {
            std::uniform_real_distribution<float> small(-1.0f, 1.0f);
            for (auto &val : data)
            {
                val = small(rng);
            }
        }
        else if (pattern == "large")
        {
            std::uniform_real_distribution<float> large(-1e6f, 1e6f);
            for (auto &val : data)
            {
                val = large(rng);
            }
        }
        else if (pattern == "mixed")
        {
            for (size_t i = 0; i < data.size(); ++i)
            {
                if (i % 4 == 0)
                    data[i] = 0.0f;
                else if (i % 4 == 1)
                    data[i] = uniform(rng);
                else if (i % 4 == 2)
                    data[i] = std::numeric_limits<float>::infinity();
                else
                    data[i] = -std::numeric_limits<float>::infinity();
            }
        }
        else if (pattern == "subnormal")
        {
            for (auto &val : data)
            {
                val = std::pow(2.0f, -126.0f) * uniform(rng);
            }
        }
    }

    /**
     * @brief Compare two BF16 arrays for bitwise equality
     */
    bool compare_bf16_arrays(const uint16_t *a, const uint16_t *b, size_t count,
                             std::string &error_msg)
    {
        size_t mismatch_count = 0;
        size_t first_mismatch = 0;

        for (size_t i = 0; i < count; ++i)
        {
            if (a[i] != b[i])
            {
                if (mismatch_count == 0)
                {
                    first_mismatch = i;
                }
                mismatch_count++;
            }
        }

        if (mismatch_count > 0)
        {
            error_msg = "Found " + std::to_string(mismatch_count) + " mismatches out of " +
                        std::to_string(count) + " elements. First at index " +
                        std::to_string(first_mismatch) +
                        " (expected: 0x" + std::to_string(a[first_mismatch]) +
                        ", got: 0x" + std::to_string(b[first_mismatch]) + ")";
            return false;
        }

        return true;
    }

} // anonymous namespace

// ============================================================================
// AVX-512 vs Scalar Tests
// ============================================================================

#if defined(__AVX512F__) && defined(__AVX512BW__)

TEST(BF16_SIMD_Accuracy, AVX512_vs_Scalar_Uniform)
{
    constexpr size_t count = 1024; // 32 AVX-512 iterations
    std::vector<float> input(count);
    init_test_data(input, "uniform");

    std::vector<uint16_t> output_avx512(count);
    std::vector<uint16_t> output_scalar(count);

    convert_fp32_to_bf16_avx512(input.data(), output_avx512.data(), count);
    convert_fp32_to_bf16_scalar(input.data(), output_scalar.data(), count);

    std::string error_msg;
    EXPECT_TRUE(compare_bf16_arrays(output_scalar.data(), output_avx512.data(), count, error_msg))
        << error_msg;
}

TEST(BF16_SIMD_Accuracy, AVX512_vs_Scalar_Small)
{
    constexpr size_t count = 512;
    std::vector<float> input(count);
    init_test_data(input, "small");

    std::vector<uint16_t> output_avx512(count);
    std::vector<uint16_t> output_scalar(count);

    convert_fp32_to_bf16_avx512(input.data(), output_avx512.data(), count);
    convert_fp32_to_bf16_scalar(input.data(), output_scalar.data(), count);

    std::string error_msg;
    EXPECT_TRUE(compare_bf16_arrays(output_scalar.data(), output_avx512.data(), count, error_msg))
        << error_msg;
}

TEST(BF16_SIMD_Accuracy, AVX512_vs_Scalar_Large)
{
    constexpr size_t count = 2048;
    std::vector<float> input(count);
    init_test_data(input, "large");

    std::vector<uint16_t> output_avx512(count);
    std::vector<uint16_t> output_scalar(count);

    convert_fp32_to_bf16_avx512(input.data(), output_avx512.data(), count);
    convert_fp32_to_bf16_scalar(input.data(), output_scalar.data(), count);

    std::string error_msg;
    EXPECT_TRUE(compare_bf16_arrays(output_scalar.data(), output_avx512.data(), count, error_msg))
        << error_msg;
}

TEST(BF16_SIMD_Accuracy, AVX512_vs_Scalar_Special)
{
    constexpr size_t count = 512;
    std::vector<float> input = {
        0.0f, -0.0f, 1.0f, -1.0f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::epsilon(),
        1e-10f, 1e10f, -1e-10f, -1e10f,
        0.33333333f, 0.66666667f, 0.99999999f};
    input.resize(count, 1.0f);

    std::vector<uint16_t> output_avx512(count);
    std::vector<uint16_t> output_scalar(count);

    convert_fp32_to_bf16_avx512(input.data(), output_avx512.data(), count);
    convert_fp32_to_bf16_scalar(input.data(), output_scalar.data(), count);

    std::string error_msg;
    EXPECT_TRUE(compare_bf16_arrays(output_scalar.data(), output_avx512.data(), count, error_msg))
        << error_msg;
}

#endif // AVX512

// ============================================================================
// AVX2 vs Scalar Tests
// ============================================================================

#if defined(__AVX2__)

TEST(BF16_SIMD_Accuracy, AVX2_vs_Scalar_Uniform)
{
    constexpr size_t count = 1024; // 64 AVX2 iterations
    std::vector<float> input(count);
    init_test_data(input, "uniform");

    std::vector<uint16_t> output_avx2(count);
    std::vector<uint16_t> output_scalar(count);

    convert_fp32_to_bf16_avx2(input.data(), output_avx2.data(), count);
    convert_fp32_to_bf16_scalar(input.data(), output_scalar.data(), count);

    std::string error_msg;
    EXPECT_TRUE(compare_bf16_arrays(output_scalar.data(), output_avx2.data(), count, error_msg))
        << error_msg;
}

TEST(BF16_SIMD_Accuracy, AVX2_vs_Scalar_Normal)
{
    constexpr size_t count = 512;
    std::vector<float> input(count);
    init_test_data(input, "normal");

    std::vector<uint16_t> output_avx2(count);
    std::vector<uint16_t> output_scalar(count);

    convert_fp32_to_bf16_avx2(input.data(), output_avx2.data(), count);
    convert_fp32_to_bf16_scalar(input.data(), output_scalar.data(), count);

    std::string error_msg;
    EXPECT_TRUE(compare_bf16_arrays(output_scalar.data(), output_avx2.data(), count, error_msg))
        << error_msg;
}

TEST(BF16_SIMD_Accuracy, AVX2_vs_Scalar_Large)
{
    constexpr size_t count = 2048;
    std::vector<float> input(count);
    init_test_data(input, "large");

    std::vector<uint16_t> output_avx2(count);
    std::vector<uint16_t> output_scalar(count);

    convert_fp32_to_bf16_avx2(input.data(), output_avx2.data(), count);
    convert_fp32_to_bf16_scalar(input.data(), output_scalar.data(), count);

    std::string error_msg;
    EXPECT_TRUE(compare_bf16_arrays(output_scalar.data(), output_avx2.data(), count, error_msg))
        << error_msg;
}

TEST(BF16_SIMD_Accuracy, AVX2_vs_Scalar_Special)
{
    constexpr size_t count = 512;
    std::vector<float> input = {
        0.0f, -0.0f, 1.0f, -1.0f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::epsilon(),
        1e-10f, 1e10f, -1e-10f, -1e10f};
    input.resize(count, 0.5f);

    std::vector<uint16_t> output_avx2(count);
    std::vector<uint16_t> output_scalar(count);

    convert_fp32_to_bf16_avx2(input.data(), output_avx2.data(), count);
    convert_fp32_to_bf16_scalar(input.data(), output_scalar.data(), count);

    std::string error_msg;
    EXPECT_TRUE(compare_bf16_arrays(output_scalar.data(), output_avx2.data(), count, error_msg))
        << error_msg;
}

#endif // AVX2

// ============================================================================
// Scalar-only Test (always runs)
// ============================================================================

TEST(BF16_SIMD_Accuracy, Scalar_Correctness)
{
    constexpr size_t count = 256;
    std::vector<float> input(count);
    init_test_data(input, "uniform");

    std::vector<uint16_t> output(count);

    convert_fp32_to_bf16_scalar(input.data(), output.data(), count);

    // Verify each conversion individually using bfloat16::from_float
    for (size_t i = 0; i < count; ++i)
    {
        uint16_t expected = fp32_to_bf16(input[i]);
        EXPECT_EQ(output[i], expected)
            << "Mismatch at index " << i << ": input=" << input[i];
    }
}

// ============================================================================
// Round-trip Tests
// ============================================================================

TEST(BF16_SIMD_Accuracy, RoundTrip_Scalar)
{
    std::vector<float> test_values = {
        0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
        1.23456789f, -9.87654321f,
        1e-5f, 1e5f, -1e-5f, -1e5f,
        std::numeric_limits<float>::min()
        // Note: FLT_MAX exceeds BF16 range (max ~3.4e38), so we omit it
    };

    size_t count = test_values.size();
    std::vector<uint16_t> bf16(count);

    convert_fp32_to_bf16_scalar(test_values.data(), bf16.data(), count);

    // Verify that converting back gives expected precision loss
    for (size_t i = 0; i < count; ++i)
    {
        float recovered = bf16_to_fp32(bf16[i]);

        // BF16 has 7-bit mantissa, so relative error should be < 2^-7 (~0.78%)
        if (std::isfinite(test_values[i]) && test_values[i] != 0.0f)
        {
            float rel_error = std::abs((recovered - test_values[i]) / test_values[i]);
            EXPECT_LT(rel_error, 0.01f) << "Value: " << test_values[i]
                                        << " -> " << recovered;
        }
    }
}

// ============================================================================
// BF16 → FP32 Conversion Tests (Lossless Expansion)
// ============================================================================

#if defined(__AVX512F__) && defined(__AVX512BW__)

TEST(BF16_SIMD_Accuracy, BF16toFP32_AVX512_vs_Scalar_Uniform)
{
    constexpr size_t count = 1024; // 32 AVX-512 iterations
    std::vector<uint16_t> bf16_input(count);

    // Generate BF16 values from random FP32
    std::vector<float> fp32_source(count);
    init_test_data(fp32_source, "uniform");
    convert_fp32_to_bf16_scalar(fp32_source.data(), bf16_input.data(), count);

    std::vector<float> output_avx512(count);
    std::vector<float> output_scalar(count);

    convert_bf16_to_fp32_avx512(bf16_input.data(), output_avx512.data(), count);
    convert_bf16_to_fp32_scalar(bf16_input.data(), output_scalar.data(), count);

    // BF16→FP32 is lossless - expect exact bitwise equality
    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(output_avx512[i], output_scalar[i])
            << "Mismatch at index " << i;
    }
}

TEST(BF16_SIMD_Accuracy, BF16toFP32_AVX512_vs_Scalar_Special)
{
    constexpr size_t count = 512;
    std::vector<float> special_values = {
        0.0f, -0.0f, 1.0f, -1.0f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        1e-10f, 1e10f, -1e-10f, -1e10f,
        0.33333333f, 0.66666667f, 0.99999999f};

    std::vector<uint16_t> bf16_input(count);
    // Convert special values to BF16
    for (size_t i = 0; i < special_values.size(); ++i)
    {
        bf16_input[i] = fp32_to_bf16(special_values[i]);
    }
    // Fill rest with pattern
    for (size_t i = special_values.size(); i < count; ++i)
    {
        bf16_input[i] = fp32_to_bf16(static_cast<float>(i) * 0.1f);
    }

    std::vector<float> output_avx512(count);
    std::vector<float> output_scalar(count);

    convert_bf16_to_fp32_avx512(bf16_input.data(), output_avx512.data(), count);
    convert_bf16_to_fp32_scalar(bf16_input.data(), output_scalar.data(), count);

    // Exact bitwise equality (lossless conversion)
    for (size_t i = 0; i < count; ++i)
    {
        // For NaN, check both are NaN (NaN != NaN in IEEE-754)
        if (std::isnan(output_scalar[i]))
        {
            EXPECT_TRUE(std::isnan(output_avx512[i])) << "Index " << i;
        }
        else
        {
            EXPECT_EQ(output_avx512[i], output_scalar[i])
                << "Mismatch at index " << i;
        }
    }
}

TEST(BF16_SIMD_Accuracy, BF16toFP32_AVX512_Lossless)
{
    constexpr size_t count = 2048;
    std::vector<float> original(count);
    init_test_data(original, "normal");

    // FP32 → BF16 → FP32 round-trip
    std::vector<uint16_t> bf16(count);
    std::vector<float> recovered(count);

    convert_fp32_to_bf16_scalar(original.data(), bf16.data(), count);
    convert_bf16_to_fp32_avx512(bf16.data(), recovered.data(), count);

    // Verify BF16→FP32 preserves the quantized value exactly
    for (size_t i = 0; i < count; ++i)
    {
        float expected = bf16_to_fp32(bf16[i]);
        EXPECT_EQ(recovered[i], expected)
            << "Lossless conversion failed at index " << i;
    }
}

#endif // AVX512

// ============================================================================
// BF16 → FP32 AVX2 Tests
// ============================================================================

#if defined(__AVX2__)

TEST(BF16_SIMD_Accuracy, BF16toFP32_AVX2_vs_Scalar_Uniform)
{
    constexpr size_t count = 1024; // 64 AVX2 iterations
    std::vector<uint16_t> bf16_input(count);

    // Generate BF16 values from random FP32
    std::vector<float> fp32_source(count);
    init_test_data(fp32_source, "uniform");
    convert_fp32_to_bf16_scalar(fp32_source.data(), bf16_input.data(), count);

    std::vector<float> output_avx2(count);
    std::vector<float> output_scalar(count);

    convert_bf16_to_fp32_avx2(bf16_input.data(), output_avx2.data(), count);
    convert_bf16_to_fp32_scalar(bf16_input.data(), output_scalar.data(), count);

    // BF16→FP32 is lossless - expect exact bitwise equality
    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(output_avx2[i], output_scalar[i])
            << "Mismatch at index " << i;
    }
}

TEST(BF16_SIMD_Accuracy, BF16toFP32_AVX2_vs_Scalar_Normal)
{
    constexpr size_t count = 512;
    std::vector<uint16_t> bf16_input(count);

    std::vector<float> fp32_source(count);
    init_test_data(fp32_source, "normal");
    convert_fp32_to_bf16_scalar(fp32_source.data(), bf16_input.data(), count);

    std::vector<float> output_avx2(count);
    std::vector<float> output_scalar(count);

    convert_bf16_to_fp32_avx2(bf16_input.data(), output_avx2.data(), count);
    convert_bf16_to_fp32_scalar(bf16_input.data(), output_scalar.data(), count);

    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(output_avx2[i], output_scalar[i])
            << "Mismatch at index " << i;
    }
}

TEST(BF16_SIMD_Accuracy, BF16toFP32_AVX2_Lossless)
{
    constexpr size_t count = 2048;
    std::vector<float> original(count);
    init_test_data(original, "large");

    // FP32 → BF16 → FP32 round-trip
    std::vector<uint16_t> bf16(count);
    std::vector<float> recovered(count);

    convert_fp32_to_bf16_scalar(original.data(), bf16.data(), count);
    convert_bf16_to_fp32_avx2(bf16.data(), recovered.data(), count);

    // Verify BF16→FP32 preserves the quantized value exactly
    for (size_t i = 0; i < count; ++i)
    {
        float expected = bf16_to_fp32(bf16[i]);
        EXPECT_EQ(recovered[i], expected)
            << "Lossless conversion failed at index " << i;
    }
}

#endif // AVX2

// ============================================================================
// BF16 → FP32 Scalar Correctness
// ============================================================================

TEST(BF16_SIMD_Accuracy, BF16toFP32_Scalar_Lossless)
{
    // Test that BF16→FP32 conversion is truly lossless
    std::vector<uint16_t> test_bf16 = {
        0x0000, // 0.0
        0x8000, // -0.0
        0x3F80, // 1.0
        0xBF80, // -1.0
        0x7F80, // +inf
        0xFF80, // -inf
        0x7FC0, // NaN
        0x3E80, // 0.25
        0x4000, // 2.0
        0x447A, // 1000.0
    };

    std::vector<float> output(test_bf16.size());
    convert_bf16_to_fp32_scalar(test_bf16.data(), output.data(), test_bf16.size());

    // Verify each conversion matches scalar implementation
    for (size_t i = 0; i < test_bf16.size(); ++i)
    {
        float expected = bf16_to_fp32(test_bf16[i]);

        if (std::isnan(expected))
        {
            EXPECT_TRUE(std::isnan(output[i])) << "Index " << i;
        }
        else
        {
            EXPECT_EQ(output[i], expected)
                << "Index " << i << ", BF16=0x" << std::hex << test_bf16[i];
        }
    }
}

TEST(BF16_SIMD_Accuracy, BF16toFP32_RoundTrip)
{
    // Verify FP32→BF16→FP32 produces expected quantized result
    std::vector<float> original = {
        1.23456789f, // Should quantize to reduced precision
        3.14159265f,
        2.71828182f,
        0.123456789f,
        -9.87654321f};

    for (float val : original)
    {
        uint16_t bf16 = fp32_to_bf16(val);
        float recovered_scalar = bf16_to_fp32(bf16);

        std::vector<uint16_t> bf16_vec = {bf16};
        std::vector<float> recovered_vec(1);
        convert_bf16_to_fp32_scalar(bf16_vec.data(), recovered_vec.data(), 1);

        // Both methods should produce identical result
        EXPECT_EQ(recovered_scalar, recovered_vec[0])
            << "Original: " << val;

        // Result should have at most 7 mantissa bits
        // (i.e., lower 16 bits of FP32 should be zero)
        uint32_t bits;
        std::memcpy(&bits, &recovered_scalar, sizeof(float));
        EXPECT_EQ(bits & 0xFFFF, 0u)
            << "BF16→FP32 didn't zero-extend properly";
    }
}
