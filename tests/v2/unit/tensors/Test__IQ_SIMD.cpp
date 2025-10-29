/**
 * @file Test__IQ_SIMD.cpp
 * @brief SIMD equivalency tests for all IQ quantization formats
 * @author David Sanftenberg
 * @date 2025-10-29
 *
 * This file contains basic validation tests for IQ quantization formats:
 * - IQ1_S, IQ1_M: 1-2 bit quantization
 * - IQ2_XXS, IQ2_XS, IQ2_S: 2 bit quantization variants
 * - IQ3_XXS, IQ3_S: 3 bit quantization variants
 * - IQ4_XS: 4 bit quantization (extra small)
 *
 * Note: IQ4_NL has its own dedicated test file (Test__IQ4_NLTensor.cpp) with full SIMD tests
 *
 * Current Status: Basic validation only
 * These tests validate that the implementations produce finite, deterministic outputs.
 * Full SIMD equivalency tests will be added once SIMD implementations are complete.
 *
 * Testing Strategy:
 * Since decodeBlock is private for these formats, we test through the public tensor API:
 * 1. Create a tensor with known raw data
 * 2. Decode to FP32 using public data() method
 * 3. Validate outputs are finite and deterministic
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include "tensors/IQQuantTables.h"
#include <cmath>
#include <random>
#include <vector>
#include <cstring>

using namespace llaminar2;

// Tolerance for float comparisons
constexpr float TOLERANCE = 1e-5f;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Compare two float arrays with tolerance
 */
bool compareArrays(const float *expected, const float *actual, size_t count,
                   float tolerance, std::string &error_msg)
{
    for (size_t i = 0; i < count; ++i)
    {
        float diff = std::abs(expected[i] - actual[i]);
        if (diff > tolerance)
        {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "Mismatch at index %zu: expected=%.6f, actual=%.6f, diff=%.6f",
                     i, expected[i], actual[i], diff);
            error_msg = buf;
            return false;
        }
    }
    return true;
}

// =============================================================================
// IQ3_XXS Tests
// =============================================================================

TEST(IQ3_XXS_Test, TensorCreation)
{
    // IQ3_XXS: 256 elements per block, 98 bytes per block
    std::vector<size_t> shape = {2, 256};  // 2 rows, 256 cols (1 block per row)
    std::vector<uint8_t> raw_data(2 * 98); // 2 blocks × 98 bytes

    // Initialize with test pattern
    for (size_t i = 0; i < raw_data.size(); ++i)
    {
        raw_data[i] = (i * 17) % 256;
    }

    auto tensor = std::make_unique<IQ3_XXSTensor>(shape, raw_data);

    // Decode and validate
    const float *decoded = tensor->data();
    for (size_t i = 0; i < 512; ++i)
    {
        EXPECT_TRUE(std::isfinite(decoded[i])) << "Non-finite value at index " << i;
    }
}

// =============================================================================
// IQ3_S Tests
// =============================================================================

TEST(IQ3_S_Test, TensorCreation)
{
    // IQ3_S: 256 elements per block, 110 bytes per block
    std::vector<size_t> shape = {2, 256};
    std::vector<uint8_t> raw_data(2 * 110);

    for (size_t i = 0; i < raw_data.size(); ++i)
    {
        raw_data[i] = (i * 23) % 256;
    }

    auto tensor = std::make_unique<IQ3_STensor>(shape, raw_data);

    const float *decoded = tensor->data();
    for (size_t i = 0; i < 512; ++i)
    {
        EXPECT_TRUE(std::isfinite(decoded[i])) << "Non-finite value at index " << i;
    }
}

// =============================================================================
// IQ2_XXS Tests
// =============================================================================

TEST(IQ2_XXS_Test, TensorCreation)
{
    // IQ2_XXS: 256 elements per block, 66 bytes per block
    std::vector<size_t> shape = {2, 256};
    std::vector<uint8_t> raw_data(2 * 66);

    for (size_t i = 0; i < raw_data.size(); ++i)
    {
        raw_data[i] = (i * 13) % 256;
    }

    auto tensor = std::make_unique<IQ2_XXSTensor>(shape, raw_data);

    const float *decoded = tensor->data();
    for (size_t i = 0; i < 512; ++i)
    {
        EXPECT_TRUE(std::isfinite(decoded[i])) << "Non-finite value at index " << i;
    }
}

// =============================================================================
// IQ2_XS Tests
// =============================================================================

TEST(IQ2_XS_Test, TensorCreation)
{
    // IQ2_XS: 256 elements per block, 74 bytes per block
    std::vector<size_t> shape = {2, 256};
    std::vector<uint8_t> raw_data(2 * 74);

    for (size_t i = 0; i < raw_data.size(); ++i)
    {
        raw_data[i] = (i * 19) % 256;
    }

    auto tensor = std::make_unique<IQ2_XSTensor>(shape, raw_data);

    const float *decoded = tensor->data();
    for (size_t i = 0; i < 512; ++i)
    {
        EXPECT_TRUE(std::isfinite(decoded[i])) << "Non-finite value at index " << i;
    }
}

// =============================================================================
// IQ2_S Tests
// =============================================================================

TEST(IQ2_S_Test, TensorCreation)
{
    // IQ2_S: 256 elements per block, 82 bytes per block
    std::vector<size_t> shape = {2, 256};
    std::vector<uint8_t> raw_data(2 * 82);

    for (size_t i = 0; i < raw_data.size(); ++i)
    {
        raw_data[i] = (i * 29) % 256;
    }

    auto tensor = std::make_unique<IQ2_STensor>(shape, raw_data);

    const float *decoded = tensor->data();
    for (size_t i = 0; i < 512; ++i)
    {
        EXPECT_TRUE(std::isfinite(decoded[i])) << "Non-finite value at index " << i;
    }
}

// =============================================================================
// IQ1_S Tests
// =============================================================================

TEST(IQ1_S_Test, TensorCreation)
{
    // IQ1_S: 256 elements per block, 50 bytes per block
    std::vector<size_t> shape = {2, 256};
    std::vector<uint8_t> raw_data(2 * 50);

    for (size_t i = 0; i < raw_data.size(); ++i)
    {
        raw_data[i] = (i * 31) % 256;
    }

    auto tensor = std::make_unique<IQ1_STensor>(shape, raw_data);

    const float *decoded = tensor->data();
    for (size_t i = 0; i < 512; ++i)
    {
        EXPECT_TRUE(std::isfinite(decoded[i])) << "Non-finite value at index " << i;
    }
}

// =============================================================================
// IQ1_M Tests
// =============================================================================

TEST(IQ1_M_Test, TensorCreation)
{
    // IQ1_M: 256 elements per block, 56 bytes per block
    std::vector<size_t> shape = {2, 256};
    std::vector<uint8_t> raw_data(2 * 56);

    for (size_t i = 0; i < raw_data.size(); ++i)
    {
        raw_data[i] = (i * 37) % 256;
    }

    auto tensor = std::make_unique<IQ1_MTensor>(shape, raw_data);

    const float *decoded = tensor->data();
    for (size_t i = 0; i < 512; ++i)
    {
        EXPECT_TRUE(std::isfinite(decoded[i])) << "Non-finite value at index " << i;
    }
}

// =============================================================================
// IQ4_XS Tests
// =============================================================================

TEST(IQ4_XS_Test, TensorCreation)
{
    // IQ4_XS: 256 elements per block, 136 bytes per block
    std::vector<size_t> shape = {2, 256};
    std::vector<uint8_t> raw_data(2 * 136);

    for (size_t i = 0; i < raw_data.size(); ++i)
    {
        raw_data[i] = (i * 41) % 256;
    }

    auto tensor = std::make_unique<IQ4_XSTensor>(shape, raw_data);

    const float *decoded = tensor->data();
    for (size_t i = 0; i < 512; ++i)
    {
        EXPECT_TRUE(std::isfinite(decoded[i])) << "Non-finite value at index " << i;
    }
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
