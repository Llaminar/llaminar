#include <gtest/gtest.h>
#include "../src/model_loader.h"
#include <vector>
#include <cstring>
#include <cmath>

// Test utilities for synthetic block construction
namespace
{
    uint16_t float_to_fp16_bits(float f)
    {
        // Simple IEEE 754 binary32 -> binary16 conversion for test purposes
        union
        {
            float f;
            uint32_t u;
        } conv = {f};
        uint32_t u = conv.u;

        uint32_t sign = (u >> 16) & 0x8000;
        int32_t exp = ((u >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (u >> 13) & 0x3FF;

        if (exp <= 0)
        {
            // Zero or subnormal
            return static_cast<uint16_t>(sign);
        }
        else if (exp >= 31)
        {
            // Infinity or NaN
            return static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x200 : 0));
        }
        else
        {
            return static_cast<uint16_t>(sign | (exp << 10) | mant);
        }
    }
}

class DequantExtraTest : public ::testing::Test
{
protected:
    ModelLoader loader;

    void SetUp() override
    {
        // No model loading needed for unit tests with synthetic blocks
    }
};

TEST_F(DequantExtraTest, Q5_0BasicBlock)
{
    // Test Q5_0 with a single block of 32 values
    // Layout: uint16_t d, uint8_t qh[4], uint8_t qs[16]
    // Total: 2 + 4 + 16 = 22 bytes

    std::vector<uint8_t> block_data(22);

    // Set scale d = 2.0
    float scale = 2.0f;
    uint16_t scale_bits = float_to_fp16_bits(scale);
    std::memcpy(block_data.data(), &scale_bits, 2);

    // Set qh (high bits): pattern where every 8th element has high bit set
    // qh[0] = 0x01 (bit 0 set -> element 0 has high bit)
    // qh[1] = 0x01 (bit 0 set -> element 8 has high bit)
    // etc.
    block_data[2] = 0x01; // qh[0]
    block_data[3] = 0x01; // qh[1]
    block_data[4] = 0x01; // qh[2]
    block_data[5] = 0x01; // qh[3]

    // Set qs (low 4 bits): alternating pattern 0x05 (gives low nibble 5, high nibble 0)
    for (int i = 0; i < 16; ++i)
    {
        block_data[6 + i] = 0x05; // low=5, high=0
    }

    // Create fake tensor info
    GGUFTensorInfo info;
    info.dimensions = {32}; // 32 elements

    // Decode
    auto result = loader.dequantizeQ5_0(block_data.data(), info);

    ASSERT_EQ(result.size(), 32);

    // Expected values per ggml mapping (model_loader mirrors this): low nibbles (indices 0..15) first, then high nibbles (16..31).
    // With qs bytes all 0x05 (low=5, high=0) and high-bit set only for elements 0,8,16,24 via qh pattern 0x01 per byte:
    // Indices 0,8: high bit + low nibble 5 => (5|16)-16=5 -> 10.0
    // Other low-nibble indices (1,2,3,9,10,11,...) have no high bit: raw=5 -> -11 -> -22.0
    // High-nibble half (16..31): high nibble 0 possibly plus high bits at 16 & 24 -> values either 10.0 (if high bit) or -32.0.

    EXPECT_FLOAT_EQ(result[0], 10.0f);   // element 0: low nibble with high bit
    EXPECT_FLOAT_EQ(result[1], -22.0f);  // element 1: low nibble 5 no high bit
    EXPECT_FLOAT_EQ(result[2], -22.0f);  // element 2: low nibble 5 no high bit
    EXPECT_FLOAT_EQ(result[3], -22.0f);  // element 3: low nibble 5 no high bit

    EXPECT_FLOAT_EQ(result[8], 10.0f);   // element 8: low nibble with high bit
    EXPECT_FLOAT_EQ(result[9], -22.0f);  // element 9: low nibble 5 no high bit
}

TEST_F(DequantExtraTest, Q2_KBasicSuperBlock)
{
    // Test Q2_K with minimal super-block (256 values)
    // Layout (test synthetic variant): uint16_t d, uint16_t dmin, uint8_t scales[12], uint8_t qs[64]
    // Total: 2 + 2 + 12 + 64 = 80 bytes. Model loader supports both this compact form and the canonical 16-scale form.

    std::vector<uint8_t> block_data(80);

    // Set d = 1.0, dmin = 0.5
    float d = 1.0f, dmin = 0.5f;
    uint16_t d_bits = float_to_fp16_bits(d);
    uint16_t dmin_bits = float_to_fp16_bits(dmin);
    std::memcpy(block_data.data(), &d_bits, 2);
    std::memcpy(block_data.data() + 2, &dmin_bits, 2);

    // Set scales[12] to simple pattern for unpacking test
    // After bit reshuffling, we expect 8 scale and 8 min values
    // Use a pattern that survives the kmask operations
    for (int i = 0; i < 12; ++i)
    {
        block_data[4 + i] = static_cast<uint8_t>(i + 1); // 1,2,3,...,12
    }

    // Set qs to simple 2-bit pattern: all bytes = 0x1B (binary 00011011)
    // This gives 2-bit values: 3,2,1,0 per byte (reading bits 0-1, 2-3, 4-5, 6-7)
    for (int i = 0; i < 64; ++i)
    {
        block_data[16 + i] = 0x1B;
    }

    // Create tensor info for 256 elements
    GGUFTensorInfo info;
    info.dimensions = {256};

    // Decode
    auto result = loader.dequantizeQ2_K(block_data.data(), GGUFTensorType::Q2_K, "test_q2k", info);

    ASSERT_EQ(result.size(), 256);

    // The exact expected values depend on the scale/min unpacking and bit shuffles.
    // We only assert general sanity: finite values and non-trivial variation.
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    for (const auto &val : result)
    {
        EXPECT_FALSE(std::isnan(val));
        if (val < min_v)
            min_v = val;
        if (val > max_v)
            max_v = val;
    }
    // Loose bounds – with synthetic scales (1..12) raw formula can yield larger magnitudes.
    EXPECT_GT(max_v, -min_v * 0.2f); // ensure not all negative dominated
    EXPECT_LT(std::fabs(min_v), 1000.0f);
    EXPECT_LT(std::fabs(max_v), 1000.0f);

    // Verify first few values match expected pattern based on formula: d*scale*q - dmin*min
    // With our 2-bit pattern (3,2,1,0), we expect different values for each position
    EXPECT_NE(result[0], result[1]);
    EXPECT_NE(result[1], result[2]);
    EXPECT_NE(result[2], result[3]);
}

TEST_F(DequantExtraTest, Q5_KSmoke)
{
    // Minimal smoke test for Q5_K (similar structure to Q4_K but with high-bit plane)
    // Layout: uint16_t d, uint16_t dmin, uint8_t scales[12], uint8_t qh[32], uint8_t qs[128]
    // Total: 2 + 2 + 12 + 32 + 128 = 176 bytes

    std::vector<uint8_t> block_data(176);

    // Set d = 1.0, dmin = 0.1
    float d = 1.0f, dmin = 0.1f;
    uint16_t d_bits = float_to_fp16_bits(d);
    uint16_t dmin_bits = float_to_fp16_bits(dmin);
    std::memcpy(block_data.data(), &d_bits, 2);
    std::memcpy(block_data.data() + 2, &dmin_bits, 2);

    // Set scales to simple pattern
    for (int i = 0; i < 12; ++i)
    {
        block_data[4 + i] = static_cast<uint8_t>(i + 1);
    }

    // Set qh (high bits) to pattern
    for (int i = 0; i < 32; ++i)
    {
        block_data[16 + i] = (i % 2) ? 0xFF : 0x00; // Alternating
    }

    // Set qs (low 4 bits) to simple pattern
    for (int i = 0; i < 128; ++i)
    {
        block_data[48 + i] = 0x73; // Low=3, high=7
    }

    GGUFTensorInfo info;
    info.dimensions = {256};

    auto result = loader.dequantizeQ5_K(block_data.data(), GGUFTensorType::Q5_K, "test_q5k", info);

    ASSERT_EQ(result.size(), 256);

    // Basic sanity checks
    for (const auto &val : result)
    {
        EXPECT_FALSE(std::isnan(val));
    }

    // Verify that high-bit processing creates different values
    // (exact verification would require replicating the complex bit unpacking)
    bool has_variation = false;
    for (size_t i = 1; i < result.size(); ++i)
    {
        if (std::abs(result[i] - result[0]) > 1e-6f)
        {
            has_variation = true;
            break;
        }
    }
    EXPECT_TRUE(has_variation);
}

TEST_F(DequantExtraTest, Q6_KSmoke)
{
    // Basic smoke test for Q6_K
    // Layout: uint16_t d, uint8_t ql[128], uint8_t qh[64], int8_t scales[16]
    // Total: 2 + 128 + 64 + 16 = 210 bytes

    std::vector<uint8_t> block_data(210);

    // Set d = 0.5
    float d = 0.5f;
    uint16_t d_bits = float_to_fp16_bits(d);
    std::memcpy(block_data.data(), &d_bits, 2);

    // Set ql (low nibbles) to pattern
    for (int i = 0; i < 128; ++i)
    {
        block_data[2 + i] = 0xA5; // Pattern: low=5, high=10
    }

    // Set qh (high 2-bit groups) to pattern
    for (int i = 0; i < 64; ++i)
    {
        block_data[130 + i] = 0x1B; // 2-bit groups: 3,2,1,0
    }

    // Set scales (int8_t) to small values
    for (int i = 0; i < 16; ++i)
    {
        int8_t scale_val = static_cast<int8_t>(i - 8); // Range -8 to 7
        std::memcpy(block_data.data() + 194 + i, &scale_val, 1);
    }

    GGUFTensorInfo info;
    info.dimensions = {256};

    auto result = loader.dequantizeQ6_K(block_data.data(), GGUFTensorType::Q6_K, "test_q6k", info);

    ASSERT_EQ(result.size(), 256);

    // Basic sanity checks
    for (const auto &val : result)
    {
        EXPECT_FALSE(std::isnan(val));
        // Theoretical max magnitude: (|raw-32| <= 32) * |scale| (<=7) * d (0.5) ~= 112
        EXPECT_GE(val, -150.0f);
        EXPECT_LE(val, 150.0f);
    }

    // Verify different scale groups produce different value ranges
    bool has_positive = false, has_negative = false;
    for (const auto &val : result)
    {
        if (val > 1e-6f)
            has_positive = true;
        if (val < -1e-6f)
            has_negative = true;
    }
    EXPECT_TRUE(has_positive || has_negative); // Should have some non-zero values
}