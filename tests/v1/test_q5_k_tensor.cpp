/**
 * @file test_q5_k_tensor.cpp
 * @brief Unit tests for Q5_KTensor implementation
 *
 * Validates Q5_KTensor::decodeRow() produces correct dequantization results
 * for 5-bit K-quant format with hierarchical scales and mins.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Q5_KTensor.h"
#include "tensors/TensorFactory.h"
#include <vector>
#include <cmath>
#include <random>

using namespace llaminar;

/**
 * @brief Helper: Convert FP32 to FP16
 */
uint16_t fp32_to_fp16_q5k(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, 4);

    uint32_t sign = (bits >> 16) & 0x8000;
    uint32_t exp = ((bits >> 23) & 0xFF);
    uint32_t mant = (bits & 0x7FFFFF);

    if (exp == 0)
    {
        return sign; // Zero or denormal
    }
    else if (exp == 255)
    {
        return sign | 0x7C00 | (mant ? 0x0200 : 0); // Inf or NaN
    }
    else
    {
        int32_t new_exp = exp - 127 + 15;
        if (new_exp <= 0)
        {
            return sign; // Underflow
        }
        else if (new_exp >= 31)
        {
            return sign | 0x7C00; // Overflow
        }
        else
        {
            return sign | (new_exp << 10) | (mant >> 13);
        }
    }
}

/**
 * @brief Helper: Create Q5_K block data
 *
 * Q5_K block format (256 elements):
 *   - 2 bytes: d (FP16) - super-block scale for scales
 *   - 2 bytes: dmin (FP16) - super-block scale for mins
 *   - 12 bytes: scales - 6-bit quantized scales and mins (hierarchical)
 *   - 128 bytes: qs - lower 4 bits (2 values per byte)
 *   - 32 bytes: qh - upper 1 bit (8 values per byte)
 *   - Total: 176 bytes
 *
 * @param d Super-block scale for scales
 * @param dmin Super-block scale for mins
 * @param scales_mins 8 pairs of (scale, min) as 6-bit values (0-63 each)
 * @param values 256 5-bit values (0-31 each)
 */
std::vector<uint8_t> create_q5_k_block(
    float d,
    float dmin,
    const std::vector<std::pair<uint8_t, uint8_t>> &scales_mins,
    const std::vector<uint8_t> &values)
{
    std::vector<uint8_t> raw_data;

    // 1. Write d (FP16 super-block scale)
    uint16_t d_fp16 = fp32_to_fp16_q5k(d);
    raw_data.push_back(d_fp16 & 0xFF);
    raw_data.push_back((d_fp16 >> 8) & 0xFF);

    // 2. Write dmin (FP16 super-block min scale)
    uint16_t dmin_fp16 = fp32_to_fp16_q5k(dmin);
    raw_data.push_back(dmin_fp16 & 0xFF);
    raw_data.push_back((dmin_fp16 >> 8) & 0xFF);

    // 3. Pack 12 bytes of scales (hierarchical 6-bit packing)
    // This follows GGML get_scale_min_k4 pattern
    std::vector<uint8_t> scales_packed(12, 0);
    
    // First 4 pairs: scale in bytes 0-3, min in bytes 4-7 (lower 6 bits each)
    for (int i = 0; i < 4; i++)
    {
        scales_packed[i] = scales_mins[i].first & 0x3F;       // scale (6 bits)
        scales_packed[i + 4] = scales_mins[i].second & 0x3F;  // min (6 bits)
    }
    
    // Last 4 pairs: hierarchical packing using upper 2 bits
    for (int i = 4; i < 8; i++)
    {
        uint8_t sc = scales_mins[i].first;
        uint8_t m = scales_mins[i].second;
        
        // Lower 4 bits of sc and m go into byte i+4
        scales_packed[i + 4] = (sc & 0x0F) | ((m & 0x0F) << 4);
        
        // Upper 2 bits of sc go into upper bits of byte i-4
        scales_packed[i - 4] |= ((sc >> 4) & 0x03) << 6;
        
        // Upper 2 bits of m go into upper bits of byte i
        scales_packed[i] |= ((m >> 4) & 0x03) << 6;
    }
    
    for (uint8_t b : scales_packed)
    {
        raw_data.push_back(b);
    }

    // 4. Pack qs: lower 4 bits (256 values → 128 bytes, 2 values per byte)
    for (int i = 0; i < 128; i++)
    {
        uint8_t low0 = values[i * 2] & 0x0F;
        uint8_t low1 = values[i * 2 + 1] & 0x0F;
        raw_data.push_back(low0 | (low1 << 4));
    }

    // 5. Pack qh: upper 1 bit (256 values → 32 bytes, 8 values per byte)
    for (int i = 0; i < 32; i++)
    {
        uint8_t packed = 0;
        for (int j = 0; j < 8; j++)
        {
            uint8_t high_bit = (values[i * 8 + j] >> 4) & 0x01;
            packed |= (high_bit << j);
        }
        raw_data.push_back(packed);
    }

    return raw_data;
}

/**
 * @brief Helper: Create multiple Q5_K blocks
 */
std::vector<uint8_t> create_q5_k_blocks_multi(
    const std::vector<float> &d_values,
    const std::vector<float> &dmin_values,
    const std::vector<std::vector<std::pair<uint8_t, uint8_t>>> &all_scales_mins,
    const std::vector<std::vector<uint8_t>> &all_values)
{
    std::vector<uint8_t> raw_data;
    for (size_t block_idx = 0; block_idx < d_values.size(); block_idx++)
    {
        auto block_data = create_q5_k_block(
            d_values[block_idx],
            dmin_values[block_idx],
            all_scales_mins[block_idx],
            all_values[block_idx]);
        raw_data.insert(raw_data.end(), block_data.begin(), block_data.end());
    }
    return raw_data;
}

// ===== Test Cases =====

/**
 * @test Basic decoding with simple values
 */
TEST(Q5_KTensorTest, BasicDecoding)
{
    // Create 1x256 tensor (1 block)
    std::vector<int> shape = {1, 256};
    
    // Set up block parameters
    float d = 0.5f;
    float dmin = 0.1f;
    
    // 8 scale/min pairs (one per 32 elements)
    std::vector<std::pair<uint8_t, uint8_t>> scales_mins = {
        {10, 5}, {12, 6}, {8, 4}, {15, 7},
        {11, 5}, {9, 3}, {13, 6}, {14, 8}
    };
    
    // 256 5-bit values (0-31 range)
    std::vector<uint8_t> values(256);
    for (int i = 0; i < 256; i++)
    {
        values[i] = i % 32; // Cycle through 0-31
    }
    
    auto raw_data = create_q5_k_block(d, dmin, scales_mins, values);
    Q5_KTensor tensor(shape, raw_data);
    
    // Decode row
    std::vector<float> decoded(256);
    tensor.decodeRow(0, decoded.data());
    
    // Verify some values manually
    // First group (elements 0-31): scale = d * 10 = 5.0, min = dmin * 5 = 0.5
    // Element 0: value = 0, result = 5.0 * 0 - 0.5 = -0.5
    EXPECT_NEAR(decoded[0], -0.5f, 0.01f);
    
    // Element 10: value = 10, result = 5.0 * 10 - 0.5 = 49.5
    EXPECT_NEAR(decoded[10], 49.5f, 0.01f);
    
    // Second group (elements 32-63): scale = d * 12 = 6.0, min = dmin * 6 = 0.6
    // Element 32: value = 0, result = 6.0 * 0 - 0.6 = -0.6
    EXPECT_NEAR(decoded[32], -0.6f, 0.01f);
    
    // Element 40: value = 8, result = 6.0 * 8 - 0.6 = 47.4
    EXPECT_NEAR(decoded[40], 47.4f, 0.01f);
}

/**
 * @test Hierarchical scale extraction for groups 4-7
 */
TEST(Q5_KTensorTest, HierarchicalScaleExtraction)
{
    std::vector<int> shape = {1, 256};
    
    float d = 1.0f;
    float dmin = 0.5f;
    
    // Test hierarchical extraction (groups 4-7 use different packing)
    std::vector<std::pair<uint8_t, uint8_t>> scales_mins = {
        {10, 5}, {12, 6}, {8, 4}, {15, 7},
        {20, 10}, {25, 15}, {30, 20}, {35, 25}  // These use hierarchical packing
    };
    
    std::vector<uint8_t> values(256, 15); // All values = 15
    
    auto raw_data = create_q5_k_block(d, dmin, scales_mins, values);
    Q5_KTensor tensor(shape, raw_data);
    
    std::vector<float> decoded(256);
    tensor.decodeRow(0, decoded.data());
    
    // Group 4 (elements 128-159): scale = 1.0 * 20 = 20.0, min = 0.5 * 10 = 5.0
    // value = 15, result = 20.0 * 15 - 5.0 = 295.0
    EXPECT_NEAR(decoded[128], 295.0f, 0.1f);
    
    // Group 5 (elements 160-191): scale = 1.0 * 25 = 25.0, min = 0.5 * 15 = 7.5
    // value = 15, result = 25.0 * 15 - 7.5 = 367.5
    EXPECT_NEAR(decoded[160], 367.5f, 0.1f);
}

/**
 * @test High bit extraction (5-bit = 4 low + 1 high)
 */
TEST(Q5_KTensorTest, HighBitExtraction)
{
    std::vector<int> shape = {1, 256};
    
    float d = 1.0f;
    float dmin = 0.0f; // Zero min for simplicity
    
    std::vector<std::pair<uint8_t, uint8_t>> scales_mins(8, {1, 0});
    
    // Test 5-bit values with high bit set
    std::vector<uint8_t> values(256);
    for (int i = 0; i < 256; i++)
    {
        // Alternate between values with and without high bit
        values[i] = (i % 2 == 0) ? 31 : 15;  // 31 = 0b11111, 15 = 0b01111
    }
    
    auto raw_data = create_q5_k_block(d, dmin, scales_mins, values);
    Q5_KTensor tensor(shape, raw_data);
    
    std::vector<float> decoded(256);
    tensor.decodeRow(0, decoded.data());
    
    // Even indices: 31.0 (high bit set)
    EXPECT_NEAR(decoded[0], 31.0f, 0.01f);
    EXPECT_NEAR(decoded[2], 31.0f, 0.01f);
    
    // Odd indices: 15.0 (high bit clear)
    EXPECT_NEAR(decoded[1], 15.0f, 0.01f);
    EXPECT_NEAR(decoded[3], 15.0f, 0.01f);
}

/**
 * @test Multiple blocks
 */
TEST(Q5_KTensorTest, MultipleBlocks)
{
    std::vector<int> shape = {2, 256}; // 2 rows, 2 blocks
    
    std::vector<float> d_values = {0.25f, 0.75f};
    std::vector<float> dmin_values = {0.05f, 0.1f};
    
    std::vector<std::vector<std::pair<uint8_t, uint8_t>>> all_scales_mins = {
        {{10, 5}, {12, 6}, {8, 4}, {15, 7}, {11, 5}, {9, 3}, {13, 6}, {14, 8}},
        {{20, 10}, {22, 12}, {18, 9}, {25, 15}, {21, 11}, {19, 8}, {23, 13}, {24, 14}}
    };
    
    std::vector<std::vector<uint8_t>> all_values(2, std::vector<uint8_t>(256));
    for (int block = 0; block < 2; block++)
    {
        for (int i = 0; i < 256; i++)
        {
            all_values[block][i] = (i + block * 10) % 32;
        }
    }
    
    auto raw_data = create_q5_k_blocks_multi(d_values, dmin_values, all_scales_mins, all_values);
    Q5_KTensor tensor(shape, raw_data);
    
    // Decode both rows
    std::vector<float> decoded_row0(256);
    std::vector<float> decoded_row1(256);
    tensor.decodeRow(0, decoded_row0.data());
    tensor.decodeRow(1, decoded_row1.data());
    
    // Verify first element of each row
    // Row 0, group 0: scale = 0.25 * 10 = 2.5, min = 0.05 * 5 = 0.25, value = 0
    // result = 2.5 * 0 - 0.25 = -0.25
    EXPECT_NEAR(decoded_row0[0], -0.25f, 0.01f);
    
    // Row 1, group 0: scale = 0.75 * 20 = 15.0, min = 0.1 * 10 = 1.0, value = 10
    // result = 15.0 * 10 - 1.0 = 149.0
    EXPECT_NEAR(decoded_row1[0], 149.0f, 0.1f);
}

/**
 * @test Cross-group boundary
 */
TEST(Q5_KTensorTest, CrossGroupBoundary)
{
    std::vector<int> shape = {1, 256};
    
    float d = 0.5f;
    float dmin = 0.1f;
    
    std::vector<std::pair<uint8_t, uint8_t>> scales_mins = {
        {10, 5}, {20, 10}, {15, 7}, {25, 12},
        {12, 6}, {18, 9}, {22, 11}, {16, 8}
    };
    
    std::vector<uint8_t> values(256, 10);
    
    auto raw_data = create_q5_k_block(d, dmin, scales_mins, values);
    Q5_KTensor tensor(shape, raw_data);
    
    std::vector<float> decoded(256);
    tensor.decodeRow(0, decoded.data());
    
    // Last element of group 0 (element 31): scale = 0.5 * 10 = 5.0, min = 0.1 * 5 = 0.5
    // value = 10, result = 5.0 * 10 - 0.5 = 49.5
    EXPECT_NEAR(decoded[31], 49.5f, 0.01f);
    
    // First element of group 1 (element 32): scale = 0.5 * 20 = 10.0, min = 0.1 * 10 = 1.0
    // value = 10, result = 10.0 * 10 - 1.0 = 99.0
    EXPECT_NEAR(decoded[32], 99.0f, 0.1f);
}

/**
 * @test BF16 decoding
 */
TEST(Q5_KTensorTest, BF16Decoding)
{
    std::vector<int> shape = {1, 256};
    
    float d = 0.5f;
    float dmin = 0.1f;
    
    std::vector<std::pair<uint8_t, uint8_t>> scales_mins(8, {10, 5});
    std::vector<uint8_t> values(256);
    for (int i = 0; i < 256; i++)
    {
        values[i] = i % 32;
    }
    
    auto raw_data = create_q5_k_block(d, dmin, scales_mins, values);
    Q5_KTensor tensor(shape, raw_data);
    
    // Decode to BF16
    std::vector<uint16_t> decoded_bf16(256);
    tensor.decodeRowToBF16(0, reinterpret_cast<bfloat16 *>(decoded_bf16.data()));
    
    // Decode to FP32 for comparison
    std::vector<float> decoded_fp32(256);
    tensor.decodeRow(0, decoded_fp32.data());
    
    // Compare BF16 → FP32 with direct FP32
    for (int i = 0; i < 256; i++)
    {
        float bf16_val = static_cast<float>(reinterpret_cast<bfloat16 *>(decoded_bf16.data())[i]);
        // BF16 has 7 mantissa bits, so tolerance is ~1/128 of value
        float tolerance = std::max(0.01f, std::abs(decoded_fp32[i]) * 0.01f);
        EXPECT_NEAR(bf16_val, decoded_fp32[i], tolerance) << "Mismatch at index " << i;
    }
}

/**
 * @test Span decoding
 */
TEST(Q5_KTensorTest, SpanDecoding)
{
    std::vector<int> shape = {2, 256};
    
    std::vector<float> d_values = {0.5f, 1.0f};
    std::vector<float> dmin_values = {0.1f, 0.2f};
    
    std::vector<std::vector<std::pair<uint8_t, uint8_t>>> all_scales_mins(2);
    for (int block = 0; block < 2; block++)
    {
        all_scales_mins[block] = {{10, 5}, {12, 6}, {8, 4}, {15, 7}, {11, 5}, {9, 3}, {13, 6}, {14, 8}};
    }
    
    std::vector<std::vector<uint8_t>> all_values(2, std::vector<uint8_t>(256));
    for (int block = 0; block < 2; block++)
    {
        for (int i = 0; i < 256; i++)
        {
            all_values[block][i] = (i + block * 5) % 32;
        }
    }
    
    auto raw_data = create_q5_k_blocks_multi(d_values, dmin_values, all_scales_mins, all_values);
    Q5_KTensor tensor(shape, raw_data);
    
    // Decode span crossing block boundary
    std::vector<float> span_decoded(128);
    tensor.decodeSpan(200, 128, span_decoded.data());
    
    // Decode full rows for comparison
    std::vector<float> full_row0(256), full_row1(256);
    tensor.decodeRow(0, full_row0.data());
    tensor.decodeRow(1, full_row1.data());
    
    // Verify span matches full decode
    for (int i = 0; i < 56; i++)
    { // First 56 elements from row 0
        EXPECT_NEAR(span_decoded[i], full_row0[200 + i], 0.01f);
    }
    for (int i = 0; i < 72; i++)
    { // Next 72 elements from row 1
        EXPECT_NEAR(span_decoded[56 + i], full_row1[i], 0.01f);
    }
}

/**
 * @test Error handling
 */
TEST(Q5_KTensorTest, ErrorHandling)
{
    // Invalid shape size mismatch
    std::vector<int> shape = {1, 256};
    std::vector<uint8_t> raw_data(100); // Too small
    EXPECT_THROW(Q5_KTensor(shape, raw_data), std::invalid_argument);
    
    // Out of bounds row access
    std::vector<std::pair<uint8_t, uint8_t>> scales_mins(8, {10, 5});
    std::vector<uint8_t> values(256, 10);
    auto valid_data = create_q5_k_block(0.5f, 0.1f, scales_mins, values);
    Q5_KTensor tensor(shape, valid_data);
    
    std::vector<float> buffer(256);
    EXPECT_THROW(tensor.decodeRow(1, buffer.data()), std::out_of_range);
}
