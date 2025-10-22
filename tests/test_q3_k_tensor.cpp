/**
 * @file test_q3_k_tensor.cpp
 * @brief Unit tests for Q3_KTensor implementation
 *
 * Validates Q3_KTensor::decodeRow() produces correct dequantization results
 * for 3-bit K-quant format with hierarchical scales and sign-modified values.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Q3_KTensor.h"
#include "tensors/TensorFactory.h"
#include <vector>
#include <cmath>
#include <random>
#include <cstring>

using namespace llaminar;

/**
 * @brief Helper: Convert FP32 to FP16
 */
uint16_t fp32_to_fp16_q3k(float value)
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
 * @brief Helper: Pack 16 scales into 12 bytes using Q3_K hierarchical packing
 * 
 * Follows GGML quantize_row_q3_K_impl packing scheme.
 */
std::vector<uint8_t> pack_q3k_scales(const std::vector<int8_t> &scales_biased)
{
    std::vector<uint8_t> packed(12, 0);
    
    for (int j = 0; j < 16; ++j)
    {
        int l = scales_biased[j];
        if (j < 8)
        {
            packed[j] = l & 0xF;
        }
        else
        {
            packed[j - 8] |= ((l & 0xF) << 4);
        }
        l >>= 4;
        packed[j % 4 + 8] |= (l << (2 * (j / 4)));
    }
    
    return packed;
}

/**
 * @brief Helper: Create Q3_K block data
 *
 * Q3_K block format (256 elements):
 *   - 32 bytes: hmask - high bit mask (8 values per byte)
 *   - 64 bytes: qs - lower 2 bits (4 values per byte)
 *   - 12 bytes: scales - 6-bit quantized scales (hierarchical)
 *   - 2 bytes: d (FP16) - super-block scale
 *   - Total: 110 bytes
 *
 * @param d Super-block scale
 * @param scales 16 scale values (will be biased by +32 internally)
 * @param values 256 3-bit signed values (-4 to 3 range)
 */
std::vector<uint8_t> create_q3_k_block(
    float d,
    const std::vector<int8_t> &scales,
    const std::vector<int8_t> &values)
{
    std::vector<uint8_t> raw_data;

    // 1. Pack hmask: GGML pattern - first 32 elements→bit0, next 32→bit1, etc.
    // hmask[m] bit hm represents element (hm_bit*32 + m)
    std::vector<uint8_t> hmask(32, 0);
    int m = 0;
    uint8_t hm = 1;
    for (int j = 0; j < 256; ++j)
    {
        int8_t val = values[j];
        // In GGML: if L[j] > 3, set hmask bit and subtract 4
        // L is already adjusted (+4 for negatives), so L>3 means original val >= 0
        if (val >= 0)
        {
            hmask[m] |= hm;
        }
        if (++m == 32)
        {
            m = 0;
            hm <<= 1;
        }
    }
    raw_data.insert(raw_data.end(), hmask.begin(), hmask.end());

    // 2. Pack qs: GGML pattern - each byte packs 4 elements 32 positions apart
    // qs[l] = L[j+l] | (L[j+l+32] << 2) | (L[j+l+64] << 4) | (L[j+l+96] << 6)
    // for j = 0, 128
    for (int j = 0; j < 256; j += 128)
    {
        for (int l = 0; l < 32; ++l)
        {
            uint8_t qs_byte = 0;
            
            // Extract the 4 values that map to this qs byte
            int8_t val0 = values[j + l];
            int8_t val1 = values[j + l + 32];
            int8_t val2 = values[j + l + 64];
            int8_t val3 = values[j + l + 96];
            
            // Convert each to 2-bit representation
            uint8_t q0 = (val0 >= 0) ? (val0 & 3) : ((val0 + 4) & 3);
            uint8_t q1 = (val1 >= 0) ? (val1 & 3) : ((val1 + 4) & 3);
            uint8_t q2 = (val2 >= 0) ? (val2 & 3) : ((val2 + 4) & 3);
            uint8_t q3 = (val3 >= 0) ? (val3 & 3) : ((val3 + 4) & 3);
            
            // Pack into byte with shifts 0, 2, 4, 6
            qs_byte = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6);
            raw_data.push_back(qs_byte);
        }
    }

    // 3. Pack scales: 16 scales with bias of 32
    std::vector<int8_t> scales_biased(16);
    for (int i = 0; i < 16; i++)
    {
        scales_biased[i] = scales[i] + 32;
    }
    auto packed_scales = pack_q3k_scales(scales_biased);
    for (uint8_t b : packed_scales)
    {
        raw_data.push_back(b);
    }

    // 4. Write d (FP16 super-block scale)
    uint16_t d_fp16 = fp32_to_fp16_q3k(d);
    raw_data.push_back(d_fp16 & 0xFF);
    raw_data.push_back((d_fp16 >> 8) & 0xFF);

    return raw_data;
}

/**
 * @brief Helper: Create multiple Q3_K blocks
 */
std::vector<uint8_t> create_q3_k_blocks_multi(
    const std::vector<float> &d_values,
    const std::vector<std::vector<int8_t>> &all_scales,
    const std::vector<std::vector<int8_t>> &all_values)
{
    std::vector<uint8_t> raw_data;
    for (size_t block_idx = 0; block_idx < d_values.size(); block_idx++)
    {
        auto block_data = create_q3_k_block(
            d_values[block_idx],
            all_scales[block_idx],
            all_values[block_idx]);
        raw_data.insert(raw_data.end(), block_data.begin(), block_data.end());
    }
    return raw_data;
}

// ===== Test Cases =====

/**
 * @test Basic decoding with simple values
 */
TEST(Q3_KTensorTest, BasicDecoding)
{
    // Create 1x256 tensor (1 block)
    std::vector<int> shape = {1, 256};
    
    float d = 0.5f;
    
    // 16 scales (one per 16 elements) - unbiased values
    std::vector<int8_t> scales = {
        5, 6, 7, 8, 9, 10, 11, 12,
        -5, -6, -7, -8, -9, -10, -11, -12
    };
    
    // 256 3-bit signed values (-4 to 3 range)
    std::vector<int8_t> values(256);
    for (int i = 0; i < 256; i++)
    {
        values[i] = (i % 8) - 4; // Cycle through -4, -3, -2, -1, 0, 1, 2, 3
    }
    
    auto raw_data = create_q3_k_block(d, scales, values);
    Q3_KTensor tensor(shape, raw_data);
    
    // Decode row
    std::vector<float> decoded(256);
    tensor.decodeRow(0, decoded.data());
    
    // Verify some values manually
    // First group (elements 0-15): scale = 0.5 * 5 = 2.5
    // Element 0: value = -4, result = 2.5 * (-4) = -10.0
    EXPECT_NEAR(decoded[0], -10.0f, 0.01f);
    
    // Element 4: value = 0, result = 2.5 * 0 = 0.0
    EXPECT_NEAR(decoded[4], 0.0f, 0.01f);
    
    // Element 7: value = 3, result = 2.5 * 3 = 7.5
    EXPECT_NEAR(decoded[7], 7.5f, 0.01f);
    
    // Second group (elements 16-31): scale = 0.5 * 6 = 3.0
    // Element 16: value = -4, result = 3.0 * (-4) = -12.0
    EXPECT_NEAR(decoded[16], -12.0f, 0.01f);
}

/**
 * @test Hierarchical scale extraction
 */
TEST(Q3_KTensorTest, HierarchicalScaleExtraction)
{
    std::vector<int> shape = {1, 256};
    
    float d = 1.0f;
    
    // Test all 16 scale positions
    std::vector<int8_t> scales = {
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16
    };
    
    std::vector<int8_t> values(256, 1); // All values = 1
    
    auto raw_data = create_q3_k_block(d, scales, values);
    Q3_KTensor tensor(shape, raw_data);
    
    std::vector<float> decoded(256);
    tensor.decodeRow(0, decoded.data());
    
    // Verify each group has correct scale
    for (int group = 0; group < 16; group++)
    {
        int idx = group * 16;
        float expected = 1.0f * scales[group] * 1.0f;
        EXPECT_NEAR(decoded[idx], expected, 0.1f) << "Group " << group;
    }
}

/**
 * @test Sign-modified values (hmask bit handling)
 */
TEST(Q3_KTensorTest, SignModifiedValues)
{
    std::vector<int> shape = {1, 256};
    
    float d = 1.0f;
    
    std::vector<int8_t> scales(16, 1); // All scales = 1
    
    // Test all 8 possible 3-bit signed values
    std::vector<int8_t> values(256);
    for (int i = 0; i < 256; i++)
    {
        values[i] = (i % 8) - 4; // -4, -3, -2, -1, 0, 1, 2, 3
    }
    
    auto raw_data = create_q3_k_block(d, scales, values);
    Q3_KTensor tensor(shape, raw_data);
    
    std::vector<float> decoded(256);
    tensor.decodeRow(0, decoded.data());
    
    // Verify values cycle through -4 to 3
    for (int i = 0; i < 256; i++)
    {
        float expected = static_cast<float>((i % 8) - 4);
        EXPECT_NEAR(decoded[i], expected, 0.01f) << "Index " << i;
    }
}

/**
 * @test Negative scales
 */
TEST(Q3_KTensorTest, NegativeScales)
{
    std::vector<int> shape = {1, 256};
    
    float d = 2.0f;
    
    // Mix of positive and negative scales
    std::vector<int8_t> scales = {
        -10, -8, -6, -4, -2, 0, 2, 4,
        6, 8, 10, 12, 14, 16, 18, 20
    };
    
    std::vector<int8_t> values(256, 2); // All values = 2
    
    auto raw_data = create_q3_k_block(d, scales, values);
    Q3_KTensor tensor(shape, raw_data);
    
    std::vector<float> decoded(256);
    tensor.decodeRow(0, decoded.data());
    
    // First group: scale = 2.0 * (-10) = -20.0, value = 2, result = -40.0
    EXPECT_NEAR(decoded[0], -40.0f, 0.1f);
    
    // Group 5: scale = 2.0 * 0 = 0.0, value = 2, result = 0.0
    EXPECT_NEAR(decoded[80], 0.0f, 0.01f);
    
    // Last group: scale = 2.0 * 20 = 40.0, value = 2, result = 80.0
    EXPECT_NEAR(decoded[240], 80.0f, 0.1f);
}

/**
 * @test Multiple blocks
 */
TEST(Q3_KTensorTest, MultipleBlocks)
{
    std::vector<int> shape = {2, 256}; // 2 rows, 2 blocks
    
    std::vector<float> d_values = {0.5f, 1.5f};
    
    std::vector<std::vector<int8_t>> all_scales = {
        {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
        {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    };
    
    std::vector<std::vector<int8_t>> all_values(2, std::vector<int8_t>(256));
    for (int block = 0; block < 2; block++)
    {
        for (int i = 0; i < 256; i++)
        {
            all_values[block][i] = ((i + block * 2) % 8) - 4;
        }
    }
    
    auto raw_data = create_q3_k_blocks_multi(d_values, all_scales, all_values);
    Q3_KTensor tensor(shape, raw_data);
    
    // Decode both rows
    std::vector<float> decoded_row0(256);
    std::vector<float> decoded_row1(256);
    tensor.decodeRow(0, decoded_row0.data());
    tensor.decodeRow(1, decoded_row1.data());
    
    // Verify first element of each row
    // Row 0, group 0: scale = 0.5 * 5 = 2.5, value = -4, result = -10.0
    EXPECT_NEAR(decoded_row0[0], -10.0f, 0.01f);
    
    // Row 1, group 0: scale = 1.5 * (-5) = -7.5, value = -2, result = 15.0
    EXPECT_NEAR(decoded_row1[0], 15.0f, 0.1f);
}

/**
 * @test Cross-group boundary
 */
TEST(Q3_KTensorTest, CrossGroupBoundary)
{
    std::vector<int> shape = {1, 256};
    
    float d = 1.0f;
    
    std::vector<int8_t> scales = {
        10, 20, 15, 25, 12, 18, 22, 16,
        8, 14, 19, 21, 11, 17, 23, 13
    };
    
    std::vector<int8_t> values(256, 1);
    
    auto raw_data = create_q3_k_block(d, scales, values);
    Q3_KTensor tensor(shape, raw_data);
    
    std::vector<float> decoded(256);
    tensor.decodeRow(0, decoded.data());
    
    // Last element of group 0 (element 15): scale = 1.0 * 10 = 10.0
    EXPECT_NEAR(decoded[15], 10.0f, 0.01f);
    
    // First element of group 1 (element 16): scale = 1.0 * 20 = 20.0
    EXPECT_NEAR(decoded[16], 20.0f, 0.01f);
}

/**
 * @test BF16 decoding
 */
TEST(Q3_KTensorTest, BF16Decoding)
{
    std::vector<int> shape = {1, 256};
    
    float d = 0.5f;
    
    std::vector<int8_t> scales(16, 10);
    std::vector<int8_t> values(256);
    for (int i = 0; i < 256; i++)
    {
        values[i] = (i % 8) - 4;
    }
    
    auto raw_data = create_q3_k_block(d, scales, values);
    Q3_KTensor tensor(shape, raw_data);
    
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
        float tolerance = std::max(0.01f, std::abs(decoded_fp32[i]) * 0.01f);
        EXPECT_NEAR(bf16_val, decoded_fp32[i], tolerance) << "Mismatch at index " << i;
    }
}

/**
 * @test Span decoding
 */
TEST(Q3_KTensorTest, SpanDecoding)
{
    std::vector<int> shape = {2, 256};
    
    std::vector<float> d_values = {0.5f, 1.0f};
    
    std::vector<std::vector<int8_t>> all_scales(2);
    for (int block = 0; block < 2; block++)
    {
        all_scales[block] = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    }
    
    std::vector<std::vector<int8_t>> all_values(2, std::vector<int8_t>(256));
    for (int block = 0; block < 2; block++)
    {
        for (int i = 0; i < 256; i++)
        {
            all_values[block][i] = ((i + block * 3) % 8) - 4;
        }
    }
    
    auto raw_data = create_q3_k_blocks_multi(d_values, all_scales, all_values);
    Q3_KTensor tensor(shape, raw_data);
    
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
TEST(Q3_KTensorTest, ErrorHandling)
{
    // Invalid shape size mismatch
    std::vector<int> shape = {1, 256};
    std::vector<uint8_t> raw_data(100); // Too small
    EXPECT_THROW(Q3_KTensor(shape, raw_data), std::invalid_argument);
    
    // Out of bounds row access
    std::vector<int8_t> scales(16, 10);
    std::vector<int8_t> values(256, 1);
    auto valid_data = create_q3_k_block(0.5f, scales, values);
    Q3_KTensor tensor(shape, valid_data);
    
    std::vector<float> buffer(256);
    EXPECT_THROW(tensor.decodeRow(1, buffer.data()), std::out_of_range);
}
