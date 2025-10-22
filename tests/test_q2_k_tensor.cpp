/**
 * @file test_q2_k_tensor.cpp
 * @brief Unit tests for Q2_KTensor implementation
 *
 * Validates Q2_KTensor::decodeRow() produces correct dequantization results
 * for 2-bit K-quant format with affine quantization (scale + min).
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Q2_KTensor.h"
#include "tensors/TensorFactory.h"
#include <vector>
#include <cmath>
#include <random>
#include <cstring>

using namespace llaminar;

/**
 * @brief Helper: Convert FP32 to FP16
 */
uint16_t fp32_to_fp16_q2k(float value)
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
 * @brief Helper: Pack 4-bit scale + 4-bit min into single byte
 */
uint8_t pack_q2k_scale_min(uint8_t scale, uint8_t min)
{
    return (min << 4) | (scale & 0xF);
}

/**
 * @brief Helper: Create Q2_K block data
 *
 * Q2_K block format (256 elements):
 *   - 16 bytes: scales - 4-bit scale + 4-bit min per 16 elements
 *   - 64 bytes: qs - 2 bits per value (4 values per byte)
 *   - 2 bytes: d (FP16) - super-block scale for scales
 *   - 2 bytes: dmin (FP16) - super-block scale for mins
 *   - Total: 84 bytes
 *
 * @param d Super-block scale for quantized scales
 * @param dmin Super-block scale for quantized mins
 * @param scales 16 4-bit scale values (0-15)
 * @param mins 16 4-bit min values (0-15)
 * @param quants 256 2-bit quantized values (0-3)
 */
std::vector<uint8_t> create_q2_k_block(
    float d,
    float dmin,
    const std::vector<uint8_t> &scales,
    const std::vector<uint8_t> &mins,
    const std::vector<uint8_t> &quants)
{
    std::vector<uint8_t> raw_data;

    // 1. Pack scales and mins into 16 bytes
    for (int i = 0; i < 16; ++i)
    {
        raw_data.push_back(pack_q2k_scale_min(scales[i], mins[i]));
    }

    // 2. Pack 2-bit quantized values into 64 bytes
    // 4 values per byte: qs[byte] = (v3 << 6) | (v2 << 4) | (v1 << 2) | v0
    // GGML pattern: elements 32 apart share same byte, shift cycles 0,2,4,6
    for (int i = 0; i < 64; ++i)
    {
        raw_data.push_back(0);  // Initialize
    }
    
    // Fill qs using GGML pattern (same as Q3_K)
    for (int i = 0; i < 256; ++i)
    {
        int group_128 = i / 128;
        int in_group_128 = i % 128;
        int subgroup_32 = in_group_128 / 32;
        int in_subgroup_32 = in_group_128 % 32;
        int part_16 = in_subgroup_32 / 16;
        int in_part_16 = in_subgroup_32 % 16;
        
        int q_offset = group_128 * 32;
        int shift = subgroup_32 * 2;
        int qs_idx = q_offset + part_16 * 16 + in_part_16;
        
        raw_data[16 + qs_idx] |= ((quants[i] & 3) << shift);
    }

    // 3. Append FP16 d
    uint16_t d_fp16 = fp32_to_fp16_q2k(d);
    raw_data.push_back(d_fp16 & 0xFF);
    raw_data.push_back((d_fp16 >> 8) & 0xFF);

    // 4. Append FP16 dmin
    uint16_t dmin_fp16 = fp32_to_fp16_q2k(dmin);
    raw_data.push_back(dmin_fp16 & 0xFF);
    raw_data.push_back((dmin_fp16 >> 8) & 0xFF);

    return raw_data;
}

class Q2_KTensorTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test BasicDecoding
 * Verifies basic Q2_K decoding with simple scale/min values
 */
TEST_F(Q2_KTensorTest, BasicDecoding)
{
    // Create simple Q2_K block: d=2.0, dmin=1.0
    // First 16 elements: scale=8, min=4 → dl=16, ml=4
    // quants = [0,1,2,3,0,1,2,3,...] (repeating pattern)
    std::vector<uint8_t> scales(16, 8);  // All 4-bit scale = 8
    std::vector<uint8_t> mins(16, 4);    // All 4-bit min = 4
    std::vector<uint8_t> quants(256);
    for (int i = 0; i < 256; ++i)
    {
        quants[i] = i % 4;  // 0,1,2,3,0,1,2,3,...
    }
    
    auto raw_data = create_q2_k_block(2.0f, 1.0f, scales, mins, quants);
    Q2_KTensor tensor({1, 256}, raw_data);

    std::vector<float> decoded(256);
    tensor.decodeRow(0, decoded.data());

    // Expected: value = dl * quant - ml
    // dl = 2.0 * 8 = 16.0, ml = 1.0 * 4 = 4.0
    // quant=0 → 16*0 - 4 = -4
    // quant=1 → 16*1 - 4 = 12
    // quant=2 → 16*2 - 4 = 28
    // quant=3 → 16*3 - 4 = 44
    
    EXPECT_NEAR(decoded[0], -4.0f, 0.1f);   // quant=0
    EXPECT_NEAR(decoded[1], 12.0f, 0.1f);   // quant=1
    EXPECT_NEAR(decoded[2], 28.0f, 0.1f);   // quant=2
    EXPECT_NEAR(decoded[3], 44.0f, 0.1f);   // quant=3
    EXPECT_NEAR(decoded[4], -4.0f, 0.1f);   // quant=0
}

/**
 * @test ScaleExtraction
 * Verifies correct extraction of scale and min from packed bytes
 */
TEST_F(Q2_KTensorTest, ScaleExtraction)
{
    // Test varying scales/mins across 16 scale groups
    std::vector<uint8_t> scales(16);
    std::vector<uint8_t> mins(16);
    std::vector<uint8_t> quants(256, 1);  // All quant=1 for simplicity
    
    // First group: scale=15, min=0
    scales[0] = 15;
    mins[0] = 0;
    
    // Second group: scale=0, min=15
    scales[1] = 0;
    mins[1] = 15;
    
    // Fill rest with default
    for (int i = 2; i < 16; ++i)
    {
        scales[i] = 8;
        mins[i] = 8;
    }
    
    auto raw_data = create_q2_k_block(1.0f, 1.0f, scales, mins, quants);
    Q2_KTensor tensor({1, 256}, raw_data);

    std::vector<float> decoded(256);
    tensor.decodeRow(0, decoded.data());

    // First 16 elements (scale_idx=0): dl=1*15=15, ml=1*0=0
    // value = 15*1 - 0 = 15
    EXPECT_NEAR(decoded[0], 15.0f, 0.1f);
    EXPECT_NEAR(decoded[15], 15.0f, 0.1f);

    // Next 16 elements (scale_idx=1): dl=1*0=0, ml=1*15=15
    // value = 0*1 - 15 = -15
    EXPECT_NEAR(decoded[16], -15.0f, 0.1f);
    EXPECT_NEAR(decoded[31], -15.0f, 0.1f);
}

/**
 * @test MinValues
 * Verifies correct handling of min subtraction in affine quantization
 */
TEST_F(Q2_KTensorTest, MinValues)
{
    // Test that min is correctly subtracted
    std::vector<uint8_t> scales(16, 10);
    std::vector<uint8_t> mins(16, 5);
    std::vector<uint8_t> quants(256);
    
    // Set all quants to 0 (minimum quant value)
    for (int i = 0; i < 256; ++i)
    {
        quants[i] = 0;
    }
    
    auto raw_data = create_q2_k_block(1.0f, 1.0f, scales, mins, quants);
    Q2_KTensor tensor({1, 256}, raw_data);

    std::vector<float> decoded(256);
    tensor.decodeRow(0, decoded.data());

    // Expected: dl=1*10=10, ml=1*5=5
    // value = 10*0 - 5 = -5
    for (int i = 0; i < 256; ++i)
    {
        EXPECT_NEAR(decoded[i], -5.0f, 0.1f) << "Element " << i;
    }
}

/**
 * @test AllQuantLevels
 * Verifies all 4 quantization levels (0,1,2,3) are correctly decoded
 */
TEST_F(Q2_KTensorTest, AllQuantLevels)
{
    std::vector<uint8_t> scales(16, 4);
    std::vector<uint8_t> mins(16, 1);
    std::vector<uint8_t> quants(256);
    
    // Create pattern: each 64 elements use one quant level
    for (int i = 0; i < 64; ++i) quants[i] = 0;
    for (int i = 64; i < 128; ++i) quants[i] = 1;
    for (int i = 128; i < 192; ++i) quants[i] = 2;
    for (int i = 192; i < 256; ++i) quants[i] = 3;
    
    auto raw_data = create_q2_k_block(2.0f, 3.0f, scales, mins, quants);
    Q2_KTensor tensor({1, 256}, raw_data);

    std::vector<float> decoded(256);
    tensor.decodeRow(0, decoded.data());

    // Expected: dl=2*4=8, ml=3*1=3
    // quant=0 → 8*0 - 3 = -3
    // quant=1 → 8*1 - 3 = 5
    // quant=2 → 8*2 - 3 = 13
    // quant=3 → 8*3 - 3 = 21
    
    EXPECT_NEAR(decoded[0], -3.0f, 0.1f);    // quant=0
    EXPECT_NEAR(decoded[64], 5.0f, 0.1f);    // quant=1
    EXPECT_NEAR(decoded[128], 13.0f, 0.1f);  // quant=2
    EXPECT_NEAR(decoded[192], 21.0f, 0.1f);  // quant=3
}

/**
 * @test MultipleBlocks
 * Verifies correct decoding across multiple Q2_K blocks
 */
TEST_F(Q2_KTensorTest, MultipleBlocks)
{
    // Create 2 blocks with different scale factors
    std::vector<uint8_t> scales1(16, 2);
    std::vector<uint8_t> mins1(16, 1);
    std::vector<uint8_t> quants1(256, 2);  // All quant=2
    
    std::vector<uint8_t> scales2(16, 4);
    std::vector<uint8_t> mins2(16, 2);
    std::vector<uint8_t> quants2(256, 1);  // All quant=1
    
    auto block1 = create_q2_k_block(1.0f, 1.0f, scales1, mins1, quants1);
    auto block2 = create_q2_k_block(2.0f, 2.0f, scales2, mins2, quants2);
    
    std::vector<uint8_t> raw_data;
    raw_data.insert(raw_data.end(), block1.begin(), block1.end());
    raw_data.insert(raw_data.end(), block2.begin(), block2.end());
    
    Q2_KTensor tensor({1, 512}, raw_data);

    std::vector<float> decoded(512);
    tensor.decodeRow(0, decoded.data());

    // Block 1: dl=1*2=2, ml=1*1=1, quant=2 → 2*2-1 = 3
    EXPECT_NEAR(decoded[0], 3.0f, 0.1f);
    EXPECT_NEAR(decoded[255], 3.0f, 0.1f);

    // Block 2: dl=2*4=8, ml=2*2=4, quant=1 → 8*1-4 = 4
    EXPECT_NEAR(decoded[256], 4.0f, 0.1f);
    EXPECT_NEAR(decoded[511], 4.0f, 0.1f);
}

/**
 * @test CrossGroupBoundary
 * Verifies correct scale/min selection at 128-element group boundaries
 */
TEST_F(Q2_KTensorTest, CrossGroupBoundary)
{
    std::vector<uint8_t> scales(16);
    std::vector<uint8_t> mins(16);
    std::vector<uint8_t> quants(256, 1);
    
    // First 8 scales (group_128=0): scale=5, min=2
    for (int i = 0; i < 8; ++i)
    {
        scales[i] = 5;
        mins[i] = 2;
    }
    
    // Last 8 scales (group_128=1): scale=10, min=4
    for (int i = 8; i < 16; ++i)
    {
        scales[i] = 10;
        mins[i] = 4;
    }
    
    auto raw_data = create_q2_k_block(1.0f, 1.0f, scales, mins, quants);
    Q2_KTensor tensor({1, 256}, raw_data);

    std::vector<float> decoded(256);
    tensor.decodeRow(0, decoded.data());

    // Elements 0-127 use scales[0-7]: dl=5, ml=2 → 5*1-2 = 3
    EXPECT_NEAR(decoded[0], 3.0f, 0.1f);
    EXPECT_NEAR(decoded[127], 3.0f, 0.1f);

    // Elements 128-255 use scales[8-15]: dl=10, ml=4 → 10*1-4 = 6
    EXPECT_NEAR(decoded[128], 6.0f, 0.1f);
    EXPECT_NEAR(decoded[255], 6.0f, 0.1f);
}

/**
 * @test BF16Decoding
 * Verifies BF16 decoding path produces similar results to FP32
 */
TEST_F(Q2_KTensorTest, BF16Decoding)
{
    std::vector<uint8_t> scales(16, 8);
    std::vector<uint8_t> mins(16, 4);
    std::vector<uint8_t> quants(256);
    for (int i = 0; i < 256; ++i)
    {
        quants[i] = i % 4;
    }
    
    auto raw_data = create_q2_k_block(2.0f, 1.0f, scales, mins, quants);
    Q2_KTensor tensor({1, 256}, raw_data);

    std::vector<bfloat16> decoded_bf16(256);
    tensor.decodeRowToBF16(0, decoded_bf16.data());

    // Convert BF16 to FP32 and compare
    std::vector<float> decoded_fp32(256);
    tensor.decodeRow(0, decoded_fp32.data());

    for (int i = 0; i < 256; ++i)
    {
        float bf16_as_fp32 = static_cast<float>(decoded_bf16[i]);
        // BF16 has lower precision, so use looser tolerance
        EXPECT_NEAR(bf16_as_fp32, decoded_fp32[i], std::abs(decoded_fp32[i]) * 0.01f + 0.1f)
            << "Element " << i;
    }
}

/**
 * @test SpanDecoding
 * Verifies decodeSpan() can decode arbitrary element ranges
 */
TEST_F(Q2_KTensorTest, SpanDecoding)
{
    std::vector<uint8_t> scales(16, 8);
    std::vector<uint8_t> mins(16, 4);
    std::vector<uint8_t> quants(256);
    for (int i = 0; i < 256; ++i)
    {
        quants[i] = i % 4;
    }
    
    auto raw_data = create_q2_k_block(2.0f, 1.0f, scales, mins, quants);
    Q2_KTensor tensor({1, 256}, raw_data);

    // Decode full row as reference
    std::vector<float> full_row(256);
    tensor.decodeRow(0, full_row.data());

    // Decode span [64, 192) (128 elements, crossing boundaries)
    std::vector<float> span_decoded(128);
    tensor.decodeSpan(64, 128, span_decoded.data());

    // Verify span matches full row
    for (int i = 0; i < 128; ++i)
    {
        EXPECT_NEAR(span_decoded[i], full_row[64 + i], 0.01f)
            << "Element " << i << " in span";
    }
}

/**
 * @test ErrorHandling
 * Verifies proper error handling for invalid inputs
 */
TEST_F(Q2_KTensorTest, ErrorHandling)
{
    // Invalid data size
    std::vector<uint8_t> bad_data(50);  // Should be 84 bytes
    EXPECT_THROW(Q2_KTensor({1, 256}, bad_data), std::invalid_argument);

    // Valid tensor
    std::vector<uint8_t> scales(16, 8);
    std::vector<uint8_t> mins(16, 4);
    std::vector<uint8_t> quants(256, 1);
    auto raw_data = create_q2_k_block(2.0f, 1.0f, scales, mins, quants);
    Q2_KTensor tensor({4, 64}, raw_data);

    // Out of bounds row access
    std::vector<float> buffer(64);
    EXPECT_THROW(tensor.decodeRow(4, buffer.data()), std::out_of_range);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
