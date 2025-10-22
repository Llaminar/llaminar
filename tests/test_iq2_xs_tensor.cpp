/**
 * @file test_iq2_xs_tensor.cpp
 * @brief Comprehensive tests for IQ2_XSTensor
 * 
 * Tests IQ2_XS quantization format implementation:
 * - Grid lookups with 512-entry table (9-bit indices)
 * - Dual scale extraction (low/high nibbles)
 * - Scale alternation pattern (db[l/2])
 * - Sign application patterns
 * - Multi-block decoding
 * - BF16 conversion path
 * - Span decoding
 * - Error handling
 * 
 * @author David Sanftenberg
 * @date 2025-10-21
 */

#include "tensors/IQ2_XSTensor.h"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>

using namespace llaminar;

// ========== Helper Functions ==========

/**
 * @brief Convert FP32 to FP16 (uint16_t representation)
 */
static uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;
    
    // Handle special cases
    if (exp <= -15) {
        // Too small for FP16, return signed zero
        return static_cast<uint16_t>(sign << 15);
    } else if (exp >= 16) {
        // Too large, return infinity
        return static_cast<uint16_t>((sign << 15) | 0x7C00);
    }
    
    // Normal case
    uint32_t h_exp = (exp + 15) & 0x1F;
    uint32_t h_mant = mant >> 13;
    
    // Rounding
    if ((mant & 0x1000) && ((mant & 0x1FFF) != 0x1000 || (h_mant & 1))) {
        h_mant++;
        if (h_mant >= 0x400) {
            h_mant = 0;
            h_exp++;
        }
    }
    
    return static_cast<uint16_t>((sign << 15) | (h_exp << 10) | h_mant);
}

/**
 * @brief Create raw IQ2_XS data with specified parameters
 * 
 * Packs 9-bit grid indices and 7-bit sign indices into qs[] array as uint16_t.
 * Packs scales into scales[] array (2 per sub-block as 4-bit nibbles).
 * Returns raw bytes suitable for IQ2_XSTensor constructor.
 * 
 * @param scale FP32 scale factor (converted to FP16)
 * @param grid_indices Array of 32 grid indices (8 sub-blocks × 4 groups), 9-bit (0-511)
 * @param sign_indices Array of 32 sign indices (8 sub-blocks × 4 groups), 7-bit (0-127)
 * @param scales Array of 8 explicit scales (one uint8_t per sub-block)
 * @param num_blocks Number of blocks to create (default 1)
 * @return Raw uint8_t vector (74 bytes per block)
 */
static std::vector<uint8_t> create_iq2_xs_raw_data(
    float scale,
    const uint16_t* grid_indices,   // 32 values (8 sub-blocks × 4), 9-bit (0-511)
    const uint8_t* sign_indices,    // 32 values (8 sub-blocks × 4), 7-bit (0-127)
    const uint8_t* scales,          // 8 values (one per sub-block, uint8_t)
    size_t num_blocks = 1
) {
    std::vector<uint8_t> raw_data(num_blocks * 74);
    
    for (size_t b = 0; b < num_blocks; ++b) {
        IQ2_XSBlock* block = reinterpret_cast<IQ2_XSBlock*>(raw_data.data() + b * 74);
        block->d = fp32_to_fp16(scale);
        
        // Pack qs[] array: 32 uint16_t = 64 bytes
        // Each uint16_t: grid_idx (9 bits) | (sign_idx << 9) (7 bits)
        for (int i = 0; i < 32; ++i) {
            uint16_t grid_idx = grid_indices[i] & 0x1FF;  // 9 bits: 0-511
            uint16_t sign_idx = sign_indices[i] & 0x7F;   // 7 bits: 0-127
            block->qs[i] = grid_idx | (sign_idx << 9);
        }
        
        // Copy scales[] array: 8 bytes
        std::memcpy(block->scales, scales, 8);
    }
    
    return raw_data;
}

/**
 * @brief Extract grid value from iq2xs_grid lookup table
 * 
 * @param grid_idx Grid index (0-511)
 * @param element_idx Element within grid entry (0-7)
 * @return Grid value as signed int8_t
 */
static int8_t get_grid_value(uint16_t grid_idx, int element_idx) {
    if (grid_idx >= 512) return 0;
    const uint64_t grid_entry = iq2xs_grid[grid_idx];
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&grid_entry);
    return static_cast<int8_t>(bytes[element_idx]);
}

/**
 * @brief Compute expected decoded value manually
 * 
 * @param d Scale factor
 * @param scale Sub-block scale (0-255, but typically low nibble 0-15)
 * @param grid_value Grid value from table
 * @param sign Sign bit (0 or 1)
 * @return Expected decoded float value
 */
static float compute_expected(float d, uint8_t scale_nibble, int8_t grid_value, bool sign) {
    float db = d * (0.5f + static_cast<float>(scale_nibble & 0xf)) * 0.25f;
    return db * static_cast<float>(grid_value) * (sign ? -1.0f : 1.0f);
}

// ========== Test Cases ==========

/**
 * @test BasicDecoding
 * @brief Validates basic grid lookup with known indices
 * 
 * Tests single block with:
 * - All grid indices = 0 (0x0808080808080808 - all 0x08 values)
 * - All signs = 0 (all positive)
 * - Scales = 0x00 (low nibble 0, high nibble 0 → both 0.5 multiplier)
 * 
 * Expected: All values = d * 0.5 * 0.25 * 0x08 = d * 1.0
 */
TEST(IQ2_XSTensorTest, BasicDecoding) {
    float scale = 2.0f;
    
    // All grid indices = 0 → grid values all 0x08 (8)
    uint16_t grid_indices[32] = {0};
    
    // All sign indices = 0 → all positive (ksigns_iq2xs[0] = 0)
    uint8_t sign_indices[32] = {0};
    
    // All scales = 0x00 → low nibble 0, high nibble 0 → both multiplier (0.5 + 0) = 0.5
    uint8_t scales[8] = {0};
    
    auto raw_data = create_iq2_xs_raw_data(scale, grid_indices, sign_indices, scales);
    
    IQ2_XSTensor tensor({1, 256}, raw_data);
    std::vector<float> decoded(tensor.element_count());
    tensor.decode_to_fp32(decoded.data());
    
    ASSERT_EQ(decoded.size(), 256);
    
    // Expected: d * (0.5 + 0) * 0.25 * 8 = 2.0 * 0.5 * 0.25 * 8 = 2.0
    float expected = 2.0f * 0.5f * 0.25f * 8.0f;
    
    for (size_t i = 0; i < decoded.size(); ++i) {
        EXPECT_NEAR(decoded[i], expected, 1e-5f) 
            << "Mismatch at index " << i;
    }
}

/**
 * @test DualScaleExtraction
 * @brief Verifies dual scale unpacking from nibbles
 * 
 * Tests scales array with different low/high nibbles.
 * Each sub-block should use db[0] for groups 0-1 and db[1] for groups 2-3.
 */
TEST(IQ2_XSTensorTest, DualScaleExtraction) {
    float d = 4.0f;
    
    uint16_t grid_indices[32] = {0}; // All grid index 0 (values = 8)
    uint8_t sign_indices[32] = {0};  // All positive
    
    // scales[0] = 0x12: low nibble 2, high nibble 1
    // db[0] = d * (0.5 + 2) * 0.25 = 4.0 * 2.5 * 0.25 = 2.5
    // db[1] = d * (0.5 + 1) * 0.25 = 4.0 * 1.5 * 0.25 = 1.5
    uint8_t scales[8] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};
    
    auto raw_data = create_iq2_xs_raw_data(d, grid_indices, sign_indices, scales);
    
    IQ2_XSTensor tensor({1, 256}, raw_data);
    std::vector<float> decoded(tensor.element_count());
    tensor.decode_to_fp32(decoded.data());
    
    // Verify each sub-block (32 elements each)
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        uint8_t low_nibble = scales[ib32] & 0xf;
        uint8_t high_nibble = scales[ib32] >> 4;
        
        float db0 = d * (0.5f + static_cast<float>(low_nibble)) * 0.25f;
        float db1 = d * (0.5f + static_cast<float>(high_nibble)) * 0.25f;
        
        // Groups 0-1 use db[0], groups 2-3 use db[1]
        float expected0 = db0 * 8.0f;
        float expected1 = db1 * 8.0f;
        
        for (int l = 0; l < 4; ++l) {
            float expected = (l < 2) ? expected0 : expected1;
            for (int j = 0; j < 8; ++j) {
                size_t idx = ib32 * 32 + l * 8 + j;
                EXPECT_NEAR(decoded[idx], expected, 1e-3f)
                    << "Mismatch in sub-block " << ib32 << " group " << l << " element " << j
                    << " (low=" << static_cast<int>(low_nibble) 
                    << ", high=" << static_cast<int>(high_nibble) << ")";
            }
        }
    }
}

/**
 * @test SignApplication
 * @brief Confirms sign bit application using ksigns_iq2xs and kmask_iq2xs
 * 
 * Tests sign index 127 (ksigns_iq2xs[127] = 255 = all bits set).
 * All values should be negative.
 */
TEST(IQ2_XSTensorTest, SignApplication) {
    float d = 1.0f;
    
    uint16_t grid_indices[32] = {0}; // All grid index 0 (values = 8)
    uint8_t sign_indices[32];
    std::fill_n(sign_indices, 32, 127); // ksigns_iq2xs[127] = 255 (all bits set)
    uint8_t scales[8] = {0}; // Scale = 0x00 (both nibbles 0 → multiplier 0.5)
    
    auto raw_data = create_iq2_xs_raw_data(d, grid_indices, sign_indices, scales);
    
    IQ2_XSTensor tensor({1, 256}, raw_data);
    std::vector<float> decoded(tensor.element_count());
    tensor.decode_to_fp32(decoded.data());
    
    // Expected: all negative (ksigns_iq2xs[127] = 255 → all bits set → all negative)
    float expected = -1.0f * 0.5f * 0.25f * 8.0f; // = -1.0
    
    for (size_t i = 0; i < decoded.size(); ++i) {
        EXPECT_NEAR(decoded[i], expected, 1e-5f)
            << "Expected negative value at index " << i;
    }
}

/**
 * @test GridLookup512
 * @brief Tests 9-bit grid indices with larger grid (512 entries)
 * 
 * Uses grid indices:
 * - 0: 0x0808080808080808 (all 0x08 = 8)
 * - 1: 0x080808080808082b (7×0x08, 1×0x2b = 7×8, 1×43)
 * - 256: First entry in second half of grid
 * - 511: Last entry in grid
 */
TEST(IQ2_XSTensorTest, GridLookup512) {
    float d = 1.0f;
    
    // Use grid indices 0, 1, 256, 511 repeatedly
    uint16_t grid_indices[32] = {
        0, 1, 256, 511,  0, 1, 256, 511,  // Sub-block 0-1
        0, 1, 256, 511,  0, 1, 256, 511,  // Sub-block 2-3
        0, 1, 256, 511,  0, 1, 256, 511,  // Sub-block 4-5
        0, 1, 256, 511,  0, 1, 256, 511   // Sub-block 6-7
    };
    
    uint8_t sign_indices[32] = {0}; // All positive
    uint8_t scales[8] = {0}; // Scale = 0x00 (multiplier 0.5)
    
    auto raw_data = create_iq2_xs_raw_data(d, grid_indices, sign_indices, scales);
    
    IQ2_XSTensor tensor({1, 256}, raw_data);
    std::vector<float> decoded(tensor.element_count());
    tensor.decode_to_fp32(decoded.data());
    
    float db = d * 0.5f * 0.25f; // = 0.125
    
    // Verify grid values
    // Grid 0: all 0x08 (8)
    for (int j = 0; j < 8; ++j) {
        EXPECT_NEAR(decoded[j], db * 8.0f, 1e-5f);
    }
    
    // Grid 1: first byte 0x2b (43), rest 0x08 (8)
    EXPECT_NEAR(decoded[8], db * 0x2b, 1e-5f); // First byte
    
    // Grid 256: First entry in second half - verify we can access it
    // iq2xs_grid[256] = 0x190808080808192b
    int8_t grid_256_0 = get_grid_value(256, 0);
    EXPECT_NEAR(decoded[16], db * static_cast<float>(grid_256_0), 1e-5f);
    
    // Grid 511: Last entry - verify we can access it
    // iq2xs_grid[511] = 0x2b2b2b2b2b2b2b2b
    EXPECT_NEAR(decoded[24], db * 0x2b, 1e-5f); // All bytes should be 0x2b
}

/**
 * @test MultipleBlocks
 * @brief Tests multi-block tensor decoding
 * 
 * Creates 3 blocks with different scales to verify block-level parallelization.
 */
TEST(IQ2_XSTensorTest, MultipleBlocks) {
    uint16_t grid_indices[32] = {0}; // All grid index 0
    uint8_t sign_indices[32] = {0};  // All positive
    
    std::vector<uint8_t> raw_data;
    
    // Block 0: scale 2.0, scales all 0x00
    uint8_t scales0[8] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    auto block0 = create_iq2_xs_raw_data(2.0f, grid_indices, sign_indices, scales0);
    raw_data.insert(raw_data.end(), block0.begin(), block0.end());
    
    // Block 1: scale 4.0, scales all 0x11 (low=1, high=1)
    uint8_t scales1[8] = {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11};
    auto block1 = create_iq2_xs_raw_data(4.0f, grid_indices, sign_indices, scales1);
    raw_data.insert(raw_data.end(), block1.begin(), block1.end());
    
    // Block 2: scale 1.0, scales all 0x22 (low=2, high=2)
    uint8_t scales2[8] = {0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22};
    auto block2 = create_iq2_xs_raw_data(1.0f, grid_indices, sign_indices, scales2);
    raw_data.insert(raw_data.end(), block2.begin(), block2.end());
    
    IQ2_XSTensor tensor({3, 256}, raw_data);
    std::vector<float> decoded(tensor.element_count());
    tensor.decode_to_fp32(decoded.data());
    
    ASSERT_EQ(decoded.size(), 768);
    
    // Block 0: d=2.0, scale=0 → db = 2.0 * 0.5 * 0.25 = 0.25, value = 0.25 * 8 = 2.0
    for (size_t i = 0; i < 256; ++i) {
        EXPECT_NEAR(decoded[i], 2.0f, 1e-4f);
    }
    
    // Block 1: d=4.0, scale=1 → db = 4.0 * 1.5 * 0.25 = 1.5, value = 1.5 * 8 = 12.0
    for (size_t i = 256; i < 512; ++i) {
        EXPECT_NEAR(decoded[i], 12.0f, 1e-4f);
    }
    
    // Block 2: d=1.0, scale=2 → db = 1.0 * 2.5 * 0.25 = 0.625, value = 0.625 * 8 = 5.0
    for (size_t i = 512; i < 768; ++i) {
        EXPECT_NEAR(decoded[i], 5.0f, 1e-4f);
    }
}

/**
 * @test ScaleAlternation
 * @brief Verifies db[l/2] scale alternation pattern
 * 
 * Tests that:
 * - Groups 0-1 (l=0,1) use db[0] (low nibble)
 * - Groups 2-3 (l=2,3) use db[1] (high nibble)
 */
TEST(IQ2_XSTensorTest, ScaleAlternation) {
    float d = 8.0f;
    
    uint16_t grid_indices[32] = {0}; // All grid index 0 (values = 8)
    uint8_t sign_indices[32] = {0};  // All positive
    
    // Use scales with distinct low/high nibbles
    // scales[0] = 0x02: low=2, high=0
    // db[0] = 8.0 * (0.5 + 2) * 0.25 = 5.0, value = 5.0 * 8 = 40.0
    // db[1] = 8.0 * (0.5 + 0) * 0.25 = 1.0, value = 1.0 * 8 = 8.0
    uint8_t scales[8] = {0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02};
    
    auto raw_data = create_iq2_xs_raw_data(d, grid_indices, sign_indices, scales);
    
    IQ2_XSTensor tensor({1, 256}, raw_data);
    std::vector<float> decoded(tensor.element_count());
    tensor.decode_to_fp32(decoded.data());
    
    // Verify each sub-block
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        // Groups 0-1 should use low nibble (2)
        float expected_low = 8.0f * (0.5f + 2.0f) * 0.25f * 8.0f; // = 40.0
        
        // Groups 2-3 should use high nibble (0)
        float expected_high = 8.0f * (0.5f + 0.0f) * 0.25f * 8.0f; // = 8.0
        
        for (int l = 0; l < 4; ++l) {
            float expected = (l < 2) ? expected_low : expected_high;
            
            for (int j = 0; j < 8; ++j) {
                size_t idx = ib32 * 32 + l * 8 + j;
                EXPECT_NEAR(decoded[idx], expected, 1e-3f)
                    << "Sub-block " << ib32 << " group " << l << " element " << j
                    << " should use " << ((l < 2) ? "low" : "high") << " nibble";
            }
        }
    }
}

/**
 * @test BF16Decoding
 * @brief Validates BF16 decode path
 * 
 * Decodes to BF16 and verifies values match FP32 within BF16 precision.
 */
TEST(IQ2_XSTensorTest, BF16Decoding) {
    float d = 1.0f;
    
    uint16_t grid_indices[32] = {0};
    uint8_t sign_indices[32] = {0};
    uint8_t scales[8] = {0};
    
    auto raw_data = create_iq2_xs_raw_data(d, grid_indices, sign_indices, scales);
    
    IQ2_XSTensor tensor({1, 256}, raw_data);
    
    std::vector<float> fp32_data(tensor.element_count());
    tensor.decode_to_fp32(fp32_data.data());
    std::vector<bfloat16> bf16_data(tensor.element_count());
    tensor.decode_to_bf16(bf16_data.data());
    
    ASSERT_EQ(bf16_data.size(), 256);
    
    // BF16 has ~3 decimal digits of precision
    // Expected: 1.0 * 0.5 * 0.25 * 8 = 1.0
    for (size_t i = 0; i < bf16_data.size(); ++i) {
        // Convert BF16 to FP32 using struct cast operator
        float bf16_as_fp32 = static_cast<float>(bf16_data[i]);
        
        EXPECT_NEAR(bf16_as_fp32, fp32_data[i], std::abs(fp32_data[i]) * 0.01f + 0.1f)
            << "BF16 value mismatch at index " << i;
    }
}

/**
 * @test SpanDecoding
 * @brief Tests arbitrary range decode (decodeSpan)
 * 
 * Decodes partial ranges and verifies they match full decode.
 */
TEST(IQ2_XSTensorTest, SpanDecoding) {
    uint16_t grid_indices[32] = {0};
    uint8_t sign_indices[32] = {0};
    uint8_t scales[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
    
    std::vector<uint8_t> raw_data;
    
    // Create 2 blocks
    auto block0 = create_iq2_xs_raw_data(1.0f, grid_indices, sign_indices, scales);
    raw_data.insert(raw_data.end(), block0.begin(), block0.end());
    
    auto block1 = create_iq2_xs_raw_data(2.0f, grid_indices, sign_indices, scales);
    raw_data.insert(raw_data.end(), block1.begin(), block1.end());
    
    IQ2_XSTensor tensor({2, 256}, raw_data);
    std::vector<float> full_decode(tensor.element_count());
    tensor.decode_to_fp32(full_decode.data());
    
    // Test various spans
    std::vector<float> span1(100);
    tensor.decodeSpan(0, 100, span1.data());      // First 100 elements
    std::vector<float> span2(150);
    tensor.decodeSpan(200, 150, span2.data());    // Middle span
    std::vector<float> span3(112);
    tensor.decodeSpan(400, 112, span3.data());    // Last 112 elements
    
    ASSERT_EQ(span1.size(), 100);
    ASSERT_EQ(span2.size(), 150);
    ASSERT_EQ(span3.size(), 112);
    
    // Verify span1
    for (size_t i = 0; i < span1.size(); ++i) {
        EXPECT_NEAR(span1[i], full_decode[i], 1e-5f);
    }
    
    // Verify span2
    for (size_t i = 0; i < span2.size(); ++i) {
        EXPECT_NEAR(span2[i], full_decode[200 + i], 1e-5f);
    }
    
    // Verify span3
    for (size_t i = 0; i < span3.size(); ++i) {
        EXPECT_NEAR(span3[i], full_decode[400 + i], 1e-5f);
    }
}

/**
 * @test ErrorHandling
 * @brief Validates exception handling
 * 
 * Tests:
 * - Invalid block count (size mismatch)
 * - Out of range span decode
 * - data() call on quantized tensor
 */
TEST(IQ2_XSTensorTest, ErrorHandling) {
    // Test 1: Block count mismatch
    {
        std::vector<uint8_t> raw_data(2 * 74); // 2 blocks = 512 elements
        EXPECT_THROW(
            IQ2_XSTensor tensor({3, 256}, raw_data), // Need 3 blocks (768 elements)
            std::invalid_argument
        );
    }
    
    // Test 2: Out of range span
    {
        uint16_t grid_indices[32] = {0};
        uint8_t sign_indices[32] = {0};
        uint8_t scales[8] = {0};
        auto raw_data = create_iq2_xs_raw_data(1.0f, grid_indices, sign_indices, scales);
        IQ2_XSTensor tensor({1, 256}, raw_data);
        
        std::vector<float> buffer1(100);
        EXPECT_THROW(tensor.decodeSpan(200, 100, buffer1.data()), std::out_of_range); // 200+100 > 256
        std::vector<float> buffer2(1);
        EXPECT_THROW(tensor.decodeSpan(256, 1, buffer2.data()), std::out_of_range);   // Start beyond end
    }
    
    // Test 3: data() call throws
    {
        uint16_t grid_indices[32] = {0};
        uint8_t sign_indices[32] = {0};
        uint8_t scales[8] = {0};
        auto raw_data = create_iq2_xs_raw_data(1.0f, grid_indices, sign_indices, scales);
        IQ2_XSTensor tensor({1, 256}, raw_data);
        
        EXPECT_THROW(tensor.data(), std::runtime_error);
        const IQ2_XSTensor& const_tensor = tensor;
        EXPECT_THROW(const_tensor.data(), std::runtime_error);
    }
}

// ========== Main ==========

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
