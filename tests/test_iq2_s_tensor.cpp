#include "../src/tensors/IQ2_STensor.h"
#include "../src/tensors/IQQuantTables.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>

using namespace llaminar;

/**
 * @brief Helper function to create raw IQ2_S quantized data
 *
 * IQ2_S uses 10-bit grid indices with high 2 bits stored separately in qh array.
 *
 * @param scale FP32 scale factor (converted to FP16)
 * @param grid_indices Array of 32 10-bit grid indices (0-1023)
 * @param sign_indices Array of 32 sign indices (0-127)
 * @param scales Array of 8 explicit scale values (nibbles, 0-15 each)
 * @param num_blocks Number of blocks to generate
 * @return Raw quantized data (82 bytes per block)
 */
std::vector<uint8_t> create_iq2_s_raw_data(
    float scale,
    const uint16_t *grid_indices,   // 32 values, 10-bit (0-1023)
    const uint8_t *sign_indices,    // 32 values, 7-bit (0-127)
    const uint8_t *scales,          // 8 values (explicit scales, 0-15)
    size_t num_blocks = 1)
{
    std::vector<uint8_t> raw_data(num_blocks * 82);

    for (size_t block = 0; block < num_blocks; ++block)
    {
        uint8_t *block_data = raw_data.data() + block * 82;

        // Convert FP32 → FP16
        uint32_t f32;
        std::memcpy(&f32, &scale, sizeof(float));
        uint16_t fp16 = ((f32 >> 16) & 0x8000) |             // Sign bit
                        ((((f32 & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | // Exponent
                        ((f32 >> 13) & 0x03ff);                                 // Mantissa
        std::memcpy(block_data, &fp16, 2);

        // Pack grid indices: low 8 bits → qs[], high 2 bits → qh[]
        uint8_t *qs = block_data + 2;       // 64 bytes
        uint8_t *qh = block_data + 2 + 64;  // 8 bytes
        uint8_t *scales_data = block_data + 2 + 64 + 8; // 8 bytes

        // Zero initialize arrays
        std::memset(qs, 0, 64);
        std::memset(qh, 0, 8);

        // Process 8 sub-blocks of 32 elements
        for (size_t ib32 = 0; ib32 < 8; ++ib32)
        {
            // Pack 4 groups of 8 elements
            for (size_t l = 0; l < 4; ++l)
            {
                size_t idx = ib32 * 4 + l;
                uint16_t grid_idx = grid_indices[idx % 32];

                // Low 8 bits → qs[]
                qs[ib32 * 4 + l] = grid_idx & 0xff;

                // High 2 bits → qh[] at position corresponding to group l
                // Bit shift formula: (8-2*l) gives 8,6,4,2
                // We need to reverse this: store high bits at correct position
                uint8_t high_bits = (grid_idx >> 8) & 0x03;
                qh[ib32] |= (high_bits << (2 * l));
            }

            // Pack explicit scales as nibbles
            uint8_t scale_low = scales[ib32 * 2 % 8];
            uint8_t scale_high = scales[(ib32 * 2 + 1) % 8];
            scales_data[ib32] = (scale_high << 4) | (scale_low & 0x0f);
        }

        // Pack sign indices after first 32 qs values
        // Note: IQ2_S uses sign bytes directly (not as ksigns_iq2xs indices), so no masking
        for (size_t i = 0; i < 32; ++i)
        {
            qs[32 + i] = sign_indices[i % 32];
        }
    }

    return raw_data;
}

/**
 * @brief Test basic IQ2_S decoding with known grid indices
 */
TEST(IQ2_STensorTest, BasicDecoding)
{
    // Setup: Create a simple test case
    const float scale = 2.0f;
    const uint16_t grid_indices[32] = {0, 1, 2, 3, 4, 5, 6, 7,
                                       8, 9, 10, 11, 12, 13, 14, 15,
                                       16, 17, 18, 19, 20, 21, 22, 23,
                                       24, 25, 26, 27, 28, 29, 30, 31};
    const uint8_t sign_indices[32] = {0}; // All positive (sign pattern 0x0000000000000000)
    const uint8_t scales[8] = {8, 8, 8, 8, 8, 8, 8, 8}; // All 8 → (0.5 + 8) * 0.25 = 2.125

    auto raw_data = create_iq2_s_raw_data(scale, grid_indices, sign_indices, scales);

    // Create tensor
    IQ2_STensor tensor({1, 256}, raw_data);

    // Decode
    std::vector<float> output(256);
    tensor.decode_to_fp32(output.data());

    // Verify first few values
    // Expected: d * (0.5 + 8) * 0.25 * grid[idx] = 2.0 * 2.125 * grid[idx] = 4.25 * grid[idx]
    const float expected_scale = scale * (0.5f + 8.0f) * 0.25f;

    for (size_t ib32 = 0; ib32 < 8; ++ib32)
    {
        for (size_t l = 0; l < 4; ++l)
        {
            size_t idx = ib32 * 4 + l;
            uint16_t grid_idx = grid_indices[idx];
            const uint8_t *grid = reinterpret_cast<const uint8_t *>(&iq2s_grid[grid_idx]);

            for (size_t j = 0; j < 8; ++j)
            {
                float expected = expected_scale * static_cast<float>(grid[j]);
                float actual = output[ib32 * 32 + l * 8 + j];
                EXPECT_NEAR(expected, actual, 1e-4f)
                    << "Mismatch at ib32=" << ib32 << ", l=" << l << ", j=" << j
                    << " (grid_idx=" << grid_idx << ")";
            }
        }
    }
}

/**
 * @brief Test dual scale extraction (same as IQ2_XS)
 */
TEST(IQ2_STensorTest, DualScaleExtraction)
{
    const float scale = 1.0f;
    const uint16_t grid_indices[32] = {0}; // All zero grid index
    const uint8_t sign_indices[32] = {0};  // All positive
    const uint8_t scales[8] = {0x3, 0x7, 0xA, 0xF, 0x0, 0x5, 0x9, 0xC}; // Various nibbles

    auto raw_data = create_iq2_s_raw_data(scale, grid_indices, sign_indices, scales);
    IQ2_STensor tensor({1, 256}, raw_data);

    std::vector<float> output(256);
    tensor.decode_to_fp32(output.data());

    // Verify scales for each sub-block
    const float expected_scales[8][2] = {
        {(0.5f + 0x3) * 0.25f, (0.5f + 0x7) * 0.25f}, // scales[0] = 0x73
        {(0.5f + 0xA) * 0.25f, (0.5f + 0xF) * 0.25f}, // scales[1] = 0xFA
        {(0.5f + 0x0) * 0.25f, (0.5f + 0x5) * 0.25f}, // scales[2] = 0x50
        {(0.5f + 0x9) * 0.25f, (0.5f + 0xC) * 0.25f}, // scales[3] = 0xC9
        {(0.5f + 0x3) * 0.25f, (0.5f + 0x7) * 0.25f}, // Repeating...
        {(0.5f + 0xA) * 0.25f, (0.5f + 0xF) * 0.25f},
        {(0.5f + 0x0) * 0.25f, (0.5f + 0x5) * 0.25f},
        {(0.5f + 0x9) * 0.25f, (0.5f + 0xC) * 0.25f}};

    // Check that decoding used correct scales
    // All grid[0] values are 0x08, so output should be scale * grid[0] = scale * 0x08
    const uint8_t *grid0 = reinterpret_cast<const uint8_t *>(&iq2s_grid[0]);
    for (size_t ib32 = 0; ib32 < 8; ++ib32)
    {
        for (size_t l = 0; l < 4; ++l)
        {
            float expected_scale = expected_scales[ib32][l / 2]; // Alternating scale
            for (size_t j = 0; j < 8; ++j)
            {
                float expected = scale * expected_scale * static_cast<float>(grid0[j]);
                float actual = output[ib32 * 32 + l * 8 + j];
                EXPECT_NEAR(expected, actual, 1e-5f)
                    << "Scale mismatch at ib32=" << ib32 << ", l=" << l << ", j=" << j;
            }
        }
    }
}

/**
 * @brief Test sign application
 */
TEST(IQ2_STensorTest, SignApplication)
{
    const float scale = 1.0f;
    const uint16_t grid_indices[32] = {0}; // All zero grid index (0x0808080808080808)
    const uint8_t sign_indices[32] = {
        0xFF, 0xFF, 0xFF, 0xFF, // All negative (all 8 bits set)
        0x00, 0x00, 0x00, 0x00, // All positive (all bits clear)
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00};
    const uint8_t scales[8] = {8, 8, 8, 8, 8, 8, 8, 8};

    auto raw_data = create_iq2_s_raw_data(scale, grid_indices, sign_indices, scales);
    IQ2_STensor tensor({1, 256}, raw_data);

    std::vector<float> output(256);
    tensor.decode_to_fp32(output.data());

    // Verify signs
    const float expected_scale = scale * (0.5f + 8.0f) * 0.25f;
    const uint8_t *grid0 = reinterpret_cast<const uint8_t *>(&iq2s_grid[0]);

    for (size_t ib32 = 0; ib32 < 8; ++ib32)
    {
        for (size_t l = 0; l < 4; ++l)
        {
            // Pattern alternates by ib32: even ib32 → negative, odd ib32 → positive
            bool should_be_negative = (ib32 % 2 == 0);
            for (size_t j = 0; j < 8; ++j)
            {
                float expected = expected_scale * static_cast<float>(grid0[j]);
                if (should_be_negative)
                    expected = -expected;
                float actual = output[ib32 * 32 + l * 8 + j];
                EXPECT_NEAR(expected, actual, 1e-5f)
                    << "Sign error at ib32=" << ib32 << ", l=" << l << ", j=" << j;
            }
        }
    }
}

/**
 * @brief Test 10-bit grid index extraction (NEW for IQ2_S)
 */
TEST(IQ2_STensorTest, GridLookup1024)
{
    const float scale = 1.0f;
    const uint8_t sign_indices[32] = {0}; // All positive
    const uint8_t scales[8] = {8, 8, 8, 8, 8, 8, 8, 8};

    // Test corner cases: 0, 1, 512, 1023
    const uint16_t grid_indices[32] = {
        0, 1, 512, 1023, // First group
        0, 1, 512, 1023,
        0, 1, 512, 1023,
        0, 1, 512, 1023,
        0, 1, 512, 1023,
        0, 1, 512, 1023,
        0, 1, 512, 1023,
        0, 1, 512, 1023};

    auto raw_data = create_iq2_s_raw_data(scale, grid_indices, sign_indices, scales);
    IQ2_STensor tensor({1, 256}, raw_data);

    std::vector<float> output(256);
    tensor.decode_to_fp32(output.data());

    // Verify that correct grids were looked up
    const float expected_scale = scale * (0.5f + 8.0f) * 0.25f;
    const uint16_t test_indices[4] = {0, 1, 512, 1023};

    for (size_t ib32 = 0; ib32 < 8; ++ib32)
    {
        for (size_t l = 0; l < 4; ++l)
        {
            uint16_t grid_idx = test_indices[l];
            const uint8_t *grid = reinterpret_cast<const uint8_t *>(&iq2s_grid[grid_idx]);

            for (size_t j = 0; j < 8; ++j)
            {
                float expected = expected_scale * static_cast<float>(grid[j]);
                float actual = output[ib32 * 32 + l * 8 + j];
                EXPECT_NEAR(expected, actual, 1e-4f)
                    << "Grid lookup error at ib32=" << ib32 << ", l=" << l
                    << ", grid_idx=" << grid_idx;
            }
        }
    }
}

/**
 * @brief Test qh bit extraction (NEW for IQ2_S)
 *
 * Verifies that the high 2 bits are correctly extracted from qh array
 * using the bit shift formula (8-2*l).
 */
TEST(IQ2_STensorTest, QHBitExtraction)
{
    const float scale = 1.0f;
    const uint8_t sign_indices[32] = {0};
    const uint8_t scales[8] = {8, 8, 8, 8, 8, 8, 8, 8};

    // Test indices that differ only in high 2 bits
    // Indices: 0x000 (0), 0x100 (256), 0x200 (512), 0x300 (768)
    const uint16_t grid_indices[32] = {
        0x000, 0x100, 0x200, 0x300, // Group with different high bits
        0x000, 0x100, 0x200, 0x300,
        0x000, 0x100, 0x200, 0x300,
        0x000, 0x100, 0x200, 0x300,
        0x000, 0x100, 0x200, 0x300,
        0x000, 0x100, 0x200, 0x300,
        0x000, 0x100, 0x200, 0x300,
        0x000, 0x100, 0x200, 0x300};

    auto raw_data = create_iq2_s_raw_data(scale, grid_indices, sign_indices, scales);
    IQ2_STensor tensor({1, 256}, raw_data);

    std::vector<float> output(256);
    tensor.decode_to_fp32(output.data());

    // Verify correct grids used
    const float expected_scale = scale * (0.5f + 8.0f) * 0.25f;
    const uint16_t test_indices[4] = {0x000, 0x100, 0x200, 0x300};

    for (size_t ib32 = 0; ib32 < 8; ++ib32)
    {
        for (size_t l = 0; l < 4; ++l)
        {
            uint16_t grid_idx = test_indices[l];
            const uint8_t *grid = reinterpret_cast<const uint8_t *>(&iq2s_grid[grid_idx]);

            for (size_t j = 0; j < 8; ++j)
            {
                float expected = expected_scale * static_cast<float>(grid[j]);
                float actual = output[ib32 * 32 + l * 8 + j];
                EXPECT_NEAR(expected, actual, 1e-4f)
                    << "QH bit extraction error at ib32=" << ib32 << ", l=" << l
                    << ", grid_idx=0x" << std::hex << grid_idx << std::dec;
            }
        }
    }
}

/**
 * @brief Test bit shift formula (NEW for IQ2_S)
 *
 * Verifies that the bit shift formula (8-2*l) correctly extracts high bits:
 * - l=0: shift 8 → bits 8-9
 * - l=1: shift 6 → bits 6-7
 * - l=2: shift 4 → bits 4-5
 * - l=3: shift 2 → bits 2-3
 */
TEST(IQ2_STensorTest, BitShiftFormula)
{
    const float scale = 1.0f;
    const uint8_t sign_indices[32] = {0};
    const uint8_t scales[8] = {8, 8, 8, 8, 8, 8, 8, 8};

    // Use indices that test all bit positions
    const uint16_t grid_indices[32] = {
        0x000, 0x001, 0x002, 0x003, // l=0,1,2,3 with minimal high bits
        0x100, 0x101, 0x102, 0x103,
        0x200, 0x201, 0x202, 0x203,
        0x300, 0x301, 0x302, 0x303,
        0x000, 0x001, 0x002, 0x003,
        0x100, 0x101, 0x102, 0x103,
        0x200, 0x201, 0x202, 0x203,
        0x300, 0x301, 0x302, 0x303};

    auto raw_data = create_iq2_s_raw_data(scale, grid_indices, sign_indices, scales);
    IQ2_STensor tensor({1, 256}, raw_data);

    std::vector<float> output(256);
    tensor.decode_to_fp32(output.data());

    // Verify correct index reconstruction
    const float expected_scale = scale * (0.5f + 8.0f) * 0.25f;

    for (size_t ib32 = 0; ib32 < 8; ++ib32)
    {
        for (size_t l = 0; l < 4; ++l)
        {
            size_t idx = ib32 * 4 + l;
            uint16_t grid_idx = grid_indices[idx % 32];
            const uint8_t *grid = reinterpret_cast<const uint8_t *>(&iq2s_grid[grid_idx]);

            for (size_t j = 0; j < 8; ++j)
            {
                float expected = expected_scale * static_cast<float>(grid[j]);
                float actual = output[ib32 * 32 + l * 8 + j];
                EXPECT_NEAR(expected, actual, 1e-4f)
                    << "Bit shift formula error at ib32=" << ib32 << ", l=" << l
                    << ", grid_idx=0x" << std::hex << grid_idx << std::dec;
            }
        }
    }
}

/**
 * @brief Test multiple blocks
 */
TEST(IQ2_STensorTest, MultipleBlocks)
{
    const size_t num_blocks = 4;
    const float scale = 1.5f;
    const uint16_t grid_indices[32] = {
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
        160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310};
    const uint8_t sign_indices[32] = {0};
    const uint8_t scales[8] = {8, 8, 8, 8, 8, 8, 8, 8};

    auto raw_data = create_iq2_s_raw_data(scale, grid_indices, sign_indices, scales, num_blocks);
    IQ2_STensor tensor({1, num_blocks * 256}, raw_data);

    std::vector<float> output(num_blocks * 256);
    tensor.decode_to_fp32(output.data());

    // Verify all blocks decoded identically
    for (size_t block = 1; block < num_blocks; ++block)
    {
        for (size_t i = 0; i < 256; ++i)
        {
            EXPECT_FLOAT_EQ(output[i], output[block * 256 + i])
                << "Block " << block << " differs at index " << i;
        }
    }
}

/**
 * @brief Test BF16 decoding
 */
TEST(IQ2_STensorTest, BF16Decoding)
{
    const float scale = 2.0f;
    const uint16_t grid_indices[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    const uint8_t sign_indices[32] = {0};
    const uint8_t scales[8] = {8, 8, 8, 8, 8, 8, 8, 8};

    auto raw_data = create_iq2_s_raw_data(scale, grid_indices, sign_indices, scales);
    IQ2_STensor tensor({1, 256}, raw_data);

    // Decode to FP32
    std::vector<float> fp32_output(256);
    tensor.decode_to_fp32(fp32_output.data());

    // Decode to BF16
    std::vector<bfloat16> bf16_output(256);
    tensor.decode_to_bf16(bf16_output.data());

    // Compare (allow BF16 precision loss)
    for (size_t i = 0; i < 256; ++i)
    {
        float bf16_as_fp32 = static_cast<float>(bf16_output[i]);
        float error = std::abs(fp32_output[i] - bf16_as_fp32);
        float rel_error = error / (std::abs(fp32_output[i]) + 1e-6f);
        EXPECT_LT(rel_error, 0.01f) << "BF16 precision loss too high at index " << i;
    }
}

/**
 * @brief Test span decoding
 */
TEST(IQ2_STensorTest, SpanDecoding)
{
    const float scale = 1.0f;
    const uint16_t grid_indices[32] = {
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
        160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310};
    const uint8_t sign_indices[32] = {0};
    const uint8_t scales[8] = {8, 8, 8, 8, 8, 8, 8, 8};

    auto raw_data = create_iq2_s_raw_data(scale, grid_indices, sign_indices, scales, 2);
    IQ2_STensor tensor({1, 512}, raw_data);

    // Decode full tensor
    std::vector<float> full_output(512);
    tensor.decode_to_fp32(full_output.data());

    // Decode span [100, 300)
    std::vector<float> span_output(200);
    tensor.decodeSpan(100, 200, span_output.data());

    // Compare
    for (size_t i = 0; i < 200; ++i)
    {
        EXPECT_FLOAT_EQ(full_output[100 + i], span_output[i])
            << "Span decode mismatch at offset " << i;
    }
}

/**
 * @brief Test error handling
 */
TEST(IQ2_STensorTest, ErrorHandling)
{
    const float scale = 1.0f;
    const uint16_t grid_indices[32] = {0};
    const uint8_t sign_indices[32] = {0};
    const uint8_t scales[8] = {8, 8, 8, 8, 8, 8, 8, 8};

    // Test: Wrong data size
    {
        auto raw_data = create_iq2_s_raw_data(scale, grid_indices, sign_indices, scales);
        raw_data.resize(80); // Too small
        EXPECT_THROW(IQ2_STensor({1, 256}, raw_data), std::invalid_argument);
    }

    // Test: Non-multiple of 256 elements
    {
        auto raw_data = create_iq2_s_raw_data(scale, grid_indices, sign_indices, scales);
        EXPECT_THROW(IQ2_STensor({1, 255}, raw_data), std::invalid_argument);
    }

    // Test: Row out of range
    {
        auto raw_data = create_iq2_s_raw_data(scale, grid_indices, sign_indices, scales);
        IQ2_STensor tensor({1, 1, 256}, raw_data);
        std::vector<float> output(256);
        EXPECT_THROW(tensor.decodeRow(1, output.data()), std::out_of_range);
    }

    // Test: Span out of range
    {
        auto raw_data = create_iq2_s_raw_data(scale, grid_indices, sign_indices, scales);
        IQ2_STensor tensor({1, 256}, raw_data);
        std::vector<float> output(200);
        EXPECT_THROW(tensor.decodeSpan(100, 200, output.data()), std::out_of_range);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
