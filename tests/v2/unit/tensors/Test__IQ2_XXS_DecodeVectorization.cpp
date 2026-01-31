/**
 * @file Test__IQ2_XXS_DecodeVectorization.cpp
 * @brief Test suite for IQ2_XXS decode_to_q8_0 vectorization (scalar/AVX2/AVX512)
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/IQQuantTables.h"
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <cstring>

using namespace llaminar2;

class IQ2_XXSDecodeTest : public ::testing::Test
{
protected:
    static constexpr size_t SUPER_BLOCK_SIZE = 256;
    static constexpr size_t SUB_BLOCKS_PER_SUPER = 8;
    static constexpr size_t SUB_BLOCK_SIZE = 32;

    void SetUp() override
    {
        // Seed random generator
        gen_.seed(42);
    }

    /**
     * @brief Create a test IQ2_XXS block with known values
     */
    IQ2_XXSBlock createTestBlock()
    {
        IQ2_XXSBlock block;
        block.d = fp32_to_fp16(0.5f); // Scale

        // Initialize qs with grid indices (0-255 valid)
        for (size_t i = 0; i < 32; ++i)
        {
            block.qs[i] = static_cast<uint16_t>((i * 123) % 256); // Deterministic pattern
        }

        return block;
    }

    /**
     * @brief Create random IQ2_XXS block
     */
    IQ2_XXSBlock createRandomBlock()
    {
        IQ2_XXSBlock block;
        std::uniform_real_distribution<float> scale_dist(0.001f, 2.0f);
        block.d = fp32_to_fp16(scale_dist(gen_));

        // Random grid indices
        std::uniform_int_distribution<uint16_t> grid_dist(0, 255);
        for (size_t i = 0; i < 32; ++i)
        {
            block.qs[i] = grid_dist(gen_);
        }

        return block;
    }

    /**
     * @brief Compare two Q8_0 blocks for approximate equality
     */
    bool compareQ8Blocks(const Q8_0Block &a, const Q8_0Block &b, float tolerance = 1e-5f)
    {
        float scale_a = fp16_to_fp32(a.d);
        float scale_b = fp16_to_fp32(b.d);

        if (std::abs(scale_a - scale_b) > tolerance)
        {
            return false;
        }

        for (size_t i = 0; i < Q8_0Block::BLOCK_SIZE; ++i)
        {
            if (a.qs[i] != b.qs[i])
            {
                return false;
            }
        }

        return true;
    }

    std::mt19937 gen_;
};

// =====================================================================
// Test 1: Scalar decode correctness
// =====================================================================
TEST_F(IQ2_XXSDecodeTest, ScalarDecodeCorrectness)
{
    IQ2_XXSBlock block = createTestBlock();

    // Decode each sub-block
    for (size_t sub = 0; sub < SUB_BLOCKS_PER_SUPER; ++sub)
    {
        Q8_0Block output;
        ASSERT_NO_THROW({
            simd::decode_iq2xxs_to_q8_0_scalar(block, sub, output.qs, &output.d);
        }) << "Sub-block "
           << sub << " failed";

        // Verify scale is valid
        float scale = fp16_to_fp32(output.d);
        EXPECT_GT(scale, 0.0f) << "Sub-block " << sub << " scale should be positive";
        EXPECT_LT(scale, 100.0f) << "Sub-block " << sub << " scale unreasonably large";

        // Verify quantized values are in valid range
        for (size_t i = 0; i < Q8_0Block::BLOCK_SIZE; ++i)
        {
            EXPECT_GE(output.qs[i], -128) << "Sub-block " << sub << " qs[" << i << "] underflow";
            EXPECT_LE(output.qs[i], 127) << "Sub-block " << sub << " qs[" << i << "] overflow";
        }
    }
}

// =====================================================================
// Test 2: Multi-block processing
// =====================================================================
TEST_F(IQ2_XXSDecodeTest, MultiBlockProcessing)
{
    constexpr size_t NUM_BLOCKS = 4;
    std::vector<IQ2_XXSBlock> blocks(NUM_BLOCKS);
    for (size_t i = 0; i < NUM_BLOCKS; ++i)
    {
        blocks[i] = createRandomBlock();
    }

    for (size_t block_idx = 0; block_idx < NUM_BLOCKS; ++block_idx)
    {
        for (size_t sub = 0; sub < SUB_BLOCKS_PER_SUPER; ++sub)
        {
            Q8_0Block output;
            ASSERT_NO_THROW({
                simd::decode_iq2xxs_to_q8_0_scalar(blocks[block_idx], sub, output.qs, &output.d);
            }) << "Block "
               << block_idx << " sub-block " << sub << " failed";

            float scale = fp16_to_fp32(output.d);
            EXPECT_GT(scale, 0.0f);
        }
    }
}

// =====================================================================
// Test 3: Grid lookup table usage
// =====================================================================
TEST_F(IQ2_XXSDecodeTest, GridLookupTableUsage)
{
    IQ2_XXSBlock block;
    block.d = fp32_to_fp16(1.0f);

    // Set specific grid indices to test lookup
    uint32_t aux32[2] = {0x03020100, 0x00000000}; // Grid indices: 0, 1, 2, 3
    std::memcpy(block.qs, aux32, sizeof(aux32));

    Q8_0Block output;
    ASSERT_NO_THROW({
        simd::decode_iq2xxs_to_q8_0_scalar(block, 0, output.qs, &output.d);
    });

    // Verify grid values were used (non-zero output expected)
    bool has_nonzero = false;
    for (size_t i = 0; i < Q8_0Block::BLOCK_SIZE; ++i)
    {
        if (output.qs[i] != 0)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Grid lookup should produce non-zero values";
}

#ifdef __AVX2__
// =====================================================================
// Test 4: AVX2 vs Scalar parity
// =====================================================================
TEST_F(IQ2_XXSDecodeTest, AVX2VsScalarParity)
{
    constexpr size_t NUM_TESTS = 20;
    for (size_t test = 0; test < NUM_TESTS; ++test)
    {
        IQ2_XXSBlock block = createRandomBlock();

        for (size_t sub = 0; sub < SUB_BLOCKS_PER_SUPER; ++sub)
        {
            Q8_0Block output_scalar, output_avx2;

            simd::decode_iq2xxs_to_q8_0_scalar(block, sub, output_scalar.qs, &output_scalar.d);
            simd::decode_iq2xxs_to_q8_0_avx2(block, sub, output_avx2.qs, &output_avx2.d);

            EXPECT_TRUE(compareQ8Blocks(output_scalar, output_avx2, 1e-4f))
                << "Test " << test << " sub-block " << sub << " AVX2 mismatch";
        }
    }
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
// =====================================================================
// Test 5: AVX-512 vs Scalar parity
// =====================================================================
TEST_F(IQ2_XXSDecodeTest, AVX512VsScalarParity)
{
    constexpr size_t NUM_TESTS = 20;
    for (size_t test = 0; test < NUM_TESTS; ++test)
    {
        IQ2_XXSBlock block = createRandomBlock();

        for (size_t sub = 0; sub < SUB_BLOCKS_PER_SUPER; ++sub)
        {
            Q8_0Block output_scalar, output_avx512;

            simd::decode_iq2xxs_to_q8_0_scalar(block, sub, output_scalar.qs, &output_scalar.d);
            simd::decode_iq2xxs_to_q8_0_avx512(block, sub, output_avx512.qs, &output_avx512.d);

            EXPECT_TRUE(compareQ8Blocks(output_scalar, output_avx512, 1e-4f))
                << "Test " << test << " sub-block " << sub << " AVX-512 mismatch";
        }
    }
}
#endif

// =====================================================================
// Test 6: Tensor integration
// =====================================================================
TEST_F(IQ2_XXSDecodeTest, TensorIntegration)
{
    // Create 2D tensor: 4 rows × 512 cols (4 rows × 2 super-blocks)
    std::vector<size_t> shape = {4, 512}; // 4 rows, 512 columns
    size_t n_blocks = 4 * 2;              // 4 rows × 2 super-blocks per row
    std::vector<uint8_t> raw_data(n_blocks * sizeof(IQ2_XXSBlock));

    IQ2_XXSBlock *blocks_ptr = reinterpret_cast<IQ2_XXSBlock *>(raw_data.data());
    for (size_t i = 0; i < n_blocks; ++i)
    {
        blocks_ptr[i] = createRandomBlock();
    }

    IQ2_XXSTensor tensor(shape, raw_data);

    // Decode via tensor interface (row_idx, k_block_offset)
    for (size_t row = 0; row < 4; ++row)
    {
        for (size_t k_block = 0; k_block < 2; ++k_block)
        {
            alignas(64) Q8_0Block output[8];
            ASSERT_NO_THROW({
                tensor.decode_to_q8_0(row, k_block, output);
            }) << "Row "
               << row << " k_block " << k_block << " decode failed";

            // Verify all sub-blocks
            for (size_t sub = 0; sub < 8; ++sub)
            {
                float scale = fp16_to_fp32(output[sub].d);
                EXPECT_GT(scale, 0.0f) << "Row " << row << " k_block " << k_block << " sub " << sub;
            }
        }
    }
}

// =====================================================================
// Test 7: Auto-dispatch function
// =====================================================================
TEST_F(IQ2_XXSDecodeTest, AutoDispatchCorrectness)
{
    IQ2_XXSBlock block = createRandomBlock();

    for (size_t sub = 0; sub < SUB_BLOCKS_PER_SUPER; ++sub)
    {
        Q8_0Block output_auto, output_scalar;

        simd::decode_iq2xxs_to_q8_0(block, sub, output_auto.qs, &output_auto.d);
        simd::decode_iq2xxs_to_q8_0_scalar(block, sub, output_scalar.qs, &output_scalar.d);

        EXPECT_TRUE(compareQ8Blocks(output_scalar, output_auto, 1e-4f))
            << "Sub-block " << sub << " auto-dispatch mismatch";
    }
}

// =====================================================================
// Test 8: Fuzz testing
// =====================================================================
TEST_F(IQ2_XXSDecodeTest, FuzzTesting)
{
    constexpr size_t NUM_FUZZ_TESTS = 100;
    for (size_t test = 0; test < NUM_FUZZ_TESTS; ++test)
    {
        IQ2_XXSBlock block = createRandomBlock();

        for (size_t sub = 0; sub < SUB_BLOCKS_PER_SUPER; ++sub)
        {
            Q8_0Block output;
            ASSERT_NO_THROW({
                simd::decode_iq2xxs_to_q8_0(block, sub, output.qs, &output.d);
            }) << "Fuzz test "
               << test << " sub-block " << sub << " failed";

            float scale = fp16_to_fp32(output.d);
            EXPECT_GT(scale, 0.0f);
            EXPECT_LT(scale, 1000.0f);
        }
    }
}

// =====================================================================
// Test 9: Error handling
// =====================================================================
TEST_F(IQ2_XXSDecodeTest, ErrorHandling)
{
    IQ2_XXSBlock block = createTestBlock();
    Q8_0Block output;

    // Out of range sub-block index
    EXPECT_THROW({ simd::decode_iq2xxs_to_q8_0_scalar(block, 8, output.qs, &output.d); }, std::out_of_range);

    EXPECT_THROW({ simd::decode_iq2xxs_to_q8_0_scalar(block, 100, output.qs, &output.d); }, std::out_of_range);

    // Tensor out of range (2D tensor with 1 row, 256 cols = 1 super-block)
    std::vector<size_t> shape = {1, 256}; // 1 row, 256 columns
    std::vector<uint8_t> raw_data(sizeof(IQ2_XXSBlock));
    IQ2_XXSBlock *block_ptr = reinterpret_cast<IQ2_XXSBlock *>(raw_data.data());
    *block_ptr = createTestBlock();

    IQ2_XXSTensor tensor(shape, raw_data);

    alignas(64) Q8_0Block outputs[8];
    // Out of range k_block_offset (only 1 block exists at k_block_offset 0)
    EXPECT_THROW({
        tensor.decode_to_q8_0(0, 1, outputs); // k_block_offset=1 is out of range
    },
                 std::out_of_range);
}

// =====================================================================
// Test 10: Performance comparison (informational)
// =====================================================================
TEST_F(IQ2_XXSDecodeTest, PerformanceComparison)
{
    constexpr size_t NUM_BLOCKS = 1000;
    std::vector<IQ2_XXSBlock> blocks(NUM_BLOCKS);
    for (auto &block : blocks)
    {
        block = createRandomBlock();
    }

    Q8_0Block output;

    // Scalar timing
    auto scalar_start = std::chrono::high_resolution_clock::now();
    for (const auto &block : blocks)
    {
        for (size_t sub = 0; sub < SUB_BLOCKS_PER_SUPER; ++sub)
        {
            simd::decode_iq2xxs_to_q8_0_scalar(block, sub, output.qs, &output.d);
        }
    }
    auto scalar_end = std::chrono::high_resolution_clock::now();
    double scalar_ms = std::chrono::duration<double, std::milli>(scalar_end - scalar_start).count();

#ifdef __AVX2__
    auto avx2_start = std::chrono::high_resolution_clock::now();
    for (const auto &block : blocks)
    {
        for (size_t sub = 0; sub < SUB_BLOCKS_PER_SUPER; ++sub)
        {
            simd::decode_iq2xxs_to_q8_0_avx2(block, sub, output.qs, &output.d);
        }
    }
    auto avx2_end = std::chrono::high_resolution_clock::now();
    double avx2_ms = std::chrono::duration<double, std::milli>(avx2_end - avx2_start).count();
    double avx2_speedup = scalar_ms / avx2_ms;
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
    auto avx512_start = std::chrono::high_resolution_clock::now();
    for (const auto &block : blocks)
    {
        for (size_t sub = 0; sub < SUB_BLOCKS_PER_SUPER; ++sub)
        {
            simd::decode_iq2xxs_to_q8_0_avx512(block, sub, output.qs, &output.d);
        }
    }
    auto avx512_end = std::chrono::high_resolution_clock::now();
    double avx512_ms = std::chrono::duration<double, std::milli>(avx512_end - avx512_start).count();
    double avx512_speedup = scalar_ms / avx512_ms;
#endif

    std::cout << "\n=== IQ2_XXS Decode Performance (" << NUM_BLOCKS << " blocks, "
              << (NUM_BLOCKS * SUB_BLOCKS_PER_SUPER) << " sub-blocks) ===\n";
    std::cout << "Scalar:  " << scalar_ms << " ms\n";
#ifdef __AVX2__
    std::cout << "AVX2:    " << avx2_ms << " ms (speedup: " << avx2_speedup << "x)\n";
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__)
    std::cout << "AVX-512: " << avx512_ms << " ms (speedup: " << avx512_speedup << "x)\n";
#endif
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
