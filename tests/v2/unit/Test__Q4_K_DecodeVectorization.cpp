/**
 * @file Test__Q4_K_DecodeVectorization.cpp
 * @brief Comprehensive test suite for Q4_K decode_to_q8_0 SIMD implementations
 * @author David Sanftenberg
 *
 * Tests:
 * 1. ScalarCorrectness - Verify scalar decode produces expected values
 * 2. MultiBlockCorrectness - Test multiple super-blocks decode correctly
 * 3. ScaleExtraction - Validate get_scale_min_k4 extracts correct 6-bit scales
 * 4. AVX2Parity - Compare AVX2 vs scalar (rel L2 < 1e-5)
 * 5. AVX512Parity - Compare AVX-512 vs scalar (rel L2 < 1e-5)
 * 6. TensorIntegration - Test Q4_KTensor::decode_to_q8_0() method
 * 7. AutoDispatch - Verify runtime dispatch selects correct implementation
 * 8. FuzzTesting - Random data stress testing
 * 9. ErrorHandling - Validate bounds checking and error cases
 * 10. Performance - Benchmark scalar vs AVX2 vs AVX-512
 */

#include <gtest/gtest.h>
#include "../../../src/v2/tensors/Tensors.h"
#include "../../../src/v2/tensors/SIMDHelpers.h"
#include "../../../src/v2/tensors/BlockStructures.h"
#include "../../../src/v2/tensors/FP16Utils.h"
#include <random>
#include <cmath>
#include <chrono>

using namespace llaminar2;

class Test__Q4_K_DecodeVectorization : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create deterministic Q4_K block for testing
        test_block_ = createTestBlock();
    }

    Q4_KBlock createTestBlock()
    {
        Q4_KBlock block;

        // Set FP16 scales
        block.d = fp32_to_fp16(0.125f);
        block.dmin = fp32_to_fp16(0.0625f);

        // Set 6-bit scales (12 bytes encoding 8 scale + 8 min values)
        // Layout for get_scale_min_k4:
        //   For j < 4: scale[j] = q[j] & 63, min[j] = q[j+4] & 63
        //   For j >= 4: scale[j] = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
        //               min[j] = (q[j+4] >> 4) | ((q[j] >> 6) << 4)

        // Scales 0-3 stored in lower 6 bits of q[0-3]
        block.scales[0] = 15; // Scale 0: 15 (bits 0-5)
        block.scales[1] = 20; // Scale 1: 20 (bits 0-5)
        block.scales[2] = 25; // Scale 2: 25 (bits 0-5)
        block.scales[3] = 30; // Scale 3: 30 (bits 0-5)

        // Mins 0-3 stored in lower 6 bits of q[4-7]
        block.scales[4] = 10; // Min 0: 10 (bits 0-5)
        block.scales[5] = 12; // Min 1: 12 (bits 0-5)
        block.scales[6] = 14; // Min 2: 14 (bits 0-5)
        block.scales[7] = 16; // Min 3: 16 (bits 0-5)

        // Scales 4-7 and Mins 4-7 stored in q[8-11]
        // Each byte contains: low 4 bits = scale[4+i], high 4 bits = min[4+i]
        // Plus we need to set high 2 bits in q[0-7] for the full 6-bit values
        block.scales[8] = (8 & 0xF) | ((18 & 0xF) << 4);   // Scale 4 = 8, Min 4 = 18
        block.scales[9] = (10 & 0xF) | ((20 & 0xF) << 4);  // Scale 5 = 10, Min 5 = 20
        block.scales[10] = (12 & 0xF) | ((22 & 0xF) << 4); // Scale 6 = 12, Min 6 = 22
        block.scales[11] = (14 & 0xF) | ((24 & 0xF) << 4); // Scale 7 = 14, Min 7 = 24

        // Fill quantized values (128 bytes, each holds 2 4-bit values)
        for (size_t i = 0; i < 128; ++i)
        {
            block.qs[i] = (i % 16) | ((15 - (i % 16)) << 4);
        }

        return block;
    }

    double computeRelativeL2(const int8_t *a, const int8_t *b, size_t count)
    {
        double sum_sq_diff = 0.0;
        double sum_sq_ref = 0.0;

        for (size_t i = 0; i < count; ++i)
        {
            double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            sum_sq_diff += diff * diff;
            sum_sq_ref += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }

        if (sum_sq_ref < 1e-10)
            return 0.0;
        return std::sqrt(sum_sq_diff / sum_sq_ref);
    }

    Q4_KBlock test_block_;
};

// Test 1: ScalarCorrectness
TEST_F(Test__Q4_K_DecodeVectorization, ScalarCorrectness)
{
    Q8_0Block output;

    // Decode first sub-block (32 elements)
    simd::decode_q4_k_to_q8_0_scalar(test_block_, 0, output.qs, &output.d);

    // Verify output scale is reasonable
    float scale = fp16_to_fp32(output.d);
    EXPECT_GT(scale, 0.0f);
    EXPECT_LT(scale, 10.0f);

    // Verify quantized values are in valid range [-127, 127]
    for (size_t i = 0; i < Q8_0Block::BLOCK_SIZE; ++i)
    {
        EXPECT_GE(output.qs[i], -127);
        EXPECT_LE(output.qs[i], 127);
    }

    // Decode all 8 sub-blocks and verify consistency
    for (size_t sb = 0; sb < 8; ++sb)
    {
        Q8_0Block sb_output;
        simd::decode_q4_k_to_q8_0_scalar(test_block_, sb, sb_output.qs, &sb_output.d);

        // Each sub-block should have valid scale
        float sb_scale = fp16_to_fp32(sb_output.d);
        EXPECT_GT(sb_scale, 0.0f) << "Sub-block " << sb << " has invalid scale";
    }
}

// Test 2: MultiBlockCorrectness
TEST_F(Test__Q4_K_DecodeVectorization, MultiBlockCorrectness)
{
    std::vector<Q4_KBlock> blocks(4);

    // Create varied test blocks
    for (size_t b = 0; b < 4; ++b)
    {
        blocks[b] = test_block_;
        blocks[b].d = fp32_to_fp16(0.1f * (b + 1));
        blocks[b].dmin = fp32_to_fp16(0.05f * (b + 1));
    }

    // Decode each block's sub-blocks
    for (size_t b = 0; b < 4; ++b)
    {
        for (size_t sb = 0; sb < 8; ++sb)
        {
            Q8_0Block output;
            simd::decode_q4_k_to_q8_0_scalar(blocks[b], sb, output.qs, &output.d);

            // Verify each decode succeeds
            float scale = fp16_to_fp32(output.d);
            EXPECT_GT(scale, 0.0f) << "Block " << b << ", sub-block " << sb;

            // Verify values are reasonable
            bool has_nonzero = false;
            for (size_t i = 0; i < Q8_0Block::BLOCK_SIZE; ++i)
            {
                if (output.qs[i] != 0)
                    has_nonzero = true;
            }
            EXPECT_TRUE(has_nonzero) << "Block " << b << ", sub-block " << sb << " is all zeros";
        }
    }
}

// Test 3: ScaleExtraction
TEST_F(Test__Q4_K_DecodeVectorization, ScaleExtraction)
{
    // Test get_scale_min_k4 helper function
    uint8_t scales[12];

    // Set up test scales (6-bit values)
    scales[0] = 15; // Scale 0: 15 (binary: 001111)
    scales[1] = 20; // Scale 1: 20 (binary: 010100)
    scales[2] = 25; // Scale 2: 25 (binary: 011001)
    scales[3] = 30; // Scale 3: 30 (binary: 011110)

    scales[4] = 35; // Min 0: 35 (binary: 100011)
    scales[5] = 40; // Min 1: 40 (binary: 101000)
    scales[6] = 45; // Min 2: 45 (binary: 101101)
    scales[7] = 50; // Min 3: 50 (binary: 110010)

    // Remaining bytes for scales 4-7
    scales[8] = 10;
    scales[9] = 20;
    scales[10] = 30;
    scales[11] = 40;

    // Extract scales for indices 0-3 (simple case)
    for (int j = 0; j < 4; ++j)
    {
        uint8_t d, m;
        simd::get_scale_min_k4(j, scales, &d, &m);

        EXPECT_EQ(d, scales[j] & 63) << "Scale index " << j;
        EXPECT_EQ(m, scales[j + 4] & 63) << "Min index " << j;
    }
}

// Test 4: AVX2Parity
TEST_F(Test__Q4_K_DecodeVectorization, AVX2Parity)
{
#if defined(__AVX2__)
    const double tolerance = 1e-5;

    // Test all 8 sub-blocks
    for (size_t sb = 0; sb < 8; ++sb)
    {
        Q8_0Block scalar_output, avx2_output;

        simd::decode_q4_k_to_q8_0_scalar(test_block_, sb, scalar_output.qs, &scalar_output.d);
        simd::decode_q4_k_to_q8_0_avx2(test_block_, sb, avx2_output.qs, &avx2_output.d);

        // Compare scales
        float scalar_scale = fp16_to_fp32(scalar_output.d);
        float avx2_scale = fp16_to_fp32(avx2_output.d);
        EXPECT_NEAR(scalar_scale, avx2_scale, 1e-4f) << "Sub-block " << sb;

        // Compare quantized values
        double rel_l2 = computeRelativeL2(avx2_output.qs, scalar_output.qs, Q8_0Block::BLOCK_SIZE);
        EXPECT_LT(rel_l2, tolerance) << "Sub-block " << sb << " AVX2 parity failed (rel L2: " << rel_l2 << ")";
    }
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

// Test 5: AVX512Parity
TEST_F(Test__Q4_K_DecodeVectorization, AVX512Parity)
{
#if defined(__AVX512F__)
    const double tolerance = 1e-5;

    // Test all 8 sub-blocks
    for (size_t sb = 0; sb < 8; ++sb)
    {
        Q8_0Block scalar_output, avx512_output;

        simd::decode_q4_k_to_q8_0_scalar(test_block_, sb, scalar_output.qs, &scalar_output.d);
        simd::decode_q4_k_to_q8_0_avx512(test_block_, sb, avx512_output.qs, &avx512_output.d);

        // Compare scales
        float scalar_scale = fp16_to_fp32(scalar_output.d);
        float avx512_scale = fp16_to_fp32(avx512_output.d);
        EXPECT_NEAR(scalar_scale, avx512_scale, 1e-4f) << "Sub-block " << sb;

        // Compare quantized values
        double rel_l2 = computeRelativeL2(avx512_output.qs, scalar_output.qs, Q8_0Block::BLOCK_SIZE);
        EXPECT_LT(rel_l2, tolerance) << "Sub-block " << sb << " AVX-512 parity failed (rel L2: " << rel_l2 << ")";
    }
#else
    GTEST_SKIP() << "AVX-512 not available on this platform";
#endif
}

// Test 6: TensorIntegration
TEST_F(Test__Q4_K_DecodeVectorization, TensorIntegration)
{
    // Create a Q4_K tensor with known data
    std::vector<size_t> shape = {2, 512}; // 2 rows, 512 columns (2 Q4_K blocks per row)
    size_t n_blocks = (shape[0] * shape[1] + Q4_KBlock::BLOCK_SIZE - 1) / Q4_KBlock::BLOCK_SIZE;

    std::vector<uint8_t> raw_data(n_blocks * sizeof(Q4_KBlock));
    Q4_KBlock *blocks = reinterpret_cast<Q4_KBlock *>(raw_data.data());

    for (size_t i = 0; i < n_blocks; ++i)
    {
        blocks[i] = test_block_;
    }

    Q4_KTensor tensor(shape, raw_data);

    // Test decode_to_q8_0 for various blocks
    for (size_t row = 0; row < shape[0]; ++row)
    {
        // Each row has 512 elements = 2 Q4_K blocks = 16 Q8_0 blocks (2 * 8)
        for (size_t kb = 0; kb < 16; ++kb)
        {
            Q8_0Block output;
            EXPECT_NO_THROW({
                tensor.decode_to_q8_0(row, kb, &output);
            }) << "Row "
               << row << ", block " << kb;

            // Verify output is valid
            float scale = fp16_to_fp32(output.d);
            EXPECT_GT(scale, 0.0f) << "Row " << row << ", block " << kb;
        }
    }
}

// Test 7: AutoDispatch
TEST_F(Test__Q4_K_DecodeVectorization, AutoDispatch)
{
    Q8_0Block auto_output, scalar_output;

    // Auto-dispatch should select best available implementation
    simd::decode_q4_k_to_q8_0(test_block_, 0, auto_output.qs, &auto_output.d);
    simd::decode_q4_k_to_q8_0_scalar(test_block_, 0, scalar_output.qs, &scalar_output.d);

    // Results should match scalar (within tolerance for SIMD)
    double rel_l2 = computeRelativeL2(auto_output.qs, scalar_output.qs, Q8_0Block::BLOCK_SIZE);
    EXPECT_LT(rel_l2, 1e-4) << "Auto-dispatch result differs from scalar (rel L2: " << rel_l2 << ")";
}

// Test 8: FuzzTesting
TEST_F(Test__Q4_K_DecodeVectorization, FuzzTesting)
{
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.001f, 1.0f);

    const size_t num_trials = 100;

    for (size_t trial = 0; trial < num_trials; ++trial)
    {
        Q4_KBlock fuzz_block;

        // Random scales
        fuzz_block.d = fp32_to_fp16(scale_dist(rng));
        fuzz_block.dmin = fp32_to_fp16(scale_dist(rng) * 0.5f);

        // Random scale bytes
        for (size_t i = 0; i < 12; ++i)
        {
            fuzz_block.scales[i] = byte_dist(rng);
        }

        // Random quantized values
        for (size_t i = 0; i < 128; ++i)
        {
            fuzz_block.qs[i] = byte_dist(rng);
        }

        // Decode all sub-blocks without crashing
        for (size_t sb = 0; sb < 8; ++sb)
        {
            Q8_0Block output;
            EXPECT_NO_THROW({
                simd::decode_q4_k_to_q8_0_scalar(fuzz_block, sb, output.qs, &output.d);
            }) << "Trial "
               << trial << ", sub-block " << sb;

            // Verify output is finite
            float scale = fp16_to_fp32(output.d);
            EXPECT_TRUE(std::isfinite(scale)) << "Trial " << trial << ", sub-block " << sb;
        }
    }
}

// Test 9: ErrorHandling
TEST_F(Test__Q4_K_DecodeVectorization, ErrorHandling)
{
    Q8_0Block output;

    // Test out-of-range sub-block index
    EXPECT_THROW({ simd::decode_q4_k_to_q8_0_scalar(test_block_, 8, output.qs, &output.d); }, std::out_of_range);

    EXPECT_THROW({ simd::decode_q4_k_to_q8_0_scalar(test_block_, 100, output.qs, &output.d); }, std::out_of_range);

    // Test tensor error handling
    std::vector<size_t> shape = {2, 256};
    std::vector<uint8_t> raw_data(sizeof(Q4_KBlock) * 2);
    Q4_KTensor tensor(shape, raw_data);

    // Test null pointer
    EXPECT_THROW({ tensor.decode_to_q8_0(0, 0, nullptr); }, std::invalid_argument);

    // Test out-of-range row
    EXPECT_THROW({ tensor.decode_to_q8_0(10, 0, &output); }, std::out_of_range);

    // Test out-of-range block offset
    EXPECT_THROW({ tensor.decode_to_q8_0(0, 100, &output); }, std::out_of_range);
}

// Test 10: Performance
TEST_F(Test__Q4_K_DecodeVectorization, Performance)
{
    const size_t num_iterations = 10000;
    Q8_0Block output;

    // Benchmark scalar
    auto scalar_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_iterations; ++i)
    {
        for (size_t sb = 0; sb < 8; ++sb)
        {
            simd::decode_q4_k_to_q8_0_scalar(test_block_, sb, output.qs, &output.d);
        }
    }
    auto scalar_end = std::chrono::high_resolution_clock::now();
    auto scalar_ms = std::chrono::duration_cast<std::chrono::microseconds>(scalar_end - scalar_start).count() / 1000.0;

#if defined(__AVX2__)
    // Benchmark AVX2
    auto avx2_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_iterations; ++i)
    {
        for (size_t sb = 0; sb < 8; ++sb)
        {
            simd::decode_q4_k_to_q8_0_avx2(test_block_, sb, output.qs, &output.d);
        }
    }
    auto avx2_end = std::chrono::high_resolution_clock::now();
    auto avx2_ms = std::chrono::duration_cast<std::chrono::microseconds>(avx2_end - avx2_start).count() / 1000.0;

    double avx2_speedup = scalar_ms / avx2_ms;
    EXPECT_GT(avx2_speedup, 1.0) << "AVX2 should be faster than scalar";

    std::cout << "[Q4_K Performance] Scalar: " << scalar_ms << " ms, "
              << "AVX2: " << avx2_ms << " ms (speedup: " << avx2_speedup << "x)" << std::endl;
#endif

#if defined(__AVX512F__)
    // Benchmark AVX-512
    auto avx512_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_iterations; ++i)
    {
        for (size_t sb = 0; sb < 8; ++sb)
        {
            simd::decode_q4_k_to_q8_0_avx512(test_block_, sb, output.qs, &output.d);
        }
    }
    auto avx512_end = std::chrono::high_resolution_clock::now();
    auto avx512_ms = std::chrono::duration_cast<std::chrono::microseconds>(avx512_end - avx512_start).count() / 1000.0;

    double avx512_speedup = scalar_ms / avx512_ms;
    EXPECT_GT(avx512_speedup, 1.0) << "AVX-512 should be faster than scalar";

    std::cout << "[Q4_K Performance] Scalar: " << scalar_ms << " ms, "
              << "AVX-512: " << avx512_ms << " ms (speedup: " << avx512_speedup << "x)" << std::endl;
#endif
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
