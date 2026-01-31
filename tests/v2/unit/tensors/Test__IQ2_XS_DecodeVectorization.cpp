/**
 * @file Test__IQ2_XS_DecodeVectorization.cpp
 * @brief Comprehensive unit tests for IQ2_XS decode_to_q8_0 SIMD implementations
 *
 * Tests cover:
 * - Scalar correctness (baseline reference)
 * - Multi-block decode correctness
 * - Grid lookup verification
 * - AVX2/AVX512 parity with scalar
 * - Tensor integration (decode_to_q8_0 method)
 * - Auto-dispatch mechanism
 * - Fuzz testing across random inputs
 * - Error handling (bounds checking)
 * - Performance comparison (scalar vs SIMD)
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/BlockStructures.h"
#include "tensors/IQQuantTables.h"
#include "tensors/FP16Utils.h"
#include "utils/CPUFeatures.h"
#include <cmath>
#include <random>
#include <chrono>

using namespace llaminar2;
using namespace llaminar2::simd;

namespace
{
    constexpr float TOLERANCE = 1e-4f;
    constexpr float REL_TOLERANCE = 1e-3f;

    /**
     * @brief Create a synthetic IQ2_XS block with known structure for testing
     */
    IQ2_XSBlock create_test_iq2xs_block(uint16_t d_fp16, uint16_t base_qs = 0, uint8_t base_scale = 8)
    {
        IQ2_XSBlock block;
        block.d = d_fp16;

        // Fill qs with sequential grid indices + sign patterns
        for (size_t i = 0; i < 32; ++i)
        {
            uint16_t grid_idx = (base_qs + i) & 511; // 9-bit grid index (0-511)
            uint16_t sign_idx = i & 127;             // 7-bit sign index (0-127)
            block.qs[i] = grid_idx | (sign_idx << 9);
        }

        // Fill scales with known values
        for (size_t i = 0; i < 8; ++i)
        {
            uint8_t scale_low = (base_scale + i) & 0xf;
            uint8_t scale_high = (base_scale + i + 1) & 0xf;
            block.scales[i] = scale_low | (scale_high << 4);
        }

        return block;
    }

    /**
     * @brief Compare two Q8_0 blocks with tolerance
     */
    bool compare_q8_0_blocks(const Q8_0Block &a, const Q8_0Block &b,
                             float abs_tol = TOLERANCE, float rel_tol = REL_TOLERANCE)
    {
        // Compare scale factors
        float scale_diff = std::abs(a.d - b.d);
        float scale_max = std::max(std::abs(a.d), std::abs(b.d));
        if (scale_diff > abs_tol && (scale_max < 1e-6f || scale_diff / scale_max > rel_tol))
        {
            return false;
        }

        // Compare quantized values
        for (size_t i = 0; i < 32; ++i)
        {
            if (a.qs[i] != b.qs[i])
            {
                // Allow ±1 difference due to rounding
                if (std::abs(a.qs[i] - b.qs[i]) > 1)
                {
                    return false;
                }
            }
        }
        return true;
    }

} // anonymous namespace

// ===================================================================
// TEST 1: ScalarCorrectness - Validate scalar decode produces expected output
// ===================================================================
TEST(Test__IQ2_XS_DecodeVectorization, ScalarCorrectness)
{
    // Create test block with d=1.0 (FP16: 0x3C00)
    IQ2_XSBlock block = create_test_iq2xs_block(0x3C00, 0, 8);

    // Decode first sub-block (32 elements)
    Q8_0Block output;
    decode_iq2xs_to_q8_0_scalar(block, 0, output.qs, &output.d);

    // Verify output is non-zero and scale is reasonable
    float scale_fp32 = fp16_to_fp32(output.d);
    EXPECT_GT(std::abs(scale_fp32), 1e-6f) << "Scale should be non-zero";

    bool has_nonzero = false;
    for (size_t i = 0; i < 32; ++i)
    {
        if (output.qs[i] != 0)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Should have at least one non-zero quantized value";

    // Verify values are in valid int8 range
    for (size_t i = 0; i < 32; ++i)
    {
        EXPECT_GE(output.qs[i], -127) << "Value at index " << i << " too low";
        EXPECT_LE(output.qs[i], 127) << "Value at index " << i << " too high";
    }
}

// ===================================================================
// TEST 2: MultiBlockCorrectness - Verify all 8 sub-blocks decode properly
// ===================================================================
TEST(Test__IQ2_XS_DecodeVectorization, MultiBlockCorrectness)
{
    IQ2_XSBlock block = create_test_iq2xs_block(0x3C00, 100, 8);

    Q8_0Block outputs[8];
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        decode_iq2xs_to_q8_0_scalar(block, sub_idx, outputs[sub_idx].qs, &outputs[sub_idx].d);
    }

    // All sub-blocks should have valid scales
    for (size_t i = 0; i < 8; ++i)
    {
        float scale_fp32 = fp16_to_fp32(outputs[i].d);
        EXPECT_GT(std::abs(scale_fp32), 1e-6f) << "Sub-block " << i << " scale is zero";
    }

    // Sub-blocks should differ (different grid lookups)
    bool found_difference = false;
    for (size_t i = 1; i < 8; ++i)
    {
        if (!compare_q8_0_blocks(outputs[0], outputs[i]))
        {
            found_difference = true;
            break;
        }
    }
    EXPECT_TRUE(found_difference) << "All sub-blocks should not be identical";
}

// ===================================================================
// TEST 3: GridLookupVerification - Verify grid table is used correctly
// ===================================================================
TEST(Test__IQ2_XS_DecodeVectorization, GridLookupVerification)
{
    IQ2_XSBlock block = create_test_iq2xs_block(0x3C00, 0, 8);

    // Set known grid indices for first sub-block
    block.qs[0] = 0;   // Grid index 0
    block.qs[1] = 1;   // Grid index 1
    block.qs[2] = 511; // Grid index 511 (max)

    Q8_0Block output;
    decode_iq2xs_to_q8_0_scalar(block, 0, output.qs, &output.d);

    // Values should be populated (grid lookup occurred)
    EXPECT_NE(output.qs[0], 0);
    EXPECT_NE(output.qs[1], 0);
}

// ===================================================================
// TEST 4: AVX2Parity - AVX2 matches scalar output
// ===================================================================
#ifdef __AVX2__
TEST(Test__IQ2_XS_DecodeVectorization, AVX2Parity)
{
    if (!cpu_supports_avx2())
    {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<uint16_t> dist_d(0x3000, 0x4000); // FP16 range ~0.25-4.0
    std::uniform_int_distribution<uint16_t> dist_qs(0, 0xFFFF);     // Full 16-bit range
    std::uniform_int_distribution<uint8_t> dist_scale(0, 15);       // 4-bit scales

    for (size_t trial = 0; trial < 10; ++trial)
    {
        IQ2_XSBlock block;
        block.d = dist_d(rng);
        for (size_t i = 0; i < 32; ++i)
        {
            block.qs[i] = dist_qs(rng);
        }
        for (size_t i = 0; i < 8; ++i)
        {
            uint8_t low = dist_scale(rng);
            uint8_t high = dist_scale(rng);
            block.scales[i] = low | (high << 4);
        }

        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            Q8_0Block scalar_out, avx2_out;
            decode_iq2xs_to_q8_0_scalar(block, sub_idx, scalar_out.qs, &scalar_out.d);
            decode_iq2xs_to_q8_0_avx2(block, sub_idx, avx2_out.qs, &avx2_out.d);

            EXPECT_TRUE(compare_q8_0_blocks(scalar_out, avx2_out))
                << "Trial " << trial << ", sub-block " << sub_idx << " mismatch";
        }
    }
}
#endif

// ===================================================================
// TEST 5: AVX512Parity - AVX-512 matches scalar output
// ===================================================================
#ifdef __AVX512F__
TEST(Test__IQ2_XS_DecodeVectorization, AVX512Parity)
{
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<uint16_t> dist_d(0x3000, 0x4000);
    std::uniform_int_distribution<uint16_t> dist_qs(0, 0xFFFF);
    std::uniform_int_distribution<uint8_t> dist_scale(0, 15);

    for (size_t trial = 0; trial < 10; ++trial)
    {
        IQ2_XSBlock block;
        block.d = dist_d(rng);
        for (size_t i = 0; i < 32; ++i)
        {
            block.qs[i] = dist_qs(rng);
        }
        for (size_t i = 0; i < 8; ++i)
        {
            uint8_t low = dist_scale(rng);
            uint8_t high = dist_scale(rng);
            block.scales[i] = low | (high << 4);
        }

        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            Q8_0Block scalar_out, avx512_out;
            decode_iq2xs_to_q8_0_scalar(block, sub_idx, scalar_out.qs, &scalar_out.d);
            decode_iq2xs_to_q8_0_avx512(block, sub_idx, avx512_out.qs, &avx512_out.d);

            EXPECT_TRUE(compare_q8_0_blocks(scalar_out, avx512_out))
                << "Trial " << trial << ", sub-block " << sub_idx << " mismatch";
        }
    }
}
#endif

// ===================================================================
// TEST 6: TensorIntegration - Test decode_to_q8_0 tensor method
// ===================================================================
TEST(Test__IQ2_XS_DecodeVectorization, TensorIntegration)
{
    // Create 2D IQ2_XS tensor (1 row, 256 columns = 1 super-block)
    std::vector<size_t> shape = {1, 256};
    std::vector<uint8_t> raw_data(sizeof(IQ2_XSBlock));

    IQ2_XSBlock *block_ptr = reinterpret_cast<IQ2_XSBlock *>(raw_data.data());
    *block_ptr = create_test_iq2xs_block(0x3C00, 0, 8);

    IQ2_XSTensor tensor(shape, raw_data);

    // Decode via tensor method
    Q8_0Block outputs[8];
    tensor.decode_to_q8_0(0, 0, outputs);

    // Verify all 8 sub-blocks were decoded
    for (size_t i = 0; i < 8; ++i)
    {
        float scale_fp32 = fp16_to_fp32(outputs[i].d);
        EXPECT_GT(std::abs(scale_fp32), 1e-6f) << "Sub-block " << i << " scale is zero";

        bool has_nonzero = false;
        for (size_t j = 0; j < 32; ++j)
        {
            if (outputs[i].qs[j] != 0)
            {
                has_nonzero = true;
                break;
            }
        }
        EXPECT_TRUE(has_nonzero) << "Sub-block " << i << " has all zeros";
    }
}

// ===================================================================
// TEST 7: AutoDispatch - Verify dispatch selects correct implementation
// ===================================================================
TEST(Test__IQ2_XS_DecodeVectorization, AutoDispatch)
{
    IQ2_XSBlock block = create_test_iq2xs_block(0x3C00, 0, 8);

    Q8_0Block auto_out, scalar_out;
    decode_iq2xs_to_q8_0(block, 0, auto_out.qs, &auto_out.d);
    decode_iq2xs_to_q8_0_scalar(block, 0, scalar_out.qs, &scalar_out.d);

    // Auto-dispatch should match scalar (or SIMD if available)
    EXPECT_TRUE(compare_q8_0_blocks(scalar_out, auto_out))
        << "Auto-dispatch does not match scalar baseline";
}

// ===================================================================
// TEST 8: FuzzTesting - Random inputs across full range
// ===================================================================
TEST(Test__IQ2_XS_DecodeVectorization, FuzzTesting)
{
    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint16_t> dist_d(0x2000, 0x5000); // Wide FP16 range
    std::uniform_int_distribution<uint16_t> dist_qs(0, 0xFFFF);     // Full 16-bit
    std::uniform_int_distribution<uint8_t> dist_scale(0, 255);      // Full byte range

    for (size_t trial = 0; trial < 100; ++trial)
    {
        IQ2_XSBlock block;
        block.d = dist_d(rng);
        for (size_t i = 0; i < 32; ++i)
        {
            block.qs[i] = dist_qs(rng);
        }
        for (size_t i = 0; i < 8; ++i)
        {
            block.scales[i] = dist_scale(rng);
        }

        // Test all sub-blocks
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            Q8_0Block output;
            EXPECT_NO_THROW({
                decode_iq2xs_to_q8_0_scalar(block, sub_idx, output.qs, &output.d);
            }) << "Scalar decode failed on trial "
               << trial << ", sub-block " << sub_idx;

            // Verify output is sane
            float scale_fp32 = fp16_to_fp32(output.d);
            EXPECT_FALSE(std::isnan(scale_fp32)) << "Scale is NaN";
            EXPECT_FALSE(std::isinf(scale_fp32)) << "Scale is inf";

            for (size_t i = 0; i < 32; ++i)
            {
                EXPECT_GE(output.qs[i], -127);
                EXPECT_LE(output.qs[i], 127);
            }
        }
    }
}

// ===================================================================
// TEST 9: ErrorHandling - Bounds checking
// ===================================================================
TEST(Test__IQ2_XS_DecodeVectorization, ErrorHandling)
{
    std::vector<size_t> shape = {1, 256};
    std::vector<uint8_t> raw_data(sizeof(IQ2_XSBlock));
    IQ2_XSBlock *block_ptr = reinterpret_cast<IQ2_XSBlock *>(raw_data.data());
    *block_ptr = create_test_iq2xs_block(0x3C00, 0, 8);

    IQ2_XSTensor tensor(shape, raw_data);

    Q8_0Block outputs[8];

    // Valid access
    EXPECT_NO_THROW(tensor.decode_to_q8_0(0, 0, outputs));

    // Out of bounds k_block_offset (only 1 block per row)
    EXPECT_THROW(tensor.decode_to_q8_0(0, 1, outputs), std::out_of_range);
}

// ===================================================================
// TEST 10: Performance - Compare scalar vs SIMD throughput
// ===================================================================
TEST(Test__IQ2_XS_DecodeVectorization, Performance)
{
    constexpr size_t NUM_BLOCKS = 1000;
    constexpr size_t NUM_ITERATIONS = 100;

    std::mt19937 rng(9999);
    std::uniform_int_distribution<uint16_t> dist_d(0x3000, 0x4000);
    std::uniform_int_distribution<uint16_t> dist_qs(0, 0xFFFF);
    std::uniform_int_distribution<uint8_t> dist_scale(0, 15);

    // Generate random blocks
    std::vector<IQ2_XSBlock> blocks(NUM_BLOCKS);
    for (auto &block : blocks)
    {
        block.d = dist_d(rng);
        for (size_t i = 0; i < 32; ++i)
        {
            block.qs[i] = dist_qs(rng);
        }
        for (size_t i = 0; i < 8; ++i)
        {
            uint8_t low = dist_scale(rng);
            uint8_t high = dist_scale(rng);
            block.scales[i] = low | (high << 4);
        }
    }

    Q8_0Block output;

    // Benchmark scalar
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
    {
        for (const auto &block : blocks)
        {
            decode_iq2xs_to_q8_0_scalar(block, 0, output.qs, &output.d);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double scalar_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "IQ2_XS decode_to_q8_0 Performance (" << NUM_BLOCKS * NUM_ITERATIONS << " decodes):\n";
    std::cout << "  Scalar:  " << scalar_ms << " ms\n";

#ifdef __AVX2__
    if (cpu_supports_avx2())
    {
        auto t2 = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (const auto &block : blocks)
            {
                decode_iq2xs_to_q8_0_avx2(block, 0, output.qs, &output.d);
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double avx2_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        std::cout << "  AVX2:    " << avx2_ms << " ms (speedup: " << (scalar_ms / avx2_ms) << "x)\n";
    }
#endif

#ifdef __AVX512F__
    if (cpu_supports_avx512())
    {
        auto t4 = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (const auto &block : blocks)
            {
                decode_iq2xs_to_q8_0_avx512(block, 0, output.qs, &output.d);
            }
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        double avx512_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
        std::cout << "  AVX-512: " << avx512_ms << " ms (speedup: " << (scalar_ms / avx512_ms) << "x)\n";
    }
#endif

    // Always pass - this is a benchmark, not a correctness test
    SUCCEED();
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
