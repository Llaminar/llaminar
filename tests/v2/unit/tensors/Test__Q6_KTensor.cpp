/**
 * @file Test__Q6_KTensor.cpp
 * @brief Unit tests for Q6_KTensor SIMD path equivalency
 * @author David Sanftenberg
 * @date October 29, 2025
 */

#include <gtest/gtest.h>
#include "../../../../src/v2/tensors/Tensors.h"
#include "../../../../src/v2/utils/DebugEnv.h"
#include <vector>
#include <cmath>
#include <cstdlib>

namespace llaminar2
{
    namespace test
    {
        class Test__Q6_KTensor : public ::testing::Test
        {
        protected:
            Q6_KBlock createTestBlock()
            {
                Q6_KBlock block;
                block.d = 0x3C00; // FP16 1.0 (global scale)

                // Initialize all scales (16 sub-blocks)
                for (int i = 0; i < 16; ++i)
                {
                    block.scales[i] = static_cast<int8_t>(i - 8); // Range: -8 to 7
                }

                // Initialize ql (lower 4 bits) with pattern
                for (int i = 0; i < 128; ++i)
                {
                    block.ql[i] = static_cast<uint8_t>(i % 256);
                }

                // Initialize qh (upper 2 bits) with pattern
                for (int i = 0; i < 64; ++i)
                {
                    block.qh[i] = static_cast<uint8_t>((i % 4) * 0x55); // 0x00, 0x55, 0xAA, 0xFF
                }

                return block;
            }

            bool compareOutputs(const float *a, const float *b, size_t count, float tolerance = 1e-6f)
            {
                float max_abs_diff = 0.0f;
                size_t mismatch_count = 0;

                for (size_t i = 0; i < count; ++i)
                {
                    float abs_diff = std::fabs(a[i] - b[i]);
                    max_abs_diff = std::max(max_abs_diff, abs_diff);

                    if (abs_diff > tolerance)
                    {
                        ++mismatch_count;
                        if (mismatch_count <= 5)
                        {
                            std::cout << "Mismatch at [" << i << "]: "
                                      << "a=" << a[i] << ", b=" << b[i]
                                      << ", diff=" << abs_diff << std::endl;
                        }
                    }
                }

                if (mismatch_count > 0)
                {
                    std::cout << "Total mismatches: " << mismatch_count << "/" << count
                              << ", max_abs_diff=" << max_abs_diff << std::endl;
                }

                return mismatch_count == 0;
            }
        };

        TEST_F(Test__Q6_KTensor, ScalarVsAVX2Equivalency)
        {
            Q6_KBlock test_block = createTestBlock();

            std::vector<float> scalar_output(Q6_KBlock::BLOCK_SIZE);
            std::vector<float> avx2_output(Q6_KBlock::BLOCK_SIZE);

            Q6_KTensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX2__)
            Q6_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q6_KBlock::BLOCK_SIZE, 1e-6f))
                << "Q6_K scalar and AVX2 paths produce different results";
#else
            GTEST_SKIP() << "AVX2 not available";
#endif
        }

        TEST_F(Test__Q6_KTensor, ScalarVsAVX512Equivalency)
        {
            Q6_KBlock test_block = createTestBlock();

            std::vector<float> scalar_output(Q6_KBlock::BLOCK_SIZE);
            std::vector<float> avx512_output(Q6_KBlock::BLOCK_SIZE);

            Q6_KTensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX512F__)
            Q6_KTensor::decodeBlockAVX512(test_block, avx512_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx512_output.data(),
                                       Q6_KBlock::BLOCK_SIZE, 1e-6f))
                << "Q6_K scalar and AVX512 paths produce different results";
#else
            GTEST_SKIP() << "AVX512 not available";
#endif
        }

        TEST_F(Test__Q6_KTensor, AVX2VsAVX512Equivalency)
        {
#if defined(__AVX2__) && defined(__AVX512F__)
            Q6_KBlock test_block = createTestBlock();

            std::vector<float> avx2_output(Q6_KBlock::BLOCK_SIZE);
            std::vector<float> avx512_output(Q6_KBlock::BLOCK_SIZE);

            Q6_KTensor::decodeBlockAVX2(test_block, avx2_output.data());
            Q6_KTensor::decodeBlockAVX512(test_block, avx512_output.data());

            EXPECT_TRUE(compareOutputs(avx2_output.data(), avx512_output.data(),
                                       Q6_KBlock::BLOCK_SIZE, 1e-6f))
                << "Q6_K AVX2 and AVX512 paths produce different results";
#else
            GTEST_SKIP() << "Both AVX2 and AVX512 required";
#endif
        }

        TEST_F(Test__Q6_KTensor, EdgeCase_ZeroGlobalScale)
        {
            Q6_KBlock test_block = createTestBlock();
            test_block.d = 0x0000; // FP16 zero (global scale)

            std::vector<float> scalar_output(Q6_KBlock::BLOCK_SIZE);

            Q6_KTensor::decodeBlockScalar(test_block, scalar_output.data());

            // With zero global scale, all outputs should be zero
            for (size_t i = 0; i < Q6_KBlock::BLOCK_SIZE; ++i)
            {
                EXPECT_FLOAT_EQ(scalar_output[i], 0.0f) << "At index " << i;
            }

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q6_KBlock::BLOCK_SIZE);
            Q6_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q6_KBlock::BLOCK_SIZE, 1e-6f));
#endif
        }

        TEST_F(Test__Q6_KTensor, EdgeCase_UniformScales)
        {
            Q6_KBlock test_block;
            test_block.d = 0x4000; // FP16 2.0

            // Set all sub-block scales to same value
            for (int i = 0; i < 16; ++i)
            {
                test_block.scales[i] = 10; // Uniform scale
            }

            // Set all ql to 0
            for (int i = 0; i < 128; ++i)
            {
                test_block.ql[i] = 0x00;
            }

            // Set all qh to 0
            for (int i = 0; i < 64; ++i)
            {
                test_block.qh[i] = 0x00;
            }

            std::vector<float> scalar_output(Q6_KBlock::BLOCK_SIZE);

            Q6_KTensor::decodeBlockScalar(test_block, scalar_output.data());

            // Each value: global_scale * sub_scale * (q - 32)
            // = 2.0 * 10 * (0 - 32) = -640.0
            for (size_t i = 0; i < Q6_KBlock::BLOCK_SIZE; ++i)
            {
                EXPECT_NEAR(scalar_output[i], -640.0f, 1e-4f) << "At index " << i;
            }

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q6_KBlock::BLOCK_SIZE);
            Q6_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q6_KBlock::BLOCK_SIZE, 1e-4f));
#endif
        }

        TEST_F(Test__Q6_KTensor, EdgeCase_MaxQuantValue)
        {
            Q6_KBlock test_block;
            test_block.d = 0x3C00; // FP16 1.0

            // Set all scales to 1
            for (int i = 0; i < 16; ++i)
            {
                test_block.scales[i] = 1;
            }

            // Set ql to maximum (all bits set)
            for (int i = 0; i < 128; ++i)
            {
                test_block.ql[i] = 0xFF;
            }

            // Set qh to maximum (all bits set)
            for (int i = 0; i < 64; ++i)
            {
                test_block.qh[i] = 0xFF;
            }

            std::vector<float> scalar_output(Q6_KBlock::BLOCK_SIZE);

            Q6_KTensor::decodeBlockScalar(test_block, scalar_output.data());

            // Maximum 6-bit value is 63 (0x3F)
            // Result: 1.0 * 1 * (63 - 32) = 31.0
            for (size_t i = 0; i < Q6_KBlock::BLOCK_SIZE; ++i)
            {
                EXPECT_NEAR(scalar_output[i], 31.0f, 1e-5f) << "At index " << i;
            }

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q6_KBlock::BLOCK_SIZE);
            Q6_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q6_KBlock::BLOCK_SIZE, 1e-5f));
#endif
        }

        TEST_F(Test__Q6_KTensor, EdgeCase_NegativeScales)
        {
            Q6_KBlock test_block;
            test_block.d = 0x3C00; // FP16 1.0

            // Set all sub-block scales to negative
            for (int i = 0; i < 16; ++i)
            {
                test_block.scales[i] = -10;
            }

            // Set all quantized values to center (32)
            for (int i = 0; i < 128; ++i)
            {
                test_block.ql[i] = 0x00; // Lower 4 bits = 0
            }

            for (int i = 0; i < 64; ++i)
            {
                test_block.qh[i] = 0xAA; // Pattern: 10101010 (upper 2 bits = 2 per element)
            }

            std::vector<float> scalar_output(Q6_KBlock::BLOCK_SIZE);

            Q6_KTensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q6_KBlock::BLOCK_SIZE);
            Q6_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q6_KBlock::BLOCK_SIZE, 1e-5f));
#endif
        }

        TEST_F(Test__Q6_KTensor, EdgeCase_SubBlockBoundaries)
        {
            // Test that sub-block boundaries are handled correctly (16 sub-blocks × 16 elements)
            Q6_KBlock test_block;
            test_block.d = 0x3C00; // FP16 1.0

            // Set each sub-block to have unique scale
            for (int i = 0; i < 16; ++i)
            {
                test_block.scales[i] = static_cast<int8_t>(i); // 0, 1, 2, ..., 15
            }

            // Set all quantized values to center (32)
            for (int i = 0; i < 128; ++i)
            {
                test_block.ql[i] = 0x00;
            }

            for (int i = 0; i < 64; ++i)
            {
                test_block.qh[i] = 0xAA; // Upper 2 bits = 2 per element
            }

            std::vector<float> scalar_output(Q6_KBlock::BLOCK_SIZE);

            Q6_KTensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q6_KBlock::BLOCK_SIZE);
            Q6_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

            // Verify sub-block boundaries are consistent
            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q6_KBlock::BLOCK_SIZE, 1e-5f));

            // Additionally verify that each sub-block of 16 elements has consistent scaling
            for (int sb = 0; sb < 16; ++sb)
            {
                int8_t expected_scale = test_block.scales[sb];
                float first_value = scalar_output[sb * 16];

                for (int j = 1; j < 16; ++j)
                {
                    // All values in same sub-block should be equal (same scale, same quant values)
                    EXPECT_NEAR(scalar_output[sb * 16 + j], first_value, 1e-5f)
                        << "Sub-block " << sb << ", element " << j;
                }
            }
#endif
        }

    } // namespace test
} // namespace llaminar2
