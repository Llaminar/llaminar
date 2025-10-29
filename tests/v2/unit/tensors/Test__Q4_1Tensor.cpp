/**
 * @file Test__Q4_1Tensor.cpp
 * @brief Unit tests for Q4_1Tensor SIMD path equivalency
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
        class Test__Q4_1Tensor : public ::testing::Test
        {
        protected:
            Q4_1Block createTestBlock()
            {
                Q4_1Block block;
                block.d = 0x3C00; // FP16 1.0 (scale)
                block.m = 0x3800; // FP16 0.5 (min)

                for (int i = 0; i < 16; ++i)
                {
                    uint8_t low_nibble = i % 16;
                    uint8_t high_nibble = (i + 8) % 16;
                    block.qs[i] = (high_nibble << 4) | low_nibble;
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
                        if (mismatch_count <= 3)
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

        TEST_F(Test__Q4_1Tensor, ScalarVsAVX2Equivalency)
        {
            Q4_1Block test_block = createTestBlock();

            std::vector<float> scalar_output(Q4_1Block::BLOCK_SIZE);
            std::vector<float> avx2_output(Q4_1Block::BLOCK_SIZE);

            Q4_1Tensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX2__)
            Q4_1Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q4_1Block::BLOCK_SIZE, 1e-6f))
                << "Q4_1 scalar and AVX2 paths produce different results";
#else
            GTEST_SKIP() << "AVX2 not available";
#endif
        }

        TEST_F(Test__Q4_1Tensor, ScalarVsAVX512Equivalency)
        {
            Q4_1Block test_block = createTestBlock();

            std::vector<float> scalar_output(Q4_1Block::BLOCK_SIZE);
            std::vector<float> avx512_output(Q4_1Block::BLOCK_SIZE);

            Q4_1Tensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX512F__)
            Q4_1Tensor::decodeBlockAVX512(test_block, avx512_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx512_output.data(),
                                       Q4_1Block::BLOCK_SIZE, 1e-6f))
                << "Q4_1 scalar and AVX512 paths produce different results";
#else
            GTEST_SKIP() << "AVX512 not available";
#endif
        }

        TEST_F(Test__Q4_1Tensor, AVX2VsAVX512Equivalency)
        {
#if defined(__AVX2__) && defined(__AVX512F__)
            Q4_1Block test_block = createTestBlock();

            std::vector<float> avx2_output(Q4_1Block::BLOCK_SIZE);
            std::vector<float> avx512_output(Q4_1Block::BLOCK_SIZE);

            Q4_1Tensor::decodeBlockAVX2(test_block, avx2_output.data());
            Q4_1Tensor::decodeBlockAVX512(test_block, avx512_output.data());

            EXPECT_TRUE(compareOutputs(avx2_output.data(), avx512_output.data(),
                                       Q4_1Block::BLOCK_SIZE, 1e-6f))
                << "Q4_1 AVX2 and AVX512 paths produce different results";
#else
            GTEST_SKIP() << "Both AVX2 and AVX512 required";
#endif
        }

        TEST_F(Test__Q4_1Tensor, EdgeCase_ZeroScaleAndMin)
        {
            Q4_1Block test_block = createTestBlock();
            test_block.d = 0x0000; // FP16 zero (scale)
            test_block.m = 0x0000; // FP16 zero (min)

            std::vector<float> scalar_output(Q4_1Block::BLOCK_SIZE);

            Q4_1Tensor::decodeBlockScalar(test_block, scalar_output.data());

            for (size_t i = 0; i < Q4_1Block::BLOCK_SIZE; ++i)
            {
                EXPECT_FLOAT_EQ(scalar_output[i], 0.0f);
            }

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q4_1Block::BLOCK_SIZE);
            Q4_1Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q4_1Block::BLOCK_SIZE, 1e-6f));
#endif
        }

        TEST_F(Test__Q4_1Tensor, EdgeCase_NonZeroMin)
        {
            Q4_1Block test_block;
            test_block.d = 0x4000; // FP16 2.0
            test_block.m = 0xC000; // FP16 -2.0

            for (int i = 0; i < 16; ++i)
            {
                test_block.qs[i] = 0x00; // All nibbles = 0
            }

            std::vector<float> scalar_output(Q4_1Block::BLOCK_SIZE);

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q4_1Block::BLOCK_SIZE);
            Q4_1Tensor::decodeBlockScalar(test_block, scalar_output.data());
            Q4_1Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            // Result should be: 2.0 * 0 + (-2.0) = -2.0 for all elements
            for (size_t i = 0; i < Q4_1Block::BLOCK_SIZE; ++i)
            {
                EXPECT_NEAR(scalar_output[i], -2.0f, 1e-5f);
            }

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q4_1Block::BLOCK_SIZE, 1e-5f));
#endif
        }

    } // namespace test
} // namespace llaminar2
