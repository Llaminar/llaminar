/**
 * @file Test__Q5_0Tensor.cpp
 * @brief Unit tests for Q5_0Tensor SIMD path equivalency
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
        class Test__Q5_0Tensor : public ::testing::Test
        {
        protected:
            Q5_0Block createTestBlock()
            {
                Q5_0Block block;
                block.d = 0x3C00; // FP16 1.0

                // Set low 4-bit values (0-15 pattern)
                for (int i = 0; i < 16; ++i)
                {
                    uint8_t low_nibble = i % 16;
                    uint8_t high_nibble = (i + 8) % 16;
                    block.qs[i] = (high_nibble << 4) | low_nibble;
                }

                // Set high bits (alternating pattern for testing)
                for (int i = 0; i < 4; ++i)
                {
                    block.qh[i] = (i % 2 == 0) ? 0xAA : 0x55; // 10101010 or 01010101
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

        TEST_F(Test__Q5_0Tensor, ScalarVsAVX2Equivalency)
        {
            Q5_0Block test_block = createTestBlock();

            std::vector<float> scalar_output(Q5_0Block::BLOCK_SIZE);
            std::vector<float> avx2_output(Q5_0Block::BLOCK_SIZE);

            Q5_0Tensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX2__)
            Q5_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q5_0Block::BLOCK_SIZE, 1e-6f))
                << "Q5_0 scalar and AVX2 paths produce different results";
#else
            GTEST_SKIP() << "AVX2 not available";
#endif
        }

        TEST_F(Test__Q5_0Tensor, ScalarVsAVX512Equivalency)
        {
            Q5_0Block test_block = createTestBlock();

            std::vector<float> scalar_output(Q5_0Block::BLOCK_SIZE);
            std::vector<float> avx512_output(Q5_0Block::BLOCK_SIZE);

            Q5_0Tensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX512F__)
            Q5_0Tensor::decodeBlockAVX512(test_block, avx512_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx512_output.data(),
                                       Q5_0Block::BLOCK_SIZE, 1e-6f))
                << "Q5_0 scalar and AVX512 paths produce different results";
#else
            GTEST_SKIP() << "AVX512 not available";
#endif
        }

        TEST_F(Test__Q5_0Tensor, AVX2VsAVX512Equivalency)
        {
#if defined(__AVX2__) && defined(__AVX512F__)
            Q5_0Block test_block = createTestBlock();

            std::vector<float> avx2_output(Q5_0Block::BLOCK_SIZE);
            std::vector<float> avx512_output(Q5_0Block::BLOCK_SIZE);

            Q5_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());
            Q5_0Tensor::decodeBlockAVX512(test_block, avx512_output.data());

            EXPECT_TRUE(compareOutputs(avx2_output.data(), avx512_output.data(),
                                       Q5_0Block::BLOCK_SIZE, 1e-6f))
                << "Q5_0 AVX2 and AVX512 paths produce different results";
#else
            GTEST_SKIP() << "Both AVX2 and AVX512 required";
#endif
        }

        TEST_F(Test__Q5_0Tensor, EdgeCase_ZeroScale)
        {
            Q5_0Block test_block = createTestBlock();
            test_block.d = 0x0000; // FP16 zero

            std::vector<float> scalar_output(Q5_0Block::BLOCK_SIZE);

            Q5_0Tensor::decodeBlockScalar(test_block, scalar_output.data());

            for (size_t i = 0; i < Q5_0Block::BLOCK_SIZE; ++i)
            {
                EXPECT_FLOAT_EQ(scalar_output[i], 0.0f);
            }

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q5_0Block::BLOCK_SIZE);
            Q5_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q5_0Block::BLOCK_SIZE, 1e-6f));
#endif
        }

        TEST_F(Test__Q5_0Tensor, EdgeCase_AllHighBitsSet)
        {
            Q5_0Block test_block;
            test_block.d = 0x3C00; // FP16 1.0

            // Set all low bits to 0
            for (int i = 0; i < 16; ++i)
            {
                test_block.qs[i] = 0x00;
            }

            // Set all high bits to 1
            for (int i = 0; i < 4; ++i)
            {
                test_block.qh[i] = 0xFF;
            }

            std::vector<float> scalar_output(Q5_0Block::BLOCK_SIZE);

            Q5_0Tensor::decodeBlockScalar(test_block, scalar_output.data());

            // Each element should be: scale * (16 - 16) = 0.0
            // (high bit contributes 16, low bits are 0, centered at 16)
            for (size_t i = 0; i < Q5_0Block::BLOCK_SIZE; ++i)
            {
                EXPECT_NEAR(scalar_output[i], 0.0f, 1e-5f) << "At index " << i;
            }

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q5_0Block::BLOCK_SIZE);
            Q5_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q5_0Block::BLOCK_SIZE, 1e-5f));
#endif
        }

        TEST_F(Test__Q5_0Tensor, EdgeCase_HighBitExtraction)
        {
            Q5_0Block test_block;
            test_block.d = 0x3C00; // FP16 1.0

            // Test specific high bit patterns with low bits = 0
            for (int i = 0; i < 16; ++i)
            {
                test_block.qs[i] = 0x00; // Low bits = 0
            }

            // Q5_0 high bit extraction (from dequantize_row_q5_0):
            //   For j=0..15:
            //     output[j]    uses bit j from qh (bits 0-15)
            //     output[j+16] uses bit j+12 from qh (bits 12-27)
            //
            // Set qh to have bits 0-11 clear, bits 12-31 set
            // Detailed trace:
            //   j=0..11:  output[j]    gets bit 0..11 (clear) → high_bit=0 → (0|0)-16 = -16
            //             output[j+16] gets bit 12..23 (set)  → high_bit=16 → (0|16)-16 = 0
            //   j=12..15: output[j]    gets bit 12..15 (set)  → high_bit=16 → (0|16)-16 = 0
            //             output[j+16] gets bit 24..27 (set)  → high_bit=16 → (0|16)-16 = 0
            //
            // Expected pattern: output[0-11]=-16, output[12-31]=0
            uint32_t qh_value = 0xFFFFF000; // Bits 12-31 set, bits 0-11 clear (little-endian)
            std::memcpy(test_block.qh, &qh_value, sizeof(qh_value));

            std::vector<float> scalar_output(Q5_0Block::BLOCK_SIZE);

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q5_0Block::BLOCK_SIZE);

            Q5_0Tensor::decodeBlockScalar(test_block, scalar_output.data());
            Q5_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            // Verify pattern based on high bit extraction
            for (size_t i = 0; i < 12; ++i)
            {
                EXPECT_NEAR(scalar_output[i], -16.0f, 1e-5f) << "First segment (bits 0-11 clear) at index " << i;
            }
            for (size_t i = 12; i < 32; ++i)
            {
                EXPECT_NEAR(scalar_output[i], 0.0f, 1e-5f) << "Second segment (bits 12-31 set) at index " << i;
            }

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q5_0Block::BLOCK_SIZE, 1e-5f));
#endif
        }

    } // namespace test
} // namespace llaminar2
