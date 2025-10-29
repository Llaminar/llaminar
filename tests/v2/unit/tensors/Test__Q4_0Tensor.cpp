/**
 * @file Test__Q4_0Tensor.cpp
 * @brief Unit tests for Q4_0Tensor SIMD path equivalency
 * @author David Sanftenberg
 * @date October 29, 2025
 *
 * Tests verify that scalar, AVX2, and AVX512 dequantization paths produce
 * numerically equivalent results using the LLAMINAR_DEQUANT_SIMD_PATH environment variable.
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
        class Test__Q4_0Tensor : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                // Save original environment variable
                const char *original = std::getenv("LLAMINAR_DEQUANT_SIMD_PATH");
                original_simd_path_ = original ? std::string(original) : "";
            }

            void TearDown() override
            {
                // Restore original environment variable
                if (original_simd_path_.empty())
                {
                    unsetenv("LLAMINAR_DEQUANT_SIMD_PATH");
                }
                else
                {
                    setenv("LLAMINAR_DEQUANT_SIMD_PATH", original_simd_path_.c_str(), 1);
                }
            }

            /**
             * @brief Force a specific SIMD path by setting environment variable and re-initializing DebugEnv
             */
            void setSIMDPath(const std::string &path)
            {
                setenv("LLAMINAR_DEQUANT_SIMD_PATH", path.c_str(), 1);
                // Force DebugEnv to re-read environment variable
                // Note: DebugEnv is a lazy static singleton, so we need to access it to trigger re-init
                // For testing, we'll create fresh blocks and decode them
            }

            /**
             * @brief Create a test Q4_0 block with known values
             */
            Q4_0Block createTestBlock()
            {
                Q4_0Block block;

                // Set scale (FP16)
                block.d = 0x3C00; // FP16 value of 1.0

                // Set quantized values (32 4-bit values in 16 bytes)
                // Values: 0, 1, 2, 3, ..., 15 repeated twice
                for (int i = 0; i < 16; ++i)
                {
                    uint8_t low_nibble = i % 16;
                    uint8_t high_nibble = (i + 8) % 16;
                    block.qs[i] = (high_nibble << 4) | low_nibble;
                }

                return block;
            }

            /**
             * @brief Compare two float arrays with tolerance
             */
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

            std::string original_simd_path_;
        };

        /**
         * @brief Test that scalar and AVX2 paths produce identical results
         */
        TEST_F(Test__Q4_0Tensor, ScalarVsAVX2Equivalency)
        {
            Q4_0Block test_block = createTestBlock();

            std::vector<float> scalar_output(Q4_0Block::BLOCK_SIZE);
            std::vector<float> avx2_output(Q4_0Block::BLOCK_SIZE);

            // Decode with scalar path
            Q4_0Tensor::decodeBlockScalar(test_block, scalar_output.data());

            // Decode with AVX2 path (if available)
#if defined(__AVX2__)
            Q4_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q4_0Block::BLOCK_SIZE, 1e-6f))
                << "Q4_0 scalar and AVX2 paths produce different results";
#else
            GTEST_SKIP() << "AVX2 not available on this platform";
#endif
        }

        /**
         * @brief Test that scalar and AVX512 paths produce identical results
         */
        TEST_F(Test__Q4_0Tensor, ScalarVsAVX512Equivalency)
        {
            Q4_0Block test_block = createTestBlock();

            std::vector<float> scalar_output(Q4_0Block::BLOCK_SIZE);
            std::vector<float> avx512_output(Q4_0Block::BLOCK_SIZE);

            // Decode with scalar path
            Q4_0Tensor::decodeBlockScalar(test_block, scalar_output.data());

            // Decode with AVX512 path (if available)
#if defined(__AVX512F__)
            Q4_0Tensor::decodeBlockAVX512(test_block, avx512_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx512_output.data(),
                                       Q4_0Block::BLOCK_SIZE, 1e-6f))
                << "Q4_0 scalar and AVX512 paths produce different results";
#else
            GTEST_SKIP() << "AVX512 not available on this platform";
#endif
        }

        /**
         * @brief Test that AVX2 and AVX512 paths produce identical results
         */
        TEST_F(Test__Q4_0Tensor, AVX2VsAVX512Equivalency)
        {
#if defined(__AVX2__) && defined(__AVX512F__)
            Q4_0Block test_block = createTestBlock();

            std::vector<float> avx2_output(Q4_0Block::BLOCK_SIZE);
            std::vector<float> avx512_output(Q4_0Block::BLOCK_SIZE);

            Q4_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());
            Q4_0Tensor::decodeBlockAVX512(test_block, avx512_output.data());

            EXPECT_TRUE(compareOutputs(avx2_output.data(), avx512_output.data(),
                                       Q4_0Block::BLOCK_SIZE, 1e-6f))
                << "Q4_0 AVX2 and AVX512 paths produce different results";
#else
            GTEST_SKIP() << "Both AVX2 and AVX512 required for this test";
#endif
        }

        /**
         * @brief Test edge cases: zero scale
         */
        TEST_F(Test__Q4_0Tensor, EdgeCase_ZeroScale)
        {
            Q4_0Block test_block = createTestBlock();
            test_block.d = 0x0000; // FP16 zero

            std::vector<float> scalar_output(Q4_0Block::BLOCK_SIZE);
            std::vector<float> avx2_output(Q4_0Block::BLOCK_SIZE);

            Q4_0Tensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX2__)
            Q4_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            // All outputs should be zero
            for (size_t i = 0; i < Q4_0Block::BLOCK_SIZE; ++i)
            {
                EXPECT_FLOAT_EQ(scalar_output[i], 0.0f) << "Scalar output not zero at index " << i;
                EXPECT_FLOAT_EQ(avx2_output[i], 0.0f) << "AVX2 output not zero at index " << i;
            }
#endif
        }

        /**
         * @brief Test edge cases: all nibbles are 0xF (maximum value)
         */
        TEST_F(Test__Q4_0Tensor, EdgeCase_MaxNibbles)
        {
            Q4_0Block test_block;
            test_block.d = 0x3C00; // FP16 1.0

            for (int i = 0; i < 16; ++i)
            {
                test_block.qs[i] = 0xFF; // All nibbles = 15
            }

            std::vector<float> scalar_output(Q4_0Block::BLOCK_SIZE);

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q4_0Block::BLOCK_SIZE);
            Q4_0Tensor::decodeBlockScalar(test_block, scalar_output.data());
            Q4_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q4_0Block::BLOCK_SIZE, 1e-6f));
#endif
        }

    } // namespace test
} // namespace llaminar2
