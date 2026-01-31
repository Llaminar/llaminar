/**
 * @file Test__Q16FixedScaleQuantization.cpp
 * @brief Unit tests for Q16_1 fixed-scale quantization with VNNI-safe clipping
 *
 * Tests the copyFrom_fp32_fixed_scale() method which is CRITICAL for integer
 * attention. Unlike adaptive quantization, this uses a fixed scale and clips
 * INT16 values to prevent VNNI INT32 overflow.
 *
 * See: docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md "VNNI OVERFLOW PREVENTION CONTRACT"
 * See: kernels/cpu/attention/q16_1/VNNISafetyConstants.h for safety limits
 *
 * @author Llaminar Team
 * @date 2025-01-01
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "kernels/cpu/attention/q16_1/VNNISafetyConstants.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

namespace llaminar2
{
    namespace test
    {

        using namespace vnni_safety;

        // ============================================================================
        // Test Constants
        // ============================================================================

        constexpr float DEFAULT_KV_CACHE_SCALE = 8.0f;
        constexpr int HEAD_DIM_64 = 64;
        constexpr int HEAD_DIM_128 = 128;
        constexpr int HEAD_DIM_192 = 192;

        // ============================================================================
        // Test Fixture
        // ============================================================================

        class Q16FixedScaleQuantizationTest : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                // Seed for reproducible tests
                gen_.seed(42);
            }

            // Generate random FP32 values in range [-range, +range]
            std::vector<float> generate_random_fp32(size_t count, float range)
            {
                std::uniform_real_distribution<float> dist(-range, range);
                std::vector<float> data(count);
                for (size_t i = 0; i < count; ++i)
                {
                    data[i] = dist(gen_);
                }
                return data;
            }

            // Generate FP32 values with some outliers beyond safe range
            std::vector<float> generate_with_outliers(size_t count, float normal_range, float outlier_range,
                                                      float outlier_ratio = 0.05f)
            {
                std::uniform_real_distribution<float> normal_dist(-normal_range, normal_range);
                std::uniform_real_distribution<float> outlier_dist(-outlier_range, outlier_range);
                std::uniform_real_distribution<float> coin(0.0f, 1.0f);

                std::vector<float> data(count);
                for (size_t i = 0; i < count; ++i)
                {
                    if (coin(gen_) < outlier_ratio)
                    {
                        data[i] = outlier_dist(gen_);
                    }
                    else
                    {
                        data[i] = normal_dist(gen_);
                    }
                }
                return data;
            }

            // Get max absolute INT16 value from Q16 tensor
            int16_t get_max_abs_int16(const Q16_1Tensor &tensor)
            {
                int16_t max_abs = 0;
                const size_t n_blocks = tensor.total_blocks();
                const auto *blocks = tensor.as_block_32();
                EXPECT_NE(blocks, nullptr);

                for (size_t b = 0; b < n_blocks; ++b)
                {
                    for (size_t i = 0; i < Q16_1Block::BLOCK_SIZE; ++i)
                    {
                        max_abs = std::max(max_abs, static_cast<int16_t>(std::abs(blocks[b].qs[i])));
                    }
                }
                return max_abs;
            }

            // Verify all INT16 values are within safe limit
            bool verify_all_within_limit(const Q16_1Tensor &tensor, int16_t limit)
            {
                const size_t n_blocks = tensor.total_blocks();
                const auto *blocks = tensor.as_block_32();
                if (!blocks)
                    return false;

                for (size_t b = 0; b < n_blocks; ++b)
                {
                    for (size_t i = 0; i < Q16_1Block::BLOCK_SIZE; ++i)
                    {
                        if (std::abs(blocks[b].qs[i]) > limit)
                        {
                            return false;
                        }
                    }
                }
                return true;
            }

            // Verify fixed scale is used (all blocks have same d)
            bool verify_fixed_scale(const Q16_1Tensor &tensor, float expected_d, float tolerance = 1e-6f)
            {
                const size_t n_blocks = tensor.total_blocks();
                const auto *blocks = tensor.as_block_32();
                if (!blocks)
                    return false;

                for (size_t b = 0; b < n_blocks; ++b)
                {
                    if (std::abs(blocks[b].d - expected_d) > tolerance)
                    {
                        return false;
                    }
                }
                return true;
            }

            // Simulate VNNI dot-product accumulation to check for overflow
            bool simulate_vnni_dot_product(const int16_t *a, const int16_t *b, size_t head_dim)
            {
                // VPDPWSSD accumulates 2 products per INT32 lane per instruction
                // Accumulate head_dim products into head_dim/16 INT32 lanes
                const size_t lanes = 16;
                const size_t products_per_lane = head_dim / lanes;

                std::vector<int64_t> accum(lanes, 0);

                for (size_t i = 0; i < head_dim; ++i)
                {
                    size_t lane = i % lanes;
                    int64_t product = static_cast<int64_t>(a[i]) * static_cast<int64_t>(b[i]);
                    accum[lane] += product;

                    // Check for overflow
                    if (accum[lane] > INT32_MAX || accum[lane] < INT32_MIN)
                    {
                        return false; // Overflow detected!
                    }
                }
                return true;
            }

            std::mt19937 gen_;
        };

        // ============================================================================
        // Basic Functionality Tests
        // ============================================================================

        TEST_F(Q16FixedScaleQuantizationTest, BasicQuantization)
        {
            // Create Q16 tensor [4, 64] - 4 rows, head_dim=64
            std::vector<size_t> shape = {4, 64};
            auto tensor = std::make_shared<Q16_1Tensor>(shape);

            // Generate data within safe range
            auto data = generate_random_fp32(4 * 64, 5.0f); // ±5.0 is within ±5.66 for head_dim=64

            // Quantize with fixed scale
            ASSERT_TRUE(tensor->copyFrom_fp32_fixed_scale(data.data(), DEFAULT_KV_CACHE_SCALE, HEAD_DIM_64));

            // Verify fixed scale: d = 8.0 / 32767 ≈ 0.000244
            const float expected_d = DEFAULT_KV_CACHE_SCALE / 32767.0f;
            EXPECT_TRUE(verify_fixed_scale(*tensor, expected_d));
        }

        TEST_F(Q16FixedScaleQuantizationTest, VerifyIntegersWithinSafeLimit_HeadDim64)
        {
            std::vector<size_t> shape = {8, 64};
            auto tensor = std::make_shared<Q16_1Tensor>(shape);

            // Generate data with some values near the scale limit
            auto data = generate_random_fp32(8 * 64, DEFAULT_KV_CACHE_SCALE * 0.9f); // ±7.2

            ASSERT_TRUE(tensor->copyFrom_fp32_fixed_scale(data.data(), DEFAULT_KV_CACHE_SCALE, HEAD_DIM_64));

            // All values should be within MAX_SAFE_INT16_64 = 23170
            EXPECT_TRUE(verify_all_within_limit(*tensor, MAX_SAFE_INT16_64));
        }

        TEST_F(Q16FixedScaleQuantizationTest, VerifyIntegersWithinSafeLimit_HeadDim128)
        {
            std::vector<size_t> shape = {8, 128};
            auto tensor = std::make_shared<Q16_1Tensor>(shape);

            // Generate data within safe range for head_dim=128 (±4.0)
            auto data = generate_random_fp32(8 * 128, 3.9f);

            ASSERT_TRUE(tensor->copyFrom_fp32_fixed_scale(data.data(), DEFAULT_KV_CACHE_SCALE, HEAD_DIM_128));

            // All values should be within MAX_SAFE_INT16_128 = 16383
            EXPECT_TRUE(verify_all_within_limit(*tensor, MAX_SAFE_INT16_128));
        }

        TEST_F(Q16FixedScaleQuantizationTest, VerifyIntegersWithinSafeLimit_HeadDim192)
        {
            std::vector<size_t> shape = {8, 192};
            auto tensor = std::make_shared<Q16_1Tensor>(shape);

            // Generate data within safe range for head_dim=192 (±3.27)
            auto data = generate_random_fp32(8 * 192, 3.0f);

            ASSERT_TRUE(tensor->copyFrom_fp32_fixed_scale(data.data(), DEFAULT_KV_CACHE_SCALE, HEAD_DIM_192));

            // All values should be within MAX_SAFE_INT16_192 = 13377
            EXPECT_TRUE(verify_all_within_limit(*tensor, MAX_SAFE_INT16_192));
        }

        // ============================================================================
        // Clipping Tests
        // ============================================================================

        TEST_F(Q16FixedScaleQuantizationTest, ClippingPreventsOverflow_HeadDim64)
        {
            std::vector<size_t> shape = {4, 64};
            auto tensor = std::make_shared<Q16_1Tensor>(shape);

            // Generate data with outliers beyond safe range (±5.66 → ±10.0)
            auto data = generate_with_outliers(4 * 64, 5.0f, 10.0f, 0.1f);

            ASSERT_TRUE(tensor->copyFrom_fp32_fixed_scale(data.data(), DEFAULT_KV_CACHE_SCALE, HEAD_DIM_64));

            // Verify all values are clipped to safe range
            int16_t max_abs = get_max_abs_int16(*tensor);
            EXPECT_LE(max_abs, MAX_SAFE_INT16_64);
        }

        TEST_F(Q16FixedScaleQuantizationTest, ClippingPreventsOverflow_HeadDim128)
        {
            std::vector<size_t> shape = {4, 128};
            auto tensor = std::make_shared<Q16_1Tensor>(shape);

            // Generate data with outliers beyond safe range (±4.0 → ±8.0)
            auto data = generate_with_outliers(4 * 128, 3.0f, 8.0f, 0.15f);

            ASSERT_TRUE(tensor->copyFrom_fp32_fixed_scale(data.data(), DEFAULT_KV_CACHE_SCALE, HEAD_DIM_128));

            // Verify all values are clipped to safe range
            int16_t max_abs = get_max_abs_int16(*tensor);
            EXPECT_LE(max_abs, MAX_SAFE_INT16_128);
        }

        TEST_F(Q16FixedScaleQuantizationTest, ExtremeOutliersAreClipped)
        {
            std::vector<size_t> shape = {2, 64};
            auto tensor = std::make_shared<Q16_1Tensor>(shape);

            // Create data with extreme values
            std::vector<float> data(2 * 64, 0.0f);
            data[0] = 100.0f;  // Way beyond ±8.0 scale
            data[1] = -50.0f;  // Way beyond ±8.0 scale
            data[32] = 20.0f;  // Beyond safe range
            data[64] = -30.0f; // Way beyond

            ASSERT_TRUE(tensor->copyFrom_fp32_fixed_scale(data.data(), DEFAULT_KV_CACHE_SCALE, HEAD_DIM_64));

            // All values must be within safe limit
            EXPECT_TRUE(verify_all_within_limit(*tensor, MAX_SAFE_INT16_64));
        }

        // ============================================================================
        // VNNI Overflow Simulation Tests
        // ============================================================================

        TEST_F(Q16FixedScaleQuantizationTest, VNNISimulation_NoOverflow_HeadDim64)
        {
            std::vector<size_t> shape = {1, 64};
            auto q_tensor = std::make_shared<Q16_1Tensor>(shape);
            auto k_tensor = std::make_shared<Q16_1Tensor>(shape);

            // Generate random Q and K values
            auto q_data = generate_random_fp32(64, DEFAULT_KV_CACHE_SCALE * 0.9f);
            auto k_data = generate_random_fp32(64, DEFAULT_KV_CACHE_SCALE * 0.9f);

            ASSERT_TRUE(q_tensor->copyFrom_fp32_fixed_scale(q_data.data(), DEFAULT_KV_CACHE_SCALE, HEAD_DIM_64));
            ASSERT_TRUE(k_tensor->copyFrom_fp32_fixed_scale(k_data.data(), DEFAULT_KV_CACHE_SCALE, HEAD_DIM_64));

            // Simulate VNNI dot-product
            const auto *q_blocks = q_tensor->as_block_32();
            const auto *k_blocks = k_tensor->as_block_32();
            ASSERT_NE(q_blocks, nullptr);
            ASSERT_NE(k_blocks, nullptr);

            EXPECT_TRUE(simulate_vnni_dot_product(q_blocks[0].qs, k_blocks[0].qs, HEAD_DIM_64));
        }

        TEST_F(Q16FixedScaleQuantizationTest, VNNISimulation_NoOverflow_HeadDim128)
        {
            std::vector<size_t> shape = {1, 128};
            auto q_tensor = std::make_shared<Q16_1Tensor>(shape);
            auto k_tensor = std::make_shared<Q16_1Tensor>(shape);

            // Generate random Q and K values
            auto q_data = generate_random_fp32(128, get_safe_fp32_range(HEAD_DIM_128, DEFAULT_KV_CACHE_SCALE) * 0.95f);
            auto k_data = generate_random_fp32(128, get_safe_fp32_range(HEAD_DIM_128, DEFAULT_KV_CACHE_SCALE) * 0.95f);

            ASSERT_TRUE(q_tensor->copyFrom_fp32_fixed_scale(q_data.data(), DEFAULT_KV_CACHE_SCALE, HEAD_DIM_128));
            ASSERT_TRUE(k_tensor->copyFrom_fp32_fixed_scale(k_data.data(), DEFAULT_KV_CACHE_SCALE, HEAD_DIM_128));

            // Extract INT16 values (first 4 blocks of 32 elements = 128)
            std::vector<int16_t> q_int16(128), k_int16(128);
            const auto *q_blocks = q_tensor->as_block_32();
            const auto *k_blocks = k_tensor->as_block_32();
            ASSERT_NE(q_blocks, nullptr);
            ASSERT_NE(k_blocks, nullptr);
            for (size_t b = 0; b < 4; ++b)
            {
                for (size_t i = 0; i < 32; ++i)
                {
                    q_int16[b * 32 + i] = q_blocks[b].qs[i];
                    k_int16[b * 32 + i] = k_blocks[b].qs[i];
                }
            }

            EXPECT_TRUE(simulate_vnni_dot_product(q_int16.data(), k_int16.data(), HEAD_DIM_128));
        }

        TEST_F(Q16FixedScaleQuantizationTest, VNNISimulation_WorstCase_AllMaxValues)
        {
            // Worst case: all values at MAX_SAFE_INT16
            std::vector<int16_t> a(64, MAX_SAFE_INT16_64);
            std::vector<int16_t> b(64, MAX_SAFE_INT16_64);

            // This MUST NOT overflow by design
            EXPECT_TRUE(simulate_vnni_dot_product(a.data(), b.data(), HEAD_DIM_64));
        }

        // ============================================================================
        // Row-wise Quantization Tests
        // ============================================================================

        TEST_F(Q16FixedScaleQuantizationTest, RowsQuantization_PartialRows)
        {
            std::vector<size_t> shape = {8, 64}; // 8 rows capacity
            auto tensor = std::make_shared<Q16_1Tensor>(shape);

            // Only fill first 3 rows
            auto data = generate_random_fp32(3 * 64, 5.0f);

            ASSERT_TRUE(tensor->copyFrom_fp32_rows_fixed_scale(data.data(), 3, DEFAULT_KV_CACHE_SCALE, HEAD_DIM_64));

            // First 3 rows should be quantized (6 blocks of 32)
            const auto *blocks = tensor->as_block_32();
            ASSERT_NE(blocks, nullptr);
            EXPECT_NE(blocks[0].d, 0.0f); // First block should have scale set
        }

        TEST_F(Q16FixedScaleQuantizationTest, RowsQuantization_AllRowsWithClipping)
        {
            std::vector<size_t> shape = {4, 128};
            auto tensor = std::make_shared<Q16_1Tensor>(shape);

            // Generate data with outliers
            auto data = generate_with_outliers(4 * 128, 3.0f, 8.0f, 0.1f);

            ASSERT_TRUE(tensor->copyFrom_fp32_rows_fixed_scale(data.data(), 4, DEFAULT_KV_CACHE_SCALE, HEAD_DIM_128));

            // All values should be within safe limit
            EXPECT_TRUE(verify_all_within_limit(*tensor, MAX_SAFE_INT16_128));
        }

        // ============================================================================
        // Comparison: Fixed vs Adaptive Quantization
        // ============================================================================

        TEST_F(Q16FixedScaleQuantizationTest, FixedVsAdaptive_DifferentScales)
        {
            std::vector<size_t> shape = {1, 32};
            auto fixed_tensor = std::make_shared<Q16_1Tensor>(shape);
            auto adaptive_tensor = std::make_shared<Q16_1Tensor>(shape);

            // Small values that don't hit the limit
            std::vector<float> data(32, 0.5f);

            // Fixed-scale quantization
            ASSERT_TRUE(fixed_tensor->copyFrom_fp32_fixed_scale(data.data(), DEFAULT_KV_CACHE_SCALE, HEAD_DIM_64));

            // Adaptive quantization
            ASSERT_TRUE(adaptive_tensor->copyFrom_fp32(data.data()));

            // Check scales
            const auto *fixed_blocks = fixed_tensor->as_block_32();
            const auto *adaptive_blocks = adaptive_tensor->as_block_32();
            ASSERT_NE(fixed_blocks, nullptr);
            ASSERT_NE(adaptive_blocks, nullptr);

            // Fixed scale: d = 8.0 / 32767 ≈ 0.000244
            float expected_fixed_d = DEFAULT_KV_CACHE_SCALE / 32767.0f;
            EXPECT_NEAR(fixed_blocks[0].d, expected_fixed_d, 1e-6f);

            // Adaptive scale: d = max_abs / 32767 = 0.5 / 32767 ≈ 0.0000153
            float expected_adaptive_d = 0.5f / 32767.0f;
            EXPECT_NEAR(adaptive_blocks[0].d, expected_adaptive_d, 1e-7f);

            // Adaptive gives larger INT16 values (better precision)
            // Fixed gives smaller INT16 values (VNNI-safe, lower precision)
            EXPECT_GT(std::abs(adaptive_blocks[0].qs[0]), std::abs(fixed_blocks[0].qs[0]));
        }

        // ============================================================================
        // Error Handling Tests
        // ============================================================================

        TEST_F(Q16FixedScaleQuantizationTest, ErrorHandling_NullData)
        {
            std::vector<size_t> shape = {4, 64};
            auto tensor = std::make_shared<Q16_1Tensor>(shape);

            EXPECT_FALSE(tensor->copyFrom_fp32_fixed_scale(nullptr, DEFAULT_KV_CACHE_SCALE, HEAD_DIM_64));
        }

        TEST_F(Q16FixedScaleQuantizationTest, ErrorHandling_InvalidScale)
        {
            std::vector<size_t> shape = {4, 64};
            auto tensor = std::make_shared<Q16_1Tensor>(shape);

            std::vector<float> data(4 * 64, 0.0f);

            EXPECT_FALSE(tensor->copyFrom_fp32_fixed_scale(data.data(), 0.0f, HEAD_DIM_64));
            EXPECT_FALSE(tensor->copyFrom_fp32_fixed_scale(data.data(), -1.0f, HEAD_DIM_64));
        }

        TEST_F(Q16FixedScaleQuantizationTest, ErrorHandling_InvalidHeadDim)
        {
            std::vector<size_t> shape = {4, 64};
            auto tensor = std::make_shared<Q16_1Tensor>(shape);

            std::vector<float> data(4 * 64, 0.0f);

            EXPECT_FALSE(tensor->copyFrom_fp32_fixed_scale(data.data(), DEFAULT_KV_CACHE_SCALE, 0));
            EXPECT_FALSE(tensor->copyFrom_fp32_fixed_scale(data.data(), DEFAULT_KV_CACHE_SCALE, -1));
        }

        TEST_F(Q16FixedScaleQuantizationTest, ErrorHandling_RowsExceedCapacity)
        {
            std::vector<size_t> shape = {4, 64};
            auto tensor = std::make_shared<Q16_1Tensor>(shape);

            std::vector<float> data(8 * 64, 0.0f);

            // Try to copy 8 rows when only 4 rows capacity
            EXPECT_FALSE(tensor->copyFrom_fp32_rows_fixed_scale(data.data(), 8, DEFAULT_KV_CACHE_SCALE, HEAD_DIM_64));
        }

        // ============================================================================
        // Precision Tests
        // ============================================================================

        TEST_F(Q16FixedScaleQuantizationTest, Precision_RoundTrip)
        {
            std::vector<size_t> shape = {1, 32};
            auto tensor = std::make_shared<Q16_1Tensor>(shape);

            // Generate values within safe range
            std::vector<float> original(32);
            for (int i = 0; i < 32; ++i)
            {
                original[i] = (i - 16) * 0.3f; // -4.8 to +4.5, within ±8.0 scale
            }

            ASSERT_TRUE(tensor->copyFrom_fp32_fixed_scale(original.data(), DEFAULT_KV_CACHE_SCALE, HEAD_DIM_64));

            // Dequantize
            const float *dequantized = tensor->data();

            // Check round-trip error
            float max_error = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                float error = std::abs(original[i] - dequantized[i]);
                max_error = std::max(max_error, error);
            }

            // With scale=8.0, quantization step = 8.0/32767 ≈ 0.000244
            // Max round-trip error should be about half the quantization step
            EXPECT_LT(max_error, 0.001f);
        }

        TEST_F(Q16FixedScaleQuantizationTest, Precision_ZeroValues)
        {
            std::vector<size_t> shape = {1, 32};
            auto tensor = std::make_shared<Q16_1Tensor>(shape);

            std::vector<float> zeros(32, 0.0f);

            ASSERT_TRUE(tensor->copyFrom_fp32_fixed_scale(zeros.data(), DEFAULT_KV_CACHE_SCALE, HEAD_DIM_64));

            // All INT16 values should be 0
            const auto *blocks = tensor->as_block_32();
            ASSERT_NE(blocks, nullptr);
            for (int i = 0; i < 32; ++i)
            {
                EXPECT_EQ(blocks[0].qs[i], 0);
            }
        }

        // ============================================================================
        // VNNISafetyConstants.h Tests (verify header is correctly integrated)
        // ============================================================================

        TEST_F(Q16FixedScaleQuantizationTest, SafetyConstants_ConsistentWithInternalCalculation)
        {
            // Verify that VNNISafetyConstants.h values match Q16_1Tensor internal calculation
            EXPECT_EQ(get_max_safe_int16(HEAD_DIM_64), MAX_SAFE_INT16_64);
            EXPECT_EQ(get_max_safe_int16(HEAD_DIM_128), MAX_SAFE_INT16_128);
            EXPECT_EQ(get_max_safe_int16(HEAD_DIM_192), MAX_SAFE_INT16_192);
        }

        TEST_F(Q16FixedScaleQuantizationTest, SafetyConstants_ComputeDynamicMatchesPrecomputed)
        {
            // Verify dynamic calculation matches pre-computed values
            EXPECT_EQ(compute_max_safe_int16(64), MAX_SAFE_INT16_64);
            EXPECT_EQ(compute_max_safe_int16(128), MAX_SAFE_INT16_128);
            EXPECT_EQ(compute_max_safe_int16(192), MAX_SAFE_INT16_192);
        }

        TEST_F(Q16FixedScaleQuantizationTest, SafetyConstants_SafeFP32Range)
        {
            // Verify safe FP32 range calculation
            float range_64 = get_safe_fp32_range(HEAD_DIM_64, DEFAULT_KV_CACHE_SCALE);
            float range_128 = get_safe_fp32_range(HEAD_DIM_128, DEFAULT_KV_CACHE_SCALE);

            EXPECT_NEAR(range_64, SAFE_FP32_RANGE_64, 0.01f);
            EXPECT_NEAR(range_128, SAFE_FP32_RANGE_128, 0.01f);

            // Larger head_dim should have smaller safe range
            EXPECT_GT(range_64, range_128);
        }

    } // namespace test
} // namespace llaminar2
