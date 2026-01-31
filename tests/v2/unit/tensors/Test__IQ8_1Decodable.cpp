/**
 * @file Test__IQ8_1Decodable.cpp
 * @brief Unit tests for Q8_1 quantization SIMD helpers and IQ8_1Decodable interface
 * @author David Sanftenberg
 *
 * Tests:
 * - Numerical accuracy of scalar, AVX2, and AVX-512 quantization
 * - Mutual agreement between SIMD implementations
 * - Pre-computed sum correctness
 * - Performance speedups (AVX2 vs scalar, AVX-512 vs scalar)
 * - Edge cases (zeros, extremes, denormals)
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include <random>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace llaminar2;
using namespace llaminar2::simd;

namespace
{

    /**
     * @brief Test fixture for Q8_1 quantization tests
     */
    class Test__IQ8_1Decodable : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Seed RNG for reproducibility
            rng_.seed(42);
        }

        /**
         * @brief Generate random FP32 test data
         */
        std::vector<float> generateRandomData(size_t count, float min_val = -10.0f, float max_val = 10.0f)
        {
            std::vector<float> data(count);
            std::uniform_real_distribution<float> dist(min_val, max_val);
            for (size_t i = 0; i < count; ++i)
            {
                data[i] = dist(rng_);
            }
            return data;
        }

        /**
         * @brief Generate edge case test data
         */
        std::vector<float> generateEdgeCaseData()
        {
            return {
                0.0f, -0.0f,     // Zeros
                1.0f, -1.0f,     // Unity
                127.0f, -127.0f, // Max int8 range
                0.001f, -0.001f, // Small values
                1e-6f, -1e-6f,   // Very small (near denormal)
                100.0f, -100.0f, // Large values
                std::numeric_limits<float>::epsilon(),
                -std::numeric_limits<float>::epsilon(),
                0.5f, -0.5f,         // Half values
                3.14159f, -3.14159f, // Irrational
                2.71828f, -2.71828f, // e
                // Fill remaining with varied values
                0.1f, 0.2f, 0.3f, 0.4f, 0.6f, 0.7f, 0.8f, 0.9f,
                -0.1f, -0.2f, -0.3f, -0.4f, -0.6f, -0.7f};
        }

        /**
         * @brief Compute relative error between two float values
         */
        float relativeError(float a, float b)
        {
            float abs_diff = std::abs(a - b);
            float abs_max = std::max(std::abs(a), std::abs(b));
            if (abs_max < 1e-10f)
            {
                return abs_diff; // Absolute error for near-zero values
            }
            return abs_diff / abs_max;
        }

        /**
         * @brief Compute max relative error between two vectors
         */
        float maxRelativeError(const std::vector<float> &a, const std::vector<float> &b)
        {
            EXPECT_EQ(a.size(), b.size());
            float max_err = 0.0f;
            for (size_t i = 0; i < a.size(); ++i)
            {
                float err = relativeError(a[i], b[i]);
                max_err = std::max(max_err, err);
            }
            return max_err;
        }

        /**
         * @brief Manually compute expected Q8_1 quantization (ground truth)
         */
        void referenceQuantizeQ8_1(
            const float *src,
            int8_t *dst_qs,
            uint16_t *dst_scale_fp16,
            uint16_t *dst_sum_fp16)
        {

            // Find max absolute value
            float max_abs = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[i]));
            }

            // Compute scale
            float d = (max_abs > 0.0f) ? (max_abs / 127.0f) : 0.0f;
            float inv_d = (d > 0.0f) ? (1.0f / d) : 0.0f;

            // Quantize and compute sum
            float sum = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                float val = src[i];

                // Quantize with rounding
                float q_float = val * inv_d;
                q_float = std::round(q_float);
                q_float = std::max(-127.0f, std::min(127.0f, q_float));
                dst_qs[i] = static_cast<int8_t>(q_float);

                // Sum quantized values (CRITICAL!)
                sum += static_cast<float>(dst_qs[i]);
            }

            // Store scale and pre-computed sum
            *dst_scale_fp16 = fp32_to_fp16(d);
            *dst_sum_fp16 = fp32_to_fp16(d * sum); // s = d × Σ(qs[i])
        }

        /**
         * @brief Dequantize Q8_1 block back to FP32
         */
        std::vector<float> dequantizeQ8_1(const Q8_1Block &block)
        {
            std::vector<float> result(32);
            float scale = fp16_to_fp32(block.d);
            for (int i = 0; i < 32; ++i)
            {
                result[i] = scale * static_cast<float>(block.qs[i]);
            }
            return result;
        }

        std::mt19937 rng_;
    };

    // ============================================================================
    // ACCURACY TESTS: Scalar Implementation
    // ============================================================================

    TEST_F(Test__IQ8_1Decodable, ScalarQuantization_RandomData_Accuracy)
    {
        auto input = generateRandomData(32);

        Q8_1Block block_scalar;
        quantize_fp32_to_q8_1_scalar(
            input.data(), 32,
            block_scalar.qs, &block_scalar.d, &block_scalar.s);

        // Reference implementation
        Q8_1Block block_ref;
        referenceQuantizeQ8_1(
            input.data(),
            block_ref.qs, &block_ref.d, &block_ref.s);

        // Compare quantized values (should be identical for scalar)
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_EQ(block_scalar.qs[i], block_ref.qs[i])
                << "Mismatch at index " << i;
        }

        // Compare scale (FP16, allow small tolerance)
        float scale_scalar = fp16_to_fp32(block_scalar.d);
        float scale_ref = fp16_to_fp32(block_ref.d);
        EXPECT_NEAR(scale_scalar, scale_ref, 1e-4f);

        // Compare pre-computed sum (FP16, allow small tolerance)
        float sum_scalar = fp16_to_fp32(block_scalar.s);
        float sum_ref = fp16_to_fp32(block_ref.s);
        EXPECT_NEAR(sum_scalar, sum_ref, 1e-3f);
    }

    TEST_F(Test__IQ8_1Decodable, ScalarQuantization_EdgeCases)
    {
        auto input = generateEdgeCaseData();
        input.resize(32, 0.0f); // Pad to 32 elements

        Q8_1Block block;
        quantize_fp32_to_q8_1_scalar(
            input.data(), 32,
            block.qs, &block.d, &block.s);

        // Dequantize and check reconstruction
        auto reconstructed = dequantizeQ8_1(block);

        // Compute expected sum from QUANTIZED values
        float d = fp16_to_fp32(block.d);
        float expected_sum = 0.0f;
        for (int i = 0; i < 32; ++i)
        {
            expected_sum += static_cast<float>(block.qs[i]);
        }

        // Check pre-computed sum matches
        float sum_stored = fp16_to_fp32(block.s);
        float sum_computed = d * expected_sum;

        // FP16 precision + accumulation error
        EXPECT_NEAR(sum_stored, sum_computed, std::abs(sum_computed) * 0.01f + 1e-4f);
    }

    TEST_F(Test__IQ8_1Decodable, ScalarQuantization_AllZeros)
    {
        std::vector<float> input(32, 0.0f);

        Q8_1Block block;
        quantize_fp32_to_q8_1_scalar(
            input.data(), 32,
            block.qs, &block.d, &block.s);

        // All quantized values should be zero
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_EQ(block.qs[i], 0);
        }

        // Scale and sum should be zero (or very close)
        float scale = fp16_to_fp32(block.d);
        float sum = fp16_to_fp32(block.s);
        EXPECT_NEAR(scale, 0.0f, 1e-6f);
        EXPECT_NEAR(sum, 0.0f, 1e-6f);
    }

    TEST_F(Test__IQ8_1Decodable, ScalarQuantization_ConstantValue)
    {
        std::vector<float> input(32, 5.0f);

        Q8_1Block block;
        quantize_fp32_to_q8_1_scalar(
            input.data(), 32,
            block.qs, &block.d, &block.s);

        // All quantized values should be max (127 or close)
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_GE(block.qs[i], 120) << "Index " << i;
        }

        // Pre-computed sum should be d * sum(qs[i])
        float d = fp16_to_fp32(block.d);
        float sum = fp16_to_fp32(block.s);

        // Compute expected sum from quantized values
        float expected_sum = 0.0f;
        for (int i = 0; i < 32; ++i)
        {
            expected_sum += static_cast<float>(block.qs[i]);
        }
        expected_sum *= d;

        EXPECT_NEAR(sum, expected_sum, std::abs(expected_sum) * 0.01f);
    }

    // ============================================================================
    // SIMD AGREEMENT TESTS: AVX2 vs Scalar
    // ============================================================================

#if defined(__AVX2__)
    TEST_F(Test__IQ8_1Decodable, AVX2vsScalar_RandomData_Agreement)
    {
        auto input = generateRandomData(32);

        Q8_1Block block_scalar, block_avx2;

        quantize_fp32_to_q8_1_scalar(
            input.data(), 32,
            block_scalar.qs, &block_scalar.d, &block_scalar.s);

        quantize_fp32_to_q8_1_avx2(
            input.data(), 32,
            block_avx2.qs, &block_avx2.d, &block_avx2.s);

        // Quantized values should match exactly
        for (int i = 0; i < 32; ++i)
        {
            // Allow ±1 difference due to rounding variations
            EXPECT_NEAR(static_cast<float>(block_scalar.qs[i]),
                        static_cast<float>(block_avx2.qs[i]), 1.0f)
                << "Mismatch at index " << i;
        }

        // Scale should match closely
        float scale_scalar = fp16_to_fp32(block_scalar.d);
        float scale_avx2 = fp16_to_fp32(block_avx2.d);
        EXPECT_NEAR(scale_scalar, scale_avx2, scale_scalar * 0.001f + 1e-6f);

        // Pre-computed sum should match closely
        float sum_scalar = fp16_to_fp32(block_scalar.s);
        float sum_avx2 = fp16_to_fp32(block_avx2.s);
        EXPECT_NEAR(sum_scalar, sum_avx2, std::abs(sum_scalar) * 0.01f + 1e-4f);
    }

    TEST_F(Test__IQ8_1Decodable, AVX2vsScalar_MultipleBlocks_Agreement)
    {
        constexpr int NUM_BLOCKS = 100;
        auto input = generateRandomData(32 * NUM_BLOCKS);

        int mismatches = 0;
        int total_elements = 0;

        for (int block = 0; block < NUM_BLOCKS; ++block)
        {
            const float *src = input.data() + block * 32;

            Q8_1Block block_scalar, block_avx2;

            quantize_fp32_to_q8_1_scalar(src, 32, block_scalar.qs, &block_scalar.d, &block_scalar.s);
            quantize_fp32_to_q8_1_avx2(src, 32, block_avx2.qs, &block_avx2.d, &block_avx2.s);

            for (int i = 0; i < 32; ++i)
            {
                if (std::abs(block_scalar.qs[i] - block_avx2.qs[i]) > 1)
                {
                    mismatches++;
                }
                total_elements++;
            }
        }

        // Allow up to 1% mismatch due to different rounding behavior
        float mismatch_rate = static_cast<float>(mismatches) / total_elements;
        EXPECT_LT(mismatch_rate, 0.01f)
            << "Too many mismatches: " << mismatches << "/" << total_elements;
    }
#endif // __AVX2__

    // ============================================================================
    // SIMD AGREEMENT TESTS: AVX-512 vs Scalar
    // ============================================================================

#if defined(__AVX512F__)
    TEST_F(Test__IQ8_1Decodable, AVX512vsScalar_RandomData_Agreement)
    {
        auto input = generateRandomData(32);

        Q8_1Block block_scalar, block_avx512;

        quantize_fp32_to_q8_1_scalar(
            input.data(), 32,
            block_scalar.qs, &block_scalar.d, &block_scalar.s);

        quantize_fp32_to_q8_1_avx512(
            input.data(), 32,
            block_avx512.qs, &block_avx512.d, &block_avx512.s);

        // Quantized values should match exactly or within ±1
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_NEAR(static_cast<float>(block_scalar.qs[i]),
                        static_cast<float>(block_avx512.qs[i]), 1.0f)
                << "Mismatch at index " << i;
        }

        // Scale should match closely
        float scale_scalar = fp16_to_fp32(block_scalar.d);
        float scale_avx512 = fp16_to_fp32(block_avx512.d);
        EXPECT_NEAR(scale_scalar, scale_avx512, scale_scalar * 0.001f + 1e-6f);

        // Pre-computed sum should match closely
        float sum_scalar = fp16_to_fp32(block_scalar.s);
        float sum_avx512 = fp16_to_fp32(block_avx512.s);
        EXPECT_NEAR(sum_scalar, sum_avx512, std::abs(sum_scalar) * 0.01f + 1e-4f);
    }

    TEST_F(Test__IQ8_1Decodable, AVX512vsScalar_MultipleBlocks_Agreement)
    {
        constexpr int NUM_BLOCKS = 100;
        auto input = generateRandomData(32 * NUM_BLOCKS);

        int mismatches = 0;
        int total_elements = 0;

        for (int block = 0; block < NUM_BLOCKS; ++block)
        {
            const float *src = input.data() + block * 32;

            Q8_1Block block_scalar, block_avx512;

            quantize_fp32_to_q8_1_scalar(src, 32, block_scalar.qs, &block_scalar.d, &block_scalar.s);
            quantize_fp32_to_q8_1_avx512(src, 32, block_avx512.qs, &block_avx512.d, &block_avx512.s);

            for (int i = 0; i < 32; ++i)
            {
                if (std::abs(block_scalar.qs[i] - block_avx512.qs[i]) > 1)
                {
                    mismatches++;
                }
                total_elements++;
            }
        }

        // Allow up to 1% mismatch
        float mismatch_rate = static_cast<float>(mismatches) / total_elements;
        EXPECT_LT(mismatch_rate, 0.01f)
            << "Too many mismatches: " << mismatches << "/" << total_elements;
    }
#endif // __AVX512F__

    // ============================================================================
    // PERFORMANCE TESTS: SIMD Speedups
    // ============================================================================

    TEST_F(Test__IQ8_1Decodable, Performance_ScalarBaseline)
    {
        constexpr int NUM_BLOCKS = 10000;
        auto input = generateRandomData(32 * NUM_BLOCKS);
        std::vector<Q8_1Block> blocks(NUM_BLOCKS);

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < NUM_BLOCKS; ++i)
        {
            quantize_fp32_to_q8_1_scalar(
                input.data() + i * 32, 32,
                blocks[i].qs, &blocks[i].d, &blocks[i].s);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        double throughput = (NUM_BLOCKS * 32.0) / (duration / 1e6); // elements/sec
        double throughput_gb = (throughput * sizeof(float)) / 1e9;  // GB/s

        std::cout << "\n[Scalar Performance]\n";
        std::cout << "  Blocks:     " << NUM_BLOCKS << "\n";
        std::cout << "  Time:       " << duration << " μs\n";
        std::cout << "  Throughput: " << (throughput / 1e6) << " M elements/sec\n";
        std::cout << "  Bandwidth:  " << throughput_gb << " GB/s\n";

        // Sanity check: should process at least 80 MB/s even on slow systems
        EXPECT_GT(throughput_gb, 0.08);
    }

#if defined(__AVX2__)
    TEST_F(Test__IQ8_1Decodable, Performance_AVX2Speedup)
    {
        constexpr int NUM_BLOCKS = 10000;
        auto input = generateRandomData(32 * NUM_BLOCKS);
        std::vector<Q8_1Block> blocks_scalar(NUM_BLOCKS);
        std::vector<Q8_1Block> blocks_avx2(NUM_BLOCKS);

        // Warmup
        for (int i = 0; i < 100; ++i)
        {
            quantize_fp32_to_q8_1_avx2(input.data(), 32, blocks_avx2[0].qs, &blocks_avx2[0].d, &blocks_avx2[0].s);
        }

        // Scalar timing
        auto start_scalar = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_BLOCKS; ++i)
        {
            quantize_fp32_to_q8_1_scalar(input.data() + i * 32, 32, blocks_scalar[i].qs, &blocks_scalar[i].d, &blocks_scalar[i].s);
        }
        auto end_scalar = std::chrono::high_resolution_clock::now();

        // AVX2 timing
        auto start_avx2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_BLOCKS; ++i)
        {
            quantize_fp32_to_q8_1_avx2(input.data() + i * 32, 32, blocks_avx2[i].qs, &blocks_avx2[i].d, &blocks_avx2[i].s);
        }
        auto end_avx2 = std::chrono::high_resolution_clock::now();

        auto duration_scalar = std::chrono::duration_cast<std::chrono::microseconds>(end_scalar - start_scalar).count();
        auto duration_avx2 = std::chrono::duration_cast<std::chrono::microseconds>(end_avx2 - start_avx2).count();

        double speedup = static_cast<double>(duration_scalar) / duration_avx2;

        std::cout << "\n[AVX2 vs Scalar Performance]\n";
        std::cout << "  Scalar time: " << duration_scalar << " μs\n";
        std::cout << "  AVX2 time:   " << duration_avx2 << " μs\n";
        std::cout << "  Speedup:     " << speedup << "x\n";

        // AVX2 should be at least 1.5x faster (conservative, often 2-3x)
        EXPECT_GT(speedup, 1.5)
            << "AVX2 should provide significant speedup over scalar";
    }
#endif // __AVX2__

#if defined(__AVX512F__)
    TEST_F(Test__IQ8_1Decodable, Performance_AVX512Speedup)
    {
        constexpr int NUM_BLOCKS = 10000;
        auto input = generateRandomData(32 * NUM_BLOCKS);
        std::vector<Q8_1Block> blocks_scalar(NUM_BLOCKS);
        std::vector<Q8_1Block> blocks_avx512(NUM_BLOCKS);

        // Warmup
        for (int i = 0; i < 100; ++i)
        {
            quantize_fp32_to_q8_1_avx512(input.data(), 32, blocks_avx512[0].qs, &blocks_avx512[0].d, &blocks_avx512[0].s);
        }

        // Scalar timing
        auto start_scalar = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_BLOCKS; ++i)
        {
            quantize_fp32_to_q8_1_scalar(input.data() + i * 32, 32, blocks_scalar[i].qs, &blocks_scalar[i].d, &blocks_scalar[i].s);
        }
        auto end_scalar = std::chrono::high_resolution_clock::now();

        // AVX-512 timing
        auto start_avx512 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_BLOCKS; ++i)
        {
            quantize_fp32_to_q8_1_avx512(input.data() + i * 32, 32, blocks_avx512[i].qs, &blocks_avx512[i].d, &blocks_avx512[i].s);
        }
        auto end_avx512 = std::chrono::high_resolution_clock::now();

        auto duration_scalar = std::chrono::duration_cast<std::chrono::microseconds>(end_scalar - start_scalar).count();
        auto duration_avx512 = std::chrono::duration_cast<std::chrono::microseconds>(end_avx512 - start_avx512).count();

        double speedup = static_cast<double>(duration_scalar) / duration_avx512;

        std::cout << "\n[AVX-512 vs Scalar Performance]\n";
        std::cout << "  Scalar time:  " << duration_scalar << " μs\n";
        std::cout << "  AVX-512 time: " << duration_avx512 << " μs\n";
        std::cout << "  Speedup:      " << speedup << "x\n";

        // AVX-512 should be at least 2x faster (often 3-4x)
        EXPECT_GT(speedup, 2.0)
            << "AVX-512 should provide significant speedup over scalar";
    }
#endif // __AVX512F__

    // ============================================================================
    // TENSOR INTERFACE TESTS: FP32/FP16/BF16 → Q8_1
    // ============================================================================

    TEST_F(Test__IQ8_1Decodable, FP32Tensor_DecodeToQ8_1)
    {
        std::vector<size_t> shape = {4, 64}; // 4 rows, 64 columns (2 blocks per row)
        auto input_data = generateRandomData(4 * 64);

        // Create FP32 tensor
        auto fp32_tensor = std::make_shared<FP32Tensor>(shape);
        // Get mutable pointer and copy data
        float *mutable_data = const_cast<float *>(fp32_tensor->data());
        std::memcpy(mutable_data, input_data.data(), input_data.size() * sizeof(float));

        // Decode to Q8_1 (returns pointer, no copy!)
        const Q8_1Block *block_ptr = fp32_tensor->decode_to_q8_1(0, 0); // Row 0, block 0
        const Q8_1Block &block = *block_ptr;

        // Verify quantization
        float scale = fp16_to_fp32(block.d);
        EXPECT_GT(scale, 0.0f);

        // Dequantize and check reconstruction error
        auto reconstructed = dequantizeQ8_1(block);
        const float *original = input_data.data(); // First 32 elements

        float max_err = 0.0f;
        for (int i = 0; i < 32; ++i)
        {
            float err = std::abs(original[i] - reconstructed[i]);
            max_err = std::max(max_err, err);
        }

        // Q8_1 quantization should preserve within ~1% relative error
        float max_val = *std::max_element(original, original + 32);
        EXPECT_LT(max_err, std::abs(max_val) * 0.02f);
    }

    TEST_F(Test__IQ8_1Decodable, FP32Tensor_MultipleBlocks)
    {
        std::vector<size_t> shape = {2, 128}; // 2 rows, 128 columns (4 blocks per row)
        auto input_data = generateRandomData(2 * 128);

        auto fp32_tensor = std::make_shared<FP32Tensor>(shape);
        float *mutable_data = const_cast<float *>(fp32_tensor->data());
        std::memcpy(mutable_data, input_data.data(), input_data.size() * sizeof(float));

        // Test all blocks
        for (size_t row = 0; row < 2; ++row)
        {
            for (size_t block_idx = 0; block_idx < 4; ++block_idx)
            {
                const Q8_1Block *block_ptr = fp32_tensor->decode_to_q8_1(row, block_idx);
                const Q8_1Block &block = *block_ptr;

                // Verify scale is reasonable
                float scale = fp16_to_fp32(block.d);
                EXPECT_GE(scale, 0.0f);
                EXPECT_LT(scale, 1000.0f); // Sanity check
            }
        }
    }

    TEST_F(Test__IQ8_1Decodable, Q8_1Tensor_IdentityConversion)
    {
        std::vector<size_t> shape = {2, 64};
        auto input_data = generateRandomData(2 * 64);

        // Quantize to Q8_1
        auto q8_1_tensor = Q8_1Tensor::quantize_from_fp32(input_data.data(), shape);

        // Decode Q8_1 → Q8_1 (should be identity - returns direct pointer!)
        const Q8_1Block *block_decoded_ptr = q8_1_tensor->decode_to_q8_1(0, 0);
        const Q8_1Block &block_decoded = *block_decoded_ptr;

        // Get a reference block by manually quantizing the same data
        Q8_1Block block_expected;
        quantize_fp32_to_q8_1_scalar(
            input_data.data(), 32,
            block_expected.qs, &block_expected.d, &block_expected.s);

        // Should match the expected quantization
        EXPECT_EQ(block_expected.d, block_decoded.d);
        EXPECT_EQ(block_expected.s, block_decoded.s);
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_EQ(block_expected.qs[i], block_decoded.qs[i]);
        }
    }

    // ============================================================================
    // PRE-COMPUTED SUM VERIFICATION
    // ============================================================================

    TEST_F(Test__IQ8_1Decodable, PreComputedSum_MatchesManualSum)
    {
        auto input = generateRandomData(32);

        Q8_1Block block;
        quantize_fp32_to_q8_1_scalar(input.data(), 32, block.qs, &block.d, &block.s);

        // Manually compute sum from dequantized values
        float scale = fp16_to_fp32(block.d);
        float manual_sum = 0.0f;
        for (int i = 0; i < 32; ++i)
        {
            manual_sum += scale * static_cast<float>(block.qs[i]);
        }

        // Compare with pre-computed sum
        float precomputed_sum = fp16_to_fp32(block.s);

        // Should match within FP16 precision + accumulation error
        EXPECT_NEAR(manual_sum, precomputed_sum, std::abs(manual_sum) * 0.01f + 1e-4f)
            << "Pre-computed sum should match manual sum";
    }

    TEST_F(Test__IQ8_1Decodable, PreComputedSum_SavesComputationTime)
    {
        constexpr int NUM_BLOCKS = 1000;
        auto input = generateRandomData(32 * NUM_BLOCKS);
        std::vector<Q8_1Block> blocks(NUM_BLOCKS);

        // Quantize with pre-computed sums
        for (int i = 0; i < NUM_BLOCKS; ++i)
        {
            quantize_fp32_to_q8_1_scalar(
                input.data() + i * 32, 32,
                blocks[i].qs, &blocks[i].d, &blocks[i].s);
        }

        // Measure time to use pre-computed sums
        auto start_precomp = std::chrono::high_resolution_clock::now();
        float sum_precomp = 0.0f;
        for (int i = 0; i < NUM_BLOCKS; ++i)
        {
            sum_precomp += fp16_to_fp32(blocks[i].s);
        }
        auto end_precomp = std::chrono::high_resolution_clock::now();

        // Measure time to compute sums manually
        auto start_manual = std::chrono::high_resolution_clock::now();
        float sum_manual = 0.0f;
        for (int i = 0; i < NUM_BLOCKS; ++i)
        {
            float scale = fp16_to_fp32(blocks[i].d);
            for (int j = 0; j < 32; ++j)
            {
                sum_manual += scale * static_cast<float>(blocks[i].qs[j]);
            }
        }
        auto end_manual = std::chrono::high_resolution_clock::now();

        auto duration_precomp = std::chrono::duration_cast<std::chrono::nanoseconds>(end_precomp - start_precomp).count();
        auto duration_manual = std::chrono::duration_cast<std::chrono::nanoseconds>(end_manual - start_manual).count();

        double speedup = static_cast<double>(duration_manual) / duration_precomp;

        std::cout << "\n[Pre-computed Sum Benefit]\n";
        std::cout << "  Pre-computed: " << duration_precomp << " ns\n";
        std::cout << "  Manual:       " << duration_manual << " ns\n";
        std::cout << "  Speedup:      " << speedup << "x\n";

        // Pre-computed sum should be much faster (at least 10x)
        EXPECT_GT(speedup, 10.0)
            << "Pre-computed sum should be significantly faster than manual computation";

        // Sums should match
        EXPECT_NEAR(sum_precomp, sum_manual, std::abs(sum_manual) * 0.01f);
    }

} // anonymous namespace
