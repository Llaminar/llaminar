/**
 * @file Test__PrecisionAwareAllreduce.cpp
 * @brief Unit tests for precision-aware MPI allreduce operations
 *
 * Tests the unified allreduce implementations for FP32, FP16, BF16, and Q8_1
 * precision types. These tests verify correctness of the N-way reduction
 * algorithms without actual MPI communication.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

#include "tensors/SIMDHelpers.h"
#include "tensors/BlockStructures.h"
#include "utils/MPIContext.h"

namespace llaminar2
{
    namespace testing
    {

        /**
         * @brief Test fixture for precision-aware allreduce tests
         */
        class Test__PrecisionAwareAllreduce : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                // Initialize random seed for reproducibility
                rng_.seed(42);
            }

            std::mt19937 rng_;

            // Helper to generate random FP32 values
            std::vector<float> generate_random_fp32(size_t count, float min_val = -1.0f, float max_val = 1.0f)
            {
                std::uniform_real_distribution<float> dist(min_val, max_val);
                std::vector<float> result(count);
                for (auto &v : result)
                {
                    v = dist(rng_);
                }
                return result;
            }

            // Helper to convert FP32 to FP16
            std::vector<uint16_t> fp32_to_fp16_vec(const std::vector<float> &fp32)
            {
                std::vector<uint16_t> fp16(fp32.size());
                for (size_t i = 0; i < fp32.size(); ++i)
                {
                    fp16[i] = simd::fp32_to_fp16(fp32[i]);
                }
                return fp16;
            }

            // Helper to convert FP32 to BF16
            std::vector<uint16_t> fp32_to_bf16_vec(const std::vector<float> &fp32)
            {
                std::vector<uint16_t> bf16(fp32.size());
                for (size_t i = 0; i < fp32.size(); ++i)
                {
                    bf16[i] = simd::fp32_to_bf16(fp32[i]);
                }
                return bf16;
            }

            // Helper to convert FP16 to FP32
            std::vector<float> fp16_to_fp32_vec(const std::vector<uint16_t> &fp16)
            {
                std::vector<float> fp32(fp16.size());
                for (size_t i = 0; i < fp16.size(); ++i)
                {
                    fp32[i] = simd::fp16_to_fp32(fp16[i]);
                }
                return fp32;
            }

            // Helper to convert BF16 to FP32
            std::vector<float> bf16_to_fp32_vec(const std::vector<uint16_t> &bf16)
            {
                std::vector<float> fp32(bf16.size());
                for (size_t i = 0; i < bf16.size(); ++i)
                {
                    fp32[i] = simd::bf16_to_fp32(bf16[i]);
                }
                return fp32;
            }

            // Helper to quantize FP32 to Q8_1 blocks
            std::vector<Q8_1Block> fp32_to_q8_1_blocks(const std::vector<float> &fp32)
            {
                const size_t n_blocks = (fp32.size() + 31) / 32;
                std::vector<Q8_1Block> blocks(n_blocks);

                // Pad to 32-element boundary
                std::vector<float> padded = fp32;
                padded.resize(n_blocks * 32, 0.0f);

                simd::quantize_fp32_to_q8_1_blocks(padded.data(), blocks.data(), padded.size());
                return blocks;
            }

            // Helper to dequantize Q8_1 blocks to FP32
            std::vector<float> q8_1_blocks_to_fp32(const std::vector<Q8_1Block> &blocks, size_t count)
            {
                std::vector<float> fp32(blocks.size() * 32);
                simd::dequantize_q8_1_to_fp32(blocks.data(), fp32.data(), blocks.size() * 32);
                fp32.resize(count);
                return fp32;
            }

            // Helper to quantize FP32 to Q16_1 blocks
            std::vector<Q16_1Block> fp32_to_q16_1_blocks(const std::vector<float> &fp32)
            {
                const size_t n_blocks = (fp32.size() + 31) / 32;
                std::vector<Q16_1Block> blocks(n_blocks);

                // Pad to 32-element boundary
                std::vector<float> padded = fp32;
                padded.resize(n_blocks * 32, 0.0f);

                for (size_t b = 0; b < n_blocks; ++b)
                {
                    const float *block_data = padded.data() + b * 32;
                    Q16_1Block &blk = blocks[b];

                    // Find max absolute value
                    float max_abs = 0.0f;
                    for (int i = 0; i < 32; ++i)
                    {
                        max_abs = std::max(max_abs, std::fabs(block_data[i]));
                    }

                    // Compute scale: scale = max_abs / 32767.0f
                    float scale = max_abs / 32767.0f;
                    if (scale < 1e-10f)
                        scale = 1e-10f; // Avoid division by zero
                    float inv_scale = (max_abs < 1e-10f) ? 0.0f : (32767.0f / max_abs);

                    // Quantize
                    int32_t sum_qs = 0;
                    for (int i = 0; i < 32; ++i)
                    {
                        int16_t q = static_cast<int16_t>(std::round(block_data[i] * inv_scale));
                        q = std::max<int16_t>(-32767, std::min<int16_t>(32767, q));
                        blk.qs[i] = q;
                        sum_qs += q;
                    }

                    blk.d = scale;
                    blk.sum_qs = sum_qs;
                }

                return blocks;
            }

            // Helper to dequantize Q16_1 blocks to FP32
            std::vector<float> q16_1_blocks_to_fp32(const std::vector<Q16_1Block> &blocks, size_t count)
            {
                std::vector<float> fp32(blocks.size() * 32);

                for (size_t b = 0; b < blocks.size(); ++b)
                {
                    const Q16_1Block &blk = blocks[b];
                    float *block_data = fp32.data() + b * 32;

                    for (int i = 0; i < 32; ++i)
                    {
                        block_data[i] = blk.d * static_cast<float>(blk.qs[i]);
                    }
                }

                fp32.resize(count);
                return fp32;
            }

            // Compute reference sum of multiple arrays
            std::vector<float> compute_reference_sum(const std::vector<std::vector<float>> &inputs)
            {
                if (inputs.empty())
                    return {};

                std::vector<float> result(inputs[0].size(), 0.0f);
                for (const auto &input : inputs)
                {
                    for (size_t i = 0; i < result.size(); ++i)
                    {
                        result[i] += input[i];
                    }
                }
                return result;
            }

            // Compute cosine similarity between two vectors
            float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
            {
                if (a.size() != b.size() || a.empty())
                    return 0.0f;

                float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
                for (size_t i = 0; i < a.size(); ++i)
                {
                    dot += a[i] * b[i];
                    norm_a += a[i] * a[i];
                    norm_b += b[i] * b[i];
                }

                if (norm_a < 1e-10f || norm_b < 1e-10f)
                    return 1.0f; // Both near zero
                return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
            }
        };

        // =============================================================================
        // FP16 N-way Sum Tests
        // =============================================================================

        TEST_F(Test__PrecisionAwareAllreduce, FP16_SumN_TwoInputs)
        {
            const size_t count = 128;

            // Generate two random FP32 inputs
            auto fp32_a = generate_random_fp32(count);
            auto fp32_b = generate_random_fp32(count);

            // Convert to FP16
            auto fp16_a = fp32_to_fp16_vec(fp32_a);
            auto fp16_b = fp32_to_fp16_vec(fp32_b);

            // Set up inputs for N-way sum
            std::vector<const uint16_t *> inputs = {fp16_a.data(), fp16_b.data()};
            std::vector<uint16_t> output(count);

            // Perform N-way sum
            simd::fp16_sum_n(inputs.data(), 2, output.data(), count);

            // Compute reference in FP32
            auto ref_sum = compute_reference_sum({fp16_to_fp32_vec(fp16_a), fp16_to_fp32_vec(fp16_b)});

            // Convert output to FP32 for comparison
            auto result = fp16_to_fp32_vec(output);

            // Check cosine similarity (allow for FP16 precision loss)
            float sim = cosine_similarity(result, ref_sum);
            EXPECT_GT(sim, 0.999f) << "FP16 2-way sum should have high similarity to reference";
        }

        TEST_F(Test__PrecisionAwareAllreduce, FP16_SumN_FourInputs)
        {
            const size_t count = 256;
            const size_t n_inputs = 4;

            // Generate N random inputs
            std::vector<std::vector<float>> fp32_inputs(n_inputs);
            std::vector<std::vector<uint16_t>> fp16_inputs(n_inputs);
            std::vector<const uint16_t *> input_ptrs(n_inputs);

            for (size_t i = 0; i < n_inputs; ++i)
            {
                fp32_inputs[i] = generate_random_fp32(count);
                fp16_inputs[i] = fp32_to_fp16_vec(fp32_inputs[i]);
                input_ptrs[i] = fp16_inputs[i].data();
            }

            std::vector<uint16_t> output(count);
            simd::fp16_sum_n(input_ptrs.data(), n_inputs, output.data(), count);

            // Compute reference
            std::vector<std::vector<float>> fp16_as_fp32(n_inputs);
            for (size_t i = 0; i < n_inputs; ++i)
            {
                fp16_as_fp32[i] = fp16_to_fp32_vec(fp16_inputs[i]);
            }
            auto ref_sum = compute_reference_sum(fp16_as_fp32);

            auto result = fp16_to_fp32_vec(output);
            float sim = cosine_similarity(result, ref_sum);
            EXPECT_GT(sim, 0.998f) << "FP16 4-way sum should have high similarity to reference";
        }

        TEST_F(Test__PrecisionAwareAllreduce, FP16_SumN_SingleInput)
        {
            const size_t count = 64;
            auto fp32 = generate_random_fp32(count);
            auto fp16 = fp32_to_fp16_vec(fp32);

            std::vector<const uint16_t *> inputs = {fp16.data()};
            std::vector<uint16_t> output(count);

            simd::fp16_sum_n(inputs.data(), 1, output.data(), count);

            // Single input should be copied directly
            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_EQ(output[i], fp16[i]) << "Single input should be copied unchanged at index " << i;
            }
        }

        // =============================================================================
        // BF16 N-way Sum Tests
        // =============================================================================

        TEST_F(Test__PrecisionAwareAllreduce, BF16_SumN_TwoInputs)
        {
            const size_t count = 128;

            auto fp32_a = generate_random_fp32(count);
            auto fp32_b = generate_random_fp32(count);

            auto bf16_a = fp32_to_bf16_vec(fp32_a);
            auto bf16_b = fp32_to_bf16_vec(fp32_b);

            std::vector<const uint16_t *> inputs = {bf16_a.data(), bf16_b.data()};
            std::vector<uint16_t> output(count);

            simd::bf16_sum_n(inputs.data(), 2, output.data(), count);

            auto ref_sum = compute_reference_sum({bf16_to_fp32_vec(bf16_a), bf16_to_fp32_vec(bf16_b)});
            auto result = bf16_to_fp32_vec(output);

            float sim = cosine_similarity(result, ref_sum);
            EXPECT_GT(sim, 0.999f) << "BF16 2-way sum should have high similarity to reference";
        }

        TEST_F(Test__PrecisionAwareAllreduce, BF16_SumN_FourInputs)
        {
            const size_t count = 256;
            const size_t n_inputs = 4;

            std::vector<std::vector<float>> fp32_inputs(n_inputs);
            std::vector<std::vector<uint16_t>> bf16_inputs(n_inputs);
            std::vector<const uint16_t *> input_ptrs(n_inputs);

            for (size_t i = 0; i < n_inputs; ++i)
            {
                fp32_inputs[i] = generate_random_fp32(count);
                bf16_inputs[i] = fp32_to_bf16_vec(fp32_inputs[i]);
                input_ptrs[i] = bf16_inputs[i].data();
            }

            std::vector<uint16_t> output(count);
            simd::bf16_sum_n(input_ptrs.data(), n_inputs, output.data(), count);

            std::vector<std::vector<float>> bf16_as_fp32(n_inputs);
            for (size_t i = 0; i < n_inputs; ++i)
            {
                bf16_as_fp32[i] = bf16_to_fp32_vec(bf16_inputs[i]);
            }
            auto ref_sum = compute_reference_sum(bf16_as_fp32);

            auto result = bf16_to_fp32_vec(output);
            float sim = cosine_similarity(result, ref_sum);
            EXPECT_GT(sim, 0.998f) << "BF16 4-way sum should have high similarity to reference";
        }

        // =============================================================================
        // Q8_1 N-way Sum Tests
        // =============================================================================

        TEST_F(Test__PrecisionAwareAllreduce, Q8_1_SumN_TwoInputs)
        {
            const size_t count = 128; // 4 blocks of 32

            auto fp32_a = generate_random_fp32(count);
            auto fp32_b = generate_random_fp32(count);

            auto q8_a = fp32_to_q8_1_blocks(fp32_a);
            auto q8_b = fp32_to_q8_1_blocks(fp32_b);

            std::vector<const Q8_1Block *> inputs = {q8_a.data(), q8_b.data()};
            std::vector<Q8_1Block> output(q8_a.size());

            simd::q8_1_sum_n(inputs.data(), 2, output.data(), q8_a.size());

            // Compute reference from quantized values (not original FP32)
            auto dequant_a = q8_1_blocks_to_fp32(q8_a, count);
            auto dequant_b = q8_1_blocks_to_fp32(q8_b, count);
            auto ref_sum = compute_reference_sum({dequant_a, dequant_b});

            auto result = q8_1_blocks_to_fp32(output, count);

            float sim = cosine_similarity(result, ref_sum);
            // Q8_1 has more quantization noise, allow slightly lower threshold
            EXPECT_GT(sim, 0.995f) << "Q8_1 2-way sum should have high similarity to reference";
        }

        TEST_F(Test__PrecisionAwareAllreduce, Q8_1_SumN_FourInputs)
        {
            const size_t count = 256;
            const size_t n_inputs = 4;
            const size_t n_blocks = (count + 31) / 32;

            std::vector<std::vector<float>> fp32_inputs(n_inputs);
            std::vector<std::vector<Q8_1Block>> q8_inputs(n_inputs);
            std::vector<const Q8_1Block *> input_ptrs(n_inputs);

            for (size_t i = 0; i < n_inputs; ++i)
            {
                fp32_inputs[i] = generate_random_fp32(count);
                q8_inputs[i] = fp32_to_q8_1_blocks(fp32_inputs[i]);
                input_ptrs[i] = q8_inputs[i].data();
            }

            std::vector<Q8_1Block> output(n_blocks);
            simd::q8_1_sum_n(input_ptrs.data(), n_inputs, output.data(), n_blocks);

            std::vector<std::vector<float>> dequant_inputs(n_inputs);
            for (size_t i = 0; i < n_inputs; ++i)
            {
                dequant_inputs[i] = q8_1_blocks_to_fp32(q8_inputs[i], count);
            }
            auto ref_sum = compute_reference_sum(dequant_inputs);

            auto result = q8_1_blocks_to_fp32(output, count);
            float sim = cosine_similarity(result, ref_sum);
            EXPECT_GT(sim, 0.99f) << "Q8_1 4-way sum should have high similarity to reference";
        }

        TEST_F(Test__PrecisionAwareAllreduce, Q8_1_SumN_SingleInput)
        {
            const size_t count = 64;
            auto fp32 = generate_random_fp32(count);
            auto q8 = fp32_to_q8_1_blocks(fp32);

            std::vector<const Q8_1Block *> inputs = {q8.data()};
            std::vector<Q8_1Block> output(q8.size());

            simd::q8_1_sum_n(inputs.data(), 1, output.data(), q8.size());

            // Single input should be copied directly
            EXPECT_EQ(memcmp(output.data(), q8.data(), q8.size() * sizeof(Q8_1Block)), 0)
                << "Single input should be copied unchanged";
        }

        // =============================================================================
        // Edge Cases
        // =============================================================================

        TEST_F(Test__PrecisionAwareAllreduce, FP16_SumN_EmptyInput)
        {
            std::vector<uint16_t> output(16, 0xFFFF); // Fill with sentinel
            simd::fp16_sum_n(nullptr, 0, output.data(), 16);
            // Should be no-op, output unchanged (or zeroed, depending on impl)
            // Just verify it doesn't crash
            SUCCEED();
        }

        TEST_F(Test__PrecisionAwareAllreduce, BF16_SumN_TailElements)
        {
            // Test with non-aligned count (17 elements = not divisible by 8 or 16)
            const size_t count = 17;

            auto fp32_a = generate_random_fp32(count);
            auto fp32_b = generate_random_fp32(count);

            auto bf16_a = fp32_to_bf16_vec(fp32_a);
            auto bf16_b = fp32_to_bf16_vec(fp32_b);

            std::vector<const uint16_t *> inputs = {bf16_a.data(), bf16_b.data()};
            std::vector<uint16_t> output(count);

            simd::bf16_sum_n(inputs.data(), 2, output.data(), count);

            auto ref_sum = compute_reference_sum({bf16_to_fp32_vec(bf16_a), bf16_to_fp32_vec(bf16_b)});
            auto result = bf16_to_fp32_vec(output);

            float sim = cosine_similarity(result, ref_sum);
            EXPECT_GT(sim, 0.999f) << "BF16 sum with tail elements should work correctly";
        }

        TEST_F(Test__PrecisionAwareAllreduce, Q8_1_SumN_LargeValues)
        {
            // Test with larger values that might cause saturation
            const size_t count = 128;
            auto fp32_a = generate_random_fp32(count, -100.0f, 100.0f);
            auto fp32_b = generate_random_fp32(count, -100.0f, 100.0f);

            auto q8_a = fp32_to_q8_1_blocks(fp32_a);
            auto q8_b = fp32_to_q8_1_blocks(fp32_b);

            std::vector<const Q8_1Block *> inputs = {q8_a.data(), q8_b.data()};
            std::vector<Q8_1Block> output(q8_a.size());

            simd::q8_1_sum_n(inputs.data(), 2, output.data(), q8_a.size());

            auto dequant_a = q8_1_blocks_to_fp32(q8_a, count);
            auto dequant_b = q8_1_blocks_to_fp32(q8_b, count);
            auto ref_sum = compute_reference_sum({dequant_a, dequant_b});

            auto result = q8_1_blocks_to_fp32(output, count);
            float sim = cosine_similarity(result, ref_sum);
            EXPECT_GT(sim, 0.99f) << "Q8_1 sum with large values should maintain precision";
        }

        // =============================================================================
        // Q16_1 N-way Sum Tests
        // =============================================================================

        TEST_F(Test__PrecisionAwareAllreduce, Q16_1_SumN_TwoInputs)
        {
            const size_t count = 128; // 4 blocks of 32

            auto fp32_a = generate_random_fp32(count);
            auto fp32_b = generate_random_fp32(count);

            auto q16_a = fp32_to_q16_1_blocks(fp32_a);
            auto q16_b = fp32_to_q16_1_blocks(fp32_b);

            std::vector<const Q16_1Block *> inputs = {q16_a.data(), q16_b.data()};
            std::vector<Q16_1Block> output(q16_a.size());

            simd::q16_1_sum_n(inputs.data(), 2, output.data(), q16_a.size());

            // Compute reference from quantized values (not original FP32)
            auto dequant_a = q16_1_blocks_to_fp32(q16_a, count);
            auto dequant_b = q16_1_blocks_to_fp32(q16_b, count);
            auto ref_sum = compute_reference_sum({dequant_a, dequant_b});

            auto result = q16_1_blocks_to_fp32(output, count);

            float sim = cosine_similarity(result, ref_sum);
            // Q16_1 has better precision than Q8_1 due to 16-bit quantization
            EXPECT_GT(sim, 0.999f) << "Q16_1 2-way sum should have high similarity to reference";
        }

        TEST_F(Test__PrecisionAwareAllreduce, Q16_1_SumN_FourInputs)
        {
            const size_t count = 256;
            const size_t n_inputs = 4;
            const size_t n_blocks = (count + 31) / 32;

            std::vector<std::vector<float>> fp32_inputs(n_inputs);
            std::vector<std::vector<Q16_1Block>> q16_inputs(n_inputs);
            std::vector<const Q16_1Block *> input_ptrs(n_inputs);

            for (size_t i = 0; i < n_inputs; ++i)
            {
                fp32_inputs[i] = generate_random_fp32(count);
                q16_inputs[i] = fp32_to_q16_1_blocks(fp32_inputs[i]);
                input_ptrs[i] = q16_inputs[i].data();
            }

            std::vector<Q16_1Block> output(n_blocks);
            simd::q16_1_sum_n(input_ptrs.data(), n_inputs, output.data(), n_blocks);

            std::vector<std::vector<float>> dequant_inputs(n_inputs);
            for (size_t i = 0; i < n_inputs; ++i)
            {
                dequant_inputs[i] = q16_1_blocks_to_fp32(q16_inputs[i], count);
            }
            auto ref_sum = compute_reference_sum(dequant_inputs);

            auto result = q16_1_blocks_to_fp32(output, count);
            float sim = cosine_similarity(result, ref_sum);
            EXPECT_GT(sim, 0.995f) << "Q16_1 4-way sum should have high similarity to reference";
        }

        TEST_F(Test__PrecisionAwareAllreduce, Q16_1_SumN_SingleInput)
        {
            const size_t count = 64;
            auto fp32 = generate_random_fp32(count);
            auto q16 = fp32_to_q16_1_blocks(fp32);

            std::vector<const Q16_1Block *> inputs = {q16.data()};
            std::vector<Q16_1Block> output(q16.size());

            simd::q16_1_sum_n(inputs.data(), 1, output.data(), q16.size());

            // Single input should be copied directly
            EXPECT_EQ(memcmp(output.data(), q16.data(), q16.size() * sizeof(Q16_1Block)), 0)
                << "Single input should be copied unchanged";
        }

        TEST_F(Test__PrecisionAwareAllreduce, Q16_1_SumN_LargeValues)
        {
            // Test with larger values that might cause saturation
            const size_t count = 128;
            auto fp32_a = generate_random_fp32(count, -100.0f, 100.0f);
            auto fp32_b = generate_random_fp32(count, -100.0f, 100.0f);

            auto q16_a = fp32_to_q16_1_blocks(fp32_a);
            auto q16_b = fp32_to_q16_1_blocks(fp32_b);

            std::vector<const Q16_1Block *> inputs = {q16_a.data(), q16_b.data()};
            std::vector<Q16_1Block> output(q16_a.size());

            simd::q16_1_sum_n(inputs.data(), 2, output.data(), q16_a.size());

            auto dequant_a = q16_1_blocks_to_fp32(q16_a, count);
            auto dequant_b = q16_1_blocks_to_fp32(q16_b, count);
            auto ref_sum = compute_reference_sum({dequant_a, dequant_b});

            auto result = q16_1_blocks_to_fp32(output, count);
            float sim = cosine_similarity(result, ref_sum);
            EXPECT_GT(sim, 0.995f) << "Q16_1 sum with large values should maintain precision";
        }

        TEST_F(Test__PrecisionAwareAllreduce, Q16_1_SumN_EightInputs)
        {
            // Test with 8 inputs (common for 8-rank MPI)
            const size_t count = 896; // d_model size
            const size_t n_inputs = 8;
            const size_t n_blocks = (count + 31) / 32;

            std::vector<std::vector<float>> fp32_inputs(n_inputs);
            std::vector<std::vector<Q16_1Block>> q16_inputs(n_inputs);
            std::vector<const Q16_1Block *> input_ptrs(n_inputs);

            for (size_t i = 0; i < n_inputs; ++i)
            {
                fp32_inputs[i] = generate_random_fp32(count, -1.0f, 1.0f);
                q16_inputs[i] = fp32_to_q16_1_blocks(fp32_inputs[i]);
                input_ptrs[i] = q16_inputs[i].data();
            }

            std::vector<Q16_1Block> output(n_blocks);
            simd::q16_1_sum_n(input_ptrs.data(), n_inputs, output.data(), n_blocks);

            std::vector<std::vector<float>> dequant_inputs(n_inputs);
            for (size_t i = 0; i < n_inputs; ++i)
            {
                dequant_inputs[i] = q16_1_blocks_to_fp32(q16_inputs[i], count);
            }
            auto ref_sum = compute_reference_sum(dequant_inputs);

            auto result = q16_1_blocks_to_fp32(output, count);
            float sim = cosine_similarity(result, ref_sum);
            EXPECT_GT(sim, 0.99f) << "Q16_1 8-way sum should have high similarity to reference";
        }

        TEST_F(Test__PrecisionAwareAllreduce, Q16_1_SumN_ZeroInputs)
        {
            const size_t count = 64;
            const size_t n_blocks = (count + 31) / 32;

            // Create inputs with all zeros
            std::vector<float> fp32_zeros(count, 0.0f);
            auto q16_a = fp32_to_q16_1_blocks(fp32_zeros);
            auto q16_b = fp32_to_q16_1_blocks(fp32_zeros);

            std::vector<const Q16_1Block *> inputs = {q16_a.data(), q16_b.data()};
            std::vector<Q16_1Block> output(n_blocks);

            simd::q16_1_sum_n(inputs.data(), 2, output.data(), n_blocks);

            // Result should be all zeros
            auto result = q16_1_blocks_to_fp32(output, count);
            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_NEAR(result[i], 0.0f, 1e-6f) << "Zero sum should produce zero at index " << i;
            }
        }

        TEST_F(Test__PrecisionAwareAllreduce, Q16_1_SumN_MixedSigns)
        {
            // Test with mixed positive and negative values that should cancel
            const size_t count = 32; // Single block

            std::vector<float> fp32_pos(count);
            std::vector<float> fp32_neg(count);
            for (size_t i = 0; i < count; ++i)
            {
                fp32_pos[i] = 0.5f;
                fp32_neg[i] = -0.5f;
            }

            auto q16_pos = fp32_to_q16_1_blocks(fp32_pos);
            auto q16_neg = fp32_to_q16_1_blocks(fp32_neg);

            std::vector<const Q16_1Block *> inputs = {q16_pos.data(), q16_neg.data()};
            std::vector<Q16_1Block> output(1);

            simd::q16_1_sum_n(inputs.data(), 2, output.data(), 1);

            // Result should be near zero
            auto result = q16_1_blocks_to_fp32(output, count);
            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_NEAR(result[i], 0.0f, 0.001f) << "Canceling values should sum to near-zero at index " << i;
            }
        }

    } // namespace testing
} // namespace llaminar2
