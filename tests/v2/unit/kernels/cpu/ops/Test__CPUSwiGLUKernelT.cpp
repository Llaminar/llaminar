/**
 * @file Test__CPUSwiGLUKernelT.cpp
 * @brief Unit tests for CPUSwiGLUKernelT (FP32/BF16/FP16/Q8_1 specializations)
 * @author David Sanftenberg
 *
 * Tests verify:
 * 1. Basic functionality for each precision
 * 2. SwiGLU formula: output = silu(gate) * up
 * 3. Cross-precision consistency (all precisions produce similar results)
 * 4. Q8_1 integer-aware path produces results comparable to FP32 reference
 * 5. Edge cases (zeros, negatives, large values)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

#include "kernels/cpu/ops/CPUSwiGLUKernelT.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include "tensors/SIMDHelpers.h"

namespace llaminar2
{
    namespace
    {
        // Use shared conversion utilities from SIMDHelpers.h
        using simd::bf16_to_fp32;
        using simd::fp32_to_bf16;
        // FP16 utilities from FP16Utils.h are in llaminar2:: namespace

        // ============================================================================
        // Helper Functions
        // ============================================================================

        /**
         * @brief Reference SiLU (Swish) implementation
         */
        inline float silu_reference(float x)
        {
            return x / (1.0f + std::exp(-x));
        }

        /**
         * @brief Reference SwiGLU implementation
         */
        inline float swiglu_reference(float gate, float up)
        {
            return silu_reference(gate) * up;
        }

        /**
         * @brief Compute cosine similarity between two vectors
         */
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

            if (norm_a < 1e-12f || norm_b < 1e-12f)
                return 1.0f; // Both zero vectors

            return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        }

        /**
         * @brief Create Q8_1 block from FP32 values
         * @note Q8_1Block stores scale as IEEE FP16 (not BF16!)
         */
        Q8_1Block create_q8_1_block(const float *values)
        {
            Q8_1Block block;

            // Find max absolute value for scale
            float max_abs = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                max_abs = std::max(max_abs, std::abs(values[i]));
            }

            // Compute scale (d) such that max maps to 127
            float d = max_abs / 127.0f;
            if (d < 1e-10f)
                d = 1e-10f;

            // Store d as IEEE FP16 (NOT bf16!)
            block.d = fp32_to_fp16(d);

            // Quantize values
            float inv_d = 1.0f / d;
            int16_t sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(values[i] * inv_d));
                q = std::max(-127, std::min(127, q));
                block.qs[i] = static_cast<int8_t>(q);
                sum_qs += block.qs[i];
            }
            block.sum_qs = sum_qs;

            return block;
        }

        /**
         * @brief Dequantize Q8_1 block to FP32 vector
         * @note Q8_1Block stores scale as IEEE FP16 (not BF16!)
         */
        std::vector<float> dequantize_q8_1_block(const Q8_1Block &block)
        {
            std::vector<float> result(32);
            float d = fp16_to_fp32(block.d); // IEEE FP16, not BF16!
            for (int i = 0; i < 32; ++i)
            {
                result[i] = static_cast<float>(block.qs[i]) * d;
            }
            return result;
        }

    } // anonymous namespace

    // ============================================================================
    // FP32 Tests
    // ============================================================================

    TEST(CPUSwiGLUKernelT_FP32, BasicFunctionality)
    {
        CPUSwiGLUKernelT<ActivationPrecision::FP32> kernel;
        EXPECT_STREQ(kernel.precision_name(), "FP32");
        EXPECT_FLOAT_EQ(kernel.compression_ratio(), 1.0f);
        EXPECT_TRUE(kernel.supports_device(-1));

        const int size = 128;
        std::vector<float> gate(size), up(size), output(size);

        // Initialize with test values
        for (int i = 0; i < size; ++i)
        {
            gate[i] = static_cast<float>(i - 64) * 0.1f; // Range: [-6.4, 6.3]
            up[i] = static_cast<float>(i) * 0.01f;       // Range: [0, 1.27]
        }

        ASSERT_TRUE(kernel.apply_typed(gate.data(), up.data(), output.data(), size));

        // Verify against reference
        for (int i = 0; i < size; ++i)
        {
            float expected = swiglu_reference(gate[i], up[i]);
            EXPECT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
        }
    }

    TEST(CPUSwiGLUKernelT_FP32, ZeroInputs)
    {
        CPUSwiGLUKernelT<ActivationPrecision::FP32> kernel;
        const int size = 64;
        std::vector<float> gate(size, 0.0f), up(size, 0.0f), output(size);

        ASSERT_TRUE(kernel.apply_typed(gate.data(), up.data(), output.data(), size));

        // silu(0) * 0 = 0
        for (int i = 0; i < size; ++i)
        {
            EXPECT_FLOAT_EQ(output[i], 0.0f);
        }
    }

    TEST(CPUSwiGLUKernelT_FP32, NegativeGate)
    {
        CPUSwiGLUKernelT<ActivationPrecision::FP32> kernel;
        const int size = 32;
        std::vector<float> gate(size), up(size, 1.0f), output(size);

        // All negative gate values
        for (int i = 0; i < size; ++i)
        {
            gate[i] = -static_cast<float>(i + 1);
        }

        ASSERT_TRUE(kernel.apply_typed(gate.data(), up.data(), output.data(), size));

        // Verify: silu(x) for negative x is small but non-zero
        for (int i = 0; i < size; ++i)
        {
            float expected = swiglu_reference(gate[i], up[i]);
            EXPECT_NEAR(output[i], expected, 1e-5f);
            EXPECT_LT(output[i], 0.0f); // silu of negative * positive is negative
        }
    }

    TEST(CPUSwiGLUKernelT_FP32, LargeValues)
    {
        CPUSwiGLUKernelT<ActivationPrecision::FP32> kernel;
        const int size = 32;
        std::vector<float> gate(size), up(size), output(size);

        // Large positive values (silu approaches x for large positive x)
        for (int i = 0; i < size; ++i)
        {
            gate[i] = 10.0f + i * 0.5f;
            up[i] = 2.0f;
        }

        ASSERT_TRUE(kernel.apply_typed(gate.data(), up.data(), output.data(), size));

        // For large positive x: silu(x) ≈ x, so output ≈ gate * up
        for (int i = 0; i < size; ++i)
        {
            float expected = swiglu_reference(gate[i], up[i]);
            EXPECT_NEAR(output[i], expected, 1e-4f);
            // silu(x) ≈ x for x >> 0
            EXPECT_NEAR(output[i], gate[i] * up[i], gate[i] * 0.01f);
        }
    }

    // ============================================================================
    // BF16 Tests
    // ============================================================================

    TEST(CPUSwiGLUKernelT_BF16, BasicFunctionality)
    {
        CPUSwiGLUKernelT<ActivationPrecision::BF16> kernel;
        EXPECT_STREQ(kernel.precision_name(), "BF16");
        EXPECT_FLOAT_EQ(kernel.compression_ratio(), 2.0f);

        const int size = 128;
        std::vector<uint16_t> gate(size), up(size), output(size);

        // Initialize with test values
        for (int i = 0; i < size; ++i)
        {
            gate[i] = fp32_to_bf16(static_cast<float>(i - 64) * 0.1f);
            up[i] = fp32_to_bf16(static_cast<float>(i) * 0.01f);
        }

        ASSERT_TRUE(kernel.apply_typed(gate.data(), up.data(), output.data(), size));

        // Verify against reference (with BF16 tolerance)
        for (int i = 0; i < size; ++i)
        {
            float g = bf16_to_fp32(gate[i]);
            float u = bf16_to_fp32(up[i]);
            float expected = swiglu_reference(g, u);
            float actual = bf16_to_fp32(output[i]);
            EXPECT_NEAR(actual, expected, std::abs(expected) * 0.02f + 1e-4f)
                << "Mismatch at index " << i;
        }
    }

    TEST(CPUSwiGLUKernelT_BF16, ConsistencyWithFP32)
    {
        CPUSwiGLUKernelT<ActivationPrecision::FP32> kernel_fp32;
        CPUSwiGLUKernelT<ActivationPrecision::BF16> kernel_bf16;

        const int size = 256;
        std::vector<float> gate_fp32(size), up_fp32(size), output_fp32(size);
        std::vector<uint16_t> gate_bf16(size), up_bf16(size), output_bf16(size);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

        for (int i = 0; i < size; ++i)
        {
            gate_fp32[i] = dist(rng);
            up_fp32[i] = dist(rng);
            gate_bf16[i] = fp32_to_bf16(gate_fp32[i]);
            up_bf16[i] = fp32_to_bf16(up_fp32[i]);
        }

        ASSERT_TRUE(kernel_fp32.apply_typed(gate_fp32.data(), up_fp32.data(), output_fp32.data(), size));
        ASSERT_TRUE(kernel_bf16.apply_typed(gate_bf16.data(), up_bf16.data(), output_bf16.data(), size));

        // Convert BF16 output to FP32 for comparison
        std::vector<float> output_bf16_as_fp32(size);
        for (int i = 0; i < size; ++i)
        {
            output_bf16_as_fp32[i] = bf16_to_fp32(output_bf16[i]);
        }

        float cos_sim = cosine_similarity(output_fp32, output_bf16_as_fp32);
        EXPECT_GT(cos_sim, 0.999f) << "BF16 output should be very close to FP32";
    }

    // ============================================================================
    // FP16 Tests
    // ============================================================================

    TEST(CPUSwiGLUKernelT_FP16, BasicFunctionality)
    {
        CPUSwiGLUKernelT<ActivationPrecision::FP16> kernel;
        EXPECT_STREQ(kernel.precision_name(), "FP16");
        EXPECT_FLOAT_EQ(kernel.compression_ratio(), 2.0f);

        const int size = 128;
        std::vector<uint16_t> gate(size), up(size), output(size);

        // Initialize with test values (staying in FP16 representable range)
        for (int i = 0; i < size; ++i)
        {
            gate[i] = fp32_to_fp16(static_cast<float>(i - 64) * 0.05f);
            up[i] = fp32_to_fp16(static_cast<float>(i) * 0.01f);
        }

        ASSERT_TRUE(kernel.apply_typed(gate.data(), up.data(), output.data(), size));

        // Verify against reference (with FP16 tolerance)
        for (int i = 0; i < size; ++i)
        {
            float g = fp16_to_fp32(gate[i]);
            float u = fp16_to_fp32(up[i]);
            float expected = swiglu_reference(g, u);
            float actual = fp16_to_fp32(output[i]);
            EXPECT_NEAR(actual, expected, std::abs(expected) * 0.02f + 1e-3f)
                << "Mismatch at index " << i;
        }
    }

    TEST(CPUSwiGLUKernelT_FP16, ConsistencyWithFP32)
    {
        CPUSwiGLUKernelT<ActivationPrecision::FP32> kernel_fp32;
        CPUSwiGLUKernelT<ActivationPrecision::FP16> kernel_fp16;

        const int size = 256;
        std::vector<float> gate_fp32(size), up_fp32(size), output_fp32(size);
        std::vector<uint16_t> gate_fp16(size), up_fp16(size), output_fp16(size);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

        for (int i = 0; i < size; ++i)
        {
            gate_fp32[i] = dist(rng);
            up_fp32[i] = dist(rng);
            gate_fp16[i] = fp32_to_fp16(gate_fp32[i]);
            up_fp16[i] = fp32_to_fp16(up_fp32[i]);
        }

        ASSERT_TRUE(kernel_fp32.apply_typed(gate_fp32.data(), up_fp32.data(), output_fp32.data(), size));
        ASSERT_TRUE(kernel_fp16.apply_typed(gate_fp16.data(), up_fp16.data(), output_fp16.data(), size));

        // Convert FP16 output to FP32 for comparison
        std::vector<float> output_fp16_as_fp32(size);
        for (int i = 0; i < size; ++i)
        {
            output_fp16_as_fp32[i] = fp16_to_fp32(output_fp16[i]);
        }

        float cos_sim = cosine_similarity(output_fp32, output_fp16_as_fp32);
        EXPECT_GT(cos_sim, 0.998f) << "FP16 output should be close to FP32";
    }

    // ============================================================================
    // Q8_1 Tests
    // ============================================================================

    TEST(CPUSwiGLUKernelT_Q8_1, BasicFunctionality)
    {
        CPUSwiGLUKernelT<ActivationPrecision::Q8_1> kernel;
        EXPECT_STREQ(kernel.precision_name(), "Q8_1");
        EXPECT_FLOAT_EQ(kernel.compression_ratio(), 4.0f);

        const int n_blocks = 4;
        const int size = n_blocks * 32;

        // Create test data
        std::vector<float> gate_fp32(size), up_fp32(size);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int i = 0; i < size; ++i)
        {
            gate_fp32[i] = dist(rng);
            up_fp32[i] = dist(rng);
        }

        // Quantize to Q8_1
        std::vector<Q8_1Block> gate_q8(n_blocks), up_q8(n_blocks), output_q8(n_blocks);
        for (int b = 0; b < n_blocks; ++b)
        {
            gate_q8[b] = create_q8_1_block(&gate_fp32[b * 32]);
            up_q8[b] = create_q8_1_block(&up_fp32[b * 32]);
        }

        ASSERT_TRUE(kernel.apply_typed(gate_q8.data(), up_q8.data(), output_q8.data(), size));

        // Verify: dequantize output and compare with reference
        std::vector<float> output_dequant;
        for (int b = 0; b < n_blocks; ++b)
        {
            auto block_dequant = dequantize_q8_1_block(output_q8[b]);
            output_dequant.insert(output_dequant.end(), block_dequant.begin(), block_dequant.end());
        }

        // Compute reference from dequantized inputs
        std::vector<float> gate_dequant, up_dequant;
        for (int b = 0; b < n_blocks; ++b)
        {
            auto g = dequantize_q8_1_block(gate_q8[b]);
            auto u = dequantize_q8_1_block(up_q8[b]);
            gate_dequant.insert(gate_dequant.end(), g.begin(), g.end());
            up_dequant.insert(up_dequant.end(), u.begin(), u.end());
        }

        std::vector<float> reference(size);
        for (int i = 0; i < size; ++i)
        {
            reference[i] = swiglu_reference(gate_dequant[i], up_dequant[i]);
        }

        // Q8_1 has quantization error, use cosine similarity
        float cos_sim = cosine_similarity(output_dequant, reference);
        EXPECT_GT(cos_sim, 0.98f) << "Q8_1 output should be reasonably close to reference";
    }

    TEST(CPUSwiGLUKernelT_Q8_1, ConsistencyWithFP32)
    {
        CPUSwiGLUKernelT<ActivationPrecision::FP32> kernel_fp32;
        CPUSwiGLUKernelT<ActivationPrecision::Q8_1> kernel_q8;

        const int n_blocks = 8;
        const int size = n_blocks * 32;

        std::vector<float> gate_fp32(size), up_fp32(size), output_fp32(size);
        std::vector<Q8_1Block> gate_q8(n_blocks), up_q8(n_blocks), output_q8(n_blocks);

        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

        for (int i = 0; i < size; ++i)
        {
            gate_fp32[i] = dist(rng);
            up_fp32[i] = dist(rng);
        }

        // Quantize
        for (int b = 0; b < n_blocks; ++b)
        {
            gate_q8[b] = create_q8_1_block(&gate_fp32[b * 32]);
            up_q8[b] = create_q8_1_block(&up_fp32[b * 32]);
        }

        ASSERT_TRUE(kernel_fp32.apply_typed(gate_fp32.data(), up_fp32.data(), output_fp32.data(), size));
        ASSERT_TRUE(kernel_q8.apply_typed(gate_q8.data(), up_q8.data(), output_q8.data(), size));

        // Dequantize Q8_1 output
        std::vector<float> output_q8_dequant;
        for (int b = 0; b < n_blocks; ++b)
        {
            auto block = dequantize_q8_1_block(output_q8[b]);
            output_q8_dequant.insert(output_q8_dequant.end(), block.begin(), block.end());
        }

        float cos_sim = cosine_similarity(output_fp32, output_q8_dequant);
        EXPECT_GT(cos_sim, 0.95f) << "Q8_1 output should be reasonably close to FP32";
    }

    TEST(CPUSwiGLUKernelT_Q8_1, MultipleBlocks)
    {
        CPUSwiGLUKernelT<ActivationPrecision::Q8_1> kernel;

        const int n_blocks = 16;
        const int size = n_blocks * 32;

        std::vector<float> gate_fp32(size), up_fp32(size);
        std::mt19937 rng(456);
        std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

        for (int i = 0; i < size; ++i)
        {
            gate_fp32[i] = dist(rng);
            up_fp32[i] = dist(rng);
        }

        std::vector<Q8_1Block> gate_q8(n_blocks), up_q8(n_blocks), output_q8(n_blocks);
        for (int b = 0; b < n_blocks; ++b)
        {
            gate_q8[b] = create_q8_1_block(&gate_fp32[b * 32]);
            up_q8[b] = create_q8_1_block(&up_fp32[b * 32]);
        }

        ASSERT_TRUE(kernel.apply_typed(gate_q8.data(), up_q8.data(), output_q8.data(), size));

        // Verify each block has valid sum_qs
        for (int b = 0; b < n_blocks; ++b)
        {
            int16_t computed_sum = 0;
            for (int i = 0; i < 32; ++i)
            {
                computed_sum += output_q8[b].qs[i];
            }
            EXPECT_EQ(output_q8[b].sum_qs, computed_sum) << "Block " << b << " sum_qs mismatch";
        }
    }

    // ============================================================================
    // Cross-Precision Tests
    // ============================================================================

    TEST(CPUSwiGLUKernelT_CrossPrecision, AllPrecisionsProduceSimilarResults)
    {
        CPUSwiGLUKernelT<ActivationPrecision::FP32> kernel_fp32;
        CPUSwiGLUKernelT<ActivationPrecision::BF16> kernel_bf16;
        CPUSwiGLUKernelT<ActivationPrecision::FP16> kernel_fp16;
        CPUSwiGLUKernelT<ActivationPrecision::Q8_1> kernel_q8;

        const int n_blocks = 4;
        const int size = n_blocks * 32;

        // Generate test data
        std::vector<float> gate_fp32(size), up_fp32(size), output_fp32(size);
        std::vector<uint16_t> gate_bf16(size), up_bf16(size), output_bf16(size);
        std::vector<uint16_t> gate_fp16(size), up_fp16(size), output_fp16(size);
        std::vector<Q8_1Block> gate_q8(n_blocks), up_q8(n_blocks), output_q8(n_blocks);

        std::mt19937 rng(789);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int i = 0; i < size; ++i)
        {
            gate_fp32[i] = dist(rng);
            up_fp32[i] = dist(rng);
            gate_bf16[i] = fp32_to_bf16(gate_fp32[i]);
            up_bf16[i] = fp32_to_bf16(up_fp32[i]);
            gate_fp16[i] = fp32_to_fp16(gate_fp32[i]);
            up_fp16[i] = fp32_to_fp16(up_fp32[i]);
        }

        for (int b = 0; b < n_blocks; ++b)
        {
            gate_q8[b] = create_q8_1_block(&gate_fp32[b * 32]);
            up_q8[b] = create_q8_1_block(&up_fp32[b * 32]);
        }

        // Run all kernels
        ASSERT_TRUE(kernel_fp32.apply_typed(gate_fp32.data(), up_fp32.data(), output_fp32.data(), size));
        ASSERT_TRUE(kernel_bf16.apply_typed(gate_bf16.data(), up_bf16.data(), output_bf16.data(), size));
        ASSERT_TRUE(kernel_fp16.apply_typed(gate_fp16.data(), up_fp16.data(), output_fp16.data(), size));
        ASSERT_TRUE(kernel_q8.apply_typed(gate_q8.data(), up_q8.data(), output_q8.data(), size));

        // Convert all outputs to FP32 for comparison
        std::vector<float> output_bf16_fp32(size), output_fp16_fp32(size), output_q8_fp32;

        for (int i = 0; i < size; ++i)
        {
            output_bf16_fp32[i] = bf16_to_fp32(output_bf16[i]);
            output_fp16_fp32[i] = fp16_to_fp32(output_fp16[i]);
        }

        for (int b = 0; b < n_blocks; ++b)
        {
            auto block = dequantize_q8_1_block(output_q8[b]);
            output_q8_fp32.insert(output_q8_fp32.end(), block.begin(), block.end());
        }

        // Compare all precisions against FP32
        float cos_bf16 = cosine_similarity(output_fp32, output_bf16_fp32);
        float cos_fp16 = cosine_similarity(output_fp32, output_fp16_fp32);
        float cos_q8 = cosine_similarity(output_fp32, output_q8_fp32);

        EXPECT_GT(cos_bf16, 0.999f) << "BF16 should be very close to FP32";
        EXPECT_GT(cos_fp16, 0.998f) << "FP16 should be close to FP32";
        EXPECT_GT(cos_q8, 0.95f) << "Q8_1 should be reasonably close to FP32";
    }

    // ============================================================================
    // Edge Cases Tests
    // ============================================================================

    TEST(CPUSwiGLUKernelT_EdgeCases, EmptyInput)
    {
        CPUSwiGLUKernelT<ActivationPrecision::FP32> kernel;
        // Use non-null but unused pointers with size=0
        float dummy[1] = {0.0f};
        EXPECT_TRUE(kernel.apply_typed(dummy, dummy, dummy, 0));
    }

    TEST(CPUSwiGLUKernelT_EdgeCases, NullPointer)
    {
        CPUSwiGLUKernelT<ActivationPrecision::FP32> kernel_fp32;
        CPUSwiGLUKernelT<ActivationPrecision::Q8_1> kernel_q8;

        EXPECT_FALSE(kernel_fp32.apply_typed(nullptr, nullptr, nullptr, 10));
        EXPECT_FALSE(kernel_q8.apply_typed(nullptr, nullptr, nullptr, 32));
    }

    TEST(CPUSwiGLUKernelT_EdgeCases, SingleElement)
    {
        // Note: Q8_1 requires multiples of 32, so this only tests FP32/BF16/FP16
        CPUSwiGLUKernelT<ActivationPrecision::FP32> kernel;
        std::vector<float> gate = {1.5f}, up = {2.0f}, output(1);

        ASSERT_TRUE(kernel.apply_typed(gate.data(), up.data(), output.data(), 1));

        float expected = swiglu_reference(1.5f, 2.0f);
        EXPECT_NEAR(output[0], expected, 1e-5f);
    }

    TEST(CPUSwiGLUKernelT_Metadata, PrecisionNames)
    {
        EXPECT_STREQ(CPUSwiGLUKernelT<ActivationPrecision::FP32>::precision_name(), "FP32");
        EXPECT_STREQ(CPUSwiGLUKernelT<ActivationPrecision::BF16>::precision_name(), "BF16");
        EXPECT_STREQ(CPUSwiGLUKernelT<ActivationPrecision::FP16>::precision_name(), "FP16");
        EXPECT_STREQ(CPUSwiGLUKernelT<ActivationPrecision::Q8_1>::precision_name(), "Q8_1");
    }

    TEST(CPUSwiGLUKernelT_Metadata, CompressionRatios)
    {
        EXPECT_FLOAT_EQ(CPUSwiGLUKernelT<ActivationPrecision::FP32>::compression_ratio(), 1.0f);
        EXPECT_FLOAT_EQ(CPUSwiGLUKernelT<ActivationPrecision::BF16>::compression_ratio(), 2.0f);
        EXPECT_FLOAT_EQ(CPUSwiGLUKernelT<ActivationPrecision::FP16>::compression_ratio(), 2.0f);
        EXPECT_FLOAT_EQ(CPUSwiGLUKernelT<ActivationPrecision::Q8_1>::compression_ratio(), 4.0f);
    }

    TEST(CPUSwiGLUKernelT_Metadata, DeviceSupport)
    {
        CPUSwiGLUKernelT<ActivationPrecision::FP32> kernel_fp32;
        CPUSwiGLUKernelT<ActivationPrecision::BF16> kernel_bf16;
        CPUSwiGLUKernelT<ActivationPrecision::FP16> kernel_fp16;
        CPUSwiGLUKernelT<ActivationPrecision::Q8_1> kernel_q8;

        // All kernels should support CPU (device_idx = -1)
        EXPECT_TRUE(kernel_fp32.supports_device(-1));
        EXPECT_TRUE(kernel_bf16.supports_device(-1));
        EXPECT_TRUE(kernel_fp16.supports_device(-1));
        EXPECT_TRUE(kernel_q8.supports_device(-1));

        // All kernels should NOT support GPU devices
        EXPECT_FALSE(kernel_fp32.supports_device(0));
        EXPECT_FALSE(kernel_bf16.supports_device(0));
        EXPECT_FALSE(kernel_fp16.supports_device(0));
        EXPECT_FALSE(kernel_q8.supports_device(0));
    }

} // namespace llaminar2
