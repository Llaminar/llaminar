/**
 * @file Test__Q16WoProjection.cpp
 * @brief Unit tests for Q16 INT16 VNNI Wo projection microkernel
 *
 * Tests the Wo projection implementation in Q16IntegerAttentionRef:
 * - wo_projection_vnni_int16() for decode (single token GEMV)
 * - wo_projection_vnni_int16_batched() for prefill (batched GEMM)
 * - FP32 reference comparison for correctness validation
 *
 * Key scenarios:
 * 1. Simple identity-like weights (debugging)
 * 2. Random weights with known FP32 reference
 * 3. Qwen2.5-0.5B dimensions (d_model=896, n_heads=14, head_dim=64)
 * 4. Scale handling validation
 */

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include <numeric>

#include "kernels/cpu/attention/q16_1/ref/microkernels/WoProjection.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include "kernels/KernelFactory.h"
#include "tensors/BlockStructures.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"

namespace llaminar2::kernels::q16_1::microkernels
{

    // ============================================================================
    // Test Fixture
    // ============================================================================

    class Test__Q16WoProjection : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Seed for reproducibility
            gen_.seed(42);
        }

        // Create random FP32 data in [-1, 1] range
        std::vector<float> createRandomFP32(int size, float scale = 1.0f)
        {
            std::uniform_real_distribution<float> dist(-scale, scale);
            std::vector<float> data(size);
            for (auto &v : data)
            {
                v = dist(gen_);
            }
            return data;
        }

        // Create random INT32 data (simulating attention context)
        std::vector<int32_t> createRandomINT32(int size, int32_t max_val = 100000)
        {
            std::uniform_int_distribution<int32_t> dist(-max_val, max_val);
            std::vector<int32_t> data(size);
            for (auto &v : data)
            {
                v = dist(gen_);
            }
            return data;
        }

        // Create identity-like weights for debugging (diagonal = 1, others = 0)
        std::vector<float> createIdentityWeights(int n, int k)
        {
            std::vector<float> weights(n * k, 0.0f);
            int diag_len = std::min(n, k);
            for (int i = 0; i < diag_len; ++i)
            {
                weights[i * k + i] = 1.0f;
            }
            return weights;
        }

        // Pack FP32 weights to QuantisedPackedWeights (Q8_1 format for VNNI)
        // Uses the production path: FP32 → Q8_1Tensor → VNNI packing
        std::pair<std::shared_ptr<Q8_1Tensor>, const gemm_v4::QuantisedPackedWeights *> packWeights(
            const std::vector<float> &weights, int N, int K)
        {
            // Step 1: Quantize FP32 to Q8_1Tensor
            std::vector<size_t> shape = {static_cast<size_t>(N), static_cast<size_t>(K)};
            auto q8_tensor = Q8_1Tensor::quantize_from_fp32(weights.data(), shape);

            // Step 2: Pack to VNNI format using KernelFactory
            // This returns a pointer to weights cached in the tensor
            const auto *packed = llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(q8_tensor.get());

            // Return both the tensor (to keep it alive) and the packed weights pointer
            return {q8_tensor, packed};
        }

        // Keep tensor alive for the duration of the test
        std::shared_ptr<Q8_1Tensor> wo_tensor_;

        // FP32 reference GEMV: output[N] = input[K] × W[N, K]^T
        std::vector<float> fp32ReferenceGEMV(
            const std::vector<float> &input,
            const std::vector<float> &weights,
            int N, int K)
        {
            std::vector<float> output(N, 0.0f);
            for (int n = 0; n < N; ++n)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    sum += input[k] * weights[n * K + k];
                }
                output[n] = sum;
            }
            return output;
        }

        // FP32 reference batched GEMM: output[B, N] = input[B, K] × W[N, K]^T
        std::vector<float> fp32ReferenceBatchedGEMM(
            const std::vector<float> &input,
            const std::vector<float> &weights,
            int B, int N, int K)
        {
            std::vector<float> output(B * N, 0.0f);
            for (int b = 0; b < B; ++b)
            {
                for (int n = 0; n < N; ++n)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k)
                    {
                        sum += input[b * K + k] * weights[n * K + k];
                    }
                    output[b * N + n] = sum;
                }
            }
            return output;
        }

        // Convert INT32 context to FP32 using scale
        std::vector<float> int32ToFP32(const std::vector<int32_t> &int32_data, float scale)
        {
            std::vector<float> fp32(int32_data.size());
            for (size_t i = 0; i < int32_data.size(); ++i)
            {
                fp32[i] = static_cast<float>(int32_data[i]) * scale;
            }
            return fp32;
        }

        // Compute cosine similarity
        float cosineSimilarity(const float *a, const float *b, int n)
        {
            double dot = 0, norm_a = 0, norm_b = 0;
            for (int i = 0; i < n; ++i)
            {
                dot += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }
            if (norm_a < 1e-10 || norm_b < 1e-10)
                return 0.0f;
            return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
        }

        // Compute MSE
        float computeMSE(const float *a, const float *b, int n)
        {
            double mse = 0;
            for (int i = 0; i < n; ++i)
            {
                double diff = a[i] - b[i];
                mse += diff * diff;
            }
            return static_cast<float>(mse / n);
        }

        std::mt19937 gen_;
    };

    // ============================================================================
    // Test: Context Normalization INT32 → INT16
    // ============================================================================

    TEST_F(Test__Q16WoProjection, ContextNormalization_RangePreservation)
    {
        // Test that INT32 context is normalized to INT16 range correctly
        const int d_model = 896;
        auto context_int32 = createRandomINT32(d_model, 1000000); // Large INT32 values

        std::vector<int16_t> context_int16(d_model);
        float scale = 0.0f;

        q16_context_normalize_to_int16(
            context_int32.data(),
            context_int16.data(),
            scale,
            d_model);

        // Check all values are in INT16 range
        for (int i = 0; i < d_model; ++i)
        {
            EXPECT_GE(context_int16[i], -32768);
            EXPECT_LE(context_int16[i], 32767);
        }

        // Check scale is positive
        EXPECT_GT(scale, 0.0f);

        // Check reconstruction: int32 ≈ int16 × scale
        for (int i = 0; i < d_model; ++i)
        {
            float reconstructed = static_cast<float>(context_int16[i]) * scale;
            float original = static_cast<float>(context_int32[i]);
            float rel_error = std::abs(reconstructed - original) / (std::abs(original) + 1e-6f);
            EXPECT_LT(rel_error, 0.01f) << "Reconstruction error at index " << i
                                        << ": original=" << original
                                        << " reconstructed=" << reconstructed;
        }
    }

    // ============================================================================
    // Test: Single Row GEMV (wo_gemv_row_vnni_int16)
    // ============================================================================

    TEST_F(Test__Q16WoProjection, SingleRowGEMV_SmallDimensions)
    {
        // Test single row GEMV with small dimensions for debugging
        // NOTE: This tests the RAW INT32 accumulator output from VNNI GEMV.
        // The weights are quantized to Q8_1 where 1.0 → ~127 (INT8 range).
        // Scales are applied LATER in wo_projection_vnni_int16().

        const int N = 64; // Output dimension
        const int K = 64; // Input dimension (must be multiple of 4 for VNNI)

        // Create simple weights: row 0 sums the first 32 elements
        std::vector<float> weights(N * K, 0.0f);
        for (int k = 0; k < 32; ++k)
        {
            weights[0 * K + k] = 1.0f; // First row: sum first 32 elements
        }
        for (int k = 0; k < K; ++k)
        {
            weights[1 * K + k] = 1.0f; // Second row: sum all elements
        }

        auto packed = packWeights(weights, N, K);

        // Create known INT16 context
        std::vector<int16_t> context_int16(K, 100); // All 100s

        // Test row 0: INT32 accumulator (before scale application)
        // Expected: 32 × 100 × ~127 (Q8 scale factor) ≈ 406,400
        int32_t result0 = wo_gemv_row_vnni_int16(
            context_int16.data(), packed.second, 0, K);

        // Test row 1: 64 × 100 × ~127 ≈ 812,800
        int32_t result1 = wo_gemv_row_vnni_int16(
            context_int16.data(), packed.second, 1, K);

        // The Q8_1 quantization for 1.0 is approximately 127 (full INT8 range)
        // Allow ±10% tolerance for quantization variance
        EXPECT_NEAR(result0, 32 * 100 * 127, 50000)
            << "Row 0 GEMV: raw INT32 accumulator incorrect";
        EXPECT_NEAR(result1, 64 * 100 * 127, 100000)
            << "Row 1 GEMV: raw INT32 accumulator incorrect";

        // More importantly, verify the ratio is correct (row1 / row0 ≈ 2.0)
        // This confirms the GEMV is computing correctly regardless of scale
        float ratio = static_cast<float>(result1) / static_cast<float>(result0);
        EXPECT_NEAR(ratio, 2.0f, 0.1f) << "Row ratio should be ~2.0 (64/32 weights)";
    }

    // ============================================================================
    // Test: Full Wo Projection (Decode Path)
    // ============================================================================

    TEST_F(Test__Q16WoProjection, DecodeWoProjection_IdentityWeights)
    {
        // Test decode Wo projection with identity-like weights
        // This should approximately preserve the input (for debugging)
        const int d_model = 128;
        const int input_dim = 128;
        const Q16BlockSize block_size = Q16BlockSize::BLOCK_64;

        // Create identity weights
        auto weights = createIdentityWeights(d_model, input_dim);
        auto packed = packWeights(weights, d_model, input_dim);

        // Create context that should pass through
        auto context_fp32 = createRandomFP32(input_dim, 10.0f);

        // Convert to INT32 with known scale
        const float input_scale = 0.001f; // Small scale so INT32 values are large
        std::vector<int32_t> context_int32(input_dim);
        for (int i = 0; i < input_dim; ++i)
        {
            context_int32[i] = static_cast<int32_t>(context_fp32[i] / input_scale);
        }

        // Allocate output
        const int num_blocks = (d_model + 63) / 64;
        std::vector<Q16_1Block_64> output_blocks(num_blocks);
        std::vector<float> snapshot_fp32(d_model);

        // Run Wo projection
        wo_projection_vnni_int16(
            context_int32.data(),
            input_scale, // context_scale
            packed.second,
            output_blocks.data(),
            d_model,
            input_dim,
            block_size,
            snapshot_fp32.data());

        // For identity weights, output ≈ input (within quantization error)
        float cosine = cosineSimilarity(context_fp32.data(), snapshot_fp32.data(), d_model);
        LOG_INFO("Identity weights decode: cosine=" << cosine);

        // Should be highly correlated (>0.9) for identity
        EXPECT_GT(cosine, 0.9f) << "Identity weight projection should preserve input direction";

        // Check output is non-zero
        float output_norm = 0;
        for (int i = 0; i < d_model; ++i)
        {
            output_norm += snapshot_fp32[i] * snapshot_fp32[i];
        }
        EXPECT_GT(output_norm, 0.0f) << "Output should be non-zero";
    }

    TEST_F(Test__Q16WoProjection, DecodeWoProjection_RandomWeights_VS_FP32)
    {
        // Test decode Wo projection against FP32 reference
        const int d_model = 896;
        const int input_dim = 896; // n_heads × head_dim = 14 × 64
        const Q16BlockSize block_size = Q16BlockSize::BLOCK_64;

        // Create random weights
        auto weights = createRandomFP32(d_model * input_dim, 0.1f);
        auto packed = packWeights(weights, d_model, input_dim);

        // Create random context (simulate attention output)
        auto context_fp32 = createRandomFP32(input_dim, 5.0f);

        // FP32 reference
        auto ref_output = fp32ReferenceGEMV(context_fp32, weights, d_model, input_dim);

        // Convert to INT32 for Q16 path
        const float input_scale = 0.0001f;
        std::vector<int32_t> context_int32(input_dim);
        for (int i = 0; i < input_dim; ++i)
        {
            context_int32[i] = static_cast<int32_t>(context_fp32[i] / input_scale);
        }

        // Allocate output
        const int num_blocks = (d_model + 63) / 64;
        std::vector<Q16_1Block_64> output_blocks(num_blocks);
        std::vector<float> q16_output(d_model);

        // Run Q16 Wo projection
        wo_projection_vnni_int16(
            context_int32.data(),
            input_scale,
            packed.second,
            output_blocks.data(),
            d_model,
            input_dim,
            block_size,
            q16_output.data());

        // Compare with FP32 reference
        float cosine = cosineSimilarity(ref_output.data(), q16_output.data(), d_model);
        float mse = computeMSE(ref_output.data(), q16_output.data(), d_model);

        LOG_INFO("Decode Wo Projection vs FP32 Reference:");
        LOG_INFO("  Cosine similarity: " << cosine);
        LOG_INFO("  MSE: " << mse);
        LOG_INFO("  Ref output[0:4]: " << ref_output[0] << ", " << ref_output[1]
                                       << ", " << ref_output[2] << ", " << ref_output[3]);
        LOG_INFO("  Q16 output[0:4]: " << q16_output[0] << ", " << q16_output[1]
                                       << ", " << q16_output[2] << ", " << q16_output[3]);

        // Expect high correlation (>0.95) with quantized weights
        EXPECT_GT(cosine, 0.95f) << "Q16 Wo projection should match FP32 reference";

        // Ensure output is non-trivial
        float ref_norm = 0, q16_norm = 0;
        for (int i = 0; i < d_model; ++i)
        {
            ref_norm += ref_output[i] * ref_output[i];
            q16_norm += q16_output[i] * q16_output[i];
        }
        EXPECT_GT(ref_norm, 0.0f) << "Reference output should be non-zero";
        EXPECT_GT(q16_norm, 0.0f) << "Q16 output should be non-zero";
    }

    // ============================================================================
    // Test: Batched Wo Projection (Prefill Path)
    // ============================================================================

    TEST_F(Test__Q16WoProjection, PrefillWoProjection_BatchedVS_FP32)
    {
        // Test batched Wo projection for prefill against FP32 reference
        const int batch_size = 9; // Typical prefill (like test prompt)
        const int d_model = 896;
        const int input_dim = 896;
        const Q16BlockSize block_size = Q16BlockSize::BLOCK_64;

        // Create random weights
        auto weights = createRandomFP32(d_model * input_dim, 0.1f);
        auto packed = packWeights(weights, d_model, input_dim);

        // Create random batched context
        auto context_fp32 = createRandomFP32(batch_size * input_dim, 5.0f);

        // FP32 reference
        auto ref_output = fp32ReferenceBatchedGEMM(context_fp32, weights, batch_size, d_model, input_dim);

        // Convert to INT32
        const float input_scale = 0.0001f;
        std::vector<int32_t> context_int32(batch_size * input_dim);
        std::vector<float> context_scales(batch_size, input_scale);
        for (int i = 0; i < batch_size * input_dim; ++i)
        {
            context_int32[i] = static_cast<int32_t>(context_fp32[i] / input_scale);
        }

        // Allocate output
        const int blocks_per_row = (d_model + 63) / 64;
        std::vector<Q16_1Block_64> output_blocks(batch_size * blocks_per_row);
        std::vector<float> q16_output(batch_size * d_model);

        // Run batched Q16 Wo projection
        wo_projection_vnni_int16_batched(
            context_int32.data(),
            context_scales.data(),
            packed.second,
            output_blocks.data(),
            batch_size,
            d_model,
            input_dim,
            input_dim,      // context_stride
            blocks_per_row, // output_stride
            block_size,
            q16_output.data());

        // Compare each batch element
        float avg_cosine = 0;
        for (int b = 0; b < batch_size; ++b)
        {
            float cosine = cosineSimilarity(
                ref_output.data() + b * d_model,
                q16_output.data() + b * d_model,
                d_model);
            avg_cosine += cosine;

            EXPECT_GT(cosine, 0.9f) << "Batch " << b << " cosine too low: " << cosine;
        }
        avg_cosine /= batch_size;

        LOG_INFO("Prefill Batched Wo Projection vs FP32 Reference:");
        LOG_INFO("  Average cosine: " << avg_cosine);

        EXPECT_GT(avg_cosine, 0.95f) << "Average cosine should be >0.95";
    }

    // ============================================================================
    // Test: Q16_1 Block Output Format
    // ============================================================================

    TEST_F(Test__Q16WoProjection, OutputBlockFormat_ValidQ16_1)
    {
        // Verify output is valid Q16_1 format
        const int d_model = 128;
        const int input_dim = 128;
        const Q16BlockSize block_size = Q16BlockSize::BLOCK_64;

        auto weights = createRandomFP32(d_model * input_dim, 0.1f);
        auto packed = packWeights(weights, d_model, input_dim);

        auto context_fp32 = createRandomFP32(input_dim, 5.0f);
        const float input_scale = 0.0001f;
        std::vector<int32_t> context_int32(input_dim);
        for (int i = 0; i < input_dim; ++i)
        {
            context_int32[i] = static_cast<int32_t>(context_fp32[i] / input_scale);
        }

        const int num_blocks = (d_model + 63) / 64;
        std::vector<Q16_1Block_64> output_blocks(num_blocks);

        wo_projection_vnni_int16(
            context_int32.data(),
            input_scale,
            packed.second,
            output_blocks.data(),
            d_model,
            input_dim,
            block_size,
            nullptr);

        // Verify Q16_1 block structure
        for (int b = 0; b < num_blocks; ++b)
        {
            // Scale should be finite and reasonable
            EXPECT_TRUE(std::isfinite(output_blocks[b].d))
                << "Block " << b << " scale is not finite";
            EXPECT_GT(std::abs(output_blocks[b].d), 1e-10f)
                << "Block " << b << " scale is too small";

            // At least some values should be non-zero
            bool has_nonzero = false;
            for (int i = 0; i < 64; ++i)
            {
                if (output_blocks[b].qs[i] != 0)
                {
                    has_nonzero = true;
                    break;
                }
            }
            EXPECT_TRUE(has_nonzero) << "Block " << b << " is all zeros";
        }
    }

    // ============================================================================
    // Test: Qwen2.5-0.5B Dimensions (Real Model Config)
    // ============================================================================

    TEST_F(Test__Q16WoProjection, Qwen2_5_0_5B_Dimensions)
    {
        // Test with exact Qwen2.5-0.5B dimensions
        const int n_heads = 14;
        const int head_dim = 64;
        const int d_model = n_heads * head_dim; // 896
        const int input_dim = d_model;          // Context is [n_heads × head_dim]
        const Q16BlockSize block_size = Q16BlockSize::BLOCK_64;

        // Create random weights
        auto weights = createRandomFP32(d_model * input_dim, 0.05f);
        auto packed = packWeights(weights, d_model, input_dim);

        // Create context simulating attention output
        // In real attention: context is INT32 from softmax-weighted V accumulation
        auto context_fp32 = createRandomFP32(input_dim, 1.0f); // Typical attention output range

        // FP32 reference
        auto ref_output = fp32ReferenceGEMV(context_fp32, weights, d_model, input_dim);

        // Convert to INT32 (simulate pv_scale normalization)
        const float pv_scale = 0.001f; // Typical from softmax
        std::vector<int32_t> context_int32(input_dim);
        for (int i = 0; i < input_dim; ++i)
        {
            context_int32[i] = static_cast<int32_t>(context_fp32[i] / pv_scale);
        }

        // Run Q16 path
        const int num_blocks = (d_model + 63) / 64; // 14 blocks
        std::vector<Q16_1Block_64> output_blocks(num_blocks);
        std::vector<float> q16_output(d_model);

        wo_projection_vnni_int16(
            context_int32.data(),
            pv_scale,
            packed.second,
            output_blocks.data(),
            d_model,
            input_dim,
            block_size,
            q16_output.data());

        // Compare
        float cosine = cosineSimilarity(ref_output.data(), q16_output.data(), d_model);
        LOG_INFO("Qwen2.5-0.5B Wo Projection: cosine=" << cosine);

        EXPECT_GT(cosine, 0.95f) << "Should match FP32 reference for real model dims";
    }

} // namespace llaminar2::kernels::q16_1::microkernels
