/**
 * @file Test__Q16AttentionOutput_Composition.cpp
 * @brief Macrokernel composition test: Context → Wo Projection → Residual Add → Q16_1 Output
 *
 * This test validates the FULL attention output path by composing:
 *   1. Attention context (simulated INT32 from softmax×V)
 *   2. Wo projection: context × Wo → FP32 accumulator → Q16_1
 *   3. Residual add: Q16_1 + Q16_1 → Q16_1
 *
 * This is the "Level 2" macrokernel test that bridges between:
 *   - Level 1: Individual microkernel tests (WoProjection, ResidualAdd)
 *   - Level 3: Full integration tests (HybridQ16Pipeline_Integration)
 *
 * The test exposes issues like:
 *   - Context concatenation from heads [n_heads][head_dim] → [d_model]
 *   - Scale propagation through the pipeline
 *   - Numerical precision through quantization boundaries
 *
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md
 * @see docs/v2/PLAN_FIXED_SCALE_ROPE_Q16.md "Addendum 2"
 */

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include <numeric>

// Microkernels under test
#include "kernels/cpu/attention/q16_1/ref/microkernels/WoProjection.h"
#include "tensors/SIMDHelpers.h"

// Supporting infrastructure
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "kernels/KernelFactory.h"
#include "utils/Logger.h"

namespace llaminar2::test
{

    using namespace llaminar2::kernels::q16_1::microkernels;
    using namespace llaminar2::simd;
    using namespace llaminar2::gemm_v4;

    /**
     * @brief Test fixture for Q16 attention output composition.
     *
     * Validates the full path from attention context to final Q16_1 output
     * with residual connection.
     */
    class Test__Q16AttentionOutput_Composition : public ::testing::Test
    {
    protected:
        // Qwen2.5-0.5B dimensions
        static constexpr int D_MODEL = 896;
        static constexpr int NUM_HEADS = 14;
        static constexpr int HEAD_DIM = 64;
        static constexpr int INPUT_DIM = NUM_HEADS * HEAD_DIM; // 896

        std::mt19937 gen_{42}; // Fixed seed for reproducibility

        void SetUp() override
        {
            // Verify dimensions
            ASSERT_EQ(INPUT_DIM, D_MODEL) << "For Qwen2.5, input_dim == d_model";
        }

        // ========================================================================
        // Helper: Create random FP32 data
        // ========================================================================
        std::vector<float> createRandomFP32(int size, float range = 1.0f)
        {
            std::uniform_real_distribution<float> dist(-range, range);
            std::vector<float> data(size);
            for (auto &v : data)
            {
                v = dist(gen_);
            }
            return data;
        }

        // ========================================================================
        // Helper: Create random INT32 context (simulating softmax×V output)
        // ========================================================================
        std::vector<int32_t> createRandomINT32Context(int size, int32_t max_val = 100000)
        {
            std::uniform_int_distribution<int32_t> dist(-max_val, max_val);
            std::vector<int32_t> data(size);
            for (auto &v : data)
            {
                v = dist(gen_);
            }
            return data;
        }

        // ========================================================================
        // Helper: Create Q16_1 tensor from FP32 (for residual)
        // ========================================================================
        std::unique_ptr<Q16_1Tensor> createQ16_1FromFP32(
            const std::vector<float> &fp32_data,
            Q16BlockSize block_size)
        {
            auto tensor = std::make_unique<Q16_1Tensor>(
                std::vector<size_t>{1, static_cast<size_t>(fp32_data.size())},
                block_size, 0);
            tensor->copyFrom_fp32(fp32_data.data());
            return tensor;
        }

        // ========================================================================
        // Helper: Pack FP32 weights to VNNI format via Q8_1Tensor
        // ========================================================================
        std::pair<std::shared_ptr<Q8_1Tensor>, const QuantisedPackedWeights *> packWeights(
            const std::vector<float> &weights, int N, int K)
        {
            std::vector<size_t> shape = {static_cast<size_t>(N), static_cast<size_t>(K)};
            auto q8_tensor = Q8_1Tensor::quantize_from_fp32(weights.data(), shape);
            const auto *packed = llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(q8_tensor.get());
            return {q8_tensor, packed};
        }

        // ========================================================================
        // Helper: FP32 reference for full pipeline
        // ========================================================================
        std::vector<float> fp32ReferencePipeline(
            const std::vector<float> &context_fp32,
            const std::vector<float> &wo_weights,
            const std::vector<float> &residual_fp32,
            int d_model, int input_dim)
        {
            // Step 1: Wo projection (GEMV)
            std::vector<float> wo_output(d_model, 0.0f);
            for (int n = 0; n < d_model; ++n)
            {
                float sum = 0.0f;
                for (int k = 0; k < input_dim; ++k)
                {
                    sum += context_fp32[k] * wo_weights[n * input_dim + k];
                }
                wo_output[n] = sum;
            }

            // Step 2: Residual add
            std::vector<float> output(d_model);
            for (int i = 0; i < d_model; ++i)
            {
                output[i] = wo_output[i] + residual_fp32[i];
            }

            return output;
        }

        // ========================================================================
        // Helper: Compute cosine similarity
        // ========================================================================
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

        // ========================================================================
        // Helper: Compute MSE
        // ========================================================================
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

        // ========================================================================
        // Helper: Dequantize Q16_1 blocks to FP32
        // ========================================================================
        std::vector<float> dequantizeQ16_1(const Q16_1Block_64 *blocks, int num_blocks)
        {
            std::vector<float> fp32(num_blocks * 64);
            for (int b = 0; b < num_blocks; ++b)
            {
                float scale = blocks[b].d;
                for (int i = 0; i < 64; ++i)
                {
                    fp32[b * 64 + i] = blocks[b].qs[i] * scale;
                }
            }
            return fp32;
        }

        std::vector<float> dequantizeQ16_1(const Q16_1Block *blocks, int num_elements)
        {
            int num_blocks = (num_elements + 31) / 32;
            std::vector<float> fp32(num_blocks * 32);
            for (int b = 0; b < num_blocks; ++b)
            {
                float scale = blocks[b].d;
                for (int i = 0; i < 32; ++i)
                {
                    fp32[b * 32 + i] = blocks[b].qs[i] * scale;
                }
            }
            // Trim to exact size
            fp32.resize(num_elements);
            return fp32;
        }
    };

    // ============================================================================
    // TEST: Wo Projection Only (Decode) - Baseline
    // ============================================================================
    TEST_F(Test__Q16AttentionOutput_Composition, WoProjectionOnly_Decode_VS_FP32)
    {
        // This test validates just the Wo projection step
        // Context → Wo × context → Q16_1 output

        const Q16BlockSize block_size = Q16BlockSize::BLOCK_64;
        const int num_output_blocks = (D_MODEL + 63) / 64;

        // Create random context (FP32, then convert to INT32)
        auto context_fp32 = createRandomFP32(INPUT_DIM, 5.0f);

        // Scale for INT32 representation
        const float context_scale = 0.0001f;
        std::vector<int32_t> context_int32(INPUT_DIM);
        for (int i = 0; i < INPUT_DIM; ++i)
        {
            context_int32[i] = static_cast<int32_t>(context_fp32[i] / context_scale);
        }

        // Create random Wo weights and pack
        auto wo_fp32 = createRandomFP32(D_MODEL * INPUT_DIM, 0.1f);
        auto [wo_tensor, wo_packed] = packWeights(wo_fp32, D_MODEL, INPUT_DIM);

        // FP32 reference: output = context × Wo^T
        std::vector<float> ref_output(D_MODEL, 0.0f);
        for (int n = 0; n < D_MODEL; ++n)
        {
            float sum = 0.0f;
            for (int k = 0; k < INPUT_DIM; ++k)
            {
                sum += context_fp32[k] * wo_fp32[n * INPUT_DIM + k];
            }
            ref_output[n] = sum;
        }

        // Q16 path: Wo projection using VNNI microkernel
        std::vector<Q16_1Block_64> q16_output_blocks(num_output_blocks);
        std::vector<float> q16_dequant(D_MODEL);

        wo_projection_vnni_int16(
            context_int32.data(),
            context_scale,
            wo_packed,
            q16_output_blocks.data(),
            D_MODEL,
            INPUT_DIM,
            block_size,
            q16_dequant.data());

        // Compare
        float cosine = cosineSimilarity(ref_output.data(), q16_dequant.data(), D_MODEL);
        float mse = computeMSE(ref_output.data(), q16_dequant.data(), D_MODEL);

        LOG_INFO("Wo Projection Only (Decode) vs FP32:");
        LOG_INFO("  Cosine similarity: " << cosine);
        LOG_INFO("  MSE: " << mse);
        LOG_INFO("  Ref output[0:4]: " << ref_output[0] << ", " << ref_output[1]
                                       << ", " << ref_output[2] << ", " << ref_output[3]);
        LOG_INFO("  Q16 output[0:4]: " << q16_dequant[0] << ", " << q16_dequant[1]
                                       << ", " << q16_dequant[2] << ", " << q16_dequant[3]);

        EXPECT_GT(cosine, 0.95f) << "Wo projection should match FP32 reference";
        EXPECT_GT(std::sqrt(std::inner_product(ref_output.begin(), ref_output.end(),
                                               ref_output.begin(), 0.0f)),
                  0.0f)
            << "Reference output should be non-zero";
    }

    // ============================================================================
    // TEST: Full Composition - Wo Projection + Residual Add (Decode)
    // ============================================================================
    TEST_F(Test__Q16AttentionOutput_Composition, FullComposition_Decode_VS_FP32)
    {
        // This is the CRITICAL test that validates the full path:
        // Context → Wo projection → Residual Add → Final Q16_1 Output

        const Q16BlockSize block_size = Q16BlockSize::BLOCK_64;
        const int num_output_blocks = (D_MODEL + 63) / 64;
        const int num_residual_blocks = (D_MODEL + 31) / 32; // Q16_1Block uses 32 elements

        // ========================================================================
        // Step 1: Create inputs
        // ========================================================================

        // Context (simulating attention output)
        auto context_fp32 = createRandomFP32(INPUT_DIM, 5.0f);
        const float context_scale = 0.0001f;
        std::vector<int32_t> context_int32(INPUT_DIM);
        for (int i = 0; i < INPUT_DIM; ++i)
        {
            context_int32[i] = static_cast<int32_t>(context_fp32[i] / context_scale);
        }

        // Wo weights
        auto wo_fp32 = createRandomFP32(D_MODEL * INPUT_DIM, 0.1f);
        auto [wo_tensor, wo_packed] = packWeights(wo_fp32, D_MODEL, INPUT_DIM);

        // Residual (Q16_1 from previous layer)
        auto residual_fp32 = createRandomFP32(D_MODEL, 2.0f);
        auto residual_tensor = createQ16_1FromFP32(residual_fp32, Q16BlockSize::BLOCK_32);

        // ========================================================================
        // Step 2: FP32 Reference Pipeline
        // ========================================================================
        auto ref_output = fp32ReferencePipeline(
            context_fp32, wo_fp32, residual_fp32, D_MODEL, INPUT_DIM);

        // ========================================================================
        // Step 3: Q16 Pipeline - Wo Projection
        // ========================================================================
        std::vector<Q16_1Block_64> wo_output_blocks(num_output_blocks);
        std::vector<float> wo_dequant(D_MODEL);

        wo_projection_vnni_int16(
            context_int32.data(),
            context_scale,
            wo_packed,
            wo_output_blocks.data(),
            D_MODEL,
            INPUT_DIM,
            block_size,
            wo_dequant.data());

        // ========================================================================
        // Step 4: Q16 Pipeline - Requantize Wo output to Q16_1Block (32-element)
        // ========================================================================
        // The Wo output is in Q16_1Block_64, but residual add uses Q16_1Block (32)
        // We need to convert/requantize

        std::vector<Q16_1Block> wo_q16_blocks(num_residual_blocks);
        for (int b = 0; b < num_residual_blocks; ++b)
        {
            const int start = b * 32;
            const int end = std::min(start + 32, D_MODEL);

            // Find max for this block
            float max_abs = 0.0f;
            for (int i = start; i < end; ++i)
            {
                max_abs = std::max(max_abs, std::abs(wo_dequant[i]));
            }

            // Compute scale
            if (max_abs < 1e-10f)
            {
                wo_q16_blocks[b].d = 0.0f;
                wo_q16_blocks[b].sum_qs = 0;
                std::memset(wo_q16_blocks[b].qs, 0, 64); // 32 × int16
            }
            else
            {
                float scale = max_abs / 32767.0f;
                wo_q16_blocks[b].d = scale;

                int32_t sum_qs = 0;
                for (int i = start; i < end; ++i)
                {
                    int16_t q = static_cast<int16_t>(std::clamp(
                        std::round(wo_dequant[i] / scale), -32767.0f, 32767.0f));
                    wo_q16_blocks[b].qs[i - start] = q;
                    sum_qs += q;
                }
                wo_q16_blocks[b].sum_qs = sum_qs;
            }
        }

        // ========================================================================
        // Step 5: Q16 Pipeline - Residual Add (Q16_1 + Q16_1 → Q16_1)
        // ========================================================================
        std::vector<Q16_1Block> final_output_blocks(num_residual_blocks);

        q16_1_add_q16_1(
            wo_q16_blocks.data(),
            residual_tensor->typed_data(),
            final_output_blocks.data(),
            D_MODEL);

        // ========================================================================
        // Step 6: Dequantize final output and compare
        // ========================================================================
        auto q16_final = dequantizeQ16_1(final_output_blocks.data(), D_MODEL);

        float cosine = cosineSimilarity(ref_output.data(), q16_final.data(), D_MODEL);
        float mse = computeMSE(ref_output.data(), q16_final.data(), D_MODEL);

        // Also check intermediate Wo output
        float wo_cosine = cosineSimilarity(
            [&]() -> std::vector<float>
                     {
                         std::vector<float> wo_ref(D_MODEL, 0.0f);
                         for (int n = 0; n < D_MODEL; ++n)
                         {
                             for (int k = 0; k < INPUT_DIM; ++k)
                             {
                                 wo_ref[n] += context_fp32[k] * wo_fp32[n * INPUT_DIM + k];
                             }
                         }
                         return wo_ref;
                     }()
                         .data(),
            wo_dequant.data(),
            D_MODEL);

        LOG_INFO("Full Composition (Wo + Residual) vs FP32:");
        LOG_INFO("  Intermediate Wo cosine: " << wo_cosine);
        LOG_INFO("  Final output cosine: " << cosine);
        LOG_INFO("  Final output MSE: " << mse);
        LOG_INFO("  Ref output[0:4]: " << ref_output[0] << ", " << ref_output[1]
                                       << ", " << ref_output[2] << ", " << ref_output[3]);
        LOG_INFO("  Q16 output[0:4]: " << q16_final[0] << ", " << q16_final[1]
                                       << ", " << q16_final[2] << ", " << q16_final[3]);

        // The composition adds quantization error, but should still be >0.9
        EXPECT_GT(cosine, 0.90f) << "Full composition should match FP32 reference";
        EXPECT_GT(wo_cosine, 0.95f) << "Wo projection stage should be accurate";
    }

    // ============================================================================
    // TEST: Residual Dominance - Verify Residual Not Zeroed Out
    // ============================================================================
    TEST_F(Test__Q16AttentionOutput_Composition, ResidualDominance_OutputNotZero)
    {
        // Edge case: If Wo output is zero, residual should pass through
        // This catches bugs where residual add is not implemented

        const int num_residual_blocks = (D_MODEL + 31) / 32;

        // Create zero Wo output
        std::vector<Q16_1Block> wo_output_blocks(num_residual_blocks);
        for (auto &block : wo_output_blocks)
        {
            block.d = 0.0f;
            block.sum_qs = 0;
            std::memset(block.qs, 0, 64);
        }

        // Create non-zero residual
        auto residual_fp32 = createRandomFP32(D_MODEL, 2.0f);
        auto residual_tensor = createQ16_1FromFP32(residual_fp32, Q16BlockSize::BLOCK_32);

        // Add: 0 + residual = residual
        std::vector<Q16_1Block> final_output_blocks(num_residual_blocks);
        q16_1_add_q16_1(
            wo_output_blocks.data(),
            residual_tensor->typed_data(),
            final_output_blocks.data(),
            D_MODEL);

        // Dequantize and check
        auto q16_final = dequantizeQ16_1(final_output_blocks.data(), D_MODEL);

        // Should match residual (with quantization error)
        float cosine = cosineSimilarity(residual_fp32.data(), q16_final.data(), D_MODEL);

        LOG_INFO("Residual Dominance Test (zero Wo):");
        LOG_INFO("  Cosine with residual: " << cosine);
        LOG_INFO("  Residual[0:4]: " << residual_fp32[0] << ", " << residual_fp32[1]
                                     << ", " << residual_fp32[2] << ", " << residual_fp32[3]);
        LOG_INFO("  Output[0:4]: " << q16_final[0] << ", " << q16_final[1]
                                   << ", " << q16_final[2] << ", " << q16_final[3]);

        EXPECT_GT(cosine, 0.99f) << "When Wo=0, output should equal residual";

        // Verify output is actually non-zero
        float output_norm = 0.0f;
        for (float v : q16_final)
        {
            output_norm += v * v;
        }
        EXPECT_GT(output_norm, 0.0f) << "Output should not be all zeros";
    }

    // ============================================================================
    // TEST: Scale Propagation Through Pipeline
    // ============================================================================
    TEST_F(Test__Q16AttentionOutput_Composition, ScalePropagation_LargeContextValues)
    {
        // Test with large context values to stress scale handling
        // This catches overflow/underflow issues in the pipeline

        const Q16BlockSize block_size = Q16BlockSize::BLOCK_64;
        const int num_output_blocks = (D_MODEL + 63) / 64;
        const int num_residual_blocks = (D_MODEL + 31) / 32;

        // Large context values (typical after softmax×V accumulation)
        auto context_fp32 = createRandomFP32(INPUT_DIM, 50.0f); // 10× larger
        const float context_scale = 0.00001f;                   // Compensate with smaller scale
        std::vector<int32_t> context_int32(INPUT_DIM);
        for (int i = 0; i < INPUT_DIM; ++i)
        {
            context_int32[i] = static_cast<int32_t>(context_fp32[i] / context_scale);
        }

        // Wo weights
        auto wo_fp32 = createRandomFP32(D_MODEL * INPUT_DIM, 0.1f);
        auto [wo_tensor, wo_packed] = packWeights(wo_fp32, D_MODEL, INPUT_DIM);

        // Residual
        auto residual_fp32 = createRandomFP32(D_MODEL, 10.0f); // Larger residual
        auto residual_tensor = createQ16_1FromFP32(residual_fp32, Q16BlockSize::BLOCK_32);

        // FP32 reference
        auto ref_output = fp32ReferencePipeline(
            context_fp32, wo_fp32, residual_fp32, D_MODEL, INPUT_DIM);

        // Q16 path
        std::vector<Q16_1Block_64> wo_output_blocks(num_output_blocks);
        std::vector<float> wo_dequant(D_MODEL);

        wo_projection_vnni_int16(
            context_int32.data(),
            context_scale,
            wo_packed,
            wo_output_blocks.data(),
            D_MODEL,
            INPUT_DIM,
            block_size,
            wo_dequant.data());

        // Requantize and add residual
        std::vector<Q16_1Block> wo_q16_blocks(num_residual_blocks);
        for (int b = 0; b < num_residual_blocks; ++b)
        {
            const int start = b * 32;
            const int end = std::min(start + 32, D_MODEL);

            float max_abs = 0.0f;
            for (int i = start; i < end; ++i)
            {
                max_abs = std::max(max_abs, std::abs(wo_dequant[i]));
            }

            if (max_abs < 1e-10f)
            {
                wo_q16_blocks[b].d = 0.0f;
                wo_q16_blocks[b].sum_qs = 0;
                std::memset(wo_q16_blocks[b].qs, 0, 64);
            }
            else
            {
                float scale = max_abs / 32767.0f;
                wo_q16_blocks[b].d = scale;

                int32_t sum_qs = 0;
                for (int i = start; i < end; ++i)
                {
                    int16_t q = static_cast<int16_t>(std::clamp(
                        std::round(wo_dequant[i] / scale), -32767.0f, 32767.0f));
                    wo_q16_blocks[b].qs[i - start] = q;
                    sum_qs += q;
                }
                wo_q16_blocks[b].sum_qs = sum_qs;
            }
        }

        std::vector<Q16_1Block> final_output_blocks(num_residual_blocks);
        q16_1_add_q16_1(
            wo_q16_blocks.data(),
            residual_tensor->typed_data(),
            final_output_blocks.data(),
            D_MODEL);

        auto q16_final = dequantizeQ16_1(final_output_blocks.data(), D_MODEL);
        float cosine = cosineSimilarity(ref_output.data(), q16_final.data(), D_MODEL);

        LOG_INFO("Scale Propagation Test (large values):");
        LOG_INFO("  Final cosine: " << cosine);
        LOG_INFO("  Ref range: [" << *std::min_element(ref_output.begin(), ref_output.end())
                                  << ", " << *std::max_element(ref_output.begin(), ref_output.end()) << "]");
        LOG_INFO("  Q16 range: [" << *std::min_element(q16_final.begin(), q16_final.end())
                                  << ", " << *std::max_element(q16_final.begin(), q16_final.end()) << "]");

        EXPECT_GT(cosine, 0.85f) << "Large values should still have reasonable correlation";
    }

    // ============================================================================
    // TEST: Batched Prefill Composition (Multiple Queries)
    // ============================================================================
    TEST_F(Test__Q16AttentionOutput_Composition, BatchedPrefill_Composition_VS_FP32)
    {
        // Test batched prefill: multiple queries through Wo + residual
        const int BATCH_SIZE = 8;
        const Q16BlockSize block_size = Q16BlockSize::BLOCK_64;
        const int num_output_blocks = (D_MODEL + 63) / 64;
        const int num_residual_blocks = (D_MODEL + 31) / 32;

        // Batched context
        auto context_fp32 = createRandomFP32(BATCH_SIZE * INPUT_DIM, 5.0f);
        const float context_scale = 0.0001f;
        std::vector<int32_t> context_int32(BATCH_SIZE * INPUT_DIM);
        for (int i = 0; i < BATCH_SIZE * INPUT_DIM; ++i)
        {
            context_int32[i] = static_cast<int32_t>(context_fp32[i] / context_scale);
        }

        // Wo weights (shared across batch)
        auto wo_fp32 = createRandomFP32(D_MODEL * INPUT_DIM, 0.1f);
        auto [wo_tensor, wo_packed] = packWeights(wo_fp32, D_MODEL, INPUT_DIM);

        // Batched residual
        auto residual_fp32 = createRandomFP32(BATCH_SIZE * D_MODEL, 2.0f);

        float total_cosine = 0.0f;
        int num_good = 0;

        for (int b = 0; b < BATCH_SIZE; ++b)
        {
            // FP32 reference for this query
            std::vector<float> ctx_b(context_fp32.begin() + b * INPUT_DIM,
                                     context_fp32.begin() + (b + 1) * INPUT_DIM);
            std::vector<float> res_b(residual_fp32.begin() + b * D_MODEL,
                                     residual_fp32.begin() + (b + 1) * D_MODEL);
            auto ref_output = fp32ReferencePipeline(ctx_b, wo_fp32, res_b, D_MODEL, INPUT_DIM);

            // Q16 path for this query
            std::vector<Q16_1Block_64> wo_output_blocks(num_output_blocks);
            std::vector<float> wo_dequant(D_MODEL);

            wo_projection_vnni_int16(
                context_int32.data() + b * INPUT_DIM,
                context_scale,
                wo_packed,
                wo_output_blocks.data(),
                D_MODEL,
                INPUT_DIM,
                block_size,
                wo_dequant.data());

            // Requantize
            std::vector<Q16_1Block> wo_q16_blocks(num_residual_blocks);
            for (int blk = 0; blk < num_residual_blocks; ++blk)
            {
                const int start = blk * 32;
                const int end = std::min(start + 32, D_MODEL);

                float max_abs = 0.0f;
                for (int i = start; i < end; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(wo_dequant[i]));
                }

                if (max_abs < 1e-10f)
                {
                    wo_q16_blocks[blk].d = 0.0f;
                    wo_q16_blocks[blk].sum_qs = 0;
                    std::memset(wo_q16_blocks[blk].qs, 0, 64);
                }
                else
                {
                    float scale = max_abs / 32767.0f;
                    wo_q16_blocks[blk].d = scale;

                    int32_t sum_qs = 0;
                    for (int i = start; i < end; ++i)
                    {
                        int16_t q = static_cast<int16_t>(std::clamp(
                            std::round(wo_dequant[i] / scale), -32767.0f, 32767.0f));
                        wo_q16_blocks[blk].qs[i - start] = q;
                        sum_qs += q;
                    }
                    wo_q16_blocks[blk].sum_qs = sum_qs;
                }
            }

            // Create residual tensor for this query
            auto residual_tensor = createQ16_1FromFP32(res_b, Q16BlockSize::BLOCK_32);

            // Residual add
            std::vector<Q16_1Block> final_output_blocks(num_residual_blocks);
            q16_1_add_q16_1(
                wo_q16_blocks.data(),
                residual_tensor->typed_data(),
                final_output_blocks.data(),
                D_MODEL);

            auto q16_final = dequantizeQ16_1(final_output_blocks.data(), D_MODEL);
            float cosine = cosineSimilarity(ref_output.data(), q16_final.data(), D_MODEL);

            total_cosine += cosine;
            if (cosine > 0.85f)
                ++num_good;
        }

        float avg_cosine = total_cosine / BATCH_SIZE;
        LOG_INFO("Batched Prefill Composition:");
        LOG_INFO("  Average cosine: " << avg_cosine);
        LOG_INFO("  Queries with cosine > 0.85: " << num_good << "/" << BATCH_SIZE);

        EXPECT_GT(avg_cosine, 0.88f) << "Average cosine should be high";
        EXPECT_GE(num_good, BATCH_SIZE - 1) << "Most queries should have good correlation";
    }

} // namespace llaminar2::test
