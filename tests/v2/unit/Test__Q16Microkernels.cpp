/**
 * @file Test__Q16Microkernels.cpp
 * @brief Unit tests for Q16_1 fused attention microkernels (PURE INTEGER PIPELINE)
 *
 * Tests each microkernel individually:
 * - Q16DotProductRef: Q×K^T dot products → INT32
 * - Exp2FixedSoftmaxRef: exp2-based softmax → INT16 weights (tested in Test__Exp2FixedSoftmaxRef.cpp)
 * - PVAccumulateRef: P×V weighted accumulation → INT32/FP32 (inc. online softmax utilities)
 * - WoProjectionVNNIRef: Wo projection with VPDPWSSD (INT16×INT16→Q16_1)
 *
 * Note: Residual add is handled by simd::q16_1_add_q16_1() in the pure integer path.
 * Index16Softmax was replaced by Exp2FixedSoftmaxRef for the pure integer pipeline.
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>

#include "kernels/cpu/attention/q16_1/ref/microkernels/Q16DotProductRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/PVAccumulateRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/WoProjectionVNNIRef.h"
#include "tensors/BlockStructures.h"
#include "tensors/Tensors.h"
#include "kernels/KernelFactory.h"

using namespace llaminar2;
using namespace llaminar2::kernels::q16_1::microkernels;

// ============================================================================
// Test Utilities
// ============================================================================

class Q16MicrokernelTest : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

    // Create random Q16_1 blocks
    std::vector<Q16_1Block> createRandomQ16Blocks(int num_blocks, float scale_range = 0.1f)
    {
        std::vector<Q16_1Block> blocks(num_blocks);
        std::uniform_int_distribution<int16_t> int_dist(-1000, 1000);
        std::uniform_real_distribution<float> scale_dist(0.001f, scale_range);

        for (auto &block : blocks)
        {
            block.d = scale_dist(rng_);
            int32_t sum = 0;
            for (int i = 0; i < 32; ++i)
            {
                block.qs[i] = int_dist(rng_);
                sum += block.qs[i];
            }
            block.sum_qs = sum;
        }
        return blocks;
    }

    // Create random Q8_0 blocks
    std::vector<Q8_0Block> createRandomQ8Blocks(int num_blocks, float scale_range = 0.1f)
    {
        std::vector<Q8_0Block> blocks(num_blocks);
        std::uniform_int_distribution<int8_t> int_dist(-100, 100);
        std::uniform_real_distribution<float> scale_dist(0.001f, scale_range);

        for (auto &block : blocks)
        {
            // Convert float to FP16 (simplified)
            float scale = scale_dist(rng_);
            uint32_t f;
            std::memcpy(&f, &scale, sizeof(f));
            uint32_t sign = (f >> 16) & 0x8000;
            int exp = ((f >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = (f >> 13) & 0x3FF;
            if (exp <= 0)
            {
                exp = 0;
                mant = 0;
            }
            if (exp >= 31)
            {
                exp = 31;
                mant = 0;
            }
            block.d = static_cast<uint16_t>(sign | (exp << 10) | mant);

            for (int i = 0; i < 32; ++i)
            {
                block.qs[i] = int_dist(rng_);
            }
        }
        return blocks;
    }

    // Compute cosine similarity
    float cosineSimilarity(const float *a, const float *b, int n)
    {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a < 1e-10f || norm_b < 1e-10f)
            return 0.0f;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    // Mean squared error
    float mse(const float *a, const float *b, int n)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum / n;
    }
};

// ============================================================================
// Q16DotProductRef Tests
// ============================================================================

TEST_F(Q16MicrokernelTest, DotProductSingle_BasicCorrectness)
{
    // Create simple test case with known values
    constexpr int HEAD_DIM = 64;
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCKS_PER_HEAD = HEAD_DIM / BLOCK_SIZE;

    // Q: [1, HEAD_DIM] = [1, 64] = 2 blocks
    std::vector<Q16_1Block> Q_blocks(BLOCKS_PER_HEAD);
    std::vector<Q16_1Block> K_blocks(BLOCKS_PER_HEAD);

    // Set uniform values for easy verification
    for (int b = 0; b < BLOCKS_PER_HEAD; ++b)
    {
        Q_blocks[b].d = 0.01f;
        K_blocks[b].d = 0.01f;
        for (int i = 0; i < 32; ++i)
        {
            Q_blocks[b].qs[i] = 100; // All Q values = 100
            K_blocks[b].qs[i] = 50;  // All K values = 50
        }
    }

    // Set up params
    Q16DotProductParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.q_blocks_per_row = BLOCKS_PER_HEAD;
    params.kv_blocks_per_row = BLOCKS_PER_HEAD;
    params.q_head = 0;
    params.kv_head = 0;
    params.head_dim = HEAD_DIM;
    params.attention_scale = 1.0f; // No scaling for this test

    // Compute dot product
    float result = q16_dot_product_single(params, 0, 0);

    // Expected: sum over 64 elements of (100 * 0.01) * (50 * 0.01)
    // = 64 * 1.0 * 0.5 = 32.0
    // But per-block: block_dot = 32 * (100*50) = 160000
    //               scaled = 160000 * 0.01 * 0.01 = 16.0
    // Total = 2 blocks * 16.0 = 32.0
    EXPECT_NEAR(result, 32.0f, 0.01f);
}

TEST_F(Q16MicrokernelTest, DotProductGemv_MultipleKeys)
{
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 16;
    constexpr int BLOCKS_PER_HEAD = HEAD_DIM / 32;

    auto Q_blocks = createRandomQ16Blocks(BLOCKS_PER_HEAD);
    auto K_blocks = createRandomQ16Blocks(KV_LEN * BLOCKS_PER_HEAD);

    Q16DotProductParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.q_blocks_per_row = BLOCKS_PER_HEAD;
    params.kv_blocks_per_row = BLOCKS_PER_HEAD;
    params.q_head = 0;
    params.kv_head = 0;
    params.head_dim = HEAD_DIM;

    std::vector<float> scores(KV_LEN);

    // GEMV: single query against KV_LEN keys
    q16_dot_product_gemv(params, 0, 0, KV_LEN, scores.data());

    // Verify against single dot product calls
    for (int k = 0; k < KV_LEN; ++k)
    {
        float expected = q16_dot_product_single(params, 0, k);
        EXPECT_NEAR(scores[k], expected, 1e-5f) << "Mismatch at kv=" << k;
    }
}

TEST_F(Q16MicrokernelTest, DotProductGemm_BatchedQueries)
{
    constexpr int HEAD_DIM = 64;
    constexpr int SEQ_LEN_Q = 4;
    constexpr int KV_LEN = 8;
    constexpr int BLOCKS_PER_HEAD = HEAD_DIM / 32;

    auto Q_blocks = createRandomQ16Blocks(SEQ_LEN_Q * BLOCKS_PER_HEAD);
    auto K_blocks = createRandomQ16Blocks(KV_LEN * BLOCKS_PER_HEAD);

    Q16DotProductParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.q_blocks_per_row = BLOCKS_PER_HEAD;
    params.kv_blocks_per_row = BLOCKS_PER_HEAD;
    params.q_head = 0;
    params.kv_head = 0;
    params.head_dim = HEAD_DIM;

    std::vector<float> scores(SEQ_LEN_Q * KV_LEN);

    // GEMM: batched queries
    q16_dot_product_gemm(params, 0, SEQ_LEN_Q, 0, KV_LEN, scores.data());

    // Verify against single calls
    for (int q = 0; q < SEQ_LEN_Q; ++q)
    {
        for (int k = 0; k < KV_LEN; ++k)
        {
            float expected = q16_dot_product_single(params, q, k);
            EXPECT_NEAR(scores[q * KV_LEN + k], expected, 1e-5f)
                << "Mismatch at q=" << q << ", k=" << k;
        }
    }
}

// ============================================================================
// PVAccumulateRef Tests
// ============================================================================

TEST_F(Q16MicrokernelTest, PVAccumulate_Gemv_FP32Weights)
{
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 8;
    constexpr int BLOCKS_PER_HEAD = HEAD_DIM / 32;

    auto V_blocks = createRandomQ16Blocks(KV_LEN * BLOCKS_PER_HEAD);

    // Create normalized FP32 weights
    std::vector<float> weights(KV_LEN);
    float sum = 0.0f;
    for (int i = 0; i < KV_LEN; ++i)
    {
        weights[i] = static_cast<float>(KV_LEN - i); // Decreasing weights
        sum += weights[i];
    }
    for (auto &w : weights)
        w /= sum;

    PVAccumulateParams params;
    params.V = V_blocks.data();
    params.kv_blocks_per_row = BLOCKS_PER_HEAD;
    params.kv_head = 0;
    params.head_dim = HEAD_DIM;

    std::vector<float> output(HEAD_DIM);
    pv_accumulate_gemv_fp32(params, weights.data(), 0, KV_LEN, output.data());

    // Verify output is reasonable (not NaN/Inf)
    for (int d = 0; d < HEAD_DIM; ++d)
    {
        EXPECT_FALSE(std::isnan(output[d])) << "NaN at d=" << d;
        EXPECT_FALSE(std::isinf(output[d])) << "Inf at d=" << d;
    }
}

TEST_F(Q16MicrokernelTest, PVAccumulate_WeightedSum)
{
    // Simple test: single KV position with weight 1.0
    constexpr int HEAD_DIM = 32;

    std::vector<Q16_1Block> V_blocks(1);
    V_blocks[0].d = 0.1f;
    for (int i = 0; i < 32; ++i)
    {
        V_blocks[0].qs[i] = static_cast<int16_t>(i + 1);
    }

    std::vector<float> weights = {1.0f};

    PVAccumulateParams params;
    params.V = V_blocks.data();
    params.kv_blocks_per_row = 1;
    params.kv_head = 0;
    params.head_dim = HEAD_DIM;

    std::vector<float> output(HEAD_DIM);
    pv_accumulate_gemv_fp32(params, weights.data(), 0, 1, output.data());

    // Output should be dequantized V
    for (int d = 0; d < HEAD_DIM; ++d)
    {
        float expected = static_cast<float>(d + 1) * 0.1f;
        EXPECT_NEAR(output[d], expected, 1e-5f) << "Mismatch at d=" << d;
    }
}

// ============================================================================
// WoProjectionVNNIRef Tests (VPDPWSSD INT16×INT16→Q16_1)
// ============================================================================

TEST_F(Q16MicrokernelTest, WoProjectionVNNI_BasicOutput)
{
    // Test basic VPDPWSSD Wo projection with INT32 context → Q16_1 output
    constexpr int INPUT_DIM = 64;
    constexpr int D_MODEL = 64;

    // Create random Wo weights as Q8_1 and pack for VNNI
    std::vector<float> Wo_fp32(D_MODEL * INPUT_DIM);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto &v : Wo_fp32)
        v = dist(rng_);

    auto Wo_q8 = Q8_1Tensor::quantize_from_fp32(
        Wo_fp32.data(),
        {static_cast<size_t>(D_MODEL), static_cast<size_t>(INPUT_DIM)});

    const auto *Wo_packed = llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(Wo_q8.get());
    ASSERT_NE(Wo_packed, nullptr) << "Failed to pack Wo weights";

    // Create INT32 context (simulating P×V output)
    std::vector<int32_t> int32_context(INPUT_DIM);
    std::uniform_int_distribution<int32_t> int_dist(-10000, 10000);
    for (auto &v : int32_context)
        v = int_dist(rng_);

    // Set up IntegerContext
    std::vector<float> v_scales(INPUT_DIM, 0.01f);
    const int32_t weight_sum = 32767; // Full softmax weight
    IntegerContext context;
    context.int32_data = int32_context.data();
    context.weight_sums = &weight_sum;
    context.v_scales = v_scales.data();
    context.num_heads = 1;
    context.head_dim = INPUT_DIM;
    context.count = INPUT_DIM;

    // Output blocks
    const int output_blocks = (D_MODEL + 31) / 32;
    std::vector<Q16_1Block> output_q16(output_blocks);

    Q16_1Projection output;
    output.blocks = output_q16.data();
    output.count = D_MODEL;

    // Set up params
    WoProjectionVNNIParams params;
    params.Wo_packed = Wo_packed;
    params.input_dim = INPUT_DIM;
    params.d_model = D_MODEL;
    params.bias = nullptr;

    // Execute
    wo_projection_vpdpwssd_to_q16_1_gemv(params, context, output);

    // Verify output blocks have valid scales and values
    for (int b = 0; b < output_blocks; ++b)
    {
        EXPECT_GT(output_q16[b].d, 0.0f) << "Invalid scale at block " << b;
        EXPECT_FALSE(std::isnan(output_q16[b].d)) << "NaN scale at block " << b;
    }
}

TEST_F(Q16MicrokernelTest, WoProjectionVNNI_NoNaNsOrInfs)
{
    // Verify no NaN/Inf values in output for various input magnitudes
    constexpr int INPUT_DIM = 128;
    constexpr int D_MODEL = 128;

    std::vector<float> Wo_fp32(D_MODEL * INPUT_DIM);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto &v : Wo_fp32)
        v = dist(rng_);

    auto Wo_q8 = Q8_1Tensor::quantize_from_fp32(
        Wo_fp32.data(),
        {static_cast<size_t>(D_MODEL), static_cast<size_t>(INPUT_DIM)});

    const auto *Wo_packed = llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(Wo_q8.get());
    ASSERT_NE(Wo_packed, nullptr);

    // Test with different context magnitudes
    for (int magnitude : {100, 1000, 10000, 100000})
    {
        std::vector<int32_t> int32_context(INPUT_DIM);
        std::uniform_int_distribution<int32_t> int_dist(-magnitude, magnitude);
        for (auto &v : int32_context)
            v = int_dist(rng_);

        std::vector<float> v_scales(INPUT_DIM, 0.01f);
        const int32_t weight_sum = 32767;
        IntegerContext context;
        context.int32_data = int32_context.data();
        context.weight_sums = &weight_sum;
        context.v_scales = v_scales.data();
        context.num_heads = 1;
        context.head_dim = INPUT_DIM;
        context.count = INPUT_DIM;

        const int output_blocks = (D_MODEL + 31) / 32;
        std::vector<Q16_1Block> output_q16(output_blocks);

        Q16_1Projection output;
        output.blocks = output_q16.data();
        output.count = D_MODEL;

        WoProjectionVNNIParams params;
        params.Wo_packed = Wo_packed;
        params.input_dim = INPUT_DIM;
        params.d_model = D_MODEL;
        params.bias = nullptr;

        wo_projection_vpdpwssd_to_q16_1_gemv(params, context, output);

        // Check all output values
        for (int b = 0; b < output_blocks; ++b)
        {
            EXPECT_FALSE(std::isnan(output_q16[b].d))
                << "NaN scale at magnitude=" << magnitude << ", block=" << b;
            EXPECT_FALSE(std::isinf(output_q16[b].d))
                << "Inf scale at magnitude=" << magnitude << ", block=" << b;
            for (int i = 0; i < 32; ++i)
            {
                // INT16 values can't be NaN/Inf by definition, but check they're in range
                EXPECT_GE(output_q16[b].qs[i], -32768)
                    << "Underflow at magnitude=" << magnitude;
                EXPECT_LE(output_q16[b].qs[i], 32767)
                    << "Overflow at magnitude=" << magnitude;
            }
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
