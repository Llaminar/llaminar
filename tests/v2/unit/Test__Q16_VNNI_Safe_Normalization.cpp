/**
 * @file Test__Q16_VNNI_Safe_Normalization.cpp
 * @brief Unit tests for VNNI-safe Q16 head normalization
 *
 * Tests the normalize_q16_head_to_vnni_safe<BlockType>() function which
 * rescales Q16 blocks to ensure VNNI-safe dot product accumulation.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "kernels/cpu/primitives/Q16HeadNormalization.h"
#include "tensors/BlockStructures.h"

using namespace llaminar2;
using namespace llaminar2::primitives;

// ============================================================================
// Test Fixtures and Helpers
// ============================================================================

class Test__Q16_VNNI_Safe_Normalization : public ::testing::Test
{
protected:
    // Helper: compute cosine similarity between two float vectors
    static double compute_cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size() || a.empty())
            return 0.0;

        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 1.0; // Both zero vectors
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    // Helper: dequantize Q16 blocks to FP32
    template <typename BlockType>
    static std::vector<float> q16_to_fp32(const std::vector<BlockType> &blocks)
    {
        std::vector<float> result;
        for (const auto &blk : blocks)
        {
            for (size_t i = 0; i < BlockType::BLOCK_SIZE; ++i)
            {
                result.push_back(static_cast<float>(blk.qs[i]) * blk.d);
            }
        }
        return result;
    }

    // Helper: create Q16 blocks from FP32 data with given scale
    template <typename BlockType>
    static std::vector<BlockType> fp32_to_q16(const std::vector<float> &data, float d)
    {
        size_t num_blocks = (data.size() + BlockType::BLOCK_SIZE - 1) / BlockType::BLOCK_SIZE;
        std::vector<BlockType> blocks(num_blocks);

        for (size_t b = 0; b < num_blocks; ++b)
        {
            blocks[b].d = d;
            blocks[b].sum_qs = 0;
            for (size_t i = 0; i < BlockType::BLOCK_SIZE; ++i)
            {
                size_t idx = b * BlockType::BLOCK_SIZE + i;
                if (idx < data.size())
                {
                    int16_t qs = static_cast<int16_t>(std::round(data[idx] / d));
                    blocks[b].qs[i] = qs;
                    blocks[b].sum_qs += qs;
                }
                else
                {
                    blocks[b].qs[i] = 0;
                }
            }
        }
        return blocks;
    }

    // Helper: simulate VNNI Q16×Q16→INT32 dot product for one head position
    static int64_t simulate_vnni_dot_product(const int16_t *q, const int16_t *k, int head_dim)
    {
        int64_t acc = 0;
        for (int i = 0; i < head_dim; ++i)
        {
            acc += static_cast<int64_t>(q[i]) * static_cast<int64_t>(k[i]);
        }
        return acc;
    }
};

// ============================================================================
// get_vnni_safe_max_qs Tests
// ============================================================================

TEST_F(Test__Q16_VNNI_Safe_Normalization, VNNISafeMaxQs_HeadDim64)
{
    // sqrt(INT32_MAX / 64) * 0.95 ≈ 5500
    EXPECT_EQ(get_vnni_safe_max_qs(64), 5500);
}

TEST_F(Test__Q16_VNNI_Safe_Normalization, VNNISafeMaxQs_HeadDim128)
{
    // sqrt(INT32_MAX / 128) * 0.95 ≈ 3890
    EXPECT_EQ(get_vnni_safe_max_qs(128), 3890);
}

TEST_F(Test__Q16_VNNI_Safe_Normalization, VNNISafeMaxQs_HeadDim192)
{
    // sqrt(INT32_MAX / 192) * 0.95 ≈ 3175
    EXPECT_EQ(get_vnni_safe_max_qs(192), 3175);
}

TEST_F(Test__Q16_VNNI_Safe_Normalization, VNNISafeMaxQs_HeadDim256)
{
    // sqrt(INT32_MAX / 256) * 0.95 ≈ 2750
    EXPECT_EQ(get_vnni_safe_max_qs(256), 2750);
}

// ============================================================================
// find_head_max_abs_qs Tests
// ============================================================================

TEST_F(Test__Q16_VNNI_Safe_Normalization, FindHeadMaxAbsQs_SingleBlock)
{
    Q16_1Block blk;
    blk.d = 1.0f;
    for (int i = 0; i < 32; ++i)
    {
        blk.qs[i] = static_cast<int16_t>(i * 100); // Max = 3100
    }
    blk.qs[15] = -5000; // Negative max

    int16_t max_abs = find_head_max_abs_qs<Q16_1Block>(&blk, 1);
    EXPECT_EQ(max_abs, 5000);
}

TEST_F(Test__Q16_VNNI_Safe_Normalization, FindHeadMaxAbsQs_MultipleBlocks)
{
    std::vector<Q16_1Block> blocks(4);
    for (auto &blk : blocks)
    {
        blk.d = 1.0f;
        for (int i = 0; i < 32; ++i)
        {
            blk.qs[i] = 1000;
        }
    }
    blocks[2].qs[10] = 20000; // Max in block 2

    int16_t max_abs = find_head_max_abs_qs<Q16_1Block>(blocks.data(), 4);
    EXPECT_EQ(max_abs, 20000);
}

TEST_F(Test__Q16_VNNI_Safe_Normalization, FindHeadMaxAbsQs_Empty)
{
    int16_t max_abs = find_head_max_abs_qs<Q16_1Block>(nullptr, 0);
    EXPECT_EQ(max_abs, 0);
}

// ============================================================================
// normalize_q16_head_to_vnni_safe Tests
// ============================================================================

TEST_F(Test__Q16_VNNI_Safe_Normalization, NoRescaleNeeded_AlreadySafe)
{
    const int head_dim = 128;
    const int num_blocks = head_dim / 32;
    const int16_t safe_max = get_vnni_safe_max_qs(head_dim); // 16383

    // Create blocks with all values well below safe limit
    std::vector<Q16_1Block> blocks(num_blocks);
    float d_unified = 0.01f;

    for (auto &blk : blocks)
    {
        blk.d = d_unified;
        blk.sum_qs = 0;
        for (int i = 0; i < 32; ++i)
        {
            blk.qs[i] = static_cast<int16_t>(i * 100); // Max = 3100, well below 16383
            blk.sum_qs += blk.qs[i];
        }
    }

    // Store original FP32 values
    auto fp32_before = q16_to_fp32(blocks);

    auto result = normalize_q16_head_to_vnni_safe<Q16_1Block>(
        blocks.data(), num_blocks, head_dim, d_unified);

    EXPECT_FALSE(result.was_rescaled);
    EXPECT_FLOAT_EQ(result.norm_factor, 1.0f);
    EXPECT_FLOAT_EQ(result.new_d, d_unified);
    EXPECT_EQ(result.max_qs, 3100);

    // Verify FP32 values unchanged
    auto fp32_after = q16_to_fp32(blocks);
    for (size_t i = 0; i < fp32_before.size(); ++i)
    {
        EXPECT_FLOAT_EQ(fp32_before[i], fp32_after[i]);
    }
}

TEST_F(Test__Q16_VNNI_Safe_Normalization, RescalesHighValues_HeadDim128)
{
    const int head_dim = 128;
    const int num_blocks = head_dim / 32;
    const int16_t safe_max = get_vnni_safe_max_qs(head_dim); // 16383

    // Create blocks with values exceeding safe limit
    std::vector<Q16_1Block> blocks(num_blocks);
    float d_unified = 0.01f;

    for (auto &blk : blocks)
    {
        blk.d = d_unified;
        blk.sum_qs = 0;
        for (int i = 0; i < 32; ++i)
        {
            blk.qs[i] = 20000; // Above safe_max of 16383
            blk.sum_qs += blk.qs[i];
        }
    }

    // Store original FP32 values for comparison
    auto fp32_before = q16_to_fp32(blocks);

    auto result = normalize_q16_head_to_vnni_safe<Q16_1Block>(
        blocks.data(), num_blocks, head_dim, d_unified);

    EXPECT_TRUE(result.was_rescaled);
    EXPECT_LT(result.norm_factor, 1.0f); // Should be ~16383/20000 ≈ 0.819

    // Verify all qs values are now <= safe_max
    for (const auto &blk : blocks)
    {
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_LE(std::abs(blk.qs[i]), safe_max)
                << "qs[" << i << "] = " << blk.qs[i] << " exceeds safe_max " << safe_max;
        }
    }

    // Verify FP32 values preserved (cosine similarity)
    auto fp32_after = q16_to_fp32(blocks);
    double cosine = compute_cosine_similarity(fp32_before, fp32_after);
    EXPECT_GT(cosine, 0.999) << "FP32 values should be preserved after rescaling";
}

TEST_F(Test__Q16_VNNI_Safe_Normalization, PreservesRelativeMagnitudes)
{
    const int head_dim = 128;
    const int num_blocks = head_dim / 32;

    // Create blocks with varying values including spike
    std::vector<float> fp32_data(head_dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    for (int i = 0; i < head_dim; ++i)
    {
        fp32_data[i] = dist(rng);
    }
    // Add spike that will require rescaling
    fp32_data[50] = 260.0f; // Peak value

    // Find max for quantization
    float max_abs = 0.0f;
    for (float v : fp32_data)
    {
        max_abs = std::max(max_abs, std::abs(v));
    }
    float d_unified = max_abs / 32767.0f; // Use full INT16 range

    auto blocks = fp32_to_q16<Q16_1Block>(fp32_data, d_unified);
    auto fp32_before = q16_to_fp32(blocks);

    auto result = normalize_q16_head_to_vnni_safe<Q16_1Block>(
        blocks.data(), num_blocks, head_dim, d_unified);

    auto fp32_after = q16_to_fp32(blocks);

    double cosine = compute_cosine_similarity(fp32_before, fp32_after);
    EXPECT_GT(cosine, 0.999)
        << "Cosine similarity should be > 0.999, got " << cosine;
}

TEST_F(Test__Q16_VNNI_Safe_Normalization, NormFactorCorrect)
{
    const int head_dim = 64;
    const int num_blocks = head_dim / 32;
    const int16_t safe_max = get_vnni_safe_max_qs(head_dim); // 23170

    // Create blocks with known values
    std::vector<Q16_1Block> blocks(num_blocks);
    float d_unified = 0.005f;
    const int16_t input_max = 30000; // Above safe_max

    for (auto &blk : blocks)
    {
        blk.d = d_unified;
        blk.sum_qs = 0;
        for (int i = 0; i < 32; ++i)
        {
            blk.qs[i] = input_max;
            blk.sum_qs += input_max;
        }
    }

    // Expected scale_ratio = safe_max / input_max = 23170 / 30000 ≈ 0.7723
    float expected_ratio = static_cast<float>(safe_max) / static_cast<float>(input_max);

    auto result = normalize_q16_head_to_vnni_safe<Q16_1Block>(
        blocks.data(), num_blocks, head_dim, d_unified);

    EXPECT_TRUE(result.was_rescaled);
    EXPECT_NEAR(result.norm_factor, expected_ratio, 0.01f);

    // Verify new_d = d_unified / scale_ratio
    EXPECT_NEAR(result.new_d, d_unified / expected_ratio, 1e-6f);
}

TEST_F(Test__Q16_VNNI_Safe_Normalization, NoOverflowInVNNI_SimulatedDotProduct)
{
    // This is the KEY test: verify that after normalization,
    // Q16×Q16→INT32 dot products don't overflow INT32

    const int head_dim = 128;
    const int num_blocks = head_dim / 32;

    // Create Q and K with max safe values after normalization
    std::vector<Q16_1Block> q_blocks(num_blocks);
    std::vector<Q16_1Block> k_blocks(num_blocks);
    float d_unified = 0.01f;

    // Fill with values that would overflow without normalization
    for (auto &blk : q_blocks)
    {
        blk.d = d_unified;
        for (int i = 0; i < 32; ++i)
        {
            blk.qs[i] = 25000; // Above safe limit
        }
    }
    for (auto &blk : k_blocks)
    {
        blk.d = d_unified;
        for (int i = 0; i < 32; ++i)
        {
            blk.qs[i] = 25000; // Above safe limit
        }
    }

    // Normalize both Q and K
    auto q_result = normalize_q16_head_to_vnni_safe<Q16_1Block>(
        q_blocks.data(), num_blocks, head_dim, d_unified);
    auto k_result = normalize_q16_head_to_vnni_safe<Q16_1Block>(
        k_blocks.data(), num_blocks, head_dim, d_unified);

    // Extract qs arrays for dot product simulation
    std::vector<int16_t> q_qs(head_dim), k_qs(head_dim);
    for (int b = 0; b < num_blocks; ++b)
    {
        for (int i = 0; i < 32; ++i)
        {
            q_qs[b * 32 + i] = q_blocks[b].qs[i];
            k_qs[b * 32 + i] = k_blocks[b].qs[i];
        }
    }

    // Simulate VNNI dot product
    int64_t dot = simulate_vnni_dot_product(q_qs.data(), k_qs.data(), head_dim);

    // Verify no overflow (fits in INT32)
    EXPECT_GE(dot, static_cast<int64_t>(std::numeric_limits<int32_t>::min()));
    EXPECT_LE(dot, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));

    // Also verify the corrected score is reasonable
    float raw_score = static_cast<float>(dot);
    float corrected_score = raw_score * q_result.new_d * k_result.new_d;

    // The corrected score should be similar to what we'd get from FP32
    // With all 25000s and d=0.01, original FP32 values were 250 each
    // FP32 dot = 250 * 250 * 128 = 8,000,000
    float expected_fp32_dot = 250.0f * 250.0f * head_dim;
    EXPECT_NEAR(corrected_score, expected_fp32_dot, expected_fp32_dot * 0.05f)
        << "Corrected score should match FP32 within 5%";
}

TEST_F(Test__Q16_VNNI_Safe_Normalization, AllBlockTypes_Supported)
{
    // Test all three block types work correctly
    const float d_unified = 0.005f;

    // Q16_1Block (32)
    {
        const int head_dim = 64;
        std::vector<Q16_1Block> blocks(2);
        for (auto &blk : blocks)
        {
            blk.d = d_unified;
            for (int i = 0; i < 32; ++i)
                blk.qs[i] = 25000;
        }
        auto result = normalize_q16_head_to_vnni_safe<Q16_1Block>(
            blocks.data(), 2, head_dim, d_unified);
        EXPECT_TRUE(result.was_rescaled);
        EXPECT_LE(find_head_max_abs_qs(blocks.data(), 2), get_vnni_safe_max_qs(head_dim));
    }

    // Q16_1Block_64
    {
        const int head_dim = 128;
        std::vector<Q16_1Block_64> blocks(2);
        for (auto &blk : blocks)
        {
            blk.d = d_unified;
            for (int i = 0; i < 64; ++i)
                blk.qs[i] = 20000;
        }
        auto result = normalize_q16_head_to_vnni_safe<Q16_1Block_64>(
            blocks.data(), 2, head_dim, d_unified);
        EXPECT_TRUE(result.was_rescaled);
        EXPECT_LE(find_head_max_abs_qs(blocks.data(), 2), get_vnni_safe_max_qs(head_dim));
    }

    // Q16_1Block_128
    {
        const int head_dim = 128;
        std::vector<Q16_1Block_128> blocks(1);
        blocks[0].d = d_unified;
        for (int i = 0; i < 128; ++i)
            blocks[0].qs[i] = 18000;
        auto result = normalize_q16_head_to_vnni_safe<Q16_1Block_128>(
            blocks.data(), 1, head_dim, d_unified);
        EXPECT_TRUE(result.was_rescaled);
        EXPECT_LE(find_head_max_abs_qs(blocks.data(), 1), get_vnni_safe_max_qs(head_dim));
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Test__Q16_VNNI_Safe_Normalization, ZeroInput)
{
    const int head_dim = 64;
    std::vector<Q16_1Block> blocks(2);
    for (auto &blk : blocks)
    {
        blk.d = 0.01f;
        for (int i = 0; i < 32; ++i)
            blk.qs[i] = 0;
        blk.sum_qs = 0;
    }

    auto result = normalize_q16_head_to_vnni_safe<Q16_1Block>(
        blocks.data(), 2, head_dim, 0.01f);

    EXPECT_FALSE(result.was_rescaled);
    EXPECT_EQ(result.max_qs, 0);
}

TEST_F(Test__Q16_VNNI_Safe_Normalization, NullPointer)
{
    auto result = normalize_q16_head_to_vnni_safe<Q16_1Block>(
        nullptr, 0, 128, 0.01f);

    EXPECT_FALSE(result.was_rescaled);
    EXPECT_FLOAT_EQ(result.norm_factor, 1.0f);
}

TEST_F(Test__Q16_VNNI_Safe_Normalization, ZeroUnifiedScale)
{
    Q16_1Block blk;
    blk.d = 0.0f;
    for (int i = 0; i < 32; ++i)
        blk.qs[i] = 10000;

    auto result = normalize_q16_head_to_vnni_safe<Q16_1Block>(
        &blk, 1, 64, 0.0f); // d_unified = 0

    EXPECT_FALSE(result.was_rescaled);
    EXPECT_FLOAT_EQ(result.norm_factor, 1.0f);
}

TEST_F(Test__Q16_VNNI_Safe_Normalization, NegativeQsValues)
{
    const int head_dim = 64;
    std::vector<Q16_1Block> blocks(2);
    float d_unified = 0.01f;

    for (auto &blk : blocks)
    {
        blk.d = d_unified;
        blk.sum_qs = 0;
        for (int i = 0; i < 32; ++i)
        {
            // Alternating positive/negative with some exceeding safe limit
            blk.qs[i] = (i % 2 == 0) ? 25000 : -25000;
            blk.sum_qs += blk.qs[i];
        }
    }

    auto result = normalize_q16_head_to_vnni_safe<Q16_1Block>(
        blocks.data(), 2, head_dim, d_unified);

    EXPECT_TRUE(result.was_rescaled);

    // Verify all values clamped correctly
    int16_t safe_max = get_vnni_safe_max_qs(head_dim);
    for (const auto &blk : blocks)
    {
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_LE(std::abs(blk.qs[i]), safe_max);
        }
    }

    // Verify sum_qs is correct
    for (const auto &blk : blocks)
    {
        int32_t expected_sum = 0;
        for (int i = 0; i < 32; ++i)
        {
            expected_sum += blk.qs[i];
        }
        EXPECT_EQ(blk.sum_qs, expected_sum);
    }
}

TEST_F(Test__Q16_VNNI_Safe_Normalization, SpikyKProjection_RealWorld)
{
    // Simulate the real problem: K projection with peaks ~130
    const int head_dim = 128;
    const int num_blocks = head_dim / 32;

    // Generate realistic K projection output with spike
    std::vector<float> k_fp32(head_dim);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 20.0f);

    for (int i = 0; i < head_dim; ++i)
    {
        k_fp32[i] = dist(rng);
    }
    // Add realistic spike (seen in attention analysis)
    k_fp32[42] = 130.0f;
    k_fp32[87] = -115.0f;

    // Find max for quantization (from dynamic-scale RoPE)
    float max_abs = 0.0f;
    for (float v : k_fp32)
    {
        max_abs = std::max(max_abs, std::abs(v));
    }
    float d_unified = max_abs / 32767.0f; // Full INT16 range

    // Quantize to Q16
    auto blocks = fp32_to_q16<Q16_1Block>(k_fp32, d_unified);
    auto fp32_before = q16_to_fp32(blocks);

    // Apply VNNI-safe normalization
    auto result = normalize_q16_head_to_vnni_safe<Q16_1Block>(
        blocks.data(), num_blocks, head_dim, d_unified);

    // Key assertions:
    // 1. Was rescaled (because max |qs| ≈ 32767 > 16383)
    EXPECT_TRUE(result.was_rescaled);

    // 2. All values now safe
    int16_t safe_max = get_vnni_safe_max_qs(head_dim);
    int16_t actual_max = find_head_max_abs_qs(blocks.data(), num_blocks);
    EXPECT_LE(actual_max, safe_max);

    // 3. FP32 values preserved
    auto fp32_after = q16_to_fp32(blocks);
    double cosine = compute_cosine_similarity(fp32_before, fp32_after);
    EXPECT_GT(cosine, 0.999)
        << "Spiky K projection should preserve cosine > 0.999, got " << cosine;

    // 4. Spike values preserved relatively
    float spike_before = 130.0f;
    float spike_after = static_cast<float>(blocks[42 / 32].qs[42 % 32]) * blocks[42 / 32].d;
    EXPECT_NEAR(spike_after, spike_before, spike_before * 0.01f)
        << "Spike at position 42 should be preserved within 1%";
}
