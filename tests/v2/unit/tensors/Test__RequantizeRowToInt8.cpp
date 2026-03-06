/**
 * @file Test__RequantizeRowToInt8.cpp
 * @brief Unit tests for IINT8Unpackable::requantizeRowToInt8()
 *
 * Tests the per-row INT8 requantization used by ROCmWeightPacker to convert
 * quantized weight blocks to uniform INT8 + row scale for ROCm GEMM kernels.
 *
 * Coverage:
 *   - Q8_0Tensor: SIMD-optimized override (AVX-512 / AVX2 / scalar)
 *   - Q8_1Tensor: SIMD-optimized override (AVX-512 / AVX2 / scalar)
 *   - Q8_KTensor: Direct-copy override (memcpy + optional rescale)
 *   - Reconstruction accuracy (dequant(requantized) ≈ dequant(original))
 *   - Edge cases: zero rows, single block, uniform values, -128 values
 *   - Multi-row tensors: each row gets independent scale
 */

#include <gtest/gtest.h>
#include "tensors/TensorClasses.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace llaminar2;

// ============================================================================
// Test fixture with helpers for tensor construction
// ============================================================================

class Test__RequantizeRowToInt8 : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

    // --- Q8_0 helpers ---

    /// Build a Q8_0Tensor from a vector of blocks laid out as [rows x blocks_per_row].
    std::unique_ptr<Q8_0Tensor> makeQ8_0Tensor(
        size_t rows, size_t cols, const std::vector<Q8_0Block> &blocks)
    {
        std::vector<uint8_t> raw(blocks.size() * sizeof(Q8_0Block));
        std::memcpy(raw.data(), blocks.data(), raw.size());
        return std::make_unique<Q8_0Tensor>(
            std::vector<size_t>{rows, cols}, raw);
    }

    /// Fill a Q8_0 block with given scale and qs values.
    Q8_0Block makeQ8_0Block(float scale, const int8_t qs[32])
    {
        Q8_0Block blk{};
        blk.d = fp32_to_fp16(scale);
        std::memcpy(blk.qs, qs, 32);
        return blk;
    }

    /// Fill a random Q8_0 block.
    Q8_0Block randomQ8_0Block(float scale_lo, float scale_hi)
    {
        std::uniform_real_distribution<float> sdist(scale_lo, scale_hi);
        std::uniform_int_distribution<int> qdist(-127, 127);
        Q8_0Block blk{};
        blk.d = fp32_to_fp16(sdist(rng_));
        for (auto &q : blk.qs)
            q = static_cast<int8_t>(qdist(rng_));
        return blk;
    }

    // --- Q8_1 helpers ---

    std::unique_ptr<Q8_1Tensor> makeQ8_1Tensor(
        size_t rows, size_t cols, const std::vector<Q8_1Block> &blocks)
    {
        std::vector<uint8_t> raw(blocks.size() * sizeof(Q8_1Block));
        std::memcpy(raw.data(), blocks.data(), raw.size());
        return std::make_unique<Q8_1Tensor>(
            std::vector<size_t>{rows, cols}, raw);
    }

    Q8_1Block randomQ8_1Block(float scale_lo, float scale_hi)
    {
        std::uniform_real_distribution<float> sdist(scale_lo, scale_hi);
        std::uniform_int_distribution<int> qdist(-127, 127);
        Q8_1Block blk{};
        blk.d = fp32_to_fp16(sdist(rng_));
        int16_t sum = 0;
        for (auto &q : blk.qs)
        {
            q = static_cast<int8_t>(qdist(rng_));
            sum += q;
        }
        blk.sum_qs = sum;
        return blk;
    }

    // --- Q8_K helpers ---

    std::unique_ptr<Q8_KTensor> makeQ8_KTensor(
        size_t rows, size_t cols, const std::vector<Q8_KBlock> &blocks)
    {
        std::vector<uint8_t> raw(blocks.size() * sizeof(Q8_KBlock));
        std::memcpy(raw.data(), blocks.data(), raw.size());
        return std::make_unique<Q8_KTensor>(
            std::vector<size_t>{rows, cols}, raw);
    }

    Q8_KBlock randomQ8_KBlock(bool include_neg128 = false)
    {
        const int lo = include_neg128 ? -128 : -127;
        std::uniform_int_distribution<int> qdist(lo, 127);
        Q8_KBlock blk{};
        for (auto &q : blk.qs)
            q = static_cast<int8_t>(qdist(rng_));
        std::memset(blk.bsums, 0, sizeof(blk.bsums));
        return blk;
    }

    // --- Verification helpers ---

    /// Dequantize a Q8_0 row to FP32 (ground truth).
    std::vector<float> dequantQ8_0Row(
        const Q8_0Block *row_blocks, size_t blocks_per_row)
    {
        std::vector<float> out(blocks_per_row * 32);
        for (size_t b = 0; b < blocks_per_row; ++b)
        {
            const float s = fp16_to_fp32(row_blocks[b].d);
            for (int i = 0; i < 32; ++i)
                out[b * 32 + i] = static_cast<float>(row_blocks[b].qs[i]) * s;
        }
        return out;
    }

    /// Dequantize a Q8_1 row to FP32 (ground truth — symmetric, min=0).
    std::vector<float> dequantQ8_1Row(
        const Q8_1Block *row_blocks, size_t blocks_per_row)
    {
        std::vector<float> out(blocks_per_row * 32);
        for (size_t b = 0; b < blocks_per_row; ++b)
        {
            const float s = fp16_to_fp32(row_blocks[b].d);
            for (int i = 0; i < 32; ++i)
                out[b * 32 + i] = static_cast<float>(row_blocks[b].qs[i]) * s;
        }
        return out;
    }

    /// Dequantize a Q8_K row to FP32 (scale=1, so qs values are the dequantized values).
    std::vector<float> dequantQ8_KRow(
        const Q8_KBlock *row_blocks, size_t superblocks_per_row)
    {
        std::vector<float> out(superblocks_per_row * 256);
        for (size_t sb = 0; sb < superblocks_per_row; ++sb)
            for (int i = 0; i < 256; ++i)
                out[sb * 256 + i] = static_cast<float>(row_blocks[sb].qs[i]);
        return out;
    }

    /// Dequantize requantized INT8 output using the returned row_scale.
    std::vector<float> dequantInt8Row(
        const int8_t *data, size_t K, float row_scale)
    {
        std::vector<float> out(K);
        for (size_t i = 0; i < K; ++i)
            out[i] = static_cast<float>(data[i]) * row_scale;
        return out;
    }

    /// Compute max relative error between two FP32 vectors (relative to max magnitude).
    float maxRelativeError(
        const std::vector<float> &ref,
        const std::vector<float> &test)
    {
        float max_mag = 0.0f;
        for (float v : ref)
            max_mag = std::max(max_mag, std::abs(v));
        if (max_mag == 0.0f)
            return 0.0f;
        float max_err = 0.0f;
        for (size_t i = 0; i < ref.size(); ++i)
        {
            float err = std::abs(ref[i] - test[i]) / max_mag;
            max_err = std::max(max_err, err);
        }
        return max_err;
    }

    /// Cosine similarity between two FP32 vectors.
    double cosineSimilarity(
        const std::vector<float> &a,
        const std::vector<float> &b)
    {
        double dot = 0.0, mag_a = 0.0, mag_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            mag_a += static_cast<double>(a[i]) * a[i];
            mag_b += static_cast<double>(b[i]) * b[i];
        }
        if (mag_a == 0.0 || mag_b == 0.0)
            return (mag_a == 0.0 && mag_b == 0.0) ? 1.0 : 0.0;
        return dot / (std::sqrt(mag_a) * std::sqrt(mag_b));
    }
};

// ============================================================================
// Q8_0 Tests
// ============================================================================

TEST_F(Test__RequantizeRowToInt8, Q8_0_SingleBlock_KnownValues)
{
    // Single block: qs = {1,2,...,32}, scale = 0.5
    int8_t qs[32];
    for (int i = 0; i < 32; ++i)
        qs[i] = static_cast<int8_t>(i + 1);
    std::vector<Q8_0Block> blocks = {makeQ8_0Block(0.5f, qs)};
    auto tensor = makeQ8_0Tensor(1, 32, blocks);

    int8_t output[32];
    float row_scale = tensor->requantizeRowToInt8(0, 32, output);

    // Max dequant value = 32 * 0.5 = 16.0, so row_scale = 16/127
    EXPECT_NEAR(row_scale, 16.0f / 127.0f, 1e-4f);

    // Verify reconstruction accuracy
    auto ref = dequantQ8_0Row(blocks.data(), 1);
    auto recon = dequantInt8Row(output, 32, row_scale);
    // With single-block Q8_0 the only quantization loss is the
    // qs[i]*block_scale → qs_out[i]*row_scale rounding.
    // Max relative error should be very small.
    EXPECT_LT(maxRelativeError(ref, recon), 0.02f)
        << "Reconstruction error too large for single Q8_0 block";
}

TEST_F(Test__RequantizeRowToInt8, Q8_0_MultiBlock_RandomData)
{
    const size_t blocks_per_row = 8;
    const size_t K = blocks_per_row * 32;
    std::vector<Q8_0Block> blocks;
    for (size_t i = 0; i < blocks_per_row; ++i)
        blocks.push_back(randomQ8_0Block(0.001f, 1.0f));
    auto tensor = makeQ8_0Tensor(1, K, blocks);

    std::vector<int8_t> output(K);
    float row_scale = tensor->requantizeRowToInt8(0, K, output.data());

    EXPECT_GT(row_scale, 0.0f);

    auto ref = dequantQ8_0Row(blocks.data(), blocks_per_row);
    auto recon = dequantInt8Row(output.data(), K, row_scale);

    double cosim = cosineSimilarity(ref, recon);
    EXPECT_GT(cosim, 0.995)
        << "Cosine similarity too low: " << cosim;
    EXPECT_LT(maxRelativeError(ref, recon), 0.03f);
}

TEST_F(Test__RequantizeRowToInt8, Q8_0_MultiRow_IndependentScales)
{
    // Two rows with very different scales to verify independence
    const size_t K = 64; // 2 blocks per row
    int8_t qs_lo[32], qs_hi[32];
    for (int i = 0; i < 32; ++i)
    {
        qs_lo[i] = static_cast<int8_t>(i - 16);     // small range
        qs_hi[i] = static_cast<int8_t>(i * 4 - 64); // wider range
    }
    std::vector<Q8_0Block> blocks = {
        makeQ8_0Block(0.01f, qs_lo), // row 0, block 0: tiny scale
        makeQ8_0Block(0.01f, qs_lo), // row 0, block 1
        makeQ8_0Block(10.0f, qs_hi), // row 1, block 0: large scale
        makeQ8_0Block(10.0f, qs_hi), // row 1, block 1
    };
    auto tensor = makeQ8_0Tensor(2, K, blocks);

    std::vector<int8_t> out0(K), out1(K);
    float scale0 = tensor->requantizeRowToInt8(0, K, out0.data());
    float scale1 = tensor->requantizeRowToInt8(1, K, out1.data());

    // Scales should differ by ~1000x
    EXPECT_GT(scale1, scale0 * 100.0f)
        << "Row scales should be very different: " << scale0 << " vs " << scale1;

    // Each row should reconstruct well independently
    auto ref0 = dequantQ8_0Row(blocks.data(), 2);
    auto ref1 = dequantQ8_0Row(blocks.data() + 2, 2);
    auto rec0 = dequantInt8Row(out0.data(), K, scale0);
    auto rec1 = dequantInt8Row(out1.data(), K, scale1);
    EXPECT_GT(cosineSimilarity(ref0, rec0), 0.99);
    EXPECT_GT(cosineSimilarity(ref1, rec1), 0.99);
}

TEST_F(Test__RequantizeRowToInt8, Q8_0_ZeroRow)
{
    int8_t qs[32] = {};
    std::vector<Q8_0Block> blocks = {makeQ8_0Block(0.0f, qs)};
    auto tensor = makeQ8_0Tensor(1, 32, blocks);

    int8_t output[32];
    float row_scale = tensor->requantizeRowToInt8(0, 32, output);

    EXPECT_EQ(row_scale, 1.0f) << "Zero row should produce scale=1.0";
    for (int i = 0; i < 32; ++i)
        EXPECT_EQ(output[i], 0) << "Zero row output should be all zeros";
}

TEST_F(Test__RequantizeRowToInt8, Q8_0_UniformValues)
{
    // All qs the same → output should be uniform too
    int8_t qs[32];
    std::fill_n(qs, 32, static_cast<int8_t>(64));
    std::vector<Q8_0Block> blocks = {makeQ8_0Block(0.25f, qs)};
    auto tensor = makeQ8_0Tensor(1, 32, blocks);

    int8_t output[32];
    float row_scale = tensor->requantizeRowToInt8(0, 32, output);

    // All values are identical → max_abs = 64 * 0.25 = 16.0
    // row_scale = 16/127, all output = round(64 * 0.25 / row_scale) = round(127) = 127
    for (int i = 0; i < 32; ++i)
        EXPECT_EQ(output[i], 127)
            << "Uniform input should map to 127 (max range)";
}

TEST_F(Test__RequantizeRowToInt8, Q8_0_OutputClamped)
{
    // Verify no output exceeds [-127, 127]
    const size_t K = 32 * 16; // 16 blocks
    std::vector<Q8_0Block> blocks;
    for (size_t i = 0; i < 16; ++i)
        blocks.push_back(randomQ8_0Block(0.01f, 5.0f));
    auto tensor = makeQ8_0Tensor(1, K, blocks);

    std::vector<int8_t> output(K);
    tensor->requantizeRowToInt8(0, K, output.data());

    for (size_t i = 0; i < K; ++i)
    {
        EXPECT_GE(output[i], -127) << "Output below -127 at index " << i;
        EXPECT_LE(output[i], 127) << "Output above 127 at index " << i;
    }
}

TEST_F(Test__RequantizeRowToInt8, Q8_0_LargeRow_Accuracy)
{
    // Simulate a realistic weight row (K=4096 → 128 blocks)
    const size_t blocks_per_row = 128;
    const size_t K = blocks_per_row * 32;
    std::vector<Q8_0Block> blocks;
    for (size_t i = 0; i < blocks_per_row; ++i)
        blocks.push_back(randomQ8_0Block(0.001f, 0.5f));
    auto tensor = makeQ8_0Tensor(1, K, blocks);

    std::vector<int8_t> output(K);
    float row_scale = tensor->requantizeRowToInt8(0, K, output.data());

    auto ref = dequantQ8_0Row(blocks.data(), blocks_per_row);
    auto recon = dequantInt8Row(output.data(), K, row_scale);

    double cosim = cosineSimilarity(ref, recon);
    EXPECT_GT(cosim, 0.995)
        << "Large-row cosine similarity too low: " << cosim;
}

// ============================================================================
// Q8_1 Tests
// ============================================================================

TEST_F(Test__RequantizeRowToInt8, Q8_1_MultiBlock_RandomData)
{
    const size_t blocks_per_row = 8;
    const size_t K = blocks_per_row * 32;
    std::vector<Q8_1Block> blocks;
    for (size_t i = 0; i < blocks_per_row; ++i)
        blocks.push_back(randomQ8_1Block(0.001f, 1.0f));
    auto tensor = makeQ8_1Tensor(1, K, blocks);

    std::vector<int8_t> output(K);
    float row_scale = tensor->requantizeRowToInt8(0, K, output.data());

    EXPECT_GT(row_scale, 0.0f);

    auto ref = dequantQ8_1Row(blocks.data(), blocks_per_row);
    auto recon = dequantInt8Row(output.data(), K, row_scale);

    double cosim = cosineSimilarity(ref, recon);
    EXPECT_GT(cosim, 0.995)
        << "Q8_1 cosine similarity too low: " << cosim;
    EXPECT_LT(maxRelativeError(ref, recon), 0.03f);
}

TEST_F(Test__RequantizeRowToInt8, Q8_1_ZeroRow)
{
    Q8_1Block blk{};
    blk.d = fp32_to_fp16(0.0f);
    blk.sum_qs = 0;
    std::memset(blk.qs, 0, 32);
    std::vector<Q8_1Block> blocks = {blk};
    auto tensor = makeQ8_1Tensor(1, 32, blocks);

    int8_t output[32];
    float row_scale = tensor->requantizeRowToInt8(0, 32, output);

    EXPECT_EQ(row_scale, 1.0f);
    for (int i = 0; i < 32; ++i)
        EXPECT_EQ(output[i], 0);
}

TEST_F(Test__RequantizeRowToInt8, Q8_1_LargeRow_Accuracy)
{
    const size_t blocks_per_row = 128;
    const size_t K = blocks_per_row * 32;
    std::vector<Q8_1Block> blocks;
    for (size_t i = 0; i < blocks_per_row; ++i)
        blocks.push_back(randomQ8_1Block(0.001f, 0.5f));
    auto tensor = makeQ8_1Tensor(1, K, blocks);

    std::vector<int8_t> output(K);
    float row_scale = tensor->requantizeRowToInt8(0, K, output.data());

    auto ref = dequantQ8_1Row(blocks.data(), blocks_per_row);
    auto recon = dequantInt8Row(output.data(), K, row_scale);

    double cosim = cosineSimilarity(ref, recon);
    EXPECT_GT(cosim, 0.995)
        << "Q8_1 large-row cosine similarity too low: " << cosim;
}

TEST_F(Test__RequantizeRowToInt8, Q8_1_OutputClamped)
{
    const size_t K = 32 * 16;
    std::vector<Q8_1Block> blocks;
    for (size_t i = 0; i < 16; ++i)
        blocks.push_back(randomQ8_1Block(0.01f, 5.0f));
    auto tensor = makeQ8_1Tensor(1, K, blocks);

    std::vector<int8_t> output(K);
    tensor->requantizeRowToInt8(0, K, output.data());

    for (size_t i = 0; i < K; ++i)
    {
        EXPECT_GE(output[i], -127);
        EXPECT_LE(output[i], 127);
    }
}

// ============================================================================
// Q8_K Tests
// ============================================================================

TEST_F(Test__RequantizeRowToInt8, Q8_K_NoClamping_DirectCopy)
{
    // Values in [-127, 127] → direct memcpy, no rescaling
    Q8_KBlock blk{};
    for (int i = 0; i < 256; ++i)
        blk.qs[i] = static_cast<int8_t>(i % 255 - 127); // range [-127, 127]
    std::vector<Q8_KBlock> blocks = {blk};
    auto tensor = makeQ8_KTensor(1, 256, blocks);

    std::vector<int8_t> output(256);
    float row_scale = tensor->requantizeRowToInt8(0, 256, output.data());

    EXPECT_GT(row_scale, 0.0f);
    EXPECT_LE(row_scale, 1.01f); // scale = max_abs/127 ≤ 1.0

    // Should be a direct copy when all values are in [-127, 127]
    for (int i = 0; i < 256; ++i)
        EXPECT_EQ(output[i], blk.qs[i])
            << "Q8_K direct copy mismatch at index " << i;
}

TEST_F(Test__RequantizeRowToInt8, Q8_K_WithNeg128_Rescales)
{
    // Insert -128 values → must rescale to fit [-127, 127]
    Q8_KBlock blk{};
    std::fill_n(blk.qs, 256, static_cast<int8_t>(0));
    blk.qs[0] = -128;
    blk.qs[100] = 127;
    blk.qs[200] = -128;
    std::vector<Q8_KBlock> blocks = {blk};
    auto tensor = makeQ8_KTensor(1, 256, blocks);

    std::vector<int8_t> output(256);
    float row_scale = tensor->requantizeRowToInt8(0, 256, output.data());

    // row_scale should be 128/127
    EXPECT_NEAR(row_scale, 128.0f / 127.0f, 1e-4f);

    // -128 should be rescaled to approximately -127
    EXPECT_GE(output[0], -127);
    EXPECT_LE(output[0], -126); // round(-128 * 127/128) = round(-126.0078) = -126

    // All values should be in [-127, 127]
    for (int i = 0; i < 256; ++i)
    {
        EXPECT_GE(output[i], -127);
        EXPECT_LE(output[i], 127);
    }
}

TEST_F(Test__RequantizeRowToInt8, Q8_K_MultiSuperblock)
{
    // Multiple superblocks per row (K=512 → 2 superblocks)
    const size_t K = 512;
    std::vector<Q8_KBlock> blocks;
    blocks.push_back(randomQ8_KBlock(false));
    blocks.push_back(randomQ8_KBlock(false));
    auto tensor = makeQ8_KTensor(1, K, blocks);

    std::vector<int8_t> output(K);
    float row_scale = tensor->requantizeRowToInt8(0, K, output.data());

    auto ref = dequantQ8_KRow(blocks.data(), 2);
    auto recon = dequantInt8Row(output.data(), K, row_scale);

    double cosim = cosineSimilarity(ref, recon);
    EXPECT_GT(cosim, 0.999)
        << "Q8_K multi-superblock cosine similarity too low: " << cosim;
}

TEST_F(Test__RequantizeRowToInt8, Q8_K_ZeroRow)
{
    Q8_KBlock blk{};
    std::memset(&blk, 0, sizeof(Q8_KBlock));
    std::vector<Q8_KBlock> blocks = {blk};
    auto tensor = makeQ8_KTensor(1, 256, blocks);

    std::vector<int8_t> output(256);
    float row_scale = tensor->requantizeRowToInt8(0, 256, output.data());

    EXPECT_EQ(row_scale, 1.0f);
    for (int i = 0; i < 256; ++i)
        EXPECT_EQ(output[i], 0);
}

// ============================================================================
// Cross-format: default implementation vs override consistency
// ============================================================================

TEST_F(Test__RequantizeRowToInt8, Q8_0_OverrideMatchesDefaultImpl)
{
    // Compare the Q8_0 SIMD override against a manual scalar implementation
    // (which mirrors the IINT8Unpackable default) to verify SIMD correctness.
    const size_t blocks_per_row = 16;
    const size_t K = blocks_per_row * 32;
    std::vector<Q8_0Block> blocks;
    for (size_t i = 0; i < blocks_per_row; ++i)
        blocks.push_back(randomQ8_0Block(0.01f, 2.0f));
    auto tensor = makeQ8_0Tensor(1, K, blocks);

    // Get SIMD result
    std::vector<int8_t> simd_out(K);
    float simd_scale = tensor->requantizeRowToInt8(0, K, simd_out.data());

    // Compute scalar reference independently
    const Q8_0Block *row_blocks = reinterpret_cast<const Q8_0Block *>(
        tensor->raw_data());

    float max_abs = 0.0f;
    for (size_t b = 0; b < blocks_per_row; ++b)
    {
        const float block_scale = fp16_to_fp32(row_blocks[b].d);
        for (int i = 0; i < 32; ++i)
        {
            float val = std::abs(static_cast<float>(row_blocks[b].qs[i]) * block_scale);
            max_abs = std::max(max_abs, val);
        }
    }
    const float ref_scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    const float inv_ref_scale = 1.0f / ref_scale;

    std::vector<int8_t> ref_out(K);
    for (size_t b = 0; b < blocks_per_row; ++b)
    {
        const float rescale = fp16_to_fp32(row_blocks[b].d) * inv_ref_scale;
        for (int i = 0; i < 32; ++i)
        {
            float val = static_cast<float>(row_blocks[b].qs[i]) * rescale;
            int32_t q = static_cast<int32_t>(std::round(val));
            q = std::clamp(q, -127, 127);
            ref_out[b * 32 + i] = static_cast<int8_t>(q);
        }
    }

    // Scales must match exactly (same FP16→FP32 conversion, same max logic)
    EXPECT_EQ(simd_scale, ref_scale)
        << "SIMD scale " << simd_scale << " != scalar scale " << ref_scale;

    // Allow ±1 quantization difference due to rounding mode differences
    // (SIMD uses _mm256_cvtps_epi32 which is round-to-nearest-even,
    //  scalar uses manual (val + 0.5) truncation)
    int max_diff = 0;
    int num_diff = 0;
    for (size_t i = 0; i < K; ++i)
    {
        int diff = std::abs(static_cast<int>(simd_out[i]) - static_cast<int>(ref_out[i]));
        if (diff > 0)
            num_diff++;
        max_diff = std::max(max_diff, diff);
    }
    EXPECT_LE(max_diff, 1)
        << "Max quantization difference between SIMD and scalar: " << max_diff
        << " (" << num_diff << " elements differ out of " << K << ")";
}

TEST_F(Test__RequantizeRowToInt8, Q8_1_OverrideMatchesDefaultImpl)
{
    const size_t blocks_per_row = 16;
    const size_t K = blocks_per_row * 32;
    std::vector<Q8_1Block> blocks;
    for (size_t i = 0; i < blocks_per_row; ++i)
        blocks.push_back(randomQ8_1Block(0.01f, 2.0f));
    auto tensor = makeQ8_1Tensor(1, K, blocks);

    std::vector<int8_t> simd_out(K);
    float simd_scale = tensor->requantizeRowToInt8(0, K, simd_out.data());

    // Independent scalar reference
    const Q8_1Block *row_blocks = reinterpret_cast<const Q8_1Block *>(
        tensor->raw_data());

    float max_abs = 0.0f;
    for (size_t b = 0; b < blocks_per_row; ++b)
    {
        const float block_scale = fp16_to_fp32(row_blocks[b].d);
        for (int i = 0; i < 32; ++i)
        {
            float val = std::abs(static_cast<float>(row_blocks[b].qs[i]) * block_scale);
            max_abs = std::max(max_abs, val);
        }
    }
    const float ref_scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    const float inv_ref_scale = 1.0f / ref_scale;

    std::vector<int8_t> ref_out(K);
    for (size_t b = 0; b < blocks_per_row; ++b)
    {
        const float rescale = fp16_to_fp32(row_blocks[b].d) * inv_ref_scale;
        for (int i = 0; i < 32; ++i)
        {
            float val = static_cast<float>(row_blocks[b].qs[i]) * rescale;
            int32_t q = static_cast<int32_t>(std::round(val));
            q = std::clamp(q, -127, 127);
            ref_out[b * 32 + i] = static_cast<int8_t>(q);
        }
    }

    EXPECT_EQ(simd_scale, ref_scale);

    int max_diff = 0;
    for (size_t i = 0; i < K; ++i)
    {
        int diff = std::abs(static_cast<int>(simd_out[i]) - static_cast<int>(ref_out[i]));
        max_diff = std::max(max_diff, diff);
    }
    EXPECT_LE(max_diff, 1)
        << "Max Q8_1 SIMD vs scalar difference: " << max_diff;
}
