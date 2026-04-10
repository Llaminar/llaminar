/**
 * @file Test__SIMDHelpersISAParity.cpp
 * @brief ISA parity tests verifying that _scalar, _avx2, and _avx512 variants
 *        of SIMDHelpers.h functions produce identical (or near-identical) results.
 *
 * For each function family: create random input → call all three variants →
 * compare outputs with appropriate tolerance.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

#include "tensors/SIMDHelpers.h"
#include "tensors/BlockStructures.h"

using namespace llaminar2::simd;
using namespace llaminar2;

// ============================================================================
// Helpers
// ============================================================================

static std::vector<float> rand_fp32(size_t n, float lo = -2.0f, float hi = 2.0f,
                                    uint32_t seed = 42)
{
    std::vector<float> v(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto &x : v)
        x = dist(rng);
    return v;
}

static float max_abs_diff(const float *a, const float *b, size_t n)
{
    float mx = 0.0f;
    for (size_t i = 0; i < n; ++i)
        mx = std::max(mx, std::abs(a[i] - b[i]));
    return mx;
}

/// Compare two Q8_1Block arrays by dequantizing to FP32 and checking element-wise
static float max_abs_diff_q8_1(const Q8_1Block *a, const Q8_1Block *b,
                               size_t n_elements)
{
    std::vector<float> fa(n_elements), fb(n_elements);
    dequantize_q8_1_to_fp32_scalar(a, fa.data(), n_elements);
    dequantize_q8_1_to_fp32_scalar(b, fb.data(), n_elements);
    return max_abs_diff(fa.data(), fb.data(), n_elements);
}

/// Compare two Q16_1Block arrays by dequantizing inline
static float max_abs_diff_q16_1(const Q16_1Block *a, const Q16_1Block *b,
                                size_t n_elements)
{
    size_t n_blocks = n_elements / 32;
    float mx = 0.0f;
    for (size_t blk = 0; blk < n_blocks; ++blk)
    {
        for (int i = 0; i < 32; ++i)
        {
            float va = a[blk].d * static_cast<float>(a[blk].qs[i]);
            float vb = b[blk].d * static_cast<float>(b[blk].qs[i]);
            mx = std::max(mx, std::abs(va - vb));
        }
    }
    return mx;
}

/// Create random Q8_1 blocks by quantizing random FP32
static void make_random_q8_1(Q8_1Block *blocks, size_t n_elements,
                             uint32_t seed = 42)
{
    auto fp = rand_fp32(n_elements, -1.5f, 1.5f, seed);
    quantize_fp32_to_q8_1_blocks_scalar(fp.data(), blocks, n_elements);
}

/// Create random Q16_1 blocks
static void make_random_q16_1(Q16_1Block *blocks, size_t n_elements,
                              uint32_t seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    size_t n_blocks = n_elements / 32;
    for (size_t b = 0; b < n_blocks; ++b)
    {
        // Generate random FP32 values, quantize to Q16_1
        float max_abs = 0.0f;
        float vals[32];
        for (int i = 0; i < 32; ++i)
        {
            vals[i] = dist(rng);
            max_abs = std::max(max_abs, std::abs(vals[i]));
        }
        float scale = max_abs / 32767.0f;
        if (scale < 1e-10f)
            scale = 1e-10f;
        blocks[b].d = scale;
        int32_t sum = 0;
        for (int i = 0; i < 32; ++i)
        {
            int16_t q = static_cast<int16_t>(std::round(vals[i] / scale));
            q = std::max<int16_t>(-32767, std::min<int16_t>(32767, q));
            blocks[b].qs[i] = q;
            sum += q;
        }
        blocks[b].sum_qs = sum;
    }
}

// ============================================================================
// fused_fp32_residual_add
// ============================================================================

TEST(Test__SIMDHelpersISAParity, FusedFP32ResidualAdd)
{
    for (size_t count : {32u, 64u, 128u, 256u, 1024u})
    {
        auto residual = rand_fp32(count, -1.0f, 1.0f, 100);
        auto input = rand_fp32(count, -1.0f, 1.0f, 200);
        std::vector<float> out_scalar(count), out_avx2(count), out_avx512(count);

        fused_fp32_residual_add_scalar(residual.data(), input.data(), out_scalar.data(), count);

#if defined(__AVX2__)
        fused_fp32_residual_add_avx2(residual.data(), input.data(), out_avx2.data(), count);
        EXPECT_LE(max_abs_diff(out_scalar.data(), out_avx2.data(), count), 1e-6f)
            << "AVX2 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX2 not available";
#endif

#if defined(__AVX512F__)
        fused_fp32_residual_add_avx512(residual.data(), input.data(), out_avx512.data(), count);
        EXPECT_LE(max_abs_diff(out_scalar.data(), out_avx512.data(), count), 1e-6f)
            << "AVX512 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX512 not available";
#endif
    }
}

// ============================================================================
// fused_bf16_residual_add
// ============================================================================

TEST(Test__SIMDHelpersISAParity, FusedBF16ResidualAdd)
{
    for (size_t count : {32u, 64u, 128u, 256u})
    {
        auto fp_vals = rand_fp32(count, -1.0f, 1.0f, 300);
        std::vector<uint16_t> residual(count);
        for (size_t i = 0; i < count; ++i)
            residual[i] = fp32_to_bf16(fp_vals[i]);
        auto input = rand_fp32(count, -0.5f, 0.5f, 400);

        std::vector<float> out_scalar(count), out_avx2(count), out_avx512(count);

        fused_bf16_residual_add_scalar(residual.data(), input.data(), out_scalar.data(), count);

#if defined(__AVX2__)
        fused_bf16_residual_add_avx2(residual.data(), input.data(), out_avx2.data(), count);
        EXPECT_LE(max_abs_diff(out_scalar.data(), out_avx2.data(), count), 1e-6f)
            << "AVX2 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX2 not available";
#endif

#if defined(__AVX512F__)
        fused_bf16_residual_add_avx512(residual.data(), input.data(), out_avx512.data(), count);
        EXPECT_LE(max_abs_diff(out_scalar.data(), out_avx512.data(), count), 1e-6f)
            << "AVX512 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX512 not available";
#endif
    }
}

// ============================================================================
// fused_fp16_residual_add
// ============================================================================

TEST(Test__SIMDHelpersISAParity, FusedFP16ResidualAdd)
{
    for (size_t count : {32u, 64u, 128u, 256u})
    {
        auto fp_vals = rand_fp32(count, -1.0f, 1.0f, 500);
        std::vector<uint16_t> residual(count);
        for (size_t i = 0; i < count; ++i)
            residual[i] = fp32_to_fp16(fp_vals[i]);
        auto input = rand_fp32(count, -0.5f, 0.5f, 600);

        std::vector<float> out_scalar(count), out_avx2(count), out_avx512(count);

        fused_fp16_residual_add_scalar(residual.data(), input.data(), out_scalar.data(), count);

#if defined(__AVX2__) && defined(__F16C__)
        fused_fp16_residual_add_avx2(residual.data(), input.data(), out_avx2.data(), count);
        EXPECT_LE(max_abs_diff(out_scalar.data(), out_avx2.data(), count), 1e-5f)
            << "AVX2 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX2+F16C not available";
#endif

#if defined(__AVX512F__)
        fused_fp16_residual_add_avx512(residual.data(), input.data(), out_avx512.data(), count);
        EXPECT_LE(max_abs_diff(out_scalar.data(), out_avx512.data(), count), 1e-5f)
            << "AVX512 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX512 not available";
#endif
    }
}

// ============================================================================
// fused_q8_1_residual_add
// ============================================================================

TEST(Test__SIMDHelpersISAParity, FusedQ8_1ResidualAdd)
{
    for (size_t count : {32u, 64u, 128u, 256u})
    {
        size_t n_blocks = count / 32;
        std::vector<Q8_1Block> residual(n_blocks);
        make_random_q8_1(residual.data(), count, 700);
        auto input = rand_fp32(count, -0.5f, 0.5f, 800);

        std::vector<float> out_scalar(count), out_avx2(count), out_avx512(count);

        fused_q8_1_residual_add_scalar(residual.data(), input.data(), out_scalar.data(), count);

#if defined(__AVX2__)
        fused_q8_1_residual_add_avx2(residual.data(), input.data(), out_avx2.data(), count);
        EXPECT_LE(max_abs_diff(out_scalar.data(), out_avx2.data(), count), 1e-5f)
            << "AVX2 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX2 not available";
#endif

#if defined(__AVX512F__)
        fused_q8_1_residual_add_avx512(residual.data(), input.data(), out_avx512.data(), count);
        EXPECT_LE(max_abs_diff(out_scalar.data(), out_avx512.data(), count), 1e-5f)
            << "AVX512 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX512 not available";
#endif
    }
}

// ============================================================================
// q8_1_add_q8_1
// ============================================================================

TEST(Test__SIMDHelpersISAParity, Q8_1AddQ8_1)
{
    for (size_t count : {32u, 64u, 128u, 256u})
    {
        size_t n_blocks = count / 32;
        std::vector<Q8_1Block> a(n_blocks), b(n_blocks);
        make_random_q8_1(a.data(), count, 900);
        make_random_q8_1(b.data(), count, 1000);

        std::vector<Q8_1Block> out_scalar(n_blocks), out_avx2(n_blocks), out_avx512(n_blocks);

        q8_1_add_q8_1_scalar(a.data(), b.data(), out_scalar.data(), count);

#if defined(__AVX2__)
        q8_1_add_q8_1_avx2(a.data(), b.data(), out_avx2.data(), count);
        EXPECT_LE(max_abs_diff_q8_1(out_scalar.data(), out_avx2.data(), count), 1e-4f)
            << "AVX2 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX2 not available";
#endif

#if defined(__AVX512F__)
        q8_1_add_q8_1_avx512(a.data(), b.data(), out_avx512.data(), count);
        EXPECT_LE(max_abs_diff_q8_1(out_scalar.data(), out_avx512.data(), count), 1e-4f)
            << "AVX512 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX512 not available";
#endif
    }
}

// ============================================================================
// q16_1_add_q16_1
// ============================================================================

TEST(Test__SIMDHelpersISAParity, Q16_1AddQ16_1)
{
    for (size_t count : {32u, 64u, 128u, 256u})
    {
        size_t n_blocks = count / 32;
        std::vector<Q16_1Block> a(n_blocks), b(n_blocks);
        make_random_q16_1(a.data(), count, 1100);
        make_random_q16_1(b.data(), count, 1200);

        std::vector<Q16_1Block> out_scalar(n_blocks), out_avx2(n_blocks), out_avx512(n_blocks);

        q16_1_add_q16_1_scalar(a.data(), b.data(), out_scalar.data(), count);

#if defined(__AVX2__)
        q16_1_add_q16_1_avx2(a.data(), b.data(), out_avx2.data(), count);
        EXPECT_LE(max_abs_diff_q16_1(out_scalar.data(), out_avx2.data(), count), 1e-4f)
            << "AVX2 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX2 not available";
#endif

#if defined(__AVX512F__)
        q16_1_add_q16_1_avx512(a.data(), b.data(), out_avx512.data(), count);
        EXPECT_LE(max_abs_diff_q16_1(out_scalar.data(), out_avx512.data(), count), 1e-4f)
            << "AVX512 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX512 not available";
#endif
    }
}

// ============================================================================
// q16_1_add_fp32 (in-place)
// ============================================================================

TEST(Test__SIMDHelpersISAParity, Q16_1AddFP32)
{
    for (size_t count : {32u, 64u, 128u, 256u})
    {
        size_t n_blocks = count / 32;
        auto delta = rand_fp32(count, -0.5f, 0.5f, 1300);

        // Create three copies of the same Q16_1 data
        std::vector<Q16_1Block> base(n_blocks);
        make_random_q16_1(base.data(), count, 1400);

        auto r_scalar = base;
        auto r_avx2 = base;
        auto r_avx512 = base;

        q16_1_add_fp32_scalar(r_scalar.data(), delta.data(), count);

#if defined(__AVX2__)
        q16_1_add_fp32_avx2(r_avx2.data(), delta.data(), count);
        EXPECT_LE(max_abs_diff_q16_1(r_scalar.data(), r_avx2.data(), count), 1e-4f)
            << "AVX2 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX2 not available";
#endif

#if defined(__AVX512F__)
        q16_1_add_fp32_avx512(r_avx512.data(), delta.data(), count);
        EXPECT_LE(max_abs_diff_q16_1(r_scalar.data(), r_avx512.data(), count), 1e-4f)
            << "AVX512 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX512 not available";
#endif
    }
}

// ============================================================================
// q16_1_to_q8_1_packed
// ============================================================================

TEST(Test__SIMDHelpersISAParity, Q16_1ToQ8_1Packed)
{
    for (size_t n_blocks : {1u, 2u, 4u, 8u})
    {
        size_t count = n_blocks * 32;
        std::vector<Q16_1Block> src(n_blocks);
        make_random_q16_1(src.data(), count, 1500);

        std::vector<Q8_1Block> out_scalar(n_blocks), out_avx2(n_blocks), out_avx512(n_blocks);

        q16_1_to_q8_1_packed_scalar(src.data(), out_scalar.data(), n_blocks);

#if defined(__AVX2__)
        q16_1_to_q8_1_packed_avx2(src.data(), out_avx2.data(), n_blocks);
        EXPECT_LE(max_abs_diff_q8_1(out_scalar.data(), out_avx2.data(), count), 1e-4f)
            << "AVX2 vs scalar, n_blocks=" << n_blocks;
#else
        GTEST_SKIP() << "AVX2 not available";
#endif

#if defined(__AVX512F__)
        q16_1_to_q8_1_packed_avx512(src.data(), out_avx512.data(), n_blocks);
        EXPECT_LE(max_abs_diff_q8_1(out_scalar.data(), out_avx512.data(), count), 1e-4f)
            << "AVX512 vs scalar, n_blocks=" << n_blocks;
#else
        GTEST_SKIP() << "AVX512 not available";
#endif
    }
}

// ============================================================================
// q16_1_add_q8_1 (in-place)
// ============================================================================

TEST(Test__SIMDHelpersISAParity, Q16_1AddQ8_1)
{
    for (size_t count : {32u, 64u, 128u, 256u})
    {
        size_t n_blocks = count / 32;
        std::vector<Q8_1Block> delta(n_blocks);
        make_random_q8_1(delta.data(), count, 1600);

        std::vector<Q16_1Block> base(n_blocks);
        make_random_q16_1(base.data(), count, 1700);

        auto r_scalar = base;
        auto r_avx2 = base;
        auto r_avx512 = base;

        q16_1_add_q8_1_scalar(r_scalar.data(), delta.data(), count);

#if defined(__AVX2__)
        q16_1_add_q8_1_avx2(r_avx2.data(), delta.data(), count);
        EXPECT_LE(max_abs_diff_q16_1(r_scalar.data(), r_avx2.data(), count), 1e-4f)
            << "AVX2 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX2 not available";
#endif

#if defined(__AVX512F__)
        q16_1_add_q8_1_avx512(r_avx512.data(), delta.data(), count);
        EXPECT_LE(max_abs_diff_q16_1(r_scalar.data(), r_avx512.data(), count), 1e-4f)
            << "AVX512 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX512 not available";
#endif
    }
}

// ============================================================================
// dequantize_q8_1_to_fp32
// ============================================================================

TEST(Test__SIMDHelpersISAParity, DequantizeQ8_1ToFP32)
{
    for (size_t count : {32u, 64u, 128u, 256u})
    {
        size_t n_blocks = count / 32;
        std::vector<Q8_1Block> src(n_blocks);
        make_random_q8_1(src.data(), count, 1800);

        std::vector<float> out_scalar(count), out_avx2(count), out_avx512(count);

        dequantize_q8_1_to_fp32_scalar(src.data(), out_scalar.data(), count);

#if defined(__AVX2__)
        dequantize_q8_1_to_fp32_avx2(src.data(), out_avx2.data(), count);
        EXPECT_LE(max_abs_diff(out_scalar.data(), out_avx2.data(), count), 1e-6f)
            << "AVX2 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX2 not available";
#endif

#if defined(__AVX512F__)
        dequantize_q8_1_to_fp32_avx512(src.data(), out_avx512.data(), count);
        EXPECT_LE(max_abs_diff(out_scalar.data(), out_avx512.data(), count), 1e-6f)
            << "AVX512 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX512 not available";
#endif
    }
}

// ============================================================================
// quantize_fp32_to_q8_1_blocks
// ============================================================================

TEST(Test__SIMDHelpersISAParity, QuantizeFP32ToQ8_1Blocks)
{
    for (size_t count : {32u, 64u, 128u, 256u})
    {
        size_t n_blocks = count / 32;
        auto src = rand_fp32(count, -2.0f, 2.0f, 1900);

        std::vector<Q8_1Block> out_scalar(n_blocks), out_avx2(n_blocks), out_avx512(n_blocks);

        quantize_fp32_to_q8_1_blocks_scalar(src.data(), out_scalar.data(), count);

#if defined(__AVX2__)
        quantize_fp32_to_q8_1_blocks_avx2(src.data(), out_avx2.data(), count);
        // Requantization can differ by ±1 in qs due to rounding, so use looser tolerance
        EXPECT_LE(max_abs_diff_q8_1(out_scalar.data(), out_avx2.data(), count), 1e-3f)
            << "AVX2 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX2 not available";
#endif

#if defined(__AVX512F__)
        quantize_fp32_to_q8_1_blocks_avx512(src.data(), out_avx512.data(), count);
        EXPECT_LE(max_abs_diff_q8_1(out_scalar.data(), out_avx512.data(), count), 1e-3f)
            << "AVX512 vs scalar, count=" << count;
#else
        GTEST_SKIP() << "AVX512 not available";
#endif
    }
}
