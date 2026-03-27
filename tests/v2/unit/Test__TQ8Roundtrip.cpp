/**
 * @file Test__TQ8Roundtrip.cpp
 * @brief Unit tests for TQ8 quantize → dequantize round-trip
 *
 * Tests the 8-bit TurboQuant quantization path:
 *   FP32 → normalize → rotate → 8-bit Lloyd-Max quantize (scalar-full)
 *        → dequantize → inverse rotate → rescale
 *
 * Validates:
 * - Scalar-full quality (cosine similarity, MSE) for D=64 and D=128
 * - TQ8 vs TQ4 quality comparison (TQ8 must be strictly better)
 * - Determinism: same input + seed → same output
 * - Zero vector handling, extreme norms
 * - Inner product preservation quality
 * - TQ8Tensor high-level API round-trip
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "kernels/cpu/turboquant/TurboQuantCodebook.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/cpu/turboquant/TurboQuantRotation.h"
#include "kernels/cpu/turboquant/TurboQuantQuantizeTQ8.h"
#include "kernels/cpu/turboquant/TurboQuantDequantizeTQ8.h"
#include "kernels/cpu/turboquant/TurboQuantQuantizeTQ4.h"
#include "kernels/cpu/turboquant/TurboQuantDequantizeTQ4.h"
#include <cmath>
#include <random>
#include <vector>
#include <numeric>
#include <iostream>

using namespace llaminar2;

namespace
{

    float compute_mse(const float *a, const float *b, int n)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum / n;
    }

    float compute_norm(const float *x, int n)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
            sum += x[i] * x[i];
        return std::sqrt(sum);
    }

    float compute_cosine(const float *a, const float *b, int n)
    {
        float dot = 0.0f, na = 0.0f, nb = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na < 1e-30f || nb < 1e-30f)
            return 0.0f;
        return dot / (std::sqrt(na) * std::sqrt(nb));
    }

    float compute_dot(const float *a, const float *b, int n)
    {
        float dot = 0.0f;
        for (int i = 0; i < n; ++i)
            dot += a[i] * b[i];
        return dot;
    }

} // anonymous namespace

// ============================================================================
// TQ8 Block-level round-trip quality
// ============================================================================

TEST(Test__TQ8Roundtrip, ScalarFull_Quality_64)
{
    constexpr int D = 64;
    constexpr int N = 200;
    TurboQuantContext ctx(D);
    const auto &head_ctx = ctx.for_layer(0);

    std::mt19937 rng(77);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    double total_cosine = 0.0;
    double total_mse = 0.0;

    for (int trial = 0; trial < N; ++trial)
    {
        alignas(64) float input[D], output[D], scratch0[D], scratch1[D];
        for (int i = 0; i < D; ++i)
            input[i] = dist(rng);

        TQ8Block_64 block;
        turboquant_quantize_tq8<D>(input, head_ctx, block, scratch0, scratch1);
        turboquant_dequantize_tq8<D>(block, head_ctx, output, scratch0);

        total_cosine += compute_cosine(input, output, D);
        total_mse += compute_mse(input, output, D);
    }

    double avg_cosine = total_cosine / N;
    double avg_mse = total_mse / N;
    std::cout << "TQ8 D=64: avg cosine=" << avg_cosine
              << " avg MSE=" << avg_mse << " (over " << N << " vectors)" << std::endl;

    // TQ8 should be substantially better than TQ4's thresholds (cosine>0.95, MSE<0.05)
    EXPECT_GT(avg_cosine, 0.999) << "TQ8 D=64 cosine unexpectedly low";
    EXPECT_LT(avg_mse, 0.005) << "TQ8 D=64 MSE unexpectedly high";
}

TEST(Test__TQ8Roundtrip, ScalarFull_Quality_128)
{
    constexpr int D = 128;
    constexpr int N = 200;
    TurboQuantContext ctx(D);
    const auto &head_ctx = ctx.for_layer(0);

    std::mt19937 rng(88);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    double total_cosine = 0.0;
    double total_mse = 0.0;

    for (int trial = 0; trial < N; ++trial)
    {
        alignas(64) float input[D], output[D], scratch0[D], scratch1[D];
        for (int i = 0; i < D; ++i)
            input[i] = dist(rng);

        TQ8Block_128 block;
        turboquant_quantize_tq8<D>(input, head_ctx, block, scratch0, scratch1);
        turboquant_dequantize_tq8<D>(block, head_ctx, output, scratch0);

        total_cosine += compute_cosine(input, output, D);
        total_mse += compute_mse(input, output, D);
    }

    double avg_cosine = total_cosine / N;
    double avg_mse = total_mse / N;
    std::cout << "TQ8 D=128: avg cosine=" << avg_cosine
              << " avg MSE=" << avg_mse << " (over " << N << " vectors)" << std::endl;

    EXPECT_GT(avg_cosine, 0.999) << "TQ8 D=128 cosine unexpectedly low";
    EXPECT_LT(avg_mse, 0.005) << "TQ8 D=128 MSE unexpectedly high";
}

// ============================================================================
// TQ8 vs TQ4 quality comparison: TQ8 must be strictly better
// ============================================================================

TEST(Test__TQ8Roundtrip, TQ8_StrictlyBetterThan_TQ4_64)
{
    constexpr int D = 64;
    constexpr int N = 500;
    TurboQuantContext ctx(D);
    const auto &head_ctx = ctx.for_layer(0);

    std::mt19937 rng(111);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    double total_mse_tq4 = 0.0, total_mse_tq8 = 0.0;
    double total_cos_tq4 = 0.0, total_cos_tq8 = 0.0;

    for (int trial = 0; trial < N; ++trial)
    {
        alignas(64) float input[D], out4[D], out8[D], s0[D], s1[D];
        for (int i = 0; i < D; ++i)
            input[i] = dist(rng);

        TQ4Block_64 block4;
        turboquant_quantize_tq4<D>(input, head_ctx, block4, s0, s1);
        turboquant_dequantize_tq4<D>(block4, head_ctx, out4, s0);

        TQ8Block_64 block8;
        turboquant_quantize_tq8<D>(input, head_ctx, block8, s0, s1);
        turboquant_dequantize_tq8<D>(block8, head_ctx, out8, s0);

        total_mse_tq4 += compute_mse(input, out4, D);
        total_mse_tq8 += compute_mse(input, out8, D);
        total_cos_tq4 += compute_cosine(input, out4, D);
        total_cos_tq8 += compute_cosine(input, out8, D);
    }

    double avg_mse_tq4 = total_mse_tq4 / N;
    double avg_mse_tq8 = total_mse_tq8 / N;
    double avg_cos_tq4 = total_cos_tq4 / N;
    double avg_cos_tq8 = total_cos_tq8 / N;

    std::cout << "D=64 TQ4: avg_mse=" << avg_mse_tq4 << " avg_cos=" << avg_cos_tq4 << std::endl;
    std::cout << "D=64 TQ8: avg_mse=" << avg_mse_tq8 << " avg_cos=" << avg_cos_tq8 << std::endl;
    std::cout << "D=64 TQ8/TQ4 MSE ratio: " << avg_mse_tq8 / avg_mse_tq4 << std::endl;

    EXPECT_LT(avg_mse_tq8, avg_mse_tq4) << "TQ8 MSE must be lower than TQ4";
    EXPECT_GT(avg_cos_tq8, avg_cos_tq4) << "TQ8 cosine must be higher than TQ4";

    // TQ8 should have ~70× lower MSE than TQ4 (0.000132 vs 0.009501 per-element)
    EXPECT_LT(avg_mse_tq8 / avg_mse_tq4, 0.1) << "TQ8 should have >10× lower MSE than TQ4";
}

TEST(Test__TQ8Roundtrip, TQ8_StrictlyBetterThan_TQ4_128)
{
    constexpr int D = 128;
    constexpr int N = 500;
    TurboQuantContext ctx(D);
    const auto &head_ctx = ctx.for_layer(0);

    std::mt19937 rng(222);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    double total_mse_tq4 = 0.0, total_mse_tq8 = 0.0;
    double total_cos_tq4 = 0.0, total_cos_tq8 = 0.0;

    for (int trial = 0; trial < N; ++trial)
    {
        alignas(64) float input[D], out4[D], out8[D], s0[D], s1[D];
        for (int i = 0; i < D; ++i)
            input[i] = dist(rng);

        TQ4Block_128 block4;
        turboquant_quantize_tq4<D>(input, head_ctx, block4, s0, s1);
        turboquant_dequantize_tq4<D>(block4, head_ctx, out4, s0);

        TQ8Block_128 block8;
        turboquant_quantize_tq8<D>(input, head_ctx, block8, s0, s1);
        turboquant_dequantize_tq8<D>(block8, head_ctx, out8, s0);

        total_mse_tq4 += compute_mse(input, out4, D);
        total_mse_tq8 += compute_mse(input, out8, D);
        total_cos_tq4 += compute_cosine(input, out4, D);
        total_cos_tq8 += compute_cosine(input, out8, D);
    }

    double avg_mse_tq4 = total_mse_tq4 / N;
    double avg_mse_tq8 = total_mse_tq8 / N;
    double avg_cos_tq4 = total_cos_tq4 / N;
    double avg_cos_tq8 = total_cos_tq8 / N;

    std::cout << "D=128 TQ4: avg_mse=" << avg_mse_tq4 << " avg_cos=" << avg_cos_tq4 << std::endl;
    std::cout << "D=128 TQ8: avg_mse=" << avg_mse_tq8 << " avg_cos=" << avg_cos_tq8 << std::endl;
    std::cout << "D=128 TQ8/TQ4 MSE ratio: " << avg_mse_tq8 / avg_mse_tq4 << std::endl;

    EXPECT_LT(avg_mse_tq8, avg_mse_tq4) << "TQ8 MSE must be lower than TQ4";
    EXPECT_GT(avg_cos_tq8, avg_cos_tq4) << "TQ8 cosine must be higher than TQ4";
    EXPECT_LT(avg_mse_tq8 / avg_mse_tq4, 0.1) << "TQ8 should have >10× lower MSE than TQ4";
}

// ============================================================================
// Determinism
// ============================================================================

TEST(Test__TQ8Roundtrip, Deterministic_SameInputSameSeed)
{
    constexpr int D = 128;
    TurboQuantContext ctx(D, /*rotation_seed=*/42, /*projection_seed=*/42);
    const auto &head_ctx = ctx.for_layer(0);

    std::mt19937 rng(333);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    alignas(64) float input[D], out1[D], out2[D], s0[D], s1[D];
    for (int i = 0; i < D; ++i)
        input[i] = dist(rng);

    TQ8Block_128 block1, block2;
    turboquant_quantize_tq8<D>(input, head_ctx, block1, s0, s1);
    turboquant_quantize_tq8<D>(input, head_ctx, block2, s0, s1);

    // Blocks must be bit-identical
    EXPECT_FLOAT_EQ(block1.norm, block2.norm);
    EXPECT_FLOAT_EQ(block1.residual_norm, block2.residual_norm);
    for (int i = 0; i < D; ++i)
        EXPECT_EQ(block1.indices[i], block2.indices[i]) << "Index mismatch at " << i;

    // Dequantized output must be bit-identical
    turboquant_dequantize_tq8<D>(block1, head_ctx, out1, s0);
    turboquant_dequantize_tq8<D>(block2, head_ctx, out2, s0);
    for (int i = 0; i < D; ++i)
        EXPECT_FLOAT_EQ(out1[i], out2[i]) << "Output mismatch at " << i;
}

// ============================================================================
// Extreme norms
// ============================================================================

TEST(Test__TQ8Roundtrip, ExtremeNorms_NoNaN)
{
    constexpr int D = 128;
    TurboQuantContext ctx(D);
    const auto &head_ctx = ctx.for_layer(0);

    std::mt19937 rng(999);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Generate unit direction
    alignas(64) float dir[D];
    for (int i = 0; i < D; ++i)
        dir[i] = dist(rng);
    float norm = compute_norm(dir, D);
    for (int i = 0; i < D; ++i)
        dir[i] /= norm;

    float scales[] = {1e-6f, 1e-3f, 1e+3f, 1e+6f};

    for (float scale : scales)
    {
        alignas(64) float input[D], output[D], s0[D], s1[D];
        for (int i = 0; i < D; ++i)
            input[i] = dir[i] * scale;

        TQ8Block_128 block;
        turboquant_quantize_tq8<D>(input, head_ctx, block, s0, s1);
        turboquant_dequantize_tq8<D>(block, head_ctx, output, s0);

        for (int i = 0; i < D; ++i)
        {
            ASSERT_FALSE(std::isnan(output[i])) << "NaN at scale=" << scale << " index=" << i;
            ASSERT_FALSE(std::isinf(output[i])) << "Inf at scale=" << scale << " index=" << i;
        }

        float cos = compute_cosine(input, output, D);
        EXPECT_GT(cos, 0.99) << "Cosine too low at scale=" << scale << ": " << cos;
    }
}

TEST(Test__TQ8Roundtrip, ZeroVector_HandledGracefully)
{
    constexpr int D = 64;
    TurboQuantContext ctx(D);
    const auto &head_ctx = ctx.for_layer(0);

    alignas(64) float input[D], output[D], s0[D], s1[D];
    std::fill(input, input + D, 0.0f);

    TQ8Block_64 block;
    turboquant_quantize_tq8<D>(input, head_ctx, block, s0, s1);
    turboquant_dequantize_tq8<D>(block, head_ctx, output, s0);

    for (int i = 0; i < D; ++i)
    {
        ASSERT_FALSE(std::isnan(output[i])) << "NaN in zero-vector output at " << i;
        ASSERT_FALSE(std::isinf(output[i])) << "Inf in zero-vector output at " << i;
    }

    // Dequantized should be near-zero
    float out_norm = compute_norm(output, D);
    EXPECT_LT(out_norm, 1e-5f) << "Zero vector output norm should be near-zero: " << out_norm;
}

// ============================================================================
// Inner product preservation (the metric that matters for attention)
// ============================================================================

TEST(Test__TQ8Roundtrip, InnerProduct_Preservation_128)
{
    constexpr int D = 128;
    constexpr int N_PAIRS = 1000;
    TurboQuantContext ctx(D);
    const auto &head_ctx = ctx.for_layer(0);

    std::mt19937 rng(444);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    double total_ip_error = 0.0;
    double total_true_ip_sq = 0.0;

    for (int trial = 0; trial < N_PAIRS; ++trial)
    {
        alignas(64) float q[D], k[D], k_hat[D], s0[D], s1[D];
        for (int i = 0; i < D; ++i)
        {
            q[i] = dist(rng);
            k[i] = dist(rng);
        }

        // Quantize K only (Q stays FP32 — matches real attention pattern)
        TQ8Block_128 block;
        turboquant_quantize_tq8<D>(k, head_ctx, block, s0, s1);
        turboquant_dequantize_tq8<D>(block, head_ctx, k_hat, s0);

        float true_dot = compute_dot(q, k, D);
        float approx_dot = compute_dot(q, k_hat, D);
        total_ip_error += (approx_dot - true_dot) * (approx_dot - true_dot);
        total_true_ip_sq += true_dot * true_dot;
    }

    double rmse_ip = std::sqrt(total_ip_error / N_PAIRS);
    double rms_true = std::sqrt(total_true_ip_sq / N_PAIRS);
    double relative_ip_error = rmse_ip / rms_true;

    std::cout << "TQ8 D=128 inner product: RMSE=" << rmse_ip
              << " RMS(true)=" << rms_true
              << " relative error=" << relative_ip_error << std::endl;

    // TQ8 inner product error should be very small (< 2% relative)
    EXPECT_LT(relative_ip_error, 0.02) << "TQ8 inner product relative error too high";
}

// ============================================================================
// TQ8Tensor high-level API tests
// ============================================================================

TEST(Test__TQ8Roundtrip, TQ8Tensor_QuantizeAndDequantize_64)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int N_TOKENS = 8;

    TurboQuantContext ctx(HEAD_DIM, 42, 42);

    std::mt19937 rng(555);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> src(N_TOKENS * KV_DIM);
    for (auto &v : src)
        v = dist(rng);

    auto tensor = TQ8Tensor::quantize_from_fp32(
        src.data(),
        {static_cast<size_t>(N_TOKENS), static_cast<size_t>(KV_DIM)},
        HEAD_DIM, ctx);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rows(), static_cast<size_t>(N_TOKENS));
    EXPECT_EQ(tensor->cols(), static_cast<size_t>(KV_DIM));
    EXPECT_EQ(tensor->head_dim(), HEAD_DIM);
    EXPECT_EQ(tensor->blocks_per_row(), static_cast<size_t>(N_KV_HEADS));
    EXPECT_EQ(tensor->native_type(), TensorType::TQ8);

    // Dequantize and check quality
    std::vector<float> dst(N_TOKENS * KV_DIM);
    tensor->dequantize_to_fp32(dst.data(), ctx);

    double total_cosine = 0.0;
    for (int r = 0; r < N_TOKENS; ++r)
    {
        float cos = compute_cosine(src.data() + r * KV_DIM, dst.data() + r * KV_DIM, KV_DIM);
        total_cosine += cos;
        EXPECT_GT(cos, 0.99) << "Row " << r << " cosine too low: " << cos;
    }
    std::cout << "TQ8Tensor D=64: avg row cosine=" << total_cosine / N_TOKENS << std::endl;
}

TEST(Test__TQ8Roundtrip, TQ8Tensor_QuantizeAndDequantize_128)
{
    constexpr int HEAD_DIM = 128;
    constexpr int N_KV_HEADS = 4;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int N_TOKENS = 16;

    TurboQuantContext ctx(HEAD_DIM, 42, 42);

    std::mt19937 rng(666);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> src(N_TOKENS * KV_DIM);
    for (auto &v : src)
        v = dist(rng);

    auto tensor = TQ8Tensor::quantize_from_fp32(
        src.data(),
        {static_cast<size_t>(N_TOKENS), static_cast<size_t>(KV_DIM)},
        HEAD_DIM, ctx);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rows(), static_cast<size_t>(N_TOKENS));
    EXPECT_EQ(tensor->cols(), static_cast<size_t>(KV_DIM));

    std::vector<float> dst(N_TOKENS * KV_DIM);
    tensor->dequantize_to_fp32(dst.data(), ctx);

    double total_cosine = 0.0;
    for (int r = 0; r < N_TOKENS; ++r)
    {
        float cos = compute_cosine(src.data() + r * KV_DIM, dst.data() + r * KV_DIM, KV_DIM);
        total_cosine += cos;
        EXPECT_GT(cos, 0.99) << "Row " << r << " cosine too low: " << cos;
    }
    std::cout << "TQ8Tensor D=128: avg row cosine=" << total_cosine / N_TOKENS << std::endl;
}

TEST(Test__TQ8Roundtrip, TQ8Tensor_CopyFromFP32Rows)
{
    constexpr int HEAD_DIM = 128;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 32;
    constexpr int N_TOKENS = 5;

    TurboQuantContext ctx(HEAD_DIM, 42, 42);

    std::mt19937 rng(777);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> src(N_TOKENS * KV_DIM);
    for (auto &v : src)
        v = dist(rng);

    // Pre-allocate tensor with MAX_SEQ capacity, fill only N_TOKENS rows
    auto tensor = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)},
        HEAD_DIM);

    ASSERT_TRUE(tensor->copyFrom_fp32_rows(src.data(), N_TOKENS, ctx));

    // Dequantize first N_TOKENS rows via to_fp32_row
    std::vector<float> dst(N_TOKENS * KV_DIM);
    tensor->set_turboquant_context(&ctx);
    for (int r = 0; r < N_TOKENS; ++r)
        tensor->to_fp32_row(r, dst.data() + r * KV_DIM);

    for (int r = 0; r < N_TOKENS; ++r)
    {
        float cos = compute_cosine(src.data() + r * KV_DIM, dst.data() + r * KV_DIM, KV_DIM);
        EXPECT_GT(cos, 0.99) << "copyFrom_fp32_rows row " << r << " cosine too low: " << cos;
    }
}

TEST(Test__TQ8Roundtrip, TQ8Tensor_Data_LazyDequant)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int N_TOKENS = 4;

    TurboQuantContext ctx(HEAD_DIM, 42, 42);

    std::mt19937 rng(888);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> src(N_TOKENS * KV_DIM);
    for (auto &v : src)
        v = dist(rng);

    auto tensor = TQ8Tensor::quantize_from_fp32(
        src.data(),
        {static_cast<size_t>(N_TOKENS), static_cast<size_t>(KV_DIM)},
        HEAD_DIM, ctx);

    // data() without context should return nullptr
    tensor->set_turboquant_context(nullptr);
    EXPECT_EQ(tensor->data(), nullptr);

    // data() with context should return valid FP32 dequantized data
    tensor->set_turboquant_context(&ctx);
    const float *fp32 = tensor->data();
    ASSERT_NE(fp32, nullptr);

    for (int r = 0; r < N_TOKENS; ++r)
    {
        float cos = compute_cosine(src.data() + r * KV_DIM, fp32 + r * KV_DIM, KV_DIM);
        EXPECT_GT(cos, 0.99) << "data() row " << r << " cosine too low: " << cos;
    }
}

TEST(Test__TQ8Roundtrip, TQ8Tensor_CopyFrom_TQ8)
{
    constexpr int HEAD_DIM = 128;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int N_TOKENS = 4;

    TurboQuantContext ctx(HEAD_DIM, 42, 42);

    std::mt19937 rng(999);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> src(N_TOKENS * KV_DIM);
    for (auto &v : src)
        v = dist(rng);

    auto orig = TQ8Tensor::quantize_from_fp32(
        src.data(),
        {static_cast<size_t>(N_TOKENS), static_cast<size_t>(KV_DIM)},
        HEAD_DIM, ctx);

    auto copy = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(N_TOKENS), static_cast<size_t>(KV_DIM)},
        HEAD_DIM);

    ASSERT_TRUE(copy->copyFrom(orig.get()));

    // Raw bytes must be identical
    ASSERT_EQ(orig->size_bytes(), copy->size_bytes());
    EXPECT_EQ(std::memcmp(orig->raw_data(), copy->raw_data(), orig->size_bytes()), 0)
        << "copyFrom should produce identical raw block data";
}

TEST(Test__TQ8Roundtrip, TQ8Tensor_BlockSizes)
{
    // Verify block size calculations
    auto t64 = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{1, 128}, 64);
    EXPECT_EQ(t64->block_bytes(), sizeof(TQ8Block_64));
    EXPECT_EQ(t64->blocks_per_row(), 2u); // 128 / 64

    auto t128 = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{1, 512}, 128);
    EXPECT_EQ(t128->block_bytes(), sizeof(TQ8Block_128));
    EXPECT_EQ(t128->blocks_per_row(), 4u); // 512 / 128

    // TQ8Block_128 = 8 (two floats) + 128 (indices) = 136 bytes
    EXPECT_EQ(sizeof(TQ8Block_128), 136u);
    // TQ8Block_64 = 8 (two floats) + 64 (indices) = 72 bytes
    EXPECT_EQ(sizeof(TQ8Block_64), 72u);
}
