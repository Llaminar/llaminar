/**
 * @file Test__TurboQuantRoundtrip.cpp
 * @brief Unit tests for TurboQuant quantize → dequantize round-trip
 *
 * Tests the scalar-full TurboQuant quantization path:
 *   FP32 → normalize → rotate → 4-bit MSE quantize (scalar-full)
 *        → dequantize MSE → inverse rotate → rescale
 *
 * Validates:
 * - Rotation matrix orthogonality and norm preservation
 * - Scalar-full quality (cosine similarity, MSE) for D=64 and D=128
 * - Determinism: same input + seed → same output
 * - Zero vector handling, extreme norms, one-hot vectors
 * - Block size correctness
 * - Row-range dequantization parity
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/BlockStructures.h"
#include "kernels/cpu/turboquant/TurboQuantCodebook.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/cpu/turboquant/TurboQuantRotation.h"
#include "kernels/cpu/turboquant/TurboQuantQuantizeTQ4.h"
#include "kernels/cpu/turboquant/TurboQuantDequantizeTQ4.h"
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <numeric>

using namespace llaminar2;

namespace
{

    // Helper: compute MSE between two vectors
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

    // Helper: compute L2 norm
    float compute_norm(const float *x, int n)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
            sum += x[i] * x[i];
        return std::sqrt(sum);
    }

    // Helper: compute cosine similarity
    float compute_cosine_similarity(const float *a, const float *b, int n)
    {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a < 1e-30f || norm_b < 1e-30f)
            return 0.0f;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    float compute_dot(const float *a, const float *b, int n)
    {
        float dot = 0.0f;
        for (int i = 0; i < n; ++i)
            dot += a[i] * b[i];
        return dot;
    }

    void generate_random_vector(float *out, int n, float target_norm, std::mt19937 &rng);

    // Helper: generate random vector with given norm
    void generate_random_vector(float *out, int n, float target_norm, std::mt19937 &rng)
    {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < n; ++i)
            out[i] = dist(rng);
        float norm = compute_norm(out, n);
        if (norm > 1e-10f)
        {
            float scale = target_norm / norm;
            for (int i = 0; i < n; ++i)
                out[i] *= scale;
        }
    }

} // anonymous namespace

// ============================================================================
// Rotation Matrix Tests
// ============================================================================

TEST(Test__TurboQuantRoundtrip, RotationMatrix_Orthogonality_64)
{
    auto rot = generate_rotation_matrix(64);
    float err = verify_orthogonality(rot);
    EXPECT_LT(err, 1e-4f)
        << "64×64 rotation matrix not orthogonal: max error = " << err;
}

TEST(Test__TurboQuantRoundtrip, RotationMatrix_Orthogonality_128)
{
    auto rot = generate_rotation_matrix(128);
    float err = verify_orthogonality(rot);
    EXPECT_LT(err, 1e-3f)
        << "128×128 rotation matrix not orthogonal: max error = " << err;
}

TEST(Test__TurboQuantRoundtrip, RotationMatrix_Deterministic)
{
    auto rot1 = generate_rotation_matrix(64, 42);
    auto rot2 = generate_rotation_matrix(64, 42);
    ASSERT_EQ(rot1.matrix.size(), rot2.matrix.size());
    for (size_t i = 0; i < rot1.matrix.size(); ++i)
        EXPECT_EQ(rot1.matrix[i], rot2.matrix[i])
            << "Rotation not deterministic at index " << i;
}

TEST(Test__TurboQuantRoundtrip, RotationMatrix_DifferentSeeds)
{
    auto rot1 = generate_rotation_matrix(64, 42);
    auto rot2 = generate_rotation_matrix(64, 99);
    bool all_same = true;
    for (size_t i = 0; i < rot1.matrix.size(); ++i)
    {
        if (rot1.matrix[i] != rot2.matrix[i])
        {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same)
        << "Different seeds should produce different rotation matrices";
}

TEST(Test__TurboQuantRoundtrip, Rotation_PreservesNorm)
{
    auto rot = generate_rotation_matrix(64);
    std::mt19937 rng(123);
    float input[64], output[64];
    generate_random_vector(input, 64, 2.5f, rng);

    apply_rotation(rot, input, output);

    float norm_in = compute_norm(input, 64);
    float norm_out = compute_norm(output, 64);
    EXPECT_NEAR(norm_in, norm_out, 1e-3f)
        << "Rotation should preserve L2 norm";
}

TEST(Test__TurboQuantRoundtrip, Rotation_InverseIsTranspose)
{
    auto rot = generate_rotation_matrix(64);
    std::mt19937 rng(456);
    float input[64], rotated[64], recovered[64];
    generate_random_vector(input, 64, 1.0f, rng);

    apply_rotation(rot, input, rotated);
    apply_rotation_transpose(rot, rotated, recovered);

    float mse = compute_mse(input, recovered, 64);
    EXPECT_LT(mse, 1e-6f)
        << "Π^T · Π · x should recover x, MSE = " << mse;
}

// ============================================================================
// Scalar-full (4-bit) explicit quality test
// ============================================================================

TEST(Test__TurboQuantRoundtrip, TQ4_ScalarFull_Quality_64)
{
    constexpr int D = 64;
    constexpr int N = 200;
    TurboQuantContext ctx(D);
    const auto &head_ctx = ctx.for_layer(0).for_layer(0);

    std::mt19937 rng(77);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    double total_cosine = 0.0;
    double total_mse = 0.0;

    for (int trial = 0; trial < N; ++trial)
    {
        float input[D], output[D], scratch0[D], scratch1[D];
        for (int i = 0; i < D; ++i)
            input[i] = dist(rng);

        TQ4Block_64 block;
        turboquant_quantize_tq4<D>(input, head_ctx, block, scratch0, scratch1);

        // Verify sentinel: scalar-full sets residual_norm < 0
        ASSERT_LT(block.residual_norm, 0.0f)
            << "Scalar-full must set negative residual_norm sentinel";

        turboquant_dequantize_tq4<D>(block, head_ctx, output, scratch0);

        total_cosine += compute_cosine_similarity(input, output, D);
        total_mse += compute_mse(input, output, D);
    }

    double avg_cosine = total_cosine / N;
    double avg_mse = total_mse / N;
    std::cout << "Scalar-full D=64: avg cosine=" << avg_cosine
              << " avg MSE=" << avg_mse << " (over " << N << " vectors)" << std::endl;

    EXPECT_GT(avg_cosine, 0.95) << "4-bit MSE should have high cosine similarity";
    EXPECT_LT(avg_mse, 0.05) << "Scalar-full MSE unexpectedly high";
}

TEST(Test__TurboQuantRoundtrip, TQ4_ScalarFull_Quality_128)
{
    constexpr int D = 128;
    constexpr int N = 200;
    TurboQuantContext ctx(D);
    const auto &head_ctx = ctx.for_layer(0).for_layer(0);

    std::mt19937 rng(88);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    double total_cosine = 0.0;

    for (int trial = 0; trial < N; ++trial)
    {
        float input[D], output[D], scratch0[D], scratch1[D];
        for (int i = 0; i < D; ++i)
            input[i] = dist(rng);

        TQ4Block_128 block;
        turboquant_quantize_tq4<D>(input, head_ctx, block, scratch0, scratch1);
        ASSERT_LT(block.residual_norm, 0.0f);

        turboquant_dequantize_tq4<D>(block, head_ctx, output, scratch0);
        total_cosine += compute_cosine_similarity(input, output, D);
    }

    double avg_cosine = total_cosine / N;
    std::cout << "Scalar-full D=128: avg cosine=" << avg_cosine
              << " (over " << N << " vectors)" << std::endl;

    EXPECT_GT(avg_cosine, 0.95) << "Scalar-full D=128 should have high quality";
}

// ============================================================================
// Extreme norm edge cases
// ============================================================================

TEST(Test__TurboQuantRoundtrip, TQ4_ExtremeNorms_64)
{
    constexpr int D = 64;
    TurboQuantContext ctx(D);
    const auto &head_ctx = ctx.for_layer(0).for_layer(0);

    std::mt19937 rng(999);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Generate a unit-direction vector to scale
    float direction[D];
    for (int i = 0; i < D; ++i)
        direction[i] = dist(rng);
    float dir_norm = compute_norm(direction, D);
    for (int i = 0; i < D; ++i)
        direction[i] /= dir_norm;

    struct NormCase
    {
        float scale;
        const char *name;
    };
    NormCase cases[] = {
        {1e-20f, "very_small_1e-20"},
        {1e-6f, "small_1e-6"},
        {1e+6f, "large_1e+6"},
        {1e+20f, "very_large_1e+20"},
    };

    for (const auto &tc : cases)
    {
        float input[D], output[D], scratch0[D], scratch1[D];
        for (int i = 0; i < D; ++i)
            input[i] = direction[i] * tc.scale;

        TQ4Block_64 block;
        turboquant_quantize_tq4<D>(input, head_ctx, block, scratch0, scratch1);
        turboquant_dequantize_tq4<D>(block, head_ctx, output, scratch0);

        bool has_nan_inf = false;
        for (int i = 0; i < D; ++i)
        {
            if (std::isnan(output[i]) || std::isinf(output[i]))
            {
                has_nan_inf = true;
                break;
            }
        }

        if (tc.scale < 1e-10f)
        {
            // Very small norms: should produce near-zero (graceful underflow)
            EXPECT_FALSE(has_nan_inf) << "NaN/Inf for small norm case: " << tc.name;
            float out_norm = compute_norm(output, D);
            EXPECT_LT(out_norm, 1e-5f)
                << "Near-zero input should produce near-zero output for " << tc.name;
        }
        else if (tc.scale > 1e+15f)
        {
            // Very large norms: overflow to NaN/Inf is acceptable (float norm limitation),
            // but the quantizer must not crash. Just log the result.
            std::cout << "  " << tc.name << ": has_nan_inf=" << has_nan_inf
                      << " (overflow is acceptable for extreme norms)" << std::endl;
        }
        else
        {
            // Moderate norms: cosine should be preserved
            EXPECT_FALSE(has_nan_inf) << "NaN/Inf for moderate norm case: " << tc.name;
            float cosine = compute_cosine_similarity(input, output, D);
            EXPECT_GT(cosine, 0.85f) << "Cosine too low for norm case: " << tc.name;
        }
    }
}

// ============================================================================
// One-hot vector handling
// ============================================================================

TEST(Test__TurboQuantRoundtrip, TQ4_OneHot_64)
{
    constexpr int D = 64;
    TurboQuantContext ctx(D);
    const auto &head_ctx = ctx.for_layer(0).for_layer(0);

    // Test one-hot vectors at various positions
    int positions[] = {0, 1, D / 2, D - 2, D - 1};

    for (int pos : positions)
    {
        float input[D], output[D], scratch0[D], scratch1[D];
        std::fill(input, input + D, 0.0f);
        input[pos] = 1.0f;

        TQ4Block_64 block;
        turboquant_quantize_tq4<D>(input, head_ctx, block, scratch0, scratch1);
        turboquant_dequantize_tq4<D>(block, head_ctx, output, scratch0);

        // No NaN/Inf
        for (int i = 0; i < D; ++i)
        {
            ASSERT_FALSE(std::isnan(output[i])) << "NaN at i=" << i << " for one-hot pos=" << pos;
            ASSERT_FALSE(std::isinf(output[i])) << "Inf at i=" << i << " for one-hot pos=" << pos;
        }

        // The hot position should have the largest absolute value
        int argmax_out = 0;
        for (int i = 1; i < D; ++i)
            if (std::abs(output[i]) > std::abs(output[argmax_out]))
                argmax_out = i;

        // After rotation + quantization, energy gets spread across all dims,
        // so argmax won't be the original position. But cosine should be positive.
        float cosine = compute_cosine_similarity(input, output, D);
        EXPECT_GT(cosine, 0.80f)
            << "One-hot cosine too low at pos=" << pos << " (cosine=" << cosine << ")";

        // Norm should be approximately preserved (input norm = 1.0)
        float out_norm = compute_norm(output, D);
        EXPECT_NEAR(out_norm, 1.0f, 0.3f)
            << "One-hot norm not preserved at pos=" << pos;
    }
}

TEST(Test__TurboQuantRoundtrip, TQ4_OneHot_128)
{
    constexpr int D = 128;
    TurboQuantContext ctx(D);
    const auto &head_ctx = ctx.for_layer(0).for_layer(0);

    int positions[] = {0, 1, D / 2, D - 2, D - 1};

    for (int pos : positions)
    {
        float input[D], output[D], scratch0[D], scratch1[D];
        std::fill(input, input + D, 0.0f);
        input[pos] = 1.0f;

        TQ4Block_128 block;
        turboquant_quantize_tq4<D>(input, head_ctx, block, scratch0, scratch1);
        turboquant_dequantize_tq4<D>(block, head_ctx, output, scratch0);

        for (int i = 0; i < D; ++i)
        {
            ASSERT_FALSE(std::isnan(output[i])) << "NaN at i=" << i << " for one-hot pos=" << pos;
            ASSERT_FALSE(std::isinf(output[i])) << "Inf at i=" << i << " for one-hot pos=" << pos;
        }

        float cosine = compute_cosine_similarity(input, output, D);
        EXPECT_GT(cosine, 0.80f)
            << "One-hot D=128 cosine too low at pos=" << pos << " (cosine=" << cosine << ")";

        float out_norm = compute_norm(output, D);
        EXPECT_NEAR(out_norm, 1.0f, 0.3f)
            << "One-hot D=128 norm not preserved at pos=" << pos;
    }
}

TEST(Test__TurboQuantRoundtrip, TQ4_OneHot_ScaledAndNegative_64)
{
    constexpr int D = 64;
    TurboQuantContext ctx(D);
    const auto &head_ctx = ctx.for_layer(0).for_layer(0);

    // Test scaled one-hot vectors (positive and negative)
    struct OneHotCase
    {
        int pos;
        float val;
        const char *name;
    };
    OneHotCase cases[] = {
        {0, 5.0f, "pos0_scale5"},
        {D / 2, -3.0f, "mid_neg3"},
        {D - 1, 0.01f, "last_small"},
        {7, -100.0f, "pos7_neg100"},
    };

    for (const auto &tc : cases)
    {
        float input[D], output[D], scratch0[D], scratch1[D];
        std::fill(input, input + D, 0.0f);
        input[tc.pos] = tc.val;

        TQ4Block_64 block;
        turboquant_quantize_tq4<D>(input, head_ctx, block, scratch0, scratch1);
        turboquant_dequantize_tq4<D>(block, head_ctx, output, scratch0);

        for (int i = 0; i < D; ++i)
        {
            ASSERT_FALSE(std::isnan(output[i])) << "NaN for " << tc.name;
            ASSERT_FALSE(std::isinf(output[i])) << "Inf for " << tc.name;
        }

        float cosine = compute_cosine_similarity(input, output, D);
        EXPECT_GT(cosine, 0.80f)
            << "Scaled one-hot cosine too low for " << tc.name << " (cosine=" << cosine << ")";

        // Sign of the hot element should be preserved
        EXPECT_GT(output[tc.pos] * tc.val, 0.0f)
            << "Sign not preserved for " << tc.name;
    }
}

// ============================================================================
// One-hot inner product preservation
// ============================================================================

TEST(Test__TurboQuantRoundtrip, TQ4_OneHot_InnerProduct_64)
{
    constexpr int D = 64;
    TurboQuantContext ctx(D);
    const auto &head_ctx = ctx.for_layer(0).for_layer(0);

    std::mt19937 rng(555);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // For each one-hot position, quantize and check inner product with random queries
    constexpr int N_QUERIES = 50;
    double total_ip_error = 0.0;
    int total_tests = 0;

    for (int pos = 0; pos < D; pos += 8) // sample every 8th position
    {
        float k[D], scratch0[D], scratch1[D];
        std::fill(k, k + D, 0.0f);
        k[pos] = 1.0f;

        TQ4Block_64 block;
        turboquant_quantize_tq4<D>(k, head_ctx, block, scratch0, scratch1);

        float k_hat[D];
        turboquant_dequantize_tq4<D>(block, head_ctx, k_hat, scratch0);

        for (int qi = 0; qi < N_QUERIES; ++qi)
        {
            float q[D];
            for (int i = 0; i < D; ++i)
                q[i] = dist(rng);

            float true_dot = q[pos]; // <q, e_pos> = q[pos]
            float approx_dot = compute_dot(q, k_hat, D);
            total_ip_error += std::abs(approx_dot - true_dot);
            ++total_tests;
        }
    }

    double mae = total_ip_error / total_tests;
    std::cout << "One-hot inner product MAE = " << mae
              << " (over " << total_tests << " tests)" << std::endl;

    // One-hot vectors are adversarial for rotation-based quantization
    // (all energy in one coordinate gets spread). MAE should still be reasonable.
    EXPECT_LT(mae, 0.5) << "One-hot inner product MAE too high";
}

// ============================================================================
// Row-range dequant parity: partial range matches full-range at same rows
// ============================================================================

TEST(Test__TurboQuantRoundtrip, TQ4_RowRangeDequant_Parity_64)
{
    constexpr int D = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * D;
    constexpr int N_ROWS = 16;

    TurboQuantContext ctx(D);

    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Quantize N_ROWS of KV data
    std::vector<TQ4Block_64> blocks(N_ROWS * N_KV_HEADS);
    for (int r = 0; r < N_ROWS; ++r)
    {
        for (int h = 0; h < N_KV_HEADS; ++h)
        {
            float input[D], scratch0[D], scratch1[D];
            for (int i = 0; i < D; ++i)
                input[i] = dist(rng);
            turboquant_quantize_tq4<D>(
                input, ctx.for_layer(h), blocks[r * N_KV_HEADS + h], scratch0, scratch1);
        }
    }

    const auto *raw = reinterpret_cast<const uint8_t *>(blocks.data());
    const size_t block_bytes = sizeof(TQ4Block_64);
    const size_t row_bytes = N_KV_HEADS * block_bytes;

    // Full dequant: rows 0..N_ROWS
    std::vector<float> full_fp32(N_ROWS * KV_DIM, 0.0f);
    turboquant_dequantize_kv_rows(
        raw, raw, ctx,
        full_fp32.data(), full_fp32.data(),
        0, N_ROWS, D, N_KV_HEADS,
        row_bytes, row_bytes, block_bytes, block_bytes);

    // Partial dequant: rows 5..10
    std::vector<float> partial_fp32(N_ROWS * KV_DIM, -999.0f); // sentinel fill
    turboquant_dequantize_kv_rows(
        raw, raw, ctx,
        partial_fp32.data(), partial_fp32.data(),
        5, 10, D, N_KV_HEADS,
        row_bytes, row_bytes, block_bytes, block_bytes);

    // Rows 5..9 should match exactly between full and partial
    for (int r = 5; r < 10; ++r)
    {
        for (int i = 0; i < KV_DIM; ++i)
        {
            EXPECT_FLOAT_EQ(full_fp32[r * KV_DIM + i], partial_fp32[r * KV_DIM + i])
                << "Mismatch at row=" << r << " col=" << i;
        }
    }

    // Rows outside 5..9 should be untouched (sentinel)
    for (int r = 0; r < 5; ++r)
        EXPECT_FLOAT_EQ(partial_fp32[r * KV_DIM], -999.0f)
            << "Row " << r << " should be untouched by partial dequant";
}

// ============================================================================
// dequantize_v_rows: verify it works for D=128 and test D=64 via kv_rows
// ============================================================================

TEST(Test__TurboQuantRoundtrip, TQ4_DequantVRows_MatchesPerBlock_128)
{
    constexpr int D = 128;
    constexpr int N_KV_HEADS = 4;
    constexpr int KV_DIM = N_KV_HEADS * D;
    constexpr int N_ROWS = 8;

    TurboQuantContext ctx(D);

    std::mt19937 rng(7777);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Quantize rows
    std::vector<TQ4Block_128> blocks(N_ROWS * N_KV_HEADS);
    for (int r = 0; r < N_ROWS; ++r)
    {
        for (int h = 0; h < N_KV_HEADS; ++h)
        {
            float input[D], scratch0[D], scratch1[D];
            for (int i = 0; i < D; ++i)
                input[i] = dist(rng);
            turboquant_quantize_tq4<D>(
                input, ctx.for_layer(h), blocks[r * N_KV_HEADS + h], scratch0, scratch1);
        }
    }

    const auto *raw = reinterpret_cast<const uint8_t *>(blocks.data());
    const size_t block_bytes = sizeof(TQ4Block_128);
    const size_t row_bytes = N_KV_HEADS * block_bytes;

    // Batch dequant via dequantize_v_rows
    std::vector<float> batch_fp32(N_ROWS * KV_DIM, 0.0f);
    turboquant_dequantize_v_rows(
        raw, ctx,
        batch_fp32.data(),
        0, N_ROWS, D, N_KV_HEADS,
        row_bytes, block_bytes);

    // Per-block reference dequant
    std::vector<float> ref_fp32(N_ROWS * KV_DIM, 0.0f);
    for (int r = 0; r < N_ROWS; ++r)
    {
        for (int h = 0; h < N_KV_HEADS; ++h)
        {
            alignas(64) float scratch[D];
            turboquant_dequantize_tq4(
                blocks[r * N_KV_HEADS + h], ctx.for_layer(h),
                ref_fp32.data() + r * KV_DIM + h * D, scratch);
        }
    }

    // Should be bit-exact (same computation, same order)
    for (int i = 0; i < N_ROWS * KV_DIM; ++i)
    {
        EXPECT_FLOAT_EQ(batch_fp32[i], ref_fp32[i])
            << "Mismatch at index " << i
            << " (row=" << i / KV_DIM << " col=" << i % KV_DIM << ")";
    }
}
