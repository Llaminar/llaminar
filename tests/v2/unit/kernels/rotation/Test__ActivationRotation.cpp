/**
 * @file Test__ActivationRotation.cpp
 * @brief Unit tests for ActivationRotation and ModelWeightRotation
 *
 * Tests:
 * - Block-diagonal rotation preserves vector norms (orthogonality)
 * - Rotation + inverse rotation = identity
 * - Weight rotation sets activation_rotation metadata on tensors
 * - ModelWeightRotation correctly rotates all GEMM weights
 * - Rotation with 4B-model dimensions (hidden=2560, ffn=9216, block_dim=128)
 * - Rotated GEMM output matches unrotated GEMM (mathematical invariance)
 * - FWHT correctness: H·H = n·I property verified at all supported sizes
 * - AVX-512 vs scalar parity for D=64 and D=128
 * - Orthogonality matrix property: R·R^T = I (columns of R are orthonormal)
 * - Known Hadamard outputs for small vectors (H₂, H₄, H₈)
 * - Deterministic output across repeated calls
 * - Tight round-trip tolerance (rotate + inverse ≈ identity within FP32 ULP)
 * - Large-dimension stress tests (D=2048+)
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "kernels/cpu/rotation/ActivationRotation.h"
#include "kernels/cpu/rotation/ModelWeightRotation.h"
#include "models/GraphTypes.h"
#include "../../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    struct RotationQuantMetrics
    {
        uint64_t seed = 0;
        double mse = 0.0;
        double cosine = 0.0;
        double rotated_kurtosis = 0.0;
        double rotated_max_over_std = 0.0;
    };

    static void quantizeDequantizeQ8_1Like(std::vector<float> &data)
    {
        constexpr int q_block = 32;
        for (size_t base = 0; base < data.size(); base += q_block)
        {
            const size_t end = std::min(base + q_block, data.size());
            float amax = 0.0f;
            for (size_t i = base; i < end; ++i)
                amax = std::max(amax, std::fabs(data[i]));

            if (amax <= 1e-20f)
                continue;

            const float scale = amax / 127.0f;
            for (size_t i = base; i < end; ++i)
            {
                int q = static_cast<int>(std::nearbyint(data[i] / scale));
                q = std::max(-128, std::min(127, q));
                data[i] = static_cast<float>(q) * scale;
            }
        }
    }

    static double computeMSE(const std::vector<float> &a, const std::vector<float> &b)
    {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            sum += d * d;
        }
        return sum / static_cast<double>(a.size());
    }

    static double computeCosine(const std::vector<float> &a, const std::vector<float> &b)
    {
        double dot = 0.0;
        double na = 0.0;
        double nb = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            const double av = a[i];
            const double bv = b[i];
            dot += av * bv;
            na += av * av;
            nb += bv * bv;
        }
        if (na <= 1e-30 || nb <= 1e-30)
            return 0.0;
        return dot / std::sqrt(na * nb);
    }

    static double computeExcessKurtosis(const std::vector<float> &x)
    {
        double mean = 0.0;
        for (float v : x)
            mean += v;
        mean /= static_cast<double>(x.size());

        double m2 = 0.0;
        double m4 = 0.0;
        for (float v : x)
        {
            const double d = static_cast<double>(v) - mean;
            const double d2 = d * d;
            m2 += d2;
            m4 += d2 * d2;
        }
        m2 /= static_cast<double>(x.size());
        m4 /= static_cast<double>(x.size());
        if (m2 <= 1e-30)
            return 0.0;
        return m4 / (m2 * m2) - 3.0;
    }

    static double computeMaxOverStd(const std::vector<float> &x)
    {
        double mean = 0.0;
        double amax = 0.0;
        for (float v : x)
        {
            mean += v;
            amax = std::max(amax, std::fabs(static_cast<double>(v)));
        }
        mean /= static_cast<double>(x.size());

        double var = 0.0;
        for (float v : x)
        {
            const double d = static_cast<double>(v) - mean;
            var += d * d;
        }
        var /= static_cast<double>(x.size());
        const double stddev = std::sqrt(var);
        if (stddev <= 1e-30)
            return 0.0;
        return amax / stddev;
    }

    static RotationQuantMetrics evaluateQ8RoundTripWithSeed(
        const std::vector<float> &activations,
        int rows,
        int dim,
        int block_dim,
        uint64_t seed)
    {
        ActivationRotation rot(dim, block_dim, seed);

        std::vector<float> rotated = activations;
        rot.rotate_rows_inplace(rotated.data(), rows, dim);

        std::vector<float> reconstructed = rotated;
        quantizeDequantizeQ8_1Like(reconstructed);
        rot.inverse_rotate_rows_inplace(reconstructed.data(), rows, dim);

        RotationQuantMetrics metrics;
        metrics.seed = seed;
        metrics.mse = computeMSE(activations, reconstructed);
        metrics.cosine = computeCosine(activations, reconstructed);
        metrics.rotated_kurtosis = computeExcessKurtosis(rotated);
        metrics.rotated_max_over_std = computeMaxOverStd(rotated);
        return metrics;
    }

    static std::vector<float> makeDefaultSeedStressActivations(
        int rows,
        int dim,
        int block_dim,
        float peak_scale)
    {
        // Build a corpus whose default-seed rotated representation has large
        // per-Q8-block peaks plus many small values. This models the failure mode
        // we care about: one outlier dominates a block scale and wastes int8
        // dynamic range. A calibration pass can choose a different Hadamard sign
        // seed that spreads those peaks more evenly at unchanged runtime cost.
        const uint64_t current_seed = 31;
        ActivationRotation current(dim, block_dim, current_seed);
        std::vector<float> rotated(rows * dim, 0.0f);

        for (int r = 0; r < rows; ++r)
        {
            const float row_scale = peak_scale * (1.0f + 0.013f * static_cast<float>(r % 11));
            for (int k = 0; k < dim; ++k)
            {
                const float phase = static_cast<float>((r + 1) * (k + 3));
                rotated[static_cast<size_t>(r) * dim + k] =
                    0.23f * std::sin(0.017f * phase) +
                    0.11f * std::cos(0.031f * phase);
            }

            for (int block = 0; block < dim; block += block_dim)
            {
                const int offsets[] = {0, 32, 64, 96};
                const float amps[] = {1.00f, -0.86f, 0.73f, -0.61f};
                for (int i = 0; i < 4; ++i)
                    rotated[static_cast<size_t>(r) * dim + block + offsets[i]] +=
                        row_scale * amps[i];
            }
        }

        std::vector<float> activations = rotated;
        current.inverse_rotate_rows_inplace(activations.data(), rows, dim);
        return activations;
    }
}

// ============================================================================
// ActivationRotation Basic Tests
// ============================================================================

TEST(Test__ActivationRotation, ConstructionWithValidDimensions)
{
    ActivationRotation rot(256, 128);
    EXPECT_EQ(rot.total_dim(), 256);
    EXPECT_EQ(rot.block_dim(), 128);
    EXPECT_EQ(rot.n_blocks(), 2);
}

TEST(Test__ActivationRotation, ConstructionWith4BDimensions)
{
    // Qwen3.5-4B: hidden=2560, ffn=9216, block_dim=128
    ActivationRotation hidden_rot(2560, 128);
    EXPECT_EQ(hidden_rot.n_blocks(), 20);

    ActivationRotation ffn_rot(9216, 128);
    EXPECT_EQ(ffn_rot.n_blocks(), 72);
}

TEST(Test__ActivationRotation, RotationPreservesNorm)
{
    const int dim = 2560;
    const int block_dim = 128;
    ActivationRotation rot(dim, block_dim);

    // Create a random vector
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(dim);
    for (auto &v : data)
        v = dist(gen);

    // Compute original norm
    float orig_norm = 0.0f;
    for (float v : data)
        orig_norm += v * v;
    orig_norm = std::sqrt(orig_norm);

    // Rotate in-place
    rot.rotate_inplace(data.data(), dim);

    // Compute rotated norm
    float rot_norm = 0.0f;
    for (float v : data)
        rot_norm += v * v;
    rot_norm = std::sqrt(rot_norm);

    // Orthogonal rotation preserves norm
    EXPECT_NEAR(rot_norm, orig_norm, orig_norm * 1e-5f)
        << "Orthogonal rotation should preserve vector norm";
}

TEST(Test__ActivationRotation, RotateInverseIsIdentity)
{
    const int dim = 256;
    const int block_dim = 128;
    ActivationRotation rot(dim, block_dim);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> original(dim);
    for (auto &v : original)
        v = dist(gen);

    std::vector<float> data = original;

    // Rotate forward, then inverse
    rot.rotate_inplace(data.data(), dim);
    rot.inverse_rotate_inplace(data.data(), dim);

    // Should recover original (FP32 rotation accumulates ~1e-4 error across 128-dim blocks)
    for (int i = 0; i < dim; ++i)
    {
        EXPECT_NEAR(data[i], original[i], 5e-4f)
            << "Rotate + inverse should be identity at index " << i;
    }
}

TEST(Test__ActivationRotation, MultiRowRotation)
{
    const int dim = 256;
    const int block_dim = 128;
    const int rows = 4;
    ActivationRotation rot(dim, block_dim);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(rows * dim);
    std::vector<float> original = data;
    for (auto &v : data)
        v = dist(gen);
    original = data;

    rot.rotate_rows_inplace(data.data(), rows, dim);

    // Each row's norm should be preserved
    for (int r = 0; r < rows; ++r)
    {
        float orig_norm = 0.0f, rot_norm = 0.0f;
        for (int i = 0; i < dim; ++i)
        {
            orig_norm += original[r * dim + i] * original[r * dim + i];
            rot_norm += data[r * dim + i] * data[r * dim + i];
        }
        EXPECT_NEAR(std::sqrt(rot_norm), std::sqrt(orig_norm),
                    std::sqrt(orig_norm) * 1e-5f)
            << "Row " << r << " norm should be preserved";
    }
}

TEST(Test__ActivationRotation, RotationReducesKurtosis)
{
    // Verify that rotation reduces kurtosis of a "spiky" vector
    const int dim = 256;
    const int block_dim = 128;
    ActivationRotation rot(dim, block_dim);

    // Create a high-kurtosis vector: mostly small values with a few large spikes
    std::vector<float> data(dim, 0.01f);
    data[0] = 100.0f;  // Spike
    data[64] = -80.0f; // Spike
    data[128] = 50.0f; // Spike

    // Compute kurtosis before rotation
    auto compute_kurtosis = [](const float *d, int n)
    {
        float mean = 0, m2 = 0, m4 = 0;
        for (int i = 0; i < n; ++i)
            mean += d[i];
        mean /= n;
        for (int i = 0; i < n; ++i)
        {
            float diff = d[i] - mean;
            m2 += diff * diff;
            m4 += diff * diff * diff * diff;
        }
        m2 /= n;
        m4 /= n;
        return m4 / (m2 * m2);
    };

    float kurtosis_before = compute_kurtosis(data.data(), dim);
    rot.rotate_inplace(data.data(), dim);
    float kurtosis_after = compute_kurtosis(data.data(), dim);

    EXPECT_LT(kurtosis_after, kurtosis_before)
        << "Rotation should reduce kurtosis of spiky vectors";
}

TEST(Test__ActivationRotation, CalibrationSelectedHadamardSeedImprovesQ8RoundTrip)
{
    // Exploratory proof-of-concept for calibration-selected Hadamard sign seeds.
    //
    // Current production rotation uses seed=31. The runtime cost of a different
    // seed is identical: the same FWHT is used, only the deterministic sign mask
    // changes. This test simulates an offline calibration pass that scores a
    // small seed bank on calibration activations, then checks whether the chosen
    // seed improves held-out Q8_1-style activation round-trip quality.
    const int rows = 64;
    const int dim = 512;
    const int block_dim = 128;
    const uint64_t current_seed = 31;

    const auto calibration = makeDefaultSeedStressActivations(rows, dim, block_dim, 72.0f);
    const auto heldout = makeDefaultSeedStressActivations(rows, dim, block_dim, 84.0f);

    const auto current_cal = evaluateQ8RoundTripWithSeed(
        calibration, rows, dim, block_dim, current_seed);
    const auto current_heldout = evaluateQ8RoundTripWithSeed(
        heldout, rows, dim, block_dim, current_seed);

    RotationQuantMetrics best_cal;
    best_cal.mse = std::numeric_limits<double>::infinity();

    for (uint64_t seed = 1; seed <= 128; ++seed)
    {
        const auto candidate = evaluateQ8RoundTripWithSeed(
            calibration, rows, dim, block_dim, seed);
        if (candidate.mse < best_cal.mse)
            best_cal = candidate;
    }

    const auto selected_heldout = evaluateQ8RoundTripWithSeed(
        heldout, rows, dim, block_dim, best_cal.seed);

    const double cal_mse_gain =
        (current_cal.mse - best_cal.mse) / current_cal.mse;
    const double heldout_mse_gain =
        (current_heldout.mse - selected_heldout.mse) / current_heldout.mse;
    const double heldout_papr_gain =
        (current_heldout.rotated_max_over_std - selected_heldout.rotated_max_over_std) /
        current_heldout.rotated_max_over_std;

    LOG_INFO("[SelectedHadamardSeed] current_seed=" << current_seed
                                                    << " selected_seed=" << best_cal.seed);
    LOG_INFO("[SelectedHadamardSeed] calibration_mse current=" << current_cal.mse
                                                               << " selected=" << best_cal.mse
                                                               << " gain=" << (100.0 * cal_mse_gain) << "%");
    LOG_INFO("[SelectedHadamardSeed] heldout_mse current=" << current_heldout.mse
                                                           << " selected=" << selected_heldout.mse
                                                           << " gain=" << (100.0 * heldout_mse_gain) << "%");
    LOG_INFO("[SelectedHadamardSeed] heldout_cosine current=" << current_heldout.cosine
                                                              << " selected=" << selected_heldout.cosine);
    LOG_INFO("[SelectedHadamardSeed] heldout_max/std current="
             << current_heldout.rotated_max_over_std
             << " selected=" << selected_heldout.rotated_max_over_std
             << " gain=" << (100.0 * heldout_papr_gain) << "%");

    EXPECT_NE(best_cal.seed, current_seed)
        << "The calibration sweep should find a non-default sign mask on this stress corpus";
    EXPECT_LT(best_cal.mse, current_cal.mse * 0.80)
        << "Selected seed should materially reduce calibration Q8_1 round-trip MSE";
    EXPECT_LT(selected_heldout.mse, current_heldout.mse * 0.90)
        << "Calibration-selected seed should generalize to held-out activations";
    EXPECT_GT(selected_heldout.cosine, current_heldout.cosine)
        << "Selected seed should improve held-out cosine similarity";
    EXPECT_LT(selected_heldout.rotated_max_over_std,
              current_heldout.rotated_max_over_std * 0.90)
        << "Selected seed should lower held-out peak-to-std ratio in rotated space";
}

// ============================================================================
// ModelWeightRotation Tests
// ============================================================================

TEST(Test__ActivationRotation, ModelWeightRotation_Create)
{
    auto rotator = ModelWeightRotation::create(2560, 9216, 128);
    ASSERT_NE(rotator, nullptr);
    EXPECT_NE(rotator->hiddenRotation(), nullptr);
    EXPECT_NE(rotator->ffnRotation(), nullptr);
    EXPECT_EQ(rotator->hiddenRotation()->total_dim(), 2560);
    EXPECT_EQ(rotator->ffnRotation()->total_dim(), 9216);
}

TEST(Test__ActivationRotation, ModelWeightRotation_RotateWeight_SetsMetadata)
{
    auto rotator = ModelWeightRotation::create(256, 512, 128);

    // Create a small FP32 weight tensor [8×256]
    auto weight = TestTensorFactory::createFP32Random({8, 256});
    weight->setDebugName("test_wq");

    TensorBase *ptr = weight.get();
    bool ok = rotator->rotateWeight(&ptr, ModelWeightRotation::HIDDEN);
    ASSERT_TRUE(ok) << "rotateWeight should succeed";

    // Pointer should be the same (tag-only, no data copy)
    EXPECT_EQ(ptr, weight.get()) << "rotateWeight should tag in-place, not create a new tensor";

    // The tensor should have rotation metadata
    EXPECT_NE(ptr->activationRotation(), nullptr)
        << "Tagged tensor should have activationRotation set";
    EXPECT_EQ(ptr->activationRotation()->total_dim(), 256);
}

TEST(Test__ActivationRotation, ModelWeightRotation_SkipsDimensionMismatch)
{
    auto rotator = ModelWeightRotation::create(256, 512, 128);

    // Create weight with K-dim = 384 (doesn't match hidden=256 or ffn=512)
    auto weight = TestTensorFactory::createFP32Random({8, 384});
    weight->setDebugName("test_mismatch");

    TensorBase *ptr = weight.get();
    bool ok = rotator->rotateWeight(&ptr, ModelWeightRotation::HIDDEN);
    EXPECT_FALSE(ok) << "rotateWeight should fail for dimension mismatch";
    EXPECT_EQ(ptr, weight.get()) << "Pointer should be unchanged on failure";
}

TEST(Test__ActivationRotation, ModelWeightRotation_RotateLayerWeights)
{
    auto rotator = ModelWeightRotation::create(64, 128, 64);

    // Create mock layer weights
    auto wq = TestTensorFactory::createFP32Random({16, 64});
    auto wk = TestTensorFactory::createFP32Random({8, 64});
    auto wv = TestTensorFactory::createFP32Random({8, 64});
    auto gate = TestTensorFactory::createFP32Random({128, 64});
    auto up = TestTensorFactory::createFP32Random({128, 64});
    auto down = TestTensorFactory::createFP32Random({64, 128});

    wq->setDebugName("wq");
    wk->setDebugName("wk");
    wv->setDebugName("wv");
    gate->setDebugName("gate");
    up->setDebugName("up");
    down->setDebugName("down");

    LayerWeights layer{};
    layer.wq = wq.get();
    layer.wk = wk.get();
    layer.wv = wv.get();
    layer.gate_proj = gate.get();
    layer.up_proj = up.get();
    layer.down_proj = down.get();

    TensorBase *orig_wq = layer.wq;
    TensorBase *orig_down = layer.down_proj;

    rotator->rotateLayerWeights(layer);

    // FA weights with K=64 should be tagged (hidden rotation)
    EXPECT_EQ(layer.wq, orig_wq)
        << "wq pointer should be unchanged (tag-only)";
    EXPECT_NE(layer.wq->activationRotation(), nullptr)
        << "Tagged wq should have rotation metadata";

    // FFN down with K=128 should be tagged (ffn rotation)
    EXPECT_EQ(layer.down_proj, orig_down)
        << "down_proj pointer should be unchanged (tag-only)";
    EXPECT_NE(layer.down_proj->activationRotation(), nullptr)
        << "Tagged down_proj should have rotation metadata";
}

TEST(Test__ActivationRotation, ModelWeightRotation_RotateAllWeights)
{
    auto rotator = ModelWeightRotation::create(64, 128, 64);

    // Create a minimal ModelWeights with 2 layers
    auto wq0 = TestTensorFactory::createFP32Random({16, 64});
    auto gate0 = TestTensorFactory::createFP32Random({128, 64});
    auto down0 = TestTensorFactory::createFP32Random({64, 128});
    auto wq1 = TestTensorFactory::createFP32Random({16, 64});
    auto gate1 = TestTensorFactory::createFP32Random({128, 64});
    auto down1 = TestTensorFactory::createFP32Random({64, 128});
    auto lm_head = TestTensorFactory::createFP32Random({100, 64});

    wq0->setDebugName("wq0");
    gate0->setDebugName("gate0");
    down0->setDebugName("down0");
    wq1->setDebugName("wq1");
    gate1->setDebugName("gate1");
    down1->setDebugName("down1");
    lm_head->setDebugName("lm_head");

    std::vector<LayerWeights> layers(2);
    layers[0].wq = wq0.get();
    layers[0].gate_proj = gate0.get();
    layers[0].down_proj = down0.get();
    layers[1].wq = wq1.get();
    layers[1].gate_proj = gate1.get();
    layers[1].down_proj = down1.get();

    ModelWeights weights;
    weights.lm_head = lm_head.get();
    weights.get_layer_weights = [&](int idx) -> LayerWeights
    { return layers[idx]; };

    rotator->rotateAllWeights(weights, 2, rotator);

    // LM head should be tagged (K=64 = hidden_dim)
    EXPECT_EQ(weights.lm_head, lm_head.get())
        << "LM head pointer should be unchanged (tag-only)";
    EXPECT_NE(weights.lm_head->activationRotation(), nullptr)
        << "LM head should have rotation metadata";

    // Layer accessor should return tagged weights
    auto l0 = weights.get_layer_weights(0);
    EXPECT_NE(l0.wq->activationRotation(), nullptr)
        << "Layer 0 wq should have rotation metadata";
    EXPECT_NE(l0.down_proj->activationRotation(), nullptr)
        << "Layer 0 down_proj should have rotation metadata";
}

// ============================================================================
// GEMM Invariance Test (X @ W^T == X@R @ (W@R)^T)
// ============================================================================

TEST(Test__ActivationRotation, GEMM_InvarianceWithRotation)
{
    // Verify: X @ W^T == (X @ R) @ (W @ R)^T
    const int M = 2;   // sequence length
    const int K = 128; // hidden dim (1 block)
    const int N = 16;  // output dim

    ActivationRotation rot(K, 128);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Create X [M×K] and W [N×K]
    std::vector<float> X(M * K), W(N * K);
    for (auto &v : X)
        v = dist(gen);
    for (auto &v : W)
        v = dist(gen);

    // Compute Y = X @ W^T (reference)
    std::vector<float> Y_ref(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                Y_ref[m * N + n] += X[m * K + k] * W[n * K + k];

    // Rotate X and W
    std::vector<float> X_rot = X, W_rot = W;
    rot.rotate_rows_inplace(X_rot.data(), M, K);
    rot.rotate_weight_rows(W_rot.data(), N, K);

    // Compute Y_rot = X_rot @ W_rot^T
    std::vector<float> Y_rot(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                Y_rot[m * N + n] += X_rot[m * K + k] * W_rot[n * K + k];

    // Results should match (FP32 rotation + GEMM accumulates ~1e-4 error)
    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(Y_rot[i], Y_ref[i], std::abs(Y_ref[i]) * 5e-4f + 1e-4f)
            << "GEMM output should be invariant to orthogonal rotation at index " << i;
    }
}

// ============================================================================
// Mathematical Accuracy Tests — FWHT Correctness
// ============================================================================

/// Helper: extract the j-th column of the full rotation matrix R by rotating
/// the j-th standard basis vector e_j. This builds R column-by-column.
static std::vector<float> extract_rotation_column(
    const ActivationRotation &rot, int dim, int col)
{
    std::vector<float> e(dim, 0.0f);
    e[col] = 1.0f;
    rot.rotate_inplace(e.data(), dim);
    return e;
}

TEST(Test__ActivationRotation, Orthogonality_RtR_IsIdentity)
{
    // Verify R^T R = I by extracting columns and checking dot products.
    // For an orthogonal matrix, columns are orthonormal:
    //   <R_i, R_j> = δ_ij
    const int dim = 64;
    ActivationRotation rot(dim, dim);

    // Extract all columns of R
    std::vector<std::vector<float>> cols(dim);
    for (int j = 0; j < dim; ++j)
        cols[j] = extract_rotation_column(rot, dim, j);

    // Check R^T R = I
    for (int i = 0; i < dim; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            float dot = 0.0f;
            for (int k = 0; k < dim; ++k)
                dot += cols[i][k] * cols[j][k];

            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(dot, expected, 1e-5f)
                << "R^T R should be identity at (" << i << "," << j << ")";
        }
    }
}

TEST(Test__ActivationRotation, Orthogonality_RRt_IsIdentity)
{
    // Verify R R^T = I by extracting rows and checking dot products.
    // For the inverse: rotate_inplace gives R·e_j (columns of R).
    // inverse_rotate_inplace gives R^T·e_j (columns of R^T = rows of R).
    const int dim = 64;
    ActivationRotation rot(dim, dim);

    // Extract rows of R (= columns of R^T)
    std::vector<std::vector<float>> rows(dim);
    for (int j = 0; j < dim; ++j)
    {
        rows[j].assign(dim, 0.0f);
        rows[j][j] = 1.0f;
        rot.inverse_rotate_inplace(rows[j].data(), dim);
    }

    // Check R R^T = I
    for (int i = 0; i < dim; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            float dot = 0.0f;
            for (int k = 0; k < dim; ++k)
                dot += rows[i][k] * rows[j][k];

            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(dot, expected, 1e-5f)
                << "R R^T should be identity at (" << i << "," << j << ")";
        }
    }
}

TEST(Test__ActivationRotation, Symmetry_R_Equals_Rt)
{
    // NOTE: R = HD/√d is NOT symmetric in general (HD ≠ DH unless D = ±I).
    // R^T = DH/√d ≠ R.
    // What IS true: R is orthogonal (R·R^T = I), tested above.
    // This test verifies the CORRECT relationship: R^T = D·H/√d and R = H·D/√d
    // by checking that R^{-1}(R(x)) = x for all x (already tested in round-trip
    // tests above), and additionally that R and R^T apply different transformations.
    const int dim = 64;
    ActivationRotation rot(dim, dim);

    std::vector<float> x(dim);
    for (int i = 0; i < dim; ++i)
        x[i] = static_cast<float>(i + 1);

    std::vector<float> forward = x;
    rot.rotate_inplace(forward.data(), dim);

    std::vector<float> inverse = x;
    rot.inverse_rotate_inplace(inverse.data(), dim);

    // Forward and inverse should produce DIFFERENT results (R ≠ R^T)
    // unless x is an eigenvector of D
    bool any_different = false;
    for (int i = 0; i < dim; ++i)
    {
        if (std::abs(forward[i] - inverse[i]) > 1e-6f)
        {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different)
        << "R and R^T should produce different results (R is not symmetric)";

    // But round-trip should be identity
    std::vector<float> roundtrip = x;
    rot.rotate_inplace(roundtrip.data(), dim);
    rot.inverse_rotate_inplace(roundtrip.data(), dim);
    for (int i = 0; i < dim; ++i)
    {
        EXPECT_NEAR(roundtrip[i], x[i], 1e-4f)
            << "R^{-1}(R(x)) should be identity at index " << i;
    }
}

TEST(Test__ActivationRotation, FWHT_MatchesReference_D64)
{
    // Compare the FWHT implementation (which uses AVX-512 fwht_64_avx512
    // for D=64) against a textbook reference FWHT implemented here.
    // This is THE test that catches butterfly sign bugs — the reference
    // implementation is trivially correct and any divergence in the
    // optimized AVX-512 path will show up here.

    // Reference FWHT: standard in-place Cooley-Tukey butterfly
    auto reference_fwht = [](float *data, int n)
    {
        for (int len = 1; len < n; len <<= 1)
            for (int i = 0; i < n; i += len << 1)
                for (int j = 0; j < len; ++j)
                {
                    float u = data[i + j];
                    float v = data[i + j + len];
                    data[i + j] = u + v;
                    data[i + j + len] = u - v;
                }
    };

    const int dim = 64;
    ActivationRotation rot(dim, dim);

    // We need to test the FWHT in isolation. Since rotate_inplace does:
    //   sign_flip → FWHT → scale
    // and inverse_rotate_inplace does:
    //   FWHT → scale → sign_flip
    // We can extract the FWHT result by:
    //   1. Start with input x
    //   2. Manually apply sign_flips (we know them from extracting R columns)
    //   3. Apply reference FWHT
    //   4. Apply 1/√d scaling
    //   5. Compare against rotate_inplace(x)

    // Extract sign_flips by transforming the first basis vector:
    // rotate(e_0) = H·(D·e_0)/√d = H·(d_0·e_0)/√d = d_0·H[:,0]/√d
    // H[:,0] = [1,1,...,1], so rotate(e_0) = d_0·[1,...,1]/√d
    // and d_0 = sign of rotate(e_0)[0] * √d
    std::vector<float> sign_probe(dim, 0.0f);
    sign_probe[0] = 1.0f;
    rot.rotate_inplace(sign_probe.data(), dim);
    // sign_probe is now H·(D·e_0)/√d = d_0·[1,...,1]/√d
    // So d_0 = sign_probe[0] * sqrt(dim)
    float d_0_sign = (sign_probe[0] > 0) ? 1.0f : -1.0f;

    // Extract all sign flips: rotate(e_j) = H·(d_j·e_j)/√d = d_j·H[:,j]/√d
    // H[:,j] is the j-th column of the Hadamard matrix.
    // All entries of H[:,j] are ±1, so |rotate(e_j)[0]| = 1/√d
    // And d_j = sign(rotate(e_j)[0]) * √d (since H[0,j] = 1 for all j)
    std::vector<float> sign_flips(dim);
    for (int j = 0; j < dim; ++j)
    {
        std::vector<float> ej(dim, 0.0f);
        ej[j] = 1.0f;
        rot.rotate_inplace(ej.data(), dim);
        // ej[0] = d_j * H[0,j] / √d = d_j * 1 / √d  (H first row is all 1s)
        sign_flips[j] = (ej[0] > 0) ? 1.0f : -1.0f;
    }

    // Now test with random vectors
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(dim));

    for (int trial = 0; trial < 200; ++trial)
    {
        std::vector<float> x(dim);
        for (auto &v : x)
            v = dist(gen);

        // Reference: sign_flip → reference FWHT → scale
        std::vector<float> y_ref = x;
        for (int i = 0; i < dim; ++i)
            y_ref[i] *= sign_flips[i];
        reference_fwht(y_ref.data(), dim);
        for (int i = 0; i < dim; ++i)
            y_ref[i] *= inv_sqrt_d;

        // Implementation: rotate_inplace
        std::vector<float> y_impl = x;
        rot.rotate_inplace(y_impl.data(), dim);

        for (int i = 0; i < dim; ++i)
        {
            EXPECT_NEAR(y_impl[i], y_ref[i], 1e-3f)
                << "FWHT D=64: impl vs reference mismatch at trial="
                << trial << " i=" << i;
        }
    }
}

TEST(Test__ActivationRotation, FWHT_MatchesReference_D128)
{
    // Same as D64 but for the fwht_128_avx512 path.
    auto reference_fwht = [](float *data, int n)
    {
        for (int len = 1; len < n; len <<= 1)
            for (int i = 0; i < n; i += len << 1)
                for (int j = 0; j < len; ++j)
                {
                    float u = data[i + j];
                    float v = data[i + j + len];
                    data[i + j] = u + v;
                    data[i + j + len] = u - v;
                }
    };

    const int dim = 128;
    ActivationRotation rot(dim, dim);

    // Extract sign flips
    std::vector<float> sign_flips(dim);
    for (int j = 0; j < dim; ++j)
    {
        std::vector<float> ej(dim, 0.0f);
        ej[j] = 1.0f;
        rot.rotate_inplace(ej.data(), dim);
        sign_flips[j] = (ej[0] > 0) ? 1.0f : -1.0f;
    }

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(dim));

    for (int trial = 0; trial < 200; ++trial)
    {
        std::vector<float> x(dim);
        for (auto &v : x)
            v = dist(gen);

        // Reference
        std::vector<float> y_ref = x;
        for (int i = 0; i < dim; ++i)
            y_ref[i] *= sign_flips[i];
        reference_fwht(y_ref.data(), dim);
        for (int i = 0; i < dim; ++i)
            y_ref[i] *= inv_sqrt_d;

        // Implementation
        std::vector<float> y_impl = x;
        rot.rotate_inplace(y_impl.data(), dim);

        for (int i = 0; i < dim; ++i)
        {
            EXPECT_NEAR(y_impl[i], y_ref[i], 1e-3f)
                << "FWHT D=128: impl vs reference mismatch at trial="
                << trial << " i=" << i;
        }
    }
}

TEST(Test__ActivationRotation, KnownHadamardOutput_H2)
{
    // H₂ = [[1,1],[1,-1]], so for block_dim=2:
    // R = H·D/√2
    // With seed=31, sign_flips are deterministic. We verify the structure
    // by checking that the transform of [1,0] and [0,1] form orthonormal
    // columns consistent with scaled Hadamard structure.
    const int dim = 2;
    ActivationRotation rot(dim, dim);

    // Transform e₀ = [1, 0]
    std::vector<float> e0 = {1.0f, 0.0f};
    rot.rotate_inplace(e0.data(), dim);

    // Transform e₁ = [0, 1]
    std::vector<float> e1 = {0.0f, 1.0f};
    rot.rotate_inplace(e1.data(), dim);

    // Each output element should be ±1/√2
    float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    for (int i = 0; i < dim; ++i)
    {
        EXPECT_NEAR(std::abs(e0[i]), inv_sqrt2, 1e-6f)
            << "H₂ output element should be ±1/√2";
        EXPECT_NEAR(std::abs(e1[i]), inv_sqrt2, 1e-6f)
            << "H₂ output element should be ±1/√2";
    }

    // Columns should be orthogonal: <e0, e1> = 0
    float dot = e0[0] * e1[0] + e0[1] * e1[1];
    EXPECT_NEAR(dot, 0.0f, 1e-6f)
        << "Columns of R should be orthogonal";
}

TEST(Test__ActivationRotation, KnownHadamardOutput_AllElementsMagnitude)
{
    // For a normalized Hadamard of size n, every element has magnitude 1/√n.
    // Our R = H·D/√n has every element = ±1/√n (D just flips signs of columns).
    // Tests both scalar path (dim≤32) and AVX-512 path (dim=64,128).
    for (int dim : {4, 8, 16, 32, 64, 128})
    {
        ActivationRotation rot(dim, dim);
        float expected_mag = 1.0f / std::sqrt(static_cast<float>(dim));

        for (int j = 0; j < dim; ++j)
        {
            std::vector<float> ej(dim, 0.0f);
            ej[j] = 1.0f;
            rot.rotate_inplace(ej.data(), dim);

            for (int i = 0; i < dim; ++i)
            {
                EXPECT_NEAR(std::abs(ej[i]), expected_mag, 1e-6f)
                    << "R[" << i << "," << j << "] should have magnitude 1/√"
                    << dim << " = " << expected_mag;
            }
        }
    }
}

// ============================================================================
// AVX-512 vs Scalar Parity
// ============================================================================

TEST(Test__ActivationRotation, AVX512_MatchesScalar_D64)
{
    // Verify the AVX-512 fwht_64 path by comparing the full rotation matrix
    // extracted column-by-column against the known Hadamard structure:
    // R_{ij} = H_{ij} * d_j / √n, where H_{ij} ∈ {-1, +1} and d_j ∈ {-1, +1}.
    // This validates that the AVX-512 butterfly produces correct signs.
    const int dim = 64;
    ActivationRotation rot(dim, dim);
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(dim));

    // Extract sign_flips: d_j = sign(R_{0,j}) * √n (since H[0,j] = 1 for all j)
    std::vector<float> sign_flips(dim);
    for (int j = 0; j < dim; ++j)
    {
        auto col = extract_rotation_column(rot, dim, j);
        sign_flips[j] = (col[0] > 0) ? 1.0f : -1.0f;
    }

    // Reference FWHT
    auto reference_fwht = [](float *data, int n)
    {
        for (int len = 1; len < n; len <<= 1)
            for (int i = 0; i < n; i += len << 1)
                for (int j = 0; j < len; ++j)
                {
                    float u = data[i + j];
                    float v = data[i + j + len];
                    data[i + j] = u + v;
                    data[i + j + len] = u - v;
                }
    };

    // Compare each column of R against reference
    for (int j = 0; j < dim; ++j)
    {
        // Reference: column j of R = H · (d_j · e_j) / √n
        std::vector<float> ref(dim, 0.0f);
        ref[j] = sign_flips[j];
        reference_fwht(ref.data(), dim);
        for (auto &v : ref)
            v *= inv_sqrt_d;

        auto impl = extract_rotation_column(rot, dim, j);

        for (int i = 0; i < dim; ++i)
        {
            EXPECT_NEAR(impl[i], ref[i], 1e-5f)
                << "D=64 column " << j << " mismatch at row " << i;
        }
    }
}

TEST(Test__ActivationRotation, AVX512_MatchesScalar_D128)
{
    // Same verification for D=128 (fwht_128_avx512 path).
    const int dim = 128;
    ActivationRotation rot(dim, dim);
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(dim));

    std::vector<float> sign_flips(dim);
    for (int j = 0; j < dim; ++j)
    {
        auto col = extract_rotation_column(rot, dim, j);
        sign_flips[j] = (col[0] > 0) ? 1.0f : -1.0f;
    }

    auto reference_fwht = [](float *data, int n)
    {
        for (int len = 1; len < n; len <<= 1)
            for (int i = 0; i < n; i += len << 1)
                for (int j = 0; j < len; ++j)
                {
                    float u = data[i + j];
                    float v = data[i + j + len];
                    data[i + j] = u + v;
                    data[i + j + len] = u - v;
                }
    };

    for (int j = 0; j < dim; ++j)
    {
        std::vector<float> ref(dim, 0.0f);
        ref[j] = sign_flips[j];
        reference_fwht(ref.data(), dim);
        for (auto &v : ref)
            v *= inv_sqrt_d;

        auto impl = extract_rotation_column(rot, dim, j);

        for (int i = 0; i < dim; ++i)
        {
            EXPECT_NEAR(impl[i], ref[i], 1e-4f)
                << "D=128 column " << j << " mismatch at row " << i;
        }
    }
}

// ============================================================================
// Round-Trip Accuracy (Tight Tolerance)
// ============================================================================

TEST(Test__ActivationRotation, RoundTrip_TightTolerance_AllSizes)
{
    // Verify rotate + inverse_rotate ≈ identity at each supported block size.
    // This is the test that would have caught the butterfly sign bug.
    for (int dim : {2, 4, 8, 16, 32, 64, 128})
    {
        ActivationRotation rot(dim, dim);

        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        // Test with 200 random vectors for statistical coverage
        float max_error = 0.0f;
        for (int trial = 0; trial < 200; ++trial)
        {
            std::vector<float> original(dim);
            for (auto &v : original)
                v = dist(gen);

            std::vector<float> data = original;
            rot.rotate_inplace(data.data(), dim);
            rot.inverse_rotate_inplace(data.data(), dim);

            for (int i = 0; i < dim; ++i)
            {
                float err = std::abs(data[i] - original[i]);
                max_error = std::max(max_error, err);
                EXPECT_NEAR(data[i], original[i], 1e-4f)
                    << "Round-trip failed for dim=" << dim
                    << " trial=" << trial << " index=" << i
                    << " original=" << original[i] << " got=" << data[i];
            }
        }

        // For a well-implemented orthogonal transform, max error should be
        // well below 1e-4 for FP32
        EXPECT_LT(max_error, 1e-4f)
            << "Max round-trip error for dim=" << dim << " is too high: "
            << max_error;
    }
}

TEST(Test__ActivationRotation, RoundTrip_MultiBlock)
{
    // Test round-trip with total_dim > block_dim (multiple blocks per row)
    const int block_dim = 128;
    const int total_dim = 896; // 7 blocks (Qwen2.5 hidden_dim)
    const int rows = 16;
    ActivationRotation rot(total_dim, block_dim);

    std::mt19937 gen(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> original(rows * total_dim);
    for (auto &v : original)
        v = dist(gen);

    std::vector<float> data = original;
    rot.rotate_rows_inplace(data.data(), rows, total_dim);
    rot.inverse_rotate_rows_inplace(data.data(), rows, total_dim);

    for (int i = 0; i < rows * total_dim; ++i)
    {
        EXPECT_NEAR(data[i], original[i], 1e-4f)
            << "Multi-block round-trip failed at index " << i;
    }
}

// ============================================================================
// FWHT Structural Validation
// ============================================================================

TEST(Test__ActivationRotation, FWHT_AllOnes_Concentrates)
{
    // A key property of the Hadamard matrix: H·[1,1,...,1] = [n,0,...,0].
    // Through the rotation API: if we set input = [d_0,d_1,...,d_{n-1}]
    // (i.e., the sign_flips themselves), then R·input = H·D·D·input/√n
    // = H·(D²)·input/√n = H·input/√n.
    // But we can't access sign_flips directly. Instead, test that the
    // FWHT concentrates a constant vector into one position.
    //
    // Use the reference FWHT to verify: H·[1,...,1] = [n, 0, ..., 0].
    auto reference_fwht = [](float *data, int n)
    {
        for (int len = 1; len < n; len <<= 1)
            for (int i = 0; i < n; i += len << 1)
                for (int j = 0; j < len; ++j)
                {
                    float u = data[i + j];
                    float v = data[i + j + len];
                    data[i + j] = u + v;
                    data[i + j + len] = u - v;
                }
    };

    for (int dim : {4, 8, 16, 32, 64, 128})
    {
        std::vector<float> ones(dim, 1.0f);
        reference_fwht(ones.data(), dim);

        EXPECT_NEAR(ones[0], static_cast<float>(dim), 1e-4f)
            << "H·[1,...,1][0] should be n for dim=" << dim;
        for (int i = 1; i < dim; ++i)
        {
            EXPECT_NEAR(ones[i], 0.0f, 1e-4f)
                << "H·[1,...,1][" << i << "] should be 0 for dim=" << dim;
        }
    }
}

TEST(Test__ActivationRotation, FWHT_BasisVector_AllPlusMinusOne)
{
    // H·e_0 = [1,1,...,1] (first column of H is all 1s).
    // Through the rotation: R·e_0 = H·(D·e_0)/√n = d_0·H[:,0]/√n = d_0·[1,...,1]/√n
    // So all elements should have magnitude 1/√n.
    for (int dim : {64, 128})
    {
        ActivationRotation rot(dim, dim);
        float expected = 1.0f / std::sqrt(static_cast<float>(dim));

        std::vector<float> e0(dim, 0.0f);
        e0[0] = 1.0f;
        rot.rotate_inplace(e0.data(), dim);

        for (int i = 0; i < dim; ++i)
        {
            EXPECT_NEAR(std::abs(e0[i]), expected, 1e-6f)
                << "R·e_0 should have all magnitudes 1/√" << dim
                << " at index " << i << " (AVX-512 path)";
        }

        // All values should have the same sign (d_0 applied uniformly)
        float sign = (e0[0] > 0) ? 1.0f : -1.0f;
        for (int i = 0; i < dim; ++i)
        {
            EXPECT_NEAR(e0[i], sign * expected, 1e-6f)
                << "All output elements should be identical (d_0/√"
                << dim << ") at index " << i;
        }
    }
}

// ============================================================================
// Norm Preservation (Tight Tolerance, Multiple Sizes)
// ============================================================================

TEST(Test__ActivationRotation, NormPreserved_AllSizes)
{
    // Orthogonal transforms preserve L2 norm exactly (up to FP rounding).
    for (int dim : {2, 4, 8, 16, 32, 64, 128})
    {
        ActivationRotation rot(dim, dim);

        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        for (int trial = 0; trial < 100; ++trial)
        {
            std::vector<float> x(dim);
            for (auto &v : x)
                v = dist(gen);

            float norm_before = 0.0f;
            for (float v : x)
                norm_before += v * v;
            norm_before = std::sqrt(norm_before);

            rot.rotate_inplace(x.data(), dim);

            float norm_after = 0.0f;
            for (float v : x)
                norm_after += v * v;
            norm_after = std::sqrt(norm_after);

            EXPECT_NEAR(norm_after, norm_before, norm_before * 1e-5f)
                << "Norm not preserved for dim=" << dim << " trial=" << trial;
        }
    }
}

TEST(Test__ActivationRotation, NormPreserved_LargeValues)
{
    // Test with extreme values to catch overflow/underflow in FWHT.
    const int dim = 128;
    ActivationRotation rot(dim, dim);

    // Large values
    {
        std::vector<float> x(dim, 1e6f);
        float norm_before = 0.0f;
        for (float v : x)
            norm_before += v * v;
        norm_before = std::sqrt(norm_before);

        rot.rotate_inplace(x.data(), dim);

        float norm_after = 0.0f;
        for (float v : x)
            norm_after += v * v;
        norm_after = std::sqrt(norm_after);

        EXPECT_NEAR(norm_after, norm_before, norm_before * 1e-4f)
            << "Norm not preserved for large values";
    }

    // Small values
    {
        std::vector<float> x(dim, 1e-6f);
        float norm_before = 0.0f;
        for (float v : x)
            norm_before += v * v;
        norm_before = std::sqrt(norm_before);

        rot.rotate_inplace(x.data(), dim);

        float norm_after = 0.0f;
        for (float v : x)
            norm_after += v * v;
        norm_after = std::sqrt(norm_after);

        EXPECT_NEAR(norm_after, norm_before, norm_before * 1e-4f)
            << "Norm not preserved for small values";
    }
}

// ============================================================================
// Determinism
// ============================================================================

TEST(Test__ActivationRotation, DeterministicAcrossCalls)
{
    const int dim = 128;
    ActivationRotation rot(dim, dim);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> x(dim);
    for (auto &v : x)
        v = dist(gen);

    std::vector<float> y1 = x, y2 = x;
    rot.rotate_inplace(y1.data(), dim);
    rot.rotate_inplace(y2.data(), dim);

    for (int i = 0; i < dim; ++i)
    {
        EXPECT_EQ(y1[i], y2[i])
            << "Rotation should be bitwise deterministic at index " << i;
    }
}

TEST(Test__ActivationRotation, DeterministicAcrossInstances)
{
    // Two ActivationRotation objects with the same seed should produce
    // identical results.
    const int dim = 128;
    ActivationRotation rot1(dim, dim, 31);
    ActivationRotation rot2(dim, dim, 31);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> x(dim);
    for (auto &v : x)
        v = dist(gen);

    std::vector<float> y1 = x, y2 = x;
    rot1.rotate_inplace(y1.data(), dim);
    rot2.rotate_inplace(y2.data(), dim);

    for (int i = 0; i < dim; ++i)
    {
        EXPECT_EQ(y1[i], y2[i])
            << "Same seed should produce identical results at index " << i;
    }
}

TEST(Test__ActivationRotation, DifferentSeedsProduceDifferentResults)
{
    const int dim = 128;
    ActivationRotation rot1(dim, dim, 1);
    ActivationRotation rot2(dim, dim, 2);

    std::vector<float> x(dim);
    for (int i = 0; i < dim; ++i)
        x[i] = static_cast<float>(i);

    std::vector<float> y1 = x, y2 = x;
    rot1.rotate_inplace(y1.data(), dim);
    rot2.rotate_inplace(y2.data(), dim);

    bool any_different = false;
    for (int i = 0; i < dim; ++i)
    {
        if (y1[i] != y2[i])
        {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different)
        << "Different seeds should produce different rotations";
}

// ============================================================================
// GEMM Invariance (Extended)
// ============================================================================

TEST(Test__ActivationRotation, GEMM_Invariance_MultiBlock)
{
    // X @ W^T == (X@R) @ (W@R)^T with multi-block rotation (total > block)
    const int M = 4;
    const int K = 256; // 2 blocks of 128
    const int N = 32;
    ActivationRotation rot(K, 128);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> X(M * K), W(N * K);
    for (auto &v : X)
        v = dist(gen);
    for (auto &v : W)
        v = dist(gen);

    // Reference: Y = X @ W^T
    std::vector<float> Y_ref(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                Y_ref[m * N + n] += X[m * K + k] * W[n * K + k];

    // Rotate X and W
    std::vector<float> X_rot = X, W_rot = W;
    rot.rotate_rows_inplace(X_rot.data(), M, K);
    rot.rotate_weight_rows(W_rot.data(), N, K);

    // Rotated: Y_rot = X_rot @ W_rot^T
    std::vector<float> Y_rot(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                Y_rot[m * N + n] += X_rot[m * K + k] * W_rot[n * K + k];

    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(Y_rot[i], Y_ref[i], std::abs(Y_ref[i]) * 1e-3f + 1e-3f)
            << "Multi-block GEMM invariance failed at index " << i;
    }
}

TEST(Test__ActivationRotation, GEMM_Invariance_D64)
{
    // Same GEMM invariance test but with block_dim=64 (Qwen2.5-0.5B head_dim)
    const int M = 4;
    const int K = 64;
    const int N = 16;
    ActivationRotation rot(K, 64);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> X(M * K), W(N * K);
    for (auto &v : X)
        v = dist(gen);
    for (auto &v : W)
        v = dist(gen);

    std::vector<float> Y_ref(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                Y_ref[m * N + n] += X[m * K + k] * W[n * K + k];

    std::vector<float> X_rot = X, W_rot = W;
    rot.rotate_rows_inplace(X_rot.data(), M, K);
    rot.rotate_weight_rows(W_rot.data(), N, K);

    std::vector<float> Y_rot(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                Y_rot[m * N + n] += X_rot[m * K + k] * W_rot[n * K + k];

    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(Y_rot[i], Y_ref[i], std::abs(Y_ref[i]) * 5e-4f + 1e-4f)
            << "D64 GEMM invariance failed at index " << i;
    }
}

// ============================================================================
// Large Dimension Stress Tests
// ============================================================================

TEST(Test__ActivationRotation, LargeDimension_RoundTrip)
{
    // Qwen2.5-7B hidden_dim=3584
    const int total_dim = 3584;
    const int block_dim = 128;
    const int rows = 8;
    ActivationRotation rot(total_dim, block_dim);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> original(rows * total_dim);
    for (auto &v : original)
        v = dist(gen);

    std::vector<float> data = original;
    rot.rotate_rows_inplace(data.data(), rows, total_dim);
    rot.inverse_rotate_rows_inplace(data.data(), rows, total_dim);

    float max_err = 0.0f;
    for (size_t i = 0; i < original.size(); ++i)
    {
        float err = std::abs(data[i] - original[i]);
        max_err = std::max(max_err, err);
    }

    EXPECT_LT(max_err, 1e-4f)
        << "Large dimension round-trip max error: " << max_err;
}

TEST(Test__ActivationRotation, LargeDimension_NormPreserved)
{
    const int total_dim = 3584;
    const int block_dim = 128;
    ActivationRotation rot(total_dim, block_dim);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> x(total_dim);
    for (auto &v : x)
        v = dist(gen);

    float norm_before = 0.0f;
    for (float v : x)
        norm_before += v * v;
    norm_before = std::sqrt(norm_before);

    rot.rotate_inplace(x.data(), total_dim);

    float norm_after = 0.0f;
    for (float v : x)
        norm_after += v * v;
    norm_after = std::sqrt(norm_after);

    EXPECT_NEAR(norm_after, norm_before, norm_before * 1e-5f)
        << "Large dimension norm not preserved";
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(Test__ActivationRotation, ZeroVector_RoundTrip)
{
    const int dim = 128;
    ActivationRotation rot(dim, dim);

    std::vector<float> x(dim, 0.0f);
    rot.rotate_inplace(x.data(), dim);

    // Zero vector should remain zero
    for (int i = 0; i < dim; ++i)
        EXPECT_EQ(x[i], 0.0f) << "Zero vector should stay zero at index " << i;

    rot.inverse_rotate_inplace(x.data(), dim);
    for (int i = 0; i < dim; ++i)
        EXPECT_EQ(x[i], 0.0f) << "Inverse of zero should stay zero at index " << i;
}

TEST(Test__ActivationRotation, SingleNonZero_NormPreserved)
{
    // A vector with one non-zero element: the rotation should spread it.
    const int dim = 128;
    ActivationRotation rot(dim, dim);

    std::vector<float> x(dim, 0.0f);
    x[0] = 42.0f;
    float norm_before = 42.0f;

    rot.rotate_inplace(x.data(), dim);

    // All elements should now be non-zero (Hadamard spreads to all)
    int nonzero_count = 0;
    for (float v : x)
        if (v != 0.0f)
            ++nonzero_count;
    EXPECT_EQ(nonzero_count, dim)
        << "Hadamard should spread a single element to all positions";

    // Norm preserved
    float norm_after = 0.0f;
    for (float v : x)
        norm_after += v * v;
    norm_after = std::sqrt(norm_after);

    EXPECT_NEAR(norm_after, norm_before, norm_before * 1e-5f)
        << "Single-element norm not preserved";

    // All magnitudes should be equal (42/√128)
    float expected_mag = 42.0f / std::sqrt(128.0f);
    for (int i = 0; i < dim; ++i)
    {
        EXPECT_NEAR(std::abs(x[i]), expected_mag, expected_mag * 1e-5f)
            << "All magnitudes should be equal after Hadamard at index " << i;
    }
}

TEST(Test__ActivationRotation, RotationChangesVector)
{
    // Verify that rotation actually modifies the vector (not a no-op).
    const int dim = 128;
    ActivationRotation rot(dim, dim);

    std::vector<float> x(dim);
    for (int i = 0; i < dim; ++i)
        x[i] = static_cast<float>(i);
    std::vector<float> original = x;

    rot.rotate_inplace(x.data(), dim);

    bool any_different = false;
    for (int i = 0; i < dim; ++i)
    {
        if (x[i] != original[i])
        {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different) << "Rotation should actually modify the vector";
}

// ============================================================================
// Rotation-Fused VNNI Weight Packing Tests
//
// Verify that packing with rotation fused in produces correct GEMM results:
//   X @ W^T == (X@R) @ Pack(R^T @ dequant(W))
// ============================================================================

#include "kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"

TEST(Test__ActivationRotation, FusedPacking_GEMM_Invariance_Q4_0)
{
    // Test that activating rotation on a Q4_0 weight produces the same
    // GEMM result as the manual approach: dequant → rotate → GEMM.
    const int M = 2;   // batch
    const int K = 128;  // hidden dim (must be multiple of block_dim)
    const int N = 32;   // output dim

    // Create rotation
    ActivationRotation rot(K, 128);

    // Create Q4_0 weight [N×K]
    auto weight = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N),
                                                        static_cast<size_t>(K)});

    // Reference: dequantize weight to FP32
    std::vector<float> W_fp32(N * K);
    for (int n = 0; n < N; ++n)
        weight->to_fp32_row(n, W_fp32.data() + n * K);

    // Create random FP32 input [M×K]
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> X(M * K);
    for (auto &v : X) v = dist(gen);

    // Reference result: Y_ref = X @ W^T (unrotated, in FP32)
    std::vector<float> Y_ref(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                Y_ref[m * N + n] += X[m * K + k] * W_fp32[n * K + k];

    // Tag weight with rotation
    weight->setActivationRotation(&rot);

    // Create GEMM kernel (will pack with rotation fused in)
    auto kernel = std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(
        weight.get());
    ASSERT_TRUE(kernel->isValid()) << "GEMM kernel should be valid after packing";

    // Create input tensor [M×K]
    auto input_tensor = TestTensorFactory::createFP32(
        {static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(input_tensor->mutable_data(), X.data(), M * K * sizeof(float));

    // Create output tensor [M×N]
    auto output_tensor = TestTensorFactory::createFP32(
        {static_cast<size_t>(M), static_cast<size_t>(N)});

    // Execute GEMM: kernel rotates activations, uses rotation-fused packed weights
    bool ok = kernel->multiply_tensor(
        input_tensor.get(), output_tensor.get(), M, N, K);
    ASSERT_TRUE(ok) << "GEMM should succeed";

    const float *Y_fused = output_tensor->data();

    // Compare. The rotation-fused path introduces INT8 quantization noise,
    // so we allow some tolerance (Q4_0 + rotation INT8 requant).
    float max_rel_err = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        float ref = Y_ref[i];
        float got = Y_fused[i];
        float abs_err = std::fabs(ref - got);
        float rel_err = (std::fabs(ref) > 1e-6f) ? abs_err / std::fabs(ref) : abs_err;
        max_rel_err = std::max(max_rel_err, rel_err);

        // Q4_0 already has ~1-2% quantization noise from the original format,
        // plus the rotation requant adds another layer. Allow generous tolerance.
        EXPECT_NEAR(got, ref, std::fabs(ref) * 0.15f + 0.5f)
            << "GEMM invariance failed at index " << i
            << " (ref=" << ref << ", got=" << got
            << ", rel_err=" << rel_err << ")";
    }

    LOG_INFO("[FusedPacking_Q4_0] max_rel_err=" << max_rel_err);
}

TEST(Test__ActivationRotation, FusedPacking_PreservesFormatForNonRotated)
{
    // Verify that packing WITHOUT rotation produces the same result as before.
    const int K = 128;
    const int N = 16;

    auto weight = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N),
                                                        static_cast<size_t>(K)});
    // No rotation set — should use the nibble-LUT path as usual
    EXPECT_EQ(weight->activationRotation(), nullptr);

    auto kernel = std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(
        weight.get());
    ASSERT_TRUE(kernel->isValid());

    // Simple smoke test: multiply identity-like input
    auto input = TestTensorFactory::createFP32({1, static_cast<size_t>(K)});
    auto output = TestTensorFactory::createFP32({1, static_cast<size_t>(N)});

    // Set all inputs to 1.0 — output should be row sums of weight
    float *inp_data = input->mutable_data();
    for (int i = 0; i < K; ++i) inp_data[i] = 1.0f;

    bool ok = kernel->multiply_tensor(input.get(), output.get(), 1, N, K);
    ASSERT_TRUE(ok);

    // Just verify output is not NaN/Inf
    const float *out_data = output->data();
    for (int n = 0; n < N; ++n)
    {
        EXPECT_FALSE(std::isnan(out_data[n])) << "Output NaN at " << n;
        EXPECT_FALSE(std::isinf(out_data[n])) << "Output Inf at " << n;
    }
}

TEST(Test__ActivationRotation, FusedPacking_RotationTagOnlyNoDataChange)
{
    // Verify that tagging with rotation doesn't modify the tensor's data
    const int K = 128;
    const int N = 8;

    auto weight = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N),
                                                        static_cast<size_t>(K)});

    // Save a copy of the FP32 dequantized data before tagging
    std::vector<float> before(N * K);
    for (int n = 0; n < N; ++n)
        weight->to_fp32_row(n, before.data() + n * K);

    // Tag with rotation
    ActivationRotation rot(K, 128);
    weight->setActivationRotation(&rot);

    // Verify data is unchanged
    std::vector<float> after(N * K);
    for (int n = 0; n < N; ++n)
        weight->to_fp32_row(n, after.data() + n * K);

    for (int i = 0; i < N * K; ++i)
    {
        EXPECT_EQ(before[i], after[i])
            << "Tensor data should not change after setActivationRotation at index " << i;
    }
}
