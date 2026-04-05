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
 */

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <numeric>

#include "kernels/cpu/rotation/ActivationRotation.h"
#include "kernels/cpu/rotation/ModelWeightRotation.h"
#include "models/GraphTypes.h"
#include "../../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

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
    data[64] = -80.0f;  // Spike
    data[128] = 50.0f;  // Spike

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

    // Pointer should be different (new tensor created)
    EXPECT_NE(ptr, weight.get()) << "rotateWeight should create a new tensor";

    // The new tensor should have rotation metadata
    EXPECT_NE(ptr->activationRotation(), nullptr)
        << "Rotated tensor should have activationRotation set";
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

    // FA weights with K=64 should be rotated (hidden rotation)
    EXPECT_NE(layer.wq, orig_wq)
        << "wq should be replaced with rotated version";
    EXPECT_NE(layer.wq->activationRotation(), nullptr)
        << "Rotated wq should have rotation metadata";

    // FFN down with K=128 should be rotated (ffn rotation)
    EXPECT_NE(layer.down_proj, orig_down)
        << "down_proj should be replaced with rotated version";
    EXPECT_NE(layer.down_proj->activationRotation(), nullptr)
        << "Rotated down_proj should have rotation metadata";
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

    // LM head should be rotated (K=64 = hidden_dim)
    EXPECT_NE(weights.lm_head, lm_head.get())
        << "LM head should be replaced with rotated version";

    // Layer accessor should return rotated weights
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
    const int M = 2;  // sequence length
    const int K = 128; // hidden dim (1 block)
    const int N = 16;  // output dim

    ActivationRotation rot(K, 128);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Create X [M×K] and W [N×K]
    std::vector<float> X(M * K), W(N * K);
    for (auto &v : X) v = dist(gen);
    for (auto &v : W) v = dist(gen);

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
