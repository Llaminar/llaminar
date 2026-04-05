/**
 * @file Test__ExpPolynomialRegression.cpp
 * @brief Regression tests for the AVX512 exp() polynomial bug
 *
 * The bug: GatedRMSNormStage and AttentionOutputGateStage used a fast exp()
 * polynomial with Taylor coefficients of 2^f, but fed the natural-log fractional
 * part f = x - n*ln2 instead of the base-2 fractional part f = x*log2e - n.
 * This caused 3-11% relative error in SiLU/sigmoid on typical gate values.
 *
 * These tests exercise the AVX512 path (>= 16 elements per row) and compare
 * against std::exp-based reference implementations with tight tolerances.
 *
 * Regression for: commit fixing range-reduction mismatch in exp() polynomial
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>

#include "execution/compute_stages/stages/GatedRMSNormStage.h"
#include "execution/compute_stages/stages/AttentionOutputGateStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/Tensors.h"

using namespace llaminar2;

namespace
{
    std::unique_ptr<IDeviceContext> makeCPUContext()
    {
        return std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 1);
    }

    std::shared_ptr<FP32Tensor> makeFP32(const std::vector<size_t> &shape)
    {
        auto t = std::make_shared<FP32Tensor>(shape, DeviceId::cpu());
        std::memset(t->mutable_data(), 0, t->numel() * sizeof(float));
        return t;
    }

    std::shared_ptr<FP32Tensor> makeFP32Const(const std::vector<size_t> &shape, float val)
    {
        auto t = std::make_shared<FP32Tensor>(shape, DeviceId::cpu());
        float *d = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
            d[i] = val;
        return t;
    }

    // Reference SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    float ref_silu(float x) { return x / (1.0f + std::exp(-x)); }

    // Reference sigmoid: 1 / (1 + exp(-x))
    float ref_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

} // namespace

// ============================================================================
// GatedRMSNormStage: SiLU accuracy on AVX512-sized vectors
// ============================================================================

TEST(Test__ExpPolynomialRegression, GatedRMSNorm_SiLU_AVX512Path_UniformGates)
{
    // Use norm_dim >= 16 to ensure the AVX512 path is exercised.
    // With gate_silu=true, the gate is passed through SiLU before multiplying.
    // We set input=1 and gamma=1 so the output is purely SiLU(gate).
    const size_t norm_dim = 128; // 8 AVX512 iterations of 16 floats
    const int seq_len = 1;

    auto input = makeFP32Const({1, norm_dim}, 1.0f);
    auto gamma = makeFP32Const({norm_dim}, 1.0f);
    auto output = makeFP32({1, norm_dim});

    // Gate values spanning the typical range seen in GDN layers
    auto gate = std::make_shared<FP32Tensor>(std::vector<size_t>{1, norm_dim}, DeviceId::cpu());
    float *g = gate->mutable_data();
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (size_t i = 0; i < norm_dim; ++i)
        g[i] = dist(rng);

    GatedRMSNormStage::Params params;
    params.input = input.get();
    params.gate = gate.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-6f;
    params.seq_len = seq_len;
    params.norm_dim = static_cast<int>(norm_dim);
    params.gate_silu = true;

    auto ctx = makeCPUContext();
    GatedRMSNormStage stage(params);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // With input=1 and gamma=1, RMSNorm([1,1,...,1]) = 1/sqrt(1+eps) ≈ 1.0
    // output ≈ SiLU(gate)
    const float inv_rms = 1.0f / std::sqrt(1.0f + params.eps);
    const float *out = output->data();
    float max_rel_err = 0.0f;

    for (size_t i = 0; i < norm_dim; ++i)
    {
        const float expected = inv_rms * ref_silu(g[i]);
        const float actual = out[i];
        const float abs_err = std::abs(actual - expected);
        const float rel_err = (std::abs(expected) > 1e-6f)
                                  ? abs_err / std::abs(expected)
                                  : abs_err;
        max_rel_err = std::max(max_rel_err, rel_err);

        // The old buggy code had 3-11% relative error.
        // The fixed code should be < 0.1% relative error.
        EXPECT_NEAR(actual, expected, std::abs(expected) * 0.005f + 1e-6f)
            << "Element " << i << " gate=" << g[i]
            << " expected=" << expected << " actual=" << actual
            << " rel_err=" << rel_err;
    }

    // Overall max relative error should be well below 1%
    EXPECT_LT(max_rel_err, 0.005f)
        << "Max relative SiLU error should be < 0.5% (was 3-11% before fix)";
}

TEST(Test__ExpPolynomialRegression, GatedRMSNorm_SiLU_AVX512Path_ExtremeValues)
{
    // Test precision at domain boundaries and near-zero inputs
    const size_t norm_dim = 64;

    auto input = makeFP32Const({1, norm_dim}, 1.0f);
    auto gamma = makeFP32Const({norm_dim}, 1.0f);
    auto output = makeFP32({1, norm_dim});

    auto gate = std::make_shared<FP32Tensor>(std::vector<size_t>{1, norm_dim}, DeviceId::cpu());
    float *g = gate->mutable_data();
    // Specific values that maximize the old bug's error
    const float test_values[] = {
        0.0f, 0.1f, 0.3f, 0.5f, 1.0f, 2.0f, 3.0f, 5.0f,
        -0.1f, -0.3f, -0.5f, -1.0f, -2.0f, -3.0f, -5.0f, -10.0f,
        0.01f, 0.05f, 0.25f, 0.75f, 1.5f, 4.0f, 7.0f, 10.0f,
        -0.01f, -0.05f, -0.25f, -0.75f, -1.5f, -4.0f, -7.0f, -15.0f,
        20.0f, 40.0f, 80.0f, -20.0f, -40.0f, -80.0f, 0.001f, -0.001f,
        0.693f, -0.693f, 1.386f, -1.386f, 2.302f, -2.302f, 88.0f, -88.0f,
        // Last 16: repeat values near ln(2) where the old bug was worst
        0.35f, -0.35f, 0.69f, -0.69f, 1.38f, -1.38f, 2.07f, -2.07f,
        0.17f, -0.17f, 0.52f, -0.52f, 1.04f, -1.04f, 1.73f, -1.73f,
    };
    static_assert(sizeof(test_values) / sizeof(float) == 64);
    std::memcpy(g, test_values, sizeof(test_values));

    GatedRMSNormStage::Params params;
    params.input = input.get();
    params.gate = gate.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-6f;
    params.seq_len = 1;
    params.norm_dim = static_cast<int>(norm_dim);
    params.gate_silu = true;

    auto ctx = makeCPUContext();
    GatedRMSNormStage stage(params);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float inv_rms = 1.0f / std::sqrt(1.0f + params.eps);
    const float *out = output->data();

    for (size_t i = 0; i < norm_dim; ++i)
    {
        const float expected = inv_rms * ref_silu(g[i]);
        EXPECT_NEAR(out[i], expected, std::abs(expected) * 0.005f + 1e-6f)
            << "Element " << i << " gate=" << g[i];
    }
}

TEST(Test__ExpPolynomialRegression, GatedRMSNorm_SiLU_MultiRow)
{
    // Multiple rows: ensures the per-head loop uses the fast path for each head
    const size_t norm_dim = 128;
    const int seq_len = 4;
    const size_t total = seq_len * norm_dim;

    auto input = makeFP32Const({static_cast<size_t>(seq_len), norm_dim}, 1.0f);
    auto gamma = makeFP32Const({norm_dim}, 1.0f);
    auto output = makeFP32({static_cast<size_t>(seq_len), norm_dim});

    auto gate = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), norm_dim}, DeviceId::cpu());
    float *g = gate->mutable_data();
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    for (size_t i = 0; i < total; ++i)
        g[i] = dist(rng);

    GatedRMSNormStage::Params params;
    params.input = input.get();
    params.gate = gate.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-6f;
    params.seq_len = seq_len;
    params.norm_dim = static_cast<int>(norm_dim);
    params.gate_silu = true;

    auto ctx = makeCPUContext();
    GatedRMSNormStage stage(params);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float inv_rms = 1.0f / std::sqrt(1.0f + params.eps);
    const float *out = output->data();

    for (size_t i = 0; i < total; ++i)
    {
        const float expected = inv_rms * ref_silu(g[i]);
        EXPECT_NEAR(out[i], expected, std::abs(expected) * 0.005f + 1e-6f)
            << "Element " << i << " gate=" << g[i];
    }
}

// ============================================================================
// AttentionOutputGateStage: sigmoid accuracy on AVX512-sized vectors
// ============================================================================

TEST(Test__ExpPolynomialRegression, AttentionOutputGate_Sigmoid_AVX512Path)
{
    // The stage computes: output = input * sigmoid(gate)
    // Use 64 elements to exercise 4 AVX512 iterations.
    const size_t dim = 64;

    auto input = makeFP32Const({1, dim}, 1.0f); // input=1 → output = sigmoid(gate)
    auto output = makeFP32({1, dim});

    auto gate = std::make_shared<FP32Tensor>(std::vector<size_t>{1, dim}, DeviceId::cpu());
    float *g = gate->mutable_data();
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    for (size_t i = 0; i < dim; ++i)
        g[i] = dist(rng);

    AttentionOutputGateStage::Params params;
    params.input = input.get();
    params.gate = gate.get();
    params.output = output.get();
    params.seq_len = 1;

    auto ctx = makeCPUContext();
    AttentionOutputGateStage stage(params);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float *out = output->data();
    float max_rel_err = 0.0f;

    for (size_t i = 0; i < dim; ++i)
    {
        const float expected = ref_sigmoid(g[i]);
        const float actual = out[i];
        const float abs_err = std::abs(actual - expected);
        const float rel_err = (expected > 1e-6f) ? abs_err / expected : abs_err;
        max_rel_err = std::max(max_rel_err, rel_err);

        EXPECT_NEAR(actual, expected, expected * 0.005f + 1e-6f)
            << "Element " << i << " gate=" << g[i]
            << " expected_sigmoid=" << expected << " actual=" << actual;
    }

    EXPECT_LT(max_rel_err, 0.005f)
        << "Max relative sigmoid error should be < 0.5% (was 2-10% before fix)";
}

TEST(Test__ExpPolynomialRegression, AttentionOutputGate_Sigmoid_ScalesInput)
{
    // Verify sigmoid * input interaction with AVX512-sized vectors
    const size_t dim = 128;

    auto input = std::make_shared<FP32Tensor>(std::vector<size_t>{1, dim}, DeviceId::cpu());
    auto gate = std::make_shared<FP32Tensor>(std::vector<size_t>{1, dim}, DeviceId::cpu());
    auto output = makeFP32({1, dim});

    float *inp = input->mutable_data();
    float *g = gate->mutable_data();
    std::mt19937 rng(99);
    std::normal_distribution<float> inp_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> gate_dist(-4.0f, 4.0f);
    for (size_t i = 0; i < dim; ++i)
    {
        inp[i] = inp_dist(rng);
        g[i] = gate_dist(rng);
    }

    AttentionOutputGateStage::Params params;
    params.input = input.get();
    params.gate = gate.get();
    params.output = output.get();
    params.seq_len = 1;

    auto ctx = makeCPUContext();
    AttentionOutputGateStage stage(params);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float *out = output->data();
    for (size_t i = 0; i < dim; ++i)
    {
        const float expected = inp[i] * ref_sigmoid(g[i]);
        EXPECT_NEAR(out[i], expected, std::abs(expected) * 0.005f + 1e-6f)
            << "Element " << i << " input=" << inp[i] << " gate=" << g[i];
    }
}

// ============================================================================
// Compounding error test: verify SiLU error doesn't compound over many layers
// ============================================================================

TEST(Test__ExpPolynomialRegression, GatedRMSNorm_SiLU_CompoundingError)
{
    // Simulate what happens when SiLU error compounds across 24 GDN layers.
    // Run the same data through GatedRMSNorm 24 times and measure final drift.
    const size_t norm_dim = 128;
    const int n_layers = 24;

    auto gamma = makeFP32Const({norm_dim}, 1.0f);

    // Start with a realistic activation
    auto activation = std::make_shared<FP32Tensor>(
        std::vector<size_t>{1, norm_dim}, DeviceId::cpu());
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < norm_dim; ++i)
        activation->mutable_data()[i] = dist(rng);

    // Reference: compute with exact SiLU
    std::vector<float> ref_act(norm_dim);
    std::memcpy(ref_act.data(), activation->data(), norm_dim * sizeof(float));

    auto ctx = makeCPUContext();

    for (int layer = 0; layer < n_layers; ++layer)
    {
        // Generate gate values for this "layer"
        auto gate = std::make_shared<FP32Tensor>(
            std::vector<size_t>{1, norm_dim}, DeviceId::cpu());
        std::mt19937 layer_rng(layer * 1000 + 42);
        std::normal_distribution<float> gate_dist(0.0f, 2.0f);
        float *g = gate->mutable_data();
        for (size_t i = 0; i < norm_dim; ++i)
            g[i] = gate_dist(layer_rng);

        // Stage execution
        auto output = makeFP32({1, norm_dim});
        GatedRMSNormStage::Params params;
        params.input = activation.get();
        params.gate = gate.get();
        params.output = output.get();
        params.gamma = gamma.get();
        params.eps = 1e-6f;
        params.seq_len = 1;
        params.norm_dim = static_cast<int>(norm_dim);
        params.gate_silu = true;

        GatedRMSNormStage stage(params);
        ASSERT_TRUE(stage.execute(ctx.get()));

        // Reference computation
        float sum_sq = 0.0f;
        for (size_t i = 0; i < norm_dim; ++i)
            sum_sq += ref_act[i] * ref_act[i];
        float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(norm_dim) + 1e-6f);

        for (size_t i = 0; i < norm_dim; ++i)
            ref_act[i] = ref_act[i] * inv_rms * ref_silu(g[i]);

        // Copy stage output back to activation for next layer
        std::memcpy(activation->mutable_data(), output->data(), norm_dim * sizeof(float));
    }

    // Compare after 24 layers of compounding
    const float *final_act = activation->data();
    double dot = 0.0, mag_a = 0.0, mag_b = 0.0;
    for (size_t i = 0; i < norm_dim; ++i)
    {
        dot += static_cast<double>(final_act[i]) * static_cast<double>(ref_act[i]);
        mag_a += static_cast<double>(final_act[i]) * static_cast<double>(final_act[i]);
        mag_b += static_cast<double>(ref_act[i]) * static_cast<double>(ref_act[i]);
    }
    const double cosine = dot / (std::sqrt(mag_a) * std::sqrt(mag_b) + 1e-30);

    // With the fix, cosine should be > 0.999 after 24 layers.
    // The old buggy code gave ~0.85 cosine after 24 layers.
    EXPECT_GT(cosine, 0.999)
        << "After 24 layers of SiLU, cosine vs reference should be > 0.999 "
        << "(was ~0.85 with the old bug). Actual: " << cosine;
}
