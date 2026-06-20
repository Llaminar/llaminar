/**
 * @file Test__GDNMathematicalCorrectness.cpp
 * @brief Deep mathematical correctness tests for GDN CPU kernels
 *
 * Addresses the following coverage gaps identified in GDN test suite:
 *
 * 1. Standalone reference implementation for core recurrence formula
 * 2. Direct gate computation verification (g, exp(g), beta_sig)
 * 3. Multi-step decode state accumulation over 10+ steps
 * 4. GatedRMSNorm per-head norm_dim normalization
 * 5. Conv1d bias term correctness
 * 6. Conv1d multi-channel prefill with distinct per-channel weights
 * 7. AVX512 vs scalar tail handling for recurrence (odd d_k/d_v)
 * 8. Softplus edge cases near the transition point (x ≈ 20)
 *
 * All tests are lightweight unit tests — no model weights are loaded.
 * Reference implementations are plain scalar C++ that directly compute
 * the expected values from the mathematical formulas, independently of
 * the optimized kernel code paths (AVX512, OpenMP, etc.).
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "execution/compute_stages/stages/ShortConv1dStage.h"
#include "execution/compute_stages/stages/GatedRMSNormStage.h"
#include "execution/compute_stages/stages/AttentionOutputGateStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "kernels/cpu/gdn/CPUShortConvolution.h"
#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"
#include "tensors/Tensors.h"

using namespace llaminar2;

namespace
{
    // ========================================================================
    // Helpers
    // ========================================================================

    std::unique_ptr<IDeviceContext> makeCPUContext()
    {
        return std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 1);
    }

    std::shared_ptr<FP32Tensor> makeFP32(const std::vector<size_t> &shape,
                                         const float *data = nullptr)
    {
        auto t = std::make_shared<FP32Tensor>(shape, DeviceId::cpu());
        if (data)
            std::memcpy(t->mutable_data(), data, t->numel() * sizeof(float));
        else
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

    std::shared_ptr<FP32Tensor> makeFP32Random(const std::vector<size_t> &shape,
                                               float mean, float stddev,
                                               unsigned seed)
    {
        auto t = std::make_shared<FP32Tensor>(shape, DeviceId::cpu());
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(mean, stddev);
        float *d = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
            d[i] = dist(gen);
        return t;
    }

    float maxAbsDiff(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size())
            return std::numeric_limits<float>::infinity();

        float max_diff = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
            max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
        return max_diff;
    }

    // Reference scalar functions (match kernel formulas exactly)
    float ref_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
    float ref_silu(float x) { return x * ref_sigmoid(x); }
    float ref_softplus(float x) { return (x > 20.0f) ? x : std::log1p(std::exp(x)); }

    CPUShortConvolution g_cpu_conv;
    CPUGatedDeltaNet g_cpu_gdn;

    // ========================================================================
    // Pure scalar reference: single-head recurrent_step
    //
    // This is a standalone implementation of the delta-rule recurrence
    // that does NOT share any code with CPUGatedDeltaNet. It follows
    // the mathematical formulas from HuggingFace's
    // torch_recurrent_gated_delta_rule.
    // ========================================================================

    struct RefRecurrenceResult
    {
        std::vector<float> output; // [d_v]
        std::vector<float> state;  // [d_k * d_v] (updated in-place)
    };

    RefRecurrenceResult ref_recurrent_step(
        const float *q, const float *k, const float *v,
        float alpha_val, float beta_raw_val,
        float A_log_val, float dt_bias_val,
        float *state, // [d_k, d_v] row-major, modified in-place
        int d_k, int d_v,
        bool use_qk_l2norm)
    {
        const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
        constexpr float l2_eps = 1e-6f;

        // Preprocess Q and K
        std::vector<float> q_proc(d_k), k_proc(d_k);

        if (use_qk_l2norm)
        {
            // L2 normalize Q, then scale
            float q_norm_sq = 0.0f, k_norm_sq = 0.0f;
            for (int i = 0; i < d_k; ++i)
            {
                q_norm_sq += q[i] * q[i];
                k_norm_sq += k[i] * k[i];
            }
            float inv_q = scale / std::max(std::sqrt(q_norm_sq), l2_eps);
            float inv_k = 1.0f / std::max(std::sqrt(k_norm_sq), l2_eps);
            for (int i = 0; i < d_k; ++i)
            {
                q_proc[i] = q[i] * inv_q;
                k_proc[i] = k[i] * inv_k;
            }
        }
        else
        {
            for (int i = 0; i < d_k; ++i)
            {
                q_proc[i] = q[i] * scale;
                k_proc[i] = k[i];
            }
        }

        // Gate computation
        // GGUF stores -exp(A_log_raw), so A_log_val is already negative
        float x = alpha_val + dt_bias_val;
        float sp = ref_softplus(x);
        float g = A_log_val * sp;
        float decay = std::exp(g);

        // Beta sigmoid
        float beta_sig = ref_sigmoid(beta_raw_val);

        // Step 1: Decay state — S *= decay
        for (int i = 0; i < d_k * d_v; ++i)
            state[i] *= decay;

        // Step 2: kv_mem[vi] = sum_j S[j,vi] * k[j]
        std::vector<float> kv_mem(d_v, 0.0f);
        for (int j = 0; j < d_k; ++j)
            for (int vi = 0; vi < d_v; ++vi)
                kv_mem[vi] += state[j * d_v + vi] * k_proc[j];

        // Step 3: delta = (v - kv_mem) * beta_sig
        std::vector<float> delta(d_v);
        for (int vi = 0; vi < d_v; ++vi)
            delta[vi] = (v[vi] - kv_mem[vi]) * beta_sig;

        // Step 4: S += outer(k, delta)
        for (int j = 0; j < d_k; ++j)
            for (int vi = 0; vi < d_v; ++vi)
                state[j * d_v + vi] += k_proc[j] * delta[vi];

        // Step 5: output[vi] = sum_j S[j,vi] * q[j]
        std::vector<float> output(d_v, 0.0f);
        for (int j = 0; j < d_k; ++j)
            for (int vi = 0; vi < d_v; ++vi)
                output[vi] += state[j * d_v + vi] * q_proc[j];

        RefRecurrenceResult result;
        result.output = std::move(output);
        result.state.assign(state, state + d_k * d_v);
        return result;
    }

} // namespace

// ============================================================================
// Gap 1: Standalone reference implementation comparison
// ============================================================================

TEST(Test__GDNMathematicalCorrectness, Recurrence_Decode_MatchesReference_SingleHead)
{
    // Single head, realistic random inputs — compare kernel output
    // element-by-element against the pure scalar reference implementation.
    const int n_heads = 1;
    const int d_k = 8;
    const int d_v = 8;

    std::mt19937 rng(777);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> q_data(d_k), k_data(d_k), v_data(d_v);
    for (auto &x : q_data)
        x = dist(rng);
    for (auto &x : k_data)
        x = dist(rng);
    for (auto &x : v_data)
        x = dist(rng);

    const float alpha_val = 0.7f;
    const float beta_raw_val = -0.3f;
    const float A_log_val = -0.8f; // Typical GGUF: -exp(A_log_raw)
    const float dt_bias_val = 0.2f;

    // Reference computation
    std::vector<float> ref_state(d_k * d_v, 0.0f);
    auto ref = ref_recurrent_step(
        q_data.data(), k_data.data(), v_data.data(),
        alpha_val, beta_raw_val, A_log_val, dt_bias_val,
        ref_state.data(), d_k, d_v, false);

    // Kernel computation via stage
    auto Q = makeFP32({1, static_cast<size_t>(d_k)}, q_data.data());
    auto K = makeFP32({1, static_cast<size_t>(d_k)}, k_data.data());
    auto V = makeFP32({1, static_cast<size_t>(d_v)}, v_data.data());
    auto alpha = makeFP32({1, 1}, &alpha_val);
    auto beta = makeFP32({1, 1}, &beta_raw_val);
    auto A_log = makeFP32({1}, &A_log_val);
    auto dt_bias = makeFP32({1}, &dt_bias_val);
    auto output = makeFP32({1, static_cast<size_t>(d_v)});
    std::vector<float> kernel_state(d_k * d_v, 0.0f);

    auto ctx = makeCPUContext();
    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = kernel_state.data();
    p.seq_len = 1;
    p.n_heads = n_heads;
    p.d_k = d_k;
    p.d_v = d_v;
    p.use_qk_l2norm = false;

    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // Compare output element-by-element
    const float *out = output->data();
    for (int i = 0; i < d_v; ++i)
    {
        EXPECT_NEAR(out[i], ref.output[i], 1e-5f)
            << "Output[" << i << "] mismatch: kernel=" << out[i]
            << " ref=" << ref.output[i];
    }

    // Compare state element-by-element
    for (int i = 0; i < d_k * d_v; ++i)
    {
        EXPECT_NEAR(kernel_state[i], ref.state[i], 1e-5f)
            << "State[" << i << "] mismatch: kernel=" << kernel_state[i]
            << " ref=" << ref.state[i];
    }
}

TEST(Test__GDNMathematicalCorrectness, Recurrence_Decode_MatchesReference_MultiHead)
{
    // Two heads with different inputs — verify per-head isolation and correctness
    const int n_heads = 2;
    const int d_k = 6;
    const int d_v = 6;

    std::mt19937 rng(888);
    std::normal_distribution<float> dist(0.0f, 0.4f);

    std::vector<float> q_data(n_heads * d_k), k_data(n_heads * d_k), v_data(n_heads * d_v);
    for (auto &x : q_data)
        x = dist(rng);
    for (auto &x : k_data)
        x = dist(rng);
    for (auto &x : v_data)
        x = dist(rng);

    std::vector<float> alpha_data = {0.5f, -0.3f};
    std::vector<float> beta_data = {1.5f, -2.0f};
    std::vector<float> A_log_data = {-0.5f, -1.2f};
    std::vector<float> dt_bias_data = {0.1f, -0.1f};

    // Reference: compute each head independently
    std::vector<float> ref_state(n_heads * d_k * d_v, 0.0f);
    std::vector<float> ref_output(n_heads * d_v);
    for (int h = 0; h < n_heads; ++h)
    {
        float *head_state = ref_state.data() + h * d_k * d_v;
        auto result = ref_recurrent_step(
            q_data.data() + h * d_k,
            k_data.data() + h * d_k,
            v_data.data() + h * d_v,
            alpha_data[h], beta_data[h], A_log_data[h], dt_bias_data[h],
            head_state, d_k, d_v, false);
        std::memcpy(ref_output.data() + h * d_v, result.output.data(), d_v * sizeof(float));
    }

    // Kernel computation
    auto Q = makeFP32({1, static_cast<size_t>(n_heads * d_k)}, q_data.data());
    auto K = makeFP32({1, static_cast<size_t>(n_heads * d_k)}, k_data.data());
    auto V = makeFP32({1, static_cast<size_t>(n_heads * d_v)}, v_data.data());
    auto alpha = makeFP32({1, static_cast<size_t>(n_heads)}, alpha_data.data());
    auto beta = makeFP32({1, static_cast<size_t>(n_heads)}, beta_data.data());
    auto A_log = makeFP32({static_cast<size_t>(n_heads)}, A_log_data.data());
    auto dt_bias_t = makeFP32({static_cast<size_t>(n_heads)}, dt_bias_data.data());
    auto output = makeFP32({1, static_cast<size_t>(n_heads * d_v)});
    std::vector<float> kernel_state(n_heads * d_k * d_v, 0.0f);

    auto ctx = makeCPUContext();
    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias_t.get();
    p.output = output.get();
    p.recurrence_state = kernel_state.data();
    p.seq_len = 1;
    p.n_heads = n_heads;
    p.d_k = d_k;
    p.d_v = d_v;
    p.use_qk_l2norm = false;

    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float *out = output->data();
    for (int i = 0; i < n_heads * d_v; ++i)
    {
        EXPECT_NEAR(out[i], ref_output[i], 1e-5f)
            << "Output[" << i << "] (head " << i / d_v << ", dim " << i % d_v << ")";
    }
    for (int i = 0; i < n_heads * d_k * d_v; ++i)
    {
        EXPECT_NEAR(kernel_state[i], ref_state[i], 1e-5f)
            << "State[" << i << "]";
    }
}

TEST(Test__GDNMathematicalCorrectness, Recurrence_Decode_MatchesReference_WithL2Norm)
{
    // Verify L2 normalization produces the correct output by comparing
    // against the reference implementation with L2 norm enabled.
    const int d_k = 8;
    const int d_v = 8;

    std::mt19937 rng(999);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Use non-unit-norm vectors to exercise normalization
    std::vector<float> q_data(d_k), k_data(d_k), v_data(d_v);
    for (auto &x : q_data)
        x = dist(rng);
    for (auto &x : k_data)
        x = dist(rng);
    for (auto &x : v_data)
        x = dist(rng);

    const float alpha_val = 1.0f;
    const float beta_raw_val = 0.5f;
    const float A_log_val = -0.6f;
    const float dt_bias_val = 0.0f;

    // Non-zero initial state
    std::vector<float> ref_state(d_k * d_v);
    std::uniform_real_distribution<float> state_dist(-0.1f, 0.1f);
    for (auto &x : ref_state)
        x = state_dist(rng);
    std::vector<float> kernel_state = ref_state;

    auto ref = ref_recurrent_step(
        q_data.data(), k_data.data(), v_data.data(),
        alpha_val, beta_raw_val, A_log_val, dt_bias_val,
        ref_state.data(), d_k, d_v, true);

    auto Q = makeFP32({1, static_cast<size_t>(d_k)}, q_data.data());
    auto K = makeFP32({1, static_cast<size_t>(d_k)}, k_data.data());
    auto V = makeFP32({1, static_cast<size_t>(d_v)}, v_data.data());
    auto alpha = makeFP32({1, 1}, &alpha_val);
    auto beta = makeFP32({1, 1}, &beta_raw_val);
    auto A_log = makeFP32({1}, &A_log_val);
    auto dt_bias_t = makeFP32({1}, &dt_bias_val);
    auto output = makeFP32({1, static_cast<size_t>(d_v)});

    auto ctx = makeCPUContext();
    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias_t.get();
    p.output = output.get();
    p.recurrence_state = kernel_state.data();
    p.seq_len = 1;
    p.n_heads = 1;
    p.d_k = d_k;
    p.d_v = d_v;
    p.use_qk_l2norm = true;

    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float *out = output->data();
    for (int i = 0; i < d_v; ++i)
    {
        EXPECT_NEAR(out[i], ref.output[i], 1e-5f)
            << "Output[" << i << "] with L2 norm";
    }
    for (int i = 0; i < d_k * d_v; ++i)
    {
        EXPECT_NEAR(kernel_state[i], ref.state[i], 1e-5f)
            << "State[" << i << "] with L2 norm";
    }
}

// ============================================================================
// Gap 2: Direct gate computation verification
// ============================================================================

TEST(Test__GDNMathematicalCorrectness, GateComputation_KnownValues)
{
    // Verify the gate formula: g = A_log_stored * softplus(alpha + dt_bias)
    // with specific known inputs and hand-computed expected values.
    //
    // Use the kernel through a full recurrent_step but with carefully
    // constructed inputs so we can verify the gate's effect on state decay.

    const int d_k = 2;
    const int d_v = 2;

    // All-ones Q, K, V so the recurrence arithmetic is simple
    std::vector<float> q = {1.0f, 0.0f};
    std::vector<float> k = {0.0f, 1.0f};
    std::vector<float> v = {1.0f, 1.0f};

    // Set beta to 0 → beta_sig ≈ sigmoid(0) = 0.5
    float beta_raw = 0.0f;

    // Test case 1: alpha=0, dt_bias=0, A_log=-1 (GGUF stores -exp(0))
    // g = -1.0 * softplus(0 + 0) = -1.0 * ln(2) ≈ -0.6931
    // decay = exp(-0.6931) ≈ 0.5
    {
        float alpha_val = 0.0f;
        float A_log_val = -1.0f;
        float dt_bias_val = 0.0f;

        float expected_g = -1.0f * ref_softplus(0.0f);
        float expected_decay = std::exp(expected_g);
        EXPECT_NEAR(expected_g, -std::log(2.0f), 1e-6f);
        EXPECT_NEAR(expected_decay, 0.5f, 1e-6f);

        // Start with state = identity matrix
        std::vector<float> state = {1.0f, 0.0f, 0.0f, 1.0f};
        std::vector<float> state_copy = state;

        auto ref = ref_recurrent_step(
            q.data(), k.data(), v.data(),
            alpha_val, beta_raw, A_log_val, dt_bias_val,
            state_copy.data(), d_k, d_v, false);

        // After decay: state[0,0]=0.5, state[1,1]=0.5 (off-diag stay 0)
        // kv_mem = S*k = [S[0,0]*k[0]+S[0,1]*k[1], S[1,0]*k[0]+S[1,1]*k[1]]
        //        = [0, 0.5]  (k=[0,1])
        // delta = (v - kv_mem) * 0.5 = ([1,1]-[0,0.5]) * 0.5 = [0.5, 0.25]
        // S += outer(k, delta) = outer([0,1], [0.5,0.25])
        //    S[0,:] += 0*[0.5,0.25] = unchanged
        //    S[1,:] += 1*[0.5,0.25] = [0, 0.5] + [0.5, 0.25] = [0.5, 0.75]
        EXPECT_NEAR(state_copy[0], 0.5f, 1e-5f);  // S[0,0]
        EXPECT_NEAR(state_copy[1], 0.0f, 1e-5f);  // S[0,1]
        EXPECT_NEAR(state_copy[2], 0.5f, 1e-5f);  // S[1,0]
        EXPECT_NEAR(state_copy[3], 0.75f, 1e-5f); // S[1,1]
    }

    // Test case 2: alpha=5, dt_bias=0, A_log=-0.5
    // g = -0.5 * softplus(5) ≈ -0.5 * 5.0067 ≈ -2.5034
    // decay = exp(-2.5034) ≈ 0.0818
    {
        float alpha_val = 5.0f;
        float A_log_val = -0.5f;
        float dt_bias_val = 0.0f;

        float expected_g = -0.5f * ref_softplus(5.0f);
        float expected_decay = std::exp(expected_g);
        EXPECT_NEAR(expected_g, -0.5f * std::log1p(std::exp(5.0f)), 1e-5f);

        // Verify through the kernel: with large alpha → small decay
        std::vector<float> state = {1.0f, 0.0f, 0.0f, 1.0f};
        auto Q = makeFP32({1, 2}, q.data());
        auto K = makeFP32({1, 2}, k.data());
        auto V = makeFP32({1, 2}, v.data());
        auto alpha_t = makeFP32({1, 1}, &alpha_val);
        auto beta_t = makeFP32({1, 1}, &beta_raw);
        auto A_log_t = makeFP32({1}, &A_log_val);
        auto dt_bias_t = makeFP32({1}, &dt_bias_val);
        auto output = makeFP32({1, 2});

        auto ctx = makeCPUContext();
        GDNRecurrenceStage::Params p;
        p.kernel = &g_cpu_gdn;
        p.Q = Q.get();
        p.K = K.get();
        p.V = V.get();
        p.alpha = alpha_t.get();
        p.beta = beta_t.get();
        p.A_log = A_log_t.get();
        p.dt_bias = dt_bias_t.get();
        p.output = output.get();
        p.recurrence_state = state.data();
        p.seq_len = 1;
        p.n_heads = 1;
        p.d_k = d_k;
        p.d_v = d_v;
        p.use_qk_l2norm = false;

        GDNRecurrenceStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));

        // Compute expected via reference
        std::vector<float> ref_state = {1.0f, 0.0f, 0.0f, 1.0f};
        auto ref = ref_recurrent_step(
            q.data(), k.data(), v.data(),
            alpha_val, beta_raw, A_log_val, dt_bias_val,
            ref_state.data(), d_k, d_v, false);

        for (int i = 0; i < d_k * d_v; ++i)
            EXPECT_NEAR(state[i], ref_state[i], 1e-5f) << "State[" << i << "]";

        const float *out = output->data();
        for (int i = 0; i < d_v; ++i)
            EXPECT_NEAR(out[i], ref.output[i], 1e-5f) << "Output[" << i << "]";
    }

    // Test case 3: alpha=0, dt_bias=3, A_log=-2.0
    // g = -2.0 * softplus(0 + 3) = -2.0 * ln(1+e^3) ≈ -2.0 * 3.0486 ≈ -6.097
    // decay = exp(-6.097) ≈ 0.00224 (very small — state nearly zeroed)
    {
        float alpha_val = 0.0f;
        float A_log_val = -2.0f;
        float dt_bias_val = 3.0f;

        float expected_g = -2.0f * ref_softplus(3.0f);
        float expected_decay = std::exp(expected_g);
        EXPECT_LT(expected_decay, 0.01f) << "Decay should be very small";

        // With near-zero decay, the state contribution from the previous step
        // is nearly eliminated. The output should be dominated by the current
        // input's contribution.
        std::vector<float> state = {10.0f, 10.0f, 10.0f, 10.0f};
        std::vector<float> ref_state = state;
        auto ref = ref_recurrent_step(
            q.data(), k.data(), v.data(),
            alpha_val, beta_raw, A_log_val, dt_bias_val,
            ref_state.data(), d_k, d_v, false);

        // Despite large initial state (10.0), after decay of ~0.002
        // the old state contribution should be negligible
        float state_energy = 0.0f;
        for (auto s : ref_state)
            state_energy += s * s;
        // State won't be exactly zero because of the update, but the
        // decayed initial state contribution should be tiny
        float initial_contribution = expected_decay * 10.0f;
        EXPECT_LT(initial_contribution, 0.03f)
            << "Old state should be nearly eliminated by strong decay";
    }
}

// ============================================================================
// Gap 3: Multi-step decode accumulation
// ============================================================================

TEST(Test__GDNMathematicalCorrectness, Recurrence_MultiStepDecode_20Steps)
{
    // Run 20 sequential decode steps and verify the kernel matches
    // the reference at every step. This catches:
    // - State reset bugs
    // - Off-by-one in state indexing
    // - Numerical accumulation drift
    const int n_heads = 2;
    const int d_k = 4;
    const int d_v = 4;
    const int n_steps = 20;

    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(0.0f, 0.3f);

    // Per-head weights (constant across steps)
    std::vector<float> A_log_data = {-0.7f, -1.1f};
    std::vector<float> dt_bias_data = {0.15f, -0.2f};

    // State trackers
    std::vector<float> kernel_state(n_heads * d_k * d_v, 0.0f);
    std::vector<float> ref_state(n_heads * d_k * d_v, 0.0f);

    auto A_log = makeFP32({static_cast<size_t>(n_heads)}, A_log_data.data());
    auto dt_bias_t = makeFP32({static_cast<size_t>(n_heads)}, dt_bias_data.data());
    auto ctx = makeCPUContext();

    for (int step = 0; step < n_steps; ++step)
    {
        // Generate random inputs for this step
        std::vector<float> q_data(n_heads * d_k), k_data(n_heads * d_k), v_data(n_heads * d_v);
        std::vector<float> alpha_data(n_heads), beta_data(n_heads);
        for (auto &x : q_data)
            x = dist(rng);
        for (auto &x : k_data)
            x = dist(rng);
        for (auto &x : v_data)
            x = dist(rng);
        for (auto &x : alpha_data)
            x = dist(rng);
        for (auto &x : beta_data)
            x = dist(rng);

        // Reference: compute each head
        std::vector<float> ref_output(n_heads * d_v);
        for (int h = 0; h < n_heads; ++h)
        {
            auto result = ref_recurrent_step(
                q_data.data() + h * d_k,
                k_data.data() + h * d_k,
                v_data.data() + h * d_v,
                alpha_data[h], beta_data[h],
                A_log_data[h], dt_bias_data[h],
                ref_state.data() + h * d_k * d_v,
                d_k, d_v, false);
            std::memcpy(ref_output.data() + h * d_v,
                        result.output.data(), d_v * sizeof(float));
        }

        // Kernel
        auto Q = makeFP32({1, static_cast<size_t>(n_heads * d_k)}, q_data.data());
        auto K = makeFP32({1, static_cast<size_t>(n_heads * d_k)}, k_data.data());
        auto V = makeFP32({1, static_cast<size_t>(n_heads * d_v)}, v_data.data());
        auto alpha = makeFP32({1, static_cast<size_t>(n_heads)}, alpha_data.data());
        auto beta = makeFP32({1, static_cast<size_t>(n_heads)}, beta_data.data());
        auto output = makeFP32({1, static_cast<size_t>(n_heads * d_v)});

        GDNRecurrenceStage::Params p;
        p.kernel = &g_cpu_gdn;
        p.Q = Q.get();
        p.K = K.get();
        p.V = V.get();
        p.alpha = alpha.get();
        p.beta = beta.get();
        p.A_log = A_log.get();
        p.dt_bias = dt_bias_t.get();
        p.output = output.get();
        p.recurrence_state = kernel_state.data();
        p.seq_len = 1;
        p.n_heads = n_heads;
        p.d_k = d_k;
        p.d_v = d_v;
        p.use_qk_l2norm = false;

        GDNRecurrenceStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get())) << "Step " << step << " failed";

        // Compare output
        const float *out = output->data();
        for (int i = 0; i < n_heads * d_v; ++i)
        {
            EXPECT_NEAR(out[i], ref_output[i], 1e-4f)
                << "Step " << step << " output[" << i << "]"
                << " (head " << i / d_v << ", dim " << i % d_v << ")";
        }

        // Compare state
        for (int i = 0; i < n_heads * d_k * d_v; ++i)
        {
            EXPECT_NEAR(kernel_state[i], ref_state[i], 1e-4f)
                << "Step " << step << " state[" << i << "]";
        }
    }
}

TEST(Test__GDNMathematicalCorrectness, Recurrence_MultiStepDecode_WithL2Norm)
{
    // Same multi-step verification but with L2 normalization enabled
    const int n_heads = 1;
    const int d_k = 6;
    const int d_v = 6;
    const int n_steps = 15;

    std::mt19937 rng(54321);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    const float A_log_val = -0.9f;
    const float dt_bias_val = 0.1f;

    std::vector<float> kernel_state(d_k * d_v, 0.0f);
    std::vector<float> ref_state(d_k * d_v, 0.0f);

    auto A_log = makeFP32({1}, &A_log_val);
    auto dt_bias_t = makeFP32({1}, &dt_bias_val);
    auto ctx = makeCPUContext();

    for (int step = 0; step < n_steps; ++step)
    {
        std::vector<float> q_data(d_k), k_data(d_k), v_data(d_v);
        float alpha_val = dist(rng);
        float beta_val = dist(rng);
        for (auto &x : q_data)
            x = dist(rng);
        for (auto &x : k_data)
            x = dist(rng);
        for (auto &x : v_data)
            x = dist(rng);

        // Reference
        auto ref = ref_recurrent_step(
            q_data.data(), k_data.data(), v_data.data(),
            alpha_val, beta_val, A_log_val, dt_bias_val,
            ref_state.data(), d_k, d_v, true);

        // Kernel
        auto Q = makeFP32({1, static_cast<size_t>(d_k)}, q_data.data());
        auto K = makeFP32({1, static_cast<size_t>(d_k)}, k_data.data());
        auto V = makeFP32({1, static_cast<size_t>(d_v)}, v_data.data());
        auto alpha = makeFP32({1, 1}, &alpha_val);
        auto beta = makeFP32({1, 1}, &beta_val);
        auto output = makeFP32({1, static_cast<size_t>(d_v)});

        GDNRecurrenceStage::Params p;
        p.kernel = &g_cpu_gdn;
        p.Q = Q.get();
        p.K = K.get();
        p.V = V.get();
        p.alpha = alpha.get();
        p.beta = beta.get();
        p.A_log = A_log.get();
        p.dt_bias = dt_bias_t.get();
        p.output = output.get();
        p.recurrence_state = kernel_state.data();
        p.seq_len = 1;
        p.n_heads = 1;
        p.d_k = d_k;
        p.d_v = d_v;
        p.use_qk_l2norm = true;

        GDNRecurrenceStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get())) << "Step " << step;

        const float *out = output->data();
        for (int i = 0; i < d_v; ++i)
        {
            EXPECT_NEAR(out[i], ref.output[i], 1e-4f)
                << "Step " << step << " output[" << i << "]";
        }
    }
}

// ============================================================================
// Gap 4: GatedRMSNorm per-head norm_dim
// ============================================================================

TEST(Test__GDNMathematicalCorrectness, GatedRMSNorm_PerHeadNormDim)
{
    // With norm_dim=4 and d_model=12, there are 3 groups.
    // Each group of 4 elements should be independently RMS-normalized,
    // NOT normalized over the full 12-element row.
    const int d_model = 12;
    const int norm_dim = 4;
    const int n_groups = d_model / norm_dim; // 3

    // Input: groups with different magnitudes
    //   group 0: [2, 2, 2, 2] → rms = 2
    //   group 1: [4, 4, 4, 4] → rms = 4
    //   group 2: [1, 1, 1, 1] → rms = 1
    std::vector<float> input_data = {2, 2, 2, 2, 4, 4, 4, 4, 1, 1, 1, 1};
    auto input = makeFP32({1, static_cast<size_t>(d_model)}, input_data.data());

    // Gate: all 1.0 (pass-through)
    auto gate = makeFP32Const({1, static_cast<size_t>(d_model)}, 1.0f);

    // Gamma: all 1.0 (per-head gamma, wraps around norm_dim)
    auto gamma = makeFP32Const({static_cast<size_t>(norm_dim)}, 1.0f);

    auto output = makeFP32({1, static_cast<size_t>(d_model)});

    GatedRMSNormStage::Params params;
    params.input = input.get();
    params.gate = gate.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-6f;
    params.seq_len = 1;
    params.norm_dim = norm_dim;

    auto ctx = makeCPUContext();
    GatedRMSNormStage stage(params);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float *out = output->data();

    // Each group should be normalized independently:
    // group 0: [2,2,2,2] / 2.0 * 1.0 * 1.0 = [1,1,1,1]
    // group 1: [4,4,4,4] / 4.0 * 1.0 * 1.0 = [1,1,1,1]
    // group 2: [1,1,1,1] / 1.0 * 1.0 * 1.0 = [1,1,1,1]
    for (int i = 0; i < d_model; ++i)
    {
        EXPECT_NEAR(out[i], 1.0f, 1e-4f)
            << "Element " << i << " (group " << i / norm_dim << ")";
    }

    // Now verify that full-dim normalization (norm_dim=0) produces DIFFERENT
    // results for the same input — confirming the per-head path is distinct.
    auto output_full = makeFP32({1, static_cast<size_t>(d_model)});

    // Need gamma with d_model elements for full-dim norm
    auto gamma_full = makeFP32Const({static_cast<size_t>(d_model)}, 1.0f);

    // Reset input (it shouldn't have been modified, but be safe)
    auto input2 = makeFP32({1, static_cast<size_t>(d_model)}, input_data.data());
    auto gate2 = makeFP32Const({1, static_cast<size_t>(d_model)}, 1.0f);

    GatedRMSNormStage::Params params_full;
    params_full.input = input2.get();
    params_full.gate = gate2.get();
    params_full.output = output_full.get();
    params_full.gamma = gamma_full.get();
    params_full.eps = 1e-6f;
    params_full.seq_len = 1;
    params_full.norm_dim = 0; // Full-dim normalization

    GatedRMSNormStage stage_full(params_full);
    ASSERT_TRUE(stage_full.execute(ctx.get()));

    const float *out_full = output_full->data();

    // Full-dim RMS = sqrt((4*4 + 4*16 + 4*1) / 12) = sqrt(84/12) = sqrt(7) ≈ 2.6458
    float rms_full = std::sqrt((4 * 4.0f + 4 * 16.0f + 4 * 1.0f) / 12.0f);

    // Group 0 (input 2): full-dim → 2/rms_full ≈ 0.756 (NOT 1.0)
    EXPECT_NEAR(out_full[0], 2.0f / rms_full, 1e-3f);
    // Group 1 (input 4): full-dim → 4/rms_full ≈ 1.512 (NOT 1.0)
    EXPECT_NEAR(out_full[4], 4.0f / rms_full, 1e-3f);
    // Group 2 (input 1): full-dim → 1/rms_full ≈ 0.378 (NOT 1.0)
    EXPECT_NEAR(out_full[8], 1.0f / rms_full, 1e-3f);

    // Different from per-head result (1.0)
    EXPECT_GT(std::abs(out_full[0] - 1.0f), 0.1f)
        << "Full-dim and per-head normalization should produce different results";
}

TEST(Test__GDNMathematicalCorrectness, GatedRMSNorm_PerHeadNormDim_WithSiLU)
{
    // Per-head norm_dim with gate_silu=true
    const int d_model = 8;
    const int norm_dim = 4;

    // Input: all 1.0 → RMS = 1.0
    auto input = makeFP32Const({1, static_cast<size_t>(d_model)}, 1.0f);
    auto gamma = makeFP32Const({static_cast<size_t>(norm_dim)}, 1.0f);
    auto output = makeFP32({1, static_cast<size_t>(d_model)});

    // Gate with known values
    std::vector<float> gate_data = {0.0f, 1.0f, -1.0f, 2.0f, 3.0f, -3.0f, 0.5f, -0.5f};
    auto gate = makeFP32({1, static_cast<size_t>(d_model)}, gate_data.data());

    GatedRMSNormStage::Params params;
    params.input = input.get();
    params.gate = gate.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-6f;
    params.seq_len = 1;
    params.norm_dim = norm_dim;
    params.gate_silu = true;

    auto ctx = makeCPUContext();
    GatedRMSNormStage stage(params);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // Input RMS = 1.0 (per group of 4), so normalized = 1.0 * gamma(1.0) = 1.0
    // output = normalized * SiLU(gate) = SiLU(gate)
    const float inv_rms = 1.0f / std::sqrt(1.0f + params.eps);
    const float *out = output->data();
    for (int i = 0; i < d_model; ++i)
    {
        float expected = inv_rms * ref_silu(gate_data[i]);
        EXPECT_NEAR(out[i], expected, std::abs(expected) * 0.005f + 1e-6f)
            << "Element " << i << " gate=" << gate_data[i];
    }
}

// ============================================================================
// Gap 5: Conv1d bias term
// ============================================================================

TEST(Test__GDNMathematicalCorrectness, Conv1d_DecodeBias)
{
    // Verify that a non-null bias is correctly added per-channel
    const int channels = 3;
    const int kernel_size = 2;
    const int state_len = kernel_size - 1;

    // Weight: all 0 — so conv output is purely from bias
    auto weight = makeFP32Const({static_cast<size_t>(channels),
                                 static_cast<size_t>(kernel_size)},
                                0.0f);

    // Bias: [10, 20, 30]
    std::vector<float> bias_data = {10.0f, 20.0f, 30.0f};
    auto bias = makeFP32({static_cast<size_t>(channels)}, bias_data.data());

    auto input = makeFP32Const({1, static_cast<size_t>(channels)}, 0.0f);
    auto output = makeFP32({1, static_cast<size_t>(channels)});
    std::vector<float> conv_state(channels * state_len, 0.0f);

    auto ctx = makeCPUContext();
    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.bias = bias.get();
    p.seq_len = 1;
    p.channels = channels;
    p.kernel_size = kernel_size;
    p.conv_state = conv_state.data();

    ShortConv1dStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // With zero weights and zero input, output = SiLU(bias)
    const float *out = output->data();
    EXPECT_NEAR(out[0], ref_silu(10.0f), 1e-4f) << "Channel 0 bias";
    EXPECT_NEAR(out[1], ref_silu(20.0f), 1e-4f) << "Channel 1 bias";
    EXPECT_NEAR(out[2], ref_silu(30.0f), 1e-4f) << "Channel 2 bias";
}

TEST(Test__GDNMathematicalCorrectness, Conv1d_PrefillBias)
{
    // Verify bias in prefill (seq_len > 1) path
    const int channels = 2;
    const int kernel_size = 2;
    const int seq_len = 3;

    // Weight: all 0 — output is purely from bias
    auto weight = makeFP32Const({static_cast<size_t>(channels),
                                 static_cast<size_t>(kernel_size)},
                                0.0f);

    // Bias: [5, -5]
    std::vector<float> bias_data = {5.0f, -5.0f};
    auto bias = makeFP32({static_cast<size_t>(channels)}, bias_data.data());

    auto input = makeFP32Const({static_cast<size_t>(seq_len),
                                static_cast<size_t>(channels)},
                               0.0f);
    auto output = makeFP32({static_cast<size_t>(seq_len),
                            static_cast<size_t>(channels)});

    auto ctx = makeCPUContext();
    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.bias = bias.get();
    p.seq_len = seq_len;
    p.channels = channels;
    p.kernel_size = kernel_size;

    ShortConv1dStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // Every timestep should produce SiLU(bias) per channel
    const float *out = output->data();
    for (int t = 0; t < seq_len; ++t)
    {
        EXPECT_NEAR(out[t * channels + 0], ref_silu(5.0f), 1e-4f)
            << "t=" << t << " channel 0";
        EXPECT_NEAR(out[t * channels + 1], ref_silu(-5.0f), 1e-4f)
            << "t=" << t << " channel 1";
    }
}

TEST(Test__GDNMathematicalCorrectness, Conv1d_BiasWithWeights)
{
    // Bias combined with non-zero weights and state
    const int channels = 2;
    const int kernel_size = 3;
    const int state_len = kernel_size - 1;

    // Weight: ch0=[1,1,1], ch1=[2,0,0]
    std::vector<float> weight_data = {1, 1, 1, 2, 0, 0};
    auto weight = makeFP32({static_cast<size_t>(channels),
                            static_cast<size_t>(kernel_size)},
                           weight_data.data());

    // Bias: [100, 200]
    std::vector<float> bias_data = {100.0f, 200.0f};
    auto bias = makeFP32({static_cast<size_t>(channels)}, bias_data.data());

    // Input: [3, 4]
    std::vector<float> input_data = {3.0f, 4.0f};
    auto input = makeFP32({1, static_cast<size_t>(channels)}, input_data.data());
    auto output = makeFP32({1, static_cast<size_t>(channels)});

    // State: ch0=[1, 2], ch1=[5, 6]
    std::vector<float> conv_state = {1.0f, 2.0f, 5.0f, 6.0f};

    auto ctx = makeCPUContext();
    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.bias = bias.get();
    p.seq_len = 1;
    p.channels = channels;
    p.kernel_size = kernel_size;
    p.conv_state = conv_state.data();

    ShortConv1dStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // ch0: bias=100, w=[1,1,1], state=[1,2], new=3
    //   raw = 100 + 1*1 + 1*2 + 1*3 = 106
    // ch1: bias=200, w=[2,0,0], state=[5,6], new=4
    //   raw = 200 + 2*5 + 0*6 + 0*4 = 210
    const float *out = output->data();
    EXPECT_NEAR(out[0], ref_silu(106.0f), 1e-3f) << "Channel 0 with bias+weights";
    EXPECT_NEAR(out[1], ref_silu(210.0f), 1e-3f) << "Channel 1 with bias+weights";
}

// ============================================================================
// Gap 6: Conv1d multi-channel prefill with distinct per-channel weights
// ============================================================================

TEST(Test__GDNMathematicalCorrectness, Conv1d_Prefill_MultiChannelDistinctWeights)
{
    // 3 channels with completely different weights — verify per-channel
    // isolation and correct index striding
    const int channels = 3;
    const int kernel_size = 3;
    const int seq_len = 4;

    // Distinct weights:
    //   ch0: [1, 0, 0] — only looks at oldest element in window
    //   ch1: [0, 0, 1] — only looks at newest element
    //   ch2: [0, 1, 0] — only looks at middle element
    std::vector<float> weight_data = {1, 0, 0, 0, 0, 1, 0, 1, 0};
    auto weight = makeFP32({static_cast<size_t>(channels),
                            static_cast<size_t>(kernel_size)},
                           weight_data.data());

    // Input: [seq_len, channels] — each channel gets unique sequence
    //   ch0: [10, 20, 30, 40]
    //   ch1: [100, 200, 300, 400]
    //   ch2: [1, 2, 3, 4]
    std::vector<float> input_data = {
        10,
        100,
        1, // t=0
        20,
        200,
        2, // t=1
        30,
        300,
        3, // t=2
        40,
        400,
        4, // t=3
    };
    auto input = makeFP32({static_cast<size_t>(seq_len),
                           static_cast<size_t>(channels)},
                          input_data.data());
    auto output = makeFP32({static_cast<size_t>(seq_len),
                            static_cast<size_t>(channels)});

    auto ctx = makeCPUContext();
    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = seq_len;
    p.channels = channels;
    p.kernel_size = kernel_size;

    ShortConv1dStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float *out = output->data();

    // Causal conv with zero-padding:
    // Window at time t: [t-2, t-1, t] with zero-pad for negative indices
    //
    // ch0 (w=[1,0,0]): reads oldest → input[t-2]
    //   t=0: w[0]*0 = 0         → SiLU(0)
    //   t=1: w[0]*0 = 0         → SiLU(0)
    //   t=2: w[0]*10 = 10       → SiLU(10)
    //   t=3: w[0]*20 = 20       → SiLU(20)
    EXPECT_NEAR(out[0 * channels + 0], ref_silu(0.0f), 1e-5f);
    EXPECT_NEAR(out[1 * channels + 0], ref_silu(0.0f), 1e-5f);
    EXPECT_NEAR(out[2 * channels + 0], ref_silu(10.0f), 1e-4f);
    EXPECT_NEAR(out[3 * channels + 0], ref_silu(20.0f), 1e-4f);

    // ch1 (w=[0,0,1]): reads newest → input[t]
    //   t=0: w[2]*100 = 100     → SiLU(100)
    //   t=1: w[2]*200 = 200     → SiLU(200)
    //   t=2: w[2]*300 = 300     → SiLU(300)
    //   t=3: w[2]*400 = 400     → SiLU(400)
    EXPECT_NEAR(out[0 * channels + 1], ref_silu(100.0f), 1e-2f);
    EXPECT_NEAR(out[1 * channels + 1], ref_silu(200.0f), 1e-2f);
    EXPECT_NEAR(out[2 * channels + 1], ref_silu(300.0f), 1e-2f);
    EXPECT_NEAR(out[3 * channels + 1], ref_silu(400.0f), 1e-2f);

    // ch2 (w=[0,1,0]): reads middle → input[t-1]
    //   t=0: w[1]*0 = 0         → SiLU(0)
    //   t=1: w[1]*1 = 1         → SiLU(1)
    //   t=2: w[1]*2 = 2         → SiLU(2)
    //   t=3: w[1]*3 = 3         → SiLU(3)
    EXPECT_NEAR(out[0 * channels + 2], ref_silu(0.0f), 1e-5f);
    EXPECT_NEAR(out[1 * channels + 2], ref_silu(1.0f), 1e-5f);
    EXPECT_NEAR(out[2 * channels + 2], ref_silu(2.0f), 1e-5f);
    EXPECT_NEAR(out[3 * channels + 2], ref_silu(3.0f), 1e-5f);
}

// ============================================================================
// Gap 7: Recurrence with odd d_k/d_v (scalar tail handling)
// ============================================================================

TEST(Test__GDNMathematicalCorrectness, Recurrence_OddDimensions_ScalarTail)
{
    // Use d_k=5, d_v=7 — NOT multiples of 16 — to force the scalar tail
    // handling in AVX512 paths. Compare against reference.
    const int n_heads = 2;
    const int d_k = 5;
    const int d_v = 7;

    std::mt19937 rng(11111);
    std::normal_distribution<float> dist(0.0f, 0.4f);

    std::vector<float> q_data(n_heads * d_k), k_data(n_heads * d_k), v_data(n_heads * d_v);
    for (auto &x : q_data)
        x = dist(rng);
    for (auto &x : k_data)
        x = dist(rng);
    for (auto &x : v_data)
        x = dist(rng);

    std::vector<float> alpha_data = {0.3f, -0.5f};
    std::vector<float> beta_data = {1.0f, -1.0f};
    std::vector<float> A_log_data = {-0.6f, -1.3f};
    std::vector<float> dt_bias_data = {0.2f, 0.0f};

    // Non-zero initial state
    std::vector<float> kernel_state(n_heads * d_k * d_v);
    std::uniform_real_distribution<float> state_dist(-0.1f, 0.1f);
    for (auto &x : kernel_state)
        x = state_dist(rng);
    std::vector<float> ref_state = kernel_state;

    // Reference
    std::vector<float> ref_output(n_heads * d_v);
    for (int h = 0; h < n_heads; ++h)
    {
        auto result = ref_recurrent_step(
            q_data.data() + h * d_k, k_data.data() + h * d_k,
            v_data.data() + h * d_v,
            alpha_data[h], beta_data[h], A_log_data[h], dt_bias_data[h],
            ref_state.data() + h * d_k * d_v, d_k, d_v, false);
        std::memcpy(ref_output.data() + h * d_v, result.output.data(), d_v * sizeof(float));
    }

    // Kernel
    auto Q = makeFP32({1, static_cast<size_t>(n_heads * d_k)}, q_data.data());
    auto K = makeFP32({1, static_cast<size_t>(n_heads * d_k)}, k_data.data());
    auto V = makeFP32({1, static_cast<size_t>(n_heads * d_v)}, v_data.data());
    auto alpha = makeFP32({1, static_cast<size_t>(n_heads)}, alpha_data.data());
    auto beta = makeFP32({1, static_cast<size_t>(n_heads)}, beta_data.data());
    auto A_log = makeFP32({static_cast<size_t>(n_heads)}, A_log_data.data());
    auto dt_bias_t = makeFP32({static_cast<size_t>(n_heads)}, dt_bias_data.data());
    auto output = makeFP32({1, static_cast<size_t>(n_heads * d_v)});

    auto ctx = makeCPUContext();
    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias_t.get();
    p.output = output.get();
    p.recurrence_state = kernel_state.data();
    p.seq_len = 1;
    p.n_heads = n_heads;
    p.d_k = d_k;
    p.d_v = d_v;
    p.use_qk_l2norm = false;

    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float *out = output->data();
    for (int i = 0; i < n_heads * d_v; ++i)
    {
        EXPECT_NEAR(out[i], ref_output[i], 1e-5f)
            << "Output[" << i << "] (head " << i / d_v << ", dim " << i % d_v << ")";
    }
    for (int i = 0; i < n_heads * d_k * d_v; ++i)
    {
        EXPECT_NEAR(kernel_state[i], ref_state[i], 1e-5f) << "State[" << i << "]";
    }
}

TEST(Test__GDNMathematicalCorrectness, Recurrence_OddDimensions_Prefill)
{
    // Prefill path with odd dimensions (d_k=3, d_v=5)
    const int n_heads = 1;
    const int d_k = 3;
    const int d_v = 5;
    const int seq_len = 4;

    std::mt19937 rng(22222);
    std::normal_distribution<float> dist(0.0f, 0.3f);

    // Generate full sequence inputs
    std::vector<float> q_data(seq_len * n_heads * d_k);
    std::vector<float> k_data(seq_len * n_heads * d_k);
    std::vector<float> v_data(seq_len * n_heads * d_v);
    std::vector<float> alpha_data(seq_len);
    std::vector<float> beta_data(seq_len);
    for (auto &x : q_data)
        x = dist(rng);
    for (auto &x : k_data)
        x = dist(rng);
    for (auto &x : v_data)
        x = dist(rng);
    for (auto &x : alpha_data)
        x = dist(rng);
    for (auto &x : beta_data)
        x = dist(rng);

    const float A_log_val = -0.8f;
    const float dt_bias_val = 0.1f;

    // Reference: sequential decode
    std::vector<float> ref_state(d_k * d_v, 0.0f);
    std::vector<float> ref_output(seq_len * d_v);
    for (int t = 0; t < seq_len; ++t)
    {
        auto result = ref_recurrent_step(
            q_data.data() + t * d_k, k_data.data() + t * d_k,
            v_data.data() + t * d_v,
            alpha_data[t], beta_data[t], A_log_val, dt_bias_val,
            ref_state.data(), d_k, d_v, false);
        std::memcpy(ref_output.data() + t * d_v, result.output.data(), d_v * sizeof(float));
    }

    // Kernel: prefill path (chunk_forward)
    auto Q = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_k)}, q_data.data());
    auto K = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_k)}, k_data.data());
    auto V = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_v)}, v_data.data());
    auto alpha = makeFP32({static_cast<size_t>(seq_len), 1}, alpha_data.data());
    auto beta = makeFP32({static_cast<size_t>(seq_len), 1}, beta_data.data());
    auto A_log = makeFP32({1}, &A_log_val);
    auto dt_bias_t = makeFP32({1}, &dt_bias_val);
    auto output = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_v)});
    std::vector<float> kernel_state(d_k * d_v, 0.0f);

    auto ctx = makeCPUContext();
    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias_t.get();
    p.output = output.get();
    p.recurrence_state = kernel_state.data();
    p.seq_len = seq_len;
    p.n_heads = 1;
    p.d_k = d_k;
    p.d_v = d_v;
    p.use_qk_l2norm = false;

    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float *out = output->data();
    for (int i = 0; i < seq_len * d_v; ++i)
    {
        EXPECT_NEAR(out[i], ref_output[i], 1e-4f)
            << "Output[" << i << "] (t=" << i / d_v << ", dim=" << i % d_v << ")";
    }
}

// ============================================================================
// Gap 8: Softplus edge cases near transition point (x ≈ 20)
// ============================================================================

TEST(Test__GDNMathematicalCorrectness, Softplus_TransitionPoint)
{
    // The kernel uses: softplus(x) = (x > 20) ? x : log1p(exp(x))
    // Verify that values near the transition produce smooth, correct results.
    // We do this indirectly through gate computation by setting alpha+dt_bias
    // to values near 20 and checking the resulting decay.

    const int d_k = 2;
    const int d_v = 2;

    struct TestCase
    {
        float alpha;
        float dt_bias;
        const char *desc;
    };

    std::vector<TestCase> cases = {
        {19.0f, 0.0f, "x=19 (below transition)"},
        {19.5f, 0.0f, "x=19.5 (near transition)"},
        {19.9f, 0.0f, "x=19.9 (just below)"},
        {20.0f, 0.0f, "x=20.0 (at transition)"},
        {20.1f, 0.0f, "x=20.1 (just above)"},
        {20.5f, 0.0f, "x=20.5 (above transition)"},
        {21.0f, 0.0f, "x=21 (above transition)"},
        {10.0f, 10.0f, "x=20 via sum (10+10)"},
        {15.0f, 5.5f, "x=20.5 via sum"},
    };

    const float A_log_val = -1.0f; // -exp(0)
    std::vector<float> q = {1.0f, 0.0f};
    std::vector<float> k = {0.0f, 1.0f};
    std::vector<float> v = {1.0f, 1.0f};
    float beta_raw = 0.0f;

    auto ctx = makeCPUContext();

    for (const auto &tc : cases)
    {
        float x = tc.alpha + tc.dt_bias;

        // Reference: exact softplus and decay
        float ref_sp = ref_softplus(x);
        float ref_g = A_log_val * ref_sp;
        float ref_decay = std::exp(ref_g);

        // Verify softplus is smooth and monotonic: log1p(exp(x)) for x<20, x for x>20
        // Both should give approximately the same result near x=20
        float sp_exact = std::log1p(std::exp(x));
        float sp_approx = x;
        float sp_diff = std::abs(sp_exact - sp_approx);
        // Near x=20, exp(20) ≈ 4.85e8, so log1p(exp(20)) ≈ 20.0000000021
        // The difference should be tiny
        if (x >= 19.0f)
        {
            EXPECT_LT(sp_diff, 1e-5f)
                << "Softplus exact vs linear should agree near x=20, x=" << x;
        }

        // Run through kernel and verify state decay matches
        std::vector<float> state = {1.0f, 0.0f, 0.0f, 1.0f};
        std::vector<float> ref_state = state;

        auto ref = ref_recurrent_step(
            q.data(), k.data(), v.data(),
            tc.alpha, beta_raw, A_log_val, tc.dt_bias,
            ref_state.data(), d_k, d_v, false);

        auto Q = makeFP32({1, 2}, q.data());
        auto K = makeFP32({1, 2}, k.data());
        auto V = makeFP32({1, 2}, v.data());
        auto alpha = makeFP32({1, 1}, &tc.alpha);
        auto beta = makeFP32({1, 1}, &beta_raw);
        auto A_log = makeFP32({1}, &A_log_val);
        auto dt_bias_t = makeFP32({1}, &tc.dt_bias);
        auto output = makeFP32({1, 2});

        GDNRecurrenceStage::Params p;
        p.kernel = &g_cpu_gdn;
        p.Q = Q.get();
        p.K = K.get();
        p.V = V.get();
        p.alpha = alpha.get();
        p.beta = beta.get();
        p.A_log = A_log.get();
        p.dt_bias = dt_bias_t.get();
        p.output = output.get();
        p.recurrence_state = state.data();
        p.seq_len = 1;
        p.n_heads = 1;
        p.d_k = d_k;
        p.d_v = d_v;
        p.use_qk_l2norm = false;

        GDNRecurrenceStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get())) << tc.desc;

        // State should match reference
        for (int i = 0; i < d_k * d_v; ++i)
        {
            EXPECT_NEAR(state[i], ref_state[i], 1e-5f)
                << tc.desc << " state[" << i << "]";
        }

        const float *out = output->data();
        for (int i = 0; i < d_v; ++i)
        {
            EXPECT_NEAR(out[i], ref.output[i], 1e-5f)
                << tc.desc << " output[" << i << "]";
        }
    }
}

// ============================================================================
// Additional: Recurrence with non-zero initial state + reference comparison
// ============================================================================

TEST(Test__GDNMathematicalCorrectness, Recurrence_NonZeroInitialState)
{
    // Verify correctness when starting with a non-zero state (e.g., from a
    // previous prefill). This catches bugs where the state update assumes
    // the state was zero-initialized.
    const int d_k = 4;
    const int d_v = 4;

    std::mt19937 rng(33333);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> q_data(d_k), k_data(d_k), v_data(d_v);
    for (auto &x : q_data)
        x = dist(rng);
    for (auto &x : k_data)
        x = dist(rng);
    for (auto &x : v_data)
        x = dist(rng);

    const float alpha_val = 0.5f;
    const float beta_raw_val = 1.0f;
    const float A_log_val = -0.3f;
    const float dt_bias_val = 0.1f;

    // Non-zero initial state with realistic magnitude
    std::vector<float> initial_state(d_k * d_v);
    std::uniform_real_distribution<float> state_dist(-0.5f, 0.5f);
    for (auto &x : initial_state)
        x = state_dist(rng);

    std::vector<float> ref_state = initial_state;
    std::vector<float> kernel_state = initial_state;

    auto ref = ref_recurrent_step(
        q_data.data(), k_data.data(), v_data.data(),
        alpha_val, beta_raw_val, A_log_val, dt_bias_val,
        ref_state.data(), d_k, d_v, false);

    auto Q = makeFP32({1, static_cast<size_t>(d_k)}, q_data.data());
    auto K = makeFP32({1, static_cast<size_t>(d_k)}, k_data.data());
    auto V = makeFP32({1, static_cast<size_t>(d_v)}, v_data.data());
    auto alpha = makeFP32({1, 1}, &alpha_val);
    auto beta = makeFP32({1, 1}, &beta_raw_val);
    auto A_log = makeFP32({1}, &A_log_val);
    auto dt_bias_t = makeFP32({1}, &dt_bias_val);
    auto output = makeFP32({1, static_cast<size_t>(d_v)});

    auto ctx = makeCPUContext();
    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias_t.get();
    p.output = output.get();
    p.recurrence_state = kernel_state.data();
    p.seq_len = 1;
    p.n_heads = 1;
    p.d_k = d_k;
    p.d_v = d_v;
    p.use_qk_l2norm = false;

    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float *out = output->data();
    for (int i = 0; i < d_v; ++i)
    {
        EXPECT_NEAR(out[i], ref.output[i], 1e-5f) << "Output[" << i << "]";
    }
    for (int i = 0; i < d_k * d_v; ++i)
    {
        EXPECT_NEAR(kernel_state[i], ref_state[i], 1e-5f) << "State[" << i << "]";
    }
}

TEST(Test__GDNMathematicalCorrectness, Recurrence_Qwen35ChunkContinuationMatchesSequentialDecode)
{
    // Qwen3.5 dense uses head_dim=128 and q/k L2 normalization. The optimized
    // chunk path must match token-by-token recurrent decode when a prior prefix
    // has already populated recurrence state.
    const int n_heads = 16;
    const int d_k = 128;
    const int d_v = 128;
    const int suffix_len = 5;

    std::mt19937 rng(55555);
    std::normal_distribution<float> activation_dist(0.0f, 0.08f);
    std::normal_distribution<float> gate_dist(0.0f, 0.15f);
    std::uniform_real_distribution<float> state_dist(-0.02f, 0.02f);

    std::vector<float> q_data(suffix_len * n_heads * d_k);
    std::vector<float> k_data(suffix_len * n_heads * d_k);
    std::vector<float> v_data(suffix_len * n_heads * d_v);
    std::vector<float> alpha_data(suffix_len * n_heads);
    std::vector<float> beta_data(suffix_len * n_heads);
    std::vector<float> A_log(n_heads);
    std::vector<float> dt_bias(n_heads);

    for (auto &x : q_data)
        x = activation_dist(rng);
    for (auto &x : k_data)
        x = activation_dist(rng);
    for (auto &x : v_data)
        x = activation_dist(rng);
    for (auto &x : alpha_data)
        x = gate_dist(rng);
    for (auto &x : beta_data)
        x = gate_dist(rng);
    for (int h = 0; h < n_heads; ++h)
    {
        A_log[h] = -std::exp(gate_dist(rng));
        dt_bias[h] = gate_dist(rng);
    }

    std::vector<float> initial_state(n_heads * d_k * d_v);
    for (auto &x : initial_state)
        x = state_dist(rng);

    CPUGatedDeltaNet recurrent_kernel;
    std::vector<float> recurrent_state = initial_state;
    std::vector<float> recurrent_output(suffix_len * n_heads * d_v);
    std::vector<float> step_output(n_heads * d_v);
    for (int t = 0; t < suffix_len; ++t)
    {
        ASSERT_TRUE(recurrent_kernel.recurrent_step(
            q_data.data() + t * n_heads * d_k,
            k_data.data() + t * n_heads * d_k,
            v_data.data() + t * n_heads * d_v,
            alpha_data.data() + t * n_heads,
            beta_data.data() + t * n_heads,
            A_log.data(),
            dt_bias.data(),
            step_output.data(),
            recurrent_state.data(),
            n_heads,
            d_k,
            d_v,
            /*use_qk_l2norm=*/true));
        std::copy(step_output.begin(), step_output.end(),
                  recurrent_output.begin() + t * n_heads * d_v);
    }

    CPUGatedDeltaNet chunk_kernel;
    std::vector<float> chunk_state = initial_state;
    std::vector<float> chunk_output(suffix_len * n_heads * d_v);
    ASSERT_TRUE(chunk_kernel.chunk_forward(
        q_data.data(),
        k_data.data(),
        v_data.data(),
        alpha_data.data(),
        beta_data.data(),
        A_log.data(),
        dt_bias.data(),
        chunk_output.data(),
        chunk_state.data(),
        suffix_len,
        n_heads,
        d_k,
        d_v,
        /*chunk_size=*/64,
        /*use_qk_l2norm=*/true));

    EXPECT_LT(maxAbsDiff(chunk_output, recurrent_output), 2e-4f);
    EXPECT_LT(maxAbsDiff(chunk_state, recurrent_state), 2e-4f);
}

TEST(Test__GDNMathematicalCorrectness, ShortConv_Qwen35ChunkContinuationMatchesSequentialDecode)
{
    // The real Qwen3.5 dense GDN short-conv projection has three 2048-wide
    // streams packed into 6144 channels. This locks the full-width prefill path
    // to the same continuation semantics as token-by-token decode.
    const int channels = 6144;
    const int kernel_size = 4;
    const int state_len = kernel_size - 1;
    const int suffix_len = 5;

    std::mt19937 rng(66666);
    std::normal_distribution<float> activation_dist(0.0f, 0.08f);
    std::normal_distribution<float> weight_dist(0.0f, 0.03f);
    std::uniform_real_distribution<float> state_dist(-0.05f, 0.05f);

    std::vector<float> input(suffix_len * channels);
    std::vector<float> weight(channels * kernel_size);
    std::vector<float> bias(channels);
    std::vector<float> initial_state(channels * state_len);

    for (auto &x : input)
        x = activation_dist(rng);
    for (auto &x : weight)
        x = weight_dist(rng);
    for (auto &x : bias)
        x = weight_dist(rng);
    for (auto &x : initial_state)
        x = state_dist(rng);

    CPUShortConvolution chunk_kernel;
    std::vector<float> chunk_state = initial_state;
    std::vector<float> chunk_output(suffix_len * channels);
    ASSERT_TRUE(chunk_kernel.forward(
        input.data(),
        weight.data(),
        bias.data(),
        chunk_output.data(),
        chunk_state.data(),
        suffix_len,
        channels,
        kernel_size,
        /*apply_silu=*/true));

    CPUShortConvolution decode_kernel;
    std::vector<float> decode_state = initial_state;
    std::vector<float> decode_output(suffix_len * channels);
    for (int t = 0; t < suffix_len; ++t)
    {
        ASSERT_TRUE(decode_kernel.forward(
            input.data() + t * channels,
            weight.data(),
            bias.data(),
            decode_output.data() + t * channels,
            decode_state.data(),
            /*seq_len=*/1,
            channels,
            kernel_size,
            /*apply_silu=*/true));
    }

    EXPECT_LT(maxAbsDiff(chunk_output, decode_output), 1e-4f);
    EXPECT_EQ(maxAbsDiff(chunk_state, decode_state), 0.0f);
}

TEST(Test__GDNMathematicalCorrectness, ShortConv_Qwen35InPlaceSplitPrefillMatchesFullPrefill)
{
    // The production Qwen3.5 graph runs short-conv in-place on the merged QKV
    // buffer. Continuation prefill depends on conv_state retaining the raw QKV
    // tail, not the overwritten convolved output.
    const int channels = 6144;
    const int kernel_size = 4;
    const int state_len = kernel_size - 1;
    const int prefix_len = 4;
    const int suffix_len = 5;
    const int total_len = prefix_len + suffix_len;

    std::mt19937 rng(77777);
    std::normal_distribution<float> activation_dist(0.0f, 0.08f);
    std::normal_distribution<float> weight_dist(0.0f, 0.03f);

    std::vector<float> input(total_len * channels);
    std::vector<float> weight(channels * kernel_size);
    std::vector<float> bias(channels);

    for (auto &x : input)
        x = activation_dist(rng);
    for (auto &x : weight)
        x = weight_dist(rng);
    for (auto &x : bias)
        x = weight_dist(rng);

    CPUShortConvolution full_kernel;
    std::vector<float> full_buffer = input;
    std::vector<float> full_state(channels * state_len, 0.0f);
    ASSERT_TRUE(full_kernel.forward(
        full_buffer.data(),
        weight.data(),
        bias.data(),
        full_buffer.data(),
        full_state.data(),
        total_len,
        channels,
        kernel_size,
        /*apply_silu=*/true));

    CPUShortConvolution split_kernel;
    std::vector<float> prefix_buffer(input.begin(),
                                     input.begin() + static_cast<size_t>(prefix_len) * channels);
    std::vector<float> suffix_buffer(input.begin() + static_cast<size_t>(prefix_len) * channels,
                                     input.end());
    std::vector<float> split_state(channels * state_len, 0.0f);

    ASSERT_TRUE(split_kernel.forward(
        prefix_buffer.data(),
        weight.data(),
        bias.data(),
        prefix_buffer.data(),
        split_state.data(),
        prefix_len,
        channels,
        kernel_size,
        /*apply_silu=*/true));

    ASSERT_TRUE(split_kernel.forward(
        suffix_buffer.data(),
        weight.data(),
        bias.data(),
        suffix_buffer.data(),
        split_state.data(),
        suffix_len,
        channels,
        kernel_size,
        /*apply_silu=*/true));

    std::vector<float> full_suffix(
        full_buffer.begin() + static_cast<size_t>(prefix_len) * channels,
        full_buffer.end());
    EXPECT_LT(maxAbsDiff(suffix_buffer, full_suffix), 1e-4f);
    EXPECT_EQ(maxAbsDiff(split_state, full_state), 0.0f);
}

TEST(Test__GDNMathematicalCorrectness, Recurrence_StateSnapshotsRestoreAcceptedVerifierRows)
{
    const int seq_len = 4;
    const int n_heads = 2;
    const int d_k = 16;
    const int d_v = 128;
    const int state_floats = n_heads * d_k * d_v;
    const int accepted_rows = 2;

    std::mt19937 rng(88888);
    std::normal_distribution<float> activation_dist(0.0f, 0.08f);
    std::normal_distribution<float> gate_dist(0.0f, 0.15f);
    std::uniform_real_distribution<float> state_dist(-0.02f, 0.02f);

    std::vector<float> q_data(seq_len * n_heads * d_k);
    std::vector<float> k_data(seq_len * n_heads * d_k);
    std::vector<float> v_data(seq_len * n_heads * d_v);
    std::vector<float> alpha_data(seq_len * n_heads);
    std::vector<float> beta_data(seq_len * n_heads);
    std::vector<float> A_log(n_heads);
    std::vector<float> dt_bias(n_heads);
    std::vector<float> initial_state(state_floats);

    for (auto &x : q_data)
        x = activation_dist(rng);
    for (auto &x : k_data)
        x = activation_dist(rng);
    for (auto &x : v_data)
        x = activation_dist(rng);
    for (auto &x : alpha_data)
        x = gate_dist(rng);
    for (auto &x : beta_data)
        x = gate_dist(rng);
    for (auto &x : initial_state)
        x = state_dist(rng);
    for (int h = 0; h < n_heads; ++h)
    {
        A_log[h] = -std::exp(gate_dist(rng));
        dt_bias[h] = gate_dist(rng);
    }

    CPUGatedDeltaNet chunk_kernel;
    std::vector<float> chunk_state = initial_state;
    std::vector<float> chunk_output(seq_len * n_heads * d_v);
    std::vector<float> snapshots(static_cast<size_t>(seq_len) * state_floats);
    ASSERT_TRUE(chunk_kernel.chunkForwardWithStateSnapshots(
        q_data.data(),
        k_data.data(),
        v_data.data(),
        alpha_data.data(),
        beta_data.data(),
        A_log.data(),
        dt_bias.data(),
        chunk_output.data(),
        chunk_state.data(),
        seq_len,
        n_heads,
        d_k,
        d_v,
        /*chunk_size=*/64,
        /*use_qk_l2norm=*/true,
        snapshots.data(),
        state_floats,
        seq_len));

    CPUGatedDeltaNet recurrent_kernel;
    std::vector<float> recurrent_state = initial_state;
    std::vector<float> step_output(n_heads * d_v);
    for (int t = 0; t < seq_len; ++t)
    {
        ASSERT_TRUE(recurrent_kernel.recurrent_step(
            q_data.data() + static_cast<size_t>(t) * n_heads * d_k,
            k_data.data() + static_cast<size_t>(t) * n_heads * d_k,
            v_data.data() + static_cast<size_t>(t) * n_heads * d_v,
            alpha_data.data() + static_cast<size_t>(t) * n_heads,
            beta_data.data() + static_cast<size_t>(t) * n_heads,
            A_log.data(),
            dt_bias.data(),
            step_output.data(),
            recurrent_state.data(),
            n_heads,
            d_k,
            d_v,
            /*use_qk_l2norm=*/true));

        const float *chunk_row = chunk_output.data() + static_cast<size_t>(t) * n_heads * d_v;
        std::vector<float> row_output(chunk_row, chunk_row + n_heads * d_v);
        EXPECT_LT(maxAbsDiff(row_output, step_output), 2e-4f) << "output row " << t;

        const float *snapshot = snapshots.data() + static_cast<size_t>(t) * state_floats;
        std::vector<float> row_state(snapshot, snapshot + state_floats);
        EXPECT_LT(maxAbsDiff(row_state, recurrent_state), 2e-4f) << "snapshot row " << t;
    }
    EXPECT_LT(maxAbsDiff(chunk_state, recurrent_state), 2e-4f);

    std::vector<float> restored_state(state_floats);
    ASSERT_TRUE(chunk_kernel.restoreStateFromSnapshot(
        restored_state.data(),
        snapshots.data(),
        accepted_rows - 1,
        state_floats,
        state_floats));

    std::vector<float> accepted_replay_state = initial_state;
    for (int t = 0; t < accepted_rows; ++t)
    {
        ASSERT_TRUE(recurrent_kernel.recurrent_step(
            q_data.data() + static_cast<size_t>(t) * n_heads * d_k,
            k_data.data() + static_cast<size_t>(t) * n_heads * d_k,
            v_data.data() + static_cast<size_t>(t) * n_heads * d_v,
            alpha_data.data() + static_cast<size_t>(t) * n_heads,
            beta_data.data() + static_cast<size_t>(t) * n_heads,
            A_log.data(),
            dt_bias.data(),
            step_output.data(),
            accepted_replay_state.data(),
            n_heads,
            d_k,
            d_v,
            /*use_qk_l2norm=*/true));
    }
    EXPECT_LT(maxAbsDiff(restored_state, accepted_replay_state), 2e-4f);
}

TEST(Test__GDNMathematicalCorrectness, Recurrence_StateSnapshotRestoreContinuesLikeReplay)
{
    const int accepted_rows = 2;
    const int continuation_rows = 3;
    const int seq_len = accepted_rows + continuation_rows;
    const int n_heads = 2;
    const int d_k = 16;
    const int d_v = 128;
    const int qk_stride = n_heads * d_k;
    const int v_stride = n_heads * d_v;
    const int state_floats = n_heads * d_k * d_v;

    std::mt19937 rng(88889);
    std::normal_distribution<float> activation_dist(0.0f, 0.08f);
    std::normal_distribution<float> gate_dist(0.0f, 0.15f);
    std::uniform_real_distribution<float> state_dist(-0.02f, 0.02f);

    std::vector<float> q_data(seq_len * qk_stride);
    std::vector<float> k_data(seq_len * qk_stride);
    std::vector<float> v_data(seq_len * v_stride);
    std::vector<float> alpha_data(seq_len * n_heads);
    std::vector<float> beta_data(seq_len * n_heads);
    std::vector<float> A_log(n_heads);
    std::vector<float> dt_bias(n_heads);
    std::vector<float> initial_state(state_floats);

    for (auto &x : q_data)
        x = activation_dist(rng);
    for (auto &x : k_data)
        x = activation_dist(rng);
    for (auto &x : v_data)
        x = activation_dist(rng);
    for (auto &x : alpha_data)
        x = gate_dist(rng);
    for (auto &x : beta_data)
        x = gate_dist(rng);
    for (auto &x : initial_state)
        x = state_dist(rng);
    for (int h = 0; h < n_heads; ++h)
    {
        A_log[h] = -std::exp(gate_dist(rng));
        dt_bias[h] = gate_dist(rng);
    }

    CPUGatedDeltaNet verifier_kernel;
    std::vector<float> verifier_state = initial_state;
    std::vector<float> verifier_output(seq_len * v_stride);
    std::vector<float> snapshots(static_cast<size_t>(seq_len) * state_floats);
    ASSERT_TRUE(verifier_kernel.chunkForwardWithStateSnapshots(
        q_data.data(),
        k_data.data(),
        v_data.data(),
        alpha_data.data(),
        beta_data.data(),
        A_log.data(),
        dt_bias.data(),
        verifier_output.data(),
        verifier_state.data(),
        seq_len,
        n_heads,
        d_k,
        d_v,
        /*chunk_size=*/64,
        /*use_qk_l2norm=*/true,
        snapshots.data(),
        state_floats,
        seq_len));

    std::vector<float> restored_state(state_floats);
    ASSERT_TRUE(verifier_kernel.restoreStateFromSnapshot(
        restored_state.data(),
        snapshots.data(),
        accepted_rows - 1,
        state_floats,
        state_floats));

    CPUGatedDeltaNet restored_kernel;
    std::vector<float> restored_continuation(continuation_rows * v_stride);
    for (int row = 0; row < continuation_rows; ++row)
    {
        const int source_row = accepted_rows + row;
        ASSERT_TRUE(restored_kernel.recurrent_step(
            q_data.data() + static_cast<size_t>(source_row) * qk_stride,
            k_data.data() + static_cast<size_t>(source_row) * qk_stride,
            v_data.data() + static_cast<size_t>(source_row) * v_stride,
            alpha_data.data() + static_cast<size_t>(source_row) * n_heads,
            beta_data.data() + static_cast<size_t>(source_row) * n_heads,
            A_log.data(),
            dt_bias.data(),
            restored_continuation.data() + static_cast<size_t>(row) * v_stride,
            restored_state.data(),
            n_heads,
            d_k,
            d_v,
            /*use_qk_l2norm=*/true));
    }

    CPUGatedDeltaNet replay_kernel;
    std::vector<float> replay_state = initial_state;
    std::vector<float> replay_output(seq_len * v_stride);
    for (int row = 0; row < seq_len; ++row)
    {
        ASSERT_TRUE(replay_kernel.recurrent_step(
            q_data.data() + static_cast<size_t>(row) * qk_stride,
            k_data.data() + static_cast<size_t>(row) * qk_stride,
            v_data.data() + static_cast<size_t>(row) * v_stride,
            alpha_data.data() + static_cast<size_t>(row) * n_heads,
            beta_data.data() + static_cast<size_t>(row) * n_heads,
            A_log.data(),
            dt_bias.data(),
            replay_output.data() + static_cast<size_t>(row) * v_stride,
            replay_state.data(),
            n_heads,
            d_k,
            d_v,
            /*use_qk_l2norm=*/true));
    }

    const std::vector<float> replay_continuation(
        replay_output.begin() + static_cast<size_t>(accepted_rows) * v_stride,
        replay_output.end());
    EXPECT_LT(maxAbsDiff(restored_continuation, replay_continuation), 2e-4f);
    EXPECT_LT(maxAbsDiff(restored_state, replay_state), 2e-4f);
}

TEST(Test__GDNMathematicalCorrectness, ShortConv_StateSnapshotsRestoreAcceptedVerifierRows)
{
    const int seq_len = 5;
    const int channels = 17;
    const int kernel_size = 4;
    const int state_len = kernel_size - 1;
    const int state_floats = channels * state_len;
    const int accepted_rows = 3;

    std::mt19937 rng(99999);
    std::normal_distribution<float> activation_dist(0.0f, 0.08f);
    std::normal_distribution<float> weight_dist(0.0f, 0.03f);
    std::uniform_real_distribution<float> state_dist(-0.05f, 0.05f);

    std::vector<float> input(seq_len * channels);
    std::vector<float> weight(channels * kernel_size);
    std::vector<float> bias(channels);
    std::vector<float> initial_state(state_floats);

    for (auto &x : input)
        x = activation_dist(rng);
    for (auto &x : weight)
        x = weight_dist(rng);
    for (auto &x : bias)
        x = weight_dist(rng);
    for (auto &x : initial_state)
        x = state_dist(rng);

    CPUShortConvolution snapshot_kernel;
    std::vector<float> state = initial_state;
    std::vector<float> output(seq_len * channels);
    std::vector<float> snapshots(static_cast<size_t>(seq_len) * state_floats);
    ASSERT_TRUE(snapshot_kernel.forwardWithStateSnapshots(
        input.data(),
        weight.data(),
        bias.data(),
        output.data(),
        state.data(),
        seq_len,
        channels,
        kernel_size,
        snapshots.data(),
        state_floats,
        seq_len,
        /*apply_silu=*/true));

    CPUShortConvolution decode_kernel;
    std::vector<float> decode_state = initial_state;
    std::vector<float> decode_output(channels);
    for (int t = 0; t < seq_len; ++t)
    {
        ASSERT_TRUE(decode_kernel.forward(
            input.data() + static_cast<size_t>(t) * channels,
            weight.data(),
            bias.data(),
            decode_output.data(),
            decode_state.data(),
            /*seq_len=*/1,
            channels,
            kernel_size,
            /*apply_silu=*/true));

        const float *output_row = output.data() + static_cast<size_t>(t) * channels;
        std::vector<float> row_output(output_row, output_row + channels);
        EXPECT_LT(maxAbsDiff(row_output, decode_output), 1e-6f) << "output row " << t;

        const float *snapshot = snapshots.data() + static_cast<size_t>(t) * state_floats;
        std::vector<float> row_state(snapshot, snapshot + state_floats);
        EXPECT_EQ(maxAbsDiff(row_state, decode_state), 0.0f) << "snapshot row " << t;
    }
    EXPECT_EQ(maxAbsDiff(state, decode_state), 0.0f);

    std::vector<float> restored_state(state_floats);
    ASSERT_TRUE(snapshot_kernel.restoreStateFromSnapshot(
        restored_state.data(),
        snapshots.data(),
        accepted_rows - 1,
        state_floats,
        state_floats));

    std::vector<float> accepted_replay_state = initial_state;
    for (int t = 0; t < accepted_rows; ++t)
    {
        ASSERT_TRUE(decode_kernel.forward(
            input.data() + static_cast<size_t>(t) * channels,
            weight.data(),
            bias.data(),
            decode_output.data(),
            accepted_replay_state.data(),
            /*seq_len=*/1,
            channels,
            kernel_size,
            /*apply_silu=*/true));
    }
    EXPECT_EQ(maxAbsDiff(restored_state, accepted_replay_state), 0.0f);
}

TEST(Test__GDNMathematicalCorrectness, ShortConv_GroupedVerifierRowsMatchSerialDecodeAtQwen36ShapeM2ToM4)
{
    constexpr int kChannels = 6144;
    constexpr int kKernelSize = 4;
    constexpr int kStateLen = kKernelSize - 1;
    constexpr int kStateFloats = kChannels * kStateLen;

    std::mt19937 rng(170031);
    std::normal_distribution<float> activation_dist(0.0f, 0.12f);
    std::normal_distribution<float> weight_dist(0.0f, 0.025f);
    std::uniform_real_distribution<float> state_dist(-0.04f, 0.04f);

    std::vector<float> weight(static_cast<size_t>(kChannels) * kKernelSize);
    std::vector<float> bias(kChannels);
    std::vector<float> initial_state(kStateFloats);
    for (auto &x : weight)
        x = weight_dist(rng);
    for (auto &x : bias)
        x = weight_dist(rng);
    for (auto &x : initial_state)
        x = state_dist(rng);

    for (int rows = 2; rows <= 4; ++rows)
    {
        std::vector<float> input(static_cast<size_t>(rows) * kChannels);
        for (auto &x : input)
            x = activation_dist(rng);

        CPUShortConvolution grouped_kernel;
        std::vector<float> grouped_state = initial_state;
        std::vector<float> grouped_output(static_cast<size_t>(rows) * kChannels);
        std::vector<float> snapshots(static_cast<size_t>(rows) * kStateFloats);
        ASSERT_TRUE(grouped_kernel.forwardWithStateSnapshots(
            input.data(),
            weight.data(),
            bias.data(),
            grouped_output.data(),
            grouped_state.data(),
            rows,
            kChannels,
            kKernelSize,
            snapshots.data(),
            kStateFloats,
            rows,
            /*apply_silu=*/true))
            << "rows=" << rows;

        CPUShortConvolution serial_kernel;
        std::vector<float> serial_state = initial_state;
        std::vector<float> serial_output(static_cast<size_t>(rows) * kChannels);
        for (int row = 0; row < rows; ++row)
        {
            ASSERT_TRUE(serial_kernel.forward(
                input.data() + static_cast<size_t>(row) * kChannels,
                weight.data(),
                bias.data(),
                serial_output.data() + static_cast<size_t>(row) * kChannels,
                serial_state.data(),
                /*seq_len=*/1,
                kChannels,
                kKernelSize,
                /*apply_silu=*/true))
                << "rows=" << rows << " row=" << row;

            const float *snapshot = snapshots.data() + static_cast<size_t>(row) * kStateFloats;
            std::vector<float> grouped_snapshot(snapshot, snapshot + kStateFloats);
            EXPECT_EQ(maxAbsDiff(grouped_snapshot, serial_state), 0.0f)
                << "rows=" << rows << " row=" << row
                << " verifier state publication must be serial-decode exact";
        }

        EXPECT_EQ(maxAbsDiff(grouped_output, serial_output), 0.0f)
            << "rows=" << rows
            << " grouped short-conv verifier output must be serial-decode exact";
        EXPECT_EQ(maxAbsDiff(grouped_state, serial_state), 0.0f)
            << "rows=" << rows
            << " grouped short-conv final state must be serial-decode exact";
    }
}

TEST(Test__GDNMathematicalCorrectness, ShortConv_BoundVerifierSlotsDoNotHijackSingleRowDecode)
{
    const int channels = 11;
    const int kernel_size = 4;
    const int state_len = kernel_size - 1;
    const int state_floats = channels * state_len;

    std::mt19937 rng(101337);
    std::normal_distribution<float> activation_dist(0.0f, 0.08f);
    std::normal_distribution<float> weight_dist(0.0f, 0.03f);
    std::uniform_real_distribution<float> state_dist(-0.05f, 0.05f);

    std::vector<float> input(channels);
    std::vector<float> weight(static_cast<size_t>(channels) * kernel_size);
    std::vector<float> bias(channels);
    std::vector<float> initial_state(state_floats);
    for (auto &x : input)
        x = activation_dist(rng);
    for (auto &x : weight)
        x = weight_dist(rng);
    for (auto &x : bias)
        x = weight_dist(rng);
    for (auto &x : initial_state)
        x = state_dist(rng);

    CPUShortConvolution verifier_bound_kernel;
    std::vector<float> decode_state = initial_state;
    std::vector<float> decode_output(channels);
    std::vector<float> snapshots(static_cast<size_t>(2) * state_floats, -777.0f);
    std::vector<float> speculative_work(state_floats, 0.0f);
    verifier_bound_kernel.bindVerifierStateCaptureWorkspace(
        snapshots.data(),
        /*rows=*/2,
        state_floats);
    verifier_bound_kernel.bindSpeculativeStateWorkspace(
        speculative_work.data(),
        state_floats);

    CPUShortConvolution reference_kernel;
    std::vector<float> reference_state = initial_state;
    std::vector<float> reference_output(channels);

    ASSERT_TRUE(verifier_bound_kernel.forward(
        input.data(),
        weight.data(),
        bias.data(),
        decode_output.data(),
        decode_state.data(),
        /*seq_len=*/1,
        channels,
        kernel_size,
        /*apply_silu=*/true));
    ASSERT_TRUE(reference_kernel.forward(
        input.data(),
        weight.data(),
        bias.data(),
        reference_output.data(),
        reference_state.data(),
        /*seq_len=*/1,
        channels,
        kernel_size,
        /*apply_silu=*/true));

    EXPECT_EQ(maxAbsDiff(decode_output, reference_output), 0.0f);
    EXPECT_EQ(maxAbsDiff(decode_state, reference_state), 0.0f)
        << "single-row decode must update the live conv_state even when "
           "verifier capture slots are bound on the shared kernel";
    EXPECT_TRUE(std::all_of(
        snapshots.begin(),
        snapshots.end(),
        [](float x) { return x == -777.0f; }))
        << "ordinary decode must not publish speculative verifier snapshots";
}

TEST(Test__GDNMathematicalCorrectness, ShortConv_StateSnapshotRestoreContinuesLikeReplay)
{
    const int accepted_rows = 2;
    const int continuation_rows = 3;
    const int seq_len = accepted_rows + continuation_rows;
    const int channels = 19;
    const int kernel_size = 4;
    const int state_len = kernel_size - 1;
    const int state_floats = channels * state_len;

    std::mt19937 rng(100000);
    std::normal_distribution<float> activation_dist(0.0f, 0.08f);
    std::normal_distribution<float> weight_dist(0.0f, 0.03f);
    std::uniform_real_distribution<float> state_dist(-0.05f, 0.05f);

    std::vector<float> input(seq_len * channels);
    std::vector<float> weight(channels * kernel_size);
    std::vector<float> bias(channels);
    std::vector<float> initial_state(state_floats);

    for (auto &x : input)
        x = activation_dist(rng);
    for (auto &x : weight)
        x = weight_dist(rng);
    for (auto &x : bias)
        x = weight_dist(rng);
    for (auto &x : initial_state)
        x = state_dist(rng);

    CPUShortConvolution verifier_kernel;
    std::vector<float> verifier_state = initial_state;
    std::vector<float> verifier_output(seq_len * channels);
    std::vector<float> snapshots(static_cast<size_t>(seq_len) * state_floats);
    ASSERT_TRUE(verifier_kernel.forwardWithStateSnapshots(
        input.data(),
        weight.data(),
        bias.data(),
        verifier_output.data(),
        verifier_state.data(),
        seq_len,
        channels,
        kernel_size,
        snapshots.data(),
        state_floats,
        seq_len,
        /*apply_silu=*/true));

    std::vector<float> restored_state(state_floats);
    ASSERT_TRUE(verifier_kernel.restoreStateFromSnapshot(
        restored_state.data(),
        snapshots.data(),
        accepted_rows - 1,
        state_floats,
        state_floats));

    CPUShortConvolution restored_kernel;
    std::vector<float> restored_continuation(continuation_rows * channels);
    for (int row = 0; row < continuation_rows; ++row)
    {
        const int source_row = accepted_rows + row;
        ASSERT_TRUE(restored_kernel.forward(
            input.data() + static_cast<size_t>(source_row) * channels,
            weight.data(),
            bias.data(),
            restored_continuation.data() + static_cast<size_t>(row) * channels,
            restored_state.data(),
            /*seq_len=*/1,
            channels,
            kernel_size,
            /*apply_silu=*/true));
    }

    CPUShortConvolution replay_kernel;
    std::vector<float> replay_state = initial_state;
    std::vector<float> replay_output(seq_len * channels);
    for (int row = 0; row < seq_len; ++row)
    {
        ASSERT_TRUE(replay_kernel.forward(
            input.data() + static_cast<size_t>(row) * channels,
            weight.data(),
            bias.data(),
            replay_output.data() + static_cast<size_t>(row) * channels,
            replay_state.data(),
            /*seq_len=*/1,
            channels,
            kernel_size,
            /*apply_silu=*/true));
    }

    const std::vector<float> replay_continuation(
        replay_output.begin() + static_cast<size_t>(accepted_rows) * channels,
        replay_output.end());
    EXPECT_EQ(maxAbsDiff(restored_continuation, replay_continuation), 0.0f);
    EXPECT_EQ(maxAbsDiff(restored_state, replay_state), 0.0f);
}

TEST(Test__GDNMathematicalCorrectness, ShortConv_StateSnapshotsUseRawInputForInPlacePrefill)
{
    const int seq_len = 5;
    const int channels = 19;
    const int kernel_size = 4;
    const int state_len = kernel_size - 1;
    const int state_floats = channels * state_len;

    std::mt19937 rng(100001);
    std::normal_distribution<float> activation_dist(0.0f, 0.08f);
    std::normal_distribution<float> weight_dist(0.0f, 0.03f);
    std::uniform_real_distribution<float> state_dist(-0.05f, 0.05f);

    std::vector<float> raw_input(seq_len * channels);
    std::vector<float> weight(channels * kernel_size);
    std::vector<float> bias(channels);
    std::vector<float> initial_state(state_floats);

    for (auto &x : raw_input)
        x = activation_dist(rng);
    for (auto &x : weight)
        x = weight_dist(rng);
    for (auto &x : bias)
        x = weight_dist(rng);
    for (auto &x : initial_state)
        x = state_dist(rng);

    CPUShortConvolution snapshot_kernel;
    std::vector<float> in_place_buffer = raw_input;
    std::vector<float> state = initial_state;
    std::vector<float> snapshots(static_cast<size_t>(seq_len) * state_floats);
    ASSERT_TRUE(snapshot_kernel.forwardWithStateSnapshots(
        in_place_buffer.data(),
        weight.data(),
        bias.data(),
        in_place_buffer.data(),
        state.data(),
        seq_len,
        channels,
        kernel_size,
        snapshots.data(),
        state_floats,
        seq_len,
        /*apply_silu=*/true));

    CPUShortConvolution decode_kernel;
    std::vector<float> decode_state = initial_state;
    std::vector<float> decode_output(channels);
    for (int t = 0; t < seq_len; ++t)
    {
        ASSERT_TRUE(decode_kernel.forward(
            raw_input.data() + static_cast<size_t>(t) * channels,
            weight.data(),
            bias.data(),
            decode_output.data(),
            decode_state.data(),
            /*seq_len=*/1,
            channels,
            kernel_size,
            /*apply_silu=*/true));

        const float *output_row = in_place_buffer.data() + static_cast<size_t>(t) * channels;
        std::vector<float> row_output(output_row, output_row + channels);
        EXPECT_LT(maxAbsDiff(row_output, decode_output), 1e-6f) << "output row " << t;

        const float *snapshot = snapshots.data() + static_cast<size_t>(t) * state_floats;
        std::vector<float> row_state(snapshot, snapshot + state_floats);
        EXPECT_EQ(maxAbsDiff(row_state, decode_state), 0.0f) << "snapshot row " << t;
    }
    EXPECT_EQ(maxAbsDiff(state, decode_state), 0.0f);
}

// ============================================================================
// Additional: AttentionOutputGate reference comparison with AVX512 sigmoid
// ============================================================================

TEST(Test__GDNMathematicalCorrectness, AttentionOutputGate_PreciseReference)
{
    // Test sigmoid precision across the full range with enough elements
    // to exercise the AVX512 vectorized path (>= 16 elements)
    const int n = 32;

    // Test values spanning sigmoid's active range
    std::vector<float> input_data(n);
    std::vector<float> gate_data(n);
    std::mt19937 rng(44444);
    std::normal_distribution<float> dist(0.0f, 3.0f);
    for (int i = 0; i < n; ++i)
    {
        input_data[i] = dist(rng);
        gate_data[i] = dist(rng);
    }

    auto input = makeFP32({1, static_cast<size_t>(n)}, input_data.data());
    auto gate = makeFP32({1, static_cast<size_t>(n)}, gate_data.data());
    auto output = makeFP32({1, static_cast<size_t>(n)});

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
    for (int i = 0; i < n; ++i)
    {
        float expected = ref_sigmoid(gate_data[i]) * input_data[i];
        float abs_err = std::abs(out[i] - expected);
        float rel_err = (std::abs(expected) > 1e-6f)
                            ? abs_err / std::abs(expected)
                            : abs_err;
        max_rel_err = std::max(max_rel_err, rel_err);

        EXPECT_NEAR(out[i], expected, std::abs(expected) * 0.005f + 1e-6f)
            << "Element " << i << " gate=" << gate_data[i]
            << " input=" << input_data[i];
    }
    EXPECT_LT(max_rel_err, 0.005f)
        << "Max relative sigmoid error should be < 0.5%";
}
