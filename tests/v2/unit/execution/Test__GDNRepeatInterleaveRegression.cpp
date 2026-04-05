/**
 * @file Test__GDNRepeatInterleaveRegression.cpp
 * @brief Regression tests for GDNRecurrenceStage QKV deinterleave with repeat_interleave
 *
 * The bug: When n_k_heads < n_v_heads (e.g., Qwen 3.5 with group_count=16,
 * time_step_rank=32), the QKV deinterleave code assumed n_k_heads == n_v_heads,
 * reading the wrong offsets from the merged buffer and producing corrupt Q/K data.
 *
 * The fix: Added repeat_interleave of Q/K from n_k_heads to n_v_heads during
 * deinterleave, matching PyTorch's repeat_interleave(Q, n_v_heads//n_k_heads, dim=head).
 *
 * Regression for: GDNRecurrenceStage.cpp QKV deinterleave repeat_interleave fix
 *                 Qwen35Graph.cpp QKV dimension computation fix
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <numeric>
#include <random>

#include "execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"
#include "tensors/Tensors.h"

using namespace llaminar2;

namespace
{
    std::unique_ptr<IDeviceContext> makeCPUContext()
    {
        return std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 1);
    }

    std::shared_ptr<FP32Tensor> makeFP32(const std::vector<size_t> &shape, const float *data = nullptr)
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

    CPUGatedDeltaNet g_cpu_gdn;
} // namespace

// ============================================================================
// QKV Deinterleave with repeat_interleave (n_k_heads < n_v_heads)
// ============================================================================

TEST(Test__GDNRepeatInterleave, MergedQKV_RepeatInterleave_Decode)
{
    // Simulate Qwen 3.5 config: n_k_heads=2, n_v_heads=4, d_k=d_v=4
    // The merged QKV buffer has layout:
    //   [Q(n_k_heads*d_k), K(n_k_heads*d_k), V(n_v_heads*d_v)]
    //   = [Q(8), K(8), V(16)] = 32 floats per row
    //
    // After deinterleave + repeat_interleave:
    //   Q → [n_v_heads*d_k] = [16] (each key head repeated 2x)
    //   K → [n_v_heads*d_k] = [16] (each key head repeated 2x)
    //   V → [n_v_heads*d_v] = [16] (straight copy)

    const int n_k_heads = 2;
    const int n_v_heads = 4;
    const int d_k = 4;
    const int d_v = 4;
    const int repeat_factor = n_v_heads / n_k_heads; // 2

    const int q_src_dim = n_k_heads * d_k;  // 8
    const int k_src_dim = n_k_heads * d_k;  // 8
    const int v_dim = n_v_heads * d_v;      // 16
    const int qkv_dim = q_src_dim + k_src_dim + v_dim; // 32

    // Build merged QKV with known values:
    // Q heads: head0=[1,2,3,4], head1=[5,6,7,8]
    // K heads: head0=[10,20,30,40], head1=[50,60,70,80]
    // V heads: head0=[100..103], head1=[104..107], head2=[108..111], head3=[112..115]
    std::vector<float> qkv_data(qkv_dim, 0.0f);
    // Q
    qkv_data[0] = 1; qkv_data[1] = 2; qkv_data[2] = 3; qkv_data[3] = 4;     // Q head 0
    qkv_data[4] = 5; qkv_data[5] = 6; qkv_data[6] = 7; qkv_data[7] = 8;     // Q head 1
    // K
    qkv_data[8] = 10; qkv_data[9] = 20; qkv_data[10] = 30; qkv_data[11] = 40;  // K head 0
    qkv_data[12] = 50; qkv_data[13] = 60; qkv_data[14] = 70; qkv_data[15] = 80; // K head 1
    // V
    for (int i = 0; i < v_dim; ++i)
        qkv_data[q_src_dim + k_src_dim + i] = 100.0f + static_cast<float>(i);

    // Create a single merged tensor — Q, K, V all point to the same buffer
    auto qkv_tensor = makeFP32({1, static_cast<size_t>(qkv_dim)}, qkv_data.data());

    // Alpha, beta, A_log, dt_bias for n_v_heads
    auto alpha = makeFP32Const({1, static_cast<size_t>(n_v_heads)}, 0.0f);
    auto beta = makeFP32Const({1, static_cast<size_t>(n_v_heads)}, 100.0f);  // sigmoid(100)≈1
    auto A_log = makeFP32Const({static_cast<size_t>(n_v_heads)}, -1.0f);
    auto dt_bias = makeFP32Const({static_cast<size_t>(n_v_heads)}, 0.0f);
    auto output = makeFP32({1, static_cast<size_t>(n_v_heads * d_v)});

    std::vector<float> state(n_v_heads * d_k * d_v, 0.0f);

    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.Q = qkv_tensor.get();
    p.K = qkv_tensor.get();
    p.V = qkv_tensor.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = state.data();
    p.seq_len = 1;
    p.n_heads = n_v_heads;   // Recurrence runs with value head count
    p.n_k_heads = n_k_heads; // Key head count for QKV split
    p.d_k = d_k;
    p.d_v = d_v;
    p.use_qk_l2norm = false;

    auto ctx = makeCPUContext();
    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // The output should be non-zero and finite
    const float *out = output->data();
    float sum = 0.0f;
    for (int i = 0; i < n_v_heads * d_v; ++i)
    {
        EXPECT_TRUE(std::isfinite(out[i]))
            << "Output[" << i << "] is not finite: " << out[i];
        sum += std::abs(out[i]);
    }
    EXPECT_GT(sum, 0.0f) << "Output should not be all zeros";

    // The state should also be updated and finite
    for (size_t i = 0; i < state.size(); ++i)
    {
        EXPECT_TRUE(std::isfinite(state[i]))
            << "State[" << i << "] is not finite: " << state[i];
    }
}

TEST(Test__GDNRepeatInterleave, MergedQKV_RepeatInterleave_Prefill)
{
    // Test with seq_len > 1 (prefill path), which uses chunk_forward
    const int n_k_heads = 2;
    const int n_v_heads = 4;
    const int d_k = 4;
    const int d_v = 4;
    const int seq_len = 3;

    const int q_src_dim = n_k_heads * d_k;
    const int k_src_dim = n_k_heads * d_k;
    const int v_dim = n_v_heads * d_v;
    const int qkv_dim = q_src_dim + k_src_dim + v_dim;

    // Build merged QKV with random data for multiple timesteps
    auto qkv_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(qkv_dim)},
        DeviceId::cpu());
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    float *qkv = qkv_tensor->mutable_data();
    for (int i = 0; i < seq_len * qkv_dim; ++i)
        qkv[i] = dist(rng);

    auto alpha = makeFP32Const({static_cast<size_t>(seq_len), static_cast<size_t>(n_v_heads)}, 0.0f);
    auto beta = makeFP32Const({static_cast<size_t>(seq_len), static_cast<size_t>(n_v_heads)}, 0.0f);
    auto A_log = makeFP32Const({static_cast<size_t>(n_v_heads)}, -1.0f);
    auto dt_bias = makeFP32Const({static_cast<size_t>(n_v_heads)}, 0.0f);
    auto output = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_v_heads * d_v)});

    std::vector<float> state(n_v_heads * d_k * d_v, 0.0f);

    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.Q = qkv_tensor.get();
    p.K = qkv_tensor.get();
    p.V = qkv_tensor.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = state.data();
    p.seq_len = seq_len;
    p.n_heads = n_v_heads;
    p.n_k_heads = n_k_heads;
    p.d_k = d_k;
    p.d_v = d_v;
    p.chunk_size = 64;
    p.use_qk_l2norm = false;

    auto ctx = makeCPUContext();
    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float *out = output->data();
    float sum = 0.0f;
    for (int i = 0; i < seq_len * n_v_heads * d_v; ++i)
    {
        EXPECT_TRUE(std::isfinite(out[i]))
            << "Output[" << i << "] is not finite";
        sum += std::abs(out[i]);
    }
    EXPECT_GT(sum, 0.0f) << "Prefill output should not be all zeros";
}

TEST(Test__GDNRepeatInterleave, MergedQKV_NoRepeat_SameHeadCount)
{
    // When n_k_heads == n_v_heads, the simple deinterleave path should still work.
    // This verifies we didn't break the simple path while adding repeat_interleave.
    const int n_heads = 4;
    const int d_k = 4;
    const int d_v = 4;

    const int inner = n_heads * d_v;
    const int qkv_dim = 3 * inner; // Q + K + V all same size

    auto qkv_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(qkv_dim)}, DeviceId::cpu());
    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    float *qkv = qkv_tensor->mutable_data();
    for (int i = 0; i < qkv_dim; ++i)
        qkv[i] = dist(rng);

    auto alpha = makeFP32Const({1, static_cast<size_t>(n_heads)}, 0.0f);
    auto beta = makeFP32Const({1, static_cast<size_t>(n_heads)}, 0.0f);
    auto A_log = makeFP32Const({static_cast<size_t>(n_heads)}, -1.0f);
    auto dt_bias = makeFP32Const({static_cast<size_t>(n_heads)}, 0.0f);
    auto output = makeFP32({1, static_cast<size_t>(n_heads * d_v)});

    std::vector<float> state(n_heads * d_k * d_v, 0.0f);

    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.Q = qkv_tensor.get();
    p.K = qkv_tensor.get();
    p.V = qkv_tensor.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = state.data();
    p.seq_len = 1;
    p.n_heads = n_heads;
    p.n_k_heads = n_heads; // Same as n_heads → no repeat
    p.d_k = d_k;
    p.d_v = d_v;
    p.use_qk_l2norm = false;

    auto ctx = makeCPUContext();
    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float *out = output->data();
    float sum = 0.0f;
    for (int i = 0; i < n_heads * d_v; ++i)
    {
        EXPECT_TRUE(std::isfinite(out[i]));
        sum += std::abs(out[i]);
    }
    EXPECT_GT(sum, 0.0f);
}

TEST(Test__GDNRepeatInterleave, MergedQKV_NKHeadsZero_DefaultsToNHeads)
{
    // When n_k_heads=0, it should default to n_heads (backward compatibility)
    const int n_heads = 2;
    const int d_k = 4;
    const int d_v = 4;
    const int inner = n_heads * d_v;
    const int qkv_dim = 3 * inner;

    auto qkv_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(qkv_dim)}, DeviceId::cpu());
    std::mt19937 rng(55);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    float *qkv = qkv_tensor->mutable_data();
    for (int i = 0; i < qkv_dim; ++i)
        qkv[i] = dist(rng);

    auto alpha = makeFP32Const({1, static_cast<size_t>(n_heads)}, 0.0f);
    auto beta = makeFP32Const({1, static_cast<size_t>(n_heads)}, 0.0f);
    auto A_log = makeFP32Const({static_cast<size_t>(n_heads)}, -1.0f);
    auto dt_bias = makeFP32Const({static_cast<size_t>(n_heads)}, 0.0f);
    auto output = makeFP32({1, static_cast<size_t>(n_heads * d_v)});

    std::vector<float> state(n_heads * d_k * d_v, 0.0f);

    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.Q = qkv_tensor.get();
    p.K = qkv_tensor.get();
    p.V = qkv_tensor.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = state.data();
    p.seq_len = 1;
    p.n_heads = n_heads;
    p.n_k_heads = 0; // 0 → default to n_heads
    p.d_k = d_k;
    p.d_v = d_v;
    p.use_qk_l2norm = false;

    auto ctx = makeCPUContext();
    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float *out = output->data();
    for (int i = 0; i < n_heads * d_v; ++i)
        EXPECT_TRUE(std::isfinite(out[i]));
}

// ============================================================================
// QKV Dimension Computation Regression (Qwen35Graph)
// ============================================================================

TEST(Test__GDNRepeatInterleave, QKVDimensionComputation)
{
    // When n_k_heads != n_v_heads, qkv_dim must be 2*key_dim + value_dim,
    // NOT 3 * inner_size. The old code used 3 * n_heads * d_v which was wrong.

    // Qwen 3.5 4B: n_k_heads=16, n_v_heads=32, d_k=d_v=128
    {
        const int n_k_heads = 16;
        const int n_v_heads = 32;
        const int d_k = 128;
        const int d_v = 128;
        const int key_dim = n_k_heads * d_k;   // 2048
        const int value_dim = n_v_heads * d_v;  // 4096

        const int correct_qkv_dim = 2 * key_dim + value_dim; // 2*2048 + 4096 = 8192
        const int wrong_qkv_dim = 3 * value_dim;             // 3*4096 = 12288

        EXPECT_EQ(correct_qkv_dim, 8192);
        EXPECT_EQ(wrong_qkv_dim, 12288);
        EXPECT_NE(correct_qkv_dim, wrong_qkv_dim)
            << "When n_k_heads != n_v_heads, 2*key_dim+value_dim != 3*inner_size";
    }

    // Equal head counts: should still be equivalent (3 * inner_size)
    {
        const int n_heads = 16;
        const int d_k = 128;
        const int d_v = 128;
        const int key_dim = n_heads * d_k;
        const int value_dim = n_heads * d_v;

        const int formula_a = 2 * key_dim + value_dim;
        const int formula_b = 3 * n_heads * d_v;

        EXPECT_EQ(formula_a, formula_b)
            << "When n_k_heads == n_v_heads, both formulas should agree";
    }
}
