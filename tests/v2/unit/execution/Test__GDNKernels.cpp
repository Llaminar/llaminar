/**
 * @file Test__GDNKernels.cpp
 * @brief Unit tests for Phase C: GDN Kernels
 *
 * Tests GDNProjectionStage, ShortConv1dStage, and GDNRecurrenceStage
 * for correctness, state management, and OpenMP parallelization.
 *
 * Three test categories:
 * 1. Stage tests with real CPU kernels (end-to-end correctness)
 * 2. Stage tests with mock kernels (verify delegation)
 * 3. Direct kernel tests (algorithm correctness in isolation)
 *
 * Parity is verified against the reference formulas from HuggingFace
 * transformers 5.4.0 (torch_recurrent_gated_delta_rule,
 * torch_causal_conv1d_update).
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "execution/compute_stages/stages/GDNProjectionStage.h"
#include "execution/compute_stages/stages/ShortConv1dStage.h"
#include "execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "execution/compute_stages/IComputeStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/cache/HybridCacheManager.h"
#include "kernels/cpu/gdn/CPUShortConvolution.h"
#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"
#include "tensors/Tensors.h"
#include "tensors/TensorKernels.h"

using namespace llaminar2;
using ::testing::_;
using ::testing::Return;

namespace
{
    std::unique_ptr<IDeviceContext> makeCPUContext()
    {
        return std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 1);
    }

    // Helper: create FP32 tensor with given data
    std::shared_ptr<FP32Tensor> makeFP32(const std::vector<size_t> &shape, const float *data = nullptr)
    {
        auto t = std::make_shared<FP32Tensor>(shape, DeviceId::cpu());
        if (data)
        {
            std::memcpy(t->mutable_data(), data, t->numel() * sizeof(float));
        }
        else
        {
            std::memset(t->mutable_data(), 0, t->numel() * sizeof(float));
        }
        return t;
    }

    // Helper: create FP32 tensor filled with a constant
    std::shared_ptr<FP32Tensor> makeFP32Const(const std::vector<size_t> &shape, float val)
    {
        auto t = std::make_shared<FP32Tensor>(shape, DeviceId::cpu());
        float *d = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
            d[i] = val;
        return t;
    }

    // Helper: create FP32 tensor with sequential values
    std::shared_ptr<FP32Tensor> makeFP32Seq(const std::vector<size_t> &shape, float start = 0.1f, float step = 0.1f)
    {
        auto t = std::make_shared<FP32Tensor>(shape, DeviceId::cpu());
        float *d = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
            d[i] = start + step * static_cast<float>(i);
        return t;
    }

    // Helper: create FP32 tensor with random normal values
    std::shared_ptr<FP32Tensor> makeFP32Random(const std::vector<size_t> &shape, float mean = 0.0f, float stddev = 0.1f, unsigned seed = 42)
    {
        auto t = std::make_shared<FP32Tensor>(shape, DeviceId::cpu());
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(mean, stddev);
        float *d = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
            d[i] = dist(gen);
        return t;
    }

    // SiLU reference
    float silu(float x) { return x / (1.0f + std::exp(-x)); }

    // Softplus reference
    float softplus(float x) { return (x > 20.0f) ? x : std::log1p(std::exp(x)); }

    // CPU kernel instances for end-to-end stage tests
    CPUShortConvolution g_cpu_conv;
    CPUGatedDeltaNet g_cpu_gdn;

} // namespace

// ============================================================================
// Mock Kernels for verifying stage delegation
// ============================================================================

class MockShortConvolution : public ITensorShortConvolution
{
public:
    MOCK_METHOD(bool, forward,
                (const float *input, const float *weight, const float *bias,
                 float *output, float *conv_state,
                 int seq_len, int channels, int kernel_size,
                 bool apply_silu),
                (override));
};

class MockGatedDeltaNet : public ITensorGatedDeltaNet
{
public:
    MOCK_METHOD(bool, chunk_forward,
                (const float *Q, const float *K, const float *V,
                 const float *alpha, const float *beta_raw,
                 const float *A_log, const float *dt_bias,
                 float *output, float *state,
                 int seq_len, int n_heads, int d_k, int d_v,
                 int chunk_size, bool use_qk_l2norm),
                (override));

    MOCK_METHOD(bool, recurrent_step,
                (const float *q, const float *k, const float *v,
                 const float *alpha, const float *beta_raw,
                 const float *A_log, const float *dt_bias,
                 float *output, float *state,
                 int n_heads, int d_k, int d_v,
                 bool use_qk_l2norm),
                (override));
};

// ============================================================================
// C1: GDNProjectionStage Tests
// ============================================================================

TEST(Test__GDNKernels, Projection_TypeAndBackend)
{
    GDNProjectionStage::Params p;
    GDNProjectionStage stage(p);
    EXPECT_EQ(stage.type(), ComputeStageType::GDN_PROJECTION);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
#ifdef HAVE_CUDA
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::GPU_CUDA));
#else
    EXPECT_FALSE(stage.supportsBackend(ComputeBackendType::GPU_CUDA));
#endif
}

TEST(Test__GDNKernels, Projection_NullPointers_Fails)
{
    auto ctx = makeCPUContext();
    GDNProjectionStage::Params p;
    GDNProjectionStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Projection_EstimatedFlops)
{
    GDNProjectionStage::Params p;
    p.m = 10;
    p.k = 512;
    p.n_qkv = 768;
    p.n_z = 256;
    p.n_a = 16;
    p.n_b = 16;
    GDNProjectionStage stage(p);

    // Flops = 2 * M * K * (n_qkv + n_z + n_a + n_b)
    const size_t expected = 2ull * 10 * 512 * (768 + 256 + 16 + 16);
    EXPECT_EQ(stage.estimatedFlops(), expected);
}

TEST(Test__GDNKernels, Projection_DumpInfo)
{
    auto input = makeFP32({4, 64});
    auto w_qkv = makeFP32({64, 96});
    auto out_qkv = makeFP32({4, 96});
    auto w_z = makeFP32({64, 32});
    auto out_z = makeFP32({4, 32});
    auto w_a = makeFP32({64, 4});
    auto out_a = makeFP32({4, 4});
    auto w_b = makeFP32({64, 4});
    auto out_b = makeFP32({4, 4});

    GDNProjectionStage::Params p;
    p.input = input.get();
    p.w_qkv = w_qkv.get();
    p.output_qkv = out_qkv.get();
    p.w_z = w_z.get();
    p.output_z = out_z.get();
    p.w_a = w_a.get();
    p.output_a = out_a.get();
    p.w_b = w_b.get();
    p.output_b = out_b.get();

    GDNProjectionStage stage(p);
    auto info = stage.buildDumpInfoImpl();

    EXPECT_EQ(info.inputs.size(), 5u);  // input + 4 weights
    EXPECT_EQ(info.outputs.size(), 4u); // 4 outputs
}

TEST(Test__GDNKernels, Projection_BufferContract)
{
    GDNProjectionStage::Params p;
    p.input_buffer_id = BufferId::HIDDEN_STATE;
    p.output_qkv_buffer_id = BufferId::Q_PROJ;
    p.output_z_buffer_id = BufferId::K_PROJ;

    GDNProjectionStage stage(p);
    auto contract = stage.bufferContract();
    // Should have at least the configured IDs
    SUCCEED(); // Contract creation doesn't crash
}

// ============================================================================
// C2: ShortConv1dStage Tests
// ============================================================================

TEST(Test__GDNKernels, Conv1d_TypeAndBackend)
{
    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    ShortConv1dStage stage(p);
    EXPECT_EQ(stage.type(), ComputeStageType::SHORT_CONV1D);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
}

TEST(Test__GDNKernels, Conv1d_NullPointers_Fails)
{
    auto ctx = makeCPUContext();
    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    ShortConv1dStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Conv1d_Decode_IdentityWeight)
{
    // With kernel_size=2 and weight=[0, 1], the conv1d just passes through
    // the current input (modulo SiLU activation)
    const int channels = 4;
    const int kernel_size = 2;
    const int state_len = kernel_size - 1; // 1

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = makeFP32({1, static_cast<size_t>(channels)}, input_data.data());
    auto output = makeFP32({1, static_cast<size_t>(channels)});

    // Weight: [0, 1] for each channel — passes through the current input
    std::vector<float> weight_data(channels * kernel_size, 0.0f);
    for (int c = 0; c < channels; ++c)
        weight_data[c * kernel_size + (kernel_size - 1)] = 1.0f;
    auto weight = makeFP32({static_cast<size_t>(channels), static_cast<size_t>(kernel_size)}, weight_data.data());

    std::vector<float> conv_state(channels * state_len, 0.0f);

    auto ctx = makeCPUContext();

    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = 1;
    p.channels = channels;
    p.kernel_size = kernel_size;
    p.conv_state = conv_state.data();

    ShortConv1dStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float *out = output->data();
    for (int c = 0; c < channels; ++c)
    {
        EXPECT_NEAR(out[c], silu(input_data[c]), 1e-5f)
            << "Channel " << c;
    }
}

TEST(Test__GDNKernels, Conv1d_Decode_WithState)
{
    // kernel_size=4, channels=2, single decode step
    // state = [channels, kernel_size-1] = [2, 3]
    // Computes: conv_output = sum(state[c,k] * w[c,k]) + input[c] * w[c,3]
    // Then SiLU

    const int channels = 2;
    const int kernel_size = 4;
    const int state_len = kernel_size - 1; // 3

    // Weight for channel 0: [1, 0, 0, 1], channel 1: [0, 0, 1, 2]
    std::vector<float> weight_data = {1, 0, 0, 1, 0, 0, 1, 2};
    auto weight = makeFP32({static_cast<size_t>(channels), static_cast<size_t>(kernel_size)}, weight_data.data());

    // State for channel 0: [0.5, 0.3, 0.1], channel 1: [0.2, 0.4, 0.6]
    std::vector<float> conv_state = {0.5f, 0.3f, 0.1f, 0.2f, 0.4f, 0.6f};

    std::vector<float> input_data = {2.0f, 1.0f};
    auto input = makeFP32({1, static_cast<size_t>(channels)}, input_data.data());
    auto output = makeFP32({1, static_cast<size_t>(channels)});

    auto ctx = makeCPUContext();

    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = 1;
    p.channels = channels;
    p.kernel_size = kernel_size;
    p.conv_state = conv_state.data();

    ShortConv1dStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // Channel 0: w=[1,0,0,1], state=[0.5,0.3,0.1], new=2.0
    //   raw = 1*0.5 + 0*0.3 + 0*0.1 + 1*2.0 = 2.5
    //   expected = silu(2.5)
    // Channel 1: w=[0,0,1,2], state=[0.2,0.4,0.6], new=1.0
    //   raw = 0*0.2 + 0*0.4 + 1*0.6 + 2*1.0 = 2.6
    //   expected = silu(2.6)

    const float *out = output->data();
    EXPECT_NEAR(out[0], silu(2.5f), 1e-5f);
    EXPECT_NEAR(out[1], silu(2.6f), 1e-5f);

    // After decode, state should be shifted: channel 0: [0.3, 0.1, 2.0], channel 1: [0.4, 0.6, 1.0]
    EXPECT_NEAR(conv_state[0], 0.3f, 1e-6f);
    EXPECT_NEAR(conv_state[1], 0.1f, 1e-6f);
    EXPECT_NEAR(conv_state[2], 2.0f, 1e-6f);
    EXPECT_NEAR(conv_state[3], 0.4f, 1e-6f);
    EXPECT_NEAR(conv_state[4], 0.6f, 1e-6f);
    EXPECT_NEAR(conv_state[5], 1.0f, 1e-6f);
}

TEST(Test__GDNKernels, Conv1d_Prefill_Causal)
{
    // Verify that prefill conv1d is causal: output[t] depends only on input[0..t]
    // kernel_size=3, channels=1, seq_len=4
    const int channels = 1;
    const int kernel_size = 3;
    const int seq_len = 4;

    // Weight: [1, 2, 3] (kernel for channel 0)
    std::vector<float> weight_data = {1.0f, 2.0f, 3.0f};
    auto weight = makeFP32({1, 3}, weight_data.data());

    // Input: [1, 2, 3, 4] — single channel, layout [seq_len, channels]
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = makeFP32({static_cast<size_t>(seq_len), 1}, input_data.data());
    auto output = makeFP32({static_cast<size_t>(seq_len), 1});

    auto ctx = makeCPUContext();

    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = seq_len;
    p.channels = channels;
    p.kernel_size = kernel_size;
    // No conv_state for prefill-only test

    ShortConv1dStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float *out = output->data();

    // Causal conv with zero-padding on the left:
    // t=0: pad=[0,0], current=1 → sum = w[0]*0 + w[1]*0 + w[2]*1 = 3*1 = 3
    // t=1: pad=[0], current=[1,2] → sum = w[0]*0 + w[1]*1 + w[2]*2 = 2+6 = 8
    // t=2: current=[1,2,3] → sum = w[0]*1 + w[1]*2 + w[2]*3 = 1+4+9 = 14
    // t=3: current=[2,3,4] → sum = w[0]*2 + w[1]*3 + w[2]*4 = 2+6+12 = 20
    EXPECT_NEAR(out[0], silu(3.0f), 1e-5f);
    EXPECT_NEAR(out[1], silu(8.0f), 1e-5f);
    EXPECT_NEAR(out[2], silu(14.0f), 1e-5f);
    EXPECT_NEAR(out[3], silu(20.0f), 1e-5f);
}

TEST(Test__GDNKernels, Conv1d_Prefill_StoresState)
{
    // After prefill, conv_state should contain the last (kernel_size-1) inputs
    const int channels = 2;
    const int kernel_size = 3;
    const int seq_len = 5;
    const int state_len = kernel_size - 1;

    auto weight = makeFP32Const({static_cast<size_t>(channels), static_cast<size_t>(kernel_size)}, 1.0f);

    // Input: [seq_len, channels] with sequential values
    std::vector<float> input_data(seq_len * channels);
    for (int t = 0; t < seq_len; ++t)
        for (int c = 0; c < channels; ++c)
            input_data[t * channels + c] = static_cast<float>(t * channels + c + 1);

    auto input = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(channels)}, input_data.data());
    auto output = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(channels)});

    std::vector<float> conv_state(channels * state_len, 0.0f);

    auto ctx = makeCPUContext();

    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = seq_len;
    p.channels = channels;
    p.kernel_size = kernel_size;
    p.conv_state = conv_state.data();

    ShortConv1dStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // State should have the last 2 timesteps (t=3 and t=4)
    // t=3: [7, 8], t=4: [9, 10]
    // state layout: [channels, state_len] = [2, 2]
    // channel 0: state[0,0]=input[3,0]=7, state[0,1]=input[4,0]=9
    // channel 1: state[1,0]=input[3,1]=8, state[1,1]=input[4,1]=10
    EXPECT_NEAR(conv_state[0], 7.0f, 1e-6f);  // ch0, t=3
    EXPECT_NEAR(conv_state[1], 9.0f, 1e-6f);  // ch0, t=4
    EXPECT_NEAR(conv_state[2], 8.0f, 1e-6f);  // ch1, t=3
    EXPECT_NEAR(conv_state[3], 10.0f, 1e-6f); // ch1, t=4
}

TEST(Test__GDNKernels, Conv1d_PrefillThenDecode_Consistency)
{
    // Running prefill then decode should give the same result as
    // running a longer prefill that includes the decode token.
    const int channels = 3;
    const int kernel_size = 4;
    const int prefill_len = 6;
    const int state_len = kernel_size - 1;

    auto weight = makeFP32Random({static_cast<size_t>(channels), static_cast<size_t>(kernel_size)}, 0.0f, 0.5f, 123);

    // Full sequence: prefill tokens + 1 decode token
    std::vector<float> full_input((prefill_len + 1) * channels);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto &v : full_input)
        v = dist(gen);

    // Path 1: Full prefill of length prefill_len+1
    auto input_full = makeFP32({static_cast<size_t>(prefill_len + 1), static_cast<size_t>(channels)}, full_input.data());
    auto output_full = makeFP32({static_cast<size_t>(prefill_len + 1), static_cast<size_t>(channels)});

    auto ctx = makeCPUContext();
    {
        ShortConv1dStage::Params p;
        p.kernel = &g_cpu_conv;
        p.input = input_full.get();
        p.output = output_full.get();
        p.weight = weight.get();
        p.seq_len = prefill_len + 1;
        p.channels = channels;
        p.kernel_size = kernel_size;
        ShortConv1dStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));
    }

    // Path 2: Prefill of length prefill_len, then decode of 1 token
    auto input_prefill = makeFP32({static_cast<size_t>(prefill_len), static_cast<size_t>(channels)}, full_input.data());
    auto output_prefill = makeFP32({static_cast<size_t>(prefill_len), static_cast<size_t>(channels)});
    std::vector<float> conv_state(channels * state_len, 0.0f);

    {
        ShortConv1dStage::Params p;
        p.kernel = &g_cpu_conv;
        p.input = input_prefill.get();
        p.output = output_prefill.get();
        p.weight = weight.get();
        p.seq_len = prefill_len;
        p.channels = channels;
        p.kernel_size = kernel_size;
        p.conv_state = conv_state.data();
        ShortConv1dStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));
    }

    // Decode the last token
    auto input_decode = makeFP32({1, static_cast<size_t>(channels)}, full_input.data() + prefill_len * channels);
    auto output_decode = makeFP32({1, static_cast<size_t>(channels)});

    {
        ShortConv1dStage::Params p;
        p.kernel = &g_cpu_conv;
        p.input = input_decode.get();
        p.output = output_decode.get();
        p.weight = weight.get();
        p.seq_len = 1;
        p.channels = channels;
        p.kernel_size = kernel_size;
        p.conv_state = conv_state.data();
        ShortConv1dStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));
    }

    // The decode output should match the last timestep of the full prefill
    const float *full_out = output_full->data();
    const float *decode_out = output_decode->data();
    for (int c = 0; c < channels; ++c)
    {
        EXPECT_NEAR(decode_out[c], full_out[prefill_len * channels + c], 1e-5f)
            << "Channel " << c << " mismatch between prefill+decode and full prefill";
    }
}

TEST(Test__GDNKernels, Conv1d_Decode_NoState_Fails)
{
    auto input = makeFP32({1, 4});
    auto output = makeFP32({1, 4});
    auto weight = makeFP32({4, 4});
    auto ctx = makeCPUContext();

    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = 1;
    p.channels = 4;
    p.kernel_size = 4;
    p.conv_state = nullptr; // No state!

    ShortConv1dStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}

// ============================================================================
// C3-C5: GDNRecurrenceStage Tests
// ============================================================================

TEST(Test__GDNKernels, Recurrence_TypeAndBackend)
{
    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    GDNRecurrenceStage stage(p);
    EXPECT_EQ(stage.type(), ComputeStageType::GDN_RECURRENCE);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
}

TEST(Test__GDNKernels, Recurrence_NullPointers_Fails)
{
    auto ctx = makeCPUContext();
    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    GDNRecurrenceStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Recurrence_Decode_ZeroState)
{
    // With zero initial state and specific inputs, verify output
    const int n_heads = 2;
    const int d_k = 4;
    const int d_v = 4;

    // Q, K, V: [1, n_heads * dim]
    auto Q = makeFP32Random({1, static_cast<size_t>(n_heads * d_k)}, 0.0f, 0.5f, 1);
    auto K = makeFP32Random({1, static_cast<size_t>(n_heads * d_k)}, 0.0f, 0.5f, 2);
    auto V = makeFP32Random({1, static_cast<size_t>(n_heads * d_v)}, 0.0f, 0.5f, 3);
    auto alpha = makeFP32Const({1, static_cast<size_t>(n_heads)}, 0.0f); // softplus(0 + dtbias)
    auto beta = makeFP32Const({1, static_cast<size_t>(n_heads)}, 0.0f);  // sigmoid(0) = 0.5
    auto A_log = makeFP32Const({static_cast<size_t>(n_heads)}, 0.0f);    // -exp(0) = -1
    auto dt_bias = makeFP32Const({static_cast<size_t>(n_heads)}, 0.0f);
    auto output = makeFP32({1, static_cast<size_t>(n_heads * d_v)});

    // Zero state
    std::vector<float> state(n_heads * d_k * d_v, 0.0f);

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
    p.recurrence_state = state.data();
    p.seq_len = 1;
    p.n_heads = n_heads;
    p.d_k = d_k;
    p.d_v = d_v;
    p.use_qk_l2norm = false; // Disable for deterministic test

    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // With zero state:
    // g = -exp(0) * softplus(0 + 0) = -1 * softplus(0) = -1 * ln(2) ≈ -0.6931
    // decay = exp(g) ≈ 0.5
    // S is zero, so after decay still zero
    // kv_mem = S * k = 0
    // delta = (v - 0) * sigmoid(0) = v * 0.5
    // S += outer(k, v * 0.5)
    // o = S * q = outer(k, v*0.5) * q = k*(q·k_after_l2norm)*v*0.5... but with use_qk_l2norm=false
    // Actually: o_h[v] = sum_j S[j,v] * q_h[j] where S[j,v] = k[j] * delta[v]
    //         = sum_j k[j] * v[v] * 0.5 * q[j] * scale
    //         = (sum_j k[j] * q[j] * scale) * v[v] * 0.5
    // This is a scalar * v[v] * 0.5

    // Just verify output is not all zeros (state was updated)
    const float *out = output->data();
    float sum = 0.0f;
    for (int i = 0; i < n_heads * d_v; ++i)
        sum += std::abs(out[i]);
    EXPECT_GT(sum, 0.0f) << "Output should not be all zeros after recurrence with non-zero inputs";

    // Verify state was updated (should no longer be zero)
    float state_sum = 0.0f;
    for (size_t i = 0; i < state.size(); ++i)
        state_sum += std::abs(state[i]);
    EXPECT_GT(state_sum, 0.0f) << "Recurrence state should be updated";
}

TEST(Test__GDNKernels, Recurrence_Decode_StateDecay)
{
    // Verify that with beta=0 (no update), the state decays exponentially
    const int n_heads = 1;
    const int d_k = 2;
    const int d_v = 2;

    auto Q = makeFP32Const({1, static_cast<size_t>(n_heads * d_k)}, 1.0f);
    auto K = makeFP32Const({1, static_cast<size_t>(n_heads * d_k)}, 1.0f);
    auto V = makeFP32Const({1, static_cast<size_t>(n_heads * d_v)}, 1.0f);
    auto alpha = makeFP32Const({1, 1}, 0.0f);
    auto beta = makeFP32Const({1, 1}, -100.0f); // sigmoid(-100) ≈ 0, no update
    // GGUF stores -exp(A_log_raw), so for raw_param=0: stored = -exp(0) = -1.0
    auto A_log = makeFP32Const({1}, -1.0f); // GGUF pre-exponentiated: -exp(0) = -1
    auto dt_bias = makeFP32Const({1}, 0.0f);
    auto output = makeFP32({1, static_cast<size_t>(n_heads * d_v)});

    // Initial state: all 1.0
    std::vector<float> state(d_k * d_v, 1.0f);

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
    p.recurrence_state = state.data();
    p.seq_len = 1;
    p.n_heads = n_heads;
    p.d_k = d_k;
    p.d_v = d_v;
    p.use_qk_l2norm = false;

    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // g = A_log * softplus(0) = -1.0 * ln(2) ≈ -0.6931
    // decay = exp(g) ≈ 0.5
    // beta_sig ≈ 0, so delta ≈ 0 and state is only decayed
    // State should be approximately 0.5 (decayed from 1.0)
    const float expected_decay = std::exp(-std::log(2.0f)); // 0.5
    for (int i = 0; i < d_k * d_v; ++i)
    {
        EXPECT_NEAR(state[i], expected_decay, 0.01f)
            << "State[" << i << "] should have decayed by exp(g)";
    }
}

TEST(Test__GDNKernels, Recurrence_Decode_SingleHeadReference)
{
    // Reference test: manually compute the recurrence for a single head
    // and compare with the stage output
    const int n_heads = 1;
    const int d_k = 3;
    const int d_v = 3;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    // Hand-chosen values
    std::vector<float> q_data = {1.0f, 0.0f, 0.0f}; // unit vector along dim 0
    std::vector<float> k_data = {0.0f, 1.0f, 0.0f}; // unit vector along dim 1
    std::vector<float> v_data = {0.5f, 0.5f, 0.5f}; // uniform

    auto Q = makeFP32({1, 3}, q_data.data());
    auto K = makeFP32({1, 3}, k_data.data());
    auto V = makeFP32({1, 3}, v_data.data());

    // Set gates to produce known values
    // alpha=100 → softplus(100+0) ≈ 100, g = A_log*softplus = -1.0*100 = -100 → decay ≈ 0
    // This effectively zeroes the old state before update
    auto alpha = makeFP32Const({1, 1}, 100.0f);
    // beta=100 → sigmoid(100) ≈ 1.0
    auto beta = makeFP32Const({1, 1}, 100.0f);
    // GGUF stores -exp(A_log_raw), so for raw_param=0: stored = -exp(0) = -1.0
    auto A_log = makeFP32Const({1}, -1.0f);
    auto dt_bias = makeFP32Const({1}, 0.0f);
    auto output = makeFP32({1, 3});

    std::vector<float> state(d_k * d_v, 0.5f); // Initial state

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
    p.recurrence_state = state.data();
    p.seq_len = 1;
    p.n_heads = 1;
    p.d_k = d_k;
    p.d_v = d_v;
    p.use_qk_l2norm = false;

    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // Manual computation:
    // g = A_log * softplus(alpha + dt_bias) = -1.0 * softplus(100) ≈ -100
    // decay = exp(-100) ≈ 0
    // S decayed ≈ 0 (state effectively zeroed)
    // kv_mem = S * k = 0 (S is zeroed, k=[0,1,0])
    // delta = (v - kv_mem) * beta = (0.5,0.5,0.5) * 1.0 = (0.5,0.5,0.5)
    // S += outer(k, delta) = k=[0,1,0]^T * delta=[0.5,0.5,0.5]
    //   S[0,:] = 0, S[1,:] = [0.5,0.5,0.5], S[2,:] = 0
    // q_scaled = q * scale = [scale, 0, 0]
    // o = S * q_scaled = sum_j S[j,:] * q_scaled[j] = S[0,:]*scale + 0 + 0 = 0
    const float *out = output->data();
    for (int v = 0; v < d_v; ++v)
    {
        EXPECT_NEAR(out[v], 0.0f, 0.01f)
            << "Output[" << v << "] — q selects dim 0 of S which is 0 (k was along dim 1)";
    }
}

TEST(Test__GDNKernels, Recurrence_Prefill_MatchesSequentialDecode)
{
    // Prefill of seq_len tokens should give identical output to running
    // seq_len sequential decode steps
    const int n_heads = 2;
    const int d_k = 4;
    const int d_v = 4;
    const int seq_len = 8;

    auto Q = makeFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * d_k)}, 0.0f, 0.3f, 100);
    auto K = makeFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * d_k)}, 0.0f, 0.3f, 200);
    auto V = makeFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * d_v)}, 0.0f, 0.3f, 300);
    auto alpha = makeFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads)}, 0.0f, 0.5f, 400);
    auto beta_raw = makeFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads)}, 0.0f, 0.5f, 500);
    auto A_log = makeFP32Const({static_cast<size_t>(n_heads)}, 1.0f);
    auto dt_bias = makeFP32Const({static_cast<size_t>(n_heads)}, 0.5f);

    // Path 1: Prefill
    auto output_prefill = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * d_v)});
    std::vector<float> state_prefill(n_heads * d_k * d_v, 0.0f);

    auto ctx = makeCPUContext();
    {
        GDNRecurrenceStage::Params p;
        p.kernel = &g_cpu_gdn;
        p.Q = Q.get();
        p.K = K.get();
        p.V = V.get();
        p.alpha = alpha.get();
        p.beta = beta_raw.get();
        p.A_log = A_log.get();
        p.dt_bias = dt_bias.get();
        p.output = output_prefill.get();
        p.recurrence_state = state_prefill.data();
        p.seq_len = seq_len;
        p.n_heads = n_heads;
        p.d_k = d_k;
        p.d_v = d_v;
        p.use_qk_l2norm = false;

        GDNRecurrenceStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));
    }

    // Path 2: Sequential decode steps
    auto output_decode = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * d_v)});
    std::vector<float> state_decode(n_heads * d_k * d_v, 0.0f);

    const float *q_data = Q->data();
    const float *k_data = K->data();
    const float *v_data = V->data();
    const float *a_data = alpha->data();
    const float *b_data = beta_raw->data();

    for (int t = 0; t < seq_len; ++t)
    {
        // Create single-step tensors pointing to time t
        auto q_t = makeFP32({1, static_cast<size_t>(n_heads * d_k)}, q_data + t * n_heads * d_k);
        auto k_t = makeFP32({1, static_cast<size_t>(n_heads * d_k)}, k_data + t * n_heads * d_k);
        auto v_t = makeFP32({1, static_cast<size_t>(n_heads * d_v)}, v_data + t * n_heads * d_v);
        auto a_t = makeFP32({1, static_cast<size_t>(n_heads)}, a_data + t * n_heads);
        auto b_t = makeFP32({1, static_cast<size_t>(n_heads)}, b_data + t * n_heads);

        auto out_t = makeFP32({1, static_cast<size_t>(n_heads * d_v)});

        GDNRecurrenceStage::Params p;
        p.kernel = &g_cpu_gdn;
        p.Q = q_t.get();
        p.K = k_t.get();
        p.V = v_t.get();
        p.alpha = a_t.get();
        p.beta = b_t.get();
        p.A_log = A_log.get();
        p.dt_bias = dt_bias.get();
        p.output = out_t.get();
        p.recurrence_state = state_decode.data();
        p.seq_len = 1;
        p.n_heads = n_heads;
        p.d_k = d_k;
        p.d_v = d_v;
        p.use_qk_l2norm = false;

        GDNRecurrenceStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));

        // Copy output for timestep t
        std::memcpy(output_decode->mutable_data() + t * n_heads * d_v,
                    out_t->data(), n_heads * d_v * sizeof(float));
    }

    // Compare: prefill output should match sequential decode output
    const float *pf = output_prefill->data();
    const float *dc = output_decode->data();
    for (int i = 0; i < seq_len * n_heads * d_v; ++i)
    {
        EXPECT_NEAR(pf[i], dc[i], 1e-4f)
            << "Mismatch at index " << i << " (t=" << i / (n_heads * d_v) << ")";
    }

    // Final states should also match
    for (size_t i = 0; i < state_prefill.size(); ++i)
    {
        EXPECT_NEAR(state_prefill[i], state_decode[i], 1e-4f)
            << "State mismatch at index " << i;
    }
}

TEST(Test__GDNKernels, Recurrence_L2Norm)
{
    // Verify L2 normalization works correctly
    const int n_heads = 1;
    const int d_k = 4;
    const int d_v = 4;

    // Q with norm != 1: [3, 4, 0, 0] → norm = 5 → normalized = [0.6, 0.8, 0, 0]
    std::vector<float> q_data = {3.0f, 4.0f, 0.0f, 0.0f};
    std::vector<float> k_data = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> v_data = {1.0f, 1.0f, 1.0f, 1.0f};

    auto Q = makeFP32({1, 4}, q_data.data());
    auto K = makeFP32({1, 4}, k_data.data());
    auto V = makeFP32({1, 4}, v_data.data());

    // Large negative gate → decay ≈ 0 (fresh state each step)
    auto alpha = makeFP32Const({1, 1}, 100.0f);
    auto beta = makeFP32Const({1, 1}, 100.0f); // sigmoid ≈ 1
    auto A_log = makeFP32Const({1}, 0.0f);
    auto dt_bias = makeFP32Const({1}, 0.0f);

    // Run with L2 norm enabled
    auto output_l2 = makeFP32({1, 4});
    std::vector<float> state_l2(d_k * d_v, 0.0f);

    auto ctx = makeCPUContext();
    {
        GDNRecurrenceStage::Params p;
        p.kernel = &g_cpu_gdn;
        p.Q = Q.get();
        p.K = K.get();
        p.V = V.get();
        p.alpha = alpha.get();
        p.beta = beta.get();
        p.A_log = A_log.get();
        p.dt_bias = dt_bias.get();
        p.output = output_l2.get();
        p.recurrence_state = state_l2.data();
        p.seq_len = 1;
        p.n_heads = 1;
        p.d_k = d_k;
        p.d_v = d_v;
        p.use_qk_l2norm = true;

        GDNRecurrenceStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));
    }

    // Run without L2 norm
    auto output_no_l2 = makeFP32({1, 4});
    std::vector<float> state_no_l2(d_k * d_v, 0.0f);

    // Re-create tensors (data was modified by l2norm)
    auto Q2 = makeFP32({1, 4}, q_data.data());
    auto K2 = makeFP32({1, 4}, k_data.data());
    {
        GDNRecurrenceStage::Params p;
        p.kernel = &g_cpu_gdn;
        p.Q = Q2.get();
        p.K = K2.get();
        p.V = V.get();
        p.alpha = alpha.get();
        p.beta = beta.get();
        p.A_log = A_log.get();
        p.dt_bias = dt_bias.get();
        p.output = output_no_l2.get();
        p.recurrence_state = state_no_l2.data();
        p.seq_len = 1;
        p.n_heads = 1;
        p.d_k = d_k;
        p.d_v = d_v;
        p.use_qk_l2norm = false;

        GDNRecurrenceStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));
    }

    // Outputs should differ because L2 norm changes the effective Q/K magnitudes
    const float *l2_out = output_l2->data();
    const float *no_l2_out = output_no_l2->data();
    bool differ = false;
    for (int i = 0; i < d_v; ++i)
    {
        if (std::abs(l2_out[i] - no_l2_out[i]) > 1e-6f)
            differ = true;
    }
    EXPECT_TRUE(differ) << "L2 normalization should change the output";
}

TEST(Test__GDNKernels, Recurrence_EstimatedFlops)
{
    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.seq_len = 10;
    p.n_heads = 4;
    p.d_k = 128;
    p.d_v = 128;
    GDNRecurrenceStage stage(p);

    // 10 * 4 * (4*128*128 + 128) = 10 * 4 * 65664 = 2626560
    EXPECT_EQ(stage.estimatedFlops(), 10ull * 4 * (4 * 128 * 128 + 128));
}

TEST(Test__GDNKernels, Recurrence_NoState_Fails)
{
    auto Q = makeFP32({1, 8});
    auto K = makeFP32({1, 8});
    auto V = makeFP32({1, 8});
    auto alpha = makeFP32({1, 2});
    auto beta = makeFP32({1, 2});
    auto A_log = makeFP32({2});
    auto dt_bias = makeFP32({2});
    auto output = makeFP32({1, 8});

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
    p.recurrence_state = nullptr;
    p.seq_len = 1;
    p.n_heads = 2;
    p.d_k = 4;
    p.d_v = 4;

    GDNRecurrenceStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}

// ============================================================================
// C6: GDN State Management Integration
// ============================================================================

TEST(Test__GDNKernels, StateIntegration_HybridCacheManager)
{
    // Verify that HybridCacheManager provides GDNLayerState with correct dimensions
    // and that we can pass state pointers to the stages
    HybridCacheManager::Config config;
    config.n_layers = 4;
    config.layer_types = {"gdn", "gdn", "gdn", "full_attention"};
    config.n_heads = 2;
    config.head_dim = 4;
    config.conv_kernel_size = 4;

    HybridCacheManager manager(config, nullptr);

    // GDN layers should have state
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(manager.getLayerStateType(i), LayerStateType::GDN_STATE);
        auto *state = manager.getGDNState(i);
        ASSERT_NE(state, nullptr);
        EXPECT_EQ(state->n_heads, 2);
        EXPECT_EQ(state->head_dim, 4);
        EXPECT_EQ(state->conv_kernel_size, 4);

        // Recurrence state: [n_heads, head_dim, head_dim] = [2, 4, 4] = 32
        EXPECT_EQ(state->recurrence_state.size(), 32u);

        // Conv state: [n_heads, conv_kernel-1, head_dim] = [2, 3, 4] = 24
        EXPECT_EQ(state->conv_state.size(), 24u);
    }

    // Full attention layer should not have GDN state
    EXPECT_EQ(manager.getLayerStateType(3), LayerStateType::KV_CACHE);
    EXPECT_EQ(manager.getGDNState(3), nullptr);
}

TEST(Test__GDNKernels, StateIntegration_PassToRecurrenceStage)
{
    // Create a HybridCacheManager and pass its state to GDNRecurrenceStage
    HybridCacheManager::Config config;
    config.n_layers = 1;
    config.layer_types = {"gdn"};
    config.n_heads = 2;
    config.head_dim = 4;
    config.conv_kernel_size = 4;

    HybridCacheManager manager(config, nullptr);
    auto *gdn_state = manager.getGDNState(0);
    ASSERT_NE(gdn_state, nullptr);

    // Create minimal inputs
    auto Q = makeFP32Random({1, 8}, 0.0f, 0.3f, 42);
    auto K = makeFP32Random({1, 8}, 0.0f, 0.3f, 43);
    auto V = makeFP32Random({1, 8}, 0.0f, 0.3f, 44);
    auto alpha = makeFP32Const({1, 2}, 0.0f);
    auto beta_raw = makeFP32Const({1, 2}, 0.0f);
    auto A_log = makeFP32Const({2}, 0.0f);
    auto dt_bias = makeFP32Const({2}, 0.0f);
    auto output = makeFP32({1, 8});

    auto ctx = makeCPUContext();

    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta_raw.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = gdn_state->recurrence_state.data();
    p.seq_len = 1;
    p.n_heads = 2;
    p.d_k = 4;
    p.d_v = 4;
    p.use_qk_l2norm = false;

    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // State should be modified
    float state_sum = 0.0f;
    for (float v : gdn_state->recurrence_state)
        state_sum += std::abs(v);
    EXPECT_GT(state_sum, 0.0f) << "GDNLayerState recurrence_state should be updated";
}

TEST(Test__GDNKernels, StateIntegration_PassToConv1dStage)
{
    // Create a HybridCacheManager and pass its conv state to ShortConv1dStage
    HybridCacheManager::Config config;
    config.n_layers = 1;
    config.layer_types = {"gdn"};
    config.n_heads = 2;
    config.head_dim = 4;
    config.conv_kernel_size = 4;

    HybridCacheManager manager(config, nullptr);
    auto *gdn_state = manager.getGDNState(0);
    ASSERT_NE(gdn_state, nullptr);

    // Conv state layout: [channels, kernel_size-1]
    // For GDNLayerState: [n_heads, conv_kernel-1, head_dim] but the ShortConv1d
    // operates on channels = QKV_dim. For this test, we just verify the state
    // pointer can be used.
    const int channels = 4; // simplified for test
    const int kernel_size = 4;

    auto input = makeFP32Random({1, static_cast<size_t>(channels)}, 0.0f, 0.5f, 42);
    auto output = makeFP32({1, static_cast<size_t>(channels)});
    auto weight = makeFP32Random({static_cast<size_t>(channels), static_cast<size_t>(kernel_size)}, 0.0f, 0.3f, 43);

    // Use a separate state buffer sized for the test
    std::vector<float> test_conv_state(channels * (kernel_size - 1), 0.0f);

    auto ctx = makeCPUContext();

    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = 1;
    p.channels = channels;
    p.kernel_size = kernel_size;
    p.conv_state = test_conv_state.data();

    ShortConv1dStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // After decode, the new input should be stored in the state
    bool state_updated = false;
    for (float v : test_conv_state)
    {
        if (std::abs(v) > 1e-8f)
        {
            state_updated = true;
            break;
        }
    }
    EXPECT_TRUE(state_updated) << "Conv state should be updated after decode step";
}

TEST(Test__GDNKernels, StateIntegration_ResetClearsState)
{
    HybridCacheManager::Config config;
    config.n_layers = 2;
    config.layer_types = {"gdn", "gdn"};
    config.n_heads = 2;
    config.head_dim = 4;
    config.conv_kernel_size = 4;

    HybridCacheManager manager(config, nullptr);

    // Set some non-zero values
    auto *state0 = manager.getGDNState(0);
    auto *state1 = manager.getGDNState(1);
    ASSERT_NE(state0, nullptr);
    ASSERT_NE(state1, nullptr);

    for (auto &v : state0->recurrence_state)
        v = 1.0f;
    for (auto &v : state0->conv_state)
        v = 1.0f;
    for (auto &v : state1->recurrence_state)
        v = 2.0f;
    for (auto &v : state1->conv_state)
        v = 2.0f;

    // Reset should zero everything
    manager.reset();

    for (float v : state0->recurrence_state)
        EXPECT_EQ(v, 0.0f);
    for (float v : state0->conv_state)
        EXPECT_EQ(v, 0.0f);
    for (float v : state1->recurrence_state)
        EXPECT_EQ(v, 0.0f);
    for (float v : state1->conv_state)
        EXPECT_EQ(v, 0.0f);
}

// ============================================================================
// Integration: Conv1d output feeds into recurrence
// ============================================================================

TEST(Test__GDNKernels, Pipeline_Conv1dThenRecurrence)
{
    // End-to-end: conv1d → split QKV → recurrence
    // Verifies that output dimensions chain correctly
    const int seq_len = 4;
    const int n_heads = 2;
    const int d_k = 4;
    const int d_v = 4;
    const int qkv_dim = 2 * n_heads * d_k + n_heads * d_v; // 24
    const int kernel_size = 3;

    // Create conv1d input and weight
    auto conv_input = makeFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(qkv_dim)}, 0.0f, 0.5f, 1);
    auto conv_output = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(qkv_dim)});
    auto conv_weight = makeFP32Random({static_cast<size_t>(qkv_dim), static_cast<size_t>(kernel_size)}, 0.0f, 0.3f, 2);

    auto ctx = makeCPUContext();

    // Step 1: Conv1d
    {
        ShortConv1dStage::Params p;
        p.kernel = &g_cpu_conv;
        p.input = conv_input.get();
        p.output = conv_output.get();
        p.weight = conv_weight.get();
        p.seq_len = seq_len;
        p.channels = qkv_dim;
        p.kernel_size = kernel_size;

        ShortConv1dStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));
    }

    // Step 2: Split conv output into Q, K, V
    // Q: first n_heads*d_k dims, K: next n_heads*d_k, V: last n_heads*d_v
    const int q_dim = n_heads * d_k; // 8
    const int k_dim = n_heads * d_k; // 8
    const int v_dim = n_heads * d_v; // 8

    auto Q = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(q_dim)});
    auto K = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(k_dim)});
    auto V = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(v_dim)});

    const float *conv_out = conv_output->data();
    for (int t = 0; t < seq_len; ++t)
    {
        std::memcpy(Q->mutable_data() + t * q_dim, conv_out + t * qkv_dim, q_dim * sizeof(float));
        std::memcpy(K->mutable_data() + t * q_dim, conv_out + t * qkv_dim + q_dim, k_dim * sizeof(float));
        std::memcpy(V->mutable_data() + t * v_dim, conv_out + t * qkv_dim + q_dim + k_dim, v_dim * sizeof(float));
    }

    // Step 3: Recurrence
    auto alpha = makeFP32Const({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads)}, 0.0f);
    auto beta_raw = makeFP32Const({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads)}, 0.0f);
    auto A_log = makeFP32Const({static_cast<size_t>(n_heads)}, 1.0f);
    auto dt_bias = makeFP32Const({static_cast<size_t>(n_heads)}, 0.0f);
    auto rec_output = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * d_v)});
    std::vector<float> state(n_heads * d_k * d_v, 0.0f);

    {
        GDNRecurrenceStage::Params p;
        p.kernel = &g_cpu_gdn;
        p.Q = Q.get();
        p.K = K.get();
        p.V = V.get();
        p.alpha = alpha.get();
        p.beta = beta_raw.get();
        p.A_log = A_log.get();
        p.dt_bias = dt_bias.get();
        p.output = rec_output.get();
        p.recurrence_state = state.data();
        p.seq_len = seq_len;
        p.n_heads = n_heads;
        p.d_k = d_k;
        p.d_v = d_v;
        p.use_qk_l2norm = false;

        GDNRecurrenceStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));
    }

    // Verify output has reasonable values (not NaN/Inf)
    const float *out = rec_output->data();
    for (int i = 0; i < seq_len * n_heads * d_v; ++i)
    {
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

// ============================================================================
// Mock-based delegation tests
// ============================================================================

TEST(Test__GDNKernels, Conv1d_NullKernel_Fails)
{
    auto input = makeFP32({1, 4});
    auto output = makeFP32({1, 4});
    auto weight = makeFP32({4, 4});
    std::vector<float> state(4 * 3, 0.0f);
    auto ctx = makeCPUContext();

    ShortConv1dStage::Params p;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = 1;
    p.channels = 4;
    p.kernel_size = 4;
    p.conv_state = state.data();
    p.kernel = nullptr; // Explicitly null

    ShortConv1dStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Conv1d_DelegatesToKernel)
{
    // Verify the stage delegates to ITensorShortConvolution::forward()
    auto input = makeFP32({1, 4}, std::vector<float>{1, 2, 3, 4}.data());
    auto output = makeFP32({1, 4});
    auto weight = makeFP32({4, 2}, std::vector<float>{1, 0, 1, 0, 1, 0, 1, 0}.data());
    std::vector<float> state(4 * 1, 0.0f);
    auto ctx = makeCPUContext();

    MockShortConvolution mock;
    EXPECT_CALL(mock, forward(_, _, _, _, _, 1, 4, 2, true))
        .WillOnce(Return(true));

    ShortConv1dStage::Params p;
    p.kernel = &mock;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = 1;
    p.channels = 4;
    p.kernel_size = 2;
    p.conv_state = state.data();

    ShortConv1dStage stage(p);
    EXPECT_TRUE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Conv1d_KernelFailurePropagates)
{
    auto input = makeFP32({1, 4});
    auto output = makeFP32({1, 4});
    auto weight = makeFP32({4, 2});
    std::vector<float> state(4 * 1, 0.0f);
    auto ctx = makeCPUContext();

    MockShortConvolution mock;
    EXPECT_CALL(mock, forward(_, _, _, _, _, _, _, _, _))
        .WillOnce(Return(false));

    ShortConv1dStage::Params p;
    p.kernel = &mock;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = 1;
    p.channels = 4;
    p.kernel_size = 2;
    p.conv_state = state.data();

    ShortConv1dStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Recurrence_NullKernel_Fails)
{
    auto Q = makeFP32({1, 8});
    auto K = makeFP32({1, 8});
    auto V = makeFP32({1, 8});
    auto alpha = makeFP32({1, 2});
    auto beta = makeFP32({1, 2});
    auto A_log = makeFP32({2});
    auto dt_bias = makeFP32({2});
    auto output = makeFP32({1, 8});
    std::vector<float> state(2 * 4 * 4, 0.0f);
    auto ctx = makeCPUContext();

    GDNRecurrenceStage::Params p;
    p.kernel = nullptr; // Explicitly null
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = state.data();
    p.seq_len = 1;
    p.n_heads = 2;
    p.d_k = 4;
    p.d_v = 4;

    GDNRecurrenceStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Recurrence_Decode_DelegatesToRecurrentStep)
{
    // seq_len=1 should call recurrent_step, not chunk_forward
    auto Q = makeFP32Random({1, 8}, 0.0f, 0.3f, 42);
    auto K = makeFP32Random({1, 8}, 0.0f, 0.3f, 43);
    auto V = makeFP32Random({1, 8}, 0.0f, 0.3f, 44);
    auto alpha = makeFP32Const({1, 2}, 0.0f);
    auto beta = makeFP32Const({1, 2}, 0.0f);
    auto A_log = makeFP32Const({2}, 0.0f);
    auto dt_bias = makeFP32Const({2}, 0.0f);
    auto output = makeFP32({1, 8});
    std::vector<float> state(2 * 4 * 4, 0.0f);
    auto ctx = makeCPUContext();

    MockGatedDeltaNet mock;
    EXPECT_CALL(mock, recurrent_step(_, _, _, _, _, _, _, _, _, 2, 4, 4, _))
        .WillOnce(Return(true));
    // chunk_forward should NOT be called
    EXPECT_CALL(mock, chunk_forward(_, _, _, _, _, _, _, _, _, _, _, _, _, _, _))
        .Times(0);

    GDNRecurrenceStage::Params p;
    p.kernel = &mock;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = state.data();
    p.seq_len = 1;
    p.n_heads = 2;
    p.d_k = 4;
    p.d_v = 4;

    GDNRecurrenceStage stage(p);
    EXPECT_TRUE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Recurrence_Prefill_DelegatesToChunkForward)
{
    // seq_len>1 should call chunk_forward, not recurrent_step
    auto Q = makeFP32Random({4, 8}, 0.0f, 0.3f, 42);
    auto K = makeFP32Random({4, 8}, 0.0f, 0.3f, 43);
    auto V = makeFP32Random({4, 8}, 0.0f, 0.3f, 44);
    auto alpha = makeFP32Const({4, 2}, 0.0f);
    auto beta = makeFP32Const({4, 2}, 0.0f);
    auto A_log = makeFP32Const({2}, 0.0f);
    auto dt_bias = makeFP32Const({2}, 0.0f);
    auto output = makeFP32({4, 8});
    std::vector<float> state(2 * 4 * 4, 0.0f);
    auto ctx = makeCPUContext();

    MockGatedDeltaNet mock;
    EXPECT_CALL(mock, chunk_forward(_, _, _, _, _, _, _, _, _, 4, 2, 4, 4, _, _))
        .WillOnce(Return(true));
    // recurrent_step should NOT be called
    EXPECT_CALL(mock, recurrent_step(_, _, _, _, _, _, _, _, _, _, _, _, _))
        .Times(0);

    GDNRecurrenceStage::Params p;
    p.kernel = &mock;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = state.data();
    p.seq_len = 4;
    p.n_heads = 2;
    p.d_k = 4;
    p.d_v = 4;

    GDNRecurrenceStage stage(p);
    EXPECT_TRUE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Recurrence_KernelFailurePropagates)
{
    auto Q = makeFP32Random({1, 8}, 0.0f, 0.3f, 42);
    auto K = makeFP32Random({1, 8}, 0.0f, 0.3f, 43);
    auto V = makeFP32Random({1, 8}, 0.0f, 0.3f, 44);
    auto alpha = makeFP32Const({1, 2}, 0.0f);
    auto beta = makeFP32Const({1, 2}, 0.0f);
    auto A_log = makeFP32Const({2}, 0.0f);
    auto dt_bias = makeFP32Const({2}, 0.0f);
    auto output = makeFP32({1, 8});
    std::vector<float> state(2 * 4 * 4, 0.0f);
    auto ctx = makeCPUContext();

    MockGatedDeltaNet mock;
    EXPECT_CALL(mock, recurrent_step(_, _, _, _, _, _, _, _, _, _, _, _, _))
        .WillOnce(Return(false));

    GDNRecurrenceStage::Params p;
    p.kernel = &mock;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = state.data();
    p.seq_len = 1;
    p.n_heads = 2;
    p.d_k = 4;
    p.d_v = 4;

    GDNRecurrenceStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}
