/**
 * @file Test__GDNDynamicParamsRegression.cpp
 * @brief Regression tests for GDN stage updateDynamicParams / hasDynamicParams
 *
 * The bug: GDNRecurrenceStage and ShortConv1dStage did not override
 * updateDynamicParams(), so cached graph reuse (via updateCachedGraphParams)
 * would leave params_.seq_len at its original prefill value instead of
 * updating to the decode seq_len. This caused execute() to call
 * chunk_forward() (prefill path) instead of recurrent_step() (decode path).
 *
 * The fix: Both stages now override updateDynamicParams() to update
 * params_.seq_len, and hasDynamicParams() returns true.
 *
 * Regression for: GDNRecurrenceStage.h / ShortConv1dStage.h
 *                 updateDynamicParams() override addition
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>

#include "execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "execution/compute_stages/stages/ShortConv1dStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"
#include "kernels/cpu/gdn/CPUShortConvolution.h"
#include "tensors/Tensors.h"

using namespace llaminar2;

namespace
{
    std::unique_ptr<IDeviceContext> makeCPUContext()
    {
        return std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 1);
    }

    std::shared_ptr<FP32Tensor> makeFP32(const std::vector<size_t> &shape, float fillVal = 0.0f)
    {
        auto t = std::make_shared<FP32Tensor>(shape, DeviceId::cpu());
        float *d = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
            d[i] = fillVal;
        return t;
    }

    std::shared_ptr<FP32Tensor> makeFP32Random(const std::vector<size_t> &shape, float lo = -0.5f, float hi = 0.5f)
    {
        auto t = std::make_shared<FP32Tensor>(shape, DeviceId::cpu());
        float *d = t->mutable_data();
        float range = hi - lo;
        for (size_t i = 0; i < t->numel(); ++i)
            d[i] = lo + range * (static_cast<float>(i % 97) / 97.0f);
        return t;
    }
} // namespace

// ============================================================================
// GDNRecurrenceStage: updateDynamicParams and hasDynamicParams
// ============================================================================

TEST(Test__GDNDynamicParams, GDNRecurrenceStage_HasDynamicParams)
{
    CPUGatedDeltaNet cpu_gdn;

    // Verify the stage reports it has dynamic params
    GDNRecurrenceStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = 512;
    params.n_heads = 4;
    params.d_k = 4;
    params.d_v = 4;

    // Need minimal valid state to construct (won't execute)
    std::vector<float> state(params.n_heads * params.d_k * params.d_v, 0.0f);
    params.recurrence_state = state.data();
    params.kernel = &cpu_gdn;

    GDNRecurrenceStage stage(std::move(params));

    EXPECT_TRUE(stage.hasDynamicParams())
        << "GDNRecurrenceStage must report hasDynamicParams()=true";
}

TEST(Test__GDNDynamicParams, GDNRecurrenceStage_UpdateDynamicParams_ChangesSeqLen)
{
    CPUGatedDeltaNet cpu_gdn;

    // Build a stage with prefill seq_len, then update to decode seq_len
    const int n_heads = 4;
    const int d_k = 4;
    const int d_v = 4;
    const int prefill_seq_len = 128;
    const int decode_seq_len = 1;

    GDNRecurrenceStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = prefill_seq_len;
    params.n_heads = n_heads;
    params.n_k_heads = 0; // same as n_heads
    params.d_k = d_k;
    params.d_v = d_v;
    params.chunk_size = 64;
    params.use_qk_l2norm = true;
    params.kernel = &cpu_gdn;

    std::vector<float> state(n_heads * d_k * d_v, 0.0f);
    params.recurrence_state = state.data();

    // Minimal tensors — won't execute, just checking param updates
    auto qkv = makeFP32({1, static_cast<size_t>(n_heads * d_k)}, 0.1f);
    auto alpha = makeFP32({1, static_cast<size_t>(n_heads)}, 0.0f);
    auto beta = makeFP32({1, static_cast<size_t>(n_heads)}, 0.0f);
    auto a_log = makeFP32({static_cast<size_t>(n_heads)}, -1.0f);
    auto dt_bias = makeFP32({static_cast<size_t>(n_heads)}, 0.0f);
    auto output = makeFP32({1, static_cast<size_t>(n_heads * d_v)}, 0.0f);

    params.Q = qkv.get();
    params.K = qkv.get();
    params.V = qkv.get();
    params.alpha = alpha.get();
    params.beta = beta.get();
    params.A_log = a_log.get();
    params.dt_bias = dt_bias.get();
    params.output = output.get();

    GDNRecurrenceStage stage(std::move(params));

    // Verify initial seq_len
    ASSERT_EQ(stage.getParams().seq_len, prefill_seq_len);

    // Simulate what updateCachedGraphParams does
    stage.updateDynamicParams(/*pos_offset=*/128, /*seq_len=*/decode_seq_len);

    // After update, seq_len must be 1 (decode)
    EXPECT_EQ(stage.getParams().seq_len, decode_seq_len)
        << "updateDynamicParams must update params_.seq_len for GDNRecurrenceStage";
}

TEST(Test__GDNDynamicParams, GDNRecurrenceStage_UpdateDynamicParams_ExecutesDecodePath)
{
    CPUGatedDeltaNet cpu_gdn;

    // Full execution test: build stage with prefill seq_len=128, update to 1,
    // verify it calls recurrent_step (seq_len==1 path) successfully.
    const int n_heads = 2;
    const int d_k = 4;
    const int d_v = 4;

    GDNRecurrenceStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = 128; // <-- prefill value, will be updated
    params.n_heads = n_heads;
    params.n_k_heads = 0;
    params.d_k = d_k;
    params.d_v = d_v;
    params.chunk_size = 64;
    params.use_qk_l2norm = true;
    params.kernel = &cpu_gdn;

    std::vector<float> state(n_heads * d_k * d_v, 0.0f);
    params.recurrence_state = state.data();

    // Single-token tensors (decode shape)
    auto q = makeFP32Random({1, static_cast<size_t>(n_heads * d_k)});
    auto k = makeFP32Random({1, static_cast<size_t>(n_heads * d_k)});
    auto v = makeFP32Random({1, static_cast<size_t>(n_heads * d_v)});
    auto alpha = makeFP32({1, static_cast<size_t>(n_heads)}, 0.5f);
    auto beta = makeFP32({1, static_cast<size_t>(n_heads)}, 0.5f);
    auto a_log = makeFP32({static_cast<size_t>(n_heads)}, -1.0f);
    auto dt_bias = makeFP32({static_cast<size_t>(n_heads)}, 0.1f);
    auto output = makeFP32({1, static_cast<size_t>(n_heads * d_v)});

    params.Q = q.get();
    params.K = k.get();
    params.V = v.get();
    params.alpha = alpha.get();
    params.beta = beta.get();
    params.A_log = a_log.get();
    params.dt_bias = dt_bias.get();
    params.output = output.get();

    GDNRecurrenceStage stage(std::move(params));

    // Update to decode mode (seq_len=1)
    stage.updateDynamicParams(/*pos_offset=*/128, /*seq_len=*/1);
    ASSERT_EQ(stage.getParams().seq_len, 1);

    // Execute — should use recurrent_step, not chunk_forward
    auto ctx = makeCPUContext();
    bool ok = stage.execute(ctx.get());
    ASSERT_TRUE(ok) << "GDNRecurrenceStage must execute successfully after updateDynamicParams to seq_len=1";

    // Output should be finite and non-zero (state was updated)
    const float *out = dynamic_cast<TensorBase *>(output.get())->data();
    bool any_nonzero = false;
    for (int i = 0; i < n_heads * d_v; ++i)
    {
        EXPECT_TRUE(std::isfinite(out[i])) << "Output[" << i << "] is not finite";
        if (std::abs(out[i]) > 1e-12f)
            any_nonzero = true;
    }
    EXPECT_TRUE(any_nonzero) << "Output is all-zero after decode step (state should have been updated)";
}

// ============================================================================
// ShortConv1dStage: updateDynamicParams and hasDynamicParams
// ============================================================================

TEST(Test__GDNDynamicParams, ShortConv1dStage_HasDynamicParams)
{
    CPUShortConvolution cpu_conv;

    ShortConv1dStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = 512;
    params.channels = 32;
    params.kernel_size = 4;
    params.kernel = &cpu_conv;

    std::vector<float> conv_state(params.channels * (params.kernel_size - 1), 0.0f);
    params.conv_state = conv_state.data();
    auto input = makeFP32({1, static_cast<size_t>(params.channels)}, 0.1f);
    auto output = makeFP32({1, static_cast<size_t>(params.channels)});
    auto weight = makeFP32({static_cast<size_t>(params.channels),
                            static_cast<size_t>(params.kernel_size)},
                           0.25f);
    params.input = input.get();
    params.output = output.get();
    params.weight = weight.get();

    ShortConv1dStage stage(std::move(params));

    EXPECT_TRUE(stage.hasDynamicParams())
        << "ShortConv1dStage must report hasDynamicParams()=true";
}

TEST(Test__GDNDynamicParams, ShortConv1dStage_UpdateDynamicParams_ChangesSeqLen)
{
    CPUShortConvolution cpu_conv;

    const int channels = 16;
    const int kernel_size = 4;
    const int prefill_seq_len = 128;
    const int decode_seq_len = 1;

    ShortConv1dStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = prefill_seq_len;
    params.channels = channels;
    params.kernel_size = kernel_size;
    params.kernel = &cpu_conv;

    std::vector<float> conv_state(channels * (kernel_size - 1), 0.0f);
    params.conv_state = conv_state.data();

    auto input = makeFP32({1, static_cast<size_t>(channels)}, 0.1f);
    auto output = makeFP32({1, static_cast<size_t>(channels)});
    auto weight = makeFP32({static_cast<size_t>(channels), static_cast<size_t>(kernel_size)}, 0.25f);

    params.input = input.get();
    params.output = output.get();
    params.weight = weight.get();

    ShortConv1dStage stage(std::move(params));

    ASSERT_EQ(stage.getParams().seq_len, prefill_seq_len);

    stage.updateDynamicParams(/*pos_offset=*/128, /*seq_len=*/decode_seq_len);

    EXPECT_EQ(stage.getParams().seq_len, decode_seq_len)
        << "updateDynamicParams must update params_.seq_len for ShortConv1dStage";
}

TEST(Test__GDNDynamicParams, ShortConv1dStage_UpdateDynamicParams_ExecutesDecodePath)
{
    CPUShortConvolution cpu_conv;

    // Build stage with prefill seq_len, update to decode, execute successfully
    const int channels = 16;
    const int kernel_size = 4;

    ShortConv1dStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = 64; // <-- prefill value
    params.channels = channels;
    params.kernel_size = kernel_size;
    params.kernel = &cpu_conv;

    std::vector<float> conv_state(channels * (kernel_size - 1), 0.0f);
    params.conv_state = conv_state.data();

    // Fill conv_state with non-zero data (simulating post-prefill state)
    for (size_t i = 0; i < conv_state.size(); ++i)
        conv_state[i] = 0.1f * static_cast<float>(i % 7 + 1);

    auto input = makeFP32Random({1, static_cast<size_t>(channels)});
    auto output = makeFP32({1, static_cast<size_t>(channels)});
    auto weight = makeFP32({static_cast<size_t>(channels), static_cast<size_t>(kernel_size)}, 0.25f);

    params.input = input.get();
    params.output = output.get();
    params.weight = weight.get();

    ShortConv1dStage stage(std::move(params));

    // Update to decode
    stage.updateDynamicParams(/*pos_offset=*/64, /*seq_len=*/1);
    ASSERT_EQ(stage.getParams().seq_len, 1);

    auto ctx = makeCPUContext();
    bool ok = stage.execute(ctx.get());
    ASSERT_TRUE(ok) << "ShortConv1dStage must execute successfully after updateDynamicParams to seq_len=1";

    // Output should be finite
    const float *out = dynamic_cast<TensorBase *>(output.get())->data();
    for (int i = 0; i < channels; ++i)
    {
        EXPECT_TRUE(std::isfinite(out[i])) << "Output[" << i << "] is not finite";
    }
}
