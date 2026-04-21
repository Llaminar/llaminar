#include <gtest/gtest.h>
#include "planning/WeightMemoryEstimator.h"
#include "planning/ModelMemoryProfile.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

namespace
{

ModelMemoryProfile createSimpleProfile()
{
    ModelMemoryProfile p;
    p.architecture = "qwen2";
    p.n_layers = 2;
    p.d_model = 896;
    p.d_ff = 4864;
    p.n_heads = 14;
    p.n_kv_heads = 2;
    p.head_dim = 64;
    p.vocab_size = 151936;
    p.max_seq_len = 4096;
    p.total_native_bytes = 0;

    auto addTensor = [&](const std::string& name, size_t rows, size_t cols,
                         const std::string& quant, int layer = -1) {
        TensorSizeInfo t;
        t.name = name;
        t.elements = rows * cols;
        t.K = cols;
        t.quant_type = quant;
        // Q8_0: 34 bytes/32 elements
        if (quant == "Q8_0")
            t.native_bytes = rows * cols * 34 / 32;
        else if (quant == "F32")
            t.native_bytes = rows * cols * 4;
        else
            t.native_bytes = rows * cols;  // Simplified
        t.layer_index = layer;
        p.total_native_bytes += t.native_bytes;
        p.tensors.push_back(t);
    };

    // Non-layer tensors
    addTensor("token_embd.weight", 896, 151936, "Q8_0");
    addTensor("output.weight", 896, 151936, "Q8_0");
    addTensor("output_norm.weight", 1, 896, "F32");

    // Layer 0
    addTensor("blk.0.attn_q.weight", 896, 896, "Q8_0", 0);
    addTensor("blk.0.attn_k.weight", 128, 896, "Q8_0", 0);
    addTensor("blk.0.attn_v.weight", 128, 896, "Q8_0", 0);
    addTensor("blk.0.attn_output.weight", 896, 896, "Q8_0", 0);
    addTensor("blk.0.ffn_gate.weight", 4864, 896, "Q8_0", 0);
    addTensor("blk.0.ffn_up.weight", 4864, 896, "Q8_0", 0);
    addTensor("blk.0.ffn_down.weight", 896, 4864, "Q8_0", 0);
    addTensor("blk.0.attn_norm.weight", 1, 896, "F32", 0);
    addTensor("blk.0.ffn_norm.weight", 1, 896, "F32", 0);

    // Layer 1 (same structure)
    addTensor("blk.1.attn_q.weight", 896, 896, "Q8_0", 1);
    addTensor("blk.1.attn_k.weight", 128, 896, "Q8_0", 1);
    addTensor("blk.1.attn_v.weight", 128, 896, "Q8_0", 1);
    addTensor("blk.1.attn_output.weight", 896, 896, "Q8_0", 1);
    addTensor("blk.1.ffn_gate.weight", 4864, 896, "Q8_0", 1);
    addTensor("blk.1.ffn_up.weight", 4864, 896, "Q8_0", 1);
    addTensor("blk.1.ffn_down.weight", 896, 4864, "Q8_0", 1);
    addTensor("blk.1.attn_norm.weight", 1, 896, "F32", 1);
    addTensor("blk.1.ffn_norm.weight", 1, 896, "F32", 1);

    return p;
}

} // anonymous namespace

TEST(Test__WeightMemoryEstimator, NativeBytesPerWeight_Q8_0)
{
    float bpw = WeightMemoryEstimator::getNativeBytesPerWeight("Q8_0");
    EXPECT_NEAR(bpw, 34.0f / 32.0f, 0.001f);  // 1.0625
}

TEST(Test__WeightMemoryEstimator, NativeBytesPerWeight_Q4_0)
{
    float bpw = WeightMemoryEstimator::getNativeBytesPerWeight("Q4_0");
    EXPECT_NEAR(bpw, 18.0f / 32.0f, 0.001f);  // 0.5625
}

TEST(Test__WeightMemoryEstimator, NativeBytesPerWeight_FP32)
{
    EXPECT_NEAR(WeightMemoryEstimator::getNativeBytesPerWeight("F32"), 4.0f, 0.001f);
}

TEST(Test__WeightMemoryEstimator, NativeBytesPerWeight_FP16)
{
    EXPECT_NEAR(WeightMemoryEstimator::getNativeBytesPerWeight("F16"), 2.0f, 0.001f);
}

TEST(Test__WeightMemoryEstimator, CUDAPackedBytesPerWeight_LargeK)
{
    // For large K, overhead approaches 1.125 bytes/weight
    float bpw = WeightMemoryEstimator::getCUDAPackedBytesPerWeight(4096);
    EXPECT_GT(bpw, 1.0f);
    EXPECT_LT(bpw, 1.2f);
}

TEST(Test__WeightMemoryEstimator, CUDAPackedBytesPerWeight_SmallK)
{
    // For small K, more scale overhead
    float bpw = WeightMemoryEstimator::getCUDAPackedBytesPerWeight(64);
    EXPECT_GT(bpw, 1.0f);
    EXPECT_LT(bpw, 1.3f);
}

TEST(Test__WeightMemoryEstimator, SingleDevice_NativeBytes)
{
    auto profile = createSimpleProfile();
    auto est = WeightMemoryEstimator::estimate(profile, DeviceId::cuda(0));

    // Should include all tensors (both layers + embedding + lm_head + norms)
    EXPECT_EQ(est.native_bytes, profile.total_native_bytes);
    EXPECT_GT(est.device_bytes, 0u);
}

TEST(Test__WeightMemoryEstimator, SingleDevice_CUDAPackedBytesGTNative)
{
    auto profile = createSimpleProfile();
    auto est = WeightMemoryEstimator::estimate(profile, DeviceId::cuda(0));

    // CUDA packed (int8 + scales) should be >= native Q8_0 bytes
    // (native Q8_0 is ~1.0625 bytes/weight, CUDA packed is ~1.125 bytes/weight)
    EXPECT_GE(est.device_bytes, est.native_bytes);
}

TEST(Test__WeightMemoryEstimator, TPSharded_ReducesDeviceBytes)
{
    auto profile = createSimpleProfile();

    auto est_single = WeightMemoryEstimator::estimate(profile, DeviceId::cuda(0), 0, 1);
    auto est_shard0 = WeightMemoryEstimator::estimate(profile, DeviceId::cuda(0), 0, 2);
    auto est_shard1 = WeightMemoryEstimator::estimate(profile, DeviceId::cuda(1), 1, 2);

    // Each TP shard should have less than full model
    EXPECT_LT(est_shard0.device_bytes, est_single.device_bytes);
    EXPECT_LT(est_shard1.device_bytes, est_single.device_bytes);

    // Both shards should be roughly equal
    EXPECT_NEAR(static_cast<double>(est_shard0.device_bytes),
                static_cast<double>(est_shard1.device_bytes),
                static_cast<double>(est_single.device_bytes) * 0.01);  // Within 1%
}

TEST(Test__WeightMemoryEstimator, TPSharded_ReplicatesNormWeights)
{
    auto profile = createSimpleProfile();

    auto est_single = WeightMemoryEstimator::estimate(profile, DeviceId::cuda(0), 0, 1);
    auto est_shard0 = WeightMemoryEstimator::estimate(profile, DeviceId::cuda(0), 0, 2);

    // TP-2 should have more than 50% of single (due to replicated norms + embedding)
    double ratio = static_cast<double>(est_shard0.device_bytes) /
                   static_cast<double>(est_single.device_bytes);
    EXPECT_GT(ratio, 0.5);
    EXPECT_LT(ratio, 1.0);
}

TEST(Test__WeightMemoryEstimator, PPSlice_OnlyCountsAssignedLayers)
{
    auto profile = createSimpleProfile();

    // Only layer 0
    auto est_layer0 = WeightMemoryEstimator::estimate(
        profile, DeviceId::cuda(0), 0, 1, 0, 0);

    // Only layer 1
    auto est_layer1 = WeightMemoryEstimator::estimate(
        profile, DeviceId::cuda(0), 0, 1, 1, 1);

    // All layers
    auto est_all = WeightMemoryEstimator::estimate(
        profile, DeviceId::cuda(0), 0, 1, 0, 1);

    // Layer slices should be less than full (they miss the other layer's weights)
    EXPECT_LT(est_layer0.native_bytes, est_all.native_bytes);
    EXPECT_LT(est_layer1.native_bytes, est_all.native_bytes);
}

TEST(Test__WeightMemoryEstimator, CPUPackedBytes)
{
    auto profile = createSimpleProfile();
    auto est = WeightMemoryEstimator::estimate(profile, DeviceId::cpu());

    EXPECT_GT(est.device_bytes, 0u);
}
