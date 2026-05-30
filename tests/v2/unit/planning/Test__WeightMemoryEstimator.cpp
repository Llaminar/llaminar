#include <gtest/gtest.h>
#include "planning/WeightMemoryEstimator.h"
#include "planning/ModelMemoryProfile.h"
#include "backends/DeviceId.h"
#include <string>
#include <vector>

/**
 * @file Test__WeightMemoryEstimator.cpp
 * @brief Unit coverage for native, GPU, and CPU weight memory sizing rules.
 *
 * The tests lock planner constants for GGUF block layouts and GPU native-VNNI
 * packed payload layouts. They intentionally use synthetic profiles so memory
 * planning can be validated without loading multi-gigabyte model files.
 */

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

        auto addTensor = [&](const std::string &name, size_t rows, size_t cols,
                             const std::string &quant, int layer = -1)
        {
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
                t.native_bytes = rows * cols; // Simplified
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

    struct ExpectedBytesPerWeight
    {
        std::string quant_type;
        float bytes_per_weight;
    };

    const std::vector<ExpectedBytesPerWeight> &nativeFormats()
    {
        static const std::vector<ExpectedBytesPerWeight> formats = {
            {"F32", 4.0f},
            {"F16", 2.0f},
            {"BF16", 2.0f},
            {"Q4_0", 18.0f / 32.0f},
            {"Q4_1", 20.0f / 32.0f},
            {"Q5_0", 22.0f / 32.0f},
            {"Q5_1", 24.0f / 32.0f},
            {"Q8_0", 34.0f / 32.0f},
            {"Q8_1", 36.0f / 32.0f},
            {"Q2_K", 84.0f / 256.0f},
            {"Q3_K", 110.0f / 256.0f},
            {"Q4_K", 144.0f / 256.0f},
            {"Q5_K", 176.0f / 256.0f},
            {"Q6_K", 210.0f / 256.0f},
            {"Q8_K", 288.0f / 256.0f},
            {"IQ4_NL", 18.0f / 32.0f},
            {"IQ4_XS", 136.0f / 256.0f},
            {"IQ2_XXS", 66.0f / 256.0f},
            {"IQ2_XS", 74.0f / 256.0f},
            {"IQ3_XXS", 98.0f / 256.0f},
            {"IQ2_S", 82.0f / 256.0f},
            {"IQ3_S", 110.0f / 256.0f},
            {"IQ1_S", 50.0f / 256.0f},
            {"IQ1_M", 56.0f / 256.0f},
        };
        return formats;
    }

    const std::vector<ExpectedBytesPerWeight> &gpuFormats()
    {
        static const std::vector<ExpectedBytesPerWeight> formats = {
            {"F32", 4.0f},
            {"F16", 2.0f},
            {"BF16", 2.0f},
            {"Q4_0", 18.0f / 32.0f},
            {"Q4_1", 20.0f / 32.0f},
            {"Q5_0", 22.0f / 32.0f},
            {"Q5_1", 24.0f / 32.0f},
            {"Q8_0", 34.0f / 32.0f},
            {"Q8_1", 34.0f / 32.0f},
            {"Q2_K", 16.0f / 32.0f},
            {"Q3_K", 16.0f / 32.0f},
            {"Q4_K", 20.0f / 32.0f},
            {"Q5_K", 24.0f / 32.0f},
            {"Q6_K", 28.0f / 32.0f},
            {"Q8_K", 288.0f / 256.0f},
            {"IQ4_NL", 18.0f / 32.0f},
            {"IQ4_XS", 18.0f / 32.0f},
            {"IQ2_XXS", 10.0f / 32.0f},
            {"IQ2_XS", 13.0f / 32.0f},
            {"IQ3_XXS", 14.0f / 32.0f},
            {"IQ2_S", 13.0f / 32.0f},
            {"IQ3_S", 15.0f / 32.0f},
            {"IQ1_S", 10.0f / 32.0f},
            {"IQ1_M", 10.0f / 32.0f},
        };
        return formats;
    }

} // anonymous namespace

TEST(Test__WeightMemoryEstimator, NativeBytesPerWeight_AllSupportedFormats)
{
    for (const auto &format : nativeFormats())
    {
        SCOPED_TRACE(format.quant_type);
        EXPECT_NEAR(WeightMemoryEstimator::getNativeBytesPerWeight(format.quant_type),
                    format.bytes_per_weight,
                    0.0001f);
    }
}

TEST(Test__WeightMemoryEstimator, GPUPackedBytesPerWeight_AllSupportedFormats)
{
    for (const auto &format : gpuFormats())
    {
        SCOPED_TRACE(format.quant_type);
        EXPECT_NEAR(WeightMemoryEstimator::getGPUPackedBytesPerWeight(format.quant_type, 4096),
                    format.bytes_per_weight,
                    0.0001f);
    }
}

TEST(Test__WeightMemoryEstimator, CPUPackedBytesPerWeight_AllSupportedQuantFormatsAreExplicit)
{
    for (const auto &format : nativeFormats())
    {
        if (format.quant_type == "F32" || format.quant_type == "F16" || format.quant_type == "BF16")
            continue;

        SCOPED_TRACE(format.quant_type);
        EXPECT_NEAR(WeightMemoryEstimator::getCPUPackedBytesPerWeight(format.quant_type), 1.125f, 0.0001f);
    }
}

TEST(Test__WeightMemoryEstimator, NativeBytesPerWeight_Q8_0)
{
    float bpw = WeightMemoryEstimator::getNativeBytesPerWeight("Q8_0");
    EXPECT_NEAR(bpw, 34.0f / 32.0f, 0.001f); // 1.0625
}

TEST(Test__WeightMemoryEstimator, NativeBytesPerWeight_Q4_0)
{
    float bpw = WeightMemoryEstimator::getNativeBytesPerWeight("Q4_0");
    EXPECT_NEAR(bpw, 18.0f / 32.0f, 0.001f); // 0.5625
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

TEST(Test__WeightMemoryEstimator, GPUPackedBytesPerWeight_UsesNativeVNNIFormat)
{
    EXPECT_NEAR(WeightMemoryEstimator::getGPUPackedBytesPerWeight("Q4_0", 4096), 18.0f / 32.0f, 0.001f);
    EXPECT_NEAR(WeightMemoryEstimator::getGPUPackedBytesPerWeight("IQ4_NL", 4096), 18.0f / 32.0f, 0.001f);
    EXPECT_NEAR(WeightMemoryEstimator::getGPUPackedBytesPerWeight("Q4_K", 4096), 20.0f / 32.0f, 0.001f);
    EXPECT_NEAR(WeightMemoryEstimator::getGPUPackedBytesPerWeight("Q5_K", 4096), 24.0f / 32.0f, 0.001f);
    EXPECT_NEAR(WeightMemoryEstimator::getGPUPackedBytesPerWeight("Q6_K", 4096), 28.0f / 32.0f, 0.001f);
    EXPECT_NEAR(WeightMemoryEstimator::getGPUPackedBytesPerWeight("Q2_K", 4096), 16.0f / 32.0f, 0.001f);
    EXPECT_NEAR(WeightMemoryEstimator::getGPUPackedBytesPerWeight("F16", 4096), 2.0f, 0.001f);
    EXPECT_NEAR(WeightMemoryEstimator::getGPUPackedBytesPerWeight("F32", 4096), 4.0f, 0.001f);
}

TEST(Test__WeightMemoryEstimator, SingleDevice_IQProfileUsesCompactGPUPacking)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen3moe";
    profile.n_layers = 1;

    auto addTensor = [&](const std::string &name, const std::string &quant_type,
                         size_t rows, size_t cols, float native_bpw)
    {
        TensorSizeInfo tensor;
        tensor.name = name;
        tensor.elements = rows * cols;
        tensor.K = cols;
        tensor.quant_type = quant_type;
        tensor.native_bytes = static_cast<size_t>(static_cast<float>(tensor.elements) * native_bpw);
        tensor.layer_index = 0;
        profile.total_native_bytes += tensor.native_bytes;
        profile.tensors.push_back(tensor);
    };

    addTensor("blk.0.ffn_gate.weight", "IQ3_S", 4096, 4096, 110.0f / 256.0f);
    addTensor("blk.0.ffn_up.weight", "IQ3_XXS", 4096, 4096, 98.0f / 256.0f);
    addTensor("blk.0.attn_q.weight", "Q2_K", 4096, 4096, 84.0f / 256.0f);
    addTensor("blk.0.attn_k.weight", "IQ1_S", 4096, 4096, 50.0f / 256.0f);

    const auto estimate = WeightMemoryEstimator::estimate(profile, DeviceId::cuda(0));
    const size_t elements = 4096ULL * 4096ULL;
    const size_t expected_gpu_bytes =
        static_cast<size_t>(static_cast<float>(elements) * (15.0f / 32.0f)) +
        static_cast<size_t>(static_cast<float>(elements) * (14.0f / 32.0f)) +
        static_cast<size_t>(static_cast<float>(elements) * (16.0f / 32.0f)) +
        static_cast<size_t>(static_cast<float>(elements) * (10.0f / 32.0f));

    EXPECT_EQ(estimate.native_bytes, profile.total_native_bytes);
    EXPECT_EQ(estimate.device_bytes, expected_gpu_bytes);
    EXPECT_LT(estimate.device_bytes, static_cast<size_t>(static_cast<float>(elements * 4) * 1.0f));
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

    // GPU native-VNNI packed Q8_0 is payload + FP16 scale, matching native Q8_0 size.
    EXPECT_GE(est.device_bytes, est.native_bytes);
}

TEST(Test__WeightMemoryEstimator, SingleDevice_Q4KUsesCompactGPUPacking)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen3moe";
    profile.n_layers = 1;

    TensorSizeInfo q4k_tensor;
    q4k_tensor.name = "blk.0.ffn_gate.weight";
    q4k_tensor.elements = 4096 * 4096;
    q4k_tensor.K = 4096;
    q4k_tensor.quant_type = "Q4_K";
    q4k_tensor.native_bytes = static_cast<size_t>(static_cast<float>(q4k_tensor.elements) * (18.0f / 32.0f));
    q4k_tensor.layer_index = 0;
    profile.total_native_bytes = q4k_tensor.native_bytes;
    profile.tensors.push_back(q4k_tensor);

    auto estimate = WeightMemoryEstimator::estimate(profile, DeviceId::cuda(0));
    auto expected_gpu_bytes = static_cast<size_t>(static_cast<float>(q4k_tensor.elements) * (20.0f / 32.0f));

    EXPECT_EQ(estimate.native_bytes, q4k_tensor.native_bytes);
    EXPECT_EQ(estimate.device_bytes, expected_gpu_bytes);
    EXPECT_LT(estimate.device_bytes, static_cast<size_t>(static_cast<float>(q4k_tensor.elements) * 1.0f));
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
                static_cast<double>(est_single.device_bytes) * 0.01); // Within 1%
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
