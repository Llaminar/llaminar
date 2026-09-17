#include <gtest/gtest.h>
#include "planning/WeightMemoryEstimator.h"
#include "planning/WeightShardGeometry.h"
#include "planning/ModelMemoryProfile.h"
#include "loaders/WeightSlicer.h"
#include "execution/local_execution/graph/SchemaFactoryRegistry.h"
#include "backends/DeviceId.h"
#include "kernels/common/EmbedQ8Block.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "loaders/PreparedWeightRepresentationContract.h"
#include "../../utils/EmbeddingVerifierFormats.h"
#include <stdexcept>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
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
using namespace llaminar2::test;

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
        addTensor("token_embd.weight", 151936, 896, "Q8_0");
        addTensor("output.weight", 151936, 896, "Q8_0");
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

    ModelMemoryProfile createMoEResidencyProfile()
    {
        ModelMemoryProfile profile;
        profile.architecture = "qwen35moe";
        profile.n_layers = 2;
        profile.d_model = 32;
        profile.d_ff = 64;
        profile.n_heads = 4;
        profile.n_kv_heads = 2;
        profile.head_dim = 8;
        profile.vocab_size = 100;
        profile.max_seq_len = 64;
        profile.expert_count = 8;
        profile.expert_used_count = 2;
        profile.expert_feed_forward_length = 64;
        profile.expert_shared_feed_forward_length = 64;

        auto addF32 = [&](const std::string &name,
                          size_t elements,
                          size_t k,
                          int layer)
        {
            TensorSizeInfo tensor;
            tensor.name = name;
            tensor.elements = elements;
            tensor.K = k;
            tensor.quant_type = "F32";
            tensor.native_bytes = elements * sizeof(float);
            tensor.layer_index = layer;
            profile.total_native_bytes += tensor.native_bytes;
            profile.tensors.push_back(std::move(tensor));
        };

        addF32("token_embd.weight", 100u * 32u, 32, -1);
        addF32("output.weight", 100u * 32u, 32, -1);
        for (int layer = 0; layer < profile.n_layers; ++layer)
        {
            const std::string prefix = "blk." + std::to_string(layer);
            addF32(prefix + ".attn_q.weight", 32u * 32u, 32, layer);
            addF32(prefix + ".ffn_gate_shexp.weight", 64u * 32u, 32, layer);
            addF32(
                prefix + ".ffn_gate_exps.weight",
                8u * 64u * 32u,
                32,
                layer);
            addF32(
                prefix + ".ffn_up_exps.weight",
                8u * 64u * 32u,
                32,
                layer);
            addF32(
                prefix + ".ffn_down_exps.weight",
                8u * 32u * 64u,
                64,
                layer);
        }
        return profile;
    }

    bool isSyntheticRoutedExpertTensor(const std::string &name)
    {
        return name.ends_with(".ffn_gate_exps.weight") ||
               name.ends_with(".ffn_up_exps.weight") ||
               name.ends_with(".ffn_down_exps.weight");
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
            {"Q8_K", 34.0f / 32.0f},
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

/**
 * @brief Input shards must retain every output row, across every format and TP1..8.
 *
 * N=65 is deliberate: flattening the shard's elements and dividing by the
 * original K truncates rows for every degree except one. This was an actual
 * under-admission, not just inaccurate performance metadata. No device or
 * tensor allocation is needed to prove the packing contract.
 */
TEST(Test__WeightMemoryEstimator, ExactInputAndOutputAxesAllFormatsTP1Through8)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen35";
    profile.n_layers = 1;
    profile.n_heads = 16;
    profile.n_kv_heads = 8;
    profile.head_dim = 32;
    profile.d_model = 65;
    // Every TP degree up to eight has native-superblock-aligned K.
    profile.d_ff = 256 * 840;
    const auto align = [](size_t bytes) { return (bytes + 255u) & ~size_t(255u); };
    for (const auto &format : nativeFormats())
    {
        SCOPED_TRACE(format.quant_type);
        TensorSizeInfo down{"blk.0.ffn_down.weight",
            static_cast<size_t>(65u * profile.d_ff * format.bytes_per_weight),
            format.quant_type, 65u * profile.d_ff, static_cast<size_t>(profile.d_ff), 0};
        TensorSizeInfo gate = down;
        gate.name = "blk.0.ffn_gate.weight";
        gate.K = 65;
        for (int degree = 1; degree <= 8; ++degree)
        {
            SCOPED_TRACE(degree);
            for (int shard = 0; shard < degree; ++shard)
            {
                for (const DeviceId device : {DeviceId::cpu(), DeviceId::cuda(shard), DeviceId::rocm(shard)})
                {
                    const size_t local_k = static_cast<size_t>(profile.d_ff / degree);
                    const auto d = WeightShardGeometryResolver(profile, device, shard, degree).resolve(down);
                    const auto g = WeightShardGeometryResolver(profile, device, shard, degree).resolve(gate);
                    ASSERT_TRUE(d.matrix());
                    ASSERT_TRUE(g.matrix());
                    EXPECT_EQ(*d.matrix(), (WeightShardMatrix{65, local_k, 1}));
                    EXPECT_EQ(*g.matrix(), (WeightShardMatrix{local_k, 65, 1}));
                    EXPECT_EQ(d.elements(), g.elements());
                    EXPECT_EQ(d.elements(), 65u * local_k);
                    profile.tensors = {down};
                    const auto bytes = WeightMemoryEstimator::estimate(profile, device, shard, degree);
                    EXPECT_EQ(bytes.native_bytes, down.native_bytes / static_cast<size_t>(degree));
                    if (device.is_gpu())
                    {
                        if (const auto *native = native_vnni_formats::forQuantType(format.quant_type))
                        {
                            const auto pool = nativeVnniPackedRegionSizes(65, local_k, *native);
                            const size_t expected = align(pool.payload_bytes) + align(pool.scales_bytes) +
                                align(pool.mins_bytes) + align(pool.emins_bytes);
                            EXPECT_EQ(bytes.device_bytes, expected) << device.to_string();
                            if (degree == 2)
                            {
                                const auto truncated = nativeVnniPackedRegionSizes(
                                    d.elements() / down.K, down.K, *native);
                                EXPECT_GT(expected, align(truncated.payload_bytes) + align(truncated.scales_bytes) +
                                    align(truncated.mins_bytes) + align(truncated.emins_bytes));
                            }
                        }
                        else
                            EXPECT_EQ(bytes.device_bytes,
                                static_cast<size_t>(d.elements() * format.bytes_per_weight));
                    }
                    else
                        EXPECT_EQ(bytes.device_bytes, static_cast<size_t>(d.elements() *
                            WeightMemoryEstimator::getCPUPackedBytesPerWeight(format.quant_type)));
                }
            }
        }
    }
}

/** @brief Exact runtime GQA assignments, including replicated KV, own local geometry. */
TEST(Test__WeightMemoryEstimator, ShardGeometryMatchesProductionSlicerGQAAndUnevenAssignments)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen2";
    profile.n_heads = 14;
    profile.n_kv_heads = 2;
    profile.head_dim = 64;
    profile.d_ff = 4864;
    profile.vocab_size = 151936;
    auto tp = std::make_shared<TensorParallelConfig>(TensorParallelConfig::proportionalSplit(
        std::vector{DeviceId::cuda(0), DeviceId::cuda(1), DeviceId::cuda(2), DeviceId::cuda(3)},
        std::vector<float>(4, 1.0f), 14, 2, 4864, 151936));
    WeightSlicer slicer({.n_heads = 14, .n_kv_heads = 2, .head_dim = 64},
        SchemaFactoryRegistry::getWeightShardingConfig(profile.architecture), tp);
    for (const auto &assignment : tp->assignments())
    {
        for (const std::string name : {"blk.0.attn_q.weight", "blk.0.attn_k.weight",
                "blk.0.ffn_gate.weight", "output.weight"})
        {
            const size_t n = name == "output.weight" ? 151936u :
                name == "blk.0.ffn_gate.weight" ? 4864u : name == "blk.0.attn_k.weight" ? 128u : 896u;
            const TensorSizeInfo tensor{name, n * 896 * 4, "F32", n * 896, 896, 0};
            const auto geometry = WeightShardGeometryResolver(profile, assignment.device, assignment.local_rank, 4, assignment).resolve(tensor);
            ASSERT_TRUE(geometry.matrix());
            EXPECT_EQ(geometry.matrix()->rows,
                slicer.computeSliceForAssignment(name, n, assignment).count);
            EXPECT_EQ(geometry.matrix()->columns, 896);
        }
    }
}

/** @brief Fused modulo-linked GDN keeps all selected Q/K/V spans, not one flat fraction. */
TEST(Test__WeightMemoryEstimator, ShardGeometryMatchesProductionFusedGDN)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen35";
    profile.n_heads = 16;
    profile.n_kv_heads = 8;
    profile.head_dim = 64;
    profile.d_ff = 4096;
    profile.vocab_size = 1024;
    profile.gdn_group_count = 8;
    profile.gdn_time_step_rank = 16;
    profile.gdn_state_size = 128;
    const size_t n = (2 * 8 + 16) * 128;
    const TensorSizeInfo tensor{"blk.0.attn_qkv.weight", n * 512 * 4, "F32", n * 512, 512, 0};
    for (int degree : {2, 4, 8})
    {
        std::vector<DeviceId> devices;
        for (int index = 0; index < degree; ++index) devices.push_back(DeviceId::rocm(index));
        auto tp = std::make_shared<TensorParallelConfig>(TensorParallelConfig::proportionalSplit(
            devices, std::vector<float>(degree, 1.0f), 16, 8, 4096, 1024));
        WeightSlicer slicer({.n_heads = 16, .n_kv_heads = 8, .head_dim = 64,
                .gdn_n_k_heads = 8, .gdn_n_v_heads = 16, .gdn_d_state = 128},
            SchemaFactoryRegistry::getWeightShardingConfig(profile.architecture), tp);
        for (const auto &assignment : tp->assignments())
        {
            const auto geometry = WeightShardGeometryResolver(profile, assignment.device, assignment.local_rank, degree, assignment).resolve(tensor);
            const auto slices = slicer.computeFusedQKVSliceForAssignment(tensor.name, n, assignment);
            ASSERT_TRUE(slices);
            ASSERT_TRUE(geometry.matrix());
            size_t rows = slices->q.count + slices->k.count;
            for (const auto &v : slices->v) rows += v.count;
            EXPECT_EQ(*geometry.matrix(), (WeightShardMatrix{rows, 512, 1}));
        }
    }
}

/** @brief Residency is an independent whole-expert axis, never another TP N/K split. */
TEST(Test__WeightMemoryEstimator, ExpertGeometryRetainsIndependentAxisAndExplicitEmptyResidency)
{
    const auto profile = createMoEResidencyProfile();
    const auto &expert = profile.tensors[4];
    ASSERT_EQ(expert.name, "blk.0.ffn_gate_exps.weight");
    for (int degree = 1; degree <= 8; ++degree)
    {
        size_t copies = 0;
        for (int shard = 0; shard < degree; ++shard)
        {
            const auto apportioned = WeightShardGeometryResolver(profile, DeviceId::cpu(), shard, degree).resolve(expert);
            ASSERT_TRUE(apportioned.matrix());
            copies += apportioned.matrix()->instances;
            EXPECT_EQ(apportioned.matrix()->rows, 64);
            EXPECT_EQ(apportioned.matrix()->columns, 32);
            for (size_t resident : {0u, 1u, 3u, 8u})
            {
                const auto overlay = WeightShardGeometryResolver(profile, DeviceId::cpu(), shard, degree, {}).resolve(expert, resident);
                ASSERT_TRUE(overlay.matrix());
                EXPECT_EQ(*overlay.matrix(), (WeightShardMatrix{64, 32, resident}));
                EXPECT_EQ(overlay.elements(), resident * 64 * 32);
            }
        }
        EXPECT_EQ(copies, 8);
    }
    EXPECT_THROW(WeightShardGeometryResolver(profile, DeviceId::cpu(), 0, 1, {}).resolve(expert, 9),
        std::invalid_argument);
    EXPECT_THROW(WeightShardGeometryResolver(profile, DeviceId::cpu(), 0, 1, {}).resolve(profile.tensors[0], 1),
        std::invalid_argument);
}

/** @brief Vector inventory never masquerades as GEMM work; malformed shapes fail before pricing. */
TEST(Test__WeightMemoryEstimator, ShardGeometryRejectsInvalidIdentityAndPreservesVectorIdentity)
{
    auto profile = createSimpleProfile();
    TensorSizeInfo vector{"blk.0.ssm_dt.bias", 64, "F32", 16, 16, 0};
    profile.architecture = "qwen35";
    profile.n_heads = 16;
    const auto slice = WeightShardGeometryResolver(profile, DeviceId::cpu(), 1, 2).resolve(vector);
    EXPECT_EQ(slice.elements(), 8);
    EXPECT_FALSE(slice.matrix());
    auto malformed = profile.tensors[0];
    malformed.elements += 1;
    EXPECT_THROW(WeightShardGeometryResolver(profile, DeviceId::cpu()).resolve(malformed), std::invalid_argument);
    EXPECT_THROW(WeightShardGeometryResolver(profile, DeviceId::cpu(), 2, 2).resolve(vector), std::invalid_argument);
    EXPECT_THROW(WeightShardGeometryResolver(profile, DeviceId::cpu(), 0, 0).resolve(vector), std::invalid_argument);
    DeviceShardingAssignment wrong;
    wrong.device = DeviceId::cuda(0);
    wrong.local_rank = 0;
    EXPECT_THROW(WeightShardGeometryResolver(profile, DeviceId::rocm(0), 0, 2, wrong).resolve(vector),
        std::invalid_argument);
    const TensorSizeInfo marker{"blk.0.attn_q.weight", 0, "F32", 0, 0, 0};
    EXPECT_EQ(WeightShardGeometryResolver(profile, DeviceId::cpu(), 0, 2).resolve(marker).elements(), 0);
    profile.n_heads = -1;
    EXPECT_THROW(WeightShardGeometryResolver(profile, DeviceId::cpu(), 0, 2).resolve(vector),
        std::invalid_argument);
}

/** @brief Valid huge metadata does not overflow the intermediate axis-coordinate product. */
TEST(Test__WeightMemoryEstimator, ShardGeometryUsesOverflowSafeAxisCoordinates)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen2";
    profile.n_heads = 8;
    const size_t rows = std::numeric_limits<size_t>::max() / 32;
    const TensorSizeInfo tensor{"blk.0.attn_q.weight", 0, "F32", rows * 32, 32, 0};
    size_t total = 0;
    for (int shard = 0; shard < 8; ++shard)
    {
        const auto shape = WeightShardGeometryResolver(profile, DeviceId::cpu(), shard, 8).resolve(tensor);
        ASSERT_TRUE(shape.matrix());
        total += shape.elements();
    }
    EXPECT_EQ(total, tensor.elements);
}


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
    profile.architecture = "qwen3";
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
    profile.architecture = "qwen3";
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

TEST(Test__WeightMemoryEstimator, QuantizedEmbeddingUsesPreparedEmbedQ8Bytes)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen35";
    profile.n_layers = 1;
    profile.d_model = 64;

    TensorSizeInfo embedding;
    embedding.name = "token_embd.weight";
    embedding.elements = 1000 * 64;
    embedding.K = 64;
    embedding.quant_type = "Q4_K";
    embedding.native_bytes = embedding.elements * 144 / 256;
    profile.total_native_bytes = embedding.native_bytes;
    profile.tensors.push_back(embedding);

    const auto estimate =
        WeightMemoryEstimator::estimate(
            profile,
            DeviceId::cuda(0));
    const size_t expected_embedding_bytes =
        1000ULL * 2ULL * 36ULL;
    const size_t expected_tied_lm_head_bytes =
        32000ULL + 4096ULL + 4096ULL;

    EXPECT_EQ(
        estimate.prepared_embedding_bytes,
        expected_embedding_bytes);
    EXPECT_EQ(
        estimate.tied_lm_head_bytes,
        expected_tied_lm_head_bytes);
    EXPECT_EQ(
        estimate.device_bytes,
        expected_embedding_bytes +
            expected_tied_lm_head_bytes);
}

TEST(Test__WeightMemoryEstimator, ExplicitLMHeadPreventsSyntheticTiedCopy)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen35";
    profile.n_layers = 1;
    profile.d_model = 64;

    for (const std::string &name :
         {"token_embd.weight", "output.weight"})
    {
        TensorSizeInfo tensor;
        tensor.name = name;
        tensor.elements = 1000 * 64;
        tensor.K = 64;
        tensor.quant_type = "Q8_0";
        tensor.native_bytes = tensor.elements * 34 / 32;
        profile.total_native_bytes += tensor.native_bytes;
        profile.tensors.push_back(tensor);
    }

    const auto estimate =
        WeightMemoryEstimator::estimate(
            profile,
            DeviceId::cuda(0));

    EXPECT_EQ(estimate.tied_lm_head_bytes, 0u);
    EXPECT_EQ(
        estimate.prepared_embedding_bytes,
        1000ULL * 2ULL * 36ULL);
}

/**
 * @brief Native embedding views remain physical allocations, not zero-cost preparation sources.
 *
 * A mirrored table lives beside the vocabulary shard. Its component estimate
 * must therefore include FP32/FP16/BF16 just as it includes prepared EmbedQ8.
 * Odd vocabulary counts expose rounding losses through every supported TP
 * degree. Device identifiers here are metadata only; no backend is initialized.
 */
TEST(Test__WeightMemoryEstimator, NativeEmbeddingComponentsCoverEveryVocabularyShard)
{
    for (const auto &format : {"F32", "F16", "BF16"})
    {
        const size_t element_bytes = std::string_view(format) == "F32" ? 4u : 2u;
        for (const DeviceType backend : {DeviceType::CUDA, DeviceType::ROCm})
        {
            ModelMemoryProfile profile;
            profile.architecture = "qwen35";
            profile.n_layers = 1;
            profile.d_model = 64;
            profile.vocab_size = 1003;
            for (const auto &name : {"token_embd.weight", "output.weight"})
            {
                TensorSizeInfo tensor;
                tensor.name = name;
                tensor.elements = static_cast<size_t>(profile.vocab_size) * profile.d_model;
                tensor.K = profile.d_model;
                tensor.quant_type = format;
                tensor.native_bytes = tensor.elements * element_bytes;
                profile.total_native_bytes += tensor.native_bytes;
                profile.tensors.push_back(tensor);
            }
            const auto full = WeightMemoryEstimator::estimate(profile, DeviceId(backend, 0));
            const size_t full_table_bytes = profile.tensors.front().native_bytes;
            EXPECT_EQ(full.prepared_embedding_bytes, full_table_bytes);
            EXPECT_EQ(full.device_bytes, 2u * full_table_bytes);
            for (int degree = 1; degree <= 8; ++degree)
            {
                size_t component_sum = 0u;
                for (int shard = 0; shard < degree; ++shard)
                {
                    SCOPED_TRACE(std::string(format) + " " + DeviceId(backend, shard).toString() +
                                 " degree=" + std::to_string(degree));
                    const auto estimate = WeightMemoryEstimator::estimate(
                        profile, DeviceId(backend, shard), shard, degree);
                    const size_t rows = profile.vocab_size / degree +
                        (shard < profile.vocab_size % degree ? 1u : 0u);
                    EXPECT_EQ(estimate.prepared_embedding_bytes, rows * profile.d_model * element_bytes);
                    EXPECT_EQ(estimate.device_bytes, 2u * rows * profile.d_model * element_bytes);
                    component_sum += estimate.prepared_embedding_bytes;
                }
                EXPECT_EQ(component_sum, full_table_bytes);
            }
        }
    }
}

TEST(Test__WeightMemoryEstimator,
     TensorParallelVocabularyViewsAreExactThroughDegreeEight)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen35";
    profile.n_layers = 1;
    profile.d_model = 64;
    profile.vocab_size = 248323;

    TensorSizeInfo embedding;
    embedding.name = "token_embd.weight";
    embedding.elements =
        static_cast<size_t>(profile.vocab_size) *
        static_cast<size_t>(profile.d_model);
    embedding.K = static_cast<size_t>(profile.d_model);
    embedding.quant_type = "Q8_0";
    embedding.native_bytes = embedding.elements * 34u / 32u;
    profile.total_native_bytes = embedding.native_bytes;
    profile.tensors.push_back(embedding);

    const auto full = WeightMemoryEstimator::estimate(
        profile, DeviceId::cuda(0));
    ASSERT_GT(full.prepared_embedding_bytes, 0u);
    ASSERT_GT(full.lm_head_bytes, 0u);

    for (int degree = 2; degree <= 8; ++degree)
    {
        size_t prepared_sum = 0;
        size_t row_sum = 0;
        for (int shard = 0; shard < degree; ++shard)
        {
            const auto estimate = WeightMemoryEstimator::estimate(
                profile,
                DeviceId::cuda(shard),
                shard,
                degree);
            const size_t expected_rows =
                static_cast<size_t>(profile.vocab_size) /
                    static_cast<size_t>(degree) +
                (static_cast<size_t>(shard) <
                         static_cast<size_t>(profile.vocab_size) %
                             static_cast<size_t>(degree)
                     ? 1u
                     : 0u);
            const size_t expected_prepared =
                expected_rows * 2u * sizeof(EmbedQ8Block);
            EXPECT_EQ(
                estimate.prepared_embedding_bytes,
                expected_prepared)
                << "degree=" << degree << " shard=" << shard;
            EXPECT_GT(estimate.lm_head_bytes, 0u);
            EXPECT_LT(estimate.lm_head_bytes, full.lm_head_bytes);
            prepared_sum += estimate.prepared_embedding_bytes;
            row_sum += expected_rows;
        }
        EXPECT_EQ(row_sum, static_cast<size_t>(profile.vocab_size));
        EXPECT_EQ(prepared_sum, full.prepared_embedding_bytes);
    }
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

/**
 * @brief Every Qwen3.5 GDN rule must be priced from its registered schema.
 *
 * These tensor names were absent from the former generic substring classifier
 * and were consequently charged as complete replicas during TP admission even
 * though WeightManager loaded participant-local slices. The aggregate check
 * proves all sharded payload bytes are conserved while only the declared norm
 * is replicated.
 */
TEST(Test__WeightMemoryEstimator,
     Qwen35GDNSchemaShardingConservesBytesThroughTP8)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen35";
    profile.n_layers = 1;
    profile.d_model = 64;
    profile.d_ff = 256;
    profile.n_heads = 16;
    profile.n_kv_heads = 8;
    profile.head_dim = 4;
    profile.vocab_size = 256;
    profile.gdn_group_count = 8;
    profile.gdn_time_step_rank = 16;
    profile.gdn_state_size = 8;

    const auto addF32 = [&profile](
                            const std::string &name,
                            size_t rows,
                            size_t columns)
    {
        TensorSizeInfo tensor;
        tensor.name = name;
        tensor.elements = rows * columns;
        tensor.K = columns;
        tensor.quant_type = "F32";
        tensor.native_bytes = tensor.elements * sizeof(float);
        tensor.layer_index = 0;
        profile.total_native_bytes += tensor.native_bytes;
        profile.tensors.push_back(std::move(tensor));
    };

    const size_t fused_rows =
        static_cast<size_t>(
            2 * profile.gdn_group_count + profile.gdn_time_step_rank) *
        static_cast<size_t>(profile.gdn_state_size);
    const size_t value_rows =
        static_cast<size_t>(profile.gdn_time_step_rank) *
        static_cast<size_t>(profile.gdn_state_size);
    addF32("blk.0.attn_qkv.weight", fused_rows, profile.d_model);
    addF32("blk.0.attn_gate.weight", value_rows, profile.d_model);
    addF32("blk.0.ssm_out.weight", profile.d_model, value_rows);
    addF32("blk.0.ssm_alpha.weight", profile.gdn_time_step_rank,
           profile.d_model);
    addF32("blk.0.ssm_beta.weight", profile.gdn_time_step_rank,
           profile.d_model);
    addF32("blk.0.ssm_conv1d.weight", fused_rows, 4u);
    addF32("blk.0.ssm_dt.bias", profile.gdn_time_step_rank, 1u);
    addF32("blk.0.ssm_a", profile.gdn_time_step_rank, 1u);
    addF32("blk.0.ssm_norm.weight", 1u, profile.gdn_state_size);

    const size_t replicated_bytes =
        profile.tensors.back().native_bytes;
    const size_t sharded_bytes =
        profile.total_native_bytes - replicated_bytes;
    const auto full = WeightMemoryEstimator::estimate(
        profile, DeviceId::rocm(0));
    ASSERT_EQ(full.device_bytes, profile.total_native_bytes);

    for (const int degree : {2, 4, 8})
    {
        size_t aggregate = 0u;
        for (int shard = 0; shard < degree; ++shard)
        {
            const auto estimate = WeightMemoryEstimator::estimate(
                profile,
                DeviceId::rocm(shard),
                shard,
                degree);
            aggregate += estimate.device_bytes;
            EXPECT_LT(estimate.device_bytes, full.device_bytes)
                << "degree=" << degree << " shard=" << shard;
        }
        EXPECT_EQ(
            aggregate,
            sharded_bytes + replicated_bytes *
                                 static_cast<size_t>(degree))
            << "degree=" << degree;
    }
}

/**
 * @brief Q8 GDN alpha/beta tensors are admitted in their runtime FP32 form.
 *
 * The optimized deterministic tiny-projection route converts these weights
 * before GPU upload.  The memory planner must therefore price FP32 bytes, not
 * the smaller GGUF codebook, for the exact TP-local interval on both GPU
 * backends.
 */
TEST(Test__WeightMemoryEstimator,
     Qwen35Q8AlphaBetaPricePreparedFP32AcrossCudaRocmAndTP)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen35";
    profile.n_layers = 1;
    profile.d_model = 1024;
    profile.d_ff = 1792;
    profile.n_heads = 16;
    profile.n_kv_heads = 4;
    profile.head_dim = 64;
    profile.vocab_size = 124160;
    profile.gdn_group_count = 8;
    profile.gdn_time_step_rank = 16;
    profile.gdn_state_size = 128;

    for (const std::string suffix : {
             "ssm_alpha.weight", "ssm_beta.weight"})
    {
        TensorSizeInfo tensor;
        tensor.name = "blk.0." + suffix;
        tensor.elements = 16u * 1024u;
        tensor.K = 1024u;
        tensor.quant_type = "Q8_0";
        tensor.native_bytes = tensor.elements * 34u / 32u;
        tensor.layer_index = 0;
        profile.total_native_bytes += tensor.native_bytes;
        profile.tensors.push_back(std::move(tensor));
    }

    constexpr size_t kCompletePreparedBytes =
        2u * 16u * 1024u * sizeof(float);
    constexpr size_t kTP2PreparedBytes =
        2u * 8u * 1024u * sizeof(float);
    for (const DeviceId device : {
             DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        EXPECT_EQ(
            WeightMemoryEstimator::estimate(profile, device).device_bytes,
            kCompletePreparedBytes);
        EXPECT_EQ(
            WeightMemoryEstimator::estimate(
                profile, device, 0, 2).device_bytes,
            kTP2PreparedBytes);
        EXPECT_EQ(
            WeightMemoryEstimator::estimate(
                profile, device, 1, 2).device_bytes,
            kTP2PreparedBytes);
    }
}

/**
 * @brief Shared-expert input-gate accounting covers every source format.
 *
 * Runtime converts this non-GEMM gate to FP32 for all non-FP32 sources.  Sweep
 * the same canonical registry used by grouped verifier coverage so adding a
 * codebook cannot silently reintroduce planner/materializer disagreement.
 */
TEST(Test__WeightMemoryEstimator,
     SharedExpertInputGatePricesFP32ForEveryCanonicalSourceFormat)
{
    constexpr size_t kElements = 256u;
    for (const auto &format : embeddingVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        const auto source = format.create({1u, kElements}, 731u);
        ASSERT_TRUE(source);

        ModelMemoryProfile profile;
        profile.architecture = "qwen35moe";
        profile.n_layers = 1;
        TensorSizeInfo tensor;
        tensor.name = "blk.0.ffn_gate_inp_shexp.weight";
        tensor.native_bytes = source->size_bytes();
        tensor.quant_type =
            source->native_type() == TensorType::FP32
                ? "F32"
                : std::string(format.label);
        tensor.elements = kElements;
        tensor.K = kElements;
        tensor.layer_index = 0;
        profile.total_native_bytes = tensor.native_bytes;
        profile.tensors.push_back(std::move(tensor));

        constexpr size_t kExpectedBytes =
            kElements * sizeof(float);
        for (const DeviceId device : {
                 DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
        {
            EXPECT_EQ(
                WeightMemoryEstimator::estimate(profile, device).device_bytes,
                kExpectedBytes)
                << "device=" << device.to_string();
        }
    }
}

/**
 * @brief Exact local-TP assignments retain replicated GQA KV heads.
 *
 * TP4 over two KV heads is intentionally not a one-quarter byte split. The
 * same DeviceShardingAssignment used by loading must keep both K/V heads on
 * every device while query rows remain partitioned.
 */
TEST(Test__WeightMemoryEstimator,
     RankLocalTPAssignmentPricesReplicatedGQAKVHeads)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen2";
    profile.n_layers = 1;
    profile.d_model = 896;
    profile.d_ff = 4864;
    profile.n_heads = 14;
    profile.n_kv_heads = 2;
    profile.head_dim = 64;
    profile.vocab_size = 151936;

    const auto addF32 = [&profile](
                            const std::string &name,
                            size_t rows,
                            size_t columns)
    {
        TensorSizeInfo tensor;
        tensor.name = name;
        tensor.elements = rows * columns;
        tensor.K = columns;
        tensor.quant_type = "F32";
        tensor.native_bytes = tensor.elements * sizeof(float);
        tensor.layer_index = 0;
        profile.total_native_bytes += tensor.native_bytes;
        profile.tensors.push_back(std::move(tensor));
    };
    addF32("blk.0.attn_q.weight", 896u, 896u);
    addF32("blk.0.attn_k.weight", 128u, 896u);

    const std::vector<DeviceId> devices{
        DeviceId::rocm(0), DeviceId::rocm(1),
        DeviceId::rocm(2), DeviceId::rocm(3)};
    const auto tp = TensorParallelConfig::proportionalSplit(
        devices,
        std::vector<float>(devices.size(), 1.0f),
        profile.n_heads,
        profile.n_kv_heads,
        profile.d_ff,
        profile.vocab_size);
    const size_t complete_k_bytes =
        128u * 896u * sizeof(float);

    size_t aggregate_q_bytes = 0u;
    for (int shard = 0; shard < 4; ++shard)
    {
        ModelMemoryProfile one_q = profile;
        one_q.tensors = {profile.tensors[0]};
        one_q.total_native_bytes = one_q.tensors[0].native_bytes;
        const std::optional<DeviceShardingAssignment> assignment =
            tp.forRank(shard);
        const auto q = WeightMemoryEstimator::estimate(
            one_q,
            devices[static_cast<size_t>(shard)],
            shard,
            4,
            0,
            0,
            {},
            assignment);
        aggregate_q_bytes += q.device_bytes;

        ModelMemoryProfile one_k = profile;
        one_k.tensors = {profile.tensors[1]};
        one_k.total_native_bytes = one_k.tensors[0].native_bytes;
        const auto k = WeightMemoryEstimator::estimate(
            one_k,
            devices[static_cast<size_t>(shard)],
            shard,
            4,
            0,
            0,
            {},
            assignment);
        EXPECT_EQ(k.device_bytes, complete_k_bytes)
            << "shard=" << shard;
    }
    EXPECT_EQ(
        aggregate_q_bytes,
        896u * 896u * sizeof(float));
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

/** @brief Pipeline endpoints partition global uses across all source formats. */
TEST(Test__WeightMemoryEstimator, PipelineGlobalComponentsAreNotReplicatedByLayerSlicing)
{
    for (const auto &format : nativeFormats())
        for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
            for (const bool tied : {false, true})
            {
                SCOPED_TRACE(format.quant_type + ":" + std::to_string(static_cast<int>(backend)) +
                             ":tied=" + std::to_string(tied));
                auto profile = createSimpleProfile();
                profile.d_model = profile.vocab_size = 256;
                profile.tensors.clear();
                for (const auto &name : {"token_embd.weight", "output.weight", "output_norm.weight",
                                        "blk.0.attn_q.weight", "blk.1.attn_q.weight"})
                {
                    if (tied && std::string_view(name) == "output.weight") continue;
                    const bool norm = std::string_view(name) == "output_norm.weight";
                    const size_t elements = norm ? 256u : 256u * 256u;
                    profile.tensors.push_back({name,
                        static_cast<size_t>(elements * format.bytes_per_weight),
                        format.quant_type, elements, 256u,
                        std::string_view(name).starts_with("blk.0.") ? 0 :
                        std::string_view(name).starts_with("blk.1.") ? 1 : -1});
                }
                const DeviceId device(backend, 0);
                const auto full = WeightMemoryEstimator::estimate(profile, device);
                const auto entry = WeightMemoryEstimator::estimate(profile, device, 0, 1, 0, 0,
                    {}, {}, WeightComponentScope::Embedding);
                const auto terminal = WeightMemoryEstimator::estimate(profile, device, 0, 1, 1, 1,
                    {}, {}, WeightComponentScope::Terminal);
                const auto middle = WeightMemoryEstimator::estimate(profile, device, 0, 1, 0, 1,
                    {}, {}, WeightComponentScope::Intermediate);
                EXPECT_EQ(entry.lm_head_bytes, 0u);
                EXPECT_EQ(entry.tied_lm_head_bytes, 0u);
                EXPECT_EQ(terminal.prepared_embedding_bytes, 0u);
                EXPECT_EQ(middle.lm_head_bytes, 0u);
                EXPECT_EQ(middle.prepared_embedding_bytes, 0u);
                EXPECT_GT(terminal.lm_head_bytes, 0u);
                if (!tied || device.is_gpu())
                    EXPECT_EQ(entry.device_bytes + terminal.device_bytes, full.device_bytes);
                else
                    // Distinct CPU stages cannot share the full model's tied source.
                    EXPECT_EQ(entry.device_bytes + terminal.device_bytes,
                              full.device_bytes + terminal.tied_lm_head_bytes);
            }
}

/** @brief A predictor-only weight set excludes globals without subtractive accounting. */
TEST(Test__WeightMemoryEstimator, LayerOnlyScopeDoesNotInventGlobalBindings)
{
    auto profile = createSimpleProfile();
    profile.tensors.push_back({"shared_constant.weight", 64, "F32", 16, 16, -1});
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    {
        const DeviceId device(backend, 0);
        const auto exact = WeightMemoryEstimator::estimate(profile, device, 0, 1, 1, 1,
            {}, {}, WeightComponentScope::LayersOnly);
        auto layers = profile;
        std::erase_if(layers.tensors, [](const auto &tensor) { return tensor.layer_index < 0; });
        const auto oracle = WeightMemoryEstimator::estimate(layers, device, 0, 1, 1, 1);
        EXPECT_EQ(exact.native_bytes, oracle.native_bytes);
        EXPECT_EQ(exact.device_bytes, oracle.device_bytes);
        EXPECT_EQ(exact.prepared_embedding_bytes, 0u);
        EXPECT_EQ(exact.lm_head_bytes, 0u);
    }
    EXPECT_THROW(WeightMemoryEstimator::estimate(profile, DeviceId::cpu(), 0, 1, 0, -1,
        {}, {}, static_cast<WeightComponentScope>(99)), std::invalid_argument);
}

TEST(Test__WeightMemoryEstimator, CPUPackedBytes)
{
    auto profile = createSimpleProfile();
    auto est = WeightMemoryEstimator::estimate(profile, DeviceId::cpu());

    EXPECT_GT(est.device_bytes, 0u);
}

TEST(Test__WeightMemoryEstimator, ContinuationResidencyCountsDenseSharedAndExactExpertFractions)
{
    const auto profile = createMoEResidencyProfile();
    const auto residency =
        DeviceWeightResidency::continuationWithSelectedRoutedExperts(
            profile.expert_count,
            {2, 4});

    const auto estimate = WeightMemoryEstimator::estimate(
        profile,
        DeviceId::cpu(),
        0,
        1,
        0,
        1,
        residency);

    size_t expected = 0;
    for (const auto &tensor : profile.tensors)
    {
        if (!isSyntheticRoutedExpertTensor(tensor.name))
        {
            expected += tensor.native_bytes;
            continue;
        }
        const int selected = tensor.layer_index == 0 ? 2 : 4;
        expected += tensor.native_bytes * static_cast<size_t>(selected) /
                    static_cast<size_t>(profile.expert_count);
    }
    EXPECT_EQ(estimate.native_bytes, expected);
    EXPECT_EQ(estimate.device_bytes, expected)
        << "F32 CPU preparation must preserve the exact selected byte count";
    EXPECT_LT(estimate.device_bytes, profile.total_native_bytes);
}

TEST(Test__WeightMemoryEstimator, ExpertOnlyResidencyExcludesEveryDenseAndSharedTensor)
{
    const auto profile = createMoEResidencyProfile();
    const auto residency =
        DeviceWeightResidency::selectedRoutedExpertsOnly(
            profile.expert_count,
            {3, 1});

    const auto estimate = WeightMemoryEstimator::estimate(
        profile,
        DeviceId::cpu(),
        0,
        1,
        0,
        1,
        residency);

    size_t expected = 0;
    for (const auto &tensor : profile.tensors)
    {
        if (!isSyntheticRoutedExpertTensor(tensor.name))
            continue;
        const int selected = tensor.layer_index == 0 ? 3 : 1;
        expected += tensor.native_bytes * static_cast<size_t>(selected) /
                    static_cast<size_t>(profile.expert_count);
    }
    EXPECT_EQ(estimate.native_bytes, expected);
    EXPECT_EQ(estimate.device_bytes, expected);
}

TEST(Test__WeightMemoryEstimator, SelectedResidencyRejectsModelGeometryDrift)
{
    const auto profile = createMoEResidencyProfile();
    const auto wrong_denominator =
        DeviceWeightResidency::selectedRoutedExpertsOnly(
            /*model_expert_count=*/7,
            {1, 1});

    EXPECT_THROW(
        WeightMemoryEstimator::estimate(
            profile,
            DeviceId::cuda(0),
            0,
            1,
            0,
            1,
            wrong_denominator),
        std::invalid_argument);
}
