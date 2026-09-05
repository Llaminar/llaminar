#include <gtest/gtest.h>
#include "planning/WeightMemoryEstimator.h"
#include "planning/ModelMemoryProfile.h"
#include "backends/DeviceId.h"
#include "kernels/common/EmbedQ8Block.h"
#include "loaders/PreparedWeightRepresentationContract.h"
#include "../../utils/EmbeddingVerifierFormats.h"
#include <stdexcept>
#include <string>
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
