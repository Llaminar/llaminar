#include <gtest/gtest.h>

#include "execution/mtp/MTPWeightManifest.h"
#include "loaders/WeightPlan.h"
#include "models/GraphTypes.h"
#include "mocks/MockModelLoader.h"
#include "tensors/Tensors.h"

#include <memory>
#include <string>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    void addTensor(MockModelLoaderBuilder &builder, const std::string &name)
    {
        builder.addFP32Tensor(name, {4, 4});
    }

    void addNextNDepth(MockModelLoaderBuilder &builder, int source_layer)
    {
        const std::string prefix = "blk." + std::to_string(source_layer) + ".";
        for (const auto *suffix : {
                 "nextn.eh_proj.weight",
                 "nextn.hnorm.weight",
                 "nextn.enorm.weight",
                 "nextn.shared_head_norm.weight",
                 "attn_norm.weight",
                 "attn_q.weight",
                 "attn_k.weight",
                 "attn_v.weight",
                 "attn_output.weight",
                 "attn_q_norm.weight",
                 "attn_k_norm.weight",
                 "post_attention_norm.weight",
                 "ffn_gate.weight",
                 "ffn_up.weight",
                 "ffn_down.weight",
             })
        {
            addTensor(builder, prefix + suffix);
        }
    }

    void addNextNMoEDepth(MockModelLoaderBuilder &builder, int source_layer)
    {
        const std::string prefix = "blk." + std::to_string(source_layer) + ".";
        for (const auto *suffix : {
                 "nextn.eh_proj.weight",
                 "nextn.hnorm.weight",
                 "nextn.enorm.weight",
                 "nextn.shared_head_norm.weight",
                 "attn_norm.weight",
                 "attn_q.weight",
                 "attn_k.weight",
                 "attn_v.weight",
                 "attn_output.weight",
                 "attn_q_norm.weight",
                 "attn_k_norm.weight",
                 "post_attention_norm.weight",
                 "ffn_gate_inp.weight",
                 "ffn_gate_exps.weight",
                 "ffn_up_exps.weight",
                 "ffn_down_exps.weight",
                 "ffn_gate_shexp.weight",
                 "ffn_up_shexp.weight",
                 "ffn_down_shexp.weight",
                 "ffn_gate_inp_shexp.weight",
             })
        {
            addTensor(builder, prefix + suffix);
        }
    }

    void addGenericMTPDepth(MockModelLoaderBuilder &builder)
    {
        for (const auto *name : {
                 "mtp.fc.weight",
                 "mtp.pre_fc_norm_hidden.weight",
                 "mtp.pre_fc_norm_embedding.weight",
                 "mtp.norm.weight",
                 "mtp.layers.0.input_layernorm.weight",
                 "mtp.layers.0.self_attn.q_proj.weight",
                 "mtp.layers.0.self_attn.k_proj.weight",
                 "mtp.layers.0.self_attn.v_proj.weight",
                 "mtp.layers.0.self_attn.o_proj.weight",
                 "mtp.layers.0.self_attn.q_norm.weight",
                 "mtp.layers.0.self_attn.k_norm.weight",
                 "mtp.layers.0.post_attention_layernorm.weight",
                 "mtp.layers.0.mlp.gate_proj.weight",
                 "mtp.layers.0.mlp.up_proj.weight",
                 "mtp.layers.0.mlp.down_proj.weight",
             })
        {
            addTensor(builder, name);
        }
    }

    struct FrozenFixture
    {
        std::vector<std::shared_ptr<TensorBase>> tensors;
        ModelWeightSetBuilder builder;

        FrozenFixture()
            : builder(makeStrategy())
        {
        }

        static InferenceStrategy makeStrategy()
        {
            InferenceStrategy strategy;
            strategy.mode = WeightInferenceMode::SingleDevice;
            strategy.model_id = ModelContextId{7};
            strategy.devices = {DeviceId::cpu()};
            return strategy;
        }

        void add(const std::string &name)
        {
            auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 4});
            WeightBinding binding;
            binding.identity = makeSourceWeightIdentity(
                name,
                ModelContextId{7},
                static_cast<uint64_t>(tensors.size() + 1));
            binding.tensor = tensor.get();
            tensors.push_back(std::move(tensor));
            builder.addBinding(std::move(binding));
        }

        FrozenModelWeightSet freeze()
        {
            return FrozenModelWeightSet(builder.strategy(), builder.freezeBindings());
        }
    };
} // namespace

TEST(Test__MTPWeightManifest, DiscoversNextNBlockLayoutFromQwen36Metadata)
{
    MockModelLoaderBuilder builder;
    builder.setArchitecture("qwen35")
        .setBlockCount(65)
        .setInt("qwen35.nextn_predict_layers", 1);
    addNextNDepth(builder, 64);
    auto loader = builder.build();

    auto manifest = discoverMTPWeightManifest(
        *loader,
        "qwen35",
        65,
        /*explicit_mtp=*/true);

    ASSERT_TRUE(manifest.available) << manifest.diagnostic;
    ASSERT_EQ(manifest.depth, 1);
    ASSERT_EQ(manifest.depths.size(), 1u);
    EXPECT_TRUE(manifest.depths[0].nextn_block_layout);
    EXPECT_EQ(manifest.depths[0].source_layer_index, 64);
    EXPECT_EQ(manifest.depths[0].fc, "blk.64.nextn.eh_proj.weight");
    EXPECT_EQ(manifest.depths[0].wq, "blk.64.attn_q.weight");
}

TEST(Test__MTPWeightManifest, InfersNextNDepthWhenMetadataIsAbsent)
{
    MockModelLoaderBuilder builder;
    builder.setArchitecture("qwen35").setBlockCount(64);
    addNextNDepth(builder, 64);
    auto loader = builder.build();

    auto manifest = discoverMTPWeightManifest(
        *loader,
        "qwen35",
        64,
        /*explicit_mtp=*/true);

    ASSERT_TRUE(manifest.available) << manifest.diagnostic;
    EXPECT_EQ(manifest.depth, 1);
    EXPECT_EQ(manifest.depths[0].source_layer_index, 64);
}

TEST(Test__MTPWeightManifest, DiscoversNextNMoEBlockLayoutFromQwen36Metadata)
{
    MockModelLoaderBuilder builder;
    builder.setArchitecture("qwen35moe")
        .setBlockCount(41)
        .setInt("qwen35moe.nextn_predict_layers", 1);
    addNextNMoEDepth(builder, 40);
    auto loader = builder.build();

    auto manifest = discoverMTPWeightManifest(
        *loader,
        "qwen35moe",
        41,
        /*explicit_mtp=*/true);

    ASSERT_TRUE(manifest.available) << manifest.diagnostic;
    ASSERT_EQ(manifest.depth, 1);
    ASSERT_EQ(manifest.depths.size(), 1u);
    EXPECT_TRUE(manifest.depths[0].nextn_block_layout);
    EXPECT_TRUE(manifest.depths[0].moe_ffn_layout);
    EXPECT_EQ(manifest.depths[0].source_layer_index, 40);
    EXPECT_EQ(manifest.depths[0].moe_gate_exps, "blk.40.ffn_gate_exps.weight");
    EXPECT_EQ(manifest.depths[0].shared_expert_gate_inp, "blk.40.ffn_gate_inp_shexp.weight");
}

TEST(Test__MTPWeightManifest, DiscoversGenericMTPLayersLayout)
{
    MockModelLoaderBuilder builder;
    builder.setArchitecture("qwen35")
        .setBlockCount(64)
        .setInt("mtp.num_hidden_layers", 1);
    addGenericMTPDepth(builder);
    auto loader = builder.build();

    auto manifest = discoverMTPWeightManifest(
        *loader,
        "qwen35",
        64,
        /*explicit_mtp=*/true);

    ASSERT_TRUE(manifest.available) << manifest.diagnostic;
    ASSERT_EQ(manifest.depths.size(), 1u);
    EXPECT_FALSE(manifest.depths[0].nextn_block_layout);
    EXPECT_EQ(manifest.depths[0].fc, "mtp.fc.weight");
    EXPECT_EQ(manifest.depths[0].wq, "mtp.layers.0.self_attn.q_proj.weight");
}

TEST(Test__MTPWeightManifest, ExplicitMTPReportsMissingRequiredWeights)
{
    MockModelLoaderBuilder builder;
    builder.setArchitecture("qwen35")
        .setBlockCount(64)
        .setInt("qwen35.nextn_predict_layers", 1);
    addTensor(builder, "blk.64.nextn.eh_proj.weight");
    auto loader = builder.build();

    auto manifest = discoverMTPWeightManifest(
        *loader,
        "qwen35",
        64,
        /*explicit_mtp=*/true);

    EXPECT_FALSE(manifest.available);
    EXPECT_EQ(manifest.depth, 1);
    EXPECT_FALSE(manifest.missing_required.empty());
}

TEST(Test__MTPWeightManifest, ModelWeightBindingsExposeNextNWeights)
{
    FrozenFixture fixture;
    fixture.add("token_embd.weight");
    fixture.add("output_norm.weight");
    fixture.add("output.weight");
    for (const auto *suffix : {
             "nextn.eh_proj.weight",
             "nextn.hnorm.weight",
             "nextn.enorm.weight",
             "nextn.shared_head_norm.weight",
             "attn_norm.weight",
             "attn_q.weight",
             "attn_k.weight",
             "attn_v.weight",
             "attn_output.weight",
             "attn_q_norm.weight",
             "attn_k_norm.weight",
             "post_attention_norm.weight",
             "ffn_gate.weight",
             "ffn_up.weight",
             "ffn_down.weight",
         })
    {
        fixture.add(std::string("blk.64.") + suffix);
    }

    auto frozen = fixture.freeze();
    auto bindings = makeModelWeightBindings(frozen);
    auto legacy = toLegacyModelWeights(bindings);

    ASSERT_EQ(bindings.mtp.depth, 1);
    ASSERT_EQ(legacy.mtp.depth, 1);
    ASSERT_EQ(legacy.mtp.depths.size(), 1u);
    EXPECT_EQ(legacy.mtp.depths[0].source_layer_index, 64);
    EXPECT_TRUE(legacy.mtp.depths[0].nextn_block_layout);
    EXPECT_NE(legacy.mtp.depths[0].fc, nullptr);
    EXPECT_NE(legacy.mtp.depths[0].pre_fc_norm_hidden, nullptr);
    EXPECT_NE(legacy.mtp.depths[0].fa_block.wq, nullptr);
    EXPECT_NE(legacy.mtp.depths[0].fa_block.q_norm, nullptr);
}
