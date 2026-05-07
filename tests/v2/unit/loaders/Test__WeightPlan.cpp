#include <gtest/gtest.h>

#include "loaders/WeightPlan.h"
#include "loaders/WeightManager.h"
#include "loaders/PreparedWeightStore.h"
#include "models/GraphTypes.h"
#include "../../mocks/MockModelLoader.h"
#include "tensors/Tensors.h"

#include <memory>
#include <utility>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    WeightBinding makeBinding(
        const std::string &name,
        DeviceId device,
        PreparedWeightKind prepared_kind = PreparedWeightKind::None,
        TensorBase *tensor = nullptr)
    {
        WeightBinding binding;
        binding.identity = makeSourceWeightIdentity(name, ModelContextId{7}, 0);
        binding.residency.home_device = device;
        binding.residency.resident_device = device;
        binding.residency.host_policy = WeightHostPolicy::ReleasableAfterPreparation;
        binding.tensor = tensor;
        if (prepared_kind != PreparedWeightKind::None)
        {
            binding.prepared = PreparedWeightRef{ModelContextId{7}, 0, prepared_kind, device};
        }
        return binding;
    }
}

TEST(Test__WeightPlan, NormalizesRequirementMetadata)
{
    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::HybridPPTP;
    strategy.model_id = ModelContextId{7};
    strategy.devices = {DeviceId::rocm(0), DeviceId::cpu()};

    WeightPlan plan(strategy);
    WeightRequirement requirement;
    requirement.canonical_name = "blk.4.ffn_down.weight";
    requirement.target_device = DeviceId::rocm(0);
    requirement.expected_prepared_kind = PreparedWeightKind::RocmInt8PackedGemm;
    plan.add(requirement);

    ASSERT_EQ(plan.size(), 1u);
    const auto &stored = plan.requirements().front();
    EXPECT_EQ(stored.role, WeightRole::FFNDown);
    EXPECT_EQ(stored.layer, 4);
    EXPECT_EQ(stored.target_device, DeviceId::rocm(0));
    EXPECT_NE(plan.renderAuditTable().find("RocmInt8PackedGemm"), std::string::npos);
}

TEST(Test__FrozenModelWeightSet, LooksUpGlobalAndLayerBindings)
{
    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::LocalTP;
    strategy.model_id = ModelContextId{7};

    ModelWeightSetBuilder builder(strategy);
    auto &embedding = builder.addBinding(makeBinding("token_embd.weight", DeviceId::rocm(0), PreparedWeightKind::PreparedEmbedding));
    auto &ffn_down = builder.addBinding(makeBinding("blk.2.ffn_down.weight", DeviceId::rocm(1), PreparedWeightKind::RocmInt8PackedGemm));
    embedding.prepared->binding_id = embedding.binding_id;
    ffn_down.prepared->binding_id = ffn_down.binding_id;

    FrozenModelWeightSet frozen(strategy, builder.freezeBindings());
    ASSERT_NO_THROW(frozen.validateForGraph());

    EXPECT_EQ(frozen.global("token_embd.weight").identity.role, WeightRole::Embedding);
    EXPECT_EQ(frozen.layer(2, "ffn_down.weight").identity.role, WeightRole::FFNDown);
    EXPECT_EQ(frozen.optionalLayer(2, "missing.weight"), nullptr);

    auto rocm1_bindings = frozen.forDevice(DeviceId::rocm(1));
    ASSERT_EQ(rocm1_bindings.size(), 1u);
    EXPECT_EQ(rocm1_bindings[0]->identity.canonical_name, "blk.2.ffn_down.weight");
}

TEST(Test__FrozenModelWeightSet, ValidatesPreparedBindingIds)
{
    InferenceStrategy strategy;
    strategy.model_id = ModelContextId{7};
    ModelWeightSetBuilder builder(strategy);
    auto &binding = builder.addBinding(makeBinding("blk.0.attn_q.weight", DeviceId::cuda(0), PreparedWeightKind::CudaInt8PackedGemm));
    binding.prepared->binding_id = binding.binding_id + 99;

    FrozenModelWeightSet frozen(strategy, builder.freezeBindings());
    EXPECT_THROW(frozen.validateForGraph(), std::runtime_error);
}

TEST(Test__WeightManagerMaterialize, ProducesFrozenBindingsFromPlan)
{
    auto loader = MockModelLoaderBuilder()
                      .addFP32RandomTensor("token_embd.weight", {128, 16})
                      .addFP32RandomTensor("blk.0.ffn_down.weight", {16, 64})
                      .build();

    WeightManager manager(*loader);

    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::SingleDevice;
    strategy.model_id = ModelContextId{99};
    strategy.devices = {DeviceId::cpu()};

    WeightPlan plan(strategy);
    WeightRequirement embedding;
    embedding.canonical_name = "token_embd.weight";
    embedding.target_device = DeviceId::cpu();
    embedding.expected_prepared_kind = PreparedWeightKind::PreparedEmbedding;
    embedding.host_policy = WeightHostPolicy::ReleasableAfterPreparation;
    plan.add(embedding);

    WeightRequirement ffn_down;
    ffn_down.canonical_name = "blk.0.ffn_down.weight";
    ffn_down.layer = 0;
    ffn_down.target_device = DeviceId::cpu();
    ffn_down.expected_prepared_kind = PreparedWeightKind::CpuPackedGemm;
    plan.add(ffn_down);

    WeightRequirement optional_missing;
    optional_missing.canonical_name = "blk.0.missing.weight";
    optional_missing.required = false;
    optional_missing.layer = 0;
    plan.add(optional_missing);

    FrozenModelWeightSet frozen = manager.materialize(plan);
    ASSERT_NO_THROW(frozen.validateForGraph());
    ASSERT_EQ(frozen.bindings().size(), 2u);

    const auto &embed = frozen.global("token_embd.weight");
    EXPECT_EQ(embed.identity.model_id.value, 99u);
    EXPECT_EQ(embed.identity.role, WeightRole::Embedding);
    ASSERT_TRUE(embed.prepared.has_value());
    EXPECT_EQ(embed.prepared->kind, PreparedWeightKind::PreparedEmbedding);
    EXPECT_EQ(embed.prepared->binding_id, embed.binding_id);
    EXPECT_EQ(embed.residency.host_policy, WeightHostPolicy::ReleasableAfterPreparation);

    const auto &down = frozen.layer(0, "ffn_down.weight");
    EXPECT_EQ(down.identity.role, WeightRole::FFNDown);
    ASSERT_TRUE(down.prepared.has_value());
    EXPECT_EQ(down.prepared->kind, PreparedWeightKind::CpuPackedGemm);
    EXPECT_NE(down.tensor, nullptr);
    EXPECT_EQ(frozen.optionalLayer(0, "missing.weight"), nullptr);
}

TEST(Test__WeightManagerMaterialize, ProducesTiedAliasBindingFromSourceName)
{
    auto loader = MockModelLoaderBuilder()
                      .addFP32RandomTensor("token_embd.weight", {128, 16})
                      .build();

    WeightManager manager(*loader);

    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::SingleDevice;
    strategy.model_id = ModelContextId{123};
    strategy.devices = {DeviceId::cpu()};

    WeightPlan plan(strategy);
    WeightRequirement embedding;
    embedding.canonical_name = "token_embd.weight";
    embedding.role = WeightRole::Embedding;
    embedding.target_device = DeviceId::cpu();
    plan.add(embedding);

    WeightRequirement lm_head_alias;
    lm_head_alias.canonical_name = "output.weight";
    lm_head_alias.source_name = "token_embd.weight";
    lm_head_alias.role = WeightRole::LMHead;
    lm_head_alias.derivation = WeightDerivationKind::TiedAlias;
    lm_head_alias.target_device = DeviceId::cpu();
    lm_head_alias.expected_prepared_kind = PreparedWeightKind::CpuPackedGemm;
    plan.add(lm_head_alias);

    FrozenModelWeightSet frozen = manager.materialize(plan);
    ASSERT_NO_THROW(frozen.validateForGraph());

    const auto &embedding_binding = frozen.global("token_embd.weight");
    const auto &lm_head_binding = frozen.global("output.weight");

    EXPECT_EQ(lm_head_binding.identity.canonical_name, "output.weight");
    EXPECT_EQ(lm_head_binding.identity.role, WeightRole::LMHead);
    EXPECT_EQ(lm_head_binding.identity.derivation, WeightDerivationKind::TiedAlias);
    EXPECT_EQ(lm_head_binding.tensor, embedding_binding.tensor);
    ASSERT_TRUE(lm_head_binding.prepared.has_value());
    EXPECT_EQ(lm_head_binding.prepared->kind, PreparedWeightKind::CpuPackedGemm);
    EXPECT_EQ(lm_head_binding.prepared->binding_id, lm_head_binding.binding_id);

    auto bindings = makeModelWeightBindings(frozen);
    EXPECT_EQ(bindings.embedding_table, &embedding_binding);
    EXPECT_EQ(bindings.lm_head, &lm_head_binding);
    auto legacy = toLegacyModelWeights(bindings);
    EXPECT_EQ(legacy.embedding_table, embedding_binding.tensor);
    EXPECT_EQ(legacy.lm_head, embedding_binding.tensor);
}

TEST(Test__WeightManagerPrepare, RegistersExactFrozenBindingRefs)
{
    auto loader = MockModelLoaderBuilder()
                      .addFP32RandomTensor("token_embd.weight", {128, 16})
                      .addFP32RandomTensor("blk.0.ffn_down.weight", {16, 64})
                      .build();

    WeightManager manager(*loader);
    auto store = std::make_shared<PreparedWeightStore>(ModelContextId{321});
    manager.setPreparedWeightStore(store);

    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::HybridPPTP;
    strategy.model_id = ModelContextId{321};
    strategy.devices = {DeviceId::cpu()};

    WeightPlan plan(strategy);
    WeightRequirement embedding;
    embedding.canonical_name = "token_embd.weight";
    embedding.role = WeightRole::Embedding;
    embedding.target_device = DeviceId::cpu();
    plan.add(embedding);

    WeightRequirement lm_head_alias;
    lm_head_alias.canonical_name = "output.weight";
    lm_head_alias.source_name = "token_embd.weight";
    lm_head_alias.role = WeightRole::LMHead;
    lm_head_alias.derivation = WeightDerivationKind::TiedAlias;
    lm_head_alias.target_device = DeviceId::cpu();
    lm_head_alias.expected_prepared_kind = PreparedWeightKind::CpuPackedGemm;
    plan.add(lm_head_alias);

    WeightRequirement ffn_down;
    ffn_down.canonical_name = "blk.0.ffn_down.weight";
    ffn_down.layer = 0;
    ffn_down.target_device = DeviceId::cpu();
    ffn_down.expected_prepared_kind = PreparedWeightKind::CpuPackedGemm;
    plan.add(ffn_down);

    FrozenModelWeightSet frozen = manager.materialize(plan);
    ASSERT_TRUE(manager.prepareWeightsForDevice(frozen, DeviceId::cpu()));

    const auto &embedding_binding = frozen.global("token_embd.weight");
    const auto &lm_head_binding = frozen.global("output.weight");
    const auto &down_binding = frozen.layer(0, "ffn_down.weight");

    ASSERT_TRUE(lm_head_binding.prepared.has_value());
    ASSERT_TRUE(down_binding.prepared.has_value());
    EXPECT_TRUE(store->contains(*lm_head_binding.prepared));
    EXPECT_TRUE(store->contains(*down_binding.prepared));
    EXPECT_FALSE(store->preparedRefForBinding(embedding_binding.binding_id, DeviceId::cpu()).has_value());

    auto resolved_lm_head = store->preparedRefForBinding(lm_head_binding.binding_id, DeviceId::cpu());
    ASSERT_TRUE(resolved_lm_head.has_value());
    EXPECT_EQ(resolved_lm_head->binding_id, lm_head_binding.binding_id);
    EXPECT_EQ(lm_head_binding.tensor, embedding_binding.tensor);
}

TEST(Test__WeightManagerMaterialize, LookupDeviceCanDifferFromTargetDevice)
{
    auto loader = MockModelLoaderBuilder()
                      .addFP32RandomTensor("blk.0.ffn_down.weight", {16, 64})
                      .build();

    WeightManager manager(*loader);

    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::LocalTP;
    strategy.model_id = ModelContextId{456};
    strategy.tp_degree = 2;
    strategy.devices = {DeviceId::cuda(0), DeviceId::cuda(1)};

    WeightPlan plan(strategy);
    WeightRequirement requirement;
    requirement.canonical_name = "blk.0.ffn_down.weight";
    requirement.target_device = DeviceId::cuda(0);
    requirement.lookup_device = DeviceId::cpu();
    requirement.tp_domain = 0;
    requirement.tp_rank_or_device_index = 0;
    requirement.expected_prepared_kind = PreparedWeightKind::CudaInt8PackedGemm;
    requirement.host_policy = WeightHostPolicy::RequiredUntilGraphMaterialized;
    plan.add(requirement);

    FrozenModelWeightSet frozen = manager.materialize(plan);
    const auto &binding = frozen.layer(0, "ffn_down.weight");

    EXPECT_EQ(binding.residency.home_device, DeviceId::cuda(0));
    ASSERT_TRUE(binding.residency.resident_device.has_value());
    EXPECT_EQ(*binding.residency.resident_device, DeviceId::cuda(0));
    EXPECT_EQ(binding.identity.tp_domain, 0);
    EXPECT_EQ(binding.identity.tp_rank_or_device_index, 0);
    ASSERT_TRUE(binding.prepared.has_value());
    EXPECT_EQ(binding.prepared->device, DeviceId::cuda(0));
    EXPECT_EQ(binding.prepared->kind, PreparedWeightKind::CudaInt8PackedGemm);
    EXPECT_NE(binding.tensor, nullptr);
}

TEST(Test__ModelWeightBindings, AdaptsFrozenBindingsToLegacyPointers)
{
    auto embedding_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{16, 8});
    auto gdn_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{8, 8});
    auto gdn_dt_bias = std::make_shared<FP32Tensor>(std::vector<size_t>{8});
    auto post_attn_norm = std::make_shared<FP32Tensor>(std::vector<size_t>{8});
    auto moe_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 8, 2});

    InferenceStrategy strategy;
    strategy.model_id = ModelContextId{7};

    ModelWeightSetBuilder builder(strategy);
    auto embedding_binding = makeBinding(
        "token_embd.weight",
        DeviceId::cpu(),
        PreparedWeightKind::PreparedEmbedding,
        embedding_tensor.get());
    auto gdn_binding = makeBinding(
        "blk.3.ssm_alpha.weight",
        DeviceId::cpu(),
        PreparedWeightKind::CpuPackedGemm,
        gdn_tensor.get());
    auto moe_binding = makeBinding(
        "blk.3.ffn_gate_exps.weight",
        DeviceId::cpu(),
        PreparedWeightKind::MoeExpertSlab,
        moe_tensor.get());
    auto dt_bias_binding = makeBinding(
        "blk.3.ssm_dt.bias",
        DeviceId::cpu(),
        PreparedWeightKind::None,
        gdn_dt_bias.get());
    auto post_attn_norm_binding = makeBinding(
        "blk.3.post_attention_norm.weight",
        DeviceId::cpu(),
        PreparedWeightKind::None,
        post_attn_norm.get());
    embedding_binding.identity.role = WeightRole::Embedding;
    gdn_binding.identity.role = WeightRole::GDNSsmParam;
    moe_binding.identity.role = WeightRole::MoEExpertGate;
    dt_bias_binding.identity.role = WeightRole::Bias;
    post_attn_norm_binding.identity.role = WeightRole::Norm;
    builder.addBinding(std::move(embedding_binding));
    builder.addBinding(std::move(gdn_binding));
    builder.addBinding(std::move(moe_binding));
    builder.addBinding(std::move(dt_bias_binding));
    builder.addBinding(std::move(post_attn_norm_binding));

    FrozenModelWeightSet frozen(strategy, builder.freezeBindings());
    auto bindings = makeModelWeightBindings(frozen);
    ASSERT_NE(bindings.embedding_table, nullptr);
    ASSERT_TRUE(bindings.get_layer_weights != nullptr);

    EXPECT_EQ(bindings.embedding_table->identity.role, WeightRole::Embedding);
    auto layer_bindings = bindings.get_layer_weights(3);
    ASSERT_NE(layer_bindings.ssm_alpha, nullptr);
    ASSERT_NE(layer_bindings.ssm_dt_bias, nullptr);
    ASSERT_NE(layer_bindings.ffn_norm, nullptr);
    ASSERT_NE(layer_bindings.moe_gate_exps, nullptr);
    EXPECT_EQ(layer_bindings.ssm_alpha->identity.role, WeightRole::GDNSsmParam);
    EXPECT_EQ(layer_bindings.ssm_dt_bias->tensor, gdn_dt_bias.get());
    EXPECT_EQ(layer_bindings.ffn_norm->tensor, post_attn_norm.get());
    EXPECT_EQ(layer_bindings.moe_gate_exps->identity.role, WeightRole::MoEExpertGate);

    auto legacy = toLegacyModelWeights(bindings);
    EXPECT_EQ(legacy.embedding_table, embedding_tensor.get());
    auto legacy_layer = legacy.get_layer_weights(3);
    EXPECT_EQ(legacy_layer.ssm_alpha, gdn_tensor.get());
    EXPECT_EQ(legacy_layer.ssm_dt_bias, gdn_dt_bias.get());
    EXPECT_EQ(legacy_layer.ffn_norm, post_attn_norm.get());
    EXPECT_EQ(legacy_layer.moe_gate_exps, moe_tensor.get());
    EXPECT_EQ(legacy.get_layer_weights(4).ssm_alpha, nullptr);
}

TEST(Test__WeightManagerMaterialize, ThrowsForMissingRequiredWeight)
{
    auto loader = MockModelLoaderBuilder().build();
    WeightManager manager(*loader);

    InferenceStrategy strategy;
    WeightPlan plan(strategy);
    WeightRequirement missing;
    missing.canonical_name = "token_embd.weight";
    missing.required = true;
    plan.add(missing);

    EXPECT_THROW(manager.materialize(plan), std::runtime_error);
}
