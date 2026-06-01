#include <gtest/gtest.h>

#include "loaders/WeightIdentity.h"
#include "loaders/WeightLifecycleTrace.h"
#include "loaders/WeightMetadataRegistry.h"
#include "loaders/WeightManager.h"
#include "../../mocks/MockModelLoader.h"
#include "config/TensorParallelConfig.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"

#include <cstdlib>
#include <memory>

using namespace llaminar2;
using namespace llaminar2::test;

TEST(Test__WeightIdentity, InfersCommonWeightRoles)
{
    EXPECT_EQ(inferWeightRole("token_embd.weight"), WeightRole::Embedding);
    EXPECT_EQ(inferWeightRole("output.weight"), WeightRole::LMHead);
    EXPECT_EQ(inferWeightRole("output_norm.weight"), WeightRole::OutputNorm);
    EXPECT_EQ(inferWeightRole("blk.3.attn_qkv.weight"), WeightRole::FusedQKV);
    EXPECT_EQ(inferWeightRole("blk.3.attn_output.weight"), WeightRole::AttentionWO);
    EXPECT_EQ(inferWeightRole("blk.3.ffn_gate_exps.weight"), WeightRole::MoEExpertGate);
    EXPECT_EQ(inferWeightRole("blk.3.ffn_gate_inp.weight"), WeightRole::MoERouter);
    EXPECT_EQ(inferWeightRole("blk.40.ffn_gate_inp.weight"), WeightRole::MoERouter);
    EXPECT_EQ(inferWeightRole("blk.3.ffn_gate_inp_shexp.weight"), WeightRole::SharedExpertGate);
    EXPECT_EQ(inferWeightRole("blk.3.ffn_gate_shexp.weight"), WeightRole::SharedExpertGate);
    EXPECT_EQ(inferWeightRole("blk.3.ffn_up_shexp.weight"), WeightRole::SharedExpertUp);
    EXPECT_EQ(inferWeightRole("blk.3.ffn_down_shexp.weight"), WeightRole::SharedExpertDown);
    EXPECT_EQ(inferWeightRole("blk.3.shared_expert_down.weight"), WeightRole::SharedExpertDown);
}

TEST(Test__WeightIdentity, InfersLayerAndExpert)
{
    EXPECT_EQ(inferWeightLayer("blk.17.attn_q.weight"), 17);
    EXPECT_EQ(inferWeightLayer("token_embd.weight"), -1);
    EXPECT_EQ(inferWeightExpert("blk.2.experts.42.ffn_gate.weight"), 42);
    EXPECT_EQ(inferWeightExpert("blk.2.ffn_gate_exps.weight"), -1);
}

TEST(Test__WeightIdentity, CreatesStableSourceIdentity)
{
    auto first = makeSourceWeightIdentity("blk.7.ffn_down.weight", ModelContextId{11}, 3);
    auto second = makeSourceWeightIdentity("blk.7.ffn_down.weight", ModelContextId{11}, 4);

    EXPECT_EQ(first.model_id.value, 11u);
    EXPECT_EQ(first.logical_id, second.logical_id);
    EXPECT_NE(first.instance_id, second.instance_id);
    EXPECT_EQ(first.role, WeightRole::FFNDown);
    EXPECT_EQ(first.layer, 7);
    EXPECT_EQ(first.derivation, WeightDerivationKind::Source);
}

TEST(Test__WeightMetadataRegistry, RegistersSourceAndDerivedClone)
{
    auto source = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 8});
    auto clone = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 8});

    WeightMetadataRegistry registry;
    ASSERT_TRUE(registry.registerSource(source.get(), "blk.0.attn_norm.weight", DeviceId::cpu()));
    ASSERT_TRUE(registry.has(source.get()));

    WeightSliceSpec slice;
    slice.source_rows = 4;
    slice.source_cols = 8;
    slice.row_count = 4;
    slice.col_count = 8;
    ASSERT_TRUE(registry.registerDerived(
        clone.get(), source.get(), WeightDerivationKind::DeviceClone, slice, DeviceId::rocm(0)));

    auto source_meta = registry.metadata(source.get());
    auto clone_meta = registry.metadata(clone.get());
    ASSERT_TRUE(source_meta.has_value());
    ASSERT_TRUE(clone_meta.has_value());

    EXPECT_EQ(source_meta->identity.derivation, WeightDerivationKind::Source);
    EXPECT_EQ(clone_meta->identity.derivation, WeightDerivationKind::DeviceClone);
    ASSERT_TRUE(clone_meta->identity.source_instance_id.has_value());
    EXPECT_EQ(*clone_meta->identity.source_instance_id, source_meta->identity.instance_id);
    EXPECT_EQ(clone_meta->residency.home_device, DeviceId::rocm(0));
    EXPECT_EQ(clone_meta->slice.source_rows, 4u);
}

TEST(Test__WeightMetadataRegistry, DescribeIncludesIdentity)
{
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
    WeightMetadataRegistry registry;
    ASSERT_TRUE(registry.registerSource(tensor.get(), "blk.1.ffn_up.weight", DeviceId::cpu()));

    auto description = registry.describe(tensor.get());
    EXPECT_NE(description.find("blk.1.ffn_up.weight"), std::string::npos);
    EXPECT_NE(description.find("FFNUp"), std::string::npos);
    EXPECT_NE(description.find("Source"), std::string::npos);
}

TEST(Test__WeightLifecycleTrace, RecordsOnlyWhenEnabled)
{
    unsetenv("LLAMINAR_WEIGHT_LIFECYCLE_TRACE");
    mutableDebugEnv().reload();
    WeightLifecycleTrace::clear();
    WeightLifecycleTrace::record(WeightLifecycleEventType::SourceLoad, "token_embd.weight");
    EXPECT_TRUE(WeightLifecycleTrace::snapshot().empty());

    setenv("LLAMINAR_WEIGHT_LIFECYCLE_TRACE", "1", 1);
    mutableDebugEnv().reload();
    WeightLifecycleTrace::clear();
    WeightLifecycleTrace::setMode(WeightInferenceMode::HybridPPTP);
    WeightLifecycleTrace::record(
        WeightLifecycleEventType::Prepare,
        "blk.0.ffn_gate.weight",
        WeightRole::FFNGate,
        0,
        DeviceId::rocm(0),
        "unit test");

    auto events = WeightLifecycleTrace::snapshot();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].mode, WeightInferenceMode::HybridPPTP);
    EXPECT_EQ(events[0].type, WeightLifecycleEventType::Prepare);
    EXPECT_EQ(events[0].role, WeightRole::FFNGate);
    EXPECT_EQ(events[0].layer, 0);
    unsetenv("LLAMINAR_WEIGHT_LIFECYCLE_TRACE");
    mutableDebugEnv().reload();
}

TEST(Test__WeightManagerMetadata, RegistersSourceAndDeviceClone)
{
    auto loader = MockModelLoaderBuilder()
                      .addFP32RandomTensor("blk.0.attn_norm.weight", {8, 1})
                      .build();

    WeightManager manager(*loader);

    auto first = manager.getWeightForDevice("blk.0.attn_norm.weight", DeviceId::cpu());
    ASSERT_NE(first, nullptr);

    auto clone = manager.getWeightForDevice("blk.0.attn_norm.weight", DeviceId::cuda(0));
    ASSERT_NE(clone, nullptr);
    ASSERT_NE(clone.get(), first.get());

    auto *registry = manager.weightMetadataRegistry();
    ASSERT_NE(registry, nullptr);

    auto source_meta = registry->metadata(first.get());
    auto clone_meta = registry->metadata(clone.get());
    ASSERT_TRUE(source_meta.has_value());
    ASSERT_TRUE(clone_meta.has_value());

    EXPECT_EQ(source_meta->identity.canonical_name, "blk.0.attn_norm.weight");
    EXPECT_EQ(source_meta->identity.derivation, WeightDerivationKind::Source);
    EXPECT_EQ(clone_meta->identity.derivation, WeightDerivationKind::DeviceClone);
    ASSERT_TRUE(clone_meta->identity.source_instance_id.has_value());
    EXPECT_EQ(*clone_meta->identity.source_instance_id, source_meta->identity.instance_id);
    EXPECT_EQ(clone_meta->residency.home_device, DeviceId::cuda(0));
}

TEST(Test__WeightManagerMetadata, RegistersLocalTPSliceMetadata)
{
    auto loader = MockModelLoaderBuilder()
                      .addFP32RandomTensor("blk.0.ffn_gate.weight", {64, 16})
                      .addFP32RandomTensor("blk.0.ffn_down.weight", {16, 64})
                      .build();

    WeightManager manager(*loader);

    WeightShardingConfig sharding;
    sharding.exact_matches["blk.0.ffn_gate.weight"] = WeightShardingMode::ColumnParallel;
    sharding.exact_dimension_matches["blk.0.ffn_gate.weight"] = WeightDimensionType::FFNHidden;
    sharding.exact_matches["blk.0.ffn_down.weight"] = WeightShardingMode::InputParallel;
    sharding.exact_dimension_matches["blk.0.ffn_down.weight"] = WeightDimensionType::FFNHidden;
    manager.setWeightShardingConfig(sharding);

    auto tp_config = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(
            2, 4, 2, 64, 128,
            std::vector<DeviceId>{DeviceId::cuda(0), DeviceId::cuda(1)}));
    manager.setTensorParallelConfig(tp_config);

    const auto &rank1 = tp_config->forDevice(DeviceId::cuda(1));
    auto gate = manager.getShardedWeightForAssignment(
        "blk.0.ffn_gate.weight", DeviceId::cuda(1), rank1, 0);
    auto down = manager.getShardedWeightForAssignment(
        "blk.0.ffn_down.weight", DeviceId::cuda(1), rank1, 0);
    ASSERT_NE(gate, nullptr);
    ASSERT_NE(down, nullptr);

    auto *registry = manager.weightMetadataRegistry();
    auto gate_meta = registry->metadata(gate.get());
    auto down_meta = registry->metadata(down.get());
    ASSERT_TRUE(gate_meta.has_value());
    ASSERT_TRUE(down_meta.has_value());

    EXPECT_EQ(gate_meta->identity.derivation, WeightDerivationKind::RowSlice);
    EXPECT_EQ(gate_meta->slice.source_rows, 64u);
    EXPECT_EQ(gate_meta->slice.row_start, 32u);
    EXPECT_EQ(gate_meta->slice.row_count, 32u);
    EXPECT_TRUE(gate_meta->slice.inner_is_presliced);

    EXPECT_EQ(down_meta->identity.derivation, WeightDerivationKind::ColumnSlice);
    EXPECT_EQ(down_meta->slice.source_cols, 64u);
    EXPECT_EQ(down_meta->slice.col_start, 32u);
    EXPECT_EQ(down_meta->slice.col_count, 32u);
    EXPECT_TRUE(down_meta->slice.inner_is_presliced);
}

TEST(Test__WeightManagerMetadata, RegistersTiedEmbeddingAliasMetadata)
{
    auto loader = MockModelLoaderBuilder()
                      .addFP32RandomTensor("token_embd.weight", {128, 16})
                      .build();

    WeightManager manager(*loader);

    WeightShardingConfig sharding;
    sharding.exact_matches["output.weight"] = WeightShardingMode::ColumnParallel;
    sharding.exact_dimension_matches["output.weight"] = WeightDimensionType::Vocab;
    manager.setWeightShardingConfig(sharding);

    auto tp_config = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(
            2, 4, 2, 64, 128,
            std::vector<DeviceId>{DeviceId::cuda(0), DeviceId::cuda(1)}));
    manager.setTensorParallelConfig(tp_config);

    const auto &rank1 = tp_config->forDevice(DeviceId::cuda(1));
    auto lm_head = manager.getShardedWeightForAssignment(
        "output.weight", DeviceId::cuda(1), rank1, -1);
    ASSERT_NE(lm_head, nullptr);

    auto meta = manager.weightMetadataRegistry()->metadata(lm_head.get());
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->identity.canonical_name, "output.weight");
    EXPECT_EQ(meta->identity.derivation, WeightDerivationKind::TiedAlias);
    EXPECT_EQ(meta->identity.role, WeightRole::LMHead);
    EXPECT_EQ(meta->slice.row_start, 64u);
    EXPECT_EQ(meta->slice.row_count, 64u);
}
