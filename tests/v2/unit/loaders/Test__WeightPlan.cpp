/**
 * @file Test__WeightPlan.cpp
 * @brief Unit tests for declarative weight planning and frozen materialization.
 */

#include <gtest/gtest.h>
#include <string_view>
#include <initializer_list>

#include "loaders/WeightPlan.h"
#include "loaders/WeightManager.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/PreparedWeightRepresentationContract.h"
#include "loaders/PreparedHalfWeights.h"
#include <array>
#include <bit>
#include "utils/DebugEnv.h"
#include "backends/ComputeBackend.h"
#include "config/TensorParallelConfig.h"
#include "models/GraphTypes.h"
#include "../../mocks/MockModelLoader.h"
#include "../../utils/EmbeddingVerifierFormats.h"
#include "tensors/Tensors.h"

#include <cstring>
#include <memory>
#include <optional>
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

    std::optional<DeviceId> firstAvailableGpuDevice()
    {
        auto &manager = DeviceManager::instance();
        manager.initialize(-1, /*log_inventory=*/false);
        for (const auto &device : manager.devices())
        {
            if (device.type == ComputeBackendType::GPU_CUDA)
                return DeviceId::cuda(device.device_id);
            if (device.type == ComputeBackendType::GPU_ROCM)
                return DeviceId::rocm(device.device_id);
        }
        return std::nullopt;
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

TEST(Test__FrozenModelWeightSet,
     DeviceQualifiedLookupKeepsHeterogeneousExpertSlicesIndependent)
{
    auto cuda_expert_parent =
        std::make_shared<FP32Tensor>(std::vector<size_t>{4, 8, 2});
    auto cpu_expert_parent =
        std::make_shared<FP32Tensor>(std::vector<size_t>{4, 8, 2});

    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::ExpertOverlayRank;
    strategy.model_id = ModelContextId{701};
    strategy.devices = {DeviceId::cuda(0), DeviceId::cpu()};

    ModelWeightSetBuilder builder(strategy);
    auto cuda_binding = makeBinding(
        "blk.3.ffn_gate_exps.weight",
        DeviceId::cuda(0),
        PreparedWeightKind::MoeExpertSlab,
        cuda_expert_parent.get());
    cuda_binding.identity.role = WeightRole::MoEExpertGate;
    cuda_binding.identity.derivation = WeightDerivationKind::ExpertSlice;
    cuda_binding.identity.overlay_domain = "cuda_hot";
    cuda_binding.slice.expert_ids = {0, 1};

    auto cpu_binding = makeBinding(
        "blk.3.ffn_gate_exps.weight",
        DeviceId::cpu(),
        PreparedWeightKind::MoeExpertSlab,
        cpu_expert_parent.get());
    cpu_binding.identity.role = WeightRole::MoEExpertGate;
    cpu_binding.identity.derivation = WeightDerivationKind::ExpertSlice;
    cpu_binding.identity.overlay_domain = "cpu_cold";
    cpu_binding.slice.expert_ids = {2, 3};

    builder.addBinding(std::move(cuda_binding));
    builder.addBinding(std::move(cpu_binding));
    FrozenModelWeightSet frozen(strategy, builder.freezeBindings());

    /* The legacy unqualified accessor observes last-write indexing. */
    ASSERT_NE(frozen.optionalLayer(3, "ffn_gate_exps.weight"), nullptr);
    EXPECT_EQ(
        frozen.optionalLayer(3, "ffn_gate_exps.weight")->tensor,
        cpu_expert_parent.get());

    const WeightBinding *cuda = frozen.optionalLayerForDevice(
        3, "ffn_gate_exps.weight", DeviceId::cuda(0));
    const WeightBinding *cpu = frozen.optionalLayerForDevice(
        3, "ffn_gate_exps.weight", DeviceId::cpu());
    ASSERT_NE(cuda, nullptr);
    ASSERT_NE(cpu, nullptr);
    EXPECT_EQ(cuda->tensor, cuda_expert_parent.get());
    EXPECT_EQ(cpu->tensor, cpu_expert_parent.get());
    EXPECT_EQ(cuda->slice.expert_ids, (std::vector<int>{0, 1}));
    EXPECT_EQ(cpu->slice.expert_ids, (std::vector<int>{2, 3}));

    const auto cuda_bindings =
        makeModelWeightBindings(frozen, DeviceId::cuda(0));
    const auto cuda_layer = cuda_bindings.get_layer_weights(3);
    ASSERT_NE(cuda_layer.moe_gate_exps, nullptr);
    EXPECT_EQ(cuda_layer.moe_gate_exps->tensor, cuda_expert_parent.get());
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

/**
 * @brief Replicated GPU graphs share sources but never raw device operands.
 *
 * Replicated-dense ExpertOverlay creates one complete graph per GPU while the
 * routed collective remains multi-participant.  GEMM preparation may safely
 * read one immutable host tensor because each participant publishes its own
 * packed device allocation.  Router, norm, bias, and recurrent operands are
 * read directly by the graph and therefore require distinct TensorBase
 * identities.  This is the device-free reproduction of a four-ROCm startup
 * race where every graph bound the same `@CPU` router tensor and concurrent
 * preparation migrated it to the last device.
 */
TEST(Test__WeightManagerMaterialize,
     ReplicatedDenseGraphsKeepRawOperandsDeviceQualified)
{
    constexpr const char *kRouter = "blk.0.ffn_gate_inp.weight";
    constexpr const char *kBias = "blk.0.ssm_dt.bias";
    constexpr const char *kGemm = "blk.0.ffn_up.weight";

    auto loader = MockModelLoaderBuilder()
                      .addFP32RandomTensor(kRouter, {8, 16})
                      .addFP32RandomTensor(kBias, {8})
                      .addFP32RandomTensor(kGemm, {32, 16})
                      .build();

    WeightManager manager(*loader);
    WeightShardingConfig sharding;
    manager.setWeightShardingConfig(sharding);

    const DeviceId first_device = DeviceId::rocm(0);
    const DeviceId second_device = DeviceId::rocm(1);

    auto router_first = manager.getWeightForDevice(kRouter, first_device);
    auto router_second = manager.getWeightForDevice(kRouter, second_device);
    auto bias_first = manager.getWeightForDevice(kBias, first_device);
    auto bias_second = manager.getWeightForDevice(kBias, second_device);
    auto gemm_first = manager.getWeightForDevice(kGemm, first_device);
    auto gemm_second = manager.getWeightForDevice(kGemm, second_device);

    ASSERT_TRUE(router_first);
    ASSERT_TRUE(router_second);
    ASSERT_TRUE(bias_first);
    ASSERT_TRUE(bias_second);
    ASSERT_TRUE(gemm_first);
    ASSERT_TRUE(gemm_second);
    EXPECT_NE(router_first.get(), router_second.get());
    EXPECT_NE(bias_first.get(), bias_second.get());
    EXPECT_EQ(gemm_first.get(), gemm_second.get());
    EXPECT_TRUE(gemm_first->isHostResident())
        << "a preparation-only source must not be copied once per GPU";

    auto materialize_for = [&](DeviceId device, uint64_t model_id)
    {
        InferenceStrategy strategy;
        strategy.mode = WeightInferenceMode::SingleDevice;
        strategy.model_id = ModelContextId{model_id};
        strategy.devices = {device};

        WeightPlan plan(strategy);
        for (const char *name : {kRouter, kBias, kGemm})
        {
            WeightRequirement requirement;
            requirement.canonical_name = name;
            requirement.target_device = device;
            requirement.lookup_device = device;
            requirement.host_policy =
                WeightHostPolicy::RequiredUntilGraphMaterialized;
            plan.add(requirement);
        }
        return manager.materialize(plan);
    };

    auto first = materialize_for(first_device, 301);
    auto second = materialize_for(second_device, 302);
    EXPECT_EQ(first.layer(0, "ffn_gate_inp.weight").tensor,
              router_first.get());
    EXPECT_EQ(second.layer(0, "ffn_gate_inp.weight").tensor,
              router_second.get());
    EXPECT_NE(first.layer(0, "ffn_gate_inp.weight").tensor,
              second.layer(0, "ffn_gate_inp.weight").tensor);
    EXPECT_NE(first.layer(0, "ssm_dt.bias").tensor,
              second.layer(0, "ssm_dt.bias").tensor);
    EXPECT_EQ(first.layer(0, "ffn_up.weight").tensor,
              second.layer(0, "ffn_up.weight").tensor);
}

TEST(Test__WeightManagerMaterialize,
     SharedExpertInputGateOwnsFP32StorageWithoutReformattingGateMatrix)
{
    constexpr size_t hidden = 64;
    constexpr size_t intermediate = 32;
    const std::string input_gate_name = "blk.0.ffn_gate_inp_shexp.weight";
    const std::string gate_matrix_name = "blk.0.ffn_gate_shexp.weight";

    auto loader = MockModelLoaderBuilder()
                      .addQ8_0RandomTensor(input_gate_name, {1, hidden})
                      .addQ8_0RandomTensor(gate_matrix_name, {intermediate, hidden})
                      .build();

    const auto source_input_gate = loader->loadTensor(input_gate_name);
    ASSERT_NE(source_input_gate, nullptr);
    ASSERT_EQ(source_input_gate->native_type(), TensorType::Q8_0);
    std::vector<float> expected_input_gate(source_input_gate->numel());
    source_input_gate->to_fp32(expected_input_gate.data());

    WeightManager manager(*loader);
    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::SingleDevice;
    strategy.model_id = ModelContextId{100};
    strategy.devices = {DeviceId::cpu()};

    WeightPlan plan(strategy);
    for (const std::string &name : {input_gate_name, gate_matrix_name})
    {
        WeightRequirement requirement;
        requirement.canonical_name = name;
        requirement.target_device = DeviceId::cpu();
        requirement.host_policy = WeightHostPolicy::ReleasableAfterPreparation;
        plan.add(requirement);
    }

    FrozenModelWeightSet frozen = manager.materialize(plan);
    ASSERT_NO_THROW(frozen.validateForGraph());

    const auto &input_gate = frozen.layer(0, "ffn_gate_inp_shexp.weight");
    ASSERT_NE(input_gate.tensor, nullptr);
    ASSERT_TRUE(input_gate.tensor_owner);
    EXPECT_EQ(input_gate.identity.role, WeightRole::SharedExpertInputGate);
    EXPECT_EQ(input_gate.tensor_owner.get(), input_gate.tensor);
    EXPECT_EQ(input_gate.tensor->native_type(), TensorType::FP32);
    EXPECT_EQ(input_gate.tensor->shape(), (std::vector<size_t>{1, hidden}));
    ASSERT_EQ(input_gate.tensor->numel(), expected_input_gate.size());
    for (size_t index = 0; index < expected_input_gate.size(); ++index)
    {
        EXPECT_FLOAT_EQ(input_gate.tensor->data()[index], expected_input_gate[index])
            << "Prepared shared-expert input gate differs at element " << index;
    }

    const auto &gate_matrix = frozen.layer(0, "ffn_gate_shexp.weight");
    ASSERT_NE(gate_matrix.tensor, nullptr);
    EXPECT_EQ(gate_matrix.identity.role, WeightRole::SharedExpertGate);
    EXPECT_EQ(gate_matrix.tensor->native_type(), TensorType::Q8_0)
        << "The shared-expert GEMM gate matrix must retain its native codebook";
    EXPECT_EQ(gate_matrix.tensor->shape(),
              (std::vector<size_t>{intermediate, hidden}));
}

/**
 * @brief Prove derived input-gate storage survives runner retirement.
 *
 * Production campaigns reuse one ModelContext across matrix cells.  The first
 * runner may release the raw GGUF bytes after device preparation, so a derived
 * FP32 input gate cannot be owned solely by that runner's frozen bindings.
 * Exercise the canonical floating-point and quantized model-weight inventory
 * to ensure the lifetime contract is representation-independent.
 */
TEST(Test__WeightManagerMaterialize,
     ModelOwnedFP32OverridesSurviveRawReleaseForEverySourceFormat)
{
    constexpr size_t hidden = 256;
    const std::vector<size_t> shape{1, hidden};
    size_t tested_formats = 0;

    for (const auto &format : embeddingVerifierFormats())
    {
        auto source_unique = format.create(shape, 9000u + tested_formats);
        ASSERT_TRUE(source_unique) << format.label;

        /*
         * Native FP32 already is the canonical model-owned representation and
         * intentionally needs no derived override.  Every other loadable model
         * format must pass through the same typed cache contract.
         */
        if (source_unique->native_type() == TensorType::FP32)
            continue;

        SCOPED_TRACE(format.label);
        ++tested_formats;
        std::shared_ptr<TensorBase> source(std::move(source_unique));
        std::vector<float> expected(source->numel());
        source->to_fp32(expected.data());

        const std::string name = "blk.0.ffn_gate_inp_shexp.weight";
        auto loader = MockModelLoaderBuilder()
                          .addTensor(name, source)
                          .build();
        WeightManager manager(*loader);

        InferenceStrategy strategy;
        strategy.mode = WeightInferenceMode::SingleDevice;
        strategy.model_id = ModelContextId{101};
        strategy.devices = {DeviceId::cuda(0)};

        WeightPlan plan(strategy);
        WeightRequirement requirement;
        requirement.canonical_name = name;
        requirement.target_device = DeviceId::cuda(0);
        requirement.lookup_device = DeviceId::cpu();
        requirement.host_policy = WeightHostPolicy::ReleasableAfterPreparation;
        plan.add(requirement);

        std::weak_ptr<TensorBase> model_owned_override;
        const TensorBase *first_override = nullptr;
        {
            FrozenModelWeightSet first = manager.materialize(plan);
            const auto &binding = first.layer(0, "ffn_gate_inp_shexp.weight");
            ASSERT_TRUE(binding.tensor_owner);
            ASSERT_EQ(binding.tensor->native_type(), TensorType::FP32);
            ASSERT_EQ(binding.tensor->shape(), shape);
            ASSERT_EQ(
                std::memcmp(
                    binding.tensor->data(),
                    expected.data(),
                    expected.size() * sizeof(float)),
                0)
                << "Initial FP32 conversion changed arithmetic";
            model_owned_override = binding.tensor_owner;
            first_override = binding.tensor;
        }

        /* Runner-owned storage would expire at the end of the preceding scope. */
        ASSERT_FALSE(model_owned_override.expired());
        source->release_host_weight_data();
        ASSERT_TRUE(source->is_raw_data_released());

        FrozenModelWeightSet second = manager.materialize(plan);
        const auto &binding = second.layer(0, "ffn_gate_inp_shexp.weight");
        ASSERT_TRUE(binding.tensor_owner);
        EXPECT_EQ(binding.tensor, first_override)
            << "Rematerialization must acquire the same model-owned authority";
        EXPECT_EQ(binding.tensor_owner.get(), first_override);
        EXPECT_EQ(binding.tensor->native_type(), TensorType::FP32);
        EXPECT_EQ(
            std::memcmp(
                binding.tensor->data(),
                expected.data(),
                expected.size() * sizeof(float)),
            0)
            << "Cached override changed after source release";
    }

    EXPECT_EQ(tested_formats, embeddingVerifierFormats().size() - 1u)
        << "Every non-FP32 format in the canonical registry must be covered";
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

TEST(Test__WeightManagerMaterialize, TiedAliasTPSliceHintPreservesMaterializedSliceMetadata)
{
    auto loader = MockModelLoaderBuilder()
                      .addFP32RandomTensor("token_embd.weight", {128, 16})
                      .build();

    WeightManager manager(*loader, nullptr, nullptr,
                          WeightDistributionStrategy::SHARDED,
                          WeightPrecision::NATIVE);

    WeightShardingConfig sharding;
    sharding.exact_matches["output.weight"] = WeightShardingMode::ColumnParallel;
    sharding.exact_dimension_matches["output.weight"] = WeightDimensionType::Vocab;
    sharding.exact_matches["token_embd.weight"] = WeightShardingMode::ColumnParallel;
    sharding.exact_dimension_matches["token_embd.weight"] = WeightDimensionType::Vocab;
    manager.setWeightShardingConfig(sharding);

    auto tp_config = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(
            2,
            4,
            4,
            64,
            128,
            std::vector<DeviceId>{DeviceId::cuda(0), DeviceId::cuda(1)}));
    manager.setTensorParallelConfig(tp_config);
    const auto &rank1 = tp_config->forRank(1);

    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::LocalTP;
    strategy.model_id = ModelContextId{124};
    strategy.tp_degree = 2;
    strategy.devices = {DeviceId::cuda(0), DeviceId::cuda(1)};

    WeightSliceSpec partial_vocab_hint;
    partial_vocab_hint.source_rows = 128;
    partial_vocab_hint.row_start = static_cast<size_t>(rank1.vocab_start);
    partial_vocab_hint.row_count = static_cast<size_t>(rank1.vocab_count);

    WeightPlan plan(strategy);
    WeightRequirement lm_head_alias;
    lm_head_alias.canonical_name = "output.weight";
    lm_head_alias.source_name = "output.weight";
    lm_head_alias.role = WeightRole::LMHead;
    lm_head_alias.derivation = WeightDerivationKind::TiedAlias;
    lm_head_alias.target_device = DeviceId::cuda(1);
    lm_head_alias.lookup_device = DeviceId::cuda(1);
    lm_head_alias.tp_domain = 0;
    lm_head_alias.tp_rank_or_device_index = 1;
    lm_head_alias.expected_prepared_kind = PreparedWeightKind::CudaInt8PackedGemm;
    lm_head_alias.slice = partial_vocab_hint;
    plan.add(lm_head_alias);

    FrozenModelWeightSet frozen = manager.materialize(plan);
    const auto &binding = frozen.global("output.weight");

    EXPECT_EQ(binding.identity.canonical_name, "output.weight");
    EXPECT_EQ(binding.identity.derivation, WeightDerivationKind::TiedAlias);
    ASSERT_NE(binding.tensor, nullptr);
    ASSERT_EQ(binding.tensor->shape().size(), 2u);
    EXPECT_EQ(binding.tensor->shape()[0], static_cast<size_t>(rank1.vocab_count));
    EXPECT_EQ(binding.tensor->shape()[1], 16u);
    EXPECT_EQ(binding.slice.source_rows, 128u);
    EXPECT_EQ(binding.slice.source_cols, 16u);
    EXPECT_EQ(binding.slice.row_start, static_cast<size_t>(rank1.vocab_start));
    EXPECT_EQ(binding.slice.row_count, static_cast<size_t>(rank1.vocab_count));
    EXPECT_EQ(binding.slice.col_start, 0u);
    EXPECT_EQ(binding.slice.col_count, 16u);
    EXPECT_TRUE(binding.slice.inner_is_presliced);
}

/**
 * @brief Replicated routed experts must bypass only the expert-axis TP slice.
 *
 * LocalTP normally apportions a three-dimensional routed-expert tensor along
 * its expert axis. A graph whose routed compute policy is Replicated instead
 * needs the complete source tensor on every participant. This regression
 * exercises both materialization contracts against the same loader geometry
 * so a future weight-plan or cache change cannot quietly turn a replicated
 * participant back into a half-expert owner.
 */
TEST(Test__WeightManagerMaterialize,
     ReplicatedExpertRequirementKeepsCompleteExpertAxis)
{
    constexpr const char *kExpertWeight =
        "blk.0.ffn_gate_exps.weight";
    constexpr size_t kExpertCount = 8;

    auto make_manager = [&]()
    {
        auto loader = MockModelLoaderBuilder()
                          .addFP32RandomTensor(
                              kExpertWeight,
                              {4, 3, kExpertCount})
                          .build();
        auto manager = std::make_unique<WeightManager>(
            *loader,
            nullptr,
            nullptr,
            WeightDistributionStrategy::SHARDED,
            WeightPrecision::NATIVE);

        WeightShardingConfig sharding;
        sharding.exact_matches[kExpertWeight] =
            WeightShardingMode::ExpertIdApportioned;
        manager->setWeightShardingConfig(sharding);
        manager->setTensorParallelConfig(
            std::make_shared<TensorParallelConfig>(
                TensorParallelConfig::equalSplit(
                    2,
                    4,
                    4,
                    64,
                    128,
                    std::vector<DeviceId>{
                        DeviceId::cuda(0),
                        DeviceId::cuda(1)})));
        return std::pair{std::move(loader), std::move(manager)};
    };

    auto materialize = [&](bool bypass_tensor_parallel)
    {
        auto [loader, manager] = make_manager();

        InferenceStrategy strategy;
        strategy.mode = WeightInferenceMode::LocalTP;
        strategy.model_id = ModelContextId{
            bypass_tensor_parallel ? uint64_t{202} : uint64_t{201}};
        strategy.tp_degree = 2;
        strategy.devices = {DeviceId::cuda(0), DeviceId::cuda(1)};

        WeightPlan plan(strategy);
        WeightRequirement requirement;
        requirement.canonical_name = kExpertWeight;
        requirement.target_device = DeviceId::cuda(1);
        requirement.lookup_device = DeviceId::cuda(1);
        requirement.tp_domain = 0;
        requirement.tp_rank_or_device_index = 1;
        requirement.bypass_tensor_parallel = bypass_tensor_parallel;
        plan.add(requirement);

        auto frozen = manager->materialize(plan);
        const auto &binding = frozen.layer(0, "ffn_gate_exps.weight");
        EXPECT_NE(binding.tensor, nullptr);
        return binding.tensor ? binding.tensor->shape() : std::vector<size_t>{};
    };

    const auto apportioned_shape = materialize(false);
    ASSERT_EQ(apportioned_shape.size(), 3u);
    EXPECT_EQ(apportioned_shape[2], kExpertCount / 2);

    const auto replicated_shape = materialize(true);
    ASSERT_EQ(replicated_shape.size(), 3u);
    EXPECT_EQ(replicated_shape[2], kExpertCount);
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

TEST(Test__WeightManagerPrepare, StagePreparedStoresRemainIndependentAcrossMaterializations)
{
    auto loader = MockModelLoaderBuilder()
                      .addFP32RandomTensor("blk.0.ffn_down.weight", {16, 64})
                      .addFP32RandomTensor("blk.1.ffn_down.weight", {16, 64})
                      .build();

    WeightManager manager(*loader);

    auto make_plan = [](ModelContextId model_id, int layer, int pp_stage, int tp_domain) {
        InferenceStrategy strategy;
        strategy.mode = WeightInferenceMode::HybridPPTP;
        strategy.model_id = model_id;
        strategy.pp_stages = 2;
        strategy.tp_degree = 1;
        strategy.devices = {DeviceId::cpu()};

        WeightPlan plan(strategy);
        WeightRequirement ffn_down;
        ffn_down.canonical_name = "blk." + std::to_string(layer) + ".ffn_down.weight";
        ffn_down.layer = layer;
        ffn_down.pp_stage = pp_stage;
        ffn_down.tp_domain = tp_domain;
        ffn_down.target_device = DeviceId::cpu();
        ffn_down.expected_prepared_kind = PreparedWeightKind::CpuPackedGemm;
        plan.add(ffn_down);
        return plan;
    };

    auto store_a = std::make_shared<PreparedWeightStore>(ModelContextId{1001});
    manager.setPreparedWeightStore(store_a);
    WeightPlan plan_a = make_plan(ModelContextId{1001}, 0, 0, 10);
    FrozenModelWeightSet frozen_a = manager.materialize(plan_a);
    ASSERT_TRUE(manager.prepareWeightsForDevice(frozen_a, DeviceId::cpu()));

    const auto &binding_a = frozen_a.layer(0, "ffn_down.weight");
    ASSERT_TRUE(binding_a.prepared.has_value());
    const PreparedWeightRef ref_a = *binding_a.prepared;
    ASSERT_TRUE(store_a->contains(ref_a));
    ASSERT_NE(store_a->gemmKernel(ref_a), nullptr);

    auto store_b = std::make_shared<PreparedWeightStore>(ModelContextId{1002});
    manager.setPreparedWeightStore(store_b);
    WeightPlan plan_b = make_plan(ModelContextId{1002}, 1, 1, 20);
    FrozenModelWeightSet frozen_b = manager.materialize(plan_b);
    ASSERT_TRUE(manager.prepareWeightsForDevice(frozen_b, DeviceId::cpu()));

    const auto &binding_b = frozen_b.layer(1, "ffn_down.weight");
    ASSERT_TRUE(binding_b.prepared.has_value());
    const PreparedWeightRef ref_b = *binding_b.prepared;

    EXPECT_NE(store_a, store_b);
    EXPECT_TRUE(store_a->contains(ref_a));
    EXPECT_NE(store_a->gemmKernel(ref_a), nullptr);
    EXPECT_FALSE(store_b->contains(ref_a));
    EXPECT_FALSE(store_b->preparedRefForBinding(ref_a.binding_id, DeviceId::cpu()).has_value());

    EXPECT_TRUE(store_b->contains(ref_b));
    EXPECT_NE(store_b->gemmKernel(ref_b), nullptr);
    EXPECT_FALSE(store_a->contains(ref_b));
    EXPECT_FALSE(store_a->preparedRefForBinding(ref_b.binding_id, DeviceId::cpu()).has_value());
    EXPECT_EQ(store_a->size(), 1u);
    EXPECT_EQ(store_b->size(), 1u);
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

TEST(Test__WeightManagerPrepare, GpuFrozenBindingWithoutPreparedHintIsPreparedByStructure)
{
    const auto gpu = firstAvailableGpuDevice();
    if (!gpu)
        GTEST_SKIP() << "No CUDA/ROCm GPU available";

    auto loader = MockModelLoaderBuilder()
                      .addFP32RandomTensor("blk.0.ffn_down.weight", {16, 64})
                      .build();

    WeightManager manager(*loader);
    WeightShardingConfig sharding;
    manager.setWeightShardingConfig(sharding);

    auto store = std::make_shared<PreparedWeightStore>(ModelContextId{909});
    manager.setPreparedWeightStore(store);

    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::SingleDevice;
    strategy.model_id = ModelContextId{909};
    strategy.devices = {*gpu};

    WeightPlan plan(strategy);
    WeightRequirement requirement;
    requirement.canonical_name = "blk.0.ffn_down.weight";
    requirement.layer = 0;
    requirement.target_device = *gpu;
    requirement.lookup_device = DeviceId::cpu();
    requirement.expected_prepared_kind = PreparedWeightKind::None;
    requirement.host_policy = WeightHostPolicy::RequiredUntilGraphMaterialized;
    plan.add(requirement);

    FrozenModelWeightSet frozen = manager.materialize(plan);
    const auto &binding = frozen.layer(0, "ffn_down.weight");
    ASSERT_FALSE(binding.prepared.has_value());

    ASSERT_TRUE(manager.prepareWeightsForDevice(frozen, *gpu, /*include_expert_jobs=*/false));

    auto prepared = store->preparedRefForBinding(binding.binding_id, *gpu);
    ASSERT_TRUE(prepared.has_value());
    EXPECT_EQ(prepared->binding_id, binding.binding_id);
    EXPECT_EQ(prepared->device, *gpu);
    EXPECT_EQ(prepared->kind, gpu->is_cuda()
                                  ? PreparedWeightKind::CudaInt8PackedGemm
                                  : PreparedWeightKind::RocmInt8PackedGemm);
}

TEST(Test__WeightManagerPrepare, GpuGdnAlphaBetaProjectionBindingsArePreparedByStructure)
{
    const auto gpu = firstAvailableGpuDevice();
    if (!gpu)
        GTEST_SKIP() << "No CUDA/ROCm GPU available";

    auto loader = MockModelLoaderBuilder()
                      .addFP32RandomTensor("blk.0.ssm_alpha.weight", {4, 8})
                      .addFP32RandomTensor("blk.0.ssm_beta.weight", {4, 8})
                      .build();

    WeightManager manager(*loader);
    WeightShardingConfig sharding;
    sharding.exact_matches["blk.0.ssm_alpha.weight"] = WeightShardingMode::ColumnParallel;
    sharding.exact_dimension_matches["blk.0.ssm_alpha.weight"] = WeightDimensionType::ProportionalHeads;
    sharding.exact_matches["blk.0.ssm_beta.weight"] = WeightShardingMode::ColumnParallel;
    sharding.exact_dimension_matches["blk.0.ssm_beta.weight"] = WeightDimensionType::ProportionalHeads;
    manager.setWeightShardingConfig(sharding);

    auto store = std::make_shared<PreparedWeightStore>(ModelContextId{910});
    manager.setPreparedWeightStore(store);

    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::SingleDevice;
    strategy.model_id = ModelContextId{910};
    strategy.devices = {*gpu};

    WeightPlan plan(strategy);
    for (const std::string &name : {"blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"})
    {
        WeightRequirement requirement;
        requirement.canonical_name = name;
        requirement.layer = 0;
        requirement.target_device = *gpu;
        requirement.lookup_device = DeviceId::cpu();
        requirement.expected_prepared_kind = PreparedWeightKind::None;
        requirement.host_policy = WeightHostPolicy::RequiredUntilGraphMaterialized;
        plan.add(requirement);
    }

    FrozenModelWeightSet frozen = manager.materialize(plan);
    const auto &alpha_binding = frozen.layer(0, "ssm_alpha.weight");
    const auto &beta_binding = frozen.layer(0, "ssm_beta.weight");
    EXPECT_EQ(alpha_binding.identity.role, WeightRole::GDNAlphaBetaProjection);
    EXPECT_EQ(beta_binding.identity.role, WeightRole::GDNAlphaBetaProjection);
    ASSERT_FALSE(alpha_binding.prepared.has_value());
    ASSERT_FALSE(beta_binding.prepared.has_value());

    ASSERT_TRUE(manager.prepareWeightsForDevice(frozen, *gpu, /*include_expert_jobs=*/false));

    auto alpha_prepared = store->preparedRefForBinding(alpha_binding.binding_id, *gpu);
    auto beta_prepared = store->preparedRefForBinding(beta_binding.binding_id, *gpu);
    ASSERT_TRUE(alpha_prepared.has_value());
    ASSERT_TRUE(beta_prepared.has_value());
    EXPECT_EQ(alpha_prepared->binding_id, alpha_binding.binding_id);
    EXPECT_EQ(beta_prepared->binding_id, beta_binding.binding_id);
}

/**
 * @brief A prepared FP32 override retains the TP slice identity of its source.
 *
 * The bulk GPU preparation pass registers a prepared handle before the frozen
 * graph binding is materialized.  PreparedWeightStore can adopt that handle
 * only when the derived FP32 tensor carries the same logical interval as the
 * Q8 source.  Losing this metadata caused every alpha/beta projection to be
 * allocated a second time in heterogeneous LocalTP startup.
 */
TEST(Test__WeightManagerMaterialize,
     GdnAlphaBetaFP32OverridePreservesTensorParallelSliceIdentity)
{
    constexpr size_t kValueHeads = 16;
    constexpr size_t kHidden = 256;
    const std::string name = "blk.0.ssm_alpha.weight";

    auto loader = MockModelLoaderBuilder()
                      .addQ8_0RandomTensor(
                          name, {kValueHeads / 2u, kHidden})
                      .build();
    WeightManager manager(*loader);

    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::LocalTP;
    strategy.model_id = ModelContextId{911};
    strategy.tp_degree = 2;
    strategy.devices = {DeviceId::cuda(0), DeviceId::cuda(1)};

    WeightPlan plan(strategy);
    WeightRequirement requirement;
    requirement.canonical_name = name;
    requirement.target_device = DeviceId::cuda(0);
    requirement.lookup_device = DeviceId::cuda(0);
    requirement.tp_domain = 0;
    requirement.tp_rank_or_device_index = 0;
    requirement.slice.source_rows = kValueHeads;
    requirement.slice.source_cols = kHidden;
    requirement.slice.row_start = 0u;
    requirement.slice.row_count = kValueHeads / 2u;
    requirement.slice.col_start = 0u;
    requirement.slice.col_count = kHidden;
    requirement.slice.inner_is_presliced = true;
    plan.add(requirement);

    FrozenModelWeightSet frozen = manager.materialize(plan);
    const auto &binding = frozen.layer(0, "ssm_alpha.weight");
    ASSERT_NE(binding.tensor, nullptr);
    EXPECT_EQ(binding.identity.role,
              WeightRole::GDNAlphaBetaProjection);
    EXPECT_EQ(binding.tensor->native_type(), TensorType::FP32);
    EXPECT_EQ(binding.tensor->shape(),
              (std::vector<size_t>{kValueHeads / 2u, kHidden}));

    const auto metadata =
        manager.weightMetadataRegistry()->metadata(binding.tensor);
    ASSERT_TRUE(metadata.has_value());
    EXPECT_EQ(metadata->slice.source_rows, binding.slice.source_rows);
    EXPECT_EQ(metadata->slice.source_cols, binding.slice.source_cols);
    EXPECT_EQ(metadata->slice.row_start, binding.slice.row_start);
    EXPECT_EQ(metadata->slice.row_count, binding.slice.row_count);
    EXPECT_EQ(metadata->slice.col_start, binding.slice.col_start);
    EXPECT_EQ(metadata->slice.col_count, binding.slice.col_count);
    EXPECT_EQ(metadata->slice.inner_is_presliced,
              binding.slice.inner_is_presliced);
}

TEST(Test__WeightManagerMaterialize, FrozenBindingsRetainMaterializedTPSlices)
{
    auto loader = MockModelLoaderBuilder()
                      .addFP32RandomTensor("blk.0.ssm_a", {32})
                      .build();

    WeightManager manager(*loader, nullptr, nullptr,
                          WeightDistributionStrategy::SHARDED,
                          WeightPrecision::NATIVE);

    WeightShardingConfig sharding;
    sharding.patterns.push_back(
        {"ssm_a", WeightShardingMode::ColumnParallel, WeightDimensionType::ProportionalHeads, "GDN decay"});
    manager.setWeightShardingConfig(sharding);
    manager.setTensorParallelConfig(std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(
            2,
            4,
            4,
            64,
            128,
            std::vector<DeviceId>{DeviceId::rocm(0), DeviceId::rocm(1)})));

    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::LocalTP;
    strategy.model_id = ModelContextId{654};
    strategy.tp_degree = 2;
    strategy.devices = {DeviceId::rocm(0), DeviceId::rocm(1)};

    WeightPlan plan(strategy);
    WeightRequirement requirement;
    requirement.canonical_name = "blk.0.ssm_a";
    requirement.role = WeightRole::GDNSsmParam;
    requirement.layer = 0;
    requirement.target_device = DeviceId::rocm(0);
    requirement.lookup_device = DeviceId::rocm(0);
    requirement.tp_domain = 0;
    requirement.tp_rank_or_device_index = 0;
    plan.add(requirement);

    FrozenModelWeightSet first_frozen = manager.materialize(plan);
    const auto &first_binding = first_frozen.layer(0, "ssm_a");
    ASSERT_NE(first_binding.tensor, nullptr);
    ASSERT_TRUE(first_binding.tensor_owner);
    EXPECT_EQ(first_binding.tensor_owner.get(), first_binding.tensor);
    ASSERT_EQ(first_binding.tensor->shape().size(), 1u);
    EXPECT_EQ(first_binding.tensor->shape()[0], 16u);

    FrozenModelWeightSet second_frozen = manager.materialize(plan);
    const auto &second_binding = second_frozen.layer(0, "ssm_a");
    ASSERT_NE(second_binding.tensor, nullptr);
    ASSERT_TRUE(second_binding.tensor_owner);

    /*
     * Immutable TP slices are cached by their source/device/assignment tuple.
     * Re-materializing the same declarative plan must therefore retain one
     * authoritative slice instead of allocating a competing host shadow. Both
     * frozen sets hold shared ownership, so the first binding remains valid
     * while the second view is constructed.
     */
    EXPECT_EQ(second_binding.tensor, first_binding.tensor);
    EXPECT_EQ(second_binding.tensor_owner.get(), first_binding.tensor_owner.get());

    ASSERT_EQ(first_binding.tensor->shape().size(), 1u);
    EXPECT_EQ(first_binding.tensor->shape()[0], 16u);
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
    gdn_binding.identity.role = WeightRole::GDNAlphaBetaProjection;
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
    EXPECT_EQ(layer_bindings.ssm_alpha->identity.role, WeightRole::GDNAlphaBetaProjection);
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

TEST(Test__WeightPlan, FrozenSetRetainsItsExactPhysicalMemoryOwner)
{
    InferenceStrategy strategy;
    strategy.model_id = ModelContextId{90210};

    WeightPlan additional(
        strategy,
        PhysicalMemoryOwner::AdditionalModelWeights);
    EXPECT_EQ(
        additional.physicalMemoryOwner(),
        PhysicalMemoryOwner::AdditionalModelWeights);

    ModelWeightSetBuilder builder(strategy);
    FrozenModelWeightSet frozen(
        strategy,
        builder.freezeBindings(),
        additional.physicalMemoryOwner());
    EXPECT_EQ(
        frozen.physicalMemoryOwner(),
        PhysicalMemoryOwner::AdditionalModelWeights);

    EXPECT_THROW(
        WeightPlan(
            strategy,
            PhysicalMemoryOwner::ExecutionWorkspace),
        std::invalid_argument);
    EXPECT_THROW(
        FrozenModelWeightSet(
            strategy,
            {},
            PhysicalMemoryOwner::KVCache),
        std::invalid_argument);
}

/** @brief Reject aliases/unknown arithmetic modes instead of selecting another route. */
TEST(Test__GDNProjectionArithmetic, ExplicitPolicyRejectsUnknownValues)
{
    EXPECT_EQ(parseGDNProjectionArithmetic("native"), GDNProjectionArithmetic::Native);
    EXPECT_EQ(parseGDNProjectionArithmetic("fp32"), GDNProjectionArithmetic::FP32);
    EXPECT_EQ(parseGDNProjectionArithmetic("fp16"), GDNProjectionArithmetic::FP16);
    EXPECT_EQ(parseGDNProjectionArithmetic("fp16_alpha_beta"), GDNProjectionArithmetic::FP16AlphaBeta);
    for (const auto mode : {GDNProjectionArithmetic::Native, GDNProjectionArithmetic::FP32,
                           GDNProjectionArithmetic::FP16, GDNProjectionArithmetic::FP16AlphaBeta, GDNProjectionArithmetic::FP16MoEAlphaBeta})
        EXPECT_EQ(parseGDNProjectionArithmetic(gdnProjectionArithmeticName(mode)), mode);
    EXPECT_THROW(gdnProjectionArithmeticName(static_cast<GDNProjectionArithmetic>(255)), std::invalid_argument);
    for (const auto value : {"", "FP32", "FP16", "auto", "native32"})
        EXPECT_THROW(parseGDNProjectionArithmetic(value), std::invalid_argument);
}

/** @brief Both planning and materialization convert exactly the same GDN sources. */
TEST(Test__GDNProjectionArithmetic, SourceAwareSelectionPreservesOtherOperators)
{
    using Contract = PreparedWeightRepresentationContract;
    for (const auto architecture : {"qwen35", "qwen35moe", "qwen4exp"})
    {
        for (const auto name : {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight",
                                "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight", "blk.0.ssm_out.weight"})
        {
            const auto role = inferWeightRole(name);
            EXPECT_TRUE(Contract::isGDNProjectionSource(role, name, architecture));
            EXPECT_EQ(Contract::resolve(role, TensorType::Q6_K, name, architecture, GDNProjectionArithmetic::FP32),
                      ModelPreparedWeightRepresentation::FP32);
            EXPECT_EQ(Contract::resolve(role, "Q6_K", name, architecture, GDNProjectionArithmetic::FP32),
                      ModelPreparedWeightRepresentation::FP32);
            EXPECT_EQ(Contract::resolve(role, TensorType::FP32, name, architecture, GDNProjectionArithmetic::FP32),
                      ModelPreparedWeightRepresentation::SourceNative);
            EXPECT_EQ(Contract::resolve(role, "F32", name, architecture, GDNProjectionArithmetic::FP32),
                      ModelPreparedWeightRepresentation::SourceNative);
            EXPECT_EQ(Contract::resolve(role, TensorType::Q6_K, name, architecture, GDNProjectionArithmetic::Native),
                      Contract::resolve(role, TensorType::Q6_K));
        }
    }
    for (const auto name : {"blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight",
                            "blk.3.attn_output.weight", "blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_exps.weight",
                            "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight", "output.weight", "token_embd.weight"})
    {
        const auto role = inferWeightRole(name);
        EXPECT_FALSE(Contract::isGDNProjectionSource(role, name, "qwen35moe"));
        EXPECT_EQ(Contract::resolve(role, TensorType::Q6_K, name, "qwen35moe", GDNProjectionArithmetic::FP32),
                  Contract::resolve(role, TensorType::Q6_K));
    }
    for (const auto name : {"blk.x.attn_qkv.weight", "blk..attn_qkv.weight", "blk.-1.attn_qkv.weight",
                            "blk.0.attn_qkv.weight.extra", "other.blk.0.attn_qkv.weight"})
        EXPECT_FALSE(Contract::isGDNProjectionSource(WeightRole::FusedQKV, name, "qwen35moe"));
    EXPECT_FALSE(Contract::isGDNProjectionSource(WeightRole::MoERouter, "blk.0.attn_qkv.weight", "qwen35moe"));
    EXPECT_FALSE(Contract::isGDNProjectionSource(WeightRole::FusedQKV, "blk.0.attn_qkv.weight", "llama"));
}

/**
 * @brief The real materializer copies original Q6 values and preserves unrelated sources.
 *
 * Setup mode is restored even on a fatal assertion. No device or GEMM kernel is
 * created; the test exercises the canonical source decoder and model-owned FP32
 * override lifetime with small complete blocks and an existing FP32 gate.
 */
TEST(Test__GDNProjectionArithmetic, ActualMaterializerPreservesOriginalValuesAndNativeExperts)
{
    struct Restore
    {
        GDNProjectionArithmetic previous = debugEnv().gemm.gdn_projection_arithmetic;
        ~Restore() { mutableDebugEnv().gemm.gdn_projection_arithmetic = previous; }
    } restore;
    mutableDebugEnv().gemm.gdn_projection_arithmetic = GDNProjectionArithmetic::FP32;
    std::shared_ptr<TensorBase> source(TestTensorFactory::createQ6_KRandom({2, 256}, 817));
    ASSERT_TRUE(source);
    std::vector<float> expected(source->numel());
    source->to_fp32(expected.data());
    std::vector<uint8_t> original(source->size_bytes());
    std::memcpy(original.data(), source->raw_data(), original.size());
    auto loader = MockModelLoaderBuilder().setArchitecture("qwen35moe")
        .addTensor("blk.0.attn_qkv.weight", std::make_shared<Q6_KTensor>(source->shape(), original))
        .addTensor("blk.0.attn_gate.weight", std::make_shared<Q6_KTensor>(source->shape(), original))
        .addTensor("blk.0.ssm_out.weight", std::make_shared<Q6_KTensor>(source->shape(), original))
        .addTensor("blk.0.ffn_up.weight", source)
        .addFP32RandomTensor("blk.0.ssm_alpha.weight", {2, 256})
        .build();
    WeightManager manager(*loader);
    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::SingleDevice;
    strategy.model_id = ModelContextId{18721};
    strategy.devices = {DeviceId::cpu()};
    WeightPlan plan(strategy);
    for (const auto name : {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight",
                            "blk.0.ffn_up.weight", "blk.0.ssm_alpha.weight"})
    {
        WeightRequirement requirement;
        requirement.canonical_name = name;
        requirement.target_device = DeviceId::cpu();
        requirement.lookup_device = DeviceId::cpu();
        requirement.host_policy = WeightHostPolicy::RequiredUntilGraphMaterialized;
        plan.add(requirement);
    }
    auto frozen = manager.materialize(plan);
    for (const auto suffix : {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"})
    {
        const auto &binding = frozen.layer(0, suffix);
        ASSERT_TRUE(binding.tensor_owner);
        ASSERT_EQ(binding.tensor->native_type(), TensorType::FP32);
        EXPECT_EQ(std::memcmp(binding.tensor->data(), expected.data(), expected.size() * sizeof(float)), 0);
    }
    EXPECT_EQ(frozen.layer(0, "ffn_up.weight").tensor->native_type(), TensorType::Q6_K);
    EXPECT_EQ(frozen.layer(0, "ssm_alpha.weight").tensor,
              loader->loadTensor("blk.0.ssm_alpha.weight").get());
    EXPECT_EQ(source->native_type(), TensorType::Q6_K);
    EXPECT_EQ(std::memcmp(source->raw_data(), original.data(), original.size()), 0);
}

/** @brief Source scope, physical representation and immutable engine mode agree. */
TEST(Test__GDNProjectionArithmetic, HalfSourceAndOperandPolicyStayScoped)
{
    using Contract = PreparedWeightRepresentationContract;
    for (const auto architecture : {"qwen35", "qwen35moe", "qwen4exp"})
        for (const auto name : {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_alpha.weight",
                               "blk.0.ssm_beta.weight", "blk.0.ssm_out.weight"})
        {
            const auto role = inferWeightRole(name);
            for (const auto type : {TensorType::Q6_K, TensorType::FP32, TensorType::BF16})
                EXPECT_EQ(Contract::resolve(role, type, name, architecture, GDNProjectionArithmetic::FP16),
                    ModelPreparedWeightRepresentation::FP16);
            for (const auto format : {"Q6_K", "F32", "FP32", "BF16"})
                EXPECT_EQ(Contract::resolve(role, format, name, architecture, GDNProjectionArithmetic::FP16),
                    ModelPreparedWeightRepresentation::FP16);
            EXPECT_EQ(Contract::resolve(role, TensorType::FP16, name, architecture, GDNProjectionArithmetic::FP16),
                ModelPreparedWeightRepresentation::SourceNative);
            EXPECT_EQ(Contract::resolve(role, "F16", name, architecture, GDNProjectionArithmetic::FP16),
                ModelPreparedWeightRepresentation::SourceNative);
            EXPECT_EQ(Contract::operandPrecision(role, name, architecture, GDNProjectionArithmetic::Native),
                FloatingActivationOperandPrecision::Native);
            EXPECT_EQ(Contract::operandPrecision(role, name, architecture, GDNProjectionArithmetic::FP32),
                FloatingActivationOperandPrecision::FP32);
            EXPECT_EQ(Contract::operandPrecision(role, name, architecture, GDNProjectionArithmetic::FP16),
                FloatingActivationOperandPrecision::FP16RNE);
        }
    for (const auto name : {"blk.3.attn_q.weight", "blk.3.attn_output.weight", "blk.0.ffn_gate_inp.weight",
                           "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up.weight", "output.weight", "token_embd.weight"})
    {
        const auto role = inferWeightRole(name);
        EXPECT_EQ(Contract::resolve(role, TensorType::Q6_K, name, "qwen35moe", GDNProjectionArithmetic::FP16),
            Contract::resolve(role, TensorType::Q6_K));
        EXPECT_EQ(Contract::operandPrecision(role, name, "qwen35moe", GDNProjectionArithmetic::FP16),
            FloatingActivationOperandPrecision::Native);
    }
    EXPECT_EQ(Contract::operandPrecision(WeightRole::FusedQKV, "blk.0.attn_qkv.weight", "llama",
        GDNProjectionArithmetic::FP16), FloatingActivationOperandPrecision::Native);
}

/**
 * @brief Real source tiles retain Q6 factors while literal ties prove half RNE.
 *
 * This tests preparation only, with no CPU half-activation execution claim. The
 * source tensor and its words survive; the independently checked half words own
 * a separate model-prepared tensor. Nonmultiple K covers the live source tail.
 */
TEST(Test__GDNProjectionArithmetic, HalfMaterializationPreservesSourceFactorsAndTies)
{
    constexpr std::array<uint32_t, 13> inputs{
        0x00000000u, 0x80000000u, 0x33000000u, 0xb3000000u, 0x33800000u, 0xb3800000u,
        0x33c00000u, 0xb3c00000u, 0x3f801000u, 0x3f803000u, 0xbf803000u, 0x477fe000u, 0xc77fe000u};
    constexpr std::array<uint16_t, 13> expected{
        0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0001u, 0x8001u,
        0x0002u, 0x8002u, 0x3c00u, 0x3c02u, 0xbc02u, 0x7bffu, 0xfbffu};
    FP32Tensor original({2, inputs.size()}, DeviceId::cpu());
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < inputs.size(); ++column)
            original.mutable_typed_data()[row * inputs.size() + column] = std::bit_cast<float>(inputs[column]);
    const auto half = materializePreparedHalfWeights(original);
    ASSERT_EQ(half->shape(), original.shape());
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < inputs.size(); ++column)
        {
            EXPECT_EQ(half->typed_data()[row * inputs.size() + column], expected[column]);
            EXPECT_EQ(std::bit_cast<uint32_t>(original.data()[row * inputs.size() + column]), inputs[column]);
        }
    std::shared_ptr<TensorBase> q6(TestTensorFactory::createQ6_KRandom({2, 256}, 918));
    std::vector<uint8_t> source_bytes(q6->size_bytes());
    std::memcpy(source_bytes.data(), q6->raw_data(), source_bytes.size());
    std::vector<float> decoded(q6->numel());
    q6->to_fp32(decoded.data());
    const auto prepared = materializePreparedHalfWeights(*q6);
    for (size_t i = 0; i < decoded.size(); ++i)
        EXPECT_EQ(prepared->typed_data()[i], canonicalPreparedHalfBits(decoded[i]));
    EXPECT_EQ(std::memcmp(source_bytes.data(), q6->raw_data(), source_bytes.size()), 0);
    // Existing source half words need no conversion by the representation policy.
    FP16Tensor native_half({1, 1});
    native_half.mutable_typed_data()[0] = 0x8000;
    EXPECT_EQ(PreparedWeightRepresentationContract::resolve(WeightRole::GDNProjection,
        TensorType::FP16, "blk.0.ssm_out.weight", "qwen35moe", GDNProjectionArithmetic::FP16),
        ModelPreparedWeightRepresentation::SourceNative);
    EXPECT_EQ(native_half.typed_data()[0], 0x8000u);
}

/** @brief Invalid half operands fail during setup rather than publishing infinity. */
TEST(Test__GDNProjectionArithmetic, HalfMaterializationRejectsOverflowAndAdmitsCPUExecution)
{
    FP32Tensor source({1, 1}, DeviceId::cpu());
    for (const uint32_t word : {0x477ff000u, 0xc77ff000u, 0x7f800000u, 0x7fc00000u})
    {
        source.mutable_typed_data()[0] = std::bit_cast<float>(word);
        EXPECT_THROW(materializePreparedHalfWeights(source), std::invalid_argument);
    }
    struct Restore
    {
        GDNProjectionArithmetic previous = debugEnv().gemm.gdn_projection_arithmetic;
        ~Restore() { mutableDebugEnv().gemm.gdn_projection_arithmetic = previous; }
    } restore;
    mutableDebugEnv().gemm.gdn_projection_arithmetic = GDNProjectionArithmetic::FP16;
    auto loader = MockModelLoaderBuilder().setArchitecture("qwen35moe")
        .addFP32RandomTensor("blk.0.ssm_alpha.weight", {1, 32}).build();
    WeightManager manager(*loader);
    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::SingleDevice;
    strategy.model_id = ModelContextId{18722};
    strategy.devices = {DeviceId::cpu()};
    WeightPlan plan(strategy);
    WeightRequirement requirement;
    requirement.canonical_name = "blk.0.ssm_alpha.weight";
    requirement.target_device = DeviceId::cpu();
    requirement.lookup_device = DeviceId::cpu();
    plan.add(requirement);
    const auto frozen = manager.materialize(plan);
    const auto &binding = frozen.layer(0, "ssm_alpha.weight");
    EXPECT_EQ(binding.tensor->native_type(), TensorType::FP16);
    EXPECT_EQ(binding.floating_operand_precision, FloatingActivationOperandPrecision::FP16RNE);
}

/** @brief Router policy changes only its operands and leaves expert activation math native. */
TEST(Test__MoERouterArithmetic, OriginalSourceAndOperandPolicyAreScoped)
{
    using Contract = PreparedWeightRepresentationContract;
    const char *name = "blk.0.ffn_gate_inp.weight";
    ASSERT_EQ(inferWeightRole(name), WeightRole::MoERouter);
    for (const auto type : {TensorType::Q6_K, TensorType::Q8_0, TensorType::FP32, TensorType::BF16})
    {
        EXPECT_EQ(Contract::resolve(WeightRole::MoERouter, type, name, "qwen35moe",
            GDNProjectionArithmetic::Native, MoERouterArithmetic::FP16), ModelPreparedWeightRepresentation::FP16);
        EXPECT_EQ(Contract::resolve(WeightRole::MoERouter, type, name, "qwen35moe",
            GDNProjectionArithmetic::Native, MoERouterArithmetic::Native),
            Contract::resolve(WeightRole::MoERouter, type));
    }
    EXPECT_EQ(Contract::resolve(WeightRole::MoERouter, TensorType::FP16, name, "qwen35moe",
        GDNProjectionArithmetic::Native, MoERouterArithmetic::FP16), ModelPreparedWeightRepresentation::SourceNative);
    for (const auto source_name : {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                                  "blk.0.ffn_down_exps.weight", "blk.3.attn_output.weight",
                                  "output.weight", "token_embd.weight"})
    {
        const auto role = inferWeightRole(source_name);
        EXPECT_EQ(Contract::resolve(role, TensorType::Q6_K, source_name, "qwen35moe",
            GDNProjectionArithmetic::Native, MoERouterArithmetic::FP16), Contract::resolve(role, TensorType::Q6_K));
        EXPECT_EQ(Contract::operandPrecision(role, source_name, "qwen35moe",
            GDNProjectionArithmetic::Native, MoERouterArithmetic::FP16), FloatingActivationOperandPrecision::Native);
    }
}

/** @brief Frozen model bindings own original-source half routers once and preserve source bytes. */
TEST(Test__MoERouterArithmetic, MaterializationSealsHalfOperandsAndPreservesNativeExperts)
{
    struct Restore {
        MoERouterArithmetic previous = debugEnv().gemm.moe_router_arithmetic;
        ~Restore() { mutableDebugEnv().gemm.moe_router_arithmetic = previous; }
    } restore;
    mutableDebugEnv().gemm.moe_router_arithmetic = MoERouterArithmetic::FP16;
    auto source = std::make_shared<FP32Tensor>(std::vector<size_t>{3, 32});
    for (size_t i = 0; i < source->numel(); ++i)
        source->mutable_typed_data()[i] = static_cast<float>(i + 1) / 129.0f;
    const std::vector<float> original(source->data(), source->data() + source->numel());
    auto loader = MockModelLoaderBuilder().setArchitecture("qwen35moe")
        .addTensor("blk.0.ffn_gate_inp.weight", source)
        .addFP32RandomTensor("blk.0.ffn_up.weight", {3, 32}).build();
    WeightManager manager(*loader);
    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::SingleDevice;
    strategy.model_id = ModelContextId{18723};
    strategy.devices = {DeviceId::cpu()};
    WeightPlan plan(strategy);
    for (const auto name : {"blk.0.ffn_gate_inp.weight", "blk.0.ffn_up.weight"})
    {
        WeightRequirement requirement;
        requirement.canonical_name = name;
        requirement.target_device = DeviceId::cpu();
        requirement.lookup_device = DeviceId::cpu();
        plan.add(requirement);
    }
    const auto frozen = manager.materialize(plan);
    const auto &router = frozen.layer(0, "ffn_gate_inp.weight");
    ASSERT_TRUE(router.tensor_owner);
    ASSERT_EQ(router.tensor->native_type(), TensorType::FP16);
    EXPECT_EQ(router.floating_operand_precision, FloatingActivationOperandPrecision::FP16RNE);
    const auto *half = static_cast<const uint16_t *>(router.tensor->raw_data());
    for (size_t i = 0; i < original.size(); ++i)
        EXPECT_EQ(half[i], canonicalPreparedHalfBits(original[i]));
    EXPECT_EQ(std::memcmp(source->data(), original.data(), original.size() * sizeof(float)), 0);
    EXPECT_EQ(frozen.layer(0, "ffn_up.weight").floating_operand_precision,
        FloatingActivationOperandPrecision::Native);
    EXPECT_EQ(frozen.layer(0, "ffn_up.weight").tensor->native_type(), TensorType::FP32);
}

/** @brief Hadamard is a first quantization of floating router sources only. */
TEST(Test__MoERouterArithmetic, HadamardRejectsEveryQuantizedSourceAndPreservesOtherRoles)
{
    using C = PreparedWeightRepresentationContract;
    constexpr auto mode = MoERouterArithmetic::NativeHadamard32;
    for (int raw = 0; raw <= static_cast<int>(TensorType::AQ8); ++raw)
    {
        const auto type = static_cast<TensorType>(raw);
        if (type == TensorType::FP32 || type == TensorType::FP16 || type == TensorType::BF16) continue;
        EXPECT_THROW(C::resolve(WeightRole::MoERouter, type, "blk.0.ffn_gate_inp.weight", "qwen35moe",
            GDNProjectionArithmetic::Native, mode), std::invalid_argument);
    }
    for (const auto format : {"Q8_0", "Q6_K", "IQ3_S", "Q4_K", "unknown"})
        EXPECT_THROW(C::resolve(WeightRole::MoERouter, format, "blk.0.ffn_gate_inp.weight", "qwen35moe",
            GDNProjectionArithmetic::Native, mode), std::invalid_argument);
    for (const auto type : {TensorType::FP32, TensorType::FP16, TensorType::BF16})
        EXPECT_EQ(C::resolve(WeightRole::MoERouter, type, "blk.0.ffn_gate_inp.weight", "qwen35moe",
            GDNProjectionArithmetic::Native, mode), type == TensorType::FP32 ?
                ModelPreparedWeightRepresentation::SourceNative : ModelPreparedWeightRepresentation::FP32);
    for (const auto name : {"blk.0.attn_qkv.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight",
                            "blk.0.attn_gate.weight", "blk.0.ssm_out.weight", "blk.0.ffn_gate_exps.weight",
                            "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight", "token_embd.weight"})
    {
        const auto role = inferWeightRole(name);
        EXPECT_EQ(C::resolve(role, TensorType::Q6_K, name, "qwen35moe", GDNProjectionArithmetic::Native, mode),
                  C::resolve(role, TensorType::Q6_K));
        EXPECT_EQ(C::operandPrecision(role, name, "qwen35moe", GDNProjectionArithmetic::Native, mode),
                  FloatingActivationOperandPrecision::Native);
    }
}

/** @brief Selective half controls preserve every wide source and reject codebook conversion. */
TEST(Test__GDNProjectionArithmetic, HalfAlphaBetaSelectsOnlyOriginalFloatingControls)
{
    using Contract = PreparedWeightRepresentationContract;
    constexpr auto mode = GDNProjectionArithmetic::FP16AlphaBeta;
    for (const auto architecture : {"qwen35", "qwen35moe", "qwen4exp"})
        for (const auto name : {"blk.0.ssm_alpha.weight", "blk.12.ssm_beta.weight"})
        {
            const auto role = inferWeightRole(name);
            EXPECT_EQ(Contract::effectiveGDNArithmetic(role, name, architecture, mode), GDNProjectionArithmetic::FP16);
            EXPECT_EQ(Contract::operandPrecision(role, name, architecture, mode), FloatingActivationOperandPrecision::FP16RNE);
            for (const auto type : {TensorType::FP32, TensorType::BF16})
                EXPECT_EQ(Contract::resolve(role, type, name, architecture, mode), ModelPreparedWeightRepresentation::FP16);
            for (const auto format : {"F32", "FP32", "BF16"})
                EXPECT_EQ(Contract::resolve(role, format, name, architecture, mode), ModelPreparedWeightRepresentation::FP16);
            EXPECT_EQ(Contract::resolve(role, TensorType::FP16, name, architecture, mode), ModelPreparedWeightRepresentation::SourceNative);
            for (const auto format : {"F16", "FP16"})
                EXPECT_EQ(Contract::resolve(role, format, name, architecture, mode), ModelPreparedWeightRepresentation::SourceNative);
            for (const auto type : {TensorType::Q6_K, TensorType::Q8_0, TensorType::IQ3_S})
                EXPECT_THROW(Contract::resolve(role, type, name, architecture, mode), std::invalid_argument);
            for (const auto format : {"Q6_K", "Q8_0", "IQ3_S", "unknown"})
                EXPECT_THROW(Contract::resolve(role, format, name, architecture, mode), std::invalid_argument);
        }
    for (const auto name : {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight",
                           "blk.0.gdn_qkv.weight", "blk.3.attn_q.weight", "blk.3.attn_output.weight",
                           "blk.0.ffn_gate_inp.weight", "blk.0.ffn_up_exps.weight", "output.weight", "token_embd.weight",
                           "blk.x.ssm_alpha.weight", "blk.0.ssm_beta.weight.extra"})
    {
        const auto role = inferWeightRole(name);
        EXPECT_EQ(Contract::effectiveGDNArithmetic(role, name, "qwen35moe", mode), GDNProjectionArithmetic::Native);
        EXPECT_EQ(Contract::operandPrecision(role, name, "qwen35moe", mode), FloatingActivationOperandPrecision::Native);
        for (const auto type : {TensorType::FP32, TensorType::FP16, TensorType::Q6_K, TensorType::IQ3_S})
            EXPECT_EQ(Contract::resolve(role, type, name, "qwen35moe", mode), Contract::resolve(role, type));
        for (const auto format : {"F32", "F16", "Q6_K", "IQ3_S"})
            EXPECT_EQ(Contract::resolve(role, format, name, "qwen35moe", mode), Contract::resolve(role, format));
    }
    EXPECT_EQ(Contract::effectiveGDNArithmetic(WeightRole::GDNAlphaBetaProjection,
        "blk.0.ssm_alpha.weight", "llama", mode), GDNProjectionArithmetic::Native);
    EXPECT_EQ(Contract::effectiveGDNArithmetic(WeightRole::MoERouter,
        "blk.0.ssm_alpha.weight", "qwen35moe", mode), GDNProjectionArithmetic::Native);
}

/** @brief The default is explicitly MoE-scoped and retains original-source conversion guards. */
TEST(Test__GDNProjectionArithmetic, DefaultHalfMoEAlphaBetaPreservesDenseNative)
{
    using Contract = PreparedWeightRepresentationContract;
    constexpr auto mode = GDNProjectionArithmetic::FP16MoEAlphaBeta;
    EXPECT_EQ(kDefaultGDNProjectionArithmetic, mode);
    EXPECT_EQ(parseGDNProjectionArithmetic("fp16_moe_alpha_beta"), mode);
    EXPECT_EQ(gdnProjectionArithmeticName(mode), "fp16_moe_alpha_beta");
    EXPECT_EQ(Contract::effectiveGDNArithmetic(WeightRole::MoERouter,
        "blk.0.ssm_alpha.weight", "qwen35moe", mode), GDNProjectionArithmetic::Native);
    for (const auto architecture : {"qwen35", "qwen35moe", "llama"})
        for (const auto name : {"blk.0.ssm_alpha.weight", "blk.12.ssm_beta.weight",
                               "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight",
                               "blk.0.ffn_gate_inp.weight", "blk.0.ffn_up_exps.weight", "output.weight",
                               "blk.x.ssm_alpha.weight", "blk.0.ssm_beta.weight.extra"})
        {
            const auto role = inferWeightRole(name);
            const bool selected = std::string_view(architecture) == "qwen35moe" &&
                Contract::isGDNProjectionSource(role, name, architecture) &&
                role == WeightRole::GDNAlphaBetaProjection;
            EXPECT_EQ(Contract::effectiveGDNArithmetic(role, name, architecture, mode),
                selected ? GDNProjectionArithmetic::FP16 : GDNProjectionArithmetic::Native);
            EXPECT_EQ(Contract::operandPrecision(role, name, architecture, mode), selected ?
                FloatingActivationOperandPrecision::FP16RNE : FloatingActivationOperandPrecision::Native);
            for (const auto type : {TensorType::FP32, TensorType::FP16, TensorType::BF16})
                EXPECT_EQ(Contract::resolve(role, type, name, architecture, mode), selected ?
                    (type == TensorType::FP16 ? ModelPreparedWeightRepresentation::SourceNative :
                        ModelPreparedWeightRepresentation::FP16) : Contract::resolve(role, type));
            for (const auto format : {"F32", "FP32", "F16", "FP16", "BF16"})
                EXPECT_EQ(Contract::resolve(role, format, name, architecture, mode), selected ?
                    (std::string_view(format) == "F16" || std::string_view(format) == "FP16" ?
                        ModelPreparedWeightRepresentation::SourceNative : ModelPreparedWeightRepresentation::FP16) :
                    Contract::resolve(role, format));
            for (const auto type : {TensorType::Q6_K, TensorType::Q8_0, TensorType::IQ3_S})
                if (selected) { EXPECT_THROW(Contract::resolve(role, type, name, architecture, mode), std::invalid_argument); }
                else { EXPECT_EQ(Contract::resolve(role, type, name, architecture, mode), Contract::resolve(role, type)); }
            for (const auto format : {"Q6_K", "Q8_0", "IQ3_S"})
                if (selected) { EXPECT_THROW(Contract::resolve(role, format, name, architecture, mode), std::invalid_argument); }
                else { EXPECT_EQ(Contract::resolve(role, format, name, architecture, mode), Contract::resolve(role, format)); }
        }
}

/** @brief The real loader prepares only alpha/beta and preserves source ownership and bytes. */
TEST(Test__GDNProjectionArithmetic, HalfAlphaBetaMaterializationLeavesWideQuantizationUntouched)
{
    struct Restore {
        GDNProjectionArithmetic previous = debugEnv().gemm.gdn_projection_arithmetic;
        ~Restore() { mutableDebugEnv().gemm.gdn_projection_arithmetic = previous; }
    } restore;
    mutableDebugEnv().gemm.gdn_projection_arithmetic = GDNProjectionArithmetic::FP16MoEAlphaBeta;
    std::shared_ptr<TensorBase> wide(TestTensorFactory::createQ6_KRandom({2, 256}, 9817));
    std::vector<uint8_t> original(wide->size_bytes());
    std::memcpy(original.data(), wide->raw_data(), original.size());
    auto loader = MockModelLoaderBuilder().setArchitecture("qwen35moe")
        .addTensor("blk.0.attn_qkv.weight", wide).addTensor("blk.0.attn_gate.weight", wide)
        .addTensor("blk.0.ssm_out.weight", wide).addTensor("blk.0.ffn_up.weight", wide)
        .addFP32RandomTensor("blk.0.ssm_alpha.weight", {2, 256})
        .addFP32RandomTensor("blk.0.ssm_beta.weight", {2, 256}).build();
    WeightManager manager(*loader);
    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::SingleDevice;
    strategy.model_id = ModelContextId{18729};
    strategy.devices = {DeviceId::cpu()};
    WeightPlan plan(strategy);
    for (const auto name : {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight",
                           "blk.0.ffn_up.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"})
    {
        WeightRequirement requirement;
        requirement.canonical_name = name;
        requirement.target_device = DeviceId::cpu();
        requirement.lookup_device = DeviceId::cpu();
        requirement.host_policy = WeightHostPolicy::RequiredUntilGraphMaterialized;
        plan.add(requirement);
    }
    const auto frozen = manager.materialize(plan);
    for (const auto suffix : {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "ffn_up.weight"})
    {
        const auto &binding = frozen.layer(0, suffix);
        EXPECT_EQ(binding.tensor->native_type(), TensorType::Q6_K);
        EXPECT_EQ(binding.floating_operand_precision, FloatingActivationOperandPrecision::Native);
        EXPECT_EQ(std::memcmp(binding.tensor->raw_data(), original.data(), original.size()), 0);
    }
    for (const auto suffix : {"ssm_alpha.weight", "ssm_beta.weight"})
    {
        const auto &binding = frozen.layer(0, suffix);
        const auto source = loader->loadTensor(std::string("blk.0.") + suffix);
        ASSERT_EQ(binding.tensor->native_type(), TensorType::FP16);
        ASSERT_EQ(source->native_type(), TensorType::FP32);
        EXPECT_EQ(binding.floating_operand_precision, FloatingActivationOperandPrecision::FP16RNE);
        EXPECT_NE(binding.tensor, source.get());
        EXPECT_EQ(binding.tensor->size_bytes() * 2, source->size_bytes());
        const auto *words = static_cast<const uint16_t *>(binding.tensor->raw_data());
        for (size_t i = 0; i < source->numel(); ++i)
            EXPECT_EQ(words[i], canonicalPreparedHalfBits(source->data()[i]));
    }
    EXPECT_EQ(std::memcmp(wide->raw_data(), original.data(), original.size()), 0);
}

/** @test Explicit MoE alpha/beta precision includes the exact Flash Next source family. */
TEST(Test__GDNProjectionArithmetic, FlashNextMoESelectivePolicyMatchesSourceContract)
{
    using Contract = PreparedWeightRepresentationContract;
    EXPECT_EQ(Contract::effectiveGDNArithmetic(WeightRole::GDNAlphaBetaProjection,
        "blk.0.ssm_alpha.weight", "qwen4exp", GDNProjectionArithmetic::FP16MoEAlphaBeta),
        GDNProjectionArithmetic::FP16);
    EXPECT_EQ(Contract::effectiveGDNArithmetic(WeightRole::GDNProjection,
        "blk.0.attn_gate.weight", "qwen4exp", GDNProjectionArithmetic::FP16MoEAlphaBeta),
        GDNProjectionArithmetic::Native);
}

/** @brief Only the registered QFN source schema changes its omitted router preparation default. */
TEST(Test__MoERouterArithmetic, QFNDefaultPlanningAndMaterializationUseOneResolvedPolicy)
{
    using Contract = PreparedWeightRepresentationContract;
    constexpr auto role = WeightRole::MoERouter;
    constexpr auto name = "blk.0.ffn_gate_inp.weight";
    constexpr auto gdn = kDefaultGDNProjectionArithmetic;
    const auto omitted = MoERouterArithmetic::ModelDefault;
    for (const auto architecture : {"qwen4exp", "qwen35moe", "qwen35", "llama"})
    {
        SCOPED_TRACE(architecture);
        const auto expected = architecture == std::string_view("qwen4exp") ?
            MoERouterArithmetic::FP16 : MoERouterArithmetic::Native;
        for (const auto &[type, format] : std::initializer_list<std::pair<TensorType, const char *>>{
                 {TensorType::FP32, "F32"}, {TensorType::FP16, "F16"},
                 {TensorType::BF16, "BF16"}, {TensorType::Q8_0, "Q8_0"}})
        {
            SCOPED_TRACE(format);
            EXPECT_EQ(Contract::resolve(role, type, name, architecture, gdn, omitted),
                      Contract::resolve(role, type, name, architecture, gdn, expected));
            EXPECT_EQ(Contract::resolve(role, format, name, architecture, gdn, omitted),
                      Contract::resolve(role, type, name, architecture, gdn, expected));
            EXPECT_EQ(Contract::operandPrecision(role, name, architecture, gdn, omitted),
                      expected == MoERouterArithmetic::FP16 ? FloatingActivationOperandPrecision::FP16RNE :
                          FloatingActivationOperandPrecision::Native);
            EXPECT_EQ(Contract::resolve(role, type, name, architecture, gdn, MoERouterArithmetic::Native),
                      Contract::resolve(role, type));
        }
        for (const auto source : {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                                  "blk.0.ffn_down_exps.weight", "output.weight"})
        {
            const auto other_role = inferWeightRole(source);
            EXPECT_EQ(Contract::resolve(other_role, TensorType::Q8_0, source, architecture, gdn, omitted),
                      Contract::resolve(other_role, TensorType::Q8_0));
            EXPECT_EQ(Contract::operandPrecision(other_role, source, architecture, gdn, omitted),
                      FloatingActivationOperandPrecision::Native);
        }
    }
    EXPECT_EQ(Contract::resolve(role, TensorType::FP32, name, "qwen4exp", gdn),
              ModelPreparedWeightRepresentation::FP16);
    EXPECT_EQ(Contract::operandPrecision(role, name, "qwen4exp", gdn),
              FloatingActivationOperandPrecision::FP16RNE);
    EXPECT_THROW(Contract::resolve(role, TensorType::FP32, name, "qwen4exp", gdn,
                                  static_cast<MoERouterArithmetic>(255)), std::invalid_argument);
}

/**
 * @test Materialization seals every predictor projection before prepared adoption.
 * @details Metadata-only existing handles reproduce bulk preparation without
 * loading a GPU or executing GEMM. The real WeightManager path must publish
 * identical frozen source48/device identities to both graph and prepared store.
 */
TEST(Test__WeightManagerMaterialize, PreparedAdoptionPublishesImmutablePredictorSources)
{
    using Factory = llaminar::v2::kernels::KernelFactory;
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
        for (const std::string suffix : {"hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
                 "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight", "nextn.hc_head_down.weight",
                 "nextn.hc_head_up.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                 "nextn.eh_proj.weight", "output.weight"})
        {
            const std::string name = suffix == "output.weight" ? suffix : "blk.48." + suffix;
            SCOPED_TRACE(::testing::Message() << name << " device=" << device.toString());
            auto loader = MockModelLoaderBuilder().addFP32RandomTensor(name, {8, 8}).build();
            WeightManager manager(*loader);
            auto store = std::make_shared<PreparedWeightStore>(ModelContextId{1441});
            manager.setPreparedWeightStore(store);
            const auto kind = device.is_cpu() ? PreparedWeightKind::CpuPackedGemm :
                device.is_cuda() ? PreparedWeightKind::CudaInt8PackedGemm : PreparedWeightKind::RocmInt8PackedGemm;
            InferenceStrategy strategy;
            strategy.mode = WeightInferenceMode::SingleDevice;
            strategy.model_id = ModelContextId{1441};
            strategy.devices = {device};
            WeightPlan plan(strategy);
            WeightRequirement requirement;
            requirement.canonical_name = name;
            requirement.layer = inferWeightLayer(name);
            requirement.target_device = device;
            requirement.lookup_device = DeviceId::cpu();
            requirement.expected_prepared_kind = kind;
            plan.add(requirement);
            const auto first = manager.materialize(plan);
            ASSERT_EQ(first.bindings().size(), 1u);
            auto bulk = first.bindings().front();
            bulk.binding_id = bulk.binding_id == 1 ? 2 : 1;
            bulk.identity.instance_id = bulk.binding_id;
            bulk.prepared.reset();
            auto handle = std::make_shared<Factory::PreparedGemmHandle>();
            handle->tensor = bulk.tensor;
            handle->device_id = device;
            handle->kind = Factory::GemmPreparationKind::FLOATING_POINT;
            handle->prepared_weights = std::make_shared<Factory::PreparedGemmWeights>();
            const auto bulk_ref = store->registerPreparedGemmHandle(bulk, kind, device, handle);
            const auto frozen = manager.materialize(plan);
            ASSERT_EQ(frozen.bindings().size(), 1u);
            const auto &binding = frozen.bindings().front();
            ASSERT_TRUE(binding.prepared.has_value());
            ASSERT_NE(binding.binding_id, bulk.binding_id);
            ASSERT_TRUE(store->contains(*binding.prepared));
            const auto stored = store->binding(*binding.prepared);
            ASSERT_TRUE(stored.has_value());
            EXPECT_TRUE(binding.immutable);
            EXPECT_TRUE(stored->immutable) << "Prepared adoption must not retain a pre-freeze copy";
            EXPECT_EQ(stored->binding_id, binding.binding_id);
            EXPECT_EQ(stored->identity, binding.identity);
            EXPECT_EQ(stored->identity.layer, inferWeightLayer(name));
            EXPECT_EQ(stored->tensor, binding.tensor);
            EXPECT_EQ(stored->residency.home_device, device);
            EXPECT_EQ(binding.prepared->device, device);
            EXPECT_TRUE(store->binding(bulk_ref)->immutable);
        }
}
