/**
 * @file Test__GPUWeightPlanPreparation.cpp
 * @brief Exact-backend preparation of frozen bindings without predeclared packed hints.
 *
 * These native tests retain the actual Manager/Store preparation assertions from
 * WeightPlan while keeping Unit device-free. Each selected backend must execute;
 * absent devices fail instead of silently selecting another vendor or skipping.
 */
#include "../../mocks/MockModelLoader.h"
#include "backends/BackendManager.h"
#include "execution/moe/ExpertPreparedMemoryGeometry.h"
#include "loaders/GPUVramPreflight.h"
#include "loaders/PreparedWeightRepresentationContract.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/WeightManager.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "tensors/TensorKernels.h"
#include "utils/DebugEnv.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <optional>

using namespace llaminar2;
using namespace llaminar2::test;

namespace {
/**
 * @brief Admit the fixture's actual final GPU matrices and maximum shared upload workspace.
 * @param loader Original in-memory source for each final floating matrix.
 * @param plan Complete original binding inventory for this one native participant.
 * @param device Actual selected backend and ordinal, never an automatic substitute.
 * @return One authority consumed by Manager and its retained Store.
 */
std::shared_ptr<PhysicalMemoryAuthority> sourceMemory(MockModelLoader &loader, const WeightPlan &plan, DeviceId device)
{
    size_t final_bytes = 0;
    size_t maximum_source = 0;
    for (const auto &requirement : plan.requirements()) {
        const auto source = loader.loadTensor(requirement.canonical_name);
        if (!source)
            throw std::invalid_argument("GPU source fixture requires its declared original matrix");
        const auto representation = PreparedWeightRepresentationContract::resolve(
            inferWeightRole(requirement.canonical_name), source->native_type(), requirement.canonical_name,
            loader.architecture(), debugEnv().gemm.gdn_projection_arithmetic);
        auto scalar = source->native_type();
        if (representation == ModelPreparedWeightRepresentation::FP32) scalar = TensorType::FP32;
        if (representation == ModelPreparedWeightRepresentation::FP16) scalar = TensorType::FP16;
        final_bytes += resolveExpertPreparedProjectionMemoryGeometry(source->rows(), source->cols(),
            ExpertWeightFormat::floating(scalar)).gpu_live_bytes;
        maximum_source = std::max(maximum_source, source->numel() * (scalar == TensorType::FP32 ? 4u : 2u));
    }
    const auto staging = resolveGPUWeightLoadMemoryGeometry(maximum_source, configuredGPUWeightLoadMemoryPolicy());
    const PhysicalMemoryResource host{.world_rank=0,.device=DeviceId::cpu(),
        .total_bytes=256u<<20,.admission_available_bytes=256u<<20};
    const PhysicalMemoryResource gpu{.world_rank=0,.device=device,
        .total_bytes=256u<<20,.admission_available_bytes=256u<<20};
    PhysicalMemoryPlanBuilder memory;
    memory.add(gpu,PhysicalMemoryOwner::PrimaryModelWeights,final_bytes)
        .add(gpu,PhysicalMemoryOwner::WeightLoadStaging,staging.staging_bytes)
        .add(host,PhysicalMemoryOwner::WeightLoadStaging,staging.host_staging_bytes);
    return std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(memory.build()),0);
}
/** @brief Required native participant; a missing compiled backend or device is a failure. */
class GPUWeightPlanPreparation : public ::testing::TestWithParam<DeviceId> {
protected:
    /** @brief Authenticate the exact backend before source population. */
    void SetUp() override {
        auto *backend = getBackendFor(GetParam());
        ASSERT_NE(backend,nullptr);
        ASSERT_GT(backend->deviceCount(),GetParam().ordinal);
    }
};
}

TEST_P(GPUWeightPlanPreparation, FrozenBindingWithoutPreparedHintIsPreparedByStructure)
{
    const std::optional<DeviceId> gpu = GetParam();

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
    requirement.lookup_device = *gpu;
    requirement.expected_prepared_kind = PreparedWeightKind::None;
    requirement.host_policy = WeightHostPolicy::RequiredUntilGraphMaterialized;
    plan.add(requirement);

    manager.installPhysicalMemoryAuthority(sourceMemory(*loader, plan, *gpu));
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

TEST_P(GPUWeightPlanPreparation, GdnAlphaBetaProjectionBindingsArePreparedByStructure)
{
    const std::optional<DeviceId> gpu = GetParam();

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
        requirement.lookup_device = *gpu;
        requirement.expected_prepared_kind = PreparedWeightKind::None;
        requirement.host_policy = WeightHostPolicy::RequiredUntilGraphMaterialized;
        plan.add(requirement);
    }

    manager.installPhysicalMemoryAuthority(sourceMemory(*loader, plan, *gpu));
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

/** @test GPU packing preserves the source-selected precision of each frozen GDN control. */
TEST_P(GPUWeightPlanPreparation, MixedMoEGDNControlsKeepOriginalOperandPolicyAfterPacking)
{
    const auto device = GetParam();
    auto loader = MockModelLoaderBuilder().setArchitecture("qwen35moe")
        .addQ8_0RandomTensor("blk.0.ssm_alpha.weight", {4, 256})
        .addFP32RandomTensor("blk.0.ssm_beta.weight", {4, 256}).build();
    WeightManager manager(*loader);
    manager.setWeightShardingConfig(WeightShardingConfig{});
    auto store = std::make_shared<PreparedWeightStore>(ModelContextId{911});
    manager.setPreparedWeightStore(store);
    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::SingleDevice;
    strategy.model_id = store->modelId();
    strategy.devices = {device};
    WeightPlan plan(strategy);
    for (const auto name : {"blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"})
    {
        WeightRequirement requirement;
        requirement.canonical_name = name;
        requirement.layer = 0;
        requirement.target_device = device;
        requirement.lookup_device = device;
        requirement.host_policy = WeightHostPolicy::RequiredUntilGraphMaterialized;
        plan.add(requirement);
    }
    manager.installPhysicalMemoryAuthority(sourceMemory(*loader, plan, device));
    const auto frozen = manager.materialize(plan);
    ASSERT_TRUE(manager.prepareWeightsForDevice(frozen, device, false));
    for (const auto suffix : {std::string_view{"ssm_alpha.weight"}, std::string_view{"ssm_beta.weight"}})
    {
        const bool quantized_original = suffix == "ssm_alpha.weight";
        const auto operand = quantized_original ? FloatingActivationOperandPrecision::Native
                                                : FloatingActivationOperandPrecision::FP16RNE;
        const auto &binding = frozen.layer(0, std::string(suffix));
        ASSERT_NE(binding.tensor, nullptr);
        EXPECT_EQ(binding.tensor->native_type(), quantized_original ? TensorType::FP32 : TensorType::FP16);
        EXPECT_EQ(binding.floating_operand_precision, operand);
        const auto prepared = store->preparedRefForBinding(binding.binding_id, device);
        ASSERT_TRUE(prepared);
        const auto engine = store->gemmKernelLifetime(*prepared);
        ASSERT_TRUE(engine);
        EXPECT_EQ(engine->floatingActivationOperandPrecision(), operand);
    }
}

#ifdef HAVE_CUDA
INSTANTIATE_TEST_SUITE_P(CUDA,GPUWeightPlanPreparation,::testing::Values(DeviceId::cuda(0)));
#endif
#ifdef HAVE_ROCM
INSTANTIATE_TEST_SUITE_P(ROCm,GPUWeightPlanPreparation,::testing::Values(DeviceId::rocm(0)));
#endif
