/**
 * @file CPUOverlaySourceOwnerContract.inc
 * @brief Original overlay sources keep their routed owner inside a mixed graph plan.
 *
 * The public preparation compiler supplies exact participant/expert identities.
 * A tensor-backed original reader performs real bounded CPU preparation; no
 * GPU readiness, MPI execution or prepacked test handle is manufactured.
 */
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
namespace {
/**
 * @brief Compile ordinary complete-expert source declarations with original overlay identity.
 * @param loader Original encoded expert source and metadata provider.
 * @param model Model identity retained by the mixed plan and prepared store.
 * @return The production compiler's source plan, owned by RoutedExpertWeights.
 */
WeightPlan cpuOverlayOwnerSources(IModelLoader &loader, ModelContextId model)
{
    auto source = std::make_shared<MoERoutedExpertPlacementPlan>();
    source->enabled = true;
    source->topology = RoutedExpertPlacementTopology::SingleDomain;
    source->continuation_domain = source->shared_expert_domain = "compute";
    source->residency_policy = RoutedExpertResidencyPolicy::StaticById;
    RoutedExpertDomain domain;
    domain.name = "compute";
    domain.scope = ExecutionDomainScope::RANK_LOCAL;
    domain.owner_rank = 0;
    domain.backend = CollectiveBackendType::UPI;
    domain.participants = {GlobalDeviceAddress::cpu(0)};
    source->domains = {domain};
    source->routed_tiers = {{.name="resident",.domain="compute",.priority=0}};
    source->placements = {{.layer=0,.routed_expert_tier={0,0,0}}};
    const auto runtime = resolveMoEExpertOverlayRuntimePlan(source);
    return MoEExpertOverlayPreparationPlan::build(*runtime,loader).sourceWeightPlan(loader,model);
}
/**
 * @brief Provide independent encoded source parents for real bounded preparation.
 * @return A tensor-backed loader with three native experts and one floating projection.
 */
std::shared_ptr<MockModelLoader> cpuOverlayOwnerLoader()
{
    auto loader = MockModelLoaderBuilder().setArchitecture("qwen35moe")
        .addFP32RandomTensor("blk.0.attn_q.weight",{3,256}).build();
    for (const auto suffix : {"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"}) {
        const bool down = std::string_view(suffix)=="ffn_down_exps.weight";
        const size_t n=down?256:64,k=down?64:256;
        const auto matrix=TestTensorFactory::createQ4_0Random({3*n,k},817);
        std::vector<uint8_t> native(matrix->size_bytes());
        std::memcpy(native.data(),matrix->raw_data(),native.size());
        loader->addTensor(std::string("blk.0.")+suffix,
            std::make_shared<Q4_0Tensor>(std::vector<size_t>{k,n,3},native));
    }
    return loader;
}
}
/** @brief Hosted-primary zero and full continuation quotas cannot absorb routed roots. */
TEST(Test__WeightPlan, CPUOverlaySourcesKeepRoutedOwnerInMixedGraphPlan)
{
    for (const bool continuation : {false,true}) {
        SCOPED_TRACE(continuation);
        auto loader=cpuOverlayOwnerLoader();WeightManager manager(*loader);
        InferenceStrategy strategy{.mode=WeightInferenceMode::SingleDevice,.model_id={29301},.devices={DeviceId::cpu()}};
        WeightPlan dense(strategy);
        if(continuation) {
            WeightRequirement requirement;
            requirement.canonical_name="blk.0.attn_q.weight";
            dense.add(requirement);
        }
        const auto sources=cpuOverlayOwnerSources(*loader,strategy.model_id);
        ASSERT_EQ(sources.physicalMemoryOwner(),PhysicalMemoryOwner::RoutedExpertWeights);
        WeightPlan combined(strategy,dense.physicalMemoryOwner());
        for(const auto &requirement:dense.requirements())combined.add(requirement);
        for(const auto &requirement:sources.requirements())combined.add(requirement);
        admitCPUWeightPlansForTest(manager,*loader,{&dense,&sources});
        const auto memory=manager.physicalMemoryAuthority();
        const auto primary=memory->plannedBytes(DeviceId::cpu(),PhysicalMemoryOwner::PrimaryModelWeights);
        EXPECT_EQ(primary,continuation?CPUStoragePlacement::local().physicalBytes<float>(3*256):0u);
        const auto routed=memory->plannedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights);
        ASSERT_GT(routed,0u);
        const auto frozen=manager.materialize(combined);
        EXPECT_EQ(frozen.bindings().size(),9u+size_t(continuation));
        // Graph preparation reuses the committed owners even though the mixed
        // frozen set still carries the continuation plan's default owner.
        const auto source_reads=loader->nativeSourceReadCallCount();
        ASSERT_TRUE(manager.prepareWeightsForDevice(frozen,DeviceId::cpu(),true));
        ASSERT_TRUE(manager.prepareWeightsForDevice(frozen,DeviceId::cpu(),true));
        EXPECT_EQ(loader->nativeSourceReadCallCount(),source_reads);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::PrimaryModelWeights,
            PhysicalMemoryMaterializationKind::NewAllocation),primary);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
            PhysicalMemoryMaterializationKind::NewAllocation),routed);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
        for(const auto &binding:frozen.bindings())if(isRoutedExpertRole(binding.identity.role)) {
            ASSERT_TRUE(binding.prepared);
            EXPECT_EQ(binding.identity.overlay_domain,"compute");
            EXPECT_EQ(binding.identity.overlay_participant_index,0);
            const auto engine=manager.preparedWeightStore()->gemmKernelLifetime(*binding.prepared);
            ASSERT_TRUE(engine);EXPECT_TRUE(engine->hasWeights());
        }
    }
}
/** @brief Spare primary quota cannot mask a missing routed declaration. */
TEST(Test__WeightPlan, CPUOverlayOwnerDoesNotFallBackToSparePrimaryQuota)
{
    auto loader=cpuOverlayOwnerLoader();WeightManager manager(*loader);
    const auto sources=cpuOverlayOwnerSources(*loader,ModelContextId{29302});
    WeightPlan misowned(sources.strategy(),PhysicalMemoryOwner::PrimaryModelWeights);
    for(const auto &requirement:sources.requirements())misowned.add(requirement);
    // Deliberately admit exactly these final roots on the wrong line. The
    // materializer must reject Routed allocation rather than spend that room.
    WeightManager quote_manager(*loader);
    admitCPUWeightPlansForTest(quote_manager,*loader,{&sources});
    const auto quote=quote_manager.physicalMemoryAuthority();
    const auto final_bytes=quote->plannedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights);
    const auto staging=quote->plannedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging);
    PhysicalMemoryPlanBuilder admission;
    const PhysicalMemoryResource cpu{0,DeviceId::cpu(),final_bytes+staging,final_bytes+staging};
    admission.add(cpu,PhysicalMemoryOwner::PrimaryModelWeights,final_bytes);
    admission.add(cpu,PhysicalMemoryOwner::WeightLoadStaging,staging);
    manager.installPhysicalMemoryAuthority(std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(admission.build()),0));
    const auto memory=manager.physicalMemoryAuthority();
    ASSERT_GT(memory->plannedBytes(DeviceId::cpu(),PhysicalMemoryOwner::PrimaryModelWeights),0u);
    ASSERT_EQ(memory->plannedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights),0u);
    EXPECT_THROW((void)manager.materialize(misowned),std::logic_error);
    for(const auto owner:{PhysicalMemoryOwner::PrimaryModelWeights,PhysicalMemoryOwner::RoutedExpertWeights,
        PhysicalMemoryOwner::WeightLoadStaging})
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),owner,PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
/** @brief Ordinary native experts and dense sidecars retain their explicit plan owner. */
TEST(Test__WeightPlan, CPUOrdinarySourcesKeepPrimaryAndAdditionalOwners)
{
    for(const auto owner:{PhysicalMemoryOwner::PrimaryModelWeights,PhysicalMemoryOwner::AdditionalModelWeights}) {
        SCOPED_TRACE(toString(owner));
        auto loader=cpuOverlayOwnerLoader();WeightManager manager(*loader);
        InferenceStrategy strategy{.mode=WeightInferenceMode::SingleDevice,.model_id={29303},.devices={DeviceId::cpu()}};
        WeightPlan plan(strategy,owner);
        WeightRequirement requirement;
        requirement.canonical_name="blk.0.ffn_gate_exps.weight";
        plan.add(requirement);
        requirement={};requirement.canonical_name="blk.0.attn_q.weight";plan.add(requirement);
        admitCPUWeightPlansForTest(manager,*loader,{&plan});
        const auto memory=manager.physicalMemoryAuthority();
        const auto frozen=manager.materialize(plan);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),owner,PhysicalMemoryMaterializationKind::NewAllocation),
            memory->plannedBytes(DeviceId::cpu(),owner));
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
}
/** @brief Domain labels alone cannot authenticate a routed source purpose. */
TEST(Test__WeightPlan, CPUOverlayOwnerRejectsForeignSourceIdentity)
{
    for(int defect=0;defect<3;++defect) {
        SCOPED_TRACE(defect);
        auto loader=cpuOverlayOwnerLoader();WeightManager manager(*loader);
        const auto sources=cpuOverlayOwnerSources(*loader,ModelContextId{29304});
        auto requirement=sources.requirements().front();
        if(defect==0)requirement.role=requirement.role==WeightRole::MoEExpertGate?WeightRole::MoEExpertUp:WeightRole::MoEExpertGate;
        if(defect==1)requirement.derivation=WeightDerivationKind::Source;
        if(defect==2)requirement.overlay_participant_index=-1;
        WeightPlan foreign(sources.strategy());foreign.add(requirement);
        admitCPUWeightPlansForTest(manager,*loader,{&sources});
        EXPECT_THROW((void)manager.materialize(foreign),std::invalid_argument);
        EXPECT_EQ(loader->nativeSourceReadCallCount(),0u);
        const auto memory=manager.physicalMemoryAuthority();
        for(const auto owner:{PhysicalMemoryOwner::PrimaryModelWeights,PhysicalMemoryOwner::RoutedExpertWeights,
            PhysicalMemoryOwner::WeightLoadStaging})
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),owner,PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
}
