#include "../../utils/TestCPUPreparedLayoutInventory.h"
/**
 * @file CPUTensorTP1PreparedStaticReplay.inc
 * @brief Replay explicit CPU TensorShardedTP1 through real source preparation and Static setup.
 *
 * Reuses the original tensor-directory fixture and actual producer bodies.
 * A degree-one CPU policy has complete matrix ownership; no cohort, fake
 * fragment payload, new context or GPU/native readiness is manufactured.
 */
TEST(CPUTensorTP1PreparedStatic, OriginalSourceAndStaticSetupRetainWholeMatrixPolicy) {
 using namespace llaminar2;
 const auto source=staticJointOriginal();
 auto plan=*source.plan;
 plan.domains[1].participants.resize(1);plan.domains[1].world_ranks.resize(1);
 ASSERT_EQ(plan.domains[1].routed_compute_policy,RoutedExpertComputePolicy::TensorSharded);
 ASSERT_FALSE(MoEExpertOwnerMap::usesCPUJointFragments(plan));
 const auto runtime=resolveMoEExpertOverlayRuntimePlan(std::make_shared<const MoERoutedExpertPlacementPlan>(plan),
  {.current_world_rank=0,.validate_mvp_root_reachability=false});
 const auto prepared=MoEExpertOverlayPreparationPlan::build(*runtime,*source.loader);
 EXPECT_FALSE(prepared.cpuJointOwnerMap());
 const auto cpu=prepared.filteredForDevice(DeviceId::cpu());
 ASSERT_EQ(cpu.requests().size(),448u*3);
 for(const auto& request:cpu.requests()){EXPECT_FALSE(request.cpu_joint_source);EXPECT_FALSE(request.projection_ownership);}
 const auto weights=cpu.sourceWeightPlan(*source.loader,ModelContextId{175});
 ASSERT_EQ(weights.requirements().size(),3u);
 for(const auto& requirement:weights.requirements()){
  EXPECT_FALSE(requirement.cpu_joint_source);EXPECT_FALSE(requirement.projection_ownership);
  EXPECT_EQ(requirement.slice.expert_ids.size(),448u);
  EXPECT_EQ(requirement.slice.row_start,0u);EXPECT_EQ(requirement.slice.col_start,0u);
  EXPECT_EQ(requirement.slice.row_count,requirement.slice.source_rows);
  EXPECT_EQ(requirement.slice.col_count,requirement.slice.source_cols);
 }
 const auto manifest=jointManifest();
 const auto footprint=MoEOverlayCapacityResolver::preparedFootprints({manifest}, DeviceMoEProjectionSet::CompleteExpert, cpu::native_quantised::kHostPreparedLayout).front();
 const auto budget=[](const char* name,DeviceId device,size_t bytes){
  PhysicalMemoryBOMBuilder b({0,device,bytes,bytes});
  return MoEOverlayBoundPhysicalMemoryBudget(name,PhysicalMemoryAdmissionCertificate(b.build()));
 };
 const MoEOverlayCapacityAdmissionPolicy storage{.overlay_world_size=1};
 const auto observed=test::cpuUniformCanonicalInventory({0});
 const auto capacity=MoEOverlayCapacityAdmission::resolveCapacity(plan,512,{manifest},
  {budget("gpu",DeviceId::cuda(0),64*footprint.gpu_live_bytes),budget("cpu",DeviceId::cpu(),448*footprint.cpu_live_bytes)},storage,&observed);
 const auto frozen=MoEOverlayCapacityResolver::installResolvedQuotas(plan,capacity);
 const auto memory=std::make_shared<PhysicalMemoryAuthority>(capacity.physical_memory_admission,0);
 const MoERebalanceRuntimeConfig maintenance{.mode=MoERebalanceRuntimeMode::Off};
 const auto setup=MoEOverlayResidencySetup::create({.plan=frozen,.metadata=staticJointConfig(source).model_metadata,
  .loader=*source.loader,.rebalance=maintenance,.storage_policy=storage,.capacity=capacity,.memory=memory,
  .world_rank=0,.world_size=1});
 EXPECT_EQ(setup.authority()->snapshot()->ownershipKind(),MoEOverlayResidencyOwnershipKind::WholeExperts);
 EXPECT_FALSE(setup.residency()->projectionPreparation());
 ASSERT_EQ(setup.residency()->localParticipantIds().size(),2u);
 for(const auto pid:setup.residency()->localParticipantIds())EXPECT_FALSE(setup.residency()->isCPUJointParticipant(pid));
 EXPECT_FALSE(setup.residency()->allInitialBanksReady());
}
