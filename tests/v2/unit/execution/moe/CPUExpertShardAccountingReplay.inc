/**
 * @file CPUExpertShardAccountingReplay.inc
 * @brief Actual source/cohort/placement/PMA replays without native readiness.
 *
 * Each logical expert costs one native fragment on every original CPU socket.
 * Tests retain the original whole-expert rejection at execution accessors and
 * check that a partial/foreign cohort cannot obtain a cheap capacity plan.
 */
#include "execution/moe/MoEExpertShardCohort.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "loaders/MoEExpertSourceView.h"
#include "kernels/common/NativeEmbeddingLayout.h"
#include "utils/QuantizedVerifierFormats.h"
#include "../../../utils/TestCPUPreparedLayoutInventory.h"
namespace llaminar2 {
namespace {
/** @return Original compatible gate/up/down source, with an explicit PP layer offset. */
MoEOverlayLayerWeightManifest cpuShardSource(ExpertWeightFormat format,int layer=3,int intermediate=512){
 return {.layer_idx=layer,.projections={{{ExpertTierWeightProjection::Gate,intermediate,256,format},
 {ExpertTierWeightProjection::Up,intermediate,256,format},{ExpertTierWeightProjection::Down,256,intermediate,format}}}};
}
/** @return Exact original two-socket membership; both native CPU ordinals are zero. */
std::vector<MoEExpertShardCohort::Participant> cpuShardMembers(){return {{0,0,GlobalDeviceAddress::cpu(0)},{1,1,GlobalDeviceAddress::cpu(1)}};}
/** @return Original typed CPU tensor-sharded domain, not complete-expert apportionment. */
MoERoutedExpertPlacementPlan cpuShardPlan(int experts=4){
 MoERoutedExpertPlacementPlan p;p.enabled=true;p.first_model_layer=3;p.topology=RoutedExpertPlacementTopology::TieredOverlay;
 p.continuation_domain=p.base_model_domain=p.shared_expert_domain="cpu_joint";
 RoutedExpertDomain d;d.name="cpu_joint";d.scope=ExecutionDomainScope::NODE_LOCAL;d.backend=CollectiveBackendType::UPI;
 d.participants={GlobalDeviceAddress::cpu(0),GlobalDeviceAddress::cpu(1)};d.world_ranks={0,1};d.owner_rank=-1;
 d.routed_compute_policy=RoutedExpertComputePolicy::TensorSharded;d.routed_phase_policy=RoutedExpertPhasePolicy::Uniform;
 d.routed_decode_assignment_policy=d.routed_prefill_assignment_policy=RoutedExpertAssignmentPolicy::StaticOwner;p.domains={d};
 p.routed_tiers={{.name="joint",.domain=d.name,.priority=0,.max_experts_per_layer=experts,.fallback=true}};
 p.placements={{.layer=3,.routed_expert_tier=std::vector<int>(experts,0)}};
 return p;
}
/** @return Device-free live inventory for this plan's two CPU sockets. */
ClusterInventory cpuShardInventory(){
 return test::cpuUniformCanonicalInventory({0,1});
}
/** @return Original PMA resource views with an exact requested capacity, no anonymous reserve. */
std::vector<MoEOverlayBoundPhysicalMemoryBudget> cpuShardBudgets(std::size_t first,std::size_t second){
 return {boundBudget(0,physicalBudget("cpu-socket0",DeviceId::cpu(),first,0,0)),
         boundBudget(1,physicalBudget("cpu-socket1",DeviceId::cpu(),second,0,0))};
}
}
/** @test All source formats retain exact original N/K axes and native block groups. */
TEST(CPUExpertShardAccounting, OriginalFormatsSlicesAndSocketIdentity){
 std::vector<ExpertWeightFormat> formats;
 for(const auto& f:native_quantised_formats::kAllSourceFormats)formats.push_back(ExpertWeightFormat::nativeQuantised({f.metadata->codebook_id,f.metadata->is_superblock,true}));
 for(const auto f:{TensorType::FP16,TensorType::BF16,TensorType::FP32})formats.push_back(ExpertWeightFormat::floating(f));
 for(const auto& f:formats){
  const auto source=cpuShardSource(f);const auto cohort=MoEExpertShardCohort::checked(source,6,cpuShardMembers(),{{0,256},{256,256}});
  EXPECT_EQ(cohort.source(),source);ASSERT_EQ(cohort.participants().size(),2u);
  for(std::size_t p=0;p<2;++p){
   const auto& layout=cohort.ownership(p);EXPECT_EQ(layout.geometry().experts,6);EXPECT_EQ(layout.participant(),static_cast<int>(p));
   const auto local=cohort.localManifest(p);EXPECT_EQ(local.projections[0].N,256);EXPECT_EQ(local.projections[2].K,256);EXPECT_EQ(local.projections[2].N,256);
   const auto request=cohort.sourceLoadSlice(p,WeightRole::MoEExpertDown,{0,2,5});EXPECT_FALSE(request.inner_is_presliced);
   const auto down=cohort.sourceSlice(p,WeightRole::MoEExpertDown,{0,2,5});EXPECT_TRUE(down.inner_is_presliced);EXPECT_EQ(down.source_rows,512u);EXPECT_EQ(down.row_start,p*256);EXPECT_EQ(down.row_count,256u);
   const std::array<std::size_t,3> shape{256,256,3};MoEExpertSourceView view(shape,down);
   const auto matrix=view.matrix(5,WeightRole::MoEExpertDown,&layout);EXPECT_EQ(matrix.rows,256u);EXPECT_EQ(matrix.columns,256u);EXPECT_EQ(matrix.source_identity.col_start,p*256);
   const std::array<std::size_t,3> full_shape{512,256,6};MoEExpertSourceView full_parent(full_shape,{});
   EXPECT_THROW(full_parent.matrix(5,WeightRole::MoEExpertDown,&layout),std::invalid_argument);
   EXPECT_TRUE(cohort.participants()[p].address.toLocalDeviceId().is_cpu());EXPECT_EQ(cohort.participants()[p].address.device_ordinal,0);
   EXPECT_THROW(cohort.sourceSlice(p,WeightRole::MoEExpertDown,{2,2}),std::invalid_argument);
  }
 }
 for(const auto& registered:test::quantizedVerifierFormats()){
  const auto source_format=ExpertWeightFormat::nativeQuantised({registered.source_codebook_id,registered.source_is_superblock,true});
  const auto native_layout=NativeEmbeddingLayout::forSource(registered.tensor_type,1,768);
  const auto block=static_cast<int>(native_layout.block_elements);
  const auto cuts=MoEExpertShardCohort::balancedIntervals(cpuShardSource(source_format,3,3*block),2);
  EXPECT_EQ(cuts,(std::vector<MoEExpertShardCohort::Interval>{{0,2*block},{2*block,block}}));
 }
 const auto native=cpuShardSource(ExpertWeightFormat::nativeQuantised({native_quantised_formats::Q4_K.codebook_id,true,true}));
 EXPECT_THROW(MoEExpertShardCohort::checked(native,6,cpuShardMembers(),{{0,128},{128,384}}),std::invalid_argument);
 auto bad=cpuShardMembers();bad[1].address=GlobalDeviceAddress::cpu(0);
 EXPECT_THROW(MoEExpertShardCohort::checked(native,6,bad,{{0,256},{256,256}}),std::invalid_argument);
 bad=cpuShardMembers();bad[1].world_rank=0;
 EXPECT_THROW(MoEExpertShardCohort::checked(native,6,bad,{{0,256},{256,256}}),std::invalid_argument);
 EXPECT_THROW(MoEExpertShardCohort::checked(native,6,cpuShardMembers(),{{0,256}}),std::invalid_argument);
 auto duplicate_role=native;duplicate_role.projections[1].projection=ExpertTierWeightProjection::Gate;
 EXPECT_THROW(MoEExpertShardCohort::checked(duplicate_role,6,cpuShardMembers(),{{0,256},{256,256}}),std::invalid_argument);
 auto aliases=cpuShardMembers();aliases[1].address.hostname="bound-node-alias";
 EXPECT_NO_THROW(MoEExpertShardCohort::checked(native,6,aliases,{{0,256},{256,256}}));
 const auto original_axis=MoEExpertShardCohort::checked(native,512,cpuShardMembers(),{{0,256},{256,256}});
 std::vector<int> cpu_tier;for(int e=64;e<512;++e)cpu_tier.push_back(e);
 const auto selected=original_axis.sourceLoadSlice(0,WeightRole::MoEExpertDown,cpu_tier);
 EXPECT_EQ(original_axis.ownership(0).geometry().experts,512);EXPECT_EQ(selected.expert_count,448u);EXPECT_EQ(selected.expert_ids.back(),511);
 EXPECT_THROW(original_axis.sourceLoadSlice(0,WeightRole::MoEExpertDown,{512}),std::invalid_argument);
 EXPECT_THROW(MoEExpertShardCohort::balancedIntervals(cpuShardSource(native.projections[0].format,3,640),2),std::invalid_argument);
}
/** @test One logical owner row retains both exact fragments; whole-owner execution stays forbidden. */
TEST(CPUExpertShardAccounting, CanonicalPlacementNeverAliasesOneWholeOwner){
 const auto source=cpuShardSource(ExpertWeightFormat::floating(TensorType::FP32));auto plan=cpuShardPlan();
 EXPECT_THROW(MoEExpertOwnerMap::build(plan),std::invalid_argument);
 const auto before=MoEExpertOwnerMap::initialBuildAttempts();
 const auto placement=MoEExpertOwnerMap::buildShardPlacement(plan,{source});
 EXPECT_EQ(MoEExpertOwnerMap::initialBuildAttempts(),before+1);ASSERT_EQ(placement.owners().size(),4u);
 const auto* first=placement.placementFor(3,0);ASSERT_NE(first,nullptr);ASSERT_TRUE(first->shard_cohort);
 for(int e=0;e<4;++e){const auto* row=placement.placementFor(3,e);ASSERT_NE(row,nullptr);EXPECT_EQ(row->owner_participant,-1);EXPECT_FALSE(row->device.is_valid());EXPECT_EQ(row->shard_cohort,first->shard_cohort);EXPECT_EQ(placement.fragmentCountForExpert(3,e),2u);EXPECT_THROW(placement.ownerFor(3,e),std::logic_error);}
 for(int p=0;p<2;++p)EXPECT_EQ(placement.expertsForParticipant(3,p),(std::vector<int>{0,1,2,3}));
 EXPECT_EQ(placement.placementFor(0,0),nullptr);EXPECT_THROW(placement.layeredOwnership(1,4,3),std::logic_error);
 const auto rebuilt=MoEExpertOwnerMap::buildShardPlacement(plan,{source});EXPECT_EQ(*rebuilt.placementFor(3,0)->shard_cohort,*first->shard_cohort);
 EXPECT_NE(rebuilt.placementFor(3,0)->shard_cohort,first->shard_cohort);
 EXPECT_EQ(rebuilt.owners(),placement.owners());
 auto changed_interval=*first;changed_interval.shard_cohort=std::make_shared<const MoEExpertShardCohort>(
  MoEExpertShardCohort::checked(source,4,cpuShardMembers(),{{0,128},{128,384}}));
 EXPECT_NE(changed_interval,*first);
 auto changed_members=cpuShardMembers();changed_members[0].world_rank=4;changed_members[1].world_rank=5;
 auto changed_endpoint=*first;changed_endpoint.shard_cohort=std::make_shared<const MoEExpertShardCohort>(
  MoEExpertShardCohort::checked(source,4,changed_members,{{0,256},{256,256}}));
 EXPECT_NE(changed_endpoint,*first);
 EXPECT_THROW(MoEExpertOwnerMap::buildShardPlacement(plan,{}),std::invalid_argument);
 auto unresolved=plan;unresolved.domains[0].world_ranks.clear();
 EXPECT_THROW(MoEExpertOwnerMap::buildShardPlacement(unresolved,{source}),std::invalid_argument);
 auto foreign_layer=source;foreign_layer.layer_idx=4;
 EXPECT_THROW(MoEExpertOwnerMap::buildShardPlacement(plan,{source,foreign_layer}),std::invalid_argument);
 plan.domains[0].participants[1]=GlobalDeviceAddress::cpu(0);
 EXPECT_THROW(MoEExpertOwnerMap::buildShardPlacement(plan,{source}),std::invalid_argument);
}
/** @test Actual admission charges four jointly resident experts, not two apportionments or eight full copies. */
TEST(CPUExpertShardAccounting, JointLogicalQuotaUsesExactOriginalPMAFragments){
 const auto cpuShardObserved=cpuShardInventory();
 const auto format=ExpertWeightFormat::floating(TensorType::FP32);const auto source=cpuShardSource(format);
 const auto cohort=MoEExpertShardCohort::checked(source,4,cpuShardMembers(),{{0,256},{256,256}});
 const auto per=MoEOverlayCapacityResolver::preparedFootprints({cohort.localManifest(0)}).front().cpu_live_bytes;
 const auto plan=cpuShardPlan();const MoEOverlayCapacityAdmissionPolicy policy{.overlay_world_size=2};
 auto input=MoEOverlayCapacityAdmission::buildResolverInput(plan,4,{source},cpuShardBudgets(4*per,4*per),policy,&cpuShardObserved);
 const auto admitted=MoEOverlayCapacityResolver::resolve(input);const auto* tier=admitted.tier(0);ASSERT_NE(tier,nullptr);
 EXPECT_EQ(tier->live_experts_per_layer,(std::vector<int>{4}));ASSERT_EQ(tier->participant_live_copies.size(),2u);
 EXPECT_EQ(tier->participant_live_copies[0],(std::vector<int>{4}));EXPECT_EQ(tier->participant_live_copies[1],(std::vector<int>{4}));
 ASSERT_EQ(tier->shard_cohorts.size(),1u);EXPECT_EQ(*tier->shard_cohorts[0],cohort);
 EXPECT_EQ(admitted.resource("cpu-socket0")->usedBytes(),4*per);EXPECT_EQ(admitted.resource("cpu-socket1")->usedBytes(),4*per);
 EXPECT_THROW(MoEOverlayCapacityResolver::resolve(MoEOverlayCapacityAdmission::buildResolverInput(plan,4,{source},cpuShardBudgets(4*per,4*per-1),policy,&cpuShardObserved)),std::invalid_argument);
 auto partial=input;partial.tiers[0].shard_cohorts.clear();EXPECT_THROW(MoEOverlayCapacityResolver::resolve(partial),std::invalid_argument);
 auto foreign=input;foreign.tiers[0].participants[0].participant_id=99;EXPECT_THROW(MoEOverlayCapacityResolver::resolve(foreign),std::invalid_argument);
 auto wrong=input;wrong.tiers[0].copy_policy=MoEOverlayTierCopyPolicy::Replicated;EXPECT_THROW(MoEOverlayCapacityResolver::resolve(wrong),std::invalid_argument);
 const auto repeated=MoEOverlayCapacityResolver::resolve(input);EXPECT_EQ(repeated.resource("cpu-socket1")->usedBytes(),4*per);
}
} // namespace llaminar2

namespace llaminar2 {
/** @test New joint placement does not change the ordinary one-socket authority or PMA bytes. */
TEST(CPUExpertShardAccounting, SingleSocketKeepsOriginalWholeExpertAuthority){
 const auto cpuShardObserved=cpuShardInventory();
 auto plan=cpuShardPlan();auto& domain=plan.domains.front();domain.scope=ExecutionDomainScope::SINGLE;
 domain.backend=CollectiveBackendType::HOST;domain.participants.resize(1);domain.world_ranks={0};domain.owner_rank=0;
 domain.routed_compute_policy=RoutedExpertComputePolicy::Apportioned;
 const auto source=cpuShardSource(ExpertWeightFormat::floating(TensorType::FP32));
 const auto map=MoEExpertOwnerMap::build(plan);
 for(int e=0;e<4;++e){ASSERT_NE(map.ownerFor(3,e),nullptr);EXPECT_FALSE(map.ownerFor(3,e)->shard_cohort);EXPECT_EQ(map.fragmentCountForExpert(3,e),1u);}
 const auto per=MoEOverlayCapacityResolver::preparedFootprints({source}).front().cpu_live_bytes;
 const auto budget=boundBudget(0,physicalBudget("cpu-single",DeviceId::cpu(),4*per,0,0));
 const auto result=MoEOverlayCapacityAdmission::resolveCapacity(plan,4,{source},{budget},{},&cpuShardObserved);
 EXPECT_EQ(result.resource("cpu-single")->usedBytes(),4*per);EXPECT_TRUE(result.tier(0)->shard_cohorts.empty());
}
/** @test Original native superblock cuts may be unequal, and every actual fragment limits admission. */
TEST(CPUExpertShardAccounting, UnequalFragmentsCannotBorrowPeerCapacity){
 const auto cpuShardObserved=cpuShardInventory();
 const auto format=ExpertWeightFormat::nativeQuantised({native_quantised_formats::Q4_K.codebook_id,true,true});
 const auto source=cpuShardSource(format,3,768);const auto cuts=MoEExpertShardCohort::balancedIntervals(source,2);
 ASSERT_EQ(cuts,(std::vector<MoEExpertShardCohort::Interval>{{0,512},{512,256}}));
 const auto cohort=MoEExpertShardCohort::checked(source,4,cpuShardMembers(),cuts);
 const auto first=MoEOverlayCapacityResolver::preparedFootprints({cohort.localManifest(0)}).front().cpu_live_bytes;
 const auto second=MoEOverlayCapacityResolver::preparedFootprints({cohort.localManifest(1)}).front().cpu_live_bytes;
 ASSERT_GT(first,second);const auto plan=cpuShardPlan();const MoEOverlayCapacityAdmissionPolicy policy{.overlay_world_size=2};
 const auto result=MoEOverlayCapacityAdmission::resolveCapacity(plan,4,{source},cpuShardBudgets(4*first,4*second),policy,&cpuShardObserved);
 EXPECT_EQ(result.resource("cpu-socket0")->usedBytes(),4*first);EXPECT_EQ(result.resource("cpu-socket1")->usedBytes(),4*second);
 EXPECT_THROW(MoEOverlayCapacityAdmission::resolveCapacity(plan,4,{source},cpuShardBudgets(4*second,4*first),policy,&cpuShardObserved),std::invalid_argument);
}
} // namespace llaminar2

namespace llaminar2 {
/** @test Actual QFN source geometry retains E512 while GPU64/CPU448 logical quotas charge two CPU fragments. */
TEST(CPUExpertShardAccounting, QFNOriginal512AxisAnd448JointTierStayDistinct){
 auto source=cpuShardSource(ExpertWeightFormat::nativeQuantised({native_quantised_formats::Q4_K.codebook_id,true,true}),3,640);
 source.projections[0].K=source.projections[1].K=source.projections[2].N=2560;
 source.projections[2].format=ExpertWeightFormat::nativeQuantised({native_quantised_formats::Q5_1.codebook_id,false,true});
 const auto geometry=MoEExpertShardCohort::checked(source,512,cpuShardMembers(),{{0,320},{320,320}});
 const auto fragment=MoEOverlayCapacityResolver::preparedFootprints({geometry.localManifest(0)}).front().cpu_live_bytes;
 for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)}){
  auto plan=cpuShardPlan(448);plan.domains[0].name="cpu_cold";plan.routed_tiers[0].domain="cpu_cold";plan.routed_tiers[0].priority=10;
  auto gpu=boundDomain("gpu_hot",0,device.is_cuda()?GlobalDeviceAddress::cuda(0):GlobalDeviceAddress::rocm(0));
  plan.domains.insert(plan.domains.begin(),gpu);
  plan.continuation_domain=plan.base_model_domain=plan.shared_expert_domain=gpu.name;
  plan.routed_tiers.insert(plan.routed_tiers.begin(),{.name="hot",.domain=gpu.name,.priority=0,.max_experts_per_layer=64});
  plan.placements[0].routed_expert_tier.assign(512,1);std::fill_n(plan.placements[0].routed_expert_tier.begin(),64,0);
  auto budgets=cpuShardBudgets(448*fragment,448*fragment);
  const auto gpu_per=MoEOverlayCapacityResolver::preparedFootprints({source}).front().gpu_live_bytes;
  budgets.push_back(boundBudget(0,physicalBudget("gpu",device,64*gpu_per,0,0)));
  const MoEOverlayCapacityAdmissionPolicy policy{.overlay_world_size=2};
  const auto cpuShardObserved=cpuShardInventory();
  const auto result=MoEOverlayCapacityAdmission::resolveCapacity(plan,512,{source},budgets,policy,&cpuShardObserved);
  EXPECT_EQ(result.tier(0)->live_experts_per_layer,(std::vector<int>{64}));EXPECT_EQ(result.tier(1)->live_experts_per_layer,(std::vector<int>{448}));
  ASSERT_EQ(result.tier(1)->shard_cohorts.size(),1u);EXPECT_EQ(result.tier(1)->shard_cohorts[0]->ownership(0).geometry().experts,512);
  EXPECT_EQ(result.resource("cpu-socket0")->usedBytes(),448*fragment);EXPECT_EQ(result.resource("cpu-socket1")->usedBytes(),448*fragment);
  const auto placement=MoEExpertOwnerMap::buildShardPlacement(plan,{source});
  EXPECT_EQ(placement.owners().size(),512u);EXPECT_EQ(placement.expertsForParticipant(1,1).size(),0u);
  EXPECT_EQ(placement.expertsForParticipant(3,1).size(),448u);EXPECT_EQ(placement.expertsForParticipant(3,2).size(),448u);
  EXPECT_EQ(placement.fragmentCountForExpert(3,63),1u);EXPECT_EQ(placement.fragmentCountForExpert(3,64),2u);
 }
}
} // namespace llaminar2
