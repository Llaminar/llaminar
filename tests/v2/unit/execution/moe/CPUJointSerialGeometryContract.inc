/**
 * @file CPUJointSerialGeometryContract.inc
 * @brief Verify original source partition and serial storage contracts without native readiness.
 *
 * NUMA0..7 below are explicitly recorded declarations, not observed sockets.
 * Source E512 is unchanged;448 is a tier quota and never a matrix/expert axis.
 */
#include "memory/CPUJointExpertRouteStorageLayout.h"
#include "memory/CPUSerialExpertStorageLayout.h"
#include "collective/NativeCPUCollectiveProgram.h"
#include "collective/backends/ShmemSpinBackend.h"
#include "execution/moe/MoEExpertOwnerMap.h"
namespace llaminar2 {
namespace {
ExpertWeightFormat serialSourceFormat(const char* name) {
 const auto* source=native_vnni_formats::forQuantType(name);
 if(!source)throw std::invalid_argument("Missing original source format");
 return ExpertWeightFormat::nativeVnni({source->codebook_id,source->is_superblock,true});
}
MoEOverlayLayerWeightManifest serialSourceLayer(int layer,const char* gate,const char* down,int intermediate=640) {
 return {.layer_idx=layer,.projections={{{ExpertTierWeightProjection::Gate,intermediate,2560,serialSourceFormat(gate)},
  {ExpertTierWeightProjection::Up,intermediate,2560,serialSourceFormat(gate)},
  {ExpertTierWeightProjection::Down,2560,intermediate,serialSourceFormat(down)}}}};
}
std::vector<MoEExpertShardCohort::Participant> serialMembers(int count) {
 std::vector<MoEExpertShardCohort::Participant> result;
 for(int i=0;i<count;++i)result.push_back({i+1,i+1,GlobalDeviceAddress::cpu(i)});
 return result;
}
std::vector<MoEOverlayLayerWeightManifest> serialActualClasses() {
 return {serialSourceLayer(0,"Q4_K","Q5_1"),serialSourceLayer(1,"Q4_K","Q8_0"),serialSourceLayer(2,"Q5_K","Q8_0")};
}
}
TEST(CPUJointSerialGeometry, ActualMixedClassesRetainNativeUnequalPiecesAndNUMARootsThroughEight) {
 const auto layers=serialActualClasses();
 for(int count=2;count<=8;++count) {
  SCOPED_TRACE(count);
  const auto cohort=CPUJointExpertRouteStorageLayout::checkedSerialCohort(layers,{512,2560,640},serialMembers(count));
  int frontier=0;
  for(int member=0;member<count;++member) {
   SCOPED_TRACE(member);
   const auto projection=cohort->ownership(size_t(member)).projection(WeightRole::MoEExpertDown);
   EXPECT_EQ(projection.source_columns,640);EXPECT_EQ(projection.first_column,frontier);
   EXPECT_EQ(projection.columns%32,0);frontier+=projection.columns;
   CPUJointExpertRouteStorageLayout joint(cohort,size_t(member),8,8);
   CPUSerialExpertStorageEnvelope envelope(DeviceId::cpu(),8,1,2560,8,{8},1,8,joint.isRoot(),
    CPUSerialExpertFamilyPolicy::PublishedSchedule,joint.intermediate(),512,joint);
   const auto layout=envelope.admitted();
   ASSERT_TRUE(layout->grouped());EXPECT_EQ(layout->grouped()->expert_intermediate,projection.columns);
   EXPECT_EQ(layout->grouped()->num_experts,512);EXPECT_EQ(layout->grouped()->batch.placement().node(),member);
   EXPECT_EQ(layout->jointRoutes()->packed().placement().node(),member);
   EXPECT_EQ(bool(layout->canonical()),member==0);
   if(layout->canonical())EXPECT_EQ(layout->canonical()->placement().node(),member);
   for(const auto& family:layout->families()) {
    EXPECT_EQ(family.hidden.placement().node(),member);EXPECT_EQ(family.output.placement().node(),member);
    EXPECT_EQ(family.indices.placement().node(),member);EXPECT_EQ(family.weights.placement().node(),member);
   }
   EXPECT_EQ(bool(joint.ranges()),member==0);EXPECT_EQ(bool(joint.maps()),member==0);
   const size_t records=8u*8u*(member==0?size_t(count):1u);
   EXPECT_EQ(joint.packed().capacity(),1u+records*2561u);
   const auto smaller=envelope.project(3,1,retainedRawPrefillGraphBucketLadder({8},3,1,8));
   EXPECT_LE(smaller->physicalBytes(),layout->physicalBytes());
   EXPECT_LE(smaller->workspaceBytes(),layout->workspaceBytes());
  }
  EXPECT_EQ(frontier,640);
 }
}
TEST(CPUJointSerialGeometry, EveryNativeCodebookUsesOriginalBlockAuthorityAtFourAndEight) {
 std::vector<ExpertWeightFormat> formats;
 for(const auto& format:native_vnni_formats::kAllSourceFormats)
  formats.push_back(ExpertWeightFormat::nativeVnni({format.metadata->codebook_id,format.metadata->is_superblock,true}));
 for(const auto type:{TensorType::FP16,TensorType::BF16,TensorType::FP32})formats.push_back(ExpertWeightFormat::floating(type));
 for(const auto& f:formats)for(const int count:{2,4,8}) {
  const MoEOverlayLayerWeightManifest source{.layer_idx=4,.projections={{{ExpertTierWeightProjection::Gate,2048,2560,f},
   {ExpertTierWeightProjection::Up,2048,2560,f},{ExpertTierWeightProjection::Down,2560,2048,f}}}};
  const std::array layers{source};
  const auto cohort=CPUJointExpertRouteStorageLayout::checkedSerialCohort(layers,{512,2560,2048},serialMembers(count));
  int columns=0;for(int member=0;member<count;++member)columns+=cohort->ownership(size_t(member)).projection(WeightRole::MoEExpertDown).columns;
  EXPECT_EQ(columns,2048);
 }
}
TEST(CPUJointSerialGeometry, IncompleteForeignAndDifferentSerialGeometryRejectBeforeRoots) {
 auto layers=serialActualClasses();
 EXPECT_THROW((void)CPUJointExpertRouteStorageLayout::checkedSerialCohort({}, {512,2560,640},serialMembers(2)),std::invalid_argument);
 layers[1].layer_idx=0;
 EXPECT_THROW((void)CPUJointExpertRouteStorageLayout::checkedSerialCohort(layers,{512,2560,640},serialMembers(2)),std::invalid_argument);
 layers=serialActualClasses();layers[1]=serialSourceLayer(1,"Q4_K","Q8_0",512);
 EXPECT_THROW((void)CPUJointExpertRouteStorageLayout::checkedSerialCohort(layers,{512,2560,640},serialMembers(2)),std::invalid_argument);
 const std::array mixed{serialSourceLayer(0,"Q4_K","Q5_1",768),serialSourceLayer(1,"Q4_K","Q4_K",768)};
 EXPECT_THROW((void)CPUJointExpertRouteStorageLayout::checkedSerialCohort(mixed,{512,2560,768},serialMembers(2)),std::invalid_argument);
 auto members=serialMembers(4);members[3].address=GlobalDeviceAddress::cpu(0);
 EXPECT_THROW((void)CPUJointExpertRouteStorageLayout::checkedSerialCohort(serialActualClasses(),{512,2560,640},members),std::invalid_argument);
}
TEST(CPUJointSerialGeometry, DomainPoolAndCreatorArenaAreExactAndNonmemberHasOnlyPool) {
 for(const int count:{2,4,8})for(const auto backend:{CollectiveBackendType::UPI,CollectiveBackendType::MPI}) {
  std::vector<NativeCPUCollectiveProgram::Participant> participants;
  for(int i=0;i<count;++i)participants.push_back({i+1,GlobalDeviceAddress::cpu(i)});
  const auto program=NativeCPUCollectiveProgram::expertShardRoutePartials(1,backend,std::move(participants));
  PhysicalMemoryPlanBuilder builder;
  const auto arena=ShmemSpinArena::allocationGeometry(count);
  EXPECT_EQ(arena.logical_bytes,ShmemSpinArena::compute_size(count));EXPECT_GE(arena.mapped_bytes,arena.logical_bytes);
  if(count==2)EXPECT_EQ(arena.mapped_bytes,8392704u);
  for(int rank=0;rank<=count;++rank) {
   PhysicalMemoryResource cpu{rank,DeviceId::cpu(),1u<<30,1u<<30};
   NativeCPUCollectiveProgram::contributeProcessPoolBOM(builder,cpu);
   if(rank>0)program.contributeDomainBOM(builder,cpu,rank);
   if(rank==1 && backend==CollectiveBackendType::UPI)builder.add(cpu,PhysicalMemoryOwner::ActivationTransportStaging,arena.mapped_bytes);
  }
  const auto plan=builder.build();
  for(int rank=0;rank<=count;++rank) {
   const auto* cpu=plan.find({rank,DeviceId::cpu()});ASSERT_NE(cpu,nullptr);
   EXPECT_EQ(cpu->bytes(PhysicalMemoryOwner::NativeCPUCollectiveProcessPool),program.processPoolBytes());
   EXPECT_EQ(cpu->bytes(PhysicalMemoryOwner::NativeCPUCollectiveDomain),rank?program.domainBytes():0u);
   EXPECT_EQ(cpu->bytes(PhysicalMemoryOwner::ActivationTransportStaging),rank==1 && backend==CollectiveBackendType::UPI?arena.mapped_bytes:0u);
  }
 }
}
TEST(CPUJointSerialGeometry, TypedInvocationScratchUsesLocalFAndRejectsForeignOriginalNK) {
 ModelMemoryProfile profile;profile.architecture="qwen3moe";profile.n_layers=3;profile.d_model=2560;
 profile.expert_count=512;profile.expert_used_count=8;profile.expert_feed_forward_length=640;
 const auto layers=serialActualClasses();
 for(const auto& layer:layers)for(const auto& p:layer.projections) {
  const char* role=p.projection==ExpertTierWeightProjection::Gate?"gate":p.projection==ExpertTierWeightProjection::Up?"up":"down";
  const char* format=layer.layer_idx==2 && p.projection!=ExpertTierWeightProjection::Down?"Q5_K":
   p.projection==ExpertTierWeightProjection::Down?(layer.layer_idx==0?"Q5_1":"Q8_0"):"Q4_K";
  profile.tensors.push_back({.name="blk."+std::to_string(layer.layer_idx)+".ffn_"+role+"_exps.weight",
   .quant_type=format,.elements=size_t(p.N)*size_t(p.K)*512u,.K=size_t(p.K),.layer_index=layer.layer_idx});
 }
 const WorkspaceMemoryGeometry geometry{.device=DeviceId::cpu(),.device_compute_units=8,
  .cpu_execution=test::kSyntheticCPUExecutionGeometry,.resident_graph_rows=3,.first_layer=0,.last_layer=2};
 for(const int count:{2,4,8}) {
  const auto cohort=CPUJointExpertRouteStorageLayout::checkedSerialCohort(layers,{512,2560,640},serialMembers(count));
  for(int member=0;member<count;++member) {
   const auto& own=cohort->ownership(size_t(member));
   auto expected=cpuSwiGLUWorkspaceRequirements(24,own.projection(WeightRole::MoEExpertUp).rows);
   for(const auto& t:profile.tensors) {
    const auto p=own.projection(inferWeightRole(t.name));
    expected.merge(CPUProjectionWorkspaceContract::sourceNative(t.quant_type,{.rows=24,.n=p.rows,.k=p.columns,.workers=8,
     .execution=geometry.cpu_execution,.numerical_policy=CPUProjectionNumericalPolicy::GPUAlignedExpert}));
   }
   EXPECT_EQ(WorkspaceMemoryEstimator::estimateRoutedExpertParticipant(profile,geometry,own),expected.total_bytes_with_alignment());
  }
 }
 const auto quota_as_source=MoEExpertProjectionOwnership::withinExpertIntermediateSlices({448,2560,640},0,{{0,320},{320,320}});
 EXPECT_THROW((void)WorkspaceMemoryEstimator::estimateRoutedExpertParticipant(profile,geometry,quota_as_source),std::invalid_argument);
 const auto foreign=MoEExpertProjectionOwnership::withinExpertIntermediateSlices({512,2560,512},0,{{0,256},{256,256}});
 EXPECT_THROW((void)WorkspaceMemoryEstimator::estimateRoutedExpertParticipant(profile,geometry,foreign),std::invalid_argument);
 const auto own=CPUJointExpertRouteStorageLayout::checkedSerialCohort(layers,{512,2560,640},serialMembers(2))->ownership(0);
 profile.tensors.front().elements+=profile.tensors.front().K*512u;
 EXPECT_THROW((void)WorkspaceMemoryEstimator::estimateRoutedExpertParticipant(profile,geometry,own),std::invalid_argument);
}
} // namespace llaminar2
namespace llaminar2 {
TEST(CPUJointSerialGeometry, ExplicitCPUTensorTP1IsFullMatrixOrdinaryStorageWithUnchangedPolicy) {
 MoERoutedExpertPlacementPlan plan;plan.enabled=true;plan.topology=RoutedExpertPlacementTopology::TieredOverlay;
 plan.continuation_domain=plan.base_model_domain=plan.shared_expert_domain="gpu";
 RoutedExpertDomain gpu;gpu.name="gpu";gpu.scope=ExecutionDomainScope::SINGLE;gpu.owner_rank=0;
 gpu.participants={GlobalDeviceAddress::cuda(0)};gpu.routed_compute_policy=RoutedExpertComputePolicy::Apportioned;
 RoutedExpertDomain cpu;cpu.name="cpu";cpu.scope=ExecutionDomainScope::NODE_LOCAL;cpu.world_ranks={1};
 cpu.participants={GlobalDeviceAddress::cpu(1)};cpu.routed_compute_policy=RoutedExpertComputePolicy::TensorSharded;
 plan.domains={gpu,cpu};plan.routed_tiers={{.name="gpu",.domain="gpu",.priority=0,.max_experts_per_layer=64},
  {.name="cpu",.domain="cpu",.priority=1,.max_experts_per_layer=448,.fallback=true}};
 std::vector<int> tiers(512,1);std::fill_n(tiers.begin(),64,0);plan.placements={{.layer=0,.routed_expert_tier=std::move(tiers)}};
 EXPECT_FALSE(MoEExpertOwnerMap::usesCPUJointFragments(plan));
 EXPECT_EQ(plan.domains[1].routed_compute_policy,RoutedExpertComputePolicy::TensorSharded);
 const auto owners=MoEExpertOwnerMap::build(plan);
 int selected=0;for(const auto& row:owners.owners())if(row.device.is_cpu()){++selected;EXPECT_FALSE(row.shard_cohort);EXPECT_GE(row.owner_participant,0);}
 EXPECT_EQ(selected,448);
 const CPUSerialExpertStorageEnvelope ordinary(DeviceId::cpu(),8,1,2560,8,{8},1,8,true,
  CPUSerialExpertFamilyPolicy::PublishedSchedule,640,512);
 const auto layout=ordinary.admitted();EXPECT_FALSE(layout->jointRoutes());
 ASSERT_TRUE(layout->grouped());EXPECT_EQ(layout->grouped()->expert_intermediate,640);
 EXPECT_EQ(layout->grouped()->num_experts,512);EXPECT_EQ(layout->grouped()->batch.placement(),CPUStoragePlacement::local());
 for(const auto& family:layout->families())EXPECT_EQ(family.hidden.placement(),CPUStoragePlacement::local());
}
}

namespace llaminar2 {
/** @test Only the original root gains one mapped bank; all source-native roots remain identical. */
TEST(CPUJointSerialGeometry, OriginalJointRootPublishesOneMappedBankUnderExactControllerBOM) {
 const auto layers=serialActualClasses();
 for(const int count:{2,3,4,8})for(const auto consumer:{DeviceId::cuda(0),DeviceId::rocm(0)}) {
  auto members=serialMembers(count);for(int rank=0;rank<count;++rank)members[size_t(rank)].world_rank=rank;
  const auto cohort=CPUJointExpertRouteStorageLayout::checkedSerialCohort(layers,{512,2560,640},members);
  const CPUJointExpertRouteStorageLayout roots(cohort,0,8,8);
  auto envelope=std::make_shared<const CPUSerialExpertStorageEnvelope>(DeviceId::cpu(),8,1,2560,8,
   std::vector<int>{8},1,8,true,CPUSerialExpertFamilyPolicy::LocalPowerOfTwo,roots.intermediate(),512,roots);
  const auto local=envelope->admitted();ASSERT_TRUE(local->canonical());ASSERT_FALSE(local->mappedCanonical());
  const auto mapped=envelope->forController(consumer);ASSERT_TRUE(mapped->mappedCanonical());
  EXPECT_EQ(mapped->canonical(),local->canonical());EXPECT_EQ(mapped->grouped(),local->grouped());
  EXPECT_EQ(mapped->families(),local->families());EXPECT_EQ(mapped->jointRoutes(),local->jointRoutes());
  EXPECT_EQ(mapped->workspaceBytes(),local->workspaceBytes());
  const CPUMappedPublicationRoot publication(local->canonical()->capacity()*sizeof(float),DeviceId::cpu(),consumer);
  EXPECT_EQ(*mapped->mappedCanonical(),publication);
  EXPECT_EQ(mapped->physicalBytes(),local->physicalBytes()+publication.physicalBytes());
  auto admission=std::make_shared<CPUSerialExpertStorageAdmission>(CPUSerialExpertStorageAdmission{
   .allocator={0,DeviceId::cpu()},.participant_ids={members[0].participant_id},.controller_devices={consumer},.envelope=envelope});
  EXPECT_EQ(admission->physicalBytes(),mapped->physicalBytes());
  PhysicalMemoryPlanBuilder plan;const PhysicalMemoryResource cpu{0,DeviceId::cpu(),1ull<<34,1ull<<34};
  plan.add(cpu,PhysicalMemoryOwner::ActivationArena,admission->physicalBytes());
  const auto bom=plan.build();ASSERT_NE(bom.find({0,DeviceId::cpu()}),nullptr);
  EXPECT_EQ(bom.find({0,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::ActivationArena),local->physicalBytes()+publication.physicalBytes());
  const std::array<std::shared_ptr<const CPUSerialExpertStorageAdmission>,1> declarations{admission};
  EXPECT_EQ(&requireCPUSerialExpertStorageAdmission(declarations,{0,DeviceId::cpu()},members[0].participant_id,consumer),admission.get());
  EXPECT_THROW((void)requireCPUSerialExpertStorageAdmission(declarations,{0,DeviceId::cpu()},members[0].participant_id,
   consumer.is_cuda()?DeviceId::cuda(1):DeviceId::rocm(1)),std::invalid_argument);
  const auto smaller=envelope->projectLocal(3,1,consumer);
  EXPECT_LE(smaller->physicalBytes(),mapped->physicalBytes());EXPECT_LE(smaller->workspaceBytes(),mapped->workspaceBytes());
 }
}
/** @test Contributors retain only local partial records; missing metadata-only GPU return never creates another matrix. */
TEST(CPUJointSerialGeometry, JointContributorCannotDuplicateRootMappedPublication) {
 const auto layers=serialActualClasses();
 for(const int count:{2,3,4,8}) {
  const auto cohort=CPUJointExpertRouteStorageLayout::checkedSerialCohort(layers,{512,2560,640},serialMembers(count));
  for(int member=1;member<count;++member) {
   const CPUJointExpertRouteStorageLayout roots(cohort,size_t(member),8,8);
   CPUSerialExpertStorageEnvelope local(DeviceId::cpu(),8,1,2560,8,{8},1,8,false,
    CPUSerialExpertFamilyPolicy::LocalPowerOfTwo,roots.intermediate(),512,roots);
   const auto layout=local.forController(DeviceId::cpu());
   EXPECT_FALSE(layout->canonical());EXPECT_FALSE(layout->mappedCanonical());
   EXPECT_FALSE(layout->jointRoutes()->ranges());EXPECT_FALSE(layout->jointRoutes()->maps());EXPECT_FALSE(layout->jointRoutes()->required());
   EXPECT_EQ(layout->jointRoutes()->packed().capacity(),1u+8u*8u*2561u);
   for(const auto consumer:{DeviceId::cuda(0),DeviceId::rocm(0)}) {
    try{(void)local.forController(consumer);FAIL()<<"A contributor cannot own a duplicate mapped matrix";}
    catch(const std::invalid_argument& error){EXPECT_NE(std::string(error.what()).find("metadata-only ticket"),std::string::npos);}
   }
  }
  const CPUJointExpertRouteStorageLayout root(cohort,0,8,8);
  CPUSerialExpertStorageEnvelope remote(DeviceId::cpu(),8,1,2560,8,{8},1,8,true,
   CPUSerialExpertFamilyPolicy::PublishedSchedule,root.intermediate(),512,root);
  EXPECT_THROW((void)remote.forController(DeviceId::cuda(0)),std::invalid_argument);
 }
}
}

#include "kernels/cpu/gemm/CPUWeightPreparationGeometry.h"
namespace llaminar2 {
/** @test The common cap bounds native input plus simultaneous prepared-consumer scratch for every source format. */
TEST(CPUJointSerialGeometry, NativeSourceWindowAllFormatsUsesExactAllocatorBound) {
 EXPECT_EQ(configuredNativeWeightLoadStagingBudgetBytes(),gpuPerDeviceLoadStagingBudgetBytes(debugEnv().rocm.repack_budget_mb));
 const auto placement=CPUStoragePlacement::onNode(7); // Declaration only, no hardware claim.
 const size_t cap=4u<<20;
 for(const auto& registered:test::quantizedVerifierFormats()) {
  const auto type=registered.tensor_type;
  const auto original=NativeEmbeddingLayout::forSource(type,256,256);
  const auto request=quoteCPUWeightPreparation(type,256,256,WeightPrecision::NATIVE,CPUWeightConsumer::PreparedGemm);
  ASSERT_TRUE(request);
  EXPECT_EQ(request->native_bytes_per_unit*request->maximum_units,original.row_stride_bytes*256u);
  const auto geometry=resolveNativeWeightLoadStaging(*request,placement,cap);
  EXPECT_GT(geometry.maximum_units_per_window,0u);
  EXPECT_LE(geometry.maximum_units_per_window,request->maximum_units);
  EXPECT_EQ(geometry.native_bytes,request->native_bytes_per_unit*geometry.maximum_units_per_window);
  EXPECT_EQ(geometry.scratch_bytes,request->fixed_scratch_bytes+request->scratch_bytes_per_unit*geometry.maximum_units_per_window);
  EXPECT_EQ(geometry.physical_bytes,placement.physicalBytes<uint8_t>(geometry.scratch_offset+geometry.scratch_bytes));
  EXPECT_LE(geometry.physical_bytes,cap);
  if(geometry.maximum_units_per_window<request->maximum_units) {
   auto next=*request;next.maximum_units=geometry.maximum_units_per_window+1;
   EXPECT_GT(resolveNativeWeightLoadStaging(next,placement,std::nullopt).physical_bytes,cap);
  }
  auto unit=*request;unit.maximum_units=1;
  const auto small=resolveNativeWeightLoadStaging(unit,placement,cap);
  EXPECT_EQ(small.maximum_units_per_window,1u);
  EXPECT_THROW((void)resolveNativeWeightLoadStaging(*request,placement,small.physical_bytes-1),std::invalid_argument);
  EXPECT_EQ(resolveNativeWeightLoadStaging(*request,placement,std::nullopt).maximum_units_per_window,request->maximum_units);
  EXPECT_THROW((void)quoteCPUWeightPreparation(type,256,257,WeightPrecision::NATIVE,CPUWeightConsumer::PreparedGemm),std::invalid_argument);
 }
 for(const auto type:{TensorType::FP16,TensorType::BF16,TensorType::FP32})
  for(const auto consumer:{CPUWeightConsumer::PreparedGemm,CPUWeightConsumer::ResidentTensor})
   EXPECT_FALSE(quoteCPUWeightPreparation(type,256,256,WeightPrecision::NATIVE,consumer));
 EXPECT_THROW((void)resolveNativeWeightLoadStaging({0,0,0,1},placement,cap),std::invalid_argument);
 EXPECT_THROW((void)resolveNativeWeightLoadStaging({1,0,0,0},placement,cap),std::invalid_argument);
 const NativeWeightLoadStagingRequest huge{2,0,0,std::numeric_limits<size_t>::max()};
 EXPECT_THROW((void)resolveNativeWeightLoadStaging(huge,placement,std::nullopt),std::overflow_error);
 EXPECT_LE(resolveNativeWeightLoadStaging(huge,placement,cap).physical_bytes,cap);
}
/** @test Native FP32 final owners need no transient source root on either rank; GPU upload staging remains separate. */
TEST(CPUJointSerialGeometry, SourceWindowIsPrepricedOnRootAndFollowerWithoutChangingRoutedBytes) {
 auto profile=smallModelProfile();addRoutedExpertMetadata(profile,"F32");
 auto overlay=cpuShardPlan();overlay.first_model_layer=0;overlay.placements.clear();
 overlay.continuation_domain=overlay.base_model_domain=overlay.shared_expert_domain="dense";
 overlay.domains.push_back(boundDomain("dense",0,GlobalDeviceAddress::cpu()));
 const auto format=ExpertWeightFormat::floating(TensorType::FP32);
 std::vector<MoEOverlayLayerWeightManifest> manifest;
 for(int i=0;i<profile.n_layers;++i)manifest.push_back({.layer_idx=i,.projections={{{ExpertTierWeightProjection::Gate,96,64,format},
  {ExpertTierWeightProjection::Up,96,64,format},{ExpertTierWeightProjection::Down,64,96,format}}}});
 ClusterInventory cluster;cluster.world_size=2;cluster.ranks.resize(2);
 for(int rank=0;rank<2;++rank) {
  auto& inventory=cluster.ranks[rank];inventory.rank=rank;inventory.node_id=0;
  inventory.cpu.type=DeviceType::CPU;inventory.cpu.local_device_id=0;inventory.cpu.numa_node=rank;
  inventory.cpu.memory_bytes=inventory.cpu.free_memory_bytes=inventory.cpu_memory_bytes=4ull<<30;
  inventory.cpu_cores=inventory.cpu_worker_threads=8;inventory.cpu_execution=test::kSyntheticCPUExecutionGeometry;
 }
 for(int rank=0;rank<2;++rank) {
  RankExecutionPlan local;local.rank=rank;local.first_layer=0;local.last_layer=profile.n_layers-1;
  local.primary_device=GlobalDeviceAddress::cpu(rank);local.runtime.batch_size=1;local.runtime.max_seq_len=32;
  local.runtime.moe_rebalance.mode=MoERebalanceRuntimeMode::Off;
  local.runtime.prefix_cache.enabled=false;local.runtime.prefix_cache.storage_mode=PrefixCacheStorageMode::Disabled;
  const MoEOverlayGPUWeightLoadCapacityInput load{.policy={.staging_stream_count=1,.staging_budget_bytes=32768},.maximum_source_bytes=65536};
  const auto result=test::planWithRecordedHostServices({.model_profile=&profile,.rank_plan=&local,.overlay_plan=&overlay,
   .rank_inventory=&cluster.ranks[rank],.cluster_inventory=&cluster,
   .rank_execution_kind=rank?OverlayRankExecutionKind::ExpertOnlyFollower:OverlayRankExecutionKind::ContinuationAuthority,
   .resident_graph_rows=4,.activation_channel_row_capacity=4,.activation_graph_family_count=1,.gpu_weight_load=load,.graph_family=testGraphFamily(profile),.original_expert_manifest=manifest});
  const auto fixed=result.fixed_memory_plan.physicalPlan();const auto* cpu=fixed.find({rank,DeviceId::cpu()});ASSERT_NE(cpu,nullptr);
  EXPECT_EQ(cpu->bytes(PhysicalMemoryOwner::WeightLoadStaging),0u);
  EXPECT_EQ(cpu->bytes(PhysicalMemoryOwner::RoutedExpertWeights),0u);
  const auto localCPU=std::find_if(result.device_inputs.begin(),result.device_inputs.end(),[](const auto& input){return input.device.is_cpu();});
  ASSERT_NE(localCPU,result.device_inputs.end());
  auto wrongBacking=*localCPU;wrongBacking.weight_load_staging.device_bytes=4096;
  EXPECT_THROW((void)MemoryPlanner::plan(profile,{wrongBacking}),std::invalid_argument);
  auto missingSource=*localCPU;missingSource.cpu_serial_expert_storage.reset();
  EXPECT_THROW((void)MemoryPlanner::plan(profile,{missingSource}),std::invalid_argument);
  const std::vector<std::vector<DevicePlanConfig>> groups{{*localCPU,*localCPU},{*localCPU}};
  const auto contributions=MemoryPlanner::planGroups(profile,groups);
  const auto first=contributions[0].physicalPlan(),second=contributions[1].physicalPlan();
  EXPECT_EQ(first.find({rank,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::WeightLoadStaging),0u);
  EXPECT_EQ(second.find({rank,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::WeightLoadStaging),0u);
  auto inconsistent=*localCPU;inconsistent.weight_load_staging.host_bytes+=4096;
  EXPECT_THROW((void)MemoryPlanner::plan(profile,{*localCPU,inconsistent}),std::invalid_argument);
  auto gpu=*localCPU;gpu.device=DeviceId::cuda(0);gpu.cpu_serial_expert_storage.reset();
  gpu.kv_precision="fp16";gpu.device_compute_units=80;gpu.associated_host_memory=cpu->resource();
  const auto ring=resolveGPUWeightLoadMemoryGeometry(65536,{.staging_stream_count=1,.staging_budget_bytes=65536});
  gpu.ordinary_weight_load_geometry=ring;
  gpu.weight_load_staging={.device_bytes=ring.staging_bytes,.host_bytes=ring.host_staging_bytes};
  const auto concurrent=MemoryPlanner::plan(profile,{*localCPU,gpu}).physicalPlan();
  EXPECT_EQ(concurrent.find({rank,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::WeightLoadStaging),ring.host_staging_bytes);
 }
}
} // namespace llaminar2
