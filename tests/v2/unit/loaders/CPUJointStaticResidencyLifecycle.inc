/**
 * @file CPUJointStaticResidencyLifecycle.inc
 * @brief Verify Static source-map, ticket and fragment publication lifecycles with production owners.
 *
 * Synthetic original GGUF data creates the same source preparation as runtime.
 * Metadata engine doubles test publication identity, not CPU2 arithmetic/NUMA
 * execution. Existing actual prepared-stage and two-rank gather tests remain
 * required; no whole-expert payload is manufactured from fragment triplets.
 * The retained-executor case uses real prepared CPU engines and repeated sparse packet execution.
 */
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "utils/CPUProjectionTestWorkspace.h"
#include "execution/moe/CanonicalMoERouteRecord.h"
#include "mocks/MockComputeStage.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "execution/moe/MoEOverlayResidencySetup.h"
#include "execution/moe/MoEOverlayPreparedGraphBank.h"
#include "execution/moe/MoECPUExpertWorkspaceSource.h"
#include "kernels/cpu/gemm/CPUWeightPreparationGeometry.h"
#include "../../utils/TestCPUPreparedLayoutInventory.h"
namespace {
/** @brief Retain both original declared plan and its single source-authenticated preparation. */
struct StaticJointOriginal {
 std::shared_ptr<const llaminar2::MoERoutedExpertPlacementPlan> plan;
 std::shared_ptr<const llaminar2::MoEExpertOverlayPreparationPlan> preparation;
 std::shared_ptr<JointOriginalFile> file;
 std::shared_ptr<llaminar2::MPIContext> mpi;
 std::shared_ptr<llaminar2::TensorFactory> factory;
 std::shared_ptr<llaminar2::ModelLoader> loader;
};
/** @brief Build actual tiny GGUF source tensors and the canonical source preparation.
 * @param rank Original metadata rank whose real local CPU backing executes this actor.
 * @param members Complete original CPU cohort width.
 * @param serial_layers Include distinct real layers six/seven for shared serial-root validation.
 * @return Actual source-aware preparation; no private selection or map constructor bypass.
 */
StaticJointOriginal staticJointOriginal(int rank=0,int members=2,bool serial_layers=false) {
 using namespace llaminar2;auto file=std::make_shared<JointOriginalFile>();
 {
  std::ofstream stream(file->path,std::ios::binary|std::ios::trunc);
  stream.write("GGUF",4);jointScalar(stream,uint32_t{3});jointScalar(stream,uint64_t(serial_layers?6:3));jointScalar(stream,uint64_t{3});
  jointString(stream,"general.architecture");jointScalar(stream,uint32_t{8});jointString(stream,"llama");
  jointString(stream,"llama.expert_count");jointScalar(stream,uint32_t{4});jointScalar(stream,uint32_t{512});
  jointString(stream,"llama.expert_used_count");jointScalar(stream,uint32_t{4});jointScalar(stream,uint32_t{4});
  for(int source=0;source<(serial_layers?2:1);++source)for(int role=0;role<3;++role) {
   jointString(stream,"blk."+std::to_string(serial_layers?6+source:7)+".ffn_"+std::string(role==0?"gate":role==1?"up":"down")+"_exps.weight");
   jointScalar(stream,uint32_t{3});jointScalar(stream,uint64_t(role==2?20:8));jointScalar(stream,uint64_t(role==2?8:20));
   jointScalar(stream,uint64_t{512});jointScalar(stream,uint32_t{0});jointScalar(stream,uint64_t(source*3+role)*512*20*8*sizeof(float));
  }
  const auto at=size_t(stream.tellp());std::vector<char> pad((at+31)/32*32-at,0);stream.write(pad.data(),std::streamsize(pad.size()));
  for(int source=0;source<(serial_layers?2:1);++source)for(int role=0;role<3;++role)for(int expert=0;expert<512;++expert)for(int i=0;i<160;++i)jointScalar(stream,float(expert+source));
  if(!stream)throw std::runtime_error("Static original source fixture write failed");
 }
 const auto mpi=std::make_shared<MPIContext>(0,1,MPI_COMM_NULL);
 const auto factory=std::make_shared<TensorFactory>(*mpi);const auto loader=std::make_shared<ModelLoader>(factory.get());
 if(!loader->loadModel(file->path.string()))throw std::runtime_error("Static original source fixture load failed");
 auto plan=jointSourcePlan();
 if(serial_layers) {
  auto first=plan.placements.front();first.layer=6;
  plan.placements.insert(plan.placements.begin(),std::move(first));plan.first_model_layer=6;
 }
 if(members<2 || members>8 || rank<0 || rank>=members)throw std::invalid_argument("Original CPU cohort fixture requires a supported exact member");
 for(auto& domain:plan.domains)if(domain.name=="cpu_joint") {
  domain.participants.clear();domain.world_ranks.clear();
  for(int member=0;member<members;++member){domain.participants.push_back(GlobalDeviceAddress::cpu(member));domain.world_ranks.push_back(member);}
 }
 // Each device-free component actor uses this process's genuine local backing.
 // Other ranks remain original metadata, not physical socket execution.
 for(auto& domain:plan.domains)if(domain.name=="cpu_joint")
  for(size_t p=0;p<domain.participants.size();++p)
   domain.participants[p]=GlobalDeviceAddress::cpu(test::localNativeSourceTestNUMANode()+int((p+domain.participants.size()-size_t(rank))%domain.participants.size()));
 plan.residency_policy=RoutedExpertResidencyPolicy::StaticById;
 plan.authority_execution=resolveMoEOverlayAuthorityExecutionKind(plan);
 auto original=std::make_shared<const MoERoutedExpertPlacementPlan>(std::move(plan));
 const auto runtime=resolveMoEExpertOverlayRuntimePlan(original,{.current_world_rank=rank,.validate_mvp_root_reachability=false});
 return {original,std::make_shared<const MoEExpertOverlayPreparationPlan>(MoEExpertOverlayPreparationPlan::build(*runtime,*loader)),file,mpi,factory,loader};
}
/** @return Exact original source geometry and closed Static maintenance intent. */
llaminar2::MoEOverlayResidencyAuthority::Config staticJointConfig(const StaticJointOriginal& original) {
 using namespace llaminar2;
 return {.initial_plan=*original.plan,
  .model_metadata={.num_layers=static_cast<int>(original.plan->placements.size()),.num_experts=512,.d_model=8,.routed_intermediate_size=20,
    .main_inference_layer_count=8,.first_model_layer=original.plan->first_model_layer},
  .projection_preparation=original.preparation,.maintenance_mode=MoERebalanceRuntimeMode::Off,
  .participant_rebalance_policy={.enabled=false}};
}
/** @return Exact CPU member ids from the retained original map, not device ordinals. */
std::vector<int> staticJointMembers(const StaticJointOriginal& original) {
 std::vector<int> ids;for(const auto& member:original.preparation->cpuJointOwnerMap()->participants())
  if(member.domain_name=="cpu_joint")ids.push_back(member.participant_id);
 return ids;
}
}
/** @test Ticket publication retains one original source map without a scalar whole-expert alias. */
TEST(CPUJointStaticResidency, SameSourceMapAndExplicitJointTicketSchema) {
 using namespace llaminar2;const auto original=staticJointOriginal();
 const auto builds=MoEExpertOwnerMap::initialBuildAttempts();MoEOverlayResidencyAuthority authority(staticJointConfig(original));
 EXPECT_EQ(MoEExpertOwnerMap::initialBuildAttempts(),builds);
 const auto snapshot=authority.snapshot();ASSERT_TRUE(snapshot);ASSERT_TRUE(snapshot->valid());
 EXPECT_EQ(snapshot->ownershipKind(),MoEOverlayResidencyOwnershipKind::CPUJointFragments);
 EXPECT_EQ(snapshot->jointOwnerMap(),original.preparation->cpuJointOwnerMap());
 EXPECT_EQ(&snapshot->originalOwnerMap(),original.preparation->cpuJointOwnerMap().get());
 EXPECT_TRUE(snapshot->owner_map.owners().empty());EXPECT_TRUE(snapshot->layered_ownership.empty());
 EXPECT_THROW(snapshot->originalOwnerMap().ownerFor(7,64),std::logic_error);
 for(int i=0;i<20;++i) {
  auto ticket=authority.tryAcquireTicketSnapshot();ASSERT_TRUE(ticket);EXPECT_EQ((*ticket)->epoch,1u);
  auto sequence=authority.tryAcquireGraphSequenceSnapshot();ASSERT_TRUE(sequence);
  EXPECT_EQ(&(**ticket).originalOwnerMap(),&(**sequence).originalOwnerMap());
 }
 auto ambiguous=*snapshot;ambiguous.owner_map=*snapshot->jointOwnerMap();EXPECT_FALSE(ambiguous.valid());
 auto rebound=*snapshot;rebound.placement_plan=std::make_shared<const MoERoutedExpertPlacementPlan>(*snapshot->placement_plan);
 EXPECT_FALSE(rebound.valid());
 auto future=std::make_shared<MoEOverlayResidencySnapshot>(*snapshot);future->epoch=2;EXPECT_FALSE(future->valid());
 MoEOverlayResidencyTransaction fake;fake.expected_epoch=1;fake.previous=snapshot;fake.candidate=future;
 EXPECT_FALSE(fake.valid());EXPECT_TRUE(authority.initialPreparedPlacementPublished());
 EXPECT_THROW(authority.exportAuthoritativeResidencyPlan(fake),std::logic_error);
 EXPECT_THROW(authority.proposeFromHistogram(),std::logic_error);
 EXPECT_THROW(authority.proposeInitialPreparedPlacementRestoration(),std::logic_error);
 std::array<uint32_t,512> output;output.fill(71);std::string error;
 EXPECT_FALSE(authority.tryAcquireCurrentBatchLLEPLease(7,"cpu_joint",2,output,&error));
 EXPECT_EQ(error,"Current-batch LLEP cannot reinterpret complete CPU cohort fragment ownership");
 for(auto value:output)EXPECT_EQ(value,71u);
 EXPECT_EQ(authority.snapshot(),snapshot);
}
/** @test Maintenance on a Static plan and foreign source/member/placement arms reject before publication. */
TEST(CPUJointStaticResidency, RejectsObserveDynamicMissingAndForeignSource) {
 using namespace llaminar2;const auto original=staticJointOriginal();
 for(const auto mode:{MoERebalanceRuntimeMode::Observe,MoERebalanceRuntimeMode::Dynamic}) {
  auto config=staticJointConfig(original);config.maintenance_mode=mode;
  EXPECT_THROW(MoEOverlayResidencyAuthority(std::move(config)),std::invalid_argument);
 }
 auto missing=staticJointConfig(original);missing.projection_preparation.reset();
 EXPECT_THROW(MoEOverlayResidencyAuthority(std::move(missing)),std::invalid_argument);
 auto foreign=staticJointConfig(original);foreign.initial_plan.domains[1].participants[1].numa_node+=3;
 EXPECT_THROW(MoEOverlayResidencyAuthority(std::move(foreign)),std::invalid_argument);
 auto changed=staticJointConfig(original);changed.initial_plan.placements[0].routed_expert_tier[64]=0;
 EXPECT_THROW(MoEOverlayResidencyAuthority(std::move(changed)),std::invalid_argument);
}
/** @test Both actual fragment endpoints publish exact leases/epochs; whole-payload and stale reuse reject. */
TEST(CPUJointStaticResidency, FragmentPublicationEpochIdentityAndFinalSourceRetention) {
 using namespace llaminar2;size_t physical_fragments=0;
 for(int rank:{0,1}) {
 const auto original=staticJointOriginal(rank);const auto all=staticJointMembers(original);ASSERT_EQ(all.size(),2u);
 const auto id=all[size_t(rank)];
 MoEOverlayParticipantResidencyRegistry banks({.local_participant_ids={id},.num_layers=1,.num_experts=512,.initial_epoch=29,
  .projection_preparation=original.preparation,.first_model_layer=7,.original_model=ModelContextId{155}});
 EXPECT_EQ(&banks.initialOwnerMap(),original.preparation->cpuJointOwnerMap().get());
 EXPECT_EQ(banks.localParticipantIds(),std::vector<int>{id});EXPECT_FALSE(banks.allInitialBanksReady());
 ExpertGemmRegistry engines;std::string error;
 EXPECT_TRUE(banks.isCPUJointParticipant(id));EXPECT_FALSE(banks.isCPUJointParticipant(all[size_t(1-rank)]));
 EXPECT_TRUE(banks.endpoint(id));
 EXPECT_THROW(banks.initialCPUJointLayer(id,7),std::logic_error);
 EXPECT_FALSE(banks.registerInitialCPUJointBank(engines,ModelContextId{156},id,&error));
 const auto* participant=original.preparation->cpuJointOwnerMap()->participantForId(id);ASSERT_NE(participant,nullptr);
 const auto selected=original.preparation->cpuShardSourceFor(7,*participant);auto sources=jointSources(selected);
 sources.registerInto(engines,0,1);
 EXPECT_FALSE(banks.registerInitialCPUJointBank(engines,ModelContextId{155},id,&error));
 EXPECT_THROW(banks.initialCPUJointLayer(id,7),std::logic_error);
 sources.registerInto(engines,1);
 ASSERT_TRUE(banks.registerInitialCPUJointBank(engines,ModelContextId{155},id,&error))<<error;
 const auto first=banks.initialCPUJointLayer(id,7);ASSERT_TRUE(first);ASSERT_TRUE(first->sourceSelection());
 EXPECT_EQ(first->epoch(),29u);EXPECT_EQ(first->participantId(),id);EXPECT_EQ(first->sourceSelection(),selected);
 ASSERT_TRUE(banks.registerInitialCPUJointBank(engines,ModelContextId{155},id,&error));
 EXPECT_EQ(banks.initialCPUJointLayer(id,7),first);EXPECT_EQ(banks.cpuJointLayer(id,7,29),first);
 EXPECT_THROW(banks.cpuJointLayer(id,7,1),std::invalid_argument);
 EXPECT_THROW(banks.cpuJointLayer(id,8,29),std::invalid_argument);
 EXPECT_FALSE(banks.registerInitialLayer(id,7,std::vector<bool>(512),std::vector<MoEOverlayPreparedExpertPayload>(512),&error));
 EXPECT_TRUE(banks.allInitialBanksReady());EXPECT_TRUE(banks.incompleteInitialBanks().empty());
 physical_fragments+=banks.verifyImmutablePreparedSources(engines);
 const auto selections=banks.initialBankExpertSelections();ASSERT_EQ(selections.size(),1u);
 EXPECT_EQ(selections[0].ownership_kind,MoEOverlayParticipantResidencyRegistry::InitialOwnershipKind::CPUJointFragments);
 EXPECT_EQ(selections[0].expert_ids.size(),448u);EXPECT_TRUE(selections[0].matches_frozen_owner_map);
 engines.clear();EXPECT_THROW((void)banks.verifyImmutablePreparedSources(engines),std::logic_error);
 EXPECT_TRUE(first->acquireInitialConstruction()->layerForModelLayer(7).cpu_fragments[511].valid());
 }
 EXPECT_EQ(physical_fragments,896u); //448 original experts per independently tested local actor; no physical two-socket claim.
}
/** @test Static setup reuses one original preparation/map under the exact admitted rank resources. */
TEST(CPUJointStaticResidency, OriginalPhysicalAdmissionCreatesAllLocalTypedEndpoints) {
 using namespace llaminar2;const auto original=staticJointOriginal();const auto source=jointManifest();
 const auto& map=*original.preparation->cpuJointOwnerMap();
 const auto local=map.placementFor(7,64)->shard_cohort->localManifest(0);
 const auto cpu=MoEOverlayCapacityResolver::preparedFootprints({local}).front().cpu_live_bytes;
 const auto gpu=MoEOverlayCapacityResolver::preparedFootprints({source}).front().gpu_live_bytes;
 const auto budget=[](const std::string& name,int rank,DeviceId device,size_t bytes) {
  PhysicalMemoryBOMBuilder builder({rank,device,bytes,bytes});
  return MoEOverlayBoundPhysicalMemoryBudget(name,PhysicalMemoryAdmissionCertificate(builder.build()));
 };
 const MoEOverlayCapacityAdmissionPolicy storage{.overlay_world_size=2};
 const auto observed=test::cpuUniformCanonicalInventory({0,1});
 const auto capacity=MoEOverlayCapacityAdmission::resolveCapacity(*original.plan,512,{source},
  {budget("gpu",0,DeviceId::cuda(0),64*gpu),budget("cpu0",0,DeviceId::cpu(),448*cpu),
   budget("cpu1",1,DeviceId::cpu(),448*cpu)},storage,&observed);
 const auto frozen=MoEOverlayCapacityResolver::installResolvedQuotas(*original.plan,capacity);
 const auto memory=std::make_shared<PhysicalMemoryAuthority>(capacity.physical_memory_admission,0);
 const auto metadata=staticJointConfig(original).model_metadata;
 const MoERebalanceRuntimeConfig maintenance{.mode=MoERebalanceRuntimeMode::Off};
 const auto before=MoEExpertOwnerMap::initialBuildAttempts();
 {
  const auto setup=MoEOverlayResidencySetup::create({.plan=frozen,.metadata=metadata,.loader=*original.loader,
    .rebalance=maintenance,.storage_policy=storage,.capacity=capacity,.memory=memory,
    .world_rank=0,.world_size=2,.original_model=ModelContextId{155}});
  EXPECT_EQ(MoEExpertOwnerMap::initialBuildAttempts(),before+1);
  ASSERT_TRUE(setup.authority());ASSERT_TRUE(setup.residency());
  EXPECT_FALSE(setup.histogram());EXPECT_FALSE(setup.interference());
  const auto& preparation=setup.residency()->projectionPreparation();ASSERT_TRUE(preparation);
  EXPECT_EQ(setup.authority()->snapshot()->jointOwnerMap(),preparation->cpuJointOwnerMap());
  EXPECT_EQ(&setup.residency()->initialOwnerMap(),preparation->cpuJointOwnerMap().get());
  ASSERT_EQ(setup.residency()->localParticipantIds().size(),2u);
  const auto members=staticJointMembers(original);EXPECT_TRUE(setup.residency()->isCPUJointParticipant(members[0]));
  EXPECT_FALSE(setup.residency()->isCPUJointParticipant(members[1]));
  EXPECT_FALSE(setup.residency()->allInitialBanksReady());
 }
 EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
    PhysicalMemoryMaterializationKind::NewAllocation),0u);
}

/** @test Retained sparse CPU executors preserve original selected engines across setup and live-prefix replay. */
TEST(CPUJointStaticResidency, RetainedExecutorStartsWithOriginalSourceEngines) {
 using namespace llaminar2;
 const auto original=staticJointOriginal(0);const auto ids=staticJointMembers(original);const int id=ids[0];
 const auto* member=original.preparation->cpuJointOwnerMap()->participantForId(id);ASSERT_TRUE(member);
 const auto selected=original.preparation->cpuShardSourceFor(7,*member);
 ExpertGemmRegistry engines;
 MoEOverlayParticipantResidencyRegistry banks({.local_participant_ids={id},.num_layers=1,.num_experts=512,.initial_epoch=29,
  .projection_preparation=original.preparation,.first_model_layer=7,.original_model=ModelContextId{155}});

 CPUJointExpertRouteStorageLayout joint(selected->cohort(),selected->participantIndex(),2,1);
 const auto layout=std::make_shared<const CPUSerialExpertStorageLayout>(DeviceId::cpu(),2,8,1,
  std::vector<size_t>{1,2},true,10,512,std::nullopt,joint);
 auto sources=jointSources(selected,layout->physicalBytes(),layout->workspaceBytes());
 const auto memory=sources.actor->memory;auto& store=*sources.actor->manager->preparedWeightStore();
 sources.registerInto(engines);
 std::string error;ASSERT_TRUE(banks.registerInitialCPUJointBank(engines,ModelContextId{155},id,&error))<<error;
 const auto published=banks.initialCPUJointLayer(id,7);
 const auto arena=std::make_shared<MoELocalExpertSerialBufferArena>(MoELocalExpertSerialBufferArena::Config{
  .device_id=DeviceId::cpu(),.row_capacity=2,.row_capacity_buckets={1,2},.d_model=8,.routing_top_k=1,
  .cpu_canonical_route_storage=MoELocalExpertSerialBufferArena::CPUCanonicalRouteStoragePolicy::RetainSerialMaximum,
  .cpu_grouped_scratch_storage=MoELocalExpertSerialBufferArena::CPUGroupedScratchStoragePolicy::RetainSerialMaximum,
  .num_experts=512,.expert_intermediate=10,.logical_participant_id=id,.debug_name="original_joint_cold",
  .cpu_storage_layout=layout,.physical_memory_authority=memory});
  std::array<int32_t,2> row_ids{0,1},expert_ids{64,65},output_ids{};
 std::array<int32_t,3> offsets{0,1,2};std::array<float,2> weights{0.25f,0.75f};
 std::array<float,16> hidden;hidden.fill(0.01f);std::array<float,16> output_values{};
 MoEOverlaySparseRows input;input.row_capacity=2;input.entry_capacity=2;input.hidden_row_capacity=2;input.d_model=8;input.top_k=1;
 input.residency_epoch=29;input.source_participant=0;input.target_participant=id;
 input.row_ids_host=row_ids.data();input.entry_offsets_host=offsets.data();input.expert_ids_host=expert_ids.data();
 input.route_weights_host=weights.data();input.hidden_rows_fp32=hidden.data();
 MoEOverlayReturnRows output;output.row_capacity=2;output.d_model=8;output.row_ids_host=output_ids.data();output.output_rows_fp32=output_values.data();
 output.layout=MoEOverlayReturnLayout::CanonicalExpertRoutes;
 std::array<int32_t,2> original_slots{0,1},compact_slots{0,1};
 input.original_route_slots_host=original_slots.data();input.compact_route_slots_host=compact_slots.data();
 MoELocalExpertStage::Params params;params.device_id=DeviceId::cpu();params.input_rows=&input;params.output_rows=&output;
 params.serial_compact_buffer_arena=arena;params.graph_row_capacity=2;params.num_experts=512;params.top_k=1;
 params.d_model=8;params.expert_intermediate=10;params.layer_idx=7;params.runtime_participant_index=id;
 params.cpu_joint_layer=published;params.expert_registry=&engines;params.physical_memory_authority=memory;
 params.expert_weight_resolution_policy=MoELocalExpertStage::ExpertWeightResolutionPolicy::RegistryOnly;
 params.cpu_joint_route_producer=MoELocalExpertStage::CPUJointRouteProducerBinding{
  .packed_rows=arena->cpuJointPackedRecords(),.original_route_slots=original_slots.data(),
  .compact_route_slots=compact_slots.data(),.route_slot_capacity=2};
 params.expert_mask.assign(512,false);params.prepared_gate_gemm.assign(512,nullptr);
 params.prepared_up_gemm.assign(512,nullptr);params.prepared_down_gemm.assign(512,nullptr);
 const auto formats=originalCPUExpertWorkspaceSourceFormats(*original.loader,7,8,20,512);
 params.cpu_workspace_source=MoELocalExpertStage::CPUExpertWorkspaceSource{
  .gate=formats[0],.up=formats[1],.down=formats[2],.workers=omp_in_parallel()?omp_get_num_threads():omp_get_max_threads(),.execution=CPUExecutionGeometry::local()};
 {
  // The generic complete-bank graph has its own explicit valid cold lifecycle.
  // It authenticates actual initial engines; the packet graph below retains none.
  const auto construction=published->acquireInitialConstruction();ASSERT_TRUE(construction);
  FP32Tensor complete_input({2,8}),complete_indices({2,1}),complete_weights({2,1}),complete_output({2,8});
  MoEExpertComputeStage::Params complete;
  complete.device_id=DeviceId::cpu();complete.input=&complete_input;complete.output=&complete_output;
  complete.routing_indices=&complete_indices;complete.routing_weights=&complete_weights;
  complete.seq_len=2;complete.d_model=8;complete.num_experts=512;complete.top_k=1;
  complete.expert_intermediate=10;complete.layer_idx=7;complete.cpu_joint_layer=published;
  complete.expert_weight_resolution_policy=MoEExpertWeightResolutionPolicy::PreparedRegistryOnly;
  complete.prepared_engine_binding_lifecycle=MoEPreparedEngineBindingLifecycle::CompleteAtConstruction;
  complete.canonical_route_contributions=arena->cpuCanonicalRoutes().get();
  complete.canonical_route_arithmetic=MoECanonicalRouteArithmeticPolicy::UnweightedExpertRowThenOrderedFMA;
  complete.canonical_route_layout=MoECanonicalRoutePublicationLayout::DenseOriginalRouteSlots;
  complete.cpu_grouped_serial_workspace=arena->cpuGroupedWorkspace();
  complete.cpu_router_q8_input_publication=CPURouterQ8InputPublicationPolicy::PublishTransportedRows;
  complete.expert_mask.assign(512,false);complete.prepared_gate_gemm.assign(512,nullptr);
  complete.prepared_up_gemm.assign(512,nullptr);complete.prepared_down_gemm.assign(512,nullptr);
  for(int expert:selected->expertIds()) {
   const auto& engine=construction->layerForModelLayer(7).cpu_fragments[expert];
   complete.expert_mask[expert]=true;complete.prepared_gate_gemm[expert]=engine.gate().get();
   complete.prepared_up_gemm[expert]=engine.up().get();complete.prepared_down_gemm[expert]=engine.down().get();
  }
  EXPECT_NO_THROW((void)MoEExpertComputeStage(std::move(complete)));
 }
 MoELocalExpertStage stage(params);test::CPUStageTestWorkspace workspace(stage);
 // Source publication, not mutable registry/store entries, retains the engines.
 store.clear();engines.clear();
 llaminar2::testing::MockDeviceContext context(DeviceId::cpu(),ComputeBackendType::CPU);
 for(int repeat=0;repeat<3;++repeat)for(size_t live:{2u,0u,1u}) {
  input.live_row_count=live;input.live_entry_count=live;
  ASSERT_TRUE(stage.execute(&context));
  EXPECT_EQ(canonical_moe_route_record::readRecordCount(arena->cpuJointPackedRecords()->data(),
   arena->cpuJointPackedRecords()->numel()),live);
  EXPECT_EQ(published->sourceSelection(),selected);
  EXPECT_EQ(output.layout,MoEOverlayReturnLayout::CanonicalExpertRoutes);
  EXPECT_EQ(output.live_row_count,0u);
 }
 auto replaced=params;replaced.prepared_gate_gemm[64]=published->acquireInitialConstruction()->layerForModelLayer(7).cpu_fragments[64].gate().get();
 EXPECT_THROW((void)MoELocalExpertStage(replaced),std::invalid_argument);
 auto missing=params;missing.cpu_workspace_source.reset();
 EXPECT_THROW((void)MoELocalExpertStage(missing),std::invalid_argument);
}

/** @test Distinct cold admission, loading and graph adoption retain one original CPU source authority. */
TEST(CPUJointStaticResidency, ColdLoadGraphPreservesOriginalSourceAuthority) {
 using namespace llaminar2;
 const auto original=staticJointOriginal();const auto source=jointManifest();
 const auto initial_member=staticJointMembers(original).front();
 const auto initial_selection=original.preparation->cpuShardSourceFor(7,
  *original.preparation->cpuJointOwnerMap()->participantForId(initial_member));
 const auto local=initial_selection->cohort()->localManifest(initial_selection->participantIndex());
 const auto cpu=MoEOverlayCapacityResolver::preparedFootprints({local}).front().cpu_live_bytes;
 const auto gpu=MoEOverlayCapacityResolver::preparedFootprints({source}).front().gpu_live_bytes;
 size_t staging_bytes=0;
 for(const auto& projection:local.projections) {
  const auto type=projection.format.floatingTensorType();
  ASSERT_TRUE(type); // This original fixture retains actual FP32 source matrices.
  const auto request=quoteCPUWeightPreparation(*type,projection.N,projection.K,
   WeightPrecision::NATIVE,CPUWeightConsumer::PreparedGemm);
  if(request)staging_bytes=std::max(staging_bytes,resolveNativeWeightLoadStaging(*request,
   CPUStoragePlacement::onNode(initial_selection->participant().address.numa_node),
   configuredNativeWeightLoadStagingBudgetBytes()).physical_bytes);
 }
 EXPECT_EQ(staging_bytes,0u); // Native floating execution writes its sole final owner directly.
 const auto cpu_total=448*cpu+staging_bytes;
 // Price source staging before capacity admission so all resource views, cold
 // setup and the live PMA retain the same final certificate.
 const auto budget=[&](const std::string& name,int rank,DeviceId device,size_t bytes) {
  PhysicalMemoryBOMBuilder builder({rank,device,bytes,bytes});
  if(device.is_cpu())builder.add(PhysicalMemoryOwner::WeightLoadStaging,staging_bytes);
  return MoEOverlayBoundPhysicalMemoryBudget(name,PhysicalMemoryAdmissionCertificate(builder.build()));
 };
 const MoEOverlayCapacityAdmissionPolicy storage{.overlay_world_size=2};
 const auto observed=test::cpuUniformCanonicalInventory({0,1});
 auto capacity=MoEOverlayCapacityAdmission::resolveCapacity(*original.plan,512,{source},
  {budget("gpu",0,DeviceId::cuda(0),64*gpu),budget("cpu0",0,DeviceId::cpu(),cpu_total),
   budget("cpu1",1,DeviceId::cpu(),cpu_total)},storage,&observed);
 const auto frozen=MoEOverlayCapacityResolver::installResolvedQuotas(*original.plan,capacity);
 const auto memory=std::make_shared<PhysicalMemoryAuthority>(capacity.physical_memory_admission,0);
 const auto metadata=staticJointConfig(original).model_metadata;
 const MoERebalanceRuntimeConfig maintenance{.mode=MoERebalanceRuntimeMode::Off};
 const ModelContextId model{359};
 const auto cold=MoEOverlayResidencySetup::create({.plan=frozen,.metadata=metadata,.loader=*original.loader,
  .rebalance=maintenance,.storage_policy=storage,.capacity=capacity,.memory=memory,
  .world_rank=0,.world_size=2,.original_model=model});
 const auto retained=cold.residency()->projectionPreparation();ASSERT_TRUE(retained);
 const auto owners=retained->cpuJointOwnerMap();ASSERT_TRUE(owners);
 const auto* participant=owners->participantForId(initial_member);ASSERT_TRUE(participant);
 const auto selected=retained->cpuShardSourceFor(7,*participant);ASSERT_TRUE(selected);
 const auto runtime=resolveMoEExpertOverlayRuntimePlan(std::make_shared<const MoERoutedExpertPlacementPlan>(frozen),
  {.current_world_rank=0,.validate_mvp_root_reachability=false});ASSERT_TRUE(runtime);
 std::string error;
 // A separately rebuilt but structurally identical source is not the original
 // cold authority. Actual preparation must not make it publishable into A.
 {
  const auto foreign=MoEExpertOverlayPreparationPlan::build(*runtime,*original.loader).filteredForDevice(DeviceId::cpu());
  const auto foreign_selection=foreign.cpuShardSourceFor(7,*participant);
  EXPECT_EQ(*foreign_selection,*selected);EXPECT_NE(foreign_selection->cohort(),selected->cohort());
  WeightManager competing(*original.loader,original.mpi);competing.installPhysicalMemoryAuthority(memory);
  const auto loaded=competing.materialize(foreign.sourceWeightPlan(*original.loader,model));
  ASSERT_EQ(loaded.bindings().size(),448u*3);
  const auto competing_lease=competing.expertGemmRegistry().acquireCPUJointLayer(model,*participant,7);
  ASSERT_TRUE(competing_lease);ASSERT_TRUE(competing_lease->cpu_joint_source);
  EXPECT_EQ(*competing_lease->cpu_joint_source,*selected);
  EXPECT_NE(competing_lease->cpu_joint_source->cohort(),selected->cohort());
  for(const auto& source_binding:competing_lease->sources)EXPECT_EQ(source_binding.identity.model_id,model);
  EXPECT_FALSE(cold.residency()->registerInitialCPUJointBank(competing.expertGemmRegistry(),model,
   participant->participant_id,&error));
  EXPECT_NE(error.find("exact original source/cohort lease"),std::string::npos)<<error;
  EXPECT_THROW(cold.residency()->initialCPUJointLayer(participant->participant_id,7),std::logic_error);
 }
 EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
  PhysicalMemoryMaterializationKind::NewAllocation),0u);
 const auto before=MoEExpertOwnerMap::initialBuildAttempts();
 EXPECT_THROW(MoEExpertOverlayPreparationPlan::forOriginalSource(*runtime,*original.loader,{}),std::invalid_argument);
 EXPECT_EQ(MoEExpertOwnerMap::initialBuildAttempts(),before);
 const auto loading=MoEExpertOverlayPreparationPlan::forOriginalSource(*runtime,*original.loader,retained)
  .filteredForDevice(DeviceId::cpu());
 EXPECT_EQ(loading.cpuJointOwnerMap(),owners);
 EXPECT_EQ(loading.cpuShardSourceFor(7,*participant)->cohort(),selected->cohort());
 EXPECT_EQ(MoEExpertOwnerMap::initialBuildAttempts(),before);
 WeightManager manager(*original.loader,original.mpi);manager.installPhysicalMemoryAuthority(memory);
 const auto loaded=manager.materialize(loading.sourceWeightPlan(*original.loader,model));
 ASSERT_EQ(loaded.bindings().size(),448u*3);
 ASSERT_TRUE(manager.prepareMoEExpertOverlayWeights(*runtime,DeviceId::cpu(),&loaded,nullptr,
  PreparedWeightAdmission::AllocateCompleteSet,retained));
 EXPECT_EQ(MoEExpertOwnerMap::initialBuildAttempts(),before);
 ASSERT_TRUE(cold.residency()->registerInitialCPUJointBank(manager.expertGemmRegistry(),model,
  participant->participant_id,&error))<<error;
 const auto published=cold.residency()->initialCPUJointLayer(participant->participant_id,7);ASSERT_TRUE(published);
 EXPECT_EQ(published->sourceSelection()->cohort(),selected->cohort());
 std::vector<bool> mask(512,false);for(const int expert:selected->expertIds())mask[expert]=true;
 MoEOverlayPreparedGraphBankRequest request{.device=DeviceId::cpu(),.tp_context=nullptr,.tp_participant=0,
  .domain=participant->domain_name,.participant=participant,.layer=7,.experts=512,.d_model=8,.expert_intermediate=20,
  .first_expert=0,.local_experts=-1,.expert_mask=std::cref(mask),.original_model=model,
  .original_residency=cold.residency(),.original_owners=owners};
 const auto graph=adoptMoEOverlayPreparedGraphBank(*runtime,manager.expertGemmRegistry(),request);
 EXPECT_EQ(graph.cpu_joint_layer,published);EXPECT_EQ(graph.local_intermediate,10);
 for(int expert=0;expert<512;++expert)EXPECT_EQ(graph.gate[expert]!=nullptr,selected->contains(expert));
 request.original_model=ModelContextId{360};
 EXPECT_THROW(adoptMoEOverlayPreparedGraphBank(*runtime,manager.expertGemmRegistry(),request),std::invalid_argument);
 EXPECT_FALSE(cold.residency()->registerInitialCPUJointBank(manager.expertGemmRegistry(),ModelContextId{360},
  participant->participant_id,&error));
 EXPECT_EQ(cold.residency()->initialCPUJointLayer(participant->participant_id,7),published);
 EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
  PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
