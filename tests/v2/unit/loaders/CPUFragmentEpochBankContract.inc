/**
 * @file CPUFragmentEpochBankContract.inc
 * @brief Exercise a genuine current-source candidate in the existing participant RCU bank.
 *
 * A prepared candidate endpoint is not global routing publication. The original
 * authority remains at epoch one until physical fanout and commit are qualified.
 */
#include "fixtures/moe/MoEOverlayJointPlacementCandidateTestAccess.h"
#include "execution/moe/MoEOverlayParticipantMigration.h"
#include <optional>
#include <cstdlib>

/** @test Original context, closed arrival and fixed CPU family survive a local candidate bank. */
namespace {
/** @brief Exercise original source, exact epoch packets and optional post-retirement lazy graph construction.
 * @param late Construct the graph only after the bootstrap bank is reclaimed.
 * @param members Original typed cohort width; only this process's local member executes physically.
 * @param serial_layers Use distinct real layer sources and a separately checked serial arena cohort.
 */
void exerciseOriginalCPUFragmentGraphLifecycle(bool late,int members=2,bool serial_layers=false)
{
    using namespace llaminar2;
    // The original collector owns reset/configuration; this test retains no
    // record address or policy epoch while genuine local GEMMs execute.
    /** @brief Serial test-owned collector policy; restores caller configuration after real CPU execution. */
    struct EvidenceScope {
        std::optional<std::string> json,filter;
        /** @brief Enable only the original residency evidence domain for this fixture. */
        EvidenceScope(){if(const auto* v=std::getenv("LLAMINAR_PERF_STATS_JSON"))json=v;
            if(const auto* v=std::getenv("LLAMINAR_PERF_STATS_FILTER"))filter=v;
            ::setenv("LLAMINAR_PERF_STATS_JSON","1",1);::setenv("LLAMINAR_PERF_STATS_FILTER","moe_overlay_residency",1);PerfStatsCollector::reset();}
        /** @brief Reset observations and restore the exact caller policy. */
        ~EvidenceScope(){if(json)::setenv("LLAMINAR_PERF_STATS_JSON",json->c_str(),1);else ::unsetenv("LLAMINAR_PERF_STATS_JSON");
            if(filter)::setenv("LLAMINAR_PERF_STATS_FILTER",filter->c_str(),1);else ::unsetenv("LLAMINAR_PERF_STATS_FILTER");PerfStatsCollector::reset();}
    } evidence;
    const auto original = staticJointOriginal(0,members,serial_layers);
    auto context = coldSourceContext(original);
    const ModelContextId model{reinterpret_cast<uint64_t>(context.get())};
    const auto ids = staticJointMembers(original);
    ASSERT_FALSE(ids.empty());
    const int id = ids.front();
    const auto *participant = original.preparation->cpuJointOwnerMap()->participantForId(id);
    ASSERT_TRUE(participant);
    const auto selected = original.preparation->cpuShardSourceFor(7, *participant);
    ASSERT_TRUE(selected);
    const int local_intermediate=selected->ownership().projection(WeightRole::MoEExpertUp).rows;
    MoEOverlayResidencyAuthority authority(staticJointConfig(original));
    auto placements=original.plan->placements;
    std::swap(placements.back().routed_expert_tier[1],placements.back().routed_expert_tier[64]);
    auto candidate = test::MoEOverlayJointPlacementCandidateTestAccess::prepare(
        authority, std::move(placements));
    ASSERT_TRUE(candidate); ASSERT_EQ(candidate->epoch, 2u);
    auto manager = context->concreteWeightManager();
    InferenceStrategy strategy{.model_id=model};
    WeightPlan plan(strategy, PhysicalMemoryOwner::RoutedExpertWeights);
    for(int layer=serial_layers?6:7;layer<=7;++layer)
    for (const auto role : {WeightRole::MoEExpertGate, WeightRole::MoEExpertUp, WeightRole::MoEExpertDown})
    {
        const auto name = std::string("blk.")+std::to_string(layer)+".ffn_" +
            (role==WeightRole::MoEExpertGate?"gate":role==WeightRole::MoEExpertUp?"up":"down") + "_exps.weight";
        plan.add(test::nativeJointRequirementForTest(original.preparation->cpuShardSourceFor(layer,*participant),role,name));
    }
    // Destination roots are prepared as a component setup oracle before movement.
    // This does not exercise or claim the native physical fanout transport.
    WeightPlan incoming_plan(strategy,PhysicalMemoryOwner::RoutedExpertWeights);
    for(const auto role:{WeightRole::MoEExpertGate,WeightRole::MoEExpertUp,WeightRole::MoEExpertDown}) {
        const auto projection=selected->ownership().projection(role);
        WeightRequirement request;
        request.canonical_name=std::string("blk.7.ffn_")+
            (role==WeightRole::MoEExpertGate?"gate":role==WeightRole::MoEExpertUp?"up":"down")+"_exps.weight";
        request.role=role;request.layer=7;request.expert=1;request.target_device=participant->device;
        request.derivation=WeightDerivationKind::ExpertSlice;request.bypass_tensor_parallel=true;
        request.expected_prepared_kind=PreparedWeightKind::CpuPackedGemm;
        request.overlay_domain=participant->domain_name;
        request.overlay_participant_index=participant->domain_participant_index;
        request.overlay_participant_world_rank=participant->world_rank;
        request.slice={.source_rows=size_t(projection.source_columns),.source_cols=size_t(projection.source_rows),
            .row_start=size_t(projection.first_column),.row_count=size_t(projection.columns),
            .col_start=size_t(projection.first_row),.col_count=size_t(projection.rows),
            .expert_count=1,.expert_ids={1}};
        incoming_plan.add(std::move(request));
    }
    const auto placement=CPUStoragePlacement::onNode(participant->address.numa_node);
    size_t weight_bytes=0;
    for(const auto &projection:selected->cohort()->localManifest(selected->participantIndex()).projections) {
        const auto elements=size_t(projection.N)*projection.K;
        weight_bytes += size_t(serial_layers?2:1)*selected->expertIds().size()*placement.physicalBytes<float>(elements);
        weight_bytes += context->concreteLoader().nativeWeightSourcePlacement().physicalBytes<float>(elements);
    }
    std::shared_ptr<const MoEExpertShardCohort> arena_cohort=selected->cohort();
    if(serial_layers) {
        const auto earlier=original.preparation->cpuShardSourceFor(6,*participant);
        ASSERT_TRUE(earlier);ASSERT_NE(earlier->cohort(),selected->cohort());
        ASSERT_NE(earlier->cohort()->source().layer_idx,selected->cohort()->source().layer_idx);
        const std::array manifests{earlier->cohort()->source(),selected->cohort()->source()};
        const auto members=selected->cohort()->participants();
        arena_cohort=CPUJointExpertRouteStorageLayout::checkedSerialCohort(manifests,
            selected->ownership().geometry(),{members.begin(),members.end()});
        ASSERT_NE(arena_cohort,earlier->cohort());ASSERT_NE(arena_cohort,selected->cohort());
        ASSERT_NE(*arena_cohort,*selected->cohort()); // Real source layers differ; only physical cuts agree.
    }
    CPUJointExpertRouteStorageLayout joint(arena_cohort,selected->participantIndex(),2,1);
    const auto layout=std::make_shared<const CPUSerialExpertStorageLayout>(participant->device,2,8,1,
        std::vector<size_t>{1,2},true,local_intermediate,512,std::nullopt,joint);
    const auto total=weight_bytes+layout->physicalBytes()+layout->workspaceBytes();
    PhysicalMemoryPlanBuilder budget;
    const PhysicalMemoryResource host{participant->world_rank,participant->device,total,total};
    budget.add(host,PhysicalMemoryOwner::RoutedExpertWeights,weight_bytes);
    budget.add(host,PhysicalMemoryOwner::ActivationArena,layout->physicalBytes());
    budget.add(host,PhysicalMemoryOwner::ExecutionWorkspace,layout->workspaceBytes());
    auto memory=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),host.world_rank);
    manager->installPhysicalMemoryAuthority(memory);
    ASSERT_EQ(manager->materialize(plan).bindings().size(),size_t(serial_layers?2:1)*1344u);
    MoEOverlayParticipantResidencyRegistry banks({.local_participant_ids={id},.num_layers=serial_layers?2:1,
        .num_experts=512,.initial_epoch=1,.retained_epoch_capacity=3,.projection_preparation=original.preparation,
        .first_model_layer=serial_layers?6:7,.original_model=model});
    std::string error;
    // Reproduce factory ordering: real weights exist, but neither the original
    // context nor the cold source family has been admitted before graph setup.
    EXPECT_FALSE(banks.acquireOriginalPreparedSource(&error));
    EXPECT_NE(error.find("no original ModelContext binding"),std::string::npos)<<error;
    try {
        (void)adoptMoEOverlayCPUJointGraphDeclaration(banks,manager->expertGemmRegistry(),
            model,id,7,512,8,20);
        FAIL()<<"Unbound cold graph declaration unexpectedly succeeded";
    } catch(const std::logic_error& failure) {
        EXPECT_EQ(std::string(failure.what()),"CPU joint graph lost its original prepared source: Prepared source has no original ModelContext binding");
    }
    EXPECT_FALSE(banks.admitInitialCPUJointGraphSources({},&error));
    EXPECT_FALSE(banks.acquireOriginalPreparedSource(&error));
    ASSERT_TRUE(banks.admitInitialCPUJointGraphSources(context,&error))<<error;
    ASSERT_TRUE(banks.admitInitialCPUJointGraphSources(context,&error))<<error;
    ASSERT_TRUE(adoptMoEOverlayCPUJointGraphDeclaration(banks,manager->expertGemmRegistry(),
        model,id,7,512,8,20));
    const auto cold=banks.initialCPUJointLayer(id,7);
    ASSERT_TRUE(cold); ASSERT_FALSE(cold->bankLease());
    auto old=cold->acquireEpoch(1); ASSERT_TRUE(old);
    std::weak_ptr<ITensorGemm> departed=old->bankLease()->layerForModelLayer(7).cpu_fragments[64].gate();
    // The cold source is authenticated before this component oracle prepares
    // candidate-only roots. They are not members of its original selection.
    const auto incoming=manager->materialize(incoming_plan);
    ASSERT_EQ(incoming.bindings().size(),3u);
    std::array<std::shared_ptr<ITensorGemm>,3> destination_engines;
    for(size_t role=0;role<3;++role) {
        ASSERT_TRUE(incoming.bindings()[role].prepared);
        destination_engines[role]=manager->preparedWeightStore()->gemmKernelLifetime(*incoming.bindings()[role].prepared);
        ASSERT_TRUE(destination_engines[role]);
    }
    auto arrival=banks.prepareCPUJointArrival(*candidate,id,7,1,&error);
    ASSERT_TRUE(arrival)<<error;
    EXPECT_TRUE(arrival->isCPUJointFragment()); EXPECT_EQ(arrival->destinationEpoch(),2u);
    EXPECT_FALSE(banks.prepareCPUJointArrival(*candidate,ids.back(),7,1,&error));
    EXPECT_FALSE(banks.prepareCPUJointArrival(*candidate,id,7,64,&error));
    ASSERT_TRUE(arrival->publish(ExpertTierWeightProjection::Gate,destination_engines[0],&error))<<error;
    ASSERT_TRUE(arrival->publish(ExpertTierWeightProjection::Up,destination_engines[1],&error))<<error;
    ASSERT_TRUE(arrival->publish(ExpertTierWeightProjection::Down,destination_engines[2],&error))<<error;
    MoEOverlayPreparedExpertPayload whole;
    EXPECT_FALSE(arrival->completePayload(whole,&error));
    MoEOverlayPreparedCPUFragmentPayload fragment;
    ASSERT_TRUE(arrival->completeCPUFragment(fragment,&error))<<error;
    EXPECT_EQ(fragment.arrivalEpoch(),2u);
    EXPECT_EQ(fragment.model(),model); EXPECT_EQ(fragment.cohort(),selected->cohort());
    EXPECT_EQ(fragment.preparation(),original.preparation); EXPECT_EQ(fragment.expertId(),1);
    EXPECT_EQ(old->bankLease()->layerForModelLayer(7).cpu_fragments[64].arrivalEpoch(),0u);
    EXPECT_THROW(fragment.recordCompletedExecution(1,1),std::logic_error);
    auto next=*old->bankLease(); next.epoch=2;
    next.layerForModelLayer(7).clearExpert(64);
    next.layerForModelLayer(7).setResidentCPUFragment(1,fragment);
    auto endpoint=banks.endpoint(id); ASSERT_TRUE(endpoint);
    auto foreign_family=next;
    const auto canonical_family=foreign_family.layerForModelLayer(7).cpu_fragment_family;
    foreign_family.layerForModelLayer(7).cpu_fragment_family=
        std::shared_ptr<const MoEOverlayCPUFragmentFamily>(canonical_family.get(),[](const auto*){});
    EXPECT_FALSE(endpoint->prepareReadyBank(std::move(foreign_family),&error));
    auto alias_config=staticJointConfig(original);
    alias_config.projection_preparation=std::shared_ptr<const MoEExpertOverlayPreparationPlan>(
        original.preparation.get(),[](const auto*){});
    MoEOverlayResidencyAuthority alias_authority(std::move(alias_config));
    auto alias_candidate=test::MoEOverlayJointPlacementCandidateTestAccess::prepare(
        alias_authority,candidate->placement_plan->placements);
    EXPECT_FALSE(banks.prepareCPUJointArrival(*alias_candidate,id,7,1,&error));
    auto prepared=endpoint->prepareReadyBank(std::move(next),&error); ASSERT_TRUE(prepared)<<error;
    ASSERT_EQ(endpoint->installReadyBank(std::move(*prepared),&error),MoEOverlayParticipantBankInstallStatus::Installed)<<error;
    auto active=cold->acquireEpoch(2); ASSERT_TRUE(active);
    EXPECT_EQ(active->bankLease()->epoch,2u); EXPECT_EQ(old->bankLease()->epoch,1u);
    EXPECT_EQ(active->bankLease()->layerForModelLayer(7).cpu_fragments[1].gate(),destination_engines[0]);
    EXPECT_EQ(cold->sourceSelection(),selected);
    EXPECT_FALSE(active->bankLease()->layerForModelLayer(7).resident_mask[64]);
    EXPECT_TRUE(old->bankLease()->layerForModelLayer(7).resident_mask[64]);
    EXPECT_FALSE(old->bankLease()->layerForModelLayer(7).resident_mask[1]);
    EXPECT_FALSE(old->bankLease()->layerForModelLayer(7).cpu_fragments[1].valid());
    // Installed candidate2 is authentic physical storage while global authority remains1.
    auto terminal_arrival = banks.prepareCPUJointArrivalFromInstalledBank(
        active->bankLease(), id, 7, 1, &error);
    ASSERT_TRUE(terminal_arrival) << error;
    EXPECT_EQ(terminal_arrival->destinationEpoch(), 2u);
    MoEOverlayPreparedCPUFragmentPayload unready_terminal;
    EXPECT_FALSE(terminal_arrival->completeCPUFragment(unready_terminal, &error));
    EXPECT_FALSE(banks.prepareCPUJointArrivalFromInstalledBank({}, id, 7, 1, &error));
    EXPECT_FALSE(banks.prepareCPUJointArrivalFromInstalledBank(active->bankLease(), ids.back(), 7, 1, &error));
    EXPECT_FALSE(banks.prepareCPUJointArrivalFromInstalledBank(active->bankLease(), id, 8, 1, &error));
    EXPECT_FALSE(banks.prepareCPUJointArrivalFromInstalledBank(active->bankLease(), id, 7, 64, &error));
    // A separate installed physical node is not a global epoch3 publication.
    auto retired_copy = *active->bankLease();
    retired_copy.epoch = 3;
    auto retired_ready = endpoint->prepareReadyBank(std::move(retired_copy), &error);
    ASSERT_TRUE(retired_ready) << error;
    ASSERT_EQ(endpoint->installReadyBank(std::move(*retired_ready), &error),
              MoEOverlayParticipantBankInstallStatus::Installed);
    auto retired_borrow = endpoint->acquire(3);
    ASSERT_TRUE(retired_borrow);
    ASSERT_TRUE(endpoint->retire(3));
    EXPECT_FALSE(banks.prepareCPUJointArrivalFromInstalledBank(retired_borrow, id, 7, 1, &error));
    EXPECT_NE(error.find("retired bank node"), std::string::npos);
    retired_borrow.reset();
    (void)endpoint->retire(3);
    EXPECT_EQ(authority.snapshot()->epoch,1u); // No fabricated global commit.
    EXPECT_FALSE(banks.prepareCPUJointArrival(*authority.snapshot(),id,7,71,&error));
    if(late) {
        manager->preparedWeightStore()->clear();manager->expertGemmRegistry().clear();
    // Actual original bootstrap bytes retire before lazy graph setup. Initial
    // declaration metadata and captured raw engine indexes must not keep a
    // departed root alive; the existing RCU reader is its final owner.
    const auto before_retirement=memory->claimedBytes(participant->device,
        PhysicalMemoryOwner::RoutedExpertWeights,PhysicalMemoryMaterializationKind::NewAllocation);
    const auto observed_initial=banks.initialBankExpertSelections();
    ASSERT_EQ(observed_initial.size(),size_t(serial_layers?2:1));
    EXPECT_TRUE(observed_initial.front().matches_frozen_owner_map);
    EXPECT_NE(std::find(observed_initial.front().expert_ids.begin(),observed_initial.front().expert_ids.end(),64),
              observed_initial.front().expert_ids.end());
    ASSERT_TRUE(endpoint->retire(1));
    EXPECT_THROW((void)banks.initialBankExpertSelections(),std::logic_error);
    EXPECT_FALSE(departed.expired());
    EXPECT_EQ(memory->claimedBytes(participant->device,PhysicalMemoryOwner::RoutedExpertWeights,
        PhysicalMemoryMaterializationKind::NewAllocation),before_retirement);
    old.reset();
    EXPECT_FALSE(endpoint->retire(1)); // Existing maintenance edge reclaims drained retired nodes.
    EXPECT_TRUE(departed.expired());
    EXPECT_THROW((void)banks.initialBankExpertSelections(),std::logic_error);
    const auto departed_bytes=3*placement.physicalBytes<float>(size_t(local_intermediate*8));
    EXPECT_EQ(memory->claimedBytes(participant->device,PhysicalMemoryOwner::RoutedExpertWeights,
        PhysicalMemoryMaterializationKind::NewAllocation),before_retirement-departed_bytes);

        EXPECT_FALSE(banks.admitInitialCPUJointGraphSources(context,&error));
        EXPECT_NE(error.find("retired or absent"),std::string::npos)<<error;
        EXPECT_FALSE(banks.registerInitialCPUJointBank(manager->expertGemmRegistry(),model,id,&error));
        EXPECT_NE(error.find("retired or absent"),std::string::npos);
        const auto declaration=adoptMoEOverlayCPUJointGraphDeclaration(banks,manager->expertGemmRegistry(),model,id,7,512,8,20);
        EXPECT_EQ(declaration,cold);EXPECT_FALSE(declaration->bankLease());
        EXPECT_FALSE(declaration->acquireInitialConstruction());
        ExpertGemmRegistry foreign_registry;
        EXPECT_THROW(adoptMoEOverlayCPUJointGraphDeclaration(banks,foreign_registry,model,id,7,512,8,20),std::invalid_argument);
        EXPECT_THROW(adoptMoEOverlayCPUJointGraphDeclaration(banks,manager->expertGemmRegistry(),ModelContextId{model.value+1},id,7,512,8,20),std::invalid_argument);
        EXPECT_THROW(adoptMoEOverlayCPUJointGraphDeclaration(banks,manager->expertGemmRegistry(),model,ids.back(),7,512,8,20),std::invalid_argument);
    }
 {
 const auto arena=std::make_shared<MoELocalExpertSerialBufferArena>(MoELocalExpertSerialBufferArena::Config{
  .device_id=DeviceId::cpu(),.row_capacity=2,.row_capacity_buckets={1,2},.d_model=8,.routing_top_k=1,
  .cpu_canonical_route_storage=MoELocalExpertSerialBufferArena::CPUCanonicalRouteStoragePolicy::RetainSerialMaximum,
  .cpu_grouped_scratch_storage=MoELocalExpertSerialBufferArena::CPUGroupedScratchStoragePolicy::RetainSerialMaximum,
  .num_experts=512,.expert_intermediate=local_intermediate,.logical_participant_id=id,.debug_name="original_joint_cold",
  .cpu_storage_layout=layout,.physical_memory_authority=memory});
  std::array<int32_t,2> row_ids{0,1},expert_ids{1,65},output_ids{};
 std::array<int32_t,3> offsets{0,1,2};std::array<float,2> weights{0.25f,0.75f};
 std::array<float,16> hidden;hidden.fill(0.01f);std::array<float,16> output_values{};
 MoEOverlaySparseRows input;input.row_capacity=2;input.entry_capacity=2;input.hidden_row_capacity=2;input.d_model=8;input.top_k=1;
 input.residency_epoch=2;input.source_participant=0;input.target_participant=id;
 input.row_ids_host=row_ids.data();input.entry_offsets_host=offsets.data();input.expert_ids_host=expert_ids.data();
 input.route_weights_host=weights.data();input.hidden_rows_fp32=hidden.data();
 MoEOverlayReturnRows output;output.row_capacity=2;output.d_model=8;output.row_ids_host=output_ids.data();output.output_rows_fp32=output_values.data();
 std::array<int32_t,2> original_slots{0,1},compact_slots{0,1};
 input.original_route_slots_host=original_slots.data();input.compact_route_slots_host=compact_slots.data();
 MoELocalExpertStage::Params params;params.device_id=DeviceId::cpu();params.input_rows=&input;params.output_rows=&output;
 params.serial_compact_buffer_arena=arena;params.graph_row_capacity=2;params.num_experts=512;params.top_k=1;
 params.d_model=8;params.expert_intermediate=local_intermediate;params.layer_idx=7;params.runtime_participant_index=id;
 params.cpu_joint_layer=cold;params.expert_registry=&manager->expertGemmRegistry();params.physical_memory_authority=memory;
 params.expert_weight_resolution_policy=MoELocalExpertStage::ExpertWeightResolutionPolicy::RegistryOnly;
 params.cpu_joint_route_producer=MoELocalExpertStage::CPUJointRouteProducerBinding{
  .packed_rows=arena->cpuJointPackedRecords(),.original_route_slots=original_slots.data(),
  .compact_route_slots=compact_slots.data(),.route_slot_capacity=2};
 params.expert_mask.assign(512,false);params.prepared_gate_gemm.assign(512,nullptr);
 params.prepared_up_gemm.assign(512,nullptr);params.prepared_down_gemm.assign(512,nullptr);
 const auto formats=originalCPUExpertWorkspaceSourceFormats(*original.loader,7,8,20,512);
 params.cpu_workspace_source=MoELocalExpertStage::CPUExpertWorkspaceSource{
  .gate=formats[0],.up=formats[1],.down=formats[2],.workers=omp_in_parallel()?omp_get_num_threads():omp_get_max_threads(),.execution=CPUExecutionGeometry::local()};
 const auto before_late_weight_bytes=memory->claimedBytes(participant->device,
  PhysicalMemoryOwner::RoutedExpertWeights,PhysicalMemoryMaterializationKind::NewAllocation);
 if(late) {
  auto foreign=params;
  const auto other=staticJointOriginal(0,members);
  MoEOverlayParticipantResidencyRegistry other_banks({.local_participant_ids={staticJointMembers(other).front()},
   .num_layers=1,.num_experts=512,.initial_epoch=1,.projection_preparation=other.preparation,.first_model_layer=7,.original_model=model});
  // A packet-view cannot become a cold graph declaration or retain an RCU node for its lifetime.
  foreign.cpu_joint_layer=banks.cpuJointLayer(id,7,2);
  EXPECT_THROW((void)MoELocalExpertStage(foreign),std::invalid_argument);
  foreign=params;
  PhysicalMemoryPlanBuilder empty_budget;
  empty_budget.add(host,PhysicalMemoryOwner::RoutedExpertWeights,0);
  foreign.physical_memory_authority=std::make_shared<PhysicalMemoryAuthority>(
   std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(empty_budget.build()),0);
  EXPECT_THROW((void)MoELocalExpertStage(foreign),std::invalid_argument);
 }
 MoELocalExpertStage stage(params);test::CPUStageTestWorkspace workspace(stage);
 // Source publication, not mutable registry/store entries, retains the engines.
 manager->preparedWeightStore()->clear();manager->expertGemmRegistry().clear();

 llaminar2::testing::MockDeviceContext execution_context(DeviceId::cpu(),ComputeBackendType::CPU);
 std::array<std::array<float,8>,2> reference{};
 for(int repeat=0;repeat<3;++repeat)for(size_t live:{2u,0u,1u}) {
  input.live_row_count=live;input.live_entry_count=live;
  ASSERT_TRUE(stage.execute(&execution_context));
  EXPECT_EQ(canonical_moe_route_record::readRecordCount(arena->cpuJointPackedRecords()->data(),
   arena->cpuJointPackedRecords()->numel()),live);
  EXPECT_EQ(cold->sourceSelection(),selected);
  for(size_t row=0;row<live;++row) {
   const auto* record=canonical_moe_route_record::record(arena->cpuJointPackedRecords()->data(),row,8);
   EXPECT_EQ(canonical_moe_route_record::readFlatRouteSlot(record,8),row);
   if(repeat==0 && live==2)std::copy_n(record,8,reference[row].data());
   EXPECT_EQ(std::memcmp(record,reference[row].data(),8*sizeof(float)),0);
   for(size_t column=0;column<8;++column)EXPECT_TRUE(std::isfinite(record[column]));
  }
 }

 if(late) {
  // The existing serial-workspace exclusion throws inside the real compute
  // body after its original packet-epoch bank has been bound. No kernel double.
  input.live_row_count=2;input.live_entry_count=2;
  {
   auto busy=arena->cpuGroupedWorkspace()->acquire(2,1,8,local_intermediate,512,7);
   bool original_exception=false;
   try {(void)stage.execute(&execution_context);}
   catch(const std::logic_error& e){EXPECT_NE(std::string(e.what()).find("acquired concurrently"),std::string::npos);original_exception=true;}
   EXPECT_TRUE(original_exception);
  }
  auto successor=*active->bankLease();successor.epoch=4;
  auto next=endpoint->prepareReadyBank(std::move(successor),&error);ASSERT_TRUE(next)<<error;
  ASSERT_EQ(endpoint->installReadyBank(std::move(*next),&error),MoEOverlayParticipantBankInstallStatus::Installed);
  terminal_arrival.reset();active.reset();
  ASSERT_TRUE(endpoint->retire(2));
  EXPECT_FALSE(endpoint->retiredEpochReaders(2).has_value()); // No graph or failed invocation retains this node.
  const auto requirements=stage.getWorkspaceRequirements(2);
  EXPECT_EQ(requirements.total_bytes_with_alignment(),stage.getWorkspaceRequirements(2).total_bytes_with_alignment());
  auto* original_workspace=stage.getWorkspace();
  stage.unbindWorkspace();stage.bindWorkspace(original_workspace);stage.unbindWorkspace();stage.bindWorkspace(original_workspace);
  input.residency_epoch=4;
  for(size_t live:{2u,0u,1u}){
   input.live_row_count=live;input.live_entry_count=live;ASSERT_TRUE(stage.execute(&execution_context));
   for(size_t row=0;row<live;++row)EXPECT_EQ(std::memcmp(canonical_moe_route_record::record(
    arena->cpuJointPackedRecords()->data(),row,8),reference[row].data(),8*sizeof(float)),0);
  }
  active=cold->acquireEpoch(4);ASSERT_TRUE(active);
  auto empty=*active->bankLease();empty.epoch=5;
  auto &empty_layer=empty.layerForModelLayer(7);
  const auto leaving=empty_layer.cpu_fragment_expert_ids;
  for(int expert:leaving)empty_layer.clearExpert(expert);
  auto empty_ready=endpoint->prepareReadyBank(std::move(empty),&error);ASSERT_TRUE(empty_ready)<<error;
  ASSERT_EQ(endpoint->installReadyBank(std::move(*empty_ready),&error),MoEOverlayParticipantBankInstallStatus::Installed);
  input.residency_epoch=5;input.live_row_count=0;input.live_entry_count=0;
  ASSERT_TRUE(stage.execute(&execution_context));
  stage.unbindWorkspace();stage.bindWorkspace(original_workspace);
  ASSERT_TRUE(endpoint->retire(5));EXPECT_FALSE(endpoint->retiredEpochReaders(5).has_value());
  auto last=*active->bankLease();last.epoch=6;
  auto last_ready=endpoint->prepareReadyBank(std::move(last),&error);ASSERT_TRUE(last_ready)<<error;
  ASSERT_EQ(endpoint->installReadyBank(std::move(*last_ready),&error),MoEOverlayParticipantBankInstallStatus::Installed);
  active.reset();ASSERT_TRUE(endpoint->retire(4));EXPECT_FALSE(endpoint->retiredEpochReaders(4).has_value());
  input.residency_epoch=6;input.live_row_count=2;input.live_entry_count=2;ASSERT_TRUE(stage.execute(&execution_context));
  for(size_t row=0;row<2;++row)EXPECT_EQ(std::memcmp(canonical_moe_route_record::record(
   arena->cpuJointPackedRecords()->data(),row,8),reference[row].data(),8*sizeof(float)),0);
  active=cold->acquireEpoch(6);ASSERT_TRUE(active); // Actual installed component observation, not global publication.
  EXPECT_EQ(memory->claimedBytes(participant->device,PhysicalMemoryOwner::RoutedExpertWeights,
   PhysicalMemoryMaterializationKind::NewAllocation),before_late_weight_bytes);
 }

 const auto completed=PerfStatsCollector::snapshot({"moe_overlay_residency.completed_cpu_fragment_execution"});
 ASSERT_EQ(completed.size(),1u);EXPECT_EQ(completed.front().tags.at("expert"),"1");
 EXPECT_GT(completed.front().value,0.0);EXPECT_EQ(completed.front().sequence_minimum_words,(std::vector<uint64_t>{2,2}));
 EXPECT_EQ(completed.front().sequence_maximum_words,(std::vector<uint64_t>{2u,late?6u:2u}));
 EXPECT_EQ(active->bankLease()->layerForModelLayer(7).cpu_fragments[1].arrivalEpoch(),2u);

 } // The fixed executor retires before its setup registry owner.
    std::weak_ptr<ModelContext> original_context=context;
    manager.reset(); context.reset();
    EXPECT_TRUE(original_context.expired());
    EXPECT_FALSE(banks.prepareCPUJointArrival(*candidate,id,7,1,&error));
    const auto actual_gate=active->bankLease()->layerForModelLayer(7).cpu_fragments[1].gate();
    test::CPUProjectionTestWorkspace retained_workspace(1,8,test::cpuProjectionTestRequirements(1,{actual_gate.get()}));
    FP32Tensor retained_input({1,8}),retained_output({1,size_t(local_intermediate)});
    std::fill_n(retained_input.mutable_data(),8,0.25f);
    ASSERT_TRUE(actual_gate->multiply_tensor(&retained_input,&retained_output,1,local_intermediate,8,true,1.f,0.f,nullptr,nullptr,-1,retained_workspace.get()));
    // Layer seven's real source values are expert+1 in the two-layer file;
    // layer six remains expert. This also detects borrowing the representative's weights.
    for(size_t value=0;value<size_t(local_intermediate);++value)EXPECT_EQ(retained_output.data()[value],serial_layers?4.f:2.f);

    if(!late) {
    // The source Store/Registry were deliberately retired above. Initial
    // declaration metadata and captured raw engine indexes must not keep a
    // departed root alive; the existing RCU reader is its final owner.
    const auto before_retirement=memory->claimedBytes(participant->device,
        PhysicalMemoryOwner::RoutedExpertWeights,PhysicalMemoryMaterializationKind::NewAllocation);
    const auto observed_initial=banks.initialBankExpertSelections();
    ASSERT_EQ(observed_initial.size(),1u);
    EXPECT_TRUE(observed_initial.front().matches_frozen_owner_map);
    EXPECT_NE(std::find(observed_initial.front().expert_ids.begin(),observed_initial.front().expert_ids.end(),64),
              observed_initial.front().expert_ids.end());
    ASSERT_TRUE(endpoint->retire(1));
    EXPECT_THROW((void)banks.initialBankExpertSelections(),std::logic_error);
    EXPECT_FALSE(departed.expired());
    EXPECT_EQ(memory->claimedBytes(participant->device,PhysicalMemoryOwner::RoutedExpertWeights,
        PhysicalMemoryMaterializationKind::NewAllocation),before_retirement);
    old.reset();
    EXPECT_FALSE(endpoint->retire(1)); // Existing maintenance edge reclaims drained retired nodes.
    EXPECT_TRUE(departed.expired());
    EXPECT_THROW((void)banks.initialBankExpertSelections(),std::logic_error);
    const auto departed_bytes=3*placement.physicalBytes<float>(size_t(local_intermediate*8));
    EXPECT_EQ(memory->claimedBytes(participant->device,PhysicalMemoryOwner::RoutedExpertWeights,
        PhysicalMemoryMaterializationKind::NewAllocation),before_retirement-departed_bytes);
    }

    auto foreign=staticJointOriginal();
    MoEOverlayResidencyAuthority foreign_authority(staticJointConfig(foreign));
    auto foreign_candidate=test::MoEOverlayJointPlacementCandidateTestAccess::prepare(foreign_authority,foreign.plan->placements);
    EXPECT_FALSE(banks.prepareCPUJointArrival(*foreign_candidate,id,7,71,&error));
}

} // namespace
/** @test Original physical candidate and retired-source retention contract remains unchanged. */
TEST(CPUFragmentEpochBankContract, OriginalSourceIssuesCandidateAndRetainsOldLease)
{exerciseOriginalCPUFragmentGraphLifecycle(false);}
/** @test Every positive CPU cohort width constructs late metadata and replays exact later banks without stale pointers. */
TEST(CPUFragmentEpochBankContract, LateSourceOnlyGraphSurvivesBankRetirementAndExceptions)
{for(int members=2;members<=8;++members){SCOPED_TRACE(members);exerciseOriginalCPUFragmentGraphLifecycle(true,members);}}

/** @test Cold admission binds the actual original context before graph use and cannot resurrect bootstrap. */
TEST(CPUFragmentEpochBankContract, ColdGraphAdmissionBindsOriginalSourceBeforeDeclaration)
{
    exerciseOriginalCPUFragmentGraphLifecycle(true);
}

/** @test Real distinct layer source lifetimes share only the checked serial arena geometry. */
TEST(CPUFragmentEpochBankContract, DistinctLayerSourcesShareOriginalSerialGeometry)
{
    exerciseOriginalCPUFragmentGraphLifecycle(true,2,true);
}
