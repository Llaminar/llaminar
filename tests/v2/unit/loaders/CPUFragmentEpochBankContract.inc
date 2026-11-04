/**
 * @file CPUFragmentEpochBankContract.inc
 * @brief Exercise a genuine current-source candidate in the existing participant RCU bank.
 *
 * A prepared candidate endpoint is not global routing publication. The original
 * authority remains at epoch one until physical fanout and commit are qualified.
 */
#include "fixtures/moe/MoEOverlayJointPlacementCandidateTestAccess.h"
#include "execution/moe/MoEOverlayParticipantMigration.h"

/** @test Original context, closed arrival and fixed CPU family survive a local candidate bank. */
TEST(CPUFragmentEpochBankContract, OriginalSourceIssuesCandidateAndRetainsOldLease)
{
    using namespace llaminar2;
    const auto original = staticJointOriginal();
    auto context = coldSourceContext(original);
    const ModelContextId model{reinterpret_cast<uint64_t>(context.get())};
    const auto ids = staticJointMembers(original);
    ASSERT_FALSE(ids.empty());
    const int id = ids.front();
    const auto *participant = original.preparation->cpuJointOwnerMap()->participantForId(id);
    ASSERT_TRUE(participant);
    const auto selected = original.preparation->cpuShardSourceFor(7, *participant);
    ASSERT_TRUE(selected);
    MoEOverlayResidencyAuthority authority(staticJointConfig(original));
    auto placements=original.plan->placements;
    std::swap(placements.front().routed_expert_tier[1],placements.front().routed_expert_tier[64]);
    auto candidate = test::MoEOverlayJointPlacementCandidateTestAccess::prepare(
        authority, std::move(placements));
    ASSERT_TRUE(candidate); ASSERT_EQ(candidate->epoch, 2u);
    auto manager = context->concreteWeightManager();
    InferenceStrategy strategy{.model_id=model};
    WeightPlan plan(strategy, PhysicalMemoryOwner::RoutedExpertWeights);
    for (const auto role : {WeightRole::MoEExpertGate, WeightRole::MoEExpertUp, WeightRole::MoEExpertDown})
    {
        const auto name = std::string("blk.7.ffn_") +
            (role==WeightRole::MoEExpertGate?"gate":role==WeightRole::MoEExpertUp?"up":"down") + "_exps.weight";
        plan.add(test::nativeJointRequirementForTest(selected,role,name));
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
        weight_bytes += selected->expertIds().size()*placement.physicalBytes<float>(elements);
        weight_bytes += context->concreteLoader().nativeWeightSourcePlacement().physicalBytes<float>(elements);
    }
    CPUJointExpertRouteStorageLayout joint(selected->cohort(),selected->participantIndex(),2,1);
    const auto layout=std::make_shared<const CPUSerialExpertStorageLayout>(participant->device,2,8,1,
        std::vector<size_t>{1,2},true,10,512,std::nullopt,joint);
    const auto total=weight_bytes+layout->physicalBytes()+layout->workspaceBytes();
    PhysicalMemoryPlanBuilder budget;
    const PhysicalMemoryResource host{participant->world_rank,participant->device,total,total};
    budget.add(host,PhysicalMemoryOwner::RoutedExpertWeights,weight_bytes);
    budget.add(host,PhysicalMemoryOwner::ActivationArena,layout->physicalBytes());
    budget.add(host,PhysicalMemoryOwner::ExecutionWorkspace,layout->workspaceBytes());
    auto memory=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),host.world_rank);
    manager->installPhysicalMemoryAuthority(memory);
    ASSERT_EQ(manager->materialize(plan).bindings().size(),1344u);
    MoEOverlayParticipantResidencyRegistry banks({.local_participant_ids={id},.num_layers=1,
        .num_experts=512,.initial_epoch=1,.retained_epoch_capacity=3,.projection_preparation=original.preparation,
        .first_model_layer=7,.original_model=model});
    std::string error;
    ASSERT_TRUE(banks.bindOriginalPreparedContext(context,original.preparation,&error))<<error;
    ASSERT_TRUE(banks.registerInitialCPUJointBank(manager->expertGemmRegistry(),model,id,&error))<<error;
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
    EXPECT_EQ(fragment.model(),model); EXPECT_EQ(fragment.cohort(),selected->cohort());
    EXPECT_EQ(fragment.preparation(),original.preparation); EXPECT_EQ(fragment.expertId(),1);
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
    const auto active=cold->acquireEpoch(2); ASSERT_TRUE(active);
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
 {
 const auto arena=std::make_shared<MoELocalExpertSerialBufferArena>(MoELocalExpertSerialBufferArena::Config{
  .device_id=DeviceId::cpu(),.row_capacity=2,.row_capacity_buckets={1,2},.d_model=8,.routing_top_k=1,
  .cpu_canonical_route_storage=MoELocalExpertSerialBufferArena::CPUCanonicalRouteStoragePolicy::RetainSerialMaximum,
  .cpu_grouped_scratch_storage=MoELocalExpertSerialBufferArena::CPUGroupedScratchStoragePolicy::RetainSerialMaximum,
  .num_experts=512,.expert_intermediate=10,.logical_participant_id=id,.debug_name="original_joint_cold",
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
 params.d_model=8;params.expert_intermediate=10;params.layer_idx=7;params.runtime_participant_index=id;
 params.cpu_joint_layer=cold;params.expert_registry=&manager->expertGemmRegistry();params.physical_memory_authority=memory;
 params.expert_weight_resolution_policy=MoELocalExpertStage::ExpertWeightResolutionPolicy::RegistryOnly;
 params.cpu_joint_route_producer=MoELocalExpertStage::CPUJointRouteProducerBinding{
  .packed_rows=arena->cpuJointPackedRecords(),.original_route_slots=original_slots.data(),
  .compact_route_slots=compact_slots.data(),.route_slot_capacity=2};
 params.expert_mask.assign(512,false);params.prepared_gate_gemm.assign(512,nullptr);
 params.prepared_up_gemm.assign(512,nullptr);params.prepared_down_gemm.assign(512,nullptr);
 for(int expert:selected->expertIds()) {
  params.expert_mask[expert]=true;const auto& e=cold->acquireInitialConstruction()->layerForModelLayer(7).cpu_fragments[expert];
  params.prepared_gate_gemm[expert]=e.gate().get();params.prepared_up_gemm[expert]=e.up().get();params.prepared_down_gemm[expert]=e.down().get();
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

 } // The fixed executor retires before its setup registry owner.
    std::weak_ptr<ModelContext> original_context=context;
    manager.reset(); context.reset();
    EXPECT_TRUE(original_context.expired());
    EXPECT_FALSE(banks.prepareCPUJointArrival(*candidate,id,7,1,&error));
    const auto actual_gate=active->bankLease()->layerForModelLayer(7).cpu_fragments[1].gate();
    test::CPUProjectionTestWorkspace retained_workspace(1,8,test::cpuProjectionTestRequirements(1,{actual_gate.get()}));
    FP32Tensor retained_input({1,8}),retained_output({1,10});
    std::fill_n(retained_input.mutable_data(),8,0.25f);
    ASSERT_TRUE(actual_gate->multiply_tensor(&retained_input,&retained_output,1,10,8,true,1.f,0.f,nullptr,nullptr,-1,retained_workspace.get()));
    for(size_t value=0;value<10;++value)EXPECT_EQ(retained_output.data()[value],2.f);

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
    const auto departed_bytes=3*placement.physicalBytes<float>(size_t(10*8));
    EXPECT_EQ(memory->claimedBytes(participant->device,PhysicalMemoryOwner::RoutedExpertWeights,
        PhysicalMemoryMaterializationKind::NewAllocation),before_retirement-departed_bytes);

    auto foreign=staticJointOriginal();
    MoEOverlayResidencyAuthority foreign_authority(staticJointConfig(foreign));
    auto foreign_candidate=test::MoEOverlayJointPlacementCandidateTestAccess::prepare(foreign_authority,foreign.plan->placements);
    EXPECT_FALSE(banks.prepareCPUJointArrival(*foreign_candidate,id,7,71,&error));
}
