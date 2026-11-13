/**
 * @file CPUJointReusableRegistryContract.inc
 * @brief Actual CPU source/Store/bank consumer after atomic terminal engine replacement.
 *
 * The existing one-actor fixture owns real original ModelContext, GGUF, PMA,
 * Store and installed fragment bank. Component-only private map validation
 * substitutes genuine same-byte prepared engines; it issues no physical seal
 * or native readiness. The real two-rank Fabric test owns that positive proof.
 */
#include "fixtures/loaders/ExpertGemmRegistryResidencyTestAccess.h"

TEST(CPUJointReusableRegistryContract, PreservesCommittedSourceAndConsumesReplacementEnginesAtomically)
{
    using namespace llaminar2;
    const auto original=staticJointOriginal();
    auto context=coldSourceContext(original);
    const ModelContextId model{reinterpret_cast<std::uint64_t>(context.get())};
    auto manager=context->concreteWeightManager();
    const int id=staticJointMembers(original).front();
    const auto *participant=original.preparation->cpuJointOwnerMap()->participantForId(id);
    ASSERT_TRUE(participant);
    const auto selected=original.preparation->cpuShardSourceFor(7,*participant);
    ASSERT_TRUE(selected);
    const int expert=selected->expertIds().front();ASSERT_EQ(expert,64);
    InferenceStrategy strategy{.model_id=model};
    WeightPlan source_plan(strategy,PhysicalMemoryOwner::RoutedExpertWeights);
    WeightPlan destination_plan(strategy,PhysicalMemoryOwner::RoutedExpertWeights);
    const auto placement=CPUStoragePlacement::onNode(participant->address.numa_node);
    std::size_t weight_bytes=0;
    for(const auto role:{WeightRole::MoEExpertGate,WeightRole::MoEExpertUp,WeightRole::MoEExpertDown}) {
        const auto name=std::string("blk.7.ffn_")+(role==WeightRole::MoEExpertGate?"gate":role==WeightRole::MoEExpertUp?"up":"down")+"_exps.weight";
        source_plan.add(test::nativeJointRequirementForTest(selected,role,name));
        const auto projection=selected->ownership().projection(role);
        WeightRequirement request;
        request.canonical_name=name;request.role=role;request.layer=7;request.expert=expert;
        request.target_device=participant->device;request.derivation=WeightDerivationKind::ExpertSlice;
        request.bypass_tensor_parallel=true;request.expected_prepared_kind=PreparedWeightKind::CpuPackedGemm;
        // Independent component destination, never registered as an original
        // source or advertised as a declared inference participant.
        request.overlay_domain=participant->domain_name+"/terminal_component";
        request.overlay_participant_index=participant->domain_participant_index;
        request.overlay_participant_world_rank=participant->world_rank;
        request.slice={.source_rows=std::size_t(projection.source_columns),.source_cols=std::size_t(projection.source_rows),
            .row_start=std::size_t(projection.first_column),.row_count=std::size_t(projection.columns),
            .col_start=std::size_t(projection.first_row),.col_count=std::size_t(projection.rows),
            .expert_count=1,.expert_ids={expert}};
        destination_plan.add(std::move(request));
        const auto elements=std::size_t(projection.rows)*projection.columns;
        weight_bytes+=selected->expertIds().size()*placement.physicalBytes<float>(elements);
        weight_bytes+=context->concreteLoader().nativeWeightSourcePlacement().physicalBytes<float>(elements);
    }
    PhysicalMemoryPlanBuilder budget;
    const PhysicalMemoryResource host{participant->world_rank,participant->device,weight_bytes,weight_bytes};
    budget.add(host,PhysicalMemoryOwner::RoutedExpertWeights,weight_bytes);
    auto memory=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),host.world_rank);
    manager->installPhysicalMemoryAuthority(memory);
    (void)manager->materialize(source_plan);
    auto &registry=manager->expertGemmRegistry();
    MoEOverlayParticipantResidencyRegistry banks({.local_participant_ids={id},.num_layers=1,
        .num_experts=512,.initial_epoch=1,.retained_epoch_capacity=3,
        .projection_preparation=original.preparation,.first_model_layer=7,.original_model=model});
    std::string error;
    ASSERT_TRUE(banks.bindOriginalPreparedContext(context,original.preparation,&error))<<error;
    ASSERT_TRUE(banks.registerInitialCPUJointBank(registry,model,id,&error))<<error;
    auto installed=banks.endpoint(id)->acquire(1);ASSERT_TRUE(installed);
    const auto before=registry.acquireCPUJointLayer(model,*participant,7);ASSERT_TRUE(before);
    const auto incoming=manager->materialize(destination_plan);
    ASSERT_EQ(incoming.bindings().size(),3u);
    std::array<std::shared_ptr<ITensorGemm>,3> engines;
    for(const auto &binding:incoming.bindings()) {
        const auto role=binding.identity.role==WeightRole::MoEExpertGate?0u:binding.identity.role==WeightRole::MoEExpertUp?1u:2u;
        engines[role]=manager->preparedWeightStore()->gemmKernelLifetime(*binding.prepared);
    }
    ASSERT_NE(engines[0],before->experts[expert].gate);
    auto arrival=banks.prepareCPUJointArrivalFromInstalledBank(installed,id,7,expert,&error);
    ASSERT_TRUE(arrival)<<error;
    for(std::size_t role=0;role<3;++role)
        ASSERT_TRUE(arrival->publish(static_cast<ExpertTierWeightProjection>(role),engines[role],&error))<<error;
    MoEOverlayPreparedCPUFragmentPayload replacement;
    ASSERT_TRUE(arrival->completeCPUFragment(replacement,&error))<<error;
    auto component=*installed;
    component.layers[0].setResidentCPUFragment(expert,replacement);
    ASSERT_TRUE(component.layers[0].experts.empty());
    // This is deliberately only private component map validation. No physical
    // seal is constructed and no participant bank/capture becomes ready here.
    for(int replay=0;replay<20;++replay) {
        ASSERT_TRUE(test::ExpertGemmRegistryResidencyTestAccess::replaceCPUFragmentBank(registry,banks,component,&error))<<error;
        const auto acquired=registry.acquireCPUJointLayer(model,*participant,7);ASSERT_TRUE(acquired);
        EXPECT_EQ(acquired->experts[expert].gate,engines[0]);
        EXPECT_EQ(acquired->experts[expert].up,engines[1]);EXPECT_EQ(acquired->experts[expert].down,engines[2]);
        ASSERT_EQ(acquired->sources.size(),before->sources.size());
        for(std::size_t i=0;i<before->sources.size();++i) {
            const auto &old=before->sources[i],&now=acquired->sources[i];
            EXPECT_EQ(now.binding_id,old.binding_id);ASSERT_TRUE(now.prepared);ASSERT_TRUE(old.prepared);
            EXPECT_EQ(now.prepared->model_id,old.prepared->model_id);EXPECT_EQ(now.prepared->binding_id,old.prepared->binding_id);
            EXPECT_EQ(now.prepared->device,old.prepared->device);EXPECT_EQ(now.prepared->kind,old.prepared->kind);
            EXPECT_EQ(now.native_source,old.native_source);EXPECT_EQ(now.cpu_joint_source,old.cpu_joint_source);
            EXPECT_EQ(now.tensor,old.tensor);EXPECT_EQ(now.tensor_owner,old.tensor_owner);
            EXPECT_FALSE(now.tensor_owner.owner_before(old.tensor_owner));EXPECT_FALSE(old.tensor_owner.owner_before(now.tensor_owner));
        }
        const auto qualified=registry.acquireStaticProjectionLayer(before->scope,before->projection,before->sources);
        ASSERT_TRUE(qualified);EXPECT_EQ(qualified->experts[expert].gate,engines[0]);
        EXPECT_EQ(registry.getEngineForDomain(participant->domain_name,participant->device,7,expert,ExpertGemmRegistry::WeightRole::GATE),nullptr);
    }
    for(int mutation=0;mutation<3;++mutation) {
        auto bad=component;
        if(mutation==0)bad.layers[0].cpu_fragment_family.reset();
        if(mutation==1)bad.layers[0].cpu_fragments[expert]={};
        if(mutation==2)bad.participant_id=staticJointMembers(original).back();
        EXPECT_FALSE(test::ExpertGemmRegistryResidencyTestAccess::replaceCPUFragmentBank(registry,banks,bad,&error));
        const auto unchanged=registry.acquireCPUJointLayer(model,*participant,7);ASSERT_TRUE(unchanged);
        EXPECT_EQ(unchanged->experts[expert].gate,engines[0]);
    }
}
