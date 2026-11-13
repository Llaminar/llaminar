/**
 * @file TiedCPUWeightFinalOwnerLifecycle.inc
 * @brief Actual bounded materializer publishes tied consumers over one original final owner.
 *
 * These CPU-only fixtures use original plan declarations, actual Store engines
 * and PMA claims. Equal values in unrelated named sources are not an ownership
 * capability. No source or native readiness is supplied by a test peer.
 * Temporary source policies cannot shorten the lifetime of committed CPU
 * execution bytes, regardless of alias order or repeated materialization.
 */
TEST(TiedCPUWeightFinalOwnerLifecycle, AliasFirstAndRepeatedMaterializationReuseOriginalOwnerWithoutReads)
{
    for(const bool native:{false,true})for(const bool alias_first:{false,true})
    for(const auto source_policy:{WeightHostPolicy::RequiredUntilGraphMaterialized,
        WeightHostPolicy::ReleasableAfterPreparation,WeightHostPolicy::RequiredForCPUExecution}) {
        SCOPED_TRACE(::testing::Message()<<"native="<<native<<" alias_first="<<alias_first
            <<" source_policy="<<static_cast<int>(source_policy));
        MockModelLoaderBuilder input;
        if(native)input.addQ8_0RandomTensor("token_embd.weight",{4,64})
            .addQ8_0RandomTensor("other_embd.weight",{4,64});
        else input.addFP32RandomTensor("token_embd.weight",{4,64})
            .addFP32RandomTensor("other_embd.weight",{4,64});
        auto loader=input.build();
        WeightManager manager(*loader);
        InferenceStrategy strategy;strategy.model_id=ModelContextId{18761};strategy.devices={DeviceId::cpu()};
        WeightPlan plan(strategy);
        WeightRequirement source;source.canonical_name="token_embd.weight";source.role=WeightRole::Embedding;
        source.target_device=DeviceId::cpu();source.expected_prepared_kind=PreparedWeightKind::CPUResidentTensor;
        source.host_policy=source_policy;
        WeightRequirement alias;alias.canonical_name="output.weight";alias.source_name=source.canonical_name;
        alias.role=WeightRole::LMHead;alias.derivation=WeightDerivationKind::TiedAlias;
        alias.target_device=DeviceId::cpu();alias.expected_prepared_kind=PreparedWeightKind::CpuPackedGemm;
        alias.host_policy=WeightHostPolicy::ReleasableAfterPreparation;
        WeightRequirement other=source;other.canonical_name="other_embd.weight";
        if(alias_first){plan.add(alias);plan.add(source);}else{plan.add(source);plan.add(alias);}
        plan.add(other);
        admitCPUWeightPlansForTest(manager,*loader,{&plan});
        const auto memory=manager.physicalMemoryAuthority();
        const auto first=manager.materialize(plan);
        ASSERT_EQ(first.bindings().size(),3u);
        const auto& original=first.global("token_embd.weight");
        const auto& tied=first.global("output.weight");
        const auto& unrelated=first.global("other_embd.weight");
        ASSERT_TRUE(original.tensor_owner);ASSERT_TRUE(tied.tensor_owner);ASSERT_TRUE(unrelated.tensor_owner);
        EXPECT_EQ(tied.tensor,original.tensor);
        EXPECT_EQ(tied.tensor_owner,original.tensor_owner);
        EXPECT_EQ(original.residency.host_policy,WeightHostPolicy::RequiredForCPUExecution);
        EXPECT_EQ(tied.residency.host_policy,original.residency.host_policy);
        EXPECT_NE(unrelated.tensor_owner,original.tensor_owner);
        ASSERT_EQ(unrelated.tensor->size_bytes(),original.tensor->size_bytes());
        EXPECT_EQ(std::memcmp(unrelated.tensor->raw_data(),original.tensor->raw_data(),original.tensor->size_bytes()),0)
            <<"Equal encoded values must not merge unrelated original named sources";
        EXPECT_EQ(tied.identity.source_instance_id,std::optional<uint64_t>{original.binding_id});
        ASSERT_TRUE(tied.prepared);
        EXPECT_EQ(tied.prepared->kind,PreparedWeightKind::CpuPackedGemm);
        EXPECT_EQ(tied.prepared->binding_id,tied.binding_id);
        EXPECT_TRUE(tied.residency.raw_host_data_available);
        ASSERT_NE(tied.tensor->raw_data(),nullptr);
        const auto store=manager.preparedWeightStore();
        const auto stored=store->binding(*tied.prepared);ASSERT_TRUE(stored);
        EXPECT_EQ(stored->tensor_owner,original.tensor_owner);
        EXPECT_TRUE(stored->residency.raw_host_data_available);
        const auto engine=store->gemmKernelLifetime(*tied.prepared);ASSERT_TRUE(engine);
        EXPECT_TRUE(engine->weights_converted());
        EXPECT_EQ(engine->get_n(),4);EXPECT_EQ(engine->get_k(),64);
        EXPECT_EQ(loader->nativeSourceReadCallCount(),2u)
            <<"Only the two genuine original resident sources may be read";
        const auto reads=loader->nativeSourceReadCallCount();
        const auto claims=memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::PrimaryModelWeights,
            PhysicalMemoryMaterializationKind::NewAllocation);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
        const auto second=manager.materialize(plan);
        const auto& repeated=second.global("output.weight");
        EXPECT_EQ(repeated.tensor_owner,original.tensor_owner);
        EXPECT_TRUE(repeated.residency.raw_host_data_available);
        EXPECT_EQ(repeated.residency.host_policy,WeightHostPolicy::RequiredForCPUExecution);
        EXPECT_EQ(store->gemmKernelLifetime(*repeated.prepared).get(),engine.get());
        EXPECT_EQ(loader->nativeSourceReadCallCount(),reads);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::PrimaryModelWeights,
            PhysicalMemoryMaterializationKind::NewAllocation),claims);
    }
}

TEST(TiedCPUWeightFinalOwnerLifecycle, AmbiguousOriginalSourceRejectsBeforeAnyMaterialization)
{
    auto loader=MockModelLoaderBuilder().addFP32RandomTensor("token_embd.weight",{4,64}).build();
    WeightManager manager(*loader);
    InferenceStrategy strategy;strategy.model_id=ModelContextId{18762};strategy.devices={DeviceId::cpu()};
    WeightPlan plan(strategy);
    WeightRequirement alias;alias.canonical_name="output.weight";alias.source_name="token_embd.weight";
    alias.role=WeightRole::LMHead;alias.derivation=WeightDerivationKind::TiedAlias;
    alias.target_device=DeviceId::cpu();alias.expected_prepared_kind=PreparedWeightKind::CpuPackedGemm;
    WeightRequirement source;source.canonical_name="token_embd.weight";source.role=WeightRole::Embedding;
    source.target_device=DeviceId::cpu();source.expected_prepared_kind=PreparedWeightKind::CPUResidentTensor;
    plan.add(alias);plan.add(source);source.pp_stage=1;plan.add(source);
    admitCPUWeightPlansForTest(manager,*loader,{&plan});
    EXPECT_THROW((void)manager.materialize(plan),std::invalid_argument);
    EXPECT_EQ(loader->nativeSourceReadCallCount(),0u);
    const auto memory=manager.physicalMemoryAuthority();
    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::PrimaryModelWeights,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
