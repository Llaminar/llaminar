/**
 * @file LocalNativePreparationBOMContract.inc
 * @brief Verify global source registry and rank-local billing contracts through MemoryPlanner.
 */
/** @test Foreign zero rows stay in original source, never in local capacity contributions. */
TEST(Test__MemoryPlanner, OriginalNativeGlobalRegistryBillsOnlyLocalResourcesOnceAcrossGroups) {
    using Access=test::NativeContextPreparationReplayAccess;
    for(const auto device:{DeviceId::cuda(83),DeviceId::rocm(83)})
    for(bool remote_gpu:{false,true}) {
        const std::array localDevices{device};
        std::vector<PhysicalMemoryResource> observed{{0,DeviceId::cpu(),8ull<<30,7ull<<30},
            {0,device,4ull<<30,4ull<<30},{1,DeviceId::cpu(),8ull<<30,7ull<<30}};
        if(remote_gpu)observed.push_back({1,device,4ull<<30,3ull<<30});
        Access::seed(device,[device](std::shared_ptr<PreparedGPUWorkerStack> root) {
            return Access::construct<SourcePlanningWorker>(std::move(root),device.gpu_ordinal());
        });
        auto source=Access::preparedSource<SourcePlanningWorker>(localDevices,observed);
        PlanningSourceRetirement retirement(source,localDevices);
        const auto original=source->mandatoryBOM();ASSERT_EQ(original.resources().size(),remote_gpu?4u:3u);
        DevicePlanConfig cfg;cfg.world_rank=0;cfg.device=device;
        cfg.device_total_bytes=cfg.device_free_bytes=4ull<<30;cfg.associated_host_memory=observed.front();
        cfg.device_compute_units=80;cfg.max_seq_len=64;cfg.activation_seq_len=1;cfg.first_layer=0;cfg.last_layer=23;
        cfg.native_context_preparation=source;
        const auto profile=createTestProfile();
        const std::vector<std::vector<DevicePlanConfig>> groups{{cfg,cfg},{cfg}};
        const auto planned=MemoryPlanner::planGroups(profile,groups);ASSERT_EQ(planned.size(),2u);
        for(const auto& plan:planned) {
            for(const auto& emitted:plan.devices)EXPECT_EQ(emitted.bom().resource().world_rank,0)
                << "Foreign zero rows are global source observations, not rank-local contributions";
            const auto physical=plan.physicalPlan();
            EXPECT_EQ(physical.find({1,DeviceId::cpu()}),nullptr);
            EXPECT_EQ(physical.find({1,device}),nullptr);
        }
        const auto first=planned[0].physicalPlan(),second=planned[1].physicalPlan();
        for(const auto& declared:original.resources())if(declared.resource().world_rank==0) {
            const PhysicalMemoryAllocatorIdentity identity{0,declared.resource().device};
            const auto* a=first.find(identity);const auto* b=second.find(identity);ASSERT_NE(a,nullptr);
            EXPECT_EQ(a->bytes(PhysicalMemoryOwner::NativeExecutionContext),declared.bytes(PhysicalMemoryOwner::NativeExecutionContext));
            EXPECT_EQ(b?b->bytes(PhysicalMemoryOwner::NativeExecutionContext):0u,0u);
        }
        EXPECT_EQ(source->mandatoryBOM().summary(),original.summary());
        EXPECT_EQ(source->mandatoryBOM().resources().size(),remote_gpu?4u:3u);
        // A second planning transaction owns its own setup deduplication scope;
        // it bills one original source again, never an extra native allocation.
        const auto repeated=MemoryPlanner::planGroups(profile,groups);
        for(size_t i=0;i<planned.size();++i)EXPECT_EQ(repeated[i].physicalPlan().summary(),planned[i].physicalPlan().summary());
    }
}
