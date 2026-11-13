/**
 * @file NativeWeightSourceBOMContract.inc
 * @brief Common source staging belongs to the original CPU allocator, not a joint workspace.
 */

namespace llaminar2 {
/** @test The planner prices the common source root independently of expert graph workspace. */
TEST(NativeWeightLoadStagingContract, CommonSourceDeclarationIsOncePerRankAndRejectsMissingQuote)
{
    Fixture fixture(DeviceType::CPU,1);
    auto configs=buildRankMemoryPlanInputs(fixture.request());ASSERT_EQ(configs.size(),1u);
    auto cfg=configs.front();
    const auto geometry=resolveNativeWeightLoadStaging({4,4,64,1000000},CPUStoragePlacement::local(),4096);
    cfg.native_weight_load_staging=geometry;
    cfg.weight_load_staging.host_bytes=geometry.physical_bytes;
    EXPECT_FALSE(cfg.cpu_serial_expert_storage);
    const std::vector<std::vector<DevicePlanConfig>> groups{{cfg,cfg},{cfg}};
    const auto plans=MemoryPlanner::planGroups(fixture.model,groups);
    EXPECT_EQ(plans[0].physicalPlan().find({0,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::WeightLoadStaging),geometry.physical_bytes);
    EXPECT_EQ(plans[1].physicalPlan().find({0,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::WeightLoadStaging),0u);
    auto wrong=cfg;wrong.weight_load_staging.host_bytes+=64;
    EXPECT_THROW((void)MemoryPlanner::plan(fixture.model,{wrong}),std::invalid_argument);
    auto missing=cfg;missing.native_weight_load_staging.reset();
    EXPECT_THROW((void)MemoryPlanner::plan(fixture.model,{missing}),std::invalid_argument);
    auto unpriced=cfg;unpriced.weight_load_staging.host_bytes=0;
    EXPECT_THROW((void)MemoryPlanner::plan(fixture.model,{unpriced}),std::invalid_argument);
}
} // namespace llaminar2
