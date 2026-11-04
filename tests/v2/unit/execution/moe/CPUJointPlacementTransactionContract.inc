/**
 * @file CPUJointPlacementTransactionContract.inc
 * @brief Authenticate complete logical changes and every physical CPU fragment destination.
 *
 * These tests use the actual source-bound placement producer and transaction
 * validator. Populations two through eight are metadata tests, not extra MPI
 * sockets. Physical copying, readiness and runtime publication stay separate
 * obligations and are never fabricated by a successful structural validation.
 */

TEST(CPUJointPlacementTransactionContract, OriginalProducerCoversEveryMemberWithoutDuplicatingDemand)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    for (int members = 2; members <= 8; ++members)
    {
        JointPlacementOriginal original(members);
        MoEOverlayResidencyAuthority authority(original.config());
        auto placements = original.plan.placements;
        std::swap(placements[0].routed_expert_tier[0], placements[0].routed_expert_tier[1]);
        auto transaction = MoEOverlayJointPlacementCandidateTestAccess::prepareTransaction(
            authority, placements);
        ASSERT_TRUE(transaction.valid());
        EXPECT_EQ(transaction.previous, authority.snapshot());
        EXPECT_EQ(transaction.expected_epoch, 1u);
        EXPECT_EQ(transaction.candidate->epoch, 2u);
        EXPECT_EQ(transaction.previous->firstModelLayer(), 0);
        EXPECT_EQ(transaction.previous->layerCount(), 1);
        EXPECT_EQ(transaction.previous->expertCount(), 4);
        ASSERT_EQ(transaction.migrations.size(), 2u);
        ASSERT_EQ(transaction.migration_cycles.size(), 1u);
        EXPECT_EQ(transaction.shadow_requirements.size(), std::size_t(members + 1));
        for (const auto &requirement : transaction.shadow_requirements)
            EXPECT_EQ(requirement.slot_count, 1u);

        const auto observed = frozenWindow(17, {7, 11, 3, 5});
        transaction.histogram_window = observed;
        transaction.histogram_generation = observed->generation;
        uint64_t logical_count = 0;
        for (auto &migration : transaction.migrations)
        {
            migration.activation_count = observed->activationCount(migration.layer_idx,
                                                                   migration.expert_id);
            logical_count += migration.activation_count;
        }
        EXPECT_TRUE(transaction.valid());
        EXPECT_EQ(logical_count, 18u);
        transaction.migrations.front().activation_count *= members;
        EXPECT_FALSE(transaction.valid());
        // A typed candidate grants no Static-mode maintenance or publication.
        EXPECT_THROW(authority.proposeFromHistogram(), std::logic_error);
        EXPECT_THROW(authority.proposeInitialPreparedPlacementRestoration(), std::logic_error);
    }
}

TEST(CPUJointPlacementTransactionContract, MissingBalancedChangesCannotPublishUncopiedExperts)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    JointPlacementOriginal original(2, 1, 6);
    MoEOverlayResidencyAuthority authority(original.config());
    auto placements = original.plan.placements;
    std::swap(placements[0].routed_expert_tier[0], placements[0].routed_expert_tier[3]);
    std::swap(placements[0].routed_expert_tier[1], placements[0].routed_expert_tier[4]);
    auto transaction = MoEOverlayJointPlacementCandidateTestAccess::prepareTransaction(
        authority, placements);
    ASSERT_TRUE(transaction.valid());
    ASSERT_EQ(transaction.migrations.size(), 4u);
    auto duplicated = transaction;
    // Preserve the same edge count and balanced slots while omitting one
    // genuine swap. The common capacity analyzer already owns uniqueness.
    duplicated.migrations = {transaction.migrations[0], transaction.migrations[2],
                             transaction.migrations[0], transaction.migrations[2]};
    duplicated.migration_cycles = {{.layer_idx = 0, .migration_indices = {0, 1}},
                                  {.layer_idx = 0, .migration_indices = {2, 3}}};
    const auto duplicate_capacity = analyzeMoEOverlayMigrationCapacity(duplicated.migrations);
    EXPECT_EQ(duplicate_capacity.malformed_edges, 2u);
    EXPECT_EQ(duplicate_capacity.participant_flow_violations, 0u);
    EXPECT_EQ(duplicate_capacity.tier_flow_violations, 0u);
    EXPECT_FALSE(duplicated.valid());
    std::erase_if(transaction.migrations, [](const auto &edge)
                  { return edge.expert_id == 1 || edge.expert_id == 4; });
    transaction.migration_cycles = buildMoEOverlayMigrationCycles(transaction.migrations);
    transaction.shadow_requirements = buildMoEOverlayShadowRequirements(transaction.migrations);
    ASSERT_TRUE(analyzeMoEOverlayMigrationCapacity(transaction.migrations).capacityPreserved());
    EXPECT_FALSE(transaction.valid());
    transaction.migrations.clear();
    transaction.migration_cycles.clear();
    transaction.shadow_requirements.clear();
    EXPECT_FALSE(transaction.valid());
}

TEST(CPUJointPlacementTransactionContract, RejectsForeignSourceCohortAndIncompleteShadowCapacity)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    JointPlacementOriginal original;
    MoEOverlayResidencyAuthority authority(original.config());
    auto placements = original.plan.placements;
    std::swap(placements[0].routed_expert_tier[0], placements[0].routed_expert_tier[1]);
    const auto valid = MoEOverlayJointPlacementCandidateTestAccess::prepareTransaction(
        authority, placements);
    ASSERT_TRUE(valid.valid());
    for (bool source : {false, true})
    {
        auto bad = valid;
        for (auto &edge : bad.migrations)
        {
            auto &owner = source ? edge.source : edge.destination;
            if (owner.shard_cohort)
            {
                owner.shard_cohort = std::shared_ptr<const MoEExpertShardCohort>(
                    owner.shard_cohort.get(), [](auto *) {});
                break;
            }
        }
        EXPECT_FALSE(bad.valid());
    }
    auto bad = valid;
    bad.shadow_requirements.pop_back();
    EXPECT_FALSE(bad.valid());
    bad = valid;
    ++bad.shadow_requirements.front().slot_count;
    EXPECT_FALSE(bad.valid());
    bad = valid;
    bad.shadow_requirements.push_back(bad.shadow_requirements.front());
    EXPECT_FALSE(bad.valid());
    MoEOverlayResidencyAuthority other(original.config());
    bad = valid;
    bad.candidate = MoEOverlayJointPlacementCandidateTestAccess::prepare(other, placements);
    EXPECT_FALSE(bad.valid());
    EXPECT_EQ(authority.snapshot(), valid.previous);
}

TEST(CPUJointPlacementTransactionContract, NoChangeIsStructuralOnlyAndPurposesRemainChecked)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    JointPlacementOriginal original;
    MoEOverlayResidencyAuthority authority(original.config());
    auto unchanged = MoEOverlayJointPlacementCandidateTestAccess::prepareTransaction(
        authority, original.plan.placements);
    ASSERT_TRUE(unchanged.valid());
    EXPECT_TRUE(unchanged.empty());
    unchanged.purpose = MoEOverlayResidencyTransactionPurpose::PreparedContextRestoration;
    EXPECT_TRUE(unchanged.valid());
    unchanged.purpose = MoEOverlayResidencyTransactionPurpose::EconomyCalibration;
    unchanged.calibration_sequence = 1;
    EXPECT_FALSE(unchanged.valid());
    unchanged.purpose = static_cast<MoEOverlayResidencyTransactionPurpose>(255);
    unchanged.calibration_sequence = 0;
    EXPECT_FALSE(unchanged.valid());

    auto placements = original.plan.placements;
    std::swap(placements[0].routed_expert_tier[0], placements[0].routed_expert_tier[1]);
    auto changed = MoEOverlayJointPlacementCandidateTestAccess::prepareTransaction(
        authority, placements);
    ASSERT_TRUE(changed.valid());
    changed.purpose = MoEOverlayResidencyTransactionPurpose::PreparedContextRestoration;
    EXPECT_FALSE(changed.valid()); // Moving away from the original source is not restoration.
}

TEST(CPUJointPlacementTransactionContract, GeometryAcceptsEveryFragmentWithoutCreatingScalarAliases)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    for (int members = 2; members <= 8; ++members)
    {
        JointPlacementOriginal original(members);
        MoEOverlayResidencyAuthority authority(original.config());
        auto placements = original.plan.placements;
        std::swap(placements[0].routed_expert_tier[0], placements[0].routed_expert_tier[1]);
        const auto next = MoEOverlayJointPlacementCandidateTestAccess::prepare(authority, placements);
        for (const auto &snapshot : {authority.snapshot(), next})
        {
            const auto &map = snapshot->placementOwnerMap();
            EXPECT_NO_THROW(map.requireLayerGeometry(1, 4, 0));
            EXPECT_THROW(map.requireLayerGeometry(1, 4, 1), std::logic_error);
            EXPECT_THROW(map.requireLayerGeometry(1, 5, 0), std::logic_error);
            EXPECT_THROW(map.layeredOwnership(1, 4, 0), std::logic_error);
        }
    }
    MoEOverlayResidencyAuthority whole_authority({
        .initial_plan = oneTierTwoParticipantPlan(RoutedExpertResidencyPolicy::StaticById),
        .model_metadata = modelMetadata(),
        .maintenance_mode = MoERebalanceRuntimeMode::Off});
    const auto snapshot = whole_authority.snapshot();
    const auto &whole = snapshot->placementOwnerMap();
    EXPECT_NO_THROW(whole.requireLayerGeometry(snapshot->layerCount(), snapshot->expertCount(),
                                              snapshot->firstModelLayer()));
    EXPECT_NO_THROW(whole.layeredOwnership(snapshot->layerCount(), snapshot->expertCount(),
                                          snapshot->firstModelLayer()));
}

TEST(CPUJointPlacementTransactionContract, RestorationNoOpRetainsOriginalSourceAndSealedGeometry)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    JointPlacementOriginal original(3, 1, 6, 2, 1);
    MoEOverlayResidencyAuthority authority(original.config());
    ASSERT_TRUE(authority.initialPreparedPlacementPublished());
    const auto before = authority.snapshot();
    const auto restoration = MoEOverlayJointPlacementCandidateTestAccess::restoration(authority);
    ASSERT_TRUE(restoration.valid());
    EXPECT_TRUE(restoration.empty());
    EXPECT_EQ(restoration.purpose, MoEOverlayResidencyTransactionPurpose::PreparedContextRestoration);
    EXPECT_EQ(restoration.previous, before);
    EXPECT_EQ(restoration.candidate->epoch, before->epoch + 1);
    EXPECT_EQ(restoration.candidate->originalProjectionPreparation(), before->originalProjectionPreparation());
    EXPECT_EQ(restoration.candidate->layerCount(), 2);
    EXPECT_EQ(restoration.candidate->expertCount(), 6);
    EXPECT_EQ(restoration.candidate->firstModelLayer(), 1);
    EXPECT_EQ(authority.snapshot(), before); // Construction never installs a candidate.
    EXPECT_THROW(authority.proposeInitialPreparedPlacementRestoration(), std::logic_error);
}

/** @test Recorded logical demand uses the actual shared source-bound candidate body, not publication. */
TEST(CPUJointPlacementTransactionContract, HistogramBodyBuildsFixedTargetCandidatesAndBoundedCycles)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    for (bool permuted : {false, true})
    {
        JointPlacementOriginal original(2, 1, 6, 1, 0, permuted);
        auto config = original.config();
        config.max_concurrent_cycles = 1;
        config.shadow_slots_per_endpoint_layer = 1;
        auto enabled_axis = config;
        enabled_axis.participant_rebalance_policy.enabled = true;
        EXPECT_THROW(MoEOverlayResidencyAuthority{enabled_axis}, std::invalid_argument);
        MoEOverlayResidencyAuthority authority(config);
        const auto before = authority.snapshot();
        const auto observed = frozenWindow(17, {11, 9, 7, 5, 3, 1});
        const auto result = MoEOverlayJointPlacementCandidateTestAccess::histogram(authority, observed);
        ASSERT_TRUE(result.valid());
        EXPECT_EQ(result.previous, before);
        EXPECT_EQ(result.histogram_window, observed);
        EXPECT_EQ(result.candidate->originalProjectionPreparation(), original.preparation);
        EXPECT_EQ(authority.snapshot(), before);
        EXPECT_EQ(result.empty(), !permuted);
        if (permuted)
        {
            ASSERT_EQ(result.migration_cycles.size(), 1u);
            ASSERT_EQ(result.migrations.size(), 2u);
            for (const auto &edge : result.migrations)
                EXPECT_EQ(edge.activation_count, observed->activationCount(edge.layer_idx, edge.expert_id));
            EXPECT_EQ(result.shadow_requirements.size(), 3u); // Whole destination + both CPU members.
        }
        auto invalid = std::make_shared<DecodeExpertHistogramWindow>(*observed);
        invalid->num_experts = 5;
        EXPECT_THROW(MoEOverlayJointPlacementCandidateTestAccess::histogram(authority, invalid), std::invalid_argument);
        EXPECT_EQ(authority.snapshot(), before);
        EXPECT_THROW(authority.proposeFromFrozenHistogramWindow(observed), std::logic_error);
    }
}
