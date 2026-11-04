/**
 * @file CPUJointGraphOwnershipContract.inc
 * @brief Verify Qwen cold preparation and scalar projection contracts for CPU cohorts.
 *
 * The existing tiny original GGUF fixture owns source metadata. These checks
 * exercise production map construction, graph preparation and routing projection
 * without creating a GPU or MPI communicator. Native CPU2 inference remains a
 * separate admission and numerical gate.
 */
#include "execution/moe/MoEGraphOwnerProjection.h"
#include "utils/ForwardGraphTemplateFixture.h"

/** @test Continuation tables keep joint work external and never select a fragment as a whole owner. */
TEST(CPUJointGraphOwnership, ExternalJointProjectionPreservesOriginalMembership)
{
    using namespace llaminar2;
    for (int members = 2; members <= 8; ++members)
    for (const auto order : {RoutedExpertOwnerOrder::Ordinal, RoutedExpertOwnerOrder::Random})
    {
        auto plan = jointSourcePlan();
        plan.owner_order = order;
        plan.domains[1].participants.clear();
        plan.domains[1].world_ranks.clear();
        for (int member = 0; member < members; ++member)
        {
            plan.domains[1].participants.push_back(GlobalDeviceAddress::cpu(member));
            plan.domains[1].world_ranks.push_back(member + 1);
        }
        const auto owners = MoEExpertOwnerMap::buildShardPlacement(plan, {jointManifest()});
        const auto global = ownerParticipantsFromMap(owners, 7, 512);
        const auto continuation = domainLocalOwnerParticipantsFromMap(owners, 7, 512, "compute");
        ASSERT_EQ(global.size(), 512u);
        ASSERT_EQ(continuation.size(), 512u);
        for (int expert = 0; expert < 512; ++expert)
        {
            EXPECT_EQ(global[expert], expert < 64 ? 0 : -1);
            EXPECT_EQ(continuation[expert], expert < 64 ? 0 : -1);
            EXPECT_EQ(owners.fragmentCountForExpert(7, expert), expert < 64 ? 1u : static_cast<std::size_t>(members));
        }
        for (const auto &participant : owners.participants())
        {
            const auto membership = jointRouteMembershipFromMap(owners, 7, 512, participant.participant_id, 11);
            const bool joint = participant.domain_name == "cpu_joint";
            EXPECT_EQ(membership.enabled(), joint);
            for (int expert = 0; expert < 512; ++expert)
            {
                EXPECT_EQ(membership.contains(expert), joint && expert >= 64);
                int resolved = -99;
                ASSERT_TRUE(membership.resolve(expert, global[expert], &resolved));
                EXPECT_EQ(resolved, joint && expert >= 64 ? participant.participant_id : global[expert]);
            }
        }
        EXPECT_THROW(owners.ownerFor(7, 64), std::logic_error);
        EXPECT_THROW(domainLocalOwnerParticipantsFromMap(owners, 7, 512, "cpu_joint"), std::logic_error);
        EXPECT_EQ(domainLocalOwnerParticipantsFromMap(owners, 7, 512, "absent"), std::vector<int>(512, -1));
        EXPECT_TRUE(ownerParticipantsFromMap(owners, 7, 0).empty());
        EXPECT_EQ(ownerParticipantsFromMap(owners, 8, 4), std::vector<int>(4, -1));
        EXPECT_THROW(jointRouteMembershipFromMap(owners, 7, 512, 999, 11), std::invalid_argument);
        EXPECT_THROW(jointRouteMembershipFromMap(owners, 7, 512, 1, 0), std::invalid_argument);
    }
}

/** @test Ordinary whole-expert routing retains both global and domain-local coordinates. */
TEST(CPUJointGraphOwnership, OrdinaryWholeOwnerProjectionIsUnchanged)
{
    using namespace llaminar2;
    auto plan = jointSourcePlan();
    plan.domains[1].routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
    for (const auto order : {RoutedExpertOwnerOrder::Ordinal, RoutedExpertOwnerOrder::Random})
    {
        plan.owner_order = order;
        const auto owners = MoEExpertOwnerMap::build(plan);
        const auto global = ownerParticipantsFromMap(owners, 7, 512);
        const auto local = domainLocalOwnerParticipantsFromMap(owners, 7, 512, "cpu_joint");
        for (int expert = 0; expert < 512; ++expert)
        {
            const auto *row = owners.ownerFor(7, expert);
            ASSERT_NE(row, nullptr);
            const auto *participant = owners.participantForId(row->owner_participant);
            ASSERT_NE(participant, nullptr);
            EXPECT_EQ(global[expert], row->owner_participant);
            EXPECT_EQ(local[expert], participant->domain_name == "cpu_joint"
                ? participant->domain_participant_index : -1);
        }
    }
}

/** @test Repeated actual Qwen preparation borrows one original source map without rebuilding whole ownership. */
TEST(CPUJointGraphOwnership, ActualQwenFamilyBorrowsOriginalPreparedMap)
{
    using namespace llaminar2;
    const auto original = staticJointOriginal();
    test::ForwardTemplateConstructionFixture fixture;
    auto config = fixture.source.config;
    // The runtime already retains the original immutable placement plan.
    config.moe.expert_overlay_runtime_plan = resolveMoEExpertOverlayRuntimePlan(
        original.plan, {.current_world_rank = 0, .validate_mvp_root_reachability = false});
    config.moe.num_experts = 512;
    config.moe.expert_overlay_participant_residency =
        std::make_shared<MoEOverlayParticipantResidencyRegistry>(
            MoEOverlayParticipantResidencyRegistry::Config{
                .local_participant_ids = {}, .num_layers = 1, .num_experts = 512,
                .initial_epoch = 1, .projection_preparation = original.preparation,
                .first_model_layer = 7, .original_model = ModelContextId{155}});
    Qwen35MoEGraph graph(config, fixture.source.mpi);
    graph.setWeights(fixture.weights());
    const auto builds = MoEExpertOwnerMap::initialBuildAttempts();
    for (int iteration = 0; iteration < 20; ++iteration)
        EXPECT_NO_THROW(graph.prepareForwardGraphTemplate(DeviceId::cpu()));
    const auto stats = graph.forwardGraphTemplateStats();
    EXPECT_EQ(MoEExpertOwnerMap::initialBuildAttempts(), builds);
    EXPECT_EQ(stats.preparations, 1u);
    EXPECT_EQ(stats.canonical_owner_map_builds, 0u);
    EXPECT_EQ(stats.original_owner_map_reuses, 1u);
    EXPECT_EQ(stats.continuation_resolutions, 1u);
    EXPECT_EQ(&config.moe.expert_overlay_participant_residency->initialOwnerMap(),
              original.preparation->cpuJointOwnerMap().get());

    // A selected joint mode without its original source preparation must fail;
    // no whole-owner reconstruction is a valid replacement.
    config.moe.expert_overlay_participant_residency.reset();
    Qwen35MoEGraph missing(config, fixture.source.mpi);
    missing.setWeights(fixture.weights());
    EXPECT_THROW(missing.prepareForwardGraphTemplate(DeviceId::cpu()), std::invalid_argument);
}

#include "execution/moe/MoECPUExpertWorkspaceSource.h"
/** @test Main-layer CPU cohort workspace formats use the same original loader geometry. */
TEST(CPUJointGraphOwnership, WorkspaceFormatsRetainOriginalSourceGeometry)
{
    using namespace llaminar2;
    const auto original=staticJointOriginal();
    EXPECT_EQ(originalCPUExpertWorkspaceSourceFormats(*original.loader,7,8,20,512),
        (std::array<TensorType,3>{TensorType::FP32,TensorType::FP32,TensorType::FP32}));
    EXPECT_THROW(originalCPUExpertWorkspaceSourceFormats(*original.loader,7,8,10,512),std::logic_error);
    EXPECT_THROW(originalCPUExpertWorkspaceSourceFormats(*original.loader,7,8,20,448),std::logic_error);
    EXPECT_THROW(originalCPUExpertWorkspaceSourceFormats(*original.loader,8,8,20,512),std::logic_error);
}
