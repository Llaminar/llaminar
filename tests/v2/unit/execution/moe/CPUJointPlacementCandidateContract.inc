/**
 * @file CPUJointPlacementCandidateContract.inc
 * @brief Source-native live-map construction without enabling static CPU cohort movement.
 *
 * Real tiny GGUF directory metadata supplies the original source/cuts. Cohort
 * populations2/4/8 are declaration tests, not socket or MPI execution claims.
 * No weight payload, physical bank, device context or readiness is fabricated.
 */
#include "../../../fixtures/moe/MoEOverlayJointPlacementCandidateTestAccess.h"
#include "../../../utils/PlanningGGUFFixture.h"
#include "../../../utils/CPUJointSourceTestPlan.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "tensors/TensorFactory.h"

namespace llaminar2::test
{
    namespace
    {
        /** @brief Own genuine source metadata and its single complete fixed-cohort preparation. */
        struct JointPlacementOriginal
        {
            int expert_count;
            int layers;
            int first_layer;
            PlanningGGUFFixture file;
            std::shared_ptr<MPIContext> mpi = MPIContextFactory::self();
            TensorFactory factory{*mpi};
            ModelLoader loader{&factory};
            MoERoutedExpertPlacementPlan plan;
            std::shared_ptr<const MoEExpertOverlayPreparationPlan> preparation;
            /**
             * @param members Full declared CPU cohort; never a hardware-availability probe.
             * @param whole_members Original inert accelerator owner count for stable-map tests.
             * @param experts Original tiny source E; three experts remain in the CPU tier.
             * @param layer_count Original contiguous layer count.
             * @param first Nonzero model-global layer origin when testing selected scopes.
             * @param permuted Swap two complete whole/cohort pairs before original preparation.
             * @param whole_backend Original inert GPU metadata for the whole-owner domain.
             */
            explicit JointPlacementOriginal(int members = 2, int whole_members = 1, int experts = 4, int layer_count = 1, int first = 0, bool permuted = false, DeviceType whole_backend = DeviceType::CUDA)
                : expert_count(experts), layers(layer_count), first_layer(first),
                  file(true, false, GGUFTensorType::F32, 64, std::nullopt, 8,
                       PlanningGGUFFixture::Geometry::FullAttentionGraph, uint32_t(first + layer_count), uint32_t(experts))
            {
                if (!loader.loadModel(file.path()))
                    throw std::runtime_error("Joint placement metadata source failed to parse");
                plan = cpuJointSourceTestPlan();
                plan.first_model_layer = first_layer;
                plan.routed_tiers[0].max_experts_per_layer = experts - 3;
                plan.routed_tiers[1].max_experts_per_layer = 3;
                std::vector<int> tiers(std::size_t(experts), 1);
                std::fill_n(tiers.begin(), experts - 3, 0);
                plan.placements.clear();
                for (int layer = first_layer; layer < first_layer + layers; ++layer)
                    plan.placements.push_back({.layer = layer, .routed_expert_tier = tiers});
                plan.domains[0].participants.clear();
                if (whole_backend != DeviceType::CUDA && whole_backend != DeviceType::ROCm)
                    throw std::invalid_argument("Joint fixture whole owners require explicit GPU backend metadata");
                plan.domains[0].backend = whole_backend == DeviceType::CUDA
                    ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
                for (int owner = 0; owner < whole_members; ++owner)
                    plan.domains[0].participants.push_back(GlobalDeviceAddress::fromLocalDeviceId({whole_backend, owner}));
                auto &cohort = plan.domains[1];
                cohort.participants.clear();
                cohort.world_ranks.clear();
                for (int member = 0; member < members; ++member)
                {
                    cohort.participants.push_back(GlobalDeviceAddress::cpu(member));
                    cohort.world_ranks.push_back(member);
                }
                if (permuted)
                    for (auto &placement : plan.placements)
                    {
                        std::swap(placement.routed_expert_tier.at(0), placement.routed_expert_tier.at(experts - 3));
                        std::swap(placement.routed_expert_tier.at(1), placement.routed_expert_tier.at(experts - 2));
                    }
                plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
                plan.authority_execution = resolveMoEOverlayAuthorityExecutionKind(plan);
                const auto original = std::make_shared<const MoERoutedExpertPlacementPlan>(plan);
                const auto runtime = resolveMoEExpertOverlayRuntimePlan(
                    original, {.current_world_rank = 0, .validate_mvp_root_reachability = false});
                preparation = std::make_shared<const MoEExpertOverlayPreparationPlan>(
                    MoEExpertOverlayPreparationPlan::build(*runtime, loader));
            }
            /** @return Original Static-only production config with no physical movement permission.
             */
            MoEOverlayResidencyAuthority::Config config() const
            {
                return {.initial_plan = plan,
                        .model_metadata = {.num_layers = layers,
                                           .num_experts = expert_count,
                                           .d_model = 256,
                                           .routed_intermediate_size = 64,
                                           .main_inference_layer_count = first_layer + layers,
                                           .first_model_layer = first_layer},
                        .projection_preparation = preparation,
                        .maintenance_mode = MoERebalanceRuntimeMode::Off,
                        .participant_rebalance_policy = {.enabled = false}};
            }
        };
    } // namespace
} // namespace llaminar2::test

TEST(CPUJointPlacementCandidateContract,
     ActualProducerKeepsOriginalSourceAndPublishesDistinctCheckedLivePlacement)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    for (const int members : {2, 4, 8})
    {
        JointPlacementOriginal original(members);
        MoEOverlayResidencyAuthority authority(original.config());
        const auto initial = authority.snapshot();
        ASSERT_TRUE(initial->valid());
        const auto source = original.preparation->cpuJointOwnerMap();
        ASSERT_EQ(initial->jointOwnerMap(), source);
        ASSERT_EQ(&initial->placementOwnerMap(), source.get());
        const auto cohort = source->placementFor(0, 1)->shard_cohort;
        ASSERT_TRUE(cohort);
        const auto *localMember =
            source->participantForId(cohort->participants().front().participant_id);
        ASSERT_TRUE(localMember);
        const auto originalSelection = original.preparation->cpuShardSourceFor(0, *localMember);
        ASSERT_TRUE(originalSelection);
        EXPECT_FALSE(originalSelection->contains(0));
        EXPECT_TRUE(originalSelection->contains(1));
        const auto builds = MoEExpertOwnerMap::initialBuildAttempts();
        for (int repeat = 0; repeat < 3; ++repeat)
        {
            auto placements = original.plan.placements;
            std::swap(placements[0].routed_expert_tier[0], placements[0].routed_expert_tier[1]);
            const auto candidate =
                MoEOverlayJointPlacementCandidateTestAccess::prepare(authority, placements);
            ASSERT_TRUE(candidate->valid());
            EXPECT_EQ(candidate->epoch, 2u);
            EXPECT_EQ(candidate->originalProjectionPreparation(), original.preparation);
            EXPECT_EQ(&candidate->originalOwnerMap(), source.get());
            EXPECT_NE(candidate->jointOwnerMap(), source);
            EXPECT_EQ(&candidate->placementOwnerMap(), candidate->jointOwnerMap().get());
            const auto *incoming = candidate->placementOwnerMap().placementFor(0, 0);
            ASSERT_TRUE(incoming && incoming->shard_cohort);
            EXPECT_EQ(incoming->shard_cohort, cohort);
            EXPECT_EQ(incoming->owner_participant, -1);
            EXPECT_FALSE(incoming->device.is_valid());
            EXPECT_EQ(candidate->placementOwnerMap().placementFor(0, 1)->owner_participant,
                      source->placementFor(0, 0)->owner_participant);
            for (const auto &member : cohort->participants())
            {
                EXPECT_EQ(
                    candidate->placementOwnerMap().expertsForParticipant(0, member.participant_id),
                    (std::vector<int>{0, 2, 3}));
                EXPECT_EQ(source->expertsForParticipant(0, member.participant_id),
                          (std::vector<int>{1, 2, 3}));
            }
            // This is a checked placement, NOT a physical transaction/ready token.
            MoEOverlayResidencyTransaction blocked;
            blocked.expected_epoch = 1;
            blocked.previous = initial;
            blocked.candidate = candidate;
            EXPECT_FALSE(blocked.valid());
            EXPECT_EQ(authority.snapshot(), initial);
        }
        EXPECT_EQ(MoEExpertOwnerMap::initialBuildAttempts(), builds);
        EXPECT_THROW(authority.proposeFromHistogram(), std::logic_error);
        EXPECT_THROW(authority.proposeInitialPreparedPlacementRestoration(), std::logic_error);
    }
}

TEST(CPUJointPlacementCandidateContract,
     ChangedCapacityForeignTopologyAndPublicSnapshotForgeryReject)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    JointPlacementOriginal original;
    MoEOverlayResidencyAuthority authority(original.config());
    const auto initial = authority.snapshot();
    auto placements = original.plan.placements;
    placements[0].routed_expert_tier[0] = 1;
    EXPECT_THROW(MoEOverlayJointPlacementCandidateTestAccess::prepare(authority, placements),
                 std::invalid_argument);
    placements = original.plan.placements;
    placements[0].layer = 99;
    EXPECT_THROW(MoEOverlayJointPlacementCandidateTestAccess::prepare(authority, placements),
                 std::invalid_argument);
    placements = original.plan.placements;
    placements[0].routed_expert_tier.pop_back();
    EXPECT_THROW(MoEOverlayJointPlacementCandidateTestAccess::prepare(authority, placements),
                 std::invalid_argument);
    MoEOverlayResidencyAuthority foreignAuthority(original.config());
    EXPECT_THROW(MoEOverlayJointPlacementCandidateTestAccess::prepareExpected(
                     authority, foreignAuthority.snapshot(), 1, original.plan.placements),
                 std::logic_error);
    EXPECT_THROW(MoEOverlayJointPlacementCandidateTestAccess::prepareExpected(
                     authority, initial, 2, original.plan.placements),
                 std::logic_error);
    const auto unpublished =
        MoEOverlayJointPlacementCandidateTestAccess::prepare(authority, original.plan.placements);
    EXPECT_THROW(MoEOverlayJointPlacementCandidateTestAccess::prepareExpected(
                     authority, unpublished, 2, original.plan.placements),
                 std::logic_error);
    auto independentOwner =
        std::shared_ptr<const MoEOverlayResidencySnapshot>(initial.get(), [](auto *) {});
    EXPECT_THROW(MoEOverlayJointPlacementCandidateTestAccess::prepareExpected(
                     authority, independentOwner, 1, original.plan.placements),
                 std::logic_error);
    auto forged = *initial;
    forged.epoch = 2;
    EXPECT_FALSE(forged.valid());
    forged = *initial;
    forged.placement_plan =
        std::make_shared<const MoERoutedExpertPlacementPlan>(*initial->placement_plan);
    EXPECT_FALSE(forged.valid());
    forged = *initial;
    forged.placement_plan = std::shared_ptr<const MoERoutedExpertPlacementPlan>(
        initial->placement_plan.get(), [](auto *) {});
    EXPECT_FALSE(forged.valid());
    auto changed = original.plan;
    changed.domains[1].participants[1].numa_node += 3;
    EXPECT_THROW(MoEExpertOwnerMap::buildShardTransition(changed, initial->placementOwnerMap(),
                                                         initial->originalOwnerMap()),
                 std::invalid_argument);
    // Equal geometry from another source wrapper is not the SAME original cohort owner.
    const auto foreign = MoEExpertOwnerMap::buildShardPlacement(
        original.plan, {initial->jointOwnerMap()->placementFor(0, 1)->shard_cohort->source()});
    EXPECT_THROW(MoEExpertOwnerMap::buildShardTransition(original.plan, foreign,
                                                         initial->originalOwnerMap()),
                 std::invalid_argument);
    EXPECT_EQ(authority.snapshot(), initial);
}

TEST(CPUJointPlacementCandidateContract, WholeOwnerAccessorsAndStaticAuthorityRemainUnchanged)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    MoEOverlayResidencyAuthority authority(
        {.initial_plan = oneTierTwoParticipantPlan(RoutedExpertResidencyPolicy::StaticById),
         .model_metadata = modelMetadata(),
         .maintenance_mode = MoERebalanceRuntimeMode::Off});
    const auto live = authority.snapshot();
    ASSERT_TRUE(live->valid());
    EXPECT_EQ(&live->placementOwnerMap(), &live->owner_map);
    EXPECT_EQ(&live->originalOwnerMap(), &live->owner_map);
    EXPECT_EQ(live->jointOwnerMap(), nullptr);
    EXPECT_EQ(live->originalProjectionPreparation(), nullptr);
    EXPECT_THROW(MoEOverlayJointPlacementCandidateTestAccess::prepare(
                     authority, live->placement_plan->placements),
                 std::logic_error);
    EXPECT_EQ(authority.snapshot(), live);
}

TEST(CPUJointPlacementCandidateContract,
     CohortExchangeRetainsUnchangedWholeOwnersAndActualSourceCuts)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    JointPlacementOriginal original(2, 2, 6);
    MoEOverlayResidencyAuthority authority(original.config());
    const auto initial = authority.snapshot();
    auto placements = original.plan.placements;
    std::swap(placements[0].routed_expert_tier[0], placements[0].routed_expert_tier[3]);
    const auto candidate =
        MoEOverlayJointPlacementCandidateTestAccess::prepare(authority, placements);
    ASSERT_TRUE(candidate->valid());
    for (const int unchanged : {1, 2})
        EXPECT_EQ(*candidate->placementOwnerMap().placementFor(0, unchanged),
                  *initial->placementOwnerMap().placementFor(0, unchanged));
    EXPECT_EQ(candidate->placementOwnerMap().placementFor(0, 0)->shard_cohort,
              initial->originalOwnerMap().placementFor(0, 3)->shard_cohort);
    EXPECT_EQ(candidate->placementOwnerMap().placementFor(0, 3)->owner_participant,
              initial->placementOwnerMap().placementFor(0, 0)->owner_participant);
    EXPECT_EQ(authority.snapshot(), initial);
}

/** The existing source-bound CPU cohort owner is retained by each layer range. */
TEST(CPUJointPlacementCandidateContract, CanonicalLayerRangeMatchesCohortFullScan)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    for (const int members : {2, 4, 8})
    {
        JointPlacementOriginal original(members, 2, 6, 3, 32, true);
        const auto map = original.preparation->cpuJointOwnerMap();
        ASSERT_TRUE(map);
        for (int replay = 0; replay < 3; ++replay)
            for (const int layer : {31, 32, 33, 34, 35})
            {
                std::vector<MoEExpertOwner> expected;
                for (const auto &owner : map->owners())
                    if (owner.layer_idx == layer) expected.push_back(owner);
                const auto actual = map->placementsForLayer(layer);
                EXPECT_EQ(std::vector<MoEExpertOwner>(actual.begin(), actual.end()), expected);
                for (const auto &owner : actual)
                {
                    EXPECT_EQ(&owner, map->placementFor(layer, owner.expert_id));
                    if (owner.shard_cohort)
                    {
                        EXPECT_EQ(map->fragmentCountForExpert(layer, owner.expert_id), std::size_t(members));
                        EXPECT_THROW((void)map->ownerFor(layer, owner.expert_id), std::logic_error);
                    }
                }
            }
    }
}
