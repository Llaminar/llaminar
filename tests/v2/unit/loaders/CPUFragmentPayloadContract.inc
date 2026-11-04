/**
 * @file CPUFragmentPayloadContract.inc
 * @brief Validate genuine prepared fragment geometry without manufacturing source authority.
 *
 * Existing native source/Manager/PMA/Store fixtures supply real local engines.
 * Metadata-only foreign cohorts exercise physical rejection; they do not mint a
 * candidate, initial static source, model owner or unavailable socket readiness.
 */
#include "execution/moe/MoEOverlayParticipantMigration.h"
#include <type_traits>

static_assert(!std::is_constructible_v<
              llaminar2::MoEOverlayPreparedCPUFragmentPayload, llaminar2::ModelContextId,
              std::shared_ptr<const llaminar2::MoEExpertOverlayPreparationPlan>,
              std::shared_ptr<const llaminar2::MoEExpertShardCohort>, std::size_t, int,
              std::array<std::shared_ptr<llaminar2::ITensorGemm>, 3>>);

static_assert(!std::is_convertible_v<llaminar2::MoEOverlayPreparedCPUFragmentPayload,
                                    llaminar2::MoEOverlayPreparedExpertPayload>);

TEST(CPUFragmentPayloadContract,
     ActualPreparedRolesRequireOriginalLocalGeometryAndNativeSourceFormat)
{
    using namespace llaminar2;
    if (!hasCPUBackend())
        initCPUBackend(-1);
    const auto selected = localJointSelection(0);
    const auto sources = jointSources(selected);
    ASSERT_GE(sources.size(), 3u);
    std::array<std::shared_ptr<ITensorGemm>, 3> engines;
    const auto store = sources.actor->manager->preparedWeightStore();
    for (const auto &source : sources)
    {
        if (source.identity.expert != selected->expertIds().front())
            continue;
        const auto role = source.identity.role == WeightRole::MoEExpertGate ? 0u
                          : source.identity.role == WeightRole::MoEExpertUp ? 1u
                                                                            : 2u;
        ASSERT_TRUE(source.prepared);
        engines[role] = store->gemmKernelLifetime(*source.prepared);
    }
    for (const auto &engine : engines)
        ASSERT_TRUE(engine);
    std::string error;
    for (int replay = 0; replay < 3; ++replay)
    {
        for (std::size_t role = 0; role < 3; ++role)
            EXPECT_TRUE(MoEOverlayPreparedCPUFragmentPayload::validateRole(
                *selected->cohort(), selected->participantIndex(),
                static_cast<ExpertTierWeightProjection>(role), engines[role], &error))
                << error;
        EXPECT_FALSE(MoEOverlayPreparedCPUFragmentPayload::validateRole(
            *selected->cohort(), selected->participantIndex(), ExpertTierWeightProjection::Gate,
            engines[2], &error));
        EXPECT_NE(error.find("geometry/source format"), std::string::npos);
        EXPECT_FALSE(MoEOverlayPreparedCPUFragmentPayload::validateRole(
            *selected->cohort(), 99, ExpertTierWeightProjection::Gate, engines[0], &error));
        EXPECT_FALSE(MoEOverlayPreparedCPUFragmentPayload::validateRole(
            *selected->cohort(), 0, ExpertTierWeightProjection::Gate, {}, &error));
        EXPECT_FALSE(MoEOverlayPreparedCPUFragmentPayload::validateRole(
            *selected->cohort(), 0, static_cast<ExpertTierWeightProjection>(77), engines[0],
            &error));
    }
    // This is deliberately only foreign physical metadata, not a source permit.
    auto foreignManifest = selected->cohort()->source();
    for (auto &projection : foreignManifest.projections)
        projection.format = ExpertWeightFormat::floating(TensorType::FP16);
    std::vector<MoEExpertShardCohort::Participant> participants(
        selected->cohort()->participants().begin(), selected->cohort()->participants().end());
    std::vector<MoEExpertProjectionOwnership::IntermediateInterval> intervals(
        selected->ownership().intermediateIntervals().begin(),
        selected->ownership().intermediateIntervals().end());
    const auto foreign =
        MoEExpertShardCohort::checked(foreignManifest, 512, participants, intervals);
    EXPECT_FALSE(MoEOverlayPreparedCPUFragmentPayload::validateRole(
        foreign, selected->participantIndex(), ExpertTierWeightProjection::Gate, engines[0],
        &error));
    EXPECT_NE(error.find("geometry/source format"), std::string::npos);
}

TEST(CPUFragmentPayloadContract,
     OrdinaryArrivalCannotMintFragmentAndEmptyFragmentCannotCertifyWholeExpert)
{
    using namespace llaminar2;
    MoEOverlayPreparedExpertArrival ordinary;
    MoEOverlayPreparedCPUFragmentPayload fragment;
    EXPECT_TRUE(fragment.empty());
    EXPECT_FALSE(fragment.valid());
    EXPECT_FALSE(ordinary.isCPUJointFragment());
    EXPECT_EQ(ordinary.destinationEpoch(), 0u);
    std::string error;
    EXPECT_FALSE(ordinary.completeCPUFragment(fragment, &error));
    EXPECT_TRUE(fragment.empty());
    EXPECT_NE(error.find("wrong-arm"), std::string::npos);
    EXPECT_TRUE(fragment.sameIdentity(MoEOverlayPreparedCPUFragmentPayload{}));
}
