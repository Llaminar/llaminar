/**
 * @file Test__MTPSurvivalDepthSelector.cpp
 * @brief Device-free ranking/filter proofs for bounded survival proposals.
 *
 * Authored arithmetic cases exercise real selector code, independent wide-score
 * ordering and the existing immutable cost profile. They do not certify model
 * observations or throughput. Typed confirmation tests additionally prove bounded
 * target history and cache-preserving real gate consumption. No device
 * state or policy candidate is installed by these fixtures.
 */
#include "execution/mtp/MTPSurvivalDepthSelector.h"
#include "execution/mtp/MTPDepthCostProfile.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace
{
    using namespace llaminar2;
    using Kind = MTPSurvivalProposalKind;
    using Error = MTPSurvivalSelectorError;
    using CacheKind = MTPRejectedVerifiedComparisonKind;

    /** @brief Arithmetic-only borrowed profile; production owns no local array. */
    struct Costs
    {
        std::array<uint32_t,16> values{};
        /** @brief Fill every supported depth with the same positive fixture cost. */
        explicit Costs(uint32_t value = 100) { values.fill(value); values[0] = 0; }
        /** @param d Requested depth. @return Immutable fixture scalar or zero. */
        uint32_t costAt(uint32_t d) const { return d < values.size() ? values[d] : 0; }
    };

    /** @return Explicit supported commit scope; no rounded rate is consumed. */
    MTPPrefixWindowEvidence evidence(uint32_t d, uint32_t r, uint64_t a,
        uint32_t z, uint32_t f)
    {
        return {MTPPrefixEvidenceScope::NonStoppedUnclippedVerifiedState,d,r,a,z,f};
    }

    /** @brief Assert a fail-closed result cannot be interpreted as a valid hold. */
    void expectError(const MTPSurvivalDepthProposal &p, Error error)
    {
        EXPECT_FALSE(p.valid());
        EXPECT_EQ(p.error,error);
        EXPECT_EQ(p.target_depth,0u);
        EXPECT_EQ(p.progress.incumbent,0u);
        EXPECT_EQ(p.progress.lower,0u);
        EXPECT_EQ(p.progress.upper,0u);
    }

    /** @brief Independent wide product ordering of two supported lower scores. */
    bool oracleBetter(const MTPSurvivalDepthProposal &a, uint32_t ac,
        const MTPSurvivalDepthProposal &b, uint32_t bc, uint32_t current)
    {
        if (a.kind != b.kind) return a.kind == Kind::SupportedProbe;
        if (a.kind == Kind::SupportedProbe)
        {
            const unsigned __int128 left = static_cast<unsigned __int128>(a.progress.lower)*bc;
            const unsigned __int128 right = static_cast<unsigned __int128>(b.progress.lower)*ac;
            if (left != right) return left > right;
        }
        const auto ad = a.target_depth > current ? a.target_depth-current : current-a.target_depth;
        const auto bd = b.target_depth > current ? b.target_depth-current : current-b.target_depth;
        if (ad != bd) return ad < bd;
        if (ac != bc) return ac < bc;
        return a.target_depth < b.target_depth;
    }
}

TEST(MTPSurvivalDepthSelector, ExistingCostsExposeUnresolvedLowerBeyondDominatedAdjacent)
{
    const MTPDepthCostProfile costs{MTPDepthCostProfileKind::Greedy};
    const auto e = evidence(5,10,26,0,3);
    EXPECT_EQ(evaluateMTPPrefixSurvivalScore(e,4,costs.costAt(5),costs.costAt(4)).classification,
        MTPSurvivalScoreClass::Dominated);
    for (auto k : {2u,3u})
        EXPECT_EQ(evaluateMTPPrefixSurvivalScore(e,k,costs.costAt(5),costs.costAt(k)).classification,
            MTPSurvivalScoreClass::Unresolved);
    const auto selected = selectMTPSurvivalDepthProposal(e,1,15,costs);
    ASSERT_TRUE(selected.valid());
    EXPECT_EQ(selected.kind,Kind::UnresolvedProbe);
    EXPECT_EQ(selected.target_depth,3u);
    EXPECT_EQ(selected.progress.incumbent,36u);
    EXPECT_EQ(selected.progress.lower,29u);
    EXPECT_EQ(selected.progress.upper,30u);
}

TEST(MTPSurvivalDepthSelector, SupportedRanksByLowerScoreBeforeDistanceAndAllTiesAreDeterministic)
{
    Costs costs(1000);
    const auto full = evidence(5,10,50,0,10);
    costs.values[1] = 100; costs.values[4] = 700;
    auto selected = selectMTPSurvivalDepthProposal(full,1,5,costs);
    ASSERT_TRUE(selected.valid());
    EXPECT_EQ(selected.kind,Kind::SupportedProbe);
    EXPECT_EQ(selected.target_depth,1u); // Greater lower score beats nearer4.

    for (uint32_t k = 1; k < 5; ++k) costs.values[k] = 100*(k+1);
    selected = selectMTPSurvivalDepthProposal(full,1,5,costs);
    EXPECT_EQ(selected.target_depth,4u); // Equal score:nearest depth.

    costs.values[6] = 600;
    selected = selectMTPSurvivalDepthProposal(full,1,6,costs);
    EXPECT_EQ(selected.target_depth,4u); // Equal score and distance:lower cost500.

    // No accepted prefixes:all progress bounds equal. Nonmonotone fixture costs
    // expose the final same-score/same-distance/same-cost smaller-depth tie.
    costs = Costs(100);
    costs.values[5] = 1000;
    selected = selectMTPSurvivalDepthProposal(evidence(5,10,0,10,0),1,15,costs);
    ASSERT_TRUE(selected.valid());
    EXPECT_EQ(selected.kind,Kind::SupportedProbe);
    EXPECT_EQ(selected.target_depth,4u);
}

TEST(MTPSurvivalDepthSelector, UnresolvedRanksByDistanceThenCostRatherThanOptimisticMagnitude)
{
    Costs costs(1000000);
    costs.values[5] = 1000;
    costs.values[3] = 820; costs.values[7] = 1100;
    const auto e = evidence(5,10,26,0,3);
    const auto selected = selectMTPSurvivalDepthProposal(e,1,15,costs);
    ASSERT_TRUE(selected.valid());
    EXPECT_EQ(selected.kind,Kind::UnresolvedProbe);
    EXPECT_EQ(selected.target_depth,3u); // Same distance as7; lower borrowed cost.
    costs.values[6] = 1050;
    const auto nearer = selectMTPSurvivalDepthProposal(e,1,15,costs);
    EXPECT_EQ(nearer.kind,Kind::UnresolvedProbe);
    EXPECT_EQ(nearer.target_depth,6u); // Distance wins regardless of upper-score rank.
}

TEST(MTPSurvivalDepthSelector, RejectedNearestFiltersOnlyItsTargetAndEqualityBlocks)
{
    const MTPDepthCostProfile costs{MTPDepthCostProfileKind::Greedy};
    const auto e = evidence(5,10,50,0,10);
    auto selected = selectMTPSurvivalDepthProposal(e,1,15,costs);
    ASSERT_TRUE(selected.valid());
    EXPECT_EQ(selected.kind,Kind::UnresolvedProbe);
    EXPECT_EQ(selected.target_depth,6u);
    const MTPRejectedVerifiedComparison rejected{CacheKind::RejectedTarget,6,10,60};
    selected = selectMTPSurvivalDepthProposal(e,1,15,costs,rejected);
    EXPECT_EQ(selected.target_depth,7u); // Blocked6 does not monopolize larger actions.
    EXPECT_EQ(selected.kind,Kind::UnresolvedProbe);
    selected = selectMTPSurvivalDepthProposal(e,1,15,costs,
        {CacheKind::RejectedTarget,7,10,60});
    EXPECT_EQ(selected.target_depth,6u); // No interval/direction-wide veto.
    selected = selectMTPSurvivalDepthProposal(e,1,15,costs,
        {CacheKind::RejectedTarget,6,10,70});
    EXPECT_EQ(selected.target_depth,6u); // Failed target beat current, so retry allowed.

    const Costs equalCosts;
    selected = selectMTPSurvivalDepthProposal(e,5,15,equalCosts,
        {CacheKind::RejectedTarget,6,20,120});
    EXPECT_EQ(selected.target_depth,7u); // Equal rates with DIFFERENT R block6.
    selected = selectMTPSurvivalDepthProposal(e,5,6,equalCosts,
        {CacheKind::RejectedTarget,6,20,120});
    EXPECT_EQ(selected.kind,Kind::Hold);
    EXPECT_EQ(selected.target_depth,5u);
}

TEST(MTPSurvivalDepthSelector, MalformedEvidenceBoundsCostsAndComparisonFailBeforeAnyProposal)
{
    Costs costs;
    const auto e = evidence(5,10,50,0,10);
    for (const auto bounds : {std::array<uint32_t,2>{0,15},{1,16},{6,5},{6,15},{1,4}})
        expectError(selectMTPSurvivalDepthProposal(e,bounds[0],bounds[1],costs),Error::InvalidBounds);
    expectError(selectMTPSurvivalDepthProposal({},1,15,costs),Error::InvalidBounds);
    auto unsupported = e; unsupported.scope = MTPPrefixEvidenceScope::Unsupported;
    expectError(selectMTPSurvivalDepthProposal(unsupported,1,15,costs),Error::InvalidEvidence);
    expectError(selectMTPSurvivalDepthProposal(evidence(5,0,0,0,0),1,15,costs),Error::InvalidEvidence);
    expectError(selectMTPSurvivalDepthProposal(evidence(5,10,51,0,10),1,15,costs),Error::InvalidEvidence);
    for (const auto rejected : {
        MTPRejectedVerifiedComparison{CacheKind::None,6,0,0},
        MTPRejectedVerifiedComparison{static_cast<CacheKind>(99),6,10,60},
        MTPRejectedVerifiedComparison{CacheKind::RejectedTarget,0,10,60},
        MTPRejectedVerifiedComparison{CacheKind::RejectedTarget,16,10,60},
        MTPRejectedVerifiedComparison{CacheKind::RejectedTarget,5,10,60},
        MTPRejectedVerifiedComparison{CacheKind::RejectedTarget,6,0,0},
        MTPRejectedVerifiedComparison{CacheKind::RejectedTarget,6,10,9},
        MTPRejectedVerifiedComparison{CacheKind::RejectedTarget,6,10,71},
        MTPRejectedVerifiedComparison{CacheKind::RejectedTarget,6,UINT32_MAX,UINT64_MAX}})
        expectError(selectMTPSurvivalDepthProposal(e,1,15,costs,rejected),Error::InvalidRejectedComparison);
    costs.values[5] = 0;
    expectError(selectMTPSurvivalDepthProposal(e,1,15,costs),Error::InvalidCost);
    expectError(selectMTPSurvivalDepthProposal(e,1,15,MTPDepthCostProfile{}),Error::InvalidCost);
    costs = Costs(1000); costs.values[1] = 1; costs.values[15] = 0;
    // A valid early supported action must not hide a later missing cost.
    expectError(selectMTPSurvivalDepthProposal(e,1,15,costs),Error::InvalidCost);
    costs = Costs{}; costs.values[6] = 0;
    expectError(selectMTPSurvivalDepthProposal(e,1,15,costs,
        {CacheKind::RejectedTarget,6,10,60}),Error::InvalidCost);
}

TEST(MTPSurvivalDepthSelector, EveryAdmittedBoundsPairMatchesIndependentWideRanking)
{
    for (const auto law : {MTPDepthCostProfileKind::Greedy,MTPDepthCostProfileKind::Stochastic})
    {
        const MTPDepthCostProfile costs{law};
        for (uint32_t d = 1; d <= 15; ++d)
            for (uint32_t minimum = 1; minimum <= d; ++minimum)
                for (uint32_t maximum = d; maximum <= 15; ++maximum)
                {
                    const auto e = evidence(d,4,2*d,2,2);
                    std::vector<MTPSurvivalDepthProposal> candidates;
                    for (uint32_t k = minimum; k <= maximum; ++k)
                    {
                        if (k == d) continue;
                        const auto score = evaluateMTPPrefixSurvivalScore(e,k,costs.costAt(d),costs.costAt(k));
                        ASSERT_TRUE(score.valid());
                        if (score.classification == MTPSurvivalScoreClass::Supported ||
                            score.classification == MTPSurvivalScoreClass::Unresolved)
                            candidates.push_back({Error::None,
                                score.classification == MTPSurvivalScoreClass::Supported ? Kind::SupportedProbe : Kind::UnresolvedProbe,
                                k,score.progress});
                    }
                    std::sort(candidates.begin(),candidates.end(),[&](const auto &a,const auto &b)
                    {
                        return oracleBetter(a,costs.costAt(a.target_depth),b,costs.costAt(b.target_depth),d);
                    });
                    const auto selected = selectMTPSurvivalDepthProposal(e,minimum,maximum,costs);
                    ASSERT_TRUE(selected.valid());
                    EXPECT_GE(selected.target_depth,minimum);
                    EXPECT_LE(selected.target_depth,maximum);
                    if (candidates.empty())
                    {
                        EXPECT_EQ(selected.kind,Kind::Hold);
                        EXPECT_EQ(selected.target_depth,d);
                    }
                    else
                    {
                        EXPECT_EQ(selected.kind,candidates.front().kind);
                        EXPECT_EQ(selected.target_depth,candidates.front().target_depth);
                        EXPECT_EQ(selected.progress.lower,candidates.front().progress.lower);
                        EXPECT_EQ(selected.progress.upper,candidates.front().progress.upper);
                    }
                }
    }
}

TEST(MTPSurvivalDepthSelector, UintExtremesAndRepeatedCacheViewsRemainPure)
{
    const Costs costs(UINT32_MAX);
    const auto e = evidence(5,UINT32_MAX,uint64_t{5}*UINT32_MAX,0,UINT32_MAX);
    const MTPRejectedVerifiedComparison rejected{CacheKind::RejectedTarget,6,
        UINT32_MAX,uint64_t{6}*UINT32_MAX};
    for (int repeat = 0; repeat < 100; ++repeat)
    {
        const auto blocked = selectMTPSurvivalDepthProposal(e,1,15,costs,rejected);
        const auto fresh = selectMTPSurvivalDepthProposal(e,1,15,costs);
        ASSERT_TRUE(blocked.valid());
        ASSERT_TRUE(fresh.valid());
        EXPECT_EQ(blocked.kind,Kind::UnresolvedProbe);
        EXPECT_EQ(blocked.target_depth,7u);
        EXPECT_EQ(blocked.progress.incumbent,uint64_t{6}*UINT32_MAX);
        EXPECT_EQ(fresh.kind,Kind::UnresolvedProbe);
        EXPECT_EQ(fresh.target_depth,6u);
    }
    EXPECT_EQ(rejected.target_depth,6u);
    EXPECT_EQ(rejected.reference_runs,UINT32_MAX);
    EXPECT_EQ(rejected.reference_verified_progress,uint64_t{6}*UINT32_MAX);
    EXPECT_EQ(e.accepted,uint64_t{5}*UINT32_MAX);
}

/** @test Only completed disjoint windows advance the sole target history. */
TEST(MTPSurvivalDepthSelector, ConfirmationCountsCompletedWindowsAndRestartsChangedTarget)
{
    using A=MTPProposalConfirmationAction;
    MTPProposalConfirmation state{5,0};
    auto first=confirmMTPSurvivalProposal(state,5,1,15,3,4);
    ASSERT_TRUE(first.valid());EXPECT_EQ(first.action,A::AwaitingConfirmation);
    EXPECT_EQ(first.next.target_depth,4u);EXPECT_EQ(first.next.completed_windows,1u);
    EXPECT_EQ(state.target_depth,5u);EXPECT_EQ(state.completed_windows,0u);
    auto second=confirmMTPSurvivalProposal(first.next,5,1,15,3,4);
    ASSERT_TRUE(second.valid());EXPECT_EQ(second.next.completed_windows,2u);
    auto third=confirmMTPSurvivalProposal(second.next,5,1,15,3,4);
    ASSERT_TRUE(third.valid());EXPECT_EQ(third.action,A::TrialReady);
    EXPECT_EQ(third.next.target_depth,5u);EXPECT_EQ(third.next.completed_windows,0u);
    const auto changed=confirmMTPSurvivalProposal(second.next,5,1,15,3,6);
    ASSERT_TRUE(changed.valid());EXPECT_EQ(changed.next.target_depth,6u);EXPECT_EQ(changed.next.completed_windows,1u);
    const auto hold=confirmMTPSurvivalProposal(changed.next,5,1,15,3,5);
    ASSERT_TRUE(hold.valid());EXPECT_EQ(hold.action,A::Hold);EXPECT_EQ(hold.next.target_depth,5u);
    EXPECT_EQ(hold.next.completed_windows,0u);
    const auto immediate=confirmMTPSurvivalProposal(state,5,1,15,1,6);
    ASSERT_TRUE(immediate.valid());EXPECT_EQ(immediate.action,A::TrialReady);
}

/** @test Invalid history and pending poison fail without repairing their input. */
TEST(MTPSurvivalDepthSelector, ConfirmationRejectsPoisonPendingAndOverflowBeforeMutation)
{
    for(const auto state:{MTPProposalConfirmation{0,0},{16,1},{4,0},{5,1},{4,3},{4,UINT32_MAX}})
    {
        EXPECT_FALSE(validMTPProposalConfirmation(state,5,1,15,3,false));
        EXPECT_EQ(confirmMTPSurvivalProposal(state,5,1,15,3,4).error,MTPProposalConfirmationError::InvalidState);
    }
    EXPECT_FALSE(validMTPProposalConfirmation({4,1},5,1,15,3,true));
    EXPECT_TRUE(validMTPProposalConfirmation({5,0},5,1,15,3,true));
    EXPECT_EQ(confirmMTPSurvivalProposal({5,0},5,1,15,0,4).error,MTPProposalConfirmationError::InvalidPolicy);
    EXPECT_EQ(confirmMTPSurvivalProposal({5,0},5,1,15,3,0).error,MTPProposalConfirmationError::InvalidTarget);
    EXPECT_EQ(confirmMTPSurvivalProposal({5,0},5,1,15,3,16).error,MTPProposalConfirmationError::InvalidTarget);
    const auto last=confirmMTPSurvivalProposal({2,UINT32_MAX-1},1,1,2,UINT32_MAX,2);
    ASSERT_TRUE(last.valid());EXPECT_EQ(last.action,MTPProposalConfirmationAction::TrialReady);
    EXPECT_EQ(last.next.completed_windows,0u);
}

/** @test Deferral consumes actual Y through Hold without erasing the rejected target. */
TEST(MTPSurvivalDepthSelector, ConfirmationDeferralPreservesGateCacheUntilActualTrial)
{
    const MTPDepthCostProfile costs{MTPDepthCostProfileKind::Greedy};
    const MTPReversibleProposalPolicy policy{MTPReversibleProposalMode::Dynamic,1,4,4};
    MTPReversibleProposalState gate{MTPReversibleProposalPhase::RejectedComparison,3,4,4,12};
    MTPProposalConfirmation history{4,0};
    for(int window=1;window<=3;++window)
    {
        gate.window_emitted=12; // Actual four-frame authored publication, independently owned Y.
        const auto confirmed=confirmMTPSurvivalProposal(history,4,1,4,3,1);
        ASSERT_TRUE(confirmed.valid());
        const bool ready=confirmed.action==MTPProposalConfirmationAction::TrialReady;
        const auto decision=decideMTPReversibleProposalGate(policy,gate,
            {.executed_depth=4,.window_runs=4,.eligible=true,
             .proposal_origin=ready?MTPDepthProposalOrigin::SupportedScore:MTPDepthProposalOrigin::Hold,
             .parent_target=ready?1u:4u,.verified_progress={12}},costs);
        ASSERT_TRUE(decision.valid());EXPECT_TRUE(decision.consume_window);EXPECT_EQ(gate.window_emitted,0u);
        if(!ready)
        {
            EXPECT_EQ(decision.next_depth,4u);EXPECT_FALSE(decision.reset_cooldown);
            EXPECT_EQ(gate.phase,MTPReversibleProposalPhase::RejectedComparison);
            EXPECT_EQ(gate.reference_depth,3u);EXPECT_EQ(gate.reference_runs,4u);
            EXPECT_EQ(gate.reference_verified_progress,4u);history=confirmed.next;
        }
        else
        {
            EXPECT_EQ(window,3);EXPECT_EQ(decision.action,MTPReversibleProposalAction::ProbeStarted);
            EXPECT_EQ(decision.next_depth,1u);EXPECT_EQ(gate.phase,MTPReversibleProposalPhase::ProbePending);
            EXPECT_EQ(gate.reference_depth,4u);EXPECT_EQ(gate.reference_verified_progress,12u);
        }
    }
}
