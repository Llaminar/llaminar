/**
 * @file Test__MTPReversibleProposalGate.cpp
 * @brief Device-free verified-progress, actual-emission and trial ownership proofs.
 *
 * Every transition invokes the shared production gate. Synthetic immutable costs
 * and actual bounded observations test arithmetic and lifecycle, not model scope
 * or adaptive economy. Actual publication validation belongs to the commit owner.
 * A host-only128-bit oracle independently checks the portable fraction comparator.
 */
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <limits>
#include <random>
#include "execution/mtp/MTPReversibleProposalGate.h"

namespace
{
using namespace llaminar2;
using Origin=MTPDepthProposalOrigin;
using Phase=MTPReversibleProposalPhase;
using Mode=MTPReversibleProposalMode;
using Action=MTPReversibleProposalAction;
using Error=MTPReversibleProposalError;
using Order=MTPUtilityOrder;
using State=MTPReversibleProposalState;
using Input=MTPReversibleProposalInput;

/** @brief Synthetic immutable costs; never a measured model profile. */
struct Costs
{
    std::array<uint32_t,16> values{};
    /** @brief All supported positive depths have equal synthetic cost initially. */
    Costs(){values.fill(100);values[0]=0;}
    /** @param depth Requested depth. @return Exact positive cost, or missing0. */
    uint32_t costAt(uint32_t depth)const{return depth<values.size()?values[depth]:0;}
};
/** @return The fixture's ordinary Dynamic policy; owners provide its cadence. */
MTPReversibleProposalPolicy policy()
{return {.mode=Mode::Dynamic,.minimum_depth=1,.maximum_depth=15,.minimum_samples=2};}
/** @brief Assert every word, including actual Y independent of verified V.
 * @param a Actual state. @param b Independently expected state. */
void expectState(const State&a,const State&b)
{
    EXPECT_EQ(a.phase,b.phase);EXPECT_EQ(a.reference_depth,b.reference_depth);
    EXPECT_EQ(a.reference_runs,b.reference_runs);
    EXPECT_EQ(a.reference_verified_progress,b.reference_verified_progress);
    EXPECT_EQ(a.window_emitted,b.window_emitted);
}
/** @brief Author one actual emission and cumulative validated publication view.
 * @param d Executed depth. @param r Eligible cumulative runs. @param y Actual emitted delta.
 * @param v Cumulative validated published states. @param origin Actual scored proposal origin.
 * @param target Actual proposed depth. @param eligible Actual commit-owner eligibility.
 * @return Distinct Y and typed V; no additional live V accumulator. */
Input input(uint32_t d,uint32_t r,uint32_t y,uint32_t v,
    Origin origin=Origin::Unevaluated,uint32_t target=0,bool eligible=true)
{
    return {.executed_depth=d,.window_runs=r,.newly_emitted_tokens=y,.eligible=eligible,
        .proposal_origin=origin,.parent_target=target,.verified_progress={.states=v}};
}
/** @brief Feed two authored equal-yield samples and one scored proposal.
 * @param s Actual five-word owner. @param d Executed depth. @param progress Each actual Y/V.
 * @param target Scored target. @param costs Immutable costs. @param origin Scored origin.
 * @return Second production transition; feedback after a changed selector is separate. */
MTPReversibleProposalResult window(State&s,uint32_t d,uint32_t progress,uint32_t target,
    const Costs&costs=Costs{},Origin origin=Origin::SupportedScore)
{
    EXPECT_TRUE(applyMTPReversibleProposalGate(policy(),s,input(d,1,progress,progress),costs).valid());
    return applyMTPReversibleProposalGate(policy(),s,
        input(d,2,progress,2*progress,target==d?Origin::Hold:origin,target),costs);
}
}

TEST(MTPReversibleProposalGate,ExactUtilityMatchesIndependentWideOracleAtCounterExtremes)
{
    const std::array<uint32_t,8> words{0,1,2,3,65535,65536,UINT32_MAX-1,UINT32_MAX};
    for(auto v1:words)for(auto v2:words)for(auto r1:words)for(auto r2:words)
    {
        if(!r1||!r2)continue;
        const uint32_t c1=UINT32_MAX,c2=UINT32_MAX-1;
        const unsigned __int128 lhs=static_cast<unsigned __int128>(v1)*r2*c2;
        const unsigned __int128 rhs=static_cast<unsigned __int128>(v2)*r1*c1;
        EXPECT_EQ(compareMTPVerifiedProgressUtility(v1,r1,c1,v2,r2,c2),
            lhs<rhs?Order::Less:lhs>rhs?Order::Greater:Order::Equal);
    }
    std::mt19937_64 rng(0x42e175);
    for(int i=0;i<20000;++i)
    {
        const auto v1=static_cast<uint32_t>(rng()),v2=static_cast<uint32_t>(rng());
        const auto r1=static_cast<uint32_t>(rng())|1u,r2=static_cast<uint32_t>(rng())|1u;
        const auto c1=static_cast<uint32_t>(rng())|1u,c2=static_cast<uint32_t>(rng())|1u;
        const unsigned __int128 lhs=static_cast<unsigned __int128>(v1)*r2*c2;
        const unsigned __int128 rhs=static_cast<unsigned __int128>(v2)*r1*c1;
        ASSERT_EQ(compareMTPVerifiedProgressUtility(v1,r1,c1,v2,r2,c2),
            lhs<rhs?Order::Less:lhs>rhs?Order::Greater:Order::Equal)<<i;
    }
    EXPECT_EQ(compareMTPVerifiedProgressUtility(6,2,100,9,3,100),Order::Equal);
    EXPECT_EQ(compareMTPUtilityFractions(UINT64_MAX-1,UINT64_MAX,UINT64_MAX-2,UINT64_MAX-1),Order::Greater);
}

/** @test Starting a trial saves validated V, while append retains different actual Y. */
TEST(MTPReversibleProposalGate,VerifiedProgressReferenceIsDistinctFromActualEmission)
{
    State s;Costs c;
    ASSERT_TRUE(applyMTPReversibleProposalGate(policy(),s,input(5,1,4,3),c).valid());
    EXPECT_EQ(s.window_emitted,4u);
    const auto start=applyMTPReversibleProposalGate(policy(),s,input(5,2,4,7,Origin::UnresolvedScore,7),c);
    ASSERT_TRUE(start.valid());EXPECT_EQ(start.action,Action::ProbeStarted);
    EXPECT_EQ(start.next_depth,7u);EXPECT_TRUE(start.consume_window);EXPECT_TRUE(start.reset_cooldown);
    expectState(s,{Phase::ProbePending,5,2,7,0}); // Reference V7, never actual window Y8.
}

/** @test Equal emitted rates cannot hide a strict verified-progress improvement. */
TEST(MTPReversibleProposalGate,VerifiedProgressKeepUsesPublicationInsteadOfEmissionTie)
{
    State s{Phase::ProbePending,5,4,12,0};Costs c;
    ASSERT_TRUE(applyMTPReversibleProposalGate(policy(),s,input(3,1,3,3),c).valid());
    const auto keep=applyMTPReversibleProposalGate(policy(),s,input(3,2,3,7,Origin::UnresolvedScore,15),c);
    ASSERT_TRUE(keep.valid());EXPECT_EQ(keep.action,Action::ProbeKept);EXPECT_EQ(keep.next_depth,3u);
    EXPECT_TRUE(keep.consume_window);EXPECT_FALSE(keep.reset_cooldown);expectState(s,{});
    // Actual trial Y6/2 ties the reference V12/4, but actual trial V7/2 improves it.
}

/** @test Extra emitted frontier credit cannot turn equal verified rates into a keep. */
TEST(MTPReversibleProposalGate,VerifiedProgressTieReturnsDespiteHigherEmittedReward)
{
    State s{Phase::ProbePending,4,2,7,0};Costs c;
    ASSERT_TRUE(applyMTPReversibleProposalGate(policy(),s,input(5,1,4,3),c).valid());
    const auto done=applyMTPReversibleProposalGate(policy(),s,input(5,2,4,7),c);
    ASSERT_TRUE(done.valid());EXPECT_EQ(done.action,Action::ProbeReturned);EXPECT_EQ(done.next_depth,4u);
    EXPECT_TRUE(done.reset_cooldown);expectState(s,{Phase::RejectedComparison,5,2,7,0});
    // Actual trial Y8 does not replace the independently validated V7 reference value.
}

/** @test Every scored direction is an actual trial; a completion cannot apply a second target. */
TEST(MTPReversibleProposalGate,VerifiedProgressBothDirectionsAndOriginsAreRealTrials)
{
    for(auto origin:{Origin::SupportedScore,Origin::UnresolvedScore})
    for(uint32_t target:{3u,7u})
    {
        State s;const auto start=window(s,5,3,target,Costs{},origin);
        ASSERT_TRUE(start.valid());EXPECT_EQ(start.action,Action::ProbeStarted);EXPECT_EQ(start.next_depth,target);
        expectState(s,{Phase::ProbePending,5,2,6,0});
        const auto keep=window(s,target,4,15);
        ASSERT_TRUE(keep.valid());EXPECT_EQ(keep.action,Action::ProbeKept);EXPECT_EQ(keep.next_depth,target);
        expectState(s,{});
    }
}

TEST(MTPReversibleProposalGate,TieOrWorseReturnsAndCachesFailedActualTargetSample)
{
    for(uint32_t progress:{2u,3u})
    {
        State s;ASSERT_EQ(window(s,5,3,3).action,Action::ProbeStarted);
        const auto done=window(s,3,progress,1);
        ASSERT_TRUE(done.valid());EXPECT_EQ(done.action,Action::ProbeReturned);EXPECT_EQ(done.next_depth,5u);
        expectState(s,{Phase::RejectedComparison,3,2,2*progress,0});
    }
}

TEST(MTPReversibleProposalGate,UnequalSampleSizesAndCostsUseRatesRatherThanRawTokenTotals)
{
    Costs c;c.values[5]=200;State s{Phase::ProbePending,5,4,12,0};
    ASSERT_TRUE(applyMTPReversibleProposalGate(policy(),s,input(3,1,2,2),c).valid());
    const auto keep=applyMTPReversibleProposalGate(policy(),s,input(3,2,2,4),c);
    EXPECT_EQ(keep.action,Action::ProbeKept);expectState(s,{}); // V4/(2*100)>V12/(4*200).
}

TEST(MTPReversibleProposalGate,PendingSuppressesParentUntilActualMinimumAndConsumesOnlyOneAction)
{
    auto p=policy();p.minimum_samples=4;State s{Phase::ProbePending,5,4,12,0};Costs c;
    for(uint32_t r=1;r<4;++r)
    {
        const auto out=applyMTPReversibleProposalGate(p,s,input(3,r,3,3*r,Origin::UnresolvedScore,15),c);
        ASSERT_TRUE(out.valid());EXPECT_EQ(out.action,Action::Accumulating);EXPECT_FALSE(out.consume_window);
        EXPECT_EQ(s.window_emitted,3*r);EXPECT_EQ(out.next_depth,3u);
    }
    const auto out=applyMTPReversibleProposalGate(p,s,input(3,4,3,12,static_cast<Origin>(99),99),c);
    ASSERT_TRUE(out.valid());EXPECT_EQ(out.action,Action::ProbeReturned);EXPECT_EQ(out.next_depth,5u);
    expectState(s,{Phase::RejectedComparison,3,4,12,0});
}

/** @test Defense veto uses V for either direction/origin and preserves exact cached evidence. */
TEST(MTPReversibleProposalGate,VerifiedProgressCacheVetoAndReopenIgnoreEmissionPhase)
{
    Costs c;
    for(auto origin:{Origin::SupportedScore,Origin::UnresolvedScore})
    for(uint32_t target:{3u,7u})
    {
        State s{Phase::RejectedComparison,target,2,7,7};
        const auto reopen=decideMTPReversibleProposalGate(policy(),s,input(5,2,0,6,origin,target),c);
        ASSERT_TRUE(reopen.valid());EXPECT_EQ(reopen.action,Action::ProbeStarted);
        expectState(s,{Phase::ProbePending,5,2,6,0}); // V6 reopens although actual Y7 ties old V7.
        for(uint32_t progress:{7u,8u})
        {
            s={Phase::RejectedComparison,target,2,7,progress==7?6u:8u};
            const auto block=decideMTPReversibleProposalGate(policy(),s,input(5,2,0,progress,origin,target),c);
            ASSERT_TRUE(block.valid());EXPECT_EQ(block.action,Action::RejectedTargetBlocked);
            EXPECT_EQ(block.next_depth,5u);EXPECT_TRUE(block.consume_window);EXPECT_FALSE(block.reset_cooldown);
            expectState(s,{Phase::RejectedComparison,target,2,7,0});
        }
    }
}

TEST(MTPReversibleProposalGate,DifferentTargetReopensAndReplacesOneReferenceWithoutHistory)
{
    State s{Phase::RejectedComparison,3,2,6,0};
    const auto out=window(s,5,4,4);EXPECT_EQ(out.action,Action::ProbeStarted);
    expectState(s,{Phase::ProbePending,5,2,8,0});
}

/** @test Holds consume actual Y with the canonical window and preserve failed-target V. */
TEST(MTPReversibleProposalGate,VerifiedProgressHoldOwnsEmissionAndPreservesRejectedReference)
{
    State s{Phase::RejectedComparison,3,2,6,7};Costs c;
    const auto hold=decideMTPReversibleProposalGate(policy(),s,input(5,2,0,6,Origin::Hold,5),c);
    ASSERT_TRUE(hold.valid());EXPECT_EQ(hold.action,Action::ParentHold);EXPECT_EQ(hold.next_depth,5u);
    EXPECT_TRUE(hold.consume_window);EXPECT_FALSE(hold.reset_cooldown);
    expectState(s,{Phase::RejectedComparison,3,2,6,0});
}

/** @test Observe is descriptive in both directions and cannot reuse a Dynamic reference. */
TEST(MTPReversibleProposalGate,VerifiedProgressObserveWouldProbeEitherDirectionWithoutMutation)
{
    auto p=policy();p.mode=Mode::Observe;Costs c;
    for(auto origin:{Origin::SupportedScore,Origin::UnresolvedScore})for(uint32_t target:{3u,7u})
    {
        State s;s.window_emitted=7;
        const auto out=decideMTPReversibleProposalGate(p,s,input(5,2,0,6,origin,target),c);
        ASSERT_TRUE(out.valid());EXPECT_EQ(out.action,Action::WouldProbe);EXPECT_EQ(out.next_depth,5u);
        EXPECT_TRUE(out.consume_window);EXPECT_FALSE(out.reset_cooldown);expectState(s,{});
    }
    State s{Phase::ProbePending,5,2,6,0};const auto old=s;
    EXPECT_EQ(applyMTPReversibleProposalGate(p,s,input(3,1,3,3),c).error,Error::InvalidState);expectState(s,old);
}

TEST(MTPReversibleProposalGate,InactiveClippedAndPartialSamplesNeverManufactureComparison)
{
    auto p=policy();p.mode=Mode::Inactive;p.minimum_depth=0;p.maximum_depth=0;p.minimum_samples=0;
    State s{static_cast<Phase>(99),99,99,99,99};const auto old=s;Costs missing;missing.values.fill(0);
    EXPECT_EQ(applyMTPReversibleProposalGate(p,s,input(5,0,0,0,Origin::SupportedScore,3,false),missing).action,Action::Inactive);
    expectState(s,old);p=policy();s={Phase::ProbePending,5,2,6,0};const auto pending=s;
    EXPECT_EQ(applyMTPReversibleProposalGate(p,s,input(3,0,0,0,Origin::Unevaluated,0,false),missing).action,Action::Ineligible);
    expectState(s,pending);
    EXPECT_EQ(applyMTPReversibleProposalGate(p,s,input(3,1,1,1),Costs{}).action,Action::Accumulating);
    EXPECT_EQ(s.window_emitted,1u);EXPECT_EQ(s.phase,Phase::ProbePending);
}

TEST(MTPReversibleProposalGate,SeparateAppendAndDecisionPreserveParentCadenceWithoutDoubleReward)
{
    State s;Costs c;
    auto first=input(5,1,2,0); // Append ignores the borrowed V; commit validation has another owner.
    ASSERT_TRUE(appendMTPReversibleProposalEmission(policy(),s,first).valid());EXPECT_EQ(s.window_emitted,2u);
    first.verified_progress.states=2;
    ASSERT_TRUE(decideMTPReversibleProposalGate(policy(),s,first,c).valid());EXPECT_EQ(s.window_emitted,2u);
    const auto second=input(5,2,4,5,Origin::SupportedScore,3);
    ASSERT_TRUE(appendMTPReversibleProposalEmission(policy(),s,second).valid());EXPECT_EQ(s.window_emitted,6u);
    const auto start=decideMTPReversibleProposalGate(policy(),s,second,c);
    EXPECT_EQ(start.action,Action::ProbeStarted);expectState(s,{Phase::ProbePending,5,2,5,0});
    // The old executed depth is now the Pending reference, not the active
    // selector. State validation rejects that stale join before reading Y.
    const auto before=s;
    EXPECT_EQ(decideMTPReversibleProposalGate(policy(),s,second,c).error,Error::InvalidState);expectState(s,before);
    // Even with the actual new selector, the consumed Y window cannot support
    // another decision without new committed observations.
    const auto consumed=input(3,2,0,5);
    EXPECT_EQ(decideMTPReversibleProposalGate(policy(),s,consumed,c).error,Error::InvalidObservation);expectState(s,before);
}

TEST(MTPReversibleProposalGate,InclusiveDepthBoundsAndActualEmissionExtentsRemainSupported)
{
    State s;EXPECT_EQ(window(s,1,1,15).action,Action::ProbeStarted);
    expectState(s,{Phase::ProbePending,1,2,2,0});
    EXPECT_EQ(window(s,15,16,1).action,Action::ProbeKept);expectState(s,{});
    EXPECT_EQ(window(s,15,16,1).action,Action::ProbeStarted);
    EXPECT_EQ(window(s,1,2,15).action,Action::ProbeReturned);
    expectState(s,{Phase::RejectedComparison,1,2,4,0});
}

TEST(MTPReversibleProposalGate,ResetClearsWordsAndPausedContinuationPreservesPartialAuthority)
{
    State s{Phase::RejectedComparison,3,2,6,9};s={};expectState(s,{});
    ASSERT_EQ(window(s,5,3,3).action,Action::ProbeStarted);
    ASSERT_TRUE(applyMTPReversibleProposalGate(policy(),s,input(3,1,4,4),Costs{}).valid());const auto before=s;
    EXPECT_EQ(applyMTPReversibleProposalGate(policy(),s,input(3,1,0,0,Origin::Unevaluated,0,false),Costs{}).action,Action::Ineligible);
    expectState(s,before);
    EXPECT_EQ(applyMTPReversibleProposalGate(policy(),s,input(3,2,4,8),Costs{}).action,Action::ProbeKept);expectState(s,{});
}

/** @test Typed V presence/bounds and scoped proposal validity fail before mutation. */
TEST(MTPReversibleProposalGate,VerifiedProgressInvalidViewOriginAndTargetRejectAtomically)
{
    Costs c;
    for(uint32_t v:{0u,1u,13u})
    {
        State s;s.window_emitted=6;const auto old=s;
        EXPECT_EQ(decideMTPReversibleProposalGate(policy(),s,input(5,2,0,v,Origin::SupportedScore,3),c).error,Error::InvalidObservation);
        expectState(s,old);
    }
    for(auto in:{input(5,2,0,6,Origin::SupportedScore,5),input(5,2,0,6,Origin::UnresolvedScore,5),
        input(5,2,0,6,Origin::Hold,3),input(5,2,0,6,static_cast<Origin>(99),3),input(5,2,0,6,Origin::SupportedScore,16)})
    {
        State s;s.window_emitted=6;const auto old=s;
        EXPECT_EQ(decideMTPReversibleProposalGate(policy(),s,in,c).error,Error::InvalidObservation);expectState(s,old);
    }
}

TEST(MTPReversibleProposalGate,InvalidBoundsStateCostAndOverflowRejectWithoutMutation)
{
    Costs c;State s;auto p=policy();p.minimum_depth=0;
    EXPECT_EQ(applyMTPReversibleProposalGate(p,s,input(5,1,1,1),c).error,Error::InvalidPolicy);expectState(s,{});
    p=policy();p.mode=static_cast<Mode>(99);
    EXPECT_EQ(applyMTPReversibleProposalGate(p,s,input(5,1,1,1),c).error,Error::InvalidPolicy);p=policy();
    for(auto in:{input(0,1,1,1),input(5,0,1,1),input(5,1,0,1),input(5,1,7,1),input(5,1,1,1,Origin::SupportedScore,3)})
    {
        EXPECT_EQ(applyMTPReversibleProposalGate(p,s,in,c).error,Error::InvalidObservation);expectState(s,{});
    }
    s={Phase::ProbePending,3,0,0,0};const auto bad=s;
    EXPECT_EQ(applyMTPReversibleProposalGate(p,s,input(5,1,1,1),c).error,Error::InvalidState);expectState(s,bad);
    s={Phase::ProbePending,5,2,6,UINT32_MAX};const auto huge=s;
    EXPECT_EQ(applyMTPReversibleProposalGate(p,s,input(3,UINT32_MAX,1,UINT32_MAX),c).error,Error::CounterOverflow);expectState(s,huge);
    s={};s.window_emitted=3;const auto old=s;c.values[3]=0;
    EXPECT_EQ(applyMTPReversibleProposalGate(p,s,input(5,2,3,6,Origin::SupportedScore,3),c).error,Error::InvalidCost);expectState(s,old);
}
