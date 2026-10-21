/**
 * @file Test__AdaptiveImmutableRowCacheOps.cpp
 * @brief Adversarial device-free traces of the production adaptive GPU metadata algorithm.
 *
 * These tests execute the shared integer operations on actual declared record
 * arrays. They prove bounded ownership/order/generation state, not source I/O,
 * GPU kernel arithmetic, stream publication or serving integration. No backend
 * startup, native payload, model, clock sampling or independent cache is used.
 */
#include "AdaptiveImmutableRowCacheTestFixture.h"
#include <array>
#include <limits>
namespace llaminar2::test {
namespace {
namespace a=adaptive::ops;
using adaptive::ActualMetadata;
using Fixture=ActualMetadata<a::Slot32>;
/** @return A stable external snapshot of the three actual linked-list heads/tails. */
auto lists(const a::OwnerControl &o){return std::array{o.head[0],o.head[1],o.head[2],o.tail[0],o.tail[1],o.tail[2]};}
/** @brief Prove a genuinely narrow/wide directory using boundary source keys, not truncated aliases. */
template<class Slot,std::size_t N> void boundaryRows(const std::array<std::uint64_t,N> &rows,std::uint64_t sourceRows){
    ActualMetadata<Slot> f(N,sourceRows);auto actual=f.operations();
    const auto decisions=f.complete(rows);ASSERT_EQ(decisions.size(),N);
    for(std::size_t i=0;i<N;++i){ASSERT_NE(decisions[i].slot,a::None);EXPECT_EQ(actual.find(rows[i]),decisions[i].slot);}
    EXPECT_EQ(actual.find(17),a::None);f.expectInvariants();
}
}
TEST(AdaptiveImmutableRowCacheOps, ZeroSlotsCompleteRepeatedAndEmptyFramesWithoutOptionalOwner){
    Fixture f(0);auto actual=f.operations();
    for(const auto &ids: {std::vector<std::uint64_t>{},std::vector<std::uint64_t>{1,1,2,1}}){
        const auto result=f.complete(ids);
        for(const auto d:result)EXPECT_EQ(d.slot,a::None);
        EXPECT_EQ(f.frame.original_ids,ids.size());EXPECT_EQ(f.frame.observed_ids,ids.size());
        EXPECT_EQ(actual.find(1),a::None);EXPECT_EQ(f.frame.phase,a::Phase::Idle);f.expectInvariants();
    }
}
TEST(AdaptiveImmutableRowCacheOps, Source32And64BoundaryKeysRetainDistinctDirectoryEntries){
    boundaryRows<a::Slot32>(std::array<std::uint64_t,4>{0,1,0x80000000ULL,0xffffffffULL},0x100000000ULL);
    boundaryRows<a::Slot64>(std::array<std::uint64_t,4>{0,0x100000000ULL,0x8000000000000000ULL,0xfffffffffffffffeULL},0xffffffffffffffffULL);
}
TEST(AdaptiveImmutableRowCacheOps, OneSlotHotAdmissionAndCancellationRetainBoundedDirectory){
    Fixture f(1);auto actual=f.operations();const std::array<std::uint64_t,1> cold{0},hot{9};
    f.complete(cold);EXPECT_NE(actual.find(0),a::None);
    f.complete(hot); // The first equal-frequency candidate is not required to enter.
    f.complete(hot);ASSERT_NE(actual.find(9),a::None);EXPECT_EQ(actual.find(0),a::None);
    EXPECT_EQ(f.owner.directory_root,a::LeafTag);EXPECT_EQ(f.owner.node_free,a::None);
    EXPECT_EQ(f.owner.filled,1u);f.expectInvariants();
}
TEST(AdaptiveImmutableRowCacheOps, EmptyRequestPreservesWarmedRowsAndModelLifetimePolicy){
    Fixture f(4);const std::array<std::uint64_t,4> ids{0,1,0,2};f.complete(ids);
    const auto clock=f.owner.logical_clock,generation=f.owner.next_slot_generation;const auto order=lists(f.owner);
    const auto counts=f.counts;const auto epochs=f.epochs;
    f.complete(std::span<const std::uint64_t>{});
    EXPECT_EQ(f.owner.logical_clock,clock);EXPECT_EQ(f.owner.next_slot_generation,generation);
    EXPECT_EQ(lists(f.owner),order);EXPECT_EQ(f.counts,counts);EXPECT_EQ(f.epochs,epochs);
    EXPECT_EQ(f.frame.original_ids,0u);EXPECT_EQ(f.frame.held_pins,0u);f.expectInvariants();
}
TEST(AdaptiveImmutableRowCacheOps, DelayedReadyDoesNotReorderOriginalTouchesOrPublishReservedRows){
    Fixture f(4);auto actual=f.operations();
    ASSERT_TRUE(actual.begin(4));ASSERT_TRUE(actual.beginPolicy(3));
    auto first=actual.observe(10);const auto second=actual.observe(20);
    first=actual.observe(10,first);const auto third=actual.observe(30);
    ASSERT_NE(first.slot,a::None);ASSERT_NE(second.slot,a::None);ASSERT_NE(third.slot,a::None);
    EXPECT_EQ(actual.find(10),a::None);EXPECT_EQ(actual.find(20),a::None);
    const auto order=lists(f.owner);const auto clock=f.owner.logical_clock;
    std::vector<std::uint64_t> touches;for(const auto &s:f.slots)touches.push_back(s.touched);
    ASSERT_TRUE(actual.prepare());f.validateResponse(3);
    ASSERT_TRUE(actual.makeReady(third));ASSERT_TRUE(actual.makeReady(second));ASSERT_TRUE(actual.makeReady(first));
    EXPECT_EQ(lists(f.owner),order);EXPECT_EQ(f.owner.logical_clock,clock);
    for(std::size_t i=0;i<f.slots.size();++i)EXPECT_EQ(f.slots[i].touched,touches[i]);
    EXPECT_EQ(actual.find(10),first.slot);EXPECT_EQ(actual.find(20),second.slot);EXPECT_EQ(actual.find(30),third.slot);
    for(const auto d:{first,second,third})ASSERT_TRUE(actual.release(d));
    ASSERT_TRUE(actual.finish());f.expectInvariants();
}
TEST(AdaptiveImmutableRowCacheOps, EveryInitialHitIsPinnedBeforeMissVictimsAndAllPinnedBypassProgresses){
    Fixture f(4);const std::array<std::uint64_t,4> initial{0,1,2,3};f.complete(initial);
    std::array<std::uint64_t,4> generations{};for(std::size_t i=0;i<4;++i)generations[i]=f.slots[i].generation;
    const std::array<std::uint64_t,6> ids{99,0,1,2,3,99};const auto decisions=f.complete(ids);
    ASSERT_EQ(decisions.size(),5u);EXPECT_EQ(decisions.front().slot,a::None);
    EXPECT_EQ(f.operations().find(99),a::None);EXPECT_EQ(f.owner.next_slot_generation,4u);
    for(std::size_t i=0;i<4;++i){EXPECT_EQ(f.slots[i].row,i);EXPECT_EQ(f.slots[i].generation,generations[i]);}
    EXPECT_EQ(f.frame.held_pins,0u);EXPECT_EQ(f.frame.observed_ids,6u);f.expectInvariants();
}
TEST(AdaptiveImmutableRowCacheOps, DemotedAndNewProbationUseLogicalOrderWithoutPinnedRescans){
    Fixture f(4,1024,1);const std::array<std::uint64_t,4> fill{0,1,2,3};f.complete(fill);
    const std::array<std::uint64_t,3> touched{0,1,2};f.complete(touched);
    auto actual=f.operations();
    for(const auto row:{0u,1u,2u,3u}){ASSERT_NE(actual.find(row),a::None);}
    EXPECT_EQ(f.slots[actual.find(3)].touched,4u);
    EXPECT_EQ(a::Operations<a::Slot32>::stream(f.slots[actual.find(0)]),a::Stream::DemotedProbation);
    EXPECT_EQ(a::Operations<a::Slot32>::stream(f.slots[actual.find(1)]),a::Stream::DemotedProbation);
    EXPECT_EQ(a::Operations<a::Slot32>::stream(f.slots[actual.find(2)]),a::Stream::Protected);
    const std::array<std::uint64_t,6> pinnedThenMiss{0,1,2,100,100,100};f.complete(pinnedThenMiss);
    ASSERT_NE(actual.find(100),a::None);EXPECT_EQ(actual.find(3),a::None);
    for(const auto row:{0u,1u,2u})EXPECT_NE(actual.find(row),a::None);
    f.expectInvariants();
}
TEST(AdaptiveImmutableRowCacheOps, DirectoryNodeReuseHasNoSourceSizedOrTombstoneGrowth){
    Fixture f(8,1000000,3);auto actual=f.operations();
    const std::array<std::uint64_t,8> initial{0,1,2,3,4,5,6,7};f.complete(initial);
    for(std::uint64_t round=0;round<64;++round){
        // Seed an exact adversarial sketch baseline. Choose a source key in
        // an unused first-lane bucket so every actual resident has one lane
        // still at1 after candidate observation. Thus admission must happen,
        // without duplicating the victim or directory algorithm in the test.
        std::fill(f.counts.begin(),f.counts.end(),1);
        std::uint64_t candidate=100+round*1000;bool distinct=false;
        for(;candidate<100+(round+1)*1000;++candidate){
            const auto bucket=a::Operations<a::Slot32>::hash(candidate)&15;
            distinct=std::none_of(f.slots.begin(),f.slots.end(),[&](const auto &slot){
                return (a::Operations<a::Slot32>::hash(slot.row)&15)==bucket;
            });
            if(distinct)break;
        }
        ASSERT_TRUE(distinct);const std::array<std::uint64_t,1> ids{candidate};
        const auto result=f.complete(ids);ASSERT_EQ(result.size(),1u);ASSERT_NE(result.front().slot,a::None);
        EXPECT_EQ(actual.find(candidate),result.front().slot);f.expectInvariants();
        EXPECT_EQ(f.nodes.size(),7u);EXPECT_EQ(f.slots.size(),8u);
        EXPECT_EQ(f.epochs.size(),64u);EXPECT_EQ(f.counts.size(),64u);
    }
    EXPECT_EQ(f.owner.next_slot_generation,72u);EXPECT_EQ(f.owner.filled,8u);
    EXPECT_EQ(f.frame.held_pins,0u);EXPECT_EQ(f.frame.status,a::Status::Success);
    EXPECT_EQ(actual.find(999999),a::None);
}
TEST(AdaptiveImmutableRowCacheOps, CancellationRecyclesOnlyReservedSlotAndNeverReusesGeneration){
    Fixture f(1);auto actual=f.operations();ASSERT_TRUE(actual.begin(1));ASSERT_TRUE(actual.beginPolicy(1));
    const auto cancelled=actual.observe(7);ASSERT_NE(cancelled.slot,a::None);ASSERT_TRUE(actual.prepare());
    EXPECT_FALSE(actual.fail(a::Status::Cancelled));ASSERT_TRUE(actual.release(cancelled));
    EXPECT_FALSE(actual.finish());EXPECT_EQ(f.frame.phase,a::Phase::Idle);EXPECT_EQ(f.frame.status,a::Status::Cancelled);
    EXPECT_EQ(actual.find(7),a::None);EXPECT_EQ(f.owner.slot_free,cancelled.slot);
    ASSERT_TRUE(actual.begin(1));ASSERT_TRUE(actual.beginPolicy(1));const auto next=actual.observe(8);
    EXPECT_EQ(next.slot,cancelled.slot);EXPECT_GT(next.generation,cancelled.generation);
    ASSERT_TRUE(actual.prepare());f.validateResponse(1);
    ASSERT_TRUE(actual.makeReady(next));ASSERT_TRUE(actual.release(next));ASSERT_TRUE(actual.finish());
    EXPECT_EQ(actual.find(7),a::None);EXPECT_EQ(actual.find(8),next.slot);f.expectInvariants();
}
TEST(AdaptiveImmutableRowCacheOps, StaleGenerationCannotMakeReadyOrReleaseCurrentReservation){
    for(const bool publish:{false,true}){
        Fixture f(1);auto actual=f.operations();ASSERT_TRUE(actual.begin(1));ASSERT_TRUE(actual.beginPolicy(1));
        const auto d=actual.observe(7);ASSERT_TRUE(actual.prepare());f.validateResponse(1);
        auto stale=d;++stale.generation;const auto flags=f.slots[d.slot].flags;
        EXPECT_FALSE(publish?actual.makeReady(stale):actual.release(stale));
        EXPECT_EQ(f.frame.status,a::Status::Invariant);EXPECT_EQ(f.frame.held_pins,1u);
        EXPECT_EQ(f.slots[d.slot].flags,flags);EXPECT_EQ(f.slots[d.slot].generation,d.generation);
        EXPECT_EQ(actual.find(7),a::None);ASSERT_TRUE(actual.release(d));
        EXPECT_FALSE(actual.finish()); // No successful completion is fabricated after failure.
    }
}
TEST(AdaptiveImmutableRowCacheOps, FinishCannotRetireBackingWithOutstandingReaderPin){
    Fixture f(1);auto actual=f.operations();ASSERT_TRUE(actual.begin(1));ASSERT_TRUE(actual.beginPolicy(1));
    const auto d=actual.observe(3);ASSERT_TRUE(actual.prepare());f.validateResponse(1);
    ASSERT_TRUE(actual.makeReady(d));EXPECT_FALSE(actual.finish());EXPECT_EQ(f.frame.phase,a::Phase::Validated);
    EXPECT_EQ(f.frame.held_pins,1u);EXPECT_EQ(actual.find(3),d.slot);ASSERT_TRUE(actual.release(d));
    EXPECT_FALSE(actual.release(d));EXPECT_EQ(f.frame.held_pins,0u);EXPECT_EQ(actual.find(3),d.slot);
}
TEST(AdaptiveImmutableRowCacheOps, InvalidCoordinateAndIncompleteFrameRejectBeforePolicyMutation){
    Fixture f(2,16);auto actual=f.operations();ASSERT_TRUE(actual.begin(2));ASSERT_TRUE(actual.beginPolicy(1));
    const auto listsBefore=lists(f.owner);const auto clock=f.owner.logical_clock,generation=f.owner.next_slot_generation;
    EXPECT_EQ(actual.observe(16).slot,a::None);EXPECT_EQ(f.frame.observed_ids,0u);
    EXPECT_EQ(f.owner.logical_clock,clock);EXPECT_EQ(f.owner.next_slot_generation,generation);EXPECT_EQ(lists(f.owner),listsBefore);
    EXPECT_FALSE(actual.prepare());EXPECT_EQ(f.frame.phase,a::Phase::Building);
}
TEST(AdaptiveImmutableRowCacheOps, PinFrontierAndPriorRowMismatchAreExplicitFailures){
    for(const bool latePin:{false,true}){
        Fixture f(2);const std::array<std::uint64_t,1> fill{3};f.complete(fill);auto actual=f.operations();
        const auto slot=actual.find(3);ASSERT_NE(slot,a::None);const auto generation=f.slots[slot].generation;
        ASSERT_TRUE(actual.begin(1));ASSERT_TRUE(actual.pinInitial(slot,generation));ASSERT_TRUE(actual.beginPolicy(1));
        const auto oldRow=f.slots[slot].row,oldFlags=f.slots[slot].flags;
        if(latePin){EXPECT_FALSE(actual.pinInitial(slot,generation));EXPECT_EQ(f.frame.status,a::Status::InvalidPhase);}
        else {EXPECT_EQ(actual.observe(4,{slot,generation}).slot,a::None);EXPECT_EQ(f.frame.status,a::Status::Invariant);}
        EXPECT_EQ(f.slots[slot].row,oldRow);EXPECT_EQ(f.slots[slot].flags,oldFlags);EXPECT_EQ(f.frame.held_pins,1u);
    }
}
TEST(AdaptiveImmutableRowCacheOps, FullGenerationClockAndEpochBudgetsRejectBeforeMutation){
    for(const unsigned failure:{0u,1u,2u}){
        Fixture f(2,1024,1,1);auto actual=f.operations();
        if(failure==0)f.owner.next_slot_generation=std::numeric_limits<std::uint64_t>::max();
        if(failure==1)f.owner.logical_clock=std::numeric_limits<std::uint64_t>::max();
        if(failure==2)f.owner.logical_clock=std::numeric_limits<std::uint32_t>::max();
        const auto clock=f.owner.logical_clock,generation=f.owner.next_slot_generation;
        EXPECT_FALSE(actual.begin(1));EXPECT_EQ(f.frame.status,a::Status::Exhausted);EXPECT_EQ(f.frame.phase,a::Phase::Idle);
        EXPECT_EQ(f.owner.logical_clock,clock);EXPECT_EQ(f.owner.next_slot_generation,generation);EXPECT_EQ(f.owner.filled,0u);
    }
}
TEST(AdaptiveImmutableRowCacheOps, FourLaneAgingIsLazySaturatingAndNeverWraps){
    Fixture f(1,1024,0,4);auto actual=f.operations();
    std::fill(f.counts.begin(),f.counts.end(),std::numeric_limits<std::uint16_t>::max());
    const std::array<std::uint64_t,1> row{7};f.complete(row);
    for(std::uint64_t lane=0;lane<4;++lane){
        const auto at=lane*16+(a::Operations<a::Slot32>::hash(7+lane*0x9e3779b97f4a7c15ULL)&15);
        EXPECT_EQ(f.counts[at],std::numeric_limits<std::uint16_t>::max());
    }
    f.owner.logical_clock=64;f.complete(row);
    for(std::uint64_t lane=0;lane<4;++lane){
        const auto at=lane*16+(a::Operations<a::Slot32>::hash(7+lane*0x9e3779b97f4a7c15ULL)&15);
        EXPECT_EQ(f.counts[at],1u);EXPECT_EQ(f.epochs[at],16u);
    }
    EXPECT_EQ(actual.find(7),0u);f.expectInvariants();
}
TEST(AdaptiveImmutableRowCacheOps, NonemptySourceCannotSkipPublicationOrResponseValidation){
    for(const bool skipResponse:{false,true}){
        Fixture f(1);auto actual=f.operations();ASSERT_TRUE(actual.begin(1));ASSERT_TRUE(actual.beginPolicy(1));
        const auto d=actual.observe(7);ASSERT_TRUE(actual.prepare());f.frame.source_rows=1;
        if(skipResponse){
            ASSERT_TRUE(actual.transition(a::Phase::Prepared,a::Phase::Published));
            EXPECT_FALSE(actual.transition(a::Phase::Published,a::Phase::Validated));
            EXPECT_EQ(f.frame.phase,a::Phase::Published);
        }else {
            EXPECT_FALSE(actual.transition(a::Phase::Prepared,a::Phase::Validated));
            EXPECT_EQ(f.frame.phase,a::Phase::Prepared);
        }
        EXPECT_EQ(f.frame.status,a::Status::InvalidPhase);EXPECT_EQ(actual.find(7),a::None);
        EXPECT_FALSE(actual.makeReady(d));EXPECT_EQ(actual.find(7),a::None);
        ASSERT_TRUE(actual.release(d));EXPECT_FALSE(actual.finish());EXPECT_EQ(f.frame.phase,a::Phase::Idle);
    }
}
TEST(AdaptiveImmutableRowCacheOps, FailedFrameOrPoisonedOwnerCannotPublishReadyAfterLegalPhaseCleanup){
    for(const bool ownerPoison:{false,true}){
        Fixture f(1);auto actual=f.operations();ASSERT_TRUE(actual.begin(1));ASSERT_TRUE(actual.beginPolicy(1));
        const auto d=actual.observe(7);ASSERT_TRUE(actual.prepare());f.frame.source_rows=1;
        if(ownerPoison){actual.poison();EXPECT_EQ(f.frame.status,a::Status::Success);}
        else {EXPECT_FALSE(actual.fail(a::Status::SourceFailure));EXPECT_EQ(f.owner.flags,a::OwnerInitialized);}
        // Legal frontier cleanup must not authorize failed native bytes. The
        // actual publish operation must independently reject BOTH health arms.
        ASSERT_TRUE(actual.transition(a::Phase::Prepared,a::Phase::Published));
        ASSERT_TRUE(actual.transition(a::Phase::Published,a::Phase::ResponseCopied));
        ASSERT_TRUE(actual.transition(a::Phase::ResponseCopied,a::Phase::Validated));
        const auto flags=f.slots[d.slot].flags;
        EXPECT_FALSE(actual.makeReady(d));EXPECT_EQ(f.slots[d.slot].flags,flags);
        EXPECT_EQ(actual.find(7),a::None);EXPECT_EQ(f.frame.held_pins,1u);
        EXPECT_EQ(f.frame.status,ownerPoison?a::Status::InvalidPhase:a::Status::SourceFailure);
        ASSERT_TRUE(actual.release(d));EXPECT_FALSE(actual.finish());EXPECT_EQ(f.frame.phase,a::Phase::Idle);
    }
}
TEST(AdaptiveImmutableRowCacheOps, SourceFailurePoisonsOwnerWithoutPublishingReservedRows){
    Fixture f(2);auto actual=f.operations();ASSERT_TRUE(actual.begin(2));ASSERT_TRUE(actual.beginPolicy(2));
    const auto one=actual.observe(1),two=actual.observe(2);ASSERT_TRUE(actual.prepare());
    EXPECT_FALSE(actual.fail(a::Status::SourceFailure));actual.poison();
    ASSERT_TRUE(actual.release(one));ASSERT_TRUE(actual.release(two));EXPECT_FALSE(actual.finish());
    EXPECT_EQ(actual.find(1),a::None);EXPECT_EQ(actual.find(2),a::None);
    EXPECT_FALSE(actual.begin(0));EXPECT_NE(f.owner.flags&a::OwnerFailed,0u);EXPECT_EQ(f.owner.next_slot_generation,2u);
}
/** @brief Reject an unrepresentable original-coordinate budget even when optional cache capacity is zero. */
TEST(AdaptiveImmutableRowCacheOps, OversizedOriginalCoordinateBudgetRejectsBeforeMutationIncludingC0)
{
    for (const auto capacity : {0u, 2u})
    {
        Fixture f(capacity);
        auto actual = f.operations();
        const auto clock = f.owner.logical_clock;
        const auto generation = f.owner.next_slot_generation;
        const auto original = f.frame.original_ids;
        EXPECT_FALSE(actual.begin(static_cast<std::uint64_t>(a::MaximumSlots) + 1));
        EXPECT_EQ(f.frame.status, a::Status::InvalidGeometry);
        EXPECT_EQ(f.frame.phase, a::Phase::Idle);
        EXPECT_EQ(f.frame.original_ids, original);
        EXPECT_EQ(f.frame.observed_ids, 0u);
        EXPECT_EQ(f.frame.held_pins, 0u);
        EXPECT_EQ(f.owner.logical_clock, clock);
        EXPECT_EQ(f.owner.next_slot_generation, generation);
        EXPECT_EQ(f.owner.filled, 0u);
    }
}
/**
 * @brief Cancel before publication or after a simulated validated private ACK without losing initial Ready rows.
 *
 * This is actual shared metadata code, not a source-authentication or native ACK
 * proof. The native fixture separately owns real source/controller publication.
 */
TEST(AdaptiveImmutableRowCacheOps, CancellationAtSourceFrontiersPreservesReadyRowsAndRetiresOnlyReservedPins)
{
    for (const bool afterPrivateAck : {false, true})
    {
        Fixture f(2);
        const std::array<std::uint64_t, 1> initial{3};
        f.complete(initial);
        auto actual = f.operations();
        const auto warm = actual.find(3);
        ASSERT_NE(warm, a::None);
        const auto warmGeneration = f.slots[warm].generation;
        ASSERT_TRUE(actual.begin(2));
        ASSERT_TRUE(actual.pinInitial(warm, warmGeneration));
        ASSERT_TRUE(actual.beginPolicy(2));
        const auto hit = actual.observe(3, {warm, warmGeneration});
        const auto reserved = actual.observe(7);
        ASSERT_NE(reserved.slot, a::None);
        ASSERT_TRUE(actual.prepare());
        if (afterPrivateAck)
        {
            f.validateResponse(1); // Explicit pure-metadata validation stand-in; no CPU/GPU source claim.
        }
        EXPECT_FALSE(actual.fail(a::Status::Cancelled));
        EXPECT_FALSE(actual.makeReady(reserved));
        ASSERT_TRUE(actual.release(hit));
        ASSERT_TRUE(actual.release(reserved));
        EXPECT_FALSE(actual.finish());
        EXPECT_EQ(f.frame.phase, a::Phase::Idle);
        EXPECT_EQ(f.frame.held_pins, 0u);
        EXPECT_EQ(f.owner.flags, a::OwnerInitialized);
        EXPECT_EQ(actual.find(7), a::None);
        EXPECT_EQ(actual.find(3), warm);
        EXPECT_EQ(f.slots[warm].generation, warmGeneration);
        const auto nextGeneration = f.owner.next_slot_generation;
        f.complete(initial);
        EXPECT_EQ(f.owner.next_slot_generation, nextGeneration);
        EXPECT_EQ(actual.find(3), warm);
        f.expectInvariants();
    }
}
} // namespace llaminar2::test
