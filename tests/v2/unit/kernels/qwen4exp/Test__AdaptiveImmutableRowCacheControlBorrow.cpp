/**
 * @file Test__AdaptiveImmutableRowCacheControlBorrow.cpp
 * @brief Actual-Operations equivalence and failure publication for lexical control borrows.
 *
 * Small device-free arrays exercise production integer operations, not a cache
 * policy mirror. The ordinary oracle publishes controls directly; the candidate
 * borrows only controls and keeps the same actual slot/radix/sketch operations.
 * Source-response transitions here qualify integer lifecycle only. Native
 * protocol/capture/control publication remains in explicit backend preflight.
 */
#include "AdaptiveImmutableRowCacheTestFixture.h"
#include "kernels/qwen4exp/AdaptiveImmutableRowCacheControlBorrow.h"
#include <array>
#include <cstring>
#include <tuple>
namespace llaminar2::test::adaptive
{
namespace
{
/** @brief Compare complete initialized control state, including historical cursors. */
template<class Slot> void expectSame(const ActualMetadata<Slot> &a, const ActualMetadata<Slot> &b)
{
    EXPECT_EQ(std::memcmp(&a.owner, &b.owner, sizeof(a.owner)), 0);
    EXPECT_EQ(std::memcmp(&a.frame, &b.frame, sizeof(a.frame)), 0);
    ASSERT_EQ(a.slots.size(), b.slots.size());
    for (std::size_t i=0; i<a.slots.size(); ++i)
    {
        const auto &x=a.slots[i], &y=b.slots[i];
        EXPECT_EQ(std::tie(x.generation,x.touched,x.row,x.previous,x.next,x.flags),
                  std::tie(y.generation,y.touched,y.row,y.previous,y.next,y.flags));
    }
    ASSERT_EQ(a.nodes.size(), b.nodes.size());
    for (std::size_t i=0; i<a.nodes.size(); ++i)
    {
        EXPECT_EQ(a.nodes[i].child[0], b.nodes[i].child[0]);
        EXPECT_EQ(a.nodes[i].child[1], b.nodes[i].child[1]);
        EXPECT_EQ(a.nodes[i].split_bit, b.nodes[i].split_bit);
    }
    EXPECT_EQ(a.epochs,b.epochs);
    EXPECT_EQ(a.counts,b.counts);
}
/** @brief Execute the same production operations with lexical control publication. */
template<class Slot> void completeBorrowed(ActualMetadata<Slot> &m, std::span<const std::uint64_t> ids)
{
    using M=ActualMetadata<Slot>;
    auto actual=m.operations(); M::require(actual.begin(ids.size()));
    std::vector<std::uint64_t> unique;
    std::vector<ops::Decision> decisions;
    std::uint64_t sourceRows=0;
    for (auto row:ids)
    {
        if (std::find(unique.begin(),unique.end(),row)!=unique.end()) continue;
        unique.push_back(row);
        const auto slot=actual.find(row);
        decisions.push_back(slot==ops::None ? ops::Decision{} : ops::Decision{slot,m.slots[slot].generation});
        sourceRows+=slot==ops::None;
    }
    {
        ops::FrameControl frame; ops::OwnerControl owner;
        ops::OrderedControlBorrow<Slot> borrow(m.view,frame,owner);
        ops::Operations<Slot> writer(borrow.metadata());
        for (const auto d:decisions) if (d.slot!=ops::None) M::require(writer.pinInitial(d.slot,d.generation));
        M::require(writer.beginPolicy(unique.size()));
        for (auto row:ids)
        {
            const auto index=static_cast<std::size_t>(std::find(unique.begin(),unique.end(),row)-unique.begin());
            decisions[index]=writer.observe(row,decisions[index]);
            M::require(frame.status==ops::Status::Success);
        }
        M::require(writer.prepare());
    }
    m.validateResponse(sourceRows);
    for (auto d:decisions) if (d.slot!=ops::None && ops::Operations<Slot>::state(m.slots[d.slot])==ops::SlotState::Reserved)
        M::require(actual.makeReady(d));
    {
        ops::FrameControl frame; ops::OwnerControl owner;
        ops::OrderedControlBorrow<Slot> borrow(m.view,frame,owner);
        ops::Operations<Slot> writer(borrow.metadata());
        for (auto d:decisions) M::require(writer.release(d));
        M::require(writer.finish());
    }
}
/** @brief Exercise actual misses, hits, duplicates, aging and subsequent victims. */
template<class Slot> void orderedEquivalence(std::uint64_t sourceRows)
{
    for (std::uint32_t capacity: {0u,1u,3u,8u})
        for (std::uint32_t limit: {0u,1u,2u})
        {
            ActualMetadata<Slot> direct(capacity,sourceRows,limit,3), borrowed(capacity,sourceRows,limit,3);
            const auto base=sourceRows>0x100000000ULL ? 0x100000001ULL : 0ULL;
            for (const auto &trace: std::vector<std::vector<std::uint64_t>>{
                {base,base+1,base+2}, {base+1,base,base+1,base+2,base},
                {base+3,base+1,base+4,base+4}, {}, {base+2,base+3,base,base+1},
                {base+5,base+6,base+1,base+7}})
            {
                direct.complete(trace); completeBorrowed(borrowed,trace); expectSame(direct,borrowed);
            }
        }
}
TEST(AdaptiveControlBorrow, ActualOrderedOperationsPreserveCompleteRow32State)
{ orderedEquivalence<ops::Slot32>(1024); }
TEST(AdaptiveControlBorrow, ActualOrderedOperationsPreserveCompleteRow64State)
{ orderedEquivalence<ops::Slot64>(0x100000100ULL); }
TEST(AdaptiveControlBorrow, ControlsStayLexicalUntilExplicitPublicationAndScopeExit)
{
    ActualMetadata<ops::Slot32> m(3); auto actual=m.operations(); ASSERT_TRUE(actual.begin(1));
    {
        ops::FrameControl frame; ops::OwnerControl owner;
        ops::OrderedControlBorrow<ops::Slot32> borrow(m.view,frame,owner);
        ops::Operations<ops::Slot32> writer(borrow.metadata());
        ASSERT_TRUE(writer.beginPolicy(1));
        const auto d=writer.observe(1);
        ASSERT_NE(d.slot,ops::None);
        EXPECT_EQ(m.frame.observed_ids,0u); EXPECT_EQ(m.owner.logical_clock,0u);
        borrow.publish();
        EXPECT_EQ(m.frame.observed_ids,1u); EXPECT_EQ(m.owner.logical_clock,1u);
        ASSERT_TRUE(writer.prepare());
    }
    EXPECT_EQ(m.frame.phase,ops::Phase::Prepared); EXPECT_EQ(m.frame.held_pins,1u);
}
TEST(AdaptiveControlBorrow, EarlyReturnPublishesExactPartialOperationBeforeFailureCleanup)
{
    ActualMetadata<ops::Slot32> direct(3), candidate(3);
    ASSERT_TRUE(direct.operations().begin(1)); ASSERT_TRUE(candidate.operations().begin(1));
    auto ordinary=direct.operations(); ASSERT_TRUE(ordinary.beginPolicy(1));
    const auto original=ordinary.observe(9); ASSERT_NE(original.slot,ops::None);
    EXPECT_FALSE(ordinary.fail(ops::Status::Cancelled));
    [&] {
        ops::FrameControl frame; ops::OwnerControl owner;
        ops::OrderedControlBorrow<ops::Slot32> borrow(candidate.view,frame,owner);
        ops::Operations<ops::Slot32> writer(borrow.metadata());
        ActualMetadata<ops::Slot32>::require(writer.beginPolicy(1));
        const auto chosen=writer.observe(9);
        ActualMetadata<ops::Slot32>::require(chosen.slot!=ops::None);
        (void)writer.fail(ops::Status::Cancelled);
        return; // The actual destructor publishes this early failure exit.
    }();
    expectSame(direct,candidate);
    ASSERT_TRUE(ordinary.release(original));
    ASSERT_TRUE(candidate.operations().release({original.slot,original.generation}));
    EXPECT_FALSE(ordinary.finish()); EXPECT_FALSE(candidate.operations().finish());
    expectSame(direct,candidate);
}
TEST(AdaptiveControlBorrow, ExistingFirstFailureSurvivesLatePublicationAndRefresh)
{
    ActualMetadata<ops::Slot32> m(3); ASSERT_TRUE(m.operations().begin(1));
    {
        ops::FrameControl frame; ops::OwnerControl owner;
        ops::OrderedControlBorrow<ops::Slot32> borrow(m.view,frame,owner);
        EXPECT_FALSE(m.operations().fail(ops::Status::SourceFailure));
        borrow.publish();
        EXPECT_EQ(frame.status,ops::Status::SourceFailure);
        frame.status=ops::Status::Cancelled;
        borrow.publish();
        EXPECT_EQ(m.frame.status,ops::Status::SourceFailure);
        EXPECT_FALSE(m.operations().fail(ops::Status::Invariant));
        borrow.refreshStatus();
        EXPECT_EQ(frame.status,ops::Status::SourceFailure);
    }
    EXPECT_EQ(m.frame.status,ops::Status::SourceFailure);
}
TEST(AdaptiveControlBorrow, FailureExitPublishesPinsBeforeOriginalStatusMutation)
{
    ActualMetadata<ops::Slot32> m(3); m.complete(std::array<std::uint64_t,1>{7});
    auto actual=m.operations(); ASSERT_TRUE(actual.begin(1)); const auto slot=actual.find(7); ASSERT_NE(slot,ops::None);
    {
        ops::FrameControl frame; ops::OwnerControl owner;
        ops::OrderedControlBorrow<ops::Slot32> borrow(m.view,frame,owner);
        ops::Operations<ops::Slot32> writer(borrow.metadata());
        ASSERT_TRUE(writer.pinInitial(slot,m.slots[slot].generation));
        borrow.publish(); EXPECT_EQ(m.frame.held_pins,1u);
        m.frame.status=ops::Status::Cancelled; borrow.refreshStatus();
        EXPECT_EQ(frame.status,ops::Status::Cancelled);
    }
    EXPECT_EQ(m.frame.status,ops::Status::Cancelled); EXPECT_EQ(m.frame.held_pins,1u);
    ASSERT_TRUE(actual.release({slot,m.slots[slot].generation}));
    EXPECT_FALSE(actual.finish()); EXPECT_EQ(m.frame.phase,ops::Phase::Idle);
}
TEST(AdaptiveControlBorrow, ZeroCapacityBorrowNeverTouchesAnOwnerPointer)
{
    ActualMetadata<ops::Slot32> m(0); ASSERT_EQ(m.view.owner,nullptr); ASSERT_TRUE(m.operations().begin(0));
    {
        ops::FrameControl frame; ops::OwnerControl owner;
        ops::OrderedControlBorrow<ops::Slot32> borrow(m.view,frame,owner);
        EXPECT_EQ(borrow.metadata().owner,nullptr);
        ops::Operations<ops::Slot32> writer(borrow.metadata());
        ASSERT_TRUE(writer.beginPolicy(0)); ASSERT_TRUE(writer.prepare());
        ASSERT_TRUE(writer.transition(ops::Phase::Prepared,ops::Phase::Validated)); ASSERT_TRUE(writer.finish());
    }
    EXPECT_EQ(m.frame.phase,ops::Phase::Idle);
}
} // namespace
} // namespace llaminar2::test::adaptive
