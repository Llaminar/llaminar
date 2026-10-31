/**
 * @file Test__PooledImmutableOwnerPlan.cpp
 * @brief Device-free replay of actual leased physical-owner allocation and cancellation.
 *
 * Records are decoded component inputs, never authenticated native transport,
 * source-service or serving capabilities. Arrays are actual owner state, not a
 * second cache policy or physical memory ledger.
 */
#include "kernels/qwen4exp/PooledImmutablePhysicalOwnerGeometry.h"
#include "kernels/qwen4exp/PooledImmutableOwnerPlanOps.h"
#include "loaders/ModelLoader.h"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <memory>
#include <cstring>
#include <vector>
namespace llaminar2::test {
namespace {
namespace a = qwen4exp::adaptive_lookup;
namespace p = a::pooled;
using Geometry = qwen4exp::PooledImmutablePhysicalOwnerGeometry;
/** @return Checked heterogeneous storage declaration independent of coordinator and consumer. */
Geometry geometry(std::uint64_t rows=0x100000100ULL) {
    const std::array<PhysicalMemoryAllocatorIdentity,3> owners{{{0,DeviceId::cuda(0)},
        {1,DeviceId::rocm(0)},{2,DeviceId::rocm(1)}}};
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
    const auto layout = ImmutableRowTableLayoutGeometry::checked(
        ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,rows,90),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
        {{{owners[0],ImmutableRowStorage::VRAM},0,1,{}},
         {{owners[1],ImmutableRowStorage::VRAM},0,2,{}},
         {{owners[2],ImmutableRowStorage::VRAM},0,0,{}}},
        {{3,DeviceId::cuda(1)},{4,DeviceId::cpu()}},{},
        {{{5,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},
        {},PhysicalMemoryAllocatorIdentity{3,DeviceId::cuda(1)},8);
    return Geometry::fromLayout(layout,owners,{3,DeviceId::cuda(1)});
}
/** @brief One actual physical owner plus its mandatory accepted-plan banks. */
template<class Slot> struct ActualOwnerStorage {
    std::vector<Slot> slots;
    p::PhysicalOwnerControl control{};
    p::OwnerFrame frame{};
    p::AllocationFrame plan{};
    std::array<p::AllocationDecision,8> accepted{};
    std::array<p::Grant,8> grants{}, initial_pins{};
    std::array<std::uint64_t,8> keys_a{},keys_b{};
    std::array<std::uint32_t,256> radix_counts{};
    p::OwnerView<Slot> view;
    explicit ActualOwnerStorage(p::OwnerParameters parameters)
        :slots(parameters.capacity()),view{slots.data(),slots.empty()?nullptr:&control,&frame,71,parameters,initial_pins.data()} {
        for(std::uint32_t i=0;i<slots.size();++i) operations().initializeSlot(i);
        if(!operations().initialize()) throw std::logic_error("Actual owner initialization failed");
    }
    p::OwnerOperations<Slot> operations(){return p::OwnerOperations<Slot>(view);}
    p::OwnerPlanOperations<Slot> allocation(){return {view,plan,{accepted.data(),grants.data(),{keys_a.data(),keys_b.data(),radix_counts.data()},8}};}
};
/** @brief Actual participant header frontier, including zero-slot storage owner. */
template<class Slot> struct GroupStorage {
    Geometry declaration;
    std::vector<std::unique_ptr<ActualOwnerStorage<Slot>>> owners;
    p::Interval interval{71,1,1};
    explicit GroupStorage(std::uint64_t rows=0x100000100ULL):declaration(geometry(rows)){
        for(std::uint32_t i=0;i<declaration.owners().size();++i)
            owners.push_back(std::make_unique<ActualOwnerStorage<Slot>>(declaration.ownerParameters(i)));
        std::array<p::PinHeader,3> headers{};
        for(std::size_t i=0;i<owners.size();++i){
            if(!owners[i]->operations().begin(interval))throw std::logic_error("Owner begin failed");
            headers[i]=owners[i]->operations().sealPins(interval);
        }
        for(auto &owner:owners)
            if(!owner->operations().acceptPinFrontier(interval,headers.data(),headers.size()))
                throw std::logic_error("Owner pin frontier failed");
    }
};
using Group=GroupStorage<a::Slot64>;
/** @brief Execute actual peer free waves and bind their one complete owner-authored frontier. */
template<class Owners> void freeFrontier(Owners &g, bool consumeAll = false)
{
    std::array<p::FreeAllocationReceipt,3> receipts{};
    for(std::size_t i=0;i<g.owners.size();++i){
        auto &owner=*g.owners[i];
        if(owner.plan.phase==p::AllocationPhase::Idle && !owner.allocation().acquire(g.interval))
            throw std::logic_error("Peer availability failed");
        if(owner.plan.phase==p::AllocationPhase::Leased){
            std::array<p::AllocationDecision,8> decisions{};
            const auto count=consumeAll?owner.plan.lease.free_slots:0u;
            if(count>decisions.size())throw std::logic_error("Fixture free bound exceeded");
            for(std::uint32_t at=0;at<count;++at)
                decisions[at]={40+i*8+at,static_cast<std::uint32_t>(i*2+at),p::CommandKind::ReserveFree,{},p::CommandOrdinal{at+1}};
            if(!owner.allocation().accept(p::FreeAllocationPlan{owner.plan.lease,decisions.data(),count}))
                throw std::logic_error("Peer free acceptance failed");
        }
        if(owner.plan.phase==p::AllocationPhase::Accepted && !owner.allocation().executeFree())
            throw std::logic_error("Peer free execution failed");
        receipts[i]=owner.allocation().freeReceipt();
    }
    for(auto &owner:g.owners)
        if(!owner->allocation().acceptFreeFrontier(receipts.data(),receipts.size()))
            throw std::logic_error("Complete free frontier failed");
}
/** @brief Warm one row through complete typed free/source/ACK/retirement phases, then begin the next frame. */
template<class Owners> p::Grant warmOwner(Owners &g,std::size_t index,std::uint64_t row)
{
    const auto require=[](bool ok){if(!ok)throw std::logic_error("Warm owner transaction failed");};
    auto &target=*g.owners[index];require(target.allocation().acquire(g.interval));
    const p::AllocationDecision d{row,0,p::CommandKind::ReserveFree,{},p::CommandOrdinal{1}};
    require(target.allocation().accept(p::FreeAllocationPlan{target.plan.lease,&d,1}));
    freeFrontier(g);
    const auto warm=target.grants[0];
    const p::SourceWindow source{71,1,g.interval.sequence};
    std::array<p::PublishHeader,3> headers{};
    for(std::size_t i=0;i<g.owners.size();++i){
        auto &owner=*g.owners[i];require(owner.allocation().sealSource(source));
        const p::FillAcknowledgement ack{source,warm.destination_token};
        headers[i]=owner.operations().validateFills(g.interval,i==index?&warm:nullptr,i==index?&ack:nullptr,i==index?1:0);
    }
    for(auto &owner:g.owners)require(owner->operations().acceptPublishFrontier(g.interval,headers.data(),headers.size()));
    require(target.operations().makeReady(g.interval,warm,{source,warm.destination_token}));
    require(target.operations().release(g.interval,warm));
    for(auto &owner:g.owners)require(owner->allocation().retire(p::SourceRetirement::Consumed,source));
    ++g.interval.sequence;
    std::array<p::PinHeader,3> pins{};
    for(std::size_t i=0;i<g.owners.size();++i){require(g.owners[i]->operations().begin(g.interval));pins[i]=g.owners[i]->operations().sealPins(g.interval);}
    for(auto &owner:g.owners)require(owner->operations().acceptPinFrontier(g.interval,pins.data(),pins.size()));
    return warm;
}
/** @return One real free-slot decision with globally scoped promotion coordinate. */
p::AllocationDecision freeDecision(std::uint64_t row=7,std::uint32_t unique=4){
    return {row,unique,p::CommandKind::ReserveFree,{},p::CommandOrdinal{1}};
}
}
TEST(PooledImmutableOwnerPlan, IndependentMixedVendorStorageAndC0CoordinatorHaveExactGeometry){
    Group g;
    EXPECT_FALSE(g.declaration.requesterCoordinate());
    EXPECT_EQ(g.declaration.coordinator(),(PhysicalMemoryAllocatorIdentity{3,DeviceId::cuda(1)}));
    EXPECT_NE(g.declaration.owners()[0].allocator,g.declaration.owners()[1].allocator);
    for(auto &owner:g.owners) ASSERT_TRUE(owner->allocation().acquire(g.interval));
    EXPECT_EQ(g.owners[0]->plan.lease.free_slots,1u);
    EXPECT_EQ(g.owners[1]->plan.lease.free_slots,2u);
    EXPECT_EQ(g.owners[2]->plan.lease.free_slots,0u);
    ASSERT_TRUE(g.owners[2]->allocation().accept(p::FreeAllocationPlan{g.owners[2]->plan.lease,nullptr,0}));
    ASSERT_TRUE(g.owners[2]->allocation().cancel(p::SourceRetirement::Unpublished));
    EXPECT_TRUE(g.owners[2]->operations().retire(g.interval,p::SourceRetirement::Unpublished));
}
TEST(PooledImmutableOwnerPlan, UnacceptedCancellationClosesLeaseWithoutSlotMutation){
    Group g;auto &owner=*g.owners[0];
    ASSERT_TRUE(owner.allocation().acquire(g.interval));
    ASSERT_TRUE(owner.allocation().cancel(p::SourceRetirement::Unpublished));
    EXPECT_EQ(owner.control.next_slot_generation,0u);
    EXPECT_EQ(owner.control.filled,0u);
    for(const auto &slot:owner.slots)EXPECT_EQ(a::PhysicalSlot<a::Slot64>::state(slot),a::SlotState::Empty);
    EXPECT_TRUE(owner.operations().retire(g.interval,p::SourceRetirement::Unpublished));
}
TEST(PooledImmutableOwnerPlan, AcceptedCancellationDrainsImmutablePlanAndRetiresItsActualGrant){
    Group g;auto &owner=*g.owners[0];
    ASSERT_TRUE(owner.allocation().acquire(g.interval));
    auto d=freeDecision();
    ASSERT_TRUE(owner.allocation().accept(p::FreeAllocationPlan{owner.plan.lease,&d,1}));
    d.row=99; // Acceptance retained its own immutable decoded plan bank.
    ASSERT_TRUE(owner.allocation().cancel(p::SourceRetirement::Unpublished));
    const auto grant=owner.grants[0];
    EXPECT_EQ(grant.row,7u);EXPECT_EQ(grant.destination_token,5u);
    EXPECT_EQ(grant.generation,1u);EXPECT_NE(grant.destination_token,grant.generation);
    EXPECT_EQ(a::PhysicalSlot<a::Slot64>::state(owner.slots[grant.slot]),a::SlotState::Reserved);
    const auto receipt=owner.operations().releaseWithReceipt(g.interval,grant);
    EXPECT_EQ(receipt.disposition,p::ReleaseDisposition::RecycledEmpty);
    EXPECT_TRUE(owner.operations().retire(g.interval,p::SourceRetirement::Unpublished));
}
TEST(PooledImmutableOwnerPlan, DuplicateStaleForeignAndOverLeasePlansFailBeforePhysicalMutation){
    for(int failure=0;failure<4;++failure){
        Group g;auto &owner=*g.owners[0];ASSERT_TRUE(owner.allocation().acquire(g.interval));
        auto lease=owner.plan.lease;
        std::array decisions{freeDecision(),freeDecision(8,5)};decisions[1].ordinal.value=2;
        if(failure==0)decisions[1].unique=decisions[0].unique;
        if(failure==1)++lease.interval.sequence;
        if(failure==2)++lease.owner;
        if(failure==3)lease.free_slots=0;
        EXPECT_FALSE(owner.allocation().accept(p::FreeAllocationPlan{lease,decisions.data(),decisions.size()}));
        EXPECT_EQ(owner.control.next_slot_generation,0u);EXPECT_EQ(owner.control.filled,0u);
    }
}
TEST(PooledImmutableOwnerPlan, LeasedOwnerRejectsIndependentReservationAndDuplicatePlanExecution){
    {Group g;auto &owner=*g.owners[0];ASSERT_TRUE(owner.allocation().acquire(g.interval));
     EXPECT_EQ(owner.operations().reserveFree(g.interval,7,0,{1}).kind,p::GrantKind::Absent);
     EXPECT_EQ(owner.control.next_slot_generation,0u);}
    {Group g;auto &owner=*g.owners[0];ASSERT_TRUE(owner.allocation().acquire(g.interval));auto d=freeDecision();
     ASSERT_TRUE(owner.allocation().accept(p::FreeAllocationPlan{owner.plan.lease,&d,1}));ASSERT_TRUE(owner.allocation().executeFree());
     EXPECT_FALSE(owner.allocation().executeFree());EXPECT_EQ(owner.control.next_slot_generation,1u);}
}
TEST(PooledImmutableOwnerPlan, SourceACKRequiresActualGlobalTokenAndConsumedRetirement){
    for(int bad=0;bad<3;++bad){
        Group g;auto &owner=*g.owners[0];ASSERT_TRUE(owner.allocation().acquire(g.interval));
        const auto d=freeDecision();ASSERT_TRUE(owner.allocation().accept(p::FreeAllocationPlan{owner.plan.lease,&d,1}));
        ASSERT_TRUE(owner.allocation().executeFree());const auto grant=owner.grants[0];
        const p::SourceWindow source{71,9,3};
        ASSERT_NO_THROW(freeFrontier(g));
        ASSERT_TRUE(owner.allocation().sealSource(source));
        p::FillAcknowledgement ack{source,grant.destination_token};
        if(bad==0)++ack.destination_token;
        if(bad==1)++ack.source.sequence;
        if(bad<2){
            const auto header=owner.operations().validateFills(g.interval,&grant,&ack,1);
            EXPECT_NE(header.status,a::Status::Success);
            EXPECT_EQ(a::PhysicalSlot<a::Slot64>::state(owner.slots[grant.slot]),a::SlotState::Reserved);
        }else{
            EXPECT_FALSE(owner.allocation().cancel(p::SourceRetirement::Unpublished));
            EXPECT_EQ(owner.frame.status,a::Status::SourceFailure);
        }
    }
}
TEST(PooledImmutableOwnerPlan, VictimMutationOccursOnlyAfterCompletePlanAcceptance){
    for(bool accept:{false,true}){
        Group g;auto &owner=*g.owners[0];
        // Actual owner operations prepare a warm readable row before the next frame.
        const auto warm=owner.operations().reserveFree(g.interval,2,0,{1});
        const p::SourceWindow source{71,1,1};
        std::array<p::PublishHeader,3> headers{};
        for(std::size_t i=0;i<g.owners.size();++i){
            auto &endpoint=*g.owners[i];ASSERT_TRUE(endpoint.operations().sealReservations(g.interval,source));
            const p::FillAcknowledgement ack{source,warm.destination_token};
            headers[i]=endpoint.operations().validateFills(g.interval,i?nullptr:&warm,i?nullptr:&ack,i?0:1);
        }
        for(auto &endpoint:g.owners)ASSERT_TRUE(endpoint->operations().acceptPublishFrontier(g.interval,headers.data(),headers.size()));
        ASSERT_TRUE(owner.operations().makeReady(g.interval,warm,{source,warm.destination_token}));
        ASSERT_TRUE(owner.operations().release(g.interval,warm));
        std::array<p::PinHeader,3> pins{};
        for(std::size_t i=0;i<g.owners.size();++i){
            ASSERT_TRUE(g.owners[i]->operations().retire(g.interval,p::SourceRetirement::Consumed,source));
            const p::Interval next{71,1,2};ASSERT_TRUE(g.owners[i]->operations().begin(next));
            pins[i]=g.owners[i]->operations().sealPins(next);
        }
        g.interval.sequence=2;
        for(auto &endpoint:g.owners)ASSERT_TRUE(endpoint->operations().acceptPinFrontier(g.interval,pins.data(),pins.size()));
        ASSERT_TRUE(owner.allocation().acquire(g.interval));
        ASSERT_NO_THROW(freeFrontier(g,true));
        const p::AllocationDecision victim{9,4,p::CommandKind::ReserveVictim,{warm.row,warm.generation,warm.owner,warm.slot},{1}};
        if(accept)ASSERT_TRUE(owner.allocation().accept(p::VictimAllocationPlan{owner.plan.lease,&victim,1}));
        ASSERT_TRUE(owner.allocation().cancel(p::SourceRetirement::Unpublished));
        EXPECT_EQ(owner.slots[warm.slot].row,accept?9u:2u);
        EXPECT_EQ(owner.slots[warm.slot].generation,accept?2u:1u);
        if(accept)ASSERT_TRUE(owner.operations().release(g.interval,owner.grants[0]));
        ASSERT_TRUE(owner.operations().retire(g.interval,p::SourceRetirement::Unpublished));
    }
}

TEST(PooledImmutableOwnerPlan, MandatoryAcceptedPlanBankIsPricedOnceOnEachExactAllocator){
    const auto declaration=geometry();
    for(const auto &owner:declaration.owners()){
        const PhysicalMemoryResource observed{owner.allocator.world_rank,owner.allocator.device,1ULL<<30,1ULL<<30};
        EXPECT_EQ(declaration.mandatoryBOM(owner.coordinate,observed).totalBytes(),owner.frame.bytes);
        EXPECT_EQ(declaration.cacheBOM(owner.coordinate,observed).totalBytes(),owner.cache.bytes);
        EXPECT_EQ(owner.frame.free_headers,owner.frame.publish_headers);
        EXPECT_EQ(owner.frame.victim_headers,owner.frame.publish_headers);
        EXPECT_EQ(owner.frame.fill_order,owner.frame.validation_a);
        EXPECT_EQ(owner.frame.allocation_control%64,0u);
        EXPECT_EQ(owner.frame.accepted_plan%64,0u);
        EXPECT_GE(owner.frame.commands,owner.frame.accepted_plan+8*sizeof(p::AllocationDecision));
        EXPECT_GE(owner.frame.accepted_plan,owner.frame.allocation_control+sizeof(p::AllocationFrame));
        auto foreign=observed;foreign.world_rank+=1;
        EXPECT_THROW(declaration.mandatoryBOM(owner.coordinate,foreign),std::invalid_argument);
    }
}

TEST(PooledImmutableOwnerPlan, ValidationWritesOnlyLiveScratchAndRecycledAvailabilitySurvivesReset){
    Group g;auto &owner=*g.owners[1];
    owner.keys_a.fill(0xaabbccddeeff0011ULL);owner.keys_b.fill(0x1122334455667788ULL);
    ASSERT_TRUE(owner.allocation().acquire(g.interval));
    const auto prior=owner.plan.lease;
    const auto d=freeDecision();ASSERT_TRUE(owner.allocation().accept(p::FreeAllocationPlan{prior,&d,1}));
    for(std::size_t i=1;i<owner.keys_a.size();++i){
        EXPECT_EQ(owner.keys_a[i],0xaabbccddeeff0011ULL);
        EXPECT_EQ(owner.keys_b[i],0x1122334455667788ULL);
    }
    ASSERT_TRUE(owner.allocation().cancel(p::SourceRetirement::Unpublished));
    ASSERT_TRUE(owner.operations().release(g.interval,owner.grants[0]));
    EXPECT_EQ(owner.control.filled,1u);EXPECT_EQ(owner.control.recycled_free,1u);
    ASSERT_TRUE(owner.allocation().retire(p::SourceRetirement::Unpublished));
    for(std::size_t i=0;i<g.owners.size();++i)if(i!=1){
        ASSERT_TRUE(g.owners[i]->operations().cancel(g.interval,p::SourceRetirement::Unpublished));
        ASSERT_TRUE(g.owners[i]->operations().retire(g.interval,p::SourceRetirement::Unpublished));
    }
    ++g.interval.sequence;
    std::array<p::PinHeader,3> pins{};
    for(std::size_t i=0;i<g.owners.size();++i){
        ASSERT_TRUE(g.owners[i]->operations().begin(g.interval));
        pins[i]=g.owners[i]->operations().sealPins(g.interval);
    }
    for(auto &endpoint:g.owners)ASSERT_TRUE(endpoint->operations().acceptPinFrontier(g.interval,pins.data(),pins.size()));
    ASSERT_TRUE(owner.allocation().acquire(g.interval));
    EXPECT_EQ(owner.plan.lease.free_slots,2u); // Recycled slot plus untouched virgin slot.
    EXPECT_GT(owner.plan.lease.interval.sequence,prior.interval.sequence);
    EXPECT_FALSE(owner.allocation().accept(p::FreeAllocationPlan{prior,&d,1}));
    EXPECT_EQ(owner.control.next_slot_generation,1u);
    EXPECT_EQ(owner.control.recycled_free,1u);
}
TEST(PooledImmutableOwnerPlan, LinearValidationRejectsDistinctUniqueRowsWithDuplicateSourceKey){
    Group g;auto &owner=*g.owners[1];ASSERT_TRUE(owner.allocation().acquire(g.interval));
    std::array decisions{freeDecision(7,1),freeDecision(7,2)};decisions[1].ordinal.value=2;
    EXPECT_FALSE(owner.allocation().accept(p::FreeAllocationPlan{owner.plan.lease,decisions.data(),decisions.size()}));
    EXPECT_EQ(owner.control.filled,0u);EXPECT_EQ(owner.control.next_slot_generation,0u);
}

TEST(PooledImmutableOwnerPlan, NullOwnerAndAliasedBanksRejectWithoutDereferenceOrPlanPublication){
    Group g;auto &owner=*g.owners[1];
    const auto storage=p::AllocationPlanStorage{owner.accepted.data(),owner.grants.data(),
        {owner.keys_a.data(),owner.keys_b.data(),owner.radix_counts.data()},8};
    auto invalid=owner.view;invalid.frame=nullptr;
    p::OwnerPlanOperations<a::Slot64> null(invalid,owner.plan,storage);
    EXPECT_FALSE(null.acquire(g.interval));EXPECT_FALSE(null.accept(p::FreeAllocationPlan{{},nullptr,0}));
    EXPECT_FALSE(null.executeFree());EXPECT_FALSE(null.cancel(p::SourceRetirement::Unpublished));
    EXPECT_FALSE(null.retire(p::SourceRetirement::Unpublished));
    auto aliased=storage;aliased.scratch.keys_b=aliased.scratch.keys_a;
    p::OwnerPlanOperations<a::Slot64> alias(owner.view,owner.plan,aliased);
    EXPECT_FALSE(alias.acquire(g.interval));EXPECT_EQ(owner.control.filled,0u);
}
TEST(PooledImmutableOwnerPlan, IncomingDecisionAliasAndOversizedCountRejectBeforeScratchWrites){
    for(bool alias:{false,true}){
        Group g;auto &owner=*g.owners[1];ASSERT_TRUE(owner.allocation().acquire(g.interval));
        owner.accepted[0]=freeDecision();const auto before=owner.keys_a;
        const auto decision=freeDecision();
        EXPECT_FALSE(owner.allocation().accept(p::FreeAllocationPlan{owner.plan.lease,alias?owner.accepted.data():&decision,alias?1u:9u}));
        EXPECT_EQ(owner.keys_a,before);EXPECT_EQ(owner.control.next_slot_generation,0u);
        EXPECT_EQ(owner.plan.phase,p::AllocationPhase::Leased);
    }
}
TEST(PooledImmutableOwnerPlan, GenerationExhaustionRejectsWholePlanWithoutReservation){
    Group g;auto &owner=*g.owners[1];owner.control.next_slot_generation=~std::uint64_t{0};
    ASSERT_TRUE(owner.allocation().acquire(g.interval));const auto d=freeDecision();
    EXPECT_FALSE(owner.allocation().accept(p::FreeAllocationPlan{owner.plan.lease,&d,1}));EXPECT_EQ(owner.control.filled,0u);
}
TEST(PooledImmutableOwnerPlan, AllocationCannotDuplicateActualInitialPinnedReadyRow){
    Group g;auto &owner=*g.owners[1];
    const auto warm=owner.operations().reserveFree(g.interval,7,0,{1});
    const p::SourceWindow source{71,1,1};std::array<p::PublishHeader,3> headers{};
    for(std::size_t i=0;i<g.owners.size();++i){
        auto &endpoint=*g.owners[i];ASSERT_TRUE(endpoint.operations().sealReservations(g.interval,source));
        const p::FillAcknowledgement ack{source,warm.destination_token};
        headers[i]=endpoint.operations().validateFills(g.interval,i==1?&warm:nullptr,i==1?&ack:nullptr,i==1?1:0);
    }
    for(auto &endpoint:g.owners)ASSERT_TRUE(endpoint->operations().acceptPublishFrontier(g.interval,headers.data(),headers.size()));
    ASSERT_TRUE(owner.operations().makeReady(g.interval,warm,{source,warm.destination_token}));
    ASSERT_TRUE(owner.operations().release(g.interval,warm));
    for(auto &endpoint:g.owners)ASSERT_TRUE(endpoint->operations().retire(g.interval,p::SourceRetirement::Consumed,source));
    ++g.interval.sequence;
    for(auto &endpoint:g.owners)ASSERT_TRUE(endpoint->operations().begin(g.interval));
    const auto read=owner.operations().pinInitial(g.interval,{warm.row,warm.generation,warm.owner,warm.slot},0,{1});
    ASSERT_EQ(read.kind,p::GrantKind::Read);ASSERT_EQ(owner.frame.initial_pin_count,1u);
    std::array<p::PinHeader,3> pins{};
    for(std::size_t i=0;i<g.owners.size();++i)pins[i]=g.owners[i]->operations().sealPins(g.interval);
    for(auto &endpoint:g.owners)ASSERT_TRUE(endpoint->operations().acceptPinFrontier(g.interval,pins.data(),pins.size()));
    ASSERT_TRUE(owner.allocation().acquire(g.interval));
    auto duplicate=freeDecision(7,4);duplicate.ordinal.value=2;
    EXPECT_FALSE(owner.allocation().accept(p::FreeAllocationPlan{owner.plan.lease,&duplicate,1}));
    EXPECT_EQ(owner.control.filled,1u);EXPECT_EQ(owner.control.next_slot_generation,1u);
    EXPECT_EQ(a::PhysicalSlot<a::Slot64>::state(owner.slots[warm.slot]),a::SlotState::Ready);
    EXPECT_TRUE(a::PhysicalSlot<a::Slot64>::pinned(owner.slots[warm.slot]));
}

TEST(PooledImmutableOwnerPlan, StaleStatusAndMalformedPinCountRejectBeforeDecisionOrScratchMutation){
    for(bool malformed:{false,true}){
        Group g;auto &owner=*g.owners[1];ASSERT_TRUE(owner.allocation().acquire(g.interval));
        if(malformed)owner.frame.initial_pin_count=9;
        else owner.frame.status=a::Status::SourceFailure;
        const auto before=owner.keys_a;const auto d=freeDecision();
        EXPECT_FALSE(owner.allocation().accept(p::FreeAllocationPlan{owner.plan.lease,&d,1}));
        EXPECT_EQ(owner.keys_a,before);EXPECT_EQ(owner.control.next_slot_generation,0u);
        EXPECT_EQ(owner.plan.phase,p::AllocationPhase::Leased);
    }
}

TEST(PooledImmutableOwnerPlan, UnusedFreeQuotaOnAnotherOwnerForbidsValidLocalVictim){
    Group g;const auto warm=warmOwner(g,0,7);auto &owner=*g.owners[0];
    ASSERT_NO_THROW(freeFrontier(g));ASSERT_FALSE(owner.plan.all_free_consumed);
    const p::AllocationDecision victim{9,4,p::CommandKind::ReserveVictim,{warm.row,warm.generation,warm.owner,warm.slot},{1}};
    EXPECT_FALSE(owner.allocation().accept(p::VictimAllocationPlan{owner.plan.lease,&victim,1}));
    EXPECT_EQ(owner.slots[warm.slot].row,7u);EXPECT_EQ(owner.slots[warm.slot].generation,warm.generation);
    EXPECT_EQ(g.owners[1]->plan.lease.free_slots,2u);EXPECT_EQ(g.owners[1]->plan.free_count,0u);
}
TEST(PooledImmutableOwnerPlan, TwoWavesRetainGrantPrefixAndOneSourceUntilAllReadersRetire){
    Group g;const auto warm=warmOwner(g,1,7);auto &owner=*g.owners[1];
    ASSERT_NO_THROW(freeFrontier(g,true));ASSERT_TRUE(owner.plan.all_free_consumed);
    const auto first=owner.grants[0];const auto interval=g.interval;
    const p::AllocationDecision victim{9,1,p::CommandKind::ReserveVictim,{warm.row,warm.generation,warm.owner,warm.slot},{2}};
    ASSERT_TRUE(owner.allocation().accept(p::VictimAllocationPlan{owner.plan.lease,&victim,1}));
    ASSERT_TRUE(owner.allocation().executeVictims());
    const auto victim_receipt=owner.allocation().victimReceipt();
    EXPECT_EQ(victim_receipt.granted,1u);EXPECT_EQ(victim_receipt.status,a::Status::Success);
    EXPECT_EQ(victim_receipt.lease.interval.sequence,interval.sequence);
    EXPECT_EQ(owner.plan.count,2u);EXPECT_EQ(owner.plan.wave_offset,1u);
    EXPECT_EQ(owner.grants[0].generation,first.generation);EXPECT_EQ(owner.grants[0].row,first.row);
    EXPECT_GT(owner.grants[1].generation,first.generation);EXPECT_EQ(owner.frame.interval.sequence,interval.sequence);
    const p::SourceWindow source{71,2,2};std::array<p::PublishHeader,3> headers{};
    std::array<std::array<p::FillAcknowledgement,8>,3> acks{};
    for(std::size_t i=0;i<g.owners.size();++i){
        auto &endpoint=*g.owners[i];ASSERT_TRUE(endpoint.allocation().sealSource(source));
        for(std::uint32_t at=0;at<endpoint.plan.count;++at)acks[i][at]={source,endpoint.grants[at].destination_token};
        headers[i]=endpoint.allocation().validateFills(acks[i].data(),endpoint.plan.count);
        ASSERT_EQ(headers[i].status,a::Status::Success);
    }
    for(auto &endpoint:g.owners)ASSERT_TRUE(endpoint->operations().acceptPublishFrontier(g.interval,headers.data(),headers.size()));
    for(std::size_t i=0;i<g.owners.size();++i){
        auto &endpoint=*g.owners[i];const auto count=endpoint.plan.count;
        for(std::uint32_t at=0;at<count;++at){
            ASSERT_TRUE(endpoint.operations().makeReady(g.interval,endpoint.grants[at],acks[i][at]));
            ASSERT_TRUE(endpoint.operations().release(g.interval,endpoint.grants[at]));
        }
        ASSERT_TRUE(endpoint.allocation().retire(p::SourceRetirement::Consumed,source));
        EXPECT_EQ(endpoint.plan.phase,p::AllocationPhase::Idle);
    }
    EXPECT_EQ(owner.grants[0].unique,first.unique);
    EXPECT_EQ(owner.grants[1].unique,1u);
    EXPECT_EQ(owner.slots[warm.slot].row,9u);EXPECT_EQ(owner.control.next_slot_generation,3u);
    // Repeated all-hit replay uses the retained physical rows and new original
    // frame identity. There is no second source window and no generation reset.
    for(unsigned replay=0;replay<3;++replay){
        ++g.interval.sequence;
        std::array<std::array<p::Grant,2>,3> reads{};
        std::array<p::PinHeader,3> pins{};
        for(std::size_t i=0;i<g.owners.size();++i){
            auto &endpoint=*g.owners[i];ASSERT_TRUE(endpoint.operations().begin(g.interval));
            for(std::uint32_t at=0;at<endpoint.slots.size();++at){
                const auto &slot=endpoint.slots[at];
                reads[i][at]=endpoint.operations().pinInitial(g.interval,{slot.row,slot.generation,static_cast<std::uint32_t>(i),at},
                    static_cast<std::uint32_t>(i*2+at),{at+1});
                ASSERT_EQ(reads[i][at].kind,p::GrantKind::Read);
            }
            pins[i]=endpoint.operations().sealPins(g.interval);
        }
        for(auto &endpoint:g.owners)ASSERT_TRUE(endpoint->operations().acceptPinFrontier(g.interval,pins.data(),pins.size()));
        ASSERT_NO_THROW(freeFrontier(g));
        for(std::size_t i=0;i<g.owners.size();++i){auto &endpoint=*g.owners[i];
            ASSERT_TRUE(endpoint.allocation().sealSource({}));
            headers[i]=endpoint.allocation().validateFills(nullptr,0);
            ASSERT_EQ(headers[i].status,a::Status::Success);
        }
        for(std::size_t i=0;i<g.owners.size();++i){auto &endpoint=*g.owners[i];
            ASSERT_TRUE(endpoint.operations().acceptPublishFrontier(g.interval,headers.data(),headers.size()));
            for(std::uint32_t at=0;at<endpoint.slots.size();++at)ASSERT_TRUE(endpoint.operations().release(g.interval,reads[i][at]));
            ASSERT_TRUE(endpoint.allocation().retire(p::SourceRetirement::Unpublished));
        }
        EXPECT_EQ(owner.control.next_slot_generation,3u);
        EXPECT_EQ(owner.slots[warm.slot].row,9u);
    }
}
TEST(PooledImmutableOwnerPlan, PriorWaveRowCannotBeAllocatedAgainUnderAnotherUniqueCoordinate){
    Group g;const auto warm=warmOwner(g,1,7);auto &owner=*g.owners[1];ASSERT_NO_THROW(freeFrontier(g,true));
    const auto prior=owner.grants[0];
    const p::AllocationDecision duplicate{prior.row,4,p::CommandKind::ReserveVictim,{warm.row,warm.generation,warm.owner,warm.slot},{2}};
    EXPECT_FALSE(owner.allocation().accept(p::VictimAllocationPlan{owner.plan.lease,&duplicate,1}));
    EXPECT_EQ(owner.plan.count,1u);EXPECT_EQ(owner.slots[warm.slot].row,7u);
    EXPECT_EQ(owner.grants[0].generation,prior.generation);
}
TEST(PooledImmutableOwnerPlan, MissingDuplicateAndStaleFreeFrontiersCannotAuthorizeVictimWave){
    for(int failure=0;failure<3;++failure){
        Group g;std::array<p::FreeAllocationReceipt,3> receipts{};
        for(std::size_t i=0;i<g.owners.size();++i){auto &owner=*g.owners[i];ASSERT_TRUE(owner.allocation().acquire(g.interval));
            ASSERT_TRUE(owner.allocation().accept(p::FreeAllocationPlan{owner.plan.lease,nullptr,0}));
            ASSERT_TRUE(owner.allocation().executeFree());receipts[i]=owner.allocation().freeReceipt();}
        if(failure==1)receipts[1].lease.owner=0;
        if(failure==2)++receipts[1].lease.interval.sequence;
        EXPECT_FALSE(g.owners[0]->allocation().acceptFreeFrontier(receipts.data(),failure==0?2:3));
        EXPECT_EQ(g.owners[0]->control.next_slot_generation,0u);
    }
}
TEST(PooledImmutableOwnerPlan, VictimWaveCancellationDrainsOnlyAfterItsOwnAcceptance){
    for(bool accept:{false,true}){
        Group g;const auto warm=warmOwner(g,1,7);auto &owner=*g.owners[1];ASSERT_NO_THROW(freeFrontier(g,true));
        const p::AllocationDecision victim{9,4,p::CommandKind::ReserveVictim,{warm.row,warm.generation,warm.owner,warm.slot},{2}};
        if(accept)ASSERT_TRUE(owner.allocation().accept(p::VictimAllocationPlan{owner.plan.lease,&victim,1}));
        for(auto &endpoint:g.owners){
            ASSERT_TRUE(endpoint->allocation().cancel(p::SourceRetirement::Unpublished));
            const auto receipt=endpoint->allocation().victimReceipt();
            ASSERT_EQ(receipt.status,a::Status::Cancelled);
            EXPECT_EQ(receipt.granted,endpoint.get()==&owner && accept?1u:0u);
            const auto count=endpoint->plan.count;
            for(std::uint32_t at=0;at<count;++at)ASSERT_TRUE(endpoint->operations().release(g.interval,endpoint->grants[at]));
            ASSERT_TRUE(endpoint->allocation().retire(p::SourceRetirement::Unpublished));
        }
        EXPECT_EQ(owner.slots[warm.slot].row,accept?9u:7u);
        EXPECT_EQ(owner.slots[warm.slot].generation,accept?3u:warm.generation);
        EXPECT_EQ(a::PhysicalSlot<a::Slot64>::state(owner.slots[warm.slot]),accept?a::SlotState::Empty:a::SlotState::Ready);
    }
}

TEST(PooledImmutableOwnerPlan, CancelledRemoteFreeReceiptCannotAuthorizeVictimEvenAfterFullConsumption){
    for(bool accepted:{false,true}){
        Group g;const auto warm=warmOwner(g,0,7);std::array<p::FreeAllocationReceipt,3> receipts{};
        for(std::size_t i=0;i<g.owners.size();++i){
            auto &owner=*g.owners[i];ASSERT_TRUE(owner.allocation().acquire(g.interval));
            if(i==1){
                std::array<p::AllocationDecision,2> decisions{{freeDecision(40,2),freeDecision(41,3)}};
                decisions[1].ordinal.value=2;
                if(accepted)ASSERT_TRUE(owner.allocation().accept(p::FreeAllocationPlan{owner.plan.lease,decisions.data(),2}));
                ASSERT_TRUE(owner.allocation().cancel(p::SourceRetirement::Unpublished));
            }else{
                ASSERT_TRUE(owner.allocation().accept(p::FreeAllocationPlan{owner.plan.lease,nullptr,0}));
                ASSERT_TRUE(owner.allocation().executeFree());
            }
            receipts[i]=owner.allocation().freeReceipt();
        }
        ASSERT_EQ(receipts[1].status,a::Status::Cancelled);
        ASSERT_EQ(receipts[1].granted,accepted?2u:0u);
        auto &owner=*g.owners[0];ASSERT_TRUE(owner.allocation().acceptFreeFrontier(receipts.data(),receipts.size()));
        ASSERT_FALSE(owner.plan.all_free_consumed);
        const p::AllocationDecision victim{9,4,p::CommandKind::ReserveVictim,{warm.row,warm.generation,warm.owner,warm.slot},{1}};
        EXPECT_FALSE(owner.allocation().accept(p::VictimAllocationPlan{owner.plan.lease,&victim,1}));
        EXPECT_EQ(owner.slots[warm.slot].row,7u);
    }
}
TEST(PooledImmutableOwnerPlan, AllocationGrantsCannotRetireBeforeSourceOrCancellationBoundary){
    Group g;warmOwner(g,1,7);auto &owner=*g.owners[1];
    ASSERT_NO_THROW(freeFrontier(g,true));const auto grant=owner.grants[0];
    EXPECT_FALSE(owner.operations().release(g.interval,grant));
    EXPECT_TRUE(a::PhysicalSlot<a::Slot64>::pinned(owner.slots[grant.slot]));
    EXPECT_EQ(a::PhysicalSlot<a::Slot64>::state(owner.slots[grant.slot]),a::SlotState::Reserved);
    EXPECT_EQ(owner.control.recycled_free,0u);
}
TEST(PooledImmutableOwnerPlan, OrderedFillPermutationRejectsInvalidIndexUniqueAndDuplicateBeforePhaseMutation){
    for(int bad=0;bad<3;++bad){
        Group g;const auto warm=warmOwner(g,1,7);auto &owner=*g.owners[1];
        ASSERT_NO_THROW(freeFrontier(g,true));
        const p::AllocationDecision victim{9,1,p::CommandKind::ReserveVictim,{warm.row,warm.generation,warm.owner,warm.slot},{2}};
        ASSERT_TRUE(owner.allocation().accept(p::VictimAllocationPlan{owner.plan.lease,&victim,1}));
        ASSERT_TRUE(owner.allocation().executeVictims());
        const auto original=owner.grants;const auto accepted=owner.accepted;
        const p::SourceWindow source{71,2,2};ASSERT_TRUE(owner.allocation().sealSource(source));
        ASSERT_EQ(static_cast<std::uint32_t>(owner.keys_a[0]),1u);
        ASSERT_EQ(static_cast<std::uint32_t>(owner.keys_a[1]),0u);
        std::array<p::FillAcknowledgement,2> acks{{{source,owner.grants[0].destination_token},{source,owner.grants[1].destination_token}}};
        if(bad==0)owner.keys_a[0]=(std::uint64_t{1}<<32)|8;
        if(bad==1)owner.keys_a[0]^=std::uint64_t{1}<<32;
        if(bad==2)owner.keys_a[1]=owner.keys_a[0];
        EXPECT_NE(owner.allocation().validateFills(acks.data(),acks.size()).status,a::Status::Success);
        EXPECT_EQ(owner.frame.phase,p::OwnerPhase::Filling);
        EXPECT_EQ(owner.frame.reservations,2u);
        EXPECT_EQ(std::memcmp(owner.grants.data(),original.data(),sizeof(original)),0);
        EXPECT_EQ(std::memcmp(owner.accepted.data(),accepted.data(),sizeof(accepted)),0);
    }
}
TEST(PooledImmutableOwnerPlan, EmptyVictimWaveClosesUnusedQuotaWithoutSourceOrGenerationReset){
    Group g;ASSERT_NO_THROW(freeFrontier(g));
    std::array<p::PublishHeader,3> headers{};
    for(std::size_t i=0;i<g.owners.size();++i){auto &owner=*g.owners[i];
        ASSERT_FALSE(owner.plan.all_free_consumed);
        ASSERT_TRUE(owner.allocation().accept(p::VictimAllocationPlan{owner.plan.lease,nullptr,0}));
        ASSERT_TRUE(owner.allocation().executeVictims());
        ASSERT_TRUE(owner.allocation().sealSource({}));
        headers[i]=owner.allocation().validateFills(nullptr,0);
        ASSERT_EQ(headers[i].status,a::Status::Success);
    }
    for(auto &owner:g.owners){
        ASSERT_TRUE(owner->operations().acceptPublishFrontier(g.interval,headers.data(),headers.size()));
        ASSERT_TRUE(owner->allocation().retire(p::SourceRetirement::Unpublished));
        EXPECT_EQ(owner->control.next_slot_generation,0u);
    }
}

TEST(PooledImmutableOwnerPlan, SourceAndVictimCannotStartBeforeAllOwnerFreeCompletions){
    for(bool source:{false,true}){
        Group g;const auto warm=warmOwner(g,0,7);auto &owner=*g.owners[0];
        ASSERT_TRUE(owner.allocation().acquire(g.interval));
        ASSERT_TRUE(owner.allocation().accept(p::FreeAllocationPlan{owner.plan.lease,nullptr,0}));
        ASSERT_TRUE(owner.allocation().executeFree());
        const p::AllocationDecision victim{9,4,p::CommandKind::ReserveVictim,{warm.row,warm.generation,warm.owner,warm.slot},{1}};
        if(source)EXPECT_FALSE(owner.allocation().sealSource({71,2,2}));
        else EXPECT_FALSE(owner.allocation().accept(p::VictimAllocationPlan{owner.plan.lease,&victim,1}));
        EXPECT_EQ(owner.slots[warm.slot].row,7u);
        EXPECT_EQ(owner.slots[warm.slot].generation,warm.generation);
        EXPECT_EQ(owner.frame.phase,p::OwnerPhase::PlanReady);
    }
}

TEST(PooledImmutableOwnerPlan, UInt32PhysicalSlotsRunWarmTwoWavesAndOrderedSourceReplay)
{
    GroupStorage<a::Slot32> g(100);const auto warm=warmOwner(g,1,7);auto &owner=*g.owners[1];
    ASSERT_EQ(g.declaration.rowKeyWidth(),a::RowKeyWidth::UInt32);
    ASSERT_NO_THROW(freeFrontier(g,true));
    const p::AllocationDecision victim{9,1,p::CommandKind::ReserveVictim,{warm.row,warm.generation,warm.owner,warm.slot},{2}};
    ASSERT_TRUE(owner.allocation().accept(p::VictimAllocationPlan{owner.plan.lease,&victim,1}));
    ASSERT_TRUE(owner.allocation().executeVictims());
    const p::SourceWindow source{71,2,2};std::array<p::PublishHeader,3> headers{};
    std::array<std::array<p::FillAcknowledgement,8>,3> acknowledgements{};
    for(std::size_t i=0;i<g.owners.size();++i){auto &endpoint=*g.owners[i];
        ASSERT_TRUE(endpoint.allocation().sealSource(source));
        for(std::uint32_t at=0;at<endpoint.plan.count;++at)acknowledgements[i][at]={source,endpoint.grants[at].destination_token};
        headers[i]=endpoint.allocation().validateFills(acknowledgements[i].data(),endpoint.plan.count);
        ASSERT_EQ(headers[i].status,a::Status::Success);
    }
    for(std::size_t i=0;i<g.owners.size();++i){auto &endpoint=*g.owners[i];
        ASSERT_TRUE(endpoint.operations().acceptPublishFrontier(g.interval,headers.data(),headers.size()));
        for(std::uint32_t at=0;at<endpoint.plan.count;++at){
            ASSERT_TRUE(endpoint.operations().makeReady(g.interval,endpoint.grants[at],acknowledgements[i][at]));
            ASSERT_TRUE(endpoint.operations().release(g.interval,endpoint.grants[at]));
        }
        ASSERT_TRUE(endpoint.allocation().retire(p::SourceRetirement::Consumed,source));
    }
    EXPECT_EQ(owner.control.next_slot_generation,3u);EXPECT_EQ(owner.slots[warm.slot].row,9u);
}

}
