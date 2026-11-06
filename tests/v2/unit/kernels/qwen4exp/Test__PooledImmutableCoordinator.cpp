/**
 * @file Test__PooledImmutableCoordinator.cpp
 * @brief Actual global coalescing, directed two-wave owner runtime and cancellation replay.
 *
 * These small CPU arrays are the actual common component state. Decoded source
 * ACKs and headers do not authenticate native transport, payload copies, source
 * service, GPU graph capture or serving eligibility. Every owner is independent
 * of the CPU consumer and independent cross-vendor coordinator declaration.
 */
#include "kernels/qwen4exp/PooledImmutableCoordinatorGeometry.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotTransportOps.h"
#include "execution/qwen4exp/PooledImmutableNativeTransfer.h"
#include "execution/qwen4exp/PooledImmutableLocalHotQueryBound.h"
#include "kernels/qwen4exp/PooledImmutableNativeTypes.h"
#include "execution/qwen4exp/PooledImmutableLocalHotTransportGeometry.h"
#include "execution/qwen4exp/PooledImmutableReplicatedReaderGeometry.h"
#include "kernels/qwen4exp/PooledImmutableCoordinatorOps.h"
#include "loaders/ModelLoader.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
namespace llaminar2::test {
namespace {
namespace a=qwen4exp::adaptive_lookup;
namespace p=a::pooled;
using PhysicalGeometry=qwen4exp::PooledImmutablePhysicalOwnerGeometry;
using Geometry=qwen4exp::PooledImmutableCoordinatorGeometry;
constexpr std::uint32_t K=16,N=64;
/** @return Explicit uneven mixed-vendor storage order, independent CPU consumer and independent coordinator. */
PhysicalGeometry declaration(std::span<const std::uint32_t> capacities)
{
    std::vector<PhysicalMemoryAllocatorIdentity> order;
    std::vector<ImmutableRowTableLayoutOwner> owners;
    bool positive=false;
    for(std::uint32_t i=0;i<capacities.size();++i){
        order.push_back({static_cast<int>(i),i%2?DeviceId::rocm(static_cast<int>(i)):DeviceId::cuda(static_cast<int>(i))});
        owners.push_back({{order.back(),ImmutableRowStorage::VRAM},0,capacities[i],{}});
        positive|=capacities[i]!=0;
    }
    const PhysicalMemoryAllocatorIdentity coordinator{101,DeviceId::rocm(11)};
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
    const auto layout=ImmutableRowTableLayoutGeometry::checked(
        ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,0x100000100ULL,90),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,owners,
        {{102,DeviceId::cpu()},coordinator},{},
        {{{103,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},
        {},positive?std::optional(coordinator):std::nullopt,K);
    return PhysicalGeometry::fromLayout(layout,order,coordinator);
}
/** @brief One real owner and its separately retained append-only banks. */
struct Physical
{
    std::vector<a::Slot64> slots;
    p::PhysicalOwnerControl control{};
    p::OwnerFrame frame{};
    p::AllocationFrame allocation_frame{};
    std::array<p::Grant,K> initial{},grants{};
    std::array<p::AllocationDecision,K> accepted{};
    std::array<std::uint64_t,K> a{},b{};
    std::array<std::uint32_t,256> counts{};
    p::OwnerView<a::Slot64> view;
    /** @brief Initialize the actual physical leaf once over exact declared owner capacity. */
    explicit Physical(p::OwnerParameters parameters):slots(parameters.capacity()),
        view{slots.data(),slots.empty()?nullptr:&control,&frame,701,parameters,initial.data()}
    {
        for(std::uint32_t i=0;i<slots.size();++i) operations().initializeSlot(i);
        if(!operations().initialize())throw std::logic_error("Owner setup failed");
    }
    /** @return Actual physical operations, never a shadow slot model. */
    p::OwnerOperations<a::Slot64> operations(){return p::OwnerOperations<a::Slot64>(view);}
    /** @return Actual typed two-wave owner protocol over retained exclusive roots. */
    p::OwnerPlanOperations<a::Slot64> allocation(){return {view,allocation_frame,{accepted.data(),grants.data(),{a.data(),b.data(),counts.data()},K}};}
};
/** @brief One actual coordinator and arbitrary actual owners with deterministic decoded source acknowledgements. */
struct Pool
{
    PhysicalGeometry physical;
    Geometry geometry;
    p::CoordinatorPolicyControl policy{};
    p::LookupFrame frame{};
    std::vector<p::DirectoryEntry> entries;
    std::vector<a::RadixNode> nodes;
    std::vector<std::uint32_t> epochs;
    std::vector<std::uint16_t> counts;
    std::array<p::CoordinatorUnique,K> unique{};
    std::array<p::CoordinatorOccurrence,N> sort_a{},sort_b{};
    std::array<std::uint32_t,N> permutation{};
    std::array<std::uint32_t,256> radix{};
    std::vector<p::CoordinatorOwnerPlan> owner_plans;
    std::array<p::AllocationDecision,K> decisions{};
    std::vector<std::unique_ptr<Physical>> owners;
    p::CoordinatorView view;
    std::uint64_t source_sequence=0;
    /** @brief Allocate only small CPU parser roots according to exact checked metadata. */
    explicit Pool(std::span<const std::uint32_t> capacities):physical(declaration(capacities)),
        geometry(Geometry::fromOwners(physical,{65536,1,2},N)),entries(geometry.parameters().entries()),
        nodes(entries.empty()?0:entries.size()-1),epochs(std::size_t{4}*geometry.parameters().laneWidth()),
        counts(epochs.size()),owner_plans(capacities.size()),
        view{entries.empty()?nullptr:&policy,entries.data(),nodes.data(),epochs.data(),counts.data(),
            &frame,unique.data(),sort_a.data(),sort_b.data(),radix.data(),permutation.data(),owner_plans.data(),
            decisions.data(),geometry.owners().data(),geometry.parameters(),K,N,
            static_cast<std::uint32_t>(capacities.size()),701,17}
    {
        for(std::uint32_t i=0;i<capacities.size();++i)owners.push_back(std::make_unique<Physical>(physical.ownerParameters(i)));
        if(!operations().initialize())throw std::logic_error("Coordinator setup failed");
    }
    /** @return Actual common production coordinator implementation. */
    p::CoordinatorOperations operations(){return p::CoordinatorOperations(view);}
    /** @brief Convert a protocol rejection into a concise fixture exception. */
    static void require(bool ok){if(!ok)throw std::logic_error("Actual pooled protocol rejected");}
    /** @brief Coalesce and acquire every actual directed initial pin and complete owner frontier. */
    void begin(std::span<const std::uint64_t> original,std::uint64_t admission=1,
        p::LookupPurpose purpose=p::LookupPurpose::CurrentDemand)
    {
        require(operations().begin(admission,original.data(),static_cast<std::uint32_t>(original.size()),purpose));
        for(auto &owner:owners)require(owner->operations().begin(frame.interval));
        for(std::uint32_t i=0;i<frame.unique_rows;++i){
            p::OwnerCommand command{};
            if(!operations().pinCommand(i,command)){require(frame.status==a::Status::Success);continue;}
            auto &owner=*owners[command.target.owner];
            const auto &t=command.target;
            require(operations().acceptPin(i,owner.operations().pinInitial(frame.interval,{t.row,t.generation,t.owner,t.slot},i,command.ordinal)));
        }
        std::vector<p::PinHeader> pins(owners.size());
        for(std::size_t i=0;i<owners.size();++i)pins[i]=owners[i]->operations().sealPins(frame.interval);
        for(auto &owner:owners)require(owner->operations().acceptPinFrontier(frame.interval,pins.data(),pins.size()));
        require(operations().sealPins(pins.data(),pins.size()));
    }
    /** @brief Obtain original owner leases and construct only the canonical directed free plans. */
    void planFree()
    {
        std::vector<p::AvailabilityLease> leases;
        for(auto &owner:owners){require(owner->allocation().acquire(frame.interval));leases.push_back(owner->allocation_frame.lease);}
        require(operations().planFree(leases.data(),leases.size()));
    }
    /** @brief Execute/drain actual free waves and acquire their exact complete all-owner receipts. */
    void freeWave(std::uint32_t cancelled=a::None,bool acceptCancelled=false)
    {
        planFree();
        std::vector<p::FreeAllocationReceipt> receipts(owners.size());
        for(std::uint32_t i=0;i<owners.size();++i){
            auto &owner=*owners[i];const auto plan=operations().freePlan(i);
            if(i==cancelled){
                if(acceptCancelled)require(owner.allocation().accept(plan));
                require(owner.allocation().cancel(p::SourceRetirement::Unpublished));
            }else{require(owner.allocation().accept(plan));require(owner.allocation().executeFree());}
            for(std::uint32_t at=0;at<owner.allocation_frame.count;++at)
                require(operations().acceptReservation(owner.grants[at].unique,owner.grants[at]));
            receipts[i]=owner.allocation().freeReceipt();
        }
        require(operations().completeFree(receipts.data(),receipts.size()));
        for(auto &owner:owners)if(owner->allocation_frame.phase!=p::AllocationPhase::Cancelled)
            require(owner->allocation().acceptFreeFrontier(receipts.data(),receipts.size()));
    }
    /** @brief Execute/drain actual victim waves, preserving all original free/grant prefixes. */
    void victimWave(std::uint32_t cancelled=a::None,bool acceptCancelled=false)
    {
        require(operations().planVictims());
        std::vector<p::VictimAllocationReceipt> receipts(owners.size());
        for(std::uint32_t i=0;i<owners.size();++i){
            auto &owner=*owners[i];const auto plan=operations().victimPlan(i);
            if(owner.allocation_frame.phase!=p::AllocationPhase::Cancelled){
                if(i==cancelled){
                    if(acceptCancelled)require(owner.allocation().accept(plan));
                    require(owner.allocation().cancel(p::SourceRetirement::Unpublished));
                }else{require(owner.allocation().accept(plan));require(owner.allocation().executeVictims());}
            }
            for(std::uint32_t at=owner.allocation_frame.free_count;at<owner.allocation_frame.count;++at)
                require(operations().acceptReservation(owner.grants[at].unique,owner.grants[at]));
            receipts[i]=owner.allocation().victimReceipt();
        }
        require(operations().completeVictims(receipts.data(),receipts.size()));
    }
    /** @brief Run actual source seal/fill/ACK/Ready/publication/release/retirement parser operations. */
    void finish()
    {
        bool misses=false;for(std::uint32_t i=0;i<frame.unique_rows;++i)misses|=unique[i].phase!=p::UniquePhase::Read;
        const p::SourceWindow source=misses?p::SourceWindow{701,17,++source_sequence}:p::SourceWindow{};
        const auto retirement=misses?p::SourceRetirement::Consumed:p::SourceRetirement::Unpublished;
        require(operations().sealSource(source));
        std::vector<p::PublishHeader> published(owners.size());
        std::vector<std::array<p::FillAcknowledgement,K>> acknowledgements(owners.size());
        for(std::size_t i=0;i<owners.size();++i){
            auto &owner=*owners[i];require(owner.allocation().sealSource(source));
            for(std::uint32_t at=0;at<owner.allocation_frame.count;++at)
                acknowledgements[i][at]={source,owner.grants[at].destination_token};
            published[i]=owner.allocation().validateFills(acknowledgements[i].data(),owner.allocation_frame.count);
            require(published[i].status==a::Status::Success);
        }
        std::vector<p::ReadyHeader> ready(owners.size());
        for(std::size_t i=0;i<owners.size();++i){
            auto &owner=*owners[i];require(owner.operations().acceptPublishFrontier(frame.interval,published.data(),published.size()));
            for(std::uint32_t at=0;at<owner.allocation_frame.count;++at)
                require(owner.operations().makeReady(frame.interval,owner.grants[at],acknowledgements[i][at]));
            ready[i]=owner.operations().readyHeader(frame.interval);
        }
        require(operations().publish(ready.data(),ready.size(),retirement,source));
        std::vector<p::RetiredHeader> retired(owners.size());
        for(std::size_t i=0;i<owners.size();++i){
            auto &owner=*owners[i];
            for(std::uint32_t at=0;at<owner.frame.initial_pin_count;++at)require(owner.operations().release(frame.interval,owner.initial[at]));
            for(std::uint32_t at=0;at<owner.allocation_frame.count;++at)require(owner.operations().release(frame.interval,owner.grants[at]));
            require(owner.allocation().retire(retirement,source));
            retired[i]=owner.operations().retiredHeader(frame.interval);
        }
        require(operations().finish(retired.data(),retired.size()));
    }
    /** @brief Retire actual cancellation using complete immutable actual owner release receipts. */
    void cancel()
    {
        std::vector<p::ReleaseReceipt> releases;
        for(auto &pointer:owners){
            auto &owner=*pointer;
            if(owner.allocation_frame.phase!=p::AllocationPhase::Cancelled)
                require(owner.allocation().cancel(p::SourceRetirement::Unpublished));
            for(std::uint32_t at=0;at<owner.frame.initial_pin_count;++at)
                releases.push_back(owner.operations().releaseWithReceipt(frame.interval,owner.initial[at]));
            for(std::uint32_t at=0;at<owner.allocation_frame.count;++at)
                releases.push_back(owner.operations().releaseWithReceipt(frame.interval,owner.grants[at]));
        }
        std::sort(releases.begin(),releases.end(),[](const auto &x,const auto &y){return x.original.unique<y.original.unique;});
        require(operations().cancel(p::SourceRetirement::Unpublished,{},releases.data(),releases.size()));
        std::vector<p::RetiredHeader> retired(owners.size());
        for(std::size_t i=0;i<owners.size();++i){
            require(owners[i]->allocation().retire(p::SourceRetirement::Unpublished));
            retired[i]=owners[i]->operations().retiredHeader(frame.interval);
        }
        require(operations().finish(retired.data(),retired.size()));
    }
    /** @brief Cancel after an actual source/ACK frontier and an explicit prefix of actual Ready publications. */
    void cancelSourcePrefix(std::uint32_t readyCount)
    {
        const p::SourceWindow source{701,17,++source_sequence};
        require(operations().sealSource(source));
        std::vector<p::PublishHeader> headers(owners.size());
        for(std::size_t i=0;i<owners.size();++i){
            auto &owner=*owners[i];require(owner.allocation().sealSource(source));
            std::array<p::FillAcknowledgement,K> acknowledgements{};
            for(std::uint32_t at=0;at<owner.allocation_frame.count;++at)
                acknowledgements[at]={source,owner.grants[at].destination_token};
            headers[i]=owner.allocation().validateFills(acknowledgements.data(),owner.allocation_frame.count);
            require(headers[i].status==a::Status::Success);
        }
        for(auto &owner:owners)require(owner->operations().acceptPublishFrontier(frame.interval,headers.data(),headers.size()));
        for(std::uint32_t i=0;i<readyCount;++i){
            const auto &grant=unique[i].grant;
            require(owners[grant.owner]->operations().makeReady(frame.interval,grant,{source,grant.destination_token}));
        }
        std::vector<p::ReleaseReceipt> releases;
        for(auto &pointer:owners){
            auto &owner=*pointer;require(owner.allocation().cancel(p::SourceRetirement::Consumed,source));
            for(std::uint32_t at=0;at<owner.frame.initial_pin_count;++at)
                releases.push_back(owner.operations().releaseWithReceipt(frame.interval,owner.initial[at]));
            for(std::uint32_t at=0;at<owner.allocation_frame.count;++at)
                releases.push_back(owner.operations().releaseWithReceipt(frame.interval,owner.grants[at]));
        }
        std::sort(releases.begin(),releases.end(),[](const auto &x,const auto &y){return x.original.unique<y.original.unique;});
        require(operations().cancel(p::SourceRetirement::Consumed,source,releases.data(),releases.size()));
        std::vector<p::RetiredHeader> retired(owners.size());
        for(std::size_t i=0;i<owners.size();++i){
            require(owners[i]->allocation().retire(p::SourceRetirement::Consumed,source));
            retired[i]=owners[i]->operations().retiredHeader(frame.interval);
        }
        require(operations().finish(retired.data(),retired.size()));
    }
    /** @brief Execute one complete actual owner runtime frame. */
    void run(std::span<const std::uint64_t> original,std::uint64_t admission=1)
    {begin(original,admission);freeWave();victimWave();finish();}
};
}
/** @brief Object-local layout proof cannot hide subsequent mutable authority changes. */
TEST(PooledImmutableCoordinator, LayoutBorrowerReplayMutableAuthorityAndFreshBindings)
{
    for(unsigned defect=0;defect<5;++defect){
        Pool pool(std::array{2u});const std::array<std::uint64_t,1> row{17};
        ASSERT_NO_THROW(pool.run(row));auto op=pool.operations();
        ASSERT_TRUE(op.begin(2,row.data(),row.size()));
        if(defect==0)pool.frame.phase=p::LookupPhase::Idle;
        if(defect==1)pool.frame.status=a::Status::Invariant;
        if(defect==2)++pool.frame.source_storage_generation;
        if(defect==3)++pool.policy.aging_accesses;
        if(defect==4)pool.policy.flags=0;
        p::OwnerCommand command{};EXPECT_FALSE(op.pinCommand(0,command));
    }
    Pool original(std::array{2u}),fresh(std::array{2u});
    auto op=original.operations();auto rebound=fresh.operations();
    original.view.sort_a=original.view.sort_b; // Caller declaration cannot rebind this existing borrower.
    ASSERT_TRUE(op.begin<std::uint64_t>(1,nullptr,0));
    EXPECT_FALSE(original.operations().begin<std::uint64_t>(2,nullptr,0));
    ASSERT_TRUE(rebound.begin<std::uint64_t>(1,nullptr,0));
    EXPECT_EQ(fresh.frame.status,a::Status::Success);
}
/** @brief An invalid copied declaration remains rejected even if the caller repairs its own copy. */
TEST(PooledImmutableCoordinator, LayoutBorrowerReplayMalformedRoots)
{
    for(unsigned defect=0;defect<4;++defect){
        Pool pool(std::array{2u});auto view=pool.view;
        if(defect==0)view.sort_a=view.sort_b;
        if(defect==1)view.unique_capacity=K-1;
        if(defect==2)view.frame=nullptr;
        if(defect==3)view.source_storage_generation=0;
        p::CoordinatorOperations op(view);view=pool.view;
        const auto scratch=pool.sort_a;
        EXPECT_FALSE(op.begin<std::uint64_t>(1,nullptr,0));
        EXPECT_EQ(std::memcmp(scratch.data(),pool.sort_a.data(),sizeof(scratch)),0);
    }
}
/** @brief Actual pin issuance preserves a successful prefix and rejects the first repeated command. */
TEST(PooledImmutableCoordinator, LayoutBorrowerReplayPrefixAndRepeatedReset)
{
    Pool pool(std::array{2u});const std::array<std::uint64_t,2> rows{17,19};
    ASSERT_NO_THROW(pool.run(rows));
    for(unsigned replay=0;replay<3;++replay)ASSERT_NO_THROW(pool.run(rows,2+replay));
    auto op=pool.operations();ASSERT_TRUE(op.begin(5,rows.data(),rows.size()));
    p::OwnerCommand first{},second{};
    ASSERT_TRUE(op.pinCommand(0,first));ASSERT_TRUE(op.pinCommand(1,second));
    const auto outstanding=pool.frame.outstanding;const auto ordinal=pool.frame.command_ordinal;
    EXPECT_FALSE(op.pinCommand(0,first));
    EXPECT_EQ(pool.frame.outstanding,outstanding);EXPECT_EQ(pool.frame.command_ordinal,ordinal);
    EXPECT_EQ(pool.unique[0].phase,p::UniquePhase::PinPending);
    EXPECT_EQ(pool.unique[1].phase,p::UniquePhase::PinPending);
}
TEST(PooledImmutableCoordinator, StableGlobalCoalescingPreservesEveryOriginalPositionAndHighRowBits)
{
    Pool pool(std::array{1u,0u,2u,0u,1u});
    const std::array<std::uint64_t,7> original{0x100000007ULL,9,0x100000007ULL,3,9,3,0x100000007ULL};
    ASSERT_NO_THROW(pool.begin(original));
    ASSERT_EQ(pool.frame.unique_rows,3u);
    EXPECT_EQ(pool.unique[0].row,original[0]);EXPECT_EQ(pool.unique[1].row,9u);EXPECT_EQ(pool.unique[2].row,3u);
    const std::array<std::uint32_t,7> expected{0,1,0,2,1,2,0};
    EXPECT_TRUE(std::equal(expected.begin(),expected.end(),pool.permutation.begin()));
    ASSERT_NO_THROW(pool.freeWave());
    ASSERT_NO_THROW(pool.victimWave());
    ASSERT_NO_THROW(pool.finish());
    EXPECT_EQ(pool.policy.logical_clock,original.size());
}
TEST(PooledImmutableCoordinator, ActualRuntimeAcrossArbitraryOwnerCountsMixedVendorsC0AndIndependentCPUConsumer)
{
    for(const auto capacities:std::vector<std::vector<std::uint32_t>>{{0},{1},{1,0,2},{1,0,2,0,1},{0,0,0,0,0}}){
        Pool pool(capacities);
        const std::array<std::uint64_t,5> original{17,3,17,9,3};
        ASSERT_NO_THROW(pool.run(original));
        EXPECT_EQ(pool.frame.phase,p::LookupPhase::Idle);
        EXPECT_EQ(pool.frame.ordinal,1u);
        EXPECT_FALSE(pool.physical.requesterCoordinate());
        EXPECT_EQ(pool.physical.coordinator(),(PhysicalMemoryAllocatorIdentity{101,DeviceId::rocm(11)}));
        for(const auto &owner:pool.owners)EXPECT_EQ(owner->frame.phase,p::OwnerPhase::Idle);
    }
}
TEST(PooledImmutableCoordinator, FullFreeFrontierPrecedesVictimsAndAllHitResetRetainsGenerations)
{
    Pool pool(std::array{1u,0u,2u});
    ASSERT_NO_THROW(pool.run(std::array<std::uint64_t,3>{101,102,103}));
    const std::array<std::uint64_t,10> hot{900,900,900,900,900,900,900,900,900,900};
    ASSERT_NO_THROW(pool.begin(hot));
    ASSERT_NO_THROW(pool.freeWave());
    EXPECT_TRUE(pool.frame.free_exhausted);
    ASSERT_NO_THROW(pool.victimWave());ASSERT_EQ(pool.frame.victim_planned,1u);
    const auto actual=pool.unique[0].grant;
    ASSERT_EQ(actual.kind,p::GrantKind::Reservation);
    ASSERT_NO_THROW(pool.finish());
    for(unsigned replay=0;replay<3;++replay){
        ASSERT_NO_THROW(pool.run(hot,2+replay));
        EXPECT_EQ(pool.unique[0].phase,p::UniquePhase::Read);
        EXPECT_EQ(pool.unique[0].grant.generation,actual.generation);
        EXPECT_EQ(pool.unique[0].grant.slot,actual.slot);
    }
}
TEST(PooledImmutableCoordinator, FreePlanAndGrantPrefixesRemainImmutableAcrossSecondWaveAndSource)
{
    Pool pool(std::array{2u,0u,2u});
    ASSERT_NO_THROW(pool.run(std::array<std::uint64_t,2>{101,102}));
    const std::array<std::uint64_t,14> original{700,701,900,900,900,900,900,900,900,900,900,900,900,900};
    ASSERT_NO_THROW(pool.begin(original));
    ASSERT_NO_THROW(pool.freeWave());
    const auto free=pool.frame.free_planned;ASSERT_EQ(free,2u);
    const auto decisions=pool.decisions;const auto owner_plans=pool.owner_plans;
    const auto unique0=pool.unique[0].grant;const auto unique1=pool.unique[1].grant;
    ASSERT_NO_THROW(pool.victimWave());ASSERT_EQ(pool.frame.victim_planned,1u);
    EXPECT_EQ(std::memcmp(decisions.data(),pool.decisions.data(),free*sizeof(p::AllocationDecision)),0);
    for(std::size_t i=0;i<pool.owner_plans.size();++i){
        EXPECT_EQ(pool.owner_plans[i].free_offset,owner_plans[i].free_offset);
        EXPECT_EQ(pool.owner_plans[i].free_count,owner_plans[i].free_count);
        EXPECT_EQ(pool.owner_plans[i].free_received,owner_plans[i].free_received);
    }
    EXPECT_EQ(std::memcmp(&unique0,&pool.unique[0].grant,sizeof(unique0)),0);
    EXPECT_EQ(std::memcmp(&unique1,&pool.unique[1].grant,sizeof(unique1)),0);
    ASSERT_NO_THROW(pool.finish());
}
TEST(PooledImmutableCoordinator, ActualVictimCompletionDistinguishesUnacceptedAndAcceptedCancellation)
{
    for(bool accepted:{false,true}){
        Pool pool(std::array{1u,0u,2u});
        ASSERT_NO_THROW(pool.run(std::array<std::uint64_t,3>{101,102,103}));
        const auto old=pool.owners[0]->slots[0];
        const std::array<std::uint64_t,10> hot{900,900,900,900,900,900,900,900,900,900};
        ASSERT_NO_THROW(pool.begin(hot));
    ASSERT_NO_THROW(pool.freeWave());
        ASSERT_NO_THROW(pool.victimWave(0,accepted));
        ASSERT_TRUE(pool.frame.cancelled_completion);
        EXPECT_EQ(pool.unique[0].phase,accepted?p::UniquePhase::VictimGranted:p::UniquePhase::SourceOnly);
        ASSERT_NO_THROW(pool.cancel());
        EXPECT_EQ(pool.owners[0]->slots[0].row,accepted?900u:old.row);
        EXPECT_EQ(pool.owners[0]->slots[0].generation,accepted?old.generation+1:old.generation);
        EXPECT_EQ(a::PhysicalSlot<a::Slot64>::state(pool.owners[0]->slots[0]),accepted?a::SlotState::Empty:a::SlotState::Ready);
        ASSERT_NO_THROW(pool.run(std::array<std::uint64_t,1>{accepted?102u:101u},2));
    }
}
TEST(PooledImmutableCoordinator, CancelledFreeCompletionClosesUnacceptedPlansWithoutInventedGrants)
{
    for(bool accepted:{false,true}){
        Pool pool(std::array{1u,0u,2u});
        ASSERT_NO_THROW(pool.begin(std::array<std::uint64_t,3>{17,18,19}));
        ASSERT_NO_THROW(pool.freeWave(0,accepted));
        EXPECT_FALSE(pool.frame.free_exhausted);ASSERT_TRUE(pool.frame.cancelled_completion);
        ASSERT_NO_THROW(pool.victimWave());EXPECT_EQ(pool.frame.victim_planned,0u);
        ASSERT_NO_THROW(pool.cancel());
        EXPECT_EQ(pool.owners[0]->control.next_slot_generation,accepted?1u:0u);
        EXPECT_EQ(pool.owners[0]->control.recycled_free,accepted?1u:0u);
    }
}
TEST(PooledImmutableCoordinator, U0FramesAdvanceLookupIdentityWithoutIssuingSourceTransactions)
{
    Pool pool(std::array{0u,0u,0u,0u,0u});
    for(std::uint64_t admission=1;admission<=4;++admission){
        ASSERT_NO_THROW(pool.run({},admission));
        EXPECT_EQ(pool.frame.ordinal,admission);EXPECT_EQ(pool.source_sequence,0u);
    }
}
TEST(PooledImmutableCoordinator, MandatoryBOMPricesExactIndependentNAndKAndArbitraryPWithoutPhysicalDuplication)
{
    for(const auto capacities:std::vector<std::vector<std::uint32_t>>{{0},{1,0,2,0,1},{0,0,0,0,0}}){
        Pool pool(capacities);
        const PhysicalMemoryResource resource{101,DeviceId::rocm(11),1ULL<<30,1ULL<<30};
        const auto mandatory=pool.geometry.frameOffsets();const auto policy=pool.geometry.policyOffsets();
        EXPECT_EQ(pool.geometry.mandatoryBOM(resource).totalBytes(),mandatory.bytes);
        EXPECT_EQ(pool.geometry.policyBOM(resource).totalBytes(),policy.bytes);
        const std::array ranges{
            std::pair{mandatory.control,sizeof(p::LookupFrame)},
            std::pair{mandatory.unique,K*sizeof(p::CoordinatorUnique)},
            std::pair{mandatory.sort_a,N*sizeof(p::CoordinatorOccurrence)},
            std::pair{mandatory.sort_b,N*sizeof(p::CoordinatorOccurrence)},
            std::pair{mandatory.radix_counts,256*sizeof(std::uint32_t)},
            std::pair{mandatory.original_to_unique,N*sizeof(std::uint32_t)},
            std::pair{mandatory.owner_plans,capacities.size()*sizeof(p::CoordinatorOwnerPlan)},
            std::pair{mandatory.decisions,K*sizeof(p::AllocationDecision)},
            std::pair{mandatory.owners,capacities.size()*sizeof(p::CoordinatorOwnerGeometry)}};
        for(std::size_t at=0;at<ranges.size();++at){
            EXPECT_EQ(ranges[at].first%64,0u);
            EXPECT_LE(ranges[at].first+ranges[at].second,mandatory.bytes);
            for(std::size_t prior=0;prior<at;++prior)
                EXPECT_TRUE(ranges[at].first>=ranges[prior].first+ranges[prior].second ||
                    ranges[prior].first>=ranges[at].first+ranges[at].second);
        }
        const auto smaller=Geometry::fromOwners(pool.physical,{65536,1,2},N/2);
        EXPECT_EQ(mandatory.bytes-smaller.frameOffsets().bytes,(N/2)*(2*sizeof(p::CoordinatorOccurrence)+sizeof(std::uint32_t)));
        auto foreign=resource;foreign.world_rank=0;
        EXPECT_THROW(pool.geometry.policyBOM(foreign),std::invalid_argument);
        EXPECT_THROW(pool.geometry.mandatoryBOM(foreign),std::invalid_argument);
        if(pool.entries.empty())EXPECT_EQ(policy.bytes,0u);
    }
}
TEST(PooledImmutableCoordinator, InvalidOriginalBoundsAndAliasedOrIncompleteRootsFailBeforeScratchWrites)
{
    {
        Pool pool(std::array{1u,0u,2u});
        EXPECT_THROW(Geometry::fromOwners(pool.physical,{65536,1,2},K-1),std::invalid_argument);
        EXPECT_THROW(Geometry::fromOwners(pool.physical,{0,1,2},N),std::invalid_argument);
        EXPECT_THROW(Geometry::fromOwners(pool.physical,{65536,2,1},N),std::invalid_argument);
    }
    for(int bad=0;bad<4;++bad){
        Pool pool(std::array{1u,0u,2u});auto view=pool.view;
        if(bad==0)view.sort_a=view.sort_b;
        if(bad==1)view.unique_capacity=K-1;
        if(bad==2)view.frame=nullptr;
        if(bad==3)view.owner_capacity=2;
        const auto before=pool.sort_a;
        EXPECT_FALSE(p::CoordinatorOperations(view).begin<std::uint64_t>(1,nullptr,0));
        EXPECT_EQ(std::memcmp(before.data(),pool.sort_a.data(),sizeof(before)),0);
    }
    {
        Pool pool(std::array{1u,0u,2u});const auto before=pool.sort_a;
        pool.unique[0].row=7;
        EXPECT_FALSE(pool.operations().begin(1,&pool.unique[0].row,1));
        EXPECT_EQ(std::memcmp(before.data(),pool.sort_a.data(),sizeof(before)),0);
        EXPECT_EQ(pool.frame.ordinal,0u);
    }
}
TEST(PooledImmutableCoordinator, OriginalAndDistinctBoundsRejectIncompleteTraceWithoutPhysicalAllocation)
{
    for(bool distinct:{false,true}){
        Pool pool(std::array{1u,0u,2u});
        std::vector<std::uint64_t> rows(distinct?K+1:N+1,7);
        if(distinct)for(std::size_t i=0;i<rows.size();++i)rows[i]=i;
        EXPECT_FALSE(pool.operations().begin(1,rows.data(),rows.size()));
        for(const auto &owner:pool.owners)EXPECT_EQ(owner->control.next_slot_generation,0u);
    }
}
TEST(PooledImmutableCoordinator, ActualUnusedFreeQuotaPreservesWarmVictimAndForcesZeroVictimPlan)
{
    Pool pool(std::array{1u,0u,2u});
    ASSERT_NO_THROW(pool.run(std::array<std::uint64_t,1>{101}));
    const auto generation=pool.owners[0]->slots[0].generation;
    ASSERT_NO_THROW(pool.begin(std::array<std::uint64_t,3>{900,900,900}));
    ASSERT_NO_THROW(pool.freeWave());
    ASSERT_FALSE(pool.frame.free_exhausted);EXPECT_EQ(pool.unique[0].grant.owner,2u);
    ASSERT_NO_THROW(pool.victimWave());EXPECT_EQ(pool.frame.victim_planned,0u);
    EXPECT_EQ(pool.owners[0]->slots[0].row,101u);EXPECT_EQ(pool.owners[0]->slots[0].generation,generation);
    ASSERT_NO_THROW(pool.finish());
}
TEST(PooledImmutableCoordinator, MalformedFreeCompletionFrontierRejectsBeforeAnyCoordinatorPhaseCommit)
{
    for(int bad=0;bad<5;++bad){
        Pool pool(std::array{1u,0u,2u});
        ASSERT_NO_THROW(pool.begin(std::array<std::uint64_t,3>{17,18,19}));
        ASSERT_NO_THROW(pool.planFree());
        std::vector<p::FreeAllocationReceipt> receipts(pool.owners.size());
        for(std::uint32_t i=0;i<pool.owners.size();++i){
            auto &owner=*pool.owners[i];ASSERT_TRUE(owner.allocation().accept(pool.operations().freePlan(i)));
            ASSERT_TRUE(owner.allocation().executeFree());
            for(std::uint32_t at=0;at<owner.allocation_frame.count;++at)
                ASSERT_TRUE(pool.operations().acceptReservation(owner.grants[at].unique,owner.grants[at]));
            receipts[i]=owner.allocation().freeReceipt();
        }
        if(bad==0)receipts[2].lease.owner=0;
        if(bad==1)++receipts[2].lease.interval.sequence;
        if(bad==2)--receipts[2].granted;
        if(bad==3)++receipts[2].lease.free_slots;
        const auto plans=pool.decisions;const auto unique=pool.unique;
        EXPECT_FALSE(pool.operations().completeFree(receipts.data(),bad==4?2:3));
        EXPECT_EQ(pool.frame.phase,p::LookupPhase::FreePlanned);
        EXPECT_FALSE(pool.frame.queues_applied);
        EXPECT_EQ(std::memcmp(plans.data(),pool.decisions.data(),sizeof(plans)),0);
        EXPECT_EQ(std::memcmp(unique.data(),pool.unique.data(),sizeof(unique)),0);
    }
}
TEST(PooledImmutableCoordinator, MissingOrForgedActualVictimCompletionCannotAuthorizeSourcePublication)
{
    for(bool count:{false,true}){
        Pool pool(std::array{1u,0u,2u});
        ASSERT_NO_THROW(pool.run(std::array<std::uint64_t,3>{101,102,103}));
        const std::array<std::uint64_t,10> hot{900,900,900,900,900,900,900,900,900,900};
        ASSERT_NO_THROW(pool.begin(hot));
        ASSERT_NO_THROW(pool.freeWave());
        ASSERT_TRUE(pool.operations().planVictims());ASSERT_EQ(pool.frame.victim_planned,1u);
        std::vector<p::VictimAllocationReceipt> receipts(pool.owners.size());
        for(std::uint32_t i=0;i<pool.owners.size();++i){
            auto &owner=*pool.owners[i];ASSERT_TRUE(owner.allocation().accept(pool.operations().victimPlan(i)));
            ASSERT_TRUE(owner.allocation().executeVictims());
            for(std::uint32_t at=owner.allocation_frame.free_count;at<owner.allocation_frame.count;++at)
                ASSERT_TRUE(pool.operations().acceptReservation(owner.grants[at].unique,owner.grants[at]));
            receipts[i]=owner.allocation().victimReceipt();
        }
        if(!count)receipts[0].granted=0;
        EXPECT_FALSE(pool.operations().completeVictims(receipts.data(),count?2:3));
        EXPECT_EQ(pool.frame.phase,p::LookupPhase::VictimPlanned);
        EXPECT_FALSE(pool.frame.queues_applied);
        EXPECT_EQ(pool.entries[pool.unique[0].entry].present,0u);
    }
}
TEST(PooledImmutableCoordinator, ActualSourceCancellationRetainsOnlyReadyRowsBeforeDirectoryPublication)
{
    for(const auto ready:std::array{0u,1u,3u}){
        Pool pool(std::array{1u,0u,2u});
        const std::array<std::uint64_t,3> original{17,18,19};
        ASSERT_NO_THROW(pool.begin(original));
        ASSERT_NO_THROW(pool.freeWave());
        ASSERT_NO_THROW(pool.victimWave());
        ASSERT_NO_THROW(pool.cancelSourcePrefix(ready));
        for(std::uint32_t i=0;i<original.size();++i)
            EXPECT_EQ(pool.operations().find(original[i])!=a::None,i<ready);
        ASSERT_NO_THROW(pool.begin(original,2));
        std::uint32_t reads=0;
        for(std::uint32_t i=0;i<pool.frame.unique_rows;++i)reads+=pool.unique[i].phase==p::UniquePhase::Read;
        EXPECT_EQ(reads,ready);
        ASSERT_NO_THROW(pool.freeWave());
        ASSERT_NO_THROW(pool.victimWave());
        ASSERT_NO_THROW(pool.finish());
    }
}

TEST(PooledImmutableCoordinator, LiveCountsKeepCoalescingAndPlanTailsDirtyAcrossLargeToSmallReplay)
{
    Pool pool(std::array{1u,0u,2u,0u,1u});
    std::array<std::uint64_t,N> original{};
    for(std::uint32_t i=0;i<N;++i)original[i]=0x100000000ULL+(i%K);
    ASSERT_NO_THROW(pool.run(original));
    const auto permutation=pool.permutation;const auto a=pool.sort_a;const auto b=pool.sort_b;
    const auto unique=pool.unique;const auto decisions=pool.decisions;
    const auto clock=pool.policy.logical_clock;
    ASSERT_NO_THROW(pool.begin(std::array<std::uint64_t,2>{original[0],original[0]},2));
    ASSERT_EQ(pool.frame.original_ids,2u);ASSERT_EQ(pool.frame.unique_rows,1u);
    for(std::uint32_t at=2;at<N;++at){
        EXPECT_EQ(pool.permutation[at],permutation[at]);
        EXPECT_EQ(std::memcmp(&pool.sort_a[at],&a[at],sizeof(a[at])),0);
        EXPECT_EQ(std::memcmp(&pool.sort_b[at],&b[at],sizeof(b[at])),0);
    }
    for(std::uint32_t at=1;at<K;++at)EXPECT_EQ(std::memcmp(&pool.unique[at],&unique[at],sizeof(unique[at])),0);
    ASSERT_NO_THROW(pool.freeWave());
    ASSERT_NO_THROW(pool.victimWave());
    EXPECT_EQ(std::memcmp(pool.decisions.data(),decisions.data(),sizeof(decisions)),0);
    ASSERT_NO_THROW(pool.finish());
    EXPECT_EQ(pool.policy.logical_clock,clock+2);
}

TEST(PooledImmutableCoordinator, SourcePolicyAdmissionAndOrdinalReplayIdentityCannotChangeAcrossReset)
{
    for(int bad=0;bad<5;++bad){
        Pool pool(std::array{1u,0u,2u});
        ASSERT_NO_THROW(pool.run(std::array<std::uint64_t,1>{17},2));
        auto view=pool.view;
        if(bad==0)++view.source_storage_generation;
        if(bad==1)++view.table;
        if(bad==2)view.parameters=Geometry::fromOwners(pool.physical,{65535,1,2},N).parameters();
        if(bad==3)view.parameters=Geometry::fromOwners(pool.physical,{65536,0,2},N).parameters();
        if(bad==4)pool.frame.ordinal=~std::uint64_t{0};
        const auto ordinal=pool.frame.ordinal;const auto scratch=pool.sort_a;
        const std::uint64_t row=17;
        EXPECT_FALSE(p::CoordinatorOperations(view).begin(2,&row,1));
        EXPECT_EQ(pool.frame.ordinal,ordinal);
        EXPECT_EQ(std::memcmp(scratch.data(),pool.sort_a.data(),sizeof(scratch)),0);
    }
    {
        Pool pool(std::array{1u,0u,2u});
        ASSERT_NO_THROW(pool.run(std::array<std::uint64_t,1>{17},2));
        const std::uint64_t row=17;
        EXPECT_FALSE(pool.operations().begin(1,&row,1));
        EXPECT_EQ(pool.frame.ordinal,1u);
    }
}

TEST(PooledImmutableCoordinator, UInt32OriginalRowsUseTheSameStableGlobalCoalescingContract)
{
    Pool pool(std::array{1u,0u,2u,0u,1u});
    const std::array<std::uint32_t,6> original{99,7,99,3,7,3};
    ASSERT_TRUE(pool.operations().begin(1,original.data(),original.size()));
    ASSERT_EQ(pool.frame.unique_rows,3u);
    EXPECT_EQ(pool.unique[0].row,99u);EXPECT_EQ(pool.unique[1].row,7u);EXPECT_EQ(pool.unique[2].row,3u);
    const std::array<std::uint32_t,6> expected{0,1,0,2,1,2};
    EXPECT_TRUE(std::equal(expected.begin(),expected.end(),pool.permutation.begin()));
}

}

namespace llaminar2::test {
/** @brief Replay exact result/ACK state through the actual shared device/CPU metadata operations. */
TEST(PooledImmutableCoordinator, ReplicatedReadersResultACKLiveExtentReplay)
{
    namespace rr = qwen4exp::replicated_reader;
    namespace state = qwen4exp::adaptive_lookup::pooled;
    constexpr rr::Geometry geometry{1, 2, 256, 64};
    rr::Control control{};
    int error = 0;
    state::LookupFrame frame{};
    frame.interval = {11, 7, 1};
    frame.phase = state::LookupPhase::Ready;
    frame.status = qwen4exp::adaptive_lookup::Status::Success;
    for (const auto count : {256u, 16u, 0u, 64u}) {
        frame.original_ids = count;
        ASSERT_TRUE(rr::prepare(frame, geometry, 1, &error, control.incoming));
        ASSERT_TRUE(rr::accept(control.incoming, sizeof(rr::Header), geometry, 1, frame.interval.table,
            count, &error, control.outgoing));
        ASSERT_TRUE(rr::acknowledge(control.outgoing, rr::outputBytes(geometry, count), geometry, &error, CapturedTransferStatus::Ready));
        ASSERT_TRUE(rr::confirm(control.outgoing, sizeof(rr::Header), frame, geometry, 1, &error));
        const auto retired = control.outgoing;
        ++frame.interval.sequence;
        EXPECT_FALSE(rr::confirm(retired, sizeof(rr::Header), frame, geometry, 1, &error));
    }
    EXPECT_EQ(error, 0);
}
/** @brief Missing, partial, foreign and failed replies cannot become last-reader evidence. */
TEST(PooledImmutableCoordinator, ReplicatedReadersMissingCancelledAndStickyErrorReplay)
{
    namespace rr = qwen4exp::replicated_reader;
    namespace state = qwen4exp::adaptive_lookup::pooled;
    constexpr rr::Geometry geometry{1, 2, 256, 64};
    int error = 0;
    state::LookupFrame frame{};
    frame.interval = {11, 7, 1};
    frame.original_ids = 16;
    frame.phase = state::LookupPhase::Ready;
    frame.status = qwen4exp::adaptive_lookup::Status::Success;
    rr::Header original{}, reply{};
    ASSERT_TRUE(rr::prepare(frame, geometry, 1, &error, original));
    for (const auto bytes : {0u, 47u, 49u}) {
        EXPECT_FALSE(rr::accept(original, bytes, geometry, 1, frame.interval.table, 16, &error, reply));
        EXPECT_FALSE(rr::confirm(reply, sizeof(reply), frame, geometry, 1, &error));
    }
    EXPECT_FALSE(rr::accept(original, sizeof(original), geometry, 1, frame.interval.table + 1, 16, &error, reply));
    EXPECT_FALSE(rr::accept(original, sizeof(original), geometry, 1, frame.interval.table, 32, &error, reply));
    ASSERT_TRUE(rr::accept(original, sizeof(original), geometry, 1, frame.interval.table, 16, &error, reply));
    EXPECT_FALSE(rr::acknowledge(reply, 0, geometry, &error, CapturedTransferStatus::Ready));
    EXPECT_FALSE(rr::confirm(reply, sizeof(reply), frame, geometry, 1, &error));
    error = static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure);
    EXPECT_FALSE(rr::prepare(frame, geometry, 1, &error, reply));
    EXPECT_EQ(error, static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
    error = 0; // Only the original request-reset owner clears first error.
    ++frame.interval.admission;
    ++frame.interval.sequence;
    frame.original_ids = 0;
    ASSERT_TRUE(rr::prepare(frame, geometry, 1, &error, original));
    EXPECT_FALSE(rr::accept(original, 0, geometry, 1, frame.interval.table, 0, &error, reply));
    ASSERT_TRUE(rr::accept(original, sizeof(original), geometry, 1, frame.interval.table, 0, &error, reply));
    auto cancelled = reply;
    EXPECT_FALSE(rr::acknowledge(cancelled, 0, geometry, &error, CapturedTransferStatus::Cancelled));
    EXPECT_FALSE(rr::confirm(cancelled, sizeof(cancelled), frame, geometry, 1, &error));
    EXPECT_TRUE(rr::acknowledge(reply, 0, geometry, &error, CapturedTransferStatus::Ready));
    EXPECT_TRUE(rr::confirm(reply, sizeof(reply), frame, geometry, 1, &error));
}
/** @brief Price actual concurrent reader channels without multiplying the original query envelope. */
TEST(PooledImmutableCoordinator, ReplicatedReadersGeometryBOMReplay)
{
    using ReaderGeometry = qwen4exp::PooledImmutableReplicatedReaderGeometry;
    const PhysicalMemoryAllocatorIdentity source{0, DeviceId::rocm(1)};
    const PhysicalMemoryResource host{0, DeviceId::cpu(), 1ULL << 30, 1ULL << 30};
    for (const auto count : {2u, 4u, 8u}) {
        std::vector<PhysicalMemoryAllocatorIdentity> consumers;
        std::vector<PhysicalMemoryResource> resources;
        for (std::uint32_t i = 0; i < count; ++i) {
            consumers.push_back({0, DeviceId::rocm(i)});
            resources.push_back({0, DeviceId::rocm(i), 1ULL << 30, 1ULL << 30});
        }
        for (const auto coverage : {PeerAccessCoverage::None, PeerAccessCoverage::Complete}) {
            std::vector<ImmutableRowTableReaderPeerAccess> edges;
            for (const auto consumer : consumers)
                if (consumer != source) edges.push_back({source, consumer, coverage});
            const auto geometry = ReaderGeometry::checked(source, consumers, 256, 64, edges);
            EXPECT_EQ(geometry.maximumOriginal(), 256u);
            EXPECT_EQ(geometry.resultMaximum(), 256u * 64u * sizeof(float));
            const auto bom = geometry.completeBOM(host, resources);
            const auto target = edges.front().reader;
            const auto forward = geometry.resultMemory(target);
            const auto reverse = geometry.acknowledgementMemory(target);
            const auto channels = count - 1;
            ASSERT_NE(bom.find({0, DeviceId::cpu()}), nullptr);
            EXPECT_EQ(bom.find({0, DeviceId::cpu()})->bytes(PhysicalMemoryOwner::ActivationTransportStaging),
                channels * (forward.mapped_host_bytes + reverse.mapped_host_bytes));
            EXPECT_EQ(bom.find(source)->bytes(PhysicalMemoryOwner::ActivationTransportStaging),
                channels * (forward.cursor_bytes_per_device + reverse.cursor_bytes_per_device +
                    sizeof(qwen4exp::replicated_reader::Control) + forward.producer_payload_bytes));
            for (const auto consumer : consumers) {
                if (consumer == source) continue;
                EXPECT_EQ(bom.find(consumer)->bytes(PhysicalMemoryOwner::ActivationTransportStaging),
                    forward.cursor_bytes_per_device + reverse.cursor_bytes_per_device +
                        sizeof(qwen4exp::replicated_reader::Control) + reverse.producer_payload_bytes);
            }
            EXPECT_EQ(forward.producer_payload_bytes != 0, coverage == PeerAccessCoverage::Complete);
            EXPECT_EQ(reverse.producer_payload_bytes != 0, coverage == PeerAccessCoverage::Complete);
            EXPECT_THROW(geometry.resultMemory(source), std::invalid_argument);
            auto malformed = edges;
            malformed.front().coverage = PeerAccessCoverage::Partial;
            EXPECT_THROW(ReaderGeometry::checked(source, consumers, 256, 64, malformed), std::invalid_argument);
            malformed = edges;
            malformed.front().coordinator = target;
            EXPECT_THROW(ReaderGeometry::checked(source, consumers, 256, 64, malformed), std::invalid_argument);
            EXPECT_THROW(ReaderGeometry::checked(source, consumers, 256, 64), std::invalid_argument);
        }
        consumers.back() = consumers.front();
        EXPECT_THROW(ReaderGeometry::checked(source, consumers, 256, 64), std::invalid_argument);
    }
}
} // namespace llaminar2::test

namespace llaminar2::test {
/** @brief Fixed native extent depends on original launch/failure authority, never a stale mapped byte publication. */
TEST(PooledImmutableCoordinator, NativeFixedHeaderPreservesOriginalFailureReplay)
{
    namespace rr = qwen4exp::replicated_reader;
    std::uint32_t abort = 0;
    int communicator = 0;
    const ncclLlaminarCompletionAuthority original{&abort, &communicator};
    EXPECT_EQ(rr::nativeFixedHeaderBytes(original, 0, 0), sizeof(rr::Header));
    EXPECT_EQ(rr::nativeFixedHeaderBytes(original, 1, 0), 0u);
    EXPECT_EQ(rr::nativeFixedHeaderBytes(original, 0, 7), 0u);
    EXPECT_EQ(rr::nativeFixedHeaderBytes({nullptr, &communicator}, 0, 0), 0u);
    EXPECT_EQ(rr::nativeFixedHeaderBytes({&abort, nullptr}, 0, 0), 0u);
    EXPECT_EQ(rr::nativeFixedHeaderBytes({}, 0, 0), 0u);
}

/** @brief Decoded native receipt law rejects stale/partial/empty replay even with zero abort/error flags.
 * These recorded structs test the actual production matching/reader arithmetic;
 * they do not manufacture a native-ready context or prove a GPU launch completed.
 */
TEST(PooledImmutableCoordinator, NativeReaderFreshReceiptAndCurrentACKReplay)
{
    namespace rr = qwen4exp::replicated_reader;
    namespace state = qwen4exp::adaptive_lookup::pooled;
    constexpr rr::Geometry geometry{1, 2, 256, 64};
    std::uint32_t abort = 0;
    int communicator = 0;
    const ncclLlaminarCompletionAuthority original{&abort, &communicator};
    constexpr std::uint64_t message = 99;
    int error = 0;
    state::LookupFrame frame{};
    frame.interval = {11, 7, 1};
    frame.phase = state::LookupPhase::Ready;
    frame.status = qwen4exp::adaptive_lookup::Status::Success;
    for (const auto count : {256u, 16u, 0u, 64u}) {
        frame.original_ids = count;
        rr::Header header{}, reply{};
        ASSERT_TRUE(rr::prepare(frame, geometry, 1, &error, header));
        const auto fixedBytes = rr::nativeFixedHeaderBytes(original, 0, error);
        ASSERT_TRUE(rr::accept(header, fixedBytes, geometry, 1, frame.interval.table, count, &error, reply));
        const auto bytes = rr::outputBytes(geometry, count);
        ncclLlaminarByteCompletion receipt{frame.interval.table, frame.interval.admission,
            frame.interval.sequence, bytes, original.abortFlag, original.deviceCommunicator,
            message, ncclLlaminarCompletionPhase::Completed, 0};
        const auto matches = [&](const ncclLlaminarByteCompletion& actual, std::uint64_t expectedBytes) {
            return ncclLlaminarCompletionMatches(actual, reply.interval.table, reply.interval.admission,
                reply.interval.sequence, expectedBytes, message, original, 0, error);
        };
        ASSERT_TRUE(matches(receipt, bytes));
        auto stale = receipt;
        ++stale.sequence;
        EXPECT_FALSE(matches(stale, bytes));
        stale = receipt;
        stale.phase = ncclLlaminarCompletionPhase::Pending;
        EXPECT_FALSE(matches(stale, bytes));
        stale = receipt;
        stale.phase = ncclLlaminarCompletionPhase::Aborted;
        EXPECT_FALSE(matches(stale, bytes));
        stale = receipt;
        ++stale.bytes;
        EXPECT_FALSE(matches(stale, bytes));
        auto rejected = reply;
        EXPECT_FALSE(rr::acknowledge(rejected, bytes, geometry, &error,
            matches(stale, bytes) ? CapturedTransferStatus::Ready : CapturedTransferStatus::InvalidPhase));
        EXPECT_FALSE(rr::confirm(rejected, fixedBytes, frame, geometry, 1, &error));
        ASSERT_TRUE(rr::acknowledge(reply, bytes, geometry, &error, CapturedTransferStatus::Ready));
        ASSERT_TRUE(rr::confirm(reply, fixedBytes, frame, geometry, 1, &error));
        ++frame.interval.sequence;
        EXPECT_FALSE(rr::confirm(reply, fixedBytes, frame, geometry, 1, &error));
        // A stale Header can pass shape acceptance; a fresh current receipt
        // cannot match its old interval, so it cannot produce a successful ACK.
        ASSERT_TRUE(rr::accept(header, fixedBytes, geometry, 1, frame.interval.table, count, &error, reply));
        ++receipt.sequence;
        EXPECT_FALSE(matches(receipt, bytes));
    }
}

/** @test The actual union enqueuer's immutable selector and capture policy cover zero/boundaries. */
TEST(PooledImmutableCoordinator, CooperativeUnionLaunchPolicyIsCaptureComplete)
{
    namespace t=qwen4exp::local_hot_transport;
    EXPECT_FALSE(t::usesCooperativeUnion(0));
    for(std::uint32_t bound:{1u,16u,32u,128u,256u})EXPECT_TRUE(t::usesCooperativeUnion(bound));
    for(std::uint32_t bound:{257u,512u,8192u})EXPECT_FALSE(t::usesCooperativeUnion(bound));
    EXPECT_NE(t::unionLaunchIdentity(256),t::unionLaunchIdentity(257));
    EXPECT_EQ(t::unionLaunchIdentity(32),t::unionLaunchIdentity(16));
}

/** @test Actual compact launch policy follows declared query rows, not retained frame capacity. */
TEST(PooledImmutableCoordinator, CompactCapturedQueryBoundIsTightAndCaptureComplete)
{
    namespace t=qwen4exp::local_hot_transport;
    using Bound=qwen4exp::PooledImmutableLocalHotQueryBound;
    std::array<t::ConsumerGeometry,2> consumers{{{0,2,8192,8192,0x100000100ULL,18,0,41},
                                               {1,2,8192,32,0x100000100ULL,18,0,42}}};
    std::int32_t count=0,other=0;
    const auto small=Bound::forCanonicalQuery(DeviceRowRange::deviceCounted(1,&count),consumers,16384);
    EXPECT_EQ(small.canonicalOriginal(),16u);EXPECT_EQ(small.consumerMaximum(0),16u);
    EXPECT_EQ(small.consumerMaximum(1),16u);EXPECT_EQ(small.maximumOriginal(),32u);
    EXPECT_TRUE(qwen4exp::pooled_native::coordinatorUsesSmallOriginal(DeviceRowRange::deviceCounted(small.maximumOriginal(),&count)));
    const auto large=Bound::forCanonicalQuery(DeviceRowRange::fullyActive(512),consumers,16384);
    EXPECT_EQ(large.maximumOriginal(),8224u);EXPECT_EQ(large.consumerMaximum(1),32u);
    EXPECT_FALSE(qwen4exp::pooled_native::coordinatorUsesSmallOriginal(DeviceRowRange::fullyActive(large.maximumOriginal())));
    for(int live:{0,1,0,1}){count=live;EXPECT_EQ(small,Bound::forCanonicalQuery(DeviceRowRange::deviceCounted(1,&count),consumers,16384));}
    EXPECT_FALSE(small==large);
    EXPECT_FALSE(small==Bound::forCanonicalQuery(DeviceRowRange::deviceCounted(1,&other),consumers,16384));
    EXPECT_THROW(Bound::forCanonicalQuery(DeviceRowRange::fullyActive(512),consumers,32),std::invalid_argument);
    EXPECT_THROW(Bound::forCanonicalQuery(DeviceRowRange::fullyActive(1),std::span<const t::ConsumerGeometry>{},16384),std::invalid_argument);
    auto foreign=consumers;foreign[1].coordinate=0;
    EXPECT_THROW(Bound::forCanonicalQuery(DeviceRowRange::fullyActive(1),foreign,16384),std::invalid_argument);
}

/** @test Unequal compact native requests use the actual original pool's stable union/inverse. */
TEST(PooledImmutableCoordinator, LocalHotCompactMissesReuseOriginalUnionAndRejectForeignRosters)
{
    namespace t=qwen4exp::local_hot_transport;
    const t::QueryIdentity query{701,17,40,4};
    std::array<a::KeyIndex,16> canonical{};
    const std::array<std::uint64_t,16> original{5,9,5,2,5,9,5,2,5,9,5,2,5,9,5,2};
    for(std::uint32_t i=0;i<16;++i)canonical[i]={original[i],i,0};
    std::stable_sort(canonical.begin(),canonical.end(),[](auto x,auto y){return x.row<y.row;});
    std::uint64_t originalCount=16;
    std::array<std::uint64_t,2> keys0{5,9},keys1{9,2};
    std::array<t::RequestHeader,3> requests{};std::array<std::uint64_t,3> headerBytes{},keyBytes{};
    t::UnionControl control{};std::array<std::uint64_t,48> merged{};
    const t::SourceAdmissionHeader admission{701,17,23,37};
    t::UnionArgs args{};args.expected=&query;args.source_admission=&admission;args.canonical={canonical.data(),&originalCount,16};
    args.consumer_count=3;args.capacity=merged.size();args.output=&control;args.merged_keys=merged.data();
    for(std::uint32_t c=0;c<3;++c){
        requests[c].source_admission=23;requests[c].source_invocation=37;
        requests[c].query={701,17,40+c,4+c};requests[c].consumer=c;requests[c].original_ids=16;
        requests[c].native_row_bytes=18;requests[c].native_format=static_cast<std::uint32_t>(GGUFTensorType::IQ4_NL);
        requests[c].miss_rows=c==2?0:2;headerBytes[c]=sizeof(t::RequestHeader);keyBytes[c]=requests[c].miss_rows*8;
        args.consumers[c]={c,3,16,16,0x100000100ULL,18,static_cast<std::uint32_t>(GGUFTensorType::IQ4_NL),40+c};
        args.requests[c]={&requests[c],c==0?keys0.data():keys1.data(),&headerBytes[c],&keyBytes[c]};
    }
    // Same admitted buffers, unequal per-consumer compact ceilings and a tight
    // checked sum. Actual zero prefixes remain ordinary progress, not fallback.
    args.consumers[0].maximum_misses=2;args.consumers[1].maximum_misses=3;args.consumers[2].maximum_misses=1;
    const auto tight=qwen4exp::PooledImmutableLocalHotQueryBound::forCanonicalQuery(
        DeviceRowRange::fullyActive(1),std::span{args.consumers,std::size_t{3}},merged.size());
    args.capacity=tight.maximumOriginal();args.canonical.capacity=tight.canonicalOriginal();
    EXPECT_EQ(args.capacity,6u);
    for(unsigned repeat=0;repeat<20;++repeat){
        for(auto& request:requests)request.miss_rows=0;keyBytes={0,0,0};
        ASSERT_TRUE(t::prepareUnion(args));EXPECT_EQ(control.pool_original_ids,0);
        requests[0].miss_rows=2;requests[1].miss_rows=2;keyBytes={16,16,0};
        ASSERT_TRUE(t::prepareUnion(args));EXPECT_EQ(control.pool_original_ids,4);
    }
    requests[0].miss_rows=3;keyBytes[0]=24;
    EXPECT_FALSE(t::prepareUnion(args));EXPECT_EQ(control.pool_original_ids,-1);
    requests[0].miss_rows=2;keyBytes[0]=16;
    ASSERT_TRUE(t::prepareUnion(args));ASSERT_EQ(control.pool_original_ids,4);
    for(std::uint32_t c=0;c<3;++c)for(std::uint32_t m=0;m<requests[c].miss_rows;++m)ASSERT_TRUE(t::copyMiss(args,c,m));
    const std::array<std::uint64_t,4> expected{5,9,9,2};EXPECT_TRUE(std::equal(expected.begin(),expected.end(),merged.begin()));
    Pool pool(std::array{0u});pool.begin(std::span(merged.data(),control.pool_original_ids));
    ASSERT_EQ(pool.frame.unique_rows,3u);
    EXPECT_EQ(pool.unique[0].row,5);EXPECT_EQ(pool.unique[1].row,9);EXPECT_EQ(pool.unique[2].row,2);
    const std::array<std::uint32_t,4> inverse{0,1,1,2};EXPECT_TRUE(std::equal(inverse.begin(),inverse.end(),pool.permutation.begin()));
    keys1[0]=6;EXPECT_FALSE(t::copyMiss(args,1,0)); // in source range, absent from actual query
    keys1={9,9};EXPECT_FALSE(t::copyMiss(args,1,1)); // duplicate is unnecessary native traffic
    keys1={2,9};EXPECT_FALSE(t::copyMiss(args,1,1)); // wrong stable original ordering
    keys1={9,2};--keyBytes[1];EXPECT_FALSE(t::prepareUnion(args));EXPECT_EQ(control.pool_original_ids,-1);
    ++keyBytes[1];++requests[1].query.consumer_storage_identity;
    EXPECT_FALSE(t::prepareUnion(args));EXPECT_EQ(control.pool_original_ids,-1);
    --requests[1].query.consumer_storage_identity;++requests[1].query.source_storage_generation;
    EXPECT_FALSE(t::prepareUnion(args));EXPECT_EQ(control.pool_original_ids,-1);
    --requests[1].query.source_storage_generation;requests[1].intent=qwen4exp::PooledImmutableLocalHotAccessIntent::FutureReadAhead;
    EXPECT_FALSE(t::prepareUnion(args));EXPECT_EQ(control.pool_original_ids,-1);
}

/** @test Native compact rows are accepted only by exact original query/source/reader/extent receipts. */
TEST(PooledImmutableCoordinator, LocalHotNativeReceiptsKeepArbitraryMissCountsAndAbsorbingFailures)
{
    namespace t=qwen4exp::local_hot_transport;
    const t::QueryIdentity query{701,17,77,9};
    const t::ConsumerGeometry geometry{1,2,32,32,1000,18,static_cast<std::uint32_t>(GGUFTensorType::IQ4_NL),77};
    for(unsigned replay=0;replay<20;++replay)for(const auto misses:{0u,1u,3u,17u,32u}){
        t::RequestHeader request{};request.query=query;request.original_ids=32;request.consumer=1;
        request.native_format=geometry.native_format;request.native_row_bytes=18;request.miss_rows=misses;
        request.source_admission=2;request.source_invocation=37;
        const t::SourceAdmissionHeader admission{701,17,2,37};
        ASSERT_TRUE(t::acceptRequest(request,query,geometry,32,admission,sizeof(request),misses*8));
        t::ResultHeader result{};result.query=query;result.pool_interval={701,2,11};result.consumer=1;
        result.native_format=geometry.native_format;result.miss_rows=misses;result.native_bytes=misses*18;
        ASSERT_TRUE(t::acceptResult(result,request,geometry,sizeof(result),misses*18));
        for(unsigned mutation=0;mutation<10;++mutation){auto invalid=result;
            if(mutation==0)++invalid.query.sequence;
            if(mutation==1)++invalid.query.source_storage_generation;
            if(mutation==2)++invalid.pool_interval.table;
            if(mutation==3)++invalid.query.consumer_storage_identity;
            if(mutation==4)invalid.pool_interval.sequence=0;
            if(mutation==5)invalid.consumer=0;
            if(mutation==6)++invalid.native_format;
            if(mutation==7)++invalid.miss_rows;
            if(mutation==8)invalid.status=a::Status::Cancelled;
            if(mutation==9)invalid.intent=qwen4exp::PooledImmutableLocalHotAccessIntent::FutureReadAhead;
            EXPECT_FALSE(t::acceptResult(invalid,request,geometry,sizeof(result),misses*18))<<mutation;
        }
        EXPECT_FALSE(t::acceptResult(result,request,geometry,0,misses*18));
        EXPECT_FALSE(t::acceptResult(result,request,geometry,sizeof(result),misses*18+1));
        if(misses)EXPECT_FALSE(t::acceptResult(result,request,geometry,sizeof(result),misses*4*16));
        ++request.original_ids;EXPECT_FALSE(t::acceptRequest(request,query,geometry,32,admission,sizeof(request),misses*8));
    }
}

/** @test Optional compact completion slots and mandatory source scratch are exact without changing shared-only bytes. */
TEST(PooledImmutableCoordinator, LocalHotGeometryKeepsOriginalUniqueBoundAndOptionalNativeSlots)
{
    namespace q=qwen4exp;
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(const auto degree:{1u,2u,4u,8u}){
        std::vector<PhysicalMemoryAllocatorIdentity> consumers;
        for(unsigned i=0;i<degree;++i)consumers.push_back({0,{backend,static_cast<int>(i)}});
        const auto extent=ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,1000,18);
        const auto shape=degree==1?q::PooledImmutableLocalHotTransportGeometry::singleton(consumers[0],16,extent):
            q::PooledImmutableLocalHotTransportGeometry::declare(q::PooledImmutableReplicatedReaderGeometry::native(consumers[0],consumers,16,32),extent);
        EXPECT_EQ(shape.maximumPoolUniqueRows(),16u);EXPECT_EQ(shape.maximumPoolOriginalRows(),16u*degree);
        EXPECT_EQ(shape.resultCapacityBytes(),16u*18u);EXPECT_EQ(shape.incomingKeyBytes(),(degree-1)*16u*8u);
        EXPECT_EQ(shape.outgoingNativeBytes(),(degree-1)*16u*18u);
        if(degree==1){EXPECT_EQ(shape.mergedKeyBytes(),0u);EXPECT_FALSE(shape.originalReaders());continue;}
        std::vector<std::uint8_t> positive(degree);for(unsigned i=1;i<degree;++i)positive[i]=i%2;
        const q::PooledNativeCompletionLayout shared(degree,0,positive),compact(degree,0,positive,q::PooledNativeReaderMessageProfile::CompactNative);
        EXPECT_EQ(shared.readerProfile(),q::PooledNativeReaderMessageProfile::ReplicatedFP32);
        for(unsigned i=1;i<degree;++i){
            EXPECT_EQ(shared.receiptBytesPerParticipant(i),(positive[i]?12u:1u)*sizeof(ncclLlaminarByteCompletion));
            EXPECT_EQ(compact.receiptBytesPerParticipant(i),(positive[i]?13u:2u)*sizeof(ncclLlaminarByteCompletion));
            EXPECT_THROW(shared.operandOffset(q::PooledNativeCountedMessage::LocalHotMissKeys,0,i),std::invalid_argument);
            EXPECT_NO_THROW(compact.operandOffset(q::PooledNativeCountedMessage::LocalHotMissKeys,0,i));
        }
    }
}

/** @test Actual request admission stays separate from unequal L1 ordinals and pool intervals. */
TEST(PooledImmutableCoordinator, LocalHotRequestPublicationRejectsStaleAdmissionAndKeepsEmptyProgress)
{
    namespace t=qwen4exp::local_hot_transport;
    std::uint64_t admission=23,invocation=37;
    t::SourceAdmissionHeader source{};
    ASSERT_TRUE(t::publishSourceAdmission({701,17,&admission,&invocation,&source}));
    std::uint64_t headerBytes=sizeof(source),misses=3,bytes=999;
    std::uint64_t original=32;
    t::QueryIdentity query{701,17,77,9};
    const t::ConsumerGeometry geometry{1,2,32,32,1000,18,static_cast<std::uint32_t>(GGUFTensorType::IQ4_NL),77};
    t::RequestHeader request{};t::RequestLaneIdentity lane{};
    t::RequestArgs args{&query,&original,&misses,&source,&headerBytes,geometry,
        qwen4exp::PooledImmutableLocalHotAccessIntent::CurrentInvocation,&request,&lane,&bytes};
    for(unsigned replay=0;replay<20;++replay){
        for(const auto live:{32u,3u,0u,1u}){
            misses=live;++query.sequence;
            ASSERT_TRUE(t::prepareRequest(args));EXPECT_EQ(bytes,live*8u);
            EXPECT_EQ(lane.table,701);EXPECT_EQ(lane.source_admission,23);EXPECT_EQ(lane.sequence,query.sequence);
            t::RequestLaneIdentity received{};
            const t::QueryIdentity sourceQuery{701,17,66,1}; // source ordinal is deliberately unrelated
            ASSERT_TRUE(t::acceptRequestLane(request,sourceQuery,geometry,32,source,sizeof(request),received));
            EXPECT_EQ(received.sequence,query.sequence);
            t::ResultHeader result{};std::uint64_t nativeBytes=999;
            ASSERT_TRUE(t::prepareResult(request,geometry,source,{701,23,101},result,nativeBytes));
            EXPECT_EQ(nativeBytes,live*18u);
            EXPECT_TRUE(t::acceptResult(result,request,geometry,sizeof(result),nativeBytes));
            auto stale=source;++stale.source_invocation;
            EXPECT_FALSE(t::acceptRequestLane(request,sourceQuery,geometry,32,stale,sizeof(request),received));
            EXPECT_EQ(received.sequence,0);
            EXPECT_FALSE(t::prepareResult(request,geometry,source,{701,24,101},result,nativeBytes));
            EXPECT_EQ(nativeBytes,0);
        }
    }
    --headerBytes;EXPECT_FALSE(t::prepareRequest(args));EXPECT_EQ(bytes,0);EXPECT_EQ(lane.sequence,0);
    ++headerBytes;args.intent=qwen4exp::PooledImmutableLocalHotAccessIntent::FutureReadAhead;
    ASSERT_TRUE(t::prepareRequest(args));
    EXPECT_FALSE(t::acceptRequest(request,query,geometry,32,source,sizeof(request),bytes));
    EXPECT_TRUE(t::acceptRequest(request,query,geometry,32,source,sizeof(request),bytes,args.intent));
    args.intent=qwen4exp::PooledImmutableLocalHotAccessIntent::CurrentInvocation;
    source.status=a::Status::Cancelled;EXPECT_FALSE(t::prepareRequest(args));EXPECT_EQ(bytes,0);
    admission=0;EXPECT_FALSE(t::publishSourceAdmission({701,17,&admission,&invocation,&source}));
    EXPECT_NE(source.status,a::Status::Success);
}

/** @test Lookup-only uses actual empty owner receipts, retained pins and no demand policy credit. */
TEST(PooledImmutableCoordinator, FutureLookupOnlyRetainsPinsAndNeverMutatesDemandPolicy)
{
    const std::array<std::uint32_t,3> capacities{2,1,0};
    Pool pool(capacities), twin(capacities);
    const std::array<std::uint64_t,3> seed{0,1,2};
    for(auto* owner:{&pool,&twin}){
        owner->begin(seed);owner->freeWave();owner->victimWave();owner->finish();
    }
    for(unsigned replay=0;replay<20;++replay){
        const auto before=pool.policy;
        const auto epochs=pool.epochs;const auto counts=pool.counts;
        const auto entries=pool.entries;
        const std::array<std::uint64_t,5> forecast{0,80,80,1,81};
        pool.begin(forecast,1,p::LookupPurpose::FutureLookupOnly);
        EXPECT_GT(pool.owners[0]->frame.initial_pin_count,0u);
        pool.freeWave();pool.victimWave();
        EXPECT_EQ(pool.frame.free_planned,0u);EXPECT_EQ(pool.frame.victim_planned,0u);
        for(const auto& owner:pool.owners)EXPECT_EQ(owner->allocation_frame.count,0u);
        if(replay%2)pool.cancel();else pool.finish();
        EXPECT_EQ(std::memcmp(&pool.policy,&before,sizeof(before)),0);
        EXPECT_EQ(pool.epochs,epochs);EXPECT_EQ(pool.counts,counts);
        for(std::size_t at=0;at<entries.size();++at){
            EXPECT_EQ(pool.entries[at].touched,entries[at].touched);
            EXPECT_EQ(pool.entries[at].previous,entries[at].previous);
            EXPECT_EQ(pool.entries[at].next,entries[at].next);
            EXPECT_EQ(pool.entries[at].stream,entries[at].stream);
            EXPECT_EQ(pool.entries[at].present,entries[at].present);
        }
        const std::array<std::uint64_t,3> demand{3+replay%7,3+replay%7,1};
        for(auto* owner:{&pool,&twin}){
            owner->begin(demand);owner->freeWave();owner->victimWave();owner->finish();
        }
        EXPECT_EQ(std::memcmp(&pool.policy,&twin.policy,sizeof(pool.policy)),0);
        EXPECT_EQ(pool.epochs,twin.epochs);EXPECT_EQ(pool.counts,twin.counts);
    }
    Pool invalid(capacities);
    EXPECT_FALSE(invalid.operations().begin(1,seed.data(),seed.size(),static_cast<p::LookupPurpose>(99)));
}

/** @test Future metadata requires its original purpose and actual live native extents. */
TEST(PooledImmutableCoordinator, FutureCompactWireRequiresPurposeAndExactPartialExtents)
{
    namespace t=qwen4exp::local_hot_transport;
    const auto future=qwen4exp::PooledImmutableLocalHotAccessIntent::FutureReadAhead;
    std::uint64_t admission=23,invocation=37,original=32,misses=0,bytes=0;
    t::SourceAdmissionHeader source{};
    ASSERT_TRUE(t::publishSourceAdmission({701,17,&admission,&invocation,&source}));
    std::uint64_t admissionBytes=sizeof(source);
    t::QueryIdentity query{701,17,77,9};
    const t::ConsumerGeometry geometry{1,2,32,32,1000,18,static_cast<std::uint32_t>(GGUFTensorType::IQ4_NL),77};
    t::RequestHeader request{};t::RequestLaneIdentity lane{};
    t::RequestArgs args{&query,&original,&misses,&source,&admissionBytes,geometry,future,&request,&lane,&bytes};
    for(unsigned replay=0;replay<20;++replay)for(const auto live:{0u,1u,3u,32u}){
        misses=live;++query.sequence;
        ASSERT_TRUE(t::prepareRequest(args));EXPECT_EQ(bytes,live*8u);
        EXPECT_FALSE(t::acceptRequest(request,query,geometry,32,source,sizeof(request),bytes));
        EXPECT_TRUE(t::acceptRequest(request,query,geometry,32,source,sizeof(request),bytes,future));
        EXPECT_FALSE(t::acceptRequest(request,query,geometry,32,source,sizeof(request),bytes+8,future));
        t::ResultHeader result{};std::uint64_t nativeBytes=0;
        ASSERT_TRUE(t::prepareResult(request,geometry,source,{701,23,101},result,nativeBytes,future));
        EXPECT_EQ(nativeBytes,live*18u);
        EXPECT_TRUE(t::acceptResult(result,request,geometry,sizeof(result),nativeBytes,future));
        EXPECT_FALSE(t::acceptResult(result,request,geometry,sizeof(result),nativeBytes));
        EXPECT_FALSE(t::acceptResult(result,request,geometry,sizeof(result),nativeBytes+18,future));
    }
}

} // namespace llaminar2::test
