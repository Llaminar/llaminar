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
    void begin(std::span<const std::uint64_t> original,std::uint64_t admission=1)
    {
        require(operations().begin(admission,original.data(),static_cast<std::uint32_t>(original.size())));
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
