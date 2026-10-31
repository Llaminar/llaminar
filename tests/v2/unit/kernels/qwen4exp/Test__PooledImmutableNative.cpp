/**
 * @file Test__PooledImmutableNative.cpp
 * @brief Replay actual shared native metadata phases using decoded, acquired CPU fixtures.
 *
 * These roots exercise the same phase parsers used by CUDA/HIP kernels. The
 * fixture supplies decoded transport/source observations, and its CPU payload
 * producer does not certify GPU copy, native capture, transport or serving.
 */
#include "kernels/qwen4exp/PooledImmutableNativeOps.h"
#include "kernels/qwen4exp/PooledImmutableNativePlan.h"
#include "kernels/qwen4exp/PooledImmutableNativeOperands.h"
#include "kernels/qwen4exp/PooledImmutableSourceOps.h"
#include "loaders/ImmutableRowTable.h"
#include "loaders/ModelLoader.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
namespace llaminar2::test {
namespace {
namespace q=qwen4exp;
namespace n=q::pooled_native;
namespace a=q::adaptive_lookup;
namespace p=a::pooled;
/** @brief Exact aligned parser roots; these arrays are not native physical allocations. */
struct Bytes {
    struct alignas(64) Block {std::byte data[64];};
    std::vector<Block> blocks;
    explicit Bytes(std::size_t bytes):blocks((bytes+63)/64){if(bytes)std::memset(data(),0xcc,blocks.size()*64);}
    std::byte* data(){return blocks.empty()?nullptr:reinterpret_cast<std::byte*>(blocks.data());}
};
/** @return Uneven arbitrary owner order with independent cross-vendor coordinator and CPU consumer. */
q::PooledImmutablePhysicalOwnerGeometry declared(std::span<const std::uint32_t> capacities)
{
    std::vector<PhysicalMemoryAllocatorIdentity> order;
    std::vector<ImmutableRowTableLayoutOwner> owners;
    bool positive=false;
    for(std::uint32_t i=0;i<capacities.size();++i){
        order.push_back({static_cast<int>(i),i%2?DeviceId::rocm(static_cast<int>(i)):DeviceId::cuda(static_cast<int>(i))});
        owners.push_back({{order.back(),ImmutableRowStorage::VRAM},0,capacities[i],{}});positive|=capacities[i]!=0;
    }
    const PhysicalMemoryAllocatorIdentity coordinator{101,DeviceId::rocm(11)};
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
    const auto layout=ImmutableRowTableLayoutGeometry::checked(
        ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,0x100000100ULL,90),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,owners,
        {{102,DeviceId::cpu()},coordinator},{},
        {{{103,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},
        {},positive?std::optional(coordinator):std::nullopt,16);
    return q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,order,coordinator);
}
/** @brief One actual native owner parser with immutable original bank geometry. */
struct Owner {
    q::PooledImmutableOwnerNativePlan plan;
    Bytes cache,frame,native;
    n::OwnerArgs args;
    explicit Owner(q::PooledImmutableOwnerNativePlan declaration):plan(std::move(declaration)),
        cache(plan.parameters().cache.bytes),frame(plan.parameters().frame.bytes),native(plan.parameters().native.bytes),
        args{plan.parameters(),cache.data(),frame.data(),native.data(),{}}
    {
        n::seedOwner<a::Slot64>(args);
        auto physical=p::OwnerOperations<a::Slot64>(n::ownerView<a::Slot64>(args));
        for(std::uint32_t i=0;i<args.plan.owner.capacity();++i)physical.initializeSlot(i);
        if(!n::sealOwner<a::Slot64>(args))throw std::logic_error("native owner setup");
    }
    template<n::OwnerAction Action> bool run(){return n::owner<Action,a::Slot64>(args);}
    template<class H> H header()const{return *reinterpret_cast<const H*>(native.blocks.data()+args.plan.native.header/64);}
};
/** @brief Actual phase replay; only decoded peer/source acquisition is supplied by this fixture. */
struct Pool {
    q::PooledImmutablePhysicalOwnerGeometry physical;
    q::PooledImmutableCoordinatorGeometry geometry;
    q::PooledImmutableCoordinatorNativePlan plan;
    Bytes cache,frame,native;
    n::CoordinatorArgs args;
    std::vector<std::unique_ptr<Owner>> owners;
    std::uint64_t source_sequence=0;
    explicit Pool(std::span<const std::uint32_t> capacities,std::uint32_t maximumOriginal=64,a::Policy policy={65536,1,2}):
        physical(declared(capacities)),geometry(q::PooledImmutableCoordinatorGeometry::fromOwners(physical,policy,maximumOriginal)),
        plan(q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,geometry,701,17)),
        cache(plan.parameters().cache.bytes),frame(plan.parameters().frame.bytes),native(plan.parameters().native.bytes),
        args{plan.parameters(),cache.data(),frame.data(),native.data(),{}}
    {
        const n::CoordinatorSetupArgs setup{args.plan,args.cache,args.frame,args.native};n::seedCoordinator(setup);
        std::copy(plan.owners().begin(),plan.owners().end(),n::at<p::CoordinatorOwnerGeometry>(args.frame,args.plan.frame.owners));
        require(p::CoordinatorOperations(n::coordinatorView(args)).initialize());
        for(std::uint32_t i=0;i<capacities.size();++i)owners.push_back(std::make_unique<Owner>(
            q::PooledImmutableOwnerNativePlan::fromGeometry(physical,geometry,i,701,17)));
    }
    static void require(bool value){if(!value)throw std::logic_error("native decoded phase rejected");}
    template<n::CoordinatorAction Action> void run(){
        if constexpr(Action==n::CoordinatorAction::PlanFree){
            auto v=view();constexpr auto width=p::OriginalObservationRuns::PartitionRows;
            for(std::uint32_t first=0;first<v.frame->original_ids;first+=width){
                const auto end=std::min(v.frame->original_ids,first+width);std::uint32_t count=0;
                for(auto pos=first;pos<end;){const auto unique=v.original_to_unique[pos];auto next=pos+1;
                    while(next<end && v.original_to_unique[next]==unique)++next;
                    v.sort_a[first+count++]={v.unique[unique].row,pos,next-pos};pos=next;
                }
                v.sort_b[first/width].unique=count;
            }
        }
        require(n::coordinator<Action>(args));
    }
    template<n::OwnerAction Action> void all(){for(auto& owner:owners)require(owner->run<Action>());}
    p::CoordinatorView view(){return n::coordinatorView(args);}
    const n::Invocation* invocation(){return n::at<n::Invocation>(args.native,args.plan.native.invocation);}
    /** @brief Feed the actual compact grant prefix without changing owner banks or their observed order. */
    template<n::CoordinatorAction Action> void grants(bool pins=false){
        std::vector<p::Grant> acquired;
        for(auto& owner:owners){
            auto& x=owner->args;const auto count=n::progress(x).reply_bytes/sizeof(p::Grant);
            const auto* source=pins?n::at<p::Grant>(x.frame,x.plan.frame.initial_pins):n::grants(x)+n::allocationFrame(x).wave_offset;
            acquired.insert(acquired.end(),source,source+count);
        }
        const std::uint64_t bytes=acquired.size()*sizeof(p::Grant);
        args.input.records=acquired.empty()?nullptr:acquired.data();args.input.record_bytes=args.plan.native.reply_capacity?&bytes:nullptr;
        run<Action>();args.input={};
    }
    template<class H> std::vector<H> headers(){std::vector<H> result;for(auto& owner:owners)result.push_back(owner->header<H>());return result;}
    /** @brief Common Begin is the CPU coalescing oracle; GPU Begin's cooperative producer is tested separately natively. */
    void begin(std::span<const std::uint64_t> original,std::uint64_t admission=1){
        require(p::CoordinatorOperations(view()).begin(admission,original.data(),original.size()));n::invocation(args);
        for(auto& owner:owners){owner->args.input={invocation(),nullptr,nullptr,nullptr,nullptr,nullptr};require(owner->run<n::OwnerAction::Begin>());}
        run<n::CoordinatorAction::PinCommands>();
        for(std::size_t i=0;i<owners.size();++i){
            auto& x=owners[i]->args;const auto slice=n::at<n::DirectedSlice>(args.native,args.plan.native.pin_slices)[i];
            x.input.records=x.plan.native.reply_capacity?n::at<p::OwnerCommand>(args.native,args.plan.native.pin_commands)+slice.byte_offset/sizeof(p::OwnerCommand):nullptr;
            x.input.record_bytes=x.plan.native.reply_capacity?&slice.live_bytes:nullptr;
            require(owners[i]->run<n::OwnerAction::Pin>());x.input.records=nullptr;x.input.record_bytes=nullptr;
        }
        grants<n::CoordinatorAction::AcquirePins>(true);all<n::OwnerAction::SealPins>();const auto pins=headers<p::PinHeader>();
        args.input.frontier=pins.data();run<n::CoordinatorAction::SealPins>();args.input={};
        for(auto& owner:owners){owner->args.input.frontier=pins.data();require(owner->run<n::OwnerAction::AcquirePins>());owner->args.input.frontier=nullptr;}
    }
    /** @brief Direct plans, actual wave execution/grants, and complete ALL-owner completion frontiers. */
    template<bool Victims> void wave(bool directed=false){
        if constexpr(Victims)run<n::CoordinatorAction::PlanVictims>();
        else{all<n::OwnerAction::Availability>();const auto availability=headers<n::Availability>();args.input.frontier=availability.data();run<n::CoordinatorAction::PlanFree>();args.input={};}
        for(std::size_t i=0;i<owners.size();++i){
            auto& owner=*owners[i];auto& x=owner.args;
            const auto offset=Victims?args.plan.native.victim_slices:args.plan.native.free_slices;
            const auto slice=n::at<n::DirectedSlice>(args.native,offset)[i];
            x.input.records=x.plan.native.reply_capacity?args.frame+slice.byte_offset:nullptr;
            x.input.frontier=Victims?static_cast<const void*>(n::at<n::VictimPlanHeader>(args.native,args.plan.native.victim_headers)+i):
                static_cast<const void*>(n::at<n::FreePlanHeader>(args.native,args.plan.native.free_headers)+i);
            if(directed && x.plan.native.reply_capacity){
                x.input.records=args.frame+args.plan.frame.decisions;
                x.input.record_layout=n::RecordLayout::DirectedAllocationBank;
                x.input.allocation_slice=n::at<n::DirectedSlice>(args.native,offset)+i;
                x.input.record_origin=args.plan.frame.decisions;
                x.input.record_capacity=std::uint64_t{x.plan.owner.maximumUnique()}*sizeof(p::AllocationDecision);
            }
            if constexpr(Victims)require(owner.run<n::OwnerAction::AcceptVictim>());
            else require(owner.run<n::OwnerAction::AcceptFree>());
            x.input.record_layout=n::RecordLayout::Compact;x.input.allocation_slice=nullptr;
            x.input.record_origin=x.input.record_capacity=0;
            if constexpr(Victims)require(owner.run<n::OwnerAction::ExecuteVictim>());
            else require(owner.run<n::OwnerAction::ExecuteFree>());
            x.input.records=nullptr;x.input.frontier=nullptr;
        }
        if constexpr(Victims){grants<n::CoordinatorAction::AcquireVictims>();all<n::OwnerAction::VictimReceipt>();const auto receipts=headers<p::VictimAllocationReceipt>();args.input.frontier=receipts.data();run<n::CoordinatorAction::CompleteVictims>();args.input={};}
        else{grants<n::CoordinatorAction::AcquireFree>();all<n::OwnerAction::FreeReceipt>();const auto receipts=headers<p::FreeAllocationReceipt>();args.input.frontier=receipts.data();run<n::CoordinatorAction::CompleteFree>();args.input={};
            for(auto& owner:owners){owner->args.input.frontier=receipts.data();require(owner->run<n::OwnerAction::AcquireFree>());owner->args.input.frontier=nullptr;}}
    }
    /** @brief Replay exact source response validation, unchanged native row copy, ACK/Ready and final retirement. */
    void finish(){
        bool misses=false;for(std::uint32_t i=0;i<view().frame->unique_rows;++i)misses|=view().unique[i].phase!=p::UniquePhase::Read;
        n::SourceFrontier source{view().frame->interval,misses?p::SourceWindow{701,17,++source_sequence}:p::SourceWindow{},
            misses?n::SourceDisposition::Published:n::SourceDisposition::Unpublished,a::Status::Success,n::Version};
        args.input.source=&source;run<n::CoordinatorAction::SealSource>();args.input={};
        for(auto& owner:owners){owner->args.input.frontier=&source;require(owner->run<n::OwnerAction::SealSource>());}
        if(misses)source.disposition=n::SourceDisposition::Consumed;
        for(auto& owner:owners){
            auto& x=owner->args;const auto count=n::allocationFrame(x).count;
            std::vector<n::Fill> fills(count);std::vector<std::byte> payload(std::size_t{count}*x.plan.row_bytes);
            for(std::uint32_t i=0;i<count;++i){fills[i]={n::grants(x)[i],{source.source,n::grants(x)[i].destination_token}};
                for(std::size_t j=0;j<x.plan.row_bytes;++j)payload[std::size_t{i}*x.plan.row_bytes+j]=std::byte((n::grants(x)[i].row+j)&255);}
            const std::uint64_t bytes=fills.size()*sizeof(n::Fill),payload_bytes=payload.size();
            x.input.records=x.plan.native.reply_capacity?fills.data():nullptr;x.input.record_bytes=x.plan.native.reply_capacity?&bytes:nullptr;
            x.input.payload=x.plan.native.reply_capacity?payload.data():nullptr;x.input.payload_bytes=x.plan.native.reply_capacity?&payload_bytes:nullptr;
            require(owner->run<n::OwnerAction::ApplySourceDelivery>());
            // Fixture decoded CPU producer, not a claim that GPU byte-copy/capture has run.
            for(std::uint32_t i=0;i<count;++i)std::memcpy(x.cache+x.plan.cache.payload+std::size_t{n::grants(x)[i].slot}*x.plan.slot_stride,
                payload.data()+std::size_t{i}*x.plan.row_bytes,x.plan.row_bytes);
            require(n::copied<a::Slot64>(x));x.input={invocation(),nullptr,nullptr,nullptr,nullptr,nullptr};
            require(owner->run<n::OwnerAction::ValidateFills>());
        }
        const auto published=headers<p::PublishHeader>();
        for(auto& owner:owners){owner->args.input.frontier=published.data();require(owner->run<n::OwnerAction::AcquirePublication>());owner->args.input.frontier=nullptr;require(owner->run<n::OwnerAction::MakeReady>());require(owner->run<n::OwnerAction::ReadyReceipt>());}
        const auto ready=headers<p::ReadyHeader>();args.input.frontier=ready.data();args.input.source=&source;run<n::CoordinatorAction::Publish>();args.input={};
        all<n::OwnerAction::Release>();all<n::OwnerAction::Retire>();const auto retired=headers<p::RetiredHeader>();args.input.frontier=retired.data();run<n::CoordinatorAction::Finish>();args.input={};
    }
    void complete(std::span<const std::uint64_t> original,std::uint64_t admission=1){begin(original,admission);wave<false>();wave<true>();finish();}
};
}
TEST(PooledImmutableNative, ActualDecodedNativePhasesReplayFreeVictimSourceAndRepeatedReset)
{
    Pool pool(std::array{2u,0u,2u});
    ASSERT_NO_THROW(pool.complete(std::array<std::uint64_t,2>{101,102}));
    const std::array<std::uint64_t,14> original{700,701,900,900,900,900,900,900,900,900,900,900,900,900};
    ASSERT_NO_THROW(pool.begin(original));
    ASSERT_NO_THROW(pool.wave<false>());
    ASSERT_EQ(pool.view().frame->free_planned,2u);
    std::vector<std::vector<p::AllocationDecision>> accepted;
    std::vector<std::vector<p::Grant>> granted;
    for(auto& owner:pool.owners){auto& x=owner->args;const auto count=n::allocationFrame(x).count;
        accepted.emplace_back(n::at<p::AllocationDecision>(x.frame,x.plan.frame.accepted_plan),n::at<p::AllocationDecision>(x.frame,x.plan.frame.accepted_plan)+count);
        granted.emplace_back(n::grants(x),n::grants(x)+count);}
    ASSERT_NO_THROW(pool.wave<true>());
    ASSERT_EQ(pool.view().frame->victim_planned,1u);
    for(std::size_t i=0;i<pool.owners.size();++i){const auto& x=pool.owners[i]->args;
        if(!accepted[i].empty())EXPECT_EQ(std::memcmp(accepted[i].data(),n::at<p::AllocationDecision>(x.frame,x.plan.frame.accepted_plan),accepted[i].size()*sizeof(p::AllocationDecision)),0);
        if(!granted[i].empty())EXPECT_EQ(std::memcmp(granted[i].data(),n::grants(x),granted[i].size()*sizeof(p::Grant)),0);}
    ASSERT_NO_THROW(pool.finish());
    for(unsigned replay=0;replay<4;++replay)ASSERT_NO_THROW(pool.complete(original,replay+2));
    EXPECT_EQ(pool.view().frame->phase,p::LookupPhase::Idle);
}
TEST(PooledImmutableNative, C0EmptyAndLargeToSmallUseExactLiveCountsAndRetainIdentity)
{
    for(const auto capacities:{std::vector{0u},std::vector{0u,0u,0u,0u,0u},std::vector{1u,0u,2u,0u,1u}}){
        Pool pool(capacities,8192);const auto identity=pool.plan.parameters().identity;
        std::vector<std::uint64_t> original(8192);for(std::size_t i=0;i<original.size();++i)original[i]=31+i%3;
        ASSERT_NO_THROW(pool.complete(original));
    ASSERT_EQ(pool.view().frame->unique_rows,3u);
        ASSERT_NO_THROW(pool.complete(std::array<std::uint64_t,1>{31},2));
    ASSERT_EQ(pool.view().frame->original_ids,1u);
        ASSERT_NO_THROW(pool.complete({},3));
    EXPECT_EQ(pool.plan.parameters().identity,identity);
        for(auto& owner:pool.owners)if(!owner->args.plan.owner.capacity()){
            EXPECT_EQ(owner->args.cache,nullptr);
    EXPECT_EQ(owner->args.plan.native.bytes,128u);
    EXPECT_EQ(owner->args.plan.native.reply_capacity,0u);}
    }
}
TEST(PooledImmutableNative, GlobalUnusedFreeQuotaForbidsNativeVictimPlan)
{
    Pool pool(std::array{1u,0u,2u});
    ASSERT_NO_THROW(pool.complete(std::array<std::uint64_t,1>{101}));
    ASSERT_NO_THROW(pool.begin(std::array<std::uint64_t,1>{900}));
    ASSERT_NO_THROW(pool.wave<false>());
    EXPECT_FALSE(pool.view().frame->free_exhausted);
    ASSERT_NO_THROW(pool.wave<true>());
    EXPECT_EQ(pool.view().frame->victim_planned,0u);
    ASSERT_NO_THROW(pool.finish());
}
TEST(PooledImmutableNative, LateMalformedOrDuplicatePinCannotAcquireAnUnreportedPrefix)
{
    for(unsigned defect=0;defect<3;++defect){
        Pool pool(std::array{2u});
    ASSERT_NO_THROW(pool.complete(std::array<std::uint64_t,2>{101,102}));
        auto& owner=*pool.owners[0];auto& x=owner.args;
        auto view=p::CoordinatorOperations(pool.view());const std::array<std::uint64_t,2> rows{101,102};
        ASSERT_TRUE(view.begin(2,rows.data(),rows.size()));n::invocation(pool.args);x.input={pool.invocation(),nullptr,nullptr,nullptr,nullptr,nullptr};
    ASSERT_TRUE(owner.run<n::OwnerAction::Begin>());
        std::array<p::OwnerCommand,2> commands;
    ASSERT_TRUE(view.pinCommand(0,commands[0]));
    ASSERT_TRUE(view.pinCommand(1,commands[1]));
        if(defect==0)++commands[1].target.generation;
        if(defect==1){commands[1].target=commands[0].target;commands[1].requested_row=commands[0].requested_row;}
        if(defect==2)commands[1].ordinal=commands[0].ordinal;
        const std::uint64_t bytes=sizeof(commands);x.input.records=commands.data();x.input.record_bytes=&bytes;
        EXPECT_FALSE(owner.run<n::OwnerAction::Pin>());
    EXPECT_EQ(n::ownerView<a::Slot64>(x).frame->held_pins,0u);
        EXPECT_EQ(n::ownerView<a::Slot64>(x).frame->initial_pin_count,0u);
    EXPECT_EQ(n::progress(x).reply_bytes,0u);
        for(std::uint32_t i=0;i<2;++i)EXPECT_FALSE(a::PhysicalSlot<a::Slot64>::pinned(n::ownerView<a::Slot64>(x).slots[i]));
    }
}
TEST(PooledImmutableNative, StaleInvocationFailsBeforeAnyPhysicalMutation)
{
    for(unsigned field=0;field<5;++field){Pool pool(std::array{1u,0u,2u});
    ASSERT_NO_THROW(pool.begin(std::array<std::uint64_t,1>{31}));
        auto& owner=*pool.owners[0];auto forged=*pool.invocation();
        if(field==0)++forged.version;if(field==1)++forged.source_storage_generation;if(field==2)++forged.interval.table;
        if(field==3)++forged.interval.sequence;if(field==4)forged.unique_rows=17;
        owner.args.input.invocation=&forged;const auto before=*n::ownerView<a::Slot64>(owner.args).owner;
        EXPECT_FALSE(owner.run<n::OwnerAction::Availability>());
    EXPECT_EQ(n::progress(owner.args).header_bytes,0u);
        EXPECT_EQ(std::memcmp(&before,n::ownerView<a::Slot64>(owner.args).owner,sizeof(before)),0);
    }
}
TEST(PooledImmutableNative, ExactDeclaredNativeGeometryAndClosedC0InputSchemas)
{
    Pool pool(std::array{1u,0u,2u,0u,1u});
    for(auto& owner:pool.owners){const auto& plan=owner->args.plan;
    EXPECT_EQ(plan.native.bytes,128+64*plan.native.reply_capacity);
        const auto pin=q::pooledNativeSchema(n::OwnerAction::Pin,plan),fill=q::pooledNativeSchema(n::OwnerAction::ApplySourceDelivery,plan);
        EXPECT_EQ(pin[static_cast<unsigned>(q::PooledNativeOperand::Records)].bytes,plan.native.reply_capacity*sizeof(p::OwnerCommand));
        EXPECT_EQ(fill[static_cast<unsigned>(q::PooledNativeOperand::Payload)].bytes,plan.native.reply_capacity*plan.row_bytes);
        const auto retire=q::pooledNativeSchema(n::OwnerAction::Retire,plan);
    EXPECT_EQ(retire[static_cast<unsigned>(q::PooledNativeOperand::Frontier)].bytes,0u);
    }
    EXPECT_THROW(q::pooledNativeSchema(n::CoordinatorAction::PlanFree,pool.args.plan,n::OriginalRows{DeviceRowRange::fullyActive(1),n::OriginalIdWidth::UInt64}),std::invalid_argument);
    EXPECT_THROW(q::pooledNativeSchema(n::CoordinatorAction::Begin,pool.args.plan,std::nullopt),std::invalid_argument);
    EXPECT_EQ(pool.args.plan.native.radix_blocks,1u);
}
TEST(PooledImmutableNative, AcceptedFreeCancellationDrainsActualGrantsAndSortsWithoutOverwritingReceipts)
{
    Pool pool(std::array{2u,0u,2u});
    ASSERT_NO_THROW(pool.begin(std::array<std::uint64_t,3>{101,102,103}));
    ASSERT_NO_THROW(pool.all<n::OwnerAction::Availability>());
    const auto available=pool.headers<n::Availability>();pool.args.input.frontier=available.data();
    ASSERT_NO_THROW(pool.run<n::CoordinatorAction::PlanFree>());pool.args.input={};
    for(std::size_t i=0;i<pool.owners.size();++i){auto& owner=*pool.owners[i];auto& x=owner.args;
        const auto slice=n::at<n::DirectedSlice>(pool.args.native,pool.args.plan.native.free_slices)[i];
        x.input.frontier=n::at<n::FreePlanHeader>(pool.args.native,pool.args.plan.native.free_headers)+i;
        x.input.records=x.plan.native.reply_capacity?pool.args.frame+slice.byte_offset:nullptr;
        ASSERT_TRUE(owner.run<n::OwnerAction::AcceptFree>());
        x.input.records=nullptr;x.input.frontier=nullptr;
        ASSERT_TRUE(owner.run<n::OwnerAction::CancelFree>());
        EXPECT_EQ(n::progress(x).reply_bytes,std::uint64_t{n::allocationFrame(x).count}*sizeof(p::Grant));
    }
    ASSERT_NO_THROW(pool.grants<n::CoordinatorAction::AcquireFree>());
    const auto complete=pool.headers<p::FreeAllocationReceipt>();pool.args.input.frontier=complete.data();
    ASSERT_NO_THROW(pool.run<n::CoordinatorAction::CompleteFree>());pool.args.input={};
    ASSERT_NO_THROW(pool.all<n::OwnerAction::Release>());
    std::vector<p::ReleaseReceipt> releases;
    for(auto& owner:pool.owners){auto& x=owner->args;const auto count=n::progress(x).released;
        const auto* receipts=n::at<p::ReleaseReceipt>(x.frame,x.plan.frame.release_receipts);releases.insert(releases.end(),receipts,receipts+count);}
    std::reverse(releases.begin(),releases.end());const auto original=releases;
    const n::SourceFrontier source{pool.view().frame->interval,{},n::SourceDisposition::Unpublished,a::Status::Success,n::Version};
    const std::uint64_t bytes=releases.size()*sizeof(p::ReleaseReceipt);pool.args.input={releases.data(),&bytes,nullptr,&source};
    ASSERT_NO_THROW(pool.run<n::CoordinatorAction::Cancel>());pool.args.input={};
    EXPECT_EQ(std::memcmp(original.data(),releases.data(),bytes),0);
    ASSERT_NO_THROW(pool.all<n::OwnerAction::Retire>());
    const auto retired=pool.headers<p::RetiredHeader>();pool.args.input.frontier=retired.data();
    ASSERT_NO_THROW(pool.run<n::CoordinatorAction::Finish>());
}
TEST(PooledImmutableNative, LateSourceCorruptionCannotPublishAcknowledgementsAndReadyRequiresActualCopy)
{
    for(bool corrupt:{false,true}){
        Pool pool(std::array{2u});
        ASSERT_NO_THROW(pool.begin(std::array<std::uint64_t,2>{101,102}));
        ASSERT_NO_THROW(pool.wave<false>());
        ASSERT_NO_THROW(pool.wave<true>());
        n::SourceFrontier source{pool.view().frame->interval,{701,17,1},n::SourceDisposition::Published,a::Status::Success,n::Version};
        pool.args.input.source=&source;
        ASSERT_NO_THROW(pool.run<n::CoordinatorAction::SealSource>());
        auto& owner=*pool.owners[0];auto& x=owner.args;x.input.frontier=&source;
        ASSERT_TRUE(owner.run<n::OwnerAction::SealSource>());
        source.disposition=n::SourceDisposition::Consumed;
        std::array<n::Fill,2> fills;
        for(unsigned i=0;i<2;++i)fills[i]={n::grants(x)[i],{source.source,n::grants(x)[i].destination_token}};
        if(corrupt)++fills[1].acknowledgement.destination_token;
        const auto count=n::allocationFrame(x).count;
        ASSERT_EQ(count,2u);
        const auto* acknowledgements=n::at<p::FillAcknowledgement>(x.frame,x.plan.frame.acknowledgements);
        std::array<p::FillAcknowledgement,2> before;std::memcpy(before.data(),acknowledgements,sizeof(before));
        const std::uint64_t bytes=sizeof(fills),payload_bytes=2*x.plan.row_bytes;
        std::vector<std::byte> payload(payload_bytes);x.input.records=fills.data();x.input.record_bytes=&bytes;x.input.payload=payload.data();x.input.payload_bytes=&payload_bytes;
        if(corrupt){EXPECT_FALSE(owner.run<n::OwnerAction::ApplySourceDelivery>());
            EXPECT_EQ(std::memcmp(before.data(),acknowledgements,sizeof(before)),0);
            EXPECT_EQ(n::progress(x).delivery,n::Delivery::None);
        }else{ASSERT_TRUE(owner.run<n::OwnerAction::ApplySourceDelivery>());
            EXPECT_EQ(n::progress(x).delivery,n::Delivery::Validated);
            EXPECT_FALSE(owner.run<n::OwnerAction::ValidateFills>());
        }
        for(unsigned i=0;i<count;++i)EXPECT_EQ(a::PhysicalSlot<a::Slot64>::state(n::ownerView<a::Slot64>(x).slots[n::grants(x)[i].slot]),a::SlotState::Reserved);
    }
}
TEST(PooledImmutableNative, ClosedProducerSpansRejectUnusedInvalidRoleAndOverlappingSourceDeclarations)
{
    Pool pool(std::array{0u});const auto& plan=pool.owners[0]->plan;
    auto root=std::make_shared<INT8Tensor>(std::vector<std::size_t>{128});
    q::PooledNativeOperandSpans spans;const auto invocation=static_cast<unsigned>(q::PooledNativeOperand::Invocation);
    spans[invocation]=q::PooledNativeTensorSpan{root,BufferId::QFN_NGRAM_LOOKUP_STATE,0,sizeof(n::Invocation)};
    EXPECT_NO_THROW(q::PooledImmutableOwnerOperands(n::OwnerAction::Pin,plan,spans));
    q::PooledImmutableOwnerOperands declared(n::OwnerAction::Pin,plan,spans);
    EXPECT_THROW(declared.captureIdentity(),std::logic_error);
    EXPECT_THROW(declared.seal(),std::logic_error); // Missing GPU address rejects before backend/runtime discovery.

    auto invalid=spans;invalid[invocation]->buffer=static_cast<BufferId>(static_cast<unsigned>(BufferId::_COUNT)+1);
    EXPECT_THROW(q::PooledImmutableOwnerOperands(n::OwnerAction::Pin,plan,invalid),std::invalid_argument);
    auto extra=spans;extra[static_cast<unsigned>(q::PooledNativeOperand::Payload)]=q::PooledNativeTensorSpan{root,BufferId::QFN_NGRAM_LOOKUP_STATE,64,1};
    EXPECT_THROW(q::PooledImmutableOwnerOperands(n::OwnerAction::Pin,plan,extra),std::invalid_argument);
    auto overlap=spans;overlap[static_cast<unsigned>(q::PooledNativeOperand::Frontier)]=q::PooledNativeTensorSpan{root,BufferId::QFN_NGRAM_LOOKUP_STATE,0,sizeof(n::SourceFrontier)};
    EXPECT_THROW(q::PooledImmutableOwnerOperands(n::OwnerAction::SealSource,plan,overlap),std::invalid_argument);
}

TEST(PooledImmutableNative, NativeCapturedPlanCancellationTraceRetiresUnacceptedSmallAndEmptyOwnersAcrossReset)
{
    for(const auto slots:{0u,2u}){
        Pool pool(std::array{slots},8192);
        for(unsigned replay=0;replay<4;++replay){
            std::vector<std::uint64_t> original(replay%2?1:8192);
            for(std::size_t i=0;i<original.size();++i)original[i]=31+i%3;
            ASSERT_NO_THROW(pool.begin(original));
            ASSERT_NO_THROW(pool.all<n::OwnerAction::Availability>());
            const auto availability=pool.headers<n::Availability>();pool.args.input.frontier=availability.data();
            ASSERT_NO_THROW(pool.run<n::CoordinatorAction::PlanFree>());pool.args.input={};
            ASSERT_NO_THROW(pool.all<n::OwnerAction::CancelFree>());
            const auto completed=pool.headers<p::FreeAllocationReceipt>();pool.args.input.frontier=completed.data();
            ASSERT_NO_THROW(pool.run<n::CoordinatorAction::CompleteFree>());pool.args.input={};
            ASSERT_NO_THROW(pool.all<n::OwnerAction::Release>());
            ASSERT_NO_THROW(pool.all<n::OwnerAction::Retire>());
            const auto retired=pool.headers<p::RetiredHeader>();
            const n::SourceFrontier source{pool.view().frame->interval,{},n::SourceDisposition::Unpublished,a::Status::Success,n::Version};
            const std::uint64_t bytes=0;pool.args.input={nullptr,slots?&bytes:nullptr,nullptr,&source};
            ASSERT_NO_THROW(pool.run<n::CoordinatorAction::Cancel>());pool.args.input={};
            pool.args.input.frontier=retired.data();
            ASSERT_NO_THROW(pool.run<n::CoordinatorAction::Finish>());pool.args.input={};
            if(slots)EXPECT_EQ(n::ownerView<a::Slot64>(pool.owners[0]->args).owner->filled,0u);
            EXPECT_EQ(pool.view().frame->phase,p::LookupPhase::Idle);
        }
    }
}

TEST(PooledImmutableNative, NativeHeadersPublishOnlyTheirActualBytesAndPreserveUncommunicatedPadding)
{
    Pool pool(std::array{0u,2u,0u});
    ASSERT_NO_THROW(pool.begin(std::array<std::uint64_t,1>{31}));
    ASSERT_NO_THROW(pool.all<n::OwnerAction::Availability>());
    for(auto& owner:pool.owners){auto& x=owner->args;
        EXPECT_EQ(n::progress(x).header_bytes,sizeof(n::Availability));
        for(std::size_t i=sizeof(n::Availability);i<64;++i)EXPECT_EQ(x.native[x.plan.native.header+i],std::byte{0xcc});
        const auto header=owner->header<n::Availability>();EXPECT_EQ(header.lease.owner,x.plan.owner.coordinate());
        EXPECT_EQ(header.lease.free_slots,x.plan.owner.capacity());EXPECT_EQ(header.version,n::Version);
    }
}

TEST(PooledImmutableNative, SuccessfulNativeSourceTraceHasActualFreeVictimAndLaterHitDecisionsAcrossRequestReset)
{
    for(const auto maximum:{32u,32768u}){
        Pool pool(std::array{4u},maximum);
        ASSERT_NO_THROW(pool.complete(std::array<std::uint64_t,2>{101,102}));
        std::vector<std::uint64_t> original(maximum,900);original[0]=700;original[1]=701;
        ASSERT_NO_THROW(pool.begin(original));
        ASSERT_NO_THROW(pool.wave<false>());
        EXPECT_EQ(pool.view().frame->free_planned,2u);
        ASSERT_NO_THROW(pool.wave<true>());
        EXPECT_EQ(pool.view().frame->victim_planned,1u);
        const auto hot=pool.view().unique[2].grant;
        ASSERT_NO_THROW(pool.finish());
        ASSERT_NO_THROW(pool.complete(std::array<std::uint64_t,1>{900},2));
        EXPECT_EQ(pool.view().unique[0].phase,p::UniquePhase::Read);
        EXPECT_EQ(pool.view().unique[0].grant.generation,hot.generation);
        ASSERT_NO_THROW(pool.complete({},2));
        ASSERT_NO_THROW(pool.complete(original,2));
        EXPECT_EQ(pool.view().frame->free_planned,0u);EXPECT_EQ(pool.view().frame->victim_planned,0u);
        for(unsigned i=0;i<3;++i)EXPECT_EQ(pool.view().unique[i].phase,p::UniquePhase::Read);
    }
}

TEST(PooledImmutableNative, ZeroCountReleaseStillRequiresActualSourceOrCancellationBoundary)
{
    for(const auto slots:{0u,2u}){
        Pool pool(std::array{slots});
        ASSERT_NO_THROW(pool.begin({}));
        auto& owner=*pool.owners[0];
        EXPECT_FALSE(owner.run<n::OwnerAction::Release>());
        EXPECT_EQ(n::progress(owner.args).released,0u);
        EXPECT_EQ(n::progress(owner.args).release_bytes,0u);
    }
}

TEST(PooledImmutableNative, ImmutableRadixBoundsCoverEveryKeyAndPreserveBankParity)
{
    for(const auto bound:std::array<std::uint64_t,10>{1,2,16,17,512,1024,32768,65536,0x100000100ULL,~std::uint64_t{0}}){
        const auto passes=n::radixPasses(bound);
        EXPECT_GE(passes,2u);EXPECT_LE(passes,16u);EXPECT_EQ(passes%2,0u);
        if(passes<16)EXPECT_LT(bound-1,std::uint64_t{1}<<(passes*4));
        if(passes>2)EXPECT_GE(bound-1,std::uint64_t{1}<<((passes-2)*4));
    }
    EXPECT_EQ(n::radixPasses(1024),4u);EXPECT_EQ(n::radixPasses(32768),4u);EXPECT_EQ(n::radixPasses(0x100000100ULL),10u);
}

TEST(PooledImmutableNative, OriginalProducerWidthsOwnExactCompactBytesAndCannotChangeMetadataPhases)
{
    Pool pool(std::array{2u});
    for(const auto width:{n::OriginalIdWidth::UInt32,n::OriginalIdWidth::UInt64}){
        const auto schema=q::pooledNativeSchema(n::CoordinatorAction::Begin,pool.args.plan,n::OriginalRows{DeviceRowRange::fullyActive(32),width});
        const auto requirement=schema[static_cast<unsigned>(q::PooledNativeOperand::Ids)];
        EXPECT_EQ(requirement.bytes,32u*(width==n::OriginalIdWidth::UInt32?4u:8u));
        EXPECT_EQ(requirement.alignment,width==n::OriginalIdWidth::UInt32?4u:8u);
    }
    EXPECT_THROW(q::pooledNativeSchema(n::CoordinatorAction::Begin,pool.args.plan,n::OriginalRows{DeviceRowRange::fullyActive(32),static_cast<n::OriginalIdWidth>(42)}),std::invalid_argument);
}

TEST(PooledImmutableNative, ParallelStableRunsMatchEveryScalarFrequencyCellAcrossAgingSaturationAndCollisions)
{
    for(const auto aging:std::array<std::uint64_t,5>{1,2,3,17,65536})
    for(unsigned pattern=0;pattern<4;++pattern){
        Pool actual(std::array{4u,0u,4u},2048,{aging,1,2}),reference(std::array{4u,0u,4u},2048,{aging,1,2});
        for(auto* pool:{&actual,&reference}){
            auto v=pool->view();v.policy->logical_clock=aging*2-1;
            for(std::uint32_t at=0;at<4*v.parameters.laneWidth();++at){
                v.frequency_epochs[at]=at%3;v.frequency_counts[at]=static_cast<std::uint16_t>(at%2?65534:at*3071);
            }
        }
        std::vector<std::uint64_t> original(2048);
        for(std::uint32_t pos=0;pos<original.size();++pos)original[pos]=pattern==0?31:pattern==1?31+pos%3:pattern==2?31+(pos/257)%7:31+((pos*1664525u+1013904223u)>>16)%11;
        ASSERT_NO_THROW(actual.begin(original));
        ASSERT_NO_THROW(reference.begin(original));
        ASSERT_NO_THROW(actual.all<n::OwnerAction::Availability>());
        ASSERT_NO_THROW(reference.all<n::OwnerAction::Availability>());
        const auto actualReceipts=actual.headers<n::Availability>(),referenceReceipts=reference.headers<n::Availability>();
        actual.args.input.frontier=actualReceipts.data();
        ASSERT_NO_THROW(actual.run<n::CoordinatorAction::PlanFree>());
        std::vector<p::AvailabilityLease> leases;for(const auto& receipt:referenceReceipts)leases.push_back(receipt.lease);
        ASSERT_TRUE(p::CoordinatorOperations(reference.view()).planFree(leases.data(),leases.size()));
        const auto a=actual.view(),b=reference.view();
        EXPECT_EQ(std::memcmp(a.policy,b.policy,sizeof(*a.policy)),0)<<"aging="<<aging<<" pattern="<<pattern;
        EXPECT_EQ(std::memcmp(a.frequency_epochs,b.frequency_epochs,4*a.parameters.laneWidth()*sizeof(std::uint32_t)),0);
        EXPECT_EQ(std::memcmp(a.frequency_counts,b.frequency_counts,4*a.parameters.laneWidth()*sizeof(std::uint16_t)),0);
        EXPECT_EQ(std::memcmp(a.frame,b.frame,sizeof(*a.frame)),0);
        EXPECT_EQ(std::memcmp(a.unique,b.unique,a.frame->unique_rows*sizeof(p::CoordinatorUnique)),0);
    }
}
TEST(PooledImmutableNative, MalformedRunPartitionRejectsBeforeAnyFrequencyCellOrClockMutation)
{
    Pool pool(std::array{2u},512);
    ASSERT_NO_THROW(pool.begin(std::array<std::uint64_t,3>{31,31,31}));
    ASSERT_NO_THROW(pool.all<n::OwnerAction::Availability>());
    const auto receipts=pool.headers<n::Availability>();pool.args.input.frontier=receipts.data();
    auto v=pool.view();const auto before=*v.policy;
    v.sort_a[0]={31,0,2};v.sort_b[0].unique=1; // Authored three-position partition has a missing final observation.
    EXPECT_FALSE(n::coordinator<n::CoordinatorAction::PlanFree>(pool.args));
    EXPECT_EQ(v.policy->logical_clock,before.logical_clock);
    for(std::uint32_t i=0;i<4*v.parameters.laneWidth();++i){EXPECT_EQ(v.frequency_counts[i],0);EXPECT_EQ(v.frequency_epochs[i],0);}
}

TEST(PooledImmutableNative, StableRunQueuePolicyMatchesOriginalScalarOrderAcrossPromotionDemotionAndRepeatBoundaries)
{
    // Golden records come from the original scalar applyQueues operation over
    // the same actual source/owner replay. They hash explicit policy fields,
    // including original logical clocks, never padding, host addresses or allocation observations.
    const std::array<std::uint64_t,12> expected{6307381114020998531ULL,14484222756488971030ULL,7673470971267910682ULL,2598223876836132572ULL,3322370960222685971ULL,8347933099706359158ULL,6243837180883191290ULL,902579556877491292ULL,3322370960222685971ULL,8347933099706359158ULL,6243837180883191290ULL,902579556877491292ULL};
    std::size_t index=0;
    for(const auto protect:std::array<unsigned,3>{0,1,2})for(unsigned pattern=0;pattern<4;++pattern){
        Pool pool(std::array{4u,0u,4u},2048,{17,protect,2});
        ASSERT_NO_THROW(pool.complete(std::array<std::uint64_t,4>{31,32,33,34}));
        for(unsigned replay=0;replay<3;++replay){
            std::vector<std::uint64_t> original(2048);
            for(std::uint32_t pos=0;pos<original.size();++pos)original[pos]=31+(pattern==0?replay%4:pattern==1?(pos+replay)%4:pattern==2?(pos/257+replay)%4:((pos*1664525u+1013904223u)>>16)%4);
            ASSERT_NO_THROW(pool.complete(original,replay+2));
        }
        const auto v=pool.view();std::uint64_t hash=1469598103934665603ULL;const auto mix=[&](std::uint64_t value){hash=(hash^value)*1099511628211ULL;};
        mix(v.policy->logical_clock);mix(v.policy->directory_root);mix(v.policy->node_free);mix(v.policy->protected_count);mix(v.policy->flags);
        for(unsigned list=0;list<3;++list){mix(v.policy->head[list]);mix(v.policy->tail[list]);mix(v.frame->eligible_cursor[list]);}
        for(std::uint32_t at=0;at<v.parameters.entries();++at){const auto& e=v.entries[at];
            mix(e.locator.row);mix(e.locator.generation);mix(e.locator.owner);mix(e.locator.slot);mix(e.touched);mix(e.observed_frame);mix(e.previous);mix(e.next);mix(static_cast<unsigned>(e.stream));mix(e.present);}
        for(std::uint32_t at=0;at<4*v.parameters.laneWidth();++at){mix(v.frequency_epochs[at]);mix(v.frequency_counts[at]);}
        EXPECT_EQ(hash,expected[index])<<"scalar_queue_record="<<index; ++index;
    }
}

TEST(PooledImmutableNative, SourceOutcomeGatesKeepActualCompletionAndWorkerAbortDistinct)
{
    const p::Interval interval{701,3,8};const p::SourceWindow source{701,17,11};
    n::SourceFrontier outcome{interval,source,n::SourceDisposition::Consumed,a::Status::Success,n::Version};
    EXPECT_EQ(n::sourceGate(n::SourceCondition::AfterSuccess,&outcome,interval,701,source),n::SourceGate::Run);
    EXPECT_EQ(n::sourceGate(n::SourceCondition::AfterFailure,&outcome,interval,701,source),n::SourceGate::Skip);
    outcome.status=a::Status::SourceFailure;
    EXPECT_EQ(n::sourceGate(n::SourceCondition::AfterSuccess,&outcome,interval,701,source),n::SourceGate::Skip);
    EXPECT_EQ(n::sourceGate(n::SourceCondition::AfterFailure,&outcome,interval,701,source),n::SourceGate::Run);
    outcome.disposition=n::SourceDisposition::Aborted;
    EXPECT_EQ(n::sourceRetirement(outcome.disposition),p::SourceRetirement::Aborted);
    EXPECT_EQ(n::sourceGate(n::SourceCondition::AfterFailure,&outcome,interval,701,source),n::SourceGate::Run);
    outcome.status=a::Status::Success;
    EXPECT_EQ(n::sourceGate(n::SourceCondition::AfterSuccess,&outcome,interval,701,source),n::SourceGate::Invalid);
    outcome.status=a::Status::Cancelled;outcome.disposition=n::SourceDisposition::Published;
    EXPECT_EQ(n::sourceGate(n::SourceCondition::AfterFailure,&outcome,interval,701,source),n::SourceGate::Invalid);
    outcome.disposition=n::SourceDisposition::Aborted;++outcome.source.sequence;
    EXPECT_EQ(n::sourceGate(n::SourceCondition::AfterFailure,&outcome,interval,701,source),n::SourceGate::Invalid);
    EXPECT_EQ(n::sourceGate(n::SourceCondition::Always,&outcome,interval,701,source),n::SourceGate::Invalid);
    EXPECT_EQ(n::sourceGate(n::SourceCondition::Always,nullptr,interval,701,source),n::SourceGate::Run);
}

TEST(PooledImmutableNative, ActualAbortedSourceCancelsWithoutReadyAndRetiresEveryOriginalGrantOnce)
{
    for(const auto capacities:{std::vector{0u},std::vector{2u,0u,2u}}){
        Pool pool(capacities);
        ASSERT_NO_THROW(pool.begin(std::array<std::uint64_t,4>{31,32,33,31}));
        ASSERT_NO_THROW(pool.wave<false>());
    ASSERT_NO_THROW(pool.wave<true>());
        n::SourceFrontier source{pool.view().frame->interval,{701,17,1},n::SourceDisposition::Published,a::Status::Success,n::Version};
        pool.args.input.source=&source;
    ASSERT_NO_THROW(pool.run<n::CoordinatorAction::SealSource>());pool.args.input={};
        std::vector<std::vector<p::Grant>> originals;
        for(auto& owner:pool.owners){
            auto& x=owner->args;x.input.frontier=&source;ASSERT_TRUE(owner->run<n::OwnerAction::SealSource>());
            originals.emplace_back(n::grants(x),n::grants(x)+n::allocationFrame(x).count);
        }
        source.disposition=n::SourceDisposition::Aborted;source.status=a::Status::Cancelled;
        for(auto& owner:pool.owners){
            auto& x=owner->args;x.input.condition=n::SourceCondition::AfterSuccess;x.input.outcome=&source;
            ASSERT_TRUE(owner->run<n::OwnerAction::MakeReady>());EXPECT_EQ(n::progress(x).header_bytes,0);
            EXPECT_EQ(n::ownerView<a::Slot64>(x).frame->phase,p::OwnerPhase::Filling);
        }
        pool.args.input.source=&source;pool.args.input.outcome=&source;pool.args.input.condition=n::SourceCondition::AfterSuccess;
        ASSERT_TRUE(n::coordinator<n::CoordinatorAction::Publish>(pool.args));
        EXPECT_EQ(pool.view().frame->phase,p::LookupPhase::Source);pool.args.input={};
        std::vector<p::ReleaseReceipt> receipts;
        for(std::size_t index=0;index<pool.owners.size();++index){
            auto& owner=*pool.owners[index];auto& x=owner.args;
            x.input.frontier=&source;x.input.outcome=&source;x.input.condition=n::SourceCondition::AfterFailure;
            ASSERT_TRUE(owner.run<n::OwnerAction::CancelSource>());
            if(!originals[index].empty())EXPECT_EQ(std::memcmp(originals[index].data(),n::grants(x),originals[index].size()*sizeof(p::Grant)),0);
            x.input={pool.invocation(),nullptr,nullptr,nullptr,nullptr,nullptr};ASSERT_TRUE(owner.run<n::OwnerAction::Release>());
            const auto count=n::progress(x).release_bytes/sizeof(p::ReleaseReceipt);
            const auto* actual=n::at<p::ReleaseReceipt>(x.frame,x.plan.frame.release_receipts);
            receipts.insert(receipts.end(),actual,actual+count);
            EXPECT_EQ(count,originals[index].size());
        }
        const std::uint64_t bytes=receipts.size()*sizeof(p::ReleaseReceipt);
        pool.args.input={receipts.empty()?nullptr:receipts.data(),pool.args.plan.native.reply_capacity?&bytes:nullptr,nullptr,&source,n::SourceCondition::AfterFailure,&source};
        ASSERT_EQ(n::sourceGate(pool.args.input.condition,pool.args.input.outcome,pool.view().frame->interval,pool.args.plan.table,pool.view().frame->source),n::SourceGate::Run);
        ASSERT_TRUE(n::source(pool.args,true,true));
        std::uint32_t acquiredCount=0;ASSERT_TRUE(n::recordCount<p::ReleaseReceipt>(pool.args,acquiredCount));
        ASSERT_TRUE(n::coordinator<n::CoordinatorAction::Cancel>(pool.args))<<"owners="<<capacities.size()<<" receipts="<<receipts.size()<<" status="<<static_cast<unsigned>(pool.view().frame->status);pool.args.input={};
        ASSERT_NO_THROW(pool.all<n::OwnerAction::Retire>());
        const auto retired=pool.headers<p::RetiredHeader>();pool.args.input.frontier=retired.data();
    ASSERT_NO_THROW(pool.run<n::CoordinatorAction::Finish>());pool.args.input={};
        EXPECT_EQ(pool.view().frame->phase,p::LookupPhase::Idle);
        for(auto& owner:pool.owners){EXPECT_EQ(n::ownerView<a::Slot64>(owner->args).frame->held_pins,0u);EXPECT_EQ(n::ownerView<a::Slot64>(owner->args).frame->reservations,0u);}
        pool.source_sequence=1;
    ASSERT_NO_THROW(pool.complete(std::array<std::uint64_t,1>{99},2));
    }
}

TEST(PooledImmutableNative, DirectedAllocationViewsPreserveOriginalBanksAcrossNonzeroVictimOffset)
{
    Pool pool(std::array{4u});
    ASSERT_NO_THROW(pool.complete(std::array<std::uint64_t,2>{101,102}));
    std::vector<std::uint64_t> original{700,701};original.insert(original.end(),12,900);
    ASSERT_NO_THROW(pool.begin(original));
    ASSERT_NO_THROW(pool.wave<false>(true));
    ASSERT_EQ(pool.view().frame->free_planned,2u);
    const auto& owner=*pool.owners[0];const auto free=n::allocationFrame(owner.args).count;
    std::vector<p::Grant> prefix(n::grants(owner.args),n::grants(owner.args)+free);
    ASSERT_NO_THROW(pool.wave<true>(true));EXPECT_EQ(pool.view().frame->victim_planned,1u);
    const auto slice=n::at<n::DirectedSlice>(pool.args.native,pool.args.plan.native.victim_slices)[0];
    EXPECT_GT(slice.byte_offset,pool.args.plan.frame.decisions);
    EXPECT_EQ(std::memcmp(prefix.data(),n::grants(owner.args),prefix.size()*sizeof(p::Grant)),0);
    ASSERT_NO_THROW(pool.finish());
}

TEST(PooledImmutableNative, IdentityFreeNativeBOMMatchesActualSourceBoundMaterializationGeometry)
{
    Pool pool(std::array{2u,0u,3u});
    const auto resource=[](PhysicalMemoryAllocatorIdentity id){return PhysicalMemoryResource{id.world_rank,id.device,1ull<<30,1ull<<30};};
    for(std::uint32_t coordinate=0;coordinate<pool.owners.size();++coordinate){
        const auto& native=pool.owners[coordinate]->plan;const auto observed=resource(native.allocator());
        EXPECT_EQ(q::PooledImmutableOwnerNativePlan::mandatoryBOM(pool.physical,pool.geometry,coordinate,observed).totalBytes(),native.mandatoryBOM(observed).totalBytes());
    }
    const auto observed=resource(pool.plan.allocator());
    EXPECT_EQ(q::PooledImmutableCoordinatorNativePlan::mandatoryBOM(pool.physical,pool.geometry,observed).totalBytes(),pool.plan.mandatoryBOM(observed).totalBytes());
}


/** @brief Actual source parsers reject stale envelopes and partial/malformed ACKs before producing any accepted fill. */
TEST(PooledImmutableNative, ActualSourceCompletionAndPartialAcknowledgementsPreserveOriginalGrantBanks)
{
    namespace source=q::pooled_source;
    for(const auto capacity:{0u,4u})for(const auto corrupt:{-1,0,2}){
        Pool pool(std::array{capacity});
        ASSERT_NO_THROW(pool.complete(std::array<std::uint64_t,2>{101,102}));
        std::vector<std::uint64_t> original{700,701};original.insert(original.end(),12,900);
        ASSERT_NO_THROW(pool.begin(original));
        ASSERT_NO_THROW(pool.wave<false>());
        ASSERT_NO_THROW(pool.wave<true>());
        const ImmutableRowTableSource descriptor{.path="cpu_parser_original_source",.tensor_name="per_layer_token_embd.weight",
            .file_offset=37,.rows=0x100000100ULL,.row_elements=160,.format=GGUFTensorType::IQ4_NL};
        const auto geometry=q::PooledImmutableSourceGeometry::fromPlans(pool.owners.front()->plan,pool.plan,descriptor);
        const auto parameters=geometry.parameters();Bytes frame(parameters.offsets.frame_bytes),native(parameters.offsets.native_bytes);
        std::memset(native.data(),0,parameters.offsets.native_bytes);
        std::array<std::int64_t,16> ids{};std::array<ImmutableRowSubordinateIntent,16> intents{};
        std::array<ImmutableRowInvalidationAcknowledgement,16> acknowledgements{};
        int healthy=1,error=0;
        source::DeviceArgs args{parameters,pool.owners.front()->args,pool.args,frame.data(),native.data(),
            {{nullptr,nullptr,nullptr,ids.data(),nullptr,701,16,90},intents.data(),acknowledgements.data()},&healthy,&error};
        source::protocol(args).admissionGeneration=17;source::invocation(args)=23;
        ASSERT_TRUE(source::planRequest(args));ASSERT_EQ(source::protocol(args).idCount,3u);
        EXPECT_EQ((std::array{ids[0],ids[1],ids[2]}),(std::array<std::int64_t,3>{700,701,900}));
        const auto* grants=n::grants(args.owner);const auto count=n::allocationFrame(args.owner).count;
        const std::vector<p::Grant> immutable(grants,grants+count);
        for(unsigned i=0;i<3;++i){
            const auto& unique=pool.view().unique[i];
            EXPECT_EQ(intents[i].intent,source::reserved(unique.phase)?ImmutableRowReadIntent::ReadAndInvalidate:ImmutableRowReadIntent::Read);
            EXPECT_EQ(intents[i].destination_grant,source::reserved(unique.phase)?unique.grant.destination_token:0u);
        }
        pool.args.input.source=&source::frontier(args);ASSERT_NO_THROW(pool.run<n::CoordinatorAction::SealSource>());pool.args.input={};
        args.owner.input.frontier=&source::frontier(args);ASSERT_TRUE((n::owner<n::OwnerAction::SealSource,a::Slot64>(args.owner)));
        auto& io=source::protocol(args);io.issuedSequence=source::frontier(args).source.sequence;
        q::ImmutableLookupCompletion completion;completion.publication=io.issuedSequence;completion.sequenceComplement=~completion.publication;
        completion.generation=io.admissionGeneration;completion.generationComplement=~completion.generation;
        completion.table=701;completion.tableComplement=~completion.table;completion.idCount=io.idCount;completion.idCountComplement=~completion.idCount;
        completion.nativeBytes=io.idCount*90;completion.nativeBytesComplement=~completion.nativeBytes;
        completion.protocol=q::kImmutableLookupSubordinateProtocol;completion.protocolComplement=~completion.protocol;
        ASSERT_TRUE(source::acquiredCompletion(args,completion,completion.publication));
        auto stale=completion;++stale.generation;stale.generationComplement=~stale.generation;
        EXPECT_FALSE(source::acquiredCompletion(args,stale,stale.publication));
        EXPECT_FALSE(source::acquiredCompletion(args,completion,completion.publication-1));
        auto failed=completion;failed.status=static_cast<std::uint64_t>(q::ImmutableLookupStatus::StorageFailure);failed.statusComplement=~failed.status;
        failed.nativeBytes=0;failed.nativeBytesComplement=~failed.nativeBytes;
        EXPECT_TRUE(source::acquiredCompletion(args,failed,failed.publication));
        failed.nativeBytes=90;failed.nativeBytesComplement=~failed.nativeBytes;
        EXPECT_FALSE(source::acquiredCompletion(args,failed,failed.publication));
        io.nativeBytes=completion.nativeBytes;source::stage(args)=source::ResponseStage::Acquired;
        auto* actual=n::at<ImmutableRowInvalidationAcknowledgement>(args.frame,parameters.offsets.invalidations);
        for(unsigned i=0;i<3;++i)actual[i].destination_grant=intents[i].destination_grant;
        if(corrupt>=0)actual[corrupt].destination_grant=actual[corrupt].destination_grant+73;
        std::vector<std::byte> previous(count*sizeof(n::Fill));
        if(!previous.empty())std::memcpy(previous.data(),args.frame+parameters.offsets.fills,previous.size());
        EXPECT_EQ(source::prepareDelivery(args),corrupt<0);
        const auto bytes=*n::at<std::uint64_t>(args.native,parameters.offsets.record_bytes);
        EXPECT_EQ(bytes,corrupt<0?count*sizeof(n::Fill):0u);
        if(corrupt>=0){
            EXPECT_EQ(io.status,static_cast<std::uint64_t>(q::ImmutableLookupStatus::InvalidAcknowledgement));
            if(!previous.empty())EXPECT_EQ(std::memcmp(previous.data(),args.frame+parameters.offsets.fills,previous.size()),0);
        }else{
            const auto* fills=n::at<n::Fill>(args.frame,parameters.offsets.fills);
            for(unsigned i=0;i<count;++i){EXPECT_TRUE(n::same(fills[i].original,immutable[i]));EXPECT_EQ(fills[i].acknowledgement.destination_token,immutable[i].destination_token);}
        }
        if(!immutable.empty())EXPECT_EQ(std::memcmp(immutable.data(),n::grants(args.owner),immutable.size()*sizeof(p::Grant)),0);
        EXPECT_EQ(n::ownerView<a::Slot64>(args.owner).frame->phase,p::OwnerPhase::Filling);
    }
}


/** @brief Early source parse failure seals only a valid fresh lifetime; unresolved previous I/O cannot be repaired. */
TEST(PooledImmutableNative, RejectedSourcePlanSealsFailureWithoutInventingRetirement)
{
    namespace source=q::pooled_source;
    for(const auto capacity:{0u,4u}){
        Pool pool(std::array{capacity});
        ASSERT_NO_THROW(pool.begin(std::array<std::uint64_t,3>{700,701,900}));
        ASSERT_NO_THROW(pool.wave<false>());
        ASSERT_NO_THROW(pool.wave<true>());
        const ImmutableRowTableSource descriptor{.path="cpu_parser_original_source",.tensor_name="per_layer_token_embd.weight",
            .file_offset=37,.rows=0x100000100ULL,.row_elements=160,.format=GGUFTensorType::IQ4_NL};
        const auto geometry=q::PooledImmutableSourceGeometry::fromPlans(pool.owners.front()->plan,pool.plan,descriptor);
        const auto parameters=geometry.parameters();Bytes frame(parameters.offsets.frame_bytes),native(parameters.offsets.native_bytes);
        std::memset(native.data(),0,parameters.offsets.native_bytes);
        std::array<std::int64_t,16> ids{};std::array<ImmutableRowSubordinateIntent,16> intents{};
        std::array<ImmutableRowInvalidationAcknowledgement,16> acknowledgements{};int healthy=1,error=0;
        source::DeviceArgs args{parameters,pool.owners.front()->args,pool.args,frame.data(),native.data(),
            {{nullptr,nullptr,nullptr,ids.data(),nullptr,701,16,90},intents.data(),acknowledgements.data()},&healthy,&error};
        auto& io=source::protocol(args);io.admissionGeneration=17;source::invocation(args)=23;
        const auto count=n::allocationFrame(args.owner).count;const auto* grants=n::grants(args.owner);
        const std::vector<p::Grant> original(grants,grants+count);
        const auto initial=source::frontier(args);io.issuedSequence=1;
        EXPECT_FALSE(source::rejectedPlan(args));EXPECT_EQ(std::memcmp(&initial,&source::frontier(args),sizeof(initial)),0);
        EXPECT_EQ(source::stage(args),source::ResponseStage::Idle);io.issuedSequence=0;
        pool.view().unique[1].first_position=0; // Recorded malformed source input; owner grants remain original.
        ASSERT_FALSE(source::planRequest(args));ASSERT_TRUE(source::rejectedPlan(args));
        EXPECT_EQ(source::stage(args),source::ResponseStage::Rejected);
        EXPECT_EQ(source::frontier(args).disposition,n::SourceDisposition::Published);
        EXPECT_EQ(source::frontier(args).status,a::Status::SourceFailure);
        EXPECT_EQ(io.issuedSequence,0u);EXPECT_EQ(io.consumedSequence,0u); // No completion or request publication is invented.
        pool.args.input.source=&source::frontier(args);ASSERT_NO_THROW(pool.run<n::CoordinatorAction::SealSource>());pool.args.input={};
        args.owner.input.frontier=&source::frontier(args);ASSERT_TRUE((n::owner<n::OwnerAction::SealSource,a::Slot64>(args.owner)));
        EXPECT_EQ(n::sourceGate(n::SourceCondition::AfterFailure,&source::frontier(args),pool.view().frame->interval,701,pool.view().frame->source),n::SourceGate::Invalid);
        if(count)EXPECT_EQ(std::memcmp(original.data(),n::grants(args.owner),count*sizeof(p::Grant)),0);
        EXPECT_EQ(n::ownerView<a::Slot64>(args.owner).frame->reservations,count); // Await actual worker witness before any grant release.
    }
}


/** @brief A source-capacity overflow rejects at the actual coalescing boundary before owner admission or pin issue. */
TEST(PooledImmutableNative, SourceUniqueBoundRejectsOneBeyondBeforeAnyOwnerFrame)
{
    for(const auto capacity:{0u,4u}){
        Pool pool(std::array{capacity},513);std::array<std::uint64_t,17> original;
        for(std::size_t i=0;i<original.size();++i)original[i]=3000+i;
        EXPECT_THROW(pool.begin(original),std::logic_error);
        EXPECT_NE(pool.view().frame->status,a::Status::Success);EXPECT_NE(pool.view().frame->phase,p::LookupPhase::Ready);
        const auto& owner=*pool.owners.front();const auto physical=n::ownerView<a::Slot64>(owner.args).frame;
        EXPECT_EQ(physical->phase,p::OwnerPhase::Idle);EXPECT_EQ(physical->held_pins,0u);EXPECT_EQ(physical->reservations,0u);
        EXPECT_EQ(n::allocationFrame(owner.args).count,0u);EXPECT_EQ(n::allocationFrame(owner.args).phase,p::AllocationPhase::Idle);
        EXPECT_EQ(pool.source_sequence,0u);
    }
}

}
