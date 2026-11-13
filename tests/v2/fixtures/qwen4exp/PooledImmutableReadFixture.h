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
#include "kernels/common/SamplingMath.h"
#include "fixtures/qwen4exp/original_PooledImmutableSourceOwner.inc"
#include "loaders/ImmutableRowTable.h"
#include "loaders/ModelLoader.h"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
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
/**
 * @brief Declare arbitrary owner order with an independent coordinator and CPU consumer.
 * @param capacities Exact slots per original owner, including C0.
 * @param replacement Optional recorded physical-cohort corruption at one coordinate.
 * @return Checked declarations only, without device discovery or allocation.
 */
q::PooledImmutablePhysicalOwnerGeometry declared(std::span<const std::uint32_t> capacities,
    std::optional<std::pair<std::size_t,PhysicalMemoryAllocatorIdentity>> replacement=std::nullopt,
    GGUFTensorType format=GGUFTensorType::IQ4_NL,std::size_t rowBytes=90)
{
    std::vector<PhysicalMemoryAllocatorIdentity> order;
    std::vector<ImmutableRowTableLayoutOwner> owners;
    bool positive=false;
    for(std::uint32_t i=0;i<capacities.size();++i){
        order.push_back({static_cast<int>(i),i%2?DeviceId::rocm(static_cast<int>(i)):DeviceId::cuda(static_cast<int>(i))});
        if(replacement && replacement->first==i)order.back()=replacement->second;
        owners.push_back({{order.back(),ImmutableRowStorage::VRAM},0,capacities[i],{}});positive|=capacities[i]!=0;
    }
    const PhysicalMemoryAllocatorIdentity coordinator{101,DeviceId::rocm(11)};
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
    const auto layout=ImmutableRowTableLayoutGeometry::checked(
        ImmutableRowTableExtent::fromNative(format,0x100000100ULL,rowBytes),
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
    explicit Pool(std::span<const std::uint32_t> capacities,std::uint32_t maximumOriginal=64,a::Policy policy={65536,1,2},std::optional<std::size_t> local=std::nullopt,GGUFTensorType format=GGUFTensorType::IQ4_NL,std::size_t rowBytes=90):
        physical(declared(capacities,local?std::optional(std::pair{*local,PhysicalMemoryAllocatorIdentity{101,DeviceId::rocm(11)}}):std::nullopt,format,rowBytes)),geometry(q::PooledImmutableCoordinatorGeometry::fromOwners(physical,policy,maximumOriginal)),
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
    void begin(std::span<const std::uint64_t> original,std::uint64_t admission=1,bool receivedIngress=false,bool replyIngress=false,std::function<void()> beforeReceivedSeal={}){
        require(p::CoordinatorOperations(view()).begin(admission,original.data(),original.size()));n::invocation(args);
        for(auto& owner:owners){
            const auto* received=invocation();
            if(receivedIngress){
                if(!owner->args.plan.native.ingress)throw std::logic_error("remote fixture requires original ingress declaration");
                auto* ingress=n::at<n::OwnerIngress>(owner->args.native,owner->args.plan.native.ingress);
                std::memcpy(&ingress->invocation,invocation(),sizeof(n::Invocation));received=&ingress->invocation;
            }
            owner->args.input={received,nullptr,nullptr,nullptr,nullptr,nullptr};require(owner->run<n::OwnerAction::Begin>());
        }
        run<n::CoordinatorAction::PinCommands>();
        for(std::size_t i=0;i<owners.size();++i){
            auto& x=owners[i]->args;const auto slice=n::at<n::DirectedSlice>(args.native,args.plan.native.pin_slices)[i];
            x.input.records=x.plan.native.reply_capacity?n::at<p::OwnerCommand>(args.native,args.plan.native.pin_commands)+slice.byte_offset/sizeof(p::OwnerCommand):nullptr;
            x.input.record_bytes=x.plan.native.reply_capacity?&slice.live_bytes:nullptr;
            if(receivedIngress){
                auto* ingress=n::at<n::OwnerIngress>(x.native,x.plan.native.ingress);
                if(slice.live_bytes>std::uint64_t{x.plan.native.reply_capacity}*sizeof(p::OwnerCommand))throw std::logic_error("oversized decoded pin wire");
                if(slice.live_bytes)std::memcpy(x.frame+x.plan.frame.commands,x.input.records,slice.live_bytes);
                ingress->record_bytes=slice.live_bytes;
                x.input.records=x.plan.native.reply_capacity?x.frame+x.plan.frame.commands:nullptr;
                x.input.record_bytes=x.plan.native.reply_capacity?&ingress->record_bytes:nullptr;
                n::progress(x).reply_bytes=~std::uint64_t{0};
                require(owners[i]->run<n::OwnerAction::Pin>());
                if(ingress->record_bytes!=slice.live_bytes || std::memcmp(&ingress->invocation,invocation(),sizeof(n::Invocation)))
                    throw std::logic_error("immutable invocation/incoming count overwritten by outgoing publication");
            }else require(owners[i]->run<n::OwnerAction::Pin>());
            x.input.records=nullptr;x.input.record_bytes=nullptr;
        }
        if(replyIngress){
            // All actual Pin command readers above have finished before this
            // source bank's first receive overwrite, as the graph lifetime
            // helper enforces for asynchronous native participant endpoints.
            auto* wire=args.native+args.plan.native.pin_commands;
            auto* frontier=n::at<p::PinHeader>(args.native,args.plan.native.ingress_frontier);
            for(std::size_t i=0;i<owners.size();++i){auto& owner=*owners[i];auto& x=owner.args;
                const auto bytes=n::progress(x).reply_bytes;
                if(bytes>std::uint64_t{x.plan.native.reply_capacity}*sizeof(p::Grant))throw std::logic_error("invalid original Pin reply extent");
                // Decode the existing GPU publishReplies observation from its
                // actual immutable initial-pin producer; GPU publication itself
                // remains a separate native gate, not a CPU-copy certificate.
                if(x.plan.native.reply_capacity){
                    if(bytes)std::memcpy(x.native+x.plan.native.replies,x.frame+x.plan.frame.initial_pins,bytes);
                    if(bytes)std::memcpy(wire,x.native+x.plan.native.replies,bytes);
                    auto* count=n::at<std::uint64_t>(args.native,args.plan.native.ingress_count);*count=bytes;
                    args.input.records=wire;args.input.record_bytes=count;run<n::CoordinatorAction::AcquirePins>();args.input={};
                }else if(bytes)throw std::logic_error("C0 published a fake Grant payload");
                require(owner.run<n::OwnerAction::SealPins>());
                std::memcpy(frontier+i,x.native+x.plan.native.header,sizeof(p::PinHeader));
            }
            if(beforeReceivedSeal)beforeReceivedSeal();
            args.input.frontier=frontier;run<n::CoordinatorAction::SealPins>();args.input={};
            for(auto& owner:owners){auto& x=owner->args;
                std::memcpy(x.frame+x.plan.frame.pin_headers,frontier,owners.size()*sizeof(p::PinHeader));
                x.input.frontier=x.frame+x.plan.frame.pin_headers;require(owner->run<n::OwnerAction::AcquirePins>());x.input.frontier=nullptr;}
        }else{
            grants<n::CoordinatorAction::AcquirePins>(true);all<n::OwnerAction::SealPins>();const auto pins=headers<p::PinHeader>();
            args.input.frontier=pins.data();run<n::CoordinatorAction::SealPins>();args.input={};
            for(auto& owner:owners){owner->args.input.frontier=pins.data();require(owner->run<n::OwnerAction::AcquirePins>());owner->args.input.frontier=nullptr;}
        }
    }
    /** @brief Decode only exact original free-wave wire messages into their actual phase-reused banks.
     * The fixture observes parser operands, not GPU transport/capture/storage qualification. */
    std::vector<std::pair<uint64_t,uint64_t>> receivedFree(
        std::function<void()> beforePlan={},std::function<void(size_t)> beforeAccept={},
        bool cancel=false,std::function<void()> beforeComplete={}){
        auto* frontier=args.native+args.plan.native.ingress_frontier;
        all<n::OwnerAction::Availability>();
        for(size_t i=0;i<owners.size();++i)std::memcpy(frontier+i*sizeof(n::Availability),owners[i]->args.native+owners[i]->args.plan.native.header,sizeof(n::Availability));
        if(beforePlan)beforePlan();args.input.frontier=frontier;run<n::CoordinatorAction::PlanFree>();args.input={};
        std::vector<std::pair<uint64_t,uint64_t>> extents;
        // Actual graph sends all original plan headers/slices before serial
        // Grant reception; this decoded fixture first acquires every plan too.
        for(size_t i=0;i<owners.size();++i){auto& x=owners[i]->args;const auto slice=n::at<n::DirectedSlice>(args.native,args.plan.native.free_slices)[i];
            const auto* header=n::at<n::FreePlanHeader>(args.native,args.plan.native.free_headers)+i;
            if(slice.byte_offset<args.plan.frame.decisions || slice.byte_offset-args.plan.frame.decisions>uint64_t{args.plan.policy.maximumUnique()}*sizeof(p::AllocationDecision) ||
               slice.live_bytes>uint64_t{x.plan.native.reply_capacity}*sizeof(p::AllocationDecision) || slice.live_bytes!=header->decision_bytes)throw std::logic_error("invalid decoded original free slice");
            std::memcpy(x.native+x.plan.native.header,header,sizeof(n::FreePlanHeader));
            if(slice.live_bytes)std::memcpy(x.frame+x.plan.frame.commands,args.frame+slice.byte_offset,slice.live_bytes);
            if(x.plan.native.reply_capacity)*n::at<uint64_t>(x.native,x.plan.native.ingress+offsetof(n::OwnerIngress,record_bytes))=slice.live_bytes;
            extents.push_back({slice.live_bytes,0});
        }
        for(size_t i=0;i<owners.size();++i){auto& owner=*owners[i];auto& x=owner.args;
            x.input.frontier=x.native+x.plan.native.header;x.input.records=x.plan.native.reply_capacity?x.frame+x.plan.frame.commands:nullptr;
            x.input.record_bytes=x.plan.native.reply_capacity?n::at<uint64_t>(x.native,x.plan.native.ingress+offsetof(n::OwnerIngress,record_bytes)):nullptr;
            x.input.record_layout=n::RecordLayout::TransportedAllocationPrefix;if(beforeAccept)beforeAccept(i);
            require(owner.run<n::OwnerAction::AcceptFree>());x.input.records=nullptr;x.input.record_bytes=nullptr;x.input.frontier=nullptr;x.input.record_layout=n::RecordLayout::Compact;
            if(cancel)require(owner.run<n::OwnerAction::CancelFree>());else require(owner.run<n::OwnerAction::ExecuteFree>());
            const auto bytes=n::progress(x).reply_bytes;extents[i].second=bytes;
            if(bytes%sizeof(p::Grant) || bytes>uint64_t{x.plan.native.reply_capacity}*sizeof(p::Grant))throw std::logic_error("invalid decoded original free Grant prefix");
            // Exact output selection of the native publishReplies producer;
            // CPU memcpy supplies decoded acquisition, not a GPU kernel proof.
            if(bytes)std::memcpy(x.native+x.plan.native.replies,n::grants(x)+n::allocationFrame(x).wave_offset,bytes);
            if(x.plan.native.reply_capacity){if(bytes)std::memcpy(args.native+args.plan.native.pin_commands,x.native+x.plan.native.replies,bytes);
                auto* incoming=n::at<uint64_t>(args.native,args.plan.native.ingress_count);*incoming=bytes;args.input.records=args.native+args.plan.native.pin_commands;args.input.record_bytes=incoming;run<n::CoordinatorAction::AcquireFree>();args.input={};
            }else if(bytes)throw std::logic_error("C0 free owner invented a Grant message");
            if(!cancel)require(owner.run<n::OwnerAction::FreeReceipt>());
            std::memcpy(frontier+i*sizeof(p::FreeAllocationReceipt),x.native+x.plan.native.header,sizeof(p::FreeAllocationReceipt));
        }
        if(beforeComplete)beforeComplete();args.input.frontier=frontier;run<n::CoordinatorAction::CompleteFree>();args.input={};
        // CancelFree is terminal/Draining and must never enter ordinary
        // AcquireFree, which authorizes a subsequent victim wave. CompleteFree
        // still acquires all exact cancellation receipts before real release.
        if(!cancel)for(auto& owner:owners){auto& x=owner->args;std::memcpy(x.frame+x.plan.frame.free_headers,frontier,owners.size()*sizeof(p::FreeAllocationReceipt));x.input.frontier=x.frame+x.plan.frame.free_headers;require(owner->run<n::OwnerAction::AcquireFree>());x.input.frontier=nullptr;}
        return extents;
    }
    /** @brief Acquire original victim suffixes and current-wave Grant prefixes in the existing banks only. */
    std::vector<std::pair<uint64_t,uint64_t>> receivedVictims(std::function<void(size_t)> beforeAccept={},bool cancel=false,std::function<void()> beforeComplete={},bool cancelBeforeAccept=false){
        const auto freeCount=view().frame->free_planned;std::vector<std::byte> freePrefix(args.frame+args.plan.frame.decisions,args.frame+args.plan.frame.decisions+uint64_t{freeCount}*sizeof(p::AllocationDecision));
        std::vector<std::vector<std::byte>> accepted,granted,frontiers;
        for(auto& owner:owners){auto& x=owner->args;const auto count=n::allocationFrame(x).count;accepted.emplace_back(x.frame+x.plan.frame.accepted_plan,x.frame+x.plan.frame.accepted_plan+uint64_t{count}*sizeof(p::AllocationDecision));granted.emplace_back(reinterpret_cast<std::byte*>(n::grants(x)),reinterpret_cast<std::byte*>(n::grants(x)+count));frontiers.emplace_back(x.frame+x.plan.frame.free_headers,x.frame+x.plan.frame.free_headers+owners.size()*sizeof(p::FreeAllocationReceipt));}
        run<n::CoordinatorAction::PlanVictims>();
        if(!freePrefix.empty() && std::memcmp(freePrefix.data(),args.frame+args.plan.frame.decisions,freePrefix.size()))throw std::logic_error("Victim planner overwrote original free decisions");
        std::vector<std::pair<uint64_t,uint64_t>> extents;auto* received=args.native+args.plan.native.ingress_frontier;
        for(size_t i=0;i<owners.size();++i){auto& x=owners[i]->args;const auto slice=n::at<n::DirectedSlice>(args.native,args.plan.native.victim_slices)[i];const auto* h=n::at<n::VictimPlanHeader>(args.native,args.plan.native.victim_headers)+i;const auto origin=args.plan.frame.decisions;const auto span=uint64_t{args.plan.policy.maximumUnique()}*sizeof(p::AllocationDecision);
            if(slice.byte_offset<origin || slice.byte_offset-origin>span || slice.live_bytes>span-(slice.byte_offset-origin) || slice.live_bytes>uint64_t{x.plan.native.reply_capacity}*sizeof(p::AllocationDecision) || slice.live_bytes!=h->decision_bytes)throw std::logic_error("Malformed original victim suffix descriptor");
            std::memcpy(x.native+x.plan.native.header,h,sizeof(n::VictimPlanHeader));if(slice.live_bytes)std::memcpy(x.frame+x.plan.frame.commands,args.frame+slice.byte_offset,slice.live_bytes);if(x.plan.native.reply_capacity)*n::at<uint64_t>(x.native,x.plan.native.ingress+offsetof(n::OwnerIngress,record_bytes))=slice.live_bytes;extents.push_back({slice.live_bytes,0});
        }
        for(size_t i=0;i<owners.size();++i){auto& owner=*owners[i];auto& x=owner.args;x.input.frontier=x.native+x.plan.native.header;x.input.records=x.plan.native.reply_capacity?x.frame+x.plan.frame.commands:nullptr;x.input.record_bytes=x.plan.native.reply_capacity?n::at<uint64_t>(x.native,x.plan.native.ingress+offsetof(n::OwnerIngress,record_bytes)):nullptr;x.input.record_layout=n::RecordLayout::TransportedAllocationPrefix;if(beforeAccept)beforeAccept(i);if(!cancelBeforeAccept)require(owner.run<n::OwnerAction::AcceptVictim>());x.input.records=nullptr;x.input.record_bytes=nullptr;x.input.frontier=nullptr;x.input.record_layout=n::RecordLayout::Compact;
            if(cancel)require(owner.run<n::OwnerAction::CancelVictim>());else require(owner.run<n::OwnerAction::ExecuteVictim>());const auto bytes=n::progress(x).reply_bytes;extents[i].second=bytes;if(bytes%sizeof(p::Grant) || bytes>uint64_t{x.plan.native.reply_capacity}*sizeof(p::Grant))throw std::logic_error("Malformed current-wave victim Grant prefix");if(bytes)std::memcpy(x.native+x.plan.native.replies,n::grants(x)+n::allocationFrame(x).wave_offset,bytes);
            if(x.plan.native.reply_capacity){if(bytes)std::memcpy(args.native+args.plan.native.pin_commands,x.native+x.plan.native.replies,bytes);auto* count=n::at<uint64_t>(args.native,args.plan.native.ingress_count);*count=bytes;args.input.records=args.native+args.plan.native.pin_commands;args.input.record_bytes=count;run<n::CoordinatorAction::AcquireVictims>();args.input={};}else if(bytes)throw std::logic_error("C0 invented a victim Grant message");
            if(!cancel)require(owner.run<n::OwnerAction::VictimReceipt>());std::memcpy(received+i*sizeof(p::VictimAllocationReceipt),x.native+x.plan.native.header,sizeof(p::VictimAllocationReceipt));
            if((!accepted[i].empty() && std::memcmp(accepted[i].data(),x.frame+x.plan.frame.accepted_plan,accepted[i].size())) || (!granted[i].empty() && std::memcmp(granted[i].data(),n::grants(x),granted[i].size())) || std::memcmp(frontiers[i].data(),x.frame+x.plan.frame.free_headers,frontiers[i].size()))throw std::logic_error("Victim execution changed original free prefix/frontier");
        }
        if(beforeComplete)beforeComplete();args.input.frontier=received;run<n::CoordinatorAction::CompleteVictims>();args.input={};return extents;
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
    void finish(std::function<void(const n::SourceFrontier&)> beforeRetirement={},std::function<void()> afterRetirement={}){
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
        if(beforeRetirement)beforeRetirement(source);
        all<n::OwnerAction::Release>();all<n::OwnerAction::Retire>();const auto retired=headers<p::RetiredHeader>();args.input.frontier=retired.data();run<n::CoordinatorAction::Finish>();args.input={};
        if(afterRetirement)afterRetirement();
    }
    void complete(std::span<const std::uint64_t> original,std::uint64_t admission=1){begin(original,admission);wave<false>();wave<true>();finish();}
};
}

} // namespace llaminar2::test
