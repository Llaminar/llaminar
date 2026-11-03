/**
 * @file Test__PooledImmutableNative.cpp
 * @brief Replay actual shared native metadata phases using decoded, acquired CPU fixtures.
 *
 * These roots exercise the same phase parsers used by CUDA/HIP kernels. The
 * fixture supplies decoded transport/source observations, and its CPU payload
 * producer does not certify GPU copy, native capture, transport or serving.
 */
#include "fixtures/qwen4exp/PooledImmutableNativeReplayBytes.h"
#include "kernels/qwen4exp/PooledImmutableNativeOps.h"
#include "kernels/qwen4exp/PooledImmutableOwnerRecordValidation.h"
#include "kernels/qwen4exp/PooledImmutableNativePlan.h"
#include "kernels/qwen4exp/PooledImmutableNativeOperands.h"
#include "kernels/qwen4exp/PooledImmutableSourceOps.h"
#include "kernels/qwen4exp/PooledImmutableRetirementOps.h"
#include "kernels/qwen4exp/PooledImmutablePublicationOps.h"
#include "kernels/common/SamplingMath.h"
#include "../../../fixtures/qwen4exp/original_PooledImmutableSourceOwner.inc"
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
 * @param maximumUnique Exact admitted identity scratch/live-row bound.
 * @return Checked declarations only, without device discovery or allocation.
 */
q::PooledImmutablePhysicalOwnerGeometry declared(std::span<const std::uint32_t> capacities,
    std::optional<std::pair<std::size_t,PhysicalMemoryAllocatorIdentity>> replacement=std::nullopt,std::uint32_t maximumUnique=16)
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
        ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,0x100000100ULL,90),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,owners,
        {{102,DeviceId::cpu()},coordinator},{},
        {{{103,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},
        {},positive?std::optional(coordinator):std::nullopt,maximumUnique);
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
    std::optional<std::uint32_t> direct_pin_owner;
    std::uint64_t observed_direct_pin_bytes=0,observed_direct_pin_offset=0;
    explicit Pool(std::span<const std::uint32_t> capacities,std::uint32_t maximumOriginal=64,a::Policy policy={65536,1,2},std::optional<std::size_t> local=std::nullopt,std::uint32_t maximumUnique=16):
        physical(declared(capacities,local?std::optional(std::pair{*local,PhysicalMemoryAllocatorIdentity{101,DeviceId::rocm(11)}}):std::nullopt,maximumUnique)),geometry(q::PooledImmutableCoordinatorGeometry::fromOwners(physical,policy,maximumOriginal)),
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
            }else{
                if(direct_pin_owner && i==*direct_pin_owner && x.plan.native.reply_capacity){
                    x.input.records=args.native+args.plan.native.pin_commands;x.input.record_bytes=nullptr;
                    x.input.allocation_slice=n::at<n::DirectedSlice>(args.native,args.plan.native.pin_slices)+i;
                    x.input.record_layout=n::RecordLayout::DirectedCommandBank;x.input.record_origin=0;
                    x.input.record_capacity=std::uint64_t{args.plan.native.reply_capacity}*sizeof(p::OwnerCommand);
                    observed_direct_pin_bytes=std::max(observed_direct_pin_bytes,slice.live_bytes);
                    observed_direct_pin_offset=std::max(observed_direct_pin_offset,slice.byte_offset);
                }
                require(owners[i]->run<n::OwnerAction::Pin>());
            }
            x.input.records=nullptr;x.input.record_bytes=nullptr;x.input.allocation_slice=nullptr;
            x.input.record_layout=n::RecordLayout::Compact;x.input.record_origin=x.input.record_capacity=0;
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
    EXPECT_EQ(owner->args.plan.native.bytes,256u); // Invocation/count64 plus retained terminal source outcome64.
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
    for(unsigned defect=0;defect<5;++defect){
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
        if(defect==3)commands[1].target.unique=commands[0].target.unique;
        if(defect==4)commands[1].target.slot=commands[0].target.slot;
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
    const std::array<size_t,3> actualRemoteExtents{256,576,768};
        ASSERT_LT(plan.native.reply_capacity,actualRemoteExtents.size());
        EXPECT_EQ(plan.native.bytes,actualRemoteExtents[plan.native.reply_capacity]);
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
    const auto identity=declared.declarationIdentity();
    EXPECT_NO_THROW(declared.requireDeclarations(n::OwnerAction::Pin,plan));
    EXPECT_THROW(declared.requireDeclarations(n::OwnerAction::Begin,plan),std::invalid_argument);
    Pool foreign(std::array{4u});
    EXPECT_THROW(declared.requireDeclarations(n::OwnerAction::Pin,foreign.owners[0]->plan),std::invalid_argument);
    EXPECT_THROW(declared.captureIdentity(),std::logic_error);
    EXPECT_THROW(declared.require(n::OwnerAction::Pin,plan),std::logic_error);
    EXPECT_THROW(declared.seal(),std::logic_error);
    EXPECT_EQ(declared.declarationIdentity(),identity);
    EXPECT_EQ(root->gpu_data_ptr(),nullptr);
    q::PooledImmutableCoordinatorOperands metadata(n::CoordinatorAction::PinCommands,pool.plan,std::nullopt,{});
    EXPECT_NO_THROW(metadata.requireDeclarations(n::CoordinatorAction::PinCommands,pool.plan));
    EXPECT_THROW(metadata.requireDeclarations(n::CoordinatorAction::PlanVictims,pool.plan),std::invalid_argument);
    EXPECT_THROW(metadata.requireDeclarations(n::CoordinatorAction::PinCommands,foreign.plan),std::invalid_argument); // Missing GPU address rejects before backend/runtime discovery.

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


/** @brief Source geometry must reject same-sized foreign owners and any replaced member of the original cohort. */
TEST(PooledImmutableNative, SourceOwnerCohortIdentityRejectsForeignAllocatorAndReplacedPeer)
{
    const ImmutableRowTableSource descriptor{.path="cpu_parser_original_source",.tensor_name="per_layer_token_embd.weight",
        .file_offset=37,.rows=0x100000100ULL,.row_elements=160,.format=GGUFTensorType::IQ4_NL};
    for(const auto participants:{1u,2u,4u,8u}){
        std::vector<std::uint32_t> capacities(participants,3u);capacities.front()=0;
        const auto original=declared(capacities);
        const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(original,{65536,1,2},64);
        const auto coordinator=q::PooledImmutableCoordinatorNativePlan::fromGeometry(original,policy,701,17);
        for(std::uint32_t coordinate=0;coordinate<participants;++coordinate){
            SCOPED_TRACE(::testing::Message()<<"participants="<<participants<<" owner="<<coordinate);
            const auto actual=q::PooledImmutableOwnerNativePlan::fromGeometry(original,policy,coordinate,701,17);
            EXPECT_NO_THROW(coordinator.requireOwnerPlan(actual));
            std::vector<q::PooledImmutableOwnerNativePlan> cohort;for(std::uint32_t i=0;i<participants;++i)cohort.push_back(q::PooledImmutableOwnerNativePlan::fromGeometry(original,policy,i,701,17));
            EXPECT_NO_THROW(q::PooledImmutableSourceGeometry::fromPlans(cohort,coordinator,descriptor));
            if(participants>1)EXPECT_THROW(q::PooledImmutableSourceGeometry::fromPlans(actual,coordinator,descriptor),std::invalid_argument);
            const auto allocator=actual.allocator();
            const std::array replacements{
                PhysicalMemoryAllocatorIdentity{allocator.world_rank+1000,allocator.device},
                PhysicalMemoryAllocatorIdentity{allocator.world_rank,{allocator.device.type,allocator.device.ordinal+16}},
                PhysicalMemoryAllocatorIdentity{allocator.world_rank,allocator.device.type==DeviceType::CUDA?
                    DeviceId::rocm(allocator.device.ordinal):DeviceId::cuda(allocator.device.ordinal)}};
            for(const auto replacement:replacements){
                const auto foreign=declared(capacities,std::pair<std::size_t,PhysicalMemoryAllocatorIdentity>{coordinate,replacement});
                const auto foreignPolicy=q::PooledImmutableCoordinatorGeometry::fromOwners(foreign,{65536,1,2},64);
                for(std::uint32_t candidate=0;candidate<participants;++candidate){
                    // The complete cohort is part of each native identity:
                    // replacing only another peer must reject this owner too.
                    const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(foreign,foreignPolicy,candidate,701,17);
                    ASSERT_EQ(owner.parameters().owner.capacity(),capacities[candidate]);
                    EXPECT_THROW(coordinator.requireOwnerPlan(owner),std::invalid_argument);
                    EXPECT_THROW(q::PooledImmutableSourceGeometry::fromPlans(owner,coordinator,descriptor),std::invalid_argument);
                }
            }
        }
    }
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
        pool.args.input.source=&source::frontier(args);
        ASSERT_NO_THROW(pool.run<n::CoordinatorAction::SealSource>());pool.args.input={};
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
        EXPECT_EQ(source::validateResponse(args),corrupt<0);
        if(corrupt<0)EXPECT_TRUE(source::prepareDelivery(args,0));
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
        pool.args.input.source=&source::frontier(args);
        ASSERT_NO_THROW(pool.run<n::CoordinatorAction::SealSource>());pool.args.input={};
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

namespace llaminar2::test {
/** @test Exact original phase work reaches terminal words once, only after all physical/read/source retirement. */
TEST(PooledImmutableNative, SplitTerminalObservationWaitsForActualRetirementAndReplaysResetEmptyAndHighRows)
{
    namespace source=q::pooled_source;
    const ImmutableRowTableSource descriptor{.path="decoded_observation_trace",.tensor_name="per_layer_token_embd.weight",
        .rows=0x100000100ULL,.row_elements=160,.format=GGUFTensorType::IQ4_NL};
    for(const auto slots:{0u,1u,4u}){
        Pool pool(std::array{slots});
        const auto geometry=q::PooledImmutableSourceGeometry::fromPlans(pool.owners.front()->plan,pool.plan,descriptor);
        Bytes frame(geometry.parameters().offsets.frame_bytes),native(geometry.parameters().offsets.native_bytes);
        std::memset(native.data(),0,geometry.parameters().offsets.native_bytes);
        std::array<int,sampling_math::kDeviceGenerationControlCount> control{};control[sampling_math::kDeviceGenerationControlOk]=1;
        source::DeviceArgs actual{geometry.parameters(),pool.owners.front()->args,pool.args,frame.data(),native.data(),{},
            control.data()+sampling_math::kDeviceGenerationControlOk,control.data()+sampling_math::kDeviceGenerationControlErrorCode,control.data()};
        q::QwenAdaptiveCounterTotals expected;
        for(unsigned replay=0;replay<5;++replay){
            if(replay==3){
                // Reset only canonical reporting data; source ordinals, warmed
                // slots and all original root identities remain unchanged.
                std::fill_n(control.data()+sampling_math::kDeviceGenerationControlAdaptiveCallsLow,18,0);
                expected={};
            }
            std::vector<std::uint64_t> ids(replay==2?0:replay==3?16:32,0x100000007ULL);
            for(std::size_t i=0;i<ids.size();++i)if(i%3==1)ids[i]=replay==4?701:700;
            ASSERT_NO_THROW(pool.begin(ids,replay<3?1:2));
            ASSERT_NO_THROW(pool.wave<false>());
            ASSERT_NO_THROW(pool.wave<true>());
            ASSERT_NO_THROW(pool.finish([&](const n::SourceFrontier& frontier){
                source::frontier(actual)=frontier;source::stage(actual)=source::ResponseStage::Consumed;
                auto v=pool.view();std::uint64_t misses=0;
                for(std::uint32_t i=0;i<v.frame->unique_rows;++i)misses+=v.unique[i].phase!=p::UniquePhase::Read;
                source::protocol(actual).nativeBytes=misses*90;
                ASSERT_TRUE(source::snapshotObservation<a::Slot64>(actual));
                const auto tuple=*n::at<q::QwenAdaptiveCounterTotals>(actual.native,geometry.parameters().offsets.observation);
                expected.calls+=tuple.calls;expected.completed_live_ids+=tuple.completed_live_ids;
                expected.hit_rows+=tuple.hit_rows;expected.miss_rows+=tuple.miss_rows;expected.source_rows+=tuple.source_rows;
                expected.source_bytes+=tuple.source_bytes;expected.promotions+=tuple.promotions;expected.evictions+=tuple.evictions;
                source::stage(actual)=source::ResponseStage::ReadersComplete;
                q::QwenAdaptiveCounterTotals before;ASSERT_TRUE(q::readQwenAdaptiveCounters(control.data(),control.size(),before));
                EXPECT_FALSE(source::observeRetired(actual));
                q::QwenAdaptiveCounterTotals after;ASSERT_TRUE(q::readQwenAdaptiveCounters(control.data(),control.size(),after));EXPECT_EQ(before,after);
            },[&]{
                const auto bytes=n::progress(actual.owner).header_bytes;n::progress(actual.owner).header_bytes=0;
                EXPECT_FALSE(source::observeRetired(actual));n::progress(actual.owner).header_bytes=bytes;
                ASSERT_TRUE(source::observeRetired(actual));EXPECT_FALSE(source::observeRetired(actual));
            }));
            q::QwenAdaptiveCounterTotals observed;ASSERT_TRUE(q::readQwenAdaptiveCounters(control.data(),control.size(),observed));
            EXPECT_EQ(observed,expected);EXPECT_TRUE(q::validQwenAdaptiveCounters(observed,90,64));
            if(!slots){EXPECT_EQ(observed.promotions,0u);EXPECT_EQ(observed.evictions,0u);EXPECT_EQ(observed.hit_rows,0u);}
        }
    }
}

}

namespace llaminar2::test {
/** @test The actual source authorization preserves consumed bytes/sequence while cancellation forbids the Ready branch across reset. */
TEST(PooledImmutableNative, BeforeReadyCancellationPreservesConsumedWitnessAndReplaysResetPartialEmpty)
{
    namespace source=q::pooled_source;
    for(const auto capacity:{0u,2u}){
        Pool pool(std::array{capacity});
        const ImmutableRowTableSource descriptor{.path="cpu_original_source",.tensor_name="per_layer_token_embd.weight",
            .file_offset=37,.rows=0x100000100ULL,.row_elements=160,.format=GGUFTensorType::IQ4_NL};
        const auto geometry=q::PooledImmutableSourceGeometry::fromPlans(pool.owners.front()->plan,pool.plan,descriptor);
        Bytes frame(geometry.parameters().offsets.frame_bytes),native(geometry.parameters().offsets.native_bytes);
        std::memset(native.data(),0,geometry.parameters().offsets.native_bytes);
        int healthy=1,error=0;
        source::DeviceArgs actual{geometry.parameters(),pool.owners.front()->args,pool.args,frame.data(),native.data(),{},&healthy,&error};
        for(unsigned replay=0;replay<20;++replay)for(const auto count:{64u,17u,1u,0u}){
            const p::Interval interval{701,17+replay,1+count};const p::SourceWindow window=count?p::SourceWindow{701,17+replay,1+count}:p::SourceWindow{};
            source::frontier(actual)={interval,window,count?n::SourceDisposition::Consumed:n::SourceDisposition::Unpublished,a::Status::Success,n::Version};
            source::stage(actual)=source::ResponseStage::Consumed;
            auto& io=source::protocol(actual);io.issuedSequence=io.consumedSequence=1+count;io.nativeBytes=count?90:0;io.idCount=count?1:0;
            io.status=0;healthy=1;error=0;
            ASSERT_TRUE(source::beforeReady(actual));
            EXPECT_EQ(n::sourceGate(n::SourceCondition::AfterSuccess,&source::frontier(actual),interval,701,window),n::SourceGate::Run);
            const auto consumed=source::frontier(actual).disposition;const auto bytes=io.nativeBytes;
            healthy=0;error=static_cast<int>(sampling_math::DeviceGenerationError::InvalidController);
            EXPECT_FALSE(source::beforeReady(actual));EXPECT_FALSE(source::beforeReady(actual));
            EXPECT_EQ(source::frontier(actual).status,a::Status::Cancelled);EXPECT_EQ(source::frontier(actual).disposition,consumed);
            EXPECT_EQ(io.nativeBytes,bytes);EXPECT_EQ(io.issuedSequence,io.consumedSequence);
            EXPECT_EQ(n::sourceGate(n::SourceCondition::AfterSuccess,&source::frontier(actual),interval,701,window),n::SourceGate::Skip);
            EXPECT_EQ(n::sourceGate(n::SourceCondition::AfterFailure,&source::frontier(actual),interval,701,window),n::SourceGate::Run);
        }
    }
}
}

namespace llaminar2::test {
/** @brief Replay original remote ingress roles through actual Pin parsers, then the full existing retirement protocol. */
TEST(PooledImmutableNative, RemoteIngressKeepsInvocationAndIncomingCountAcrossPinAndShrinkingReplay)
{
    for(const auto participants:{1u,2u,4u,8u}){
        std::vector<std::uint32_t> capacities(participants,3u);if(participants>1)capacities[1]=0;
        Pool pool(capacities);const auto identity=pool.plan.parameters().identity;
        ASSERT_NO_THROW(pool.complete(std::array<std::uint64_t,3>{101,102,103}));
        for(const auto rows:{std::vector<std::uint64_t>{101,102,103,101},std::vector<std::uint64_t>{101},std::vector<std::uint64_t>{},std::vector<std::uint64_t>{101,102}}){
            ASSERT_NO_THROW(pool.begin(rows,2,true));
            ASSERT_NO_THROW(pool.wave<false>());
            ASSERT_NO_THROW(pool.wave<true>());
            ASSERT_NO_THROW(pool.finish());
            EXPECT_EQ(pool.view().frame->phase,p::LookupPhase::Idle);EXPECT_EQ(pool.plan.parameters().identity,identity);
        }
    }
}

/** @brief Actual parsers consume each original owner return and complete ALL-owner frontier across repeated partial/empty requests. */
TEST(PooledImmutableNative, SharedPinReplyIngressReplaysOrderedC0EmptyAndShrinkingFrames)
{
    for(unsigned participants:{1u,2u,4u,8u})for(bool allEmpty:{false,true}){
        std::vector<std::uint32_t> capacities(participants,allEmpty?0u:3u);if(participants>1)capacities[1]=0;
        Pool pool(capacities);
        ASSERT_NO_THROW(pool.complete(std::array<std::uint64_t,3>{101,102,103}));
        const auto identity=pool.plan.parameters().identity;
        for(const auto rows:{std::vector<std::uint64_t>{101,102,103,101,103},std::vector<std::uint64_t>{101},std::vector<std::uint64_t>{},std::vector<std::uint64_t>{101,102}}){
            ASSERT_NO_THROW(pool.begin(rows,2,true,true));
            ASSERT_EQ(pool.view().frame->phase,p::LookupPhase::Availability);
            const auto* frontier=n::at<p::PinHeader>(pool.args.native,pool.args.plan.native.ingress_frontier);
            for(size_t i=0;i<pool.owners.size();++i){const auto& x=pool.owners[i]->args;
                EXPECT_EQ(frontier[i].owner,i);EXPECT_EQ(frontier[i].status,a::Status::Success);
                EXPECT_EQ(std::memcmp(x.frame+x.plan.frame.pin_headers,frontier,participants*sizeof(p::PinHeader)),0);
                EXPECT_LE(n::progress(x).reply_bytes,std::uint64_t{x.plan.native.reply_capacity}*sizeof(p::Grant));
                if(!x.plan.native.reply_capacity)EXPECT_EQ(n::progress(x).reply_bytes,0u);
            }
            ASSERT_NO_THROW(pool.wave<false>());
            ASSERT_NO_THROW(pool.wave<true>());
            ASSERT_NO_THROW(pool.finish());
            EXPECT_EQ(pool.plan.parameters().identity,identity);EXPECT_EQ(pool.view().frame->phase,p::LookupPhase::Idle);
        }
    }
}
/** @brief Missing/stale owner headers cannot become a complete frontier or advance into allocation. */
TEST(PooledImmutableNative, SharedPinReplyIngressRejectsStaleAndPartialOriginalFrontiers)
{
    for(unsigned defect:{0u,1u}){
        Pool pool(std::array{2u,1u});
        ASSERT_NO_THROW(pool.complete(std::array<std::uint64_t,2>{101,102}));
        EXPECT_THROW(pool.begin(std::array<std::uint64_t,2>{101,102},2,true,true,[&]{
            auto* frontier=n::at<p::PinHeader>(pool.args.native,pool.args.plan.native.ingress_frontier);
            if(defect==0)--frontier[1].interval.sequence;else frontier[1]=frontier[0];
        }),std::logic_error);
        EXPECT_NE(pool.view().frame->phase,p::LookupPhase::Availability);EXPECT_NE(pool.view().frame->status,a::Status::Success);
    }
}
TEST(PooledImmutableNative, SharedPinReplyBankReusesExactCancellationReceiptsWithoutOverwritingObservations)
{
    Pool pool(std::array{2u,0u,2u});
    ASSERT_NO_THROW(pool.begin(std::array<std::uint64_t,3>{101,102,103},1,true,true));
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
    const std::uint64_t bytes=releases.size()*sizeof(p::ReleaseReceipt);
    auto* received=pool.args.native+pool.args.plan.native.pin_commands;
    auto* count=n::at<std::uint64_t>(pool.args.native,pool.args.plan.native.ingress_count);
    std::memcpy(received,releases.data(),bytes);*count=bytes;pool.args.input={received,count,nullptr,&source};
    ASSERT_NO_THROW(pool.run<n::CoordinatorAction::Cancel>());pool.args.input={};
    EXPECT_EQ(std::memcmp(original.data(),received,bytes),0);
    EXPECT_EQ(*count,bytes);
    ASSERT_NO_THROW(pool.all<n::OwnerAction::Retire>());
    const auto retired=pool.headers<p::RetiredHeader>();pool.args.input.frontier=retired.data();
    ASSERT_NO_THROW(pool.run<n::CoordinatorAction::Finish>());
}

}

namespace llaminar2::test {
/** @brief Actual source parsers consume exact original free-wave banks across C0/empty/shrinking/reset replays. */
TEST(PooledImmutableNative, ReceivedFreeWaveUsesOriginalCountsAndCompleteFrontiersAcrossReplay)
{
    for(unsigned participants:{1u,2u,4u,8u})for(bool allC0:{false,true}){
        std::vector<uint32_t> capacities(participants,allC0?0u:3u);if(participants>1)capacities[0]=0;
        Pool pool(capacities);uint64_t admission=10;
        for(const auto rows:{std::vector<uint64_t>{101,102,101,103},std::vector<uint64_t>{101},std::vector<uint64_t>{},std::vector<uint64_t>{105,106}}){
            ASSERT_NO_THROW(pool.begin(rows,admission++,true,true));std::vector<std::pair<uint64_t,uint64_t>> extents;
        ASSERT_NO_THROW(extents=pool.receivedFree());
        ASSERT_EQ(extents.size(),participants);
            for(size_t i=0;i<extents.size();++i){const auto& x=pool.owners[i]->args;
        EXPECT_LE(extents[i].first,uint64_t{x.plan.native.reply_capacity}*sizeof(p::AllocationDecision));
        EXPECT_LE(extents[i].second,uint64_t{x.plan.native.reply_capacity}*sizeof(p::Grant));if(!capacities[i])EXPECT_EQ(extents[i],(std::pair<uint64_t,uint64_t>{0,0}));if(rows.empty())EXPECT_EQ(extents[i],(std::pair<uint64_t,uint64_t>{0,0}));}
            ASSERT_NO_THROW(pool.wave<true>(true));
        ASSERT_NO_THROW(pool.finish());
        }
    }
}
/** @brief Count/header mismatch, stale leases, partial bytes and over-capacity records fail before reservation mutation. */
TEST(PooledImmutableNative, ReceivedFreeWaveRejectsMalformedTransportBeforeReservation)
{
    for(unsigned defect=0;defect<10;++defect){Pool pool(std::array{3u,0u});
        ASSERT_NO_THROW(pool.begin(std::array<uint64_t,2>{101,102},1,true,true));
        p::AllocationFrame before{};std::vector<std::byte> cache;bool observed=false;
        EXPECT_THROW(pool.receivedFree({},[&](size_t i){if(i)return;auto& x=pool.owners[0]->args;auto* h=n::at<n::FreePlanHeader>(x.native,x.plan.native.header);auto* count=n::at<uint64_t>(x.native,x.plan.native.ingress+offsetof(n::OwnerIngress,record_bytes));before=n::allocationFrame(x);cache.assign(x.cache,x.cache+x.plan.cache.bytes);observed=true;
            if(defect==0)*count=0;
            if(defect==1)*count=h->decision_bytes+sizeof(p::AllocationDecision);
            if(defect==2){h->decision_bytes=1;*count=1;}
            if(defect==3){h->decision_bytes=uint64_t{x.plan.native.reply_capacity+1}*sizeof(p::AllocationDecision);*count=h->decision_bytes;}
            if(defect==4)++h->lease.interval.sequence;
            if(defect==5)x.input.record_bytes=nullptr;
            if(defect==6){h->decision_bytes=0;*count=sizeof(p::AllocationDecision);}
            if(defect==7)x.input.records=static_cast<const std::byte*>(x.input.records)+1;
            if(defect==8)++h->version;
            if(defect==9)x.input.record_bytes=reinterpret_cast<const uint64_t*>(reinterpret_cast<const std::byte*>(x.input.record_bytes)+1);
        }),std::logic_error)<<defect;
        ASSERT_TRUE(observed);const auto& x=pool.owners[0]->args;const auto& after=n::allocationFrame(x);
        EXPECT_EQ(after.phase,before.phase);
        EXPECT_EQ(after.count,before.count);
        EXPECT_EQ(after.wave_count,before.wave_count);
        EXPECT_EQ(after.free_count,before.free_count);
        EXPECT_EQ(std::memcmp(cache.data(),x.cache,cache.size()),0)<<defect;
    }
}
/** @brief Missing/stale original owner frontiers cannot authorize free planning or completion. */
TEST(PooledImmutableNative, ReceivedFreeWaveRejectsStaleAvailabilityAndPartialCompletion)
{
    for(unsigned defect=0;defect<3;++defect){Pool pool(std::array{2u,0u,2u});
        ASSERT_NO_THROW(pool.begin(std::array<uint64_t,2>{101,102},1,true,true));
        EXPECT_THROW(pool.receivedFree([&]{if(defect==0)++n::at<n::Availability>(pool.args.native,pool.args.plan.native.ingress_frontier)[1].lease.interval.sequence;}, {}, false,[&]{if(defect==1)n::at<p::FreeAllocationReceipt>(pool.args.native,pool.args.plan.native.ingress_frontier)[2]={};if(defect==2)++n::at<p::FreeAllocationReceipt>(pool.args.native,pool.args.plan.native.ingress_frontier)[0].lease.interval.sequence;}),std::logic_error)<<defect;
    }
}
/** @brief Actual CancelFree drains accepted remote plans and permits unchanged request topology reuse after retirement. */
TEST(PooledImmutableNative, ReceivedFreeWaveCancellationReusesOriginalReceiptAndCountBanks)
{
    Pool pool(std::array{2u,0u,2u});
    for(uint64_t admission=1;admission<=3;++admission){ASSERT_NO_THROW(pool.begin(std::array<uint64_t,3>{101,102,103},admission,true,true));
        ASSERT_NO_THROW(pool.receivedFree({}, {}, true));
        ASSERT_NO_THROW(pool.all<n::OwnerAction::Release>());std::vector<p::ReleaseReceipt> releases;
        for(auto& owner:pool.owners){auto& x=owner->args;const auto count=n::progress(x).released;const auto* source=n::at<p::ReleaseReceipt>(x.frame,x.plan.frame.release_receipts);releases.insert(releases.end(),source,source+count);}
        const uint64_t bytes=releases.size()*sizeof(p::ReleaseReceipt);if(bytes)std::memcpy(pool.args.native+pool.args.plan.native.pin_commands,releases.data(),bytes);
        if(pool.args.plan.native.reply_capacity)*n::at<uint64_t>(pool.args.native,pool.args.plan.native.ingress_count)=bytes;
        const n::SourceFrontier source{pool.view().frame->interval,{},n::SourceDisposition::Unpublished,a::Status::Success,n::Version};pool.args.input={pool.args.plan.native.reply_capacity?pool.args.native+pool.args.plan.native.pin_commands:nullptr,pool.args.plan.native.reply_capacity?n::at<uint64_t>(pool.args.native,pool.args.plan.native.ingress_count):nullptr,nullptr,&source};
        ASSERT_NO_THROW(pool.run<n::CoordinatorAction::Cancel>());pool.args.input={};
        ASSERT_NO_THROW(pool.all<n::OwnerAction::Retire>());
        const auto retired=pool.headers<p::RetiredHeader>();pool.args.input.frontier=retired.data();
        ASSERT_NO_THROW(pool.run<n::CoordinatorAction::Finish>());pool.args.input={};
    }
}
}

namespace llaminar2::test {
/** @brief Actual second-wave prefix origins and current-wave replies preserve every original free plan/grant. */
TEST(PooledImmutableNative, ReceivedVictimWaveKeepsNonzeroSuffixAndCurrentWaveGrants)
{
    Pool pool(std::array{2u,0u,2u});
    ASSERT_NO_THROW(pool.complete(std::array<uint64_t,2>{101,102}));const std::array<uint64_t,14> rows{700,701,900,900,900,900,900,900,900,900,900,900,900,900};
    ASSERT_NO_THROW(pool.begin(rows,2,true,true));
    ASSERT_NO_THROW(pool.receivedFree());
    ASSERT_EQ(pool.view().frame->free_planned,2u);std::vector<std::pair<uint64_t,uint64_t>> extents;
    ASSERT_NO_THROW(extents=pool.receivedVictims());
    ASSERT_EQ(pool.view().frame->victim_planned,1u);uint64_t decisions=0,replies=0;
    for(size_t i=0;i<pool.owners.size();++i){const auto slice=n::at<n::DirectedSlice>(pool.args.native,pool.args.plan.native.victim_slices)[i];
    EXPECT_GE(slice.byte_offset,pool.args.plan.frame.decisions+2*sizeof(p::AllocationDecision));decisions+=extents[i].first;replies+=extents[i].second;}
    EXPECT_EQ(decisions,sizeof(p::AllocationDecision));
    EXPECT_EQ(replies,sizeof(p::Grant));
    ASSERT_NO_THROW(pool.finish());
    for(uint64_t request=3;request<7;++request){ASSERT_NO_THROW(pool.begin(request%2?std::vector<uint64_t>{900}:std::vector<uint64_t>{},request,true,true));
    ASSERT_NO_THROW(pool.receivedFree());
    ASSERT_NO_THROW(pool.receivedVictims());
    ASSERT_NO_THROW(pool.finish());}
}
/** @brief Complete C0/empty/unequal cohorts retain exact byte publications and no owner receipt fanout across reset. */
TEST(PooledImmutableNative, ReceivedVictimWaveCohortsC0EmptyShrinkingAndUnusedFree)
{
    for(unsigned participants:{1u,2u,4u,8u})for(bool c0:{false,true}){std::vector<uint32_t> capacity(participants,c0?0u:3u);if(participants>1)capacity[0]=0;Pool pool(capacity);uint64_t request=1;
        for(const auto rows:{std::vector<uint64_t>{31,32,31},std::vector<uint64_t>{33},std::vector<uint64_t>{},std::vector<uint64_t>{31,32}}){ASSERT_NO_THROW(pool.begin(rows,request++,true,true));
    ASSERT_NO_THROW(pool.receivedFree());std::vector<std::pair<uint64_t,uint64_t>> extents;
    ASSERT_NO_THROW(extents=pool.receivedVictims());for(size_t i=0;i<capacity.size();++i)if(!capacity[i])EXPECT_EQ(extents[i],(std::pair<uint64_t,uint64_t>{0,0}));
    ASSERT_NO_THROW(pool.finish());}
    }
    Pool unused(std::array{4u,0u});
    ASSERT_NO_THROW(unused.begin(std::array<uint64_t,1>{31},1,true,true));
    ASSERT_NO_THROW(unused.receivedFree());
    EXPECT_FALSE(unused.view().frame->free_exhausted);
    ASSERT_NO_THROW(unused.receivedVictims());
    EXPECT_EQ(unused.view().frame->victim_planned,0u);
    ASSERT_NO_THROW(unused.finish());
}
/** @brief Invalid count/header/alignment/stale lease cannot mutate actual slots or append accepted victim records. */
TEST(PooledImmutableNative, ReceivedVictimWaveRejectsMalformedBeforeReservation)
{
    for(unsigned defect=0;defect<9;++defect){Pool pool(std::array{2u,0u,2u});
    ASSERT_NO_THROW(pool.complete(std::array<uint64_t,2>{101,102}));const std::array<uint64_t,14> rows{700,701,900,900,900,900,900,900,900,900,900,900,900,900};
    ASSERT_NO_THROW(pool.begin(rows,2,true,true));
    ASSERT_NO_THROW(pool.receivedFree());p::AllocationFrame before{};std::vector<std::byte> cache;size_t observed=~size_t{0};
    EXPECT_THROW(pool.receivedVictims([&](size_t i){auto& x=pool.owners[i]->args;auto* h=n::at<n::VictimPlanHeader>(x.native,x.plan.native.header);if(!h->decision_bytes)return;auto* count=n::at<uint64_t>(x.native,x.plan.native.ingress+48);observed=i;before=n::allocationFrame(x);cache.assign(x.cache,x.cache+x.plan.cache.bytes);
            if(defect==0)*count=0;if(defect==1)*count=h->decision_bytes+48;if(defect==2){h->decision_bytes=1;*count=1;}if(defect==3){h->decision_bytes=uint64_t{x.plan.native.reply_capacity+1}*48;*count=h->decision_bytes;}if(defect==4)++h->lease.interval.sequence;if(defect==5)x.input.record_bytes=nullptr;if(defect==6)x.input.records=static_cast<const std::byte*>(x.input.records)+1;if(defect==7)++h->version;if(defect==8)x.input.record_bytes=reinterpret_cast<const uint64_t*>(reinterpret_cast<const std::byte*>(x.input.record_bytes)+1);
        }),std::logic_error)<<defect;
    ASSERT_NE(observed,~size_t{0});const auto& x=pool.owners[observed]->args;
    EXPECT_EQ(n::allocationFrame(x).count,before.count);
    EXPECT_EQ(n::allocationFrame(x).phase,before.phase);
    EXPECT_EQ(std::memcmp(cache.data(),x.cache,cache.size()),0);
    }
}
}

namespace llaminar2::test {
/** @brief Accepted and unaccepted actual victim cancellation preserves prior grants and retires the complete original receipt set. */
TEST(PooledImmutableNative, ReceivedVictimWaveCancellationDrainsAcceptedAndUnacceptedPlansAcrossReset)
{
    for(bool beforeAccept:{false,true}){Pool pool(std::array{2u,0u,2u});
        ASSERT_NO_THROW(pool.complete(std::array<uint64_t,2>{101,102}));
        for(uint64_t request=2;request<5;++request){const std::array<uint64_t,14> rows{700,701,900,900,900,900,900,900,900,900,900,900,900,900};
            ASSERT_NO_THROW(pool.begin(rows,request,true,true));
            ASSERT_NO_THROW(pool.receivedFree());
            ASSERT_NO_THROW(pool.receivedVictims({},true,{},beforeAccept));
            ASSERT_NO_THROW(pool.all<n::OwnerAction::Release>());std::vector<p::ReleaseReceipt> releases;
            for(auto& owner:pool.owners){auto& x=owner->args;const auto count=n::progress(x).released;const auto* source=n::at<p::ReleaseReceipt>(x.frame,x.plan.frame.release_receipts);releases.insert(releases.end(),source,source+count);}
            const uint64_t bytes=releases.size()*sizeof(p::ReleaseReceipt);if(bytes)std::memcpy(pool.args.native+pool.args.plan.native.pin_commands,releases.data(),bytes);if(pool.args.plan.native.reply_capacity)*n::at<uint64_t>(pool.args.native,pool.args.plan.native.ingress_count)=bytes;
            const n::SourceFrontier source{pool.view().frame->interval,{},n::SourceDisposition::Unpublished,a::Status::Success,n::Version};pool.args.input={pool.args.plan.native.reply_capacity?pool.args.native+pool.args.plan.native.pin_commands:nullptr,pool.args.plan.native.reply_capacity?n::at<uint64_t>(pool.args.native,pool.args.plan.native.ingress_count):nullptr,nullptr,&source};
            ASSERT_NO_THROW(pool.run<n::CoordinatorAction::Cancel>());pool.args.input={};
            ASSERT_NO_THROW(pool.all<n::OwnerAction::Retire>());const auto retired=pool.headers<p::RetiredHeader>();pool.args.input.frontier=retired.data();
            ASSERT_NO_THROW(pool.run<n::CoordinatorAction::Finish>());pool.args.input={};
        }
    }
}
}
namespace llaminar2::test {
/** @brief CompleteVictims requires the exact ALL-owner actual receipt set including C0 before policy publication. */
TEST(PooledImmutableNative, ReceivedVictimWaveRejectsStaleOrMissingCompletionIncludingC0)
{
    for(unsigned defect=0;defect<2;++defect){Pool pool(std::array{2u,0u,2u});
        ASSERT_NO_THROW(pool.complete(std::array<uint64_t,2>{101,102}));
        const std::array<uint64_t,14> rows{700,701,900,900,900,900,900,900,900,900,900,900,900,900};
        ASSERT_NO_THROW(pool.begin(rows,2,true,true));
        ASSERT_NO_THROW(pool.receivedFree());
        EXPECT_THROW(pool.receivedVictims({},false,[&]{auto* receipts=n::at<p::VictimAllocationReceipt>(pool.args.native,pool.args.plan.native.ingress_frontier);if(defect==0)++receipts[0].lease.interval.sequence;else receipts[1]={};}),std::logic_error);
    }
}
}

namespace llaminar2::test {
namespace {
/** @brief Decoded original source protocol fixture; no provider or native storage capability is minted. */
struct DistributedSourceReplay {
    Pool pool;
    q::PooledImmutableSourceGeometry geometry;
    Bytes frame,native;
    std::vector<int64_t> ids;
    std::vector<ImmutableRowSubordinateIntent> intents;
    std::vector<ImmutableRowInvalidationAcknowledgement> mappedAcks;
    int healthy=1,error=0;
    q::pooled_source::DeviceArgs args;
    /** @brief Derive complete original cohort metadata, never one representative owner as authority. */
    static q::PooledImmutableSourceGeometry sourceGeometry(Pool& pool){
        std::vector<q::PooledImmutableOwnerNativePlan> owners;for(const auto& owner:pool.owners)owners.push_back(owner->plan);
        const ImmutableRowTableSource descriptor{.path="decoded_source_replay",.tensor_name="per_layer_token_embd.weight",.file_offset=37,.rows=0x100000100ULL,.row_elements=160,.format=GGUFTensorType::IQ4_NL};
        return q::PooledImmutableSourceGeometry::fromPlans(owners,pool.plan,descriptor);
    }
    explicit DistributedSourceReplay(const std::vector<uint32_t>& capacities,std::optional<std::size_t> local=std::nullopt):pool(capacities,64,{65536,1,2},local),geometry(sourceGeometry(pool)),frame(geometry.parameters().offsets.frame_bytes),native(geometry.parameters().offsets.native_bytes),ids(16),intents(16),mappedAcks(16),args{geometry.parameters(),{geometry.parameters().owner,nullptr,nullptr,nullptr,{}},pool.args,frame.data(),native.data(),{{nullptr,nullptr,nullptr,ids.data(),nullptr,701,16,90},intents.data(),mappedAcks.data()},&healthy,&error}{
        std::memset(native.data(),0,geometry.parameters().offsets.native_bytes);q::pooled_source::protocol(args).admissionGeneration=17;q::pooled_source::invocation(args)=23;
    }
    /** @brief Execute the actual original two allocation waves and global source plan before any acquired source response. */
    void plan(std::span<const uint64_t> rows){pool.begin(rows);pool.wave<false>();pool.wave<true>();Pool::require(q::pooled_source::planRequest(args));pool.args.input.source=&q::pooled_source::frontier(args);pool.run<n::CoordinatorAction::SealSource>();pool.args.input={};}
    /** @brief Seal actual owners and acquire the resulting typed64 records, not copies of the source input. */
    void seals(){for(size_t i=0;i<pool.owners.size();++i){auto& owner=*pool.owners[i];owner.args.input.frontier=&q::pooled_source::frontier(args);Pool::require(owner.run<n::OwnerAction::SealSource>());Pool::require(q::pooled_source::acquireSeal(args,owner.header<n::OwnerSourceSealReceipt>(),i));}}
    /** @brief Supply an acquired decoded envelope/private native bytes with poison outside the actual M-row extent. */
    void response(){namespace source=q::pooled_source;auto& io=source::protocol(args);
        // A genuine all-hit/empty request issues no new source sequence. Preserve the
        // actual consumed history so the original no-request protocol can validate it.
        if(io.idCount)io.issuedSequence=source::frontier(args).source.sequence;
        io.nativeBytes=io.idCount*90;
        auto* acks=n::at<ImmutableRowInvalidationAcknowledgement>(args.frame,args.plan.offsets.invalidations);for(size_t row=0;row<io.idCount;++row){acks[row].destination_grant=intents[row].destination_grant;for(size_t byte=0;byte<90;++byte)args.frame[args.plan.offsets.responses+row*90+byte]=std::byte((ids[row]+byte)%251);}
        source::stage(args)=source::ResponseStage::Acquired;Pool::require(source::validateResponse(args));}
};
}
/** @brief Original acquired grants drive1/2/4/8-owner Fill/payload prefixes, including C0/source-only and unchanged accepted banks. */
TEST(PooledImmutableNative, DistributedSourceFillActualGrantPacketsAndTypedOutcomes){namespace source=q::pooled_source;
    for(uint32_t participants:{1u,2u,4u,8u})for(bool empty:{false,true}){std::vector<uint32_t> capacities;for(uint32_t i=0;i<participants;++i)capacities.push_back(empty || i%3==0?0:2);DistributedSourceReplay c(capacities);const std::array<uint64_t,8> rows{700,701,900,901,700,901,902,903};
        ASSERT_NO_THROW(c.plan(rows));
        ASSERT_NO_THROW(c.seals());
        ASSERT_NO_THROW(c.response());
        EXPECT_EQ(source::progress(c.args).seals,participants);uint64_t delivered=0;
        for(uint32_t owner=0;owner<participants;++owner){auto& own=c.pool.owners[owner]->args;const auto count=n::allocationFrame(own).count;std::vector<p::Grant> original(n::grants(own),n::grants(own)+count);ASSERT_TRUE(source::prepareDelivery(c.args,owner));const auto* fills=n::at<n::Fill>(c.args.frame,c.args.plan.offsets.fills);
            EXPECT_EQ(*n::at<uint64_t>(c.args.native,c.args.plan.offsets.record_bytes),count*sizeof(n::Fill));EXPECT_EQ(*n::at<uint64_t>(c.args.native,c.args.plan.offsets.payload_bytes),count*90u);for(uint32_t at=0;at<count;++at){EXPECT_TRUE(n::same(fills[at].original,original[at]));const auto* row=source::deliveryRow(c.args,at);for(size_t byte=0;byte<90;++byte)EXPECT_EQ(row[byte],std::byte((original[at].row+byte)%251));}
            delivered+=count;if(count)EXPECT_EQ(std::memcmp(original.data(),n::grants(own),count*sizeof(p::Grant)),0);
        }
        EXPECT_LE(delivered,source::protocol(c.args).idCount);EXPECT_EQ(source::progress(c.args).packets,participants);if(empty)EXPECT_EQ(delivered,0u);
    }
}
/** @brief An attempted failed seal cannot echo Success, and later source publication cannot accept it. */
TEST(PooledImmutableNative, DistributedSourceFillFailedAttemptedSealProducesActualFailureReceipt){namespace source=q::pooled_source;
    for(unsigned defect=0;defect<3;++defect){DistributedSourceReplay c(std::vector{2u,0u});
        ASSERT_NO_THROW(c.plan(std::array<uint64_t,3>{700,701,900}));auto& own=c.pool.owners[0]->args;own.input.frontier=&source::frontier(c.args);ASSERT_EQ(source::frontier(c.args).status,a::Status::Success);
        if(defect==0)n::allocationFrame(own).phase=p::AllocationPhase::Accepted;
        if(defect==1)n::ownerView<a::Slot64>(own).frame->status=a::Status::SourceFailure;
        if(defect==2){auto* input=n::at<n::Invocation>(c.pool.args.native,c.pool.args.plan.native.invocation);++input->interval.sequence;}
        EXPECT_FALSE((n::owner<n::OwnerAction::SealSource,a::Slot64>(own)));const auto receipt=c.pool.owners[0]->header<n::OwnerSourceSealReceipt>();EXPECT_NE(receipt.status,a::Status::Success);EXPECT_EQ(n::progress(own).header_bytes,64u);EXPECT_FALSE(source::acquireSeal(c.args,receipt,0));EXPECT_NE(source::frontier(c.args).status,a::Status::Success);EXPECT_EQ(source::stage(c.args),source::ResponseStage::Rejected);EXPECT_EQ(source::protocol(c.args).issuedSequence,0u);
    }
}
/** @brief Parser/copy/gate failure always publishes typed56, while the dedicated terminal64 remains authoritative. */
TEST(PooledImmutableNative, DistributedSourceFillFailureHeaderNeverAliasesTerminalOutcome){namespace source=q::pooled_source;
    for(unsigned defect=0;defect<4;++defect){DistributedSourceReplay c(std::vector{2u});
        ASSERT_NO_THROW(c.plan(std::array<uint64_t,2>{700,701}));
        ASSERT_NO_THROW(c.seals());
        ASSERT_NO_THROW(c.response());ASSERT_TRUE(source::prepareDelivery(c.args,0));auto& own=c.pool.owners[0]->args;auto outcome=source::frontier(c.args);outcome.disposition=n::SourceDisposition::Consumed;
        *n::at<n::SourceFrontier>(own.native,own.plan.native.source_outcome)=outcome;const auto saved=outcome;own.input.frontier=n::at<n::SourceFrontier>(own.native,own.plan.native.source_outcome);own.input.outcome=static_cast<const n::SourceFrontier*>(own.input.frontier);own.input.condition=n::SourceCondition::AfterSuccess;
        own.input.records=n::at<n::Fill>(c.args.frame,c.args.plan.offsets.fills);own.input.record_bytes=n::at<uint64_t>(c.args.native,c.args.plan.offsets.record_bytes);own.input.payload=c.args.frame+c.args.plan.offsets.payload;own.input.payload_bytes=n::at<uint64_t>(c.args.native,c.args.plan.offsets.payload_bytes);
        if(defect==0)++*const_cast<uint64_t*>(own.input.record_bytes);
        if(defect==1)++n::at<n::Fill>(c.args.frame,c.args.plan.offsets.fills)[0].acknowledgement.destination_token;
        if(defect<2)EXPECT_FALSE((n::owner<n::OwnerAction::ApplySourceDelivery,a::Slot64>(own)));
        if(defect==2){ASSERT_TRUE((n::owner<n::OwnerAction::ApplySourceDelivery,a::Slot64>(own)));/* recorded missing copy completion */}
        if(defect==3){auto* active=n::at<n::SourceFrontier>(own.native,own.plan.native.source_outcome);active->status=a::Status::Cancelled;}
        (void)n::owner<n::OwnerAction::ValidateFills,a::Slot64>(own);const auto receipt=c.pool.owners[0]->header<p::PublishHeader>();EXPECT_EQ(n::progress(own).header_bytes,56u);EXPECT_NE(receipt.status,a::Status::Success);EXPECT_EQ(receipt.owner,0u);const auto* retained=n::at<n::SourceFrontier>(own.native,own.plan.native.source_outcome);EXPECT_TRUE(n::same(retained->source,saved.source));EXPECT_EQ(retained->version,n::Version);EXPECT_EQ(retained->disposition,n::SourceDisposition::Consumed);
    }
}
}

namespace llaminar2::test {
/** @brief Authenticated private-copy failures drain every later zero-count packet and still consume the real response. */
TEST(PooledImmutableNative, DistributedSourceFillPacketFailureDrainsZeroMetadataAndRetiresActualResponse){namespace source=q::pooled_source;
    for(unsigned defect=0;defect<3;++defect){DistributedSourceReplay c(std::vector{2u,0u,2u});
        ASSERT_NO_THROW(c.plan(std::array<uint64_t,5>{700,701,900,901,902}));
        ASSERT_NO_THROW(c.seals());
        ASSERT_NO_THROW(c.response());
        if(defect==0){source::stage(c.args)=source::ResponseStage::Acquired;++n::at<ImmutableRowInvalidationAcknowledgement>(c.args.frame,c.args.plan.offsets.invalidations)[0].destination_grant;EXPECT_FALSE(source::validateResponse(c.args));}
        if(defect==1)EXPECT_FALSE(source::prepareDelivery(c.args,2));
        if(defect<2){EXPECT_EQ(source::stage(c.args),source::ResponseStage::Private);for(uint32_t owner=0;owner<3;++owner){EXPECT_FALSE(source::prepareDelivery(c.args,owner));EXPECT_EQ(*n::at<uint64_t>(c.args.native,c.args.plan.offsets.record_bytes),0u);EXPECT_EQ(*n::at<uint64_t>(c.args.native,c.args.plan.offsets.payload_bytes),0u);}}
        // Missing successful packet progress is an error, never an excuse to skip actual source-reader retirement.
        EXPECT_EQ(source::retirementDecision(c.args),source::ResponseRetirement::Consume);
        EXPECT_NE(source::frontier(c.args).status,a::Status::Success);
        ASSERT_TRUE(source::retireConsumed(c.args));EXPECT_EQ(source::frontier(c.args).disposition,n::SourceDisposition::Consumed);EXPECT_EQ(source::protocol(c.args).consumedSequence,source::protocol(c.args).issuedSequence);
    }
}
}

namespace llaminar2::test {
/** @brief Private-copy rejection cannot mint Consumed for another original source generation, interval or table. */
TEST(PooledImmutableNative, DistributedSourceFillRetirementRejectsForeignOriginalIdentity){namespace source=q::pooled_source;
    for(unsigned defect=0;defect<5;++defect){DistributedSourceReplay c(std::vector{2u,0u});
        ASSERT_NO_THROW(c.plan(std::array<uint64_t,3>{700,701,900}));
        ASSERT_NO_THROW(c.seals());
        ASSERT_NO_THROW(c.response());
        auto& outcome=source::frontier(c.args);
        if(defect==0)++outcome.source.generation;
        if(defect==1)++outcome.interval.sequence;
        if(defect==2)++outcome.source.table;
        if(defect==3)++outcome.version;
        if(defect==4)++source::protocol(c.args).admissionGeneration;
        source::stage(c.args)=source::ResponseStage::Acquired;EXPECT_FALSE(source::validateResponse(c.args));
        EXPECT_EQ(source::retirementDecision(c.args),source::ResponseRetirement::Invalid);EXPECT_FALSE(source::retireConsumed(c.args));EXPECT_EQ(source::protocol(c.args).consumedSequence,0u);EXPECT_EQ(outcome.disposition,n::SourceDisposition::Published);
    }
}
/** @brief The one shared source BOM rejects same-cardinality but foreign per-owner slot geometry before materialization. */
TEST(PooledImmutableNative, DistributedSourceFillPreSourceFactoryUsesOriginalCohortSlotMap){
    const auto original=declared(std::array{1u,3u});const auto foreign=declared(std::array{3u,1u});const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(original,{65536,1,2},64);
    const ImmutableRowTableSource descriptor{.path="decoded_source_replay",.tensor_name="per_layer_token_embd.weight",.file_offset=37,.rows=0x100000100ULL,.row_elements=160,.format=GGUFTensorType::IQ4_NL};
    EXPECT_NO_THROW(q::PooledImmutableSourceGeometry::rootOffsets(original,policy,descriptor));
    EXPECT_THROW(q::PooledImmutableSourceGeometry::rootOffsets(foreign,policy,descriptor),std::invalid_argument);
}
/** @brief Actual owner publication is copied to its existing original cohort slot, including C0. */
TEST(PooledImmutableNative, MixedLocalImportsOriginalPinPublicationAndRejectsPoisonedIdentityBeforeWrite)
{
    for(const auto participants:{2u,4u,8u})for(std::uint32_t local=0;local<participants;++local)
    for(unsigned mutation=0;mutation<6;++mutation){
        std::vector<std::uint32_t> capacities(participants);for(std::uint32_t i=0;i<participants;++i)capacities[i]=i%2?0:3;
        Pool pool(capacities);ASSERT_TRUE(p::CoordinatorOperations(pool.view()).begin(1,static_cast<const std::uint64_t*>(nullptr),0));n::invocation(pool.args);
        auto& owner=*pool.owners[local];owner.args.input.invocation=pool.invocation();
        ASSERT_TRUE(owner.run<n::OwnerAction::Begin>());const std::uint64_t emptyCommands=0;
        owner.args.input.record_bytes=owner.args.plan.native.reply_capacity?&emptyCommands:nullptr;
        ASSERT_TRUE(owner.run<n::OwnerAction::Pin>());owner.args.input.record_bytes=nullptr;
        ASSERT_TRUE(owner.run<n::OwnerAction::SealPins>());
        auto* source=n::at<p::PinHeader>(owner.args.native,owner.args.plan.native.header);
        const auto original=*source;auto* destination=n::at<p::PinHeader>(pool.args.native,pool.args.plan.native.ingress_frontier)+local;
        std::memset(destination,0xa5,sizeof(*destination));std::array<std::byte,sizeof(p::PinHeader)> poisoned{};
        std::memcpy(poisoned.data(),destination,poisoned.size());
        pool.args.input.frontier=source;pool.args.input.record_bytes=&n::progress(owner.args).header_bytes;
        pool.args.input.local_owner=local;pool.args.input.local_reply_capacity=owner.args.plan.native.reply_capacity;
        pool.args.input.local_control=n::ownerView<a::Slot64>(owner.args).frame;pool.args.input.local_allocation=&n::allocationFrame(owner.args);pool.args.input.local_header_kind=&n::progress(owner.args).header_kind;
        if(mutation==1)++source->interval.admission;
        if(mutation==2)source->owner=(local+1)%participants;
        if(mutation==3)--n::progress(owner.args).header_bytes;
        if(mutation==4)source->status=static_cast<a::Status>(99);
        if(mutation==5)n::progress(owner.args).header_kind=n::OwnerHeaderKind::FreeReceipt;
        const bool imported=n::coordinator<n::CoordinatorAction::ImportLocalPinHeader>(pool.args);
        if(!mutation){ASSERT_TRUE(imported);EXPECT_EQ(std::memcmp(destination,&original,sizeof(original)),0);}
        else {
            EXPECT_FALSE(imported);EXPECT_EQ(std::memcmp(destination,poisoned.data(),poisoned.size()),0);EXPECT_NE(pool.view().frame->status,a::Status::Success);
        }
    }
}
/** @brief Invalid command slicing rejects before the original physical pin mutation. */
TEST(PooledImmutableNative, MixedLocalDirectedCommandSliceRejectsBoundsBeforePhysicalMutation)
{
    for(const auto participants:{2u,4u,8u})for(std::uint32_t local=0;local<participants;++local)
    for(unsigned mutation=0;mutation<4;++mutation){
        std::vector<std::uint32_t> capacities(participants,3);Pool pool(capacities);
        ASSERT_TRUE(p::CoordinatorOperations(pool.view()).begin(1,static_cast<const std::uint64_t*>(nullptr),0));n::invocation(pool.args);
        auto& owner=*pool.owners[local];owner.args.input.invocation=pool.invocation();ASSERT_TRUE(owner.run<n::OwnerAction::Begin>());
        n::DirectedSlice slice{};const auto span=std::uint64_t{pool.args.plan.native.reply_capacity}*sizeof(p::OwnerCommand);
        owner.args.input.records=pool.args.native+pool.args.plan.native.pin_commands;
        owner.args.input.record_layout=n::RecordLayout::DirectedCommandBank;owner.args.input.allocation_slice=&slice;
        owner.args.input.record_capacity=span;
        if(mutation==1)slice.byte_offset=1;
        if(mutation==2)slice.byte_offset=span+sizeof(p::OwnerCommand);
        if(mutation==3)slice.live_bytes=span+sizeof(p::OwnerCommand);
        const auto before=n::ownerView<a::Slot64>(owner.args).frame->held_pins;
        EXPECT_EQ(owner.run<n::OwnerAction::Pin>(),mutation==0);EXPECT_EQ(n::ownerView<a::Slot64>(owner.args).frame->held_pins,before);
        EXPECT_EQ(n::progress(owner.args).reply_bytes,0u);
    }
}

/** @brief Every nominal local receipt comes from its actual owner phase and actual producer controls. */
TEST(PooledImmutableNative, MixedLocalAvailabilityFreeVictimImportsUseActualNominalProducers)
{
    for(const auto participants:{2u,4u,8u})for(bool empty:{false,true}){
        Pool pool(std::vector<std::uint32_t>(participants,empty?0:3));
            ASSERT_NO_THROW(pool.begin({}));
        const auto imported=[&]<n::CoordinatorAction Action>(std::size_t coordinate){
            auto& owner=*pool.owners[coordinate];pool.args.input={};pool.args.input.frontier=owner.args.native+owner.args.plan.native.header;
            pool.args.input.record_bytes=&n::progress(owner.args).header_bytes;pool.args.input.local_owner=coordinate;
            pool.args.input.local_reply_capacity=owner.args.plan.native.reply_capacity;pool.args.input.local_control=n::ownerView<a::Slot64>(owner.args).frame;
            pool.args.input.local_allocation=&n::allocationFrame(owner.args);pool.args.input.local_header_kind=&n::progress(owner.args).header_kind;
            EXPECT_TRUE(n::coordinator<Action>(pool.args));pool.args.input={};
        };
        ASSERT_NO_THROW(pool.all<n::OwnerAction::Availability>());
        for(std::size_t i=0;i<participants;++i)imported.template operator()<n::CoordinatorAction::ImportLocalAvailability>(i);
        pool.args.input.frontier=pool.args.native+pool.args.plan.native.ingress_frontier;
            ASSERT_NO_THROW(pool.run<n::CoordinatorAction::PlanFree>());pool.args.input={};
        for(std::size_t i=0;i<participants;++i){auto& owner=*pool.owners[i];owner.args.input.frontier=n::at<n::FreePlanHeader>(pool.args.native,pool.args.plan.native.free_headers)+i;
            ASSERT_TRUE(owner.run<n::OwnerAction::AcceptFree>());owner.args.input.frontier=nullptr;
            ASSERT_TRUE(owner.run<n::OwnerAction::ExecuteFree>());
            ASSERT_TRUE(owner.run<n::OwnerAction::FreeReceipt>());
            imported.template operator()<n::CoordinatorAction::ImportLocalFreeReceipt>(i);
        }
        pool.args.input.frontier=pool.args.native+pool.args.plan.native.ingress_frontier;
            ASSERT_NO_THROW(pool.run<n::CoordinatorAction::CompleteFree>());pool.args.input={};
        for(auto& owner:pool.owners){owner->args.input.frontier=pool.args.native+pool.args.plan.native.ingress_frontier;
            ASSERT_TRUE(owner->run<n::OwnerAction::AcquireFree>());owner->args.input.frontier=nullptr;}
        ASSERT_NO_THROW(pool.run<n::CoordinatorAction::PlanVictims>());
        for(std::size_t i=0;i<participants;++i){auto& owner=*pool.owners[i];owner.args.input.frontier=n::at<n::VictimPlanHeader>(pool.args.native,pool.args.plan.native.victim_headers)+i;
            ASSERT_TRUE(owner.run<n::OwnerAction::AcceptVictim>());owner.args.input.frontier=nullptr;
            ASSERT_TRUE(owner.run<n::OwnerAction::ExecuteVictim>());
            ASSERT_TRUE(owner.run<n::OwnerAction::VictimReceipt>());
            imported.template operator()<n::CoordinatorAction::ImportLocalVictimReceipt>(i);
        }
        pool.args.input.frontier=pool.args.native+pool.args.plan.native.ingress_frontier;
            ASSERT_NO_THROW(pool.run<n::CoordinatorAction::CompleteVictims>());pool.args.input={};
        ASSERT_NO_THROW(pool.finish());
            EXPECT_EQ(pool.view().frame->phase,p::LookupPhase::Idle);
    }
}
/** @brief Actual original compact P1 path and direct slice reader retain complete byte results across hit/reset/replay. */
TEST(PooledImmutableNative, MixedLocalDirectedPinMatchesOriginalBytesForP1AndOriginalCohortOrder)
{
    for(const auto participants:{1u,2u,4u,8u})for(std::uint32_t local=0;local<participants;++local){
        SCOPED_TRACE(::testing::Message()<<"participants="<<participants<<" local="<<local);
        const std::vector<std::uint32_t> capacities(participants,3);Pool original(capacities),candidate(capacities);candidate.direct_pin_owner=local;
        // Each admitted query has at most the original K distinct rows. Two
        // disjoint warm families can fill a P8 cache larger than one K bucket.
        for(unsigned family=0;family<2;++family){std::vector<std::uint64_t> warm;
            for(std::uint64_t i=0;i<original.args.plan.policy.maximumUnique();++i)warm.push_back(0x100000000ULL+100+family*64+i);
            for(unsigned replay=0;replay<4;++replay){
                SCOPED_TRACE(::testing::Message()<<"family="<<family<<" warm="<<replay<<" K="<<warm.size());
                ASSERT_NO_THROW(original.complete(warm));
                ASSERT_NO_THROW(candidate.complete(warm));
            }
        }
        std::vector<std::uint64_t> hits;
        for(auto& owner:original.owners){auto view=n::ownerView<a::Slot64>(owner->args);for(std::uint32_t i=0;i<owner->args.plan.owner.capacity();++i)
            if(a::PhysicalSlot<a::Slot64>::state(view.slots[i])==a::SlotState::Ready){hits.push_back(view.slots[i].row);break;}}
        ASSERT_FALSE(hits.empty());
        for(unsigned replay=0;replay<4;++replay){const std::vector<std::uint64_t> request=replay==0 || replay==3?hits:replay==1?std::vector<std::uint64_t>{hits.front()}:std::vector<std::uint64_t>{};
            ASSERT_NO_THROW(original.complete(request,replay==3?2:1));
            ASSERT_NO_THROW(candidate.complete(request,replay==3?2:1));
            EXPECT_EQ(std::memcmp(original.args.cache,candidate.args.cache,original.args.plan.cache.bytes),0);
            EXPECT_EQ(std::memcmp(original.args.frame,candidate.args.frame,original.args.plan.frame.bytes),0);
            for(std::size_t i=0;i<participants;++i){const auto& a=original.owners[i]->args;const auto& b=candidate.owners[i]->args;
                EXPECT_EQ(std::memcmp(a.cache,b.cache,a.plan.cache.bytes),0);
            EXPECT_EQ(std::memcmp(a.frame,b.frame,a.plan.frame.bytes),0);
            EXPECT_EQ(std::memcmp(a.native,b.native,a.plan.native.bytes),0);}
        }
        EXPECT_GT(candidate.observed_direct_pin_bytes,0u);
        if(local)EXPECT_GT(candidate.observed_direct_pin_offset,0u);
    }
}

/** @brief A real observed owner error stays an error in the complete local frontier. */
TEST(PooledImmutableNative, MixedLocalActualErrorReceiptNeverBecomesSuccess)
{
    Pool pool(std::array{3u,0u});
        ASSERT_TRUE(p::CoordinatorOperations(pool.view()).begin(1,static_cast<const std::uint64_t*>(nullptr),0));n::invocation(pool.args);
    for(auto& owner:pool.owners){owner->args.input.invocation=pool.invocation();
        ASSERT_TRUE(owner->run<n::OwnerAction::Begin>());
        const std::uint64_t zero=0;owner->args.input.record_bytes=owner->args.plan.native.reply_capacity?&zero:nullptr;
        ASSERT_TRUE(owner->run<n::OwnerAction::Pin>());owner->args.input.record_bytes=nullptr;
        ASSERT_TRUE(owner->run<n::OwnerAction::SealPins>());}
    ASSERT_FALSE(pool.owners.front()->run<n::OwnerAction::SealPins>());
    for(std::size_t i=0;i<pool.owners.size();++i){auto& owner=*pool.owners[i];pool.args.input={};pool.args.input.frontier=owner.args.native+owner.args.plan.native.header;
        pool.args.input.record_bytes=&n::progress(owner.args).header_bytes;pool.args.input.local_owner=i;pool.args.input.local_reply_capacity=owner.args.plan.native.reply_capacity;
        pool.args.input.local_control=n::ownerView<a::Slot64>(owner.args).frame;pool.args.input.local_allocation=&n::allocationFrame(owner.args);pool.args.input.local_header_kind=&n::progress(owner.args).header_kind;
        ASSERT_TRUE(n::coordinator<n::CoordinatorAction::ImportLocalPinHeader>(pool.args));}
    const auto* frontier=n::at<p::PinHeader>(pool.args.native,pool.args.plan.native.ingress_frontier);
    EXPECT_EQ(frontier[0].status,a::Status::InvalidPhase);pool.args.input={};pool.args.input.frontier=frontier;
    EXPECT_FALSE(n::coordinator<n::CoordinatorAction::SealPins>(pool.args));
        EXPECT_NE(pool.view().frame->phase,p::LookupPhase::Availability);
    auto& untouched=*pool.owners.back();untouched.args.input.frontier=frontier;
        EXPECT_FALSE(untouched.run<n::OwnerAction::AcquirePins>());
    EXPECT_NE(n::ownerView<a::Slot64>(untouched.args).frame->phase,p::OwnerPhase::Reserving);
}
/** @brief Identical40B layouts cannot turn an actual Free producer into a successful Victim publication. */
TEST(PooledImmutableNative, MixedLocalWrongNominalAllocationProducerRejectsBeforeFrontierWrite)
{
    for(bool cancelled:{false,true}){
    SCOPED_TRACE(cancelled);
    Pool pool(std::array{3u,0u});
        ASSERT_NO_THROW(pool.begin({}));
        ASSERT_NO_THROW(pool.all<n::OwnerAction::Availability>());
    const auto availability=pool.headers<n::Availability>();pool.args.input.frontier=availability.data();
        ASSERT_NO_THROW(pool.run<n::CoordinatorAction::PlanFree>());pool.args.input={};
    for(std::size_t i=0;i<pool.owners.size();++i){auto& owner=*pool.owners[i];owner.args.input.frontier=n::at<n::FreePlanHeader>(pool.args.native,pool.args.plan.native.free_headers)+i;
        ASSERT_TRUE(owner.run<n::OwnerAction::AcceptFree>());owner.args.input.frontier=nullptr;
        ASSERT_TRUE(owner.run<n::OwnerAction::ExecuteFree>());
        if(cancelled && i==0)ASSERT_TRUE(owner.run<n::OwnerAction::CancelFree>());
        ASSERT_TRUE(owner.run<n::OwnerAction::FreeReceipt>());}
    const auto receipts=pool.headers<p::FreeAllocationReceipt>();pool.args.input.frontier=receipts.data();
        ASSERT_NO_THROW(pool.run<n::CoordinatorAction::CompleteFree>());pool.args.input={};
    ASSERT_NO_THROW(pool.run<n::CoordinatorAction::PlanVictims>());
    auto& owner=*pool.owners.front();
        ASSERT_EQ(n::allocationFrame(owner.args).wave,p::AllocationWave::Free);
    ASSERT_EQ(n::allocationFrame(owner.args).phase,cancelled?p::AllocationPhase::Cancelled:p::AllocationPhase::Executed);
    auto* destination=pool.args.native+pool.args.plan.native.ingress_frontier;std::array<std::byte,sizeof(p::VictimAllocationReceipt)> poison{};poison.fill(std::byte{0xa5});std::memcpy(destination,poison.data(),poison.size());
    pool.args.input.frontier=owner.args.native+owner.args.plan.native.header;pool.args.input.record_bytes=&n::progress(owner.args).header_bytes;
    pool.args.input.local_owner=0;pool.args.input.local_reply_capacity=owner.args.plan.native.reply_capacity;
    pool.args.input.local_control=n::ownerView<a::Slot64>(owner.args).frame;pool.args.input.local_allocation=&n::allocationFrame(owner.args);pool.args.input.local_header_kind=&n::progress(owner.args).header_kind;
    EXPECT_FALSE(n::coordinator<n::CoordinatorAction::ImportLocalVictimReceipt>(pool.args));
        EXPECT_EQ(std::memcmp(destination,poison.data(),poison.size()),0);
    ASSERT_DEATH(n::submitCoordinator<n::CoordinatorAction::ImportLocalVictimReceipt>(pool.args),"");
    }
}

/** @brief A real new Begin invalidates both original publication operands before stale header bytes can escape. */
TEST(PooledImmutableNative, MixedLocalResetInvalidatesOriginalHeaderKindAndExtent)
{
    for(const auto participants:{2u,4u,8u})for(std::uint32_t local=0;local<participants;++local){
        Pool pool(std::vector<std::uint32_t>(participants,3));
        ASSERT_NO_THROW(pool.complete({}));
        auto& owner=*pool.owners[local];
        ASSERT_EQ(n::progress(owner.args).header_kind,n::OwnerHeaderKind::Retired);
        ASSERT_EQ(n::progress(owner.args).header_bytes,sizeof(p::RetiredHeader));
        ASSERT_TRUE(p::CoordinatorOperations(pool.view()).begin(2,static_cast<const std::uint64_t*>(nullptr),0));n::invocation(pool.args);
        owner.args.input.invocation=pool.invocation();
        ASSERT_TRUE(owner.run<n::OwnerAction::Begin>());
        EXPECT_EQ(n::progress(owner.args).header_kind,n::OwnerHeaderKind::None);
        EXPECT_EQ(n::progress(owner.args).header_bytes,0u);
        auto* destination=pool.args.native+pool.args.plan.native.ingress_frontier+std::size_t{local}*sizeof(p::PinHeader);
        std::array<std::byte,sizeof(p::PinHeader)> poison{};poison.fill(std::byte{0xa5});std::memcpy(destination,poison.data(),poison.size());
        pool.args.input.frontier=owner.args.native+owner.args.plan.native.header;pool.args.input.record_bytes=&n::progress(owner.args).header_bytes;
        pool.args.input.local_owner=local;pool.args.input.local_reply_capacity=owner.args.plan.native.reply_capacity;
        pool.args.input.local_control=n::ownerView<a::Slot64>(owner.args).frame;pool.args.input.local_allocation=&n::allocationFrame(owner.args);
        pool.args.input.local_header_kind=&n::progress(owner.args).header_kind;
        EXPECT_FALSE(n::coordinator<n::CoordinatorAction::ImportLocalPinHeader>(pool.args));
        EXPECT_EQ(std::memcmp(destination,poison.data(),poison.size()),0);
    }
}

/** @brief Every actual import template stops native continuation for an invalid publication. */
TEST(PooledImmutableNative, MixedLocalEveryImportSubmissionRejectsMissingActualPublication)
{
    Pool pool(std::array{3u,0u});
    EXPECT_DEATH(n::submitCoordinator<n::CoordinatorAction::ImportLocalPinHeader>(pool.args),"");
    EXPECT_DEATH(n::submitCoordinator<n::CoordinatorAction::ImportLocalAvailability>(pool.args),"");
    EXPECT_DEATH(n::submitCoordinator<n::CoordinatorAction::ImportLocalFreeReceipt>(pool.args),"");
    EXPECT_DEATH(n::submitCoordinator<n::CoordinatorAction::ImportLocalVictimReceipt>(pool.args),"");
}

}

namespace llaminar2::test {
/** @brief Mixed actual pre/post-seal cancellation drains original ownership without fabricating a shared SourceWindow. */
TEST(PooledImmutableNative, DistributedSourceFillMixedSealFailureWaitsForExactWorkerThenDrainsActualPhases){namespace source=q::pooled_source;
    DistributedSourceReplay c(std::vector{2u,2u});
    ASSERT_NO_THROW(c.plan(std::array<uint64_t,3>{700,701,900}));
    auto& sealed=*c.pool.owners[0];auto& cancelled=*c.pool.owners[1];sealed.args.input.frontier=&source::frontier(c.args);ASSERT_TRUE(sealed.run<n::OwnerAction::SealSource>());ASSERT_TRUE(source::acquireSeal(c.args,sealed.header<n::OwnerSourceSealReceipt>(),0));
    ASSERT_TRUE(cancelled.run<n::OwnerAction::CancelVictim>());cancelled.args.input.frontier=&source::frontier(c.args);EXPECT_FALSE(cancelled.run<n::OwnerAction::SealSource>());const auto receipt=cancelled.header<n::OwnerSourceSealReceipt>();EXPECT_EQ(receipt.status,a::Status::Cancelled);EXPECT_EQ(receipt.source.table,0u);EXPECT_FALSE(source::acquireSeal(c.args,receipt,1));
    struct RecordedWorker {uint64_t ticket=0,generation=0,completed=0;uint64_t retiredInvocation(){return ticket;}uint64_t retiredGeneration(){return generation;}uint64_t completion(){return completed;}} witness;
    EXPECT_FALSE(source::retireAborted(c.args,witness));witness.ticket=23;witness.generation=18;EXPECT_FALSE(source::retireAborted(c.args,witness));witness.ticket=22;witness.generation=17;EXPECT_FALSE(source::retireAborted(c.args,witness));EXPECT_EQ(source::frontier(c.args).disposition,n::SourceDisposition::Published);witness.ticket=23;ASSERT_TRUE(source::retireAborted(c.args,witness));EXPECT_EQ(source::frontier(c.args).disposition,n::SourceDisposition::Aborted);EXPECT_EQ(source::protocol(c.args).issuedSequence,0u);EXPECT_EQ(source::protocol(c.args).consumedSequence,0u);
    for(size_t i=0;i<c.pool.owners.size();++i){auto& owner=*c.pool.owners[i];const auto count=n::allocationFrame(owner.args).count;const std::vector<p::Grant> original(n::grants(owner.args),n::grants(owner.args)+count);owner.args.input.frontier=&source::frontier(c.args);owner.args.input.condition=n::SourceCondition::Always;owner.args.input.outcome=nullptr;ASSERT_TRUE(owner.run<n::OwnerAction::DrainSourceFailure>());EXPECT_EQ(n::ownerView<a::Slot64>(owner.args).frame->phase,p::OwnerPhase::Draining);EXPECT_EQ(n::progress(owner.args).retirement,i?p::SourceRetirement::Unpublished:p::SourceRetirement::Aborted);EXPECT_EQ(n::ownerView<a::Slot64>(owner.args).frame->source.table,i?0u:701u);EXPECT_EQ(owner.header<p::PublishHeader>().status,a::Status::Cancelled);if(count)EXPECT_EQ(std::memcmp(original.data(),n::grants(owner.args),count*sizeof(p::Grant)),0);owner.args.input={c.pool.invocation(),nullptr,nullptr,nullptr,nullptr,nullptr};ASSERT_TRUE(owner.run<n::OwnerAction::Release>());ASSERT_TRUE(owner.run<n::OwnerAction::Retire>());EXPECT_EQ(n::ownerView<a::Slot64>(owner.args).frame->held_pins,0u);EXPECT_EQ(n::ownerView<a::Slot64>(owner.args).frame->reservations,0u);}
    // Global mixed-source retirement/Finish composition is item6, not fabricated by this owner-fill component.
}
}

namespace llaminar2::test {
/** @brief Frozen actual r5 operation bodies causally lack the guaranteed receipts; current wrapper closes both ignored-bool failures. */
TEST(PooledImmutableNative, DistributedSourceFillOriginalOperationCausalControls){namespace source=q::pooled_source;
    {DistributedSourceReplay c(std::vector{2u});ASSERT_NO_THROW(c.plan(std::array<uint64_t,2>{7,9}));auto& own=c.pool.owners[0]->args;own.input.frontier=&source::frontier(c.args);n::allocationFrame(own).phase=p::AllocationPhase::Accepted;
        EXPECT_FALSE((n::originalSourceOutcomeOwner<n::OwnerAction::SealSource,a::Slot64>(own)));EXPECT_EQ(n::progress(own).header_bytes,0u);EXPECT_EQ(source::frontier(c.args).status,a::Status::Success); // Blind echo would still claim success.
        EXPECT_FALSE((n::owner<n::OwnerAction::SealSource,a::Slot64>(own)));EXPECT_EQ(n::progress(own).header_bytes,sizeof(n::OwnerSourceSealReceipt));EXPECT_NE(c.pool.owners[0]->header<n::OwnerSourceSealReceipt>().status,a::Status::Success);}
    {DistributedSourceReplay c(std::vector{2u});ASSERT_NO_THROW(c.plan(std::array<uint64_t,2>{7,9}));
        ASSERT_NO_THROW(c.seals());auto& own=c.pool.owners[0]->args;auto terminal=source::frontier(c.args);terminal.disposition=n::SourceDisposition::Consumed;*n::at<n::SourceFrontier>(own.native,own.plan.native.source_outcome)=terminal;*n::at<n::SourceFrontier>(own.native,own.plan.native.header)=terminal;own.input.frontier=nullptr;own.input.condition=n::SourceCondition::AfterSuccess;own.input.outcome=n::at<n::SourceFrontier>(own.native,own.plan.native.source_outcome);
        EXPECT_FALSE((n::originalSourceOutcomeOwner<n::OwnerAction::ValidateFills,a::Slot64>(own)));EXPECT_EQ(n::progress(own).header_bytes,0u);EXPECT_EQ(std::memcmp(own.native+own.plan.native.header,&terminal,sizeof(terminal)),0); // Old early return leaves64-byte terminal header in place.
        EXPECT_FALSE((n::owner<n::OwnerAction::ValidateFills,a::Slot64>(own)));EXPECT_EQ(n::progress(own).header_bytes,sizeof(p::PublishHeader));EXPECT_NE(c.pool.owners[0]->header<p::PublishHeader>().status,a::Status::Success);EXPECT_EQ(std::memcmp(own.native+own.plan.native.source_outcome,&terminal,sizeof(terminal)),0);}
}
}

namespace llaminar2::test {
/**
 * @brief Common parser-only local-last packaging retains its final packet across Consumed and actual Apply/copyComplete.
 * This is deliberately not mixed-local graph qualification: the earlier-wave
 * participant-local adapter is separately guarded and not constructed here.
 */
TEST(PooledImmutableNative, DistributedSourceFillCommonLocalLastPacketSurvivesConsumedThroughActualCopy){namespace source=q::pooled_source;
    for(uint32_t participants:{1u,2u,4u,8u})for(uint32_t local:{0u,participants/2,participants-1}){std::vector<uint32_t> capacities(participants,2);DistributedSourceReplay c(capacities,local);
        ASSERT_NO_THROW(c.plan(std::array<uint64_t,6>{7,9,3,5,7,11}));
        ASSERT_NO_THROW(c.seals());
        ASSERT_NO_THROW(c.response());
        EXPECT_EQ(c.args.plan.local_owner,local);std::vector<std::vector<p::Grant>> accepted;for(const auto& owner:c.pool.owners){const auto count=n::allocationFrame(owner->args).count;accepted.emplace_back(n::grants(owner->args),n::grants(owner->args)+count);}
        std::vector<uint32_t> order;for(uint32_t ordinal=0;ordinal<participants;++ordinal){const auto owner=source::packetOwner(c.args,ordinal);order.push_back(owner);ASSERT_TRUE(source::prepareDelivery(c.args,owner));auto* payload=c.args.frame+c.args.plan.offsets.payload;const auto count=*n::at<uint64_t>(c.args.native,c.args.plan.offsets.payload_bytes)/90;for(uint64_t row=0;row<count;++row)std::memcpy(payload+row*90,source::deliveryRow(c.args,row),90);}
        EXPECT_EQ(order.back(),local);std::vector<uint32_t> remote;for(uint32_t owner=0;owner<participants;++owner)if(owner!=local)remote.push_back(owner);EXPECT_TRUE(std::equal(remote.begin(),remote.end(),order.begin()));auto& own=c.pool.owners[local]->args;const auto bytes=*n::at<uint64_t>(c.args.native,c.args.plan.offsets.payload_bytes),records=*n::at<uint64_t>(c.args.native,c.args.plan.offsets.record_bytes);std::vector<std::byte> retained(c.args.frame+c.args.plan.offsets.payload,c.args.frame+c.args.plan.offsets.payload+bytes);
        ASSERT_TRUE(source::retireConsumed(c.args));EXPECT_EQ(std::memcmp(retained.data(),c.args.frame+c.args.plan.offsets.payload,bytes),0);own.input.frontier=&source::frontier(c.args);own.input.outcome=&source::frontier(c.args);own.input.condition=n::SourceCondition::AfterSuccess;own.input.records=c.args.frame+c.args.plan.offsets.fills;own.input.record_bytes=n::at<uint64_t>(c.args.native,c.args.plan.offsets.record_bytes);own.input.payload=c.args.frame+c.args.plan.offsets.payload;own.input.payload_bytes=n::at<uint64_t>(c.args.native,c.args.plan.offsets.payload_bytes);ASSERT_TRUE((n::owner<n::OwnerAction::ApplySourceDelivery,a::Slot64>(own)));const auto* fills=n::at<n::Fill>(c.args.frame,c.args.plan.offsets.fills);for(uint64_t row=0;row<records/sizeof(n::Fill);++row)std::memcpy(own.cache+own.plan.cache.payload+fills[row].original.slot*own.plan.slot_stride,own.input.payload+row*90,90);ASSERT_TRUE(n::copied<a::Slot64>(own));ASSERT_TRUE((n::owner<n::OwnerAction::ValidateFills,a::Slot64>(own)));EXPECT_EQ(c.pool.owners[local]->header<p::PublishHeader>().status,a::Status::Success);
        for(size_t owner=0;owner<participants;++owner)if(!accepted[owner].empty())EXPECT_EQ(std::memcmp(accepted[owner].data(),n::grants(c.pool.owners[owner]->args),accepted[owner].size()*sizeof(p::Grant)),0);
    }
}
}

namespace llaminar2::test {
namespace {
/**
 * @brief Replay actual r6 producers through copied fills before the new publication successor.
 * Exact native bytes and wire acquisitions are decoded CPU observations only;
 * no provider, Tensor device storage, transport, capture or native runtime is certified.
 */
void publicationFilled(DistributedSourceReplay& c,std::span<const uint64_t> rows){namespace source=q::pooled_source;
    c.plan(rows);c.seals();c.response();
    for(uint32_t i=0;i<c.pool.owners.size();++i){Pool::require(source::prepareDelivery(c.args,i));auto& own=c.pool.owners[i]->args;const auto records=*n::at<uint64_t>(c.args.native,c.args.plan.offsets.record_bytes),bytes=*n::at<uint64_t>(c.args.native,c.args.plan.offsets.payload_bytes);auto* payload=c.args.frame+c.args.plan.offsets.payload;for(uint64_t row=0;row<records/sizeof(n::Fill);++row)std::memcpy(payload+row*90,source::deliveryRow(c.args,row),90);if(records)std::memcpy(own.native+own.plan.native.source_fills,c.args.frame+c.args.plan.offsets.fills,records);if(bytes)std::memcpy(own.native+own.plan.native.source_payload,payload,bytes);auto* input=n::at<n::OwnerIngress>(own.native,own.plan.native.ingress);input->record_bytes=records;input->payload_bytes=bytes;}
    Pool::require(source::retireConsumed(c.args));
    for(auto& original:c.pool.owners){auto& own=original->args;auto* outcome=n::at<n::SourceFrontier>(own.native,own.plan.native.source_outcome);*outcome=source::frontier(c.args);auto* input=n::at<n::OwnerIngress>(own.native,own.plan.native.ingress);own.input.frontier=outcome;own.input.outcome=outcome;own.input.condition=n::SourceCondition::AfterSuccess;own.input.records=own.plan.native.reply_capacity?own.native+own.plan.native.source_fills:nullptr;own.input.record_bytes=own.plan.native.reply_capacity?&input->record_bytes:nullptr;own.input.payload=own.plan.native.reply_capacity?own.native+own.plan.native.source_payload:nullptr;own.input.payload_bytes=own.plan.native.reply_capacity?&input->payload_bytes:nullptr;Pool::require((n::owner<n::OwnerAction::ApplySourceDelivery,a::Slot64>(own)));const auto* fills=n::at<n::Fill>(own.native,own.plan.native.source_fills);for(uint64_t row=0;row<input->record_bytes/sizeof(n::Fill);++row)std::memcpy(own.cache+own.plan.cache.payload+fills[row].original.slot*own.plan.slot_stride,own.input.payload+row*90,90);Pool::require(n::copied<a::Slot64>(own));Pool::require((n::owner<n::OwnerAction::ValidateFills,a::Slot64>(own)));}
}
/** @brief Acquire the actual ALL-owner fill frontier; keep each original source outcome after header reuse. */
void publicationAcquire(DistributedSourceReplay& c){const auto publish=c.pool.headers<p::PublishHeader>();for(auto& original:c.pool.owners){auto& own=original->args;const auto* invocation=own.input.invocation;own.input={invocation,nullptr,nullptr,publish.data(),nullptr,nullptr,n::SourceCondition::AfterSuccess,n::at<n::SourceFrontier>(own.native,own.plan.native.source_outcome)};Pool::require((n::owner<n::OwnerAction::AcquirePublication,a::Slot64>(own)));own.input.frontier=nullptr;}}
/** @brief Actual Ready producers over original physical pins/reservations; no fixed-size header echo is supplied. */
void publicationReady(DistributedSourceReplay& c){for(auto& original:c.pool.owners){Pool::require((n::owner<n::OwnerAction::MakeReady,a::Slot64>(original->args)));Pool::require((n::owner<n::OwnerAction::ReadyReceipt,a::Slot64>(original->args)));}}
}
/** @brief Exact original all-owner publication succeeds for1/2/4/8 including C0/empty without releasing pins or private responses. */
TEST(PooledImmutableNative, PublicationReadyActualConsensusKeepsOriginalReadersAndGrants){namespace source=q::pooled_source;
    for(uint32_t participants:{1u,2u,4u,8u})for(bool empty:{false,true}){std::vector<uint32_t> capacities;for(uint32_t i=0;i<participants;++i)capacities.push_back(empty || i%3==0?0:2);DistributedSourceReplay c(capacities);const std::array<uint64_t,8> rows{700,701,900,901,700,901,902,903};
        ASSERT_NO_THROW(publicationFilled(c,empty?std::span<const uint64_t>{}:std::span<const uint64_t>{rows}));
        std::vector<std::byte> privateRows(c.args.frame+c.args.plan.offsets.responses,c.args.frame+c.args.plan.offsets.responses+source::protocol(c.args).nativeBytes);std::vector<uint32_t> pins;for(const auto& own:c.pool.owners)pins.push_back(n::ownerView<a::Slot64>(own->args).frame->held_pins);
        ASSERT_NO_THROW(publicationAcquire(c));
        ASSERT_TRUE(source::beforeReady(c.args));
        ASSERT_NO_THROW(publicationReady(c));const auto ready=c.pool.headers<p::ReadyHeader>();
        ASSERT_TRUE(source::acquireReadyConsensus(c.args,ready.data(),ready.size()));EXPECT_EQ(source::stage(c.args),source::ResponseStage::Consumed);EXPECT_EQ(c.pool.view().frame->phase,p::LookupPhase::Source);c.pool.args.input.frontier=ready.data();c.pool.args.input.source=&source::frontier(c.args);
        ASSERT_NO_THROW(c.pool.run<n::CoordinatorAction::Publish>());EXPECT_EQ(c.pool.view().frame->phase,p::LookupPhase::Ready);EXPECT_EQ(std::memcmp(privateRows.data(),c.args.frame+c.args.plan.offsets.responses,privateRows.size()),0);for(size_t i=0;i<c.pool.owners.size();++i){const auto& own=c.pool.owners[i]->args;EXPECT_EQ(n::ownerView<a::Slot64>(own).frame->held_pins,pins[i]);EXPECT_EQ(n::progress(own).header_bytes,56u);EXPECT_EQ(n::ownerView<a::Slot64>(own).frame->reservations,0u);}
    }
}
/** @brief Gate cancellation publishes a real Ready error even while physical owner status remainsSuccess; r6 old body leaves stale Publish56. */
TEST(PooledImmutableNative, PublicationReadyCancelledGateOverwritesActualOldPublishHeader){namespace source=q::pooled_source;DistributedSourceReplay c(std::vector{2u});
    ASSERT_NO_THROW(publicationFilled(c,std::array<uint64_t,2>{7,9}));
    ASSERT_NO_THROW(publicationAcquire(c));auto& own=c.pool.owners[0]->args;auto* outcome=n::at<n::SourceFrontier>(own.native,own.plan.native.source_outcome);outcome->status=a::Status::Cancelled;const auto prior=c.pool.owners[0]->header<p::PublishHeader>();ASSERT_EQ(prior.status,a::Status::Success);
    EXPECT_TRUE((n::ownerOperation<n::OwnerAction::ReadyReceipt,a::Slot64>(own)));EXPECT_EQ(n::progress(own).header_bytes,0u);EXPECT_EQ(std::memcmp(own.native+own.plan.native.header,&prior,sizeof(prior)),0);
    EXPECT_FALSE((n::owner<n::OwnerAction::ReadyReceipt,a::Slot64>(own)));EXPECT_EQ(n::progress(own).header_bytes,56u);EXPECT_EQ(c.pool.owners[0]->header<p::ReadyHeader>().status,a::Status::Cancelled);EXPECT_EQ(n::ownerView<a::Slot64>(own).frame->status,a::Status::Success);EXPECT_EQ(n::ownerView<a::Slot64>(own).frame->reservations,2u);EXPECT_EQ(source::stage(c.args),source::ResponseStage::Consumed);
}
/** @brief Lost physical readability cannot be certified by layout/phase alone even when old readyHeader reportsSuccess. */
TEST(PooledImmutableNative, PublicationReadyRevalidatesEveryOriginalPhysicalPin){DistributedSourceReplay c(std::vector{2u});
    ASSERT_NO_THROW(publicationFilled(c,std::array<uint64_t,2>{7,9}));
    ASSERT_NO_THROW(publicationAcquire(c));
    ASSERT_NO_THROW(publicationReady(c));auto& own=c.pool.owners[0]->args;const auto grant=n::grants(own)[0];a::PhysicalSlot<a::Slot64>::unpin(n::ownerView<a::Slot64>(own).slots[grant.slot]);EXPECT_TRUE((n::ownerOperation<n::OwnerAction::ReadyReceipt,a::Slot64>(own)));EXPECT_EQ(c.pool.owners[0]->header<p::ReadyHeader>().status,a::Status::Success);
    EXPECT_FALSE((n::owner<n::OwnerAction::ReadyReceipt,a::Slot64>(own)));EXPECT_EQ(c.pool.owners[0]->header<p::ReadyHeader>().status,a::Status::SourceFailure);EXPECT_EQ(n::ownerView<a::Slot64>(own).frame->held_pins,2u); // No invented release repairs the invalid trace.
}
/** @brief Foreign/partial/malformed Ready observations cannot publish or retire an original frame; acquired source errors remain sticky. */
TEST(PooledImmutableNative, PublicationReadyConsensusRejectsIncompleteAndForeignBeforeCommit){namespace source=q::pooled_source;
    for(unsigned defect=0;defect<9;++defect){DistributedSourceReplay c(std::vector{1u,0u,2u});
        ASSERT_NO_THROW(publicationFilled(c,std::array<uint64_t,3>{7,9,11}));
        ASSERT_NO_THROW(publicationAcquire(c));
        ASSERT_NO_THROW(publicationReady(c));auto headers=c.pool.headers<p::ReadyHeader>();auto count=static_cast<uint32_t>(headers.size());
        if(defect==0)--count;if(defect==1)++headers[1].owner;if(defect==2)++headers[1].interval.sequence;if(defect==3)++headers[1].source.generation;if(defect==4)headers[1].status=a::Status::Cancelled;if(defect==5)++source::frontier(c.args).version;if(defect==6)++source::protocol(c.args).admissionGeneration;if(defect==7){source::frontier(c.args).status=a::Status::Cancelled;}if(defect==8){source::stage(c.args)=source::ResponseStage::ReadersComplete;}
        const auto consumed=source::protocol(c.args).consumedSequence;EXPECT_FALSE(source::acquireReadyConsensus(c.args,headers.data(),count));EXPECT_NE(source::frontier(c.args).status,a::Status::Success);if(defect==7)EXPECT_EQ(source::frontier(c.args).status,a::Status::Cancelled);EXPECT_EQ(c.pool.view().frame->phase,p::LookupPhase::Source);EXPECT_EQ(c.pool.view().frame->status,a::Status::Success);EXPECT_EQ(source::protocol(c.args).consumedSequence,consumed);
    }
}
}

namespace llaminar2::test {
namespace {
/** @brief Only decoded transport copies the exact live60 bytes; native root padding is deliberately not supplied. */
template<class Header,n::OwnerAction Action>void publicationImport(DistributedSourceReplay& c,uint32_t owner){auto& a=c.pool.owners[owner]->args;const auto* invocation=a.input.invocation;a.input={invocation,nullptr,nullptr,nullptr,nullptr,nullptr};Pool::require((n::owner<Action,a::Slot64>(a)));std::memcpy(c.args.native+c.args.plan.offsets.seal_receipt,a.native+a.plan.native.header,n::NominalHeaderWireBytes);Pool::require(q::pooled_source::importPublicationWire<Header>(c.args,owner));}
}
/** @brief Actual nominal producers fit60 live bytes in original64-byte banks, with source readers/pins retained across consensus. */
TEST(PooledImmutableNative, PublicationReadyNominalWireUsesOriginalWitnessAndExact60){namespace source=q::pooled_source;
    static_assert(n::NominalHeaderWireBytes==60 && sizeof(n::NominalHeaderWire<p::PublishHeader>)==64);
    static_assert(sizeof(n::OwnerProgress)==64 && offsetof(n::OwnerProgress,header_kind)==48);
    for(uint32_t participants:{1u,2u,4u,8u}){std::vector<uint32_t> capacities(participants,2);capacities[0]=0;DistributedSourceReplay c(capacities);
        ASSERT_NO_THROW(publicationFilled(c,std::array<uint64_t,4>{7,9,11,7}));
        for(uint32_t owner=0;owner<participants;++owner){auto& a=c.pool.owners[owner]->args;*n::at<uint32_t>(a.native,a.plan.native.header+60)=0xdecafbad;ASSERT_EQ(n::progress(a).header_kind,n::OwnerHeaderKind::Publish);
            ASSERT_NO_THROW((publicationImport<p::PublishHeader,n::OwnerAction::PublishWire>(c,owner)));EXPECT_EQ(*n::at<uint32_t>(a.native,a.plan.native.header+60),0xdecafbadu);EXPECT_EQ(n::progress(a).header_kind,n::OwnerHeaderKind::Publish);EXPECT_EQ(n::progress(a).header_bytes,56u);}
        ASSERT_TRUE(source::acquirePublicationConsensus<p::PublishHeader>(c.args));
        ASSERT_NO_THROW(publicationAcquire(c));for(uint32_t owner=0;owner<participants;++owner){ASSERT_EQ(n::progress(c.pool.owners[owner]->args).header_kind,n::OwnerHeaderKind::PublicationAcquired);
            ASSERT_NO_THROW((publicationImport<n::PublicationAcquiredHeader,n::OwnerAction::PublicationAcquiredWire>(c,owner)));}
        ASSERT_TRUE(source::acquirePublicationConsensus<n::PublicationAcquiredHeader>(c.args));ASSERT_TRUE(source::beforeReady(c.args));
        ASSERT_NO_THROW(publicationReady(c));for(uint32_t owner=0;owner<participants;++owner){
            ASSERT_NO_THROW((publicationImport<p::ReadyHeader,n::OwnerAction::ReadyWire>(c,owner)));}
        ASSERT_TRUE(source::acquirePublicationConsensus<p::ReadyHeader>(c.args));const auto* ready=n::at<p::ReadyHeader>(c.pool.args.native,c.pool.args.plan.native.ingress_frontier);ASSERT_TRUE(source::acquireReadyConsensus(c.args,ready,participants));EXPECT_EQ(source::stage(c.args),source::ResponseStage::Consumed);EXPECT_EQ(source::progress(c.args).publication_headers,participants);EXPECT_EQ(source::progress(c.args).publication_acquired,participants);EXPECT_EQ(source::progress(c.args).ready_headers,participants);
        c.pool.args.input.frontier=ready;c.pool.args.input.source=&source::frontier(c.args);
        ASSERT_NO_THROW(c.pool.run<n::CoordinatorAction::Publish>());EXPECT_EQ(c.pool.view().frame->phase,p::LookupPhase::Ready);
    }
}
/** @brief The ordinary reset-before-pack body destroys the original witness; new wire projections execute before that reset and invalidate errors. */
TEST(PooledImmutableNative, PublicationReadyWireRejectsResetWrongRoleStaleAndPartialProducer){
    for(unsigned defect=0;defect<6;++defect){DistributedSourceReplay c(std::vector{2u});
        ASSERT_NO_THROW(publicationFilled(c,std::array<uint64_t,2>{7,9}));auto& own=c.pool.owners[0]->args;const auto* invocation=own.input.invocation;own.input={invocation,nullptr,nullptr,nullptr,nullptr,nullptr};
        *n::at<n::OwnerHeaderKind>(own.native,own.plan.native.header+56)=n::OwnerHeaderKind::Ready;
        if(defect==0){n::progress(own).header_bytes=0;n::progress(own).header_kind=n::OwnerHeaderKind::None;} // Exact ordinary operation reset, before original witness acquisition.
        if(defect==1)n::progress(own).header_kind=n::OwnerHeaderKind::Ready;
        if(defect==2)--n::progress(own).header_bytes;
        if(defect==3)++n::at<p::PublishHeader>(own.native,own.plan.native.header)->interval.sequence;
        if(defect==4)++n::at<p::PublishHeader>(own.native,own.plan.native.header)->source.generation;
        if(defect==5)own.input.condition=n::SourceCondition::AfterSuccess;
        EXPECT_FALSE((n::owner<n::OwnerAction::PublishWire,a::Slot64>(own)));EXPECT_EQ(n::progress(own).header_kind,n::OwnerHeaderKind::None);EXPECT_EQ(n::progress(own).header_bytes,0u);EXPECT_EQ(*n::at<n::OwnerHeaderKind>(own.native,own.plan.native.header+56),n::OwnerHeaderKind::None);EXPECT_EQ(n::ownerView<a::Slot64>(own).frame->held_pins,2u);
    }
}
/** @brief Equal56-byte layouts alone let the original wire parser publish before physical Ready; real nominal imports reject that causal trace. */
TEST(PooledImmutableNative, PublicationReadyNominalKindRejectsOriginalLayoutOnlyFalseReady){namespace source=q::pooled_source;
    {DistributedSourceReplay original(std::vector{2u});ASSERT_NO_THROW(publicationFilled(original,std::array<uint64_t,2>{7,9}));const auto publish=original.pool.headers<p::PublishHeader>();static_assert(sizeof(p::PublishHeader)==sizeof(p::ReadyHeader));std::vector<p::ReadyHeader> decoded(publish.size());std::memcpy(decoded.data(),publish.data(),publish.size()*sizeof(p::PublishHeader));EXPECT_TRUE(p::CoordinatorOperations(original.pool.view()).publish(decoded.data(),decoded.size(),p::SourceRetirement::Consumed,source::frontier(original.args).source));EXPECT_EQ(original.pool.view().frame->phase,p::LookupPhase::Ready);EXPECT_EQ(n::ownerView<a::Slot64>(original.pool.owners[0]->args).frame->reservations,2u);}
    {DistributedSourceReplay candidate(std::vector{2u});ASSERT_NO_THROW(publicationFilled(candidate,std::array<uint64_t,2>{7,9}));auto& own=candidate.pool.owners[0]->args;
        ASSERT_NO_THROW((publicationImport<p::PublishHeader,n::OwnerAction::PublishWire>(candidate,0)));std::array<std::byte,60> oldPublish;std::memcpy(oldPublish.data(),candidate.args.native+candidate.args.plan.offsets.seal_receipt,oldPublish.size());
        ASSERT_NO_THROW(publicationAcquire(candidate));
        ASSERT_NO_THROW((publicationImport<n::PublicationAcquiredHeader,n::OwnerAction::PublicationAcquiredWire>(candidate,0)));ASSERT_EQ(source::progress(candidate.args).publication_headers,1u);ASSERT_EQ(source::progress(candidate.args).publication_acquired,1u);std::memcpy(candidate.args.native+candidate.args.plan.offsets.seal_receipt,oldPublish.data(),oldPublish.size());EXPECT_FALSE(source::importPublicationWire<p::ReadyHeader>(candidate.args,0));EXPECT_EQ(source::progress(candidate.args).ready_headers,0u);EXPECT_EQ(candidate.pool.view().frame->phase,p::LookupPhase::Source);EXPECT_EQ(candidate.pool.view().frame->status,a::Status::Success);EXPECT_EQ(n::ownerView<a::Slot64>(own).frame->reservations,2u);}
}
/** @brief Publication acquisition acknowledgment is actual compute completion; cancelled gate emits its real error kind without inventing state. */
TEST(PooledImmutableNative, PublicationReadyAcquiredReceiptHasActualGateAndPhase){DistributedSourceReplay c(std::vector{2u});
    ASSERT_NO_THROW(publicationFilled(c,std::array<uint64_t,2>{7,9}));auto& own=c.pool.owners[0]->args;const auto* input=own.input.invocation;auto* outcome=n::at<n::SourceFrontier>(own.native,own.plan.native.source_outcome);outcome->status=a::Status::Cancelled;const auto publish=c.pool.headers<p::PublishHeader>();own.input={input,nullptr,nullptr,publish.data(),nullptr,nullptr,n::SourceCondition::AfterSuccess,outcome};
    EXPECT_FALSE((n::owner<n::OwnerAction::AcquirePublication,a::Slot64>(own)));EXPECT_EQ(n::progress(own).header_kind,n::OwnerHeaderKind::PublicationAcquired);EXPECT_EQ(c.pool.owners[0]->header<n::PublicationAcquiredHeader>().status,a::Status::Cancelled);EXPECT_EQ(n::ownerView<a::Slot64>(own).frame->status,a::Status::Success);EXPECT_EQ(n::ownerView<a::Slot64>(own).frame->phase,p::OwnerPhase::FillValidated);EXPECT_EQ(n::ownerView<a::Slot64>(own).frame->held_pins,2u);
}
}

namespace llaminar2::test {
/** @brief Direct-local metadata uses actual original owner root/kind/byte publication, while remote members retain exact60 envelopes. */
TEST(PooledImmutableNative, PublicationReadyLocalOriginalNominalWitnessAcrossCohorts){namespace source=q::pooled_source;for(size_t count:{2u,4u,8u})for(uint32_t local=0;local<count;++local){DistributedSourceReplay c(std::vector<uint32_t>(count,2));
    ASSERT_NO_THROW(publicationFilled(c,std::array<uint64_t,4>{7,9,11,13}));
    c.args.plan.local_owner=local;c.args.owner=c.pool.owners[local]->args;
    for(uint32_t i=0;i<count;++i){if(i==local)ASSERT_TRUE(source::importLocalPublication<p::PublishHeader>(c.args,i));else ASSERT_NO_THROW((publicationImport<p::PublishHeader,n::OwnerAction::PublishWire>(c,i)));}
    ASSERT_TRUE(source::acquirePublicationConsensus<p::PublishHeader>(c.args));
    ASSERT_NO_THROW(publicationAcquire(c));
    for(uint32_t i=0;i<count;++i){if(i==local)ASSERT_TRUE(source::importLocalPublication<n::PublicationAcquiredHeader>(c.args,i));else ASSERT_NO_THROW((publicationImport<n::PublicationAcquiredHeader,n::OwnerAction::PublicationAcquiredWire>(c,i)));}
    ASSERT_TRUE(source::acquirePublicationConsensus<n::PublicationAcquiredHeader>(c.args));
    ASSERT_NO_THROW(publicationReady(c));
    for(uint32_t i=0;i<count;++i){if(i==local)ASSERT_TRUE(source::importLocalPublication<p::ReadyHeader>(c.args,i));else ASSERT_NO_THROW((publicationImport<p::ReadyHeader,n::OwnerAction::ReadyWire>(c,i)));}
    ASSERT_TRUE(source::acquirePublicationConsensus<p::ReadyHeader>(c.args));
    EXPECT_EQ(source::stage(c.args),source::ResponseStage::Consumed);EXPECT_EQ(source::progress(c.args).ready_headers,count);
}}
/** @brief Equal-layout stale/partial/wrong nominal local records reject before the original ordered slot/cursor changes. */
TEST(PooledImmutableNative, PublicationReadyLocalRejectsWrongOriginalProducerBeforeImport){namespace source=q::pooled_source;for(unsigned defect=0;defect<6;++defect){DistributedSourceReplay c(std::vector<uint32_t>{2,0});
    ASSERT_NO_THROW(publicationFilled(c,std::array<uint64_t,2>{7,9}));
    c.args.plan.local_owner=0;c.args.owner=c.pool.owners[0]->args;auto& own=c.args.owner;
    if(defect==0)n::progress(own).header_kind=n::OwnerHeaderKind::Ready;
    if(defect==1)n::progress(own).header_bytes=55;
    if(defect==2)n::at<p::PublishHeader>(own.native,own.plan.native.header)->interval.sequence++;
    if(defect==3)n::at<p::PublishHeader>(own.native,own.plan.native.header)->source.generation++;
    if(defect==4)n::allocationFrame(own).phase=p::AllocationPhase::Idle;
    if(defect==5)c.args.owner=c.pool.owners[1]->args;
    const auto before=*n::at<p::PublishHeader>(c.args.coordinator.native,c.args.coordinator.plan.native.ingress_frontier);
    EXPECT_FALSE(source::importLocalPublication<p::PublishHeader>(c.args,0));EXPECT_EQ(source::progress(c.args).publication_headers,0u);EXPECT_EQ(std::memcmp(&before,n::at<p::PublishHeader>(c.args.coordinator.native,c.args.coordinator.plan.native.ingress_frontier),sizeof(before)),0);
}}
}

namespace llaminar2::test {
/** @brief Failed unsealed owners may truthfully retain an old window; error metadata drains without granting it source authority. */
TEST(PooledImmutableNative, PublicationReadyFailedOwnerOldWindowDrainsOnlyErrorMetadata){namespace source=q::pooled_source;for(bool local:{false,true}){DistributedSourceReplay c(std::vector<uint32_t>{2,0});
    ASSERT_NO_THROW(publicationFilled(c,std::array<uint64_t,2>{7,9}));
    auto& own=c.pool.owners[0]->args;auto& frame=*n::ownerView<a::Slot64>(own).frame;const auto original=source::frontier(c.args).source;frame.source.generation++;frame.status=a::Status::SourceFailure;n::header(own,p::PublishHeader{frame.interval,frame.source,0,a::Status::SourceFailure});
    if(local){c.args.plan.local_owner=0;c.args.owner=own;EXPECT_FALSE(source::importLocalPublication<p::PublishHeader>(c.args,0));}else{const auto* invocation=own.input.invocation;own.input={invocation,nullptr,nullptr,nullptr,nullptr,nullptr};ASSERT_TRUE((n::owner<n::OwnerAction::PublishWire,a::Slot64>(own)));std::memcpy(c.args.native+c.args.plan.offsets.seal_receipt,own.native+own.plan.native.header,n::NominalHeaderWireBytes);EXPECT_FALSE(source::importPublicationWire<p::PublishHeader>(c.args,0));}
    EXPECT_EQ(source::progress(c.args).publication_headers,1u);EXPECT_EQ(source::stage(c.args),source::ResponseStage::Consumed);EXPECT_TRUE(n::same(source::frontier(c.args).source,original));EXPECT_NE(source::frontier(c.args).status,a::Status::Success);EXPECT_EQ(c.pool.view().frame->phase,p::LookupPhase::Source);
}}
}

namespace llaminar2::test {
/**
 * @brief Actual mixed sealed/unsealed cancellation reconciles only complete original release evidence.
 * @note Decoded original worker retirement drives the real response state machine; no source
 * window is added to an unsealed owner and no pin count is inferred from coordinator policy.
 */
TEST(PooledImmutableNative, TerminalMixedCancelledRetirementRequiresCompleteOriginalReleaseSet)
{
    namespace source=q::pooled_source;
    for(uint32_t participants:{2u,4u,8u})for(uint32_t unsealed:{0u,participants/2,participants-1})
    for(bool empty:{false,true})for(unsigned corruption=0;corruption<8;++corruption){
        std::vector<uint32_t> capacities(participants,empty?0u:2u);
        DistributedSourceReplay c(capacities);
        ASSERT_NO_THROW(c.plan(std::array<uint64_t,3>{700,701,900}));
        for(uint32_t coordinate=0;coordinate<participants;++coordinate){
            auto& owner=*c.pool.owners[coordinate];owner.args.input.frontier=&source::frontier(c.args);
            if(coordinate==unsealed){
                ASSERT_TRUE(owner.run<n::OwnerAction::CancelVictim>());
                EXPECT_FALSE(owner.run<n::OwnerAction::SealSource>());
            }else{
                ASSERT_TRUE(owner.run<n::OwnerAction::SealSource>());
                // The first failure remains sticky; later actual receipts still identify real physical windows.
                const auto actual=owner.header<n::OwnerSourceSealReceipt>();
                EXPECT_EQ(actual.source.table,701u);
            }
        }
        for(uint32_t coordinate=0;coordinate<participants;++coordinate){
            const bool acquired=source::acquireSeal(c.args,c.pool.owners[coordinate]->header<n::OwnerSourceSealReceipt>(),coordinate);
            EXPECT_EQ(acquired,coordinate<unsealed);
        }
        struct OriginalWorkerRetirement {
            uint64_t retiredInvocation(){return 23;} uint64_t retiredGeneration(){return 17;}
            uint64_t completion(){return 0;}
        } worker;
        ASSERT_TRUE(source::retireAborted(c.args,worker));
        std::vector<p::ReleaseReceipt> releases;
        for(auto& pointer:c.pool.owners){auto& owner=*pointer;auto& args=owner.args;
            args.input.frontier=&source::frontier(c.args);args.input.condition=n::SourceCondition::Always;args.input.outcome=nullptr;
            ASSERT_TRUE(owner.run<n::OwnerAction::DrainSourceFailure>());
            args.input={c.pool.invocation(),nullptr,nullptr,nullptr,nullptr,nullptr};
            ASSERT_TRUE(owner.run<n::OwnerAction::Release>());
            const auto count=n::progress(args).release_bytes/sizeof(p::ReleaseReceipt);
            const auto* original=n::at<p::ReleaseReceipt>(args.frame,args.plan.frame.release_receipts);
            releases.insert(releases.end(),original,original+count);
            EXPECT_EQ(n::ownerView<a::Slot64>(args).frame->held_pins,0u);
            EXPECT_EQ(n::ownerView<a::Slot64>(args).frame->reservations,0u);
        }
        const bool corruptRelease=corruption==1 && !releases.empty();
        if(corruptRelease)releases.pop_back();
        const uint64_t bytes=releases.size()*sizeof(p::ReleaseReceipt);
        const auto& outcome=source::frontier(c.args);
        c.pool.args.input={releases.empty()?nullptr:releases.data(),c.pool.args.plan.native.reply_capacity?&bytes:nullptr,nullptr,&outcome,n::SourceCondition::AfterFailure,&outcome};
        const bool cancelled=corruption==7?false:n::coordinator<n::CoordinatorAction::Cancel>(c.pool.args);
        EXPECT_EQ(cancelled,!corruptRelease && corruption!=7);c.pool.args.input={};
        for(auto& owner:c.pool.owners)ASSERT_TRUE(owner->run<n::OwnerAction::Retire>());
        auto retired=c.pool.headers<p::RetiredHeader>();
        EXPECT_EQ(retired[unsealed].source.table,0u);EXPECT_EQ(retired[unsealed].status,a::Status::Cancelled);
        if(corruption==2)retired[unsealed].source.sequence=1; // Not an all-zero absent window.
        if(corruption==3)retired[unsealed].status=a::Status::Success; // Cannot recast failed unsealed work as success.
        if(corruption==4)++retired[unsealed].interval.admission;
        if(corruption==5)retired[unsealed].owner=(unsealed+1)%participants;
        if(corruption==6)++retired[(unsealed+1)%participants].source.sequence;
        c.pool.args.input.frontier=retired.data();
        const bool finished=n::coordinator<n::CoordinatorAction::Finish>(c.pool.args);
        EXPECT_EQ(finished,!corruptRelease && (corruption==0 || corruption==1));
        EXPECT_EQ(c.pool.view().frame->phase==p::LookupPhase::Idle,finished);
        if(finished){
            c.pool.args.input={};
            ASSERT_NO_THROW(c.pool.complete(std::array<uint64_t,1>{99},2));
            EXPECT_EQ(c.pool.view().frame->phase,p::LookupPhase::Idle);
        }
    }
}
}

namespace llaminar2::test {
/** @brief Authentic cancelled allocation-wave completion cannot substitute for complete release reconciliation. */
TEST(PooledImmutableNative, TerminalCancelledWaveAloneNeverAuthorizesFinish)
{
    for(bool empty:{false,true}){
        Pool pool(std::vector<uint32_t>{empty?0u:2u,0u});
        ASSERT_NO_THROW(pool.begin(std::array<uint64_t,2>{701,702}));
        ASSERT_NO_THROW(pool.receivedFree({},{},true));
        EXPECT_TRUE(pool.view().frame->cancelled_completion);
        EXPECT_EQ(pool.view().frame->status,a::Status::Success);
        EXPECT_EQ(pool.view().frame->phase,p::LookupPhase::FreeCompleted);
        ASSERT_NO_THROW(pool.all<n::OwnerAction::Release>());
        ASSERT_NO_THROW(pool.all<n::OwnerAction::Retire>());
        const auto retired=pool.headers<p::RetiredHeader>();pool.args.input.frontier=retired.data();
        EXPECT_FALSE(n::coordinator<n::CoordinatorAction::Finish>(pool.args));
        EXPECT_NE(pool.view().frame->phase,p::LookupPhase::Idle);
    }
}
/**
 * @brief Actual Release72 parsers phase-reuse the serial Fill bank and preserve original physical counters.
 * @note The recorded input ReadersComplete state represents the separately authenticated output/read-drain
 * graph frontier. This device-free parser replay does not certify that upstream graph or a GPU completion.
 */
TEST(PooledImmutableNative, TerminalReleaseAggregationUsesExactOriginalPacketsAndRejectsBeforeAppend)
{
    namespace source=q::pooled_source;
    for(uint32_t participants:{2u,4u,8u})for(bool empty:{false,true})for(unsigned defect=0;defect<9;++defect){
        DistributedSourceReplay c(std::vector<uint32_t>(participants,empty?0u:2u));
        ASSERT_NO_THROW(c.plan(std::array<uint64_t,6>{701,702,703,704,705,706}));
        ASSERT_NO_THROW(c.seals());
        struct OriginalWorkerRetirement {uint64_t retiredInvocation(){return 23;}uint64_t retiredGeneration(){return 17;}uint64_t completion(){return 0;}} worker;
        ASSERT_TRUE(source::retireAborted(c.args,worker));
        uint64_t total=0;
        for(auto& pointer:c.pool.owners){auto& owner=*pointer;auto& args=owner.args;
            args.input.frontier=&source::frontier(c.args);args.input.condition=n::SourceCondition::Always;args.input.outcome=nullptr;
            ASSERT_TRUE(owner.run<n::OwnerAction::DrainSourceFailure>());
            args.input={c.pool.invocation(),nullptr,nullptr,nullptr,nullptr,nullptr};
            ASSERT_TRUE(owner.run<n::OwnerAction::Release>());total+=n::progress(args).release_bytes;
        }
        const auto priorStage=source::stage(c.args);
        EXPECT_FALSE(source::planReleases(c.args));EXPECT_EQ(source::stage(c.args),priorStage);
        source::stage(c.args)=source::ResponseStage::ReadersComplete; // Recorded actual last-reader input, not an invented event/retirement capability.
        if(defect>=6){
            if(defect==6)++source::protocol(c.args).admissionGeneration;
            if(defect==7)++source::frontier(c.args).interval.admission;
            if(defect==8)++source::frontier(c.args).version;
            const auto before=source::releasedBytes(c.args);
            EXPECT_FALSE(source::planReleases(c.args));EXPECT_EQ(source::releasedBytes(c.args),before);
            EXPECT_EQ(source::stage(c.args),source::ResponseStage::ReadersComplete);continue;
        }
        ASSERT_TRUE(source::planReleases(c.args));EXPECT_EQ(source::releasedBytes(c.args),0u);
        bool rejected=false;
        for(uint32_t coordinate=0;coordinate<participants;++coordinate){
            auto& owner=c.pool.owners[coordinate]->args;const auto actual=n::progress(owner).release_bytes;
            const auto* original=n::at<p::ReleaseReceipt>(owner.frame,owner.plan.frame.release_receipts);
            auto* received=n::at<p::ReleaseReceipt>(c.args.frame,c.args.plan.offsets.fills);
            if(actual)std::memcpy(received,original,actual);
            auto bytes=actual;auto addressed=coordinate;
            const bool corrupt=coordinate==0 && defect && (defect==1 || actual);
            if(corrupt){
                if(defect==1)addressed=1;
                if(defect==2)--bytes;
                if(defect==3)++received[0].original.destination_token;
                if(defect==4)received[0].reserved=1;
                if(defect==5)received[0].disposition=p::ReleaseDisposition::Failed;
            }
            const std::vector<p::ReleaseReceipt> retained(original,original+actual/sizeof(p::ReleaseReceipt));
            const auto before=source::releasedBytes(c.args);const auto cursor=source::progress(c.args).packets;
            const bool accepted=source::importReleases(c.args,addressed,actual?received:nullptr,bytes);
            EXPECT_EQ(accepted,!corrupt);
            EXPECT_EQ(n::progress(owner).release_bytes,actual);
            if(actual)EXPECT_EQ(std::memcmp(retained.data(),owner.frame+owner.plan.frame.release_receipts,actual),0);
            if(corrupt){EXPECT_EQ(source::releasedBytes(c.args),before);EXPECT_EQ(source::progress(c.args).packets,cursor);rejected=true;break;}
            if(coordinate==0){const auto retainedBytes=source::releasedBytes(c.args);const auto retainedCursor=source::progress(c.args).packets;
                EXPECT_FALSE(source::planReleases(c.args));EXPECT_EQ(source::releasedBytes(c.args),retainedBytes);EXPECT_EQ(source::progress(c.args).packets,retainedCursor);}
        }
        if(rejected)continue;
        EXPECT_EQ(source::releasedBytes(c.args),total);
        ASSERT_TRUE(source::reconcileReleases(c.args));
        EXPECT_EQ(c.pool.view().frame->phase,p::LookupPhase::Draining);
        EXPECT_EQ(c.pool.view().frame->status,a::Status::Cancelled);
        EXPECT_EQ(source::stage(c.args),source::ResponseStage::ReleasesComplete);
        std::array<int32_t,sampling_math::kDeviceGenerationControlCount> control{};c.args.control=control.data();
        ASSERT_TRUE(source::snapshotReleasedObservation(c.args));
        ASSERT_NO_THROW(c.pool.all<n::OwnerAction::Retire>());
        for(uint32_t coordinate=0;coordinate<participants;++coordinate){
            const auto retired=c.pool.owners[coordinate]->header<p::RetiredHeader>();
            std::memcpy(c.args.native+c.args.plan.offsets.seal_receipt,&retired,sizeof(retired));
            *n::at<n::OwnerHeaderKind>(c.args.native,c.args.plan.offsets.seal_receipt+sizeof(retired))=n::OwnerHeaderKind::Retired;
            ASSERT_TRUE(source::importRemoteRetired(c.args,coordinate));
        }
        ASSERT_TRUE(source::finishReleased(c.args));EXPECT_EQ(c.pool.view().frame->phase,p::LookupPhase::Idle);
        ASSERT_TRUE(source::observeReleased(c.args));
        q::QwenAdaptiveCounterTotals observed;ASSERT_TRUE(q::readQwenAdaptiveCounters(control.data(),control.size(),observed));
        EXPECT_EQ(observed.calls,1u);EXPECT_EQ(observed.failed_frames,1u);EXPECT_EQ(observed.completed_live_ids,0u);
        EXPECT_EQ(observed.promotions,0u);EXPECT_EQ(observed.source_bytes,0u);
        const auto previous=control;EXPECT_FALSE(source::observeReleased(c.args));EXPECT_EQ(control,previous);
        EXPECT_EQ(source::stage(c.args),source::ResponseStage::ReadersComplete);
    }
}
}

namespace llaminar2::test {
/** @brief Actual local Idle/kind/count publication is mandatory before its poisoned full-cohort slot changes. */
TEST(PooledImmutableNative, TerminalLocalNominalRetirementRejectsPoisonBeforeFrontierWrite)
{
    namespace source=q::pooled_source;
    for(uint32_t participants:{2u,4u,8u})for(uint32_t local=0;local<participants;++local)
    for(bool empty:{false,true})for(unsigned defect=0;defect<6;++defect){
        DistributedSourceReplay c(std::vector<uint32_t>(participants,empty?0u:2u),local);
        ASSERT_NO_THROW(c.plan(std::array<uint64_t,3>{701,702,703}));
        ASSERT_NO_THROW(c.seals());
        struct OriginalWorkerRetirement {uint64_t retiredInvocation(){return 23;}uint64_t retiredGeneration(){return 17;}uint64_t completion(){return 0;}} worker;
        ASSERT_TRUE(source::retireAborted(c.args,worker));
        for(auto& pointer:c.pool.owners){auto& owner=*pointer;owner.args.input.frontier=&source::frontier(c.args);
            owner.args.input.condition=n::SourceCondition::Always;owner.args.input.outcome=nullptr;
            ASSERT_TRUE(owner.run<n::OwnerAction::DrainSourceFailure>());
            owner.args.input={c.pool.invocation(),nullptr,nullptr,nullptr,nullptr,nullptr};
            ASSERT_TRUE(owner.run<n::OwnerAction::Release>());
        }
        source::stage(c.args)=source::ResponseStage::ReadersComplete; // Recorded real last-reader boundary; concrete graph proof is separate.
        ASSERT_TRUE(source::planReleases(c.args));c.args.owner=c.pool.owners[local]->args;
        for(uint32_t coordinate=0;coordinate<participants;++coordinate){auto& owner=c.pool.owners[coordinate]->args;
            if(coordinate==local)ASSERT_TRUE(source::importLocalReleases<a::Slot64>(c.args));
            else{
                const auto bytes=n::progress(owner).release_bytes;
                if(bytes)std::memcpy(c.args.frame+c.args.plan.offsets.fills,owner.frame+owner.plan.frame.release_receipts,bytes);
                *n::at<uint64_t>(c.args.native,c.args.plan.offsets.record_bytes)=bytes;
                ASSERT_TRUE(source::importRemoteReleases(c.args,coordinate));
            }
        }
        ASSERT_TRUE(source::reconcileReleases(c.args));
        ASSERT_NO_THROW(c.pool.all<n::OwnerAction::Retire>());
        for(uint32_t coordinate=0;coordinate<participants;++coordinate){auto& owner=c.pool.owners[coordinate]->args;
            if(coordinate==local){
                auto* slot=n::at<p::RetiredHeader>(c.pool.args.native,c.pool.args.plan.native.ingress_frontier)+coordinate;
                std::memset(slot,0xa5,sizeof(*slot));std::array<std::byte,sizeof(p::RetiredHeader)> poisoned{};std::memcpy(poisoned.data(),slot,sizeof(*slot));
                if(defect==1)n::progress(owner).header_kind=n::OwnerHeaderKind::Ready;
                if(defect==2)n::progress(owner).header_bytes=32;
                if(defect==3)n::ownerView<a::Slot64>(owner).frame->phase=p::OwnerPhase::Draining;
                if(defect==4)++n::at<p::RetiredHeader>(owner.native,owner.plan.native.header)->interval.admission;
                if(defect==5)++n::at<p::RetiredHeader>(owner.native,owner.plan.native.header)->source.sequence;
                const bool imported=source::importLocalRetired<a::Slot64>(c.args);EXPECT_EQ(imported,defect==0);
                if(defect){EXPECT_EQ(std::memcmp(slot,poisoned.data(),sizeof(*slot)),0);EXPECT_EQ(source::progress(c.args).seals,coordinate);EXPECT_FALSE(source::finishReleased(c.args));break;}
            }else{
                const auto record=c.pool.owners[coordinate]->header<p::RetiredHeader>();
                std::memcpy(c.args.native+c.args.plan.offsets.seal_receipt,&record,sizeof(record));
                *n::at<n::OwnerHeaderKind>(c.args.native,c.args.plan.offsets.seal_receipt+sizeof(record))=n::progress(owner).header_kind;
                ASSERT_TRUE(source::importRemoteRetired(c.args,coordinate));
            }
        }
        if(!defect){ASSERT_TRUE(source::finishReleased(c.args));EXPECT_EQ(c.pool.view().frame->phase,p::LookupPhase::Idle);}
    }
}
}

namespace llaminar2::test {
/**
 * @brief A single independently located owner uses actual source/Ready and exact remote terminal parsing.
 * @note ReadersComplete is the recorded real output boundary for this common replay;
 * original source planning/response/consumption and physical Ready are executed, never assigned.
 */
TEST(PooledImmutableNative, TerminalP1RemoteActualReadyReleaseRetireFinishAcrossReset)
{
    namespace source=q::pooled_source;
    for(uint32_t capacity:{0u,2u}){
        DistributedSourceReplay c(std::vector<uint32_t>{capacity});
        ASSERT_EQ(c.args.plan.coordinator.policy.participants(),1u);
        ASSERT_EQ(c.args.plan.local_owner,a::None);
        ASSERT_NE(c.pool.owners[0]->plan.allocator(),c.pool.plan.allocator());
        std::array<int32_t,sampling_math::kDeviceGenerationControlCount> controller{};c.args.control=controller.data();
        const std::array<std::vector<uint64_t>,4> requests{{{701,702,703,701},{701},{},{0x100000003ULL}}};
        uint64_t completed=0,sourceBytes=0;
        for(size_t request=0;request<requests.size();++request){
            ASSERT_NO_THROW(publicationFilled(c,requests[request]));
            ASSERT_NO_THROW(publicationAcquire(c));
            ASSERT_TRUE(source::beforeReady(c.args));
            ASSERT_NO_THROW(publicationReady(c));
            const auto ready=c.pool.headers<p::ReadyHeader>();
            ASSERT_TRUE(source::acquireReadyConsensus(c.args,ready.data(),ready.size()));
            c.pool.args.input.frontier=ready.data();c.pool.args.input.source=&source::frontier(c.args);
            ASSERT_NO_THROW(c.pool.run<n::CoordinatorAction::Publish>());c.pool.args.input={};
            ASSERT_EQ(c.pool.view().frame->phase,p::LookupPhase::Ready);
            auto& owner=*c.pool.owners[0];auto& args=owner.args;
            ASSERT_TRUE(owner.run<n::OwnerAction::Release>());
            const auto bytes=n::progress(args).release_bytes;
            const std::vector<std::byte> retained(args.frame+args.plan.frame.release_receipts,args.frame+args.plan.frame.release_receipts+bytes);
            sourceBytes+=source::protocol(c.args).nativeBytes;completed+=requests[request].size();
            source::stage(c.args)=source::ResponseStage::ReadersComplete;
            ASSERT_TRUE(source::planReleases(c.args));
            if(bytes)std::memcpy(c.args.frame+c.args.plan.offsets.fills,retained.data(),bytes);
            *n::at<uint64_t>(c.args.native,c.args.plan.offsets.record_bytes)=bytes;
            ASSERT_TRUE(source::importRemoteReleases(c.args,0));EXPECT_EQ(source::releasedBytes(c.args),bytes);
            ASSERT_TRUE(source::reconcileReleases(c.args));EXPECT_EQ(c.pool.view().frame->phase,p::LookupPhase::Ready);
            ASSERT_TRUE(source::snapshotReleasedObservation(c.args));
            ASSERT_TRUE(owner.run<n::OwnerAction::Retire>());
            const auto retired=owner.header<p::RetiredHeader>();EXPECT_EQ(retired.status,a::Status::Success);
            std::memcpy(c.args.native+c.args.plan.offsets.seal_receipt,&retired,sizeof(retired));
            *n::at<n::OwnerHeaderKind>(c.args.native,c.args.plan.offsets.seal_receipt+sizeof(retired))=n::progress(args).header_kind;
            ASSERT_TRUE(source::importRemoteRetired(c.args,0));
            ASSERT_TRUE(source::finishReleased(c.args));
            ASSERT_TRUE(source::observeReleased(c.args));
            EXPECT_EQ(n::ownerView<a::Slot64>(args).frame->held_pins,0u);EXPECT_EQ(n::ownerView<a::Slot64>(args).frame->reservations,0u);
            if(bytes)EXPECT_EQ(std::memcmp(retained.data(),args.frame+args.plan.frame.release_receipts,bytes),0);
            q::QwenAdaptiveCounterTotals actual;ASSERT_TRUE(q::readQwenAdaptiveCounters(controller.data(),controller.size(),actual));
            EXPECT_EQ(actual.calls,request+1);EXPECT_EQ(actual.completed_live_ids,completed);EXPECT_EQ(actual.source_bytes,sourceBytes);EXPECT_EQ(actual.failed_frames,0u);
            EXPECT_EQ(source::stage(c.args),source::ResponseStage::ReadersComplete);EXPECT_EQ(c.pool.view().frame->phase,p::LookupPhase::Idle);
        }
    }
}
/** @brief Original same-device P1 remains on its original local path; remote P1 unsealed failures retire truthfully. */
TEST(PooledImmutableNative, TerminalP1RemoteCancellationAndOriginalLocalBoundary)
{
    namespace source=q::pooled_source;
    for(bool local:{false,true})for(uint32_t capacity:{0u,2u}){
        DistributedSourceReplay c(std::vector<uint32_t>{capacity},local?std::optional<size_t>{0}:std::nullopt);
        EXPECT_EQ(c.args.plan.local_owner,local?0u:a::None);
        ASSERT_NO_THROW(c.plan(std::array<uint64_t,2>{701,702}));auto& owner=*c.pool.owners[0];
        ASSERT_TRUE(owner.run<n::OwnerAction::CancelVictim>());owner.args.input.frontier=&source::frontier(c.args);
        EXPECT_FALSE(owner.run<n::OwnerAction::SealSource>());EXPECT_FALSE(source::acquireSeal(c.args,owner.header<n::OwnerSourceSealReceipt>(),0));
        struct Worker {uint64_t retiredInvocation(){return 23;}uint64_t retiredGeneration(){return 17;}uint64_t completion(){return 0;}} worker;
        ASSERT_TRUE(source::retireAborted(c.args,worker));owner.args.input.frontier=&source::frontier(c.args);owner.args.input.condition=n::SourceCondition::Always;owner.args.input.outcome=nullptr;
        ASSERT_TRUE(owner.run<n::OwnerAction::DrainSourceFailure>());
        owner.args.input={c.pool.invocation(),nullptr,nullptr,nullptr,nullptr,nullptr};ASSERT_TRUE(owner.run<n::OwnerAction::Release>());
        source::stage(c.args)=source::ResponseStage::ReadersComplete;
        if(local){const auto before=n::progress(owner.args).release_bytes;EXPECT_FALSE(source::planReleases(c.args));EXPECT_EQ(n::progress(owner.args).release_bytes,before);continue;}
        ASSERT_TRUE(source::planReleases(c.args));const auto bytes=n::progress(owner.args).release_bytes;
        if(bytes)std::memcpy(c.args.frame+c.args.plan.offsets.fills,owner.args.frame+owner.args.plan.frame.release_receipts,bytes);
        *n::at<uint64_t>(c.args.native,c.args.plan.offsets.record_bytes)=bytes;ASSERT_TRUE(source::importRemoteReleases(c.args,0));
        ASSERT_TRUE(source::reconcileReleases(c.args));ASSERT_TRUE(owner.run<n::OwnerAction::Retire>());
        const auto retired=owner.header<p::RetiredHeader>();EXPECT_EQ(retired.source.table,0u);EXPECT_EQ(retired.status,a::Status::Cancelled);
        std::memcpy(c.args.native+c.args.plan.offsets.seal_receipt,&retired,sizeof(retired));*n::at<n::OwnerHeaderKind>(c.args.native,c.args.plan.offsets.seal_receipt+sizeof(retired))=n::progress(owner.args).header_kind;
        ASSERT_TRUE(source::importRemoteRetired(c.args,0));ASSERT_TRUE(source::finishReleased(c.args));EXPECT_EQ(c.pool.view().frame->phase,p::LookupPhase::Idle);
    }
}
}

namespace llaminar2::test {
/** @brief C0 has no count receive; stale positive/poison words cannot become its payload or contaminate the next owner. */
TEST(PooledImmutableNative, TerminalC0ReleaseSerialCountPoisonAcrossPositiveAndEmptyReplay)
{
    namespace source=q::pooled_source;
    for(const auto& capacities:std::array<std::vector<uint32_t>,2>{{{2,0,2,0},{0,2,0,2}}}){
        DistributedSourceReplay c(capacities);std::array<int32_t,sampling_math::kDeviceGenerationControlCount> control{};c.args.control=control.data();
        const std::array<std::vector<uint64_t>,4> requests{{{701,702,703,704,705,706},{701},{},{0x100000003ULL,701}}};
        for(size_t ordinal=0;ordinal<requests.size();++ordinal){
            ASSERT_NO_THROW(publicationFilled(c,requests[ordinal]));
            ASSERT_NO_THROW(publicationAcquire(c));
            ASSERT_TRUE(source::beforeReady(c.args));
            ASSERT_NO_THROW(publicationReady(c));
            const auto ready=c.pool.headers<p::ReadyHeader>();ASSERT_TRUE(source::acquireReadyConsensus(c.args,ready.data(),ready.size()));
            c.pool.args.input.frontier=ready.data();c.pool.args.input.source=&source::frontier(c.args);
            ASSERT_NO_THROW(c.pool.run<n::CoordinatorAction::Publish>());c.pool.args.input={};
            ASSERT_NO_THROW(c.pool.all<n::OwnerAction::Release>());
            source::stage(c.args)=source::ResponseStage::ReadersComplete; // Recorded actual output/drain boundary; this is common-parser evidence.
            ASSERT_TRUE(source::planReleases(c.args));uint64_t aggregate=0;
            for(uint32_t owner=0;owner<capacities.size();++owner){auto& original=c.pool.owners[owner]->args;const auto bytes=n::progress(original).release_bytes;
                if(capacities[owner]){
                    if(bytes)std::memcpy(c.args.frame+c.args.plan.offsets.fills,original.frame+original.plan.frame.release_receipts,bytes);
                    *n::at<uint64_t>(c.args.native,c.args.plan.offsets.record_bytes)=bytes; // Actual decoded counted receive, including a positive-capacity live0 packet.
                }else{
                    EXPECT_EQ(bytes,0u);
                    *n::at<uint64_t>(c.args.native,c.args.plan.offsets.record_bytes)=~uint64_t{0}; // No C0 endpoint is issued; old serial storage remains inactive.
                }
                ASSERT_TRUE(source::importRemoteReleases(c.args,owner))<<"owner="<<owner<<" request="<<ordinal;
                aggregate+=bytes;EXPECT_EQ(source::releasedBytes(c.args),aggregate);EXPECT_EQ(source::progress(c.args).packets,owner+1);
                EXPECT_EQ(n::progress(original).release_bytes,bytes);
            }
            ASSERT_TRUE(source::reconcileReleases(c.args));ASSERT_TRUE(source::snapshotReleasedObservation(c.args));
            ASSERT_NO_THROW(c.pool.all<n::OwnerAction::Retire>());
            for(uint32_t owner=0;owner<capacities.size();++owner){const auto retired=c.pool.owners[owner]->header<p::RetiredHeader>();
                std::memcpy(c.args.native+c.args.plan.offsets.seal_receipt,&retired,sizeof(retired));*n::at<n::OwnerHeaderKind>(c.args.native,c.args.plan.offsets.seal_receipt+sizeof(retired))=n::progress(c.pool.owners[owner]->args).header_kind;
                ASSERT_TRUE(source::importRemoteRetired(c.args,owner));
            }
            ASSERT_TRUE(source::finishReleased(c.args));ASSERT_TRUE(source::observeReleased(c.args));
            EXPECT_EQ(c.pool.view().frame->phase,p::LookupPhase::Idle);q::QwenAdaptiveCounterTotals totals;ASSERT_TRUE(q::readQwenAdaptiveCounters(control.data(),control.size(),totals));EXPECT_EQ(totals.calls,ordinal+1);EXPECT_EQ(totals.failed_frames,0u);
        }
    }
}
/** @brief A geometrically absent payload is not an owner completion: missing C0 Retired60 still blocks Finish. */
TEST(PooledImmutableNative, TerminalC0ReleaseStillRequiresActualNominalRetiredReceipt)
{
    namespace source=q::pooled_source;DistributedSourceReplay c(std::vector<uint32_t>{2,0});
    ASSERT_NO_THROW(publicationFilled(c,std::array<uint64_t,1>{701}));
    ASSERT_NO_THROW(publicationAcquire(c));
    ASSERT_NO_THROW(publicationReady(c));
    const auto ready=c.pool.headers<p::ReadyHeader>();ASSERT_TRUE(source::acquireReadyConsensus(c.args,ready.data(),ready.size()));
    c.pool.args.input.frontier=ready.data();c.pool.args.input.source=&source::frontier(c.args);ASSERT_NO_THROW(c.pool.run<n::CoordinatorAction::Publish>());c.pool.args.input={};
    ASSERT_NO_THROW(c.pool.all<n::OwnerAction::Release>());source::stage(c.args)=source::ResponseStage::ReadersComplete;
    ASSERT_TRUE(source::planReleases(c.args));
    for(uint32_t owner=0;owner<2;++owner){auto& original=c.pool.owners[owner]->args;const auto bytes=n::progress(original).release_bytes;if(bytes)std::memcpy(c.args.frame+c.args.plan.offsets.fills,original.frame+original.plan.frame.release_receipts,bytes);*n::at<uint64_t>(c.args.native,c.args.plan.offsets.record_bytes)=bytes;ASSERT_TRUE(source::importRemoteReleases(c.args,owner));}
    ASSERT_TRUE(source::reconcileReleases(c.args));ASSERT_NO_THROW(c.pool.all<n::OwnerAction::Retire>());
    const auto retired=c.pool.owners[0]->header<p::RetiredHeader>();std::memcpy(c.args.native+c.args.plan.offsets.seal_receipt,&retired,sizeof(retired));*n::at<n::OwnerHeaderKind>(c.args.native,c.args.plan.offsets.seal_receipt+sizeof(retired))=n::OwnerHeaderKind::Retired;
    ASSERT_TRUE(source::importRemoteRetired(c.args,0));EXPECT_FALSE(source::finishReleased(c.args));EXPECT_NE(c.pool.view().frame->phase,p::LookupPhase::Idle);EXPECT_EQ(source::progress(c.args).seals,1u);
}
}

namespace llaminar2::test {
/** @brief Reject P and UINT32_MAX before any original geometry indexing, also with a real local coordinate0. */
TEST(PooledImmutableNative, TerminalC0ReleaseRejectsForeignOwnerBeforeGeometryAccess)
{
    namespace source=q::pooled_source;
    for(uint32_t owner:{2u,UINT32_MAX}){
        ASSERT_EXIT({
            DistributedSourceReplay c(std::vector<uint32_t>{2,0},0);
            auto* extent=n::at<uint64_t>(c.pool.args.native,c.pool.args.plan.native.ingress_count);*extent=0x123456789abcdef0ULL;
            const auto before=*extent;
            const bool accepted=source::importRemoteReleases(c.args,owner);
            std::_Exit(!accepted && *extent==before?0:87);
        },::testing::ExitedWithCode(0),"");
    }
}
/** @test Retained identity validation remains mandatory in empty Free/Victim waves across reset. */
TEST(PooledImmutableNative, RetainedIdentityEmptyAllocationWavesReplay)
{
    for(const unsigned live:{0u,1u,63u,64u,65u,127u,128u,129u,255u,256u,257u,513u}){
        Pool pool(std::array{live},std::max(1u,live),a::Policy{65536,1,2},std::nullopt,std::max(1u,live));
        std::vector<std::uint64_t> rows(live);
        for(unsigned i=0;i<live;++i)rows[i]=100+i;
        ASSERT_NO_THROW(pool.complete(rows));
        for(unsigned replay=0;replay<3;++replay){
            ASSERT_NO_THROW(pool.begin(rows,replay+2));
            auto& x=pool.owners.front()->args;
            EXPECT_EQ(n::ownerView<a::Slot64>(x).frame->initial_pin_count,live);
            ASSERT_NO_THROW(pool.wave<false>());
            EXPECT_EQ(n::allocationFrame(x).count,0u);
            ASSERT_NO_THROW(pool.wave<true>());
            EXPECT_EQ(n::allocationFrame(x).count,0u);
            ASSERT_NO_THROW(pool.finish());
            EXPECT_EQ(n::ownerView<a::Slot64>(x).frame->held_pins,0u);
            ASSERT_NO_THROW(pool.complete(rows.empty()?std::span<const std::uint64_t>{}:std::span<const std::uint64_t>{rows.data(),1},replay+2));
            ASSERT_NO_THROW(pool.complete({},replay+2));
        }
    }
}
/** @test An empty incoming plan still rejects duplicate/foreign retained pins before plan mutation. */
TEST(PooledImmutableNative, EmptyAllocationRejectsCorruptRetainedIdentityBeforeCommit)
{
    for(bool victim:{false,true})for(unsigned defect=0;defect<4;++defect){
        Pool pool(std::array{128u},128,a::Policy{65536,1,2},std::nullopt,128);
        std::vector<std::uint64_t> rows(128);for(unsigned i=0;i<128;++i)rows[i]=100+i;
        ASSERT_NO_THROW(pool.complete(rows));
        ASSERT_NO_THROW(pool.begin(rows,2));
        if(victim)ASSERT_NO_THROW(pool.wave<false>());
        auto& x=pool.owners.front()->args;
        auto before=n::allocationFrame(x);
        auto slotBytes=pool.owners.front()->cache;
        auto corrupt=[&](size_t){
            before=n::allocationFrame(x);
            auto* pins=n::ownerView<a::Slot64>(x).initial_pins;
            if(defect==0)pins[127].row=pins[0].row;
            if(defect==1)pins[127].unique=pins[0].unique;
            if(defect==2)++pins[127].owner;
            if(defect==3)++pins[127].interval.sequence;
        };
        if(victim)EXPECT_THROW(pool.receivedVictims(corrupt),std::logic_error);
        else EXPECT_THROW(pool.receivedFree({},corrupt),std::logic_error);
        EXPECT_EQ(n::allocationFrame(x).count,before.count);
        EXPECT_EQ(n::allocationFrame(x).phase,before.phase);
        EXPECT_EQ(n::ownerView<a::Slot64>(x).frame->held_pins,128u);
        EXPECT_EQ(n::ownerView<a::Slot64>(x).frame->initial_pin_count,128u);
        EXPECT_EQ(std::memcmp(pool.owners.front()->cache.data(),slotBytes.data(),slotBytes.blocks.size()*64),0);
    }
}

/** @test Recorded scalar inputs retain actual native64-byte alignment and exact logical extents. */
TEST(PooledImmutableNative, RecordedSnapshotAlignmentAndActualOwnerReplay)
{
    static_assert(PooledImmutableNativeReplayBytes::Alignment>=alignof(p::OwnerFrame));
    static_assert(PooledImmutableNativeReplayBytes::Alignment>=alignof(p::PhysicalOwnerControl));
    static_assert(PooledImmutableNativeReplayBytes::Alignment>=alignof(n::OwnerProgress));
    for(const std::size_t count:{0u,1u,63u,64u,65u,129u,1027u}){
        PooledImmutableNativeReplayBytes bytes(count);
        EXPECT_EQ(bytes.size(),count);
        if(count){
            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(bytes.data())%64,0u);
            for(std::size_t i=0;i<count;++i)bytes.data()[i]=std::byte(i%251);
        }else EXPECT_EQ(bytes.data(),nullptr);
        auto copy=bytes;
        EXPECT_EQ(copy.size(),count);
        if(count){EXPECT_EQ(reinterpret_cast<std::uintptr_t>(copy.data())%64,0u);EXPECT_EQ(std::memcmp(copy.data(),bytes.data(),count),0);}
        PooledImmutableNativeReplayBytes assigned;assigned=copy;
        auto moved=std::move(assigned);
        EXPECT_EQ(assigned.size(),0u);EXPECT_EQ(assigned.data(),nullptr);EXPECT_EQ(moved.size(),count);
        if(count){EXPECT_EQ(reinterpret_cast<std::uintptr_t>(moved.data())%64,0u);EXPECT_EQ(std::memcmp(moved.data(),bytes.data(),count),0);}
    }
    Pool pool(std::array{2u});const std::array<std::uint64_t,2> rows{101,102};
    ASSERT_NO_THROW(pool.complete(rows));
    ASSERT_NO_THROW(pool.begin(rows,2));
    const auto& original=pool.owners.front()->args;
    PooledImmutableNativeReplayBytes cache(original.plan.cache.bytes),frame(original.plan.frame.bytes),native(original.plan.native.bytes);
    std::memcpy(cache.data(),original.cache,cache.size());std::memcpy(frame.data(),original.frame,frame.size());std::memcpy(native.data(),original.native,native.size());
    n::OwnerArgs recorded{original.plan,cache.data(),frame.data(),native.data(),original.input};
    ASSERT_TRUE((n::owner<n::OwnerAction::Availability,a::Slot64>(recorded)));
    const auto lease=n::allocationFrame(recorded).lease;
    n::FreePlanHeader header{lease,0,a::Status::Success,n::Version};
    recorded.input.frontier=&header;recorded.input.records=nullptr;recorded.input.record_bytes=nullptr;
    ASSERT_TRUE((n::owner<n::OwnerAction::AcceptFree,a::Slot64>(recorded)));
    EXPECT_EQ(n::allocationFrame(recorded).phase,p::AllocationPhase::Accepted);
    EXPECT_EQ(n::allocationFrame(recorded).count,0u);
    EXPECT_EQ(n::ownerView<a::Slot64>(recorded).frame->held_pins,2u);
    EXPECT_EQ(n::ownerView<a::Slot64>(original).frame->phase,p::OwnerPhase::Reserving);
}

/** @test GPU-used per-record fields match the independent scalar oracle before any physical mutation. */
TEST(PooledImmutableNative, CooperativePinRecordFieldsMatchIndependentScalarReplay)
{
    constexpr unsigned count=257;
    Pool pool(std::array{count},count,a::Policy{65536,1,2},std::nullopt,count);
    std::vector<std::uint64_t> rows(count);
    for(unsigned i=0;i<count;++i)rows[i]=100+i;
    ASSERT_NO_THROW(pool.complete(rows));
    auto& x=pool.owners.front()->args;
    auto coordinator=p::CoordinatorOperations(pool.view());
    ASSERT_TRUE(coordinator.begin(2,rows.data(),rows.size()));
    n::invocation(pool.args);
    x.input={pool.invocation(),nullptr,nullptr,nullptr,nullptr,nullptr};
    ASSERT_TRUE((n::owner<n::OwnerAction::Begin,a::Slot64>(x)));
    std::vector<p::OwnerCommand> original(count);
    for(unsigned i=0;i<count;++i)ASSERT_TRUE(coordinator.pinCommand(i,original[i]));
    const auto view=n::ownerView<a::Slot64>(x);
    for(const auto position:{0u,count-1})for(unsigned defect=0;defect<18;++defect){
        SCOPED_TRACE(::testing::Message()<<"position="<<position<<" defect="<<defect);
        auto commands=original;
        auto& command=commands[position];
        auto& grant=command.target;
        auto& slot=view.slots[grant.slot];
        const auto originalSlot=slot;
        const auto prior=position?commands[position-1].ordinal.value:view.frame->last_command;
        switch(defect){
        case 1:command.kind=p::CommandKind::ReserveFree;break;
        case 2:command.reserved=1;break;
        case 3:++grant.interval.table;break;
        case 4:grant.kind=p::GrantKind::Reservation;break;
        case 5:grant.destination_token=1;break;
        case 6:grant.unique=x.input.invocation->unique_rows;break;
        case 7:grant.row=x.plan.owner.sourceRows();break;
        case 8:++grant.owner;break;
        case 9:grant.slot=x.plan.owner.capacity();break;
        case 10:grant.generation=0;break;
        case 11:++command.requested_row;break;
        case 12:command.ordinal.value=prior;break;
        case 13:++slot.row;break;
        case 14:++slot.generation;break;
        case 15:slot.flags=0;break;
        case 16:a::PhysicalSlot<a::Slot64>::pin(slot);break;
        case 17:slot.flags|=0x80000000u;break;
        default:break;
        }
        const auto expectedSlot=slot;
        view.frame->status=a::Status::Success;
        bool recordsValid=true;
        for(unsigned i=0;i<count;++i){
            const auto previous=i?commands[i-1].ordinal.value:view.frame->last_command;
            recordsValid=n::record_validation::pin<a::Slot64>(x,commands[i],previous) && recordsValid;
        }
        EXPECT_EQ(view.frame->status,a::Status::Success);
        const bool scalarValid=n::validatePinFields<a::Slot64>(x,commands.data(),commands.size());
        EXPECT_EQ(recordsValid,scalarValid);
        EXPECT_EQ(recordsValid,defect==0 || defect==17);
        EXPECT_EQ(std::memcmp(&slot,&expectedSlot,sizeof(slot)),0);
        EXPECT_EQ(view.frame->held_pins,0u);
        EXPECT_EQ(view.frame->initial_pin_count,0u);
        if(defect && defect!=17){
            view.frame->status=a::Status::SourceFailure;
            EXPECT_FALSE((n::validatePinFields<a::Slot64>(x,commands.data(),commands.size())));
            EXPECT_EQ(view.frame->status,a::Status::SourceFailure);
        }
        slot=originalSlot;
    }
    auto invalid=original.front();invalid.target.slot=x.plan.owner.capacity();
    auto inaccessible=x;inaccessible.cache=nullptr;
    EXPECT_FALSE((n::record_validation::pin<a::Slot32>(inaccessible,invalid,0)));
    EXPECT_FALSE((n::record_validation::pin<a::Slot64>(inaccessible,invalid,0)));
}
/** @test Retained-field projection checks remain mandatory with zero incoming allocation decisions. */
TEST(PooledImmutableNative, CooperativeRetainedRecordFieldsMatchZeroIncomingScalarReplay)
{
    for(const bool victim:{false,true})for(const unsigned position:{0u,64u})for(unsigned defect=0;defect<4;++defect){
        Pool pool(std::array{65u},65,a::Policy{65536,1,2},std::nullopt,65);
        std::vector<std::uint64_t> rows(65);
        for(unsigned i=0;i<65;++i)rows[i]=100+i;
        ASSERT_NO_THROW(pool.complete(rows));
        ASSERT_NO_THROW(pool.begin(rows,2));
        if(victim)ASSERT_NO_THROW(pool.wave<false>());
        auto& x=pool.owners.front()->args;
        auto slots=pool.owners.front()->cache;
        const auto corrupt=[&](std::size_t){
            const auto view=n::ownerView<a::Slot64>(x);
            auto& pin=view.initial_pins[position];
            if(defect==0)++pin.interval.sequence;
            if(defect==1)++pin.owner;
            if(defect==2)pin.kind=p::GrantKind::Reservation;
            if(defect==3)pin.kind=p::GrantKind::Absent;
            EXPECT_EQ(n::allocationFrame(x).count,0u);
            for(unsigned key=0;key<2;++key){
                bool valid=true;
                for(unsigned i=0;i<view.frame->initial_pin_count;++i)
                    valid=n::record_validation::retained(view.initial_pins[i],n::allocationFrame(x).lease.interval,
                        p::GrantKind::Read,x.plan.owner.coordinate()) && valid;
                EXPECT_FALSE(valid);
            }
        };
        if(victim)EXPECT_THROW(pool.receivedVictims(corrupt),std::logic_error);
        else EXPECT_THROW(pool.receivedFree({},corrupt),std::logic_error);
        EXPECT_EQ(n::allocationFrame(x).count,0u);
        EXPECT_EQ(n::ownerView<a::Slot64>(x).frame->held_pins,65u);
        EXPECT_EQ(std::memcmp(pool.owners.front()->cache.data(),slots.data(),slots.blocks.size()*64),0);
    }
    const p::Interval interval{701,17,3};
    p::Grant reservation{};reservation.interval=interval;reservation.owner=2;reservation.kind=p::GrantKind::Reservation;
    EXPECT_TRUE(n::record_validation::retained(reservation,interval,p::GrantKind::Reservation,2));
    EXPECT_FALSE(n::record_validation::retained(reservation,interval,p::GrantKind::Read,2));
}

}

namespace llaminar2::test {
/** @test Exact original cancellation vetoes empty/positive source success before any request publication. */
TEST(PooledImmutableNative, OriginalParentCancellationSuppressesSourcePublicationAndSuccess)
{
    namespace source=q::pooled_source;
    for(const bool positive:{false,true})for(const bool cancel:{false,true}){
        DistributedSourceReplay c({0u,0u});
        const std::array<std::uint64_t,2> rows{101,102};
        c.plan(positive?std::span<const std::uint64_t>(rows):std::span<const std::uint64_t>{});c.seals();
        c.healthy=0; // Original standalone prefill is valid before response admission.
        auto& io=source::protocol(c.args);
        const auto issued=io.issuedSequence;
        EXPECT_EQ(source::prepareRequestPublication(c.args,cancel?23:22,17),positive&&!cancel);
        if(!cancel){EXPECT_EQ(c.error,0);EXPECT_EQ(io.status,0u);continue;}
        EXPECT_EQ(source::stage(c.args),source::ResponseStage::Rejected);
        EXPECT_EQ(io.status,static_cast<std::uint64_t>(q::ImmutableLookupStatus::Cancelled));
        EXPECT_EQ(io.issuedSequence,issued);
        EXPECT_EQ(c.error,static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
        EXPECT_EQ(*n::at<std::uint64_t>(c.args.native,c.args.plan.offsets.record_bytes),0u);
        EXPECT_EQ(*n::at<std::uint64_t>(c.args.native,c.args.plan.offsets.payload_bytes),0u);
        EXPECT_FALSE(source::beforeReady(c.args));
        EXPECT_EQ(source::readableRow<a::Slot64>(c.args,0),nullptr);
        /** @brief Replay acquired worker retirement, not the cancellation flag, as the original drain witness. */
        struct Retired {
            std::uint64_t generation=17;
            std::uint64_t retiredInvocation()const{return 23;}
            std::uint64_t retiredGeneration()const{return generation;}
            std::uint64_t completion()const{return 0;}
        } observed;
        observed.generation=16;EXPECT_FALSE(source::retireAborted(c.args,observed));
        observed.generation=17;ASSERT_TRUE(source::retireAborted(c.args,observed));
        const auto& frontier=source::frontier(c.args);const auto& coordinator=*c.pool.view().frame;
        EXPECT_EQ(n::sourceGate(n::SourceCondition::AfterSuccess,&frontier,coordinator.interval,
            c.args.plan.owner.table,coordinator.source),n::SourceGate::Skip);
        EXPECT_EQ(c.error,static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
    }
}
/** @test Original I/O/controller faults remain first when cancellation arrives concurrently. */
TEST(PooledImmutableNative, OriginalSourcePublicationPreservesFirstErrorOnCancellation)
{
    namespace source=q::pooled_source;
    for(const auto first:{q::ImmutableLookupStatus::TimedOut,q::ImmutableLookupStatus::InvalidProtocol}){
        DistributedSourceReplay c({0u});c.plan({});c.seals();
        auto& io=source::protocol(c.args);io.status=static_cast<std::uint64_t>(first);
        const auto original=static_cast<int>(sampling_math::DeviceGenerationError::InvalidVerifierCounts);
        c.error=original;
        EXPECT_FALSE(source::prepareRequestPublication(c.args,23,17));
        EXPECT_EQ(io.status,static_cast<std::uint64_t>(first));EXPECT_EQ(c.error,original);
        EXPECT_EQ(source::frontier(c.args).status,a::Status::SourceFailure);
        EXPECT_FALSE(source::forwardValid(c.args));
    }
}
}

namespace llaminar2::test {
/** @test Cancellation cannot replace outstanding owner duties or prior errors with an Idle drain. */
TEST(PooledImmutableNative, CancelledUnlaunchedOwnerRequiresOriginalInitializedDutyFreeIdle)
{
    for (const std::uint32_t capacity : {0u, 4u})
    {
        const std::array capacities{capacity};
        Pool pool(capacities);
        auto& args = pool.owners.front()->args;
        auto& frame = *n::ownerView<a::Slot64>(args).frame;
        auto& allocation = n::allocationFrame(args);
        auto& native = n::progress(args);
        ASSERT_TRUE(n::idleOwnerForCancelledInvocation<a::Slot64>(args));
        // Absence of an original cancelled receive never authorizes a drain.
        EXPECT_FALSE(n::cancelledIdleOwner<a::Slot64>(args, 1));
        const auto originalFrame = frame;
        const auto originalAllocation = allocation;
        const auto originalNative = native;
        for (const auto phase : {p::OwnerPhase::Pinning, p::OwnerPhase::PinsSealed,
                 p::OwnerPhase::Filling, p::OwnerPhase::Draining})
        {
            frame.phase = phase;
            EXPECT_FALSE(n::idleOwnerForCancelledInvocation<a::Slot64>(args));
        }
        frame = originalFrame;
        frame.held_pins = 1;
        EXPECT_FALSE(n::idleOwnerForCancelledInvocation<a::Slot64>(args));
        frame = originalFrame;
        frame.reservations = 1;
        EXPECT_FALSE(n::idleOwnerForCancelledInvocation<a::Slot64>(args));
        frame = originalFrame;
        frame.status = a::Status::Invariant;
        EXPECT_FALSE(n::idleOwnerForCancelledInvocation<a::Slot64>(args));
        frame = originalFrame;
        ++frame.interval.table;
        EXPECT_FALSE(n::idleOwnerForCancelledInvocation<a::Slot64>(args));
        frame = originalFrame;
        allocation.phase = p::AllocationPhase::Leased;
        EXPECT_FALSE(n::idleOwnerForCancelledInvocation<a::Slot64>(args));
        allocation = originalAllocation;
        native.initialized = 1;
        EXPECT_FALSE(n::idleOwnerForCancelledInvocation<a::Slot64>(args));
        native = originalNative;
        ++native.setup_identity;
        EXPECT_FALSE(n::idleOwnerForCancelledInvocation<a::Slot64>(args));
        native = originalNative;
        EXPECT_TRUE(n::idleOwnerForCancelledInvocation<a::Slot64>(args));
        EXPECT_EQ(frame.held_pins, 0u);
        EXPECT_EQ(frame.reservations, 0u);
        EXPECT_EQ(frame.status, originalFrame.status);
    }
}

} // namespace llaminar2::test
