/**
 * @file ImmutableLookupService.cpp
 * @brief Exact native-row I/O processing and PMA-owned retained service lifetime.
 *
 * The worker polls only shared protocol publications; it cannot read device
 * token/controller memory. Acquisition authenticates every extent before IDs
 * are read. Completion publishes native rows before the release word, and an
 * acknowledgement is mandatory before any response reuse. Failed lanes remain
 * failed, retaining precise I/O diagnostics instead of changing storage modes.
 * GPU source abort, CPU cancellation and complete-parent close have disjoint
 * writers. Exact-ticket I/O abort accounting occurs after the actual worker
 * joins so publication racing its retirement cannot disappear from the ledger.
 */
#include "ImmutableLookupService.h"
#include "backends/NativeOrderingEventGroup.h"
#include "../local_execution/graph/GraphCaptureGuard.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "utils/ScopedOpenMPPartitionAffinity.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <thread>
#include <utility>

namespace llaminar2::qwen4exp
{
namespace
{
/** @brief Acquire a GPU release word without inventing a host shadow value. */
std::uint64_t acquire(std::uint64_t &word)
{
    return std::atomic_ref<std::uint64_t>(word).load(std::memory_order_acquire);
}
/** @brief Release CPU payload/metadata after all native-row writes are complete. */
void release(std::uint64_t &word,std::uint64_t value)
{
    std::atomic_ref<std::uint64_t>(word).store(value,std::memory_order_release);
}
/** @brief Allocate a process-unique immutable table/service binding identity. */
std::uint64_t nextTableId()
{
    static std::atomic<std::uint64_t> next{1};
    auto id=next.load(std::memory_order_relaxed);
    while(id!=0&&id!=std::numeric_limits<std::uint64_t>::max())
    {
        if(next.compare_exchange_weak(id,id+1,std::memory_order_relaxed))return id;
    }
    throw std::overflow_error("Immutable lookup table-instance identity exhausted");
}
} // namespace

ImmutableLookupHostProcessor::ImmutableLookupHostProcessor(HostImmutableRowProvider &table,ImmutableLookupLayout layout,
    std::span<std::byte> channel,std::uint64_t tableId,std::atomic<ImmutableLookupStatus> *fatalStatus,
    const std::atomic<bool> *cancelled,DeviceId requester)
    :table_(&table),layout_(layout),channel_(channel),tableId_(tableId),fatalStatus_(fatalStatus),cancelled_(cancelled)
{
    if(tableId==0||channel.size()<layout.bytes()||reinterpret_cast<std::uintptr_t>(channel.data())%64!=0||
        table.source().rowBytes()!=layout.rowBytes() ||
        layout.subordinate()!=!table.subordinateRequesters().empty() ||
        (table.maximumLiveRows() && layout.maximumIds()>*table.maximumLiveRows()))
        throw std::invalid_argument("Immutable lookup requires aligned complete channel storage and exact table-row geometry");
    // Mode/source/bounds checks precede metadata initialization and binding.
    if(layout.subordinate())table.bindSubordinate(this,requester);
    std::construct_at(reinterpret_cast<ImmutableLookupRequest *>(channel.data()+layout.requestOffset()));
    std::construct_at(reinterpret_cast<ImmutableLookupCompletion *>(channel.data()+layout.completionOffset()));
    std::construct_at(reinterpret_cast<ImmutableLookupAcknowledgement *>(channel.data()+layout.acknowledgementOffset()));
}

ImmutableLookupHostProcessor::~ImmutableLookupHostProcessor()
{
    subordinateGrant_.reset();
    if(layout_.subordinate())table_->retireSubordinate(this);
}
ImmutableLookupHostProcessor::SubordinateReadBorrow ImmutableLookupHostProcessor::acquireSubordinateRead()
{
    if(cancelled_ && cancelled_->load(std::memory_order_acquire))
    {
        lastStatus_=ImmutableLookupStatus::Cancelled;subordinateGrant_.reset();
        throw std::logic_error("Subordinate source ingress was cancelled");
    }
    if(!layout_.subordinate() || !subordinateGrant_ || lastStatus_!=ImmutableLookupStatus::Success ||
        observeAcknowledgement()==lastSequence_)
        throw std::logic_error("Subordinate source borrow requires an actual successful unconsumed request");
    const auto &grant=*subordinateGrant_;
    const auto *completion=reinterpret_cast<const ImmutableLookupCompletion*>(channel_.data()+layout_.completionOffset());
    if(grant.channel_!=this || grant.sequence_!=lastSequence_ || !grant.generation_ ||
        ImmutableLookupProtocol::validateCompletion(*completion,lastSequence_,grant.sequence_,grant.generation_,
            tableId_,grant.row_count_,layout_.rowBytes(),layout_.protocol())!=ImmutableLookupStatus::Success)
        throw std::logic_error("Subordinate source read grant differs from its original completed interval");
    return SubordinateReadBorrow(this,grant.generation_,grant.sequence_);
}
std::optional<CpuAdaptiveImmutableRowCache::Pin> ImmutableLookupHostProcessor::finishSubordinateRead(
    const SubordinateReadBorrow &borrow,std::size_t responseRow)
{
    if(!subordinateGrant_ || borrow.owner_!=this || borrow.sequence_!=lastSequence_ ||
        borrow.sequence_!=subordinateGrant_->sequence_ || borrow.generation_!=subordinateGrant_->generation_ ||
        subordinateGrant_->channel_!=this || lastStatus_!=ImmutableLookupStatus::Success)
        throw std::logic_error("Subordinate source borrow names a stale acquired grant");
    // This lookup reads only the retained CPU-owned GrantedRow array. GPU ACK
    // and request-frame refill cannot redirect it to an unauthorised source row.
    auto pin=table_->pinSubordinate(*subordinateGrant_,responseRow);
    if((cancelled_ && cancelled_->load(std::memory_order_acquire)) || observeAcknowledgement()==lastSequence_)
    {
        pin.reset();
        throw std::logic_error("Subordinate source response retired during new borrow acquisition");
    }
    // Retirement immediately after this point is legal: the already acquired
    // old reader has an independent generation pin and retains its exact bytes.
    return pin;
}
std::optional<CpuAdaptiveImmutableRowCache::Pin> ImmutableLookupHostProcessor::borrowSubordinateResponseRow(
    std::size_t responseRow)
{
    const auto borrow=acquireSubordinateRead();
    return finishSubordinateRead(borrow,responseRow);
}

std::uint64_t ImmutableLookupHostProcessor::observeAcknowledgement()
{
    auto *ack=reinterpret_cast<ImmutableLookupAcknowledgement *>(channel_.data()+layout_.acknowledgementOffset());
    const auto consumed=acquire(ack->consumedSequence);
    if(consumed<lastAcknowledgement_||consumed>lastSequence_)
    {
        lastStatus_=ImmutableLookupStatus::InvalidAcknowledgement;
        lastFailure_="Immutable lookup acknowledgement is stale or names an unserviced sequence";
        throw std::runtime_error(lastFailure_);
    }
    stats_.acknowledgementBytes+=(consumed-lastAcknowledgement_)*sizeof(std::uint64_t);
    lastAcknowledgement_=consumed;
    if(subordinateGrant_ && consumed==lastSequence_)subordinateGrant_.reset();
    return consumed;
}

void ImmutableLookupHostProcessor::complete(const ImmutableLookupRequest &request,std::uint64_t sequence,
    ImmutableLookupStatus status)
{
    ImmutableLookupCompletion body;
    body.publication=sequence;body.sequenceComplement=~sequence;
    body.generation=request.generation;body.generationComplement=~body.generation;
    body.table=tableId_;body.tableComplement=~tableId_;
    body.idCount=request.idCount;body.idCountComplement=~body.idCount;
    body.status=static_cast<std::uint64_t>(status);body.statusComplement=~body.status;
    body.nativeBytes=status==ImmutableLookupStatus::Success?request.idCount*layout_.rowBytes():0;
    body.nativeBytesComplement=~body.nativeBytes;
    body.protocol=layout_.protocol();body.protocolComplement=~body.protocol;
    auto *destination=channel_.data()+layout_.completionOffset();
    // Physical metadata is cache-line aligned, but padding is never copied.
    std::memcpy(destination+sizeof(std::uint64_t),reinterpret_cast<const std::byte *>(&body)+sizeof(std::uint64_t),
                kLookupCompletionLiveBytes-sizeof(std::uint64_t));
    if(status!=ImmutableLookupStatus::Success)
    {
        if(lastStatus_==ImmutableLookupStatus::Success)lastStatus_=status;
        // Generation admission may run after GPU completion. Publish CPU I/O
        // failure first, so reinitializing controller words cannot mask it.
        if(fatalStatus_)fatalStatus_->store(lastStatus_,std::memory_order_release);
    }
    release(reinterpret_cast<ImmutableLookupCompletion *>(destination)->publication,sequence);
    lastSequence_=sequence;
    ++stats_.requests;stats_.metadataBytes+=kLookupRequestLiveBytes+kLookupCompletionLiveBytes;
    if(status==ImmutableLookupStatus::Success)
    {
        stats_.ids+=request.idCount;stats_.idBytes+=request.idBytes;stats_.nativeBytes+=body.nativeBytes;
    }
    else {++stats_.failures;if(lastStatus_==ImmutableLookupStatus::Success)lastStatus_=status;}
    if(layout_.subordinate())
    {
        if(status==ImmutableLookupStatus::Success)
            stats_.metadataBytes+=request.idCount*sizeof(ImmutableRowInvalidationAcknowledgement);
    }
}

ImmutableLookupPoll ImmutableLookupHostProcessor::poll(std::uint64_t generation,std::uint64_t invocation)
{
    if(generation==0)throw std::invalid_argument("Immutable lookup admission generation must be positive");
    if(layout_.subordinate() && ((cancelled_ && cancelled_->load(std::memory_order_acquire)) || sourceAbortRequested(generation,invocation)))
    {
        lastStatus_=ImmutableLookupStatus::Cancelled;
        lastFailure_="Subordinate source ingress was cancelled";
        subordinateGrant_.reset();
        throw std::runtime_error(lastFailure_);
    }
    if(layout_.subordinate() && lastStatus_!=ImmutableLookupStatus::Success)
        throw std::runtime_error("Subordinate source ingress is failed");
    if(subordinateGrant_ && subordinateGrant_->generation_!=generation && observeAcknowledgement()!=lastSequence_)
    {
        lastStatus_=ImmutableLookupStatus::StaleGeneration;
        lastFailure_="Subordinate source interval changed before the original grant was consumed";
        throw std::runtime_error(lastFailure_);
    }
    auto *shared=reinterpret_cast<ImmutableLookupRequest *>(channel_.data()+layout_.requestOffset());
    const auto sequence=acquire(shared->publication);
    if(sequence==0||sequence==lastSequence_)return ImmutableLookupPoll::Pending;
    if(sequence<lastSequence_)
    {
        lastStatus_=ImmutableLookupStatus::InvalidSequence;
        lastFailure_="Immutable lookup GPU request sequence regressed while the channel was retained";
        throw std::runtime_error(lastFailure_);
    }
    if(observeAcknowledgement()!=lastSequence_)
    {
        lastStatus_=ImmutableLookupStatus::InvalidAcknowledgement;
        lastFailure_="Immutable lookup attempted response-slot reuse before GPU acknowledgement";
        throw std::runtime_error(lastFailure_);
    }
    ImmutableLookupRequest request;
    request.publication=sequence;
    std::memcpy(reinterpret_cast<std::byte *>(&request)+sizeof(std::uint64_t),
                reinterpret_cast<const std::byte *>(shared)+sizeof(std::uint64_t),
                kLookupRequestLiveBytes-sizeof(std::uint64_t));
    auto status=ImmutableLookupProtocol::validateRequest(request,sequence,generation,tableId_,lastSequence_,
                                                        layout_.maximumIds(),layout_.rowBytes(),layout_.protocol());
    if(status==ImmutableLookupStatus::Success && layout_.subordinate())
    {
        constexpr auto max=std::numeric_limits<std::uint64_t>::max();
        constexpr auto headers=kLookupRequestLiveBytes+kLookupCompletionLiveBytes;
        constexpr auto perRow=sizeof(ImmutableRowSubordinateIntent)+sizeof(ImmutableRowInvalidationAcknowledgement);
        if(request.idCount>(max-headers)/perRow || stats_.metadataBytes>max-headers-request.idCount*perRow)
            throw std::overflow_error("Subordinate source live metadata observation exhausted");
    }
    if(status==ImmutableLookupStatus::Success)
    {
        try
        {
            if(layout_.subordinate())stats_.metadataBytes+=request.idCount*sizeof(ImmutableRowSubordinateIntent);
            const auto *ids=reinterpret_cast<const std::int64_t *>(channel_.data()+layout_.idsOffset());
            const auto count=static_cast<std::size_t>(request.idCount);
            const std::span<std::byte> response{channel_.data()+layout_.payloadOffset(),count*layout_.rowBytes()};
            HostImmutableRowGatherObservation observation;
            if(layout_.subordinate())
            {
                const auto *intents=reinterpret_cast<const ImmutableRowSubordinateIntent *>(channel_.data()+layout_.intentsOffset());
                auto *invalidations=reinterpret_cast<ImmutableRowInvalidationAcknowledgement *>(channel_.data()+layout_.invalidationsOffset());
                auto grant=table_->authorizeSubordinate(this,request.generation,sequence,{ids,count},
                    {intents,count},{invalidations,count});
                observation=table_->gatherSubordinate(grant,response);
                grant.complete(); // Completed capabilities retain no mutable mapped ID/intent/ACK spans.
                subordinateGrant_=std::move(grant);
            }
            else observation=table_->gather({ids,count},response);
            // This processor alone owns its counters. The returned observation
            // excludes concurrent clients of the same table and is issued only
            // after all requested native bytes were gathered successfully.
            accumulateHostImmutableRowWork(stats_,table_->mode(),observation,request.idCount,
                static_cast<std::size_t>(request.idCount)*layout_.rowBytes());
        }
        catch(const std::out_of_range &error)
        {
            status=ImmutableLookupStatus::InvalidRowId;lastFailure_=error.what();
        }
        catch(const std::invalid_argument &error)
        {
            status=layout_.subordinate()?ImmutableLookupStatus::InvalidProtocol:ImmutableLookupStatus::StorageFailure;
            lastFailure_=error.what();
        }
        catch(const std::exception &error)
        {
            status=ImmutableLookupStatus::StorageFailure;lastFailure_=error.what();
        }
    }
    else lastFailure_="Immutable lookup rejected request protocol/identity/extent status "+
                      std::to_string(static_cast<std::uint64_t>(status));
    // Cancellation may race this check. A GPU consumer must still validate its
    // live admission/reservation before publication; CPU cannot enforce that edge.
    if(layout_.subordinate() && ((cancelled_ && cancelled_->load(std::memory_order_acquire)) || sourceAbortRequested(generation,invocation)))
    {
        status=ImmutableLookupStatus::Cancelled;
        lastFailure_="Subordinate source read cancelled before response publication";
        subordinateGrant_.reset();
    }
    if(status!=ImmutableLookupStatus::Success)subordinateGrant_.reset();
    complete(request,sequence,status);
    return ImmutableLookupPoll::Completed;
}

bool ImmutableLookupHostProcessor::closed(std::uint64_t generation)
{
    auto *ack=reinterpret_cast<ImmutableLookupAcknowledgement *>(channel_.data()+layout_.acknowledgementOffset());
    const auto closed=acquire(ack->closedGeneration);
    if(closed!=generation)return false;
    auto *request=reinterpret_cast<ImmutableLookupRequest *>(channel_.data()+layout_.requestOffset());
    if(observeAcknowledgement()!=lastSequence_||acquire(request->publication)!=lastSequence_)
    {
        lastStatus_=ImmutableLookupStatus::InvalidAcknowledgement;
        lastFailure_="Immutable lookup terminal close preceded request service or response acknowledgement";
        throw std::runtime_error(lastFailure_);
    }
    if(lastClose_!=generation){stats_.terminalBytes+=sizeof(std::uint64_t);lastClose_=generation;}
    return true;
}

std::size_t ImmutableLookupService::hostBytes(ImmutableLookupLayout layout)
{
    return TransferEngine::mappedHostRegionAllocationBytes(layout.bytes());
}

std::shared_ptr<ImmutableLookupService> ImmutableLookupService::create(std::shared_ptr<HostImmutableRowProvider> table,
    ImmutableLookupLayout layout,std::span<const DeviceId> devices,std::shared_ptr<PhysicalMemoryAuthority> memory,
    DeviceId hostDevice,RetainedHostServiceUseToken worker,TransferEngine &transfers)
{
    return createWorker(std::move(table),layout,devices,std::move(memory),hostDevice,std::move(worker),transfers);
}
std::shared_ptr<ImmutableLookupService> ImmutableLookupService::create(std::shared_ptr<HostImmutableRowProvider> table,
    ImmutableLookupLayout layout,std::span<const DeviceId> devices,std::shared_ptr<PhysicalMemoryAuthority> memory,
    DeviceId hostDevice,RetainedHostServiceBorrow worker,TransferEngine &transfers)
{
    return createWorker(std::move(table),layout,devices,std::move(memory),hostDevice,std::move(worker),transfers);
}
std::shared_ptr<ImmutableLookupService> ImmutableLookupService::createWorker(std::shared_ptr<HostImmutableRowProvider> table,
    ImmutableLookupLayout layout,std::span<const DeviceId> devices,std::shared_ptr<PhysicalMemoryAuthority> memory,
    DeviceId hostDevice,std::variant<RetainedHostServiceUseToken,RetainedHostServiceBorrow> worker,TransferEngine &transfers)
{
    const auto& program=std::visit([](const auto& original)->const RetainedHostServiceProgram&{
        if constexpr(std::is_same_v<std::decay_t<decltype(original)>,RetainedHostServiceUseToken>)return original.program();
        else return original->program();},worker);
    const auto& originalMemory=std::visit([](const auto& original)->const std::shared_ptr<PhysicalMemoryAuthority>&{
        if constexpr(std::is_same_v<std::decay_t<decltype(original)>,RetainedHostServiceUseToken>)return original.memory();
        else return original->memory();},worker);
    if(program.role()!=RetainedHostServiceRole::ImmutableLookupIO ||
       originalMemory.get()!=memory.get() || originalMemory.owner_before(memory) || memory.owner_before(originalMemory))
        throw std::logic_error("Immutable lookup requires its original admitted I/O worker role and PMA control block");
    if(!table||!memory||!hostDevice.is_cpu()||devices.empty()||table->source().rowBytes()!=layout.rowBytes()||
        table->retainedOwnerObservation().allocator()!=PhysicalMemoryAllocatorIdentity{memory->worldRank(),hostDevice})
        throw std::invalid_argument("Immutable lookup service requires table, canonical admission and explicit mapped endpoints");
    if(const auto bound=table->maximumLiveRows();bound && layout.maximumIds()>*bound)
        throw std::invalid_argument("Hosted lookup channel exceeds the provider's admitted original ID envelope");
    const auto source_owner=table->retainedOwnerObservation();
    if(source_owner.authority().get()!=memory.get())
        throw std::invalid_argument("Hosted lookup cannot borrow a provider from a different PMA");
    if(layout.subordinate()) {
        const auto requesters=table->subordinateRequesters();
        if(requesters.size()!=devices.size())throw std::invalid_argument("Subordinate service requires the complete original consumer population");
        for(const auto requester:requesters)
            if(requester.world_rank!=memory->worldRank() || std::find(devices.begin(),devices.end(),requester.device)==devices.end())
                throw std::invalid_argument("Subordinate service has a foreign or missing original consumer");
    } else if(!table->subordinateRequesters().empty())
        throw std::invalid_argument("Subordinate source cannot enter the ordinary channel ABI");
    const auto geometry=ImmutableLookupServiceGeometry::checked(layout,devices);
    const auto bytes=geometry.hostBytes();
    auto physical=memory->claimNewAllocation(hostDevice,PhysicalMemoryOwner::ImmutableLookupStaging,bytes);
    auto mapped=transfers.allocateMappedHostRegion(geometry.bytes(),devices);
    if(!mapped||!mapped->isBound()||mapped->sizeBytes()!=bytes)
        throw std::runtime_error("Immutable lookup mapped allocation differs from its admitted physical extent");
    auto backing=std::make_shared<ImmutableLookupChannelBacking>(ImmutableLookupChannelBacking{
        std::move(physical),std::move(mapped),layout,nextTableId()});
    // Only this independently scheduled native-row I/O owner inherits the
    // admitted OpenMP partition. Restore the creator before returning; generic
    // retained workers and CPU expert compute placement remain unchanged.
    const ScopedOpenMPPartitionAffinity io_worker_affinity;
    auto execution=std::visit([](auto&& original)->RetainedHostServiceBorrow{
        if constexpr(std::is_same_v<std::decay_t<decltype(original)>,RetainedHostServiceUseToken>)
            return RetainedHostServiceBorrow::own(PreparedRetainedHostService::materialize(std::move(original)));
        else return std::move(original);},std::move(worker));
    return std::shared_ptr<ImmutableLookupService>(
        new ImmutableLookupService(std::move(table), std::move(backing), geometry,std::move(memory),
            std::move(execution)));
}

ImmutableLookupService::ImmutableLookupService(std::shared_ptr<HostImmutableRowProvider> table,
                                               std::shared_ptr<ImmutableLookupChannelBacking> backing,ImmutableLookupServiceGeometry geometry,
                                               std::shared_ptr<const PhysicalMemoryAuthority> memory,
                                               RetainedHostServiceBorrow worker)
    : table_(std::move(table)), backing_(std::move(backing)), memory_(std::move(memory)),
      geometry_(std::move(geometry)),processor_(*table_, backing_->layout,
                 {static_cast<std::byte *>(backing_->mapped->mutableHostData()), backing_->layout.bytes()},
                 backing_->tableId, &fatalStatus_, &cancelled_,geometry_.device(0)),
      worker_owner_(std::move(worker)),worker_(worker_owner_->worker())
{
    adopted_.resize(geometry_.channelCount());retired_.resize(geometry_.channelCount());
    pooled_storages_.resize(geometry_.channelCount());
    for(std::size_t i=1;i<geometry_.channelCount();++i)
        extra_processors_.push_back(std::make_unique<ImmutableLookupHostProcessor>(*table_,backing_->layout,
            std::span<std::byte>{static_cast<std::byte*>(backing_->mapped->mutableHostData())+geometry_.offset(i),backing_->layout.bytes()},
            backing_->tableId,&fatalStatus_,&cancelled_,geometry_.device(i)));
}
std::size_t ImmutableLookupService::channelIndex(DeviceId device) const {
    for(std::size_t i=0;i<geometry_.channelCount();++i)if(geometry_.device(i)==device)return i;
    throw std::invalid_argument("Immutable lookup names a foreign original consumer");
}
ImmutableLookupHostProcessor& ImmutableLookupService::processor(std::size_t index) {
    return index?*extra_processors_.at(index-1):processor_;
}
const ImmutableLookupHostProcessor& ImmutableLookupService::processor(std::size_t index) const {
    return index?*extra_processors_.at(index-1):processor_;
}

ImmutableLookupService::~ImmutableLookupService()
{
    if(outstanding_.valid())
        (void)cancel(outstanding_);
    if(outstanding_.valid())
    {
        try{(void)worker_.await({outstanding_.invocation});}catch(...){/* Teardown retains bytes until the worker stops. */}
    }
}

ImmutableLookupService::Binding ImmutableLookupService::bind(DeviceId device)
{
    if(backing_->layout.subordinate())
        throw std::logic_error("Subordinate source requires the future explicit GPU request/ACK kernel binding");
    if(!device.is_gpu()||!backing_->mapped->hasDevice(device))
        throw std::invalid_argument("Immutable lookup binding names an unmapped GPU endpoint");
    auto *base=static_cast<std::byte *>(backing_->mapped->deviceAlias(device))+geometry_.offset(channelIndex(device));
    const auto &layout=backing_->layout;
    // This is the explicit storage infrastructure boundary: TransferEngine owns
    // every mapping/alias and PMA owns its physical lease. Kernels receive only
    // this setup-authenticated immutable view, never arbitrary mapped pointers.
    return Binding(shared_from_this(),device,{
        reinterpret_cast<ImmutableLookupRequest *>(base+layout.requestOffset()),
        reinterpret_cast<ImmutableLookupCompletion *>(base+layout.completionOffset()),
        reinterpret_cast<ImmutableLookupAcknowledgement *>(base+layout.acknowledgementOffset()),
        reinterpret_cast<std::int64_t *>(base+layout.idsOffset()),base+layout.payloadOffset(),
        backing_->tableId,layout.maximumIds(),layout.rowBytes()});
}

ImmutableSubordinateLookupBinding ImmutableLookupService::bindSubordinate(DeviceId device)
{
    if (!backing_->layout.subordinate() || table_->subordinateRequesters().empty() || !device.is_gpu() ||
        !backing_->mapped->hasDevice(device))
        throw std::invalid_argument(
            "Adaptive native binding requires its exact declared protocol2 requester");
    table_->authenticateSource();
    auto *base = static_cast<std::byte *>(backing_->mapped->deviceAlias(device))+geometry_.offset(channelIndex(device));
    const auto &l = backing_->layout;
    return ImmutableSubordinateLookupBinding(
        shared_from_this(), device,
        {{reinterpret_cast<ImmutableLookupRequest *>(base + l.requestOffset()),
          reinterpret_cast<ImmutableLookupCompletion *>(base + l.completionOffset()),
          reinterpret_cast<ImmutableLookupAcknowledgement *>(base + l.acknowledgementOffset()),
          reinterpret_cast<std::int64_t *>(base + l.idsOffset()), base + l.payloadOffset(), backing_->tableId,
          l.maximumIds(), l.rowBytes()},
         reinterpret_cast<ImmutableRowSubordinateIntent *>(base + l.intentsOffset()),
         reinterpret_cast<ImmutableRowInvalidationAcknowledgement *>(base + l.invalidationsOffset())});
}
const ImmutableRowTableSource& ImmutableSubordinateLookupBinding::source()const{
    if(!owner_)throw std::logic_error("Subordinate source binding has lost its original service");
    return owner_->table_->source();
}
void ImmutableSubordinateLookupBinding::requireFor(const PooledImmutableSourceGeometry& geometry)const{
    const auto& p=geometry.parameters();
    if(!owner_ || !owner_->backing_->layout.subordinate() || geometry.allocator().device!=device_ ||
        geometry.allocator().world_rank!=owner_->memory_->worldRank() || !owner_->backing_->mapped->hasDevice(device_))
        throw std::invalid_argument("Pooled source binding has foreign provider/channel/rank/device ownership");
    const auto requesters=owner_->table_->subordinateRequesters();const auto& actual=owner_->table_->source();const auto& layout=owner_->backing_->layout;
    if(std::find(requesters.begin(),requesters.end(),geometry.allocator())==requesters.end() || actual.rows!=p.coordinator.policy.sourceRows() || actual.row_elements!=p.row_elements ||
        ggufToTensorType(actual.format)!=p.format || actual.rowBytes()!=p.owner.row_bytes || layout.protocol()!=kImmutableLookupSubordinateProtocol ||
        layout.maximumIds()!=p.coordinator.policy.maximumUnique() || layout.rowBytes()!=p.owner.row_bytes ||
        owner_->backing_->tableId!=view_.base.table || p.owner.table!=view_.base.table || p.coordinator.table!=view_.base.table ||
        p.owner.source_storage_generation!=storageGeneration() || p.coordinator.source_storage_generation!=storageGeneration())
        throw std::invalid_argument("Pooled source native fields do not match the real protocol2 provider and original source generation");
    auto* base=static_cast<std::byte*>(owner_->backing_->mapped->deviceAlias(device_))+owner_->geometry_.offset(owner_->channelIndex(device_));
    if(view_.base.request!=reinterpret_cast<ImmutableLookupRequest*>(base+layout.requestOffset()) ||
        view_.base.completion!=reinterpret_cast<ImmutableLookupCompletion*>(base+layout.completionOffset()) ||
        view_.base.acknowledgement!=reinterpret_cast<ImmutableLookupAcknowledgement*>(base+layout.acknowledgementOffset()) ||
        view_.base.ids!=reinterpret_cast<std::int64_t*>(base+layout.idsOffset()) || view_.base.nativeRows!=base+layout.payloadOffset() ||
        view_.base.maximumIds!=layout.maximumIds() || view_.base.rowBytes!=layout.rowBytes() ||
        view_.intents!=reinterpret_cast<ImmutableRowSubordinateIntent*>(base+layout.intentsOffset()) ||
        view_.acknowledgements!=reinterpret_cast<ImmutableRowInvalidationAcknowledgement*>(base+layout.invalidationsOffset()))
        throw std::invalid_argument("Pooled source original mapped aliases changed after binding");
}
std::shared_ptr<PooledImmutableSourcePreparedStorage> ImmutableSubordinateLookupBinding::preparePooled(
    const PooledImmutableSourceGeometry& geometry,std::vector<std::shared_ptr<PooledImmutableOwnerPreparedStorage>> owners,
    std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream)const{
    return preparePooledImpl(geometry,std::move(owners),std::move(coordinator),std::move(memory),stream,{});
}
std::shared_ptr<PooledImmutableSourcePreparedStorage> ImmutableSubordinateLookupBinding::preparePooledImpl(
    const PooledImmutableSourceGeometry& geometry,std::vector<std::shared_ptr<PooledImmutableOwnerPreparedStorage>> owners,
    std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,std::shared_ptr<PhysicalMemoryAuthority> memory,
    void* stream,std::unique_ptr<NativeOrderingEventGroup> group)const{
    requireFor(geometry);
    return owner_->declarePooledSourceStorage(*this,geometry,std::move(owners),std::move(coordinator),std::move(memory),stream,std::move(group));
}
std::shared_ptr<PooledImmutableSourcePreparedStorage> ImmutableLookupService::declarePooledSourceStorage(
    const ImmutableSubordinateLookupBinding& binding,const PooledImmutableSourceGeometry& geometry,
    std::vector<std::shared_ptr<PooledImmutableOwnerPreparedStorage>> owners,std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,
    std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream,std::unique_ptr<NativeOrderingEventGroup> group){
    if(isGraphCaptureActive())throw std::logic_error("Pooled source registration must be materialized before capture");
    binding.requireFor(geometry);requireHealthy();
    if(!memory || memory.get()!=memory_.get() || owners.empty() || !coordinator)
        throw std::invalid_argument("Pooled source registration requires its actual original PMA and native authorities");
    table_->authenticateSource();
    // Cold declaration is serialized once. No model state or host policy shadow
    // is consulted; failure never publishes partial roots into this service.
    std::lock_guard lock(pooled_setup_);
    auto& pooled_storage_=pooled_storages_.at(channelIndex(binding.device()));
    if(pooled_storage_){
        if(group)throw std::invalid_argument("Existing pooled source cannot consume a second initialization claim");
        if(pooled_storage_->geometry().parameters().identity!=geometry.parameters().identity || pooled_storage_->originalOwners().size()!=owners.size() || !std::equal(owners.begin(),owners.end(),pooled_storage_->originalOwners().begin()) || pooled_storage_->coordinator()!=coordinator)
            throw std::invalid_argument("Pooled source cannot replace its original native source/accepted-owner/coordinator registration");
        pooled_storage_->captureIdentity();return pooled_storage_;
    }
    auto storage=PooledImmutableSourcePreparedStorage::materialize(geometry,std::move(owners),std::move(coordinator),std::move(memory),stream,std::move(group));
    pooled_storage_=std::move(storage);return pooled_storage_;
}
std::uint64_t ImmutableSubordinateLookupBinding::captureIdentity() const noexcept
{
    std::uint64_t h = 1469598103934665603ULL;
    const auto mix = [&](std::uint64_t n)
    {
        h ^= n;
        h *= 1099511628211ULL;
    };
    mix(2);mix(kImmutableLookupAcknowledgementVersion);
    mix(reinterpret_cast<std::uintptr_t>(owner_.get()));
    mix(static_cast<std::uint64_t>(device_.type));
    mix(static_cast<std::uint64_t>(device_.ordinal));
    mix(reinterpret_cast<std::uintptr_t>(view_.base.request));
    mix(reinterpret_cast<std::uintptr_t>(view_.base.completion));
    mix(reinterpret_cast<std::uintptr_t>(view_.base.acknowledgement));
    mix(reinterpret_cast<std::uintptr_t>(view_.base.ids));
    mix(reinterpret_cast<std::uintptr_t>(view_.base.nativeRows));
    mix(reinterpret_cast<std::uintptr_t>(view_.intents));
    mix(reinterpret_cast<std::uintptr_t>(view_.acknowledgements));
    mix(view_.base.table);
    mix(view_.base.maximumIds);
    mix(view_.base.rowBytes);
    return h ? h : 1;
}

ImmutableLookupService::Ticket ImmutableLookupService::arm(std::uint64_t generation) {
    return armSingleChannel(generation,{&ImmutableLookupService::run,this});
}
ImmutableLookupService::Ticket ImmutableLookupService::armSingleChannel(std::uint64_t generation,WorkerEntry entry) {
    if(geometry_.channelCount()!=1)throw std::logic_error("Multiple source consumers require complete original group admission");
    auto ticket=armImpl(generation,entry);
    std::lock_guard lock(lifecycle_);adopted_[0]=true;return ticket;
}
ImmutableLookupService::GroupAdmission ImmutableLookupService::armGroup(std::uint64_t generation) {
    return GroupAdmission(shared_from_this(),armImpl(generation,{&ImmutableLookupService::run,this}));
}
ImmutableLookupService::Ticket ImmutableLookupService::adoptGroupChannel(const GroupAdmission& group,DeviceId consumer) {
    std::lock_guard lock(lifecycle_);const auto index=channelIndex(consumer);
    if(group.owner_.get()!=this || state_!=Lifecycle::Serving || group.ticket_.generation!=outstanding_.generation ||
       group.ticket_.invocation!=outstanding_.invocation || adopted_[index])
        throw std::logic_error("Source channel requires its unique original complete-group admission");
    adopted_[index]=true;return {outstanding_.generation,outstanding_.invocation,index};
}
ImmutableLookupService::Ticket ImmutableLookupService::armImpl(std::uint64_t generation,WorkerEntry entry)
{
    if(!entry.task || !entry.context)throw std::invalid_argument("Immutable lookup worker entry requires its borrowed task and context");
    std::lock_guard lock(lifecycle_);
    if(state_!=Lifecycle::Idle||outstanding_.valid()||generation==0||generation<=generation_)
        throw std::logic_error("Immutable lookup service requires an idle healthy lane and a newer admission generation");
    std::fill(adopted_.begin(),adopted_.end(),false);std::fill(retired_.begin(),retired_.end(),false);joined_=false;joined_result_=false;
    generation_=generation;cancelled_.store(false,std::memory_order_release);state_=Lifecycle::Serving;
    // run() acquires lifecycle_ before reading this immutable ticket. dispatch
    // may wake the retained thread before returning, so publication stays under
    // the same lock until the actual worker invocation has been assigned.
    const auto invocation=worker_.dispatch(entry.task,entry.context);
    outstanding_={generation,invocation.generation};
    return outstanding_;
}

ImmutableLookupService::GroupAdmission::~GroupAdmission() noexcept {
    if(!owner_)return;
    owner_->abandonGroup(*this);
}
ImmutableLookupService::GroupAdmission& ImmutableLookupService::GroupAdmission::operator=(GroupAdmission&& other) noexcept {
    if(this!=&other){
        if(owner_)owner_->abandonGroup(*this);
        owner_=std::move(other.owner_);ticket_=other.ticket_;
    }
    return *this;
}
void ImmutableLookupService::abandonGroup(const GroupAdmission& group) noexcept {
    try{
        {
            std::lock_guard lock(lifecycle_);
            if(group.owner_.get()!=this || state_!=Lifecycle::Serving ||
               group.ticket_.generation!=outstanding_.generation || group.ticket_.invocation!=outstanding_.invocation)return;
        }
        (void)cancel(group.ticket_);(void)finishGroup(group);
    }catch(...){std::terminate();}
}
std::function<void()> ImmutableLookupService::failureCallback(const GroupAdmission& group) const
{
    std::lock_guard lock(lifecycle_);
    if(group.owner_.get()!=this || !group.valid() || state_!=Lifecycle::Serving ||
       group.ticket_.generation!=outstanding_.generation || group.ticket_.invocation!=outstanding_.invocation)
        throw std::logic_error("Source failure callback requires its original active group");
    return [original=group.owner_,ticket=group.ticket_]{(void)original->cancel(ticket);};
}
std::function<void()> ImmutableLookupService::failureCallbackWithNativeAbort(
    const GroupAdmission& group, std::function<void()> signal) const
{
    std::lock_guard lock(lifecycle_);
    if (!signal || group.owner_.get() != this || !group.valid() || state_ != Lifecycle::Serving ||
        group.ticket_.generation != outstanding_.generation || group.ticket_.invocation != outstanding_.invocation)
        throw std::logic_error("Native source abort requires its original active complete-group admission");
    return [original = group.owner_, ticket = group.ticket_, signal = std::move(signal)] {
        std::lock_guard current(original->lifecycle_);
        // One service arm requires a strictly newer generation. A completed CPU
        // close may precede native failure; retain its truthful I/O outcome while
        // still cancelling this SAME original GPU request. A newer arm rejects
        // this callback before touching that communicator's absorbing abort.
        if (!ticket.valid() || ticket.generation != original->generation_ ||
            (original->state_ == Lifecycle::Serving &&
                ticket.invocation != original->outstanding_.invocation))
            return;
        (void)original->cancelLocked(ticket);
        signal();
    };
}
bool ImmutableLookupService::finishGroup(const GroupAdmission& group) {
    bool cancelMissing=false;
    {
        std::lock_guard lock(lifecycle_);
        if(group.owner_.get()!=this || group.ticket_.generation!=generation_)
            throw std::logic_error("Source terminal group lost its original admission");
        if(state_==Lifecycle::Serving && !joined_){
            if(group.ticket_.invocation!=outstanding_.invocation)
                throw std::logic_error("Source terminal group lost its original worker invocation");
            for(std::size_t i=0;i<geometry_.channelCount();++i){
                auto* ack=reinterpret_cast<ImmutableLookupAcknowledgement*>(static_cast<std::byte*>(backing_->mapped->mutableHostData())+
                    geometry_.offset(i)+backing_->layout.acknowledgementOffset());
                cancelMissing=acquire(ack->closedGeneration)!=group.ticket_.generation || cancelMissing;
            }
        }
    }
    // A participant may already be in the shared terminal wait. Publish the
    // parent's actual failure before taking the join mutex, so it can drain.
    if(cancelMissing)(void)cancel(group.ticket_);
    std::lock_guard terminalLock(terminal_join_);
    bool join=false;
    {
        std::lock_guard lock(lifecycle_);
        if(group.owner_.get()!=this || group.ticket_.generation!=generation_)
            throw std::logic_error("Source terminal group lost its original admission");
        if(state_!=Lifecycle::Serving)return joined_result_;
        if(group.ticket_.invocation!=outstanding_.invocation)
            throw std::logic_error("Source terminal group lost its original worker invocation");
        join=!joined_;
    }
    if(join)joinInvocation(group.ticket_,true);
    std::lock_guard lock(lifecycle_);
    // A group handles unlaunched peers. It never steals an adopted boundary's
    // right to drain its original native stream and await its own interval.
    for(std::size_t i=0;i<geometry_.channelCount();++i)if(!adopted_[i])retired_[i]=true;
    if(std::all_of(retired_.begin(),retired_.end(),[](bool value){return value;})){
        outstanding_={};state_=joined_result_?Lifecycle::Idle:Lifecycle::Failed;
    }
    return joined_result_;
}
bool ImmutableLookupService::await(Ticket ticket) {
    bool missingOwnClose=false;
    {
        std::lock_guard lock(lifecycle_);
        if(state_!=Lifecycle::Serving || !ticket.valid() || ticket.generation!=outstanding_.generation ||
           ticket.invocation!=outstanding_.invocation || ticket.channel>=adopted_.size() ||
           !adopted_[ticket.channel] || retired_[ticket.channel])
            throw std::logic_error("Immutable lookup await names a stale or unrelated channel invocation");
        auto* ack=reinterpret_cast<ImmutableLookupAcknowledgement*>(static_cast<std::byte*>(backing_->mapped->mutableHostData())+
            geometry_.offset(ticket.channel)+backing_->layout.acknowledgementOffset());
        missingOwnClose=acquire(ack->closedGeneration)!=ticket.generation;
    }
    // A peer may already wait inside the terminal join. The caller owns its
    // native terminal, so its missing close is a concrete failure: publish
    // original cancellation before the join mutex and let both callers drain.
    if(missingOwnClose)(void)cancel(ticket);
    std::lock_guard terminalLock(terminal_join_);
    bool join=false;
    {
        std::lock_guard lock(lifecycle_);
        if(state_!=Lifecycle::Serving || !ticket.valid() || ticket.generation!=outstanding_.generation ||
           ticket.invocation!=outstanding_.invocation || ticket.channel>=adopted_.size() ||
           !adopted_[ticket.channel] || retired_[ticket.channel])
            throw std::logic_error("Immutable lookup await names a stale or unrelated channel invocation");
        join=!joined_;
    }
    if(join)joinInvocation(ticket,false);
    std::lock_guard lock(lifecycle_);retired_[ticket.channel]=true;
    const bool result=joined_result_;
    if(std::all_of(retired_.begin(),retired_.end(),[](bool value){return value;})){
        outstanding_={};state_=result?Lifecycle::Idle:Lifecycle::Failed;
    }
    return result;
}

void ImmutableLookupService::joinInvocation(Ticket ticket,bool complete_parent) {
        bool missing=false;
        // A participant's own native terminal does not prove its peers have
        // reached theirs. Only the complete parent may require every close;
        // ordinary await lets the existing worker fairly drain late peers.
        const auto first=complete_parent?std::size_t{0}:ticket.channel;
        const auto end=complete_parent?geometry_.channelCount():ticket.channel+1;
        for(std::size_t i=first;i<end;++i){
            auto* ack=reinterpret_cast<ImmutableLookupAcknowledgement*>(static_cast<std::byte*>(backing_->mapped->mutableHostData())+
                geometry_.offset(i)+backing_->layout.acknowledgementOffset());
            missing=acquire(ack->closedGeneration)!=ticket.generation || missing;
        }
        if(missing)(void)cancel(ticket);
        const bool result=worker_.await({ticket.invocation});
        std::lock_guard lock(lifecycle_);
        bool aborted=false;
        for(std::size_t i=0;i<geometry_.channelCount();++i)aborted=processor(i).observeTerminalAbort(ticket.generation,ticket.invocation)||aborted;
        joined_=true;joined_result_=result&&!aborted&&!missing;
        if(aborted)serviceFailure_="Immutable lookup GPU source aborted its exact admitted I/O invocation";
        else if(missing)serviceFailure_="Immutable lookup GPU terminal did not publish every original service close";

}

bool ImmutableLookupService::cancel(Ticket ticket)
{
    std::lock_guard lock(lifecycle_);
    return cancelLocked(ticket);
}
bool ImmutableLookupService::cancelLocked(Ticket ticket)
{
    if(state_!=Lifecycle::Serving || !ticket.valid() || ticket.generation!=outstanding_.generation ||
       ticket.invocation!=outstanding_.invocation)
        return false;
    const bool alreadyCancelled=cancelled_.load(std::memory_order_acquire);
    bool live=false;
    for(std::size_t i=0;i<geometry_.channelCount();++i){
        auto* ack=reinterpret_cast<ImmutableLookupAcknowledgement*>(static_cast<std::byte*>(backing_->mapped->mutableHostData())+
            geometry_.offset(i)+backing_->layout.acknowledgementOffset());
        if(ImmutableLookupProtocol::matchesInvocation(acquire(ack->retiredInvocation),acquire(ack->retiredGeneration),ticket.generation,ticket.invocation))continue;
        live=true;
        if(!alreadyCancelled){release(ack->cancelledGeneration,ticket.generation);release(ack->cancelledInvocation,ticket.invocation);}
    }
    if(!live)return false;
    cancelled_.store(true,std::memory_order_release);
    return true;
}

bool ImmutableLookupHostProcessor::sourceAbortRequested(std::uint64_t generation,std::uint64_t invocation) const noexcept
{
    if(!invocation)return false; // Unticketed direct processor tests cannot acquire a worker abort.
    auto *ack=reinterpret_cast<ImmutableLookupAcknowledgement *>(channel_.data()+layout_.acknowledgementOffset());
    const auto abortInvocation=acquire(ack->abortInvocation);
    const auto abortGeneration=acquire(ack->abortGeneration);
    return ImmutableLookupProtocol::matchesInvocation(abortInvocation,abortGeneration,generation,invocation);
}

bool ImmutableLookupHostProcessor::observeTerminalAbort(std::uint64_t generation,std::uint64_t invocation)
{
    if(!sourceAbortRequested(generation,invocation))return false;
    stats_.terminalBytes+=2*sizeof(std::uint64_t);
    if(lastStatus_==ImmutableLookupStatus::Success)lastStatus_=ImmutableLookupStatus::Cancelled;
    if(lastFailure_.empty())lastFailure_="GPU source aborted its exact admitted I/O invocation";
    return true;
}

void ImmutableLookupHostProcessor::retireInvocation(std::uint64_t generation,std::uint64_t invocation) noexcept
{
    subordinateGrant_.reset();
    auto *ack=reinterpret_cast<ImmutableLookupAcknowledgement *>(channel_.data()+layout_.acknowledgementOffset());
    // Old borrowed Pins retain their own original source backing. Dropping the
    // acquired request never frees the channel or any captured source frame.
    const auto cancelledInvocation=acquire(ack->cancelledInvocation);
    const auto cancelledGeneration=acquire(ack->cancelledGeneration);
    if(ImmutableLookupProtocol::matchesInvocation(cancelledInvocation,cancelledGeneration,generation,invocation))
        stats_.terminalBytes+=2*sizeof(std::uint64_t);
    release(ack->retiredGeneration,generation);
    release(ack->retiredInvocation,invocation);
    stats_.terminalBytes+=2*sizeof(std::uint64_t);
}

bool ImmutableLookupService::run(void *context)
{
    auto &self=*static_cast<ImmutableLookupService *>(context);
    Ticket admitted;
    {
        std::lock_guard lock(self.lifecycle_);
        admitted=self.outstanding_;
    }
    const auto retire=[&](void*) noexcept {
        // This lock orders only host I/O cancellation and retirement. Device
        // model/control state is neither queried nor mirrored by this worker.
        std::lock_guard lock(self.lifecycle_);
        for(std::size_t i=0;i<self.geometry_.channelCount();++i)self.processor(i).retireInvocation(admitted.generation,admitted.invocation);
    };
    const std::unique_ptr<void,decltype(retire)> retirement(&self,retire);
    try
    {
        while(!self.cancelled_.load(std::memory_order_acquire))
        {
            bool closed=true;
            for(std::size_t i=0;i<self.geometry_.channelCount();++i){
                auto& processor=self.processor(i);
                (void)processor.poll(admitted.generation,admitted.invocation);
                if(processor.lastStatus()!=ImmutableLookupStatus::Success){
                    self.fatalStatus_.store(processor.lastStatus(),std::memory_order_release);
                    self.serviceFailure_=processor.lastFailure();
                    if(self.geometry_.channelCount()>1)(void)self.cancel(admitted);
                    return false;
                }
                closed=processor.closed(admitted.generation)&&closed;
            }
            if(closed)return true;
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
        self.serviceFailure_="Immutable lookup service cancelled before the GPU terminal close";
        self.fatalStatus_.store(ImmutableLookupStatus::Cancelled,std::memory_order_release);
    }
    catch(const std::exception &error)
    {
        self.serviceFailure_=error.what();
        if(self.geometry_.channelCount()>1)(void)self.cancel(admitted);
        const auto status=self.fatalStatus_.load(std::memory_order_acquire);
        self.fatalStatus_.store(status==ImmutableLookupStatus::Success?ImmutableLookupStatus::StorageFailure:status,std::memory_order_release);
    }
    return false;
}

ImmutableLookupServiceStats ImmutableLookupService::stats() const
{
    std::lock_guard lock(lifecycle_);
    if(state_==Lifecycle::Serving && !joined_)throw std::logic_error("Immutable lookup stats require terminal service ownership");
    ImmutableLookupServiceStats result;
    const auto add=[](std::uint64_t& value,std::uint64_t extra){
        if(extra>std::numeric_limits<std::uint64_t>::max()-value)throw std::overflow_error("Grouped source counter overflow");
        value+=extra;
    };
    for(std::size_t i=0;i<geometry_.channelCount();++i){
        const auto next=processor(i).stats();
        add(result.requests,next.requests);
        add(result.ids,next.ids);
        add(result.idBytes,next.idBytes);
        add(result.nativeBytes,next.nativeBytes);
        add(result.metadataBytes,next.metadataBytes);
        add(result.failures,next.failures);
        add(result.acknowledgementBytes,next.acknowledgementBytes);
        add(result.terminalBytes,next.terminalBytes);
        add(result.windowHits,next.windowHits);
        add(result.windowMisses,next.windowMisses);
        add(result.windowEvictions,next.windowEvictions);
        add(result.adaptiveBatches,next.adaptiveBatches);
        add(result.ramRows,next.ramRows);
        add(result.sourceRows,next.sourceRows);
        add(result.duplicateRows,next.duplicateRows);
        add(result.cacheAdmissions,next.cacheAdmissions);
        add(result.cacheEvictions,next.cacheEvictions);
        add(result.cacheAdmissionRejections,next.cacheAdmissionRejections);
        add(result.allPinnedResponses,next.allPinnedResponses);
    }
    return result;
}
ImmutableLookupServiceStats ImmutableLookupService::stats(DeviceId consumer) const {
    std::lock_guard lock(lifecycle_);
    if(state_==Lifecycle::Serving && !joined_)throw std::logic_error("Immutable lookup stats require terminal service ownership");
    return processor(channelIndex(consumer)).stats();
}
HostImmutableRowOwnerObservation ImmutableLookupService::retainedSourceOwnerObservation() const
{
    std::lock_guard lock(lifecycle_);
    if (state_ != Lifecycle::Idle || outstanding_.valid())
        throw std::logic_error("Immutable source retention requires an idle joined worker");
    const auto source = table_->retainedOwnerObservation();
    if (!memory_ || !backing_ || !backing_->mapped || !backing_->mapped->isBound() ||
        source.authority().get() != memory_.get() || source.allocator().world_rank != memory_->worldRank() ||
        !backing_->physical.valid() ||
        backing_->physical.owner() != PhysicalMemoryOwner::ImmutableLookupStaging ||
        backing_->physical.bytes() != backing_->mapped->sizeBytes())
        throw std::logic_error("Immutable source/channel retained claims lost their original authority");
    const auto channel = backing_->physical.bytes();
    if (channel > std::numeric_limits<std::size_t>::max() - source.lookupStagingBytes())
        throw std::overflow_error("Immutable source/channel retained staging observation overflow");
    return HostImmutableRowOwnerObservation(source.authority(), source.allocator(), source.hostPlacement(),
                                            source.tableStorageBytes(),
                                            source.lookupStagingBytes() + channel);
}

std::string ImmutableLookupService::lastFailure() const
{
    std::lock_guard lock(lifecycle_);
    if(state_==Lifecycle::Serving && !joined_)throw std::logic_error("Immutable lookup diagnostics require terminal service ownership");
    return serviceFailure_.empty()?processor_.lastFailure():serviceFailure_;
}
void ImmutableLookupService::requireHealthy() const
{
    const auto status=fatalStatus_.load(std::memory_order_acquire);
    if(status!=ImmutableLookupStatus::Success)
        throw std::runtime_error("Immutable lookup storage lane has a fatal I/O protocol status "+
            std::to_string(static_cast<std::uint64_t>(status))+"; the native terminal retains its precise I/O diagnostic");
}
} // namespace llaminar2::qwen4exp
