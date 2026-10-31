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
#include "kernels/qwen4exp/AdaptiveLookupPreparedStorage.h"
#include "ImmutableLookupService.h"
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
    const std::atomic<bool> *cancelled)
    :table_(&table),layout_(layout),channel_(channel),tableId_(tableId),fatalStatus_(fatalStatus),cancelled_(cancelled)
{
    if(tableId==0||channel.size()<layout.bytes()||reinterpret_cast<std::uintptr_t>(channel.data())%64!=0||
        table.source().rowBytes()!=layout.rowBytes() ||
        layout.subordinate()!=table.subordinateRequester().has_value() ||
        (table.maximumLiveRows() && layout.maximumIds()>*table.maximumLiveRows()))
        throw std::invalid_argument("Immutable lookup requires aligned complete channel storage and exact table-row geometry");
    // Mode/source/bounds checks precede metadata initialization and binding.
    if(layout.subordinate())table.bindSubordinate(this);
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
    DeviceId hostDevice,TransferEngine &transfers)
{
    if(!table||!memory||!hostDevice.is_cpu()||devices.empty()||table->source().rowBytes()!=layout.rowBytes()||
        table->retainedOwnerObservation().allocator()!=PhysicalMemoryAllocatorIdentity{memory->worldRank(),hostDevice})
        throw std::invalid_argument("Immutable lookup service requires table, canonical admission and explicit mapped endpoints");
    if(const auto bound=table->maximumLiveRows();bound && layout.maximumIds()>*bound)
        throw std::invalid_argument("Hosted lookup channel exceeds the provider's admitted original ID envelope");
    const auto source_owner=table->retainedOwnerObservation();
    if(source_owner.authority().get()!=memory.get())
        throw std::invalid_argument("Hosted lookup cannot borrow a provider from a different PMA");
    if(layout.subordinate())
    {
        const auto requester=table->subordinateRequester();
        if(!requester || devices.size()!=1 || requester->world_rank!=memory->worldRank() ||
            devices.front()!=requester->device)
            throw std::invalid_argument("Subordinate source service requires its exact single declared GPU requester");
    }
    else if(table->subordinateRequester())
        throw std::invalid_argument("Subordinate source cannot enter the ordinary channel ABI");
    const auto bytes=hostBytes(layout);
    auto physical=memory->claimNewAllocation(hostDevice,PhysicalMemoryOwner::ImmutableLookupStaging,bytes);
    auto mapped=transfers.allocateMappedHostRegion(layout.bytes(),devices);
    if(!mapped||!mapped->isBound()||mapped->sizeBytes()!=bytes)
        throw std::runtime_error("Immutable lookup mapped allocation differs from its admitted physical extent");
    auto backing=std::make_shared<ImmutableLookupChannelBacking>(ImmutableLookupChannelBacking{
        std::move(physical),std::move(mapped),layout,nextTableId()});
    // Only this independently scheduled native-row I/O owner inherits the
    // admitted OpenMP partition. Restore the creator before returning; generic
    // retained workers and CPU expert compute placement remain unchanged.
    const ScopedOpenMPPartitionAffinity io_worker_affinity;
    return std::shared_ptr<ImmutableLookupService>(
        new ImmutableLookupService(std::move(table), std::move(backing), std::move(memory)));
}

ImmutableLookupService::ImmutableLookupService(std::shared_ptr<HostImmutableRowProvider> table,
                                               std::shared_ptr<ImmutableLookupChannelBacking> backing,
                                               std::shared_ptr<const PhysicalMemoryAuthority> memory)
    : table_(std::move(table)), backing_(std::move(backing)), memory_(std::move(memory)),
      processor_(*table_, backing_->layout,
                 {static_cast<std::byte *>(backing_->mapped->mutableHostData()), backing_->layout.bytes()},
                 backing_->tableId, &fatalStatus_, &cancelled_)
{
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
    auto *base=static_cast<std::byte *>(backing_->mapped->deviceAlias(device));
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
    const auto requester = table_->subordinateRequester();
    if (!backing_->layout.subordinate() || !requester || requester->device != device || !device.is_gpu() ||
        !backing_->mapped->hasDevice(device))
        throw std::invalid_argument(
            "Adaptive native binding requires its exact declared protocol2 requester");
    table_->authenticateSource();
    auto *base = static_cast<std::byte *>(backing_->mapped->deviceAlias(device));
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
std::shared_ptr<AdaptiveLookupPreparedStorage> ImmutableSubordinateLookupBinding::declareStorage(
    const AdaptiveImmutableRowCacheNativePlan &plan, std::shared_ptr<TensorBase> cache,
    std::shared_ptr<TensorBase> frame, std::shared_ptr<TensorBase> protocol) const
{
    requireFor(plan.parameters());
    return owner_->declareAdaptiveStorage(*this, plan, std::move(cache), std::move(frame),
                                          std::move(protocol));
}
std::shared_ptr<AdaptiveLookupPreparedStorage> ImmutableLookupService::declareAdaptiveStorage(
    const ImmutableSubordinateLookupBinding &binding, const AdaptiveImmutableRowCacheNativePlan &plan,
    std::shared_ptr<TensorBase> cache, std::shared_ptr<TensorBase> frame,
    std::shared_ptr<TensorBase> protocol)
{
    if (isGraphCaptureActive())
        throw std::logic_error("Adaptive storage owner must be declared before capture");
    binding.requireFor(plan.parameters());
    requireHealthy();
    // This mutex serializes declarations only; no file operation, native upload,
    // physical claim or live GPU state is performed while holding it.
    std::lock_guard lock(adaptive_setup_);
    if(pooled_storage_)throw std::invalid_argument("Original source channel already belongs to the pooled source arm");
    if (adaptive_storage_)
    {
        if (adaptive_storage_->plan_ != plan || adaptive_storage_->cache_ != cache ||
            adaptive_storage_->frame_ != frame || adaptive_storage_->protocol_ != protocol)
            throw std::invalid_argument(
                "Adaptive source channel cannot replace its original storage/plan owner");
        adaptive_storage_->declarations();
        return adaptive_storage_;
    }
    adaptive_storage_ = std::shared_ptr<AdaptiveLookupPreparedStorage>(
        new AdaptiveLookupPreparedStorage(plan, std::move(cache), std::move(frame), std::move(protocol)));
    return adaptive_storage_;
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
    const auto requester=owner_->table_->subordinateRequester();const auto& actual=owner_->table_->source();const auto& layout=owner_->backing_->layout;
    if(!requester || *requester!=geometry.allocator() || actual.rows!=p.coordinator.policy.sourceRows() || actual.row_elements!=p.row_elements ||
        ggufToTensorType(actual.format)!=p.format || actual.rowBytes()!=p.owner.row_bytes || layout.protocol()!=kImmutableLookupSubordinateProtocol ||
        layout.maximumIds()!=p.coordinator.policy.maximumUnique() || layout.rowBytes()!=p.owner.row_bytes ||
        owner_->backing_->tableId!=view_.base.table || p.owner.table!=view_.base.table || p.coordinator.table!=view_.base.table ||
        p.owner.source_storage_generation!=storageGeneration() || p.coordinator.source_storage_generation!=storageGeneration())
        throw std::invalid_argument("Pooled source native fields do not match the real protocol2 provider and original source generation");
    auto* base=static_cast<std::byte*>(owner_->backing_->mapped->deviceAlias(device_));
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
    const PooledImmutableSourceGeometry& geometry,std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner,
    std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream)const{
    requireFor(geometry);
    return owner_->declarePooledSourceStorage(*this,geometry,std::move(owner),std::move(coordinator),std::move(memory),stream);
}
std::shared_ptr<PooledImmutableSourcePreparedStorage> ImmutableLookupService::declarePooledSourceStorage(
    const ImmutableSubordinateLookupBinding& binding,const PooledImmutableSourceGeometry& geometry,
    std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner,std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,
    std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream){
    if(isGraphCaptureActive())throw std::logic_error("Pooled source registration must be materialized before capture");
    binding.requireFor(geometry);requireHealthy();
    if(!memory || memory.get()!=memory_.get() || !owner || !coordinator)
        throw std::invalid_argument("Pooled source registration requires its actual original PMA and native authorities");
    table_->authenticateSource();
    // Cold declaration is serialized once. No model state or host policy shadow
    // is consulted; failure never publishes partial roots into this service.
    std::lock_guard lock(adaptive_setup_);
    if(adaptive_storage_)throw std::invalid_argument("Original source channel already belongs to the single-GPU adaptive arm");
    if(pooled_storage_){
        if(pooled_storage_->geometry().parameters().identity!=geometry.parameters().identity || pooled_storage_->owner()!=owner || pooled_storage_->coordinator()!=coordinator)
            throw std::invalid_argument("Pooled source cannot replace its original native source/accepted-owner/coordinator registration");
        pooled_storage_->captureIdentity();return pooled_storage_;
    }
    auto storage=PooledImmutableSourcePreparedStorage::materialize(geometry,std::move(owner),std::move(coordinator),std::move(memory),stream);
    pooled_storage_=std::move(storage);return pooled_storage_;
}
void ImmutableSubordinateLookupBinding::requireFor(const AdaptiveNativeParameters &p) const
{
    if (!owner_ || p.device != device_ || !owner_->backing_->layout.subordinate() ||
        !owner_->backing_->mapped->hasDevice(device_))
        throw std::invalid_argument("Adaptive native source has changed channel/device ownership");
    // Identity authentication must remain usable for terminal cleanup of a
    // failed source lane. The enclosing launch boundary owns requireHealthy();
    // native sticky failure still prevents a failed component from doing work.
    const auto requester = owner_->table_->subordinateRequester();
    const auto &source = owner_->table_->source();
    if (!requester || requester->device != device_ || requester->world_rank != p.world_rank ||
        source.rows != p.source_rows || source.row_elements != static_cast<std::uint64_t>(p.row_elements) ||
        ggufToTensorType(source.format) != p.format || source.rowBytes() != p.row_bytes ||
        owner_->backing_->tableId != view_.base.table ||
        owner_->backing_->layout.maximumIds() != p.maximum_ids || view_.base.maximumIds != p.maximum_ids ||
        view_.base.rowBytes != p.row_bytes)
        throw std::invalid_argument("Adaptive native source/layout differs from its checked declaration");
    auto *base = static_cast<std::byte *>(owner_->backing_->mapped->deviceAlias(device_));
    const auto &l = owner_->backing_->layout;
    if (view_.base.request != reinterpret_cast<ImmutableLookupRequest *>(base + l.requestOffset()) ||
        view_.base.completion != reinterpret_cast<ImmutableLookupCompletion *>(base + l.completionOffset()) ||
        view_.base.acknowledgement !=
            reinterpret_cast<ImmutableLookupAcknowledgement *>(base + l.acknowledgementOffset()) ||
        view_.base.ids != reinterpret_cast<std::int64_t *>(base + l.idsOffset()) ||
        view_.base.nativeRows != base + l.payloadOffset() ||
        view_.intents != reinterpret_cast<ImmutableRowSubordinateIntent *>(base + l.intentsOffset()) ||
        view_.acknowledgements !=
            reinterpret_cast<ImmutableRowInvalidationAcknowledgement *>(base + l.invalidationsOffset()))
        throw std::invalid_argument("Adaptive native mapped aliases changed after binding");
}
void ImmutableSubordinateLookupBinding::requirePreparedStorage(const AdaptiveImmutableRowCacheNativePlan &plan)const
{
    requireFor(plan.parameters());
    std::shared_ptr<AdaptiveLookupPreparedStorage> storage;
    {std::lock_guard lock(owner_->adaptive_setup_);storage=owner_->adaptive_storage_;}
    if(!storage)throw std::logic_error("Adaptive native factory requires its original prepared source owner");
    storage->requireFor(plan);
}
void ImmutableSubordinateLookupBinding::requireStorage(const AdaptiveLookupPreparedStorage &storage) const
{
    if (!owner_)
        throw std::invalid_argument("Adaptive storage binding has no source service");
    std::lock_guard lock(owner_->adaptive_setup_);
    if (owner_->adaptive_storage_.get() != &storage)
        throw std::invalid_argument(
            "Adaptive cache/frame/protocol belong to another original source service");
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

ImmutableLookupService::Ticket ImmutableLookupService::arm(std::uint64_t generation)
{
    std::lock_guard lock(lifecycle_);
    if(state_!=Lifecycle::Idle||outstanding_.valid()||generation==0||generation<=generation_)
        throw std::logic_error("Immutable lookup service requires an idle healthy lane and a newer admission generation");
    generation_=generation;cancelled_.store(false,std::memory_order_release);state_=Lifecycle::Serving;
    // run() acquires lifecycle_ before reading this immutable ticket. dispatch
    // may wake the retained thread before returning, so publication stays under
    // the same lock until the actual worker invocation has been assigned.
    const auto invocation=worker_.dispatch(&ImmutableLookupService::run,this);
    outstanding_={generation,invocation.generation};
    return outstanding_;
}

bool ImmutableLookupService::await(Ticket ticket)
{
    {
        std::lock_guard lock(lifecycle_);
        if(state_!=Lifecycle::Serving||!ticket.valid()||ticket.generation!=outstanding_.generation||ticket.invocation!=outstanding_.invocation)
            throw std::logic_error("Immutable lookup await names a stale or unrelated service invocation");
    }
    // await() is a terminal boundary. A complete GPU parent must have published
    // its close metadata; never mask a missing close by waiting indefinitely.
    auto *ack=reinterpret_cast<ImmutableLookupAcknowledgement *>(
        backing_->mapped->mutableHostData(backing_->layout.acknowledgementOffset()));
    const bool missingClose=acquire(ack->closedGeneration)!=generation_;
    if(missingClose)(void)cancel(ticket);
    const bool workerResult=worker_.await({ticket.invocation});
    std::lock_guard lock(lifecycle_);
    // The complete source parent has retired before await. Only mapped I/O
    // metadata is inspected here; a publication racing CPU retirement is still
    // counted once by this ticket's unique terminal transition.
    const bool aborted=processor_.observeTerminalAbort(ticket.generation,ticket.invocation);
    const bool result=workerResult && !aborted;
    if(aborted)fatalStatus_.store(ImmutableLookupStatus::Cancelled,std::memory_order_release);
    // The worker is retired before touching its diagnostic string.
    if(aborted)serviceFailure_="Immutable lookup GPU source aborted its exact admitted I/O invocation";
    else if(missingClose)serviceFailure_="Immutable lookup GPU terminal did not publish its explicit service close";
    outstanding_={};state_=result?Lifecycle::Idle:Lifecycle::Failed;
    return result;
}

bool ImmutableLookupService::cancel(Ticket ticket)
{
    std::lock_guard lock(lifecycle_);
    if(state_!=Lifecycle::Serving || !ticket.valid() || ticket.generation!=outstanding_.generation ||
       ticket.invocation!=outstanding_.invocation)
        return false;
    auto *ack=reinterpret_cast<ImmutableLookupAcknowledgement *>(
        backing_->mapped->mutableHostData(backing_->layout.acknowledgementOffset()));
    // The worker publishes retirement under this same lifecycle lock. A late
    // caller cannot add an unobserved cancellation message after retirement.
    const auto retiredInvocation=acquire(ack->retiredInvocation);
    const auto retiredGeneration=acquire(ack->retiredGeneration);
    if(ImmutableLookupProtocol::matchesInvocation(retiredInvocation,retiredGeneration,ticket.generation,ticket.invocation))
        return false;
    // The complete terminal boundary can repeat cancellation while joining the
    // same live invocation. Its one metadata message is published exactly once.
    if(cancelled_.load(std::memory_order_acquire))return true;
    // This is CPU I/O control only. No GPU state or terminal result is read.
    // Both fields are atomic because readers can reject an older publication
    // while a later admission publishes its distinct generation body.
    release(ack->cancelledGeneration,ticket.generation);
    release(ack->cancelledInvocation,ticket.invocation);
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
        self.processor_.retireInvocation(admitted.generation,admitted.invocation);
    };
    const std::unique_ptr<void,decltype(retire)> retirement(&self,retire);
    try
    {
        while(!self.cancelled_.load(std::memory_order_acquire))
        {
            (void)self.processor_.poll(admitted.generation,admitted.invocation);
            if(self.processor_.lastStatus()!=ImmutableLookupStatus::Success)
            {
                self.fatalStatus_.store(self.processor_.lastStatus(),std::memory_order_release);
                return false; // The failed completion is already visible; this lane cannot accept another request.
            }
            if(self.processor_.closed(admitted.generation))return self.processor_.lastFailure().empty();
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
        self.serviceFailure_="Immutable lookup service cancelled before the GPU terminal close";
        self.fatalStatus_.store(ImmutableLookupStatus::Cancelled,std::memory_order_release);
    }
    catch(const std::exception &error)
    {
        self.serviceFailure_=error.what();
        const auto status=self.processor_.lastStatus();
        self.fatalStatus_.store(status==ImmutableLookupStatus::Success?ImmutableLookupStatus::StorageFailure:status,std::memory_order_release);
    }
    return false;
}

ImmutableLookupServiceStats ImmutableLookupService::stats() const
{
    std::lock_guard lock(lifecycle_);
    if(state_==Lifecycle::Serving)throw std::logic_error("Immutable lookup stats require terminal service ownership");
    return processor_.stats();
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
    if(state_==Lifecycle::Serving)throw std::logic_error("Immutable lookup diagnostics require terminal service ownership");
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
