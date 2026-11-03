/**
 * @file PooledImmutableDirectedTransport.cpp
 * @brief Materialize only original two-channel mapped or peer edges priced by the shared runtime geometry.
 *
 * All candidate/floor/growth BOMs use one source-owned channel shape. No table
 * identity is invented for planning. Cold setup borrows the already prepared
 * original participant workers; no capability, alternative transport or hidden
 * nested graph is introduced. Expert placement/rebalancing is independent.
 */
#include "PooledImmutableDirectedTransport.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <algorithm>
#include <stdexcept>
namespace llaminar2::qwen4exp {
namespace {
/** @brief Distinguish original mandatory protocol roots, candidate roots and optional growth. */
enum class Contribution {Mandatory,Complete,Growth};
/** @return The exact immutable original allocator observation, rejecting duplicated or missing resources. */
PhysicalMemoryResource resource(std::span<const PhysicalMemoryResource> resources,PhysicalMemoryAllocatorIdentity allocator){
    const PhysicalMemoryResource* found=nullptr;
    for(const auto& r:resources)if(r.world_rank==allocator.world_rank && r.device==allocator.device){if(found)throw std::invalid_argument("Duplicate original directed resource");found=&r;}
    if(!found)throw std::invalid_argument("Missing original directed GPU resource");return *found;
}
/** @brief Authenticate explicit mapped endpoint policy before quoting or creating transport. */
void requireRemote(PhysicalMemoryAllocatorIdentity coordinator, PhysicalMemoryAllocatorIdentity owner,
    std::optional<PeerAccessCoverage> observation)
{
    ImmutableRowTableDirectedTransportPolicy::require(coordinator, owner, observation);
}
/** @return One exact source-owned contribution, never subtracting a capacity balance. */
PhysicalMemoryPlan contribution(const PooledImmutablePhysicalOwnerGeometry& physical,const PooledImmutableCoordinatorGeometry& policy,
    PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> resources,Contribution kind){
    PooledImmutableCoordinatorNativePlan::requireOriginalCohort(physical,policy);
    if(!host.device.is_cpu() || host.world_rank!=physical.coordinator().world_rank)throw std::invalid_argument("Directed channel changed original host allocator");
    const auto coordinator=resource(resources,physical.coordinator());PhysicalMemoryPlanBuilder builder;
    for(const auto& owner:physical.owners()){
        requireRemote(physical.coordinator(),owner.allocator,physical.peerAccess());if(owner.allocator==physical.coordinator())continue;
        const auto observed=resource(resources,owner.allocator);
        const auto floor=PooledImmutableDirectedChannelGeometry::mandatory(physical,policy,owner.coordinate);
        const auto actual=PooledImmutableDirectedChannelGeometry::fromGeometry(physical,policy,owner.coordinate);
        const std::array baseline{floor.forwardMemory(),floor.reverseMemory()},complete{actual.forwardMemory(),actual.reverseMemory()};
        for(std::size_t direction=0;direction<2;++direction){
            auto memory=kind==Contribution::Mandatory?baseline[direction]:complete[direction];
            if(kind==Contribution::Growth){
                if(memory.mapped_host_bytes<baseline[direction].mapped_host_bytes ||
                    memory.producer_payload_bytes<baseline[direction].producer_payload_bytes ||
                    memory.cursor_bytes_per_device!=baseline[direction].cursor_bytes_per_device)
                    throw std::logic_error("Directed transport growth changed original mandatory cursor roots");
                memory.mapped_host_bytes-=baseline[direction].mapped_host_bytes;
                memory.producer_payload_bytes-=baseline[direction].producer_payload_bytes;memory.cursor_bytes_per_device=0;
            }
            if(memory.producer_payload_bytes)builder.add(direction==0?coordinator:observed,PhysicalMemoryOwner::ActivationTransportStaging,memory.producer_payload_bytes);
            if(memory.mapped_host_bytes)builder.add(host,PhysicalMemoryOwner::ActivationTransportStaging,memory.mapped_host_bytes);
            if(memory.cursor_bytes_per_device){builder.add(coordinator,PhysicalMemoryOwner::ActivationTransportStaging,memory.cursor_bytes_per_device);builder.add(observed,PhysicalMemoryOwner::ActivationTransportStaging,memory.cursor_bytes_per_device);}
        }
    }
    return builder.build();
}
}
PhysicalMemoryPlan PooledImmutableDirectedTransport::mandatoryBOM(const PooledImmutablePhysicalOwnerGeometry& p,const PooledImmutableCoordinatorGeometry& c,PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> resources){return contribution(p,c,host,resources,Contribution::Mandatory);}
PhysicalMemoryPlan PooledImmutableDirectedTransport::completeBOM(const PooledImmutablePhysicalOwnerGeometry& p,const PooledImmutableCoordinatorGeometry& c,PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> resources){return contribution(p,c,host,resources,Contribution::Complete);}
PhysicalMemoryPlan PooledImmutableDirectedTransport::slotDependentBOM(const PooledImmutablePhysicalOwnerGeometry& p,const PooledImmutableCoordinatorGeometry& c,PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> resources){return contribution(p,c,host,resources,Contribution::Growth);}
std::shared_ptr<const PooledImmutableDirectedTransport> PooledImmutableDirectedTransport::create(std::shared_ptr<ImmutableLookupExecutionBoundary> boundary, std::shared_ptr<const ClusterInventory> originalInventory,
    std::optional<PooledImmutableReplicatedReaderGeometry> readers){
    if(isGraphCaptureActive() || !boundary || !boundary->isPooled())throw std::invalid_argument("Directed transport requires original cold pooled source");
    const auto storage=boundary->pooledStorage();const auto& retained=storage->retainedPreparedSourceWorkers();
    if(!retained || storage->originalOwners().empty() || retained->owners().size()!=storage->originalOwners().size())throw std::logic_error("Directed transport requires actual original prepared source participants");
    const auto memory=retained->source()->memory();storage->coordinator()->requireAuthority(memory);
    const auto& coordinator=retained->coordinator();
    const auto current=GPUDeviceContextPool::instance().retainExistingPreparedContext(coordinator.device(),memory);
    coordinator.requireWorker(current.worker());
    if(current.generation()!=coordinator.generation())throw std::logic_error("Directed coordinator original generation changed");
    if(coordinator.worker()->ownsCurrentThread())throw std::invalid_argument("Directed cohort setup requires its off-worker cold coordinator");
    std::vector<PhysicalMemoryAllocatorIdentity> originalOwners;
    for (const auto& owner : storage->originalOwners()) originalOwners.push_back(owner->plan().allocator());
    const auto observation = storage->originalOwners().front()->plan().peerAccess();
    for (const auto& owner : storage->originalOwners())
        if (owner->plan().peerAccess() != observation)
            throw std::invalid_argument("Directed owners changed the original transport observation");
    ImmutableRowTableDirectedTransportPolicy::requireOriginal(storage->coordinator()->plan().allocator(),
        originalOwners, originalInventory, observation);
    // Reader membership is independent of physical owner order and CPU source
    // channel population. Authenticate every original endpoint before creating
    // any channel, so a later invalid reader cannot partially select transport.
    std::vector<std::pair<std::uint32_t,PreparedGPUContextUsePermit>> readerPermits;
    if (readers) {
        if (readers->coordinator() != storage->coordinator()->plan().allocator() ||
            readers->consumers().size() != storage->geometry().parameters().consumers ||
            readers->maximumOriginal() != storage->geometry().parameters().coordinator.policy.maximumOriginal() ||
            readers->rowElements() != storage->geometry().parameters().row_elements)
            throw std::invalid_argument("Shared reader transport changed original source geometry");
        if (!originalInventory)
            throw std::invalid_argument("Shared readers require their retained original rank inventory");
        const auto actualEdges = ImmutableRowTableDirectedTransportPolicy::observeReaders(readers->coordinator(),
            readers->consumers(), originalInventory->getRank(readers->coordinator().world_rank));
        for (const auto &edge : actualEdges)
            if (readers->peerAccess(edge.reader) != edge.coverage)
                throw std::invalid_argument("Shared reader transport changed its original observed peer route");
        for (std::uint32_t i = 0; i < readers->consumers().size(); ++i) {
            const auto consumer = readers->consumers()[i];
            if (consumer == readers->coordinator()) continue;
            auto permit = retained->source()->acquire(consumer.device);
            const auto actual = GPUDeviceContextPool::instance().retainExistingPreparedContext(consumer.device, memory);
            permit.requireWorker(actual.worker());
            if (permit.generation() != actual.generation() || permit.memory().get() != memory.get() ||
                permit.memory().owner_before(memory) || memory.owner_before(permit.memory()) ||
                permit.worker()->ownsCurrentThread())
                throw std::logic_error("Shared reader transport lost its original off-worker source/PMA generation");
            readerPermits.emplace_back(i, std::move(permit));
        }
    } else if (storage->geometry().parameters().consumers != 1) {
        throw std::invalid_argument("Plural shared source requires every original compute-reader channel");
    }
    std::vector<void*> ownerStreams(retained->owners().size());
    // Authenticate the COMPLETE actual participant population before either
    // channel materializes. An unsupported later owner cannot create a partial
    // apparently usable mapped topology.
    for(std::uint32_t i=0;i<retained->owners().size();++i){
        const auto& owner=storage->originalOwners()[i];const auto& permit=retained->owners()[i];owner->requireAuthority(memory);
        const auto actual=GPUDeviceContextPool::instance().retainExistingPreparedContext(permit.device(),memory);
        permit.requireWorker(actual.worker());if(actual.generation()!=permit.generation())throw std::logic_error("Directed owner original generation changed");
        requireRemote(storage->coordinator()->plan().allocator(),owner->plan().allocator(),owner->plan().peerAccess());
        if(permit.worker()->ownsCurrentThread())throw std::invalid_argument("Directed setup cannot occupy an original participant worker");
    }
    auto result=std::shared_ptr<PooledImmutableDirectedTransport>(new PooledImmutableDirectedTransport(std::move(boundary),std::move(originalInventory)));
    result->requireOriginalSource();result->remotes_.reserve(retained->owners().size());
    void* coordinatorStream=nullptr;coordinator.worker()->submitAndWait([&]{coordinatorStream=coordinator.worker()->defaultStream();});
    for(std::uint32_t i=0;i<retained->owners().size();++i){
        const auto& owner=storage->originalOwners()[i];const auto& permit=retained->owners()[i];
        if(owner->plan().allocator()==storage->coordinator()->plan().allocator())continue;
        permit.worker()->submitAndWait([&]{ownerStreams[i]=permit.worker()->defaultStream();});
    }
    for(std::uint32_t i=0;i<retained->owners().size();++i){
        const auto& owner=storage->originalOwners()[i];
        if(owner->plan().allocator()==storage->coordinator()->plan().allocator())continue;
        void* ownerStream=ownerStreams[i];
        const auto pin=PooledImmutableCapturedPinEdge::bind(storage->coordinator(),owner,memory,coordinatorStream,ownerStream,result->original_inventory_);
        const auto returned=PooledImmutableCapturedPinReturn::bind(pin,memory,ownerStream,coordinatorStream);
        const auto free=PooledImmutableCapturedFreeWave::bind(returned,memory);
        const auto victim=PooledImmutableCapturedVictimWave::bind(free,memory);
        const auto source=PooledImmutableCapturedSourceWave::bind(victim,result->boundary_,memory);
        const auto publication=PooledImmutableCapturedPublicationWave::bind(source,memory);
        const auto read=PooledImmutableCapturedReadWave::bind(publication,memory);
        const auto retirement=PooledImmutableCapturedRetirementWave::bind(read,memory);
        result->remotes_.push_back({i,pin,returned,free,victim,source,publication,read,retirement});
    }
    if (readers) {
        result->reader_geometry_ = std::move(readers);
        for (auto& [coordinate, permit] : readerPermits) {
            void* readerStream = nullptr;
            permit.worker()->submitAndWait([&] { readerStream = permit.worker()->defaultStream(); });
            const auto& geometry = *result->reader_geometry_;
            auto& transfer = TransferEngine::instance();
            const auto peer = geometry.payloadStorage(geometry.consumers()[coordinate]) ==
                CapturedTransferPayloadStorage::PeerDevice;
            const auto &rank = result->original_inventory_->getRank(geometry.coordinator().world_rank);
            const auto capacity = std::max(geometry.resultMaximum(), sizeof(replicated_reader::Header));
            auto output = peer ? transfer.createPeerCapturedTransferChannel(*memory, DeviceId::cpu(),
                coordinator.device(), coordinatorStream, permit.device(), readerStream, capacity, rank) :
                transfer.createCapturedTransferChannel(*memory, DeviceId::cpu(), coordinator.device(),
                    coordinatorStream, permit.device(), readerStream, capacity);
            auto ack = peer ? transfer.createPeerCapturedTransferChannel(*memory, DeviceId::cpu(),
                permit.device(), readerStream, coordinator.device(), coordinatorStream,
                sizeof(replicated_reader::Header), rank) :
                transfer.createCapturedTransferChannel(*memory, DeviceId::cpu(), permit.device(), readerStream,
                    coordinator.device(), coordinatorStream, sizeof(replicated_reader::Header));
            std::array<PhysicalMemoryAllocationLease,2> controlClaims;
            std::array<std::shared_ptr<DeviceTransferBuffer>,2> controls;
            const std::array endpoints{coordinator.device(), permit.device()};
            const std::array workers{coordinator.worker(), permit.worker()};
            for (std::size_t endpoint = 0; endpoint < endpoints.size(); ++endpoint) {
                controlClaims[endpoint] = memory->claimNewAllocation(endpoints[endpoint],
                    PhysicalMemoryOwner::ActivationTransportStaging, sizeof(replicated_reader::Control));
                workers[endpoint]->submitAndWait([&] {
                    controls[endpoint] = transfer.allocateDeviceTransferBuffer(
                        sizeof(replicated_reader::Control), endpoints[endpoint]);
                });
            }
            result->readers_.push_back(std::shared_ptr<const PooledImmutableReplicatedReaderTransport>(
                new PooledImmutableReplicatedReaderTransport(coordinate, coordinator, std::move(permit),
                    std::move(output), std::move(ack), std::move(controlClaims), std::move(controls))));
        }
    }
    result->requireCurrent();return result;
}
const PooledImmutableRemoteTransport& PooledImmutableDirectedTransport::remote(std::uint32_t coordinate)const{
    requireCurrent();const auto found=std::find_if(remotes_.begin(),remotes_.end(),[&](const auto& value){return value.coordinate==coordinate;});
    if(found==remotes_.end())throw std::invalid_argument("Directed transport coordinate is local or absent");return *found;
}
const std::shared_ptr<DeviceTransferBuffer>& PooledImmutableReplicatedReaderTransport::control(
    CapturedTransferEndpoint side) const
{
    if (side == CapturedTransferEndpoint::Producer) return controls_[0];
    if (side == CapturedTransferEndpoint::Consumer) return controls_[1];
    throw std::invalid_argument("Unknown shared PLE result endpoint");
}
void PooledImmutableReplicatedReaderTransport::requireCurrent() const
{
    const auto& memory = source_.memory();
    if (!memory || reader_.memory().get() != memory.get() ||
        reader_.memory().owner_before(memory) || memory.owner_before(reader_.memory()) ||
        !result_ || !acknowledgement_ || source_.device() == reader_.device() ||
        result_->endpointDevice(CapturedTransferEndpoint::Producer) != source_.device() ||
        result_->endpointDevice(CapturedTransferEndpoint::Consumer) != reader_.device() ||
        acknowledgement_->endpointDevice(CapturedTransferEndpoint::Producer) != reader_.device() ||
        acknowledgement_->endpointDevice(CapturedTransferEndpoint::Consumer) != source_.device())
        throw std::logic_error("Shared PLE reader lost original source/endpoint/PMA ownership");
    for (std::size_t i = 0; i < controls_.size(); ++i) {
        const auto device = i ? reader_.device() : source_.device();
        if (!controls_[i] || !controls_[i]->isBound() || controls_[i]->device() != device ||
            controls_[i]->sizeBytes() != sizeof(replicated_reader::Control) ||
            reinterpret_cast<std::uintptr_t>(controls_[i]->deviceData()) % alignof(replicated_reader::Control) != 0 ||
            control_claims_[i].bytes() != sizeof(replicated_reader::Control))
            throw std::logic_error("Shared PLE reader lost its exact original metadata storage");
    }
    for (const auto* original : {&source_, &reader_}) {
        const auto current = GPUDeviceContextPool::instance().retainExistingPreparedContext(original->device(), memory);
        original->requireWorker(current.worker());
        if (current.generation() != original->generation())
            throw std::logic_error("Shared PLE reader worker generation changed");
    }
}
std::vector<std::shared_ptr<CapturedTransferChannel>> PooledImmutableDirectedTransport::originalChannels(std::optional<CancellationEdges> edges) const
{
    std::vector<std::shared_ptr<CapturedTransferChannel>> channels;
    channels.reserve(2 * (remotes_.size() + readers_.size()));
    if (!edges || *edges != CancellationEdges::ComputeReaders) for (const auto& remote : remotes_) {
        channels.push_back(remote.pin->channel_);
        channels.push_back(remote.returned->channel_);
    }
    if (!edges || *edges != CancellationEdges::StorageOwners) for (const auto& reader : readers_) {
        channels.push_back(reader->result());
        channels.push_back(reader->acknowledgement());
    }
    return channels;
}
void PooledImmutableDirectedTransport::beginInvocation(CapturedTransferInvocation original) const
{
    requireCurrent();
    const auto channels = originalChannels();
    std::size_t begun = 0;
    try {
        for (const auto& channel : channels) {
            TransferEngine::instance().beginCapturedTransferInvocation(channel, original);
            ++begun;
        }
    } catch (...) {
        // No participant has launched yet. Preserve the original failure and
        // terminally cancel every already-bound owner; never silently reuse it.
        for (std::size_t i = 0; i < begun; ++i) {
            if (!TransferEngine::instance().cancelCapturedTransferInvocation(channels[i], original)) std::terminate();
            // This cold operation has not returned to any participant launcher:
            // every begun native endpoint is still unsubmitted. Retire only
            // these exact cancelled admissions, not a foreign active ticket.
            if (TransferEngine::instance().finishCapturedTransferInvocation(channels[i], original) !=
                CapturedTransferInvocationRetirement::Cancelled) std::terminate();
        }
        throw;
    }
}
std::function<void()> PooledImmutableDirectedTransport::failureCallback(CapturedTransferInvocation original, CancellationEdges edges) const
{
    requireCurrent();
    return [channels = originalChannels(edges), original] {
        std::exception_ptr failure;
        for (const auto& channel : channels) {
            try {
                if (!TransferEngine::instance().cancelCapturedTransferInvocation(channel, original))
                    throw std::logic_error("Shared source cancellation lost its original channel invocation");
            } catch (...) {
                if (!failure) failure = std::current_exception();
            }
        }
        if (failure) std::rethrow_exception(failure);
    };
}
CapturedTransferInvocationRetirement PooledImmutableDirectedTransport::finishInvocation(CapturedTransferInvocation original) const
{
    requireCurrent();
    std::exception_ptr failure;
    auto outcome = CapturedTransferInvocationRetirement::Completed;
    for (const auto& channel : originalChannels()) {
        try {
            if (TransferEngine::instance().finishCapturedTransferInvocation(channel, original) ==
                CapturedTransferInvocationRetirement::Cancelled) outcome = CapturedTransferInvocationRetirement::Cancelled;
        }
        catch (...) { if (!failure) failure = std::current_exception(); }
    }
    if (failure) std::rethrow_exception(failure);
    return outcome;
}
void PooledImmutableDirectedTransport::requireOriginalSource() const {
    const auto storage=boundary_->pooledStorage();const auto& retained=storage->retainedPreparedSourceWorkers();
    if(!retained)throw std::logic_error("Directed source lost original worker cohort");
    const auto& memory=retained->source()->memory();const auto observed=storage->retainedOwnerObservation().authority();
    if(observed.get()!=memory.get() || observed.owner_before(memory) || memory.owner_before(observed) || retained->rootIdentity()!=storage->captureIdentity())
        throw std::logic_error("Directed source changed original worker/PMA/root authority");
}
void PooledImmutableDirectedTransport::requireCurrent()const{
    requireOriginalSource();
    const auto storage = boundary_->pooledStorage();
    std::size_t expectedRemotes = 0;
    for (const auto& owner : storage->originalOwners())
        if (owner->plan().allocator() != storage->coordinator()->plan().allocator()) ++expectedRemotes;
    if (remotes_.size() != expectedRemotes)
        throw std::logic_error("Directed transport lost an original physical owner edge");
    if (reader_geometry_) {
        if (reader_geometry_->consumers().size() != storage->geometry().parameters().consumers ||
            readers_.size() + 1 != reader_geometry_->consumers().size())
            throw std::logic_error("Shared transport lost an original compute reader edge");
        std::size_t next = 0;
        for (std::uint32_t i = 0; i < reader_geometry_->consumers().size(); ++i) {
            if (reader_geometry_->consumers()[i] == reader_geometry_->coordinator()) continue;
            if (!readers_[next] || readers_[next]->coordinate() != i ||
                readers_[next]->originalReader().device() != reader_geometry_->consumers()[i].device)
                throw std::logic_error("Shared transport changed its original ordered reader membership");
            ++next;
        }
    } else if (!readers_.empty() || storage->geometry().parameters().consumers != 1) {
        throw std::logic_error("Shared transport lost its original compute-reader declaration");
    }
    for (const auto& reader : readers_) {
        if (!reader || !reader_geometry_)
            throw std::logic_error("Shared reader transport lost its original declaration");
        reader->requireCurrent();
        if (reader->coordinate() >= reader_geometry_->consumers().size() ||
            reader_geometry_->consumers()[reader->coordinate()].device != reader->originalReader().device() ||
            reader->result()->capacityBytes() != std::max(reader_geometry_->resultMaximum(), sizeof(replicated_reader::Header)) ||
            reader->acknowledgement()->capacityBytes() != sizeof(replicated_reader::Header) ||
            reader->result()->payloadStorage() != reader_geometry_->payloadStorage(reader_geometry_->consumers()[reader->coordinate()]) ||
            reader->acknowledgement()->payloadStorage() != reader_geometry_->payloadStorage(reader_geometry_->consumers()[reader->coordinate()]))
            throw std::logic_error("Shared reader channel geometry changed");
    }
    for(const auto& remote:remotes_){remote.pin->requireCurrent();remote.returned->requireCurrent();remote.free->requireCurrent();remote.victim->requireCurrent();remote.source->requireCurrent();remote.publication->requireCurrent();remote.read->requireCurrent();remote.retirement->requireCurrent();}
}
}
