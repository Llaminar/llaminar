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
    PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> resources,Contribution kind, const PooledImmutableReplicatedReaderGeometry* readers){
    PooledImmutableCoordinatorNativePlan::requireOriginalCohort(physical,policy);
    if(!host.device.is_cpu() || host.world_rank!=physical.coordinator().world_rank)throw std::invalid_argument("Directed channel changed original host allocator");
    const auto coordinator=resource(resources,physical.coordinator());PhysicalMemoryPlanBuilder builder;
    const bool native = readers && readers->nativeCollective();
    if (native) {
        const auto count=readers->consumers().size();
        if(readers->coordinator()!=physical.coordinator() || count<2 || count>kMaximumPooledNativeParticipants)
            throw std::invalid_argument("Native pooled BOM requires exact original addressed reader coordinates");
        std::array<PhysicalMemoryResource,kMaximumPooledNativeParticipants> endpoints;
        std::array<std::uint8_t,kMaximumPooledNativeParticipants> positive{};
        std::size_t source=count;
        for(std::size_t i=0;i<count;++i){
            endpoints[i]=resource(resources,readers->consumers()[i]);
            if(readers->consumers()[i]==physical.coordinator())source=i;
        }
        for(const auto& owner:physical.owners()) {
            if(owner.allocator==physical.coordinator() || owner.allocator.device.type!=physical.coordinator().device.type)continue;
            const auto found=std::find(readers->consumers().begin(),readers->consumers().end(),owner.allocator);
            if(found==readers->consumers().end())
                throw std::invalid_argument("Shared PLE same-vendor storage owner is outside the original native communicator");
            positive[found-readers->consumers().begin()]=kind!=Contribution::Mandatory && owner.slots!=0;
        }
        const auto flags=std::span(positive.data(),count);
        const auto endpointsView=std::span(endpoints.data(),count);
        const auto actual=PooledImmutableNativeCompletionStorage::memoryBOM(host,endpointsView,
            PooledNativeCompletionLayout(count,source,flags));
        if(kind==Contribution::Growth) {
            positive.fill(0);
            const auto floor=PooledImmutableNativeCompletionStorage::memoryBOM(host,endpointsView,
                PooledNativeCompletionLayout(count,source,flags));
            for(const auto& line:actual.resources()) {
                const auto* baseline=floor.find({line.resource().world_rank,line.resource().device});
                if(!baseline)throw std::logic_error("Native metadata growth changed original endpoints");
                const auto owner=PhysicalMemoryOwner::ActivationTransportStaging;
                if(line.bytes(owner)<baseline->bytes(owner))throw std::logic_error("Native metadata shrank below original protocol floor");
                builder.add(line.resource(),owner,line.bytes(owner)-baseline->bytes(owner));
            }
        }else for(const auto& line:actual.resources())builder.add(line);
    }

    for(const auto& owner:physical.owners()){
        if (native && owner.allocator != physical.coordinator() && owner.allocator.device.type == physical.coordinator().device.type)
            continue; // No mapped payload/cursor bank exists for an original native operation.
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
PhysicalMemoryPlan PooledImmutableDirectedTransport::mandatoryBOM(const PooledImmutablePhysicalOwnerGeometry& p,const PooledImmutableCoordinatorGeometry& c,PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> resources,const PooledImmutableReplicatedReaderGeometry* readers){return contribution(p,c,host,resources,Contribution::Mandatory,readers);}
PhysicalMemoryPlan PooledImmutableDirectedTransport::completeBOM(const PooledImmutablePhysicalOwnerGeometry& p,const PooledImmutableCoordinatorGeometry& c,PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> resources,const PooledImmutableReplicatedReaderGeometry* readers){return contribution(p,c,host,resources,Contribution::Complete,readers);}
PhysicalMemoryPlan PooledImmutableDirectedTransport::slotDependentBOM(const PooledImmutablePhysicalOwnerGeometry& p,const PooledImmutableCoordinatorGeometry& c,PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> resources,const PooledImmutableReplicatedReaderGeometry* readers){return contribution(p,c,host,resources,Contribution::Growth,readers);}
std::shared_ptr<const PooledImmutableDirectedTransport> PooledImmutableDirectedTransport::create(std::shared_ptr<ImmutableLookupExecutionBoundary> boundary, std::shared_ptr<const ClusterInventory> originalInventory,
    std::optional<PooledImmutableReplicatedReaderGeometry> readers,
    std::shared_ptr<const Qwen4ExpReplicatedPLECohort> nativeCohort){
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
    // A homogeneous edge is an original native CCL operation, independent of
    // direct-peer availability. Mixed-vendor channels retain their original
    // inventory/endpoint owner. No failed native selection retries a channel.
    const auto nativeMember = [&](DeviceId device) {
        return nativeCohort && std::any_of(nativeCohort->context().devices().begin(),
            nativeCohort->context().devices().end(), [&](const auto& member) {
                return member.toLocalDeviceId() == device;
            });
    };
    for (const auto& owner : originalOwners) {
        if (owner == storage->coordinator()->plan().allocator()) continue;
        if (owner.device.type == coordinator.device().type) {
            if (!nativeMember(owner.device) || !nativeMember(coordinator.device()))
                throw std::invalid_argument("Homogeneous pooled owner requires its actual original inline native communicator");
        } else {
            const std::array mixed{owner};
            ImmutableRowTableDirectedTransportPolicy::requireOriginal(storage->coordinator()->plan().allocator(),
                mixed, originalInventory, observation);
        }
    }
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
        if (!nativeCohort || std::size_t(nativeCohort->degree())!=readers->consumers().size())
            throw std::invalid_argument("Shared homogeneous readers require their original addressed native communicator");
        for (const auto& consumer : readers->consumers())
            if (!nativeMember(consumer.device))
                throw std::invalid_argument("Shared reader is absent from its original native communicator");
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
        if (owner->plan().allocator().device.type != coordinator.device().type)
            requireRemote(storage->coordinator()->plan().allocator(),owner->plan().allocator(),owner->plan().peerAccess());
        if(permit.worker()->ownsCurrentThread())throw std::invalid_argument("Directed setup cannot occupy an original participant worker");
    }
    auto result=std::shared_ptr<PooledImmutableDirectedTransport>(new PooledImmutableDirectedTransport(std::move(boundary),std::move(originalInventory)));
    result->requireOriginalSource();result->remotes_.reserve(retained->owners().size());
    if(nativeCohort) {
        if(!readers || std::size_t(nativeCohort->degree())!=readers->consumers().size() ||
            nativeCohort->degree()<2 || std::size_t(nativeCohort->degree())>kMaximumPooledNativeParticipants)
            throw std::invalid_argument("Native pooled metadata requires exact original addressed readers");
        const auto& devices=nativeCohort->context().devices();
        PooledImmutableNativeCompletionStorage::Participants permits;
        PooledImmutableNativeCompletionStorage::Frames frames;
        std::array<std::uint8_t,kMaximumPooledNativeParticipants> positive{};
        std::size_t source=devices.size();
        for(std::size_t participant=0;participant<devices.size();++participant) {
            permits[participant].emplace(retained->source()->acquire(devices[participant].toLocalDeviceId()));
            const auto device=permits[participant]->device();
            if(device==coordinator.device()) {
                source=participant;const auto& original=storage->coordinator();const auto offset=original->plan().parameters().frame.control;
                frames[participant].emplace(PooledNativeTransferSpan(original->roots().frame,
                    offset+offsetof(adaptive_lookup::pooled::LookupFrame,interval),sizeof(adaptive_lookup::pooled::Interval)),
                    PooledNativeTransferSpan(original->roots().frame,offset+offsetof(adaptive_lookup::pooled::LookupFrame,status),sizeof(std::int32_t)),
                    static_cast<std::int32_t>(adaptive_lookup::Status::SourceFailure));
            }else for(const auto& original:storage->originalOwners()) {
                if(original->plan().allocator().device!=device)continue;
                const auto offset=original->plan().parameters().frame.control;
                frames[participant].emplace(PooledNativeTransferSpan(original->roots().frame,
                    offset+offsetof(adaptive_lookup::pooled::OwnerFrame,interval),sizeof(adaptive_lookup::pooled::Interval)),
                    PooledNativeTransferSpan(original->roots().frame,offset+offsetof(adaptive_lookup::pooled::OwnerFrame,status),sizeof(std::int32_t)),
                    static_cast<std::int32_t>(adaptive_lookup::Status::SourceFailure));
                positive[participant]=original->plan().parameters().owner.capacity()!=0;
            }
        }
        result->native_completion_=PooledImmutableNativeCompletionStorage::create(std::move(nativeCohort),
            std::move(permits),std::move(frames),PooledNativeCompletionLayout(devices.size(),source,std::span(positive.data(),devices.size())));
    }

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
        const auto pin=PooledImmutableCapturedPinEdge::bind(storage->coordinator(),owner,memory,coordinatorStream,ownerStream,result->original_inventory_,
            owner->plan().allocator().device.type == coordinator.device().type ? result->native_completion_ : nullptr);
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
            if (!result->native_completion_)
                throw std::logic_error("Shared native reader lost its original communicator metadata owner");
            const auto capacity = std::max(geometry.resultMaximum(), sizeof(replicated_reader::Header));
            auto output = PooledImmutableTransferChannel::native(result->native_completion_, coordinator.device(), permit.device(), capacity);
            auto ack = PooledImmutableTransferChannel::native(result->native_completion_, permit.device(), coordinator.device(), sizeof(replicated_reader::Header));
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
                    {}, {}, std::move(controlClaims), std::move(controls), std::move(output), std::move(ack))));
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
    const bool native = nativeCollective();
    if (!memory || reader_.memory().get() != memory.get() ||
        reader_.memory().owner_before(memory) || memory.owner_before(reader_.memory()) ||
        source_.device() == reader_.device() ||
        (native && (result_ || acknowledgement_ || !native_acknowledgement_.nativeCollective() ||
            native_result_.nativeStorage() != native_acknowledgement_.nativeStorage())) ||
        (!native && (!result_ || !acknowledgement_ || bool(native_acknowledgement_))))
        throw std::logic_error("Shared PLE reader lost its original unique transport/PMA owner");
    if (native) {
        native_result_.nativeStorage()->requireCurrent();
        if (native_result_.endpointDevice(CapturedTransferEndpoint::Producer) != source_.device() ||
            native_result_.endpointDevice(CapturedTransferEndpoint::Consumer) != reader_.device() ||
            native_acknowledgement_.endpointDevice(CapturedTransferEndpoint::Producer) != reader_.device() ||
            native_acknowledgement_.endpointDevice(CapturedTransferEndpoint::Consumer) != source_.device())
            throw std::logic_error("Native shared PLE reader changed original communicator coordinates");
    } else if (result_->endpointDevice(CapturedTransferEndpoint::Producer) != source_.device() ||
        result_->endpointDevice(CapturedTransferEndpoint::Consumer) != reader_.device() ||
        acknowledgement_->endpointDevice(CapturedTransferEndpoint::Producer) != reader_.device() ||
        acknowledgement_->endpointDevice(CapturedTransferEndpoint::Consumer) != source_.device()) {
        throw std::logic_error("Shared PLE reader changed its original captured endpoints");
    }
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
        if (const auto& original = remote.pin->channel_.captured()) channels.push_back(original);
        if (const auto& original = remote.returned->channel_.captured()) channels.push_back(original);
    }
    if (!edges || *edges != CancellationEdges::StorageOwners) for (const auto& reader : readers_) {
        if (reader->result()) channels.push_back(reader->result());
        if (reader->acknowledgement()) channels.push_back(reader->acknowledgement());
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
void PooledImmutableDirectedTransport::retireOriginalRuntime() const noexcept
{
    if (native_completion_) native_completion_->retireOriginalRuntime();
}
std::function<void()> PooledImmutableDirectedTransport::nativeFailureSignal() const
{
    if (!native_completion_) return {};
    const auto original = native_completion_;
    original->requireCurrent();
    return [original] {
        if (!original->cohort()->context().signalNativeCompletionAbort())
            throw std::logic_error("Original pooled native communicator could not publish its absorbing abort");
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
            (reader->nativeCollective() &&
                (reader->nativeResult().nativeStorage() != native_completion_ ||
                 reader->nativeResult().maximumBytes() != std::max(reader_geometry_->resultMaximum(), sizeof(replicated_reader::Header)) ||
                 reader->nativeAcknowledgement().maximumBytes() != sizeof(replicated_reader::Header))) ||
            (!reader->nativeCollective() &&
                (reader->result()->capacityBytes() != std::max(reader_geometry_->resultMaximum(), sizeof(replicated_reader::Header)) ||
                 reader->acknowledgement()->capacityBytes() != sizeof(replicated_reader::Header) ||
                 reader->result()->payloadStorage() != reader_geometry_->payloadStorage(reader_geometry_->consumers()[reader->coordinate()]) ||
                 reader->acknowledgement()->payloadStorage() != reader_geometry_->payloadStorage(reader_geometry_->consumers()[reader->coordinate()]))))
            throw std::logic_error("Shared reader channel geometry changed");
    }
    for(const auto& remote:remotes_){remote.pin->requireCurrent();remote.returned->requireCurrent();remote.free->requireCurrent();remote.victim->requireCurrent();remote.source->requireCurrent();remote.publication->requireCurrent();remote.read->requireCurrent();remote.retirement->requireCurrent();}
}
}
