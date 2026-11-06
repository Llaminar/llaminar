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
#include "loaders/ModelLoader.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotPreparedStorage.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotOperands.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotTransportKernels.h"
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
    PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> resources,Contribution kind, const PooledImmutableReplicatedReaderGeometry* readers,PooledNativeReaderMessageProfile profile){
    if(profile!=PooledNativeReaderMessageProfile::ReplicatedFP32 && profile!=PooledNativeReaderMessageProfile::CompactNative)
        throw std::invalid_argument("Unknown pooled native reader BOM profile");
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
            PooledNativeCompletionLayout(count,source,flags,profile));
        if(kind==Contribution::Growth) {
            positive.fill(0);
            const auto floor=PooledImmutableNativeCompletionStorage::memoryBOM(host,endpointsView,
                PooledNativeCompletionLayout(count,source,flags,profile));
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
PhysicalMemoryPlan PooledImmutableDirectedTransport::mandatoryBOM(const PooledImmutablePhysicalOwnerGeometry& p,const PooledImmutableCoordinatorGeometry& c,PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> resources,const PooledImmutableReplicatedReaderGeometry* readers,PooledNativeReaderMessageProfile profile){return contribution(p,c,host,resources,Contribution::Mandatory,readers,profile);}
PhysicalMemoryPlan PooledImmutableDirectedTransport::completeBOM(const PooledImmutablePhysicalOwnerGeometry& p,const PooledImmutableCoordinatorGeometry& c,PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> resources,const PooledImmutableReplicatedReaderGeometry* readers,PooledNativeReaderMessageProfile profile){return contribution(p,c,host,resources,Contribution::Complete,readers,profile);}
PhysicalMemoryPlan PooledImmutableDirectedTransport::slotDependentBOM(const PooledImmutablePhysicalOwnerGeometry& p,const PooledImmutableCoordinatorGeometry& c,PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> resources,const PooledImmutableReplicatedReaderGeometry* readers,PooledNativeReaderMessageProfile profile){return contribution(p,c,host,resources,Contribution::Growth,readers,profile);}
std::shared_ptr<const PooledImmutableDirectedTransport> PooledImmutableDirectedTransport::create(std::shared_ptr<ImmutableLookupExecutionBoundary> boundary, std::shared_ptr<const ClusterInventory> originalInventory,
    std::optional<PooledImmutableReplicatedReaderGeometry> readers,
    std::shared_ptr<const Qwen4ExpReplicatedPLECohort> nativeCohort,
    std::optional<PooledImmutableLocalHotTransportGeometry> localHot){
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
            (localHot ? localHot->maximumPoolOriginalRows() : readers->maximumOriginal()) != storage->geometry().parameters().coordinator.policy.maximumOriginal() ||
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
    if(localHot){
        if(localHot->coordinator()!=storage->coordinator()->plan().allocator() ||
            localHot->consumers().size()!=storage->geometry().parameters().consumers ||
            localHot->maximumPoolOriginalRows()!=storage->geometry().parameters().coordinator.policy.maximumOriginal() ||
            localHot->sourceExtent().rows()!=storage->geometry().parameters().owner.owner.sourceRows() ||
            localHot->sourceExtent().rowBytes()!=storage->geometry().parameters().owner.row_bytes ||
            ggufToTensorType(localHot->sourceExtent().format())!=storage->geometry().parameters().format)
            throw std::invalid_argument("Compact transport changed its actual source/union geometry");
        if(bool(readers)!=bool(localHot->originalReaders()) ||
            (readers && (readers->coordinator()!=localHot->coordinator() ||
                !std::equal(readers->consumers().begin(),readers->consumers().end(),localHot->consumers().begin(),localHot->consumers().end()))))
            throw std::invalid_argument("Compact transport lost its exact original reader roster");
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
    result->local_hot_geometry_=std::move(localHot);
    const auto profile=result->local_hot_geometry_?PooledNativeReaderMessageProfile::CompactNative:PooledNativeReaderMessageProfile::ReplicatedFP32;
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
            std::move(permits),std::move(frames),PooledNativeCompletionLayout(devices.size(),source,std::span(positive.data(),devices.size()),profile));
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
            const auto capacity = result->local_hot_geometry_ ?
                std::max(result->local_hot_geometry_->resultCapacityBytes(),sizeof(local_hot_transport::ResultHeader)) :
                std::max(geometry.resultMaximum(), sizeof(replicated_reader::Header));
            auto output = PooledImmutableTransferChannel::native(result->native_completion_, coordinator.device(), permit.device(), capacity);
            auto ack = PooledImmutableTransferChannel::native(result->native_completion_, permit.device(), coordinator.device(), result->local_hot_geometry_?
                std::max(result->local_hot_geometry_->keyCapacityBytes(),sizeof(local_hot_transport::RequestHeader)):sizeof(replicated_reader::Header));
            std::array<PhysicalMemoryAllocationLease,2> controlClaims;
            std::array<std::shared_ptr<DeviceTransferBuffer>,2> controls;
            const std::array endpoints{coordinator.device(), permit.device()};
            const std::array workers{coordinator.worker(), permit.worker()};
            const auto controlBytes=result->local_hot_geometry_?sizeof(local_hot_transport::Control):sizeof(replicated_reader::Control);
            for (std::size_t endpoint = 0; endpoint < endpoints.size(); ++endpoint) {
                controlClaims[endpoint] = memory->claimNewAllocation(endpoints[endpoint],
                    PhysicalMemoryOwner::ActivationTransportStaging, controlBytes);
                workers[endpoint]->submitAndWait([&] {
                    controls[endpoint] = transfer.allocateDeviceTransferBuffer(
                        controlBytes, endpoints[endpoint]);
                });
            }
            result->readers_.push_back(std::shared_ptr<const PooledImmutableReplicatedReaderTransport>(
                new PooledImmutableReplicatedReaderTransport(coordinate, coordinator, std::move(permit),
                    {}, {}, std::move(controlClaims), std::move(controls), std::move(output), std::move(ack), profile)));
        }
    }
    if(result->local_hot_geometry_){
        const auto& geometry=*result->local_hot_geometry_;
        const std::array<std::size_t,5> bytes{sizeof(local_hot_transport::Control),sizeof(local_hot_transport::UnionControl),
            geometry.mergedKeyBytes(),geometry.incomingKeyBytes(),geometry.outgoingNativeBytes()};
        for(std::size_t i=0;i<bytes.size();++i){
            if(!bytes[i])continue;
            result->local_hot_source_claims_[i]=memory->claimNewAllocation(coordinator.device(),PhysicalMemoryOwner::ActivationTransportStaging,bytes[i]);
            coordinator.worker()->submitAndWait([&]{result->local_hot_source_roots_[i]=
                TransferEngine::instance().allocateDeviceTransferBuffer(bytes[i],coordinator.device());});
        }
    }
    if(result->local_hot_geometry_){
        result->local_hot_admission_.reserve(result->readers_.size());
        const auto& parameters=storage->geometry().parameters();
        for(const auto& reader:result->readers_){
            auto key=0x51464e4c3141444dULL;key=(key^parameters.owner.table)*1099511628211ULL;
            key=(key^parameters.owner.source_storage_generation)*1099511628211ULL;key=(key^reader->coordinate())*1099511628211ULL;key|=1;
            auto send=reader->nativeResult().bind(CapturedTransferEndpoint::Producer,{key,sizeof(local_hot_transport::SourceAdmissionHeader)},result->localHotAdmission(coordinator.device()));
            auto receive=reader->nativeResult().bind(CapturedTransferEndpoint::Consumer,{key,sizeof(local_hot_transport::SourceAdmissionHeader)},result->localHotAdmission(reader->originalReader().device()));
            coordinator.worker()->submitAndWait([&]{send.prepare(coordinator.worker()->defaultStream());});
            reader->originalReader().worker()->submitAndWait([&]{receive.prepare(reader->originalReader().worker()->defaultStream());});
            result->local_hot_admission_.push_back({reader->coordinate(),std::move(send),std::move(receive)});
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
    if(profile_!=PooledNativeReaderMessageProfile::ReplicatedFP32)
        throw std::logic_error("Compact request metadata is not an expanded reader control");
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
    const auto controlBytes=profile_==PooledNativeReaderMessageProfile::CompactNative?sizeof(local_hot_transport::Control):sizeof(replicated_reader::Control);
    for (std::size_t i = 0; i < controls_.size(); ++i) {
        const auto device = i ? reader_.device() : source_.device();
        if (!controls_[i] || !controls_[i]->isBound() || controls_[i]->device() != device ||
            controls_[i]->sizeBytes() != controlBytes ||
            reinterpret_cast<std::uintptr_t>(controls_[i]->deviceData()) % alignof(replicated_reader::Control) != 0 ||
            control_claims_[i].bytes() != controlBytes)
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
    const auto resultCapacity=local_hot_geometry_?std::max(local_hot_geometry_->resultCapacityBytes(),sizeof(local_hot_transport::ResultHeader)):
        (reader_geometry_?std::max(reader_geometry_->resultMaximum(),sizeof(replicated_reader::Header)):0);
    const auto requestCapacity=local_hot_geometry_?std::max(local_hot_geometry_->keyCapacityBytes(),sizeof(local_hot_transport::RequestHeader)):sizeof(replicated_reader::Header);
    if(local_hot_geometry_){
        const auto& g=*local_hot_geometry_;
        const std::array<std::size_t,5> bytes{sizeof(local_hot_transport::Control),sizeof(local_hot_transport::UnionControl),g.mergedKeyBytes(),g.incomingKeyBytes(),g.outgoingNativeBytes()};
        for(std::size_t i=0;i<bytes.size();++i){
            const auto& root=local_hot_source_roots_[i];
            if(bool(root)!=bool(bytes[i]) || (root && (!root->isBound() || root->device()!=g.coordinator().device ||
                root->sizeBytes()!=bytes[i] || local_hot_source_claims_[i].bytes()!=bytes[i])))
                throw std::logic_error("Compact transport lost exact source protocol/scratch roots");
        }
    }
    for (const auto& reader : readers_) {
        if (!reader || !reader_geometry_)
            throw std::logic_error("Shared reader transport lost its original declaration");
        reader->requireCurrent();
        if (reader->coordinate() >= reader_geometry_->consumers().size() ||
            reader_geometry_->consumers()[reader->coordinate()].device != reader->originalReader().device() ||
            (reader->nativeCollective() &&
                (reader->nativeResult().nativeStorage() != native_completion_ ||
                 reader->nativeResult().maximumBytes() != resultCapacity ||
                 reader->nativeAcknowledgement().maximumBytes() != requestCapacity)) ||
            (!reader->nativeCollective() &&
                (reader->result()->capacityBytes() != resultCapacity ||
                 reader->acknowledgement()->capacityBytes() != requestCapacity ||
                 reader->result()->payloadStorage() != reader_geometry_->payloadStorage(reader_geometry_->consumers()[reader->coordinate()]) ||
                 reader->acknowledgement()->payloadStorage() != reader_geometry_->payloadStorage(reader_geometry_->consumers()[reader->coordinate()]))))
            throw std::logic_error("Shared reader channel geometry changed");
    }
    for(const auto& remote:remotes_){remote.pin->requireCurrent();remote.returned->requireCurrent();remote.free->requireCurrent();remote.victim->requireCurrent();remote.source->requireCurrent();remote.publication->requireCurrent();remote.read->requireCurrent();remote.retirement->requireCurrent();}
}
const std::shared_ptr<DeviceTransferBuffer>& PooledImmutableDirectedTransport::localHotControlRoot(DeviceId participant,std::uint32_t consumer)const
{
    requireCurrent();
    if(!local_hot_geometry_ || consumer>=local_hot_geometry_->consumers().size())
        throw std::invalid_argument("Compact control requires its actual declared consumer");
    const auto source=local_hot_geometry_->coordinator().device;
    if(local_hot_geometry_->consumers()[consumer].device==source){
        if(participant!=source)throw std::invalid_argument("Source-self compact control cannot be read from a foreign GPU");
        return local_hot_source_roots_[0];
    }
    for(const auto& reader:readers_)if(reader->coordinate()==consumer){
        if(participant==source)return reader->controls_[0];
        if(participant==reader->originalReader().device())return reader->controls_[1];
        throw std::invalid_argument("Compact control is not local to this original endpoint");
    }
    throw std::logic_error("Compact consumer lost its original transport endpoint");
}
PooledNativeTransferSpan PooledImmutableDirectedTransport::localHotControl(DeviceId participant,std::uint32_t consumer)const
{return {localHotControlRoot(participant,consumer),0,sizeof(local_hot_transport::Control)};}
PooledNativeTransferSpan PooledImmutableDirectedTransport::localHotOriginalRequest(DeviceId participant)const
{
    requireCurrent();if(!local_hot_geometry_)throw std::logic_error("Shared-only transport has no local compact request");
    for(std::uint32_t i=0;i<local_hot_geometry_->consumers().size();++i)
        if(local_hot_geometry_->consumers()[i].device==participant)
            return {localHotControlRoot(participant,i),offsetof(local_hot_transport::Control,request_out),sizeof(local_hot_transport::RequestHeader)};
    throw std::invalid_argument("Compact request requires its actual consumer device");
}
PooledNativeTransferSpan PooledImmutableDirectedTransport::localHotAdmission(DeviceId participant)const
{
    requireCurrent();if(!local_hot_geometry_)throw std::logic_error("Shared-only transport has no local compact admission");
    for(std::uint32_t i=0;i<local_hot_geometry_->consumers().size();++i)
        if(local_hot_geometry_->consumers()[i].device==participant)
            return {localHotControlRoot(participant,i),offsetof(local_hot_transport::Control,original_admission),sizeof(local_hot_transport::SourceAdmissionHeader)};
    throw std::invalid_argument("Compact admission requires its actual consumer device");
}
PooledNativeTransferSpan PooledImmutableDirectedTransport::localHotUnion()const
{
    requireCurrent();if(!local_hot_geometry_)throw std::logic_error("Shared-only transport has no compact union");
    return {local_hot_source_roots_[1],0,sizeof(local_hot_transport::UnionControl)};
}
PooledNativeTransferSpan PooledImmutableDirectedTransport::localHotSourceScratch(LocalHotSourceScratch role)const
{
    requireCurrent();if(!local_hot_geometry_)throw std::logic_error("Shared-only transport has no compact source scratch");
    const auto index=role==LocalHotSourceScratch::MergedKeys?2u:role==LocalHotSourceScratch::IncomingKeys?3u:
        role==LocalHotSourceScratch::OutgoingRows?4u:5u;
    if(index==5 || !local_hot_source_roots_[index])throw std::invalid_argument("Compact source scratch role is absent or invalid");
    const auto& root=local_hot_source_roots_[index];return {root,0,root->sizeBytes()};
}

PooledImmutableLocalHotPoolQuery PooledImmutableDirectedTransport::localHotPoolQuery(
    std::shared_ptr<PooledImmutableLocalHotPreparedStorage> sourceLocal,PooledImmutableLocalHotQueryBound bound,
    adaptive_lookup::pooled::LookupPurpose purpose)const
{
    requireCurrent();const auto source=boundary_->pooledStorage();
    if(!local_hot_geometry_ || !sourceLocal || sourceLocal->originalSource()!=source ||
        sourceLocal->geometry().allocator()!=local_hot_geometry_->coordinator() ||
        sourceLocal->geometry().maximumOriginalIds()!=local_hot_geometry_->maximumPoolUniqueRows())
        throw std::invalid_argument("Compact Begin requires the actual source participant's original local query");
    sourceLocal->requireAuthority(source->retainedOwnerObservation().authority());
    if(bound.consumers()!=local_hot_geometry_->consumers().size() ||
        bound.canonicalOriginal()>sourceLocal->geometry().maximumOriginalIds() ||
        bound.maximumOriginal()>local_hot_geometry_->maximumPoolOriginalRows())
        throw std::invalid_argument("Compact Begin launch bound exceeds original roots");
    for(std::uint32_t i=0;i<bound.consumers();++i)
        if(!bound.consumerMaximum(i) || bound.consumerMaximum(i)>std::min(bound.canonicalOriginal(),local_hot_geometry_->maximumPoolUniqueRows()))
            throw std::invalid_argument("Compact Begin launch bound changed original consumer subset");
    const auto& f=sourceLocal->geometry().frameOffsets();
    auto ids=local_hot_geometry_->consumers().size()==1?
        PooledNativeTransferSpan(sourceLocal->roots().frame,f.miss_keys,local_hot_geometry_->keyCapacityBytes()):
        localHotSourceScratch(LocalHotSourceScratch::MergedKeys);
    ids=ids.subspan(0,std::size_t{bound.maximumOriginal()}*sizeof(std::uint64_t));
    const auto& sourceParameters=source->geometry().parameters();
    PooledNativeTransferSpan admission(source->roots().native,
        sourceParameters.offsets.protocol+offsetof(ImmutableLookupDeviceState,admissionGeneration),sizeof(std::uint64_t));
    return {shared_from_this(),std::move(sourceLocal),std::move(ids),
        PooledNativeTransferSpan(local_hot_source_roots_[1],offsetof(local_hot_transport::UnionControl,pool_original_ids),sizeof(std::int32_t)),
        std::move(admission),std::move(bound),purpose};
}
PooledImmutableLocalHotPoolQuery::PooledImmutableLocalHotPoolQuery(
    std::shared_ptr<const PooledImmutableDirectedTransport> transport,std::shared_ptr<PooledImmutableLocalHotPreparedStorage> local,
    PooledNativeTransferSpan ids,PooledNativeTransferSpan count,PooledNativeTransferSpan admission,PooledImmutableLocalHotQueryBound bound,
    adaptive_lookup::pooled::LookupPurpose purpose)
    :transport_(std::move(transport)),source_local_(std::move(local)),ids_(std::move(ids)),count_(std::move(count)),
     admission_(std::move(admission)),bound_(std::move(bound)),capacity_(bound_.maximumOriginal()),purpose_(purpose)
{
    if((purpose_!=adaptive_lookup::pooled::LookupPurpose::CurrentDemand && purpose_!=adaptive_lookup::pooled::LookupPurpose::FutureLookupOnly) ||
        !transport_ || !source_local_ || !capacity_ || capacity_>static_cast<std::uint32_t>(INT32_MAX) ||
        ids_.bytes()!=std::size_t{capacity_}*sizeof(std::uint64_t) || count_.bytes()!=sizeof(std::int32_t) || admission_.bytes()!=sizeof(std::uint64_t))
        throw std::invalid_argument("Compact Begin lost its exact original producer extents");
}
void PooledImmutableLocalHotPoolQuery::seal()
{
    const auto device=source_local_->geometry().allocator().device;
    transport_->requireCurrent();ids_.require(device,8);count_.require(device,4);admission_.require(device,8);
    const auto* ids=ids_.pointer(device,8);const auto* count=static_cast<const std::int32_t*>(count_.pointer(device,4));
    const auto* admission=static_cast<const std::uint64_t*>(admission_.pointer(device,8));
    if(sealed_ids_ && (sealed_ids_!=ids || sealed_count_!=count || sealed_admission_!=admission))
        throw std::logic_error("Compact Begin original producer addresses changed after seal");
    sealed_ids_=ids;sealed_count_=count;sealed_admission_=admission;
}
void PooledImmutableLocalHotPoolQuery::requireDeclaration(const PooledImmutableCoordinatorNativePlan& plan)const
{
    transport_->requireCurrent();const auto source=transport_->sourceBoundary()->pooledStorage();
    const auto device=source_local_->geometry().allocator().device;
    if(source_local_->originalSource()!=source || source->coordinator()->plan().parameters().identity!=plan.parameters().identity ||
        plan.allocator().device!=device || capacity_>plan.parameters().policy.maximumOriginal() ||
        bound_.canonicalOriginal()>source_local_->geometry().maximumOriginalIds() ||
        ids_.bytes()!=std::size_t{capacity_}*sizeof(std::uint64_t) || count_.bytes()!=sizeof(std::int32_t) ||
        admission_.bytes()!=sizeof(std::uint64_t))
        throw std::logic_error("Compact Begin changed its original coordinator/source/query declaration");
    source_local_->captureIdentity();
}
void PooledImmutableLocalHotPoolQuery::require(const PooledImmutableCoordinatorNativePlan& plan)const
{
    requireDeclaration(plan);const auto device=source_local_->geometry().allocator().device;
    if(!sealed_ids_ || sealed_ids_!=ids_.pointer(device,8) || sealed_count_!=count_.pointer(device,4) ||
        sealed_admission_!=admission_.pointer(device,8))
        throw std::logic_error("Compact Begin lost its sealed original query producer addresses");
}
std::uint64_t PooledImmutableLocalHotPoolQuery::declarationIdentity()const
{
    transport_->requireCurrent();if(sealed_ids_)require(transport_->sourceBoundary()->pooledStorage()->coordinator()->plan());std::uint64_t h=1469598103934665603ULL;
    const auto mix=[&](std::uint64_t x){h=(h^x)*1099511628211ULL;};
    mix(2);const auto& rows=bound_.tokenIdentity();mix(reinterpret_cast<std::uintptr_t>(rows.count));
    mix(rows.count_capacity);mix(rows.first_physical_row);mix(rows.physical_rows);mix(rows.repeats_per_row);mix(rows.schema_version);
    mix(bound_.canonicalOriginal());mix(bound_.consumers());for(std::uint32_t i=0;i<bound_.consumers();++i)mix(bound_.consumerMaximum(i));
    mix(static_cast<std::uint32_t>(purpose_));mix(source_local_->captureIdentity());mix(ids_.identity());mix(count_.identity());mix(admission_.identity());mix(capacity_);return h?h:1;
}
std::uint64_t PooledImmutableLocalHotPoolQuery::captureIdentity()const
{
    require(transport_->sourceBoundary()->pooledStorage()->coordinator()->plan());
    auto h=declarationIdentity();for(const auto pointer:{sealed_ids_,static_cast<const void*>(sealed_count_),static_cast<const void*>(sealed_admission_)})
        h=(h^reinterpret_cast<std::uintptr_t>(pointer))*1099511628211ULL;
    return h?h:1;
}
pooled_native::CoordinatorQuery PooledImmutableLocalHotPoolQuery::native()const
{
    require(transport_->sourceBoundary()->pooledStorage()->coordinator()->plan());
    return {{DeviceRowRange::deviceCounted(static_cast<int>(capacity_),sealed_count_),pooled_native::OriginalIdWidth::UInt64,purpose_},sealed_ids_,sealed_admission_,
        static_cast<const local_hot_transport::UnionControl*>(transport_->localHotUnion().pointer(source_local_->geometry().allocator().device,alignof(local_hot_transport::UnionControl))),
        static_cast<const local_hot_transport::QueryIdentity*>(PooledNativeTransferSpan(source_local_->roots().frame,
            source_local_->geometry().frameOffsets().identity,sizeof(local_hot_transport::QueryIdentity)).pointer(
                source_local_->geometry().allocator().device,alignof(local_hot_transport::QueryIdentity)))};
}

PooledImmutableLocalHotNativePack PooledImmutableDirectedTransport::localHotNativePack(
    std::shared_ptr<PooledImmutableLocalHotPreparedStorage> consumer)const
{
    requireCurrent();const auto source=boundary_->pooledStorage();
    if(!local_hot_geometry_ || !consumer || consumer->originalSource()!=source ||
        consumer->geometry().maximumOriginalIds()!=local_hot_geometry_->maximumPoolUniqueRows())
        throw std::invalid_argument("Compact native pack requires its actual original local consumer");
    consumer->requireAuthority(source->retainedOwnerObservation().authority());
    const auto device=local_hot_geometry_->coordinator().device;
    std::uint32_t coordinate=static_cast<std::uint32_t>(local_hot_geometry_->consumers().size()),remoteIndex=0;
    for(std::uint32_t i=0;i<local_hot_geometry_->consumers().size();++i){
        if(local_hot_geometry_->consumers()[i]==consumer->geometry().allocator()){coordinate=i;break;}
        if(local_hot_geometry_->consumers()[i]!=local_hot_geometry_->coordinator())++remoteIndex;
    }
    if(coordinate==local_hot_geometry_->consumers().size())throw std::invalid_argument("Native pack consumer is outside the actual transport roster");
    const auto control=localHotControlRoot(device,coordinate);
    const auto capacity=local_hot_geometry_->resultCapacityBytes();
    const bool self=consumer->geometry().allocator()==local_hot_geometry_->coordinator();
    std::array<PooledNativeTransferSpan,5> spans{
        PooledNativeTransferSpan(control,self?offsetof(local_hot_transport::Control,request_out):offsetof(local_hot_transport::Control,request_in),sizeof(local_hot_transport::RequestHeader)),
        self?PooledNativeTransferSpan(consumer->roots().frame,consumer->geometry().frameOffsets().result_header,sizeof(local_hot_transport::ResultHeader)):
            PooledNativeTransferSpan(control,offsetof(local_hot_transport::Control,result_out),sizeof(local_hot_transport::ResultHeader)),
        self?PooledNativeTransferSpan(consumer->roots().frame,consumer->geometry().frameOffsets().response,capacity):
            PooledNativeTransferSpan(local_hot_source_roots_[4],std::size_t{remoteIndex}*capacity,capacity),
        self?PooledNativeTransferSpan(consumer->roots().frame,consumer->geometry().frameOffsets().received_payload_bytes,sizeof(std::uint64_t)):
            PooledNativeTransferSpan(control,offsetof(local_hot_transport::Control,result_send_bytes),sizeof(std::uint64_t)),
        localHotUnion()};
    std::optional<PooledNativeTransferSpan> headerBytes;
    if(self)headerBytes.emplace(consumer->roots().frame,consumer->geometry().frameOffsets().received_header_bytes,sizeof(std::uint64_t));
    return {shared_from_this(),std::move(consumer),coordinate,std::move(spans),std::move(headerBytes)};
}
PooledImmutableLocalHotNativePack::PooledImmutableLocalHotNativePack(
    std::shared_ptr<const PooledImmutableDirectedTransport> transport,std::shared_ptr<PooledImmutableLocalHotPreparedStorage> consumer,
    std::uint32_t coordinate,std::array<PooledNativeTransferSpan,5> spans,std::optional<PooledNativeTransferSpan> header)
    :transport_(std::move(transport)),consumer_(std::move(consumer)),coordinate_(coordinate),spans_(std::move(spans)),local_header_bytes_(std::move(header)){}
void PooledImmutableLocalHotNativePack::seal()
{
    transport_->requireCurrent();const auto device=transport_->localHotGeometry()->coordinator().device;
    const std::array<std::size_t,5> alignment{alignof(local_hot_transport::RequestHeader),alignof(local_hot_transport::ResultHeader),1,8,alignof(local_hot_transport::UnionControl)};
    for(std::size_t i=0;i<spans_.size();++i){
        spans_[i].require(device,alignment[i]);auto* pointer=spans_[i].pointer(device,alignment[i]);
        if(sealed_[i] && sealed_[i]!=pointer)throw std::logic_error("Native pack original producer changed after seal");
        sealed_[i]=pointer;
    }
    if(local_header_bytes_){
        auto* pointer=local_header_bytes_->pointer(device,8);
        if(sealed_header_bytes_ && sealed_header_bytes_!=pointer)throw std::logic_error("Native pack original header publication changed after seal");
        sealed_header_bytes_=pointer;
    }
}
void PooledImmutableLocalHotNativePack::require(const PooledImmutableSourcePreparedStorage& source)const
{
    transport_->requireCurrent();const auto device=transport_->localHotGeometry()->coordinator().device;
    if(transport_->sourceBoundary()->pooledStorage().get()!=&source || consumer_->originalSource().get()!=&source ||
        coordinate_>=transport_->localHotGeometry()->consumers().size() ||
        transport_->localHotGeometry()->consumers()[coordinate_]!=consumer_->geometry().allocator())
        throw std::logic_error("Native pack changed its actual original source/consumer");
    consumer_->captureIdentity();
    for(std::size_t i=0;i<spans_.size();++i)if(!sealed_[i] || sealed_[i]!=spans_[i].pointer(device))
        throw std::logic_error("Native pack changed its sealed original physical root");
    if(local_header_bytes_ && sealed_header_bytes_!=local_header_bytes_->pointer(device,8))
        throw std::logic_error("Native pack changed its source-self terminal publication");
}
std::uint64_t PooledImmutableLocalHotNativePack::declarationIdentity()const
{
    transport_->requireCurrent();if(sealed_[0])require(*transport_->sourceBoundary()->pooledStorage());auto h=consumer_->captureIdentity();
    for(const auto& span:spans_)h=(h^span.identity())*1099511628211ULL;
    if(local_header_bytes_)h=(h^local_header_bytes_->identity())*1099511628211ULL;
    h=(h^coordinate_)*1099511628211ULL;return h?h:1;
}
std::uint64_t PooledImmutableLocalHotNativePack::captureIdentity()const
{
    require(*transport_->sourceBoundary()->pooledStorage());auto h=declarationIdentity();
    for(const auto* pointer:sealed_)h=(h^reinterpret_cast<std::uintptr_t>(pointer))*1099511628211ULL;
    h=(h^reinterpret_cast<std::uintptr_t>(sealed_header_bytes_))*1099511628211ULL;return h?h:1;
}
local_hot_transport::NativePackDescriptor PooledImmutableLocalHotNativePack::native()const
{
    require(*transport_->sourceBoundary()->pooledStorage());const auto& geometry=*transport_->localHotGeometry();
    return {static_cast<const local_hot_transport::UnionControl*>(sealed_[4]),static_cast<const local_hot_transport::RequestHeader*>(sealed_[0]),
        static_cast<local_hot_transport::ResultHeader*>(sealed_[1]),static_cast<std::byte*>(sealed_[2]),
        static_cast<std::uint64_t*>(sealed_[3]),static_cast<std::uint64_t*>(sealed_header_bytes_),coordinate_,
        static_cast<std::uint32_t>(geometry.consumers().size()),geometry.maximumPoolUniqueRows(),
        static_cast<std::uint32_t>(geometry.sourceExtent().format()),consumer_->consumerStorageIdentity(),geometry.resultCapacityBytes()};
}

void PooledImmutableDirectedTransport::enqueueLocalHotAdmission(
    const std::shared_ptr<PooledImmutableLocalHotPreparedStorage>& local,
    const std::shared_ptr<const IPooledImmutableLocalHotKernels>& code,
    const DeviceGenerationControlBinding& control,void* stream)const
{
    requireCurrent();
    if(isGraphCaptureActive() || !local_hot_geometry_ || !local || !code || !stream || reinterpret_cast<std::uintptr_t>(stream)<=2 ||
        local->originalSource()!=boundary_->pooledStorage())
        throw std::invalid_argument("Local-hot request admission requires its original outside-capture source boundary");
    const auto allocator=local->geometry().allocator();const auto device=allocator.device;
    const auto found=std::find(local_hot_geometry_->consumers().begin(),local_hot_geometry_->consumers().end(),allocator);
    if(found==local_hot_geometry_->consumers().end() || code->backend()!=device.type || !code->codeIdentity())
        throw std::invalid_argument("Local-hot admission changed its actual consumer/native code");
    const auto coordinate=static_cast<std::uint32_t>(found-local_hot_geometry_->consumers().begin());
    const auto memory=boundary_->pooledStorage()->retainedPreparedSourceWorkers()->source()->memory();local->requireAuthority(memory);
    const auto original=GPUDeviceContextPool::instance().retainExistingPreparedContext(device,memory);
    if(!original.worker()->ownsCurrentThread())throw std::invalid_argument("Local-hot admission must execute on its original prepared worker");
    control.requireFor(device);local->acquireSetup(stream);
    local_hot_transport::MetadataArgs args{};args.device=device;args.first_error=control.error();args.healthy=control.healthy();
    args.reporting=static_cast<local_hot::Counters*>(PooledNativeTransferSpan(local->roots().frame,
        local->geometry().frameOffsets().counters,sizeof(local_hot::Counters)).pointer(device,alignof(local_hot::Counters)));
    const auto& p=boundary_->pooledStorage()->geometry().parameters();args.source.table=p.owner.table;args.source.source_storage_generation=p.owner.source_storage_generation;
    const auto header=localHotAdmission(device);
    const auto bank=localHotControlRoot(device,coordinate);
    auto* count=static_cast<std::uint64_t*>(PooledNativeTransferSpan(bank,offsetof(local_hot_transport::Control,admission_receive_bytes),8).pointer(device,8));
    const auto dispatch=[&](local_hot_transport::MetadataAction action){
        if(device.is_cuda()){
#if defined(HAVE_CUDA)
            local_hot_transport::enqueueCUDAPooledLocalHotTransport(action,args,stream);
#else
            throw std::logic_error("Local-hot request admission requires compiled CUDA");
#endif
        }else{
#if defined(HAVE_ROCM)
            local_hot_transport::enqueueROCmPooledLocalHotTransport(action,args,stream);
#else
            throw std::logic_error("Local-hot request admission requires compiled HIP");
#endif
        }
    };
    if(allocator==local_hot_geometry_->coordinator()){
        const auto source=boundary_->pooledStorage();source->acquireSetup(stream);
        args.source.original_admission=static_cast<const std::uint64_t*>(PooledNativeTransferSpan(source->roots().native,
            p.offsets.protocol+offsetof(ImmutableLookupDeviceState,admissionGeneration),8).pointer(device,8));
        args.source.original_invocation=static_cast<const std::uint64_t*>(PooledNativeTransferSpan(source->roots().native,p.offsets.invocation,8).pointer(device,8));
        args.source.output=static_cast<local_hot_transport::SourceAdmissionHeader*>(header.pointer(device,8));args.fixed_header_bytes=count;
        dispatch(local_hot_transport::MetadataAction::PublishSourceAdmission);
        for(const auto& edge:local_hot_admission_)edge.source.enqueue(stream);
    }else{
        const auto edge=std::find_if(local_hot_admission_.begin(),local_hot_admission_.end(),[&](const auto& value){return value.coordinate==coordinate;});
        if(edge==local_hot_admission_.end())throw std::logic_error("Local-hot admission lost its original native header endpoint");
        edge->reader.enqueue(stream);args.admission=static_cast<const local_hot_transport::SourceAdmissionHeader*>(header.pointer(device,8));args.payload_bytes=count;
        const auto reader=std::find_if(readers_.begin(),readers_.end(),[&](const auto& value){return value->coordinate()==coordinate;});
        if(reader==readers_.end())throw std::logic_error("Local-hot admission lost its actual reader");
        args.native_fixed_header=(*reader)->nativeResult().nativeStorage()->authority(device).binding();
        dispatch(local_hot_transport::MetadataAction::AcceptSourceAdmission);
    }
    auto operands=PooledImmutableLocalHotOperands::requestReset(*local,local_hot_geometry_->consumer(coordinate,local->consumerStorageIdentity()),header,control);
    operands.seal();code->enqueue(PooledImmutableLocalHotAction::ResetRequest,*local,operands,stream);
}

}
