/**
 * @file ImmutableLookupExecutionBoundary.cpp
 * @brief Admission and complete-parent terminal edges for immutable GPU row I/O.
 *
 * A persistent worker starts before the first captured model submission, so HIP
 * launch backpressure cannot withhold its CPU producer. The worker reads only
 * immutable lookup IDs and protocol metadata. Existing terminal observation is
 * the only join; no stage, token or retained transaction synchronizes the host.
 */
#include "ImmutableLookupExecutionBoundary.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "../local_execution/graph/GraphCaptureGuard.h"
#include "../../loaders/ModelLoader.h"
#include "../../backends/BackendManager.h"
#include "../../transfer/TransferEngine.h"
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>
namespace llaminar2::qwen4exp
{
namespace {
/** @brief Validate the exact already-materialized original protocol root without reading its contents.
 * @param owner Retained canonical tensor declaration. @param device Exact native participant.
 * @return Aligned original state address; no allocation or coherence transition occurs.
 */
ImmutableLookupDeviceState *protocolState(const std::shared_ptr<TensorBase> &owner,DeviceId device) {
    if(!device.is_gpu() || !owner || owner->native_type()!=TensorType::INT32 || owner->is_view() ||
        owner->transferStorageOwner()!=owner.get() || owner->size_bytes()!=sizeof(ImmutableLookupDeviceState) ||
        owner->numel()!=sizeof(ImmutableLookupDeviceState)/sizeof(std::int32_t) || owner->current_device()!=device ||
        !owner->gpu_data_ptr() || reinterpret_cast<std::uintptr_t>(owner->gpu_data_ptr())%alignof(ImmutableLookupDeviceState))
        throw std::invalid_argument("Immutable lookup requires its exact retained INT32/64B native protocol owner");
    return static_cast<ImmutableLookupDeviceState *>(owner->gpu_data_ptr());
}
/** @return Original exact INT8 source-native state; a subrange is never mistaken for an allocator extent. */
ImmutableLookupDeviceState* pooledProtocolState(const PooledImmutableSourcePreparedStorage& storage,DeviceId device){
    const auto root=storage.roots().native;const auto& offsets=storage.geometry().parameters().offsets;
    if(device!=storage.geometry().allocator().device || !root || root->native_type()!=TensorType::INT8 || root->is_view() ||
       root->transferStorageOwner()!=root.get() || root->size_bytes()!=offsets.native_bytes || root->numel()!=offsets.native_bytes ||
       root->current_device()!=device || !root->gpu_data_ptr() || offsets.protocol ||
       reinterpret_cast<std::uintptr_t>(root->gpu_data_ptr())%alignof(ImmutableLookupDeviceState))
        throw std::invalid_argument("Pooled source boundary lost its original exact INT8 native root");
    return static_cast<ImmutableLookupDeviceState*>(root->gpu_data_ptr());
}
/**
 * @brief Complete the original setup frontier before its last physical owner is released.
 * @param backend Exact original native backend. @param device Exact participant.
 * @param event Setup-created ordering event. @param stream Exact setup producer stream.
 * @param recorded Whether successful setup already recorded its final metadata frontier.
 * @note Only failed setup/admission or final submission-owner destruction calls
 * this infrastructure boundary. Successful admission, capture, replay and
 * request terminal publication never wait here. Failed submissions record after
 * partial writes; successful submissions retain their recorded event without a wait.
 */
void retireImmutableLookupSubmissionOwner(IBackend &backend, DeviceId device, void *event, void *stream,
                                          bool recorded) noexcept
{
    if ((!recorded && !backend.recordEvent(event, device.ordinal, stream)) ||
        !backend.waitForEvent(event, device.ordinal))
        std::terminate();
}
} // namespace

/**
 * @brief Retain initialization/admission frontiers through last boundary destruction.
 *
 * This setup-only owner keeps source, prepared roots and physical leases alive
 * even when runner construction fails before its first graph launch. The event
 * is recorded after initialization, and no successful setup waits for it. This
 * is not a copy of cache state or an inference/request lifecycle authority.
 */
class ImmutableLookupSubmissionRetirement final
{
  public:
    /** @brief Create the event and retain all owners before the first metadata enqueue. */
    ImmutableLookupSubmissionRetirement(DeviceId device, void *stream,
                                        std::shared_ptr<ImmutableLookupService> service,
                                        std::shared_ptr<TensorBase> protocol,
                                        std::shared_ptr<PooledImmutableSourcePreparedStorage> pooled = {})
        : device_(device), stream_(stream), backend_(getBackendFor(device)), service_(std::move(service)),
          protocol_(std::move(protocol)),pooled_(std::move(pooled))
    {
        if (!backend_ || !(event_ = backend_->createEvent(device.ordinal)))
            throw std::runtime_error("Immutable lookup could not create its submission-retirement event");
    }
    /**
     * @brief Seal successful initialization without blocking its setup stream.
     * @throws std::runtime_error If its exact producer event cannot be recorded.
     * @note On failure the destructor must still establish retirement or terminate.
     */
    void recordCompletion()
    {
        if (!stream_ || recorded_ || !backend_->recordEvent(event_, device_.ordinal, stream_))
            throw std::runtime_error("Immutable lookup could not seal its submission frontier");
        recorded_ = true;
    }
    /**
     * @brief Name a new admission after its previous producer frontier is ordered.
     * @param stream Exact upcoming protocol producer, outside capture.
     * @note The enclosing boundary proves no old interval is active. Adaptive
     * initialization is joined through the actual root producer events before
     * this event is re-recorded; ordinary intervals already retired their prior
     * admission. No bytes are read or native operations submitted here.
     */
    void beginSubmission(void *stream) noexcept
    {
        stream_ = stream;
        recorded_ = false;
    }
    /**
     * @brief Retire a partially submitted admission before failed runner setup unwinds.
     * @param stream Exact stream on which admission may have submitted writes.
     * @note Complete initialization first: admission may use another stream,
     * and re-recording an event alone would discard its earlier frontier. This
     * exception-only path never runs for a successful admission or model token.
     */
    void retireFailedAdmission(void *stream) noexcept
    {
        if (stream_ && stream_ != stream)
            retireImmutableLookupSubmissionOwner(*backend_, device_, event_, stream_, recorded_);
        retireImmutableLookupSubmissionOwner(*backend_, device_, event_, stream, false);
        stream_ = stream;
        recorded_ = true;
    }
    /** @brief Prove retirement before source, roots or their actual claims can destruct. */
    ~ImmutableLookupSubmissionRetirement() noexcept
    {
        if (stream_)
            retireImmutableLookupSubmissionOwner(*backend_, device_, event_, stream_, recorded_);
        backend_->destroyEvent(event_, device_.ordinal);
    }
    ImmutableLookupSubmissionRetirement(const ImmutableLookupSubmissionRetirement &) = delete;
    ImmutableLookupSubmissionRetirement &operator=(const ImmutableLookupSubmissionRetirement &) = delete;

  private:
    DeviceId device_;
    void *stream_, *event_ = nullptr;
    IBackend *backend_;
    bool recorded_ = false; ///< Immutable setup-frontier seal, never live GPU state.
    std::shared_ptr<ImmutableLookupService> service_;
    std::shared_ptr<TensorBase>
        protocol_; ///< Original ordinary/adaptive protocol owner, never replacement bytes.
    std::shared_ptr<PooledImmutableSourcePreparedStorage> pooled_; ///< Original source roots through partial setup/admission retirement.
};

std::shared_ptr<ImmutableLookupExecutionBoundary> ImmutableLookupExecutionBoundary::create(
    std::shared_ptr<HostImmutableRowProvider> table,DeviceId device,std::shared_ptr<TensorBase> protocol,
    std::size_t maximumTokenRows,std::shared_ptr<PhysicalMemoryAuthority> memory)
{
    (void)protocolState(protocol,device);
    if(!table||!memory||!device.is_gpu()||
        !maximumTokenRows||maximumTokenRows>std::numeric_limits<std::size_t>::max()/16||
        table->source().row_elements>static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Immutable lookup setup requires exact GPU/table/arena/PMA bindings and representable token geometry");
    const auto format=ggufToTensorType(table->source().format);
    const auto width=static_cast<int>(table->source().row_elements);const auto rowBytes=table->source().rowBytes();
    const auto layout=ImmutableLookupLayout::create(maximumTokenRows*16,rowBytes);
    auto service=ImmutableLookupService::create(table,layout,std::array{device},memory,table->retainedOwnerObservation().allocator().device);
    std::unique_ptr<IImmutableLookupKernels> kernels;
#ifdef HAVE_CUDA
    if(device.is_cuda())kernels=createCUDAImmutableLookupKernels(device,format,width,rowBytes);
#endif
#ifdef HAVE_ROCM
    if(device.is_rocm())kernels=createROCmImmutableLookupKernels(device,format,width,rowBytes);
#endif
    if(!kernels)throw std::runtime_error("Immutable lookup requested native GPU backend was not compiled");
    return std::make_shared<ImmutableLookupExecutionBoundary>(std::move(service),device,std::move(kernels),std::move(protocol));
}
ImmutableLookupExecutionBoundary::ImmutableLookupExecutionBoundary(
    std::shared_ptr<ImmutableLookupService> service, DeviceId device,
    std::unique_ptr<IImmutableLookupKernels> kernels, std::shared_ptr<TensorBase> protocol)
    : service_(std::move(service)), device_(device),
      binding_(service_
                   ? service_->bind(device)
                   : throw std::invalid_argument("Immutable lookup boundary requires its service owner")),
      kernels_(std::move(kernels)), protocol_(std::move(protocol)), backend_(getBackendFor(device)),
      state_(protocolState(protocol_, device)),
      generation_(reinterpret_cast<std::uint64_t *>(
          reinterpret_cast<std::byte *>(state_) + offsetof(ImmutableLookupDeviceState, admissionGeneration)))
{
    runtimeGeneration_ = backend_ ? backend_->deviceRuntimeGeneration(device.ordinal) : 0;
    if (!kernels_ || !runtimeGeneration_)
        throw std::invalid_argument("Immutable lookup requires prepared kernels "
                                    "and its live native backend generation");
    const std::array<TensorBase *, 1> roots{protocol_.get()};
    protocolSeal_ = std::make_unique<Qwen4ExpPreparedStorage>(device, roots);
    protocolSeal_->seal();
    identity_ = reinterpret_cast<std::uintptr_t>(this);
    for (const auto word :
         {reinterpret_cast<std::uintptr_t>(protocol_.get()), reinterpret_cast<std::uintptr_t>(state_),
          reinterpret_cast<std::uintptr_t>(generation_), reinterpret_cast<std::uintptr_t>(backend_),
          static_cast<std::uintptr_t>(runtimeGeneration_), static_cast<std::uintptr_t>(device.type),
          static_cast<std::uintptr_t>(device.ordinal),
          static_cast<std::uintptr_t>(sizeof(ImmutableLookupDeviceState)),
          static_cast<std::uintptr_t>(kImmutableLookupAcknowledgementVersion)})
    {
        identity_ ^= word;
        identity_ *= 1099511628211ULL;
    }
    if (!identity_)
        identity_ = 1;
    // Allocate once during ordinary lane setup. A failed admission can enqueue
    // before DGO publishes its state-ready event, so caller retirement alone
    // cannot cover it. No stream/work is named until a real admission begins.
    admissionRetirement_ = std::make_shared<ImmutableLookupSubmissionRetirement>(device, nullptr, service_,
                                                                                 protocol_);
}
std::shared_ptr<ImmutableLookupExecutionBoundary> ImmutableLookupExecutionBoundary::createPooled(
    std::shared_ptr<ImmutableLookupService> service,PooledImmutableSourceGeometry geometry,
    std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner,std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,
    std::shared_ptr<PhysicalMemoryAuthority> authority,void* stream){
    if(isGraphCaptureActive() || !service || !authority || !stream || reinterpret_cast<std::uintptr_t>(stream)<=2)
        throw std::invalid_argument("Pooled source boundary requires its real admitted worker and exact ordinary setup stream");
    const auto device=geometry.allocator().device;auto source=service->bindSubordinate(device);
    auto storage=source.preparePooled(geometry,std::move(owner),std::move(coordinator),std::move(authority),stream);
    std::unique_ptr<IPooledImmutableSourceKernels> kernels;
#ifdef HAVE_CUDA
    if(device.is_cuda())kernels=createCUDAPooledImmutableSourceKernels(geometry,source);
#endif
#ifdef HAVE_ROCM
    if(device.is_rocm())kernels=createROCmPooledImmutableSourceKernels(geometry,source);
#endif
    if(!kernels)throw std::invalid_argument("Pooled source native backend was not compiled");
    auto protocol=storage->roots().native;
    // Retain the actual backing and an event before the first initialization
    // enqueue. Failed factory/constructor paths drain only their own producer.
    auto retirement=std::make_shared<ImmutableLookupSubmissionRetirement>(device,stream,service,protocol,storage);
    kernels->initialize(*storage,stream);retirement->recordCompletion();
    Pooled pooled{std::move(storage),std::move(kernels),std::nullopt,std::move(retirement)};
    return std::shared_ptr<ImmutableLookupExecutionBoundary>(new ImmutableLookupExecutionBoundary(std::move(service),std::move(protocol),std::move(pooled)));
}
ImmutableLookupExecutionBoundary::ImmutableLookupExecutionBoundary(std::shared_ptr<ImmutableLookupService> service,
    std::shared_ptr<TensorBase> protocol,Pooled pooled)
    :service_(std::move(service)),device_(pooled.storage->geometry().allocator().device),pooled_(std::move(pooled)),
     protocol_(std::move(protocol)),backend_(getBackendFor(device_)),state_(pooledProtocolState(*pooled_->storage,device_)),
     generation_(&state_->admissionGeneration){
    runtimeGeneration_=backend_?backend_->deviceRuntimeGeneration(device_.ordinal):0;
    if(!runtimeGeneration_)throw std::invalid_argument("Pooled source boundary requires its live original native runtime");
    const std::array<TensorBase*,1> roots{protocol_.get()};protocolSeal_=std::make_unique<Qwen4ExpPreparedStorage>(device_,roots);protocolSeal_->seal();
    identity_=pooled_->kernels->captureIdentity(*pooled_->storage);
}
const std::shared_ptr<PooledImmutableSourcePreparedStorage>& ImmutableLookupExecutionBoundary::pooledStorage()const{
    if(!pooled_)throw std::logic_error("Storage boundary has no real pooled provider arm");requireProtocolOwner();return pooled_->storage;
}
const IPooledImmutableSourceKernels& ImmutableLookupExecutionBoundary::pooledKernels()const{
    (void)pooledStorage();return *pooled_->kernels;
}
void ImmutableLookupExecutionBoundary::bindPooledGenerationControl(DeviceGenerationControlBinding control){
    (void)pooledStorage();control.requireFor(device());
    if(pooled_->control && pooled_->control->captureIdentity()!=control.captureIdentity())
        throw std::invalid_argument("Pooled source boundary cannot replace its original canonical controller");
    bindGenerationControl({control.healthy(),control.error()});
    pooled_->control=control;
}

const ImmutableLookupDeviceView &ImmutableLookupExecutionBoundary::channel() const
{
    if (!binding_)
        throw std::logic_error("Adaptive subordinate storage cannot be used as an "
                               "ordinary lookup channel");
    return binding_->deviceView();
}

const IImmutableLookupKernels &ImmutableLookupExecutionBoundary::kernels() const
{
    if (!kernels_)
        throw std::logic_error("Adaptive subordinate storage cannot use ordinary lookup kernels");
    return *kernels_;
}

HostImmutableRowOwnerObservation ImmutableLookupExecutionBoundary::retainedSourceOwnerObservation() const
{
    return service_->retainedSourceOwnerObservation();
}

void ImmutableLookupExecutionBoundary::requireProtocolOwner() const
{
    protocolSeal_->validate();
    if(pooled_ && (protocol_!=pooled_->storage->roots().native ||
       pooled_->kernels->captureIdentity(*pooled_->storage)!=identity_))
        throw std::logic_error("Pooled source original roots/provider/code identity changed after setup");
    if((pooled_?pooledProtocolState(*pooled_->storage,device()):protocolState(protocol_,device()))!=state_ || getBackendFor(device())!=backend_ ||
        backend_->deviceRuntimeGeneration(device().ordinal)!=runtimeGeneration_)
        throw std::logic_error("Immutable lookup protocol owner/storage/backend generation changed after setup");
}
void ImmutableLookupExecutionBoundary::beginAdmission(std::uint64_t generation,void *stream)
{
    requireProtocolOwner();
    if(isGraphCaptureActive()||!generation||!stream||reinterpret_cast<std::uintptr_t>(stream)<=2)
        throw std::logic_error("Immutable lookup admission requires a positive request epoch outside capture and an exact stream");
    if(armed_.load(std::memory_order_acquire))
    {
        service_->requireHealthy();
        if(requestEpoch_!=generation)throw std::logic_error("Immutable lookup request replaced an unretired storage interval");
        return; // Prefill and generation-ledger admission share one retained I/O interval.
    }
    if(intervalGeneration_==~std::uint64_t{0})throw std::overflow_error("Immutable lookup I/O admission identity exhausted");
    const auto interval=intervalGeneration_+1;
    intervalStart_=service_->stats();terminal_.reset();
    service_->requireHealthy();
    if (pooled_)
    {
        // Authenticate the original file/provider once at this ordinary request
        // boundary. Binding performs no gather, policy touch, sequence change
        // or GPU-value observation; all-hit captured transactions still do U0.
        const auto source = service_->bindSubordinate(device_);
        if (source.captureIdentity() != pooled_->kernels->source().captureIdentity())
            throw std::logic_error("Adaptive admission replaced its original source channel");
    }
    ticket_=service_->arm(interval);
    if (!pooled_)
        admissionRetirement_->beginSubmission(stream);
    try {
        if (pooled_) {
            pooled_->storage->acquireSetup(stream);
            for(const auto roots:{pooled_->storage->roots(),pooled_->storage->owner()->roots(),pooled_->storage->coordinator()->roots()})
                for(const auto& root:{roots.cache,roots.frame,roots.native})if(root)
                    TransferEngine::requireDeviceInput(root.get(),device(),stream);
            pooled_->setupRetirement->beginSubmission(stream);
            pooled_->kernels->admit(*pooled_->storage,ticket_.generation,ticket_.invocation,stream);
            pooled_->setupRetirement->recordCompletion();
        }
        else
            kernels_->admit({interval, generation_, state_, initialization_}, stream);
        // The admission kernel is the real producer of these bytes. Publish its
        // exact stream/event before RequireReadyBytes joins any capture frontier.
        // The adaptive component publishes this same original root itself.
        // Ordinary protocol kernels retain their existing boundary publication.
        if (!pooled_)
        {
            TransferEngine::publishDeviceWrite(protocol_.get(), device(), stream);
            admissionRetirement_->recordCompletion();
        }
    } catch(...) {
        // Admission can enqueue a protocol write before publication fails.
        // Retire both original initialization and this exact failed producer
        // before source/root references or the already-armed worker can unwind.
        // Successful admissions and all captured transactions never wait here.
        if (pooled_)
            pooled_->setupRetirement->retireFailedAdmission(stream);
        else
            admissionRetirement_->retireFailedAdmission(stream);
        (void)service_->cancel(ticket_);
        try{(void)service_->await(ticket_);}catch(...){std::terminate();}
        ticket_={};throw;
    }
    initialization_=ImmutableLookupAdmissionArgs::ProtocolInitialization::Preserve;
    requestEpoch_=generation;intervalGeneration_=interval;
    armed_.store(true,std::memory_order_release);
}
void ImmutableLookupExecutionBoundary::bindGenerationControl(RetainedStorageGenerationControl control)
{
    if(!control.healthy||!control.error||control.healthy==control.error||
       (control_.healthy&&(control_.healthy!=control.healthy||control_.error!=control.error)))
        throw std::invalid_argument("Immutable lookup requires immutable canonical generation health/error addresses");
    control_=control;
}
RetainedStorageGenerationControl ImmutableLookupExecutionBoundary::generationControl() const
{
    if(!control_.healthy||!control_.error)throw std::logic_error("Immutable lookup stage has no bound canonical generation failure words");
    return control_;
}
void ImmutableLookupExecutionBoundary::requireArmedForLaunch() const
{
    requireProtocolOwner();
    service_->requireHealthy();
    if(!armed_.load(std::memory_order_acquire))throw std::logic_error("Immutable lookup graph launch has no armed retained-generation storage interval");
}
void ImmutableLookupExecutionBoundary::recordTerminalClose(void *stream,const int *healthy,const int *complete) const
{
    requireProtocolOwner();
    if (pooled_) {
        if(!pooled_->control)throw std::invalid_argument("Pooled source terminal requires its original canonical controller");
        if(!healthy && !complete)pooled_->kernels->closeForReset(*pooled_->storage,*pooled_->control,stream);
        else {
            if(healthy!=pooled_->control->healthy() || complete!=pooled_->control->row()+sampling_math::kDeviceGenerationControlRequestComplete)
                throw std::invalid_argument("Pooled terminal predicates must be the original canonical controller");
            pooled_->kernels->closeWhenTerminal(*pooled_->storage,*pooled_->control,stream);
        }
    }
    else
        kernels_->close({generation_, state_, binding_->deviceView(), healthy, complete}, stream);
}
void ImmutableLookupExecutionBoundary::awaitTerminal()
{
    // A failed I/O lane still owes its worker join; launch admission's health
    // check must never bypass terminal retirement and its precise diagnostic.
    if(!armed_.load(std::memory_order_acquire))throw std::logic_error("Immutable lookup terminal has no armed storage interval");
    const bool success=service_->await(ticket_);ticket_={};
    terminal_=RetainedStorageTerminalObservation::between(intervalGeneration_,service_->mode(),intervalStart_,service_->stats());
    if (pooled_)
        terminal_->transport = RetainedStorageTransport::PooledSubordinate;
    armed_.store(false,std::memory_order_release);
    if(!success)throw std::runtime_error("Immutable lookup retained-generation storage failed: "+service_->lastFailure());
}
RetainedStorageTerminalObservation ImmutableLookupExecutionBoundary::terminalObservation() const
{
    if(armed_.load(std::memory_order_acquire)||!terminal_)
        throw std::logic_error("Immutable lookup terminal statistics require this interval's completed worker join");
    return *terminal_;
}
} // namespace llaminar2::qwen4exp
