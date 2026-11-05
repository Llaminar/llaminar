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
#include "PooledImmutableRetainedParticipants.h"
#include "execution/local_execution/graph/IRetainedGraphParticipants.h"
#include "PooledImmutableCompleteCohort.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/NativeOrderingEventGroup.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
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
bool ImmutableLookupExecutionBoundary::requiresSourceGroupAdmission()const noexcept
{
    return service_->channelDevices().size() > 1 ||
        (pooled_ && pooled_->storage->geometry().parameters().consumers > 1);
}

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
    /** @brief Validate every cold preclaim before the cohort constructs its first actual frontier. */
    static void requireOrderingGroup(const NativeOrderingEventGroup& ordering,
        const std::shared_ptr<PhysicalMemoryAuthority>& memory,const GPUContextProvenance& original){
        ordering.requireOriginal(memory,original,NativeOrderingEventProgram::lookupSubmissionRetirement());
    }
    /** @brief Create the event and retain all owners before the first metadata enqueue. */
    ImmutableLookupSubmissionRetirement(DeviceId device, void *stream,
                                        std::shared_ptr<ImmutableLookupService> service,
                                        std::shared_ptr<TensorBase> protocol,
                                        std::shared_ptr<PhysicalMemoryAuthority> originalMemory,
                                        std::shared_ptr<PooledImmutableSourcePreparedStorage> pooled = {},
                                        std::shared_ptr<IWorkerGPUContext> worker = {},
                                        std::unique_ptr<NativeOrderingEventGroup> ordering = {})
        : device_(device), stream_(stream), backend_(getBackendFor(device)), service_(std::move(service)),
          protocol_(std::move(protocol)),pooled_(std::move(pooled)),worker_(std::move(worker))
    {
        if(!backend_ || !service_ || !originalMemory)
            throw std::invalid_argument("Immutable lookup retirement requires its original backend, service and authority");
        const auto observation=service_->retainedSourceOwnerObservation();
        const auto& actual=observation.authority();
        if(actual.get()!=originalMemory.get() || actual.owner_before(originalMemory) || originalMemory.owner_before(actual))
            throw std::invalid_argument("Immutable lookup retirement cannot substitute its original source authority");
        const auto provenance=GPUDeviceContextPool::instance().retainExistingContextProvenance(device);
        if(worker_)provenance.requireWorker(worker_);
        if(ordering){
            ordering->requireOriginal(originalMemory,provenance,NativeOrderingEventProgram::lookupSubmissionRetirement());
            ordering_=std::move(ordering);
        }else ordering_=NativeOrderingEventGroup::claim(std::move(originalMemory),provenance,
            NativeOrderingEventProgram::lookupSubmissionRetirement());
        ordering_->beginNativeAttempt();
        try {
            if (!backend_ || !(event_ = backend_->createEvent(device.ordinal)))
                throw std::runtime_error("Immutable lookup could not create its submission-retirement event");
        } catch (...) {
            // No native handle escaped a failed creator. Retained physical
            // commitment still follows the original context/process lifetime.
            ordering_->completeNativeHandles();
            throw;
        }
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
        const auto retire = [&] {
            if (stream_)
                retireImmutableLookupSubmissionOwner(*backend_, device_, event_, stream_, recorded_);
            backend_->destroyEvent(event_, device_.ordinal);
            ordering_->completeNativeHandles();
        };
        // Remote setup events belong to their original retained worker. The
        // ordinary boundary keeps its existing calling-thread contract.
        if (worker_) {
            try { worker_->submitAndWait(retire); } catch (...) { std::terminate(); }
        } else retire();
    }
    ImmutableLookupSubmissionRetirement(const ImmutableLookupSubmissionRetirement &) = delete;
    ImmutableLookupSubmissionRetirement &operator=(const ImmutableLookupSubmissionRetirement &) = delete;

  private:
    std::unique_ptr<NativeOrderingEventGroup> ordering_; ///< Outlives every native handle and retained root.
    DeviceId device_;
    void *stream_, *event_ = nullptr;
    IBackend *backend_;
    bool recorded_ = false; ///< Immutable setup-frontier seal, never live GPU state.
    std::shared_ptr<ImmutableLookupService> service_;
    std::shared_ptr<TensorBase>
        protocol_; ///< Original ordinary/adaptive protocol owner, never replacement bytes.
    std::shared_ptr<PooledImmutableSourcePreparedStorage> pooled_; ///< Original source roots through partial setup/admission retirement.
    std::shared_ptr<IWorkerGPUContext> worker_; ///< Exact remote worker survives event retirement and root destruction.
};

/** @brief One original boundary owner, transferable across caller/destructor threads. */
struct PooledImmutableCompleteCohort::OwnershipClaim {
    std::weak_ptr<ImmutableLookupExecutionBoundary> boundary;
    explicit OwnershipClaim(const std::shared_ptr<ImmutableLookupExecutionBoundary>& original):boundary(original) {
        if(original->cohortOwnerClaim_.test_and_set(std::memory_order_acquire))
            throw std::logic_error("Original pooled boundary is already constructing, owned or retiring");
    }
    ~OwnershipClaim(){
        // State retains the original boundary through every native retirement.
        // Afterwards an expired boundary cannot be rebound; a surviving one
        // stays alive under this lock until its ownership claim is released.
        if(const auto original=boundary.lock())original->cohortOwnerClaim_.clear(std::memory_order_release);
    }
};
/** @copydoc PooledImmutableCompleteCohort::submissionRetirementBOM */
PhysicalMemoryPlan PooledImmutableCompleteCohort::submissionRetirementBOM(
    PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> remotes)
{
    if(!host.valid() || !host.device.is_cpu())
        throw std::invalid_argument("Legacy cohort retirement requires its original CPU observation");
    PhysicalMemoryPlanBuilder builder;
    for(std::size_t i=0;i<remotes.size();++i) {
        if(!remotes[i].device.is_gpu() || remotes[i].world_rank!=host.world_rank)
            throw std::invalid_argument("Legacy cohort retirement requires exact rank-local remote GPU observations");
        for(std::size_t j=0;j<i;++j)if(remotes[j].device==remotes[i].device)
            throw std::invalid_argument("Legacy cohort retirement duplicated a remote GPU");
    }
    for(const auto& remote:remotes) {
        const auto ordering=NativeOrderingEventGroup::bom(
            NativeOrderingEventProgram::lookupSubmissionRetirement(),host,remote);
        for(const auto& bom:ordering.resources())builder.add(bom);
    }
    return builder.build();
}
/** @brief Retained original graph/capture metadata; never device execution state. */
struct PooledImmutableCompleteCohort::State {
    struct FrozenParticipant {
        PooledImmutableCohortParticipant original;
        std::shared_ptr<RetainedParticipantGraphExecutable> executable;
        IGPUGraphCapture* capture=nullptr; ///< Borrowed recording facade; the strong executable remains its only native authority.
        mutable std::uint64_t invocation=0; ///< Original CPU-owned submission lifetime, never GPU/controller progress.

        DeviceId device;
        std::uint64_t topology;
        struct Leaf {std::string name;std::vector<std::string> dependencies;const IComputeStage* stage;std::uint64_t identity;};
        std::vector<Leaf> stages;
    };
    FrozenParticipant source;
    PooledImmutableRetirementCompletion completion;
    std::vector<PooledImmutableCohortRemote> remotes;
    std::vector<FrozenParticipant> frozenRemotes;
    std::vector<std::shared_ptr<ImmutableLookupSubmissionRetirement>> frontiers;
    mutable std::mutex submissionOrder; ///< Serializes CPU enqueue groups only; no GPU completion is awaited.

};
namespace {
/** @brief Authenticate one existing pool worker and every exact cold prepared leaf. */
void requireCohortParticipant(const PooledImmutableCohortParticipant& participant,DeviceId device) {
    if (!participant.graph || !participant.worker || !participant.context || !participant.arena ||
        participant.context->deviceId()!=device ||
        !participant.stream || reinterpret_cast<std::uintptr_t>(participant.stream)<=2 ||
        participant.worker!=GPUDeviceContextPool::instance().retainExistingContext(device) ||
        participant.worker->deviceOrdinal()!=device.ordinal)
        throw std::invalid_argument("Complete pooled cohort requires its original worker/context/arena/stream");
    if (participant.graph->getExecutionOrder().empty())
        throw std::invalid_argument("Complete pooled cohort cannot publish an empty participant");
    for (const auto& name:participant.graph->getExecutionOrder()) {
        const auto* node=participant.graph->getNode(name);
        if (!node || node->device!=device || !node->stage || node->stage->device()!=device ||
            !node->stage->supportsBackend(participant.context->backendType()) ||
            (node->stage->type()!=ComputeStageType::QWEN4EXP_POOLED_NATIVE_PHASE &&
             node->stage->type()!=ComputeStageType::QWEN4EXP_POOLED_SOURCE_PHASE &&
             node->stage->type()!=ComputeStageType::QWEN4EXP_NGRAM_LOOKUP &&
             node->stage->type()!=ComputeStageType::QWEN4EXP_POOLED_DIRECTED_TRANSFER) ||
            node->stage->isManualGraphBoundary() ||
            node->stage->graphLaunchPreparationPolicy()==GraphLaunchPreparationPolicy::CaptureAndReplay)
            throw std::invalid_argument("Complete pooled cohort changed its participant-local capture declaration");
    }
}
/** @brief Join all exact root producers without allocating, copying or observing their device contents. */
void requireCohortInputs(const PooledImmutableNativeRoots& roots,DeviceId device,void* stream) {
    for (const auto& root:{roots.cache,roots.frame,roots.native}) if (root)
        TransferEngine::requireDeviceInput(root.get(),device,stream);
}
}
PooledImmutableCompleteCohort::PooledImmutableCompleteCohort(std::unique_ptr<OwnershipClaim> ownership,std::unique_ptr<State> state)
    :ownership_(std::move(ownership)),state_(std::move(state)){}
std::shared_ptr<PooledImmutableCompleteCohort> PooledImmutableCompleteCohort::bind(
    PooledImmutableCohortParticipant source,PooledImmutableRetirementCompletion completion,
    std::vector<PooledImmutableCohortRemote> remotes) {
    return bindImpl(std::move(source),std::move(completion),std::move(remotes),std::nullopt);
}
std::shared_ptr<PooledImmutableCompleteCohort> PooledImmutableCompleteCohort::bindWithOrderingGroups(
    PooledImmutableCohortParticipant source,PooledImmutableRetirementCompletion completion,
    std::vector<PooledImmutableCohortRemote> remotes,std::vector<std::unique_ptr<NativeOrderingEventGroup>> ordering){
    if(ordering.size()!=remotes.size() || std::any_of(ordering.begin(),ordering.end(),[](const auto& group){return !group;}))
        throw std::invalid_argument("Recorded complete cohort requires every original remote event program exactly once");
    return bindImpl(std::move(source),std::move(completion),std::move(remotes),std::move(ordering));
}
std::shared_ptr<PooledImmutableCompleteCohort> PooledImmutableCompleteCohort::bindImpl(
    PooledImmutableCohortParticipant source,PooledImmutableRetirementCompletion completion,
    std::vector<PooledImmutableCohortRemote> remotes,std::optional<std::vector<std::unique_ptr<NativeOrderingEventGroup>>> ordering){
    const auto boundary=completion.read.publication.source.boundary;
    if(isGraphCaptureActive() || !boundary || !source.worker ||
       source.worker!=GPUDeviceContextPool::instance().retainExistingContext(boundary->device()))
        throw std::invalid_argument("Complete pooled cold binding requires its original source worker outside capture");
    if(source.worker->ownsCurrentThread())
        throw std::invalid_argument("Complete pooled cold coordinator cannot occupy its original source worker");
    auto ownership=std::make_unique<OwnershipClaim>(boundary);
    return bindClaimed(std::move(source),std::move(completion),std::move(remotes),std::move(ownership),std::move(ordering));
}
std::shared_ptr<PooledImmutableCompleteCohort> PooledImmutableCompleteCohort::bindClaimed(
    PooledImmutableCohortParticipant source,PooledImmutableRetirementCompletion completion,
    std::vector<PooledImmutableCohortRemote> remotes,std::unique_ptr<OwnershipClaim> ownership,
    std::optional<std::vector<std::unique_ptr<NativeOrderingEventGroup>>> ordering) {
    if (isGraphCaptureActive() || !source.graph || !completion.read.publication.source.boundary)
        throw std::invalid_argument("Complete pooled cohort publication requires actual cold terminal producers");
    const auto boundary=completion.read.publication.source.boundary;
    if (boundary->hasActiveInterval())
        throw std::logic_error("Cannot replace a complete pooled cohort during its original active interval");
    if (boundary->completeCohort_.load(std::memory_order_acquire).lock())
        throw std::logic_error("Original pooled source already belongs to a retained complete cohort");
    requirePooledRetirementCompletion(*source.graph,completion);
    requireCohortParticipant(source,boundary->device());
    if (source.graph->retainedStorageBoundary()!=boundary)
        throw std::invalid_argument("Complete pooled source graph lost its original retained boundary");
    const auto storage=boundary->pooledStorage();
    const auto& parameters=storage->geometry().parameters();
    const auto count=storage->originalOwners().size();
    if (remotes.size()!=count-(parameters.local_owner!=adaptive_lookup::None?1:0))
        throw std::invalid_argument("Complete pooled cohort omits or duplicates an original owner");
    auto state=std::make_unique<State>();
    const auto freeze=[](PooledImmutableCohortParticipant participant,DeviceId device) {
        State::FrozenParticipant result{};result.original=std::move(participant);result.device=device;
        result.executable=result.original.executable;
        result.topology=result.original.graph->topologyGeneration();
        for (const auto& name:result.original.graph->getExecutionOrder())
            result.stages.push_back({name,result.original.graph->getNode(name)->dependencies,
                result.original.graph->getNode(name)->stage.get(),result.original.graph->getNode(name)->stage->graphCaptureVariantSignature()});
        return result;
    };
    state->source=freeze(std::move(source),boundary->device());
    state->completion=std::move(completion);state->remotes=std::move(remotes);
    std::size_t next=0;
    for (std::size_t coordinate=0;coordinate<count;++coordinate) {
        if (coordinate==parameters.local_owner) continue;
        auto& remote=state->remotes[next++];
        const auto* holder=std::get_if<std::shared_ptr<const PooledImmutableCapturedRetirementWave>>(&state->completion.owners[coordinate]);
        if (remote.coordinate!=coordinate || !holder || !*holder || !remote.participant.graph ||
            pooledSourceOwner((*holder)->read()->publication()->sourceWave())!=storage->originalOwners()[coordinate])
            throw std::invalid_argument("Complete pooled cohort changed original ordered physical owner/root identity");
        const auto device=storage->originalOwners()[coordinate]->plan().allocator().device;
        if (device==boundary->device())
            throw std::invalid_argument("Complete pooled cohort cannot hide a local owner behind a remote program");
        requirePooledRetirementOwnerCompletion(*remote.participant.graph,*holder,remote.retiredSent);
        requireCohortParticipant(remote.participant,device);
        if(remote.participant.worker->ownsCurrentThread())
            throw std::invalid_argument("Complete pooled cold coordinator cannot occupy an original remote worker");
        if (remote.participant.graph==state->source.original.graph)
            throw std::invalid_argument("Complete pooled cohort requires exclusive participant graph/capture owners");
        for (const auto& previous:state->frozenRemotes)
            if (previous.original.graph==remote.participant.graph)
                throw std::invalid_argument("Complete pooled cohort duplicated a participant execution owner");
        state->frozenRemotes.push_back(freeze(std::move(remote.participant),device));
    }
    const auto requireOriginalExecutable=[&](const State::FrozenParticipant& participant){
        const auto& executable=participant.executable;
        if(!executable || executable->self_.owner_before(executable) || executable.owner_before(executable->self_) || executable->phase_!=RetainedParticipantGraphExecutable::Phase::Fresh ||
            executable->worker_.get()!=participant.original.worker.get() ||
            executable->worker_.owner_before(participant.original.worker) || participant.original.worker.owner_before(executable->worker_) ||
            executable->device_!=participant.device || executable->stream_!=participant.original.stream ||
            executable->memory_.get()!=boundary->originalMemory_.get() || executable->memory_.owner_before(boundary->originalMemory_) ||
            boundary->originalMemory_.owner_before(executable->memory_))
            throw std::invalid_argument("Complete pooled participant requires its same original prepared-PMA/worker/device/stream graph owner");
        const auto original=GPUDeviceContextPool::instance().retainExistingPreparedContext(participant.device,boundary->originalMemory_);
        original.requireWorker(participant.original.worker);
        if(original.generation()!=executable->generation_)throw std::invalid_argument("Complete pooled participant has stale original executable generation");
    };
    requireOriginalExecutable(state->source);
    for(const auto& participant:state->frozenRemotes)requireOriginalExecutable(participant);
    if(ordering){
        if(ordering->size()!=state->frozenRemotes.size())throw std::invalid_argument("Complete cohort changed recorded event membership");
        for(std::size_t i=0;i<ordering->size();++i){
            if(!(*ordering)[i])throw std::invalid_argument("Complete cohort omitted its original recorded event group");
            const auto original=GPUDeviceContextPool::instance().retainExistingContextProvenance(state->frozenRemotes[i].device);
            original.requireWorker(state->frozenRemotes[i].original.worker);
            ImmutableLookupSubmissionRetirement::requireOrderingGroup(*(*ordering)[i],boundary->originalMemory_,original);
        }
    }
    // No two retained programs may record the same definition. A rejected
    // later claim closes only this attempt's already-claimed fresh owners;
    // another program's owner is never torn down by this failure.
    std::vector<std::shared_ptr<RetainedParticipantGraphExecutable>> claimed;
    claimed.reserve(1+state->frozenRemotes.size()); // All bookkeeping allocation precedes the first irreversible claim.
    try{
        const auto claim=[&](const State::FrozenParticipant& participant){participant.executable->bindCohortDefinition();claimed.push_back(participant.executable);};
        claim(state->source);for(const auto& participant:state->frozenRemotes)claim(participant);
    }catch(...){for(const auto& executable:claimed)executable->retireDefinition();throw;}
    // Every declaration and one-shot owner claim rejects before this binding's
    // native frontier creation or original graph recording. Publish only after this owner has recorded
    // the actual graphs; its destructor covers all partial cold submissions.
    auto result=std::shared_ptr<PooledImmutableCompleteCohort>(new PooledImmutableCompleteCohort(std::move(ownership),std::move(state)));
    auto& owned=*result->state_;
    for (std::size_t i=0;i<owned.frozenRemotes.size();++i) {
        const auto& remote=owned.frozenRemotes[i];
        std::shared_ptr<ImmutableLookupSubmissionRetirement> frontier;
        remote.original.worker->submitAndWait([&] {
            frontier=std::make_shared<ImmutableLookupSubmissionRetirement>(remote.device,remote.original.stream,
                boundary->service_,boundary->protocol_,boundary->originalMemory_,storage,remote.original.worker,
                ordering?std::move((*ordering)[i]):std::unique_ptr<NativeOrderingEventGroup>{});
            frontier->recordCompletion();
        });
        owned.frontiers.push_back(std::move(frontier));
    }
    // Original setup/root-input producers are immutable for this retained
    // cohort. Acquire them once from the off-worker cold coordinator; request
    // admission must never hold source A while synchronously waiting on B.
    result->acquireRemoteSetup();
    const auto record=[](State::FrozenParticipant& participant) {
        participant.original.worker->submitAndWait([&] {
            participant.capture=&participant.executable->capture();
            if (participant.capture->executionStream()!=participant.original.stream ||
                participant.capture->hasExecutable() || participant.capture->nodeCount()!=0)
                throw std::runtime_error("Complete pooled recorder requires its pristine sole original capture owner");
            DeviceGraphExecutor executor;
            executor.setArena(participant.original.arena.get());
            // This closed source program includes explicit captured transfer
            // collectives. The generic retained-child recorder correctly rejects
            // collectives, so use the original public setup recorder/ledger here
            // without relaxing its policy or accepting a caller executable.
            constexpr auto label="complete original pooled reader/retirement participant";
            if(!executor.allocateGraphStorageForCapture(*participant.original.graph,participant.original.context.get(),label))
                throw std::runtime_error("Complete pooled participant storage prebinding failed");
            for(const auto& name:participant.original.graph->getExecutionOrder()) {
                auto* stage=participant.original.graph->getNode(name)->stage.get();
                if(!stage->prepareGraphLaunch(participant.original.context.get(),participant.original.stream) || !stage->isGraphCapturable())
                    throw std::runtime_error("Complete pooled participant stage preparation failed: "+name);
            }
            if(!executor.prepareGraphStorageForCapture(*participant.original.graph,participant.original.context.get(),participant.original.stream,label))
                throw std::runtime_error("Complete pooled participant capture frontier preparation failed");
            auto dependencies=executor.planGraphCaptureDependencies(*participant.original.graph,
                participant.original.graph->getExecutionOrder(),participant.device,participant.original.stream,label);
            if(!dependencies)throw std::runtime_error("Complete pooled participant capture dependency ledger is incomplete");
            {
                ScopedBackendGraphCapture capture(*participant.original.worker,*participant.capture,label,dependencies.get());
                if(!capture.begin())throw std::runtime_error("Complete pooled participant beginCapture failed");
                if(!executor.recordSetupGraphMaterialization(*participant.original.graph,participant.original.context.get()))
                    throw std::runtime_error("Complete pooled participant actual graph recording failed");
                capture.finish();
            }
            if(!participant.capture->nodeCount())throw std::runtime_error("Complete pooled participant recorded an empty native graph");
            std::string detail;
            if (!participant.capture->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&detail) ||
                !participant.capture->instantiate())
                throw std::runtime_error("Complete pooled participant executable admission failed: "+detail);
        });
    };
    record(owned.source);
    for (auto& remote:owned.frozenRemotes)record(remote);
    result->requireCurrent();boundary->bindCompleteCohort(result);return result;
}
void PooledImmutableCompleteCohort::requireCurrent()const {
    const auto check=[](const State::FrozenParticipant& frozen) {
        requireCohortParticipant(frozen.original,frozen.device);
        if (!frozen.executable || !frozen.executable->readyForSubmission() || !frozen.capture || !frozen.capture->hasExecutable() || frozen.capture->executionStream()!=frozen.original.stream)
            throw std::logic_error("Complete pooled cohort lost its privately recorded original executable");
        if (frozen.original.graph->topologyGeneration()!=frozen.topology ||
            frozen.original.graph->getExecutionOrder().size()!=frozen.stages.size())
            throw std::logic_error("Complete pooled cohort graph topology changed after capture publication");
        for (const auto& leaf:frozen.stages) {
            const auto* current=frozen.original.graph->getNode(leaf.name);
            // ComputeNode::stage is publicly replaceable. Compare the current
            // owner before inspecting any stage bytes: the cached raw pointer
            // may already refer to released storage despite unchanged topology.
            if(!current || current->stage.get()!=leaf.stage || current->dependencies!=leaf.dependencies)
                throw std::logic_error("Complete pooled cohort replaced an original captured stage or predecessor");
            const auto* stage=current->stage.get();
            if(!stage->hasGPUStream() || stage->requireGPUStream()!=frozen.original.stream ||
               stage->graphCaptureVariantSignature()!=leaf.identity)
                throw std::logic_error("Complete pooled cohort stage/root identity changed after capture publication");
        }
    };
    check(state_->source);
    for (const auto& remote:state_->frozenRemotes) check(remote);
    // Cold graph guards inspect concrete predecessor/operand objects. Capture
    // consumes the already sealed topology/leaf/stream identities above; these
    // original guards intentionally reject being called from inside capture.
    if (!isGraphCaptureActive()) {
        requirePooledRetirementCompletion(*state_->source.original.graph,state_->completion);
        for (std::size_t i=0;i<state_->remotes.size();++i)
            requirePooledRetirementOwnerCompletion(*state_->frozenRemotes[i].original.graph,
                std::get<std::shared_ptr<const PooledImmutableCapturedRetirementWave>>(state_->completion.owners[state_->remotes[i].coordinate]),state_->remotes[i].retiredSent);
    }
}
void PooledImmutableCompleteCohort::launch()const {
    // Keep each participant's FIFO invocation pairing identical even if two
    // admitted CPU callers enqueue concurrently. This holds no GPU wait.
    std::lock_guard submission(state_->submissionOrder);
    const auto boundary=state_->completion.read.publication.source.boundary;
    boundary->requireArmedForLaunch();
    if(state_->source.original.worker->ownsCurrentThread())throw std::invalid_argument("Cohort launch requires its off-worker coordinator");
    for(const auto& remote:state_->frozenRemotes)if(remote.original.worker->ownsCurrentThread())throw std::invalid_argument("Cohort launch cannot occupy an original peer worker");
    // After the first actual launch, an exception is not proof that a peer
    // stopped reading. Failed submission is fatal before any owner unwinds.
    const auto submit=[](const State::FrozenParticipant& participant) {
        participant.original.worker->submitAndWait([&] {
            if(participant.invocation==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("Complete pooled participant invocation ownership exhausted");
            if(!participant.executable->submit(++participant.invocation))throw std::runtime_error("Original pooled participant owned submission failed");
        });
    };
    try { submit(state_->source);for(const auto& remote:state_->frozenRemotes)submit(remote); }
    catch (...) { std::terminate(); }
}
std::vector<GPUGraphKernelNodeInfo> PooledImmutableCompleteCohort::kernelInventory(std::optional<std::uint32_t> coordinate)const {
    if (isGraphCaptureActive() || state_->completion.read.publication.source.boundary->hasActiveInterval())
        throw std::logic_error("Kernel inventory requires its cold original cohort outside an active interval/capture");
    requireCurrent();
    if(state_->source.original.worker->ownsCurrentThread())throw std::invalid_argument("Kernel inventory requires its off-worker coordinator");
    for(const auto& remote:state_->frozenRemotes)if(remote.original.worker->ownsCurrentThread())throw std::invalid_argument("Kernel inventory cannot occupy an original peer worker");
    const State::FrozenParticipant* participant=&state_->source;
    if(coordinate){
        participant=nullptr;
        for(std::size_t i=0;i<state_->remotes.size();++i)if(state_->remotes[i].coordinate==*coordinate)participant=&state_->frozenRemotes[i];
        if(!participant)throw std::invalid_argument("Kernel inventory requires an actual original remote owner coordinate");
    }
    std::vector<GPUGraphKernelNodeInfo> nodes;
    participant->original.worker->submitAndWait([&]{std::string detail;if(!participant->capture->inspectKernelNodes(nodes,&detail))throw std::runtime_error("Original pooled kernel inventory failed: "+detail);});
    return nodes;
}
std::vector<GPUGraphKernelNodeInfo> PooledImmutableCompleteCohort::sourceKernelInventory()const{return kernelInventory({});}
std::vector<GPUGraphKernelNodeInfo> PooledImmutableCompleteCohort::ownerKernelInventory(std::uint32_t coordinate)const{return kernelInventory(coordinate);}
void PooledImmutableCompleteCohort::requireSourceStream(void* stream)const {
    requireCurrent();
    if (stream!=state_->source.original.stream || !state_->source.original.worker->ownsCurrentThread())
        throw std::invalid_argument("Complete pooled admission/terminal must use its exact retained source worker and stream");
}
void PooledImmutableCompleteCohort::requireSource(const PooledImmutableSourcePreparedStorage& storage,void* stream)const {
    requireSourceStream(stream);
    if (&storage!=state_->completion.read.publication.source.boundary->pooledStorage().get())
        throw std::invalid_argument("Complete pooled reset substituted the original source storage");
}
void PooledImmutableCompleteCohort::acquireRemoteSetup()const {
    for (std::size_t i=0;i<state_->frozenRemotes.size();++i) {
        const auto& remote=state_->frozenRemotes[i];
        remote.original.worker->submitAndWait([&] {
            state_->frontiers[i]->beginSubmission(remote.original.stream);
            const auto owner=state_->completion.read.publication.source.boundary->pooledStorage()->originalOwners()[state_->remotes[i].coordinate];
            owner->acquireSetup(remote.original.stream);
            requireCohortInputs(owner->roots(),remote.device,remote.original.stream);
            state_->frontiers[i]->recordCompletion();
        });
    }
}
void PooledImmutableCompleteCohort::retireFailedSetup()const noexcept {
    for (std::size_t i=0;i<state_->frontiers.size();++i) {
        const auto& remote=state_->frozenRemotes[i];
        try { remote.original.worker->submitAndWait([&] { state_->frontiers[i]->retireFailedAdmission(remote.original.stream); }); }
        catch (...) { std::terminate(); }
    }
}
PooledImmutableCompleteCohort::~PooledImmutableCompleteCohort() noexcept {
    // This check occurs before State/member destruction. No callback, graph,
    // alias, root or worker can disappear while the original service is active.
    const auto boundary=state_->completion.read.publication.source.boundary;
    if (boundary->hasActiveInterval())std::terminate();
    // Final peer retirement is coordinated off-worker for the same reason as
    // construction: holding A while synchronously draining B can cycle with
    // an independent B-owned session draining A. Reject before any peer wait.
    if(state_->source.original.worker->ownsCurrentThread())std::terminate();
    for(const auto& remote:state_->frozenRemotes)if(remote.original.worker->ownsCurrentThread())std::terminate();
    // Join every actual last submission before destroying executable aliases.
    // This is final session retirement, never a token/request-success wait.
    retireFailedSetup();
    try {
        state_->source.original.worker->submitAndWait([&] {
            boundary->pooled_->setupRetirement->retireFailedAdmission(state_->source.original.stream);
        });
        const auto retire=[](State::FrozenParticipant& participant){
            if(!participant.executable->observeTerminal())throw std::logic_error("Joined complete pooled participant retained native readers");
            participant.executable->retireDefinition();participant.capture=nullptr;
        };
        retire(state_->source);for(auto& remote:state_->frozenRemotes)retire(remote);
    } catch (...) { std::terminate(); }
}
void PooledImmutableCompleteCohort::bindSourceSubmissionStream(const RetainedGraphParticipantParentStream& owner){
    owner.original().requireWorker(state_->source.original.worker);
    const auto boundary=ownership_->boundary.lock();
    if(!boundary || owner.original().memory().get()!=boundary->originalMemory_.get() || owner.stream()!=state_->source.original.stream)
        throw std::invalid_argument("Legacy component source cannot replace its original recorded stream");
}
void PooledImmutableCompleteCohort::awaitNativeTerminal(std::chrono::steady_clock::time_point deadline)const{
    for(;;){
        bool complete=state_->source.executable->observeTerminal();
        for(const auto& remote:state_->frozenRemotes)complete=remote.executable->observeTerminal()&&complete;
        if(complete)return;
        if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("Original pooled cohort missed whole-request native terminal");
        std::this_thread::yield();
    }
}
std::optional<PreparedGPUContextUsePermit> ImmutableLookupExecutionBoundary::retainedParticipantSource()const {
    if(!isPooled() || !completeCohort_.load(std::memory_order_acquire).lock())return std::nullopt;
    const auto& witness=pooledStorage()->retainedPreparedSourceWorkers();
    if(!witness)throw std::logic_error("Retained remote boundary lost its original source worker");
    return witness->coordinator();
}
void ImmutableLookupExecutionBoundary::bindSourceSubmissionStream(const RetainedGraphParticipantParentStream& owner){
    const auto authority=requireCompleteCohort();
    if(!authority)throw std::logic_error("Participant parent stream has no original distributed authority");
    authority->bindSourceSubmissionStream(owner);
}
void ImmutableLookupExecutionBoundary::bindRetainedParticipants(const std::shared_ptr<PooledImmutableRetainedParticipants>& owner){
    if(!owner || !pooled_ || hasActiveInterval() || !cohortOwnerClaim_.test(std::memory_order_acquire))
        throw std::logic_error("Recurring participant authority publication has no original cold claim");
    if(const auto prior=completeCohort_.load(std::memory_order_acquire).lock();prior && prior!=owner)
        throw std::logic_error("Original pooled source already has a different retained authority");
    owner->requireCurrent();completeCohort_.store(std::weak_ptr<PooledImmutableRetirementAuthority>{owner},std::memory_order_release);
}
void ImmutableLookupExecutionBoundary::bindCompleteCohort(const std::shared_ptr<PooledImmutableCompleteCohort>& cohort) {
    if (!cohort || !pooled_ || hasActiveInterval())throw std::logic_error("Invalid complete pooled cohort publication");
    if (const auto current=completeCohort_.load(std::memory_order_acquire).lock();current && current!=cohort)
        throw std::logic_error("Original pooled source already belongs to a retained complete cohort");
    completeCohort_.store(std::weak_ptr<PooledImmutableRetirementAuthority>{cohort},std::memory_order_release);
}
std::shared_ptr<PooledImmutableRetirementAuthority> ImmutableLookupExecutionBoundary::requireCompleteCohort()const {
    if (!pooled_)return {};
    const auto& p=pooled_->storage->geometry().parameters();
    auto cohort=completeCohort_.load(std::memory_order_acquire).lock();
    if (!cohort && (p.consumers>1 || p.coordinator.policy.participants()!=1 || p.local_owner==adaptive_lookup::None))
        throw std::logic_error("Distributed pooled admission requires its retained complete Read/Retirement cohort");
    if (cohort)cohort->requireCurrent();
    return cohort;
}

std::shared_ptr<ImmutableLookupExecutionBoundary> ImmutableLookupExecutionBoundary::create(
    std::shared_ptr<HostImmutableRowProvider> table,DeviceId device,std::shared_ptr<TensorBase> protocol,
    std::size_t maximumTokenRows,std::shared_ptr<PhysicalMemoryAuthority> memory,RetainedHostServiceUseToken worker)
{
    return createWorker(std::move(table),device,std::move(protocol),maximumTokenRows,std::move(memory),std::move(worker));
}
std::shared_ptr<ImmutableLookupExecutionBoundary> ImmutableLookupExecutionBoundary::create(
    std::shared_ptr<HostImmutableRowProvider> table,DeviceId device,std::shared_ptr<TensorBase> protocol,
    std::size_t maximumTokenRows,std::shared_ptr<PhysicalMemoryAuthority> memory,RetainedHostServiceBorrow worker)
{
    return createWorker(std::move(table),device,std::move(protocol),maximumTokenRows,std::move(memory),std::move(worker));
}
std::shared_ptr<ImmutableLookupExecutionBoundary> ImmutableLookupExecutionBoundary::createWorker(
    std::shared_ptr<HostImmutableRowProvider> table,DeviceId device,std::shared_ptr<TensorBase> protocol,
    std::size_t maximumTokenRows,std::shared_ptr<PhysicalMemoryAuthority> memory,
    std::variant<RetainedHostServiceUseToken,RetainedHostServiceBorrow> worker)
{
    (void)protocolState(protocol,device);
    if(!table||!memory||!device.is_gpu()||
        !maximumTokenRows||maximumTokenRows>std::numeric_limits<std::size_t>::max()/16||
        table->source().row_elements>static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Immutable lookup setup requires exact GPU/table/arena/PMA bindings and representable token geometry");
    const auto format=ggufToTensorType(table->source().format);
    const auto width=static_cast<int>(table->source().row_elements);const auto rowBytes=table->source().rowBytes();
    const auto layout=ImmutableLookupLayout::create(maximumTokenRows*16,rowBytes);
    auto service=std::visit([&](auto&& original){return ImmutableLookupService::create(table,layout,std::array{device},memory,
        table->retainedOwnerObservation().allocator().device,std::move(original));},std::move(worker));
    std::unique_ptr<IImmutableLookupKernels> kernels;
#ifdef HAVE_CUDA
    if(device.is_cuda())kernels=createCUDAImmutableLookupKernels(device,format,width,rowBytes);
#endif
#ifdef HAVE_ROCM
    if(device.is_rocm())kernels=createROCmImmutableLookupKernels(device,format,width,rowBytes);
#endif
    if(!kernels)throw std::runtime_error("Immutable lookup requested native GPU backend was not compiled");
    return std::make_shared<ImmutableLookupExecutionBoundary>(std::move(service),device,std::move(kernels),std::move(protocol),std::move(memory));
}
ImmutableLookupExecutionBoundary::ImmutableLookupExecutionBoundary(
    std::shared_ptr<ImmutableLookupService> service, DeviceId device,
    std::unique_ptr<IImmutableLookupKernels> kernels, std::shared_ptr<TensorBase> protocol,
    std::shared_ptr<PhysicalMemoryAuthority> originalMemory)
    : originalMemory_(std::move(originalMemory)),service_(std::move(service)), device_(device),
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
                                                                                 protocol_,originalMemory_);
}
std::shared_ptr<ImmutableLookupExecutionBoundary> ImmutableLookupExecutionBoundary::createPooled(
    std::shared_ptr<ImmutableLookupService> service,PooledImmutableSourceGeometry geometry,
    std::vector<std::shared_ptr<PooledImmutableOwnerPreparedStorage>> owners,std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,
    std::shared_ptr<PhysicalMemoryAuthority> authority,void* stream){
    return createPooledImpl(std::move(service),std::move(geometry),std::move(owners),std::move(coordinator),std::move(authority),stream,{},{});
}
std::shared_ptr<ImmutableLookupExecutionBoundary> ImmutableLookupExecutionBoundary::createPooledImpl(
    std::shared_ptr<ImmutableLookupService> service,PooledImmutableSourceGeometry geometry,
    std::vector<std::shared_ptr<PooledImmutableOwnerPreparedStorage>> owners,std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,
    std::shared_ptr<PhysicalMemoryAuthority> authority,void* stream,std::unique_ptr<NativeOrderingEventGroup> sourceGroup,
    std::unique_ptr<NativeOrderingEventGroup> boundaryGroup){
    if(isGraphCaptureActive() || !service || !authority || !stream || reinterpret_cast<std::uintptr_t>(stream)<=2)
        throw std::invalid_argument("Pooled source boundary requires its real admitted worker and exact ordinary setup stream");
    const auto device=geometry.allocator().device;auto source=service->bindSubordinate(device);
    auto originalMemory=authority;
    auto storage=source.preparePooledImpl(geometry,std::move(owners),std::move(coordinator),std::move(authority),stream,std::move(sourceGroup));
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
    auto retirement=std::make_shared<ImmutableLookupSubmissionRetirement>(device,stream,service,protocol,originalMemory,storage,std::shared_ptr<IWorkerGPUContext>{},std::move(boundaryGroup));
    kernels->initialize(*storage,stream);retirement->recordCompletion();
    Pooled pooled{std::move(storage),std::move(kernels),std::nullopt,std::move(retirement)};
    return std::shared_ptr<ImmutableLookupExecutionBoundary>(new ImmutableLookupExecutionBoundary(std::move(service),std::move(protocol),std::move(pooled),std::move(originalMemory)));
}
ImmutableLookupExecutionBoundary::ImmutableLookupExecutionBoundary(std::shared_ptr<ImmutableLookupService> service,
    std::shared_ptr<TensorBase> protocol,Pooled pooled,std::shared_ptr<PhysicalMemoryAuthority> originalMemory)
    :originalMemory_(std::move(originalMemory)),service_(std::move(service)),device_(pooled.storage->geometry().allocator().device),pooled_(std::move(pooled)),
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
ImmutableLookupExecutionBoundary::SourceGroupAdmission::SourceGroupAdmission(
    std::shared_ptr<ImmutableLookupService> service,ImmutableLookupService::GroupAdmission admission,
    std::vector<SourceAdmissionParticipant> participants)
    :service_(std::move(service)),participants_(std::move(participants)),admission_(std::move(admission))
{
    const auto& boundary = participants_.front().boundary;
    if (boundary->pooled_ && boundary->pooled_->storage->geometry().parameters().consumers > 1) {
        const auto retained = std::dynamic_pointer_cast<PooledImmutableRetainedParticipants>(boundary->requireCompleteCohort());
        if (!retained || !retained->transport()->readerGeometry())
            throw std::logic_error("Shared source admission requires its original complete reader transport");
        shared_reader_transport_ = retained->transport();
        shared_reader_transport_->beginInvocation({admission_.generation(), admission_.invocation()});
    }
}
ImmutableLookupExecutionBoundary::SourceGroupAdmission::~SourceGroupAdmission() noexcept
{
    try {
        // Abandonment cancels only CPU/reader progress. A missing source
        // proof must not erase storage release/SourceFrontier/Retired packets.
        // Original model/graph owners retain those channels and native duties.
        if (shared_reader_transport_)
            shared_reader_transport_->failureCallback({admission_.generation(), admission_.invocation()}, PooledImmutableDirectedTransport::CancellationEdges::ComputeReaders)();
        if (service_ && !finished_) (void)finish();
    } catch (...) { std::terminate(); }
}
bool ImmutableLookupExecutionBoundary::SourceGroupAdmission::reuses(
    std::span<const SourceAdmissionParticipant> participants)const
{
    if(finished_||participants.size()!=participants_.size())return false;
    const auto identity=[](const SourceAdmissionParticipant& member){
        return SourceAdmissionIdentity{member.boundary?member.boundary->device_:DeviceId::cpu(),
            member.boundary,member.request_epoch};
    };
    if(!sourceAdmissionReuses<SourceAdmissionParticipant>(participants_,participants,identity))return false;
    for(const auto& member:participants)
        if(!member.boundary||(!member.boundary->armed_.load(std::memory_order_acquire)&&
           !member.boundary->pending_group_admission_))return false;
    return true;
}
std::function<void()> ImmutableLookupExecutionBoundary::SourceGroupAdmission::failureCallback() const
{
    auto source = service_->failureCallback(admission_);
    if (!shared_reader_transport_) return source;
    auto channels = shared_reader_transport_->failureCallback({admission_.generation(), admission_.invocation()}, PooledImmutableDirectedTransport::CancellationEdges::ComputeReaders);
    return [source = std::move(source), channels = std::move(channels)] {
        std::exception_ptr failure;
        try { source(); } catch (...) { failure = std::current_exception(); }
        try { channels(); } catch (...) { if (!failure) failure = std::current_exception(); }
        if (failure) std::rethrow_exception(failure);
    };
}
bool ImmutableLookupExecutionBoundary::SourceGroupAdmission::finish()
{
    if(finished_)return result_;
    result_=service_->finishGroup(admission_);
    // A staged channel has an adopted CPU ticket but no native admission. Consume
    // only that right; actual admitted boundaries retain their own native close.
    for(auto& participant:participants_){
        auto& pending=participant.boundary->pending_group_admission_;
        if(pending){(void)service_->await(pending->ticket);pending.reset();}
    }
    if (shared_reader_transport_ && !result_)
        shared_reader_transport_->failureCallback({admission_.generation(), admission_.invocation()}, PooledImmutableDirectedTransport::CancellationEdges::ComputeReaders)();
    finished_=true;return result_;
}
void ImmutableLookupExecutionBoundary::SourceGroupAdmission::cancelStorageChannelsAfterSourceProof()
{
    if (shared_reader_transport_)
        shared_reader_transport_->failureCallback({admission_.generation(), admission_.invocation()},
            PooledImmutableDirectedTransport::CancellationEdges::StorageOwners)();
}
bool ImmutableLookupExecutionBoundary::SourceGroupAdmission::finishNativeChannelsAfterProof()
{
    if (!finished_)
        throw std::logic_error("Shared reader native retirement preceded original CPU/adopted-right cleanup");
    if (!shared_reader_transport_) return result_;
    const auto outcome = shared_reader_transport_->finishInvocation({admission_.generation(), admission_.invocation()});
    result_ = result_ && outcome == CapturedTransferInvocationRetirement::Completed;
    shared_reader_transport_.reset();
    return result_;
}
std::unique_ptr<ImmutableLookupExecutionBoundary::SourceGroupAdmission>
ImmutableLookupExecutionBoundary::armSourceGroup(std::span<const SourceAdmissionParticipant> participants)
{
    if(participants.empty()||!participants.front().boundary)
        throw std::invalid_argument("Immutable lookup group requires its original participants");
    const auto service=participants.front().boundary->service_;
    const auto devices=service->channelDevices();
    if(devices.empty()||devices.size()!=participants.size()||
       (devices.size()==1&&!participants.front().boundary->requiresSourceGroupAdmission()))
        throw std::invalid_argument("Immutable lookup group requires every declared source channel");
    const auto previous=participants.front().boundary->intervalGeneration_;
    if(previous==~std::uint64_t{0})throw std::overflow_error("Immutable lookup group interval exhausted");
    std::vector<SourceAdmissionIdentity> identities;identities.reserve(participants.size());
    for(const auto& participant:participants){
        if(!participant.boundary)throw std::invalid_argument("Original source group has a null participant");
        identities.push_back({participant.boundary->device_,participant.boundary,participant.request_epoch});
    }
    const auto order=sourceAdmissionOrder(devices,identities);
    std::vector<SourceAdmissionParticipant> retained;retained.reserve(participants.size());
    for(const auto index:order)retained.push_back(participants[index]);
    std::vector<ImmutableLookupServiceStats> starts;starts.reserve(participants.size());
    for(std::size_t i=0;i<participants.size();++i){
        const auto& member=retained[i];const auto& boundary=member.boundary;
        if(!boundary||!member.request_epoch||boundary->device_!=devices[i]||
           boundary->service_!=service||boundary->service_.owner_before(service)||service.owner_before(boundary->service_)||
           boundary->intervalGeneration_!=previous||boundary->armed_.load(std::memory_order_acquire)||
           boundary->pending_group_admission_)
            throw std::logic_error("Immutable lookup group replaced an original channel, epoch or interval");
        boundary->requireProtocolOwner();
        starts.push_back(service->stats(devices[i]));
    }
    auto result=std::unique_ptr<SourceGroupAdmission>(new SourceGroupAdmission(
        service,service->armGroup(previous+1),std::move(retained)));
    for(std::size_t i=0;i<participants.size();++i)
        result->participants_[i].boundary->pending_group_admission_=PendingGroupAdmission{
            service->adoptGroupChannel(result->admission_,devices[i]),result->participants_[i].request_epoch,starts[i]};
    return result;
}
void ImmutableLookupExecutionBoundary::beginAdmission(std::uint64_t generation,void *stream)
{
    requireProtocolOwner();
    const auto cohort=requireCompleteCohort();
    struct TemporaryLegacyClaim {
        std::atomic_flag* claim=nullptr;
        ~TemporaryLegacyClaim(){if(claim)claim->clear(std::memory_order_release);}
    } legacy;
    if(pooled_ && !cohort){
        if(cohortOwnerClaim_.test_and_set(std::memory_order_acquire))
            throw std::logic_error("Pooled legacy admission collided with original cohort construction/retirement");
        legacy.claim=&cohortOwnerClaim_;
    }
    if (cohort)cohort->requireSourceStream(stream);
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
    if(requiresSourceGroupAdmission()){
        if(!pending_group_admission_||pending_group_admission_->request_epoch!=generation||
           pending_group_admission_->ticket.generation!=interval)
            throw std::logic_error("Immutable lookup channel requires its original parent group admission");
        intervalStart_=pending_group_admission_->start;
    }else intervalStart_=service_->stats(device_);
    terminal_.reset();
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
    if(requiresSourceGroupAdmission()){
        ticket_=pending_group_admission_->ticket;pending_group_admission_.reset();
    }else ticket_=service_->arm(interval);
    if (!pooled_)
        admissionRetirement_->beginSubmission(stream);
    try {
        if (pooled_) {
            pooled_->storage->acquireSetup(stream);
            requireCohortInputs(pooled_->storage->roots(),device(),stream);
            requireCohortInputs(pooled_->storage->coordinator()->roots(),device(),stream);
            if (pooled_->storage->geometry().parameters().local_owner!=adaptive_lookup::None)
                requireCohortInputs(pooled_->storage->owner()->roots(),device(),stream);
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
        if (pooled_) {
            pooled_->setupRetirement->retireFailedAdmission(stream);
        } else
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
    const auto cohort=requireCompleteCohort();
    requireProtocolOwner();
    service_->requireHealthy();
    if(!armed_.load(std::memory_order_acquire))throw std::logic_error("Immutable lookup graph launch has no armed retained-generation storage interval");
}
void ImmutableLookupExecutionBoundary::recordTerminalClose(void *stream,const int *healthy,const int *complete) const
{
    const auto cohort=requireCompleteCohort();
    if (cohort)cohort->requireSourceStream(stream);
    requireProtocolOwner();
    if (pooled_) {
        if(!pooled_->control)throw std::invalid_argument("Pooled source terminal requires its original canonical controller");
        if(!healthy && !complete) {
            if (cohort)pooled_->kernels->closeForReset(*pooled_->storage,*pooled_->control,stream,*cohort);
            else pooled_->kernels->closeForReset(*pooled_->storage,*pooled_->control,stream);
        }
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
    // Lock lifetime before joining, but never let metadata drift bypass the
    // original failed-worker retirement. Validation follows the actual join.
    const auto originalCohort=completeCohort_.load(std::memory_order_acquire).lock();
    // A failed I/O lane still owes its worker join; launch admission's health
    // check must never bypass terminal retirement and its precise diagnostic.
    if(!armed_.load(std::memory_order_acquire))throw std::logic_error("Immutable lookup terminal has no armed storage interval");
    const auto nativeDeadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    const bool success=service_->await(ticket_);ticket_={};
    terminal_=RetainedStorageTerminalObservation::between(intervalGeneration_,service_->mode(),intervalStart_,service_->stats(device_));
    if (pooled_)
        terminal_->transport = RetainedStorageTransport::PooledSubordinate;
    // Validation cannot bypass the original CPU/adopted join. It must still
    // reject an expired required cohort or a replaced original ownership block
    // before inactive state or safe-retired failure can be published.
    const auto cohort = requireCompleteCohort();
    if (cohort != originalCohort || cohort.owner_before(originalCohort) ||
        originalCohort.owner_before(cohort))
        throw std::logic_error("Immutable lookup terminal replaced its original retained cohort");
    if(cohort){cohort->awaitNativeTerminal(nativeDeadline);cohort->requireCurrent();}
    // Keep the original native duty armed if its proof fails. A consumed CPU
    // ticket/terminal statistic alone never authorizes retry or safe teardown.
    armed_.store(false,std::memory_order_release);
    if(!success)throw RetiredStorageRequestFailure("Immutable lookup retained-generation storage failed: "+service_->lastFailure());
}
RetainedStorageTerminalObservation ImmutableLookupExecutionBoundary::terminalObservation() const
{
    if(armed_.load(std::memory_order_acquire)||!terminal_)
        throw std::logic_error("Immutable lookup terminal statistics require this interval's completed worker join");
    return *terminal_;
}
} // namespace llaminar2::qwen4exp
