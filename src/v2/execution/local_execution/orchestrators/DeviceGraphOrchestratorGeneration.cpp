/**
 * @file DeviceGraphOrchestratorGeneration.cpp
 * @brief Ordinary captured generation and terminal request retirement using the existing request/controller owner.
 *
 * Prefill sampling consumes no KV row. Each subsequent retained transaction
 * consumes the pending device token, then publishes one response and its next
 * frontier. No host token, speculative verifier or fake forward participates.
 * Sampler recordings borrow arena storage; their parent retires first. HIP
 * retains one complete transaction, never a host-submitted list of model stages.
 */
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "execution/qwen4exp/PooledImmutableRetainedParticipants.h"
#include "execution/qwen4exp/PooledImmutableLocalHotRequestPublication.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotKernels.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotPreparedStorage.h"
#include "models/qwen4exp/Qwen4ExpReplicatedPLECohort.h"
#include "interfaces/IModelContext.h"
#include "execution/local_execution/graph/IRetainedGraphParticipants.h"
#include "DeviceGraphOrchestrator.h"
#include "models/qwen4exp/Qwen4ExpConfig.h"
#include "PipelineForwardGraphEdges.h"
#include "backends/BackendManager.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/MoEOverlayInferenceTransactionService.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "execution/moe/DeviceMoEOverlayEpochArena.h"
#include "execution/mtp/DeviceGenerationGraphProgram.h"
#include "utils/Logger.h"
#include <exception>
#include <stdexcept>

namespace llaminar2
{
namespace {
/** @brief Submit one request-boundary native storage mutation to its original worker.
 * The caller retains coordination and terminal service joins. Local boundaries
 * without remote participants preserve their existing explicit native contract.
 * @param boundary Existing immutable I/O owner, never a replacement cohort.
 * @param memory Original DGO admitted authority and ownership control block.
 * @param device Exact source participant, independent from remote storage owners.
 * @param publication Existing request admission check or reset close publication only.
 * @throws std::invalid_argument For foreign original device/authority.
 * @throws std::logic_error For stale provenance or failed worker entry.
 * @note submitAndWait joins host enqueue completion, never GPU execution.
 * Whole-request event waits and CPU I/O joins remain with the caller.
 */
template<class Publication>
void submitRetainedStoragePublication(const std::shared_ptr<IRetainedGraphStorageBoundary>& boundary,
    const std::shared_ptr<PhysicalMemoryAuthority>& memory,DeviceId device,Publication&& publication)
{
    const auto original=boundary->retainedParticipantSource();
    if(!original){publication();return;}
    const auto expected=original->memory();
    if(device!=boundary->device() || original->device()!=device || !memory ||
       memory.get()!=expected.get() || memory.owner_before(expected) || expected.owner_before(memory) ||
       memory->phase()!=PhysicalMemoryAuthorityPhase::Admitted)
        throw std::invalid_argument("Retained storage publication requires the original participant/PMA control block");
    const auto current=GPUDeviceContextPool::instance().retainExistingContextProvenance(device);
    original->requireWorker(current.worker());
    if(original->generation()!=current.generation())
        throw std::logic_error("Retained storage publication has a stale original source generation");
    const auto& worker=original->worker();
    const auto publish=[&]{
        if(!worker->ownsCurrentThread())throw std::logic_error("Retained storage publication did not enter its original worker");
        publication();
    };
    if(worker->ownsCurrentThread())publish();
    else worker->submitAndWait(publish);
}
}
std::size_t DeviceGraphOrchestrator::sparseQueryProgramCount() const
{
    if (!graph_builder_ || !graph_builder_->config().flash_next) return 1u;
    const auto &model = *graph_builder_->config().flash_next;
    const auto &mixers = model.mixers();
    if (model.mainLayers() > mixers.size())
        throw std::logic_error("Sparse-query program inventory lost validated main-layer geometry");
    return std::any_of(mixers.begin(), mixers.begin() + model.mainLayers(),
        [](auto mixer) { return mixer == Qwen4ExpMixer::SparseAttention; }) ? kSparseQueryExecutionProgramCount : 1u;
}

void DeviceGraphOrchestrator::admitSparseQueryProgram(const DeviceGenerationAdmissionRequest &admission,
    SparseQueryAdmissionBoundary boundary)
{
    if (!admission.valid() || compiling_sparse_query_program_ || sparseQueryParentLaunched())
        throw std::logic_error("Sparse-query program selection requires a valid idle generation admission");
    // A local forced controller has no response budget authority. Preserve
    // the already-proved outer host response rather than deriving a program
    // from its budget0. Standalone proved forced admissions resolve P+1 in
    // the same common contract; missing proof retains the complete program.
    if (boundary == SparseQueryAdmissionBoundary::ResidentGeneration &&
        admission.depth_policy.isForwardOnly() && admitted_sparse_query_program_ &&
        admitted_sparse_query_program_->boundary == SparseQueryAdmissionBoundary::HostControlledResponse)
        return;
    admitted_sparse_query_program_ = SparseQueryProgramAdmission{
        sparseQueryProgramCount() == 1u ? SparseQueryExecutionProgram::RankingPossible :
            resolveSparseQueryExecutionProgram(admission), boundary};
}

bool DeviceGraphOrchestrator::admitSparseQueryExecutionProgram(const DeviceGenerationAdmissionRequest &admission)
{
    if (!admission.valid() || device_generation_storage_.active_request_count != 0 ||
        active_device_generation_admission_ || compiling_sparse_query_program_ || sparseQueryParentLaunched() ||
        (admitted_sparse_query_program_ &&
         admitted_sparse_query_program_->boundary == SparseQueryAdmissionBoundary::HostControlledResponse))
        return false;
    admitSparseQueryProgram(admission, SparseQueryAdmissionBoundary::HostControlledResponse);
    return true;
}

void DeviceGraphOrchestrator::endSparseQueryExecutionProgram() noexcept
{
    if (admitted_sparse_query_program_ &&
        admitted_sparse_query_program_->boundary == SparseQueryAdmissionBoundary::HostControlledResponse)
        admitted_sparse_query_program_.reset();
}

std::optional<ForwardGraphSignature> &DeviceGraphOrchestrator::pairedVerifierForwardSignature(
    MTPVerifierPreparationGraphCache &cache) const
{
    return cache.paired_forward_signatures[sparseQueryExecutionProgramIndex(sparseQueryGraphProgram())];
}

std::pair<std::shared_ptr<IRetainedGraphStorageBoundary>,std::uint64_t>
DeviceGraphOrchestrator::retainedStorageSourceAdmission() const
{
    const auto declared=graph_builder_?graph_builder_->retainedStorageBoundary(state_.device_id):nullptr;
    if(retained_storage_boundary_&&declared!=retained_storage_boundary_)
        throw std::logic_error("Original storage admission declaration changed");
    return {declared,session_epoch_};
}

bool DeviceGraphOrchestrator::sharedPooledQueryOnlyReader() const
{
    if (!graph_builder_) return false;
    const auto& graph = graph_builder_->config();
    const auto& placement = graph.ngram_table_placement;
    if (!placement || !placement->isGPUAdaptive() ||
        placement->gpuAdaptiveLayout().consumers().size() <= 1 ||
        placement->gpuAdaptiveRequest().placement.mode != AdaptiveGPUCachePlacementMode::DeclaredRankLocal)
        return false;
    const auto& declaration = placement->gpuAdaptiveRequest().placement;
    if (!declaration.coordinator)
        throw std::logic_error("Shared reader terminal lost its authored source coordinator");
    if (declaration.coordinator->device() == state_.device_id) return false;
    const auto& cohort = graph.ngram_replicated_ple_cohort;
    if (!cohort || !injected_model_ctx_ || cohort->modelOwner() != injected_model_ctx_.get() ||
        cohort->placement() != *placement || &cohort->context() != graph.tp_ctx)
        throw std::logic_error("Shared reader terminal lost its original Rank/model/native collective");
    cohort->requireConsumer(state_.device_id, graph.tp_device_idx);
    const auto original = injected_model_ctx_->adaptiveImmutableLookupParticipants(state_.device_id);
    if (!original || !original->transport()->readerGeometry() ||
        retained_storage_boundary_ || graph_builder_->retainedStorageBoundary(state_.device_id))
        throw std::logic_error("Shared reader terminal cannot borrow or invent a CPU source boundary");
    original->requireForPolicy(graph);
    return true;
}

bool DeviceGraphOrchestrator::finishGroupedDeviceResidentGeneration(DeviceGenerationTerminalResult* result)
{
    const auto boundary=std::dynamic_pointer_cast<qwen4exp::ImmutableLookupExecutionBoundary>(retained_storage_boundary_);
    if (boundary && boundary->requiresSourceGroupAdmission())
        return finishDeviceResidentGenerationImpl(result,RetainedStorageTerminalPhase::PublishOnly);
    if (!sharedPooledQueryOnlyReader())
        throw std::logic_error("Grouped terminal materialization requires its original source or query-only reader");
    // No CPU close belongs to this participant. Its existing terminal bridge
    // still records/joins the exact native endpoint completion before returning.
    return finishDeviceResidentGenerationImpl(result,RetainedStorageTerminalPhase::PublishAndAwait);
}
void DeviceGraphOrchestrator::awaitGroupedRetainedStorageTerminal()
{
    const auto [declared, epoch] = retainedStorageSourceAdmission();
    (void)epoch;
    const auto boundary = std::dynamic_pointer_cast<qwen4exp::ImmutableLookupExecutionBoundary>(declared);
    if (!boundary) {
        if (!sharedPooledQueryOnlyReader())
            throw std::logic_error("Grouped terminal await lost its original query-only reader");
        // Rank reaches this private aggregation edge only after every ordinary
        // terminal bridge, or after its reset producer-join phase below. Both
        // record this same original event for the current invocation. No CPU
        // service close, zero byte count or cancellation flag supplies proof.
        auto* backend = getBackendFor(state_.device_id);
        if (!backend || !backend->setDevice(state_.device_id.ordinal) ||
            !device_generation_terminal_host_ready_event_ ||
            !backend->waitForEvent(device_generation_terminal_host_ready_event_.get(), state_.device_id.ordinal))
            throw std::runtime_error("Shared reader original native terminal completion is unproven");
        retireLocalHotRequestAdmissionAfterTerminal();
        return;
    }
    if (!boundary->requiresSourceGroupAdmission())
        throw std::logic_error("Grouped terminal await requires its original grouped source boundary");
    if(!boundary->hasActiveInterval()){
        retained_storage_reset_close_prepared_=false;
        return;
    }
    if (retained_storage_boundary_ != boundary)
        throw std::logic_error("Active grouped terminal lost its original admitted boundary");
    try{awaitRetainedStorageTerminal();}
    catch(...){
        if(!boundary->hasActiveInterval())retained_storage_reset_close_prepared_=false;
        throw;
    }
    retained_storage_reset_close_prepared_=false;
}

void DeviceGraphOrchestrator::prepareGroupedRetainedStorageReset(const InferenceStateResetRequest& request)
{
    prepareGroupedRetainedStorageProducerProof(request.reason);
}
void DeviceGraphOrchestrator::prepareGroupedRetainedStorageProducerProof(const char* reason)
{
    const auto [declared,epoch]=retainedStorageSourceAdmission();
    (void)epoch;
    const auto boundary=std::dynamic_pointer_cast<qwen4exp::ImmutableLookupExecutionBoundary>(declared);
    const bool reader = !boundary && sharedPooledQueryOnlyReader();
    if (!reader && (!boundary || !boundary->requiresSourceGroupAdmission() || !boundary->hasActiveInterval())) return;
    auto* backend=getBackendFor(state_.device_id);
    if(!backend||!backend->setDevice(state_.device_id.ordinal))
        throw std::runtime_error("Original grouped reset could not activate its exact GPU");
    // Preserve the original reset order. This drains only already-published
    // maintenance diagnostics; the later reset sees the same owner retired.
    drainCompletedDeviceMoERebalanceMaintenanceDiagnostics("request_reset",
        reason?reason:"request-boundary");
    auto* stream=explicitGPUStreamForOperation("requestStateResetTransaction");
    if(!joinPriorDeviceWorkForRequestStateReset(stream,"grouped-source-terminal",RetainedStorageTerminalPhase::PublishOnly))
        throw std::runtime_error("Original grouped reset could not join prior GPU producers");
    if (reader) {
        // The same original reset stream has joined all queued producers. An
        // unlaunched peer has no work to fabricate; recording its existing
        // event here proves this exact frontier before any channel is retired.
        if (!device_generation_terminal_host_ready_event_ ||
            !backend->recordEvent(device_generation_terminal_host_ready_event_.get(), state_.device_id.ordinal, stream) ||
            !backend->waitForEvent(device_generation_terminal_host_ready_event_.get(), state_.device_id.ordinal))
            throw std::runtime_error("Shared reader reset could not prove its original native endpoint frontier");
    }
}

void DeviceGraphOrchestrator::prepareRetainedStorageBinding()
{
    const auto declared=graph_builder_?graph_builder_->retainedStorageBoundary(state_.device_id):nullptr;
    if(retained_storage_boundary_&&declared!=retained_storage_boundary_)
        throw std::logic_error("Immutable storage lane changed while captured generation retained its addresses");
    if(declared)
    {
        if(declared->device()!=state_.device_id)
            throw std::logic_error("Immutable storage admission names another GPU participant");
        if(!device_generation_storage_.control_device)
            throw std::logic_error("Immutable storage binding requires its original canonical controller storage");
        declared->bindGenerationControl({device_generation_storage_.control_device+sampling_math::kDeviceGenerationControlOk,
            device_generation_storage_.control_device+sampling_math::kDeviceGenerationControlErrorCode});
    }
    // Only successful acquisition publishes a retirement duty. A foreign or
    // throwing bind never grants this runner ownership of somebody else's borrow.
    retained_storage_boundary_=declared;
}
void DeviceGraphOrchestrator::retireRetainedStorageGenerationBinding()noexcept
{
    // The original producer and every native parent have already retired. Drop
    // the local admission's controller borrow before the arena can be reused,
    // including query-only readers that own no CPU source boundary.
    retained_local_hot_admission_.reset();
    if(!retained_storage_boundary_)return;
    try{
        if(!device_generation_storage_.control_device)
            throw std::logic_error("Original storage graph borrow lost its canonical controller storage");
        retained_storage_boundary_->retireGenerationControl({
            device_generation_storage_.control_device+sampling_math::kDeviceGenerationControlOk,
            device_generation_storage_.control_device+sampling_math::kDeviceGenerationControlErrorCode});
        retained_storage_boundary_.reset();
    }catch(const std::exception& error){
        LOG_ERROR("[DeviceGraphOrchestrator] Original storage graph-borrow retirement failed: "<<error.what());
        std::terminate();
    }
}
bool DeviceGraphOrchestrator::beginRetainedStorageAdmission(void *stream)
{
    prepareRetainedStorageBinding();
    if(retained_storage_boundary_){
        // Separate normal prefill/forward entries can revisit this admission.
        // The boundary's same-epoch check prevents a second arm/native admit,
        // but this wrapper still joins one host enqueue task on each entry.
        // Retained token/transaction replay bypasses this normal-forward prelude.
        const auto boundary=retained_storage_boundary_;
        submitRetainedStoragePublication(boundary,physical_memory_authority_,state_.device_id,
            [&]{boundary->beginAdmission(session_epoch_,stream);});
    }
    beginLocalHotRequestAdmission(stream);
    return true;
}

/** @brief Exact original request dependencies; mutable cache and source state remain on device. */
struct DeviceGraphOrchestrator::LocalHotRequestAdmission final {
    std::shared_ptr<qwen4exp::PooledImmutableRetainedParticipants> participants;
    std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage> local;
    std::shared_ptr<const qwen4exp::IPooledImmutableLocalHotKernels> code;
    DeviceGenerationControlBinding control;
    PreparedGPUContextUsePermit original;
    qwen4exp::PooledImmutableLocalHotRequestPublication publication;

    /** @brief Retain the actual original model/worker/controller owners before request submission. */
    LocalHotRequestAdmission(std::shared_ptr<qwen4exp::PooledImmutableRetainedParticipants> owner,
        std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage> storage,
        DeviceGenerationControlBinding controller)
        : participants(std::move(owner)), local(std::move(storage)), control(controller),
          original(participants->originalParticipant(local->geometry().allocator().device))
    {
        const auto device = original.device();
#ifdef HAVE_CUDA
        if (device.is_cuda()) code = qwen4exp::createCUDAPooledImmutableLocalHotKernels();
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) code = qwen4exp::createROCmPooledImmutableLocalHotKernels();
#endif
        if (!code) throw std::invalid_argument("Local-cache request admission requires its compiled native backend");
    }
};

std::size_t DeviceGraphOrchestrator::localHotTerminalMetadataBytes(int requests)const
{
    const auto placement=graph_builder_?graph_builder_->config().ngram_table_placement:nullptr;
    if(!placement || !placement->isGPUAdaptive() || !placement->gpuAdaptiveRequest().local_hot_cache)return 0;
    return qwen4exp::local_hot::terminalObservationBytes(requests,sampling_math::kDeviceGenerationControlCount);
}
void DeviceGraphOrchestrator::packLocalHotTerminalObservation(int requests,void* destination,std::size_t bytes,void* stream)
{
    if(!retained_local_hot_admission_ || bytes!=localHotTerminalMetadataBytes(requests))
        throw std::logic_error("Local terminal observation has no actual admitted request/local owner");
    auto& original=*retained_local_hot_admission_;
    original.local->requireAuthority(physical_memory_authority_);
    const auto output=arena_->getSharedTensor(BufferId::MTP_LOCAL_HOT_TERMINAL_OBSERVATION);
    if(!output || output->gpu_data_ptr()!=destination)
        throw std::logic_error("Local terminal observation changed its actual separate arena owner");
    const auto enqueue=[&]{original.code->packTerminal(*original.local,original.control,requests,
        qwen4exp::PooledNativeTransferSpan(output,0,bytes),stream);};
    if(original.original.worker()->ownsCurrentThread())enqueue();
    else original.original.worker()->submitAndWait(enqueue);
}
void DeviceGraphOrchestrator::publishLocalHotTerminalObservation(const void* packed,std::size_t bytes,int requests)
{
    if(!packed || !retained_local_hot_admission_ || bytes!=localHotTerminalMetadataBytes(requests))
        throw std::logic_error("Local terminal observation changed its complete immutable extent");
    qwen4exp::local_hot::TerminalObservation current;
    std::memcpy(&current,packed,sizeof(current));
    auto& original=*retained_local_hot_admission_;
    const auto publication=original.local->observeCompletedTerminal(current);
    const auto& delta=publication.delta;
    for(unsigned i=0;i<15;++i)
        PerfStatsCollector::addCounterWithSequence("ngram_lookup",qwen4exp::local_hot::TerminalMetricNames[i],
            static_cast<double>(delta.values[i]),{delta.values[i]},"generation_terminal",state_.device_id.toString(),
            {{"authority","completed_local_hot_terminal"},{"storage_identity",std::to_string(original.local->consumerStorageIdentity())},
             {"unit",i<4?"original_l1_ids":i<10?"native_rows":"bytes"},{"wire_attribution","receive_completed_only"},
             {"window",publication.since_storage_initialization?"since_original_storage_initialization":"between_completed_terminals"}});
}

void DeviceGraphOrchestrator::beginLocalHotRequestAdmission(void* stream)
{
    const auto placement = graph_builder_ ? graph_builder_->config().ngram_table_placement : nullptr;
    const bool selected = placement && placement->isGPUAdaptive() && placement->gpuAdaptiveRequest().local_hot_cache.has_value();
    if (!selected) {
        if (retained_local_hot_admission_)
            throw std::logic_error("Local-cache request admission changed its original placement policy");
        return;
    }
    if (!stream || reinterpret_cast<std::uintptr_t>(stream) <= 2 || isGraphCaptureActive() ||
        !injected_model_ctx_ || !arena_)
        throw std::logic_error("Local-cache request admission requires its original outside-capture model/arena/stream");
    const auto local = injected_model_ctx_->adaptiveImmutableLookupLocalHot(state_.device_id);
    const auto participants = injected_model_ctx_->adaptiveImmutableLookupParticipants(state_.device_id);
    if (!local || !participants || !participants->localHotGeometry() ||
        local->originalSource() != participants->boundary()->pooledStorage())
        throw std::logic_error("Local-cache request admission lost its original local/shared owners");
    local->requireAuthority(physical_memory_authority_);
    participants->requireForPolicy(graph_builder_->config());
    const auto control = DeviceGenerationControlBinding::admit(*arena_, state_.device_id, 0);
    if (!retained_local_hot_admission_)
        retained_local_hot_admission_ = std::make_shared<LocalHotRequestAdmission>(participants, local, control);
    auto& admission = *retained_local_hot_admission_;
    const auto same = [](const auto& a, const auto& b) {
        return a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a);
    };
    if (!same(admission.participants, participants) || !same(admission.local, local) ||
        admission.control.captureIdentity() != control.captureIdentity() ||
        !same(admission.original.memory(), physical_memory_authority_))
        throw std::logic_error("Local-cache request admission changed original storage/controller/PMA identity");
    const auto current = GPUDeviceContextPool::instance().retainExistingContextProvenance(state_.device_id);
    admission.original.requireWorker(current.worker());
    if (admission.original.generation() != current.generation())
        throw std::logic_error("Local-cache request admission changed its original prepared worker generation");
    const auto submit = [&] {
        if (!admission.original.worker()->ownsCurrentThread())
            throw std::logic_error("Local-cache request admission did not enter its original participant worker");
        const auto frame = local->roots().frame;
        admission.publication.enter(session_epoch_, [&] {
            // Initial storage and reset publish this exact controller root on
            // their producer stream. Admission precedes the ordinary forward
            // state wait, so its input edge must exist before any health read.
            // Binding an address alone supplies no publication dependency.
            TransferEngine::requireDeviceInput(control.owner(), state_.device_id, stream);
        }, [&] {
            // The source DGO has enqueued its native source admission above.
            // Every other DGO receives that fixed grant on its own worker. The
            // enclosing Rank submits participants together; no GPU wait occurs
            // here. The device reset keeps warmed rows and validates retirement.
            participants->transport()->enqueueLocalHotAdmission(local, admission.code, control, stream);
            TransferEngine::publishDeviceWrite(frame, state_.device_id, stream);
        }, [&] {
            // Reuse the admitted Tensor publication event. This also orders a
            // later prefill/condition stream without another source round trip.
            TransferEngine::requireDeviceInput(frame.get(), state_.device_id, stream);
        });
    };
    if (admission.original.worker()->ownsCurrentThread()) submit();
    else admission.original.worker()->submitAndWait(submit);
}

void DeviceGraphOrchestrator::retireLocalHotRequestAdmissionAfterTerminal()
{
    if (retained_local_hot_admission_)
        retained_local_hot_admission_->publication.retireAfterTerminal(session_epoch_);
}

std::span<const DeviceControlledLoopEntryFragment> DeviceGraphOrchestrator::retainedStorageTerminalFragments()
{
    if(!retained_storage_boundary_)return {};
    void *stream=deviceGenerationLoopGraph().stream.get();
    if(!stream)throw std::logic_error("Immutable storage terminal requires its exact generation stream");
    if(retained_storage_terminal_capture_&&retained_storage_terminal_capture_->executionStream()!=stream)
        throw std::logic_error("Immutable storage terminal stream changed before retained graph retirement");
    if(!retained_storage_terminal_capture_)
    {
        auto &worker=GPUDeviceContextPool::instance().getContext(state_.device_id);
        retained_storage_terminal_capture_.bind(deviceGenerationLoopGraph().stream,worker.createGraphCapture(stream));
        ScopedBackendGraphCapture recording(worker,*retained_storage_terminal_capture_.get(),"immutable_storage_generation_close");
        if(!recording.begin())throw std::runtime_error("Immutable storage terminal recording failed");
        retained_storage_boundary_->recordTerminalClose(stream);recording.finish();
        if(!retained_storage_terminal_capture_->instantiate())
            throw std::runtime_error("Immutable storage terminal executable materialization failed");
        retained_storage_terminal_fragment_[0]={"immutable storage terminal close",retained_storage_terminal_capture_.get()};
    }
    return retained_storage_terminal_fragment_;
}

bool DeviceGraphOrchestrator::submitRetainedStorageTerminal()
{
    if(!retained_storage_boundary_)return true;
    retained_storage_boundary_->requireArmedForLaunch();
    if(!retained_storage_terminal_capture_||!retained_storage_terminal_capture_->hasExecutable())
        throw std::logic_error("Hosted terminal ticket names an unprepared immutable storage close branch");
    return retained_storage_terminal_capture_->launchOnStream(deviceGenerationLoopGraph().stream.get());
}
void DeviceGraphOrchestrator::releaseRetainedStorageGenerationTopology() noexcept
{
    for (auto &parent : sparse_query_generation_parents_) parent.release();
    compiling_sparse_query_program_.reset();
    admitted_sparse_query_program_.reset();
    // The terminal owner retained the original stream while the parent and its
    // child family retired. Its own recording now retires before the last stream.
    retained_storage_terminal_fragment_={};
    retained_storage_terminal_capture_.reset();
}
void DeviceGraphOrchestrator::retireRetainedStorageForRequestReset(void *stream,RetainedStorageTerminalPhase phase)
{
    if(!retained_storage_boundary_ || !retained_storage_boundary_->hasActiveInterval())return;
    // Terminal aggregation can query other participant workers and join the
    // CPU I/O service. Keep that whole-request obligation off the source worker.
    if(const auto original=retained_storage_boundary_->retainedParticipantSource();
       original && original->worker()->ownsCurrentThread())
        throw std::logic_error("Retained storage request retirement requires its off-worker terminal coordinator");
    if(retained_storage_reset_close_prepared_){
        if(phase==RetainedStorageTerminalPhase::PublishAndAwait){awaitRetainedStorageTerminal();retained_storage_reset_close_prepared_=false;}
        return;
    }
    auto *backend=getBackendFor(state_.device_id);
    if(!state_.device_id.is_gpu() || !backend || !stream ||
        reinterpret_cast<std::uintptr_t>(stream)<=2 || !device_generation_terminal_host_ready_event_)
        throw std::logic_error("Immutable storage reset terminal requires its exact GPU stream and setup-owned terminal event");
    // All prior forward/publication producers have joined this reset stream.
    // A standalone prefill has no complete generation parent to publish close;
    // this is its terminal request boundary, before protocol/session mutation.
    // The DGO's request API owns this event exclusively: response observation
    // and reset cannot overlap. Reusing it creates no new allocation or transfer.
    // Do not requireHealthy here: a failed storage worker still owes its join.
    const auto boundary=retained_storage_boundary_;
    submitRetainedStoragePublication(boundary,physical_memory_authority_,state_.device_id,
        [&]{boundary->recordTerminalClose(stream);});
    PerfStatsCollector::addCounter("immutable_lookup","reset_terminal_closes",1.0,
        "request_reset",state_.device_id.toString());
    if(!backend->recordEvent(device_generation_terminal_host_ready_event_.get(),state_.device_id.ordinal,stream) ||
        !backend->waitForEvent(device_generation_terminal_host_ready_event_.get(),state_.device_id.ordinal))
        throw std::runtime_error("Immutable storage reset terminal event publication/completion failed");
    // Service await authenticates the mapped close immediately. Only this exact
    // terminal event (not a stream/device-wide sync) makes that publication safe
    // to observe. No token, stage or retained-transaction boundary waits here.
    PerfStatsCollector::addCounter("immutable_lookup","reset_terminal_waits",1.0,
        "request_reset",state_.device_id.toString());
    retained_storage_reset_close_prepared_=true;
    if(phase==RetainedStorageTerminalPhase::PublishAndAwait){awaitRetainedStorageTerminal();retained_storage_reset_close_prepared_=false;}
}
void DeviceGraphOrchestrator::awaitRetainedStorageTerminal()
{
    if(!retained_storage_boundary_) {
        // Query-only readers still completed their original native terminal
        // bridge. They own no CPU source interval, but their next response may
        // legitimately reuse this session with a fresh source admission.
        retireLocalHotRequestAdmissionAfterTerminal();
        return;
    }
    std::exception_ptr failure;
    try{retained_storage_boundary_->awaitTerminal();}catch(...){failure=std::current_exception();}
    // Both successful and failed workers have joined before counters become
    // observable. Recording adds no device read, transfer or synchronization.
    try{retained_storage_perf_stats_.publish(retained_storage_boundary_->terminalObservation(),state_.device_id.toString());}
    catch(...){if(!failure)throw;} // Preserve a more precise terminal-lifecycle failure when no joined observation exists.
    if(failure)std::rethrow_exception(failure);
    retireLocalHotRequestAdmissionAfterTerminal();
}

bool DeviceGraphOrchestrator::observeOrdinaryGenerationForwardCompletion(int64_t completed_invocations)
{
    const auto &loop = deviceGenerationLoopGraph();
    std::string error;
    if (loop.ordinary_hosted_forward &&
        loop.execution_kind == MTPDeviceGenerationLoopGraphCache::ExecutionKind::HostedDispatchTicketPublisher)
    {
        /*
         * Hosted ExpertOverlay forwards pass through
         * replayRetainedDecodeGraph(), which records every real replay as it is
         * submitted. The terminal only authenticates that the same retained
         * cache entry survived; publishing composed-parent evidence here would
         * double-count work that did not execute inside a parent.
         */
        const auto retained = forward_engine_ && loop.ordinary_forward_identity
            ? forward_engine_->retainedDecodeGraph(*loop.ordinary_forward_identity, &error) : std::nullopt;
        if (!retained || !loop.ordinary_hosted_forward ||
            *retained != *loop.ordinary_hosted_forward || !loop.source_fragments.empty())
        {
            LOG_ERROR("Ordinary ExpertOverlay terminal lost its retained forward identity: " << error);
            return false;
        }
        return completed_invocations >= 0;
    }
    if (!forward_engine_ || !loop.ordinary_forward_identity || loop.source_fragments.size() != 1 ||
        !forward_engine_->observeComposedForwardCompletion(*loop.ordinary_forward_identity,
            loop.source_fragments.front().capture, completed_invocations, &error))
    {
        LOG_ERROR("Ordinary generation terminal lost its captured forward identity: " << error);
        return false;
    }
    return true;
}

OrdinaryForwardRetentionKind DeviceGraphOrchestrator::ordinaryForwardRetentionKind() const
{
    if (!graph_builder_)
        throw std::logic_error("Ordinary forward retention requires its admitted graph builder");
    const auto execution = deviceGenerationExecutionPolicy(DeviceGenerationLoopTopology::FixedDepth);
    const bool boundary = hasHeterogeneousCollectiveExecutionDomain() ||
        static_cast<bool>(moe_overlay_inference_transaction_coordinator_);
    OrdinaryForwardDomain domain = OrdinaryForwardDomain::ParticipantLocal;
    if (boundary && graph_builder_ && graph_builder_->config().moe.routed_expert_plan &&
        graph_builder_->config().moe.routed_expert_plan->usesExpertOverlayAuthority())
    {
        const auto &moe = graph_builder_->config().moe;
        // The builder owns the admitted residency registry and its exact epoch
        // arena. Placement snapshots may legitimately change during maintenance;
        // they are not graph/export identity and are never read to select replay.
        const auto declared = graph_builder_->deviceMoEOverlayEpochExecutionBinding(state_.device_id);
        if (!moe.expert_overlay_residency_authority || !moe.expert_overlay_participant_residency ||
            !declared || declared.arena != moe_overlay_epoch_execution_binding_.arena ||
            declared.request_slot != moe_overlay_epoch_execution_binding_.request_slot ||
            !moe_overlay_epoch_execution_binding_ ||
            moe_overlay_epoch_execution_binding_.arena->deviceId() != state_.device_id ||
            moe_overlay_epoch_execution_binding_.request_slot >=
                moe_overlay_epoch_execution_binding_.arena->requestSlotCapacity())
            throw std::logic_error("Hosted ordinary forward lacks its exact admitted overlay and epoch owner");
        domain = OrdinaryForwardDomain::HostedExpertOverlay;
    }
    return resolveOrdinaryForwardRetention(execution, domain);
}

std::optional<DeviceGraphOrchestrator::OrdinaryGenerationForward>
DeviceGraphOrchestrator::materializeOrdinaryGenerationForward(
    const int32_t *condition, OrdinaryForwardRetentionKind retention, std::string *error, OrdinaryForwardPurpose purpose)
{
    const auto fail = [&](const std::string &reason) -> std::optional<OrdinaryGenerationForward> {
        if (error) *error = reason;
        return std::nullopt;
    };
    if (purpose != OrdinaryForwardPurpose::OutputGeneratingDecode && purpose != OrdinaryForwardPurpose::ForwardOnlyCondition)
        return fail("Unknown resident forward semantic purpose");
    // Resolve the requested ownership before preparing any model nodes. A
    // failed native export never selects the hosted arm.
    const auto admitted = ordinaryForwardRetentionKind();
    if (retention == OrdinaryForwardRetentionKind::HostedExpertOverlay && retention != admitted)
        return fail("Hosted ordinary forward is outside its admitted ExpertOverlay domain");
    if (retention != OrdinaryForwardRetentionKind::NativeComposable &&
        retention != OrdinaryForwardRetentionKind::HostedExpertOverlay)
        return fail("Unknown ordinary forward retention kind");
    auto *const backend = getBackendFor(state_.device_id);
    void *const stream = deviceGenerationLoopGraph().stream.get();
    if (!backend || !stream || !condition || !state_.kv_cache || !request_position_ids_dev_ ||
        condition != device_resident_logical_sequence_state_storage_.next_condition_tokens_device)
        return fail("Resident ordinary forward requires its exact logical token owner, stream and owned KV position");
    // A receive writes directly into the participant's existing hidden bank.
    // This is deliberately a self-binding: the engine must not insert a host
    // handoff or copy/rebind the preceding participant's tensor during replay.
    ForwardInput input;
    input.external_hidden_state = pp_stage_config_ && !pp_stage_config_->has_embedding
        ? state_.hidden.get() : nullptr;
    input.token_ids_device = condition;
    input.position_ids_device = request_position_ids_dev_;
    input.device_decode_position = DeviceDecodePositionBinding{
        backend, state_.kv_cache->deviceSequenceCachedTokenCountPtr(0),
        static_cast<int32_t *>(request_position_ids_dev_)};
    input.position_policy = ForwardPositionPolicy::ExplicitRows;
    input.execution_role = ForwardExecutionRole::MainInference;
    input.state_transaction = purpose == OrdinaryForwardPurpose::ForwardOnlyCondition
        ? ForwardStateTransaction::ForwardOnlyCondition : ForwardStateTransaction::Ordinary;
    input.execution_phase = ForwardExecutionPhase::Decode;
    input.sparse_query_execution_program = sparseQueryForwardProgram(input.execution_phase);
    input.graph_submission_intent = ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch;
    if (pipeline_forward_edges_ && !pipeline_forward_edges_->hasDomainBoundary())
        input.pipeline_activation_ownership = ForwardPipelineActivationOwnership::NativeGenerationParent;
    input.batch_size = input.seq_len = input.real_seq_len = 1;
    input.position_offset = input.token_offset = 1;
    input.device_state_publication_stream = stream;
    input.device = state_.device_id;
    input.kv_cache = state_.kv_cache.get();
    if (!bindShiftedMTPPrefillTransaction(input, 1) || !bindMTPMainTerminalHiddenTransaction(input, 1))
        return fail("Ordinary forward cannot bind the retained model-state producers");
    setBuffers(state_.toModelBuffers());
    ensureForwardEngine();
    ForwardOutput output;
    output.logits = state_.logits.get();
    output.hidden = state_.hidden.get();
    std::optional<ForwardGraphSignature> signature;
    if (!forward_engine_->execute(input, output, *this, &signature) || !signature)
        return fail("Ordinary resident-input forward could not materialize its exact identity");
    std::string detail;
    if (retention == OrdinaryForwardRetentionKind::HostedExpertOverlay)
    {
        const auto forward = forward_engine_->retainedDecodeGraph(*signature, &detail);
        if (!forward)
            return fail("Ordinary hosted ExpertOverlay requires its complete retained body: " + detail);
        PerfStatsCollector::addCounter("forward_graph", "ordinary_hosted_forward_exports", 1.0,
            "graph_setup", state_.device_id.toString());
        return OrdinaryGenerationForward{*forward, *signature};
    }
    const auto forward = forward_engine_->deviceLoopGraphTemplate(*signature, &detail);
    if (!forward)
        return fail("Ordinary generation requires a complete native-composable forward: " + detail);
    return OrdinaryGenerationForward{*forward, *signature};
}

bool DeviceGraphOrchestrator::bindRetainedStorageSubmissionStream(
    const std::shared_ptr<void>& stream, std::string* error)
{
    const auto fail = [&](const char* reason) {
        if (error) *error = reason;
        LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
        return false;
    };
    if (!graph_builder_)
        return fail("Retained storage parent binding requires its original graph builder");
    const auto boundary = graph_builder_->retainedStorageBoundary(state_.device_id);
    if (!boundary){
        if(!sharedPooledQueryOnlyReader())return true;
        const auto participants=injected_model_ctx_->adaptiveImmutableLookupParticipants(state_.device_id);
        if(!participants->hasEmbeddedParticipant(state_.device_id))return true;
        const auto original=participants->originalParticipant(state_.device_id);
        if(original.memory().get()!=physical_memory_authority_.get() ||
           original.memory().owner_before(physical_memory_authority_) || physical_memory_authority_.owner_before(original.memory()))
            return fail("Embedded reader parent lost its original physical-memory authority");
        participants->bindEmbeddedParentSubmissionStream(RetainedGraphParticipantParentStream(original,stream));
        return true;
    }
    auto original = boundary->retainedParticipantSource();
    if (!original) return true;
    const auto& config = graph_builder_->config();
    const auto& execution = config.moe.expert_overlay_execution_plan;
    if (!execution || execution->generationExecutionDomain()!=DeviceGenerationExecutionDomain::HostedBoundary ||
        !execution->ownsContinuationGraph() || execution->world_size!=1 ||
        !config.pp_contexts.empty() || config.pp_layer_offset!=0 ||
        (config.total_n_layers>0 && config.n_layers!=config.total_n_layers) ||
        original->device()!=state_.device_id || original->memory().get()!=physical_memory_authority_.get() ||
        original->memory().owner_before(physical_memory_authority_) ||
        physical_memory_authority_.owner_before(original->memory()))
        return fail("Recurring remote lookup requires its original same-PMA hosted ExpertOverlay parent");
    boundary->bindSourceSubmissionStream(RetainedGraphParticipantParentStream(std::move(*original),stream));
    return true;
}

bool DeviceGraphOrchestrator::materializeOrdinaryDeviceGenerationLoopGraph(
    DeviceGenerationSamplingMode sampling_mode, std::string *error,
    OrdinaryGenerationComposition composition)
{
    using namespace sampling_math;
    using SamplerStage = OrdinaryGenerationSamplingStage;
    using Plan = OrdinaryGenerationGraphPlan;
    auto &loop = deviceGenerationLoopGraph();
    const auto fail = [&](const std::string &reason) {
        // An invalid second materialization must not destroy a graph which
        // already owns an in-flight request. Its existing terminal/reset edge
        // remains responsible for retirement, even on the failure path.
        if (!loop.launched)
            loop.invalidateGraph();
        if (error) *error = reason;
        return false;
    };
    if (error) error->clear();
    const bool native_no_peer_tail = composition == OrdinaryGenerationComposition::PipelineNoPeerOverlayTail;
    const bool pipeline_tail = composition == OrdinaryGenerationComposition::PipelineTail || native_no_peer_tail;
    const bool expert_overlay = composition == OrdinaryGenerationComposition::ExpertOverlay ||
        composition == OrdinaryGenerationComposition::PipelineDomainExpertOverlayTail || native_no_peer_tail;
    const bool pipeline_domain_tail = composition == OrdinaryGenerationComposition::PipelineDomainTail ||
        composition == OrdinaryGenerationComposition::PipelineDomainExpertOverlayTail;
    if (native_no_peer_tail)
    {
        if (!pp_stage_config_ || !graph_builder_ || !moe_overlay_epoch_execution_binding_ ||
            moe_overlay_epoch_execution_binding_.arena->deviceId() != state_.device_id ||
            moe_overlay_epoch_execution_binding_.request_slot >=
                moe_overlay_epoch_execution_binding_.arena->requestSlotCapacity() ||
            (pipeline_forward_edges_ && pipeline_forward_edges_->hasDomainBoundary()) ||
            usesParticipantLocalDeviceMoERebalanceController())
            return fail("Native no-peer pipeline tail has no exact epoch, scope or parent ownership");
        const auto &config = graph_builder_->config();
        const auto &authority = config.moe.expert_overlay_residency_authority;
        if (!authority || config.pp_layer_offset != pp_stage_config_->first_layer ||
            config.n_layers != pp_stage_config_->last_layer - pp_stage_config_->first_layer)
            return fail("Native no-peer pipeline tail has no exact admitted model-layer authority");
        try
        {
            MoEOverlayNoPeerResidency::requireNativePipelineTail(
                authority->noPeerResidency(), *pp_stage_config_, state_.device_id, config.moe.num_experts);
        }
        catch (const std::exception &exception)
        {
            return fail(exception.what());
        }
    }
    // This compiler accepts a complete participant-local graph. A pipeline or
    // sparse boundary needs its explicit transfer/follower composition, not an
    // apparently successful loop around only this participant's model shard.
    if (!active_device_generation_admission_ ||
        !active_device_generation_admission_->depth_policy.isOrdinary() ||
        active_device_generation_admission_->request_count != 1 ||
        !active_device_generation_admission_->ordinary_sampling ||
        !state_.kv_cache || !state_.logits || !request_position_ids_dev_ ||
        activeMainLogitsAreColumnParallel() ||
        (pp_stage_config_ && (!pp_stage_config_->has_lm_head ||
            (!pp_stage_config_->has_embedding && !pipeline_tail &&
             !pipeline_domain_tail))) ||
        (pipeline_domain_tail &&
            (!pipeline_forward_edges_ || !pipeline_forward_edges_->hasDomainBoundary())) ||
        expert_overlay !=
            static_cast<bool>(moe_overlay_epoch_execution_binding_))
        return fail("Ordinary generation requires an admitted scalar sampler and an exact local, pipeline-tail, or ExpertOverlay composition");

    const auto &admission = *active_device_generation_admission_;
    const auto &law = *admission.ordinary_sampling;
    const bool greedy = law.is_greedy();
    if ((greedy ? DeviceGenerationSamplingMode::Greedy : DeviceGenerationSamplingMode::Stochastic) != sampling_mode)
        return fail("Ordinary materialization differs from the admitted sampling law");
    if (law.presence_penalty != mtp_request_penalty_policy_.presence_penalty ||
        law.frequency_penalty != mtp_request_penalty_policy_.frequency_penalty ||
        law.repetition_penalty != mtp_request_penalty_policy_.repetition_penalty)
        return fail("Ordinary sampling penalties differ from the published request policy");

    const auto generation = workspaceGeneration(state_.device_id);
    auto *const backend = getBackendFor(state_.device_id);
    const auto execution = deviceGenerationExecutionPolicy(DeviceGenerationLoopTopology::FixedDepth);
    const bool hosted = execution == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
    if (!generation || !backend || !loop.stream || !loop.capture || loop.launched ||
        execution == DeviceGenerationExecutionPolicy::Unsupported)
        return fail("Ordinary generation has no idle, admitted native graph owner");
    const bool resident_read_ahead = graph_builder_ &&
        graph_builder_->config().ngram_lookup_schedule ==
            GraphConfig::NgramLookupSchedule::EarlyCurrentResidentDecodeGPUCandidate;
    auto retained_read_ahead_worker = resident_read_ahead
        ? GPUDeviceContextPool::instance().retainExistingContext(state_.device_id)
        : std::shared_ptr<IWorkerGPUContext>{};
    if (resident_read_ahead && !retained_read_ahead_worker)
        return fail("Resident read-ahead requires its already admitted original worker");
    auto &context = retained_read_ahead_worker ? *retained_read_ahead_worker
        : GPUDeviceContextPool::instance().getContext(state_.device_id);
    void *const stream = loop.stream.get();
    if (resident_read_ahead)
    {
        const auto& original = graph_builder_->config().ngram_read_ahead_resources;
        if (!original)
            graph_builder_->bindReadAheadNativeResources(ReadAheadNativeResourceFamily::materialize(
                physical_memory_authority_, retained_read_ahead_worker, state_.device_id, stream));
        else if (original->device() != state_.device_id || original->primaryStream() != stream)
            return fail("Resident read-ahead source factory changed its original admitted device/primary stream");
    }
    auto &logical = device_resident_logical_sequence_state_storage_;
    if (!logical.validFor(1))
        return fail("Ordinary generation has no persistent logical frontier");

    // This authority retains the actual controller-owned stream independently
    // of every child recording stream. No new native resource is created.
    if (const auto boundary=graph_builder_->retainedStorageBoundary(state_.device_id);
        boundary && boundary->retainedParticipantSource() && (!hosted || !expert_overlay || pipeline_tail))
        return fail("Recurring remote lookup requires its original same-PMA hosted ExpertOverlay parent");
    if (!bindRetainedStorageSubmissionStream(loop.stream, error)) return false;

    const auto retention = (pipeline_tail || !expert_overlay)
        ? OrdinaryForwardRetentionKind::NativeComposable : ordinaryForwardRetentionKind();
    const auto prepared = materializeOrdinaryGenerationForward(logical.next_condition_tokens_device, retention, error);
    if (!prepared) return false;
    const auto &signature = prepared->signature;
    const auto *native_forward = prepared->native();
    const auto *hosted_forward = prepared->hosted();
    const bool hosted_overlay = retention == OrdinaryForwardRetentionKind::HostedExpertOverlay;
    if ((hosted_overlay && (!hosted || !expert_overlay || pipeline_tail)) ||
        hosted_overlay != static_cast<bool>(hosted_forward) ||
        (!hosted_overlay && !native_forward))
        return fail("Ordinary generation export differs from its explicit native/hosted composition");
    const auto read_ahead = native_forward ? native_forward->read_ahead_resources
        : hosted_forward ? hosted_forward->read_ahead_resources : nullptr;
    if(resident_read_ahead != static_cast<bool>(read_ahead))
        return fail("Selected resident read-ahead lost or forged its original retained forward resource owner");
    if (read_ahead)
    {
        // The native parent or authenticated whole-request ticket cursor owns
        // this family across every retained forward and maintenance fragment.
        // Hosted exports must name their actual complete retained composition;
        // an ordinary segmented replay cannot acquire this resource lifetime.
        if (read_ahead->device() != state_.device_id ||
            read_ahead->primaryStream() != stream || !device_generation_state_ready_.event ||
            (hosted_forward && (!hosted_forward->composed_parent || hosted_forward->segmented ||
                                hosted_forward->replay_unit_count != 1u ||
                                hosted_forward->device != state_.device_id)))
            return fail("Ordinary read-ahead requires its exact complete retained forward and whole-request primary/event");
        read_ahead->bindTerminal(device_generation_state_ready_.event);
    }
    std::string detail;
    const bool owns_maintenance =
        usesParticipantLocalDeviceMoERebalanceController();
    const uint32_t *maintenance_due = nullptr;
    const IGPUGraphCapture *maintenance_capture = nullptr;
    if (owns_maintenance)
    {
        const auto &maintenance = device_moe_rebalance_maintenance_graph_;
        const auto *const controller =
            deviceMoERebalanceControllerStateDevice();
        if (!expert_overlay ||
            !controller || !maintenance.graph ||
            maintenance.workspace_generation != generation)
            return fail("Ordinary ExpertOverlay has no current device maintenance owner");
        maintenance_due = reinterpret_cast<const uint32_t *>(
            reinterpret_cast<const std::byte *>(controller) +
            offsetof(DeviceMoERebalanceGraphControllerState,
                     maintenance_due));
        const auto retained =
            maintenance.segment_cache.deviceLoopGraphTemplate(
                *maintenance.graph, &detail);
        if (!retained || !retained->capture)
            return fail("Ordinary ExpertOverlay has no retained maintenance graph: " + detail);
        maintenance_capture = retained->capture;
    }

    const auto seeds = arena_->getSharedTensor(BufferId::SAMPLING_REQUEST_SEEDS);
    SamplerStage::Params params;
    params.device_id = state_.device_id;
    params.backend = backend;
    params.logits = static_cast<float *>(state_.logits->gpu_data_ptr());
    params.vocab_size = params.logits_row_stride = state_.vocab_size;
    if (!greedy)
        params.policy = SamplerStage::Stochastic{law.top_k, law.top_p, law.temperature,
            seeds ? static_cast<const uint64_t *>(seeds->gpu_data_ptr()) : nullptr};
    params.workspace = {
        static_cast<float *>(stochastic_target_probs_dev_),
        static_cast<int32_t *>(stochastic_target_token_ids_dev_), kMaxTopK,
        static_cast<float *>(stochastic_topk_partial_vals_dev_),
        static_cast<int32_t *>(stochastic_topk_partial_idxs_dev_), stochastic_topk_partial_capacity_};
    params.publication = {
        .request_count = 1,
        .sampled_tokens = static_cast<int32_t *>(stochastic_target_sample_tokens_dev_),
        .stop_tokens = {static_cast<const int32_t *>(mtp_verifier_stop_tokens_dev_), kSpeculativeBatchMaxStopTokens, 0},
        .source = OrdinaryGenerationSampleSource::PrefillLogits,
        .response_tokens = device_generation_storage_.response_tokens_device,
        .response_token_stride = device_generation_storage_.response_token_stride,
        .control = device_generation_storage_.control_device,
        .control_stride = device_generation_storage_.control_stride,
        .frontier = {logical.target_cached_tokens_device, logical.next_condition_tokens_device,
            logical.stopped_flags_device, logical.publication_ok_flags_device, 1, state_.max_seq_len},
        .history = {static_cast<int32_t *>(mtp_generated_token_counts_dev_), state_.vocab_size,
            state_.vocab_size, 1}};
    params.penalties = law.has_penalties()
        ? static_cast<const MTPGreedyPenaltyPolicy *>(mtp_greedy_penalty_policy_dev_) : nullptr;

    if (loop.valid && loop.workspace_generation == generation &&
        loop.ordinary_sampling_identity == params && loop.ordinary_forward_identity == signature &&
        loop.read_ahead_resources == read_ahead &&
        loop.ordinary_composition == composition &&
        loop.ordinary_maintenance_capture == maintenance_capture &&
        loop.ordinary_maintenance_due == maintenance_due &&
        (hosted_overlay
            ? (loop.source_fragments.empty() && loop.ordinary_hosted_forward == *hosted_forward)
            : (!loop.ordinary_hosted_forward && loop.source_fragments.size() == 1 &&
               loop.source_fragments.front().capture == native_forward->capture)))
    {
        // Leading-row ownership is request data, not graph topology. CUDA
        // reads the admitted control word; HIP uses this immutable admission
        // to submit its first complete transaction before observing tickets.
        loop.ordinary_leading = admission.initial_leading_row_disposition;
        PerfStatsCollector::addCounter("generation", "ordinary_graph_reuses", 1.0,
            "graph_setup", state_.device_id.toString());
        return true;
    }

    loop.invalidateGraph();
    // RAII closes capture even if a backend reports an asynchronous error.
    // These recordings are cloned by the complete parent, not launched as
    // independent per-stage work. Only HIP's initial sampler is executable.
    for (const auto index : {Plan::prefill_sampler, Plan::decode_sampler})
    {
        auto &child = loop.ordinary_children[index];
        child = context.createGraphCapture(stream);
        if (!child) return fail("Ordinary sampler has no capture owner");
        auto binding = params;
        binding.publication.source = index == Plan::prefill_sampler
            ? OrdinaryGenerationSampleSource::PrefillLogits : OrdinaryGenerationSampleSource::DecodeLogits;
        SamplerStage stage(binding);
        stage.setGPUStream(stream);
        bool recorded = false;
        {
            ScopedBackendGraphCapture recording(context, *child, "ordinary_generation_sampler");
            if (!recording.begin()) return fail("Ordinary sampler capture could not begin");
            recorded = stage.execute(getDeviceContext(state_.device_id));
            recording.finish();
        }
        if (!recorded) return fail("Ordinary sampler could not record its exact policy");
    }
    const DeviceGenerationGraphControl control{admission.depth_policy,
        device_generation_storage_.control_device, device_generation_storage_.control_stride, 1};
    const DeviceControlledLoopFragment initialization[] = {{
        .name = "ordinary_prefill_sample", .capture = loop.ordinary_children[Plan::prefill_sampler].get(),
        .execution = DeviceControlledLoopFragmentExecution::IfDeviceWordZero,
        .condition_word_device = reinterpret_cast<const uint32_t *>(
            control.rows + kDeviceGenerationControlNextLeadingCommittedOutputCount)}};
    /*
     * A MainInference forward is the owner of the ExpertOverlay reader lease:
     * appendCapturedMainForwardMoEOverlayEpochTransaction decorates this exact
     * retained forward during materializeOrdinaryGenerationForward.  Keeping
     * the boundary in that graph is essential because it spans the real sparse
     * collective body.  Adding a second acquire/release around the sampler
     * would both duplicate the lease and borrow an independently invalidated
     * boundary cache after a normal decode recapture.
     */
    if (pipeline_tail)
    {
        // The rank installs activation/command edges around these recordings.
        // A local-only tail loop would advance KV without upstream layers.
    }
    else if (!hosted)
    {
        std::array<DeviceControlledLoopFragment, 3> iteration = {{
            {.name = "ordinary_resident_forward", .capture = native_forward->capture},
            {.name = "ordinary_decode_sample", .capture = loop.ordinary_children[Plan::decode_sampler].get()},
            {}}};
        if (owns_maintenance)
        {
            // The model's final routed layer publishes the serial commit clock.
            // Maintenance consumes that due word only after the sampled response
            // and the forward's captured epoch release have completed.
            iteration[2] = {
                .name = "ordinary_device_moe_maintenance",
                .capture = maintenance_capture,
                .execution = DeviceControlledLoopFragmentExecution::IfDeviceWordNonZero,
                .condition_word_device = maintenance_due};
        }
        if (!DeviceGenerationGraphProgram::native(*loop.capture, control,
                {.initialization = initialization,
                 .iteration = std::span<const DeviceControlledLoopFragment>(
                     iteration.data(), 2u + static_cast<size_t>(owns_maintenance)),
                 .terminal=retainedStorageTerminalFragments()},
                detail))
            return fail(detail);
    }
    else if (hosted_overlay)
    {
        std::string retained_error;
        if (!forward_engine_->retainedDecodeGraph(signature, &retained_error) ||
            !loop.ordinary_children[Plan::prefill_sampler]->instantiate() ||
            !loop.ordinary_children[Plan::decode_sampler]->instantiate() ||
            !DeviceGenerationGraphProgram::ticketPublisher(*loop.capture, context, *backend, state_.device_id,
                control, maintenance_due, device_generation_storage_.dispatch_tickets_device, detail,retained_storage_boundary_.get()))
        {
            return fail("Ordinary ExpertOverlay hosted graph compilation failed: " +
                        (retained_error.empty() ? detail : retained_error));
        }
        loop.hosted_fragments.push_back({
            .name = "ordinary_expert_overlay_transaction",
            .kind = HostedDeviceGenerationFragment::Kind::OrdinaryExpertOverlayReplay,
            .capture = loop.ordinary_children[Plan::decode_sampler].get(),
            .forward_signature = signature,
        });
    }
    else
    {
        auto &transaction = loop.ordinary_children[Plan::hosted_transaction];
        transaction = context.createGraphCapture(stream);
        const GPUOrderedTimelineStep steps[] = {
            {.name = "ordinary_resident_forward", .capture = native_forward->capture},
            {.name = "ordinary_decode_sample", .capture = loop.ordinary_children[Plan::decode_sampler].get()}};
        if (!transaction || !transaction->buildOrderedTimelineTransaction(steps) || !transaction->instantiate() ||
            !loop.ordinary_children[Plan::prefill_sampler]->instantiate() ||
            !DeviceGenerationGraphProgram::ticketPublisher(*loop.capture, context, *backend, state_.device_id,
                control, maintenance_due, device_generation_storage_.dispatch_tickets_device, detail,retained_storage_boundary_.get()))
            return fail("Ordinary hosted graph compilation failed: " + detail);
        loop.hosted_fragments.push_back({.name = "ordinary_complete_transaction",
            .kind = HostedDeviceGenerationFragment::Kind::CapturedLocal, .capture = transaction.get()});
    }
    if (hosted && !pipeline_tail)
    {
        if (owns_maintenance)
        {
            // Hosted tickets retire this conditional tail before they admit
            // the next body; a period boundary can never be crossed early.
            loop.hosted_fragments.push_back({
                .name = "ordinary_device_moe_maintenance",
                .kind = HostedDeviceGenerationFragment::Kind::CapturedLocal,
                .capture = maintenance_capture,
                .execution = DeviceControlledLoopFragmentExecution::IfDeviceWordNonZero,
                .condition_word_device = maintenance_due});
        }
        loop.branch_fragment_counts[0] = loop.hosted_fragments.size();
    }
    // The forward identity and bindings are immutable; all mutable counters,
    // seeds and frontier contents stay in their existing arena-owned banks.
    // Completion attribution belongs to the retained model forward, never to
    // the overlay lease child that happens to precede it in this composition.
    if (hosted_overlay)
        loop.ordinary_hosted_forward = *hosted_forward;
    else
        loop.source_fragments.push_back(
            {.name = "ordinary_resident_forward", .capture = native_forward->capture});
    loop.ordinary_sampling_identity = params;
    loop.ordinary_forward_identity = signature;
    loop.read_ahead_resources = read_ahead;
    loop.ordinary_composition = composition;
    loop.ordinary_maintenance_capture = maintenance_capture;
    loop.ordinary_maintenance_due = maintenance_due;
    loop.ordinary_leading = admission.initial_leading_row_disposition;
    loop.workspace_generation = generation;
    loop.request_count = 1;
    loop.depth_policy_mode = static_cast<int>(DeviceGenerationPolicyMode::Ordinary);
    loop.sampling_mode = sampling_mode;
    loop.fragment_count = hosted
        ? loop.branch_fragment_counts[0]
        : 3u + static_cast<size_t>(owns_maintenance);
    loop.execution_kind = hosted ? MTPDeviceGenerationLoopGraphCache::ExecutionKind::HostedDispatchTicketPublisher
                                : MTPDeviceGenerationLoopGraphCache::ExecutionKind::NativeConditionalParent;
    loop.valid = true;
    if (native_no_peer_tail)
        PerfStatsCollector::addCounter("generation", "ordinary_native_pipeline_nopeer_tail_materializations", 1.0,
            "graph_setup", state_.device_id.toString());
    PerfStatsCollector::addCounter("generation", "ordinary_graph_materializations", 1.0,
        "graph_setup", state_.device_id.toString());
    return true;
}

bool DeviceGraphOrchestrator::replayHostedOrdinaryExpertOverlayTransaction(
    const ForwardGraphSignature &signature,
    const IGPUGraphCapture *sampler,
    void **out_producer_stream,
    std::string *error)
{
    const auto fail = [&](const std::string &reason)
    {
        if (error)
            *error = reason;
        return false;
    };
    if (error)
        error->clear();
    if (out_producer_stream)
        *out_producer_stream = nullptr;

    auto &loop = deviceGenerationLoopGraph();
    if (!forward_engine_ || !sampler || !sampler->hasExecutable() ||
        !loop.stream ||
        (loop.ordinary_composition != OrdinaryGenerationComposition::ExpertOverlay &&
         loop.ordinary_composition != OrdinaryGenerationComposition::PipelineDomainExpertOverlayTail) ||
        !loop.ordinary_forward_identity ||
        *loop.ordinary_forward_identity != signature)
    {
        LOG_ERROR("[DeviceGraphOrchestrator] Hosted ordinary ExpertOverlay ownership mismatch"
                  << " device=" << state_.device_id.toString()
                  << " forward_engine=" << static_cast<bool>(forward_engine_)
                  << " sampler=" << static_cast<bool>(sampler)
                  << " sampler_executable="
                  << (sampler && sampler->hasExecutable())
                  << " stream=" << static_cast<bool>(loop.stream)
                  << " composition="
                  << (loop.ordinary_composition
                          ? static_cast<int>(*loop.ordinary_composition)
                          : -1)
                  << " forward_identity="
                  << static_cast<bool>(loop.ordinary_forward_identity)
                  << " signature_match="
                  << (loop.ordinary_forward_identity &&
                      *loop.ordinary_forward_identity == signature)
                  << " coordinator="
                  << static_cast<bool>(
                         moe_overlay_inference_transaction_coordinator_));
        return fail(
            "hosted ordinary ExpertOverlay replay has incomplete or divergent retained ownership");
    }

    std::string retained_error;
    const auto retained =
        forward_engine_->retainedDecodeGraph(signature, &retained_error);
    if (!retained)
        return fail(retained_error);
    if (!loop.ordinary_hosted_forward || *loop.ordinary_hosted_forward != *retained ||
        !loop.source_fragments.empty() || ordinaryForwardRetentionKind() != OrdinaryForwardRetentionKind::HostedExpertOverlay)
        return fail("Hosted ordinary ExpertOverlay retained owner changed before replay");

    MoEOverlayInferenceExecutionDescriptor descriptor{
        .graph_role = MoEOverlayInferenceGraphRole::MainDecode,
        .logical_step_id = 0,
        .placement_epoch =
            moe_overlay_inference_transaction_coordinator_
                ? moe_overlay_inference_transaction_coordinator_
                      ->currentPlacementEpoch()
                : 0,
        .request_count = 1,
        .logical_rows_per_request = 1,
        .physical_rows_per_request = 1,
        .draft_depth = -1,
        .sidecar_depth = -1,
    };
    MoEOverlayInferenceParticipantGraphScope graph_scope(
        moe_overlay_inference_transaction_coordinator_,
        descriptor,
        moe_overlay_inference_transaction_participant_index_);
    if (!graph_scope.ready() ||
        (moe_overlay_inference_transaction_coordinator_ &&
         !graph_scope.active()))
    {
        return fail(
            graph_scope.error().empty()
                ? "hosted ordinary ExpertOverlay replay has no active transaction binding"
                : graph_scope.error());
    }

    const auto request_identity = currentMoESparseRequestIdentity();
    const auto &binding = graph_scope.binding();
    if (!request_identity ||
        (graph_scope.active() &&
         (request_identity->generation() != binding.request_generation ||
          binding.descriptor.graph_role !=
              MoEOverlayInferenceGraphRole::MainDecode ||
          binding.descriptor.draft_depth != -1 ||
          binding.descriptor.sidecar_depth != -1)))
    {
        return fail(
            "hosted ordinary ExpertOverlay binding disagrees with its request generation or serial decode role");
    }
    const IComputeStage::MoEOverlayCollectiveRuntimeParams sparse_params{
        .generation_id = graph_scope.active()
                             ? binding.request_generation
                             : request_identity->generation(),
        .step_id = graph_scope.active()
                       ? binding.descriptor.logical_step_id
                       : 0,
        .execution_semantics =
            IComputeStage::MoEOverlayCollectiveRuntimeParams::
                ExecutionSemantics::Decode,
        .mtp_depth = -1,
        .placement_epoch = graph_scope.active()
                               ? binding.descriptor.placement_epoch
                               : 0,
    };
    if (!sparse_params.valid() || !sparse_params.hasExecutionSemantics())
    {
        return fail(
            "hosted ordinary ExpertOverlay replay produced an invalid sparse wire identity");
    }

    DeviceGraphExecutor::GraphLaunchDependencyHook launch_dependency =
        [&graph_scope, device = state_.device_id](
            DeviceGraphExecutor::GraphExecutableLaunchPhase,
            void *execution_stream) -> bool
    {
        // The complete main forward already captures its acquire/release
        // around sparse work. Only a remote follower needs a host-side ticket.
        if (!graph_scope.ownsTicketAuthority())
            return true;
        std::string arm_error;
        if (!graph_scope.armForExecutableLaunch(
                execution_stream, &arm_error))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Hosted ordinary decode could not arm its ExpertOverlay follower at the retained executable launch edge on "
                << device.toString() << ": " << arm_error);
            return false;
        }
        return true;
    };

    IDeviceContext *const ctx = getDeviceContext(state_.device_id);
    void *forward_stream = nullptr;
    bool submitted = forward_engine_->replayRetainedDecodeGraph(
        signature,
        ctx,
        sparse_params,
        launch_dependency,
        *this,
        loop.stream.get(),
        &forward_stream,
        &retained_error);
    if (submitted)
        submitted = sampler->launchOnStream(loop.stream.get());

    std::string finish_error;
    if (!graph_scope.finish(submitted, &finish_error))
        return fail(finish_error);
    if (!submitted)
    {
        return fail(
            retained_error.empty()
                ? "hosted ordinary ExpertOverlay sampler submission failed"
                : retained_error);
    }
    if (forward_stream != retained->stream)
    {
        return fail(
            "hosted ordinary ExpertOverlay forward published an invalid completion stream");
    }
    if (out_producer_stream)
        *out_producer_stream = loop.stream.get();
    return true;
}
}
