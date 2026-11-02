/**
 * @file PooledImmutableRetainedParticipants.cpp
 * @brief Original recurring hosted owner captures, source seals and whole-request retirement.
 *
 * No source model graph is cloned or substituted. Each actual original source
 * parent invocation submits each addressed owner program once. Protocol packet
 * retirement and native executable completion remain distinct authorities.
 */
#include "PooledImmutableRetainedParticipants.h"
#include "PooledImmutableParticipantExecutionPolicy.h"
#include "backends/RetainedParticipantGraphExecutable.h"
#include "backends/GPUGraphMemoryContract.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableNativePhaseStage.h"
#include "models/GraphTypes.h"
#include "utils/PerfStatsCollector.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include <mutex>
#include <algorithm>
#include <limits>
#include <thread>
#include <stdexcept>
namespace llaminar2::qwen4exp {
struct PooledImmutableRetainedParticipants::State {
    struct Owner {
        PreparedGPUContextUsePermit original;
        std::uint32_t coordinate;
        PooledRemoteGraphProgram program; ///< Exact same source scope declared before admission.
        std::shared_ptr<IDeviceContext> context;
        std::shared_ptr<BufferArena> arena;
        std::shared_ptr<ComputeGraph> graph;
        std::shared_ptr<const IPooledImmutableNativeKernels> code;
        std::shared_ptr<RetainedParticipantGraphExecutable> executable;
        struct Leaf {std::string name;const IComputeStage* stage;std::vector<std::string> dependencies;std::uint64_t identity;};
        std::vector<Leaf> leaves;
        std::uint64_t topology=0;
        mutable std::uint64_t submissionOrdinal=0; ///< Ownership sequence of consumed original source invocations, never GPU progress.
        std::string terminal;
    };
    std::shared_ptr<ImmutableLookupExecutionBoundary> boundary;
    std::shared_ptr<const PooledImmutableDirectedTransport> transport;
    PreparedGPUContextUsePermit source;
    std::shared_ptr<const MoEExpertOverlayExecutionPlan> executionPolicy; ///< Original immutable topology; expert movement epochs remain device-owned and separate.
    std::vector<Owner> owners;
    std::unique_ptr<RetainedGraphParticipantEnqueueGroup> enqueueGroup;
    std::vector<PooledRemoteGraphProgram> programs; ///< Original physical order; never recomputed from live counts.
    std::uint64_t rootIdentity;
    mutable std::mutex submissionMutex;
    std::optional<RetainedGraphParticipantParentStream> submissionStream;
    std::vector<std::weak_ptr<const RetainedGraphParticipantPreparationStream>> preparationStreams;
    bool ownsBoundaryClaim=false; ///< One cold CPU ownership claim, released only after original participants retire.
    State(std::shared_ptr<ImmutableLookupExecutionBoundary> b,PreparedGPUContextUsePermit original,std::uint64_t identity)
        :boundary(std::move(b)),source(std::move(original)),rootIdentity(identity){}
};
class PooledImmutableRetainedParticipants::Invocation final:public IRetainedGraphParticipantInvocation {
public:
    Invocation(std::shared_ptr<PooledImmutableRetainedParticipants> owner,RetainedGraphParticipantSourceIdentity source)
        :owner_(std::move(owner)),source_(source){}
    void requireCurrent(const RetainedGraphParticipantSourceSeal& current)const override {
        if(current.identity()!=source_)throw std::logic_error("Recurring source changed its original parent/arena/stream/variant/generation");
        owner_->requireCurrent();
    }
    RetainedGraphParticipantSubmission enqueueOriginalParticipants()const override{return owner_->enqueueOriginalParticipants();}
private:
    std::shared_ptr<PooledImmutableRetainedParticipants> owner_;
    RetainedGraphParticipantSourceIdentity source_;
};
class PooledImmutableRetainedParticipants::SourcePlan final:public IRetainedGraphParticipantSourcePlan {
public:
    SourcePlan(std::shared_ptr<PooledImmutableRetainedParticipants> owner,BufferArena& arena,
        PooledImmutableRetirementCompletion completion):owner_(std::move(owner)),arena_(&arena),completion_(std::move(completion)){
        // Descriptor identity includes the exact source output roots, not merely
        // a cohort's owner count. Actual topology/recording identity seals later.
        identity_=1469598103934665603ULL;
        const auto fold=[&](std::uint64_t value){identity_^=value;identity_*=1099511628211ULL;};
        fold(owner_->state_->rootIdentity);fold(reinterpret_cast<std::uintptr_t>(arena_));
        fold(reinterpret_cast<std::uintptr_t>(owner_->state_->transport.get()));
        fold(reinterpret_cast<std::uintptr_t>(owner_->state_->executionPolicy.get()));
        for(const auto& owner:owner_->state_->owners){fold(owner.program.captureIdentity());fold(owner.topology);
            fold(reinterpret_cast<std::uintptr_t>(owner.executable.get()));}
        for(const auto& name:{completion_.coordinator.first,completion_.coordinator.last,
            completion_.readersRetired,completion_.observed})for(const unsigned char c:name)fold(c);
        if(!identity_)identity_=1;
    }
    const PreparedGPUContextUsePermit& originalSource()const noexcept override{return owner_->state_->source;}
    void authenticatePreparationStream(std::shared_ptr<const RetainedGraphParticipantPreparationStream> preparation)const override{
        if(!preparation)throw std::invalid_argument("Original source preparation lost its cache authority");
        preparation->requireLive(owner_->state_->source);
        std::lock_guard lock(owner_->state_->submissionMutex);
        auto& retained=owner_->state_->preparationStreams;
        for(auto it=retained.begin();it!=retained.end();){
            if(const auto existing=it->lock()){
                if(existing==preparation)return;
                ++it;
            }else it=retained.erase(it);
        }
        retained.push_back(std::move(preparation));
    }
    std::uint64_t captureIdentity()const noexcept override{return identity_;}
    void authenticateRecordingOwner(const IGPUGraphCapture& actual)const override {
        owner_->requireCurrent();
        if(!owner_->state_->source.worker()->ownsCurrentThread() ||
           !actual.executionStream() || reinterpret_cast<std::uintptr_t>(actual.executionStream())<=2)
            throw std::invalid_argument("Recurring source requires its actual original worker and native capture stream");
        // The actual capture supplies its recording stream. Each cache seal
        // retains that exact identity; serialized variants may own different streams.
    }
    std::shared_ptr<IRetainedGraphParticipantInvocation> bindOriginalSource(const RetainedGraphParticipantSourceSeal& sealed)const override {
        const auto& source=sealed.identity();
        if(!source.graph || source.arena!=arena_ || source.device!=owner_->state_->source.device() ||
           !source.original_parent || !source.original_parent->hasExecutable() ||
           source.original_parent->executionStream()!=source.recording_stream ||
           !source.recording_stream || reinterpret_cast<std::uintptr_t>(source.recording_stream)<=2 ||
           !source.capture_generation || !source.capture_variant_signature ||
           !ownsHeterogeneousTicketBoundary(source.graph->nativeCaptureEnvelope()) ||
           source.graph->retainedParticipantSourcePlan().get()!=this)
            throw std::invalid_argument("Recurring participant binding requires the actual complete original hosted source parent");
        requirePooledRetirementCompletion(*source.graph,completion_);
        owner_->requireCurrent();
        return std::make_shared<Invocation>(owner_,source);
    }
private:
    std::shared_ptr<PooledImmutableRetainedParticipants> owner_;
    BufferArena* arena_;
    PooledImmutableRetirementCompletion completion_;
    std::uint64_t identity_;
};
PooledImmutableRetainedParticipants::PooledImmutableRetainedParticipants(std::unique_ptr<State> state):state_(std::move(state)){}
std::shared_ptr<PooledImmutableRetainedParticipants> PooledImmutableRetainedParticipants::create(
    std::shared_ptr<ImmutableLookupExecutionBoundary> boundary,const GraphConfig& policy){
    PooledImmutableParticipantExecutionPolicy::requireConfiguration(policy.mtp);
    if(isGraphCaptureActive() || !boundary || !boundary->isPooled() || policy.default_device!=boundary->device() ||
       policy.n_layers<=0 || policy.pp_layer_offset!=0 ||
       (policy.total_n_layers>0 && policy.n_layers!=policy.total_n_layers) || !policy.pp_contexts.empty() ||
       !policy.moe.expert_overlay_execution_plan)
        throw std::invalid_argument("Recurring declared owners require original hosted ExpertOverlay source policy");
    const auto& execution=*policy.moe.expert_overlay_execution_plan;
    if(execution.world_size!=1 || !execution.ownsContinuationGraph() ||
       execution.generationExecutionDomain()!=DeviceGenerationExecutionDomain::HostedBoundary)
        throw std::invalid_argument("Recurring remote owners require one actual rank-local heterogeneous hosted ticket parent");
    const auto dense=std::find_if(execution.domains.begin(),execution.domains.end(),[&](const auto& domain){return domain.name==execution.continuation_domain;});
    if(dense==execution.domains.end() || dense->participants.size()!=1 ||
       !dense->participants.front().owned_by_current_rank || !dense->participants.front().locally_addressable ||
       dense->participants.front().local_device!=boundary->device())
        throw std::invalid_argument("First recurring owner composition requires its actual singleton continuation; multi-compute domains need their coordinated source participant path");
    const auto storage=boundary->pooledStorage();const auto& witness=storage->retainedPreparedSourceWorkers();
    if(!witness || witness->rootIdentity()!=storage->captureIdentity() || witness->coordinator().device()!=boundary->device())
        throw std::invalid_argument("Recurring remote owners require actually consumed original prepared source workers");
    bool cpu=false;
    for(const auto& domain:execution.domains)for(const auto& participant:domain.participants){
        cpu=cpu || participant.local_device.is_cpu();
        for(const auto& owner:witness->owners())
            if(owner.device()!=boundary->device() && participant.local_device==owner.device())
                throw std::invalid_argument("First recurring owner program requires remote storage-only devices, never relabeled compute/expert domains");
    }
    if(!cpu)throw std::invalid_argument("Recurring remote owner policy lost its explicitly declared CPU expert boundary");
    if(witness->coordinator().worker()->ownsCurrentThread())throw std::logic_error("Recurring owner preparation must run off its original source worker");
    std::unique_ptr<RetainedGraphParticipantEnqueueGroup> enqueueGroup;
    std::vector<PooledRemoteGraphProgram> programs;
    for(const auto& owner:witness->owners()){
        if(owner.worker()->ownsCurrentThread())throw std::logic_error("Recurring owner preparation must run off every original participant worker");
        if(owner.device()!=boundary->device()){
            const auto coordinate=static_cast<std::uint32_t>(&owner-witness->owners().data());
            auto program=PooledImmutableDirectedOwnerProgram::fromOriginalPlans(storage->originalOwners()[coordinate]->plan(),storage->coordinator()->plan());
            // Revalidate every installed host class before creating even the
            // first directed channel. This selects no numeric caller budget.
            (void)RetainedParticipantGraphExecutable::hostBOM(program,witness->source()->originalResource(DeviceId::cpu()),
                witness->source()->originalResource(owner.device()));
            programs.push_back(std::move(program));
        }
    }
    auto result=std::shared_ptr<PooledImmutableRetainedParticipants>(new PooledImmutableRetainedParticipants(
        std::make_unique<State>(boundary,witness->coordinator(),storage->captureIdentity())));
    if(boundary->cohortOwnerClaim_.test_and_set(std::memory_order_acquire))
        throw std::logic_error("Original source is already constructing, owned or retiring its participant domain");
    result->state_->ownsBoundaryClaim=true;
    result->state_->executionPolicy=policy.moe.expert_overlay_execution_plan;
    result->state_->programs=std::move(programs);
    result->state_->transport=PooledImmutableDirectedTransport::create(boundary);
    if(result->state_->transport->remotes().empty())throw std::invalid_argument("Recurring owner binding has no actual declared remote");
    result->prepareOriginalOwners();
    boundary->bindRetainedParticipants(result);
    return result;
}
void PooledImmutableRetainedParticipants::requireForPolicy(const GraphConfig& policy)const {
    PooledImmutableParticipantExecutionPolicy::requireConfiguration(policy.mtp);
    requireCurrent();
    if(policy.moe.expert_overlay_execution_plan!=state_->executionPolicy ||
       policy.default_device!=state_->source.device() ||
       policy.n_layers<=0 || policy.pp_layer_offset!=0 || (policy.total_n_layers>0 && policy.n_layers!=policy.total_n_layers) ||
       !policy.pp_contexts.empty() || !policy.moe.expert_overlay_execution_plan ||
       policy.moe.expert_overlay_execution_plan->world_size!=1 ||
       !policy.moe.expert_overlay_execution_plan->ownsContinuationGraph() ||
       policy.moe.expert_overlay_execution_plan->generationExecutionDomain()!=DeviceGenerationExecutionDomain::HostedBoundary)
        throw std::invalid_argument("Retained remote source requires its original hosted continuation policy");
}
PooledImmutableLookupGraphEndpoints PooledImmutableRetainedParticipants::appendPLE(
    ComputeGraph& graph,BufferArena& arena,std::shared_ptr<const IPooledImmutableNativeKernels> code,
    DeviceRowRange tokens,PooledNativeTensorSpan ids,std::optional<PooledNativeTensorSpan> count,
    DeviceGenerationControlBinding control,PooledNativeTensorSpan output,
    const std::string& identity,const std::string& outputIdentity){
    requireCurrent();
    if(bool(count)!=bool(tokens.countOwner()))throw std::invalid_argument("Directed PLE lost original counted rows");
    const auto expanded=tokens.repeatEachRow(16);
    const auto storage=state_->boundary->pooledStorage();const auto& declared=storage->geometry().parameters();
    if(ids.bytes!=std::size_t{expanded.physicalRows()}*sizeof(std::uint64_t) || ids.offset%alignof(std::uint64_t) ||
       output.bytes!=std::size_t{expanded.physicalRows()}*declared.row_elements*sizeof(float))
        throw std::invalid_argument("Directed PLE requires exact original hash and output extents");
    PooledNativeOperandSpans begin{};
    begin[static_cast<unsigned>(PooledNativeOperand::Ids)]=std::move(ids);
    if(count)begin[static_cast<unsigned>(PooledNativeOperand::OriginalCount)]=std::move(*count);
    begin[static_cast<unsigned>(PooledNativeOperand::Admission)]=PooledNativeTensorSpan{
        storage->roots().native,BufferId::QFN_POOLED_SOURCE_NATIVE,
        declared.offsets.protocol+offsetof(ImmutableLookupDeviceState,admissionGeneration),sizeof(std::uint64_t)};
    auto endpoints=appendPooledImmutableDirectedLookupGraph(graph,arena,state_->boundary,state_->transport,std::move(code),
        {expanded,pooled_native::OriginalIdWidth::UInt64},std::move(begin),control,std::move(output),identity,outputIdentity);
    bindSourcePlan(graph,arena,std::move(endpoints.retirement));
    return endpoints.source;
}
void PooledImmutableRetainedParticipants::prepareOriginalOwners(){
    const auto& witness=*state_->boundary->pooledStorage()->retainedPreparedSourceWorkers();
    for(const auto& edge:state_->transport->remotes()){
        const auto& permit=witness.owners()[edge.coordinate];
        std::shared_ptr<const IPooledImmutableNativeKernels> code;
#ifdef HAVE_CUDA
        if(permit.device().is_cuda())code=createCUDAPooledImmutableNativeKernels();
#endif
#ifdef HAVE_ROCM
        if(permit.device().is_rocm())code=createROCmPooledImmutableNativeKernels();
#endif
        if(!code)throw std::invalid_argument("Recurring owner native backend was not compiled");
        const auto declared=std::find_if(state_->programs.begin(),state_->programs.end(),[&](const auto& program){return program.geometry().coordinate==edge.coordinate;});
        if(declared==state_->programs.end())throw std::logic_error("Recurring original owner lost its admitted source program");
        State::Owner owner{.original=permit,.coordinate=edge.coordinate,.program=*declared,
            .arena=std::make_shared<BufferArena>(),.graph=std::make_shared<ComputeGraph>(),.code=std::move(code)};
        void* originalStream=nullptr;
        permit.worker()->submitAndWait([&]{originalStream=permit.worker()->defaultStream();});
        owner.executable=RetainedParticipantGraphExecutable::create(permit,originalStream,owner.program);
        owner.executable->bindCohortDefinition();
        permit.worker()->submitAndWait([&]{
            auto context=IDeviceContext::create(permit.device(),1);
            if(!context)throw std::runtime_error("Original remote context wrapper is unavailable");
            owner.context=std::shared_ptr<IDeviceContext>(context.release(),[worker=permit.worker()](IDeviceContext* original){
                // Even the lightweight device wrapper's destructor selects its
                // device. Keep that native operation on the original worker.
                try{if(worker->ownsCurrentThread())delete original;
                    else worker->submitAndWait([original]{delete original;});}
                catch(...){std::terminate();}
            });
            if(!owner.context)throw std::runtime_error("Recurring owner lost its original device context");
            const auto nodes=appendPooledImmutableDirectedOwnerGraph(*owner.graph,*owner.arena,edge,owner.code,owner.program,"immutable_owner");
            owner.terminal=nodes.last;
            DeviceGraphExecutor executor;executor.setArena(owner.arena.get());
            constexpr auto label="original recurring immutable owner";
            auto& native=owner.executable->capture();
            if(!executor.captureSetupGraphFragment(*owner.graph,owner.context.get(),&native,permit,label))
                throw std::runtime_error("Recurring owner original setup graph recording failed");
            std::string error;
            if(!native.nodeCount() || !native.validateExecutableMemoryClass(GPUGraphExecutableClass::General,&error) ||
               !native.prepareRuntimeContextStorage(witness.source()->memory()) || !native.instantiate())
                throw std::runtime_error("Recurring owner General native instantiation failed: "+error);
            owner.topology=owner.graph->topologyGeneration();
            for(const auto& name:owner.graph->getExecutionOrder()){
                const auto* node=owner.graph->getNode(name);
                owner.leaves.push_back({name,node->stage.get(),node->dependencies,node->stage->graphCaptureVariantSignature()});
            }
        });
        state_->owners.push_back(std::move(owner));
    }
    std::vector<RetainedGraphParticipantEnqueueGroup::Job> jobs;
    jobs.reserve(state_->owners.size());
    for(auto& owner:state_->owners)jobs.push_back({owner.original.worker(),[original=&owner]{
        if(original->submissionOrdinal==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Original remote invocation ownership exhausted");
        if(!original->executable->submit(++original->submissionOrdinal))
            throw std::runtime_error("Original remote native enqueue failed");
    }});
    state_->enqueueGroup.reset(new RetainedGraphParticipantEnqueueGroup(std::move(jobs)));
}
const std::shared_ptr<ImmutableLookupExecutionBoundary>& PooledImmutableRetainedParticipants::boundary()const noexcept{return state_->boundary;}
const std::shared_ptr<const PooledImmutableDirectedTransport>& PooledImmutableRetainedParticipants::transport()const noexcept{return state_->transport;}
void PooledImmutableRetainedParticipants::bindSourcePlan(ComputeGraph& graph,BufferArena& arena,PooledImmutableRetirementCompletion completion){
    if(completion.read.publication.source.boundary!=state_->boundary)throw std::invalid_argument("Recurring source plan substituted original storage");
    graph.setRetainedParticipantSourcePlan(std::make_shared<SourcePlan>(shared_from_this(),arena,std::move(completion)));
}
void PooledImmutableRetainedParticipants::requireCurrent()const{
    if(state_->rootIdentity!=state_->boundary->pooledStorage()->captureIdentity())throw std::logic_error("Recurring owner original roots changed");
    state_->transport->requireCurrent();
    state_->source.requireWorker(GPUDeviceContextPool::instance().retainExistingContext(state_->source.device()));
    for(const auto& owner:state_->owners){
        owner.original.requireWorker(GPUDeviceContextPool::instance().retainExistingContext(owner.original.device()));
        if(!owner.executable || !owner.executable->readyForSubmission() ||
           owner.graph->topologyGeneration()!=owner.topology || owner.graph->getExecutionOrder().size()!=owner.leaves.size())
            throw std::logic_error("Recurring owner original executable/topology changed");
        for(const auto& leaf:owner.leaves){const auto* node=owner.graph->getNode(leaf.name);
            if(!node || node->stage.get()!=leaf.stage || node->dependencies!=leaf.dependencies ||
               node->stage->requireGPUStream()!=owner.original.worker()->defaultStream() || node->stage->graphCaptureVariantSignature()!=leaf.identity)
                throw std::logic_error("Recurring owner original stage/stream/geometry changed");}
    }
}
RetainedGraphParticipantSubmission PooledImmutableRetainedParticipants::enqueueOriginalParticipants()const{
    requireCurrent();state_->boundary->requireArmedForLaunch();
    // A failure after the first native enqueue cannot unwind any shared frame.
    // The exact original CPU ticket remains the sole transaction selection.
    std::unique_lock lock(state_->submissionMutex);
    state_->enqueueGroup->dispatch();
    return RetainedGraphParticipantSubmission(shared_from_this(),std::move(lock));
}
void PooledImmutableRetainedParticipants::completeOriginalEnqueue()const noexcept{
    state_->enqueueGroup->completeOriginalEnqueue();
    // These count successful actual source/remote SDK submissions, not inferred
    // controller progress. Compact live byte counts remain TransferEngine-owned.
    PerfStatsCollector::addCounter("ngram_lookup","retained_source_native_invocations",1.0,
        "hosted_transaction",state_->source.device().toString(),{{"policy","retained_hosted"}});
    PerfStatsCollector::addCounter("ngram_lookup","retained_remote_native_enqueues",static_cast<double>(state_->owners.size()),
        "hosted_transaction",state_->source.device().toString(),{{"policy","retained_hosted"}});
}
void PooledImmutableRetainedParticipants::requireSourceStream(void* stream)const{
    requireCurrent();std::lock_guard lock(state_->submissionMutex);
    if(!state_->source.worker()->ownsCurrentThread() || !stream)
        throw std::invalid_argument("Recurring source operation requires its original source worker and explicit stream");
    if(stream==state_->source.worker()->defaultStream() ||
       (state_->submissionStream && stream==state_->submissionStream->stream()))return;
    for(const auto& borrowed:state_->preparationStreams)if(const auto original=borrowed.lock()){
        if(original->stream()==stream){original->requireLive(state_->source);return;}
    }
    throw std::invalid_argument("Recurring source operation requires its actual retained preparation or parent stream");
}
void PooledImmutableRetainedParticipants::requireSource(const PooledImmutableSourcePreparedStorage& storage,void* stream)const{
    requireSourceStream(stream);
    if(&storage!=state_->boundary->pooledStorage().get())throw std::invalid_argument("Recurring source reset replaced its original storage");
}
void PooledImmutableRetainedParticipants::bindSourceSubmissionStream(const RetainedGraphParticipantParentStream& owner){
    owner.original().requireWorker(state_->source.worker());
    const auto original=state_->source.memory(),parent=owner.original().memory();
    if(owner.original().device()!=state_->source.device() || original.get()!=parent.get() ||
       original.owner_before(parent) || parent.owner_before(original))
        throw std::invalid_argument("Recurring parent stream lost its original source/PMA control block");
    std::lock_guard lock(state_->submissionMutex);
    if(state_->submissionStream && (state_->submissionStream->stream()!=owner.stream() ||
       state_->submissionStream->owner().owner_before(owner.owner()) || owner.owner().owner_before(state_->submissionStream->owner())))
        throw std::invalid_argument("Recurring source cannot reinterpret its original parent stream ownership");
    state_->submissionStream=owner;
}
void PooledImmutableRetainedParticipants::awaitNativeTerminal(std::chrono::steady_clock::time_point deadline)const{
    requireCurrent();
    for(;;){bool complete=true;
        for(const auto& owner:state_->owners)complete=owner.executable->observeTerminal() && complete;
        if(complete)return;
        if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("Recurring owner native completion exceeded the original request terminal deadline");
        std::this_thread::yield();}
}
PooledImmutableRetainedParticipants::~PooledImmutableRetainedParticipants()noexcept{
    if(!state_->ownsBoundaryClaim)return;
    if(state_->boundary->hasActiveInterval() || state_->source.worker()->ownsCurrentThread())std::terminate();
    for(const auto& owner:state_->owners)if(owner.original.worker()->ownsCurrentThread())std::terminate();
    // Each closed executable rejects unobserved live readers before destroying
    // handles; its original device tokens remain process-retained afterwards.
    state_->enqueueGroup.reset();
    state_->owners.clear();
    state_->transport.reset();
    state_->boundary->cohortOwnerClaim_.clear(std::memory_order_release);
}
}
