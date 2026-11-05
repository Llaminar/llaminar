/**
 * @file PooledImmutableRetainedParticipants.cpp
 * @brief Original recurring hosted owner captures, source seals and whole-request retirement.
 *
 * No source model graph is cloned or substituted. Each actual original source
 * parent invocation submits each addressed owner program once. Protocol packet
 * retirement and native executable completion remain distinct authorities.
 */
#include "PooledImmutableRetainedParticipants.h"
#include "PooledImmutableReplicatedReaderGraph.h"
#include "PooledImmutableParticipantExecutionPolicy.h"
#include "backends/RetainedParticipantGraphExecutable.h"
#include "backends/GPUGraphMemoryContract.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableNativePhaseStage.h"
#include "models/GraphTypes.h"
#include "utils/PerfStatsCollector.h"
#include "execution/local_execution/graph/IGraphCaptureAuxiliaryBranch.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include <mutex>
#include <algorithm>
#include <limits>
#include "models/qwen4exp/Qwen4ExpReplicatedPLECohort.h"
#include <thread>
#include <stdexcept>
namespace llaminar2::qwen4exp {
struct PooledImmutableRetainedParticipants::State {
    /** @brief Original cache-local parent lifetime, not a GPU execution/progress ledger. */
    struct AttachedParent {
        std::shared_ptr<const RetainedGraphParticipantPreparationStream> definition;
        std::vector<std::shared_ptr<const RetainedGraphParticipantPreparationStream>> variants; ///< Same original finite owner body, exact source signature variants.
        const ComputeGraph* recipient=nullptr; ///< Actual follower graph identity, never a source ownership substitute.
        const IGPUGraphCapture* native=nullptr;
        void* captureStream=nullptr;
        PreparedGPUContextUsePermit original;
        std::optional<RetainedGraphParticipantParentStream> enclosing;
        bool attempted=false,attached=false; ///< Cold construction only; never GPU progress/retirement.
    };
    enum class ExecutionDisposition {Standalone, EmbeddedParent, InlineReader, EmbeddedExpert};
    struct Owner {
        PreparedGPUContextUsePermit original;
        std::uint32_t coordinate;
        std::optional<PooledRemoteGraphProgram> program; ///< Present only for a separately recorded storage definition.
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
        ExecutionDisposition disposition=ExecutionDisposition::Standalone;
        std::optional<RetainedGraphParticipantParentStream> enclosingParent;
        std::vector<std::weak_ptr<AttachedParent>> parents; ///< Exact cache borrows, never lookup by role/depth/latest.

    };
    std::shared_ptr<ImmutableLookupExecutionBoundary> boundary;
    std::shared_ptr<const PooledImmutableDirectedTransport> transport;
    PreparedGPUContextUsePermit source;
    std::shared_ptr<const Qwen4ExpReplicatedPLECohort> originalCohort; ///< Passive original parent identity; never dereference a retired TP context.
    const IModelContext* originalModel = nullptr; ///< Original model owner, never a copied consumer/source context.
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
    Invocation(std::shared_ptr<PooledImmutableRetainedParticipants> owner,RetainedGraphParticipantSourceIdentity source,
        std::shared_ptr<const RetainedGraphParticipantPreparationStream> definition)
        :owner_(std::move(owner)),source_(source),definition_(std::move(definition)){}
    void requireCurrent(const RetainedGraphParticipantSourceSeal& current)const override {
        if(current.identity()!=source_)throw std::logic_error("Recurring source changed its original parent/arena/stream/variant/generation");
        owner_->requireInvocation(*definition_);
    }
    RetainedGraphParticipantSubmission enqueueOriginalParticipants()const override{return owner_->enqueueOriginalParticipants(*definition_);}
private:
    std::shared_ptr<PooledImmutableRetainedParticipants> owner_;
    RetainedGraphParticipantSourceIdentity source_;
    std::shared_ptr<const RetainedGraphParticipantPreparationStream> definition_;
};
/** @brief One native parent owns this original recorded-body borrow until its existing cache fence retires. */
class PooledImmutableRetainedParticipants::StorageAttachment final:public IRetainedGraphParticipantStorageAttachment {
public:
    StorageAttachment(std::optional<RetainedParticipantGraphExecutable::RecordedStorageDefinition> body,std::shared_ptr<State::AttachedParent> parent)
        :body_(std::move(body)),parent_(std::move(parent)){}
    bool attachTo(IGPUGraphCapture& parent,IGraphCaptureAuxiliaryBranch* progress)override{
        if(parent_->attempted || parent_->native!=&parent || parent.executionStream()!=parent_->captureStream)
            throw std::logic_error("Embedded storage attachment changed its original cold native parent");
        parent_->attempted=true;
        bool attached=false;
        const auto assemble=[&] {
            // Inline storage is already in the recorded main DAG. The original
            // CPU-service progress branch still belongs to this SAME parent;
            // skipping its ordinary attachment would strand hosted execution.
            attached = body_ ? body_->attachTo(parent, progress) : (!progress || progress->attach(parent));
        };
        if(parent_->original.worker()->ownsCurrentThread())assemble();else parent_->original.worker()->submitAndWait(assemble);
        parent_->attached=attached;return attached;
    }
    void requireCurrent()const override{
        if (body_) body_->requireCurrent();
        else parent_->definition->requireDefinition();
        for (const auto& variant : parent_->variants) variant->requireCurrentDefinition();
        if(!parent_->attached || !parent_->native || !parent_->native->hasExecutable() ||
           parent_->native->executionStream()!=parent_->captureStream)
            throw std::logic_error("Embedded storage lost its exact instantiated enclosing FEE parent");
    }
    ~StorageAttachment()override{
        // The existing cache fenced its direct capture stream before destroying
        // native handles. A cloned DGO parent has its separate original owner;
        // never mistake the child recording stream for that terminal proof.
        try{
            const auto complete=[&](void* stream){return parent_->original.worker()->queryStreamExecutionState(
                stream,"original embedded storage cache retirement")==GPUStreamExecutionState::Complete;};
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            while(!complete(parent_->captureStream) || (parent_->enclosing && !complete(parent_->enclosing->stream()))){
                if(std::chrono::steady_clock::now()>=deadline)std::terminate();
                std::this_thread::yield();
            }
        }catch(...){std::terminate();}
        parent_->native=nullptr;
    }
private:
    std::optional<RetainedParticipantGraphExecutable::RecordedStorageDefinition> body_;
    std::shared_ptr<State::AttachedParent> parent_;
};
class PooledImmutableRetainedParticipants::SourcePlan final:public IRetainedGraphParticipantSourcePlan {
    friend class PooledImmutableRetainedParticipants;
public:
    SourcePlan(std::shared_ptr<PooledImmutableRetainedParticipants> owner,BufferArena& arena,
        PooledImmutableSourceRetirementCompletion completion):owner_(std::move(owner)),arena_(&arena),completion_(std::move(completion)){
        // Descriptor identity includes the exact source output roots, not merely
        // a cohort's owner count. Actual topology/recording identity seals later.
        identity_=1469598103934665603ULL;
        const auto fold=[&](std::uint64_t value){identity_^=value;identity_*=1099511628211ULL;};
        fold(owner_->state_->rootIdentity);fold(reinterpret_cast<std::uintptr_t>(arena_));
        fold(reinterpret_cast<std::uintptr_t>(owner_->state_->transport.get()));
        fold(reinterpret_cast<std::uintptr_t>(owner_->state_->executionPolicy.get()));
        for(const auto& owner:owner_->state_->owners){fold(owner.program ? owner.program->captureIdentity() : owner.coordinate);fold(owner.topology);
            fold(reinterpret_cast<std::uintptr_t>(owner.executable.get()));}
        std::visit([&](const auto& completion){
            using T=std::decay_t<decltype(completion)>;
            if constexpr(std::is_same_v<T,PooledImmutableRetirementCompletion>){
                for(const auto& name:{completion.coordinator.first,completion.coordinator.last,
                    completion.readersRetired,completion.observed})for(const unsigned char c:name)fold(c);
            }else{
                if(!completion)throw std::invalid_argument("Source plan requires its original local terminal inventory");
                for(const auto& name:{completion->firstNode(),completion->lastNode(),completion->readersRetiredNode(),completion->readNode()})
                    for(const unsigned char c:name)fold(c);
            }
        },completion_);
        if(!identity_)identity_=1;
    }
    const PreparedGPUContextUsePermit& originalSource()const noexcept override{return owner_->state_->source;}
    RetainedGraphParticipantDefinitionDomain definitionDomain()const noexcept override{
        return {owner_->state_->source.device(),owner_->state_->transport.get()};
    }
    void requirePreparedSourceDefinition(const RetainedGraphParticipantPreparationStream& preparation)const override{
        owner_->requireCurrent();
        preparation.requireLive(originalSource());
        const auto& definition=preparation.definition();
        if(definition.plan.get()!=this || definition.domain!=definitionDomain())
            throw std::invalid_argument("Prepared source changed its original source/transport plan");
        requireDeclaration(definition.graph,definition.arena,originalSource().device(),preparation.stream(),
            definition.topology_generation,definition.variant);
    }
    void requireCurrentPreparedSourceDefinition(const RetainedGraphParticipantPreparationStream& preparation)const override{
        owner_->requireCurrent();
        preparation.requireLive(originalSource());const auto& definition=preparation.definition();
        originalSource().requireWorker(GPUDeviceContextPool::instance().retainExistingContext(originalSource().device()));
        if(definition.plan.get()!=this || definition.domain!=definitionDomain() || !definition.graph || definition.arena!=arena_ ||
           definition.graph->topologyGeneration()!=definition.topology_generation || definition.graph->retainedParticipantSourcePlan().get()!=this)
            throw std::logic_error("Sealed source definition changed its original cache/topology/plan");
    }
    void requireCurrentStorageDefinition(const RetainedParticipantGraphExecutable& definition)const override{
        owner_->requireCurrent();
        const auto& owners=owner_->state_->owners;
        const auto found=std::find_if(owners.begin(),owners.end(),[&](const auto& physical){return physical.executable.get()==&definition;});
        if(found==owners.end() || found->graph->topologyGeneration()!=found->topology)
            throw std::logic_error("Sealed storage definition changed its original owner/topology");
        found->original.requireWorker(GPUDeviceContextPool::instance().retainExistingContext(found->original.device()));
    }
    void requireDeclaration(const ComputeGraph* graph,const void* arena,DeviceId device,void* stream,
        uint64_t topology,uint64_t variant)const{
        owner_->requireCurrent();
        if(owner_->state_->rootIdentity!=owner_->state_->boundary->pooledStorage()->captureIdentity())
            throw std::logic_error("Original declarative source roots changed");
        owner_->state_->transport->requireCurrent();
        originalSource().requireWorker(GPUDeviceContextPool::instance().retainExistingContext(originalSource().device()));
        if(!graph || arena!=arena_ || device!=owner_->state_->source.device() ||
           !stream || reinterpret_cast<std::uintptr_t>(stream)<=2 || !topology || !variant ||
           graph->topologyGeneration()!=topology || graph->retainedParticipantSourcePlan().get()!=this ||
           !ownsHeterogeneousTicketBoundary(graph->nativeCaptureEnvelope()))
            throw std::invalid_argument("Source definition lost its original graph/arena/count transaction");
        requirePooledSourceRetirementCompletion(*graph,completion_);
    }
    void requirePreparedStorageDefinition(const RetainedParticipantGraphExecutable& definition)const override{
        const auto found=std::find_if(owner_->state_->owners.begin(),owner_->state_->owners.end(),
            [&](const auto& physical){return physical.executable.get()==&definition;});
        if(found==owner_->state_->owners.end())throw std::invalid_argument("Source definition substituted its original storage graph owner");
        found->original.requireWorker(GPUDeviceContextPool::instance().retainExistingContext(found->original.device()));
        if(found->graph->topologyGeneration()!=found->topology || found->graph->getExecutionOrder().size()!=found->leaves.size())
            throw std::logic_error("Original storage definition topology changed");
        for(const auto& leaf:found->leaves){const auto* node=found->graph->getNode(leaf.name);
            if(!node || node->stage.get()!=leaf.stage || node->dependencies!=leaf.dependencies ||
               node->stage->requireGPUStream()!=found->original.worker()->defaultStream() ||
               node->stage->graphCaptureVariantSignature()!=leaf.identity)
                throw std::logic_error("Original storage definition stages changed");
        }
    }
    void acquirePreparedStorageDefinition(const RetainedParticipantGraphExecutable& definition,void* parentStream)const override{
        requirePreparedStorageDefinition(definition);
        const auto found=std::find_if(owner_->state_->owners.begin(),owner_->state_->owners.end(),
            [&](const auto& physical){return physical.executable.get()==&definition;});
        if(!parentStream || reinterpret_cast<std::uintptr_t>(parentStream)<=2 || !found->original.worker()->ownsCurrentThread())
            throw std::invalid_argument("Storage definition input frontier requires its original worker and exact parent stream");
        owner_->state_->boundary->pooledStorage()->originalOwners()[found->coordinate]->acquireSetup(parentStream);
    }
    std::unique_ptr<IRetainedGraphParticipantStorageAttachment> prepareOriginalStorageAttachment(
        DeviceId receiver,IGPUGraphCapture& parent,IGraphCaptureAuxiliaryBranch* progress,
        std::shared_ptr<const RetainedGraphParticipantPreparationStream> definition,
        const ComputeGraph& receiverGraph)const override{
        if(!definition || definition->definition().plan.get()!=this)throw std::invalid_argument("Finite storage attachment lost its exact original source plan");
        definition->requireDefinition();
        if (!receiverGraph.retainedParticipantDefinitionDomain() ||
            *receiverGraph.retainedParticipantDefinitionDomain() != definition->definition().domain)
            throw std::invalid_argument("Storage attachment changed its original actual recipient graph/domain");
        auto& owners=owner_->state_->owners;
        const auto found=std::find_if(owners.begin(),owners.end(),[&](const auto& physical){return physical.original.device()==receiver;});
        if(found==owners.end() || found->disposition==State::ExecutionDisposition::Standalone)return {};
        if(progress && (progress->device()!=receiver || progress->originalSourceDefinition()!=definition))
            throw std::logic_error("Embedded mapped storage requires its actual original progress branch and matched token");
        if(parent.hasExecutable() || !parent.nodeCount() || reinterpret_cast<std::uintptr_t>(parent.executionStream())<=2)
            throw std::invalid_argument("Finite storage attachment requires its original uninstantiated parent");
        for(const auto& weak:found->parents)if(const auto prior=weak.lock())if(prior->definition==definition || std::find(prior->variants.begin(),prior->variants.end(),definition)!=prior->variants.end())
            throw std::logic_error("One source-cache definition already owns this physical participant attachment");
        const auto& variants = receiverGraph.retainedParticipantSourceVariants();
        if (found->disposition == State::ExecutionDisposition::EmbeddedExpert) {
            if (variants.empty() || variants.front() != definition)
                throw std::invalid_argument("Expert owner attachment lost its exact finite source-family association");
            for (const auto& variant : variants) {
                variant->requireCurrentDefinition();
                const auto original = std::dynamic_pointer_cast<const SourcePlan>(variant->definition().plan);
                if (!original || original->owner_ != owner_)
                    throw std::invalid_argument("Expert owner attachment substituted its original source plan");
            }
        } else if (!variants.empty()) {
            throw std::invalid_argument("Reader attachment cannot adopt an expert source-family association");
        }
        std::optional<RetainedParticipantGraphExecutable::RecordedStorageDefinition> body;
        if (found->disposition == State::ExecutionDisposition::InlineReader) {
            const auto consumer = findPooledReplicatedConsumerCompletion(receiverGraph, owner_->state_->transport, receiver);
            const auto& edge = owner_->state_->transport->remote(found->coordinate);
            const auto rows = findPooledReadOwnerCompletion(receiverGraph, edge.read);
            (void)findPooledRetirementOwnerCompletion(receiverGraph, edge.retirement, consumer);
            const auto* header = receiverGraph.getNode(consumer.header);
            if (!header || std::find(header->dependencies.begin(), header->dependencies.end(), rows) == header->dependencies.end())
                throw std::invalid_argument("Inline owner result lost its exact original Read frontier");
        } else {
            body.emplace(found->executable->retainRecordedStorageDefinition(found->original,definition));
        }
        // Finish every fallible host allocation/weak registration BEFORE native
        // mutation. The canonical DGE installs this RAII object into its cache
        // before attachTo; even partial SDK failure retains the borrow until
        // the original parent handle has been discarded.
        auto lifetime=std::make_shared<State::AttachedParent>(State::AttachedParent{.definition=definition,.variants=receiverGraph.retainedParticipantSourceVariants(),
            .recipient=&receiverGraph,.native=&parent,.captureStream=parent.executionStream(),.original=found->original,.enclosing=found->enclosingParent});
        auto result=std::make_unique<StorageAttachment>(std::move(body),lifetime);
        found->parents.push_back(lifetime);
        return result;
    }
    void authenticatePreparationStream(std::shared_ptr<const RetainedGraphParticipantPreparationStream> preparation)const override{
        if(!preparation)throw std::invalid_argument("Original source preparation lost its cache authority");
        preparation->requireDefinition();
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
        requireDeclaration(source.graph,source.arena,source.device,source.recording_stream,
            source.topology_generation,source.capture_variant_signature);
        if(!source.graph || source.arena!=arena_ || source.device!=owner_->state_->source.device() ||
           !source.original_parent || !source.original_parent->hasExecutable() ||
           source.original_parent->executionStream()!=source.recording_stream ||
           !source.recording_stream || reinterpret_cast<std::uintptr_t>(source.recording_stream)<=2 ||
           !source.capture_generation || !source.capture_variant_signature ||
           !ownsHeterogeneousTicketBoundary(source.graph->nativeCaptureEnvelope()) ||
           source.graph->retainedParticipantSourcePlan().get()!=this)
            throw std::invalid_argument("Recurring participant binding requires the actual complete original hosted source parent");
        requirePooledSourceRetirementCompletion(*source.graph,completion_);
        owner_->requireCurrent();
        std::shared_ptr<const RetainedGraphParticipantPreparationStream> definition;
        for(const auto& weak:owner_->state_->preparationStreams)if(const auto candidate=weak.lock()){
            if(candidate->definition().graph==source.graph && candidate->stream()==source.recording_stream &&
               candidate->definition().topology_generation==source.topology_generation &&
               candidate->definition().variant==source.capture_variant_signature){
                if(definition && definition!=candidate)throw std::logic_error("Source seal aliases distinct original cache tokens");
                definition=candidate;
            }
        }
        if(!definition)throw std::logic_error("Executable source seal lost its exact original preparation token");
        definition->requireDefinition();
        return std::make_shared<Invocation>(owner_,source,std::move(definition));
    }
private:
    std::shared_ptr<PooledImmutableRetainedParticipants> owner_;
    BufferArena* arena_;
    PooledImmutableSourceRetirementCompletion completion_;
    std::uint64_t identity_;
};
PooledImmutableRetainedParticipants::PooledImmutableRetainedParticipants(std::unique_ptr<State> state):state_(std::move(state)){}
std::shared_ptr<PooledImmutableRetainedParticipants> PooledImmutableRetainedParticipants::create(
    std::shared_ptr<ImmutableLookupExecutionBoundary> boundary,const GraphConfig& policy,std::shared_ptr<const ClusterInventory> originalInventory,
    std::optional<PooledImmutableReplicatedReaderGeometry> readers){
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
    if (dense == execution.domains.end())
        throw std::invalid_argument("Recurring owner composition lost its actual continuation domain");
    if (readers) {
        if (readers->coordinator().device != boundary->device() ||
            readers->consumers().size() != dense->participants.size() ||
            !policy.ngram_replicated_ple_cohort || !policy.ngram_table_placement ||
            policy.ngram_replicated_ple_cohort->placement() != *policy.ngram_table_placement ||
            &policy.ngram_replicated_ple_cohort->context() != policy.tp_ctx)
            throw std::invalid_argument("Shared owners require their actual complete Rank replicated-query cohort");
        for (const auto& participant : dense->participants) {
            if (!participant.owned_by_current_rank || !participant.locally_addressable ||
                participant.world_rank != readers->coordinator().world_rank ||
                std::none_of(readers->consumers().begin(), readers->consumers().end(), [&](const auto& reader) {
                    return reader.world_rank == participant.world_rank && reader.device == participant.local_device;
                }))
                throw std::invalid_argument("Shared owners changed an original continuation reader or rank");
        }
    } else if (dense->participants.size() != 1 ||
        !dense->participants.front().owned_by_current_rank || !dense->participants.front().locally_addressable ||
        dense->participants.front().local_device != boundary->device()) {
        throw std::invalid_argument("Plural continuation requires its original shared reader/source wiring");
    }
    const auto storage=boundary->pooledStorage();const auto& witness=storage->retainedPreparedSourceWorkers();
    if(!witness || witness->rootIdentity()!=storage->captureIdentity() || witness->coordinator().device()!=boundary->device())
        throw std::invalid_argument("Recurring remote owners require actually consumed original prepared source workers");
    bool cpu=false;
    for(const auto& domain:execution.domains)for(const auto& participant:domain.participants){
        cpu=cpu || participant.local_device.is_cpu();
        for(const auto& owner:witness->owners())
            if(owner.device()!=boundary->device() && participant.local_device==owner.device()){
                const bool embedded=readers && std::any_of(readers->consumers().begin(),readers->consumers().end(),[&](const auto& consumer){
                    return consumer.device==owner.device() && consumer.world_rank==participant.world_rank;
                });
                if (!embedded && (!participant.owned_by_current_rank || !participant.locally_addressable ||
                        participant.world_rank != storage->coordinator()->plan().allocator().world_rank))
                    throw std::invalid_argument("Colocated expert storage requires its exact original rank-local follower");
            }
    }
    const auto isReader = [&](const auto& owner) {
        return readers && std::any_of(readers->consumers().begin(),readers->consumers().end(),[&](const auto& reader) {
            return reader.device == owner.device();
        });
    };
    const auto dedicatedOwners = std::count_if(witness->owners().begin(),witness->owners().end(),[&](const auto& owner) {
        return owner.device() != boundary->device() && !isReader(owner);
    });
    if (dedicatedOwners > 1) for (const auto& owner : witness->owners()) {
        if (owner.device() == boundary->device() || isReader(owner)) continue;
        const bool expert = PooledImmutableParticipantExecutionPolicy::declaresExpertOwner(execution,owner);
        if (!expert) throw std::invalid_argument("Multiple mapped storage owners require their actual original expert follower parents");
    }
    if(!cpu)throw std::invalid_argument("Recurring remote owner policy lost its explicitly declared CPU expert boundary");
    if(witness->coordinator().worker()->ownsCurrentThread())throw std::logic_error("Recurring owner preparation must run off its original source worker");
    std::unique_ptr<RetainedGraphParticipantEnqueueGroup> enqueueGroup;
    std::vector<PooledRemoteGraphProgram> programs;
    for(const auto& owner:witness->owners()){
        if(owner.worker()->ownsCurrentThread())throw std::logic_error("Recurring owner preparation must run off every original participant worker");
        if(owner.device()!=boundary->device()){
            const auto coordinate=static_cast<std::uint32_t>(&owner-witness->owners().data());
            const bool inlineNative = readers && policy.ngram_replicated_ple_cohort &&
                owner.device().type == boundary->device().type &&
                std::any_of(readers->consumers().begin(), readers->consumers().end(),
                    [&](const auto& reader) { return reader.device == owner.device(); });
            if (inlineNative) continue;
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
    if (readers) {
        if (!policy.ngram_replicated_ple_cohort || !policy.ngram_table_placement ||
            policy.ngram_replicated_ple_cohort->placement() != *policy.ngram_table_placement ||
            readers->coordinator().device != boundary->device() ||
            readers->consumers().size() != storage->geometry().parameters().consumers)
            throw std::invalid_argument("Shared retained owners require the original Rank replicated query declaration");
        policy.ngram_replicated_ple_cohort->requireConsumer(policy.default_device, policy.tp_device_idx);
        result->state_->originalModel = policy.ngram_replicated_ple_cohort->modelOwner();
        result->state_->originalCohort = policy.ngram_replicated_ple_cohort;
    }
    result->state_->programs=std::move(programs);
    result->state_->transport=PooledImmutableDirectedTransport::create(boundary,std::move(originalInventory), std::move(readers),
        policy.ngram_replicated_ple_cohort);
    if(result->state_->transport->remotes().empty() && result->state_->transport->readers().empty())
        throw std::invalid_argument("Retained participant binding has neither remote storage owners nor compute readers");
    result->prepareOriginalOwners();
    boundary->bindRetainedParticipants(result);
    return result;
}
bool PooledImmutableParticipantExecutionPolicy::requiresRetainedParticipants(
    const PooledImmutableSourceGeometry& source) noexcept
{
    // Storage ownership and result consumption are independent. In particular,
    // one coordinator-owned cache may serve every member of a native TP cohort.
    return source.parameters().consumers > 1 ||
        std::any_of(source.originalOwners().begin(), source.originalOwners().end(),
            [&](const auto& owner) { return owner.allocator() != source.allocator(); });
}

bool PooledImmutableParticipantExecutionPolicy::declaresExpertOwner(
    const MoEExpertOverlayExecutionPlan& original, const PreparedGPUContextUsePermit& owner)
{
    if (!owner.device().is_gpu() || !owner.memory()) return false;
    return std::any_of(original.domains.begin(),original.domains.end(),[&](const auto& domain) {
        return domain.name != original.continuation_domain &&
            std::any_of(domain.participants.begin(),domain.participants.end(),[&](const auto& participant) {
                return participant.local_device == owner.device() && participant.owned_by_current_rank &&
                    participant.locally_addressable && participant.world_rank == owner.memory()->worldRank();
            });
    });
}
void PooledImmutableParticipantExecutionPolicy::requireGraphDescriptor(
    const std::shared_ptr<const MoEExpertOverlayExecutionPlan>& original,
    const GraphConfig& participant, GraphRole role)
{
    const auto& current = participant.moe.expert_overlay_execution_plan;
    if (!original || !current || participant.n_layers <= 0 || participant.pp_layer_offset != 0 ||
        (participant.total_n_layers > 0 && participant.n_layers != participant.total_n_layers) ||
        !participant.pp_contexts.empty() || current->world_size != 1 ||
        !current->ownsContinuationGraph() ||
        current->generationExecutionDomain() != DeviceGenerationExecutionDomain::HostedBoundary)
        throw std::invalid_argument("Retained remote source requires its complete hosted continuation descriptor");
    // Each sibling factory derives a new immutable descriptor from its frozen
    // placement. Pointer equality is meaningful for the source's own retained
    // graph, but would reject every genuine reader. The caller independently
    // authenticates the exact model, Rank cohort, communicator and participant.
    switch (role)
    {
    case GraphRole::Source:
        if (current != original || current.owner_before(original) || original.owner_before(current))
            throw std::invalid_argument("Retained source replaced its original execution descriptor");
        return;
    case GraphRole::ReplicatedReader:
        if (*current != *original)
            throw std::invalid_argument("Retained reader changed its original hosted execution descriptor");
        return;
    }
    throw std::invalid_argument("Retained graph has an unknown source/reader role");
}

void PooledImmutableRetainedParticipants::requireForPolicy(const GraphConfig& policy)const {
    PooledImmutableParticipantExecutionPolicy::requireConfiguration(policy.mtp);
    requireCurrent();
    const bool source = policy.default_device == state_->source.device();
    const auto& readers = state_->transport->readerGeometry();
    if (readers) {
        const auto& cohort = policy.ngram_replicated_ple_cohort;
        if (!cohort || cohort.get() != state_->originalCohort.get() ||
            cohort.owner_before(state_->originalCohort) || state_->originalCohort.owner_before(cohort) ||
            !state_->originalModel || cohort->modelOwner() != state_->originalModel ||
            !policy.ngram_table_placement || cohort->placement() != *policy.ngram_table_placement ||
            &cohort->context() != policy.tp_ctx ||
            std::none_of(readers->consumers().begin(), readers->consumers().end(), [&](const auto& consumer) {
                return consumer.world_rank == readers->coordinator().world_rank && consumer.device == policy.default_device;
            }))
            throw std::invalid_argument("Shared retained reader lost its original Rank, model, source or device membership");
        cohort->requireConsumer(policy.default_device, policy.tp_device_idx);
    } else if (!source) {
        throw std::invalid_argument("Singleton retained source cannot manufacture a foreign compute reader");
    }
    PooledImmutableParticipantExecutionPolicy::requireGraphDescriptor(state_->executionPolicy, policy,
        source ? PooledImmutableParticipantExecutionPolicy::GraphRole::Source :
                 PooledImmutableParticipantExecutionPolicy::GraphRole::ReplicatedReader);
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
    if(ids.bytes!=static_cast<std::size_t>(expanded.physicalRows())*sizeof(std::uint64_t) || ids.offset%alignof(std::uint64_t) ||
       output.bytes!=static_cast<std::size_t>(expanded.physicalRows())*declared.row_elements*sizeof(float))
        throw std::invalid_argument("Directed PLE requires exact original hash and output extents");
    PooledNativeOperandSpans begin{};
    begin[static_cast<unsigned>(PooledNativeOperand::Ids)]=std::move(ids);
    if(count)begin[static_cast<unsigned>(PooledNativeOperand::OriginalCount)]=std::move(*count);
    begin[static_cast<unsigned>(PooledNativeOperand::Admission)]=PooledNativeTensorSpan{
        storage->roots().native,BufferId::QFN_POOLED_SOURCE_NATIVE,
        declared.offsets.protocol+offsetof(ImmutableLookupDeviceState,admissionGeneration),sizeof(std::uint64_t)};
    auto endpoints=appendPooledImmutableDirectedLookupGraph(graph,arena,state_->boundary,state_->transport,std::move(code),
        {expanded,pooled_native::OriginalIdWidth::UInt64},std::move(begin),control,std::move(output),identity,outputIdentity);
    graph.setRetainedParticipantDefinitionDomain({state_->source.device(),state_->transport.get()});
    bindSourcePlan(graph,arena,std::move(endpoints.retirement));
    return endpoints.source;
}
PooledImmutableLookupGraphEndpoints PooledImmutableRetainedParticipants::appendReader(
    ComputeGraph& graph, BufferArena& arena, DeviceGenerationControlBinding control, DeviceRowRange rows,
    PooledNativeTensorSpan output, const std::string& identity, const std::string& outputIdentity,
    std::optional<PooledNativeTensorSpan> count)
{
    requireCurrent();
    const auto actualDevice = control.owner()->current_device();
    if (!actualDevice) throw std::invalid_argument("Inline reader lost its original controller device");
    const auto device = *actualDevice;
    const auto owner = std::find_if(state_->owners.begin(), state_->owners.end(), [&](const auto& candidate) {
        return candidate.original.device() == device && candidate.disposition == State::ExecutionDisposition::InlineReader;
    });
    if (owner == state_->owners.end())
        return appendPooledReplicatedReader(graph, arena, state_->transport, control, rows, output, identity, outputIdentity, std::move(count));
    const auto& edge = state_->transport->remote(owner->coordinate);
    PooledImmutableDemandFrontier demand;
    const auto prefix = appendPooledImmutableInlineOwnerPrefix(graph, arena, edge, owner->code, identity + "/owner",&demand);
    const auto reader = appendPooledReplicatedReader(graph, arena, state_->transport, control, rows, output,
        identity, outputIdentity, std::move(count));
    graph.addDependency(reader.first, prefix.last);
    const auto consumer = findPooledReplicatedConsumerCompletion(graph, state_->transport, device);
    const auto suffix = appendPooledImmutableInlineOwnerRetirement(graph, arena, edge, owner->code,
        prefix.last, consumer, identity + "/owner_retirement");
    return {prefix.first, suffix.last,std::move(demand)};
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
        const bool inlineNative = state_->transport->readerGeometry() &&
            permit.device().type == state_->source.device().type &&
            std::any_of(state_->transport->readerGeometry()->consumers().begin(),state_->transport->readerGeometry()->consumers().end(),
                [&](const auto& reader){return reader.device==permit.device();});
        if (inlineNative) {
            state_->owners.push_back(State::Owner{.original=permit, .coordinate=edge.coordinate,
                .code=std::move(code), .disposition=State::ExecutionDisposition::InlineReader});
            continue;
        }
        const auto declared=std::find_if(state_->programs.begin(),state_->programs.end(),[&](const auto& program){return program.geometry().coordinate==edge.coordinate;});
        if(declared==state_->programs.end())throw std::logic_error("Recurring original owner lost its admitted source program");
        State::Owner owner{.original=permit,.coordinate=edge.coordinate,.program=*declared,
            .arena=std::make_shared<BufferArena>(),.graph=std::make_shared<ComputeGraph>(),.code=std::move(code)};
        if(state_->transport->readerGeometry() && std::any_of(state_->transport->readerGeometry()->consumers().begin(),state_->transport->readerGeometry()->consumers().end(),
            [&](const auto& reader){return reader.device==permit.device();}))owner.disposition=State::ExecutionDisposition::EmbeddedParent;
        if (state_->executionPolicy && PooledImmutableParticipantExecutionPolicy::declaresExpertOwner(*state_->executionPolicy,permit) &&
            owner.disposition == State::ExecutionDisposition::Standalone)
            owner.disposition = State::ExecutionDisposition::EmbeddedExpert;
        void* originalStream=nullptr;
        permit.worker()->submitAndWait([&]{originalStream=permit.worker()->defaultStream();});
        owner.executable=RetainedParticipantGraphExecutable::create(permit,originalStream,*owner.program);
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
            const auto nodes=appendPooledImmutableDirectedOwnerGraph(*owner.graph,*owner.arena,edge,owner.code,*owner.program,"immutable_owner");
            owner.terminal=nodes.last;
            DeviceGraphExecutor executor;executor.setArena(owner.arena.get());
            constexpr auto label="original recurring immutable owner";
            auto& native=owner.executable->capture();
            if(!executor.captureSetupGraphFragment(*owner.graph,owner.context.get(),&native,permit,label))
                throw std::runtime_error("Recurring owner original setup graph recording failed");
            std::string error;
            if(!native.nodeCount() || !native.validateExecutableMemoryClass(GPUGraphExecutableClass::General,&error) ||
               !native.prepareRuntimeContextStorage(witness.source()->memory()) ||
               (owner.disposition==State::ExecutionDisposition::Standalone && !native.instantiate()))
                throw std::runtime_error("Recurring owner General native instantiation failed: "+error);
            if(owner.disposition!=State::ExecutionDisposition::Standalone)owner.executable->sealRecordedStorageBody();
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
    for(auto& owner:state_->owners)if(owner.disposition==State::ExecutionDisposition::Standalone)jobs.push_back({owner.original.worker(),[original=&owner]{
        if(original->submissionOrdinal==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Original remote invocation ownership exhausted");
        if(!original->executable->submit(++original->submissionOrdinal))
            throw std::runtime_error("Original remote native enqueue failed");
    }});
    if(!jobs.empty())state_->enqueueGroup.reset(new RetainedGraphParticipantEnqueueGroup(std::move(jobs)));
}
const std::shared_ptr<ImmutableLookupExecutionBoundary>& PooledImmutableRetainedParticipants::boundary()const noexcept{return state_->boundary;}
const std::shared_ptr<const PooledImmutableDirectedTransport>& PooledImmutableRetainedParticipants::transport()const noexcept{return state_->transport;}
void PooledImmutableRetainedParticipants::bindSourcePlan(ComputeGraph& graph,BufferArena& arena,PooledImmutableSourceRetirementCompletion completion){
    const auto boundary=std::visit([](const auto& value){
        using T=std::decay_t<decltype(value)>;
        if constexpr(std::is_same_v<T,PooledImmutableRetirementCompletion>)return value.read.publication.source.boundary;
        else return value?value->sourceBoundary():std::shared_ptr<ImmutableLookupExecutionBoundary>{};
    },completion);
    if(boundary!=state_->boundary)throw std::invalid_argument("Recurring source plan substituted original storage");
    requirePooledSourceRetirementCompletion(graph,completion);
    graph.setRetainedParticipantSourcePlan(std::make_shared<SourcePlan>(shared_from_this(),arena,std::move(completion)));
}
void PooledImmutableRetainedParticipants::requireCurrent()const{
    if (!state_->ownsBoundaryClaim) throw std::logic_error("Original pooled execution participants retired with their runner");
    if(state_->rootIdentity!=state_->boundary->pooledStorage()->captureIdentity())throw std::logic_error("Recurring owner original roots changed");
    state_->transport->requireCurrent();
    state_->source.requireWorker(GPUDeviceContextPool::instance().retainExistingContext(state_->source.device()));
    for(const auto& owner:state_->owners){
        owner.original.requireWorker(GPUDeviceContextPool::instance().retainExistingContext(owner.original.device()));
        if (owner.disposition == State::ExecutionDisposition::InlineReader) {
            if (owner.executable || owner.program || owner.graph || owner.arena || !owner.code)
                throw std::logic_error("Inline owner acquired an independent graph or lost its original native code");
            continue;
        }
        if(!owner.executable || (owner.disposition==State::ExecutionDisposition::Standalone ? !owner.executable->readyForSubmission() :
            owner.executable->phase_!=RetainedParticipantGraphExecutable::Phase::RecordedStorageBody) ||
           owner.graph->topologyGeneration()!=owner.topology || owner.graph->getExecutionOrder().size()!=owner.leaves.size())
            throw std::logic_error("Recurring owner original executable/topology changed");
        for(const auto& leaf:owner.leaves){const auto* node=owner.graph->getNode(leaf.name);
            if(!node || node->stage.get()!=leaf.stage || node->dependencies!=leaf.dependencies ||
               node->stage->requireGPUStream()!=owner.original.worker()->defaultStream() || node->stage->graphCaptureVariantSignature()!=leaf.identity)
                throw std::logic_error("Recurring owner original stage/stream/geometry changed");}
    }
}
bool PooledImmutableRetainedParticipants::hasEmbeddedParticipant(DeviceId device)const noexcept{
    return std::any_of(state_->owners.begin(),state_->owners.end(),[&](const auto& owner){return owner.original.device()==device && owner.disposition!=State::ExecutionDisposition::Standalone;});
}
PreparedGPUContextUsePermit PooledImmutableRetainedParticipants::originalParticipant(DeviceId device)const{
    if(device==state_->source.device())return state_->source;
    const auto found=std::find_if(state_->owners.begin(),state_->owners.end(),[&](const auto& owner){return owner.original.device()==device && owner.disposition!=State::ExecutionDisposition::Standalone;});
    if(found==state_->owners.end())throw std::invalid_argument("Parent stream requires an original embedded storage participant");
    return found->original;
}
void PooledImmutableRetainedParticipants::bindEmbeddedParentSubmissionStream(const RetainedGraphParticipantParentStream& parent){
    auto& owners=state_->owners;const auto found=std::find_if(owners.begin(),owners.end(),[&](const auto& owner){return owner.original.device()==parent.original().device() && owner.disposition!=State::ExecutionDisposition::Standalone;});
    if(found==owners.end())throw std::invalid_argument("Enclosing parent is not an original embedded participant");
    parent.original().requireWorker(found->original.worker());
    const auto same=[](const auto& a,const auto& b){return a.get()==b.get() && !a.owner_before(b) && !b.owner_before(a);};
    if(!same(parent.original().memory(),found->original.memory()) || (found->enclosingParent &&
       (parent.stream()!=found->enclosingParent->stream() || !same(parent.owner(),found->enclosingParent->owner()))))
        throw std::invalid_argument("Embedded parent changed its original PMA or native stream owner");
    found->enclosingParent=parent;
    for(const auto& weak:found->parents)if(const auto original=weak.lock())original->enclosing=parent;
    // Join the exact original setup/reset publication onto the actual enclosing
    // stream while still cold; no default-stream inference or host wait occurs.
    const auto acquire=[&]{state_->boundary->pooledStorage()->originalOwners()[found->coordinate]->acquireSetup(parent.stream());};
    if(found->original.worker()->ownsCurrentThread())acquire();else found->original.worker()->submitAndWait(acquire);
}
std::vector<PooledImmutableRetainedParticipants::ExpertOwnerParentObservation>
PooledImmutableRetainedParticipants::expertOwnerParentObservations() const {
    requireCurrent();
    std::vector<ExpertOwnerParentObservation> result;
    for (const auto& owner : state_->owners) {
        if (owner.disposition != State::ExecutionDisposition::EmbeddedExpert) continue;
        ExpertOwnerParentObservation observed{owner.coordinate,
            state_->boundary->pooledStorage()->originalOwners()[owner.coordinate]->plan().allocator(),
            owner.original.memory().get(),owner.original.generation(),owner.original.worker(),{},0};
        for (const auto& weak : owner.parents) if (const auto parent=weak.lock()) {
            if (!parent->attached || !parent->native || !parent->native->hasExecutable() || parent->variants.empty())
                throw std::logic_error("Original expert storage observation found an incomplete native parent association");
            for (const auto& variant : parent->variants) variant->requireCurrentDefinition();
            observed.parent_lifetimes.emplace_back(parent);
            observed.source_variant_count += parent->variants.size();
        }
        result.push_back(std::move(observed));
    }
    return result;
}
bool PooledImmutableRetainedParticipants::hasOriginalExpertOwner(DeviceId receiver) const noexcept {
    return std::any_of(state_->owners.begin(), state_->owners.end(), [&](const auto& owner) {
        return owner.original.device() == receiver && owner.disposition == State::ExecutionDisposition::EmbeddedExpert;
    });
}
void PooledImmutableRetainedParticipants::requireOriginalExpertSourceVariants(DeviceId receiver,
    std::span<const std::shared_ptr<const RetainedGraphParticipantPreparationStream>> definitions) const {
    if (!hasOriginalExpertOwner(receiver) || definitions.empty() || isGraphCaptureActive())
        throw std::invalid_argument("Expert source association requires its original physical owner and complete cold family");
    for (std::size_t i = 0; i < definitions.size(); ++i) {
        const auto& definition = definitions[i];
        if (!definition || std::find(definitions.begin(), definitions.begin()+i, definition) != definitions.begin()+i)
            throw std::invalid_argument("Expert source association contains absent or duplicate definitions");
        definition->requireCurrentDefinition();
        const auto plan = std::dynamic_pointer_cast<const SourcePlan>(definition->definition().plan);
        if (!plan || plan->owner_.get() != this)
            throw std::invalid_argument("Expert source association changed its original pooled source");
        std::size_t occurrences = 0;
        for (const auto& name : definition->definition().graph->getExecutionOrder()) {
            const auto* node = definition->definition().graph->getNode(name);
            const auto* phase = node ? dynamic_cast<const PooledImmutableNativePhaseStage*>(node->stage.get()) : nullptr;
            if (!phase) continue;
            const auto* coordinator = std::get_if<PooledImmutableNativePhaseStage::Coordinator>(&phase->phase());
            if (coordinator && coordinator->action == pooled_native::CoordinatorAction::Begin) {
                if (coordinator->storage != state_->boundary->pooledStorage()->coordinator())
                    throw std::invalid_argument("Expert source association contains a foreign coordinator occurrence");
                ++occurrences;
            }
        }
        if (occurrences != 1)
            throw std::invalid_argument("Expert source family requires one actual finite declared source occurrence");
    }
}
void PooledImmutableRetainedParticipants::bindOriginalExpertSourceVariants(DeviceId receiver, ComputeGraph& graph,
    std::vector<std::shared_ptr<const RetainedGraphParticipantPreparationStream>> definitions) const {
    requireOriginalExpertSourceVariants(receiver,definitions);
    graph.setRetainedParticipantSourceVariants(std::move(definitions));
}
void PooledImmutableRetainedParticipants::refreshOriginalExpertSourceVariants(
    std::span<const OriginalExpertSourceBinding> bindings) const {
    requireCurrent();
    struct Staged {
        std::shared_ptr<State::AttachedParent> parent;
        std::vector<std::shared_ptr<const RetainedGraphParticipantPreparationStream>> variants;
    };
    std::vector<Staged> staged;
    for (const auto& binding : bindings) {
        if (!binding.graph || !hasOriginalExpertOwner(binding.receiver))
            throw std::invalid_argument("Expert source refresh lost its original follower graph/owner");
        if (binding.definitions.empty()) {
            if (!binding.graph->retainedParticipantSourceVariants().empty())
                throw std::invalid_argument("Expert source refresh removed its original native storage sibling");
            continue;
        }
        if (binding.graph->retainedParticipantSourceVariants().empty())
            throw std::invalid_argument("Expert source refresh cannot add a native sibling to a source-free parent");
        requireOriginalExpertSourceVariants(binding.receiver,binding.definitions);
        const auto owner = std::find_if(state_->owners.begin(),state_->owners.end(),[&](const auto& owner) {
            return owner.original.device() == binding.receiver;
        });
        std::shared_ptr<State::AttachedParent> original;
        for (const auto& weak : owner->parents) if (const auto parent = weak.lock()) {
            if (parent->recipient != binding.graph) continue;
            if (original || !parent->attached || !parent->native || !parent->native->hasExecutable())
                throw std::invalid_argument("Expert source refresh requires one exact sealed original native parent");
            original = parent;
        }
        if (!original || std::any_of(staged.begin(),staged.end(),[&](const auto& value){return value.parent==original;}))
            throw std::invalid_argument("Expert source refresh omitted or duplicated its original attachment");
        auto variants = retainedSourceVariantExtension(original->variants,binding.definitions);
        for (const auto& next : binding.definitions) {
            for (const auto& weak : owner->parents) if (const auto other = weak.lock()) {
                if (other != original && std::find(other->variants.begin(),other->variants.end(),next)!=other->variants.end())
                    throw std::invalid_argument("One original source variant cannot select two expert row parents");
            }
            for (const auto& prior : staged) if (prior.parent->original.device()==binding.receiver &&
                std::find(prior.variants.begin(),prior.variants.end(),next)!=prior.variants.end())
                throw std::invalid_argument("Source refresh duplicates an original variant across expert parents");
        }
        staged.push_back({std::move(original),std::move(variants)});
    }
    // Every currently attached expert parent must participate. Validate and
    // allocate the complete candidate before changing even its first alias.
    std::size_t expected=0;
    for (const auto& owner : state_->owners) if (owner.disposition==State::ExecutionDisposition::EmbeddedExpert)
        for (const auto& weak : owner.parents) if (weak.lock()) ++expected;
    if (staged.size()!=expected)
        throw std::invalid_argument("Expert source refresh is a partial original follower family");
    for (auto& value : staged) value.parent->variants.swap(value.variants);
}
void PooledImmutableRetainedParticipants::requireInvocation(const RetainedGraphParticipantPreparationStream& definition)const{
    requireCurrent();definition.requireCurrentDefinition();
    if(definition.definition().domain!=RetainedGraphParticipantDefinitionDomain{state_->source.device(),state_->transport.get()})
        throw std::logic_error("Source invocation substituted its original declared cohort/physical definitions");
    for (const auto& owner : state_->owners) {
        if (owner.disposition != State::ExecutionDisposition::EmbeddedExpert) continue;
        bool matched = false;
        for (const auto& weak : owner.parents) if (const auto parent = weak.lock()) {
            const bool same = parent->definition.get() == &definition ||
                std::any_of(parent->variants.begin(),parent->variants.end(),[&](const auto& variant) { return variant.get() == &definition; });
            if (!same) continue;
            if (matched || !parent->attached || !parent->native || !parent->native->hasExecutable())
                throw std::logic_error("Expert owner source invocation lacks one exact sealed original follower parent");
            matched = true;
        }
        if (!matched) throw std::logic_error("Expert owner source invocation has no original finite family attachment");
    }
    // The matched capture-end precedes each peer's attach/instantiate. Original
    // Rank dispatch owns those independent submissions; checking another CPU
    // thread's current readiness here would race or demand a second barrier.
    // Each receiver validates its SAME token and own instantiated parent at its
    // existing local submission edge. Native ACKs and all-parent terminal proof
    // remain the sole evidence that physical readers actually retired.
}
RetainedGraphParticipantSubmission PooledImmutableRetainedParticipants::enqueueOriginalParticipants(const RetainedGraphParticipantPreparationStream& definition)const{
    requireInvocation(definition);state_->boundary->requireArmedForLaunch();
    // A failure after the first native enqueue cannot unwind any shared frame.
    // The exact original CPU ticket remains the sole transaction selection.
    std::unique_lock lock(state_->submissionMutex);
    if(state_->enqueueGroup)state_->enqueueGroup->dispatch();
    return RetainedGraphParticipantSubmission(shared_from_this(),std::move(lock));
}
void PooledImmutableRetainedParticipants::completeOriginalEnqueue()const noexcept{
    if(state_->enqueueGroup)state_->enqueueGroup->completeOriginalEnqueue();
    // These count successful actual source/remote SDK submissions, not inferred
    // controller progress. Compact live byte counts remain TransferEngine-owned.
    PerfStatsCollector::addCounter("ngram_lookup","retained_source_native_invocations",1.0,
        "hosted_transaction",state_->source.device().toString(),{{"policy","retained_hosted"}});
    PerfStatsCollector::addCounter("ngram_lookup","retained_remote_native_enqueues",static_cast<double>(std::count_if(state_->owners.begin(),state_->owners.end(),[](const auto& physical){return physical.disposition==State::ExecutionDisposition::Standalone;})),
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
        for(const auto& owner:state_->owners){
            if(owner.disposition==State::ExecutionDisposition::Standalone){complete=owner.executable->observeTerminal() && complete;continue;}
            for(const auto& weak:owner.parents)if(const auto parent=weak.lock()){
                const auto observed=[&](void* stream){return owner.original.worker()->queryStreamExecutionState(stream,"original embedded storage request terminal")==GPUStreamExecutionState::Complete;};
                complete=observed(parent->captureStream) && complete;
                if(parent->enclosing)complete=observed(parent->enclosing->stream()) && complete;
            }
        }
        if(complete)return;
        if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("Recurring owner native completion exceeded the original request terminal deadline");
        std::this_thread::yield();}
}
bool PooledImmutableRetainedParticipants::belongsToOriginalCohort(
    const std::shared_ptr<const Qwen4ExpReplicatedPLECohort>& original) const noexcept
{
    // Cancellation revokes native admission, not the identity of its retirement
    // owner. Compare the existing immutable parent/control block without calling
    // context(), requireCurrent(), or any native API after an absorbing abort.
    const auto& retained = state_->originalCohort;
    return original && retained.get() == original.get() &&
        !retained.owner_before(original) && !original.owner_before(retained);
}
void PooledImmutableRetainedParticipants::retireOriginalRuntimeBeforeArenaRelease()
{
    if (!state_->ownsBoundaryClaim) return;
    if (isGraphCaptureActive() || state_->boundary->hasActiveInterval() || state_->source.worker()->ownsCurrentThread())
        throw std::logic_error("Original pooled runtime retirement requires closed source/native work off the original worker");
    std::unique_lock lock(state_->submissionMutex);
    // Cache-owned attachment leases prove no original native parent still
    // borrows these definitions. A context may retain a FAILED cold handoff
    // token; it grants no executable/arena borrow and is revoked below.
    for (const auto& owner : state_->owners)
        for (const auto& weak : owner.parents)
            if (!weak.expired()) throw std::logic_error("Original pooled parent/cache borrower is still live");
    if (state_->transport) {
        awaitNativeTerminal(std::chrono::steady_clock::now() + std::chrono::seconds(30));
        if (state_->submissionStream) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (state_->source.worker()->queryStreamExecutionState(state_->submissionStream->stream(),
                    "original pooled source runtime retirement") != GPUStreamExecutionState::Complete) {
                if (std::chrono::steady_clock::now() >= deadline)
                    throw std::runtime_error("Original pooled source parent retirement remains unproven");
                std::this_thread::yield();
            }
        }
        // Independently recorded owners may borrow coordinator arena roots.
        // Their captures/stages MUST die here, while every child arena lives.
        state_->enqueueGroup.reset();
        state_->owners.clear();
        for (const auto& remote : state_->transport->remotes()) remote.pin->retireOriginalInvocation();
        state_->transport->retireOriginalRuntime();
    }
    const auto current = state_->boundary->completeCohort_.load(std::memory_order_acquire).lock();
    if (current && current.get() != this)
        throw std::logic_error("Original pooled runtime retirement replaced its source authority");
    // All DGO graph borrowers crossed their own exact producer/topology
    // frontier first. Only now may the retained source forget runner addresses.
    state_->boundary->retireParticipantGenerationControl(*this);
    state_->boundary->completeCohort_.store({}, std::memory_order_release);
    state_->submissionStream.reset();
    state_->preparationStreams.clear();
    state_->transport.reset();
    state_->programs.clear();
    state_->ownsBoundaryClaim = false;
    state_->boundary->cohortOwnerClaim_.clear(std::memory_order_release);
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
