/**
 * @file PooledImmutableRetainedParticipants.cpp
 * @brief Original recurring hosted owner captures, source seals and whole-request retirement.
 *
 * No source model graph is cloned or substituted. Each actual original source
 * parent invocation submits each addressed owner program once. Protocol packet
 * retirement and native executable completion remain distinct authorities.
 * A mixed-GPU I/O auxiliary body borrows the same original execution descriptor;
 * no CPU expert tier or alternative source policy is created to admit it.
 */
#include "PooledImmutableRetainedParticipants.h"
#include "PooledImmutableReplicatedReaderGraph.h"
#include "PooledImmutableParticipantExecutionPolicy.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotPreparedStorage.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableLocalHotPhaseStage.h"
#include "loaders/ModelLoader.h"
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
        std::shared_ptr<const RetainedGraphParticipantPreparationStream> auxiliary; ///< Actual local body, distinct from its whole-source lineage.
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
        std::optional<PooledStorageOwnerProgram> program; ///< Present only for a separately recorded storage definition.
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
    /** @brief Existing local-root lifetime metadata; no cache occupancy or second allocator ledger. */
    struct LocalParticipant {
        PreparedGPUContextUsePermit original;
        std::uint64_t rootIdentity;
        std::optional<RetainedGraphParticipantParentStream> enclosingParent;
        std::vector<std::weak_ptr<AttachedParent>> parents;
    };
    std::optional<PooledImmutableLocalHotTransportGeometry> localHotGeometry;
    std::vector<std::shared_ptr<PooledImmutableLocalHotPreparedStorage>> localHotParticipants; ///< Sole immutable local-root roster, in declared consumer order.
    std::vector<LocalParticipant> locals; ///< Same-roster original worker and parent lifetimes only.
    std::unique_ptr<RetainedGraphParticipantEnqueueGroup> enqueueGroup;
    std::vector<PooledStorageOwnerProgram> programs; ///< Original physical order; never recomputed from live counts.
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
    StorageAttachment(std::optional<RetainedParticipantGraphExecutable::RecordedStorageDefinition> body,
        std::shared_ptr<State::AttachedParent> parent,
        std::optional<RetainedGraphParticipantAuxiliaryBodySeal> auxiliary={})
        :body_(std::move(body)),parent_(std::move(parent)),auxiliary_(std::move(auxiliary)){}
    bool attachTo(IGPUGraphCapture& parent,IGraphCaptureAuxiliaryBranch* progress)override{
        if(parent_->attempted || parent_->native!=&parent || parent.executionStream()!=parent_->captureStream)
            throw std::logic_error("Embedded storage attachment changed its original cold native parent");
        parent_->attempted=true;
        bool attached=false;
        const auto assemble=[&] {
            // Inline storage is already in the recorded main DAG. The original
            // CPU-service progress branch still belongs to this SAME parent;
            // skipping its ordinary attachment would strand hosted execution.
            if(auxiliary_) {
                auxiliary_->preparation()->requireCurrentDefinition();
                // Start only after THIS replay's main Probe. The body remains
                // a separate parent terminal; joining it to the producer unit
                // would deadlock when it waits for LocalRead in a later unit.
                if(!parent.appendFiniteBodyAfterCapturedFragment(
                    *auxiliary_->identity().producer_fragment,*auxiliary_->recordedBody()))return;
            }
            // Existing progress Open/Close observes the complete parent after
            // the finite body has been added, including its retirement leaf.
            attached=body_ ? body_->attachTo(parent,progress) : (!progress || progress->attach(parent));
        };
        if(parent_->original.worker()->ownsCurrentThread())assemble();else parent_->original.worker()->submitAndWait(assemble);
        parent_->attached=attached;return attached;
    }
    void requireCurrent()const override{
        if (body_) body_->requireCurrent();
        else parent_->definition->requireCurrentDefinition();
        if(auxiliary_) {
            auxiliary_->preparation()->requireCurrentDefinition();
            if(auxiliary_->identity().enclosing_parent!=parent_->native ||
               auxiliary_->identity().enclosing_recording_stream!=parent_->captureStream ||
               !auxiliary_->identity().producer_fragment ||
               !auxiliary_->recordedBody() || auxiliary_->recordedBody()->hasExecutable())
                throw std::logic_error("Local auxiliary lost its exact graph-only body and enclosing native parent");
        }
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
    std::optional<RetainedGraphParticipantAuxiliaryBodySeal> auxiliary_; ///< Strong original body and preparation borrows until parent destruction.
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
        for(const auto& local:owner_->state_->locals)fold(local.rootIdentity);
        for(const auto& owner:owner_->state_->owners){fold(owner.program ? pooledStorageOwnerIdentity(*owner.program) : owner.coordinate);fold(owner.topology);
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
        if(preparation.definition().scope!=RetainedGraphParticipantDefinitionScope::WholeSource) {
            requireAuxiliaryDeclaration(preparation,true);return;
        }
        preparation.requireLive(originalSource());
        const auto& definition=preparation.definition();
        if(definition.plan.get()!=this || definition.domain!=definitionDomain())
            throw std::invalid_argument("Prepared source changed its original source/transport plan");
        requireDeclaration(definition.graph,definition.arena,originalSource().device(),preparation.stream(),
            definition.topology_generation,definition.variant);
    }
    void requireCurrentPreparedSourceDefinition(const RetainedGraphParticipantPreparationStream& preparation)const override{
        owner_->requireCurrent();
        if(preparation.definition().scope!=RetainedGraphParticipantDefinitionScope::WholeSource) {
            requireAuxiliaryDeclaration(preparation,false);return;
        }
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
    /**
     * @brief Authenticate the original source envelope independently of CPU expert presence.
     * @param graph Same source graph later sealed with the actual native parent.
     * @throws std::exception For an unsupported envelope or replaced original owner borrow.
     */
    void requireSourceEnvelope(const ComputeGraph& graph) const
    {
        if (ownsHeterogeneousTicketBoundary(graph.nativeCaptureEnvelope()))
            return;
        graph.requireRetainedParticipantNativeAuxiliaryParent();
        const auto& original = graph.retainedParticipantAuxiliaryRange()->original_execution;
        const auto& expected = owner_->state_->executionPolicy;
        if (!original || original != expected || original.owner_before(expected) || expected.owner_before(original))
            throw std::invalid_argument("Native I/O source substituted its original execution descriptor");
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
           graph->topologyGeneration()!=topology || graph->retainedParticipantSourcePlan().get()!=this)
            throw std::invalid_argument("Source definition lost its original graph/arena/count transaction");
        requireSourceEnvelope(*graph);
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
    /**
     * @brief Retain the actual participant-local auxiliary beside any original physical owner body.
     * @param sealed Executor-issued graph-only recording with its exact participant/parent identity.
     * @param progress Existing original per-device Open/Close authority.
     * @return One attachment owning the complete sibling set through native retirement.
     * @throws std::exception Before native mutation for missing/foreign/stale or duplicated bodies.
     */
    std::unique_ptr<IRetainedGraphParticipantStorageAttachment> prepareOriginalAuxiliaryAttachment(
        const RetainedGraphParticipantAuxiliaryBodySeal& sealed,IGraphCaptureAuxiliaryBranch* progress)const override {
        const auto& id=sealed.identity();const auto& prep=sealed.preparation();
        if(!prep || prep->definition().plan.get()!=this || !sealed.recordedBody())
            throw std::invalid_argument("Local auxiliary lost its actual original preparation/body");
        prep->requireDefinition();
        const auto& declaration=prep->definition();
        const auto& source=declaration.original_source_definition;
        const auto local=std::find_if(owner_->state_->locals.begin(),owner_->state_->locals.end(),
            [&](const auto& value){return value.original.device()==id.participant;});
        if(local==owner_->state_->locals.end() || !id.graph || !id.arena ||
           id.graph!=declaration.graph || id.arena!=declaration.arena || id.participant!=declaration.participant ||
           id.topology_generation!=declaration.topology_generation || id.capture_variant_signature!=declaration.variant ||
           !id.capture_generation || id.range_identity!=declaration.range_identity ||
           !id.enclosing_parent || id.enclosing_parent->hasExecutable() || !id.enclosing_parent->nodeCount() ||
           !id.producer_fragment || !id.producer_fragment->nodeCount() || id.producer_fragment->hasExecutable() ||
           id.producer_fragment->executionStream()!=id.enclosing_recording_stream ||
           id.producer_fragment==id.enclosing_parent || id.producer_fragment==sealed.recordedBody().get() ||
           id.enclosing_parent->executionStream()!=id.enclosing_recording_stream || prep->stream()!=id.enclosing_recording_stream ||
           !sealed.recordedBody()->nodeCount() || sealed.recordedBody()->hasExecutable() ||
           sealed.recordedBody()->executionStream()!=id.enclosing_recording_stream ||
           (progress && (progress->device()!=id.participant || progress->originalSourceDefinition()!=source)))
            throw std::invalid_argument("Local auxiliary changed its actual native parent, recording or complete source lineage");
        sealed.originalParticipant().requireWorker(local->original.worker());
        prep->requireLive(local->original);
        if(!local->original.worker()->ownsCurrentThread())
            throw std::logic_error("Local auxiliary attachment requires its actual recording worker");
        for(const auto& weak:local->parents)if(const auto prior=weak.lock())
            if(prior->definition==source || prior->auxiliary==prep || prior->native==id.enclosing_parent)
                throw std::logic_error("One local consumer already owns this exact source/parent auxiliary body");
        auto physical=std::find_if(owner_->state_->owners.begin(),owner_->state_->owners.end(),
            [&](const auto& value){return value.original.device()==id.participant;});
        std::optional<RetainedParticipantGraphExecutable::RecordedStorageDefinition> body;
        if(physical!=owner_->state_->owners.end()) {
            if(physical->disposition==State::ExecutionDisposition::Standalone ||
               physical->disposition==State::ExecutionDisposition::EmbeddedExpert)
                throw std::invalid_argument("Local consumer cannot acquire an independent storage/expert parent role");
            for(const auto& weak:physical->parents)if(const auto prior=weak.lock())
                if(prior->definition==source || prior->native==id.enclosing_parent)
                    throw std::logic_error("Original physical owner already attached to this source/parent");
            if(physical->disposition==State::ExecutionDisposition::EmbeddedParent)
                body.emplace(physical->executable->retainRecordedStorageDefinition(physical->original,source));
        }
        auto lifetime=std::make_shared<State::AttachedParent>(State::AttachedParent{
            .definition=source,.auxiliary=prep,.recipient=id.graph,.native=id.enclosing_parent,
            .captureStream=id.enclosing_recording_stream,.original=local->original,.enclosing=local->enclosingParent});
        auto result=std::make_unique<StorageAttachment>(std::move(body),lifetime,sealed);
        local->parents.push_back(lifetime);
        if(physical!=owner_->state_->owners.end())physical->parents.push_back(lifetime);
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
    /**
     * @brief Validate the real local scope without treating a reader as a whole source.
     * @param preparation Exact executor-issued local-body token.
     * @param inspect Cold validation additionally authenticates concrete native stages and dependencies.
     * @throws std::exception For a replaced source, participant, arena, root, role or capture topology.
     */
    void requireAuxiliaryDeclaration(const RetainedGraphParticipantPreparationStream& preparation,bool inspect)const {
        const auto& d=preparation.definition();const auto& source=d.original_source_definition;
        const auto local=std::find_if(owner_->state_->locals.begin(),owner_->state_->locals.end(),
            [&](const auto& value){return value.original.device()==d.participant;});
        const bool isSource=d.scope==RetainedGraphParticipantDefinitionScope::AuxiliarySource;
        if(!source || d.plan.get()!=this || d.domain!=definitionDomain() || !d.graph || !d.arena ||
           local==owner_->state_->locals.end() || d.graph->topologyGeneration()!=d.topology_generation ||
           d.scope!= (isSource?RetainedGraphParticipantDefinitionScope::AuxiliarySource:RetainedGraphParticipantDefinitionScope::AuxiliaryReader) ||
           isSource!=(d.participant==originalSource().device()) ||
           !d.graph->retainedParticipantDefinitionDomain() || *d.graph->retainedParticipantDefinitionDomain()!=d.domain ||
           (isSource ? d.graph->retainedParticipantSourcePlan().get()!=this || d.arena!=arena_ : bool(d.graph->retainedParticipantSourcePlan())) ||
           !d.graph->retainedParticipantSourceVariants().empty())
            throw std::invalid_argument("Auxiliary definition changed its actual local participant, source or graph role");
        source->requireCurrentDefinition();preparation.requireLive(local->original);
        const auto& range=d.graph->retainedParticipantAuxiliaryRange();
        if(!range || range->participant!=d.participant || range->identity!=d.range_identity ||
           (range->role==RetainedGraphParticipantAuxiliaryRole::Source)!=isSource)
            throw std::invalid_argument("Auxiliary definition lost its exact declared native range");
        if (requiresSingleNativeExecutable(d.graph->nativeCaptureEnvelope()))
        {
            // Cold authentication proves the full native I/O declaration once.
            // Current borrows retain its immutable identity/topology without
            // rescanning the model DAG on every invocation.
            if (inspect)
                d.graph->requireRetainedParticipantNativeAuxiliaryParent();
            const auto& declared = range->original_execution;
            const auto& expected = owner_->state_->executionPolicy;
            if (!declared || declared != expected || declared.owner_before(expected) || expected.owner_before(declared))
                throw std::invalid_argument("Native auxiliary source changed its original execution descriptor");
        }
        if(inspect) {
            d.graph->requireRetainedParticipantAuxiliaryRange();
            const auto index=static_cast<std::size_t>(local-owner_->state_->locals.begin());
            requirePooledLocalHotAuxiliary(*d.graph,*range,owner_->state_->transport,owner_->state_->localHotParticipants[index]);
        }
    }
    std::shared_ptr<PooledImmutableRetainedParticipants> owner_;
    BufferArena* arena_;
    PooledImmutableSourceRetirementCompletion completion_;
    std::uint64_t identity_;
};
PooledImmutableRetainedParticipants::PooledImmutableRetainedParticipants(std::unique_ptr<State> state):state_(std::move(state)){}
std::shared_ptr<PooledImmutableRetainedParticipants> PooledImmutableRetainedParticipants::create(
    std::shared_ptr<ImmutableLookupExecutionBoundary> boundary,const GraphConfig& policy,std::shared_ptr<const ClusterInventory> originalInventory,
    std::optional<PooledImmutableReplicatedReaderGeometry> readers,
    std::optional<PooledImmutableLocalHotTransportGeometry> localHot,
    std::span<const std::shared_ptr<PooledImmutableLocalHotPreparedStorage>> localHotParticipants){
    PooledImmutableParticipantExecutionPolicy::requireConfiguration(policy.mtp);
    if(isGraphCaptureActive() || !boundary || !boundary->isPooled() || policy.default_device!=boundary->device() ||
       policy.n_layers<=0 || policy.pp_layer_offset!=0 ||
       (policy.total_n_layers>0 && policy.n_layers!=policy.total_n_layers) || !policy.pp_contexts.empty() ||
       !policy.moe.expert_overlay_execution_plan)
        throw std::invalid_argument("Recurring declared owners require original hosted ExpertOverlay source policy");
    const auto& execution=*policy.moe.expert_overlay_execution_plan;
    PooledImmutableParticipantExecutionPolicy::requireHostedSourceExecution(execution);
    if (!localHot && std::none_of(execution.domains.begin(), execution.domains.end(), [](const auto& domain) {
            return std::any_of(domain.participants.begin(), domain.participants.end(), [](const auto& participant) {
                return participant.local_device.is_cpu();
            });
        }))
        throw std::invalid_argument("Native mixed-GPU pooled source requires its actual combined auxiliary geometry");
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
    if(bool(localHot)!=!localHotParticipants.empty())
        throw std::invalid_argument("Combined retained participants require their complete original local-root roster");
    if(localHot) {
        (void)PooledImmutableParticipantExecutionPolicy::requiresRetainedParticipants(storage->geometry(),&*localHot);
        if(localHot->consumers().size()!=localHotParticipants.size() ||
           localHot->consumers().size()!=dense->participants.size() || bool(readers)!=bool(localHot->originalReaders()))
            throw std::invalid_argument("Combined retained participant population changed the actual continuation roster");
        for(std::size_t index=0;index<localHotParticipants.size();++index) {
            const auto& local=localHotParticipants[index];const auto endpoint=localHot->consumers()[index];
            if(!local || local->geometry().allocator()!=endpoint ||
               local->geometry().extent()!=localHot->sourceExtent() ||
               local->geometry().maximumOriginalIds()!=localHot->maximumPoolUniqueRows() ||
               (readers && readers->consumers()[index]!=endpoint))
                throw std::invalid_argument("Combined retained local roots changed exact table, query or consumer order");
            local->requireFor(local->geometry(),storage,witness->coordinator().memory());
        }
    }
    for(const auto& domain:execution.domains)for(const auto& participant:domain.participants){
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
    if(witness->coordinator().worker()->ownsCurrentThread())throw std::logic_error("Recurring owner preparation must run off its original source worker");
    std::unique_ptr<RetainedGraphParticipantEnqueueGroup> enqueueGroup;
    std::vector<PooledStorageOwnerProgram> programs;
    for(const auto& owner:witness->owners()){
        if(owner.worker()->ownsCurrentThread())throw std::logic_error("Recurring owner preparation must run off every original participant worker");
        if(owner.device()!=boundary->device()){
            const auto coordinate=static_cast<std::uint32_t>(&owner-witness->owners().data());
            const bool inlineNative = readers && policy.ngram_replicated_ple_cohort &&
                owner.device().type == boundary->device().type &&
                std::any_of(readers->consumers().begin(), readers->consumers().end(),
                    [&](const auto& reader) { return reader.device == owner.device(); });
            if (inlineNative) continue;
            auto program=PooledImmutableDirectedOwnerProgram::fromOriginalStoragePlans(storage->originalOwners()[coordinate]->plan(),storage->coordinator()->plan());
            // Validate each exact mapped or native-General resource declaration
            // before creating the first directed channel. No caller supplies a bound.
            std::visit([&](const auto& contract){
                using Contract=std::decay_t<decltype(contract)>;
                if constexpr(std::is_same_v<Contract,PooledRemoteGraphProgram>)
                    (void)RetainedParticipantGraphExecutable::hostBOM(contract,witness->source()->originalResource(DeviceId::cpu()),witness->source()->originalResource(owner.device()));
                else (void)RetainedParticipantGraphExecutable::nativeStorageBOM(contract,witness->source()->originalResource(DeviceId::cpu()),witness->source()->originalResource(owner.device()));
            },program);
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
    result->state_->localHotGeometry=localHot;
    result->state_->localHotParticipants.assign(localHotParticipants.begin(),localHotParticipants.end());
    for(const auto& local:localHotParticipants) {
        auto original=GPUDeviceContextPool::instance().retainExistingPreparedContext(
            local->geometry().allocator().device,witness->coordinator().memory());
        if(original.worker()->ownsCurrentThread())
            throw std::logic_error("Combined local participant preparation must remain off its original worker");
        result->state_->locals.push_back(State::LocalParticipant{.original=std::move(original),.rootIdentity=local->captureIdentity()});
    }
    result->state_->transport=PooledImmutableDirectedTransport::create(boundary,std::move(originalInventory), std::move(readers),
        policy.ngram_replicated_ple_cohort,localHot);
    if(!localHot && result->state_->transport->remotes().empty() && result->state_->transport->readers().empty())
        throw std::invalid_argument("Retained participant binding has neither remote storage owners nor compute readers");
    result->prepareOriginalOwners();
    boundary->bindRetainedParticipants(result);
    return result;
}
void PooledImmutableParticipantExecutionPolicy::requireHostedSourceExecution(
    const MoEExpertOverlayExecutionPlan& original)
{
    if (original.world_size != 1 || !original.ownsContinuationGraph() ||
        original.generationExecutionDomain() != DeviceGenerationExecutionDomain::HostedBoundary)
        throw std::invalid_argument("Recurring remote owners require one actual rank-local heterogeneous hosted ticket parent");
    const bool hasCPU = std::any_of(original.domains.begin(), original.domains.end(), [](const auto& domain) {
        return std::any_of(domain.participants.begin(), domain.participants.end(), [](const auto& participant) {
            return participant.local_device.is_cpu();
        });
    });
    if (!hasCPU)
        requireRankLocalMixedGPUAuxiliaryExecution(original);
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

bool PooledImmutableParticipantExecutionPolicy::requiresRetainedParticipants(
    const PooledImmutableSourceGeometry& source,const PooledImmutableLocalHotTransportGeometry* localHot)
{
    if(!localHot)return requiresRetainedParticipants(source);
    const auto& p=source.parameters();
    if(localHot->coordinator()!=source.allocator() || localHot->consumers().size()!=p.consumers ||
       localHot->sourceExtent().rows()!=p.coordinator.policy.sourceRows() ||
       ggufToTensorType(localHot->sourceExtent().format())!=p.format || localHot->sourceExtent().rowBytes()!=p.owner.row_bytes ||
       localHot->maximumPoolOriginalRows()!=p.coordinator.policy.maximumOriginal() ||
       localHot->maximumPoolUniqueRows()!=p.coordinator.policy.maximumUnique())
        throw std::invalid_argument("Combined source declaration changed the original pool/table/query envelope");
    return true;
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
        const auto declared=std::find_if(state_->programs.begin(),state_->programs.end(),[&](const auto& program){return pooledStorageOwnerGeometry(program).coordinate==edge.coordinate;});
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
        owner.executable=state_->boundary->pooledStorage()->acquireOriginalRemoteDefinition(*owner.program,originalStream);
        owner.submissionOrdinal=owner.executable->lastSubmission_;
        try {
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
            const auto nodes=std::visit([&](const auto& contract){
                using Contract=std::decay_t<decltype(contract)>;
                if constexpr(std::is_same_v<Contract,PooledRemoteGraphProgram>)
                    return appendPooledImmutableDirectedOwnerGraph(*owner.graph,*owner.arena,edge,owner.code,contract,"immutable_owner");
                else {
                    const auto actual=PooledImmutableDirectedOwnerProgram::fromOriginalStoragePlans(edge.pin->ownerStorage()->plan(),edge.pin->coordinatorStorage()->plan());
                    if(actual!=*owner.program)throw std::logic_error("Native retained owner substituted its original source semantic declaration");
                    return appendPooledImmutableNativeOwnerGraph(*owner.graph,*owner.arena,edge,owner.code,"immutable_owner");
                }
            },*owner.program);
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
        } catch(...) {
            state_->boundary->pooledStorage()->abandonOriginalRemoteDefinition(*owner.program,owner.executable,std::current_exception());
            throw;
        }
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
const std::shared_ptr<const MoEExpertOverlayExecutionPlan>&
PooledImmutableRetainedParticipants::originalExecutionPolicy() const noexcept
{
    return state_->executionPolicy;
}
const std::optional<PooledImmutableLocalHotTransportGeometry>& PooledImmutableRetainedParticipants::localHotGeometry()const noexcept{return state_->localHotGeometry;}
std::span<const std::shared_ptr<PooledImmutableLocalHotPreparedStorage>> PooledImmutableRetainedParticipants::localHotParticipants()const noexcept{return state_->localHotParticipants;}

std::shared_ptr<const IPooledImmutableNativeKernels>
PooledImmutableRetainedParticipants::localHotInlineOwnerCode(DeviceId device) const
{
    requireCurrent();
    if (!state_->localHotGeometry || std::none_of(state_->locals.begin(), state_->locals.end(),
        [&](const auto& local) { return local.original.device() == device; }))
        throw std::invalid_argument("Combined inline-owner code requires an original local-cache consumer");
    const auto owner = std::find_if(state_->owners.begin(), state_->owners.end(), [&](const auto& candidate) {
        return candidate.original.device() == device && candidate.disposition == State::ExecutionDisposition::InlineReader;
    });
    return owner == state_->owners.end() ? nullptr : owner->code;
}

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
RetainedGraphParticipantDefinitionDomain PooledImmutableRetainedParticipants::definitionDomain() const noexcept
{
    return {state_->source.device(), state_->transport.get()};
}

std::uint64_t PooledImmutableRetainedParticipants::captureIdentity() const noexcept
{
    auto value = state_->rootIdentity;
    const auto mix = [&](std::uint64_t word) { value = (value ^ word) * 1099511628211ULL; };
    mix(reinterpret_cast<std::uintptr_t>(state_->transport.get()));
    mix(reinterpret_cast<std::uintptr_t>(state_->executionPolicy.get()));
    for (const auto& local : state_->locals) mix(local.rootIdentity);
    return value ? value : 1;
}

void PooledImmutableRetainedParticipants::bindLocalConsumption(ComputeGraph& graph, BufferArena& arena,
    const std::shared_ptr<PooledImmutableLocalHotPreparedStorage>& local, RetainedGraphParticipantAuxiliaryRange range)
{
    requireCurrent();
    const auto found = std::find(state_->localHotParticipants.begin(), state_->localHotParticipants.end(), local);
    if (!local || found == state_->localHotParticipants.end() || range.participant != local->geometry().allocator().device ||
        !localConsumptionOnly(range.lifecycle) || graph.retainedParticipantSourcePlan() ||
        !graph.retainedParticipantDefinitionDomain() || *graph.retainedParticipantDefinitionDomain() != definitionDomain())
        throw std::invalid_argument("Local-only declaration lost its actual prepared local/source owner");
    // Alias the same canonical owner: no declaration allocation, source program
    // or independently counted captured-branch population is introduced.
    range.local_consumption = std::static_pointer_cast<const IRetainedGraphParticipantLocalConsumptionPlan>(shared_from_this());
    range.local_arena = &arena;
    graph.setRetainedParticipantAuxiliaryRange(std::move(range));
    requirePooledLocalHotLocalConsumption(graph, *graph.retainedParticipantAuxiliaryRange(), local);
}

void PooledImmutableRetainedParticipants::requireLocalDefinition(
    const RetainedGraphParticipantPreparationStream& preparation, bool inspect) const
{
    requireCurrent();
    const auto& d = preparation.definition();
    const auto original = shared_from_this();
    const auto local = std::find_if(state_->locals.begin(), state_->locals.end(),
        [&](const auto& value) { return value.original.device() == d.participant; });
    if (d.scope != RetainedGraphParticipantDefinitionScope::LocalConsumption || d.plan || d.original_source_definition ||
        d.local_consumption.get() != this || d.local_consumption.owner_before(original) || original.owner_before(d.local_consumption) ||
        !d.graph || !d.arena || !d.topology_generation || !d.variant || d.graph->topologyGeneration() != d.topology_generation ||
        d.domain != definitionDomain() || local == state_->locals.end() || d.graph->retainedParticipantSourcePlan() ||
        !d.graph->retainedParticipantDefinitionDomain() || *d.graph->retainedParticipantDefinitionDomain() != d.domain ||
        !d.graph->retainedParticipantSourceVariants().empty())
        throw std::invalid_argument("Local consumption cannot claim a source program or substitute its actual owner/cache");
    preparation.requireLive(local->original);
    const auto& range = d.graph->retainedParticipantAuxiliaryRange();
    if (!range || range->local_arena != d.arena || !localConsumptionOnly(range->lifecycle) ||
        range->participant != d.participant || range->identity != d.range_identity ||
        range->local_consumption != d.local_consumption || range->local_consumption.owner_before(d.local_consumption) ||
        d.local_consumption.owner_before(range->local_consumption) || range->original_execution != state_->executionPolicy ||
        range->original_execution.owner_before(state_->executionPolicy) || state_->executionPolicy.owner_before(range->original_execution))
        throw std::invalid_argument("Local consumption changed its exact immutable local range/execution identity");
    if (inspect)
    {
        d.graph->requireRetainedParticipantAuxiliaryRange();
        if (requiresSingleNativeExecutable(d.graph->nativeCaptureEnvelope()))
            d.graph->requireRetainedParticipantNativeAuxiliaryParent();
        const auto index = std::size_t(local - state_->locals.begin());
        requirePooledLocalHotLocalConsumption(*d.graph, *range, state_->localHotParticipants[index]);
    }
}

std::unique_ptr<IRetainedGraphParticipantStorageAttachment> PooledImmutableRetainedParticipants::prepareLocalAttachment(
    const RetainedGraphParticipantAuxiliaryBodySeal& sealed, IGraphCaptureAuxiliaryBranch* progress) const
{
    const auto& prep = sealed.preparation();
    const auto& id = sealed.identity();
    if (!prep || !sealed.recordedBody())
        throw std::invalid_argument("Local consumption lost its actual finite recording");
    prep->requireDefinition();
    const auto& d = prep->definition();
    const auto local = std::find_if(state_->locals.begin(), state_->locals.end(),
        [&](const auto& value) { return value.original.device() == id.participant; });
    if (local == state_->locals.end() || id.graph != d.graph || id.arena != d.arena || id.participant != d.participant ||
        id.topology_generation != d.topology_generation || id.capture_variant_signature != d.variant ||
        !id.capture_generation || id.range_identity != d.range_identity || !id.enclosing_parent ||
        id.enclosing_parent->hasExecutable() || !id.enclosing_parent->nodeCount() || !id.producer_fragment ||
        !id.producer_fragment->nodeCount() || id.producer_fragment->hasExecutable() ||
        id.producer_fragment->executionStream() != id.enclosing_recording_stream || id.producer_fragment == id.enclosing_parent ||
        id.producer_fragment == sealed.recordedBody().get() || id.enclosing_parent->executionStream() != id.enclosing_recording_stream ||
        prep->stream() != id.enclosing_recording_stream || !sealed.recordedBody()->nodeCount() || sealed.recordedBody()->hasExecutable() ||
        sealed.recordedBody()->executionStream() != id.enclosing_recording_stream ||
        (progress && (progress->device() != id.participant || progress->originalSourceDefinition())))
        throw std::invalid_argument("Local consumption substituted native parent/body or introduced source progress");
    sealed.originalParticipant().requireWorker(local->original.worker());
    prep->requireLive(local->original);
    if (!local->original.worker()->ownsCurrentThread())
        throw std::logic_error("Local attachment requires original recording worker");
    for (const auto& weak : local->parents)
        if (const auto prior = weak.lock())
            if (prior->native == id.enclosing_parent || prior->auxiliary == prep)
                throw std::logic_error("Local body already belongs to this exact original parent");
    // Retain original local/native identity only. No physical owner body or
    // source submission, servicing or completion is imported by this branch.
    auto lifetime = std::make_shared<State::AttachedParent>(State::AttachedParent{
        .definition = prep, .auxiliary = prep, .recipient = id.graph, .native = id.enclosing_parent,
        .captureStream = id.enclosing_recording_stream, .original = local->original, .enclosing = local->enclosingParent});
    auto result = std::make_unique<StorageAttachment>(std::nullopt, lifetime, sealed);
    local->parents.push_back(lifetime);
    return result;
}

void PooledImmutableRetainedParticipants::requireCurrent()const{
    if (!state_->ownsBoundaryClaim) throw std::logic_error("Original pooled execution participants retired with their runner");
    if(state_->rootIdentity!=state_->boundary->pooledStorage()->captureIdentity())throw std::logic_error("Recurring owner original roots changed");
    state_->transport->requireCurrent();
    state_->source.requireWorker(GPUDeviceContextPool::instance().retainExistingContext(state_->source.device()));
    if(state_->locals.size()!=state_->localHotParticipants.size())
        throw std::logic_error("Combined participant lost its immutable original local roster");
    for(std::size_t index=0;index<state_->locals.size();++index) {
        const auto& local=state_->localHotParticipants[index];const auto& lifetime=state_->locals[index];
        if(!local || local->captureIdentity()!=lifetime.rootIdentity || local->geometry().allocator().device!=lifetime.original.device())
            throw std::logic_error("Combined participant changed an original local physical root");
        lifetime.original.requireWorker(GPUDeviceContextPool::instance().retainExistingContext(lifetime.original.device()));
    }
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
    return std::any_of(state_->locals.begin(),state_->locals.end(),[&](const auto& local){return local.original.device()==device;}) ||
        std::any_of(state_->owners.begin(),state_->owners.end(),[&](const auto& owner){return owner.original.device()==device && owner.disposition!=State::ExecutionDisposition::Standalone;});
}
PreparedGPUContextUsePermit PooledImmutableRetainedParticipants::originalParticipant(DeviceId device)const{
    if(device==state_->source.device())return state_->source;
    const auto local=std::find_if(state_->locals.begin(),state_->locals.end(),[&](const auto& value){return value.original.device()==device;});
    if(local!=state_->locals.end())return local->original;
    const auto found=std::find_if(state_->owners.begin(),state_->owners.end(),[&](const auto& owner){return owner.original.device()==device && owner.disposition!=State::ExecutionDisposition::Standalone;});
    if(found==state_->owners.end())throw std::invalid_argument("Parent stream requires an original embedded storage participant");
    return found->original;
}
void PooledImmutableRetainedParticipants::bindEmbeddedParentSubmissionStream(const RetainedGraphParticipantParentStream& parent){
    auto& owners=state_->owners;const auto found=std::find_if(owners.begin(),owners.end(),[&](const auto& owner){return owner.original.device()==parent.original().device() && owner.disposition!=State::ExecutionDisposition::Standalone;});
    const auto local=std::find_if(state_->locals.begin(),state_->locals.end(),[&](const auto& value){return value.original.device()==parent.original().device();});
    if(found==owners.end() && local==state_->locals.end())throw std::invalid_argument("Enclosing parent is not an original embedded participant");
    if(local!=state_->locals.end()) {
        parent.original().requireWorker(local->original.worker());
        const auto same=[](const auto& a,const auto& b){return a.get()==b.get() && !a.owner_before(b) && !b.owner_before(a);};
        if(!same(parent.original().memory(),local->original.memory()) || (local->enclosingParent &&
           (parent.stream()!=local->enclosingParent->stream() || !same(parent.owner(),local->enclosingParent->owner()))))
            throw std::invalid_argument("Local parent changed its original PMA or native stream owner");
        local->enclosingParent=parent;
        for(const auto& weak:local->parents)if(const auto actual=weak.lock())actual->enclosing=parent;
        const auto index=static_cast<std::size_t>(local-state_->locals.begin());
        const auto acquire=[&]{state_->localHotParticipants[index]->acquireSetup(parent.stream());};
        if(local->original.worker()->ownsCurrentThread())acquire();else local->original.worker()->submitAndWait(acquire);
        if(found==owners.end())return;
    }
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
            const auto* compact=node?dynamic_cast<const PooledImmutableLocalHotBeginStage*>(node->stage.get()):nullptr;
            if(compact) {
                if(compact->source()!=state_->boundary->pooledStorage())
                    throw std::invalid_argument("Expert source association contains a foreign compact coordinator occurrence");
                requirePooledCompactQueryProducer(*definition->definition().graph,compact->query(),name);
                ++occurrences;
                continue;
            }
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
    const auto local=std::find_if(state_->locals.begin(),state_->locals.end(),[&](const auto& value){return value.original.device()==owner.original().device();});
    if(local!=state_->locals.end()) {
        local->enclosingParent=owner;
        for(const auto& weak:local->parents)if(const auto actual=weak.lock())actual->enclosing=owner;
    }
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
        for(const auto& local:state_->locals)for(const auto& weak:local.parents)if(const auto parent=weak.lock()) {
            const auto observed=[&](void* stream){return local.original.worker()->queryStreamExecutionState(stream,"original local auxiliary request terminal")==GPUStreamExecutionState::Complete;};
            complete=observed(parent->captureStream) && complete;
            if(parent->enclosing)complete=observed(parent->enclosing->stream()) && complete;
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
    for(const auto& local:state_->locals)for(const auto& weak:local.parents)
        if(!weak.expired())throw std::logic_error("Original local auxiliary parent/cache borrower is still live");
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
        for(const auto& owner:state_->owners)if(owner.executable && owner.program)
            state_->boundary->pooledStorage()->releaseOriginalRemoteDefinition(*owner.program,owner.executable);
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
    state_->locals.clear();
    state_->localHotParticipants.clear();
    state_->programs.clear();
    state_->ownsBoundaryClaim = false;
    state_->boundary->cohortOwnerClaim_.clear(std::memory_order_release);
}
PooledImmutableRetainedParticipants::~PooledImmutableRetainedParticipants()noexcept{
    if(!state_->ownsBoundaryClaim)return;
    if(state_->boundary->hasActiveInterval() || state_->source.worker()->ownsCurrentThread())std::terminate();
    for(const auto& owner:state_->owners)if(owner.original.worker()->ownsCurrentThread())std::terminate();
    for(const auto& local:state_->locals) {
        if(local.original.worker()->ownsCurrentThread())std::terminate();
        for(const auto& weak:local.parents)if(!weak.expired())std::terminate();
    }
    // Each closed executable rejects unobserved live readers before destroying
    // handles; its original device tokens remain process-retained afterwards.
    state_->enqueueGroup.reset();
    try {
        for(const auto& owner:state_->owners)if(owner.executable && owner.program)
            state_->boundary->pooledStorage()->releaseOriginalRemoteDefinition(*owner.program,owner.executable);
    }catch(...){std::terminate();}
    state_->owners.clear();
    state_->transport.reset();
    state_->boundary->cohortOwnerClaim_.clear(std::memory_order_release);
}
}
