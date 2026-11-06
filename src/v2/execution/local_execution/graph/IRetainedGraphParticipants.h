/**
 * @file IRetainedGraphParticipants.h
 * @brief Explicit participant captures surrounding one original retained source transaction.
 *
 * A source graph carries a typed collective plan. Cold capture binds that plan
 * to the actual complete source executable identity; the source executor then
 * submits its retained participant programs once per original parent launch.
 * Participants never substitute or capture the source model graph.
 */
#pragma once
#include <atomic>
#include <algorithm>
#include <span>
#include <vector>
#include "collective/LocalTPCaptureSourceDefinition.h"
#include "RetainedGraphParticipantEnqueueGroup.h"
#include "backends/DeviceId.h"
#include "backends/NativeContextPreparation.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <exception>
#include <stdexcept>
#include <string>
namespace llaminar2 {
namespace qwen4exp {class PooledImmutableRetainedParticipants;}
class ComputeGraph;
struct MoEExpertOverlayExecutionPlan;
/**
 * @brief Validate immutable rank-local mixed-GPU ownership for a native I/O auxiliary parent.
 * @param original Same original resolved execution descriptor retained by the participant owner.
 * @throws std::invalid_argument For CPU/homogeneous, distributed, foreign or incomplete ownership.
 * @note This metadata check grants no worker, source, graph or attachment authority.
 */
void requireRankLocalMixedGPUAuxiliaryExecution(const MoEExpertOverlayExecutionPlan& original);
class IRetainedGraphParticipantSourcePlan;
class IGPUGraphCapture;
class RetainedParticipantGraphExecutable;
class IGraphCaptureAuxiliaryBranch;
/** @brief Cache-owned original finite-body borrow; destruction follows native parent retirement. */
class IRetainedGraphParticipantStorageAttachment {
public:
    virtual ~IRetainedGraphParticipantStorageAttachment()=default;
    /** @brief Mutate the exact native parent only AFTER this borrow belongs to its original cache. */
    virtual bool attachTo(IGPUGraphCapture&,IGraphCaptureAuxiliaryBranch*)=0;
    /** @brief Reject changed source cache/native parent identity before actual replay. */
    virtual void requireCurrent()const=0;
};
class DeviceGraphExecutor;
class DeviceGraphOrchestrator;
/** @brief Actual original generation-loop stream owner, distinct from child recording streams.
 * Only the canonical DGO may mint this lifetime/ordering authority. It grants no
 * opaque resource certificate and creates no stream/event or alternative queue. */
class RetainedGraphParticipantParentStream final {
public:
    /** @return Original selected source worker/PMA/context, never ordinary context adoption. */
    const PreparedGPUContextUsePermit& original()const noexcept{return original_;}
    /** @return Exact original parent stream, not a guessed alias of child recording. */
    void* stream()const noexcept{return stream_.get();}
    /** @return Existing strong resource owner used to detect same-pointer/different-owner drift. */
    const std::shared_ptr<void>& owner()const noexcept{return stream_;}
private:
    friend class DeviceGraphOrchestrator;
    /** @brief Borrow only the DGO's existing stream after its original source permit is authenticated. */
    RetainedGraphParticipantParentStream(PreparedGPUContextUsePermit original,std::shared_ptr<void> stream)
        :original_(std::move(original)),stream_(std::move(stream)){
        if(!stream_ || reinterpret_cast<std::uintptr_t>(stream_.get())<=2)
            throw std::invalid_argument("Retained original parent stream cannot be null/default");
    }
    PreparedGPUContextUsePermit original_; ///< Keeps original context alive until the existing stream owner releases.
    std::shared_ptr<void> stream_;
};
/** @brief Per-device auxiliary ownership, never a declaration of source authority for a reader. */
enum class RetainedGraphParticipantAuxiliaryRole : std::uint8_t {Source,Reader};
/**
 * @brief Actual participant-local side range in the original graph.
 *
 * The canonical builder records these real stage names and identity. The
 * executor validates topology before recording; only its private body seal
 * can authorize attachment. Main LocalRead is explicitly outside this range.
 */
struct RetainedGraphParticipantAuxiliaryRange final {
    RetainedGraphParticipantAuxiliaryRole role=RetainedGraphParticipantAuxiliaryRole::Source;
    DeviceId participant=DeviceId::invalid();
    std::vector<std::string> nodes;
    std::string probe_ready,ready_publication,main_read,retirement;
    std::uint64_t identity=0;
    /** Same original immutable owner descriptor; no graph/source back-reference or policy copy. */
    std::shared_ptr<const MoEExpertOverlayExecutionPlan> original_execution;
    bool operator==(const RetainedGraphParticipantAuxiliaryRange&)const=default;
};
/** @brief Whole source and exact participant bodies have disjoint preparation meanings. */
enum class RetainedGraphParticipantDefinitionScope : std::uint8_t {
    WholeSource,AuxiliarySource,AuxiliaryReader
};
/** @brief Cache-owned preparation stream borrow, revoked before the original cache destroys its stream.
 * This capability creates no resource. The exact original worker serializes
 * native preparation with cache teardown; a retired borrow cannot authorize a
 * later request or a same-address replacement cache. */
class RetainedGraphParticipantPreparationStream final {
public:
    /** @brief Immutable declarative definition; it grants no native executable or launch. */
    struct Definition final {
        const ComputeGraph* graph=nullptr;
        const void* arena=nullptr;
        uint64_t topology_generation=0, variant=0;
        RetainedGraphParticipantDefinitionDomain domain;
        std::shared_ptr<const IRetainedGraphParticipantSourcePlan> plan;
        RetainedGraphParticipantDefinitionScope scope=RetainedGraphParticipantDefinitionScope::WholeSource;
        DeviceId participant=DeviceId::invalid();
        std::uint64_t range_identity=0;
        std::shared_ptr<const RetainedGraphParticipantPreparationStream> original_source_definition;
    };
    /** @return Exact executor-minted cold definition, absent for legacy stream-only borrows. */
    const Definition& definition()const noexcept{return definition_;}
    /** @return Original admitted actor: source for whole definitions, actual participant for auxiliary bodies. */
    const PreparedGPUContextUsePermit& originalSource()const noexcept{return original_;}
    /** @brief Revalidate original live cache, graph/count/retirement definition without an executable seal. */
    void requireDefinition()const;
    /** @brief Require already-authenticated cold structure plus live exact cache/provenance, without rescanning its DAG. */
    void requireCurrentDefinition()const;
    /** @return Actual existing cache stream, distinct from controller submission. */
    void* stream()const noexcept{return stream_;}
    /** @brief Reject retired borrows and changed original worker/PMA generations. */
    void requireLive(const PreparedGPUContextUsePermit& original)const{
        if(!live_.load(std::memory_order_acquire) || original.device()!=original_.device() ||
           original.memory().get()!=original_.memory().get() ||
           original.memory().owner_before(original_.memory()) || original_.memory().owner_before(original.memory()))
            throw std::logic_error("Original source preparation stream is retired or foreign");
        original_.requireWorker(original.worker());
    }
private:
    friend class DeviceGraphExecutor;
    /** @brief Mint only from the actual existing cache and its original source permit. */
    friend struct test::RetainedGraphParticipantsReplayAccess;
    RetainedGraphParticipantPreparationStream(PreparedGPUContextUsePermit original,void* stream)
        :original_(std::move(original)),stream_(stream){
        if(!stream || reinterpret_cast<std::uintptr_t>(stream)<=2)
            throw std::invalid_argument("Original source preparation stream is null/default");
    }
    /** @brief Mint the complete read-only definition only from the actual canonical cache. */
    RetainedGraphParticipantPreparationStream(PreparedGPUContextUsePermit original,void* stream,Definition definition)
        :RetainedGraphParticipantPreparationStream(std::move(original),stream){definition_=std::move(definition);}
    /** @brief Revoke on the original cache lifecycle before any native resource teardown. */
    void retire()noexcept{live_.store(false,std::memory_order_release);}
    PreparedGPUContextUsePermit original_;
    void* stream_;
    std::atomic<bool> live_{true};
    Definition definition_;
    mutable std::atomic<bool> definition_authenticated_{false};
};
/**
 * @brief Stage an additive cold alias extension without modifying any existing attachment.
 * @param previous Exact original source tokens already retained by one finite parent.
 * @param proposed Complete validated compatible source variants selected at the cold setup frontier.
 * @return Deduplicated old-plus-new aliases after every token/domain/PMA/generation check succeeds.
 * @throws std::exception Before publication for absent, stale or foreign source identity.
 * This is immutable topology metadata staging, not an execution permit. The
 * original participant owner separately validates each actual source occurrence
 * and commits all parent vectors only after the whole family passes.
 */
inline std::vector<std::shared_ptr<const RetainedGraphParticipantPreparationStream>>
retainedSourceVariantExtension(
    std::span<const std::shared_ptr<const RetainedGraphParticipantPreparationStream>> previous,
    std::span<const std::shared_ptr<const RetainedGraphParticipantPreparationStream>> proposed)
{
    if (previous.empty() || proposed.empty() || !previous.front())
        throw std::invalid_argument("Source alias refresh requires its original complete finite association");
    const auto& original = previous.front()->originalSource();
    const auto domain = previous.front()->definition().domain;
    std::vector<std::shared_ptr<const RetainedGraphParticipantPreparationStream>> result;
    const auto append = [&](const auto& values) {
        for (const auto& value : values) {
            if (!value) throw std::invalid_argument("Source alias refresh contains an absent original definition");
            value->requireCurrentDefinition();
            if(value->definition().scope!=RetainedGraphParticipantDefinitionScope::WholeSource)
                throw std::invalid_argument("Source alias refresh cannot adopt an auxiliary preparation body");
            value->requireLive(original);
            if (value->definition().domain != domain)
                throw std::invalid_argument("Source alias refresh contains a foreign original domain");
            const auto prior = std::find(result.begin(),result.end(),value);
            if (prior == result.end()) result.push_back(value);
            else if (prior->owner_before(value) || value.owner_before(*prior))
                throw std::invalid_argument("Source alias refresh substituted an original token control block");
        }
    };
    append(previous);append(proposed);return result;
}
/** @brief Exact original source capture ownership; no mutable token/model state is observed. */
struct RetainedGraphParticipantSourceIdentity final {
    const ComputeGraph* graph=nullptr;
    const void* arena=nullptr;
    DeviceId device=DeviceId::invalid();
    void* recording_stream=nullptr;
    std::uint64_t topology_generation=0;
    std::uint64_t capture_variant_signature=0;
    std::uint64_t capture_generation=0;
    const IGPUGraphCapture* original_parent=nullptr;
    bool operator==(const RetainedGraphParticipantSourceIdentity&)const=default;
};
/** @brief Only the canonical executor can mint an actual original source-cache seal. */
class RetainedGraphParticipantSourceSeal final {
public:
    /** @return Exact immutable source ownership validated by the original executor. */
    const RetainedGraphParticipantSourceIdentity& identity()const noexcept{return identity_;}
private:
    friend class DeviceGraphExecutor;
    friend struct test::RetainedGraphParticipantsReplayAccess;
    explicit RetainedGraphParticipantSourceSeal(RetainedGraphParticipantSourceIdentity identity):identity_(identity){}
    RetainedGraphParticipantSourceIdentity identity_;
};
/** @brief Exact enclosing parent observation for one real auxiliary body, never a reader SourceSeal. */
struct RetainedGraphParticipantAuxiliaryBodyIdentity final {
    const ComputeGraph* graph=nullptr;
    const void* arena=nullptr;
    DeviceId participant=DeviceId::invalid();
    void* enclosing_recording_stream=nullptr;
    const IGPUGraphCapture* enclosing_parent=nullptr;
    const IGPUGraphCapture* producer_fragment=nullptr; ///< Exact retained main unit ending at this replay's probe.
    std::uint64_t topology_generation=0,capture_variant_signature=0,capture_generation=0,range_identity=0;
    bool operator==(const RetainedGraphParticipantAuxiliaryBodyIdentity&)const=default;
};
/**
 * @brief Only the canonical executor seals an actual graph-only recorded body under its exact enclosing parent.
 * Strong original participant/body/preparation owners survive until attachment
 * and original parent retirement. A reader never receives source launch rights.
 */
class RetainedGraphParticipantAuxiliaryBodySeal final {
public:
    /** @return Exact enclosing native parent/range identity, never readiness. */
    const auto& identity()const noexcept{return identity_;}
    /** @return Actual admitted participant permit retained by its original executor. */
    const auto& originalParticipant()const noexcept{return participant_;}
    /** @return Actual record-only native owner; it must never be instantiated/enqueued independently. */
    const auto& recordedBody()const noexcept{return body_;}
    /** @return Revocable exact body preparation with its whole-source lineage. */
    const auto& preparation()const noexcept{return preparation_;}
private:
    friend class DeviceGraphExecutor;
    /** @brief Mint only after actual source-range recording and enclosing-parent identity are authenticated. */
    RetainedGraphParticipantAuxiliaryBodySeal(RetainedGraphParticipantAuxiliaryBodyIdentity identity,
        PreparedGPUContextUsePermit participant,std::shared_ptr<const IGPUGraphCapture> body,
        std::shared_ptr<const RetainedGraphParticipantPreparationStream> preparation)
        :identity_(identity),participant_(std::move(participant)),body_(std::move(body)),preparation_(std::move(preparation)){}
    RetainedGraphParticipantAuxiliaryBodyIdentity identity_;
    PreparedGPUContextUsePermit participant_;
    std::shared_ptr<const IGPUGraphCapture> body_;
    std::shared_ptr<const RetainedGraphParticipantPreparationStream> preparation_;
};
/** @brief Keep participant and original-source enqueues in one FIFO ownership scope.
 * This lock orders CPU enqueue ownership only; it never waits for native execution.
 * A partial native enqueue cannot be unwound as a successful invocation. */
class RetainedGraphParticipantSubmission final {
public:
    RetainedGraphParticipantSubmission(RetainedGraphParticipantSubmission&&)=default;
    RetainedGraphParticipantSubmission(const RetainedGraphParticipantSubmission&)=delete;
    RetainedGraphParticipantSubmission& operator=(const RetainedGraphParticipantSubmission&)=delete;
    /** @brief Fatal if the original source was not enqueued after its remote participants. */
    ~RetainedGraphParticipantSubmission(){if(lock_.owns_lock() && !source_enqueued_)std::terminate();}
private:
    friend class DeviceGraphExecutor;
    friend class qwen4exp::PooledImmutableRetainedParticipants;
    /** @brief Mint only around the original model-owned participant enqueue group. */
    explicit RetainedGraphParticipantSubmission(std::shared_ptr<const IRetainedGraphParticipantEnqueueCompletion> owner,std::unique_lock<std::mutex> lock)
        :owner_(std::move(owner)),lock_(std::move(lock)){}
    /** @brief Called by the sole canonical source launcher after its exact enqueue succeeds. */
    void completeOriginalEnqueue()noexcept{owner_->completeOriginalEnqueue();source_enqueued_=true;}
    std::shared_ptr<const IRetainedGraphParticipantEnqueueCompletion> owner_; ///< Native roots and mutex survive even if the invocation cache is concurrently withdrawn.
    std::unique_lock<std::mutex> lock_;
    bool source_enqueued_=false;
};
/** @brief One sealed source invocation and its separately retained participant programs. */
class IRetainedGraphParticipantInvocation {
public:
    /** @brief Retain operands until original native participant completion has been observed. */
    virtual ~IRetainedGraphParticipantInvocation()=default;
    /** @brief Reject source owner, geometry, topology, native variant or recording-stream drift. */
    virtual void requireCurrent(const RetainedGraphParticipantSourceSeal&)const=0;
    /** @brief Enqueue complete participant captures, never waiting for GPU progress or launching the source. */
    virtual RetainedGraphParticipantSubmission enqueueOriginalParticipants()const=0;
};
/** @brief Declarative collective metadata attached to the actual participant-local source graph. */
class IRetainedGraphParticipantSourcePlan {
public:
    /** @brief Source and participant strong ownership must survive every submitted native reader. */
    virtual ~IRetainedGraphParticipantSourcePlan()=default;
    /** @return Original consumed source permit; observing it never acquires or creates a worker. */
    virtual const PreparedGPUContextUsePermit& originalSource()const noexcept=0;
    /** @brief Authenticate the actual cache's revocable preparation-stream authority before admission. */
    virtual void authenticatePreparationStream(std::shared_ptr<const RetainedGraphParticipantPreparationStream>)const=0;
    /** @return Complete immutable original source/participant policy and root identity. */
    virtual std::uint64_t captureIdentity()const noexcept=0;
    /** @brief Authenticate the recording stream from its actual existing native capture owner before recording. */
    virtual void authenticateRecordingOwner(const IGPUGraphCapture& original)const=0;
    /**
     * @brief Bind only after the original full source parent has materialized without launch.
     * @param original Exact actual source cache identity, not a caller executable.
     * @return Sealed invocation owner with cold-prepared participant captures.
     * @throws std::exception Before unsupported parent policy, partial ownership or native preparation.
     */
    virtual std::shared_ptr<IRetainedGraphParticipantInvocation> bindOriginalSource(
        const RetainedGraphParticipantSourceSeal& original)const=0;
    /** @brief Reject absent or foreign declarative source definitions without minting an invocation. */
    virtual void requirePreparedSourceDefinition(const RetainedGraphParticipantPreparationStream&)const
    {throw std::logic_error("Source plan has no declarative preparation contract");}
    /** @brief Revalidate an already sealed source cache without cold DAG traversal/allocation. */
    virtual void requireCurrentPreparedSourceDefinition(const RetainedGraphParticipantPreparationStream&)const
    {throw std::logic_error("Source plan has no sealed preparation identity contract");}
    /** @brief Revalidate an immutable original storage body without its cold stage walk. */
    virtual void requireCurrentStorageDefinition(const RetainedParticipantGraphExecutable&)const
    {throw std::logic_error("Source plan has no sealed original storage identity contract");}
    /** @return Existing source/transport domain, not a graph role or selected depth. */
    virtual RetainedGraphParticipantDefinitionDomain definitionDomain()const noexcept{return {};}
    /**
     * @brief Attach an actual colocated storage body to its matched original native parent.
     * @param receiver Actual parent device from the canonical capture controller.
     * @param receiverGraph Actual original recipient graph, whose inline owner/reader stages must authenticate before attachment.
     * @param parent Original sealed, uninstantiated per-device parent.
     * @param progress Original optional progress contribution; never replaced or fabricated.
     * @param definition Exact source-cache token returned by the matched cold boundary.
     * @return Cache-owned attachment, or empty when this device owns no separate storage body.
     * @throws std::exception For unsupported transport/parent/progress or changed original identity.
     */
    virtual std::unique_ptr<IRetainedGraphParticipantStorageAttachment> prepareOriginalStorageAttachment(
        DeviceId,IGPUGraphCapture&,IGraphCaptureAuxiliaryBranch*,
        std::shared_ptr<const RetainedGraphParticipantPreparationStream>,const ComputeGraph&)const{return {};}
    /**
     * @brief Attach the actual compact Source/Reader auxiliary range to its original enclosing parent.
     * @param body Private canonical executor seal of the real record-only body and preparation scope.
     * @param progress Existing original auxiliary contribution, never a replacement controller.
     * @return Cache-owned existing finite-body attachment RAII; no standalone enqueue is permitted.
     * @throws std::exception For ordinary mode, foreign role/root/range/token/stream or partial sealing.
     */
    virtual std::unique_ptr<IRetainedGraphParticipantStorageAttachment> prepareOriginalAuxiliaryAttachment(
        const RetainedGraphParticipantAuxiliaryBodySeal&,IGraphCaptureAuxiliaryBranch*)const
    {throw std::logic_error("Source plan has no original participant auxiliary body contract");}
    /** @brief Authenticate one exact original storage definition independently of kernel/library recipe. */
    virtual void requirePreparedStorageDefinition(const RetainedParticipantGraphExecutable&)const
    {throw std::logic_error("Source plan has no original storage definition binding");}
    /** @brief Join the exact original storage input event on the actual original parent stream before cloning. */
    virtual void acquirePreparedStorageDefinition(const RetainedParticipantGraphExecutable&,void*)const
    {throw std::logic_error("Source plan has no original storage input frontier");}

};
/** @brief Validate only after the original revocable cache token proves its graph/plan lifetime. */
inline void RetainedGraphParticipantPreparationStream::requireDefinition()const {
    requireLive(original_);
    // Name only the absent declaration, never log pointers or live device words.
    const char* missing = !definition_.plan ? "plan" : !definition_.graph ? "graph" :
        !definition_.arena ? "arena" : !definition_.topology_generation ? "topology_generation" :
        !definition_.variant ? "variant" : !definition_.domain.source.is_gpu() ? "domain.source" :
        !definition_.domain.authority ? "domain.authority" : nullptr;
    if(missing)
        throw std::logic_error(std::string("Original source preparation has no complete declarative definition: missing ")+missing);
    if(definition_.scope==RetainedGraphParticipantDefinitionScope::WholeSource){
        if(definition_.participant.is_valid() || definition_.range_identity || definition_.original_source_definition)
            throw std::logic_error("Whole source preparation cannot carry participant auxiliary scope");
    }else if(definition_.scope==RetainedGraphParticipantDefinitionScope::AuxiliarySource ||
             definition_.scope==RetainedGraphParticipantDefinitionScope::AuxiliaryReader){
        const auto& source=definition_.original_source_definition;
        if(!definition_.participant.is_gpu() || !definition_.range_identity || !source || source.get()==this ||
           source->definition().scope!=RetainedGraphParticipantDefinitionScope::WholeSource ||
           source->definition().plan!=definition_.plan || source->definition().domain!=definition_.domain)
            throw std::logic_error("Auxiliary preparation lost its actual whole-source lineage/range");
        source->requireCurrentDefinition();source->requireLive(source->originalSource());
        const auto& sourceActor=source->originalSource();
        const auto& sourceMemory=sourceActor.memory();const auto& participantMemory=original_.memory();
        if(definition_.participant!=original_.device() || sourceActor.device()!=definition_.domain.source ||
           !sourceMemory || !participantMemory || sourceMemory.get()!=participantMemory.get() ||
           sourceMemory.owner_before(participantMemory) || participantMemory.owner_before(sourceMemory) ||
           sourceMemory->worldRank()<0 || sourceMemory->worldRank()!=participantMemory->worldRank() ||
           (definition_.scope==RetainedGraphParticipantDefinitionScope::AuxiliarySource && original_.device()!=definition_.domain.source) ||
           (definition_.scope==RetainedGraphParticipantDefinitionScope::AuxiliaryReader && original_.device()==definition_.domain.source))
            throw std::logic_error("Auxiliary preparation changed original participant/source role or PMA/rank");
    }else throw std::logic_error("Source preparation has unknown participant scope");
    definition_.plan->requirePreparedSourceDefinition(*this);
    definition_authenticated_.store(true,std::memory_order_release);
}
/** @brief Revocation is checked before dereferencing the original graph/plan. */
inline void RetainedGraphParticipantPreparationStream::requireCurrentDefinition()const {
    requireLive(original_);
    if(!definition_authenticated_.load(std::memory_order_acquire) || !definition_.plan)
        throw std::logic_error("Original source definition was never structurally authenticated");
    definition_.plan->requireCurrentPreparedSourceDefinition(*this);
}

}
