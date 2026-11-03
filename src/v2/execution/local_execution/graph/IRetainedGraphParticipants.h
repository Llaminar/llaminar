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
#include "collective/LocalTPCaptureSourceDefinition.h"
#include "RetainedGraphParticipantEnqueueGroup.h"
#include "backends/DeviceId.h"
#include "backends/NativeContextPreparation.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <exception>
#include <stdexcept>
namespace llaminar2 {
namespace qwen4exp {class PooledImmutableRetainedParticipants;}
class ComputeGraph;
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
    };
    /** @return Exact executor-minted cold definition, absent for legacy stream-only borrows. */
    const Definition& definition()const noexcept{return definition_;}
    /** @return Original admitted source permit; observing it creates no context. */
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
     * @param parent Original sealed, uninstantiated per-device parent.
     * @param progress Original optional progress contribution; never replaced or fabricated.
     * @param definition Exact source-cache token returned by the matched cold boundary.
     * @return Cache-owned attachment, or empty when this device owns no separate storage body.
     * @throws std::exception For unsupported transport/parent/progress or changed original identity.
     */
    virtual std::unique_ptr<IRetainedGraphParticipantStorageAttachment> prepareOriginalStorageAttachment(
        DeviceId,IGPUGraphCapture&,IGraphCaptureAuxiliaryBranch*,
        std::shared_ptr<const RetainedGraphParticipantPreparationStream>)const{return {};}
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
    if(!definition_.plan || !definition_.graph || !definition_.arena || !definition_.topology_generation ||
       !definition_.variant || !definition_.domain.source.is_gpu() || !definition_.domain.authority)
        throw std::logic_error("Original source preparation has no complete declarative definition");
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
