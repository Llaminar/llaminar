/**
 * @file PooledImmutableRetainedParticipants.h
 * @brief Model-owned recurring remote programs around the original hosted source parent.
 *
 * Ordinary MTP-off hosted ExpertOverlay transactions retain one original owner
 * capture per storage-only remote coordinate. The model engine alone captures
 * and launches its full source graph. Original immutable scheduler tickets
 * choose each transaction; this authority adds no host model/controller state.
 */
#pragma once
#include "PooledImmutableRetirementAuthority.h"
#include "PooledImmutableDirectedLookupGraph.h"
#include "PooledImmutableReplicatedReaderGeometry.h"
#include <optional>
#include "execution/local_execution/graph/IRetainedGraphParticipants.h"
namespace llaminar2 {struct GraphConfig;}
namespace llaminar2 { struct ClusterInventory; }
namespace llaminar2::qwen4exp {
/** @brief Closed original source/owner native lifetime, shared by exclusive model graph variants. */
class PooledImmutableRetainedParticipants final:public PooledImmutableRetirementAuthority,public IRetainedGraphParticipantEnqueueCompletion,
    public std::enable_shared_from_this<PooledImmutableRetainedParticipants> {
public:
    /**
     * @brief Validate the real ordinary hosted policy and prepare original remote captures off all workers.
     * @param boundary Original model-owned consumed source storage.
     * @param policy Actual full model policy; storage coordinates never become expert/compute roles.
     * @param originalInventory Original rank discovery owner; required for explicit same-vendor NoPeer transport.
     * @param readers Exact declared compute-reader cohort, distinct from actual CPU source channels.
     * @return Strong model lifetime; missing opaque host qualification rejects before native capture.
     */
    static std::shared_ptr<PooledImmutableRetainedParticipants> create(
        std::shared_ptr<ImmutableLookupExecutionBoundary> boundary,const GraphConfig& policy,
        std::shared_ptr<const ClusterInventory> originalInventory = {},
        std::optional<PooledImmutableReplicatedReaderGeometry> readers = {});
    /** @brief Reject active or unobserved readers before actual participant definitions or roots destruct. */
    ~PooledImmutableRetainedParticipants()noexcept override;
    /** @return Original source boundary retained by the existing directed transport. */
    const std::shared_ptr<ImmutableLookupExecutionBoundary>& boundary()const noexcept;
    /** @return Original immutable two-channel bundle; no local-self channel exists. */
    const std::shared_ptr<const PooledImmutableDirectedTransport>& transport()const noexcept;
    /**
     * @brief Attach one declarative source plan to the actual model fragment before merge.
     * @param graph Source fragment; the canonical recorder authenticates its final merged owner.
     * @param arena Original source arena, never an owner-local replacement.
     * @param completion Actual Read→last-reader→physical-retirement source descriptors.
     */
    void bindSourcePlan(ComputeGraph& graph,BufferArena& arena,PooledImmutableRetirementCompletion completion);
    /** @brief Reauthenticate a serialized model variant without replacing its original participant programs. */
    void requireForPolicy(const GraphConfig&)const;
    /** @brief Append the actual counted PLE source program and bind its retained participant plan. */
    PooledImmutableLookupGraphEndpoints appendPLE(ComputeGraph&,BufferArena&,
        std::shared_ptr<const IPooledImmutableNativeKernels>,DeviceRowRange,PooledNativeTensorSpan,
        std::optional<PooledNativeTensorSpan>,DeviceGenerationControlBinding,PooledNativeTensorSpan,
        const std::string&,const std::string&);
    /** @return The exact original prepared compute/storage participant, never a newly adopted worker. */
    PreparedGPUContextUsePermit originalParticipant(DeviceId device)const;
    /** @return True only for an actual original physical owner embedded in a declared compute reader. */
    bool hasEmbeddedParticipant(DeviceId device)const noexcept;
    /** @brief Retain one original DGO enclosing stream, independently of each FEE variant's capture stream. */
    void bindEmbeddedParentSubmissionStream(const RetainedGraphParticipantParentStream& owner);
    void requireCurrent()const override;
    void requireSource(const PooledImmutableSourcePreparedStorage&,void* stream)const override;
    void requireSourceStream(void* stream)const override;
    void bindSourceSubmissionStream(const RetainedGraphParticipantParentStream& owner)override;
    void awaitNativeTerminal(std::chrono::steady_clock::time_point deadline)const override;
private:
    /** @brief Join only original remote host enqueue calls after source submission. */
    void completeOriginalEnqueue()const noexcept override;
    class SourcePlan;
    class StorageAttachment;
    class Invocation;
    struct State;
    /** @brief Build only through the checked actual model policy factory. */
    explicit PooledImmutableRetainedParticipants(std::unique_ptr<State>);
    /** @brief Prepare all owner programs before original source graph materialization can run. */
    void prepareOriginalOwners();
    /** @brief Submit remote definitions once for the authenticated actual source parent invocation. */
    RetainedGraphParticipantSubmission enqueueOriginalParticipants(const RetainedGraphParticipantPreparationStream&)const;
    /** @brief Require every exact matched finite attachment before the original source can launch. */
    void requireInvocation(const RetainedGraphParticipantPreparationStream&)const;
    std::unique_ptr<State> state_;
};
}
