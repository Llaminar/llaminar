/**
 * @file PooledImmutableRetainedParticipants.h
 * @brief Model-owned recurring remote programs around the original hosted source parent.
 *
 * Each selected finite continuation family retains its exact original source
 * definition. Dedicated storage coordinates own their existing native parents;
 * expert followers borrow finite storage siblings inside their actual expert
 * parents. The original scheduler ticket chooses the transaction, while this
 * authority retains source/PMA/worker identity and exact retirement duties.
 */
#pragma once
#include "PooledImmutableRetirementAuthority.h"
#include "PooledImmutableDirectedLookupGraph.h"
#include "PooledImmutableReplicatedReaderGeometry.h"
#include "PooledImmutableLocalHotGraph.h"
#include "PooledImmutableLocalHotTransportGeometry.h"
#include <optional>
#include "execution/local_execution/graph/IRetainedGraphParticipants.h"
namespace llaminar2 {struct GraphConfig; class RankOrchestrator; class MoEOverlayParticipantGraphRunner; class Qwen4ExpReplicatedPLECohort;}
namespace llaminar2 { struct ClusterInventory; }
namespace llaminar2::qwen4exp {
class PooledImmutableLocalHotPreparedStorage;
class IPooledImmutableLocalHotKernels;
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
        std::optional<PooledImmutableReplicatedReaderGeometry> readers = {},
        std::optional<PooledImmutableLocalHotTransportGeometry> localHot = {},
        std::span<const std::shared_ptr<PooledImmutableLocalHotPreparedStorage>> localHotParticipants = {});
    /** @brief Reject active or unobserved readers before actual participant definitions or roots destruct. */
    ~PooledImmutableRetainedParticipants()noexcept override;
    /** @return Original source boundary retained by the existing directed transport. */
    const std::shared_ptr<ImmutableLookupExecutionBoundary>& boundary()const noexcept;
    /** @brief Passive projection of one actually attached original expert storage owner, without retaining its native parent. */
    struct ExpertOwnerParentObservation {
        std::uint32_t owner_coordinate;
        PhysicalMemoryAllocatorIdentity allocator;
        const PhysicalMemoryAuthority* memory_identity;
        std::uint64_t worker_generation;
        std::weak_ptr<IWorkerGPUContext> worker;
        std::vector<std::weak_ptr<const void>> parent_lifetimes; ///< Exact existing attachment control blocks, never a second readiness ledger.
        std::size_t source_variant_count;
    };
    /**
     * @brief Inspect actual cold expert-owner attachments after original serving family materialization.
     * @return Original coordinate/PMA/worker identities and weak actual parent borrows; storage-only/reader owners are excluded.
     * @throws std::exception If an observed attachment lost its original sealed native parent or source variants.
     * This performs host metadata validation only. Parent presence proves cold
     * composition, never kernel execution, numerical output or native terminal completion.
     */
    std::vector<ExpertOwnerParentObservation> expertOwnerParentObservations() const;
    /** @return Exact optional combined geometry; absence retains shared-only behavior. */
    const std::optional<PooledImmutableLocalHotTransportGeometry>& localHotGeometry()const noexcept;
    /** @return Complete immutable local root roster supplied atomically before any graph recording. */
    std::span<const std::shared_ptr<PooledImmutableLocalHotPreparedStorage>> localHotParticipants()const noexcept;
    /**
     * @brief Borrow the original colocated physical owner's native code for combined graph construction.
     * @param device Actual declared local-cache consumer, independent of pool ownership.
     * @return The existing InlineReader owner's code; empty for a valid consumer with no inline owner.
     * @throws std::invalid_argument For absent combined policy or a foreign consumer.
     * No owner, worker, source role or executable is created by this projection.
     */
    std::shared_ptr<const IPooledImmutableNativeKernels> localHotInlineOwnerCode(DeviceId device) const;
    /** @return The SAME original immutable owner descriptor; this borrow creates no policy copy or source authority. */
    const std::shared_ptr<const MoEExpertOverlayExecutionPlan>& originalExecutionPolicy() const noexcept;
    /** @return Original immutable two-channel bundle; no local-self channel exists. */
    const std::shared_ptr<const PooledImmutableDirectedTransport>& transport()const noexcept;
    /**
     * @brief Attach one declarative source plan to the actual model fragment before merge.
     * @param graph Source fragment; the canonical recorder authenticates its final merged owner.
     * @param arena Original source arena, never an owner-local replacement.
     * @param completion Actual Read→last-reader→physical-retirement source descriptors.
     */
    void bindSourcePlan(ComputeGraph& graph,BufferArena& arena,PooledImmutableSourceRetirementCompletion completion);
    /** @brief Reauthenticate a serialized model variant without replacing its original participant programs. */
    void requireForPolicy(const GraphConfig&)const;
    /** @brief Append the actual counted PLE source program and bind its retained participant plan. */
    PooledImmutableLookupGraphEndpoints appendPLE(ComputeGraph&,BufferArena&,
        std::shared_ptr<const IPooledImmutableNativeKernels>,DeviceRowRange,PooledNativeTensorSpan,
        std::optional<PooledNativeTensorSpan>,DeviceGenerationControlBinding,PooledNativeTensorSpan,
        const std::string&,const std::string&);
    /** @brief Declare actual reader result and any colocated owner phases in the SAME original PLE graph.
     * Inline native storage has no independently submitted graph. Its original
     * source token authenticates this full graph before native instantiate.
     */
    PooledImmutableLookupGraphEndpoints appendReader(ComputeGraph&, BufferArena&,
        DeviceGenerationControlBinding, DeviceRowRange, PooledNativeTensorSpan,
        const std::string&, const std::string&, std::optional<PooledNativeTensorSpan>);
    /**
     * @brief Declare the actual original source's combined local/shared PLE stages.
     * @param graph Same source model graph; no nested or replacement graph.
     * @param arena Original participant activation/private protocol arena.
     * @param poolCode Existing original pool code bundle.
     * @param local Actual same-source/PMA local roots, including C0.
     * @param localCode Original selected local native code bundle.
     * @param tokens Original counted model-token rows. @param ids Original UInt64 sixteen-head hash span.
     * @param count Exact original token count producer, absent only for fixed rows.
     * @param control Original generation error/controller root. @param output Exact FP32 model embedding.
     * @param identity Original declaration namespace. @param outputIdentity Original model output name.
     * @return Probe→LocalRead model-ready edges plus separately retained auxiliary retirement.
     */
    PooledImmutableLocalHotGraphEndpoints appendLocalHotPLE(ComputeGraph& graph,BufferArena& arena,
        std::shared_ptr<const IPooledImmutableNativeKernels> poolCode,
        std::shared_ptr<PooledImmutableLocalHotPreparedStorage> local,
        std::shared_ptr<const IPooledImmutableLocalHotKernels> localCode,
        DeviceRowRange tokens,PooledNativeTensorSpan ids,std::optional<PooledNativeTensorSpan> count,
        DeviceGenerationControlBinding control,PooledNativeTensorSpan output,
        const std::string& identity,const std::string& outputIdentity);
    /**
     * @brief Declare an actual foreign compute reader's symmetric combined stages.
     * @param graph Original reader model graph, receiving no SourcePlan/source launch authority.
     * @param arena Original reader arena. @param local Actual same-source/PMA local roots, including C0.
     * @param localCode Original reader native code. @param control Original generation controller.
     * @param tokens Original counted token rows. @param ids Original sixteen-head hash span.
     * @param output Exact FP32 model output. @param identity Actual declaration namespace.
     * @param outputIdentity Original model output name. @param count Exact optional token-count producer.
     * @return Main readiness and separately sealed Reader-role auxiliary range.
     */
    PooledImmutableLocalHotGraphEndpoints appendLocalHotReader(ComputeGraph& graph,BufferArena& arena,
        std::shared_ptr<PooledImmutableLocalHotPreparedStorage> local,
        std::shared_ptr<const IPooledImmutableLocalHotKernels> localCode,
        DeviceGenerationControlBinding control,DeviceRowRange tokens,PooledNativeTensorSpan ids,
        PooledNativeTensorSpan output,const std::string& identity,const std::string& outputIdentity,
        std::optional<PooledNativeTensorSpan> count);
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
    friend class ::llaminar2::ModelContext;
    friend class ::llaminar2::RankOrchestrator;
    friend class ::llaminar2::MoEOverlayParticipantGraphRunner;
    /** @brief Bind only actual original expert-owner source variants before its parent capture. */
    void bindOriginalExpertSourceVariants(DeviceId receiver, ComputeGraph& graph,
        std::vector<std::shared_ptr<const RetainedGraphParticipantPreparationStream>> definitions) const;
    /** @brief One actual retained follower parent and its proposed exact source signature variants. */
    struct OriginalExpertSourceBinding {
        DeviceId receiver;
        const ComputeGraph* graph;
        std::vector<std::shared_ptr<const RetainedGraphParticipantPreparationStream>> definitions;
    };
    /** @brief Validate the whole cold family before atomically adding aliases to unchanged original native parents. */
    void refreshOriginalExpertSourceVariants(std::span<const OriginalExpertSourceBinding> bindings) const;
    /** @brief Authenticate actual source occurrences without mutating graph or parent ownership. */
    void requireOriginalExpertSourceVariants(DeviceId receiver,
        std::span<const std::shared_ptr<const RetainedGraphParticipantPreparationStream>> definitions) const;
    /** @return Whether this exact original physical owner is also a declared expert follower. */
    bool hasOriginalExpertOwner(DeviceId receiver) const noexcept;
    /**
     * @brief Match only the actual parent identity, including after fatal cancellation.
     * @param original Original Rank-issued cohort/control block; an absent parent never matches.
     * @return Whether this published source belongs to that exact parent, without native admission.
     */
    bool belongsToOriginalCohort(const std::shared_ptr<const Qwen4ExpReplicatedPLECohort>& original) const noexcept;
    /** @brief Close runner-only definitions after every original parent/cache borrower retired while all arenas still exist. */
    void retireOriginalRuntimeBeforeArenaRelease();
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
