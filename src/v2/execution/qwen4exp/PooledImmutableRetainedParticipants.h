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
namespace llaminar2 { struct ClusterInventory; struct PooledImmutableLocalHotFutureSourceBody; }
namespace llaminar2::qwen4exp {
class PooledImmutableLocalHotPreparedStorage;
class IPooledImmutableLocalHotKernels;
/** @brief Closed original source/owner native lifetime, shared by exclusive model graph variants. */
class PooledImmutableRetainedParticipants final:public PooledImmutableRetirementAuthority,public IRetainedGraphParticipantEnqueueCompletion,
    public IRetainedGraphParticipantLocalConsumptionPlan,
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
    /**
     * @brief Build actual future source delivery, retaining the one local frame for later adoption.
     * @param graph Original source graph. @param arena Same prepared arena.
     * @param poolCode Original source implementation. @param local Actual coordinator local root.
     * @param code Same local implementation. @param tokens Genuine original token extent.
     * @param ids Already-authored ordered hashes. @param count Exact optional count producer.
     * @param control Same request controller row. @param identity Unique declaration namespace.
     * @return Actual source/ACK retirement; no model read, policy credit or local retirement.
     * @note The caller independently orders original accepted history and previous local retirement.
     */
    PooledImmutableLocalHotFutureSourceBody appendLocalHotFuturePLE(ComputeGraph& graph,BufferArena& arena,
        std::shared_ptr<const IPooledImmutableNativeKernels> poolCode,
        std::shared_ptr<PooledImmutableLocalHotPreparedStorage> local,
        std::shared_ptr<const IPooledImmutableLocalHotKernels> code,DeviceRowRange tokens,PooledNativeTensorSpan ids,
        std::optional<PooledNativeTensorSpan> count,DeviceGenerationControlBinding control,const std::string& identity);
    /**
     * @brief Build actual future reader request/delivery/ACK without waiting for a later model read.
     * @param graph Original participant graph. @param arena Same prepared arena.
     * @param local Actual reader local root. @param code Same local implementation.
     * @param control Original request controller. @param tokens Genuine original token extent.
     * @param ids Already-authored ordered hashes. @param identity Unique declaration namespace.
     * @param count Exact optional token-count producer.
     * @return Actual response acquisition and ACK retirement, retaining the one local frame.
     */
    PooledImmutableLocalHotFutureProducerEndpoints appendLocalHotFutureReader(ComputeGraph& graph,BufferArena& arena,
        std::shared_ptr<PooledImmutableLocalHotPreparedStorage> local,
        std::shared_ptr<const IPooledImmutableLocalHotKernels> code,DeviceGenerationControlBinding control,
        DeviceRowRange tokens,PooledNativeTensorSpan ids,const std::string& identity,
        std::optional<PooledNativeTensorSpan> count);
    /**
     * @brief Append complete actual-demand comparison without selecting its native branch.
     * @param graph Same participant graph. @param arena Same prepared arena.
     * @param local Original local root. @param code Same local implementation.
     * @param control Same request controller. @param tokens Actual complete demand extent.
     * @param ids Genuine ordered demand IDs. @param count Exact optional count producer.
     * @param identity Unique comparison node name.
     * @return Actual Adopt endpoint; original controller explicitly selects read or cleanup afterward.
     */
    std::string appendLocalHotAdoptionCheck(ComputeGraph& graph,BufferArena& arena,
        std::shared_ptr<PooledImmutableLocalHotPreparedStorage> local,
        std::shared_ptr<const IPooledImmutableLocalHotKernels> code,DeviceGenerationControlBinding control,
        DeviceRowRange tokens,PooledNativeTensorSpan ids,std::optional<PooledNativeTensorSpan> count,
        const std::string& identity);
    /**
     * @brief Build an explicitly selected local-only accepted or unused-future branch.
     * @param graph Same original participant graph. @param arena Same prepared arena.
     * @param local Original retained local root. @param code Same native implementation.
     * @param control Same request controller. @param branch Explicit controller-selected disposition.
     * @param output Sole model output for Accepted; absent for Discard.
     * @param identity Unique declaration namespace. @param outputIdentity Actual accepted LocalRead name.
     * @return Actual main and separately attached maintenance endpoints; no new source completion.
     * @note Source/transfer ACK retirement must already precede this submission. PredictionMiss uses
     * appendLocalHotPredictionMissRetirement because Adopt has already discarded that frame.
     */
    PooledImmutableLocalHotConsumerEndpoints appendLocalHotConsumer(ComputeGraph& graph,BufferArena& arena,
        std::shared_ptr<PooledImmutableLocalHotPreparedStorage> local,
        std::shared_ptr<const IPooledImmutableLocalHotKernels> code,DeviceGenerationControlBinding control,
        PooledImmutableLocalHotConsumerBranch branch,std::optional<PooledNativeTensorSpan> output,
        const std::string& identity,const std::string& outputIdentity);
    /**
     * @brief Retire a selected prediction-miss branch after its actual adoption check.
     * @param graph Same graph holding the actual r779 adoption stage.
     * @param arena Same local frame arena. @param local Original prepared local root.
     * @param code Same native local implementation. @param control Same request controller row.
     * @param adoptionCheck Actual complete-demand comparison stage on this graph.
     * @param identity Unique retirement node name.
     * @return Actual local Retire endpoint, with no SourcePlan, auxiliary or source completion.
     * @note The original controller must select this branch only for PredictionMiss.
     * Adoption already discarded the frame; a second Discard is invalid. Actual
     * source/transfer ACK retirement must precede that adoption submission.
     */
    std::string appendLocalHotPredictionMissRetirement(ComputeGraph& graph,BufferArena& arena,
        std::shared_ptr<PooledImmutableLocalHotPreparedStorage> local,
        std::shared_ptr<const IPooledImmutableLocalHotKernels> code,DeviceGenerationControlBinding control,
        const std::string& adoptionCheck,const std::string& identity);
    /** @return Same original prepared source/local roster; no new declaration owner is allocated. */
    RetainedGraphParticipantDefinitionDomain definitionDomain()const noexcept override;
    /** @return Immutable original retained source/local-owner identity, not live cache state. */
    std::uint64_t captureIdentity()const noexcept override;
    /** @brief Validate only original local ownership, never a new source execution/completion.
     * @param preparation Exact executor-issued definition. @param inspect Inspect stage topology only while cold. */
    void requireLocalDefinition(const RetainedGraphParticipantPreparationStream& preparation,bool inspect)const override;
    /** @brief Attach an actual local-only finite body without importing physical owner/source work.
     * @param body Exact sealed recording. @param progress Optional generic original progress, without source binding.
     * @return Attachment retaining same original local owner and native parent through retirement. */
    std::unique_ptr<IRetainedGraphParticipantStorageAttachment> prepareLocalAttachment(
        const RetainedGraphParticipantAuxiliaryBodySeal& body,IGraphCaptureAuxiliaryBranch* progress)const override;
    /** @return The exact original prepared compute/storage participant, never a newly adopted worker. */
    PreparedGPUContextUsePermit originalParticipant(DeviceId device)const override;
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
    /** @brief Share the original reader protocol between current and future lifetimes without copying stages. */
    PooledImmutableLocalHotGraphEndpoints appendLocalHotReaderBody(ComputeGraph&,BufferArena&,
        std::shared_ptr<PooledImmutableLocalHotPreparedStorage>,std::shared_ptr<const IPooledImmutableLocalHotKernels>,
        DeviceGenerationControlBinding,DeviceRowRange,PooledNativeTensorSpan,PooledNativeTensorSpan,
        const std::string&,const std::string&,std::optional<PooledNativeTensorSpan>,PooledImmutableLocalHotAccessIntent);
    /** @brief Install a local-only declaration over existing prepared roots; never mint a source plan.
     * @param graph Actual local graph. @param arena Same original arena.
     * @param local Original local root. @param range Complete typed finite body declaration. */
    void bindLocalConsumption(ComputeGraph& graph,BufferArena& arena,
        const std::shared_ptr<PooledImmutableLocalHotPreparedStorage>& local,RetainedGraphParticipantAuxiliaryRange range);
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
