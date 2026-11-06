/**
 * @file PooledImmutableDirectedTransport.h
 * @brief Original same-PMA directed channel ownership for declared remote immutable cache participants.
 *
 * Opposite-vendor remote owners reuse two original captured channels across
 * Pin, Free, Victim, Source, Publication, Read and retirement. Homogeneous native cohorts
 * continuation owners use their original native completed-byte communicator.
 * Same-vendor owners outside that communicator are unsupported; local owners
 * create no remote channel.
 * Participant graphs and recurring hosted transaction submission remain explicit
 * consumers; this owner launches no graph and holds no model or residency state.
 */
#pragma once
#include "PooledImmutableDirectedChannelGeometry.h"
#include "PooledImmutableLocalHotTransportGeometry.h"
#include "PooledImmutableLocalHotPoolQuery.h"
#include "PooledImmutableLocalHotNativePack.h"
#include "PooledImmutableTransferBinding.h"
#include "PooledImmutableReplicatedReaderGeometry.h"
#include "PooledImmutableParticipantRetirementGraph.h"
#include "ImmutableLookupExecutionBoundary.h"
namespace llaminar2::qwen4exp {
class IPooledImmutableLocalHotKernels;
/** @brief Complete immutable original chain for one addressed remote coordinate. */
struct PooledImmutableRemoteTransport final {
    std::uint32_t coordinate;
    std::shared_ptr<const PooledImmutableCapturedPinEdge> pin;
    std::shared_ptr<const PooledImmutableCapturedPinReturn> returned;
    std::shared_ptr<const PooledImmutableCapturedFreeWave> free;
    std::shared_ptr<const PooledImmutableCapturedVictimWave> victim;
    std::shared_ptr<const PooledImmutableCapturedSourceWave> source;
    std::shared_ptr<const PooledImmutableCapturedPublicationWave> publication;
    std::shared_ptr<const PooledImmutableCapturedReadWave> read;
    std::shared_ptr<const PooledImmutableCapturedRetirementWave> retirement;
};
/** @brief Original result/ACK channel pair for one actual foreign compute reader. */
class PooledImmutableReplicatedReaderTransport final {
public:
    /** @return Original declared reader coordinate, independent of physical storage coordinates. */
    std::uint32_t coordinate() const noexcept { return coordinate_; }
    /** @return Original source worker, independent of compute enumeration order. */
    const PreparedGPUContextUsePermit& originalSource() const noexcept { return source_; }
    /** @return Original reader worker permit retained until both native endpoints retire. */
    const PreparedGPUContextUsePermit& originalReader() const noexcept { return reader_; }
    /** @return Existing source-to-reader channel, phase-reused for result header and live output. */
    const std::shared_ptr<CapturedTransferChannel>& result() const noexcept { return result_; }
    /** @return Existing reader-to-source ACK channel. */
    const std::shared_ptr<CapturedTransferChannel>& acknowledgement() const noexcept { return acknowledgement_; }
    /** @return Explicit original NCCL/RCCL route, absent on an original cross-vendor channel. */
    bool nativeCollective() const noexcept { return bool(native_result_); }
    /** @return Source-to-reader original native fixed/header and counted payload endpoint. */
    const PooledImmutableTransferChannel& nativeResult() const noexcept { return native_result_; }
    /** @return Reader-to-source original native fixed ACK endpoint. */
    const PooledImmutableTransferChannel& nativeAcknowledgement() const noexcept { return native_acknowledgement_; }
    /** @return Exact source/reader metadata storage priced by the same channel geometry. */
    const std::shared_ptr<DeviceTransferBuffer>& control(CapturedTransferEndpoint side) const;
    /** @brief Reject substituted PMA/control blocks, retired worker generations or endpoint identities. */
    void requireCurrent() const;
private:
    friend class PooledImmutableDirectedTransport;
    /** @brief Construct only through the original cold transport owner after complete inventory validation. */
    PooledImmutableReplicatedReaderTransport(std::uint32_t coordinate,
        PreparedGPUContextUsePermit source, PreparedGPUContextUsePermit reader,
        std::shared_ptr<CapturedTransferChannel> result, std::shared_ptr<CapturedTransferChannel> acknowledgement,
        std::array<PhysicalMemoryAllocationLease,2> claims,
        std::array<std::shared_ptr<DeviceTransferBuffer>,2> controls,
        PooledImmutableTransferChannel nativeResult = {}, PooledImmutableTransferChannel nativeAck = {},
        PooledNativeReaderMessageProfile profile = PooledNativeReaderMessageProfile::ReplicatedFP32)
        : coordinate_(coordinate), source_(std::move(source)), reader_(std::move(reader)),
          result_(std::move(result)), acknowledgement_(std::move(acknowledgement)),
          control_claims_(std::move(claims)), controls_(std::move(controls)),
          native_result_(std::move(nativeResult)), native_acknowledgement_(std::move(nativeAck)), profile_(profile) {}
    std::uint32_t coordinate_;
    PreparedGPUContextUsePermit source_, reader_;
    std::shared_ptr<CapturedTransferChannel> result_, acknowledgement_;
    std::array<PhysicalMemoryAllocationLease,2> control_claims_;
    std::array<std::shared_ptr<DeviceTransferBuffer>,2> controls_;
    PooledImmutableTransferChannel native_result_, native_acknowledgement_;
    PooledNativeReaderMessageProfile profile_;
};
/** @brief Pure BOM projection and actual original TransferEngine materialization, never another admission ledger. */
class PooledImmutableDirectedTransport final:public std::enable_shared_from_this<PooledImmutableDirectedTransport> {
public:
    /** @brief Exact two-channel protocol floor at zero optional owner payload capacity. */
    static PhysicalMemoryPlan mandatoryBOM(const PooledImmutablePhysicalOwnerGeometry&,const PooledImmutableCoordinatorGeometry&,
        PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> originalGPUResources,
        const PooledImmutableReplicatedReaderGeometry* readers = nullptr,
        PooledNativeReaderMessageProfile profile = PooledNativeReaderMessageProfile::ReplicatedFP32);
    /** @brief Exact full two-channel contribution for this candidate's actual owner capacities. */
    static PhysicalMemoryPlan completeBOM(const PooledImmutablePhysicalOwnerGeometry&,const PooledImmutableCoordinatorGeometry&,
        PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> originalGPUResources,
        const PooledImmutableReplicatedReaderGeometry* readers = nullptr,
        PooledNativeReaderMessageProfile profile = PooledNativeReaderMessageProfile::ReplicatedFP32);
    /** @brief Source-owned exact growth above the same ordered protocol floor; add before candidate fits. */
    static PhysicalMemoryPlan slotDependentBOM(const PooledImmutablePhysicalOwnerGeometry&,const PooledImmutableCoordinatorGeometry&,
        PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> originalGPUResources,
        const PooledImmutableReplicatedReaderGeometry* readers = nullptr,
        PooledNativeReaderMessageProfile profile = PooledNativeReaderMessageProfile::ReplicatedFP32);
    /** @brief Bind original source permits/streams and existing directed waves once outside capture.
     * @param boundary Actual service/source owner retaining the original prepared participant cohort.
     * @param originalInventory Original rank discovery owner; required for every explicit same-vendor mapped or peer route.
     * @return Strong two-channel chain owner; no peer graph/worker/stream is created here.
     */
    static std::shared_ptr<const PooledImmutableDirectedTransport> create(std::shared_ptr<ImmutableLookupExecutionBoundary> boundary,
        std::shared_ptr<const ClusterInventory> originalInventory = {},
        std::optional<PooledImmutableReplicatedReaderGeometry> readers = {},
        std::shared_ptr<const Qwen4ExpReplicatedPLECohort> nativeCohort = {},
        std::optional<PooledImmutableLocalHotTransportGeometry> localHot = {});
    /** @return Original source boundary; foreign readers never reinterpret its device as their own. */
    const std::shared_ptr<ImmutableLookupExecutionBoundary>& sourceBoundary() const noexcept { return boundary_; }
    /** @return Actual foreign compute-reader channels; they are never storage-owner programs. */
    std::span<const std::shared_ptr<const PooledImmutableReplicatedReaderTransport>> readers() const noexcept
    { return readers_; }
    /** @return Exact declared compute-reader shape, absent on the original singleton path. */
    const std::optional<PooledImmutableReplicatedReaderGeometry>& readerGeometry() const noexcept
    { return reader_geometry_; }
    /** @brief Once-request original source publication/receive and local reset, before any retained Probe. */
    void enqueueLocalHotAdmission(const std::shared_ptr<PooledImmutableLocalHotPreparedStorage>&,
        const std::shared_ptr<const IPooledImmutableLocalHotKernels>&,
        const DeviceGenerationControlBinding&,void* exact_stream)const;
    /** @return Exact original native response destination for one actual consumer. */
    PooledImmutableLocalHotNativePack localHotNativePack(std::shared_ptr<PooledImmutableLocalHotPreparedStorage>)const;
    /** @return Closed actual compact query producers for the unchanged original pool Begin. */
    /**
     * @brief Retain the same actual compact producer with immutable read purpose.
     * @param local Original source participant's local cache/query storage.
     * @param bound Exact captured canonical query/consumer subset limits; storage stays unchanged.
     * @param purpose Expected union purpose, validated on device before coordinator policy work.
     * @return Closed original union/count/keys borrower; no new source or storage.
     * @throws std::exception For foreign source, coordinator, roots, bounds or purpose.
     */
    PooledImmutableLocalHotPoolQuery localHotPoolQuery(std::shared_ptr<PooledImmutableLocalHotPreparedStorage>,
        PooledImmutableLocalHotQueryBound bound,
        adaptive_lookup::pooled::LookupPurpose purpose=adaptive_lookup::pooled::LookupPurpose::CurrentDemand)const;
    /** @return Original optional compact geometry; old shared-only semantics remain distinct. */
    const auto& localHotGeometry()const noexcept{return local_hot_geometry_;}
    /** @return Exact participant-local actual request metadata, including source-self. */
    PooledNativeTransferSpan localHotControl(DeviceId participant,std::uint32_t consumer)const;
    /** @return Actual local consumer-authored Request88, never reconstructed from a response. */
    PooledNativeTransferSpan localHotOriginalRequest(DeviceId participant)const;
    /** @return Exact local once-per-request original source admission header. */
    PooledNativeTransferSpan localHotAdmission(DeviceId participant)const;
    /** @return Source-owned union metadata followed by distinct admitted native scratch roots. */
    PooledNativeTransferSpan localHotUnion()const;
    /** @return Source scratch by nominal role; absent singleton scratch is not fabricated. */
    enum class LocalHotSourceScratch {MergedKeys,IncomingKeys,OutgoingRows};
    PooledNativeTransferSpan localHotSourceScratch(LocalHotSourceScratch role)const;
    /** @return Actual remotes in original coordinate order; co-located owners are absent. */
    std::span<const PooledImmutableRemoteTransport> remotes()const noexcept{return remotes_;}
    /** @return Actual original addressed remote chain; a local/foreign coordinate rejects. */
    const PooledImmutableRemoteTransport& remote(std::uint32_t coordinate)const;
private:
    friend class ImmutableLookupExecutionBoundary;
    friend class PooledImmutableRetainedParticipants;
    /** @brief Revoke only original native caller bindings after all enclosing cache/parent borrowers have retired. */
    void retireOriginalRuntime() const noexcept;

    /** @brief Bind the actual whole-request source ticket before any native participant submits. */
    void beginInvocation(CapturedTransferInvocation original) const;
    /** @brief Original physical edge subset; source retirement precedes storage-edge cancellation. */
    enum class CancellationEdges { StorageOwners, ComputeReaders };
    /**
     * @return Strong original edge cancellation captured before partial Rank submission.
     * @param original Exact admitted group invocation, never a replacement generation.
     * @param edges Readers cancel immediately; storage owners cancel after the original source terminal proof.
     */
    std::function<void()> failureCallback(CapturedTransferInvocation original, CancellationEdges edges) const;
    /** @return Signal-only original TP failure action, bound by the service's exact ticket/lifecycle owner before launch. */
    std::function<void()> nativeFailureSignal() const;
    /** @brief Retire successful channel admission after the original complete native reader frontier. */
    [[nodiscard]] CapturedTransferInvocationRetirement finishInvocation(CapturedTransferInvocation original) const;
public:
    /** @brief Reject source/worker/generation/channel/root drift before graph composition. */
    void requireCurrent()const;
private:
    /** @brief Construct only after all actual source permits have authenticated. */
    explicit PooledImmutableDirectedTransport(std::shared_ptr<ImmutableLookupExecutionBoundary> boundary,
        std::shared_ptr<const ClusterInventory> originalInventory)
        :boundary_(std::move(boundary)),original_inventory_(std::move(originalInventory)){}
    /** @return Exact private control owner for an original local endpoint/consumer pair. */
    const std::shared_ptr<DeviceTransferBuffer>& localHotControlRoot(DeviceId,std::uint32_t)const;
    /** @brief Authenticate original source roots before any channel is materialized. */
    void requireOriginalSource() const;
    /** @return Existing physical channel owners only, with no new registry or progress state. */
    std::vector<std::shared_ptr<CapturedTransferChannel>> originalChannels(std::optional<CancellationEdges> edges = {}) const;
    std::shared_ptr<ImmutableLookupExecutionBoundary> boundary_;
    /** Original immutable discovery owner, retained until all original channels retire. */
    std::shared_ptr<const ClusterInventory> original_inventory_;
    std::vector<PooledImmutableRemoteTransport> remotes_;
    std::optional<PooledImmutableLocalHotTransportGeometry> local_hot_geometry_;
    /** Immutable original fixed-header endpoints; created once, never per lookup. */
    struct AdmissionEdge {std::uint32_t coordinate;PooledImmutableTransferBinding source,reader;};
    std::vector<AdmissionEdge> local_hot_admission_;
    /** Exact materialized roots, in self-control/union/merged/ingress/pack order. */
    std::array<PhysicalMemoryAllocationLease,5> local_hot_source_claims_;
    std::array<std::shared_ptr<DeviceTransferBuffer>,5> local_hot_source_roots_;
    std::shared_ptr<const PooledImmutableNativeCompletionStorage> native_completion_;
    std::optional<PooledImmutableReplicatedReaderGeometry> reader_geometry_;
    std::vector<std::shared_ptr<const PooledImmutableReplicatedReaderTransport>> readers_;
};
}
