/**
 * @file RetainedParticipantGraphExecutable.h
 * @brief Closed process-retained owner of an original participant's General graph device budget.
 *
 * Generic retained owners price conservative host backing beside the existing
 * General device unit. Both original PMA tokens remain committed through
 * process reclamation; handle destruction and
 * context reset do not imply backing release. The closed directed remote
 * owner program adds its separately sized host allowance. Generic source and
 * terminal graphs use the same typed General contract as production capture;
 * their conservative General host allowance is separate from remote-program rows.
 */
#pragma once
#include "NativeContextPreparation.h"
#include "IGPUGraphCapture.h"
#include "NativeContextRetirement.h"
#include "NativeAuxiliaryResourceCatalog.h"
#include "memory/CPUStorageRoot.h"
#include <optional>
#include <memory>
#include <atomic>
namespace llaminar2 {
namespace qwen4exp {class PooledImmutableRetainedParticipants;class PooledImmutableCompleteCohort;}
namespace test {struct RetainedParticipantGraphExecutableReplayAccess;struct RetainedParticipantGraphExecutableFixtureAccess;struct PooledRemoteGraphHostReplayAccess;}
/** @brief Explicit graph-host authority; recorded/raw seams never claim installed qualification. */
enum class GraphHostQualification:std::uint8_t {DeviceBoundOnly,GeneralProcessAllowance,InstalledCertified,RecordedReplay,UnqualifiedDiagnostic};
/** @brief Exactly one original capture, stream and original host/device commitments. */
class RetainedParticipantGraphExecutable final {
public:
    /**
     * @brief Reserve the existing General bound before constructing its original capture.
     * @param original Actual original source/PMA/worker permit, not caller-supplied accounting or provenance.
     * @param stream Exact retained original worker stream; no stream/event is created.
     * @return Closed capture owner, initially not submitted.
     * @note Both General host/device tokens remain committed through ProcessRuntime; known wrapper objects are charged separately.
     */
    static std::shared_ptr<RetainedParticipantGraphExecutable> create(
        const PreparedGPUContextUsePermit& original,void* stream);
    /**
     * @brief Declare original General host/device slots and exact known wrapper backing before admission.
     * @param host Original CPU0 allocator observation.
     * @param gpu Original addressed GPU observation, with the same rank.
     * @param slots Number of actual retained executables; checked arithmetic rejects overflow.
     * @return Same-PMA typed BOM, with conservative host/device ProcessRuntime tokens and exact known objects.
     */
    static PhysicalMemoryPlan generalBOM(PhysicalMemoryResource host,PhysicalMemoryResource gpu,std::size_t slots=1);
    /**
     * @brief Price original CPU0 known graph objects/scratch and the selected opaque host program.
     * @param program Closed descriptor authored by the actual directed remote builder before admission.
     * @param host Original canonical CPU0 observation, independent of table NUMA placement.
     * @param gpu Exact original addressed remote observation; device General bytes stay separate.
     * @return Host-only named BOM; no native creation, provider hashing or live admission occurs.
     * @throws std::runtime_error For a missing/incompatible installed class.
     */
    static PhysicalMemoryPlan hostBOM(const PooledRemoteGraphProgram& program,
        PhysicalMemoryResource host,PhysicalMemoryResource gpu);
    /**
     * @brief Consume one installed original-PMA remote host program before its first native creator.
     * @param original Actual original prepared worker/PMA/generation capability.
     * @param stream Exact original non-default worker stream.
     * @param program Source-authored program used by the actual directed owner builder.
     * @return One retained definition with its original known/opaque host and device obligations.
     */
    static std::shared_ptr<RetainedParticipantGraphExecutable> create(
        const PreparedGPUContextUsePermit& original,void* stream,const PooledRemoteGraphProgram& program);
    /** @return Explicit immutable qualification marker, never inferred from a successful replay. */
    GraphHostQualification hostQualification()const noexcept{return hostQualification_;}
    /** @brief Destroy actual handles only after terminal observation, retaining original physical tokens. */
    ~RetainedParticipantGraphExecutable()noexcept;
    RetainedParticipantGraphExecutable(const RetainedParticipantGraphExecutable&)=delete;
    RetainedParticipantGraphExecutable& operator=(const RetainedParticipantGraphExecutable&)=delete;
    /** @return Distinct policy identity; it asserts conservative retention, not native host qualification. */
    static const char* policyIdentity()noexcept;
private:
    friend class qwen4exp::PooledImmutableRetainedParticipants;
    friend class NativeOrderingEventBudgetProjection;
    friend class qwen4exp::PooledImmutableCompleteCohort;
    friend struct test::RetainedParticipantGraphExecutableReplayAccess;
    friend struct test::RetainedParticipantGraphExecutableFixtureAccess;
    friend struct test::PooledRemoteGraphHostReplayAccess;
    /** @brief Actual device-token/handle lifecycle, privately replayed without minting a host certificate. */
    static std::shared_ptr<RetainedParticipantGraphExecutable> createDeviceBound(
        const PreparedGPUContextUsePermit&,void*);
    struct HostProgramBinding {
        PooledRemoteGraphProgram program;
        std::optional<NativeAuxiliaryResourceSelection> selected;
        std::size_t nativeObjectBytes;
        GraphHostQualification qualification;
    };
    /** @brief Distinguish ordinary General host ownership from the closed device-only replay/diagnostic boundary. */
    enum class GeneralHostOwnership {DeviceOnly,OriginalAllowance};
    /** @brief Same native creator for original General ownership, device-only diagnostics and closed remote programs. */
    static std::shared_ptr<RetainedParticipantGraphExecutable> createBound(
        const PreparedGPUContextUsePermit&,void*,std::optional<HostProgramBinding>,
        GeneralHostOwnership=GeneralHostOwnership::DeviceOnly);
    /** @brief Closed recorded selection enters the actual original reservation/creator body. */
    static std::shared_ptr<RetainedParticipantGraphExecutable> createRecordedRemote(
        const PreparedGPUContextUsePermit&,void*,const PooledRemoteGraphProgram&,
        NativeAuxiliaryResourceSelection,std::size_t compiledRecordedObjectBytes);
    /** @brief Explicit raw measurement with exact known roots and no invented opaque host allowance. */
    static std::shared_ptr<RetainedParticipantGraphExecutable> createDiagnosticRemote(
        const PreparedGPUContextUsePermit&,void*,const PooledRemoteGraphProgram&);
    /** @brief Shared typed known-root and selected opaque projection for recorded/installed planning. */
    static PhysicalMemoryPlan hostBOMWithSelection(const PooledRemoteGraphProgram&,
        PhysicalMemoryResource,PhysicalMemoryResource,const std::optional<NativeAuxiliaryResourceSelection>&,
        std::size_t compiledNativeObjectBytes);
    /** @brief Construct only after original PMA tokens and retirement slot exist. */
    RetainedParticipantGraphExecutable(std::shared_ptr<IWorkerGPUContext>,std::shared_ptr<PhysicalMemoryAuthority>,std::uint64_t,DeviceId,void*,NativeContextRetirementSlot,
        std::unique_ptr<IContextNativeRetirementObligation>);
    class RecordingCapture;
    enum class Phase:std::uint8_t {Fresh,Recording,Recorded,Ready,Failed,Retired};
    /** @brief Reject late recording changes without querying native state. */
    void requireConstruction()const;
    /** @brief Submit the next original invocation exactly once on its retained stream.
     * @param invocationOrdinal Consecutive lifecycle ownership ordinal, never GPU/controller progress.
     * @return Native enqueue success; possible partial failure retains all readers and forbids reuse.
     * @throws std::logic_error For unsealed, duplicated, skipped or failed submission.
     */
    bool submit(std::uint64_t invocationOrdinal);
    /** @return Whether the original sealed definition still admits its next owned invocation. */
    bool readyForSubmission()const;
    /** @return Recording-only facade; native launch/reset cannot bypass this owner through a retained base reference. */
    IGPUGraphCapture& capture()const;
    /** @return Actual original stream completion, queried only at whole-request terminal aggregation. */
    bool observeTerminal();
    /** @brief Close actual native handles after whole-request retirement, preserving original ProcessRuntime tokens.
     * Strong metadata aliases may survive, but can never submit/reset the retired definition.
     * This operation must run outside the original worker; actual teardown executes on that worker.
     */
    void retireDefinition();
    /** @brief Claim one fresh definition for exactly one retained participant program, without creating native resources. */
    void bindCohortDefinition();
    /** @brief Authenticate mutation against the live original worker, generation and admitted authority. */
    void requireRecordingAuthority()const;
    /** @brief Reject stale/reset/foreign original provenance before touching native state. */
    void requireLiveOriginal()const;
    std::shared_ptr<IWorkerGPUContext> worker_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_; ///< Same original control block; never a copied ledger.
    std::uint64_t generation_;
    DeviceId device_;
    void* stream_;
    NativeContextRetirementSlot retirement_;
    std::unique_ptr<IContextNativeRetirementObligation> pendingObligation_; ///< Not adopted until immediately before the first possible native creator.
    PhysicalMemoryAllocationLease nativeObjectLease_,recordingObjectLease_;
    AlignedVector<std::byte> inspectionScratch_; ///< Original bounded CPU0 root; never embedded in native work.
    std::optional<PooledRemoteGraphProgram> remoteProgram_;
    GraphHostQualification hostQualification_=GraphHostQualification::DeviceBoundOnly;
    std::unique_ptr<IGPUGraphCapture> capture_;
    std::unique_ptr<RecordingCapture> recording_;
    std::weak_ptr<RetainedParticipantGraphExecutable> self_; ///< Original factory control block, rejecting counterfeit aliases.
    std::atomic<Phase> phase_{Phase::Fresh};
    std::atomic<bool> cohortBound_{false}; ///< One-shot definition ownership, never GPU/model progress.

    std::uint64_t lastSubmission_=0; ///< Host ownership only; no model/controller state is mirrored.
    bool nativeAttempted_=false; ///< Sole native-handle lifecycle guard, never GPU model state.
    bool pending_=false; ///< Native-reader lifetime guard; never model/token/controller state.
};
}
