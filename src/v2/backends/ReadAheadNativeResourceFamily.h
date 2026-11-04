/**
 * @file ReadAheadNativeResourceFamily.h
 * @brief Same-PMA certified ownership of one retained stream and two ordering events.
 *
 * This family owns infrastructure only, never GPU inference state. A closed
 * source-bound certificate must price both physical allocators before native
 * creation. Reservations are certified opaque commitments, never reported as
 * exact materialized allocations. There is deliberately no production
 * certificate until the actual closure and retirement bound are qualified.
 * An explicit private diagnostic variant grants no certificate or commitment;
 * its constructors are defined only in the named guarded test fixture.
 */
#pragma once
#include "backends/DeviceId.h"
#include "backends/IWorkerGPUContext.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "ReadAheadNativeResourceObligation.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <variant>

namespace llaminar2 {
class GPUContextProvenance;
class DeviceGraphOrchestrator;
namespace test {struct ReadAheadNativeResourceReplayAccess;struct PooledNativeProgramDiagnosticAccess;}
/** @brief Immutable provenance, never a selectable production execution mode. */
enum class NativeAuxiliaryOwnershipKind :std::uint8_t {InstalledCertified,UnqualifiedDiagnostic};
/** @brief Immutable physical/runtime/source closure checked before resource birth. */
struct ReadAheadNativeResourceClosure final {
    DeviceId device=DeviceId::invalid();
    std::uint64_t contextGeneration=0;
    std::string physicalUuid,architecture,driver,runtime,backendSource,resourceSource;
    std::string sourceIdentity; ///< Complete immutable actual original source-key identity.
    bool operator==(const ReadAheadNativeResourceClosure&)const=default;
};
/** @brief Exact primary-stream binding for one family; terminal binding is sealed separately. */
struct ReadAheadNativeResourceRequest final {
    ReadAheadNativeResourceClosure closure;
    void* primaryStream=nullptr;
};
/**
 * @brief Source-minted certified upper limits, not caller-supplied reserve bytes.
 * @details A future installed certificate must prove teardown retires its
 * opaque pool commitment, not merely that native handle destruction succeeds.
 */
class ReadAheadNativeResourceCertificate final {
public:
    /** @return Certified host commitment for precisely one stream/two events. */
    std::size_t hostBytes()const noexcept{return hostBytes_;}
    /** @return Certified device commitment for precisely one stream/two events. */
    std::size_t deviceBytes()const noexcept{return deviceBytes_;}
    /** @return Immutable source/evidence identity, never a mutable readiness flag. */
    const std::string& proofIdentity()const noexcept{return proof_;}
private:
    friend class ReadAheadNativeResourceFamily;
    friend struct test::ReadAheadNativeResourceReplayAccess;
    ReadAheadNativeResourceCertificate(ReadAheadNativeResourceClosure closure,std::size_t host,
        std::size_t device,std::string proof,ReadAheadNativeRetirementPolicy retirement=ReadAheadNativeRetirementPolicy::HandleDestruction)
        :closure_(std::move(closure)),hostBytes_(host),deviceBytes_(device),proof_(std::move(proof)),retirement_(retirement){}
    ReadAheadNativeResourceClosure closure_;
    std::size_t hostBytes_,deviceBytes_;
    std::string proof_;
    ReadAheadNativeRetirementPolicy retirement_;
};
/** @brief Exclusive retained resource family; active native readers make destruction fatal. */
// Mutually exclusive retained variants may share this exact owner only when
// their primary stream and closure are identical. One outstanding submission
// rejects concurrent reuse. Concurrent programs require separately certified
// physical families; no hidden global cache creates or shares one implicitly.
class ReadAheadNativeResourceFamily final {
public:
    /** @brief Capability minted only by the original successful terminal publisher. */
    class PublishedTerminal final {
    private:
        friend class ReadAheadNativeResourceFamily;
        friend class DeviceGraphOrchestrator;
        friend struct test::ReadAheadNativeResourceReplayAccess;
        friend struct test::PooledNativeProgramDiagnosticAccess;
        /** @brief Bind one exact family submission to its original event owner and producer. */
        PublishedTerminal(const std::shared_ptr<ReadAheadNativeResourceFamily>&,std::shared_ptr<void>,void*);
        std::weak_ptr<const ReadAheadNativeResourceFamily> family_;
        std::uint64_t submission_;
        std::shared_ptr<void> event_;
        void* producer_;
    };
    /**
     * @brief Require the installed certificate and claim both PMA commitments before native creation.
     * @param memory The existing complete original authority; no capacity resampling occurs.
     * @param worker Strong original context owner retained through native destruction.
     * @param device Exact original device, internally projected through installed certification.
     * @param primaryStream Original exact parent stream; no caller-authored closure is accepted.
     * @return Original shared owner for retained graph/executable aliases.
     * @throws std::runtime_error If the actual closure lacks an installed certificate.
     * @throws std::exception On invalid identity, admission, or native creation failure.
     */
    static std::shared_ptr<ReadAheadNativeResourceFamily> materialize(
        std::shared_ptr<PhysicalMemoryAuthority> memory,std::shared_ptr<IWorkerGPUContext> worker,
        DeviceId device,void* primaryStream);
    /**
     * @brief Project certified limits through the normal BOM before final admission.
     * @param selected Original selected backend/device whose installed allowance is priced without a local context.
     * @param host Original CPU0 physical observation from the same source plan.
     * @param device Original exact GPU observation from that plan.
     * @return Named host/device opaque-family BOM; absent installed families fail closed. Live creation separately requires the original certificate.
     */
    static PhysicalMemoryPlan bom(DeviceId selected,
        PhysicalMemoryResource host,PhysicalMemoryResource device);
    /** @brief Destroy only retired exact handles, then release the certified PMA commitments. */
    ~ReadAheadNativeResourceFamily()noexcept;
    ReadAheadNativeResourceFamily(const ReadAheadNativeResourceFamily&)=delete;
    ReadAheadNativeResourceFamily& operator=(const ReadAheadNativeResourceFamily&)=delete;
    /** @return Exact admitted device. */
    DeviceId device()const noexcept{return request_.closure.device;}
    /** @return Original one retained nondefault read-ahead stream. */
    void* stream()const noexcept{return side_;}
    /** @return Exact primary-to-side ordering event. */
    void* forkEvent()const noexcept{return fork_;}
    /** @return Exact side-to-primary ordering event. */
    void* joinEvent()const noexcept{return join_;}
    /** @return Original parent stream, never guessed from a current context. */
    void* primaryStream()const noexcept{return request_.primaryStream;}
    /** @return Original strong context authority used by capture adapters. */
    IWorkerGPUContext& workerContext()const noexcept{return *worker_;}
    /** @return Complete closure/handle/generation identity for retained capture caching. */
    std::uint64_t captureIdentity()const noexcept;
    /**
     * @brief Bind one existing parent terminal event, or authenticate its same strong owner for serialized variants.
     * @param event Original strong parent event owner, retained through family destruction.
     * @throws std::logic_error On replaced/active terminal binding or a forged same-address owner.
     */
    void bindTerminal(std::shared_ptr<void> event);
    /** @brief Mark native readers live before actual parent launch; capture recording never calls this. */
    void beginSubmission();
    /** @brief Record the bound terminal event on the exact parent stream after actual launch enqueue. */
    void recordTerminal();
    /** @brief Accept original successful publication without recording another event.
     * @param proof Original producer's exact event/controlblock/stream/submission capability.
     * @throws std::logic_error On stale, substituted, or duplicate publication.
     */
    void acceptTerminalPublication(const PublishedTerminal& proof);
    /** @return True only after the bound latest terminal record has completed; no host synchronization. */
    bool observeTerminal();
    /** @return Whether actual parent readers still prevent resource retirement; no native query occurs. */
    bool hasOutstandingSubmission()const;
    /** @return Actual ownership provenance; diagnostic handles are never installed certification. */
    NativeAuxiliaryOwnershipKind ownershipKind()const noexcept;
private:
    friend struct test::ReadAheadNativeResourceReplayAccess;
    friend struct test::PooledNativeProgramDiagnosticAccess;
    /** @brief Explicit raw diagnostic state; only the named test fixture can construct it. */
    struct UnqualifiedDiagnostic final {
        UnqualifiedDiagnostic(const UnqualifiedDiagnostic&)=default;
    private:
        friend struct test::PooledNativeProgramDiagnosticAccess;
        UnqualifiedDiagnostic()=default;
    };
    /** @brief Defined only in the dedicated test fixture; it grants no certified commitment. */
    ReadAheadNativeResourceFamily(std::shared_ptr<PhysicalMemoryAuthority>,std::shared_ptr<IWorkerGPUContext>,
        ReadAheadNativeResourceRequest,UnqualifiedDiagnostic);
    /** @brief Run the original three SDK creators for either explicitly represented ownership state. */
    void createNativeHandles();
    /** @brief Resolve only compiled source certificates; no public byte/document injection exists. */
    static ReadAheadNativeResourceCertificate requireInstalledCertificate(DeviceId device);
    /** @brief Internal positive path shared with device-free replay, after authentic certificate validation. */
    static std::shared_ptr<ReadAheadNativeResourceFamily> materializeCertified(
        std::shared_ptr<PhysicalMemoryAuthority>,std::shared_ptr<IWorkerGPUContext>,
        ReadAheadNativeResourceRequest,ReadAheadNativeResourceCertificate,const GPUContextProvenance&);
    /** @brief Retain canonical commitments before any native handle exists. */
    ReadAheadNativeResourceFamily(std::shared_ptr<PhysicalMemoryAuthority>,std::shared_ptr<IWorkerGPUContext>,
        ReadAheadNativeResourceRequest,ReadAheadNativeResourceCertificate,
        PhysicalMemoryOwnerReservation,PhysicalMemoryOwnerReservation,std::optional<NativeContextRetirementSlot>);
    /** @brief Destroy each successfully created handle in reverse order on the original worker. */
    void destroyHandles()noexcept;
    enum class Submission :std::uint8_t {Idle,Launched,TerminalRecorded};
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    std::shared_ptr<IWorkerGPUContext> worker_;
    std::weak_ptr<ReadAheadNativeResourceFamily> self_; ///< Original holder controlblock; no self-retention.
    ReadAheadNativeResourceRequest request_;
    std::variant<ReadAheadNativeResourceCertificate,UnqualifiedDiagnostic> certificate_;
    PhysicalMemoryOwnerReservation hostCommitment_,deviceCommitment_;
    std::optional<NativeContextRetirementSlot> retirementSlot_; ///< Adopted ORIGINAL tokens, never another claim.
    void* side_=nullptr;void* fork_=nullptr;void* join_=nullptr;
    std::shared_ptr<void> terminal_; ///< Original parent event retained, never a third created event.
    mutable std::mutex lifecycle_;
    Submission submission_=Submission::Idle;
    std::uint64_t submissionSequence_=0; ///< Host ownership epoch, never GPU execution state or capture identity.
};
} // namespace llaminar2
