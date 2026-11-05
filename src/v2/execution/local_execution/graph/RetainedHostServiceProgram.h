/**
 * @file RetainedHostServiceProgram.h
 * @brief Immutable CPU service birth declaration, priced before admission and consumed afterward.
 *
 * Exact stack/object pages and opaque libc/TLS backing have distinct PMA owners.
 * No CPU service thread participates in GPU source observation or Preparing.
 */
#pragma once
#include "planning/PhysicalMemoryAuthority.h"
#include <memory>
#include <mutex>
#include <string>
#include <pthread.h>
#include <variant>
namespace llaminar2 {
class PreparedRetainedHostService;
class RetainedHostServiceBorrow;
class RetainedHostServiceUseToken;
struct RetainedHostServiceInstanceState;
class RetainedParentTicketServiceWorker;
namespace test {struct RetainedHostServiceReplayAccess;struct RetainedHostServiceCalibrationAccess;}
/** @brief Two existing independently progressing CPU programs, never GPU endpoints. */
enum class RetainedHostServiceRole {ImmutableLookupIO,ExecutorParentTicket};
/** @brief Immutable evidence role; diagnostics never become an installed resource certificate. */
enum class RetainedHostServiceCreatorPolicy {OriginalCaller,OriginalOpenMPPartition};
/** @brief Explicit evidence role; diagnostics cannot authorize production admission. */
enum class RetainedHostServiceEvidence {Installed,Recorded,UnqualifiedDiagnostic,InstalledWireProjection};
/** @brief Immutable original-rank exact geometry and coarse opaque contribution; never a live owner. */
struct RetainedHostServiceResourceDeclaration final {
    std::size_t stackBytes=0,guardBytes=0,pageBytes=0,objectBytes=0,objectAlignment=0,tlsBytes=0,opaqueBytes=0;
    /** @brief Reject invalid mapping, arithmetic overflow or a budget not derived by the shared policy. */
    void requireValid()const;
    bool operator==(const RetainedHostServiceResourceDeclaration&)const=default;
};
/** @brief Original attributes and source-bound immutable demand for one retained thread. */
class RetainedHostServiceProgram final {
public:
    /** @brief Immutable original native evidence shared by one declaration session, never a global cache. */
    class Evidence;
    /** @brief Cold observed work only; never an allocation/admission estimate. */
    struct EvidenceObservation {std::uint64_t filesHashed,bytesHashed,elapsedNanoseconds;};
    /** @return Actual one-session immutable hashing observations. */
    static EvidenceObservation evidenceObservation(const std::shared_ptr<const Evidence>&);
    /** @return Monotonic actual cold hash attempts, enabling a no-rehash replay without resetting global state. */
    static std::uint64_t nativeEvidenceHashAttempts()noexcept;
    /** @return One authenticated loaded-provider/TLS snapshot; no thread is created. */
    static std::shared_ptr<const Evidence> originalEvidence();
    /** @brief Derive a role using the same original session evidence without rehashing libraries. */
    static RetainedHostServiceProgram original(RetainedHostServiceRole,std::shared_ptr<const Evidence>);
    /** @brief Snapshot original pthread defaults and loaded TLS closure without creating a thread. */
    static RetainedHostServiceProgram original(RetainedHostServiceRole);
    /** @return Same immutable exact/opaque declaration before optional quota allocation. */
    PhysicalMemoryPlan bom(PhysicalMemoryResource originalCPU0)const;
    /** @return Exact writable mapping extent; PROT_NONE guard pages are excluded. */
    std::size_t allocationBytes()const noexcept;
    /** @return Original stack extent, excluding object pages. */
    std::size_t stackBytes()const noexcept;
    /** @return Exact virtual guard extent; never claimed as physical bytes. */
    std::size_t guardBytes()const noexcept;
    /** @return Original page granule. */
    std::size_t pageBytes()const noexcept;
    /** @return Actual immutable declaration used identically by local BOM and rank receipt. */
    RetainedHostServiceResourceDeclaration resourceDeclaration()const noexcept;
    /** @return Fixed compiled worker role. */
    RetainedHostServiceRole role()const noexcept;
    /** @return Existing creator affinity semantics, independent of physical CPU0 allocator identity. */
    RetainedHostServiceCreatorPolicy creatorPolicy()const noexcept;
    /** @return Runtime/ABI/default-attribute and original live-owner identity of this immutable declaration. */
    const std::string& identity()const noexcept;
    /** @return Portable runtime/ABI/default-policy class, excluding physical CPU index selectors. */
    const std::string& resourceIdentity()const noexcept;
    /** @return Explicit evidence ownership used by diagnostic receipts. */
    RetainedHostServiceEvidence evidenceKind()const noexcept;
private:
    friend class PreparedRetainedHostService;
    friend class RetainedHostServiceUseToken;
    friend class MemoryPlanner;
    friend class RetainedHostServiceBudgetProjection;
    friend class RetainedParentTicketServiceWorker;
    friend struct test::RetainedHostServiceReplayAccess;
    friend struct test::RetainedHostServiceCalibrationAccess;
    class TLSProviderEvidence;
    /** @return Actual selected loader/TLS provider, authenticated independently of passive TLS modules. */
    static std::shared_ptr<const TLSProviderEvidence> originalTLSProviderEvidence();
    /** @brief Same production provider validator with recorded selected-address input for device-free replay. */
    static void requireTLSProviderEvidence(const std::shared_ptr<const TLSProviderEvidence>&,const void*);
    enum class MetadataAuthority {Installed,Recorded,UnqualifiedDiagnostic,InstalledWire};
    static RetainedHostServiceProgram snapshot(RetainedHostServiceRole,MetadataAuthority,std::shared_ptr<const Evidence>);
    struct State;
    explicit RetainedHostServiceProgram(std::shared_ptr<const State>);
    /** @brief Closed replay selection; no installed override or public numeric constructor exists. */
    /** @brief Authenticate peer declaration against installed data; result can price but can NEVER materialize. */
    static RetainedHostServiceProgram fromInstalledWire(RetainedHostServiceRole,const std::string& resourceClass,const std::string& liveIdentity,const RetainedHostServiceResourceDeclaration&);
    static RetainedHostServiceProgram unqualifiedDiagnostic(RetainedHostServiceRole);
    static RetainedHostServiceProgram recorded(RetainedHostServiceRole,std::size_t opaqueBytes);
    /** @return Sealed recorded diagnostic origin for fast lifecycle replay, with no installed/native bound. */
    static RetainedHostServiceProgram recordedDiagnostic(RetainedHostServiceRole);
    static RetainedHostServiceProgram recordedWithEvidence(RetainedHostServiceRole,std::shared_ptr<const Evidence>,std::size_t opaqueBytes);
    /** @brief Reject delayed token policy/module mutation without selecting new evidence or hashing libraries. */
    void requireCurrent()const;
    /** @brief Start on original defaults after binding only this admitted stack. */
    int createThread(pthread_t*,void* stack,void*(*entry)(void*),void*,void(*beforeAttempt)(void*),void*)const;
    std::shared_ptr<const State> state_;
};
/** @brief Original opaque CPU obligation retained by process entry; never a second ledger. */
class RetainedHostServiceProcessObligation final {
private:
    friend class PreparedRetainedHostService;
    friend class GPUDeviceProcessLifetime;
    friend class RetainedHostServiceUseToken;
    struct UnqualifiedDiagnostic final {}; ///< Closed tag carries no invented byte allowance.
    using Original=std::variant<PhysicalMemoryOwnerReservation,UnqualifiedDiagnostic>;
    RetainedHostServiceProcessObligation(std::shared_ptr<PhysicalMemoryAuthority>,Original,std::string);
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    Original original_;
    std::string identity_;
    bool active_=false; ///< Original process owner alone publishes/closes actual creator use.
};
/** @brief Noncopyable admitted instance owns worker object, exact mapping and lifecycle token. */
class PreparedRetainedHostService final {
public:
    /** @brief Reserve original admitted bytes before mmap/pthread; unknown libc/TLS classes reject first. */
    static std::unique_ptr<PreparedRetainedHostService> materialize(RetainedHostServiceUseToken);
    /** @brief Drain/join, destroy object and unmap before releasing original exact claim. */
    ~PreparedRetainedHostService()noexcept;
    PreparedRetainedHostService(const PreparedRetainedHostService&)=delete;
    PreparedRetainedHostService& operator=(const PreparedRetainedHostService&)=delete;
    /** @return Actual unchanged worker state machine retained by this exact instance. */
    RetainedParentTicketServiceWorker& worker()const noexcept;
    /** @return Original immutable program, never a repriced substitute. */
    const RetainedHostServiceProgram& program()const noexcept;
    /** @return Exact original authority retained throughout teardown. */
    const std::shared_ptr<PhysicalMemoryAuthority>& memory()const noexcept;
private:
    friend class RetainedHostServiceBorrow;
    explicit PreparedRetainedHostService(std::unique_ptr<RetainedHostServiceInstanceState>);
    std::mutex borrow_mutex_; ///< Serializes cold exclusive execution ownership, never dispatch.
    bool borrowed_=false; ///< One protocol/executor may name the worker at a time.
    std::unique_ptr<RetainedHostServiceInstanceState> state_;
};
/** @brief Exclusive execution borrow of the same original model-retained worker and PMA.
 * Return requires Idle, including consumption of a completed invocation by await().
 * The owner remains alive through the borrow; acquisition never creates another thread.
 */
class RetainedHostServiceBorrow final {
public:
    RetainedHostServiceBorrow() noexcept=default;
    /** @brief Fail fatally rather than let an outstanding task retain destroyed protocol state. */
    ~RetainedHostServiceBorrow() noexcept;
    RetainedHostServiceBorrow(const RetainedHostServiceBorrow&)=delete;
    RetainedHostServiceBorrow& operator=(const RetainedHostServiceBorrow&)=delete;
    /** @brief Transfer exclusive execution ownership without changing the worker. */
    RetainedHostServiceBorrow(RetainedHostServiceBorrow&&) noexcept;
    /** @brief Return the prior idle borrow before transferring exclusive ownership. */
    RetainedHostServiceBorrow& operator=(RetainedHostServiceBorrow&&) noexcept;
    /** @brief Checked standalone adapter; original token materialization remains one-shot.
     * @param owner Exact already materialized original service, consumed by this call.
     * @return Exclusive idle execution borrow retaining the service and its original PMA.
     * @throws std::logic_error If owner is absent, already borrowed or not idle.
     */
    static RetainedHostServiceBorrow own(std::unique_ptr<PreparedRetainedHostService> owner);
    /** @brief Borrow one retained original worker, rejecting concurrent or non-idle use.
     * @param owner Same original prepared instance retained by its model binding.
     * @return Exclusive execution scope without rematerialization or another PMA claim.
     * @throws std::logic_error If owner is absent, already borrowed or has unconsumed work.
     */
    static RetainedHostServiceBorrow acquire(std::shared_ptr<PreparedRetainedHostService> owner);
    /** @brief Return only after the actual worker has cleared its invocation descriptor.
     * @throws std::logic_error If the actual worker is not Idle; the borrow remains held.
     * Empty scopes are already returned and require no operation.
     */
    void release();
    /** @return Whether this scope exclusively owns execution. */
    explicit operator bool()const noexcept{return bool(owner_);}
    /** @return Same original prepared instance; empty borrows reject access. */
    PreparedRetainedHostService* operator->()const;
    /** @return Same original prepared instance; empty borrows reject access. */
    PreparedRetainedHostService& operator*()const{return *operator->();}
private:
    explicit RetainedHostServiceBorrow(std::shared_ptr<PreparedRetainedHostService> owner):owner_(std::move(owner)){}
    std::shared_ptr<PreparedRetainedHostService> owner_;
};

}
