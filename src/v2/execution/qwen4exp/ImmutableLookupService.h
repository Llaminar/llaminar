/**
 * @file ImmutableLookupService.h
 * @brief Retained immutable-table I/O worker with explicit mapped-channel ownership.
 *
 * The host processor sees immutable lookup IDs and I/O protocol metadata only.
 * GPU token, sampler, verifier, KV and response state remain device-owned.
 * A model-lifetime worker is armed before the complete parent launches and
 * drains until an explicit GPU terminal close. PhysicalMemoryAuthority leases
 * mapped backing; strong binding lifetime retains those bytes after teardown
 * begins, while typed sequence/ack rules prohibit early response reuse.
 */
#pragma once
#include "ImmutableLookupProtocol.h"
#include "kernels/qwen4exp/ImmutableSubordinateLookupBinding.h"
#include "../../kernels/qwen4exp/ImmutableLookupKernels.h"
#include "../../loaders/HostImmutableRowProvider.h"
#include "../local_execution/graph/RetainedParentTicketServiceWorker.h"
#include "../local_execution/graph/RetainedStorageStatistics.h"
#include "../../transfer/TransferEngine.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>

namespace llaminar2::qwen4exp
{
/** @brief CPU I/O observations; never sampled mutable inference/controller state. */
using ImmutableLookupServiceStats=RetainedStorageCounters;
/** @brief Outcome of one nonblocking CPU poll of a release-published immutable request. */
enum class ImmutableLookupPoll { Pending, Completed };

/** @brief Device-free host half of the exact immutable row protocol.
 * @details The caller supplies already-admitted aligned channel bytes. The
 * processor owns only host I/O sequence bookkeeping; all protocol publication
 * words remain in the shared channel and are accessed with acquire/release.
 * One host owner serializes poll, closed and source borrowing. Concurrent host
 * calls are unsupported; GPU publications may progress concurrently. Returned
 * immutable row Pins have their own owner mutex/lifetime and may outlive the processor.
 */
class ImmutableLookupHostProcessor final
{
public:
    /** @brief Bind and initialize one newly admitted channel before any GPU capture.
     * @param table Model-owned native RAM/mmap source, retained by the service.
     * @param layout Checked maximum geometry, never used as a live payload count.
     * @param channel Aligned CPU view of the complete admitted mapped backing.
     * @param tableId Positive immutable service-instance identity.
     * @param fatalStatus Optional CPU I/O failure publication, released before a failed GPU completion becomes visible.
     * @param cancelled Existing service cancellation flag, retained through this processor; never a substitute for GPU publication validation.
     */
    ImmutableLookupHostProcessor(HostImmutableRowProvider &table,ImmutableLookupLayout layout,
        std::span<std::byte> channel,std::uint64_t tableId,
        std::atomic<ImmutableLookupStatus> *fatalStatus=nullptr,const std::atomic<bool> *cancelled=nullptr);
    /** @brief Retire the sole subordinate ingress; old read pins retain their exact source/PMA backing. */
    ~ImmutableLookupHostProcessor();
    ImmutableLookupHostProcessor(const ImmutableLookupHostProcessor &)=delete;
    ImmutableLookupHostProcessor &operator=(const ImmutableLookupHostProcessor &)=delete;
    /**
     * @brief Borrow only a row of this processor's actual successful unconsumed subordinate request.
     * @param responseRow Live response coordinate already authenticated from the acquired channel.
     * @return Existing immutable RAM pin, or absence when the response used SSD/no caching.
     * @throws std::exception For cancellation, failed/stale/consumed grants, promotion rows or foreign coordinates.
     * @note This is a read-lease API for the storage infrastructure, not an independent lookup or proof factory.
     */
    std::optional<CpuAdaptiveImmutableRowCache::Pin> borrowSubordinateResponseRow(std::size_t responseRow);
    /** @brief Process at most one newly published exact row request.
     * @param generation Current API-admission generation, independent of model execution decisions.
     * @param invocation Actual retained-worker dispatch, or zero for an unticketed direct processor diagnostic.
     * @return Pending without touching payload, or Completed after release-publishing a response.
     */
    ImmutableLookupPoll poll(std::uint64_t generation,std::uint64_t invocation=0);
    /** @brief Verify that the GPU terminal closed after acknowledging every serviced response.
     * @param generation Exact API-admission generation being drained.
     * @return True after its explicit device close, false while it remains open.
     * @throws std::runtime_error On an out-of-order acknowledgement or terminal close.
     */
    bool closed(std::uint64_t generation);
    /** @brief Snapshot terminal-safe CPU-owned I/O counters. */
    ImmutableLookupServiceStats stats() const noexcept { return stats_; }
    /** @brief Return bounded fatal diagnostic captured by this I/O owner. */
    const std::string &lastFailure() const noexcept { return lastFailure_; }
    /** @brief Current serviced protocol position, not a mirrored inference-loop counter. */
    std::uint64_t lastSequence() const noexcept { return lastSequence_; }
    /** @brief CPU-owned first fatal protocol result, independent of model state. */
    ImmutableLookupStatus lastStatus() const noexcept{return lastStatus_;}
private:
    friend struct ImmutableLookupSubordinateTestAccess;
    friend class ImmutableLookupService;
    /** @brief Drop this worker's acquired source grant before releasing its exact terminal witness.
     * @param generation Original admitted I/O interval, never a model-state observation.
     * @param invocation Actual retained-worker dispatch ticket.
     * @note This does not release channel/frame backing or independently held reader pins.
     */
    void retireInvocation(std::uint64_t generation,std::uint64_t invocation) noexcept;
    /** @return A matching GPU-owned I/O abort after acquiring invocation before its generation body.
     * @param generation Exact admitted service interval. @param invocation Actual worker dispatch, zero only for direct processor diagnostics.
     */
    bool sourceAbortRequested(std::uint64_t generation,std::uint64_t invocation) const noexcept;
    /** @brief Account a GPU abort once after this exact worker joins, including publication racing retirement.
     * @param generation Original joined admission. @param invocation Original joined dispatch ticket.
     * @return Whether that original GPU I/O abort was published; no model/controller state is observed.
     * @note Service await owns the unique Serving-to-terminal transition and cannot call this twice for one ticket.
     */
    bool observeTerminalAbort(std::uint64_t generation,std::uint64_t invocation);
    /** @brief Immutable identity of one actual acquire; it neither keeps a mutable-frame reference nor promises borrowed bytes. */
    class SubordinateReadBorrow final {
        friend class ImmutableLookupHostProcessor;
        /** @brief Only this processor's successful acquire snapshots its original completed interval. */
        SubordinateReadBorrow(const ImmutableLookupHostProcessor *owner,std::uint64_t generation,std::uint64_t sequence)
            :owner_(owner),generation_(generation),sequence_(sequence) {}
        const ImmutableLookupHostProcessor *owner_;
        std::uint64_t generation_,sequence_;
    };
    /** @return Actual current acquired interval after response, cancellation and consumption checks. */
    SubordinateReadBorrow acquireSubordinateRead();
    /** @brief Reject stale interval identity before touching a later batch's CPU-owned row bank. */
    std::optional<CpuAdaptiveImmutableRowCache::Pin> finishSubordinateRead(
        const SubordinateReadBorrow &borrow,std::size_t responseRow);
    /** @brief Release exactly the required completion metadata after native-row writes. */
    void complete(const ImmutableLookupRequest &request,std::uint64_t sequence,ImmutableLookupStatus status);
    /** @brief Count each acquired GPU retirement word once and reject future acknowledgements. */
    std::uint64_t observeAcknowledgement();
    HostImmutableRowProvider *table_;
    ImmutableLookupLayout layout_;
    std::span<std::byte> channel_;
    std::uint64_t tableId_,lastSequence_=0,lastAcknowledgement_=0,lastClose_=0;
    ImmutableLookupServiceStats stats_;
    std::string lastFailure_;
    ImmutableLookupStatus lastStatus_=ImmutableLookupStatus::Success;
    std::atomic<ImmutableLookupStatus> *fatalStatus_=nullptr;
    const std::atomic<bool> *cancelled_=nullptr; ///< Existing service cancellation; never a GPU publication authority.
    std::optional<CpuAdaptiveImmutableRowCache::ReadGrant> subordinateGrant_; ///< Actual acquired frame, no second channel sequence ledger.
};

/** @brief Exact mapped backing and its sole PMA lease, retained by graph bindings. */
struct ImmutableLookupChannelBacking final
{
    PhysicalMemoryAllocationLease physical; ///< Destroyed after mapped pages retire.
    std::shared_ptr<MappedHostTransferRegion> mapped;
    ImmutableLookupLayout layout;
    std::uint64_t tableId;
};

/** @brief Persistent CPU worker for one explicitly declared GPU/CPU storage lane. */
class ImmutableLookupService final : public std::enable_shared_from_this<ImmutableLookupService>
{
public:
    /** @brief Exact admitted I/O interval and actual retained-worker invocation; neither can be substituted. */
    struct Ticket final {
        std::uint64_t generation=0, invocation=0;
        /** @return Whether both original identities name a real admission. */
        constexpr bool valid() const noexcept { return generation && invocation; }
    };
    /** @brief Strong service/backing lifetime plus one exact native alias address space. */
    class Binding final
    {
    public:
        /** @brief Return immutable channel pointers authenticated by TransferEngine setup. */
        const ImmutableLookupDeviceView &deviceView() const noexcept { return view_; }
        /** @brief Return exact endpoint used by these addresses. */
        DeviceId device() const noexcept { return device_; }
    private:
        friend class ImmutableLookupService;
        /** @brief Retain the complete I/O owner until every captured binding retires. */
        Binding(std::shared_ptr<ImmutableLookupService> owner,DeviceId device,ImmutableLookupDeviceView view)
            :owner_(std::move(owner)),device_(device),view_(view) {}
        std::shared_ptr<ImmutableLookupService> owner_;
        DeviceId device_;
        ImmutableLookupDeviceView view_;
    };
    /** @brief Exact page-rounded CPU mapped staging demand for the canonical BOM. */
    static std::size_t hostBytes(ImmutableLookupLayout layout);
    /** @brief Allocate exactly one PMA-admitted mapped slot and persistent worker.
     * @param table Existing RAM/mmap native row authority; no table copy is created.
     * @param layout Exact maximum ID/native-row geometry.
     * @param devices Explicit mapped endpoint set, supplied by graph placement.
     * @param memory Canonical rank-bound physical admission authority.
     * @param hostDevice Exact CPU/NUMA allocator charged for staging.
     * @param transfers Public mapping authority, retaining its exact backend bindings; setup injection permits device-free worker proofs.
     * @return Shared service whose binding can safely outlive caller teardown.
     */
    static std::shared_ptr<ImmutableLookupService> create(std::shared_ptr<HostImmutableRowProvider> table,
        ImmutableLookupLayout layout,std::span<const DeviceId> devices,
        std::shared_ptr<PhysicalMemoryAuthority> memory,DeviceId hostDevice,
        TransferEngine &transfers=TransferEngine::instance());
    /** @brief Cancel/retire any outstanding host invocation before owned state is released. */
    ~ImmutableLookupService();
    /** @brief Freeze one exact device alias during setup, retaining this service lifetime. */
    Binding bind(DeviceId device);
    /** @brief Bind explicit protocol2 to the original source-issued pooled native storage; ordinary protocol1 remains distinct. */
    ImmutableSubordinateLookupBinding bindSubordinate(DeviceId device);
    /** @brief Arm I/O before launching the retained generation parent.
     * @param generation Monotonic API-admission identity; never a host token/loop decision.
     * @return Exact worker invocation awaited only after the GPU parent terminal.
     */
    Ticket arm(std::uint64_t generation);
    /** @brief Join one exact completed service interval after native terminal observation. */
    bool await(Ticket ticket);
    /** @brief Cancel only the exact outstanding I/O admission without claiming response consumption.
     * @param ticket Original admission and actual worker invocation returned by arm().
     * @return False for a stale/foreign invocation, leaving the current admission untouched.
     * @note Worker retirement is separately release-published after acquired source grants retire.
     */
    bool cancel(Ticket ticket);
    /** @brief Return terminal I/O statistics; rejected while an invocation is live. */
    ImmutableLookupServiceStats stats() const;
    /** @brief Immutable actual table policy; no worker or device observation is performed. */
    HostImmutableRowMode mode() const noexcept{return table_->mode();}
    /**
     * @brief Observe actual retained CPU source and mapped-channel claims only after worker retirement.
     * @return Same source authority/allocator/locality and checked actual lease-byte sums.
     * @throws std::exception During a live interval, stale source, foreign authority or invalid channel
     * lease.
     * @note This neither performs a gather nor observes GPU state; it is not a reservation or storage pin.
     */
    HostImmutableRowOwnerObservation retainedSourceOwnerObservation() const;
    /** @brief Return terminal fatal I/O text; rejected while an invocation is live. */
    std::string lastFailure() const;
    /** @brief Reject an I/O lane that already release-published a fatal completion. */
    void requireHealthy() const;
private:
    friend struct ImmutableLookupSubordinateTestAccess;
    friend class ImmutableSubordinateLookupBinding; ///< Original source/backing validation and closed setup
                                                    ///< factory.
    /** @brief Create/reuse this original source channel's exact pooled registration before capture. */
    std::shared_ptr<PooledImmutableSourcePreparedStorage> declarePooledSourceStorage(const ImmutableSubordinateLookupBinding& binding,
        const PooledImmutableSourceGeometry& geometry,std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner,
        std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream);
    std::shared_ptr<PooledImmutableSourcePreparedStorage> pooled_storage_; ///< Roots only; never retains service/binding and creates no ownership cycle.
    std::mutex pooled_setup_; ///< Setup metadata publication only, independent of live device state.
    /** @brief Construct only through PMA-authenticated create(). */
    ImmutableLookupService(std::shared_ptr<HostImmutableRowProvider> table,
                           std::shared_ptr<ImmutableLookupChannelBacking> backing,
                           std::shared_ptr<const PhysicalMemoryAuthority> memory);
    /** @brief Allocation-free retained-worker entry. */
    static bool run(void *context);
    /** @brief Complete host I/O lifecycle; failed channels cannot be silently reused. */
    enum class Lifecycle { Idle,Serving,Failed };
    Lifecycle state_=Lifecycle::Idle;
    std::string serviceFailure_;
    std::atomic<ImmutableLookupStatus> fatalStatus_{ImmutableLookupStatus::Success};
    std::shared_ptr<HostImmutableRowProvider> table_;
    std::shared_ptr<ImmutableLookupChannelBacking> backing_;
    std::shared_ptr<const PhysicalMemoryAuthority>
        memory_;                         ///< Exact channel claim authority, never a byte ledger.
    std::atomic<bool> cancelled_{false}; ///< Constructed before and retained beyond its processor binding.
    ImmutableLookupHostProcessor processor_;
    mutable std::mutex lifecycle_;
    RetainedParentTicketServiceWorker worker_;
    Ticket outstanding_;
    std::uint64_t generation_=0;
};
} // namespace llaminar2::qwen4exp
