/**
 * @file CPUFP32PagedKVTable.h
 * @brief PMA-backed immutable CPU page-table banks with bounded version borrowers.
 *
 * Metadata is materialized before work. A prepared bank is inaccessible to
 * consumers; commit validates complete native page geometry and transfers its
 * ownership to an immutable snapshot. Old banks retain every page publication
 * until their last borrower retires. No update allocates, rebinds live storage,
 * copies retained KV payload or overwrites a pinned bank. This CPU-only witness
 * is not a GPU table controller or complete request-history/pressure authority.
 */
#pragma once
#include "CPUFP32PagedKVView.h"
#include "EffectiveUseOrdering.h"
#include "PagedKVArchive.h"
#include <memory>
#include <optional>

namespace llaminar2
{
/** @brief Explicit admitted table metadata and concurrent snapshot envelope. */
struct CPUFP32PagedKVTableGeometry
{
    size_t page_capacity = 0;
    size_t bank_count = 0;
    size_t readers_per_bank = 0;
    int page_tokens = 0;
    /** @brief Finite nonce issuance budget (metadata). Zero or headroom>=budget is invalid. */
    std::uint64_t issuance_budget = 0;
    /** @brief Maximum outstanding reservation capacity; must be < budget. */
    std::uint64_t issuance_headroom = 0;
    /** @brief Reject empty, overflowing or unrepresentable metadata geometry before allocation. */
    void validate() const;
};

/** @brief Single CPU runner-thread metadata authority; immutable page bytes may be consumed synchronously. */
class CPUFP32PagedKVTable final : public std::enable_shared_from_this<CPUFP32PagedKVTable>
{
public:
    /** @brief One immutable operation snapshot retaining actual page referents, not just a header. */
    /**
     * @brief Authority-issued consumable reservation for accepted-model work.
     *
     * plan P1 accepted-work completion: only a reservation HOLDER is allowed
     * to consume the headroom, and only up to its admitted capacity. The
     * identity is minted by the table under the finite budget and cannot be
     * copied; foreign, exhausted or revoked handles are refused before any
     * issuance. Normal admission never counts against outstanding reserved
     * capacity, so accepted work always completes within its own reservation.
     */
    class IssuanceReservation final
    {
    public:
        IssuanceReservation() = default;
        ~IssuanceReservation();
        IssuanceReservation(IssuanceReservation &&other) noexcept;
        IssuanceReservation &operator=(IssuanceReservation &&other) noexcept;
        IssuanceReservation(const IssuanceReservation &) = delete;
        IssuanceReservation &operator=(const IssuanceReservation &) = delete;
        /** @return Whether this handle holds a live reservable grant. */
        explicit operator bool() const noexcept { return table_ != nullptr; }
        /** @return Remaining consumable issuances of this reservation. */
        std::size_t remaining() const noexcept { return remaining_; }
    private:
        friend class CPUFP32PagedKVTable;
        IssuanceReservation(std::shared_ptr<CPUFP32PagedKVTable> table, std::size_t capacity,
            std::uint64_t serial);
        void release() noexcept;
        std::shared_ptr<CPUFP32PagedKVTable> table_;
        std::size_t remaining_ = 0;
        std::uint64_t serial_ = 0;
    };
    class Snapshot final
    {
    public:
        Snapshot() = default;
        ~Snapshot();
        Snapshot(Snapshot &&other) noexcept;
        Snapshot &operator=(Snapshot &&other) noexcept;
        Snapshot(const Snapshot &) = delete;
        Snapshot &operator=(const Snapshot &) = delete;
        /** @return Native view borrowing this exact retained bank and page publications. */
        CPUFP32PagedKVView view() const;
        /** @return A new bounded borrower, or empty when the declared reader envelope is exhausted. */
        std::optional<Snapshot> tryBorrow() const;
        /** @return Non-reused bank incarnation for identity checks and evidence. */
        std::uint64_t version() const noexcept { return version_; }
    private:
        friend class CPUFP32PagedKVTable;
        Snapshot(std::shared_ptr<CPUFP32PagedKVTable> table, size_t bank, std::uint64_t version);
        void release() noexcept;
        std::shared_ptr<CPUFP32PagedKVTable> table_;
        size_t bank_ = 0;
        std::uint64_t version_ = 0;
        // NOTE: a snapshot carries NO touch identity. Every successful touch
        // returns a move-only lease that owns its acknowledgement authority;
        // borrowing a snapshot obtains no capability and copies no ordinal
        // (R11: independent borrowers must touch and own their own lease).
    };
    /** @brief Exclusive unpublished bank preparation; abandonment retires its original page owners. */
    class Update final
    {
    public:
        Update() = default;
        ~Update();
        Update(Update &&other) noexcept;
        Update &operator=(Update &&other) noexcept;
        Update(const Update &) = delete;
        Update &operator=(const Update &) = delete;
        /** @brief Transfer one immutable page publication into the next logical table position. */
        void append(CPUFP32KVPage &&page);
        /** @brief Adopt one shared immutable page reference into the next logical table position. */
        void appendShared(PhysicalPagePool::SharedPage &&shared, int heads, int dimension);
        /**
         * @brief Attach a validated archived prefix into fresh private pages.
         * @param archive Authenticated FP32 attention section (provenance checked).
         * @param grant Caller request with protected growth for every page.
         * @param page_tokens Native block size for page slicing.
         * @param trust_domain Required authenticated isolation domain; any other
         * value refuses before a single byte is staged.
         * @param ancestry_id Required exact ancestry proof identity; a mismatch
         * refuses as an incompatible section, never an interpretation.
         * @note The prefix is materialized privately; no sampler/RNG/history or
         * domain-wide placement state is restored, and the attachment is fully
         * gated by the archive provenance before any byte is copied.
         */
        void attachArchivedPrefix(const PagedKVArchive &archive, PhysicalPagePool::Grant &grant, int page_tokens,
            std::uint64_t trust_domain = 0, std::uint64_t ancestry_id = 0);
        /** @brief Validate and publish a complete nonempty operation bank, without an unowned interval. */
        Snapshot commit();
    private:
        friend class CPUFP32PagedKVTable;
        Update(std::shared_ptr<CPUFP32PagedKVTable> table, size_t bank, std::uint64_t version);
        void release() noexcept;
        std::shared_ptr<CPUFP32PagedKVTable> table_;
        size_t bank_ = 0;
        std::uint64_t version_ = 0;
    };
    /** @brief Price actual bounded object cells and bank metadata through CPUStorageRoot. */
    static size_t physicalBytes(CPUFP32PagedKVTableGeometry geometry, DeviceId device,
        CPUStoragePlacement placement = CPUStoragePlacement::local());
    /** @brief Materialize the declared metadata once under its original PMA allocator. */
    static std::shared_ptr<CPUFP32PagedKVTable> create(std::shared_ptr<PhysicalMemoryAuthority> authority,
        DeviceId device, CPUFP32PagedKVTableGeometry geometry,
        CPUStoragePlacement placement = CPUStoragePlacement::local());
    ~CPUFP32PagedKVTable();
    /** @brief Reserve an unused metadata bank, or defer without changing any published bank. */
    std::optional<Update> tryBeginUpdate();
    /**
     * @brief Mint one consumable accepted-work reservation under headroom.
     * @param capacity Issuances this accepted obligation may consume (or 0 to
     * probe headroom only).
     * @return A move-only reservation handle, or empty without mutation when
     * the finite budget cannot admit the requested capacity (headroom is
     * shared among live reservations and never exceeds budget minus issued).
     */
    std::optional<IssuanceReservation> reserveAcceptedWork(std::size_t capacity);
    /**
     * @brief Consume one issuance from an accepted-work reservation.
     * @param reservation Exact minted handle; a foreign/exhausted handle is
     * refused before any bank or version state changes.
     * @return A prepared update, or empty when no bank is free or the
     * reservation is exhausted.
     * @note Reserved begins never consume normal-admission capacity, and
     * normal begins never consume reservation capacity: accepted work always
     * completes within its own finite grant.
     */
    std::optional<Update> tryBeginReserved(IssuanceReservation &reservation);
    /** @return Finite admitted issuance budget (never an independent maximum). */
    std::uint64_t issuanceBudget() const noexcept { return geometry_.issuance_budget; }
    /** @return Total issued versions, burned and live, monotonic across banks. */
    std::uint64_t issuedVersions() const noexcept { return next_version_; }
    /** @return Remaining headroom capacity not already reserved (budget-aware). */
    std::uint64_t headroomAvailable() const noexcept
    {
        const auto rem = geometry_.issuance_budget - next_version_;
        const auto spare = geometry_.issuance_headroom - reserved_outstanding_;
        return rem < spare ? rem : spare;
    }
    /** @return Outstanding accepted-work capacity currently reserved. */
    std::uint64_t outstandingReserved() const noexcept { return reserved_outstanding_; }
    /** @return Remaining normal-admission capacity (budget minus headroom minus issued). */
    std::uint64_t normalAdmissionRemaining() const noexcept
    {
        const auto cap = geometry_.issuance_budget - geometry_.issuance_headroom;
        return next_version_ >= cap ? 0 : cap - next_version_;
    }
    /** @return FINITE version exhaustion state: normal admission closed. */
    bool issuanceBudgetExhausted() const noexcept
    {
        return next_version_ >= geometry_.issuance_budget;
    }
    /**
     * @brief Borow the published snapshot of one request slot (P2 stage pool).
     * @param slot Request slot index mapped to a bank (0..bank_count-1).
     * @return A borrowed snapshot, or empty when the slot has no published
     * bank or the reader envelope is exhausted (the stage then fails; it never
     * silently substitutes the ring cache).
     */
    std::optional<Snapshot> tryBorrowForStage(int slot);

    /** @brief Freeze new borrows of an exact published version before joining its existing readers. */
    void closeBorrowers(const Snapshot &snapshot);
    /** @return Original PMA instance identity for same-authority admission checks. */
    const PhysicalMemoryAuthority *issuerForAdmission() const noexcept { return authority_.get(); }

    /**
     * @brief Touch one successful attachment (§8.2 effective use).
     * @param snapshot EXACT published snapshot the touch adopted (the table
     * authenticates its bank, version and incarnation against live state;
     * a burned/unpublished version or foreign table refuses).
     * @return A move-only lease owning this touch's acknowledgement authority,
     * or empty with NO state change on rollback-free refusal (duplicate,
     * obsolete, journal full, sealed). The snapshot is never mutated and
     * carries no touch identity; borrowing a snapshot obtains no capability
     * (independent consumers touch and own their own lease).
     */
    std::optional<EffectiveUseLease> touchEffectiveUse(const Snapshot &snapshot);
    /** @return Highest accepted retention watermark (§8.2 ordering point). */
    std::uint64_t effectiveUseWatermark() const noexcept { return effective_use_.watermark(); }
    /** @return Whether @p incarnation may be evicted (all touches acked). */
    bool effectiveUseMayEvict(std::uint64_t incarnation) const noexcept
    {
        return effective_use_.mayEvict(incarnation);
    }
    /** @brief Explicit retirement of one retained replica (refuses while live). */
    bool retireEffectiveUseReplica(std::uint64_t incarnation) noexcept
    {
        return effective_use_.retireReplica(incarnation);
    }
    /** @return Admitted bank count (witness class maximum). */
    std::size_t bankCount() const noexcept { return banks_.size(); }
    /** @return Admitted effective-use lease-cell count (witness class maximum). */
    std::size_t leaseCellCount() const noexcept { return effective_use_.leaseCapacity(); }
    /** @brief REAL live class demand (occupied banks and live lease cells). */
    struct LiveClassDemand
    {
        std::size_t occupied_banks = 0; ///< Every non-Free bank (Preparing + Published + readers).
        std::size_t live_leases = 0;    ///< Live effective-use lease cells right now.
    };
    /**
     * @return Observed live class demand; never a hard-coded per-request 1.
     * C4: Preparing banks (open updates/COW) hold REAL banks, so every
     * non-Free phase is counted — a witness must not admit a bank need that
     * exceeds what the table actually has free.
     */
    LiveClassDemand liveClassDemand() const noexcept
    {
        LiveClassDemand demand;
        for (const auto &bank : banks_)
            if (bank.phase != Phase::Free) ++demand.occupied_banks;
        demand.live_leases = effective_use_.pendingCount();
        return demand;
    }
private:
    enum class Phase : std::uint8_t { Free, Preparing, Published };
    /** @brief Exact CPU ownership state within the charged metadata root. */
    struct Bank
    {
        std::uint64_t version = 0;
        size_t pages = 0;
        size_t readers = 0;
        Phase phase = Phase::Free;
        bool accepting_readers = false;
    };
    CPUFP32PagedKVTable(std::shared_ptr<PhysicalMemoryAuthority> authority, DeviceId device,
        CPUFP32PagedKVTableGeometry geometry, CPUStoragePlacement placement);
    /** @brief Authenticate an original bank incarnation and lifecycle phase. */
    Bank &require(size_t bank, std::uint64_t version, Phase phase);
    /** @brief Retire page referents and free one original bank, never its physical storage. */
    void clear(size_t bank) noexcept;
    /** @brief Release a published bank only after its final borrower retires. */
    void releaseSnapshot(size_t bank, std::uint64_t version) noexcept;
    std::shared_ptr<PhysicalMemoryAuthority> authority_; ///< Original instance identity, not a separate budget.
    CPUFP32PagedKVTableGeometry geometry_;
    std::uint64_t next_version_ = 0; ///< Monotonic across every bank, including aborted preparations.
    std::uint64_t reserved_outstanding_ = 0; ///< Sum of live reservation capacities.
    std::uint64_t reservation_serial_ = 0;   ///< Monotonic handle nonce, never recycled.
    std::uint64_t effective_use_sequence_ = 0; ///< Fresh touch ordinal (§8.2).
    AlignedVector<std::byte> cells_;
    AlignedVector<Bank> banks_;
    AlignedVector<EffectiveUseJournal::LeaseCell> use_cells_; ///< PMA-admitted lease cells.
    AlignedVector<EffectiveUseJournal::ReplicaRecord> use_records_; ///< PMA-admitted replica records.
    EffectiveUseJournal effective_use_;       ///< Lease + replica-record journal (non-owning).
    CPUFP32KVPage *pages_ = nullptr; ///< Constructed stable metadata array, never KV payload.
};
}
