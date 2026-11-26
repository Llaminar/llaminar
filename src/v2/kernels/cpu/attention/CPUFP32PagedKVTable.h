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
#include "PagedKVArchive.h"
#include <memory>
#include <optional>

namespace llaminar2
{
namespace internal
{
    /**
     * @brief Finite nonce issuance admission: budget minus reserved headroom.
     *
     * plan P1: finite grants never sum independent maxima. A prepared update
     * monopolizes one issuance that stays burned through abort; normal
     * admission refuses before the reserved headroom can be consumed so
     * already-accepted work always completes. Reserved issuance (accepted
     * work draining) may consume the remaining headroom but never the next
     * normal update's protection, and no version is ever recycled.
     */
    struct IssuanceAdmission
    {
    public:
        explicit IssuanceAdmission(std::uint64_t budget, std::uint64_t headroom)
            : budget_(budget), headroom_(headroom)
        {
            if (!budget_ || headroom_ >= budget_) throw std::invalid_argument("Issuance budget must exceed its headroom");
            normal_cap_ = budget_ - headroom_;
        }
        /** @return Whether one ordinary begin is allowed. */
        bool allowsNormal(std::uint64_t issued) const noexcept { return issued < normal_cap_; }
        /** @return Whether one reserved begin (accepted work) is allowed. */
        bool allowsReserved(std::uint64_t issued) const noexcept { return issued < budget_; }
        std::uint64_t budget() const noexcept { return budget_; }
    private:
        std::uint64_t budget_ = 0;
        std::uint64_t headroom_ = 0;
        std::uint64_t normal_cap_ = 0;
    };
}

/** @brief Explicit admitted table metadata and concurrent snapshot envelope. */
struct CPUFP32PagedKVTableGeometry
{
    size_t page_capacity = 0;
    size_t bank_count = 0;
    size_t readers_per_bank = 0;
    int page_tokens = 0;
    /** @brief Finite nonce issuance budget (metadata). Zero or headroom>=budget is invalid. */
    std::uint64_t issuance_budget = 0;
    /** @brief Reserved issuance for already-accepted work/aborts/drain; must be < budget. */
    std::uint64_t issuance_headroom = 0;
    /** @brief Reject empty, overflowing or unrepresentable metadata geometry before allocation. */
    void validate() const;
};

/** @brief Single CPU runner-thread metadata authority; immutable page bytes may be consumed synchronously. */
class CPUFP32PagedKVTable final : public std::enable_shared_from_this<CPUFP32PagedKVTable>
{
public:
    /** @brief One immutable operation snapshot retaining actual page referents, not just a header. */
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
     * @brief Begin under the reserved headroom for already-accepted work.
     *
     * The ordinary admission cap is budget minus headroom; an accepted update
     * that still needs to complete may consume the reserved headroom even
     * when normal admission has stopped. It still never exceeds the finite
     * budget and never exhausts it below a nonnegative remainder.
     */
    std::optional<Update> tryBeginUpdateReserved();
    /** @return Finite admitted issuance budget (never an independent maximum). */
    std::uint64_t issuanceBudget() const noexcept { return geometry_.issuance_budget; }
    /** @return Total issued versions, burned and live, monotonic across banks. */
    std::uint64_t issuedVersions() const noexcept { return next_version_; }
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
    /** @brief Freeze new borrows of an exact published version before joining its existing readers. */
    void closeBorrowers(const Snapshot &snapshot);
    /** @return Original PMA instance identity for same-authority admission checks. */
    const PhysicalMemoryAuthority *issuerForAdmission() const noexcept { return authority_.get(); }
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
    AlignedVector<std::byte> cells_;
    AlignedVector<Bank> banks_;
    CPUFP32KVPage *pages_ = nullptr; ///< Constructed stable metadata array, never KV payload.
};
}
