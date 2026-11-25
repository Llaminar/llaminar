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
    /** @brief Freeze new borrows of an exact published version before joining its existing readers. */
    void closeBorrowers(const Snapshot &snapshot);
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
