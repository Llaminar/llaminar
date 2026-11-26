/**
 * @file PagedKVInstance.h
 * @brief ONE admitted pooled-KV instance: pool + table under one PMA authority.
 *
 * Planned candidates from `PagedKVInstancePlan` become a live instance only
 * through this constructor, which acquires the pool sub-authority and the
 * table from the SAME PMA instance, and verifies the physical bytes actually
 * charged equal the planned complete BOM. No independent constructor exists:
 * an instance cannot be assembled from a foreign pool/table pair, and the
 * authority remains the sole physical admission ledger.
 *
 * Destruction retires the table and the pool in dependency order; the
 * instance is the unit of retirement (no orphan table without its pool).
 */
#pragma once
#include "kernels/cpu/attention/CPUFP32PagedKVTable.h"
#include "planning/PagedKVInstancePlan.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <memory>

namespace llaminar2
{
class PagedKVInstance final
{
public:
    PagedKVInstance() = default;
    ~PagedKVInstance();
    PagedKVInstance(PagedKVInstance &&other) noexcept;
    PagedKVInstance &operator=(PagedKVInstance &&other) noexcept;
    PagedKVInstance(const PagedKVInstance &) = delete;
    PagedKVInstance &operator=(const PagedKVInstance &) = delete;

    /**
     * @brief Construct one admitted instance from an exact planned candidate.
     * @param authority Original PMA instance (sole physical admission owner).
     * @param plan Complete pool/table candidate from the planner.
     * @param device Original CPU allocator identity.
     * @param placement Explicit first-touch policy.
     * @return The live instance, or empty when the authority cannot admit the
     * complete planned BOM (refusal is total: no pool/table partial exists).
     * @throws std::invalid_argument For a foreign authority, device mismatch,
     * or a planned geometry that disagrees with the pool/table factories.
     */
    static std::optional<PagedKVInstance> create(
        std::shared_ptr<PhysicalMemoryAuthority> authority,
        const PagedKVInstancePlan &plan, DeviceId device,
        CPUStoragePlacement placement = CPUStoragePlacement::local());

    /** @return The pool sub-authority (same PMA instance). */
    PhysicalPagePool &pool() const { return *pool_; }
    /** @return The table owner (same PMA instance). */
    CPUFP32PagedKVTable &table() const { return *table_; }
    /** @return Exact admitted complete BOM (pool + table metadata). */
    std::size_t admittedBytes() const noexcept { return plan_.total_bytes; }
    /** @return The PMA instance this instance was admitted from. */
    const std::shared_ptr<PhysicalMemoryAuthority> &authority() const noexcept
    {
        return authority_;
    }
    explicit operator bool() const noexcept { return pool_ && table_; }

private:
    PagedKVInstance(std::shared_ptr<PhysicalMemoryAuthority> authority,
        std::shared_ptr<PhysicalPagePool> pool,
        std::shared_ptr<CPUFP32PagedKVTable> table,
        PagedKVInstancePlan plan);
    void retire() noexcept;
    std::shared_ptr<PhysicalMemoryAuthority> authority_;
    std::shared_ptr<PhysicalPagePool> pool_;
    std::shared_ptr<CPUFP32PagedKVTable> table_;
    PagedKVInstancePlan plan_;
};
} // namespace llaminar2
