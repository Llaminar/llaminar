/**
 * @file PagedKVInstance.cpp
 * @brief One-admitted-instance construction and retirement implementation.
 */
#include "PagedKVInstance.h"
#include <limits>

namespace llaminar2
{
PagedKVInstance::PagedKVInstance(std::shared_ptr<PhysicalMemoryAuthority> authority,
    std::shared_ptr<PhysicalPagePool> pool,
    std::shared_ptr<CPUFP32PagedKVTable> table,
    std::unique_ptr<PagedKVArchiveStore> disk_store,
    PagedKVInstancePlan plan)
    : authority_(std::move(authority)), pool_(std::move(pool)),
      table_(std::move(table)), disk_store_(std::move(disk_store)),
      plan_(std::move(plan)) {}

std::optional<PagedKVInstance> PagedKVInstance::create(
    std::shared_ptr<PhysicalMemoryAuthority> authority,
    const PagedKVInstancePlan &plan, DeviceId device,
    CPUStoragePlacement placement, std::string *refusal)
{
    if (refusal) refusal->clear();
    if (!authority) throw std::invalid_argument("Paged KV instance requires its original PMA authority");
    if (!device.is_cpu()) throw std::invalid_argument("Paged KV instance requires a CPU device");
    // Complete plan validity first, before any construction, so a bad plan
    // never leaves a pool without its table (or vice versa).
    plan.pool_geometry.validate();
    plan.table_geometry.validate();
    // The plan's complete BOM must be EXACTLY what the factories charge;
    // a mismatched plan (stale/zero/hand-edited bytes) is never admitted.
    const auto exact_pool = PhysicalPagePool::physicalBytes(plan.pool_geometry, device, placement);
    const auto exact_table = CPUFP32PagedKVTable::physicalBytes(plan.table_geometry, device, placement);
    if (exact_table > std::numeric_limits<std::size_t>::max() - exact_pool)
        throw std::overflow_error("Paged KV instance exact BOM overflows");
    if (plan.total_bytes != exact_pool + exact_table)
        throw std::invalid_argument("Paged KV instance plan BOM disagrees with its exact factories");
    // Same-authority admission: the pool and the table are bought from ONE
    // PMA instance; the plan's BOM is the total physical charge expected.
    auto pool = authority->createCPUPagePool(device, plan.pool_geometry, placement);
    std::shared_ptr<CPUFP32PagedKVTable> table;
    try
    {
        table = CPUFP32PagedKVTable::create(authority, device, plan.table_geometry, placement);
    }
    catch (...)
    {
        pool.reset(); // Retire the pool sub-authority first; no partial instance.
        throw;
    }
    // The composed disk store is opened as part of THIS instance (one
    // validated route, one quota, one retirement unit); a store that cannot
    // open its validated route refuses the WHOLE instance — no pool/table
    // partial survives and no disk store outlives a refused admission.
    std::unique_ptr<PagedKVArchiveStore> disk_store;
    if (plan.disk_route.has_value())
    {
        try
        {
            disk_store = std::make_unique<PagedKVArchiveStore>(*plan.disk_route);
        }
        catch (const std::exception &e)
        {
            table.reset();
            pool.reset(); // Retire the pool sub-authority first; no partial.
            if (refusal)
                *refusal = std::string("composed disk route refused: ") + e.what();
            return std::nullopt;
        }
    }
    // Construct directly (optional::emplace uses is_constructible, which
    // cannot observe the private constructor); the public move constructor
    // carries the instance into the result.
    PagedKVInstance instance(std::move(authority), std::move(pool),
        std::move(table), std::move(disk_store), plan);
    return instance;
}

PagedKVInstance::~PagedKVInstance() { retire(); }
PagedKVInstance::PagedKVInstance(PagedKVInstance &&other) noexcept
    : authority_(std::move(other.authority_)), pool_(std::move(other.pool_)),
      table_(std::move(other.table_)), disk_store_(std::move(other.disk_store_)),
      plan_(std::move(other.plan_)) {}
PagedKVInstance &PagedKVInstance::operator=(PagedKVInstance &&other) noexcept
{
    if (this != &other)
    {
        retire();
        authority_ = std::move(other.authority_);
        pool_ = std::move(other.pool_);
        table_ = std::move(other.table_);
        disk_store_ = std::move(other.disk_store_);
        plan_ = std::move(other.plan_);
    }
    return *this;
}
void PagedKVInstance::retire() noexcept
{
    // Disk store first: it outlives nothing it does not own, and the instance
    // is the single retirement unit for the composed route (extent files are
    // only removed by explicit retire calls; the store destructor releases
    // its ledger only).
    disk_store_.reset();
    if (table_) { table_.reset(); } // Table first: its snapshot vomits pages back to the pool.
    if (pool_) { pool_.reset(); }
    authority_.reset();
}
} // namespace llaminar2
