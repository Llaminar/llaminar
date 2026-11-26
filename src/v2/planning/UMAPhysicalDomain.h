/**
 * @file UMAPhysicalDomain.h
 * @brief Domain-aware single-charge ledger for shared CPU/GPU physical memory.
 *
 * Design §4.4: shared CPU/GPU visibility does not create two independent
 * physical capacities. On UMA, resident GPU and CPU allocations compete in one
 * canonical PMA domain; distinct allocations each cost bytes, but device
 * mappings/aliases of the SAME allocation never duplicate its physical charge.
 * Bounded host I/O staging is a distinct resource: turning RAM reservation off
 * does not remove its charge.
 *
 * This is the typed rule consumed by the pool/tier reservation contract; the
 * physical bytes themselves remain admitted through PhysicalMemoryAuthority.
 * Observed domain identity comes from the hardware inventory (real UMA versus
 * discrete is therefore a device-dependent qualification, not a unit claim).
 */
#pragma once
#include <cstddef>
#include <cstdint>

namespace llaminar2
{
/** @brief One physical allocation identity plus its observed device aliases. */
struct UMAAllocationIdentity
{
    std::uint64_t allocation_id = 0;
    std::uint64_t domain_id = 0;
    std::size_t bytes = 0;
    bool operator==(const UMAAllocationIdentity &) const = default;
};

/** @brief Bounded single-charge UMA domain ledger. */
class UMAPhysicalDomainLedger final
{
public:
    /**
     * @brief Charge one allocation in the domain; aliases never double-charge.
     * @return True when this charge was newly applied (bytes added), false for
     *         a duplicate alias of an already-charged allocation.
     */
    bool charge(UMAAllocationIdentity allocation);
    /** @brief Release the exact allocation (aliases leave via the same charge). */
    bool release(UMAAllocationIdentity allocation);
    /** @return Total single-charged physical bytes in the domain. */
    std::size_t chargedBytes() const noexcept { return charged_; }
    /** @return Count of distinct charged allocations. */
    std::size_t allocationCount() const noexcept { return count_; }
    static constexpr std::size_t capacity() noexcept { return kCapacity; }
private:
    static constexpr std::size_t kCapacity = 256;
    UMAAllocationIdentity entries_[kCapacity] = {};
    std::size_t count_ = 0;
    std::size_t charged_ = 0;
};

/** @brief Bounded I/O staging charge: RAM reservation off never removes it. */
class UMAStagingCharge final
{
public:
    explicit UMAStagingCharge(std::size_t max_bytes) : max_(max_bytes) {}
    bool reserve(std::size_t bytes);
    bool release(std::size_t bytes);
    std::size_t staged() const noexcept { return staged_; }
private:
    std::size_t max_;
    std::size_t staged_ = 0;
};
} // namespace llaminar2
