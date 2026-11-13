/**
 * @file NativeCPUCollectiveProgram.h
 * @brief Inert external pool types for actual MPI session entry replay.
 *
 * Entrypoints create no CPU domain. Any pool reservation is an unexpected
 * native resource admission and aborts; this fixture never models live bytes.
 */
#pragma once
#include <cstdlib>
#include <cstddef>
namespace llaminar2 {
struct PhysicalMemoryOwnerReservation {};
/** @brief Inert CPU coordinate; no allocator is ever admitted. */
struct DeviceId { static DeviceId cpu() { return {}; } };
enum class PhysicalMemoryOwner { NativeCPUCollectiveProcessPool };
class PhysicalMemoryAuthority {
public:
    /** @brief Reject unexpected resource admission; the entry creates no domains.
     * @return Never returns; every call violates the replay's isolation.
     */
    PhysicalMemoryOwnerReservation reserveNewAllocations(DeviceId, PhysicalMemoryOwner, std::size_t) { std::abort(); }
    /** @brief Reject any unexpected process-pool borrow in the entry body. */
    void borrowCPUCollectiveProcessPool(const PhysicalMemoryOwnerReservation &) { std::abort(); }
};
class NativeCPUCollectiveProgram {
public:
    /** @brief Reject unexpected resource sizing; no native pool is exercised.
     * @return Never returns; a domain declaration is outside this replay.
     */
    static std::size_t processPoolBytes() { std::abort(); }
};
}
