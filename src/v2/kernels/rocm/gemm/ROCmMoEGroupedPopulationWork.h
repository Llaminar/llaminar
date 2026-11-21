/**
 * @file ROCmMoEGroupedPopulationWork.h
 * @brief Retained population worker geometry and shared directory publication layout.
 *
 * Capacity bounds storage; a trailing device word publishes this replay's live
 * tile extent. Both roles reuse the same canonical work bank in stream order.
 * The odd worker stride rotates cold column ownership across iterations instead
 * of permanently stranding three quarters of workers on unused native columns.
 */
#pragma once
#include <cstddef>
#include <stdexcept>
#if defined(__HIPCC__)
#define LLAMINAR_POPULATION_WORK_HD __host__ __device__
#else
#define LLAMINAR_POPULATION_WORK_HD
#endif
namespace llaminar2::rocm
{
/** Fixed retained launch width; live work remains exclusively device-owned. */
inline constexpr unsigned kMoEGroupedPopulationWorkers = 2047;
/** @brief Shared layout for compact tile entries followed by one live-count word. */
class MoEGroupedPopulationDirectoryLayout final
{
public:
    /** @param capacity Conservative tile capacity excluding the count word.
     * @throws std::invalid_argument For nonpositive capacity on the host; device misuse traps. */
    LLAMINAR_POPULATION_WORK_HD explicit MoEGroupedPopulationDirectoryLayout(int capacity) : capacity_(capacity)
    {
        if (capacity <= 0)
        {
#if defined(__HIP_DEVICE_COMPILE__)
            __builtin_trap();
#else
            throw std::invalid_argument("population directory requires positive tile capacity");
#endif
        }
    }
    /** @return Offset of the sole live-count publication, outside tile payload. */
    [[nodiscard]] LLAMINAR_POPULATION_WORK_HD constexpr int liveCountIndex() const noexcept { return capacity_; }
    /** @return Physical words required in the canonical borrowed arena bank. */
    [[nodiscard]] LLAMINAR_POPULATION_WORK_HD constexpr std::size_t words() const noexcept
    { return static_cast<std::size_t>(capacity_) + 1; }
private:
    int capacity_;
};
}
#undef LLAMINAR_POPULATION_WORK_HD
