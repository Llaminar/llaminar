/**
 * @file ROCmMoEGroupedPopulationPublication.h
 * @brief Typed writable lease for the sole device population-decision producer.
 *
 * Gate/up planning publishes into canonical arena storage before its consumers.
 * Down planning and projections retain read-only bindings to the same word.
 */
#pragma once
#include "ROCmMoEGroupedPopulationPolicy.h"
namespace llaminar2::rocm
{
/** @brief Non-null producer binding; copying it never reads the device decision. */
class MoEGroupedPopulationPublisher final
{
public:
    /** @param publication Canonical arena word retained through graph retirement.
     * @throws std::invalid_argument For a missing publication. */
    explicit MoEGroupedPopulationPublisher(uint32_t *publication) : publication_(publication)
    { if (!publication) throw std::invalid_argument("population publisher requires device storage"); }
    /** @return Immutable storage identity without observing device contents. */
    [[nodiscard]] constexpr uint32_t *address() const noexcept { return publication_; }
#if defined(__HIPCC__)
    /** @param family Valid decision computed from this replay's authoritative counts. */
    __device__ __forceinline__ void publish(MoEGroupedPopulationFamily family) const
    {
        if (family >= MoEGroupedPopulationFamily::Invalid) __builtin_trap();
        *publication_ = static_cast<uint32_t>(family);
    }
#endif
private:
    uint32_t *publication_;
};
static_assert(std::is_trivially_copyable_v<MoEGroupedPopulationPublisher>);
}
