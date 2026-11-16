/**
 * @file MoEGroupedOffsetSlots.h
 * @brief Duplicate-safe WideFloor8 descriptors inside original expert offsets.
 *
 * Cold experts publish row zero once; hot experts publish every native16 tile.
 * Original exclusive intervals guarantee unique slots, even when top-k input
 * repeats an expert more than sixteen times. Unwritten slots remain sentinels;
 * consumers must inspect each slot independently rather than stop at a hole.
 */
#pragma once
#include "MoEGroupedDirectory.h"
#include <stdexcept>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_OFFSET_HD __host__ __device__
#else
#define LLAMINAR_MOE_OFFSET_HD
#endif

namespace llaminar2::moe_grouped_directory
{
/** @brief One validated live expert interval in the physical grouping slot bank. */
class WideFloor8OffsetSlots final
{
public:
    /** @brief Validate a device-derived count and exclusive offset before publication.
     * @param count Original expert route count, including duplicate top-k slots.
     * @param offset Original exclusive expert offset.
     * @param total_slots Captured physical route capacity.
     * @return Exact interval with no host count mirror.
     * @throws std::invalid_argument On invalid host inputs; device inputs trap fatally. */
    [[nodiscard]] LLAMINAR_MOE_OFFSET_HD static constexpr WideFloor8OffsetSlots bind(
        int count,int offset,int total_slots)
    {
        if (count < 0 || offset < 0 || total_slots <= 0 ||
            static_cast<uint32_t>(total_slots) >= kMaximumRows ||
            offset > total_slots || count > total_slots-offset)
            invalid();
        return {count,offset};
    }
    /** @return Number of cold/native descriptors needed to cover every original route. */
    [[nodiscard]] LLAMINAR_MOE_OFFSET_HD constexpr int tiles() const noexcept
    {
        return count_ == 0 ? 0 : count_ < 8 ? 1 : (count_+15)/16;
    }
    /** @brief Resolve one tile's original expert row offset.
     * @param tile Zero-based descriptor ordinal. @return Exact row within this expert.
     * @throws std::invalid_argument On an invalid host ordinal; device ordinals trap. */
    [[nodiscard]] LLAMINAR_MOE_OFFSET_HD constexpr int firstRow(int tile) const
    {
        if (tile < 0 || tile >= tiles()) invalid();
        return tile*16;
    }
    /** @brief Resolve the unique physical word this expert owns for a tile.
     * @param tile Zero-based descriptor ordinal. @return Original offset plus tile row.
     * @throws std::invalid_argument On an invalid host ordinal; device ordinals trap. */
    [[nodiscard]] LLAMINAR_MOE_OFFSET_HD constexpr int slot(int tile) const { return offset_+firstRow(tile); }
    /** @brief Encode the common native descriptor without reinterpreting raw active IDs.
     * @param expert Original expert identity. @param experts Descriptor extent.
     * @param tile Zero-based descriptor ordinal. @return Packed first-row/expert word.
     * @throws std::invalid_argument On invalid host geometry; device geometry traps. */
    [[nodiscard]] LLAMINAR_MOE_OFFSET_HD constexpr uint32_t descriptor(int expert,int experts,int tile) const
    {
        if (expert < 0 || experts <= 0 || static_cast<uint32_t>(experts) > kMaximumExperts || expert >= experts) invalid();
        return (static_cast<uint32_t>(firstRow(tile)) << kExpertBits) | static_cast<uint32_t>(expert);
    }
private:
    /** @brief Only validated original intervals can be instantiated. */
    LLAMINAR_MOE_OFFSET_HD constexpr WideFloor8OffsetSlots(int count,int offset):count_(count),offset_(offset){}
    /** @brief Corrupt grouping metadata must fail before writing outside its original interval. */
    LLAMINAR_MOE_OFFSET_HD static void invalid()
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        __builtin_trap();
#else
        throw std::invalid_argument("WideFloor8 offset-slot directory has invalid original grouping geometry");
#endif
    }
    int count_,offset_;
};
}
#undef LLAMINAR_MOE_OFFSET_HD
