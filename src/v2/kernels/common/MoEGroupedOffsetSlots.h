/**
 * @file MoEGroupedOffsetSlots.h
 * @brief Duplicate-safe row and column work inside original expert offsets.
 *
 * Cold experts publish row zero once; hot experts publish every native16 tile.
 * Original exclusive intervals guarantee unique slots, even when top-k input
 * repeats an expert more than sixteen times. Unwritten slots remain sentinels;
 * consumers must inspect each slot independently rather than stop at a hole.
 * The bounded column layout uses compact ordinals for four native quadrants,
 * while its cold GU4/down12 roles retain their original ordered arithmetic.
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
/** @brief Compact column work inside each original expert interval.
 * Cold experts emit four-row GU tiles, with only row zero consumed by down.
 * Native experts emit four column quadrants for each sixteen-row tile.
 * Compact ordinals, rather than first-row offsets, keep count17 inside17 slots.
 * The original count/offset banks remain the only live ownership authority. */
class ColumnFloor8OffsetSlots final
{
public:
    /** @brief Authenticate one expert interval before publishing or consuming it.
     * @param count Original expert count, including duplicate routes.
     * @param offset Original exclusive offset. @param total_slots Physical route span.
     * @return Validated interval. @throws std::invalid_argument For invalid host metadata. */
    [[nodiscard]] LLAMINAR_MOE_OFFSET_HD static constexpr ColumnFloor8OffsetSlots bind(
        int count,int offset,int total_slots)
    {
        if (count < 0 || offset < 0 || total_slots <= 0 || total_slots > 128 ||
            offset > total_slots || count > total_slots-offset) invalid();
        return {count,offset};
    }
    /** @return Compact words needed for all cold row tiles or native column quadrants. */
    [[nodiscard]] LLAMINAR_MOE_OFFSET_HD constexpr int tiles() const noexcept
    { return count_ < 8 ? (count_+3)/4 : 4*((count_+15)/16); }
    /** @param ordinal Compact word ordinal. @return Original expert-relative row. */
    [[nodiscard]] LLAMINAR_MOE_OFFSET_HD constexpr int firstRow(int ordinal) const
    { check(ordinal); return count_ < 8 ? 4*ordinal : 16*(ordinal/4); }
    /** @param ordinal Compact word ordinal. @return Native column quadrant, or zero for cold work. */
    [[nodiscard]] LLAMINAR_MOE_OFFSET_HD constexpr int quadrant(int ordinal) const
    { check(ordinal); return count_ < 8 ? 0 : ordinal%4; }
    /** @param ordinal Compact word ordinal. @return Sole physical slot owning this word. */
    [[nodiscard]] LLAMINAR_MOE_OFFSET_HD constexpr int slot(int ordinal) const
    { check(ordinal); return offset_+ordinal; }
    /** @param expert Original expert ID. @param experts Exact bounded descriptor domain.
     * @param ordinal Compact word ordinal. @return Packed expert/column/row work. */
    [[nodiscard]] LLAMINAR_MOE_OFFSET_HD constexpr uint32_t descriptor(int expert,int experts,int ordinal) const
    {
        if (experts != 256 || expert < 0 || expert >= experts) invalid();
        return uint32_t(expert) | (uint32_t(quadrant(ordinal))<<8) | (uint32_t(firstRow(ordinal))<<10);
    }
    /** @brief Prove a consumed word is exactly the one owned by this original slot.
     * This rejects duplicate, shifted and malformed entries before an opposite-role
     * return could hide corruption. Sentinels are handled by the caller.
     * @param word Packed word. @param expert Decoded expert ID. @param slot_index Actual directory index.
     * @return Whether this interval uniquely owns the complete word. */
    [[nodiscard]] LLAMINAR_MOE_OFFSET_HD constexpr bool owns(uint32_t word,int expert,int slot_index) const
    {
        if(slot_index<offset_)return false;
        const int ordinal=slot_index-offset_;
        return ordinal<tiles() && descriptor(expert,256,ordinal)==word;
    }
private:
    /** @brief Construction follows interval validation. */
    LLAMINAR_MOE_OFFSET_HD constexpr ColumnFloor8OffsetSlots(int count,int offset):count_(count),offset_(offset){}
    /** @brief Reject an ordinal before any slot arithmetic can escape its interval. */
    LLAMINAR_MOE_OFFSET_HD constexpr void check(int ordinal) const
    { if(ordinal<0 || ordinal>=tiles())invalid(); }
    /** @brief Fail corrupt geometry on its owning execution domain. */
    LLAMINAR_MOE_OFFSET_HD static void invalid()
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        __builtin_trap();
#else
        throw std::invalid_argument("column Floor8 directory requires original bounded expert intervals");
#endif
    }
    int count_,offset_;
};

}
#undef LLAMINAR_MOE_OFFSET_HD
