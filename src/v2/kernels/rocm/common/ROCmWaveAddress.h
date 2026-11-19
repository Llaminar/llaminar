/** @file ROCmWaveAddress.h
 * @brief Preserve both unsigned halves when recovering a wave-uniform address.
 */
#pragma once
#include <cstdint>

namespace llaminar2::rocm {
/** @brief Join two exact unsigned address words without sign extension.
 * @param low Bits 0 through 31, including the high bit of the low word.
 * @param high Bits 32 through 63, including the high bit of the full address.
 * @return The original 64-bit representation without interpreting it as a pointer.
 */
constexpr std::uint64_t composeWaveAddress(std::uint32_t low,std::uint32_t high) noexcept
{
    return std::uint64_t(low)|(std::uint64_t(high)<<32);
}

/** @brief Recover the exact 64-bit address from the first active lane.
 * @param address The same address in every active lane of this wave.
 * @return Both unsigned address words, unchanged.
 * @pre The caller has proved wave-wide uniformity. A lane-varying projection
 * would borrow the wrong address via this operation.
 */
__device__ __forceinline__ std::uint64_t waveUniformAddress(std::uint64_t address)
{
    // The intrinsic returns signed int. Widen only after restoring each half.
    const std::uint32_t low=__builtin_amdgcn_readfirstlane(static_cast<std::uint32_t>(address));
    const std::uint32_t high=__builtin_amdgcn_readfirstlane(static_cast<std::uint32_t>(address>>32));
    return composeWaveAddress(low,high);
}
} // namespace llaminar2::rocm
