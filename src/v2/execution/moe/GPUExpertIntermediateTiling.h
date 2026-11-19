/**
 * @file GPUExpertIntermediateTiling.h
 * @brief Immutable GPU routed-expert execution and source-partition policy.
 */
#pragma once

#include <cstdint>

namespace llaminar2
{
/** @brief Kernel eligibility derived from the original model and bound hardware inventory. */
enum class LocalGPUExpertKernelPolicy : std::uint8_t { Generic, QFNMI50TP4 };

/** @brief Preferred activation tile for the original source-native intermediate partition. */
enum class GPUExpertIntermediateTiling : std::uint8_t { Coarse128, BalancedNativeK32 };

/** @return The preferred tile. Source-native block alignment still governs each partition. */
constexpr int preferredIntermediateGrain(GPUExpertIntermediateTiling tiling) noexcept
{
    return tiling == GPUExpertIntermediateTiling::BalancedNativeK32 ? 32 : 128;
}
}
