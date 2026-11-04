/**
 * @file NativeContextQualifiedLimits.h
 * @brief Conservative selected-worker family allowances; actual identity belongs to the live owner.
 *
 * Entries require successful native creation/use/actual-reset evidence and
 * explicit conservative headroom. CUDA selected-worker birth also provisions
 * CUDANativeProgramStack::capacityBytes within this SAME opaque commitment;
 * it adds no separate physical-byte charge. Bootstrap/runtime-inventory populations are
 * separate. Current compiled stack/object geometry remains exactly owned.
 * Each row retains the evidence for its conservative rounded allowance.
 * The supported backend and native resource family define opaque eligibility;
 * runtime builds and current explicit object sizes do not require new allowance data.
 */
#pragma once
#include "NativeContextOpaqueContract.h"
#include <array>
#include <string_view>
namespace llaminar2::detail {
/** @brief Conservative per-original-worker opaque units; exact stack/object storage is separately owned. */
struct NativeContextQualifiedClass final {
    DeviceType backend;
    NativeContextBirthPopulation population;
    NativePhysicalLifetime physical_lifetime;
    std::size_t host_upper_bytes,device_upper_bytes;
    std::string_view evidence_sha256;
};
/** @brief Reviewed selected-worker limits; opaque backing remains process-owned on both backends. */
inline constexpr std::array kNativeContextQualifiedClasses{
    NativeContextQualifiedClass{
        .backend = DeviceType::CUDA,
        .population = NativeContextBirthPopulation::AdditionalContext,
        .physical_lifetime = NativePhysicalLifetime::ProcessRuntime,
        .host_upper_bytes = 268435456u,
        .device_upper_bytes = 402653184u,
        .evidence_sha256 = "b82fa882e6bb6b7a5175df363bd1f591175c8754000afbbd2c94ac943d7fd7ce"},
    NativeContextQualifiedClass{
        .backend = DeviceType::ROCm,
        .population = NativeContextBirthPopulation::AdditionalContext,
        .physical_lifetime = NativePhysicalLifetime::ProcessRuntime,
        .host_upper_bytes = 268435456u,
        .device_upper_bytes = 268435456u,
        .evidence_sha256 = "b82fa882e6bb6b7a5175df363bd1f591175c8754000afbbd2c94ac943d7fd7ce"},
};
}
