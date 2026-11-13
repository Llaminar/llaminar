/**
 * @file GDNRecurrenceStatePolicy.h
 * @brief Cold selection of the physical format of a persistent GDN state bank.
 *
 * A cache copies this policy at construction and retains it for its lifetime.
 * Selection changes neither arithmetic nor capacity, and never converts a live
 * bank. Backend-specific diagnostics permit matched comparisons without changing
 * another participant's format. CPU state retains its implemented key-major
 * layout. CUDA and ROCm select their independently qualified defaults. Linked
 * state exchange requires one matching format within its actual clique;
 * independent PP banks do not require a pipeline-wide physical format.
 * Each GPU implementation validates its selected format before capture.
 */
#pragma once

#include "backends/DeviceId.h"
#include "kernels/common/GDNRecurrenceStateLayout.h"

#include <initializer_list>
#include <stdexcept>
#include <string_view>

namespace llaminar2
{
/**
 * @brief Independently qualified backend defaults, copied before bank allocation.
 * @details Both canonical layouts remain explicit cold selections. A different
 * participant backend never rewrites another owner's policy or converts state.
 */
inline constexpr GDNRecurrenceStateLayout kDefaultCUDAGDNStateLayout =
    GDNRecurrenceStateLayout::KeyMajorValueContiguous;
inline constexpr GDNRecurrenceStateLayout kDefaultROCmGDNStateLayout =
    GDNRecurrenceStateLayout::ValueMajorKeyContiguous;

/**
 * @brief Parse exactly one canonical physical-format name.
 * @param text Complete setting, with no aliases or implicit whitespace trimming.
 * @return The requested immutable format.
 * @throws std::invalid_argument for an empty, unknown or malformed setting.
 */
[[nodiscard]] inline GDNRecurrenceStateLayout parseGDNRecurrenceStateLayout(
    std::string_view text)
{
    for (const auto layout : {GDNRecurrenceStateLayout::KeyMajorValueContiguous,
                              GDNRecurrenceStateLayout::ValueMajorKeyContiguous})
        if (text == gdnRecurrenceStateLayoutName(layout))
            return layout;
    throw std::invalid_argument(
        "GDN state layout must be key_major_value_contiguous or value_major_key_contiguous");
}

/**
 * @brief Resolve one participant's format before its cache is materialized.
 * @param device Actual participant device, independent of rank and TP degree.
 * @param cuda_layout Explicit CUDA policy already parsed by DebugEnv.
 * @param rocm_layout Explicit ROCm policy already parsed by DebugEnv.
 * @return CPU's key-major format or the policy for the participant's GPU backend.
 * @throws std::invalid_argument for an invalid device or unknown policy value.
 * @details This is a pure cold selection, not a capability probe. A selected
 * backend must implement the returned format or fail admission; no alternate
 * layout is selected on an execution failure.
 */
[[nodiscard]] inline GDNRecurrenceStateLayout gdnRecurrenceStateLayoutForDevice(
    DeviceId device, GDNRecurrenceStateLayout cuda_layout,
    GDNRecurrenceStateLayout rocm_layout)
{
    if (!validGDNRecurrenceStateLayout(cuda_layout) ||
        !validGDNRecurrenceStateLayout(rocm_layout))
        throw std::invalid_argument("Unknown GDN recurrence state layout policy");
    if (device.is_cpu()) return GDNRecurrenceStateLayout::KeyMajorValueContiguous;
    if (device.is_cuda()) return cuda_layout;
    if (device.is_rocm()) return rocm_layout;
    throw std::invalid_argument("GDN state layout selection requires a valid participant device");
}
} // namespace llaminar2
