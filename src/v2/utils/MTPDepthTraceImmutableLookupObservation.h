/**
 * @file MTPDepthTraceImmutableLookupObservation.h
 * @brief Persist the resolved immutable lookup interpretation of borrowed trace words.
 *
 * Resident depth replay owns none of these words. Labeling the observed
 * placement keeps raw receipts unambiguous while leaving its controller-only
 * arithmetic and the historical 108-word wire image intact.
 */
#pragma once

#include "loaders/ImmutableRowTableObservationMode.h"

#include <array>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace llaminar2 {

/** @brief Original and terminal resident snapshots carry the same admitted lookup interpretation. */
inline constexpr std::string_view kMTPDepthTraceImmutableLookupObservation = "immutable_lookup_observation_mode";

/**
 * @brief Spell the exact resolved layout in a resident depth trace.
 * @param mode Sealed placement's immutable terminal observation layout.
 * @return Stable external name shared by publication and replay admission.
 * @throws std::invalid_argument For an unknown enum value.
 */
inline std::string_view mtpDepthTraceImmutableLookupName(ImmutableRowTableObservationMode mode)
{
    using Mode = ImmutableRowTableObservationMode;
    switch (mode) {
    case Mode::None: return "none";
    case Mode::GPUAdaptive: return "gpu_adaptive";
    case Mode::FixedHeadStandalone: return "fixed_head_standalone";
    case Mode::FixedHeadPipeline: return "fixed_head_pipeline";
    }
    throw std::invalid_argument("Unknown immutable lookup layout in resident depth trace");
}

/**
 * @brief Parse a layout label before replay can accept an observed raw controller image.
 * @param label Exact layout published with the original trace.
 * @return Typed layout, or absence for malformed and unsupported labels.
 */
inline std::optional<ImmutableRowTableObservationMode> parseMTPDepthTraceImmutableLookupName(std::string_view label)
{
    using Mode = ImmutableRowTableObservationMode;
    constexpr std::array modes{Mode::None, Mode::GPUAdaptive, Mode::FixedHeadStandalone, Mode::FixedHeadPipeline};
    for (const auto mode : modes)
        if (mtpDepthTraceImmutableLookupName(mode) == label)
            return mode;
    return {};
}
}
