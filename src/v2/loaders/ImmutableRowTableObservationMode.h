/**
 * @file ImmutableRowTableObservationMode.h
 * @brief Immutable resolved interpretation of original generation lookup observations.
 *
 * A sealed placement grants one projection. Controller replay still owns only
 * its own fields and never interprets any lookup-specific observation values.
 */
#pragma once
namespace llaminar2 {
/** @brief Mutually exclusive interpretations of the same original observation interval. */
enum class ImmutableRowTableObservationMode { None, GPUAdaptive, FixedHeadStandalone, FixedHeadPipeline };
} // namespace llaminar2
