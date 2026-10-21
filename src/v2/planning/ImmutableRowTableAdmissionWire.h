/**
 * @file ImmutableRowTableAdmissionWire.h
 * @brief Versioned setup-only exchange of exact source/consumer/candidate BOM inputs.
 *
 * This transports immutable planning metadata through the existing startup
 * agreement. It contains no row payload, GPU execution state or runtime fallback
 * decision, and cannot grant materialization before the final PMA certificate.
 */
#pragma once
#include "ImmutableRowTablePlacementPlanner.h"
#include <cstdint>
#include <span>
#include <vector>
namespace llaminar2
{
/**
 * @brief Encode one rank's optional PLE request without ABI padding.
 * @param requests Zero or one canonical request for that exact rank.
 * @return Checked versioned little-endian source/resource/BOM metadata.
 * @throws std::invalid_argument For duplicate rank requests or malformed policy.
 * @throws std::overflow_error For unrepresentable wire lengths.
 */
std::vector<std::uint8_t> serializeImmutableRowTableAdmissionRequests(
    std::span<const ImmutableRowTableAdmissionRequest> requests);
/**
 * @brief Decode and authenticate a startup contribution against its actual MPI sender.
 * @param bytes Exact one rank's framed wire segment.
 * @param expected_rank Authenticated communicator sender rank, never taken from the payload.
 * @param canonical_source Existing receiver-owned model metadata; mandatory for provision-bearing frames.
 * @return Checked exact original extent, resources and candidate plans.
 * @throws std::invalid_argument For changed version, truncation, trailing bytes or foreign ownership.
 * @throws std::overflow_error For nonrepresentable capacities.
 */
std::vector<ImmutableRowTableAdmissionRequest> parseImmutableRowTableAdmissionRequests(
    std::span<const std::uint8_t> bytes, int expected_rank,const ModelMemoryProfile *canonical_source = nullptr);
}
