/**
 * @file ImmutableRowTableAdmissionWire.h
 * @brief Versioned setup-only exchange of exact source/consumer/candidate BOM inputs.
 *
 * This transports immutable planning metadata through the existing startup
 * agreement. It contains no row payload, GPU execution state or runtime fallback
 * decision, and cannot grant materialization before the final PMA certificate.
 * Immutable sender frames are prepared inside readiness consensus, before peers
 * may enter metadata collectives; malformed local input cannot strand a peer.
 */
#pragma once
#include "ImmutableRowTablePlacementPlanner.h"
#include <cstdint>
#include <span>
#include <vector>
#include <utility>
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
/**
 * @brief Immutable sender bytes validated before distributed readiness succeeds.
 *
 * Preparation owns only metadata. Its exact live byte count, never vector
 * capacity, is the MPI contribution. The private constructor prevents callers
 * from supplying unvalidated or oversized bytes to the runner's gather helper.
 */
class ImmutableRowTableAdmissionSendFrame final
{
public:
    /**
     * @brief Serialize and authenticate local metadata inside its readiness phase.
     * @param requests Zero or one local canonical request.
     * @param sender_rank Actual communicator rank from the startup owner.
     * @param canonical_source Independent local source; required for provision frames.
     * @return Owned, immutable, exactly sized contribution ready for collective use.
     * @throws std::invalid_argument For malformed policy or foreign sender/source.
     * @throws std::overflow_error For an extent outside the MPI int range.
     */
    [[nodiscard]] static ImmutableRowTableAdmissionSendFrame prepare(
        std::span<const ImmutableRowTableAdmissionRequest> requests, int sender_rank,
        const ModelMemoryProfile *canonical_source = nullptr);

    /** @return Exact live wire bytes; allocation capacity is never communicated. */
    [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept { return bytes_; }
    /** @return Exact live contribution size, proven representable by prepare(). */
    [[nodiscard]] int byteCount() const noexcept { return static_cast<int>(bytes_.size()); }
    /** @return Sender identity already authenticated against the request frame. */
    [[nodiscard]] int senderRank() const noexcept { return sender_rank_; }

private:
    /** @brief Adopt only bytes minted by the checked factory above. */
    ImmutableRowTableAdmissionSendFrame(std::vector<std::uint8_t> bytes, int sender_rank)
        : bytes_(std::move(bytes)), sender_rank_(sender_rank) {}
    std::vector<std::uint8_t> bytes_; ///< Metadata backing; no tensor/physical ledger.
    int sender_rank_; ///< Authenticated communicator-local sender identity.
};

}
