/**
 * @file PeerPushReduction.h
 * @brief Immutable layout and participant binding for small lossless peer reductions.
 *
 * Each block owns a fixed range in two inbox slots and its own device epoch.
 * Retained graphs with different row counts therefore share storage without
 * changing which block protects a byte. Storage belongs to the collective's
 * physical owner; this value object neither allocates nor keeps an admission
 * ledger. Membership is a device table, never a fixed-size GPU array.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    /** @brief One original participant's uncached, peer-visible native storage. */
    struct PeerPushEndpoint
    {
        float *inbox = nullptr; ///< Two slots, each participant-major with fixed block ranges.
        std::uint64_t *flags = nullptr; ///< Cache-line-separated readiness epochs; self entries own progress.
    };

    /** @brief Checked, exact storage geometry independent of a particular model. */
    struct PeerPushLayout
    {
        static constexpr std::size_t threads = 256;
        static constexpr std::size_t elements_per_block = 4 * threads;
        static constexpr std::size_t flag_stride = 64 / sizeof(std::uint64_t);

        std::size_t bank_elements = 0; ///< Whole fixed-block bank extent, never the transfer extent.
        std::size_t blocks = 0; ///< Retained progress lanes; launch only the selected row geometry.
        int participants = 0; ///< Original communicator membership, with no engine count cap.

        /**
         * @brief Construct overflow-checked storage for an admitted payload range.
         * @param maximum_elements Largest selected FP32 operation, including partial final tiles.
         * @param members Original communicator size, at least two.
         * @param resident_blocks Smallest simultaneous block capacity across original devices.
         * @return Geometry whose whole grid can reside, avoiding inter-GPU scheduling deadlock.
         * @throws std::invalid_argument For absent membership/work or insufficient residency.
         * @throws std::overflow_error For an unrepresentable physical extent.
         */
        static PeerPushLayout create(std::size_t maximum_elements, int members,
                                     std::size_t resident_blocks)
        {
            if (!maximum_elements || members < 2 || !resident_blocks)
                throw std::invalid_argument("Peer push requires work, original members and resident block capacity");
            const auto n = (maximum_elements - 1) / elements_per_block + 1;
            if (n > resident_blocks)
                throw std::invalid_argument("Peer push grid exceeds simultaneous participant residency");
            constexpr auto limit = std::numeric_limits<std::size_t>::max();
            if (n > limit / elements_per_block / sizeof(float) / 2 / std::size_t(members))
                throw std::overflow_error("Peer push inbox extent overflow");
            PeerPushLayout result{n * elements_per_block, n, members};
            if (result.inboxBytes() > limit - result.flagBytes() - result.endpointBytes())
                throw std::overflow_error("Peer push total extent overflow");
            return result;
        }

        /** @return Exact local two-slot inbox capacity; only live values cross a peer edge. */
        [[nodiscard]] std::size_t inboxBytes() const noexcept
        { return 2 * std::size_t(participants) * bank_elements * sizeof(float); }
        /** @return Exact local flag capacity; one writer owns each source/block line. */
        [[nodiscard]] std::size_t flagBytes() const noexcept
        { return std::size_t(participants) * blocks * flag_stride * sizeof(std::uint64_t); }
        /** @return Exact device pointer-table bytes for the original membership. */
        [[nodiscard]] std::size_t endpointBytes() const noexcept
        { return std::size_t(participants) * sizeof(PeerPushEndpoint); }
        /** @return Smallest nonempty grid covering the exact captured operand. */
        [[nodiscard]] static constexpr std::size_t blocksFor(std::size_t elements) noexcept
        { return elements ? (elements - 1) / elements_per_block + 1 : 0; }
    };

    /** @brief Borrowed immutable graph identity; all pointed-to resources outlive its captures. */
    struct PeerPushBinding
    {
        const PeerPushEndpoint *endpoints = nullptr; ///< Device table in original communicator order.
        PeerPushLayout layout; ///< Exact admitted layout, common to all participants.
        const volatile std::uint32_t *abort = nullptr; ///< Borrowed original communicator abort observation.
        int rank = -1; ///< Original member index, never a physical GPU ordinal.
    };

    /**
     * @brief Query this kernel's cold simultaneous block capacity on the current GPU.
     * @param capacity Host result; not an execution-state observation.
     * @return False on runtime/property failure. Never called during capture or replay.
     */
    bool rocmPeerPushResidentCapacity(std::size_t &capacity);

    /**
     * @brief Enqueue one lossless FP32 direct-push collective on its exact stream.
     * @param binding Prepared original participant and storage identities.
     * @param input Local FP32 contribution; may alias output exactly.
     * @param output Result; inactive capacity remains unchanged.
     * @param live_rows Ordered device-owned equal row count, or null for fully active rows.
     * @param rows Exact retained transaction row geometry, not arena capacity.
     * @param columns FP32 values per row.
     * @param payload_bytes Optional device-only receipt of useful outgoing bytes.
     * @param metadata_bytes Optional device-only receipt of readiness writes.
     * @param stream Exact non-default producer/consumer stream.
     * @return Whether arguments and the launch were accepted; no alternative path on failure.
     */
    bool launchROCmPeerPushFP32(const PeerPushBinding &binding, const float *input, float *output,
        const std::int32_t *live_rows, std::size_t rows, std::size_t columns,
        unsigned long long *payload_bytes, unsigned long long *metadata_bytes, void *stream);
}
