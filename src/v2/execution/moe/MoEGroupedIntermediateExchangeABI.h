/**
 * @file MoEGroupedIntermediateExchangeABI.h
 * @brief Lossless device packet boundary between expert activation and down GEMM.
 *
 * The packet copies the representation already consumed by down projection:
 * block-Q8 plus FP32 scales for quantized experts, or FP32 for floating experts.
 * It never quantizes a new boundary. Fixed native-collective packets are indexed
 * by original router slot. Device-counted packets instead contain contiguous
 * producer-grouped records carrying that original slot ID. Both import into
 * the consumer's own expert-major grouping. The native collective or captured
 * TransferEngine message owns publication; this ABI adds no host-owned epochs.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace llaminar2
{
    /** @brief Existing expert-activation encodings; never a requested conversion. */
    enum class MoEGroupedIntermediateEncoding : std::uint32_t
    {
        BlockQ8FP32Scales = 1,
        FP32 = 2,
    };

    /** @brief Capture-time packet geometry shared by every participant. */
    struct MoEGroupedIntermediateLayout
    {
        MoEGroupedIntermediateEncoding encoding = MoEGroupedIntermediateEncoding::BlockQ8FP32Scales;
        std::uint32_t columns = 0;
        std::uint32_t route_capacity = 0;
        std::uint32_t participants = 0;

        /** @return Existing activation uint32 words, excluding scale planes. */
        [[nodiscard]] constexpr std::size_t valueWords() const noexcept
        {
            return encoding == MoEGroupedIntermediateEncoding::FP32 ? columns : columns / 4;
        }

        /** @return Number of existing FP32 block scales; floating rows need none. */
        [[nodiscard]] constexpr std::size_t scaleWords() const noexcept
        {
            return encoding == MoEGroupedIntermediateEncoding::FP32 ? 0 : columns / 32;
        }

        /** @return All uint32 words for one original route's immutable payload. */
        [[nodiscard]] constexpr std::size_t routeWords() const noexcept { return valueWords() + scaleWords(); }

        /** @return Fixed native-allgather byte count for one participant. */
        [[nodiscard]] constexpr std::size_t packetBytes() const noexcept
        {
            return static_cast<std::size_t>(route_capacity) * routeWords() * sizeof(std::uint32_t);
        }

        /** @return Compact record width: original route ID followed by unchanged payload bits. */
        [[nodiscard]] constexpr std::size_t compactRecordWords() const noexcept { return 1 + routeWords(); }

        /** @return Positive worst-case compact storage, not the GPU-selected transfer extent. */
        [[nodiscard]] constexpr std::size_t compactCapacityBytes() const noexcept
        { return static_cast<std::size_t>(route_capacity) * compactRecordWords() * sizeof(std::uint32_t); }

        /** @return Whether adding route IDs keeps all compact participant banks representable. */
        [[nodiscard]] constexpr bool compactValid() const noexcept
        {
            return valid() && compactRecordWords() <= std::numeric_limits<std::size_t>::max() /
                sizeof(std::uint32_t) / route_capacity / participants;
        }

        /** @return Whether geometry and the complete receive bank are representable. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return (encoding == MoEGroupedIntermediateEncoding::FP32 ||
                    encoding == MoEGroupedIntermediateEncoding::BlockQ8FP32Scales) &&
                columns > 0 &&
                (encoding == MoEGroupedIntermediateEncoding::FP32 || columns % 32 == 0) &&
                route_capacity > 0 &&
                route_capacity <= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) &&
                participants > 0 && participants <= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) &&
                routeWords() <= std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t) /
                    route_capacity / participants;
        }
    };

    /** @brief Pack only routes belonging to one authoritative participant. */
    struct MoEGroupedIntermediatePackLaunch
    {
        MoEGroupedIntermediateLayout layout;
        const std::int32_t *route_owners = nullptr; ///< Local domain participant IDs; -1 means inactive.
        const std::int32_t *original_to_grouped = nullptr; ///< Local producer grouping; -1 for unowned rows.
        const void *grouped_values = nullptr; ///< Existing Q8 or FP32 values, not a converted mirror.
        const float *grouped_scales = nullptr; ///< Existing FP32 scales, absent for FP32 values.
        std::uint32_t *packet = nullptr; ///< One fixed-capacity participant packet.
        std::int32_t participant = -1;

        /** @return Whether all host-verifiable capture bindings are complete. */
        [[nodiscard]] bool valid() const noexcept
        {
            return layout.valid() && route_owners && original_to_grouped && grouped_values && packet &&
                participant >= 0 && static_cast<std::uint32_t>(participant) < layout.participants &&
                ((layout.encoding == MoEGroupedIntermediateEncoding::FP32 && !grouped_scales) ||
                 (layout.encoding == MoEGroupedIntermediateEncoding::BlockQ8FP32Scales && grouped_scales));
        }
    };

    /** @brief Materialize every active route from its sole published owner. */
    struct MoEGroupedIntermediateConsumeLaunch
    {
        MoEGroupedIntermediateLayout layout;
        const std::int32_t *route_owners = nullptr; ///< Same immutable routing transaction used by packers.
        const std::int32_t *original_to_grouped = nullptr; ///< Consumer's complete expert-major grouping.
        const std::uint32_t *participant_packets = nullptr; ///< Native allgather's participant-major bank.
        void *grouped_values = nullptr;
        float *grouped_scales = nullptr;

        /** @return Whether all host-verifiable capture bindings are complete. */
        [[nodiscard]] bool valid() const noexcept
        {
            return layout.valid() && route_owners && original_to_grouped && participant_packets && grouped_values &&
                ((layout.encoding == MoEGroupedIntermediateEncoding::FP32 && !grouped_scales) ||
                 (layout.encoding == MoEGroupedIntermediateEncoding::BlockQ8FP32Scales && grouped_scales));
        }
    };

    /**
     * @brief Compact existing dense expert-major rows, without a second scan or atomics.
     *
     * Grouping already assigns each owned route a unique contiguous row. The
     * final expert's exclusive offset plus count is its total live extent.
     * Pack writes that count in bytes and one original-route ID per live row;
     * it never writes or clears unowned/inactive payload capacity.
     */
    struct MoECompactIntermediatePackLaunch
    {
        MoEGroupedIntermediatePackLaunch payload;
        const std::int32_t *last_group_offset = nullptr;
        const std::int32_t *last_group_count = nullptr;
        std::uint64_t *packet_bytes = nullptr; ///< GPU-authored TransferEngine count, distinct from packet capacity.

        /** @return Whether the complete capture-time compact contract is bound. */
        [[nodiscard]] bool valid() const noexcept
        {
            return payload.valid() && payload.layout.compactValid() && last_group_offset && last_group_count && packet_bytes;
        }
    };

    /** @brief Consume one owner's compact packet into the complete consumer grouping. */
    struct MoECompactIntermediateConsumeLaunch
    {
        MoEGroupedIntermediateLayout layout;
        const std::int32_t *route_owners = nullptr;
        const std::int32_t *original_to_grouped = nullptr;
        const std::uint32_t *packet = nullptr;
        const std::uint64_t *packet_bytes = nullptr; ///< Acquired producer extent, not independently recomputed.
        std::int32_t participant = -1;
        void *grouped_values = nullptr;
        float *grouped_scales = nullptr;

        /** @return Whether one exact producer and representation are capture-bound. */
        [[nodiscard]] bool valid() const noexcept
        {
            return layout.compactValid() && route_owners && original_to_grouped && packet && packet_bytes &&
                participant >= 0 && static_cast<std::uint32_t>(participant) < layout.participants && grouped_values &&
                ((layout.encoding == MoEGroupedIntermediateEncoding::FP32 && !grouped_scales) ||
                 (layout.encoding == MoEGroupedIntermediateEncoding::BlockQ8FP32Scales && grouped_scales));
        }
    };

    static_assert(std::is_trivially_copyable_v<MoEGroupedIntermediatePackLaunch>);
    static_assert(std::is_trivially_copyable_v<MoEGroupedIntermediateConsumeLaunch>);
    static_assert(std::is_trivially_copyable_v<MoECompactIntermediatePackLaunch>);
    static_assert(std::is_trivially_copyable_v<MoECompactIntermediateConsumeLaunch>);
} // namespace llaminar2
