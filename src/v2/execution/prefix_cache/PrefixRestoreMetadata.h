/**
 * @file PrefixRestoreMetadata.h
 * @brief Small value describing the archive sections and tiers selected for restore.
 *
 * Lookup deliberately omits payload pointers until the common frontier is
 * selected. Reporting therefore uses authenticated layout metadata, never
 * payload residency, scans or hashes. Composites merge every participant's
 * selected chain; the result owns no payload and performs no memory accounting.
 */
#pragma once

#include <cstdint>
#include <span>

namespace llaminar2
{
    struct PrefixBlockHandle;

    /** @brief Mergeable request diagnostics with a single-word collective encoding. */
    class PrefixRestoreMetadata final
    {
    public:
        /**
         * @brief Describe selected immutable records without touching their payloads.
         * @param blocks Exact clamped chain admitted for restoration on one participant.
         * @return Sections and source tiers represented by those records.
         */
        static PrefixRestoreMetadata fromBlocks(std::span<const PrefixBlockHandle> blocks);

        /** @brief Include another participant's selected sections and tiers. */
        void merge(PrefixRestoreMetadata other) noexcept { bits_ |= other.bits_; }
        /** @return Whether any participant restored shifted predictor state. */
        bool hasMTPState() const noexcept { return (bits_ & kMTP) != 0; }
        /** @return Whether any participant restored a recurrent checkpoint. */
        bool hasHybridState() const noexcept { return (bits_ & kHybrid) != 0; }
        /** @return Stable public tier name, including mixed participant/tier sources. */
        const char *storageTierName() const noexcept;
        /** @return Exactly one protocol word; no capacities or payload bytes are transmitted. */
        uint32_t wireBits() const noexcept { return bits_; }
        /**
         * @brief Validate a collective result before publishing diagnostics.
         * @param bits Union of authenticated participant protocol words.
         * @return Typed metadata containing only supported bits.
         * @throws std::invalid_argument for unknown bits or sections without a source tier.
         */
        static PrefixRestoreMetadata fromWireBits(uint32_t bits);

    private:
        static constexpr uint32_t kMTP = 1u;
        static constexpr uint32_t kHybrid = 2u;
        static constexpr uint32_t kDevice = 4u;
        static constexpr uint32_t kRam = 8u;
        static constexpr uint32_t kDisk = 16u;
        static constexpr uint32_t kTiers = kDevice | kRam | kDisk;
        uint32_t bits_ = 0;
    };
}
