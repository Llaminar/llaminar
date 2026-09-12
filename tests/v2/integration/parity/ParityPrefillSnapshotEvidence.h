/**
 * @file ParityPrefillSnapshotEvidence.h
 * @brief Request-scoped proof from cumulative segmented-prefill diagnostics.
 *
 * PerfStats counters coalesce requests sharing the same tags. Training and
 * prefix-cache seeding therefore remain in their lifetime totals. Immutable
 * observations around the authenticated prefill prove exactly one new prompt
 * without clearing telemetry, changing capture, or labelling every request.
 */
#pragma once

#include "utils/PerfStatsCollector.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace llaminar2::test::parity
{
    /** @brief Validated cumulative snapshot publications for one expected geometry. */
    class ParityPrefillSnapshotEvidence
    {
    public:
        /**
         * @brief Validate every aggregation record and retain bounded totals.
         * @param records Immutable collector snapshot, including unrelated metrics.
         * @param expected_chunks Chunk count of the authenticated parity prompt.
         * @return Observation usable at either boundary of that request.
         * @throws std::logic_error For malformed records, geometry, or overflow.
         */
        static ParityPrefillSnapshotEvidence capture(
            std::span<const PerfStatRecord> records,
            uint64_t expected_chunks)
        {
            if (expected_chunks == 0u)
                throw std::logic_error("Prefill evidence requires positive chunk geometry");
            ParityPrefillSnapshotEvidence result;
            result.expected_chunks_ = expected_chunks;
            for (const auto &record : records)
            {
                if (record.domain != "forward_graph" ||
                    record.name != "prefill_chunk_snapshot_sequence_keys")
                    continue;

                const auto chunks_tag = record.tags.find("chunks");
                const auto diagnostic_tag = record.tags.find("diagnostic_only");
                if (record.kind != PerfStatRecord::Kind::Counter ||
                    record.phase != "prefill" || record.count == 0u ||
                    !std::isfinite(record.value) || record.value <= 0.0 ||
                    chunks_tag == record.tags.end() ||
                    diagnostic_tag == record.tags.end() ||
                    diagnostic_tag->second != "true")
                    throw std::logic_error("Malformed segmented-prefill snapshot record");

                uint64_t chunks = 0;
                const auto &text = chunks_tag->second;
                const auto parsed = std::from_chars(
                    text.data(), text.data() + text.size(), chunks);
                if (parsed.ec != std::errc{} ||
                    parsed.ptr != text.data() + text.size() || chunks == 0u)
                    throw std::logic_error("Malformed segmented-prefill snapshot chunk count");
                if (record.count > std::numeric_limits<uint64_t>::max() - result.total_)
                    throw std::logic_error("Segmented-prefill snapshot count overflow");
                result.total_ += record.count;
                if (chunks == expected_chunks)
                    result.matching_ += record.count;
            }
            return result;
        }

        /**
         * @brief Require exactly one new aggregation with the admitted geometry.
         * @param before Observation immediately before the authenticated prefill.
         * @throws std::logic_error For reset, missing/extra work, or wrong geometry.
         */
        void requireSingleRequestSince(const ParityPrefillSnapshotEvidence &before) const
        {
            // Validate monotonicity before subtraction: unsigned wraparound
            // must not turn a collector reset into a successful request delta.
            if (expected_chunks_ != before.expected_chunks_ ||
                total_ < before.total_ || matching_ < before.matching_ ||
                total_ - before.total_ != 1u ||
                matching_ - before.matching_ != 1u)
                throw std::logic_error(
                    "Authenticated prefill must publish exactly one snapshot aggregation with its chunk geometry");
        }

        /** @return Number of validated publications, not number of coalesced rows. */
        uint64_t transactions() const noexcept { return total_; }

    private:
        uint64_t expected_chunks_ = 0; ///< Geometry identity shared by both observations.
        uint64_t total_ = 0; ///< All requests, including training and cache seeds.
        uint64_t matching_ = 0; ///< Requests with the authenticated chunk count.
    };
}
