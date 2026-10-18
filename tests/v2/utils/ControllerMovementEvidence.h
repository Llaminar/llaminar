/**
 * @file ControllerMovementEvidence.h
 * @brief Independent sequence replay for device-controller parity evidence.
 *
 * The fixture rebuilds the collector's ordered metadata witness from the exact
 * typed terminal history, without touching the global collector. It rejects
 * missing interior waves even when endpoint epochs and aggregate values match.
 */
#pragma once
#include "execution/moe/MoEControllerMovementPerfStats.h"
#include <algorithm>
#include <array>
#include <limits>
#include <string>

namespace llaminar2::test
{
    /** @brief Independent, device-free reference for the public metadata wire encoding. */
    struct MovementSequenceEvidence
    {
        PerfStatRecord record;

        /**
         * @brief Append one ordered counter observation without accessing PerfStats state.
         * @param value Actual counter increment from the immutable completed wave.
         * @param words Lossless, constant-width semantic metadata for this family.
         */
        void append(double value, std::initializer_list<std::uint64_t> words)
        {
            if (!record.count)
            {
                record.sequence_digest_lo = 14695981039346656037ull;
                record.sequence_digest_hi = 7809847782465536322ull;
                record.sequence_minimum_words.assign(words.begin(), words.end());
                record.sequence_maximum_words.assign(words.begin(), words.end());
            }
            /** Fold explicit little-endian bytes rather than native struct storage. */
            const auto mix = [](std::uint64_t &lane, std::uint64_t word, std::uint64_t prime)
            {
                for (unsigned shift = 0; shift < 64; shift += 8)
                    lane = (lane ^ ((word >> shift) & 255u)) * prime;
            };
            mix(record.sequence_digest_lo, 0x6c6c616d696e6172ull ^ words.size(), 1099511628211ull);
            mix(record.sequence_digest_hi, ~0x6c6c616d696e6172ull ^ words.size(), 14029467366897019727ull);
            std::size_t index = 0;
            for (auto word : words)
            {
                mix(record.sequence_digest_lo, word, 1099511628211ull);
                mix(record.sequence_digest_hi, ~word, 14029467366897019727ull);
                record.sequence_minimum_words[index] = std::min(record.sequence_minimum_words[index], word);
                record.sequence_maximum_words[index] = std::max(record.sequence_maximum_words[index], word);
                ++index;
            }
            mix(record.sequence_digest_lo, 0x73657175656e6365ull, 1099511628211ull);
            mix(record.sequence_digest_hi, ~0x73657175656e6365ull, 14029467366897019727ull);
            record.sequence_word_count += words.size();
            ++record.count;
            record.value += value;
        }

        /** @return Whether all counts, extrema and both ordered lanes agree exactly. */
        bool matches(const PerfStatRecord &other) const
        {
            return other.kind == PerfStatRecord::Kind::Counter && other.count == record.count &&
                other.value == record.value && other.sequence_word_count == record.sequence_word_count &&
                other.sequence_digest_lo == record.sequence_digest_lo && other.sequence_digest_hi == record.sequence_digest_hi &&
                other.sequence_minimum_words == record.sequence_minimum_words &&
                other.sequence_maximum_words == record.sequence_maximum_words;
        }
    };

    /**
     * @param records Counter snapshot after the movement authority has quiesced.
     * @param ledger Same authority's complete immutable movement history.
     * @param cycle_cap Configured maximum accepted cycles in each physical wave.
     * @return Empty on exact agreement, otherwise a diagnostic of the first defect.
     */
    inline std::string validateControllerMovementPerfStats(std::span<const PerfStatRecord> records,
        const MoEOptimizationMovementLedger &ledger, std::uint64_t cycle_cap)
    {
        if (!ledger.complete())
            return "controller movement ledger discarded evidence";
        std::map<std::string, MovementSequenceEvidence> expected;
        std::size_t offset = 0;
        try
        {
            for (const auto &publication : ledger.device_publications)
            {
                if (!publication.controller || publication.controller->accepted_cycles > cycle_cap ||
                    publication.command_count > ledger.edges.size() - offset)
                    return "controller movement receipt violates admitted cycle capacity or history";
                visitMoEControllerMovementEvidence(publication,
                    std::span<const MoEOptimizationMovementEdge>(ledger.edges).subspan(offset, publication.command_count),
                    [&](const char *name, double value, std::initializer_list<std::uint64_t> words)
                    { expected[name].append(value, words); });
                offset += publication.command_count;
            }
        }
        catch (const std::exception &error)
        {
            return error.what();
        }
        if (offset != ledger.edges.size())
            return "controller movement edges lost a physical receipt";
        const std::array names{"dynamic_movement_transactions", "dynamic_movement_commands", "dynamic_physical_bytes",
            "dynamic_promotions", "dynamic_demotions", "dynamic_same_priority_moves", "dynamic_cross_domain_moves",
            "dynamic_cross_rank_moves", "dynamic_cross_backend_moves", "dynamic_capacity_conservation_certifications",
            "dynamic_migration_edges", "dynamic_migration_edge_identities"};
        std::set<std::string> seen;
        std::string device;
        for (const auto &record : records)
        {
            if (record.domain != "moe_overlay_controller" || std::find(names.begin(), names.end(), record.name) == names.end())
                continue;
            if (!seen.insert(record.name).second || record.phase != "maintenance" || record.device.empty() ||
                (!device.empty() && device != record.device) || record.tags != moeControllerMovementTags() ||
                !expected.contains(record.name) || !expected.at(record.name).matches(record))
                return "controller counter disagrees with exact completed history: " + record.name;
            device = record.device;
        }
        if (seen.size() != expected.size())
            return "controller movement lacks completed counter families";
        return {};
    }
}
