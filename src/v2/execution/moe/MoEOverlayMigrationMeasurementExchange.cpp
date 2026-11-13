/**
 * @file MoEOverlayMigrationMeasurementExchange.cpp
 * @brief Bounded local observation handoff and conservative distributed reduction.
 */

#include "MoEOverlayMigrationMeasurementExchange.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @brief Store one journal diagnostic only when requested. */
        void assignExchangeError(
            std::string *error,
            const char *message) noexcept
        {
            if (error)
                *error = message;
        }

        /** @return Whether two rows name the identical migration coordinate. */
        bool sameMigrationIdentity(
            const MoEOverlayCompletedMigrationMeasurement &left,
            const MoEOverlayCompletedMigrationMeasurement &right) noexcept
        {
            return left.expected_epoch == right.expected_epoch &&
                   left.candidate_epoch == right.candidate_epoch &&
                   left.source_participant == right.source_participant &&
                   left.destination_participant ==
                       right.destination_participant &&
                   left.layer == right.layer && left.expert == right.expert &&
                   bool(left.geometry) == bool(right.geometry) &&
                   (!left.geometry || left.geometry->sameIdentity(*right.geometry));
        }
    } // namespace

    MoEOverlayMigrationMeasurementJournal::
        MoEOverlayMigrationMeasurementJournal(Config config)
        : config_(config)
    {
        if (config_.maximum_migrations_per_wave == 0)
        {
            throw std::invalid_argument(
                "ExpertOverlay migration measurement journal requires positive wave capacity");
        }
        wave_.reserve(config_.maximum_migrations_per_wave);
    }

    bool MoEOverlayMigrationMeasurementJournal::recordCompletedWave(
        const std::vector<MoEOverlayCompletedMigrationMeasurement> &measurements,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (std::any_of(measurements.begin(), measurements.end(), [](const auto &row) { return bool(row.geometry) || !row.piece_observations.empty(); }))
        {
            rejected_waves_.fetch_add(1, std::memory_order_relaxed);
            assignExchangeError(error, "Piece observations require original owned-wave handoff, never a nested completion copy");
            return false;
        }
        if (unread_ || measurements.empty() ||
            measurements.size() > config_.maximum_migrations_per_wave ||
            !std::all_of(
                measurements.begin(),
                measurements.end(),
                [](const auto &measurement)
                { return measurement.valid(); }))
        {
            rejected_waves_.fetch_add(1, std::memory_order_relaxed);
            assignExchangeError(
                error,
                unread_
                    ? "ExpertOverlay migration journal would overwrite an unread wave"
                    : "ExpertOverlay migration journal received invalid or oversized evidence");
            return false;
        }

        /* Outer cardinality is reserved; nested passive geometry follows exact
         * authenticated piece counts. No source payload or timing is synthesized. */
        try { wave_.assign(measurements.begin(), measurements.end()); }
        catch (...)
        {
            wave_.clear();
            rejected_waves_.fetch_add(1, std::memory_order_relaxed);
            assignExchangeError(error, "ExpertOverlay journal could not retain exact piece metadata");
            return false;
        }
        unread_ = true;
        waves_recorded_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool MoEOverlayMigrationMeasurementJournal::recordCompletedWave(
        std::vector<MoEOverlayCompletedMigrationMeasurement> &&measurements, std::string *error) noexcept
    {
        if (error) error->clear();
        if (unread_ || measurements.empty() || measurements.size() > config_.maximum_migrations_per_wave ||
            !std::all_of(measurements.begin(), measurements.end(), [](const auto &row) { return row.valid(); }))
        {
            rejected_waves_.fetch_add(1, std::memory_order_relaxed);
            assignExchangeError(error, "Journal cannot accept incomplete evidence or overwrite an unread wave");
            return false;
        }
        // The original vector, geometry owners and sample allocations cross
        // this edge unchanged. No completion-time copy, reserve or allocation.
        wave_.swap(measurements);
        unread_ = true;
        waves_recorded_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool MoEOverlayMigrationMeasurementJournal::consume(
        std::vector<MoEOverlayCompletedMigrationMeasurement> *output,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!output || !unread_)
        {
            assignExchangeError(
                error,
                output
                    ? "ExpertOverlay migration journal has no unread wave"
                    : "ExpertOverlay migration journal requires an output owner");
            return false;
        }
        *output = std::move(wave_);
        wave_.clear();
        wave_.reserve(config_.maximum_migrations_per_wave);
        unread_ = false;
        waves_consumed_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    MoEOverlayMigrationMeasurementJournalStats
    MoEOverlayMigrationMeasurementJournal::stats() const noexcept
    {
        return {
            .waves_recorded = waves_recorded_.load(std::memory_order_relaxed),
            .waves_consumed = waves_consumed_.load(std::memory_order_relaxed),
            .rejected_waves = rejected_waves_.load(std::memory_order_relaxed),
        };
    }

    std::vector<MoEOverlayCompletedMigrationMeasurement>
    MoEOverlayMigrationMeasurementMerger::merge(
        const std::vector<std::vector<
            MoEOverlayCompletedMigrationMeasurement>> &rank_measurements)
    {
        if (rank_measurements.empty() || rank_measurements.front().empty())
        {
            throw std::invalid_argument(
                "ExpertOverlay migration merge requires non-empty rank evidence");
        }
        const std::size_t migration_count = rank_measurements.front().size();
        for (const auto &rank : rank_measurements)
        {
            if (rank.size() != migration_count)
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration rank evidence has divergent wave cardinality");
            }
        }

        std::vector<MoEOverlayCompletedMigrationMeasurement> result;
        result.reserve(migration_count);
        for (std::size_t migration = 0;
             migration < migration_count;
             ++migration)
        {
            MoEOverlayCompletedMigrationMeasurement merged =
                rank_measurements.front()[migration];
            merged.projections = {};
            for (auto &piece : merged.piece_observations) { piece.observation.reset(); piece.quoted_payload_bytes.reset(); }
            merged.wave_wall_nanoseconds = 0;
            for (const auto &rank : rank_measurements)
            {
                const auto &local = rank[migration];
                if (!local.valid() ||
                    !sameMigrationIdentity(
                        rank_measurements.front()[migration], local))
                {
                    throw std::invalid_argument(
                        "ExpertOverlay migration ranks disagree on completed-wave identity");
                }
                merged.wave_wall_nanoseconds = std::max(
                    merged.wave_wall_nanoseconds,
                    local.wave_wall_nanoseconds);
                if (merged.geometry)
                {
                    for (std::size_t piece = 0; piece < merged.piece_observations.size(); ++piece)
                    {
                        const auto &sample = local.piece_observations[piece];
                        auto &aggregate = merged.piece_observations[piece];
                        if (sample.quoted_payload_bytes)
                        {
                            if (aggregate.quoted_payload_bytes && aggregate.quoted_payload_bytes != sample.quoted_payload_bytes)
                                throw std::invalid_argument("Physical piece endpoints disagree on bound payload quote");
                            aggregate.quoted_payload_bytes = sample.quoted_payload_bytes;
                        }
                        if (!sample.observation) continue;
                        if (!aggregate.observation) { aggregate.observation = sample.observation; continue; }
                        const auto &observed = *sample.observation;
                        auto &value = *aggregate.observation;
                        if (value.bytes != observed.bytes)
                            throw std::invalid_argument("Physical piece ranks disagree on exact live bytes");
                        value.sequence = std::max(value.sequence, observed.sequence);
                        value.wall_nanoseconds = std::max(value.wall_nanoseconds, observed.wall_nanoseconds);
                        value.device_nanoseconds = std::max(value.device_nanoseconds, observed.device_nanoseconds);
                        value.host_nanoseconds = std::max(value.host_nanoseconds, observed.host_nanoseconds);
                        value.transport_nanoseconds = std::max(value.transport_nanoseconds, observed.transport_nanoseconds);
                    }
                    continue;
                }
                for (std::size_t projection = 0; projection < 3; ++projection)
                {
                    if (!local.projections[projection])
                        continue;
                    const auto &observation =
                        *local.projections[projection];
                    if (!observation.valid())
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay migration rank supplied invalid projection evidence");
                    }
                    auto &aggregate = merged.projections[projection];
                    if (!aggregate)
                    {
                        aggregate = observation;
                        continue;
                    }
                    if (aggregate->bytes != observation.bytes)
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay migration ranks disagree on projection bytes");
                    }
                    aggregate->sequence = std::max(
                        aggregate->sequence, observation.sequence);
                    aggregate->wall_nanoseconds = std::max(
                        aggregate->wall_nanoseconds,
                        observation.wall_nanoseconds);
                    aggregate->device_nanoseconds = std::max(
                        aggregate->device_nanoseconds,
                        observation.device_nanoseconds);
                    aggregate->host_nanoseconds = std::max(
                        aggregate->host_nanoseconds,
                        observation.host_nanoseconds);
                    aggregate->transport_nanoseconds = std::max(
                        aggregate->transport_nanoseconds,
                        observation.transport_nanoseconds);
                }
            }
            const bool missing_pieces = merged.geometry && std::any_of(
                merged.piece_observations.begin(), merged.piece_observations.end(),
                [](const auto &piece) { return !piece.observation || !piece.quoted_payload_bytes || !piece.valid(); });
            if (!merged.valid() || missing_pieces ||
                (!merged.geometry && std::any_of(
                    merged.projections.begin(),
                    merged.projections.end(),
                    [](const auto &projection)
                    { return !projection || !projection->valid(); })))
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration ranks omitted a complete projection observation");
            }
            result.push_back(std::move(merged));
        }
        return result;
    }
} // namespace llaminar2
