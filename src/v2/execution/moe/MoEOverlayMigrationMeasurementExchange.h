/**
 * @file MoEOverlayMigrationMeasurementExchange.h
 * @brief Rank-local journaling and deterministic merge of migration evidence.
 *
 * A distributed projection operation is observed only by its source and
 * destination ranks; uninvolved ranks retain explicit empty optionals so the
 * global operation order remains identical.  These types separate the
 * non-blocking transport callback from later collective exchange: a bounded
 * journal captures one local wave without allocation, then the calibration
 * owner merges authenticated rank rows into one complete conservative wave.
 */

#pragma once

#include "MoEOverlayTierMigrationTransport.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace llaminar2
{
    /** @brief Cumulative proof counters for a bounded local measurement journal. */
    struct MoEOverlayMigrationMeasurementJournalStats
    {
        std::uint64_t waves_recorded = 0;
        std::uint64_t waves_consumed = 0;
        std::uint64_t rejected_waves = 0;
    };

    /**
     * @brief Single-producer/single-consumer handoff for one completed wave.
     *
     * The transport callback and calibration driver execute on the same
     * maintenance worker. Constructor-time reserve makes `recordCompletedWave`
     * allocation-free; an unread observation cannot be overwritten.
     */
    class MoEOverlayMigrationMeasurementJournal final
        : public IMoEOverlayMigrationMeasurementSink
    {
    public:
        /** @brief Fixed largest expert-migration cardinality of one wave. */
        struct Config
        {
            std::size_t maximum_migrations_per_wave = 0;
        };

        /**
         * @brief Allocate one bounded journal before any migration runs.
         * @throws std::invalid_argument For zero capacity.
         */
        explicit MoEOverlayMigrationMeasurementJournal(Config config);

        /** @brief Copy one valid local partial wave without allocation or wait. */
        bool recordCompletedWave(
            const std::vector<MoEOverlayCompletedMigrationMeasurement> &
                measurements,
            std::string *error) noexcept override;

        /**
         * @brief Move the unread wave to its maintenance-side exchange owner.
         * @param output Replaced by the exact local observation.
         * @param error Optional lifecycle diagnostic.
         * @return True only when one unread wave was consumed.
         */
        bool consume(
            std::vector<MoEOverlayCompletedMigrationMeasurement> *output,
            std::string *error = nullptr) noexcept;

        /** @return Whether one complete local callback is waiting for exchange. */
        [[nodiscard]] bool hasUnreadWave() const noexcept
        {
            return unread_;
        }

        /** @return Cumulative journal lifecycle evidence. */
        [[nodiscard]] MoEOverlayMigrationMeasurementJournalStats stats()
            const noexcept;

    private:
        Config config_;
        std::vector<MoEOverlayCompletedMigrationMeasurement> wave_;
        bool unread_ = false;
        std::atomic<std::uint64_t> waves_recorded_{0};
        std::atomic<std::uint64_t> waves_consumed_{0};
        std::atomic<std::uint64_t> rejected_waves_{0};
    };

    /**
     * @brief Pure deterministic merger for partial observations from all ranks.
     *
     * Duplicate source/destination endpoint observations are reduced
     * conservatively: bytes must agree, while wall, device, host, transport,
     * and whole-wave durations use their maximum. This prevents a faster rank
     * from hiding the actual critical path and gives every rank identical
     * pointer-free evidence after the same authenticated exchange.
     */
    class MoEOverlayMigrationMeasurementMerger final
    {
    public:
        /**
         * @brief Merge transaction-ordered local rank observations.
         * @param rank_measurements One row vector per participating world rank.
         * @return Complete rows with all three projections present.
         * @throws std::invalid_argument For missing ranks, identity divergence,
         *         inconsistent bytes, duplicate rank-local values, or an
         *         unobserved projection.
         */
        [[nodiscard]] static std::vector<
            MoEOverlayCompletedMigrationMeasurement>
        merge(const std::vector<std::vector<
                  MoEOverlayCompletedMigrationMeasurement>> &rank_measurements);
    };
} // namespace llaminar2
