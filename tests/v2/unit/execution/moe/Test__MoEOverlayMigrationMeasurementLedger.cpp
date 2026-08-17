/**
 * @file Test__MoEOverlayMigrationMeasurementLedger.cpp
 * @brief Device-free lifecycle and robust-median tests for migration evidence.
 */

#include "execution/moe/MoEOverlayMigrationMeasurementLedger.h"
#include "execution/moe/MoEOverlayMigrationMeasurementExchange.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Canonical two-direction, one-layer calibration geometry. */
        std::vector<MoEOverlayMigrationMeasurementCoordinate> coordinates()
        {
            return {
                {.source_participant = 0,
                 .destination_participant = 1,
                 .layer = 0},
                {.source_participant = 1,
                 .destination_participant = 0,
                 .layer = 0},
            };
        }

        /** @brief Every production phase that must independently prove overlap. */
        constexpr std::array<ExpertHistogramSource, 3> kProductionPhases{
            ExpertHistogramSource::DecodeToken,
            ExpertHistogramSource::PrefillChunk,
            ExpertHistogramSource::GroupedVerifier,
        };

        /** @brief Construct one complete three-projection physical observation. */
        MoEOverlayCompletedMigrationMeasurement measurement(
            int source,
            int destination,
            std::uint64_t sequence,
            std::uint64_t timing_base,
            std::uint64_t bytes_base = 1024)
        {
            MoEOverlayCompletedMigrationMeasurement result{
                .expected_epoch = sequence,
                .candidate_epoch = sequence + 1,
                .source_participant = source,
                .destination_participant = destination,
                .layer = 0,
                .expert = static_cast<int>(sequence),
                .wave_wall_nanoseconds = timing_base * 10,
            };
            for (std::size_t projection = 0; projection < 3; ++projection)
            {
                result.projections[projection] = {
                    .sequence = sequence,
                    .bytes = bytes_base + projection,
                    .wall_nanoseconds = timing_base + projection,
                    .device_nanoseconds = timing_base / 2 + projection + 1,
                    .host_nanoseconds = 0,
                };
            }
            return result;
        }

        /** @brief Feed one warmup and three deliberately unsorted samples. */
        void fillLedger(MoEOverlayMigrationMeasurementLedger &ledger)
        {
            constexpr std::array<std::uint64_t, 4> timing{
                50, 300, 100, 200};
            constexpr std::array<std::uint64_t, 4> interference{
                50, 30, 10, 20};
            for (std::size_t sample = 0; sample < timing.size(); ++sample)
            {
                std::vector<MoEOverlayCompletedMigrationMeasurement> wave;
                /* Reverse input order to prove canonical coordinate lookup. */
                wave.push_back(measurement(
                    1, 0, sample + 1, timing[sample] + 10));
                wave.push_back(measurement(
                    0, 1, sample + 1, timing[sample]));
                ASSERT_TRUE(ledger.recordCompletedWave(wave, nullptr));

                for (const auto &coordinate : coordinates())
                {
                    for (const auto phase : kProductionPhases)
                    {
                        const auto phase_penalty =
                            static_cast<std::uint64_t>(phase) * 100u;
                        ASSERT_TRUE(ledger.recordInterferenceSample(
                            coordinate,
                            phase,
                            1'000,
                            1'000 + interference[sample] + phase_penalty,
                            nullptr));
                    }
                }
            }
        }
    } // namespace

    TEST(
        MoEOverlayMigrationMeasurementLedger,
        RejectsIncompleteOrAmbiguousCorpusGeometry)
    {
        EXPECT_THROW(
            (MoEOverlayMigrationMeasurementLedger({
                .required_coordinates = {},
                .measurement_identity = "test",
            })),
            std::invalid_argument);
        EXPECT_THROW(
            (MoEOverlayMigrationMeasurementLedger({
                .required_coordinates = coordinates(),
                .measured_samples_per_coordinate = 2,
                .measurement_identity = "test",
            })),
            std::invalid_argument);

        auto duplicate = coordinates();
        duplicate.push_back(duplicate.front());
        EXPECT_THROW(
            (MoEOverlayMigrationMeasurementLedger({
                .required_coordinates = std::move(duplicate),
                .measurement_identity = "test",
            })),
            std::invalid_argument);
    }

    TEST(
        MoEOverlayMigrationMeasurementLedger,
        WarmupIsDiscardedAndThreeSamplesProduceDeterministicMedians)
    {
        MoEOverlayMigrationMeasurementLedger ledger({
            .required_coordinates = coordinates(),
            .warmup_samples_per_coordinate = 1,
            .measured_samples_per_coordinate = 3,
            .measurement_identity = "loaded-model/topology-generation-7",
        });
        EXPECT_FALSE(ledger.ready());
        fillLedger(ledger);
        ASSERT_TRUE(ledger.ready());

        const auto sealed = ledger.seal();
        ASSERT_TRUE(sealed.valid());
        ASSERT_EQ(sealed.rows.size(), 2u);
        EXPECT_EQ(sealed.rows[0].source_participant, 0);
        EXPECT_EQ(sealed.rows[0].destination_participant, 1);
        EXPECT_EQ(sealed.rows[0].wave_wall_nanoseconds, 2'000u);
        EXPECT_EQ(
            sealed.rows[0].inference_interference_nanoseconds,
            220u);
        EXPECT_EQ(sealed.rows[0].interference_sample_count, 9u);
        EXPECT_EQ(sealed.rows[0].projections[0].wall_nanoseconds, 200u);
        EXPECT_EQ(sealed.rows[0].projections[2].wall_nanoseconds, 202u);
        EXPECT_EQ(sealed.rows[1].wave_wall_nanoseconds, 2'100u);

        const auto normalized =
            MoEOverlayEconomyProfileComposer::normalizeMigrationMeasurements(
                sealed.rows);
        ASSERT_EQ(normalized.size(), 2u);
        EXPECT_EQ(normalized[0].transfer_and_repack_ns, 2'000u);
        EXPECT_EQ(normalized[0].inference_interference_ns, 220u);

        const auto stats = ledger.stats();
        EXPECT_EQ(stats.completed_waves_observed, 4u);
        EXPECT_EQ(stats.migration_observations_seen, 8u);
        EXPECT_EQ(stats.warmup_migration_observations, 2u);
        EXPECT_EQ(stats.stored_migration_observations, 6u);
        EXPECT_EQ(stats.warmup_interference_observations, 6u);
        EXPECT_EQ(stats.stored_interference_observations, 18u);
        EXPECT_EQ(stats.rejected_observations, 0u);
        EXPECT_FALSE(ledger.ready());
        EXPECT_THROW((void)ledger.seal(), std::logic_error);
    }

    TEST(
        MoEOverlayMigrationMeasurementLedger,
        WholeWaveValidationIsAtomicAndSealedLedgerRejectsWrites)
    {
        MoEOverlayMigrationMeasurementLedger ledger({
            .required_coordinates = coordinates(),
            .warmup_samples_per_coordinate = 0,
            .measured_samples_per_coordinate = 3,
            .measurement_identity = "atomic-wave",
        });
        auto valid = measurement(0, 1, 1, 100);
        auto partial = measurement(1, 0, 1, 100);
        partial.projections[1] = std::nullopt;
        std::string error;
        EXPECT_FALSE(ledger.recordCompletedWave({valid, partial}, &error));
        EXPECT_NE(error.find("partial observation"), std::string::npos);
        EXPECT_EQ(ledger.stats().migration_observations_seen, 0u);

        for (std::uint64_t sample = 1; sample <= 3; ++sample)
        {
            ASSERT_TRUE(ledger.recordCompletedWave(
                {measurement(0, 1, sample, 100 + sample),
                 measurement(1, 0, sample, 110 + sample)},
                nullptr));
            for (const auto &coordinate : coordinates())
            {
                for (const auto phase : kProductionPhases)
                {
                    ASSERT_TRUE(ledger.recordInterferenceSample(
                        coordinate, phase, 1'000, 1'000, nullptr));
                }
            }
        }
        ASSERT_TRUE(ledger.ready());
        const auto sealed = ledger.seal();
        ASSERT_TRUE(sealed.valid());
        EXPECT_FALSE(ledger.recordCompletedWave({valid}, &error));
        EXPECT_EQ(error, "ExpertOverlay migration ledger is sealed");
        EXPECT_FALSE(ledger.recordInterferenceSample(
            coordinates().front(),
            ExpertHistogramSource::DecodeToken,
            1,
            1,
            &error));
        EXPECT_NE(error.find("sealed"), std::string::npos);
    }

    TEST(
        MoEOverlayMigrationMeasurementLedger,
        ProjectionByteGeometryCannotChangeBetweenMeasuredSamples)
    {
        MoEOverlayMigrationMeasurementLedger ledger({
            .required_coordinates = {coordinates().front()},
            .warmup_samples_per_coordinate = 0,
            .measured_samples_per_coordinate = 3,
            .measurement_identity = "stable-bytes",
        });
        ASSERT_TRUE(ledger.recordCompletedWave(
            {measurement(0, 1, 1, 100)}, nullptr));
        std::string error;
        EXPECT_FALSE(ledger.recordCompletedWave(
            {measurement(0, 1, 2, 100, 2048)}, &error));
        EXPECT_NE(error.find("changing projection bytes"), std::string::npos);
        EXPECT_EQ(ledger.stats().migration_observations_seen, 1u);
    }

    TEST(
        MoEOverlayMigrationMeasurementExchange,
        JournalDoesNotOverwriteAndMergerCompletesComplementaryRanks)
    {
        auto source_rank = measurement(0, 1, 1, 100);
        auto destination_rank = source_rank;
        source_rank.projections[1] = std::nullopt;
        destination_rank.projections[0] = std::nullopt;
        destination_rank.projections[2] = std::nullopt;
        destination_rank.wave_wall_nanoseconds += 50;
        destination_rank.projections[1]->transport_nanoseconds = 77;

        MoEOverlayMigrationMeasurementJournal journal({
            .maximum_migrations_per_wave = 2,
        });
        std::string error;
        ASSERT_TRUE(journal.recordCompletedWave({source_rank}, &error))
            << error;
        EXPECT_TRUE(journal.hasUnreadWave());
        EXPECT_FALSE(journal.recordCompletedWave({source_rank}, &error));
        EXPECT_NE(error.find("overwrite"), std::string::npos);

        std::vector<MoEOverlayCompletedMigrationMeasurement> consumed;
        ASSERT_TRUE(journal.consume(&consumed, &error)) << error;
        ASSERT_EQ(consumed.size(), 1u);
        EXPECT_FALSE(journal.hasUnreadWave());
        EXPECT_FALSE(journal.consume(&consumed, &error));

        const auto merged =
            MoEOverlayMigrationMeasurementMerger::merge(
                {{source_rank}, {destination_rank}});
        ASSERT_EQ(merged.size(), 1u);
        ASSERT_TRUE(merged.front().valid());
        EXPECT_EQ(
            merged.front().wave_wall_nanoseconds,
            destination_rank.wave_wall_nanoseconds);
        for (const auto &projection : merged.front().projections)
            ASSERT_TRUE(projection.has_value());
        EXPECT_EQ(
            merged.front().projections[1]->transport_nanoseconds,
            77u);

        const auto stats = journal.stats();
        EXPECT_EQ(stats.waves_recorded, 1u);
        EXPECT_EQ(stats.waves_consumed, 1u);
        EXPECT_EQ(stats.rejected_waves, 1u);
    }

    TEST(
        MoEOverlayMigrationMeasurementExchange,
        MergerRejectsDivergentIdentityBytesAndMissingProjection)
    {
        auto first = measurement(0, 1, 1, 100);
        auto divergent = first;
        divergent.expert += 1;
        EXPECT_THROW(
            (void)MoEOverlayMigrationMeasurementMerger::merge(
                {{first}, {divergent}}),
            std::invalid_argument);

        auto conflicting_bytes = first;
        conflicting_bytes.projections[0]->bytes += 1;
        EXPECT_THROW(
            (void)MoEOverlayMigrationMeasurementMerger::merge(
                {{first}, {conflicting_bytes}}),
            std::invalid_argument);

        first.projections[2] = std::nullopt;
        EXPECT_THROW(
            (void)MoEOverlayMigrationMeasurementMerger::merge({{first}}),
            std::invalid_argument);
    }
} // namespace llaminar2::test
