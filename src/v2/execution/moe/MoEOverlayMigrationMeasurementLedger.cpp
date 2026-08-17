/**
 * @file MoEOverlayMigrationMeasurementLedger.cpp
 * @brief Bounded sample retention and robust medians for migration economy.
 */

#include "MoEOverlayMigrationMeasurementLedger.h"

#include "../../utils/FNV1a.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** Sentinel returned when a required coordinate is absent. */
        constexpr std::size_t kMissingCoordinate =
            std::numeric_limits<std::size_t>::max();

        /** @brief Store one failure diagnostic only when requested. */
        void assignLedgerError(
            std::string *error,
            const char *message) noexcept
        {
            if (error)
                *error = message;
        }

        /** @brief Compute the deterministic lower median of a copied sample set. */
        template <typename Value>
        Value median(std::vector<Value> values)
        {
            std::sort(values.begin(), values.end());
            return values[(values.size() - 1u) / 2u];
        }

        /** @brief Add a scalar to the platform-independent corpus digest. */
        void hashUnsigned(std::uint64_t &hash, std::uint64_t value) noexcept
        {
            std::array<unsigned char, sizeof(value)> bytes{};
            for (std::size_t index = 0; index < bytes.size(); ++index)
            {
                bytes[index] = static_cast<unsigned char>(value & 0xffu);
                value >>= 8u;
            }
            hash = fnv1a64(bytes.data(), bytes.size(), hash);
        }

        /** @brief Add a length-delimited string to the corpus digest. */
        void hashString(std::uint64_t &hash, const std::string &value) noexcept
        {
            hashUnsigned(hash, static_cast<std::uint64_t>(value.size()));
            hash = fnv1a64(value.data(), value.size(), hash);
        }
    } // namespace

    bool MoEOverlayMigrationMeasurementCoordinate::valid() const noexcept
    {
        return source_participant >= 0 && destination_participant >= 0 &&
               source_participant != destination_participant && layer >= 0;
    }

    bool MoEOverlaySealedMigrationMeasurements::valid() const noexcept
    {
        return !identity.empty() && !rows.empty() &&
               std::all_of(
                   rows.begin(),
                   rows.end(),
                   [](const auto &row)
                   {
                       if (row.source_participant < 0 ||
                           row.destination_participant < 0 ||
                           row.source_participant ==
                               row.destination_participant ||
                           row.layer < 0 || row.wave_wall_nanoseconds == 0 ||
                           row.wave_sample_count <
                               MoEOverlayEconomyProfileComposer::
                                   kMinimumMigrationSamples ||
                           row.interference_sample_count <
                               MoEOverlayEconomyProfileComposer::
                                   kMinimumMigrationSamples)
                       {
                           return false;
                       }
                       return std::all_of(
                           row.projections.begin(),
                           row.projections.end(),
                           [](const auto &projection)
                           { return projection.valid(); });
                   });
    }

    MoEOverlayMigrationMeasurementLedger::
        MoEOverlayMigrationMeasurementLedger(Config config)
        : config_(std::move(config)),
          coordinates_(config_.required_coordinates)
    {
        if (coordinates_.empty() || config_.measurement_identity.empty() ||
            config_.measured_samples_per_coordinate <
                MoEOverlayEconomyProfileComposer::kMinimumMigrationSamples ||
            !validExpertHistogramProductionSourceMask(
                config_.required_sources))
        {
            throw std::invalid_argument(
                "ExpertOverlay migration ledger requires coordinates, identity, runtime phases, and at least three measured samples");
        }
        std::sort(coordinates_.begin(), coordinates_.end());
        for (std::size_t index = 0; index < coordinates_.size(); ++index)
        {
            if (!coordinates_[index].valid() ||
                (index != 0 && coordinates_[index] == coordinates_[index - 1]))
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration ledger coordinates are invalid or duplicated");
            }
        }

        if (coordinates_.size() >
            std::numeric_limits<std::size_t>::max() /
                config_.measured_samples_per_coordinate)
        {
            throw std::invalid_argument(
                "ExpertOverlay migration ledger sample geometry overflows host size");
        }
        const std::size_t retained_samples =
            coordinates_.size() *
            static_cast<std::size_t>(
                config_.measured_samples_per_coordinate);
        if (retained_samples >
            std::numeric_limits<std::size_t>::max() /
                kExpertHistogramProductionSourceCount)
        {
            throw std::invalid_argument(
                "ExpertOverlay migration ledger phase sample geometry overflows host size");
        }
        coordinate_state_ =
            std::make_unique<CoordinateState[]>(coordinates_.size());
        wave_samples_.resize(retained_samples);
        interference_samples_.resize(
            retained_samples * kExpertHistogramProductionSourceCount);
        for (auto &samples : projection_samples_)
            samples.resize(retained_samples);
    }

    std::size_t MoEOverlayMigrationMeasurementLedger::coordinateIndex(
        const MoEOverlayMigrationMeasurementCoordinate &coordinate)
        const noexcept
    {
        const auto found =
            std::lower_bound(coordinates_.begin(), coordinates_.end(), coordinate);
        if (found == coordinates_.end() || *found != coordinate)
            return kMissingCoordinate;
        return static_cast<std::size_t>(found - coordinates_.begin());
    }

    std::size_t MoEOverlayMigrationMeasurementLedger::sampleOffset(
        std::size_t coordinate_index,
        std::uint64_t measured_sample_index) const noexcept
    {
        return coordinate_index *
                   static_cast<std::size_t>(
                       config_.measured_samples_per_coordinate) +
               static_cast<std::size_t>(measured_sample_index);
    }

    std::size_t
    MoEOverlayMigrationMeasurementLedger::interferenceSampleOffset(
        std::size_t coordinate_index,
        std::size_t phase_index,
        std::uint64_t measured_sample_index) const noexcept
    {
        return (coordinate_index * kExpertHistogramProductionSourceCount +
                phase_index) *
                   static_cast<std::size_t>(
                       config_.measured_samples_per_coordinate) +
               static_cast<std::size_t>(measured_sample_index);
    }

    bool MoEOverlayMigrationMeasurementLedger::recordCompletedWave(
        const std::vector<MoEOverlayCompletedMigrationMeasurement> &measurements,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (sealed_.load(std::memory_order_acquire))
        {
            rejected_observations_.fetch_add(1, std::memory_order_relaxed);
            assignLedgerError(error, "ExpertOverlay migration ledger is sealed");
            return false;
        }
        if (measurements.empty())
        {
            rejected_observations_.fetch_add(1, std::memory_order_relaxed);
            assignLedgerError(error, "ExpertOverlay migration ledger received an empty wave");
            return false;
        }

        /*
         * Validate the whole vector before mutating any cell.  The transaction
         * is small and bounded by participant shadow capacity, so an O(n^2)
         * duplicate check avoids allocating scratch memory in this callback.
         */
        for (std::size_t index = 0; index < measurements.size(); ++index)
        {
            const auto &measurement = measurements[index];
            const MoEOverlayMigrationMeasurementCoordinate coordinate{
                .source_participant = measurement.source_participant,
                .destination_participant = measurement.destination_participant,
                .layer = measurement.layer,
            };
            const std::size_t coordinate_index = coordinateIndex(coordinate);
            if (!measurement.valid() ||
                coordinate_index == kMissingCoordinate ||
                std::any_of(
                    measurement.projections.begin(),
                    measurement.projections.end(),
                    [](const auto &projection)
                    {
                        return !projection || !projection->valid() ||
                               (projection->device_nanoseconds == 0 &&
                                projection->host_nanoseconds == 0 &&
                                projection->transport_nanoseconds == 0);
                    }))
            {
                rejected_observations_.fetch_add(
                    1, std::memory_order_relaxed);
                assignLedgerError(
                    error,
                    "ExpertOverlay migration ledger received an invalid, unexpected, or partial observation");
                return false;
            }
            for (std::size_t previous = 0; previous < index; ++previous)
            {
                if (measurements[previous].source_participant ==
                        measurement.source_participant &&
                    measurements[previous].destination_participant ==
                        measurement.destination_participant &&
                    measurements[previous].layer == measurement.layer)
                {
                    rejected_observations_.fetch_add(
                        1, std::memory_order_relaxed);
                    assignLedgerError(
                        error,
                        "ExpertOverlay migration ledger wave repeats a directed coordinate");
                    return false;
                }
            }
            const auto observed = coordinate_state_[coordinate_index]
                                      .migration_observations.load(
                                          std::memory_order_relaxed);
            if (observed >= config_.warmup_samples_per_coordinate)
            {
                for (std::size_t projection = 0; projection < 3; ++projection)
                {
                    const auto bytes =
                        measurement.projections[projection]->bytes;
                    const auto expected = coordinate_state_[coordinate_index]
                                              .projection_bytes[projection];
                    if (expected != 0 && expected != bytes)
                    {
                        rejected_observations_.fetch_add(
                            1, std::memory_order_relaxed);
                        assignLedgerError(
                            error,
                            "ExpertOverlay migration ledger observed changing projection bytes for one coordinate");
                        return false;
                    }
                }
            }
        }

        for (const auto &measurement : measurements)
        {
            const MoEOverlayMigrationMeasurementCoordinate coordinate{
                .source_participant = measurement.source_participant,
                .destination_participant = measurement.destination_participant,
                .layer = measurement.layer,
            };
            const std::size_t coordinate_index = coordinateIndex(coordinate);
            auto &state = coordinate_state_[coordinate_index];
            const std::uint64_t observed =
                state.migration_observations.load(std::memory_order_relaxed);
            migration_observations_seen_.fetch_add(
                1, std::memory_order_relaxed);
            if (observed < config_.warmup_samples_per_coordinate)
            {
                warmup_migration_observations_.fetch_add(
                    1, std::memory_order_relaxed);
            }
            else
            {
                const std::uint64_t measured_index =
                    observed - config_.warmup_samples_per_coordinate;
                if (measured_index <
                    config_.measured_samples_per_coordinate)
                {
                    const std::size_t offset =
                        sampleOffset(coordinate_index, measured_index);
                    wave_samples_[offset] =
                        measurement.wave_wall_nanoseconds;
                    for (std::size_t projection = 0; projection < 3; ++projection)
                    {
                        const auto value =
                            *measurement.projections[projection];
                        projection_samples_[projection][offset] = value;
                        if (state.projection_bytes[projection] == 0)
                            state.projection_bytes[projection] = value.bytes;
                    }
                    stored_migration_observations_.fetch_add(
                        1, std::memory_order_relaxed);
                }
                else
                {
                    excess_migration_observations_.fetch_add(
                        1, std::memory_order_relaxed);
                }
            }
            /* Publish completed sample bytes before the visible count. */
            state.migration_observations.store(
                observed + 1u, std::memory_order_release);
        }
        completed_waves_observed_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool MoEOverlayMigrationMeasurementLedger::recordInterferenceSample(
        MoEOverlayMigrationMeasurementCoordinate coordinate,
        ExpertHistogramSource source,
        std::uint64_t inference_only_nanoseconds,
        std::uint64_t concurrent_nanoseconds,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        const std::size_t coordinate_index = coordinateIndex(coordinate);
        const auto phase_index = static_cast<std::size_t>(source);
        if (sealed_.load(std::memory_order_acquire) ||
            coordinate_index == kMissingCoordinate ||
            source == ExpertHistogramSource::SyntheticTest ||
            phase_index >= kExpertHistogramProductionSourceCount ||
            !config_.required_sources[phase_index] ||
            inference_only_nanoseconds == 0 || concurrent_nanoseconds == 0)
        {
            rejected_observations_.fetch_add(1, std::memory_order_relaxed);
            assignLedgerError(
                error,
                "ExpertOverlay migration ledger received an invalid interference pair or is sealed");
            return false;
        }

        auto &state = coordinate_state_[coordinate_index];
        const std::uint64_t observed =
            state.interference_observations[phase_index].load(
                std::memory_order_relaxed);
        if (observed < config_.warmup_samples_per_coordinate)
        {
            warmup_interference_observations_.fetch_add(
                1, std::memory_order_relaxed);
        }
        else
        {
            const std::uint64_t measured_index =
                observed - config_.warmup_samples_per_coordinate;
            if (measured_index < config_.measured_samples_per_coordinate)
            {
                interference_samples_[interferenceSampleOffset(
                    coordinate_index, phase_index, measured_index)] =
                    concurrent_nanoseconds > inference_only_nanoseconds
                        ? concurrent_nanoseconds - inference_only_nanoseconds
                        : 0u;
                stored_interference_observations_.fetch_add(
                    1, std::memory_order_relaxed);
            }
            else
            {
                excess_interference_observations_.fetch_add(
                    1, std::memory_order_relaxed);
            }
        }
        state.interference_observations[phase_index].store(
            observed + 1u, std::memory_order_release);
        return true;
    }

    bool MoEOverlayMigrationMeasurementLedger::ready() const noexcept
    {
        if (sealed_.load(std::memory_order_acquire))
            return false;
        const std::uint64_t required =
            config_.warmup_samples_per_coordinate +
            config_.measured_samples_per_coordinate;
        for (std::size_t index = 0; index < coordinates_.size(); ++index)
        {
            if (coordinate_state_[index].migration_observations.load(
                    std::memory_order_acquire) < required)
            {
                return false;
            }
            for (std::size_t phase = 0;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                if (!config_.required_sources[phase])
                    continue;
                if (coordinate_state_[index]
                        .interference_observations[phase]
                        .load(std::memory_order_acquire) < required)
                {
                    return false;
                }
            }
        }
        return true;
    }

    MoEOverlaySealedMigrationMeasurements
    MoEOverlayMigrationMeasurementLedger::seal()
    {
        if (sealed_.load(std::memory_order_acquire))
        {
            throw std::logic_error(
                "ExpertOverlay migration ledger was already sealed");
        }
        if (!ready())
        {
            throw std::logic_error(
                "ExpertOverlay migration ledger cannot seal an incomplete corpus");
        }

        MoEOverlaySealedMigrationMeasurements result;
        result.rows.reserve(coordinates_.size());
        std::vector<
            std::array<
                std::uint64_t,
                kExpertHistogramProductionSourceCount>>
            phase_interference_medians;
        phase_interference_medians.reserve(coordinates_.size());
        const std::size_t sample_count = static_cast<std::size_t>(
            config_.measured_samples_per_coordinate);
        for (std::size_t coordinate_index = 0;
             coordinate_index < coordinates_.size();
             ++coordinate_index)
        {
            const std::size_t begin = sampleOffset(coordinate_index, 0);
            const std::size_t end = begin + sample_count;
            MoEOverlayParticipantLayerMigrationMeasurement row;
            row.source_participant =
                coordinates_[coordinate_index].source_participant;
            row.destination_participant =
                coordinates_[coordinate_index].destination_participant;
            row.layer = coordinates_[coordinate_index].layer;
            row.wave_wall_nanoseconds = median(std::vector<std::uint64_t>(
                wave_samples_.begin() + static_cast<std::ptrdiff_t>(begin),
                wave_samples_.begin() + static_cast<std::ptrdiff_t>(end)));
            row.wave_sample_count =
                config_.measured_samples_per_coordinate;
            std::array<
                std::uint64_t,
                kExpertHistogramProductionSourceCount>
                phase_medians{};
            for (std::size_t phase = 0;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                if (!config_.required_sources[phase])
                    continue;
                const std::size_t phase_begin = interferenceSampleOffset(
                    coordinate_index, phase, 0);
                phase_medians[phase] = median(std::vector<std::uint64_t>(
                    interference_samples_.begin() +
                        static_cast<std::ptrdiff_t>(phase_begin),
                    interference_samples_.begin() +
                        static_cast<std::ptrdiff_t>(
                            phase_begin + sample_count)));
            }
            /*
             * Placement is shared by all inference phases.  Charge the worst
             * robust phase median so a cheap decode sample cannot conceal a
             * prefill or grouped-verifier contention regression.
             */
            row.inference_interference_nanoseconds = *std::max_element(
                phase_medians.begin(), phase_medians.end());
            const auto required_phase_count = static_cast<std::uint64_t>(
                std::count(
                    config_.required_sources.begin(),
                    config_.required_sources.end(),
                    true));
            row.interference_sample_count =
                config_.measured_samples_per_coordinate *
                required_phase_count;
            phase_interference_medians.push_back(phase_medians);

            for (std::size_t projection = 0; projection < 3; ++projection)
            {
                std::vector<std::uint64_t> wall;
                std::vector<std::uint64_t> device;
                std::vector<std::uint64_t> host;
                std::vector<std::uint64_t> transport;
                wall.reserve(sample_count);
                device.reserve(sample_count);
                host.reserve(sample_count);
                transport.reserve(sample_count);
                for (std::size_t sample = begin; sample < end; ++sample)
                {
                    const auto &value =
                        projection_samples_[projection][sample];
                    wall.push_back(value.wall_nanoseconds);
                    device.push_back(value.device_nanoseconds);
                    host.push_back(value.host_nanoseconds);
                    transport.push_back(value.transport_nanoseconds);
                }
                row.projections[projection] = {
                    .sequence = 1,
                    .bytes = coordinate_state_[coordinate_index]
                                 .projection_bytes[projection],
                    .wall_nanoseconds = median(std::move(wall)),
                    .device_nanoseconds = median(std::move(device)),
                    .host_nanoseconds = median(std::move(host)),
                    .transport_nanoseconds = median(std::move(transport)),
                };
            }
            result.rows.push_back(std::move(row));
        }

        std::uint64_t hash = kFNV1a64OffsetBasis;
        hashString(hash, "MoEOverlayMigrationMeasurementLedger/v2");
        hashString(hash, config_.measurement_identity);
        for (const bool required : config_.required_sources)
            hashUnsigned(hash, required ? 1u : 0u);
        for (std::size_t row_index = 0; row_index < result.rows.size(); ++row_index)
        {
            const auto &row = result.rows[row_index];
            hashUnsigned(hash, static_cast<std::uint64_t>(row.source_participant));
            hashUnsigned(hash, static_cast<std::uint64_t>(row.destination_participant));
            hashUnsigned(hash, static_cast<std::uint64_t>(row.layer));
            hashUnsigned(hash, row.wave_wall_nanoseconds);
            hashUnsigned(hash, row.inference_interference_nanoseconds);
            for (const auto phase_median :
                 phase_interference_medians[row_index])
            {
                hashUnsigned(hash, phase_median);
            }
            for (const auto &projection : row.projections)
            {
                hashUnsigned(hash, projection.bytes);
                hashUnsigned(hash, projection.wall_nanoseconds);
                hashUnsigned(hash, projection.device_nanoseconds);
                hashUnsigned(hash, projection.host_nanoseconds);
                hashUnsigned(hash, projection.transport_nanoseconds);
            }
        }
        std::ostringstream identity;
        identity << "expert-overlay-migration-measurements-v2/"
                 << std::hex << std::setfill('0') << std::setw(16) << hash;
        result.identity = identity.str();
        if (!result.valid())
        {
            throw std::logic_error(
                "ExpertOverlay migration ledger produced an invalid sealed corpus");
        }

        bool expected = false;
        if (!sealed_.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel))
        {
            throw std::logic_error(
                "ExpertOverlay migration ledger raced with another seal operation");
        }
        return result;
    }

    MoEOverlayMigrationMeasurementLedgerStats
    MoEOverlayMigrationMeasurementLedger::stats() const noexcept
    {
        return {
            .completed_waves_observed = completed_waves_observed_.load(
                std::memory_order_relaxed),
            .migration_observations_seen = migration_observations_seen_.load(
                std::memory_order_relaxed),
            .warmup_migration_observations =
                warmup_migration_observations_.load(
                    std::memory_order_relaxed),
            .stored_migration_observations =
                stored_migration_observations_.load(
                    std::memory_order_relaxed),
            .excess_migration_observations =
                excess_migration_observations_.load(
                    std::memory_order_relaxed),
            .warmup_interference_observations =
                warmup_interference_observations_.load(
                    std::memory_order_relaxed),
            .stored_interference_observations =
                stored_interference_observations_.load(
                    std::memory_order_relaxed),
            .excess_interference_observations =
                excess_interference_observations_.load(
                    std::memory_order_relaxed),
            .rejected_observations = rejected_observations_.load(
                std::memory_order_relaxed),
        };
    }
} // namespace llaminar2
