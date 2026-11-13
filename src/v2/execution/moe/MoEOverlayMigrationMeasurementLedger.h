/**
 * @file MoEOverlayMigrationMeasurementLedger.h
 * @brief Bounded robust evidence ledger for ExpertOverlay migration economy.
 *
 * Physical migration lanes publish scalar observations and shared immutable
 * original geometry for completed projection pieces. This setup-owned ledger
 * stores a fixed number of warmup and measured samples for every explicitly
 * declared directed complete-target/layer coordinate.  Recording performs no allocation, waiting,
 * device work, or topology inference.  Once its single maintenance producer
 * is quiescent, sealing computes deterministic medians suitable for
 * `MoEOverlayEconomyProfileComposer`.
 */

#pragma once

#include "MoEOverlayEconomyProfileComposer.h"
#include "MoEOverlayTierMigrationTransport.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <compare>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace llaminar2
{
    /** @brief Stable directed endpoint/layer key required by calibration. */
    struct MoEOverlayMigrationMeasurementCoordinate
    {
        int source_participant = -1;
        int destination_participant = -1;
        int layer = -1;

        /** @brief Compare the complete coordinate for canonical ordering. */
        auto operator<=>(
            const MoEOverlayMigrationMeasurementCoordinate &) const = default;

        /** @return Whether this names a non-self directed layer edge. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /** @brief Setup-declared original target geometry; a coordinate alone cannot size a cohort. */
    struct MoEOverlayRequiredTargetMigrationMeasurement
    {
        MoEOverlayMigrationTargetCoordinate coordinate;
        std::shared_ptr<const MoEOverlayMigrationMeasurementGeometry> geometry;
        /** Optional original operation/layout quotes; exact piece order, never inferred from incoming evidence. */
        std::vector<std::uint64_t> quoted_payload_bytes;
    };

    /** @brief Borrowed completed observation; metadata grants no source or readiness authority. */
    struct MoEOverlayBorrowedTargetMigrationObservation
    {
        MoEOverlayMigrationTargetCoordinate coordinate;
        const MoEOverlayCompletedMigrationMeasurement &measurement;
    };

    /** @brief Race-safe proof counters for one calibration ledger. */
    struct MoEOverlayMigrationMeasurementLedgerStats
    {
        std::uint64_t completed_waves_observed = 0;
        std::uint64_t migration_observations_seen = 0;
        std::uint64_t warmup_migration_observations = 0;
        std::uint64_t stored_migration_observations = 0;
        std::uint64_t excess_migration_observations = 0;
        std::uint64_t warmup_interference_observations = 0;
        std::uint64_t stored_interference_observations = 0;
        std::uint64_t excess_interference_observations = 0;
        std::uint64_t rejected_observations = 0;
    };

    /**
     * @brief Immutable robust migration corpus emitted by a sealed ledger.
     *
     * `identity` is a lightweight deterministic digest over the exact
     * coordinate set and median observations.  It is evidence identity, not a
     * cryptographic model checksum.
     */
    struct MoEOverlaySealedMigrationMeasurements
    {
        std::string identity;
        std::vector<MoEOverlayTargetLayerMigrationMeasurement> rows;

        /** @return Checked whole-only projection, never a retained duplicate corpus.
         * @throws std::invalid_argument For any cohort or incomplete row. */
        [[nodiscard]] std::vector<MoEOverlayParticipantLayerMigrationMeasurement> wholeRows() const;

        /** @return Whether identity and every required row are present. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /**
     * @brief Allocation-free sink and robust median builder for calibration.
     *
     * One maintenance worker owns all record calls.  The setup coordinator may
     * inspect `ready()` concurrently because sample counts are published with
     * release ordering.  It must stop or otherwise quiesce that producer before
     * calling `seal()`.  This explicit lifecycle avoids a mutex that could make
     * completed migration publication wait behind diagnostic inspection.
     */
    class MoEOverlayMigrationMeasurementLedger final
        : public IMoEOverlayMigrationMeasurementSink
    {
    public:
        /** @brief Immutable bounded corpus geometry and robust-sampling policy. */
        struct Config
        {
            /** Canonical typed declarations; mutually exclusive with the scalar input adapter. */
            std::vector<MoEOverlayRequiredTargetMigrationMeasurement> required_target_measurements;
            /** Explicit legacy whole-only input adapter; no geometry/cut claim is invented. */
            std::vector<MoEOverlayMigrationMeasurementCoordinate>
                required_coordinates;
            std::uint64_t warmup_samples_per_coordinate = 1;
            std::uint64_t measured_samples_per_coordinate =
                MoEOverlayEconomyProfileComposer::kMinimumMigrationSamples;
            /** Runtime-reachable phases used by subsequent service profiling. */
            ExpertHistogramProductionSourceMask required_sources =
                kAllExpertHistogramProductionSources;
            std::string measurement_identity;
        };

        /**
         * @brief Canonicalize coordinates and preallocate the complete corpus.
         * @param config Exact required edges, sample counts, and setup identity.
         * @throws std::invalid_argument For invalid/duplicate coordinates,
         *         inadequate samples, empty identity, or allocation overflow.
         */
        explicit MoEOverlayMigrationMeasurementLedger(Config config);

        /**
         * @brief Retain complete local projection observations without waiting.
         * @param measurements Transaction-ordered completed migration rows.
         * @param error Optional exact validation or lifecycle diagnostic.
         * @return True when the whole input was accepted atomically.
         *
         * Every input row must contain all three local projection observations.
         * Distributed partial observations are merged into complete rows by the
         * distributed calibration owner before reaching this ledger.
         */
        bool recordCompletedWave(
            const std::vector<MoEOverlayCompletedMigrationMeasurement> &
                measurements,
            std::string *error) noexcept;

        /** @brief Take an already-owned completed wave without copying nested metadata.
         * @param measurements Exact wave-owned observations. @param error Optional rejection reason.
         * @return Whether the complete observation was accepted atomically. */
        bool recordCompletedWave(std::vector<MoEOverlayCompletedMigrationMeasurement> &&measurements,
            std::string *error) noexcept override;

        /** @brief Record one borrowed original-target wave without allocation or geometry inference.
         * @param observations Exact typed coordinates and complete original observations.
         * @param error Optional rejection diagnostic. @return Whether the entire wave was accepted atomically. */
        bool recordCompletedTargetWave(std::span<const MoEOverlayBorrowedTargetMigrationObservation> observations,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Retain one paired inference-only/with-movement latency sample.
         * @param coordinate Exact directed edge and layer exercised.
         * @param source Production phase whose identical workload is paired.
         * @param inference_only_nanoseconds Baseline inference wall time.
         * @param concurrent_nanoseconds Same workload during background movement.
         * @param error Optional validation or lifecycle diagnostic.
         * @return True when the pair was accepted; a faster concurrent sample
         *         records a real zero interference delta rather than underflow.
         */
        bool recordInterferenceSample(
            MoEOverlayMigrationMeasurementCoordinate coordinate,
            ExpertHistogramSource source,
            std::uint64_t inference_only_nanoseconds,
            std::uint64_t concurrent_nanoseconds,
            std::string *error = nullptr) noexcept;

        /** @return Whether every coordinate has robust physical movement samples. */
        [[nodiscard]] bool ready() const noexcept;

        /**
         * @brief Quiescently seal the corpus and compute deterministic medians.
         * @return Immutable rows in canonical coordinate order.
         * @throws std::logic_error If incomplete, already sealed, or recording
         *         raced with the required quiescent setup boundary.
         */
        [[nodiscard]] MoEOverlaySealedMigrationMeasurements seal();

        /** @return Race-safe copy of bounded-ledger proof counters. */
        [[nodiscard]] MoEOverlayMigrationMeasurementLedgerStats stats()
            const noexcept;

        /** @return Canonical number of directed coordinates in this corpus. */
        [[nodiscard]] std::size_t coordinateCount() const noexcept
        {
            return coordinates_.size();
        }

        /**
         * @return Immutable canonical coordinate identities admitted at setup.
         * @note A matching count alone cannot bind a planner to this corpus.
         */
        [[nodiscard]] std::vector<MoEOverlayMigrationMeasurementCoordinate>
        requiredCoordinates() const;

        /** @return The one canonical typed coordinate list admitted at setup. */
        [[nodiscard]] const std::vector<MoEOverlayMigrationTargetCoordinate> &
        requiredTargetCoordinates() const noexcept { return coordinates_; }

        /** @brief Authenticate presealed evidence against original setup declarations without mutation.
         * @param measurements Imported corpus at the setup/quiescent boundary.
         * @param error Optional exact incompatibility diagnostic.
         * @return Exact coordinate/geometry/quote compatibility; cohort reuse requires setup-bound quotes. */
        [[nodiscard]] bool matchesDeclarations(const MoEOverlaySealedMigrationMeasurements &measurements,
            std::string *error = nullptr) const noexcept;

        /** @return Warmup plus retained observations required per coordinate/phase. */
        [[nodiscard]] std::uint64_t requiredObservationsPerCoordinate()
            const noexcept
        {
            return config_.warmup_samples_per_coordinate +
                   config_.measured_samples_per_coordinate;
        }

        /** @return Exact runtime-reachable phase mask sealed by this ledger. */
        [[nodiscard]] const ExpertHistogramProductionSourceMask &
        requiredSources() const noexcept
        {
            return config_.required_sources;
        }

    private:
        /** @brief Per-coordinate release-published progress counters. */
        struct CoordinateState
        {
            std::atomic<std::uint64_t> migration_observations{0};
            std::array<
                std::atomic<std::uint64_t>,
                kExpertHistogramProductionSourceCount>
                interference_observations{};
            std::shared_ptr<const MoEOverlayMigrationMeasurementGeometry> geometry;
            std::size_t first_piece = 0;
            std::size_t piece_count = 0;
            bool setup_quotes_bound = false;
        };

        /** @brief Find one canonical coordinate without allocation. */
        [[nodiscard]] std::size_t coordinateIndex(
            const MoEOverlayMigrationTargetCoordinate &coordinate)
            const noexcept;

        /** @brief Flatten a coordinate and retained sample into fixed storage. */
        [[nodiscard]] std::size_t sampleOffset(
            std::size_t coordinate_index,
            std::uint64_t measured_sample_index) const noexcept;

        /** @brief Flatten coordinate, production phase, and retained sample. */
        [[nodiscard]] std::size_t interferenceSampleOffset(
            std::size_t coordinate_index,
            std::size_t phase_index,
            std::uint64_t measured_sample_index) const noexcept;

        /** @brief Shared adapter implementation; exactly one borrowed input arm is nonempty.
         * @param whole Existing whole-only input adapter. @param targets Explicit borrowed typed inputs.
         * @param error Optional diagnostic. @return Atomic whole-wave acceptance. */
        bool recordWave(std::span<const MoEOverlayCompletedMigrationMeasurement> whole,
            std::span<const MoEOverlayBorrowedTargetMigrationObservation> targets,std::string *error) noexcept;

        /** @brief Flatten one canonical piece and measured sample into setup-owned storage.
         * @param coordinate_index Canonical coordinate. @param piece Canonical piece ordinal.
         * @param measured_index Post-warmup sample ordinal. @return Checked setup-established offset. */
        [[nodiscard]] std::size_t pieceSampleOffset(std::size_t coordinate_index,std::size_t piece,
            std::uint64_t measured_index) const noexcept;

        Config config_;
        std::vector<MoEOverlayMigrationTargetCoordinate> coordinates_;
        std::unique_ptr<CoordinateState[]> coordinate_state_;
        std::vector<std::uint64_t> wave_samples_;
        std::vector<ExpertTierProjectionTransferMeasurement> piece_samples_;
        std::vector<std::uint64_t> piece_bytes_;
        std::vector<std::uint64_t> interference_samples_;
        std::atomic<bool> sealed_{false};

        std::atomic<std::uint64_t> completed_waves_observed_{0};
        std::atomic<std::uint64_t> migration_observations_seen_{0};
        std::atomic<std::uint64_t> warmup_migration_observations_{0};
        std::atomic<std::uint64_t> stored_migration_observations_{0};
        std::atomic<std::uint64_t> excess_migration_observations_{0};
        std::atomic<std::uint64_t> warmup_interference_observations_{0};
        std::atomic<std::uint64_t> stored_interference_observations_{0};
        std::atomic<std::uint64_t> excess_interference_observations_{0};
        std::atomic<std::uint64_t> rejected_observations_{0};
    };
} // namespace llaminar2
