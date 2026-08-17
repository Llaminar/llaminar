/**
 * @file MoEOverlayEconomyProfileComposer.h
 * @brief Builds immutable tier-economy profiles from exact participant evidence.
 *
 * Dynamic ExpertOverlay placement is allowed to move weights only when it can
 * compare the service benefit with the measured movement cost.  Device-facing
 * profilers naturally produce participant rows, while the placement optimizer
 * assigns whole experts to tiers.  This file owns the typed, deterministic
 * boundary between those two representations.  It performs no benchmarking,
 * allocation on an inference stream, or topology inference from names.
 */

#pragma once

#include "ExpertTierTransferMeasurement.h"
#include "MoEOverlayParticipantResidency.h"
#include "MoEOverlayResidencyAuthority.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{
    class MoEOverlayEconomyCalibrationLayerCatalog;

    /**
     * @brief Exact service observation for one participant, layer, and phase.
     *
     * Profilers report already-normalized nanoseconds per routed activation.
     * `sample_count` proves that a positive duration was measured for every
     * production phase rather than filled with a guessed backend constant.
     */
    struct MoEOverlayParticipantLayerServiceMeasurement
    {
        int participant_id = -1;
        int layer = -1;
        std::array<uint64_t, kExpertHistogramProductionSourceCount>
            nanoseconds_per_activation{};
        std::array<uint64_t, kExpertHistogramProductionSourceCount>
            sample_count{};
    };

    /**
     * @brief Complete setup-time evidence consumed by the profile composer.
     *
     * Identities name the exact profiler run and prepared-weight generation.
     * They are included in the resulting lightweight FNV identity, but are not
     * treated as a cryptographic model-file checksum.
     */
    struct MoEOverlayEconomyMeasurements
    {
        std::string service_measurement_identity;
        std::string migration_measurement_identity;
        /** Phases reachable under the runtime policy being certified. */
        ExpertHistogramProductionSourceMask active_sources =
            kAllExpertHistogramProductionSources;
        std::vector<MoEOverlayParticipantLayerServiceMeasurement>
            participant_service;
        std::vector<MoEOverlayParticipantLayerMigrationCost>
            directed_migration;
    };

    /**
     * @brief Robust complete-expert movement observation for one directed edge.
     *
     * Gate, up, and down projections use independent persistent lanes and may
     * overlap. `wave_wall_nanoseconds` therefore measures the complete wave
     * directly; adding projection durations would incorrectly charge parallel
     * work three times. The component observations retain diagnostic evidence
     * for the conversion, DMA, and host-relay portions of that wave.
     */
    struct MoEOverlayParticipantLayerMigrationMeasurement
    {
        int source_participant = -1;
        int destination_participant = -1;
        int layer = -1;
        std::array<ExpertTierProjectionTransferMeasurement, 3> projections;
        /** Median end-to-end gate/up/down wave latency after warmup. */
        std::uint64_t wave_wall_nanoseconds = 0;
        /** Independent complete-wave samples contributing to the median. */
        std::uint64_t wave_sample_count = 0;
        /** Median added inference latency measured during concurrent movement. */
        std::uint64_t inference_interference_nanoseconds = 0;
        /** Paired inference-only/inference-plus-movement samples. */
        std::uint64_t interference_sample_count = 0;
    };

    /**
     * @brief Validated immutable profiles ready for residency-authority sealing.
     *
     * Shared ownership is intentional: a background proposal may retain these
     * values while the runner and profiler objects that produced them are gone.
     */
    struct MoEOverlayCertifiedEconomyProfiles
    {
        std::shared_ptr<const MoERoutedTierServiceProfile> service;
        std::shared_ptr<const MoEOverlayMigrationCostProfile> migration;
        MoEOverlayMigrationEconomyPolicy policy;

        /** @return Whether all immutable components and policy are present. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /**
     * @brief Deterministically certify participant measurements for placement.
     *
     * The composer requires one service row per participant/layer/phase and one
     * migration row per directed participant pair/layer.  Participant service
     * costs are reduced to a tier row with a maximum, because every participant
     * in a tier is a possible whole-expert owner and the slowest endpoint is the
     * safe critical-path cost. Runtime-disabled phases must remain explicitly
     * zero throughout certification; a later non-zero demand in such a phase is
     * therefore a lifecycle error rather than an invented cost.
     */
    class MoEOverlayEconomyProfileComposer final
    {
    public:
        /** Minimum independent samples accepted for a robust setup median. */
        static constexpr std::uint64_t kMinimumMigrationSamples = 3;

        /**
         * @brief Normalize coherent raw endpoint totals with conservative ceil division.
         *
         * The participant accumulator retains integer duration and activation
         * sums so many sub-microsecond GPU observations do not lose precision.
         * Certification rounds each aggregate upward, preserving a non-zero
         * cost for every runtime-active production phase. Disabled phases must
         * contain exact zero totals and remain zero in the normalized result.
         *
         * @param totals One coherent row per participant/layer.
         * @param active_sources Immutable runtime phase availability.
         * @return Canonically ordered normalized rows accepted by @ref compose.
         * @throws std::invalid_argument For empty, overflowed, duplicated, or
         *         partially sampled totals.
         */
        [[nodiscard]] static std::vector<
            MoEOverlayParticipantLayerServiceMeasurement>
        normalizeServiceTotals(
            std::vector<MoEOverlayParticipantLayerServiceTotals> totals,
            ExpertHistogramProductionSourceMask active_sources =
                kAllExpertHistogramProductionSources);

        /**
         * @brief Pool live service totals over exact layer-equivalence classes.
         *
         * One service scalar models a prepared projection contract, while live
         * sparse routing gives individual layers different incidental route
         * counts. Summing integer durations and activations across every layer
         * with the same authenticated manifest contract produces the single
         * through-origin estimator used by placement, then expands that value
         * back to each member. This prevents packet sparsity and timing noise
         * from pretending that identical kernels have different capabilities.
         *
         * @param totals Complete participant/layer geometry. Individual class
         *        members may be coherently empty when sparse traffic did not
         *        route to them, but every participant/class/active-phase pool
         *        must contain at least one real observation.
         * @param active_sources Immutable runtime phase availability.
         * @param catalog Exact manifest-derived partition of model layers.
         * @return Participant/layer rows with one shared cost per exact class.
         * @throws std::invalid_argument For incomplete geometry, overflow,
         *         disabled-phase evidence, or malformed observations.
         */
        [[nodiscard]] static std::vector<
            MoEOverlayParticipantLayerServiceMeasurement>
        normalizeEquivalentServiceTotals(
            std::vector<MoEOverlayParticipantLayerServiceTotals> totals,
            ExpertHistogramProductionSourceMask active_sources,
            const MoEOverlayEconomyCalibrationLayerCatalog &catalog);

        /**
         * @brief Reduce robust gate/up/down observations to optimizer costs.
         *
         * Rows are canonicalized by directed participant pair and layer. The
         * charged transfer/repack cost is the larger of the measured whole-wave
         * latency and any projection latency, never the sum of three lanes that
         * execute concurrently. A zero measured interference delta is valid;
         * its positive paired-sample count proves it was observed rather than
         * omitted.
         *
         * @param measurements Complete normalized calibration observations.
         * @return Canonically ordered directed migration-cost rows.
         * @throws std::invalid_argument For duplicate coordinates, self edges,
         *         incomplete timing evidence, or insufficient robust samples.
         */
        [[nodiscard]] static std::vector<
            MoEOverlayParticipantLayerMigrationCost>
        normalizeMigrationMeasurements(
            std::vector<MoEOverlayParticipantLayerMigrationMeasurement>
                measurements);

        /**
         * @brief Validate, canonicalize, and compose one complete profile pair.
         *
         * @param plan Frozen, model-aware tier plan with explicit priorities.
         * @param metadata Exact routed-expert model geometry.
         * @param owner_map Resolved logical participant topology for the plan.
         * @param measurements Exact setup profiler output.
         * @param policy Smoothing, payoff, and residency policy to certify.
         * @return Immutable service/migration profiles with stable identities.
         * @throws std::invalid_argument when any row, identity, geometry, or
         *         priority relation is incomplete or contradictory.
         */
        [[nodiscard]] static MoEOverlayCertifiedEconomyProfiles compose(
            const MoERoutedExpertPlacementPlan &plan,
            const MoERoutedExpertModelMetadata &metadata,
            const MoEExpertOwnerMap &owner_map,
            MoEOverlayEconomyMeasurements measurements,
            MoEOverlayMigrationEconomyPolicy policy);
    };

} // namespace llaminar2
