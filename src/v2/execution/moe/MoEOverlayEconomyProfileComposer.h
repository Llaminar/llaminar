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
#include "MoEOverlayMigrationTargetCoordinate.h"
#include "MoEOverlayTierMigrationTransport.h"
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
     * economy-priced production phase rather than filled with a guessed
     * backend constant. Reachable but exceptional phases remain exact zero.
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
        /** Graph reachability and its economy-priced recurring-service subset. */
        ExpertHistogramProductionTopology production_topology;
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
        /** Reserved zero: runtime contention is governed by maintenance budgets. */
        std::uint64_t inference_interference_nanoseconds = 0;
        /** Reserved zero: startup performs no paired synthetic inference. */
        std::uint64_t interference_sample_count = 0;
    };

    /** @brief One independently quoted robust median at its original canonical piece ordinal. */
    struct MoEOverlayQuotedPieceMigrationMeasurement
    {
        std::uint64_t quoted_payload_bytes = 0;
        ExpertTierProjectionTransferMeasurement measurement;
        /** @return Whether the median retains the exact original bound byte extent and real work. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /**
     * @brief One complete-target robust row, retaining every original physical piece.
     *
     * Geometry is passive source metadata, never an engine/source/publication permit.
     * Only explicit legacy whole adapters may omit geometry, retaining the original
     * three-role scalar contract without claiming cuts or source-format evidence.
     */
    struct MoEOverlayTargetLayerMigrationMeasurement
    {
        MoEOverlayMigrationTargetCoordinate coordinate;
        std::shared_ptr<const MoEOverlayMigrationMeasurementGeometry> geometry;
        std::vector<MoEOverlayQuotedPieceMigrationMeasurement> pieces;
        std::uint64_t wave_wall_nanoseconds = 0;
        std::uint64_t wave_sample_count = 0;
        std::uint64_t inference_interference_nanoseconds = 0;
        std::uint64_t interference_sample_count = 0;
        /** @brief Check typed target identity against original passive geometry.
         * @param coordinate Explicit original targets and model layer. @param geometry Checked original geometry.
         * @return Whether both target alternatives and physical member identities agree. */
        [[nodiscard]] static bool geometryMatches(const MoEOverlayMigrationTargetCoordinate &coordinate,
            const MoEOverlayMigrationMeasurementGeometry &geometry) noexcept;
        /** @return Whether original target geometry and complete robust piece evidence agree. */
        [[nodiscard]] bool valid() const noexcept;
        /** @return Checked singleton legacy row. @throws std::invalid_argument For any cohort target. */
        [[nodiscard]] MoEOverlayParticipantLayerMigrationMeasurement whole() const;
        /** @brief Explicitly import an existing whole-only robust row without inventing geometry.
         * @param row Original whole-only evidence. @return Checked typed whole row.
         * @throws std::invalid_argument For incomplete or invalid scalar evidence. */
        [[nodiscard]] static MoEOverlayTargetLayerMigrationMeasurement fromWhole(
            const MoEOverlayParticipantLayerMigrationMeasurement &row);
    };

    /** @brief Complete-target movement price; no service or controller capability is implied. */
    struct MoEOverlayTargetLayerMigrationCost
    {
        MoEOverlayMigrationTargetCoordinate coordinate;
        std::uint64_t transfer_and_repack_ns = 0;
        std::uint64_t inference_interference_ns = 0;
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
     * costs are retained exactly for live critical-path admission and are also
     * reduced to a tier row with a maximum for conservative cold-start capacity
     * placement. Graph-unreachable phases reject service observations and
     * demand. Reachable but economy-unpriced phases may be observed (for
     * example fixed-MTP terminal catch-up), yet remain zero in the certified
     * profile so exceptional traffic cannot distort recurring economics.
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
         * cost for every economy-priced production phase. Graph-unreachable
         * phases must contain exact zero totals. Reachable but unpriced phases
         * may contain real observations but remain zero in the normalized
         * result.
         *
         * @param totals One coherent row per participant/layer.
         * @param production_topology Immutable reachability and economy mask.
         * @return Canonically ordered normalized rows accepted by @ref compose.
         * @throws std::invalid_argument For empty, overflowed, duplicated, or
         *         partially sampled totals.
         */
        [[nodiscard]] static std::vector<
            MoEOverlayParticipantLayerServiceMeasurement>
        normalizeServiceTotals(
            std::vector<MoEOverlayParticipantLayerServiceTotals> totals,
            const ExpertHistogramProductionTopology &production_topology);

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
         *        route to them, but every participant/class/economy-phase pool
         *        must contain at least one real observation.
         * @param production_topology Immutable reachability and economy mask.
         * @param catalog Exact manifest-derived partition of model layers.
         * @return Participant/layer rows with one shared cost per exact class.
         * @throws std::invalid_argument For incomplete geometry, overflow,
         *         unreachable-phase evidence, or malformed observations.
         */
        [[nodiscard]] static std::vector<
            MoEOverlayParticipantLayerServiceMeasurement>
        normalizeEquivalentServiceTotals(
            std::vector<MoEOverlayParticipantLayerServiceTotals> totals,
            const ExpertHistogramProductionTopology &production_topology,
            const MoEOverlayEconomyCalibrationLayerCatalog &catalog);

        /**
         * @brief Reduce robust gate/up/down observations to optimizer costs.
         *
         * Rows are canonicalized by directed participant pair and layer. The
         * charged transfer/repack cost is the larger of the measured whole-wave
         * latency and any projection latency, never the sum of three lanes that
         * execute concurrently. Startup interference fields remain reserved
         * zero; runtime contention is governed by existing maintenance budgets,
         * not synthetic paired-inference samples.
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

        /** @brief Normalize complete original-target observations using the unchanged wave maximum.
         * @param measurements Complete typed rows with all original pieces.
         * @return Canonical typed directed prices, never summed across concurrent pieces.
         * @throws std::invalid_argument For incomplete, duplicate, or foreign geometry/evidence. */
        [[nodiscard]] static std::vector<MoEOverlayTargetLayerMigrationCost>
        normalizeMigrationMeasurements(std::vector<MoEOverlayTargetLayerMigrationMeasurement> measurements);

        /** @brief Project typed prices at an explicitly whole-only profile consumer.
         * @param costs Complete typed prices. @return Existing canonical whole-owner costs.
         * @throws std::invalid_argument For any cohort or invalid/duplicate coordinate. */
        [[nodiscard]] static std::vector<MoEOverlayParticipantLayerMigrationCost>
        checkedWholeMigrationCosts(std::vector<MoEOverlayTargetLayerMigrationCost> costs);

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
