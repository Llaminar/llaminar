/**
 * @file MoEOverlayEconomyCalibrationPlanner.h
 * @brief Deterministic non-publishable swap waves for migration calibration.
 *
 * Every directed complete-target edge must be measured with real prepared model
 * weights and the production physical fabric before economic placement can be
 * certified.  A calibration wave copies one resident expert in each direction
 * between two original whole/cohort targets, preserving shadow capacity as a closed cycle.  It
 * retains the live residency snapshot unchanged and is typed so neither the
 * transport nor the residency authority can commit it.
 */

#pragma once

#include "MoEOverlayMigrationMeasurementLedger.h"
#include "MoEOverlayMigrationTargetCoordinate.h"
#include "MoEOverlayPhysicalResidencyFabric.h"
#include "MoEOverlayParticipantResidency.h"
#include "MoEOverlayResidencyAuthority.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace llaminar2
{
    /**
     * @brief One unmeasured participant/exact-format/production-phase coordinate.
     *
     * A cold expert need never receive an ordinary route. Preparation therefore
     * needs the complete missing set, not a request to keep inferencing until
     * the first missing coordinate happens to execute. Eligible layers contain
     * only members of this exact manifest class priced in this phase; another
     * participant, codebook, geometry, or graph phase cannot supply its price.
     * This is a measurement requirement, never permission to publish residency
     * or routing evidence.
     */
    struct MoEOverlayServiceEvidenceGap
    {
        int participant_id = -1;
        int representative_layer = -1;
        ExpertHistogramSource source = ExpertHistogramSource::SyntheticTest;
        std::vector<int> eligible_layers;
    };

    /**
     * @brief One exact manifest-equivalence class used by economy calibration.
     *
     * Layers may share movement evidence only when all three projection roles,
     * shapes, and original NativeVNNI source identities are byte-for-byte
     * equivalent.  The numerically smallest member is the physical calibration
     * representative; `member_layers` is sorted and includes it.
     */
    struct MoEOverlayEconomyCalibrationLayerGroup
    {
        int representative_layer = -1;
        std::vector<int> member_layers;
        /**
         * Deterministic ordinal strata used by recurring service telemetry.
         *
         * Transfer calibration still needs only @ref representative_layer
         * because byte movement depends solely on the exact prepared-weight
         * contract. Runtime service includes route density and fixed launch
         * cost, so it samples several spatially distributed members instead.
         */
        std::vector<int> service_telemetry_layers;
        /** Conservative exact maximum of CPU/GPU prepared live bytes. */
        std::size_t complete_expert_bytes = 0;

        /** @return Whether this is a sorted non-empty equivalence class. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /**
     * @brief Authenticated partition of model layers by exact weight contract.
     *
     * Large MoE models commonly repeat one projection geometry and codebook
     * across every transformer layer.  Re-running the same physical lane
     * calibration for every address would add setup latency without adding a
     * distinct transfer contract.  This catalog derives equivalence strictly
     * from the GGUF-owned layer manifest, measures one real representative per
     * class for physical transfer calibration. Runtime service uses a bounded
     * deterministic stratified sample within each class so one unusually sparse
     * layer cannot turn fixed GPU launch cost into a false per-activation price.
     * Both evidence forms are expanded back to every exact member layer before
     * the complete economy profile is certified.
     */
    class MoEOverlayEconomyCalibrationLayerCatalog final
    {
    public:
        /**
         * @brief Build a complete exact partition from contiguous layer metadata.
         * @param manifest Authenticated gate/up/down contracts in layer order.
         * @throws std::invalid_argument For malformed or non-contiguous input.
         * @throws std::overflow_error When a complete expert footprint overflows.
         */
        explicit MoEOverlayEconomyCalibrationLayerCatalog(
            const std::vector<MoEOverlayLayerWeightManifest> &manifest);

        /** @return Stable lightweight identity over the complete partition. */
        [[nodiscard]] const std::string &identity() const noexcept
        {
            return identity_;
        }

        /** @return Canonical exact-equivalence classes in representative order. */
        [[nodiscard]] const std::vector<
            MoEOverlayEconomyCalibrationLayerGroup> &
        groups() const noexcept
        {
            return groups_;
        }

        /** @return One physical-transfer representative per exact class. */
        [[nodiscard]] const std::vector<int> &representativeLayers()
            const noexcept
        {
            return representative_layers_;
        }

        /**
         * @brief Test whether one layer is this class partition's representative.
         * @param layer Exact zero-based model layer index.
         * @return True only for the canonical member selected for physical
         *         migration calibration of its exact weight-equivalence class.
         */
        [[nodiscard]] bool isRepresentativeLayer(int layer) const noexcept;

        /**
         * @return Sorted union of deterministic service-telemetry strata.
         *
         * A class of N exact-equivalent layers contributes ceil(sqrt(N))
         * members, one from the midpoint of each equal ordinal stratum. This
         * gives sublinear graph-marker cost while sampling the full model depth.
         */
        [[nodiscard]] const std::vector<int> &serviceTelemetryLayers()
            const noexcept
        {
            return service_telemetry_layers_;
        }

        /**
         * @brief Test whether one layer carries recurring service telemetry.
         * @param layer Exact zero-based model layer index.
         * @return True only for a deterministic stratum representative.
         */
        [[nodiscard]] bool isServiceTelemetryLayer(int layer) const noexcept;

        /** @return Number of model layers covered by the exact partition. */
        [[nodiscard]] std::size_t layerCount() const noexcept
        {
            return complete_expert_bytes_per_layer_.size();
        }

        /** @return First global model layer in this catalog's compact interval. */
        [[nodiscard]] int firstModelLayer() const noexcept
        {
            return first_model_layer_;
        }

        /** @return Conservative complete-expert byte footprints for every layer. */
        [[nodiscard]] const std::vector<std::size_t> &
        completeExpertBytesPerLayer() const noexcept
        {
            return complete_expert_bytes_per_layer_;
        }

        /**
         * @brief Enumerate every missing exact service price without mutating evidence.
         *
         * Preparation and certification must share this query. One real sample
         * in an eligible equivalent layer closes that class/phase coordinate;
         * samples from a different participant or manifest class never do.
         * Empty participant sets are legal for relay-only ranks.
         *
         * @param rows Complete participant-major, layer-minor cumulative rows.
         * @param participant_ids Strictly increasing process/global participant IDs.
         * @param topology Canonical graph reachability and economy-priced phases.
         * @return Missing coordinates in participant/class/phase order; empty
         *         means complete, not that missing prices were assigned zero.
         * @throws std::invalid_argument For incomplete, unordered, overflowed,
         *         or phase-incompatible evidence or mismatched topology.
         */
        [[nodiscard]] std::vector<MoEOverlayServiceEvidenceGap>
        serviceEvidenceGaps(
            const std::vector<MoEOverlayParticipantLayerServiceTotals> &rows,
            const std::vector<int> &participant_ids,
            const ExpertHistogramProductionTopology &topology) const;

        /**
         * @brief Complete unobserved classes with independent prepared-kernel measurements.
         *
         * Both inputs remain immutable. This produces certification evidence,
         * never a cumulative runtime counter import. Live observations own a
         * class/phase whenever present; otherwise only measured probe rows of
         * that exact participant, manifest class and priced phase are admitted.
         * An unmeasured coordinate remains missing, not a zero-cost estimate.
         *
         * @param live Complete canonical snapshot of ordinary service counters.
         * @param prepared Complete canonical matrix of bounded setup observations.
         * @param participant_ids Exact sorted participants represented by both matrices.
         * @param topology Canonical graph reachability and economy-priced phases.
         * @return Independent combined evidence, still subject to the coverage gate.
         * @throws std::invalid_argument For malformed, overflowed or incompatible rows.
         */
        [[nodiscard]] std::vector<MoEOverlayParticipantLayerServiceTotals>
        withPreparedServiceEvidence(
            const std::vector<MoEOverlayParticipantLayerServiceTotals> &live,
            const std::vector<MoEOverlayParticipantLayerServiceTotals> &prepared,
            const std::vector<int> &participant_ids,
            const ExpertHistogramProductionTopology &topology) const;

        /**
         * @brief Expand representative-only robust rows to every exact member.
         * @param representative_measurements Sealed complete directed rows for
         *        every representative and observed endpoint pair.
         * @return A new sealed corpus containing the full layer matrix.
         * @throws std::invalid_argument For missing, duplicate, non-representative,
         *         or otherwise incomplete representative evidence.
         */
        [[nodiscard]] MoEOverlaySealedMigrationMeasurements expand(
            const MoEOverlaySealedMigrationMeasurements &
                representative_measurements) const;

        /**
         * @brief Expand measured target rows only across exact original source geometry.
         * @param measurements Physically measured rows, never already projected evidence.
         * @param snapshot Original sealed source placement retaining cohort manifests and cuts.
         * @return Full target/layer corpus with explicit measured-coordinate provenance.
         * @throws std::invalid_argument For missing samples or changed members, cuts or source formats.
         */
        [[nodiscard]] MoEOverlaySealedMigrationMeasurements expand(
            const MoEOverlaySealedMigrationMeasurements &measurements,
            std::shared_ptr<const MoEOverlayResidencySnapshot> snapshot) const;

    private:
        /**
         * @brief Translate an owned global layer into compact catalog storage.
         * @param layer Global model-layer ID from a manifest or evidence row.
         * @return Index within this stage's layer arrays.
         * @throws std::invalid_argument For a layer outside the owned interval.
         */
        [[nodiscard]] std::size_t storageIndexForModelLayer(int layer) const;

        /** @brief Shared expansion implementation for source-bound targets and the checked whole adapter.
         * @param measurements Original measured corpus. @param snapshot Original source, or null for legacy whole rows.
         * @return Fully covered corpus preserving each original measurement coordinate.
         */
        [[nodiscard]] MoEOverlaySealedMigrationMeasurements expandOriginalMeasurements(
            const MoEOverlaySealedMigrationMeasurements &measurements,
            std::shared_ptr<const MoEOverlayResidencySnapshot> snapshot) const;

        int first_model_layer_ = 0;
        std::string identity_;
        std::vector<MoEOverlayEconomyCalibrationLayerGroup> groups_;
        std::vector<int> representative_layers_;
        std::vector<int> service_telemetry_layers_;
        std::vector<std::size_t> complete_expert_bytes_per_layer_;
    };

    /**
     * @brief Builds exact pairwise calibration transactions from one live epoch.
     *
     * The planner does not infer preference from tier names or device types.
     * Promotion/demotion labels use only explicit integer tier priorities.
     */
    class MoEOverlayEconomyCalibrationPlanner final
    {
    public:
        /** @brief Immutable live topology and complete-expert layer footprints. */
        struct Config
        {
            std::shared_ptr<const MoEOverlayResidencySnapshot> live_snapshot;
            std::vector<std::size_t> complete_expert_bytes_per_layer;
            /**
             * Exact layer representatives to exercise. Empty means every layer.
             * Production supplies the authenticated catalog representatives;
             * the all-layer form remains useful for focused protocol fixtures.
             */
            std::vector<int> calibration_layers;
        };

        /**
         * @brief Validate source availability and retain one immutable epoch.
         * @param config Live snapshot plus one positive byte footprint per layer.
         * @throws std::invalid_argument When any admitted target/layer has no source
         *         expert or the model geometry/byte contract is incomplete.
         */
        explicit MoEOverlayEconomyCalibrationPlanner(Config config);

        /** @return The single canonical directed complete-target/layer planning list. */
        [[nodiscard]] const std::vector<MoEOverlayMigrationTargetCoordinate> &
        requiredTargetCoordinates() const noexcept { return required_target_coordinates_; }

        /**
         * @brief Describe the exact original prepared-wave geometry for the typed ledger.
         * @param manifests Complete source-authenticated layer manifests from the original loader.
         * @return One declaration per canonical coordinate, without fabricated operation quotes or readiness.
         * @throws std::invalid_argument For missing, duplicate or foreign layer geometry.
         */
        [[nodiscard]] std::vector<MoEOverlayRequiredTargetMigrationMeasurement> requiredTargetMeasurements(
            std::span<const MoEOverlayLayerWeightManifest> manifests) const;

        /**
         * @brief Project the canonical list for existing whole-owner evidence consumers.
         * @return Whole participant/layer coordinates in historical order.
         * @throws std::logic_error If any coordinate is a cohort; no member substitutes for it.
         * @note This checked adapter is retired when shared consumers migrate to typed targets.
         * No second planning list is retained.
         */
        [[nodiscard]] std::vector<MoEOverlayMigrationMeasurementCoordinate> requiredCoordinates() const;

        /**
         * @brief Build the same reciprocal shadow wave for complete original targets.
         * @param source_target Original whole participant or fixed cohort tag.
         * @param destination_target Different complete target retained by the live snapshot.
         * @param layer Global source layer selected by the original calibration catalog.
         * @param calibration_sequence Positive unique sample identity.
         * @return Genuine source-bound, non-publishable calibration transaction.
         * @throws std::invalid_argument For an absent/foreign/member alias or unavailable source.
         */
        [[nodiscard]] MoEOverlayResidencyTransaction buildPairSwap(
            MoEOverlayRouteTarget source_target, MoEOverlayRouteTarget destination_target,
            int layer, std::uint64_t calibration_sequence) const;

        /**
         * @brief Checked whole-owner adapter forwarding to the complete-target shadow builder.
         * @param source_participant First endpoint in the pair.
         * @param destination_participant Second endpoint in the pair.
         * @param layer Exact model layer whose prepared geometry is exercised.
         * @param calibration_sequence Positive unique sample identity.
         * @return Structurally valid non-publishable two-migration transaction.
         * @throws std::invalid_argument For an unknown/self endpoint, layer, or
         *         zero sequence.
         */
        [[nodiscard]] MoEOverlayResidencyTransaction buildPairSwap(
            int source_participant,
            int destination_participant,
            int layer,
            std::uint64_t calibration_sequence) const;

    private:
        /**
         * @brief Resolve one original resident expert without inventing source geometry.
         * @param target Checked route alternative, never a scalar cohort member.
         * @param layer Original model-global layer.
         * @return First canonical source expert at the complete original target.
         * @throws std::invalid_argument For foreign/absent/empty targets.
         */
        [[nodiscard]] const MoEExpertOwner &sourceOwner(MoEOverlayRouteTarget target, int layer) const;

        /** @brief Classify movement solely through configured integer priority. */
        [[nodiscard]] MoEOverlayTierMigrationDirection direction(
            int source_tier,
            int destination_tier) const;

        Config config_;
        std::vector<int> calibration_layers_;
        std::vector<MoEOverlayMigrationTargetCoordinate> required_target_coordinates_;
    };
} // namespace llaminar2
