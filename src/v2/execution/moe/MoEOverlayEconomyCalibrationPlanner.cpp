/**
 * @file MoEOverlayEconomyCalibrationPlanner.cpp
 * @brief Exact closed-cycle construction for non-publishable calibration waves.
 *
 * The authenticated layer catalog owns both transfer equivalence and service
 * coverage. Sparse routing cannot guarantee a sample of every cold format, so
 * preparation can enumerate missing service coordinates without fabricating
 * costs or changing the live histogram/residency epoch.
 */

#include "MoEOverlayEconomyCalibrationPlanner.h"

#include "MoEOverlayCapacityResolver.h"
#include "../../utils/FNV1a.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @brief Hash a scalar in a platform-independent little-endian form. */
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

        /** @brief Hash a length-delimited string into one evidence identity. */
        void hashString(
            std::uint64_t &hash,
            const std::string &value) noexcept
        {
            hashUnsigned(hash, static_cast<std::uint64_t>(value.size()));
            hash = fnv1a64(value.data(), value.size(), hash);
        }

        /** @brief Render one inexpensive deterministic catalog digest. */
        std::string catalogIdentity(std::uint64_t hash)
        {
            std::ostringstream output;
            output << "expert-overlay-calibration-layer-catalog-v1/"
                   << std::hex << std::setfill('0') << std::setw(16) << hash;
            return output.str();
        }

        /** @brief Guard vector-size multiplication used by evidence expansion. */
        std::size_t checkedProduct(
            std::size_t left,
            std::size_t right,
            const char *description)
        {
            if (left != 0 &&
                right > std::numeric_limits<std::size_t>::max() / left)
            {
                throw std::invalid_argument(
                    std::string("ExpertOverlay calibration ") + description +
                    " overflows host size");
            }
            return left * right;
        }

        /**
         * @brief Return ceil(sqrt(value)) using exact integer arithmetic.
         *
         * Layer counts are small, but avoiding floating point keeps catalog
         * identity and graph topology identical on every rank and host ISA.
         */
        std::size_t ceilingSquareRoot(std::size_t value) noexcept
        {
            if (value == 0u)
                return 0u;
            std::size_t root = 1u;
            while (root < value / root ||
                   (root == value / root && value % root != 0u))
            {
                ++root;
            }
            return root;
        }

        /**
         * @brief Select one midpoint from each equal ordinal layer stratum.
         *
         * A remainder accumulator distributes non-divisible members without a
         * potentially overflowing `stratum * layer_count` product.
         */
        std::vector<int> stratifiedServiceLayers(
            const std::vector<int> &members)
        {
            const std::size_t sample_count = ceilingSquareRoot(members.size());
            std::vector<int> selected;
            selected.reserve(sample_count);
            const std::size_t base_width = members.size() / sample_count;
            const std::size_t remainder = members.size() % sample_count;
            std::size_t remainder_accumulator = 0u;
            std::size_t first_ordinal = 0u;
            for (std::size_t stratum = 0u; stratum < sample_count; ++stratum)
            {
                std::size_t width = base_width;
                remainder_accumulator += remainder;
                if (remainder_accumulator >= sample_count)
                {
                    ++width;
                    remainder_accumulator -= sample_count;
                }
                const std::size_t ordinal = first_ordinal + width / 2u;
                selected.push_back(members[ordinal]);
                first_ordinal += width;
            }
            return selected;
        }
    } // namespace

    bool MoEOverlayEconomyCalibrationLayerGroup::valid() const noexcept
    {
        return representative_layer >= 0 && complete_expert_bytes > 0 &&
               !member_layers.empty() &&
               member_layers.front() == representative_layer &&
               std::is_sorted(member_layers.begin(), member_layers.end()) &&
               std::adjacent_find(
                   member_layers.begin(), member_layers.end()) ==
                   member_layers.end() &&
               !service_telemetry_layers.empty() &&
               std::is_sorted(
                   service_telemetry_layers.begin(),
                   service_telemetry_layers.end()) &&
               std::adjacent_find(
                   service_telemetry_layers.begin(),
                   service_telemetry_layers.end()) ==
                   service_telemetry_layers.end() &&
               std::all_of(
                   service_telemetry_layers.begin(),
                   service_telemetry_layers.end(),
                   [this](int layer)
                   {
                       return std::binary_search(
                           member_layers.begin(), member_layers.end(), layer);
                   });
    }

    MoEOverlayEconomyCalibrationLayerCatalog::
        MoEOverlayEconomyCalibrationLayerCatalog(
            const std::vector<MoEOverlayLayerWeightManifest> &manifest)
    {
        if (manifest.empty())
        {
            throw std::invalid_argument(
                "ExpertOverlay calibration layer catalog requires a manifest");
        }

        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(manifest);
        first_model_layer_ = manifest.front().layer_idx;
        complete_expert_bytes_per_layer_.resize(manifest.size());
        for (std::size_t layer = 0; layer < manifest.size(); ++layer)
        {
            // preparedFootprints authenticated the complete contiguous interval.
            // Catalog members retain model-global IDs; byte arrays stay compact.
            const int model_layer = manifest[layer].layer_idx;
            const std::size_t complete_bytes = std::max(
                footprints[layer].cpu_live_bytes,
                footprints[layer].gpu_live_bytes);
            if (complete_bytes == 0)
            {
                throw std::invalid_argument(
                    "ExpertOverlay calibration layer has a zero complete-expert footprint");
            }
            complete_expert_bytes_per_layer_[layer] = complete_bytes;

            auto group = std::find_if(
                groups_.begin(),
                groups_.end(),
                [this, &manifest, layer](const auto &candidate)
                {
                    return manifest[storageIndexForModelLayer(
                                        candidate.representative_layer)]
                               .projections == manifest[layer].projections;
                });
            if (group == groups_.end())
            {
                groups_.push_back({
                    .representative_layer = model_layer,
                    .member_layers = {model_layer},
                    .service_telemetry_layers = {},
                    .complete_expert_bytes = complete_bytes,
                });
            }
            else
            {
                if (group->complete_expert_bytes != complete_bytes)
                {
                    throw std::logic_error(
                        "Manifest-equivalent ExpertOverlay layers produced different exact footprints");
                }
                group->member_layers.push_back(model_layer);
            }
        }

        representative_layers_.reserve(groups_.size());
        std::uint64_t hash = kFNV1a64OffsetBasis;
        hashUnsigned(hash, static_cast<std::uint64_t>(manifest.size()));
        hashUnsigned(hash, static_cast<std::uint64_t>(groups_.size()));
        for (auto &group : groups_)
        {
            group.service_telemetry_layers =
                stratifiedServiceLayers(group.member_layers);
            if (!group.valid())
            {
                throw std::logic_error(
                    "ExpertOverlay calibration constructed an invalid layer equivalence class");
            }
            representative_layers_.push_back(group.representative_layer);
            service_telemetry_layers_.insert(
                service_telemetry_layers_.end(),
                group.service_telemetry_layers.begin(),
                group.service_telemetry_layers.end());
            hashUnsigned(
                hash,
                static_cast<std::uint64_t>(group.representative_layer));
            hashUnsigned(
                hash,
                static_cast<std::uint64_t>(group.complete_expert_bytes));
            hashUnsigned(
                hash,
                static_cast<std::uint64_t>(group.member_layers.size()));
            for (const int member : group.member_layers)
                hashUnsigned(hash, static_cast<std::uint64_t>(member));

            const auto &representative = manifest[storageIndexForModelLayer(
                group.representative_layer)];
            for (const auto &projection : representative.projections)
            {
                hashUnsigned(
                    hash,
                    static_cast<std::uint64_t>(projection.projection));
                hashUnsigned(hash, static_cast<std::uint64_t>(projection.N));
                hashUnsigned(hash, static_cast<std::uint64_t>(projection.K));
                hashUnsigned(
                    hash,
                    static_cast<std::uint8_t>(projection.format.kind));
                hashUnsigned(
                    hash,
                    projection.format.native_vnni.codebook_id);
                hashUnsigned(
                    hash,
                    projection.format.native_vnni.is_superblock ? 1u : 0u);
            }
        }
        std::sort(
            service_telemetry_layers_.begin(),
            service_telemetry_layers_.end());
        if (std::adjacent_find(
                service_telemetry_layers_.begin(),
                service_telemetry_layers_.end()) !=
            service_telemetry_layers_.end())
        {
            throw std::logic_error(
                "ExpertOverlay service telemetry strata overlap across exact layer classes");
        }
        identity_ = catalogIdentity(hash);
    }

    std::size_t MoEOverlayEconomyCalibrationLayerCatalog::storageIndexForModelLayer(
        int layer) const
    {
        if (layer < first_model_layer_ ||
            static_cast<std::size_t>(layer - first_model_layer_) >= layerCount())
            throw std::invalid_argument(
                "ExpertOverlay calibration layer is outside its owned interval");
        return static_cast<std::size_t>(layer - first_model_layer_);
    }

    bool MoEOverlayEconomyCalibrationLayerCatalog::isRepresentativeLayer(
        int layer) const noexcept
    {
        return std::binary_search(
            representative_layers_.begin(),
            representative_layers_.end(),
            layer);
    }

    bool MoEOverlayEconomyCalibrationLayerCatalog::isServiceTelemetryLayer(
        int layer) const noexcept
    {
        return std::binary_search(
            service_telemetry_layers_.begin(),
            service_telemetry_layers_.end(),
            layer);
    }

    std::vector<MoEOverlayServiceEvidenceGap>
    MoEOverlayEconomyCalibrationLayerCatalog::serviceEvidenceGaps(
        const std::vector<MoEOverlayParticipantLayerServiceTotals> &rows,
        const std::vector<int> &participant_ids,
        const ExpertHistogramProductionTopology &topology) const
    {
        if (!topology.valid() || topology.layerCount() != layerCount() ||
            topology.firstModelLayer() != firstModelLayer() ||
            !std::is_sorted(participant_ids.begin(), participant_ids.end()) ||
            std::adjacent_find(participant_ids.begin(), participant_ids.end()) !=
                participant_ids.end() ||
            (!participant_ids.empty() && participant_ids.front() < 0) ||
            rows.size() != checkedProduct(
                participant_ids.size(), layerCount(), "service matrix"))
        {
            throw std::invalid_argument(
                "ExpertOverlay service coverage requires canonical participants, complete rows, and exact graph topology");
        }

        // Validate the entire snapshot before returning even its first gap.
        // Otherwise a cold early coordinate could conceal corrupt later rows.
        for (std::size_t participant = 0; participant < participant_ids.size();
             ++participant)
        {
            const std::size_t base = participant * layerCount();
            for (std::size_t layer = 0; layer < layerCount(); ++layer)
            {
                const auto &row = rows[base + layer];
                const int model_layer = first_model_layer_ + static_cast<int>(layer);
                if (!row.valid() || row.participant_id != participant_ids[participant] ||
                    row.layer != model_layer)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay service snapshot is malformed, overflowed, or non-canonical");
                }
                for (std::size_t phase = 0;
                     phase < kExpertHistogramProductionSourceCount; ++phase)
                {
                    if (!topology.reachable(model_layer, phase) &&
                        row.sample_count[phase] != 0)
                    {
                        std::ostringstream message;
                        message << "ExpertOverlay service snapshot sampled a runtime-disabled inference phase"
                                << " participant=" << row.participant_id
                                << " layer=" << model_layer << " source_index=" << phase;
                        throw std::invalid_argument(message.str());
                    }
                }
            }
        }

        constexpr std::array sources{
            ExpertHistogramSource::DecodeToken,
            ExpertHistogramSource::PrefillChunk,
            ExpertHistogramSource::GroupedVerifier};
        std::vector<MoEOverlayServiceEvidenceGap> gaps;
        for (std::size_t participant = 0; participant < participant_ids.size();
             ++participant)
        {
            const std::size_t base = participant * layerCount();
            for (const auto &group : groups_)
            {
                for (const auto source : sources)
                {
                    const auto phase = expertHistogramProductionSourceIndex(source);
                    MoEOverlayServiceEvidenceGap gap{
                        .participant_id = participant_ids[participant],
                        .representative_layer = group.representative_layer,
                        .source = source,
                    };
                    bool observed = false;
                    for (const int layer : group.member_layers)
                    {
                        if (!topology.requiresServiceEvidence(layer, phase))
                            continue;
                        gap.eligible_layers.push_back(layer);
                        observed |= rows[base + storageIndexForModelLayer(layer)]
                                        .sample_count[phase] != 0;
                    }
                    // Reachable-but-unpriced catch-up graphs cannot create a
                    // measurement obligation or satisfy a priced phase.
                    if (!observed && !gap.eligible_layers.empty())
                        gaps.push_back(std::move(gap));
                }
            }
        }
        return gaps;
    }

    std::vector<MoEOverlayParticipantLayerServiceTotals>
    MoEOverlayEconomyCalibrationLayerCatalog::withPreparedServiceEvidence(
        const std::vector<MoEOverlayParticipantLayerServiceTotals> &live,
        const std::vector<MoEOverlayParticipantLayerServiceTotals> &prepared,
        const std::vector<int> &participant_ids,
        const ExpertHistogramProductionTopology &topology) const
    {
        const auto missing = serviceEvidenceGaps(live, participant_ids, topology);
        // Validate all setup rows even when live evidence already covers every
        // class. Irrelevance to this profile cannot excuse corrupt provenance.
        (void)serviceEvidenceGaps(prepared, participant_ids, topology);
        auto combined = live;
        for (const auto &gap : missing)
        {
            const auto participant = std::lower_bound(
                participant_ids.begin(), participant_ids.end(), gap.participant_id);
            const auto base = static_cast<std::size_t>(participant - participant_ids.begin()) * layerCount();
            const auto phase = expertHistogramProductionSourceIndex(gap.source);
            for (const int layer : gap.eligible_layers)
            {
                const auto index = base + storageIndexForModelLayer(layer);
                const auto &probe = prepared[index];
                if (probe.sample_count[phase] == 0) continue;
                // The coverage query proved there are no live samples anywhere
                // in this exact class/phase. Copy measured totals rather than
                // adding them to device-authored cumulative counters.
                combined[index].total_nanoseconds[phase] = probe.total_nanoseconds[phase];
                combined[index].activation_count[phase] = probe.activation_count[phase];
                combined[index].sample_count[phase] = probe.sample_count[phase];
            }
        }
        return combined;
    }

    MoEOverlaySealedMigrationMeasurements
    MoEOverlayEconomyCalibrationLayerCatalog::expand(
        const MoEOverlaySealedMigrationMeasurements &measurements) const
    {
        return expandOriginalMeasurements(measurements, {});
    }

    MoEOverlaySealedMigrationMeasurements
    MoEOverlayEconomyCalibrationLayerCatalog::expand(
        const MoEOverlaySealedMigrationMeasurements &measurements,
        std::shared_ptr<const MoEOverlayResidencySnapshot> snapshot) const
    {
        if (!snapshot || !snapshot->valid() ||
            snapshot->firstModelLayer() != firstModelLayer() ||
            snapshot->layerCount() != static_cast<int>(layerCount()))
            throw std::invalid_argument("Target expansion requires the original sealed cohort source and exact layer interval");
        if (snapshot->ownershipKind() != MoEOverlayResidencyOwnershipKind::CPUJointFragments)
            return expandOriginalMeasurements(measurements, {});
        if (!snapshot->originalProjectionPreparation())
            throw std::invalid_argument("Target expansion lost its original cohort source preparation");
        return expandOriginalMeasurements(measurements, std::move(snapshot));
    }

    MoEOverlaySealedMigrationMeasurements
    MoEOverlayEconomyCalibrationLayerCatalog::expandOriginalMeasurements(
        const MoEOverlaySealedMigrationMeasurements &measurements,
        std::shared_ptr<const MoEOverlayResidencySnapshot> snapshot) const
    {
        if (!measurements.valid() || groups_.empty() || identity_.empty())
            throw std::invalid_argument("Calibration expansion requires valid original measurements and catalog");

        using Coordinate = MoEOverlayMigrationTargetCoordinate;
        std::map<Coordinate, std::shared_ptr<const MoEOverlayMigrationMeasurementGeometry>> original_geometry;
        std::vector<Coordinate> required;
        if (snapshot)
        {
            // The existing planner and fabric geometry builder remain the ONLY
            // target/intersection authorities. buildShardPlacement uses one
            // fixed plan-level member order per tier and balancedIntervals for
            // each source. That rule depends only on F, source block and width;
            // exact catalog projection classes therefore have identical cuts.
            // Validate geometry anyway: arbitrary passive cuts are NOT another
            // admitted scheduler or a reason to invent additional samples.
            // Every layer's original cohort
            // retains its complete source manifest, including whole endpoint roles.
            std::vector<MoEOverlayLayerWeightManifest> manifests;
            for (int layer = firstModelLayer(); layer < firstModelLayer() + static_cast<int>(layerCount()); ++layer)
            {
                const auto &owners = snapshot->placementOwnerMap().owners();
                const auto source = std::find_if(owners.begin(), owners.end(), [&](const auto &owner) {
                    return owner.layer_idx == layer && owner.shard_cohort;
                });
                if (source == owners.end())
                    throw std::invalid_argument("Target expansion has no original cohort source manifest for a layer");
                manifests.push_back(source->shard_cohort->source());
            }
            const MoEOverlayEconomyCalibrationLayerCatalog original_catalog(manifests);
            if (original_catalog.identity() != identity_)
                throw std::invalid_argument("Target expansion catalog differs from original source formats or geometry");
            MoEOverlayEconomyCalibrationPlanner planner({.live_snapshot = snapshot,
                .complete_expert_bytes_per_layer = complete_expert_bytes_per_layer_});
            required = planner.requiredTargetCoordinates();
            for (const auto &declaration : planner.requiredTargetMeasurements(manifests))
                original_geometry.emplace(declaration.coordinate, declaration.geometry);
        }

        std::map<Coordinate, const MoEOverlayTargetLayerMigrationMeasurement *> observed;
        std::set<std::pair<int, int>> directed_pairs;
        for (const auto &row : measurements.rows)
        {
            if (row.measured_coordinate ||
                !observed.emplace(row.coordinate, &row).second)
                throw std::invalid_argument("Calibration expansion rejects projected or duplicate input measurements");
            (void)storageIndexForModelLayer(row.coordinate.layer);
            if (!std::binary_search(representative_layers_.begin(), representative_layers_.end(), row.coordinate.layer))
                throw std::invalid_argument("Calibration expansion requires physically measured catalog representatives");
            if (snapshot)
            {
                const auto expected = original_geometry.find(row.coordinate);
                if (expected == original_geometry.end() || !row.geometry ||
                    !expected->second->sameIdentity(*row.geometry))
                    throw std::invalid_argument("Measured target geometry differs from original members, cuts or source formats");
            }
            else if (!row.coordinate.source_target.isWhole() || !row.coordinate.destination_target.isWhole())
            {
                throw std::invalid_argument("Whole expansion adapter requires unprojected whole representative rows");
            }
            directed_pairs.emplace(row.coordinate.source_target.word(), row.coordinate.destination_target.word());
        }
        if (!snapshot)
            for (const auto &[source, destination] : directed_pairs)
                for (int layer = firstModelLayer(); layer < firstModelLayer() + static_cast<int>(layerCount()); ++layer)
                    required.push_back({MoEOverlayRouteTarget::fromWord(source),
                        MoEOverlayRouteTarget::fromWord(destination), layer});
        std::sort(required.begin(), required.end());

        MoEOverlaySealedMigrationMeasurements expanded;
        expanded.rows.reserve(required.size());
        for (const auto &coordinate : required)
        {
            const auto group = std::find_if(groups_.begin(), groups_.end(), [&](const auto &candidate) {
                return std::binary_search(candidate.member_layers.begin(), candidate.member_layers.end(), coordinate.layer);
            });
            if (group == groups_.end())
                throw std::logic_error("Original catalog lost an owned layer during expansion");
            const auto sample = observed.find({coordinate.source_target,
                coordinate.destination_target, group->representative_layer});
            if (sample == observed.end())
                throw std::invalid_argument("Calibration expansion is missing an original directed-pair/layer-class measurement");
            if (snapshot && !sample->second->geometry->sameIdentity(*original_geometry.at(coordinate)))
                throw std::invalid_argument("Changed original target cuts are not admitted by representative calibration");

            auto copy = *sample->second;
            if (copy.coordinate.layer != coordinate.layer)
            {
                // Preserve the genuine observation owner and its measured layer.
                // Equivalent-layer pricing does not create another physical sample.
                copy.measured_coordinate = copy.coordinate;
                copy.coordinate = coordinate;
            }
            expanded.rows.push_back(std::move(copy));
        }
        std::uint64_t hash = kFNV1a64OffsetBasis;
        hashString(hash, measurements.identity);
        hashString(hash, identity_);
        if (snapshot)
        {
            for (const auto &row : expanded.rows)
            {
                hashUnsigned(hash, static_cast<uint64_t>(row.coordinate.source_target.word()));
                hashUnsigned(hash, static_cast<uint64_t>(row.coordinate.destination_target.word()));
                hashUnsigned(hash, static_cast<uint64_t>(row.coordinate.layer));
                hashUnsigned(hash, static_cast<uint64_t>(row.measured_coordinate
                    ? row.measured_coordinate->layer : row.coordinate.layer));
            }
        }
        expanded.identity = std::string(snapshot ? "expert-overlay-expanded-migration-v2/"
                                                 : "expert-overlay-expanded-migration-v1/") +
            catalogIdentity(hash).substr(std::string("expert-overlay-calibration-layer-catalog-v1/").size());
        if (!expanded.valid())
            throw std::logic_error("Calibration expansion produced an invalid full-layer corpus");
        return expanded;
    }

    namespace
    {
        /** @return Existing typed route alternative after validating the full original target. */
        MoEOverlayRouteTarget calibrationRouteTarget(const MoEExpertOwner &owner)
        {
            const auto target = MoEExpertPlacementTarget::fromPlacement(owner);
            if (!target) throw std::invalid_argument("Calibration source has an incomplete original target");
            return target->isCohort() ? MoEOverlayRouteTarget::cpuCohort(target->tierIndex())
                                      : MoEOverlayRouteTarget::whole(target->participantId(0));
        }
    }

    MoEOverlayEconomyCalibrationPlanner::
        MoEOverlayEconomyCalibrationPlanner(Config config)
        : config_(std::move(config))
    {
        if (!config_.live_snapshot || !config_.live_snapshot->valid())
            throw std::invalid_argument("Calibration requires a valid original live snapshot");
        const auto &snapshot = *config_.live_snapshot;
        const int layers = snapshot.layerCount();
        if (layers <= 0 || config_.complete_expert_bytes_per_layer.size() != size_t(layers) ||
            std::any_of(config_.complete_expert_bytes_per_layer.begin(),
                config_.complete_expert_bytes_per_layer.end(), [](size_t bytes) { return bytes == 0; }))
            throw std::invalid_argument("Calibration requires one positive complete-expert footprint per layer");
        calibration_layers_ = config_.calibration_layers;
        if (calibration_layers_.empty())
            for (int row = 0; row < layers; ++row)
                calibration_layers_.push_back(snapshot.firstModelLayer() + row);
        if (!std::is_sorted(calibration_layers_.begin(), calibration_layers_.end()) ||
            std::adjacent_find(calibration_layers_.begin(), calibration_layers_.end()) != calibration_layers_.end() ||
            std::any_of(calibration_layers_.begin(), calibration_layers_.end(), [&](int layer) {
                return layer < snapshot.firstModelLayer() ||
                       int64_t(layer) >= int64_t(snapshot.firstModelLayer()) + layers; }))
            throw std::invalid_argument("Calibration layers must be unique original model coordinates");
        const auto &map = snapshot.placementOwnerMap();
        for (const int layer : calibration_layers_)
        {
            std::map<int32_t, MoEExpertPlacementTarget> targets;
            std::set<int> covered_participants;
            for (const auto &owner : map.owners())
            {
                if (owner.layer_idx != layer) continue;
                const auto tag = calibrationRouteTarget(owner);
                const auto target = *MoEExpertPlacementTarget::fromPlacement(owner);
                const auto [position, inserted] = targets.emplace(tag.word(), target);
                if (!inserted && !(position->second == target))
                    throw std::invalid_argument("Calibration target tag has conflicting original source geometry");
                for (size_t member = 0; member < target.participantCount(); ++member)
                    covered_participants.insert(target.participantId(member));
            }
            // A zero-quota/empty endpoint cannot be invented as a calibration
            // source. Cohort members are covered by their one complete target.
            for (const auto &participant : map.participants())
                if (!covered_participants.contains(participant.participant_id))
                    throw std::invalid_argument("Calibration requires a resident source at every admitted target/layer");
            if (targets.size() < 2)
                throw std::invalid_argument("Calibration requires at least two complete resident targets");
            for (const auto &[source, source_definition] : targets)
                for (const auto &[destination, destination_definition] : targets)
                {
                    (void)source_definition; (void)destination_definition;
                    if (source != destination)
                        required_target_coordinates_.push_back({MoEOverlayRouteTarget::fromWord(source),
                            MoEOverlayRouteTarget::fromWord(destination), layer});
                }
        }
        std::sort(required_target_coordinates_.begin(), required_target_coordinates_.end());
    }

    std::vector<MoEOverlayRequiredTargetMigrationMeasurement>
    MoEOverlayEconomyCalibrationPlanner::requiredTargetMeasurements(
        std::span<const MoEOverlayLayerWeightManifest> manifests) const
    {
        std::map<int, const MoEOverlayLayerWeightManifest *> source;
        for (const auto &manifest : manifests)
        {
            if (!manifest.valid() || !source.emplace(manifest.layer_idx, &manifest).second)
                throw std::invalid_argument("Calibration declarations require unique complete original source manifests");
        }
        if (source.size() != static_cast<std::size_t>(config_.live_snapshot->layerCount()))
            throw std::invalid_argument("Calibration declarations omitted original source layers");
        for (int layer = config_.live_snapshot->firstModelLayer();
             layer < config_.live_snapshot->firstModelLayer() + config_.live_snapshot->layerCount(); ++layer)
        {
            const auto manifest = source.find(layer);
            if (manifest == source.end())
                throw std::invalid_argument("Calibration declarations contain a foreign or omitted original source layer");
            if (config_.live_snapshot->jointOwnerMap())
            {
                const auto &owners = config_.live_snapshot->placementOwnerMap().owners();
                const auto original = std::find_if(owners.begin(), owners.end(), [layer](const auto &owner) {
                    return owner.layer_idx == layer && owner.shard_cohort;
                });
                if (original == owners.end() || original->shard_cohort->source() != *manifest->second)
                    throw std::invalid_argument("Calibration declaration changed its original cohort source geometry or format");
            }
        }
        std::vector<MoEOverlayRequiredTargetMigrationMeasurement> result;
        result.reserve(required_target_coordinates_.size());
        for (const auto &coordinate : required_target_coordinates_)
        {
            const auto manifest = source.find(coordinate.layer);
            if (manifest == source.end())
                throw std::invalid_argument("Calibration declaration has no original source layer");
            const auto wave = buildPairSwap(coordinate.source_target,
                coordinate.destination_target, coordinate.layer, 1);
            result.push_back({.coordinate = coordinate,
                .geometry = makePhysicalMoEOverlayMeasurementGeometry(wave.migrations.front(),
                    *manifest->second, config_.live_snapshot->expertCount(),
                    config_.live_snapshot->placementOwnerMap())});
        }
        return result;
    }

    std::vector<MoEOverlayMigrationMeasurementCoordinate>
    MoEOverlayEconomyCalibrationPlanner::requiredCoordinates() const
    {
        std::vector<MoEOverlayMigrationMeasurementCoordinate> result;
        result.reserve(required_target_coordinates_.size());
        for (const auto &coordinate : required_target_coordinates_)
        {
            if (!coordinate.source_target.isWhole() || !coordinate.destination_target.isWhole())
                throw std::logic_error("Scalar calibration consumers cannot certify complete CPU cohort targets");
            result.push_back({coordinate.source_target.word(), coordinate.destination_target.word(), coordinate.layer});
        }
        return result;
    }

    const MoEExpertOwner &MoEOverlayEconomyCalibrationPlanner::sourceOwner(
        MoEOverlayRouteTarget target, int layer) const
    {
        if (target.isAbsent()) throw std::invalid_argument("Calibration target is absent");
        for (const auto &owner : config_.live_snapshot->placementOwnerMap().owners())
            if (owner.layer_idx == layer && calibrationRouteTarget(owner).word() == target.word())
                return owner;
        throw std::invalid_argument("Calibration target has no complete resident original source at this layer");
    }

    MoEOverlayTierMigrationDirection
    MoEOverlayEconomyCalibrationPlanner::direction(
        int source_tier,
        int destination_tier) const
    {
        const auto &tiers =
            config_.live_snapshot->placement_plan->routed_tiers;
        if (source_tier < 0 || destination_tier < 0 ||
            static_cast<std::size_t>(source_tier) >= tiers.size() ||
            static_cast<std::size_t>(destination_tier) >= tiers.size())
        {
            throw std::invalid_argument(
                "ExpertOverlay calibration movement references an unknown tier");
        }
        const int source_priority =
            tiers[static_cast<std::size_t>(source_tier)].priority;
        const int destination_priority =
            tiers[static_cast<std::size_t>(destination_tier)].priority;
        if (destination_priority < source_priority)
            return MoEOverlayTierMigrationDirection::Promotion;
        if (destination_priority > source_priority)
            return MoEOverlayTierMigrationDirection::Demotion;
        return MoEOverlayTierMigrationDirection::SamePriority;
    }

    MoEOverlayResidencyTransaction MoEOverlayEconomyCalibrationPlanner::buildPairSwap(
        int source_participant, int destination_participant, int layer, uint64_t sequence) const
    {
        return buildPairSwap(MoEOverlayRouteTarget::whole(source_participant),
                             MoEOverlayRouteTarget::whole(destination_participant), layer, sequence);
    }

    MoEOverlayResidencyTransaction MoEOverlayEconomyCalibrationPlanner::buildPairSwap(
        MoEOverlayRouteTarget source_target, MoEOverlayRouteTarget destination_target,
        int layer, uint64_t sequence) const
    {
        const MoEOverlayMigrationTargetCoordinate coordinate{source_target, destination_target, layer};
        if (!coordinate.valid() || sequence == 0 ||
            !std::binary_search(required_target_coordinates_.begin(), required_target_coordinates_.end(), coordinate))
            throw std::invalid_argument("Calibration requires one declared complete target pair/layer and positive sequence");
        const auto &forward = sourceOwner(source_target, layer);
        const auto &reverse = sourceOwner(destination_target, layer);
        auto forward_destination = reverse;
        forward_destination.expert_id = forward.expert_id;
        auto reverse_destination = forward;
        reverse_destination.expert_id = reverse.expert_id;
        MoEOverlayResidencyTransaction result;
        result.purpose = MoEOverlayResidencyTransactionPurpose::EconomyCalibration;
        result.calibration_sequence = sequence;
        result.expected_epoch = config_.live_snapshot->epoch;
        result.previous = config_.live_snapshot;
        result.candidate = MoEOverlayResidencyAuthority::buildCalibrationShadowSnapshot(config_.live_snapshot);
        const auto bytes = config_.complete_expert_bytes_per_layer.at(size_t(layer - result.previous->firstModelLayer()));
        result.migrations = {
            {.layer_idx=layer, .expert_id=forward.expert_id, .activation_count=0, .estimated_weight_bytes=bytes,
             .direction=direction(forward.tier_idx, reverse.tier_idx), .source=forward, .destination=std::move(forward_destination)},
            {.layer_idx=layer, .expert_id=reverse.expert_id, .activation_count=0, .estimated_weight_bytes=bytes,
             .direction=direction(reverse.tier_idx, forward.tier_idx), .source=reverse, .destination=std::move(reverse_destination)}};
        result.migration_cycles = {{.layer_idx=layer, .migration_indices={0,1}}};
        // Preserve the historical forward/reverse requirement order for whole
        // targets, expanding each actual original destination member in place.
        for (const auto &edge : result.migrations)
        {
            const auto destination = *MoEExpertPlacementTarget::fromPlacement(edge.destination);
            for (size_t member = 0; member < destination.participantCount(); ++member)
                result.shadow_requirements.push_back({.layer_idx=layer, .tier_idx=destination.tierIndex(),
                    .destination_participant=destination.participantId(member), .slot_count=1});
        }
        if (!result.valid()) throw std::logic_error("Calibration producer did not seal a complete original-target shadow wave");
        return result;
    }
} // namespace llaminar2
