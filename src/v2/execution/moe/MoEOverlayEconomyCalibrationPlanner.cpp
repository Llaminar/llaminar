/**
 * @file MoEOverlayEconomyCalibrationPlanner.cpp
 * @brief Exact closed-cycle construction for non-publishable calibration waves.
 */

#include "MoEOverlayEconomyCalibrationPlanner.h"

#include "MoEOverlayCapacityResolver.h"
#include "../../utils/FNV1a.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <limits>
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
    } // namespace

    bool MoEOverlayEconomyCalibrationLayerGroup::valid() const noexcept
    {
        return representative_layer >= 0 && complete_expert_bytes > 0 &&
               !member_layers.empty() &&
               member_layers.front() == representative_layer &&
               std::is_sorted(member_layers.begin(), member_layers.end()) &&
               std::adjacent_find(
                   member_layers.begin(), member_layers.end()) ==
                   member_layers.end();
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
        complete_expert_bytes_per_layer_.resize(manifest.size());
        for (std::size_t layer = 0; layer < manifest.size(); ++layer)
        {
            if (!manifest[layer].valid() ||
                manifest[layer].layer_idx != static_cast<int>(layer) ||
                footprints[layer].layer_idx != static_cast<int>(layer))
            {
                throw std::invalid_argument(
                    "ExpertOverlay calibration layer manifest must be valid and contiguous from zero");
            }
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
                [&manifest, &layer](const auto &candidate)
                {
                    return manifest[static_cast<std::size_t>(
                                        candidate.representative_layer)]
                               .projections == manifest[layer].projections;
                });
            if (group == groups_.end())
            {
                groups_.push_back({
                    .representative_layer = static_cast<int>(layer),
                    .member_layers = {static_cast<int>(layer)},
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
                group->member_layers.push_back(static_cast<int>(layer));
            }
        }

        representative_layers_.reserve(groups_.size());
        std::uint64_t hash = kFNV1a64OffsetBasis;
        hashUnsigned(hash, static_cast<std::uint64_t>(manifest.size()));
        hashUnsigned(hash, static_cast<std::uint64_t>(groups_.size()));
        for (const auto &group : groups_)
        {
            if (!group.valid())
            {
                throw std::logic_error(
                    "ExpertOverlay calibration constructed an invalid layer equivalence class");
            }
            representative_layers_.push_back(group.representative_layer);
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

            const auto &representative = manifest[static_cast<std::size_t>(
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
        identity_ = catalogIdentity(hash);
    }

    MoEOverlaySealedMigrationMeasurements
    MoEOverlayEconomyCalibrationLayerCatalog::expand(
        const MoEOverlaySealedMigrationMeasurements &
            representative_measurements) const
    {
        if (!representative_measurements.valid() || groups_.empty() ||
            identity_.empty())
        {
            throw std::invalid_argument(
                "ExpertOverlay calibration expansion requires valid representative evidence and catalog");
        }

        using DirectedPair = std::pair<int, int>;
        std::set<DirectedPair> directed_pairs;
        std::set<std::tuple<int, int, int>> observed;
        for (const auto &row : representative_measurements.rows)
        {
            const auto representative = std::lower_bound(
                representative_layers_.begin(),
                representative_layers_.end(),
                row.layer);
            if (representative == representative_layers_.end() ||
                *representative != row.layer ||
                !observed.emplace(
                     row.source_participant,
                     row.destination_participant,
                     row.layer)
                     .second)
            {
                throw std::invalid_argument(
                    "ExpertOverlay calibration expansion received a non-representative or duplicate row");
            }
            directed_pairs.emplace(
                row.source_participant, row.destination_participant);
        }
        const std::size_t expected_representative_rows = checkedProduct(
            directed_pairs.size(), groups_.size(), "representative matrix");
        if (representative_measurements.rows.size() !=
            expected_representative_rows)
        {
            throw std::invalid_argument(
                "ExpertOverlay calibration expansion is missing a directed-pair/layer-class row");
        }
        for (const auto &[source, destination] : directed_pairs)
        {
            for (const int representative : representative_layers_)
            {
                if (!observed.contains(
                        {source, destination, representative}))
                {
                    throw std::invalid_argument(
                        "ExpertOverlay calibration expansion has an incomplete representative matrix");
                }
            }
        }

        MoEOverlaySealedMigrationMeasurements expanded;
        expanded.rows.reserve(checkedProduct(
            directed_pairs.size(),
            complete_expert_bytes_per_layer_.size(),
            "expanded matrix"));
        for (const auto &row : representative_measurements.rows)
        {
            const auto group = std::lower_bound(
                groups_.begin(),
                groups_.end(),
                row.layer,
                [](const auto &candidate, int representative)
                {
                    return candidate.representative_layer < representative;
                });
            if (group == groups_.end() ||
                group->representative_layer != row.layer)
            {
                throw std::logic_error(
                    "ExpertOverlay calibration representative disappeared during expansion");
            }
            for (const int member : group->member_layers)
            {
                auto copy = row;
                copy.layer = member;
                expanded.rows.push_back(std::move(copy));
            }
        }
        std::sort(
            expanded.rows.begin(),
            expanded.rows.end(),
            [](const auto &left, const auto &right)
            {
                return std::tie(
                           left.source_participant,
                           left.destination_participant,
                           left.layer) <
                       std::tie(
                           right.source_participant,
                           right.destination_participant,
                           right.layer);
            });

        std::uint64_t hash = kFNV1a64OffsetBasis;
        hashString(hash, representative_measurements.identity);
        hashString(hash, identity_);
        expanded.identity =
            "expert-overlay-expanded-migration-v1/" +
            catalogIdentity(hash).substr(
                std::string("expert-overlay-calibration-layer-catalog-v1/")
                    .size());
        if (!expanded.valid())
        {
            throw std::logic_error(
                "ExpertOverlay calibration expansion produced an invalid full-layer corpus");
        }
        return expanded;
    }

    MoEOverlayEconomyCalibrationPlanner::
        MoEOverlayEconomyCalibrationPlanner(Config config)
        : config_(std::move(config))
    {
        if (!config_.live_snapshot || !config_.live_snapshot->valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay calibration planner requires a valid live snapshot");
        }
        const int layers =
            config_.live_snapshot->layered_ownership.layerCount();
        if (layers <= 0 ||
            config_.complete_expert_bytes_per_layer.size() !=
                static_cast<std::size_t>(layers) ||
            std::any_of(
                config_.complete_expert_bytes_per_layer.begin(),
                config_.complete_expert_bytes_per_layer.end(),
                [](std::size_t bytes) { return bytes == 0; }))
        {
            throw std::invalid_argument(
                "ExpertOverlay calibration planner requires one positive complete-expert footprint per layer");
        }

        calibration_layers_ = config_.calibration_layers;
        if (calibration_layers_.empty())
        {
            calibration_layers_.resize(static_cast<std::size_t>(layers));
            for (int layer = 0; layer < layers; ++layer)
                calibration_layers_[static_cast<std::size_t>(layer)] = layer;
        }
        if (!std::is_sorted(
                calibration_layers_.begin(), calibration_layers_.end()) ||
            std::adjacent_find(
                calibration_layers_.begin(), calibration_layers_.end()) !=
                calibration_layers_.end() ||
            std::any_of(
                calibration_layers_.begin(),
                calibration_layers_.end(),
                [layers](int layer)
                { return layer < 0 || layer >= layers; }))
        {
            throw std::invalid_argument(
                "ExpertOverlay calibration layers must be sorted, unique, and inside model geometry");
        }

        const auto &participants =
            config_.live_snapshot->owner_map.participants();
        if (participants.size() < 2)
        {
            throw std::invalid_argument(
                "ExpertOverlay migration calibration requires at least two participants");
        }
        for (const auto &source : participants)
        {
            for (const auto &destination : participants)
            {
                if (source.participant_id == destination.participant_id)
                    continue;
                for (const int layer : calibration_layers_)
                {
                    if (config_.live_snapshot->owner_map
                            .expertsForParticipant(
                                layer, source.participant_id)
                            .empty())
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay calibration cannot measure an endpoint/layer with no resident source expert");
                    }
                    required_coordinates_.push_back({
                        .source_participant = source.participant_id,
                        .destination_participant = destination.participant_id,
                        .layer = layer,
                    });
                }
            }
        }
        std::sort(
            required_coordinates_.begin(), required_coordinates_.end());
    }

    MoEExpertOwner MoEOverlayEconomyCalibrationPlanner::destinationOwner(
        const MoEExpertOwner &source,
        int destination_participant) const
    {
        const auto *participant =
            config_.live_snapshot->owner_map.participantForId(
                destination_participant);
        if (!participant)
        {
            throw std::invalid_argument(
                "ExpertOverlay calibration references an unknown destination participant");
        }
        MoEExpertOwner destination = source;
        destination.tier_idx = participant->tier_idx;
        destination.owner_participant = participant->participant_id;
        destination.device = participant->device;
        destination.resident = true;
        destination.tier_name = participant->tier_name;
        destination.domain_name = participant->domain_name;
        destination.domain_participant_index =
            participant->domain_participant_index;
        destination.owner_world_rank = participant->world_rank;
        destination.owner_world_rank_known =
            participant->world_rank_known;
        destination.address = participant->address;
        return destination;
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

    MoEOverlayResidencyTransaction
    MoEOverlayEconomyCalibrationPlanner::buildPairSwap(
        int source_participant,
        int destination_participant,
        int layer,
        std::uint64_t calibration_sequence) const
    {
        if (source_participant == destination_participant ||
            calibration_sequence == 0 || layer < 0 ||
            layer >=
                config_.live_snapshot->layered_ownership.layerCount() ||
            !std::binary_search(
                calibration_layers_.begin(),
                calibration_layers_.end(),
                layer))
        {
            throw std::invalid_argument(
                "ExpertOverlay calibration swap requires distinct endpoints, a valid layer, and positive sequence");
        }
        const auto source_experts =
            config_.live_snapshot->owner_map.expertsForParticipant(
                layer, source_participant);
        const auto destination_experts =
            config_.live_snapshot->owner_map.expertsForParticipant(
                layer, destination_participant);
        if (source_experts.empty() || destination_experts.empty())
        {
            throw std::invalid_argument(
                "ExpertOverlay calibration swap endpoint has no resident source expert");
        }

        const auto *forward_source =
            config_.live_snapshot->owner_map.ownerFor(
                layer, source_experts.front());
        const auto *reverse_source =
            config_.live_snapshot->owner_map.ownerFor(
                layer, destination_experts.front());
        if (!forward_source || !reverse_source)
        {
            throw std::logic_error(
                "ExpertOverlay calibration lost an owner selected from its live map");
        }

        auto candidate =
            std::make_shared<MoEOverlayResidencySnapshot>(
                *config_.live_snapshot);
        candidate->epoch = config_.live_snapshot->epoch + 1;
        const std::size_t bytes =
            config_.complete_expert_bytes_per_layer[
                static_cast<std::size_t>(layer)];

        MoEOverlayResidencyTransaction transaction;
        transaction.purpose =
            MoEOverlayResidencyTransactionPurpose::EconomyCalibration;
        transaction.calibration_sequence = calibration_sequence;
        transaction.expected_epoch = config_.live_snapshot->epoch;
        transaction.previous = config_.live_snapshot;
        transaction.candidate = std::move(candidate);
        transaction.migrations = {
            {
                .layer_idx = layer,
                .expert_id = forward_source->expert_id,
                .activation_count = 0,
                .estimated_weight_bytes = bytes,
                .direction = direction(
                    forward_source->tier_idx,
                    reverse_source->tier_idx),
                .source = *forward_source,
                .destination = destinationOwner(
                    *forward_source, destination_participant),
            },
            {
                .layer_idx = layer,
                .expert_id = reverse_source->expert_id,
                .activation_count = 0,
                .estimated_weight_bytes = bytes,
                .direction = direction(
                    reverse_source->tier_idx,
                    forward_source->tier_idx),
                .source = *reverse_source,
                .destination = destinationOwner(
                    *reverse_source, source_participant),
            },
        };
        transaction.migration_cycles = {{
            .layer_idx = layer,
            .migration_indices = {0, 1},
        }};
        transaction.shadow_requirements = {
            {
                .layer_idx = layer,
                .tier_idx = transaction.migrations[0].destination.tier_idx,
                .destination_participant = destination_participant,
                .slot_count = 1,
            },
            {
                .layer_idx = layer,
                .tier_idx = transaction.migrations[1].destination.tier_idx,
                .destination_participant = source_participant,
                .slot_count = 1,
            },
        };
        if (!transaction.valid())
        {
            throw std::logic_error(
                "ExpertOverlay calibration planner produced an invalid closed swap");
        }
        return transaction;
    }
} // namespace llaminar2
