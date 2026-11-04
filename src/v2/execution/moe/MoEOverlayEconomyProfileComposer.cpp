/**
 * @file MoEOverlayEconomyProfileComposer.cpp
 * @brief Validates and canonicalizes measured ExpertOverlay economy evidence.
 *
 * The implementation intentionally stays device-free.  Exact CPU/GPU profilers
 * own streams, events, warmup, and sample collection; this component rejects
 * incomplete evidence and converts it into immutable optimizer input.  Keeping
 * that boundary pure makes topology and arithmetic policy testable without
 * occupying an accelerator.
 */

#include "MoEOverlayEconomyProfileComposer.h"

#include "MoEOverlayEconomyCalibrationPlanner.h"
#include "MoEOverlayPreparedTargetService.h"
#include "MoEOverlayCPUServiceMeasurement.h"
#include "MoEExpertShardCohort.h"
#include <map>
#include <set>
#include "../../utils/FNV1a.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace llaminar2
{
    namespace
    {
        /** @brief Multiply size coordinates while rejecting host-size overflow. */
        std::size_t checkedProduct(
            std::size_t lhs,
            std::size_t rhs,
            const char *description)
        {
            if (lhs != 0 &&
                rhs > std::numeric_limits<std::size_t>::max() / lhs)
            {
                throw std::invalid_argument(
                    std::string("ExpertOverlay economy ") + description +
                    " overflows host size");
            }
            return lhs * rhs;
        }

        /** @brief Add one evidence total while rejecting integer overflow. */
        void checkedAdd(
            uint64_t &target,
            uint64_t value,
            const char *description)
        {
            if (value > std::numeric_limits<uint64_t>::max() - target)
            {
                throw std::invalid_argument(
                    std::string("ExpertOverlay economy ") + description +
                    " overflows uint64_t");
            }
            target += value;
        }

        /**
         * @brief Add one integer using a platform-independent little-endian form.
         *
         * Profile identities are compared across MPI ranks and therefore must
         * not depend on native struct padding or host endianness.
         */
        void hashUnsigned(uint64_t &hash, uint64_t value) noexcept
        {
            std::array<unsigned char, sizeof(value)> bytes{};
            for (std::size_t index = 0; index < bytes.size(); ++index)
            {
                bytes[index] = static_cast<unsigned char>(value & 0xffu);
                value >>= 8u;
            }
            hash = fnv1a64(bytes.data(), bytes.size(), hash);
        }

        /** @brief Add a length-delimited string to a stable profile identity. */
        void hashString(uint64_t &hash, const std::string &value) noexcept
        {
            hashUnsigned(hash, static_cast<uint64_t>(value.size()));
            hash = fnv1a64(value.data(), value.size(), hash);
        }

        /** @brief Render the lightweight diagnostic digest without SHA overhead. */
        std::string profileIdentity(const char *kind, uint64_t hash)
        {
            std::ostringstream output;
            output << "expert-overlay-" << kind << "-v1/"
                   << std::hex << std::setfill('0') << std::setw(16) << hash;
            return output.str();
        }

        /**
         * @brief Ask the canonical placement planner to validate service rows.
         *
         * A zero-demand generation preserves incumbent placement while still
         * exercising the planner's totality, positivity, and integer-priority
         * checks.  This avoids maintaining a second interpretation here.
         */
        void validateServiceProfileWithPlanner(
            const MoERoutedExpertPlacementPlan &plan,
            const MoERoutedExpertModelMetadata &metadata,
            const MoERoutedTierServiceProfile &profile)
        {
            DecodeExpertHistogramWindow window;
            window.generation = 0;
            window.num_layers = metadata.num_layers;
            window.num_experts = metadata.num_experts;
            window.first_model_layer = metadata.first_model_layer;
            const std::size_t entries = checkedProduct(
                static_cast<std::size_t>(metadata.num_layers),
                static_cast<std::size_t>(metadata.num_experts),
                "service validation geometry");
            window.expert_counts.assign(entries, 0);
            window.source_expert_counts.assign(
                checkedProduct(
                    entries,
                    kExpertHistogramProductionSourceCount,
                    "phase service validation geometry"),
                0);

            MoERoutedExpertPlacementPlannerOptions options;
            options.decode_histogram_window = &window;
            options.phase_service_profile = &profile;
            options.rebalancer.enabled = true;
            options.rebalancer.previous_placements = plan.placements;
            auto maintenance_plan = plan;
            maintenance_plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            (void)MoERoutedExpertPlacementPlanner::plan(
                maintenance_plan,
                metadata,
                options);
        }
    } // namespace

    bool MoEOverlayCertifiedEconomyProfiles::valid() const noexcept
    {
        return service != nullptr && migration != nullptr &&
               !service->identity.empty() &&
               service->production_topology.valid() &&
               !migration->identity.empty() &&
               policy.valid();
    }

    std::vector<MoEOverlayParticipantLayerServiceMeasurement>
    MoEOverlayEconomyProfileComposer::normalizeServiceTotals(
        std::vector<MoEOverlayParticipantLayerServiceTotals> totals,
        const ExpertHistogramProductionTopology &production_topology)
    {
        if (totals.empty() || !production_topology.valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay service normalization requires endpoint totals and a valid production topology");
        }
        std::sort(
            totals.begin(),
            totals.end(),
            [](const auto &left, const auto &right)
            {
                if (left.participant_id != right.participant_id)
                    return left.participant_id < right.participant_id;
                return left.layer < right.layer;
            });

        std::vector<MoEOverlayParticipantLayerServiceMeasurement> result;
        result.reserve(totals.size());
        int previous_participant = -1;
        int previous_layer = -1;
        for (const auto &raw : totals)
        {
            if (!raw.valid() || raw.layer < production_topology.firstModelLayer() ||
                raw.layer >= production_topology.endModelLayer() ||
                (raw.participant_id == previous_participant &&
                 raw.layer == previous_layer))
            {
                throw std::invalid_argument(
                    "ExpertOverlay service totals are invalid, overflowed, or duplicated");
            }

            MoEOverlayParticipantLayerServiceMeasurement normalized;
            normalized.participant_id = raw.participant_id;
            normalized.layer = raw.layer;
            for (std::size_t phase = 0;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                const uint64_t duration = raw.total_nanoseconds[phase];
                const uint64_t activations = raw.activation_count[phase];
                const uint64_t samples = raw.sample_count[phase];
                if (!production_topology.reachable(raw.layer, phase))
                {
                    if (duration != 0 || activations != 0 || samples != 0)
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay service totals sampled a runtime-disabled phase");
                    }
                    continue;
                }
                if (!production_topology.requiresServiceEvidence(
                        raw.layer, phase))
                {
                    /*
                     * A retained catch-up/tail graph may contribute a real
                     * observation without becoming recurring movement
                     * economics. Keep the graph evidence legal, but publish a
                     * zero profile column so it cannot steer placement.
                     */
                    continue;
                }
                if (duration == 0 || activations == 0 || samples == 0)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay service totals omitted a production phase");
                }
                /* ceil(duration / activations), written to avoid addition overflow. */
                normalized.nanoseconds_per_activation[phase] =
                    duration / activations +
                    (duration % activations == 0 ? 0u : 1u);
                normalized.sample_count[phase] = samples;
            }
            result.push_back(normalized);
            previous_participant = raw.participant_id;
            previous_layer = raw.layer;
        }
        return result;
    }

    std::vector<MoEOverlayParticipantLayerServiceMeasurement>
    MoEOverlayEconomyProfileComposer::normalizeEquivalentServiceTotals(
        std::vector<MoEOverlayParticipantLayerServiceTotals> totals,
        const ExpertHistogramProductionTopology &production_topology,
        const MoEOverlayEconomyCalibrationLayerCatalog &catalog)
    {
        const std::size_t layer_count =
            catalog.completeExpertBytesPerLayer().size();
        if (totals.empty() || layer_count == 0 ||
            totals.size() % layer_count != 0 ||
            !production_topology.valid() ||
            production_topology.layerCount() != layer_count ||
            production_topology.firstModelLayer() != catalog.firstModelLayer())
        {
            throw std::invalid_argument(
                "ExpertOverlay equivalent service normalization requires complete endpoint/layer geometry and a matching production topology");
        }

        std::sort(
            totals.begin(),
            totals.end(),
            [](const auto &left, const auto &right)
            {
                if (left.participant_id != right.participant_id)
                    return left.participant_id < right.participant_id;
                return left.layer < right.layer;
            });

        /*
         * Validate the manifest partition independently of its constructor so
         * this arithmetic remains total if a future catalog implementation
         * changes representation. Every model layer must contribute exactly
         * once to precisely one estimator.
         */
        std::vector<bool> covered(layer_count, false);
        for (const auto &group : catalog.groups())
        {
            if (!group.valid())
            {
                throw std::invalid_argument(
                    "ExpertOverlay equivalent service normalization received an invalid layer class");
            }
            for (const int layer : group.member_layers)
            {
                if (layer < catalog.firstModelLayer() ||
                    layer >= production_topology.endModelLayer() ||
                    covered[static_cast<std::size_t>(layer - catalog.firstModelLayer())])
                {
                    throw std::invalid_argument(
                        "ExpertOverlay service layer classes overlap or exceed model geometry");
                }
                covered[static_cast<std::size_t>(layer - catalog.firstModelLayer())] = true;
            }
        }
        if (std::any_of(
                covered.begin(), covered.end(), [](bool value) { return !value; }))
        {
            throw std::invalid_argument(
                "ExpertOverlay service layer classes omit model layers");
        }

        std::vector<MoEOverlayParticipantLayerServiceTotals> pooled;
        pooled.reserve(totals.size());
        const std::size_t participant_count = totals.size() / layer_count;
        int previous_participant = -1;
        for (std::size_t participant = 0;
             participant < participant_count;
             ++participant)
        {
            const std::size_t base = participant * layer_count;
            const int participant_id = totals[base].participant_id;
            if (participant_id < 0 || participant_id <= previous_participant)
            {
                throw std::invalid_argument(
                    "ExpertOverlay service totals have duplicate or unordered participant blocks");
            }
            for (std::size_t layer = 0; layer < layer_count; ++layer)
            {
                const auto &raw = totals[base + layer];
                if (!raw.valid() || raw.participant_id != participant_id ||
                    raw.layer != catalog.firstModelLayer() + static_cast<int>(layer))
                {
                    throw std::invalid_argument(
                        "ExpertOverlay equivalent service totals omit or duplicate a participant/layer coordinate");
                }
                for (std::size_t phase = 0;
                     phase < kExpertHistogramProductionSourceCount;
                     ++phase)
                {
                    const bool has_evidence =
                        raw.total_nanoseconds[phase] != 0 &&
                        raw.activation_count[phase] != 0 &&
                        raw.sample_count[phase] != 0;
                    if (!production_topology.reachable(
                            raw.layer, phase) &&
                        has_evidence)
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay equivalent service totals sampled an unreachable layer phase");
                    }
                }
            }

            for (const auto &group : catalog.groups())
            {
                MoEOverlayParticipantLayerServiceTotals aggregate;
                aggregate.participant_id = participant_id;
                for (const int layer : group.member_layers)
                {
                    const auto &raw =
                        totals[base + static_cast<std::size_t>(layer - catalog.firstModelLayer())];
                    for (std::size_t phase = 0;
                         phase < kExpertHistogramProductionSourceCount;
                         ++phase)
                    {
                        if (!production_topology.requiresServiceEvidence(
                                layer, phase))
                            continue;
                        checkedAdd(
                            aggregate.total_nanoseconds[phase],
                            raw.total_nanoseconds[phase],
                            "equivalent service duration");
                        checkedAdd(
                            aggregate.activation_count[phase],
                            raw.activation_count[phase],
                            "equivalent service activation count");
                        checkedAdd(
                            aggregate.sample_count[phase],
                            raw.sample_count[phase],
                            "equivalent service sample count");
                    }
                }

                /*
                 * Sparse production routing need not visit every equivalent
                 * layer on every participant. The exact manifest class is the
                 * estimator boundary: at least one real observation per live
                 * phase must exist somewhere in the class, and no synthetic
                 * duration is substituted for a missing class.
                 */
                for (std::size_t phase = 0;
                     phase < kExpertHistogramProductionSourceCount;
                     ++phase)
                {
                    const bool class_phase_requires_evidence = std::any_of(
                        group.member_layers.begin(),
                        group.member_layers.end(),
                        [&production_topology, phase](int layer)
                        {
                            return production_topology
                                .requiresServiceEvidence(layer, phase);
                        });
                    const bool class_has_evidence =
                        aggregate.total_nanoseconds[phase] != 0 &&
                        aggregate.activation_count[phase] != 0 &&
                        aggregate.sample_count[phase] != 0;
                    if (class_phase_requires_evidence != class_has_evidence)
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay equivalent service class lacks measured production-phase evidence");
                    }
                }

                /*
                 * Expansion is deliberate: placement remains layer-indexed,
                 * but every member consumes the one statistically supported
                 * cost for its identical prepared projection contract.
                 */
                for (const int layer : group.member_layers)
                {
                    auto member = aggregate;
                    member.layer = layer;
                    for (std::size_t phase = 0;
                         phase < kExpertHistogramProductionSourceCount;
                         ++phase)
                    {
                        if (production_topology.requiresServiceEvidence(
                                layer, phase))
                            continue;
                        member.total_nanoseconds[phase] = 0u;
                        member.activation_count[phase] = 0u;
                        member.sample_count[phase] = 0u;
                    }
                    pooled.push_back(std::move(member));
                }
            }
            previous_participant = participant_id;
        }
        return normalizeServiceTotals(
            std::move(pooled), production_topology);
    }

    std::vector<MoEOverlayParticipantLayerMigrationCost>
    MoEOverlayEconomyProfileComposer::normalizeMigrationMeasurements(
        std::vector<MoEOverlayParticipantLayerMigrationMeasurement>
            measurements)
    {
        if (measurements.empty())
        {
            throw std::invalid_argument(
                "ExpertOverlay migration normalization requires measured directed edges");
        }
        std::sort(
            measurements.begin(),
            measurements.end(),
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

        std::vector<MoEOverlayParticipantLayerMigrationCost> result;
        result.reserve(measurements.size());
        std::array<int, 3> previous{-1, -1, -1};
        bool has_previous = false;
        for (const auto &measurement : measurements)
        {
            const std::array<int, 3> coordinate{
                measurement.source_participant,
                measurement.destination_participant,
                measurement.layer,
            };
            if (measurement.source_participant < 0 ||
                measurement.destination_participant < 0 ||
                measurement.source_participant ==
                    measurement.destination_participant ||
                measurement.layer < 0 ||
                (has_previous && coordinate == previous) ||
                measurement.wave_wall_nanoseconds == 0 ||
                measurement.wave_sample_count < kMinimumMigrationSamples ||
                measurement.inference_interference_nanoseconds != 0 ||
                measurement.interference_sample_count != 0)
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration measurement has an invalid coordinate, duplicate row, zero wave time, runtime-interference charge, or insufficient samples");
            }

            std::uint64_t critical_path =
                measurement.wave_wall_nanoseconds;
            for (const auto &projection : measurement.projections)
            {
                if (!projection.valid() ||
                    (projection.device_nanoseconds == 0 &&
                     projection.host_nanoseconds == 0 &&
                     projection.transport_nanoseconds == 0))
                {
                    throw std::invalid_argument(
                        "ExpertOverlay migration measurement omitted exact projection timing evidence");
                }
                critical_path = std::max(
                    critical_path,
                    projection.wall_nanoseconds);
            }

            result.push_back({
                .source_participant = measurement.source_participant,
                .destination_participant =
                    measurement.destination_participant,
                .layer = measurement.layer,
                .transfer_and_repack_ns = critical_path,
                .inference_interference_ns =
                    measurement.inference_interference_nanoseconds,
            });
            previous = coordinate;
            has_previous = true;
        }
        return result;
    }

    bool MoEOverlayQuotedPieceMigrationMeasurement::valid() const noexcept
    {
        return quoted_payload_bytes != 0 && measurement.valid() &&
            quoted_payload_bytes == measurement.bytes &&
            (measurement.device_nanoseconds || measurement.host_nanoseconds || measurement.transport_nanoseconds);
    }

    bool MoEOverlayTargetLayerMigrationMeasurement::geometryMatches(
        const MoEOverlayMigrationTargetCoordinate &coordinate,
        const MoEOverlayMigrationMeasurementGeometry &geometry) noexcept
    {
        const auto matches=[](MoEOverlayRouteTarget tag,const MoEOverlayPhysicalProjectionTarget &target) {
            if(tag.isWhole())return target.members.size()==1 && target.members.front().participant_id==tag.word();
            // Canonical CPU shard cohorts contain 2..8 original participants;
            // TP1 is a whole target, never an alias for a one-member cohort.
            if(!tag.isCPUCohort() || target.members.size()<2 || target.members.size()>8 ||
                std::int64_t(target.tier_idx)!=-std::int64_t(tag.word())-2)return false;
            return std::all_of(target.members.begin(),target.members.end(),[](const auto &member){return member.device.is_cpu();});
        };
        return coordinate.valid() && matches(coordinate.source_target,geometry.targets().first) &&
            matches(coordinate.destination_target,geometry.targets().second);
    }

    bool MoEOverlayTargetLayerMigrationMeasurement::valid() const noexcept
    {
        if(!coordinate.valid() || !wave_wall_nanoseconds ||
            wave_sample_count<MoEOverlayEconomyProfileComposer::kMinimumMigrationSamples ||
            inference_interference_nanoseconds || interference_sample_count || pieces.empty())return false;
        if(geometry) {
            if(!geometryMatches(coordinate,*geometry) || pieces.size()!=geometry->pieces().size())return false;
        } else if(!coordinate.source_target.isWhole() || !coordinate.destination_target.isWhole() || pieces.size()!=3)return false;
        return std::all_of(pieces.begin(),pieces.end(),[](const auto &piece){return piece.valid();});
    }

    MoEOverlayParticipantLayerMigrationMeasurement MoEOverlayTargetLayerMigrationMeasurement::whole() const
    {
        if(!valid() || !coordinate.source_target.isWhole() || !coordinate.destination_target.isWhole() || pieces.size()!=3)
            throw std::invalid_argument("Complete-target migration evidence cannot enter a whole-only consumer");
        MoEOverlayParticipantLayerMigrationMeasurement result{
            .source_participant=coordinate.source_target.word(),.destination_participant=coordinate.destination_target.word(),
            .layer=coordinate.layer,.wave_wall_nanoseconds=wave_wall_nanoseconds,.wave_sample_count=wave_sample_count,
            .inference_interference_nanoseconds=inference_interference_nanoseconds,.interference_sample_count=interference_sample_count};
        for(std::size_t role=0;role<3;++role)result.projections[role]=pieces[role].measurement;
        return result;
    }

    MoEOverlayTargetLayerMigrationMeasurement MoEOverlayTargetLayerMigrationMeasurement::fromWhole(
        const MoEOverlayParticipantLayerMigrationMeasurement &row)
    {
        MoEOverlayTargetLayerMigrationMeasurement result{
            .coordinate={MoEOverlayRouteTarget::whole(row.source_participant),MoEOverlayRouteTarget::whole(row.destination_participant),row.layer},
            .wave_wall_nanoseconds=row.wave_wall_nanoseconds,.wave_sample_count=row.wave_sample_count,
            .inference_interference_nanoseconds=row.inference_interference_nanoseconds,.interference_sample_count=row.interference_sample_count};
        result.pieces.reserve(3);
        for(const auto &sample:row.projections)result.pieces.push_back({sample.bytes,sample});
        if(!result.valid())throw std::invalid_argument("Invalid whole-only robust migration evidence");
        return result;
    }

    std::vector<MoEOverlayTargetLayerMigrationCost>
    MoEOverlayEconomyProfileComposer::normalizeMigrationMeasurements(
        std::vector<MoEOverlayTargetLayerMigrationMeasurement> measurements)
    {
        if(measurements.empty())throw std::invalid_argument("Target migration normalization requires complete measured coordinates");
        std::sort(measurements.begin(),measurements.end(),[](const auto &a,const auto &b){return a.coordinate<b.coordinate;});
        std::vector<MoEOverlayTargetLayerMigrationCost> result;result.reserve(measurements.size());
        for(std::size_t i=0;i<measurements.size();++i) {
            const auto &row=measurements[i];
            if(!row.valid() || (i && measurements[i-1].coordinate==row.coordinate))
                throw std::invalid_argument("Target migration evidence has incomplete, repeated or foreign original geometry");
            auto critical_path=row.wave_wall_nanoseconds;
            // Preserve max-of-medians: concurrent member/role costs are never added.
            for(const auto &piece:row.pieces)critical_path=std::max(critical_path,piece.measurement.wall_nanoseconds);
            result.push_back({row.coordinate,critical_path,0});
        }
        return result;
    }

    std::vector<MoEOverlayParticipantLayerMigrationCost>
    MoEOverlayEconomyProfileComposer::checkedWholeMigrationCosts(std::vector<MoEOverlayTargetLayerMigrationCost> costs)
    {
        std::sort(costs.begin(),costs.end(),[](const auto &a,const auto &b){return a.coordinate<b.coordinate;});
        std::vector<MoEOverlayParticipantLayerMigrationCost> result;result.reserve(costs.size());
        for(std::size_t i=0;i<costs.size();++i) {
            const auto &row=costs[i];
            if(!row.coordinate.valid() || !row.coordinate.source_target.isWhole() || !row.coordinate.destination_target.isWhole() ||
                !row.transfer_and_repack_ns || row.inference_interference_ns || (i && costs[i-1].coordinate==row.coordinate))
                throw std::invalid_argument("Cohort or invalid migration prices cannot enter a whole-only objective");
            result.push_back({row.coordinate.source_target.word(),row.coordinate.destination_target.word(),row.coordinate.layer,
                row.transfer_and_repack_ns,row.inference_interference_ns});
        }
        return result;
    }

    MoEOverlayCertifiedEconomyProfiles
    MoEOverlayEconomyProfileComposer::compose(
        const MoERoutedExpertPlacementPlan &plan,
        const MoERoutedExpertModelMetadata &metadata,
        const MoEExpertOwnerMap &owner_map,
        MoEOverlayEconomyMeasurements measurements,
        MoEOverlayMigrationEconomyPolicy policy)
    {
        if (!plan.usesExpertOverlayAuthority())
        {
            throw std::invalid_argument(
                "ExpertOverlay economy profiles require an enabled overlay plan");
        }
        if (metadata.num_layers <= 0 || metadata.num_experts <= 0 ||
            metadata.first_model_layer < 0 ||
            metadata.num_layers > std::numeric_limits<int>::max() - metadata.first_model_layer ||
            plan.first_model_layer != metadata.first_model_layer ||
            plan.routed_tiers.empty() || owner_map.participants().empty())
        {
            throw std::invalid_argument(
                "ExpertOverlay economy profiles require positive model, tier, and participant geometry");
        }
        owner_map.requireLayerGeometry(
            metadata.num_layers, metadata.num_experts, metadata.first_model_layer);
        if (!policy.valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay economy profile received an invalid migration policy");
        }
        if (measurements.service_measurement_identity.empty() ||
            measurements.migration_measurement_identity.empty() ||
            !measurements.production_topology.valid() ||
            measurements.production_topology.firstModelLayer() != metadata.first_model_layer ||
            measurements.production_topology.layerCount() !=
                static_cast<std::size_t>(metadata.num_layers))
        {
            throw std::invalid_argument(
                "ExpertOverlay economy measurements require setup identities and matching production topology");
        }

        const std::size_t participant_count = owner_map.participants().size();
        const std::size_t layer_count =
            static_cast<std::size_t>(metadata.num_layers);
        const std::size_t tier_count = plan.routed_tiers.size();

        /*
         * Participant ids are used as dense protocol coordinates by migration
         * profiles.  Reject a sparse or duplicated id set here rather than
         * allowing different ranks to flatten the same rows differently.
         */
        std::vector<const MoEExpertOwnerParticipant *> participants(
            participant_count,
            nullptr);
        std::vector<std::vector<int>> participants_by_tier(tier_count);
        for (const auto &participant : owner_map.participants())
        {
            if (participant.participant_id < 0 ||
                static_cast<std::size_t>(participant.participant_id) >=
                    participant_count ||
                participants[static_cast<std::size_t>(
                    participant.participant_id)] != nullptr ||
                participant.tier_idx < 0 ||
                static_cast<std::size_t>(participant.tier_idx) >= tier_count)
            {
                throw std::invalid_argument(
                    "ExpertOverlay economy requires contiguous unique participant ids and valid tiers");
            }
            participants[static_cast<std::size_t>(participant.participant_id)] =
                &participant;
            participants_by_tier[static_cast<std::size_t>(
                participant.tier_idx)]
                .push_back(participant.participant_id);
        }
        if (std::any_of(
                participants.begin(),
                participants.end(),
                [](const auto *participant)
                { return participant == nullptr; }) ||
            std::any_of(
                participants_by_tier.begin(),
                participants_by_tier.end(),
                [](const auto &ids)
                { return ids.empty(); }))
        {
            throw std::invalid_argument(
                "ExpertOverlay economy requires every participant id and tier to be represented");
        }

        const std::size_t expected_service_rows = checkedProduct(
            participant_count,
            layer_count,
            "participant service geometry");
        if (measurements.participant_service.size() != expected_service_rows)
        {
            throw std::invalid_argument(
                "ExpertOverlay service measurements must contain every participant/layer row");
        }
        std::sort(
            measurements.participant_service.begin(),
            measurements.participant_service.end(),
            [](const auto &lhs, const auto &rhs)
            {
                if (lhs.participant_id != rhs.participant_id)
                    return lhs.participant_id < rhs.participant_id;
                return lhs.layer < rhs.layer;
            });

        std::vector<const MoEOverlayParticipantLayerServiceMeasurement *>
            service_rows(expected_service_rows, nullptr);
        for (const auto &row : measurements.participant_service)
        {
            if (!metadata.containsModelLayer(row.layer))
                throw std::invalid_argument(
                    "ExpertOverlay service measurement belongs outside the owned layer interval");
            bool invalid_phase_evidence = false;
            for (std::size_t phase = 0;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                const bool has_cost =
                    row.nanoseconds_per_activation[phase] != 0;
                const bool has_samples = row.sample_count[phase] != 0;
                const bool phase_valid =
                    measurements.production_topology.requiresServiceEvidence(
                        row.layer, phase)
                                             ? has_cost && has_samples
                                             : !has_cost && !has_samples;
                invalid_phase_evidence =
                    invalid_phase_evidence || !phase_valid;
            }
            if (row.participant_id < 0 ||
                static_cast<std::size_t>(row.participant_id) >=
                    participant_count ||
                invalid_phase_evidence)
            {
                throw std::invalid_argument(
                    "ExpertOverlay service measurement has an invalid coordinate or evidence inconsistent with layer reachability");
            }
            const std::size_t offset =
                static_cast<std::size_t>(row.participant_id) * layer_count +
                metadata.storageIndexForModelLayer(row.layer);
            if (service_rows[offset] != nullptr)
            {
                throw std::invalid_argument(
                    "ExpertOverlay service measurements repeat a participant/layer row");
            }
            service_rows[offset] = &row;
        }
        if (std::any_of(
                service_rows.begin(),
                service_rows.end(),
                [](const auto *row)
                { return row == nullptr; }))
        {
            throw std::invalid_argument(
                "ExpertOverlay service measurements omitted a participant/layer row");
        }

        auto service_profile =
            std::make_shared<MoERoutedTierServiceProfile>();
        service_profile->production_topology =
            measurements.production_topology;
        service_profile->target_costs.reserve(expected_service_rows);
        for (const auto &row : measurements.participant_service)
        {
            service_profile->target_costs.push_back({
                .target = MoEOverlayRouteTarget::whole(row.participant_id),
                .layer = row.layer,
                .nanoseconds_per_activation =
                    row.nanoseconds_per_activation,
            });
        }
        service_profile->costs.reserve(checkedProduct(
            tier_count,
            layer_count,
            "tier service geometry"));
        for (std::size_t tier = 0; tier < tier_count; ++tier)
        {
            for (std::size_t layer = 0; layer < layer_count; ++layer)
            {
                MoERoutedTierLayerPhaseServiceCost aggregate;
                aggregate.tier_index = static_cast<int>(tier);
                aggregate.layer = metadata.first_model_layer + static_cast<int>(layer);
                for (const int participant_id : participants_by_tier[tier])
                {
                    const auto &row = *service_rows[
                        static_cast<std::size_t>(participant_id) * layer_count +
                        layer];
                    for (std::size_t phase = 0;
                         phase < kExpertHistogramProductionSourceCount;
                         ++phase)
                    {
                        aggregate.nanoseconds_per_activation[phase] =
                            std::max(
                                aggregate.nanoseconds_per_activation[phase],
                                row.nanoseconds_per_activation[phase]);
                    }
                }
                service_profile->costs.push_back(aggregate);
            }
        }

        uint64_t service_hash = kFNV1a64OffsetBasis;
        hashString(service_hash, "MoEOverlayTierServiceProfile/v2");
        hashString(
            service_hash,
            measurements.service_measurement_identity);
        hashUnsigned(service_hash, static_cast<uint64_t>(metadata.num_layers));
        hashUnsigned(service_hash, static_cast<uint64_t>(metadata.first_model_layer));
        hashUnsigned(service_hash, static_cast<uint64_t>(metadata.num_experts));
        hashUnsigned(service_hash, static_cast<uint64_t>(tier_count));
        for (int layer = metadata.first_model_layer;
             layer < metadata.first_model_layer + metadata.num_layers; ++layer)
        {
            for (const bool reachable :
                 measurements.production_topology.sources(layer))
            {
                hashUnsigned(service_hash, reachable ? 1u : 0u);
            }
            for (const bool priced :
                 measurements.production_topology.economySources(layer))
            {
                hashUnsigned(service_hash, priced ? 1u : 0u);
            }
        }
        for (std::size_t tier = 0; tier < tier_count; ++tier)
        {
            hashUnsigned(service_hash, tier);
            hashUnsigned(
                service_hash,
                static_cast<uint64_t>(
                    static_cast<int64_t>(plan.routed_tiers[tier].priority)));
        }
        for (const auto &participant : owner_map.participants())
        {
            hashUnsigned(
                service_hash,
                static_cast<uint64_t>(participant.participant_id));
            hashUnsigned(
                service_hash,
                static_cast<uint64_t>(participant.tier_idx));
            hashString(service_hash, participant.address.toString());
            hashUnsigned(
                service_hash,
                static_cast<uint64_t>(participant.world_rank));
            hashUnsigned(
                service_hash,
                participant.world_rank_known ? 1u : 0u);
        }
        for (const auto &row : measurements.participant_service)
        {
            hashUnsigned(
                service_hash,
                static_cast<uint64_t>(row.participant_id));
            hashUnsigned(service_hash, static_cast<uint64_t>(row.layer));
            for (std::size_t phase = 0;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                hashUnsigned(
                    service_hash,
                    row.nanoseconds_per_activation[phase]);
                hashUnsigned(service_hash, row.sample_count[phase]);
            }
        }
        service_profile->identity =
            profileIdentity("service", service_hash);
        validateServiceProfileWithPlanner(plan, metadata, *service_profile);

        const std::size_t directed_pair_count = checkedProduct(
            participant_count,
            participant_count > 0 ? participant_count - 1 : 0,
            "directed participant geometry");
        const std::size_t expected_migration_rows = checkedProduct(
            directed_pair_count,
            layer_count,
            "directed migration geometry");
        if (measurements.directed_migration.size() !=
            expected_migration_rows)
        {
            throw std::invalid_argument(
                "ExpertOverlay migration measurements must contain every directed participant pair/layer row");
        }
        std::sort(
            measurements.directed_migration.begin(),
            measurements.directed_migration.end(),
            [](const auto &lhs, const auto &rhs)
            {
                if (lhs.source_participant != rhs.source_participant)
                    return lhs.source_participant < rhs.source_participant;
                if (lhs.destination_participant !=
                    rhs.destination_participant)
                {
                    return lhs.destination_participant <
                           rhs.destination_participant;
                }
                return lhs.layer < rhs.layer;
            });

        const std::size_t square_participants = checkedProduct(
            participant_count,
            participant_count,
            "migration lookup geometry");
        std::vector<bool> migration_seen(
            checkedProduct(
                square_participants,
                layer_count,
                "migration lookup layers"),
            false);
        for (const auto &row : measurements.directed_migration)
        {
            if (row.source_participant < 0 ||
                static_cast<std::size_t>(row.source_participant) >=
                    participant_count ||
                row.destination_participant < 0 ||
                static_cast<std::size_t>(row.destination_participant) >=
                    participant_count ||
                row.source_participant == row.destination_participant ||
                !metadata.containsModelLayer(row.layer) ||
                row.transfer_and_repack_ns == 0)
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration measurement has an invalid endpoint, layer, or zero transfer cost");
            }
            const std::size_t offset =
                (static_cast<std::size_t>(row.source_participant) *
                     participant_count +
                 static_cast<std::size_t>(row.destination_participant)) *
                    layer_count +
                metadata.storageIndexForModelLayer(row.layer);
            if (migration_seen[offset])
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration measurements repeat a directed endpoint/layer row");
            }
            migration_seen[offset] = true;
        }
        for (std::size_t source = 0; source < participant_count; ++source)
        {
            for (std::size_t destination = 0;
                 destination < participant_count;
                 ++destination)
            {
                if (source == destination)
                    continue;
                for (std::size_t layer = 0; layer < layer_count; ++layer)
                {
                    const std::size_t offset =
                        (source * participant_count + destination) *
                            layer_count +
                        layer;
                    if (!migration_seen[offset])
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay migration measurements omitted a directed endpoint/layer row");
                    }
                }
            }
        }

        uint64_t migration_hash = kFNV1a64OffsetBasis;
        hashString(migration_hash, "MoEOverlayMigrationCostProfile/v1");
        hashString(
            migration_hash,
            measurements.migration_measurement_identity);
        hashUnsigned(
            migration_hash,
            static_cast<uint64_t>(metadata.num_layers));
        hashUnsigned(migration_hash, static_cast<uint64_t>(metadata.first_model_layer));
        hashUnsigned(
            migration_hash,
            static_cast<uint64_t>(participant_count));
        for (const auto &participant : owner_map.participants())
        {
            hashUnsigned(
                migration_hash,
                static_cast<uint64_t>(participant.participant_id));
            hashString(migration_hash, participant.address.toString());
            hashUnsigned(
                migration_hash,
                static_cast<uint64_t>(participant.world_rank));
            hashUnsigned(
                migration_hash,
                participant.world_rank_known ? 1u : 0u);
        }
        for (const auto &row : measurements.directed_migration)
        {
            hashUnsigned(
                migration_hash,
                static_cast<uint64_t>(row.source_participant));
            hashUnsigned(
                migration_hash,
                static_cast<uint64_t>(row.destination_participant));
            hashUnsigned(migration_hash, static_cast<uint64_t>(row.layer));
            hashUnsigned(migration_hash, row.transfer_and_repack_ns);
            hashUnsigned(migration_hash, row.inference_interference_ns);
        }

        auto migration_profile =
            std::make_shared<MoEOverlayMigrationCostProfile>();
        migration_profile->identity =
            profileIdentity("migration", migration_hash);
        migration_profile->setWholeCosts(measurements.directed_migration);

        MoEOverlayCertifiedEconomyProfiles result{
            .service = std::move(service_profile),
            .migration = std::move(migration_profile),
            .policy = policy,
        };
        if (!result.valid())
        {
            throw std::logic_error(
                "ExpertOverlay economy composer produced an invalid immutable profile bundle");
        }
        return result;
    }

    namespace {
        /** @brief Resolve one actual original execution vertex without fabricating a member owner.
         * @param map Original checked placement. @param key Original target/layer.
         * @return Exact cohort, or null for a declared whole target.
         */
        std::shared_ptr<const MoEExpertShardCohort> preparedServiceCohort(
            const MoEExpertOwnerMap& map,MoEOverlayTargetServiceCoordinate key)
        {
            if(!key.target.isCPUCohort())return {};
            const int tier=-int64_t(key.target.word())-2;
            for(const auto& owner:map.owners())
                if(owner.layer_idx==key.layer && owner.tier_idx==tier && owner.shard_cohort)
                    return owner.shard_cohort;
            throw std::invalid_argument("Prepared service target has no original source-bound cohort");
        }
        /** @brief Select the deterministic actual resident on one original target/layer.
         * @param map Original immutable placement, never a synthetic candidate.
         * @param key Exact complete target and source layer.
         * @return Lowest resident global expert ID, or -1 when this layer has no source.
         */
        int preparedServiceResidentExpert(const MoEExpertOwnerMap& map,
            MoEOverlayTargetServiceCoordinate key)
        {
            int selected=-1;
            for(const auto& owner:map.owners())
                if(owner.layer_idx==key.layer &&
                    (key.target.isCPUCohort()?owner.shard_cohort &&
                        owner.tier_idx==-int64_t(key.target.word())-2:
                        !owner.shard_cohort && owner.owner_participant==key.target.word()))
                    if(selected<0 || owner.expert_id<selected)selected=owner.expert_id;
            return selected;
        }
        /** @brief Compare fixed member/cut geometry inside an already exact source-format catalog class.
         * @param a Original first layer cohort. @param b Original second layer cohort.
         * @return Whether one real prepared sample may price both declarations.
         */
        bool samePreparedServiceCuts(const std::shared_ptr<const MoEExpertShardCohort>& a,
            const std::shared_ptr<const MoEExpertShardCohort>& b)
        {
            if(!a || !b)return !a && !b;
            if(!std::equal(a->participants().begin(),a->participants().end(),b->participants().begin(),b->participants().end()))return false;
            for(size_t member=0;member<a->participants().size();++member)
                if(a->ownership(member)!=b->ownership(member))return false;
            return true;
        }
    }

    const ExpertHistogramProductionTopology& MoEOverlayPreparedTargetServicePlan::topology() const noexcept
    {return *topology_;}

    const MoEOverlayPreparedTargetServicePlan::Entry& MoEOverlayPreparedTargetServicePlan::entry(
        MoEOverlayTargetServiceCoordinate coordinate) const
    {
        const auto found=std::lower_bound(entries_.begin(),entries_.end(),coordinate,
            [](const auto& entry,const auto& key){return entry.coordinate<key;});
        if(found==entries_.end() || found->coordinate!=coordinate)
            throw std::invalid_argument("Prepared service coordinate is outside original target declarations");
        return *found;
    }

    std::shared_ptr<const MoEExpertShardCohort> MoEOverlayPreparedTargetServicePlan::cohort(const Entry& value) const
    {return preparedServiceCohort(snapshot_->placementOwnerMap(),entry(value.coordinate).coordinate);}

    std::shared_ptr<const MoEOverlayPreparedTargetServicePlan> MoEOverlayPreparedTargetServicePlan::checked(
        std::shared_ptr<const MoEOverlayResidencySnapshot> snapshot,
        std::shared_ptr<const MoEOverlayEconomyCalibrationLayerCatalog> catalog,
        const ExpertHistogramProductionTopology& topology)
    {
        if(!snapshot || !snapshot->valid() || !catalog || !topology.valid() ||
            snapshot->firstModelLayer()!=catalog->firstModelLayer() ||
            snapshot->layerCount()!=int(catalog->layerCount()) ||
            topology.firstModelLayer()!=catalog->firstModelLayer() || topology.layerCount()!=catalog->layerCount())
            throw std::invalid_argument("Prepared service requires original sealed source/catalog/phase geometry");
        const auto& map=snapshot->placementOwnerMap();
        std::vector<Entry> entries;
        const auto phaseSource=[](size_t phase){
            return phase==0?ExpertHistogramSource::DecodeToken:phase==1?ExpertHistogramSource::PrefillChunk:
                ExpertHistogramSource::GroupedVerifier;
        };
        for(int layer=snapshot->firstModelLayer();layer<snapshot->firstModelLayer()+snapshot->layerCount();++layer) {
            std::map<int,std::pair<int,std::vector<const MoEExpertOwnerParticipant*>>> targets;
            std::set<int> fragment_ids;
            for(const auto& owner:map.owners())if(owner.layer_idx==layer && owner.shard_cohort) {
                const auto target=MoEOverlayRouteTarget::cpuCohort(owner.tier_idx);
                auto& descriptor=targets[target.word()];descriptor.first=owner.tier_idx;
                if(descriptor.second.empty())for(const auto& member:owner.shard_cohort->participants()) {
                    const auto* actual=map.participantForId(member.participant_id);
                    if(!actual || !actual->world_rank_known || actual->world_rank!=member.world_rank ||
                        actual->address!=member.address || !actual->device.is_cpu())
                        throw std::invalid_argument("Prepared service cohort differs from original member inventory");
                    descriptor.second.push_back(actual);fragment_ids.insert(actual->participant_id);
                }
            }
            for(const auto& member:map.participants())if(!fragment_ids.contains(member.participant_id)) {
                if(!member.world_rank_known || member.world_rank<0)
                    throw std::invalid_argument("Prepared service whole target has no original root rank");
                // A declared joint tier cannot be silently projected as whole when
                // its initial source/cuts are absent. No cohort is rebuilt from quota.
                const auto domain=std::find_if(snapshot->placement_plan->domains.begin(),snapshot->placement_plan->domains.end(),
                    [&](const auto& value){return value.name==member.domain_name;});
                if(domain!=snapshot->placement_plan->domains.end() && member.device.is_cpu() &&
                    domain->routed_compute_policy==RoutedExpertComputePolicy::TensorSharded && domain->participants.size()>1)
                    throw std::invalid_argument("Prepared service joint tier lacks original cohort source geometry");
                targets.emplace(MoEOverlayRouteTarget::whole(member.participant_id).word(),
                    std::pair{member.tier_idx,std::vector<const MoEExpertOwnerParticipant*>{&member}});
            }
            if(targets.empty())throw std::invalid_argument("Prepared service has no original execution targets");
            // A completed cohort is one concurrent execution vertex. Existing
            // scheduling has not certified overlapping vertices as independent;
            // reject only this NEW profile, preserving legacy whole behavior.
            for(auto a=targets.begin();a!=targets.end();++a)for(auto b=std::next(a);b!=targets.end();++b)
                for(const auto* left:a->second.second)for(const auto* right:b->second.second) {
                    const auto& x=left->address;const auto& y=right->address;
                    if(x.device_type==y.device_type && x.device_ordinal==y.device_ordinal &&
                        (!x.isCPU() || x.numa_node==y.numa_node) &&
                        (left->world_rank==right->world_rank || x.hostname==y.hostname))
                        throw std::invalid_argument("Prepared target economy does not certify overlapping physical execution targets");
                }
            for(const auto& [word,descriptor]:targets) {
                Entry entry{.coordinate={MoEOverlayRouteTarget::fromWord(word),layer},.tier=descriptor.first,
                    .root_world_rank=descriptor.second.front()->world_rank,
                    .measured_invocations=MoEOverlayCPUServiceMeasurement::kMeasuredSamples};
                entry.sample_layers.fill(-1);
                for(size_t phase=0;phase<kExpertHistogramProductionSourceCount;++phase)
                    if(topology.requiresServiceEvidence(layer,phase)) {
                        // Pool only exact source classes AND original fixed cuts.
                        // Different cuts remain separate sampled declarations. A
                        // class member without an actual target resident cannot
                        // supply a prepared sample; inspect the SAME original
                        // placement before accepting its equivalent source layer.
                        const auto group=std::find_if(catalog->groups().begin(),catalog->groups().end(),
                            [&](const auto& g){return std::binary_search(g.member_layers.begin(),g.member_layers.end(),layer);});
                        if(group==catalog->groups().end())throw std::invalid_argument("Prepared service omitted an original source class");
                        for(int candidate:group->member_layers)if(topology.requiresServiceEvidence(candidate,phase) &&
                            preparedServiceResidentExpert(map,{entry.coordinate.target,candidate})>=0 &&
                            samePreparedServiceCuts(preparedServiceCohort(map,entry.coordinate),
                                preparedServiceCohort(map,{entry.coordinate.target,candidate}))) {
                            entry.sample_layers[phase]=candidate;break;
                        }
                        if(entry.sample_layers[phase]<0)throw std::invalid_argument("Prepared service has no exact phase sample layer");
                        entry.sample_rows[phase]=MoEOverlayCPUServiceMeasurement::rowsForPhase(phaseSource(phase));
                    }
                entries.push_back(entry);
            }
        }
        std::sort(entries.begin(),entries.end(),[](const auto& a,const auto& b){return a.coordinate<b.coordinate;});
        uint64_t hash=kFNV1a64OffsetBasis;hashString(hash,"PreparedTargetInputReadyCanonicalOutput/v1");
        hashString(hash,catalog->identity());
        for(const auto& entry:entries) {
            hashUnsigned(hash,uint64_t(int64_t(entry.coordinate.target.word())));hashUnsigned(hash,entry.coordinate.layer);
            hashUnsigned(hash,entry.tier);hashUnsigned(hash,entry.root_world_rank);
            for(size_t phase=0;phase<kExpertHistogramProductionSourceCount;++phase) {
                hashUnsigned(hash,uint64_t(int64_t(entry.sample_layers[phase])));hashUnsigned(hash,entry.sample_rows[phase]);
            }
            const auto cohort=preparedServiceCohort(map,entry.coordinate);
            if(cohort)for(size_t member=0;member<cohort->participants().size();++member) {
                const auto& p=cohort->participants()[member];hashUnsigned(hash,p.participant_id);hashUnsigned(hash,p.world_rank);
                hashString(hash,p.address.toString());
                for(const auto role:{WeightRole::MoEExpertGate,WeightRole::MoEExpertUp,WeightRole::MoEExpertDown}) {
                    const auto cut=cohort->ownership(member).projection(role);
                    hashUnsigned(hash,cut.first_row);hashUnsigned(hash,cut.rows);hashUnsigned(hash,cut.first_column);hashUnsigned(hash,cut.columns);
                }
            }
        }
        std::ostringstream identity;identity<<"prepared_target_service_"<<std::hex<<hash;
        auto original=std::shared_ptr<MoEOverlayPreparedTargetServicePlan>(new MoEOverlayPreparedTargetServicePlan(
            std::move(snapshot),std::move(catalog),std::make_shared<const ExpertHistogramProductionTopology>(topology),
            std::move(entries),identity.str()));
        original->original_owner_=original;return original;
    }

    int MoEOverlayPreparedTargetServicePlan::sampleExpert(const Entry& value,size_t phase) const
    {
        const auto& declaration=entry(value.coordinate);
        if(phase>=kExpertHistogramProductionSourceCount || declaration.sample_layers[phase]<0)
            throw std::invalid_argument("Prepared service requested an undeclared phase job");
        const int selected=preparedServiceResidentExpert(snapshot_->placementOwnerMap(),
            {declaration.coordinate.target,declaration.sample_layers[phase]});
        if(selected<0)throw std::invalid_argument("Prepared service target class has no actual original resident expert");
        return selected;
    }

    void MoEOverlayPreparedTargetServicePlan::requireLocalSamples(
        std::span<const MoEOverlayPreparedTargetServiceTotals> rows,int world_rank) const
    {
        if(world_rank<0)throw std::invalid_argument("Prepared service observer has no original rank");
        std::set<MoEOverlayTargetServiceCoordinate> seen;
        for(const auto& row:rows) {
            const auto& declaration=entry(row.coordinate);
            if(declaration.root_world_rank!=world_rank || !seen.insert(row.coordinate).second)
                throw std::invalid_argument("Prepared service has duplicate or nonroot target samples");
            bool live=false;
            for(size_t phase=0;phase<kExpertHistogramProductionSourceCount;++phase) {
                const bool required=declaration.sample_layers[phase]==row.coordinate.layer;
                if(required) {
                    const auto activations=declaration.sample_rows[phase]*declaration.measured_invocations;
                    if(!row.total_nanoseconds[phase] || row.activation_count[phase]!=activations ||
                        row.sample_count[phase]!=declaration.measured_invocations ||
                        row.expert_ids[phase]!=sampleExpert(declaration,phase))
                        throw std::invalid_argument("Prepared service root omitted its exact declared phase sample");
                    live=true;
                }else if(row.total_nanoseconds[phase] || row.activation_count[phase] || row.sample_count[phase] || row.expert_ids[phase]!=-1)
                    throw std::invalid_argument("Prepared service sampled an undeclared class/layer/phase");
            }
            if(!live)throw std::invalid_argument("Prepared service payload carries an inactive layer row");
        }
        for(const auto& declaration:entries_)if(declaration.root_world_rank==world_rank &&
            std::find(declaration.sample_layers.begin(),declaration.sample_layers.end(),declaration.coordinate.layer)!=declaration.sample_layers.end() &&
            !seen.contains(declaration.coordinate))
            throw std::invalid_argument("Prepared service observer omitted a required root sample row");
    }

    std::vector<MoERoutedTargetLayerPhaseServiceCost>
    MoEOverlayEconomyProfileComposer::normalizePreparedService(
        const MoEOverlayPreparedTargetServicePlan& plan,
        std::span<const MoEOverlayPreparedTargetServiceTotals> samples)
    {
        std::map<int,std::vector<MoEOverlayPreparedTargetServiceTotals>> by_root;
        for(const auto& entry:plan.entries())by_root.try_emplace(entry.root_world_rank);
        for(const auto& row:samples)by_root.at(plan.entry(row.coordinate).root_world_rank).push_back(row);
        for(const auto& [rank,rows]:by_root)plan.requireLocalSamples(rows,rank);
        std::map<MoEOverlayTargetServiceCoordinate,const MoEOverlayPreparedTargetServiceTotals*> rows;
        for(const auto& row:samples)if(!rows.emplace(row.coordinate,&row).second)
            throw std::invalid_argument("Prepared target service repeats a complete root coordinate");
        std::vector<MoERoutedTargetLayerPhaseServiceCost> result;result.reserve(plan.entries().size());
        for(const auto& entry:plan.entries()) {
            MoERoutedTargetLayerPhaseServiceCost price{entry.coordinate.target,entry.coordinate.layer,{}};
            for(size_t phase=0;phase<kExpertHistogramProductionSourceCount;++phase)if(entry.sample_layers[phase]>=0) {
                const auto* sample=rows.at({entry.coordinate.target,entry.sample_layers[phase]});
                const auto count=sample->activation_count[phase],elapsed=sample->total_nanoseconds[phase];
                price.nanoseconds_per_activation[phase]=elapsed/count+(elapsed%count!=0);
                if(!price.nanoseconds_per_activation[phase])
                    throw std::invalid_argument("Prepared target service has no positive normalized phase rate");
            }
            result.push_back(price);
        }
        return result;
    }

    std::shared_ptr<const MoERoutedTierServiceProfile>
    MoEOverlayEconomyProfileComposer::composePreparedService(
        std::shared_ptr<const MoEOverlayPreparedTargetServicePlan> plan,
        std::span<const MoEOverlayPreparedTargetServiceTotals> samples)
    {
        if(!plan || !plan->authenticates(plan))throw std::invalid_argument("Prepared service composition requires original checked declaration ownership");
        auto result=std::make_shared<MoERoutedTierServiceProfile>();
        result->prepared_plan=plan;result->production_topology=plan->topology();
        result->target_costs=normalizePreparedService(*plan,samples);
        const auto tier_count=plan->snapshot()->placement_plan->routed_tiers.size();
        for(size_t tier=0;tier<tier_count;++tier)
            for(int layer=plan->snapshot()->firstModelLayer();layer<plan->snapshot()->firstModelLayer()+plan->snapshot()->layerCount();++layer) {
                MoERoutedTierLayerPhaseServiceCost aggregate{int(tier),layer,{}};bool found=false;
                for(const auto& price:result->target_costs)if(price.layer==layer &&
                    plan->entry({price.target,layer}).tier==int(tier)) {
                    found=true;
                    for(size_t phase=0;phase<kExpertHistogramProductionSourceCount;++phase)
                        aggregate.nanoseconds_per_activation[phase]=std::max(aggregate.nanoseconds_per_activation[phase],
                            price.nanoseconds_per_activation[phase]);
                }
                if(!found)throw std::invalid_argument("Prepared service omitted an original tier/layer execution target");
                // A cohort contributes exactly ONE already-complete price; max
                // only aggregates alternative whole targets in the same tier.
                result->costs.push_back(aggregate);
            }
        uint64_t hash=kFNV1a64OffsetBasis;hashString(hash,plan->identity());
        for(const auto& row:result->target_costs) {
            hashUnsigned(hash,uint64_t(int64_t(row.target.word())));hashUnsigned(hash,row.layer);
            for(const auto price:row.nanoseconds_per_activation)hashUnsigned(hash,price);
        }
        result->identity=profileIdentity("prepared_target_service",hash);
        return result;
    }

} // namespace llaminar2
