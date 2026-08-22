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
               !service->identity.empty() && !migration->identity.empty() &&
               policy.valid();
    }

    std::vector<MoEOverlayParticipantLayerServiceMeasurement>
    MoEOverlayEconomyProfileComposer::normalizeServiceTotals(
        std::vector<MoEOverlayParticipantLayerServiceTotals> totals,
        ExpertHistogramProductionSourceMask active_sources)
    {
        if (totals.empty() ||
            !validExpertHistogramProductionSourceMask(active_sources))
        {
            throw std::invalid_argument(
                "ExpertOverlay service normalization requires endpoint totals and valid runtime phases");
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
            if (!raw.valid() ||
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
                if (!active_sources[phase])
                {
                    if (duration != 0 || activations != 0 || samples != 0)
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay service totals sampled a runtime-disabled phase");
                    }
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
        ExpertHistogramProductionSourceMask active_sources,
        const MoEOverlayEconomyCalibrationLayerCatalog &catalog)
    {
        const std::size_t layer_count =
            catalog.completeExpertBytesPerLayer().size();
        if (totals.empty() || layer_count == 0 ||
            totals.size() % layer_count != 0 ||
            !validExpertHistogramProductionSourceMask(active_sources))
        {
            throw std::invalid_argument(
                "ExpertOverlay equivalent service normalization requires complete endpoint/layer geometry and valid runtime phases");
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
                if (layer < 0 ||
                    static_cast<std::size_t>(layer) >= layer_count ||
                    covered[static_cast<std::size_t>(layer)])
                {
                    throw std::invalid_argument(
                        "ExpertOverlay service layer classes overlap or exceed model geometry");
                }
                covered[static_cast<std::size_t>(layer)] = true;
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
                    raw.layer != static_cast<int>(layer))
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
                    if (!active_sources[phase] && has_evidence)
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay equivalent service totals sampled a runtime-disabled phase");
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
                        totals[base + static_cast<std::size_t>(layer)];
                    for (std::size_t phase = 0;
                         phase < kExpertHistogramProductionSourceCount;
                         ++phase)
                    {
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
                    const bool class_has_evidence =
                        aggregate.total_nanoseconds[phase] != 0 &&
                        aggregate.activation_count[phase] != 0 &&
                        aggregate.sample_count[phase] != 0;
                    if (active_sources[phase] != class_has_evidence)
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
                    pooled.push_back(std::move(member));
                }
            }
            previous_participant = participant_id;
        }
        return normalizeServiceTotals(std::move(pooled), active_sources);
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
            plan.routed_tiers.empty() || owner_map.participants().empty())
        {
            throw std::invalid_argument(
                "ExpertOverlay economy profiles require positive model, tier, and participant geometry");
        }
        if (!policy.valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay economy profile received an invalid migration policy");
        }
        if (measurements.service_measurement_identity.empty() ||
            measurements.migration_measurement_identity.empty() ||
            !validExpertHistogramProductionSourceMask(
                measurements.active_sources))
        {
            throw std::invalid_argument(
                "ExpertOverlay economy measurements require setup identities and valid runtime phases");
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
            bool invalid_phase_evidence = false;
            for (std::size_t phase = 0;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                const bool has_cost =
                    row.nanoseconds_per_activation[phase] != 0;
                const bool has_samples = row.sample_count[phase] != 0;
                const bool phase_valid = measurements.active_sources[phase]
                                             ? has_cost && has_samples
                                             : !has_cost && !has_samples;
                invalid_phase_evidence =
                    invalid_phase_evidence || !phase_valid;
            }
            if (row.participant_id < 0 ||
                static_cast<std::size_t>(row.participant_id) >=
                    participant_count ||
                row.layer < 0 || row.layer >= metadata.num_layers ||
                invalid_phase_evidence)
            {
                throw std::invalid_argument(
                    "ExpertOverlay service measurement has an invalid coordinate or evidence inconsistent with the active runtime phases");
            }
            const std::size_t offset =
                static_cast<std::size_t>(row.participant_id) * layer_count +
                static_cast<std::size_t>(row.layer);
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
        service_profile->active_sources = measurements.active_sources;
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
                aggregate.layer = static_cast<int>(layer);
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
        hashUnsigned(service_hash, static_cast<uint64_t>(metadata.num_experts));
        hashUnsigned(service_hash, static_cast<uint64_t>(tier_count));
        for (const bool active : measurements.active_sources)
        {
            hashUnsigned(service_hash, active ? 1u : 0u);
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
                row.layer < 0 || row.layer >= metadata.num_layers ||
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
                static_cast<std::size_t>(row.layer);
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
        migration_profile->costs =
            std::move(measurements.directed_migration);

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

} // namespace llaminar2
