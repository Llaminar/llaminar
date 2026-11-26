/**
 * @file MoEOverlayResidencyAuthority.cpp
 * @brief Residency authority lifecycle, admission, transaction tracking, and histogram window management.
 */
#include "MoEOverlayResidencyAuthorityInternal.h"

namespace llaminar2
{

    MoEOverlayResidencyOwnershipKind MoEOverlayResidencySnapshot::ownershipKind() const noexcept
    {
        return joint_ ? MoEOverlayResidencyOwnershipKind::JointFragments
                      : MoEOverlayResidencyOwnershipKind::WholeExperts;
    }

    std::shared_ptr<const MoEExpertOwnerMap> MoEOverlayResidencySnapshot::jointOwnerMap() const noexcept
    {
        return joint_ ? joint_->live_owners : nullptr;
    }

    std::shared_ptr<const MoEExpertOverlayPreparationPlan>
    MoEOverlayResidencySnapshot::originalProjectionPreparation() const noexcept
    {
        return joint_ ? joint_->preparation : nullptr;
    }

    const MoEExpertOwnerMap &MoEOverlayResidencySnapshot::placementOwnerMap() const
    {
        if (!joint_)
            return owner_map;
        if (!joint_->live_owners)
            throw std::logic_error("Joint snapshot lost its canonical live placement");
        return *joint_->live_owners;
    }

    int MoEOverlayResidencySnapshot::firstModelLayer() const noexcept
    {
        return joint_ ? (joint_->source_plan ? joint_->source_plan->first_model_layer : 0)
                      : layered_ownership.firstModelLayer();
    }

    int MoEOverlayResidencySnapshot::layerCount() const noexcept
    {
        return joint_ ? (joint_->source_plan ? int(joint_->source_plan->placements.size()) : 0)
                      : layered_ownership.layerCount();
    }

    int MoEOverlayResidencySnapshot::expertCount() const noexcept
    {
        return joint_ ? (joint_->source_plan && !joint_->source_plan->placements.empty()
                            ? int(joint_->source_plan->placements.front().routed_expert_tier.size())
                            : 0)
                      : layered_ownership.expertCount();
    }

    const MoEExpertOwnerMap& MoEOverlayResidencySnapshot::originalOwnerMap() const
    {
        if (!joint_)
            return owner_map;
        const auto original = joint_->preparation->jointOwnerMap();
        if (!original)
            throw std::logic_error("Joint snapshot lost its original source map");
        return *original;
    }

    bool MoEOverlayTierMigration::crossesBackend() const noexcept
    {
        // Cohort construction proves a common compute backend for every
        // member. Reading that type does not manufacture a representative
        // physical owner, rank or shard coordinate.
        const auto backend = [](const MoEExpertOwner &target) {
            return target.shard_cohort
                ? target.shard_cohort->participants().front().address.device_type
                : target.device.type;
        };
        return backend(source) != backend(destination);
    }

    bool MoEOverlayTierMigration::crossesWorldRank() const noexcept
    {
        if (!source.shard_cohort && !destination.shard_cohort)
            return source.owner_world_rank_known && destination.owner_world_rank_known &&
                source.owner_world_rank != destination.owner_world_rank;
        if (source.shard_cohort && destination.shard_cohort)
        {
            for (std::size_t from = 0; from < source.shard_cohort->participants().size(); ++from)
                for (std::size_t to = 0; to < destination.shard_cohort->participants().size(); ++to)
                {
                    // All three fragment projections use the SAME F
                    // interval (Gate/Up N and Down K). Empty intersections do
                    // not create remote movement, even if ranks differ.
                    const auto a = source.shard_cohort->ownership(from).projection(WeightRole::MoEExpertUp);
                    const auto b = destination.shard_cohort->ownership(to).projection(WeightRole::MoEExpertUp);
                    if (std::max(a.first_row, b.first_row) <
                            std::min(a.first_row + a.rows, b.first_row + b.rows) &&
                        source.shard_cohort->participants()[from].world_rank !=
                            destination.shard_cohort->participants()[to].world_rank)
                        return true;
                }
            return false;
        }
        const auto &cohort = source.shard_cohort ? source.shard_cohort : destination.shard_cohort;
        const auto &whole = source.shard_cohort ? destination : source;
        return whole.owner_world_rank_known && std::any_of(cohort->participants().begin(),
            cohort->participants().end(), [&](const auto &member) { return member.world_rank != whole.owner_world_rank; });
    }

    bool MoEOverlayResidencySnapshot::valid() const noexcept
    {
        if (!epoch || !placement_plan)
            return false;
        if (joint_)
            return epoch == joint_->epoch && joint_->source_plan && joint_->preparation &&
                   joint_->live_owners && !joint_->live_owners->owners().empty() &&
                   placement_plan.get() == joint_->live_plan.get() &&
                   !placement_plan.owner_before(joint_->live_plan) &&
                   !joint_->live_plan.owner_before(placement_plan) && layered_ownership.empty() &&
                   owner_map.owners().empty() && owner_map.participants().empty();
        return !layered_ownership.empty() &&
               placement_plan->first_model_layer == layered_ownership.firstModelLayer();
    }



    bool MoEOverlayMigrationEconomyPolicy::valid() const noexcept
    {
        return current_window_weight > 0 &&
               static_cast<uint64_t>(historical_window_weight) +
                       static_cast<uint64_t>(current_window_weight) <=
                   std::numeric_limits<uint32_t>::max() &&
               payoff_horizon_tokens > 0;
    }

    bool MoEOverlayMigrationEconomyEvidence::valid(
        bool has_migrations) const noexcept
    {
        if (!enabled)
        {
            return service_profile_identity.empty() &&
                   migration_profile_identity.empty() &&
                   smoothed_through_generation == 0 &&
                   forecast_fingerprint == 0 &&
                   historical_window_weight == 0 &&
                   current_window_weight == 0 &&
                   payoff_horizon_tokens == 0 &&
                   minimum_net_benefit_ns == 0 &&
                   minimum_residency_generations == 0 &&
                   projected_service_gain_ns == 0 &&
                   projected_transfer_and_repack_ns == 0 &&
                   projected_inference_interference_ns == 0 &&
                   projected_net_benefit_ns == 0 &&
                   payoff_rejected_cycles == 0 &&
                   residency_rejected_cycles == 0;
        }

        if (service_profile_identity.empty() ||
            migration_profile_identity.empty() ||
            forecast_fingerprint == 0 ||
            current_window_weight == 0 || payoff_horizon_tokens == 0 ||
            static_cast<uint64_t>(historical_window_weight) +
                    static_cast<uint64_t>(current_window_weight) >
                std::numeric_limits<uint32_t>::max())
        {
            return false;
        }
        if (!has_migrations)
        {
            return projected_service_gain_ns == 0 &&
                   projected_transfer_and_repack_ns == 0 &&
                   projected_inference_interference_ns == 0 &&
                   projected_net_benefit_ns == 0;
        }
        if (projected_transfer_and_repack_ns >
            std::numeric_limits<uint64_t>::max() -
                projected_inference_interference_ns)
        {
            return false;
        }
        const uint64_t measured_cost =
            projected_transfer_and_repack_ns +
            projected_inference_interference_ns;
        return projected_service_gain_ns >= measured_cost &&
               projected_net_benefit_ns ==
                   projected_service_gain_ns - measured_cost &&
               projected_net_benefit_ns > minimum_net_benefit_ns;
    }

    bool MoEOverlayAuthoritativeResidencyEntry::valid() const noexcept
    {
        const bool valid_axis =
            axis == MoEOptimizationMovementAxis::TierResidency ||
            axis == MoEOptimizationMovementAxis::ParticipantPlacement ||
            axis == MoEOptimizationMovementAxis::Combined;
        return candidate_tier_idx >= 0 &&
               !candidate_target.isAbsent() &&
               (!candidate_target.isCohort() ||
                candidate_target.word() == -std::int64_t(candidate_tier_idx) - 2) && valid_axis &&
               (changed ||
                (activation_count == 0 && estimated_weight_bytes == 0));
    }

    bool MoEOverlayAuthoritativeResidencyPlan::valid() const noexcept
    {
        if (expected_epoch == 0 || num_layers <= 0 || num_experts <= 0 ||
            !histogram_window || !histogram_window->valid() ||
            histogram_window->num_layers != num_layers ||
            histogram_window->num_experts != num_experts)
        {
            return false;
        }
        const auto layers = static_cast<size_t>(num_layers);
        const auto experts = static_cast<size_t>(num_experts);
        if (layers > std::numeric_limits<size_t>::max() / experts ||
            entries.size() != layers * experts)
        {
            return false;
        }
        for (size_t index = 0; index < entries.size(); ++index)
        {
            const auto &entry = entries[index];
            if (!entry.valid())
                return false;

            /*
             * The dense entry and histogram are one authenticated semantic
             * value, not merely two independently well-formed payloads. A
             * follower reconstructs the migration economy from this exact
             * evidence, so accepting a different count here would postpone a
             * coordinator defect until after publication.
             */
            if (entry.changed &&
                entry.activation_count !=
                    histogram_window->expert_counts[index])
            {
                return false;
            }
        }
        return true;
    }

    size_t MoEOverlayAuthoritativeResidencyPlan::offset(
        int layer_idx,
        int expert_id) const
    {
        if (!histogram_window || !histogram_window->containsModelLayer(layer_idx) ||
            histogram_window->num_layers != num_layers || expert_id < 0 ||
            expert_id >= num_experts)
        {
            throw std::out_of_range(
                "ExpertOverlay authoritative plan coordinate is out of range");
        }
        return static_cast<size_t>(layer_idx - firstModelLayer()) *
                   static_cast<size_t>(num_experts) +
               static_cast<size_t>(expert_id);
    }



    bool MoEOverlayTierMigrationCycle::valid(
        const std::vector<MoEOverlayTierMigration> &migrations) const noexcept
    {
        if (layer_idx < 0 || migration_indices.empty())
            return false;
        // A repeated index necessarily repeats its logical (layer, expert)
        // key. The existing key set rejects both without clearing a full-wave
        // bitmap for every two-edge cycle in a large candidate population.
        std::set<std::pair<int, int>> logical_edges;
        for (size_t offset = 0; offset < migration_indices.size(); ++offset)
        {
            const size_t index = migration_indices[offset];
            const size_t next_index = migration_indices[(offset + 1) % migration_indices.size()];
            if (index >= migrations.size() || next_index >= migrations.size())
                return false;
            const auto &edge = migrations[index];
            const auto &next = migrations[next_index];
            if (!logical_edges.emplace(edge.layer_idx, edge.expert_id).second)
                return false;
            const auto targets = migrationPlacementTargets(edge);
            const auto next_targets = migrationPlacementTargets(next);
            if (edge.layer_idx != layer_idx || next.layer_idx != layer_idx ||
                !targets || !next_targets || targets->second != next_targets->first)
                return false;
        }
        return true;
    }

    MoEOverlayMigrationCapacityEvidence analyzeMoEOverlayMigrationCapacity(
        std::span<const MoEOverlayTierMigration> migrations) noexcept
    {
        using FlowKey = std::tuple<int, int>;
        std::map<FlowKey, std::int64_t> participant_flow;
        std::map<FlowKey, std::int64_t> tier_flow;
        std::set<FlowKey> logical_edges;
        MoEOverlayMigrationCapacityEvidence evidence;
        evidence.edges_checked = migrations.size();
        for (const auto &migration : migrations)
        {
            const auto targets = migrationPlacementTargets(migration);
            if (!targets || !logical_edges.emplace(migration.layer_idx, migration.expert_id).second)
            {
                ++evidence.malformed_edges;
                continue;
            }
            const auto &[source, destination] = *targets;
            // One logical edge moves every source/destination fragment. Physical
            // slots are per member; tier slots and demand are counted only once.
            for (size_t member = 0; member < source.participantCount(); ++member)
                --participant_flow[{migration.layer_idx, source.participantId(member)}];
            for (size_t member = 0; member < destination.participantCount(); ++member)
                ++participant_flow[{migration.layer_idx, destination.participantId(member)}];
            --tier_flow[{migration.layer_idx, source.tierIndex()}];
            ++tier_flow[{migration.layer_idx, destination.tierIndex()}];
        }
        evidence.participant_coordinates_checked = participant_flow.size();
        evidence.tier_coordinates_checked = tier_flow.size();
        evidence.participant_flow_violations = static_cast<size_t>(std::count_if(
            participant_flow.begin(), participant_flow.end(),
            [](const auto &entry) { return entry.second != 0; }));
        evidence.tier_flow_violations = static_cast<size_t>(std::count_if(
            tier_flow.begin(), tier_flow.end(),
            [](const auto &entry) { return entry.second != 0; }));
        return evidence;
    }

    bool MoEOverlayResidencyTransaction::valid() const noexcept
    {
        if (!previous || !candidate || !previous->valid() ||
            !candidate->valid() || expected_epoch != previous->epoch ||
            expected_epoch == std::numeric_limits<uint64_t>::max() ||
            candidate->epoch != expected_epoch + 1 ||
            previous->ownershipKind() != candidate->ownershipKind() ||
            previous->firstModelLayer() != candidate->firstModelLayer() ||
            previous->layerCount() != candidate->layerCount() ||
            previous->expertCount() != candidate->expertCount())
        {
            return false;
        }
        const bool joint = previous->ownershipKind() ==
                           MoEOverlayResidencyOwnershipKind::JointFragments;
        const auto same_owner = [](const auto &a, const auto &b)
        { return a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a); };
        const auto matches_placement = [&](const MoEExpertOwner &canonical,
                                           const MoEExpertOwner &edge)
        {
            if (joint)
                return canonical == edge && same_owner(canonical.shard_cohort, edge.shard_cohort);
            return canonical.owner_participant == edge.owner_participant &&
                   canonical.tier_idx == edge.tier_idx && canonical.device == edge.device;
        };
        if (joint &&
            (!same_owner(previous->joint_->source_plan, candidate->joint_->source_plan) ||
             !same_owner(previous->joint_->preparation, candidate->joint_->preparation)))
            return false;
        const bool calibration =
            purpose ==
            MoEOverlayResidencyTransactionPurpose::EconomyCalibration;
        const bool restoration =
            purpose ==
            MoEOverlayResidencyTransactionPurpose::PreparedContextRestoration;
        if ((!calibration && !restoration &&
             purpose != MoEOverlayResidencyTransactionPurpose::PlacementChange))
            return false;
        if (calibration)
        {
            if (joint && (migrations.size() != 2 || migration_cycles.size() != 1))
                return false; // This admitted calibration shape is one reciprocal target pair.
            /*
             * A calibration candidate is deliberately the unchanged live
             * topology at the next private epoch. Only its shadow copies are
             * exercised, and the transport makes commit unrepresentable.
             */
            if (calibration_sequence == 0 || histogram_window ||
                histogram_generation != 0 || host_admission ||
                economy.enabled ||
                !same_owner(previous->placement_plan, candidate->placement_plan) ||
                previous->layered_ownership != candidate->layered_ownership ||
                (joint && (!same_owner(previous->joint_->live_plan, candidate->joint_->live_plan) ||
                           !same_owner(previous->joint_->live_owners, candidate->joint_->live_owners))) ||
                (!joint && (previous->owner_map.owners() != candidate->owner_map.owners() ||
                            previous->owner_map.participants() != candidate->owner_map.participants())))
            {
                return false;
            }
        }
        else if (calibration_sequence != 0)
        {
            return false;
        }
        if (restoration &&
            (histogram_window || histogram_generation != 0 ||
             host_admission || economy.enabled))
        {
            return false;
        }
        if (histogram_window &&
            (!histogram_window->valid() ||
             histogram_window->generation != histogram_generation ||
             histogram_window->first_model_layer != previous->firstModelLayer() ||
             histogram_window->num_layers != previous->layerCount() ||
             histogram_window->num_experts != previous->expertCount()))
        {
            return false;
        }

        if (!economy.valid(!migrations.empty()))
            return false;

        if (host_admission &&
            (!host_admission->valid() || !candidate ||
             host_admission->transaction != candidate->epoch ||
             host_admission->candidate_epoch != candidate->epoch ||
             host_admission->admitted_physical_cycles !=
                 migration_cycles.size()))
        {
            return false;
        }

        const auto &previous_map = previous->placementOwnerMap();
        const auto &candidate_map = candidate->placementOwnerMap();
        if (joint && !calibration)
        {
            // Every changed logical placement must have one physical edge.
            // Otherwise an empty or partial wave could publish uncopied rows.
            std::size_t changed = 0;
            for (const auto &destination : candidate_map.owners())
            {
                const auto *source = previous_map.placementFor(destination.layer_idx,
                                                               destination.expert_id);
                if (!source)
                    return false;
                changed += !(*source == destination);
            }
            if (changed != migrations.size())
                return false;
        }
        if (migrations.empty())
        {
            return migration_cycles.empty() && shadow_requirements.empty() &&
                   !host_admission;
        }

        if (!analyzeMoEOverlayMigrationCapacity(migrations)
                 .capacityPreserved())
        {
            return false;
        }

        std::vector<size_t> coverage(migrations.size(), 0);
        for (const auto &cycle : migration_cycles)
        {
            if (!cycle.valid(migrations))
                return false;
            for (const size_t index : cycle.migration_indices)
                ++coverage[index];
        }
        if (std::any_of(
                coverage.begin(),
                coverage.end(),
                [](size_t count)
                { return count != 1; }))
        {
            return false;
        }

        using RequirementKey = std::tuple<int, int, int>;
        std::map<RequirementKey, size_t> expected_requirements;
        for (const auto &migration : migrations)
        {
            if (migration.layer_idx < 0 || migration.expert_id < 0 ||
                migration.source.layer_idx != migration.layer_idx ||
                migration.destination.layer_idx != migration.layer_idx ||
                migration.source.expert_id != migration.expert_id ||
                migration.destination.expert_id != migration.expert_id ||
                migration.source.tier_idx < 0 ||
                migration.destination.tier_idx < 0)
            {
                return false;
            }
            if (calibration && migration.activation_count != 0)
                return false; // Shadow timing never authors logical routing demand.
            if (histogram_window)
            {
                if (!histogram_window->containsModelLayer(migration.layer_idx) ||
                    migration.expert_id >= histogram_window->num_experts)
                {
                    return false;
                }
                const size_t histogram_index =
                    static_cast<size_t>(migration.layer_idx - histogram_window->first_model_layer) *
                        static_cast<size_t>(histogram_window->num_experts) +
                    static_cast<size_t>(migration.expert_id);
                if (migration.activation_count !=
                    histogram_window->expert_counts[histogram_index])
                {
                    return false;
                }
            }
            if (migration.axis ==
                    MoEOptimizationMovementAxis::ParticipantPlacement &&
                migration.crossesTier())
            {
                return false;
            }
            const auto *previous_owner = previous_map.placementFor(
                migration.layer_idx, migration.expert_id);
            if (!previous_owner || !matches_placement(*previous_owner, migration.source))
            {
                return false;
            }
            if (calibration)
            {
                if (joint)
                {
                    // Shadow destinations are complete original targets already
                    // present in this exact source layer. Borrow the canonical
                    // member/cut/control block, changing only the migrant expert
                    // coordinate; calibration never rewrites live ownership.
                    bool authenticated = false;
                    for (const auto &resident : previous_map.owners())
                    {
                        if (resident.layer_idx != migration.layer_idx) continue;
                        const auto canonical = MoEExpertPlacementTarget::fromPlacement(resident);
                        const auto actual = MoEExpertPlacementTarget::fromPlacement(migration.destination);
                        const auto &destination = migration.destination;
                        // Compare the actual target without copying strings or
                        // manufacturing a temporary owner inside noexcept validation.
                        if (canonical && actual && *canonical == *actual &&
                            same_owner(resident.shard_cohort, destination.shard_cohort) &&
                            resident.owner_participant == destination.owner_participant &&
                            resident.device == destination.device && resident.resident == destination.resident &&
                            resident.tier_name == destination.tier_name && resident.domain_name == destination.domain_name &&
                            resident.domain_participant_index == destination.domain_participant_index &&
                            resident.owner_world_rank == destination.owner_world_rank &&
                            resident.owner_world_rank_known == destination.owner_world_rank_known &&
                            resident.address == destination.address)
                        {
                            authenticated = true;
                            break;
                        }
                    }
                    if (!authenticated) return false;
                }
                else
                {
                    const auto *destination_participant = previous_map.participantForId(
                        migration.destination.owner_participant);
                    if (!destination_participant ||
                        destination_participant->tier_idx != migration.destination.tier_idx ||
                        destination_participant->device != migration.destination.device)
                        return false;
                }
            }
            else
            {
                const auto *candidate_owner = candidate_map.placementFor(
                    migration.layer_idx, migration.expert_id);
                if (!candidate_owner || !matches_placement(*candidate_owner, migration.destination) ||
                    (joint && *previous_owner == *candidate_owner))
                {
                    return false;
                }
                if (joint && restoration)
                {
                    const auto *original = previous->originalOwnerMap().placementFor(
                        migration.layer_idx, migration.expert_id);
                    if (!original || !matches_placement(*original, *candidate_owner))
                        return false;
                }
            }
            const auto destination = MoEExpertPlacementTarget::fromPlacement(migration.destination);
            if (!destination)
                return false;
            for (std::size_t member = 0; member < destination->participantCount(); ++member)
                ++expected_requirements[{migration.layer_idx, destination->tierIndex(),
                                         destination->participantId(member)}];
        }

        std::map<RequirementKey, size_t> actual_requirements;
        for (const auto &requirement : shadow_requirements)
        {
            if (requirement.layer_idx < 0 || requirement.tier_idx < 0 ||
                requirement.destination_participant < 0 ||
                requirement.slot_count == 0)
            {
                return false;
            }
            const RequirementKey key{
                requirement.layer_idx,
                requirement.tier_idx,
                requirement.destination_participant,
            };
            if (!actual_requirements.emplace(key, requirement.slot_count).second)
                return false;
        }
        return actual_requirements == expected_requirements;
    }


    std::unique_ptr<MoEOverlayResidencyAuthority::EconomyState>
    MoEOverlayResidencyAuthority::buildEconomyState(
        const Config &config,
        const MoEOverlayResidencySnapshot &initial_snapshot)
    {
        if (config.migration_cost_profile &&
            !config.migration_economy_policy)
        {
            throw std::invalid_argument(
                "ExpertOverlay migration cost evidence requires an explicit economy policy");
        }
        if (config.migration_economy_policy)
        {
            if (config.maintenance_mode != MoERebalanceRuntimeMode::Dynamic)
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration economics require Dynamic maintenance");
            }
            if (!config.migration_economy_policy->valid())
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration economy policy has invalid smoothing or payoff parameters");
            }
            if (!config.phase_service_profile ||
                !config.migration_cost_profile)
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration economics require complete service and movement profiles");
            }
        }

        if (config.phase_service_profile)
        {
            /*
             * Validate the setup profile with the canonical placement planner.
             * A zero-demand frozen generation checks totality, positive values,
             * and priority monotonicity without changing the incumbent plan.
             */
            DecodeExpertHistogramWindow validation_window;
            validation_window.generation = 0;
            validation_window.first_model_layer = config.model_metadata.first_model_layer;
            validation_window.num_layers = config.model_metadata.num_layers;
            validation_window.num_experts = config.model_metadata.num_experts;
            const std::size_t entries = checkedSizeProduct(
                static_cast<std::size_t>(validation_window.num_layers),
                static_cast<std::size_t>(validation_window.num_experts),
                "service-profile validation geometry");
            validation_window.expert_counts.assign(entries, 0);
            validation_window.source_expert_counts.assign(
                checkedSizeProduct(
                    entries,
                    kExpertHistogramProductionSourceCount,
                    "phase service-profile validation geometry"),
                0);
            MoERoutedExpertPlacementPlannerOptions validation_options;
            validation_options.decode_histogram_window = &validation_window;
            validation_options.phase_service_profile =
                config.phase_service_profile.get();
            validation_options.rebalancer.enabled = true;
            validation_options.rebalancer.previous_placements =
                initial_snapshot.placement_plan->placements;
            auto validation_plan = config.initial_plan;
            validation_plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            (void)MoERoutedExpertPlacementPlanner::plan(
                validation_plan,
                config.model_metadata,
                validation_options);
        }

        if (!config.migration_economy_policy)
            return nullptr;

        const auto &service_profile = *config.phase_service_profile;
        const auto &migration_profile = *config.migration_cost_profile;
        if (migration_profile.identity.empty())
        {
            throw std::invalid_argument(
                "ExpertOverlay migration cost profile requires a non-empty setup identity");
        }

        auto state = std::make_unique<EconomyState>();
        state->first_model_layer = config.model_metadata.first_model_layer;
        state->num_layers = config.model_metadata.num_layers;
        state->num_experts = config.model_metadata.num_experts;
        state->tier_count = static_cast<int>(
            initial_snapshot.placement_plan->routed_tiers.size());
        if(service_profile.prepared_plan) {
            if(!service_profile.prepared_plan->authenticates(service_profile.prepared_plan))
                throw std::invalid_argument("Prepared service declaration does not retain its original metadata owner");
            for(const auto& entry:service_profile.prepared_plan->entries())state->execution_targets.push_back(entry.coordinate.target);
        }else for(const auto& participant:initial_snapshot.placementOwnerMap().participants())
            state->execution_targets.push_back(MoEOverlayRouteTarget::whole(participant.participant_id));
        std::sort(state->execution_targets.begin(),state->execution_targets.end(),[](auto a,auto b){return a.word()<b.word();});
        state->execution_targets.erase(std::unique(state->execution_targets.begin(),state->execution_targets.end(),
            [](auto a,auto b){return a.word()==b.word();}),state->execution_targets.end());
        if(state->execution_targets.empty() || state->execution_targets.size()>size_t(INT32_MAX))
            throw std::invalid_argument("Service profile has no bounded complete execution targets");
        state->participant_count=int(state->execution_targets.size());
        if(service_profile.prepared_plan) {
            const auto& original=*service_profile.prepared_plan->snapshot();
            if(original.firstModelLayer()!=initial_snapshot.firstModelLayer() || original.layerCount()!=initial_snapshot.layerCount() ||
                original.expertCount()!=initial_snapshot.expertCount() ||
                original.placementOwnerMap().participants()!=initial_snapshot.placementOwnerMap().participants() ||
                original.placementOwnerMap().owners()!=initial_snapshot.placementOwnerMap().owners())
                throw std::invalid_argument("Prepared service profile belongs to foreign original target/source geometry");
            const auto source=original.originalProjectionPreparation(),expected=initial_snapshot.originalProjectionPreparation();
            if(source.get()!=expected.get() || source.owner_before(expected) || expected.owner_before(source))
                throw std::invalid_argument("Prepared service profile has foreign original preparation ownership");
            if(config.economy_layer_catalog && (config.economy_layer_catalog.get()!=service_profile.prepared_plan->catalog().get() ||
                config.economy_layer_catalog.owner_before(service_profile.prepared_plan->catalog()) ||
                service_profile.prepared_plan->catalog().owner_before(config.economy_layer_catalog)))
                throw std::invalid_argument("Prepared service profile has a foreign source catalog");
        }else for(size_t index=0;index<state->execution_targets.size();++index)
            if(!state->execution_targets[index].isWhole() || state->execution_targets[index].word()!=int(index))
                throw std::invalid_argument("Legacy service evidence cannot certify cohort target prices");

        if (!service_profile.production_topology.valid() ||
            service_profile.production_topology.firstModelLayer() != state->first_model_layer ||
            service_profile.production_topology.layerCount() !=
                static_cast<std::size_t>(state->num_layers))
        {
            throw std::invalid_argument(
                "ExpertOverlay economy service topology disagrees with live residency geometry");
        }
        state->production_topology =
            service_profile.production_topology;
        const std::size_t service_rows = checkedSizeProduct(
            static_cast<std::size_t>(state->tier_count),
            static_cast<std::size_t>(state->num_layers),
            "service-profile tier/layer geometry");
        // Tier prices still certify placement ranking. Only participant prices
        // are retained for service scoring; there is no second live tier scorer.
        std::vector<bool> service_rows_seen(service_rows, false);
        for (const auto &row : service_profile.costs)
        {
            if (row.tier_index < 0 ||
                row.tier_index >= state->tier_count ||
                !config.model_metadata.containsModelLayer(row.layer))
            {
                throw std::invalid_argument(
                    "ExpertOverlay service profile row is outside live residency geometry");
            }
            const std::size_t offset =
                static_cast<std::size_t>(row.tier_index) *
                    static_cast<std::size_t>(state->num_layers) +
                state->storageIndexForModelLayer(row.layer);
            if (service_rows_seen[offset])
            {
                throw std::invalid_argument(
                    "ExpertOverlay service profile repeats a tier/layer row");
            }
            service_rows_seen[offset] = true;
        }
        if (service_rows_seen.size() != service_profile.costs.size() ||
            std::any_of(
                service_rows_seen.begin(),
                service_rows_seen.end(),
                [](bool seen) { return !seen; }))
        {
            throw std::invalid_argument(
                "ExpertOverlay service profile is not total for live residency geometry");
        }

        const std::size_t participant_service_rows = checkedSizeProduct(
            static_cast<std::size_t>(state->participant_count),
            static_cast<std::size_t>(state->num_layers),
            "participant service-profile geometry");
        state->participant_service_cost_rows.assign(
            participant_service_rows, nullptr);
        for (const auto &row : service_profile.target_costs)
        {
            if (row.target.isAbsent() || !config.model_metadata.containsModelLayer(row.layer))
            {
                throw std::invalid_argument(
                    "ExpertOverlay participant service profile row is outside live residency geometry");
            }
            const std::size_t offset =
                state->targetIndex(row.target) * static_cast<std::size_t>(state->num_layers) +
                state->storageIndexForModelLayer(row.layer);
            if (state->participant_service_cost_rows[offset])
            {
                throw std::invalid_argument(
                    "ExpertOverlay participant service profile repeats a participant/layer row");
            }
            for (std::size_t phase = 0;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                const bool has_cost =
                    row.nanoseconds_per_activation[phase] != 0u;
                if (has_cost !=
                    state->production_topology.requiresServiceEvidence(
                        row.layer, phase))
                {
                    throw std::invalid_argument(
                        "ExpertOverlay participant service profile is inconsistent with active runtime phases");
                }
            }
            state->participant_service_cost_rows[offset] = &row;
        }
        if (state->participant_service_cost_rows.size() !=
                service_profile.target_costs.size() ||
            std::any_of(
                state->participant_service_cost_rows.begin(),
                state->participant_service_cost_rows.end(),
                [](const auto *row) { return row == nullptr; }))
        {
            throw std::invalid_argument(
                "ExpertOverlay participant service profile is not total for live residency geometry");
        }

        const std::size_t participants =
            static_cast<std::size_t>(state->participant_count);
        const std::size_t participant_pairs = checkedSizeProduct(
            participants,
            participants,
            "migration-profile participant geometry");
        const std::size_t migration_slots = checkedSizeProduct(
            participant_pairs,
            static_cast<std::size_t>(state->num_layers),
            "migration-profile layer geometry");
        const std::size_t directed_pairs = checkedSizeProduct(
            participants,
            participants > 0 ? participants - 1 : 0,
            "directed migration-profile participant geometry");
        const std::size_t expected_migration_rows = checkedSizeProduct(
            directed_pairs,
            static_cast<std::size_t>(state->num_layers),
            "directed migration-profile layer geometry");
        if (migration_profile.costs.size() != expected_migration_rows)
        {
            throw std::invalid_argument(
                "ExpertOverlay migration profile must contain every directed endpoint/layer pair");
        }
        state->migration_cost_rows.assign(migration_slots, nullptr);
        for (const auto &row : migration_profile.costs)
        {
            if (!row.coordinate.valid() ||
                !config.model_metadata.containsModelLayer(row.coordinate.layer) ||
                row.transfer_and_repack_ns == 0)
            {
                throw std::invalid_argument(
                    "Migration profile has invalid original targets/layer/cost");
            }
            // Interference is a measured nonnegative cost, including positive
            // values. Retain it in the same canonical row so proposal payoff
            // charges both transfer/repack and inference interference.
            const auto source = state->targetIndex(row.coordinate.source_target);
            const auto destination = state->targetIndex(row.coordinate.destination_target);
            const auto offset =
                (source * participants + destination) *
                    static_cast<std::size_t>(state->num_layers) +
                state->storageIndexForModelLayer(row.coordinate.layer);
            if (state->migration_cost_rows[offset])
            {
                throw std::invalid_argument(
                    "Migration profile repeats an original target edge");
            }
            state->migration_cost_rows[offset] = &row;
        }
        for (int source = 0; source < state->participant_count; ++source)
        {
            for (int destination = 0;
                 destination < state->participant_count;
                 ++destination)
            {
                if (source == destination)
                    continue;
                for (int layer = 0; layer < state->num_layers; ++layer)
                {
                    const std::size_t offset =
                        (static_cast<std::size_t>(source) * participants +
                         static_cast<std::size_t>(destination)) *
                            static_cast<std::size_t>(state->num_layers) +
                        static_cast<std::size_t>(layer);
                    if (!state->migration_cost_rows[offset])
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay migration profile omitted a directed endpoint/layer row");
                    }
                }
            }
        }
        state->last_moved_generation.assign(
            checkedSizeProduct(
                static_cast<std::size_t>(state->num_layers),
                static_cast<std::size_t>(state->num_experts),
                "residency-generation geometry"),
            std::numeric_limits<uint64_t>::max());
        return state;
    }

    MoEOverlayResidencyAuthority::TicketLease::TicketLease(
        MoEOverlayResidencyAuthority *authority,
        std::shared_ptr<PublishedEpochState> epoch_state,
        TicketLeasePurpose purpose)
        : authority_(authority),
          epoch_state_(std::move(epoch_state)),
          snapshot_(epoch_state_ ? epoch_state_->snapshot : nullptr),
          purpose_(purpose)
    {
    }

    MoEOverlayResidencyAuthority::TicketLease::~TicketLease()
    {
        release();
    }

    MoEOverlayResidencyAuthority::TicketLease::TicketLease(
        TicketLease &&other) noexcept
        : authority_(std::exchange(other.authority_, nullptr)),
          epoch_state_(std::move(other.epoch_state_)),
          snapshot_(std::move(other.snapshot_)),
          purpose_(other.purpose_)
    {
    }

    MoEOverlayResidencyAuthority::TicketLease &
    MoEOverlayResidencyAuthority::TicketLease::operator=(
        TicketLease &&other) noexcept
    {
        if (this == &other)
            return *this;
        release();
        authority_ = std::exchange(other.authority_, nullptr);
        epoch_state_ = std::move(other.epoch_state_);
        snapshot_ = std::move(other.snapshot_);
        purpose_ = other.purpose_;
        return *this;
    }

    void MoEOverlayResidencyAuthority::TicketLease::release() noexcept
    {
        if (authority_)
            authority_->releaseTicket(epoch_state_, purpose_);
        authority_ = nullptr;
        epoch_state_.reset();
        snapshot_.reset();
    }

    MoEOverlayResidencyAuthority::MoEOverlayResidencyAuthority(Config config)
        : config_(std::move(config)),
          planning_template_(config_.initial_plan),
          initial_histogram_window_tokens_(
              config_.histogram ? config_.histogram->windowSize() : 0)
    {
        if (config_.model_metadata.first_model_layer < 0 ||
            config_.model_metadata.num_layers <= 0 ||
            config_.model_metadata.num_experts <= 0 ||
            config_.model_metadata.num_layers > std::numeric_limits<int>::max() -
                config_.model_metadata.first_model_layer ||
            config_.initial_plan.first_model_layer != config_.model_metadata.first_model_layer)
            throw std::invalid_argument(
                "ExpertOverlay initial plan must match a valid owned model-layer interval");
        if (!config_.initial_plan.usesExpertOverlayAuthority())
        {
            throw std::invalid_argument(
                "MoE overlay residency authority requires an enabled tiered overlay");
        }
        if (jointFragmentPlan(config_.initial_plan))
        {
            if (config_.maintenance_mode == MoERebalanceRuntimeMode::Off &&
                (config_.histogram || config_.participant_rebalance_policy.enabled ||
                 config_.phase_service_profile || config_.migration_cost_profile ||
                 config_.migration_economy_policy || config_.economy_layer_catalog))
                throw std::invalid_argument("Static joint placement cannot own maintenance or economic state");
            if (MoEExpertOwnerMap::usesCPUJointFragments(config_.initial_plan) &&
                config_.maintenance_mode == MoERebalanceRuntimeMode::Dynamic &&
                (config_.initial_plan.authority_execution != MoEOverlayAuthorityExecutionKind::HostResident ||
                 !hasTierResidencyAxis(availableMoEOptimizationMovementAxes(config_.initial_plan)) ||
                 std::none_of(config_.initial_plan.domains.begin(), config_.initial_plan.domains.end(),
                     [](const auto &domain) { return std::any_of(domain.participants.begin(), domain.participants.end(),
                         [](const auto &address) { return address.isGPU(); }); })))
                throw std::invalid_argument("Dynamic CPU cohorts require the declared mixed GPU/CPU host authority and complete original cohort targets");
        }
        if (observesHistogram() && !config_.histogram)
        {
            throw std::invalid_argument(
                "Observe or Dynamic ExpertOverlay maintenance requires a DecodeExpertHistogram");
        }
        if (config_.histogram)
        {
            if (config_.histogram->firstModelLayer() != config_.model_metadata.first_model_layer ||
                config_.histogram->config().num_layers != config_.model_metadata.num_layers ||
                config_.histogram->config().num_experts != config_.model_metadata.num_experts)
                throw std::invalid_argument("ExpertOverlay histogram must match its owned stage geometry");
            const int initial_window = config_.histogram->windowSize();
            if (!std::isfinite(
                    config_.histogram_window_growth_factor) ||
                config_.histogram_window_growth_factor <= 0.0 ||
                (config_.histogram_max_window_tokens != 0u &&
                 config_.histogram_max_window_tokens <
                     static_cast<std::uint64_t>(initial_window)) ||
                config_.histogram_max_window_tokens >
                    static_cast<std::uint64_t>(
                        std::numeric_limits<int>::max()))
            {
                throw std::invalid_argument(
                    "ExpertOverlay histogram growth requires a finite positive factor and a ceiling no smaller than its initial window");
            }
        }
        if (!config_.participant_rebalance_policy.valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay participant rebalance policy has invalid thresholds or paired-entry capacity");
        }
        if (config_.migration_cost_profile &&
            !config_.migration_economy_policy)
        {
            throw std::invalid_argument(
                "ExpertOverlay migration cost evidence requires an explicit economy policy");
        }
        if (config_.migration_economy_policy)
        {
            if (!migrationEnabled())
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration economics require Dynamic maintenance");
            }
            if (!config_.migration_economy_policy->valid())
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration economy policy has invalid smoothing or payoff parameters");
            }
            if (!config_.phase_service_profile ||
                !config_.migration_cost_profile)
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration economics require complete service and movement profiles");
            }
        }
        if (config_.economy_layer_catalog)
        {
            if (!migrationEnabled())
            {
                throw std::invalid_argument(
                    "ExpertOverlay economy layer catalog requires Dynamic maintenance");
            }
            if (config_.economy_layer_catalog->firstModelLayer() != config_.model_metadata.first_model_layer ||
                config_.economy_layer_catalog->layerCount() !=
                static_cast<std::size_t>(
                    config_.model_metadata.num_layers))
            {
                throw std::invalid_argument(
                    "ExpertOverlay economy layer catalog does not cover the complete model layer geometry");
            }
        }

        /*
         * Epoch one is the caller's exact declared layout. Successor epochs
         * use the histogram planner only when durable movement is enabled.
         * Keeping a distinct template is what permits a StaticById,
         * ExplicitMasks, or adversarial initial layout to improve over time
         * without lying about how the first physical banks were apportioned.
         */
        if (migrationEnabled())
        {
            planning_template_.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
        }

        MoERoutedExpertPlacementPlan initial = config_.initial_plan;
        if (initial.placements.empty())
        {
            /*
             * Initial residency is deterministic by expert id even for a
             * dynamic policy. Runtime evidence cannot authorize movement before
             * any resident weights exist. A later proposal consumes the live
             * histogram and moves from this installed baseline.
             */
            MoERoutedExpertPlacementPlan initial_template = initial;
            if (isDynamicPolicy(initial_template.residency_policy))
            {
                initial_template.residency_policy =
                    RoutedExpertResidencyPolicy::StaticById;
            }
            initial = MoERoutedExpertPlacementPlanner::plan(
                          initial_template,
                          config_.model_metadata)
                          .planned_plan;
            initial.residency_policy = config_.initial_plan.residency_policy;
        }

        auto initial_snapshot = buildSnapshot(
            1,
            std::move(initial),
            config_.model_metadata,
            nullptr,nullptr,config_.projection_preparation);

        no_peer_residency_ = MoEOverlayNoPeerResidency::seal(
            *initial_snapshot->placement_plan, initial_snapshot->originalOwnerMap(), config_.model_metadata);
        stationary_tensor_shard_residency_ = MoEOverlayStationaryTensorShardResidency::seal(
            *initial_snapshot->placement_plan, initial_snapshot->originalOwnerMap(), config_.model_metadata);
        if (!stationary_tensor_shard_residency_)
            economy_state_ = buildEconomyState(config_, *initial_snapshot);
        if (economy_state_ && config_.histogram)
            config_.histogram->activatePrecertifiedOptimizationDemand();
        economy_activation_state_.store(
            economy_state_
                ? EconomyActivationState::Active
                : EconomyActivationState::CollectingEvidence,
            std::memory_order_release);
        initial_snapshot_ = initial_snapshot;
        published_epoch_.store(
            std::make_shared<PublishedEpochState>(std::move(initial_snapshot)),
            std::memory_order_release);
    }

    MoEOverlayResidencyAuthority::~MoEOverlayResidencyAuthority() = default;

    std::optional<MoEOverlayResidencyAuthority::TicketLease>
    MoEOverlayResidencyAuthority::tryAcquireTicketSnapshot()
    {
        return tryAcquireCurrentSnapshot(
            TicketLeasePurpose::InferenceDispatch);
    }

    std::optional<MoEOverlayResidencyAuthority::TicketLease>
    MoEOverlayResidencyAuthority::tryAcquireGraphSequenceSnapshot()
    {
        return tryAcquireCurrentSnapshot(
            TicketLeasePurpose::InferenceGraphSequence);
    }

    MoEOverlayResidencyAuthority::TicketLease
    MoEOverlayResidencyAuthority::borrowGraphSequenceSnapshot(const TicketLease& parent)
    {
        if (parent.authority_ != this ||
            parent.purpose_ != TicketLeasePurpose::InferenceGraphSequence ||
            !parent.epoch_state_ || !parent.snapshot_)
            throw std::invalid_argument("Graph-sequence descendant requires its live original sequence parent");
        const auto state = parent.epoch_state_;
        if (state->snapshot != parent.snapshot_ ||
            published_epoch_.load(std::memory_order_acquire) != state ||
            state->active_graph_sequences.load(std::memory_order_acquire) == 0)
            throw std::logic_error("Graph-sequence parent lost its original published epoch");

        // The live parent already prevents publication. Unlike a fresh reader,
        // this descendant must not wait on the gate its own parent is draining.
        // The caller serializes parent destruction, so its nonzero ownership
        // spans all increments and closes the publisher's zero-reader race.
        state->active_tickets.fetch_add(1, std::memory_order_acq_rel);
        state->active_graph_sequences.fetch_add(1, std::memory_order_acq_rel);
        active_ticket_count_.fetch_add(1, std::memory_order_acq_rel);
        TicketLease child(this, state, TicketLeasePurpose::InferenceGraphSequence);
        PerfStatsCollector::addCounter("moe_overlay_residency", "graph_sequence_epoch_leases",
            1.0, "inference", config_.perf_device,
            {{"action", "acquire"}, {"epoch", std::to_string(child.epoch())}});
        return child;
    }
    std::optional<MoEOverlayResidencyAuthority::TicketLease>
    MoEOverlayResidencyAuthority::tryAcquireCurrentSnapshot(
        TicketLeasePurpose purpose)
    {
        for (;;)
        {
            auto candidate = published_epoch_.load(std::memory_order_acquire);
            if (!candidate || !candidate->snapshot ||
                !candidate->snapshot->valid())
            {
                throw std::logic_error(
                    "MoE overlay residency authority has no valid published snapshot");
            }

            const bool graph_sequence =
                purpose == TicketLeasePurpose::InferenceGraphSequence;
            if (graph_sequence)
            {
                const auto admission =
                    candidate->graph_sequence_admission.load(
                        std::memory_order_acquire);
                if (admission == PublishedEpochState::
                                     GraphSequenceAdmissionState::
                                         PublicationReserved)
                {
                    graph_sequence_boundary_waits_.fetch_add(
                        1, std::memory_order_relaxed);
                    PerfStatsCollector::addCounter(
                        "moe_overlay_residency",
                        "graph_sequence_boundary_waits",
                        1.0,
                        "inference",
                        config_.perf_device,
                        {{"epoch", std::to_string(candidate->snapshot->epoch)}});
                    candidate->graph_sequence_admission.wait(
                        admission, std::memory_order_acquire);
                    continue;
                }
                if (admission != PublishedEpochState::
                                     GraphSequenceAdmissionState::Open)
                {
                    ticket_acquire_retries_.fetch_add(
                        1, std::memory_order_relaxed);
                    continue;
                }
            }

            /*
             * Pin before rechecking publication. If publication won the race,
             * this provisional pin is dropped and no inference-visible lease
             * is returned. If our recheck wins, a later publisher necessarily
             * observes the pin before deciding whether the old bank can retire.
             */
            candidate->active_tickets.fetch_add(1, std::memory_order_acq_rel);
            if (graph_sequence)
            {
                candidate->active_graph_sequences.fetch_add(
                    1, std::memory_order_acq_rel);
            }
            const bool publication_unchanged =
                published_epoch_.load(std::memory_order_acquire) == candidate;
            const bool graph_sequence_still_admitted =
                !graph_sequence ||
                candidate->graph_sequence_admission.load(
                    std::memory_order_acquire) ==
                    PublishedEpochState::GraphSequenceAdmissionState::Open;
            if (publication_unchanged && graph_sequence_still_admitted)
            {
                active_ticket_count_.fetch_add(1, std::memory_order_acq_rel);
                TicketLease lease(
                    this,
                    std::move(candidate),
                    purpose);
                if (purpose == TicketLeasePurpose::InferenceGraphSequence)
                {
                    PerfStatsCollector::addCounter(
                        "moe_overlay_residency",
                        "graph_sequence_epoch_leases",
                        1.0,
                        "inference",
                        config_.perf_device,
                        {{"action", "acquire"},
                         {"epoch", std::to_string(lease.epoch())}});
                }
                return lease;
            }

            const uint64_t previous =
                candidate->active_tickets.fetch_sub(1, std::memory_order_acq_rel);
            if (previous == 0)
                std::terminate();
            if (graph_sequence)
            {
                const uint64_t previous_graph_sequences =
                    candidate->active_graph_sequences.fetch_sub(
                        1, std::memory_order_acq_rel);
                if (previous_graph_sequences == 0)
                    std::terminate();
            }
            ticket_acquire_retries_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    std::optional<MoEOverlayResidencyAuthority::TicketLease>
    MoEOverlayResidencyAuthority::tryAcquireTicketSnapshot(uint64_t epoch)
    {
        if (epoch == 0)
            return std::nullopt;

        for (;;)
        {
            auto candidate = published_epoch_.load(std::memory_order_acquire);
            if (!candidate || !candidate->snapshot ||
                candidate->snapshot->epoch != epoch)
            {
                candidate = candidate_epoch_.load(std::memory_order_acquire);
            }
            if (!candidate || !candidate->snapshot ||
                candidate->snapshot->epoch != epoch)
            {
                candidate = retiring_epoch_.load(std::memory_order_acquire);
            }
            if (!candidate || !candidate->snapshot ||
                candidate->snapshot->epoch != epoch ||
                !candidate->accepting_exact_tickets.load(
                    std::memory_order_acquire))
            {
                return std::nullopt;
            }

            /*
             * Pin before rechecking both lookup slots and the retirement gate.
             * If retirement won, this provisional count is dropped without
             * exposing a lease.  If acquisition won, retirement's final count
             * observation necessarily sees us.
             */
            candidate->active_tickets.fetch_add(1, std::memory_order_acq_rel);
            const bool still_addressable =
                candidate->accepting_exact_tickets.load(
                    std::memory_order_acquire) &&
                (published_epoch_.load(std::memory_order_acquire) == candidate ||
                 candidate_epoch_.load(std::memory_order_acquire) == candidate ||
                 retiring_epoch_.load(std::memory_order_acquire) == candidate);
            if (still_addressable)
            {
                active_ticket_count_.fetch_add(1, std::memory_order_acq_rel);
                return TicketLease(
                    this,
                    std::move(candidate),
                    TicketLeasePurpose::InferenceDispatch);
            }

            const uint64_t previous =
                candidate->active_tickets.fetch_sub(1, std::memory_order_acq_rel);
            if (previous == 0)
                std::terminate();
            ticket_acquire_retries_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    std::optional<MoEOverlayResidencyAuthority::TicketLease>
    MoEOverlayResidencyAuthority::tryAcquireCurrentBatchLLEPLease(
        int layer_idx,
        std::string_view domain_name,
        int domain_participant_count,
        std::span<uint32_t> owner_participants_out,
        std::string *error)
    {
        if (error)
            error->clear();
        const auto reject = [&](std::string message)
            -> std::optional<TicketLease>
        {
            current_batch_llep_lease_rejections_.fetch_add(
                1, std::memory_order_relaxed);
            if (error)
                *error = std::move(message);
            return std::nullopt;
        };

        if(config_.projection_preparation && config_.projection_preparation->jointOwnerMap())
            return reject("Current-batch LLEP cannot reinterpret complete CPU cohort fragment ownership");
        if (layer_idx < 0 || domain_name.empty() ||
            domain_participant_count <= 1)
        {
            return reject(
                "Current-batch LLEP requires a routed layer, named domain, "
                "and at least two domain participants");
        }

        auto lease = tryAcquireTicketSnapshot();
        if (!lease || !*lease)
            return reject("Current-batch LLEP could not pin a durable epoch");
        const auto &snapshot = **lease;
        if (layer_idx >= snapshot.layered_ownership.layerCount() ||
            owner_participants_out.size() !=
                static_cast<std::size_t>(
                    snapshot.layered_ownership.expertCount()))
        {
            return reject(
                "Current-batch LLEP owner storage does not match the durable "
                "epoch geometry");
        }

        int observed_domain_participants = 0;
        for (const auto &participant : snapshot.owner_map.participants())
        {
            if (participant.domain_name != domain_name)
                continue;
            ++observed_domain_participants;
            if (participant.domain_participant_index < 0 ||
                participant.domain_participant_index >=
                    domain_participant_count)
            {
                return reject(
                    "Current-batch LLEP domain participant index is outside "
                    "the declared transaction domain");
            }
        }
        if (observed_domain_participants != domain_participant_count)
        {
            return reject(
                "Current-batch LLEP participant count disagrees with the "
                "durable epoch domain");
        }

        for (int expert = 0;
             expert < snapshot.layered_ownership.expertCount();
             ++expert)
        {
            const int global_owner =
                snapshot.layered_ownership.owner(layer_idx, expert);
            const auto *owner =
                snapshot.owner_map.participantForId(global_owner);
            if (!owner || owner->domain_name != domain_name ||
                owner->domain_participant_index < 0 ||
                owner->domain_participant_index >=
                    domain_participant_count)
            {
                return reject(
                    "Current-batch LLEP cannot borrow an expert whose durable "
                    "owner lies outside its routed domain");
            }
            owner_participants_out[static_cast<std::size_t>(expert)] =
                static_cast<uint32_t>(
                    owner->domain_participant_index);
        }

        lease->purpose_ = TicketLeasePurpose::CurrentBatchLLEP;
        current_batch_llep_leases_acquired_.fetch_add(
            1, std::memory_order_relaxed);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "current_batch_llep_leases_acquired",
            1.0,
            "prefill",
            config_.perf_device,
            {{"epoch", std::to_string(lease->epoch())},
             {"layer", std::to_string(layer_idx)},
             {"domain", std::string(domain_name)},
             {"participants", std::to_string(domain_participant_count)}});
        return lease;
    }

    std::shared_ptr<const MoEOverlayResidencySnapshot>
    MoEOverlayResidencyAuthority::snapshot() const
    {
        const auto current = published_epoch_.load(std::memory_order_acquire);
        return current ? current->snapshot : nullptr;
    }

    RoutedExpertResidencyPolicy
    MoEOverlayResidencyAuthority::residencyPolicy() const noexcept
    {
        return config_.initial_plan.residency_policy;
    }

    MoERebalanceRuntimeMode
    MoEOverlayResidencyAuthority::maintenanceMode() const noexcept
    {
        return config_.maintenance_mode;
    }

    bool MoEOverlayResidencyAuthority::observesHistogram() const noexcept
    {
        return config_.maintenance_mode != MoERebalanceRuntimeMode::Off &&
            !(config_.maintenance_mode == MoERebalanceRuntimeMode::Dynamic &&
              hasStationaryTensorShardTopology(config_.initial_plan));
    }

    bool MoEOverlayResidencyAuthority::migrationEnabled() const noexcept
    {
        return config_.maintenance_mode == MoERebalanceRuntimeMode::Dynamic &&
               !stationary_tensor_shard_residency_;
    }

    void MoEOverlayResidencyAuthority::
        requireHostDynamicPublicationAuthority(
            const char *operation) const
    {
        if (jointFragmentPlan(config_.initial_plan) && config_.maintenance_mode == MoERebalanceRuntimeMode::Off)
            throw std::logic_error(std::string(operation) + ": Static CPU cohorts have no mutable publication authority");
        if (!migrationEnabled() ||
            config_.initial_plan.authority_execution !=
                MoEOverlayAuthorityExecutionKind::
                    DeviceResident)
        {
            return;
        }

        throw std::logic_error(
            std::string(operation ? operation : "Host ExpertOverlay publication") +
            " is forbidden because the frozen device-resident authority owns the live epoch");
    }

    void MoEOverlayResidencyAuthority::installEconomyCertification(
        std::shared_ptr<const MoERoutedTierServiceProfile> service,
        std::shared_ptr<const MoEOverlayMigrationCostProfile> migration,
        MoEOverlayMigrationEconomyPolicy policy)
    {
        requireHostDynamicPublicationAuthority(
            "Host ExpertOverlay economy certification");
        if (!migrationEnabled())
        {
            throw std::invalid_argument(
                "ExpertOverlay migration economics require Dynamic maintenance");
        }

        std::lock_guard<std::mutex> lock(economy_mutex_);
        if (checks_.load(std::memory_order_acquire) != 0)
        {
            throw std::logic_error(
                "ExpertOverlay economy certification must precede the first dynamic proposal");
        }
        if (economy_activation_state_.load(std::memory_order_acquire) !=
            EconomyActivationState::ReadyForCertification)
        {
            throw std::logic_error(
                "ExpertOverlay economy certification requires a completed calibration-demand rebase");
        }
        if (economy_state_ || config_.phase_service_profile ||
            config_.migration_cost_profile ||
            config_.migration_economy_policy)
        {
            throw std::logic_error(
                "ExpertOverlay economy certification is immutable and may be installed only once");
        }

        Config candidate_config = config_;
        candidate_config.phase_service_profile = std::move(service);
        candidate_config.migration_cost_profile = std::move(migration);
        candidate_config.migration_economy_policy = policy;
        const auto current = snapshot();
        if (!current || !current->valid())
        {
            throw std::logic_error(
                "ExpertOverlay economy certification requires a valid initial residency epoch");
        }

        /*
         * Build the complete flattened state before touching live authority
         * fields. A malformed measurement leaves the authority unchanged and
         * can therefore be diagnosed without creating a partial policy.
         */
        auto candidate_state = buildEconomyState(
            candidate_config,
            *current);
        if (!candidate_state)
        {
            throw std::invalid_argument(
                "ExpertOverlay economy certification did not contain a complete policy");
        }

        config_.phase_service_profile =
            std::move(candidate_config.phase_service_profile);
        config_.migration_cost_profile =
            std::move(candidate_config.migration_cost_profile);
        config_.migration_economy_policy = policy;
        economy_state_ = std::move(candidate_state);
        economy_activation_state_.store(
            EconomyActivationState::CertifiedAwaitingRequestBoundary,
            std::memory_order_release);

        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "economy_certifications",
            1.0,
            "model_setup",
            config_.perf_device,
            {{"service_profile", config_.phase_service_profile->identity},
             {"migration_profile", config_.migration_cost_profile->identity},
             {"immutable", "true"},
             {"blocking_inference", "false"},
             {"demand_activation", "next_request_boundary"}});
    }

    bool MoEOverlayResidencyAuthority::hasEconomyCertification() const noexcept
    {
        const auto state =
            economy_activation_state_.load(std::memory_order_acquire);
        return state ==
                   EconomyActivationState::CertifiedAwaitingRequestBoundary ||
               state == EconomyActivationState::Active;
    }

    bool MoEOverlayResidencyAuthority::optimizationDemandActive() const noexcept
    {
        return economy_activation_state_.load(std::memory_order_acquire) ==
               EconomyActivationState::Active;
    }

    MoEOverlayDemandActivationResult
    MoEOverlayResidencyAuthority::activateOptimizationDemandAtRequestBoundary()
    {
        if (!migrationEnabled())
            return MoEOverlayDemandActivationResult::NotReady;

        /*
         * Active is terminal for this authority's lifetime. The release store
         * which publishes it also publishes the opened histogram bank, so an
         * acquire read is sufficient for every subsequent request. Taking the
         * economy lock here would serialize prefill behind the entire swap
         * search on the maintenance worker, even though nothing needs changing.
         *
         * Before certification is published there is likewise no transition to
         * perform. Only the first boundary observing a complete certificate
         * takes the lock; recheck below because two callers may reach it.
         */
        const auto published =
            economy_activation_state_.load(std::memory_order_acquire);
        if (published == EconomyActivationState::Active)
            return MoEOverlayDemandActivationResult::AlreadyActive;
        if (published !=
            EconomyActivationState::CertifiedAwaitingRequestBoundary)
            return MoEOverlayDemandActivationResult::NotReady;

        std::lock_guard<std::mutex> lock(economy_mutex_);
        const auto state =
            economy_activation_state_.load(std::memory_order_acquire);
        if (state == EconomyActivationState::Active)
            return MoEOverlayDemandActivationResult::AlreadyActive;
        if (state !=
            EconomyActivationState::CertifiedAwaitingRequestBoundary)
        {
            return MoEOverlayDemandActivationResult::NotReady;
        }
        if (!economy_state_ || !config_.histogram)
        {
            throw std::logic_error(
                "Request-boundary economy activation requires an installed policy and route histogram");
        }

        config_.histogram->activateOptimizationDemand();
        economy_activation_state_.store(
            EconomyActivationState::Active,
            std::memory_order_release);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "optimization_demand_activations",
            1.0,
            "request_admission",
            config_.perf_device,
            {{"authority", "host"},
             {"blocking", "false"},
             {"boundary", "prefill"}});
        return MoEOverlayDemandActivationResult::Activated;
    }

    bool MoEOverlayResidencyAuthority::maintenanceWindowReady() const noexcept
    {
        const auto drain_state =
            histogram_drain_state_.load(std::memory_order_acquire);
        return migrationEnabled() &&
               config_.initial_plan.authority_execution !=
                   MoEOverlayAuthorityExecutionKind::
                       DeviceResident &&
               config_.histogram &&
               (drain_state == HistogramDrainState::ProposalWindow ||
                config_.histogram->windowFull());
    }

    MoEOptimizationDemandWindow
    MoEOverlayResidencyAuthority::optimizationDemandWindow() const noexcept
    {
        return config_.histogram
                   ? config_.histogram->optimizationDemandWindow()
                   : MoEOptimizationDemandWindow{};
    }

    MoEOverlayHistogramWindowResult
    MoEOverlayResidencyAuthority::progressHistogramWindow(
        MoEOverlayHistogramEvidenceScope evidence_scope)
    {
        requireHostDynamicPublicationAuthority(
            "Host ExpertOverlay histogram progress");
        if (!migrationEnabled() || !config_.histogram)
        {
            throw std::logic_error(
                "Only Dynamic ExpertOverlay maintenance may prepare a routing window");
        }
        return progressHistogramDrain(
            HistogramDrainState::ProposalWindow, evidence_scope);
    }

    MoEOverlayHistogramRebaseProgress
    MoEOverlayResidencyAuthority::progressEconomyEvidenceRebase()
    {
        requireHostDynamicPublicationAuthority(
            "Host ExpertOverlay economy-evidence rebase");
        if (!migrationEnabled() || !config_.histogram)
        {
            throw std::logic_error(
                "Economy-evidence rebase requires one uncertified Dynamic host authority");
        }
        auto activation = economy_activation_state_.load(
            std::memory_order_acquire);
        if (activation == EconomyActivationState::ReadyForCertification)
            return MoEOverlayHistogramRebaseProgress::Complete;
        if (activation != EconomyActivationState::CollectingEvidence &&
            activation !=
                EconomyActivationState::RebasingRoutingEvidence)
        {
            throw std::logic_error(
                "Economy-evidence rebase requires one uncertified Dynamic host authority");
        }
        if (activation == EconomyActivationState::CollectingEvidence)
        {
            EconomyActivationState expected =
                EconomyActivationState::CollectingEvidence;
            if (!economy_activation_state_.compare_exchange_strong(
                    expected,
                    EconomyActivationState::RebasingRoutingEvidence,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire) &&
                expected !=
                    EconomyActivationState::RebasingRoutingEvidence)
            {
                throw std::logic_error(
                    "ExpertOverlay economy activation changed before its routing-evidence rebase");
            }
            try
            {
                config_.histogram->beginOptimizationDemandRebase();
            }
            catch (...)
            {
                economy_activation_state_.store(
                    EconomyActivationState::CollectingEvidence,
                    std::memory_order_release);
                throw;
            }
        }

        if (config_.histogram->admissionState() !=
            RuntimeExpertHistogramAdmission::CertificationQuarantine)
        {
            throw std::logic_error(
                "Economy-evidence rebase lost its route-admission quarantine");
        }

        try
        {
            const auto result = progressHistogramDrain(
                HistogramDrainState::CertificationRebase,
                MoEOverlayHistogramEvidenceScope::RuntimeSources);
            if (result.progress !=
                MoEOverlayHistogramWindowProgress::Ready)
            {
                return MoEOverlayHistogramRebaseProgress::Pending;
            }
            economy_activation_state_.store(
                EconomyActivationState::ReadyForCertification,
                std::memory_order_release);
            return MoEOverlayHistogramRebaseProgress::Complete;
        }
        catch (...)
        {
            economy_activation_state_.store(
                EconomyActivationState::CollectingEvidence,
                std::memory_order_release);
            throw;
        }
    }

    MoEOverlayHistogramWindowResult
    MoEOverlayResidencyAuthority::progressHistogramDrain(
        HistogramDrainState requested_state,
        MoEOverlayHistogramEvidenceScope evidence_scope)
    {
        if (requested_state != HistogramDrainState::ProposalWindow &&
            requested_state != HistogramDrainState::CertificationRebase)
        {
            throw std::logic_error(
                "ExpertOverlay histogram drain requires one explicit purpose");
        }

        auto current =
            histogram_drain_state_.load(std::memory_order_acquire);
        if (current == HistogramDrainState::Idle)
        {
            if (requested_state == HistogramDrainState::ProposalWindow &&
                !config_.histogram->windowFull() &&
                evidence_scope ==
                    MoEOverlayHistogramEvidenceScope::HostResident)
            {
                return {};
            }
            HistogramDrainState expected = HistogramDrainState::Idle;
            if (!histogram_drain_state_.compare_exchange_strong(
                    expected,
                    requested_state,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire))
            {
                current = expected;
            }
            else
            {
                current = requested_state;
            }
        }
        if (current != requested_state)
        {
            throw std::logic_error(
                "ExpertOverlay histogram drain purpose changed while a drain was active");
        }

        const auto drain =
            config_.histogram->progressRuntimeHistogramDrains();
        if (drain.progress ==
            RuntimeExpertHistogramDrainProgress::Pending)
        {
            return {
                .progress = MoEOverlayHistogramWindowProgress::Pending,
            };
        }
        if (drain.progress ==
            RuntimeExpertHistogramDrainProgress::Failed)
        {
            histogram_drain_state_.store(
                HistogramDrainState::Idle, std::memory_order_release);
            throw std::runtime_error(
                drain.error.empty()
                    ? "Failed to drain runtime expert histograms at the overlay maintenance boundary"
                    : drain.error);
        }
        if (!config_.histogram->syncRuntimeHistograms())
        {
            histogram_drain_state_.store(
                HistogramDrainState::Idle, std::memory_order_release);
            throw std::runtime_error(
                "Failed to merge synchronous runtime expert histograms at the overlay maintenance boundary");
        }

        /* Runtime counters can be authoritative while the host RCU bank is
         * still partial. A cadence probe reconciles and resets those device
         * banks, but it must not manufacture a proposal or advance adaptive
         * window policy until the merged host total reaches its threshold. */
        if (requested_state == HistogramDrainState::ProposalWindow &&
            !config_.histogram->windowFull())
        {
            histogram_drain_state_.store(
                HistogramDrainState::Idle, std::memory_order_release);
            return {
                .progress =
                    MoEOverlayHistogramWindowProgress::Reconciled,
            };
        }

        auto window =
            std::make_shared<DecodeExpertHistogramWindow>(
                config_.histogram->freezeAndRotateWindow());
        if (!window->valid() ||
            window->num_layers != config_.model_metadata.num_layers ||
            window->first_model_layer != config_.model_metadata.first_model_layer ||
            window->num_experts != config_.model_metadata.num_experts)
        {
            /* The completed drain has no resumable work once its bank has
             * rotated.  Return the typed lifecycle to Idle, but never advance
             * the adaptive policy from malformed evidence. */
            histogram_drain_state_.store(
                HistogramDrainState::Idle, std::memory_order_release);
            throw std::runtime_error(
                "Frozen ExpertOverlay histogram does not match model geometry");
        }
        histogram_drain_state_.store(
            HistogramDrainState::Idle, std::memory_order_release);
        return {
            .progress = MoEOverlayHistogramWindowProgress::Ready,
            .window = std::move(window),
        };
    }

    void MoEOverlayResidencyAuthority::
        advanceHistogramWindowAfterProposal(
            const MoEOverlayResidencyTransaction &transaction)
    {
        if (!config_.histogram || transaction.purpose !=
                MoEOverlayResidencyTransactionPurpose::PlacementChange ||
            !transaction.previous || !transaction.candidate ||
            !transaction.histogram_window ||
            initial_histogram_window_tokens_ <= 0)
        {
            throw std::logic_error(
                "ExpertOverlay histogram cadence requires one valid observed placement proposal");
        }

        if (transaction.empty())
        {
            const bool observed_demand =
                transaction.histogram_window->token_count != 0u ||
                std::any_of(
                    transaction.histogram_window->expert_counts.begin(),
                    transaction.histogram_window->expert_counts.end(),
                    [](const std::uint64_t count) { return count != 0u; });
            if (!observed_demand)
                return;
            growHistogramWindowAfterObservedNoMovement();
            return;
        }

        const int current = config_.histogram->windowSize();
        if (current == initial_histogram_window_tokens_)
            return;

        /* A published move changes the state evaluated by the next scan. Start
         * another short evidence window so independent tier and participant
         * objectives converge before the controller returns to cooldown. */
        config_.histogram->setWindowSize(initial_histogram_window_tokens_);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "histogram_window_reset_after_movement",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"previous_tokens", std::to_string(current)},
             {"next_tokens",
              std::to_string(initial_histogram_window_tokens_)},
             {"histogram_generation",
              std::to_string(transaction.histogram_generation)},
             {"policy_owner", "host"},
             {"blocking_inference", "false"}});
    }

    void MoEOverlayResidencyAuthority::
        growHistogramWindowAfterObservedNoMovement()
    {
        if (!config_.histogram ||
            config_.histogram_max_window_tokens == 0u ||
            config_.histogram_window_growth_factor <= 1.0)
        {
            return;
        }

        const std::uint64_t current = static_cast<std::uint64_t>(
            config_.histogram->windowSize());
        const std::uint64_t maximum =
            config_.histogram_max_window_tokens;
        if (current >= maximum)
            return;

        const long double scaled =
            static_cast<long double>(current) *
            static_cast<long double>(
                config_.histogram_window_growth_factor);
        const std::uint64_t next =
            scaled >= static_cast<long double>(maximum)
                ? maximum
                : std::max<std::uint64_t>(
                      current + 1u,
                      static_cast<std::uint64_t>(std::ceil(scaled)));
        config_.histogram->setWindowSize(static_cast<int>(next));
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "histogram_window_growth",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"previous_tokens", std::to_string(current)},
             {"next_tokens", std::to_string(next)},
             {"maximum_tokens", std::to_string(maximum)},
             {"growth_factor",
              std::to_string(
                  config_.histogram_window_growth_factor)},
             {"policy_owner", "host"},
             {"blocking_inference", "false"}});
    }

    MoEOverlayResidencyTransaction
    MoEOverlayResidencyAuthority::proposeFromHistogram()
    {
        requireHostDynamicPublicationAuthority(
            "Host ExpertOverlay histogram proposal");
        if (!migrationEnabled())
        {
            checks_.fetch_add(1, std::memory_order_relaxed);
            const auto previous = snapshot();
            if (!previous || !previous->valid())
            {
                throw std::logic_error(
                    "Cannot propose MoE overlay residency without a valid snapshot");
            }

            MoEOverlayResidencyTransaction transaction;
            transaction.expected_epoch = previous->epoch;
            transaction.previous = previous;
            // Reuse the authority's checked unchanged-successor constructor;
            // a public epoch scalar cannot advance the private joint arm.
            transaction.candidate = buildCalibrationShadowSnapshot(previous);
            return transaction;
        }
        if (hasEconomyCertification() && !optimizationDemandActive())
        {
            throw std::logic_error(
                "ExpertOverlay cannot propose from a sealed economy profile before request-boundary demand activation");
        }

        return proposeFromFrozenHistogramWindow(
            freezeAndRotateHistogramWindow());
    }

    bool MoEOverlayResidencyAuthority::initialPreparedPlacementPublished()
        const
    {
        const auto current = snapshot();
        if (current && current->ownershipKind() == MoEOverlayResidencyOwnershipKind::JointFragments)
        {
            if (!current->valid() || !initial_snapshot_ || !initial_snapshot_->valid())
                return false;
            // Epoch identity changes on restoration; the complete fixed target
            // assignments, rather than the original publication pointer, define reuse.
            const auto &live = current->placementOwnerMap().owners();
            const auto &original = initial_snapshot_->placementOwnerMap().owners();
            return live == original;
        }
        if (!current || !current->valid() || !initial_snapshot_ ||
            !initial_snapshot_->valid() ||
            current->layered_ownership !=
                initial_snapshot_->layered_ownership)
        {
            return false;
        }

        const auto &current_placements =
            current->placement_plan->placements;
        const auto &initial_placements =
            initial_snapshot_->placement_plan->placements;
        if (current_placements.size() != initial_placements.size())
            return false;
        for (std::size_t index = 0; index < current_placements.size(); ++index)
        {
            if (current_placements[index].layer !=
                    initial_placements[index].layer ||
                current_placements[index].routed_expert_tier !=
                    initial_placements[index].routed_expert_tier)
            {
                return false;
            }
        }
        return true;
    }

    MoEOverlayResidencyTransaction
    MoEOverlayResidencyAuthority::
        proposeInitialPreparedPlacementRestoration()
    {
        requireHostDynamicPublicationAuthority(
            "Host ExpertOverlay prepared-context restoration proposal");
        if (!migrationEnabled())
        {
            throw std::logic_error(
                "Prepared-context restoration requires Dynamic ExpertOverlay maintenance");
        }

        return buildInitialPreparedPlacementRestoration();
    }

    MoEOverlayResidencyTransaction
    MoEOverlayResidencyAuthority::buildInitialPreparedPlacementRestoration() const
    {
        const auto previous = snapshot();
        if (!previous || !previous->valid() || !initial_snapshot_ ||
            !initial_snapshot_->valid())
        {
            throw std::logic_error(
                "Prepared-context restoration requires valid current and initial snapshots");
        }

        const auto build_transaction =
            [&](std::shared_ptr<const MoEOverlayResidencySnapshot> candidate)
        {
            MoEOverlayResidencyTransaction result;
            result.purpose = MoEOverlayResidencyTransactionPurpose::
                PreparedContextRestoration;
            result.expected_epoch = previous->epoch;
            result.previous = previous;
            result.candidate = std::move(candidate);
            std::vector<MoELayeredExpertOwnershipChange> participant_restoration;
            if (previous->originalProjectionPreparation())
                for (const auto &destination : result.candidate->placementOwnerMap().owners())
                {
                    const auto *source = previous->placementOwnerMap().placementFor(destination.layer_idx, destination.expert_id);
                    if (source && source->tier_idx == destination.tier_idx && !source->shard_cohort &&
                        !destination.shard_cohort && source->owner_participant != destination.owner_participant)
                        participant_restoration.push_back({.layer_idx = destination.layer_idx,
                            .expert_id = destination.expert_id, .previous_participant = source->owner_participant,
                            .current_participant = destination.owner_participant});
                }
            result.migrations = buildMigrations(*previous, *result.candidate, nullptr,
                /*estimated_weight_bytes=*/0u, participant_restoration);
            result.migration_cycles = buildMoEOverlayMigrationCycles(
                result.migrations);
            result.shadow_requirements = buildMoEOverlayShadowRequirements(
                result.migrations);
            if (!result.valid())
            {
                throw std::logic_error(
                    "Prepared-context restoration produced an invalid residency transaction");
            }
            return result;
        };

        const bool joint = previous->ownershipKind() ==
                           MoEOverlayResidencyOwnershipKind::JointFragments;
        auto target = joint
            ? prepareJointPlacementCandidate(previous, previous->epoch,
                  initial_snapshot_->placement_plan->placements)
            : buildSnapshot(
            previous->epoch + 1u,
            *initial_snapshot_->placement_plan,
            config_.model_metadata,
            &previous->owner_map,
            &initial_snapshot_->layered_ownership);
        if (joint)
        {
            const auto changes = wholeOwnerTargetChanges(target->placementOwnerMap(),
                initial_snapshot_->placementOwnerMap().owners());
            if (!changes.empty())
                target = prepareJointPlacementCandidate(previous, previous->epoch,
                    initial_snapshot_->placement_plan->placements, changes);
        }
        requireCapacityPreserving(*previous, *target);
        auto full = build_transaction(target);
        if (full.empty())
            return full;

        const auto fits_wave_budget =
            [&](const MoEOverlayResidencyTransaction &candidate)
        {
            if (config_.max_concurrent_cycles != 0u &&
                candidate.migration_cycles.size() >
                    config_.max_concurrent_cycles)
            {
                return false;
            }
            return std::all_of(
                candidate.shadow_requirements.begin(),
                candidate.shadow_requirements.end(),
                [&](const auto &requirement)
                {
                    return config_.shadow_slots_per_endpoint_layer == 0u ||
                           requirement.slot_count <=
                               config_.shadow_slots_per_endpoint_layer;
                });
        };
        if (fits_wave_budget(full))
            return full;

        /*
         * A restoration is decomposed only at closed-cycle boundaries. Start
         * from the live state and add deterministic target cycles until the
         * exact preallocated wave BOM is full. Rebuilding the candidate after
         * each addition is intentional: overlapping logical cycles may merge
         * into a different physical cycle set, and the transport must be
         * admitted against that recomposed reality.
         */
        std::vector<std::size_t> selected_cycles;
        std::optional<MoEOverlayResidencyTransaction> bounded;
        for (std::size_t cycle_index = 0;
             cycle_index < full.migration_cycles.size();
             ++cycle_index)
        {
            auto trial_cycles = selected_cycles;
            trial_cycles.push_back(cycle_index);

            MoERoutedExpertPlacementPlan trial_plan =
                *initial_snapshot_->placement_plan;
            trial_plan.placements = previous->placement_plan->placements;
            MoELayeredExpertOwnership trial_ownership = previous->layered_ownership;
            std::vector<MoEExpertOwner> exact_targets;
            for (const std::size_t selected : trial_cycles)
            {
                const auto &cycle = full.migration_cycles.at(selected);
                for (const std::size_t migration_index :
                     cycle.migration_indices)
                {
                    const auto &migration =
                        full.migrations.at(migration_index);
                    const auto placement = std::find_if(
                        trial_plan.placements.begin(),
                        trial_plan.placements.end(),
                        [&](const auto &entry)
                        { return entry.layer == migration.layer_idx; });
                    if (placement == trial_plan.placements.end() ||
                        migration.expert_id < 0 ||
                        migration.expert_id >= static_cast<int>(
                            placement->routed_expert_tier.size()))
                    {
                        throw std::logic_error(
                            "Prepared-context restoration cycle references invalid placement geometry");
                    }
                    placement->routed_expert_tier[
                        static_cast<std::size_t>(migration.expert_id)] =
                        migration.destination.tier_idx;
                    if (joint) exact_targets.push_back(migration.destination);
                    else trial_ownership.assignOwner(
                            migration.layer_idx,
                            migration.expert_id,
                            migration.destination.owner_participant);
                }
            }

            auto trial_snapshot = joint
                ? prepareJointPlacementTransaction(previous, previous->epoch,
                      trial_plan.placements).candidate
                : buildSnapshot(
                previous->epoch + 1u,
                std::move(trial_plan),
                config_.model_metadata,
                &previous->owner_map,
                &trial_ownership);
            if (joint)
            {
                const auto changes = wholeOwnerTargetChanges(trial_snapshot->placementOwnerMap(), exact_targets);
                if (!changes.empty())
                    trial_snapshot = prepareJointPlacementCandidate(previous, previous->epoch,
                        trial_plan.placements, changes);
            }
            requireCapacityPreserving(*previous, *trial_snapshot);
            auto trial = build_transaction(std::move(trial_snapshot));
            if (trial.empty() || !fits_wave_budget(trial))
                continue;
            selected_cycles = std::move(trial_cycles);
            bounded = std::move(trial);
            if (config_.max_concurrent_cycles != 0u &&
                bounded->migration_cycles.size() >=
                    config_.max_concurrent_cycles)
            {
                break;
            }
        }

        if (!bounded)
        {
            throw std::runtime_error(
                "Prepared-context restoration cannot express one closed cycle within the configured shadow-slot and concurrency BOM");
        }
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "prepared_context_restoration_bounded_waves",
            1.0,
            "model_teardown",
            config_.perf_device,
            {{"target_cycles", std::to_string(full.migration_cycles.size())},
             {"admitted_cycles",
              std::to_string(bounded->migration_cycles.size())},
             {"max_concurrent_cycles",
              std::to_string(config_.max_concurrent_cycles)}});
        return std::move(*bounded);
    }

    std::shared_ptr<const DecodeExpertHistogramWindow>
    MoEOverlayResidencyAuthority::freezeAndRotateHistogramWindow()
    {
        requireHostDynamicPublicationAuthority(
            "Host ExpertOverlay histogram rotation");
        if (!migrationEnabled() || !config_.histogram)
        {
            throw std::logic_error(
                "Only Dynamic ExpertOverlay maintenance may freeze a routing window");
        }
        if (!config_.histogram->syncRuntimeHistograms())
        {
            throw std::runtime_error(
                "Failed to synchronize runtime expert histograms at the overlay maintenance boundary");
        }

        auto window =
            std::make_shared<DecodeExpertHistogramWindow>(
                config_.histogram->freezeAndRotateWindow());
        if (!window->valid() ||
            window->num_layers != config_.model_metadata.num_layers ||
            window->first_model_layer != config_.model_metadata.first_model_layer ||
            window->num_experts != config_.model_metadata.num_experts)
        {
            throw std::runtime_error(
                "Frozen ExpertOverlay histogram does not match model geometry");
        }
        return window;
    }

    MoEOverlayResidencyTransaction
    MoEOverlayResidencyAuthority::proposeFromFrozenHistogramWindow(
        std::shared_ptr<const DecodeExpertHistogramWindow> window)
    {
        requireHostDynamicPublicationAuthority(
            "Host ExpertOverlay frozen-window proposal");
        if (!migrationEnabled())
        {
            throw std::logic_error(
                "ExpertOverlay histogram proposals require Dynamic maintenance");
        }
        if (hasEconomyCertification() && !optimizationDemandActive())
        {
            throw std::logic_error(
                "ExpertOverlay cannot propose from a sealed economy profile before request-boundary demand activation");
        }
        auto result = buildFromFrozenHistogramWindow(std::move(window));
        // The derivation retains the original economy lock through cadence.
        advanceHistogramWindowAfterProposal(result.transaction);
        return std::move(result.transaction);
    }

    MoEOverlayResidencyAuthority::CommittedObservation
    MoEOverlayResidencyAuthority::retainCommittedObservation(
        const MoEOverlayResidencyTransaction &committed) const
    {
        requireHostDynamicPublicationAuthority("Host ExpertOverlay committed observation");
        const auto current = snapshot();
        const CommittedObservation observation(committed.histogram_window, committed.candidate);
        std::lock_guard<std::mutex> lock(economy_mutex_);
        if (!migrationEnabled() || !committed.valid() || committed.empty() ||
            committed.purpose != MoEOverlayResidencyTransactionPurpose::PlacementChange ||
            !committed.economy.enabled || committed.economy.projected_service_gain_ns == 0u ||
            committed.economy.projected_net_benefit_ns == 0u ||
            !observation.matches(current) || !economy_state_ || !economy_state_->forecast ||
            !economy_state_->forecast->observes(committed.histogram_window))
            throw std::logic_error("ExpertOverlay continuation requires the original certified committed observation");
        return observation;
    }

    bool MoEOverlayResidencyAuthority::isCurrentCommittedObservation(
        const CommittedObservation &observation) const
    {
        return observation.matches(snapshot());
    }

    std::optional<MoEOverlayResidencyTransaction>
    MoEOverlayResidencyAuthority::reproposeFromCommittedObservation(
        const CommittedObservation &observation)
    {
        requireHostDynamicPublicationAuthority("Host ExpertOverlay observation continuation");
        if (!migrationEnabled() || !optimizationDemandActive() ||
            hasActiveBackgroundWave() || pendingRetirementCount() != 0u || pendingAbortCount() != 0u)
            throw std::logic_error("ExpertOverlay continuation requires retired original wave resources");
        auto derived = buildFromFrozenHistogramWindow(observation.window_, &observation);
        const auto &transaction = derived.transaction;
        if (transaction.empty() || !transaction.economy.enabled ||
            transaction.economy.projected_service_gain_ns == 0u ||
            transaction.economy.projected_net_benefit_ns == 0u)
        {
            // A terminal continuation is not a new demand window: in
            // particular it must not grow/reset the mutable next bank.
            recordNoMovement(false);
            return std::nullopt;
        }
        return std::move(derived.transaction);
    }

} // namespace llaminar2
