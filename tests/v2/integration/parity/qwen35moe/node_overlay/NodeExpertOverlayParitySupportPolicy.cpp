/**
 * @file NodeExpertOverlayParitySupportPolicy.cpp
 * @brief SupportPolicy implementation of the shared node-overlay parity fixture.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#include "NodeExpertOverlayParitySupport.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{

    /** Active case has one process-local definition shared by every shard. */
    const ModelParityCase *g_active_model_parity_case = nullptr;

    /** @return The active typed canonical case, or null outside a fixture. */
    const ModelParityCase *activeModelParityCase() noexcept
    {
        return g_active_model_parity_case;
    }

    /** @return Active generated case; throws outside a parameterized fixture. */
    const ModelParityCase &activeModelParityCaseOrThrow()
    {
        const auto *test_case = activeModelParityCase();
        if (!test_case)
        {
            throw std::logic_error(
                "Graph-native parity runtime requires an active typed case");
        }
        return *test_case;
    }

    /**
     * @brief Classify authoritative progress toward one convergence objective.
     *
     * Publication and physical retirement are distinct event edges. Once both
     * have reached the requested wave count, the suffix of the durable ledger
     * must prove every logical movement axis expressible by the frozen
     * topology. Multiple cycles in one wave satisfy multiple axes but never
     * masquerade as multiple publications.
     *
     * @param target Typed publication and movement-axis objective.
     * @param origin Authority position before convergence traffic began.
     * @param status Current passive status from the sole production authority.
     * @param ledger Current complete durable movement ledger.
     * @return Exact lifecycle state without consulting optional instrumentation.
     */
    DynamicResidencyConvergenceState classifyDynamicResidencyConvergence(
        const DynamicResidencyConvergenceTarget &target,
        const DynamicResidencyConvergenceOrigin &origin,
        const MoEOptimizationStatus &status,
        const MoEOptimizationMovementLedger &ledger) noexcept
    {
        if (target.minimum_published_waves == 0u || !ledger.complete() ||
            status.published_movement_waves < origin.published_waves ||
            status.completed_movement.transactions <
                origin.completed_transactions ||
            ledger.edges.size() < origin.ledger_edges ||
            ledger.host_admissions.size() < origin.host_admissions ||
            status.completed_movement.transactions >
                status.published_movement_waves)
        {
            return DynamicResidencyConvergenceState::
                InvalidAuthorityEvidence;
        }

        const std::uint64_t published =
            status.published_movement_waves - origin.published_waves;
        if (published < target.minimum_published_waves)
        {
            return DynamicResidencyConvergenceState::AwaitingPublication;
        }

        const std::uint64_t completed =
            status.completed_movement.transactions -
            origin.completed_transactions;
        if (completed < target.minimum_published_waves)
        {
            return DynamicResidencyConvergenceState::
                AwaitingPhysicalCompletion;
        }

        bool advanced_tier_residency = false;
        bool advanced_participant_placement = false;
        for (std::size_t index = origin.ledger_edges;
             index < ledger.edges.size();
             ++index)
        {
            const auto &edge = ledger.edges[index];
            if (!edge.valid())
            {
                return DynamicResidencyConvergenceState::
                    InvalidAuthorityEvidence;
            }
            advanced_tier_residency =
                advanced_tier_residency || advancesTierResidency(edge.axis);
            advanced_participant_placement =
                advanced_participant_placement ||
                advancesParticipantPlacement(edge.axis);
        }
        if (!advanced_tier_residency)
        {
            return DynamicResidencyConvergenceState::AwaitingTierResidency;
        }
        if (target.axis_contract ==
                DynamicMovementAxisContract::
                    PriorityMigrationAndParticipantBalance &&
            !advanced_participant_placement)
        {
            if (status.authority == MoEOptimizationAuthority::Host)
            {
                /*
                 * Heterogeneous policy may correctly find no economical
                 * within-tier exchange after its tier promotion subset has
                 * already removed the bottleneck. Every completed host wave
                 * carries an immutable admission proof. Require one proof per
                 * completed suffix wave, validate it, and demand a physical
                 * participant edge whenever any candidate survived the exact
                 * economy gates. This keeps a broken or starved eligible
                 * participant cycle red without manufacturing a gratuitous
                 * move for a workload whose exhaustive planner found none.
                 */
                const std::size_t admission_count =
                    ledger.host_admissions.size() -
                    origin.host_admissions;
                if (admission_count <
                    static_cast<std::size_t>(
                        target.minimum_published_waves))
                {
                    return DynamicResidencyConvergenceState::
                        AwaitingParticipantPolicyEvidence;
                }

                bool participant_cycle_was_policy_eligible = false;
                for (std::size_t index = origin.host_admissions;
                     index < ledger.host_admissions.size();
                     ++index)
                {
                    const auto &admission =
                        ledger.host_admissions[index];
                    if (!admission.valid() ||
                        admission.authority !=
                            MoEOptimizationAuthority::Host)
                    {
                        return DynamicResidencyConvergenceState::
                            InvalidAuthorityEvidence;
                    }
                    participant_cycle_was_policy_eligible =
                        participant_cycle_was_policy_eligible ||
                        admission.policy_eligible_axes
                                .participant_placement > 0u ||
                        admission.policy_eligible_axes.combined > 0u;
                }
                if (!participant_cycle_was_policy_eligible)
                    return DynamicResidencyConvergenceState::Satisfied;
            }
            return DynamicResidencyConvergenceState::
                AwaitingParticipantPlacement;
        }
        return DynamicResidencyConvergenceState::Satisfied;
    }

    /**
     * @brief Classify whether the durable suffix moved a reference-routed expert.
     *
     * Service-economy certification deliberately exercises a broad natural
     * corpus. A promotion authored from that traffic is valid production
     * movement, but a later Hugging Face checkpoint cannot prove its
     * destination math when the authenticated prompt never selects the expert.
     * The movement-only driver subsequently replays the exact reference
     * prefill. Require its demand to author at least one promotion before the
     * proof boundary settles; this joins workload causality without forcing a
     * route or consulting optional telemetry.
     *
     * @param target Complete typed convergence objective.
     * @param origin Durable ledger frontier before convergence traffic.
     * @param ledger Complete movement ledger from the production authority.
     * @param authenticated_routes Per-layer reference prefill route counts.
     * @return Awaiting, satisfied, or malformed authoritative evidence.
     */
    AuthenticatedPromotionConvergenceState
    classifyAuthenticatedPromotionConvergence(
        const DynamicResidencyConvergenceTarget &target,
        const DynamicResidencyConvergenceOrigin &origin,
        const MoEOptimizationMovementLedger &ledger,
        const std::vector<std::vector<std::uint64_t>>
            &authenticated_routes) noexcept
    {
        if (target.minimum_authenticated_promotions == 0u ||
            !ledger.complete() || ledger.edges.size() < origin.ledger_edges ||
            authenticated_routes.empty())
        {
            return AuthenticatedPromotionConvergenceState::
                InvalidAuthorityEvidence;
        }

        std::uint64_t authenticated_promotions = 0u;
        for (std::size_t index = origin.ledger_edges;
             index < ledger.edges.size(); ++index)
        {
            const auto &edge = ledger.edges[index];
            if (!edge.valid())
            {
                return AuthenticatedPromotionConvergenceState::
                    InvalidAuthorityEvidence;
            }
            if (edge.direction !=
                MoEOptimizationMovementDirection::Promotion)
            {
                continue;
            }
            if (edge.layer < 0 ||
                static_cast<std::size_t>(edge.layer) >=
                    authenticated_routes.size())
            {
                /* Sidecar-only movement is valid, but this prefill-specific
                 * proof deliberately waits for a main-model promotion. */
                continue;
            }
            const auto &layer_routes = authenticated_routes[
                static_cast<std::size_t>(edge.layer)];
            if (edge.expert < 0 ||
                static_cast<std::size_t>(edge.expert) >=
                    layer_routes.size())
            {
                return AuthenticatedPromotionConvergenceState::
                    InvalidAuthorityEvidence;
            }
            if (layer_routes[static_cast<std::size_t>(edge.expert)] == 0u)
                continue;
            ++authenticated_promotions;
            if (authenticated_promotions >=
                target.minimum_authenticated_promotions)
            {
                return AuthenticatedPromotionConvergenceState::Satisfied;
            }
        }
        return AuthenticatedPromotionConvergenceState::AwaitingPromotion;
    }

    /**
     * @brief Join topology movement and reference-workload movement evidence.
     *
     * The production authority remains the sole author of both inputs. The
     * reference histogram only decides whether a completed promotion is
     * mathematically witnessable by this test; it never changes placement.
     */
    DynamicResidencyConvergenceState requireAuthenticatedPromotion(
        DynamicResidencyConvergenceState topology_convergence,
        AuthenticatedPromotionConvergenceState promotion_convergence) noexcept
    {
        if (topology_convergence ==
                DynamicResidencyConvergenceState::
                    InvalidAuthorityEvidence ||
            promotion_convergence ==
                AuthenticatedPromotionConvergenceState::
                    InvalidAuthorityEvidence)
        {
            return DynamicResidencyConvergenceState::
                InvalidAuthorityEvidence;
        }
        if (topology_convergence !=
            DynamicResidencyConvergenceState::Satisfied)
        {
            return topology_convergence;
        }
        return promotion_convergence ==
                       AuthenticatedPromotionConvergenceState::Satisfied
                   ? DynamicResidencyConvergenceState::Satisfied
                   : DynamicResidencyConvergenceState::
                         AwaitingAuthenticatedPromotion;
    }

    /**
     * @brief Validate exact-epoch selection without inventing router demand.
     *
     * Residency is a capacity fact, not a promise that one bounded prompt
     * selects an owned expert. The pinned checkpoint must never select a final
     * idle participant. A remote participant that owns final experts must have
     * completed real device-owned sparse traffic somewhere in the production
     * workload, but that traffic may legitimately precede the final parity
     * prompt when Dynamic movement changes ownership between requests.
     *
     * @param residency Final request-pinned placement-bank residency.
     * @param pinned_route_count Routes selected by the exact parity checkpoint.
     * @param remote Whether endpoint-owned completion evidence is required.
     * @param completed_route_count Cumulative authenticated sparse completions.
     * @return Typed validity or the exact violated implication.
     */
    PublishedParticipantRouteEvidence validateParticipantRouteEvidence(
        PublishedParticipantResidency residency,
        std::uint64_t pinned_route_count,
        bool remote,
        std::uint64_t completed_route_count) noexcept
    {
        if (residency == PublishedParticipantResidency::Idle &&
            pinned_route_count != 0u)
        {
            return PublishedParticipantRouteEvidence::IdleParticipantSelected;
        }
        if (residency == PublishedParticipantResidency::OwnsExpert && remote &&
            completed_route_count == 0u)
        {
            return PublishedParticipantRouteEvidence::
                RemoteResidentNeverCompleted;
        }
        return PublishedParticipantRouteEvidence::Valid;
    }

    /**
     * @brief Fold one device-owned expert-to-participant bank into residency.
     *
     * Automatic capacity may validly exhaust the model before reaching a
     * configured lower-priority tier. Such endpoints remain part of topology
     * but must be proved idle instead of being asked to manufacture sparse
     * traffic. The published placement bank, rather than observed PerfStats,
     * is the authority that distinguishes those states.
     *
     * @param states Complete participant-state vector updated in place.
     * @param expert_owners Integral participant IDs for every expert in a layer.
     * @throws std::invalid_argument for a non-finite, fractional, or unknown ID.
     */
    void includePublishedExpertOwners(
        std::vector<PublishedParticipantResidency> &states,
        std::span<const float> expert_owners)
    {
        if (states.empty())
        {
            throw std::invalid_argument(
                "Published ExpertOverlay residency requires participants");
        }
        for (const float encoded_owner : expert_owners)
        {
            if (!std::isfinite(encoded_owner) || encoded_owner < 0.0f ||
                encoded_owner >
                    static_cast<float>(std::numeric_limits<int>::max()))
            {
                throw std::invalid_argument(
                    "Published ExpertOverlay bank contains an invalid participant ID");
            }
            const int owner = static_cast<int>(encoded_owner);
            if (encoded_owner != static_cast<float>(owner) || owner < 0 ||
                static_cast<std::size_t>(owner) >= states.size())
            {
                throw std::invalid_argument(
                    "Published ExpertOverlay bank names an unknown participant");
            }
            states[static_cast<std::size_t>(owner)] =
                PublishedParticipantResidency::OwnsExpert;
        }
    }

    /**
     * @brief Classify active CPU and non-continuation GPU sparse tiers.
     *
     * Participant IDs are allocated in routed-tier order by
     * `MoEExpertOwnerMap`; this helper walks the same typed plan order and
     * rejects a cardinality mismatch. Tier names and priority values remain
     * diagnostic only.
     *
     * @param plan Frozen capacity-resolved overlay topology.
     * @param states Published participant residency indexed by participant ID.
     * @return Active device-kind summary for transport evidence.
     * @throws std::invalid_argument for incomplete topology or state geometry.
     */
    PublishedSparseTierParticipation summarizePublishedParticipation(
        const MoERoutedExpertPlacementPlan &plan,
        std::span<const PublishedParticipantResidency> states)
    {
        PublishedSparseTierParticipation summary;
        std::size_t participant_id = 0u;
        for (const auto &tier : plan.routed_tiers)
        {
            const auto domain = std::find_if(
                plan.domains.begin(),
                plan.domains.end(),
                [&](const RoutedExpertDomain &candidate)
                { return candidate.name == tier.domain; });
            if (domain == plan.domains.end())
            {
                throw std::invalid_argument(
                    "Published participant summary cannot resolve domain '" +
                    tier.domain + "'");
            }
            for (const auto &participant : domain->participants)
            {
                if (participant_id >= states.size())
                {
                    throw std::invalid_argument(
                        "Published participant summary has fewer states than endpoints");
                }
                if (states[participant_id] ==
                    PublishedParticipantResidency::OwnsExpert)
                {
                    summary.sparse_follower =
                        summary.sparse_follower ||
                        domain->name != plan.continuation_domain;
                    summary.cpu = summary.cpu || participant.isCPU();
                    summary.secondary_gpu =
                        summary.secondary_gpu ||
                        (domain->name != plan.continuation_domain &&
                         participant.isGPU());
                }
                ++participant_id;
            }
        }
        if (participant_id != states.size())
        {
            throw std::invalid_argument(
                "Published participant summary has more states than endpoints");
        }
        return summary;
    }

    /**
     * @brief Derive the physically expressible Dynamic movement contract.
     *
     * Tier names, integer values, and accelerator vendors are deliberately
     * ignored. Rebalancing is local to one routed tier/domain, so distinct
     * domains carrying the same integer priority are not incorrectly treated
     * as one exchange set. The plan must be the capacity-resolved frozen
     * production plan: an automatic-capacity blueprint has no concrete expert
     * membership from which an expressible movement can be inferred.
     *
     * @param plan Inventory-bound or rank-agnostic ExpertOverlay plan.
     * @return Exact movement-axis contract implied by its participant catalogue.
     * @throws std::invalid_argument when a tier references no declared domain.
     */
    DynamicMovementAxisContract dynamicMovementAxisContract(
        const MoERoutedExpertPlacementPlan &plan)
    {
        for (std::size_t tier_index = 0;
             tier_index < plan.routed_tiers.size();
             ++tier_index)
        {
            const auto &tier = plan.routed_tiers[tier_index];
            const auto domain = std::find_if(
                plan.domains.begin(),
                plan.domains.end(),
                [&](const RoutedExpertDomain &candidate)
                { return candidate.name == tier.domain; });
            if (domain == plan.domains.end())
            {
                throw std::invalid_argument(
                    "Dynamic movement-axis resolution cannot find routed domain '" +
                    tier.domain + "'");
            }
            const std::size_t participant_count =
                domain->participants.size();
            if (domain->routed_compute_policy !=
                    RoutedExpertComputePolicy::Apportioned ||
                participant_count < 2u)
            {
                continue;
            }

            const bool has_exchange_degree_of_freedom = std::any_of(
                plan.placements.begin(),
                plan.placements.end(),
                [&](const RoutedExpertLayerPlacement &placement)
                {
                    return static_cast<std::size_t>(std::count(
                               placement.routed_expert_tier.begin(),
                               placement.routed_expert_tier.end(),
                               static_cast<int>(tier_index))) >
                           participant_count;
                });
            if (has_exchange_degree_of_freedom)
            {
                return DynamicMovementAxisContract::
                    PriorityMigrationAndParticipantBalance;
            }
        }
        return DynamicMovementAxisContract::PriorityMigrationOnly;
    }

    /**
     * @brief Classify whether submitted demand is visible to the authority.
     *
     * Occupancy is monotonic within one generation.  Rotation to a newer
     * generation also proves that the submitted work was consumed.  Neither
     * an unchanged snapshot nor unrelated activity authorizes a duplicate
     * admission.
     *
     * @param admission Bank generation and occupancy observed before submit.
     * @param status Current passive status from the optimization authority.
     * @return Pending, published, or malformed/regressed evidence.
     */
    DemandWindowAdmissionObservation observeDemandWindowAdmission(
        const SubmittedDemandWindowAdmission &admission,
        const MoEOptimizationStatus &status) noexcept
    {
        if (!status.demand_window.valid() ||
            status.demand_window.collected_routed_rows >
                status.demand_window.capacity_routed_rows ||
            status.demand_window.generation < admission.generation)
        {
            return DemandWindowAdmissionObservation::
                InvalidAuthorityEvidence;
        }
        if (status.demand_window.generation > admission.generation)
            return DemandWindowAdmissionObservation::Published;
        if (status.demand_window.collected_routed_rows <
            admission.observed_routed_rows)
        {
            return DemandWindowAdmissionObservation::
                InvalidAuthorityEvidence;
        }
        return status.demand_window.collected_routed_rows >
                       admission.observed_routed_rows
                   ? DemandWindowAdmissionObservation::Published
                   : DemandWindowAdmissionObservation::AwaitingPublication;
    }

    /**
     * @brief Classify whether a complete evidence cohort can begin safely.
     *
     * The observer never discards demand or pauses maintenance. If a
     * reconciled partial bank lacks timing headroom, the returned closure is
     * the exact number of ordinary routed rows required to finish that bank.
     * The caller can then admit one cache-distinct production prefill, wake the
     * authority once, and stop admission while the resulting wave rotates to
     * an empty successor bank.
     *
     * @param status Passive snapshot from the sole optimization authority.
     * @param purpose Evidence that follows this boundary.
     * @return Typed readiness, closure work, or malformed-evidence verdict.
     */
    ConvergenceBoundaryDecision classifyConvergenceBoundary(
        const MoEOptimizationStatus &status,
        ConvergenceBoundaryPurpose purpose) noexcept
    {
        if (!status.quiescentBetweenWaves())
        {
            return {
                .state = ConvergenceBoundaryState::AwaitingQuiescence,
            };
        }

        const auto required_rows =
            convergenceBoundaryProtectedRows(purpose);
        if (!required_rows)
        {
            return {
                .state = ConvergenceBoundaryState::Ready,
            };
        }
        if (!status.demand_window.valid())
        {
            return {
                .state =
                    ConvergenceBoundaryState::InvalidAuthorityEvidence,
            };
        }
        if (status.canBeginExclusiveCohort(*required_rows))
        {
            return {
                .state = ConvergenceBoundaryState::Ready,
            };
        }

        const std::uint64_t rows_to_close =
            status.demand_window.remainingRoutedRows();
        if (rows_to_close == 0u)
        {
            /* A completed bank can be visible just before the worker consumes
             * its wake. It is not safe to admit more demand, but neither is it
             * malformed; ordinary progress must rotate it. */
            return {
                .state = ConvergenceBoundaryState::AwaitingQuiescence,
            };
        }
        return {
            .state =
                ConvergenceBoundaryState::NeedsDemandWindowClosure,
            .closure = DemandWindowClosure{
                .generation = status.demand_window.generation,
                .observed_routed_rows =
                    status.demand_window.collected_routed_rows,
                .routed_rows = rows_to_close,
            },
        };
    }

    /**
     * @brief Classify one complete Dynamic convergence settlement transition.
     *
     * During the ordinary traffic horizon a partial demand bank remains normal
     * production state: later certified requests should fill it. Once that
     * finite horizon is exhausted, a reconciled host bank cannot make further
     * publication progress by itself. An empty successor requests one
     * authenticated seed; a partial bank returns its exact remaining row count
     * as cache-distinct closure traffic. These transitions are deliberately
     * distinct from post-target closure used to prepare an immutable timing
     * cohort.
     *
     * @param convergence Authoritative movement-target classification.
     * @param status Passive status from the sole optimization authority.
     * @param purpose Evidence that follows successful settlement.
     * @param horizon Whether ordinary certified traffic remains admissible.
     * @return One explicit lifecycle transition and any exact closure work.
     */
    DynamicConvergenceSettlementDecision classifyDynamicConvergenceSettlement(
        DynamicResidencyConvergenceState convergence,
        const MoEOptimizationStatus &status,
        ConvergenceBoundaryPurpose purpose,
        ConvergenceTrafficHorizon horizon) noexcept
    {
        if (convergence ==
            DynamicResidencyConvergenceState::InvalidAuthorityEvidence)
        {
            return {
                .state = DynamicConvergenceSettlementState::
                    InvalidAuthorityEvidence,
            };
        }

        if (convergence != DynamicResidencyConvergenceState::Satisfied)
        {
            /* Physical retirement already has an admitted publication to
             * complete. More demand cannot advance that event edge. */
            if (convergence == DynamicResidencyConvergenceState::
                                   AwaitingPhysicalCompletion ||
                horizon == ConvergenceTrafficHorizon::Open ||
                !status.quiescentBetweenWaves() ||
                status.authority != MoEOptimizationAuthority::Host)
            {
                return {
                    .state = DynamicConvergenceSettlementState::
                        AwaitingMovement,
                };
            }
            if (!status.demand_window.valid() ||
                status.demand_window.collected_routed_rows >
                    status.demand_window.capacity_routed_rows)
            {
                return {
                    .state = DynamicConvergenceSettlementState::
                        InvalidAuthorityEvidence,
                };
            }

            const std::uint64_t rows_to_close =
                status.demand_window.remainingRoutedRows();
            if (status.demand_window.collected_routed_rows == 0u)
            {
                return {
                    .state = DynamicConvergenceSettlementState::
                        NeedsMovementDemandWindowSeed,
                    .seed = DemandWindowSeed{
                        .generation = status.demand_window.generation,
                    },
                };
            }
            if (rows_to_close == 0u)
            {
                return {
                    .state = DynamicConvergenceSettlementState::
                        AwaitingMovement,
                };
            }
            return {
                .state = DynamicConvergenceSettlementState::
                    NeedsMovementDemandWindowClosure,
                .closure = DemandWindowClosure{
                    .generation = status.demand_window.generation,
                    .observed_routed_rows =
                        status.demand_window.collected_routed_rows,
                    .routed_rows = rows_to_close,
                },
            };
        }

        const ConvergenceBoundaryDecision boundary =
            classifyConvergenceBoundary(status, purpose);
        switch (boundary.state)
        {
        case ConvergenceBoundaryState::AwaitingQuiescence:
            return {
                .state = DynamicConvergenceSettlementState::
                    AwaitingBoundaryQuiescence,
            };
        case ConvergenceBoundaryState::Ready:
            return {
                .state = DynamicConvergenceSettlementState::Ready,
            };
        case ConvergenceBoundaryState::NeedsDemandWindowClosure:
            return {
                .state = DynamicConvergenceSettlementState::
                    NeedsBoundaryDemandWindowClosure,
                .closure = boundary.closure,
            };
        case ConvergenceBoundaryState::InvalidAuthorityEvidence:
            return {
                .state = DynamicConvergenceSettlementState::
                    InvalidAuthorityEvidence,
            };
        }
        return {
            .state = DynamicConvergenceSettlementState::
                InvalidAuthorityEvidence,
        };
    }

    /**
     * @brief Calculate requests that guarantee a typed publication target.
     *
     * Decode is deliberately excluded because EOS may end it before one routed
     * forward.  Each publication opens a fresh demand bank, so ceiling is
     * applied per bank before multiplying by the required publication count.
     *
     * @param window_rows Routed rows required to close one demand bank.
     * @param guaranteed_rows Cache-distinct real prefill rows per request.
     * @param required_publications Durable placement publications required.
     * @return Exact conservative request budget before async overlap traffic.
     * @throws std::invalid_argument for an empty geometry or target.
     * @throws std::overflow_error when the request count cannot fit in `int`.
     */
    int movementProofHistogramRequestBudget(
        int window_rows,
        int guaranteed_rows,
        std::uint64_t required_publications)
    {
        if (window_rows <= 0 || guaranteed_rows <= 0 ||
            required_publications == 0u)
        {
            throw std::invalid_argument(
                "Movement-proof histogram budget requires positive window, prompt, and publication geometry");
        }
        const std::uint64_t rows =
            static_cast<std::uint64_t>(window_rows);
        const std::uint64_t guaranteed =
            static_cast<std::uint64_t>(guaranteed_rows);
        const std::uint64_t requests_per_publication =
            (rows + guaranteed - 1u) / guaranteed;
        if (required_publications >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max()) /
                requests_per_publication)
        {
            throw std::overflow_error(
                "Movement-proof histogram request budget exceeds int range");
        }
        return static_cast<int>(
            required_publications * requests_per_publication);
    }

}
