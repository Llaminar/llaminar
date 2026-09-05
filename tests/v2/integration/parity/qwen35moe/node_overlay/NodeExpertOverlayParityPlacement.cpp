/**
 * @file NodeExpertOverlayParityPlacement.cpp
 * @brief Placement implementation of the shared node-overlay parity fixture.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#include "NodeExpertOverlayParityFixture.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{

    /** @return Whether @p candidate strictly improves the layout objective. */
    auto Qwen35MoENodeExpertOverlayParityTest::improvesAdversarialCpuPlacement(
        const AdversarialCpuPlacementScore &candidate,
        const AdversarialCpuPlacementScore &incumbent) noexcept -> bool
    {
        if (!candidate.reachable)
            return false;
        if (!incumbent.reachable)
            return true;
        if (candidate.remote_routes != incumbent.remote_routes)
            return candidate.remote_routes > incumbent.remote_routes;
        return candidate.local_routes < incumbent.local_routes;
    }

    /**
     * @brief Select exact lower-tier membership under random owner partitioning.
     *
     * Whole-expert ownership first sorts the selected expert IDs, applies the
     * production ordinal/random permutation to those positions, and gives one
     * balanced span to each participant. Selecting a different set therefore
     * changes the sorted-rank occupied by every later expert. A small dynamic
     * program solves that subsequence problem exactly instead of assuming an
     * expert ID maps directly to a participant.
     *
     * @param routes Authenticated prefill route count for every expert.
     * @param layer Model layer used by the production random permutation.
     * @param tier_index Lower-priority CPU tier index used by that permutation.
     * @param selected_count Exact tier quota for this layer.
     * @param participant_count Number of balanced NodeTP CPU owners.
     * @return Boolean expert mask with exactly @p selected_count entries.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::selectAdversarialCpuExperts(
        const std::vector<std::uint64_t> &routes,
        int layer,
        int tier_index,
        int selected_count,
        int participant_count,
        RoutedExpertOwnerOrder owner_order) -> std::vector<bool>
    {
        if (routes.empty() || layer < 0 || tier_index < 0 ||
            selected_count <= 0 ||
            selected_count > static_cast<int>(routes.size()) ||
            participant_count < 2)
        {
            throw std::invalid_argument(
                "Adversarial CPU owner selection has invalid geometry");
        }

        std::vector<int> selected_ranks(
            static_cast<std::size_t>(selected_count));
        std::iota(selected_ranks.begin(), selected_ranks.end(), 0);
        routed_expert_ownership::applyOwnerOrder(
            selected_ranks, owner_order, layer, tier_index);

        const int base = selected_count / participant_count;
        const int remainder = selected_count % participant_count;
        const int first_participant_count = base + (remainder > 0 ? 1 : 0);
        std::vector<bool> remote_selected_rank(
            static_cast<std::size_t>(selected_count), false);
        for (int offset = 0; offset < first_participant_count; ++offset)
        {
            remote_selected_rank.at(static_cast<std::size_t>(
                selected_ranks.at(static_cast<std::size_t>(offset)))) = true;
        }

        const std::size_t expert_count = routes.size();
        const std::size_t columns =
            static_cast<std::size_t>(selected_count) + 1u;
        std::vector<AdversarialCpuPlacementScore> scores(
            (expert_count + 1u) * columns);
        std::vector<std::uint8_t> took_expert(scores.size(), 0u);
        const auto index = [columns](std::size_t experts_seen,
                                     std::size_t experts_selected)
        {
            return experts_seen * columns + experts_selected;
        };
        scores[index(0u, 0u)].reachable = true;

        for (std::size_t experts_seen = 1u;
             experts_seen <= expert_count;
             ++experts_seen)
        {
            const std::size_t maximum_selected = std::min(
                experts_seen, static_cast<std::size_t>(selected_count));
            for (std::size_t experts_selected = 0u;
                 experts_selected <= maximum_selected;
                 ++experts_selected)
            {
                auto best = scores[index(
                    experts_seen - 1u, experts_selected)];
                bool take = false;
                if (experts_selected > 0u)
                {
                    auto candidate = scores[index(
                        experts_seen - 1u, experts_selected - 1u)];
                    if (candidate.reachable)
                    {
                        const std::uint64_t demand =
                            routes[experts_seen - 1u];
                        auto &total = remote_selected_rank[
                                          experts_selected - 1u]
                                          ? candidate.remote_routes
                                          : candidate.local_routes;
                        if (demand >
                            std::numeric_limits<std::uint64_t>::max() - total)
                        {
                            throw std::overflow_error(
                                "Adversarial CPU routing objective overflowed");
                        }
                        total += demand;
                        if (improvesAdversarialCpuPlacement(candidate, best))
                        {
                            best = candidate;
                            take = true;
                        }
                    }
                }
                scores[index(experts_seen, experts_selected)] = best;
                took_expert[index(experts_seen, experts_selected)] =
                    take ? 1u : 0u;
            }
        }

        if (!scores[index(expert_count,
                          static_cast<std::size_t>(selected_count))]
                 .reachable)
        {
            throw std::logic_error(
                "Adversarial CPU owner selection found no exact tier quota");
        }

        std::vector<bool> selected(expert_count, false);
        std::size_t experts_selected =
            static_cast<std::size_t>(selected_count);
        for (std::size_t experts_seen = expert_count;
             experts_seen > 0u;
             --experts_seen)
        {
            if (took_expert[index(experts_seen, experts_selected)] == 0u)
                continue;
            selected[experts_seen - 1u] = true;
            --experts_selected;
        }
        if (experts_selected != 0u ||
            static_cast<int>(std::count(
                selected.begin(), selected.end(), true)) != selected_count)
        {
            throw std::logic_error(
                "Adversarial CPU owner selection reconstruction changed its quota");
        }
        return selected;
    }

    /**
     * @brief Load exact Hugging Face prefill demand by model layer and expert.
     *
     * This histogram is immutable mathematical evidence. Runtime placement is
     * still authored exclusively by the production controller from its live
     * device histograms; the fixture uses these counts only to decide whether
     * a completed promotion can be re-exercised by the later parity prefill.
     *
     * @param layer_count Number of main-model or complete model layers to load.
     * @param expert_count Authenticated routed-expert cardinality.
     * @return Dense `[layer][expert]` route counts.
     * @throws std::exception for missing or malformed reference evidence.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::loadAuthenticatedPrefillRouteCounts(
        int layer_count,
        int expert_count) -> std::vector<std::vector<std::uint64_t>>
    {
        if (layer_count <= 0 || expert_count <= 0)
        {
            throw std::invalid_argument(
                "Authenticated route histogram requires positive model geometry");
        }
        std::vector<std::vector<std::uint64_t>> routes(
            static_cast<std::size_t>(layer_count),
            std::vector<std::uint64_t>(
                static_cast<std::size_t>(expert_count), 0u));
        for (int layer = 0; layer < layer_count; ++layer)
        {
            const auto routing = loadPyTorchSnapshot(
                "layer" + std::to_string(layer) +
                "_MOE_ROUTING_INDICES");
            if (routing.empty())
            {
                throw std::invalid_argument(
                    "Authenticated ExpertOverlay movement proof lacks reference routing for layer " +
                    std::to_string(layer));
            }
            for (const float routed_expert : routing)
            {
                const int expert = static_cast<int>(routed_expert);
                if (!std::isfinite(routed_expert) ||
                    routed_expert != static_cast<float>(expert) ||
                    expert < 0 || expert >= expert_count)
                {
                    throw std::invalid_argument(
                        "Authenticated ExpertOverlay movement proof found a malformed expert ID");
                }
                ++routes[static_cast<std::size_t>(layer)]
                        [static_cast<std::size_t>(expert)];
            }
        }
        return routes;
    }

    /**
     * @brief Install an authenticated workload-adversarial initial tier layout.
     *
     * The Hugging Face pack is already the mathematical oracle for this parity
     * cell. Its routing IDs define only the starting condition: high-demand
     * experts are deliberately left on lower-priority participants, with the
     * strongest CPU candidates owned by the rank remote from continuation.
     * Runtime movement remains driven exclusively by histograms emitted from
     * the real Llaminar sparse-collective graphs.
     *
     * @param requested Inventory-bound dynamic production request.
     * @return Same request with complete explicit per-layer initial placement.
     * @throws std::exception For missing/malformed reference evidence or a
     *         layout that cannot make remote CPU demand strictly dominant.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::installReferenceAdversarialPlacements(
        MoERoutedExpertPlacementPlan requested) -> MoERoutedExpertPlacementPlan
    {
        const auto metadata_path =
            std::filesystem::path(config_.snapshot_dir) / "metadata.txt";
        const auto layer_text =
            readSnapshotMetadataValue(metadata_path, "n_layers");
        int layer_count = 0;
        if (!layer_text)
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay placement requires reference n_layers metadata");
        }
        const char *const layer_begin = layer_text->data();
        const char *const layer_end = layer_begin + layer_text->size();
        const auto parsed_layers =
            std::from_chars(layer_begin, layer_end, layer_count);
        if (parsed_layers.ec != std::errc{} ||
            parsed_layers.ptr != layer_end || layer_count <= 0)
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay placement has invalid reference n_layers metadata");
        }
        auto cpu_tier = std::find_if(
            requested.routed_tiers.begin(),
            requested.routed_tiers.end(),
            [](const auto &tier)
            { return tier.domain == kCpuColdDomain; });
        if (cpu_tier == requested.routed_tiers.end())
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay placement has no NodeTP CPU tier");
        }
        const int cpu_tier_index = static_cast<int>(std::distance(
            requested.routed_tiers.begin(), cpu_tier));
        const auto cpu_domain = std::find_if(
            requested.domains.begin(),
            requested.domains.end(),
            [](const auto &domain)
            { return domain.name == kCpuColdDomain; });
        const auto continuation = std::find_if(
            requested.domains.begin(),
            requested.domains.end(),
            [&](const auto &domain)
            { return domain.name == requested.continuation_domain; });
        const auto continuation_rank =
            continuation == requested.domains.end()
                ? std::optional<int>{}
                : continuation->primaryWorldRank();
        if (cpu_domain == requested.domains.end() ||
            continuation == requested.domains.end() ||
            cpu_domain->participants.size() < 2u ||
            cpu_domain->world_ranks.size() !=
                cpu_domain->participants.size() ||
            !continuation_rank ||
            cpu_domain->world_ranks.front() == *continuation_rank)
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay placement requires remote-first multi-participant CPU ownership and a resolved continuation rank");
        }

        MoERoutedExpertModelMetadata metadata = topologyOnlyMetadata();
        metadata.num_layers = layer_count;
        const int expert_count = metadata.num_experts;
        auto reference_routes = loadAuthenticatedPrefillRouteCounts(
            layer_count, expert_count);

        /*
         * Do not guess the tier quota here. The production physical-capacity
         * resolver has not yet observed fixed graph, migration, and backend
         * allocations. Declare only a complete cold-to-hot expert order; after
         * exact quotas are installed, the normal planner fills smaller integer
         * priorities first and therefore leaves the authenticated hottest
         * experts in lower-priority tiers. Concrete membership remains wholly
         * production-owned.
         */
        requested.placements.clear();
        requested.initial_layer_order_overrides.clear();
        requested.initial_layer_order_overrides.reserve(
            static_cast<std::size_t>(layer_count));
        for (int layer = 0; layer < layer_count; ++layer)
        {
            std::vector<int> order(static_cast<std::size_t>(expert_count));
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(
                order.begin(), order.end(),
                [&](int lhs, int rhs)
                {
                    const auto lhs_routes = reference_routes[
                        static_cast<std::size_t>(layer)]
                        [static_cast<std::size_t>(lhs)];
                    const auto rhs_routes = reference_routes[
                        static_cast<std::size_t>(layer)]
                        [static_cast<std::size_t>(rhs)];
                    if (lhs_routes != rhs_routes)
                        return lhs_routes < rhs_routes;
                    return lhs < rhs;
                });
            requested.initial_layer_order_overrides.push_back({
                .layer = layer,
                .expert_ids = std::move(order),
            });
        }

        reference_adversarial_routes_ = std::move(reference_routes);
        reference_adversarial_cpu_tier_index_ = cpu_tier_index;
        reference_adversarial_continuation_rank_ = *continuation_rank;
        LOG_INFO(
            "[Qwen3.5 MoE GraphNative] Authenticated adversarial order deferred "
            "to production capacity: layers=" << layer_count
            << " experts=" << expert_count
            << " owner_order="
            << routedExpertOwnerOrderToString(requested.owner_order));
        return requested;
    }

    /**
     * @brief Certify the exact capacity-resolved adversarial epoch-one layout.
     *
     * The setup request carries only a complete expert permutation. This check
     * runs after the production runner has resolved physical quotas and frozen
     * concrete placements. It proves that lower-priority CPU membership owns
     * the hottest suffix and that the first, remote CPU participant received a
     * genuinely hotter expert than its colocated peer on at least one layer.
     *
     * @return True when the frozen production owner map satisfies the complete
     *         adversarial contract, or when this cell declared no such order.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::certifyInstalledReferenceAdversarialPlacement() const -> bool
    {
        if (reference_adversarial_routes_.empty())
            return true;
        if (!orch_runner_ || !orch_runner_->config().moe_routed_expert_plan)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Deferred adversarial order has no frozen production plan");
            return false;
        }

        const auto &plan = *orch_runner_->config().moe_routed_expert_plan;
        if (!plan.initial_layer_order_overrides.empty() ||
            plan.placements.size() < reference_adversarial_routes_.size())
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Production did not consume the complete deferred adversarial order: initial_orders="
                << plan.initial_layer_order_overrides.size()
                << " placements=" << plan.placements.size()
                << " expected_layers="
                << reference_adversarial_routes_.size());
            return false;
        }
        if (reference_adversarial_cpu_tier_index_ < 0 ||
            reference_adversarial_cpu_tier_index_ >=
                static_cast<int>(plan.routed_tiers.size()))
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Deferred adversarial CPU tier identity is outside the frozen plan");
            return false;
        }

        const auto owner_map = MoEExpertOwnerMap::build(plan);
        std::uint64_t remote_routes = 0u;
        std::uint64_t local_routes = 0u;
        std::uint64_t strongest_remote = 0u;
        std::uint64_t strongest_local = 0u;
        std::uint64_t strongest_cpu = 0u;
        std::uint64_t strongest_preferred = 0u;
        std::int64_t best_layer_margin =
            std::numeric_limits<std::int64_t>::min();
        bool every_layer_has_both_tier_sides = true;
        bool cold_to_hot_order_preserved = true;

        for (std::size_t layer = 0;
             layer < reference_adversarial_routes_.size(); ++layer)
        {
            const auto &routes = reference_adversarial_routes_[layer];
            std::uint64_t layer_remote_peak = 0u;
            std::uint64_t layer_local_peak = 0u;
            std::uint64_t layer_cpu_min =
                std::numeric_limits<std::uint64_t>::max();
            std::uint64_t layer_cpu_max = 0u;
            std::uint64_t layer_preferred_max = 0u;
            std::size_t cpu_experts = 0u;
            std::size_t preferred_experts = 0u;

            for (std::size_t expert = 0; expert < routes.size(); ++expert)
            {
                const auto *owner = owner_map.ownerFor(
                    static_cast<int>(layer), static_cast<int>(expert));
                if (!owner)
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Frozen adversarial owner map is incomplete at layer="
                        << layer << " expert=" << expert);
                    return false;
                }
                const std::uint64_t demand = routes[expert];
                if (owner->tier_idx ==
                    reference_adversarial_cpu_tier_index_)
                {
                    ++cpu_experts;
                    layer_cpu_min = std::min(layer_cpu_min, demand);
                    layer_cpu_max = std::max(layer_cpu_max, demand);
                    if (owner->owner_world_rank !=
                        reference_adversarial_continuation_rank_)
                    {
                        remote_routes += demand;
                        layer_remote_peak = std::max(
                            layer_remote_peak, demand);
                    }
                    else
                    {
                        local_routes += demand;
                        layer_local_peak = std::max(
                            layer_local_peak, demand);
                    }
                }
                else
                {
                    ++preferred_experts;
                    layer_preferred_max = std::max(
                        layer_preferred_max, demand);
                }
            }

            every_layer_has_both_tier_sides =
                every_layer_has_both_tier_sides &&
                cpu_experts > 0u && preferred_experts > 0u;
            if (cpu_experts > 0u && preferred_experts > 0u)
            {
                cold_to_hot_order_preserved =
                    cold_to_hot_order_preserved &&
                    layer_cpu_min >= layer_preferred_max;
            }
            strongest_cpu = std::max(strongest_cpu, layer_cpu_max);
            strongest_preferred = std::max(
                strongest_preferred, layer_preferred_max);
            strongest_remote = std::max(
                strongest_remote, layer_remote_peak);
            strongest_local = std::max(
                strongest_local, layer_local_peak);
            best_layer_margin = std::max(
                best_layer_margin,
                static_cast<std::int64_t>(layer_remote_peak) -
                    static_cast<std::int64_t>(layer_local_peak));
        }

        if (!every_layer_has_both_tier_sides ||
            !cold_to_hot_order_preserved || remote_routes == 0u ||
            best_layer_margin <= 0)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Capacity-resolved adversarial layout failed certification: both_tier_sides="
                << every_layer_has_both_tier_sides
                << " cold_to_hot_order=" << cold_to_hot_order_preserved
                << " remote_routes=" << remote_routes
                << " best_layer_margin=" << best_layer_margin);
            return false;
        }

        LOG_INFO(
            "[Qwen3.5 MoE GraphNative] Capacity-resolved adversarial layout certified: layers="
            << reference_adversarial_routes_.size()
            << " remote_cpu_routes=" << remote_routes
            << " local_cpu_routes=" << local_routes
            << " strongest_remote_expert=" << strongest_remote
            << " strongest_local_expert=" << strongest_local
            << " strongest_cpu_expert=" << strongest_cpu
            << " strongest_preferred_expert=" << strongest_preferred
            << " best_layer_margin=" << best_layer_margin);
        return true;
    }

}
