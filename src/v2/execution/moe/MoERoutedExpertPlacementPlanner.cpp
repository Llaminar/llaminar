/**
 * @file MoERoutedExpertPlacementPlanner.cpp
 * @brief Deterministic implementation of routed-expert placement objectives.
 *
 * Static policies fill setup-resolved quotas in strict integer-priority order.
 * Dynamic policies may consume exact decode, prefill, and grouped-verifier
 * demand together with a complete service profile. That path solves a fixed-
 * quota bipartite min-cost flow using integer arithmetic, so every distributed
 * participant derives the same placement without floating-point or declaration-
 * order ambiguity.
 */

#include "execution/moe/MoERoutedExpertPlacementPlanner.h"

#include "execution/moe/DecodeExpertHistogram.h"
#include "planning/WeightMemoryEstimator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace llaminar2
{
    namespace
    {

        size_t ceilBytes(long double bytes)
        {
            if (bytes <= 0.0L)
                return 0;
            return static_cast<size_t>(std::ceil(bytes));
        }

        void validateMetadata(const MoERoutedExpertModelMetadata &metadata)
        {
            if (metadata.num_layers <= 0)
                throw std::invalid_argument("MoE expert planner metadata num_layers must be > 0");
            if (metadata.num_experts <= 0)
                throw std::invalid_argument("MoE expert planner metadata num_experts must be > 0");
            if (metadata.d_model <= 0)
                throw std::invalid_argument("MoE expert planner metadata d_model must be > 0");
            if (metadata.routed_intermediate_size <= 0)
                throw std::invalid_argument("MoE expert planner metadata routed_intermediate_size must be > 0");
            if (metadata.has_shared_expert && metadata.shared_intermediate_size <= 0)
                throw std::invalid_argument("MoE expert planner metadata shared_intermediate_size must be > 0 when has_shared_expert is true");
        }

        std::string formatValidationErrors(const MoERoutedExpertPlacementValidationResult &validation)
        {
            std::ostringstream message;
            message << "Invalid routed-expert placement plan for planner:";
            for (const auto &error : validation.errors)
            {
                message << "\n - " << error;
            }
            return message.str();
        }

        int findFallbackTierIndex(const MoERoutedExpertPlacementPlan &plan)
        {
            for (size_t tier_idx = 0; tier_idx < plan.routed_tiers.size(); ++tier_idx)
            {
                if (plan.routed_tiers[tier_idx].fallback)
                    return static_cast<int>(tier_idx);
            }
            return -1;
        }

        std::vector<int> sortedPlanningTierIndices(
            const MoERoutedExpertPlacementPlan &plan,
            int fallback_tier)
        {
            std::vector<int> tier_indices;
            tier_indices.reserve(plan.routed_tiers.size());
            for (size_t tier_idx = 0; tier_idx < plan.routed_tiers.size(); ++tier_idx)
            {
                if (static_cast<int>(tier_idx) != fallback_tier)
                    tier_indices.push_back(static_cast<int>(tier_idx));
            }

            std::sort(tier_indices.begin(), tier_indices.end(), [&](int lhs, int rhs)
                      {
            const auto &left = plan.routed_tiers[lhs];
            const auto &right = plan.routed_tiers[rhs];
            return left.priority < right.priority; });
            return tier_indices;
        }

        std::vector<int> byIdExpertOrder(int num_experts)
        {
            std::vector<int> expert_order(static_cast<size_t>(num_experts));
            std::iota(expert_order.begin(), expert_order.end(), 0);
            return expert_order;
        }

        /**
         * @brief Resolve the setup-only fill order for one model layer.
         *
         * Model-aware validation has already proven that each override is an
         * exact permutation. A layer without an override deliberately uses the
         * deterministic expert-id order; this is how trailing MTP source layers
         * discovered from the GGUF coexist with main-layer warm-start evidence.
         */
        std::vector<int> initialExpertOrder(
            const MoERoutedExpertPlacementPlan &plan,
            int layer,
            int num_experts)
        {
            if (plan.initial_layer_order_overrides.empty())
                return byIdExpertOrder(num_experts);

            const auto found = std::find_if(
                plan.initial_layer_order_overrides.begin(),
                plan.initial_layer_order_overrides.end(),
                [layer](const RoutedExpertInitialLayerOrder &order)
                { return order.layer == layer; });
            if (found == plan.initial_layer_order_overrides.end())
                return byIdExpertOrder(num_experts);
            if (static_cast<int>(found->expert_ids.size()) != num_experts)
            {
                throw std::invalid_argument(
                    "Deferred initial expert order has invalid expert count at layer " +
                    std::to_string(layer));
            }
            return found->expert_ids;
        }

        /** @brief Uniform read-only view over live or frozen histogram evidence. */
        struct HistogramEvidence
        {
            const DecodeExpertHistogram *live = nullptr;
            std::optional<ValidatedDecodeExpertHistogramWindowView> frozen;

            explicit operator bool() const noexcept
            {
                return live != nullptr || frozen.has_value();
            }

            int numLayers() const noexcept
            {
                return frozen ? frozen->numLayers()
                              : (live ? live->config().num_layers : 0);
            }

            int numExperts() const noexcept
            {
                return frozen ? frozen->numExperts()
                              : (live ? live->config().num_experts : 0);
            }

            uint64_t activationCount(int layer, int expert) const
            {
                return frozen ? frozen->activationCount(layer, expert)
                              : live->activationCount(layer, expert);
            }

            uint64_t activationCount(
                ExpertHistogramSource source,
                int layer,
                int expert) const
            {
                return frozen
                           ? frozen->activationCount(source, layer, expert)
                           : live->activationCount(source, layer, expert);
            }
        };

        /** @brief Convert one retained production source to its profile column. */
        std::size_t servicePhaseIndex(ExpertHistogramSource source)
        {
            switch (source)
            {
            case ExpertHistogramSource::DecodeToken:
                return 0;
            case ExpertHistogramSource::PrefillChunk:
                return 1;
            case ExpertHistogramSource::GroupedVerifier:
                return 2;
            case ExpertHistogramSource::SyntheticTest:
                break;
            }
            throw std::invalid_argument(
                "Synthetic or unknown histogram sources have no service-profile column");
        }

        /** @brief Validated constant-time view over a complete service profile. */
        struct PhaseServiceProfileView
        {
            const MoERoutedTierServiceProfile *profile = nullptr;
            int tier_count = 0;
            int layer_count = 0;
            std::vector<const MoERoutedTierLayerPhaseServiceCost *> rows;

            explicit operator bool() const noexcept
            {
                return profile != nullptr;
            }

            /** @return Whether this exact retained layer may execute @p source. */
            bool reachable(int layer, ExpertHistogramSource source) const
            {
                return profile->production_topology.reachable(
                    layer, servicePhaseIndex(source));
            }

            /** @return Whether this exact layer/phase participates in economics. */
            bool priced(int layer, ExpertHistogramSource source) const
            {
                return profile->production_topology.requiresServiceEvidence(
                    layer, servicePhaseIndex(source));
            }

            /** @return Certified cost, rejecting an unpriced layer/phase. */
            uint64_t cost(
                int tier,
                int layer,
                ExpertHistogramSource source) const
            {
                if (!priced(layer, source))
                {
                    throw std::logic_error(
                        "MoE service cost requested for an economy-unpriced layer phase");
                }
                const auto offset =
                    static_cast<std::size_t>(tier) *
                        static_cast<std::size_t>(layer_count) +
                    static_cast<std::size_t>(layer);
                const auto phase = servicePhaseIndex(source);
                const auto cost =
                    rows.at(offset)->nanoseconds_per_activation[phase];
                if (!profile->production_topology.requiresServiceEvidence(
                        layer, phase) ||
                    cost == 0u)
                {
                    throw std::logic_error(
                        "MoE service cost is missing for an economy-priced layer phase");
                }
                return cost;
            }
        };

        /**
         * @brief Validate and index one complete measured service profile.
         *
         * Integer priority owns capacity fill and deterministic tie-breaking;
         * it is not a claim that one backend wins every phase and geometry.
         * Real CPU/GPU kernels can cross over for small sparse batches, so the
         * exact phase costs remain authoritative for the runtime objective.
         */
        PhaseServiceProfileView phaseServiceProfileView(
            const MoERoutedExpertPlacementPlan &plan,
            const MoERoutedExpertModelMetadata &metadata,
            const MoERoutedTierServiceProfile *profile)
        {
            if (!profile)
                return {};
            if (profile->identity.empty())
            {
                throw std::invalid_argument(
                    "MoE tier phase service profile requires a non-empty setup identity");
            }
            if (!profile->production_topology.valid() ||
                profile->production_topology.layerCount() !=
                    static_cast<std::size_t>(metadata.num_layers))
            {
                throw std::invalid_argument(
                    "MoE tier phase service profile has invalid layer reachability");
            }

            PhaseServiceProfileView view;
            view.profile = profile;
            view.tier_count = static_cast<int>(plan.routed_tiers.size());
            view.layer_count = metadata.num_layers;
            const auto expected_rows =
                static_cast<std::size_t>(view.tier_count) *
                static_cast<std::size_t>(view.layer_count);
            if (profile->costs.size() != expected_rows)
            {
                throw std::invalid_argument(
                    "MoE tier phase service profile must contain exactly one row per tier and layer");
            }
            view.rows.assign(expected_rows, nullptr);
            for (const auto &row : profile->costs)
            {
                if (row.tier_index < 0 || row.tier_index >= view.tier_count ||
                    row.layer < 0 || row.layer >= view.layer_count)
                {
                    throw std::invalid_argument(
                        "MoE tier phase service profile row lies outside plan geometry");
                }
                bool invalid_phase_cost = false;
                for (std::size_t phase = 0;
                     phase < kExpertHistogramProductionSourceCount;
                     ++phase)
                {
                    const uint64_t value =
                        row.nanoseconds_per_activation[phase];
                    invalid_phase_cost = invalid_phase_cost ||
                                         (profile->production_topology
                                                  .requiresServiceEvidence(
                                                      row.layer, phase)
                                              ? value == 0
                                              : value != 0);
                }
                if (invalid_phase_cost)
                {
                    throw std::invalid_argument(
                        "MoE tier phase service times must be positive exactly for economy-priced phases");
                }
                const auto offset =
                    static_cast<std::size_t>(row.tier_index) *
                        static_cast<std::size_t>(view.layer_count) +
                    static_cast<std::size_t>(row.layer);
                if (view.rows[offset] != nullptr)
                {
                    throw std::invalid_argument(
                        "MoE tier phase service profile repeats a tier/layer row");
                }
                view.rows[offset] = &row;
            }
            if (std::any_of(
                    view.rows.begin(),
                    view.rows.end(),
                    [](const auto *row)
                    { return row == nullptr; }))
            {
                throw std::invalid_argument(
                    "MoE tier phase service profile omitted a tier/layer row");
            }

            return view;
        }

        HistogramEvidence histogramEvidence(
            const MoERoutedExpertPlacementPlannerOptions &options)
        {
            if (options.decode_histogram && options.decode_histogram_window)
            {
                throw std::invalid_argument(
                    "MoE placement planner accepts either a live or frozen histogram, not both");
            }
            HistogramEvidence evidence{
                .live = options.decode_histogram,
            };
            if (options.decode_histogram_window)
            {
                /* Authenticate the distributed/frozen payload once. Every
                 * placement objective below can then read its immutable phase
                 * banks in O(1) without revalidating all model entries. */
                evidence.frozen.emplace(
                    options.decode_histogram_window->validatedView());
            }
            return evidence;
        }

        bool histogramLayerHasCounts(
            const HistogramEvidence &histogram,
            int layer,
            int num_experts)
        {
            for (int expert_id = 0; expert_id < num_experts; ++expert_id)
            {
                if (histogram.activationCount(layer, expert_id) != 0)
                    return true;
            }
            return false;
        }

        std::vector<int> histogramExpertOrder(
            const HistogramEvidence &histogram,
            int layer,
            int num_experts)
        {
            std::vector<std::pair<int, uint64_t>> expert_counts;
            expert_counts.reserve(static_cast<size_t>(num_experts));
            for (int expert_id = 0; expert_id < num_experts; ++expert_id)
            {
                expert_counts.emplace_back(expert_id, histogram.activationCount(layer, expert_id));
            }

            std::sort(expert_counts.begin(), expert_counts.end(), [](const auto &lhs, const auto &rhs)
                      {
            if (lhs.second != rhs.second)
                return lhs.second > rhs.second;
            return lhs.first < rhs.first; });

            std::vector<int> expert_order;
            expert_order.reserve(expert_counts.size());
            for (const auto &[expert_id, _] : expert_counts)
            {
                expert_order.push_back(expert_id);
            }
            return expert_order;
        }

        size_t tierCapacityPerLayer(
            const RoutedExpertTier &tier,
            int layer,
            int num_experts,
            size_t routed_expert_bytes_per_expert)
        {
            if (!tier.resolved_live_experts_per_layer.empty())
            {
                if (layer < 0 || static_cast<size_t>(layer) >=
                                     tier.resolved_live_experts_per_layer.size())
                {
                    throw std::invalid_argument(
                        "Resolved routed-tier quota does not cover requested layer");
                }
                return static_cast<size_t>(
                    tier.resolved_live_experts_per_layer[
                        static_cast<size_t>(layer)]);
            }
            size_t capacity = static_cast<size_t>(num_experts);
            if (tier.max_experts_per_layer > 0)
            {
                capacity = std::min(capacity, static_cast<size_t>(tier.max_experts_per_layer));
            }
            if (tier.memory_budget_bytes > 0)
            {
                const size_t memory_capacity = routed_expert_bytes_per_expert == 0
                                                   ? 0
                                                   : tier.memory_budget_bytes / routed_expert_bytes_per_expert;
                capacity = std::min(capacity, memory_capacity);
            }
            return capacity;
        }

        RoutedExpertLayerPlacement buildPlacementFromExpertOrder(
            const MoERoutedExpertPlacementPlan &plan,
            const MoERoutedExpertModelMetadata &metadata,
            int layer,
            const std::vector<int> &expert_order,
            size_t routed_expert_bytes_per_expert)
        {
            const int fallback_tier = findFallbackTierIndex(plan);
            RoutedExpertLayerPlacement placement;
            placement.layer = layer;
            placement.routed_expert_tier.assign(static_cast<size_t>(metadata.num_experts), fallback_tier);

            size_t next_expert = 0;
            const auto tier_indices = sortedPlanningTierIndices(plan, fallback_tier);
            for (const int tier_idx : tier_indices)
            {
                const auto &tier = plan.routed_tiers[tier_idx];
                const size_t capacity = tierCapacityPerLayer(
                    tier,
                    layer,
                    metadata.num_experts,
                    routed_expert_bytes_per_expert);
                size_t assigned = 0;
                while (assigned < capacity && next_expert < expert_order.size())
                {
                    const int expert_id = expert_order[next_expert++];
                    placement.routed_expert_tier[static_cast<size_t>(expert_id)] = tier_idx;
                    ++assigned;
                }
            }

            if (fallback_tier < 0 && next_expert < expert_order.size())
            {
                std::ostringstream message;
                message << "Routed-expert placement planner no-fallback tier capacity cannot cover every expert for layer "
                        << layer << ": assigned " << next_expert << " of " << expert_order.size()
                        << " experts; increase routed tier max_experts_per_layer or memory_budget_bytes, or configure one fallback tier";
                throw std::invalid_argument(message.str());
            }

            return placement;
        }

        /** @brief Lexicographic exact objective used by fixed-quota assignment. */
        struct ExactAssignmentCost
        {
            __int128 service_time_ns = 0;
            int64_t moved_experts = 0;
            int64_t expert_id_tie = 0;

            friend bool operator<(
                const ExactAssignmentCost &lhs,
                const ExactAssignmentCost &rhs) noexcept
            {
                return std::tie(
                           lhs.service_time_ns,
                           lhs.moved_experts,
                           lhs.expert_id_tie) <
                       std::tie(
                           rhs.service_time_ns,
                           rhs.moved_experts,
                           rhs.expert_id_tie);
            }

            friend bool operator==(
                const ExactAssignmentCost &lhs,
                const ExactAssignmentCost &rhs) noexcept = default;
        };

        /** @brief Checked component-wise addition for min-cost residual paths. */
        ExactAssignmentCost addExactCost(
            const ExactAssignmentCost &lhs,
            const ExactAssignmentCost &rhs)
        {
            ExactAssignmentCost result;
            if (__builtin_add_overflow(
                    lhs.service_time_ns,
                    rhs.service_time_ns,
                    &result.service_time_ns) ||
                __builtin_add_overflow(
                    lhs.moved_experts,
                    rhs.moved_experts,
                    &result.moved_experts) ||
                __builtin_add_overflow(
                    lhs.expert_id_tie,
                    rhs.expert_id_tie,
                    &result.expert_id_tie))
            {
                throw std::overflow_error(
                    "MoE exact tier-assignment objective overflowed");
            }
            return result;
        }

        /** @brief Checked additive inverse for one residual edge cost. */
        ExactAssignmentCost negateExactCost(
            const ExactAssignmentCost &value)
        {
            ExactAssignmentCost result;
            if (__builtin_sub_overflow(
                    static_cast<__int128>(0),
                    value.service_time_ns,
                    &result.service_time_ns) ||
                __builtin_sub_overflow(
                    static_cast<int64_t>(0),
                    value.moved_experts,
                    &result.moved_experts) ||
                __builtin_sub_overflow(
                    static_cast<int64_t>(0),
                    value.expert_id_tie,
                    &result.expert_id_tie))
            {
                throw std::overflow_error(
                    "MoE exact tier-assignment residual objective overflowed");
            }
            return result;
        }

        /** @brief Build one non-negative exact phase-weighted service cost. */
        __int128 phaseWeightedServiceTime(
            const HistogramEvidence &histogram,
            const PhaseServiceProfileView &profile,
            int tier,
            int layer,
            int expert)
        {
            constexpr std::array<ExpertHistogramSource,
                                 kExpertHistogramProductionSourceCount>
                sources{
                    ExpertHistogramSource::DecodeToken,
                    ExpertHistogramSource::PrefillChunk,
                    ExpertHistogramSource::GroupedVerifier,
                };
            constexpr unsigned __int128 kSigned128Max =
                (~static_cast<unsigned __int128>(0)) >> 1;
            unsigned __int128 total = 0;
            for (const auto source : sources)
            {
                const auto demand = histogram.activationCount(
                    source, layer, expert);
                if (!profile.reachable(layer, source))
                {
                    if (demand != 0)
                    {
                        throw std::logic_error(
                            "MoE histogram contains demand for a runtime-disabled inference phase");
                    }
                    continue;
                }
                if (!profile.priced(layer, source))
                {
                    /* Fixed positive-depth MTP can execute a bounded serial
                     * catch-up/tail graph. Its demand is valid lifecycle
                     * evidence, but it is deliberately excluded from the
                     * recurring steady-state placement objective. */
                    continue;
                }
                const auto service = profile.cost(
                    tier, layer, source);
                unsigned __int128 product = 0;
                if (__builtin_mul_overflow(
                        static_cast<unsigned __int128>(demand),
                        static_cast<unsigned __int128>(service),
                        &product) ||
                    product > kSigned128Max - total)
                {
                    throw std::overflow_error(
                        "MoE phase-weighted service objective overflowed signed 128-bit capacity");
                }
                total += product;
            }
            return static_cast<__int128>(total);
        }

        /** @brief Validate and index optional incumbent placements by layer. */
        std::vector<const RoutedExpertLayerPlacement *> previousByLayer(
            const MoERoutedExpertPlacementPlan &plan,
            const MoERoutedExpertModelMetadata &metadata,
            const MoERoutedTierRebalancerOptions &options)
        {
            std::vector<const RoutedExpertLayerPlacement *> result(
                static_cast<std::size_t>(metadata.num_layers), nullptr);
            if (options.previous_placements.empty())
                return result;

            for (const auto &placement : options.previous_placements)
            {
                if (placement.layer < 0 ||
                    placement.layer >= metadata.num_layers ||
                    placement.routed_expert_tier.size() !=
                        static_cast<std::size_t>(metadata.num_experts))
                {
                    throw std::invalid_argument(
                        "MoE previous placements do not match model geometry");
                }
                auto &slot = result.at(
                    static_cast<std::size_t>(placement.layer));
                if (slot != nullptr)
                {
                    throw std::invalid_argument(
                        "MoE previous placements repeat a model layer");
                }
                for (const int tier : placement.routed_expert_tier)
                {
                    if (tier < 0 ||
                        tier >= static_cast<int>(plan.routed_tiers.size()))
                    {
                        throw std::invalid_argument(
                            "MoE previous placement references an unknown tier");
                    }
                }
                slot = &placement;
            }
            if (std::any_of(
                    result.begin(),
                    result.end(),
                    [](const auto *placement)
                    { return placement == nullptr; }))
            {
                throw std::invalid_argument(
                    "MoE previous placements must cover every model layer");
            }
            return result;
        }

        /**
         * @brief Solve the exact phase-weighted fixed-quota assignment.
         *
         * This is a deterministic successive-shortest-path solver over the
         * bipartite expert-to-tier graph. Primary cost is exact integer
         * nanoseconds, secondary cost is the number of incumbent moves, and
         * the final expert-id term assigns lower ids to more-preferred tiers.
         * Reverse residual edges make the result globally optimal rather than
         * a sequence of locally attractive swaps.
         */
        RoutedExpertLayerPlacement buildMinimumCostPlacement(
            const MoERoutedExpertPlacementPlan &plan,
            const MoERoutedExpertModelMetadata &metadata,
            int layer,
            const HistogramEvidence &histogram,
            const PhaseServiceProfileView &profile,
            const RoutedExpertLayerPlacement *previous,
            size_t routed_expert_bytes_per_expert)
        {
            const auto quota_seed = buildPlacementFromExpertOrder(
                plan,
                metadata,
                layer,
                byIdExpertOrder(metadata.num_experts),
                routed_expert_bytes_per_expert);
            const int tier_count = static_cast<int>(plan.routed_tiers.size());
            std::vector<int> quotas(static_cast<std::size_t>(tier_count), 0);
            for (const int tier : quota_seed.routed_expert_tier)
                ++quotas.at(static_cast<std::size_t>(tier));

            std::vector<int> priority_order(
                static_cast<std::size_t>(tier_count));
            std::iota(priority_order.begin(), priority_order.end(), 0);
            std::sort(
                priority_order.begin(),
                priority_order.end(),
                [&](int lhs, int rhs)
                {
                    return plan.routed_tiers[static_cast<std::size_t>(lhs)]
                               .priority <
                           plan.routed_tiers[static_cast<std::size_t>(rhs)]
                               .priority;
                });
            std::vector<int> priority_rank(
                static_cast<std::size_t>(tier_count), -1);
            for (int rank = 0; rank < tier_count; ++rank)
            {
                priority_rank[static_cast<std::size_t>(
                    priority_order[static_cast<std::size_t>(rank)])] = rank;
            }

            struct ResidualEdge
            {
                int to = -1;
                int reverse = -1;
                int capacity = 0;
                ExactAssignmentCost cost;
                int assignment_tier = -1;
            };
            const int source = 0;
            const int expert_begin = 1;
            const int tier_begin = expert_begin + metadata.num_experts;
            const int sink = tier_begin + tier_count;
            const int node_count = sink + 1;
            std::vector<std::vector<ResidualEdge>> graph(
                static_cast<std::size_t>(node_count));
            const auto add_edge = [&graph](
                                      int from,
                                      int to,
                                      int capacity,
                                      ExactAssignmentCost cost,
                                      int assignment_tier = -1)
            {
                const int reverse_at_to =
                    static_cast<int>(graph[static_cast<std::size_t>(to)].size());
                const int reverse_at_from =
                    static_cast<int>(graph[static_cast<std::size_t>(from)].size());
                graph[static_cast<std::size_t>(from)].push_back({
                    .to = to,
                    .reverse = reverse_at_to,
                    .capacity = capacity,
                    .cost = cost,
                    .assignment_tier = assignment_tier,
                });
                graph[static_cast<std::size_t>(to)].push_back({
                    .to = from,
                    .reverse = reverse_at_from,
                    .capacity = 0,
                    .cost = negateExactCost(cost),
                    .assignment_tier = -1,
                });
            };

            const ExactAssignmentCost zero;
            for (int expert = 0; expert < metadata.num_experts; ++expert)
            {
                add_edge(source, expert_begin + expert, 1, zero);
                for (int tier = 0; tier < tier_count; ++tier)
                {
                    const auto tie_value =
                        static_cast<__int128>(expert) *
                        static_cast<__int128>(
                            tier_count - 1 -
                            priority_rank[static_cast<std::size_t>(tier)]);
                    if (tie_value >
                        static_cast<__int128>(
                            std::numeric_limits<int64_t>::max()))
                    {
                        throw std::overflow_error(
                            "MoE deterministic expert-id assignment tie overflowed");
                    }
                    add_edge(
                        expert_begin + expert,
                        tier_begin + tier,
                        1,
                        {
                            .service_time_ns = phaseWeightedServiceTime(
                                histogram,
                                profile,
                                tier,
                                layer,
                                expert),
                            .moved_experts =
                                previous &&
                                        previous->routed_expert_tier[
                                            static_cast<std::size_t>(expert)] !=
                                            tier
                                    ? 1
                                    : 0,
                            .expert_id_tie =
                                static_cast<int64_t>(tie_value),
                        },
                        tier);
                }
            }
            for (int tier = 0; tier < tier_count; ++tier)
            {
                add_edge(
                    tier_begin + tier,
                    sink,
                    quotas[static_cast<std::size_t>(tier)],
                    zero);
            }

            std::vector<ExactAssignmentCost> potential(
                static_cast<std::size_t>(node_count));
            for (int flow = 0; flow < metadata.num_experts; ++flow)
            {
                struct QueueEntry
                {
                    ExactAssignmentCost distance;
                    int node = -1;
                };
                struct QueueGreater
                {
                    bool operator()(
                        const QueueEntry &lhs,
                        const QueueEntry &rhs) const noexcept
                    {
                        if (rhs.distance < lhs.distance)
                            return true;
                        if (lhs.distance < rhs.distance)
                            return false;
                        return lhs.node > rhs.node;
                    }
                };

                std::vector<std::optional<ExactAssignmentCost>> distance(
                    static_cast<std::size_t>(node_count));
                std::vector<int> previous_node(
                    static_cast<std::size_t>(node_count), -1);
                std::vector<int> previous_edge(
                    static_cast<std::size_t>(node_count), -1);
                std::priority_queue<
                    QueueEntry,
                    std::vector<QueueEntry>,
                    QueueGreater>
                    queue;
                distance[static_cast<std::size_t>(source)] = zero;
                queue.push({zero, source});

                while (!queue.empty())
                {
                    const auto current = queue.top();
                    queue.pop();
                    const auto &known =
                        distance[static_cast<std::size_t>(current.node)];
                    if (!known || !(current.distance == *known))
                        continue;
                    const auto &edges =
                        graph[static_cast<std::size_t>(current.node)];
                    for (int edge_index = 0;
                         edge_index < static_cast<int>(edges.size());
                         ++edge_index)
                    {
                        const auto &edge =
                            edges[static_cast<std::size_t>(edge_index)];
                        if (edge.capacity <= 0)
                            continue;
                        auto reduced = addExactCost(
                            edge.cost,
                            potential[static_cast<std::size_t>(current.node)]);
                        reduced = addExactCost(
                            reduced,
                            negateExactCost(
                                potential[static_cast<std::size_t>(edge.to)]));
                        if (reduced < zero)
                        {
                            throw std::logic_error(
                                "MoE exact tier-assignment residual cost became negative");
                        }
                        const auto candidate = addExactCost(
                            current.distance, reduced);
                        auto &next_distance =
                            distance[static_cast<std::size_t>(edge.to)];
                        if (!next_distance || candidate < *next_distance)
                        {
                            next_distance = candidate;
                            previous_node[static_cast<std::size_t>(edge.to)] =
                                current.node;
                            previous_edge[static_cast<std::size_t>(edge.to)] =
                                edge_index;
                            queue.push({candidate, edge.to});
                        }
                    }
                }
                if (!distance[static_cast<std::size_t>(sink)])
                {
                    throw std::logic_error(
                        "MoE exact tier-assignment graph cannot satisfy fixed quotas");
                }
                for (int node = 0; node < node_count; ++node)
                {
                    const auto &node_distance =
                        distance[static_cast<std::size_t>(node)];
                    if (node_distance)
                    {
                        potential[static_cast<std::size_t>(node)] =
                            addExactCost(
                                potential[static_cast<std::size_t>(node)],
                                *node_distance);
                    }
                }
                for (int node = sink; node != source;)
                {
                    const int from =
                        previous_node[static_cast<std::size_t>(node)];
                    const int edge_index =
                        previous_edge[static_cast<std::size_t>(node)];
                    if (from < 0 || edge_index < 0)
                    {
                        throw std::logic_error(
                            "MoE exact tier-assignment augmenting path is incomplete");
                    }
                    auto &edge = graph[static_cast<std::size_t>(from)]
                                      [static_cast<std::size_t>(edge_index)];
                    --edge.capacity;
                    ++graph[static_cast<std::size_t>(node)]
                           [static_cast<std::size_t>(edge.reverse)]
                               .capacity;
                    node = from;
                }
            }

            RoutedExpertLayerPlacement placement;
            placement.layer = layer;
            placement.routed_expert_tier.assign(
                static_cast<std::size_t>(metadata.num_experts), -1);
            for (int expert = 0; expert < metadata.num_experts; ++expert)
            {
                for (const auto &edge :
                     graph[static_cast<std::size_t>(expert_begin + expert)])
                {
                    if (edge.assignment_tier >= 0 && edge.capacity == 0)
                    {
                        placement.routed_expert_tier[
                            static_cast<std::size_t>(expert)] =
                            edge.assignment_tier;
                        break;
                    }
                }
                if (placement.routed_expert_tier[
                        static_cast<std::size_t>(expert)] < 0)
                {
                    throw std::logic_error(
                        "MoE exact tier-assignment omitted an expert");
                }
            }
            return placement;
        }

        std::vector<RoutedExpertLayerPlacement> convertExplicitMasksToPlacements(
            const std::vector<MoERoutedExpertLayerTierMask> &explicit_masks,
            const MoERoutedExpertModelMetadata &metadata,
            const MoERoutedExpertPlacementPlan &plan)
        {
            std::vector<RoutedExpertLayerPlacement> placements;
            placements.reserve(static_cast<size_t>(metadata.num_layers));
            for (int layer = 0; layer < metadata.num_layers; ++layer)
            {
                RoutedExpertLayerPlacement placement;
                placement.layer = layer;
                placement.routed_expert_tier.assign(static_cast<size_t>(metadata.num_experts), -1);
                placements.push_back(std::move(placement));
            }

            for (const auto &mask : explicit_masks)
            {
                if (mask.layer < 0 || mask.layer >= metadata.num_layers)
                    throw std::invalid_argument("Explicit MoE expert tier mask references layer outside model metadata range: " + std::to_string(mask.layer));
                if (mask.tier_index < 0 || mask.tier_index >= static_cast<int>(plan.routed_tiers.size()))
                    throw std::invalid_argument("Explicit MoE expert tier mask references unknown tier index: " + std::to_string(mask.tier_index));

                auto &placement = placements[static_cast<size_t>(mask.layer)];
                for (const int expert_id : mask.expert_ids)
                {
                    if (expert_id < 0 || expert_id >= metadata.num_experts)
                        throw std::invalid_argument("Explicit MoE expert tier mask references expert outside model metadata range: " + std::to_string(expert_id));
                    auto &assigned_tier = placement.routed_expert_tier[static_cast<size_t>(expert_id)];
                    if (assigned_tier != -1)
                    {
                        throw std::invalid_argument("Explicit MoE expert tier masks assign layer " + std::to_string(mask.layer) +
                                                    " expert " + std::to_string(expert_id) + " more than once");
                    }
                    assigned_tier = mask.tier_index;
                }
            }

            return placements;
        }

        std::vector<RoutedExpertLayerPlacement> explicitPlacements(
            const MoERoutedExpertPlacementPlan &plan,
            const MoERoutedExpertModelMetadata &metadata,
            const MoERoutedExpertPlacementPlannerOptions &options)
        {
            if (!options.explicit_placements.empty())
                return options.explicit_placements;
            if (!options.explicit_masks.empty())
                return convertExplicitMasksToPlacements(options.explicit_masks, metadata, plan);
            if (!plan.placements.empty())
                return plan.placements;
            throw std::invalid_argument("ExplicitMasks MoE expert residency policy requires explicit placements or masks");
        }

        std::vector<RoutedExpertLayerPlacement> staticByIdPlacements(
            const MoERoutedExpertPlacementPlan &plan,
            const MoERoutedExpertModelMetadata &metadata,
            size_t routed_expert_bytes_per_expert)
        {
            std::vector<RoutedExpertLayerPlacement> placements;
            placements.reserve(static_cast<size_t>(metadata.num_layers));
            for (int layer = 0; layer < metadata.num_layers; ++layer)
            {
                const auto expert_order = initialExpertOrder(
                    plan, layer, metadata.num_experts);
                placements.push_back(buildPlacementFromExpertOrder(
                    plan,
                    metadata,
                    layer,
                    expert_order,
                    routed_expert_bytes_per_expert));
            }
            return placements;
        }

        std::vector<RoutedExpertLayerPlacement> histogramTieredCachePlacements(
            const MoERoutedExpertPlacementPlan &plan,
            const MoERoutedExpertModelMetadata &metadata,
            const MoERoutedExpertPlacementPlannerOptions &options,
            size_t routed_expert_bytes_per_expert)
        {
            const auto histogram = histogramEvidence(options);
            if (!histogram)
                return staticByIdPlacements(plan, metadata, routed_expert_bytes_per_expert);

            if (histogram.numLayers() < metadata.num_layers ||
                histogram.numExperts() < metadata.num_experts)
            {
                throw std::invalid_argument("DecodeExpertHistogram shape is smaller than MoE expert planner model metadata");
            }

            const auto service_profile = phaseServiceProfileView(
                plan,
                metadata,
                options.phase_service_profile);
            const auto previous = previousByLayer(
                plan,
                metadata,
                options.rebalancer);

            std::vector<RoutedExpertLayerPlacement> placements;
            placements.reserve(static_cast<size_t>(metadata.num_layers));
            for (int layer = 0; layer < metadata.num_layers; ++layer)
            {
                const bool has_counts = histogramLayerHasCounts(
                    histogram, layer, metadata.num_experts);
                if (has_counts && service_profile)
                {
                    placements.push_back(buildMinimumCostPlacement(
                        plan,
                        metadata,
                        layer,
                        histogram,
                        service_profile,
                        previous[static_cast<std::size_t>(layer)],
                        routed_expert_bytes_per_expert));
                    continue;
                }
                if (!has_counts &&
                    previous[static_cast<std::size_t>(layer)] != nullptr)
                {
                    placements.push_back(
                        *previous[static_cast<std::size_t>(layer)]);
                    continue;
                }
                const auto expert_order = has_counts
                                              ? histogramExpertOrder(
                                                    histogram,
                                                    layer,
                                                    metadata.num_experts)
                                              : initialExpertOrder(
                                                    plan,
                                                    layer,
                                                    metadata.num_experts);
                placements.push_back(buildPlacementFromExpertOrder(
                    plan,
                    metadata,
                    layer,
                    expert_order,
                    routed_expert_bytes_per_expert));
            }
            return placements;
        }

        MoERoutedExpertDomainMemoryEstimate &domainEstimate(
            MoERoutedExpertPlacementMemoryEstimate &estimate,
            const std::string &domain)
        {
            auto found = std::find_if(estimate.domains.begin(), estimate.domains.end(), [&](const auto &entry)
                                      { return entry.domain == domain; });
            if (found != estimate.domains.end())
                return *found;

            MoERoutedExpertDomainMemoryEstimate entry;
            entry.domain = domain;
            estimate.domains.push_back(std::move(entry));
            return estimate.domains.back();
        }

        MoERoutedExpertPlacementMemoryEstimate estimateMemory(
            const MoERoutedExpertPlacementPlan &planned_plan,
            const MoERoutedExpertModelMetadata &metadata,
            size_t routed_expert_bytes_per_expert)
        {
            MoERoutedExpertPlacementMemoryEstimate estimate;
            estimate.shared_expert_domain = planned_plan.shared_expert_domain;
            estimate.routed_expert_bytes_per_expert = routed_expert_bytes_per_expert;
            estimate.shared_expert_bytes_per_layer = MoERoutedExpertPlacementPlanner::estimateSharedExpertBytesPerLayer(metadata);
            estimate.total_shared_expert_bytes = MoERoutedExpertPlacementPlanner::estimateTotalSharedExpertBytes(metadata);
            estimate.tiers.reserve(planned_plan.routed_tiers.size());

            if (!planned_plan.shared_expert_domain.empty())
            {
                auto &shared_domain = domainEstimate(estimate, planned_plan.shared_expert_domain);
                shared_domain.shared_expert_bytes = estimate.total_shared_expert_bytes;
            }

            for (size_t tier_idx = 0; tier_idx < planned_plan.routed_tiers.size(); ++tier_idx)
            {
                const auto &tier = planned_plan.routed_tiers[tier_idx];
                MoERoutedExpertTierMemoryEstimate tier_estimate;
                tier_estimate.tier_index = static_cast<int>(tier_idx);
                tier_estimate.tier_name = tier.name;
                tier_estimate.domain = tier.domain;
                estimate.tiers.push_back(std::move(tier_estimate));
                (void)domainEstimate(estimate, tier.domain);
            }

            for (const auto &placement : planned_plan.placements)
            {
                for (const int tier_idx : placement.routed_expert_tier)
                {
                    if (tier_idx < 0 || tier_idx >= static_cast<int>(estimate.tiers.size()))
                        continue;
                    auto &tier_estimate = estimate.tiers[static_cast<size_t>(tier_idx)];
                    ++tier_estimate.routed_expert_count;
                    tier_estimate.routed_expert_bytes += routed_expert_bytes_per_expert;
                    estimate.total_routed_expert_bytes += routed_expert_bytes_per_expert;
                }
            }

            for (const auto &tier_estimate : estimate.tiers)
            {
                auto &domain = domainEstimate(estimate, tier_estimate.domain);
                domain.routed_expert_bytes += tier_estimate.routed_expert_bytes;
            }

            return estimate;
        }

        // ---------------------------------------------------------------------------
        // RoutedTierRebalanced: deterministic demand-ranked placement with diagnostics
        // ---------------------------------------------------------------------------

        int findFallbackTierIndexRebalanced(const MoERoutedExpertPlacementPlan &plan)
        {
            return findFallbackTierIndex(plan);
        }

        MoERoutedTierRebalanceLayerDiagnostics buildLayerDiagnostics(
            const MoERoutedExpertPlacementPlan &plan,
            const MoERoutedExpertModelMetadata &metadata,
            const RoutedExpertLayerPlacement &placement,
            const HistogramEvidence &histogram,
            int layer,
            size_t routed_expert_bytes_per_expert)
        {
            MoERoutedTierRebalanceLayerDiagnostics diag;
            diag.layer = layer;
            diag.tier_expert_counts.assign(plan.routed_tiers.size(), 0);

            const int fallback_tier = findFallbackTierIndex(plan);

            uint64_t total_activations = 0;
            uint64_t gpu_tier_activations = 0;
            int gpu_experts = 0;

            for (int expert_id = 0; expert_id < metadata.num_experts; ++expert_id)
            {
                const int tier_idx = placement.routed_expert_tier[static_cast<size_t>(expert_id)];
                if (tier_idx >= 0 && tier_idx < static_cast<int>(plan.routed_tiers.size()))
                    ++diag.tier_expert_counts[static_cast<size_t>(tier_idx)];

                if (histogram)
                    total_activations += histogram.activationCount(layer, expert_id);

                const bool is_fallback = (tier_idx == fallback_tier) && (fallback_tier >= 0);
                if (!is_fallback && tier_idx >= 0)
                {
                    ++gpu_experts;
                    if (histogram)
                        gpu_tier_activations += histogram.activationCount(layer, expert_id);
                    diag.gpu_tier_memory_bytes += routed_expert_bytes_per_expert;
                }
            }

            diag.gpu_coverage_ratio = metadata.num_experts > 0
                                          ? static_cast<float>(gpu_experts) / static_cast<float>(metadata.num_experts)
                                          : 0.0f;

            if (histogram && total_activations > 0)
            {
                diag.expected_gpu_hit_rate = static_cast<float>(gpu_tier_activations) / static_cast<float>(total_activations);
                diag.expected_cpu_fallback_rows = 1.0f - diag.expected_gpu_hit_rate;
            }
            else
            {
                diag.expected_gpu_hit_rate = diag.gpu_coverage_ratio;
                diag.expected_cpu_fallback_rows = 1.0f - diag.gpu_coverage_ratio;
            }

            return diag;
        }

        struct RoutedTierRebalancedResult
        {
            std::vector<RoutedExpertLayerPlacement> placements;
            MoERoutedTierRebalanceDiagnostics diagnostics;
        };

        RoutedTierRebalancedResult routedTierRebalancedPlacements(
            const MoERoutedExpertPlacementPlan &plan,
            const MoERoutedExpertModelMetadata &metadata,
            const MoERoutedExpertPlacementPlannerOptions &options,
            size_t routed_expert_bytes_per_expert)
        {
            const auto histogram = histogramEvidence(options);

            if (histogram &&
                (histogram.numLayers() < metadata.num_layers ||
                 histogram.numExperts() < metadata.num_experts))
            {
                throw std::invalid_argument("DecodeExpertHistogram shape is smaller than MoE expert planner model metadata");
            }

            const auto service_profile = phaseServiceProfileView(
                plan,
                metadata,
                options.phase_service_profile);
            const auto previous = previousByLayer(
                plan,
                metadata,
                options.rebalancer);

            RoutedTierRebalancedResult result;
            result.placements.reserve(static_cast<size_t>(metadata.num_layers));
            result.diagnostics.histogram_used = static_cast<bool>(histogram);
            result.diagnostics.phase_service_profile_used =
                static_cast<bool>(service_profile);
            if (service_profile)
            {
                result.diagnostics.phase_service_profile_identity =
                    service_profile.profile->identity;
            }

            float sum_gpu_hit_rate = 0.0f;
            float sum_cpu_fallback = 0.0f;
            float sum_gpu_coverage = 0.0f;

            for (int layer = 0; layer < metadata.num_layers; ++layer)
            {
                const bool has_counts = histogram && histogramLayerHasCounts(
                    histogram, layer, metadata.num_experts);
                RoutedExpertLayerPlacement placement;
                if (has_counts && service_profile)
                {
                    placement = buildMinimumCostPlacement(
                        plan,
                        metadata,
                        layer,
                        histogram,
                        service_profile,
                        previous[static_cast<std::size_t>(layer)],
                        routed_expert_bytes_per_expert);
                }
                else if (!has_counts &&
                         previous[static_cast<std::size_t>(layer)] != nullptr)
                {
                    placement =
                        *previous[static_cast<std::size_t>(layer)];
                }
                else
                {
                    const auto expert_order = has_counts
                                                  ? histogramExpertOrder(
                                                        histogram,
                                                        layer,
                                                        metadata.num_experts)
                                                  : initialExpertOrder(
                                                        plan,
                                                        layer,
                                                        metadata.num_experts);
                    placement = buildPlacementFromExpertOrder(
                        plan,
                        metadata,
                        layer,
                        expert_order,
                        routed_expert_bytes_per_expert);
                }
                result.placements.push_back(placement);

                auto layer_diag = buildLayerDiagnostics(
                    plan, metadata, placement, histogram, layer, routed_expert_bytes_per_expert);
                sum_gpu_hit_rate += layer_diag.expected_gpu_hit_rate;
                sum_cpu_fallback += layer_diag.expected_cpu_fallback_rows;
                sum_gpu_coverage += layer_diag.gpu_coverage_ratio;
                result.diagnostics.layers.push_back(std::move(layer_diag));
            }

            const float n = static_cast<float>(metadata.num_layers);
            if (n > 0.0f)
            {
                result.diagnostics.avg_gpu_hit_rate = sum_gpu_hit_rate / n;
                result.diagnostics.avg_cpu_fallback_rows = sum_cpu_fallback / n;
                result.diagnostics.avg_gpu_coverage_ratio = sum_gpu_coverage / n;
            }

            return result;
        }

    } // namespace

    MoERoutedExpertPlacementPlannerResult MoERoutedExpertPlacementPlanner::plan(const MoERoutedExpertPlacementPlannerInput &input)
    {
        return plan(input.plan, input.metadata, input.options);
    }

    MoERoutedExpertPlacementPlannerResult MoERoutedExpertPlacementPlanner::plan(
        const MoERoutedExpertPlacementPlan &base_plan,
        const MoERoutedExpertModelMetadata &metadata,
        const MoERoutedExpertPlacementPlannerOptions &options)
    {
        if (!base_plan.enabled)
        {
            MoERoutedExpertPlacementPlannerResult result;
            result.planned_plan = base_plan;
            return result;
        }

        validateMetadata(metadata);

        const auto base_validation = validateMoERoutedExpertPlacementPlan(base_plan);
        if (!base_validation.ok())
            throw std::invalid_argument(formatValidationErrors(base_validation));

        MoERoutedExpertPlacementPlan planned_plan = base_plan;
        const size_t routed_expert_bytes_per_expert = estimateRoutedExpertBytesPerExpert(metadata);

        MoERoutedExpertPlacementPlannerResult result;

        switch (base_plan.residency_policy)
        {
        case RoutedExpertResidencyPolicy::StaticById:
            planned_plan.placements = staticByIdPlacements(base_plan, metadata, routed_expert_bytes_per_expert);
            break;
        case RoutedExpertResidencyPolicy::ExplicitMasks:
            planned_plan.placements = explicitPlacements(base_plan, metadata, options);
            break;
        case RoutedExpertResidencyPolicy::HistogramTieredCache:
            planned_plan.placements = histogramTieredCachePlacements(base_plan, metadata, options, routed_expert_bytes_per_expert);
            break;
        case RoutedExpertResidencyPolicy::RoutedTierRebalanced:
        {
            auto rebalanced = routedTierRebalancedPlacements(base_plan, metadata, options, routed_expert_bytes_per_expert);
            planned_plan.placements = std::move(rebalanced.placements);
            result.rebalance_diagnostics = std::move(rebalanced.diagnostics);
            break;
        }
        case RoutedExpertResidencyPolicy::Disabled:
            if (base_plan.placements.empty())
                throw std::invalid_argument("Routed-expert placement planner cannot plan an enabled overlay with Disabled residency policy and no placements");
            planned_plan.placements = base_plan.placements;
            break;
        }

        // Concrete placements are now the sole epoch-one authority. Retaining
        // the setup permutation would leave two equivalent but independently
        // mutable descriptions in the frozen graph identity.
        planned_plan.initial_layer_order_overrides.clear();

        const MoERoutedExpertPlacementValidationOptions validation_options{
            .layer_count = metadata.num_layers,
            .routed_expert_count = metadata.num_experts,
        };
        const auto planned_validation = validateMoERoutedExpertPlacementPlan(planned_plan, validation_options);
        if (!planned_validation.ok())
            throw std::invalid_argument(formatValidationErrors(planned_validation));

        result.planned_plan = std::move(planned_plan);
        result.memory = estimateMemory(result.planned_plan, metadata, routed_expert_bytes_per_expert);
        return result;
    }

    size_t MoERoutedExpertPlacementPlanner::estimateRoutedExpertBytesPerExpert(const MoERoutedExpertModelMetadata &metadata)
    {
        validateMetadata(metadata);
        const long double elements = 3.0L * static_cast<long double>(metadata.d_model) *
                                     static_cast<long double>(metadata.routed_intermediate_size);
        const long double bytes_per_weight = WeightMemoryEstimator::getNativeBytesPerWeight(metadata.routed_quant_type);
        return ceilBytes(elements * bytes_per_weight);
    }

    size_t MoERoutedExpertPlacementPlanner::estimateSharedExpertBytesPerLayer(const MoERoutedExpertModelMetadata &metadata)
    {
        validateMetadata(metadata);
        if (!metadata.has_shared_expert)
            return 0;
        const long double elements = 3.0L * static_cast<long double>(metadata.d_model) *
                                     static_cast<long double>(metadata.shared_intermediate_size);
        const long double bytes_per_weight = WeightMemoryEstimator::getNativeBytesPerWeight(metadata.shared_quant_type);
        return ceilBytes(elements * bytes_per_weight);
    }

    size_t MoERoutedExpertPlacementPlanner::estimateTotalSharedExpertBytes(const MoERoutedExpertModelMetadata &metadata)
    {
        validateMetadata(metadata);
        return estimateSharedExpertBytesPerLayer(metadata) * static_cast<size_t>(metadata.num_layers);
    }

} // namespace llaminar2
