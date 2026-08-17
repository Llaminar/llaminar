/**
 * @file MoERoutingBoundary.h
 * @brief Fail-closed numerical equivalence for MoE top-k routing boundaries.
 *
 * CPU and accelerator softmax implementations can place two nearly equal
 * experts on opposite sides of the top-k cutoff.  Comparing index tensors
 * literally would reject a mathematically unresolved cutoff, while accepting
 * arbitrary set overlap would hide real routing defects.  This helper proves
 * that each index set is the top-k selection of its own live router tensor and
 * that every changed expert can cross the reference boundary within the
 * observed componentwise router error.  The top-1 expert remains exact.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <vector>

namespace llaminar2::test::parity
{
    /**
     * @brief Compute the worst row-wise symmetric KL between probabilities.
     *
     * Router snapshots are already post-softmax probability distributions.
     * Applying softmax again would flatten both rows toward uniform and hide
     * meaningful routing drift.  This helper therefore consumes probability
     * rows directly, normalizing only the small sum error introduced by FP32
     * storage.  Returning the worst row prevents a good batch row from
     * averaging away a bad one.
     *
     * The comparison fails closed: null, empty, non-finite, negative, zero-sum,
     * or incomplete rows return positive infinity.
     *
     * @param reference Reference probabilities flattened by row.
     * @param production Production probabilities flattened by row.
     * @param size Total number of probabilities in each input.
     * @param row_width Number of experts in one logical router row.
     * @return Worst symmetric KL across rows, or infinity for invalid input.
     */
    inline double symmetricProbabilityKLDivergence(
        const float *reference,
        const float *production,
        size_t size,
        size_t row_width)
    {
        if (!reference || !production || size == 0 || row_width == 0 ||
            size % row_width != 0)
        {
            return std::numeric_limits<double>::infinity();
        }

        constexpr double kProbabilityFloor = 1.0e-300;
        double worst = 0.0;
        const size_t rows = size / row_width;
        for (size_t row = 0; row < rows; ++row)
        {
            const size_t offset = row * row_width;
            double reference_sum = 0.0;
            double production_sum = 0.0;
            for (size_t expert = 0; expert < row_width; ++expert)
            {
                const double p =
                    static_cast<double>(reference[offset + expert]);
                const double q =
                    static_cast<double>(production[offset + expert]);
                if (!std::isfinite(p) || !std::isfinite(q) || p < 0.0 ||
                    q < 0.0)
                {
                    return std::numeric_limits<double>::infinity();
                }
                reference_sum += p;
                production_sum += q;
            }
            if (!(reference_sum > 0.0) || !(production_sum > 0.0))
            {
                return std::numeric_limits<double>::infinity();
            }

            double reference_to_production = 0.0;
            double production_to_reference = 0.0;
            for (size_t expert = 0; expert < row_width; ++expert)
            {
                /*
                 * The floor makes exact zero probabilities logarithmically
                 * comparable without changing ordinary FP32 router values.
                 */
                const double p = std::max(
                    static_cast<double>(reference[offset + expert]) /
                        reference_sum,
                    kProbabilityFloor);
                const double q = std::max(
                    static_cast<double>(production[offset + expert]) /
                        production_sum,
                    kProbabilityFloor);
                reference_to_production += p * std::log(p / q);
                production_to_reference += q * std::log(q / p);
            }
            worst = std::max(
                worst,
                0.5 *
                    (reference_to_production + production_to_reference));
        }
        return worst;
    }

    /** @brief Evidence produced by one complete routing-boundary comparison. */
    struct MoERoutingBoundaryResult
    {
        bool evaluated = false; ///< Input geometry was complete and comparable.
        bool equivalent = false; ///< Every row satisfied the semantic contract.
        double maximum_boundary_gap =
            std::numeric_limits<double>::quiet_NaN();
        double maximum_error_limit =
            std::numeric_limits<double>::quiet_NaN();
    };

    /**
     * @brief Compare top-k expert selections using their originating routers.
     *
     * Both index tensors must contain unique, integral, in-range experts.  Each
     * set must select its own router's top-k values, and its first entry must be
     * that router's top-1 expert.  Top-1 must agree exactly across routers.  A
     * lower-rank set substitution is accepted only when its reference cutoff
     * gap is no greater than twice the observed maximum componentwise router
     * error plus a small FP32 comparison allowance.
     *
     * @param reference_indices Reference top-k indices, flattened by row.
     * @param production_indices Production top-k indices, flattened by row.
     * @param indices_size Number of entries in each index tensor.
     * @param reference_router Reference router probabilities, flattened by row.
     * @param reference_router_size Number of reference router values.
     * @param production_router Production router probabilities, flattened by row.
     * @param production_router_size Number of production router values.
     * @param top_k Routed experts per row.
     * @return Structural validity, semantic equivalence, and cutoff evidence.
     */
    inline MoERoutingBoundaryResult compareMoERoutingBoundarySelections(
        const float *reference_indices,
        const float *production_indices,
        size_t indices_size,
        const float *reference_router,
        size_t reference_router_size,
        const float *production_router,
        size_t production_router_size,
        size_t top_k = 8)
    {
        MoERoutingBoundaryResult result;
        if (!reference_indices || !production_indices || !reference_router ||
            !production_router || top_k == 0 || indices_size == 0 ||
            indices_size % top_k != 0 ||
            reference_router_size != production_router_size)
        {
            return result;
        }

        const size_t rows = indices_size / top_k;
        if (rows == 0 || production_router_size == 0 ||
            production_router_size % rows != 0)
        {
            return result;
        }
        const size_t num_experts = production_router_size / rows;
        if (num_experts < top_k)
        {
            return result;
        }

        result.evaluated = true;
        result.equivalent = true;
        result.maximum_boundary_gap = 0.0;
        result.maximum_error_limit = 0.0;

        for (size_t row_index = 0; row_index < rows; ++row_index)
        {
            const float *reference_scores =
                reference_router + row_index * num_experts;
            const float *production_scores =
                production_router + row_index * num_experts;
            std::set<int> reference_set;
            std::set<int> production_set;
            bool valid_indices = true;
            for (size_t route = 0; route < top_k; ++route)
            {
                const size_t index = row_index * top_k + route;
                const float reference_value = reference_indices[index];
                const float production_value = production_indices[index];
                const bool reference_valid =
                    std::isfinite(reference_value) &&
                    std::trunc(reference_value) == reference_value &&
                    reference_value >= 0.0f &&
                    static_cast<double>(reference_value) <
                        static_cast<double>(num_experts);
                const bool production_valid =
                    std::isfinite(production_value) &&
                    std::trunc(production_value) == production_value &&
                    production_value >= 0.0f &&
                    static_cast<double>(production_value) <
                        static_cast<double>(num_experts);
                valid_indices =
                    valid_indices && reference_valid && production_valid;
                const int reference_expert = reference_valid
                    ? static_cast<int>(reference_value)
                    : -1;
                const int production_expert = production_valid
                    ? static_cast<int>(production_value)
                    : -1;
                reference_set.insert(reference_expert);
                production_set.insert(production_expert);
            }
            valid_indices = valid_indices && reference_set.size() == top_k &&
                production_set.size() == top_k;

            double router_max_abs_error = 0.0;
            double score_scale = 0.0;
            bool finite_scores = true;
            for (size_t expert = 0; expert < num_experts; ++expert)
            {
                finite_scores = finite_scores &&
                    std::isfinite(reference_scores[expert]) &&
                    std::isfinite(production_scores[expert]);
                router_max_abs_error = std::max(
                    router_max_abs_error,
                    std::abs(
                        static_cast<double>(reference_scores[expert]) -
                        static_cast<double>(production_scores[expert])));
                score_scale = std::max(
                    score_scale,
                    std::max(
                        std::abs(static_cast<double>(reference_scores[expert])),
                        std::abs(static_cast<double>(production_scores[expert]))));
            }
            const double roundoff =
                32.0 * std::numeric_limits<float>::epsilon() *
                std::max(
                    score_scale,
                    static_cast<double>(std::numeric_limits<float>::min()));
            const double error_limit =
                2.0 * router_max_abs_error + roundoff;

            auto selects_own_top_k =
                [&](const std::set<int> &selected,
                    int selected_leader,
                    const float *scores) -> bool
            {
                if (!valid_indices || !finite_scores ||
                    !selected.contains(selected_leader))
                {
                    return false;
                }
                double selected_floor =
                    std::numeric_limits<double>::infinity();
                double excluded_ceiling =
                    -std::numeric_limits<double>::infinity();
                double global_ceiling =
                    -std::numeric_limits<double>::infinity();
                for (size_t expert = 0; expert < num_experts; ++expert)
                {
                    const double score = scores[expert];
                    global_ceiling = std::max(global_ceiling, score);
                    if (selected.contains(static_cast<int>(expert)))
                    {
                        selected_floor = std::min(selected_floor, score);
                    }
                    else
                    {
                        excluded_ceiling = std::max(excluded_ceiling, score);
                    }
                }
                return selected_floor + roundoff >= excluded_ceiling &&
                    static_cast<double>(scores[selected_leader]) + roundoff >=
                        global_ceiling;
            };

            std::vector<double> dropped_reference_scores;
            std::vector<double> added_reference_scores;
            for (const int expert : reference_set)
            {
                if (!production_set.contains(expert) && expert >= 0 &&
                    static_cast<size_t>(expert) < num_experts)
                {
                    dropped_reference_scores.push_back(
                        reference_scores[static_cast<size_t>(expert)]);
                }
            }
            for (const int expert : production_set)
            {
                if (!reference_set.contains(expert) && expert >= 0 &&
                    static_cast<size_t>(expert) < num_experts)
                {
                    added_reference_scores.push_back(
                        reference_scores[static_cast<size_t>(expert)]);
                }
            }
            std::sort(
                dropped_reference_scores.begin(),
                dropped_reference_scores.end());
            std::sort(
                added_reference_scores.begin(),
                added_reference_scores.end());

            double boundary_gap = 0.0;
            bool boundary_can_flip =
                dropped_reference_scores.size() ==
                added_reference_scores.size();
            if (boundary_can_flip)
            {
                for (size_t i = 0; i < dropped_reference_scores.size(); ++i)
                {
                    boundary_gap = std::max(
                        boundary_gap,
                        std::max(
                            0.0,
                            dropped_reference_scores[i] -
                                added_reference_scores[i]));
                }
                boundary_can_flip = boundary_gap <= error_limit;
            }

            const int reference_leader =
                static_cast<int>(reference_indices[row_index * top_k]);
            const int production_leader =
                static_cast<int>(production_indices[row_index * top_k]);
            const bool row_equivalent = valid_indices && finite_scores &&
                reference_leader == production_leader &&
                selects_own_top_k(
                    reference_set,
                    reference_leader,
                    reference_scores) &&
                selects_own_top_k(
                    production_set,
                    production_leader,
                    production_scores) &&
                boundary_can_flip;
            result.equivalent = result.equivalent && row_equivalent;
            result.maximum_boundary_gap = std::max(
                result.maximum_boundary_gap,
                boundary_gap);
            result.maximum_error_limit = std::max(
                result.maximum_error_limit,
                error_limit);
        }

        return result;
    }
} // namespace llaminar2::test::parity
