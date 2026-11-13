/**
 * @file ParityNumericalAggregation.h
 * @brief Typed rules for aggregating heterogeneous MoE parity checkpoints.
 *
 * Every checkpoint remains individually measured and serialized. These rules
 * only decide whether two tensors describe the same numerical branch and can
 * therefore contribute to one elementwise layer-cosine aggregate.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace llaminar2::test::parity
{
    /**
     * @brief Hard lower bound for a production recursive-MTP branch aggregate.
     *
     * Recursive prediction repeatedly feeds the preceding sidecar result into
     * the next depth. A merely marginal aggregate at the final retained depth
     * is therefore an early warning for state or expert-execution drift even
     * when the sampled token remains unchanged. Model-specific thresholds may
     * be stricter. A typed model/MTP-policy definition may explicitly supply
     * a different recursive aggregate floor when a measured quantized
     * recurrence needs it; callers that do not carry such a contract retain
     * this strict default.
     */
    inline constexpr double
        kMinimumProductionRecursiveMTPAggregateCosine = 0.99;

    /**
     * @brief Smallest norm product for which cosine is independently stable.
     *
     * This is the established resolution boundary used by the canonical
     * parity tensor comparator.  Below it, two independently accumulated
     * nonzero vectors still prove execution, but their angle is conditioned by
     * errors at the numerical floor and must be certified through the
     * completed semantic output instead of pretending that cosine is a stable
     * standalone metric.
     */
    inline constexpr double kParityCosineNormProductResolution = 1.0e-10;

    /**
     * @brief Resolve the recursive-MTP aggregate threshold for one typed case.
     * @param model_threshold Model/backend-specific decode threshold.
     * @param production_floor Typed case floor, or the strict default.
     * @return Stronger of the model threshold and recursive aggregate floor.
     */
    constexpr double productionRecursiveMTPAggregateRequiredCosine(
        double model_threshold,
        double production_floor =
            kMinimumProductionRecursiveMTPAggregateCosine) noexcept
    {
        return std::max(model_threshold, production_floor);
    }

    /**
     * @brief Apply the production recursive-MTP aggregate gate.
     * @param aggregate_cosine Mean cosine of comparable numerical checkpoints.
     * @param model_threshold Model/backend-specific decode threshold.
     * @param production_floor Typed case floor, or the strict default.
     * @return True only when the aggregate satisfies the resolved lower bound.
     */
    constexpr bool productionRecursiveMTPAggregatePasses(
        double aggregate_cosine,
        double model_threshold,
        double production_floor =
            kMinimumProductionRecursiveMTPAggregateCosine) noexcept
    {
        return aggregate_cosine >=
               productionRecursiveMTPAggregateRequiredCosine(
                   model_threshold, production_floor);
    }

    /**
     * @brief Mutually exclusive numerical state of one routed contribution.
     *
     * A contribution is either unavailable, invalid, exactly zero on both
     * paths, zero on only one path, or a comparable nonzero vector. Keeping
     * that distinction typed prevents an arbitrary magnitude threshold from
     * turning a valid low-energy expert result into a missing execution.
     */
    enum class RoutedExpertContributionState
    {
        InvalidGeometry,
        InvalidEvidence,
        NoComparableRows,
        OneSidedZero,
        ExactZeroEquality,
        ComparableNonzeroBelowCosineResolution,
        ComparableNonzero,
    };

    /**
     * @brief Causal relationship between one routed layer and its reference.
     *
     * Per-route evidence needs only identical layer input for the matched
     * expert. A dense post-return aggregate additionally needs the current
     * top-k route set to match because it no longer exposes the individual
     * addends. Keeping current-route and prior-route divergence distinct
     * prevents a different branch from either certifying or convicting an
     * asynchronous remote expert publication.
     */
    enum class RoutedExpertReferenceLineage
    {
        Canonical, ///< Same input lineage and same current route set.
        DivergedAtCurrentRouting, ///< Same input, different current route set.
        DivergedByPriorRouting, ///< Input already differs due to an earlier route.
    };

    /**
     * @brief Typed lineage result for one layer and its successor.
     *
     * `current_layer` describes which numerical witnesses are valid at the
     * current routed layer. `next_layer` is the input lineage propagated to
     * the next transformer layer. A current route-set difference therefore
     * preserves per-route comparability now but becomes prior divergence at
     * the next layer.
     */
    struct RoutedExpertReferenceLineageTransition
    {
        RoutedExpertReferenceLineage current_layer =
            RoutedExpertReferenceLineage::Canonical;
        RoutedExpertReferenceLineage next_layer =
            RoutedExpertReferenceLineage::Canonical;
    };

    /**
     * @brief Advance routed-reference lineage through one layer.
     * @param input_lineage Lineage entering the current routed layer.
     * @param current_routes_equal Whether production and reference top-k sets match.
     * @return Current-layer witness state and next-layer input state.
     */
    constexpr RoutedExpertReferenceLineageTransition
    advanceRoutedExpertReferenceLineage(
        RoutedExpertReferenceLineage input_lineage,
        bool current_routes_equal) noexcept
    {
        if (input_lineage != RoutedExpertReferenceLineage::Canonical)
        {
            return {
                .current_layer =
                    RoutedExpertReferenceLineage::DivergedByPriorRouting,
                .next_layer =
                    RoutedExpertReferenceLineage::DivergedByPriorRouting,
            };
        }
        if (current_routes_equal)
            return {};
        return {
            .current_layer =
                RoutedExpertReferenceLineage::DivergedAtCurrentRouting,
            .next_layer =
                RoutedExpertReferenceLineage::DivergedByPriorRouting,
        };
    }

    /** @return Whether matched per-route values still share the same input. */
    constexpr bool routedExpertInputLineageIsCanonical(
        RoutedExpertReferenceLineage lineage) noexcept
    {
        return lineage !=
               RoutedExpertReferenceLineage::DivergedByPriorRouting;
    }

    /** @return Whether a dense route aggregate describes the same operation. */
    constexpr bool routedExpertAggregateLineageIsCanonical(
        RoutedExpertReferenceLineage lineage) noexcept
    {
        return lineage == RoutedExpertReferenceLineage::Canonical;
    }

    /** @brief Publication form observed for one routed-expert contribution. */
    enum class RoutedExpertContributionPublication
    {
        ContinuationCanonical, ///< Continuation publishes the per-route slot.
        ReturnedCanonical, ///< Remote return materializes the per-route slot.
        DeferredRemoteAggregate, ///< Remote addend appears only in the dense fold.
    };

    /**
     * @brief Final numerical proof used for one routed expert publication.
     *
     * Route execution and numerical agreement are deliberately separate.  A
     * nonzero route below the canonical cosine resolution may use the
     * independently compared post-return MoE output, while malformed,
     * one-sided-zero, and resolvable mismatches always fail closed.
     */
    enum class RoutedExpertContributionProof
    {
        Invalid,
        NoComparableRows,
        OneSidedZero,
        ExactZeroEquality,
        PerRouteCosine,
        PostReturnAggregateDeferredPublication,
        PostReturnAggregateBelowCosineResolution,
        InconclusiveAfterRouteDivergence,
        NumericalMismatch,
    };

    /**
     * @brief Test disposition of one routed-expert numerical observation.
     *
     * `Inconclusive` is deliberately distinct from both success and failure.
     * It cannot satisfy a positive movement witness, but neither can a
     * different-input comparison convict an expert implementation. Structural
     * publication defects remain failures regardless of lineage.
     */
    enum class RoutedExpertContributionDisposition
    {
        Certified,
        Inconclusive,
        Failed,
    };

    /** @return Stable artifact spelling for one routed contribution state. */
    constexpr std::string_view routedExpertContributionStateName(
        RoutedExpertContributionState state) noexcept
    {
        switch (state)
        {
        case RoutedExpertContributionState::InvalidGeometry:
            return "invalid_geometry";
        case RoutedExpertContributionState::InvalidEvidence:
            return "invalid_evidence";
        case RoutedExpertContributionState::NoComparableRows:
            return "no_comparable_rows";
        case RoutedExpertContributionState::OneSidedZero:
            return "one_sided_zero";
        case RoutedExpertContributionState::ExactZeroEquality:
            return "exact_zero_equality";
        case RoutedExpertContributionState::
            ComparableNonzeroBelowCosineResolution:
            return "comparable_nonzero_below_cosine_resolution";
        case RoutedExpertContributionState::ComparableNonzero:
            return "comparable_nonzero";
        }
        return "invalid";
    }

    /** @return Stable artifact spelling for a routed-reference lineage. */
    constexpr std::string_view routedExpertReferenceLineageName(
        RoutedExpertReferenceLineage lineage) noexcept
    {
        switch (lineage)
        {
        case RoutedExpertReferenceLineage::Canonical:
            return "canonical";
        case RoutedExpertReferenceLineage::DivergedAtCurrentRouting:
            return "diverged_at_current_routing";
        case RoutedExpertReferenceLineage::DivergedByPriorRouting:
            return "diverged_by_prior_routing";
        }
        return "invalid";
    }

    /** @return Stable artifact spelling for one contribution publication form. */
    constexpr std::string_view routedContributionPublicationName(
        RoutedExpertContributionPublication publication) noexcept
    {
        switch (publication)
        {
        case RoutedExpertContributionPublication::ContinuationCanonical:
            return "continuation_canonical";
        case RoutedExpertContributionPublication::ReturnedCanonical:
            return "returned_canonical";
        case RoutedExpertContributionPublication::DeferredRemoteAggregate:
            return "deferred_remote_aggregate";
        }
        return "invalid";
    }

    /** @return Stable artifact spelling for one routed numerical proof. */
    constexpr std::string_view routedExpertContributionProofName(
        RoutedExpertContributionProof proof) noexcept
    {
        switch (proof)
        {
        case RoutedExpertContributionProof::Invalid:
            return "invalid";
        case RoutedExpertContributionProof::NoComparableRows:
            return "no_comparable_rows";
        case RoutedExpertContributionProof::OneSidedZero:
            return "one_sided_zero";
        case RoutedExpertContributionProof::ExactZeroEquality:
            return "exact_zero_equality";
        case RoutedExpertContributionProof::PerRouteCosine:
            return "per_route_cosine";
        case RoutedExpertContributionProof::
            PostReturnAggregateDeferredPublication:
            return "post_return_aggregate_deferred_publication";
        case RoutedExpertContributionProof::
            PostReturnAggregateBelowCosineResolution:
            return "post_return_aggregate_below_cosine_resolution";
        case RoutedExpertContributionProof::
            InconclusiveAfterRouteDivergence:
            return "inconclusive_after_route_divergence";
        case RoutedExpertContributionProof::NumericalMismatch:
            return "numerical_mismatch";
        }
        return "invalid";
    }

    /** @return Whether a typed routed contribution proof certifies parity. */
    constexpr bool routedExpertContributionProofPasses(
        RoutedExpertContributionProof proof) noexcept
    {
        return proof ==
                   RoutedExpertContributionProof::ExactZeroEquality ||
               proof == RoutedExpertContributionProof::PerRouteCosine ||
               proof == RoutedExpertContributionProof::
                            PostReturnAggregateDeferredPublication ||
               proof == RoutedExpertContributionProof::
                            PostReturnAggregateBelowCosineResolution;
    }

    /** @return Complete three-state disposition for a routed proof. */
    constexpr RoutedExpertContributionDisposition
    routedExpertContributionProofDisposition(
        RoutedExpertContributionProof proof) noexcept
    {
        if (routedExpertContributionProofPasses(proof))
            return RoutedExpertContributionDisposition::Certified;
        if (proof == RoutedExpertContributionProof::NoComparableRows ||
            proof == RoutedExpertContributionProof::
                         InconclusiveAfterRouteDivergence)
        {
            return RoutedExpertContributionDisposition::Inconclusive;
        }
        return RoutedExpertContributionDisposition::Failed;
    }

    /** @return Stable artifact spelling for a routed proof disposition. */
    constexpr std::string_view routedExpertContributionDispositionName(
        RoutedExpertContributionDisposition disposition) noexcept
    {
        switch (disposition)
        {
        case RoutedExpertContributionDisposition::Certified:
            return "certified";
        case RoutedExpertContributionDisposition::Inconclusive:
            return "inconclusive";
        case RoutedExpertContributionDisposition::Failed:
            return "failed";
        }
        return "invalid";
    }

    /**
     * @brief Route-conditioned evidence for one physically moved expert.
     *
     * Production and Hugging Face each publish one weighted contribution per
     * selected route. Matching by expert ID isolates the moved expert even
     * when an unrelated low-weight top-k member differs. Route selection,
     * physical execution, and numerical agreement remain separate facts.
     */
    struct RoutedExpertContributionComparison
    {
        RoutedExpertContributionState state =
            RoutedExpertContributionState::InvalidGeometry; ///< Exclusive evidence state.
        std::size_t routed_rows = 0u; ///< Production rows selecting the expert.
        std::size_t comparable_rows = 0u; ///< Rows sharing the selected expert.
        std::size_t production_executed_rows = 0u; ///< Nonzero production rows.
        std::size_t reference_executed_rows = 0u; ///< Nonzero reference rows.
        std::size_t exact_zero_rows = 0u; ///< Rows exactly zero in both paths.
        std::size_t one_sided_zero_rows = 0u; ///< Rows zero in only one path.
        std::size_t compared_elements = 0u; ///< Elements entering the cosine.
        float cosine_similarity = 0.0f; ///< Matched-contribution cosine.
        float production_l2_norm = 0.0f; ///< L2 norm of matched production rows.
        float reference_l2_norm = 0.0f; ///< L2 norm of matched reference rows.
        float absolute_l2_error = 0.0f; ///< L2 distance between matched rows.
        float root_mean_square_error = 0.0f; ///< Per-element matched-row RMSE.

        /** @return Whether row, width, and contribution geometry is valid. */
        [[nodiscard]] constexpr bool validGeometry() const noexcept
        {
            return state !=
                       RoutedExpertContributionState::InvalidGeometry &&
                   state !=
                       RoutedExpertContributionState::InvalidEvidence;
        }

        /** @return Whether route IDs and compared values were all finite. */
        [[nodiscard]] constexpr bool finite() const noexcept
        {
            return state != RoutedExpertContributionState::InvalidGeometry &&
                   state != RoutedExpertContributionState::InvalidEvidence;
        }

        /** @return Whether both paths selected this expert on at least one row. */
        [[nodiscard]] constexpr bool numericallyComparable() const noexcept
        {
            return state == RoutedExpertContributionState::OneSidedZero ||
                   state ==
                       RoutedExpertContributionState::ExactZeroEquality ||
                   state ==
                       RoutedExpertContributionState::
                           ComparableNonzeroBelowCosineResolution ||
                   state ==
                       RoutedExpertContributionState::ComparableNonzero;
        }

        /**
         * @brief Apply a cosine gate without conflating magnitude and absence.
         * @param cosine_threshold Required cosine for a nonzero contribution.
         * @return True for exact zero equality or a passing nonzero cosine.
         */
        [[nodiscard]] constexpr bool passes(
            float cosine_threshold) const noexcept
        {
            return state ==
                       RoutedExpertContributionState::ExactZeroEquality ||
                   ((state ==
                         RoutedExpertContributionState::ComparableNonzero ||
                     state == RoutedExpertContributionState::
                                  ComparableNonzeroBelowCosineResolution) &&
                    cosine_similarity >= cosine_threshold);
        }
    };

    /**
     * @brief Classify the complete numerical proof for one routed addend.
     *
     * The aggregate witness is consulted only when both independent paths
     * executed a nonzero route whose norm product is below the same cosine
     * resolution used by the canonical parity comparator.  It cannot forgive
     * missing execution, malformed evidence, or a resolvable per-route
     * mismatch.
     *
     * @param comparison Typed route-conditioned comparison.
     * @param cosine_threshold Required per-route and aggregate cosine.
     * @param post_return_aggregate_cosine Completed MoE output cosine.
     * @param reference_lineage Current layer's typed route/input lineage.
     * @return One explicit, auditable proof outcome.
     */
    constexpr RoutedExpertContributionProof
    classifyRoutedExpertContributionProof(
        const RoutedExpertContributionComparison &comparison,
        float cosine_threshold,
        float post_return_aggregate_cosine,
        RoutedExpertReferenceLineage reference_lineage) noexcept
    {
        switch (comparison.state)
        {
        case RoutedExpertContributionState::InvalidGeometry:
        case RoutedExpertContributionState::InvalidEvidence:
            return RoutedExpertContributionProof::Invalid;
        case RoutedExpertContributionState::NoComparableRows:
            return RoutedExpertContributionProof::NoComparableRows;
        case RoutedExpertContributionState::OneSidedZero:
            return RoutedExpertContributionProof::OneSidedZero;
        case RoutedExpertContributionState::ExactZeroEquality:
            return RoutedExpertContributionProof::ExactZeroEquality;
        case RoutedExpertContributionState::
            ComparableNonzeroBelowCosineResolution:
            if (comparison.cosine_similarity >= cosine_threshold)
                return RoutedExpertContributionProof::PerRouteCosine;
            if (post_return_aggregate_cosine >= cosine_threshold)
            {
                return RoutedExpertContributionProof::
                    PostReturnAggregateBelowCosineResolution;
            }
            return !routedExpertInputLineageIsCanonical(reference_lineage)
                       ? RoutedExpertContributionProof::
                             InconclusiveAfterRouteDivergence
                       : RoutedExpertContributionProof::NumericalMismatch;
        case RoutedExpertContributionState::ComparableNonzero:
            if (comparison.cosine_similarity >= cosine_threshold)
                return RoutedExpertContributionProof::PerRouteCosine;
            return !routedExpertInputLineageIsCanonical(reference_lineage)
                       ? RoutedExpertContributionProof::
                             InconclusiveAfterRouteDivergence
                       : RoutedExpertContributionProof::NumericalMismatch;
        }
        return RoutedExpertContributionProof::Invalid;
    }

    /**
     * @brief Join one publication form with its independent numerical proof.
     *
     * Returned and continuation slots retain per-route values and therefore
     * use the ordinary matched-expert classifier. A deferred heterogeneous
     * endpoint intentionally leaves that slot empty; its dense post-return
     * aggregate is a valid substitute only when both the input and complete
     * current route set are canonical. A different route set is explicitly
     * inconclusive rather than a false missing-execution failure.
     *
     * @param publication Physical form of the production contribution.
     * @param comparison Route-conditioned production/reference evidence.
     * @param cosine_threshold Required numerical cosine.
     * @param post_return_expert_output_cosine Completed routed-expert sum cosine.
     * @param reference_lineage Current layer's typed route/input lineage.
     * @return One explicit, auditable proof outcome.
     */
    constexpr RoutedExpertContributionProof
    classifyPublishedRoutedExpertContribution(
        RoutedExpertContributionPublication publication,
        const RoutedExpertContributionComparison &comparison,
        float cosine_threshold,
        float post_return_expert_output_cosine,
        RoutedExpertReferenceLineage reference_lineage) noexcept
    {
        if (publication !=
            RoutedExpertContributionPublication::DeferredRemoteAggregate)
        {
            return classifyRoutedExpertContributionProof(
                comparison,
                cosine_threshold,
                post_return_expert_output_cosine,
                reference_lineage);
        }
        if (!comparison.validGeometry())
            return RoutedExpertContributionProof::Invalid;
        if (comparison.comparable_rows == 0u)
            return RoutedExpertContributionProof::NoComparableRows;
        if (!routedExpertAggregateLineageIsCanonical(reference_lineage))
        {
            return RoutedExpertContributionProof::
                InconclusiveAfterRouteDivergence;
        }
        if (post_return_expert_output_cosine >= cosine_threshold)
        {
            return RoutedExpertContributionProof::
                PostReturnAggregateDeferredPublication;
        }
        return RoutedExpertContributionProof::NumericalMismatch;
    }

    /**
     * @brief Compare one moved expert's unsummed production and HF addends.
     *
     * The reference may contain leading rows absent from an incremental
     * production checkpoint; trailing rows are aligned using the ordinary
     * decode parity rule. Route order and the other selected experts may
     * differ. Duplicate, fractional, negative, or non-finite IDs fail closed.
     * Both contribution tensors use `[row, route_slot, output_column]`.
     *
     * @param production_routes Flattened production top-k expert IDs.
     * @param reference_routes Flattened Hugging Face top-k expert IDs.
     * @param production_domain_participants Per-route destination-domain IDs.
     * @param routed_expert Expert whose moved execution is being certified.
     * @param expected_domain_participant Destination-domain participant, or
     *        `-1` for a heterogeneous remote-domain route.
     * @param top_k Number of routed experts per row.
     * @param production_route_contributions Production FP32 route addends.
     * @param reference_route_contributions Hugging Face FP32 route addends.
     * @return Typed route, execution, and numerical evidence.
     */
    inline RoutedExpertContributionComparison
    compareRoutedExpertContribution(
        std::span<const float> production_routes,
        std::span<const float> reference_routes,
        std::span<const float> production_domain_participants,
        int routed_expert,
        int expected_domain_participant,
        std::size_t top_k,
        std::span<const float> production_route_contributions,
        std::span<const float> reference_route_contributions)
    {
        RoutedExpertContributionComparison result;
        if (routed_expert < 0 || expected_domain_participant < -1 ||
            top_k == 0u || production_routes.empty() ||
            production_routes.size() % top_k != 0u ||
            reference_routes.size() % top_k != 0u ||
            production_domain_participants.size() !=
                production_routes.size())
        {
            return result;
        }

        const std::size_t production_rows =
            production_routes.size() / top_k;
        const std::size_t reference_rows = reference_routes.size() / top_k;
        if (reference_rows < production_rows ||
            production_route_contributions.empty() ||
            reference_route_contributions.empty() ||
            production_route_contributions.size() %
                    production_routes.size() !=
                0u ||
            reference_route_contributions.size() % reference_routes.size() !=
                0u)
        {
            return result;
        }

        const std::size_t output_width =
            production_route_contributions.size() /
            production_routes.size();
        const std::size_t reference_width =
            reference_route_contributions.size() / reference_routes.size();
        if (output_width == 0u || reference_width != output_width)
            return result;

        const auto parse_expert_id = [](float encoded, int &decoded)
        {
            if (!std::isfinite(encoded) || encoded < 0.0f ||
                encoded > static_cast<float>(
                              std::numeric_limits<int>::max()))
            {
                return false;
            }
            decoded = static_cast<int>(encoded);
            return encoded == static_cast<float>(decoded);
        };
        const auto parse_participant_id = [](float encoded, int &decoded)
        {
            if (!std::isfinite(encoded) || encoded < -1.0f ||
                encoded > static_cast<float>(
                              std::numeric_limits<int>::max()))
            {
                return false;
            }
            decoded = static_cast<int>(encoded);
            return encoded == static_cast<float>(decoded);
        };

        const std::size_t reference_row_offset =
            reference_rows - production_rows;
        double dot = 0.0;
        double production_norm = 0.0;
        double reference_norm = 0.0;
        double squared_error = 0.0;
        result.state = RoutedExpertContributionState::NoComparableRows;
        std::vector<int> production_set(top_k);
        std::vector<int> reference_set(top_k);

        for (std::size_t row = 0u; row < production_rows; ++row)
        {
            const std::size_t production_route_offset = row * top_k;
            const std::size_t reference_route_offset =
                (reference_row_offset + row) * top_k;
            std::size_t production_slot = top_k;
            std::size_t reference_slot = top_k;
            for (std::size_t slot = 0u; slot < top_k; ++slot)
            {
                int production_expert = -1;
                int reference_expert = -1;
                int domain_participant = -2;
                if (!parse_expert_id(
                        production_routes[production_route_offset + slot],
                        production_expert) ||
                    !parse_expert_id(
                        reference_routes[reference_route_offset + slot],
                        reference_expert) ||
                    !parse_participant_id(
                        production_domain_participants[
                            production_route_offset + slot],
                        domain_participant))
                {
                    result.state =
                        RoutedExpertContributionState::InvalidEvidence;
                    return result;
                }
                production_set[slot] = production_expert;
                reference_set[slot] = reference_expert;
                if (production_expert == routed_expert &&
                    domain_participant == expected_domain_participant)
                {
                    production_slot = slot;
                }
                if (reference_expert == routed_expert)
                    reference_slot = slot;
            }

            std::sort(production_set.begin(), production_set.end());
            std::sort(reference_set.begin(), reference_set.end());
            if (std::adjacent_find(
                    production_set.begin(), production_set.end()) !=
                    production_set.end() ||
                std::adjacent_find(
                    reference_set.begin(), reference_set.end()) !=
                    reference_set.end())
            {
                result.state =
                    RoutedExpertContributionState::InvalidEvidence;
                return result;
            }

            if (production_slot == top_k)
                continue;
            ++result.routed_rows;
            if (reference_slot == top_k)
                continue;
            ++result.comparable_rows;

            const std::size_t production_offset =
                (production_route_offset + production_slot) * output_width;
            const std::size_t reference_offset =
                (reference_route_offset + reference_slot) * output_width;
            bool production_nonzero = false;
            bool reference_nonzero = false;
            for (std::size_t column = 0u; column < output_width; ++column)
            {
                const float actual =
                    production_route_contributions[
                        production_offset + column];
                const float expected =
                    reference_route_contributions[reference_offset + column];
                if (!std::isfinite(actual) || !std::isfinite(expected))
                {
                    result.state =
                        RoutedExpertContributionState::InvalidEvidence;
                    return result;
                }
                production_nonzero |= actual != 0.0f;
                reference_nonzero |= expected != 0.0f;
                dot += static_cast<double>(actual) * expected;
                production_norm += static_cast<double>(actual) * actual;
                reference_norm += static_cast<double>(expected) * expected;
                const double difference =
                    static_cast<double>(actual) -
                    static_cast<double>(expected);
                squared_error += difference * difference;
            }
            result.production_executed_rows +=
                production_nonzero ? 1u : 0u;
            result.reference_executed_rows += reference_nonzero ? 1u : 0u;
            result.exact_zero_rows +=
                !production_nonzero && !reference_nonzero ? 1u : 0u;
            result.one_sided_zero_rows +=
                production_nonzero != reference_nonzero ? 1u : 0u;
            result.compared_elements += output_width;
        }

        if (result.compared_elements == 0u)
            return result;
        result.production_l2_norm =
            static_cast<float>(std::sqrt(production_norm));
        result.reference_l2_norm =
            static_cast<float>(std::sqrt(reference_norm));
        result.absolute_l2_error =
            static_cast<float>(std::sqrt(squared_error));
        result.root_mean_square_error = static_cast<float>(
            std::sqrt(
                squared_error /
                static_cast<double>(result.compared_elements)));
        if (result.one_sided_zero_rows > 0u)
        {
            result.state = RoutedExpertContributionState::OneSidedZero;
            return result;
        }
        const double denominator =
            std::sqrt(production_norm) * std::sqrt(reference_norm);
        if (result.exact_zero_rows == result.comparable_rows)
        {
            /* A real routed expert may have an exactly zero packed weight
             * tensor. Exact zero on both independently produced paths is a
             * stronger equality result than an undefined 0/0 cosine. Keep it
             * separate from one-sided zero so missing execution fails closed. */
            result.state =
                RoutedExpertContributionState::ExactZeroEquality;
            result.cosine_similarity = 1.0f;
            return result;
        }
        if (production_norm == 0.0 || reference_norm == 0.0 ||
            denominator == 0.0)
        {
            /* Row accounting says both paths are nonzero, so an exact zero
             * norm would contradict the evidence state. Float inputs are
             * accumulated in double and cannot legitimately underflow here. */
            result.state =
                RoutedExpertContributionState::InvalidEvidence;
            return result;
        }
        result.cosine_similarity = static_cast<float>(dot / denominator);
        if (denominator <= kParityCosineNormProductResolution)
        {
            result.state =
                RoutedExpertContributionState::
                    ComparableNonzeroBelowCosineResolution;
            return result;
        }
        result.state = RoutedExpertContributionState::ComparableNonzero;
        return result;
    }

    /**
     * @brief Exact reference payload window selected by the ordinary decode crop.
     *
     * This immutable diagnostic projection carries the actual element interval
     * used for values. Its ID projection must use that same interval and prove
     * full rows/column width; it cannot independently guess a trailing ID slice.
     */
    class RouteContributionReferenceWindow final
    {
    public:
        /**
         * @brief Bind the existing full/trailing payload selection at its actual crop site.
         * @param original_elements Complete reference payload extent before cropping.
         * @param first_element Actual begin offset used to copy that payload.
         * @param selected_elements Actual copied extent, ending at the original end.
         * @return Immutable interval tied to the existing value crop.
         * @throws std::invalid_argument For empty, overflowing, nontrailing or out-of-bounds windows.
         */
        static RouteContributionReferenceWindow checked(std::size_t original_elements,
            std::size_t first_element, std::size_t selected_elements)
        {
            if (!original_elements || !selected_elements || first_element > original_elements ||
                selected_elements != original_elements - first_element)
                throw std::invalid_argument("Route contribution reference window must match the actual full/trailing crop");
            return {original_elements, first_element, selected_elements};
        }

        /**
         * @brief Project original reference IDs through exactly the captured payload row window.
         * @param ids Complete reference route IDs before selection.
         * @param production_ids Number of IDs owning the selected native payload rows.
         * @param selected_payload_elements Actual reference payload extent after cropping.
         * @param top_k Original complete route width.
         * @return Same-row reference ID subview, never a separately inferred crop.
         * @throws std::invalid_argument For mismatched widths, extents or incomplete row intervals.
         */
        std::span<const float> projectIds(std::span<const float> ids, std::size_t production_ids,
            std::size_t selected_payload_elements, std::size_t top_k) const
        {
            if (!top_k || ids.empty() || ids.size() % top_k || !production_ids || production_ids % top_k ||
                original_ % ids.size() || selected_ != selected_payload_elements)
                throw std::invalid_argument("Route contribution ID frame does not own the selected payload window");
            const auto width = original_ / ids.size();
            if (!width || first_ % width || selected_ % width || (first_ / width) % top_k ||
                selected_ / width != production_ids || production_ids > ids.size() - first_ / width)
                throw std::invalid_argument("Route contribution reference crop crosses an ID row or changes output width");
            return ids.subspan(first_ / width, production_ids);
        }
    private:
        /**
         * @brief Retain only the already checked original payload interval.
         * @param original Original complete element extent.
         * @param first Actual selected begin offset. @param selected Selected element extent.
         */
        RouteContributionReferenceWindow(std::size_t original, std::size_t first, std::size_t selected)
            : original_(original), first_(first), selected_(selected) {}
        std::size_t original_, first_, selected_;
    };

    /** @brief Complete same-row expert correspondence, or a genuinely different route set. */
    enum class RouteContributionAlignmentStatus { Aligned, DifferentExpertSets };

    /** @brief Owned reference diagnostic rows; no production or reference snapshot is mutated. */
    struct RouteContributionAlignment
    {
        RouteContributionAlignmentStatus status = RouteContributionAlignmentStatus::DifferentExpertSets;
        std::vector<float> reference_in_production_order;
    };

    /**
     * @brief Match weighted contribution blocks by original row and exact expert ID.
     * @param production_ids Native row-major selected expert IDs, encoded as exact FP32 integers.
     * @param reference_ids Independent row-major reference IDs for the same rows.
     * @param reference_values Reference [row,route,output] blocks in its own slot order.
     * @param production_elements Exact native contribution extent, not allocation capacity.
     * @param top_k Complete original route width. @param num_experts Original expert-ID domain.
     * @return Aligned immutable reference copy, or explicit unavailable correspondence for different sets.
     * @throws std::invalid_argument For missing/incomplete geometry or duplicate/nonintegral/foreign IDs.
     * @note Like compareRoutedExpertContribution, slots are found separately in each same-row ID list.
     *       This full-bank diagnostic never crosses rows, changes routes, drops unmatched experts,
     *       reconstructs expert arithmetic, or replaces the existing semantic expert sum.
     */
    inline RouteContributionAlignment alignRouteContributionReference(
        std::span<const float> production_ids, std::span<const float> reference_ids,
        std::span<const float> reference_values, std::size_t production_elements,
        std::size_t top_k, std::size_t num_experts)
    {
        if (!top_k || !num_experts || top_k > num_experts || production_ids.empty() ||
            production_ids.size() != reference_ids.size() || production_ids.size() % top_k ||
            !production_elements || production_elements != reference_values.size() ||
            production_elements % production_ids.size())
            throw std::invalid_argument("Route contribution alignment requires complete same-row geometry");
        // Validate the entire bank before reporting different sets: a malformed
        // later row must not hide behind an earlier legitimate top-k change.
        const auto valid_ids = [&](std::span<const float> ids) {
            for (std::size_t i = 0; i < ids.size(); ++i)
            {
                const float id = ids[i];
                if (!std::isfinite(id) || id < 0 || std::floor(id) != id ||
                    static_cast<double>(id) >= static_cast<double>(num_experts))
                    return false;
                const auto begin = ids.begin() + static_cast<std::ptrdiff_t>((i / top_k) * top_k);
                if (std::find(begin, ids.begin() + static_cast<std::ptrdiff_t>(i), id) !=
                    ids.begin() + static_cast<std::ptrdiff_t>(i))
                    return false;
            }
            return true;
        };
        if (!valid_ids(production_ids) || !valid_ids(reference_ids))
            throw std::invalid_argument("Route contribution alignment rejects duplicate or invalid expert IDs");
        std::vector<std::size_t> reference_slots(production_ids.size());
        for (std::size_t i = 0; i < production_ids.size(); ++i)
        {
            const auto row = (i / top_k) * top_k;
            const auto begin = reference_ids.begin() + static_cast<std::ptrdiff_t>(row);
            const auto end = begin + static_cast<std::ptrdiff_t>(top_k);
            const auto match = std::find(begin, end, production_ids[i]);
            if (match == end)
                return {}; // No fake aligned pass or intersection-only comparison.
            reference_slots[i] = row + static_cast<std::size_t>(match - begin);
        }
        const auto width = production_elements / production_ids.size();
        RouteContributionAlignment result{RouteContributionAlignmentStatus::Aligned,
                                           std::vector<float>(production_elements)};
        for (std::size_t i = 0; i < production_ids.size(); ++i)
            std::copy_n(reference_values.begin() + static_cast<std::ptrdiff_t>(reference_slots[i] * width),
                        width, result.reference_in_production_order.begin() + static_cast<std::ptrdiff_t>(i * width));
        return result;
    }

    /**
     * @brief Return whether a stage belongs in a layer cosine aggregate.
     *
     * Routing indices and weights have dedicated set/sparse-vector metrics and
     * never contribute cosine values. A raw routed-expert sum is also branch
     * dependent: when production and the reference select different expert
     * sets at the permitted low-weight top-k boundary, the two sums are not the
     * same elementwise operation. Its checkpoint remains visible in the stage
     * CSV, while the downstream combined MoE output and subsequent stages
     * continue to certify the semantic result.
     *
     * @param stage Semantic parity stage name.
     * @param routed_expert_set_exact Whether production and reference selected
     *        the same routed expert set for the current layer and row.
     * @return True only when the stage's cosine is mathematically comparable
     *         and should contribute to the layer aggregate.
     */
    constexpr bool parityStageContributesToLayerCosine(
        std::string_view stage,
        bool routed_expert_set_exact) noexcept
    {
        if (stage == "MOE_ROUTING_INDICES" ||
            stage == "MOE_ROUTING_WEIGHTS" ||
            stage == "MOE_ROUTE_CONTRIBUTIONS" ||
            stage == "MOE_ROUTE_CONTRIBUTIONS_EXPERT_ALIGNED")
        {
            return false;
        }
        if (stage == "MOE_EXPERT_OUTPUT" &&
            !routed_expert_set_exact)
        {
            return false;
        }
        return true;
    }
} // namespace llaminar2::test::parity
