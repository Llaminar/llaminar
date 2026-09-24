/**
 * @file MoERouteConditionedProof.h
 * @brief Independent HF-input expert equations for unresolved MoE route cutoffs.
 *
 * A discontinuous top-k choice is not an expert arithmetic error. This proof
 * keeps the original HF result untouched and explicitly certifies the selected
 * equation only after routing, input, and weight-normalization checks pass.
 * No production output or input can author the independent expert bank.
 */
#pragma once

#include "MoERoutingBoundary.h"

#include <array>
#include <optional>
#include <ostream>
#include <span>

namespace llaminar2::test::parity
{
    /** @brief Independent model-owned RMS parameters, never inferred from native output. */
    struct MoEReferenceNormalization
    {
        std::vector<float> scale;
        float epsilon = 0;
    };

    /** @brief One authenticated HF-input bank and its model-owned terminal norm. */
    struct MoEIndependentReference
    {
        std::vector<float> experts;
        MoEReferenceNormalization normalization;
    };

    /** @brief The finite suffix of a routed sidecar equation; logits are not included. */
    enum class MoESuffixStage : size_t { Combined, Residual, FinalNorm, Count };
    using MoESuffixRows = std::array<std::vector<float>, static_cast<size_t>(MoESuffixStage::Count)>;

    /** @brief Independently captured non-routed HF inputs to the bounded suffix. */
    struct MoEHFSuffixOperands
    {
        std::span<const float> fc;
        std::span<const float> attention;
        std::span<const float> gated_shared;
    };

    /**
     * @brief Reconstruct combined, residual and RMS norm from independent HF operands.
     * @param routed Unchanged or route-conditioned HF-input expert equation.
     * @param operands Immutable HF operands; no production activation enters this API.
     * @param normalization Exact GGUF/HF parameter transform and model epsilon.
     * @return Complete ordered suffix, or no value for malformed/nonfinite evidence.
     *
     * FP32 additions retain the reference's parenthesization. RMS reduction is
     * independently accumulated in double; canonical HF reconstruction must
     * authenticate the resulting equation before it can adjudicate a native row.
     */
    inline std::optional<MoESuffixRows> reconstructMoEHFSuffix(
        std::span<const float> routed, const MoEHFSuffixOperands &operands,
        const MoEReferenceNormalization &normalization)
    {
        const size_t width = normalization.scale.size();
        if (!width || routed.empty() || routed.size() % width ||
            operands.fc.size() != routed.size() || operands.attention.size() != routed.size() ||
            operands.gated_shared.size() != routed.size() ||
            !std::isfinite(normalization.epsilon) || normalization.epsilon <= 0 ||
            std::any_of(normalization.scale.begin(), normalization.scale.end(),
                [](float x) { return !std::isfinite(x); }))
            return std::nullopt;
        MoESuffixRows result;
        for (auto &row : result) row.resize(routed.size());
        auto &[combined, residual, norm] = result;
        for (size_t i = 0; i < routed.size(); ++i)
        {
            if (!std::isfinite(routed[i]) || !std::isfinite(operands.fc[i]) ||
                !std::isfinite(operands.attention[i]) || !std::isfinite(operands.gated_shared[i]))
                return std::nullopt;
            combined[i] = routed[i] + operands.gated_shared[i];
            residual[i] = (operands.fc[i] + operands.attention[i]) + combined[i];
        }
        for (size_t begin = 0; begin < routed.size(); begin += width)
        {
            double sum = 0;
            for (size_t i = 0; i < width; ++i)
                sum += static_cast<double>(residual[begin + i]) * residual[begin + i];
            const float inverse = static_cast<float>(1.0 / std::sqrt(sum / width + normalization.epsilon));
            for (size_t i = 0; i < width; ++i)
                norm[begin + i] = (residual[begin + i] * inverse) * normalization.scale[i];
        }
        for (const auto &row : result)
            if (std::any_of(row.begin(), row.end(), [](float x) { return !std::isfinite(x); }))
                return std::nullopt;
        return result;
    }

    /** @brief Normalization evidence bound to a live full router and its IDs. */
    struct MoERouteWeightProof
    {
        bool valid = false;
        double maximum_error = std::numeric_limits<double>::infinity();
    };

    /**
     * @brief Verify every selected weight equals its router probability/sum.
     * @param router Full post-softmax rows, not pre-softmax logits.
     * @param indices Unique selected IDs in slot order for each row.
     * @param weights Live slot weights with the same geometry as indices.
     * @param top_k Number of slots in each row.
     * @return Fail-closed validity and worst componentwise FP32 rounding error.
     */
    inline MoERouteWeightProof proveMoENormalizedWeights(
        std::span<const float> router, std::span<const float> indices,
        std::span<const float> weights, size_t top_k)
    {
        MoERouteWeightProof result;
        if (!top_k || indices.empty() || indices.size() % top_k ||
            weights.size() != indices.size())
            return result;
        const size_t rows = indices.size() / top_k;
        if (router.empty() || router.size() % rows)
            return result;
        const size_t experts = router.size() / rows;
        if (experts < top_k || std::any_of(router.begin(), router.end(),
                [](float p) { return !std::isfinite(p) || p < 0.0f; }))
            return result;
        result.maximum_error = 0;
        for (size_t row = 0; row < rows; ++row)
        {
            std::set<size_t> unique;
            double sum = 0;
            for (size_t slot = 0; slot < top_k; ++slot)
            {
                const size_t offset = row * top_k + slot;
                const float id = indices[offset];
                if (!std::isfinite(id) || id < 0 || std::trunc(id) != id ||
                    static_cast<double>(id) >= static_cast<double>(experts) ||
                    !std::isfinite(weights[offset]) || weights[offset] < 0 ||
                    !unique.insert(static_cast<size_t>(id)).second)
                    return result;
                sum += router[row * experts + static_cast<size_t>(id)];
            }
            if (!(sum > 0))
                return result;
            for (size_t slot = 0; slot < top_k; ++slot)
            {
                const size_t offset = row * top_k + slot;
                const double expected = router[row * experts +
                    static_cast<size_t>(indices[offset])] / sum;
                result.maximum_error = std::max(result.maximum_error,
                    std::abs(expected - weights[offset]));
            }
        }
        // This allowance covers FP32 sum/division rounding, not quantization
        // drift: IDs, probabilities, and weights describe one live operation.
        result.valid = result.maximum_error <=
            16.0 * std::numeric_limits<float>::epsilon();
        return result;
    }

    /** @brief Finite vector metrics; exact zero equality is a valid equation. */
    struct MoEEquationMetrics
    {
        bool finite = false;
        double cosine = std::numeric_limits<double>::quiet_NaN();
        double relative_l2 = std::numeric_limits<double>::infinity();
        double maximum_error = std::numeric_limits<double>::infinity();
    };

    /**
     * @brief Measure one complete equation without averaging away malformed rows.
     * @param actual Observed or independently reconstructed output.
     * @param reference Independent CPU FP32 target of identical geometry.
     * @return Double-accumulated cosine, relative L2, and maximum absolute error.
     */
    inline MoEEquationMetrics measureMoEEquation(
        std::span<const float> actual, std::span<const float> reference)
    {
        MoEEquationMetrics result;
        if (actual.empty() || actual.size() != reference.size())
            return result;
        double aa = 0, rr = 0, ar = 0, error = 0, maximum = 0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            const double a = actual[i], r = reference[i], delta = a - r;
            if (!std::isfinite(a) || !std::isfinite(r))
                return result;
            aa += a * a;
            rr += r * r;
            ar += a * r;
            error += delta * delta;
            maximum = std::max(maximum, std::abs(delta));
        }
        result.finite = true;
        result.cosine = aa == 0 || rr == 0 ? (aa == rr ? 1.0 : 0.0) :
            std::clamp(ar / std::sqrt(aa * rr), -1.0, 1.0);
        result.relative_l2 = rr == 0 ? (error == 0 ? 0 :
            std::numeric_limits<double>::infinity()) : std::sqrt(error / rr);
        result.maximum_error = maximum;
        return result;
    }

    /**
     * @brief Sum an independent [expert,row,hidden] bank in selected slot order.
     * @param bank FP32 equations evaluated on HF inputs, never native inputs.
     * @param expert_ids Sorted unique ID carried by each bank plane.
     * @param indices Selected IDs for each row/slot.
     * @param weights Independently validated normalized live or HF weights.
     * @param top_k Route width.
     * @param hidden Output width per row.
     * @return Complete FP32 result, or an empty vector for malformed evidence.
     */
    inline std::vector<float> sumMoEReferenceBank(
        std::span<const float> bank, std::span<const int> expert_ids,
        std::span<const float> indices, std::span<const float> weights,
        size_t top_k, size_t hidden)
    {
        if (!top_k || !hidden || indices.empty() || indices.size() % top_k ||
            indices.size() != weights.size() || expert_ids.empty() ||
            expert_ids.front() < 0 ||
            !std::is_sorted(expert_ids.begin(), expert_ids.end()) ||
            std::adjacent_find(expert_ids.begin(), expert_ids.end()) != expert_ids.end())
            return {};
        const size_t rows = indices.size() / top_k;
        if (bank.size() / expert_ids.size() / rows != hidden ||
            bank.size() % (expert_ids.size() * rows) != 0 ||
            std::any_of(bank.begin(), bank.end(), [](float x) { return !std::isfinite(x); }))
            return {};
        std::vector<double> sum(rows * hidden, 0);
        for (size_t row = 0; row < rows; ++row)
        {
            for (size_t slot = 0; slot < top_k; ++slot)
            {
                const size_t offset = row * top_k + slot;
                const float id = indices[offset];
                if (!std::isfinite(id) || id < 0 || std::trunc(id) != id ||
                    static_cast<double>(id) > std::numeric_limits<int>::max() ||
                    !std::isfinite(weights[offset]) || weights[offset] < 0)
                    return {};
                const auto found = std::lower_bound(expert_ids.begin(), expert_ids.end(), static_cast<int>(id));
                if (found == expert_ids.end() || *found != static_cast<int>(id))
                    return {};
                const size_t base = ((found - expert_ids.begin()) * rows + row) * hidden;
                for (size_t column = 0; column < hidden; ++column)
                    sum[row * hidden + column] +=
                        static_cast<double>(bank[base + column]) * weights[offset];
            }
        }
        return {sum.begin(), sum.end()};
    }

    /** @brief Named prerequisites; none can be replaced by an output cosine. */
    struct MoERouteConditionedPrerequisites
    {
        MoERoutingBoundaryResult boundary;
        MoERouteWeightProof production_weights;
        MoERouteWeightProof reference_weights;
        double router_kl = std::numeric_limits<double>::infinity();
        double maximum_router_kl = std::numeric_limits<double>::quiet_NaN();
        bool hf_input_equivalent = false;
        bool selected_set_changed = false;

        /** @brief Admit only a validated unresolved cutoff, never an arbitrary route. */
        bool valid() const noexcept
        {
            return boundary.evaluated && boundary.equivalent &&
                production_weights.valid && reference_weights.valid &&
                hf_input_equivalent && selected_set_changed &&
                std::isfinite(router_kl) && std::isfinite(maximum_router_kl) &&
                maximum_router_kl >= 0 && router_kl <= maximum_router_kl;
        }
    };

    /** @brief Separate certification evidence; canonical metrics stay immutable. */
    struct MoERouteConditionedProof
    {
        bool evaluated = false;
        bool equivalent = false;
        bool canonical_passed = false;
        MoEEquationMetrics canonical_reconstruction;
        MoEEquationMetrics conditioned;
        double maximum_weight_error = std::numeric_limits<double>::infinity();
        std::optional<MoESuffixStage> suffix_stage;
    };

    /**
     * @brief Certify the independent equation with the unchanged cosine gate.
     * @param prerequisites Authenticated route selection, weights, and HF input.
     * @param canonical_passed Original different-route HF verdict, retained as-is.
     * @param actual Native routed output.
     * @param conditioned Independently recombined selected-route HF-input output.
     * @param canonical Original HF routed output.
     * @param reconstructed Original routes recombined from the same independent bank.
     * @param minimum_cosine Existing cell threshold, never lowered by this proof.
     * @return Explicit authority evidence, including both original and new verdicts.
     */
    inline MoERouteConditionedProof proveMoERouteConditionedEquation(
        const MoERouteConditionedPrerequisites &prerequisites,
        bool canonical_passed, std::span<const float> actual,
        std::span<const float> conditioned, std::span<const float> canonical,
        std::span<const float> reconstructed, double minimum_cosine)
    {
        MoERouteConditionedProof result;
        result.canonical_passed = canonical_passed;
        if (!prerequisites.valid() || !std::isfinite(minimum_cosine) ||
            minimum_cosine <= 0 || minimum_cosine > 1 ||
            actual.size() != canonical.size())
            return result;
        result.canonical_reconstruction = measureMoEEquation(reconstructed, canonical);
        result.conditioned = measureMoEEquation(actual, conditioned);
        result.maximum_weight_error = std::max(prerequisites.production_weights.maximum_error,
            prerequisites.reference_weights.maximum_error);
        result.evaluated = result.canonical_reconstruction.finite && result.conditioned.finite;
        // A cosine alone cannot catch an erroneously doubled publication.
        // The L2 budget is the unit-vector distance implied by the same cosine,
        // with no independent tolerance relaxation or backend-specific rule.
        result.equivalent = result.evaluated &&
            result.canonical_reconstruction.relative_l2 <= 2.0e-5 &&
            result.conditioned.cosine >= minimum_cosine &&
            result.conditioned.relative_l2 <= std::sqrt(2.0 * (1.0 - minimum_cosine));
        return result;
    }

    /** @brief Original HF input verdicts, not inferred from a better suffix cosine. */
    enum class MoEHFInputVerdict { Failed, Equivalent };

    /** @brief All non-routed dependencies must retain their original HF comparisons. */
    struct MoESuffixInputProof
    {
        MoEHFInputVerdict fc = MoEHFInputVerdict::Failed;
        MoEHFInputVerdict attention = MoEHFInputVerdict::Failed;
        MoEHFInputVerdict gated_shared = MoEHFInputVerdict::Failed;

        /** @brief Reject a missing/failed independent input before conditioning any suffix. */
        bool valid() const noexcept
        {
            return fc == MoEHFInputVerdict::Equivalent &&
                attention == MoEHFInputVerdict::Equivalent &&
                gated_shared == MoEHFInputVerdict::Equivalent;
        }
    };

    /**
     * @brief Propagate one authenticated route equation through its bounded HF suffix.
     * @param prerequisites Full-router, normalized-weight and HF-input route proof.
     * @param routed Previously certified expert equation, required even if suffix looks good.
     * @param inputs Original non-routed input verdicts; conditioning cannot replace them.
     * @param actual Native combined/residual/norm rows, used only as comparison outputs.
     * @param canonical Original immutable HF output rows and their original verdicts.
     * @param canonical_passed Original stage verdicts retained in each proof record.
     * @param reconstructed Original HF routes passed through the same independent suffix.
     * @param conditioned Native-selected routes on HF inputs through that suffix.
     * @param minimum_cosine Unchanged cell cosine gate, also bounding relative L2.
     * @return Ordered explicit proofs. No stage may pass after a failed dependency.
     */
    inline std::array<MoERouteConditionedProof, 3> proveMoERouteConditionedSuffix(
        const MoERouteConditionedPrerequisites &prerequisites,
        const MoERouteConditionedProof &routed, const MoESuffixInputProof &inputs,
        const MoESuffixRows &actual, const MoESuffixRows &canonical,
        const std::array<bool, 3> &canonical_passed,
        const std::optional<MoESuffixRows> &reconstructed,
        const std::optional<MoESuffixRows> &conditioned, double minimum_cosine)
    {
        std::array<MoERouteConditionedProof, 3> result;
        bool admitted = prerequisites.valid() && routed.evaluated && routed.equivalent &&
            inputs.valid() && reconstructed && conditioned;
        for (size_t i = 0; i < result.size(); ++i)
        {
            result[i].canonical_passed = canonical_passed[i];
            if (admitted)
            {
                result[i] = proveMoERouteConditionedEquation(prerequisites,
                    canonical_passed[i], actual[i], (*conditioned)[i], canonical[i],
                    (*reconstructed)[i], minimum_cosine);
                admitted = result[i].equivalent;
            }
            result[i].suffix_stage = static_cast<MoESuffixStage>(i);
        }
        return result;
    }

    /** @brief One CSV schema shared by canonical and detailed sidecar artifacts. */
    inline void writeMoERouteProofCSVHeader(std::ostream &out)
    {
        out << "proof_authority,canonical_passed,route_conditioned_evaluated,"
               "route_conditioned_passed,route_conditioned_cosine,route_conditioned_rel_l2,"
               "route_conditioned_max_abs,route_weights_max_error,hf_reconstruction_rel_l2";
    }

    /** @brief Serialize explicit proof evidence without replacing original HF metrics. */
    inline void writeMoERouteProofCSV(std::ostream &out,
        const std::optional<MoERouteConditionedProof> &proof, bool passed)
    {
        if (!proof)
        {
            out << "canonical_hf," << (passed ? 1 : 0) << ",,,,,,,";
            return;
        }
        out << (proof->suffix_stage ? "route_conditioned_hf_suffix," : "route_conditioned_hf,")
            << proof->canonical_passed << ','
            << proof->evaluated << ',' << proof->equivalent << ','
            << proof->conditioned.cosine << ',' << proof->conditioned.relative_l2 << ','
            << proof->conditioned.maximum_error << ',' << proof->maximum_weight_error << ','
            << proof->canonical_reconstruction.relative_l2;
    }
}
