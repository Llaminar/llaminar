/**
 * @file FinalLogitMeanKLContract.h
 * @brief Explicit ordinary final-logit acceptance, independent of raw diagnostic cosines.
 *
 * Limits apply to each final-position distribution separately. They never
 * average different decode steps or change recursive MTP, exact integer or
 * request/prefix state contracts. An absent contract retains legacy gates.
 */
#pragma once
#include <cmath>
#include <cstddef>
#include <span>

namespace llaminar2::test::parity
{
/** @brief Explicit provisional ordinary-logit budgets and exact teacher trajectory length. */
struct FinalLogitMeanKLContract
{
    float prefill_maximum = 0.05f; ///< Existing canonical prefill budget, nats/position.
    float decode_maximum = 0.10f; ///< Provisional ordinary decode regression budget, nats/position.
    int decode_steps = 2; ///< Every original step must be present, exactly once.

    /** @return Whether every explicit budget and trajectory cardinality is meaningful. */
    [[nodiscard]] bool valid() const noexcept
    {
        return std::isfinite(prefill_maximum) && prefill_maximum > 0.0f &&
            std::isfinite(decode_maximum) && decode_maximum > 0.0f && decode_steps > 0;
    }

    /** @return Whether one finite, nonnegative mean KL is strictly within its phase budget. */
    [[nodiscard]] bool accepts(float mean_kl, bool prefill) const noexcept
    {
        return valid() && std::isfinite(mean_kl) && mean_kl >= 0.0f &&
            mean_kl < (prefill ? prefill_maximum : decode_maximum);
    }
};

/** @return Whether an actual nonempty floating observation has finite operands throughout. */
inline bool finiteParityOperands(std::span<const float> values) noexcept
{
    if (values.empty()) return false;
    for (float value : values) if (!std::isfinite(value)) return false;
    return true;
}

/**
 * @brief Select the final complete vocabulary row without treating capacity as positions.
 * @param data Original observed scalar root. @param elements Actual observed extent.
 * @param vocabulary Original model vocabulary, not a copied test dimension.
 * @return Final row, or empty for absent, partial-row or nonfinite evidence.
 */
inline std::span<const float> finalParityLogitRow(const float *data, std::size_t elements,
                                                 std::size_t vocabulary) noexcept
{
    if (!data || !vocabulary || elements < vocabulary || elements % vocabulary != 0) return {};
    const std::span<const float> row{data + elements - vocabulary, vocabulary};
    return finiteParityOperands(row) ? row : std::span<const float>{};
}

/**
 * @brief Bind the actual physical ID extents to the original compared routing-weight rows.
 * @param actual_weights Actual live values. @param expected_weights Reference values after the same row window.
 * @param actual_ids Actual complete ID bank. @param expected_ids Same-window reference IDs.
 * @return True only for one nonempty exact common extent, before reading any ID element.
 */
inline bool parityRouteFrameHasExactExtents(std::span<const float> actual_weights,
    std::span<const float> expected_weights, std::span<const float> actual_ids,
    std::span<const float> expected_ids) noexcept
{
    return !actual_weights.empty() && actual_weights.size() == expected_weights.size() &&
        actual_weights.size() == actual_ids.size() && actual_weights.size() == expected_ids.size();
}

/**
 * @brief Validate live normalized-probability rows without inventing a sum-to-one tolerance.
 * @param weights Complete row-major probabilities. @param top_k Original live route width.
 * @return True for finite [0,1] entries with a positive total in every live row.
 * @note Only the QFN schema's norm_topk_prob contract consumes this bound; scaled legacy models do not.
 */
inline bool validNormalizedParityWeights(std::span<const float> weights, int top_k) noexcept
{
    if (top_k <= 0 || weights.empty() || weights.size() % top_k) return false;
    for (std::size_t row = 0; row < weights.size(); row += top_k)
    {
        double total = 0.0;
        for (int index = 0; index < top_k; ++index)
        {
            const float value = weights[row + index];
            if (!std::isfinite(value) || value < 0.0f || value > 1.0f) return false;
            total += value;
        }
        if (!(total > 0.0)) return false;
    }
    return true;
}

/**
 * @brief Validate routing representation without demanding the reference ranking/membership.
 * @param values Actual or reference complete row-major expert IDs.
 * @param top_k Exact original route width. @param experts Exact original expert domain.
 * @return True only for nonempty complete rows of unique finite integral in-domain IDs.
 *
 * Expected expert membership drift is numerical evidence, not malformed data.
 * Validation precedes any float-to-int cast or sparse-vector index.
 */
inline bool validParityRouteIds(std::span<const float> values, int top_k, int experts) noexcept
{
    if (top_k <= 0 || experts < top_k || values.empty() || values.size() % top_k != 0) return false;
    for (std::size_t row = 0; row < values.size(); row += top_k)
        for (int index = 0; index < top_k; ++index)
        {
            const float value = values[row + index];
            if (!std::isfinite(value) || value < 0.0f || value >= experts || std::trunc(value) != value) return false;
            for (int prior = 0; prior < index; ++prior)
                if (value == values[row + prior]) return false;
        }
    return true;
}
} // namespace llaminar2::test::parity
