/**
 * @file GenerationPenaltyHistory.h
 * @brief Explicit history provenance for captured ordinary and speculative sampling.
 *
 * Both samplers consume the same durable histogram and penalty arithmetic.
 * Only a verifier additionally reads an uncommitted branch and a resident
 * active-row limit. Factories reject incomplete bindings before graph recording;
 * ordinary sampling never fabricates verifier inputs to reuse a fused reducer.
 */
#pragma once

#include <stdexcept>

namespace llaminar2
{
/** @brief Borrowed immutable binding, not an owner or host mirror of token counts. */
class GenerationPenaltyHistory
{
public:
    /** @brief One ordinary sampling row observes committed generated tokens only.
     *  @throws std::invalid_argument when either persistent binding is missing. */
    static GenerationPenaltyHistory committed(const void *counts, const void *policy)
    {
        return GenerationPenaltyHistory(counts, policy, nullptr, nullptr);
    }

    /** @brief Each active verifier row also observes its speculative prefix.
     *  @throws std::invalid_argument for a missing histogram, policy, branch or width. */
    static GenerationPenaltyHistory speculative(
        const void *counts, const void *policy, const void *tokens, const void *active_rows)
    {
        if (!tokens || !active_rows)
            throw std::invalid_argument("Speculative penalty history requires branch tokens and resident active rows");
        return GenerationPenaltyHistory(counts, policy, tokens, active_rows);
    }

    /** @return Whether this binding admits the requested physical row capacity. */
    bool admitsRows(int rows) const noexcept { return rows > 0 && (branch_tokens_ || rows == 1); }
    /** @return Durable device INT32 histogram shared by this row/branch. */
    const void *counts() const noexcept { return counts_; }
    /** @return Device MTPGreedyPenaltyPolicy; arithmetic is common to both histories. */
    const void *policy() const noexcept { return policy_; }
    /** @return Verifier INT32 input tokens, or null for explicitly committed history. */
    const void *branchTokens() const noexcept { return branch_tokens_; }
    /** @return Resident verifier width, or null for the single ordinary row. */
    const void *activeRows() const noexcept { return active_rows_; }

private:
    /** @brief Only named factories may select a complete history provenance. */
    GenerationPenaltyHistory(const void *counts, const void *policy,
        const void *tokens, const void *active_rows)
        : counts_(counts), policy_(policy), branch_tokens_(tokens), active_rows_(active_rows)
    {
        if (!counts || !policy)
            throw std::invalid_argument("Generation penalty history requires resident counts and policy");
    }
    const void *counts_; ///< Request-owned durable counts; immutable address, mutable device bytes.
    const void *policy_; ///< Admitted penalty policy, published before captured consumers.
    const void *branch_tokens_; ///< Null only for the explicit committed-history mode.
    const void *active_rows_; ///< Paired with branch_tokens_; never optional for a verifier.
};
} // namespace llaminar2
