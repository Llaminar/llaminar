/**
 * @file ProductionParityPrefixRestorePlan.h
 * @brief Shared request geometry for the mandatory partial-prefix proof.
 *
 * Ordinary decode may archive the entire prompt-plus-token request. The proof
 * therefore purges that archive, seeds the base prompt, proves a complete hit,
 * then appends one authenticated token to prove a partial hit. Both execution
 * and demand-window admission consume this plan: the seed is real routed work,
 * even though the intervening complete hit executes no model rows.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llaminar2::test::parity
{
    /** Immutable, validated geometry; this owns no runtime or placement state. */
    class ProductionParityPrefixRestorePlan final
    {
    public:
        /**
         * @param prefix_tokens Nonempty authenticated base prompt length.
         * @throws std::invalid_argument When the extended request cannot fit
         * the production runner's positive int token-count interface.
         */
        explicit constexpr ProductionParityPrefixRestorePlan(std::size_t prefix_tokens)
            : prefix_tokens_(prefix_tokens)
        {
            if (prefix_tokens == 0u ||
                prefix_tokens >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Partial-prefix proof requires a nonempty int-sized prompt and one suffix token");
        }

        /** @return Rows recomputed after the mandatory public archive purge. */
        [[nodiscard]] constexpr int seedTokens() const noexcept
        {
            return static_cast<int>(prefix_tokens_);
        }

        /** @return Exactly the first independently certified decode row. */
        [[nodiscard]] static constexpr int suffixTokens() noexcept { return 1; }

        /** @return Logical request length at the final partial-restore boundary. */
        [[nodiscard]] constexpr int extendedTokens() const noexcept
        {
            return seedTokens() + suffixTokens();
        }

        /** @return New main-model rows: cold seed plus uncached suffix, not hits. */
        [[nodiscard]] constexpr std::uint64_t routedRows() const noexcept
        {
            return static_cast<std::uint64_t>(extendedTokens());
        }

    private:
        const std::size_t prefix_tokens_; ///< Validated base request geometry.
    };
}
