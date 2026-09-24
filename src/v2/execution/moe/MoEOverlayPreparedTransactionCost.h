/**
 * @file MoEOverlayPreparedTransactionCost.h
 * @brief Exact, reusable CPU pricing of one immutable routed transaction.
 *
 * A background swap search prices thousands of placements against the same
 * routed rows. Count each expert once per transaction before that search, then
 * multiply its integer activation count by the candidate owner's integer price.
 * This preserves the raw scorer's exact arithmetic and overflow rejection.
 * Never merge distinct transactions: their participant maxima must still be
 * summed separately. Identical sufficient statistics may share one priced
 * result with an exact occurrence multiplier. This is proposal-local scratch,
 * not another demand authority.
 */
#pragma once

#include "MoEOverlayTransactionCost.h"

#include <algorithm>
#include <compare>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace llaminar2::moe_overlay_economy
{
    /**
     * @brief Immutable sufficient statistic for one transaction's service cost.
     *
     * Preparation owns only nonzero per-expert integer counts. Owner maps and
     * service prices are supplied afresh for every candidate, so a provisional
     * swap cannot leave stale cached ownership or economic policy behind.
     */
    class PreparedTransactionCost final
    {
    public:
        /**
         * @brief Authenticate and compact exactly one real routed invocation.
         * @param routes Logical rows and stride; padding is intentionally ignored.
         * @param expert_count Complete model expert-ID range.
         * @throws std::invalid_argument for malformed geometry or expert IDs.
         */
        PreparedTransactionCost(TransactionRoutes routes, std::uint32_t expert_count)
            : expert_count_(expert_count)
        {
            if (!routes.expert_ids || expert_count == 0 || routes.logical_rows == 0 ||
                routes.top_k == 0 || routes.top_k > expert_count ||
                routes.row_stride < routes.top_k ||
                static_cast<std::uint64_t>(routes.logical_rows - 1u) * routes.row_stride +
                        routes.top_k > routes.capacity_slots)
                throw std::invalid_argument("Prepared expert transaction has invalid route geometry");

            std::vector<std::uint64_t> counts(expert_count, 0);
            for (std::uint32_t row = 0; row < routes.logical_rows; ++row)
                for (std::uint32_t slot = 0; slot < routes.top_k; ++slot)
                {
                    const auto expert = routes.expert_ids[
                        static_cast<std::uint64_t>(row) * routes.row_stride + slot];
                    if (expert < 0 || static_cast<std::uint32_t>(expert) >= expert_count)
                        throw std::invalid_argument("Prepared expert transaction has an invalid expert ID");
                    // At most uint32(rows) * uint32(top_k) routes exist, so a
                    // uint64 count cannot overflow even for repeated expert IDs.
                    ++counts[static_cast<std::size_t>(expert)];
                }
            const auto nonzero = std::count_if(counts.begin(), counts.end(),
                [](std::uint64_t count) { return count != 0; });
            counts_.reserve(static_cast<std::size_t>(nonzero));
            for (std::uint32_t expert = 0; expert < expert_count; ++expert)
                if (counts[expert] != 0)
                    counts_.push_back({expert, counts[expert]});
        }

        /**
         * @brief Compare immutable sufficient statistics, not current placements.
         *
         * Equality means both transactions have the same cost under every
         * possible owner map and price table in one phase. It does not mean
         * that two different transactions with equal window marginals can be
         * merged. Ordering permits bounded O(log unique-transactions) grouping
         * without a hash or a scan through all previous observations.
         */
        [[nodiscard]] auto operator<=>(const PreparedTransactionCost &) const = default;

        /**
         * @brief Price a candidate without walking its original token rows again.
         * @param placement Current/candidate owners and positive endpoint prices.
         * @param scratch Exclusive participant scratch, overwritten on entry.
         * @param scratch_count Available entries, not a fixed topology bound.
         * @return Exact parallel maxima, or a fatal typed validation/overflow status.
         *
         * Integer addition of nonnegative work is associative until overflow.
         * Widen multiplication and reject overflow before addition, retaining the
         * raw scorer's failure semantics rather than saturating a cost estimate.
         */
        [[nodiscard]] TransactionCostResult score(
            TransactionPlacementCosts placement, ServiceCostPair *scratch,
            std::uint32_t scratch_count) const noexcept
        {
            if (!placement.before_owners || !placement.after_owners ||
                !placement.participant_ns_per_activation || !scratch ||
                placement.expert_count != expert_count_ || placement.participant_count == 0 ||
                scratch_count < placement.participant_count)
                return {TransactionCostStatus::InvalidGeometry, {}};
            for (std::uint32_t participant = 0; participant < placement.participant_count;
                 ++participant)
            {
                if (placement.participant_ns_per_activation[participant] == 0)
                    return {TransactionCostStatus::UnpricedParticipant, {}};
                scratch[participant] = {};
            }

            constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
            for (const auto &entry : counts_)
            {
                const auto before = placement.before_owners[entry.expert];
                const auto after = placement.after_owners[entry.expert];
                if (before < 0 || after < 0 ||
                    static_cast<std::uint32_t>(before) >= placement.participant_count ||
                    static_cast<std::uint32_t>(after) >= placement.participant_count)
                    return {TransactionCostStatus::InvalidOwner, {}};
                const auto before_work = static_cast<unsigned __int128>(entry.activations) *
                    placement.participant_ns_per_activation[before];
                const auto after_work = static_cast<unsigned __int128>(entry.activations) *
                    placement.participant_ns_per_activation[after];
                if (before_work > maximum - scratch[before].before_ns ||
                    after_work > maximum - scratch[after].after_ns)
                    return {TransactionCostStatus::Overflow, {}};
                scratch[before].before_ns += static_cast<std::uint64_t>(before_work);
                scratch[after].after_ns += static_cast<std::uint64_t>(after_work);
            }

            ServiceCostPair critical;
            for (std::uint32_t participant = 0; participant < placement.participant_count;
                 ++participant)
            {
                critical.before_ns = std::max(critical.before_ns, scratch[participant].before_ns);
                critical.after_ns = std::max(critical.after_ns, scratch[participant].after_ns);
            }
            return {TransactionCostStatus::Complete, critical};
        }

    private:
        friend class PreparedTransactionSwapSearch;
        /** @brief One nonzero count; ordered by expert ID, never by a candidate owner. */
        struct ExpertCount
        {
            std::uint32_t expert;
            std::uint64_t activations;
            /** @brief Exact lexicographic count identity, including expert ID. */
            auto operator<=>(const ExpertCount &) const = default;
        };
        std::uint32_t expert_count_;
        std::vector<ExpertCount> counts_;
    };

    /**
     * @brief Immutable, exact pricing of swaps from one complete source layout.
     *
     * A search considers thousands of two-expert exchanges. All other expert
     * contributions are identical, so price the source layout once and replace
     * only the two changed contributions. This is derived proposal-local work,
     * not a live ownership mirror: it owns its inputs and has no update method.
     * An accepted swap requires constructing a new search from the new layout.
     */
    class PreparedTransactionSwapSearch final
    {
    public:
        /**
         * @brief Bind one invocation to its immutable source layout and prices.
         * @param transaction Authenticated per-expert activation counts.
         * @param owners Complete expert-to-participant map for this search.
         * @param prices Positive service prices for every physical participant.
         * @param selected Tier participants to price; empty selects all of them.
         * @throws std::invalid_argument For invalid ownership, prices or selection.
         * @throws std::overflow_error If the source layout cannot be priced exactly.
         *
         * Every participant is checked even when only one tier is selected.
         * Selection changes the reported maximum, not the validity of work
         * outside that tier. No caller-owned span survives construction.
         */
        PreparedTransactionSwapSearch(
            const PreparedTransactionCost &transaction,
            std::span<const std::int32_t> owners,
            std::span<const std::uint64_t> prices,
            std::span<const int> selected = {})
            : experts_(owners.size()), participants_(prices.size())
        {
            if (owners.size() != transaction.expert_count_ || prices.empty() ||
                prices.size() > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("Prepared expert swap has invalid geometry");
            for (std::size_t participant = 0; participant < prices.size(); ++participant)
            {
                if (prices[participant] == 0)
                    throw std::invalid_argument("Prepared expert swap has an unpriced participant");
                participants_[participant] = {prices[participant], 0, selected.empty()};
            }
            for (const int participant : selected)
            {
                if (participant < 0 || static_cast<std::size_t>(participant) >= prices.size() ||
                    participants_[participant].selected)
                    throw std::invalid_argument("Prepared expert swap has an invalid tier selection");
                participants_[participant].selected = true;
            }
            for (std::size_t expert = 0; expert < owners.size(); ++expert)
            {
                if (owners[expert] < 0 || static_cast<std::size_t>(owners[expert]) >= prices.size())
                    throw std::invalid_argument("Prepared expert swap has an invalid owner");
                experts_[expert] = {0, static_cast<std::uint32_t>(owners[expert])};
            }
            constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
            for (const auto &entry : transaction.counts_)
            {
                auto &expert = experts_[entry.expert];
                expert.activations = entry.activations;
                auto &participant = participants_[expert.owner];
                const auto work = static_cast<unsigned __int128>(entry.activations) * participant.price;
                if (work > maximum - participant.before)
                    throw std::overflow_error("Prepared expert swap source service overflows");
                participant.before += static_cast<std::uint64_t>(work);
            }
            for (const auto &participant : participants_)
                if (participant.selected)
                {
                    before_maximum_ = std::max(before_maximum_, participant.before);
                    before_minimum_ = std::min(before_minimum_, participant.before);
                }
        }

        /** @return Selected-tier source minimum for the unchanged imbalance gate. */
        [[nodiscard]] std::uint64_t minimumBefore() const noexcept { return before_minimum_; }

        /**
         * @brief Price exchanging two expert owners without mutating the search.
         * @param first First logical expert ID.
         * @param second Second logical expert ID, possibly equal to the first.
         * @return Exact selected-tier before/after maxima or a typed fatal status.
         *
         * Subtract each outgoing contribution before adding the incoming one.
         * The source contribution is proven representable by construction; the
         * replacement is widened and checked even for an unselected participant.
         * This preserves full-map scorer overflow semantics without walking all
         * unchanged experts or materializing an after-owner vector per candidate.
         */
        [[nodiscard]] TransactionCostResult scoreSwap(
            std::uint32_t first, std::uint32_t second) const noexcept
        {
            if (first >= experts_.size() || second >= experts_.size())
                return {TransactionCostStatus::InvalidExpert, {}};
            const auto &a = experts_[first];
            const auto &b = experts_[second];
            if (a.owner == b.owner || a.activations == b.activations)
                return {TransactionCostStatus::Complete, {before_maximum_, before_maximum_}};

            const auto &source_a = participants_[a.owner];
            const auto &source_b = participants_[b.owner];
            const auto after_a = source_a.before - a.activations * source_a.price +
                static_cast<unsigned __int128>(b.activations) * source_a.price;
            const auto after_b = source_b.before - b.activations * source_b.price +
                static_cast<unsigned __int128>(a.activations) * source_b.price;
            constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
            if (after_a > maximum || after_b > maximum)
                return {TransactionCostStatus::Overflow, {}};

            std::uint64_t after_maximum = 0;
            for (std::size_t participant = 0; participant < participants_.size(); ++participant)
                if (participants_[participant].selected)
                    after_maximum = std::max(after_maximum,
                        participant == a.owner ? static_cast<std::uint64_t>(after_a) :
                        participant == b.owner ? static_cast<std::uint64_t>(after_b) :
                                                 participants_[participant].before);
            return {TransactionCostStatus::Complete, {before_maximum_, after_maximum}};
        }

    private:
        /** @brief Source owner and exact count for one expert, including zero demand. */
        struct Expert { std::uint64_t activations; std::uint32_t owner; };
        /** @brief One physical endpoint's immutable source work and service price. */
        struct Participant { std::uint64_t price; std::uint64_t before; bool selected; };
        std::vector<Expert> experts_;
        std::vector<Participant> participants_;
        std::uint64_t before_maximum_ = 0;
        std::uint64_t before_minimum_ = std::numeric_limits<std::uint64_t>::max();
    };

    /**
     * @brief Price repeated identical transactions without merging their rows.
     * @param transaction One already priced transaction's participant maxima.
     * @param occurrences Positive count of identical phase-local invocations.
     * @return Sequential sum, or fatal overflow/validation status.
     *
     * Multiplication follows the maximum, not the other way around. Every
     * original invocation therefore retains its independent critical path.
     * This is valid only after exact sufficient-statistic equality and phase
     * identity have been established by the caller's immutable grouping.
     */
    [[nodiscard]] inline TransactionCostResult repeatPreparedTransaction(
        TransactionCostResult transaction, std::uint64_t occurrences) noexcept
    {
        if (!transaction.complete())
            return transaction;
        if (occurrences == 0)
            return {TransactionCostStatus::InvalidGeometry, {}};
        constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
        const auto before = static_cast<unsigned __int128>(transaction.cost.before_ns) * occurrences;
        const auto after = static_cast<unsigned __int128>(transaction.cost.after_ns) * occurrences;
        if (before > maximum || after > maximum)
            return {TransactionCostStatus::Overflow, {}};
        return {TransactionCostStatus::Complete,
                {static_cast<std::uint64_t>(before), static_cast<std::uint64_t>(after)}};
    }
}
