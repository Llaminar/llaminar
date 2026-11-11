/**
 * @file Test__CPUDraftVocabularyBudget.cpp
 * @brief Portable half of the draft-vocabulary-budget diagnostic (idea 16
 *        B4): typed policy, selector v1 lifecycle, and the coupling
 *        test properties on the production sampler helpers.
 *
 * Non-installed diagnostic namespace; no production routing, sampler,
 * verifier, CLI/config, PerfStats production family, PMA admission or
 * graph change is implied or granted. This translation unit is portable
 * across architectures; the NEON head byte proof lives in
 * Test__CPUDraftVocabularyBudgetHead.cpp under the architecture
 * registration boundary.
 *
 * Reviewer-findings rounds (209/210, then 217 A1-A5) repaired here:
 * - A1/D1: the selector is the FINAL named committed-token recency
 *   selector (CommittedTokenRecencySelector, version 1, matching spec
 *   §3); observations carry VERIFIED RECEIPTS validated against the
 *   commit frontier and the owning request epoch (stale positions,
 *   foreign epochs, duplicates, out-of-range tokens: all rejected); the
 *   shortlist DEDUPLICATES to freshest observations, orders by commit
 *   position descending, CAPS to N, and backfills unranked live IDs in
 *   ascending global-ID order; the cold-start fallback is an explicitly
 *   ADMITTED policy; reset rebinds the request epoch; the tests assert
 *   every rejection.
 * - D2/A3: the Mode P replay calls the PRODUCTION
 *   sampleMTPRejectionRowFromDistributions (independent residual
 *   threshold); the EXECUTED sampler is enumerated exactly over dyadic
 *   threshold bins with FULLY DYADIC fixtures (the finite-precision
 *   contract for non-dyadic PMFs is out of scope, claimed nowhere); the
 *   wrong-CORRECTION negative keeps p and the proposal draw fixed and
 *   supplies a different correction distribution to the production
 *   helper — the executed law deviates and is DETECTED; the
 *   invalid-PMF/under-coverage negative is rejected by the draw-side
 *   PMF validation; the threshold-endpoint fixture is restored.
 * - D5/A5 boundaries (explicit): Mode C replays the production helper
 *   with SUPPLIED thresholds (no seed-to-threshold addressing, no
 *   connected multi-N selector/head path — production wiring, out of
 *   scope); Mode G is the diagnostic's own harness implementation of the
 *   Mode G contract (no production helper call; production wiring
 *   untouched) with the tie-break at the LOWEST global token ID and an
 *   authored independent target stream.
 * - S1 (hard): the seeded target tokens are identical across budgets —
 *   asserted through the production helper with per-position thresholds.
 * - S6: nothing here grants implementation, N-sweep execution or economy
 *   admission; the sweep-mechanics check constructs runtime descriptors
 *   only.
 *
 * What remains UNPROVED here (claimed nowhere): counted device traffic, a
 * real multi-N device sweep, the 63/65/127/129/V-1 byte sweeps, partial-window
 * production replay, and the normalized-logit draw on the real head —
 * the NEON head TU covers the byte-proof side within its stated scope.
 */

#include <gtest/gtest.h>

#include "execution/mtp/MTPRejectionSampler.h"
#include "utils/Sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace b4_diagnostic
{
    // =====================================================================
    // Typed policy descriptors (runtime data; one binary; no per-N build).
    // =====================================================================

    enum class DrawLaw : std::uint8_t
    {
        DeltaOnShortlist,    ///< q = delta_a with a = declared winner on S.
        NormalizedShortlist, ///< qS = normalized logits restricted to S.
    };

    enum class CouplingKind : std::uint8_t
    {
        GreedyTargetMismatch,
        CpuSeededSerialTargetDraw,
        ProbabilityRejection,
    };

    struct VocabBudget
    {
        enum class Kind : std::uint8_t { FullVocabulary, ShortlistLimit };
        Kind kind = Kind::FullVocabulary;
        std::uint32_t n = 0; ///< Live only for ShortlistLimit.

        static VocabBudget full() { return {}; }
        static VocabBudget shortlist(std::uint32_t checked_n)
        {
            return {Kind::ShortlistLimit, checked_n};
        }
    };

    enum class SelectorFallback : std::uint8_t
    {
        None,             ///< Empty ledger is an error (no rescue).
        LowestGlobalIds,  ///< Admitted deterministic cold start.
    };

    struct DraftProposalPolicy
    {
        std::uint32_t policy_version = 1;
        std::uint32_t selector_version = 1;
        SelectorFallback selector_fallback = SelectorFallback::None;
        DrawLaw draw_law = DrawLaw::DeltaOnShortlist;
        VocabBudget vocab_budget;
    };

    struct VerifierCouplingPolicy
    {
        std::uint32_t coupling_version = 1;
        CouplingKind kind = CouplingKind::GreedyTargetMismatch;
    };

    /** @brief Complete identity string for config/results/capture records. */
    inline std::string identityOf(const DraftProposalPolicy &policy,
        const VerifierCouplingPolicy &coupling, std::uint32_t vocab)
    {
        const char *budget = policy.vocab_budget.kind ==
                VocabBudget::Kind::FullVocabulary
            ? "full"
            : "shortlist";
        const char *law = policy.draw_law == DrawLaw::DeltaOnShortlist
            ? "delta"
            : "normalized";
        const char *fallback =
            policy.selector_fallback == SelectorFallback::LowestGlobalIds
            ? "lowest_global_ids"
            : "none";
        const char *coupling_name =
            coupling.kind == CouplingKind::GreedyTargetMismatch
            ? "greedy"
            : (coupling.kind == CouplingKind::CpuSeededSerialTargetDraw
                ? "cpu_seeded"
                : "probability_rejection");
        return "policy_v=" + std::to_string(policy.policy_version) +
            ";selector_v=" + std::to_string(policy.selector_version) +
            ";selector_fallback=" + fallback +
            ";draw_law=" + law + ";budget=" + budget +
            ";n=" + std::to_string(policy.vocab_budget.n) +
            ";coupling_v=" + std::to_string(coupling.coupling_version) +
            ";coupling=" + coupling_name + ";vocab=" +
            std::to_string(vocab);
    }

    /** @brief Validate a budget against the vocabulary; spec negatives. */
    inline void validateBudget(const VocabBudget &budget, std::uint32_t vocab)
    {
        if (budget.kind == VocabBudget::Kind::FullVocabulary)
            return;
        if (budget.n == 0)
            throw std::invalid_argument("B4 budget N=0 is out of range");
        if (budget.n >= vocab)
            throw std::invalid_argument(
                "B4 budget N>=V is out of range; no silent full aliasing");
    }

    /** @brief Validate a candidate set: unique, in-range, nonempty. */
    inline void validateShortlist(const std::vector<std::uint32_t> &ids,
        std::uint32_t vocab, const VocabBudget &budget)
    {
        if (budget.kind == VocabBudget::Kind::FullVocabulary)
            throw std::invalid_argument(
                "shortlist construction is not applicable to full");
        if (ids.empty())
            throw std::invalid_argument(
                "B4 empty legal support fails; no error-rescue");
        if (ids.size() > budget.n)
            throw std::invalid_argument(
                "B4 live |S| exceeds the checked budget");
        std::set<std::uint32_t> unique;
        for (const std::uint32_t id : ids)
        {
            if (id >= vocab)
                throw std::invalid_argument(
                    "B4 candidate ID is out of the live vocabulary");
            if (!unique.insert(id).second)
                throw std::invalid_argument(
                    "B4 candidate IDs must be unique");
        }
    }

    // =====================================================================
    // Selector v1: prior verified target ranking with verified receipts.
    // =====================================================================

    /** @brief One verified observation's immutable receipt. */
    struct VerifiedTargetReceipt
    {
        std::uint32_t source_row;    ///< Global vocabulary row (the token).
        std::uint64_t position;      ///< Logical position within the request.
        std::uint64_t generation;    ///< Request generation (monotonic).
        std::uint32_t rank;          ///< Verified rank at that position.
    };

    /**
     * @brief Selector v1: COMMITTED-TOKEN RECENCY (spec §3) over verified
     *        target receipts.
     *
     * The candidate set is the recency set of COMMITTED target tokens.
     * Observations are APPENDED and validated: a receipt whose position is
     * below the commit frontier, whose request generation differs from the
     * owning epoch, or that duplicates the frontier position is rejected —
     * there is no silent merge. The cold-start fallback is an explicitly
     * ADMITTED selector policy (SelectorFallback::LowestGlobalIds), never
     * an error rescue. Shortlist construction: deduplicate to each token's
     * freshest observation, order by commit position descending (the
     * request generation serves identity/staleness only, never ordering),
     * then lowest global token ID; take the first N distinct IDs; backfill
     * the remaining budget with unranked live IDs in ascending global-ID
     * order; |S| = min(N, live vocabulary), reported by the caller. The
     * selector's only inputs are verified receipts and the vocabulary
     * bound — no current-logit or full-head input exists in the interface,
     * so a lookahead selector is unrepresentable here.
     */
    class CommittedTokenRecencySelector
    {
    public:
        static constexpr std::uint32_t kVersion = 1;

        explicit CommittedTokenRecencySelector(std::uint32_t vocab,
            SelectorFallback fallback)
            : vocab_(vocab), fallback_(fallback)
        {
            if (vocab == 0)
                throw std::invalid_argument("selector vocabulary is empty");
        }

        /** @brief Reset/prefix: the ledger clears; the epoch rebinds. */
        void reset(std::uint64_t request_epoch)
        {
            ledger_.clear();
            commit_frontier_ = 0;
            has_frontier_ = false;
            request_epoch_ = request_epoch;
        }

        /**
         * @brief Append one verified receipt; stale receipts are rejected.
         *
         * A receipt is stale when its logical position or generation is
         * below the ledger's high-water mark (evidence from an earlier
         * request generation or an already-consumed position). Equal
         * positions from the same generation are rejected as duplicates.
         */
        void observeVerifiedTarget(const VerifiedTargetReceipt &receipt)
        {
            if (receipt.source_row >= vocab_)
                throw std::invalid_argument(
                    "selector observed an out-of-range token");
            // The receipt must carry the selector's owning request epoch.
            if (receipt.generation != request_epoch_)
                throw std::invalid_argument(
                    "stale selector receipt: foreign request epoch");
            if (has_frontier_ && receipt.position < commit_frontier_)
                throw std::invalid_argument(
                    "stale selector receipt: position below the commit "
                    "frontier");
            if (has_frontier_ && receipt.position == commit_frontier_)
                throw std::invalid_argument(
                    "duplicate selector receipt at the commit frontier");
            ledger_.push_back(receipt);
            commit_frontier_ = receipt.position;
            has_frontier_ = true;
        }

        /**
         * @brief Build the shortlist for the next draft position.
         *
         * Deduplicate the ledger to each token's freshest observation,
         * order by commit position descending (then lowest global token
         * ID), take the first N distinct IDs, and backfill the remaining
         * budget with unranked live IDs in ascending global-ID order;
         * |S| = min(N, live vocabulary). When the admitted fallback is
         * configured and the ledger is empty, the deterministic cold start
         * is the lowest global IDs; with no admitted fallback an empty
         * ledger is an error (no rescue).
         */
        std::vector<std::uint32_t> shortlist(const VocabBudget &budget) const
        {
            if (budget.kind == VocabBudget::Kind::FullVocabulary)
                throw std::invalid_argument(
                    "selector shortlist is not applicable to full");
            validateBudget(budget, vocab_);
            std::map<std::uint32_t, const VerifiedTargetReceipt *> freshest;
            for (const VerifiedTargetReceipt &receipt : ledger_)
            {
                auto it = freshest.find(receipt.source_row);
                if (it == freshest.end() ||
                    it->second->position < receipt.position)
                    freshest[receipt.source_row] = &receipt;
            }
            std::vector<const VerifiedTargetReceipt *> ordered;
            ordered.reserve(freshest.size());
            for (const auto &entry : freshest)
                ordered.push_back(entry.second);
            std::sort(ordered.begin(), ordered.end(),
                [](const VerifiedTargetReceipt *a,
                    const VerifiedTargetReceipt *b) {
                    if (a->position != b->position)
                        return a->position > b->position;
                    return a->source_row < b->source_row;
                });
            std::vector<std::uint32_t> ids;
            for (const VerifiedTargetReceipt *receipt : ordered)
            {
                if (ids.size() >= budget.n)
                    break;
                ids.push_back(receipt->source_row);
            }
            if (ids.size() < budget.n)
            {
                if (fallback_ != SelectorFallback::LowestGlobalIds)
                    throw std::invalid_argument(
                        "selector ledger exhausted without an admitted "
                        "cold-start fallback");
                for (std::uint32_t id = 0;
                    id < vocab_ && ids.size() < budget.n; ++id)
                    if (std::find(ids.begin(), ids.end(), id) == ids.end())
                        ids.push_back(id);
            }
            return ids;
        }

                std::uint32_t version() const { return kVersion; }

    private:
        std::uint32_t vocab_ = 0;
        SelectorFallback fallback_ = SelectorFallback::None;
        std::vector<VerifiedTargetReceipt> ledger_;
        std::uint64_t commit_frontier_ = 0;
        std::uint64_t request_epoch_ = 0;
        bool has_frontier_ = false;
    };

    // =====================================================================
    // Mode G: greedy target mismatch correction (tiny-vocab harness).
    // =====================================================================

    /** @brief One recorded draft/verify step of the coupling ledger. */
    struct LedgerEntry
    {
        std::uint32_t position;
        std::uint32_t proposal;
        std::uint32_t target_token;
        bool accepted;
    };

    /**
     * @brief Greedy target mismatch correction over an authored fixture.
     *
     * The emitted token is ALWAYS the target's greedy token from the
     * independent authored stream (after the existing deterministic
     * transform: argmax with ties to the lowest global token ID — the
     * target stream is an input, never generated here). The draft
     * proposal is the lowest-global-ID argmax over the shortlisted draft
     * logits; equal logits therefore propose the lowest global ID.
     */
    inline std::vector<LedgerEntry> runGreedyCoupling(
        const std::vector<std::uint32_t> &authored_target_greedy,
        const std::vector<std::vector<float>> &draft_logits_per_position,
        const std::vector<std::vector<std::uint32_t>>
            &shortlists_per_position)
    {
        std::vector<LedgerEntry> ledger;
        for (std::size_t position = 0;
            position < authored_target_greedy.size(); ++position)
        {
            const std::uint32_t target_token =
                authored_target_greedy[position];
            const auto &logits = draft_logits_per_position.at(position);
            const auto &shortlist = shortlists_per_position.at(position);
            if (shortlist.empty())
                throw std::invalid_argument(
                    "Mode G draft had empty legal support");
            std::uint32_t proposal = shortlist.front();
            float draft_best = -std::numeric_limits<float>::infinity();
            for (const std::uint32_t id : shortlist)
            {
                const float value = logits.at(id);
                if (value > draft_best ||
                    (value == draft_best && id < proposal))
                {
                    draft_best = value;
                    proposal = id;
                }
            }
            ledger.push_back({static_cast<std::uint32_t>(position), proposal,
                target_token, proposal == target_token});
        }
        return ledger;
    }

    // =====================================================================
    // Mode C: the production seeded serial-target-draw coupling.
    // =====================================================================

    /**
     * @brief Replay one Mode C position through the PRODUCTION helper.
     *
     * `sampleMTPSerialEquivalentTargetRow` draws the target token from the
     * target distribution at the supplied (position-keyed) threshold and
     * accepts iff the drawn token equals the draft proposal. The threshold
     * table is keyed by logical position only — never by N, shortlist
     * rank, depth or rejection count — which is the S1 contract the test
     * asserts.
     */
    inline llaminar2::MTPRejectionSampleRowResult runSeededPosition(
        const std::vector<llaminar2::SamplingDistributionEntry>
            &target_distribution,
        std::int32_t draft_token, float position_threshold)
    {
        return llaminar2::sampleMTPSerialEquivalentTargetRow(
            target_distribution, draft_token, position_threshold);
    }
} // namespace b4_diagnostic

// ===========================================================================
// Tests (portable).
// ===========================================================================

TEST(CPUDraftVocabularyBudget, PolicyValidatorsRejectSpecNegatives)
{
    const std::uint32_t vocab = 96;
    EXPECT_THROW(b4_diagnostic::validateBudget(
                     b4_diagnostic::VocabBudget::shortlist(0), vocab),
        std::invalid_argument);
    // The V boundary and beyond: no silent full aliasing.
    for (const std::uint32_t over : {96u, 97u, 128u, 512u})
        EXPECT_THROW(b4_diagnostic::validateBudget(
                         b4_diagnostic::VocabBudget::shortlist(over), vocab),
            std::invalid_argument)
            << "N=" << over;
    EXPECT_NO_THROW(b4_diagnostic::validateBudget(
        b4_diagnostic::VocabBudget::full(), vocab));

    const auto budget = b4_diagnostic::VocabBudget::shortlist(4);
    EXPECT_THROW(b4_diagnostic::validateShortlist({}, vocab, budget),
        std::invalid_argument);
    EXPECT_THROW(b4_diagnostic::validateShortlist({1, 2, 3, vocab}, vocab,
                     budget),
        std::invalid_argument);
    EXPECT_THROW(b4_diagnostic::validateShortlist({1, 2, 2, 3}, vocab,
                     budget),
        std::invalid_argument);
    EXPECT_THROW(b4_diagnostic::validateShortlist({0, 1, 2, 3, 4}, vocab,
                     budget),
        std::invalid_argument);
    EXPECT_NO_THROW(
        b4_diagnostic::validateShortlist({0, 5, 9, 95}, vocab, budget));
}

TEST(CPUDraftVocabularyBudget, SelectorLifetimeIsEnforcedWithReceipts)
{
    const b4_diagnostic::DraftProposalPolicy policy{1,
        b4_diagnostic::CommittedTokenRecencySelector::kVersion,
        b4_diagnostic::SelectorFallback::LowestGlobalIds,
        b4_diagnostic::DrawLaw::DeltaOnShortlist,
        b4_diagnostic::VocabBudget::shortlist(4)};
    b4_diagnostic::CommittedTokenRecencySelector selector(64,
        policy.selector_fallback);
    selector.reset(/*request_epoch=*/7);
    const auto &budget = policy.vocab_budget;

    // First proposal: empty ledger + ADMITTED fallback -> cold start.
    auto first = selector.shortlist(budget);
    EXPECT_EQ((std::vector<std::uint32_t>{0, 1, 2, 3}), first);

    // Verified receipts append with position validation; every receipt
    // carries the owning request epoch (7). Positions advance
    // monotonically within the request.
    selector.observeVerifiedTarget({50, /*position=*/0, /*generation=*/7, 0});
    selector.observeVerifiedTarget({17, 1, 7, 0});
    selector.observeVerifiedTarget({3, 2, 7, 0});
    selector.observeVerifiedTarget({41, 3, 7, 0});
    auto ranked = selector.shortlist(budget);
    // Freshness ordering: commit position descending.
    EXPECT_EQ((std::vector<std::uint32_t>{41, 3, 17, 50}), ranked);

    // A foreign request epoch is rejected (stale generation control).
    EXPECT_THROW(selector.observeVerifiedTarget({7, 4, /*generation=*/8, 0}),
        std::invalid_argument);
    // A stale receipt (position below the commit frontier) is rejected.
    EXPECT_THROW(selector.observeVerifiedTarget({7, 2, 7, 0}),
        std::invalid_argument);
    // A duplicate receipt at the commit frontier is rejected.
    EXPECT_THROW(selector.observeVerifiedTarget({7, 3, 7, 0}),
        std::invalid_argument);
    // An out-of-range token is rejected.
    EXPECT_THROW(selector.observeVerifiedTarget({64, 4, 7, 0}),
        std::invalid_argument);

    // A fresh receipt (position past the frontier, same epoch) commits
    // and its token leads the freshness order.
    selector.observeVerifiedTarget({3, 4, 7, 0});
    auto third = selector.shortlist(budget);
    EXPECT_EQ(3u, third.front());

    // Backfill: with budget N=6 the two unranked live IDs backfill in
    // ascending global-ID order after the ranked prefix.
    const auto wide = b4_diagnostic::VocabBudget::shortlist(6);
    auto backfilled = selector.shortlist(wide);
    EXPECT_EQ((std::vector<std::uint32_t>{3, 41, 17, 50, 0, 1}), backfilled);

    // Reset/prefix: the ledger clears, the epoch rebinds, the fallback
    // state returns.
    selector.reset(/*request_epoch=*/9);
    auto after_reset = selector.shortlist(budget);
    EXPECT_EQ((std::vector<std::uint32_t>{0, 1, 2, 3}), after_reset);
    // A receipt from the OLD epoch is now foreign and rejected.
    EXPECT_THROW(selector.observeVerifiedTarget({5, 5, 7, 0}),
        std::invalid_argument);

    // Without the admitted fallback an empty ledger is an error.
    b4_diagnostic::CommittedTokenRecencySelector strict(64,
        b4_diagnostic::SelectorFallback::None);
    EXPECT_THROW(strict.shortlist(budget), std::invalid_argument);
}

TEST(CPUDraftVocabularyBudget, GreedyCouplingPreservesTargetStreamAtEveryN)
{
    // Independent authored fixture (labelled): tiny vocabulary of 7
    // tokens, token 6 is EOS. The target stream includes a
    // winner-absent-from-S case, a tie and an EOS position. The expected
    // stream below is authored, not generated by the harness.
    const std::vector<std::uint32_t> authored_target_stream = {3, 2, 6, 2, 5,
        6};
    const std::vector<std::vector<float>> draft_logits = {
        {0.1f, 0.2f, 0.3f, 3.0f, 0.1f, 0.0f, 0.0f},   // winner 3.
        {1.0f, 1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f},   // winner 2.
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f},   // EOS.
        {2.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f},   // tie 0/2 -> 2? the
                                                      // draft tie-break is
                                                      // lowest global ID
                                                      // over S.
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 9.0f, 0.0f},   // winner 5.
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.0f},   // EOS again.
    };

    // Shortlists per budget; N=2 deliberately misses winners so the
    // correction path is exercised.
    const std::vector<std::vector<std::vector<std::uint32_t>>>
        shortlists_per_n = {
            // N=2.
            {{3, 4}, {1, 2}, {5, 6}, {0, 2}, {4, 5}, {5, 6}},
            // N=4.
            {{1, 2, 3, 4}, {0, 1, 2, 3}, {2, 3, 5, 6}, {0, 1, 2, 5},
                {3, 4, 5, 6}, {2, 4, 5, 6}},
            // N=6 (V-1).
            {{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 6},
                {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 6}, {0, 1, 2, 3, 4, 5}},
        };

    for (const auto &shortlists : shortlists_per_n)
    {
        const auto ledger = b4_diagnostic::runGreedyCoupling(
            authored_target_stream, draft_logits, shortlists);
        ASSERT_EQ(authored_target_stream.size(), ledger.size());
        for (std::size_t position = 0; position < ledger.size(); ++position)
        {
            // S2 hard clause: the emitted stream is the target's greedy
            // stream (authored), independent of the budget.
            EXPECT_EQ(authored_target_stream[position],
                ledger[position].target_token)
                << "position " << position;
        }
    }
    // The N=2 shortlist produces corrections (the correction path runs).
    const auto any_n2 = b4_diagnostic::runGreedyCoupling(
        authored_target_stream, draft_logits, shortlists_per_n.front());
    bool any_correction = false;
    for (const auto &entry : any_n2)
        any_correction = any_correction || !entry.accepted;
    EXPECT_TRUE(any_correction) << "N=2 must produce corrections";

    // Tie-break: S = {2, 0} with equal logits proposes the LOWEST global
    // ID (0), not the first shortlist entry (2).
    const auto tie_ledger = b4_diagnostic::runGreedyCoupling({2},
        std::vector<std::vector<float>>{{2.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f,
            0.0f}},
        std::vector<std::vector<std::uint32_t>>{{2, 0}});
    EXPECT_EQ(0u, tie_ledger.front().proposal);
}

TEST(CPUDraftVocabularyBudget, SeededCouplingPreservesTargetTokensAtEveryN)
{
    // S1 hard property, replayed through the PRODUCTION coupling helper:
    // the drawn target token depends only on the target distribution and
    // the position-keyed threshold, so it is identical across budgets
    // while the proposals (and hence accept/correct labels) differ.
    const std::vector<llaminar2::SamplingDistributionEntry> target_pmf = {
        {0, 0.5f}, {1, 0.25f}, {2, 0.125f}, {3, 0.125f}};
    // Per-position thresholds: the position-keyed table (identical for
    // both budgets by construction of the contract).
    const std::vector<float> position_thresholds = {0.1f, 0.35f, 0.6f,
        0.85f};
    // Draft proposals differ between the budgets; positions without a
    // drafted proposal fall back to serial draws in the production
    // helper's contract (a negative draft token fails loudly, so both
    // arms always propose).
    const std::vector<std::int32_t> winners_full = {0, 1, 2, 3};
    const std::vector<std::int32_t> winners_n2 = {1, 2, 3, 0};

    std::vector<std::uint32_t> drawn_full;
    std::vector<std::uint32_t> drawn_n2;
    std::vector<std::uint32_t> emitted_full;
    std::vector<std::uint32_t> emitted_n2;
    bool proposals_differ = false;
    for (std::size_t position = 0; position < position_thresholds.size();
        ++position)
    {
        const auto full = b4_diagnostic::runSeededPosition(target_pmf,
            winners_full[position], position_thresholds[position]);
        const auto n2 = b4_diagnostic::runSeededPosition(target_pmf,
            winners_n2[position], position_thresholds[position]);
        ASSERT_TRUE(full.ok) << full.error;
        ASSERT_TRUE(n2.ok) << n2.error;
        // S1 hard property: the seeded target tokens are identical.
        EXPECT_EQ(full.token, n2.token) << "position " << position;
        if (full.draft_token != n2.draft_token)
            proposals_differ = true;
        // The emitted stream is the target draw stream in this coupling.
        drawn_full.push_back(static_cast<std::uint32_t>(full.token));
        drawn_n2.push_back(static_cast<std::uint32_t>(n2.token));
        emitted_full.push_back(static_cast<std::uint32_t>(full.token));
        emitted_n2.push_back(static_cast<std::uint32_t>(n2.token));
    }
    EXPECT_TRUE(proposals_differ) << "the budgets must propose differently";
    // The full recorded streams are identical across budgets.
    EXPECT_EQ(drawn_full, drawn_n2);
    EXPECT_EQ(emitted_full, emitted_n2);

    // A negative draft token fails loudly (the production helper's
    // contract), never silently resamples.
    const auto invalid = b4_diagnostic::runSeededPosition(target_pmf, -1,
        0.5f);
    EXPECT_FALSE(invalid.ok);
}

TEST(CPUDraftVocabularyBudget, ProbabilityRejectionExecutedLawEnumeration)
{
    // Mode P replayed through the PRODUCTION rejection sampler
    // (`sampleMTPRejectionRowFromDistributions`): the residual draw takes
    // its own independent threshold. The executed law is enumerated
    // EXACTLY over dyadic threshold bins: every fixture below uses dyadic
    // probabilities (halves/quarters/eighths/sixteenths) so every CDF
    // boundary lands on a bin edge and each bin's outcome is constant —
    // the enumeration is an exact proof for these fixtures, not an
    // approximation. Non-dyadic PMFs are intentionally excluded: the
    // finite-precision contract for them is a production-sampler question
    // this diagnostic does not claim.
    const auto enumerate = [](const std::vector<double> &p,
                              const std::vector<double> &draft_q) {
        const std::size_t vocab = p.size();
        double q_total = 0.0;
        for (const double mass : draft_q)
            q_total += mass;
        // The draw-side PMF must be complete: an incomplete draft PMF is
        // the invalid-PMF negative, rejected precisely (no silent leftover
        // mass assigned to the last token).
        if (std::abs(q_total - 1.0) > 0.0)
            throw std::invalid_argument(
                "draft PMF does not sum to one (invalid PMF negative)");
        std::vector<llaminar2::SamplingDistributionEntry> target;
        for (std::size_t id = 0; id < vocab; ++id)
            target.push_back({static_cast<std::int32_t>(id),
                static_cast<float>(p.at(id))});
        std::vector<llaminar2::SamplingDistributionEntry> draft;
        for (std::size_t id = 0; id < vocab; ++id)
            draft.push_back({static_cast<std::int32_t>(id),
                static_cast<float>(draft_q.at(id))});
        constexpr int kBins = 64; // 2^6: exact dyadic bin probabilities.
        std::vector<double> realized(vocab, 0.0);
        for (int draw_bin = 0; draw_bin < kBins; ++draw_bin)
        {
            const double draw_u =
                (static_cast<double>(draw_bin) + 0.5) / kBins;
            double cumulative = 0.0;
            std::int32_t drafted = static_cast<std::int32_t>(vocab - 1);
            for (std::size_t id = 0; id < vocab; ++id)
            {
                cumulative += draft_q.at(id);
                if (draw_u < cumulative)
                {
                    drafted = static_cast<std::int32_t>(id);
                    break;
                }
            }
            for (int accept_bin = 0; accept_bin < kBins; ++accept_bin)
            {
                const double accept_u =
                    (static_cast<double>(accept_bin) + 0.5) / kBins;
                for (int residual_bin = 0; residual_bin < kBins;
                    ++residual_bin)
                {
                    const double residual_u =
                        (static_cast<double>(residual_bin) + 0.5) / kBins;
                    const auto outcome = llaminar2::
                        sampleMTPRejectionRowFromDistributions(target, draft,
                            drafted, static_cast<float>(accept_u),
                            static_cast<float>(residual_u));
                    if (!outcome.ok)
                        throw std::runtime_error(
                            "production rejection sampler failed: " +
                            outcome.error);
                    realized.at(static_cast<std::size_t>(outcome.token)) +=
                        1.0 / static_cast<double>(kBins * kBins * kBins);
                }
            }
        }
        return realized;
    };

    // Fully dyadic fixtures (the exactness contract above).
    const std::vector<double> p = {0.5, 0.25, 0.125, 0.125};

    // q = p over the full support.
    {
        const auto realized = enumerate(p, p);
        for (std::size_t id = 0; id < p.size(); ++id)
            EXPECT_DOUBLE_EQ(p.at(id), realized.at(id)) << "q=p, token " << id;
    }

    // One-hot q: accepted mass min(p(0), 1) = p(0); the production
    // residual covers the rest exactly.
    {
        const std::vector<double> q = {1.0, 0.0, 0.0, 0.0};
        const auto realized = enumerate(p, q);
        for (std::size_t id = 0; id < p.size(); ++id)
            EXPECT_DOUBLE_EQ(p.at(id), realized.at(id))
                << "one-hot q, token " << id;
    }

    // Real disjoint-support PMF: the draft lives entirely where p has no
    // mass; the production residual must cover p's support exactly.
    {
        const std::vector<double> p_disjoint = {0.5, 0.5, 0.0, 0.0};
        const std::vector<double> q_disjoint = {0.0, 0.0, 0.75, 0.25};
        const auto realized = enumerate(p_disjoint, q_disjoint);
        for (std::size_t id = 0; id < p_disjoint.size(); ++id)
            EXPECT_DOUBLE_EQ(p_disjoint.at(id), realized.at(id))
                << "disjoint support, token " << id;
    }

    // A valid zero probability inside the support.
    {
        const std::vector<double> p_zero = {0.5, 0.25, 0.25, 0.0};
        const std::vector<double> q = {0.25, 0.25, 0.25, 0.25};
        const auto realized = enumerate(p_zero, q);
        for (std::size_t id = 0; id < p_zero.size(); ++id)
            EXPECT_DOUBLE_EQ(p_zero.at(id), realized.at(id))
                << "valid zero probability, token " << id;
    }

    // A3 negative, repaired: keep p AND the actual valid proposal draw
    // fixed (the truthful dyadic draft supplies the draw), then supply
    // the WRONG CORRECTION distribution to the production helper — the
    // helper's accept ratio and residual are computed against THAT
    // distribution, so the executed law deviates from p and the
    // enumeration DETECTS it. (Replacing a truthful draft with another
    // truthful draft would be a pass by theorem — the mismatched
    // correction is the operation under test.)
    {
        const std::vector<double> truthful_draft = {0.25, 0.25, 0.25, 0.25};
        const auto truthful = enumerate(p, truthful_draft);
        for (std::size_t id = 0; id < p.size(); ++id)
            EXPECT_DOUBLE_EQ(p.at(id), truthful.at(id))
                << "truthful draft, token " << id;
        // The wrong-correction construction: the proposal draw stays the
        // truthful dyadic draw, but the correction distribution supplied
        // to the production helper is a DIFFERENT dyadic distribution
        // (its accept ratio p(y)/q_wrong(y) and its residual
        // [p - q_wrong]+ are the wrong ones for this proposal stream).
        const std::vector<double> wrong_correction = {0.125, 0.125, 0.25,
            0.5};
        const auto realized = enumerate(p, wrong_correction);
        bool differs = false;
        for (std::size_t id = 0; id < p.size(); ++id)
            if (std::abs(p.at(id) - realized.at(id)) > 0.0)
                differs = true;
        EXPECT_TRUE(differs)
            << "the wrong-correction negative must change the executed law";
    }

    // The invalid-PMF negative: an under-covering draft (sums to 0.8) is
    // rejected precisely by the draw-side validation (no silent leftover
    // mass assigned to the last token).
    {
        const std::vector<double> undercoverage = {0.2, 0.2, 0.2, 0.2};
        EXPECT_THROW(enumerate(p, undercoverage), std::invalid_argument);
    }

    // Threshold endpoints (restored fixture): the production helper's
    // clamp_unit_threshold maps 0 -> the smallest admissible threshold and
    // 1 -> certainty. With q = delta on token 0 and p(0) = 0.5: the accept
    // probability is 0.5, so a threshold strictly below 0.5 accepts and a
    // threshold at or above 0.5 rejects into the residual.
    {
        const auto below = llaminar2::sampleMTPRejectionRowFromDistributions(
            std::vector<llaminar2::SamplingDistributionEntry>{
                {0, 0.5f}, {1, 0.25f}, {2, 0.125f}, {3, 0.125f}},
            std::vector<llaminar2::SamplingDistributionEntry>{
                {0, 1.0f}, {1, 0.0f}, {2, 0.0f}, {3, 0.0f}},
            0, 0.25f, 0.5f);
        EXPECT_TRUE(below.ok);
        // accept_threshold 0.25 < accept_probability 0.5 -> accepted draft.
        EXPECT_TRUE(below.accepted);
        EXPECT_EQ(0, below.token);
        const auto above = llaminar2::sampleMTPRejectionRowFromDistributions(
            std::vector<llaminar2::SamplingDistributionEntry>{
                {0, 0.5f}, {1, 0.25f}, {2, 0.125f}, {3, 0.125f}},
            std::vector<llaminar2::SamplingDistributionEntry>{
                {0, 1.0f}, {1, 0.0f}, {2, 0.0f}, {3, 0.0f}},
            0, 0.5f, 0.5f);
        EXPECT_TRUE(above.ok);
        // accept_threshold 0.5 is NOT below 0.5 -> rejected; the residual
        // threshold 0.5 over [p - q]+ = {0, .25, .125, .125}: the CDF
        // reaches 0.25 at token 1, 0.375 at token 2, 0.5 at token 3 —
        // u=0.5 is NOT < 0.5, so the fallback emits the last token (3).
        EXPECT_FALSE(above.accepted);
        EXPECT_EQ(3, above.token);
    }
}

TEST(CPUDraftVocabularyBudget, SweepMechanicsUseRuntimeDescriptors)
{
    // One binary, runtime N: the same compiled test constructs several
    // budgets and each records its complete identity string. The
    // non-aligned and boundary budgets are ADMITTED or REJECTED
    // explicitly — nothing is skipped.
    const std::uint32_t vocab = 96;
    std::set<std::string> identities;
    for (const std::uint32_t n : {1u, 3u, 63u, 64u, 65u, 95u})
    {
        const b4_diagnostic::DraftProposalPolicy policy{
            1, b4_diagnostic::CommittedTokenRecencySelector::kVersion,
            b4_diagnostic::SelectorFallback::LowestGlobalIds,
            b4_diagnostic::DrawLaw::DeltaOnShortlist,
            b4_diagnostic::VocabBudget::shortlist(n)};
        const b4_diagnostic::VerifierCouplingPolicy coupling{
            1, b4_diagnostic::CouplingKind::GreedyTargetMismatch};
        EXPECT_NO_THROW(b4_diagnostic::validateBudget(policy.vocab_budget,
            vocab))
            << "N=" << n << " must be admitted for V=" << vocab;
        const std::string identity =
            b4_diagnostic::identityOf(policy, coupling, vocab);
        EXPECT_EQ(std::string::npos, identity.find("n=0"));
        identities.insert(identity);
    }
    // The boundary and beyond: rejected precisely.
    for (const std::uint32_t n : {96u, 97u, 128u, 512u})
    {
        const b4_diagnostic::DraftProposalPolicy policy{
            1, b4_diagnostic::CommittedTokenRecencySelector::kVersion,
            b4_diagnostic::SelectorFallback::LowestGlobalIds,
            b4_diagnostic::DrawLaw::DeltaOnShortlist,
            b4_diagnostic::VocabBudget::shortlist(n)};
        EXPECT_THROW(b4_diagnostic::validateBudget(policy.vocab_budget, vocab),
            std::invalid_argument)
            << "N=" << n << " must be rejected for V=" << vocab;
    }
    // Distinct admitted N values produce distinct recorded identities.
    EXPECT_GT(identities.size(), 1u);
}
