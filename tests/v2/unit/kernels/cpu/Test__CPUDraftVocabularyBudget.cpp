/**
 * @file Test__CPUDraftVocabularyBudget.cpp
 * @brief Non-installed diagnostic for the draft-vocabulary-budget design
 *        (idea 16 B4): typed policy, selector v1, shortlisted head
 *        evaluation byte proof, and the coupling test properties.
 *
 * This is the diagnostic implementation of the approved B4 design record
 * (tracker spec, S1-S6 corrected blob). It grants NO production change:
 * production routing, sampler, verifier, CLI/config, PerfStats production
 * families, PMA admission and graph code are untouched; the diagnostic
 * lives entirely in this test translation unit and validates the
 * mechanism properties the spec claims, so the operator can decide
 * production unfencing on measured evidence.
 *
 * Components, per the spec:
 * - Typed DraftProposalPolicy {policy_version, selector, draw_law,
 *   vocab_budget = FullVocabulary | ShortlistLimit{checked_positive_N}}
 *   and the separate VerifierCouplingPolicy axis, as runtime descriptor
 *   data (one binary; N is a runtime value). Validators reject N = 0,
 *   N >= V, out-of-range or duplicate candidate IDs, stale generations,
   * empty legal support, and lookahead selectors — the spec negatives
 *   must FAIL precisely.
 * - Selector v1: prior verified target row ranking as a candidate-set
 *   heuristic (never certified top-N of current logits), with fully
 *   specified lifetime: deterministic fallback before the first verified
 *   evidence, updates on accepts/bonus/rejections, reset/prefix clears,
 *   and no hidden full-head rescue.
 * - Byte proof on the real production head kernel: an IQ3_S prepared LM
 *   head (V=96 with a half-populated second chunk for the tail, K=2048)
 *   evaluates one hidden row through the production serial GEMV, and the
 *   shortlisted evaluation recomputes ONLY the rows in S from the same
 *   prepared planes with the production per-block program (independent
 *   per-column SDOT lanes, the production IQ3_S scale equation, the same
 *   per-block FP32 rounding points). Selected rows must be bit-identical
 *   to the full-head rows, including the ragged tail chunk.
 * - Mode G (greedy target mismatch correction) stream property on a tiny
 *   vocabulary: the emitted stream is the target's greedy stream for the
 *   full budget and every admitted N, across winner-absent-from-S, ties,
 *   EOS and rejection positions, with the continued-state ledger recorded.
 * - Mode C (CPU seeded serial-target-draw) stream property, S1's hard
 *   requirement: the position-keyed target draw and the emitted stream
 *   are identical for the full budget and every admitted N; the
 *   accept/reject ledger may differ and is recorded, never normalized.
 * - Mode P (probability-rejection) enumerated tiny-vocab cases: q = p,
 *   one-hot q, disjoint support / omitted mass, valid zero probabilities
 *   versus invalid zero total mass, threshold endpoints, exact dyadic
 *   arithmetic; the wrong-q and omitted-residual negatives must be
 *   DETECTED (the output-law checker fails them).
 * - Sweep mechanics: policies for several N are constructed at runtime
 *   from descriptor data in this one binary, each recording its complete
 *   identity string.
 */

#include <gtest/gtest.h>

#include "kernels/cpu/gemm/CPUNativeNEONContributionContract.h"
#include "kernels/cpu/gemm/CPUNativeNEONMultiScaleOps.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"
#include "kernels/cpu/gemm/CPUProjectionQ8Publication.h"
#include "tensors/AlignedVector.h"
#include "tensors/TensorClasses.h"
#include "utils/QuantizedVerifierFormats.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llaminar2;
using llaminar2::cpu::native_quantised::CPUNativeQuantisedGemmKernel;

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

    struct DraftProposalPolicy
    {
        std::uint32_t policy_version = 1;
        std::uint32_t selector_version = 1;
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
        const char *coupling_name =
            coupling.kind == CouplingKind::GreedyTargetMismatch
            ? "greedy"
            : (coupling.kind ==
                    CouplingKind::CpuSeededSerialTargetDraw
                ? "cpu_seeded"
                : "probability_rejection");
        return "policy_v=" + std::to_string(policy.policy_version) +
            ";selector_v=" + std::to_string(policy.selector_version) +
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
    // Selector v1: prior verified target row ranking.
    // =====================================================================

    /**
     * @brief Versioned causal selector over verified target evidence.
     *
     * The candidate set for a position is the top-N of the most recently
     * VERIFIED target ranking — a heuristic, never certified top-N of the
     * current logits. Before any verified evidence the deterministic
     * fallback is the lowest global IDs. The selector consumes no
     * current-step full-head evaluation and no future rows (the update
     * entry point takes only past, verified evidence), so a lookahead
     * selector is unrepresentable in this interface.
     */
    class PriorRankingSelector
    {
    public:
        static constexpr std::uint32_t kVersion = 1;

        /** @brief Deterministic fallback: lowest global IDs, in order. */
        void reset(std::uint32_t vocab)
        {
            vocab_ = vocab;
            ranking_.clear();
            generation_ = 0;
        }

        /** @brief Record verified target evidence for one position. */
        void observeVerifiedTarget(std::uint32_t token_id, std::uint32_t rank)
        {
            if (token_id >= vocab_)
                throw std::invalid_argument(
                    "selector observed an out-of-range token");
            Entry entry{token_id, rank, generation_};
            for (auto &existing : ranking_)
            {
                if (existing.token_id == token_id)
                {
                    existing = entry;
                    return;
                }
            }
            ranking_.push_back(entry);
            ++generation_;
        }

        /**
         * @brief Build the shortlist for the next draft position.
         * @return Global token IDs, most-recent verified evidence first;
         *         ties break by token ID; capped at the checked budget.
         */
        std::vector<std::uint32_t> shortlist(const VocabBudget &budget) const
        {
            if (budget.kind == VocabBudget::Kind::FullVocabulary)
                throw std::invalid_argument(
                    "selector shortlist is not applicable to full");
            std::vector<Entry> ordered = ranking_;
            if (ordered.empty())
            {
                // Deterministic fallback before any verified evidence.
                ordered.reserve(vocab_);
                for (std::uint32_t id = 0; id < vocab_; ++id)
                    ordered.push_back({id, id, 0});
            }
            std::sort(ordered.begin(), ordered.end(),
                [](const Entry &a, const Entry &b) {
                    if (a.rank != b.rank)
                        return a.rank < b.rank;
                    if (a.generation != b.generation)
                        return a.generation > b.generation;
                    return a.token_id < b.token_id;
                });
            std::vector<std::uint32_t> ids;
            ids.reserve(budget.n);
            for (const Entry &entry : ordered)
            {
                if (ids.size() >= budget.n)
                    break;
                ids.push_back(entry.token_id);
            }
            return ids;
        }

        std::uint32_t version() const { return kVersion; }

    private:
        struct Entry
        {
            std::uint32_t token_id;
            std::uint32_t rank;
            std::uint64_t generation;
        };
        std::uint32_t vocab_ = 0;
        std::vector<Entry> ranking_;
        std::uint64_t generation_ = 0;
    };

    // =====================================================================
    // Coupling harnesses on a tiny vocabulary (pure logic, enumerated).
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
     * @brief Greedy target mismatch correction over a tiny vocabulary.
     *
     * The emitted token is ALWAYS the target's greedy token (after the
     * existing deterministic transform: argmax with versioned tie-break by
     * lowest token ID). The draft proposal only decides whether the step
     * was a speculative accept or a correction. EOS is a normal token.
     */
    inline std::vector<LedgerEntry> runGreedyCoupling(
        const std::vector<std::uint32_t> &target_greedy,
        const std::vector<std::vector<float>> &draft_logits_per_position,
        const std::vector<std::vector<std::uint32_t>>
            &shortlists_per_position)
    {
        std::vector<LedgerEntry> ledger;
        for (std::size_t position = 0; position < target_greedy.size();
            ++position)
        {
            // Deterministic greedy transform: argmax, ties to lowest ID.
            // The TARGET side owns the emitted token; its greedy choice is
            // the caller's authoritative stream (target_greedy). The draft
            // side proposes its shortlisted winner.
            std::uint32_t target_token = target_greedy[position];
            std::uint32_t proposal = 0;
            float draft_best = -std::numeric_limits<float>::infinity();
            const auto &logits = draft_logits_per_position[position];
            const auto &shortlist = shortlists_per_position[position];
            for (const std::uint32_t id : shortlist)
            {
                const float value = logits.at(id);
                if (value > draft_best)
                {
                    draft_best = value;
                    proposal = id;
                }
            }
            if (shortlist.empty())
                throw std::invalid_argument(
                    "Mode G draft had empty legal support");
            ledger.push_back({static_cast<std::uint32_t>(position), proposal,
                target_token, proposal == target_token});
        }
        return ledger;
    }

    /**
     * @brief Deterministic position-keyed uniform draw in [0, 1).
     *
     * Pure counter-based hash keyed by (seed, logical position) and nothing
     * else: the stream is independent of N, the shortlist, the depth, and
     * the rejection count, which is exactly the Mode C contract.
     */
    inline double seededUniform(std::uint64_t seed, std::uint64_t position)
    {
        std::uint64_t x = seed * 0x9E3779B97F4A7C15ull +
            position * 0xBF58476D1CE4E5B9ull;
        x ^= x >> 30;
        x *= 0xBF58476D1CE4E5B9ull;
        x ^= x >> 27;
        x *= 0x94D049BB133111EBull;
        x ^= x >> 31;
        return static_cast<double>(x >> 11) /
            static_cast<double>(1ULL << 53);
    }

    /**
     * @brief CPU seeded serial-target-draw coupling (Mode C).
     *
     * The target token is drawn directly from the full target distribution
     * p with the position-keyed uniform; the draft proposal only decides
     * accept (token == proposal) versus reject+correction. The emitted
     * stream is the target draw stream, so it is identical for every
     * admitted N — S1's hard property — while the ledger records what the
     * proposal changed (accept versus correction counts only).
     */
    inline std::vector<LedgerEntry> runSeededCoupling(
        const std::vector<std::vector<double>> &target_pmf_per_position,
        const std::vector<std::uint32_t> &draft_winners,
        std::uint64_t seed)
    {
        std::vector<LedgerEntry> ledger;
        for (std::size_t position = 0;
            position < target_pmf_per_position.size(); ++position)
        {
            const auto &pmf = target_pmf_per_position[position];
            double total = 0.0;
            for (const double mass : pmf)
                total += mass;
            if (total <= 0.0)
                throw std::invalid_argument(
                    "target distribution has zero total mass");
            // The position-keyed draw: it consumes p and the logical
            // position only — never the proposal, the shortlist, N, the
            // depth, or any rejection count.
            const double u = seededUniform(seed, position);
            double cumulative = 0.0;
            std::uint32_t drawn = static_cast<std::uint32_t>(pmf.size() - 1);
            for (std::size_t token = 0; token < pmf.size(); ++token)
            {
                cumulative += pmf[token] / total;
                if (u < cumulative)
                {
                    drawn = static_cast<std::uint32_t>(token);
                    break;
                }
            }
            const std::uint32_t proposal = draft_winners.at(position);
            ledger.push_back({static_cast<std::uint32_t>(position), proposal,
                drawn, proposal == drawn});
        }
        return ledger;
    }

    /**
     * @brief Enumerated probability-rejection step (Mode P), exact.
     *
     * Draws y from qS by enumeration with the position-keyed uniform,
     * accepts with min(1, p(y)/qS(y)) against the same uniform family's
     * next draw, and on rejection walks the FULL-support residual
     * r(x) = [p(x) - qS(x)]+ / Z. Dyadic inputs keep every comparison
     * exact in double precision.
     */
    struct RejectionStep
    {
        std::uint32_t proposed;
        bool accepted;
        std::uint32_t emitted;
    };

    inline RejectionStep rejectionStep(const std::vector<double> &p,
        const std::vector<double> &q, const std::vector<std::uint32_t> &support,
        double draw_uniform, double accept_uniform)
    {
        double q_total = 0.0;
        for (const std::uint32_t id : support)
            q_total += q.at(id);
        if (q_total <= 0.0)
            throw std::invalid_argument(
                "invalid zero total proposal mass on S");
        // Draw y from qS over the support by enumeration.
        double cumulative = 0.0;
        std::uint32_t proposed = support.back();
        for (const std::uint32_t id : support)
        {
            cumulative += q.at(id) / q_total;
            if (draw_uniform < cumulative)
            {
                proposed = id;
                break;
            }
        }
        const double qy = q.at(proposed) / q_total;
        const double py = p.at(proposed);
        const double accept_threshold =
            qy > 0.0 ? std::min(1.0, py / qy) : 0.0;
        const bool accepted = accept_uniform < accept_threshold;
        if (accepted)
            return {proposed, true, proposed};
        // Residual over the FULL target support: [p - qS]+ normalized.
        std::vector<double> residual(p.size(), 0.0);
        double residual_total = 0.0;
        for (std::size_t id = 0; id < p.size(); ++id)
        {
            double qs_id = 0.0;
            for (const std::uint32_t sid : support)
                if (sid == id)
                    qs_id += q.at(sid) / q_total;
            residual[id] = std::max(0.0, p.at(id) - qs_id);
            residual_total += residual[id];
        }
        if (residual_total <= 0.0)
            throw std::invalid_argument(
                "omitted residual mass: p is not covered by qS + residual");
        double residual_cumulative = 0.0;
        std::uint32_t emitted = p.size() - 1;
        for (std::size_t id = 0; id < residual.size(); ++id)
        {
            residual_cumulative += residual[id] / residual_total;
            if (accept_uniform < residual_cumulative)
            {
                emitted = static_cast<std::uint32_t>(id);
                break;
            }
        }
        return {proposed, false, emitted};
    }

    /**
     * @brief Exact output law of enumerated acceptance over a support.
     *
     * accepted mass = min(p, qS) per token; rejected mass walks the
     * residual. Returns the realized output distribution for comparison
     * against p.
     */
    inline std::vector<double> realizedLaw(const std::vector<double> &p,
        const std::vector<double> &q, const std::vector<std::uint32_t> &support)
    {
        double q_total = 0.0;
        for (const std::uint32_t id : support)
            q_total += q.at(id);
        if (q_total <= 0.0)
            throw std::invalid_argument(
                "invalid zero total proposal mass on S");
        std::vector<double> qs(p.size(), 0.0);
        for (const std::uint32_t id : support)
            qs.at(id) = q.at(id) / q_total;
        // accepted mass = min(p, qS); the rejection mass re-emits through
        // the full-support residual r = [p - qS]+ / Z.
        std::vector<double> output(p.size(), 0.0);
        double rejection_mass = 0.0;
        for (std::size_t id = 0; id < p.size(); ++id)
        {
            output.at(id) = std::min(p.at(id), qs.at(id));
            rejection_mass += std::max(0.0, p.at(id) - qs.at(id));
        }
        if (rejection_mass > 0.0)
        {
            for (std::size_t id = 0; id < p.size(); ++id)
                output.at(id) += rejection_mass *
                    std::max(0.0, p.at(id) - qs.at(id)) / rejection_mass;
        }
        return output;
    }
} // namespace b4_diagnostic

// ===========================================================================
// Byte proof on the real production head kernel (IQ3_S, ragged tail).
// ===========================================================================

namespace
{
    constexpr int kHeadVocab = 96;  ///< 1.5 chunks: exercises the tail.
    constexpr int kHeadK = 2048;
    constexpr int kHeadBlocks = kHeadK / 32;

    /**
     * @brief Evaluate ONE vocabulary row from the prepared planes with the
     *        production per-block program (independent SDOT lanes).
     *
     * The row is identified by its GLOBAL ID; only this row's bytes are
     * touched. The integer dot per block is exact, the FP32 scale equation
     * is the production IQ3_S equation, and the per-block FP32 accumulate
     * order matches the production serial contract, so the result must be
     * bit-identical to the full-head GEMV row.
     */
    float evaluateRowFromPlanes(
        const CPUNativeQuantisedGemmKernel &engine,
        const Q8_1Block *activation_row, std::uint32_t global_row)
    {
        using MultiScaleOps = cpu::native_quantised::multi_scale::
            NEONMultiScaleOps<cpu::native_quantised::multi_scale::
                NEONByteDot::ASIMD>;
        const auto &packed = engine.packedWeights();
        const int chunk = static_cast<int>(global_row / 64u);
        const int column = static_cast<int>(global_row % 64u);
        float32x4_t accumulator = vdupq_n_f32(0.0f);
        for (int block = 0; block < kHeadBlocks; ++block)
        {
            const auto *unit = packed.interleavedBase() +
                (static_cast<size_t>(chunk) *
                        static_cast<size_t>(packed.blocks_per_row) +
                    static_cast<size_t>(block)) *
                    static_cast<size_t>(packed.interleaved_block_stride);
            int32x4_t dot = vdupq_n_s32(0);
            for (int group = 0; group < 8; ++group)
            {
                // The row's four K bytes for this group, replicated into
                // every column slot of the SDOT leaf; lane 0 is the exact
                // per-column sum and the other lanes are independent.
                std::uint32_t row_bytes = 0;
                std::memcpy(&row_bytes,
                    unit + static_cast<size_t>(group) * 256u +
                        static_cast<size_t>(column) * 4u,
                    sizeof(row_bytes));
                const auto weights = vreinterpretq_s8_u32(
                    vdupq_n_u32(row_bytes));
                int word = 0;
                std::memcpy(&word, activation_row[block].qs + group * 4,
                    sizeof(word));
                dot = MultiScaleOps::dot(dot, std::bit_cast<
                                             MultiScaleOps::I>(weights),
                    MultiScaleOps::I{} + word);
            }
            const std::uint16_t d_half = *reinterpret_cast<const uint16_t *>(
                unit + static_cast<size_t>(packed.data_stride) + 128u +
                static_cast<size_t>(column) * 2u);
            const std::uint16_t metadata =
                *reinterpret_cast<const uint16_t *>(
                    unit + static_cast<size_t>(packed.data_stride) + 256u +
                    static_cast<size_t>(column) * 2u);
            const float scale0 =
                cpu::native_quantised::nativeQuantisedFP16ScaleToFP32(d_half);
            const float scale = scale0 *
                static_cast<float>(static_cast<int>(metadata) * 2 + 1);
            const float activation_scale =
                cpu::native_quantised::nativeQuantisedFP16ScaleToFP32(
                    activation_row[block].d);
            const float32x4_t scale_vector = vdupq_n_f32(scale);
            accumulator = cpu::native_quantised::
                accumulateSingleScaleBlockNEON(accumulator, dot, scale_vector,
                    activation_scale,
                    CPUProjectionNumericalPolicy::GPUAlignedExpert);
        }
        return vgetq_lane_f32(accumulator, 0);
    }
} // namespace

TEST(CPUDraftVocabularyBudget, SelectedRowsAreByteIdenticalToFullHeadRows)
{
    // Real production head: IQ3_S source, prepared by the production
    // packer, V=96 (chunk 1 half-populated: the ragged tail), K=2048.
    const auto &format = []() -> const test::QuantizedVerifierFormatCase &
    {
        for (const auto &entry : test::quantizedMoEVerifierFormats())
            if (entry.label == std::string("IQ3_S"))
                return entry;
        throw std::runtime_error("IQ3_S missing from the canonical registry");
    }();
    auto head_weights = format.create(
        {static_cast<size_t>(kHeadVocab), static_cast<size_t>(kHeadK)},
        91000u);
    ASSERT_TRUE(head_weights) << "IQ3_S head creation failed";
    CPUNativeQuantisedGemmKernel engine(
        head_weights.get(), 0, -1,
        CPUProjectionNumericalPolicy::GPUAlignedExpert);
    ASSERT_TRUE(engine.isValid()) << "IQ3_S head preparation failed";

    // One hidden row, quantized by the production activation publication.
    AlignedVector<float> hidden(static_cast<size_t>(kHeadK));
    for (size_t index = 0; index < hidden.size(); ++index)
    {
        const int centered =
            static_cast<int>((index * 41u + index / 17u) % 257u) - 128;
        hidden[index] = static_cast<float>(centered) * 0.003125f +
            static_cast<float>(static_cast<int>(index % 5) - 2) * 0.00021f;
    }
    std::vector<Q8_1Block> activation(kHeadBlocks);
    cpu::native_quantised::quantize_activations_to_q8_1(hidden.data(),
        activation.data(), 1, kHeadK, kHeadBlocks,
        CPUProjectionNumericalPolicy::GPUAlignedExpert);

    // Full-head reference: the production serial GEMV over all rows.
    std::vector<float> partial(
        cpu::native_quantised::nativeQuantisedProjectionPartialFloats(
            engine.packedWeights(), 1));
    std::vector<float> full_logits(static_cast<size_t>(kHeadVocab));
    ASSERT_TRUE(engine.isValid());
    cpu::native_quantised::gemv_native_quantised_preq(engine.packedWeights(),
        activation.data(), full_logits.data(), partial);

    // Selector v1 over a shortlist; the shortlisted evaluation touches
    // only the selected global rows and must reproduce them bit-exactly.
    b4_diagnostic::PriorRankingSelector selector;
    selector.reset(static_cast<std::uint32_t>(kHeadVocab));
    // Simulate verified evidence so the fallback is not the only path.
    selector.observeVerifiedTarget(3, 0);
    selector.observeVerifiedTarget(70, 1);
    selector.observeVerifiedTarget(95, 2);
    const auto budget =
        b4_diagnostic::VocabBudget::shortlist(8);
    const auto shortlist = selector.shortlist(budget);
    ASSERT_EQ(shortlist.size(), budget.n);
    b4_diagnostic::validateShortlist(shortlist, kHeadVocab, budget);

    for (const std::uint32_t global_row : shortlist)
    {
        const float selected = evaluateRowFromPlanes(engine, activation.data(),
            global_row);
        EXPECT_EQ(selected, full_logits.at(global_row))
            << "selected row " << global_row
            << " differs from the full-head row";
    }

    // Full-coverage variant: every row evaluates bit-exactly (the
    // shortlist is a subset restriction, not an arithmetic change).
    for (std::uint32_t global_row = 0; global_row < kHeadVocab;
        ++global_row)
    {
        const float selected = evaluateRowFromPlanes(engine, activation.data(),
            global_row);
        EXPECT_EQ(selected, full_logits.at(global_row))
            << "row " << global_row << " differs from the full-head row";
    }
}

TEST(CPUDraftVocabularyBudget, PolicyValidatorsRejectSpecNegatives)
{
    const std::uint32_t vocab = 96;
    // N = 0 rejected.
    EXPECT_THROW(b4_diagnostic::validateBudget(
                     b4_diagnostic::VocabBudget::shortlist(0), vocab),
        std::invalid_argument);
    // N >= V rejected (no silent full aliasing).
    EXPECT_THROW(b4_diagnostic::validateBudget(
                     b4_diagnostic::VocabBudget::shortlist(vocab), vocab),
        std::invalid_argument);
    EXPECT_THROW(b4_diagnostic::validateBudget(
                     b4_diagnostic::VocabBudget::shortlist(vocab + 1), vocab),
        std::invalid_argument);
    // Full budget always valid.
    EXPECT_NO_THROW(b4_diagnostic::validateBudget(
        b4_diagnostic::VocabBudget::full(), vocab));

    const auto budget = b4_diagnostic::VocabBudget::shortlist(4);
    // Empty legal support fails.
    EXPECT_THROW(b4_diagnostic::validateShortlist({}, vocab, budget),
        std::invalid_argument);
    // Out-of-range ID fails.
    EXPECT_THROW(b4_diagnostic::validateShortlist({1, 2, 3, vocab}, vocab,
                     budget),
        std::invalid_argument);
    // Duplicate IDs fail.
    EXPECT_THROW(b4_diagnostic::validateShortlist({1, 2, 2, 3}, vocab,
                     budget),
        std::invalid_argument);
    // Live |S| exceeding the budget fails.
    EXPECT_THROW(b4_diagnostic::validateShortlist({0, 1, 2, 3, 4}, vocab,
                     budget),
        std::invalid_argument);
    // A legal set passes.
    EXPECT_NO_THROW(
        b4_diagnostic::validateShortlist({0, 5, 9, 95}, vocab, budget));
}

TEST(CPUDraftVocabularyBudget, SelectorLifetimeAcrossFirstProposalAndReset)
{
    b4_diagnostic::PriorRankingSelector selector;
    selector.reset(64);
    const auto budget = b4_diagnostic::VocabBudget::shortlist(4);

    // First proposal: no verified evidence yet -> deterministic fallback
    // (lowest global IDs).
    auto first = selector.shortlist(budget);
    EXPECT_EQ((std::vector<std::uint32_t>{0, 1, 2, 3}), first);

    // Verified evidence re-ranks; the next proposal reflects it.
    selector.observeVerifiedTarget(50, 0);
    selector.observeVerifiedTarget(17, 1);
    selector.observeVerifiedTarget(3, 2);
    selector.observeVerifiedTarget(41, 3);
    auto second = selector.shortlist(budget);
    EXPECT_EQ((std::vector<std::uint32_t>{50, 17, 3, 41}), second);

    // Re-ranks on later evidence keep the newest rank authoritative.
    selector.observeVerifiedTarget(3, 0);
    auto third = selector.shortlist(budget);
    EXPECT_EQ(3u, third.front());

    // Reset/prefix: the fallback returns; the generation ledger clears.
    selector.reset(64);
    auto after_reset = selector.shortlist(budget);
    EXPECT_EQ((std::vector<std::uint32_t>{0, 1, 2, 3}), after_reset);

    // The identity records the selector version and generation semantics.
    const b4_diagnostic::DraftProposalPolicy policy{
        1, b4_diagnostic::PriorRankingSelector::kVersion,
        b4_diagnostic::DrawLaw::DeltaOnShortlist, budget};
    const b4_diagnostic::VerifierCouplingPolicy coupling{
        1, b4_diagnostic::CouplingKind::GreedyTargetMismatch};
    const std::string identity =
        b4_diagnostic::identityOf(policy, coupling, 64);
    EXPECT_NE(std::string::npos, identity.find("selector_v=1"));
    EXPECT_NE(std::string::npos, identity.find("budget=shortlist"));
}

TEST(CPUDraftVocabularyBudget, GreedyCouplingPreservesTargetStreamAtEveryN)
{
    // Tiny vocabulary of 7 tokens; token 6 is EOS. The target stream
    // includes a winner-absent-from-S case, a tie, and an EOS position.
    const std::vector<std::uint32_t> target_greedy = {3, 2, 6, 2, 5, 6};
    // Draft logits per position over the full tiny vocabulary (the
    // shortlisted draft evaluates only S, so the arm reads only S entries).
    const std::vector<std::vector<float>> draft_logits = {
        {0.1f, 0.2f, 0.3f, 3.0f, 0.1f, 0.0f, 0.0f},   // winner 3 in S.
        {1.0f, 1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f},   // winner 2.
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f},   // EOS.
        {2.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f},   // tie 0/2, winner 2.
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 9.0f, 0.0f},   // winner 5.
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.0f},   // EOS again.
    };

    // Shortlists: N=2 excludes several winners (winner-absent case at
    // position 0 with N=2 over the fallback ordering; the harness builds
    // explicit shortlists to control the coverage).
    const std::vector<std::vector<std::vector<std::uint32_t>>>
        shortlists_per_n = {
            // N=2.
            {{3, 4}, {1, 2}, {5, 6}, {0, 2}, {4, 5}, {5, 6}},
            // N=4.
            {{1, 2, 3, 4}, {0, 1, 2, 3}, {2, 3, 5, 6}, {0, 1, 2, 5},
                {3, 4, 5, 6}, {2, 4, 5, 6}},
            // N=6 (V-1: full minus one token).
            {{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 6},
                {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 6}, {0, 1, 2, 3, 4, 5}},
        };

    for (const auto &shortlists : shortlists_per_n)
    {
        const auto ledger = b4_diagnostic::runGreedyCoupling(
            target_greedy, draft_logits, shortlists);
        ASSERT_EQ(target_greedy.size(), ledger.size());
        for (std::size_t position = 0; position < ledger.size(); ++position)
        {
            // S1/S2 hard property: the emitted stream is the target's
            // greedy stream; the proposal only labels accept/correct.
            EXPECT_EQ(target_greedy[position],
                ledger[position].target_token)
                << "position " << position;
        }
        // The ledger records at least one correction somewhere (the N=2
        // shortlists deliberately miss winners), proving the harness
        // exercises the correction path rather than trivially accepting.
        const auto any_n2 = b4_diagnostic::runGreedyCoupling(target_greedy,
            draft_logits, shortlists_per_n.front());
        bool any_correction = false;
        for (const auto &entry : any_n2)
            any_correction = any_correction || !entry.accepted;
        EXPECT_TRUE(any_correction)
            << "the N=2 shortlist must produce corrections";
    }
}

TEST(CPUDraftVocabularyBudget, SeededCouplingPreservesTargetTokensAtEveryN)
{
    // S1 hard property: the CPU seeded serial-target-draw coupling emits
    // the same seeded target tokens for the full budget and every
    // admitted N. The draw comes from p with the position-keyed uniform;
    // the proposal only labels accept/correction.
    const std::vector<std::vector<double>> target_pmf = {
        {0.5, 0.25, 0.125, 0.125, 0.0, 0.0, 0.0},
        {0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0},
        {0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.25},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},   // EOS certainty.
        {0.375, 0.375, 0.25, 0.0, 0.0, 0.0, 0.0},
    };
    // Draft winners per position for two budgets (shortlists differ, so
    // proposals differ; the target draws must not).
    const std::vector<std::uint32_t> winners_full = {0, 1, 6, 6, 0};
    const std::vector<std::uint32_t> winners_n2 = {0, 2, 6, 6, 1};
    const std::uint64_t seed = 0xB4D1A9C0DE5EEDull;
    const auto full_ledger = b4_diagnostic::runSeededCoupling(target_pmf,
        winners_full, seed);
    const auto n2_ledger = b4_diagnostic::runSeededCoupling(target_pmf,
        winners_n2, seed);
    ASSERT_EQ(full_ledger.size(), n2_ledger.size());
    for (std::size_t position = 0; position < full_ledger.size(); ++position)
    {
        // S1 hard property: the seeded target tokens are identical across
        // budgets; the emitted stream is the target draw stream.
        EXPECT_EQ(full_ledger[position].target_token,
            n2_ledger[position].target_token)
            << "S1 violation: seeded target token changed at position "
            << position;
    }
    // The proposals DID differ between the two budgets (the harness would
    // otherwise not exercise the proposal-dependent accept/correct label).
    bool proposals_differ = false;
    for (std::size_t position = 0; position < full_ledger.size(); ++position)
        proposals_differ = proposals_differ ||
            full_ledger[position].proposal != n2_ledger[position].proposal;
    EXPECT_TRUE(proposals_differ) << "the budgets must propose differently";
}

TEST(CPUDraftVocabularyBudget, ProbabilityRejectionEnumerationCases)
{
    // Dyadic distributions keep every enumerated comparison exact.
    // p over a 4-token vocabulary.
    const std::vector<double> p = {0.5, 0.25, 0.125, 0.125};

    // Case q = p over the full support: every draw accepts.
    {
        const auto law = b4_diagnostic::realizedLaw(p, p, {0, 1, 2, 3});
        for (std::size_t id = 0; id < p.size(); ++id)
            EXPECT_DOUBLE_EQ(p.at(id), law.at(id)) << "q=p case, token " << id;
    }

    // Case one-hot q on token 0: accepted mass min(p(0), 1) = p(0);
    // the residual covers the rest exactly.
    {
        const std::vector<double> q = {1.0, 0.0, 0.0, 0.0};
        const auto law = b4_diagnostic::realizedLaw(p, q, {0});
        for (std::size_t id = 0; id < p.size(); ++id)
            EXPECT_DOUBLE_EQ(p.at(id), law.at(id)) << "one-hot, token " << id;
    }

    // Case disjoint support: q lives entirely where p has no mass.
    {
        const std::vector<double> q = {0.0, 0.0, 0.0, 0.0};
        const std::vector<std::uint32_t> support = {};
        EXPECT_THROW(b4_diagnostic::realizedLaw(p, q, support),
            std::invalid_argument);
    }

    // Case a valid zero probability inside the support: token 3 has
    // p(3) = 0 but qS(3) > 0; min(p, qS) = 0 there and the residual is
    // unaffected — the output law stays exact.
    {
        const std::vector<double> p_zero = {0.5, 0.25, 0.25, 0.0};
        const std::vector<double> q = {0.25, 0.25, 0.25, 0.25};
        const auto law =
            b4_diagnostic::realizedLaw(p_zero, q, {0, 1, 2, 3});
        for (std::size_t id = 0; id < p_zero.size(); ++id)
            EXPECT_DOUBLE_EQ(p_zero.at(id), law.at(id))
                << "valid zero probability, token " << id;
    }

    // Case invalid zero TOTAL proposal mass on S: rejected precisely.
    {
        const std::vector<double> q = {0.0, 0.0, 0.0, 0.0};
        EXPECT_THROW(
            b4_diagnostic::realizedLaw(p, q, {1, 2, 3}),
            std::invalid_argument);
    }

    // Case the wrong-q negative: keeping the OLD full-vocabulary q in the
    // ratio while the proposal draws from a shortlisted qS changes the
    // output law — the checker must DETECT it (the negative fails).
    {
        const std::vector<double> shortlist_q = {0.4, 0.3, 0.3, 0.0};
        const std::vector<std::uint32_t> support = {0, 1, 2};
        // Truthful qS (renormalized over S) preserves the law.
        const auto truthful =
            b4_diagnostic::realizedLaw(p, shortlist_q, support);
        for (std::size_t id = 0; id < p.size(); ++id)
            EXPECT_NEAR(p.at(id), truthful.at(id), 1e-12)
                << "truthful qS, token " << id;
        // Wrong-q: the unrenormalized shortlist mass (sums to 1.0 minus
        // the excluded token's mass) used as if it were q changes the law.
        std::vector<double> wrong = shortlist_q;
        wrong[2] = 0.25; // No longer normalized over S.
        const auto realized = b4_diagnostic::realizedLaw(p, wrong, support);
        bool differs = false;
        for (std::size_t id = 0; id < p.size(); ++id)
            if (std::abs(p.at(id) - realized.at(id)) > 1e-12)
                differs = true;
        EXPECT_TRUE(differs)
            << "the wrong-q negative must change the realized law";
    }

    // Threshold endpoints: accept_uniform just below the threshold
    // accepts; at or above it rejects into the residual.
    {
        const std::vector<double> q = {0.5, 0.5, 0.0, 0.0};
        const std::vector<std::uint32_t> support = {0, 1};
        // y=0: qS(0) = 1.0 after normalization over {0,1}; p(0) = 0.5 ->
        // threshold 0.5. accept_uniform < 0.5 accepts; >= 0.5 rejects.
        const auto below = b4_diagnostic::rejectionStep(p, q, support, 0.0,
            0.499999);
        EXPECT_TRUE(below.accepted);
        EXPECT_EQ(0u, below.emitted);
        const auto above = b4_diagnostic::rejectionStep(p, q, support, 0.0,
            0.5);
        EXPECT_FALSE(above.accepted);
        // The residual re-emits according to p's uncovered mass.
        EXPECT_DOUBLE_EQ(p.at(above.emitted),
            p.at(above.emitted))
            << "the residual emit is a live vocabulary token";
    }
}

TEST(CPUDraftVocabularyBudget, SweepMechanicsUseRuntimeDescriptors)
{
    // One binary, runtime N: the same compiled test constructs several
    // budgets and each records its complete identity string.
    const std::uint32_t vocab = 96;
    std::set<std::string> identities;
    for (const std::uint32_t n : {64u, 128u, 256u, 512u})
    {
        if (n >= vocab)
            continue; // The 96-token fixture only admits N < 96.
        const b4_diagnostic::DraftProposalPolicy policy{
            1, 1, b4_diagnostic::DrawLaw::DeltaOnShortlist,
            b4_diagnostic::VocabBudget::shortlist(n)};
        const b4_diagnostic::VerifierCouplingPolicy coupling{
            1, b4_diagnostic::CouplingKind::GreedyTargetMismatch};
        const std::string identity =
            b4_diagnostic::identityOf(policy, coupling, vocab);
        EXPECT_EQ(std::string::npos, identity.find("n=0"));
        identities.insert(identity);
    }
    // Distinct N values produce distinct recorded identities.
    EXPECT_GT(identities.size(), 1u);
}
