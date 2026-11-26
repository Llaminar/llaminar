/**
 * @file ServingContinuationManifest.h
 * @brief Exact request-continuation manifest for transaction-boundary
 *        suspension (plan P4 / design §7).
 *
 * The manifest captures the COMPLETE continuation of one serving request
 * WITHOUT reseeding: the original request identity and sampling law, the
 * sampler/RNG history (prompt tokens and every committed decision), prompt
 * progress, KV length, output credit, forced tokens and cancellation. It is
 * the typed contract the residency module uses to preserve a request while
 * its slot serves another request, and to restore it so the next decision is
 * bitwise identical to the no-movement control.
 *
 * It is PURE STATE, not a ledger: no physical accounting, no allocation in
 * the hot path (fixed capacities declared by admission), and no model
 * sampler/placement state is invented on restore.
 */
#pragma once
#include "utils/Sampler.h"
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace llaminar2
{
/** @brief One exact continuation manifest; restore validates every field. */
class ServingContinuationManifest final
{
public:
    /** @brief Complete continuation of one admitted request (fixed extent). */
    struct Content
    {
        std::uint64_t request_id = 0;
        int slot = -1;
        SamplingParams sampling;          ///< Original law (seed, filters, DRY).
        std::size_t prompt_consumed = 0;  ///< Prompt tokens already in KV.
        int cached_tokens = 0;            ///< KV length of this slot.
        int completion_tokens = 0;        ///< Committed output count.
        int32_t last_token = 0;           ///< Last committed token.
        bool needs_feed = false;          ///< last_token is not yet in KV.
        bool has_logits = false;          ///< Decision row holds next logits.
        int grant = 0;                    ///< Paced tokens still allowed.
        std::vector<int32_t> forced;      ///< Tokens committed instead of sampling.
        bool cancelled = false;
        int cached_prompt_tokens = 0;     ///< Restored prefix-cache tokens.
        // Sampler/RNG history: prompt tokens + every committed decision, so
        // restore regenerates the exact RNG position (never reseeds).
        std::vector<int32_t> sampler_prompt;   ///< Prompt tokens recorded.
        std::vector<int32_t> sampler_decisions; ///< Every committed decision.
    };

    /** @brief Capture one continuous row's complete continuation. */
    bool capture(std::uint64_t request_id, int slot, const SamplingParams &sampling,
        std::size_t prompt_consumed, int cached_tokens, int completion_tokens,
        int32_t last_token, bool needs_feed, bool has_logits, int grant,
        const std::vector<int32_t> &forced, bool cancelled, int cached_prompt_tokens,
        const std::vector<int32_t> &sampler_history = {},
        std::string *error = nullptr);

    /** @brief Validate the manifest is complete and internally consistent. */
    bool valid(std::string *error = nullptr) const noexcept;

    /** @return The captured continuation (immutable after capture). */
    const Content &content() const noexcept { return content_; }
    /** @return Whether a manifest was captured. */
    explicit operator bool() const noexcept { return captured_; }

    /**
     * @brief Restore a sampler from the manifest (no reseed) and replay the
     *        complete captured history so the next RNG state is exact.
     *
     * The sampler is constructed from the ORIGINAL seed, then every captured
     * history token is replayed through the same penalty/RNG path. Greedy
     * laws need no history (decision 0 continues). This is the restore-side
     * contract: a continuation is only restorable when replay yields the exact
     * same RNG state as the no-movement control.
     */
    Sampler restoreSampler() const;

private:
    Content content_;
    bool captured_ = false;
};
} // namespace llaminar2
