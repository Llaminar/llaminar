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
 * CAPABILITY (R15 B3): the manifest is MOVE-ONLY, issued and sealed by the
 * session (capture is private; only ServingSession mints one), and consumed
 * once on restore (the single-use seal refuses a second restore of any
 * retained object). It exposes no public mutation — an edited or forged
 * manifest cannot claim another request, sampling law or KV payload. Full
 * invariant validation covers identity, prompt, progress, budget, KV
 * consistency and the sampler state.
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
class ServingSession; // capture: the session is the sole issuer.
namespace test { struct ServingContinuationManifestTestAccess; } // unit contracts only.

/** @brief One exact continuation manifest; restore validates every field. */
class ServingContinuationManifest final
{
public:
    /** @brief Complete continuation of one admitted request (fixed extent). */
    struct Content
    {
        std::uint64_t request_id = 0;
        int slot = -1;
        std::vector<int32_t> prompt;      ///< ORIGINAL prompt tokens (prefill recoverable).
        int max_new_tokens = 0;           ///< ORIGINAL requested output budget.
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
        // EXACT typed sampler state (RNG position + penalty counts + DRY
        // history). This is the ONLY faithful restoration of a stochastic
        // continuation: a seeded re-construction plus token replay does NOT
        // reproduce the RNG position, generated counts or DRY history.
        Sampler::State sampler_state;
        // ACTUAL executor KV/auxiliary continuation (plan P4: host metadata
        // alone is not a continuation). Empty means the executor could not
        // export its slot state and suspension was refused.
        std::vector<std::byte> kv_payload;
    };

    ServingContinuationManifest() = default;
    ~ServingContinuationManifest() = default;
    // MOVE-ONLY: a retained copy is impossible by construction; a single-use
    // manifest cannot be shadowed or replayed from a duplicate object.
    ServingContinuationManifest(const ServingContinuationManifest &) = delete;
    ServingContinuationManifest &operator=(const ServingContinuationManifest &) = delete;
    ServingContinuationManifest(ServingContinuationManifest &&other) noexcept
        : content_(std::move(other.content_)), captured_(other.captured_),
          moved_from_(false), used_(other.used_)
    {
        other.captured_ = false;
        other.moved_from_ = true;
        other.used_ = true; // The moved-from object carries no capability.
    }
    ServingContinuationManifest &operator=(ServingContinuationManifest &&other) noexcept
    {
        if (this != &other)
        {
            content_ = std::move(other.content_);
            captured_ = other.captured_;
            moved_from_ = false;
            used_ = other.used_;
            other.captured_ = false;
            other.moved_from_ = true;
            other.used_ = true;
        }
        return *this;
    }

    /** @brief Validate the manifest is complete, consistent and NOT consumed. */
    bool valid(std::string *error = nullptr) const noexcept;

    /** @return The captured continuation (immutable after capture). */
    const Content &content() const noexcept { return content_; }
    /** @return Whether a manifest was captured, not moved-from and not consumed. */
    explicit operator bool() const noexcept { return captured_ && !moved_from_ && !used_; }

    /**
     * @brief Restore a sampler from the manifest's EXACT typed state.
     *
     * The original RNG position, generated token counts and DRY history are
     * restored verbatim — never reseeded, never replayed from tokens (token
     * replay cannot reproduce the RNG position or the DRY ring contents).
     * The next stochastic decision is therefore identical to an uninterrupted
     * control.
     */
    Sampler restoreSampler() const;

private:
    friend class ServingSession; // sole production issuer + consumer.
    friend struct ::llaminar2::test::ServingContinuationManifestTestAccess; // unit contracts only.

    /**
     * @brief Capture one continuous row's complete continuation (session only).
     * @param prompt ORIGINAL prompt tokens (required; never invented on restore).
     * @param max_new_tokens ORIGINAL requested output budget (never a cap change).
     */
    bool capture(std::uint64_t request_id, int slot, const SamplingParams &sampling,
        std::vector<int32_t> prompt, int max_new_tokens,
        std::size_t prompt_consumed, int cached_tokens, int completion_tokens,
        int32_t last_token, bool needs_feed, bool has_logits, int grant,
        const std::vector<int32_t> &forced, bool cancelled, int cached_prompt_tokens,
        const std::vector<int32_t> &sampler_history = {},
        const std::vector<std::byte> &kv_payload = {},
        const Sampler::State *sampler_state = nullptr,
        std::string *error = nullptr);

    /** @brief Consume the single-use seal (session on successful restore). */
    void consume() noexcept { used_ = true; }

    Content content_;
    bool captured_ = false;
    bool moved_from_ = false; ///< Set by the move-source: no capability remains.
    bool used_ = false; ///< Single-use seal: consumed exactly once on restore.
};
} // namespace llaminar2
