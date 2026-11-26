/**
 * @file ServingContinuationManifest.cpp
 * @brief Continuation manifest implementation.
 */
#include "ServingContinuationManifest.h"

namespace llaminar2
{
bool ServingContinuationManifest::capture(std::uint64_t request_id, int slot,
    const SamplingParams &sampling, std::size_t prompt_consumed, int cached_tokens,
    int completion_tokens, int32_t last_token, bool needs_feed, bool has_logits,
    int grant, const std::vector<int32_t> &forced, bool cancelled,
    int cached_prompt_tokens, const std::vector<int32_t> &sampler_history,
    const std::vector<std::byte> &kv_payload, const Sampler::State *sampler_state,
    std::string *error)
{
    if (request_id == 0 || slot < 0 || cached_tokens < 0 || completion_tokens < 0 ||
        (cached_tokens == 0 && (prompt_consumed != 0 || last_token != 0)))
    {
        if (error) *error = "continuation manifest requires a live request identity and nonnegative state";
        return false;
    }
    if (completion_tokens > 0 && last_token == 0 && cached_tokens == 0)
    {
        if (error) *error = "continuation manifest has committed output without KV progress";
        return false;
    }
    if (kv_payload.empty())
    {
        // A continuation without the ACTUAL executor KV payload is not a
        // complete continuation; refuse (the caller must export or defer).
        if (error) *error = "continuation manifest requires the executor KV payload";
        return false;
    }
    content_ = Content{};
    content_.request_id = request_id;
    content_.slot = slot;
    content_.sampling = sampling;
    content_.prompt_consumed = prompt_consumed;
    content_.cached_tokens = cached_tokens;
    content_.completion_tokens = completion_tokens;
    content_.last_token = last_token;
    content_.needs_feed = needs_feed;
    content_.has_logits = has_logits;
    content_.grant = grant;
    content_.forced = forced;
    content_.cancelled = cancelled;
    content_.cached_prompt_tokens = cached_prompt_tokens;
    content_.sampler_prompt.clear();
    content_.sampler_decisions = sampler_history;
    content_.kv_payload = kv_payload;
    if (sampler_state)
        content_.sampler_state = *sampler_state; // Exact RNG/penalty/DRY state.
    captured_ = true;
    return true;
}

Sampler ServingContinuationManifest::restoreSampler() const
{
    Sampler sampler(content_.sampling.seed);
    // RESTORE THE EXACT TYPED STATE: the original RNG position, generated
    // token counts and DRY history are imported verbatim. Token replay cannot
    // reproduce the RNG position or the DRY ring contents (record_prompt_token
    // only inserts a zero count); the state is the only faithful continuation.
    sampler.importState(content_.sampler_state);
    return sampler;
}

bool ServingContinuationManifest::valid(std::string *error) const noexcept
{
    if (!captured_)
    {
        if (error) *error = "continuation manifest has no captured content";
        return false;
    }
    if (content_.request_id == 0 || content_.slot < 0 ||
        content_.cached_tokens < 0 || content_.completion_tokens < 0)
    {
        if (error) *error = "continuation manifest fields violate their invariants";
        return false;
    }
    // The sampler/RNG history is authoritative as captured: it is never
    // invented or repaired here. The manifest only requires that a request
    // with committed output has a matching sampler history WHEN THE LAW IS
    // RNG-bound (temperature > 0: each decision advances RNG); greedy history
    // is empty by construction and validated by re-execution, not fabrication.
    if (content_.completion_tokens > 0 && content_.sampling.temperature > 0.0f &&
        content_.sampler_decisions.size() != static_cast<std::size_t>(content_.completion_tokens))
    {
        if (error) *error = "continuation manifest sampler history disagrees with RNG-bound decisions";
        return false;
    }
    if (content_.cached_tokens == 0 &&
        (content_.prompt_consumed != 0 || content_.last_token != 0 ||
         content_.completion_tokens != 0))
    {
        if (error) *error = "continuation manifest has physical progress without KV backing";
        return false;
    }
    return true;
}
} // namespace llaminar2
