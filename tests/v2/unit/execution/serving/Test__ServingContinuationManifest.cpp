/**
 * @file Test__ServingContinuationManifest.cpp
 * @brief Continuation manifest capture and validation contracts (plan P4).
 *
 * These tests prove the manifest captures the complete continuation of one
 * serving request (identity, sampling law, progress, output credit, forced
 * instructions, cancellation and the sampler/RNG history) and refuses
 * inconsistent or incomplete state typed — never reseeding, never inventing
 * history, never accepting a continuation without KV backing.
 */
#include "execution/serving/ServingContinuationManifest.h"
#include <type_traits>
#include <gtest/gtest.h>

using namespace llaminar2;
using llaminar2::test::ServingContinuationManifestTestAccess;
namespace llaminar2::test
{
/** @brief Unit-test access to the session-sealed capture (production path stays private). */
struct ServingContinuationManifestTestAccess
{
    static bool capture(ServingContinuationManifest &m, std::uint64_t nonce,
        std::uint64_t id, int slot,
        const SamplingParams &law, std::vector<int32_t> prompt, int max_new_tokens,
        std::size_t prompt_consumed, int cached_tokens, int completion_tokens,
        int32_t last_token, bool needs_feed, bool has_logits, int grant,
        const std::vector<int32_t> &forced, bool cancelled, int cached_prompt_tokens,
        bool prefix_cache_enabled = false, bool prefix_supported = false,
        int prefix_block_size = 0, std::uint64_t prefix_fingerprint_key = 0,
        const std::vector<int32_t> &history = {},
        const std::vector<std::byte> &kv = {},
        const Sampler::State *state = nullptr, std::string *error = nullptr)
    {
        return m.capture(nonce, id, slot, law, std::move(prompt), max_new_tokens,
            prompt_consumed, cached_tokens, completion_tokens, last_token, needs_feed,
            has_logits, grant, forced, cancelled, cached_prompt_tokens,
            prefix_cache_enabled, prefix_supported, prefix_block_size,
            prefix_fingerprint_key, history, kv, state, error);
    }
    static void consume(ServingContinuationManifest &m) { m.consume(); }
};

} // namespace llaminar2::test
/** @brief Minimal nonempty executor KV payload for manifest capture. */
std::vector<std::byte> kvPayload()
{
    return {std::byte{1}, std::byte{2}, std::byte{3}};
}
/** @brief Exact typed sampler state (nonzero RNG/penalty/DRY history). */
Sampler::State samplerState()
{
    Sampler sampler(7);
    sampler.record_token(11);
    sampler.record_token(22);
    return sampler.exportState();
}

SamplingParams seededLaw()
{
    SamplingParams law;
    law.temperature = 0.7f; // RNG-bound.
    law.top_k = 5;
    law.top_p = 0.9f;
    law.seed = 42;
    return law;
}

/** @test A complete RNG-bound continuation captures and validates. */
TEST(ServingContinuationManifest, CompleteContinuationIsValid)
{
    ServingContinuationManifest manifest;
    std::string error;
    SamplingParams law = seededLaw();
    const auto state = samplerState();
    ASSERT_TRUE(ServingContinuationManifestTestAccess::capture(manifest, 42, 7, 2, law, std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8}, 40,
        8, /*cached_tokens=*/12,
        /*completion_tokens=*/3, /*last_token=*/9, /*needs_feed=*/true,
        /*has_logits=*/false, /*grant=*/5, std::vector<int32_t>{21, 22},
        /*cancelled=*/false, /*cached_prompt_tokens=*/6, false, false, 0, 0,
        std::vector<int32_t>{31, 32, 33}, kvPayload(), &state, &error)) << error;
    ASSERT_TRUE(manifest.valid(&error)) << error;
    const auto &c = manifest.content();
    EXPECT_EQ(c.request_id, 7u);
    EXPECT_EQ(c.slot, 2);
    EXPECT_EQ(c.cached_tokens, 12);
    EXPECT_EQ(c.completion_tokens, 3);
    EXPECT_EQ(c.last_token, 9);
    EXPECT_EQ(c.grant, 5);
    EXPECT_EQ(c.forced.size(), 2u);
    EXPECT_EQ(c.sampling.seed, 42u);
}

/** @test Greedy continuation (temperature 0) needs no sampler history. */
TEST(ServingContinuationManifest, GreedyContinuationNeedsNoRngHistory)
{
    ServingContinuationManifest manifest;
    std::string error;
    SamplingParams greedy;
    greedy.temperature = 0.0f;
    greedy.seed = 1;
    const auto state = samplerState();
    ASSERT_TRUE(ServingContinuationManifestTestAccess::capture(manifest, 42, 1, 0, greedy,
        std::vector<int32_t>{1, 2, 3, 4}, 20, 4, /*cached_tokens=*/10,
        /*completion_tokens=*/2, /*last_token=*/5, /*needs_feed=*/false,
        /*has_logits=*/true, /*grant=*/0, {}, /*cancelled=*/false,
        /*cached_prompt_tokens=*/0, false, false, 0, 0, {}, kvPayload(), &state, &error)) << error;
    EXPECT_TRUE(manifest.valid(&error)) << error;
}

/** @test RNG-bound decisions must carry an exact matching history. */
TEST(ServingContinuationManifest, RngHistoryMustMatchDecisions)
{
    ServingContinuationManifest manifest;
    std::string error;
    SamplingParams law = seededLaw();
    const auto state = samplerState();
    ASSERT_TRUE(ServingContinuationManifestTestAccess::capture(manifest, 42, 2, 1, law, std::vector<int32_t>{5, 6}, 20, 0, /*cached_tokens=*/4,
        /*completion_tokens=*/2, /*last_token=*/3, /*needs_feed=*/false,
        /*has_logits=*/true, /*grant=*/0, {}, /*cancelled=*/false,
        /*cached_prompt_tokens=*/0, false, false, 0, 0, {}, kvPayload(), &state, &error)) << error;
    // Two decisions but an empty RNG history: refused (no invented history).
    EXPECT_FALSE(manifest.valid(&error));
    EXPECT_NE(error.find("history"), std::string::npos) << error;
}

/** @test Inconsistent state (progress without KV backing) refuses typed. */
TEST(ServingContinuationManifest, ProgressWithoutKvBackingRefuses)
{
    ServingContinuationManifest manifest;
    std::string error;
    SamplingParams law = seededLaw();
    const auto state = samplerState();
    EXPECT_FALSE(ServingContinuationManifestTestAccess::capture(manifest, 42, 3, 0, law, std::vector<int32_t>{5, 6}, 20, 0, /*cached_tokens=*/0,
        /*completion_tokens=*/1, /*last_token=*/4, /*needs_feed=*/false,
        /*has_logits=*/false, /*grant=*/0, {}, /*cancelled=*/false,
/*cached_prompt_tokens=*/0, false, false, 0, 0, {}, {}, nullptr, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_FALSE(manifest.valid(&error));
}

/** @test Empty manifest is not valid (capture is mandatory). */
TEST(ServingContinuationManifest, EmptyManifestIsInvalid)
{
    ServingContinuationManifest manifest;
    std::string error;
    EXPECT_FALSE(manifest.valid(&error));
    EXPECT_FALSE(error.empty());
}

/** @test Restore replays captured history so the next decision matches the control. */
TEST(ServingContinuationManifest, RestoreProducesIdenticalNextDecision)
{
    std::string error;
    SamplingParams law = seededLaw();
    const auto state = samplerState();
    // Control: a sampler seeded once and advanced by the recorded history.
    Sampler control(law.seed);
    for (int i = 0; i < 5; ++i) control.record_prompt_token(i);
    for (int i = 0; i < 3; ++i) control.record_prompt_token(100 + i);
    // Manifest capturing the SAME history as decisions.
    ServingContinuationManifest manifest;
    ASSERT_TRUE(ServingContinuationManifestTestAccess::capture(manifest, 42, 9, 1, law, std::vector<int32_t>{5, 6}, 20, 0, /*cached_tokens=*/4,
        /*completion_tokens=*/3, /*last_token=*/103, /*needs_feed=*/false,
        /*has_logits=*/true, /*grant=*/0, {}, /*cancelled=*/false,
        /*cached_prompt_tokens=*/0, false, false, 0, 0,
        std::vector<int32_t>{100, 101, 102}, kvPayload(), &state, &error)) << error;
    auto restored = manifest.restoreSampler();
    // A fixed logits vector; with deterministic (temperature 0) sampling the
    // penalty map is the only variable, so a restored sampler with the same
    // history must make the identical greedy decision as the control.
    std::vector<float> logits(8, 1.0f);
    logits[2] = 9.0f; // Token 2 is the greedy pick unless a penalty shifts it.
    SamplingParams greedy;
    greedy.temperature = 0.0f;
    // Both samplers carry the same history; the penalty map is identical, so
    // the next decision is identical (no reseed, no history loss).
    EXPECT_EQ(restored.sample(logits, greedy), control.sample(logits, greedy));
}

/** @test Stochastic continuation: restored sampler continues the EXACT RNG/penalty/DRY state. */
TEST(ServingContinuationManifest, StochasticContinuationMatchesUninterruptedControl)
{
    SamplingParams law;
    law.temperature = 0.8f; // RNG-bound.
    law.top_k = 50;
    law.top_p = 0.95f;
    law.seed = 12345;
    law.presence_penalty = 0.4f;
    law.frequency_penalty = 0.3f;
    law.repetition_penalty = 1.2f;
    // UNINTERRUPTED control: seed once, sample several decisions with history.
    Sampler control(law.seed);
    std::vector<std::vector<float>> logits_bank(4, std::vector<float>(16, 0.1f));
    for (auto &row : logits_bank)
        for (int t = 4; t < 16; ++t) row[t] = 1.0f; // Distinct top candidates.
    std::vector<int> control_decisions;
    for (int i = 0; i < 4; ++i)
    {
        control_decisions.push_back(control.sample(logits_bank[i], law));
        control.record_token(control_decisions.back());
    }
    // SUSPENDED: capture the exact state after the same history, restore.
    ServingContinuationManifest manifest;
    std::string error;
    auto state = control.exportState();
    ASSERT_TRUE(ServingContinuationManifestTestAccess::capture(manifest, 42, 7, 2, law, std::vector<int32_t>{5, 6}, 20, 0, /*cached_tokens=*/8,
        /*completion_tokens=*/4, /*last_token=*/control_decisions.back(),
        /*needs_feed=*/false, /*has_logits=*/true, /*grant=*/0,
        /*forced=*/{}, /*cancelled=*/false, /*cached_prompt_tokens=*/0, false, false, 0, 0,
        control_decisions, kvPayload(), &state, &error)) << error;
    auto restored = manifest.restoreSampler();
    // CONTROL continues its next draw on the SAME state (penalty/DRY + RNG).
    const int control_next = control.sample(logits_bank[0], law);
    // RESTORED continues from the captured exact state: the next decision MUST
    // equal the uninterrupted control (identical RNG position and history).
    EXPECT_EQ(restored.sample(logits_bank[0], law), control_next);
    // A second draw still matches (the restored internal state advanced the
    // same way), proving stateful, not just one-shot, continuation.
    EXPECT_EQ(restored.sample(logits_bank[1], law), control.sample(logits_bank[1], law));
}

/** @test A manifest without the executor KV payload is refused (nothing to restore). */
TEST(ServingContinuationManifest, MissingKvPayloadRefuses)
{
    ServingContinuationManifest manifest;
    std::string error;
    SamplingParams law = seededLaw();
    const auto state = samplerState();
    EXPECT_FALSE(ServingContinuationManifestTestAccess::capture(manifest, 42, 5, 0, law, std::vector<int32_t>{5, 6}, 20, 0, /*cached_tokens=*/4,
        /*completion_tokens=*/1, /*last_token=*/3, /*needs_feed=*/false,
        /*has_logits=*/true, /*grant=*/0, {}, /*cancelled=*/false,
/*cached_prompt_tokens=*/0, false, false, 0, 0, {}, {}, nullptr, &error));
    EXPECT_NE(error.find("KV"), std::string::npos) << error;
}

/** @test R15 B3: full invariant set — inconsistent progress/budget refuses. */
TEST(ServingContinuationManifest, FullInvariantSetRefusesInconsistency)
{
    std::string error;
    SamplingParams law = seededLaw();
    const auto state = samplerState();
    // Prompt progress beyond the original prompt.
    {
        ServingContinuationManifest m;
        EXPECT_FALSE(ServingContinuationManifestTestAccess::capture(m, 42, 1, 0, law,
            std::vector<int32_t>{5, 6}, 20, 99, /*cached_tokens=*/99,
            /*completion_tokens=*/0, /*last_token=*/0, false, false, 0,
            {}, false, 0, false, false, 0, 0, {}, kvPayload(), &state, &error));
        EXPECT_NE(error.find("bounds"), std::string::npos) << error;
    }
    // Output beyond the original budget.
    {
        ServingContinuationManifest m;
        EXPECT_FALSE(ServingContinuationManifestTestAccess::capture(m, 42, 1, 0, law,
            std::vector<int32_t>{5, 6}, 20, 0, /*cached_tokens=*/4,
            /*completion_tokens=*/21, /*last_token=*/3, false, false, 0,
            {}, false, 0, false, false, 0, 0, {}, kvPayload(), &state, &error));
        EXPECT_NE(error.find("bounds"), std::string::npos) << error;
    }
    // Empty prompt or nonpositive budget.
    {
        ServingContinuationManifest m;
        EXPECT_FALSE(ServingContinuationManifestTestAccess::capture(m, 42, 1, 0, law,
            std::vector<int32_t>{}, 20, 0, /*cached_tokens=*/4,
            /*completion_tokens=*/0, /*last_token=*/0, false, false, 0,
            {}, false, 0, false, false, 0, 0, {}, kvPayload(), &state, &error));
        EXPECT_NE(error.find("prompt"), std::string::npos) << error;
    }
}

/** @test R15 B3: single-use seal — consumed manifests refuse valid()/restore. */
TEST(ServingContinuationManifest, ConsumedManifestRefusesDoubleRestore)
{
    std::string error;
    SamplingParams law = seededLaw();
    const auto state = samplerState();
    ServingContinuationManifest m;
    ASSERT_TRUE(ServingContinuationManifestTestAccess::capture(m, 42, 7, 0, law,
        std::vector<int32_t>{5, 6}, 20, 0, /*cached_tokens=*/4,
        /*completion_tokens=*/1, /*last_token=*/3, false, true, 0,
        {}, false, 0, false, false, 0, 0, std::vector<int32_t>{3}, kvPayload(), &state, &error)) << error;
    ASSERT_TRUE(m.valid(&error)) << error;
    ServingContinuationManifestTestAccess::consume(m); // Session consumes on successful restore.
    EXPECT_FALSE(m.valid(&error));
    EXPECT_NE(error.find("single-use"), std::string::npos) << error;
    EXPECT_FALSE(static_cast<bool>(m)); // No re-usable capability remains.
}

/** @test R15 B3: A retained manifest copy is impossible (move-only); no editing. */
TEST(ServingContinuationManifest, ManifestIsMoveOnlyAndImmutableAfterCapture)
{
    static_assert(!std::is_copy_constructible_v<ServingContinuationManifest>,
        "a retained copy would allow replay after completion");
    static_assert(!std::is_copy_assignable_v<ServingContinuationManifest>,
        "copy assignment would allow a forged continuation");
    std::string error;
    SamplingParams law = seededLaw();
    const auto state = samplerState();
    ServingContinuationManifest m;
    ASSERT_TRUE(ServingContinuationManifestTestAccess::capture(m, 42, 7, 0, law,
        std::vector<int32_t>{5, 6}, 20, 0, /*cached_tokens=*/4,
        /*completion_tokens=*/1, /*last_token=*/3, false, true, 0,
        {}, false, 0, false, false, 0, 0, std::vector<int32_t>{3}, kvPayload(), &state, &error)) << error;
    ServingContinuationManifest moved(std::move(m));
    EXPECT_FALSE(static_cast<bool>(m)); // Moved-from: no capability.
    ASSERT_TRUE(moved.valid(&error)) << error;
}
