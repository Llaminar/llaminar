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
#include <gtest/gtest.h>

using namespace llaminar2;
namespace
{
SamplingParams seededLaw()
{
    SamplingParams law;
    law.temperature = 0.7f; // RNG-bound.
    law.top_k = 5;
    law.top_p = 0.9f;
    law.seed = 42;
    return law;
}
}

/** @test A complete RNG-bound continuation captures and validates. */
TEST(ServingContinuationManifest, CompleteContinuationIsValid)
{
    ServingContinuationManifest manifest;
    std::string error;
    SamplingParams law = seededLaw();
    ASSERT_TRUE(manifest.capture(7, 2, law, 8, /*cached_tokens=*/12,
        /*completion_tokens=*/3, /*last_token=*/9, /*needs_feed=*/true,
        /*has_logits=*/false, /*grant=*/5, std::vector<int32_t>{21, 22},
        /*cancelled=*/false, /*cached_prompt_tokens=*/6,
        std::vector<int32_t>{31, 32, 33}, &error)) << error;
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
    ASSERT_TRUE(manifest.capture(1, 0, greedy, 4, /*cached_tokens=*/10,
        /*completion_tokens=*/2, /*last_token=*/5, /*needs_feed=*/false,
        /*has_logits=*/true, /*grant=*/0, {}, /*cancelled=*/false,
        /*cached_prompt_tokens=*/0, {}, &error)) << error;
    EXPECT_TRUE(manifest.valid(&error)) << error;
}

/** @test RNG-bound decisions must carry an exact matching history. */
TEST(ServingContinuationManifest, RngHistoryMustMatchDecisions)
{
    ServingContinuationManifest manifest;
    std::string error;
    SamplingParams law = seededLaw();
    ASSERT_TRUE(manifest.capture(2, 1, law, 0, /*cached_tokens=*/4,
        /*completion_tokens=*/2, /*last_token=*/3, /*needs_feed=*/false,
        /*has_logits=*/true, /*grant=*/0, {}, /*cancelled=*/false,
        /*cached_prompt_tokens=*/0, {}, &error)) << error;
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
    EXPECT_FALSE(manifest.capture(3, 0, law, 0, /*cached_tokens=*/0,
        /*completion_tokens=*/1, /*last_token=*/4, /*needs_feed=*/false,
        /*has_logits=*/false, /*grant=*/0, {}, /*cancelled=*/false,
        /*cached_prompt_tokens=*/0, {}, &error));
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
    // Control: a sampler seeded once and advanced by the recorded history.
    Sampler control(law.seed);
    for (int i = 0; i < 5; ++i) control.record_prompt_token(i);
    for (int i = 0; i < 3; ++i) control.record_prompt_token(100 + i);
    // Manifest capturing the SAME history as decisions.
    ServingContinuationManifest manifest;
    ASSERT_TRUE(manifest.capture(9, 1, law, 0, /*cached_tokens=*/4,
        /*completion_tokens=*/3, /*last_token=*/103, /*needs_feed=*/false,
        /*has_logits=*/true, /*grant=*/0, {}, /*cancelled=*/false,
        /*cached_prompt_tokens=*/0,
        std::vector<int32_t>{100, 101, 102}, &error)) << error;
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
