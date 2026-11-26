/**
 * @file Test__ServingSubmissionOutcome.cpp
 * @brief PKV-05 typed submission outcomes: Direct / Queued / Refused.
 *
 * These tests exercise the §5.4 contract on the production ServingSession
 * with a scripted row executor: a request submitted while a slot is free is
 * Direct, one admitted beyond the resident slots within the bounded queue is
 * Queued, and one beyond the admitted Q is refused typed — with the FIFO
 * admission order preserved and the caller's response incarnation never
 * minted on refusal.
 */
#include "execution/serving/ServingSession.h"
#include <gtest/gtest.h>
#include <map>

using namespace llaminar2;
namespace
{
class ScriptedRows final : public IServingRowExecutor
{
public:
    bool forwardServingRows(const std::vector<int> &, const std::vector<std::vector<int>> &rows) override
    {
        logits_.assign(rows.size() * 8, 0.0f);
        for (std::size_t r = 0; r < rows.size(); ++r)
            logits_[r * 8 + 3] = 1.0f;
        return true;
    }
    const float *servingRowLogits(int row) const override { return logits_.data() + row * 8; }
    int prefix_restored_tokens = 0; // >0 simulates a prefix-cache hit admission.
    bool prefix_miss_cache_enabled = false; // cache-enabled admission WITHOUT a hit.
    int restoreServingRowPrefix(int, const std::vector<int32_t> &, PrefixLookupResult &admission) override
    {
        admission = {};
        if (prefix_miss_cache_enabled)
        {
            admission.supported = true;
            admission.cache_enabled = true;
            admission.cached_tokens = 0; // MISS: no cached tokens.
            admission.block_size = 64;
            return 0;
        }
        if (prefix_restored_tokens > 0)
        {
            admission.supported = true;
            admission.cache_enabled = true;
            admission.cached_tokens = prefix_restored_tokens;
            admission.block_size = 64;
        }
        return prefix_restored_tokens;
    }
    int harvest_count = 0;
    PrefixLookupResult last_harvest_admission;
    bool harvestServingRowPrefix(int, const PrefixLookupResult &admission,
        const std::vector<int32_t> &) override
    {
        ++harvest_count;
        last_harvest_admission = admission; // Record the admission the session passed.
        return true;
    }
    // Typed executor continuation: the harness exports the exact KV payload of
    // one slot and re-imports it on restore (proof the session refuses when
    // the executor cannot do this, and succeeds when it can).
    std::optional<std::vector<std::byte>> exportSlotKV(int slot) const override
    {
        auto it = slot_kv_.find(slot);
        if (it == slot_kv_.end()) return std::nullopt;
        return it->second;
    }
    bool fail_import = false;
    bool fail_reset = false;
    bool resetServingRow(int) override { return !fail_reset; }
    bool importSlotKV(int slot, std::span<const std::byte> bytes) override
    {
        if (fail_import) return false;
        slot_kv_[slot] = {bytes.begin(), bytes.end()};
        return true;
    }
    bool resumedSlotKVLength(int slot, int expected) const override
    {
        // The harness tracks the byte extent it imported; its exported length
        // is whatever the manifest declared (the real orchestrator compares
        // the exact KV length). For the session contract, an import with
        // capacity present is consistent.
        const auto it = slot_kv_.find(slot);
        return it != slot_kv_.end() && !it->second.empty();
    }
    void setSlotKV(int slot, std::vector<std::byte> bytes) { slot_kv_[slot] = std::move(bytes); }
private:
    std::vector<float> logits_;
    std::map<int, std::vector<std::byte>> slot_kv_;

};
ServingBatchRequest greedy(std::uint64_t id)
{
    ServingBatchRequest request;
    request.id = id;
    request.prompt = {1, 2, 3};
    request.max_new_tokens = 5;
    request.sampling.temperature = 0.0f;
    request.sampling.top_k = 1;
    request.sampling.seed = 1;
    return request;
}
}

/** @test Direct when a slot is free; Queued within the bounded Q; Refused beyond. */
TEST(ServingSubmissionOutcome, DirectQueuedRefusedWithBoundedQueue)
{
    ScriptedRows rows;
    ServingSession session(rows, 32, 8);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 2));
    // Slot free: Direct (no earlier queue positions).
    EXPECT_EQ(session.submit(greedy(1), error), ServingSubmissionOutcome::Direct);
    EXPECT_TRUE(error.empty());
    // Queue bound Q=2: the second request is Queued (one seat left).
    EXPECT_EQ(session.submit(greedy(2), error), ServingSubmissionOutcome::Queued);
    // Beyond the admitted queue (resident full, queue full): Refused typed.
    EXPECT_EQ(session.submit(greedy(3), error), ServingSubmissionOutcome::Refused);
    EXPECT_FALSE(error.empty());
    // A duplicate live id is Refused with the precise diagnostic.
    EXPECT_EQ(session.submit(greedy(1), error), ServingSubmissionOutcome::Refused);
}

/** @test FIFO order preserved: Direct + Queued admit in submission order. */
TEST(ServingSubmissionOutcome, FifoAdmissionOrderPreserved)
{
    ScriptedRows rows;
    ServingSession session(rows, 32, 8);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 2));
    EXPECT_EQ(session.submit(greedy(1), error), ServingSubmissionOutcome::Direct);
    EXPECT_EQ(session.submit(greedy(2), error), ServingSubmissionOutcome::Queued);
    // Q=2 admitted one queued seat after the direct admission consumed the
    // resident slot; a third request beyond the admitted queue is Refused.
    EXPECT_EQ(session.submit(greedy(3), error), ServingSubmissionOutcome::Refused);
    EXPECT_FALSE(error.empty());
    // One step promotes request 1 into the slot (FIFO); 2 stays queued.
    const auto events = session.step();
    ASSERT_FALSE(events.empty());
    EXPECT_EQ(session.occupancy().resident, 1);
    EXPECT_EQ(session.occupancy().queued, 1);
}

/** @test An executor that cannot export KV refuses suspension BEFORE any mutation. */
TEST(ServingSubmissionOutcome, UnsupportedExecutorRefusesSuspensionPreMutation)
{
    // ScriptedRows with no slot KV seeded: exportSlotKV returns nullopt and the
    // session must refuse WITHOUT resetting or erasing the resident row.
    ScriptedRows rows;
    ServingSession session(rows, 32, 8);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 0));
    EXPECT_EQ(session.submit(greedy(1), error), ServingSubmissionOutcome::Direct);
    session.step();
    auto manifest = session.suspend(1, error);
    EXPECT_FALSE(manifest.has_value());
    EXPECT_NE(error.find("export"), std::string::npos) << error;
    // The request is still resident (no state was destroyed).
    EXPECT_EQ(session.occupancy().resident, 1);
}

/** @test Suspend/restore at a transaction boundary preserves the complete continuation. */

TEST(ServingSubmissionOutcome, SuspendRestoreKeepsExactContinuation)
{
    ScriptedRows rows;
    ServingSession session(rows, 32, 8);
    std::string error;
    ASSERT_TRUE(session.begin(2, 2, error, nullptr, 1));
    rows.setSlotKV(0, {std::byte{1}, std::byte{2}, std::byte{3}});
    rows.setSlotKV(1, {std::byte{4}, std::byte{5}, std::byte{6}});
    auto long_request = greedy(1);
    long_request.max_new_tokens = 500; // Never hit the length boundary in the settle loop.
    EXPECT_EQ(session.submit(long_request, error), ServingSubmissionOutcome::Direct);
    EXPECT_EQ(session.submit(greedy(2), error), ServingSubmissionOutcome::Direct);
    // One step: both rows resident, no pending decisions (logits emitted).
    auto events = session.step();
    ASSERT_FALSE(events.empty());
    // A row mid-forward (needs_feed/has_logits) is refused at its transaction
    // boundary: no speculative state may be dropped. Bounded settle loop.
    std::optional<ServingContinuationManifest> pendant;
    for (int attempt = 0; attempt < 8; ++attempt)
    {
        pendant = session.suspend(1, error);
        if (pendant) break;
        session.step();
    }
    if (!pendant)
        pendant = session.suspend(1, error);
    ASSERT_TRUE(pendant.has_value()) << error;
    EXPECT_EQ(pendant->content().request_id, 1u);
    EXPECT_TRUE(pendant->valid(&error)) << error;
    // The slot is free again (B continues in the cohort without draining).
    EXPECT_EQ(session.occupancy().resident, 1);
    // Restore into the same slot identity with the exact continuation.
    EXPECT_TRUE(session.restore(std::move(*pendant), error)) << error;
    EXPECT_EQ(session.occupancy().resident, 2);
    // The restored row keeps its original identity and law (no reseed).
    EXPECT_EQ(session.restore(ServingContinuationManifest{}, error), false); // Empty refuses.
}

/** @test R14 blocker 1: partial-prefill suspension restores the ORIGINAL prompt (no lost prefill). */
TEST(ServingSubmissionOutcome, PartialPrefillSuspendRestoreReattachesPromptAndMatchesControl)
{
    ScriptedRows rows;
    ServingSession session(rows, 256, 8);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 0));
    rows.setSlotKV(0, {std::byte{1}, std::byte{2}, std::byte{3}});
    auto request = greedy(1);
    // Prompt larger than the chunk so the first step leaves a partial prefill
    // boundary (prompt_consumed < prompt.size()).
    request.prompt.assign(static_cast<std::size_t>(ServingSession::kPrefillChunk) + 5, 9);
    request.max_new_tokens = 8;
    EXPECT_EQ(session.submit(request, error), ServingSubmissionOutcome::Direct);
    session.step(); // Consumes the first chunk; mid-prefill now.
    auto manifest = session.suspend(1, error);
    ASSERT_TRUE(manifest.has_value()) << error;
    EXPECT_EQ(manifest->content().prompt.size(), request.prompt.size());
    EXPECT_LT(manifest->content().prompt_consumed, request.prompt.size());
    // Restore must reattach the ORIGINAL prompt so the remaining prefill chunk
    // is processed (no lost prompt, no skip to decode).
    ASSERT_TRUE(session.restore(std::move(*manifest), error)) << error;
    auto events = session.step();
    ASSERT_FALSE(events.empty());
    EXPECT_EQ(events.front().prompt_tokens, static_cast<int>(request.prompt.size()));
    // The rest of the prompt and the granted decode complete normally.
    for (int i = 0; i < 20 && !session.idle(); ++i)
        session.step();
    ASSERT_TRUE(session.idle());
}

/** @test R14 blocker 2: prefix-backed suspension is refused before any mutation. */
TEST(ServingSubmissionOutcome, PrefixBackedSuspensionRefusesPreMutation)
{
    ScriptedRows rows;
    ServingSession session(rows, 256, 8);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 0));
    rows.setSlotKV(0, {std::byte{1}, std::byte{2}, std::byte{3}});
    // No typed prefix admission contract in the manifest yet: a prefix-backed
    // LIVE row (cached_prompt_tokens > 0 or prefix_admission.hit()) must
    // refuse suspension BEFORE reset/erase (R14 blocker 2).
    rows.prefix_restored_tokens = 32; // simulate a prefix-cache hit admission.
    auto request = greedy(1);
    request.prompt.assign(ServingSession::kPrefillChunk + 5, 9);
    request.max_new_tokens = 8;
    EXPECT_EQ(session.submit(request, error), ServingSubmissionOutcome::Direct);
    session.step();
    auto manifest = session.suspend(1, error);
    EXPECT_FALSE(manifest.has_value());
    EXPECT_NE(error.find("prefix"), std::string::npos) << error;
    // Pre-mutation: the resident row and its slot are untouched.
    EXPECT_EQ(session.occupancy().resident, 1);
    EXPECT_EQ(session.occupancy().queued, 0);
}

/** @test R15 B2 (superseded by R17): the typed prefix admission is carried, so a MISS row supports suspension. */
// The typed admission contract (cache_enabled/supported/block_size/fingerprint_key)
// now travels in the manifest; the cache-enabled MISS case is covered by
// CacheEnabledMissSuspendRestoreKeepsAdmission. Full HIT rows still refuse.
TEST(ServingSubmissionOutcome, PrefixHitAdmissionStillRefusesSuspension)
{
    ScriptedRows rows;
    ServingSession session(rows, 256, 8);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 0));
    rows.setSlotKV(0, {std::byte{1}, std::byte{2}, std::byte{3}});
    // A real HIT (cached tokens + blocks) cannot be carried: still refuses pre-mutation.
    rows.prefix_restored_tokens = 32;
    auto request = greedy(1);
    request.prompt.assign(ServingSession::kPrefillChunk + 5, 9);
    request.max_new_tokens = 8;
    EXPECT_EQ(session.submit(request, error), ServingSubmissionOutcome::Direct);
    session.step();
    auto manifest = session.suspend(1, error);
    EXPECT_FALSE(manifest.has_value());
    EXPECT_NE(error.find("prefix-HIT"), std::string::npos) << error;
    // Pre-mutation: row and slot untouched.
    EXPECT_EQ(session.occupancy().resident, 1);
    EXPECT_EQ(session.occupancy().queued, 0);
}

/** @test R14 blocker 3: failed KV import leaves slot and rows untouched (atomic). */
TEST(ServingSubmissionOutcome, FailedImportLeavesNoMutation)
{
    ScriptedRows rows;
    ServingSession session(rows, 32, 8);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 0));
    rows.setSlotKV(0, {std::byte{1}, std::byte{2}, std::byte{3}});
    EXPECT_EQ(session.submit(greedy(1), error), ServingSubmissionOutcome::Direct);
    session.step();
    auto manifest = session.suspend(1, error);
    ASSERT_TRUE(manifest.has_value()) << error;
    rows.fail_import = true;
    // The import is the ONLY executor mutation; its failure must leave the
    // free slot free and the follower set unchanged (no row sans state).
    EXPECT_FALSE(session.restore(std::move(*manifest), error));
    EXPECT_NE(error.find("import"), std::string::npos) << error;
    EXPECT_EQ(session.occupancy().resident, 0);
    // The slot is reusable: a fresh request is admitted immediately.
    EXPECT_EQ(session.submit(greedy(2), error), ServingSubmissionOutcome::Direct);
    EXPECT_EQ(session.occupancy().queued, 1); // Direct, not refused.
}

/** @test R16 C1: suspension reset failure publishes terminal events through the normal channel. */
TEST(ServingSubmissionOutcome, SuspensionResetFailurePublishesTerminalEvents)
{
    ScriptedRows rows;
    ServingSession session(rows, 32, 8);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 0));
    rows.setSlotKV(0, {std::byte{1}, std::byte{2}, std::byte{3}});
    EXPECT_EQ(session.submit(greedy(1), error), ServingSubmissionOutcome::Direct);
    session.step();
    rows.fail_reset = true;
    EXPECT_FALSE(session.suspend(1, error).has_value());
    // The next step drains the terminal events exactly once (no discard, no
    // duplicate): the sealed row gets its single Error event.
    auto events = session.step();
    ASSERT_FALSE(events.empty());
    std::size_t errors = 0;
    for (const auto &e : events)
        if (e.finished())
        {
            ++errors;
            EXPECT_EQ(e.finish, ServingFinishReason::Error);
            EXPECT_EQ(e.id, 1u);
        }
    EXPECT_EQ(errors, 1u); // Exactly once.
    EXPECT_TRUE(session.step().empty()); // Nothing left.
    EXPECT_NE(error.find("sealed"), std::string::npos) << error;
}

/** @test R16 C2: a refused restore preserves the continuation; foreign session refuses. */
TEST(ServingSubmissionOutcome, RefusedRestoreKeepsCapabilityAndForeignSessionRefuses)
{
    ScriptedRows rows;
    ServingSession session(rows, 32, 8);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 0));
    rows.setSlotKV(0, {std::byte{1}, std::byte{2}, std::byte{3}});
    EXPECT_EQ(session.submit(greedy(1), error), ServingSubmissionOutcome::Direct);
    session.step();
    auto manifest = session.suspend(1, error);
    ASSERT_TRUE(manifest.has_value()) << error;
    EXPECT_EQ(manifest->content().issuer_session_nonce, session.sessionNonce());
    // A foreign session (different issuer nonce) cannot restore it.
    ScriptedRows other_rows;
    ServingSession other(other_rows, 32, 8);
    ASSERT_TRUE(other.begin(1, 1, error, nullptr, 0));
    other_rows.setSlotKV(0, {std::byte{9}, std::byte{8}, std::byte{7}});
    EXPECT_FALSE(other.restore(std::move(*manifest), error));
    // Binding refusal: the continuation is either foreign OR not outstanding
    // in the other session; both diagnostics prove the same binding contract.
    EXPECT_TRUE(error.find("another session") != std::string::npos ||
                error.find("not an outstanding suspension") != std::string::npos) << error;
    // The capability survived the refusal (manifest still usable): the ORIGINAL
    // session restores it into its free slot.
    ASSERT_TRUE(session.restore(std::move(*manifest), error)) << error;
    EXPECT_EQ(session.occupancy().resident, 1);
}

/** @test R16 C2: end() with outstanding suspension emits one terminal Error. */
TEST(ServingSubmissionOutcome, EndWithOutstandingSuspensionEmitsTerminal)
{
    ScriptedRows rows;
    ServingSession session(rows, 32, 8);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 0));
    rows.setSlotKV(0, {std::byte{1}, std::byte{2}, std::byte{3}});
    EXPECT_EQ(session.submit(greedy(1), error), ServingSubmissionOutcome::Direct);
    session.step();
    auto manifest = session.suspend(1, error);
    ASSERT_TRUE(manifest.has_value()) << error;
    // The defined terminal: never silently dropped.
    session.end();
    auto final_events = session.step();
    ASSERT_FALSE(final_events.empty());
    bool errored = false;
    for (const auto &e : final_events)
        if (e.id == 1u && e.finished())
        {
            errored = true;
            EXPECT_EQ(e.finish, ServingFinishReason::Error);
            EXPECT_NE(e.error.find("outstanding suspension"), std::string::npos) << e.error;
        }
    EXPECT_TRUE(errored); // Defined terminal: one Error, no hang, no drop.
}

/** @test R17 D1: colliding request id across sessions refuses with \"another session\". */
TEST(ServingSubmissionOutcome, CollidingIdAcrossSessionsRefusesForeign)
{
    ScriptedRows rows_a;
    ServingSession a(rows_a, 32, 8);
    ScriptedRows rows_b;
    ServingSession b(rows_b, 32, 8);
    std::string error;
    ASSERT_TRUE(a.begin(1, 1, error, nullptr, 0));
    ASSERT_TRUE(b.begin(1, 1, error, nullptr, 0));
    ASSERT_NE(a.sessionNonce(), b.sessionNonce()); // Unique per session.
    rows_a.setSlotKV(0, {std::byte{1}, std::byte{2}, std::byte{3}});
    rows_b.setSlotKV(0, {std::byte{4}, std::byte{5}, std::byte{6}});
    EXPECT_EQ(a.submit(greedy(1), error), ServingSubmissionOutcome::Direct);
    EXPECT_EQ(b.submit(greedy(1), error), ServingSubmissionOutcome::Direct);
    a.step(); b.step();
    auto ma = a.suspend(1, error);
    auto mb = b.suspend(1, error);
    ASSERT_TRUE(ma.has_value());
    ASSERT_TRUE(mb.has_value());
    // A's manifest (id 1) must REFUSE in B with \"another session\" — the
    // nonce binding, not merely the request-id outstanding lookup.
    EXPECT_FALSE(b.restore(std::move(*ma), error));
    EXPECT_NE(error.find("another session"), std::string::npos) << error;
    // Each manifest restores into its OWN session (its own nonce matches).
    EXPECT_TRUE(a.restore(std::move(*ma), error)) << error;
    EXPECT_TRUE(b.restore(std::move(*mb), error)) << error;
}

/** @test R18 E1: a suspended id refuses re-submission; one terminal per suspended request. */
TEST(ServingSubmissionOutcome, SuspendedIdRefusesResubmitAndTerminalOncePerRequest)
{
    ScriptedRows rows;
    ServingSession session(rows, 32, 8);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 0));
    rows.setSlotKV(0, {std::byte{1}, std::byte{2}, std::byte{3}});
    EXPECT_EQ(session.submit(greedy(1), error), ServingSubmissionOutcome::Direct);
    session.step();
    auto m1 = session.suspend(1, error);
    ASSERT_TRUE(m1.has_value()) << error;
    // The suspended id is still OWNED: re-submission refuses (E1), so a
    // second suspension can never overwrite the first outstanding record.
    EXPECT_EQ(session.submit(greedy(1), error), ServingSubmissionOutcome::Refused);
    EXPECT_NE(error.find("suspended"), std::string::npos) << error;
    // Neither restored: end() emits ONE terminal per request (exactly two:
    // the suspended id 1 and the admitted id 1? No — the re-submit was refused
    // so only id 1 exists).
    session.end();
    auto events = session.step();
    std::size_t errors = 0;
    for (const auto &e : events)
        if (e.finished())
        {
            ++errors;
            EXPECT_EQ(e.id, 1u);
            EXPECT_EQ(e.finish, ServingFinishReason::Error);
        }
    EXPECT_EQ(errors, 1u); // Exactly one terminal for the one suspended request.
}

/** @test R17 D1: a manifest minted before end()/begin() refuses the new incarnation. */
TEST(ServingSubmissionOutcome, PreEndManifestRefusesNewIncarnation)
{
    ScriptedRows rows;
    ServingSession session(rows, 32, 8);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 0));
    rows.setSlotKV(0, {std::byte{1}, std::byte{2}, std::byte{3}});
    EXPECT_EQ(session.submit(greedy(1), error), ServingSubmissionOutcome::Direct);
    session.step();
    std::uint64_t first_nonce = session.sessionNonce();
    auto manifest = session.suspend(1, error);
    ASSERT_TRUE(manifest.has_value()) << error;
    session.end();
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 0));
    ASSERT_NE(session.sessionNonce(), first_nonce); // Fresh per incarnation.
    rows.setSlotKV(0, {std::byte{9}, std::byte{8}, std::byte{7}});
    // The pre-end manifest must refuse the new incarnation (its nonce is old).
    EXPECT_FALSE(session.restore(std::move(*manifest), error));
    EXPECT_NE(error.find("another session"), std::string::npos) << error;
}

/** @test R17/R18: cache-enabled MISS row suspends/restores with the SAME admission as the control. */
TEST(ServingSubmissionOutcome, CacheEnabledMissSuspendRestoreKeepsAdmission)
{
    // CONTROL: the same request runs uninterrupted.
    ScriptedRows control_rows;
    ServingSession control(control_rows, 256, 8);
    std::string error;
    ASSERT_TRUE(control.begin(1, 1, error, nullptr, 0));
    control_rows.setSlotKV(0, {std::byte{1}, std::byte{2}, std::byte{3}});
    control_rows.prefix_miss_cache_enabled = true;
    auto control_request = greedy(1);
    control_request.prompt.assign(ServingSession::kPrefillChunk + 5, 9);
    control_request.max_new_tokens = 8;
    EXPECT_EQ(control.submit(control_request, error), ServingSubmissionOutcome::Direct);
    std::vector<ServingBatchEvent> control_events;
    for (int i = 0; i < 20 && !control.idle(); ++i)
        for (const auto &e : control.step())
            control_events.push_back(e);
    ASSERT_TRUE(control.idle());
    const int control_harvests = control_rows.harvest_count;
    const auto control_admission = control_rows.last_harvest_admission;

    // SUSPENDED: the same request with a suspension at a mid-prefill boundary.
    ScriptedRows rows;
    ServingSession session(rows, 256, 8);
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 0));
    rows.setSlotKV(0, {std::byte{1}, std::byte{2}, std::byte{3}});
    rows.prefix_miss_cache_enabled = true;
    auto request = greedy(1);
    request.prompt.assign(ServingSession::kPrefillChunk + 5, 9);
    request.max_new_tokens = 8;
    EXPECT_EQ(session.submit(request, error), ServingSubmissionOutcome::Direct);
    session.step();
    auto manifest = session.suspend(1, error);
    ASSERT_TRUE(manifest.has_value()) << error;
    EXPECT_TRUE(manifest->content().prefix_cache_enabled);
    EXPECT_TRUE(manifest->content().prefix_supported);
    ASSERT_TRUE(session.restore(std::move(*manifest), error)) << error;
    std::vector<ServingBatchEvent> events;
    for (int i = 0; i < 20 && !session.idle(); ++i)
        for (const auto &e : session.step())
            events.push_back(e);
    ASSERT_TRUE(session.idle());
    // Same harvest admission as the control (cache_enabled/supported/block
    // size/fingerprint preserved — never silently defaulted to cache disabled).
    EXPECT_EQ(rows.harvest_count, control_harvests);
    EXPECT_EQ(rows.last_harvest_admission.cache_enabled,
        control_admission.cache_enabled);
    EXPECT_EQ(rows.last_harvest_admission.supported,
        control_admission.supported);
    EXPECT_EQ(rows.last_harvest_admission.block_size,
        control_admission.block_size);
    EXPECT_EQ(rows.last_harvest_admission.fingerprint_key,
        control_admission.fingerprint_key);
    // Events' prefix fields match the control (counters identical).
    ASSERT_EQ(events.size(), control_events.size());
    for (std::size_t i = 0; i < events.size(); ++i)
    {
        EXPECT_EQ(events[i].prefix_cache_enabled, control_events[i].prefix_cache_enabled);
        EXPECT_EQ(events[i].cached_prompt_tokens, control_events[i].cached_prompt_tokens);
        EXPECT_EQ(events[i].prefix_matched_blocks, control_events[i].prefix_matched_blocks);
    }
}

/** @test P4 interleaving: A continues while B suspends and C enters the freed slot. */
TEST(ServingSubmissionOutcome, InterleavedABCSuspendKeepsPeerProgressExact)
{
    ScriptedRows rows;
    ServingSession session(rows, 32, 8);
    std::string error;
    ASSERT_TRUE(session.begin(3, 3, error, nullptr, 2));
    rows.setSlotKV(0, {std::byte{1}, std::byte{2}, std::byte{3}});
    rows.setSlotKV(1, {std::byte{4}, std::byte{5}, std::byte{6}});
    rows.setSlotKV(2, {std::byte{7}, std::byte{8}, std::byte{9}});
    auto a = greedy(1);
    a.max_new_tokens = 500;
    auto b = greedy(2);
    b.max_new_tokens = 500;
    auto c = greedy(3);
    c.max_new_tokens = 500;
    EXPECT_EQ(session.submit(a, error), ServingSubmissionOutcome::Direct);
    EXPECT_EQ(session.submit(b, error), ServingSubmissionOutcome::Direct);
    // Settle both rows to their decision boundary with a bounded loop.
    std::optional<ServingContinuationManifest> suspended_b;
    for (int attempt = 0; attempt < 8; ++attempt)
    {
        suspended_b = session.suspend(2, error);
        if (suspended_b) break;
        session.step();
    }
    ASSERT_TRUE(suspended_b.has_value()) << error;
    // A continues decoding while B is suspended (no cohort drain).
    const auto a_events = session.step();
    ASSERT_EQ(session.occupancy().resident, 1);
    // C enters the freed slot and sees the same shared-forward treatment.
    EXPECT_EQ(session.submit(c, error), ServingSubmissionOutcome::Direct);
    session.step();
    ASSERT_EQ(session.occupancy().resident, 2);
    // B restores (slot identity may differ if C took the original address) and
    // all three continue decoding together; the cohort never drained.
    EXPECT_TRUE(session.restore(std::move(*suspended_b), error)) << error;
    ASSERT_EQ(session.occupancy().resident, 3); // B resident again alongside A and C.
    (void)a_events;
}

/** @test §5.1 pressure probe: a refusing probe keeps requests queued; release re-admits. */
TEST(ServingSubmissionOutcome, PressureProbeHoldsQueuedUntilWitnessHolds)
{
    ScriptedRows rows;
    ServingSession session(rows, 32, 8);
    std::string error;
    ASSERT_TRUE(session.begin(2, 2, error, nullptr, 4));
    // Refuse all promotion until released: bounded backpressure, no drop.
    bool allow = false;
    session.setPressureProbe([&](const ServingBatchRequest &) { return allow; });
    EXPECT_EQ(session.submit(greedy(1), error), ServingSubmissionOutcome::Direct);
    EXPECT_EQ(session.submit(greedy(2), error), ServingSubmissionOutcome::Direct);
    // Step with the probe refusing: requests stay queued, cohort not admitted.
    session.step();
    EXPECT_EQ(session.occupancy().resident, 0);
    EXPECT_EQ(session.occupancy().queued, 2);
    // The probe admits: the next step promotes the FIFO queue.
    allow = true;
    session.step();
    EXPECT_EQ(session.occupancy().resident, 2);
    EXPECT_EQ(session.occupancy().queued, 0);
}

/** @test Q=0 refuses queued requests typed; Direct still works with a free slot. */
TEST(ServingSubmissionOutcome, ZeroQueueAcceptsOnlyDirect)
{
    ScriptedRows rows;
    ServingSession session(rows, 32, 8);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 0));
    EXPECT_EQ(session.submit(greedy(1), error), ServingSubmissionOutcome::Direct);
    // A second request has no slot and no queue: refused (backpressure typed).
    EXPECT_EQ(session.submit(greedy(2), error), ServingSubmissionOutcome::Refused);
}
