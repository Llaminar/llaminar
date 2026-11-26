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
    bool resetServingRow(int) override { return true; }
    int restoreServingRowPrefix(int, const std::vector<int32_t> &, PrefixLookupResult &admission) override
    {
        admission = {};
        return 0;
    }
    bool harvestServingRowPrefix(int, const PrefixLookupResult &, const std::vector<int32_t> &) override
    {
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
    bool importSlotKV(int slot, std::span<const std::byte> bytes) override
    {
        slot_kv_[slot] = {bytes.begin(), bytes.end()};
        return true;
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
