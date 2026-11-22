/**
 * @file Test__SharedExecutionProof.cpp
 * @brief A-continues / B-finishes / C-enters-B's-retired-slot protocol proof + discriminating
 *        lifecycle regressions (grants consumed+backed, explicit restore readiness, owned
 *        snapshot receipts, partial final windows, mixed quanta, cancellation, MTP exactly-once).
 *
 * The decode callback is a pure per-row mock; this file proves the lifecycle protocol through
 * the engine's actual owned interfaces (no const-cast, no renamed unrelated fixtures).
 */

#include <gtest/gtest.h>

#include "execution/serving/GroupedServingEngine.h"

namespace llaminar2::test
{
    namespace
    {
        std::string soloDecode(std::uint64_t seed, std::uint64_t position,
                               std::uint64_t window, const std::string &)
        {
            return "x" + std::to_string(seed % 97) + ":" + std::to_string(position) + "/" +
                   std::to_string(window);
        }
        RowDecodeFn solo = &soloDecode;
        std::string windowSlice(std::uint64_t seed, std::uint64_t from, std::uint64_t count,
                                std::uint64_t quantum)
        {
            std::string out;
            for (std::uint64_t pos = from; pos < from + count; ++pos)
                out += soloDecode(seed, pos, quantum, std::string{});
            return out;
        }
        constexpr std::uint64_t kDest = 1;
        constexpr std::uint64_t kDestB = 2;
        constexpr std::uint64_t kDestC = 3;
        constexpr std::uint64_t kDestD = 4;
        void bankDest(GroupedServingEngine &engine)
        {
            // Unique per-request destinations, all MATERIALIZED by the producer.
            for (auto d : {kDest, kDestB, kDestC, kDestD})
            {
                engine.bankDestination(d, 1u << 20u);
                engine.materializeDestination(d);
            }
        }
    } // namespace

    TEST(SharedExecutionProof, AContinuesWhileBFinishesAndCEntersBSlot)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256);  // A
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 256);  // B
        engine.queueRequest(1, 2, 32, 8, 3, kDestC, 256); // C waits for B's slot (NewRequest)

        const auto step1 = engine.step();
        EXPECT_EQ(step1.at(0), windowSlice(1, 1, 16, 16)); // A == solo
        EXPECT_NE(step1.at(1).find("|entry:2"), std::string::npos); // C entered in the same step

        EXPECT_EQ(engine.rows().at(1).state.generation, 2u); // C owns the slot
        EXPECT_EQ(engine.rows().at(1).state.kind, AdmissionKind::NewRequest); // §14 C=NewRequest

        engine.issueNextGrant(0); // A continues (same request; single-use grant re-issued)
        const auto step2 = engine.step();
        EXPECT_EQ(step2.at(0), windowSlice(1, 17, 16, 16)); // A window 2 == solo
        EXPECT_EQ(step2.at(1), windowSlice(3, 1, 8, 8));    // C first slice
        EXPECT_TRUE(engine.rows().at(0).complete);
        // C receives its SECOND grant through the queued/active path and completes.
        engine.issueNextGrant(1);
        const auto step3 = engine.step();
        EXPECT_EQ(step3.at(1), windowSlice(3, 9, 8, 8)); // C second window (9-16)
        EXPECT_EQ(engine.rows().at(1).state.committed, 16u); // C limit reached
    }

    TEST(SharedExecutionProof, SlowConsumerDefersOnlyItself)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256);
        engine.admitNew(1, 1, 32, 16, 2, kDestB, 256);
        engine.throttle(1);
        const auto step1 = engine.step();
        EXPECT_TRUE(step1.count(0) == 1);
        EXPECT_TRUE(step1.count(1) == 0);
        EXPECT_EQ(engine.rows().at(0).state.committed, 16u);
        EXPECT_EQ(engine.rows().at(1).state.committed, 0u);
    }

    TEST(SharedExecutionProof, OccupiedSlotRefusesAdmission)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256);
        EXPECT_THROW(engine.admitNew(0, 3, 32, 16, 4, kDest, 256), std::logic_error);
    }

    TEST(SharedExecutionProof, FreshSlotQueuedRequestAdmits)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.queueRequest(3, 1, 32, 16, 7, kDest, 256); // never-populated slot
        const auto step1 = engine.step();
        EXPECT_EQ(engine.rows().at(3).state.generation, 1u);
        EXPECT_EQ(step1.at(3), windowSlice(7, 1, 16, 16));
        engine.issueNextGrant(3); // fresh-slot row gets its second grant too
        const auto step2 = engine.step();
        EXPECT_EQ(step2.at(3), windowSlice(7, 17, 16, 16));
        EXPECT_EQ(engine.rows().at(3).state.committed, 32u);
    }

    TEST(SharedExecutionProof, PartialFinalWindowIsBoundedAndTerminal)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 20, 16, 1, kDest, 256); // limit not a quantum multiple
        const auto step1 = engine.step();
        EXPECT_EQ(step1.at(0), windowSlice(1, 1, 16, 16));
        engine.issueNextGrant(0); // budget-backed: final 8 remaining allowed
        const auto step2 = engine.step();
        std::string partial;
        for (std::uint64_t pos = 17; pos <= 20; ++pos)
            partial += soloDecode(1, pos, 16, std::string{});
        EXPECT_EQ(step2.at(0), partial);
        EXPECT_EQ(engine.rows().at(0).state.committed, 20u);
        EXPECT_TRUE(engine.rows().at(0).complete);
        EXPECT_THROW(engine.issueNextGrant(0), std::logic_error); // completed row
    }

    TEST(SharedExecutionProof, RestoreReadinessGatesAdmission)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 256);
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 256);
        engine.queueRequest(1, 2, 16, 16, 3, kDestC, 4096); // prompt == restore geometry
        engine.beginQueuedRestore(1, 4096); // exact geometry
        // Not published -> Pending -> not admitted.
        const auto step1 = engine.step();
        EXPECT_EQ(engine.rows().at(1).state.generation, 1u);
        // Readiness is gated on a completed producer copy: minting without one refused.
        EXPECT_THROW(engine.issueRestoreReceipt(1, 2, 4096), std::logic_error);
        {
            std::array<std::byte, 4096> src{};
            for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<std::byte>(i & 0xFF);
            engine.completeRestoreCopy(1, engine.queuedRestoreSession(1), 4096, src.data());
            EXPECT_EQ(engine.restorePayload(1),
                      std::string(reinterpret_cast<const char *>(src.data()), src.size()));
        }
        {
            std::array<std::byte, 4096> src2{};
            EXPECT_THROW(engine.completeRestoreCopy(1, engine.queuedRestoreSession(1), 4096, src2.data()),
                         std::logic_error); // exactly once (no second copy source)
        }
        // A mismatched-byte receipt is refused at MINTING (producer gate).
        EXPECT_THROW(engine.issueRestoreReceipt(1, 2, 8192), std::logic_error);
        auto correct = engine.issueRestoreReceipt(1, 2, 4096);
        engine.publishRestore(1, std::move(correct));
        // The copied payload is restored INTO the materialized destination backing
        // (restored execution state), not left in an isolated lane.
        EXPECT_EQ(engine.destinationContents(kDestC).substr(0, 4096),
                  std::string(engine.restorePayload(1)));
        const auto step2 = engine.step();
        EXPECT_EQ(engine.rows().at(1).state.generation, 2u); // attached after readiness
        EXPECT_EQ(step2.at(1), windowSlice(3, 1, 16, 16));
        // Layout: [0,4096) is restored prompt, output is written AFTER it (no
        // overwrite of the restored region).
        EXPECT_EQ(engine.destinationContents(kDestC).substr(0, 4096),
                  engine.restorePayload(1));
        EXPECT_EQ(engine.destinationContents(kDestC).substr(4096, step2.at(1).size()),
                  step2.at(1));
        // Delivery THROUGH publication: the consumer receives ONLY the committed
        // output (never the 4096 restored prompt bytes).
        std::string delivered_bytes;
        engine.setConsumer([&](std::uint64_t, const std::byte *p, std::size_t n)
                           {
                               delivered_bytes.assign(reinterpret_cast<const char *>(p), n);
                           });
        engine.markDelivered(kDestC, 1);
        EXPECT_EQ(delivered_bytes.size(), step2.at(1).size());
        EXPECT_EQ(delivered_bytes, step2.at(1)); // BYTES match the output slice
        EXPECT_EQ(engine.restoredExtentFor(kDestC), 4096u);
    }

    TEST(SharedExecutionProof, HarvestHoldsSourceUntilOwnedReceiptReleases)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256);
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 256);
        engine.queueRequest(1, 2, 16, 8, 3, kDestC, 256);
        engine.requestHarvest(1); // before B completes
        const auto step1 = engine.step();
        EXPECT_EQ(engine.rows().at(1).state.generation, 1u);   // held
        EXPECT_EQ(engine.queued().count(1), 1u);               // C waits
        // Frozen-source / harvest-requested slot cannot be overwritten:
        EXPECT_THROW(engine.admitNew(1, 9, 16, 8, 9, kDest, 256), std::logic_error);
        engine.freezeSnapshot(1);
        engine.beginSnapshotCopy(1);
        auto receipt = engine.borrowSnapshot(1);
        EXPECT_THROW(engine.admitNew(1, 9, 16, 8, 9, kDest, 256), std::logic_error); // not released yet
        receipt.release(); // authenticating receipt releases the source
        engine.issueNextGrant(0);
        const auto step2 = engine.step();
        EXPECT_EQ(engine.rows().at(1).state.generation, 2u); // C attached after release
        EXPECT_EQ(step2.at(1), windowSlice(3, 1, 8, 8));
    }

    TEST(SharedExecutionProof, MixedQuantaCancellationAndMTPInOneBatch)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 8, 1, kDest, 256);
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 256);
        engine.setMTPDepth(1, 2);
        engine.queueRequest(1, 2, 16, 8, 3, kDestC, 256);
        engine.cancelQueued(1); // C cancelled before B finishes
        engine.queueRequest(1, 3, 16, 8, 4, kDestD, 256); // D replaces C (unique reservation)

        const auto step1 = engine.step();
        EXPECT_EQ(step1.at(0), windowSlice(1, 1, 8, 8));
        EXPECT_NE(step1.at(1).find("|entry:3"), std::string::npos);
        EXPECT_EQ(engine.rows().at(1).state.generation, 3u);

        engine.setMTPDepth(1, 2); // D admitted in step1; depth before its production
        engine.issueNextGrant(0);
        const auto step2 = engine.step();
        EXPECT_EQ(step2.at(0), windowSlice(1, 9, 8, 8));
        EXPECT_EQ(step2.at(1), windowSlice(4, 1, 8, 8));
        EXPECT_THROW(engine.setMTPDepth(1, 2), std::logic_error); // exactly once
        EXPECT_THROW(engine.setMTPDepth(0, 2), std::logic_error); // after execution: immutable
    }

    TEST(SharedExecutionProof, GrantIsSingleUseAndBacked)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        // Unbacked admission is refused by the authority.
        EXPECT_THROW(engine.admitNew(0, 1, 16, 16, 1, 0), std::logic_error);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256); // not complete after one window
        // A second issue while one is outstanding (cross-window) is refused.
        EXPECT_THROW(engine.issueNextGrant(0), std::logic_error);
        const auto step1 = engine.step(); // consumes the grant
        EXPECT_EQ(step1.count(0), 1u);
        engine.issueNextGrant(0); // re-issue after consumption (new ID)
        EXPECT_TRUE(engine.rowMayExecute(0));
    }

    TEST(SharedExecutionProof, MTPVerifierIsExactlyOncePerStepAndRequiresDepth)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256);
        // Verification requires produced work AND positive depth.
        EXPECT_THROW(engine.verifyMTP(0), std::logic_error); // no window yet
        engine.setMTPDepth(0, 2);
        const auto step1 = engine.step(); // produces + verifies once internally
        EXPECT_EQ(step1.count(0), 1u);
        EXPECT_THROW(engine.verifyMTP(0), std::logic_error); // already verified this step
        engine.issueNextGrant(0);
        const auto step2 = engine.step(); // fresh step: resets, verifies exactly once
        EXPECT_EQ(step2.count(0), 1u);
        EXPECT_THROW(engine.verifyMTP(0), std::logic_error); // exactly once per step
    }

    TEST(SharedExecutionProof, RestoreReceiptGenerationAndIdentityAreAuthenticated)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 256);
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 256);
        engine.queueRequest(1, 2, 16, 16, 3, kDestC, 4096);
        engine.beginQueuedRestore(1, 4096);
        // Readiness is gated on a completed producer copy (foreign too).
        EXPECT_THROW(engine.issueRestoreReceipt(1, 1, 4096), std::logic_error);
        {
            std::array<std::byte, 4096> src{};
            engine.completeRestoreCopy(1, engine.queuedRestoreSession(1), 4096, src.data());
        }
        // Stale generation refused AT MINTING (generation validated by the issuer).
        EXPECT_THROW(engine.issueRestoreReceipt(1, 1, 4096), std::logic_error);
        GroupedServingEngine other(solo);
        bankDest(other);
        other.admitNew(0, 1, 16, 16, 2, kDestC, 256);
        other.queueRequest(1, 2, 16, 16, 3, kDest, 4096);
        other.beginQueuedRestore(1, 4096);
        {
            std::array<std::byte, 4096> src{};
            other.completeRestoreCopy(1, other.queuedRestoreSession(1), 4096, src.data());
        }
        auto foreign = other.issueRestoreReceipt(1, 2, 4096); // other engine
        EXPECT_THROW(engine.publishRestore(1, std::move(foreign)), std::logic_error);
        auto correct = engine.issueRestoreReceipt(1, 2, 4096);
        engine.publishRestore(1, std::move(correct));
        const auto step = engine.step();
        EXPECT_EQ(engine.rows().at(1).state.generation, 2u); // attached
    }

    TEST(SharedExecutionProof, RecordedControllerTraceReplaysIdentically)
    {
        // Device-free recorded-input controller replay through the ACTUAL
        // implementation: the trace (admits, queue, issue, cancel, harvest) is
        // recorded once, then replayed on a fresh engine and must reproduce the
        // identical per-step transcripts and lifecycle state.
        struct Step
        {
            enum class Op { Admit, Queue, Issue, Cancel, Harvest, Step } op;
            std::uint32_t slot = 0;
        };
        // A VALID controller trace (cancel of an already-admitted row is its own
        // regression): admit A, queue C, step, issue A's next grant, step.
        // Recorded controller trace: A admitted; C queued; step; both next grants;
        // two more steps (C completes through its SECOND queued-path grant).
        const std::vector<Step> trace = {
            {Step::Op::Admit, 0}, {Step::Op::Queue, 1},
            {Step::Op::Step, 0}, {Step::Op::Issue, 0}, {Step::Op::Issue, 1},
            {Step::Op::Step, 0}, {Step::Op::Issue, 0}, {Step::Op::Issue, 1},
            {Step::Op::Step, 0}, {Step::Op::Issue, 1}, {Step::Op::Step, 0},
        };
        auto run = [&](const std::vector<Step> &tr) {
            GroupedServingEngine engine(solo);
        bankDest(engine);
            std::vector<std::string> transcript;
            for (const auto &st : tr)
            {
                switch (st.op)
                {
                case Step::Op::Admit: engine.admitNew(0, 1, 48, 16, 1, kDest, 256); break; // 3 A windows
                case Step::Op::Queue: engine.queueRequest(1, 2, 32, 8, 3, kDestC, 256); break;
                case Step::Op::Issue: engine.issueNextGrant(st.slot); break;
                case Step::Op::Cancel: engine.cancelQueued(st.slot); break;
                case Step::Op::Step:
                    for (const auto &[slot, out] : engine.step())
                        transcript.push_back(std::to_string(slot) + ":" + out);
                    break;
                }
            }
            return transcript;
        };
        const auto first = run(trace);
        // ORACLE: pinned position-by-position. A: seed1 q16 -> [1-16],[17-32],[33-48]
        // (3 windows). C: seed3 q8 -> completes its full 32-token limit over FOUR
        // q8 windows [1-8],[9-16],[17-24],[25-32] through queued-path grants.
        const std::string a1 = windowSlice(1, 1, 16, 16);
        const std::string a2 = windowSlice(1, 17, 16, 16);
        const std::string c1 = windowSlice(3, 1, 8, 8);
        const std::string c2 = windowSlice(3, 9, 8, 8);
        const std::string c3 = windowSlice(3, 17, 8, 8);
        const std::string c4 = windowSlice(3, 25, 8, 8);
        const std::string a3 = windowSlice(1, 33, 16, 16);
        const std::vector<std::string> expected = {
            "0:" + a1, "1:" + c1,
            "0:" + a2, "1:" + c2,
            "0:" + a3, "1:" + c3,
            "1:" + c4,
        };
        ASSERT_EQ(expected.size(), 7u); // A3 + C4 windows
        ASSERT_EQ(first.size(), expected.size());
        EXPECT_EQ(first[0], expected[0]);
        EXPECT_EQ(first[1], expected[1]);
        EXPECT_EQ(first[2], expected[2]);
        EXPECT_EQ(first[3], expected[3]);
        EXPECT_EQ(first[4], expected[4]); // A's THIRD window at index 4 (was unchecked)
        EXPECT_EQ(first[5], expected[5]); // C's third window (was unchecked)
        EXPECT_EQ(first[6], expected[6]); // C's fourth window (was unchecked)
        // Replay on a fresh engine: byte-identical transcription.
        const auto second = run(trace);
        ASSERT_EQ(second, first);
    }

    TEST(SharedExecutionProof, SnapshotLifetimeFocusedRegressions)
    {
        // (a) A LIVE receipt must survive engine destruction: the lease is owned
        // by the receipt (no raw engine pointer), so release after the engine is
        // gone never touches a destroyed engine or its mutex.
        std::unique_ptr<GroupedServingEngine::SnapshotBorrowReceipt> survivor;
        {
            GroupedServingEngine engine(solo);
            bankDest(engine);
            engine.admitNew(0, 1, 32, 16, 1, kDest, 256);
            engine.admitNew(1, 1, 16, 16, 2, kDestB, 256);
            engine.queueRequest(1, 2, 16, 8, 3, kDestC, 256);
            engine.requestHarvest(1);
            engine.step();
            engine.freezeSnapshot(1);
            engine.beginSnapshotCopy(1);
            survivor = std::make_unique<GroupedServingEngine::SnapshotBorrowReceipt>(
                engine.borrowSnapshot(1));
            // Overlap guards AFTER the first borrow: a second borrow and a second
            // copy are both refused (no overlapping borrows / receipt invalidation).
            EXPECT_THROW(engine.borrowSnapshot(1), std::logic_error);
            EXPECT_THROW(engine.beginSnapshotCopy(1), std::logic_error);
        }
        // Engine destroyed; the survivor releases its own lease safely.
        EXPECT_NO_THROW(survivor->release());
        EXPECT_NO_THROW(survivor->release()); // idempotent
    }

    TEST(SharedExecutionProof, MoveAssignReleasesPriorReceiptRow)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256);
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 256);
        engine.admitNew(2, 1, 16, 16, 4, kDestC, 256); // second harvestable row
        engine.requestHarvest(1);
        engine.requestHarvest(2);
        engine.step();
        engine.freezeSnapshot(1);
        engine.beginSnapshotCopy(1);
        auto r1 = engine.borrowSnapshot(1);
        engine.freezeSnapshot(2); // completed in the same step? row2 limit16 -> complete
        engine.beginSnapshotCopy(2);
        auto r2 = engine.borrowSnapshot(2);
        // Move-assign r2 over r1: the prior (r1) row must be RELEASED.
        r2 = std::move(r1); // move-assign: releases r2's prior (row2) lease FIRST
        EXPECT_TRUE(engine.rowSnapshotReleased(2)); // row2 observed released
        EXPECT_FALSE(engine.rowSnapshotReleased(1)); // row1 still held by the receipt
        r2.release();       // releases row1's lease
        EXPECT_TRUE(engine.rowSnapshotReleased(1));
        engine.issueNextGrant(0);
        const auto step = engine.step();
        (void)step;
    }

    TEST(SharedExecutionProof, ReleaseIsGatedOnDeliveryNotExecution)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 256);
        engine.step(); // A completes (execution done)
        // Execution done but delivery pending: backing NOT released yet.
        EXPECT_FALSE(engine.destinationReleased(kDest));
        // PUBLIC release before delivery is refused (delivery ledger gate).
        EXPECT_THROW(engine.releaseDestination(kDest), std::logic_error);
        engine.markDelivered(kDest, 1); // delivery completes
        engine.step();           // next step runs the post-pass release
        EXPECT_TRUE(engine.destinationReleased(kDest));
        // ONE more step: the retained backing is REALLY released (markReleased ran),
        // so no second release is attempted and nothing throws.
        EXPECT_NO_THROW(engine.step());
    }

    TEST(SharedExecutionProof, PartialFinalWindowAcceptsQuantumBeyondLimit)
    {
        // §7.1: L=5, W=16 is a VALID partial final window (quantum > limit must not
        // be refused; only quantum == 0 is invalid).
        GroupedServingEngine engine(solo);
        bankDest(engine);
        EXPECT_THROW(engine.admitNew(0, 1, 5, 0, 1, kDest, 64), std::logic_error);
        EXPECT_NO_THROW(engine.admitNew(0, 1, 5, 16, 1, kDest, 64));
        EXPECT_THROW(engine.queueRequest(1, 2, 5, 0, 3, kDestC, 64), std::logic_error);
        EXPECT_NO_THROW(engine.queueRequest(1, 2, 5, 16, 3, kDestC, 64));
        engine.step();
        // quotient 0 -> the +2 floor applies; the authority is request-bounded.
        EXPECT_EQ(engine.grantBound(0), 2u);
    }

    TEST(SharedExecutionProof, SameInstanceBoundMoveCarriesIncomingMax)
    {
        // The queued-attach wait: A holds a SMALL bound (3); C's larger bound
        // (14) must survive the same-instance move-assign, not A's stale 3.
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 0, kDest, 64);        // A: bound 16/16+2 = 3
        engine.queueRequest(0, 2, 200, 16, 0, kDestC, 64);   // C: bound 200/16+2 = 14
        engine.step(); // A completes AND C attaches into slot 0 in the same step
                        // (completion post-pass); the attach move-assign merges
                        // C's bound over A's authority. A stale 3 would remain if
                        // the same-instance move did not carry the incoming bound.
        EXPECT_EQ(engine.grantBound(0), (200 / 16) + 2); // moved bound 14, NOT stale 3
    }

    TEST(SharedExecutionProof, LateBDeliveryReleasesOwnBackingWithoutStallingC)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);   // A slot 0
        engine.admitNew(1, 2, 16, 16, 2, kDestB, 64);  // B slot 1
        engine.step(); // both complete (slots retired)
        engine.queueRequest(1, 3, 32, 16, 3, kDestC, 64); // C REUSES B's slot 1; L=32/W=16
        engine.step(); // C attaches INTO B's slot and completes its FIRST window only
                       // (committed 16 < 32): C is MID-FLIGHT, not complete.
        EXPECT_FALSE(engine.rowComplete(1));
        // B delivers only AFTER C reused its slot AND is still executing: the
        // delivery is keyed by the RESPONSE reservation, so it must hit B, never C.
        engine.markDelivered(kDestB, 1);
        engine.issueNextGrant(1); // C's second window (backpressure: one per grant)
        engine.step(); // post-pass releases B's OWN backing while C still executes
                       // (C produces its second window in the same step).
        EXPECT_TRUE(engine.destinationReleased(kDestB));
        EXPECT_FALSE(engine.destinationReleased(kDestC)); // C is not delivered yet
        EXPECT_TRUE(engine.rowComplete(1));                // and C was NOT stalled
        EXPECT_THROW(engine.markDelivered(kDestB, 1), std::logic_error); // exactly once
        // A still delivers normally (slot 0 untouched)
        engine.markDelivered(kDest, 1);
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDest));
        // C delivers its own reservation afterwards
        engine.markDelivered(kDestC, 1);
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDestC));
    }

    TEST(SharedExecutionProof, GrantLedgerSaturatesNotCapsAtDefault)
    {
        // L=4097/q1 needs a 4099 grant bound; the old kDefaultMaxGrants cap would
        // fail the 4097th grant mid-request. Saturation must NOT cap at 4096.
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 4097, 1, 1, kDest, 64);
        EXPECT_EQ(engine.grantBound(0), (4097 / 1) + 2);
        // admitNew already issues the FIRST grant (id 1); consume it, then each
        // further loop iteration issues+consumes one more (ids 2..4096).
        EXPECT_NO_THROW(engine.step());
        for (int i = 1; i < 4096; ++i)
        {
            EXPECT_NO_THROW(engine.issueNextGrant(0)) << "issue at i=" << i;
            EXPECT_NO_THROW(engine.step()) << "step at i=" << i;
        }
        EXPECT_EQ(engine.grantBound(0), (4097 / 1) + 2);
        EXPECT_NO_THROW(engine.issueNextGrant(0)); // the 4097th grant is legal
        engine.step();
    }

    TEST(SharedExecutionProof, DestinationReservationsAreExclusive)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);
        // An active row already owns kDest: re-claim refused (active AND queued).
        EXPECT_THROW(engine.admitNew(1, 2, 16, 16, 2, kDest, 64), std::logic_error);
        EXPECT_THROW(engine.queueRequest(1, 2, 16, 16, 2, kDest, 64), std::logic_error);
    }

    TEST(SharedExecutionProof, PrematureDeliveryIsRefused)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 64); // still executing (32 tokens, one step = 16)
        EXPECT_THROW(engine.markDelivered(kDest, 1), std::logic_error); // not completed yet
    }



    TEST(SharedExecutionProof, ResponseOwnershipOutlivesRowRemoval)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);   // A
        engine.admitNew(1, 2, 16, 16, 2, kDestB, 64);  // B
        engine.step(); // both complete
        engine.queueRequest(1, 3, 16, 16, 3, kDestD, 64); // C (kDestD) reuses B's slot
        engine.step(); // C replaces B's row; B's response is STILL undelivered
        // B's ownership must outlive the execution-row removal: D cannot claim B.
        EXPECT_THROW(engine.admitNew(2, 4, 16, 16, 4, kDestB, 64), std::logic_error);
        EXPECT_THROW(engine.queueRequest(2, 4, 16, 16, 4, kDestB, 64), std::logic_error);
        // After B delivers and its backing retires, the ID may be claimed again by
        // a FRESH response (bounded identity reuse; the re-bank bumps generation).
        engine.markDelivered(kDestB, 1);
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDestB));
        // Bounded identity REUSE: the released ID may be re-banked (generation
        // bumped) and claimed by a fresh response.
        engine.bankDestination(kDestB, 1u << 20u);
        engine.materializeDestination(kDestB);
        EXPECT_NO_THROW(engine.admitNew(2, 4, 16, 16, 4, kDestB, 64));
    }

    TEST(SharedExecutionProof, DestinationPoolIsBoundedAndIDsRebankWithGeneration)
    {
        // Canonical bounded identity: a fixed-size wheel (pool full refused), and a
        // RETIRED ID may be re-banked (bounded reuse, NO unbounded per-ID memory);
        // the re-bank bumps the ID's generation so a stale response can never act
        // on the new occupant (the claim/delivery gates are generation-clean).
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);
        engine.step();
        engine.markDelivered(kDest, 1);
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDest));
        EXPECT_NO_THROW(engine.bankDestination(5, 1u << 20u)); // fresh wheel slot
        // Bounded wheel: the pool width comes from the ADMITTED PLAN (default 32),
        // never from a hard-coded HTTP constant; the wheel cannot exceed it.
        GroupedServingEngine full(solo);
        for (std::uint64_t id = 1; id <= engine.maxDestinations(); ++id)
            full.bankDestination(id, 1u << 20u);
        EXPECT_THROW(full.bankDestination(engine.maxDestinations() + 1, 1u << 20u),
                     std::logic_error);
        // A plan-derived width (4) rejects slot 5; width is not copied from HTTP.
        GroupedServingEngine narrow(solo, 1, GroupedServingEngine::ServingPoolPlan{4});
        EXPECT_NO_THROW(narrow.bankDestination(4, 1u << 20u));
        EXPECT_THROW(narrow.bankDestination(5, 1u << 20u), std::logic_error);
        EXPECT_EQ(narrow.maxDestinations(), 4u);
    }

    TEST(SharedExecutionProof, LiveOutputBoundUsesRealProducedBytes)
    {
        // seed 96 / L16 / W16 / prompt 0: the decode function emits MORE than the
        // assumed 8 bytes/token (real output 135 bytes > 128 assumed).
        GroupedServingEngine engine(solo);
        engine.bankDestination(kDest, 128);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 1, 16, 16, 96, kDest, 0);
        // Admission (the typed MINIMUM 8 bytes/token) fits; the ATOMIC publication
        // boundary validates REAL produced bytes (135) BEFORE any mutation.
        EXPECT_THROW(engine.step(), std::logic_error);
        EXPECT_TRUE(engine.failed());
        // The failed response was never completed: delivery is refused (no
        // accepted-failed response) and the engine stays dead (no stranded grant).
        EXPECT_THROW(engine.markDelivered(kDest, 1), std::logic_error);
        EXPECT_THROW(engine.step(), std::logic_error);
    }

    TEST(SharedExecutionProof, OutputIsPhysicallyWrittenToTheDestinationBacking)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);
        const auto step1 = engine.step();
        const auto slice = step1.at(0);
        // The produced output bytes are physically present in the materialized
        // destination buffer AFTER the prompt region (output offset = prompt bytes).
        EXPECT_NE(engine.destinationContents(kDest).find(slice), std::string::npos);
    }

    TEST(SharedExecutionProof, QueuedSaturationPathUsesRequestBoundedLedger)
    {
        // The queued path must also carry the request-derived bound (not 4096):
        // L=4097/q1 through a QUEUED request.
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);       // A (small, completes)
        engine.queueRequest(0, 2, 4097, 1, 1, kDestC, 64);  // C queued, L=4097/q1
        engine.step(); // A completes; C attaches (move-ASSIGN) with bound 4099
        EXPECT_EQ(engine.grantBound(0), (4097 / 1) + 2);
        EXPECT_NO_THROW(engine.step()); // consume C's admission grant (token 1)
        for (int i = 1; i < 4096; ++i)
        {
            EXPECT_NO_THROW(engine.issueNextGrant(0)) << "queued issue at i=" << i;
            EXPECT_NO_THROW(engine.step()) << "queued step at i=" << i;
        }
        EXPECT_NO_THROW(engine.issueNextGrant(0)); // grant 4097
        EXPECT_NO_THROW(engine.step());
        EXPECT_TRUE(engine.rowComplete(0));
    }

    TEST(SharedExecutionProof, RestoreFeedsExecutionState)
    {
        // A decoder that ACTUALLY READS the restored prefix: proves the restored
        // bytes reach execution state (not an isolated lane).
        static std::string last_restored; // decoder observation
        const RowDecodeFn readingDecode = [](
            std::uint64_t seed, std::uint64_t position, std::uint64_t window,
            const std::string &restored)
        {
            last_restored = restored;
            return "r" + std::to_string(seed) + ":" + std::to_string(position) + "/" +
                   std::to_string(window) + ":" + std::to_string(restored.size());
        };
        GroupedServingEngine engine(readingDecode);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);
        engine.admitNew(1, 2, 16, 16, 2, kDestB, 64);
        engine.queueRequest(1, 3, 16, 16, 3, kDestC, 4096); // prompt == restore geometry
        engine.beginQueuedRestore(1, 4096);
        {
            std::array<std::byte, 4096> src{};
            for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<std::byte>(i & 0xFF);
            engine.completeRestoreCopy(1, engine.queuedRestoreSession(1), 4096, src.data());
            auto correct = engine.issueRestoreReceipt(1, 3, 4096);
            engine.publishRestore(1, std::move(correct));
        }
        engine.step(); // C attaches (move-assign); its decoder runs WITH the prefix
        last_restored.clear();
        engine.step(); // C executes a window using the restored prefix
        EXPECT_EQ(last_restored.size(), 4096u); // decoder read 4096 restored bytes
        EXPECT_NE(engine.rows().at(1).state.committed, 0u);
        (void)windowSlice(0, 0, 0, 16);
    }

    TEST(SharedExecutionProof, ConsumerReceivesPublishedBytesOnce)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        std::size_t calls = 0;
        std::size_t total = 0;
        std::uint64_t seen_reservation = 0;
        engine.setConsumer([&](std::uint64_t reservation, const std::byte *, std::size_t n)
                           {
                               ++calls; total += n; seen_reservation = reservation;
                           });
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);
        const auto step1 = engine.step();
        engine.markDelivered(kDest, 1);
        EXPECT_EQ(calls, 1u);
        EXPECT_EQ(seen_reservation, kDest);
        // The consumer receives the COMMITTED OUTPUT ONLY (§5.1/§10.1): the
        // produced window — never the unwritten prompt prefix, never holes,
        // never allocation capacity.
        EXPECT_EQ(total, step1.at(0).size());
        engine.step();
        EXPECT_EQ(calls, 1u); // exactly once, no second publication
    }

    TEST(SharedExecutionProof, StaleGenerationDeliveryIsRefusedAfterRebank)
    {
        // B occupies slot 2 (generation 1), retires; D re-banks slot 2
        // (generation 2), completes. An OLD-B duplicate calling delivery with
        // generation 1 must be REFUSED — the ABA is closed by generation keys.
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);   // A (slot 1)
        engine.admitNew(1, 2, 16, 16, 2, kDestB, 64);  // B (slot 2, gen 1)
        engine.step();
        engine.markDelivered(kDestB, 1);
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDestB));
        EXPECT_EQ(engine.completedGeneration(kDestB), 0u); // release erased the record
        EXPECT_EQ(engine.destinationGeneration(kDestB), 1u);
        engine.bankDestination(kDestB, 1u << 20u); // re-bank: slot 2 -> generation 2
        engine.materializeDestination(kDestB);
        EXPECT_NO_THROW(engine.admitNew(1, 3, 16, 16, 3, kDestB, 64)); // D (gen 2)
        engine.step();
        EXPECT_TRUE(engine.rowComplete(1)); // D completed in one window
        EXPECT_EQ(engine.destinationGeneration(kDestB), 2u);
        EXPECT_EQ(engine.completedGeneration(kDestB), 2u);
        // The stale B duplicate cannot deliver D's completed response.
        EXPECT_THROW(engine.markDelivered(kDestB, 1), std::logic_error);
        EXPECT_NO_THROW(engine.markDelivered(kDestB, 2)); // D's own delivery works
    }

    TEST(SharedExecutionProof, ThrowingConsumerLeavesResponseUndelivered)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        bool throw_once = true;
        engine.setConsumer([&](std::uint64_t, const std::byte *, std::size_t)
                           {
                               if (throw_once) { throw_once = false; throw std::runtime_error("transport"); }
                           });
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);
        engine.step();
        // A throwing consumer means the delivery NEVER completed: no ledger
        // record, backing retained, retryable.
        EXPECT_THROW(engine.markDelivered(kDest, 1), std::runtime_error);
        EXPECT_FALSE(engine.destinationReleased(kDest));
        engine.step();
        EXPECT_FALSE(engine.destinationReleased(kDest)); // still not freed
        EXPECT_NO_THROW(engine.markDelivered(kDest, 1)); // retry succeeds
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDest));
    }

    TEST(SharedExecutionProof, FailedEngineRefusesNewAdmissionsAndGrants)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine); // all four slots banked+materialized (VALID operations)
        // Row 1 is a GOOD row that consumes its admission grant first (resolved),
        // so issueNextGrant(1) is legal BEFORE the failure gate exists.
        engine.admitNew(1, 1, 32, 16, 1, kDestB, 64);
        engine.step();
        EXPECT_FALSE(engine.rowComplete(1));
        // Bad row (capacity 128 provokes 135 > 128) poisons the engine.
        engine.bankDestination(5, 128);
        engine.materializeDestination(5);
        engine.admitNew(0, 1, 16, 16, 96, 5, 0);
        EXPECT_THROW(engine.step(), std::logic_error);
        EXPECT_TRUE(engine.failed());
        // kDestC / kDestD are banked, materialized and UNCLAIMED: without the
        // failed gate these VALID operations would succeed; they must refuse.
        EXPECT_THROW(engine.admitNew(2, 1, 16, 16, 1, kDestC, 64), std::logic_error);
        EXPECT_THROW(engine.queueRequest(2, 2, 16, 16, 2, kDestD, 64), std::logic_error);
        // Row 1's grant is RESOLVED and the row is unfinished: without the gate
        // issueNextGrant(1) would be legal; it must refuse.
        EXPECT_THROW(engine.issueNextGrant(1), std::logic_error);
    }

    TEST(SharedExecutionProof, FailedAdmissionLeavesNoStrandedClaim)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        // A fresh wheel slot banked WITHOUT materialization: admission refused.
        engine.bankDestination(6, 1u << 20u);
        EXPECT_THROW(engine.admitNew(0, 1, 16, 16, 1, 6, 64), std::logic_error);
        // Materialize and RETRY THE SAME reservation: must succeed (no claim left).
        engine.materializeDestination(6);
        EXPECT_NO_THROW(engine.admitNew(0, 1, 16, 16, 1, 6, 64));
        engine.step();
        EXPECT_TRUE(engine.rowComplete(0));
    }

    TEST(SharedExecutionProof, MultiRowAdverseStepMutatesNothingOnViolation)
    {
        // Row 1 is GOOD and mid-flight; row 3 violates the real-bytes bound. The
        // step validates ALL executable rows (and the queued attach candidate)
        // BEFORE mutating ANY, so the good row is untouched, the attach is NOT
        // applied, and no response was half-completed.
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);   // good: completes in one window
        engine.admitNew(1, 1, 32, 16, 1, kDestB, 64);  // good: mid-flight L32/W16
        engine.queueRequest(0, 2, 16, 16, 3, kDestC, 64); // attach candidate (slot 0)
        engine.step(); // row0 completes; row1 -> first window (committed 16)
        EXPECT_EQ(engine.rows().at(1).state.committed, 16u);
        // C attached into slot 0 in the same step (completion post-pass) but has
        // NOT executed yet (attachment happens after the compute loop).
        EXPECT_EQ(engine.rows().at(0).state.generation, 2u);
        EXPECT_EQ(engine.rows().at(0).state.committed, 0u);
        // Violating row (capacity 128; seed 96 emits 135 > 128).
        engine.bankDestination(5, 128);
        engine.materializeDestination(5);
        engine.admitNew(2, 1, 16, 16, 96, 5, 0);
        EXPECT_THROW(engine.step(), std::logic_error);
        EXPECT_TRUE(engine.failed());
        // Nothing was mutated by the ADVERSE step: row1 not advanced past its
        // previous window, the freshly attached C (slot 0) not advanced, and the
        // violating row (slot 2) not completed.
        EXPECT_EQ(engine.rows().at(1).state.committed, 16u);
        EXPECT_FALSE(engine.rows().at(1).complete);
        EXPECT_EQ(engine.rows().at(0).state.generation, 2u); // C attached earlier
        EXPECT_EQ(engine.rows().at(0).state.committed, 0u);  // but NOT advanced
        EXPECT_FALSE(engine.rows().at(0).complete);
        EXPECT_FALSE(engine.rows().at(2).complete);
        EXPECT_EQ(engine.rows().at(2).state.committed, 0u);
    }

    TEST(SharedExecutionProof, RetirementDuringInProgressPublicationIsRefused)
    {
        GroupedServingEngine engine(solo);
        bankDest(engine);
        bool inside = false;
        engine.setConsumer([&](std::uint64_t, const std::byte *, std::size_t)
                           {
                               inside = true;
                               // A consumer must never retire the buffer it is
                               // still borrowing: refused while Publishing.
                               EXPECT_THROW(engine.releaseDestination(kDest),
                                            std::logic_error);
                           });
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);
        engine.step();
        EXPECT_NO_THROW(engine.markDelivered(kDest, 1)); // consumer accepted
        EXPECT_TRUE(inside);
        // After acceptance the retirement is legal (exactly once).
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDest));
    }

    TEST(SharedExecutionProof, AdverseStepWithPendingAttachAndFreshGrant)
    {
        // A has a FRESH outstanding grant (mid-flight), C is a PENDING attach
        // candidate, and the violating row poisons the step: the fresh grant must
        // NOT be consumed and C must NOT attach.
        GroupedServingEngine engine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 64);   // A: L32/W16
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 64);  // B: completes in one window
        engine.queueRequest(1, 2, 16, 16, 3, kDestC, 64); // C (attach after B)
        engine.step(); // B completes AND C attaches in the same step (post-pass)
        // Give A a FRESH grant (mid-flight granted) and queue a NEW pending attach
        // for the just-completed B slot? B's slot now holds C (not complete).
        // Use slot 2: complete a fresh row to create a pending attach.
        engine.admitNew(2, 1, 16, 16, 4, kDestD, 64); // D: completes in one window
        engine.step();                                 // D completes
        engine.bankDestination(7, 1u << 20u); // E's own fresh reservation
        engine.materializeDestination(7);
        engine.queueRequest(2, 3, 16, 16, 5, 7, 64); // E: attach candidate (slot 2)
        EXPECT_EQ(engine.queued().count(2), 1u);
        engine.issueNextGrant(0);                      // A: fresh outstanding grant
        // Violating row on slot 3 (capacity 128; seed 96 emits 135 > 128).
        engine.bankDestination(6, 128);
        engine.materializeDestination(6);
        engine.admitNew(3, 1, 16, 16, 96, 6, 0);
        EXPECT_THROW(engine.step(), std::logic_error);
        EXPECT_TRUE(engine.failed());
        // A's fresh grant was NOT consumed (committed unchanged), E did NOT attach,
        // and the violating row is untouched.
        EXPECT_EQ(engine.rows().at(0).state.committed, 16u);
        EXPECT_TRUE(engine.rowMayExecute(0)); // grant still outstanding
        EXPECT_EQ(engine.queued().count(2), 1u);
        EXPECT_EQ(engine.rows().at(2).state.generation, 1u); // D, not E
        EXPECT_FALSE(engine.rows().at(3).complete);
    }

    TEST(SharedExecutionProof, InConsumerStepAdvancesCAndRetainsBWhilePublishing)
    {
        // B's consumer runs an INDEPENDENT grouped step while B is Publishing:
        // C computes/commits/consumes its grant (correct progress bytes), B's
        // backing is RETAINED (never auto-retired mid-publication), and B retires
        // only after acceptance. C is NOT stalled behind B's delivery.
        GroupedServingEngine engine(solo);
        bankDest(engine);
        bool inside = false;
        std::string c_bytes;
        engine.setConsumer([&](std::uint64_t, const std::byte *, std::size_t)
                           {
                               if (inside)
                                   return;
                               inside = true;
                               // Re-entrant step from inside the consumer: C must
                               // advance normally (its returned bytes correct) and
                               // B's backing must NOT be released.
                               const auto inner = engine.step();
                               c_bytes = inner.at(2);
                               EXPECT_EQ(c_bytes, windowSlice(4, 17, 16, 16));
                               EXPECT_FALSE(engine.destinationReleased(kDestB));
                               inside = false;
                           });
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 64); // B: slot 1 (completes step 1)
        engine.admitNew(2, 1, 32, 16, 4, kDestC, 64); // C: slot 2 (MID-FLIGHT L32)
        engine.step();                                // B completes; C -> first window
        EXPECT_EQ(engine.rows().at(2).state.committed, 16u);
        EXPECT_FALSE(engine.rows().at(2).complete);
        engine.issueNextGrant(2); // C's second window is pending (backpressure)
        engine.markDelivered(kDestB, 1);              // B delivery -> consumer runs
        EXPECT_FALSE(engine.destinationReleased(kDestB)); // retained while Publishing
        // After acceptance B retires; C was NOT stalled and completed its second
        // window during the in-consumer step.
        EXPECT_EQ(engine.rows().at(2).state.committed, 32u);
        EXPECT_TRUE(engine.rows().at(2).complete);
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDestB));
        EXPECT_FALSE(engine.destinationReleased(kDestC)); // C not delivered yet
    }
} // namespace llaminar2::test
