/**
 * @file Test__SSEPublishChannel.cpp
 * @brief Device-free proofs for the bounded SSE publish channel (§10.1).
 */
#include <gtest/gtest.h>

#include "app/modes/SSEPublishChannel.h"

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

namespace
{
    bool wait_for_true(const std::atomic<bool> &flag,
                       std::chrono::milliseconds budget = std::chrono::seconds(5))
    {
        const auto end = std::chrono::steady_clock::now() + budget;
        while (!flag.load())
        {
            if (std::chrono::steady_clock::now() >= end)
                return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return true;
    }

    template<class Predicate>
    bool wait_for_true(Predicate predicate,
                       std::chrono::milliseconds budget = std::chrono::seconds(5))
    {
        const auto end = std::chrono::steady_clock::now() + budget;
        while (!predicate())
        {
            if (std::chrono::steady_clock::now() >= end)
                return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return true;
    }
} // namespace


using namespace llaminar2;

TEST(SSEPublishChannel, BoundedProducerConsumerRoundTrip)
{
    SSEPublishChannel channel(4);
    for (int i = 0; i < 3; ++i)
        EXPECT_TRUE(channel.push("line" + std::to_string(i)));
    EXPECT_EQ(channel.size(), 3u);
    EXPECT_EQ(channel.popOrClosed().value(), "line0");
    EXPECT_EQ(channel.popOrClosed().value(), "line1");
    EXPECT_FALSE(channel.closed());
    channel.close();
    EXPECT_EQ(channel.popOrClosed().value(), "line2");
    EXPECT_FALSE(channel.popOrClosed().has_value()); // closed and drained
    // Push after close is refused.
    EXPECT_FALSE(channel.push("late"));
}

TEST(SSEPublishChannel, SlowWriterExertsBackpressureAndCloseUnblocksProducer)
{
    // REAL WAIT-ENTRY WITNESS: blockedPushers() counts a thread INSIDE the
    // wait, so the main can prove the producer is genuinely blocked — not a
    // pre-call flag race. The cleanup OWNER closes before joining, so an early
    // ASSERT can never strand a blocked worker.
    SSEPublishChannel channel(1);
    bool late_result = true;
    std::atomic<bool> finished{false};
    std::thread producer([&] {
        EXPECT_TRUE(channel.push("x"));
        late_result = channel.push("y"); // blocks (full, capacity 1)
        finished.store(true);
    });
    {
        // CLOSE-BEFORE-JOIN OWNER: owns the channel and closes it in the dtor
        // no matter how the body unwinds (an ASSERT timeout cannot skip it).
        struct OwnerGuard
        {
            std::thread &t;
            SSEPublishChannel &ch;
            ~OwnerGuard()
            {
                ch.close();
                if (t.joinable())
                    t.join();
            }
        } guard{producer, channel};
        ASSERT_TRUE(wait_for_true([&] { return channel.blockedPushers() == 1; }));
        EXPECT_FALSE(finished.load()); // deterministically inside the wait
        // (guard destructor closes+joins at the end of this block)
    }
    EXPECT_TRUE(finished.load()); // close woke the blocked push
    EXPECT_FALSE(late_result);    // push after close refused
    EXPECT_TRUE(channel.closed());
}

TEST(SSEPublishChannel, BlockedPushResumesOnConsumerPop)
{
    // The backpressure resume path: a blocked push completes on POP.
    SSEPublishChannel channel(1);
    std::atomic<bool> finished{false};
    bool second_push_result = false;
    std::thread producer([&] {
        EXPECT_TRUE(channel.push("a"));
        second_push_result = channel.push("b"); // blocks (full, capacity 1)
        finished.store(true);
    });
    {
        struct OwnerGuard
        {
            std::thread &t;
            SSEPublishChannel &ch;
            ~OwnerGuard()
            {
                ch.close();
                if (t.joinable())
                    t.join();
            }
        } guard{producer, channel};
        ASSERT_TRUE(wait_for_true([&] { return channel.blockedPushers() == 1; }));
        EXPECT_FALSE(finished.load());
        // Drain one: the blocked push resumes (pop wake, not close). The pop
        // result and the exact queued/drained state are REQUIRED below.
        const auto first = channel.popOrClosed();
        EXPECT_EQ(first.value(), "a");
        // The pop wake happened while OPEN (the guard closes only at block end).
        EXPECT_FALSE(channel.closed());
        ASSERT_TRUE(wait_for_true(finished));
        // CAPTURED result + FATAL checks before any further pop.
        EXPECT_TRUE(second_push_result);
        ASSERT_EQ(channel.size(), 1u); // exact: "b" queued, nothing more
        const auto b = channel.popOrClosed();
        ASSERT_TRUE(b.has_value());
        EXPECT_EQ(b.value(), "b");
        EXPECT_EQ(channel.size(), 0u);
    }
    EXPECT_TRUE(finished.load());
}

TEST(SSEPublishChannel, EmptyConsumerWaitWakesOnClose)
{
    SSEPublishChannel channel(2);
    std::atomic<bool> returned{false};
    std::thread consumer([&] {
        const auto r = channel.popOrClosed(); // WAITS (empty, not closed)
        EXPECT_FALSE(r.has_value());
        returned.store(true);
    });
    {
        struct OwnedConsumerGuard
        {
            std::thread &t;
            SSEPublishChannel &ch;
            ~OwnedConsumerGuard()
            {
                ch.close(); // wakes the waiting consumer
                if (t.joinable())
                    t.join();
            }
        } guard{consumer, channel};
        // REAL WAIT-ENTRY WITNESS: the consumer is inside popOrClosed's wait.
        ASSERT_TRUE(wait_for_true([&] { return channel.blockedConsumers() == 1; }));
        EXPECT_FALSE(returned.load());
    }
    EXPECT_TRUE(returned.load()); // close woke the waiting consumer
}

TEST(SSEPublishChannel, ZeroCapacityIsRejectedAndOptionalDrainIsExplicit)
{
    EXPECT_THROW(SSEPublishChannel(0), std::invalid_argument);
    SSEPublishChannel channel(1);
    channel.close();
    // popOrClosed uses std::optional: closed+drained is nullopt, never a sentinel.
    EXPECT_FALSE(channel.popOrClosed().has_value());
}

TEST(SSEPublishChannel, WaitRegistrationVisibleWhileBlockedThenReleased)
{
    // Controlled single participant: the registration is EXACTLY 1 while the
    // producer is blocked, and returns to 0 after it resumes (pop wake).
    SSEPublishChannel channel(1);
    std::atomic<bool> finished{false};
    bool second_push_result = false;
    std::thread producer([&] {
        EXPECT_TRUE(channel.push("a"));
        second_push_result = channel.push("b"); // blocks (capacity 1, nothing draining)
        finished.store(true);
    });
    {
        struct OwnerGuard
        {
            std::thread &t;
            SSEPublishChannel &ch;
            ~OwnerGuard()
            {
                ch.close();
                if (t.joinable()) t.join();
            }
        } guard{producer, channel};
        ASSERT_TRUE(wait_for_true([&] { return channel.blockedPushers() == 1; }));
        EXPECT_FALSE(finished.load());
        EXPECT_EQ(channel.popOrClosed().value(), "a"); // pop wake
        ASSERT_TRUE(wait_for_true(finished));
        // CAPTURED result + FATAL checks BEFORE any further pop (a faulty push
        // that lost "b" must fail here, not hang on an open empty queue).
        EXPECT_TRUE(second_push_result);
        ASSERT_EQ(channel.size(), 1u);
        EXPECT_FALSE(channel.closed());
        const auto b = channel.popOrClosed();
        ASSERT_TRUE(b.has_value());
        EXPECT_EQ(b.value(), "b");
        EXPECT_EQ(channel.size(), 0u);
    }
    EXPECT_EQ(channel.blockedPushers(), 0u); // registration released
}

TEST(SSEPublishChannel, NonBlockingCallDoesNotTouchWaitCounters)
{
    // A NON-blocking call never registers (no spurious count, no steal): with
    // room available the counters stay 0 through the call.
    SSEPublishChannel channel(2);
    EXPECT_TRUE(channel.push("a"));
    EXPECT_TRUE(channel.push("b")); // still has room? capacity 2 -> full now
    EXPECT_EQ(channel.blockedPushers(), 0u); // no wait was needed
    // Drain one and push again non-blocking: counters stay 0.
    EXPECT_EQ(channel.popOrClosed().value(), "a");
    EXPECT_TRUE(channel.push("c"));
    EXPECT_EQ(channel.blockedPushers(), 0u);
    EXPECT_EQ(channel.size(), 2u);
}

TEST(SSEPublishChannel, CloseReleasesOutstandingWaitRegistration)
{
    // A blocked producer's registration is outstanding at close and released
    // exactly when that call returns (no positive-only zero).
    SSEPublishChannel channel(1);
    std::atomic<bool> finished{false};
    bool late_result = true;
    std::thread producer([&] {
        EXPECT_TRUE(channel.push("x"));
        late_result = channel.push("y"); // blocked, registration outstanding
        finished.store(true);
    });
    {
        // OWNER GUARD (close-before-join) installed immediately: an early return/
        // exception can never leave a joinable thread behind.
        struct OwnerGuard
        {
            std::thread &t;
            SSEPublishChannel &ch;
            ~OwnerGuard()
            {
                ch.close();
                if (t.joinable()) t.join();
            }
        } guard{producer, channel};
        ASSERT_TRUE(wait_for_true([&] { return channel.blockedPushers() == 1; }));
        channel.close(); // close wake; the registration remains until return
        ASSERT_TRUE(wait_for_true(finished));
        // The registration is released with THIS call's return; push after close
        // is refused (explicit result).
        EXPECT_EQ(channel.blockedPushers(), 0u);
        EXPECT_FALSE(late_result);
    }
    EXPECT_TRUE(finished.load());
}

TEST(SSEPublishChannel, NonWaitCompletionLeavesOutstandingWaiterRegistration)
{
    // ORDINARY-PATH DISCRIMINATOR via a GUARDED COORDINATOR: after the pop wakes
    // P1, a coordinator thread performs an ORDINARY push (never tryPush). If the
    // coordinator wins the freed slot, its non-wait completion runs through the
    // production push while P1's registration is STILL outstanding (count MUST
    // stay 1 — faulty shared arithmetic shows 0). If P1 wins, the coordinator's
    // ordinary push would wait; the OWNER GUARD closes the channel at scope end,
    // so the coordinator can never hang. Bounded trials to observe the branch.
    bool observed_main_won = false;
    for (int trial = 0; trial < 200 && !observed_main_won; ++trial)
    {
        SSEPublishChannel channel(1);
        std::atomic<bool> p1_finished{false};
        std::atomic<bool> coordinator_returned{false};
        bool coordinator_result = false;
        // OWNERS INSTALLED OVER DEFAULT HANDLES BEFORE ANY LAUNCH: a throwing
        // second std::thread ctor can never leave the first unguarded.
        std::thread producer;
        std::thread coordinator;
        struct OwnerGuard
        {
            std::thread &a;
            std::thread &b;
            SSEPublishChannel &ch;
            ~OwnerGuard()
            {
                ch.close(); // unblocks ANY blocked ordinary call
                if (a.joinable()) a.join();
                if (b.joinable()) b.join();
            }
        } guard{producer, coordinator, channel};
        producer = std::thread([&] {
            EXPECT_TRUE(channel.push("a"));
            (void)channel.push("b");
            p1_finished.store(true);
        });
        // WITNESS THE OPPONENT'S WAIT BEFORE THE COMPETITOR STARTS (initial
        // ordering: the coordinator can never complete before P1 registers).
        ASSERT_TRUE(wait_for_true([&] { return channel.blockedPushers() == 1; }));
        coordinator = std::thread([&] {
            const auto first = channel.popOrClosed(); // frees the slot (wakes P1)
            EXPECT_TRUE(first.has_value());
            coordinator_result = channel.push("c"); // ORDINARY push; may race P1
            coordinator_returned.store(true);
        });
        // WAIT FOR EITHER OUTCOME (the loser schedule must not trip the 5s bound):
        // if P1 wins, the coordinator's ordinary push blocks until the guard's
        // close below — that is the expected losing branch, not a failure.
        bool coord_won = false;
        for (int i = 0; i < 500; ++i)
        {
            if (coordinator_returned.load() && coordinator_result)
            {
                coord_won = true;
                break;
            }
            if (p1_finished.load())
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (coord_won)
        {
            observed_main_won = true;
            // The coordinator's ORDINARY push completed WITHOUT waiting; P1 is
            // deterministically still blocked (queue refilled, nothing drains)
            // => its registration MUST be 1 (faulty arithmetic shows 0).
            EXPECT_FALSE(p1_finished.load());
            EXPECT_EQ(channel.blockedPushers(), 1u);
        }
        // else: either the opponent won (the coordinator's call waits and the
        // guard close retires it) OR the poll expired without either observed
        // completion — both are non-discriminating outcomes; the trial continues.
        // (The required positive overlap observation is asserted at the end.)
    }
    EXPECT_TRUE(observed_main_won); // the discriminating branch ran
}

TEST(SSEPublishChannel, NonWaitConsumerCompletionLeavesOtherRegistration)
{
    // ORDINARY-PATH consumer symmetry via a guarded coordinator: after the push
    // wakes the consumer, the coordinator performs an ORDINARY popOrClosed. If
    // the coordinator wins the line, its non-wait completion runs through the
    // production pop while the consumer's registration is still outstanding
    // (count MUST stay 1; faulty arithmetic shows 0). The owner guard closes at
    // scope end so neither participant can hang.
    bool observed_main_won = false;
    for (int trial = 0; trial < 200 && !observed_main_won; ++trial)
    {
        SSEPublishChannel channel(2);
        std::atomic<bool> consumer_returned{false};
        std::atomic<bool> coordinator_returned{false};
        std::optional<std::string> coordinator_line;
        std::thread consumer;
        std::thread coordinator;
        struct OwnerGuard
        {
            std::thread &a;
            std::thread &b;
            SSEPublishChannel &ch;
            ~OwnerGuard()
            {
                ch.close();
                if (a.joinable()) a.join();
                if (b.joinable()) b.join();
            }
        } guard{consumer, coordinator, channel};
        consumer = std::thread([&] {
            (void)channel.popOrClosed(); // waits (empty)
            consumer_returned.store(true);
        });
        // WITNESS the consumer's wait BEFORE the competitor starts.
        ASSERT_TRUE(wait_for_true([&] { return channel.blockedConsumers() == 1; }));
        coordinator = std::thread([&] {
            EXPECT_TRUE(channel.push("a")); // wakes the consumer; may race it
            coordinator_line = channel.popOrClosed(); // ORDINARY pop; may wait
            coordinator_returned.store(true);
        });
        // WAIT FOR EITHER OUTCOME (loser schedule is not a failure).
        bool coord_won = false;
        for (int i = 0; i < 500; ++i)
        {
            if (coordinator_returned.load() && coordinator_line.has_value())
            {
                coord_won = true;
                break;
            }
            if (consumer_returned.load())
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (coord_won)
        {
            observed_main_won = true;
            EXPECT_EQ(coordinator_line.value(), "a");
            // The coordinator's ORDINARY pop completed WITHOUT waiting; the
            // consumer's registration MUST still be 1 (faulty shows 0).
            EXPECT_EQ(channel.blockedConsumers(), 1u);
        }
        // else: either the consumer won (the guard close retires the coordinator
        // wait) OR the poll expired without either observed completion — both are
        // non-discriminating outcomes; the trial continues. (The required
        // positive overlap observation is asserted at the end.)
    }
    EXPECT_TRUE(observed_main_won);
}
/** @brief QUEUED-PAYLOAD byte window: push blocks at the byte limit (NOT the
 *  entry limit) and resumes on POP; bufferedBytes accounts exactly; tryPush
 *  refuses beyond the queued-payload byte window. (Admission/physical bytes
 *  are a separate authority — this quota is window geometry, not an admission
 *  claim.) */
TEST(SSEPublishChannel, QueuedPayloadByteBoundBackpressuresAndAccounting)
{
    // Entry bound (8) is large: the BYTE bound (30) is the binding authority.
    SSEPublishChannel channel(SSEPublishChannelBudget{/*max_entries*/ 8,
                                                      /*max_buffered_bytes*/ 30});
    const std::string big(10, 'x');
    EXPECT_TRUE(channel.push(big));
    EXPECT_TRUE(channel.push(big));
    EXPECT_TRUE(channel.push(big));
    EXPECT_EQ(channel.size(), 3u);
    EXPECT_EQ(channel.bufferedBytes(), 30u); // exact admitted bytes
    // Byte bound exactly reached: a further entry would exceed the admitted
    // extent; tryPush (non-blocking) refuses without registering any wait.
    EXPECT_FALSE(channel.tryPush(big));
    EXPECT_EQ(channel.blockedPushers(), 0u);
    EXPECT_EQ(channel.blockedConsumers(), 0u);

    // BLOCKING push at the byte limit: waits (real wait-entry witness), then
    // resumes on POP (not close); accounting stays exact end to end.
    std::atomic<bool> finished{false};
    bool late_result = false;
    std::thread producer([&] {
        late_result = channel.push(big); // blocks (30 bytes buffered, +10 > 30)
        finished.store(true);
    });
    {
        struct OwnerGuard
        {
            std::thread &t;
            SSEPublishChannel &ch;
            ~OwnerGuard()
            {
                ch.close();
                if (t.joinable())
                    t.join();
            }
        } guard{producer, channel};
        ASSERT_TRUE(wait_for_true([&] { return channel.blockedPushers() == 1; }));
        EXPECT_FALSE(finished.load()); // deterministically inside the byte wait
        // Drain one line: 10 admitted bytes return, so the blocked push resumes
        // while OPEN (the guard closes only at block end).
        const auto first = channel.popOrClosed();
        ASSERT_TRUE(first.has_value());
        EXPECT_EQ(first->size(), 10u);
        ASSERT_TRUE(wait_for_true(finished));
        EXPECT_TRUE(late_result); // the byte-bound push WAS admitted after pop
        EXPECT_FALSE(channel.closed());
        EXPECT_EQ(channel.size(), 3u);        // exact: 3 buffered again
        EXPECT_EQ(channel.bufferedBytes(), 30u);
        EXPECT_EQ(channel.budget().max_buffered_bytes, 30u);
        EXPECT_EQ(channel.budget().max_entries, 8u);
    }
    EXPECT_TRUE(finished.load());
}

/** @brief IMPOSSIBLE-FIT REJECTION BEFORE WAITING (seq-198 deadlock): a line
 *  larger than the whole byte window can NEVER fit (no pop can help), so the
 *  channel rejects it deterministically before any wait registration — no
 *  producer/consumer deadlock (empty channel + oversized line). tryPush is a
 *  definite refuse with zero registration. */
TEST(SSEPublishChannel, ImpossibleLineRejectedBeforeWaiting)
{
    SSEPublishChannel channel(SSEPublishChannelBudget{/*max_entries*/ 1,
                                                      /*max_buffered_bytes*/ 30});
    const std::string too_big(31, 'x');
    // OWNED WORKER + CAPTURED OUTCOME: the rejected push runs on a worker owned
    // from before launch; the main observes the outcome with a BOUNDED wait and
    // the owner closes+joins even when the observation fails — no main-thread
    // blocking, no timeout-dependent retirement.
    std::exception_ptr caught;
    std::atomic<bool> done{false};
    std::thread worker;
    {
        struct OwnerGuard
        {
            std::thread &t;
            SSEPublishChannel &ch;
            ~OwnerGuard()
            {
                ch.close();
                if (t.joinable()) t.join();
            }
        } guard{worker, channel};
        worker = std::thread([&] {
            try
            {
                channel.push(too_big); // deterministic typed rejection (throw)
            }
            catch (...)
            {
                caught = std::current_exception();
            }
            done.store(true);
        });
        ASSERT_TRUE(wait_for_true(done)); // bounded observation
        EXPECT_TRUE(caught != nullptr);
        bool length_seen = false;
        try
        {
            if (caught)
                std::rethrow_exception(caught);
        }
        catch (const std::length_error &)
        {
            length_seen = true;
        }
        EXPECT_TRUE(length_seen); // captured typed outcome
        EXPECT_EQ(channel.size(), 0u);
        EXPECT_EQ(channel.bufferedBytes(), 0u);
        EXPECT_EQ(channel.blockedPushers(), 0u); // no wait was ever registered
        // tryPush: definite refuse without registration.
        EXPECT_FALSE(channel.tryPush(too_big));
        EXPECT_EQ(channel.blockedPushers(), 0u);
        EXPECT_EQ(channel.blockedConsumers(), 0u);
        EXPECT_EQ(channel.size(), 0u);
        // A fitting line still works after the rejection.
        EXPECT_TRUE(channel.tryPush(std::string(10, 'y')));
        EXPECT_EQ(channel.bufferedBytes(), 10u);
    } // close-before-join owner (failed observation is still retired safely)
}

/** @brief SKEWED-SIZE WAITERS (eligible wake policy): with a byte cap, a pop
 *  + notify_all must let the ELIGIBLE (smaller) waiter proceed while the
 *  INELIGIBLE (larger) waiter re-waits — never the reverse. */
TEST(SSEPublishChannel, SkewedSizeWaiterEligibleWakePolicy)
{
    // 2 + 8 = 10 bytes buffered: the BYTE CAP (10) is exactly full (two of the
    // four entry slots are used — the entry bound is NOT what blocks here). A
    // 10-byte push and a 2-byte push both wait on the byte cap.
    SSEPublishChannel channel(SSEPublishChannelBudget{/*max_entries*/ 4,
                                                      /*max_buffered_bytes*/ 10});
    ASSERT_TRUE(channel.tryPush(std::string(2, 'a')));
    ASSERT_TRUE(channel.tryPush(std::string(8, 'b')));
    EXPECT_EQ(channel.bufferedBytes(), 10u);

    std::atomic<bool> big_done{false}, small_done{false};
    bool big_result = false, small_result = false;
    // OWNER INSTALLED BEFORE EITHER LAUNCH (v46 pattern): the guard holds the
    // thread handles from the start (default-constructed: not joinable yet), so
    // a throw from either launch can never leave an unowned joinable thread.
    std::thread big, small;
    {
        struct OwnerGuard
        {
            std::thread &tb;
            std::thread &ts;
            SSEPublishChannel &ch;
            ~OwnerGuard()
            {
                ch.close();
                if (tb.joinable()) tb.join();
                if (ts.joinable()) ts.join();
            }
        } guard{big, small, channel};
        big = std::thread([&] {
            big_result = channel.push(std::string(10, 'c')); // blocks (10 + 10 > 10)
            big_done.store(true);
        });
        small = std::thread([&] {
            small_result = channel.push(std::string(2, 'd')); // blocks (2 + 10 > 10)
            small_done.store(true);
        });
        ASSERT_TRUE(wait_for_true([&] { return channel.blockedPushers() == 2; }));
        // Drain 2 bytes: the ELIGIBLE 2-byte waiter must proceed (8 + 2 = 10 fits)
        // while the 10-byte waiter re-waits (10 + 8 > 10).
        const auto first = channel.popOrClosed();
        ASSERT_TRUE(first.has_value());
        EXPECT_EQ(first->size(), 2u);
        ASSERT_TRUE(wait_for_true(small_done));
        EXPECT_TRUE(small_result);   // eligible waiter admitted
        EXPECT_FALSE(big_done.load()); // ineligible waiter still WAITING (witness)
        EXPECT_EQ(channel.blockedPushers(), 1u);
        // The big waiter is proven blocked; the guard close then retires it.
    }
    EXPECT_TRUE(big_done.load()); // close woke the ineligible waiter
    EXPECT_FALSE(big_result);    // push after close refused
}

/** @brief CHECKED CONSTRUCTION (seq-242 #4): a legacy capacity whose byte
 *  multiplication wraps is rejected with a typed overflow error — never a
 *  silently-wrapped zero/unbounded budget. */
TEST(SSEPublishChannel, LegacyCapacityByteBoundOverflowRejected)
{
    constexpr std::size_t huge = std::numeric_limits<std::size_t>::max() / 4096 + 1;
    // NOTE: without the (void) cast, `SSEPublishChannel(huge)` inside the
    // ASSERT_THROW statement is parsed as a DECLARATION of a different object
    // named `huge` — it DEFAULT-CONSTRUCTS with that same size_t constructor
    // (capacity 64), so the OUTER `huge` argument never reaches the checked
    // constructor and no overflow is observed. The cast forces expression
    // evaluation with the outer `huge` value.
    ASSERT_THROW((void)SSEPublishChannel(huge), std::overflow_error);
    // A capacity at exactly one below the wrap boundary is still valid.
    SSEPublishChannel valid(std::numeric_limits<std::size_t>::max() / 4096);
    EXPECT_EQ(valid.budget().max_entries, std::numeric_limits<std::size_t>::max() / 4096);
}


/** @brief SYNTHETIC PRODUCTION-FACTORY/CHANNEL REGRESSION (seq-201): the
 *  production factory policy selects the byte-UNBOUNDED quota (max_buffered_
 *  bytes == 0), so ORDINARY push admits a large frame and the consumer drains
 *  the exact payload. This is a channel-level regression only — it is NOT
 *  proof of all long frames or of native HTTP/model-echo behavior; actual
 *  handler/native qualification is separate. */
TEST(SSEPublishChannel, ProductionFactoryPreservesLongFrames)
{
    // Production factory policy: entry quota from the provisional policy and
    // BYTES UNBOUNDED — the unbounded selection is ASSERTED explicitly.
    SSEPublishChannel channel(SSEPublishChannelBudget::forRequest(/*max_tokens*/1));
    ASSERT_EQ(channel.budget().max_buffered_bytes, 0u); // UNBOUNDED policy
    EXPECT_EQ(channel.budget().max_entries, 2u + 6u);   // provisional policy

    // ORDINARY push on a PRE-OWNED WORKER (complete v56 pattern): BOTH the
    // exception_ptr and the push result are captured, and `pushed` is
    // published after EITHER outcome — an unexpected push exception can never
    // terminate the process or hang the coordinator.
    const std::string long_frame(5000, 'x'); // 5000-byte model-style frame
    std::exception_ptr caught;
    bool push_result = false;
    std::atomic<bool> pushed{false};
    std::thread worker;
    {
        struct OwnerGuard
        {
            std::thread &t;
            SSEPublishChannel &ch;
            ~OwnerGuard()
            {
                ch.close();
                if (t.joinable()) t.join();
            }
        } guard{worker, channel};
        worker = std::thread([&] {
            try
            {
                push_result = channel.push(long_frame); // ordinary push
            }
            catch (...)
            {
                caught = std::current_exception();
            }
            pushed.store(true); // after EITHER outcome
        });
        ASSERT_TRUE(wait_for_true(pushed)); // bounded observation
        // FATAL checks before consuming anything:
        ASSERT_TRUE(caught == nullptr);   // no unexpected push exception
        ASSERT_TRUE(push_result);         // successful enqueue (fatal, not EXPECT)
        ASSERT_EQ(channel.bufferedBytes(), 5000u); // exact admitted bytes
        // Post-completion inspection via tryPop (fatal establishment before any
        // blocking drain): the item is present while OPEN, so a faulty candidate
        // that refused publication cannot leave the main block before cleanup.
        const auto line = channel.tryPop();
        ASSERT_TRUE(line.has_value());    // present while open
        EXPECT_EQ(*line, long_frame);     // EXACT payload
        EXPECT_EQ(channel.size(), 0u);
        EXPECT_EQ(channel.bufferedBytes(), 0u); // queued bytes back to 0
        EXPECT_FALSE(channel.closed());
    } // guard performs the close+join (completion is NOT the join)
}
