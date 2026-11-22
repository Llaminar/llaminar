/**
 * @file Test__SerializedInferenceExecutor.cpp
 * @brief Unit tests for typed serving admission in the serialized HTTP executor.
 *
 * The executor is the response-lease admission authority at the serving
 * boundary (design §9.4). These tests prove that both typed axes are enforced:
 * the response-lease total R (capacity) and the queue/preparing bound Q, where
 * the single execution-slot occupant is never counted against Q. An authored
 * Q = 0 therefore still admits the one response occupying the worker while
 * refusing every additional accepted response.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "app/modes/SerializedInferenceExecutor.h"
#include "utils/ServingCapacityTestIssuer.h"

namespace llaminar2::test
{
    TEST(SerializedInferenceExecutor, QueueZeroAdmitsOnlyTheExecutionSlot)
    {
        SerializedInferenceExecutor executor(4, 0);
        auto first = executor.tryReserve();
        EXPECT_TRUE(first);
        // Q = 0: beyond the execution slot nothing may be accepted, even though
        // R = 4 would allow it.
        EXPECT_FALSE(executor.tryReserve());
        // The execution-slot occupant may still submit and run.
        first->run([]{});
        // Release ends the response lease; a new single response is admitted.
        first.reset();
        auto replacement = executor.tryReserve();
        EXPECT_TRUE(replacement);
        EXPECT_FALSE(executor.tryReserve());
        replacement->run([]{});
    }

    TEST(SerializedInferenceExecutor, QueueBoundLimitsAcceptedBeyondExecutionSlot)
    {
        SerializedInferenceExecutor executor(8, 1);
        auto slot = executor.tryReserve();
        EXPECT_TRUE(slot);
        auto extra = executor.tryReserve();
        EXPECT_TRUE(extra);
        EXPECT_FALSE(executor.tryReserve()); // Q = 1: only one beyond the slot
        slot->run([]{});
        extra->run([]{});
    }

    TEST(SerializedInferenceExecutor, ResponseLeaseTotalStillBoundsAdmission)
    {
        SerializedInferenceExecutor executor(2, 1);
        auto first = executor.tryReserve();
        EXPECT_TRUE(first);
        auto second = executor.tryReserve();
        EXPECT_TRUE(second);
        EXPECT_FALSE(executor.tryReserve()); // R = 2 reached
        first->run([]{});
        second->run([]{});
    }

    TEST(SerializedInferenceExecutor, QueueCapacityValidatesConstructor)
    {
        EXPECT_THROW((SerializedInferenceExecutor(8, 1025)), std::invalid_argument);
        EXPECT_THROW((SerializedInferenceExecutor(0, 0)), std::invalid_argument);
    }

    TEST(SerializedInferenceExecutor, DefaultsPreserveQualifiedPoolSemantics)
    {
        SerializedInferenceExecutor executor;
        EXPECT_EQ(executor.snapshot().capacity, SerializedInferenceExecutor::kDefaultCapacity);
        // Automatic envelope: R = 32, Q = R - 1 -> behavior identical to the
        // historical 32-total admission.
        std::vector<std::shared_ptr<SerializedInferenceExecutor::Reservation>> held;
        for (int i = 0; i < 32; ++i)
        {
            auto reservation = executor.tryReserve();
            EXPECT_TRUE(reservation) << i;
            held.push_back(std::move(reservation));
        }
        // 33rd is refused by the response-lease total.
        EXPECT_FALSE(executor.tryReserve());
    }

    TEST(SerializedInferenceExecutor, LifetimeCountsDistinguishAttachedWaitingAndDelivering)
    {
        SerializedInferenceExecutor executor(8, 1);
        auto occupant = executor.tryReserve();
        ASSERT_TRUE(occupant);
        auto waiting = executor.tryReserve();
        ASSERT_TRUE(waiting);
        EXPECT_FALSE(executor.tryReserve());

        occupant->run([]{});
        auto snap = executor.snapshot();
        // Occupant finished production: delivery retained, slot freed.
        EXPECT_EQ(snap.delivering, 1u);
        EXPECT_EQ(snap.queued, 0u);
        // Waiting reservation may now run; slot can accept a new occupant.
        auto replacement = executor.tryReserve();
        ASSERT_TRUE(replacement);
        waiting->run([]{});
        replacement->run([]{});
        EXPECT_EQ(executor.snapshot().delivering, 3u);
    }

    TEST(SerializedInferenceExecutor, NoSecondOccupantWhileFirstIsRunning)
    {
        // Deterministic: the occupant blocks inside the worker until released;
        // while attached/running, Q = 0 must refuse a second 'occupant'.
        SerializedInferenceExecutor executor(4, 0);
        auto occupant = executor.tryReserve();
        ASSERT_TRUE(occupant);
        bool released = false;
        std::mutex gate_mutex;
        std::condition_variable gate;
        std::thread runner([&] {
            occupant->run([&] {
                std::unique_lock gate_lock(gate_mutex);
                gate.wait(gate_lock, [&] { return released; });
            });
        });
        // Wait until the worker is executing the occupant.
        {
            std::unique_lock gate_lock(gate_mutex);
            gate.wait_for(gate_lock, std::chrono::seconds(5), [&] {
                return executor.snapshot().active;
            });
        }
        EXPECT_TRUE(executor.snapshot().active);
        EXPECT_FALSE(executor.tryReserve()); // Q = 0: no second occupant while running
        {
            std::lock_guard gate_lock(gate_mutex);
            released = true;
        }
        gate.notify_all();
        runner.join();
        // Production finished: slot free, delivery retained; a new occupant is admitted.
        EXPECT_TRUE(executor.tryReserve());
        EXPECT_EQ(executor.snapshot().delivering, 1u);
    }

    TEST(SerializedInferenceExecutor, WorkerOwnsCompletionBeforeRunReturns)
    {
        // run() returns only after the worker finished this reservation's
        // completion bookkeeping: the slot is observably idle and the lease
        // already delivering, independent of caller scheduling.
        SerializedInferenceExecutor executor(4, 0);
        auto occupant = executor.tryReserve();
        ASSERT_TRUE(occupant);
        occupant->run([] {});
        const auto snap = executor.snapshot();
        EXPECT_FALSE(snap.active);
        EXPECT_EQ(snap.delivering, 1u);
        EXPECT_EQ(snap.queued, 0u);
    }

    TEST(SerializedInferenceExecutor, ThrowingOccupantReleasesTheSlotWithoutResponseDestruction)
    {
        // A retained throwing submission must not pin the idle slot until
        // response destruction: the worker-side hook frees the slot and moves
        // the lease to delivering even though production threw.
        SerializedInferenceExecutor executor(4, 0);
        auto occupant = executor.tryReserve();
        ASSERT_TRUE(occupant);
        EXPECT_THROW(occupant->run([] { throw std::runtime_error("inference failed"); }),
                     std::runtime_error);
        const auto snap = executor.snapshot();
        EXPECT_EQ(snap.delivering, 1u);
        EXPECT_FALSE(snap.active);
        // The slot is free for a new occupant while the failed response is
        // still delivering (Q = 0 bounds waiting responses only).
        auto replacement = executor.tryReserve();
        EXPECT_TRUE(replacement);
        replacement->run([] {});
    }

    TEST(SerializedInferenceExecutor, QueueBoundExcludesTheOccupantsOwnSubmission)
    {
        // Automatic envelope R = 32, Q = 31: with the occupant queued for
        // execution and 30 responses preparing, the 32nd response must still
        // be admitted. The occupant's own submission is execution work, never
        // waiting-room capacity; only the response-lease total refuses the 33rd.
        SerializedInferenceExecutor executor(32, 31);
        auto occupant = executor.tryReserve();
        ASSERT_TRUE(occupant);
        bool released = false;
        std::mutex gate_mutex;
        std::condition_variable gate;
        std::thread runner([&] {
            occupant->run([&] {
                std::unique_lock gate_lock(gate_mutex);
                gate.wait(gate_lock, [&] { return released; });
            });
        });
        {
            std::unique_lock gate_lock(gate_mutex);
            gate.wait_for(gate_lock, std::chrono::seconds(5), [&] {
                return executor.snapshot().active;
            });
        }
        std::vector<std::shared_ptr<SerializedInferenceExecutor::Reservation>> waiting;
        for (int i = 0; i < 30; ++i)
        {
            auto reservation = executor.tryReserve();
            ASSERT_TRUE(reservation) << i;
            waiting.push_back(std::move(reservation));
        }
        // 31 waiting (30 preparing) + the executing occupant: the 32nd is admitted.
        auto thirty_second = executor.tryReserve();
        EXPECT_TRUE(thirty_second);
        // The 33rd is refused by the response-lease total R = 32.
        EXPECT_FALSE(executor.tryReserve());
        {
            std::lock_guard gate_lock(gate_mutex);
            released = true;
        }
        gate.notify_all();
        runner.join();
        EXPECT_EQ(executor.snapshot().delivering, 1u);
    }

    TEST(SerializedInferenceExecutor, ExecutingWaitingRequestLeavesTheWaitingRoom)
    {
        // R8/Q1 interleaving: A holds and completes; B (waiting) submits, starts
        // and holds the worker. While B executes it is not waiting and is
        // promoted to slot holder, so the waiting room has space again and C
        // is admitted although Q = 1 was full at B's admission.
        SerializedInferenceExecutor executor(8, 1);
        auto holder = executor.tryReserve();
        ASSERT_TRUE(holder);
        auto waiting = executor.tryReserve(); // Q = 1 filled
        ASSERT_TRUE(waiting);
        EXPECT_FALSE(executor.tryReserve());
        holder->run([] {});
        bool released = false;
        std::mutex gate_mutex;
        std::condition_variable gate;
        std::thread runner([&] {
            waiting->run([&] {
                std::unique_lock gate_lock(gate_mutex);
                gate.wait(gate_lock, [&] { return released; });
            });
        });
        {
            std::unique_lock gate_lock(gate_mutex);
            // B executing: it left the waiting room at start, not at completion.
            gate.wait_for(gate_lock, std::chrono::seconds(5), [&] {
                const auto snap = executor.snapshot();
                return snap.active && snap.queued == 0;
            });
        }
        // No request is waiting now: C is admitted.
        auto third = executor.tryReserve();
        EXPECT_TRUE(third);
        {
            std::lock_guard gate_lock(gate_mutex);
            released = true;
        }
        gate.notify_all();
        runner.join();
        // B completed as the promoted holder: its delivery is retained.
        EXPECT_EQ(executor.snapshot().delivering, 2u);
    }

    TEST(SerializedInferenceExecutor, DisplacedHolderChargedWhileAnotherExecutes)
    {
        // F1/F2 regression at the actual boundary: A is admitted first (holder)
        // but never submits; the waiting B submits and starts executing. The S
        // exemption TRANSFERS to B and the displaced pending A is recharged
        // into the waiting room. A then submits while B still holds the worker,
        // so A's job sits queued BEFORE dequeue (deterministic handshake:
        // pending == 1) and counts against Q: with 30 more preparing responses
        // the waiting room is full (Q = 31 of R = 64) and the next admission is
        // refused — the two-exemption state would wrongly admit it.
        SerializedInferenceExecutor executor(64, 31);
        auto displaced = executor.tryReserve(); // holder while pending
        ASSERT_TRUE(displaced);
        auto blocker = executor.tryReserve();  // waiting
        ASSERT_TRUE(blocker);
        bool released = false;
        std::mutex gate_mutex;
        std::condition_variable gate;
        std::thread blocker_runner([&] {
            blocker->run([&] {
                std::unique_lock gate_lock(gate_mutex);
                gate.wait(gate_lock, [&] { return released; });
            });
        });
        {
            std::unique_lock gate_lock(gate_mutex);
            // B executes with the transferred exemption; A is charged pending.
            gate.wait_for(gate_lock, std::chrono::seconds(5), [&] {
                const auto snap = executor.snapshot();
                return snap.active && snap.queued == 0 && snap.preparing == 1;
            });
        }
        // Fill the remaining waiting room: A + 30 more = Q = 31.
        std::vector<std::shared_ptr<SerializedInferenceExecutor::Reservation>> filling;
        for (int i = 0; i < 30; ++i)
        {
            auto reservation = executor.tryReserve();
            ASSERT_TRUE(reservation) << i;
            filling.push_back(std::move(reservation));
        }
        // A submits while the worker is still held by B: its job is enqueued
        // and waits before dequeue (pending == 1 proves the enqueue, no sleep).
        std::thread displaced_runner([&] { displaced->run([] {}); });
        {
            std::unique_lock gate_lock(gate_mutex);
            gate.wait_for(gate_lock, std::chrono::seconds(5), [&] {
                return executor.snapshot().pending == 1;
            });
        }
        const auto snap = executor.snapshot();
        EXPECT_TRUE(snap.active);       // B still executing
        EXPECT_EQ(snap.queued, 1u);     // A: submitted waiting (charged, not exempt)
        EXPECT_EQ(snap.preparing, 30u); // the 30 fillers
        // Waiting room full (30 preparing + A queued = Q = 31): boundary refusal.
        EXPECT_FALSE(executor.tryReserve());
        {
            std::lock_guard gate_lock(gate_mutex);
            released = true;
        }
        gate.notify_all();
        blocker_runner.join();
        displaced_runner.join();
    }

    TEST(SerializedInferenceExecutor, CapturedOwnerRetiresOnTheWorkerOutsideTheQueueMutex)
    {
        // The closure owns the FINAL Reservation reference (movable-only
        // functor; no caller-side shared copies survive the submit), and its
        // re-entering cleanup (snapshot) must run ON THE WORKER at closure
        // retirement — exactly the point where the inherited serve() shape
        // held the queue mutex. The retirement is observed directly: the
        // destructor records its thread, releases the final reference and
        // signals this test; under the old ordering that destructor blocks
        // forever inside snapshot() and the observation deadline fails.
        SerializedInferenceExecutor executor(4, 0);
        std::weak_ptr<SerializedInferenceExecutor::Reservation> observer;
        std::mutex retirement_mutex;
        std::condition_variable retired;
        bool retired_on_worker = false;
        std::thread::id production_thread{};
        std::thread::id retirement_thread{};
        {
            auto reservation = executor.tryReserve();
            ASSERT_TRUE(reservation);
            observer = reservation;
            // Raw invoke pointer: the functor below holds the ONLY remaining
            // shared reference, so its worker-side destruction is provably
            // the final-owner retirement.
            SerializedInferenceExecutor::Reservation *const invoke = reservation.get();
            struct ReenteringWork
            {
                SerializedInferenceExecutor *executor = nullptr;
                std::shared_ptr<SerializedInferenceExecutor::Reservation> owner;
                std::thread::id *production_thread = nullptr;
                std::thread::id *retirement_thread = nullptr;
                std::mutex *retirement_mutex = nullptr;
                std::condition_variable *retired = nullptr;
                bool *retired_on_worker = nullptr;

                ReenteringWork(SerializedInferenceExecutor *exec,
                               std::shared_ptr<SerializedInferenceExecutor::Reservation> own,
                               std::thread::id *production, std::thread::id *retirement,
                               std::mutex *mutex, std::condition_variable *retire_flag,
                               bool *on_worker)
                    : executor(exec), owner(std::move(own)), production_thread(production),
                      retirement_thread(retirement), retirement_mutex(mutex),
                      retired(retire_flag), retired_on_worker(on_worker) {}

                ReenteringWork(ReenteringWork &&other) noexcept
                    : executor(other.executor), owner(std::move(other.owner)),
                      production_thread(other.production_thread),
                      retirement_thread(other.retirement_thread),
                      retirement_mutex(other.retirement_mutex), retired(other.retired),
                      retired_on_worker(other.retired_on_worker)
                {
                    // The moved-from shell must be inert: its destructor is a
                    // no-op on the caller thread.
                    other.executor = nullptr;
                }
                ReenteringWork(const ReenteringWork &) = delete;
                ReenteringWork &operator=(const ReenteringWork &) = delete;
                ReenteringWork &operator=(ReenteringWork &&) = delete;

                void operator()() { *production_thread = std::this_thread::get_id(); }

                ~ReenteringWork()
                {
                    if (executor == nullptr)
                        return; // moved-from shell destroyed on the caller
                    *retirement_thread = std::this_thread::get_id();
                    // Owner cleanup re-enters the executor mutex: it must
                    // observe the worker NOT holding it (the inherited serve()
                    // shape deadlocked exactly here).
                    (void)executor->snapshot();
                    // Release the FINAL reference on the worker, then signal.
                    owner.reset();
                    {
                        std::lock_guard lock(*retirement_mutex);
                        *retired_on_worker = true;
                    }
                    retired->notify_all();
                }
            };
            ReenteringWork work{&executor, std::move(reservation), &production_thread,
                                &retirement_thread, &retirement_mutex, &retired,
                                &retired_on_worker};
            invoke->run(std::move(work));
            // run() returned (completion hook ran); the closure now solely
            // owns the reservation and the callable.
        }
        {
            std::unique_lock lock(retirement_mutex);
            retired.wait_for(lock, std::chrono::seconds(5),
                             [&] { return retired_on_worker && observer.expired(); });
        }
        EXPECT_TRUE(retired_on_worker);
        // The cleanup ran on the worker thread, at the formerly-locked point.
        EXPECT_NE(production_thread, std::thread::id{});
        EXPECT_EQ(production_thread, retirement_thread);
        // The final reference is provably released by that worker-side cleanup.
        EXPECT_TRUE(observer.expired());
        // The worker must keep serving after the captured owner retired.
        auto successor = executor.tryReserve();
        ASSERT_TRUE(successor);
        successor->run([] {});
    }

    namespace completion_proof
    {
        template <class GateT>
        void openGate(const std::shared_ptr<GateT> &gate)
        {
            {
                std::lock_guard l(gate->mutex);
                gate->open = true;
            }
            gate->cv.notify_all();
        }
    } // namespace completion_proof

    TEST(SerializedInferenceExecutor, CompletionIsRequestBoundNotAnyJobProgress)
    {
        // Deterministic frontier proof (move-only armed-role fixture; protocol
        // identical to the executed control). BlockingResult is MOVE-ONLY and
        // only the ORIGINAL worker temporary stays armed: its destructor (end
        // of promise_.set_value(std::invoke(fn_)) — after future readiness,
        // BEFORE this job's completion hook) blocks on the gate; the move
        // constructor yields UNARMED destinations so the promise and caller
        // values are inert. Thread OWNERSHIP: both std::thread handles are
        // default-constructed and the RAII Cleanup guard is installed BEFORE
        // any thread starts, so a construction failure can never leave a
        // joinable thread unguarded. ORDERING: A is launched and its start is
        // WITNESSED (bounded wait) BEFORE B is launched, so the B-enqueue
        // snapshot unambiguously describes A active / B queued. While the
        // gate is CLOSED the worker is provably inside the original temporary
        // destructor with B's hook pending; the test observes B's return
        // status for a bounded window (an observed return is an early
        // return), then opens the gate and requires B's return only after its
        // hook (active == false). Cleanup guard releases A, opens the gate and
        // joins both threads on EVERY exit path, so fatal ASSERTs are safe.
        struct Gate
        {
            std::mutex mutex;
            std::condition_variable cv;
            bool entered = false;
            bool open = false;
        };
        struct BlockingResult
        {
            std::shared_ptr<Gate> gate;
            bool armed = true;
            explicit BlockingResult(std::shared_ptr<Gate> g) : gate(std::move(g)) {}
            BlockingResult(BlockingResult &&o) noexcept : gate(o.gate), armed(false)
            {
                // The moved-from source keeps armed=true (the worker
                // temporary's destructor still blocks at the publication
                // boundary); every destination is unarmed/inert.
            }
            BlockingResult(const BlockingResult &) = delete;
            BlockingResult &operator=(const BlockingResult &) = delete;
            BlockingResult &operator=(BlockingResult &&) = delete;
            ~BlockingResult()
            {
                if (!armed || !gate)
                    return;
                std::unique_lock l(gate->mutex);
                gate->entered = true;
                gate->cv.notify_all();
                gate->cv.wait_for(l, std::chrono::seconds(10), [&] { return gate->open; });
            }
        };
        SerializedInferenceExecutor executor(8, 4);
        auto a = executor.tryReserve();
        auto b = executor.tryReserve();
        ASSERT_TRUE(a);
        ASSERT_TRUE(b);
        auto gate = std::make_shared<Gate>();
        std::mutex m;
        std::condition_variable cv;
        bool a_started = false;
        bool a_release = false;
        bool b_run_returned = false;
        bool b_active_at_return = false;
        // Handles default-constructed; the cleanup guard owns them before any start.
        std::thread a_thread;
        std::thread b_thread;
        struct Cleanup
        {
            std::mutex *m = nullptr;
            std::condition_variable *cv = nullptr;
            bool *a_release = nullptr;
            std::shared_ptr<Gate> gate;
            std::thread *a = nullptr;
            std::thread *b = nullptr;
            ~Cleanup()
            {
                {
                    std::lock_guard l(*m);
                    *a_release = true;
                }
                cv->notify_all();
                if (gate)
                    completion_proof::openGate(gate);
                if (a && a->joinable())
                    a->join();
                if (b && b->joinable())
                    b->join();
            }
        };
        Cleanup cleanup{&m, &cv, &a_release, gate, &a_thread, &b_thread};
        a_thread = std::thread([&] {
            a->run([&] {
                {
                    std::lock_guard l(m);
                    a_started = true;
                }
                cv.notify_all();
                std::unique_lock l(m);
                cv.wait(l, [&] { return a_release; });
            });
        });
        // ORDERING: witness A's start BEFORE launching B.
        {
            std::unique_lock l(m);
            ASSERT_TRUE(cv.wait_for(l, std::chrono::seconds(5),
                                    [&] { return a_started; }));
        }
        b_thread = std::thread([&] {
            b->run([&] { return BlockingResult{gate}; });
            {
                std::lock_guard l(m);
                b_active_at_return = executor.snapshot().active;
                b_run_returned = true;
            }
            cv.notify_all();
        });
        // B-enqueued witness BEFORE releasing A (bounded predicate, no sleep).
        {
            std::unique_lock l(m);
            ASSERT_TRUE(cv.wait_for(l, std::chrono::seconds(2), [&] {
                return executor.snapshot().pending == 1;
            }));
        }
        const auto before_release = executor.snapshot();
        ASSERT_TRUE(before_release.active);    // A still executing
        ASSERT_EQ(before_release.pending, 1u); // B submitted, not yet picked up
        ASSERT_FALSE(b_run_returned);
        {
            std::lock_guard l(m);
            a_release = true;
        }
        cv.notify_all();
        // Wait until the worker is INSIDE the original temporary destructor
        // (promise ready; B's hook NOT yet run; gate still closed).
        bool entered = false;
        {
            std::unique_lock l(gate->mutex);
            entered = gate->cv.wait_for(l, std::chrono::seconds(10),
                                        [&] { return gate->entered; });
        }
        ASSERT_TRUE(entered);
        // OBSERVE while the gate is still CLOSED (the hook is provably held):
        // a bounded window during which an early-returning design records.
        {
            std::unique_lock l(m);
            const bool early = cv.wait_for(l, std::chrono::seconds(3),
                                           [&] { return b_run_returned; });
            EXPECT_FALSE(b_run_returned)
                << "B returned while its completion hook was provably pending (early return)";
            EXPECT_FALSE(b_active_at_return);
            (void)early;
        }
        completion_proof::openGate(gate);
        bool returned_in_time = false;
        {
            std::unique_lock l(m);
            returned_in_time = cv.wait_for(l, std::chrono::seconds(10),
                                           [&] { return b_run_returned; });
            // b_active_at_return is written by B under m: read it under m.
            ASSERT_TRUE(returned_in_time);
            EXPECT_FALSE(b_active_at_return); // returned only after its hook
        }
        EXPECT_FALSE(executor.snapshot().active);
        EXPECT_EQ(executor.snapshot().queued, 0u);
        // Cleanup guard joins both threads on every path.
    }
    TEST(SerializedInferenceExecutor, LvalueOnlyWorkKeepsResultTyping)
    {
        // seq 138 F3: TypedJob must keep the OUTER result/call-category typing.
        // An lvalue-only callable (operator()() &) is valid when passed as an
        // lvalue; re-deducing Result from the decayed (prvalue) type rejected it.
        SerializedInferenceExecutor executor(2, 0);
        auto reservation = executor.tryReserve();
        ASSERT_TRUE(reservation);
        struct LvalueWork
        {
            int calls = 0;
            int operator()() & { return ++calls; }
        };
        LvalueWork work;
        const int result = reservation->run(work);
        EXPECT_EQ(result, 1);
        // The stored copy (not the caller's object) executed.
        EXPECT_EQ(work.calls, 0);
    }

    TEST(SerializedInferenceExecutor, ReferenceReturningWorkPreservesTheReference)
    {
        // Result may be a reference type: run() must return the exact
        // reference, never a decayed copy (the same promise/future path).
        SerializedInferenceExecutor executor(2, 0);
        auto reservation = executor.tryReserve();
        ASSERT_TRUE(reservation);
        struct RefWork
        {
            int &out;
            explicit RefWork(int &value) : out(value) {}
            int &operator()() { return out; }
        };
        int value = 42;
        int &result = reservation->run(RefWork{value});
        EXPECT_EQ(&result, &value);
        EXPECT_EQ(result, 42);
    }

    TEST(SerializedInferenceExecutor, HttpWorkerCountUsesTheTypedPoolAuthority)
    {
        // The pool width must come from servingHttpWorkerCount(resolved, headroom)
        // — the single authority — never from an ad-hoc capacity + 2 rule.
        const auto resolved = ServingCapacityTestIssuer::fromComponents(1, 8, 0);
        SerializedInferenceExecutor executor(resolved);
        // The executor consumes the canonical admission; its pool width follows
        // the typed authority (R + headroom) with no exported capacity mint.
        EXPECT_EQ(executor.httpWorkerCount(), 8u + 2u);
        // A different headroom must change the result PROVING the typed authority
        // value (and not a stale +2) is used.
        SerializedInferenceExecutor executor5(resolved, 5);
        EXPECT_EQ(executor5.httpWorkerCount(), 8u + 5u);
        // Q derives from the SAME immutable resolved value: resolve(1,8,0) has
        // Q=0, so a second tryReserve while the first is Preparing must FAIL.
        SerializedInferenceExecutor zero_queue(resolved);
        auto first = zero_queue.tryReserve();
        ASSERT_NE(first, nullptr);
        // Q=0 AND the first reservation still occupies the exempt slot: a second
        // waiting response must be REFUSED (the Q bound is real, not a default).
        EXPECT_EQ(zero_queue.tryReserve(), nullptr);
        // Invalid components never bypass the checked authority.
        EXPECT_THROW(ServingCapacityTestIssuer::fromComponents(1, -1, 0), std::invalid_argument);
        EXPECT_THROW(ServingCapacityTestIssuer::fromComponents(1, 0, 0), std::invalid_argument);
        EXPECT_THROW(ServingCapacityTestIssuer::fromComponents(1, 1025, 0), std::invalid_argument);
        // Widened S+Q check: the sum can never overflow into UB (int64 arithmetic).
        EXPECT_THROW(ServingCapacityTestIssuer::fromComponents(
                         std::numeric_limits<int>::max(), 1, 1), std::invalid_argument);
        // INT_MAX headroom is refused BEFORE worker creation (no terminate on
        // the noexcept pool-width helper).
        EXPECT_THROW(SerializedInferenceExecutor(resolved, std::numeric_limits<int>::max()),
                     std::invalid_argument);
    }

    TEST(SerializedInferenceExecutor, PublicationTerminalAuthenticatesBothVerdictAndProducer)
    {
        // Preparing + native success: the producer never finished (e.g. a
        // pre-provider shutdown reported native success) => NO-OP, no throw, the
        // destructor owns retirement — never a terminate from response teardown.
        auto executor = std::make_shared<SerializedInferenceExecutor>();
        {
            auto reservation = executor->tryReserve();
            ASSERT_NE(reservation, nullptr);
            EXPECT_NO_THROW(reservation->publicationTerminal(true));
            EXPECT_EQ(executor->drainedPublications(), 0u);
            EXPECT_EQ(executor->snapshot().delivering, 0u);
        }
        EXPECT_EQ(executor->snapshot().admitted, 0u);
    }

    TEST(SerializedInferenceExecutor, PublicationTerminalDrainsAFullyCompletedSSE)
    {
        // Delivering + Reserved (this binding never emits InFlight/Detached
        // explicitly) + native success: the terminal receipt performs the
        // authenticated transitions and DRAINS exactly once.
        auto executor = std::make_shared<SerializedInferenceExecutor>();
        auto reservation = executor->tryReserve();
        ASSERT_NE(reservation, nullptr);
        reservation->run([] { return 42; }); // Delivering
        EXPECT_EQ(executor->snapshot().delivering, 1u);
        EXPECT_NO_THROW(reservation->publicationTerminal(true));
        EXPECT_EQ(executor->drainedPublications(), 1u);
        EXPECT_EQ(executor->snapshot().delivering, 0u);
        // Exactly once: a repeated terminal receipt is a no-op.
        EXPECT_NO_THROW(reservation->publicationTerminal(true));
        EXPECT_EQ(executor->drainedPublications(), 1u);
        EXPECT_NO_THROW(reservation->publicationTerminal(false));
        EXPECT_EQ(executor->drainedPublications(), 1u);
        reservation.reset();
        EXPECT_EQ(executor->snapshot().admitted, 0u);
    }

    TEST(SerializedInferenceExecutor, PublicationTerminalFailureAbandonsNotDrains)
    {
        auto executor = std::make_shared<SerializedInferenceExecutor>();
        auto reservation = executor->tryReserve();
        ASSERT_NE(reservation, nullptr);
        reservation->run([] { return 1; });
        EXPECT_EQ(executor->snapshot().delivering, 1u);
        EXPECT_NO_THROW(reservation->publicationTerminal(false)); // transport failure
        EXPECT_EQ(executor->drainedPublications(), 0u);           // never fabricated
        EXPECT_EQ(executor->snapshot().delivering, 1u);           // still retained
        reservation.reset(); // abandoned: destructor releases delivering once
        EXPECT_EQ(executor->snapshot().delivering, 0u);
        EXPECT_EQ(executor->snapshot().admitted, 0u);
    }
} // namespace llaminar2::test
