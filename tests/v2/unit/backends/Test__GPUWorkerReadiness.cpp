/**
 * @file Test__GPUWorkerReadiness.cpp
 * @brief Device-free replay of the actual owning-worker readiness publication.
 *
 * The constructor-to-wait gap is held open under the real predicate mutex.
 * An observed publisher lock attempt proves exclusion before condition-variable
 * registration. Recorded flags separately authenticate complete publication
 * order and notification after unlock, without sleeps or timing thresholds.
 */
#include "backends/GPUWorkerInitialization.h"
#include <gtest/gtest.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

namespace
{
    /** @brief Actual mutual exclusion with an optional deterministic lock-attempt receipt. */
    struct ObservedMutex
    {
        std::mutex native;
        std::atomic<bool> held{false}, observe_next{false};
        std::counting_semaphore<2> *transition = nullptr;
        /** @brief Report an attempted publisher acquisition before waiting for the constructor. */
        void lock()
        {
            if (observe_next.exchange(false) && transition) transition->release();
            native.lock();
            held.store(true);
        }
        /** @brief Release actual exclusion and the test-visible ownership observation. */
        void unlock() { held.store(false); native.unlock(); }
    };
    /** @brief Record each existing production flag write under the actual publication mutex. */
    struct ObservedFlag
    {
        ObservedMutex &mutex;
        std::vector<std::string> &trace;
        const char *name;
        bool value = false;
        /** @brief Authenticate mutation ownership and preserve the requested boolean value. */
        void store(bool next, std::memory_order)
        {
            EXPECT_TRUE(mutex.held.load());
            trace.emplace_back(name);
            value = next;
        }
    };
    /** @brief Observe notification only after the production helper releases exclusion. */
    struct ObservedCondition
    {
        ObservedMutex &mutex;
        std::vector<std::string> &trace;
        /** @brief Constructor wake must follow the complete unlocked outcome. */
        void notify_all()
        {
            EXPECT_FALSE(mutex.held.load());
            trace.emplace_back("notify");
        }
    };
}

/** @test Complete readiness precedes initialized publication; failure publishes only after rollback. */
TEST(GPUWorkerReadiness, CompleteOutcomeIsLockedAndNotificationFollowsUnlock)
{
    for (const auto outcome : {llaminar2::GPUWorkerReadiness::Ready, llaminar2::GPUWorkerReadiness::Failed})
    {
        ObservedMutex mutex;
        std::vector<std::string> trace;
        ObservedFlag initialized{mutex, trace, "initialized"};
        ObservedFlag running{mutex, trace, "running"};
        ObservedFlag shutdown{mutex, trace, "shutdown"};
        ObservedCondition condition{mutex, trace};
        llaminar2::publishGPUWorkerReadiness(mutex, condition, initialized, running, shutdown, outcome);
        if (outcome == llaminar2::GPUWorkerReadiness::Ready)
        {
            EXPECT_EQ(trace, (std::vector<std::string>{"running", "initialized", "notify"}));
            EXPECT_TRUE(initialized.value && running.value);
            EXPECT_FALSE(shutdown.value);
        }
        else
        {
            EXPECT_EQ(trace, (std::vector<std::string>{"shutdown", "notify"}));
            EXPECT_TRUE(shutdown.value);
            EXPECT_FALSE(initialized.value || running.value);
        }
    }
}

/** @test Success and failure cannot publish into the constructor's predicate-check/wait gap. */
TEST(GPUWorkerReadiness, ConstructorCheckToWaitGapIsExcludedForSuccessAndFailure)
{
    for (int replay = 0; replay < 20; ++replay)
        for (const auto outcome : {llaminar2::GPUWorkerReadiness::Ready, llaminar2::GPUWorkerReadiness::Failed})
        {
            ObservedMutex mutex;
            std::condition_variable_any condition;
            std::atomic<bool> initialized{false}, running{false}, shutdown{false};
            std::counting_semaphore<2> transition{0};
            mutex.transition = &transition;
            std::unique_lock constructor_lock(mutex);
            ASSERT_FALSE(initialized.load() || shutdown.load());
            mutex.observe_next.store(true);
            std::thread publisher([&] {
                llaminar2::publishGPUWorkerReadiness(mutex, condition, initialized, running, shutdown, outcome);
                // A lockless defective publisher also releases this receipt,
                // causing the pre-wait flag assertions to fail without hanging.
                transition.release();
            });
            transition.acquire();
            EXPECT_FALSE(initialized.load());
            EXPECT_FALSE(running.load());
            EXPECT_FALSE(shutdown.load());
            condition.wait(constructor_lock, [&] { return initialized.load() || shutdown.load(); });
            if (outcome == llaminar2::GPUWorkerReadiness::Ready)
            {
                EXPECT_TRUE(initialized.load() && running.load());
                EXPECT_FALSE(shutdown.load());
            }
            else
            {
                EXPECT_TRUE(shutdown.load());
                EXPECT_FALSE(initialized.load() || running.load());
            }
            constructor_lock.unlock();
            publisher.join();
        }
}

/** @test The existing initialization transaction retires partial state before constructor failure publication. */
TEST(GPUWorkerReadiness, FailureBecomesVisibleOnlyAfterOwningThreadRollback)
{
    ObservedMutex mutex;
    std::vector<std::string> trace;
    ObservedFlag initialized{mutex, trace, "initialized"};
    ObservedFlag running{mutex, trace, "running"};
    ObservedFlag shutdown{mutex, trace, "shutdown"};
    ObservedCondition condition{mutex, trace};
    EXPECT_FALSE(llaminar2::runGPUWorkerInitialization(
        [&] { trace.emplace_back("initialize"); return false; },
        [&] { trace.emplace_back("rollback"); }));
    llaminar2::publishGPUWorkerReadiness(mutex, condition, initialized, running, shutdown,
        llaminar2::GPUWorkerReadiness::Failed);
    EXPECT_EQ(trace, (std::vector<std::string>{"initialize", "rollback", "shutdown", "notify"}));
}
