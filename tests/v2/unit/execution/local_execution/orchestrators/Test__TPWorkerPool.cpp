/**
 * @file Test__TPWorkerPool.cpp
 * @brief Concurrency regressions for generation-owned LocalTP worker fan-out.
 *
 * These tests intentionally use only host threads. TPWorkerPool is an
 * orchestration primitive, so its unit contract must be provable without
 * initializing CUDA or ROCm. GPU execution belongs to integration suites that
 * exercise the child runners and collective backends behind this fan-out.
 */

#include "execution/local_execution/orchestrators/TPWorkerPool.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <thread>

namespace llaminar2::test
{
    namespace
    {
        using namespace std::chrono_literals;

        /**
         * @brief Require every participant result to describe one completed success.
         *
         * @param results Immutable generation snapshot returned by the pool.
         */
        void expectSuccessfulGeneration(
            const std::vector<TPWorkerPool::WorkerResult> &results)
        {
            ASSERT_EQ(results.size(), 2u);
            for (size_t index = 0; index < results.size(); ++index)
            {
                EXPECT_EQ(results[index].worker_index, index);
                EXPECT_TRUE(results[index].completed);
                EXPECT_TRUE(results[index].success);
                EXPECT_EQ(results[index].exception, nullptr);
            }
        }
    } // namespace

    /**
     * @brief A terminal worker completion wakes every observer of its generation.
     *
     * The historical pool used `notify_one()` after the last participant. If a
     * diagnostic observer and the rank collector were both asleep, one could
     * remain blocked forever even though the completion predicate was already
     * true. Both collectors are admitted before the workers are released so the
     * regression exercises that exact terminal-notification window.
     */
    TEST(Test__TPWorkerPool, CompletedGenerationWakesEveryConcurrentCollector)
    {
        TPWorkerPool pool(/*num_workers=*/2);
        std::promise<void> release_workers;
        const std::shared_future<void> release =
            release_workers.get_future().share();

        pool.dispatch(
            [release](size_t) mutable
            {
                release.wait();
                return true;
            });

        std::atomic<int> collectors_admitted{0};
        auto collect = [&]()
        {
            collectors_admitted.fetch_add(1, std::memory_order_release);
            return pool.collectAll(/*timeout_ms=*/2000);
        };
        auto first = std::async(std::launch::async, collect);
        auto second = std::async(std::launch::async, collect);

        while (collectors_admitted.load(std::memory_order_acquire) != 2)
            std::this_thread::yield();
        std::this_thread::sleep_for(20ms);
        release_workers.set_value();

        EXPECT_EQ(first.wait_for(500ms), std::future_status::ready);
        EXPECT_EQ(second.wait_for(500ms), std::future_status::ready);
        expectSuccessfulGeneration(first.get());
        expectSuccessfulGeneration(second.get());
    }

    /**
     * @brief Back-to-back tiny generations never inherit neighboring results.
     *
     * MTP sidecar mock work can complete before the caller reaches collectAll().
     * Repeating that shape with scheduler yields proves that an early completion
     * remains attached to its own generation and that the next dispatch starts
     * only after the previous result snapshot has been consumed.
     */
    TEST(Test__TPWorkerPool, BackToBackTinyDispatchesRemainGenerationExact)
    {
        TPWorkerPool pool(/*num_workers=*/2);
        std::array<std::atomic<int>, 2> observed = {
            std::atomic<int>{0},
            std::atomic<int>{0},
        };

        constexpr int kGenerationCount = 2000;
        for (int generation = 1; generation <= kGenerationCount; ++generation)
        {
            pool.dispatch(
                [generation, &observed](size_t worker)
                {
                    if (((generation + static_cast<int>(worker)) % 7) == 0)
                        std::this_thread::yield();
                    observed[worker].store(generation, std::memory_order_release);
                    return true;
                });

            const auto results = pool.collectAll(/*timeout_ms=*/2000);
            expectSuccessfulGeneration(results);
            for (size_t worker = 0; worker < observed.size(); ++worker)
            {
                EXPECT_EQ(
                    observed[worker].load(std::memory_order_acquire),
                    generation)
                    << "worker=" << worker;
            }
        }
    }

    /**
     * @brief A blocking collective-abort callback cannot own participant completion.
     *
     * LocalTP installs a first-failure callback that can enter NCCL/RCCL
     * communicator teardown. Teardown may wait while a neighboring participant
     * leaves its collective. The callback therefore runs on the pool's dedicated
     * control thread, after the failing worker has published its terminal result.
     *
     * This test models the former deadlock deliberately. Worker zero fails.
     * Failure handling releases worker one and then remains blocked, just as a
     * communicator abort can. Both participant results must nevertheless become
     * collectible while failure handling is still blocked. Running the callback
     * inline on worker zero would leave that worker incomplete and time out.
     */
    TEST(Test__TPWorkerPool,
         BlockingFailureCallbackCannotDelayParticipantCompletion)
    {
        TPWorkerPool pool(/*num_workers=*/2);

        std::promise<void> callback_entered_promise;
        std::future<void> callback_entered =
            callback_entered_promise.get_future();
        std::promise<void> callback_finished_promise;
        std::future<void> callback_finished =
            callback_finished_promise.get_future();
        std::promise<void> release_callback_promise;
        const std::shared_future<void> release_callback =
            release_callback_promise.get_future().share();

        std::promise<void> release_peer_promise;
        const std::shared_future<void> release_peer =
            release_peer_promise.get_future().share();
        std::once_flag release_peer_once;
        const auto releasePeer = [&]()
        {
            std::call_once(
                release_peer_once,
                [&release_peer_promise]()
                { release_peer_promise.set_value(); });
        };

        pool.setFailureCallback(
            [&]()
            {
                callback_entered_promise.set_value();
                releasePeer();
                release_callback.wait();
                callback_finished_promise.set_value();
            });

        pool.dispatch(
            [release_peer](size_t worker)
            {
                if (worker == 0)
                    return false;
                release_peer.wait();
                return true;
            });

        const auto callback_status = callback_entered.wait_for(500ms);
        const auto results = pool.collectAll(/*timeout_ms=*/500);

        // Always open both gates before assertions so a failed regression cannot
        // strand either a participant or the pool's callback thread in teardown.
        releasePeer();
        release_callback_promise.set_value();
        const auto callback_finished_status = callback_finished.wait_for(500ms);

        EXPECT_EQ(callback_status, std::future_status::ready);
        ASSERT_EQ(results.size(), 2u);
        EXPECT_TRUE(results[0].completed);
        EXPECT_FALSE(results[0].success);
        EXPECT_TRUE(results[1].completed);
        EXPECT_TRUE(results[1].success);
        EXPECT_EQ(pool.completedCount(), 2u);
        EXPECT_EQ(pool.firstFailureIndex(), 0u);
        EXPECT_EQ(callback_finished_status, std::future_status::ready);
    }

} // namespace llaminar2::test
