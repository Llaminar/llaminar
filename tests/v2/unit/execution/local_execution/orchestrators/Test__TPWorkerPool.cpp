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

} // namespace llaminar2::test
