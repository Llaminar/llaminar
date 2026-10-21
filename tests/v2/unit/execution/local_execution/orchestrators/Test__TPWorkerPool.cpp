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

    namespace
    {
        /** @return Actual native mask, independent of the production scope. */
        cpu_set_t observedWorkerMask()
        {
            cpu_set_t result;
            CPU_ZERO(&result);
            if (sched_getaffinity(0, sizeof(result), &result) != 0)
                throw std::runtime_error("Cannot observe worker affinity");
            return result;
        }

        /** @return Initial rank's admitted places, independently enumerated for this test. */
        cpu_set_t expectedWorkerPartition()
        {
            if (omp_get_proc_bind() == omp_proc_bind_false) return observedWorkerMask();
            cpu_set_t result;
            CPU_ZERO(&result);
            // Registered bound tests start at the initial host level, where
            // all runtime places form the original participant partition.
            for (int place = 0; place < omp_get_num_places(); ++place)
            {
                std::vector<int> cpus(static_cast<size_t>(omp_get_place_num_procs(place)));
                omp_get_place_proc_ids(place, cpus.data());
                for (const int cpu : cpus) CPU_SET(cpu, &result);
            }
            return result;
        }
    }

    TEST(TPWorkerAffinity, IntentUsesActualDevicesAndPreservesComputeScopes)
    {
        using Intent = TPWorkerHostPlacement;
        using Scope = TPWorkerHostScope;
        const std::array cuda{DeviceId::cuda(0), DeviceId::cuda(1)};
        const std::array rocm{DeviceId::rocm(0), DeviceId::rocm(3)};
        const std::array cpu{DeviceId::cpu(), DeviceId::cpu()};
        const std::array mixed{DeviceId::rocm(0), DeviceId::cpu()};
        const std::array vendors{DeviceId::rocm(0), DeviceId::cuda(0)};
        EXPECT_EQ(tpWorkerHostPlacement(cuda, Scope::FlatRank), Intent::IndependentGPUControl);
        EXPECT_EQ(tpWorkerHostPlacement(rocm, Scope::FlatRank), Intent::IndependentGPUControl);
        EXPECT_EQ(tpWorkerHostPlacement(cpu, Scope::FlatRank), Intent::InheritedCompute);
        EXPECT_EQ(tpWorkerHostPlacement(mixed, Scope::FlatRank), Intent::InheritedCompute);
        EXPECT_EQ(tpWorkerHostPlacement(vendors, Scope::FlatRank), Intent::InheritedCompute);
        EXPECT_EQ(tpWorkerHostPlacement(rocm, Scope::PipelineOrComposed), Intent::InheritedCompute);
        EXPECT_THROW(tpWorkerHostPlacement(cuda, static_cast<Scope>(99)), std::invalid_argument);
        EXPECT_THROW(tpWorkerHostPlacement({}, Scope::FlatRank), std::invalid_argument);
        const std::array invalid{DeviceId::invalid()};
        EXPECT_THROW(tpWorkerHostPlacement(invalid, Scope::FlatRank), std::invalid_argument);
    }

    TEST(TPWorkerAffinity, ActualParticipantsAndAbortWorkerRetainPartitionWhileParentRestores)
    {
        const auto parent = observedWorkerMask();
        const auto partition = expectedWorkerPartition();
        const int team = omp_get_max_threads();
        // Callback observation owners outlive the pool's actual worker join,
        // including an early assertion return before callback completion.
        std::promise<cpu_set_t> abort_worker;
        auto observed_abort = abort_worker.get_future();
        TPWorkerPool pool(2, TPWorkerHostPlacement::IndependentGPUControl);
        const auto after_construction = observedWorkerMask();
        EXPECT_TRUE(CPU_EQUAL(&parent, &after_construction));
        for (int generation = 0; generation < 3; ++generation)
        {
            pool.dispatch([&](size_t) {
                const auto actual = observedWorkerMask();
                return CPU_EQUAL(&actual, &partition);
            });
            expectSuccessfulGeneration(pool.collectAll(2000));
        }
        pool.setFailureCallback([&] { abort_worker.set_value(observedWorkerMask()); });
        pool.dispatch([](size_t index) { return index != 0; });
        const auto failed = pool.collectAll(2000);
        ASSERT_EQ(failed.size(), 2u);
        EXPECT_TRUE(failed[0].completed);
        EXPECT_FALSE(failed[0].success);
        ASSERT_EQ(observed_abort.wait_for(2s), std::future_status::ready);
        const auto callback_mask = observed_abort.get();
        EXPECT_TRUE(CPU_EQUAL(&callback_mask, &partition));
        const auto after = observedWorkerMask();
        EXPECT_TRUE(CPU_EQUAL(&parent, &after));
        EXPECT_EQ(omp_get_max_threads(), team);
    }

    TEST(TPWorkerAffinity, InheritedComputeAndScopeUnwindPreserveExactCallerMask)
    {
        const auto parent = observedWorkerMask();
        TPWorkerPool pool(2, TPWorkerHostPlacement::InheritedCompute);
        pool.dispatch([&](size_t) {
            const auto actual = observedWorkerMask();
            return CPU_EQUAL(&actual, &parent);
        });
        expectSuccessfulGeneration(pool.collectAll(2000));
        EXPECT_THROW(([] {
            const ScopedOpenMPPartitionAffinity scope;
            throw std::runtime_error("controlled construction unwind");
        }()), std::runtime_error);
        const auto after = observedWorkerMask();
        EXPECT_TRUE(CPU_EQUAL(&parent, &after));
        EXPECT_THROW(TPWorkerPool(2, static_cast<TPWorkerHostPlacement>(99)), std::invalid_argument);
    }

    TEST(TPWorkerAffinity, ScopeRejectsActiveComputeBeforeChangingItsPlace)
    {
        bool rejected = false, preserved = false;
#pragma omp parallel num_threads(1)
        {
            const auto original = observedWorkerMask();
            try { const ScopedOpenMPPartitionAffinity scope; }
            catch (const std::runtime_error &) { rejected = true; }
            const auto after = observedWorkerMask();
            preserved = CPU_EQUAL(&original, &after);
        }
        EXPECT_TRUE(rejected);
        EXPECT_TRUE(preserved);
    }

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
