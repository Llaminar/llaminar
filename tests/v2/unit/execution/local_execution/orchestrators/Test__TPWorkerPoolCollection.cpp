/**
 * @file Test__TPWorkerPoolCollection.cpp
 * @brief Device-free exception and retry contracts for the actual persistent worker result collector.
 *
 * A thread-local allocation fault targets only the calling collector's result
 * vector. Real TPWorkerPool workers, generation state, result publication and
 * condition variables execute unchanged. A failed snapshot must release its
 * collector admission while leaving the same generation available for retry.
 */
#include "execution/local_execution/orchestrators/TPWorkerPool.h"
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <future>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

namespace collection_fault {
thread_local bool next_allocation = false;
/** @brief Arm only one original collector thread's next result-snapshot allocation. */
struct SnapshotAllocation {
    SnapshotAllocation() { next_allocation = true; }
    ~SnapshotAllocation() { next_allocation = false; }
};
}

/** @brief Delegate every ordinary allocation; fault only the explicitly armed collector. */
void *operator new(std::size_t bytes) {
    if (collection_fault::next_allocation) {
        collection_fault::next_allocation = false;
        throw std::bad_alloc();
    }
    if (auto *pointer = std::malloc(bytes ? bytes : 1)) return pointer;
    throw std::bad_alloc();
}
void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept { std::free(pointer); }

namespace llaminar2::test {
namespace {
using namespace std::chrono_literals;
/** @return Whether the actual result-vector copy rejected the injected allocation. */
bool failedSnapshot(TPWorkerPool &pool, int timeout_ms = 0) {
    const collection_fault::SnapshotAllocation fault;
    try { (void)pool.collectAll(timeout_ms); }
    catch (const std::bad_alloc &) { return true; }
    return false;
}
/** @brief Check the original participant result ordering after successful collection. */
void expectSuccess(const std::vector<TPWorkerPool::WorkerResult> &results, std::size_t count) {
    ASSERT_EQ(results.size(), count);
    for (std::size_t participant = 0; participant < count; ++participant) {
        EXPECT_EQ(results[participant].worker_index, participant);
        EXPECT_TRUE(results[participant].completed);
        EXPECT_TRUE(results[participant].success);
        EXPECT_EQ(results[participant].exception, nullptr);
    }
}
/** @brief Always release a fixture's blocked real callbacks before the pool joins them. */
struct WorkerRelease {
    std::promise<void> &promise;
    bool opened = false;
    void open() { if (!opened) { promise.set_value(); opened = true; } }
    ~WorkerRelease() { open(); }
};
}

/** @brief A completed generation remains unconsumed after bad_alloc, then retries and dispatches normally. */
TEST(TPWorkerPoolCollectionLifecycle, CompletedSnapshotFailureRetriesOriginalGeneration) {
    for (const std::size_t participants : {std::size_t{1}, std::size_t{3}, std::size_t{8}, std::size_t{13}}) {
        TPWorkerPool pool(participants);
        for (int generation = 0; generation < 20; ++generation) {
            pool.dispatch([](std::size_t) { return true; });
            ASSERT_TRUE(failedSnapshot(pool));
            // Snapshot failure must not consume a completed but unread generation.
            EXPECT_THROW(pool.dispatch([](std::size_t) { return true; }), std::logic_error);
            expectSuccess(pool.collectAll(), participants);
            // This public protocol check also detects a leaked active_collectors_ count.
            ASSERT_NO_THROW(pool.dispatch([](std::size_t) { return true; }));
            expectSuccess(pool.collectAll(), participants);
        }
    }
}

/** @brief An incomplete timed-out snapshot cannot release its still-active generation or callable. */
TEST(TPWorkerPoolCollectionLifecycle, IncompleteSnapshotFailureRetainsCallbacksUntilRetry) {
    std::promise<void> release;
    const auto gate = release.get_future().share();
    TPWorkerPool pool(3);
    WorkerRelease retirement{release};
    pool.dispatch([gate](std::size_t) { gate.wait(); return true; });
    EXPECT_TRUE(failedSnapshot(pool, 1));
    const auto partial = pool.collectAll(1);
    EXPECT_EQ(partial.size(), 3u);
    for (const auto &result : partial) EXPECT_FALSE(result.completed);
    EXPECT_THROW(pool.dispatch([](std::size_t) { return true; }), std::logic_error);
    retirement.open();
    expectSuccess(pool.collectAll(), 3);
    ASSERT_NO_THROW(pool.dispatch([](std::size_t) { return true; }));
    expectSuccess(pool.collectAll(), 3);
}

/** @brief A failed snapshot preserves false results and original worker exceptions through retry. */
TEST(TPWorkerPoolCollectionLifecycle, RetryPreservesFailureAndExceptionResults) {
    TPWorkerPool pool(3);
    pool.dispatch([](std::size_t participant) {
        if (participant == 2) throw std::runtime_error("controlled worker failure");
        return participant != 1;
    });
    ASSERT_TRUE(failedSnapshot(pool));
    const auto results = pool.collectAll();
    ASSERT_EQ(results.size(), 3u);
    EXPECT_TRUE(results[0].completed && results[0].success);
    EXPECT_TRUE(results[1].completed); EXPECT_FALSE(results[1].success); EXPECT_EQ(results[1].exception, nullptr);
    EXPECT_TRUE(results[2].completed); EXPECT_FALSE(results[2].success); ASSERT_NE(results[2].exception, nullptr);
    EXPECT_THROW(std::rethrow_exception(results[2].exception), std::runtime_error);
    ASSERT_NO_THROW(pool.dispatch([](std::size_t) { return true; }));
    expectSuccess(pool.collectAll(), 3);
}

/** @brief One throwing collector cannot retain admission after another observer consumes the same generation. */
TEST(TPWorkerPoolCollectionLifecycle, SnapshotFailureAndConcurrentObserverRetireIndependently) {
    std::promise<void> release;
    const auto gate = release.get_future().share();
    TPWorkerPool pool(3);
    WorkerRelease retirement{release};
    pool.dispatch([gate](std::size_t) { gate.wait(); return true; });
    std::atomic<unsigned> entered{0};
    auto failing = std::async(std::launch::async, [&] { ++entered; return failedSnapshot(pool); });
    auto observer = std::async(std::launch::async, [&] { ++entered; return pool.collectAll(); });
    while (entered.load() != 2) std::this_thread::yield();
    retirement.open();
    EXPECT_TRUE(failing.get());
    expectSuccess(observer.get(), 3);
    // The successful observer may consume the generation, but neither collector
    // may leave its admission behind when the next generation is dispatched.
    ASSERT_NO_THROW(pool.dispatch([](std::size_t) { return true; }));
    expectSuccess(pool.collectAll(), 3);
}
}
