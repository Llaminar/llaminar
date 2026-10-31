/**
 * @file Test__GPUWorkerInitialization.cpp
 * @brief Replay the production initialization transaction over arbitrary handle prefixes.
 *
 * The transaction is the same helper called around actual CUDA/HIP initializer
 * bodies. Opaque obligations vary independently of vendor implementation: tests
 * do not reproduce a second native creation algorithm or query a driver.
 */
#include "backends/GPUWorkerInitialization.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>
using namespace llaminar2;
namespace
{
    /** @brief Opaque native obligations used only to observe transaction ordering. */
    struct Obligations
    {
        std::vector<int> live;
        std::vector<int> released;
        int cleanup_calls = 0;
        std::thread::id initialization_thread{}, cleanup_thread{};
        /** @brief Materialize a prefix without duplicating a backend initializer. */
        void materialize(int count)
        {
            initialization_thread = std::this_thread::get_id();
            for (int slot = 0; slot < count; ++slot) live.push_back(slot);
        }
        /** @brief Discharge every exact obligation once, in reverse ownership order. */
        void cleanup()
        {
            ++cleanup_calls;
            cleanup_thread = std::this_thread::get_id();
            while (!live.empty()) { released.push_back(live.back()); live.pop_back(); }
        }
    };
    /** @brief Keep deliberate fatal probes from creating filesystem core payloads. */
    void disableCorePayload()
    {
        const rlimit limit{0, 0};
        if (setrlimit(RLIMIT_CORE, &limit) != 0) ::_exit(98);
    }
    /** @brief Exception identity must survive rollback unchanged. */
    struct InitializationFailure final : std::runtime_error
    {
        int boundary;
        explicit InitializationFailure(int value) : std::runtime_error("injected initialization failure"), boundary(value) {}
    };
    struct Initialize { bool operator()() const { return false; } };
    struct StatusRollback { bool operator()() const { return false; } };
    template<class Rollback>
    concept AcceptedRollback = requires(Rollback callback) { runGPUWorkerInitialization(Initialize{}, callback); };
    static_assert(!AcceptedRollback<StatusRollback>);
}

/** @test Every partially constructed prefix retires before false is returned. */
TEST(GPUWorkerInitialization, EveryHandlePrefixFalseRollsBackExactlyOnce)
{
    // This arbitrary prefix range contains all current native worker handle
    // boundaries, including their persistent 32-event sets, without copying
    // either vendor's sequence or depending on their SDKs in a Unit test.
    for (int count = 0; count <= 64; ++count)
    {
        SCOPED_TRACE(count); Obligations resources;
        EXPECT_FALSE(runGPUWorkerInitialization([&] {
            resources.materialize(count); return false;
        }, [&] { resources.cleanup(); }));
        EXPECT_EQ(resources.cleanup_calls, 1);
        EXPECT_TRUE(resources.live.empty());
        ASSERT_EQ(resources.released.size(), static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index) EXPECT_EQ(resources.released[index], count - 1 - index);
        EXPECT_EQ(resources.cleanup_thread, resources.initialization_thread);
    }
}

/** @test Initializer exceptions escape only after owning-thread cleanup completes. */
TEST(GPUWorkerInitialization, EveryHandlePrefixExceptionRetiresBeforeRethrow)
{
    for (int count = 0; count <= 64; ++count)
    {
        SCOPED_TRACE(count); Obligations resources;
        try
        {
            (void)runGPUWorkerInitialization([&]() -> bool {
                resources.materialize(count); throw InitializationFailure(count);
            }, [&] { resources.cleanup(); });
            FAIL() << "Initializer exception disappeared";
        }
        catch (const InitializationFailure &error)
        {
            EXPECT_EQ(error.boundary, count);
            EXPECT_TRUE(resources.live.empty());
            EXPECT_EQ(resources.cleanup_calls, 1);
            EXPECT_EQ(resources.released.size(), static_cast<std::size_t>(count));
        }
        EXPECT_EQ(resources.cleanup_thread, resources.initialization_thread);
    }
}

/** @test Full success transfers cleanup back to the owning worker lifecycle. */
TEST(GPUWorkerInitialization, CommitKeepsHandlesUntilOrdinaryWorkerCleanup)
{
    for (int count = 0; count <= 64; ++count)
    {
        Obligations resources;
        ASSERT_TRUE(runGPUWorkerInitialization([&] {
            resources.materialize(count); return true;
        }, [&] { resources.cleanup(); }));
        EXPECT_EQ(resources.cleanup_calls, 0);
        EXPECT_EQ(resources.live.size(), static_cast<std::size_t>(count));
        resources.cleanup();
        EXPECT_EQ(resources.cleanup_calls, 1);
        EXPECT_EQ(resources.released.size(), static_cast<std::size_t>(count));
        EXPECT_TRUE(resources.live.empty());
    }
}

/** @test The real owning worker, not its waiting parent, performs rollback. */
TEST(GPUWorkerInitialization, RollbackRunsOnInitializerWorker)
{
    Obligations resources; bool ready = true;
    const auto parent = std::this_thread::get_id();
    std::thread worker([&] {
        ready = runGPUWorkerInitialization([&] {
            resources.materialize(35); return false;
        }, [&] { resources.cleanup(); });
    });
    worker.join();
    EXPECT_FALSE(ready);
    EXPECT_NE(resources.initialization_thread, parent);
    EXPECT_EQ(resources.cleanup_thread, resources.initialization_thread);
    EXPECT_EQ(resources.cleanup_calls, 1);
    EXPECT_TRUE(resources.live.empty());
}

/** @test Failed retirement cannot turn initialization failure into a recoverable result. */
TEST(GPUWorkerInitialization, CleanupFailureIsFatalForFalseAndException)
{
    EXPECT_DEATH({
        disableCorePayload();
        (void)runGPUWorkerInitialization([] { return false; }, [] { throw std::runtime_error("cleanup false path"); });
    }, "Native initialization rollback failed: cleanup false path");
    EXPECT_DEATH({
        disableCorePayload();
        (void)runGPUWorkerInitialization([]() -> bool { throw InitializationFailure(1); }, [] { throw std::runtime_error("cleanup exception path"); });
    }, "Native initialization rollback failed: cleanup exception path");
    EXPECT_DEATH({
        disableCorePayload();
        (void)runGPUWorkerInitialization([] { return false; }, [] { throw 1; });
    }, "Native initialization rollback failed with an unknown exception");
}
