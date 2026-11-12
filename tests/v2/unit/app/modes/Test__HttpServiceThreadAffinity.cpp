/**
 * @file Test__HttpServiceThreadAffinity.cpp
 * @brief Native CPU-mask proofs for independent HTTP workers under OpenMP binding.
 *
 * A rank can own many CPU places while its initial thread occupies only one.
 * These model-free tests inspect actual scheduler masks on newly created HTTP
 * workers and a previously created inference worker. The scope must preserve
 * the existing inference placement and restore its caller, including unwind.
 * Real loopback requests also inspect the production listener's HTTP pool;
 * directly testing the guard cannot prove its use at the serving boundary.
 * Actual process collectors and prefix archive workers have the same independent
 * progress requirement. Their observed masks and retirement are checked through
 * the public constructors, without substituting a test-created service thread.
 */
#include "app/modes/HttpServiceThreadAffinity.h"
#include "app/modes/HttpProcessStats.h"
#include "app/modes/SerializedInferenceExecutor.h"
#include "app/modes/ServerMode.h"
#include "execution/prefix_cache/DiskPrefixStorageBackend.h"
#include "httplib.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <future>
#include <set>
#include <thread>
#include <vector>
#include <cstdlib>

using namespace llaminar2;

namespace
{
    /** @return An independent read of the current thread's published scheduler mask. */
    cpu_set_t observedMask()
    {
        cpu_set_t result;
        CPU_ZERO(&result);
        if (sched_getaffinity(0, sizeof(result), &result) != 0)
            throw std::runtime_error("Test cannot read current CPU affinity");
        return result;
    }

    /** @return The initial rank partition inventory, independent of the guard's implementation. */
    cpu_set_t expectedPartition()
    {
        if (omp_get_proc_bind() == omp_proc_bind_false)
            return observedMask();
        cpu_set_t expected;
        CPU_ZERO(&expected);
        // These tests run at the initial level, where every runtime place is
        // part of the rank. The implementation must also preserve subpartitions.
        for (int place = 0; place < omp_get_num_places(); ++place)
        {
            std::vector<int> cpus(static_cast<std::size_t>(omp_get_place_num_procs(place)));
            omp_get_place_proc_ids(place, cpus.data());
            for (int cpu : cpus)
                CPU_SET(cpu, &expected);
        }
        return expected;
    }

    /** @return Live task identities in this fixture process, never another job's threads. */
    std::set<pid_t> taskIds()
    {
        std::set<pid_t> result;
        for (const auto &entry : std::filesystem::directory_iterator("/proc/self/task"))
            result.insert(static_cast<pid_t>(std::stol(entry.path().filename().string())));
        return result;
    }

    /** @brief Own one exclusively created temporary archive directory through worker retirement. */
    class TemporaryArchiveDirectory
    {
    public:
        /** @brief Create a fresh task-owned path without touching another test's cache. */
        TemporaryArchiveDirectory()
        {
            auto pattern = (std::filesystem::temp_directory_path() / "llaminar-prefix-affinity-XXXXXX").string();
            const char *created = ::mkdtemp(pattern.data());
            if (!created) throw std::runtime_error("Cannot create prefix affinity fixture directory");
            path = created;
        }
        /** @brief Remove only this fixture's files after all archive workers have joined. */
        ~TemporaryArchiveDirectory()
        {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
        std::filesystem::path path; ///< Exclusive fixture root, never a configured user cache.
    };
}

TEST(HttpServiceThreadAffinity, ProductionProcessCollectorUsesPartitionAndRestoresCaller)
{
    const auto original = observedMask();
    const auto partition = expectedPartition();
    std::promise<cpu_set_t> sampled;
    auto observation = sampled.get_future();
    {
        HttpProcessStats stats({.interval = std::chrono::hours(1)}, [&] {
            sampled.set_value(observedMask());
            return nlohmann::json::object();
        });
        const auto after_creation = observedMask();
        EXPECT_TRUE(CPU_EQUAL(&after_creation, &original));
        ASSERT_EQ(observation.wait_for(std::chrono::seconds(2)), std::future_status::ready);
        const auto worker = observation.get();
        EXPECT_TRUE(CPU_EQUAL(&worker, &partition));
    }
    const auto after_retirement = observedMask();
    EXPECT_TRUE(CPU_EQUAL(&after_retirement, &original));
}

TEST(PrefixArchiveWorkerAffinity, ProductionWriterAndMaintenanceUsePartitionAndRetire)
{
    const auto original = observedMask();
    const auto partition = expectedPartition();
    TemporaryArchiveDirectory directory;
    const std::string identity(64, 'a');
    // Reopen the actual archive after retirement: neither a stale worker nor
    // a retained caller pin may become the next lifetime's inherited placement.
    for (int lifetime = 0; lifetime < 2; ++lifetime)
    {
        const auto prior_tasks = taskIds();
        std::set<pid_t> workers;
        {
            DiskPrefixStorageBackend archive(directory.path / (identity + ".kvcache"), 4096, identity);
            ASSERT_TRUE(archive.ready()) << archive.initializationError();
            const auto after_creation = observedMask();
            EXPECT_TRUE(CPU_EQUAL(&after_creation, &original));
            for (const pid_t tid : taskIds())
                if (!prior_tasks.contains(tid)) workers.insert(tid);
            // The archive owns one durable writer and one metadata-maintenance
            // worker. Inspect BOTH actual OS tasks, including their idle state.
            ASSERT_EQ(workers.size(), 2u);
            for (const pid_t tid : workers)
            {
                cpu_set_t mask{};
                ASSERT_EQ(sched_getaffinity(tid, sizeof(mask), &mask), 0);
                EXPECT_TRUE(CPU_EQUAL(&mask, &partition)) << "archive worker " << tid;
            }
            std::string error;
            ASSERT_TRUE(archive.requestCompaction(&error)) << error;
            ASSERT_TRUE(archive.waitForCompaction(&error)) << error;
            ASSERT_TRUE(archive.waitForPersistence(&error)) << error;
        }
        const auto after_retirement = observedMask();
        EXPECT_TRUE(CPU_EQUAL(&after_retirement, &original));
        const auto remaining = taskIds();
        for (const pid_t tid : workers) EXPECT_FALSE(remaining.contains(tid));
    }
}

TEST(HttpServiceThreadAffinity, HttpChildrenUseAdmittedPartitionWhileInferenceKeepsItsPlacement)
{
    const auto original = observedMask();
    const auto partition = expectedPartition();
    SerializedInferenceExecutor inference;
    auto reservation = inference.tryReserve();
    ASSERT_NE(reservation, nullptr);
    const auto inference_before = reservation->run(observedMask);
    {
        const HttpServiceThreadAffinity service;
        const auto listener = observedMask();
        EXPECT_TRUE(CPU_EQUAL(&listener, &partition));
        auto worker = std::async(std::launch::async, observedMask);
        const auto child = worker.get();
        EXPECT_TRUE(CPU_EQUAL(&child, &partition));
        auto next = inference.tryReserve();
        ASSERT_NE(next, nullptr);
        const auto inference_during = next->run(observedMask);
        EXPECT_TRUE(CPU_EQUAL(&inference_before, &inference_during));
    }
    const auto restored = observedMask();
    EXPECT_TRUE(CPU_EQUAL(&restored, &original));
}

TEST(HttpServiceThreadAffinity, ExceptionRestoresCallerPlacement)
{
    const auto original = observedMask();
    EXPECT_THROW(([] {
        const HttpServiceThreadAffinity service;
        throw std::runtime_error("controlled HTTP exit");
    }()), std::runtime_error);
    const auto restored = observedMask();
    EXPECT_TRUE(CPU_EQUAL(&original, &restored));
}

TEST(HttpServiceThreadAffinity, RejectsOpenMPComputeRegionsBeforeChangingPlacement)
{
    bool rejected = false;
    bool preserved = false;
#pragma omp parallel num_threads(1)
    {
        const auto original = observedMask();
        try
        {
            const HttpServiceThreadAffinity service;
        }
        catch (const std::runtime_error &)
        {
            rejected = true;
        }
        const auto actual = observedMask();
        preserved = CPU_EQUAL(&actual, &original);
    }
    EXPECT_TRUE(rejected);
    EXPECT_TRUE(preserved);
}

TEST(HttpServiceThreadAffinity, ProductionListenerPlacesActualHttpPoolAndRestoresItsCaller)
{
    const auto partition = expectedPartition();
    SerializedInferenceExecutor inference;
    httplib::Server server;
    configureInferenceHttpServer(server, inference);
    std::promise<cpu_set_t> worker_mask;
    server.Get("/worker-mask", [&](const httplib::Request &, httplib::Response &response) {
        worker_mask.set_value(observedMask());
        response.set_content("observed", "text/plain");
    });
    const auto port = server.bind_to_any_port("127.0.0.1");
    ASSERT_GT(port, 0);
    auto listener = std::async(std::launch::async, [&] {
        const auto before = observedMask();
        const bool served = listenInferenceHttpServer(server);
        const auto after = observedMask();
        return std::pair{served, CPU_EQUAL(&before, &after)};
    });
    // This guard stops the server before the future's joining destructor,
    // including failed assertions and exceptions in the loopback client.
    struct StopOnExit
    {
        httplib::Server &server;
        /** @brief Release the listener before its future and route captures retire. */
        ~StopOnExit() { server.stop(); }
    } stop{server};
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(std::chrono::seconds(3));
    const auto response = client.Get("/worker-mask");
    ASSERT_TRUE(response);
    ASSERT_EQ(response->status, 200);
    ASSERT_EQ(response->body, "observed");
    const auto observed = worker_mask.get_future().get();
    EXPECT_TRUE(CPU_EQUAL(&observed, &partition));
    server.stop();
    const auto [served, restored] = listener.get();
    EXPECT_TRUE(served);
    EXPECT_TRUE(restored);
}
