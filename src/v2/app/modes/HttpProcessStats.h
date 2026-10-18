/**
 * @file HttpProcessStats.h
 * @brief Cached OS health observations independent of HTTP request frequency.
 *
 * One background owner samples process, cgroup and filesystem metadata. HTTP
 * readers only copy an immutable publication, including age and collection
 * errors; no request can trigger, queue, or wait for a kernel/filesystem probe.
 * These are observations, never an alternative physical-memory admission ledger.
 */
#pragma once

#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace llaminar2
{
    /** @brief Fixed, startup-selected filesystem locations; cache payloads are never visited. */
    struct HttpProcessStatsOptions
    {
        std::chrono::milliseconds interval{5000}; ///< Minimum delay between completed probes.
        std::vector<std::filesystem::path> filesystems{"."}; ///< statvfs only, no directory walk.
    };

    /** @brief Linux metadata reader with explicit roots for device-free parser fixtures. */
    class HttpProcessStatsReader
    {
    public:
        /**
         * @brief Bind immutable metadata locations before sampling starts.
         * @param filesystems Startup-selected paths queried with statvfs.
         * @param proc Procfs root, normally /proc; tests may provide small text fixtures.
         * @param cgroup Unified cgroup mount, normally /sys/fs/cgroup.
         */
        explicit HttpProcessStatsReader(std::vector<std::filesystem::path> filesystems,
            std::filesystem::path proc = "/proc", std::filesystem::path cgroup = "/sys/fs/cgroup");
        /** @brief Collect metadata once; missing sections retain explicit availability errors. */
        [[nodiscard]] nlohmann::json operator()();

    private:
        std::vector<std::filesystem::path> filesystems_;
        std::filesystem::path proc_, cgroup_;
        std::optional<std::pair<std::chrono::steady_clock::time_point, double>> previous_cpu_;
    };

    /** @brief A single periodic collector publishing bounded immutable observations. */
    class HttpProcessStats final
    {
    public:
        using Clock = std::chrono::steady_clock;
        using Collector = std::function<nlohmann::json()>;
        /**
         * @brief Start one owner without delaying endpoint admission for an OS probe.
         * @param options Fixed polling cadence and filesystem selection.
         * @param collector Explicit test reader; empty selects the production OS reader.
         */
        explicit HttpProcessStats(HttpProcessStatsOptions options = {}, Collector collector = {});
        /** @brief Interrupt the cadence wait and join the sole collector before destruction. */
        ~HttpProcessStats();
        HttpProcessStats(const HttpProcessStats &) = delete;
        HttpProcessStats &operator=(const HttpProcessStats &) = delete;
        /**
         * @brief Read a cached publication; performs no procfs, driver or filesystem I/O.
         * @param now Monotonic observation instant; injectable to prove age/staleness behavior.
         * @return Last successful sample, or explicit warming/unavailable state, plus freshness.
         */
        [[nodiscard]] nlohmann::json snapshot(Clock::time_point now = Clock::now()) const;

    private:
        /** @brief Retain last good data and current collection failure as one coherent publication. */
        struct Publication
        {
            nlohmann::json data = nullptr;
            uint64_t attempts = 0, successful_samples = 0;
            Clock::time_point sampled_at{};
            double sampled_unix_seconds = 0, collection_seconds = 0;
            std::string error;
        };
        /** @brief Refresh only on the owner cadence; readers have no refresh entrypoint. */
        void collect(std::stop_token stop);

        const std::chrono::milliseconds interval_;
        Collector collector_;
        std::atomic<std::shared_ptr<const Publication>> publication_;
        std::mutex wait_mutex_;
        std::condition_variable_any wake_;
        std::jthread worker_; ///< Last member: its join precedes every captured dependency's destruction.
    };
}
