/**
 * @file HttpProcessStats.cpp
 * @brief Bounded Linux health sampling and immutable HTTP publications.
 *
 * Procfs reads are size bounded; filesystem probes use statvfs on a fixed list
 * of paths rather than scanning directories. CPU percentages use monotonic
 * interval deltas, with 100% meaning one occupied core. Shared-memory/file-cache
 * observations keep their kernel scopes instead of being added to process RSS.
 */
#include "HttpProcessStats.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <sys/resource.h>
#include <sys/statvfs.h>
#include <unistd.h>

namespace llaminar2
{
    namespace
    {
        using json = nlohmann::json;

        /** @brief Read one small kernel metadata file, rejecting truncation or hidden unbounded work. */
        std::string readMetadata(const std::filesystem::path &path)
        {
            std::ifstream input(path);
            if (!input) throw std::runtime_error("cannot read " + path.string());
            std::array<char, 65537> bytes{};
            input.read(bytes.data(), bytes.size());
            const auto size = input.gcount();
            if (size == static_cast<std::streamsize>(bytes.size()) || input.bad())
                throw std::runtime_error("oversized or failed metadata read: " + path.string());
            return {bytes.data(), static_cast<size_t>(size)};
        }

        /** @brief Parse unsigned kernel counters without accepting signed text or overflow. */
        uint64_t integer(const std::string &text)
        {
            if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
                throw std::runtime_error("invalid unsigned kernel counter");
            return std::stoull(text);
        }

        /** @brief Convert byte-valued procfs fields with their explicit binary-kilobyte unit. */
        uint64_t scaled(uint64_t value, uint64_t factor)
        {
            if (factor == 0 || value > std::numeric_limits<uint64_t>::max() / factor)
                throw std::runtime_error("kernel byte counter overflow");
            return value * factor;
        }

        /** @brief Parse fixed key/value metadata; unsupported value forms are omitted, never zeroed. */
        json counters(const std::filesystem::path &path)
        {
            json result = json::object();
            std::istringstream lines(readMetadata(path));
            for (std::string line; std::getline(lines, line);)
            {
                std::istringstream fields(line);
                std::string name, text, unit;
                if (!(fields >> name >> text) || text.find_first_not_of("0123456789") != std::string::npos)
                    continue;
                if (name.ends_with(':')) name.pop_back();
                uint64_t value = integer(text);
                if (fields >> unit)
                {
                    if (unit != "kB") continue;
                    value = scaled(value, 1024);
                }
                result[name] = value;
            }
            return result;
        }

        /** @brief Mark an independently unavailable diagnostic section without fabricating metrics. */
        template<typename Function>
        json section(Function &&read)
        {
            try
            {
                json result = read();
                result["available"] = true;
                return result;
            }
            catch (const std::exception &error)
            {
                return {{"available", false}, {"error", error.what()}};
            }
        }

        /** @brief Copy required procfs fields under public names, retaining byte units. */
        json named(const json &source, std::initializer_list<std::pair<const char *, const char *>> names)
        {
            json result = json::object();
            for (const auto &[input, output] : names) result[output] = source.at(input);
            return result;
        }

        /** @brief Parse Linux pressure-stall totals separately from moving-average percentages. */
        json pressure(const std::filesystem::path &path)
        {
            json result = json::object();
            std::istringstream lines(readMetadata(path));
            for (std::string line; std::getline(lines, line);)
            {
                std::istringstream fields(line);
                std::string kind, token;
                fields >> kind;
                json values = json::object();
                while (fields >> token)
                {
                    const auto equal = token.find('=');
                    if (equal == std::string::npos) throw std::runtime_error("invalid pressure metadata");
                    const auto name = token.substr(0, equal), value = token.substr(equal + 1);
                    if (name == "total") values["total_microseconds"] = integer(value);
                    else
                    {
                        const double percent = std::stod(value);
                        if (!std::isfinite(percent) || percent < 0 || percent > 100)
                            throw std::runtime_error("invalid pressure percentage");
                        values[name + "_percent"] = percent;
                    }
                }
                result[kind] = std::move(values);
            }
            return result;
        }
    }

    HttpProcessStatsReader::HttpProcessStatsReader(std::vector<std::filesystem::path> filesystems,
        std::filesystem::path proc, std::filesystem::path cgroup)
        : filesystems_(std::move(filesystems)), proc_(std::move(proc)), cgroup_(std::move(cgroup))
    {
        for (auto &path : filesystems_) path = std::filesystem::absolute(path).lexically_normal();
        std::sort(filesystems_.begin(), filesystems_.end());
        filesystems_.erase(std::unique(filesystems_.begin(), filesystems_.end()), filesystems_.end());
        if (filesystems_.size() > 16) throw std::invalid_argument("at most 16 fixed filesystem probes are supported");
    }

    nlohmann::json HttpProcessStatsReader::operator()()
    {
        const auto now = std::chrono::steady_clock::now();
        const auto status = counters(proc_ / "self/status");
        json process = named(status, {{"VmRSS", "rss_bytes"}, {"VmHWM", "peak_rss_bytes"},
            {"VmSize", "virtual_bytes"}, {"RssAnon", "anonymous_rss_bytes"},
            {"RssFile", "file_rss_bytes"}, {"RssShmem", "shared_rss_bytes"},
            {"VmSwap", "swap_bytes"}, {"Threads", "threads"}, {"Pid", "pid"}});
        const auto raw_stat = readMetadata(proc_ / "self/stat");
        const auto end_name = raw_stat.rfind(')');
        if (end_name == std::string::npos) throw std::runtime_error("invalid process stat name");
        std::istringstream fields(raw_stat.substr(end_name + 1));
        std::vector<std::string> values;
        for (std::string value; fields >> value;) values.push_back(std::move(value));
        if (values.size() < 20) throw std::runtime_error("incomplete process stat metadata");
        const auto ticks = ::sysconf(_SC_CLK_TCK);
        if (ticks <= 0) throw std::runtime_error("invalid process clock tick rate");
        const double user = static_cast<double>(integer(values[11])) / ticks;
        const double system = static_cast<double>(integer(values[12])) / ticks;
        const double total = user + system;
        process["start_time_ticks"] = integer(values[19]);
        process["state"] = values[0];
        process["cpu"] = {{"user_seconds", user}, {"system_seconds", system},
            {"total_seconds", total}, {"interval_usage_percent", nullptr},
            {"usage_basis", "one_core_is_100_percent"}};
        if (previous_cpu_)
        {
            const double elapsed = std::chrono::duration<double>(now - previous_cpu_->first).count();
            if (elapsed <= 0 || total < previous_cpu_->second)
                throw std::runtime_error("process CPU counter/time regressed");
            process["cpu"]["interval_usage_percent"] = 100 * (total - previous_cpu_->second) / elapsed;
        }
        process["io"] = section([&] { return counters(proc_ / "self/io"); });
        process["file_descriptors"] = section([&] {
            constexpr size_t limit = 65536;
            size_t count = 0;
            for ([[maybe_unused]] const auto &entry : std::filesystem::directory_iterator(proc_ / "self/fd"))
                if (++count == limit) break;
            struct rlimit bound{};
            if (::getrlimit(RLIMIT_NOFILE, &bound) != 0) throw std::runtime_error("getrlimit failed");
            return json{{"count", count == limit ? json(nullptr) : json(count)},
                {"count_lower_bound", count}, {"truncated", count == limit},
                {"soft_limit", bound.rlim_cur == RLIM_INFINITY ? json(nullptr) : json(bound.rlim_cur)}};
        });
        process["available"] = true;

        json host = section([&] {
            auto memory = named(counters(proc_ / "meminfo"), {{"MemTotal", "total_bytes"},
                {"MemAvailable", "available_bytes"}, {"MemFree", "free_bytes"},
                {"Cached", "cached_bytes"}, {"Buffers", "buffers_bytes"},
                {"SwapTotal", "swap_total_bytes"}, {"SwapFree", "swap_free_bytes"},
                {"Dirty", "dirty_bytes"}, {"Writeback", "writeback_bytes"}});
            std::istringstream load(readMetadata(proc_ / "loadavg"));
            double one, five, fifteen;
            if (!(load >> one >> five >> fifteen) || !std::isfinite(one) || !std::isfinite(five) ||
                !std::isfinite(fifteen) || std::min({one, five, fifteen}) < 0)
                throw std::runtime_error("invalid host load average");
            json result{{"memory", std::move(memory)}, {"load_average", {one, five, fifteen}},
                        {"logical_cpus", ::sysconf(_SC_NPROCESSORS_ONLN)}};
            for (const auto *kind : {"cpu", "memory", "io"})
                result["pressure"][kind] = section([&] { return pressure(proc_ / "pressure" / kind); });
            return result;
        });
        json cgroup = section([&] {
            std::istringstream lines(readMetadata(proc_ / "self/cgroup"));
            std::filesystem::path relative;
            bool found = false;
            for (std::string line; std::getline(lines, line);)
                if (line.starts_with("0::/")) { relative = line.substr(4); found = true; break; }
            if (!found) throw std::runtime_error("unified cgroup v2 metadata unavailable");
            for (const auto &component : relative)
                if (component == ".." || component == ".") throw std::runtime_error("invalid cgroup identity");
            const auto root = cgroup_ / relative;
            json result{{"version", 2}, {"scope", "current_cgroup"}, {"path", "/" + relative.string()}};
            for (const auto &[file, output] : std::initializer_list<std::pair<const char *, const char *>>{
                    {"memory.current", "memory_current_bytes"}, {"memory.peak", "memory_peak_bytes"},
                    {"memory.max", "memory_limit_bytes"}, {"memory.swap.current", "swap_current_bytes"}})
            {
                std::istringstream stream(readMetadata(root / file));
                std::string value;
                stream >> value;
                result[output] = value == "max" ? json(nullptr) : json(integer(value));
            }
            result["memory_stat"] = counters(root / "memory.stat");
            result["memory_events"] = counters(root / "memory.events");
            result["cpu_stat"] = counters(root / "cpu.stat");
            result["memory_accounting"] = "file_includes_shmem; process_RSS_and_cgroup_are_distinct_scopes";
            for (const auto *kind : {"cpu", "memory", "io"})
                result["pressure"][kind] = section([&] { return pressure(root / (std::string(kind) + ".pressure")); });
            return result;
        });
        json filesystems = json::array();
        for (const auto &path : filesystems_)
        {
            auto observation = section([&] {
                struct statvfs value{};
                if (::statvfs(path.c_str(), &value) != 0)
                    throw std::system_error(errno, std::generic_category(), "statvfs");
                const auto capacity = scaled(value.f_blocks, value.f_frsize);
                const auto free = scaled(value.f_bfree, value.f_frsize);
                const auto available = scaled(value.f_bavail, value.f_frsize);
                if (free > capacity || available > free) throw std::runtime_error("invalid filesystem capacity");
                return json{{"filesystem_id", value.f_fsid}, {"capacity_bytes", capacity},
                    {"free_bytes", free}, {"available_bytes", available}, {"used_bytes", capacity - free},
                    {"utilization_percent", capacity ? json(100.0 * (capacity - free) / capacity) : json(nullptr)},
                    {"total_inodes", value.f_files}, {"available_inodes", value.f_favail},
                    {"scope", "filesystem_not_directory_size"}};
            });
            observation["path"] = path.string();
            filesystems.push_back(std::move(observation));
        }
        previous_cpu_ = std::pair{now, total};
        return {{"scope", "serving_process"}, {"process", std::move(process)},
                {"host", std::move(host)}, {"cgroup", std::move(cgroup)}, {"filesystems", std::move(filesystems)}};
    }

    HttpProcessStats::HttpProcessStats(HttpProcessStatsOptions options, Collector collector)
        : interval_(options.interval), collector_(std::move(collector)),
          publication_(std::make_shared<const Publication>())
    {
        if (interval_.count() <= 0) throw std::invalid_argument("process stats interval must be positive");
        if (!collector_) collector_ = HttpProcessStatsReader(std::move(options.filesystems));
        worker_ = std::jthread([this](std::stop_token stop) { collect(stop); });
    }

    HttpProcessStats::~HttpProcessStats()
    {
        worker_.request_stop();
        wake_.notify_all();
        worker_.join();
    }

    void HttpProcessStats::collect(std::stop_token stop)
    {
        while (!stop.stop_requested())
        {
            auto next = std::make_shared<Publication>(*publication_.load());
            const auto started = Clock::now();
            const auto wall = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
            ++next->attempts;
            try
            {
                auto data = collector_();
                if (!data.is_object()) throw std::runtime_error("process collector did not return an object");
                next->data = std::move(data);
                next->sampled_at = started;
                next->sampled_unix_seconds = wall;
                ++next->successful_samples;
                next->error.clear();
            }
            catch (const std::exception &error) { next->error = error.what(); }
            catch (...) { next->error = "unknown process collection failure"; }
            next->collection_seconds = std::chrono::duration<double>(Clock::now() - started).count();
            publication_.store(std::move(next));
            std::unique_lock lock(wait_mutex_);
            wake_.wait_for(lock, stop, interval_, [] { return false; });
        }
    }

    nlohmann::json HttpProcessStats::snapshot(Clock::time_point now) const
    {
        const auto current = publication_.load();
        const bool available = current->successful_samples != 0;
        const double interval = std::chrono::duration<double>(interval_).count();
        const double age = available ? std::max(0.0, std::chrono::duration<double>(now - current->sampled_at).count()) : 0;
        auto result = current->data.is_null() ? json::object() : current->data;
        result["collection"] = {{"available", available},
            {"status", available ? "ready" : current->attempts ? "unavailable" : "warming"},
            {"interval_seconds", interval}, {"attempts", current->attempts},
            {"successful_samples", current->successful_samples},
            {"sampled_unix_seconds", available ? json(current->sampled_unix_seconds) : json(nullptr)},
            {"age_seconds", available ? json(age) : json(nullptr)},
            {"stale", !available || age > interval * 2}, {"collection_seconds", current->collection_seconds},
            {"last_error", current->error.empty() ? json(nullptr) : json(current->error)}};
        return result;
    }
}
