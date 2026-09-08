/**
 * @file HostMemoryCapacity.cpp
 * @brief Canonical host/NUMA memory observation implementation.
 *
 * The implementation intentionally performs one read per observation and
 * derives every published field from those same bytes.  This prevents a
 * capacity decision from mixing totals and availability sampled at different
 * points in a rapidly changing model-load lifecycle.
 */

#include "HostMemoryCapacity.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numa.h>
#include <sstream>
#include <string>

namespace llaminar2
{
    namespace
    {
        /** @brief Saturating byte addition for kernel counters. */
        [[nodiscard]] std::size_t saturatedAdd(
            std::size_t left,
            std::size_t right) noexcept
        {
            if (right > std::numeric_limits<std::size_t>::max() - left)
                return std::numeric_limits<std::size_t>::max();
            return left + right;
        }

        /** @brief Convert a kernel memory-info value to bytes without wrap. */
        [[nodiscard]] std::size_t valueToBytes(
            std::uint64_t value,
            const std::string &unit) noexcept
        {
            const std::uint64_t multiplier =
                unit.empty() ? 1ULL : unit == "kB" ? 1024ULL : 0ULL;
            if (multiplier == 0ULL ||
                value > std::numeric_limits<std::size_t>::max() / multiplier)
            {
                return 0;
            }
            return static_cast<std::size_t>(value * multiplier);
        }

        /** @brief Read a small kernel text file completely. */
        [[nodiscard]] std::string readTextFile(const std::string &path)
        {
            std::ifstream stream(path);
            if (!stream.is_open())
                return {};
            return std::string(
                std::istreambuf_iterator<char>(stream),
                std::istreambuf_iterator<char>());
        }

        /** @brief Finalize the one conservative admission-memory definition. */
        void finalizeObservation(
            HostMemoryCapacityObservation &observation,
            bool saw_inactive_file) noexcept
        {
            std::size_t available = observation.free_bytes;
            if (saw_inactive_file)
            {
                /*
                 * Linux classifies regular filesystem-backed folios on the file
                 * LRU and swap-backed tmpfs/shmem folios on the anonymous LRU.
                 * Inactive(file) is therefore already the discardable cache
                 * class. Subtracting the independently reported Shmem counter
                 * would double-charge every ramdisk page and collapse a node's
                 * admission authority to raw MemFree under a large tmpfs model.
                 */
                available = saturatedAdd(
                    observation.free_bytes,
                    observation.inactive_file_bytes);
            }
            else if (observation.kernel_available_bytes > 0)
            {
                // Older/non-NUMA views without LRU detail retain the kernel's
                // own availability estimate instead of falling back to total.
                available = observation.kernel_available_bytes;
            }

            observation.admission_available_bytes =
                observation.total_bytes > 0
                    ? std::min(available, observation.total_bytes)
                    : 0;
        }
    } // namespace

    HostMemoryCapacityObservation parseLinuxHostMemoryInfo(
        std::string_view text)
    {
        HostMemoryCapacityObservation observation;
        bool saw_inactive_file = false;
        std::istringstream lines(std::string{text});
        std::string line;
        while (std::getline(lines, line))
        {
            const std::size_t colon = line.find(':');
            if (colon == std::string::npos)
                continue;

            // NUMA sysfs prepends "Node N "; the final whitespace-delimited
            // token before ':' is the same canonical field name as /proc.
            const std::string prefix = line.substr(0, colon);
            const std::size_t field_start = prefix.find_last_of(" \t");
            const std::string field =
                field_start == std::string::npos
                    ? prefix
                    : prefix.substr(field_start + 1);

            std::istringstream value_stream(line.substr(colon + 1));
            std::uint64_t value = 0;
            std::string unit;
            if (!(value_stream >> value))
                continue;
            (void)(value_stream >> unit);
            const std::size_t bytes = valueToBytes(value, unit);

            if (field == "MemTotal")
                observation.total_bytes = bytes;
            else if (field == "MemFree")
                observation.free_bytes = bytes;
            else if (field == "MemAvailable")
                observation.kernel_available_bytes = bytes;
            else if (field == "Inactive(file)")
            {
                observation.inactive_file_bytes = bytes;
                saw_inactive_file = true;
            }
            else if (field == "Shmem")
                observation.shared_memory_bytes = bytes;
        }

        finalizeObservation(observation, saw_inactive_file);
        return observation;
    }

    HostMemoryCapacityObservation observeNUMAMemoryCapacity(int numa_node)
    {
        if (numa_node < 0)
            return {};

        const std::string path =
            "/sys/devices/system/node/node" +
            std::to_string(numa_node) + "/meminfo";
        auto observation = parseLinuxHostMemoryInfo(readTextFile(path));
        if (observation.valid())
            return observation;

        if (numa_available() < 0)
            return {};
        long long free_bytes = 0;
        const long long total_bytes =
            numa_node_size64(numa_node, &free_bytes);
        if (total_bytes <= 0)
            return {};
        observation.total_bytes = static_cast<std::size_t>(total_bytes);
        observation.free_bytes =
            free_bytes > 0 ? static_cast<std::size_t>(free_bytes) : 0;
        observation.admission_available_bytes = std::min(
            observation.free_bytes,
            observation.total_bytes);
        return observation;
    }

    HostMemoryCapacityObservation observeSystemMemoryCapacity()
    {
        return parseLinuxHostMemoryInfo(readTextFile("/proc/meminfo"));
    }
} // namespace llaminar2
