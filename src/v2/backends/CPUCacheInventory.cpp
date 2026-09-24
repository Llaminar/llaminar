/**
 * @file CPUCacheInventory.cpp
 * @brief Read cache sharing identities once, without socket/processor-name guesses.
 *
 * Linux supplies canonical shared_cpu_list identities. Equal identities at the
 * same cache level must have equal capacities; each contributes exactly once.
 * Instruction-only caches are excluded. Incomplete observations stay unknown,
 * so a later streaming probe cannot mistake a cache-sized buffer for DRAM work.
 */
#include "CPUCacheInventory.h"
#include <charconv>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    namespace
    {
        /** @return One whitespace-delimited sysfs value, empty when unavailable. */
        std::string read(const std::filesystem::path &path)
        {
            std::ifstream stream(path);
            std::string value;
            stream >> value;
            return value;
        }

        /** @return Exact positive decimal value with an optional binary K/M/G suffix. */
        size_t quantity(std::string value, bool allow_suffix)
        {
            size_t multiplier = 1;
            if (allow_suffix && !value.empty())
            {
                switch (value.back())
                {
                case 'K': multiplier = size_t{1} << 10; value.pop_back(); break;
                case 'M': multiplier = size_t{1} << 20; value.pop_back(); break;
                case 'G': multiplier = size_t{1} << 30; value.pop_back(); break;
                }
            }
            size_t result = 0;
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
            if (parsed.ec == std::errc::result_out_of_range || result > std::numeric_limits<size_t>::max() / multiplier)
                throw std::overflow_error("CPU cache observation overflow");
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !result)
                throw std::invalid_argument("Malformed CPU cache observation");
            return result * multiplier;
        }
    }

    size_t observeCPUSharedCacheBytes(std::span<const int> cores, const std::filesystem::path &cpu_root)
    {
        std::map<std::pair<size_t, std::string>, size_t> domains;
        bool complete = !cores.empty();
        for (int core : cores)
        {
            if (core < 0) throw std::invalid_argument("Negative CPU cache observer ID");
            const auto directory = cpu_root / ("cpu" + std::to_string(core)) / "cache";
            if (!std::filesystem::is_directory(directory)) { complete = false; continue; }
            size_t highest_level = 0, highest_bytes = 0;
            std::string highest_peers;
            for (const auto &entry : std::filesystem::directory_iterator(directory))
            {
                if (!entry.path().filename().string().starts_with("index")) continue;
                const auto type = read(entry.path() / "type");
                if (type.empty()) { complete = false; continue; }
                if (type != "Data" && type != "Unified") continue;
                const auto level_text = read(entry.path() / "level");
                const auto size_text = read(entry.path() / "size");
                const auto peers = read(entry.path() / "shared_cpu_list");
                if (level_text.empty() || size_text.empty() || peers.empty()) { complete = false; continue; }
                const auto level = quantity(level_text, false), bytes = quantity(size_text, true);
                if (level == highest_level && (bytes != highest_bytes || peers != highest_peers))
                    throw std::invalid_argument("Contradictory highest CPU cache domains");
                if (level > highest_level) { highest_level = level; highest_bytes = bytes; highest_peers = peers; }
            }
            if (!highest_level) { complete = false; continue; }
            const auto [found, inserted] = domains.emplace(std::make_pair(highest_level, highest_peers), highest_bytes);
            if (!inserted && found->second != highest_bytes)
                throw std::invalid_argument("Conflicting capacities for shared CPU cache domain");
        }
        if (!complete) return 0;
        size_t total = 0;
        for (const auto &[identity, bytes] : domains)
        {
            if (bytes > std::numeric_limits<size_t>::max() - total)
                throw std::overflow_error("Aggregate CPU cache observation overflow");
            total += bytes;
        }
        return total;
    }
}
