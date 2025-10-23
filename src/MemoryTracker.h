/**
 * @file MemoryTracker.h
 * @brief Memory usage tracking utility for benchmarking
 * @author David Sanftenberg
 * @date 2025-10-20
 */

#pragma once

#include <cstddef>
#include <string>
#include <fstream>
#include <sstream>

namespace llaminar
{

    /**
     * @class MemoryTracker
     * @brief Tracks process memory usage via /proc/self/status
     *
     * Provides utilities to measure resident memory (VmRSS) for benchmarking
     * memory consumption with different configurations.
     */
    class MemoryTracker
    {
    public:
        /**
         * @brief Get current resident memory usage in bytes
         * @return Resident memory (VmRSS) in bytes, or 0 if unavailable
         */
        static size_t getResidentMemoryBytes()
        {
            std::ifstream status("/proc/self/status");
            if (!status.is_open())
            {
                return 0;
            }

            std::string line;
            while (std::getline(status, line))
            {
                if (line.substr(0, 6) == "VmRSS:")
                {
                    std::istringstream iss(line.substr(6));
                    size_t memory_kb;
                    iss >> memory_kb;
                    return memory_kb * 1024; // Convert KB to bytes
                }
            }

            return 0;
        }

        /**
         * @brief Get current resident memory usage in megabytes
         * @return Resident memory (VmRSS) in MB, or 0 if unavailable
         */
        static double getResidentMemoryMB()
        {
            return static_cast<double>(getResidentMemoryBytes()) / (1024.0 * 1024.0);
        }

        /**
         * @brief Format memory size as human-readable string
         * @param bytes Memory size in bytes
         * @return Formatted string (e.g., "256.5 MB")
         */
        static std::string formatBytes(size_t bytes)
        {
            const double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
            const double gb = mb / 1024.0;

            std::ostringstream oss;
            oss.precision(2);
            oss << std::fixed;

            if (gb >= 1.0)
            {
                oss << gb << " GB";
            }
            else
            {
                oss << mb << " MB";
            }

            return oss.str();
        }
    };

} // namespace llaminar
