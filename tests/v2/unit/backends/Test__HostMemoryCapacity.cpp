/**
 * @file Test__HostMemoryCapacity.cpp
 * @brief Unit proofs for canonical host/NUMA admission-memory accounting.
 *
 * These tests model the production parity layout directly: a large immutable
 * tmpfs model remains on the anonymous LRU, while cold source-GGUF filesystem
 * cache is reported independently on the file LRU and is reclaimable. The
 * suite is pure and device-free so the accounting contract remains part of the
 * fast unit gate.
 */

#include <gtest/gtest.h>

#include "backends/HostMemoryCapacity.h"

#include <cstddef>
#include <string>

namespace llaminar2::test
{
    namespace
    {
        constexpr std::size_t kKiB = 1024;
    }

    TEST(Test__HostMemoryCapacity, ParsesNumaGrammarWithoutDoubleChargingTmpfs)
    {
        const auto observation = parseLinuxHostMemoryInfo(
            "Node 0 MemTotal:       100000 kB\n"
            "Node 0 MemFree:         10000 kB\n"
            "Node 0 Inactive(file):  50000 kB\n"
            "Node 0 Shmem:           20000 kB\n");

        ASSERT_TRUE(observation.valid());
        EXPECT_EQ(observation.total_bytes, 100000 * kKiB);
        EXPECT_EQ(observation.free_bytes, 10000 * kKiB);
        EXPECT_EQ(observation.inactive_file_bytes, 50000 * kKiB);
        EXPECT_EQ(observation.shared_memory_bytes, 20000 * kKiB);
        EXPECT_EQ(observation.admission_available_bytes, 60000 * kKiB);
    }

    TEST(Test__HostMemoryCapacity, SharedMemoryDoesNotOverlapTheFileLru)
    {
        const auto observation = parseLinuxHostMemoryInfo(
            "Node 1 MemTotal:       100000 kB\n"
            "Node 1 MemFree:         12000 kB\n"
            "Node 1 Inactive(file):   4000 kB\n"
            "Node 1 Shmem:           30000 kB\n");

        EXPECT_EQ(observation.admission_available_bytes, 16000 * kKiB);
    }

    TEST(Test__HostMemoryCapacity, CapsReclaimableEstimateAtPhysicalTotal)
    {
        const auto observation = parseLinuxHostMemoryInfo(
            "MemTotal:       100000 kB\n"
            "MemFree:         80000 kB\n"
            "Inactive(file):  50000 kB\n"
            "Shmem:               0 kB\n");

        EXPECT_EQ(observation.admission_available_bytes, 100000 * kKiB);
    }

    TEST(Test__HostMemoryCapacity, UsesKernelAvailableWhenLruDetailIsAbsent)
    {
        const auto observation = parseLinuxHostMemoryInfo(
            "MemTotal:       100000 kB\n"
            "MemFree:         10000 kB\n"
            "MemAvailable:    70000 kB\n");

        EXPECT_EQ(observation.kernel_available_bytes, 70000 * kKiB);
        EXPECT_EQ(observation.admission_available_bytes, 70000 * kKiB);
    }

    TEST(Test__HostMemoryCapacity, MissingPhysicalTotalCannotAuthorizeMemory)
    {
        const auto observation = parseLinuxHostMemoryInfo(
            "MemFree:         10000 kB\n"
            "Inactive(file):  50000 kB\n");

        EXPECT_FALSE(observation.valid());
        EXPECT_EQ(observation.admission_available_bytes, 0u);
    }

    TEST(Test__HostMemoryCapacity, LiveSystemObservationIsBounded)
    {
        const auto observation = observeSystemMemoryCapacity();
        ASSERT_TRUE(observation.valid());
        EXPECT_GT(observation.admission_available_bytes, 0u);
        EXPECT_LE(
            observation.admission_available_bytes,
            observation.total_bytes);
    }
} // namespace llaminar2::test
