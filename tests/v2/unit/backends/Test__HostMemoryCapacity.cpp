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
#include "backends/HardwareInventory.h"

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
    /**
     * @brief A host without NUMA memory topology is one memory domain.
     *
     * Regression: on Docker Desktop's linuxkit kernel (no node sysfs, libnuma
     * unavailable) every socket kept zero memory and `llaminar2 plan` failed
     * with "Physical memory BOM requires a valid resource" before reading the
     * model. The single package now receives the system observation; ambiguous
     * multi-package or partial topologies fail with a precise diagnostic.
     */
    TEST(HostMemoryCapacity, NonNUMAHostAttributesSystemMemoryToItsSinglePackage)
    {
        HostMemoryCapacityObservation system;
        system.total_bytes = 32ull << 30;
        system.free_bytes = 30ull << 30;
        system.admission_available_bytes = 29ull << 30;

        std::vector<CPUSocketInfo> single(1);
        single[0].socket_id = 0;
        single[0].numa_node = 0;
        assignNonNUMAHostMemory(single, system);
        EXPECT_EQ(single[0].memory_bytes, system.total_bytes);
        EXPECT_EQ(single[0].available_memory_bytes, system.admission_available_bytes);

        // NUMA-observed sockets are never replaced.
        std::vector<CPUSocketInfo> numa(2);
        numa[0].memory_bytes = numa[1].memory_bytes = 64ull << 30;
        numa[0].available_memory_bytes = numa[1].available_memory_bytes = 60ull << 30;
        assignNonNUMAHostMemory(numa, system);
        EXPECT_EQ(numa[0].memory_bytes, 64ull << 30);
        EXPECT_EQ(numa[1].available_memory_bytes, 60ull << 30);

        std::vector<CPUSocketInfo> ambiguous(2);
        EXPECT_THROW(assignNonNUMAHostMemory(ambiguous, system), std::runtime_error);
        std::vector<CPUSocketInfo> partial(2);
        partial[0].memory_bytes = 1;
        EXPECT_THROW(assignNonNUMAHostMemory(partial, system), std::runtime_error);
        std::vector<CPUSocketInfo> unobservable(1);
        EXPECT_THROW(assignNonNUMAHostMemory(unobservable, HostMemoryCapacityObservation{}), std::runtime_error);
    }
    /**
     * Regression (DGX Spark/GB10): an integrated GPU's driver free figure
     * tracks MemFree and omitted 39 GiB of reclaimable page cache, so a model
     * that fit the shared pool was rejected ("only 78.1 GB available") until
     * the cache was dropped by hand. The accelerator member of a shared pool
     * must admit the same canonical host observation as the CPU member,
     * bounded by its own addressable total, and must never invent capacity
     * for an unobservable pool.
     */
    TEST(HostMemoryCapacity, IntegratedAcceleratorAdmitsItsReclaimableHostPool)
    {
        // The Spark observation: 121 GiB total, 78 GiB MemFree, 39 GiB cold cache.
        const auto pool = parseLinuxHostMemoryInfo(
            "MemTotal:       127535104 kB\n"
            "MemFree:         81788928 kB\n"
            "MemAvailable:   121905332 kB\n"
            "Inactive(file):  38797312 kB\n"
            "Shmem:              80324 kB\n");
        ASSERT_TRUE(pool.valid());
        const std::size_t driver_total = 125000000000ull;
        const auto available = integratedAcceleratorPoolAvailableBytes(pool, driver_total);
        EXPECT_EQ(available, (81788928ull + 38797312ull) * kKiB);
        EXPECT_GT(available, pool.free_bytes) << "reclaimable cache must be admitted";

        // The accelerator can never address more than its own total.
        EXPECT_EQ(integratedAcceleratorPoolAvailableBytes(pool, 64ull << 30), 64ull << 30);

        EXPECT_THROW((void)integratedAcceleratorPoolAvailableBytes(HostMemoryCapacityObservation{}, driver_total),
                     std::runtime_error);
        EXPECT_THROW((void)integratedAcceleratorPoolAvailableBytes(pool, 0u), std::runtime_error);
    }

    /** Dedicated devices keep their driver figure; pooled ones adopt the pool. */
    TEST(HostMemoryCapacity, OnlyRankHostPoolDevicesAdoptThePoolObservation)
    {
        ComputeDevice dedicated{};
        dedicated.total_memory_bytes = 24ull << 30;
        dedicated.free_memory_bytes = 3ull << 30;
        dedicated.numa_node = -1;
        dedicated.memory_backing = PhysicalMemoryBacking::Dedicated;
        adoptIntegratedAcceleratorPoolCapacity(dedicated);
        EXPECT_EQ(dedicated.free_memory_bytes, 3ull << 30);

        ComputeDevice pooled = dedicated;
        pooled.memory_backing = PhysicalMemoryBacking::RankHostPool;
        pooled.total_memory_bytes = std::size_t{1} << 50; // bound by the live pool, not the device
        pooled.free_memory_bytes = 1u;
        adoptIntegratedAcceleratorPoolCapacity(pooled);
        const auto live = observeIntegratedAcceleratorPool(-1);
        ASSERT_TRUE(live.valid());
        // A live observation moves between samples; it must be the pool's
        // order of magnitude, never the 1-byte driver figure.
        EXPECT_GT(pooled.free_memory_bytes, 1u);
        EXPECT_LE(pooled.free_memory_bytes, live.total_bytes);
    }
} // namespace llaminar2::test
