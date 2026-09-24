/**
 * @file Test__CPUCacheInventory.cpp
 * @brief Device-free sysfs fixtures for shared-cache identity and completeness.
 *
 * Unequal chiplets and repeated SMT/cache aliases must not become a single
 * guessed socket cache. The fixture owns only its mkdtemp directory and never
 * changes host topology or interrogates a real accelerator.
 */
#include "backends/CPUCacheInventory.h"
#include "backends/HardwareInventory.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
#include <fstream>
#include <limits>

using namespace llaminar2;

namespace
{
    /** @brief Own a disposable, synthetic Linux CPU cache tree. */
    class CPUCacheInventoryTest : public ::testing::Test
    {
    protected:
        std::filesystem::path root;

        /** @brief Create one unique directory, without touching the real sysfs. */
        void SetUp() override
        {
            char name[] = "/tmp/llaminar-cache-inventory-XXXXXX";
            const auto created = mkdtemp(name);
            ASSERT_NE(created, nullptr);
            root = created;
        }

        /** @brief Remove only this test's positively identified temporary tree. */
        void TearDown() override
        {
            if (!root.empty()) std::filesystem::remove_all(root);
        }

        /** @brief Publish one cache observation exactly as Linux exposes it. */
        void cache(int cpu, int index, int level, std::string size, std::string peers,
            std::string type = "Unified")
        {
            const auto path = root / ("cpu" + std::to_string(cpu)) / "cache" / ("index" + std::to_string(index));
            std::filesystem::create_directories(path);
            std::ofstream(path / "level") << level << '\n';
            std::ofstream(path / "type") << type << '\n';
            std::ofstream(path / "size") << size << '\n';
            std::ofstream(path / "shared_cpu_list") << peers << '\n';
        }
    };
}

TEST_F(CPUCacheInventoryTest, UnequalChipletsCountOnceEachAndIgnoreInstructionCache)
{
    const std::array cores{2, 4, 9};
    for (int cpu : cores)
    {
        cache(cpu, 0, 1, "32K", std::to_string(cpu), "Data");
        cache(cpu, 1, 2, "1024K", std::to_string(cpu));
        cache(cpu, 2, 3, cpu == 9 ? "96M" : "32M", cpu == 9 ? "9,11" : "2,4,6,8");
        cache(cpu, 3, 4, "1G", "0-127", "Instruction");
    }
    EXPECT_EQ(observeCPUSharedCacheBytes(cores, root), 128u << 20);
    EXPECT_EQ(observeCPUSharedCacheBytes(std::array{2, 2, 4}, root), 32u << 20);
}

TEST_F(CPUCacheInventoryTest, MissingMetadataCannotMasqueradeAsCompleteCapacity)
{
    cache(2, 0, 2, "1024K", "2");
    cache(2, 1, 3, "32M", "2,4");
    EXPECT_EQ(observeCPUSharedCacheBytes(std::array{2, 4}, root), 0u);
    std::filesystem::remove(root / "cpu2/cache/index1/size");
    EXPECT_EQ(observeCPUSharedCacheBytes(std::array{2}, root), 0u);
    EXPECT_EQ(observeCPUSharedCacheBytes({}, root), 0u);
}

TEST_F(CPUCacheInventoryTest, ConflictingDomainAndMalformedSizeAreRejected)
{
    cache(2, 0, 3, "32M", "2,4");
    cache(4, 0, 3, "64M", "2,4");
    EXPECT_THROW(observeCPUSharedCacheBytes(std::array{2, 4}, root), std::invalid_argument);
    cache(4, 0, 3, "64Mgarbage", "4");
    EXPECT_THROW(observeCPUSharedCacheBytes(std::array{4}, root), std::invalid_argument);
    cache(4, 0, 3, "0", "4");
    EXPECT_THROW(observeCPUSharedCacheBytes(std::array{4}, root), std::invalid_argument);
    EXPECT_THROW(observeCPUSharedCacheBytes(std::array{-1}, root), std::invalid_argument);
}

TEST_F(CPUCacheInventoryTest, IndividualAndAggregateOverflowAreRejected)
{
    cache(2, 0, 3, std::to_string(std::numeric_limits<size_t>::max()), "2");
    cache(4, 0, 3, "1", "4");
    EXPECT_THROW(observeCPUSharedCacheBytes(std::array{2, 4}, root), std::overflow_error);
    cache(2, 0, 3, std::to_string(std::numeric_limits<size_t>::max()) + "K", "2");
    EXPECT_THROW(observeCPUSharedCacheBytes(std::array{2}, root), std::overflow_error);
}

TEST(CPUCacheInventoryProjection, PartialSocketObservationIsUnknownNotPartialSum)
{
    HardwareInventory inventory;
    inventory.cpu_sockets = {{.socket_id = 0, .numa_node = 3}, {.socket_id = 1, .numa_node = 7}};
    inventory.cpu_last_level_cache_bytes = {{0, 32u << 20}};
    EXPECT_EQ(inventory.cpuDevice().last_level_cache_bytes, 0u);
    EXPECT_EQ(inventory.cpuDevice(3).last_level_cache_bytes, 32u << 20);
    EXPECT_EQ(inventory.cpuDevice(7).last_level_cache_bytes, 0u);
}
