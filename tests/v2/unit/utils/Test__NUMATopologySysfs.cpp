/**
 * @file Test__NUMATopologySysfs.cpp
 * @brief Unit tests for NUMA sysfs detection - hardware-independent using temp files
 * @author David Sanftenberg
 * @date January 2026
 *
 * These tests execute CPU ownership resolution over authored procfs/sysfs
 * fixtures, as well as historical GPU metadata fixture checks, without GPUs.
 * They create temporary files that mimic /sys/bus/pci/devices/<bus_id>/numa_node
 */

#include <gtest/gtest.h>
#include "utils/NUMATopology.h"
#include "utils/NodeTopology.h"
#include <limits>
#include <filesystem>
#include <fstream>

using namespace llaminar2;
namespace fs = std::filesystem;

// =============================================================================
// Test Fixture - Creates temp directory structure mimicking sysfs
// =============================================================================

class Test__Unit_NUMATopologySysfs : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create temp directory for mock sysfs
        temp_dir_ = fs::temp_directory_path() / ("numa_test_" + std::to_string(getpid()));
        fs::create_directories(temp_dir_);
    }

    void TearDown() override
    {
        // Cleanup temp directory
        if (fs::exists(temp_dir_))
        {
            fs::remove_all(temp_dir_);
        }
    }

    /**
     * @brief Create a mock numa_node file with specified content
     * @param pci_bus_id The PCI bus ID (e.g., "0000:3b:00.0")
     * @param numa_node_value The value to write to numa_node file (-1, 0, 1, etc.)
     * @return Full path to the mock sysfs device directory
     */
    std::string createMockNUMANode(const std::string &pci_bus_id, int numa_node_value)
    {
        fs::path device_dir = temp_dir_ / pci_bus_id;
        fs::create_directories(device_dir);

        fs::path numa_file = device_dir / "numa_node";
        std::ofstream ofs(numa_file);
        ofs << numa_node_value;
        ofs.close();

        return device_dir.string();
    }

    fs::path temp_dir_;
};

// =============================================================================
// detectGPUViaSysfs Tests (via test helper)
// =============================================================================

// Note: detectGPUViaSysfs is private, so we test it indirectly through
// a test-accessible wrapper or by verifying full NUMA detection behavior.
// For now we use the filesystem to verify the logic would work.

TEST_F(Test__Unit_NUMATopologySysfs, MockSysfs_ReadsNUMANode0)
{
    // Create mock sysfs structure
    createMockNUMANode("0000:3b:00.0", 0);

    // Read the file directly to verify our mock setup works
    fs::path numa_file = temp_dir_ / "0000:3b:00.0" / "numa_node";
    std::ifstream ifs(numa_file);
    ASSERT_TRUE(ifs.is_open()) << "Failed to open mock numa_node file";

    int value;
    ifs >> value;
    EXPECT_EQ(value, 0);
}

TEST_F(Test__Unit_NUMATopologySysfs, MockSysfs_ReadsNUMANode1)
{
    createMockNUMANode("0000:af:00.0", 1);

    fs::path numa_file = temp_dir_ / "0000:af:00.0" / "numa_node";
    std::ifstream ifs(numa_file);
    ASSERT_TRUE(ifs.is_open());

    int value;
    ifs >> value;
    EXPECT_EQ(value, 1);
}

TEST_F(Test__Unit_NUMATopologySysfs, MockSysfs_HandlesNegativeOne)
{
    // Some systems return -1 for devices without NUMA affinity
    createMockNUMANode("0000:00:00.0", -1);

    fs::path numa_file = temp_dir_ / "0000:00:00.0" / "numa_node";
    std::ifstream ifs(numa_file);
    ASSERT_TRUE(ifs.is_open());

    int value;
    ifs >> value;
    EXPECT_EQ(value, -1);
}

TEST_F(Test__Unit_NUMATopologySysfs, MockSysfs_MultipleDevices)
{
    // Create multiple mock devices on different NUMA nodes
    createMockNUMANode("0000:3b:00.0", 0); // CUDA GPU0 on NUMA 0
    createMockNUMANode("0000:af:00.0", 1); // CUDA GPU1 on NUMA 1
    createMockNUMANode("0000:c1:00.0", 0); // ROCm GPU0 on NUMA 0
    createMockNUMANode("0000:c3:00.0", 1); // ROCm GPU1 on NUMA 1

    // Verify each
    std::vector<std::pair<std::string, int>> expected = {
        {"0000:3b:00.0", 0},
        {"0000:af:00.0", 1},
        {"0000:c1:00.0", 0},
        {"0000:c3:00.0", 1}};

    for (const auto &[bus_id, expected_numa] : expected)
    {
        fs::path numa_file = temp_dir_ / bus_id / "numa_node";
        std::ifstream ifs(numa_file);
        ASSERT_TRUE(ifs.is_open()) << "Failed to open " << numa_file;

        int value;
        ifs >> value;
        EXPECT_EQ(value, expected_numa)
            << "PCI bus " << bus_id << " expected NUMA " << expected_numa << " got " << value;
    }
}

TEST_F(Test__Unit_NUMATopologySysfs, MockSysfs_MissingFile)
{
    // Don't create the numa_node file, just the directory
    fs::path device_dir = temp_dir_ / "0000:xx:00.0";
    fs::create_directories(device_dir);

    fs::path numa_file = device_dir / "numa_node";
    std::ifstream ifs(numa_file);

    // Should fail to open
    EXPECT_FALSE(ifs.is_open());
}

// =============================================================================
// PCI Bus ID Format Tests
// =============================================================================

TEST_F(Test__Unit_NUMATopologySysfs, PCIBusID_ValidFormat)
{
    // Standard PCI bus ID format: domain:bus:device.function
    std::vector<std::string> valid_ids = {
        "0000:3b:00.0",
        "0000:af:00.0",
        "0000:00:00.0",
        "0001:00:00.0", // Non-zero domain
        "0000:ff:1f.7"  // Max values
    };

    for (const auto &bus_id : valid_ids)
    {
        createMockNUMANode(bus_id, 0);
        fs::path numa_file = temp_dir_ / bus_id / "numa_node";
        EXPECT_TRUE(fs::exists(numa_file)) << "Failed to create file for " << bus_id;
    }
}


/** @brief Authored CPU bindings exercise the actual production resolver. */
class CPUCallerNUMAOwnership : public Test__Unit_NUMATopologySysfs
{
protected:
    /** @brief Give one authored CPU an exact OS NUMA directory identity.
     * @param cpu Logical CPU ID.
     * @param node Physical, possibly sparse NUMA ID.
     */
    void cpuNode(int cpu, int node)
    {
        fs::create_directories(temp_dir_ / "cpu" / ("cpu" + std::to_string(cpu)) /
                               ("node" + std::to_string(node)));
    }

    /** @brief Replay an authored allowed mask through the production helper.
     * @param allowed Linux comma/range CPU list.
     * @return Actual resolver result for the authored filesystem.
     */
    int resolve(const std::string &allowed)
    {
        const auto status = temp_dir_ / "status";
        { std::ofstream output(status); output << "Name:\tauthored\nCpus_allowed_list:\t" << allowed << '\n'; }
        return NUMATopology::detectLocalNUMANodeFromStatus(status,temp_dir_ / "cpu");
    }
};

/** @test Interior CPU IDs and decimal substrings do not stand in for ranges. */
TEST_F(CPUCallerNUMAOwnership, InteriorAndSubstringIDs)
{
    for(int cpu=28;cpu<=55;++cpu)cpuNode(cpu,1);
    cpuNode(4,0);cpuNode(12,0);
    EXPECT_EQ(resolve("29-31,47,55"),1);
    EXPECT_EQ(resolve("4,12"),0);
    // The old textual test misses29 inside28-55 and can mistake4 for84.
    EXPECT_EQ(resolve("29"),1);
    EXPECT_EQ(resolve("4"),0);
}

/** @test Physical NUMA IDs are independent of count, order and CPU ordinal. */
TEST_F(CPUCallerNUMAOwnership, SparseAndLargeNodeIDs)
{
    cpuNode(129,17);cpuNode(130,17);cpuNode(4095,1024);
    EXPECT_EQ(resolve("129-130"),17);
    EXPECT_EQ(resolve("4095"),1024);
}

/** @test A caller spanning multiple real nodes has no one-node ownership. */
TEST_F(CPUCallerNUMAOwnership, RejectsCrossNodeAndMissingFacts)
{
    cpuNode(0,0);cpuNode(28,1);
    EXPECT_EQ(resolve("0,28"),-1);
    EXPECT_EQ(resolve("0-1"),-1); // CPU1's owner is absent, not silentlyCPU0.
    EXPECT_EQ(resolve(""),-1);
    EXPECT_EQ(NUMATopology::detectLocalNUMANodeFromStatus(temp_dir_/"absent",temp_dir_/"cpu"),-1);
    { std::ofstream output(temp_dir_/"status"); output<<"Name:\tauthored\n"; }
    EXPECT_EQ(NUMATopology::detectLocalNUMANodeFromStatus(temp_dir_/"status",temp_dir_/"cpu"),-1);
}

/** @test An ambiguous or non-directory CPU node cannot manufacture ownership. */
TEST_F(CPUCallerNUMAOwnership, RejectsAmbiguousNodeLinks)
{
    cpuNode(12,23);
    fs::create_directories(temp_dir_/"cpu/cpu12/node5suffix");
    { std::ofstream output(temp_dir_/"cpu/cpu12/node7"); output<<"not a node directory"; }
    EXPECT_EQ(resolve("12"),23);
    cpuNode(12,91);
    EXPECT_EQ(resolve("12"),-1);
}

/** @test Repeated large-to-small caller observations never reuse a stale owner. */
TEST_F(CPUCallerNUMAOwnership, RepeatedBindingChanges)
{
    for(int cpu=28;cpu<=55;++cpu)cpuNode(cpu,9);
    cpuNode(4,0);
    for(int replay=0;replay<3;++replay) {
        EXPECT_EQ(resolve("28-55"),9);
        EXPECT_EQ(resolve("4"),0);
        EXPECT_EQ(resolve("4,29"),-1);
        EXPECT_EQ(resolve("31"),9);
    }
}

/** @test An unobserved NUMAInfo never asserts ownership of physical node0. */
TEST(CPUCallerNUMAOwnershipContract, DefaultIsUnknown)
{
    const NUMAInfo unknown;
    EXPECT_EQ(unknown.local_numa_node,-1);
    EXPECT_FALSE(unknown.detection_succeeded);
    EXPECT_EQ(unknown.detection_method,"unknown");
}


/** @test A malformed token must not leave an apparently valid ownership prefix. */
TEST_F(CPUCallerNUMAOwnership, MalformedCPUListsNeverPublishAnOwner)
{
    cpuNode(42,9);cpuNode(43,9);
    for(const std::string text:{"42x","42,broken","42,,43","42,","42-41",
                               "42-43junk","-42","+42","2147483648","42-2147483648"}) {
        SCOPED_TRACE(text);
        EXPECT_TRUE(NodeTopology::parseCPUListString(text).empty());
        EXPECT_EQ(resolve(text),-1);
    }
    EXPECT_EQ(NodeTopology::parseCPUListString(" 42-43, 47 "),(std::vector<int>{42,43,47}));
    const int largest=std::numeric_limits<int>::max();
    EXPECT_EQ(NodeTopology::parseCPUListString(std::to_string(largest)+"-"+std::to_string(largest)),
              (std::vector<int>{largest}));
    EXPECT_EQ(resolve("42-43"),9);
}
