/**
 * @file Test__DeviceRegistry.cpp
 * @brief Unit tests for DeviceRegistry
 *
 * Tests:
 * - Singleton instance access
 * - CPU device discovery
 * - CUDA device discovery (if available)
 * - ROCm device discovery (if available)
 * - Device validation
 * - Memory queries
 * - Backend access
 * - Topology queries
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "backends/DeviceRegistry.h"
#include "backends/IBackend.h"

using namespace llaminar2;

// ============================================================================
// Singleton Tests
// ============================================================================

TEST(Test__DeviceRegistry, SingletonInstance)
{
    auto &registry1 = DeviceRegistry::instance();
    auto &registry2 = DeviceRegistry::instance();

    // Same instance
    EXPECT_EQ(&registry1, &registry2);
}

// ============================================================================
// Discovery Tests
// ============================================================================

TEST(Test__DeviceRegistry, DiscoveryCpuDevices)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    // Should have discovered at least one CPU device
    auto cpus = registry.devicesByType(DeviceType::CPU);
    EXPECT_GE(cpus.size(), 1) << "Should discover at least one CPU device";

    // First CPU should be NUMA node 0
    if (!cpus.empty())
    {
        EXPECT_EQ(cpus[0].device_type, DeviceType::CPU);
        EXPECT_TRUE(cpus[0].isLocal());
    }
}

TEST(Test__DeviceRegistry, DiscoveryIsDiscoveredFlag)
{
    auto &registry = DeviceRegistry::instance();

    // After discover(), flag should be set
    registry.discover();
    EXPECT_TRUE(registry.isDiscovered());
}

TEST(Test__DeviceRegistry, DiscoveryAllDevices)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    auto all = registry.allDevices();
    EXPECT_GE(all.size(), 1) << "Should have at least one device (CPU)";

    // All devices should be local
    for (const auto &addr : all)
    {
        EXPECT_TRUE(addr.isLocal());
    }
}

TEST(Test__DeviceRegistry, DiscoveryTotalCount)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    auto all = registry.allDevices();
    EXPECT_EQ(all.size(), registry.totalDeviceCount());
}

TEST(Test__DeviceRegistry, DiscoveryRefresh)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    size_t count1 = registry.totalDeviceCount();

    // Refresh should work without error
    registry.refresh();

    size_t count2 = registry.totalDeviceCount();

    // Count should be the same (topology doesn't change)
    EXPECT_EQ(count1, count2);
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST(Test__DeviceRegistry, ValidatesCpuDevice)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    // CPU:0 should always be valid
    GlobalDeviceAddress cpu0 = GlobalDeviceAddress::cpu(0);
    EXPECT_TRUE(registry.isValid(cpu0));
}

TEST(Test__DeviceRegistry, ValidatesInvalidDevice)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    // CUDA:999 should not be valid (unless you have 1000 GPUs!)
    GlobalDeviceAddress invalid = GlobalDeviceAddress::cuda(999);
    EXPECT_FALSE(registry.isValid(invalid));
}

TEST(Test__DeviceRegistry, ValidatesRemoteDevice)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    // Remote device can't be validated locally
    GlobalDeviceAddress remote;
    remote.hostname = "remote-host";
    remote.numa_node = 0;
    remote.device_type = DeviceType::CPU;
    remote.device_ordinal = 0;

    EXPECT_FALSE(registry.isValid(remote));
}

TEST(Test__DeviceRegistry, DeviceAvailability)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    auto cpus = registry.devicesByType(DeviceType::CPU);
    if (!cpus.empty())
    {
        // CPU should be available
        EXPECT_TRUE(registry.isAvailable(cpus[0]));
    }
}

// ============================================================================
// Memory Query Tests
// ============================================================================

TEST(Test__DeviceRegistry, MemoryCapacityCpu)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    auto cpus = registry.devicesByType(DeviceType::CPU);
    ASSERT_FALSE(cpus.empty());

    size_t capacity = registry.memoryCapacity(cpus[0]);
    EXPECT_GT(capacity, 0) << "CPU memory capacity should be positive";

    // Sanity check: should be at least 100 MB
    EXPECT_GT(capacity, 100 * 1024 * 1024);
}

TEST(Test__DeviceRegistry, MemoryCapacityInvalidDevice)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    GlobalDeviceAddress invalid = GlobalDeviceAddress::cuda(999);
    size_t capacity = registry.memoryCapacity(invalid);
    EXPECT_EQ(capacity, 0) << "Invalid device should return 0 capacity";
}

TEST(Test__DeviceRegistry, DeviceNameCpu)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    auto cpus = registry.devicesByType(DeviceType::CPU);
    ASSERT_FALSE(cpus.empty());

    std::string name = registry.deviceName(cpus[0]);
    EXPECT_FALSE(name.empty()) << "CPU device should have a name";
    EXPECT_NE(name.find("CPU"), std::string::npos) << "Name should contain 'CPU'";
}

// ============================================================================
// Backend Access Tests
// ============================================================================

TEST(Test__DeviceRegistry, BackendForCpu)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    GlobalDeviceAddress cpu0 = GlobalDeviceAddress::cpu(0);
    IBackend *backend = registry.backendFor(cpu0);

    EXPECT_NE(backend, nullptr) << "CPU backend should be available";
}

TEST(Test__DeviceRegistry, BackendForInvalidType)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    // Vulkan backend not implemented
    GlobalDeviceAddress vulkan;
    vulkan.hostname = "localhost";
    vulkan.numa_node = 0;
    vulkan.device_type = DeviceType::Vulkan;
    vulkan.device_ordinal = 0;

    IBackend *backend = registry.backendFor(vulkan);
    EXPECT_EQ(backend, nullptr) << "Vulkan backend should not be available";
}

// ============================================================================
// Topology Tests
// ============================================================================

TEST(Test__DeviceRegistry, NumaAffinityCpu)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    auto cpus = registry.devicesByType(DeviceType::CPU);
    ASSERT_FALSE(cpus.empty());

    int numa = registry.numaAffinity(cpus[0]);
    EXPECT_GE(numa, 0) << "CPU NUMA affinity should be valid";
    EXPECT_EQ(numa, cpus[0].numa_node) << "NUMA affinity should match address";
}

TEST(Test__DeviceRegistry, PcieBusIdCpu)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    GlobalDeviceAddress cpu0 = GlobalDeviceAddress::cpu(0);
    std::string pcie = registry.pcieBusId(cpu0);

    // CPU doesn't have PCIe bus ID
    EXPECT_TRUE(pcie.empty()) << "CPU should not have PCIe bus ID";
}

TEST(Test__DeviceRegistry, P2PCpuDevices)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    GlobalDeviceAddress cpu0 = GlobalDeviceAddress::cpu(0);
    GlobalDeviceAddress cpu1 = GlobalDeviceAddress::cpu(1);

    // CPU devices don't do P2P
    EXPECT_FALSE(registry.canP2P(cpu0, cpu0));
    EXPECT_FALSE(registry.canP2P(cpu0, cpu1));
}

// ============================================================================
// Filter Tests
// ============================================================================

TEST(Test__DeviceRegistry, DevicesByNuma)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    auto numa0 = registry.devicesByNuma(0);
    EXPECT_GE(numa0.size(), 1) << "Should have at least one device on NUMA 0";

    for (const auto &addr : numa0)
    {
        EXPECT_EQ(addr.numa_node, 0);
    }
}

TEST(Test__DeviceRegistry, DevicesByType)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    auto cpus = registry.devicesByType(DeviceType::CPU);
    for (const auto &addr : cpus)
    {
        EXPECT_EQ(addr.device_type, DeviceType::CPU);
    }

    // CUDA devices (may be empty)
    auto cudas = registry.devicesByType(DeviceType::CUDA);
    for (const auto &addr : cudas)
    {
        EXPECT_EQ(addr.device_type, DeviceType::CUDA);
    }

    // ROCm devices (may be empty)
    auto rocms = registry.devicesByType(DeviceType::ROCm);
    for (const auto &addr : rocms)
    {
        EXPECT_EQ(addr.device_type, DeviceType::ROCm);
    }
}

TEST(Test__DeviceRegistry, DeviceCountByType)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    auto cpus = registry.devicesByType(DeviceType::CPU);
    EXPECT_EQ(cpus.size(), registry.deviceCount(DeviceType::CPU));

    auto cudas = registry.devicesByType(DeviceType::CUDA);
    EXPECT_EQ(cudas.size(), registry.deviceCount(DeviceType::CUDA));

    auto rocms = registry.devicesByType(DeviceType::ROCm);
    EXPECT_EQ(rocms.size(), registry.deviceCount(DeviceType::ROCm));
}

// ============================================================================
// Default Device Tests
// ============================================================================

TEST(Test__DeviceRegistry, DefaultDeviceCpu)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    auto default_cpu = registry.defaultDevice(DeviceType::CPU);
    ASSERT_TRUE(default_cpu.has_value());

    EXPECT_EQ(default_cpu->device_type, DeviceType::CPU);
    EXPECT_EQ(default_cpu->device_ordinal, 0);
}

TEST(Test__DeviceRegistry, DefaultDeviceNonexistent)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    // Vulkan devices are not discovered
    auto default_vulkan = registry.defaultDevice(DeviceType::Vulkan);
    EXPECT_FALSE(default_vulkan.has_value());
}

// ============================================================================
// GPU Tests (conditional)
// ============================================================================

#ifdef HAVE_CUDA
TEST(Test__DeviceRegistry, DiscoversCudaDevices)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    auto cudas = registry.devicesByType(DeviceType::CUDA);

    // May or may not have CUDA devices depending on hardware
    // Just verify the API works
    for (const auto &addr : cudas)
    {
        EXPECT_EQ(addr.device_type, DeviceType::CUDA);
        EXPECT_TRUE(addr.isGPU());
        EXPECT_FALSE(addr.isCPU());

        // Should have positive memory
        size_t mem = registry.memoryCapacity(addr);
        EXPECT_GT(mem, 0);
    }
}

TEST(Test__DeviceRegistry, CudaBackendAccess)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    auto cudas = registry.devicesByType(DeviceType::CUDA);
    if (!cudas.empty())
    {
        IBackend *backend = registry.backendFor(cudas[0]);
        EXPECT_NE(backend, nullptr);
    }
}
#endif

#ifdef HAVE_ROCM
TEST(Test__DeviceRegistry, DiscoversRocmDevices)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    auto rocms = registry.devicesByType(DeviceType::ROCm);

    // May or may not have ROCm devices depending on hardware
    for (const auto &addr : rocms)
    {
        EXPECT_EQ(addr.device_type, DeviceType::ROCm);
        EXPECT_TRUE(addr.isGPU());

        // Should have positive memory
        size_t mem = registry.memoryCapacity(addr);
        EXPECT_GT(mem, 0);
    }
}

TEST(Test__DeviceRegistry, RocmBackendAccess)
{
    auto &registry = DeviceRegistry::instance();
    registry.discover();

    auto rocms = registry.devicesByType(DeviceType::ROCm);
    if (!rocms.empty())
    {
        IBackend *backend = registry.backendFor(rocms[0]);
        EXPECT_NE(backend, nullptr);
    }
}
#endif
