/**
 * @file Test__NUMATopology.cpp
 * @brief Integration tests for NUMATopology NUMA detection with real GPUs
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "utils/NUMATopology.h"
#include "backends/ComputeBackend.h"
#include <fstream>

using namespace llaminar2;

// =============================================================================
// Local NUMA Node Detection Tests
// =============================================================================

TEST(Test__Integration_NUMATopology, DetectLocalNUMANode_Succeeds)
{
    auto info = NUMATopology::detectLocalNUMANode();

    // Should always succeed on Linux systems
    EXPECT_TRUE(info.detection_succeeded);
    EXPECT_GE(info.local_numa_node, 0);
    EXPECT_GE(info.total_numa_nodes, 1);
    EXPECT_FALSE(info.detection_method.empty());
}

TEST(Test__Integration_NUMATopology, GetNumNUMANodes_ReturnsPositive)
{
    int num_nodes = NUMATopology::getNumNUMANodes();

    EXPECT_GE(num_nodes, 1);
}

TEST(Test__Integration_NUMATopology, DetectionMethod_IsValid)
{
    auto info = NUMATopology::detectLocalNUMANode();

    // Should be one of the known methods
    EXPECT_TRUE(info.detection_method == "hwloc" ||
                info.detection_method == "procfs" ||
                info.detection_method == "sysfs" ||
                info.detection_method == "fallback");
}

// =============================================================================
// GPU NUMA Affinity Helper Tests
// =============================================================================

TEST(Test__Integration_NUMATopology, IsGPULocalToProcess_SameNode)
{
    // GPU and process on same node should return true
    EXPECT_TRUE(NUMATopology::isGPULocalToProcess(0, 0));
    EXPECT_TRUE(NUMATopology::isGPULocalToProcess(1, 1));
}

TEST(Test__Integration_NUMATopology, IsGPULocalToProcess_DifferentNode)
{
    // GPU and process on different nodes should return false
    EXPECT_FALSE(NUMATopology::isGPULocalToProcess(0, 1));
    EXPECT_FALSE(NUMATopology::isGPULocalToProcess(1, 0));
}

TEST(Test__Integration_NUMATopology, IsGPULocalToProcess_UnknownNode)
{
    // If either is unknown (-1), assume compatible (fallback)
    EXPECT_TRUE(NUMATopology::isGPULocalToProcess(-1, 0));
    EXPECT_TRUE(NUMATopology::isGPULocalToProcess(0, -1));
    EXPECT_TRUE(NUMATopology::isGPULocalToProcess(-1, -1));
}

// =============================================================================
// ROCm GPU NUMA Detection Tests (requires ROCm device)
// =============================================================================

#ifdef HAVE_ROCM
class Test__Integration_NUMATopology_ROCm : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize DeviceManager to enumerate devices
        auto &dm = DeviceManager::instance();
        dm.initialize(-1);

        // Check if we have any ROCm devices
        has_rocm_device_ = false;
        const auto &devices = dm.devices();
        for (size_t i = 0; i < devices.size(); ++i)
        {
            if (devices[i].type == ComputeBackendType::GPU_ROCM)
            {
                has_rocm_device_ = true;
                break;
            }
        }
    }

    bool has_rocm_device_ = false;
};

TEST_F(Test__Integration_NUMATopology_ROCm, DetectsROCmGPUNUMANode)
{
    if (!has_rocm_device_)
    {
        GTEST_SKIP() << "No ROCm device available";
    }

    // Test device 0
    auto info = NUMATopology::getROCmGPUNUMANode(0);

    // Should detect NUMA node via sysfs
    EXPECT_TRUE(info.affinity_detected) << "ROCm GPU 0 NUMA detection failed";
    EXPECT_GE(info.numa_node, 0) << "NUMA node should be >= 0, got " << info.numa_node;
    EXPECT_EQ(info.gpu_id, 0);
}

TEST_F(Test__Integration_NUMATopology_ROCm, DetectionMethod_NotFallback)
{
    if (!has_rocm_device_)
    {
        GTEST_SKIP() << "No ROCm device available";
    }

    auto info = NUMATopology::getROCmGPUNUMANode(0);

    // Should NOT be "fallback" - we have working sysfs detection
    EXPECT_NE(info.detection_method, "fallback")
        << "ROCm GPU NUMA detection fell back - sysfs detection not working";
}

TEST_F(Test__Integration_NUMATopology_ROCm, MatchesSysfs)
{
    if (!has_rocm_device_)
    {
        GTEST_SKIP() << "No ROCm device available";
    }

    auto info = NUMATopology::getROCmGPUNUMANode(0);

    // Cross-check with DRM sysfs
    // ROCm GPU 0 is typically renderD128
    std::ifstream numa_file("/sys/class/drm/renderD128/device/numa_node");
    if (numa_file.is_open())
    {
        int sysfs_numa;
        numa_file >> sysfs_numa;

        EXPECT_EQ(info.numa_node, sysfs_numa)
            << "NUMATopology returned " << info.numa_node
            << " but sysfs says " << sysfs_numa;
    }
    else
    {
        // Can't verify via renderD128, just make sure we got a valid result
        EXPECT_GE(info.numa_node, 0);
    }
}

TEST_F(Test__Integration_NUMATopology_ROCm, AllROCmDevicesHaveValidNUMA)
{
    if (!has_rocm_device_)
    {
        GTEST_SKIP() << "No ROCm device available";
    }

    auto &dm = DeviceManager::instance();
    const auto &devices = dm.devices();

    int rocm_device_idx = 0;
    for (size_t i = 0; i < devices.size(); ++i)
    {
        const auto &dev = devices[i];
        if (dev.type == ComputeBackendType::GPU_ROCM)
        {
            auto info = NUMATopology::getROCmGPUNUMANode(rocm_device_idx);

            EXPECT_TRUE(info.affinity_detected)
                << "ROCm device " << rocm_device_idx << " NUMA detection failed";
            EXPECT_GE(info.numa_node, 0)
                << "ROCm device " << rocm_device_idx << " has invalid NUMA node";

            ++rocm_device_idx;
        }
    }

    EXPECT_GT(rocm_device_idx, 0) << "No ROCm devices found in DeviceManager";
}
#endif // HAVE_ROCM

// =============================================================================
// CUDA GPU NUMA Detection Tests (requires CUDA device)
// =============================================================================

#ifdef HAVE_CUDA
class Test__Integration_NUMATopology_CUDA : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize DeviceManager to enumerate devices
        auto &dm = DeviceManager::instance();
        dm.initialize(-1);

        // Check if we have any CUDA devices
        has_cuda_device_ = false;
        const auto &devices = dm.devices();
        for (size_t i = 0; i < devices.size(); ++i)
        {
            if (devices[i].type == ComputeBackendType::GPU_CUDA)
            {
                has_cuda_device_ = true;
                break;
            }
        }
    }

    bool has_cuda_device_ = false;
};

TEST_F(Test__Integration_NUMATopology_CUDA, DetectsCUDAGPUNUMANode)
{
    if (!has_cuda_device_)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    // Test device 0
    auto info = NUMATopology::getCUDAGPUNUMANode(0);

    // Should detect NUMA node
    EXPECT_GE(info.numa_node, 0) << "NUMA node should be >= 0, got " << info.numa_node;
    EXPECT_EQ(info.gpu_id, 0);
}

TEST_F(Test__Integration_NUMATopology_CUDA, AllCUDADevicesHaveValidNUMA)
{
    if (!has_cuda_device_)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    auto &dm = DeviceManager::instance();
    const auto &devices = dm.devices();

    int cuda_device_idx = 0;
    for (size_t i = 0; i < devices.size(); ++i)
    {
        const auto &dev = devices[i];
        if (dev.type == ComputeBackendType::GPU_CUDA)
        {
            auto info = NUMATopology::getCUDAGPUNUMANode(cuda_device_idx);

            EXPECT_GE(info.numa_node, 0)
                << "CUDA device " << cuda_device_idx << " has invalid NUMA node";

            ++cuda_device_idx;
        }
    }

    EXPECT_GT(cuda_device_idx, 0) << "No CUDA devices found in DeviceManager";
}

/**
 * @test Verifies that dual-backend builds (CUDA+ROCm) actually detect NUMA via sysfs
 *
 * This test catches the bug where getCUDAGPUNUMANode() returns a hardcoded
 * NUMA node 0 with detection_method="fallback_multi_gpu" instead of actually
 * reading from sysfs.
 *
 * BUG: In dual-backend builds, the code path at NUMATopology.cpp:358-365
 * unconditionally returns numa_node=0 without calling detectGPUViaSysfs().
 */
TEST_F(Test__Integration_NUMATopology_CUDA, DetectionMethod_NotFallbackMultiGPU)
{
    if (!has_cuda_device_)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    auto &dm = DeviceManager::instance();
    const auto &devices = dm.devices();

    int cuda_device_idx = 0;
    for (size_t i = 0; i < devices.size(); ++i)
    {
        const auto &dev = devices[i];
        if (dev.type == ComputeBackendType::GPU_CUDA)
        {
            auto info = NUMATopology::getCUDAGPUNUMANode(cuda_device_idx);

            // This test MUST FAIL with the current buggy code!
            // The dual-backend path returns "fallback_multi_gpu" instead of
            // actually detecting via sysfs.
            EXPECT_NE(info.detection_method, "fallback_multi_gpu")
                << "CUDA device " << cuda_device_idx
                << " used fallback_multi_gpu instead of actual sysfs detection. "
                << "This indicates the dual-backend NUMA detection bug is present.";

            // If detection worked, method should be nvml, sysfs, or rocm_smi
            EXPECT_TRUE(info.detection_method == "nvml" ||
                        info.detection_method == "sysfs" ||
                        info.detection_method == "rocm_smi")
                << "CUDA device " << cuda_device_idx
                << " detection_method=" << info.detection_method
                << " is not a valid hardware detection method";

            ++cuda_device_idx;
        }
    }
}
#endif // HAVE_CUDA
