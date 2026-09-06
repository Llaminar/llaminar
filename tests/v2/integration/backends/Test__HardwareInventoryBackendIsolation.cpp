/**
 * @file Test__HardwareInventoryBackendIsolation.cpp
 * @brief Real-driver regressions for foreign-backend inventory isolation.
 *
 * Each test runs in a fresh process with one vendor explicitly excluded. The
 * allowed vendor remains enabled so this exercises the production mixed-build
 * condition that exposed accidental foreign primary contexts during parity
 * campaign concurrency. CUDA provides an exact primary-context activity query;
 * ROCm isolation is observed at the Linux KFD device boundary without calling
 * HIP and thereby perturbing the state under test.
 */

#include <gtest/gtest.h>

#include "app/MPIBootstrapPhase.h"
#include "backends/HardwareInventory.h"
#include "utils/DebugEnv.h"

#include <cuda.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <string>
#include <unistd.h>
#include <vector>

using namespace llaminar2;

namespace
{
    /**
     * @brief Snapshot every CUDA device's primary-context activity bit.
     * @return One activity bit per driver-visible CUDA device.
     */
    std::vector<int> cudaPrimaryContextActivity()
    {
        const CUresult init_result = cuInit(0);
        if (init_result != CUDA_SUCCESS)
            return {};

        int count = 0;
        if (cuDeviceGetCount(&count) != CUDA_SUCCESS || count <= 0)
            return {};

        std::vector<int> activity;
        activity.reserve(static_cast<std::size_t>(count));
        for (int ordinal = 0; ordinal < count; ++ordinal)
        {
            CUdevice device = 0;
            EXPECT_EQ(cuDeviceGet(&device, ordinal), CUDA_SUCCESS);
            unsigned int flags = 0;
            int active = 0;
            EXPECT_EQ(
                cuDevicePrimaryCtxGetState(device, &flags, &active),
                CUDA_SUCCESS);
            activity.push_back(active);
        }
        return activity;
    }

    /**
     * @brief Assert that no CUDA primary context is active in this process.
     * @param activity Activity snapshot returned by cudaPrimaryContextActivity().
     */
    void expectEveryCudaPrimaryContextInactive(
        const std::vector<int> &activity)
    {
        ASSERT_FALSE(activity.empty())
            << "CUDA integration hardware is required for this preflight";
        for (std::size_t ordinal = 0; ordinal < activity.size(); ++ordinal)
        {
            EXPECT_EQ(activity[ordinal], 0)
                << "CUDA primary context unexpectedly active on ordinal "
                << ordinal;
        }
    }

    /**
     * @brief Determine whether this process owns an open KFD device handle.
     * @return True when any `/proc/self/fd` entry resolves to `/dev/kfd`.
     */
    bool hasOpenKfdDescriptor()
    {
        DIR *directory = opendir("/proc/self/fd");
        if (directory == nullptr)
        {
            ADD_FAILURE() << "Cannot inspect /proc/self/fd: "
                          << std::strerror(errno);
            return true;
        }

        bool found = false;
        while (const dirent *entry = readdir(directory))
        {
            if (entry->d_name[0] == '.')
                continue;
            const std::string path =
                std::string("/proc/self/fd/") + entry->d_name;
            std::array<char, 4096> target{};
            const ssize_t length =
                readlink(path.c_str(), target.data(), target.size() - 1u);
            if (length <= 0)
                continue;
            target[static_cast<std::size_t>(length)] = '\0';
            if (std::string(target.data()) == "/dev/kfd")
            {
                found = true;
                break;
            }
        }
        closedir(directory);
        return found;
    }
} // namespace

TEST(Test__HardwareInventoryBackendIsolation,
     CudaExcludedRocmDiscoveryLeavesCudaPrimaryContextsInactive)
{
    const auto &startup = debugEnv().backend_startup;
    ASSERT_FALSE(startup.cudaEnabled());
    ASSERT_TRUE(startup.rocmEnabled());

    expectEveryCudaPrimaryContextInactive(cudaPrimaryContextActivity());

    // This is the exact authority used by ClusterInventoryGatherer and
    // MPITopology. ROCm discovery remains real while CUDA must be unreachable.
    const HardwareInventory inventory = HardwareInventory::detect();
    EXPECT_TRUE(inventory.cuda_devices.empty());
    ASSERT_FALSE(inventory.rocm_devices.empty())
        << "ROCm integration hardware is required for this preflight";

    expectEveryCudaPrimaryContextInactive(cudaPrimaryContextActivity());
}

TEST(Test__HardwareInventoryBackendIsolation,
     RocmExcludedCudaDiscoveryNeverOpensKfd)
{
    const auto &startup = debugEnv().backend_startup;
    ASSERT_TRUE(startup.cudaEnabled());
    ASSERT_FALSE(startup.rocmEnabled());
    ASSERT_FALSE(hasOpenKfdDescriptor())
        << "ROCm was initialized before the inventory authority ran";

    // CUDA discovery remains real; a linked ROCm implementation must neither
    // enumerate devices nor open its process-level KFD runtime state.
    const HardwareInventory inventory = HardwareInventory::detect();
    ASSERT_FALSE(inventory.cuda_devices.empty())
        << "CUDA integration hardware is required for this preflight";
    EXPECT_TRUE(inventory.rocm_devices.empty());
    EXPECT_FALSE(hasOpenKfdDescriptor());
}

TEST(Test__HardwareInventoryBackendIsolation,
     CpuNamedDomainBootstrapExcludesBothVendorsBeforeDiscovery)
{
    // Start with both vendors permitted: CLI intent, not a test-only
    // environment exclusion, must close the accelerator startup boundary.
    ASSERT_TRUE(debugEnv().backend_startup.cudaEnabled());
    ASSERT_TRUE(debugEnv().backend_startup.rocmEnabled());
    expectEveryCudaPrimaryContextInactive(cudaPrimaryContextActivity());
    ASSERT_FALSE(hasOpenKfdDescriptor());

    OrchestrationConfig config;
    config.mpi_no_bootstrap = true;
    config.domain_definitions.push_back(DomainDefinition::parse(
        "cpu_node=localhost:0:cpu:0,localhost:1:cpu:0;scope=node_local;backend=mpi"));
    MPIBootstrapPhase bootstrap;
    ASSERT_EQ(bootstrap.execute(config, 0, nullptr).action,
              BootstrapResult::Action::CONTINUE);

    const HardwareInventory inventory = HardwareInventory::detect();
    EXPECT_TRUE(inventory.cuda_devices.empty());
    EXPECT_TRUE(inventory.rocm_devices.empty());
    expectEveryCudaPrimaryContextInactive(cudaPrimaryContextActivity());
    EXPECT_FALSE(hasOpenKfdDescriptor());
}
