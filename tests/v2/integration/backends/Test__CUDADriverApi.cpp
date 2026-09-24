/**
 * @file Test__CUDADriverApi.cpp
 * @brief Full-backend startup on a driver-free CPU host, and native CUDA binding.
 *
 * The driver-free case runs with a test-only ELF audit module, never a stub or
 * CPU-only replacement build. The native case prepares the actual CUDA backend
 * and proves one immutable table across threads. Existing graph/stream tests
 * exercise these same function pointers during captured inference protocols.
 */
#include <gtest/gtest.h>
#include "backends/HardwareInventory.h"
#include "backends/cuda/CUDABackend.h"
#include "backends/cuda/CUDADriverApi.h"

#include <array>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>

using namespace llaminar2;

TEST(Test__CUDADriverApi, FullBackendCPUHostWithoutNvidiaDriver)
{
    // This is production CPU inventory discovery, not a mock of CUDA startup.
    auto &manager = DeviceManager::instance();
    manager.initialize(-1, false);
    const auto *inventory = manager.hardware();
    ASSERT_NE(inventory, nullptr);
    EXPECT_GT(inventory->cpuDevice().total_memory_bytes, 0u);
    EXPECT_EQ(inventory->cuda_device_count(), 0);
    EXPECT_EQ(inventory->rocm_device_count(), 0);

    // An explicit CUDA operation must not turn absence into a successful empty
    // table. Its diagnostic must identify the deployment dependency exactly.
    try
    {
        (void)CUDADriverApi::instance();
        FAIL() << "Test isolation failed: NVIDIA driver should be unavailable";
    }
    catch (const std::runtime_error &error)
    {
        EXPECT_NE(std::string(error.what()).find(
            "CUDA execution requires host NVIDIA driver libcuda.so.1"), std::string::npos);
    }
    std::ifstream maps("/proc/self/maps");
    ASSERT_TRUE(maps.good());
    const std::string loaded{std::istreambuf_iterator<char>(maps), {}};
    EXPECT_EQ(loaded.find("/libcuda.so"), std::string::npos);
}

TEST(Test__CUDADriverApi, NativePreparationPublishesOneCompleteTable)
{
    CUDABackend backend;
    const auto &api = CUDADriverApi::instance();
    ASSERT_EQ(api.init(0), CUDA_SUCCESS);
    int count = 0;
    ASSERT_EQ(api.deviceGetCount(&count), CUDA_SUCCESS);
    ASSERT_GT(count, 0) << "CUDA integration hardware is required";

#define LLAMINAR_CUDA_DRIVER_ENTRY(member, symbol) EXPECT_NE(api.member, nullptr);
#include "backends/cuda/CUDADriverFunctions.def"
#undef LLAMINAR_CUDA_DRIVER_ENTRY

    std::array<const CUDADriverApi *, 8> observations{};
    std::array<std::thread, 8> readers;
    for (std::size_t i = 0; i < readers.size(); ++i)
        readers[i] = std::thread([&, i] { observations[i] = &CUDADriverApi::instance(); });
    for (auto &reader : readers) reader.join();
    for (const auto *observed : observations) EXPECT_EQ(observed, &api);
}
