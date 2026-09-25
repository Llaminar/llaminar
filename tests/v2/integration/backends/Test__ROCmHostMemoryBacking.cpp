/**
 * @file Test__ROCmHostMemoryBacking.cpp
 * @brief Prove native ROCm transfer slabs use driver-owned, coherent backing.
 *
 * The regression checks the actual Linux mapping, not a startup log or timing
 * threshold. Anonymous USERPTR backing can enter unbounded huge-page compaction
 * when a large model fragments RAM. Native KFD/GTT backing must also retain the
 * exact-stream, byte-correct upload/download contract of TransferEngine.
 * Caller-owned registrations must use explicit pinned buffer objects as well,
 * rather than silently retaining demand-paged HMM ranges after unregistration.
 */

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "transfer/TransferEngine.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <dlfcn.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @return Linux VMA containing the byte, or an empty string if unmapped. */
    std::string mappingFor(const void *pointer)
    {
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        std::ifstream maps("/proc/self/maps");
        std::string line;
        while (std::getline(maps, line))
        {
            unsigned long begin = 0, end = 0;
            if (std::sscanf(line.c_str(), "%lx-%lx", &begin, &end) == 2 &&
                address >= begin && address < end)
                return line;
        }
        return {};
    }
}

TEST(ROCmHostMemoryBacking, RegistrationUsesExplicitOwnershipNotHmmRanges)
{
    // Inspect the live runtime, not just getenv(): the environment could have
    // been changed after HSA initialized. Backend admission initializes HIP.
    ASSERT_NE(getBackendFor(DeviceId::rocm(0)), nullptr);
    const auto query = reinterpret_cast<decltype(&hsa_system_get_info)>(
        dlsym(RTLD_DEFAULT, "hsa_system_get_info"));
    ASSERT_NE(query, nullptr);
    bool svm_enabled = true;
    ASSERT_EQ(query(static_cast<hsa_system_info_t>(HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED),
        &svm_enabled), HSA_STATUS_SUCCESS);
    EXPECT_FALSE(svm_enabled)
        << "Explicit host pin/unpin must not enter deferred HMM range ownership";
}

TEST(ROCmHostMemoryBacking, NativeSlabUsesDriverPagesAndTransfersExactBytes)
{
    const auto device = DeviceId::rocm(0);
    auto *backend = getBackendFor(device);
    ASSERT_NE(backend, nullptr);
    ASSERT_GT(backend->deviceCount(), 0);
    TransferEngine transfers;
    // Cross the ROCr huge-page hint threshold and retain independent odd-tail
    // slices. The test does not need a large model or fragmented host memory.
    constexpr std::size_t bytes = 4u * 1024u * 1024u + 13u;
    const auto slices = transfers.allocateMappedHostTransferSlices(bytes, 2u, device);
    ASSERT_EQ(slices.size(), 2u);
    for (const auto &slice : slices)
    {
        const auto mapping = mappingFor(slice->mutableHostData());
        ASSERT_NE(mapping.find("/dev/dri/renderD"), std::string::npos) << mapping;
        const auto end_mapping = mappingFor(
            static_cast<unsigned char *>(slice->mutableHostData()) + bytes - 1u);
        ASSERT_NE(end_mapping.find("/dev/dri/renderD"), std::string::npos) << end_mapping;
    }
    const auto staging = transfers.allocatePersistentTransferStagingSlices(bytes, 1u, device);
    ASSERT_EQ(staging.size(), 1u);
    // Both public host-storage forms share the native backing policy. A fix
    // limited to mapped slabs would leave ordinary staging vulnerable too.
    const auto staging_mapping = mappingFor(staging.front().mutablePinnedData());
    ASSERT_NE(staging_mapping.find("/dev/dri/renderD"), std::string::npos) << staging_mapping;

    const auto lanes = transfers.allocatePersistentTransferExecutionLanes(
        1u, device, "native_host_backing_regression");
    auto storage = transfers.allocateDeviceTransferBuffer(bytes, device);
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    std::vector<unsigned char> expected(bytes);
    for (std::size_t index = 0; index < bytes; ++index)
        expected[index] = static_cast<unsigned char>((index * 43u + (index >> 3u)) & 255u);
    std::memcpy(slices[0]->mutableHostData(), expected.data(), bytes);
    std::memset(slices[1]->mutableHostData(), 0, bytes);
    context.submitAndWait([&] {
        const auto stream = lanes.front().stream();
        void *terminal = context.createEvent();
        ASSERT_NE(terminal, nullptr);
        transfers.enqueueMappedHostToPersistentDeviceRegion(*slices[0], 0u,
            storage->mutableDeviceData(), bytes, 0u, bytes, device, stream);
        transfers.enqueuePersistentDeviceRegionToMappedHost(storage->deviceData(),
            bytes, 0u, *slices[1], 0u, bytes, device, stream);
        const bool recorded = context.recordEventChecked(terminal, stream);
        const bool completed = recorded && context.synchronizeEventChecked(terminal);
        context.destroyEvent(terminal);
        ASSERT_TRUE(completed);
    });
    EXPECT_EQ(std::memcmp(slices[1]->mutableHostData(), expected.data(), bytes), 0);
}

/**
 * @brief Shared same-family route pages use native portable ownership.
 *
 * ExpertOverlay and node-local tensor-parallel route exchanges expose one
 * mapped control/payload region to several ROCm devices.  Keep this focused
 * preflight proof separate from the single-device slab test: anonymous mmap
 * plus late hipHostRegisterPortable is the path that can overflow Vega IH2.
 */
TEST(ROCmHostMemoryBacking, MultiDeviceRegionUsesPortableNativeBacking)
{
    const auto *backend = getBackendFor(DeviceId::rocm(0));
    ASSERT_NE(backend, nullptr);
    if (backend->deviceCount() < 2)
        GTEST_SKIP() << "requires two ROCm devices for the shared mapped-region contract";

    const std::array devices{DeviceId::rocm(0), DeviceId::rocm(1)};
    TransferEngine transfers;
    constexpr std::size_t bytes = 4u * 1024u * 1024u + 13u;
    auto region = transfers.allocateMappedHostRegion(bytes, devices);

    ASSERT_NE(region, nullptr);
    ASSERT_TRUE(region->isBound());
    const auto mapping = mappingFor(region->mutableHostData());
    ASSERT_NE(mapping.find("/dev/dri/renderD"), std::string::npos) << mapping;
    for (const auto device : devices)
        ASSERT_NE(region->deviceAlias(device), nullptr);
}
