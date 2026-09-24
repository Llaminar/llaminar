/**
 * @file Test__HostReleaseAfterROCmUpload.cpp
 * @brief Exhaustive ROCm host-registration ownership regressions.
 *
 * Tensor uploads register caller-owned host pages in the exact HIP context
 * that performs DMA. Multi-device graph preparation may leave another device
 * current when model weights are released or destroyed. These tests prove that
 * every loader-visible codebook and ordinary tensor storage family retires its
 * registration on the owning device, restores the caller's device, and only
 * then relinquishes the backing allocation. TensorSlice receives a dedicated
 * regression because it delegates storage and registration to an inner tensor.
 */

#include <gtest/gtest.h>

#include "backends/DeviceId.h"
#include "tensors/TensorSlice.h"
#include "tensors/Tensors.h"

#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/ScopedGPUStream.h"

#include <hip/hip_runtime.h>

#include <functional>
#include <memory>
#include <utility>
#include <vector>

using namespace llaminar2;

namespace
{
    using TensorCreator = std::function<std::unique_ptr<TensorBase>()>;

    /** @return Number of HIP devices visible to this process, or zero on error. */
    int rocmDeviceCount()
    {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess ? count : 0;
    }

    /**
     * @brief Build every host-storage family accepted by model preparation.
     *
     * The quantized portion is sourced from the canonical codebook registry so
     * additions cannot silently miss this lifecycle proof. Floating, integer,
     * Q16, and TurboQuant storage are appended because they can also traverse
     * TransferEngine even though they are not all GGUF weight codebooks.
     */
    std::vector<std::pair<const char *, TensorCreator>> allStorageCases()
    {
        std::vector<std::pair<const char *, TensorCreator>> cases;
        for (const auto &format : llaminar2::test::quantizedVerifierFormats())
        {
            cases.emplace_back(
                format.label,
                [creator = format.create]()
                {
                    return creator({1u, 256u}, 0x74c3u);
                });
        }

        cases.emplace_back("FP32", []()
                           { return llaminar2::test::TestTensorFactory::createFP32Random({1u, 256u}); });
        cases.emplace_back("FP16", []()
                           { return llaminar2::test::TestTensorFactory::createFP16Random({1u, 256u}); });
        cases.emplace_back("BF16", []()
                           { return llaminar2::test::TestTensorFactory::createBF16Random({1u, 256u}); });
        cases.emplace_back("INT8", []()
                           { return std::make_unique<INT8Tensor>(std::vector<size_t>{1u, 256u}); });
        cases.emplace_back("INT32", []()
                           { return std::make_unique<INT32Tensor>(std::vector<size_t>{1u, 256u}); });
        cases.emplace_back("Q16_1", []()
                           { return llaminar2::test::TestTensorFactory::createQ16_1Random({1u, 256u}); });
        cases.emplace_back("TQ4", []()
                           { return std::make_unique<TQ4Tensor>(std::vector<size_t>{1u, 256u}, 64); });
        cases.emplace_back("TQ8", []()
                           { return std::make_unique<TQ8Tensor>(std::vector<size_t>{1u, 256u}, 64); });
        return cases;
    }

    /**
     * @brief Prove an address is no longer registered in its owning HIP context.
     *
     * `hipHostRegisterDefault` publishes only into the selected device's page
     * tables, so the query deliberately binds the registration owner. The
     * foreign device is restored afterward to preserve the adversarial caller
     * context for the next lifecycle transition.
     */
    void expectROCmAddressUnregistered(
        const void *address,
        const char *format_label,
        int owner_device,
        int foreign_device)
    {
        ASSERT_EQ(hipSetDevice(owner_device), hipSuccess);
        hipPointerAttribute_t attributes{};
        const hipError_t status =
            hipPointerGetAttributes(&attributes, address);
        ASSERT_EQ(status, hipSuccess)
            << format_label << " pointer query failed: "
            << hipGetErrorString(status);
        EXPECT_NE(attributes.type, hipMemoryTypeHost)
            << format_label << " left address " << address
            << " registered as HIP host memory on ROCm:" << owner_device;
        ASSERT_EQ(hipSetDevice(foreign_device), hipSuccess);
    }

    /** @brief Assert that host retirement restored the adversarial HIP device. */
    void expectCurrentROCmDevice(int expected_device, const char *format_label)
    {
        int current_device = -1;
        ASSERT_EQ(hipGetDevice(&current_device), hipSuccess);
        EXPECT_EQ(current_device, expected_device)
            << format_label << " changed the caller's HIP device context";
    }
} // namespace

/**
 * @brief Destroy every storage family while another HIP device is current.
 */
TEST(Test__HostReleaseAfterROCmUpload,
     EveryFormatDestructorRetiresOwningDeviceRegistration)
{
    if (rocmDeviceCount() < 2)
        GTEST_SKIP() << "Cross-device registration retirement requires two ROCm devices";

    constexpr int owner_device = 0;
    constexpr int foreign_device = 1;
    const DeviceId owner = DeviceId::rocm(owner_device);
    llaminar2::test::ScopedGPUStream upload_stream(owner);

    for (const auto &[label, create] : allStorageCases())
    {
        SCOPED_TRACE(label);
        std::unique_ptr<TensorBase> tensor = create();
        ASSERT_NE(tensor, nullptr);
        const void *const registered_address = tensor->raw_data();
        ASSERT_NE(registered_address, nullptr);
        ASSERT_TRUE(tensor->ensureOnDevice(owner, upload_stream.get()));

        ASSERT_EQ(hipSetDevice(foreign_device), hipSuccess);
        tensor.reset();
        expectCurrentROCmDevice(foreign_device, label);
        expectROCmAddressUnregistered(
            registered_address, label, owner_device, foreign_device);
    }
}

/**
 * @brief Explicitly release every storage family under a foreign HIP context.
 */
TEST(Test__HostReleaseAfterROCmUpload,
     EveryFormatReleaseRetiresOwningDeviceRegistration)
{
    if (rocmDeviceCount() < 2)
        GTEST_SKIP() << "Cross-device registration retirement requires two ROCm devices";

    constexpr int owner_device = 0;
    constexpr int foreign_device = 1;
    const DeviceId owner = DeviceId::rocm(owner_device);
    llaminar2::test::ScopedGPUStream upload_stream(owner);

    for (const auto &[label, create] : allStorageCases())
    {
        SCOPED_TRACE(label);
        std::unique_ptr<TensorBase> tensor = create();
        ASSERT_NE(tensor, nullptr);
        const void *const registered_address = tensor->raw_data();
        ASSERT_NE(registered_address, nullptr);
        ASSERT_TRUE(tensor->ensureOnDevice(owner, upload_stream.get()));

        ASSERT_EQ(hipSetDevice(foreign_device), hipSuccess);
        tensor->release_host_weight_data();
        EXPECT_TRUE(tensor->is_raw_data_released());
        expectCurrentROCmDevice(foreign_device, label);
        expectROCmAddressUnregistered(
            registered_address, label, owner_device, foreign_device);
    }
}

/**
 * @brief Prove TensorSlice retires its inner registration on ROCm.
 */
TEST(Test__HostReleaseAfterROCmUpload,
     TensorSliceReleaseRetiresInnerRegistrationOnOwningDevice)
{
    if (rocmDeviceCount() < 2)
        GTEST_SKIP() << "Cross-device registration retirement requires two ROCm devices";

    constexpr int owner_device = 0;
    constexpr int foreign_device = 1;
    const DeviceId owner = DeviceId::rocm(owner_device);
    auto inner = llaminar2::test::TestTensorFactory::createFP32Random(
        {8u, 3072u});
    const void *const registered_address = inner->raw_data();
    const SliceMetadata metadata = SliceMetadata::forRowParallel(
        8u, 3072u, 1, 2, true);
    std::unique_ptr<TensorBase> inner_storage = std::move(inner);
    TensorSlice slice(std::move(inner_storage), metadata);
    llaminar2::test::ScopedGPUStream upload_stream(owner);
    ASSERT_TRUE(slice.ensureOnDevice(owner, upload_stream.get()));

    ASSERT_EQ(hipSetDevice(foreign_device), hipSuccess);
    slice.release_host_weight_data();
    EXPECT_TRUE(slice.is_raw_data_released());
    expectCurrentROCmDevice(foreign_device, "TensorSlice<FP32>");
    expectROCmAddressUnregistered(
        registered_address,
        "TensorSlice<FP32>",
        owner_device,
        foreign_device);
}
