/**
 * @file Test__GPUPackedWeightsStubs.cpp
 * @brief Unit tests for Phase 4 GPU packed weight stubs (CUDA + ROCm).
 *
 * Validates that CUDAPackedWeights and ROCmPackedWeights implement the
 * IPackedWeights interface correctly. These are compile-time stubs —
 * no actual GPU memory operations are tested.
 */

#include <gtest/gtest.h>
#include "kernels/cuda/gemm/CUDAPackedWeights.h"
#include "kernels/rocm/gemm/ROCmPackedWeights.h"

using namespace llaminar2;

// ─── CUDAPackedWeights ───────────────────────────────────────────

TEST(Test__GPUPackedWeightsStubs, CUDA_ConstructionAndDimensions)
{
    std::vector<uint8_t> data(1024, 0xAB);
    cuda::CUDAPackedWeights pw(128, 256, data);

    EXPECT_EQ(pw.N(), 128);
    EXPECT_EQ(pw.K(), 256);
    EXPECT_EQ(pw.sizeBytes(), 1024u);
}

TEST(Test__GPUPackedWeightsStubs, CUDA_FormatAndDeviceType)
{
    cuda::CUDAPackedWeights pw(64, 64, std::vector<uint8_t>(512));

    EXPECT_EQ(pw.format(), PackedWeightsFormat::CUDA_INT8);
    EXPECT_EQ(pw.deviceType(), DeviceType::CUDA);
}

TEST(Test__GPUPackedWeightsStubs, CUDA_InterfacePolymorphism)
{
    auto pw = std::make_unique<cuda::CUDAPackedWeights>(32, 64, std::vector<uint8_t>(256));
    IPackedWeights *iface = pw.get();

    EXPECT_EQ(iface->N(), 32);
    EXPECT_EQ(iface->K(), 64);
    EXPECT_EQ(iface->format(), PackedWeightsFormat::CUDA_INT8);
    EXPECT_EQ(iface->deviceType(), DeviceType::CUDA);
}

TEST(Test__GPUPackedWeightsStubs, CUDA_CloneReturnsCopy)
{
    std::vector<uint8_t> data(512, 0xCD);
    cuda::CUDAPackedWeights pw(96, 192, data);

    auto cloned = pw.clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->N(), 96);
    EXPECT_EQ(cloned->K(), 192);
    EXPECT_EQ(cloned->sizeBytes(), 512u);
    EXPECT_EQ(cloned->format(), PackedWeightsFormat::CUDA_INT8);
    EXPECT_EQ(cloned->deviceType(), DeviceType::CUDA);

    // Verify data content via downcast
    auto *cuda_cloned = dynamic_cast<cuda::CUDAPackedWeights *>(cloned.get());
    ASSERT_NE(cuda_cloned, nullptr);
    EXPECT_EQ(cuda_cloned->deviceData(), data);
}

TEST(Test__GPUPackedWeightsStubs, CUDA_ConvertToSameDeviceReturnsClone)
{
    cuda::CUDAPackedWeights pw(64, 128, std::vector<uint8_t>(256, 0xFF));

    auto converted = pw.convertTo(DeviceType::CUDA);
    ASSERT_NE(converted, nullptr);
    EXPECT_EQ(converted->N(), 64);
    EXPECT_EQ(converted->K(), 128);
    EXPECT_EQ(converted->deviceType(), DeviceType::CUDA);
}

TEST(Test__GPUPackedWeightsStubs, CUDA_ConvertToCrossDeviceReturnsNull)
{
    cuda::CUDAPackedWeights pw(64, 128, std::vector<uint8_t>(256));

    EXPECT_EQ(pw.convertTo(DeviceType::CPU), nullptr);
    EXPECT_EQ(pw.convertTo(DeviceType::ROCm), nullptr);
}

TEST(Test__GPUPackedWeightsStubs, CUDA_SerializeReturnsEmpty)
{
    cuda::CUDAPackedWeights pw(64, 128, std::vector<uint8_t>(256));

    auto serialized = pw.serialize();
    EXPECT_TRUE(serialized.empty());
}

TEST(Test__GPUPackedWeightsStubs, CUDA_CompatibilityChecks)
{
    cuda::CUDAPackedWeights pw(64, 128, std::vector<uint8_t>(256));

    EXPECT_TRUE(pw.isCompatibleWith(DeviceType::CUDA));
    EXPECT_FALSE(pw.isCompatibleWith(DeviceType::CPU));
    EXPECT_FALSE(pw.isCompatibleWith(DeviceType::ROCm));

    EXPECT_TRUE(pw.canTransferTo(PackedWeightsFormat::CUDA_INT8));
    EXPECT_FALSE(pw.canTransferTo(PackedWeightsFormat::CPU_NATIVE_VNNI));
    EXPECT_FALSE(pw.canTransferTo(PackedWeightsFormat::ROCM_INT8));
}

// ─── ROCmPackedWeights ───────────────────────────────────────────

TEST(Test__GPUPackedWeightsStubs, ROCm_ConstructionAndDimensions)
{
    std::vector<uint8_t> data(2048, 0xBE);
    rocm::ROCmPackedWeights pw(256, 512, data);

    EXPECT_EQ(pw.N(), 256);
    EXPECT_EQ(pw.K(), 512);
    EXPECT_EQ(pw.sizeBytes(), 2048u);
}

TEST(Test__GPUPackedWeightsStubs, ROCm_FormatAndDeviceType)
{
    rocm::ROCmPackedWeights pw(64, 64, std::vector<uint8_t>(512));

    EXPECT_EQ(pw.format(), PackedWeightsFormat::ROCM_INT8);
    EXPECT_EQ(pw.deviceType(), DeviceType::ROCm);
}

TEST(Test__GPUPackedWeightsStubs, ROCm_InterfacePolymorphism)
{
    auto pw = std::make_unique<rocm::ROCmPackedWeights>(32, 64, std::vector<uint8_t>(256));
    IPackedWeights *iface = pw.get();

    EXPECT_EQ(iface->N(), 32);
    EXPECT_EQ(iface->K(), 64);
    EXPECT_EQ(iface->format(), PackedWeightsFormat::ROCM_INT8);
    EXPECT_EQ(iface->deviceType(), DeviceType::ROCm);
}

TEST(Test__GPUPackedWeightsStubs, ROCm_CloneReturnsCopy)
{
    std::vector<uint8_t> data(768, 0xEF);
    rocm::ROCmPackedWeights pw(48, 96, data);

    auto cloned = pw.clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->N(), 48);
    EXPECT_EQ(cloned->K(), 96);
    EXPECT_EQ(cloned->sizeBytes(), 768u);
    EXPECT_EQ(cloned->format(), PackedWeightsFormat::ROCM_INT8);
    EXPECT_EQ(cloned->deviceType(), DeviceType::ROCm);

    auto *rocm_cloned = dynamic_cast<rocm::ROCmPackedWeights *>(cloned.get());
    ASSERT_NE(rocm_cloned, nullptr);
    EXPECT_EQ(rocm_cloned->deviceData(), data);
}

TEST(Test__GPUPackedWeightsStubs, ROCm_ConvertToSameDeviceReturnsClone)
{
    rocm::ROCmPackedWeights pw(64, 128, std::vector<uint8_t>(256, 0xFF));

    auto converted = pw.convertTo(DeviceType::ROCm);
    ASSERT_NE(converted, nullptr);
    EXPECT_EQ(converted->N(), 64);
    EXPECT_EQ(converted->K(), 128);
    EXPECT_EQ(converted->deviceType(), DeviceType::ROCm);
}

TEST(Test__GPUPackedWeightsStubs, ROCm_ConvertToCrossDeviceReturnsNull)
{
    rocm::ROCmPackedWeights pw(64, 128, std::vector<uint8_t>(256));

    EXPECT_EQ(pw.convertTo(DeviceType::CPU), nullptr);
    EXPECT_EQ(pw.convertTo(DeviceType::CUDA), nullptr);
}

TEST(Test__GPUPackedWeightsStubs, ROCm_SerializeReturnsEmpty)
{
    rocm::ROCmPackedWeights pw(64, 128, std::vector<uint8_t>(256));

    auto serialized = pw.serialize();
    EXPECT_TRUE(serialized.empty());
}

TEST(Test__GPUPackedWeightsStubs, ROCm_CompatibilityChecks)
{
    rocm::ROCmPackedWeights pw(64, 128, std::vector<uint8_t>(256));

    EXPECT_TRUE(pw.isCompatibleWith(DeviceType::ROCm));
    EXPECT_FALSE(pw.isCompatibleWith(DeviceType::CPU));
    EXPECT_FALSE(pw.isCompatibleWith(DeviceType::CUDA));

    EXPECT_TRUE(pw.canTransferTo(PackedWeightsFormat::ROCM_INT8));
    EXPECT_FALSE(pw.canTransferTo(PackedWeightsFormat::CPU_NATIVE_VNNI));
    EXPECT_FALSE(pw.canTransferTo(PackedWeightsFormat::CUDA_INT8));
}

// ─── Empty data edge case ────────────────────────────────────────

TEST(Test__GPUPackedWeightsStubs, CUDA_EmptyData)
{
    cuda::CUDAPackedWeights pw(0, 0, {});
    EXPECT_EQ(pw.sizeBytes(), 0u);
    EXPECT_EQ(pw.N(), 0);
    EXPECT_EQ(pw.K(), 0);

    auto cloned = pw.clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->sizeBytes(), 0u);
}

TEST(Test__GPUPackedWeightsStubs, ROCm_EmptyData)
{
    rocm::ROCmPackedWeights pw(0, 0, {});
    EXPECT_EQ(pw.sizeBytes(), 0u);
    EXPECT_EQ(pw.N(), 0);
    EXPECT_EQ(pw.K(), 0);

    auto cloned = pw.clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->sizeBytes(), 0u);
}
