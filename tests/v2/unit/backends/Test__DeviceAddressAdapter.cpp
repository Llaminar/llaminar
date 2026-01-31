/**
 * @file Test__DeviceAddressAdapter.cpp
 * @brief Unit tests for DeviceAddressAdapter
 *
 * Tests:
 * - Conversion from legacy types to GlobalDeviceAddress
 * - Conversion from GlobalDeviceAddress to legacy types
 * - Type predicates
 * - Validation
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "backends/DeviceAddressAdapter.h"
#include "backends/GlobalDeviceAddress.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

// ============================================================================
// From Legacy to GlobalDeviceAddress Tests
// ============================================================================

TEST(Test__DeviceAddressAdapter, FromTypeAndOrdinal_Cuda)
{
    auto addr = DeviceAddressAdapter::fromTypeAndOrdinal(DeviceType::CUDA, 0);

    EXPECT_EQ(addr.hostname, "localhost");
    EXPECT_EQ(addr.numa_node, 0);
    EXPECT_EQ(addr.device_type, DeviceType::CUDA);
    EXPECT_EQ(addr.device_ordinal, 0);
}

TEST(Test__DeviceAddressAdapter, FromTypeAndOrdinal_CudaWithNuma)
{
    auto addr = DeviceAddressAdapter::fromTypeAndOrdinal(DeviceType::CUDA, 1, 2);

    EXPECT_EQ(addr.numa_node, 2);
    EXPECT_EQ(addr.device_ordinal, 1);
}

TEST(Test__DeviceAddressAdapter, FromTypeAndOrdinal_Rocm)
{
    auto addr = DeviceAddressAdapter::fromTypeAndOrdinal(DeviceType::ROCm, 2);

    EXPECT_EQ(addr.device_type, DeviceType::ROCm);
    EXPECT_EQ(addr.device_ordinal, 2);
}

TEST(Test__DeviceAddressAdapter, FromTypeAndOrdinal_Cpu)
{
    auto addr = DeviceAddressAdapter::fromTypeAndOrdinal(DeviceType::CPU, 0);

    EXPECT_EQ(addr.device_type, DeviceType::CPU);
    EXPECT_EQ(addr.device_ordinal, 0);
}

TEST(Test__DeviceAddressAdapter, FromDeviceId_RawInt)
{
    auto addr = DeviceAddressAdapter::fromDeviceId(3, DeviceType::CUDA);

    EXPECT_EQ(addr.device_type, DeviceType::CUDA);
    EXPECT_EQ(addr.device_ordinal, 3);
}

TEST(Test__DeviceAddressAdapter, FromDeviceId_DeviceIdStruct)
{
    DeviceId device_id = DeviceId::cuda(2);
    auto addr = DeviceAddressAdapter::fromDeviceId(device_id, 1, "gpu-host");

    EXPECT_EQ(addr.hostname, "gpu-host");
    EXPECT_EQ(addr.numa_node, 1);
    EXPECT_EQ(addr.device_type, DeviceType::CUDA);
    EXPECT_EQ(addr.device_ordinal, 2);
}

TEST(Test__DeviceAddressAdapter, FromCudaDevice)
{
    auto addr = DeviceAddressAdapter::fromCudaDevice(0);

    EXPECT_EQ(addr.device_type, DeviceType::CUDA);
    EXPECT_EQ(addr.device_ordinal, 0);
    EXPECT_TRUE(addr.isLocal());
}

TEST(Test__DeviceAddressAdapter, FromCudaDevice_WithNuma)
{
    auto addr = DeviceAddressAdapter::fromCudaDevice(1, 3);

    EXPECT_EQ(addr.device_type, DeviceType::CUDA);
    EXPECT_EQ(addr.device_ordinal, 1);
    EXPECT_EQ(addr.numa_node, 3);
}

TEST(Test__DeviceAddressAdapter, FromRocmDevice)
{
    auto addr = DeviceAddressAdapter::fromRocmDevice(0);

    EXPECT_EQ(addr.device_type, DeviceType::ROCm);
    EXPECT_EQ(addr.device_ordinal, 0);
}

TEST(Test__DeviceAddressAdapter, FromRocmDevice_WithNuma)
{
    auto addr = DeviceAddressAdapter::fromRocmDevice(2, 1);

    EXPECT_EQ(addr.device_type, DeviceType::ROCm);
    EXPECT_EQ(addr.device_ordinal, 2);
    EXPECT_EQ(addr.numa_node, 1);
}

TEST(Test__DeviceAddressAdapter, FromCpuSocket)
{
    auto addr = DeviceAddressAdapter::fromCpuSocket(1);

    EXPECT_EQ(addr.device_type, DeviceType::CPU);
    EXPECT_EQ(addr.numa_node, 1);
    EXPECT_EQ(addr.device_ordinal, 0);
}

// ============================================================================
// From GlobalDeviceAddress to Legacy Tests
// ============================================================================

TEST(Test__DeviceAddressAdapter, ToOrdinal)
{
    GlobalDeviceAddress addr = GlobalDeviceAddress::cuda(2);
    EXPECT_EQ(DeviceAddressAdapter::toOrdinal(addr), 2);
}

TEST(Test__DeviceAddressAdapter, ToOrdinal_Cpu)
{
    GlobalDeviceAddress addr = GlobalDeviceAddress::cpu(0);
    EXPECT_EQ(DeviceAddressAdapter::toOrdinal(addr), 0);
}

TEST(Test__DeviceAddressAdapter, ToDeviceId_Cuda)
{
    GlobalDeviceAddress addr = GlobalDeviceAddress::cuda(1);
    DeviceId device_id = DeviceAddressAdapter::toDeviceId(addr);

    EXPECT_EQ(device_id.type, DeviceType::CUDA);
    EXPECT_EQ(device_id.ordinal, 1);
    EXPECT_TRUE(device_id.is_cuda());
}

TEST(Test__DeviceAddressAdapter, ToDeviceId_Rocm)
{
    GlobalDeviceAddress addr = GlobalDeviceAddress::rocm(2);
    DeviceId device_id = DeviceAddressAdapter::toDeviceId(addr);

    EXPECT_EQ(device_id.type, DeviceType::ROCm);
    EXPECT_EQ(device_id.ordinal, 2);
    EXPECT_TRUE(device_id.is_rocm());
}

TEST(Test__DeviceAddressAdapter, ToDeviceId_Cpu)
{
    GlobalDeviceAddress addr = GlobalDeviceAddress::cpu(0);
    DeviceId device_id = DeviceAddressAdapter::toDeviceId(addr);

    EXPECT_EQ(device_id.type, DeviceType::CPU);
    EXPECT_TRUE(device_id.is_cpu());
}

TEST(Test__DeviceAddressAdapter, ToCudaDevice)
{
    GlobalDeviceAddress addr = GlobalDeviceAddress::cuda(3);
    EXPECT_EQ(DeviceAddressAdapter::toCudaDevice(addr), 3);
}

TEST(Test__DeviceAddressAdapter, ToCudaDevice_ThrowsForNonCuda)
{
    GlobalDeviceAddress addr = GlobalDeviceAddress::rocm(0);
    EXPECT_THROW(DeviceAddressAdapter::toCudaDevice(addr), std::invalid_argument);
}

TEST(Test__DeviceAddressAdapter, ToCudaDevice_ThrowsForCpu)
{
    GlobalDeviceAddress addr = GlobalDeviceAddress::cpu(0);
    EXPECT_THROW(DeviceAddressAdapter::toCudaDevice(addr), std::invalid_argument);
}

TEST(Test__DeviceAddressAdapter, ToRocmDevice)
{
    GlobalDeviceAddress addr = GlobalDeviceAddress::rocm(2);
    EXPECT_EQ(DeviceAddressAdapter::toRocmDevice(addr), 2);
}

TEST(Test__DeviceAddressAdapter, ToRocmDevice_ThrowsForNonRocm)
{
    GlobalDeviceAddress addr = GlobalDeviceAddress::cuda(0);
    EXPECT_THROW(DeviceAddressAdapter::toRocmDevice(addr), std::invalid_argument);
}

TEST(Test__DeviceAddressAdapter, ToNumaNode)
{
    GlobalDeviceAddress addr;
    addr.hostname = "localhost";
    addr.numa_node = 3;
    addr.device_type = DeviceType::CUDA;
    addr.device_ordinal = 0;

    EXPECT_EQ(DeviceAddressAdapter::toNumaNode(addr), 3);
}

// ============================================================================
// Type Predicate Tests
// ============================================================================

TEST(Test__DeviceAddressAdapter, ExtractType)
{
    GlobalDeviceAddress cuda = GlobalDeviceAddress::cuda(0);
    GlobalDeviceAddress rocm = GlobalDeviceAddress::rocm(0);
    GlobalDeviceAddress cpu = GlobalDeviceAddress::cpu(0);

    EXPECT_EQ(DeviceAddressAdapter::extractType(cuda), DeviceType::CUDA);
    EXPECT_EQ(DeviceAddressAdapter::extractType(rocm), DeviceType::ROCm);
    EXPECT_EQ(DeviceAddressAdapter::extractType(cpu), DeviceType::CPU);
}

TEST(Test__DeviceAddressAdapter, IsGpu)
{
    GlobalDeviceAddress cuda = GlobalDeviceAddress::cuda(0);
    GlobalDeviceAddress rocm = GlobalDeviceAddress::rocm(0);
    GlobalDeviceAddress cpu = GlobalDeviceAddress::cpu(0);

    EXPECT_TRUE(DeviceAddressAdapter::isGpu(cuda));
    EXPECT_TRUE(DeviceAddressAdapter::isGpu(rocm));
    EXPECT_FALSE(DeviceAddressAdapter::isGpu(cpu));
}

TEST(Test__DeviceAddressAdapter, IsCpu)
{
    GlobalDeviceAddress cuda = GlobalDeviceAddress::cuda(0);
    GlobalDeviceAddress rocm = GlobalDeviceAddress::rocm(0);
    GlobalDeviceAddress cpu = GlobalDeviceAddress::cpu(0);

    EXPECT_FALSE(DeviceAddressAdapter::isCpu(cuda));
    EXPECT_FALSE(DeviceAddressAdapter::isCpu(rocm));
    EXPECT_TRUE(DeviceAddressAdapter::isCpu(cpu));
}

TEST(Test__DeviceAddressAdapter, IsCuda)
{
    GlobalDeviceAddress cuda = GlobalDeviceAddress::cuda(0);
    GlobalDeviceAddress rocm = GlobalDeviceAddress::rocm(0);

    EXPECT_TRUE(DeviceAddressAdapter::isCuda(cuda));
    EXPECT_FALSE(DeviceAddressAdapter::isCuda(rocm));
}

TEST(Test__DeviceAddressAdapter, IsRocm)
{
    GlobalDeviceAddress cuda = GlobalDeviceAddress::cuda(0);
    GlobalDeviceAddress rocm = GlobalDeviceAddress::rocm(0);

    EXPECT_FALSE(DeviceAddressAdapter::isRocm(cuda));
    EXPECT_TRUE(DeviceAddressAdapter::isRocm(rocm));
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST(Test__DeviceAddressAdapter, IsValidForCompute_ValidTypes)
{
    GlobalDeviceAddress cuda = GlobalDeviceAddress::cuda(0);
    GlobalDeviceAddress rocm = GlobalDeviceAddress::rocm(0);
    GlobalDeviceAddress cpu = GlobalDeviceAddress::cpu(0);

    EXPECT_TRUE(DeviceAddressAdapter::isValidForCompute(cuda));
    EXPECT_TRUE(DeviceAddressAdapter::isValidForCompute(rocm));
    EXPECT_TRUE(DeviceAddressAdapter::isValidForCompute(cpu));
}

TEST(Test__DeviceAddressAdapter, IsValidForCompute_NegativeOrdinal)
{
    GlobalDeviceAddress addr;
    addr.hostname = "localhost";
    addr.numa_node = 0;
    addr.device_type = DeviceType::CUDA;
    addr.device_ordinal = -1;

    EXPECT_FALSE(DeviceAddressAdapter::isValidForCompute(addr));
}

TEST(Test__DeviceAddressAdapter, IsValidForCompute_UnsupportedType)
{
    GlobalDeviceAddress vulkan;
    vulkan.hostname = "localhost";
    vulkan.numa_node = 0;
    vulkan.device_type = DeviceType::Vulkan;
    vulkan.device_ordinal = 0;

    EXPECT_FALSE(DeviceAddressAdapter::isValidForCompute(vulkan));

    GlobalDeviceAddress metal;
    metal.hostname = "localhost";
    metal.numa_node = 0;
    metal.device_type = DeviceType::Metal;
    metal.device_ordinal = 0;

    EXPECT_FALSE(DeviceAddressAdapter::isValidForCompute(metal));
}

// ============================================================================
// Round-Trip Tests
// ============================================================================

TEST(Test__DeviceAddressAdapter, RoundTrip_CudaDevice)
{
    // Start with CUDA device ordinal
    int original_ordinal = 3;

    // Convert to GlobalDeviceAddress
    GlobalDeviceAddress addr = DeviceAddressAdapter::fromCudaDevice(original_ordinal, 1);

    // Convert back
    int recovered_ordinal = DeviceAddressAdapter::toCudaDevice(addr);
    int recovered_numa = DeviceAddressAdapter::toNumaNode(addr);

    EXPECT_EQ(recovered_ordinal, original_ordinal);
    EXPECT_EQ(recovered_numa, 1);
}

TEST(Test__DeviceAddressAdapter, RoundTrip_DeviceId)
{
    DeviceId original = DeviceId::rocm(2);

    // Convert to GlobalDeviceAddress
    GlobalDeviceAddress addr = DeviceAddressAdapter::fromDeviceId(original, 3);

    // Convert back
    DeviceId recovered = DeviceAddressAdapter::toDeviceId(addr);

    EXPECT_EQ(recovered.type, original.type);
    EXPECT_EQ(recovered.ordinal, original.ordinal);
}

TEST(Test__DeviceAddressAdapter, RoundTrip_TypeAndOrdinal)
{
    DeviceType original_type = DeviceType::CUDA;
    int original_ordinal = 5;
    int original_numa = 2;

    // Convert to GlobalDeviceAddress
    GlobalDeviceAddress addr = DeviceAddressAdapter::fromTypeAndOrdinal(
        original_type, original_ordinal, original_numa);

    // Extract back
    DeviceType recovered_type = DeviceAddressAdapter::extractType(addr);
    int recovered_ordinal = DeviceAddressAdapter::toOrdinal(addr);
    int recovered_numa = DeviceAddressAdapter::toNumaNode(addr);

    EXPECT_EQ(recovered_type, original_type);
    EXPECT_EQ(recovered_ordinal, original_ordinal);
    EXPECT_EQ(recovered_numa, original_numa);
}
