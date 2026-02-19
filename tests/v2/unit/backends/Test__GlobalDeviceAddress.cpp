/**
 * @file Test__GlobalDeviceAddress.cpp
 * @brief Unit tests for GlobalDeviceAddress
 *
 * Tests:
 * - Factory methods (cpu, cuda, rocm)
 * - Parsing full form, shorthand forms
 * - Invalid format handling
 * - toString/toShortString round-trips
 * - Predicates (isLocal, isGPU, sameNuma, sameHost)
 * - Comparison operators
 * - Hash function (use in unordered_set)
 * - DeviceId conversion
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <unordered_set>
#include <set>
#include "backends/GlobalDeviceAddress.h"

using namespace llaminar2;

// ============================================================================
// Factory Method Tests
// ============================================================================

TEST(Test__GlobalDeviceAddress, FactoryMethod_CPU)
{
    auto addr = GlobalDeviceAddress::cpu();

    EXPECT_EQ(addr.hostname, "localhost");
    EXPECT_EQ(addr.numa_node, NUMA_NODE_UNKNOWN);
    EXPECT_EQ(addr.device_type, DeviceType::CPU);
    EXPECT_EQ(addr.device_ordinal, 0);
    EXPECT_TRUE(addr.isCPU());
    EXPECT_FALSE(addr.isGPU());
    EXPECT_FALSE(addr.hasValidNuma());
}

TEST(Test__GlobalDeviceAddress, FactoryMethod_CPU_WithNuma)
{
    auto addr = GlobalDeviceAddress::cpu(1);

    EXPECT_EQ(addr.hostname, "localhost");
    EXPECT_EQ(addr.numa_node, 1);
    EXPECT_EQ(addr.device_type, DeviceType::CPU);
    EXPECT_EQ(addr.device_ordinal, 0);
}

TEST(Test__GlobalDeviceAddress, FactoryMethod_CPU_WithHostname)
{
    auto addr = GlobalDeviceAddress::cpu(2, "node1");

    EXPECT_EQ(addr.hostname, "node1");
    EXPECT_EQ(addr.numa_node, 2);
    EXPECT_EQ(addr.device_type, DeviceType::CPU);
}

TEST(Test__GlobalDeviceAddress, FactoryMethod_CUDA)
{
    auto addr = GlobalDeviceAddress::cuda(0);

    EXPECT_EQ(addr.hostname, "localhost");
    EXPECT_EQ(addr.numa_node, NUMA_NODE_UNKNOWN);
    EXPECT_EQ(addr.device_type, DeviceType::CUDA);
    EXPECT_EQ(addr.device_ordinal, 0);
    EXPECT_TRUE(addr.isCUDA());
    EXPECT_TRUE(addr.isGPU());
    EXPECT_FALSE(addr.isCPU());
    EXPECT_FALSE(addr.hasValidNuma());
}

TEST(Test__GlobalDeviceAddress, FactoryMethod_CUDA_WithOrdinal)
{
    auto addr = GlobalDeviceAddress::cuda(3, 1, "gpu-server");

    EXPECT_EQ(addr.hostname, "gpu-server");
    EXPECT_EQ(addr.numa_node, 1);
    EXPECT_EQ(addr.device_type, DeviceType::CUDA);
    EXPECT_EQ(addr.device_ordinal, 3);
}

TEST(Test__GlobalDeviceAddress, FactoryMethod_ROCm)
{
    auto addr = GlobalDeviceAddress::rocm(0);

    EXPECT_EQ(addr.device_type, DeviceType::ROCm);
    EXPECT_TRUE(addr.isROCm());
    EXPECT_TRUE(addr.isGPU());
    EXPECT_FALSE(addr.isCUDA());
}

// ============================================================================
// Parsing Tests - Full Form
// ============================================================================

TEST(Test__GlobalDeviceAddress, Parse_FullForm_CUDA)
{
    auto addr = GlobalDeviceAddress::parse("node1:0:cuda:0");

    EXPECT_EQ(addr.hostname, "node1");
    EXPECT_EQ(addr.numa_node, 0);
    EXPECT_EQ(addr.device_type, DeviceType::CUDA);
    EXPECT_EQ(addr.device_ordinal, 0);
}

TEST(Test__GlobalDeviceAddress, Parse_FullForm_ROCm)
{
    auto addr = GlobalDeviceAddress::parse("node2:1:rocm:2");

    EXPECT_EQ(addr.hostname, "node2");
    EXPECT_EQ(addr.numa_node, 1);
    EXPECT_EQ(addr.device_type, DeviceType::ROCm);
    EXPECT_EQ(addr.device_ordinal, 2);
}

TEST(Test__GlobalDeviceAddress, Parse_FullForm_CPU)
{
    auto addr = GlobalDeviceAddress::parse("localhost:0:cpu:0");

    EXPECT_EQ(addr.hostname, "localhost");
    EXPECT_EQ(addr.numa_node, 0);
    EXPECT_EQ(addr.device_type, DeviceType::CPU);
    EXPECT_EQ(addr.device_ordinal, 0);
}

TEST(Test__GlobalDeviceAddress, Parse_FullForm_HIP_Alias)
{
    // "hip" should be accepted as alias for ROCm
    auto addr = GlobalDeviceAddress::parse("localhost:0:hip:0");

    EXPECT_EQ(addr.device_type, DeviceType::ROCm);
}

// ============================================================================
// Parsing Tests - Shorthand Forms
// ============================================================================

TEST(Test__GlobalDeviceAddress, Parse_ShortForm_TypeOrdinal)
{
    // "cuda:0" -> localhost:NUMA_NODE_UNKNOWN:cuda:0 (NUMA unknown when not specified)
    auto addr = GlobalDeviceAddress::parse("cuda:0");

    EXPECT_EQ(addr.hostname, "localhost");
    EXPECT_EQ(addr.numa_node, NUMA_NODE_UNKNOWN);
    EXPECT_EQ(addr.device_type, DeviceType::CUDA);
    EXPECT_EQ(addr.device_ordinal, 0);
}

TEST(Test__GlobalDeviceAddress, Parse_ShortForm_TypeOrdinal_WithExplicitNuma)
{
    // When current_numa is explicitly provided, use it
    auto addr = GlobalDeviceAddress::parse("cuda:0", 0);

    EXPECT_EQ(addr.hostname, "localhost");
    EXPECT_EQ(addr.numa_node, 0);
    EXPECT_EQ(addr.device_type, DeviceType::CUDA);
    EXPECT_EQ(addr.device_ordinal, 0);
    EXPECT_TRUE(addr.hasValidNuma());
}

TEST(Test__GlobalDeviceAddress, Parse_ShortForm_TypeOrdinal_WithCurrentNuma)
{
    auto addr = GlobalDeviceAddress::parse("cuda:1", 2);

    EXPECT_EQ(addr.hostname, "localhost");
    EXPECT_EQ(addr.numa_node, 2); // Uses current_numa
    EXPECT_EQ(addr.device_type, DeviceType::CUDA);
    EXPECT_EQ(addr.device_ordinal, 1);
}

TEST(Test__GlobalDeviceAddress, Parse_MediumForm_NumaTypeOrdinal)
{
    // "0:cuda:0" -> localhost:0:cuda:0
    auto addr = GlobalDeviceAddress::parse("0:cuda:0");

    EXPECT_EQ(addr.hostname, "localhost");
    EXPECT_EQ(addr.numa_node, 0);
    EXPECT_EQ(addr.device_type, DeviceType::CUDA);
    EXPECT_EQ(addr.device_ordinal, 0);
}

TEST(Test__GlobalDeviceAddress, Parse_MediumForm_DifferentNuma)
{
    auto addr = GlobalDeviceAddress::parse("1:rocm:3");

    EXPECT_EQ(addr.hostname, "localhost");
    EXPECT_EQ(addr.numa_node, 1);
    EXPECT_EQ(addr.device_type, DeviceType::ROCm);
    EXPECT_EQ(addr.device_ordinal, 3);
}

// ============================================================================
// Parsing Tests - Case Insensitivity
// ============================================================================

TEST(Test__GlobalDeviceAddress, Parse_CaseInsensitive_CUDA)
{
    EXPECT_EQ(GlobalDeviceAddress::parse("CUDA:0").device_type, DeviceType::CUDA);
    EXPECT_EQ(GlobalDeviceAddress::parse("Cuda:0").device_type, DeviceType::CUDA);
    EXPECT_EQ(GlobalDeviceAddress::parse("cuda:0").device_type, DeviceType::CUDA);
}

TEST(Test__GlobalDeviceAddress, Parse_CaseInsensitive_ROCm)
{
    EXPECT_EQ(GlobalDeviceAddress::parse("ROCM:0").device_type, DeviceType::ROCm);
    EXPECT_EQ(GlobalDeviceAddress::parse("rocm:0").device_type, DeviceType::ROCm);
    EXPECT_EQ(GlobalDeviceAddress::parse("HIP:0").device_type, DeviceType::ROCm);
}

// ============================================================================
// Parsing Tests - Invalid Formats
// ============================================================================

TEST(Test__GlobalDeviceAddress, TryParse_EmptyString_ReturnsNullopt)
{
    auto result = GlobalDeviceAddress::tryParse("");
    EXPECT_FALSE(result.has_value());
}

TEST(Test__GlobalDeviceAddress, TryParse_InvalidType_ReturnsNullopt)
{
    auto result = GlobalDeviceAddress::tryParse("invalid:0");
    EXPECT_FALSE(result.has_value());
}

TEST(Test__GlobalDeviceAddress, TryParse_NegativeOrdinal_ReturnsNullopt)
{
    auto result = GlobalDeviceAddress::tryParse("cuda:-1");
    EXPECT_FALSE(result.has_value());
}

TEST(Test__GlobalDeviceAddress, TryParse_NegativeNuma_MediumForm_ReturnsNullopt)
{
    // Explicitly parsed NUMA < -1 is invalid
    auto result = GlobalDeviceAddress::tryParse("-2:cuda:0");
    EXPECT_FALSE(result.has_value());
}

TEST(Test__GlobalDeviceAddress, TryParse_NegativeOneNuma_MediumForm_Succeeds)
{
    // NUMA_NODE_UNKNOWN (-1) is now a valid sentinel value
    auto result = GlobalDeviceAddress::tryParse("-1:cuda:0");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->numa_node, NUMA_NODE_UNKNOWN);
    EXPECT_FALSE(result->hasValidNuma());
}

TEST(Test__GlobalDeviceAddress, TryParse_NonNumericOrdinal_ReturnsNullopt)
{
    auto result = GlobalDeviceAddress::tryParse("cuda:abc");
    EXPECT_FALSE(result.has_value());
}

TEST(Test__GlobalDeviceAddress, TryParse_TooManyParts_ReturnsNullopt)
{
    auto result = GlobalDeviceAddress::tryParse("a:b:c:d:e");
    EXPECT_FALSE(result.has_value());
}

TEST(Test__GlobalDeviceAddress, TryParse_SinglePart_ReturnsNullopt)
{
    auto result = GlobalDeviceAddress::tryParse("cuda");
    EXPECT_FALSE(result.has_value());
}

TEST(Test__GlobalDeviceAddress, Parse_InvalidFormat_Throws)
{
    EXPECT_THROW(GlobalDeviceAddress::parse("invalid:format"), std::invalid_argument);
}

// ============================================================================
// Serialization Tests
// ============================================================================

TEST(Test__GlobalDeviceAddress, ToString_FullForm)
{
    auto addr = GlobalDeviceAddress::cuda(0, 0, "node1");
    EXPECT_EQ(addr.toString(), "node1:0:cuda:0");
}

TEST(Test__GlobalDeviceAddress, ToString_CPU)
{
    auto addr = GlobalDeviceAddress::cpu(1, "localhost");
    EXPECT_EQ(addr.toString(), "localhost:1:cpu:0");
}

TEST(Test__GlobalDeviceAddress, ToShortString_LocalhostNuma0)
{
    auto addr = GlobalDeviceAddress::cuda(0, 0, "localhost");
    EXPECT_EQ(addr.toShortString(), "cuda:0");
}

TEST(Test__GlobalDeviceAddress, ToShortString_LocalhostDifferentNuma)
{
    auto addr = GlobalDeviceAddress::cuda(0, 1, "localhost");
    EXPECT_EQ(addr.toShortString(), "1:cuda:0");
}

TEST(Test__GlobalDeviceAddress, ToShortString_RemoteHost)
{
    auto addr = GlobalDeviceAddress::cuda(0, 0, "node1");
    EXPECT_EQ(addr.toShortString(), "node1:0:cuda:0");
}

TEST(Test__GlobalDeviceAddress, ToString_RoundTrip_FullForm)
{
    std::string original = "node1:0:cuda:0";
    auto addr = GlobalDeviceAddress::parse(original);
    EXPECT_EQ(addr.toString(), original);
}

TEST(Test__GlobalDeviceAddress, ToString_RoundTrip_ShortForm)
{
    auto addr = GlobalDeviceAddress::parse("cuda:0");
    auto str = addr.toString();
    auto reparsed = GlobalDeviceAddress::parse(str);
    EXPECT_EQ(addr, reparsed);
}

// ============================================================================
// Predicate Tests
// ============================================================================

TEST(Test__GlobalDeviceAddress, IsLocal_Localhost_ReturnsTrue)
{
    auto addr = GlobalDeviceAddress::cuda(0, 0, "localhost");
    EXPECT_TRUE(addr.isLocal());
}

TEST(Test__GlobalDeviceAddress, IsLocal_EmptyHostname_ReturnsTrue)
{
    GlobalDeviceAddress addr;
    addr.hostname = "";
    EXPECT_TRUE(addr.isLocal());
}

TEST(Test__GlobalDeviceAddress, IsLocal_RemoteHost_ReturnsFalse)
{
    auto addr = GlobalDeviceAddress::cuda(0, 0, "node1");
    EXPECT_FALSE(addr.isLocal());
}

TEST(Test__GlobalDeviceAddress, SameNuma_SameHostAndNuma_ReturnsTrue)
{
    auto addr1 = GlobalDeviceAddress::cuda(0, 0, "node1");
    auto addr2 = GlobalDeviceAddress::rocm(0, 0, "node1");
    EXPECT_TRUE(addr1.sameNuma(addr2));
}

TEST(Test__GlobalDeviceAddress, SameNuma_DifferentNuma_ReturnsFalse)
{
    auto addr1 = GlobalDeviceAddress::cuda(0, 0, "node1");
    auto addr2 = GlobalDeviceAddress::cuda(0, 1, "node1");
    EXPECT_FALSE(addr1.sameNuma(addr2));
}

TEST(Test__GlobalDeviceAddress, SameNuma_DifferentHost_ReturnsFalse)
{
    auto addr1 = GlobalDeviceAddress::cuda(0, 0, "node1");
    auto addr2 = GlobalDeviceAddress::cuda(0, 0, "node2");
    EXPECT_FALSE(addr1.sameNuma(addr2));
}

TEST(Test__GlobalDeviceAddress, SameNuma_UnknownNuma_ReturnsFalse)
{
    // If either side has unknown NUMA, sameNuma should be conservative and return false
    auto addr1 = GlobalDeviceAddress::cuda(0); // NUMA_NODE_UNKNOWN
    auto addr2 = GlobalDeviceAddress::cuda(1); // NUMA_NODE_UNKNOWN
    EXPECT_FALSE(addr1.sameNuma(addr2));
}

TEST(Test__GlobalDeviceAddress, SameNuma_OneUnknown_ReturnsFalse)
{
    auto addr1 = GlobalDeviceAddress::cuda(0, 0, "localhost"); // NUMA 0 (explicit)
    auto addr2 = GlobalDeviceAddress::cuda(1);                  // NUMA_NODE_UNKNOWN
    EXPECT_FALSE(addr1.sameNuma(addr2));
}

TEST(Test__GlobalDeviceAddress, SameHost_SameHost_ReturnsTrue)
{
    auto addr1 = GlobalDeviceAddress::cuda(0, 0, "node1");
    auto addr2 = GlobalDeviceAddress::rocm(1, 1, "node1");
    EXPECT_TRUE(addr1.sameHost(addr2));
}

TEST(Test__GlobalDeviceAddress, SameHost_DifferentHost_ReturnsFalse)
{
    auto addr1 = GlobalDeviceAddress::cuda(0, 0, "node1");
    auto addr2 = GlobalDeviceAddress::cuda(0, 0, "node2");
    EXPECT_FALSE(addr1.sameHost(addr2));
}

// ============================================================================
// Comparison Operator Tests
// ============================================================================

TEST(Test__GlobalDeviceAddress, Equality_SameValues_ReturnsTrue)
{
    auto addr1 = GlobalDeviceAddress::cuda(0, 0, "node1");
    auto addr2 = GlobalDeviceAddress::cuda(0, 0, "node1");
    EXPECT_EQ(addr1, addr2);
}

TEST(Test__GlobalDeviceAddress, Equality_DifferentOrdinal_ReturnsFalse)
{
    auto addr1 = GlobalDeviceAddress::cuda(0);
    auto addr2 = GlobalDeviceAddress::cuda(1);
    EXPECT_NE(addr1, addr2);
}

TEST(Test__GlobalDeviceAddress, Equality_DifferentType_ReturnsFalse)
{
    auto addr1 = GlobalDeviceAddress::cuda(0);
    auto addr2 = GlobalDeviceAddress::rocm(0);
    EXPECT_NE(addr1, addr2);
}

TEST(Test__GlobalDeviceAddress, Equality_DifferentNuma_ReturnsFalse)
{
    auto addr1 = GlobalDeviceAddress::cuda(0, 0);
    auto addr2 = GlobalDeviceAddress::cuda(0, 1);
    EXPECT_NE(addr1, addr2);
}

TEST(Test__GlobalDeviceAddress, Equality_DifferentHost_ReturnsFalse)
{
    auto addr1 = GlobalDeviceAddress::cuda(0, 0, "node1");
    auto addr2 = GlobalDeviceAddress::cuda(0, 0, "node2");
    EXPECT_NE(addr1, addr2);
}

TEST(Test__GlobalDeviceAddress, LessThan_OrdersByHostname)
{
    auto addr1 = GlobalDeviceAddress::cuda(0, 0, "aaa");
    auto addr2 = GlobalDeviceAddress::cuda(0, 0, "bbb");
    EXPECT_LT(addr1, addr2);
}

TEST(Test__GlobalDeviceAddress, LessThan_OrdersByNuma)
{
    auto addr1 = GlobalDeviceAddress::cuda(0, 0);
    auto addr2 = GlobalDeviceAddress::cuda(0, 1);
    EXPECT_LT(addr1, addr2);
}

TEST(Test__GlobalDeviceAddress, LessThan_OrdersByType)
{
    auto addr1 = GlobalDeviceAddress::cpu();
    auto addr2 = GlobalDeviceAddress::cuda(0);
    EXPECT_LT(addr1, addr2); // CPU < CUDA in enum order
}

TEST(Test__GlobalDeviceAddress, LessThan_OrdersByOrdinal)
{
    auto addr1 = GlobalDeviceAddress::cuda(0);
    auto addr2 = GlobalDeviceAddress::cuda(1);
    EXPECT_LT(addr1, addr2);
}

// ============================================================================
// Hash Function Tests
// ============================================================================

TEST(Test__GlobalDeviceAddress, Hash_SameValues_SameHash)
{
    auto addr1 = GlobalDeviceAddress::cuda(0, 0, "node1");
    auto addr2 = GlobalDeviceAddress::cuda(0, 0, "node1");

    std::hash<GlobalDeviceAddress> hasher;
    EXPECT_EQ(hasher(addr1), hasher(addr2));
}

TEST(Test__GlobalDeviceAddress, Hash_DifferentValues_DifferentHash)
{
    auto addr1 = GlobalDeviceAddress::cuda(0);
    auto addr2 = GlobalDeviceAddress::cuda(1);

    std::hash<GlobalDeviceAddress> hasher;
    EXPECT_NE(hasher(addr1), hasher(addr2));
}

TEST(Test__GlobalDeviceAddress, Hash_WorksInUnorderedSet)
{
    std::unordered_set<GlobalDeviceAddress> devices;

    devices.insert(GlobalDeviceAddress::cuda(0));
    devices.insert(GlobalDeviceAddress::cuda(1));
    devices.insert(GlobalDeviceAddress::rocm(0));
    devices.insert(GlobalDeviceAddress::cpu());

    EXPECT_EQ(devices.size(), 4);

    // Duplicates should not increase size
    devices.insert(GlobalDeviceAddress::cuda(0));
    EXPECT_EQ(devices.size(), 4);

    // Check contains
    EXPECT_TRUE(devices.count(GlobalDeviceAddress::cuda(0)) > 0);
    EXPECT_TRUE(devices.count(GlobalDeviceAddress::rocm(0)) > 0);
    EXPECT_FALSE(devices.count(GlobalDeviceAddress::cuda(2)) > 0);
}

TEST(Test__GlobalDeviceAddress, Comparison_WorksInSet)
{
    std::set<GlobalDeviceAddress> devices;

    devices.insert(GlobalDeviceAddress::cuda(1));
    devices.insert(GlobalDeviceAddress::cuda(0));
    devices.insert(GlobalDeviceAddress::cpu());

    EXPECT_EQ(devices.size(), 3);

    // Verify ordering
    auto it = devices.begin();
    EXPECT_EQ(it->device_type, DeviceType::CPU); // CPU comes first
    ++it;
    EXPECT_EQ(it->device_ordinal, 0); // cuda:0 comes before cuda:1
}

// ============================================================================
// DeviceId Conversion Tests
// ============================================================================

TEST(Test__GlobalDeviceAddress, ToLocalDeviceId_CPU)
{
    auto addr = GlobalDeviceAddress::cpu(1, "node1");
    auto device_id = addr.toLocalDeviceId();

    EXPECT_TRUE(device_id.is_cpu());
    EXPECT_EQ(device_id.ordinal, 0);
}

TEST(Test__GlobalDeviceAddress, ToLocalDeviceId_CUDA)
{
    auto addr = GlobalDeviceAddress::cuda(2, 1, "node1");
    auto device_id = addr.toLocalDeviceId();

    EXPECT_TRUE(device_id.is_cuda());
    EXPECT_EQ(device_id.ordinal, 2);
}

TEST(Test__GlobalDeviceAddress, ToLocalDeviceId_ROCm)
{
    auto addr = GlobalDeviceAddress::rocm(3);
    auto device_id = addr.toLocalDeviceId();

    EXPECT_TRUE(device_id.is_rocm());
    EXPECT_EQ(device_id.ordinal, 3);
}

TEST(Test__GlobalDeviceAddress, FromLocalDeviceId_CPU)
{
    auto device_id = DeviceId::cpu();
    auto addr = GlobalDeviceAddress::fromLocalDeviceId(device_id, "node1", 2);

    EXPECT_EQ(addr.hostname, "node1");
    EXPECT_EQ(addr.numa_node, 2);
    EXPECT_EQ(addr.device_type, DeviceType::CPU);
}

TEST(Test__GlobalDeviceAddress, FromLocalDeviceId_CUDA)
{
    auto device_id = DeviceId::cuda(1);
    auto addr = GlobalDeviceAddress::fromLocalDeviceId(device_id);

    EXPECT_EQ(addr.hostname, "localhost");
    EXPECT_EQ(addr.numa_node, NUMA_NODE_UNKNOWN);
    EXPECT_EQ(addr.device_type, DeviceType::CUDA);
    EXPECT_EQ(addr.device_ordinal, 1);
    EXPECT_FALSE(addr.hasValidNuma());
}

TEST(Test__GlobalDeviceAddress, RoundTrip_DeviceId)
{
    auto original = DeviceId::cuda(3);
    auto addr = GlobalDeviceAddress::fromLocalDeviceId(original);
    auto converted = addr.toLocalDeviceId();

    EXPECT_EQ(original, converted);
}

// ============================================================================
// Stream Output Tests
// ============================================================================

TEST(Test__GlobalDeviceAddress, StreamOutput)
{
    auto addr = GlobalDeviceAddress::cuda(0, 1, "node1");

    std::ostringstream oss;
    oss << addr;

    EXPECT_EQ(oss.str(), "node1:1:cuda:0");
}
