/**
 * @file Test__DevicePCIAddress.cpp
 * @brief Prove OS resource identities survive discovery, rank exchange and serving evidence.
 *
 * These fixtures deliberately separate native endpoints from visible device
 * ordinals and upstream PCIe bridges. They run without a device or MPI session;
 * stale wire versions and incomplete identities must fail before measurement.
 */
#include <gtest/gtest.h>
#include <cstring>
#include "backends/DevicePCIAddress.h"
#include "app/modes/ServerExecutionEvidence.h"
#include "utils/MPITopology.h"
#include "../../utils/CPUExecutionTestGeometry.h"

using namespace llaminar2;

TEST(DevicePCIIdentity, NativeSpellingRetainsDomainAndFunction)
{
    EXPECT_EQ(DevicePCIAddress::parse("000000AB:CD:1F.7").toString(), "00ab:cd:1f.7");
    EXPECT_EQ(DevicePCIAddress::parse("12345678:ff:1f.7").toString(), "12345678:ff:1f.7");
    EXPECT_EQ(DevicePCIAddress::parse("0000:00:00.0").toString(), "0000:00:00.0");
    for (const auto text : {"", "0", "41:00.0", "0000:41:00", "0000:41:20.0",
            "0000:41:00.8", "0000:gg:00.0", "0000:41:00.0 ", " 000:41:00.0",
            "000:41:00.0", "0000:41:00.0x", "0000:41:00.+"})
        EXPECT_THROW(DevicePCIAddress::parse(text), std::invalid_argument) << text;
}

TEST(DevicePCIIdentity, MPIWireAndServingEvidencePreserveNativeEndpoints)
{
    RankInventory rank{.rank = 0, .node_id = 0};
    rank.cpu_execution = test::kSyntheticCPUExecutionGeometry;
    rank.gpus = {{.type = DeviceType::CUDA, .local_device_id = 7, .uuid = "cuda-uuid",
                  .pci_bus_address = "0001:41:03.2"},
                 {.type = DeviceType::ROCm, .local_device_id = 2, .uuid = "rocm-uuid",
                  .pci_bus_address = "0002:84:1f.7"}};
    rank.gpus[1].pcie_bottleneck_bdf = "0002:81:00.0";
    const auto bytes = MPITopology::serializeRankInventory(rank);
    auto decoded = MPITopology::deserializeRankInventory(bytes.data(), bytes.size());
    ASSERT_EQ(decoded.gpus.size(), 2u);
    for (const auto device : {DeviceId::cuda(7), DeviceId::rocm(2)})
        EXPECT_EQ(serverExecutionParticipantTags(device, rank, 0),
                  serverExecutionParticipantTags(device, decoded, 0));
    EXPECT_EQ(serverExecutionParticipantTags(DeviceId::rocm(2), decoded, 0).at("pci_bus_address"),
              "0002:84:1f.7");
    decoded.gpus[1].local_device_id = 9;
    EXPECT_EQ(serverExecutionParticipantTags(DeviceId::rocm(9), decoded, 0),
              serverExecutionParticipantTags(DeviceId::rocm(2), rank, 0));
    decoded.gpus[1].pci_bus_address.clear();
    EXPECT_THROW(serverExecutionParticipantTags(DeviceId::rocm(9), decoded, 0), std::invalid_argument);
    auto stale = bytes;
    const uint32_t old_version = 3;
    std::memcpy(stale.data() + sizeof(uint32_t), &old_version, sizeof(old_version));
    EXPECT_THROW(MPITopology::deserializeRankInventory(stale.data(), stale.size()), std::runtime_error);
}

TEST(DevicePCIIdentity, DuplicatePhysicalViewsCannotDisagreeOnEndpoint)
{
    ClusterInventory inventory;
    RankInventory rank{.rank = 0, .node_id = 0};
    rank.gpus = {{.type = DeviceType::ROCm, .local_device_id = 2, .uuid = "physical-uuid",
                  .pci_bus_address = "0002:84:1f.7"}};
    inventory.ranks = {rank, rank};
    inventory.ranks[1].rank = 1;
    inventory.ranks[1].gpus[0].local_device_id = 7;
    inventory.buildNodeAggregations();
    EXPECT_EQ(inventory.total_gpus, 1);
    inventory.ranks[1].gpus[0].pci_bus_address = "0002:85:1f.7";
    EXPECT_THROW(inventory.buildNodeAggregations(), std::invalid_argument);
}

TEST(DevicePCIIdentity, NativeProcessIdentityRejectsMissingPid)
{
    EXPECT_EQ(serverProcessIdentityTags(17).at("pid"), "17");
    EXPECT_EQ(serverProcessIdentityTags(17).at("identity_source"), "native_pid_namespace");
    EXPECT_THROW(serverProcessIdentityTags(0), std::invalid_argument);
    EXPECT_THROW(serverProcessIdentityTags(-1), std::invalid_argument);
}
