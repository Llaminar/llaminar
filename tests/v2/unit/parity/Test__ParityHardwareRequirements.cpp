/**
 * @file Test__ParityHardwareRequirements.cpp
 * @brief Device-free exact-participant hardware admission and collective capability regressions.
 *
 * Inventory values are injected into the same pure validator used before model
 * setup. The original participant-list projection also proves mixed pipelines
 * retain GPU snapshot sizing independently of their primary CPU endpoint.
 * No backend, MPI world, model source or device allocation is initialized.
 */
#include "integration/parity/qwen4exp/Qwen4ExpNgramAdmissionCases.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen4exp;

/** @test The actual ROCm0/ROCm1/CPU0 admission topology has no CUDA prerequisite. */
TEST(ParityHardwareRequirements, ActualTypedROCmPipelineNeedsTwoROCmAndNoCUDA)
{
    const auto &cases = qwen4ExpNgramAdmissionCases();
    const auto found = std::find_if(cases.begin(), cases.end(), [](const auto &cell) {
        return cell.ngram_table_policy->storage == ImmutableRowStoragePreference::VRAM;
    });
    ASSERT_NE(found, cases.end());
    const auto config = found->toTestConfig();
    ASSERT_TRUE(config.local_hardware_requirements);
    EXPECT_EQ(config.collective, Collective::HETEROGENEOUS);
    EXPECT_EQ(config.local_hardware_requirements->cuda_devices, 0);
    EXPECT_EQ(config.local_hardware_requirements->rocm_devices, 2);
    EXPECT_FALSE(parityHardwareInventoryFailure(*config.local_hardware_requirements, {0,2}));
    EXPECT_TRUE(parityHardwareInventoryFailure(*config.local_hardware_requirements, {0,1}));
    EXPECT_FALSE(parityCollectiveCapabilityFailure(config.collective, false, false));
}

/** @test CPU and CUDA/CPU retain their actual participant-derived backend isolation. */
TEST(ParityHardwareRequirements, CPUAndCUDAOnlyTopologiesNeverAcquireForeignVendorRequirements)
{
    const auto cpu = ParityHardwareRequirements::fromParticipants(
        std::array{GlobalDeviceAddress::cpu(0)});
    EXPECT_EQ(cpu.cuda_devices, 0); EXPECT_EQ(cpu.rocm_devices, 0);
    EXPECT_FALSE(parityHardwareInventoryFailure(cpu, {0,0}));
    const auto cuda = ParityHardwareRequirements::fromParticipants(
        std::array{GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cpu(0)});
    EXPECT_EQ(cuda.cuda_devices, 1); EXPECT_EQ(cuda.rocm_devices, 0);
    EXPECT_FALSE(parityHardwareInventoryFailure(cuda, {1,0}));
    EXPECT_TRUE(parityHardwareInventoryFailure(cuda, {0,8}));
}

/** @test Exact ordinals require the visible range, not just the number of participants. */
TEST(ParityHardwareRequirements, SparseOrdinalsAndRealMixedTopologyAreStrict)
{
    const auto required = ParityHardwareRequirements::fromParticipants(
        std::array{GlobalDeviceAddress::cuda(3), GlobalDeviceAddress::rocm(2), GlobalDeviceAddress::cpu(0)});
    EXPECT_EQ(required.cuda_devices, 4); EXPECT_EQ(required.rocm_devices, 3);
    EXPECT_FALSE(parityHardwareInventoryFailure(required, {4,3}));
    EXPECT_TRUE(parityHardwareInventoryFailure(required, {3,3}));
    EXPECT_TRUE(parityHardwareInventoryFailure(required, {4,2}));
    EXPECT_TRUE(parityHardwareInventoryFailure(required, {8,0}));
    EXPECT_TRUE(parityHardwareInventoryFailure(required, {0,8}));
}

/** @test The typed legacy adapter preserves sparse ordinal metadata end to end. */
TEST(ParityHardwareRequirements, TypedAdapterKeepsSparseOrdinalRatherThanFamilyCardinality)
{
    auto definition = qwen4ExpProductionDefinition(Qwen4ExpParityTopology::CPU);
    definition.topology.participants.front().address = GlobalDeviceAddress::cuda(3);
    const auto cells = expandModelParityDefinition(definition);
    ASSERT_FALSE(cells.empty());
    const auto config = cells.front().toTestConfig();
    ASSERT_EQ(config.devices.size(), 1u);
    EXPECT_EQ(config.devices.front(), ParityDeviceType::CUDA);
    ASSERT_TRUE(config.local_hardware_requirements);
    EXPECT_EQ(config.local_hardware_requirements->cuda_devices, 4);
    EXPECT_EQ(config.local_hardware_requirements->rocm_devices, 0);
    EXPECT_TRUE(parityHardwareInventoryFailure(*config.local_hardware_requirements, {1,0}));
}

/** @test Repeated logical consumers of one ordinal do not invent extra physical GPUs. */
TEST(ParityHardwareRequirements, SharedPhysicalOrdinalUsesItsExactVisibleRange)
{
    const auto required = ParityHardwareRequirements::fromParticipants(
        std::array{GlobalDeviceAddress::rocm(1), GlobalDeviceAddress::rocm(1), GlobalDeviceAddress::cpu(0)});
    EXPECT_EQ(required.cuda_devices, 0); EXPECT_EQ(required.rocm_devices, 2);
    EXPECT_FALSE(parityHardwareInventoryFailure(required, {0,2}));
}

/** @test Invalid or overflowing declarations cannot become permissive prerequisites. */
TEST(ParityHardwareRequirements, EmptyUnknownNegativeAndOverflowGeometryFailClosed)
{
    EXPECT_THROW((void)ParityHardwareRequirements::fromParticipants({}), std::invalid_argument);
    for (int ordinal : {-1, std::numeric_limits<int>::max()})
    {
        GlobalDeviceAddress address;
        address.device_type = DeviceType::ROCm; address.device_ordinal = ordinal;
        if (ordinal < 0)
            EXPECT_THROW((void)ParityHardwareRequirements::fromParticipants(std::array{address}), std::invalid_argument);
        else
            EXPECT_THROW((void)ParityHardwareRequirements::fromParticipants(std::array{address}), std::overflow_error);
    }
    GlobalDeviceAddress unknown;
    unknown.device_type = static_cast<DeviceType>(99);
    EXPECT_THROW((void)ParityHardwareRequirements::fromParticipants(std::array{unknown}), std::invalid_argument);
    EXPECT_THROW((void)parityHardwareInventoryFailure({-1,0},{0,0}), std::invalid_argument);
    EXPECT_THROW((void)parityHardwareInventoryFailure({0,0},{0,-1}), std::invalid_argument);
}

/** @test Removing the false heterogeneous vendor pair never relaxes NCCL/RCCL capability admission. */
TEST(ParityHardwareRequirements, CollectiveCapabilitiesRemainExactAndTotal)
{
    EXPECT_TRUE(parityCollectiveCapabilityFailure(Collective::NCCL, false, true));
    EXPECT_TRUE(parityCollectiveCapabilityFailure(Collective::RCCL, true, false));
    EXPECT_FALSE(parityCollectiveCapabilityFailure(Collective::NCCL, true, false));
    EXPECT_FALSE(parityCollectiveCapabilityFailure(Collective::RCCL, false, true));
    for (auto policy : {Collective::HETEROGENEOUS, Collective::HOST, Collective::MPI, Collective::None})
        EXPECT_FALSE(parityCollectiveCapabilityFailure(policy, false, false));
    EXPECT_TRUE(parityCollectiveCapabilityFailure(static_cast<Collective>(99), true, true));
}

/** @test Snapshot ownership follows every authored participant, never only the primary device. */
TEST(ParityHardwareRequirements, SnapshotServingEnvelopeSeesCPUFirstGPUAndCPUOnlyParticipants)
{
    for (const auto gpu : {ParityDeviceType::CUDA, ParityDeviceType::ROCm})
    {
        TestConfig mixed;
        mixed.devices = {ParityDeviceType::CPU, gpu};
        EXPECT_EQ(mixed.primary_device(), ParityDeviceType::CPU);
        ASSERT_TRUE(mixed.has_gpu_participant());
        EXPECT_EQ(paritySnapshotServingGraphRows(9, 4096, {64, 128, 512},
            mixed.has_gpu_participant(), {}), 512);
        std::reverse(mixed.devices.begin(), mixed.devices.end());
        EXPECT_TRUE(mixed.has_gpu_participant());
    }
    TestConfig cpu;
    cpu.devices = {ParityDeviceType::CPU, ParityDeviceType::CPU};
    EXPECT_FALSE(cpu.has_gpu_participant());
    EXPECT_EQ(paritySnapshotServingGraphRows(9, 4096, {64, 128, 512},
        cpu.has_gpu_participant(), {}), 9);
}
