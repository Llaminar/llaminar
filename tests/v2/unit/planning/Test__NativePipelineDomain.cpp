/**
 * @file Test__NativePipelineDomain.cpp
 * @brief Device-free replay of actual native pipeline extent, rank ownership and typed frame BOM.
 *
 * The same production row projection and completed native receipt validator run
 * through empty, partial, reset and stale work. Arbitrary unequal stage widths
 * exercise the actual rank-plan memory projection without loading a model/GPU.
 */
#include "collective/NativePipelineByteFrame.h"
#include "kernels/common/SamplingMath.h"
#include "kernels/qwen4exp/QwenFixedHeadPipelineCounters.h"
#include "execution/local_execution/orchestrators/PipelineNativeOwnership.h"
#include "execution/local_execution/orchestrators/PipelineNativeDomain.h"
#include "planning/RankMemoryPlanInputs.h"
#include "planning/PipelineTransferMemory.h"
#include "mocks/MockMPIContext.h"
#include "mocks/MockLocalTPContext.h"
#include <gtest/gtest.h>
#include <numeric>
#include <array>
using namespace llaminar2;

/** @test Composition retains the exact address authority for arbitrary native domain widths. */
TEST(NativePipelineDomain, CompositionPreservesOriginalHostAndNUMA)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (const int count : {2, 3, 4, 7, 8, 17}) {
            test::MockLocalTPContext original;
            std::vector<GlobalDeviceAddress> members;
            for (int member = 0; member < count; ++member)
                members.push_back(GlobalDeviceAddress::fromLocalDeviceId(
                    DeviceId(backend, 3 * member + 2), "original-owner.example", 5));
            original.setDevices(members);
            original.setBackend(backend == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
            std::vector<GlobalDeviceAddress> composed;
            for (int member = 0; member < count; ++member) {
                const PipelineNativeDomain domain(members[member].toLocalDeviceId(), &original, member);
                composed.push_back(domain.memberAddress());
                EXPECT_NE(composed.back(), GlobalDeviceAddress::fromLocalDeviceId(members[member].toLocalDeviceId()));
            }
            EXPECT_EQ(composed, members);
            const PipelineNativeDomain retained(members.front().toLocalDeviceId(), &original, 0);
            members.front().numa_node = 6;
            original.setDevices(members);
            EXPECT_THROW((void)retained.memberAddress(), std::logic_error);
        }
    const PipelineNativeDomain singleton(DeviceId::rocm(9), nullptr, 0);
    EXPECT_EQ(singleton.memberAddress(), GlobalDeviceAddress::rocm(9));
}

namespace {
/** @brief Original published test transport; no hardware/context reconstruction occurs in the production body. */
class PipelineInventoryMPI final : public test::MockMPIContext {
public:
    std::shared_ptr<ClusterInventory> observed;
    /** @brief Retain one exact rank-indexed immutable publication for this test role. */
    PipelineInventoryMPI(int rank, std::shared_ptr<ClusterInventory> inventory)
        : MockMPIContext(rank, inventory->world_size), observed(std::move(inventory)) {}
    /** @return The original test publication under this same context identity. */
    std::shared_ptr<const ClusterInventory> clusterInventory() const override { return observed; }
};
}
/** @test Actual shared endpoint body retains sequence/error authority across empty/shrinking/reset replay. */
TEST(NativePipelineDomain, ExactRowsAndCompletedReceiptsReplay)
{
    const NativeCountedAllGatherLayout layout(1);
    std::vector<std::uint64_t> storage((layout.bytes() + 7) / 8, 0);
    auto *frame = reinterpret_cast<NativeCountedAllGatherFrame *>(storage.data());
    auto *extent = reinterpret_cast<std::uint64_t *>(reinterpret_cast<std::byte *>(storage.data()) + layout.extents());
    auto *operand = reinterpret_cast<ncclLlaminarCompletionOperand *>(reinterpret_cast<std::byte *>(storage.data()) + layout.operands());
    auto *receipt = reinterpret_cast<ncclLlaminarByteCompletion *>(reinterpret_cast<std::byte *>(storage.data()) + layout.receipts());
    std::uint32_t abort = 0; int original = 0; std::int32_t rows = 0;
    const ncclLlaminarCompletionAuthority authority{&abort, &original};
    NativePipelineByteInvocation invocation{storage.data(), 0x4143540000000010ULL, 16 * 12, &rows, 12, authority};
    ASSERT_TRUE(invocation.valid());
    for (std::uint64_t replay = 1; replay <= 20; ++replay) {
        rows = std::array<std::int32_t, 8>{16, 0, 1, 7, 0, 2, 3, 0}[replay % 8];
        ASSERT_TRUE(prepareNativePipelineByteInvocation(invocation));
        EXPECT_EQ(frame->sequence, replay);
        EXPECT_EQ(*extent, std::uint64_t(rows) * 12);
        *receipt = {frame->table, frame->admission, frame->sequence, *extent, authority.abortFlag,
            authority.deviceCommunicator, operand->message, ncclLlaminarCompletionPhase::Completed, 0};
        EXPECT_TRUE(completeNativePipelineByteInvocation(invocation, 0));
    }
    ASSERT_TRUE(prepareNativePipelineByteInvocation(invocation));
    EXPECT_FALSE(completeNativePipelineByteInvocation(invocation, 0)); // Actual previous receipt is stale.
    EXPECT_NE(frame->first_error, 0);
    EXPECT_FALSE(prepareNativePipelineByteInvocation(invocation));
    for (const int fault : {0, 1, 2, 3}) {
        std::fill(storage.begin(), storage.end(), 0); rows = fault == 0 ? -1 : fault == 1 ? 17 : 2;
        if (fault < 2) { EXPECT_FALSE(prepareNativePipelineByteInvocation(invocation)); continue; }
        ASSERT_TRUE(prepareNativePipelineByteInvocation(invocation));
        *receipt = {frame->table, frame->admission, frame->sequence, *extent, authority.abortFlag,
            authority.deviceCommunicator, operand->message, ncclLlaminarCompletionPhase::Completed, 0};
        if (fault == 2) { *extent = 12; receipt->bytes = 12; } // Complete native receipt, incompatible original local rows.
        EXPECT_FALSE(completeNativePipelineByteInvocation(invocation, fault == 3 ? 1 : 0));
        EXPECT_FALSE(prepareNativePipelineByteInvocation(invocation));
    }
    std::fill(storage.begin(), storage.end(), 0);
    invocation.rows = nullptr; invocation.row_bytes = 0; invocation.capacity = 4 * sizeof(std::int32_t);
    ASSERT_TRUE(prepareNativePipelineByteInvocation(invocation));
    EXPECT_EQ(*extent, 16u); // Fixed semantic metadata never sends allocation capacity.
}
/** @test Only a fresh native producer receipt can publish storage-only transport rows. */
TEST(NativePipelineDomain, ProducerPublicationRowsReplay)
{
    const NativeCountedAllGatherLayout layout(1);
    std::vector<std::uint64_t> storage((layout.bytes() + 7) / 8, 0);
    auto *frame = reinterpret_cast<NativeCountedAllGatherFrame *>(storage.data());
    auto *extent = reinterpret_cast<std::uint64_t *>(reinterpret_cast<std::byte *>(storage.data()) + layout.extents());
    auto *operand = reinterpret_cast<ncclLlaminarCompletionOperand *>(reinterpret_cast<std::byte *>(storage.data()) + layout.operands());
    auto *receipt = reinterpret_cast<ncclLlaminarByteCompletion *>(reinterpret_cast<std::byte *>(storage.data()) + layout.receipts());
    std::uint32_t abort = 0;
    int original = 0;
    std::int32_t published = 99;
    alignas(std::uint64_t) std::array<std::int32_t, sampling_math::kDeviceGenerationControlCount> control{};
    using Metric = qwen4exp::FixedHeadPipelineTerminalMetric;
    const qwen4exp::FixedHeadPipelineCounterRow observations{control.data()};
    const auto completed = observations.metric(Metric::CompletedReceipts);
    const ncclLlaminarCompletionAuthority authority{&abort, &original};
    const NativePipelineByteInvocation invocation{
        storage.data(), 0x51464e0000000001ULL, 31 * 16, nullptr, 16, authority, &published, {nullptr, completed}};
    ASSERT_TRUE(invocation.valid());
    auto conflicting = invocation;
    conflicting.rows = &published;
    EXPECT_FALSE(conflicting.valid());
    auto oversized = invocation;
    oversized.capacity = (std::uint64_t{std::numeric_limits<std::int32_t>::max()} + 1) * oversized.row_bytes;
    EXPECT_FALSE(oversized.valid());
    const auto complete = [&](std::uint64_t received) {
        *extent = received; // The native fixed header writes the receiver extent before payload.
        *receipt = {frame->table, frame->admission, frame->sequence, received,
            authority.abortFlag, authority.deviceCommunicator, operand->message,
            ncclLlaminarCompletionPhase::Completed, 0};
    };
    for (std::uint64_t replay = 1; replay <= 20; ++replay) {
        const auto expected = std::array<std::int32_t, 5>{0, 31, 1, 0, 7}[replay % 5];
        ASSERT_TRUE(prepareNativePipelineByteInvocation(invocation));
        EXPECT_EQ(published, -1);
        EXPECT_EQ(*extent, 0u);
        EXPECT_EQ(frame->sequence, replay);
        complete(std::uint64_t(expected) * 16);
        ASSERT_TRUE(completeNativePipelineByteInvocation(invocation, 0));
        EXPECT_EQ(published, expected);
        EXPECT_EQ(readNativeCounter64(completed.words), replay); // Empty is a fresh metadata completion too.
    }
    ASSERT_TRUE(prepareNativePipelineByteInvocation(invocation));
    EXPECT_FALSE(completeNativePipelineByteInvocation(invocation, 0)); // Previous success is stale.
    EXPECT_EQ(published, -1);
    EXPECT_EQ(readNativeCounter64(completed.words), 20u);
    EXPECT_FALSE(prepareNativePipelineByteInvocation(invocation));
    for (const auto received : {std::uint64_t{17}, invocation.capacity + 16, std::uint64_t{16}}) {
        std::fill(storage.begin(), storage.end(), 0);
        ASSERT_TRUE(prepareNativePipelineByteInvocation(invocation));
        complete(received);
        EXPECT_FALSE(completeNativePipelineByteInvocation(invocation, received == 16));
        EXPECT_EQ(published, -1);
        EXPECT_NE(frame->first_error, 0);
        EXPECT_EQ(readNativeCounter64(completed.words), 20u);
    }

    using namespace sampling_math;
    const auto policy = DeviceGenerationPolicy::ordinary();
    const DeviceGenerationInitialization preserve{.forward_failure_policy =
        DeviceGenerationForwardFailurePolicy::PreserveUntilRequestReset};
    ASSERT_TRUE(initialize_device_generation_control(1, 1, policy, control.data(),
        DeviceGenerationLeadingRowDisposition::PendingResponse, preserve));
    EXPECT_EQ(readNativeCounter64(completed.words), 20u); // Initial admission retains real prefill work.
    std::array<std::int32_t, 1> response{};
    ASSERT_TRUE(append_ordinary_sample_to_device_generation(7, false, OrdinaryGenerationSampleSource::PrefillLogits,
        response.data(), 1, control.data()));
    DeviceGenerationDispatchTicket prior;
    ASSERT_TRUE(initialize_device_generation_dispatch_ticket(2, 3, &prior));
    ASSERT_TRUE(publish_device_generation_dispatch_ticket(control.data(), nullptr, &prior));
    const DeviceGenerationInitialization continuation{.kind = DeviceGenerationAdmissionKind::ContinueResponse,
        .session_epoch = 2, .workspace_generation = 3, .prior_tickets = &prior,
        .forward_failure_policy = DeviceGenerationForwardFailurePolicy::PreserveUntilRequestReset};
    ASSERT_TRUE(initialize_device_generation_control(1, 1, policy, control.data(),
        DeviceGenerationLeadingRowDisposition::AlreadyEmitted, continuation));
    EXPECT_EQ(readNativeCounter64(completed.words), 0u); // The prior terminal window cannot be counted twice.
}
/** @test Actual stage projection prices unequal positive domains and every adjacent edge without a participant cap. */
TEST(NativePipelineDomain, UnequalStageEndpointMemoryHasNoParticipantCap)
{
    for (const auto widths : {std::vector<std::size_t>{1, 3, 2}, {5, 1, 7}, {9, 2}, {2, 1, 4, 3}})
        for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm}) {
            RankExecutionPlan plan;
            plan.rank = 0;
            std::size_t ordinal = 0;
            for (std::size_t stage = 0; stage < widths.size(); ++stage) {
                plan.local_pp_layer_boundaries.push_back(static_cast<int>(stage));
                RankExecutionPlan::LocalPPStageTPInfo domain;
                for (std::size_t member = 0; member < widths[stage]; ++member)
                    domain.devices.push_back(GlobalDeviceAddress::fromLocalDeviceId(DeviceId(backend, ordinal++), "node", 1));
                plan.local_pp_devices.push_back(domain.devices.front());
                plan.local_pp_stage_tp_info.push_back(std::move(domain));
            }
            plan.local_pp_layer_boundaries.push_back(static_cast<int>(widths.size()));
            std::size_t quoted = 0, expected_total = 0;
            for (std::size_t stage = 0; stage < widths.size(); ++stage) {
                const auto projection = PipelineStageTransferMemory::forStage(plan, stage);
                for (std::size_t member = 0; member < widths[stage]; ++member) {
                    DevicePlanConfig config;
                    config.world_rank = 0; config.first_layer = stage; config.last_layer = stage;
                    config.device = plan.local_pp_stage_tp_info[stage].devices[member].toLocalDeviceId();
                    config.execution_role = DeviceExecutionMemoryRole::ContinuationGraph;
                    projection.bind(config);
                    const auto expected = std::size_t(stage > 0) + std::size_t(stage + 1 < widths.size()) +
                        (member ? 0 : (stage ? widths[stage - 1] : 0) + (stage + 1 < widths.size() ? widths[stage + 1] : 0));
                    EXPECT_EQ(config.native_pipeline_byte_endpoints, expected);
                    EXPECT_TRUE(config.captured_pipeline_boundaries.empty());
                    EXPECT_FALSE(config.pipeline_native_activation_fanout);
                    EXPECT_EQ(config.local_pipeline_backend, backend == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
                    EXPECT_FALSE(projection.requiresHostMemory(config.device));
                    quoted += config.native_pipeline_byte_endpoints;
                }
                if (stage + 1 < widths.size()) expected_total += 2 * (widths[stage] + widths[stage + 1]);
            }
            EXPECT_EQ(quoted, expected_total);
        }
    const auto geometry = PipelineTransferMemory::nativeForRows(13, 17, 4);
    EXPECT_EQ(geometry.activation_capacity_bytes, 13u * 17 * sizeof(float));
    EXPECT_EQ(geometry.metadata_capacity_bytes, 4u * sizeof(std::int32_t));
    EXPECT_EQ(geometry.host_bytes_per_boundary, 0u);
    EXPECT_EQ(geometry.device_bytes_per_boundary, 0u); // Endpoint frames have their own exact typed incidence BOM.
}
/** @test Original MPI rank, NUMA and prepared physical UUID remain mandatory for every member. */
TEST(NativePipelineDomain, OriginalRankNumaAndPreparedUuidCannotBeAliased)
{
    auto inventory = std::make_shared<ClusterInventory>(); inventory->world_size = 2; inventory->ranks.resize(2);
    std::vector<DeviceId> devices; std::vector<std::string> uuids;
    for (int rank = 0; rank < 2; ++rank) {
        auto &observed = inventory->ranks[rank]; observed.rank = rank; observed.cpu.numa_node = rank;
        for (int gpu = 0; gpu < 11; ++gpu) {
            DeviceInfo device; device.type = DeviceType::ROCm; device.local_device_id = gpu;
            device.numa_node = 1; device.uuid = "actual-recorded-uuid-" + std::to_string(gpu);
            observed.gpus.push_back(device);
        }
    }
    for (int gpu = 0; gpu < 11; ++gpu) { devices.push_back(DeviceId::rocm(gpu)); uuids.push_back(inventory->ranks[1].gpus[gpu].uuid); }
    PipelineInventoryMPI owner(1, inventory), remote(0, inventory);
    EXPECT_NO_THROW(requireOriginalNativePipelineOwnership(owner, 1, devices, uuids));
    EXPECT_THROW(requireOriginalNativePipelineOwnership(remote, 0, devices, uuids), std::invalid_argument);
    EXPECT_THROW(requireOriginalNativePipelineOwnership(owner, 0, devices, uuids), std::invalid_argument);
    uuids[7] += "-foreign-birth";
    EXPECT_THROW(requireOriginalNativePipelineOwnership(owner, 1, devices, uuids), std::invalid_argument);
    uuids[7] = inventory->ranks[1].gpus[7].uuid;
    inventory->ranks[1].gpus[10].numa_node = 0;
    EXPECT_THROW(requireOriginalNativePipelineOwnership(owner, 1, devices, uuids), std::invalid_argument);
    inventory->ranks[1].gpus[10].numa_node = 1;
    inventory->ranks[1].cpu.numa_node = -1;
    EXPECT_THROW(requireOriginalNativePipelineOwnership(owner, 1, devices, uuids), std::invalid_argument);
}
