/**
 * @file Test__MoEOverlayReturnValidation.cpp
 * @brief Captured CUDA/HIP return authentication rejects malformed peer packets.
 *
 * All four production return entrypoints share one block-cooperative validator.
 * This fixture keeps a retained graph and persistent mapped/device addresses,
 * changes only packet data between completed submissions, and proves both the
 * terminal status and absence of payload publication on every rejected replay.
 * Each exact case is a separate original MPI2 process lifetime: the observed
 * NUMA owner alone prepares its GPU/source/PMA; the idle original rank retains
 * setup/retirement participation. One General graph and all mapped/device roots
 * are admitted before capture. Data resets reuse topology; only terminal queries
 * expose results. These single-GPU component cases isolate descriptor validation,
 * while the original-rank peer suite owns peer arrival and anti-ABA progress.
 */
#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "fixtures/backends/OriginalRankLocalNativeReplay.h"
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "backends/IGPUGraphCapture.h"
#include "kernels/cuda/moe/CUDAMoEKernel.h"
#include "kernels/rocm/moe/ROCmMoEKernel.h"
#include "transfer/TransferEngine.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <chrono>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <memory>
#include <string>
#include <vector>

using namespace llaminar2;

namespace
{
    /** The four public captured return paths, independent of backend roles. */
    enum class ReturnPath { Single, SingleBatch, Multi, MultiBatch };

    /** One independent corruption per replay, plus good replays between faults. */
    enum class ReturnFault
    {
        None, Digest, Epoch, Stage, Layer, PayloadBytes, LiveEntries,
        LiveRows, PhysicalRows, OriginalNegative, OriginalDuplicate,
        OriginalOverflow, CompactNegative, CompactOverflow, GrantOrder,
    };

    /** Backend and exact production entrypoint for one captured proof. */
    struct ReturnCase
    {
        DeviceId device;
        ReturnPath path;
        const char *name;
    };

    /** Model-free packet regressions belong to the existing preflight target. */
    class MoEOverlayReturnValidation : public ::testing::TestWithParam<ReturnCase> {};

    /** @brief Reject setup/enqueue failures before releasing any original native owner. */
    void requireReturn(bool condition, const char *message)
    { if (!condition) throw std::runtime_error(message); }

    /** @brief Drain the exact original stream before exceptional graph/buffer destruction. */
    struct ReturnTerminalRetirement final
    {
        std::function<void()> retire;
        /** @brief Preserve the first diagnostic and fail fatally if native retirement is unprovable. */
        ~ReturnTerminalRetirement() noexcept
        {
            try { if (retire) retire(); }
            catch (const std::exception &error) {
                std::cerr << "Original return-validation retirement failed: " << error.what() << '\n';
                std::terminate();
            }
        }
    };

    /** @brief Replay one declared return entrypoint on its actual original NUMA-owned GPU. */
    void runOriginalReturnValidation(const ReturnCase &parameter)
    {
        using Graph = llaminar2::test::RetainedParticipantGraphExecutableFixtureAccess;
        const std::array devices{parameter.device};
        auto original = llaminar2::test::PreparedNativeLookupFixture::prepare(
            devices, MPIContextFactory::global()->rank());
        const bool single = parameter.path == ReturnPath::Single || parameter.path == ReturnPath::SingleBatch;
        const int rows = single ? 1 : 65;
        constexpr int top_k = 4;
        constexpr int width = 67; // Vector-copy tails and unaligned rows.
        const std::size_t capacity = static_cast<std::size_t>(rows) * top_k;
        const std::size_t entries = static_cast<std::size_t>(rows) * 2u;
        const std::size_t bytes = capacity * width * sizeof(float);
        // One immutable declaration feeds both BOM and materialization. These
        // roots are component protocol operands, not fabricated model tensors.
        const std::array mapped_bytes{
            sizeof(MoEOverlayActivationEpochControl), 7u * capacity * sizeof(int), bytes,
            sizeof(MoEOverlayActivationDeviceEpochGrant),
            sizeof(MoEOverlayActivationSingleRowReturnConsumeLaunch), bytes, bytes};
        const std::array device_bytes{sizeof(MoEOverlayActivationDeviceEpochGrant), bytes,
            sizeof(int), sizeof(MoEOverlayActivationSingleRowReturnConsumeLaunch)};
        PhysicalMemoryPlanBuilder plan;
        for (const auto size : mapped_bytes)
            plan.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
                TransferEngine::mappedHostRegionAllocationBytes(size));
        for (const auto size : device_bytes)
            plan.add(original->resource(parameter.device), PhysicalMemoryOwner::ActivationTransportStaging, size);
        const auto graph_bom = RetainedParticipantGraphExecutable::generalBOM(
            original->resource(DeviceId::cpu()), original->resource(parameter.device), 1);
        for (const auto &line : graph_bom.resources()) plan.add(line);
        original->admit(plan.build());
        {
            auto memory = original->memory();
            auto worker = original->permit(parameter.device).worker();
            // This is the worker-owned admitted native stream, not the
            // CUDA/HIP null or legacy runtime-default stream.
            void *const stream = worker->defaultStream();
            requireReturn(reinterpret_cast<std::uintptr_t>(stream) > 2u, "Original return validation lost its exact stream");
            TransferEngine engine;
            // Claims outlive native allocations/registrations; the terminal
            // guard below drains before graph and operand destruction.
            std::vector<PhysicalMemoryAllocationLease> claims;
            std::array<std::shared_ptr<MappedHostTransferRegion>, 7> mapped;
            std::array<std::shared_ptr<DeviceTransferBuffer>, 4> device;
            worker->submitAndWait([&] {
                for (std::size_t index = 0; index < mapped.size(); ++index) {
                    claims.push_back(memory->claimNewAllocation(DeviceId::cpu(),
                        PhysicalMemoryOwner::ActivationTransportStaging,
                        TransferEngine::mappedHostRegionAllocationBytes(mapped_bytes[index])));
                    mapped[index] = engine.allocateMappedHostRegion(mapped_bytes[index], devices);
                }
                for (std::size_t index = 0; index < device.size(); ++index) {
                    claims.push_back(memory->claimNewAllocation(parameter.device,
                        PhysicalMemoryOwner::ActivationTransportStaging, device_bytes[index]));
                    device[index] = engine.allocateDeviceTransferBuffer(device_bytes[index], parameter.device);
                }
            });
            const auto &control_memory = mapped[0], &metadata = mapped[1], &payload = mapped[2],
                &grant_source = mapped[3], &lane_source = mapped[4], &poison_source = mapped[5], &snapshot = mapped[6];
            const auto &grant_device = device[0], &output = device[1], &lane_valid = device[2], &lane_device = device[3];
            auto *const control = static_cast<MoEOverlayActivationEpochControl *>(control_memory->mutableHostData());
            auto *const mapped_control = static_cast<MoEOverlayActivationEpochControl *>(control_memory->deviceAlias(parameter.device));
            auto *const slots = static_cast<int *>(metadata->mutableHostData());
            auto *const mapped_slots = static_cast<int *>(metadata->deviceAlias(parameter.device));
            auto *const grant = static_cast<MoEOverlayActivationDeviceEpochGrant *>(grant_source->mutableHostData());
            auto *const source = static_cast<float *>(payload->mutableHostData());
            auto *const poison = static_cast<float *>(poison_source->mutableHostData());
            auto *const actual = static_cast<const float *>(snapshot->mutableHostData());
            const auto elements = capacity * width;
            std::vector<float> expected(elements);
            std::fill_n(poison, elements, -333.0f);
            std::unique_ptr<IMoEKernel> kernel;
            worker->submitAndWait([&] {
                kernel = parameter.device.is_cuda()
                    ? std::unique_ptr<IMoEKernel>(std::make_unique<CUDAMoEKernel>(parameter.device.ordinal))
                    : std::unique_ptr<IMoEKernel>(std::make_unique<ROCmMoEKernel>(parameter.device.ordinal));
            });
            auto graph = Graph::original(original->permit(parameter.device), stream);
            const auto await = [&] {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(
                    collective_timeout_policy::kDefaultCollectiveTimeoutMs);
                for (;;) {
                    GPUStreamExecutionState state = GPUStreamExecutionState::Pending;
                    worker->submitAndWait([&] { state = worker->queryStreamExecutionState(
                        stream, "original return-validation terminal"); });
                    if (state == GPUStreamExecutionState::Complete) break;
                    requireReturn(std::chrono::steady_clock::now() < deadline,
                        "Original return-validation terminal timed out");
                    std::this_thread::yield();
                }
            };
            ReturnTerminalRetirement retirement{[&] {
                await();
                requireReturn(Graph::observeTerminal(graph), "Original return-validation graph remained pending");
            }};
            constexpr std::uint32_t stage = 1u;
            constexpr int layer = 5;
            const auto bank = moeOverlayActivationBufferIndex(stage);
            const auto timeline = moeOverlayActivationLeasedTimelineValue(moeOverlayActivationBufferVisit(stage));
            const MoEOverlayActivationReturnConsumeLaunch packet{
                .dispatch = {
                    .row_ids = mapped_slots + 2u * capacity, .entry_offsets = mapped_slots + 3u * capacity,
                    .expert_ids = mapped_slots + 4u * capacity,
                    .route_weights = reinterpret_cast<float *>(mapped_slots + 5u * capacity),
                    .original_route_slots = mapped_slots, .compact_route_slots = mapped_slots + capacity,
                    .hidden_rows_fp32 = static_cast<float *>(payload->deviceAlias(parameter.device)),
                    .row_capacity = static_cast<std::size_t>(rows), .entry_capacity = capacity,
                    .d_model = width, .top_k = top_k},
                .returned = {static_cast<float *>(payload->deviceAlias(parameter.device)), capacity, width},
                .control = mapped_control,
                .grant = static_cast<MoEOverlayActivationDeviceEpochGrant *>(grant_device->mutableDeviceData()),
                .canonical_route_contributions_fp32 = static_cast<float *>(output->mutableDeviceData()),
                .physical_rows = rows, .stage_ordinal = stage, .model_layer_index = layer,
            };
            const MoEOverlayActivationSingleRowReturnConsumeLaunch single_packet{
                .packet = packet, .acquire = {&mapped_control->buffers[bank].return_signal.value, timeline},
            };
            if (single)
                *static_cast<MoEOverlayActivationSingleRowReturnConsumeLaunch *>(lane_source->mutableHostData()) = single_packet;
            else
                *static_cast<MoEOverlayActivationReturnConsumeLaunch *>(lane_source->mutableHostData()) = packet;
            worker->submitAndWait([&] {
                engine.enqueueMappedHostToPersistentDeviceRegion(*lane_source, 0u,
                    lane_device->mutableDeviceData(), sizeof(single_packet), 0u, sizeof(single_packet),
                    parameter.device, stream);
            });
            await(); // Cold immutable descriptor upload precedes native recording.
            worker->submitAndWait([&] {
                auto &definition = Graph::capture(graph);
                ScopedBackendGraphCapture capture(*worker, definition, "original return validation");
                requireReturn(capture.begin(), "Original return recording did not begin");
                switch (parameter.path)
                {
                case ReturnPath::Single:
                    requireReturn(kernel->consumeSingleRowMoEOverlayActivationReturn({.stream = stream}, single_packet), "Original single return recording failed");
                    break;
                case ReturnPath::SingleBatch:
                    requireReturn(kernel->consumeSingleRowMoEOverlayActivationReturnBatch({.stream = stream}, {
                        .lanes = static_cast<const MoEOverlayActivationSingleRowReturnConsumeLaunch *>(lane_device->deviceData()),
                        .lane_valid = static_cast<int *>(lane_valid->mutableDeviceData()),
                        .canonical_route_contributions_fp32 = packet.canonical_route_contributions_fp32,
                        .lane_count = 1u, .d_model = width, .top_k = top_k}), "Original single Batch return recording failed");
                    break;
                case ReturnPath::Multi:
                    requireReturn(kernel->consumeMoEOverlayActivationReturn({.stream = stream}, packet), "Original multi return recording failed");
                    break;
                case ReturnPath::MultiBatch:
                    requireReturn(kernel->consumeMultiRowMoEOverlayActivationReturnBatch({.stream = stream}, {
                        .lanes = static_cast<const MoEOverlayActivationReturnConsumeLaunch *>(lane_device->deviceData()),
                        .lane_valid = static_cast<int *>(lane_valid->mutableDeviceData()),
                        .canonical_route_contributions_fp32 = packet.canonical_route_contributions_fp32,
                        .lane_count = 1u, .physical_rows = rows, .d_model = width, .top_k = top_k}), "Original multi Batch return recording failed");
                    break;
                }
                capture.finish();
                requireReturn(definition.prepareRuntimeContextStorage(memory) && definition.instantiate(),
                    "Original return executable admission failed");
            });

            const std::array faults{
                ReturnFault::None, ReturnFault::Digest, ReturnFault::Epoch, ReturnFault::Stage,
                ReturnFault::Layer, ReturnFault::PayloadBytes, ReturnFault::LiveEntries,
                ReturnFault::LiveRows, ReturnFault::PhysicalRows, ReturnFault::OriginalNegative,
                ReturnFault::OriginalDuplicate, ReturnFault::OriginalOverflow,
                ReturnFault::CompactNegative, ReturnFault::CompactOverflow, ReturnFault::GrantOrder};
            std::size_t replay = 0;
            for (const auto fault : faults)
            {
                // Re-arm the same captured addresses after each terminal result.
                // A good replay after every corruption catches stale valid bits.
                for (const auto current : {fault, ReturnFault::None})
                {
                    SCOPED_TRACE(static_cast<int>(current));
                    ++replay;
                    *control = {};
                    std::fill_n(slots, 7u * capacity, 0);
                    // Alternate valid route ownership and payload bytes at the
                    // same retained addresses. Cached metadata must follow each
                    // new publication, not only repeat an unchanged first map.
                    for (std::size_t i = 0; i < elements; ++i)
                        source[i] = static_cast<float>(static_cast<int>(i % 113u) - 56 + static_cast<int>(replay));
                    for (std::size_t entry = 0; entry < entries; ++entry)
                    {
                        slots[entry] = static_cast<int>(2u * entry + (replay & 1u));
                        slots[capacity + entry] = static_cast<int>(entry);
                    }
                    *grant = {
                        .digest = {.low = 17u, .high = 23u}, .generation = 7u, .placement_epoch = 3u,
                        .live_rows = static_cast<std::uint64_t>(rows), .live_entries = entries, .stage_count = 2u,
                        .physical_rows = rows, .last_published_stage = 1, .last_consumed_stage = 0,
                        .endpoint = static_cast<std::uint32_t>(MoEOverlayActivationEndpoint::Continuation),
                        .state = static_cast<std::uint32_t>(MoEOverlayActivationEndpointState::Active),
                        .code = static_cast<std::uint32_t>(MoEOverlayActivationStatusCode::Success),
                        .graph_role = MoEOverlayInferenceGraphRole::MainPrefill,
                    };
                    auto &descriptor = control->buffers[bank].return_descriptor;
                    descriptor = {
                        .digest = grant->digest, .timeline = timeline, .placement_epoch = grant->placement_epoch,
                        .live_rows = grant->live_rows, .live_entries = entries,
                        .payload_bytes = moeOverlayReturnPayloadBytes(entries, width),
                        .stage_ordinal = stage, .model_layer_index = layer,
                    };
                    control->buffers[bank].return_signal.value = timeline;
                    switch (current)
                    {
                    case ReturnFault::None: break;
                    case ReturnFault::Digest: ++descriptor.digest.high; break;
                    case ReturnFault::Epoch: ++descriptor.placement_epoch; break;
                    case ReturnFault::Stage: ++descriptor.stage_ordinal; break;
                    case ReturnFault::Layer: ++descriptor.model_layer_index; break;
                    case ReturnFault::PayloadBytes: ++descriptor.payload_bytes; break;
                    case ReturnFault::LiveEntries: --descriptor.live_entries; break;
                    case ReturnFault::LiveRows: --descriptor.live_rows; break;
                    case ReturnFault::PhysicalRows: ++descriptor.live_rows; ++grant->live_rows; break;
                    case ReturnFault::OriginalNegative: slots[entries - 1u] = -1; break;
                    case ReturnFault::OriginalDuplicate: slots[entries - 1u] = slots[entries - 2u]; break;
                    case ReturnFault::OriginalOverflow: slots[entries - 1u] = static_cast<int>(capacity); break;
                    case ReturnFault::CompactNegative: slots[capacity + entries - 1u] = -1; break;
                    case ReturnFault::CompactOverflow: slots[capacity + entries - 1u] = static_cast<int>(capacity); break;
                    case ReturnFault::GrantOrder: ++grant->last_consumed_stage; break;
                    }
                    worker->submitAndWait([&] {
                        engine.enqueueMappedHostToPersistentDeviceRegion(*poison_source, 0u,
                            output->mutableDeviceData(), bytes, 0u, bytes, parameter.device, stream);
                        engine.enqueueMappedHostToPersistentDeviceRegion(*grant_source, 0u,
                            grant_device->mutableDeviceData(), sizeof(*grant), 0u, sizeof(*grant), parameter.device, stream);
                        requireReturn(Graph::submit(graph, replay), "Original return replay submission failed");
                        engine.enqueuePersistentDeviceRegionToMappedHost(output->deviceData(), bytes, 0u,
                            *snapshot, 0u, bytes, parameter.device, stream);
                    });
                    await();
                    requireReturn(Graph::observeTerminal(graph), "Original return replay did not publish its terminal");
                    std::copy_n(poison, elements, expected.data());
                    const bool accepted = current == ReturnFault::None;
                    if (accepted)
                        for (std::size_t entry = 0; entry < entries; ++entry)
                        {
                            const auto offset = static_cast<size_t>(slots[entry]) * width;
                            std::copy_n(source + offset, width, expected.data() + offset);
                        }
                    EXPECT_EQ(std::memcmp(actual, expected.data(), bytes), 0);
                    EXPECT_EQ(control->continuation_status.code, static_cast<std::uint32_t>(accepted
                        ? MoEOverlayActivationStatusCode::Success
                        : current == ReturnFault::GrantOrder ? MoEOverlayActivationStatusCode::OutOfOrder
                        : MoEOverlayActivationStatusCode::PayloadMismatch));
                }
            }
        }
        original->requirePayloadRetired();
        original->retire();
        original->requireExactRetired();
    }

    TEST_P(MoEOverlayReturnValidation, RetainedGraphRejectsCorruptionWithoutPayloadPublication)
    {
        const auto parameter = GetParam();
        llaminar2::test::originalRankLocalReplay(parameter.device.type, 1, [&](const auto &devices) {
            requireReturn(devices.size() == 1u, "Original return replay lost its single-GPU geometry");
            runOriginalReturnValidation({devices.front(), parameter.path, parameter.name});
        });
    }

    INSTANTIATE_TEST_SUITE_P(Backends, MoEOverlayReturnValidation, ::testing::Values(
        ReturnCase{DeviceId::cuda(0), ReturnPath::Single, "CUDASingle"},
        ReturnCase{DeviceId::cuda(0), ReturnPath::SingleBatch, "CUDASingleBatch"},
        ReturnCase{DeviceId::cuda(0), ReturnPath::Multi, "CUDAMulti"},
        ReturnCase{DeviceId::cuda(0), ReturnPath::MultiBatch, "CUDAMultiBatch"},
        ReturnCase{DeviceId::rocm(0), ReturnPath::Single, "ROCmSingle"},
        ReturnCase{DeviceId::rocm(0), ReturnPath::SingleBatch, "ROCmSingleBatch"},
        ReturnCase{DeviceId::rocm(0), ReturnPath::Multi, "ROCmMulti"},
        ReturnCase{DeviceId::rocm(0), ReturnPath::MultiBatch, "ROCmMultiBatch"}),
        [](const ::testing::TestParamInfo<ReturnCase> &info) { return info.param.name; });
}
