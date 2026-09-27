/**
 * @file Test__MoEOverlayReturnValidation.cpp
 * @brief Captured CUDA/HIP return authentication rejects malformed peer packets.
 *
 * All four production return entrypoints share one block-cooperative validator.
 * This fixture keeps a retained graph and persistent mapped/device addresses,
 * changes only packet data between completed submissions, and proves both the
 * terminal status and absence of payload publication on every rejected replay.
 * The full peer-arrival and anti-ABA lifetime proofs remain in the owning mapped
 * packet suite; this test isolates adversarial descriptor/route validation.
 */
#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "kernels/cuda/moe/CUDAMoEKernel.h"
#include "kernels/rocm/moe/ROCmMoEKernel.h"
#include "transfer/TransferEngine.h"

#include <algorithm>
#include <array>
#include <cstring>
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

    TEST_P(MoEOverlayReturnValidation, RetainedGraphRejectsCorruptionWithoutPayloadPublication)
    {
        const auto parameter = GetParam();
        auto *const backend = getBackendFor(parameter.device);
        ASSERT_NE(backend, nullptr);
        ASSERT_GT(backend->deviceCount(), parameter.device.gpu_ordinal());
        auto &context = GPUDeviceContextPool::instance().getContext(parameter.device);
        context.submitAndWait([&] {
            const bool single = parameter.path == ReturnPath::Single ||
                parameter.path == ReturnPath::SingleBatch;
            const int rows = single ? 1 : 65;
            constexpr int top_k = 4;
            constexpr int width = 67; // Vector-copy tails and unaligned rows.
            const std::size_t capacity = static_cast<std::size_t>(rows) * top_k;
            const std::size_t entries = static_cast<std::size_t>(rows) * 2u;
            const std::size_t bytes = capacity * width * sizeof(float);
            TransferEngine engine;
            const std::array devices{parameter.device};
            void *const stream = context.getOrCreateAuxiliaryStream("return_validation_regression");
            ASSERT_NE(stream, nullptr);
            auto control_memory = engine.allocateMappedHostRegion(
                sizeof(MoEOverlayActivationEpochControl), devices);
            auto metadata = engine.allocateMappedHostRegion(7u * capacity * sizeof(int), devices);
            auto payload = engine.allocateMappedHostRegion(bytes, devices);
            auto grant_source = engine.allocateMappedHostRegion(
                sizeof(MoEOverlayActivationDeviceEpochGrant), devices);
            auto grant_device = engine.allocateDeviceTransferBuffer(
                sizeof(MoEOverlayActivationDeviceEpochGrant), parameter.device);
            auto output = engine.allocateDeviceTransferBuffer(bytes, parameter.device);
            auto lane_valid = engine.allocateDeviceTransferBuffer(sizeof(int), parameter.device);
            auto lane_source = engine.allocateMappedHostRegion(
                sizeof(MoEOverlayActivationSingleRowReturnConsumeLaunch), devices);
            auto lane_device = engine.allocateDeviceTransferBuffer(
                sizeof(MoEOverlayActivationSingleRowReturnConsumeLaunch), parameter.device);

            auto *const control = static_cast<MoEOverlayActivationEpochControl *>(control_memory->mutableHostData());
            auto *const mapped_control = static_cast<MoEOverlayActivationEpochControl *>(control_memory->deviceAlias(parameter.device));
            auto *const slots = static_cast<int *>(metadata->mutableHostData());
            auto *const mapped_slots = static_cast<int *>(metadata->deviceAlias(parameter.device));
            auto *const grant = static_cast<MoEOverlayActivationDeviceEpochGrant *>(grant_source->mutableHostData());
            auto *const source = static_cast<float *>(payload->mutableHostData());
            std::vector<float> poison(capacity * width, -333.0f), actual(poison.size()), expected(poison.size());
            for (std::size_t i = 0; i < poison.size(); ++i)
                source[i] = static_cast<float>(static_cast<int>(i % 997u) - 498) * 0.125f;

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
            engine.enqueueMappedHostToPersistentDeviceRegion(*lane_source, 0u,
                lane_device->mutableDeviceData(), sizeof(single_packet), 0u, sizeof(single_packet),
                parameter.device, stream);
            std::unique_ptr<IMoEKernel> kernel = parameter.device.is_cuda()
                ? std::unique_ptr<IMoEKernel>(std::make_unique<CUDAMoEKernel>(parameter.device.ordinal))
                : std::unique_ptr<IMoEKernel>(std::make_unique<ROCmMoEKernel>(parameter.device.ordinal));
            auto graph = context.createGraphCapture(stream);
            // Only failure/terminal collection may block the host in this test.
            // Declare the drain after graph/buffers so it retires before them.
            auto drain = [&](void *) { (void)backend->synchronizeStream(stream, parameter.device.ordinal); };
            std::unique_ptr<void, decltype(drain)> terminal_guard(&context, drain);
            ASSERT_TRUE(graph->beginCapture());
            switch (parameter.path)
            {
            case ReturnPath::Single:
                ASSERT_TRUE(kernel->consumeSingleRowMoEOverlayActivationReturn({.stream = stream}, single_packet));
                break;
            case ReturnPath::SingleBatch:
                ASSERT_TRUE(kernel->consumeSingleRowMoEOverlayActivationReturnBatch({.stream = stream}, {
                    .lanes = static_cast<const MoEOverlayActivationSingleRowReturnConsumeLaunch *>(lane_device->deviceData()),
                    .lane_valid = static_cast<int *>(lane_valid->mutableDeviceData()),
                    .canonical_route_contributions_fp32 = packet.canonical_route_contributions_fp32,
                    .lane_count = 1u, .d_model = width, .top_k = top_k}));
                break;
            case ReturnPath::Multi:
                ASSERT_TRUE(kernel->consumeMoEOverlayActivationReturn({.stream = stream}, packet));
                break;
            case ReturnPath::MultiBatch:
                ASSERT_TRUE(kernel->consumeMultiRowMoEOverlayActivationReturnBatch({.stream = stream}, {
                    .lanes = static_cast<const MoEOverlayActivationReturnConsumeLaunch *>(lane_device->deviceData()),
                    .lane_valid = static_cast<int *>(lane_valid->mutableDeviceData()),
                    .canonical_route_contributions_fp32 = packet.canonical_route_contributions_fp32,
                    .lane_count = 1u, .physical_rows = rows, .d_model = width, .top_k = top_k}));
                break;
            }
            ASSERT_TRUE(graph->endCapture());
            ASSERT_TRUE(graph->instantiate());

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
                    for (std::size_t i = 0; i < poison.size(); ++i)
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
                    ASSERT_TRUE(backend->hostToDevice(output->mutableDeviceData(), poison.data(), bytes,
                        parameter.device.ordinal, stream));
                    engine.enqueueMappedHostToPersistentDeviceRegion(*grant_source, 0u,
                        grant_device->mutableDeviceData(), sizeof(*grant), 0u, sizeof(*grant), parameter.device, stream);
                    ASSERT_TRUE(graph->launchOnStream(stream));
                    ASSERT_TRUE(backend->synchronizeStream(stream, parameter.device.ordinal));
                    ASSERT_TRUE(backend->deviceToHost(actual.data(), output->deviceData(), bytes,
                        parameter.device.ordinal, stream));
                    expected = poison;
                    const bool accepted = current == ReturnFault::None;
                    if (accepted)
                        for (std::size_t entry = 0; entry < entries; ++entry)
                        {
                            const auto offset = static_cast<size_t>(slots[entry]) * width;
                            std::copy_n(source + offset, width, expected.data() + offset);
                        }
                    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), bytes), 0);
                    EXPECT_EQ(control->continuation_status.code, static_cast<std::uint32_t>(accepted
                        ? MoEOverlayActivationStatusCode::Success
                        : current == ReturnFault::GrantOrder ? MoEOverlayActivationStatusCode::OutOfOrder
                        : MoEOverlayActivationStatusCode::PayloadMismatch));
                }
            }
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
