/**
 * @file Perf__MoECompactIntermediate.cpp
 * @brief Complete pack/exchange/import A/B for padded native and counted MoE packets.
 *
 * Both candidates preserve the exact existing intermediate bits and original
 * route identity. Timing includes pack, all peer traffic and import, not input
 * upload or terminal validation. The counted candidate is an explicit probe,
 * not an installed production collective policy. In particular, these results
 * cannot authorize bypassing an enabled native GPU P2P path.
 */
#include <gtest/gtest.h>
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "collective/LocalTPContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/common/MoEGroupedIntermediateExchangeKernels.h"
#include "planning/PlanningObservedResource.h"
#include "transfer/CapturedTransferChannel.h"
#include "utils/MPIContext.h"
#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <exception>
#include <optional>
#include <thread>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Explicit experimental exchange contract, never selected on failure. */
    enum class Exchange { NativePadded, DeviceCounted };

    /** @brief Abort the probe immediately on an invalid operation. */
    void require(bool value, const char *message)
    { if (!value) throw std::runtime_error(message); }

    /** @brief Dispatch one representation-preserving producer. */
    bool pack(DeviceId device, const MoEGroupedIntermediatePackLaunch &fixed,
        const MoECompactIntermediatePackLaunch &compact, Exchange mode, void *stream)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda()) return mode == Exchange::NativePadded ? cuda::packGroupedIntermediate(fixed, stream) : cuda::packCompactIntermediate(compact, stream);
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) return mode == Exchange::NativePadded ? rocm::packGroupedIntermediate(fixed, stream) : rocm::packCompactIntermediate(compact, stream);
#endif
        return false;
    }

    /** @brief Dispatch one fixed receive-bank importer. */
    bool importFixed(DeviceId device, const MoEGroupedIntermediateConsumeLaunch &launch, void *stream)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda()) return cuda::consumeGroupedIntermediate(launch, stream);
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) return rocm::consumeGroupedIntermediate(launch, stream);
#endif
        return false;
    }

    /** @brief Dispatch one live-extent importer, with no arithmetic conversion. */
    bool importCompact(DeviceId device, const MoECompactIntermediateConsumeLaunch &launch, void *stream)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda()) return cuda::consumeCompactIntermediate(launch, stream);
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) return rocm::consumeCompactIntermediate(launch, stream);
#endif
        return false;
    }

    /** @return Stable arbitrary bits including FP special encodings; no precision-dependent oracle. */
    std::uint32_t word(std::uint32_t route, size_t column)
    { return static_cast<std::uint32_t>((route + 1u) * 0x9e3779b9u ^ column * 0x85ebca6bu); }

    /** @brief One participant's admitted physical owners and retained graph. */
    struct Participant final
    {
        DeviceId device;
        IBackend *backend;
        IWorkerGPUContext *worker;
        void *stream;
        std::unique_ptr<DeviceWorkspaceManager> workspace;
        std::optional<CapturedTransferBinding> send, receive;
        PhysicalMemoryOwnerReservation graph_claim;
        std::unique_ptr<IGPUGraphCapture> graph;
        void *begin = nullptr, *end = nullptr;

        /** @brief Retire only after the outer timing/validation terminal joins. */
        ~Participant()
        {
            graph.reset();
            if (end) backend->destroyEvent(end, device.ordinal);
            if (begin) backend->destroyEvent(begin, device.ordinal);
        }
        /** @return Typed address in this admitted, immutable named workspace. */
        template<class T> T *buffer(const char *name) const
        { return static_cast<T *>(workspace->getBuffer(name)); }
    };

    /**
     * @brief Measure a complete two-device exchange with one captured graph per participant.
     * @param first First endpoint of a same-vendor two-GPU domain.
     * @param mode Explicit native or device-counted experiment.
     * @param layout Immutable packet geometry shared by both candidates.
     * @param live_routes Real active routes inside the captured bucket capacity.
     * @param first_share Number of each eight routes owned by the first participant (0..8).
     * @return Median maximum-participant GPU interval, in microseconds.
     */
    double measure(DeviceId first, Exchange mode, MoEGroupedIntermediateLayout layout,
        std::uint32_t live_routes, unsigned first_share)
    {
        require(layout.valid() && layout.compactValid() && layout.participants == 2 &&
            live_routes <= layout.route_capacity && first_share <= 8, "Invalid compact economy geometry");
        const std::array devices{first, first.is_cuda() ? DeviceId::cuda(first.ordinal + 1) : DeviceId::rocm(first.ordinal + 1)};
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        require(bool(inventory), "Compact economy needs canonical inventory");
        const auto &rank = inventory->ranks.at(mpi->rank());
        const bool compact = mode == Exchange::DeviceCounted;
        const auto packet_bytes = compact ? layout.compactCapacityBytes() : layout.packetBytes();
        const auto values_bytes = layout.route_capacity * layout.valueWords() * sizeof(std::uint32_t);
        const auto scales_bytes = layout.route_capacity * layout.scaleWords() * sizeof(std::uint32_t);
        WorkspaceRequirements requirements;
        requirements.buffers = {
            {"owners", layout.route_capacity * sizeof(std::int32_t)}, {"local_map", layout.route_capacity * sizeof(std::int32_t)},
            {"all_map", layout.route_capacity * sizeof(std::int32_t)}, {"group_end", 2 * sizeof(std::int32_t)},
            {"counts", 2 * sizeof(std::uint64_t)}, {"values", values_bytes}, {"output", values_bytes},
            {"packet", packet_bytes}, {"received", packet_bytes * (compact ? 1 : 2)}};
        if (scales_bytes) { requirements.buffers.emplace_back("scales", scales_bytes); requirements.buffers.emplace_back("output_scales", scales_bytes); }
        const auto workspace_bytes = requirements.total_bytes_with_alignment();
        const auto channel_geometry = CapturedTransferChannel::memoryFor(packet_bytes);
        PhysicalMemoryPlanBuilder builder;
        if (compact) builder.add(planningObservedResource(rank, DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
            2 * channel_geometry.mapped_host_bytes);
        for (const auto device : devices)
        {
            const auto resource = planningObservedResource(rank, device);
            builder.add(resource, PhysicalMemoryOwner::ExecutionWorkspace, workspace_bytes);
            builder.add(resource, PhysicalMemoryOwner::NativeGraphExecutable, GPUGraphMemoryContract::reservationBytesPerExecutable(device));
            if (compact) builder.add(resource, PhysicalMemoryOwner::ActivationTransportStaging, 2 * channel_geometry.cursor_bytes_per_device);
        }
        auto authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(builder.build()), rank.rank);
        // Communicators outlive their captured participants. RCCL/NCCL graph
        // retirement must finish before destroying the associated TP domain.
        std::shared_ptr<ILocalTPContext> native;
        std::array<std::unique_ptr<Participant>, 2> participants;
        std::vector<std::int32_t> owners(layout.route_capacity, -1), all_map(layout.route_capacity, -1);
        for (std::uint32_t route = 0; route < live_routes; ++route)
        { owners[route] = route % 8 < first_share ? 0 : 1; all_map[route] = route; }
        for (size_t i = 0; i < 2; ++i)
        {
            auto p = std::make_unique<Participant>();
            p->device = devices[i]; p->backend = getBackendFor(devices[i]);
            require(p->backend && p->backend->deviceCount() > devices[i].ordinal, "Compact economy needs two real GPUs");
            p->worker = &GPUDeviceContextPool::instance().getContext(devices[i]);
            p->worker->submitAndWait([&] {
                p->stream = p->worker->getOrCreateAuxiliaryStream("compact_moe_economy");
                p->workspace = std::make_unique<DeviceWorkspaceManager>(devices[i], workspace_bytes, authority);
                require(p->workspace->allocate(requirements), "Compact economy workspace admission");
                p->begin = p->backend->createTimingEvent(devices[i].ordinal);
                p->end = p->backend->createTimingEvent(devices[i].ordinal);
                require(p->begin && p->end, "Compact economy timing events");
                std::vector<std::int32_t> local_map(layout.route_capacity, -1);
                std::vector<std::uint32_t> values(values_bytes / 4, 0), scales(scales_bytes / 4, 0);
                int count = 0;
                for (std::uint32_t route = 0; route < live_routes; ++route)
                    if (owners[route] == static_cast<int>(i))
                    {
                        local_map[route] = count;
                        for (size_t j = 0; j < layout.valueWords(); ++j) values[count * layout.valueWords() + j] = word(route, j);
                        for (size_t j = 0; j < layout.scaleWords(); ++j) scales[count * layout.scaleWords() + j] = word(route, layout.valueWords() + j);
                        ++count;
                    }
                const std::array group_end{count / 2, count - count / 2};
                const auto upload = [&](const char *name, const void *data, size_t bytes) {
                    require(p->backend->hostToDeviceOnStream(p->buffer<void>(name), data, bytes, devices[i].ordinal, p->stream), "Compact economy setup upload");
                };
                upload("owners", owners.data(), owners.size() * sizeof(owners[0]));
                upload("local_map", local_map.data(), local_map.size() * sizeof(local_map[0]));
                upload("all_map", all_map.data(), all_map.size() * sizeof(all_map[0]));
                upload("group_end", group_end.data(), sizeof(group_end));
                upload("values", values.data(), values_bytes);
                if (scales_bytes) upload("scales", scales.data(), scales_bytes);
                require(p->backend->memset(p->buffer<void>("output"), 0xa5, values_bytes, devices[i].ordinal, p->stream), "Compact output sentinel");
                if (scales_bytes) require(p->backend->memset(p->buffer<void>("output_scales"), 0xa5, scales_bytes, devices[i].ordinal, p->stream), "Compact scale sentinel");
                require(p->backend->recordEvent(p->end, devices[i].ordinal, p->stream) && p->backend->waitForEvent(p->end, devices[i].ordinal), "Compact setup terminal");
            });
            participants[i] = std::move(p);
        }
        auto &transfer = TransferEngine::instance();
        if (compact)
            for (size_t producer = 0; producer < 2; ++producer)
            {
                auto &a = *participants[producer]; auto &b = *participants[1 - producer];
                auto channel = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), a.device, a.stream, b.device, b.stream, packet_bytes);
                a.send = transfer.bindDeviceCountedTransfer(transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Producer,
                    {73 + producer, packet_bytes}, a.workspace->retainBuffer("packet", 0, packet_bytes)),
                    a.workspace->retainBuffer("counts", 0, 16));
                b.receive = transfer.bindDeviceCountedTransfer(transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Consumer,
                    {73 + producer, packet_bytes}, b.workspace->retainBuffer("received", 0, packet_bytes)),
                    b.workspace->retainBuffer("counts", 0, 16), 8);
            }
        else
            native = createLocalTPContext({GlobalDeviceAddress::fromLocalDeviceId(devices[0]), GlobalDeviceAddress::fromLocalDeviceId(devices[1])}, {},
                first.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        require(compact || bool(native), "Missing native economy communicator");
        constexpr int warmups = 5, samples = 31;
        std::array<std::array<double, samples>, 2> timings{};
        std::array<std::exception_ptr, 2> errors{};
        std::array<std::thread, 2> threads;
        std::barrier rendezvous(2);
        for (size_t i = 0; i < 2; ++i)
            threads[i] = std::thread([&, i] {
                try
                {
                    auto &p = *participants[i];
                    p.worker->submitAndWait([&] {
                        const MoEGroupedIntermediatePackLaunch fixed{.layout = layout, .route_owners = p.buffer<std::int32_t>("owners"),
                            .original_to_grouped = p.buffer<std::int32_t>("local_map"), .grouped_values = p.buffer<void>("values"),
                            .grouped_scales = scales_bytes ? p.buffer<float>("scales") : nullptr, .packet = p.buffer<std::uint32_t>("packet"),
                            .participant = static_cast<std::int32_t>(i)};
                        const MoECompactIntermediatePackLaunch compact_pack{fixed, p.buffer<std::int32_t>("group_end"),
                            p.buffer<std::int32_t>("group_end") + 1, p.buffer<std::uint64_t>("counts")};
                        p.graph_claim = authority->reserveNewAllocations(p.device, PhysicalMemoryOwner::NativeGraphExecutable,
                            GPUGraphMemoryContract::reservationBytesPerExecutable(p.device));
                        p.graph = p.worker->createGraphCapture(p.stream);
                        require(bool(p.graph), "Compact economy graph factory");
                        rendezvous.arrive_and_wait();
                        ScopedBackendGraphCapture capture(*p.worker, *p.graph, "compact MoE exchange economy");
                        require(capture.begin(), "Compact economy capture begin");
                        require(pack(p.device, fixed, compact_pack, mode, p.stream), "Compact economy pack");
                        if (compact)
                        {
                            transfer.enqueueCapturedTransfer(*p.send, p.stream);
                            transfer.enqueueCapturedTransfer(*p.receive, p.stream);
                            for (size_t source = 0; source < 2; ++source)
                            {
                                const bool local = source == i;
                                const MoECompactIntermediateConsumeLaunch consume{.layout = layout,
                                    .route_owners = p.buffer<std::int32_t>("owners"), .original_to_grouped = p.buffer<std::int32_t>("all_map"),
                                    .packet = p.buffer<std::uint32_t>(local ? "packet" : "received"),
                                    .packet_bytes = p.buffer<std::uint64_t>("counts") + (local ? 0 : 1), .participant = static_cast<std::int32_t>(source),
                                    .grouped_values = p.buffer<void>("output"), .grouped_scales = scales_bytes ? p.buffer<float>("output_scales") : nullptr};
                                require(importCompact(p.device, consume, p.stream), "Compact economy import");
                            }
                        }
                        else
                        {
                            require(native->allgatherRawOnStream(p.buffer<void>("packet"), p.buffer<void>("received"), packet_bytes,
                                CollectiveDataType::INT8, i, p.stream, "padded_intermediates"), "Native padded allgather");
                            const MoEGroupedIntermediateConsumeLaunch consume{.layout = layout, .route_owners = p.buffer<std::int32_t>("owners"),
                                .original_to_grouped = p.buffer<std::int32_t>("all_map"), .participant_packets = p.buffer<std::uint32_t>("received"),
                                .grouped_values = p.buffer<void>("output"), .grouped_scales = scales_bytes ? p.buffer<float>("output_scales") : nullptr};
                            require(importFixed(p.device, consume, p.stream), "Padded economy import");
                        }
                        capture.finish();
                        require(p.graph->instantiate(), "Compact economy instantiate");
                        for (int sample = -warmups; sample < samples; ++sample)
                        {
                            rendezvous.arrive_and_wait();
                            require(p.backend->recordEvent(p.begin, p.device.ordinal, p.stream) && p.graph->launch() &&
                                p.backend->recordEvent(p.end, p.device.ordinal, p.stream) && p.backend->waitForEvent(p.end, p.device.ordinal), "Compact economy timed replay");
                            float milliseconds = 0;
                            require(p.backend->eventElapsedTimeMs(p.begin, p.end, p.device.ordinal, &milliseconds), "Compact economy elapsed time");
                            if (sample >= 0) timings[i][sample] = milliseconds * 1000.0;
                        }
                        for (const bool scale : {false, true})
                        {
                            if (scale && !scales_bytes) continue;
                            const auto words = scale ? layout.scaleWords() : layout.valueWords();
                            std::vector<std::uint32_t> actual(layout.route_capacity * words);
                            require(p.backend->deviceToHostOnStream(actual.data(), p.buffer<void>(scale ? "output_scales" : "output"),
                                actual.size() * sizeof(actual[0]), p.device.ordinal, p.stream) &&
                                p.backend->recordEvent(p.end, p.device.ordinal, p.stream) && p.backend->waitForEvent(p.end, p.device.ordinal), "Compact economy validate terminal");
                            for (std::uint32_t route = 0; route < layout.route_capacity; ++route)
                                for (size_t j = 0; j < words; ++j)
                                    require(actual[route * words + j] == (route < live_routes
                                        ? word(route, j + (scale ? layout.valueWords() : 0)) : 0xa5a5a5a5u), "Compact economy changed output bits or untouched tail");
                        }
                    });
                }
                catch (...) { errors[i] = std::current_exception(); rendezvous.arrive_and_drop(); }
            });
        for (auto &thread : threads) thread.join();
        for (auto &error : errors) if (error) std::rethrow_exception(error);
        std::array<double, samples> maxima{};
        for (int i = 0; i < samples; ++i) maxima[i] = std::max(timings[0][i], timings[1][i]);
        std::sort(maxima.begin(), maxima.end());
        const auto median = maxima[samples / 2];
        std::printf("COMPACT_MOE_EXCHANGE backend=%s mode=%s encoding=%u capacity_routes=%u live_routes=%u owner0_eighths=%u wire_bytes=%zu median_us=%.3f p90_us=%.3f samples=%d\n",
            first.is_cuda() ? "CUDA" : "ROCm", compact ? "device_counted" : "native_padded", static_cast<unsigned>(layout.encoding),
            layout.route_capacity, live_routes, first_share, compact ? live_routes * layout.compactRecordWords() * sizeof(std::uint32_t) : 2 * layout.packetBytes(),
            median, maxima[samples * 9 / 10], samples);
        return median;
    }

    /** @brief Matched repeated-order comparison; correctness is mandatory, timing is evidence only. */
    void compare(DeviceId first)
    {
        for (const auto encoding : {MoEGroupedIntermediateEncoding::BlockQ8FP32Scales, MoEGroupedIntermediateEncoding::FP32})
            // Include serial/depth-three/depth-fifteen and small-tail
            // geometries as well as saturated prefill. Fusion must not earn
            // its decode win by throttling the large-payload parallel copy.
            for (const auto shape : std::array<std::array<unsigned, 3>, 8>{{
                {8, 8, 4}, {32, 32, 4}, {128, 8, 4}, {128, 128, 4},
                {512, 448, 4}, {4096, 3584, 4}, {4096, 3584, 7}, {4096, 4096, 8}}})
            {
                const MoEGroupedIntermediateLayout layout{encoding, 512, shape[0], 2};
                for (auto mode : {Exchange::NativePadded, Exchange::DeviceCounted, Exchange::DeviceCounted, Exchange::NativePadded})
                    measure(first, mode, layout, shape[1], shape[2]);
            }
    }
}
#ifdef HAVE_CUDA
TEST(MoECompactIntermediateEconomy, CUDA2) { compare(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
TEST(MoECompactIntermediateEconomy, ROCm2) { compare(DeviceId::rocm(0)); }
#endif
