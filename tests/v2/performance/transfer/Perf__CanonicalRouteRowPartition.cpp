/**
 * @file Perf__CanonicalRouteRowPartition.cpp
 * @brief Captured economy probe for distributing exact MoE folds by token row.
 *
 * Both candidates call the existing production sparse exchange kernels and
 * native NCCL/RCCL collectives. The rooted candidate folds all rows on one GPU
 * and broadcasts them; the row-partitioned candidate folds disjoint halves and
 * allgathers the finished rows. Neither changes route ownership, wire precision
 * or the ascending top-k FP32 addition order. This is an explicit experiment,
 * not an installed model-graph policy or a replacement transport.
 *
 * Timing includes the complete captured exchange and final dense publication.
 * Setup, reset uploads and byte validation are outside the interval. Each
 * sample changes ownership and data to expose stale epochs; adversarial route
 * values distinguish the canonical fold from participant-subtotal reduction.
 * A separate compact-intermediate probe measures native allgather of existing
 * quantized SwiGLU/scales followed by native allgather of finished column
 * shards. It is transport evidence only, not a claim of model-graph support.
 */
#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include "collective/LocalTPContext.h"
#include "execution/moe/MoEOverlayNodeLocalRouteExchange.h"
#ifdef HAVE_CUDA
#include "kernels/cuda/moe/CUDAMoEKernel.h"
#endif
#ifdef HAVE_ROCM
#include "kernels/rocm/moe/ROCmMoEKernel.h"
#endif
#include "transfer/TransferEngine.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace llaminar2;

namespace
{
    /** Complete, explicitly chosen candidate; never selected on an error. */
    enum class FoldPlacement { Rooted, TokenRows };

    /** @brief Propagate a fixture error through participant teardown. */
    void require(bool result, const char *message)
    {
        if (!result) throw std::runtime_error(message);
    }

    /** @return The production kernel facade for the exact selected endpoint. */
    std::unique_ptr<IMoEKernel> makeKernel(DeviceId device)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda()) return std::make_unique<CUDAMoEKernel>(device.ordinal);
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) return std::make_unique<ROCmMoEKernel>(device.ordinal);
#endif
        throw std::invalid_argument("row-fold probe requires an enabled GPU backend");
    }

    /**
     * @brief Own one participant's persistent buffers, graph and timing events.
     *
     * The outer fixture aborts all exchange lanes before joining a failed peer.
     * This scope then drains its exact stream before releasing captured addresses.
     */
    struct Participant final
    {
        /** @brief Bind the exact device/stream and allocate only timing handles. */
        Participant(DeviceId endpoint, void *execution_stream)
            : device(endpoint), backend(getBackendFor(endpoint)), stream(execution_stream)
        {
            begin = backend->createTimingEvent(device.ordinal);
            end = backend->createTimingEvent(device.ordinal);
            require(begin && end, "row-fold timing-event allocation failed");
        }

        /** @brief Retire the graph before its buffers and event handles. */
        ~Participant()
        {
            (void)backend->synchronizeStream(stream, device.ordinal);
            graph.reset();
            backend->destroyEvent(end, device.ordinal);
            backend->destroyEvent(begin, device.ordinal);
        }

        Participant(const Participant &) = delete;
        Participant &operator=(const Participant &) = delete;

        /** @return Stable TransferEngine-owned bytes retained until graph retirement. */
        void *allocate(std::size_t bytes)
        {
            auto allocation = TransferEngine::instance().allocateDeviceTransferBuffer(bytes, device);
            require(allocation && allocation->isBound(), "row-fold buffer allocation failed");
            void *address = allocation->mutableDeviceData();
            buffers.push_back(std::move(allocation));
            return address;
        }

        DeviceId device;
        IBackend *backend;
        void *stream;
        void *begin = nullptr;
        void *end = nullptr;
        std::vector<std::shared_ptr<DeviceTransferBuffer>> buffers;
        std::unique_ptr<IGPUGraphCapture> graph;
    };

    /**
     * @brief Compare whole captured exchanges with an identical canonical oracle.
     * @param first First of two adjacent native GPU ordinals.
     * @param rows Even physical row count; equal-size native allgather is explicit.
     * @param placement Rooted publication or disjoint row-owner publication.
     * @return Median maximum participant event interval in microseconds.
     *
     * This probe excludes shared-expert reduction and model compute. A winning
     * result would justify a separate production graph/memory design, not certify
     * that design or predict whole-model speedup by itself.
     */
    double measure(DeviceId first, std::uint32_t rows, FoldPlacement placement)
    {
        constexpr std::uint32_t top_k = 8u;
        constexpr std::uint32_t width = 2048u;
        constexpr int warmups = 5;
        constexpr int samples = 31;
        require(rows > 0 && rows % 2u == 0u, "row-fold probe requires positive even rows");
        const std::array devices{first, first.is_cuda()
            ? DeviceId::cuda(first.ordinal + 1) : DeviceId::rocm(first.ordinal + 1)};
        const bool partitioned = placement == FoldPlacement::TokenRows;
        const std::uint32_t fold_rows = partitioned ? rows / 2u : rows;
        const std::size_t route_count = static_cast<std::size_t>(rows) * top_k;
        const std::size_t route_elements = route_count * width;
        const std::size_t dense_elements = static_cast<std::size_t>(rows) * width;
        auto tp = createLocalTPContext(first.is_cuda()
                ? std::vector<GlobalDeviceAddress>{GlobalDeviceAddress::cuda(first.ordinal),
                    GlobalDeviceAddress::cuda(first.ordinal + 1)}
                : std::vector<GlobalDeviceAddress>{GlobalDeviceAddress::rocm(first.ordinal),
                    GlobalDeviceAddress::rocm(first.ordinal + 1)}, {},
            first.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        require(tp != nullptr, "row-fold native collective setup failed");
        std::vector<std::unique_ptr<MoEOverlayNodeLocalRouteExchange>> exchanges;
        for (int owner = 0; owner < (partitioned ? 2 : 1); ++owner)
        {
            auto exchange = std::make_unique<MoEOverlayNodeLocalRouteExchange>(
                MoEOverlayNodeLocalRouteExchange::Config{
                    .devices = {devices[0], devices[1]}, .root_device = devices[owner],
                    .identity = "captured_row_fold_economy"});
            exchange->materialize({{.participant_id = 0, .device = devices[0]},
                {.participant_id = 1, .device = devices[1]}}, owner, fold_rows, top_k, width);
            exchanges.push_back(std::move(exchange));
        }
        std::barrier rendezvous(2);
        std::array<std::exception_ptr, 2> errors{};
        std::array<std::array<double, samples>, 2> times{};
        std::array<std::thread, 2> threads;
        for (int participant = 0; participant < 2; ++participant)
        {
            threads[participant] = std::thread([&, participant]
            {
                try
                {
                    const auto device = devices[participant];
                    auto &gpu = GPUDeviceContextPool::instance().getContext(device);
                    gpu.submitAndWait([&]
                    {
                        Participant storage(device, gpu.getOrCreateAuxiliaryStream("row_fold_economy"));
                        try
                        {
                            auto kernel = makeKernel(device);
                            auto *routes = static_cast<float *>(storage.allocate(route_elements * sizeof(float)));
                            auto *owners = static_cast<std::int32_t *>(storage.allocate(route_count * sizeof(std::int32_t)));
                            auto *output = static_cast<float *>(storage.allocate(dense_elements * sizeof(float)));
                            auto *status = static_cast<std::int32_t *>(storage.allocate(sizeof(std::int32_t)));
                            auto *peer_device = static_cast<MoENodeLocalRoutePeerDeviceBinding *>(
                                storage.allocate(sizeof(MoENodeLocalRoutePeerDeviceBinding)));
                            const bool consumer = partitioned || participant == 0;
                            const bool producer = partitioned || participant != 0;
                            const int consume_owner = partitioned ? participant : 0;
                            const int publish_owner = partitioned ? 1 - participant : 0;
                            const std::size_t consume_first = partitioned ? participant * fold_rows : 0;
                            const std::size_t publish_first = partitioned ? (1 - participant) * fold_rows : 0;
                            MoENodeLocalRouteConsumeLaunch consume;
                            MoENodeLocalRoutePublishLaunch publish;
                            if (consumer)
                            {
                                const auto bindings = exchanges[consume_owner]->rootPeerBindings(device);
                                require(bindings.size() == 1u, "row-fold peer binding count");
                                require(storage.backend->hostToDeviceOnStream(peer_device, bindings.data(),
                                    sizeof(bindings.front()), device.ordinal, storage.stream), "row-fold binding upload");
                                consume = {
                                    .peers = peer_device, .peer_count = 1u,
                                    .root_canonical_route_contributions = routes + consume_first * top_k * width,
                                    .domain_assignment = {.participant_ids = owners + consume_first * top_k,
                                        .capacity = fold_rows * top_k},
                                    .external_route_source = MoEExternalCanonicalRouteSource::DeferredDenseMerge,
                                    .dense_output = output + consume_first * width,
                                    .validation_status = status, .root_participant = participant,
                                    .physical_rows = fold_rows, .top_k = top_k, .d_model = width};
                                require(consume.valid(), "row-fold consume descriptor");
                            }
                            if (producer)
                            {
                                publish = {
                                    .lane = exchanges[publish_owner]->producerBinding(device),
                                    .canonical_route_contributions = routes + publish_first * top_k * width,
                                    .domain_assignment = {.participant_ids = owners + publish_first * top_k,
                                        .capacity = fold_rows * top_k},
                                    .live_route_slots = fold_rows * top_k};
                                require(publish.valid(), "row-fold publish descriptor");
                            }
                            require(gpu.synchronizeStreamChecked(storage.stream), "row-fold setup completion");
                            storage.graph = gpu.createGraphCapture(storage.stream);
                            require(storage.graph != nullptr, "row-fold graph allocation");
                            rendezvous.arrive_and_wait();
                            require(storage.graph->beginCapture(), "row-fold begin capture");
                            // Each endpoint publishes before waiting. The two disjoint
                            // row lanes therefore cannot wait on one another's future work.
                            const MoEKernelLaunchContext launch{.stream = storage.stream};
                            if (producer) require(kernel->publishNodeLocalCanonicalRoutes(launch, publish), "row-fold publish");
                            if (consumer)
                            {
                                require(kernel->acquireNodeLocalCanonicalRoutes(launch, consume), "row-fold acquire");
                                require(kernel->stageNodeLocalCanonicalRoutes(launch, consume), "row-fold stage");
                                require(kernel->foldNodeLocalCanonicalRoutes(launch, consume), "row-fold ordered fold");
                            }
                            if (partitioned)
                                require(tp->allgatherRawOnStream(output + consume_first * width, output,
                                    static_cast<std::size_t>(fold_rows) * width, CollectiveDataType::FLOAT32,
                                    participant, storage.stream, "row_fold_dense_allgather"), "row-fold allgather");
                            else
                                require(tp->broadcastRawOnStream(output, output, dense_elements,
                                    CollectiveDataType::FLOAT32, 0, participant, storage.stream,
                                    "root_fold_dense_broadcast"), "root-fold broadcast");
                            require(storage.graph->endCapture() && storage.graph->instantiate(), "row-fold graph instantiate");

                            std::vector<float> source(route_elements), expected(dense_elements), actual(dense_elements);
                            std::vector<std::int32_t> assignment(route_count);
                            for (int sample = 0; sample < warmups + samples; ++sample)
                            {
                                std::fill(expected.begin(), expected.end(), 0.0f);
                                for (std::size_t slot = 0; slot < route_count; ++slot)
                                {
                                    assignment[slot] = static_cast<int>((slot + sample) % 2u);
                                    for (std::uint32_t column = 0; column < width; ++column)
                                    {
                                        const auto route = slot % top_k;
                                        const float canonical = route == 0u ? 1.0e20f : route == 2u ? -1.0e20f
                                            : static_cast<float>(1 + sample + column % 7u) * 0.125f;
                                        source[slot * width + column] = assignment[slot] == participant
                                            ? canonical : -8192.0f; // Reading an unowned row must fail the oracle.
                                        expected[(slot / top_k) * width + column] += canonical;
                                    }
                                }
                                require(storage.backend->hostToDeviceOnStream(routes, source.data(),
                                    source.size() * sizeof(float), device.ordinal, storage.stream), "row-fold source reset");
                                require(storage.backend->hostToDeviceOnStream(owners, assignment.data(),
                                    assignment.size() * sizeof(std::int32_t), device.ordinal, storage.stream), "row-fold owner reset");
                                require(gpu.synchronizeStreamChecked(storage.stream), "row-fold reset completion");
                                rendezvous.arrive_and_wait();
                                require(storage.backend->recordEvent(storage.begin, device.ordinal, storage.stream), "row-fold start");
                                require(storage.graph->launch(), "row-fold replay");
                                require(storage.backend->recordEvent(storage.end, device.ordinal, storage.stream)
                                    && storage.backend->waitForEvent(storage.end, device.ordinal), "row-fold terminal event");
                                float milliseconds = 0;
                                require(storage.backend->eventElapsedTimeMs(storage.begin, storage.end,
                                    device.ordinal, &milliseconds), "row-fold elapsed time");
                                if (sample >= warmups) times[participant][sample - warmups] = milliseconds * 1000.0;
                                require(storage.backend->deviceToHostOnStream(actual.data(), output,
                                    actual.size() * sizeof(float), device.ordinal, storage.stream)
                                    && gpu.synchronizeStreamChecked(storage.stream), "row-fold output observation");
                                require(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
                                    "row-fold serial contract violated");
                            }
                            rendezvous.arrive_and_wait();
                        }
                        catch (...)
                        {
                            // Release any device-side waiter before stack unwinding
                            // drains this participant's stream and captured addresses.
                            for (auto &exchange : exchanges) exchange->abortForShutdown();
                            tp->requestAbort();
                            throw;
                        }
                    });
                }
                catch (...)
                {
                    errors[participant] = std::current_exception();
                    for (auto &exchange : exchanges) exchange->abortForShutdown();
                    tp->requestAbort();
                    rendezvous.arrive_and_drop();
                }
            });
        }
        for (auto &thread : threads) thread.join();
        for (const auto &error : errors) if (error) std::rethrow_exception(error);
        std::array<double, samples> critical{};
        for (int sample = 0; sample < samples; ++sample)
            critical[sample] = std::max(times[0][sample], times[1][sample]);
        std::sort(critical.begin(), critical.end());
        const double median = critical[samples / 2];
        std::printf("CANONICAL_ROW_FOLD,%s,placement=%s,rows=%u,width=%u,top_k=%u,median_us=%.3f,p10_us=%.3f,p90_us=%.3f,byte_exact=1\n",
            first.is_cuda() ? "CUDA" : "ROCm", partitioned ? "token_rows" : "rooted", rows,
            width, top_k, median, critical[samples / 10], critical[9 * samples / 10]);
        return median;
    }

    /**
     * @brief Measure the wire portion of projection-split expert execution.
     *
     * Each participant publishes a fixed-capacity, exclusive-owner packet in
     * original route order. One native byte allgather preserves every existing
     * INT8 and FP32-scale bit; no activation conversion or FP32 sum occurs.
     * After local down/fold compute, a second native allgather would publish
     * disjoint dense column shards. Here those finished shards are prepared
     * outside timing to isolate transport. Packing, projection, transpose and
     * orchestration costs are deliberately NOT claimed by this measurement.
     */
    double measureCompactProjectionExchange(DeviceId first, std::uint32_t rows)
    {
        constexpr std::size_t top_k = 8, intermediate = 512, width = 2048;
        constexpr int warmups = 5, samples = 31;
        const std::size_t route_count = rows * top_k;
        const std::size_t route_bytes = intermediate + (intermediate / 32) * sizeof(float);
        const std::size_t packet_bytes = route_count * route_bytes;
        const std::size_t shard_elements = rows * (width / 2);
        const std::array devices{first, first.is_cuda()
            ? DeviceId::cuda(first.ordinal + 1) : DeviceId::rocm(first.ordinal + 1)};
        auto tp = createLocalTPContext(first.is_cuda()
                ? std::vector<GlobalDeviceAddress>{GlobalDeviceAddress::cuda(first.ordinal),
                    GlobalDeviceAddress::cuda(first.ordinal + 1)}
                : std::vector<GlobalDeviceAddress>{GlobalDeviceAddress::rocm(first.ordinal),
                    GlobalDeviceAddress::rocm(first.ordinal + 1)}, {},
            first.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        require(tp != nullptr, "compact projection collective setup failed");
        std::barrier rendezvous(2);
        std::array<std::exception_ptr, 2> errors{};
        std::array<std::array<double, samples>, 2> times{};
        std::array<std::thread, 2> threads;
        for (int participant = 0; participant < 2; ++participant)
        {
            threads[participant] = std::thread([&, participant] {
                try
                {
                    auto &gpu = GPUDeviceContextPool::instance().getContext(devices[participant]);
                    gpu.submitAndWait([&] {
                        Participant storage(devices[participant], gpu.getOrCreateAuxiliaryStream("compact_projection_economy"));
                        try
                        {
                            auto *packet = static_cast<uint8_t *>(storage.allocate(packet_bytes));
                            auto *packets = static_cast<uint8_t *>(storage.allocate(2 * packet_bytes));
                            auto *shard = static_cast<float *>(storage.allocate(shard_elements * sizeof(float)));
                            auto *columns = static_cast<float *>(storage.allocate(2 * shard_elements * sizeof(float)));
                            storage.graph = gpu.createGraphCapture(storage.stream);
                            require(storage.graph != nullptr, "compact projection graph allocation");
                            rendezvous.arrive_and_wait();
                            require(storage.graph->beginCapture(), "compact projection capture begin");
                            require(tp->allgatherRawOnStream(packet, packets, packet_bytes,
                                CollectiveDataType::INT8, participant, storage.stream,
                                "compact_projection_intermediates"), "compact intermediate allgather");
                            require(tp->allgatherRawOnStream(shard, columns, shard_elements,
                                CollectiveDataType::FLOAT32, participant, storage.stream,
                                "compact_projection_columns"), "compact column allgather");
                            require(storage.graph->endCapture() && storage.graph->instantiate(), "compact projection instantiate");
                            std::vector<uint8_t> local(packet_bytes), expected_packets(2 * packet_bytes), actual_packets(2 * packet_bytes);
                            std::vector<float> local_shard(shard_elements), expected_columns(2 * shard_elements), actual_columns(2 * shard_elements);
                            for (int sample = 0; sample < warmups + samples; ++sample)
                            {
                                std::fill(expected_packets.begin(), expected_packets.end(), uint8_t{0});
                                for (std::size_t slot = 0; slot < route_count; ++slot)
                                    for (std::size_t byte = 0; byte < route_bytes; ++byte)
                                        expected_packets[((slot + sample) % 2) * packet_bytes + slot * route_bytes + byte] =
                                            static_cast<uint8_t>((slot * 19 + byte * 13 + sample * 7) & 255u);
                                std::copy_n(expected_packets.data() + participant * packet_bytes, packet_bytes, local.data());
                                for (std::size_t i = 0; i < expected_columns.size(); ++i)
                                    expected_columns[i] = static_cast<float>(static_cast<int>((i * 13 + sample) % 101) - 50) * 0.125f;
                                std::copy_n(expected_columns.data() + participant * shard_elements, shard_elements, local_shard.data());
                                require(storage.backend->hostToDeviceOnStream(packet, local.data(), packet_bytes,
                                    storage.device.ordinal, storage.stream), "compact packet reset");
                                require(storage.backend->hostToDeviceOnStream(shard, local_shard.data(), shard_elements * sizeof(float),
                                    storage.device.ordinal, storage.stream), "compact column reset");
                                require(gpu.synchronizeStreamChecked(storage.stream), "compact reset join");
                                rendezvous.arrive_and_wait();
                                require(storage.backend->recordEvent(storage.begin, storage.device.ordinal, storage.stream), "compact timing start");
                                require(storage.graph->launch(), "compact graph replay");
                                require(storage.backend->recordEvent(storage.end, storage.device.ordinal, storage.stream) &&
                                    storage.backend->waitForEvent(storage.end, storage.device.ordinal), "compact terminal event");
                                float milliseconds = 0;
                                require(storage.backend->eventElapsedTimeMs(storage.begin, storage.end,
                                    storage.device.ordinal, &milliseconds), "compact elapsed time");
                                if (sample >= warmups) times[participant][sample - warmups] = milliseconds * 1000.0;
                                require(storage.backend->deviceToHostOnStream(actual_packets.data(), packets, 2 * packet_bytes,
                                    storage.device.ordinal, storage.stream), "compact packet observation");
                                require(storage.backend->deviceToHostOnStream(actual_columns.data(), columns, actual_columns.size() * sizeof(float),
                                    storage.device.ordinal, storage.stream) && gpu.synchronizeStreamChecked(storage.stream), "compact column observation");
                                require(actual_packets == expected_packets, "compact packet bit contract violated");
                                require(std::memcmp(actual_columns.data(), expected_columns.data(), actual_columns.size() * sizeof(float)) == 0,
                                    "compact output-column bit contract violated");
                            }
                            rendezvous.arrive_and_wait();
                        }
                        catch (...) { tp->requestAbort(); throw; }
                    });
                }
                catch (...)
                {
                    errors[participant] = std::current_exception();
                    tp->requestAbort();
                    rendezvous.arrive_and_drop();
                }
            });
        }
        for (auto &thread : threads) thread.join();
        for (const auto &error : errors) if (error) std::rethrow_exception(error);
        std::array<double, samples> critical{};
        for (int sample = 0; sample < samples; ++sample)
            critical[sample] = std::max(times[0][sample], times[1][sample]);
        std::sort(critical.begin(), critical.end());
        std::printf("COMPACT_PROJECTION_EXCHANGE,%s,rows=%u,packet_bytes=%zu,column_shard_bytes=%zu,median_us=%.3f,p10_us=%.3f,p90_us=%.3f,byte_exact=1\n",
            first.is_cuda() ? "CUDA" : "ROCm", rows, packet_bytes, shard_elements * sizeof(float),
            critical[samples / 2], critical[samples / 10], critical[9 * samples / 10]);
        return critical[samples / 2];
    }

    /** Explicit backend/shape/placement cases support uncontaminated profiling. */
    class CanonicalRowFoldEconomy : public ::testing::TestWithParam<std::uint32_t> {};

#ifdef HAVE_ROCM
    TEST_P(CanonicalRowFoldEconomy, ROCmRooted)
    {
        if (getBackendFor(DeviceId::rocm(0))->deviceCount() < 2) GTEST_SKIP();
        measure(DeviceId::rocm(0), GetParam(), FoldPlacement::Rooted);
    }
    TEST_P(CanonicalRowFoldEconomy, ROCmTokenRows)
    {
        if (getBackendFor(DeviceId::rocm(0))->deviceCount() < 2) GTEST_SKIP();
        measure(DeviceId::rocm(0), GetParam(), FoldPlacement::TokenRows);
    }
    TEST_P(CanonicalRowFoldEconomy, ROCmCompactProjectionExchange)
    {
        if (getBackendFor(DeviceId::rocm(0))->deviceCount() < 2) GTEST_SKIP();
        measureCompactProjectionExchange(DeviceId::rocm(0), GetParam());
    }
#endif
#ifdef HAVE_CUDA
    TEST_P(CanonicalRowFoldEconomy, CUDARooted)
    {
        if (getBackendFor(DeviceId::cuda(0))->deviceCount() < 2) GTEST_SKIP();
        measure(DeviceId::cuda(0), GetParam(), FoldPlacement::Rooted);
    }
    TEST_P(CanonicalRowFoldEconomy, CUDATokenRows)
    {
        if (getBackendFor(DeviceId::cuda(0))->deviceCount() < 2) GTEST_SKIP();
        measure(DeviceId::cuda(0), GetParam(), FoldPlacement::TokenRows);
    }
    TEST_P(CanonicalRowFoldEconomy, CUDACompactProjectionExchange)
    {
        if (getBackendFor(DeviceId::cuda(0))->deviceCount() < 2) GTEST_SKIP();
        measureCompactProjectionExchange(DeviceId::cuda(0), GetParam());
    }
#endif
    INSTANTIATE_TEST_SUITE_P(Captured, CanonicalRowFoldEconomy,
        ::testing::Values(16u, 64u, 448u, 512u),
        [](const ::testing::TestParamInfo<std::uint32_t> &info) {
            return "Rows" + std::to_string(info.param);
        });
}
