/**
 * @file ROCmMoEProjectionExchangeFixture.h
 * @brief Shared captured two-device protocol for functional/economy gates.
 *
 * Both callers execute production kernels and native RCCL. The functional
 * caller checks twenty changing-owner replays without enabling timing. The
 * Release performance caller interleaves paired timing samples. Weights are
 * synthetic, correctly packed and redundant for an exact oracle; this fixture
 * is not a production residency or memory-admission certificate.
 */
#pragma once

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include "collective/LocalTPContext.h"
#include "execution/moe/MoEOverlayNodeLocalRouteExchange.h"
#include "kernels/common/MoEGroupedIntermediateExchangeKernels.h"
#include "kernels/rocm/moe/ROCmMoEKernel.h"
#include "ROCmMoEProjectionBoundaryFixture.h"

#include <array>
#include <barrier>
#include <bit>
#include <cmath>
#include <exception>
#include <thread>
#include <vector>

namespace llaminar2::test::projection_exchange
{
    /** @brief Compare independent expert dots with the fused per-token loop. */
    enum class DownPublication { FusedToken, IndependentRoutes };

    /** @brief Synthetic geometry and replay coverage; never production policy. */
    struct Shape
    {
        int rows = 65;
        int width = 512;
        int intermediate = 256;
        int experts = 8;
        int top_k = 3;
        int warmups = 0;
        int samples = 20;
        DownPublication down_publication = DownPublication::FusedToken;
        const char *gate_up_format = "IQ3_S";
        const char *down_format = "IQ3_S";
    };

    /** @brief Optional economy evidence; functional callers never request timing. */
    struct Timing
    {
        double median_us = 0, p10_us = 0, p90_us = 0, control_us = 0;
    };

    /** @brief Throw while the transaction still owns its abort/retirement handles. */
    inline void require(bool result, const char *message)
    {
        if (!result) throw std::runtime_error(message);
    }

    /** @brief Exact byte comparison without treating signed zero or NaNs as equal. */
    inline void requireExact(const std::vector<float> &actual, const std::vector<float> &expected)
    {
        require(actual.size() == expected.size(), "projection exchange output geometry changed");
        for (std::size_t i = 0; i < expected.size(); ++i)
            if (!std::isfinite(expected[i]) || std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(expected[i]))
                throw std::runtime_error("projection exchange changed output byte at element " + std::to_string(i));
    }

    /** @brief Retain exact timing events until the measured stream has completed. */
    struct Events final
    {
        hipStream_t stream;
        hipEvent_t begin = nullptr, end = nullptr;
        /** @brief Create timing resources outside graph recording. */
        explicit Events(hipStream_t execution_stream) : stream(execution_stream)
        {
            checkProjectionHip(hipEventCreate(&begin), "create projection begin event");
            checkProjectionHip(hipEventCreate(&end), "create projection end event");
        }
        /** @brief Teardown joins only this diagnostic's explicit stream. */
        ~Events()
        {
            (void)hipStreamSynchronize(stream);
            (void)hipEventDestroy(end);
            (void)hipEventDestroy(begin);
        }
    };

    /**
     * @brief Measure a real two-GPU projection transaction against complete math.
     * @param shape Physical geometry and replay budget; every replay changes ownership.
     * @param timing Optional performance output; null disables all timing collection.
     *
     * Fixture generation and oracle launches occur outside timing. The maximum
     * participant event interval is retained per sample, never the sum of two
     * overlapping intervals. Empty/short replays precede the timing cohort.
     */
    inline void run(const Shape &shape, Timing *timing = nullptr)
    {
        const auto [rows, width, intermediate, experts, top_k, warmups, samples,
            down_publication, gate_up_format, down_format] = shape;
        require(rows > 0 && width > 0 && width % 64 == 0 && intermediate > 0 && intermediate % 256 == 0 &&
            top_k > 0 && experts >= top_k && warmups >= 0 && samples > 0, "invalid projection exchange geometry");
        auto tp = createLocalTPContext({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)}, {},
            CollectiveBackendType::RCCL);
        require(tp != nullptr, "projection exchange RCCL setup failed");
        MoEOverlayNodeLocalRouteExchange exchange({.devices = {DeviceId::rocm(0), DeviceId::rocm(1)},
            .root_device = DeviceId::rocm(0), .identity = "projection_exchange_economy_control"});
        exchange.materialize({{0, DeviceId::rocm(0)}, {1, DeviceId::rocm(1)}}, 0, rows, top_k, width);
        std::barrier rendezvous(2);
        std::array<std::exception_ptr, 2> failures{};
        std::array<std::vector<double>, 2> times{std::vector<double>(samples), std::vector<double>(samples)};
        std::array<std::vector<double>, 2> control_times{std::vector<double>(samples), std::vector<double>(samples)};
        std::array<std::thread, 2> threads;
        for (int participant = 0; participant < 2; ++participant)
            threads[participant] = std::thread([&, participant]
            {
                try
                {
                    auto &worker = GPUDeviceContextPool::instance().getContext(DeviceId::rocm(participant));
                    worker.submitAndWait([&]
                    {
                        const ProjectionFixtureFormats formats{
                            quantizedMoEVerifierFormat(gate_up_format),
                            quantizedMoEVerifierFormat(down_format)};
                        const ProjectionFixturePlacement placement{DeviceId::rocm(participant), experts, top_k};
                        const auto ownership = MoEExpertProjectionOwnership::gateUpOwnedDownColumns(
                            {experts, width, intermediate}, participant, 2);
                        const auto down_projection = ownership.projection(WeightRole::MoEExpertDown);
                        ROCmMoEProjectionBoundaryFixture producer(formats, width, intermediate, rows, placement);
                        ROCmMoEProjectionBoundaryFixture consumer(formats, width, intermediate, rows, placement);
                        Events events(producer.stream);
                        std::unique_ptr<IGPUGraphCapture> transaction;
                        std::unique_ptr<IGPUGraphCapture> control;
                        try
                        {
                            const MoEGroupedIntermediateLayout layout{MoEGroupedIntermediateEncoding::BlockQ8FP32Scales,
                                static_cast<std::uint32_t>(intermediate), static_cast<std::uint32_t>(rows * top_k), 2};
                            auto *packet = producer.allocate<std::uint32_t>(layout.packetBytes() / sizeof(std::uint32_t));
                            auto *packets = producer.allocate<std::uint32_t>(2 * layout.packetBytes() / sizeof(std::uint32_t));
                            auto *shard = producer.allocate<float>(static_cast<std::size_t>(rows) * down_projection.rows);
                            auto *columns = producer.allocate<float>(static_cast<std::size_t>(rows) * width);
                            auto *assembled = producer.allocate<float>(static_cast<std::size_t>(rows) * width);
                            auto *routes = producer.allocate<float>(static_cast<std::size_t>(rows) * top_k * width);
                            auto *control_output = producer.allocate<float>(static_cast<std::size_t>(rows) * width);
                            auto *status = producer.allocate<std::int32_t>(1);
                            MoENodeLocalRouteConsumeLaunch control_consume;
                            MoENodeLocalRoutePublishLaunch control_publish;
                            if (participant == 0)
                            {
                                auto *peers = producer.upload(exchange.rootPeerBindings(DeviceId::rocm(0)));
                                control_consume = {.peers = peers, .peer_count = 1,
                                    .root_canonical_route_contributions = routes,
                                    .domain_assignment = {.participant_ids = producer.routeOwners(), .capacity = static_cast<std::uint32_t>(rows * top_k)},
                                    .external_route_source = MoEExternalCanonicalRouteSource::DeferredDenseMerge,
                                    .dense_output = control_output, .validation_status = status, .root_participant = 0,
                                    .physical_rows = static_cast<std::uint32_t>(rows), .top_k = static_cast<std::uint32_t>(top_k), .d_model = static_cast<std::uint32_t>(width)};
                            }
                            else
                                control_publish = {.lane = exchange.producerBinding(DeviceId::rocm(1)),
                                    .canonical_route_contributions = routes,
                                    .domain_assignment = {.participant_ids = producer.routeOwners(), .capacity = static_cast<std::uint32_t>(rows * top_k)},
                                    .live_route_slots = static_cast<std::uint32_t>(rows * top_k)};
                            auto *down = consumer.slice(ownership);
                            producer.reset(rows, rows, 1, {2, participant, ProjectionGroupingCoverage::OwnedRoutes});
                            consumer.reset(rows, rows, 1, {2, participant, ProjectionGroupingCoverage::AllRoutes});
                            ProjectionBoundaryGraph oracle(consumer.stream, [&] {
                                consumer.launch(rows, MoEPrefillProjectionExecution::complete(width));
                            });
                            const MoEGroupedIntermediatePackLaunch pack{.layout = layout,
                                .route_owners = producer.routeOwners(), .original_to_grouped = producer.originalToGrouped(),
                                .grouped_values = producer.swiglu, .grouped_scales = producer.swiglu_scales,
                                .packet = packet, .participant = participant};
                            const MoEGroupedIntermediateConsumeLaunch consume{.layout = layout,
                                .route_owners = consumer.routeOwners(), .original_to_grouped = consumer.originalToGrouped(),
                                .participant_packets = packets, .grouped_values = consumer.swiglu,
                                .grouped_scales = consumer.swiglu_scales};
                            checkProjectionHip(hipStreamSynchronize(producer.stream), "projection exchange setup completion");
                            transaction = worker.createGraphCapture(producer.stream);
                            rendezvous.arrive_and_wait();
                            require(transaction && transaction->beginCapture(), "projection exchange begin capture");
                            producer.launch(rows, MoEPrefillProjectionExecution::gateUp(width));
                            require(rocm::packGroupedIntermediate(pack, producer.stream), "projection exchange pack");
                            require(tp->allgatherRawOnStream(packet, packets, layout.packetBytes(),
                                CollectiveDataType::INT8, participant, producer.stream, "projection_intermediates"),
                                "projection exchange packet allgather");
                            require(rocm::consumeGroupedIntermediate(consume, producer.stream), "projection exchange consume");
                            // Only the tiny top-k addition is serial. The route
                            // variant computes independent dots concurrently,
                            // then performs the unchanged ordered fold. Reuse
                            // the control's persistent route bank: the two graphs
                            // never execute concurrently and overwrite all rows.
                            const auto down_execution = MoEPrefillProjectionExecution::down(
                                down_projection.source_rows, down_projection.first_row, down_projection.rows);
                            if (down_publication == DownPublication::IndependentRoutes)
                            {
                                consumer.launchOnStream(producer.stream, rows, down_execution,
                                    down, nullptr, routes);
                                require(rocmMoE_reduce_canonical_route_contributions(routes, shard,
                                    rows, top_k, down_projection.rows, participant, producer.stream),
                                    "projection exchange ordered route fold");
                            }
                            else
                                consumer.launchOnStream(producer.stream, rows, down_execution, down, shard);
                            require(tp->allgatherRawOnStream(shard, columns, static_cast<std::size_t>(rows) * down_projection.rows,
                                CollectiveDataType::FLOAT32, participant, producer.stream, "projection_output_columns"),
                                "projection exchange column allgather");
                            for (int peer = 0; peer < 2; ++peer)
                                checkProjectionHip(hipMemcpy2DAsync(assembled + peer * down_projection.rows, width * sizeof(float),
                                    columns + peer * static_cast<std::size_t>(rows) * down_projection.rows,
                                    down_projection.rows * sizeof(float), down_projection.rows * sizeof(float),
                                    rows, hipMemcpyDeviceToDevice, producer.stream),
                                    "projection exchange column assembly");
                            require(transaction->endCapture() && transaction->instantiate(), "projection exchange instantiate");
                            // The control uses the existing production route fabric
                            // and the identical weights/inputs, not summed estimates.
                            ROCmMoEKernel kernel(participant);
                            control = worker.createGraphCapture(producer.stream);
                            rendezvous.arrive_and_wait();
                            require(control && control->beginCapture(), "projection control begin capture");
                            producer.launchOnStream(producer.stream, rows,
                                MoEPrefillProjectionExecution::complete(width), nullptr, nullptr, routes);
                            const MoEKernelLaunchContext launch{.stream = producer.stream};
                            if (participant == 0)
                            {
                                require(kernel.acquireNodeLocalCanonicalRoutes(launch, control_consume), "projection control acquire");
                                require(kernel.stageNodeLocalCanonicalRoutes(launch, control_consume), "projection control stage");
                                require(kernel.foldNodeLocalCanonicalRoutes(launch, control_consume), "projection control fold");
                            }
                            else
                                require(kernel.publishNodeLocalCanonicalRoutes(launch, control_publish), "projection control publish");
                            require(tp->broadcastRawOnStream(control_output, control_output, static_cast<std::size_t>(rows) * width,
                                CollectiveDataType::FLOAT32, 0, participant, producer.stream, "projection_control_output"),
                                "projection control broadcast");
                            require(control->endCapture() && control->instantiate(), "projection control instantiate");
                            for (int sample = -3; sample < warmups + samples; ++sample)
                            {
                                const int live = sample == -2 ? 0 : (sample == -1 ? rows / 2 : rows);
                                producer.reset(rows, live, sample + 7, {2, participant, ProjectionGroupingCoverage::OwnedRoutes});
                                consumer.reset(rows, live, sample + 7, {2, participant, ProjectionGroupingCoverage::AllRoutes});
                                oracle.replay();
                                const auto expected = consumer.download(consumer.output, static_cast<std::size_t>(rows) * width);
                                for (int round = 0; round < 2; ++round)
                                {
                                    const bool candidate = ((sample + 3 + round) % 2) == 0;
                                    rendezvous.arrive_and_wait();
                                    if (timing) checkProjectionHip(hipEventRecord(events.begin, producer.stream), "projection timing start");
                                    require((candidate ? transaction : control)->launch(), "projection exchange replay");
                                    checkProjectionHip(hipEventRecord(events.end, producer.stream), "projection timing end");
                                    checkProjectionHip(hipEventSynchronize(events.end), "projection terminal event");
                                    float ms = 0;
                                    if (timing) checkProjectionHip(hipEventElapsedTime(&ms, events.begin, events.end), "projection elapsed time");
                                    if (timing && sample >= warmups)
                                        (candidate ? times : control_times)[participant][sample - warmups] = ms * 1000.0;
                                    requireExact(producer.download(candidate ? assembled : control_output, expected.size()), expected);
                                }
                                if (live > 0)
                                    require(std::any_of(expected.begin(), expected.end(), [](float x) { return x != 0; }),
                                        "projection exchange requires nonzero math witness");
                            }
                            rendezvous.arrive_and_wait();
                        }
                        catch (...)
                        {
                            // Abort native peers before graph/buffer destructors can
                            // join a stream whose collective has lost a participant.
                            exchange.abortForShutdown();
                            tp->requestAbort();
                            throw;
                        }
                    });
                }
                catch (...)
                {
                    failures[participant] = std::current_exception();
                    exchange.abortForShutdown();
                    tp->requestAbort();
                    rendezvous.arrive_and_drop();
                }
            });
        for (auto &thread : threads) thread.join();
        for (const auto &failure : failures) if (failure) std::rethrow_exception(failure);
        if (!timing) return;
        std::vector<double> critical(samples), control_critical(samples);
        for (int sample = 0; sample < samples; ++sample)
        {
            critical[sample] = std::max(times[0][sample], times[1][sample]);
            control_critical[sample] = std::max(control_times[0][sample], control_times[1][sample]);
        }
        std::sort(critical.begin(), critical.end());
        std::sort(control_critical.begin(), control_critical.end());
        *timing = {critical[samples / 2], critical[samples / 10], critical[9 * samples / 10], control_critical[samples / 2]};
    }
} // namespace llaminar2::test::projection_exchange
