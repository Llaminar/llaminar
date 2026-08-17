/**
 * @file Test__LocalTPHeterogeneousGraphCaptureTicket.cpp
 * @brief Real CUDA/ROCm proof for LocalTP graph-lifecycle event tickets.
 *
 * The production LocalTP capture controller must place mixed-vendor
 * participants at one safe lifecycle generation without pretending that a
 * CUDA event can be consumed by a ROCm stream (or vice versa). This test queues
 * real work and a sentinel event on each exact backend stream, invokes the
 * production LocalTP ticket boundary concurrently, and proves that return
 * implies both sentinels completed. Repeated begin/end-shaped generations lock
 * down cyclic event reuse and PerfStats observability.
 */

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "collective/LocalTPContext.h"
#include "collective/backends/HeterogeneousBackend.h"
#include "utils/PerfStatsCollector.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

using namespace llaminar2;

namespace
{
    /**
     * @brief Poll one backend event with the production collective timeout.
     * @param backend Backend that owns the event.
     * @param ordinal Backend-local device ordinal.
     * @param event Event to observe.
     * @return true when the event becomes ready within 30 seconds.
     */
    bool awaitEvent(IBackend *backend, int ordinal, void *event)
    {
        if (!backend || !event)
            return false;
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (std::chrono::steady_clock::now() < deadline)
        {
            bool ready = false;
            if (!backend->queryEvent(event, ordinal, &ready))
                return false;
            if (ready)
                return true;
            std::this_thread::yield();
        }
        return false;
    }

    /**
     * @brief Own one backend stream, payload, and independent sentinel event.
     *
     * Members are allocated during test setup, before any lifecycle boundary.
     * Destruction occurs after LocalTP releases its own persistent tickets.
     */
    class DeviceTicketFixture
    {
    public:
        /** @brief Construct an empty fixture for one backend device ordinal. */
        DeviceTicketFixture(IBackend *backend, int ordinal)
            : backend_(backend), ordinal_(ordinal)
        {
        }

        /** @brief Release every owned backend resource. */
        ~DeviceTicketFixture()
        {
            if (!backend_)
                return;
            if (sentinel_)
                backend_->destroyEvent(sentinel_, ordinal_);
            if (payload_)
                backend_->free(payload_, ordinal_);
            if (stream_)
                backend_->destroyStream(stream_, ordinal_);
        }

        DeviceTicketFixture(const DeviceTicketFixture &) = delete;
        DeviceTicketFixture &operator=(const DeviceTicketFixture &) = delete;

        /**
         * @brief Allocate persistent resources used by every test generation.
         * @return true only when stream, payload, and event are all available.
         */
        bool initialize()
        {
            if (!backend_)
                return false;
            stream_ = backend_->createStream(ordinal_);
            payload_ = backend_->allocate(kPayloadBytes, ordinal_);
            sentinel_ = backend_->createEvent(ordinal_);
            return stream_ && payload_ && sentinel_;
        }

        /**
         * @brief Queue real stream work followed by an independently queryable event.
         * @param byte_value Deterministic byte written to the payload.
         * @return true when both operations were accepted on the exact stream.
         */
        bool enqueuePreBoundaryWork(int byte_value)
        {
            return backend_->memset(
                       payload_,
                       byte_value,
                       kPayloadBytes,
                       ordinal_,
                       stream_) &&
                   backend_->recordEvent(sentinel_, ordinal_, stream_);
        }

        /**
         * @brief Query whether all work queued before the sentinel completed.
         * @param ready Destination for the nonblocking query result.
         * @return true when the backend query itself succeeds.
         */
        bool querySentinel(bool *ready) const
        {
            return backend_->queryEvent(sentinel_, ordinal_, ready);
        }

        /**
         * @brief Wait for the independently recorded sentinel with the test's
         *        bounded, nonblocking event-query loop.
         * @return true when all compute-stream work preceding the sentinel
         *         completes before the integration timeout.
         */
        bool awaitSentinel() const
        {
            return awaitEvent(backend_, ordinal_, sentinel_);
        }

        /** @brief Return the exact stream passed to LocalTP. */
        void *stream() const noexcept { return stream_; }

        /** @brief Return the owned device payload. */
        void *payload() const noexcept { return payload_; }

        /** @brief Return the payload capacity used by integration tests. */
        static constexpr size_t payloadBytes() noexcept
        {
            return kPayloadBytes;
        }

        /**
         * @brief Enqueue a host payload upload on the exact compute stream.
         * @param source Host input buffer.
         * @param bytes Bytes to upload, bounded by payloadBytes().
         * @return true when the upload was enqueued.
         */
        bool enqueueUpload(const void *source, size_t bytes)
        {
            return source && bytes <= kPayloadBytes &&
                   backend_->hostToDeviceOnStream(
                       payload_, source, bytes, ordinal_, stream_);
        }

        /** @brief Record the sentinel behind all currently queued compute work. */
        bool recordSentinel()
        {
            return backend_->recordEvent(sentinel_, ordinal_, stream_);
        }

        /**
         * @brief Read the device payload after the sentinel proves completion.
         * @param destination Host destination.
         * @param bytes Bytes to read, bounded by payloadBytes().
         * @return true when the copy completes.
         */
        bool readBack(void *destination, size_t bytes)
        {
            return destination && bytes <= kPayloadBytes &&
                   backend_->deviceToHost(
                       destination,
                       payload_,
                       bytes,
                       ordinal_,
                       stream_);
        }

    private:
        // Deliberately exceeds the bridge's 8 MiB chunk size so the collective
        // proof exercises double-buffered segmented transfer.
        static constexpr size_t kPayloadBytes = 9U * 1024U * 1024U;

        IBackend *backend_ = nullptr; ///< Borrowed process backend.
        int ordinal_ = -1;            ///< Backend-local device ordinal.
        void *stream_ = nullptr;       ///< Owned non-default stream.
        void *payload_ = nullptr;      ///< Owned device payload.
        void *sentinel_ = nullptr;     ///< Owned observation event.
    };

    /**
     * @brief Own a device timeline signal and an independent release stream.
     *
     * The gate lets the test keep one producer event incomplete while calling
     * the production asynchronous collective. A successful call must return
     * before release(), proving payload progress did not run inline.
     */
    class DeviceTimelineGate
    {
    public:
        /** @brief Bind an empty gate to one backend device. */
        DeviceTimelineGate(IBackend *backend, int ordinal)
            : backend_(backend), ordinal_(ordinal)
        {
        }

        /** @brief Drain and release the stream, event, and timeline signal. */
        ~DeviceTimelineGate()
        {
            if (!backend_)
                return;
            // An assertion after installWait() must not strand the compute
            // stream. Publish the exact armed generation before any resource
            // teardown so failure paths remain bounded and diagnosable.
            if (armed_)
                (void)release(armed_generation_);
            if (stream_ && drain_event_)
            {
                (void)backend_->recordEvent(drain_event_, ordinal_, stream_);
                (void)awaitEvent(backend_, ordinal_, drain_event_);
            }
            if (signal_)
                backend_->freeStreamTimelineSignal32(signal_, ordinal_);
            if (drain_event_)
                backend_->destroyEvent(drain_event_, ordinal_);
            if (stream_)
                backend_->destroyStream(stream_, ordinal_);
        }

        DeviceTimelineGate(const DeviceTimelineGate &) = delete;
        DeviceTimelineGate &operator=(const DeviceTimelineGate &) = delete;

        /**
         * @brief Allocate and initialize the gate to generation zero.
         * @return true when the device has proved the zero publication complete.
         */
        bool initialize()
        {
            if (!backend_ ||
                !backend_->supportsStreamTimelineSignal32(ordinal_))
            {
                return false;
            }
            stream_ = backend_->createStream(ordinal_);
            signal_ = backend_->allocateStreamTimelineSignal32(ordinal_);
            drain_event_ = backend_->createEvent(ordinal_);
            return stream_ && signal_ && drain_event_ &&
                   backend_->streamPublishTimelineSignal32(
                       stream_, signal_, 0, ordinal_) &&
                   backend_->recordEvent(
                       drain_event_, ordinal_, stream_) &&
                   awaitEvent(backend_, ordinal_, drain_event_);
        }

        /**
         * @brief Gate later work on an exact compute stream.
         * @param compute_stream Stream whose later operations must wait.
         * @param generation Future generation to await.
         * @return true when the wait was enqueued.
         */
        bool installWait(void *compute_stream, uint32_t generation)
        {
            if (armed_ || generation == 0)
                return false;
            const bool installed = backend_->streamWaitTimelineSignal32(
                compute_stream,
                signal_,
                generation,
                ordinal_);
            if (installed)
            {
                armed_ = true;
                armed_generation_ = generation;
            }
            return installed;
        }

        /**
         * @brief Release a previously installed future wait.
         * @param generation Generation to publish on the independent stream.
         * @return true when publication was enqueued.
         */
        bool release(uint32_t generation)
        {
            if (!armed_ || generation != armed_generation_)
                return false;
            const bool published = backend_->streamPublishTimelineSignal32(
                stream_,
                signal_,
                generation,
                ordinal_);
            if (published)
            {
                armed_ = false;
                armed_generation_ = 0;
            }
            return published;
        }

    private:
        IBackend *backend_ = nullptr; ///< Borrowed process backend.
        int ordinal_ = -1;            ///< Backend-local owner ordinal.
        void *stream_ = nullptr;       ///< Owned release stream.
        void *signal_ = nullptr;       ///< Owned device timeline word.
        void *drain_event_ = nullptr;  ///< Owned cleanup observation event.
        bool armed_ = false;            ///< True while a compute wait is live.
        uint32_t armed_generation_ = 0; ///< Generation needed to unblock it.
    };

    /**
     * @brief Own a small persistent side-stream pool that perturbs queue mapping.
     *
     * Production FusedQKV creates several concurrent projection streams before
     * heterogeneous collective resources are reserved. On gfx906, HIP may map
     * one of those streams, the graph stream, and the later transfer stream onto
     * shared HSA queues. This fixture reproduces that creation order and keeps
     * real work outstanding while the collective ticket advances.
     */
    class ConcurrentStreamPoolFixture
    {
    public:
        /** @brief Bind an empty pool to one backend-local device ordinal. */
        ConcurrentStreamPoolFixture(IBackend *backend, int ordinal)
            : backend_(backend), ordinal_(ordinal)
        {
        }

        /** @brief Drain and release all pool-owned resources. */
        ~ConcurrentStreamPoolFixture()
        {
            if (!backend_)
                return;
            for (size_t index = 0; index < streams_.size(); ++index)
            {
                if (events_[index])
                {
                    (void)backend_->recordEvent(
                        events_[index], ordinal_, streams_[index]);
                    (void)awaitEvent(backend_, ordinal_, events_[index]);
                    backend_->destroyEvent(events_[index], ordinal_);
                }
                if (payloads_[index])
                    backend_->free(payloads_[index], ordinal_);
                if (streams_[index])
                    backend_->destroyStream(streams_[index], ordinal_);
            }
        }

        ConcurrentStreamPoolFixture(const ConcurrentStreamPoolFixture &) = delete;
        ConcurrentStreamPoolFixture &operator=(const ConcurrentStreamPoolFixture &) = delete;

        /**
         * @brief Allocate streams in the same lifecycle position as FusedQKV.
         * @param stream_count Number of independently owned side streams.
         * @return true when every stream, payload, and event is available.
         */
        bool initialize(size_t stream_count)
        {
            if (!backend_ || stream_count == 0)
                return false;
            streams_.reserve(stream_count);
            payloads_.reserve(stream_count);
            events_.reserve(stream_count);
            for (size_t index = 0; index < stream_count; ++index)
            {
                void *const stream = backend_->createStream(ordinal_);
                void *const payload = backend_->allocate(kPayloadBytes, ordinal_);
                void *const event = backend_->createEvent(ordinal_);
                streams_.push_back(stream);
                payloads_.push_back(payload);
                events_.push_back(event);
                if (!stream || !payload || !event)
                    return false;
            }
            return true;
        }

        /**
         * @brief Queue a burst on every side stream without host synchronization.
         * @return true when all work and terminal events were accepted.
         */
        bool enqueueBurst()
        {
            for (size_t index = 0; index < streams_.size(); ++index)
            {
                for (int round = 0; round < kRounds; ++round)
                {
                    if (!backend_->memset(
                            payloads_[index],
                            static_cast<int>((index + round) & 0xffU),
                            kPayloadBytes,
                            ordinal_,
                            streams_[index]))
                    {
                        return false;
                    }
                }
                if (!backend_->recordEvent(
                        events_[index], ordinal_, streams_[index]))
                {
                    return false;
                }
            }
            return true;
        }

        /** @brief Prove every independently queued side-stream burst completed. */
        bool awaitAll() const
        {
            for (void *event : events_)
            {
                if (!awaitEvent(backend_, ordinal_, event))
                    return false;
            }
            return true;
        }

    private:
        static constexpr size_t kPayloadBytes = 256U * 1024U;
        static constexpr int kRounds = 16;

        IBackend *backend_ = nullptr; ///< Borrowed process backend.
        int ordinal_ = -1;            ///< Backend-local device ordinal.
        std::vector<void *> streams_; ///< Owned non-default side streams.
        std::vector<void *> payloads_; ///< Owned device work buffers.
        std::vector<void *> events_;  ///< Owned terminal observation events.
    };
}

/**
 * @test Real mixed-vendor tickets drain exact streams and reuse generations.
 */
TEST(Test__LocalTPHeterogeneousGraphCaptureTicket,
     MixedVendorLifecycleTicketsDrainExactStreamsAndReuseGenerations)
{
    IBackend *const cuda_backend = getCUDABackend();
    IBackend *const rocm_backend = getROCmBackend();
    ASSERT_NE(cuda_backend, nullptr);
    ASSERT_NE(rocm_backend, nullptr);
    if (cuda_backend->deviceCount() < 1 || rocm_backend->deviceCount() < 1)
    {
        GTEST_SKIP() << "Requires at least one CUDA and one ROCm device";
    }

    DeviceTicketFixture cuda(cuda_backend, 0);
    DeviceTicketFixture rocm(rocm_backend, 0);
    ASSERT_TRUE(cuda.initialize());
    ASSERT_TRUE(rocm.initialize());

    auto local_tp = createLocalTPContext(
        {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)},
        {},
        CollectiveBackendType::HETEROGENEOUS);
    ASSERT_NE(local_tp, nullptr);

    constexpr int kTimeoutMs = 30000;
    constexpr int kGenerations = 3;
    for (int generation = 0; generation < kGenerations; ++generation)
    {
        for (const char *const transition : {"capture_begin", "capture_end"})
        {
            ASSERT_TRUE(cuda.enqueuePreBoundaryWork(0x20 + generation));
            ASSERT_TRUE(rocm.enqueuePreBoundaryWork(0x40 + generation));

            const std::string boundary =
                "prefill_graph:integration_" + std::string(transition) +
                ":generation=" + std::to_string(generation);
            bool cuda_ok = false;
            bool rocm_ok = false;
            std::thread cuda_worker([&]() {
                cuda_ok = local_tp->graphCaptureBoundaryOnStream(
                    boundary,
                    0,
                    cuda.stream(),
                    kTimeoutMs);
            });
            std::thread rocm_worker([&]() {
                rocm_ok = local_tp->graphCaptureBoundaryOnStream(
                    boundary,
                    1,
                    rocm.stream(),
                    kTimeoutMs);
            });
            cuda_worker.join();
            rocm_worker.join();

            ASSERT_TRUE(cuda_ok);
            ASSERT_TRUE(rocm_ok);
            bool cuda_ready = false;
            bool rocm_ready = false;
            ASSERT_TRUE(cuda.querySentinel(&cuda_ready));
            ASSERT_TRUE(rocm.querySentinel(&rocm_ready));
            EXPECT_TRUE(cuda_ready)
                << "LocalTP returned before CUDA's exact stream ticket completed";
            EXPECT_TRUE(rocm_ready)
                << "LocalTP returned before ROCm's exact stream ticket completed";
        }
    }

    double ticket_fence_count = 0.0;
    for (const auto &record : PerfStatsCollector::snapshot({"graph_capture"}))
    {
        if (record.domain == "graph_capture" &&
            record.name == "localtp_heterogeneous_ticket_boundary_fences")
        {
            ticket_fence_count += record.value;
            EXPECT_EQ(record.tags.at("authority"),
                      "host_observed_stream_ticket");
            EXPECT_EQ(record.tags.at("steady_state_replay"), "false");
        }
    }
    EXPECT_GE(ticket_fence_count,
              static_cast<double>(2 * 2 * kGenerations));
}

/**
 * @test The real CUDA/ROCm bridge streams multiple chunks in the background.
 */
TEST(Test__LocalTPHeterogeneousGraphCaptureTicket,
     MixedVendorAllreduceSurvivesConcurrentStreamPoolsAndPublishesExactResult)
{
    IBackend *const cuda_backend = getCUDABackend();
    IBackend *const rocm_backend = getROCmBackend();
    ASSERT_NE(cuda_backend, nullptr);
    ASSERT_NE(rocm_backend, nullptr);
    if (cuda_backend->deviceCount() < 1 || rocm_backend->deviceCount() < 1)
    {
        GTEST_SKIP() << "Requires at least one CUDA and one ROCm device";
    }

    DeviceTicketFixture cuda(cuda_backend, 0);
    DeviceTicketFixture rocm(rocm_backend, 0);
    DeviceTimelineGate gate(cuda_backend, 0);
    ConcurrentStreamPoolFixture cuda_pool(cuda_backend, 0);
    ConcurrentStreamPoolFixture rocm_pool(rocm_backend, 0);
    ASSERT_TRUE(cuda.initialize());
    ASSERT_TRUE(rocm.initialize());
    ASSERT_TRUE(gate.initialize());
    ASSERT_TRUE(cuda_pool.initialize(4));
    ASSERT_TRUE(rocm_pool.initialize(4));
    ASSERT_TRUE(cuda_pool.enqueueBurst());
    ASSERT_TRUE(rocm_pool.enqueueBurst());

    constexpr size_t bytes = DeviceTicketFixture::payloadBytes();
    constexpr size_t count = bytes / sizeof(float);
    std::vector<float> cuda_input(count);
    std::vector<float> rocm_input(count);
    for (size_t i = 0; i < count; ++i)
    {
        // Small integers keep the expected FP32 sum exactly representable.
        cuda_input[i] = static_cast<float>(i % 97U);
        rocm_input[i] = static_cast<float>(i % 31U);
    }
    ASSERT_TRUE(cuda.enqueueUpload(cuda_input.data(), bytes));
    ASSERT_TRUE(rocm.enqueueUpload(rocm_input.data(), bytes));

    HeterogeneousBackend backend;
    DeviceGroupBuilder builder;
    const DeviceGroup group = builder
                                  .setName("async_cuda_rocm_integration")
                                  .setScope(CollectiveScope::LOCAL)
                                  .addDevice(DeviceId::cuda(0))
                                  .addDevice(DeviceId::rocm(0))
                                  .setLocalRank(0)
                                  .build();
    ASSERT_TRUE(backend.initialize(group));
    ASSERT_TRUE(backend.reserveTempBufferBytes(bytes))
        << backend.lastError();
    ASSERT_FALSE(backend.supportsAllreduceMultiOnStreams())
        << "mixed-vendor stream waits are unsafe under shared ROCm queues";
    ASSERT_TRUE(
        backend.supportsAllreduceMultiOnStreamsWithHostCompletionTicket());
    ASSERT_TRUE(gate.installWait(cuda.stream(), 1));

    const auto submit_started = std::chrono::steady_clock::now();
    const auto ticket =
        backend.allreduceMultiOnStreamsWithHostCompletionTicket(
        {cuda.payload(), rocm.payload()},
        count,
        CollectiveDataType::FLOAT32,
        CollectiveOp::ALLREDUCE_SUM,
        {cuda.stream(), rocm.stream()});
    const auto submit_elapsed =
        std::chrono::steady_clock::now() - submit_started;

    // Always release the deliberate gate before an ASSERT can return from the
    // test body and strand either compute stream.
    const bool released = gate.release(1);
    ASSERT_TRUE(released);
    ASSERT_TRUE(ticket.has_value()) << backend.lastError();
    EXPECT_LT(submit_elapsed, std::chrono::milliseconds(250))
        << "submission performed payload transfer/reduction inline";
    ASSERT_TRUE(backend.awaitHostCompletionTicket(*ticket, 30000))
        << backend.lastError();
    ASSERT_TRUE(cuda_pool.awaitAll());
    ASSERT_TRUE(rocm_pool.awaitAll());

    ASSERT_TRUE(cuda.recordSentinel());
    ASSERT_TRUE(rocm.recordSentinel());
    ASSERT_TRUE(cuda.awaitSentinel());
    ASSERT_TRUE(rocm.awaitSentinel());

    std::vector<float> cuda_result(count);
    std::vector<float> rocm_result(count);
    ASSERT_TRUE(cuda.readBack(cuda_result.data(), bytes));
    ASSERT_TRUE(rocm.readBack(rocm_result.data(), bytes));
    for (size_t i = 0; i < count; ++i)
    {
        const float expected = cuda_input[i] + rocm_input[i];
        ASSERT_EQ(cuda_result[i], expected) << "CUDA mismatch at i=" << i;
        ASSERT_EQ(rocm_result[i], expected) << "ROCm mismatch at i=" << i;
    }

    double handed_off = 0.0;
    double chunks = 0.0;
    double d2h_bytes = 0.0;
    double h2d_bytes = 0.0;
    double observed_tickets = 0.0;
    double terminal_h2d_participants = 0.0;
    for (const auto &record :
         PerfStatsCollector::snapshot({"heterogeneous_collective"}))
    {
        if (record.domain != "heterogeneous_collective")
            continue;
        if (record.name == "async_transactions_handed_off")
            handed_off += record.value;
        else if (record.name == "streamed_chunks")
            chunks += record.value;
        else if (record.name == "d2h_bytes")
            d2h_bytes += record.value;
        else if (record.name == "h2d_bytes")
            h2d_bytes += record.value;
        else if (record.name == "completion_tickets_observed")
            observed_tickets += record.value;
        else if (record.name == "terminal_h2d_participants_observed")
            terminal_h2d_participants += record.value;
        EXPECT_EQ(record.tags.at("path"),
                  "async_chunked_host_bridge");
    }
    EXPECT_GE(handed_off, 1.0);
    EXPECT_GE(chunks, 2.0);
    EXPECT_GE(d2h_bytes, static_cast<double>(2 * bytes));
    EXPECT_GE(h2d_bytes, static_cast<double>(2 * bytes));
    EXPECT_GE(observed_tickets, 1.0);
    EXPECT_GE(terminal_h2d_participants, 2.0);

    backend.shutdown();
}
