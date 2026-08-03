/**
 * @file Test__ROCmKVCacheGroupedVerifier.cpp
 * @brief ROCm production-path sweep for byte-exact grouped MTP KV publication.
 *
 * The shared matrix is intentionally identical to CUDA.  This file contributes
 * only HIP availability, explicit-stream ownership, and graph capture/replay so
 * backend coverage cannot diverge through duplicated format tables.
 */

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "utils/GpuKVCacheGroupedVerifierHarness.h"

using namespace llaminar2;
using namespace llaminar2::test::gpu_kv_verifier;

namespace
{
    /** @brief Own the mandatory non-default HIP stream used by every operation. */
    class ScopedHipStream
    {
    public:
        ScopedHipStream()
        {
            EXPECT_EQ(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), hipSuccess);
        }

        ~ScopedHipStream()
        {
            if (stream_)
                (void)hipStreamDestroy(stream_);
        }

        hipStream_t get() const { return stream_; }
        void *opaque() const { return static_cast<void *>(stream_); }

    private:
        hipStream_t stream_ = nullptr;
    };

    /**
     * @brief HIP resource and ordering adapter for the shared KV lifecycle model.
     *
     * Only persistent setup allocation, asynchronous copies, stream events,
     * graph replay, and the final assertion fence are exposed.  Consequently
     * the shared stress loop cannot accidentally introduce device-wide
     * synchronization or per-transaction temporary allocation.
     */
    class ROCmKVLifecycleRuntime
    {
    public:
        /** @brief Own one persistent HIP device allocation. */
        class DeviceBuffer
        {
        public:
            DeviceBuffer() = default;

            explicit DeviceBuffer(size_t bytes)
                : bytes_(bytes)
            {
                if (bytes_ == 0 ||
                    hipMalloc(&pointer_, bytes_) != hipSuccess)
                {
                    pointer_ = nullptr;
                    bytes_ = 0;
                }
            }

            ~DeviceBuffer()
            {
                if (pointer_)
                    (void)hipFree(pointer_);
            }

            DeviceBuffer(const DeviceBuffer &) = delete;
            DeviceBuffer &operator=(const DeviceBuffer &) = delete;

            DeviceBuffer(DeviceBuffer &&other) noexcept
                : pointer_(std::exchange(other.pointer_, nullptr)),
                  bytes_(std::exchange(other.bytes_, 0))
            {
            }

            DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
            {
                if (this == &other)
                    return *this;
                if (pointer_)
                    (void)hipFree(pointer_);
                pointer_ = std::exchange(other.pointer_, nullptr);
                bytes_ = std::exchange(other.bytes_, 0);
                return *this;
            }

            bool valid() const { return pointer_ != nullptr; }
            void *data() const { return pointer_; }
            size_t size() const { return bytes_; }

        private:
            void *pointer_ = nullptr;
            size_t bytes_ = 0;
        };

        /** @brief Own one timing-disabled HIP dependency event. */
        class Event
        {
        public:
            Event()
            {
                if (hipEventCreateWithFlags(
                        &event_,
                        hipEventDisableTiming) != hipSuccess)
                {
                    event_ = nullptr;
                }
            }

            ~Event()
            {
                if (event_)
                    (void)hipEventDestroy(event_);
            }

            Event(const Event &) = delete;
            Event &operator=(const Event &) = delete;

            Event(Event &&other) noexcept
                : event_(
                      std::exchange(
                          other.event_,
                          nullptr))
            {
            }

            Event &operator=(Event &&other) noexcept
            {
                if (this == &other)
                    return *this;
                if (event_)
                    (void)hipEventDestroy(event_);
                event_ =
                    std::exchange(
                        other.event_,
                        nullptr);
                return *this;
            }

            bool valid() const { return event_ != nullptr; }
            hipEvent_t get() const { return event_; }

        private:
            hipEvent_t event_ = nullptr;
        };

        /** @brief Own one captured HIP graph and executable. */
        class Graph
        {
        public:
            Graph() = default;

            Graph(
                hipGraph_t graph,
                hipGraphExec_t executable)
                : graph_(graph),
                  executable_(executable)
            {
            }

            ~Graph()
            {
                if (executable_)
                    (void)hipGraphExecDestroy(
                        executable_);
                if (graph_)
                    (void)hipGraphDestroy(graph_);
            }

            Graph(const Graph &) = delete;
            Graph &operator=(const Graph &) = delete;

            Graph(Graph &&other) noexcept
                : graph_(
                      std::exchange(
                          other.graph_,
                          nullptr)),
                  executable_(
                      std::exchange(
                          other.executable_,
                          nullptr))
            {
            }

            Graph &operator=(Graph &&other) noexcept
            {
                if (this == &other)
                    return *this;
                if (executable_)
                    (void)hipGraphExecDestroy(
                        executable_);
                if (graph_)
                    (void)hipGraphDestroy(graph_);
                graph_ =
                    std::exchange(
                        other.graph_,
                        nullptr);
                executable_ =
                    std::exchange(
                        other.executable_,
                        nullptr);
                return *this;
            }

            bool valid() const
            {
                return graph_ != nullptr &&
                       executable_ != nullptr;
            }

            hipGraphExec_t executable() const
            {
                return executable_;
            }

        private:
            hipGraph_t graph_ = nullptr;
            hipGraphExec_t executable_ = nullptr;
        };

        DeviceBuffer allocateDeviceBuffer(size_t bytes)
        {
            return DeviceBuffer(bytes);
        }

        Event createEvent()
        {
            return Event();
        }

        bool copyHostToDeviceAsync(
            void *destination,
            const void *source,
            size_t bytes,
            void *opaque_stream)
        {
            return destination && source && opaque_stream &&
                   hipMemcpyAsync(
                       destination,
                       source,
                       bytes,
                       hipMemcpyHostToDevice,
                       static_cast<hipStream_t>(
                           opaque_stream)) ==
                       hipSuccess;
        }

        bool copyDeviceToHostAsync(
            void *destination,
            const void *source,
            size_t bytes,
            void *opaque_stream)
        {
            return destination && source && opaque_stream &&
                   hipMemcpyAsync(
                       destination,
                       source,
                       bytes,
                       hipMemcpyDeviceToHost,
                       static_cast<hipStream_t>(
                           opaque_stream)) ==
                       hipSuccess;
        }

        bool copyDeviceToDeviceAsync(
            void *destination,
            const void *source,
            size_t bytes,
            void *opaque_stream)
        {
            return destination && source && opaque_stream &&
                   hipMemcpyAsync(
                       destination,
                       source,
                       bytes,
                       hipMemcpyDeviceToDevice,
                       static_cast<hipStream_t>(
                           opaque_stream)) ==
                       hipSuccess;
        }

        bool recordEvent(
            const Event &event,
            void *opaque_stream)
        {
            return event.valid() && opaque_stream &&
                   hipEventRecord(
                       event.get(),
                       static_cast<hipStream_t>(
                           opaque_stream)) ==
                       hipSuccess;
        }

        bool waitEvent(
            void *opaque_stream,
            const Event &event)
        {
            return event.valid() && opaque_stream &&
                   hipStreamWaitEvent(
                       static_cast<hipStream_t>(
                           opaque_stream),
                       event.get(),
                       0) == hipSuccess;
        }

        Graph captureGraph(
            void *opaque_stream,
            const std::function<bool()> &enqueue)
        {
            if (!opaque_stream || !enqueue)
                return {};
            const auto stream =
                static_cast<hipStream_t>(
                    opaque_stream);
            if (hipStreamBeginCapture(
                    stream,
                    hipStreamCaptureModeGlobal) !=
                hipSuccess)
            {
                return {};
            }

            bool enqueue_ok = false;
            {
                GraphCaptureGuard guard;
                enqueue_ok = enqueue();
            }
            hipGraph_t graph = nullptr;
            const hipError_t end_status =
                hipStreamEndCapture(
                    stream,
                    &graph);
            if (!enqueue_ok ||
                end_status != hipSuccess ||
                !graph)
            {
                if (graph)
                    (void)hipGraphDestroy(graph);
                return {};
            }

            hipGraphExec_t executable = nullptr;
            if (hipGraphInstantiate(
                    &executable,
                    graph,
                    nullptr,
                    nullptr,
                    0) != hipSuccess ||
                !executable)
            {
                (void)hipGraphDestroy(graph);
                return {};
            }
            return Graph(graph, executable);
        }

        bool launchGraph(
            const Graph &graph,
            void *opaque_stream)
        {
            return graph.valid() && opaque_stream &&
                   hipGraphLaunch(
                       graph.executable(),
                       static_cast<hipStream_t>(
                           opaque_stream)) ==
                       hipSuccess;
        }

        bool synchronizeStream(void *opaque_stream)
        {
            return opaque_stream &&
                   hipStreamSynchronize(
                       static_cast<hipStream_t>(
                           opaque_stream)) ==
                       hipSuccess;
        }
    };

    /**
     * @brief Execute one grouped publication entirely inside a HIP graph.
     *
     * The graph owns both payload publication and canonical device metadata
     * advancement. A positive @p logical_rows is staged into one persistent
     * device scalar before capture and bound through the production padded-row
     * API; no host sequence-state copy participates in graph execution.
     */
    bool appendGrouped(
        IKVCache &cache,
        const ITensor *k,
        const ITensor *v,
        int verifier_rows,
        int logical_rows,
        void *opaque_stream)
    {
        const auto stream = static_cast<hipStream_t>(opaque_stream);
        if (!stream || !cache.isGraphCaptureReady())
            return false;

        int32_t *device_logical_rows = nullptr;
        if (logical_rows > 0)
        {
            if (logical_rows > verifier_rows ||
                hipMalloc(&device_logical_rows, sizeof(int32_t)) != hipSuccess ||
                hipMemcpyAsync(
                    device_logical_rows,
                    &logical_rows,
                    sizeof(int32_t),
                    hipMemcpyHostToDevice,
                    stream) != hipSuccess ||
                !cache.bindGraphAppendCountSource(
                    0,
                    0,
                    device_logical_rows,
                    verifier_rows,
                    opaque_stream))
            {
                if (device_logical_rows)
                    (void)hipFree(device_logical_rows);
                return false;
            }
        }

        hipGraph_t graph = nullptr;
        hipGraphExec_t executable = nullptr;
        if (hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal) != hipSuccess)
        {
            if (device_logical_rows)
                (void)hipFree(device_logical_rows);
            return false;
        }
        bool append_ok = false;
        {
            GraphCaptureGuard guard;
            append_ok = cache.appendVerifierRowsDecodeEquivalent(
                0, 0, k, v, verifier_rows, opaque_stream);
        }
        const hipError_t end_status = hipStreamEndCapture(stream, &graph);
        if (!append_ok || end_status != hipSuccess || !graph)
        {
            if (graph)
                (void)hipGraphDestroy(graph);
            if (device_logical_rows)
                (void)hipFree(device_logical_rows);
            return false;
        }
        if (hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0) != hipSuccess ||
            !executable)
        {
            (void)hipGraphDestroy(graph);
            if (device_logical_rows)
                (void)hipFree(device_logical_rows);
            return false;
        }

        const bool replay_ok =
            hipGraphLaunch(executable, stream) == hipSuccess &&
            hipStreamSynchronize(stream) == hipSuccess;
        (void)hipGraphExecDestroy(executable);
        (void)hipGraphDestroy(graph);
        if (device_logical_rows)
        {
            const bool unbound = cache.bindGraphAppendCountSource(
                0, 0, nullptr, verifier_rows, opaque_stream);
            (void)hipFree(device_logical_rows);
            return replay_ok && unbound;
        }
        return replay_ok;
    }

    /**
     * @brief Observe canonical ROCm sequence metadata after a stream fence.
     *
     * The copied values live only long enough for assertions and therefore do
     * not become a second owner of sequence state.
     */
    bool observeDeviceState(
        const IKVCache &cache,
        int max_seq_len,
        void *opaque_stream,
        IKVCache::KVCacheSequenceState *state)
    {
        const auto stream = static_cast<hipStream_t>(opaque_stream);
        const int *device_head = cache.deviceRingHeadPtr(0, 0);
        const int *device_count = cache.deviceCachedTokenCountPtr(0, 0);
        if (!stream || !state || !device_head || !device_count)
            return false;

        int head = 0;
        int count = 0;
        if (hipMemcpyAsync(
                &head, device_head, sizeof(head), hipMemcpyDeviceToHost, stream) != hipSuccess ||
            hipMemcpyAsync(
                &count, device_count, sizeof(count), hipMemcpyDeviceToHost, stream) != hipSuccess ||
            hipStreamSynchronize(stream) != hipSuccess)
        {
            return false;
        }
        *state = IKVCache::KVCacheSequenceState{
            .cached_tokens = count,
            .implementation_head = head,
            .wrapped = count == max_seq_len,
        };
        return true;
    }

    /** @brief Capture and replay the production converted request-batch read. */
    bool readConvertedBatch(
        IKVCache &cache,
        int request_count,
        int max_kv_len,
        const IKVCache::KVReadParams &read,
        ITensor **out_k,
        ITensor **out_v,
        void *opaque_stream)
    {
        const auto stream = static_cast<hipStream_t>(opaque_stream);
        hipGraph_t graph = nullptr;
        hipGraphExec_t executable = nullptr;
        if (!stream ||
            hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal) != hipSuccess)
        {
            return false;
        }
        bool read_ok = false;
        {
            GraphCaptureGuard guard;
            read_ok = cache.get_kv_batched_converted_device_view(
                0, 0, request_count,
                ActivationPrecision::FP16, out_k, out_v, read);
        }
        const hipError_t end_status = hipStreamEndCapture(stream, &graph);
        if (!read_ok || end_status != hipSuccess || !graph ||
            hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0) != hipSuccess ||
            !executable)
        {
            if (executable)
                (void)hipGraphExecDestroy(executable);
            if (graph)
                (void)hipGraphDestroy(graph);
            return false;
        }
        const bool replay_ok =
            hipGraphLaunch(executable, stream) == hipSuccess &&
            hipStreamSynchronize(stream) == hipSuccess;
        (void)hipGraphExecDestroy(executable);
        (void)hipGraphDestroy(graph);
        return replay_ok;
    }

    /** @brief Integration-boundary device-to-host byte observation. */
    bool copyDeviceBytes(void *destination, const void *source, size_t bytes, void *)
    {
        return hipMemcpy(destination, source, bytes, hipMemcpyDeviceToHost) == hipSuccess;
    }

    /** @brief Allocate a HIP assertion buffer outside production cache code. */
    void *allocateDeviceBytes(size_t bytes)
    {
        void *pointer = nullptr;
        return hipMalloc(&pointer, bytes) == hipSuccess ? pointer : nullptr;
    }

    /** @brief Release a HIP assertion buffer after the matrix cell completes. */
    void releaseDeviceBytes(void *pointer)
    {
        if (pointer)
            (void)hipFree(pointer);
    }

    /** @brief Enqueue the suite's final device-to-host observation copy. */
    bool copyDeviceBytesAsync(
        void *destination,
        const void *source,
        size_t bytes,
        void *opaque_stream)
    {
        return opaque_stream &&
               hipMemcpyAsync(
                   destination, source, bytes, hipMemcpyDeviceToHost,
                   static_cast<hipStream_t>(opaque_stream)) == hipSuccess;
    }

    bool synchronizeStream(void *opaque_stream)
    {
        return opaque_stream &&
               hipStreamSynchronize(static_cast<hipStream_t>(opaque_stream)) == hipSuccess;
    }
} // namespace

TEST(Test__ROCmKVCacheGroupedVerifier,
     AllFormatsRuntimeMGraphCapturedReplicatedAndLocalTPMatchSerialDecodeBytes)
{
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1)
        GTEST_SKIP() << "ROCm device unavailable";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedHipStream stream;
    ASSERT_NE(stream.get(), nullptr);
    runAllFormatGroupedVerifierSweep(
        DeviceId::rocm(0),
        "ROCm",
        "rocm_kv_cache_grouped_verifier_append_calls",
        stream.opaque(),
        appendGrouped,
        observeDeviceState);
}

TEST(Test__ROCmKVCacheGroupedVerifier,
     AllFormatsCapturedConvertedBatchReadMatchesSerialBytes)
{
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1)
        GTEST_SKIP() << "ROCm device unavailable";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedHipStream stream;
    ASSERT_NE(stream.get(), nullptr);
    runAllFormatConvertedDeviceReadSweep(
        DeviceId::rocm(0), "ROCm", stream.opaque(),
        readConvertedBatch, copyDeviceBytes, synchronizeStream);
}

TEST(Test__ROCmKVCacheGroupedVerifier,
     AllNativeFormatsDeviceLogicalBlockHarvestRestoreAreByteExact)
{
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1)
        GTEST_SKIP() << "ROCm device unavailable";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedHipStream stream;
    ASSERT_NE(stream.get(), nullptr);
    runAllFormatDeviceLogicalBlockSweep(
        DeviceId::rocm(0),
        "ROCm",
        "rocm_device_logical_kv_exports",
        "rocm_device_logical_kv_imports",
        "rocm_device_logical_tq_kv_exports",
        "rocm_device_logical_tq_kv_imports",
        stream.opaque(),
        allocateDeviceBytes,
        releaseDeviceBytes,
        copyDeviceBytesAsync,
        synchronizeStream);
}

/**
 * @brief Test the complete captured MTP/main/prefix lifecycle under contention.
 */
TEST(Test__ROCmKVCacheGroupedVerifier,
     AdversarialMultiStreamGraphReusePrefixRestoreMatchesSerialState)
{
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess ||
        device_count < 1)
    {
        GTEST_SKIP() << "ROCm device unavailable";
    }
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedHipStream producer_stream;
    ScopedHipStream graph_stream;
    ScopedHipStream observer_stream;
    ASSERT_NE(producer_stream.get(), nullptr);
    ASSERT_NE(graph_stream.get(), nullptr);
    ASSERT_NE(observer_stream.get(), nullptr);

    ROCmKVLifecycleRuntime runtime;
    runAdversarialKVLifecycleStress(
        DeviceId::rocm(0),
        "ROCm",
        ActivationPrecision::FP32,
        "FP32",
        producer_stream.opaque(),
        graph_stream.opaque(),
        observer_stream.opaque(),
        runtime);
}

/**
 * @brief Stress both TurboQuant policies through captured MTP publication.
 *
 * TQ8-K/TQ4-V and TQ8-K/TQ8-V share canonical sequence metadata but own
 * different grouped quantization and logical-block paths. This test subjects
 * both real cache implementations to graph reuse, wraparound, prefix restore,
 * and explicit producer/consumer event handoffs.
 */
TEST(Test__ROCmKVCacheGroupedVerifier,
     AdversarialTurboQuantMultiStreamGraphReuseMatchesSerialState)
{
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess ||
        device_count < 1)
    {
        GTEST_SKIP() << "ROCm device unavailable";
    }
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedHipStream producer_stream;
    ScopedHipStream graph_stream;
    ScopedHipStream observer_stream;
    ROCmKVLifecycleRuntime runtime;

    for (const auto &[precision, label] :
         std::array<std::pair<ActivationPrecision, const char *>, 2>{{
             {ActivationPrecision::TQ4, "TQ8-K/TQ4-V"},
             {ActivationPrecision::TQ8, "TQ8-K/TQ8-V"},
         }})
    {
        SCOPED_TRACE(label);
        runAdversarialKVLifecycleStress(
            DeviceId::rocm(0),
            "ROCm",
            precision,
            label,
            producer_stream.opaque(),
            graph_stream.opaque(),
            observer_stream.opaque(),
            runtime);
    }
}
