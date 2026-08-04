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

#include "backends/rocm/HIPGraphCapture.h"
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

        /** @brief Own one production HIP graph capture and its executable. */
        class Graph
        {
        public:
            Graph() = default;

            explicit Graph(
                std::unique_ptr<HIPGraphCapture> capture)
                : capture_(std::move(capture))
            {
            }

            Graph(const Graph &) = delete;
            Graph &operator=(const Graph &) = delete;
            Graph(Graph &&) noexcept = default;
            Graph &operator=(Graph &&) noexcept = default;

            bool valid() const
            {
                return capture_ && capture_->hasExecutable();
            }

            bool launch(void *opaque_stream)
            {
                return valid() && opaque_stream &&
                       capture_->executionStream() == opaque_stream &&
                       capture_->launch();
            }

        private:
            std::unique_ptr<HIPGraphCapture> capture_;
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
            auto capture =
                std::make_unique<HIPGraphCapture>(stream);
            ScopedBackendGraphCapture capture_transaction(
                *capture,
                "ROCm grouped KV lifecycle graph");
            if (!capture_transaction.begin())
                return {};

            const bool enqueue_ok = enqueue();
            capture_transaction.finish();
            if (!enqueue_ok || !capture->instantiate())
                return {};
            return Graph(std::move(capture));
        }

        bool launchGraph(
            Graph &graph,
            void *opaque_stream)
        {
            return graph.launch(opaque_stream);
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

        HIPGraphCapture graph(stream);
        ScopedBackendGraphCapture capture_transaction(
            graph,
            "ROCm grouped KV append");
        if (!capture_transaction.begin())
        {
            if (device_logical_rows)
                (void)hipFree(device_logical_rows);
            return false;
        }
        const bool append_ok = cache.appendVerifierRowsDecodeEquivalent(
            0, 0, k, v, verifier_rows, opaque_stream);
        capture_transaction.finish();
        if (!append_ok || !graph.instantiate())
        {
            if (device_logical_rows)
                (void)hipFree(device_logical_rows);
            return false;
        }

        const bool replay_ok =
            graph.launch() &&
            hipStreamSynchronize(stream) == hipSuccess;
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
        if (!stream)
            return false;

        HIPGraphCapture graph(stream);
        ScopedBackendGraphCapture capture_transaction(
            graph,
            "ROCm grouped KV converted read");
        if (!capture_transaction.begin())
            return false;

        const bool read_ok = cache.get_kv_batched_converted_device_view(
            0, 0, request_count,
            ActivationPrecision::FP16, out_k, out_v, read);
        capture_transaction.finish();
        if (!read_ok || !graph.instantiate())
            return false;

        const bool replay_ok =
            graph.launch() &&
            hipStreamSynchronize(stream) == hipSuccess;
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

/**
 * @brief Run one independently process-isolated cache/source format matrix.
 *
 * A non-retiring HIP kernel can wedge its KFD process beyond ordinary signal
 * recovery. Process isolation therefore belongs at the format boundary: CTest
 * names the exact cache/source conversion before launch, and the shared
 * harness still proves every D/M/layout/topology cell for that format.
 */
void runRuntimePublicationFormat(size_t format_index)
{
    ASSERT_LT(format_index, kFormatCases.size());
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1)
        GTEST_SKIP() << "ROCm device unavailable";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedHipStream stream;
    ASSERT_NE(stream.get(), nullptr);
    runFormatGroupedVerifierSweep(
        DeviceId::rocm(0),
        "ROCm",
        "rocm_kv_cache_grouped_verifier_append_calls",
        kFormatCases[format_index],
        stream.opaque(),
        appendGrouped,
        observeDeviceState);
}

#define LLAMINAR_ROCM_KV_RUNTIME_FORMAT_TEST(suffix, index)                     \
    TEST(Test__ROCmKVCacheGroupedVerifier, RuntimePublication_##suffix)         \
    {                                                                            \
        runRuntimePublicationFormat(index);                                      \
    }

LLAMINAR_ROCM_KV_RUNTIME_FORMAT_TEST(FP32_From_FP32, 0)
LLAMINAR_ROCM_KV_RUNTIME_FORMAT_TEST(BF16_From_BF16, 1)
LLAMINAR_ROCM_KV_RUNTIME_FORMAT_TEST(FP16_From_FP32, 2)
LLAMINAR_ROCM_KV_RUNTIME_FORMAT_TEST(FP16_From_FP16, 3)
LLAMINAR_ROCM_KV_RUNTIME_FORMAT_TEST(FP16_From_BF16, 4)
LLAMINAR_ROCM_KV_RUNTIME_FORMAT_TEST(FP16_From_Q8_1, 5)
LLAMINAR_ROCM_KV_RUNTIME_FORMAT_TEST(Q8_1_From_FP32, 6)
LLAMINAR_ROCM_KV_RUNTIME_FORMAT_TEST(Q8_1_From_FP16, 7)
LLAMINAR_ROCM_KV_RUNTIME_FORMAT_TEST(Q8_1_From_BF16, 8)
LLAMINAR_ROCM_KV_RUNTIME_FORMAT_TEST(Q8_1_From_Q8_1, 9)
LLAMINAR_ROCM_KV_RUNTIME_FORMAT_TEST(TQ8K_TQ4V_From_FP32, 10)
LLAMINAR_ROCM_KV_RUNTIME_FORMAT_TEST(TQ8K_TQ4V_From_TQ8_TQ4, 11)
LLAMINAR_ROCM_KV_RUNTIME_FORMAT_TEST(TQ8K_TQ8V_From_FP32, 12)
LLAMINAR_ROCM_KV_RUNTIME_FORMAT_TEST(TQ8K_TQ8V_From_TQ8, 13)

#undef LLAMINAR_ROCM_KV_RUNTIME_FORMAT_TEST

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
