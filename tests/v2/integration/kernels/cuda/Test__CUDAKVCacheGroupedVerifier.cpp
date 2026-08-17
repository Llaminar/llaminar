/**
 * @file Test__CUDAKVCacheGroupedVerifier.cpp
 * @brief CUDA production-path sweep for byte-exact grouped MTP KV publication.
 *
 * This integration suite owns CUDA graph mechanics while the shared harness
 * owns the canonical format/depth/layout/topology matrix.  It never substitutes
 * host publication, row replay, or a deterministic-only diagnostic kernel for
 * the production cache APIs.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "backends/cuda/CUDAGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "utils/GpuKVCacheGroupedVerifierHarness.h"

using namespace llaminar2;
using namespace llaminar2::test::gpu_kv_verifier;

namespace
{
    /** @brief Own the mandatory non-default CUDA stream used by every operation. */
    class ScopedCudaStream
    {
    public:
        ScopedCudaStream()
        {
            EXPECT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
        }

        ~ScopedCudaStream()
        {
            if (stream_)
                (void)cudaStreamDestroy(stream_);
        }

        cudaStream_t get() const { return stream_; }
        void *opaque() const { return static_cast<void *>(stream_); }

    private:
        cudaStream_t stream_ = nullptr;
    };

    /**
     * @brief CUDA resource and ordering adapter for the shared KV lifecycle model.
     *
     * The adapter deliberately exposes only asynchronous copy, event, graph,
     * and final stream-fence primitives.  The shared stress driver cannot call
     * a device-wide synchronization or allocate temporary device storage while
     * graph replay is active because neither operation exists in this API.
     */
    class CUDAKVLifecycleRuntime
    {
    public:
        /** @brief Own one persistent CUDA device allocation. */
        class DeviceBuffer
        {
        public:
            DeviceBuffer() = default;

            explicit DeviceBuffer(size_t bytes)
                : bytes_(bytes)
            {
                if (bytes_ == 0 ||
                    cudaMalloc(&pointer_, bytes_) != cudaSuccess)
                {
                    pointer_ = nullptr;
                    bytes_ = 0;
                }
            }

            ~DeviceBuffer()
            {
                if (pointer_)
                    (void)cudaFree(pointer_);
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
                    (void)cudaFree(pointer_);
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

        /** @brief Own one timing-disabled CUDA dependency event. */
        class Event
        {
        public:
            Event()
            {
                if (cudaEventCreateWithFlags(
                        &event_,
                        cudaEventDisableTiming) != cudaSuccess)
                {
                    event_ = nullptr;
                }
            }

            ~Event()
            {
                if (event_)
                    (void)cudaEventDestroy(event_);
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
                    (void)cudaEventDestroy(event_);
                event_ =
                    std::exchange(
                        other.event_,
                        nullptr);
                return *this;
            }

            bool valid() const { return event_ != nullptr; }
            cudaEvent_t get() const { return event_; }

        private:
            cudaEvent_t event_ = nullptr;
        };

        /** @brief Own one production CUDA graph capture and its executable. */
        class Graph
        {
        public:
            Graph() = default;

            explicit Graph(
                std::unique_ptr<CUDAGraphCapture> capture)
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
            std::unique_ptr<CUDAGraphCapture> capture_;
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
                   cudaMemcpyAsync(
                       destination,
                       source,
                       bytes,
                       cudaMemcpyHostToDevice,
                       static_cast<cudaStream_t>(
                           opaque_stream)) ==
                       cudaSuccess;
        }

        bool copyDeviceToHostAsync(
            void *destination,
            const void *source,
            size_t bytes,
            void *opaque_stream)
        {
            return destination && source && opaque_stream &&
                   cudaMemcpyAsync(
                       destination,
                       source,
                       bytes,
                       cudaMemcpyDeviceToHost,
                       static_cast<cudaStream_t>(
                           opaque_stream)) ==
                       cudaSuccess;
        }

        bool copyDeviceToDeviceAsync(
            void *destination,
            const void *source,
            size_t bytes,
            void *opaque_stream)
        {
            return destination && source && opaque_stream &&
                   cudaMemcpyAsync(
                       destination,
                       source,
                       bytes,
                       cudaMemcpyDeviceToDevice,
                       static_cast<cudaStream_t>(
                           opaque_stream)) ==
                       cudaSuccess;
        }

        bool recordEvent(
            const Event &event,
            void *opaque_stream)
        {
            return event.valid() && opaque_stream &&
                   cudaEventRecord(
                       event.get(),
                       static_cast<cudaStream_t>(
                           opaque_stream)) ==
                       cudaSuccess;
        }

        bool waitEvent(
            void *opaque_stream,
            const Event &event)
        {
            return event.valid() && opaque_stream &&
                   cudaStreamWaitEvent(
                       static_cast<cudaStream_t>(
                           opaque_stream),
                       event.get(),
                       0) == cudaSuccess;
        }

        Graph captureGraph(
            void *opaque_stream,
            const std::function<bool()> &enqueue)
        {
            if (!opaque_stream || !enqueue)
                return {};
            const auto stream =
                static_cast<cudaStream_t>(
                    opaque_stream);
            auto capture =
                std::make_unique<CUDAGraphCapture>(stream, 0);
            ScopedBackendGraphCapture capture_transaction(
                *capture,
                "CUDA grouped KV lifecycle graph");
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
                   cudaStreamSynchronize(
                       static_cast<cudaStream_t>(
                           opaque_stream)) ==
                       cudaSuccess;
        }
    };

    /**
     * @brief Execute one grouped publication entirely inside a CUDA graph.
     *
     * The captured kernels read and advance the cache's canonical device head
     * and count. When @p logical_rows is positive, test setup publishes that
     * value into a persistent device scalar before capture and binds the scalar
     * through the same cache API used by production padded verifier graphs.
     * The captured graph itself contains no host transfer or state adoption.
     */
    bool appendGrouped(
        IKVCache &cache,
        const ITensor *k,
        const ITensor *v,
        int verifier_rows,
        int logical_rows,
        void *opaque_stream)
    {
        const auto stream = static_cast<cudaStream_t>(opaque_stream);
        if (!stream || !cache.isGraphCaptureReady())
            return false;

        int32_t *device_logical_rows = nullptr;
        if (logical_rows > 0)
        {
            if (logical_rows > verifier_rows ||
                cudaMalloc(&device_logical_rows, sizeof(int32_t)) != cudaSuccess ||
                cudaMemcpyAsync(
                    device_logical_rows,
                    &logical_rows,
                    sizeof(int32_t),
                    cudaMemcpyHostToDevice,
                    stream) != cudaSuccess ||
                !cache.bindGraphAppendCountSource(
                    0,
                    0,
                    device_logical_rows,
                    verifier_rows,
                    opaque_stream))
            {
                if (device_logical_rows)
                    (void)cudaFree(device_logical_rows);
                return false;
            }
        }

        CUDAGraphCapture graph(stream, 0);
        ScopedBackendGraphCapture capture_transaction(
            graph,
            "CUDA grouped KV append");
        if (!capture_transaction.begin())
        {
            if (device_logical_rows)
                (void)cudaFree(device_logical_rows);
            return false;
        }
        const bool append_ok = cache.appendVerifierRowsDecodeEquivalent(
            0, 0, k, v, verifier_rows, opaque_stream);
        capture_transaction.finish();
        if (!append_ok || !graph.instantiate())
        {
            if (device_logical_rows)
                (void)cudaFree(device_logical_rows);
            return false;
        }

        const bool replay_ok =
            graph.launch() &&
            cudaStreamSynchronize(stream) == cudaSuccess;
        if (device_logical_rows)
        {
            const bool unbound = cache.bindGraphAppendCountSource(
                0, 0, nullptr, verifier_rows, opaque_stream);
            (void)cudaFree(device_logical_rows);
            return replay_ok && unbound;
        }
        return replay_ok;
    }

    /**
     * @brief Observe canonical CUDA sequence metadata after a stream fence.
     *
     * This copy exists only in the integration-test assertion boundary. It
     * proves graph replay changed the authoritative device allocation and does
     * not mutate cache state or establish a persistent host mirror.
     */
    bool observeDeviceState(
        const IKVCache &cache,
        int max_seq_len,
        void *opaque_stream,
        IKVCache::KVCacheSequenceState *state)
    {
        const auto stream = static_cast<cudaStream_t>(opaque_stream);
        const int *device_head = cache.deviceRingHeadPtr(0, 0);
        const int *device_count = cache.deviceCachedTokenCountPtr(0, 0);
        if (!stream || !state || !device_head || !device_count)
            return false;

        int head = 0;
        int count = 0;
        if (cudaMemcpyAsync(
                &head, device_head, sizeof(head), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaMemcpyAsync(
                &count, device_count, sizeof(count), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess)
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
        const auto stream = static_cast<cudaStream_t>(opaque_stream);
        if (!stream)
            return false;

        CUDAGraphCapture graph(stream, 0);
        ScopedBackendGraphCapture capture_transaction(
            graph,
            "CUDA grouped KV converted read");
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
            cudaStreamSynchronize(stream) == cudaSuccess;
        return replay_ok;
    }

    /** @brief Integration-boundary device-to-host byte observation. */
    bool copyDeviceBytes(void *destination, const void *source, size_t bytes, void *)
    {
        return cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    /** @brief Allocate a CUDA assertion buffer outside production cache code. */
    void *allocateDeviceBytes(size_t bytes)
    {
        void *pointer = nullptr;
        return cudaMalloc(&pointer, bytes) == cudaSuccess ? pointer : nullptr;
    }

    /** @brief Release a CUDA assertion buffer after the matrix cell completes. */
    void releaseDeviceBytes(void *pointer)
    {
        if (pointer)
            (void)cudaFree(pointer);
    }

    /** @brief Enqueue the suite's final device-to-host observation copy. */
    bool copyDeviceBytesAsync(
        void *destination,
        const void *source,
        size_t bytes,
        void *opaque_stream)
    {
        return opaque_stream &&
               cudaMemcpyAsync(
                   destination, source, bytes, cudaMemcpyDeviceToHost,
                   static_cast<cudaStream_t>(opaque_stream)) == cudaSuccess;
    }

    bool synchronizeStream(void *opaque_stream)
    {
        return opaque_stream &&
               cudaStreamSynchronize(static_cast<cudaStream_t>(opaque_stream)) == cudaSuccess;
    }
} // namespace

/**
 * @brief Run one independently process-isolated cache/source format matrix.
 *
 * A backend kernel fault can leave a GPU context unable to retire. Keeping
 * each format in a distinct GTest lets CTest identify that format before the
 * process reaches any later matrix cell, while the shared harness still owns
 * the complete D/M/layout/topology proof for the selected format.
 */
void runRuntimePublicationFormat(size_t format_index)
{
    ASSERT_LT(format_index, kFormatCases.size());
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1)
        GTEST_SKIP() << "CUDA device unavailable";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    ScopedCudaStream stream;
    ASSERT_NE(stream.get(), nullptr);
    runFormatGroupedVerifierSweep(
        DeviceId::cuda(0),
        "CUDA",
        "cuda_kv_cache_grouped_verifier_append_calls",
        kFormatCases[format_index],
        stream.opaque(),
        appendGrouped,
        observeDeviceState);
}

#define LLAMINAR_CUDA_KV_RUNTIME_FORMAT_TEST(suffix, index)                     \
    TEST(Test__CUDAKVCacheGroupedVerifier, RuntimePublication_##suffix)         \
    {                                                                            \
        runRuntimePublicationFormat(index);                                      \
    }

LLAMINAR_CUDA_KV_RUNTIME_FORMAT_TEST(FP32_From_FP32, 0)
LLAMINAR_CUDA_KV_RUNTIME_FORMAT_TEST(BF16_From_BF16, 1)
LLAMINAR_CUDA_KV_RUNTIME_FORMAT_TEST(FP16_From_FP32, 2)
LLAMINAR_CUDA_KV_RUNTIME_FORMAT_TEST(FP16_From_FP16, 3)
LLAMINAR_CUDA_KV_RUNTIME_FORMAT_TEST(FP16_From_BF16, 4)
LLAMINAR_CUDA_KV_RUNTIME_FORMAT_TEST(FP16_From_Q8_1, 5)
LLAMINAR_CUDA_KV_RUNTIME_FORMAT_TEST(Q8_1_From_FP32, 6)
LLAMINAR_CUDA_KV_RUNTIME_FORMAT_TEST(AQ8K_TQ4V_From_FP32, 7)
LLAMINAR_CUDA_KV_RUNTIME_FORMAT_TEST(AQ8K_TQ8V_From_FP32, 8)

#undef LLAMINAR_CUDA_KV_RUNTIME_FORMAT_TEST

TEST(Test__CUDAKVCacheGroupedVerifier,
     AllFormatsCapturedConvertedBatchReadMatchesSerialBytes)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1)
        GTEST_SKIP() << "CUDA device unavailable";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    ScopedCudaStream stream;
    ASSERT_NE(stream.get(), nullptr);
    runAllFormatConvertedDeviceReadSweep(
        DeviceId::cuda(0), "CUDA", stream.opaque(),
        readConvertedBatch, copyDeviceBytes, synchronizeStream);
}

TEST(Test__CUDAKVCacheGroupedVerifier,
     AllNativeFormatsDeviceLogicalBlockHarvestRestoreAreByteExact)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1)
        GTEST_SKIP() << "CUDA device unavailable";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    ScopedCudaStream stream;
    ASSERT_NE(stream.get(), nullptr);
    runAllFormatDeviceLogicalBlockSweep(
        DeviceId::cuda(0),
        "CUDA",
        "cuda_device_logical_kv_exports",
        "cuda_device_logical_kv_imports",
        "cuda_device_logical_tq_kv_exports",
        "cuda_device_logical_tq_kv_imports",
        stream.opaque(),
        allocateDeviceBytes,
        releaseDeviceBytes,
        copyDeviceBytesAsync,
        synchronizeStream);
}

/**
 * @brief Test the complete captured MTP/main/prefix lifecycle under contention.
 */
TEST(Test__CUDAKVCacheGroupedVerifier,
     AdversarialMultiStreamGraphReusePrefixRestoreMatchesSerialState)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
        device_count < 1)
    {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    ScopedCudaStream producer_stream;
    ScopedCudaStream graph_stream;
    ScopedCudaStream observer_stream;
    ASSERT_NE(producer_stream.get(), nullptr);
    ASSERT_NE(graph_stream.get(), nullptr);
    ASSERT_NE(observer_stream.get(), nullptr);

    CUDAKVLifecycleRuntime runtime;
    runAdversarialKVLifecycleStress(
        DeviceId::cuda(0),
        "CUDA",
        ActivationPrecision::FP32,
        "FP32",
        producer_stream.opaque(),
        graph_stream.opaque(),
        observer_stream.opaque(),
        runtime);
}

/**
 * @brief Stress all asymmetric compressed policies through captured MTP publication.
 *
 * The canonical metadata publisher lives in the common CUDA ring-cache base,
 * while verifier append and logical-block serialization are format-specific.
 * Running the complete lifecycle for both physical TQ layouts proves that the
 * shared publisher does not conceal a TQ wraparound, restore, or codec defect.
 */
TEST(Test__CUDAKVCacheGroupedVerifier,
     AdversarialTurboQuantMultiStreamGraphReuseMatchesSerialState)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
        device_count < 1)
    {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    ScopedCudaStream producer_stream;
    ScopedCudaStream graph_stream;
    ScopedCudaStream observer_stream;
    CUDAKVLifecycleRuntime runtime;

    for (const auto &[precision, label] :
         std::array<std::pair<ActivationPrecision, const char *>, 3>{{
             {ActivationPrecision::Q8_1, "AQ8-K/Q8_1-V"},
             {ActivationPrecision::TQ4, "AQ8-K/TQ4-V"},
             {ActivationPrecision::TQ8, "AQ8-K/TQ8-V"},
         }})
    {
        SCOPED_TRACE(label);
        runAdversarialKVLifecycleStress(
            DeviceId::cuda(0),
            "CUDA",
            precision,
            label,
            producer_stream.opaque(),
            graph_stream.opaque(),
            observer_stream.opaque(),
            runtime);
    }
}
