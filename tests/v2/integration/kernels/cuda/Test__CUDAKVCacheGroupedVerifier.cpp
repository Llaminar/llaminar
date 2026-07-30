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

        /** @brief Own one captured CUDA graph and executable. */
        class Graph
        {
        public:
            Graph() = default;

            Graph(
                cudaGraph_t graph,
                cudaGraphExec_t executable)
                : graph_(graph),
                  executable_(executable)
            {
            }

            ~Graph()
            {
                if (executable_)
                    (void)cudaGraphExecDestroy(
                        executable_);
                if (graph_)
                    (void)cudaGraphDestroy(graph_);
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
                    (void)cudaGraphExecDestroy(
                        executable_);
                if (graph_)
                    (void)cudaGraphDestroy(graph_);
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

            cudaGraphExec_t executable() const
            {
                return executable_;
            }

        private:
            cudaGraph_t graph_ = nullptr;
            cudaGraphExec_t executable_ = nullptr;
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
            if (cudaStreamBeginCapture(
                    stream,
                    cudaStreamCaptureModeGlobal) !=
                cudaSuccess)
            {
                return {};
            }

            bool enqueue_ok = false;
            {
                GraphCaptureGuard guard;
                enqueue_ok = enqueue();
            }
            cudaGraph_t graph = nullptr;
            const cudaError_t end_status =
                cudaStreamEndCapture(
                    stream,
                    &graph);
            if (!enqueue_ok ||
                end_status != cudaSuccess ||
                !graph)
            {
                if (graph)
                    (void)cudaGraphDestroy(graph);
                return {};
            }

            cudaGraphExec_t executable = nullptr;
            if (cudaGraphInstantiate(
                    &executable,
                    graph,
                    nullptr,
                    nullptr,
                    0) != cudaSuccess ||
                !executable)
            {
                (void)cudaGraphDestroy(graph);
                return {};
            }
            return Graph(graph, executable);
        }

        bool launchGraph(
            const Graph &graph,
            void *opaque_stream)
        {
            return graph.valid() && opaque_stream &&
                   cudaGraphLaunch(
                       graph.executable(),
                       static_cast<cudaStream_t>(
                           opaque_stream)) ==
                       cudaSuccess;
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
     * and count. No host sequence metadata is uploaded before capture or
     * adopted after replay.
     */
    bool appendGrouped(
        IKVCache &cache,
        const ITensor *k,
        const ITensor *v,
        int verifier_rows,
        void *opaque_stream)
    {
        const auto stream = static_cast<cudaStream_t>(opaque_stream);
        if (!stream || !cache.isGraphCaptureReady())
            return false;

        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) != cudaSuccess)
            return false;
        bool append_ok = false;
        {
            GraphCaptureGuard guard;
            append_ok = cache.appendVerifierRowsDecodeEquivalent(
                0, 0, k, v, verifier_rows, opaque_stream);
        }
        const cudaError_t end_status = cudaStreamEndCapture(stream, &graph);
        if (!append_ok || end_status != cudaSuccess || !graph)
        {
            if (graph)
                (void)cudaGraphDestroy(graph);
            return false;
        }
        if (cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0) != cudaSuccess ||
            !executable)
        {
            (void)cudaGraphDestroy(graph);
            return false;
        }

        const bool replay_ok =
            cudaGraphLaunch(executable, stream) == cudaSuccess &&
            cudaStreamSynchronize(stream) == cudaSuccess;
        (void)cudaGraphExecDestroy(executable);
        (void)cudaGraphDestroy(graph);
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
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        if (!stream ||
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) != cudaSuccess)
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
        const cudaError_t end_status = cudaStreamEndCapture(stream, &graph);
        if (!read_ok || end_status != cudaSuccess || !graph ||
            cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0) != cudaSuccess ||
            !executable)
        {
            if (executable)
                (void)cudaGraphExecDestroy(executable);
            if (graph)
                (void)cudaGraphDestroy(graph);
            return false;
        }
        const bool replay_ok =
            cudaGraphLaunch(executable, stream) == cudaSuccess &&
            cudaStreamSynchronize(stream) == cudaSuccess;
        (void)cudaGraphExecDestroy(executable);
        (void)cudaGraphDestroy(graph);
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

TEST(Test__CUDAKVCacheGroupedVerifier,
     AllFormatsRuntimeMGraphCapturedReplicatedAndLocalTPMatchSerialDecodeBytes)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1)
        GTEST_SKIP() << "CUDA device unavailable";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    ScopedCudaStream stream;
    ASSERT_NE(stream.get(), nullptr);
    runAllFormatGroupedVerifierSweep(
        DeviceId::cuda(0),
        "CUDA",
        "cuda_kv_cache_grouped_verifier_append_calls",
        stream.opaque(),
        appendGrouped,
        observeDeviceState);
}

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
 * @brief Stress both TurboQuant policies through captured MTP publication.
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
         std::array<std::pair<ActivationPrecision, const char *>, 2>{{
             {ActivationPrecision::TQ4, "TQ8-K/TQ4-V"},
             {ActivationPrecision::TQ8, "TQ8-K/TQ8-V"},
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
