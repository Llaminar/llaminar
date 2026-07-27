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
