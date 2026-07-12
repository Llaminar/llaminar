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
     * @brief Execute one grouped publication entirely inside a HIP graph.
     *
     * The graph owns both payload publication and canonical device metadata
     * advancement. No persistent host sequence-state copy participates.
     */
    bool appendGrouped(
        IKVCache &cache,
        const ITensor *k,
        const ITensor *v,
        int verifier_rows,
        void *opaque_stream)
    {
        const auto stream = static_cast<hipStream_t>(opaque_stream);
        if (!stream || !cache.isGraphCaptureReady())
            return false;

        hipGraph_t graph = nullptr;
        hipGraphExec_t executable = nullptr;
        if (hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal) != hipSuccess)
            return false;
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
            return false;
        }
        if (hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0) != hipSuccess ||
            !executable)
        {
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
