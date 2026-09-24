/**
 * @file Test__CUDAAttentionKeyCachePartitions.cpp
 * @brief Real CUDA graph proof of cold/partial-prefix native cache identity.
 *
 * Shares the complete value-codec, head-width, chunk-boundary and graph-reset
 * inventory with the other GPU backend. No model, weight conversion or
 * performance threshold participates in this functional preflight regression.
 */
#include "../AttentionKeyCachePartitionProof.h"
#include "kernels/cuda/kvcache/CUDARingKVCacheTQ.h"
#include "backends/cuda/CUDAGraphCapture.h"
#include <cuda_runtime.h>

namespace llaminar2::test
{
/** @brief Own the explicit stream until all captured consumers have retired. */
class CUDAPartitionStream
{
public:
    /** @brief Create a non-blocking participant-local execution stream. */
    CUDAPartitionStream()
    {
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
            throw std::runtime_error("Cannot create CUDA partition-proof stream");
    }
    /** @brief Release only after the shared proof has collected graph results. */
    ~CUDAPartitionStream() { (void)cudaStreamDestroy(stream); }
    CUDAPartitionStream(const CUDAPartitionStream &) = delete;
    CUDAPartitionStream &operator=(const CUDAPartitionStream &) = delete;
    cudaStream_t stream = nullptr; ///< Borrowed by each cache and graph.
};

/** @brief Exercise every compressed value codec on one actual device. */
TEST(CUDAAttentionKeyCachePartitions, CapturedColdAndRestoredBytesAreIdentical)
{
    int devices = 0;
    ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
    ASSERT_GT(devices, 0) << "This explicit device gate requires CUDA";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    CUDAPartitionStream owner;
    proveAttentionKeyCachePartitions<CUDARingKVCacheTQ>(
        DeviceId::cuda(0), owner.stream,
        [&] { return std::make_unique<CUDAGraphCapture>(owner.stream, 0); },
        [&] { return cudaStreamSynchronize(owner.stream) == cudaSuccess; });
}
} // namespace llaminar2::test

