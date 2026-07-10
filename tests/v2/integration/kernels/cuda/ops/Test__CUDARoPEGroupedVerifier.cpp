/**
 * @file Test__CUDARoPEGroupedVerifier.cpp
 * @brief CUDA-native grouped RoPE decode-equivalence integration gate.
 *
 * This explicit CUDA suite drives CUDARoPEKernelT through its tensor-aware
 * production API on a non-default stream. The shared harness covers every CUDA
 * activation format, M=2..4, full and supported partial RoPE, and both device
 * position owners while requiring native byte equality against CUDA M=1 decode.
 */

#include <gtest/gtest.h>

#include "kernels/cuda/ops/CUDARoPEKernelT.h"
#include "utils/GpuRoPEGroupedVerifierHarness.h"

#include <cuda_runtime.h>

using namespace llaminar2;
using namespace llaminar2::test::gpu_rope_verifier;

namespace
{
    /** @brief CUDA copy/synchronization policy consumed by the shared harness. */
    struct CudaRuntime
    {
        void copyDeviceToHost(
            void *destination,
            const void *source,
            size_t bytes,
            void *stream) const
        {
            ASSERT_EQ(
                cudaMemcpyAsync(
                    destination, source, bytes, cudaMemcpyDeviceToHost,
                    static_cast<cudaStream_t>(stream)),
                cudaSuccess);
        }

        void synchronize(void *stream) const
        {
            ASSERT_EQ(
                cudaStreamSynchronize(static_cast<cudaStream_t>(stream)),
                cudaSuccess);
        }
    };

    /** @brief Own the mandatory non-default stream across fatal assertions. */
    class ScopedCudaStream
    {
    public:
        ScopedCudaStream()
        {
            if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess)
                stream_ = nullptr;
        }

        ~ScopedCudaStream()
        {
            if (stream_)
                (void)cudaStreamDestroy(stream_);
        }

        cudaStream_t get() const { return stream_; }

    private:
        cudaStream_t stream_ = nullptr;
    };
} // namespace

TEST(Test__CUDARoPEGroupedVerifier, AllNativeFormatsAndPositionRoutesMatchSerialDecodeBytes)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        GTEST_SKIP() << "No CUDA device available";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    ScopedCudaStream stream;
    ASSERT_NE(stream.get(), nullptr);
    CudaRuntime runtime;
    runAllFormats<llaminar2::cuda::CUDARoPEKernelT>(
        runtime,
        DeviceId::cuda(0),
        static_cast<void *>(stream.get()),
        "CUDA",
        "cuda_rope_grouped_verifier_rows_calls");
}
