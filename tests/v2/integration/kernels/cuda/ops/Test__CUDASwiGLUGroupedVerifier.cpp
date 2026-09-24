/**
 * @file Test__CUDASwiGLUGroupedVerifier.cpp
 * @brief CUDA-native grouped standalone SwiGLU decode-equivalence gate.
 *
 * The explicit suite enters CUDASwiGLUKernelT through its production tensor API
 * on a non-default stream. Every native activation format must match CUDA M=1
 * decode bytes across the runtime-M inventory in one flat activation launch.
 */

#include <gtest/gtest.h>

#include "kernels/cuda/ops/CUDASwiGLUKernelT.h"
#include "utils/GpuSwiGLUGroupedVerifierHarness.h"

#include <cuda_runtime.h>

using namespace llaminar2;
using namespace llaminar2::test::gpu_swiglu_verifier;

namespace
{
    /** @brief CUDA copy and synchronization policy for the shared harness. */
    struct CudaRuntime
    {
        /** @brief Enqueue one native result copy on the explicit stream. */
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

        /** @brief Synchronize only at a test observation boundary. */
        void synchronize(void *stream) const
        {
            ASSERT_EQ(
                cudaStreamSynchronize(static_cast<cudaStream_t>(stream)),
                cudaSuccess);
        }
    };

    /** @brief Own the mandatory non-default CUDA stream across assertions. */
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

        ScopedCudaStream(const ScopedCudaStream &) = delete;
        ScopedCudaStream &operator=(const ScopedCudaStream &) = delete;

        /** @return Owned non-default stream, or null after creation failure. */
        cudaStream_t get() const { return stream_; }

    private:
        cudaStream_t stream_ = nullptr;
    };
} // namespace

TEST(Test__CUDASwiGLUGroupedVerifier, AllNativeFormatsMatchSerialDecodeBytes)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        GTEST_SKIP() << "No CUDA device available";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    ScopedCudaStream stream;
    ASSERT_NE(stream.get(), nullptr);
    CudaRuntime runtime;
    runAllFormats<llaminar2::cuda::CUDASwiGLUKernelT>(
        runtime,
        DeviceId::cuda(0),
        static_cast<void *>(stream.get()),
        "CUDA",
        "cuda_swiglu_grouped_verifier_rows_calls");
}
