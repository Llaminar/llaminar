/**
 * @file Test__CUDARMSNormGroupedVerifier.cpp
 * @brief CUDA-native grouped RMSNorm decode-equivalence integration gate.
 *
 * This explicit CUDA suite drives CUDARMSNormKernelT through the tensor-aware
 * production API on a non-default stream. The shared matrix covers every native
 * activation format, every certified runtime M, and both narrow per-head and wide hidden-state
 * reductions. Grouped output must equal CUDA M=1 decode byte for byte, and the
 * production route counter must prove one block-per-row grouped launch.
 */

#include <gtest/gtest.h>

#include "kernels/cuda/ops/CUDARMSNormKernelT.h"
#include "utils/GpuRMSNormGroupedVerifierHarness.h"

#include <cuda_runtime.h>

using namespace llaminar2;
using namespace llaminar2::test::gpu_rmsnorm_verifier;

namespace
{
    /**
     * @brief CUDA runtime operations required by the backend-neutral harness.
     *
     * Copies remain asynchronous and ordered on the same explicit stream as the
     * production RMSNorm kernels. Synchronization occurs only at oracle/test
     * observation boundaries and is not part of the kernel implementation.
     */
    struct CudaRuntime
    {
        /** @brief Enqueue a native output copy on the production test stream. */
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

        /** @brief Wait until all operations preceding the observation complete. */
        void synchronize(void *stream) const
        {
            ASSERT_EQ(
                cudaStreamSynchronize(static_cast<cudaStream_t>(stream)),
                cudaSuccess);
        }
    };

    /**
     * @brief Own the mandatory non-default CUDA stream across fatal assertions.
     */
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

        /** @return Owned non-default CUDA stream, or null after creation failure. */
        cudaStream_t get() const { return stream_; }

    private:
        cudaStream_t stream_ = nullptr;
    };
} // namespace

TEST(Test__CUDARMSNormGroupedVerifier, AllNativeFormatsAndReductionWidthsMatchSerialDecodeBytes)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        GTEST_SKIP() << "No CUDA device available";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    ScopedCudaStream stream;
    ASSERT_NE(stream.get(), nullptr);
    CudaRuntime runtime;
    runAllFormats<llaminar2::cuda::CUDARMSNormKernelT>(
        runtime,
        DeviceId::cuda(0),
        static_cast<void *>(stream.get()),
        "CUDA",
        "cuda_rmsnorm_grouped_verifier_rows_calls");
}
