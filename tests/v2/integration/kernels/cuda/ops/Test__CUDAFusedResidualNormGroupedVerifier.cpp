/**
 * @file Test__CUDAFusedResidualNormGroupedVerifier.cpp
 * @brief CUDA production-stage grouped fused-residual RMSNorm integration gate.
 *
 * The shared matrix drives FusedResidualNormStage on one non-default CUDA stream
 * for FP32, BF16, and FP16 at M=2..4. Both device-published outputs must equal
 * independent CUDA M=1 stage executions byte for byte, and stage telemetry must
 * prove one native fused launch covered the complete verifier row group.
 */

#include <gtest/gtest.h>

#include "utils/GpuFusedResidualNormGroupedVerifierHarness.h"

#include <cuda_runtime.h>

using namespace llaminar2;
using namespace llaminar2::test::gpu_fused_residual_norm_verifier;

namespace
{
    /** @brief CUDA copy and observation policy used by the shared harness. */
    struct CudaRuntime
    {
        /** @brief Enqueue a native device-to-host copy on the production stream. */
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

        /** @brief Wait until all preceding work on the explicit stream completes. */
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

        /** @return Owned non-default CUDA stream, or null after creation failure. */
        cudaStream_t get() const { return stream_; }

    private:
        cudaStream_t stream_ = nullptr;
    };
} // namespace

TEST(Test__CUDAFusedResidualNormGroupedVerifier,
     AllNativeFormatsAndReductionWidthsMatchSerialDecodeBytes)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        GTEST_SKIP() << "No CUDA device available";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    ScopedCudaStream stream;
    ASSERT_NE(stream.get(), nullptr);
    auto context = IDeviceContext::create(DeviceId::cuda(0));
    ASSERT_NE(context, nullptr);

    CudaRuntime runtime;
    runAllFormats(
        runtime,
        context.get(),
        DeviceId::cuda(0),
        static_cast<void *>(stream.get()),
        "CUDA",
        "cuda_fused_residual_rmsnorm_grouped_verifier_rows_calls");
}
