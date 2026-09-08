/**
 * @file Test__ROCmFusedResidualNormGroupedVerifier.cpp
 * @brief ROCm production-stage grouped fused-residual RMSNorm integration gate.
 *
 * This suite mirrors the CUDA matrix exactly. FusedResidualNormStage executes
 * FP32, BF16, and FP16 verifier groups on one non-default HIP stream; updated
 * residual and normalized output bytes are compared with ROCm M=1 execution,
 * and telemetry proves one native fused workgroup grid handled the full group.
 */

#include <gtest/gtest.h>

#include "utils/GpuFusedResidualNormGroupedVerifierHarness.h"

#include <hip/hip_runtime.h>

using namespace llaminar2;
using namespace llaminar2::test::gpu_fused_residual_norm_verifier;

namespace
{
    /** @brief HIP copy and observation policy used by the shared harness. */
    struct ROCmRuntime
    {
        /** @brief Enqueue a native device-to-host copy on the production stream. */
        void copyDeviceToHost(
            void *destination,
            const void *source,
            size_t bytes,
            void *stream) const
        {
            ASSERT_EQ(
                hipMemcpyAsync(
                    destination, source, bytes, hipMemcpyDeviceToHost,
                    static_cast<hipStream_t>(stream)),
                hipSuccess);
        }

        /** @brief Wait until all preceding work on the explicit stream completes. */
        void synchronize(void *stream) const
        {
            ASSERT_EQ(
                hipStreamSynchronize(static_cast<hipStream_t>(stream)),
                hipSuccess);
        }
    };

    /** @brief Own the mandatory non-default HIP stream across assertions. */
    class ScopedROCmStream
    {
    public:
        ScopedROCmStream()
        {
            if (hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking) != hipSuccess)
                stream_ = nullptr;
        }

        ~ScopedROCmStream()
        {
            if (stream_)
                (void)hipStreamDestroy(stream_);
        }

        ScopedROCmStream(const ScopedROCmStream &) = delete;
        ScopedROCmStream &operator=(const ScopedROCmStream &) = delete;

        /** @return Owned non-default HIP stream, or null after creation failure. */
        hipStream_t get() const { return stream_; }

    private:
        hipStream_t stream_ = nullptr;
    };
} // namespace

TEST(Test__ROCmFusedResidualNormGroupedVerifier,
     AllNativeFormatsAndReductionWidthsMatchSerialDecodeBytes)
{
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0)
        GTEST_SKIP() << "No ROCm device available";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedROCmStream stream;
    ASSERT_NE(stream.get(), nullptr);
    auto context = IDeviceContext::create(DeviceId::rocm(0));
    ASSERT_NE(context, nullptr);

    ROCmRuntime runtime;
    runAllFormats(
        runtime,
        context.get(),
        DeviceId::rocm(0),
        static_cast<void *>(stream.get()),
        "ROCm",
        "rocm_fused_residual_rmsnorm_grouped_verifier_rows_calls");
}
