/**
 * @file Test__ROCmRMSNormGroupedVerifier.cpp
 * @brief ROCm-native grouped RMSNorm decode-equivalence integration gate.
 *
 * This explicit HIP suite mirrors the CUDA matrix exactly. It enters the
 * tensor-aware ROCmRMSNormKernelT production API on a non-default stream and
 * requires native byte equality against ROCm M=1 decode for FP32, BF16, and
 * FP16 at every verifier depth and reduction-width dispatch.
 */

#include <gtest/gtest.h>

#include "kernels/rocm/ops/ROCmRMSNormKernelT.h"
#include "utils/GpuRMSNormGroupedVerifierHarness.h"

#include <hip/hip_runtime.h>

using namespace llaminar2;
using namespace llaminar2::test::gpu_rmsnorm_verifier;

namespace
{
    /**
     * @brief HIP runtime operations required by the backend-neutral harness.
     *
     * Native output copies share the production test stream. Synchronization is
     * reserved for test observation boundaries and does not alter the grouped
     * production implementation under proof.
     */
    struct ROCmRuntime
    {
        /** @brief Enqueue a native output copy on the production test stream. */
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

        /** @brief Wait until all operations preceding the observation complete. */
        void synchronize(void *stream) const
        {
            ASSERT_EQ(
                hipStreamSynchronize(static_cast<hipStream_t>(stream)),
                hipSuccess);
        }
    };

    /** @brief Own the mandatory non-default HIP stream across fatal assertions. */
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

TEST(Test__ROCmRMSNormGroupedVerifier, AllNativeFormatsAndReductionWidthsMatchSerialDecodeBytes)
{
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0)
        GTEST_SKIP() << "No ROCm device available";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedROCmStream stream;
    ASSERT_NE(stream.get(), nullptr);
    ROCmRuntime runtime;
    runAllFormats<llaminar2::rocm::ROCmRMSNormKernelT>(
        runtime,
        DeviceId::rocm(0),
        static_cast<void *>(stream.get()),
        "ROCm",
        "rocm_rmsnorm_grouped_verifier_rows_calls");
}
