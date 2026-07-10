/**
 * @file Test__ROCmSwiGLUGroupedVerifier.cpp
 * @brief ROCm-native grouped standalone SwiGLU decode-equivalence gate.
 *
 * This suite mirrors the CUDA matrix exactly and exercises the newly complete
 * BF16 stream-binding contract. Native grouped outputs must equal ROCm M=1
 * decode bytes while remaining entirely device-owned.
 */

#include <gtest/gtest.h>

#include "kernels/rocm/ops/ROCmSwiGLUKernelT.h"
#include "utils/GpuSwiGLUGroupedVerifierHarness.h"

#include <hip/hip_runtime.h>

using namespace llaminar2;
using namespace llaminar2::test::gpu_swiglu_verifier;

namespace
{
    /** @brief HIP copy and synchronization policy for the shared harness. */
    struct ROCmRuntime
    {
        /** @brief Enqueue one native result copy on the explicit stream. */
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

        /** @brief Synchronize only at a test observation boundary. */
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

        /** @return Owned non-default stream, or null after creation failure. */
        hipStream_t get() const { return stream_; }

    private:
        hipStream_t stream_ = nullptr;
    };
} // namespace

TEST(Test__ROCmSwiGLUGroupedVerifier, AllNativeFormatsMatchSerialDecodeBytes)
{
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0)
        GTEST_SKIP() << "No ROCm device available";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedROCmStream stream;
    ASSERT_NE(stream.get(), nullptr);
    ROCmRuntime runtime;
    runAllFormats<llaminar2::rocm::ROCmSwiGLUKernelT>(
        runtime,
        DeviceId::rocm(0),
        static_cast<void *>(stream.get()),
        "ROCm",
        "rocm_swiglu_grouped_verifier_rows_calls");
}
