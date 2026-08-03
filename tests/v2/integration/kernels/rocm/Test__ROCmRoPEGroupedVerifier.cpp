/**
 * @file Test__ROCmRoPEGroupedVerifier.cpp
 * @brief ROCm-native grouped RoPE decode-equivalence integration gate.
 *
 * This explicit ROCm suite mirrors the CUDA matrix exactly and drives
 * ROCmRoPEKernelT on a non-default HIP stream. Native grouped Q/K bytes must
 * equal ROCm serial decode for every supported format and production position
 * route; route counters make an unintended owner a hard failure.
 */

#include <gtest/gtest.h>

#include "kernels/rocm/ops/ROCmRoPEKernelT.h"
#include "utils/GpuRoPEGroupedVerifierHarness.h"

#include <hip/hip_runtime.h>

using namespace llaminar2;
using namespace llaminar2::test::gpu_rope_verifier;

namespace
{
    /** @brief HIP copy/synchronization policy consumed by the shared harness. */
    struct ROCmRuntime
    {
        bool allocateDevice(void **pointer, size_t bytes) const
        {
            return hipMalloc(pointer, bytes) == hipSuccess;
        }

        void freeDevice(void *pointer) const
        {
            (void)hipFree(pointer);
        }

        bool copyHostToDevice(
            void *destination,
            const void *source,
            size_t bytes,
            void *stream) const
        {
            return hipMemcpyAsync(
                       destination, source, bytes, hipMemcpyHostToDevice,
                       static_cast<hipStream_t>(stream)) == hipSuccess;
        }

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

        hipStream_t get() const { return stream_; }

    private:
        hipStream_t stream_ = nullptr;
    };
} // namespace

TEST(Test__ROCmRoPEGroupedVerifier, AllNativeFormatsAndPositionRoutesMatchSerialDecodeBytes)
{
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0)
        GTEST_SKIP() << "No ROCm device available";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedROCmStream stream;
    ASSERT_NE(stream.get(), nullptr);
    ROCmRuntime runtime;
    runAllFormats<llaminar2::rocm::ROCmRoPEKernelT>(
        runtime,
        DeviceId::rocm(0),
        static_cast<void *>(stream.get()),
        "ROCm",
        "rocm_rope_grouped_verifier_rows_calls");
}
