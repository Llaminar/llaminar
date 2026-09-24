/**
 * @file Test__ROCmAttentionKeyCachePartitions.cpp
 * @brief Real ROCm graph proof of cold/partial-prefix native cache identity.
 *
 * Shares the complete value-codec, head-width, chunk-boundary and graph-reset
 * inventory with the other GPU backend. No model, weight conversion or
 * performance threshold participates in this functional preflight regression.
 */
#include "../AttentionKeyCachePartitionProof.h"
#include "kernels/rocm/kvcache/ROCmRingKVCacheTQ.h"
#include "backends/rocm/HIPGraphCapture.h"
#include <hip/hip_runtime.h>

namespace llaminar2::test
{
/** @brief Own the explicit stream until all captured consumers have retired. */
class ROCmPartitionStream
{
public:
    /** @brief Create a non-blocking participant-local execution stream. */
    ROCmPartitionStream()
    {
        if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess)
            throw std::runtime_error("Cannot create ROCm partition-proof stream");
    }
    /** @brief Release only after the shared proof has collected graph results. */
    ~ROCmPartitionStream() { (void)hipStreamDestroy(stream); }
    ROCmPartitionStream(const ROCmPartitionStream &) = delete;
    ROCmPartitionStream &operator=(const ROCmPartitionStream &) = delete;
    hipStream_t stream = nullptr; ///< Borrowed by each cache and graph.
};

/** @brief Exercise every compressed value codec on one actual device. */
TEST(ROCmAttentionKeyCachePartitions, CapturedColdAndRestoredBytesAreIdentical)
{
    int devices = 0;
    ASSERT_EQ(hipGetDeviceCount(&devices), hipSuccess);
    ASSERT_GT(devices, 0) << "This explicit device gate requires ROCm";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ROCmPartitionStream owner;
    proveAttentionKeyCachePartitions<ROCmRingKVCacheTQ>(
        DeviceId::rocm(0), owner.stream,
        [&] { return std::make_unique<HIPGraphCapture>(owner.stream, 0); },
        [&] { return hipStreamSynchronize(owner.stream) == hipSuccess; });
}
} // namespace llaminar2::test

