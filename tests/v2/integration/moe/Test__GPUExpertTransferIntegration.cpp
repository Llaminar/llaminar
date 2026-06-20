/**
 * @file Test__GPUExpertTransferIntegration.cpp
 * @brief Integration tests for GPU↔GPU expert weight transfer.
 *
 * Tests:
 *   1. Device context preservation after GPU transfer
 *   2. P2P transfer data integrity (if multi-GPU available)
 *   3. Host-staged fallback correctness
 *   4. Transfer with asymmetric weights (mins array present)
 *   5. Zero-size array handling (emins_bytes=0)
 *
 * Requires: HAVE_ROCM or HAVE_CUDA and at least 1 GPU.
 * Multi-GPU tests are skipped if only 1 device is available.
 */

#include <gtest/gtest.h>

#include "execution/moe/GPUExpertTransfer.h"
#include "backends/DeviceId.h"
#include "utils/Logger.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

using namespace llaminar2;

// ---------------------------------------------------------------------------
// Skip helper
// ---------------------------------------------------------------------------

static int getGPUCount()
{
#ifdef HAVE_ROCM
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) return 0;
    return count;
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class Test__GPUExpertTransferIntegration : public ::testing::Test
{
protected:
    void SetUp() override
    {
        gpu_count_ = getGPUCount();
        if (gpu_count_ == 0) {
            GTEST_SKIP() << "No GPU available, skipping GPU transfer tests";
        }
    }

    int gpu_count_ = 0;
};

// ---------------------------------------------------------------------------
// 1. Device context preserved after transfer
// ---------------------------------------------------------------------------

#ifdef HAVE_ROCM
TEST_F(Test__GPUExpertTransferIntegration, DeviceContextPreserved_SingleGPU)
{
    // Set device to 0, perform a self-transfer (src=dst=0), verify device is still 0
    hipSetDevice(0);

    // Allocate test buffers on device 0
    const size_t test_bytes = 4096;
    uint8_t* d_src = nullptr;
    uint8_t* d_dst = nullptr;
    ASSERT_EQ(hipMalloc(&d_src, test_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_dst, test_bytes), hipSuccess);

    // Fill source with pattern
    std::vector<uint8_t> host_pattern(test_bytes);
    for (size_t i = 0; i < test_bytes; ++i)
        host_pattern[i] = static_cast<uint8_t>(i & 0xFF);
    ASSERT_EQ(hipMemcpy(d_src, host_pattern.data(), test_bytes, hipMemcpyHostToDevice), hipSuccess);

    GPUExpertPointers src_ptrs, dst_ptrs;
    src_ptrs.d_vnni = d_src;
    dst_ptrs.d_vnni = d_dst;

    // Transfer (self-device — always P2P since same device)
    bool ok = GPUExpertTransfer::transferExpert(
        src_ptrs, dst_ptrs,
        DeviceId::rocm(0), DeviceId::rocm(0),
        test_bytes, 0, 0, 0, nullptr);
    EXPECT_TRUE(ok);

    // Verify device is still 0
    int current_device = -1;
    ASSERT_EQ(hipGetDevice(&current_device), hipSuccess);
    EXPECT_EQ(current_device, 0) << "Device context must be preserved after transfer";

    // Verify data integrity
    std::vector<uint8_t> host_result(test_bytes);
    ASSERT_EQ(hipMemcpy(host_result.data(), d_dst, test_bytes, hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(host_pattern, host_result) << "Transferred data must match source";

    hipFree(d_src);
    hipFree(d_dst);
}

TEST_F(Test__GPUExpertTransferIntegration, DeviceContextPreserved_MultiGPU)
{
    if (gpu_count_ < 2) {
        GTEST_SKIP() << "Need 2+ GPUs for cross-device context test";
    }

    // Set active device to 0
    hipSetDevice(0);

    // Allocate on device 0 (source)
    const size_t vnni_bytes = 8192;
    const size_t scales_bytes = 512;
    uint8_t *d_src_vnni = nullptr, *d_dst_vnni = nullptr;
    uint8_t *d_src_scales = nullptr, *d_dst_scales = nullptr;
    ASSERT_EQ(hipMalloc(&d_src_vnni, vnni_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_src_scales, scales_bytes), hipSuccess);

    // Fill with pattern
    std::vector<uint8_t> pattern_vnni(vnni_bytes);
    std::vector<uint8_t> pattern_scales(scales_bytes);
    for (size_t i = 0; i < vnni_bytes; ++i)
        pattern_vnni[i] = static_cast<uint8_t>((i * 3 + 7) & 0xFF);
    for (size_t i = 0; i < scales_bytes; ++i)
        pattern_scales[i] = static_cast<uint8_t>((i * 11 + 3) & 0xFF);
    ASSERT_EQ(hipMemcpy(d_src_vnni, pattern_vnni.data(), vnni_bytes, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_src_scales, pattern_scales.data(), scales_bytes, hipMemcpyHostToDevice), hipSuccess);

    // Allocate on device 1 (destination)
    hipSetDevice(1);
    ASSERT_EQ(hipMalloc(&d_dst_vnni, vnni_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_dst_scales, scales_bytes), hipSuccess);

    // Switch back to device 0 (caller's context)
    hipSetDevice(0);

    GPUExpertPointers src_ptrs, dst_ptrs;
    src_ptrs.d_vnni = d_src_vnni;
    src_ptrs.d_scales = d_src_scales;
    dst_ptrs.d_vnni = d_dst_vnni;
    dst_ptrs.d_scales = d_dst_scales;

    bool ok = GPUExpertTransfer::transferExpert(
        src_ptrs, dst_ptrs,
        DeviceId::rocm(0), DeviceId::rocm(1),
        vnni_bytes, scales_bytes, 0, 0, nullptr);
    EXPECT_TRUE(ok);

    // Verify caller's device context is preserved (should be 0)
    int current_device = -1;
    ASSERT_EQ(hipGetDevice(&current_device), hipSuccess);
    EXPECT_EQ(current_device, 0)
        << "Device context must be restored to caller's original device (0) after cross-device transfer";

    // Verify data integrity on destination
    hipSetDevice(1);
    std::vector<uint8_t> result_vnni(vnni_bytes);
    std::vector<uint8_t> result_scales(scales_bytes);
    ASSERT_EQ(hipMemcpy(result_vnni.data(), d_dst_vnni, vnni_bytes, hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(result_scales.data(), d_dst_scales, scales_bytes, hipMemcpyDeviceToHost), hipSuccess);

    EXPECT_EQ(pattern_vnni, result_vnni) << "VNNI data integrity failed";
    EXPECT_EQ(pattern_scales, result_scales) << "Scales data integrity failed";

    // Cleanup
    hipSetDevice(0);
    hipFree(d_src_vnni);
    hipFree(d_src_scales);
    hipSetDevice(1);
    hipFree(d_dst_vnni);
    hipFree(d_dst_scales);
}

TEST_F(Test__GPUExpertTransferIntegration, ZeroSizeArraysHandled)
{
    // Transfer with vnni only — scales/mins/emins all zero
    const size_t vnni_bytes = 2048;
    uint8_t *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(hipMalloc(&d_src, vnni_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_dst, vnni_bytes), hipSuccess);

    std::vector<uint8_t> pattern(vnni_bytes, 0xAB);
    ASSERT_EQ(hipMemcpy(d_src, pattern.data(), vnni_bytes, hipMemcpyHostToDevice), hipSuccess);

    GPUExpertPointers src_ptrs, dst_ptrs;
    src_ptrs.d_vnni = d_src;
    dst_ptrs.d_vnni = d_dst;

    bool ok = GPUExpertTransfer::transferExpert(
        src_ptrs, dst_ptrs,
        DeviceId::rocm(0), DeviceId::rocm(0),
        vnni_bytes, 0, 0, 0, nullptr);
    EXPECT_TRUE(ok);

    std::vector<uint8_t> result(vnni_bytes);
    ASSERT_EQ(hipMemcpy(result.data(), d_dst, vnni_bytes, hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(pattern, result);

    hipFree(d_src);
    hipFree(d_dst);
}

TEST_F(Test__GPUExpertTransferIntegration, NonROCmDeviceRejected)
{
    GPUExpertPointers src_ptrs, dst_ptrs;
    bool ok = GPUExpertTransfer::transferExpert(
        src_ptrs, dst_ptrs,
        DeviceId::cuda(0), DeviceId::rocm(0),
        1024, 0, 0, 0, nullptr);
    EXPECT_FALSE(ok) << "Mixed device types must be rejected";
}

TEST_F(Test__GPUExpertTransferIntegration, PeerAccessQuery)
{
    // canAccessPeer(0, 0) should always return true (same device)
    EXPECT_TRUE(GPUExpertTransfer::canAccessPeer(0, 0));

    if (gpu_count_ >= 2) {
        // P2P between different devices may or may not work — just verify no crash
        bool can = GPUExpertTransfer::canAccessPeer(0, 1);
        (void)can; // Result depends on hardware topology
    }
}
#endif // HAVE_ROCM
