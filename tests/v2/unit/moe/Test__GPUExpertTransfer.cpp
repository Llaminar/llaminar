/**
 * @file Test__GPUExpertTransfer.cpp
 * @brief Unit tests for GPU↔GPU expert weight transfer.
 *
 * Validates peer-to-peer and host-staged device-to-device transfer
 * of packed MoE expert weight arrays between ROCm GPUs.
 *
 * Requires: ≥2 ROCm GPUs (tests skip otherwise).
 */

#include <gtest/gtest.h>

#include "execution/moe/GPUExpertTransfer.h"
#include "backends/DeviceId.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

#include <vector>
#include <random>
#include <cstring>
#include <iostream>
#include <iomanip>

using namespace llaminar2;

namespace {

#ifdef HAVE_ROCM

int getROCmDeviceCount()
{
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    return (err == hipSuccess) ? count : 0;
}

/// Allocate device memory on a specific ROCm device.
template<typename T>
T* allocOnDevice(int device_ordinal, size_t count)
{
    (void)hipSetDevice(device_ordinal);
    T* ptr = nullptr;
    hipError_t err = hipMalloc(&ptr, count * sizeof(T));
    return (err == hipSuccess) ? ptr : nullptr;
}

/// Upload host data to a specific device.
template<typename T>
bool uploadToDevice(T* d_ptr, const T* h_ptr, size_t count, int device_ordinal)
{
    (void)hipSetDevice(device_ordinal);
    hipError_t err = hipMemcpy(d_ptr, h_ptr, count * sizeof(T), hipMemcpyHostToDevice);
    return (err == hipSuccess);
}

/// Download device data to host.
template<typename T>
bool downloadFromDevice(T* h_ptr, const T* d_ptr, size_t count, int device_ordinal)
{
    (void)hipSetDevice(device_ordinal);
    hipError_t err = hipMemcpy(h_ptr, d_ptr, count * sizeof(T), hipMemcpyDeviceToHost);
    return (err == hipSuccess);
}

void freeOnDevice(void* ptr, int device_ordinal)
{
    (void)hipSetDevice(device_ordinal);
    (void)hipFree(ptr);
}

#endif // HAVE_ROCM

// ============================================================================
// Test: Peer access check
// ============================================================================
TEST(Test__GPUExpertTransfer, PeerAccessCheck)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm not available";
#else
    const int count = getROCmDeviceCount();
    if (count < 2) GTEST_SKIP() << "Need >= 2 ROCm devices, have " << count;

    bool can_access_0_to_1 = GPUExpertTransfer::canAccessPeer(0, 1);
    bool can_access_1_to_0 = GPUExpertTransfer::canAccessPeer(1, 0);
    bool can_access_self   = GPUExpertTransfer::canAccessPeer(0, 0);

    std::cout << "[PeerAccessCheck] 0→1: " << std::boolalpha << can_access_0_to_1
              << "  1→0: " << can_access_1_to_0
              << "  0→0 (self): " << can_access_self << std::endl;

    // Self-access should always be true
    EXPECT_TRUE(can_access_self);

    // P2P between different devices depends on hardware topology.
    // We don't assert true/false — just verify the call succeeds without crashing.
    // The actual transfer test below handles both P2P and host-staged paths.
#endif
}

// ============================================================================
// Test: Enable peer access (idempotent)
// ============================================================================
TEST(Test__GPUExpertTransfer, EnablePeerAccess)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm not available";
#else
    const int count = getROCmDeviceCount();
    if (count < 2) GTEST_SKIP() << "Need >= 2 ROCm devices, have " << count;

    // P2P may not be available depending on hardware topology (e.g. MI50 via PCIe).
    // Test that the call doesn't crash and handles the unavailable case gracefully.
    bool peer_available = GPUExpertTransfer::canAccessPeer(0, 1);

    (void)hipSetDevice(0);
    bool ok1 = GPUExpertTransfer::enablePeerAccess(1);
    bool ok2 = GPUExpertTransfer::enablePeerAccess(1);  // idempotent

    std::cout << "[EnablePeerAccess] peer_available: " << std::boolalpha << peer_available
              << "  first: " << ok1 << "  second (idempotent): " << ok2 << std::endl;

    if (peer_available) {
        EXPECT_TRUE(ok1);
        EXPECT_TRUE(ok2);
    } else {
        // Without P2P, enablePeerAccess returns false — that's expected.
        // Transfer falls back to host-staged copy.
        std::cout << "  (P2P not available on this topology — host-staged fallback used)" << std::endl;
    }
#endif
}

// ============================================================================
// Test: D2D transfer of vnni + scales arrays
// ============================================================================
TEST(Test__GPUExpertTransfer, D2D_Transfer)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm not available";
#else
    const int count = getROCmDeviceCount();
    if (count < 2) GTEST_SKIP() << "Need >= 2 ROCm devices, have " << count;

    const int src_dev = 0, dst_dev = 1;

    // Simulate expert weight arrays
    const size_t vnni_bytes = 1024 * 1024;  // 1MB
    const size_t scales_count = 4096;       // 4K FP16 scales
    const size_t scales_bytes = scales_count * sizeof(uint16_t);

    // Generate random test data
    std::mt19937 rng(42);
    std::vector<uint8_t> h_vnni(vnni_bytes);
    std::vector<uint16_t> h_scales(scales_count);
    for (auto& b : h_vnni) b = static_cast<uint8_t>(rng() & 0xFF);
    for (auto& s : h_scales) s = static_cast<uint16_t>(rng() & 0xFFFF);

    // Allocate and upload to source device
    auto* d_src_vnni = allocOnDevice<uint8_t>(src_dev, vnni_bytes);
    auto* d_src_scales = allocOnDevice<uint16_t>(src_dev, scales_count);
    ASSERT_NE(d_src_vnni, nullptr);
    ASSERT_NE(d_src_scales, nullptr);
    ASSERT_TRUE(uploadToDevice(d_src_vnni, h_vnni.data(), vnni_bytes, src_dev));
    ASSERT_TRUE(uploadToDevice(d_src_scales, h_scales.data(), scales_count, src_dev));

    // Allocate destination buffers
    auto* d_dst_vnni = allocOnDevice<uint8_t>(dst_dev, vnni_bytes);
    auto* d_dst_scales = allocOnDevice<uint16_t>(dst_dev, scales_count);
    ASSERT_NE(d_dst_vnni, nullptr);
    ASSERT_NE(d_dst_scales, nullptr);

    // Build pointer structs
    GPUExpertPointers src_ptrs, dst_ptrs;
    src_ptrs.d_vnni = d_src_vnni;
    src_ptrs.d_scales = d_src_scales;
    dst_ptrs.d_vnni = d_dst_vnni;
    dst_ptrs.d_scales = d_dst_scales;

    // Transfer
    bool ok = GPUExpertTransfer::transferExpert(
        src_ptrs, dst_ptrs,
        DeviceId::rocm(src_dev), DeviceId::rocm(dst_dev),
        vnni_bytes, scales_bytes,
        0, 0,       // no mins, no emins
        nullptr);   // synchronous
    ASSERT_TRUE(ok);

    // Sync destination device
    (void)hipSetDevice(dst_dev);
    (void)hipDeviceSynchronize();

    // Download and verify
    std::vector<uint8_t> h_result_vnni(vnni_bytes);
    std::vector<uint16_t> h_result_scales(scales_count);
    ASSERT_TRUE(downloadFromDevice(h_result_vnni.data(), d_dst_vnni, vnni_bytes, dst_dev));
    ASSERT_TRUE(downloadFromDevice(h_result_scales.data(), d_dst_scales, scales_count, dst_dev));

    bool vnni_match = (std::memcmp(h_vnni.data(), h_result_vnni.data(), vnni_bytes) == 0);
    bool scales_match = (std::memcmp(h_scales.data(), h_result_scales.data(), scales_bytes) == 0);

    std::cout << "[D2D_Transfer] vnni_match=" << std::boolalpha << vnni_match
              << " scales_match=" << scales_match << std::endl;

    EXPECT_TRUE(vnni_match);
    EXPECT_TRUE(scales_match);

    // Cleanup
    freeOnDevice(d_src_vnni, src_dev);
    freeOnDevice(d_src_scales, src_dev);
    freeOnDevice(d_dst_vnni, dst_dev);
    freeOnDevice(d_dst_scales, dst_dev);
#endif
}

// ============================================================================
// Test: D2D transfer of all 4 arrays (vnni + scales + mins + emins)
// ============================================================================
TEST(Test__GPUExpertTransfer, D2D_Transfer_AllArrays)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm not available";
#else
    const int count = getROCmDeviceCount();
    if (count < 2) GTEST_SKIP() << "Need >= 2 ROCm devices, have " << count;

    const int src_dev = 0, dst_dev = 1;

    const size_t vnni_bytes = 512 * 1024;   // 512KB
    const size_t scales_count = 2048;
    const size_t mins_count = 2048;
    const size_t emins_count = 2048;
    const size_t scales_bytes = scales_count * sizeof(uint16_t);
    const size_t mins_bytes = mins_count * sizeof(uint16_t);
    const size_t emins_bytes = emins_count * sizeof(uint32_t);

    // Generate random test data
    std::mt19937 rng(123);
    std::vector<uint8_t> h_vnni(vnni_bytes);
    std::vector<uint16_t> h_scales(scales_count);
    std::vector<uint16_t> h_mins(mins_count);
    std::vector<uint32_t> h_emins(emins_count);
    for (auto& b : h_vnni) b = static_cast<uint8_t>(rng() & 0xFF);
    for (auto& s : h_scales) s = static_cast<uint16_t>(rng() & 0xFFFF);
    for (auto& m : h_mins) m = static_cast<uint16_t>(rng() & 0xFFFF);
    for (auto& e : h_emins) e = static_cast<uint32_t>(rng());

    // Allocate + upload source
    auto* d_src_vnni = allocOnDevice<uint8_t>(src_dev, vnni_bytes);
    auto* d_src_scales = allocOnDevice<uint16_t>(src_dev, scales_count);
    auto* d_src_mins = allocOnDevice<uint16_t>(src_dev, mins_count);
    auto* d_src_emins = allocOnDevice<uint32_t>(src_dev, emins_count);
    ASSERT_NE(d_src_vnni, nullptr);
    ASSERT_NE(d_src_scales, nullptr);
    ASSERT_NE(d_src_mins, nullptr);
    ASSERT_NE(d_src_emins, nullptr);
    ASSERT_TRUE(uploadToDevice(d_src_vnni, h_vnni.data(), vnni_bytes, src_dev));
    ASSERT_TRUE(uploadToDevice(d_src_scales, h_scales.data(), scales_count, src_dev));
    ASSERT_TRUE(uploadToDevice(d_src_mins, h_mins.data(), mins_count, src_dev));
    ASSERT_TRUE(uploadToDevice(d_src_emins, h_emins.data(), emins_count, src_dev));

    // Allocate destination
    auto* d_dst_vnni = allocOnDevice<uint8_t>(dst_dev, vnni_bytes);
    auto* d_dst_scales = allocOnDevice<uint16_t>(dst_dev, scales_count);
    auto* d_dst_mins = allocOnDevice<uint16_t>(dst_dev, mins_count);
    auto* d_dst_emins = allocOnDevice<uint32_t>(dst_dev, emins_count);
    ASSERT_NE(d_dst_vnni, nullptr);
    ASSERT_NE(d_dst_scales, nullptr);
    ASSERT_NE(d_dst_mins, nullptr);
    ASSERT_NE(d_dst_emins, nullptr);

    GPUExpertPointers src_ptrs, dst_ptrs;
    src_ptrs.d_vnni = d_src_vnni;
    src_ptrs.d_scales = d_src_scales;
    src_ptrs.d_mins = d_src_mins;
    src_ptrs.d_emins = d_src_emins;
    dst_ptrs.d_vnni = d_dst_vnni;
    dst_ptrs.d_scales = d_dst_scales;
    dst_ptrs.d_mins = d_dst_mins;
    dst_ptrs.d_emins = d_dst_emins;

    bool ok = GPUExpertTransfer::transferExpert(
        src_ptrs, dst_ptrs,
        DeviceId::rocm(src_dev), DeviceId::rocm(dst_dev),
        vnni_bytes, scales_bytes, mins_bytes, emins_bytes,
        nullptr);
    ASSERT_TRUE(ok);

    (void)hipSetDevice(dst_dev);
    (void)hipDeviceSynchronize();

    // Download and verify all arrays
    std::vector<uint8_t> r_vnni(vnni_bytes);
    std::vector<uint16_t> r_scales(scales_count);
    std::vector<uint16_t> r_mins(mins_count);
    std::vector<uint32_t> r_emins(emins_count);
    ASSERT_TRUE(downloadFromDevice(r_vnni.data(), d_dst_vnni, vnni_bytes, dst_dev));
    ASSERT_TRUE(downloadFromDevice(r_scales.data(), d_dst_scales, scales_count, dst_dev));
    ASSERT_TRUE(downloadFromDevice(r_mins.data(), d_dst_mins, mins_count, dst_dev));
    ASSERT_TRUE(downloadFromDevice(r_emins.data(), d_dst_emins, emins_count, dst_dev));

    bool vnni_ok = (std::memcmp(h_vnni.data(), r_vnni.data(), vnni_bytes) == 0);
    bool scales_ok = (std::memcmp(h_scales.data(), r_scales.data(), scales_bytes) == 0);
    bool mins_ok = (std::memcmp(h_mins.data(), r_mins.data(), mins_bytes) == 0);
    bool emins_ok = (std::memcmp(h_emins.data(), r_emins.data(), emins_bytes) == 0);

    std::cout << "[D2D_Transfer_AllArrays] vnni=" << std::boolalpha << vnni_ok
              << " scales=" << scales_ok << " mins=" << mins_ok
              << " emins=" << emins_ok << std::endl;

    EXPECT_TRUE(vnni_ok);
    EXPECT_TRUE(scales_ok);
    EXPECT_TRUE(mins_ok);
    EXPECT_TRUE(emins_ok);

    // Cleanup
    freeOnDevice(d_src_vnni, src_dev);
    freeOnDevice(d_src_scales, src_dev);
    freeOnDevice(d_src_mins, src_dev);
    freeOnDevice(d_src_emins, src_dev);
    freeOnDevice(d_dst_vnni, dst_dev);
    freeOnDevice(d_dst_scales, dst_dev);
    freeOnDevice(d_dst_mins, dst_dev);
    freeOnDevice(d_dst_emins, dst_dev);
#endif
}

// ============================================================================
// Test: D2D transfer with zero-size optional arrays (symmetric weights)
// ============================================================================
TEST(Test__GPUExpertTransfer, D2D_Transfer_ZeroSizeOptional)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm not available";
#else
    const int count = getROCmDeviceCount();
    if (count < 2) GTEST_SKIP() << "Need >= 2 ROCm devices, have " << count;

    const int src_dev = 0, dst_dev = 1;

    const size_t vnni_bytes = 256 * 1024;  // 256KB
    const size_t scales_count = 1024;
    const size_t scales_bytes = scales_count * sizeof(uint16_t);

    std::mt19937 rng(999);
    std::vector<uint8_t> h_vnni(vnni_bytes);
    std::vector<uint16_t> h_scales(scales_count);
    for (auto& b : h_vnni) b = static_cast<uint8_t>(rng() & 0xFF);
    for (auto& s : h_scales) s = static_cast<uint16_t>(rng() & 0xFFFF);

    auto* d_src_vnni = allocOnDevice<uint8_t>(src_dev, vnni_bytes);
    auto* d_src_scales = allocOnDevice<uint16_t>(src_dev, scales_count);
    ASSERT_NE(d_src_vnni, nullptr);
    ASSERT_NE(d_src_scales, nullptr);
    ASSERT_TRUE(uploadToDevice(d_src_vnni, h_vnni.data(), vnni_bytes, src_dev));
    ASSERT_TRUE(uploadToDevice(d_src_scales, h_scales.data(), scales_count, src_dev));

    auto* d_dst_vnni = allocOnDevice<uint8_t>(dst_dev, vnni_bytes);
    auto* d_dst_scales = allocOnDevice<uint16_t>(dst_dev, scales_count);
    ASSERT_NE(d_dst_vnni, nullptr);
    ASSERT_NE(d_dst_scales, nullptr);

    GPUExpertPointers src_ptrs, dst_ptrs;
    src_ptrs.d_vnni = d_src_vnni;
    src_ptrs.d_scales = d_src_scales;
    src_ptrs.d_mins = nullptr;    // symmetric: no mins
    src_ptrs.d_emins = nullptr;   // no emins
    dst_ptrs.d_vnni = d_dst_vnni;
    dst_ptrs.d_scales = d_dst_scales;
    dst_ptrs.d_mins = nullptr;
    dst_ptrs.d_emins = nullptr;

    // Transfer with mins_bytes=0 and emins_bytes=0
    bool ok = GPUExpertTransfer::transferExpert(
        src_ptrs, dst_ptrs,
        DeviceId::rocm(src_dev), DeviceId::rocm(dst_dev),
        vnni_bytes, scales_bytes,
        0, 0,       // zero-size optional arrays
        nullptr);
    ASSERT_TRUE(ok);

    (void)hipSetDevice(dst_dev);
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> r_vnni(vnni_bytes);
    std::vector<uint16_t> r_scales(scales_count);
    ASSERT_TRUE(downloadFromDevice(r_vnni.data(), d_dst_vnni, vnni_bytes, dst_dev));
    ASSERT_TRUE(downloadFromDevice(r_scales.data(), d_dst_scales, scales_count, dst_dev));

    bool vnni_ok = (std::memcmp(h_vnni.data(), r_vnni.data(), vnni_bytes) == 0);
    bool scales_ok = (std::memcmp(h_scales.data(), r_scales.data(), scales_bytes) == 0);

    std::cout << "[D2D_ZeroSizeOptional] vnni=" << std::boolalpha << vnni_ok
              << " scales=" << scales_ok
              << " (mins/emins skipped as expected)" << std::endl;

    EXPECT_TRUE(vnni_ok);
    EXPECT_TRUE(scales_ok);

    freeOnDevice(d_src_vnni, src_dev);
    freeOnDevice(d_src_scales, src_dev);
    freeOnDevice(d_dst_vnni, dst_dev);
    freeOnDevice(d_dst_scales, dst_dev);
#endif
}

} // namespace
