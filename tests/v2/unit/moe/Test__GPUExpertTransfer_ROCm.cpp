/**
 * @file Test__GPUExpertTransfer_ROCm.cpp
 * @brief ROCm D2D coverage for GPU expert packed transfer.
 */

#include <gtest/gtest.h>

#include "backends/DeviceId.h"
#include "execution/moe/GPUExpertTransfer.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace llaminar2;

namespace
{
    int rocmDeviceCount()
    {
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        return err == hipSuccess ? count : 0;
    }

    template <typename T>
    T *allocROCm(int ordinal, size_t count)
    {
        (void)hipSetDevice(ordinal);
        T *ptr = nullptr;
        hipError_t err = hipMalloc(&ptr, count * sizeof(T));
        return err == hipSuccess ? ptr : nullptr;
    }

    template <typename T>
    bool uploadROCm(T *dst, const T *src, size_t count, int ordinal)
    {
        (void)hipSetDevice(ordinal);
        return hipMemcpy(dst, src, count * sizeof(T), hipMemcpyHostToDevice) == hipSuccess;
    }

    template <typename T>
    bool downloadROCm(T *dst, const T *src, size_t count, int ordinal)
    {
        (void)hipSetDevice(ordinal);
        return hipMemcpy(dst, src, count * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess;
    }

    void freeROCm(void *ptr, int ordinal)
    {
        if (!ptr)
            return;
        (void)hipSetDevice(ordinal);
        (void)hipFree(ptr);
    }

    void requireTwoROCmDevices()
    {
        const int count = rocmDeviceCount();
        if (count < 2)
            GTEST_SKIP() << "Need >= 2 ROCm devices, have " << count;
    }

    void runROCmD2DTransfer(bool include_optional_arrays, int repetitions = 1)
    {
        requireTwoROCmDevices();

        const int src_dev = 0;
        const int dst_dev = 1;
        const size_t vnni_bytes = include_optional_arrays ? 512 * 1024 : 256 * 1024;
        const size_t block_count = include_optional_arrays ? 2048 : 1024;
        const size_t scales_bytes = block_count * sizeof(uint16_t);
        const size_t mins_bytes = include_optional_arrays ? block_count * sizeof(uint16_t) : 0;
        const size_t emins_bytes = include_optional_arrays ? block_count * sizeof(uint32_t) : 0;

        std::mt19937 rng(include_optional_arrays ? 123 : 999);
        std::vector<uint8_t> h_vnni(vnni_bytes);
        std::vector<uint16_t> h_scales(block_count);
        std::vector<uint16_t> h_mins(block_count);
        std::vector<uint32_t> h_emins(block_count);
        for (auto &b : h_vnni)
            b = static_cast<uint8_t>(rng() & 0xFF);
        for (auto &s : h_scales)
            s = static_cast<uint16_t>(rng() & 0xFFFF);
        for (auto &m : h_mins)
            m = static_cast<uint16_t>(rng() & 0xFFFF);
        for (auto &e : h_emins)
            e = static_cast<uint32_t>(rng());

        auto *d_src_vnni = allocROCm<uint8_t>(src_dev, vnni_bytes);
        auto *d_src_scales = allocROCm<uint16_t>(src_dev, block_count);
        auto *d_src_mins = include_optional_arrays ? allocROCm<uint16_t>(src_dev, block_count) : nullptr;
        auto *d_src_emins = include_optional_arrays ? allocROCm<uint32_t>(src_dev, block_count) : nullptr;
        ASSERT_NE(d_src_vnni, nullptr);
        ASSERT_NE(d_src_scales, nullptr);
        ASSERT_TRUE(uploadROCm(d_src_vnni, h_vnni.data(), vnni_bytes, src_dev));
        ASSERT_TRUE(uploadROCm(d_src_scales, h_scales.data(), block_count, src_dev));
        if (include_optional_arrays)
        {
            ASSERT_NE(d_src_mins, nullptr);
            ASSERT_NE(d_src_emins, nullptr);
            ASSERT_TRUE(uploadROCm(d_src_mins, h_mins.data(), block_count, src_dev));
            ASSERT_TRUE(uploadROCm(d_src_emins, h_emins.data(), block_count, src_dev));
        }

        auto *d_dst_vnni = allocROCm<uint8_t>(dst_dev, vnni_bytes);
        auto *d_dst_scales = allocROCm<uint16_t>(dst_dev, block_count);
        auto *d_dst_mins = include_optional_arrays ? allocROCm<uint16_t>(dst_dev, block_count) : nullptr;
        auto *d_dst_emins = include_optional_arrays ? allocROCm<uint32_t>(dst_dev, block_count) : nullptr;
        ASSERT_NE(d_dst_vnni, nullptr);
        ASSERT_NE(d_dst_scales, nullptr);
        if (include_optional_arrays)
        {
            ASSERT_NE(d_dst_mins, nullptr);
            ASSERT_NE(d_dst_emins, nullptr);
        }

        GPUExpertPointers src_ptrs;
        src_ptrs.d_vnni = d_src_vnni;
        src_ptrs.d_scales = d_src_scales;
        src_ptrs.d_mins = d_src_mins;
        src_ptrs.d_emins = d_src_emins;

        GPUExpertPointers dst_ptrs;
        dst_ptrs.d_vnni = d_dst_vnni;
        dst_ptrs.d_scales = d_dst_scales;
        dst_ptrs.d_mins = d_dst_mins;
        dst_ptrs.d_emins = d_dst_emins;

        (void)hipSetDevice(dst_dev);
        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        for (int i = 0; i < repetitions; ++i)
        {
            ASSERT_TRUE(GPUExpertTransfer::transferExpert(
                src_ptrs,
                dst_ptrs,
                DeviceId::rocm(src_dev),
                DeviceId::rocm(dst_dev),
                vnni_bytes,
                scales_bytes,
                mins_bytes,
                emins_bytes,
                stream));
        }

        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

        std::vector<uint8_t> r_vnni(vnni_bytes);
        std::vector<uint16_t> r_scales(block_count);
        std::vector<uint16_t> r_mins(block_count);
        std::vector<uint32_t> r_emins(block_count);
        ASSERT_TRUE(downloadROCm(r_vnni.data(), d_dst_vnni, vnni_bytes, dst_dev));
        ASSERT_TRUE(downloadROCm(r_scales.data(), d_dst_scales, block_count, dst_dev));
        if (include_optional_arrays)
        {
            ASSERT_TRUE(downloadROCm(r_mins.data(), d_dst_mins, block_count, dst_dev));
            ASSERT_TRUE(downloadROCm(r_emins.data(), d_dst_emins, block_count, dst_dev));
        }

        EXPECT_EQ(std::memcmp(h_vnni.data(), r_vnni.data(), vnni_bytes), 0);
        EXPECT_EQ(std::memcmp(h_scales.data(), r_scales.data(), scales_bytes), 0);
        if (include_optional_arrays)
        {
            EXPECT_EQ(std::memcmp(h_mins.data(), r_mins.data(), mins_bytes), 0);
            EXPECT_EQ(std::memcmp(h_emins.data(), r_emins.data(), emins_bytes), 0);
        }

        freeROCm(d_src_vnni, src_dev);
        freeROCm(d_src_scales, src_dev);
        freeROCm(d_src_mins, src_dev);
        freeROCm(d_src_emins, src_dev);
        freeROCm(d_dst_vnni, dst_dev);
        freeROCm(d_dst_scales, dst_dev);
        freeROCm(d_dst_mins, dst_dev);
        freeROCm(d_dst_emins, dst_dev);
    }

    GpuExpertPackedDescriptor makeROCmDescriptor(
        uint8_t *vnni,
        uint16_t *scales,
        uint16_t *mins,
        uint32_t *emins,
        int n,
        int k,
        uint8_t payload_bytes_per_block)
    {
        const uint32_t blocks_per_row = static_cast<uint32_t>(k / 32);
        const size_t block_count =
            static_cast<size_t>(n) * static_cast<size_t>(blocks_per_row);

        GpuExpertPackedDescriptor desc;
        desc.ptrs.d_vnni = vnni;
        desc.ptrs.d_scales = scales;
        desc.ptrs.d_mins = mins;
        desc.ptrs.d_emins = emins;
        desc.n = n;
        desc.k = k;
        desc.blocks_per_row = blocks_per_row;
        desc.codebook_id = 4;
        desc.payload_bytes_per_block = payload_bytes_per_block;
        desc.is_asymmetric = true;
        desc.has_emins = true;
        desc.vnni_bytes = block_count * payload_bytes_per_block;
        desc.scales_bytes = block_count * sizeof(uint16_t);
        desc.mins_bytes = block_count * sizeof(uint16_t);
        desc.emins_bytes = block_count * sizeof(uint32_t);
        return desc;
    }

    void runROCmStagedActivation()
    {
        requireTwoROCmDevices();

        const int src_dev = 0;
        const int dst_dev = 1;
        const int n = 128;
        const int k = 64;
        const uint8_t payload_bytes_per_block = 32;
        const size_t block_count = static_cast<size_t>(n) * static_cast<size_t>(k / 32);
        const size_t vnni_bytes = block_count * payload_bytes_per_block;

        std::mt19937 rng(20260626);
        std::vector<uint8_t> h_vnni(vnni_bytes);
        std::vector<uint16_t> h_scales(block_count);
        std::vector<uint16_t> h_mins(block_count);
        std::vector<uint32_t> h_emins(block_count);
        for (auto &b : h_vnni)
            b = static_cast<uint8_t>(rng() & 0xFF);
        for (auto &s : h_scales)
            s = static_cast<uint16_t>(rng() & 0xFFFF);
        for (auto &m : h_mins)
            m = static_cast<uint16_t>(rng() & 0xFFFF);
        for (auto &e : h_emins)
            e = static_cast<uint32_t>(rng());

        auto *d_src_vnni = allocROCm<uint8_t>(src_dev, vnni_bytes);
        auto *d_src_scales = allocROCm<uint16_t>(src_dev, block_count);
        auto *d_src_mins = allocROCm<uint16_t>(src_dev, block_count);
        auto *d_src_emins = allocROCm<uint32_t>(src_dev, block_count);
        auto *d_staged_vnni = allocROCm<uint8_t>(dst_dev, vnni_bytes);
        auto *d_staged_scales = allocROCm<uint16_t>(dst_dev, block_count);
        auto *d_staged_mins = allocROCm<uint16_t>(dst_dev, block_count);
        auto *d_staged_emins = allocROCm<uint32_t>(dst_dev, block_count);
        auto *d_active_vnni = allocROCm<uint8_t>(dst_dev, vnni_bytes);
        auto *d_active_scales = allocROCm<uint16_t>(dst_dev, block_count);
        auto *d_active_mins = allocROCm<uint16_t>(dst_dev, block_count);
        auto *d_active_emins = allocROCm<uint32_t>(dst_dev, block_count);
        auto *d_active2_vnni = allocROCm<uint8_t>(dst_dev, vnni_bytes);
        auto *d_active2_scales = allocROCm<uint16_t>(dst_dev, block_count);
        auto *d_active2_mins = allocROCm<uint16_t>(dst_dev, block_count);
        auto *d_active2_emins = allocROCm<uint32_t>(dst_dev, block_count);
        ASSERT_NE(d_src_vnni, nullptr);
        ASSERT_NE(d_src_scales, nullptr);
        ASSERT_NE(d_src_mins, nullptr);
        ASSERT_NE(d_src_emins, nullptr);
        ASSERT_NE(d_staged_vnni, nullptr);
        ASSERT_NE(d_staged_scales, nullptr);
        ASSERT_NE(d_staged_mins, nullptr);
        ASSERT_NE(d_staged_emins, nullptr);
        ASSERT_NE(d_active_vnni, nullptr);
        ASSERT_NE(d_active_scales, nullptr);
        ASSERT_NE(d_active_mins, nullptr);
        ASSERT_NE(d_active_emins, nullptr);
        ASSERT_NE(d_active2_vnni, nullptr);
        ASSERT_NE(d_active2_scales, nullptr);
        ASSERT_NE(d_active2_mins, nullptr);
        ASSERT_NE(d_active2_emins, nullptr);

        ASSERT_TRUE(uploadROCm(d_src_vnni, h_vnni.data(), vnni_bytes, src_dev));
        ASSERT_TRUE(uploadROCm(d_src_scales, h_scales.data(), block_count, src_dev));
        ASSERT_TRUE(uploadROCm(d_src_mins, h_mins.data(), block_count, src_dev));
        ASSERT_TRUE(uploadROCm(d_src_emins, h_emins.data(), block_count, src_dev));

        auto src_desc = makeROCmDescriptor(
            d_src_vnni, d_src_scales, d_src_mins, d_src_emins,
            n, k, payload_bytes_per_block);
        auto staged_desc = makeROCmDescriptor(
            d_staged_vnni, d_staged_scales, d_staged_mins, d_staged_emins,
            n, k, payload_bytes_per_block);
        auto active_desc = makeROCmDescriptor(
            d_active_vnni, d_active_scales, d_active_mins, d_active_emins,
            n, k, payload_bytes_per_block);
        auto active2_desc = makeROCmDescriptor(
            d_active2_vnni, d_active2_scales, d_active2_mins, d_active2_emins,
            n, k, payload_bytes_per_block);

        EXPECT_FALSE(GPUExpertTransfer::activateStagedExpert(
            staged_desc,
            active_desc,
            DeviceId::rocm(dst_dev),
            nullptr))
            << "staged activation must never silently fall back to the HIP legacy/default stream";
        std::vector<GpuExpertStagedActivation> activation_batch{
            {staged_desc, active_desc},
            {staged_desc, active2_desc}};
        EXPECT_FALSE(GPUExpertTransfer::activateStagedExperts(
            activation_batch,
            DeviceId::rocm(dst_dev),
            nullptr))
            << "staged activation batches must never silently fall back to the HIP legacy/default stream";

        (void)hipSetDevice(dst_dev);
        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        ASSERT_TRUE(GPUExpertTransfer::transferExpert(
            src_desc,
            staged_desc,
            DeviceId::rocm(src_dev),
            DeviceId::rocm(dst_dev),
            stream));
        ASSERT_TRUE(GPUExpertTransfer::activateStagedExperts(
            activation_batch,
            DeviceId::rocm(dst_dev),
            stream));
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

        std::vector<uint8_t> r_vnni(vnni_bytes);
        std::vector<uint16_t> r_scales(block_count);
        std::vector<uint16_t> r_mins(block_count);
        std::vector<uint32_t> r_emins(block_count);
        std::vector<uint8_t> r2_vnni(vnni_bytes);
        std::vector<uint16_t> r2_scales(block_count);
        std::vector<uint16_t> r2_mins(block_count);
        std::vector<uint32_t> r2_emins(block_count);
        ASSERT_TRUE(downloadROCm(r_vnni.data(), d_active_vnni, vnni_bytes, dst_dev));
        ASSERT_TRUE(downloadROCm(r_scales.data(), d_active_scales, block_count, dst_dev));
        ASSERT_TRUE(downloadROCm(r_mins.data(), d_active_mins, block_count, dst_dev));
        ASSERT_TRUE(downloadROCm(r_emins.data(), d_active_emins, block_count, dst_dev));
        ASSERT_TRUE(downloadROCm(r2_vnni.data(), d_active2_vnni, vnni_bytes, dst_dev));
        ASSERT_TRUE(downloadROCm(r2_scales.data(), d_active2_scales, block_count, dst_dev));
        ASSERT_TRUE(downloadROCm(r2_mins.data(), d_active2_mins, block_count, dst_dev));
        ASSERT_TRUE(downloadROCm(r2_emins.data(), d_active2_emins, block_count, dst_dev));

        EXPECT_EQ(std::memcmp(h_vnni.data(), r_vnni.data(), vnni_bytes), 0);
        EXPECT_EQ(std::memcmp(h_scales.data(), r_scales.data(), block_count * sizeof(uint16_t)), 0);
        EXPECT_EQ(std::memcmp(h_mins.data(), r_mins.data(), block_count * sizeof(uint16_t)), 0);
        EXPECT_EQ(std::memcmp(h_emins.data(), r_emins.data(), block_count * sizeof(uint32_t)), 0);
        EXPECT_EQ(std::memcmp(h_vnni.data(), r2_vnni.data(), vnni_bytes), 0);
        EXPECT_EQ(std::memcmp(h_scales.data(), r2_scales.data(), block_count * sizeof(uint16_t)), 0);
        EXPECT_EQ(std::memcmp(h_mins.data(), r2_mins.data(), block_count * sizeof(uint16_t)), 0);
        EXPECT_EQ(std::memcmp(h_emins.data(), r2_emins.data(), block_count * sizeof(uint32_t)), 0);

        freeROCm(d_src_vnni, src_dev);
        freeROCm(d_src_scales, src_dev);
        freeROCm(d_src_mins, src_dev);
        freeROCm(d_src_emins, src_dev);
        freeROCm(d_staged_vnni, dst_dev);
        freeROCm(d_staged_scales, dst_dev);
        freeROCm(d_staged_mins, dst_dev);
        freeROCm(d_staged_emins, dst_dev);
        freeROCm(d_active_vnni, dst_dev);
        freeROCm(d_active_scales, dst_dev);
        freeROCm(d_active_mins, dst_dev);
        freeROCm(d_active_emins, dst_dev);
        freeROCm(d_active2_vnni, dst_dev);
        freeROCm(d_active2_scales, dst_dev);
        freeROCm(d_active2_mins, dst_dev);
        freeROCm(d_active2_emins, dst_dev);
    }
}

TEST(Test__GPUExpertTransferROCm, PeerAccessCheck)
{
    requireTwoROCmDevices();

    const bool can_access_0_to_1 =
        GPUExpertTransfer::canAccessPeer(DeviceId::rocm(0), DeviceId::rocm(1));
    const bool can_access_1_to_0 =
        GPUExpertTransfer::canAccessPeer(DeviceId::rocm(1), DeviceId::rocm(0));
    const bool can_access_self =
        GPUExpertTransfer::canAccessPeer(DeviceId::rocm(0), DeviceId::rocm(0));

    std::cout << "[PeerAccessCheck ROCm] 0->1: " << std::boolalpha << can_access_0_to_1
              << " 1->0: " << can_access_1_to_0
              << " 0->0: " << can_access_self << std::endl;
    EXPECT_TRUE(can_access_self);
}

TEST(Test__GPUExpertTransferROCm, D2DTransfer)
{
    runROCmD2DTransfer(/*include_optional_arrays=*/false);
}

TEST(Test__GPUExpertTransferROCm, D2DTransferAllArrays)
{
    runROCmD2DTransfer(/*include_optional_arrays=*/true);
}

TEST(Test__GPUExpertTransferROCm, RepeatedD2DTransferUsesCachedPeerState)
{
    runROCmD2DTransfer(/*include_optional_arrays=*/false, /*repetitions=*/3);
}

TEST(Test__GPUExpertTransferROCm, StagedActivationCopiesTransferSlotIntoActiveSlot)
{
    runROCmStagedActivation();
}
