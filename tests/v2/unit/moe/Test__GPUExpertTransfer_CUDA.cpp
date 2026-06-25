/**
 * @file Test__GPUExpertTransfer_CUDA.cpp
 * @brief CUDA D2D coverage for GPU expert packed transfer.
 */

#include <gtest/gtest.h>

#include "backends/DeviceId.h"
#include "execution/moe/GPUExpertTransfer.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace llaminar2;

namespace
{
    int cudaDeviceCount()
    {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        return err == cudaSuccess ? count : 0;
    }

    template <typename T>
    T *allocCuda(int ordinal, size_t count)
    {
        (void)cudaSetDevice(ordinal);
        T *ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));
        return err == cudaSuccess ? ptr : nullptr;
    }

    template <typename T>
    bool uploadCuda(T *dst, const T *src, size_t count, int ordinal)
    {
        (void)cudaSetDevice(ordinal);
        return cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
    }

    template <typename T>
    bool downloadCuda(T *dst, const T *src, size_t count, int ordinal)
    {
        (void)cudaSetDevice(ordinal);
        return cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    void freeCuda(void *ptr, int ordinal)
    {
        if (!ptr)
            return;
        (void)cudaSetDevice(ordinal);
        (void)cudaFree(ptr);
    }

    void requireTwoCudaDevices()
    {
        const int count = cudaDeviceCount();
        if (count < 2)
            GTEST_SKIP() << "Need >= 2 CUDA devices, have " << count;
    }

    void runCudaD2DTransfer(bool include_optional_arrays)
    {
        requireTwoCudaDevices();

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

        auto *d_src_vnni = allocCuda<uint8_t>(src_dev, vnni_bytes);
        auto *d_src_scales = allocCuda<uint16_t>(src_dev, block_count);
        auto *d_src_mins = include_optional_arrays ? allocCuda<uint16_t>(src_dev, block_count) : nullptr;
        auto *d_src_emins = include_optional_arrays ? allocCuda<uint32_t>(src_dev, block_count) : nullptr;
        ASSERT_NE(d_src_vnni, nullptr);
        ASSERT_NE(d_src_scales, nullptr);
        ASSERT_TRUE(uploadCuda(d_src_vnni, h_vnni.data(), vnni_bytes, src_dev));
        ASSERT_TRUE(uploadCuda(d_src_scales, h_scales.data(), block_count, src_dev));
        if (include_optional_arrays)
        {
            ASSERT_NE(d_src_mins, nullptr);
            ASSERT_NE(d_src_emins, nullptr);
            ASSERT_TRUE(uploadCuda(d_src_mins, h_mins.data(), block_count, src_dev));
            ASSERT_TRUE(uploadCuda(d_src_emins, h_emins.data(), block_count, src_dev));
        }

        auto *d_dst_vnni = allocCuda<uint8_t>(dst_dev, vnni_bytes);
        auto *d_dst_scales = allocCuda<uint16_t>(dst_dev, block_count);
        auto *d_dst_mins = include_optional_arrays ? allocCuda<uint16_t>(dst_dev, block_count) : nullptr;
        auto *d_dst_emins = include_optional_arrays ? allocCuda<uint32_t>(dst_dev, block_count) : nullptr;
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

        (void)cudaSetDevice(dst_dev);
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_TRUE(GPUExpertTransfer::transferExpert(
            src_ptrs,
            dst_ptrs,
            DeviceId::cuda(src_dev),
            DeviceId::cuda(dst_dev),
            vnni_bytes,
            scales_bytes,
            mins_bytes,
            emins_bytes,
            stream));

        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

        std::vector<uint8_t> r_vnni(vnni_bytes);
        std::vector<uint16_t> r_scales(block_count);
        std::vector<uint16_t> r_mins(block_count);
        std::vector<uint32_t> r_emins(block_count);
        ASSERT_TRUE(downloadCuda(r_vnni.data(), d_dst_vnni, vnni_bytes, dst_dev));
        ASSERT_TRUE(downloadCuda(r_scales.data(), d_dst_scales, block_count, dst_dev));
        if (include_optional_arrays)
        {
            ASSERT_TRUE(downloadCuda(r_mins.data(), d_dst_mins, block_count, dst_dev));
            ASSERT_TRUE(downloadCuda(r_emins.data(), d_dst_emins, block_count, dst_dev));
        }

        EXPECT_EQ(std::memcmp(h_vnni.data(), r_vnni.data(), vnni_bytes), 0);
        EXPECT_EQ(std::memcmp(h_scales.data(), r_scales.data(), scales_bytes), 0);
        if (include_optional_arrays)
        {
            EXPECT_EQ(std::memcmp(h_mins.data(), r_mins.data(), mins_bytes), 0);
            EXPECT_EQ(std::memcmp(h_emins.data(), r_emins.data(), emins_bytes), 0);
        }

        freeCuda(d_src_vnni, src_dev);
        freeCuda(d_src_scales, src_dev);
        freeCuda(d_src_mins, src_dev);
        freeCuda(d_src_emins, src_dev);
        freeCuda(d_dst_vnni, dst_dev);
        freeCuda(d_dst_scales, dst_dev);
        freeCuda(d_dst_mins, dst_dev);
        freeCuda(d_dst_emins, dst_dev);
    }
}

TEST(Test__GPUExpertTransferCUDA, PeerAccessCheck)
{
    requireTwoCudaDevices();

    const bool can_access_0_to_1 =
        GPUExpertTransfer::canAccessPeer(DeviceId::cuda(0), DeviceId::cuda(1));
    const bool can_access_1_to_0 =
        GPUExpertTransfer::canAccessPeer(DeviceId::cuda(1), DeviceId::cuda(0));
    const bool can_access_self =
        GPUExpertTransfer::canAccessPeer(DeviceId::cuda(0), DeviceId::cuda(0));

    std::cout << "[PeerAccessCheck CUDA] 0->1: " << std::boolalpha << can_access_0_to_1
              << " 1->0: " << can_access_1_to_0
              << " 0->0: " << can_access_self << std::endl;
    EXPECT_TRUE(can_access_self);
}

TEST(Test__GPUExpertTransferCUDA, D2DTransfer)
{
    runCudaD2DTransfer(/*include_optional_arrays=*/false);
}

TEST(Test__GPUExpertTransferCUDA, D2DTransferAllArrays)
{
    runCudaD2DTransfer(/*include_optional_arrays=*/true);
}
