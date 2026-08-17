/**
 * @file Test__GPUExpertTransfer_CUDA.cpp
 * @brief CUDA D2D coverage for packed and floating expert transfers.
 *
 * In addition to the separated NativeVNNI layout, this suite moves raw FP16,
 * BF16, and FP32 projection payloads through the persistent production peer
 * lane. Every case joins an exact producer event, observes completion only by
 * polling the destination event, and compares every bit at the destination.
 */

#include <gtest/gtest.h>

#include "backends/DeviceId.h"
#include "execution/moe/ExpertTierGpuPeerTransferLane.h"
#include "execution/moe/GPUExpertTransfer.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <chrono>
#include <random>
#include <string>
#include <thread>
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

    void runCudaD2DTransfer(bool include_optional_arrays, int repetitions = 1)
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
        for (int i = 0; i < repetitions; ++i)
        {
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
        }

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

    GpuExpertPackedDescriptor makeCudaDescriptor(
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

    void runCudaStagedActivation()
    {
        requireTwoCudaDevices();

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

        auto *d_src_vnni = allocCuda<uint8_t>(src_dev, vnni_bytes);
        auto *d_src_scales = allocCuda<uint16_t>(src_dev, block_count);
        auto *d_src_mins = allocCuda<uint16_t>(src_dev, block_count);
        auto *d_src_emins = allocCuda<uint32_t>(src_dev, block_count);
        auto *d_staged_vnni = allocCuda<uint8_t>(dst_dev, vnni_bytes);
        auto *d_staged_scales = allocCuda<uint16_t>(dst_dev, block_count);
        auto *d_staged_mins = allocCuda<uint16_t>(dst_dev, block_count);
        auto *d_staged_emins = allocCuda<uint32_t>(dst_dev, block_count);
        auto *d_active_vnni = allocCuda<uint8_t>(dst_dev, vnni_bytes);
        auto *d_active_scales = allocCuda<uint16_t>(dst_dev, block_count);
        auto *d_active_mins = allocCuda<uint16_t>(dst_dev, block_count);
        auto *d_active_emins = allocCuda<uint32_t>(dst_dev, block_count);
        auto *d_active2_vnni = allocCuda<uint8_t>(dst_dev, vnni_bytes);
        auto *d_active2_scales = allocCuda<uint16_t>(dst_dev, block_count);
        auto *d_active2_mins = allocCuda<uint16_t>(dst_dev, block_count);
        auto *d_active2_emins = allocCuda<uint32_t>(dst_dev, block_count);
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

        ASSERT_TRUE(uploadCuda(d_src_vnni, h_vnni.data(), vnni_bytes, src_dev));
        ASSERT_TRUE(uploadCuda(d_src_scales, h_scales.data(), block_count, src_dev));
        ASSERT_TRUE(uploadCuda(d_src_mins, h_mins.data(), block_count, src_dev));
        ASSERT_TRUE(uploadCuda(d_src_emins, h_emins.data(), block_count, src_dev));

        auto src_desc = makeCudaDescriptor(
            d_src_vnni, d_src_scales, d_src_mins, d_src_emins,
            n, k, payload_bytes_per_block);
        auto staged_desc = makeCudaDescriptor(
            d_staged_vnni, d_staged_scales, d_staged_mins, d_staged_emins,
            n, k, payload_bytes_per_block);
        auto active_desc = makeCudaDescriptor(
            d_active_vnni, d_active_scales, d_active_mins, d_active_emins,
            n, k, payload_bytes_per_block);
        auto active2_desc = makeCudaDescriptor(
            d_active2_vnni, d_active2_scales, d_active2_mins, d_active2_emins,
            n, k, payload_bytes_per_block);

        EXPECT_FALSE(GPUExpertTransfer::activateStagedExpert(
            staged_desc,
            active_desc,
            DeviceId::cuda(dst_dev),
            nullptr))
            << "staged activation must never silently fall back to the CUDA legacy/default stream";
        std::vector<GpuExpertStagedActivation> activation_batch{
            {staged_desc, active_desc},
            {staged_desc, active2_desc}};
        EXPECT_FALSE(GPUExpertTransfer::activateStagedExperts(
            activation_batch,
            DeviceId::cuda(dst_dev),
            nullptr))
            << "staged activation batches must never silently fall back to the CUDA legacy/default stream";

        (void)cudaSetDevice(dst_dev);
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_TRUE(GPUExpertTransfer::transferExpert(
            src_desc,
            staged_desc,
            DeviceId::cuda(src_dev),
            DeviceId::cuda(dst_dev),
            stream));
        ASSERT_TRUE(GPUExpertTransfer::activateStagedExperts(
            activation_batch,
            DeviceId::cuda(dst_dev),
            stream));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

        std::vector<uint8_t> r_vnni(vnni_bytes);
        std::vector<uint16_t> r_scales(block_count);
        std::vector<uint16_t> r_mins(block_count);
        std::vector<uint32_t> r_emins(block_count);
        std::vector<uint8_t> r2_vnni(vnni_bytes);
        std::vector<uint16_t> r2_scales(block_count);
        std::vector<uint16_t> r2_mins(block_count);
        std::vector<uint32_t> r2_emins(block_count);
        ASSERT_TRUE(downloadCuda(r_vnni.data(), d_active_vnni, vnni_bytes, dst_dev));
        ASSERT_TRUE(downloadCuda(r_scales.data(), d_active_scales, block_count, dst_dev));
        ASSERT_TRUE(downloadCuda(r_mins.data(), d_active_mins, block_count, dst_dev));
        ASSERT_TRUE(downloadCuda(r_emins.data(), d_active_emins, block_count, dst_dev));
        ASSERT_TRUE(downloadCuda(r2_vnni.data(), d_active2_vnni, vnni_bytes, dst_dev));
        ASSERT_TRUE(downloadCuda(r2_scales.data(), d_active2_scales, block_count, dst_dev));
        ASSERT_TRUE(downloadCuda(r2_mins.data(), d_active2_mins, block_count, dst_dev));
        ASSERT_TRUE(downloadCuda(r2_emins.data(), d_active2_emins, block_count, dst_dev));

        EXPECT_EQ(std::memcmp(h_vnni.data(), r_vnni.data(), vnni_bytes), 0);
        EXPECT_EQ(std::memcmp(h_scales.data(), r_scales.data(), block_count * sizeof(uint16_t)), 0);
        EXPECT_EQ(std::memcmp(h_mins.data(), r_mins.data(), block_count * sizeof(uint16_t)), 0);
        EXPECT_EQ(std::memcmp(h_emins.data(), r_emins.data(), block_count * sizeof(uint32_t)), 0);
        EXPECT_EQ(std::memcmp(h_vnni.data(), r2_vnni.data(), vnni_bytes), 0);
        EXPECT_EQ(std::memcmp(h_scales.data(), r2_scales.data(), block_count * sizeof(uint16_t)), 0);
        EXPECT_EQ(std::memcmp(h_mins.data(), r2_mins.data(), block_count * sizeof(uint16_t)), 0);
        EXPECT_EQ(std::memcmp(h_emins.data(), r2_emins.data(), block_count * sizeof(uint32_t)), 0);

        freeCuda(d_src_vnni, src_dev);
        freeCuda(d_src_scales, src_dev);
        freeCuda(d_src_mins, src_dev);
        freeCuda(d_src_emins, src_dev);
        freeCuda(d_staged_vnni, dst_dev);
        freeCuda(d_staged_scales, dst_dev);
        freeCuda(d_staged_mins, dst_dev);
        freeCuda(d_staged_emins, dst_dev);
        freeCuda(d_active_vnni, dst_dev);
        freeCuda(d_active_scales, dst_dev);
        freeCuda(d_active_mins, dst_dev);
        freeCuda(d_active_emins, dst_dev);
        freeCuda(d_active2_vnni, dst_dev);
        freeCuda(d_active2_scales, dst_dev);
        freeCuda(d_active2_mins, dst_dev);
        freeCuda(d_active2_emins, dst_dev);
    }

    /**
     * @brief Exercise the persistent peer lane with a cross-device event edge.
     *
     * The maintenance thread starts from the source device deliberately. The
     * lane must submit on its destination-owned auxiliary stream, restore the
     * caller context, and reach readiness solely through event queries.
     */
    void runCudaPeerLaneTransfer()
    {
        requireTwoCudaDevices();
        constexpr int src_dev = 0;
        constexpr int dst_dev = 1;
        constexpr int n = 2048;
        constexpr int k = 256;
        constexpr uint8_t payload_bytes_per_block = 16;
        constexpr size_t block_count =
            static_cast<size_t>(n) * static_cast<size_t>(k / 32);
        constexpr size_t vnni_bytes =
            block_count * payload_bytes_per_block;

        std::mt19937 rng(0xC0FFEEu);
        std::vector<uint8_t> expected_vnni(vnni_bytes);
        std::vector<uint16_t> expected_scales(block_count);
        std::vector<uint16_t> expected_mins(block_count);
        std::vector<uint32_t> expected_emins(block_count);
        for (auto &value : expected_vnni)
            value = static_cast<uint8_t>(rng());
        for (auto &value : expected_scales)
            value = static_cast<uint16_t>(rng());
        for (auto &value : expected_mins)
            value = static_cast<uint16_t>(rng());
        for (auto &value : expected_emins)
            value = rng();

        auto *src_vnni = allocCuda<uint8_t>(src_dev, vnni_bytes);
        auto *src_scales = allocCuda<uint16_t>(src_dev, block_count);
        auto *src_mins = allocCuda<uint16_t>(src_dev, block_count);
        auto *src_emins = allocCuda<uint32_t>(src_dev, block_count);
        auto *dst_vnni = allocCuda<uint8_t>(dst_dev, vnni_bytes);
        auto *dst_scales = allocCuda<uint16_t>(dst_dev, block_count);
        auto *dst_mins = allocCuda<uint16_t>(dst_dev, block_count);
        auto *dst_emins = allocCuda<uint32_t>(dst_dev, block_count);
        ASSERT_NE(src_vnni, nullptr);
        ASSERT_NE(src_scales, nullptr);
        ASSERT_NE(src_mins, nullptr);
        ASSERT_NE(src_emins, nullptr);
        ASSERT_NE(dst_vnni, nullptr);
        ASSERT_NE(dst_scales, nullptr);
        ASSERT_NE(dst_mins, nullptr);
        ASSERT_NE(dst_emins, nullptr);

        ASSERT_TRUE(uploadCuda(
            src_vnni, expected_vnni.data(), vnni_bytes, src_dev));
        ASSERT_TRUE(uploadCuda(
            src_scales, expected_scales.data(), block_count, src_dev));
        ASSERT_TRUE(uploadCuda(
            src_mins, expected_mins.data(), block_count, src_dev));
        ASSERT_TRUE(uploadCuda(
            src_emins, expected_emins.data(), block_count, src_dev));

        const auto source = makeCudaDescriptor(
            src_vnni, src_scales, src_mins, src_emins,
            n, k, payload_bytes_per_block);
        const auto destination = makeCudaDescriptor(
            dst_vnni, dst_scales, dst_mins, dst_emins,
            n, k, payload_bytes_per_block);

        (void)cudaSetDevice(src_dev);
        cudaStream_t producer_stream = nullptr;
        cudaEvent_t source_ready = nullptr;
        ASSERT_EQ(
            cudaStreamCreateWithFlags(
                &producer_stream, cudaStreamNonBlocking),
            cudaSuccess);
        ASSERT_EQ(
            cudaEventCreateWithFlags(
                &source_ready, cudaEventDisableTiming),
            cudaSuccess);
        ASSERT_EQ(cudaEventRecord(source_ready, producer_stream), cudaSuccess);

        {
            ExpertTierGpuPeerTransferLane lane({
                .source_device = DeviceId::cuda(src_dev),
                .destination_device = DeviceId::cuda(dst_dev),
                .lane_name = "cuda_peer_event_polled",
                .perf_device = "cuda-peer",
                .collect_timing_measurements = true,
            });
            std::string error;
            ASSERT_TRUE(lane.materialize(&error)) << error;
            ASSERT_EQ(cudaSetDevice(src_dev), cudaSuccess);
            ASSERT_TRUE(lane.start(
                source,
                destination,
                ExpertTierSourceReadiness::producerEvent(source_ready),
                &error)) << error;

            int current_device = -1;
            ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
            EXPECT_EQ(current_device, src_dev);

            const auto deadline = std::chrono::steady_clock::now() +
                                  std::chrono::seconds(10);
            while (lane.progress() ==
                       ExpertTierGpuPeerTransferProgress::Pending &&
                   std::chrono::steady_clock::now() < deadline)
            {
                ASSERT_NE(
                    lane.poll(&error),
                    ExpertTierGpuPeerTransferProgress::Failed)
                    << error;
                std::this_thread::yield();
            }
            ASSERT_EQ(
                lane.progress(),
                ExpertTierGpuPeerTransferProgress::Ready)
                << error;

            /*
             * Reuse the persistent lane from an installed RCU bank.  This path
             * must not fabricate or wait on a producer event: the retained
             * epoch is the source-byte readiness and lifetime authority.
             */
            ASSERT_EQ(cudaSetDevice(src_dev), cudaSuccess);
            ASSERT_TRUE(lane.start(
                source,
                destination,
                ExpertTierSourceReadiness::publishedResidencyBank(17),
                &error)) << error;
            while (lane.progress() ==
                       ExpertTierGpuPeerTransferProgress::Pending &&
                   std::chrono::steady_clock::now() < deadline)
            {
                ASSERT_NE(
                    lane.poll(&error),
                    ExpertTierGpuPeerTransferProgress::Failed)
                    << error;
                std::this_thread::yield();
            }
            ASSERT_EQ(
                lane.progress(),
                ExpertTierGpuPeerTransferProgress::Ready)
                << error;
            ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
            EXPECT_EQ(current_device, src_dev);

            const auto stats = lane.stats();
            EXPECT_EQ(stats.transfers_started, 2u);
            EXPECT_EQ(stats.transfers_completed, 2u);
            EXPECT_EQ(stats.bytes_submitted, source.totalBytes() * 2u);
            EXPECT_EQ(stats.producer_event_waits, 1u);
            EXPECT_EQ(stats.published_bank_sources, 1u);
            EXPECT_EQ(stats.timing_measurement_failures, 0u);
            EXPECT_TRUE(stats.last_measurement.valid());
            EXPECT_EQ(stats.last_measurement.bytes, source.totalBytes());
            EXPECT_GT(stats.last_measurement.device_nanoseconds, 0u);
            EXPECT_EQ(stats.inference_stream_waits, 0u);
            EXPECT_EQ(stats.blocking_synchronizations, 0u);
        }

        std::vector<uint8_t> actual_vnni(vnni_bytes);
        std::vector<uint16_t> actual_scales(block_count);
        std::vector<uint16_t> actual_mins(block_count);
        std::vector<uint32_t> actual_emins(block_count);
        ASSERT_TRUE(downloadCuda(
            actual_vnni.data(), dst_vnni, vnni_bytes, dst_dev));
        ASSERT_TRUE(downloadCuda(
            actual_scales.data(), dst_scales, block_count, dst_dev));
        ASSERT_TRUE(downloadCuda(
            actual_mins.data(), dst_mins, block_count, dst_dev));
        ASSERT_TRUE(downloadCuda(
            actual_emins.data(), dst_emins, block_count, dst_dev));
        EXPECT_EQ(actual_vnni, expected_vnni);
        EXPECT_EQ(actual_scales, expected_scales);
        EXPECT_EQ(actual_mins, expected_mins);
        EXPECT_EQ(actual_emins, expected_emins);

        (void)cudaSetDevice(src_dev);
        ASSERT_EQ(cudaEventDestroy(source_ready), cudaSuccess);
        ASSERT_EQ(cudaStreamDestroy(producer_stream), cudaSuccess);
        freeCuda(src_vnni, src_dev);
        freeCuda(src_scales, src_dev);
        freeCuda(src_mins, src_dev);
        freeCuda(src_emins, src_dev);
        freeCuda(dst_vnni, dst_dev);
        freeCuda(dst_scales, dst_dev);
        freeCuda(dst_mins, dst_dev);
        freeCuda(dst_emins, dst_dev);
    }

    /**
     * @brief Prove one contiguous floating projection uses peer DMA unchanged.
     *
     * Floating formats are intentionally treated as opaque bytes by tier
     * movement. The element width controls the production-shaped allocation;
     * arbitrary bit patterns (including values that would decode as NaN) prove
     * the lane never interprets or converts a same-backend payload.
     *
     * @param precision_name Stable test/evidence identity (fp16, bf16, fp32).
     * @param element_bytes Bytes per scalar in the floating representation.
     * @param seed Deterministic source-pattern seed.
     */
    void runCudaContiguousPeerLaneTransfer(
        const char *precision_name,
        std::size_t element_bytes,
        std::uint32_t seed)
    {
        requireTwoCudaDevices();
        ASSERT_TRUE(element_bytes == 2 || element_bytes == 4);
        constexpr int src_dev = 0;
        constexpr int dst_dev = 1;
        constexpr std::size_t element_count = 32771;
        const std::size_t bytes = element_count * element_bytes;

        std::vector<std::uint8_t> expected(bytes);
        for (std::size_t index = 0; index < expected.size(); ++index)
        {
            expected[index] = static_cast<std::uint8_t>(
                (index * 43u + seed * 17u + (index >> 3u)) & 0xffu);
        }

        auto *source = allocCuda<std::uint8_t>(src_dev, bytes);
        auto *destination = allocCuda<std::uint8_t>(dst_dev, bytes);
        ASSERT_NE(source, nullptr);
        ASSERT_NE(destination, nullptr);

        ASSERT_EQ(cudaSetDevice(src_dev), cudaSuccess);
        cudaStream_t producer_stream = nullptr;
        cudaEvent_t source_ready = nullptr;
        ASSERT_EQ(
            cudaStreamCreateWithFlags(
                &producer_stream, cudaStreamNonBlocking),
            cudaSuccess);
        ASSERT_EQ(
            cudaEventCreateWithFlags(&source_ready, cudaEventDisableTiming),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpyAsync(
                source,
                expected.data(),
                bytes,
                cudaMemcpyHostToDevice,
                producer_stream),
            cudaSuccess);
        ASSERT_EQ(cudaEventRecord(source_ready, producer_stream), cudaSuccess);

        {
            ExpertTierGpuPeerTransferLane lane({
                .source_device = DeviceId::cuda(src_dev),
                .destination_device = DeviceId::cuda(dst_dev),
                .lane_name = std::string("cuda_peer_contiguous_") +
                             precision_name,
                .perf_device = "cuda-peer-floating",
                .collect_timing_measurements = true,
            });
            std::string error;
            ASSERT_TRUE(lane.materialize(&error)) << error;

            /* Start from the source device to catch ambient-device coupling. */
            ASSERT_EQ(cudaSetDevice(src_dev), cudaSuccess);
            ASSERT_TRUE(lane.startContiguous(
                source,
                destination,
                bytes,
                ExpertTierSourceReadiness::producerEvent(source_ready),
                &error))
                << error;

            int current_device = -1;
            ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
            EXPECT_EQ(current_device, src_dev);

            const auto deadline = std::chrono::steady_clock::now() +
                                  std::chrono::seconds(10);
            while (lane.progress() ==
                       ExpertTierGpuPeerTransferProgress::Pending &&
                   std::chrono::steady_clock::now() < deadline)
            {
                ASSERT_NE(
                    lane.poll(&error),
                    ExpertTierGpuPeerTransferProgress::Failed)
                    << error;
                std::this_thread::yield();
            }
            ASSERT_EQ(
                lane.progress(), ExpertTierGpuPeerTransferProgress::Ready)
                << error;

            const auto stats = lane.stats();
            EXPECT_EQ(stats.transfers_started, 1u);
            EXPECT_EQ(stats.transfers_completed, 1u);
            EXPECT_EQ(stats.bytes_submitted, bytes);
            EXPECT_EQ(stats.producer_event_waits, 1u);
            EXPECT_EQ(stats.published_bank_sources, 0u);
            EXPECT_EQ(stats.failed_transfers, 0u);
            EXPECT_EQ(stats.timing_measurement_failures, 0u);
            EXPECT_TRUE(stats.last_measurement.valid());
            EXPECT_EQ(stats.last_measurement.bytes, bytes);
            EXPECT_GT(stats.last_measurement.device_nanoseconds, 0u);
            EXPECT_EQ(stats.inference_stream_waits, 0u);
            EXPECT_EQ(stats.blocking_synchronizations, 0u);
        }

        std::vector<std::uint8_t> actual(bytes);
        ASSERT_TRUE(downloadCuda(
            actual.data(), destination, bytes, dst_dev));
        EXPECT_EQ(actual, expected);

        ASSERT_EQ(cudaSetDevice(src_dev), cudaSuccess);
        ASSERT_EQ(cudaEventDestroy(source_ready), cudaSuccess);
        ASSERT_EQ(cudaStreamDestroy(producer_stream), cudaSuccess);
        freeCuda(source, src_dev);
        freeCuda(destination, dst_dev);
    }
}

TEST(Test__GPUExpertTransferCUDA, D2DTransfer)
{
    runCudaD2DTransfer(/*include_optional_arrays=*/false);
}

TEST(Test__GPUExpertTransferCUDA, D2DTransferAllArrays)
{
    runCudaD2DTransfer(/*include_optional_arrays=*/true);
}

TEST(Test__GPUExpertTransferCUDA, RepeatedD2DTransfer)
{
    runCudaD2DTransfer(/*include_optional_arrays=*/false, /*repetitions=*/3);
}

TEST(Test__GPUExpertTransferCUDA, StagedActivationCopiesTransferSlotIntoActiveSlot)
{
    runCudaStagedActivation();
}

TEST(Test__ExpertTierGpuPeerTransferCUDA, EventPolledTransferIsByteExact)
{
    runCudaPeerLaneTransfer();
}

TEST(Test__ExpertTierGpuPeerTransferCUDA, FP16ContiguousTransferIsByteExact)
{
    runCudaContiguousPeerLaneTransfer("fp16", 2, 0xF016u);
}

TEST(Test__ExpertTierGpuPeerTransferCUDA, BF16ContiguousTransferIsByteExact)
{
    runCudaContiguousPeerLaneTransfer("bf16", 2, 0xBF16u);
}

TEST(Test__ExpertTierGpuPeerTransferCUDA, FP32ContiguousTransferIsByteExact)
{
    runCudaContiguousPeerLaneTransfer("fp32", 4, 0xF032u);
}
