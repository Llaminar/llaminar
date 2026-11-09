/**
 * @file Test__GPUExpertTransfer_ROCm.cpp
 * @brief ROCm D2D coverage for packed and floating expert transfers.
 *
 * In addition to the separated NativeQuantised layout, this suite moves raw FP16,
 * BF16, and FP32 projection payloads through the persistent production peer
 * lane. Every case joins an exact producer event, observes completion only by
 * polling the destination event, and compares every bit at the destination.
 * TransferEngine prepares the destination-owned execution stream; each lane
 * retains its typed pool lease until all event-polled work and cleanup finish.
 */

#include <gtest/gtest.h>

#include "backends/DeviceId.h"
#include "transfer/TransferEngine.h"
#include "execution/moe/ExpertTierGpuPeerTransferLane.h"
#include "execution/moe/GPUExpertTransfer.h"
#include "execution/moe/GPUExpertTransferBackend.h"
#include "kernels/qwen4exp/Qwen4ExpStateIO.h"

#include <hip/hip_runtime.h>

#include <array>
#include <exception>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <optional>
#include <random>
#include <string>
#include <thread>
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

    /** Directed peer edge whose destination stream may read source memory. */
    struct DirectedROCmPeerPair
    {
        int source = -1;      ///< Device that owns the source allocation.
        int destination = -1; ///< Device whose stream owns the peer copy.
    };

    /**
     * @brief Find a driver-authorized directed HIP peer edge on this host.
     *
     * PCIe/IOMMU topology is not implied by ordinal order and can change when
     * cards move between sockets. The production lane deliberately rejects an
     * unauthorized edge, so this hardware certificate selects an actual edge
     * instead of weakening that invariant or assuming devices zero and one.
     *
     * @return First stable directed edge, or empty when this host exposes none.
     */
    std::optional<DirectedROCmPeerPair> findDirectedROCmPeerPair()
    {
        const int count = rocmDeviceCount();
        for (int destination = 0; destination < count; ++destination)
        {
            for (int source = 0; source < count; ++source)
            {
                if (source == destination)
                    continue;
                int can_access = 0;
                if (hipDeviceCanAccessPeer(
                        &can_access, destination, source) == hipSuccess &&
                    can_access != 0)
                {
                    return DirectedROCmPeerPair{
                        .source = source,
                        .destination = destination,
                    };
                }
            }
        }
        return std::nullopt;
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

    /**
     * @brief Prove destination-stream peer DMA and event-only completion on HIP.
     *
     * Starting from the source device catches accidental dependence on the
     * caller's current HIP device while exact byte comparisons cover all four
     * separated NativeQuantised regions.
     */
    void runROCmPeerLaneTransfer()
    {
        const auto peer_pair = findDirectedROCmPeerPair();
        if (!peer_pair.has_value())
        {
            GTEST_SKIP()
                << "No driver-authorized directed ROCm peer edge across "
                << rocmDeviceCount() << " visible devices";
        }
        const int src_dev = peer_pair->source;
        const int dst_dev = peer_pair->destination;
        constexpr int n = 2048;
        constexpr int k = 256;
        constexpr uint8_t payload_bytes_per_block = 16;
        constexpr size_t block_count =
            static_cast<size_t>(n) * static_cast<size_t>(k / 32);
        constexpr size_t vnni_bytes =
            block_count * payload_bytes_per_block;

        std::mt19937 rng(0xBADC0DEu);
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

        auto *src_vnni = allocROCm<uint8_t>(src_dev, vnni_bytes);
        auto *src_scales = allocROCm<uint16_t>(src_dev, block_count);
        auto *src_mins = allocROCm<uint16_t>(src_dev, block_count);
        auto *src_emins = allocROCm<uint32_t>(src_dev, block_count);
        auto *dst_vnni = allocROCm<uint8_t>(dst_dev, vnni_bytes);
        auto *dst_scales = allocROCm<uint16_t>(dst_dev, block_count);
        auto *dst_mins = allocROCm<uint16_t>(dst_dev, block_count);
        auto *dst_emins = allocROCm<uint32_t>(dst_dev, block_count);
        ASSERT_NE(src_vnni, nullptr);
        ASSERT_NE(src_scales, nullptr);
        ASSERT_NE(src_mins, nullptr);
        ASSERT_NE(src_emins, nullptr);
        ASSERT_NE(dst_vnni, nullptr);
        ASSERT_NE(dst_scales, nullptr);
        ASSERT_NE(dst_mins, nullptr);
        ASSERT_NE(dst_emins, nullptr);

        ASSERT_TRUE(uploadROCm(
            src_vnni, expected_vnni.data(), vnni_bytes, src_dev));
        ASSERT_TRUE(uploadROCm(
            src_scales, expected_scales.data(), block_count, src_dev));
        ASSERT_TRUE(uploadROCm(
            src_mins, expected_mins.data(), block_count, src_dev));
        ASSERT_TRUE(uploadROCm(
            src_emins, expected_emins.data(), block_count, src_dev));

        const auto source = makeROCmDescriptor(
            src_vnni, src_scales, src_mins, src_emins,
            n, k, payload_bytes_per_block);
        const auto destination = makeROCmDescriptor(
            dst_vnni, dst_scales, dst_mins, dst_emins,
            n, k, payload_bytes_per_block);

        (void)hipSetDevice(src_dev);
        hipStream_t producer_stream = nullptr;
        hipEvent_t source_ready = nullptr;
        ASSERT_EQ(
            hipStreamCreateWithFlags(
                &producer_stream, hipStreamNonBlocking),
            hipSuccess);
        ASSERT_EQ(
            hipEventCreateWithFlags(
                &source_ready, hipEventDisableTiming),
            hipSuccess);
        ASSERT_EQ(hipEventRecord(source_ready, producer_stream), hipSuccess);

        {
            ExpertTierGpuPeerTransferLane lane({
                .source_device = DeviceId::rocm(src_dev),
                .destination_device = DeviceId::rocm(dst_dev),
                .execution = TransferEngine::instance()
                                 .allocatePersistentTransferExecutionLanes(
                                     1u, DeviceId::rocm(dst_dev),
                                     "rocm_peer_event_polled")
                                 .front(),
                .lane_name = "rocm_peer_event_polled",
                .perf_device = "rocm-peer",
                .collect_timing_measurements = true,
            });
            std::string error;
            ASSERT_TRUE(lane.materialize(&error)) << error;
            ASSERT_EQ(hipSetDevice(src_dev), hipSuccess);
            ASSERT_TRUE(lane.start(
                source,
                destination,
                ExpertTierSourceReadiness::producerEvent(source_ready),
                &error)) << error;

            int current_device = -1;
            ASSERT_EQ(hipGetDevice(&current_device), hipSuccess);
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

            /* Installed-bank readiness is epoch proof, not a synthetic event. */
            ASSERT_EQ(hipSetDevice(src_dev), hipSuccess);
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
            ASSERT_EQ(hipGetDevice(&current_device), hipSuccess);
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
        ASSERT_TRUE(downloadROCm(
            actual_vnni.data(), dst_vnni, vnni_bytes, dst_dev));
        ASSERT_TRUE(downloadROCm(
            actual_scales.data(), dst_scales, block_count, dst_dev));
        ASSERT_TRUE(downloadROCm(
            actual_mins.data(), dst_mins, block_count, dst_dev));
        ASSERT_TRUE(downloadROCm(
            actual_emins.data(), dst_emins, block_count, dst_dev));
        EXPECT_EQ(actual_vnni, expected_vnni);
        EXPECT_EQ(actual_scales, expected_scales);
        EXPECT_EQ(actual_mins, expected_mins);
        EXPECT_EQ(actual_emins, expected_emins);

        (void)hipSetDevice(src_dev);
        ASSERT_EQ(hipEventDestroy(source_ready), hipSuccess);
        ASSERT_EQ(hipStreamDestroy(producer_stream), hipSuccess);
        freeROCm(src_vnni, src_dev);
        freeROCm(src_scales, src_dev);
        freeROCm(src_mins, src_dev);
        freeROCm(src_emins, src_dev);
        freeROCm(dst_vnni, dst_dev);
        freeROCm(dst_scales, dst_dev);
        freeROCm(dst_mins, dst_dev);
        freeROCm(dst_emins, dst_dev);
    }

    /**
     * @brief Prove one contiguous floating projection uses peer DMA unchanged.
     *
     * Same-backend movement treats floating representations as opaque bytes.
     * Arbitrary bit patterns ensure the lane cannot silently reinterpret,
     * normalize, or repack FP16, BF16, or FP32 payloads.
     *
     * @param precision_name Stable test/evidence identity (fp16, bf16, fp32).
     * @param element_bytes Bytes per scalar in the floating representation.
     * @param seed Deterministic source-pattern seed.
     */
    void runROCmContiguousPeerLaneTransfer(
        const char *precision_name,
        std::size_t element_bytes,
        std::uint32_t seed)
    {
        ASSERT_TRUE(element_bytes == 2 || element_bytes == 4);
        const auto peer_pair = findDirectedROCmPeerPair();
        if (!peer_pair.has_value())
        {
            GTEST_SKIP()
                << "No driver-authorized directed ROCm peer edge across "
                << rocmDeviceCount() << " visible devices";
        }
        const int src_dev = peer_pair->source;
        const int dst_dev = peer_pair->destination;
        constexpr std::size_t element_count = 32771;
        const std::size_t bytes = element_count * element_bytes;

        std::vector<std::uint8_t> expected(bytes);
        for (std::size_t index = 0; index < expected.size(); ++index)
        {
            expected[index] = static_cast<std::uint8_t>(
                (index * 47u + seed * 19u + (index >> 2u)) & 0xffu);
        }

        auto *source = allocROCm<std::uint8_t>(src_dev, bytes);
        auto *destination = allocROCm<std::uint8_t>(dst_dev, bytes);
        ASSERT_NE(source, nullptr);
        ASSERT_NE(destination, nullptr);

        ASSERT_EQ(hipSetDevice(src_dev), hipSuccess);
        hipStream_t producer_stream = nullptr;
        hipEvent_t source_ready = nullptr;
        ASSERT_EQ(
            hipStreamCreateWithFlags(
                &producer_stream, hipStreamNonBlocking),
            hipSuccess);
        ASSERT_EQ(
            hipEventCreateWithFlags(&source_ready, hipEventDisableTiming),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(
                source,
                expected.data(),
                bytes,
                hipMemcpyHostToDevice,
                producer_stream),
            hipSuccess);
        ASSERT_EQ(hipEventRecord(source_ready, producer_stream), hipSuccess);

        {
            ExpertTierGpuPeerTransferLane lane({
                .source_device = DeviceId::rocm(src_dev),
                .destination_device = DeviceId::rocm(dst_dev),
                .execution = TransferEngine::instance()
                                 .allocatePersistentTransferExecutionLanes(
                                     1u, DeviceId::rocm(dst_dev),
                                     std::string("rocm_peer_contiguous_") + precision_name)
                                 .front(),
                .lane_name = std::string("rocm_peer_contiguous_") +
                             precision_name,
                .perf_device = "rocm-peer-floating",
                .collect_timing_measurements = true,
            });
            std::string error;
            ASSERT_TRUE(lane.materialize(&error)) << error;

            /* Start from the source device to catch ambient-device coupling. */
            ASSERT_EQ(hipSetDevice(src_dev), hipSuccess);
            ASSERT_TRUE(lane.startContiguous(
                source,
                destination,
                bytes,
                ExpertTierSourceReadiness::producerEvent(source_ready),
                &error))
                << error;

            int current_device = -1;
            ASSERT_EQ(hipGetDevice(&current_device), hipSuccess);
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
        ASSERT_TRUE(downloadROCm(
            actual.data(), destination, bytes, dst_dev));
        EXPECT_EQ(actual, expected);

        ASSERT_EQ(hipSetDevice(src_dev), hipSuccess);
        ASSERT_EQ(hipEventDestroy(source_ready), hipSuccess);
        ASSERT_EQ(hipStreamDestroy(producer_stream), hipSuccess);
        freeROCm(source, src_dev);
        freeROCm(destination, dst_dev);
    }
}

TEST(Test__GPUExpertTransferROCm, D2DTransfer)
{
    runROCmD2DTransfer(/*include_optional_arrays=*/false);
}

TEST(Test__GPUExpertTransferROCm, D2DTransferAllArrays)
{
    runROCmD2DTransfer(/*include_optional_arrays=*/true);
}

TEST(Test__GPUExpertTransferROCm, RepeatedD2DTransfer)
{
    runROCmD2DTransfer(/*include_optional_arrays=*/false, /*repetitions=*/3);
}

TEST(Test__GPUExpertTransferROCm, StagedActivationCopiesTransferSlotIntoActiveSlot)
{
    runROCmStagedActivation();
}

TEST(Test__ExpertTierGpuPeerTransferROCm, EventPolledTransferIsByteExact)
{
    runROCmPeerLaneTransfer();
}

TEST(Test__ExpertTierGpuPeerTransferROCm, FP16ContiguousTransferIsByteExact)
{
    runROCmContiguousPeerLaneTransfer("fp16", 2, 0xF016u);
}

TEST(Test__ExpertTierGpuPeerTransferROCm, BF16ContiguousTransferIsByteExact)
{
    runROCmContiguousPeerLaneTransfer("bf16", 2, 0xBF16u);
}

TEST(Test__ExpertTierGpuPeerTransferROCm, FP32ContiguousTransferIsByteExact)
{
    runROCmContiguousPeerLaneTransfer("fp32", 4, 0xF032u);
}


namespace
{
    /** @brief Own one tiny retained reset graph, including failed-test cleanup. */
    struct ROCmPeerResetResources
    {
        int ordinal;
        hipStream_t stream=nullptr;
        hipGraph_t graph=nullptr;
        hipGraphExec_t executable=nullptr;
        std::uint32_t *words=nullptr;
        /** @brief Retire the exact test stream before releasing its captured bytes. */
        ~ROCmPeerResetResources()
        {
            if(hipSetDevice(ordinal)!=hipSuccess)std::terminate();
            if(stream && hipStreamSynchronize(stream)!=hipSuccess)std::terminate();
            if(executable && hipGraphExecDestroy(executable)!=hipSuccess)std::terminate();
            if(graph && hipGraphDestroy(graph)!=hipSuccess)std::terminate();
            if(words && hipFree(words)!=hipSuccess)std::terminate();
            if(stream && hipStreamDestroy(stream)!=hipSuccess)std::terminate();
        }
    };

    /**
     * @brief Replay an actual partial QFN reset after repeated direct-peer setup.
     * @param source Actual native source ordinal.
     * @param destination Actual native consumer ordinal.
     * @param direct_peer Whether the queried directed edge is physically supported.
     *
     * Peer setup and ordinary auxiliary reset stay on this SAME host thread.
     * A fresh thread isolates the defect from unrelated earlier test cases;
     * no last-error clear is used to make either setup or reset pass.
     */
    void runROCmPeerPreparationReset(int source,int destination,bool direct_peer)
    {
        ASSERT_EQ(hipPeekAtLastError(),hipSuccess);
        ASSERT_EQ(hipSetDevice(destination),hipSuccess);
        ROCmPeerResetResources owner{destination};
        constexpr std::size_t capacity=37,live=17;
        constexpr std::uint32_t captured_bits=0x13579BDFu;
        ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&owner.words),capacity*sizeof(std::uint32_t)),hipSuccess);
        ASSERT_EQ(hipStreamCreateWithFlags(&owner.stream,hipStreamNonBlocking),hipSuccess);
        ASSERT_EQ(hipStreamBeginCapture(owner.stream,hipStreamCaptureModeThreadLocal),hipSuccess);
        ASSERT_NO_THROW(qwen4exp::rocmQwenStateFill({owner.words,live,captured_bits},owner.stream));
        ASSERT_EQ(hipStreamEndCapture(owner.stream,&owner.graph),hipSuccess);
        ASSERT_EQ(hipGraphInstantiate(&owner.executable,owner.graph,nullptr,nullptr,0),hipSuccess);
        for(unsigned replay=0;replay<20;++replay)
        {
            SCOPED_TRACE(replay);
            ASSERT_EQ(hipSetDevice(source),hipSuccess);
            ASSERT_EQ(detail::prepareDirectPeerAccessROCmBackend(DeviceId::rocm(source),DeviceId::rocm(destination)),direct_peer);
            ASSERT_EQ(detail::prepareDirectPeerAccessROCmBackend(DeviceId::rocm(source),DeviceId::rocm(destination)),direct_peer);
            int ambient=-1;ASSERT_EQ(hipGetDevice(&ambient),hipSuccess);ASSERT_EQ(ambient,source);
            ASSERT_EQ(hipSetDevice(destination),hipSuccess);
            const std::uint32_t poison=0xAA550000u+replay;
            // This is the ordinary cache-owned reset wrapper, before replay.
            // Repeated setup must retire only its own AlreadyEnabled status.
            ASSERT_NO_THROW(qwen4exp::rocmQwenStateFill({owner.words,capacity,poison},owner.stream));
            ASSERT_EQ(hipPeekAtLastError(),hipSuccess);
            ASSERT_EQ(hipGraphLaunch(owner.executable,owner.stream),hipSuccess);
            std::array<std::uint32_t,capacity> observed{};
            ASSERT_EQ(hipMemcpyAsync(observed.data(),owner.words,sizeof(observed),hipMemcpyDeviceToHost,owner.stream),hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(owner.stream),hipSuccess);
            for(std::size_t index=0;index<capacity;++index)
                ASSERT_EQ(observed[index],index<live ? captured_bits : poison)<<index;
        }
    }
}

TEST(Test__ExpertTierGpuPeerTransferROCm, RepeatedPreparationPreservesCapturedAuxiliaryReset)
{
    if(rocmDeviceCount()<2)GTEST_SKIP()<<"Requires two actual ROCm devices";
    const auto directed=findDirectedROCmPeerPair();
    const int source=directed ? directed->source : 0;
    const int destination=directed ? directed->destination : 1;
    const bool direct_peer=directed.has_value();
    RecordProperty("direct_peer_available",direct_peer ? 1 : 0);
    RecordProperty("source_ordinal",source);RecordProperty("destination_ordinal",destination);
    // Unsupported edges are an explicit negative capability: setup must return
    // false with clean error state while the device-local reset still works.
    std::thread worker([&]{runROCmPeerPreparationReset(source,destination,direct_peer);});
    worker.join();
}

TEST(Test__ExpertTierGpuPeerTransferROCm, PreparationPreservesUnrelatedPreexistingError)
{
    const int count=rocmDeviceCount();
    if(count<2)GTEST_SKIP()<<"Requires two actual ROCm devices";
    std::thread worker([&]{
        ASSERT_EQ(hipSetDevice(0),hipSuccess);
        ASSERT_EQ(hipPeekAtLastError(),hipSuccess);
        // The deliberately invalid ordinal is defined hip behavior, not an
        // invalid native pointer. Only this test retires its injected error.
        const auto injected=hipSetDevice(count);
        ASSERT_NE(injected,hipSuccess);
        ASSERT_EQ(hipPeekAtLastError(),injected);
        const bool prepared=detail::prepareDirectPeerAccessROCmBackend(DeviceId::rocm(0),DeviceId::rocm(1));
        EXPECT_FALSE(prepared);
        EXPECT_EQ(hipPeekAtLastError(),injected);
        EXPECT_EQ(hipGetLastError(),injected);
    });
    worker.join();
}
