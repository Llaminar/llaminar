/**
 * @file Test__MoEGroupedIntermediateExchange.cpp
 * @brief Captured byte-contract proof for CUDA/HIP expert activation exchange.
 *
 * A single device models multiple logical producer packets to isolate the
 * protocol from transport. Native multi-device collectives have separate tests.
 * Producer and consumer grouping permutations are deliberately unrelated; every
 * replay changes ownership and active rows without rebinding or recapturing.
 * Raw words include NaN payloads, signed zero and extreme scale mantissas so any
 * accidental FP conversion, quantization or participant reduction is visible.
 */
#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include "kernels/common/MoEGroupedIntermediateExchangeKernels.h"
#include "transfer/TransferEngine.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Explicit packet contracts; no runtime fallback between them. */
    enum class PacketShape { SparseCapacity, CompactRows };
    /** @brief Keep failures exception-safe inside the exact GPU worker context. */
    void require(bool condition, const char *message)
    {
        if (!condition) throw std::runtime_error(message);
    }

    /** @return The selected backend's same shared pack implementation. */
    bool pack(DeviceId device, const MoEGroupedIntermediatePackLaunch &launch, void *stream)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda()) return cuda::packGroupedIntermediate(launch, stream);
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) return rocm::packGroupedIntermediate(launch, stream);
#endif
        throw std::invalid_argument("intermediate test requires an enabled GPU backend");
    }

    /** @return The selected backend's same shared consume implementation. */
    bool consume(DeviceId device, const MoEGroupedIntermediateConsumeLaunch &launch, void *stream)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda()) return cuda::consumeGroupedIntermediate(launch, stream);
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) return rocm::consumeGroupedIntermediate(launch, stream);
#endif
        throw std::invalid_argument("intermediate test requires an enabled GPU backend");
    }

    /** @brief Record the exact compact producer, without host-computing the count. */
    bool packCompact(DeviceId device, const MoECompactIntermediatePackLaunch &launch, void *stream)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda()) return cuda::packCompactIntermediate(launch, stream);
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) return rocm::packCompactIntermediate(launch, stream);
#endif
        throw std::invalid_argument("compact intermediate requires an enabled GPU backend");
    }

    /** @brief Record the exact compact consumer with the device-published extent. */
    bool consumeCompact(DeviceId device, const MoECompactIntermediateConsumeLaunch &launch, void *stream)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda()) return cuda::consumeCompactIntermediate(launch, stream);
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) return rocm::consumeCompactIntermediate(launch, stream);
#endif
        throw std::invalid_argument("compact intermediate requires an enabled GPU backend");
    }

    /** @brief Persistent setup-only allocations outlive the complete graph. */
    struct Storage final
    {
        DeviceId device;
        IBackend *backend;
        void *stream;
        std::vector<std::shared_ptr<DeviceTransferBuffer>> owners;
        std::unique_ptr<IGPUGraphCapture> graph;

        /** @brief Drain only at test teardown, before retiring recorded pointers. */
        ~Storage()
        {
            (void)backend->synchronizeStream(stream, device.ordinal);
            graph.reset();
        }

        /** @return TransferEngine-owned persistent storage for the requested plane. */
        template <typename T> T *allocate(std::size_t count)
        {
            auto owner = TransferEngine::instance().allocateDeviceTransferBuffer(count * sizeof(T), device);
            require(owner && owner->isBound(), "intermediate allocation failed");
            auto *pointer = static_cast<T *>(owner->mutableDeviceData());
            owners.push_back(std::move(owner));
            return pointer;
        }

        /** @brief Publish inputs outside capture; retain host bytes until completion. */
        template <typename T> void upload(T *pointer, const std::vector<T> &values)
        {
            require(backend->hostToDeviceOnStream(pointer, values.data(), values.size() * sizeof(T),
                device.ordinal, stream), "intermediate upload failed");
            require(backend->synchronizeStream(stream, device.ordinal), "intermediate setup join failed");
        }

        /** @return Observed bytes after the retained graph's terminal boundary. */
        template <typename T> std::vector<T> download(T *pointer, std::size_t count)
        {
            std::vector<T> values(count);
            require(backend->deviceToHostOnStream(values.data(), pointer, count * sizeof(T), device.ordinal, stream),
                "intermediate download failed");
            require(backend->synchronizeStream(stream, device.ordinal), "intermediate observation join failed");
            return values;
        }
    };

    /** @brief Prove one complete geometry through twenty ownership/data epochs. */
    void check(DeviceId device, MoEGroupedIntermediateLayout layout, PacketShape shape = PacketShape::SparseCapacity)
    {
        auto &worker = GPUDeviceContextPool::instance().getContext(device);
        worker.submitAndWait([&]
        {
            Storage storage{device, getBackendFor(device), worker.getOrCreateAuxiliaryStream("intermediate_byte_proof")};
            constexpr std::size_t guards = 13;
            constexpr std::uint32_t sentinel = 0x7fc05a5au;
            const auto capacity = layout.route_capacity;
            const auto value_words = capacity * layout.valueWords();
            const auto scale_words = capacity * layout.scaleWords();
            const bool compact = shape == PacketShape::CompactRows;
            const auto packet_words = (compact ? layout.compactCapacityBytes() : layout.packetBytes()) / sizeof(std::uint32_t);
            auto *owners = storage.allocate<std::int32_t>(capacity);
            auto *inverse = storage.allocate<std::int32_t>(capacity);
            auto *packets = storage.allocate<std::uint32_t>(packet_words * layout.participants + guards);
            auto *output = storage.allocate<std::uint32_t>(value_words + guards);
            auto *scales = scale_words ? storage.allocate<std::uint32_t>(scale_words + guards) : nullptr;
            auto *group_ends = compact ? storage.allocate<std::int32_t>(2 * layout.participants) : nullptr;
            auto *packet_bytes = compact ? storage.allocate<std::uint64_t>(layout.participants) : nullptr;
            std::vector<MoEGroupedIntermediatePackLaunch> producers;
            for (std::uint32_t participant = 0; participant < layout.participants; ++participant)
                producers.push_back({.layout = layout, .route_owners = owners,
                    .original_to_grouped = storage.allocate<std::int32_t>(capacity),
                    .grouped_values = storage.allocate<std::uint32_t>(value_words),
                    .grouped_scales = scale_words ? reinterpret_cast<float *>(storage.allocate<std::uint32_t>(scale_words)) : nullptr,
                    .packet = packets + participant * packet_words, .participant = static_cast<std::int32_t>(participant)});
            MoEGroupedIntermediateConsumeLaunch consumer{.layout = layout, .route_owners = owners,
                .original_to_grouped = inverse, .participant_packets = packets, .grouped_values = output,
                .grouped_scales = reinterpret_cast<float *>(scales)};
            // Default streams are invalid even with otherwise valid bindings.
            for (void *invalid : std::array<void *, 3>{nullptr, reinterpret_cast<void *>(1), reinterpret_cast<void *>(2)})
            {
                EXPECT_FALSE(pack(device, producers.front(), invalid));
                EXPECT_FALSE(consume(device, consumer, invalid));
            }
            std::vector<std::uint32_t> expected_output(value_words + guards, sentinel);
            std::vector<std::uint32_t> expected_scales(scale_words + guards, sentinel);
            std::vector<std::uint32_t> expected_packets(packet_words * layout.participants + guards, sentinel);
            storage.upload(output, expected_output);
            if (scales) storage.upload(scales, expected_scales);
            storage.upload(packets, std::vector<std::uint32_t>(packet_words * layout.participants + guards, sentinel));
            storage.graph = worker.createGraphCapture(storage.stream);
            require(storage.graph && storage.graph->beginCapture(), "intermediate capture begin failed");
            for (std::uint32_t participant = 0; participant < layout.participants; ++participant)
                if (compact)
                {
                    const MoECompactIntermediatePackLaunch launch{producers[participant],
                        group_ends + 2 * participant, group_ends + 2 * participant + 1, packet_bytes + participant};
                    require(packCompact(device, launch, storage.stream), "compact pack capture failed");
                }
                else require(pack(device, producers[participant], storage.stream), "intermediate pack capture failed");
            if (compact)
                for (std::uint32_t participant = 0; participant < layout.participants; ++participant)
                {
                    const MoECompactIntermediateConsumeLaunch launch{.layout = layout, .route_owners = owners,
                        .original_to_grouped = inverse, .packet = packets + participant * packet_words,
                        .packet_bytes = packet_bytes + participant, .participant = static_cast<std::int32_t>(participant),
                        .grouped_values = output, .grouped_scales = reinterpret_cast<float *>(scales)};
                    require(consumeCompact(device, launch, storage.stream), "compact consume capture failed");
                }
            else require(consume(device, consumer, storage.stream), "intermediate consume capture failed");
            require(storage.graph->endCapture() && storage.graph->instantiate(), "intermediate capture end failed");
            for (int epoch = 0; epoch < 20; ++epoch)
            {
                SCOPED_TRACE(::testing::Message() << "columns=" << layout.columns << " capacity=" << capacity
                    << " participants=" << layout.participants << " epoch=" << epoch);
                std::mt19937 rng(731u + epoch);
                std::vector<std::int32_t> order(capacity), owner_map(capacity, -1), consumer_map(capacity, -1);
                std::iota(order.begin(), order.end(), 0);
                std::shuffle(order.begin(), order.end(), rng);
                const auto live = epoch % 4 == 2 ? 0u : (epoch % 4 == 1 ? capacity / 2 : capacity);
                for (std::uint32_t grouped = 0; grouped < live; ++grouped)
                {
                    const int slot = order[grouped];
                    owner_map[slot] = epoch % 4 == 3 ? epoch % layout.participants : (slot + epoch) % layout.participants;
                    consumer_map[slot] = grouped;
                }
                storage.upload(owners, owner_map);
                storage.upload(inverse, consumer_map);
                if (!compact) std::fill(expected_packets.begin(), expected_packets.end() - guards, 0);
                std::vector<std::int32_t> ends(2 * layout.participants, 0);
                std::vector<std::uint64_t> expected_counts(layout.participants, 0);
                for (std::uint32_t participant = 0; participant < layout.participants; ++participant)
                {
                    std::shuffle(order.begin(), order.end(), rng);
                    std::vector<std::int32_t> map(capacity, -1);
                    std::vector<std::uint32_t> values(value_words, sentinel), block_scales(scale_words, sentinel);
                    std::uint32_t grouped = 0;
                    for (const int slot : order)
                    {
                        if (owner_map[slot] != static_cast<int>(participant)) continue;
                        map[slot] = grouped;
                        const auto record = participant * packet_words + (compact
                            ? grouped * layout.compactRecordWords() : slot * layout.routeWords());
                        if (compact) expected_packets[record] = slot;
                        for (std::size_t word = 0; word < layout.routeWords(); ++word)
                        {
                            constexpr std::array<std::uint32_t, 8> special{0, 0x80000000u, 0x7f800000u,
                                0xff800000u, 0x7fc12345u, 1, 0x00800000u, 0xffffffffu};
                            const std::uint32_t value = word < special.size() ? special[word] : rng();
                            expected_packets[record + word + (compact ? 1 : 0)] = value;
                            if (word < layout.valueWords())
                            {
                                values[grouped * layout.valueWords() + word] = value;
                                expected_output[consumer_map[slot] * layout.valueWords() + word] = value;
                            }
                            else
                            {
                                block_scales[grouped * layout.scaleWords() + word - layout.valueWords()] = value;
                                expected_scales[consumer_map[slot] * layout.scaleWords() + word - layout.valueWords()] = value;
                            }
                        }
                        ++grouped;
                    }
                    const auto &producer = producers[participant];
                    storage.upload(const_cast<std::int32_t *>(producer.original_to_grouped), map);
                    storage.upload(static_cast<std::uint32_t *>(const_cast<void *>(producer.grouped_values)), values);
                    if (scale_words)
                        storage.upload(reinterpret_cast<std::uint32_t *>(const_cast<float *>(producer.grouped_scales)), block_scales);
                    ends[2 * participant] = grouped / 2;
                    ends[2 * participant + 1] = grouped - grouped / 2;
                    expected_counts[participant] = grouped * layout.compactRecordWords() * sizeof(std::uint32_t);
                }
                if (compact) storage.upload(group_ends, ends);
                require(storage.graph->launch(), "intermediate replay failed");
                EXPECT_EQ(storage.download(packets, expected_packets.size()), expected_packets);
                if (compact) EXPECT_EQ(storage.download(packet_bytes, expected_counts.size()), expected_counts);
                EXPECT_EQ(storage.download(output, expected_output.size()), expected_output);
                if (scales) EXPECT_EQ(storage.download(scales, expected_scales.size()), expected_scales);
            }
        });
    }

    /** @brief Cover quant-block tails, odd FP32 widths and arbitrary logical peers. */
    void sweep(DeviceId device, PacketShape shape = PacketShape::SparseCapacity)
    {
        for (auto encoding : {MoEGroupedIntermediateEncoding::BlockQ8FP32Scales, MoEGroupedIntermediateEncoding::FP32})
            for (auto columns : {32u, 96u, 512u})
                for (auto peers : {1u, 2u, 3u, 8u})
                    check(device, {encoding, columns, 130u, peers}, shape);
        check(device, {MoEGroupedIntermediateEncoding::FP32, 513u, 3u, 3u}, shape);
        check(device, {MoEGroupedIntermediateEncoding::BlockQ8FP32Scales, 512u, 4096u, 2u}, shape);
    }
}

#ifdef HAVE_CUDA
TEST(MoEGroupedIntermediateExchange, CUDAReplayIsBitExact) { sweep(DeviceId::cuda(0)); }
TEST(MoEGroupedIntermediateExchange, CUDACompactReplayIsBitExact) { sweep(DeviceId::cuda(0), PacketShape::CompactRows); }
/** @brief Isolate one production-sized geometry for launch/resource profiling. */
TEST(MoEGroupedIntermediateExchange, CUDAExactPacketProfile)
{
    check(DeviceId::cuda(0), {MoEGroupedIntermediateEncoding::BlockQ8FP32Scales, 512, 3584, 2});
}
/** @brief Isolate one compact Q8 geometry without profiling a peer-waiting collective. */
TEST(MoEGroupedIntermediateExchange, CUDACompactQ8PacketProfile)
{
    check(DeviceId::cuda(0), {MoEGroupedIntermediateEncoding::BlockQ8FP32Scales, 512, 4096, 2}, PacketShape::CompactRows);
}
/** @brief Profile floating packets separately so their traffic cannot contaminate Q8 evidence. */
TEST(MoEGroupedIntermediateExchange, CUDACompactFP32PacketProfile)
{
    check(DeviceId::cuda(0), {MoEGroupedIntermediateEncoding::FP32, 512, 4096, 2}, PacketShape::CompactRows);
}
#endif
#ifdef HAVE_ROCM
TEST(MoEGroupedIntermediateExchange, ROCmReplayIsBitExact) { sweep(DeviceId::rocm(0)); }
TEST(MoEGroupedIntermediateExchange, ROCmCompactReplayIsBitExact) { sweep(DeviceId::rocm(0), PacketShape::CompactRows); }
/** @brief Isolate one production-sized geometry for launch/resource profiling. */
TEST(MoEGroupedIntermediateExchange, ROCmExactPacketProfile)
{
    check(DeviceId::rocm(0), {MoEGroupedIntermediateEncoding::BlockQ8FP32Scales, 512, 3584, 2});
}
/** @brief Isolate the same compact Q8 geometry on HIP, with no cross-device wait to profile. */
TEST(MoEGroupedIntermediateExchange, ROCmCompactQ8PacketProfile)
{
    check(DeviceId::rocm(0), {MoEGroupedIntermediateEncoding::BlockQ8FP32Scales, 512, 4096, 2}, PacketShape::CompactRows);
}
/** @brief Isolate floating packet resource/traffic evidence from the quantized sweep. */
TEST(MoEGroupedIntermediateExchange, ROCmCompactFP32PacketProfile)
{
    check(DeviceId::rocm(0), {MoEGroupedIntermediateEncoding::FP32, 512, 4096, 2}, PacketShape::CompactRows);
}
#endif
