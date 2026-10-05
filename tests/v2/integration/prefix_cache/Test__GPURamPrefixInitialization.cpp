/**
 * @file Test__GPURamPrefixInitialization.cpp
 * @brief Prove exact prefix archive initialization without redundant CPU
 * clears.
 *
 * A retained device producer fills the full checkpoint/terminal sections while
 * short attention copies leave zeroed capacity tails. Event publication is the
 * only host observation boundary. Twenty different archive identities exercise
 * retirement and PMA leases inside one persistent arena; poison bytes prove no
 * section depends on malloc returning clean memory. Delayed stream frontiers
 * additionally prove eviction is nonblocking and cannot recycle in-flight DMA.
 * This is a functional gate, never a timing threshold.
 */
#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/prefix_cache/RamPrefixStorageBackend.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <limits>

using namespace llaminar2;

namespace
{
    /** @brief Keep every archive section in the reuse and initialization matrix. */
    PrefixPayloadLayout archiveLayout(DeviceId device, size_t recurrent_bytes)
    {
        PrefixPayloadLayout layout;
        layout.device = device;
        layout.block_size = 64;
        layout.total_layers = 3;
        layout.fa_layers = 2;
        layout.gdn_layers = 1;
        layout.bytes_per_fa_layer_k = 128;
        layout.bytes_per_fa_layer_v = 256;
        layout.includes_mtp_state = true;
        layout.mtp_kv_bytes = 384;
        layout.includes_hybrid_state = true;
        layout.hybrid_state_bytes = layout.hybrid_device_state_bytes = recurrent_bytes;
        layout.includes_terminal_hidden = true;
        layout.terminal_hidden_bytes = 132;
        layout.includes_terminal_logits = true;
        layout.terminal_logits_bytes = 260;
        return layout;
    }

    /** @brief Admit exactly one fixed prefix backing allocation. */
    std::shared_ptr<PhysicalMemoryAuthority> hostAuthority(size_t bytes)
    {
        PhysicalMemoryPlanBuilder plan;
        plan.add(PhysicalMemoryResource{.world_rank = 0, .device = DeviceId::cpu(),
                     .total_bytes = bytes, .admission_available_bytes = bytes},
                 PhysicalMemoryOwner::PrefixHostTier, bytes);
        return std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    }

    /** @return Canonical live physical claim, independent of cache-index occupancy. */
    size_t claimed(const std::shared_ptr<PhysicalMemoryAuthority> &authority)
    {
        return authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::PrefixHostTier,
                                        PhysicalMemoryMaterializationKind::NewAllocation);
    }
    /** @brief Read one bounded counter without making it a runtime authority. */
    double allocationCounter(const char *name)
    {
        double result = 0;
        for (const auto &record : PerfStatsCollector::snapshot({"prefix_cache", "device_memory"}))
            if (record.name == name)
                result += record.value;
        return result;
    }

    /** @brief Check raw byte preservation for dense, compressed and partial
     * extents. */
    void verifyArchiveInitialization(DeviceId device)
    {
        ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("prefix_cache"))
            << "This functional path witness requires the registered PerfStats "
               "environment";
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        ASSERT_GT(backend->deviceCount(), device.ordinal);
        auto &gpu = GPUDeviceContextPool::instance().getContext(device);
        gpu.submitAndWait([&] {
            void *const stream = gpu.getOrCreateAuxiliaryStream("prefix_archive_initialization");
            ASSERT_NE(stream, nullptr);
            constexpr size_t capacity = 1024u * 1024u + 4u;
            auto source = std::shared_ptr<void>(
                backend->allocate(capacity, device.ordinal),
                [backend, device](void *pointer) { backend->free(pointer, device.ordinal); });
            ASSERT_NE(source, nullptr);
            auto capture = gpu.createGraphCapture(stream);
            ASSERT_NE(capture, nullptr);
            {
                ScopedBackendGraphCapture recording(gpu, *capture, "prefix_archive_producer");
                ASSERT_TRUE(recording.begin());
                ASSERT_TRUE(backend->memset(source.get(), 0x5a, capacity, device.ordinal, stream));
                recording.finish();
            }
            ASSERT_TRUE(capture->instantiate());

            const size_t arena_bytes = archiveLayout(device, capacity).totalBytes() + 64u;
            auto authority = hostAuthority(arena_bytes);
            PerfStatsCollector::reset();
            auto ram = RamPrefixStorageBackend::create(device, arena_bytes, authority);
            ASSERT_NE(ram, nullptr);
            EXPECT_EQ(claimed(authority), arena_bytes);

            for (int iteration = 0; iteration < 20; ++iteration)
            {
                SCOPED_TRACE(iteration);
                // Small odd tails and large checkpoint banks share the byte
                // contract; no float/quantized format can acquire another path.
                constexpr std::array<size_t, 4> extents{4u, 68u, 4100u, capacity};
                const auto layout = archiveLayout(device, extents[iteration % 4]);
                const double zeroed_before = allocationCounter("ram_payload_cpu_zeroed_bytes");
                const double overwritten_before = allocationCounter("ram_payload_full_overwrite_bytes");
                auto handle = ram->allocate(
                    makePrefixCacheKey(0xfeed, 0, iteration, iteration * 64, {iteration, iteration + 1}),
                    layout);
                ASSERT_TRUE(handle.valid());
                EXPECT_EQ(allocationCounter("ram_payload_cpu_zeroed_bytes") - zeroed_before,
                          static_cast<double>(layout.faKVBytes() + layout.mtpKVBytes() +
                                              layout.terminal_hidden_bytes + layout.terminal_logits_bytes));
                EXPECT_EQ(allocationCounter("ram_payload_full_overwrite_bytes") - overwritten_before,
                          static_cast<double>(layout.hybrid_state_bytes));
                EXPECT_FALSE(handle.payload_readiness->published());
                ASSERT_TRUE(ram->attachModelRuntimeState(&handle,
                    std::make_shared<std::vector<uint8_t>>(13u, static_cast<uint8_t>(iteration))));
                // A device-hot replica retains this independent metadata owner,
                // while its common payload already resides in VRAM.
                auto hot_runtime_alias = handle.model_runtime_state_storage;

                const std::array<std::pair<void *, size_t>, 3> complete{
                    {{handle.hybrid_payload, layout.hybrid_state_bytes},
                     {handle.terminal_hidden, layout.terminal_hidden_bytes},
                     {handle.terminal_logits, layout.terminal_logits_bytes}}};
                for (const auto &[pointer, bytes] : complete)
                    std::memset(pointer, 0xa5,
                                bytes); // fixture poison, not production work
                const std::array<std::pair<void *, size_t>, 2> padded{
                    {{handle.kv_payload, layout.faKVBytes()}, {handle.mtp_payload, layout.mtpKVBytes()}}};
                for (const auto &[pointer, bytes] : padded)
                    ASSERT_TRUE(std::all_of(static_cast<uint8_t *>(pointer),
                                            static_cast<uint8_t *>(pointer) + bytes,
                                            [](uint8_t value) { return value == 0; }));

                auto ready = std::shared_ptr<void>(
                    backend->createEvent(device.ordinal),
                    [backend, device](void *event) { backend->destroyEvent(event, device.ordinal); });
                ASSERT_TRUE(handle.payload_readiness->prepare(ready, device, stream));
                ASSERT_TRUE(capture->launch());
                bool copied = true;
                for (const auto &[pointer, bytes] : complete)
                    copied =
                        backend->deviceToHostOnStream(pointer, source.get(), bytes, device.ordinal, stream) &&
                        copied;
                for (const auto &[pointer, bytes] : padded)
                    copied = backend->deviceToHostOnStream(pointer, source.get(), bytes / 2, device.ordinal,
                                                           stream) &&
                             copied;
                // Always publish after accepted copies, even if a later enqueue
                // fails; no handle may free an in-flight destination.
                ASSERT_TRUE(backend->recordEvent(ready.get(), device.ordinal, stream));
                ASSERT_TRUE(handle.payload_readiness->publishRecorded());
                ASSERT_TRUE(handle.payload_readiness->waitOnHost());
                ASSERT_TRUE(copied);
                for (const auto &[pointer, bytes] : complete)
                    EXPECT_TRUE(std::all_of(static_cast<uint8_t *>(pointer),
                                            static_cast<uint8_t *>(pointer) + bytes,
                                            [](uint8_t value) { return value == 0x5a; }));
                for (const auto &[pointer, bytes] : padded)
                {
                    const auto *data = static_cast<const uint8_t *>(pointer);
                    EXPECT_TRUE(
                        std::all_of(data, data + bytes / 2, [](uint8_t value) { return value == 0x5a; }));
                    EXPECT_TRUE(std::all_of(data + bytes / 2, data + bytes,
                                            [](uint8_t value) { return value == 0; }));
                }
                auto alias = handle;
                ASSERT_TRUE(ram->release(handle));
                handle = {};
                EXPECT_EQ(claimed(authority), arena_bytes);
                if (iteration % 4 == 3) EXPECT_FALSE(ram->canStore(layout.totalBytes()));
                alias = {};
                EXPECT_EQ(claimed(authority), arena_bytes);
                EXPECT_TRUE(std::ranges::all_of(hot_runtime_alias->bytes(),
                    [&](uint8_t value) { return value == iteration; }));
                hot_runtime_alias.reset();
                EXPECT_EQ(ram->availableAllocationBytes(), arena_bytes);
                EXPECT_EQ(allocationCounter("ram_arena_materializations"), 1.0);
                EXPECT_EQ(allocationCounter("pinned_host_allocations"), 1.0);
                EXPECT_EQ(allocationCounter("pinned_host_allocation_bytes"), static_cast<double>(arena_bytes));
                EXPECT_EQ(allocationCounter("ram_arena_payload_leases"), iteration + 1.0);
            }
            auto retained = ram->allocate(makePrefixCacheKey(0xfeed, 0, 99, 0, {99}),
                                          archiveLayout(device, capacity));
            ASSERT_TRUE(retained.valid());
            ASSERT_TRUE(ram->release(retained));
            ram.reset();
            EXPECT_EQ(claimed(authority), arena_bytes);
            retained = {};
            EXPECT_EQ(claimed(authority), 0u);
            capture->reset();
        });
    }
} // namespace

#ifdef HAVE_CUDA
TEST(GPURamPrefixInitialization, CUDA)
{
    verifyArchiveInitialization(DeviceId::cuda(0));
}
#endif
#ifdef HAVE_ROCM
TEST(GPURamPrefixInitialization, ROCm)
{
    verifyArchiveInitialization(DeviceId::rocm(0));
}
#endif
