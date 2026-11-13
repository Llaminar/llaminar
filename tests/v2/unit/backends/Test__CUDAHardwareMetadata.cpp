/**
 * @file Test__CUDAHardwareMetadata.cpp
 * @brief Device-free replay of CUDA physical publication and actual attention binding.
 *
 * Recorded complete inventories exercise once-only publication, failed native
 * observations, arbitrary visible counts and UUID identity independently of
 * loaded program generations. The production FA2 binding and decode envelope
 * consume the records directly; this binary links neither CUDA nor a backend.
 */
#include "backends/cuda/CUDAHardwareMetadata.h"
#include "kernels/cuda/attention/CUDAFlashAttentionHardware.h"
#include "kernels/cuda/moe/CUDARouterQ8Prefill.h"
#include "kernels/common/ColumnShardAssemblyKernels.h"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <barrier>
#include <cstdio>
#include <exception>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /**
         * @brief Author complete independent records without a driver or card-name table.
         * @param count Exact visible count, including empty and more than eight devices.
         * @return Ordered unique UUIDs with intentionally differing launch-relevant limits.
         */
        std::vector<CUDADeviceHardware> cudaRecords(unsigned count)
        {
            std::vector<CUDADeviceHardware> result;
            for (unsigned index = 0; index < count; ++index)
            {
                char pci[32]{};
                std::snprintf(pci, sizeof(pci), "%04x:%02x:00.0", index / 256, index % 256);
                CUDADeviceHardware record{
                    .ordinal = static_cast<int>(index),
                    .name = "recorded CUDA device " + std::to_string(index),
                    .pci_address = pci,
                    .pci_domain = static_cast<int>(index / 256),
                    .pci_bus = static_cast<int>(index % 256), .pci_device = 0,
                    .sm_major = 8, .sm_minor = static_cast<int>(index % 3),
                    .sm_count = 20 + static_cast<int>(index), .native_warp_width = 32,
                    .maximum_threads_per_block = 1024, .maximum_threads_per_sm = 2048,
                    .maximum_blocks_per_sm = 32, .registers_per_block = 65536,
                    .registers_per_sm = 65536, .shared_memory_per_block = 49152,
                    .shared_memory_optin_per_block = 65536 + index * 256u,
                    .shared_memory_per_sm = 262144, .total_memory = 1ull << 30,
                    .last_level_cache = 1u << 20};
                for (unsigned byte = 0; byte < 4; ++byte)
                    record.uuid[byte] = static_cast<std::uint8_t>((index + 1) >> (8 * byte));
                result.push_back(std::move(record));
            }
            return result;
        }
    }

    /** @test CPU-only construction/read never enters discovery or silently admits emptiness. */
    TEST(CUDAHardwareMetadata, ExplicitAdmissionRequiredAndEmptyVisibilityIsValid)
    {
        CUDAHardwareAuthority authority;
        EXPECT_THROW((void)authority.snapshot(), std::logic_error);
        unsigned calls = 0;
        authority.initialize([&] { ++calls; return cudaRecords(0); });
        EXPECT_EQ(calls, 1u);
        EXPECT_EQ(authority.snapshot().size(), 0u);
        EXPECT_THROW((void)authority.snapshot().at(0), std::out_of_range);
    }

    /** @test Every ordinal remains exact beyond former masked/clamped eight-device tables. */
    TEST(CUDAHardwareMetadata, ArbitraryVisibilityKeepsHighOrdinalsAndActualBindingsExact)
    {
        namespace policy = cuda::fa2_policy;
        for (unsigned count : {1u, 9u, 128u})
        {
            CUDAHardwareAuthority authority;
            authority.initialize([&] { return cudaRecords(count); });
            const auto &inventory = authority.snapshot();
            EXPECT_EQ(inventory.size(), count);
            EXPECT_THROW((void)inventory.at(-1), std::out_of_range);
            EXPECT_THROW((void)inventory.at(static_cast<int>(count)), std::out_of_range);
            for (int ordinal : {0, 7, 8, 63, 64, 127})
            {
                if (ordinal >= static_cast<int>(count)) continue;
                SCOPED_TRACE(ordinal);
                const auto &record = inventory.at(ordinal);
                const auto bound = policy::bindCUDAFA2Hardware(record);
                EXPECT_EQ(record.ordinal, ordinal);
                EXPECT_EQ(bound.sm_major, 8);
                EXPECT_EQ(bound.sm_minor, ordinal % 3);
                EXPECT_EQ(bound.sm_count, 20 + ordinal);
                EXPECT_EQ(bound.max_smem_optin, 65536u + static_cast<unsigned>(ordinal) * 256u);
                if (ordinal >= 8)
                {
                    // Original-code negative controls authenticate both lost
                    // identity mechanisms, using the real binding on wrong records.
                    const auto masked = policy::bindCUDAFA2Hardware(inventory.at(ordinal & 7));
                    const auto clamped = policy::bindCUDAFA2Hardware(inventory.at(0));
                    EXPECT_NE(bound.sm_count, masked.sm_count);
                    EXPECT_NE(bound.max_smem_optin, clamped.max_smem_optin);
                }
            }
        }
    }

    /** @test The actual decode envelope consumes exact SM limits without a family whitelist. */
    TEST(CUDAHardwareMetadata, AttentionPolicyUsesExactLimitsForUnfamiliarSMIdentity)
    {
        namespace policy = cuda::fa2_policy;
        auto records = cudaRecords(128);
        // A new physical tuple establishes no executable-image or MMA claim.
        // It must nevertheless retain its actual limits in the installed policy.
        records[127].sm_major = 99;
        records[127].sm_minor = 7;
        const CUDAHardwareInventory inventory(std::move(records));
        const auto future = policy::bindCUDAFA2Hardware(inventory.at(127));
        EXPECT_EQ(future.sm_major, 99); EXPECT_EQ(future.sm_minor, 7);
        EXPECT_EQ(future.sm_count, 147);
        EXPECT_EQ(future.max_smem_optin, 98048u);

        // Recorded decisions retain the current arithmetic partition contract:
        // four resident blocks/SM, floor division, KV work and 32-split bounds.
        // These are replay observations, not a second selector implementation.
        struct Observation { int ordinal, capacity, heads, splits; };
        const Observation observations[]{
                 {0, 16384, 28, 2}, {7, 16384, 28, 3},
                 {8, 16384, 28, 4}, {63, 16384, 28, 11},
                 {64, 16384, 28, 12}, {127, 16384, 28, 21},
                 {127, 16, 28, 1}, {127, 63, 28, 3},
                 {127, 64, 28, 4}, {127, 4096, 1, 32}, {127, 1, 28, 1}};
        for (const auto observation : observations)
        {
            SCOPED_TRACE(observation.ordinal);
            SCOPED_TRACE(observation.capacity);
            EXPECT_EQ(policy::selectCUDAFlashDecodeSplitEnvelope(
                inventory.at(observation.ordinal), observation.capacity, observation.heads),
                observation.splits);
        }
        const auto exact = policy::selectCUDAFlashDecodeSplitEnvelope(inventory.at(64), 16384, 28);
        const auto former_masked = policy::selectCUDAFlashDecodeSplitEnvelope(inventory.at(64 & 7), 16384, 28);
        EXPECT_NE(exact, former_masked);
        // Changing opt-in shared memory is visible to binding but cannot alter
        // this decode reduction partition, which is an independent contract.
        auto changed = inventory.at(127);
        changed.shared_memory_optin_per_block = 131072;
        EXPECT_EQ(policy::bindCUDAFA2Hardware(changed).max_smem_optin, 131072u);
        EXPECT_EQ(policy::selectCUDAFlashDecodeSplitEnvelope(changed, 16384, 28), 21);
        changed.sm_count = std::numeric_limits<int>::max();
        EXPECT_EQ(policy::selectCUDAFlashDecodeSplitEnvelope(changed, 16384, 1), 32);
        EXPECT_THROW((void)policy::selectCUDAFlashDecodeSplitEnvelope(changed, 0, 28), std::invalid_argument);
        EXPECT_THROW((void)policy::selectCUDAFlashDecodeSplitEnvelope(changed, 4096, 0), std::invalid_argument);
        EXPECT_THROW((void)policy::selectCUDAFlashDecodeSplitEnvelope(changed, -1, 28), std::invalid_argument);
    }

    /** @test Physical records cannot silently change the installed kernel's warp/thread contract. */
    TEST(CUDAHardwareMetadata, ActualAttentionBindingRejectsIncompatibleOrMissingLimits)
    {
        namespace policy = cuda::fa2_policy;
        const auto source = cudaRecords(1).front();
        auto invalid = source; invalid.native_warp_width = 64;
        EXPECT_THROW((void)policy::bindCUDAFA2Hardware(invalid), std::invalid_argument);
        invalid = source; invalid.maximum_threads_per_block = 128;
        EXPECT_THROW((void)policy::bindCUDAFA2Hardware(invalid), std::invalid_argument);
        invalid = source; invalid.sm_count = 0;
        EXPECT_THROW((void)policy::bindCUDAFA2Hardware(invalid), std::invalid_argument);
        invalid = source; invalid.sm_major = 0;
        EXPECT_THROW((void)policy::bindCUDAFA2Hardware(invalid), std::invalid_argument);
        invalid = source; invalid.shared_memory_optin_per_block = 0;
        EXPECT_THROW((void)policy::bindCUDAFA2Hardware(invalid), std::invalid_argument);
        invalid = source; invalid.shared_memory_optin_per_block =
            static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
        EXPECT_THROW((void)policy::bindCUDAFA2Hardware(invalid), std::invalid_argument);
    }

    /** @test Concurrent startup publishes once and hardware identity survives resource resets. */
    TEST(CUDAHardwareMetadata, ConcurrentAdmissionPublishesOneStableInventory)
    {
        CUDAHardwareAuthority authority;
        std::atomic<unsigned> calls{0};
        std::barrier start(24);
        std::array<const CUDADeviceHardware *, 24> observed{};
        std::array<std::exception_ptr, 24> failures{};
        std::vector<std::thread> workers;
        for (unsigned worker = 0; worker < observed.size(); ++worker)
            workers.emplace_back([&, worker] {
                start.arrive_and_wait();
                try
                {
                    authority.initialize([&] { ++calls; return cudaRecords(128); });
                    observed[worker] = &authority.snapshot().at(127);
                }
                catch (...) { failures[worker] = std::current_exception(); }
            });
        for (auto &worker : workers) worker.join();
        EXPECT_EQ(calls, 1u);
        for (unsigned worker = 0; worker < observed.size(); ++worker)
        {
            EXPECT_FALSE(failures[worker]);
            ASSERT_NE(observed[worker], nullptr);
            EXPECT_EQ(observed[worker], observed[0]);
        }
        // No program/runtime generation belongs to immutable physical facts.
        // Reentering startup after a native reset must not replace this record.
        authority.initialize([&]() -> std::vector<CUDADeviceHardware> {
            ++calls; throw std::runtime_error("reset must not rediscover physical hardware");
        });
        EXPECT_EQ(calls, 1u);
        EXPECT_EQ(&authority.snapshot().at(127), observed[0]);
    }

    /** @test A partially discovered inventory cannot become visible or retry into a fabricated SM8.2. */
    TEST(CUDAHardwareMetadata, DiscoveryFailureIsStickyAndCannotInventArchitecture)
    {
        CUDAHardwareAuthority authority;
        unsigned calls = 0, bindings = 0;
        for (int repeat = 0; repeat < 3; ++repeat)
        {
            try
            {
                authority.initialize([&] {
                    ++calls;
                    if (repeat == 0)
                    {
                        auto partial = cudaRecords(10);
                        (void)partial;
                        throw std::runtime_error("cudaGetDeviceProperties failed at ordinal 9: status=999");
                    }
                    auto invented = cudaRecords(1);
                    invented[0].sm_major = 8; invented[0].sm_minor = 2;
                    return invented;
                });
                (void)cuda::fa2_policy::bindCUDAFA2Hardware(authority.snapshot().at(0));
                ++bindings;
                FAIL() << "Failed source unexpectedly published fallback architecture";
            }
            catch (const std::runtime_error &error)
            {
                EXPECT_STREQ(error.what(), "cudaGetDeviceProperties failed at ordinal 9: status=999");
            }
            EXPECT_THROW((void)authority.snapshot(), std::logic_error);
        }
        EXPECT_EQ(calls, 1u);
        EXPECT_EQ(bindings, 0u);
    }

    /** @test Validation failures, including partial records, are sticky admission failures. */
    TEST(CUDAHardwareMetadata, PartialInventoryValidationCannotPublishOrRetry)
    {
        CUDAHardwareAuthority authority;
        unsigned calls = 0;
        EXPECT_THROW(authority.initialize([&] {
            ++calls; auto records = cudaRecords(128); records[64].sm_count = 0; return records;
        }), std::invalid_argument);
        EXPECT_THROW((void)authority.snapshot(), std::logic_error);
        EXPECT_THROW(authority.initialize([&] { ++calls; return cudaRecords(0); }), std::invalid_argument);
        EXPECT_EQ(calls, 1u);
        auto gap = cudaRecords(128); gap[64].ordinal = 0;
        EXPECT_THROW(CUDAHardwareInventory(std::move(gap)), std::invalid_argument);
        auto missing_architecture = cudaRecords(1); missing_architecture[0].sm_major = 0;
        EXPECT_THROW(CUDAHardwareInventory(std::move(missing_architecture)), std::invalid_argument);
    }

    /** @test Concurrent failing callers all receive the original failure without partial publication. */
    TEST(CUDAHardwareMetadata, ConcurrentDiscoveryFailureIsObservedOnceAndRetained)
    {
        CUDAHardwareAuthority authority;
        std::atomic<unsigned> calls{0}, rejected{0};
        std::barrier start(16);
        std::vector<std::thread> workers;
        for (unsigned worker = 0; worker < 16; ++worker)
            workers.emplace_back([&] {
                start.arrive_and_wait();
                try
                {
                    authority.initialize([&]() -> std::vector<CUDADeviceHardware> {
                        ++calls; throw std::runtime_error("recorded CUDA discovery failure");
                    });
                }
                catch (const std::runtime_error &error)
                {
                    if (std::string(error.what()) == "recorded CUDA discovery failure") ++rejected;
                }
            });
        for (auto &worker : workers) worker.join();
        EXPECT_EQ(calls, 1u); EXPECT_EQ(rejected, 16u);
        EXPECT_THROW((void)authority.snapshot(), std::logic_error);
    }

    /** @test Distinct allocation UUIDs may share a PCI parent; missing/duplicate UUIDs cannot alias. */
    TEST(CUDAHardwareMetadata, UUIDIdentityAllowsSharedPCIWithoutAllocationAliasing)
    {
        auto partitioned = cudaRecords(2);
        partitioned[1].pci_address = partitioned[0].pci_address;
        partitioned[1].pci_domain = partitioned[0].pci_domain;
        partitioned[1].pci_bus = partitioned[0].pci_bus;
        partitioned[1].pci_device = partitioned[0].pci_device;
        const CUDAHardwareInventory inventory(partitioned);
        EXPECT_EQ(inventory.at(0).pci_address, inventory.at(1).pci_address);
        EXPECT_NE(inventory.at(0).uuid, inventory.at(1).uuid);
        partitioned[1].uuid = partitioned[0].uuid;
        EXPECT_THROW(CUDAHardwareInventory(std::move(partitioned)), std::invalid_argument);
        auto absent = cudaRecords(2); absent[1].uuid.fill(0);
        EXPECT_THROW(CUDAHardwareInventory(std::move(absent)), std::invalid_argument);
        auto missing_pci = cudaRecords(2); missing_pci[1].pci_address.clear();
        EXPECT_THROW(CUDAHardwareInventory(std::move(missing_pci)), std::invalid_argument);
        auto malformed_pci = cudaRecords(2); malformed_pci[1].pci_address = "ordinal:1";
        EXPECT_THROW(CUDAHardwareInventory(std::move(malformed_pci)), std::invalid_argument);
    }

    /** @test Actual router setup binding retains exact high ordinals and rejects incompatible streams/capture. */
    TEST(CUDAHardwareMetadata, RouterBindingUsesAdmittedLimitsAndRejectsIncompleteCaptureIdentity)
    {
        const CUDAHardwareInventory inventory(cudaRecords(129));
        for (const int ordinal : {0, 8, 64, 128})
        {
            const auto &hardware = inventory.at(ordinal);
            cuda::RouterQ8PrefillHardware binding{7, 77, 7};
            ASSERT_TRUE(cuda::bindRouterQ8PrefillHardware(ordinal, ordinal, false, hardware, binding));
            EXPECT_EQ(binding.device, ordinal);
            EXPECT_EQ(binding.multiprocessors, hardware.sm_count);
            EXPECT_EQ(binding.block_slots_per_sm, hardware.maximum_blocks_per_sm);
            const auto stable = binding;
            for (int repeat = 0; repeat < 8; ++repeat)
            {
                EXPECT_FALSE(cuda::bindRouterQ8PrefillHardware(ordinal, ordinal, true, hardware, binding));
                EXPECT_FALSE(cuda::bindRouterQ8PrefillHardware(ordinal, ordinal + 1, false, hardware, binding));
                EXPECT_FALSE(cuda::bindRouterQ8PrefillHardware(ordinal + 1, ordinal + 1, false, hardware, binding));
                EXPECT_EQ(binding.device, stable.device);
                EXPECT_EQ(binding.multiprocessors, stable.multiprocessors);
                EXPECT_EQ(binding.block_slots_per_sm, stable.block_slots_per_sm);
            }
            for (const int bad_limit : {0, -1, 1025})
            {
                auto incomplete = hardware;
                incomplete.maximum_blocks_per_sm = bad_limit;
                EXPECT_FALSE(cuda::bindRouterQ8PrefillHardware(ordinal, ordinal, false, incomplete, binding));
            }
            auto incomplete = hardware;
            incomplete.sm_count = 0;
            EXPECT_FALSE(cuda::bindRouterQ8PrefillHardware(ordinal, ordinal, false, incomplete, binding));
            EXPECT_EQ(binding.multiprocessors, stable.multiprocessors);
        }
    }

    /** @test The production resident-grid binding keeps physical identity separate from loaded-function occupancy. */
    TEST(CUDAHardwareMetadata, ColumnAssemblyUsesExactPhysicalCountAndFreshFunctionOccupancy)
    {
        const CUDAHardwareInventory inventory(cudaRecords(129));
        for (const int ordinal : {0, 8, 64, 128})
        {
            const auto &hardware = inventory.at(ordinal);
            const int processors = hardware.sm_count;
            EXPECT_EQ(residentColumnAssemblyBlockCount(processors, 8, 0, 256), 0);
            EXPECT_EQ(residentColumnAssemblyBlockCount(processors, 8, 1, 256), 1);
            EXPECT_EQ(residentColumnAssemblyBlockCount(processors, 8, 257, 256), 2);
            // A changed loaded function/generation can have different occupancy
            // while the exact physical record remains unchanged. No resource cache.
            for (const int occupancy : {1, 8, 2, 8})
                EXPECT_EQ(residentColumnAssemblyBlockCount(processors, occupancy, 1ll << 30, 256),
                    processors * occupancy);
            EXPECT_EQ(&inventory.at(ordinal), &hardware);
        }
        EXPECT_THROW((void)residentColumnAssemblyBlockCount(0, 8, 1024, 256), std::invalid_argument);
        EXPECT_THROW((void)residentColumnAssemblyBlockCount(80, 0, 1024, 256), std::invalid_argument);
    }

}
