/**
 * @file Test__ROCmHardwareMetadata.cpp
 * @brief Device-free replay of physical-inventory publication and FA2 binding.
 *
 * Synthetic driver records exercise visibility beyond eight devices, concurrent
 * admission, immutable reset identity and persistent failure. No HIP runtime or
 * GPU library is linked; native preferred wave32 is distinct from compiled64.
 */
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "kernels/rocm/attention/ROCmFlashAttentionLaunchPolicy.h"
#include "kernels/qwen4exp/ImmutableLookupHardwareBinding.h"
#include "kernels/common/ColumnShardAssemblyKernels.h"
#include <gtest/gtest.h>
#include <atomic>
#include <barrier>
#include <cstdio>
#include <thread>

namespace llaminar2::test
{
    /**
     * @brief Build independently owned synthetic physical records without a runtime.
     * @param count Number of visible devices, including zero and counts above eight.
     * @return Complete immutable-admission inputs with unique identities.
     */
    std::vector<ROCmDeviceHardware> records(unsigned count)
    {
        std::vector<ROCmDeviceHardware> result;
        for (unsigned index = 0; index < count; ++index)
        {
            char pci[32]{};
            std::snprintf(pci, sizeof(pci), "%04x:%02x:00.0", index / 256, index % 256);
            ROCmDeviceHardware hardware{
                .ordinal = static_cast<int>(index), .name = "synthetic " + std::to_string(index),
                .isa_target = index % 2 ? "gfx1100" : "gfx906",
                .pci_address = pci,
                .pci_domain = static_cast<int>(index / 256),
                .pci_bus = static_cast<int>(index % 256),
                .compute_units = 20 + static_cast<int>(index),
                .native_wave_width = index % 2 ? 32 : 64,
                .maximum_threads_per_block = 1024,
                .wall_clock_rate_khz = 100000 + static_cast<int>(index),
                .lds_per_block = 65536, .total_memory = 1ull << 30,
            };
            for (unsigned byte = 0; byte < 4; ++byte)
                hardware.uuid[byte] = static_cast<std::uint8_t>((index + 1) >> (8 * byte));
            result.push_back(std::move(hardware));
        }
        return result;
    }
    /** @test Neither construction nor unadmitted lookup invokes a hardware source. */
    TEST(ROCmHardwareMetadata, CPUOnlyConstructionAndUnadmittedReadNeverEnumerate)
    {
        ROCmHardwareAuthority authority;
        EXPECT_THROW((void)authority.snapshot(), std::logic_error);
    }
    /** @test Publication supports empty visibility and arbitrary valid device counts. */
    TEST(ROCmHardwareMetadata, EmptyAndMoreThanEightVisibleDevicesRemainExact)
    {
        for (unsigned count : {0u, 1u, 9u, 128u})
        {
            ROCmHardwareAuthority authority;
            authority.initialize([&] { return records(count); });
            const auto &snapshot = authority.snapshot();
            EXPECT_EQ(snapshot.size(), count);
            EXPECT_THROW((void)snapshot.at(-1), std::out_of_range);
            EXPECT_THROW((void)snapshot.at(static_cast<int>(count)), std::out_of_range);
            for (unsigned index = 0; index < count; ++index)
                EXPECT_EQ(snapshot.at(index).name, "synthetic " + std::to_string(index));
        }
    }
    /** @test Simultaneous startup callers publish one complete immutable snapshot. */
    TEST(ROCmHardwareMetadata, ConcurrentAdmissionQueriesOnceAndPublishesStableIdentity)
    {
        ROCmHardwareAuthority authority;
        std::atomic<unsigned> queries{0};
        std::barrier start(24);
        std::array<const ROCmDeviceHardware *, 24> observed{};
        std::vector<std::thread> workers;
        for (unsigned worker = 0; worker < observed.size(); ++worker)
            workers.emplace_back([&, worker] {
                start.arrive_and_wait();
                authority.initialize([&] { ++queries; return records(24); });
                observed[worker] = &authority.snapshot().at(17);
            });
        for (auto &worker : workers) worker.join();
        EXPECT_EQ(queries, 1u);
        for (const auto *identity : observed) EXPECT_EQ(identity, observed[0]);
        // A context-reset generation is deliberately absent from this authority:
        // another startup boundary must retain its physical snapshot verbatim.
        authority.initialize([&]() -> std::vector<ROCmDeviceHardware> {
            ++queries; throw std::runtime_error("reset must not enumerate again");
        });
        EXPECT_EQ(queries, 1u);
        EXPECT_EQ(&authority.snapshot().at(17), observed[0]);
    }
    /** @test Partial/broken sources remain rejected and cannot retry into an empty inventory. */
    TEST(ROCmHardwareMetadata, FailedAdmissionIsPermanentAndNeverPublishesPartialDevices)
    {
        ROCmHardwareAuthority authority;
        unsigned queries = 0;
        const auto fail = [&]() -> std::vector<ROCmDeviceHardware> {
            ++queries; throw std::runtime_error("driver property query failed at ordinal 9");
        };
        EXPECT_THROW(authority.initialize(fail), std::runtime_error);
        EXPECT_THROW((void)authority.snapshot(), std::logic_error);
        EXPECT_THROW(authority.initialize([&] { ++queries; return records(0); }), std::runtime_error);
        EXPECT_EQ(queries, 1u);
        auto invalid = records(12); invalid[9].ordinal = 0;
        EXPECT_THROW(ROCmHardwareInventory(std::move(invalid)), std::invalid_argument);
    }
    /** @test Visible ordinals cannot alias absent or another device's physical identity. */
    TEST(ROCmHardwareMetadata, AbsentAndDuplicatePhysicalIdentitiesFailAdmission)
    {
        auto absent = records(12); absent[9].uuid.fill(0);
        EXPECT_THROW(ROCmHardwareInventory(std::move(absent)), std::invalid_argument);
        auto duplicate_uuid = records(12); duplicate_uuid[9].uuid = duplicate_uuid[3].uuid;
        EXPECT_THROW(ROCmHardwareInventory(std::move(duplicate_uuid)), std::invalid_argument);
        auto duplicate_pci = records(12); duplicate_pci[10].pci_address = duplicate_pci[1].pci_address;
        EXPECT_THROW(ROCmHardwareInventory(std::move(duplicate_pci)), std::invalid_argument);
        auto absent_pci = records(12); absent_pci[10].pci_address.clear();
        EXPECT_THROW(ROCmHardwareInventory(std::move(absent_pci)), std::invalid_argument);
    }
    /** @test Physical planning admits new ISA identities without a card-name table. */
    TEST(ROCmHardwareMetadata, GeometryBindingDependsOnLimitsInsteadOfCardFamily)
    {
        namespace policy = rocm::fa2_policy;
        auto devices = records(3);
        for (auto &device : devices)
        {
            device.compute_units = 120;
            device.lds_per_block = 65536;
        }
        // This synthetic identity proves only metadata and launch geometry.
        // It does not advertise native instructions or an executable image.
        devices[2].name = "future architecture fixture";
        devices[2].isa_target = "synthetic-future-isa";
        devices[2].native_wave_width = 32;
        const ROCmHardwareInventory inventory(std::move(devices));
        for (int ordinal = 0; ordinal < 3; ++ordinal)
        {
            const auto facts = policy::bindROCmFA2Hardware(inventory.at(ordinal));
            const auto plan = policy::selectROCmFA2PrefillParallelPlan({
                .batch_size = 1, .query_rows = 128, .local_query_heads = 28,
                .head_dim = 128, .kv_capacity = 131072,
                .compute_unit_count = facts.compute_unit_count,
                .lds_capacity_bytes = facts.lds_capacity_bytes,
                .requested_axis = attention::AttentionPrefillParallelAxis::GeometrySelected,
            });
            EXPECT_EQ(facts.compute_unit_count, 120);
            EXPECT_EQ(facts.lds_capacity_bytes, 65536u);
            EXPECT_TRUE(plan.valid);
            EXPECT_TRUE(plan.usesContextParallelism());
            EXPECT_EQ(policy::kROCmFA2WavefrontSize, 64);
        }
    }
    /** @test Attention's shared hardware binding has no legacy ordinal-64 cache limit. */
    TEST(ROCmHardwareMetadata, AttentionBindingKeepsHighOrdinalsExact)
    {
        const ROCmHardwareInventory inventory(records(128));
        const auto bind = [&](int ordinal) {
            return rocm::fa2_policy::bindROCmFA2Hardware(inventory.at(ordinal));
        };
        for (const int ordinal : {0, 63, 64, 96, 127})
        {
            SCOPED_TRACE(ordinal);
            EXPECT_EQ(bind(ordinal).compute_unit_count, 20 + ordinal);
            EXPECT_EQ(bind(ordinal).lds_capacity_bytes, 65536u);
        }
        EXPECT_THROW(bind(-1), std::out_of_range);
        EXPECT_THROW(bind(128), std::out_of_range);
    }
    /** @test Planning and native launch selection share actual per-workgroup LDS limits. */
    TEST(ROCmHardwareMetadata, AttentionTileBindingMatchesPlannerAcrossLDSCapacities)
    {
        namespace policy = rocm::fa2_policy;
        auto hardware = records(1).front();
        hardware.compute_units = 120;
        unsigned old_constant_disagreements = 0;
        for (const std::size_t kib : {32u, 48u, 64u, 96u})
        {
            SCOPED_TRACE(kib);
            hardware.lds_per_block = kib * 1024;
            const auto launch = policy::selectROCmFA2TileForHardware(hardware, 96);
            const auto plan = policy::selectROCmFA2PrefillParallelPlan({
                .batch_size = 1, .query_rows = 128, .local_query_heads = 28,
                .head_dim = 96, .kv_capacity = 131072,
                .compute_unit_count = hardware.compute_units,
                .lds_capacity_bytes = hardware.lds_per_block,
                .requested_axis = attention::AttentionPrefillParallelAxis::QuerySequence,
            });
            ASSERT_TRUE(launch.valid);
            ASSERT_TRUE(plan.valid);
            EXPECT_EQ(launch.query_rows, plan.tile.query_rows);
            EXPECT_EQ(launch.kv_rows, plan.tile.kv_rows);
            EXPECT_EQ(launch.lds_bytes, plan.tile.lds_bytes);
            EXPECT_LE(launch.lds_bytes, hardware.lds_per_block);
            // Replay the former launcher's assumption as a negative control:
            // both a smaller and a larger legal envelope select another tile.
            const auto old_launch = policy::selectROCmFA2Tile(96, 65536);
            if (old_launch.query_rows != launch.query_rows || old_launch.kv_rows != launch.kv_rows)
                ++old_constant_disagreements;
            if (kib == 64)
            {
                EXPECT_EQ(old_launch.query_rows, launch.query_rows);
                EXPECT_EQ(old_launch.kv_rows, launch.kv_rows);
                EXPECT_EQ(old_launch.lds_bytes, launch.lds_bytes);
            }
        }
        EXPECT_EQ(old_constant_disagreements, 2u);
    }
    /** @test Failed native current-device observations cannot become ordinal-zero launches. */
    TEST(ROCmHardwareMetadata, CurrentDeviceQueryFailureCannotLookupOrLaunch)
    {
        const ROCmHardwareInventory inventory(records(24));
        for (const bool writes_zero : {false, true})
        {
            unsigned queries = 0, lookups = 0, launches = 0;
            const auto submit = [&] {
                const auto &hardware = requireROCmCurrentHardware(
                    [&](int *ordinal) {
                        ++queries;
                        if (writes_zero) *ordinal = 0;
                        return 702;
                    },
                    [&](int ordinal) -> const ROCmDeviceHardware & {
                        ++lookups; return inventory.at(ordinal);
                    });
                (void)hardware;
                ++launches;
            };
            try { submit(); FAIL() << "Failed current-device query reached launch"; }
            catch (const std::runtime_error &error) {
                EXPECT_STREQ(error.what(),
                    "hipGetDevice failed before ROCm decode hardware lookup: status=702");
            }
            EXPECT_EQ(queries, 1u); EXPECT_EQ(lookups, 0u); EXPECT_EQ(launches, 0u);
        }
        unsigned lookups = 0;
        const auto &hardware = requireROCmCurrentHardware(
            [](int *ordinal) { *ordinal = 17; return 0; },
            [&](int ordinal) -> const ROCmDeviceHardware & {
                ++lookups; return inventory.at(ordinal);
            });
        EXPECT_EQ(&hardware, &inventory.at(17));
        EXPECT_EQ(lookups, 1u);
    }
    /** @test Physical native-wave preference never alters the installed arithmetic contract. */
    TEST(ROCmHardwareMetadata, FA2BindsExactCULDSLimitsAndCompiledWavePolicy)
    {
        namespace policy = rocm::fa2_policy;
        const ROCmHardwareInventory inventory(records(12));
        const auto &rdna = inventory.at(9);
        ASSERT_EQ(rdna.native_wave_width, 32);
        ASSERT_EQ(rocmOrderedWaveWidth, 64u);
        EXPECT_EQ(policy::kROCmFA2WavefrontSize, 64);
        const auto bound = policy::bindROCmFA2Hardware(rdna);
        EXPECT_EQ(bound.compute_unit_count, rdna.compute_units);
        EXPECT_EQ(bound.lds_capacity_bytes, rdna.lds_per_block);
        auto insufficient = rdna; insufficient.maximum_threads_per_block = 128;
        EXPECT_THROW((void)policy::bindROCmFA2Hardware(insufficient), std::invalid_argument);
        // The actual prefill policy consumes the bound LDS capacity; small-LDS
        // hardware cannot accidentally select the fixed 256-head 16-KiB tile.
        EXPECT_TRUE(policy::selectROCmFA2Tile(256, bound.lds_capacity_bytes).valid);
        EXPECT_FALSE(policy::selectROCmFA2Tile(256, 8192).valid);
        auto devices = records(2);
        devices[0].compute_units = 60;
        devices[1].compute_units = 120;
        const ROCmHardwareInventory heterogeneous(std::move(devices));
        const auto choose = [&](int ordinal) {
            const auto facts = policy::bindROCmFA2Hardware(heterogeneous.at(ordinal));
            return policy::selectROCmFA2PrefillParallelPlan({
                .batch_size = 1, .query_rows = 128, .local_query_heads = 28,
                .head_dim = 128, .kv_capacity = 131072,
                .compute_unit_count = facts.compute_unit_count,
                .lds_capacity_bytes = facts.lds_capacity_bytes,
                .requested_axis = attention::AttentionPrefillParallelAxis::GeometrySelected,
            });
        };
        const auto small = choose(0), large = choose(1);
        ASSERT_TRUE(small.valid); ASSERT_TRUE(large.valid);
        EXPECT_FALSE(small.usesContextParallelism());
        EXPECT_TRUE(large.usesContextParallelism());

    }
    /** @test Service timing binds the exact admitted wall clock without rediscovery or a core-clock guess. */
    TEST(ROCmHardwareMetadata, ServiceWallClockBindingIsExactStableAndRequired)
    {
        ROCmHardwareAuthority authority;
        unsigned discoveries = 0;
        authority.initialize([&] { ++discoveries; return records(129); });
        for (const int ordinal : {128, 8, 0, 64, 128})
        {
            const auto &hardware = authority.snapshot().at(ordinal);
            const auto *identity = &hardware;
            for (unsigned replay = 0; replay < 16; ++replay)
            {
                authority.initialize([&] { ++discoveries; return records(0); });
                EXPECT_EQ(&authority.snapshot().at(ordinal), identity);
                EXPECT_EQ(rocmServiceWallClockRateKHz(hardware), 100000u + ordinal);
                EXPECT_EQ(qwen4exp::rocmImmutableLookupTimeoutTicks(hardware),
                    static_cast<std::uint64_t>(100000u + ordinal) * 30'000u);
            }
        }
        EXPECT_EQ(discoveries, 1u);
        for (const int invalid : {0, -1})
        {
            auto incomplete = records(1);
            incomplete[0].wall_clock_rate_khz = invalid;
            EXPECT_THROW((void)rocmServiceWallClockRateKHz(incomplete[0]), std::invalid_argument);
            EXPECT_THROW((void)qwen4exp::rocmImmutableLookupTimeoutTicks(incomplete[0]), std::invalid_argument);
            EXPECT_THROW((void)ROCmHardwareInventory(incomplete), std::invalid_argument);
        }
    }

    /** @test The production resident-grid binding keeps physical identity separate from loaded-function occupancy. */
    TEST(ROCmHardwareMetadata, ColumnAssemblyUsesExactPhysicalCountAndFreshFunctionOccupancy)
    {
        const ROCmHardwareInventory inventory(records(129));
        for (const int ordinal : {0, 8, 64, 128})
        {
            const auto &hardware = inventory.at(ordinal);
            const int processors = hardware.compute_units;
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

    /** @test Successful unsupported is a real fact; failed/incomplete discovery never becomes false support. */
    TEST(ROCmHardwareMetadata, StreamWaitValueCapabilityPublishesOnceAndRejectsFailedObservation)
    {
        for (const int supported : {0, 1})
        {
            ROCmHardwareAuthority authority;
            unsigned queries = 0;
            authority.initialize([&] {
                auto observed = records(1);
                observed[0].can_use_stream_wait_value = observeROCmStreamWaitValueCapability(0,
                    [&](int *value, int ordinal) { EXPECT_EQ(ordinal, 0); ++queries; *value = supported; return 0; });
                return observed;
            });
            const auto *identity = &authority.snapshot().at(0);
            for (unsigned replay = 0; replay < 32; ++replay)
            {
                authority.initialize([&] { ++queries; return records(0); });
                EXPECT_EQ(&authority.snapshot().at(0), identity);
                EXPECT_EQ(authority.snapshot().at(0).can_use_stream_wait_value, supported != 0);
            }
            EXPECT_EQ(queries, 1u);
        }
        ROCmHardwareAuthority failed;
        unsigned queries = 0;
        const auto source = [&] {
            auto observed = records(1);
            observed[0].can_use_stream_wait_value = observeROCmStreamWaitValueCapability(0,
                [&](int *value, int) { ++queries; *value = 0; return 7; });
            return observed;
        };
        EXPECT_THROW(failed.initialize(source), std::runtime_error);
        EXPECT_THROW(failed.initialize(source), std::runtime_error);
        EXPECT_EQ(queries, 1u);
        EXPECT_THROW((void)failed.snapshot(), std::logic_error);
        for (const int incomplete : {-1, 2})
            EXPECT_THROW((void)observeROCmStreamWaitValueCapability(128,
                [&](int *value, int) { *value = incomplete; return 0; }), std::runtime_error);
    }

}
