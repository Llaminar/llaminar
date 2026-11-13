/**
 * @file Test__ROCmAttentionFixtureAdmission.cpp
 * @brief Device-free replay of the attention source fixture's actual PMA admission.
 *
 * An explicitly authored tiny profile/inventory prices graph state without
 * loading a GGUF or entering a driver. Real CPUStorageRoot claims use the same
 * production rank-input/planner/certificate helper as the real-model fixture;
 * missing authority and artificial terminal ownership remain hard errors.
 */
#include "integration/kernels/rocm/ROCmFlashAttentionFixtureAdmission.h"
#include "memory/CPUStorageRoot.h"
#include <gtest/gtest.h>

namespace llaminar2::test
{
    /** @test Semantic aliases cannot underprice distinct temporal producer descriptors. */
    TEST(ROCmAttentionFixtureAdmission, SelectedStageDescriptorsOwnSnapshotArenaCapacity)
    {
        ModelMemoryProfile profile;
        profile.d_model = 896;
        profile.n_heads = 14;
        profile.n_kv_heads = 2;
        profile.head_dim = 64;
        const auto wide = rocm_attention_fixture::snapshotLayouts(profile, 512);
        const auto scalar = rocm_attention_fixture::snapshotLayouts(profile, 1);
        ASSERT_EQ(wide.size(), 5u); // Actual projection Q/K/V followed by RoPE Q/K.
        ASSERT_EQ(scalar.size(), wide.size());
        const auto extent = [](const auto &layouts, std::size_t first) {
            std::size_t end = 0;
            for (auto i = first; i < layouts.size(); ++i)
                end = placeSnapshotStorageSlot(end, layouts[i]).end_bytes;
            return end;
        };
        const auto capacity = rocm_attention_fixture::snapshotCapacity(profile, 512);
        ASSERT_TRUE(capacity.main_phase_payload.has_value());
        const auto actual = extent(wide, 0);
        EXPECT_EQ(actual, 4456448u); // Original native failure's immutable descriptor extent.
        EXPECT_EQ(capacity.main_phase_payload->wide_main_bytes, actual);
        EXPECT_EQ(capacity.main_phase_payload->scalar_main_bytes, extent(scalar, 0));

        // Reproduce the old semantic-three-value declaration using the actual
        // V_PROJECTION/Q_ROPE/K_ROPE descriptors, not another byte formula.
        const auto former_budget = extent(wide, 2);
        ASSERT_LT(former_budget, actual);
        const auto replay = [&](std::size_t budget, bool reject) {
            PhysicalMemoryPlanBuilder builder;
            builder.add(PhysicalMemoryResource{
                .world_rank = 0, .device = DeviceId::rocm(0),
                .total_bytes = actual, .admission_available_bytes = actual},
                PhysicalMemoryOwner::GraphSnapshotArena, budget);
            PhysicalMemoryAuthority authority(
                std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
            auto pool = authority.reserveNewAllocations(
                DeviceId::rocm(0), PhysicalMemoryOwner::GraphSnapshotArena, budget);
            if (reject)
            {
                EXPECT_THROW((void)pool.claimAllocation(actual), std::logic_error);
                EXPECT_EQ(pool.materializedBytes(), 0u);
            }
            else
            {
                for (int repeat = 0; repeat < 3; ++repeat)
                {
                    {
                        auto lease = pool.claimAllocation(actual);
                        EXPECT_EQ(pool.materializedBytes(), actual);
                        EXPECT_EQ(pool.remainingBytes(), 0u);
                    }
                    EXPECT_EQ(pool.materializedBytes(), 0u);
                    EXPECT_EQ(pool.remainingBytes(), actual);
                }
            }
        };
        replay(former_budget, true);
        replay(capacity.main_phase_payload->wide_main_bytes, false);
    }

    /** @test An actual CPU root rejects the old absent issuer and retains its admitted lease. */
    TEST(ROCmAttentionFixtureAdmission, CPUFragmentUsesActualAdmittedRootAuthority)
    {
        ModelMemoryProfile profile;
        profile.architecture = "qwen2";
        profile.n_layers = 4; profile.d_model = 8; profile.d_ff = 16;
        profile.n_heads = 2; profile.n_kv_heads = 1; profile.head_dim = 4;
        profile.vocab_size = 16; profile.max_seq_len = 8;
        RankExecutionPlan plan;
        plan.rank = 0; plan.first_layer = 0; plan.last_layer = 1;
        plan.has_embedding = true; plan.has_lm_head = false;
        plan.primary_device = GlobalDeviceAddress::cpu();
        plan.runtime.max_seq_len = 8; plan.runtime.batch_size = 1;
        // This allocator-only replay owns no prefix-cache tier or archive path.
        // Select the supported disabled policy rather than admitting unused
        // production-default RAM/cache budgets into its tiny observation.
        plan.runtime.prefix_cache.enabled = false;
        plan.runtime.prefix_cache.storage_mode = PrefixCacheStorageMode::Disabled;
        RankInventory inventory;
        inventory.rank = 0;
        inventory.cpu_execution = CPUExecutionGeometry::local();
        inventory.cpu_worker_threads = 1;
        inventory.cpu.memory_bytes = 1u << 28;
        inventory.cpu.free_memory_bytes = 1u << 28;
        const CPUStorageRoot<float> root(16, PhysicalMemoryOwner::ActivationArena, DeviceId::cpu());
        try {
            (void)root.allocate({});
            FAIL() << "Original absent-authority source fixture unexpectedly allocated";
        } catch (const std::invalid_argument &error) {
            EXPECT_STREQ(error.what(), "CPU storage root requires its admitted allocator authority");
        }
        const auto authority = rocm_attention_fixture::admit(profile, plan, inventory);
        ASSERT_TRUE(authority->contains(DeviceId::cpu()));
        for (const auto owner : {PhysicalMemoryOwner::PrefixArchiveStaging,
                                 PhysicalMemoryOwner::PrefixHostTier,
                                 PhysicalMemoryOwner::PrefixDeviceTier})
            EXPECT_EQ(authority->plannedBytes(DeviceId::cpu(), owner), 0u);
        const auto before = authority->remainingAdmittedNewAllocationBytes(
            DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena);
        ASSERT_GE(before, root.physicalBytes());
        {
            auto storage = root.allocate(authority);
            ASSERT_EQ(storage.size(), 16u);
            ASSERT_NE(storage.data(), nullptr);
            EXPECT_EQ(authority->remainingAdmittedNewAllocationBytes(
                DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena), before - root.physicalBytes());
        }
        EXPECT_EQ(authority->remainingAdmittedNewAllocationBytes(
            DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena), before);
        plan.has_lm_head = true;
        EXPECT_THROW(rocm_attention_fixture::admit(profile, plan, inventory), std::invalid_argument);
    }
}
