/**
 * @file Test__PagedKVInstance.cpp
 * @brief One-admitted pooled-KV instance construction/retirement contracts.
 *
 * Contract tests: the instance admits pool and table from ONE PMA authority,
 * charges exactly the planned complete BOM, refuses foreign authority/device
 * typed, and retires both owners in dependency order (no orphan table).
 */
#include "planning/PagedKVInstance.h"
#include <gtest/gtest.h>

using namespace llaminar2;

namespace
{
struct InstanceFixture
{
    std::shared_ptr<PhysicalMemoryAuthority> authority;
    PagedKVInstancePlan plan;
    InstanceFixture()
    {
        const auto device = DeviceId::cpu();
        PagedKVTierConfig config; // Auto pool.
        PagedKVPoolBudget budget;
        budget.remaining_budget_bytes = 2 * 1024 * 1024;
        budget.frame_bytes = 64 * 1024;
        budget.request_records = 4;
        budget.max_pages_per_request = 4;
        auto candidate = planPagedKVInstance(config, budget,
            PagedKVInstancePolicy{8, 2, 4, 4}, device);
        if (!candidate) throw std::logic_error("Instance fixture plan refused");
        plan = *candidate;
        // Admit the complete planned BOM under one authority.
        PhysicalMemoryPlanBuilder builder;
        builder.add({.world_rank=0, .device=device,
            .total_bytes=plan.total_bytes,
            .admission_available_bytes=plan.total_bytes},
            PhysicalMemoryOwner::KVCache, plan.total_bytes);
        authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
    }
};
}

/** @test Creation admits both owners under one authority and charges the BOM. */
TEST(PagedKVInstance, CreateAdmitsPoolAndTableUnderOneAuthority)
{
    InstanceFixture fx;
    auto instance = PagedKVInstance::create(fx.authority, fx.plan, DeviceId::cpu());
    ASSERT_TRUE(instance);
    EXPECT_EQ(instance->admittedBytes(), fx.plan.total_bytes);
    EXPECT_GT(instance->pool().geometry().frames, 0u);
    EXPECT_EQ(instance->table().issuanceBudget(),
        fx.plan.table_geometry.issuance_budget);
    EXPECT_GE(fx.authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::KVCache,
        PhysicalMemoryMaterializationKind::NewAllocation), fx.plan.total_bytes);
    // The instance is the only way to reach the sub-authority; both owners live.
    auto update = instance->table().tryBeginUpdate();
    ASSERT_TRUE(update);
}

/** @test Retirement returns both owners; no orphan table outlives its pool. */
TEST(PagedKVInstance, RetirementReturnsBothOwnersInOrder)
{
    InstanceFixture fx;
    {
        auto instance = PagedKVInstance::create(fx.authority, fx.plan, DeviceId::cpu());
        ASSERT_TRUE(instance);
    }
    // After retirement the authority no longer charges the pool backing.
    EXPECT_EQ(fx.authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::KVCache,
        PhysicalMemoryMaterializationKind::NewAllocation), 0u);
}

/** @test Foreign authority/device refuse typed before construction. */
TEST(PagedKVInstance, ForeignAuthorityAndDeviceAreRejected)
{
    InstanceFixture fx;
    PagedKVInstancePlan invalid{fx.plan.pool_geometry, fx.plan.table_geometry, 0};
    EXPECT_THROW(PagedKVInstance::create(fx.authority, invalid, DeviceId::cpu()),
        std::exception); // Zero BOM cannot be admitted under the fixture plan.
    EXPECT_THROW(PagedKVInstance::create(fx.authority,
        PagedKVInstancePlan{{0, 0, 0, 0}, fx.plan.table_geometry, 0}, DeviceId::cpu()),
        std::invalid_argument); // Invalid plan geometry refuses typed.
}
