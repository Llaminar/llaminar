/**
 * @file Test__PagedKVInstanceFactory.cpp
 * @brief Tier-intent to live-instance factory contracts.
 *
 * These tests prove the factory is the only admitted path to a live pooled
 * instance: a valid intent plus a sufficient remaining budget yields a live
 * instance whose pool and table belong to one PMA authority, while an
 * insufficient budget or an invalid authored intent yields a precise refusal
 * (never a partial instance, never an invented default).
 */
#include "planning/PagedKVInstanceFactory.h"
#include <gtest/gtest.h>

using namespace llaminar2;

namespace
{
struct FactoryFixture
{
    std::shared_ptr<PhysicalMemoryAuthority> authority;
    PagedKVPoolBudget budget;
    PagedKVInstancePolicy policy{8, 2, 4, 4};
    FactoryFixture()
    {
        budget.remaining_budget_bytes = 2 * 1024 * 1024;
        budget.frame_bytes = 64 * 1024;
        budget.request_records = 4;
        budget.max_pages_per_request = 4;
        const auto device = DeviceId::cpu();
        // Admit exactly the viable instance BOM so the factory can charge it.
        auto plan = planPagedKVInstance(PagedKVTierConfig{}, budget, policy, device);
        if (!plan) throw std::logic_error("Factory fixture plan refused");
        PhysicalMemoryPlanBuilder builder;
        builder.add({.world_rank=0, .device=device,
            .total_bytes=plan->total_bytes,
            .admission_available_bytes=plan->total_bytes},
            PhysicalMemoryOwner::KVCache, plan->total_bytes);
        authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
    }
};
}

/** @test A valid auto intent yields one live instance under the authority. */
TEST(PagedKVInstanceFactory, ValidAutoIntentYieldsLiveInstance)
{
    FactoryFixture fx;
    auto outcome = admitConfiguredPagedKVInstance(fx.authority, PagedKVTierConfig{},
        fx.budget, fx.policy, DeviceId::cpu());
    ASSERT_TRUE(outcome.ok()) << outcome.refusal;
    EXPECT_GT(outcome.instance->pool().geometry().frames, 0u);
    EXPECT_EQ(outcome.instance->table().issuanceBudget(), 6u); // banks + records.
    EXPECT_GT(outcome.instance->admittedBytes(), 0u);
    EXPECT_FALSE(outcome.instance->authority() == nullptr);
    // The live instance is usable: a bank can be reserved and begun.
    auto reservation = outcome.instance->table().reserveAcceptedWork(1);
    ASSERT_TRUE(reservation);
    auto update = outcome.instance->table().tryBeginReserved(*reservation);
    ASSERT_TRUE(update);
}

/** @test An insufficient budget refuses precisely; no partial instance exists. */
TEST(PagedKVInstanceFactory, InsufficientBudgetRefusesPrecisely)
{
    FactoryFixture fx;
    auto small = fx.budget;
    small.remaining_budget_bytes = 64 * 1024; // One frame, no table room.
    auto outcome = admitConfiguredPagedKVInstance(fx.authority, PagedKVTierConfig{},
        small, fx.policy, DeviceId::cpu());
    EXPECT_FALSE(outcome.ok());
    EXPECT_FALSE(outcome.refusal.empty());
    EXPECT_EQ(fx.authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::KVCache,
        PhysicalMemoryMaterializationKind::NewAllocation), 0u);
}

/** @test An invalid authored intent fails typed before any instance is planned. */
TEST(PagedKVInstanceFactory, InvalidAuthoredIntentFailsTyped)
{
    FactoryFixture fx;
    PagedKVTierConfig invalid;
    invalid.disk_enabled = true; // No quota/directory.
    EXPECT_THROW(admitConfiguredPagedKVInstance(fx.authority, invalid,
        fx.budget, fx.policy, DeviceId::cpu()), std::invalid_argument);
}
