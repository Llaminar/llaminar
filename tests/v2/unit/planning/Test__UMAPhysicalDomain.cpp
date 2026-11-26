/**
 * @file Test__UMAPhysicalDomain.cpp
 * @brief UMA single-charge, alias non-double-charge and staging contracts.
 */
#include "planning/UMAPhysicalDomain.h"
#include <gtest/gtest.h>

using namespace llaminar2;

/** @test Aliases never double-charge; distinct allocations each charge bytes. */
TEST(UMAPhysicalDomain, AliasesChargeOnceDistinctAllocationsChargeEach)
{
    UMAPhysicalDomainLedger ledger;
    UMAAllocationIdentity gpu_view{1, 7, 4096};
    UMAAllocationIdentity ram_view{1, 7, 4096};      // Same allocation, RAM alias.
    UMAAllocationIdentity distinct{2, 7, 2048};
    EXPECT_TRUE(ledger.charge(gpu_view));
    EXPECT_FALSE(ledger.charge(ram_view));           // Alias: no double charge.
    EXPECT_EQ(ledger.chargedBytes(), 4096u);
    EXPECT_EQ(ledger.allocationCount(), 1u);
    EXPECT_TRUE(ledger.charge(distinct));
    EXPECT_EQ(ledger.chargedBytes(), 6144u);
    EXPECT_EQ(ledger.allocationCount(), 2u);
}

/** @test Exact release; aliases share the single charge and release exactly once. */
TEST(UMAPhysicalDomain, ExactReleaseAndAliasRetirement)
{
    UMAPhysicalDomainLedger ledger;
    UMAAllocationIdentity gpu_view{1, 7, 4096};
    UMAAllocationIdentity ram_view{1, 7, 4096};
    EXPECT_TRUE(ledger.charge(gpu_view));
    EXPECT_FALSE(ledger.charge(ram_view));
    // Releasing the RAM alias view retires the single charge exactly once.
    EXPECT_TRUE(ledger.release(ram_view));
    EXPECT_EQ(ledger.chargedBytes(), 0u);
    EXPECT_EQ(ledger.allocationCount(), 0u);
    EXPECT_FALSE(ledger.release(gpu_view)); // Double retirement refuses.
    // Distinct domains are separate accounts.
    UMAAllocationIdentity other_domain{1, 8, 4096};
    EXPECT_TRUE(ledger.charge(other_domain));
    EXPECT_EQ(ledger.chargedBytes(), 4096u);
}

/** @test RAM reservation off does not remove the staging charge. */
TEST(UMAPhysicalDomain, StagingChargePersistsWithRamOff)
{
    UMAStagingCharge staging(8192);
    EXPECT_TRUE(staging.reserve(512));
    EXPECT_EQ(staging.staged(), 512u);
    // A RAM-off configuration still pays bounded I/O staging (distinct resource).
    EXPECT_TRUE(staging.reserve(4096));
    EXPECT_EQ(staging.staged(), 4608u);
    EXPECT_FALSE(staging.reserve(8192)); // Beyond maximum: refuses.
    EXPECT_EQ(staging.staged(), 4608u);
    EXPECT_TRUE(staging.release(4096));
    EXPECT_EQ(staging.staged(), 512u);
    EXPECT_FALSE(staging.release(4096)); // Over-release refuses.
    EXPECT_EQ(staging.staged(), 512u);
}

/** @test Ledger capacity refuses before mutating. */
TEST(UMAPhysicalDomain, CapacityRefusesBeforeMutation)
{
    UMAPhysicalDomainLedger ledger;
    for (std::uint64_t i = 1; i <= UMAPhysicalDomainLedger::capacity(); ++i)
        EXPECT_TRUE(ledger.charge({i, 1, 1}));
    EXPECT_FALSE(ledger.charge({9999, 1, 1}));
    EXPECT_EQ(ledger.allocationCount(), UMAPhysicalDomainLedger::capacity());
}
