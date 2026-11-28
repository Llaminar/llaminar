/**
 * @file Test__DiskExtentQuota.cpp
 * @brief Finite quota, extent identity, checked arithmetic and exact retirement.
 */
#include "execution/prefix_cache/DiskExtentQuota.h"
#include <gtest/gtest.h>

using namespace llaminar2;

/** @test Quota admission is overcommit-free; retirement returns bytes exactly once. */
TEST(DiskExtentQuota, QuotaAndExactRetirementContracts)
{
    DiskExtentQuota quota(100, 8);
    DiskExtentKey first{1, 7, 0xabc, 60};
    DiskExtentKey second{2, 7, 0xdef, 40};
    EXPECT_TRUE(quota.admit(first));
    EXPECT_TRUE(quota.admit(second));
    EXPECT_EQ(quota.bytesAdmitted(), 100u);
    DiskExtentKey third{3, 7, 0x111, 1};
    EXPECT_FALSE(quota.admit(third)); // Quota exhausted; no partial state.
    EXPECT_EQ(quota.bytesAdmitted(), 100u);
    EXPECT_TRUE(quota.retire(first));
    EXPECT_FALSE(quota.contains(first));
    EXPECT_EQ(quota.bytesAdmitted(), 40u);
    EXPECT_FALSE(quota.retire(first)); // Double retirement refuses.
    EXPECT_TRUE(quota.admit(third));   // Space returned by retirement.
    EXPECT_EQ(quota.bytesAdmitted(), 41u);
}

/** @test Zero quota admits NOTHING: no underflow admission into an empty ledger. */
TEST(DiskExtentQuota, ZeroQuotaNeverAdmits)
{
    DiskExtentQuota zero(0, 4);
    EXPECT_FALSE(zero.admit(DiskExtentKey{1, 1, 1, 1}));
    EXPECT_EQ(zero.count(), 0u);
    EXPECT_EQ(zero.bytesAdmitted(), 0u);
    // Large extent must not underflow the subtraction either.
    EXPECT_FALSE(zero.admit(DiskExtentKey{2, 1, 1, SIZE_MAX / 2}));
    EXPECT_EQ(zero.count(), 0u);
}

/** @test Extent identity is generation/namespace/digest exact; same bytes differ. */
TEST(DiskExtentQuota, ExtentIdentityIsExact)
{
    DiskExtentQuota quota(1000, 8);
    DiskExtentKey a{1, 7, 0xabc, 50};
    DiskExtentKey same_bytes_other_gen{2, 7, 0xabc, 50};
    EXPECT_TRUE(quota.admit(a));
    EXPECT_FALSE(quota.admit(a));                        // Duplicate admission refuses.
    EXPECT_TRUE(quota.admit(same_bytes_other_gen));      // Distinct identity admitted.
    EXPECT_EQ(quota.bytesAdmitted(), 100u);
    EXPECT_TRUE(quota.retire(a));
    EXPECT_FALSE(quota.retire(a));
    EXPECT_EQ(quota.bytesAdmitted(), 50u);
}

/** @test Journal capacity overflow refuses before mutating. */
TEST(DiskExtentQuota, CapacityOverflowRefusesBeforeMutation)
{
    DiskExtentQuota quota(1ull << 40, 4);
    for (std::size_t i = 0; i < quota.capacity(); ++i)
        EXPECT_TRUE(quota.admit({static_cast<std::uint64_t>(i + 1), 1, 1, 1}));
    EXPECT_FALSE(quota.admit({9999, 1, 1, 1}));
    EXPECT_EQ(quota.count(), quota.capacity());
}

/** @test B1 (R15): 0 < quota < extent must refuse WITHOUT subtraction wrap. */
TEST(DiskExtentQuota, ExtentLargerThanQuotaRefusesWithoutWrap)
{
    DiskExtentQuota quota(8, 4);
    EXPECT_FALSE(quota.admit(DiskExtentKey{1, 1, 1, 9})); // 9 > 8: refuse, no wrap.
    EXPECT_EQ(quota.count(), 0u);
    EXPECT_EQ(quota.bytesAdmitted(), 0u);
    // Exact boundary admits; the next byte refuses.
    EXPECT_TRUE(quota.admit(DiskExtentKey{2, 1, 1, 8}));
    EXPECT_EQ(quota.bytesAdmitted(), 8u);
    EXPECT_FALSE(quota.admit(DiskExtentKey{3, 1, 1, 1}));
}

/** @test A zero-length extent is refused; partial admission never mutates. */
TEST(DiskExtentQuota, ZeroLengthExtentRefuses)
{
    DiskExtentQuota quota(100, 4);
    EXPECT_FALSE(quota.admit(DiskExtentKey{1, 1, 1, 0}));
    EXPECT_EQ(quota.count(), 0u);
}

/** @test The journal capacity is the ADMITTED membership, not a constant. */
TEST(DiskExtentQuota, JournalCapacityIsAdmittedMembership)
{
    DiskExtentQuota quota(1ull << 40, 3);
    EXPECT_EQ(quota.capacity(), 3u);
    EXPECT_TRUE(quota.admit(DiskExtentKey{1, 1, 1, 1}));
    EXPECT_TRUE(quota.admit(DiskExtentKey{2, 1, 1, 1}));
    EXPECT_TRUE(quota.admit(DiskExtentKey{3, 1, 1, 1}));
    EXPECT_FALSE(quota.admit(DiskExtentKey{4, 1, 1, 1})); // Membership full.
    EXPECT_EQ(quota.count(), 3u);
    // Zero membership is a configuration error, never a silent off state.
    EXPECT_THROW((void)DiskExtentQuota(10, 0), std::invalid_argument);
}
