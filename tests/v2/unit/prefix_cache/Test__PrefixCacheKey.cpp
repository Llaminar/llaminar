#include <gtest/gtest.h>

#include "execution/prefix_cache/PrefixCacheKey.h"

#include <unordered_set>
#include <vector>

using namespace llaminar2;

TEST(Test__PrefixCacheKey, StableHashAndHexAreDeterministic)
{
    const PrefixCacheKey key = makePrefixCacheKey(
        /*fingerprint=*/0x1234,
        /*parent_hash=*/0,
        /*block_index=*/2,
        /*token_start=*/128,
        std::vector<int32_t>{10, 11, 12});

    const PrefixCacheKey again = makePrefixCacheKey(0x1234, 0, 2, 128, {10, 11, 12});
    EXPECT_EQ(key, again);
    EXPECT_EQ(key.stableHash(), again.stableHash());
    EXPECT_EQ(key.toHex(), again.toHex());
    EXPECT_TRUE(key.valid());
}

TEST(Test__PrefixCacheKey, ChangingKeyMaterialChangesIdentity)
{
    const PrefixCacheKey baseline = makePrefixCacheKey(0x1234, 0, 0, 0, {1, 2, 3});
    EXPECT_NE(baseline, makePrefixCacheKey(0x1235, 0, 0, 0, {1, 2, 3}));
    EXPECT_NE(baseline, makePrefixCacheKey(0x1234, 9, 0, 0, {1, 2, 3}));
    EXPECT_NE(baseline, makePrefixCacheKey(0x1234, 0, 1, 0, {1, 2, 3}));
    EXPECT_NE(baseline, makePrefixCacheKey(0x1234, 0, 0, 3, {1, 2, 3}));
    EXPECT_NE(baseline, makePrefixCacheKey(0x1234, 0, 0, 0, {1, 2, 4}));
}

TEST(Test__PrefixCacheKey, HashWorksInUnorderedContainers)
{
    std::unordered_set<PrefixCacheKey, PrefixCacheKeyHasher> keys;
    keys.insert(makePrefixCacheKey(1, 0, 0, 0, {1, 2}));
    keys.insert(makePrefixCacheKey(1, 0, 1, 2, {3, 4}));

    EXPECT_TRUE(keys.contains(makePrefixCacheKey(1, 0, 0, 0, {1, 2})));
    EXPECT_FALSE(keys.contains(makePrefixCacheKey(1, 0, 0, 0, {2, 1})));
}
