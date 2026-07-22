/**
 * @file Test__NativeVNNIDispatchCache.cpp
 * @brief Host-only regressions for the shared CPU/CUDA/ROCm dispatch cache.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "kernels/common/NativeVNNIDispatchCache.h"

namespace
{
    struct CachedPolicy
    {
        int family = -1;
        int tile = -1;

        bool operator==(const CachedPolicy &) const = default;
    };

    using Cache = llaminar2::native_vnni::FixedDispatchCache<
        std::array<uint64_t, 4>,
        CachedPolicy,
        8,
        llaminar2::native_vnni::DispatchCacheArrayHasher<4>>;

    TEST(Test__NativeVNNIDispatchCache, PreservesSuccessfulSelectorResults)
    {
        Cache cache;
        const std::array<uint64_t, 4> key = {5, 4, 4096, 2048};
        const CachedPolicy expected{.family = 3, .tile = 128};
        bool selector_found = false;
        CachedPolicy value{};

        EXPECT_FALSE(cache.lookup(key, selector_found, value));
        cache.insert(key, true, expected);
        ASSERT_TRUE(cache.lookup(key, selector_found, value));
        EXPECT_TRUE(selector_found);
        EXPECT_EQ(value, expected);
    }

    TEST(Test__NativeVNNIDispatchCache, ExposesImmediateReadOnlyValueView)
    {
        Cache cache;
        const std::array<uint64_t, 4> key = {19, 1, 512, 2048};
        const CachedPolicy expected{.family = 2, .tile = 32};
        bool selector_found = false;

        EXPECT_EQ(cache.lookupValue(key, selector_found), nullptr);
        cache.insert(key, true, expected);
        const CachedPolicy *view = cache.lookupValue(key, selector_found);
        ASSERT_NE(view, nullptr);
        EXPECT_TRUE(selector_found);
        EXPECT_EQ(*view, expected);
    }

    TEST(Test__NativeVNNIDispatchCache, CachesHardMissesWithoutCreatingFallbacks)
    {
        Cache cache;
        const std::array<uint64_t, 4> key = {19, 31, 3073, 2016};
        const CachedPolicy empty{};
        bool selector_found = true;
        CachedPolicy value{.family = 7, .tile = 512};

        cache.insert(key, false, empty);
        ASSERT_TRUE(cache.lookup(key, selector_found, value));
        EXPECT_FALSE(selector_found);
        EXPECT_EQ(value, empty);
    }

    TEST(Test__NativeVNNIDispatchCache, FullGeometryPreventsPackedKeyAliasing)
    {
        Cache cache;
        const std::array<uint64_t, 4> representable = {0, 1, 4096, 2048};
        const std::array<uint64_t, 4> outside_packed_m = {0, 129, 4096, 2048};
        bool selector_found = false;
        CachedPolicy value{};

        cache.insert(
            representable,
            true,
            CachedPolicy{.family = 1, .tile = 64});
        EXPECT_FALSE(cache.lookup(outside_packed_m, selector_found, value));
    }

    TEST(Test__NativeVNNIDispatchCache, DirectMappedCollisionNeverReturnsWrongKey)
    {
        Cache cache;
        const std::array<uint64_t, 4> first = {0, 2, 4096, 2048};
        const CachedPolicy first_value{.family = 1, .tile = 64};
        cache.insert(first, true, first_value);

        std::array<uint64_t, 4> collision{};
        const auto hasher = llaminar2::native_vnni::DispatchCacheArrayHasher<4>{};
        for (uint64_t n = 4097; n < 100000; ++n)
        {
            const std::array<uint64_t, 4> candidate = {0, 2, n, 2048};
            if ((hasher(candidate) & 7U) == (hasher(first) & 7U))
            {
                collision = candidate;
                break;
            }
        }
        ASSERT_NE(collision, (std::array<uint64_t, 4>{}));
        cache.insert(
            collision,
            true,
            CachedPolicy{.family = 2, .tile = 128});

        bool selector_found = false;
        CachedPolicy value{};
        EXPECT_FALSE(cache.lookup(first, selector_found, value));
        ASSERT_TRUE(cache.lookup(collision, selector_found, value));
        EXPECT_EQ(value, (CachedPolicy{.family = 2, .tile = 128}));
    }
}
