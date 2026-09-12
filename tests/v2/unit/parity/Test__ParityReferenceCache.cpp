/**
 * @file Test__ParityReferenceCache.cpp
 * @brief Device-free tests for persistent reference-pack location boundaries.
 *
 * These tests exercise only path policy. Pack authentication and numerical
 * comparisons remain the responsibility of the existing parity framework.
 */
#include "integration/parity/ParityReferenceCache.h"
#include <gtest/gtest.h>

using llaminar2::test::parity::resolveParityReferenceCache;

TEST(ParityReferenceCache, UnconfiguredPathsAreUnchanged)
{
    EXPECT_EQ(resolveParityReferenceCache("pack"), "pack");
    EXPECT_EQ(resolveParityReferenceCache("/existing/pack"), "/existing/pack");
}

TEST(ParityReferenceCache, MountedPacksAreResolvedIdempotently)
{
    const auto root = std::filesystem::path("/cache/references");
    const auto pack = resolveParityReferenceCache("model/pack", root);
    EXPECT_EQ(pack, "/cache/references/model/pack");
    EXPECT_EQ(resolveParityReferenceCache(pack, root), pack);
}

TEST(ParityReferenceCache, ForeignAbsolutePathsAndParentTraversalAreRejected)
{
    const auto root = std::filesystem::path("/cache/references");
    for (const auto &path : {"../outside", "model/../../outside", "/other/pack", "/cache/references", ""})
        EXPECT_THROW(resolveParityReferenceCache(path, root), std::invalid_argument);
    EXPECT_THROW(resolveParityReferenceCache("pack", std::filesystem::path("relative")), std::invalid_argument);
    EXPECT_THROW(resolveParityReferenceCache("pack", std::filesystem::path("/")), std::invalid_argument);
}
