/**
 * @file Test__ExpertWeightPayloadProvider.cpp
 * @brief Unit tests for ExpertWeightPayloadProvider lifecycle tracking.
 *
 * Tests the register/query/mark/release lifecycle of the payload provider
 * which replaces string-matching shouldRetainRawForLazyMoE() with
 * metadata-based expert preparation state tracking.
 */

#include <gtest/gtest.h>
#include "execution/moe/ExpertWeightPayloadProvider.h"

using namespace llaminar2;

// ─── Helper ──────────────────────────────────────────────────────────────────

static ExpertWeightBlobs makeTestBlobs(size_t n_bytes = 64)
{
    ExpertWeightBlobs blobs;
    blobs.gate = std::vector<uint8_t>(n_bytes, 0xAB);
    blobs.up = std::vector<uint8_t>(n_bytes, 0xCD);
    blobs.down = std::vector<uint8_t>(n_bytes, 0xEF);
    return blobs;
}

// ─── Basic lifecycle ─────────────────────────────────────────────────────────

TEST(Test__ExpertWeightPayloadProvider, EmptyProviderHasNoPayloads)
{
    ExpertWeightPayloadProvider provider;
    EXPECT_EQ(provider.totalPayloadCount(), 0u);
    EXPECT_EQ(provider.totalPayloadBytes(), 0u);
    EXPECT_FALSE(provider.hasPayload(0, 0));
    EXPECT_EQ(provider.payloadPtr(0, 0), nullptr);
    EXPECT_FALSE(provider.payloadFor(0, 0).has_value());
}

TEST(Test__ExpertWeightPayloadProvider, RegisterAndQuerySinglePayload)
{
    ExpertWeightPayloadProvider provider;
    auto blobs = makeTestBlobs(128);
    provider.registerPayload(0, 3, blobs);

    EXPECT_TRUE(provider.hasPayload(0, 3));
    EXPECT_FALSE(provider.hasPayload(0, 0));
    EXPECT_FALSE(provider.hasPayload(1, 3));
    EXPECT_EQ(provider.totalPayloadCount(), 1u);

    auto retrieved = provider.payloadFor(0, 3);
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_FALSE(retrieved->empty());
    EXPECT_EQ(retrieved->gate.size(), 128u);
}

TEST(Test__ExpertWeightPayloadProvider, RegisterOverwritesExisting)
{
    ExpertWeightPayloadProvider provider;
    provider.registerPayload(0, 0, makeTestBlobs(64));
    EXPECT_EQ(provider.totalPayloadBytes(), 192u); // 64*3

    provider.registerPayload(0, 0, makeTestBlobs(256));
    EXPECT_EQ(provider.totalPayloadCount(), 1u);
    EXPECT_EQ(provider.totalPayloadBytes(), 768u); // 256*3
}

TEST(Test__ExpertWeightPayloadProvider, RegisterBatchPayloads)
{
    ExpertWeightPayloadProvider provider;
    std::unordered_map<int, ExpertWeightBlobs> batch;
    batch[0] = makeTestBlobs(32);
    batch[1] = makeTestBlobs(64);
    batch[2] = makeTestBlobs(96);
    provider.registerPayloads(5, std::move(batch));

    EXPECT_EQ(provider.totalPayloadCount(), 3u);
    EXPECT_TRUE(provider.hasPayload(5, 0));
    EXPECT_TRUE(provider.hasPayload(5, 1));
    EXPECT_TRUE(provider.hasPayload(5, 2));
    EXPECT_FALSE(provider.hasPayload(5, 3));
}

TEST(Test__ExpertWeightPayloadProvider, PayloadPtrReturnsValidPointer)
{
    ExpertWeightPayloadProvider provider;
    provider.registerPayload(2, 7, makeTestBlobs(100));

    const ExpertWeightBlobs *ptr = provider.payloadPtr(2, 7);
    ASSERT_NE(ptr, nullptr);
    EXPECT_FALSE(ptr->empty());
    EXPECT_EQ(ptr->gate.size(), 100u);

    EXPECT_EQ(provider.payloadPtr(2, 8), nullptr);
}

TEST(Test__ExpertWeightPayloadProvider, PayloadsForLayerReturnsAll)
{
    ExpertWeightPayloadProvider provider;
    provider.registerPayload(3, 0, makeTestBlobs(10));
    provider.registerPayload(3, 4, makeTestBlobs(20));
    provider.registerPayload(3, 7, makeTestBlobs(30));
    provider.registerPayload(5, 0, makeTestBlobs(40)); // different layer

    auto layer3 = provider.payloadsForLayer(3);
    EXPECT_EQ(layer3.size(), 3u);
    EXPECT_TRUE(layer3.count(0));
    EXPECT_TRUE(layer3.count(4));
    EXPECT_TRUE(layer3.count(7));

    auto layer5 = provider.payloadsForLayer(5);
    EXPECT_EQ(layer5.size(), 1u);

    auto layer99 = provider.payloadsForLayer(99);
    EXPECT_TRUE(layer99.empty());
}

// ─── Preparation state tracking ─────────────────────────────────────────────

TEST(Test__ExpertWeightPayloadProvider, InitialStateUnprepared)
{
    ExpertWeightPayloadProvider provider;

    // Even without registering a payload, state should indicate raw data required
    EXPECT_TRUE(provider.isRawDataRequired(0, 0));
    EXPECT_FALSE(provider.isExpertPrepared(0, 0));
    EXPECT_FALSE(provider.isExpertTransferred(0, 0));
}

TEST(Test__ExpertWeightPayloadProvider, MarkPreparedReleasesRawDataRequirement)
{
    ExpertWeightPayloadProvider provider;

    // Before any marking, raw data is required
    EXPECT_TRUE(provider.isRawDataRequired(0, 0));

    provider.markExpertPrepared(0, 0);
    EXPECT_TRUE(provider.isExpertPrepared(0, 0));
    EXPECT_FALSE(provider.isRawDataRequired(0, 0));
}

TEST(Test__ExpertWeightPayloadProvider, MarkTransferredReleasesRawDataRequirement)
{
    ExpertWeightPayloadProvider provider;

    // Before any marking, raw data is required
    EXPECT_TRUE(provider.isRawDataRequired(0, 0));

    provider.markExpertTransferred(0, 0);
    EXPECT_TRUE(provider.isExpertTransferred(0, 0));
    EXPECT_FALSE(provider.isRawDataRequired(0, 0));
}

TEST(Test__ExpertWeightPayloadProvider, MarkRawReleased)
{
    ExpertWeightPayloadProvider provider;
    provider.markExpertPrepared(1, 5);
    provider.markExpertRawReleased(1, 5);

    // Raw data no longer required (prepared + released)
    EXPECT_FALSE(provider.isRawDataRequired(1, 5));
}

TEST(Test__ExpertWeightPayloadProvider, AllExpertsPreparedOrTransferred_AllPrepared)
{
    ExpertWeightPayloadProvider provider;
    const int num_experts = 4;

    EXPECT_FALSE(provider.allExpertsPreparedOrTransferred(0, num_experts));

    provider.markExpertPrepared(0, 0);
    provider.markExpertPrepared(0, 1);
    provider.markExpertPrepared(0, 2);
    EXPECT_FALSE(provider.allExpertsPreparedOrTransferred(0, num_experts));

    provider.markExpertPrepared(0, 3);
    EXPECT_TRUE(provider.allExpertsPreparedOrTransferred(0, num_experts));
}

TEST(Test__ExpertWeightPayloadProvider, AllExpertsPreparedOrTransferred_MixedStates)
{
    ExpertWeightPayloadProvider provider;

    provider.markExpertPrepared(0, 0);
    provider.markExpertTransferred(0, 1);
    provider.markExpertPrepared(0, 2);
    provider.markExpertTransferred(0, 3);

    EXPECT_TRUE(provider.allExpertsPreparedOrTransferred(0, 4));
}

TEST(Test__ExpertWeightPayloadProvider, AllExpertsPreparedOrTransferred_NotAllReady)
{
    ExpertWeightPayloadProvider provider;

    provider.markExpertPrepared(0, 0);
    provider.markExpertPrepared(0, 2);
    // Expert 1 and 3 not marked

    EXPECT_FALSE(provider.allExpertsPreparedOrTransferred(0, 4));
}

// ─── Cleanup operations ─────────────────────────────────────────────────────

TEST(Test__ExpertWeightPayloadProvider, RemovePayloadSingle)
{
    ExpertWeightPayloadProvider provider;
    provider.registerPayload(0, 0, makeTestBlobs());
    provider.registerPayload(0, 1, makeTestBlobs());

    provider.removePayload(0, 0);
    EXPECT_FALSE(provider.hasPayload(0, 0));
    EXPECT_TRUE(provider.hasPayload(0, 1));
    EXPECT_EQ(provider.totalPayloadCount(), 1u);
}

TEST(Test__ExpertWeightPayloadProvider, RemoveLayer)
{
    ExpertWeightPayloadProvider provider;
    provider.registerPayload(0, 0, makeTestBlobs());
    provider.registerPayload(0, 1, makeTestBlobs());
    provider.registerPayload(1, 0, makeTestBlobs());

    provider.removeLayer(0);
    EXPECT_FALSE(provider.hasPayload(0, 0));
    EXPECT_FALSE(provider.hasPayload(0, 1));
    EXPECT_TRUE(provider.hasPayload(1, 0));
    EXPECT_EQ(provider.totalPayloadCount(), 1u);
}

TEST(Test__ExpertWeightPayloadProvider, ClearAll)
{
    ExpertWeightPayloadProvider provider;
    provider.registerPayload(0, 0, makeTestBlobs());
    provider.registerPayload(1, 0, makeTestBlobs());
    provider.markExpertPrepared(0, 0);

    provider.clear();
    EXPECT_EQ(provider.totalPayloadCount(), 0u);
    EXPECT_EQ(provider.totalPayloadBytes(), 0u);
    EXPECT_FALSE(provider.hasPayload(0, 0));
    EXPECT_FALSE(provider.isExpertPrepared(0, 0));
}

// ─── Multi-layer scenarios ───────────────────────────────────────────────────

TEST(Test__ExpertWeightPayloadProvider, RegisterPayloadImplicitlyMarksTransferred)
{
    ExpertWeightPayloadProvider provider;

    // registerPayload marks transferred=true
    provider.registerPayload(0, 0, makeTestBlobs());
    EXPECT_TRUE(provider.isExpertTransferred(0, 0));
    EXPECT_FALSE(provider.isRawDataRequired(0, 0));
}

TEST(Test__ExpertWeightPayloadProvider, MultiLayerIndependence)
{
    ExpertWeightPayloadProvider provider;

    // Layer 0: 4 experts
    for (int e = 0; e < 4; ++e)
    {
        provider.registerPayload(0, e, makeTestBlobs(100));
        provider.markExpertPrepared(0, e);
    }

    // Layer 1: 4 experts, only 2 explicitly marked prepared (but registerPayload already marks transferred)
    for (int e = 0; e < 4; ++e)
        provider.registerPayload(1, e, makeTestBlobs(100));

    // registerPayload marks all as transferred, so all should be ready
    EXPECT_TRUE(provider.allExpertsPreparedOrTransferred(0, 4));
    EXPECT_TRUE(provider.allExpertsPreparedOrTransferred(1, 4));
    EXPECT_EQ(provider.totalPayloadCount(), 8u);

    // Also explicitly mark 2 as prepared (redundant but valid)
    provider.markExpertPrepared(1, 0);
    provider.markExpertPrepared(1, 1);
    EXPECT_TRUE(provider.allExpertsPreparedOrTransferred(1, 4));
}

TEST(Test__ExpertWeightPayloadProvider, TotalPayloadBytes)
{
    ExpertWeightPayloadProvider provider;

    // Each blob has gate + up + down
    provider.registerPayload(0, 0, makeTestBlobs(100));
    provider.registerPayload(0, 1, makeTestBlobs(200));
    provider.registerPayload(1, 0, makeTestBlobs(300));

    EXPECT_EQ(provider.totalPayloadBytes(), 1800u); // (100+200+300)*3
}

// ─── Edge cases ──────────────────────────────────────────────────────────────

TEST(Test__ExpertWeightPayloadProvider, RemoveNonExistentPayloadIsNoOp)
{
    ExpertWeightPayloadProvider provider;
    provider.removePayload(99, 99); // should not throw
    provider.removeLayer(99);       // should not throw
    EXPECT_EQ(provider.totalPayloadCount(), 0u);
}

TEST(Test__ExpertWeightPayloadProvider, MarkWithoutRegister)
{
    ExpertWeightPayloadProvider provider;
    // Mark state without registering a payload — should work
    provider.markExpertPrepared(0, 0);
    EXPECT_TRUE(provider.isExpertPrepared(0, 0));
    EXPECT_FALSE(provider.isRawDataRequired(0, 0));
    // But no payload exists
    EXPECT_FALSE(provider.hasPayload(0, 0));
}

TEST(Test__ExpertWeightPayloadProvider, ZeroExpertsUnknownLayerReturnsFalse)
{
    ExpertWeightPayloadProvider provider;
    // Unknown layer returns false (conservative)
    EXPECT_FALSE(provider.allExpertsPreparedOrTransferred(0, 0));
}
