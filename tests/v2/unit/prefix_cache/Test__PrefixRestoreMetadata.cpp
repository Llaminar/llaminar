/**
 * @file Test__PrefixRestoreMetadata.cpp
 * @brief Device-free restore-reporting coverage for metadata-only and nested TP/PP lookups.
 *
 * Real rank and stage composition query injected leaf admissions. No archive
 * bytes, GPU contexts or model weights are needed to prove that a non-primary
 * participant's sections survive aggregation, clamping and a later cache miss.
 */
#include "execution/prefix_cache/PrefixRestoreMetadata.h"
#include "execution/global/GlobalOrchestrator.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "../../mocks/MockModelContext.h"
#include "../../mocks/MockRankOrchestrator.h"
#include <gtest/gtest.h>
#include <array>

namespace llaminar2::test
{
    /** @brief Leaf exposing immutable archive metadata with no payload ownership. */
    class RestoreMetadataLeaf final : public MockDeviceRunner
    {
    public:
        PrefixLookupResult admission;
        /** @brief Return the exact fixture admission used by production composition. */
        PrefixLookupResult lookupPrefix(const std::vector<int32_t> &) override { return admission; }
    };

    /**
     * @brief Inject a real local TP owner around metadata-only CPU leaves.
     * @param width Number of participant admissions in this homogeneous domain.
     * @param mtp_owner Leaf owning the late MTP/disk section, or -1 for none.
     * @return Rank composition with test-only model metadata and no backend initialization.
     */
    static std::unique_ptr<RankOrchestrator> metadataDomain(int width, int mtp_owner)
    {
        RankOrchestrator::Config config;
        config.mode = RankOrchestrator::ParallelismMode::TP;
        MockLocalTPContext::Config tp;
        std::vector<std::unique_ptr<IInferenceRunner>> leaves;
        for (int index = 0; index < width; ++index)
        {
            config.devices.push_back(GlobalDeviceAddress::cpu());
            config.weights.push_back(1.0f / width);
            tp.devices.push_back(GlobalDeviceAddress::cpu());
            tp.weights.push_back(1.0f / width);
            auto leaf = std::make_unique<RestoreMetadataLeaf>();
            auto &hit = leaf->admission;
            hit.supported = hit.cache_enabled = true;
            hit.cached_tokens = 4;
            hit.block_size = 2;
            hit.requires_terminal_hidden = hit.requires_terminal_logits = false;
            for (int block = 0; block < 2; ++block)
            {
                PrefixBlockHandle metadata;
                metadata.key.token_start = block * 2;
                metadata.key.token_count = 2;
                metadata.layout.block_size = 2;
                metadata.has_hybrid_state = true;
                metadata.layout.includes_mtp_state = index == mtp_owner && block == 1;
                metadata.tier = metadata.layout.includes_mtp_state
                    ? PrefixStorageTier::Disk : PrefixStorageTier::Ram;
                hit.blocks.push_back(metadata);
            }
            leaves.push_back(std::move(leaf));
        }
        return RankOrchestrator::createForTest(MockModelContext::createMinimal(),
            std::move(leaves), std::make_unique<MockLocalTPContext>(tp), config);
    }

    /** @test Four/eight participants preserve a terminal-only section in every ownership position. */
    TEST(PrefixRestoreMetadata, NestedDomainsIncludeEveryParticipantAndClampSources)
    {
        for (int width : {2, 4})
            for (int owner = 0; owner < 2 * width; ++owner)
            {
                SCOPED_TRACE(::testing::Message() << "participants=" << 2 * width << " mtp_owner=" << owner);
                StageRunnerRegistry registry;
                for (int stage = 0; stage < 2; ++stage)
                {
                    StageRunnerEntry entry;
                    entry.stage_id = stage;
                    entry.runner = metadataDomain(width,
                        owner / width == stage ? owner % width : -1);
                    registry.add(std::move(entry));
                }
                const auto hit = registry.lookupPrefixAll({1, 2, 3, 4});
                ASSERT_EQ(hit.cached_tokens, 4);
                const auto full = registry.prefixRestoreMetadataAll(hit);
                EXPECT_TRUE(full.hasMTPState());
                EXPECT_TRUE(full.hasHybridState());
                EXPECT_STREQ(full.storageTierName(), "mixed");
                // A later terminal section must not leak into a shorter restore.
                const auto partial = registry.prefixRestoreMetadataAll(hit.clampedTo(2));
                EXPECT_FALSE(partial.hasMTPState());
                EXPECT_TRUE(partial.hasHybridState());
                EXPECT_STREQ(partial.storageTierName(), "ram");
                const auto miss = registry.prefixRestoreMetadataAll(hit.clampedTo(0));
                EXPECT_EQ(miss.wireBits(), 0u);
                EXPECT_STREQ(miss.storageTierName(), "none");
            }
    }

    /** @test Pointer-free section declarations and all source tiers round-trip through one word. */
    TEST(PrefixRestoreMetadata, MetadataOnlySectionsAndTierUnionNeedNoPayloadPointers)
    {
        PrefixRestoreMetadata combined;
        for (const auto tier : {PrefixStorageTier::DeviceHot, PrefixStorageTier::Ram, PrefixStorageTier::Disk})
        {
            std::array<PrefixBlockHandle, 1> blocks;
            blocks[0].tier = tier;
            blocks[0].layout.includes_mtp_state = true;
            blocks[0].has_hybrid_state = true;
            const auto single = PrefixRestoreMetadata::fromBlocks(blocks);
            EXPECT_TRUE(single.hasMTPState());
            EXPECT_TRUE(single.hasHybridState());
            EXPECT_EQ(PrefixRestoreMetadata::fromWireBits(single.wireBits()).wireBits(), single.wireBits());
            combined.merge(single);
        }
        EXPECT_STREQ(combined.storageTierName(), "mixed");
        EXPECT_THROW(PrefixRestoreMetadata::fromWireBits(0x80000000u), std::invalid_argument);
        EXPECT_THROW(PrefixRestoreMetadata::fromWireBits(1u), std::invalid_argument);
    }

    /** @test Queries cannot invent stage admissions after a caller skipped lookup. */
    TEST(PrefixRestoreMetadata, MissingParticipantAdmissionsFailPrecisely)
    {
        StageRunnerRegistry registry;
        StageRunnerEntry entry;
        entry.stage_id = 0;
        entry.runner = metadataDomain(2, 1);
        registry.add(std::move(entry));
        PrefixLookupResult unowned;
        unowned.cached_tokens = 4;
        EXPECT_THROW(registry.prefixRestoreMetadataAll(unowned), std::logic_error);
    }
}
