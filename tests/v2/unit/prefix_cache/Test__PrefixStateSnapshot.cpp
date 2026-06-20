#include <gtest/gtest.h>

#include "execution/prefix_cache/PrefixStateSnapshot.h"

namespace llaminar2
{
namespace
{

    PrefixBlockHandle makeBlock(int index,
                                int start,
                                int count,
                                bool includes_hybrid_payload,
                                bool hybrid_shape,
                                bool has_hybrid,
                                bool has_terminal)
    {
        PrefixBlockHandle handle;
        handle.key.fingerprint = 0x1234;
        handle.key.block_index = index;
        handle.key.token_start = start;
        handle.key.token_count = count;
        handle.layout.block_size = 4;
        handle.layout.fa_layers = 1;
        handle.layout.bytes_per_fa_layer_k = 16;
        handle.layout.bytes_per_fa_layer_v = 16;
        handle.layout.includes_hybrid_state = includes_hybrid_payload;
        handle.layout.hybrid_state_bytes = hybrid_shape ? 8 : 0;
        handle.has_hybrid_state = has_hybrid;
        handle.has_terminal_logits = has_terminal;
        handle.has_terminal_hidden = has_terminal;
        handle.total_bytes = handle.layout.totalBytes();
        handle.kv_storage = std::make_shared<std::vector<uint8_t>>(handle.layout.faKVBytes());
        handle.kv_payload = handle.kv_storage->data();
        if (includes_hybrid_payload)
        {
            handle.hybrid_storage = std::make_shared<std::vector<uint8_t>>(handle.layout.hybrid_state_bytes);
            handle.hybrid_payload = handle.hybrid_storage->data();
        }
        return handle;
    }

} // namespace

TEST(Test__PrefixStateSnapshot, ClampedToKeepsTerminalPartialBlock)
{
    PrefixLookupResult hit;
    hit.supported = true;
    hit.cache_enabled = true;
    hit.cached_tokens = 9;
    hit.block_size = 4;
    hit.has_terminal_hidden = true;
    hit.has_terminal_logits = true;
    hit.blocks.push_back(makeBlock(0, 0, 4, false, false, false, false));
    hit.blocks.push_back(makeBlock(1, 4, 4, false, false, false, false));
    hit.blocks.push_back(makeBlock(2, 8, 1, false, false, false, true));

    PrefixLookupResult clamped = hit.clampedTo(9);

    EXPECT_EQ(clamped.cached_tokens, 9);
    ASSERT_EQ(clamped.blocks.size(), 3u);
    EXPECT_EQ(clamped.blocks.back().key.token_count, 1);
    EXPECT_TRUE(clamped.has_terminal_hidden);
    EXPECT_TRUE(clamped.has_terminal_logits);
}

TEST(Test__PrefixStateSnapshot, ClampedToTrimsHybridBlocksWithoutRestorableState)
{
    PrefixLookupResult hit;
    hit.supported = true;
    hit.cache_enabled = true;
    hit.cached_tokens = 9;
    hit.block_size = 4;
    hit.has_terminal_hidden = true;
    hit.has_terminal_logits = true;
    hit.blocks.push_back(makeBlock(0, 0, 4, false, true, false, false));
    hit.blocks.push_back(makeBlock(1, 4, 4, false, true, false, false));
    hit.blocks.push_back(makeBlock(2, 8, 1, true, true, true, true));

    PrefixLookupResult full = hit.clampedTo(9);
    EXPECT_EQ(full.cached_tokens, 9);
    ASSERT_EQ(full.blocks.size(), 3u);
    EXPECT_TRUE(full.blocks.back().has_hybrid_state);

    PrefixLookupResult block_boundary = hit.clampedTo(8);
    EXPECT_EQ(block_boundary.cached_tokens, 0);
    EXPECT_TRUE(block_boundary.blocks.empty());
    EXPECT_FALSE(block_boundary.has_terminal_hidden);
    EXPECT_FALSE(block_boundary.has_terminal_logits);
}

TEST(Test__PrefixStateSnapshot, ProvenanceMarksOnlyReplaySafeStatesDecodeEquivalent)
{
    EXPECT_TRUE(isDecodeEquivalent(PrefixStateProvenance::PayloadCheckpoint));
    EXPECT_TRUE(isDecodeEquivalent(PrefixStateProvenance::LogicalCheckpoint));
    EXPECT_TRUE(isDecodeEquivalent(PrefixStateProvenance::DecodeEquivalent));
    EXPECT_TRUE(isDecodeEquivalent(PrefixStateProvenance::VerifierPrefillRowsDecodeEquivalent));

    EXPECT_FALSE(isDecodeEquivalent(PrefixStateProvenance::Unknown));
    EXPECT_FALSE(isDecodeEquivalent(PrefixStateProvenance::VerifierPrefillRows));
    EXPECT_FALSE(isDecodeEquivalent(PrefixStateProvenance::SidecarDraftOnly));

    PrefixStateSnapshot snapshot;
    snapshot.valid = true;
    snapshot.provenance = PrefixStateProvenance::VerifierPrefillRows;
    EXPECT_FALSE(snapshot.decodeEquivalent());

    snapshot.provenance = PrefixStateProvenance::VerifierPrefillRowsDecodeEquivalent;
    EXPECT_TRUE(snapshot.decodeEquivalent());
}

TEST(Test__PrefixStateSnapshot, MoveLeavesSourceEmptyForNestedPayloadHandles)
{
    PrefixStateSnapshot nested;
    nested.valid = true;
    nested.logical_checkpoint = true;
    nested.provenance = PrefixStateProvenance::LogicalCheckpoint;
    nested.cached_tokens = 4;
    nested.mtp_cached_tokens = {3};
    nested.blocks.push_back(makeBlock(0, 0, 4, true, true, true, true));
    nested.mtp_blocks.push_back(makeBlock(1, 0, 3, false, false, false, false));

    PrefixStateSnapshot source;
    source.valid = true;
    source.logical_checkpoint = false;
    source.provenance = PrefixStateProvenance::PayloadCheckpoint;
    source.cached_tokens = 8;
    source.mtp_cached_tokens = {7, 6};
    source.blocks.push_back(makeBlock(0, 0, 4, true, true, true, false));
    source.blocks.push_back(makeBlock(1, 4, 4, true, true, true, true));
    source.mtp_blocks.push_back(makeBlock(2, 0, 7, false, false, false, false));
    source.participant_snapshots.push_back(std::move(nested));

    const void *source_block_payload = source.blocks.back().hybrid_payload;
    ASSERT_NE(source_block_payload, nullptr);

    PrefixStateSnapshot moved(std::move(source));

    EXPECT_FALSE(source.valid);
    EXPECT_FALSE(source.logical_checkpoint);
    EXPECT_EQ(source.provenance, PrefixStateProvenance::Unknown);
    EXPECT_EQ(source.cached_tokens, 0);
    EXPECT_TRUE(source.mtp_cached_tokens.empty());
    EXPECT_TRUE(source.blocks.empty());
    EXPECT_TRUE(source.mtp_blocks.empty());
    EXPECT_TRUE(source.participant_snapshots.empty());

    EXPECT_TRUE(moved.valid);
    EXPECT_EQ(moved.provenance, PrefixStateProvenance::PayloadCheckpoint);
    EXPECT_EQ(moved.cached_tokens, 8);
    ASSERT_EQ(moved.blocks.size(), 2u);
    EXPECT_EQ(moved.blocks.back().hybrid_payload, source_block_payload);
    ASSERT_EQ(moved.participant_snapshots.size(), 1u);
    EXPECT_TRUE(moved.participant_snapshots.front().logical_checkpoint);

    PrefixStateSnapshot assigned;
    assigned = std::move(moved);

    EXPECT_FALSE(moved.valid);
    EXPECT_TRUE(moved.blocks.empty());
    EXPECT_TRUE(moved.participant_snapshots.empty());
    EXPECT_TRUE(assigned.valid);
    EXPECT_EQ(assigned.cached_tokens, 8);
    ASSERT_EQ(assigned.blocks.size(), 2u);
    EXPECT_EQ(assigned.blocks.back().hybrid_payload, source_block_payload);
}

} // namespace llaminar2
