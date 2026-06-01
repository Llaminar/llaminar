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

} // namespace llaminar2
