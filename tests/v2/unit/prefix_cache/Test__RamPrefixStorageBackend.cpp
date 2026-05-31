#include <gtest/gtest.h>

#include "execution/prefix_cache/RamPrefixStorageBackend.h"

#include <algorithm>

using namespace llaminar2;

namespace
{
    PrefixPayloadLayout makeLayout(size_t k_bytes = 16, size_t v_bytes = 16)
    {
        PrefixPayloadLayout layout;
        layout.block_size = 2;
        layout.fa_layers = 1;
        layout.total_layers = 1;
        layout.bytes_per_fa_layer_k = k_bytes;
        layout.bytes_per_fa_layer_v = v_bytes;
        return layout;
    }

    PrefixCacheKey keyFor(int block)
    {
        return makePrefixCacheKey(0xfeed, 0, block, block * 2, {block, block + 1});
    }
} // namespace

TEST(Test__RamPrefixStorageBackend, AllocatesTypedPayloadSegmentsWithinBudget)
{
    RamPrefixStorageBackend backend(256);
    PrefixPayloadLayout layout = makeLayout();
    layout.includes_terminal_hidden = true;
    layout.terminal_hidden_bytes = 8;

    auto handle = backend.allocate(keyFor(0), layout);
    ASSERT_TRUE(handle.valid());
    EXPECT_EQ(handle.total_bytes, 40u);
    EXPECT_EQ(backend.usedBytes(), 40u);
    ASSERT_NE(handle.kvKData(), nullptr);
    ASSERT_NE(handle.kvVData(), nullptr);
    ASSERT_NE(handle.terminal_hidden, nullptr);

    std::fill(handle.kvKData(), handle.kvKData() + handle.kvKBytes(), 0x11);
    std::fill(handle.kvVData(), handle.kvVData() + handle.kvVBytes(), 0x22);
    EXPECT_EQ(handle.kv_storage->front(), 0x11);
    EXPECT_EQ(handle.kv_storage->at(handle.kvKBytes()), 0x22);

    EXPECT_TRUE(backend.release(handle));
    EXPECT_EQ(backend.usedBytes(), 0u);
    EXPECT_FALSE(backend.release(handle));
}

TEST(Test__RamPrefixStorageBackend, AllocatesHybridPayloadSegment)
{
    RamPrefixStorageBackend backend(256);
    PrefixPayloadLayout layout = makeLayout();
    layout.includes_hybrid_state = true;
    layout.hybrid_host_state_bytes = 10;
    layout.hybrid_state_bytes = 10;

    auto handle = backend.allocate(keyFor(0), layout);
    ASSERT_TRUE(handle.valid());
    EXPECT_EQ(handle.total_bytes, 42u);
    ASSERT_NE(handle.kvKData(), nullptr);
    ASSERT_NE(handle.kvVData(), nullptr);
    ASSERT_NE(handle.hybrid_payload, nullptr);
    ASSERT_NE(handle.hybrid_storage, nullptr);
    EXPECT_EQ(handle.hybrid_storage->size(), 10u);
    EXPECT_FALSE(handle.has_hybrid_state);
}

TEST(Test__RamPrefixStorageBackend, RejectsBlocksThatDoNotFit)
{
    RamPrefixStorageBackend backend(31);
    EXPECT_FALSE(backend.canStore(32));

    auto handle = backend.allocate(keyFor(0), makeLayout());
    EXPECT_FALSE(handle.valid());
    EXPECT_EQ(backend.usedBytes(), 0u);
}

TEST(Test__RamPrefixStorageBackend, HydrateToRamReturnsSameHandle)
{
    RamPrefixStorageBackend backend(128);
    auto handle = backend.allocate(keyFor(1), makeLayout());
    ASSERT_TRUE(handle.valid());

    PrefixBlockHandle hydrated;
    ASSERT_TRUE(backend.hydrateToRam(handle, &hydrated));
    EXPECT_EQ(hydrated.key, handle.key);
    EXPECT_EQ(hydrated.kv_storage, handle.kv_storage);
}
