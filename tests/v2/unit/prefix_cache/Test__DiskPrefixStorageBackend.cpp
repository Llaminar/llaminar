#include <gtest/gtest.h>

#include "execution/prefix_cache/DiskPrefixStorageBackend.h"
#include "execution/prefix_cache/RamPrefixStorageBackend.h"

#include <chrono>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>

using namespace llaminar2;

namespace
{
    PrefixPayloadLayout makeLayout()
    {
        PrefixPayloadLayout layout;
        layout.block_size = 2;
        layout.fa_layers = 1;
        layout.total_layers = 1;
        layout.bytes_per_fa_layer_k = 8;
        layout.bytes_per_fa_layer_v = 8;
        layout.includes_terminal_logits = true;
        layout.terminal_logits_bytes = 12;
        return layout;
    }

    PrefixCacheKey testKey()
    {
        return makePrefixCacheKey(0x12345678, 0, 0, 0, {1, 2});
    }

    std::filesystem::path tempDir()
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::filesystem::temp_directory_path() / ("llaminar_prefix_disk_" + std::to_string(stamp));
    }
} // namespace

TEST(Test__DiskPrefixStorageBackend, WritesAndReadsRamBlockWithChecksums)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    RamPrefixStorageBackend ram(1024);
    const auto layout = makeLayout();
    auto handle = ram.allocate(testKey(), layout);
    ASSERT_TRUE(handle.valid());
    for (size_t i = 0; i < handle.kv_storage->size(); ++i)
    {
        (*handle.kv_storage)[i] = static_cast<uint8_t>(i + 1);
    }
    for (size_t i = 0; i < handle.terminal_logits_storage->size(); ++i)
    {
        (*handle.terminal_logits_storage)[i] = static_cast<uint8_t>(100 + i);
    }

    DiskPrefixStorageBackend disk(dir, 1024);
    std::string error;
    ASSERT_TRUE(disk.writeBlock(handle, &error)) << error;

    PrefixBlockHandle hydrated;
    ASSERT_TRUE(disk.readBlock(testKey(), layout, &hydrated, &error)) << error;
    EXPECT_EQ(*hydrated.kv_storage, *handle.kv_storage);
    EXPECT_EQ(*hydrated.terminal_logits_storage, *handle.terminal_logits_storage);
    EXPECT_EQ(hydrated.total_bytes, handle.total_bytes);
    cleanup();
}

TEST(Test__DiskPrefixStorageBackend, PreservesHybridPayloadAndStateFlag)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    RamPrefixStorageBackend ram(1024);
    auto layout = makeLayout();
    layout.includes_hybrid_state = true;
    layout.hybrid_host_state_bytes = 10;
    layout.hybrid_state_bytes = 10;
    auto handle = ram.allocate(testKey(), layout);
    ASSERT_TRUE(handle.valid());
    ASSERT_NE(handle.hybrid_storage, nullptr);
    std::fill(handle.hybrid_storage->begin(), handle.hybrid_storage->end(), 0x5a);
    handle.has_hybrid_state = true;
    handle.has_terminal_logits = true;

    DiskPrefixStorageBackend disk(dir, 1024);
    std::string error;
    ASSERT_TRUE(disk.writeBlock(handle, &error)) << error;

    PrefixBlockHandle hydrated;
    ASSERT_TRUE(disk.readBlock(testKey(), layout, &hydrated, &error)) << error;
    ASSERT_NE(hydrated.hybrid_storage, nullptr);
    EXPECT_EQ(*hydrated.hybrid_storage, *handle.hybrid_storage);
    EXPECT_TRUE(hydrated.has_hybrid_state);
    EXPECT_TRUE(hydrated.has_terminal_logits);
    cleanup();
}

TEST(Test__DiskPrefixStorageBackend, RejectsMalformedMetadataChecksum)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    RamPrefixStorageBackend ram(1024);
    const auto layout = makeLayout();
    auto handle = ram.allocate(testKey(), layout);
    ASSERT_TRUE(handle.valid());
    (*handle.kv_storage)[0] = 42;

    DiskPrefixStorageBackend disk(dir, 1024);
    std::string error;
    ASSERT_TRUE(disk.writeBlock(handle, &error)) << error;

    const auto meta_path = disk.blockMetaPath(testKey());
    std::ifstream in(meta_path);
    std::string meta((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::string needle = "\"kv_checksum\":";
    const size_t start = meta.find(needle);
    ASSERT_NE(start, std::string::npos);
    size_t value_start = start + needle.size();
    size_t value_end = value_start;
    while (value_end < meta.size() && meta[value_end] >= '0' && meta[value_end] <= '9')
    {
        ++value_end;
    }
    meta.replace(value_start, value_end - value_start, "0");
    std::ofstream out(meta_path, std::ios::trunc);
    out << meta;
    out.close();

    PrefixBlockHandle hydrated;
    EXPECT_FALSE(disk.readBlock(testKey(), layout, &hydrated, &error));
    cleanup();
}
