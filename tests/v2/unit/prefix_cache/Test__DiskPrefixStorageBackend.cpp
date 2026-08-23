#include <gtest/gtest.h>

#include "execution/prefix_cache/DiskPrefixStorageBackend.h"
#include "execution/prefix_cache/RamPrefixStorageBackend.h"
#include "utils/Sha256.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

using namespace llaminar2;

namespace
{
    constexpr const char *kModelArtifactIdentity =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

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

    std::filesystem::path archivePath(const std::filesystem::path &directory)
    {
        return directory /
               (std::string(kModelArtifactIdentity) + ".kvcache");
    }
} // namespace

TEST(Test__DiskPrefixStorageBackend, UsesStableModelArtifactIdentityForArchiveNaming)
{
    const auto dir = tempDir();
    std::filesystem::create_directories(dir);
    const auto model = dir / "model.gguf";
    {
        std::ofstream out(model, std::ios::binary);
        out << "abc";
    }

    std::string error;
    const auto digest = sha256FileSetIdentityHex({model}, &error);
    ASSERT_TRUE(digest.has_value()) << error;
    ASSERT_EQ(digest->size(), 64u);
    EXPECT_EQ(
        (dir / (*digest + ".kvcache")).filename(),
        *digest + ".kvcache");
    std::filesystem::remove_all(dir);
}

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

    DiskPrefixStorageBackend disk(
        archivePath(dir), 1024, kModelArtifactIdentity);
    ASSERT_TRUE(disk.ready()) << disk.initializationError();
    std::string error;
    PrefixBlockHandle disk_handle;
    ASSERT_TRUE(disk.writeBlock(handle, &disk_handle, nullptr, &error)) << error;
    EXPECT_EQ(disk.archivePath(), archivePath(dir));
    EXPECT_TRUE(std::filesystem::is_regular_file(disk.archivePath()));

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

    DiskPrefixStorageBackend disk(
        archivePath(dir), 1024, kModelArtifactIdentity);
    ASSERT_TRUE(disk.ready()) << disk.initializationError();
    std::string error;
    PrefixBlockHandle disk_handle;
    ASSERT_TRUE(disk.writeBlock(handle, &disk_handle, nullptr, &error)) << error;

    PrefixBlockHandle hydrated;
    ASSERT_TRUE(disk.readBlock(testKey(), layout, &hydrated, &error)) << error;
    ASSERT_NE(hydrated.hybrid_storage, nullptr);
    EXPECT_EQ(*hydrated.hybrid_storage, *handle.hybrid_storage);
    EXPECT_TRUE(hydrated.has_hybrid_state);
    EXPECT_TRUE(hydrated.has_terminal_logits);
    cleanup();
}

TEST(Test__DiskPrefixStorageBackend, PreservesModelRuntimeStatePayload)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    RamPrefixStorageBackend ram(1024);
    const auto layout = makeLayout();
    auto handle = ram.allocate(testKey(), layout);
    ASSERT_TRUE(handle.valid());
    handle.model_runtime_state_storage =
        std::make_shared<std::vector<uint8_t>>(
            std::initializer_list<uint8_t>{9, 8, 7, 6, 5});
    handle.has_model_runtime_state = true;
    handle.total_bytes += handle.model_runtime_state_storage->size();

    DiskPrefixStorageBackend disk(
        archivePath(dir), 1024, kModelArtifactIdentity);
    ASSERT_TRUE(disk.ready()) << disk.initializationError();
    std::string error;
    PrefixBlockHandle disk_handle;
    ASSERT_TRUE(disk.writeBlock(handle, &disk_handle, nullptr, &error)) << error;

    PrefixBlockHandle hydrated;
    ASSERT_TRUE(disk.readBlock(testKey(), layout, &hydrated, &error)) << error;
    EXPECT_TRUE(hydrated.has_model_runtime_state);
    ASSERT_NE(hydrated.model_runtime_state_storage, nullptr);
    EXPECT_EQ(
        *hydrated.model_runtime_state_storage,
        *handle.model_runtime_state_storage);
    EXPECT_EQ(hydrated.total_bytes, handle.total_bytes);
    cleanup();
}

TEST(Test__DiskPrefixStorageBackend, RejectsCorruptedPayloadChecksum)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    RamPrefixStorageBackend ram(1024);
    const auto layout = makeLayout();
    auto handle = ram.allocate(testKey(), layout);
    ASSERT_TRUE(handle.valid());
    (*handle.kv_storage)[0] = 42;

    DiskPrefixStorageBackend disk(
        archivePath(dir), 1024, kModelArtifactIdentity);
    ASSERT_TRUE(disk.ready()) << disk.initializationError();
    std::string error;
    PrefixBlockHandle disk_handle;
    ASSERT_TRUE(disk.writeBlock(handle, &disk_handle, nullptr, &error)) << error;

    const auto archive_bytes = std::filesystem::file_size(disk.archivePath());
    ASSERT_GT(archive_bytes, 17u);
    std::fstream archive(
        disk.archivePath(),
        std::ios::binary | std::ios::in | std::ios::out);
    archive.seekg(static_cast<std::streamoff>(archive_bytes - 17));
    char byte = 0;
    archive.read(&byte, 1);
    byte ^= 0x5a;
    archive.seekp(static_cast<std::streamoff>(archive_bytes - 17));
    archive.write(&byte, 1);
    archive.close();

    PrefixBlockHandle hydrated;
    EXPECT_FALSE(disk.readBlock(testKey(), layout, &hydrated, &error));
    cleanup();
}

TEST(Test__DiskPrefixStorageBackend, DiscoversCommittedBlocksAfterRestart)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };
    const auto layout = makeLayout();

    {
        RamPrefixStorageBackend ram(1024);
        auto handle = ram.allocate(testKey(), layout);
        ASSERT_TRUE(handle.valid());
        std::fill(handle.kv_storage->begin(), handle.kv_storage->end(), 0x39);

        DiskPrefixStorageBackend writer(
            archivePath(dir), 1024, kModelArtifactIdentity);
        ASSERT_TRUE(writer.ready()) << writer.initializationError();
        PrefixBlockHandle disk_handle;
        std::string error;
        ASSERT_TRUE(writer.writeBlock(handle, &disk_handle, nullptr, &error)) << error;
    }

    DiskPrefixStorageBackend reopened(
        archivePath(dir), 1024, kModelArtifactIdentity);
    ASSERT_TRUE(reopened.ready()) << reopened.initializationError();
    std::string error;
    const auto entries =
        reopened.compatibleEntries(testKey().fingerprint, layout, &error);
    ASSERT_EQ(entries.size(), 1u) << error;
    EXPECT_EQ(entries.front().key, testKey());

    PrefixBlockHandle hydrated;
    ASSERT_TRUE(reopened.readBlock(testKey(), layout, &hydrated, &error)) << error;
    EXPECT_TRUE(std::all_of(
        hydrated.kv_storage->begin(),
        hydrated.kv_storage->end(),
        [](uint8_t value) { return value == 0x39; }));
    cleanup();
}

TEST(Test__DiskPrefixStorageBackend, TruncatesInterruptedTailWithoutLosingCommittedBlocks)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };
    const auto layout = makeLayout();

    RamPrefixStorageBackend ram(1024);
    auto handle = ram.allocate(testKey(), layout);
    ASSERT_TRUE(handle.valid());

    {
        DiskPrefixStorageBackend writer(
            archivePath(dir), 1024, kModelArtifactIdentity);
        ASSERT_TRUE(writer.ready()) << writer.initializationError();
        PrefixBlockHandle disk_handle;
        std::string error;
        ASSERT_TRUE(writer.writeBlock(handle, &disk_handle, nullptr, &error)) << error;
    }
    const auto committed_bytes = std::filesystem::file_size(archivePath(dir));
    {
        std::ofstream interrupted(
            archivePath(dir),
            std::ios::binary | std::ios::app);
        interrupted << "LLKVR001partial";
    }
    ASSERT_GT(std::filesystem::file_size(archivePath(dir)), committed_bytes);

    DiskPrefixStorageBackend reopened(
        archivePath(dir), 1024, kModelArtifactIdentity);
    ASSERT_TRUE(reopened.ready()) << reopened.initializationError();
    EXPECT_EQ(std::filesystem::file_size(archivePath(dir)), committed_bytes);
    PrefixBlockHandle hydrated;
    std::string error;
    EXPECT_TRUE(reopened.readBlock(testKey(), layout, &hydrated, &error)) << error;
    cleanup();
}

TEST(Test__DiskPrefixStorageBackend, EvictsOldestActiveRecordAtBudget)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    auto layout = makeLayout();
    layout.includes_terminal_logits = false;
    layout.terminal_logits_bytes = 0;
    const size_t block_bytes = layout.totalBytes();
    ASSERT_GT(block_bytes, 0u);

    RamPrefixStorageBackend ram(block_bytes * 3);
    DiskPrefixStorageBackend disk(
        archivePath(dir),
        block_bytes * 2,
        kModelArtifactIdentity);
    ASSERT_TRUE(disk.ready()) << disk.initializationError();

    std::vector<PrefixCacheKey> keys;
    std::string error;
    for (int block = 0; block < 3; ++block)
    {
        const auto key = makePrefixCacheKey(
            testKey().fingerprint,
            0,
            block,
            block,
            {block + 1});
        keys.push_back(key);
        auto handle = ram.allocate(key, layout);
        ASSERT_TRUE(handle.valid());
        PrefixBlockHandle disk_handle;
        std::vector<PrefixCacheKey> evicted;
        ASSERT_TRUE(disk.writeBlock(
            handle,
            &disk_handle,
            &evicted,
            &error))
            << error;
        if (block == 2)
        {
            ASSERT_EQ(evicted.size(), 1u);
            EXPECT_EQ(evicted.front(), keys.front());
        }
    }

    EXPECT_LE(disk.usedBytes(), block_bytes * 2);
    PrefixBlockHandle hydrated;
    EXPECT_FALSE(disk.readBlock(keys.front(), layout, &hydrated, &error));
    EXPECT_TRUE(disk.readBlock(keys[1], layout, &hydrated, &error)) << error;
    cleanup();
}

TEST(Test__DiskPrefixStorageBackend, AccessedRecordSurvivesBudgetEvictionAndRestart)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    auto layout = makeLayout();
    layout.includes_terminal_logits = false;
    layout.terminal_logits_bytes = 0;
    const size_t block_bytes = layout.totalBytes();
    ASSERT_GT(block_bytes, 0u);

    const auto make_key =
        [&](int block)
    {
        return makePrefixCacheKey(
            testKey().fingerprint,
            0,
            block,
            block,
            {100 + block});
    };
    const PrefixCacheKey a_key = make_key(0);
    const PrefixCacheKey b_key = make_key(1);
    const PrefixCacheKey c_key = make_key(2);

    {
        RamPrefixStorageBackend ram(block_bytes * 3);
        DiskPrefixStorageBackend disk(
            archivePath(dir),
            block_bytes * 2,
            kModelArtifactIdentity);
        ASSERT_TRUE(disk.ready()) << disk.initializationError();

        std::string error;
        for (const PrefixCacheKey &key : {a_key, b_key})
        {
            auto handle = ram.allocate(key, layout);
            ASSERT_TRUE(handle.valid());
            PrefixBlockHandle disk_handle;
            ASSERT_TRUE(disk.writeBlock(
                handle,
                &disk_handle,
                nullptr,
                &error))
                << error;
        }

        PrefixBlockHandle touched;
        ASSERT_TRUE(disk.readBlock(
            a_key,
            layout,
            &touched,
            &error))
            << error;

        auto c = ram.allocate(c_key, layout);
        ASSERT_TRUE(c.valid());
        PrefixBlockHandle c_disk;
        std::vector<PrefixCacheKey> evicted;
        ASSERT_TRUE(disk.writeBlock(
            c,
            &c_disk,
            &evicted,
            &error))
            << error;
        ASSERT_EQ(evicted.size(), 1u);
        EXPECT_EQ(evicted.front(), b_key)
            << "A's payload-free touch must make untouched B the oldest record";
    }

    /*
     * Reopening forces recency to be reconstructed from committed archive
     * records. This catches implementations that update only an in-memory LRU.
     */
    DiskPrefixStorageBackend reopened(
        archivePath(dir),
        block_bytes * 2,
        kModelArtifactIdentity);
    ASSERT_TRUE(reopened.ready()) << reopened.initializationError();
    std::string error;
    PrefixBlockHandle hydrated;
    EXPECT_TRUE(reopened.readBlock(
        a_key,
        layout,
        &hydrated,
        &error))
        << error;
    EXPECT_FALSE(reopened.readBlock(
        b_key,
        layout,
        &hydrated,
        &error));
    EXPECT_TRUE(reopened.readBlock(
        c_key,
        layout,
        &hydrated,
        &error))
        << error;
    cleanup();
}
