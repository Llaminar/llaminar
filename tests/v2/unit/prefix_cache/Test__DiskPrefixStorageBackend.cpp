/**
 * @file Test__DiskPrefixStorageBackend.cpp
 * @brief Device-free durability and online prefix-archive lifecycle regressions.
 *
 * Exercises the production archive worker, concurrent appends, restart-visible
 * LRU, checksum rejection and retained-inode hydration. Maintenance must never
 * require inference to rewrite old payloads or abandon a verified restore.
 */

#include <gtest/gtest.h>

#include "execution/prefix_cache/DiskPrefixStorageBackend.h"
#include "execution/prefix_cache/PrefixArchiveIOGeometry.h"
#include "execution/prefix_cache/RamPrefixStorageBackend.h"
#include "utils/Sha256.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <future>
#include <fstream>
#include <string>
#include <thread>

using namespace llaminar2;

namespace
{
    constexpr const char *kModelArtifactIdentity =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    /** @brief Small valid attention block, with a terminal payload checksum. */
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

    /** @return Stable key independent of hardware, model files or timing. */
    PrefixCacheKey testKey()
    {
        return makePrefixCacheKey(0x12345678, 0, 0, 0, {1, 2});
    }

    /** @return One uniquely owned temporary directory for this test lifetime. */
    std::filesystem::path tempDir()
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::filesystem::temp_directory_path() / ("llaminar_prefix_disk_" + std::to_string(stamp));
    }

    /** @return Model-addressed archive within the exact owned temporary root. */
    std::filesystem::path archivePath(const std::filesystem::path &directory)
    {
        return directory /
               (std::string(kModelArtifactIdentity) + ".kvcache");
    }

    /** @brief Admit exactly one production archive scratch on rank zero. */
    std::shared_ptr<PhysicalMemoryAuthority> makeArchiveAuthority()
    {
        constexpr size_t kHostBytes = 64u * 1024u * 1024u;
        PhysicalMemoryPlanBuilder builder;
        builder.add(
            PhysicalMemoryResource{
                .world_rank = 0,
                .device = DeviceId::cpu(),
                .total_bytes = kHostBytes,
                .admission_available_bytes = kHostBytes,
            },
            PhysicalMemoryOwner::PrefixArchiveStaging,
            PrefixArchiveIOGeometry::scratchBytes());
        auto admission = std::make_shared<
            const PhysicalMemoryPlanAdmissionCertificate>(builder.build());
        return std::make_shared<PhysicalMemoryAuthority>(
            std::move(admission), 0);
    }
} // namespace

TEST(Test__DiskPrefixStorageBackend,
     SharedProductionArchiveOwnsExactlyOneCanonicalScratchClaim)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };
    auto authority = makeArchiveAuthority();
    std::string error;

    auto first = DiskPrefixStorageBackend::openShared(
        archivePath(dir),
        1024u,
        kModelArtifactIdentity,
        authority,
        &error);
    ASSERT_NE(first, nullptr) << error;
    EXPECT_EQ(
        authority->claimedBytes(
            DeviceId::cpu(),
            PhysicalMemoryOwner::PrefixArchiveStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),
        PrefixArchiveIOGeometry::scratchBytes());

    auto second = DiskPrefixStorageBackend::openShared(
        archivePath(dir),
        1024u,
        kModelArtifactIdentity,
        authority,
        &error);
    ASSERT_EQ(second.get(), first.get());
    EXPECT_EQ(
        authority->claimedBytes(
            DeviceId::cpu(),
            PhysicalMemoryOwner::PrefixArchiveStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),
        PrefixArchiveIOGeometry::scratchBytes())
        << "LocalTP users must share the one admitted archive I/O buffer.";

    second.reset();
    first.reset();
    EXPECT_EQ(
        authority->claimedBytes(
            DeviceId::cpu(),
            PhysicalMemoryOwner::PrefixArchiveStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),
        0u);
    cleanup();
}

/** @test Explicit maintenance executes off the caller and preserves restart LRU. */
TEST(Test__DiskPrefixStorageBackend, BackgroundCompactionPreservesCommittedLRU)
{
    const auto dir = tempDir();
    auto layout = makeLayout();
    const auto a = testKey();
    const auto b = makePrefixCacheKey(a.fingerprint, 0, 1, 2, {3, 4});
    const auto c = makePrefixCacheKey(a.fingerprint, 0, 2, 4, {5, 6});
    const size_t budget = 2u * layout.totalBytes();
    {
        RamPrefixStorageBackend ram(3u * layout.totalBytes());
        DiskPrefixStorageBackend disk(archivePath(dir), budget, kModelArtifactIdentity);
        std::string error;
        for (const auto &key : {a, b})
        {
            auto handle = ram.allocate(key, layout);
            ASSERT_TRUE(handle.valid());
            std::fill(handle.kv_storage->begin(), handle.kv_storage->end(),
                      key == a ? 0x31 : 0x42);
            ASSERT_TRUE(disk.writeBlock(handle, nullptr, nullptr, &error)) << error;
        }
        PrefixBlockHandle hydrated;
        ASSERT_TRUE(disk.readBlock(a, layout, &hydrated, &error)) << error;
        ASSERT_TRUE(disk.requestCompaction(&error)) << error;
        ASSERT_TRUE(disk.waitForCompaction(&error)) << error;
        const auto status = disk.compactionStatus();
        EXPECT_EQ(status.state, DiskPrefixStorageBackend::CompactionState::Idle);
        EXPECT_EQ(status.publications, 1u);
        EXPECT_NE(status.last_executor, std::this_thread::get_id())
            << "Live inference must not execute archive payload maintenance";
    }
    {
        DiskPrefixStorageBackend reopened(archivePath(dir), budget, kModelArtifactIdentity);
        RamPrefixStorageBackend ram(layout.totalBytes());
        auto handle = ram.allocate(c, layout);
        std::vector<PrefixCacheKey> evicted;
        std::string error;
        ASSERT_TRUE(reopened.writeBlock(handle, nullptr, &evicted, &error)) << error;
        ASSERT_EQ(evicted.size(), 1u);
        EXPECT_EQ(evicted.front(), b) << "Compaction must preserve A's durable touch";
        PrefixBlockHandle hydrated;
        ASSERT_TRUE(reopened.readBlock(a, layout, &hydrated, &error)) << error;
        EXPECT_TRUE(std::all_of(hydrated.kv_storage->begin(), hydrated.kv_storage->end(),
                               [](uint8_t byte) { return byte == 0x31; }));
    }
    std::filesystem::remove_all(dir);
}

/** @test A verified, logically evicted record survives another backend's rename. */
TEST(Test__DiskPrefixStorageBackend, VerifiedHydrationSurvivesBackgroundInodeReplacement)
{
    const auto dir = tempDir();
    auto layout = makeLayout();
    const auto key = testKey();
    {
        auto reader = std::make_shared<DiskPrefixStorageBackend>(
            archivePath(dir), layout.totalBytes(), kModelArtifactIdentity);
        RamPrefixStorageBackend ram(layout.totalBytes());
        auto original = ram.allocate(key, layout);
        std::fill(original.kv_storage->begin(), original.kv_storage->end(), 0xab);
        std::fill(original.terminal_logits_storage->begin(),
                  original.terminal_logits_storage->end(), 0xcd);
        std::string error;
        ASSERT_TRUE(reader->writeBlock(original, nullptr, nullptr, &error)) << error;
        auto ticket = reader->beginVerifiedHydration(key, layout, &error);
        ASSERT_TRUE(ticket.has_value()) << error;

        DiskPrefixStorageBackend peer(archivePath(dir), layout.totalBytes(), kModelArtifactIdentity);
        auto replacement = original;
        replacement.key = makePrefixCacheKey(key.fingerprint, 0, 1, 2, {9, 10});
        ASSERT_TRUE(peer.writeBlock(replacement, nullptr, nullptr, &error)) << error;
        ASSERT_TRUE(peer.requestCompaction(&error)) << error;
        ASSERT_TRUE(peer.waitForCompaction(&error)) << error;
        EXPECT_EQ(peer.compactionStatus().publications, 1u);

        // Drop the RAM owner before filling final storage: the old inode, not
        // a second payload copy, is the ticket's only surviving byte authority.
        ram.release(original);
        original = {};
        replacement = {};
        PrefixBlockHandle restored;
        ASSERT_TRUE(reader->hydrateVerified(*ticket, ram, &restored, &error)) << error;
        EXPECT_EQ(restored.key, key);
        EXPECT_TRUE(std::all_of(restored.kv_storage->begin(), restored.kv_storage->end(),
                               [](uint8_t byte) { return byte == 0xab; }));
        EXPECT_TRUE(std::all_of(restored.terminal_logits_storage->begin(),
                               restored.terminal_logits_storage->end(),
                               [](uint8_t byte) { return byte == 0xcd; }));
        EXPECT_FALSE(reader->hydrateVerified(*ticket, ram, &restored, &error));
    }
    std::filesystem::remove_all(dir);
}

/** @test Concurrent writers/touches/deletes survive the worker's exact tail frontier. */
TEST(Test__DiskPrefixStorageBackend, BackgroundCompactionPreservesConcurrentMutationTail)
{
    const auto dir = tempDir();
    const auto layout = makeLayout();
    std::vector<PrefixCacheKey> expected;
    {
        DiskPrefixStorageBackend disk(archivePath(dir), 0u, kModelArtifactIdentity);
        DiskPrefixStorageBackend peer(archivePath(dir), 0u, kModelArtifactIdentity);
        auto write = [&](DiskPrefixStorageBackend &backend, int begin)
        {
            RamPrefixStorageBackend ram(layout.totalBytes());
            for (int index = begin; index < begin + 40; ++index)
            {
                const auto key = makePrefixCacheKey(testKey().fingerprint, 0,
                    index, 2 * index, {2 * index + 1, 2 * index + 2});
                auto handle = ram.allocate(key, layout);
                if (!handle.valid())
                    return std::string("writer could not allocate its one block");
                std::fill(handle.kv_storage->begin(), handle.kv_storage->end(),
                          static_cast<uint8_t>(index));
                std::string error;
                if (!backend.writeBlock(handle, nullptr, nullptr, &error))
                    return error;
                if (index % 3 == 0 && !backend.release(handle))
                    return std::string("concurrent delete failed");
                ram.release(handle);
                handle = {};
                if (index % 4 == 0 && !backend.requestCompaction(&error))
                    return error;
            }
            return std::string{};
        };
        auto first = std::async(std::launch::async, [&] { return write(disk, 0); });
        auto second = std::async(std::launch::async, [&] { return write(peer, 40); });
        EXPECT_EQ(first.get(), "");
        EXPECT_EQ(second.get(), "");
        std::string error;
        ASSERT_TRUE(disk.waitForCompaction(&error)) << error;
        ASSERT_TRUE(peer.waitForCompaction(&error)) << error;
        ASSERT_TRUE(disk.requestCompaction(&error)) << error;
        ASSERT_TRUE(disk.waitForCompaction(&error)) << error;
        EXPECT_GT(disk.compactionStatus().publications, 0u);
        for (int index = 0; index < 80; ++index)
            if (index % 3 != 0)
                expected.push_back(makePrefixCacheKey(testKey().fingerprint, 0,
                    index, 2 * index, {2 * index + 1, 2 * index + 2}));
    }
    {
        DiskPrefixStorageBackend reopened(archivePath(dir), 0u, kModelArtifactIdentity);
        std::string error;
        const auto entries = reopened.compatibleEntries(testKey().fingerprint, layout, &error);
        ASSERT_EQ(entries.size(), expected.size()) << error;
        for (const auto &key : expected)
        {
            PrefixBlockHandle restored;
            ASSERT_TRUE(reopened.readBlock(key, layout, &restored, &error)) << error;
            EXPECT_TRUE(std::all_of(restored.kv_storage->begin(), restored.kv_storage->end(),
                [&](uint8_t byte) { return byte == static_cast<uint8_t>(key.block_index); }));
        }
    }
    std::filesystem::remove_all(dir);
}

/** @test Real stale payload volume schedules automatic off-request reclamation. */
TEST(Test__DiskPrefixStorageBackend, BackgroundCompactionReclaimsAutomaticCapacityHistory)
{
    const auto dir = tempDir();
    {
        auto layout = makeLayout();
        layout.bytes_per_fa_layer_k = 4u * 1024u * 1024u;
        layout.bytes_per_fa_layer_v = 4u * 1024u * 1024u;
        layout.includes_terminal_logits = false;
        layout.terminal_logits_bytes = 0u;
        const auto bytes = layout.totalBytes();
        RamPrefixStorageBackend ram(bytes);
        auto handle = ram.allocate(testKey(), layout);
        ASSERT_TRUE(handle.valid());
        DiskPrefixStorageBackend disk(archivePath(dir), bytes, kModelArtifactIdentity);
        std::string error;
        for (uint8_t version = 0; version < 9; ++version)
        {
            std::fill(handle.kv_storage->begin(), handle.kv_storage->end(), version);
            ASSERT_TRUE(disk.writeBlock(handle, nullptr, nullptr, &error)) << error;
        }
        ASSERT_TRUE(disk.waitForCompaction(&error)) << error;
        const auto status = disk.compactionStatus();
        EXPECT_GT(status.publications, 0u);
        EXPECT_NE(status.last_executor, std::this_thread::get_id());
        // Correct restored bytes alone cannot prove economical maintenance.
        // Check actual descriptor flags as well as completed writes. Successful
        // range-flush calls can do no work on a container OverlayFS mapping;
        // the native O_DSYNC contract must be present before any bulk copy.
        const auto &writeback = status.last_writeback;
        EXPECT_EQ(writeback.native_open_flags & O_DSYNC, O_DSYNC);
        EXPECT_GT(writeback.durable_writes, 0u);
        EXPECT_GE(writeback.durable_bytes, bytes);
        EXPECT_LE(writeback.largest_write_bytes,
                  PrefixArchiveIOGeometry::compactionBytes());
        EXPECT_EQ(disk.usedBytes(), bytes);
        EXPECT_LT(std::filesystem::file_size(disk.archivePath()), 4u * bytes);
        PrefixBlockHandle restored;
        ASSERT_TRUE(disk.readBlock(testKey(), layout, &restored, &error)) << error;
        EXPECT_TRUE(std::all_of(restored.kv_storage->begin(), restored.kv_storage->end(),
                               [](uint8_t byte) { return byte == 8u; }));
    }
    std::filesystem::remove_all(dir);
}

/** @test Background failures are retained and reject further archive operations. */
TEST(Test__DiskPrefixStorageBackend, BackgroundCompactionFailureInvalidatesArchive)
{
    const auto dir = tempDir();
    {
        DiskPrefixStorageBackend disk(archivePath(dir), 1024u, kModelArtifactIdentity);
        // A directory at this exact, worker-owned output name deterministically
        // rejects native open without corrupting the committed source archive.
        std::filesystem::create_directory(
            disk.archivePath().string() + ".compact." + std::to_string(::getpid()));
        std::string error;
        ASSERT_TRUE(disk.requestCompaction(&error)) << error;
        EXPECT_FALSE(disk.waitForCompaction(&error));
        EXPECT_FALSE(disk.ready());
        EXPECT_EQ(disk.compactionStatus().state,
                  DiskPrefixStorageBackend::CompactionState::Failed);
        EXPECT_NE(error.find("failed to create compact prefix archive"), std::string::npos);
        EXPECT_FALSE(disk.requestCompaction(&error));
        RamPrefixStorageBackend ram(1024u);
        auto handle = ram.allocate(testKey(), makeLayout());
        EXPECT_FALSE(disk.writeBlock(handle, nullptr, nullptr, &error));
    }
    std::filesystem::remove_all(dir);
}

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
    ASSERT_TRUE(ram.attachModelRuntimeState(&handle,
        std::make_shared<std::vector<uint8_t>>(
            std::initializer_list<uint8_t>{9, 8, 7, 6, 5})));

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
    EXPECT_TRUE(std::ranges::equal(hydrated.model_runtime_state_storage->bytes(),
                                  handle.model_runtime_state_storage->bytes()));
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
