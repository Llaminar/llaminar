/**
 * @file Test__PrefixCacheShutdown.cpp
 * @brief Exercise terminal RAM priority, shared capacity and restart durability.
 *
 * Small real RAM/disk tiers prove selection before I/O, existing-backing reuse,
 * stale incarnation eviction, lifecycle sealing and failed native publication.
 * No model or accelerator is loaded. Multi-participant fixtures use independent
 * runtime fingerprints and the same archive just like local TP/PP owners.
 */
#include "execution/prefix_cache/PrefixCacheShutdown.h"
#include "execution/prefix_cache/PrefixStateCache.h"
#include "execution/prefix_cache/RamPrefixStorageBackend.h"
#include "execution/runner/OrchestrationRunner.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <unistd.h>

using namespace llaminar2;

namespace
{
    constexpr auto kIdentity = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    /** @brief Minimal device-free participant carrying a real prefix cache through native shutdown. */
    class CacheRunner final : public IInferenceRunner
    {
    public:
        /** @brief Retain the original cache and a test-owned lifecycle observation. */
        CacheRunner(std::shared_ptr<PrefixStateCache> cache, int &collections)
            : cache_(std::move(cache)), collections_(collections) {}
        /** @brief This fixture exercises lifecycle only; no model forward is admitted. */
        bool forward(const int *, int) override { throw std::logic_error("unexpected fixture inference"); }
        /** @return No logits are owned by this lifecycle-only fixture. */
        const float *logits() const override { return nullptr; }
        /** @return Minimal valid vocabulary geometry for runner construction. */
        int vocab_size() const override { return 1; }
        /** @brief No live device sequence exists in this fixture. */
        void clear_cache() override {}
        /** @brief A real terminal reset must name a complete lifecycle boundary. */
        void resetInferenceState(const InferenceStateResetRequest &request) override
        {
            if (request.boundary != InferenceStateResetRequest::Boundary::Request ||
                !request.resetsAllLiveRequestOwners() || !request.reset_model_runtime ||
                !request.preserve_replay_safe_graphs)
                throw std::logic_error("invalid fixture reset");
        }
        /** @return The idle sequence frontier. */
        int get_position() const override { return 0; }
        /** @return The graph runner's declared interface role. */
        ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
        /** @return Stable fixture identity. */
        const char *architecture() const override { return "shutdown-fixture"; }
        /** @brief Forward the actual cache at the production terminal boundary. */
        void collectPrefixCacheShutdown(PrefixCacheShutdown &shutdown) const override
        {
            ++collections_;
            shutdown.add(cache_);
        }
    private:
        std::shared_ptr<PrefixStateCache> cache_;
        int &collections_;
    };

    /** @brief Isolate archive writers and remove files after their last owner retires. */
    class ShutdownFixture : public testing::Test
    {
    protected:
        std::filesystem::path directory = std::filesystem::temp_directory_path() /
            ("llaminar-prefix-shutdown-" + std::to_string(::getpid()) + "-" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        /** @brief Keep all archive owners inside test bodies, before directory retirement. */
        void TearDown() override { std::filesystem::remove_all(directory); }
        /** @return A real bounded archive; direct construction is the device-free test boundary. */
        std::shared_ptr<DiskPrefixStorageBackend> disk(size_t budget, std::string suffix = "")
        {
            return std::make_shared<DiskPrefixStorageBackend>(
                directory / suffix / (std::string(kIdentity) + ".kvcache"), budget, kIdentity);
        }
        /** @return One exact serialized KV section, without optional model state. */
        static PrefixPayloadLayout layout(size_t bytes = 16)
        {
            PrefixPayloadLayout result;
            result.block_size = 1;
            result.fa_layers = result.total_layers = 1;
            result.bytes_per_fa_layer_k = bytes / 2;
            result.bytes_per_fa_layer_v = bytes - bytes / 2;
            return result;
        }
        /** @return Distinct token keys, optionally in different participant namespaces. */
        static PrefixCacheKey key(int token, uint64_t fingerprint = 0x1234)
        {
            return makePrefixCacheKey(fingerprint, 0, 0, 0, {token});
        }
        /** @return An initialized immutable RAM owner with observable restored bytes. */
        static PrefixBlockHandle block(RamPrefixStorageBackend &ram, int token, size_t bytes = 16,
                                       uint64_t fingerprint = 0x1234)
        {
            auto result = ram.allocate(key(token, fingerprint), layout(bytes));
            if (!result.valid()) throw std::runtime_error("fixture RAM allocation failed");
            std::memset(result.kv_payload, token, result.total_bytes);
            return result;
        }
        /** @return Durable token identities, discovered using metadata only. */
        static std::set<uint64_t> tokens(DiskPrefixStorageBackend &disk, uint64_t fingerprint = 0x1234)
        {
            std::string error;
            std::set<uint64_t> result;
            for (const auto &entry : disk.compatibleEntries(fingerprint, layout(), &error))
                result.insert(entry.key.token_hash);
            if (!error.empty()) throw std::runtime_error(error);
            return result;
        }
    };
}

/** @test Recently used RAM owners displace even more recently touched disk records. */
TEST_F(ShutdownFixture, RamPriorityAndRecencySurviveRestart)
{
    {
        auto archive = disk(32);
        ASSERT_TRUE(archive->ready());
        auto ram = std::make_shared<RamPrefixStorageBackend>(256);
        auto cache = std::make_shared<PrefixStateCache>(256, ram, archive);
        for (int token : {1, 2, 3}) ASSERT_TRUE(cache->insert(block(*ram, token)));
        ASSERT_TRUE(cache->find(key(1))); // RAM order is now 1, 3, 2.
        for (int token : {8, 9}) ASSERT_TRUE(archive->writeBlock(block(*ram, token), nullptr));
        PrefixCacheShutdown flush;
        flush.add(cache);
        flush.add(cache); // Aliases in a composite tree must not duplicate writes.
        const auto result = flush.flush();
        EXPECT_EQ(result.candidates, 3u);
        EXPECT_EQ(result.selected, 2u);
        EXPECT_EQ(result.written, 2u);
        EXPECT_EQ(result.payload_bytes_written, 32u);
        EXPECT_EQ(archive->usedBytes(), 32u);
        EXPECT_EQ(tokens(*archive), (std::set<uint64_t>{key(1).token_hash, key(3).token_hash}));
        EXPECT_THROW(cache->find(key(1)), std::logic_error);
        EXPECT_THROW(cache->beginLookup(), std::logic_error);
        EXPECT_THROW(cache->erase(key(1)), std::logic_error);
        EXPECT_THROW(flush.flush(), std::logic_error);
        EXPECT_THROW(flush.add(cache), std::logic_error);
    }
    auto restarted = disk(32);
    ASSERT_TRUE(restarted->ready());
    PrefixBlockHandle restored;
    ASSERT_TRUE(restarted->readBlock(key(1), layout(), &restored));
    EXPECT_EQ(std::vector<uint8_t>(static_cast<uint8_t *>(restored.kv_payload),
        static_cast<uint8_t *>(restored.kv_payload) + 16), std::vector<uint8_t>(16, 1));
    EXPECT_EQ(tokens(*restarted), (std::set<uint64_t>{key(1).token_hash, key(3).token_hash}));
    EXPECT_TRUE(tokens(*restarted, 0x5678).empty());
    auto incompatible = layout();
    incompatible.block_size = 2;
    EXPECT_TRUE(restarted->compatibleEntries(0x1234, incompatible).empty());
}

/** @test Flush consumes spare capacity before evicting cold LRU entries. */
TEST_F(ShutdownFixture, SpareCapacityKeepsColdEntriesAndPreservesRamRetentionOrder)
{
    auto archive = disk(64);
    auto ram = std::make_shared<RamPrefixStorageBackend>(256);
    for (int token : {7, 8, 9}) ASSERT_TRUE(archive->writeBlock(block(*ram, token), nullptr));
    auto cache = std::make_shared<PrefixStateCache>(256, ram, archive);
    for (int token : {1, 2}) ASSERT_TRUE(cache->insert(block(*ram, token)));
    PrefixCacheShutdown flush;
    flush.add(cache);
    EXPECT_EQ(flush.flush().written, 2u);
    EXPECT_EQ(tokens(*archive), (std::set<uint64_t>{key(1).token_hash, key(2).token_hash,
                                                   key(8).token_hash, key(9).token_hash}));
    ASSERT_TRUE(archive->writeBlock(block(*ram, 10), nullptr));
    ASSERT_TRUE(archive->writeBlock(block(*ram, 11), nullptr));
    EXPECT_EQ(tokens(*archive), (std::set<uint64_t>{key(1).token_hash, key(2).token_hash,
                                                   key(10).token_hash, key(11).token_hash}));
}

/** @test Identical existing backing produces only a metadata touch. */
TEST_F(ShutdownFixture, ExistingBackingIsReusedWithoutAnotherPayloadFile)
{
    auto archive = disk(32);
    auto ram = std::make_shared<RamPrefixStorageBackend>(32);
    auto cache = std::make_shared<PrefixStateCache>(32, ram, archive);
    auto handle = block(*ram, 1);
    ASSERT_TRUE(cache->insert(handle));
    ASSERT_TRUE(archive->writeBlock(handle, nullptr));
    std::set<std::filesystem::path> before;
    for (const auto &entry : std::filesystem::directory_iterator(archive->archivePath().string() + ".blocks"))
        before.insert(entry.path());
    PrefixCacheShutdown flush;
    flush.add(cache);
    const auto result = flush.flush();
    EXPECT_EQ(result.selected, 1u);
    EXPECT_EQ(result.reused, 1u);
    EXPECT_EQ(result.written, 0u);
    EXPECT_EQ(result.payload_bytes_written, 0u);
    std::set<std::filesystem::path> after;
    for (const auto &entry : std::filesystem::directory_iterator(archive->archivePath().string() + ".blocks"))
        after.insert(entry.path());
    EXPECT_EQ(before, after);
}

/** @test A later participant reuses selected old backing after an earlier peer makes space. */
TEST_F(ShutdownFixture, SelectedBackingSurvivesIndependentWriterOrder)
{
    auto first = disk(32);
    auto peer = disk(32);
    auto ram = std::make_shared<RamPrefixStorageBackend>(128);
    const auto a = block(*ram, 1);
    const auto b = block(*ram, 2);
    ASSERT_TRUE(first->writeBlock(a, nullptr));
    ASSERT_TRUE(first->writeBlock(block(*ram, 9), nullptr)); // Cold but newer than A.
    const PrefixShutdownArchivePlan plan{.archive = first->shutdownIdentity(), .budget_bytes = 32,
        .retained = {{a.key, a.payload_identity, a.total_bytes}, {b.key, b.payload_identity, b.total_bytes}}};
    const std::vector<PrefixBlockHandle> peer_owners{b};
    EXPECT_EQ(peer->flushShutdown(peer_owners, plan).written, 1u);
    const std::vector<PrefixBlockHandle> first_owners{a};
    const auto reused = first->flushShutdown(first_owners, plan);
    EXPECT_EQ(reused.written, 0u);
    EXPECT_EQ(reused.reused, 1u);
    EXPECT_EQ(tokens(*first), (std::set<uint64_t>{a.key.token_hash, b.key.token_hash}));
}

/** @test Invalid selections fail before changing cold archive contents. */
TEST_F(ShutdownFixture, WriterRejectsWrongArchiveCapacityAndUnselectedOwners)
{
    auto archive = disk(32);
    auto other = disk(32, "other");
    auto ram = std::make_shared<RamPrefixStorageBackend>(128);
    const auto a = block(*ram, 1);
    const auto b = block(*ram, 2);
    ASSERT_TRUE(archive->writeBlock(block(*ram, 9), nullptr));
    PrefixShutdownArchivePlan valid{.archive = archive->shutdownIdentity(), .budget_bytes = 32,
        .retained = {{a.key, a.payload_identity, a.total_bytes}}};
    const std::vector<PrefixBlockHandle> selected{a};
    auto wrong = valid;
    wrong.archive = other->shutdownIdentity();
    EXPECT_THROW(archive->flushShutdown(selected, wrong), std::invalid_argument);
    wrong = valid;
    wrong.budget_bytes = 64;
    EXPECT_THROW(archive->flushShutdown(selected, wrong), std::invalid_argument);
    wrong = valid;
    wrong.retained.front().bytes = 33;
    EXPECT_THROW(archive->flushShutdown(selected, wrong), std::invalid_argument);
    wrong = valid;
    wrong.retained.push_back(wrong.retained.front());
    EXPECT_THROW(archive->flushShutdown(selected, wrong), std::invalid_argument);
    EXPECT_THROW(archive->flushShutdown(std::vector<PrefixBlockHandle>{b}, valid), std::invalid_argument);
    EXPECT_EQ(tokens(*archive), (std::set<uint64_t>{key(9).token_hash}));
}

/** @test Local TP, PP and mixed participant trees share one bounded selection. */
TEST_F(ShutdownFixture, SharedArchivesInterleaveFourAndEightParticipantPriorities)
{
    for (const size_t participants : {4u, 8u})
    {
        auto archive = disk(participants * 16, std::to_string(participants));
        PrefixCacheShutdown flush;
        for (size_t participant = 0; participant < participants; ++participant)
        {
            auto ram = std::make_shared<RamPrefixStorageBackend>(64);
            auto cache = std::make_shared<PrefixStateCache>(64, ram, archive);
            for (int token : {1, 2, 3})
                ASSERT_TRUE(cache->insert(block(*ram, token, 16, participant + 1)));
            flush.add(cache);
        }
        const auto result = flush.flush();
        EXPECT_EQ(result.candidates, participants * 3);
        EXPECT_EQ(result.written, participants);
        EXPECT_EQ(archive->usedBytes(), participants * 16);
        for (size_t participant = 0; participant < participants; ++participant)
            EXPECT_EQ(tokens(*archive, participant + 1),
                      (std::set<uint64_t>{key(3, participant + 1).token_hash}));
    }
}

/** @test An oversized source is never read or written; a smaller lower-priority owner fits. */
TEST_F(ShutdownFixture, SelectionSkipsOversizedPayloadWithoutInspectingItsBytes)
{
    auto archive = disk(16);
    auto ram = std::make_shared<RamPrefixStorageBackend>(256);
    auto cache = std::make_shared<PrefixStateCache>(256, ram, archive);
    ASSERT_TRUE(cache->insert(block(*ram, 1)));
    auto inaccessible = block(*ram, 2, 64);
    inaccessible.kv_payload = reinterpret_cast<void *>(uintptr_t{1});
    ASSERT_TRUE(cache->insert(inaccessible));
    PrefixCacheShutdown flush;
    flush.add(cache);
    const auto result = flush.flush();
    EXPECT_EQ(result.candidates, 2u);
    EXPECT_EQ(result.written, 1u);
    EXPECT_EQ(tokens(*archive), (std::set<uint64_t>{key(1).token_hash}));
}

/** @test Disk-disabled and empty tiers have a supported, allocation-free terminal policy. */
TEST_F(ShutdownFixture, DisabledDiskAndEmptyCacheAreNoOps)
{
    auto ram = std::make_shared<RamPrefixStorageBackend>(32);
    auto cache = std::make_shared<PrefixStateCache>(32, ram);
    ASSERT_TRUE(cache->insert(block(*ram, 1)));
    PrefixCacheShutdown flush;
    flush.add(cache);
    const auto result = flush.flush();
    EXPECT_EQ(result.candidates, 0u);
    EXPECT_EQ(result.written, 0u);
    EXPECT_FALSE(std::filesystem::exists(directory));
    PrefixCacheShutdown empty;
    EXPECT_EQ(empty.flush().written, 0u);
}

/** @test A pending demotion is joined before selecting still-resident RAM owners. */
TEST_F(ShutdownFixture, PendingDemotionsCompleteBeforeRamSelection)
{
    auto archive = disk(16);
    auto ram = std::make_shared<RamPrefixStorageBackend>(32);
    auto cache = std::make_shared<PrefixStateCache>(32, ram, archive);
    for (int token : {1, 2}) ASSERT_TRUE(cache->insert(block(*ram, token)));
    EXPECT_NE(cache->prepareCapacity(16), PrefixRamInsertPreparation::Error);
    PrefixCacheShutdown flush;
    flush.add(cache);
    const auto result = flush.flush();
    EXPECT_EQ(result.candidates, 1u);
    EXPECT_EQ(result.selected, 1u);
    EXPECT_EQ(tokens(*archive), (std::set<uint64_t>{key(2).token_hash}));
}

/** @test Explicit live lookup leases cannot silently cross the shutdown boundary. */
TEST_F(ShutdownFixture, OutstandingLeaseRejectsShutdownBeforeEviction)
{
    auto archive = disk(16);
    auto ram = std::make_shared<RamPrefixStorageBackend>(32);
    auto cache = std::make_shared<PrefixStateCache>(32, ram, archive);
    ASSERT_TRUE(cache->insert(block(*ram, 1)));
    ASSERT_TRUE(cache->retain(key(1)));
    PrefixCacheShutdown flush;
    flush.add(cache);
    EXPECT_THROW(flush.flush(), std::logic_error);
    EXPECT_EQ(archive->usedBytes(), 0u);
    EXPECT_TRUE(cache->release(key(1)));
}

/** @test Unselected old versions do not pin more space than the chosen replacement. */
TEST_F(ShutdownFixture, StaleLargeIncarnationIsColdDuringSelectedPeerWrite)
{
    auto archive = disk(48);
    auto old_ram = std::make_shared<RamPrefixStorageBackend>(64);
    ASSERT_TRUE(archive->writeBlock(block(*old_ram, 1, 48), nullptr));
    auto ram = std::make_shared<RamPrefixStorageBackend>(64);
    auto cache = std::make_shared<PrefixStateCache>(64, ram, archive);
    for (int token : {2, 1}) ASSERT_TRUE(cache->insert(block(*ram, token)));
    PrefixCacheShutdown flush;
    flush.add(cache);
    EXPECT_EQ(flush.flush().written, 2u);
    EXPECT_EQ(archive->usedBytes(), 32u);
    EXPECT_EQ(tokens(*archive), (std::set<uint64_t>{key(1).token_hash, key(2).token_hash}));
}

/** @test Native persistence failure is fatal, with no fabricated completion or retry. */
TEST_F(ShutdownFixture, FailedNativeWriteCannotBecomeSuccessfulRetirement)
{
    auto archive = disk(32);
    auto ram = std::make_shared<RamPrefixStorageBackend>(32);
    auto cache = std::make_shared<PrefixStateCache>(32, ram, archive);
    ASSERT_TRUE(cache->insert(block(*ram, 1)));
    const auto payloads = archive->archivePath().string() + ".blocks";
    ASSERT_TRUE(std::filesystem::remove(payloads));
    std::ofstream(payloads) << "deliberate non-directory";
    PrefixCacheShutdown flush;
    flush.add(cache);
    EXPECT_THROW(flush.flush(), std::runtime_error);
    EXPECT_FALSE(archive->ready());
    EXPECT_NE(archive->initializationError().find("payload"), std::string::npos);
    EXPECT_THROW(flush.flush(), std::logic_error);
}

/** @test Shared identities survive restart; independent archives retain independent budgets. */
TEST_F(ShutdownFixture, ArchiveIdentityDistinguishesPhysicalArchivesAndRejectsPartialMetadata)
{
    auto first = disk(32, "one");
    auto peer = disk(32, "one");
    auto other = disk(32, "two");
    ASSERT_TRUE(first->ready());
    ASSERT_TRUE(peer->ready());
    ASSERT_TRUE(other->ready());
    EXPECT_EQ(first->shutdownIdentity(), peer->shutdownIdentity());
    EXPECT_NE(first->shutdownIdentity(), other->shutdownIdentity());
    const auto identity = first->shutdownIdentity();
    first.reset();
    peer.reset();
    first = disk(32, "one");
    EXPECT_EQ(first->shutdownIdentity(), identity);
    const auto lock = first->archivePath().string() + ".lock";
    first.reset();
    std::ofstream(lock, std::ios::trunc) << "partial";
    auto broken = disk(32, "one");
    EXPECT_FALSE(broken->ready());
    EXPECT_NE(broken->initializationError().find("identity"), std::string::npos);
}

/** @test Ordinary shutdown, explicit worker retirement and destruction all flush exactly once. */
TEST_F(ShutdownFixture, ProductionRunnerFlushesBeforeDestroyingParticipantsExactlyOnce)
{
    for (const bool explicit_worker_shutdown : {false, true})
    {
        auto archive = disk(32, explicit_worker_shutdown ? "explicit" : "ordinary");
        auto ram = std::make_shared<RamPrefixStorageBackend>(32);
        auto cache = std::make_shared<PrefixStateCache>(32, ram, archive);
        ASSERT_TRUE(cache->insert(block(*ram, 1)));
        OrchestrationConfig config;
        config.device_for_this_rank = GlobalDeviceAddress::cpu();
        config.prefix_cache.enabled = true;
        config.prefix_cache.storage_mode = PrefixCacheStorageMode::Tiered;
        config.prefix_cache.disk_budget_bytes = 32;
        RankExecutionPlan plan;
        plan.primary_device = GlobalDeviceAddress::cpu();
        plan.runtime.prefix_cache = config.prefix_cache;
        int collections = 0;
        {
            OrchestrationRunner runner(config, plan, std::make_unique<CacheRunner>(cache, collections));
            if (explicit_worker_shutdown) runner.shutdownMPIWorkers();
            runner.shutdown();
            EXPECT_EQ(collections, 1);
            EXPECT_EQ(tokens(*archive), (std::set<uint64_t>{key(1).token_hash}));
            const auto journal_bytes = std::filesystem::file_size(archive->archivePath());
            runner.shutdown();
            EXPECT_EQ(std::filesystem::file_size(archive->archivePath()), journal_bytes);
        }
        EXPECT_EQ(collections, 1);
    }
}
