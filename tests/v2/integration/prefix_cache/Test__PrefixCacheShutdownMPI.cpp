/**
 * @file Test__PrefixCacheShutdownMPI.cpp
 * @brief Real-rank terminal prefix persistence and metadata communication economy.
 *
 * Independent process-local archive indexes share real durable storage. Empty
 * participants, replicated identities and disjoint archives prove that one
 * global selection respects each physical archive's capacity. A native MPI
 * observer counts the exact metadata bytes and rejects padded transmission.
 */
#include "execution/prefix_cache/PrefixCacheShutdown.h"
#include "execution/prefix_cache/PrefixStateCache.h"
#include "execution/prefix_cache/RamPrefixStorageBackend.h"
#include "utils/MPIContext.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <map>
#include <vector>
#include <unistd.h>

using namespace llaminar2;

namespace
{
    /** @return This real communicator participant, without hardware discovery. */
    int mpiWorldRank()
    {
        int rank = -1;
        if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
            throw std::runtime_error("fixture rank query failed");
        return rank;
    }
    /** @return Actual admitted MPI cardinality. */
    int mpiWorldSize()
    {
        int size = 0;
        if (MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS)
            throw std::runtime_error("fixture communicator query failed");
        return size;
    }
    constexpr size_t kBlockBytes = 1024 * 1024;
    constexpr auto kModelIdentity = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    /** @brief Delegate every operation to real MPI while observing actual wire extents. */
    class ObservedContext final : public MPIContext
    {
    public:
        /** @brief Bind the test's already initialized communicator without hardware discovery. */
        ObservedContext() : MPIContext(mpiWorldRank(), mpiWorldSize()) {}
        /** @brief Authenticate each body against its immediately preceding exact byte count. */
        void send(const void *data, size_t count, MPI_Datatype type, int destination, int tag) const override
        {
            if (type == MPI_UINT64_T)
            {
                EXPECT_EQ(count, 1u);
                announced_[destination] = *static_cast<const uint64_t *>(data);
                sent_bytes += sizeof(uint64_t);
            }
            else
            {
                EXPECT_EQ(type, MPI_BYTE);
                EXPECT_EQ(count, announced_.at(destination));
                sent_bytes += count;
                ++bodies;
                metadata_bodies.emplace_back(static_cast<const char *>(data), count);
            }
            MPIContext::send(data, count, type, destination, tag);
        }
        mutable size_t sent_bytes = 0;
        mutable size_t bodies = 0;
        mutable std::vector<std::string> metadata_bodies; ///< Exact native sends, without capacity tails.
    private:
        mutable std::map<int, uint64_t> announced_;
    };

    /** @return One shared temporary path, broadcasting only its actual characters. */
    std::filesystem::path sharedDirectory()
    {
        std::string path;
        if (mpiWorldRank() == 0)
            path = (std::filesystem::temp_directory_path() /
                ("llaminar-shutdown-mpi-" + std::to_string(::getpid()) + "-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))).string();
        int bytes = static_cast<int>(path.size());
        MPI_Bcast(&bytes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        path.resize(static_cast<size_t>(bytes));
        MPI_Bcast(path.data(), bytes, MPI_BYTE, 0, MPI_COMM_WORLD);
        return path;
    }

    /** @return A native serialized section with an observable rank-specific payload. */
    PrefixPayloadLayout layout()
    {
        PrefixPayloadLayout result;
        result.block_size = result.fa_layers = result.total_layers = 1;
        result.bytes_per_fa_layer_k = result.bytes_per_fa_layer_v = kBlockBytes / 2;
        return result;
    }

    /**
     * @brief Authenticate every observed body as compact schema-only metadata.
     * @param context Native sender whose exact byte ranges were retained by the observer.
     * @return Candidate entries announced by a follower; zero for planner selections.
     *
     * Checks never throw before MPI progress: malformed metadata records a test
     * failure and leaves the native send/receive protocol able to retire.
     */
    size_t expectMetadataOnly(const ObservedContext &context)
    {
        size_t bytes = context.bodies * sizeof(uint64_t);
        size_t candidates = 0;
        EXPECT_EQ(context.metadata_bodies.size(), context.bodies);
        for (const auto &body : context.metadata_bodies)
        {
            bytes += body.size();
            const auto value = nlohmann::json::parse(body, nullptr, false);
            EXPECT_FALSE(value.is_discarded());
            if (value.is_discarded()) continue;
            EXPECT_TRUE(value.is_array());
            if (!value.is_array()) continue;
            EXPECT_EQ(value.dump(), body) << "The wire body must contain no whitespace or capacity padding";
            for (const auto &row : value)
            {
                if (row.is_null()) continue;
                EXPECT_TRUE(row.is_object());
                if (!row.is_object()) continue;
                const bool manifest = row.contains("entries");
                EXPECT_EQ(row.size(), manifest ? 3u : 4u);
                EXPECT_TRUE(row.contains("archive"));
                EXPECT_TRUE(row.contains("budget"));
                const char *blocks = manifest ? "entries" : "retained";
                EXPECT_TRUE(row.contains(blocks));
                if (!row.contains(blocks)) continue;
                EXPECT_TRUE(row.at(blocks).is_array());
                if (!row.at(blocks).is_array()) continue;
                if (manifest) candidates += row.at(blocks).size();
                else
                {
                    EXPECT_TRUE(row.contains("owned"));
                    if (row.contains("owned"))
                        for (const auto &owner : row.at("owned"))
                            EXPECT_EQ(owner.size(), 2u);
                }
                for (const auto &block : row.at(blocks))
                {
                    EXPECT_TRUE(block.is_object());
                    if (!block.is_object()) continue;
                    EXPECT_EQ(block.size(), 3u) << "Only key, immutable identity and live bytes belong on the wire";
                    EXPECT_TRUE(block.contains("key"));
                    EXPECT_TRUE(block.contains("identity"));
                    EXPECT_TRUE(block.contains("bytes"));
                    if (block.contains("key")) EXPECT_EQ(block.at("key").size(), 6u);
                    if (block.contains("identity")) EXPECT_EQ(block.at("identity").size(), 2u);
                    if (block.contains("bytes")) EXPECT_TRUE(block.at("bytes").is_number_unsigned());
                }
            }
        }
        EXPECT_EQ(context.sent_bytes, bytes) << "Native communication extent must equal metadata plus framing";
        EXPECT_LT(context.sent_bytes, kBlockBytes / 16) << "KV payload bytes must remain on their owning rank";
        return candidates;
    }

    /**
     * @brief Allocate one exact immutable fixture owner without involving a model.
     * @param ram Original RAM authority retaining the serialized section.
     * @param key Complete logical block identity.
     * @param value Byte value inspected independently after restart.
     * @return Original handle with a completed host payload.
     * @throws std::runtime_error If the fixture's declared capacity cannot be materialized.
     */
    PrefixBlockHandle initializedBlock(RamPrefixStorageBackend &ram, const PrefixCacheKey &key, uint8_t value)
    {
        auto handle = ram.allocate(key, layout());
        if (!handle.valid()) throw std::runtime_error("shutdown MPI fixture RAM allocation failed");
        std::memset(handle.kv_payload, value, handle.total_bytes);
        return handle;
    }

    /**
     * @brief Verify an independently reopened archive's selected payload.
     * @param archive Restarted durable authority, after all fixture writers retire.
     * @param key Selected logical identity.
     * @param value Expected immutable byte value.
     */
    void expectRestoredBytes(DiskPrefixStorageBackend &archive, const PrefixCacheKey &key, uint8_t value)
    {
        PrefixBlockHandle restored;
        ASSERT_TRUE(archive.readBlock(key, layout(), &restored));
        const auto *bytes = static_cast<const uint8_t *>(restored.kv_payload);
        EXPECT_TRUE(std::all_of(bytes, bytes + restored.total_bytes,
            [value](uint8_t actual) { return actual == value; }));
    }

    /**
     * @brief Prove skewed candidate inventories cannot consume a peer's first priority.
     *
     * Rank zero owns nine candidates; every peer owns a different smaller count.
     * The exact shared budget fits one most-recent owner from each participant.
     * No candidate capacity is sent, cleared, or used as a communication extent.
     */
    void runSkewedOwners()
    {
        const int rank = mpiWorldRank();
        const int ranks = mpiWorldSize();
        const int count = rank == 0 ? 9 : rank + 1;
        const auto root = sharedDirectory();
        const auto path = root / (std::string(kModelIdentity) + ".kvcache");
        const size_t budget = static_cast<size_t>(ranks) * kBlockBytes;
        const uint64_t fingerprint = static_cast<uint64_t>(rank + 1);
        {
            auto disk = std::make_shared<DiskPrefixStorageBackend>(path, budget, kModelIdentity);
            ASSERT_TRUE(disk->ready()) << disk->initializationError();
            auto ram = std::make_shared<RamPrefixStorageBackend>(static_cast<size_t>(count) * kBlockBytes);
            auto cache = std::make_shared<PrefixStateCache>(ram->budgetBytes(), ram, disk);
            for (int index = 0; index < count; ++index)
            {
                const auto key = makePrefixCacheKey(fingerprint, 0, 0, 0, {100 + index});
                ASSERT_TRUE(cache->insert(initializedBlock(*ram, key, static_cast<uint8_t>(rank + index))));
            }
            ObservedContext context;
            PrefixCacheShutdown shutdown;
            shutdown.add(cache);
            const auto result = shutdown.flush(&context);
            EXPECT_EQ(result.candidates, static_cast<uint64_t>(count));
            EXPECT_EQ(result.selected, 1u);
            EXPECT_EQ(result.written, 1u);
            const size_t sent_candidates = expectMetadataOnly(context);
            if (rank != 0) EXPECT_EQ(sent_candidates, static_cast<size_t>(count));
            MPI_Barrier(MPI_COMM_WORLD); // Test oracle only; production has no post-write collective.
        }
        {
            DiskPrefixStorageBackend restarted(path, budget, kModelIdentity);
            ASSERT_TRUE(restarted.ready()) << restarted.initializationError();
            EXPECT_EQ(restarted.usedBytes(), budget);
            const auto entries = restarted.compatibleEntries(fingerprint, layout());
            ASSERT_EQ(entries.size(), 1u);
            const auto expected = makePrefixCacheKey(fingerprint, 0, 0, 0, {100 + count - 1});
            EXPECT_EQ(entries.front().key, expected);
            expectRestoredBytes(restarted, expected, static_cast<uint8_t>(rank + count - 1));
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::filesystem::remove_all(root);
    }

    /**
     * @brief Check duplicate logical keys consume one shared-budget selection.
     * @param exact_replica Whether every participant owns the same immutable incarnation.
     *
     * A common first-priority key wins once; remaining capacity retains private
     * second-priority owners in stable rank order. The final rank's preexisting
     * common backing must be reused for an exact replica, or replaced by rank
     * zero's selected incarnation. Cold records cannot displace selected peers.
     */
    void runReplicatedOwners(bool exact_replica)
    {
        const int rank = mpiWorldRank();
        const int ranks = mpiWorldSize();
        const auto root = sharedDirectory();
        const auto path = root / (std::string(kModelIdentity) + ".kvcache");
        const size_t budget = static_cast<size_t>(ranks) * kBlockBytes;
        const auto common_key = makePrefixCacheKey(0xfeed, 0, 0, 0, {500});
        const auto private_key = makePrefixCacheKey(static_cast<uint64_t>(rank + 1), 0, 0, 0, {100});
        {
            auto disk = std::make_shared<DiskPrefixStorageBackend>(path, budget, kModelIdentity);
            ASSERT_TRUE(disk->ready()) << disk->initializationError();
            auto ram = std::make_shared<RamPrefixStorageBackend>(3 * kBlockBytes);
            auto cache = std::make_shared<PrefixStateCache>(3 * kBlockBytes, ram, disk);
            ASSERT_TRUE(cache->insert(initializedBlock(*ram, private_key, static_cast<uint8_t>(rank + 1))));
            auto common = initializedBlock(*ram, common_key,
                exact_replica ? uint8_t{0x5a} : static_cast<uint8_t>(0x40 + rank));
            // This is explicit immutable fixture identity, not payload hashing.
            common.payload_identity = {0x1234, exact_replica ? uint64_t{0x5678} : static_cast<uint64_t>(rank + 1)};
            ASSERT_TRUE(cache->insert(common));
            if (rank + 1 == ranks) ASSERT_TRUE(disk->writeBlock(common, nullptr));
            else
            {
                const auto cold_key = makePrefixCacheKey(0xbad, 0, 0, 0, {rank});
                ASSERT_TRUE(disk->writeBlock(initializedBlock(*ram, cold_key, 0xee), nullptr));
            }
            MPI_Barrier(MPI_COMM_WORLD); // Fixture preparation, outside measured protocol sends.
            ObservedContext context;
            PrefixCacheShutdown shutdown;
            shutdown.add(cache);
            const auto result = shutdown.flush(&context);
            const uint64_t selected = rank == 0 ? 2u : rank + 1 < ranks ? 1u : 0u;
            EXPECT_EQ(result.candidates, 2u);
            EXPECT_EQ(result.selected, selected);
            EXPECT_EQ(result.reused, exact_replica && rank == 0 ? 1u : 0u);
            EXPECT_EQ(result.written, selected - (exact_replica && rank == 0 ? 1u : 0u));
            const size_t sent_candidates = expectMetadataOnly(context);
            if (rank != 0) EXPECT_EQ(sent_candidates, 2u);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        {
            DiskPrefixStorageBackend restarted(path, budget, kModelIdentity);
            ASSERT_TRUE(restarted.ready()) << restarted.initializationError();
            EXPECT_EQ(restarted.usedBytes(), budget);
            EXPECT_EQ(restarted.compatibleEntries(0xfeed, layout()).size(), 1u);
            expectRestoredBytes(restarted, common_key, exact_replica ? uint8_t{0x5a} : uint8_t{0x40});
            const auto entries = restarted.compatibleEntries(static_cast<uint64_t>(rank + 1), layout());
            EXPECT_EQ(entries.size(), rank + 1 < ranks ? 1u : 0u);
            if (rank + 1 < ranks) expectRestoredBytes(restarted, private_key, static_cast<uint8_t>(rank + 1));
            EXPECT_TRUE(restarted.compatibleEntries(0xbad, layout()).empty());
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::filesystem::remove_all(root);
    }

    /**
     * @brief Reuse one archive with fresh large then small terminal cohorts.
     *
     * Every participant first owns eight candidates. A fresh cohort then has
     * one rank-zero candidate and empty peers, retaining the same RAM capacity
     * and disk budget. Each peer receives exactly the two live bytes of [] in
     * the smaller cohort; previous candidate/selection capacity cannot leak.
     */
    void runFreshLargeToSmallCohorts()
    {
        const int rank = mpiWorldRank();
        const int ranks = mpiWorldSize();
        const auto root = sharedDirectory();
        const auto path = root / (std::string(kModelIdentity) + ".kvcache");
        const size_t budget = static_cast<size_t>(ranks) * kBlockBytes;
        size_t larger_sent_bytes = 0;
        for (int cohort = 0; cohort < 2; ++cohort)
        {
            const int count = cohort == 0 ? 8 : rank == 0 ? 1 : 0;
            {
                auto disk = std::make_shared<DiskPrefixStorageBackend>(path, budget, kModelIdentity);
                ASSERT_TRUE(disk->ready()) << disk->initializationError();
                auto ram = std::make_shared<RamPrefixStorageBackend>(8 * kBlockBytes);
                auto cache = std::make_shared<PrefixStateCache>(8 * kBlockBytes, ram, disk);
                for (int index = 0; index < count; ++index)
                {
                    const auto key = makePrefixCacheKey(static_cast<uint64_t>(rank + 1), 0, 0, 0,
                        {1000 + cohort * 100 + index});
                    ASSERT_TRUE(cache->insert(initializedBlock(*ram, key, static_cast<uint8_t>(cohort + rank))));
                }
                ObservedContext context;
                PrefixCacheShutdown shutdown;
                shutdown.add(cache);
                const auto result = shutdown.flush(&context);
                EXPECT_EQ(result.candidates, static_cast<uint64_t>(count));
                EXPECT_EQ(result.selected, count == 0 ? 0u : 1u);
                const size_t sent_candidates = expectMetadataOnly(context);
                if (rank != 0) EXPECT_EQ(sent_candidates, static_cast<size_t>(count));
                if (cohort == 0) larger_sent_bytes = context.sent_bytes;
                else
                {
                    EXPECT_LT(context.sent_bytes, larger_sent_bytes);
                    if (rank == 0)
                    {
                        EXPECT_EQ(context.sent_bytes, static_cast<size_t>(ranks - 1) * (sizeof(uint64_t) + 2u));
                        for (const auto &body : context.metadata_bodies) EXPECT_EQ(body, "[]");
                    }
                    else
                    {
                        ASSERT_EQ(context.metadata_bodies.size(), 1u);
                        const auto manifest = nlohmann::json::parse(context.metadata_bodies.front());
                        EXPECT_TRUE(manifest.at(0).at("entries").empty());
                    }
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
            {
                DiskPrefixStorageBackend restarted(path, budget, kModelIdentity);
                ASSERT_TRUE(restarted.ready()) << restarted.initializationError();
                EXPECT_LE(restarted.usedBytes(), budget);
                if (count != 0)
                {
                    const auto key = makePrefixCacheKey(static_cast<uint64_t>(rank + 1), 0, 0, 0,
                        {1000 + cohort * 100 + count - 1});
                    expectRestoredBytes(restarted, key, static_cast<uint8_t>(cohort + rank));
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        if (rank == 0) std::filesystem::remove_all(root);
    }

    /** @brief Prove shared or independent capacity with an optional empty final rank. */
    void runFlush(bool shared, bool empty_last)
    {
        const int rank = mpiWorldRank();
        const int ranks = mpiWorldSize();
        ASSERT_GE(ranks, 2);
        const auto root = sharedDirectory();
        const auto directory = shared ? root : root / std::to_string(rank);
        const auto path = directory / (std::string(kModelIdentity) + ".kvcache");
        const int owners = ranks - (empty_last ? 1 : 0);
        const auto budget = (shared ? static_cast<size_t>(owners) : 1u) * kBlockBytes;
        {
            auto disk = std::make_shared<DiskPrefixStorageBackend>(path, budget, kModelIdentity);
            ASSERT_TRUE(disk->ready()) << disk->initializationError();
            auto ram = std::make_shared<RamPrefixStorageBackend>(3 * kBlockBytes);
            auto cache = std::make_shared<PrefixStateCache>(3 * kBlockBytes, ram, disk);
            if (!empty_last || rank + 1 < ranks)
            {
                for (int token : {1, 2, 3})
                {
                    const auto key = makePrefixCacheKey(static_cast<uint64_t>(rank + 1), 0, 0, 0, {token});
                    auto handle = ram->allocate(key, layout());
                    ASSERT_TRUE(handle.valid());
                    std::memset(handle.kv_payload, rank + token, handle.total_bytes);
                    ASSERT_TRUE(cache->insert(std::move(handle)));
                }
            }
            ObservedContext context;
            PrefixCacheShutdown shutdown;
            if (!empty_last || rank + 1 < ranks) shutdown.add(cache);
            const auto result = shutdown.flush(&context);
            EXPECT_EQ(result.written, (!empty_last || rank + 1 < ranks) ? 1u : 0u);
            EXPECT_EQ(context.bodies, rank == 0 ? static_cast<size_t>(ranks - 1) : 1u);
            EXPECT_LT(context.sent_bytes, kBlockBytes / 16)
                << "Shutdown sent payload bytes or padded capacity instead of candidate metadata";
            // This barrier belongs only to the test oracle, after every writer
            // has completed. Production retirement has no post-I/O collective.
            MPI_Barrier(MPI_COMM_WORLD);
        }
        {
            DiskPrefixStorageBackend restarted(path, budget, kModelIdentity);
            ASSERT_TRUE(restarted.ready()) << restarted.initializationError();
            EXPECT_LE(restarted.usedBytes(), budget);
            if (!empty_last || rank + 1 < ranks)
            {
                const auto fingerprint = static_cast<uint64_t>(rank + 1);
                const auto entries = restarted.compatibleEntries(fingerprint, layout());
                ASSERT_EQ(entries.size(), 1u);
                const auto key = makePrefixCacheKey(fingerprint, 0, 0, 0, {3});
                EXPECT_EQ(entries.front().key, key);
                PrefixBlockHandle restored;
                ASSERT_TRUE(restarted.readBlock(key, layout(), &restored));
                const auto *bytes = static_cast<const uint8_t *>(restored.kv_payload);
                EXPECT_TRUE(std::all_of(bytes, bytes + restored.total_bytes,
                    [rank](uint8_t value) { return value == rank + 3; }));
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::filesystem::remove_all(root);
    }
}

/** @test All ranks retain their most recent entry regardless of native writer order. */
TEST(Test__PrefixCacheShutdownMPI, SharedArchiveRetainsEveryParticipantsHighestPriority)
{
    runFlush(true, false);
}

/** @test Expert-only ranks without a RAM prefix tier still join and retire the protocol. */
TEST(Test__PrefixCacheShutdownMPI, EmptyParticipantsExchangeOnlyMetadata)
{
    runFlush(true, true);
}

/** @test Separate host-local archive identities each receive their configured capacity. */
TEST(Test__PrefixCacheShutdownMPI, IndependentArchivesHaveIndependentSelections)
{
    runFlush(false, false);
}

/** @test Skewed inventories retain every rank's highest priority within one shared budget. */
TEST(Test__PrefixCacheShutdownMPI, SkewedCandidateCountsCannotDisplacePeerPriority)
{
    runSkewedOwners();
}

/** @test Exact replicated incarnations consume one slot and reuse protected durable backing. */
TEST(Test__PrefixCacheShutdownMPI, ReplicatedIdentitiesShareOneBudgetSlotAndReuseBacking)
{
    runReplicatedOwners(true);
}

/** @test Duplicate keys with distinct incarnations preserve the selected owner's priority. */
TEST(Test__PrefixCacheShutdownMPI, DuplicateKeysRetainTheHighestPriorityIncarnation)
{
    runReplicatedOwners(false);
}

/** @test Fresh small cohorts transmit only current metadata after a large terminal selection. */
TEST(Test__PrefixCacheShutdownMPI, FreshLargeToSmallCohortsDoNotTransmitOldCapacity)
{
    runFreshLargeToSmallCohorts();
}
