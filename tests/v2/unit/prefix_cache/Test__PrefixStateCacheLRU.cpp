/**
 * @file Test__PrefixStateCacheLRU.cpp
 * @brief Unit regressions for prefix-cache lookup, ownership, and tier LRU.
 *
 * Archive pressure must keep payload leases valid without turning a request
 * thread into a disk writer. A held native archive lock models a slow durable
 * tier deterministically, instead of depending on an unusually busy SSD.
 * Restored block chains must retire before harvest admission; only the exact
 * terminal reuse witness remains a request-owned physical lease.
 */

#include <gtest/gtest.h>

#include "execution/prefix_cache/DiskPrefixStorageBackend.h"
#include "execution/prefix_cache/PrefixStateCache.h"
#include "execution/prefix_cache/RamPrefixStorageBackend.h"
#include "planning/PhysicalMemoryAuthority.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

using namespace llaminar2;

namespace
{
    constexpr const char *kModelSha256 =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    PrefixPayloadLayout layoutBytes(size_t bytes)
    {
        PrefixPayloadLayout layout;
        layout.block_size = 1;
        layout.fa_layers = 1;
        layout.total_layers = 1;
        layout.bytes_per_fa_layer_k = bytes / 2;
        layout.bytes_per_fa_layer_v = bytes - layout.bytes_per_fa_layer_k;
        return layout;
    }

    PrefixCacheKey keyFor(int block)
    {
        return makePrefixCacheKey(0xbeef, 0, block, block, {block});
    }

    PrefixCacheKey keyForFingerprint(uint64_t fingerprint, int block)
    {
        return makePrefixCacheKey(
            fingerprint,
            0,
            block,
            block,
            {block});
    }

    std::filesystem::path tempDir()
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::filesystem::temp_directory_path() / ("llaminar_prefix_state_cache_" + std::to_string(stamp));
    }

    std::shared_ptr<DiskPrefixStorageBackend> makeDiskBackend(
        const std::filesystem::path &directory,
        size_t budget_bytes)
    {
        return std::make_shared<DiskPrefixStorageBackend>(
            directory / (std::string(kModelSha256) + ".kvcache"),
            budget_bytes,
            kModelSha256);
    }

    /**
     * @brief Advance asynchronous pressure at explicit test-only writer joins.
     *
     * The already-allocated handle belongs to the fixture, not a production
     * allocation path. A retry follows one completed durable owner, never an
     * arbitrary delay or an I/O failure. The resident count bounds transitions.
     */
    bool insertAfterArchivePublication(
        PrefixStateCache &cache,
        DiskPrefixStorageBackend &disk,
        const PrefixBlockHandle &handle)
    {
        const auto frontiers = cache.size() + 1u;
        for (size_t frontier = 0; frontier < frontiers; ++frontier)
        {
            if (cache.insert(handle))
                return true;
            std::string error;
            if (!disk.waitForPersistence(&error))
                return false;
            cache.publishCompletedPersistence();
        }
        return cache.insert(handle);
    }

    /** @brief Join known pending victims before observing a verified promotion. */
    std::optional<PrefixBlockHandle> findAfterArchivePublication(
        PrefixStateCache &cache,
        DiskPrefixStorageBackend &disk,
        const PrefixCacheKey &key)
    {
        const auto frontiers = cache.size() + 1u;
        for (size_t frontier = 0; frontier < frontiers; ++frontier)
        {
            if (auto handle = cache.find(key))
                return handle;
            std::string error;
            if (!disk.waitForPersistence(&error))
                return std::nullopt;
            cache.publishCompletedPersistence();
        }
        return cache.find(key);
    }

    /** @brief Own a test-only competing writer's exact native archive lock. */
    class HeldArchiveLock final
    {
    public:
        /** @brief Acquire a separate file description, like another process. */
        explicit HeldArchiveLock(const std::filesystem::path &archive)
            : fd_(::open((archive.string() + ".lock").c_str(),
                         O_RDWR | O_CLOEXEC))
        {
            if (fd_ >= 0 && ::flock(fd_, LOCK_EX) != 0)
            {
                ::close(fd_);
                fd_ = -1;
            }
        }
        /** @brief Release before a failed assertion can strand a worker. */
        ~HeldArchiveLock() { release(); }
        HeldArchiveLock(const HeldArchiveLock &) = delete;
        HeldArchiveLock &operator=(const HeldArchiveLock &) = delete;
        /** @return Whether the fixture really owns the competing writer lock. */
        bool valid() const noexcept { return fd_ >= 0; }
        /** @brief Permit the archive worker to commit after the passive probe. */
        void release() noexcept
        {
            if (fd_ >= 0)
            {
                (void)::flock(fd_, LOCK_UN);
                (void)::close(fd_);
                fd_ = -1;
            }
        }
    private:
        int fd_ = -1;
    };

    /**
     * @brief Hold one unrelated writer's final source retirement deterministically.
     *
     * A fixture-owned vector blocks in its deleter after native writing, but
     * before its immutable receipt publishes. This proves required admission
     * waits for its own victims, not global writer idleness. No production hook
     * or additional cache reservation is involved.
     */
    class HeldPayloadRetirement final
    {
    public:
        /** @brief Create the independent test-only retirement frontier. */
        HeldPayloadRetirement() : state_(std::make_shared<State>()) {}
        /** @brief Ensure teardown cannot strand the writer in the test gate. */
        ~HeldPayloadRetirement() { release(); }
        HeldPayloadRetirement(const HeldPayloadRetirement &) = delete;
        HeldPayloadRetirement &operator=(const HeldPayloadRetirement &) = delete;

        /** @return An unrelated immutable payload with a held final deleter. */
        PrefixBlockHandle payload(const PrefixCacheKey &key, size_t bytes)
        {
            auto storage = std::shared_ptr<std::vector<uint8_t>>(
                new std::vector<uint8_t>(bytes, 0x5a),
                [state = state_](std::vector<uint8_t> *data)
                {
                    std::unique_lock lock(state->mutex);
                    if (state->phase != Phase::Released)
                    {
                        state->phase = Phase::Retiring;
                        state->changed.notify_all();
                        state->changed.wait(lock, [&] { return state->phase == Phase::Released; });
                    }
                    lock.unlock();
                    delete data;
                });
            PrefixBlockHandle handle;
            handle.key = key;
            handle.payload_identity = PrefixPayloadIdentity::fresh();
            handle.layout = layoutBytes(bytes);
            handle.total_bytes = bytes;
            handle.kv_payload = storage->data();
            handle.kv_storage = std::move(storage);
            return handle;
        }

        /** @return Whether the writer reached this exact source-release boundary. */
        bool observeRetiring()
        {
            std::unique_lock lock(state_->mutex);
            return state_->changed.wait_for(lock, std::chrono::seconds(1), [&]
            {
                return state_->phase == Phase::Retiring;
            });
        }

        /** @brief Publish the sole fixture-owned retirement permission. */
        void release() noexcept
        {
            {
                std::lock_guard lock(state_->mutex);
                state_->phase = Phase::Released;
            }
            state_->changed.notify_all();
        }

    private:
        /** @brief Source ownership advances monotonically to fixture release. */
        enum class Phase { Queued, Retiring, Released };
        /** @brief Shared test gate outlives the final payload-owner callback. */
        struct State
        {
            std::mutex mutex;
            std::condition_variable changed;
            Phase phase = Phase::Queued;
        };
        std::shared_ptr<State> state_;
    };

    /** @brief Admit one production-style bounded host archive reservation. */
    std::shared_ptr<PhysicalMemoryAuthority> prefixAuthority(
        size_t prefix_bytes)
    {
        PhysicalMemoryPlanBuilder builder;
        builder.add(
            PhysicalMemoryResource{
                .world_rank = 0,
                .device = DeviceId::cpu(),
                .total_bytes = 4096u,
                .admission_available_bytes = 4096u,
            },
            PhysicalMemoryOwner::PrefixHostTier,
            prefix_bytes);
        auto admission = std::make_shared<
            const PhysicalMemoryPlanAdmissionCertificate>(builder.build());
        return std::make_shared<PhysicalMemoryAuthority>(
            std::move(admission), 0);
    }

    /** @brief Observe the real writer frontier without injecting a test hook. */
    bool observeExecuting(const PrefixArchivePersistenceTicket &ticket)
    {
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(200);
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (ticket.phase() == PrefixArchiveMutationPhase::Executing)
                return true;
            std::this_thread::yield();
        }
        return false;
    }
} // namespace

/**
 * @test Slow durable eviction must not block early inference-side preparation.
 *
 * No disk timeout is adjusted: a competing writer is held only until the
 * bounded observation finishes. Pending physical owners produce typed Busy;
 * they must not be forgotten, overcommitted, or synchronously persisted. Required
 * publication has a separate exact-completion contract tested below.
 */
TEST(Test__PrefixStateCacheLRU,
     PendingDiskPersistenceNeverBlocksRequestAdmission)
{
    const auto directory = tempDir();
    {
        auto disk = makeDiskBackend(directory, 1024u);
        ASSERT_TRUE(disk->ready());
        HeldArchiveLock writer_lock(disk->archivePath());
        ASSERT_TRUE(writer_lock.valid());
        auto ram = std::make_shared<RamPrefixStorageBackend>(96u);
        PrefixStateCache cache(96u, ram, disk);
        for (int block = 0; block < 3; ++block)
        {
            auto handle = ram->allocate(keyFor(block), layoutBytes(32u));
            ASSERT_TRUE(handle.valid());
            ASSERT_TRUE(cache.insert(std::move(handle)));
        }
        auto admission = std::async(std::launch::async, [&]()
        {
            return cache.prepareInsert(keyFor(3), 32u);
        });
        const auto state = admission.wait_for(std::chrono::milliseconds(200));

        // Always release before get()/destruction, even for the negative
        // implementation. This regression fails promptly, never by deadlock.
        writer_lock.release();
        const auto preparation = admission.get();
        EXPECT_EQ(state, std::future_status::ready)
            << "RAM admission performed foreground archive I/O";
        EXPECT_EQ(preparation, PrefixRamInsertPreparation::Busy)
            << "Incomplete disk publication cannot manufacture free RAM";
        EXPECT_EQ(cache.usedBytes(), 96u);
        EXPECT_EQ(ram->usedBytes(), 96u);
        ASSERT_TRUE(disk->waitForPersistence());
        cache.publishCompletedPersistence();
        EXPECT_EQ(disk->usedBytes(), 32u)
            << "One incoming block must not persist the entire RAM capacity";
        EXPECT_EQ(cache.stats().ram_to_disk_demotions, 1u);
    }
    std::filesystem::remove_all(directory);
}

/**
 * @test Start every necessary victim before inference, without flushing the tier.
 *
 * A recurrent checkpoint may be larger than an ordinary KV record. Queuing
 * only the first victim leaves most of the required work until terminal
 * harvest; queuing the entire tier needlessly consumes bandwidth. Both are
 * rejected by this exact two-victim publication witness.
 */
TEST(Test__PrefixStateCacheLRU, AsyncPressurePlansAllAndOnlyNecessaryVictims)
{
    const auto directory = tempDir();
    {
        auto disk = makeDiskBackend(directory, 1024u);
        ASSERT_TRUE(disk->ready());
        HeldArchiveLock writer_lock(disk->archivePath());
        ASSERT_TRUE(writer_lock.valid());
        auto ram = std::make_shared<RamPrefixStorageBackend>(96u);
        PrefixStateCache cache(96u, ram, disk);
        for (int block = 0; block < 3; ++block)
        {
            auto handle = ram->allocate(keyFor(block), layoutBytes(32u));
            ASSERT_TRUE(handle.valid());
            ASSERT_TRUE(cache.insert(std::move(handle)));
        }
        EXPECT_EQ(cache.prepareCapacity(64u), PrefixRamInsertPreparation::Busy);
        EXPECT_EQ(cache.usedBytes(), 96u);
        EXPECT_EQ(ram->usedBytes(), 96u);
        writer_lock.release();
        ASSERT_TRUE(disk->waitForPersistence());
        cache.publishCompletedPersistence();
        EXPECT_EQ(disk->usedBytes(), 64u);
        EXPECT_EQ(cache.stats().ram_to_disk_demotions, 2u);
        EXPECT_EQ(cache.prepareCapacity(64u), PrefixRamInsertPreparation::Prepared)
            << "One prefill interval must suffice for the complete planned victim set";
        EXPECT_EQ(cache.usedBytes(), 32u);
        EXPECT_TRUE(cache.isRamResident(keyFor(2)));
    }
    std::filesystem::remove_all(directory);
}

/**
 * @test Consumed request chains stop retaining physical bytes needed by harvest.
 *
 * Exercise dense, hybrid and MTP archives at the admitted capacity ceiling.
 * Eviction must remain Busy while the restore lookup owns all three records;
 * selecting terminal-only harvest admission releases the other physical owners.
 */
TEST(Test__PrefixStateCacheLRU, RestoredChainReleasesCapacityForEveryArchiveKind)
{
    for (const bool hybrid : {false, true})
        for (const bool mtp : {false, true})
            for (int iteration = 0; iteration < 20; ++iteration)
            {
                auto layout = layoutBytes(32u);
                layout.includes_hybrid_state = hybrid;
                layout.hybrid_state_bytes = hybrid ? 16u : 0u;
                layout.includes_mtp_state = mtp;
                layout.mtp_kv_bytes = mtp ? 16u : 0u;
                const size_t bytes = layout.totalBytes();
                auto ram = RamPrefixStorageBackend::create(
                    DeviceId::cpu(), bytes * 3u, prefixAuthority(bytes * 3u));
                ASSERT_NE(ram, nullptr);
                PrefixStateCache cache(bytes * 3u, ram);
                PrefixLookupResult admission;
                admission.supported = admission.cache_enabled = true;
                admission.block_size = 1;
                admission.fingerprint_key = 0xbeef;
                admission.cached_tokens = 3;
                admission.requires_terminal_hidden = admission.requires_terminal_logits = false;
                for (int block = 0; block < 3; ++block)
                {
                    auto handle = ram->allocate(keyFor(block), layout);
                    ASSERT_TRUE(handle.valid());
                    handle.has_hybrid_state = hybrid;
                    ASSERT_TRUE(cache.insert(handle));
                    admission.blocks.push_back(std::move(handle));
                }
                // An event-owned restore may retain a temporary alias; here
                // the synchronous CPU import is complete, so harvest is its
                // only remaining request owner. Eviction alone cannot help.
                EXPECT_EQ(cache.prepareCapacity(bytes), PrefixRamInsertPreparation::Busy);
                EXPECT_EQ(ram->availableAllocationBytes(), 0u);
                admission = admission.forHarvest(3);
                ASSERT_EQ(admission.blocks.size(), 1u);
                EXPECT_EQ(admission.blocks.back().key, keyFor(2));
                EXPECT_EQ(admission.terminalHarvestDisposition(keyFor(2), 3),
                    PrefixTerminalHarvestDisposition::ReuseAdmittedArchive);
                EXPECT_EQ(ram->availableAllocationBytes(), bytes * 2u);
                EXPECT_NO_THROW(cache.completeInsertPreparation(keyFor(3), bytes));
                auto next = ram->allocate(keyFor(3), layout);
                EXPECT_TRUE(next.valid());
                EXPECT_TRUE(admission.blocks.back().valid());
            }
}

/** @test Only consumed sections remain physically charged across a rich-checkpoint row restore. */
TEST(Test__PrefixStateCacheLRU, RichRestoreRetainsOnlyConsumedPhysicalSections)
{
    for (const bool hybrid : {false, true})
        for (const bool mtp : {false, true})
            for (int iteration = 0; iteration < 20; ++iteration)
            {
                SCOPED_TRACE(hybrid);
                SCOPED_TRACE(mtp);
                SCOPED_TRACE(iteration);
                auto layout = layoutBytes(32u);
                layout.includes_hybrid_state = hybrid;
                layout.hybrid_state_bytes = hybrid ? 64u : 0u;
                layout.includes_mtp_state = mtp;
                layout.mtp_layers = mtp ? 1 : 0;
                layout.mtp_kv_bytes = mtp ? 16u : 0u;
                layout.bytes_per_mtp_layer_k = layout.bytes_per_mtp_layer_v = 8u;
                layout.includes_terminal_hidden = layout.includes_terminal_logits = true;
                layout.terminal_hidden_bytes = layout.terminal_logits_bytes = 16u;
                const size_t runtime_bytes = 13u;
                const auto allocation = PrefixPayloadAllocationPlan::archive(layout, runtime_bytes);
                const size_t capacity = 3u * allocation.totalBytes();
                auto authority = prefixAuthority(capacity);
                auto ram = RamPrefixStorageBackend::create(DeviceId::cpu(), capacity, authority);
                ASSERT_NE(ram, nullptr);
                PrefixStateCache cache(capacity, ram);
                std::vector<PrefixPayloadReadLease> reads;
                size_t retained = 0u;
                for (int block = 0; block < 3; ++block)
                {
                    auto archive = ram->allocate(keyFor(block), layout);
                    ASSERT_TRUE(archive.valid());
                    ASSERT_TRUE(ram->attachModelRuntimeState(&archive,
                        std::make_shared<std::vector<uint8_t>>(runtime_bytes, uint8_t{0x7b})));
                    ASSERT_TRUE(cache.insert(archive));
                    reads.push_back(block == 2 ? PrefixPayloadReadLease::wholeArchive(std::move(archive))
                                               : PrefixPayloadReadLease::sequenceRows(std::move(archive)));
                    retained += reads.back().retainedBytes();
                }
                ASSERT_TRUE(cache.clear());
                const auto claimed = [&] {
                    return authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::PrefixHostTier,
                        PhysicalMemoryMaterializationKind::NewAllocation);
                };
                EXPECT_EQ(retained, 2u * (layout.faKVBytes() + layout.mtpKVBytes()) + allocation.totalBytes());
                EXPECT_EQ(claimed(), retained);
                EXPECT_NO_THROW(cache.completeInsertPreparation(keyFor(3),
                    PrefixPayloadAllocationPlan::archive(layout)));
                auto next = ram->allocate(keyFor(3), layout);
                ASSERT_TRUE(next.valid());
                EXPECT_EQ(claimed(), retained + layout.totalBytes());
                ASSERT_TRUE(ram->release(next));
                next = {};
                EXPECT_EQ(claimed(), retained);
                reads.clear();
                EXPECT_EQ(claimed(), 0u);
            }
}

/** @test Section allocation geometry rejects overflow before storage admission. */
TEST(Test__PrefixStateCacheLRU, PrefixSectionAllocationPlanAuthenticatesAllSectionExtents)
{
    auto layout = layoutBytes(32u);
    layout.includes_mtp_state = true;
    layout.mtp_kv_bytes = 17u;
    EXPECT_EQ(PrefixPayloadAllocationPlan::archive(layout, 13u).totalBytes(), 62u);
    EXPECT_EQ(PrefixPayloadAllocationPlan::contiguous(23u).totalBytes(), 23u);
    layout.includes_hybrid_state = true;
    layout.hybrid_state_bytes = std::numeric_limits<size_t>::max();
    EXPECT_THROW(PrefixPayloadAllocationPlan::archive(layout), std::overflow_error);
    layout = layoutBytes(32u);
    layout.fa_layers = -1;
    EXPECT_THROW(PrefixPayloadAllocationPlan::archive(layout), std::invalid_argument);
}

/** @test Capacity admission polls the restore authority for every archive kind. */
TEST(Test__PrefixStateCacheLRU, AdmissionRetiresCompletedRestoreSourcesBeforePhysicalCapacity)
{
    struct Retirement final : IPrefixRestoreSourceRetirement
    {
        mutable std::vector<PrefixBlockHandle> sources;
        bool reads_complete = false;
        mutable int polls = 0;
        /** @brief Model a completed asynchronous consumer without a second byte ledger. */
        void retireCompletedPrefixRestoreSources() const override
        {
            ++polls;
            if (reads_complete)
                sources.clear();
        }
    };
    for (const bool hybrid : {false, true})
        for (const bool mtp : {false, true})
            for (int iteration = 0; iteration < 20; ++iteration)
            {
                SCOPED_TRACE(hybrid);
                SCOPED_TRACE(mtp);
                SCOPED_TRACE(iteration);
                auto layout = layoutBytes(32u);
                layout.includes_hybrid_state = hybrid;
                layout.hybrid_state_bytes = hybrid ? 16u : 0u;
                layout.includes_mtp_state = mtp;
                layout.mtp_kv_bytes = mtp ? 16u : 0u;
                const size_t bytes = layout.totalBytes();
                auto ram = RamPrefixStorageBackend::create(
                    DeviceId::cpu(), bytes * 3u, prefixAuthority(bytes * 3u));
                ASSERT_NE(ram, nullptr);
                Retirement retirement;
                PrefixStateCache cache(bytes * 3u, ram, nullptr, nullptr, &retirement);
                for (int block = 0; block < 3; ++block)
                {
                    auto archive = ram->allocate(keyFor(block), layout);
                    ASSERT_TRUE(archive.valid());
                    ASSERT_TRUE(cache.insert(archive));
                    retirement.sources.push_back(std::move(archive));
                }
                EXPECT_EQ(cache.prepareCapacity(bytes), PrefixRamInsertPreparation::Busy);
                EXPECT_EQ(ram->availableAllocationBytes(), 0u);
                EXPECT_GT(retirement.polls, 0);
                const int earlier_polls = retirement.polls;
                retirement.reads_complete = true;
                EXPECT_NO_THROW(cache.completeInsertPreparation(keyFor(3), bytes));
                EXPECT_GT(retirement.polls, earlier_polls);
                EXPECT_TRUE(retirement.sources.empty());
                EXPECT_EQ(ram->availableAllocationBytes(), bytes * 3u);
            }
}

/**
 * @test Required durable publication joins its capacity victims, never unrelated work.
 * Force two necessary victims to remain pending beyond the producer's work.
 * A later unrelated payload's held retirement proves the join is receipt-local.
 */
TEST(Test__PrefixStateCacheLRU, AsyncArchiveRequiredTerminalWaitsOnlyForItsCapacity)
{
    const auto directory = tempDir();
    {
        auto disk = makeDiskBackend(directory, 1024u);
        auto ram = RamPrefixStorageBackend::create(
            DeviceId::cpu(), 96u, prefixAuthority(96u));
        ASSERT_TRUE(disk->ready());
        ASSERT_NE(ram, nullptr);
        PrefixStateCache cache(96u, ram, disk);
        HeldArchiveLock writer_lock(disk->archivePath());
        ASSERT_TRUE(writer_lock.valid());
        HeldPayloadRetirement unrelated_retirement;
        for (int block = 0; block < 3; ++block)
        {
            auto handle = ram->allocate(keyFor(block), layoutBytes(32u));
            ASSERT_TRUE(handle.valid());
            ASSERT_TRUE(cache.insert(std::move(handle)));
        }
        EXPECT_EQ(cache.prepareCapacity(64u), PrefixRamInsertPreparation::Busy);
        EXPECT_EQ(ram->availableAllocationBytes(), 0u)
            << "Queued writes still own their original physical bytes";
        const auto unrelated = disk->scheduleWrite(
            unrelated_retirement.payload(keyFor(99), 32u));
        auto terminal_layout = layoutBytes(32u);
        terminal_layout.includes_hybrid_state = true;
        terminal_layout.hybrid_state_bytes = 16u;
        terminal_layout.includes_mtp_state = true;
        terminal_layout.mtp_kv_bytes = 16u;
        std::promise<void> started;
        auto started_receipt = started.get_future();
        auto required = std::async(std::launch::async, [&]()
        {
            started.set_value();
            cache.completeInsertPreparation(keyFor(3), terminal_layout.totalBytes());
            auto terminal = ram->allocate(keyFor(3), terminal_layout);
            if (!terminal.valid())
                throw std::runtime_error("required terminal owner was not admitted");
            std::fill(terminal.kv_storage->begin(), terminal.kv_storage->end(), 0x11);
            std::fill(terminal.hybrid_storage->begin(), terminal.hybrid_storage->end(), 0x22);
            std::fill(terminal.mtp_storage->begin(), terminal.mtp_storage->end(), 0x33);
            if (!cache.insert(std::move(terminal)))
                throw std::runtime_error("required terminal publication was lost");
        });
        const auto started_state = started_receipt.wait_for(std::chrono::seconds(1));
        const auto pending_state = required.wait_for(std::chrono::milliseconds(50));
        writer_lock.release();
        const bool unrelated_is_retiring = unrelated_retirement.observeRetiring();
        const auto completed_state = required.wait_for(std::chrono::seconds(1));
        const bool unrelated_has_published = unrelated.publication() != nullptr;
        // Release every held native/source edge before get() or a fatal assertion,
        // including when the negative implementation waited for global idleness.
        unrelated_retirement.release();
        EXPECT_EQ(started_state, std::future_status::ready);
        EXPECT_EQ(pending_state, std::future_status::timeout);
        EXPECT_TRUE(unrelated_is_retiring);
        EXPECT_EQ(completed_state, std::future_status::ready)
            << "Required admission joined unrelated later archive work";
        EXPECT_FALSE(unrelated_has_published);
        ASSERT_NO_THROW(required.get());
        EXPECT_EQ(cache.usedBytes(), 96u);
        EXPECT_EQ(cache.stats().ram_to_disk_demotions, 2u);
        EXPECT_EQ(ram->availableAllocationBytes(), 0u);
        const auto restored = cache.find(keyFor(3));
        ASSERT_TRUE(restored.has_value());
        EXPECT_TRUE(restored->layout.includes_hybrid_state);
        EXPECT_TRUE(restored->layout.includes_mtp_state);
        EXPECT_EQ(*restored->kv_storage, std::vector<uint8_t>(32u, 0x11));
        EXPECT_EQ(*restored->hybrid_storage, std::vector<uint8_t>(16u, 0x22));
        EXPECT_EQ(*restored->mtp_storage, std::vector<uint8_t>(16u, 0x33));
        ASSERT_TRUE(disk->waitForPersistence());
    }
    std::filesystem::remove_all(directory);
}

/** @test An absent receipt cannot manufacture completion or wait forever. */
TEST(Test__PrefixStateCacheLRU, AsyncArchiveEmptyCompletionIsRejected)
{
    const PrefixArchivePersistenceTicket absent;
    EXPECT_THROW((void)absent.waitForPublication(), std::logic_error);
}

/** @test Required publication rejects untracked owners, never overcommitting RAM. */
TEST(Test__PrefixStateCacheLRU, AsyncArchiveRequiredPublicationRejectsUntrackedOwners)
{
    auto ram = RamPrefixStorageBackend::create(
        DeviceId::cpu(), 32u, prefixAuthority(32u));
    ASSERT_NE(ram, nullptr);
    PrefixStateCache cache(32u, ram);
    auto source = ram->allocate(keyFor(0), layoutBytes(32u));
    ASSERT_TRUE(source.valid());
    std::fill(source.kv_storage->begin(), source.kv_storage->end(), 0x5a);
    ASSERT_TRUE(cache.insert(source));
    ASSERT_TRUE(cache.retain(keyFor(0)));
    EXPECT_THROW(cache.completeInsertPreparation({}, 32u), std::runtime_error);
    EXPECT_THROW(cache.completeInsertPreparation(keyFor(1), 33u), std::runtime_error);
    EXPECT_THROW(cache.completeInsertPreparation(keyFor(0), 32u), std::runtime_error);
    EXPECT_THROW(cache.completeInsertPreparation(keyFor(1), 32u), std::runtime_error);
    EXPECT_TRUE(cache.isRamResident(keyFor(0)));
    EXPECT_EQ(ram->availableAllocationBytes(), 0u);
    EXPECT_EQ(*source.kv_storage, std::vector<uint8_t>(32u, 0x5a));
    ASSERT_TRUE(cache.release(keyFor(0)));
    source = {};
    EXPECT_NO_THROW(cache.completeInsertPreparation(keyFor(1), 32u));
    EXPECT_TRUE(ram->allocate(keyFor(1), layoutBytes(32u)).valid());
}

/**
 * @test Prepare both real recurrent frontiers and count shared KV keys once.
 *
 * The checkpoint's terminal record is also a nonterminal in the final token
 * chain. Its richer image must stay one publication. This geometry includes
 * main KV, shifted MTP KV, GDN and the model-owned terminal extension.
 */
TEST(Test__PrefixStateCacheLRU, PrefillPublicationPreparesBothRecurrentFrontiers)
{
    const auto directory = tempDir();
    {
        auto disk = makeDiskBackend(directory, 1024u);
        auto ram = std::make_shared<RamPrefixStorageBackend>(130u);
        PrefixStateCache cache(130u, ram, disk);
        for (int block = 100; block < 104; ++block)
        {
            auto handle = ram->allocate(keyFor(block), layoutBytes(32u));
            ASSERT_TRUE(handle.valid());
            ASSERT_TRUE(cache.insert(std::move(handle)));
        }
        PrefixLookupResult admission;
        admission.supported = admission.cache_enabled = true;
        admission.fingerprint_key = 0xbeef;
        admission.block_size = 1;
        admission.checkpoint_policy = PrefixCheckpointPolicy::ReusableBoundary;
        const std::vector<int32_t> tokens{20, 21, 22};
        const auto schedule = PrefixHarvestSchedule::forPrefill(admission, 3, 0);
        ASSERT_EQ(schedule.reusableCheckpoints(), std::vector<int>({2}));
        auto layout = layoutBytes(4u);
        layout.gdn_layers = 1;
        layout.total_layers = 2;
        layout.includes_hybrid_state = true;
        layout.hybrid_state_bytes = 16u;
        layout.includes_mtp_state = true;
        layout.mtp_kv_bytes = 2u;
        HeldArchiveLock writer_lock(disk->archivePath());
        ASSERT_TRUE(writer_lock.valid());
        EXPECT_EQ(cache.prepareHarvest(admission, tokens, schedule, layout, 8u),
                  PrefixRamInsertPreparation::Busy);
        EXPECT_EQ(cache.usedBytes(), 128u);
        writer_lock.release();
        ASSERT_TRUE(disk->waitForPersistence());
        cache.publishCompletedPersistence();
        EXPECT_EQ(disk->usedBytes(), 64u)
            << "6-byte ordinary block plus two 30-byte terminals needs exactly two victims";
        EXPECT_EQ(cache.prepareHarvest(admission, tokens, schedule, layout, 8u),
                  PrefixRamInsertPreparation::Prepared);
        EXPECT_EQ(cache.usedBytes(), 64u);
        EXPECT_EQ(ram->availableAllocationBytes(), 66u);
    }
    std::filesystem::remove_all(directory);
}

/**
 * @test A history rewrite restores its own sparse checkpoint after RAM-to-disk churn.
 *
 * Tiny device-free state payloads exercise the real causal keys, PMA admission,
 * archive worker, metadata selection and selected-section hydration. The state
 * marker identifies its producing frontier; later terminal state must never be
 * substituted for the shared leading history. No model execution is simulated.
 */
TEST(Test__PrefixStateCacheLRU, SparseCheckpointSurvivesHistoryRewriteAndColdPromptPressure)
{
    for (const bool mtp : {false, true})
    {
        SCOPED_TRACE(mtp);
        const auto directory = tempDir();
        {
            constexpr size_t ram_bytes = 4096u;
            constexpr size_t disk_bytes = 65536u;
            auto ram = RamPrefixStorageBackend::create(DeviceId::cpu(), ram_bytes, prefixAuthority(ram_bytes));
            ASSERT_TRUE(ram);
            auto disk = makeDiskBackend(directory, disk_bytes);
            auto cache = std::make_shared<PrefixStateCache>(ram_bytes, ram, disk);
            auto layout = layoutBytes(16u);
            layout.block_size = 64;
            layout.gdn_layers = 1;
            layout.total_layers = 2;
            layout.includes_hybrid_state = true;
            layout.hybrid_host_state_bytes = layout.hybrid_state_bytes = 256u;
            layout.includes_mtp_state = mtp;
            layout.mtp_kv_bytes = mtp ? 8u : 0u;
            layout.includes_terminal_hidden = layout.includes_terminal_logits = true;
            layout.terminal_hidden_bytes = layout.terminal_logits_bytes = 8u;
            PrefixLookupResult admission;
            admission.supported = admission.cache_enabled = true;
            admission.fingerprint_key = 0xbeef;
            admission.block_size = 64;
            admission.checkpoint_policy = PrefixCheckpointPolicy::ReusableBoundary;
            const auto archive = [&](const std::vector<int32_t> &tokens)
            {
                const auto schedule = PrefixHarvestSchedule::forPrefill(admission, tokens.size(), 0);
                ASSERT_NE(cache->prepareHarvest(admission, tokens, schedule, layout, 0u),
                          PrefixRamInsertPreparation::Error);
                auto frontiers = schedule.reusableCheckpoints();
                frontiers.push_back(tokens.size());
                for (const int frontier : frontiers)
                {
                    uint64_t parent = 0;
                    for (int start = 0; start < frontier; start += 64)
                    {
                        const int end = std::min(frontier, start + 64);
                        const auto key = makePrefixCacheKey(0xbeef, parent, start / 64, start,
                            {tokens.begin() + start, tokens.begin() + end});
                        parent = key.stableHash();
                        const bool terminal = end == frontier;
                        if (!terminal && cache->contains(key)) continue;
                        auto block_layout = layout;
                        block_layout.includes_hybrid_state = terminal;
                        block_layout.includes_terminal_hidden = block_layout.includes_terminal_logits = terminal;
                        cache->completeInsertPreparation(key, PrefixPayloadAllocationPlan::archive(block_layout));
                        auto handle = ram->allocate(key, block_layout);
                        ASSERT_TRUE(handle.valid());
                        handle.has_hybrid_state = handle.has_terminal_hidden = handle.has_terminal_logits = terminal;
                        if (terminal)
                            std::memcpy(handle.hybrid_storage->data(), &frontier, sizeof(frontier));
                        ASSERT_TRUE(cache->insert(std::move(handle)));
                        EXPECT_LE(cache->usedBytes(), ram_bytes);
                        EXPECT_LE(ram->usedBytes(), ram_bytes);
                    }
                }
                ASSERT_TRUE(disk->waitForPersistence());
                cache->publishCompletedPersistence();
                EXPECT_LE(disk->usedBytes(), disk_bytes);
            };
            std::vector<int32_t> original(20037, 7);
            archive(original);
            ASSERT_FALSE(HasFatalFailure());
            archive(std::vector<int32_t>(16001, 9));
            ASSERT_FALSE(HasFatalFailure());
            EXPECT_GT(disk->usedBytes(), 0u);
            std::vector<int32_t> rewritten(9001, 11);
            std::copy_n(original.begin(), 6212, rewritten.begin());
            auto lookup = admission;
            lookup.payload_plan = cache->beginLookup();
            uint64_t parent = 0;
            for (int start = 0; start < static_cast<int>(rewritten.size()); start += 64)
            {
                const int end = std::min(static_cast<int>(rewritten.size()), start + 64);
                const auto found = lookup.payload_plan->selectLongest(0xbeef, parent, start / 64, start,
                    {rewritten.begin() + start, rewritten.begin() + end});
                if (!found) break;
                lookup.blocks.push_back(*found);
                lookup.cached_tokens = found->key.token_start + found->key.token_count;
                parent = found->key.stableHash();
            }
            const auto selected = lookup.clampedTo(lookup.payload_plan->boundedTokenCount(lookup.blocks));
            ASSERT_EQ(selected.cached_tokens, 4096);
            const auto restored = selected.materializeRestoreBlocks();
            ASSERT_EQ(restored.size(), 64u);
            int checkpoint = 0;
            ASSERT_TRUE(restored.back().has_hybrid_state);
            std::memcpy(&checkpoint, restored.back().hybrid_payload, sizeof(checkpoint));
            EXPECT_EQ(checkpoint, 4096);
            for (size_t index = 0; index + 1 < restored.size(); ++index)
                EXPECT_FALSE(restored[index].has_hybrid_state);
            EXPECT_LE(ram->usedBytes(), ram_bytes);
        }
        std::filesystem::remove_all(directory);
    }
}

/** @test Only publication releases the background writer's original PMA alias. */
TEST(Test__PrefixStateCacheLRU, AsyncArchiveWriterRetainsCanonicalPhysicalLease)
{
    const auto directory = tempDir();
    {
        auto disk = makeDiskBackend(directory, 32u);
        auto ram = RamPrefixStorageBackend::create(
            DeviceId::cpu(), 32u, prefixAuthority(32u));
        ASSERT_TRUE(disk->ready());
        ASSERT_NE(ram, nullptr);
        auto handle = ram->allocate(keyFor(0), layoutBytes(32u));
        ASSERT_TRUE(handle.valid());
        std::fill(handle.kv_storage->begin(), handle.kv_storage->end(), 0x5a);
        HeldArchiveLock writer_lock(disk->archivePath());
        ASSERT_TRUE(writer_lock.valid());
        const auto ticket = disk->scheduleWrite(handle);
        EXPECT_TRUE(observeExecuting(ticket));
        ASSERT_TRUE(ram->release(handle));
        handle = {};
        EXPECT_EQ(ram->usedBytes(), 0u);
        EXPECT_FALSE(ram->canStore(32u))
            << "The queued/native writer still owns the canonical physical lease";
        EXPECT_EQ(ticket.publication(), nullptr);
        writer_lock.release();
        ASSERT_TRUE(disk->waitForPersistence());
        const auto publication = ticket.publication();
        ASSERT_NE(publication, nullptr);
        const auto *written = std::get_if<PrefixArchiveWritePublication>(publication.get());
        ASSERT_NE(written, nullptr);
        EXPECT_NE(written->executor, std::this_thread::get_id());
        EXPECT_EQ(written->disk_handle.total_bytes, 32u);
        EXPECT_TRUE(ram->canStore(32u));
        PrefixBlockHandle restored;
        ASSERT_TRUE(disk->readBlock(keyFor(0), layoutBytes(32u), &restored));
        ASSERT_NE(restored.kv_storage, nullptr);
        EXPECT_TRUE(std::all_of(restored.kv_storage->begin(), restored.kv_storage->end(),
                                [](uint8_t byte) { return byte == 0x5a; }));
    }
    std::filesystem::remove_all(directory);
}

/** @test An ordered tombstone cancels old queued bytes without cancelling their replacement. */
TEST(Test__PrefixStateCacheLRU, AsyncArchiveRetirementCannotResurrectQueuedOldPayload)
{
    const auto directory = tempDir();
    {
        auto disk = makeDiskBackend(directory, 256u);
        auto ram = RamPrefixStorageBackend::create(
            DeviceId::cpu(), 128u, prefixAuthority(128u));
        ASSERT_TRUE(disk->ready());
        ASSERT_NE(ram, nullptr);
        auto busy = ram->allocate(keyFor(0), layoutBytes(32u));
        auto old = ram->allocate(keyFor(1), layoutBytes(32u));
        ASSERT_TRUE(busy.valid());
        ASSERT_TRUE(old.valid());
        std::fill(old.kv_storage->begin(), old.kv_storage->end(), 0x11);
        HeldArchiveLock writer_lock(disk->archivePath());
        ASSERT_TRUE(writer_lock.valid());
        const auto busy_ticket = disk->scheduleWrite(busy);
        EXPECT_TRUE(observeExecuting(busy_ticket));
        const auto old_ticket = disk->scheduleWrite(old);
        EXPECT_EQ(old_ticket.phase(), PrefixArchiveMutationPhase::Queued);
        const auto retirement = disk->scheduleRetirement(old.key);
        EXPECT_EQ(old_ticket.publication(), nullptr)
            << "Cancellation cannot publish while the queued source is still retained";
        ASSERT_TRUE(ram->release(old));
        old = {};
        EXPECT_FALSE(ram->canStore(65u))
            << "The cancelled queue entry still owns its 32-byte PMA lease";
        auto rich_layout = layoutBytes(32u);
        rich_layout.includes_mtp_state = true;
        rich_layout.mtp_kv_bytes = 16u;
        auto replacement = ram->allocate(keyFor(1), rich_layout);
        ASSERT_TRUE(replacement.valid());
        std::fill(replacement.kv_storage->begin(), replacement.kv_storage->end(), 0x22);
        std::fill(replacement.mtp_storage->begin(), replacement.mtp_storage->end(), 0x33);
        const auto replacement_ticket = disk->scheduleWrite(replacement);
        writer_lock.release();
        const auto cancelled = old_ticket.waitForPublication();
        ASSERT_NE(cancelled, nullptr);
        EXPECT_TRUE(std::holds_alternative<PrefixArchiveCancelledPublication>(*cancelled));
        ASSERT_TRUE(disk->waitForPersistence());
        ASSERT_NE(retirement.publication(), nullptr);
        EXPECT_TRUE(std::holds_alternative<PrefixArchiveRetirementPublication>(*retirement.publication()));
        ASSERT_NE(replacement_ticket.publication(), nullptr);
        EXPECT_TRUE(std::holds_alternative<PrefixArchiveWritePublication>(*replacement_ticket.publication()));
        PrefixBlockHandle restored;
        ASSERT_TRUE(disk->readBlock(keyFor(1), rich_layout, &restored));
        EXPECT_EQ(*restored.kv_storage, *replacement.kv_storage);
        EXPECT_EQ(*restored.mtp_storage, *replacement.mtp_storage);
    }
    std::filesystem::remove_all(directory);
}

/** @test A failed background put seals admission and reaches the request as an error. */
TEST(Test__PrefixStateCacheLRU, AsyncArchiveFailureIsFatalNotACacheMiss)
{
    const auto directory = tempDir();
    {
        auto disk = makeDiskBackend(directory, 128u);
        auto ram = std::make_shared<RamPrefixStorageBackend>(32u);
        ASSERT_TRUE(disk->ready());
        PrefixStateCache cache(32u, ram, disk);
        auto handle = ram->allocate(keyFor(0), layoutBytes(32u));
        ASSERT_TRUE(cache.insert(handle));
        // Sabotage only this private fixture archive. Opening a directory as
        // an append target fails deterministically, without permissions noise.
        const auto backup = directory / "retained-test-archive";
        std::filesystem::rename(disk->archivePath(), backup);
        std::filesystem::create_directory(disk->archivePath());
        EXPECT_EQ(cache.prepareCapacity(32u), PrefixRamInsertPreparation::Busy);
        EXPECT_THROW(cache.completeInsertPreparation(keyFor(1), 32u), std::runtime_error);
        std::string error;
        EXPECT_FALSE(disk->waitForPersistence(&error));
        EXPECT_FALSE(error.empty());
        EXPECT_THROW(disk->scheduleWrite(handle), std::runtime_error);
        EXPECT_EQ(cache.usedBytes(), 32u);
        EXPECT_EQ(cache.stats().disk_write_failures, 1u);
    }
    std::filesystem::remove_all(directory);
}

/**
 * @test Cache pressure must follow the physical lease, not only LRU keys.
 *
 * The oldest key is still held by a request. Evicting it lowers logical
 * occupancy but does not release PMA capacity; the next unaliased victim must
 * retire before a new archive can be allocated.
 */
TEST(Test__PrefixStateCacheLRU,
     PhysicalAliasPressureEvictsUntilTheArchiveCanBeLeased)
{
    auto authority = prefixAuthority(96u);
    auto ram = RamPrefixStorageBackend::create(
        DeviceId::cpu(), 96u, authority);
    ASSERT_NE(ram, nullptr);
    PrefixStateCache cache(96u, ram);

    auto oldest = ram->allocate(keyFor(0), layoutBytes(32u));
    auto middle = ram->allocate(keyFor(1), layoutBytes(32u));
    auto newest = ram->allocate(keyFor(2), layoutBytes(32u));
    ASSERT_TRUE(oldest.valid());
    ASSERT_TRUE(middle.valid());
    ASSERT_TRUE(newest.valid());
    ASSERT_TRUE(cache.insert(oldest));
    ASSERT_TRUE(cache.insert(middle));
    ASSERT_TRUE(cache.insert(newest));
    PrefixBlockHandle request_alias = oldest;
    oldest = {};
    middle = {};
    newest = {};

    EXPECT_EQ(cache.prepareInsert(keyFor(3), 32u),
              PrefixRamInsertPreparation::Prepared);
    EXPECT_FALSE(cache.isRamResident(keyFor(0)));
    EXPECT_FALSE(cache.isRamResident(keyFor(1)));
    EXPECT_TRUE(cache.isRamResident(keyFor(2)));
    EXPECT_EQ(ram->usedBytes(), 32u);
    auto incoming = ram->allocate(keyFor(3), layoutBytes(32u));
    ASSERT_TRUE(incoming.valid());
    EXPECT_TRUE(cache.insert(incoming));
}

/** @test All request-held aliases yield typed Busy instead of overcommit. */
TEST(Test__PrefixStateCacheLRU,
     PhysicalAliasPressureReportsBusyWhenEveryVictimIsRetained)
{
    auto authority = prefixAuthority(64u);
    auto ram = RamPrefixStorageBackend::create(
        DeviceId::cpu(), 64u, authority);
    ASSERT_NE(ram, nullptr);
    PrefixStateCache cache(64u, ram);

    auto first = ram->allocate(keyFor(0), layoutBytes(32u));
    auto second = ram->allocate(keyFor(1), layoutBytes(32u));
    ASSERT_TRUE(first.valid());
    ASSERT_TRUE(second.valid());
    ASSERT_TRUE(cache.insert(first));
    ASSERT_TRUE(cache.insert(second));
    PrefixBlockHandle retained_first = first;
    PrefixBlockHandle retained_second = second;
    first = {};
    second = {};

    EXPECT_EQ(cache.prepareInsert(keyFor(2), 32u),
              PrefixRamInsertPreparation::Busy);
    EXPECT_EQ(ram->usedBytes(), 0u);
    EXPECT_FALSE(ram->canStore(32u));
    retained_first = {};
    EXPECT_EQ(cache.prepareInsert(keyFor(2), 32u),
              PrefixRamInsertPreparation::Prepared);
    EXPECT_TRUE(ram->allocate(keyFor(2), layoutBytes(32u)).valid());
}

/** @test A physically busy hydration keeps its valid durable disk record. */
TEST(Test__PrefixStateCacheLRU,
     PhysicalAliasPressureDoesNotDiscardVerifiedDiskEntry)
{
    const auto directory = tempDir();
    auto authority = prefixAuthority(64u);
    auto ram = RamPrefixStorageBackend::create(
        DeviceId::cpu(), 64u, authority);
    auto disk = makeDiskBackend(directory, 128u);
    ASSERT_NE(ram, nullptr);
    ASSERT_TRUE(disk->ready()) << disk->initializationError();
    PrefixStateCache cache(32u, ram, disk);

    auto first = ram->allocate(keyFor(0), layoutBytes(32u));
    auto second = ram->allocate(keyFor(1), layoutBytes(32u));
    ASSERT_TRUE(first.valid());
    ASSERT_TRUE(second.valid());
    ASSERT_TRUE(cache.insert(first));
    ASSERT_TRUE(insertAfterArchivePublication(cache, *disk, second));
    ASSERT_TRUE(cache.isDiskResident(keyFor(0)));

    // Both evicted handles remain owned by the request. Hydration can retire
    // logical keys but must not claim a third physical 32-byte allocation.
    EXPECT_FALSE(cache.find(keyFor(0)).has_value());
    ASSERT_TRUE(disk->waitForPersistence());
    cache.publishCompletedPersistence();
    EXPECT_FALSE(cache.find(keyFor(0)).has_value());
    EXPECT_TRUE(cache.isDiskResident(keyFor(0)));
    first = {};
    second = {};

    auto hydrated = cache.find(keyFor(0));
    ASSERT_TRUE(hydrated.has_value());
    EXPECT_TRUE(hydrated->valid());
    ASSERT_TRUE(disk->waitForPersistence());
    std::filesystem::remove_all(directory);
}

TEST(Test__PrefixStateCacheLRU, InsertFindAndTouchUpdatesRecency)
{
    auto backend = std::make_shared<RamPrefixStorageBackend>(128);
    PrefixStateCache cache(128, backend);
    auto a = backend->allocate(keyFor(0), layoutBytes(32));
    auto b = backend->allocate(keyFor(1), layoutBytes(32));

    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(cache.insert(b));
    ASSERT_TRUE(cache.find(a.key).has_value());

    const auto keys = cache.keysMostRecentFirst();
    ASSERT_EQ(keys.size(), 2u);
    EXPECT_EQ(keys[0], a.key);
    EXPECT_EQ(keys[1], b.key);
    EXPECT_EQ(cache.stats().lookups, 1u);
    EXPECT_EQ(cache.stats().hits, 1u);
    EXPECT_EQ(cache.stats().stores, 2u);
}

/**
 * A placement-invalidating fingerprint transition must immediately recover
 * volatile capacity. Persisting stale blocks during that request boundary
 * would charge disk writes and fsync to the next inference.
 */
TEST(
    Test__PrefixStateCacheLRU,
    FingerprintRebaseRetiresVolatileEntriesWithoutWritingDisk)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    constexpr size_t kBlockBytes = 32;
    constexpr uint64_t kOldFingerprint = 0x1111;
    constexpr uint64_t kNewFingerprint = 0x2222;
    auto ram = std::make_shared<RamPrefixStorageBackend>(kBlockBytes * 3);
    auto disk = makeDiskBackend(dir, kBlockBytes * 4);
    ASSERT_TRUE(disk->ready()) << disk->initializationError();
    PrefixStateCache cache(kBlockBytes * 2, ram, disk);

    auto old_a = ram->allocate(
        keyForFingerprint(kOldFingerprint, 0),
        layoutBytes(kBlockBytes));
    auto old_b = ram->allocate(
        keyForFingerprint(kOldFingerprint, 1),
        layoutBytes(kBlockBytes));
    auto old_c = ram->allocate(
        keyForFingerprint(kOldFingerprint, 2),
        layoutBytes(kBlockBytes));
    ASSERT_TRUE(old_a.valid());
    ASSERT_TRUE(old_b.valid());
    ASSERT_TRUE(old_c.valid());
    ASSERT_TRUE(cache.insert(old_a));
    ASSERT_TRUE(cache.insert(old_b));
    ASSERT_TRUE(insertAfterArchivePublication(cache, *disk, old_c));
    ASSERT_EQ(cache.stats().ram_to_disk_demotions, 1u);
    ASSERT_GT(disk->usedBytes(), 0u);
    const size_t durable_bytes_before_rebase = disk->usedBytes();

    const auto transition = cache.rebaseFingerprint(
        kOldFingerprint,
        kNewFingerprint);
    ASSERT_TRUE(transition.has_value());
    EXPECT_TRUE(transition->changed());
    EXPECT_EQ(transition->invalidated_ram_entries, 2u);
    EXPECT_EQ(transition->unindexed_disk_entries, 1u);
    EXPECT_EQ(transition->released_ram_bytes, kBlockBytes * 2);
    EXPECT_EQ(cache.usedBytes(), 0u);
    EXPECT_EQ(ram->usedBytes(), 0u);
    EXPECT_EQ(cache.size(), 0u);
    EXPECT_FALSE(cache.isDiskResident(old_a.key));
    EXPECT_EQ(disk->usedBytes(), durable_bytes_before_rebase)
        << "Rebase must not serialize a delete or compact the durable archive";
    EXPECT_EQ(cache.stats().fingerprint_rebases, 1u);
    EXPECT_EQ(cache.stats().fingerprint_invalidated_ram_entries, 2u);
    EXPECT_EQ(cache.stats().fingerprint_unindexed_disk_entries, 1u);

    auto current = ram->allocate(
        keyForFingerprint(kNewFingerprint, 0),
        layoutBytes(kBlockBytes));
    ASSERT_TRUE(current.valid());
    ASSERT_TRUE(cache.insert(current));
    EXPECT_EQ(cache.stats().ram_to_disk_demotions, 1u)
        << "A new-epoch insert must consume recovered RAM rather than spill a stale epoch";

    ASSERT_TRUE(disk->waitForPersistence());
    cleanup();
}

/**
 * @brief A peer's archive eviction cannot masquerade as this cache's backing.
 *
 * TP participants share one durable budget but keep distinct RAM indexes.
 * Hydrate A, let a peer evict its old disk copy, then demote the still-valid
 * RAM owner. That demotion must publish A again before releasing its source;
 * a stale participant-local disk entry is not a persistence receipt.
 */
TEST(Test__PrefixStateCacheLRU, SharedArchiveEvictionRepersistsResidentOwner)
{
    for (const bool independent_writer : {false, true})
    {
        SCOPED_TRACE(independent_writer ? "independent archive writer" : "shared archive writer");
        const auto directory = tempDir();
        constexpr size_t bytes = 32;
        auto disk = makeDiskBackend(directory, 2 * bytes);
        ASSERT_TRUE(disk->ready()) << disk->initializationError();
        auto peer_disk = independent_writer ? makeDiskBackend(directory, 2 * bytes) : disk;
        ASSERT_TRUE(peer_disk->ready()) << peer_disk->initializationError();
        auto first_ram = std::make_shared<RamPrefixStorageBackend>(2 * bytes);
        auto peer_ram = std::make_shared<RamPrefixStorageBackend>(bytes);
        PrefixStateCache first(2 * bytes, first_ram, disk);
        PrefixStateCache peer(bytes, peer_ram, peer_disk);
        const auto layout = layoutBytes(bytes);
        const auto a = keyForFingerprint(0x1111, 0);
        const auto b = keyForFingerprint(0x1111, 1);
        const auto c = keyForFingerprint(0x1111, 2);
        const auto d = keyForFingerprint(0x1111, 3);
        const auto peer_a = keyForFingerprint(0x2222, 0);
        const auto peer_b = keyForFingerprint(0x2222, 1);
        const auto peer_c = keyForFingerprint(0x2222, 2);
        const auto insert = [&](PrefixStateCache &cache,
                                RamPrefixStorageBackend &ram,
                                const PrefixCacheKey &key)
        {
            cache.completeInsertPreparation(key, PrefixPayloadAllocationPlan::archive(layout));
            auto handle = ram.allocate(key, layout);
            if (!handle.valid())
                return false;
            std::fill_n(static_cast<uint8_t *>(handle.kv_payload), bytes,
                        static_cast<uint8_t>(key.block_index + 37));
            return cache.insert(handle);
        };
        ASSERT_TRUE(insert(first, *first_ram, a));
        ASSERT_TRUE(insert(first, *first_ram, b));
        ASSERT_TRUE(insert(first, *first_ram, c));
        ASSERT_TRUE(findAfterArchivePublication(first, *disk, a).has_value());
        ASSERT_TRUE(first.isRamResident(a));
        ASSERT_TRUE(first.isDiskResident(a));

        ASSERT_TRUE(insert(peer, *peer_ram, peer_a));
        ASSERT_TRUE(insert(peer, *peer_ram, peer_b));
        ASSERT_TRUE(insert(peer, *peer_ram, peer_c));
        ASSERT_TRUE(peer_disk->waitForPersistence());
        std::string error;
        ASSERT_FALSE(peer_disk->beginHydration(a, layout, &error).has_value())
            << "The peer must evict A's durable copy before this test demotes RAM";
        ASSERT_TRUE(first.find(c).has_value()); // A is now the next RAM victim.
        ASSERT_TRUE(insert(first, *first_ram, d));
        ASSERT_TRUE(disk->waitForPersistence());

        auto restored = findAfterArchivePublication(first, *disk, a);
        ASSERT_TRUE(restored.has_value())
            << "Participant-local disk metadata must not discard the only A payload";
        ASSERT_TRUE(restored->kv_payload);
        const auto *data = static_cast<const uint8_t *>(restored->kv_payload);
        EXPECT_TRUE(std::all_of(data, data + bytes, [](uint8_t value) { return value == 37; }));
        restored.reset();
        ASSERT_TRUE(disk->waitForPersistence());
        std::filesystem::remove_all(directory);
    }
}

/**
 * @brief A retain acquired during persistence cannot reuse that old receipt later.
 *
 * The native file lock holds the selected write while a legacy cache retain
 * arrives. After publication a separate archive writer removes that backing.
 * Releasing the retain must start a new demotion and preserve the original bytes.
 */
TEST(Test__PrefixStateCacheLRU, SharedArchiveRetainedVictimNeedsFreshPublication)
{
    const auto directory = tempDir();
    {
        constexpr size_t bytes = 32;
        const auto layout = layoutBytes(bytes);
        auto disk = makeDiskBackend(directory, bytes);
        auto peer = makeDiskBackend(directory, bytes);
        auto ram = std::make_shared<RamPrefixStorageBackend>(bytes);
        PrefixStateCache cache(bytes, ram, disk);
        auto source = ram->allocate(keyFor(0), layout);
        ASSERT_TRUE(source.valid());
        std::fill(source.kv_storage->begin(), source.kv_storage->end(), 0x7a);
        ASSERT_TRUE(cache.insert(source));
        source = {};
        HeldArchiveLock held(disk->archivePath());
        ASSERT_TRUE(held.valid());
        EXPECT_EQ(cache.prepareCapacity(bytes), PrefixRamInsertPreparation::Busy);
        ASSERT_TRUE(cache.retain(keyFor(0)));
        held.release();
        ASSERT_TRUE(disk->waitForPersistence());
        cache.publishCompletedPersistence();
        EXPECT_TRUE(cache.isRamResident(keyFor(0)));
        EXPECT_EQ(cache.stats().evictions, 0u);
        RamPrefixStorageBackend peer_ram(bytes);
        auto competing = peer_ram.allocate(keyForFingerprint(0x2222, 0), layout);
        ASSERT_TRUE(competing.valid());
        ASSERT_TRUE(peer->writeBlock(competing, nullptr));
        ASSERT_TRUE(cache.release(keyFor(0)));
        cache.completeInsertPreparation(keyFor(1), PrefixPayloadAllocationPlan::archive(layout));
        EXPECT_FALSE(cache.isRamResident(keyFor(0)));
        EXPECT_EQ(cache.stats().evictions, 1u);
        EXPECT_TRUE(ram->canStore(bytes));
        PrefixBlockHandle restored;
        ASSERT_TRUE(disk->readBlock(keyFor(0), layout, &restored));
        ASSERT_TRUE(restored.kv_storage);
        EXPECT_TRUE(std::all_of(restored.kv_storage->begin(), restored.kv_storage->end(),
                                [](uint8_t value) { return value == 0x7a; }));
    }
    std::filesystem::remove_all(directory);
}

/** A retained legacy lease rejects the complete rebase before any tier moves. */
TEST(Test__PrefixStateCacheLRU, BusyFingerprintRebaseIsAtomic)
{
    constexpr size_t kBlockBytes = 32;
    constexpr uint64_t kOldFingerprint = 0x1111;
    constexpr uint64_t kNewFingerprint = 0x2222;
    auto ram = std::make_shared<RamPrefixStorageBackend>(kBlockBytes * 2);
    PrefixStateCache cache(kBlockBytes * 2, ram);

    auto old = ram->allocate(
        keyForFingerprint(kOldFingerprint, 0),
        layoutBytes(kBlockBytes));
    auto current = ram->allocate(
        keyForFingerprint(kNewFingerprint, 0),
        layoutBytes(kBlockBytes));
    ASSERT_TRUE(old.valid());
    ASSERT_TRUE(current.valid());
    ASSERT_TRUE(cache.insert(old));
    ASSERT_TRUE(cache.insert(current));
    ASSERT_TRUE(cache.retain(old.key));

    EXPECT_FALSE(cache.rebaseFingerprint(
        kOldFingerprint,
        kNewFingerprint));
    EXPECT_TRUE(cache.isRamResident(old.key));
    EXPECT_TRUE(cache.isRamResident(current.key));
    EXPECT_EQ(cache.stats().fingerprint_rebases, 0u);

    EXPECT_TRUE(cache.release(old.key));
    ASSERT_TRUE(cache.rebaseFingerprint(
        kOldFingerprint,
        kNewFingerprint));
    EXPECT_FALSE(cache.isRamResident(old.key));
    EXPECT_TRUE(cache.isRamResident(current.key));
}

/**
 * @brief A prior terminal partial block must extend a later multi-turn prompt.
 *
 * Recurrent models store their continuation state on the terminal block. The
 * next request can append tokens inside that same logical block, so exact
 * full-width lookup would miss the only byte-correct continuation checkpoint.
 */
TEST(Test__PrefixStateCacheLRU, LongestTokenPrefixSelectsTerminalPartialBlock)
{
    constexpr uint64_t kFingerprint = 0xbeef;
    constexpr uint64_t kParentHash = 0x1234;
    constexpr int kBlockIndex = 2;
    constexpr int kTokenStart = 128;
    auto backend = std::make_shared<RamPrefixStorageBackend>(256);
    auto cache = std::make_shared<PrefixStateCache>(256, backend);

    const std::vector<int32_t> terminal_tokens = {41, 42, 43};
    const PrefixCacheKey terminal_key = makePrefixCacheKey(
        kFingerprint,
        kParentHash,
        kBlockIndex,
        kTokenStart,
        terminal_tokens);
    auto terminal = backend->allocate(terminal_key, layoutBytes(32));
    terminal.has_hybrid_state = true;
    ASSERT_TRUE(cache->insert(terminal));

    const std::vector<int32_t> longer_block = {41, 42, 43, 44, 45};
    auto match = cache->beginLookup()->selectLongest(
        kFingerprint,
        kParentHash,
        kBlockIndex,
        kTokenStart,
        longer_block);
    ASSERT_TRUE(match.has_value());
    EXPECT_EQ(match->key, terminal_key);
    EXPECT_TRUE(match->has_hybrid_state);
    EXPECT_EQ(cache->stats().lookups, 1u);
    EXPECT_EQ(cache->stats().hits, 1u);
    EXPECT_EQ(cache->stats().misses, 0u);

    const PrefixCacheKey full_key = makePrefixCacheKey(
        kFingerprint,
        kParentHash,
        kBlockIndex,
        kTokenStart,
        longer_block);
    auto full = backend->allocate(full_key, layoutBytes(32));
    ASSERT_TRUE(cache->insert(full));
    match = cache->beginLookup()->selectLongest(
        kFingerprint,
        kParentHash,
        kBlockIndex,
        kTokenStart,
        longer_block);
    ASSERT_TRUE(match.has_value());
    EXPECT_EQ(match->key, full_key)
        << "An exact block must take precedence over an older terminal prefix";
}

/** @test Coordinated endpoints own exact bytes across cold/mixed tiers and every archive organization. */
TEST(Test__PrefixStateCacheLRU, SelectedRestoreOwnsOnlyCoordinatedReadSet)
{
    for (const int organization : {0, 1, 2}) // Attention, hybrid, recurrent-only PP.
        for (const bool mtp : {false, true})
            for (const bool resident : {false, true})
                for (const int selected_count : {2, 4})
                {
                    SCOPED_TRACE(::testing::Message() << "organization=" << organization
                        << " mtp=" << mtp << " resident=" << resident << " selected=" << selected_count);
                    const auto directory = tempDir();
                    {
                        auto layout = layoutBytes(16);
                        if (organization == 2)
                        {
                            layout.fa_layers = 0;
                            layout.bytes_per_fa_layer_k = layout.bytes_per_fa_layer_v = 0;
                        }
                        layout.gdn_layers = organization != 0 ? 1 : 0;
                        layout.total_layers = layout.fa_layers + layout.gdn_layers;
                        layout.includes_hybrid_state = organization != 0;
                        layout.hybrid_host_state_bytes = layout.hybrid_state_bytes = organization != 0 ? 32u : 0u;
                        layout.includes_mtp_state = mtp;
                        layout.mtp_kv_bytes = mtp ? 16u : 0u;
                        layout.includes_terminal_hidden = layout.includes_terminal_logits = true;
                        layout.terminal_hidden_bytes = layout.terminal_logits_bytes = 8u;
                        constexpr size_t runtime_bytes = 8u;
                        const size_t rows = layout.faKVBytes() + layout.mtpKVBytes();
                        const size_t full = layout.totalBytes() + runtime_bytes;
                        const size_t capacity = rows * 3u + full;
                        auto ram = RamPrefixStorageBackend::create(DeviceId::cpu(), capacity, prefixAuthority(capacity));
                        ASSERT_TRUE(ram);
                        auto disk = makeDiskBackend(directory, full * 4u);
                        auto cache = std::make_shared<PrefixStateCache>(capacity, ram, disk);
                        RamPrefixStorageBackend staging(full);
                        const auto fill = [&](PrefixBlockHandle &block, RamPrefixStorageBackend &backend, int index)
                        {
                            if (block.kv_storage) std::fill(block.kv_storage->begin(), block.kv_storage->end(), index + 1);
                            if (block.mtp_storage) std::fill(block.mtp_storage->begin(), block.mtp_storage->end(), index + 11);
                            if (block.hybrid_storage) std::fill(block.hybrid_storage->begin(), block.hybrid_storage->end(), index + 21);
                            std::fill(block.terminal_hidden_storage->begin(), block.terminal_hidden_storage->end(), index + 31);
                            std::fill(block.terminal_logits_storage->begin(), block.terminal_logits_storage->end(), index + 41);
                            block.has_hybrid_state = layout.includes_hybrid_state;
                            block.has_terminal_hidden = block.has_terminal_logits = true;
                            return backend.attachModelRuntimeState(&block,
                                std::make_shared<std::vector<uint8_t>>(runtime_bytes, index + 51));
                        };
                        for (int index = 0; index < 4; ++index)
                        {
                            auto block = staging.allocate(keyFor(index), layout);
                            ASSERT_TRUE(fill(block, staging, index));
                            std::string error;
                            ASSERT_TRUE(disk->writeBlock(block, nullptr, nullptr, &error)) << error;
                            ASSERT_TRUE(staging.release(block));
                        }
                        ASSERT_TRUE(cache->installDiscoveredDiskEntries(0xbeef, layout));
                        if (resident)
                        {
                            const auto block = cache->find(keyFor(1));
                            ASSERT_TRUE(block);
                        }
                        const auto before = cache->stats();
                        PrefixLookupResult admission;
                        admission.supported = admission.cache_enabled = true;
                        admission.fingerprint_key = 0xbeef;
                        admission.block_size = 1;
                        admission.cached_tokens = 4;
                        admission.has_terminal_hidden = admission.has_terminal_logits = true;
                        admission.payload_plan = cache->beginLookup();
                        for (int index = 0; index < 4; ++index)
                        {
                            const auto block = admission.payload_plan->selectLongest(0xbeef, 0, index, index, {index});
                            ASSERT_TRUE(block);
                            EXPECT_EQ(block->kv_payload, nullptr);
                            admission.blocks.push_back(*block);
                        }
                        EXPECT_EQ(cache->stats().lookups - before.lookups, 4u);
                        EXPECT_EQ(cache->stats().hits - before.hits, 4u);
                        EXPECT_EQ(cache->stats().disk_hydrations, before.disk_hydrations);
                        EXPECT_EQ(admission.payload_plan->boundedTokenCount(admission.blocks), 4);
                        auto selected = admission.clampedTo(selected_count);
                        const bool recurrent_only = layout.organization() == PrefixPayloadOrganization::RecurrentCheckpoint;
                        if (recurrent_only)
                            selected.blocks.erase(selected.blocks.begin(), selected.blocks.end() - 1);
                        auto copied_metadata = selected;
                        // Selection may copy metadata through TP/PP coordination.
                        // Only captured source geometry can authorize omitting rows.
                        copied_metadata.blocks.back().layout.fa_layers = 0;
                        copied_metadata.blocks.back().layout.gdn_layers = 1;
                        copied_metadata.blocks.back().layout.includes_mtp_state = false;
                        auto payloads = copied_metadata.materializeRestoreBlocks();
                        ASSERT_EQ(payloads.size(), recurrent_only ? 1u : selected_count);
                        const int first = recurrent_only ? selected_count - 1 : 0;
                        for (size_t position = 0; position < payloads.size(); ++position)
                        {
                            const auto &block = payloads[position];
                            const int index = first + static_cast<int>(position);
                            EXPECT_EQ(block.key, keyFor(index));
                            if (block.kv_storage) EXPECT_TRUE(std::all_of(block.kv_storage->begin(), block.kv_storage->end(),
                                [index](uint8_t value) { return value == index + 1; }));
                            if (block.mtp_storage) EXPECT_TRUE(std::all_of(block.mtp_storage->begin(), block.mtp_storage->end(),
                                [index](uint8_t value) { return value == index + 11; }));
                            const bool terminal = position + 1u == payloads.size();
                            EXPECT_EQ(block.has_hybrid_state, terminal && layout.includes_hybrid_state);
                            EXPECT_EQ(block.has_terminal_hidden, terminal);
                            EXPECT_EQ(block.has_terminal_logits, terminal);
                            EXPECT_EQ(block.has_model_runtime_state, terminal);
                        }
                        const auto &terminal = payloads.back();
                        EXPECT_EQ(terminal.terminal_hidden_storage->front(), selected_count - 1 + 31);
                        EXPECT_EQ(terminal.terminal_logits_storage->front(), selected_count - 1 + 41);
                        EXPECT_EQ(terminal.model_runtime_state_storage->front(), selected_count - 1 + 51);
                        if (terminal.hybrid_storage)
                            EXPECT_EQ(terminal.hybrid_storage->front(), selected_count - 1 + 21);
                        EXPECT_LE(ram->budgetBytes() - ram->availableAllocationBytes(), capacity);
                        EXPECT_THROW(selected.clampedTo(selected_count - 1).materializeRestoreBlocks(), std::logic_error);

                        std::weak_ptr<std::vector<uint8_t>> earlier_rows;
                        if (payloads.size() > 1u)
                            earlier_rows = mtp ? payloads.front().mtp_storage : payloads.front().kv_storage;
                        ASSERT_TRUE(cache->clear());
                        payloads.clear();
                        auto witness = selected.forHarvest(selected_count);
                        ASSERT_EQ(witness.blocks.size(), 1u);
                        EXPECT_EQ(witness.blocks.front().key, keyFor(selected_count - 1));
                        EXPECT_FALSE(witness.payload_plan);
                        EXPECT_TRUE(earlier_rows.expired())
                            << "The original lookup alias must not pin consumed rows after harvest";
                        EXPECT_EQ(admission.forHarvest(selected_count).blocks.front().key, witness.blocks.front().key);
                        EXPECT_THROW(selected.materializeRestoreBlocks(), std::logic_error);
                    }
                    std::filesystem::remove_all(directory);
                }
}

/** @test A cache window larger than RAM is shortened before any disk read or live-state mutation. */
TEST(Test__PrefixStateCacheLRU, SelectedRestoreBoundsActualRowsAndEndpointByTierCapacity)
{
    const auto directory = tempDir();
    {
        auto layout = layoutBytes(16);
        layout.includes_hybrid_state = true;
        layout.gdn_layers = 1;
        layout.total_layers = 2;
        layout.hybrid_host_state_bytes = layout.hybrid_state_bytes = 32;
        const size_t capacity = layout.totalBytes() + layout.faKVBytes();
        auto ram = RamPrefixStorageBackend::create(DeviceId::cpu(), capacity, prefixAuthority(capacity));
        auto disk = makeDiskBackend(directory, layout.totalBytes() * 4u);
        auto cache = std::make_shared<PrefixStateCache>(capacity, ram, disk);
        RamPrefixStorageBackend staging(layout.totalBytes());
        for (int index = 0; index < 4; ++index)
        {
            auto source = staging.allocate(keyFor(index), layout);
            source.has_hybrid_state = true;
            ASSERT_TRUE(disk->writeBlock(source, nullptr));
            ASSERT_TRUE(staging.release(source));
        }
        ASSERT_TRUE(cache->installDiscoveredDiskEntries(0xbeef, layout));
        PrefixLookupResult hit;
        hit.cached_tokens = 4;
        hit.block_size = 1;
        hit.payload_plan = cache->beginLookup();
        for (int index = 0; index < 4; ++index)
        {
            auto block = hit.payload_plan->selectLongest(0xbeef, 0, index, index, {index});
            ASSERT_TRUE(block);
            hit.blocks.push_back(*block);
        }
        EXPECT_EQ(hit.payload_plan->boundedTokenCount(hit.blocks), 2);
        EXPECT_EQ(cache->stats().disk_hydrations, 0u);
        EXPECT_THROW(hit.materializeRestoreBlocks(), std::invalid_argument);
        EXPECT_EQ(cache->stats().disk_hydrations, 0u);
        EXPECT_EQ(ram->availableAllocationBytes(), capacity);
        const auto selected = hit.clampedTo(hit.payload_plan->boundedTokenCount(hit.blocks));
        const auto payloads = selected.materializeRestoreBlocks();
        ASSERT_EQ(payloads.size(), 2u);
        EXPECT_EQ(payloads.front().total_bytes + payloads.back().total_bytes, capacity);
        EXPECT_EQ(cache->stats().disk_hydrations, 2u);
        EXPECT_EQ(ram->availableAllocationBytes(), 0u);
    }
    std::filesystem::remove_all(directory);
}

/**
 * @brief Rich terminal replacement must retire stale RAM, VRAM, and disk copies.
 *
 * A block first observed inside a longer prompt carries ordinary KV payloads.
 * If a later request ends on that same key, harvest republishes it with MTP and
 * terminal state. The old disk record must not survive and reappear after the
 * richer resident block is demoted and hydrated again.
 */
TEST(Test__PrefixStateCacheLRU, PreparedReplacementCannotResurrectStaleDiskPayload)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    constexpr size_t kCacheBudget = 64;
    auto ram = std::make_shared<RamPrefixStorageBackend>(256);
    auto disk = makeDiskBackend(dir, 256);
    ASSERT_TRUE(disk->ready()) << disk->initializationError();
    PrefixStateCache cache(kCacheBudget, ram, disk);

    const PrefixCacheKey replaced_key = keyFor(7);
    auto old = ram->allocate(replaced_key, layoutBytes(32));
    ASSERT_TRUE(old.valid());
    std::fill(old.kv_storage->begin(), old.kv_storage->end(), 0x11);
    ASSERT_TRUE(cache.insert(old));

    auto pressure = ram->allocate(keyFor(8), layoutBytes(48));
    ASSERT_TRUE(pressure.valid());
    ASSERT_TRUE(insertAfterArchivePublication(cache, *disk, pressure));
    ASSERT_TRUE(cache.isDiskResident(replaced_key));

    auto rich_layout = layoutBytes(32);
    rich_layout.includes_mtp_state = true;
    rich_layout.mtp_kv_bytes = 16;
    ASSERT_EQ(cache.prepareInsert(replaced_key, rich_layout.totalBytes()),
              PrefixRamInsertPreparation::Busy);
    ASSERT_TRUE(disk->waitForPersistence());
    cache.publishCompletedPersistence();
    ASSERT_EQ(cache.prepareInsert(replaced_key, rich_layout.totalBytes()),
              PrefixRamInsertPreparation::Prepared);
    EXPECT_FALSE(cache.contains(replaced_key));

    auto rich = ram->allocate(replaced_key, rich_layout);
    ASSERT_TRUE(rich.valid());
    std::fill(rich.kv_storage->begin(), rich.kv_storage->end(), 0x22);
    ASSERT_NE(rich.mtp_storage, nullptr);
    std::fill(rich.mtp_storage->begin(), rich.mtp_storage->end(), 0x33);
    ASSERT_TRUE(cache.insert(rich));

    auto second_pressure = ram->allocate(keyFor(9), layoutBytes(32));
    ASSERT_TRUE(second_pressure.valid());
    ASSERT_TRUE(insertAfterArchivePublication(cache, *disk, second_pressure));
    ASSERT_TRUE(cache.isDiskResident(replaced_key));

    const auto hydrated = findAfterArchivePublication(cache, *disk, replaced_key);
    ASSERT_TRUE(hydrated.has_value());
    EXPECT_TRUE(hydrated->layout.includes_mtp_state);
    ASSERT_NE(hydrated->kv_storage, nullptr);
    ASSERT_NE(hydrated->mtp_storage, nullptr);
    EXPECT_TRUE(std::all_of(
        hydrated->kv_storage->begin(),
        hydrated->kv_storage->end(),
        [](uint8_t value) { return value == 0x22; }));
    EXPECT_TRUE(std::all_of(
        hydrated->mtp_storage->begin(),
        hydrated->mtp_storage->end(),
        [](uint8_t value) { return value == 0x33; }));

    ASSERT_TRUE(disk->waitForPersistence());
    cleanup();
}

TEST(Test__PrefixStateCacheLRU, EvictsLeastRecentlyUsedBlocksToFitBudget)
{
    auto backend = std::make_shared<RamPrefixStorageBackend>(96);
    PrefixStateCache cache(64, backend);
    auto a = backend->allocate(keyFor(0), layoutBytes(32));
    auto b = backend->allocate(keyFor(1), layoutBytes(32));
    auto c = backend->allocate(keyFor(2), layoutBytes(32));
    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(cache.insert(b));
    ASSERT_TRUE(cache.find(a.key).has_value()); // b becomes LRU.
    ASSERT_TRUE(cache.insert(c));

    EXPECT_TRUE(cache.contains(a.key));
    EXPECT_FALSE(cache.contains(b.key));
    EXPECT_TRUE(cache.contains(c.key));
    EXPECT_EQ(cache.usedBytes(), 64u);
    EXPECT_EQ(cache.stats().evictions, 1u);
}

/**
 * @brief Prove copied handles retain payload storage after metadata eviction.
 *
 * GPU restore uses this ownership property to avoid a contended cache lease:
 * lookup returns a cheap shared-owner copy, the cache remains free to evict its
 * metadata entry, and an event-retirement queue keeps that copied handle alive
 * until the asynchronous restore has stopped reading the payload.
 */
TEST(Test__PrefixStateCacheLRU, CopiedHandleOwnsPayloadAfterMetadataEviction)
{
    constexpr size_t kBlockBytes = 32;
    auto backend =
        std::make_shared<RamPrefixStorageBackend>(kBlockBytes * 3);
    PrefixStateCache cache(kBlockBytes * 2, backend);

    auto a = backend->allocate(keyFor(0), layoutBytes(kBlockBytes));
    auto b = backend->allocate(keyFor(1), layoutBytes(kBlockBytes));
    auto c = backend->allocate(keyFor(2), layoutBytes(kBlockBytes));
    ASSERT_TRUE(a.valid());
    ASSERT_TRUE(b.valid());
    ASSERT_TRUE(c.valid());
    ASSERT_NE(a.kv_storage, nullptr);
    std::fill(a.kv_storage->begin(), a.kv_storage->end(), uint8_t{0xa5});

    const PrefixCacheKey a_key = a.key;
    const PrefixCacheKey b_key = b.key;
    ASSERT_TRUE(cache.insert(std::move(a)));
    ASSERT_TRUE(cache.insert(std::move(b)));

    auto held_restore_source = cache.find(a_key);
    ASSERT_TRUE(held_restore_source.has_value());
    ASSERT_NE(held_restore_source->kv_storage, nullptr);
    std::weak_ptr<std::vector<uint8_t>> payload_lifetime =
        held_restore_source->kv_storage;

    /*
     * Touch B after obtaining A's restore handle so A becomes the metadata
     * LRU. Inserting C evicts A from the cache while the copied handle remains
     * the sole owner used by the simulated asynchronous restore.
     */
    ASSERT_TRUE(cache.find(b_key).has_value());
    ASSERT_TRUE(cache.insert(std::move(c)));
    EXPECT_FALSE(cache.contains(a_key));
    EXPECT_FALSE(payload_lifetime.expired());
    ASSERT_NE(held_restore_source->kv_storage, nullptr);
    EXPECT_TRUE(std::all_of(
        held_restore_source->kv_storage->begin(),
        held_restore_source->kv_storage->end(),
        [](uint8_t byte)
        {
            return byte == uint8_t{0xa5};
        }));

    held_restore_source.reset();
    EXPECT_TRUE(payload_lifetime.expired())
        << "The copied restore handle, not cache metadata or a manual lease, "
           "must own the evicted payload";
}

TEST(Test__PrefixStateCacheLRU, RetainedBlocksAreNotEvicted)
{
    auto backend = std::make_shared<RamPrefixStorageBackend>(96);
    PrefixStateCache cache(64, backend);
    auto a = backend->allocate(keyFor(0), layoutBytes(32));
    auto b = backend->allocate(keyFor(1), layoutBytes(32));
    auto c = backend->allocate(keyFor(2), layoutBytes(32));
    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(cache.insert(b));
    ASSERT_TRUE(cache.retain(a.key));
    ASSERT_TRUE(cache.insert(c)); // b can be evicted, a cannot.

    EXPECT_TRUE(cache.contains(a.key));
    EXPECT_FALSE(cache.contains(b.key));
    EXPECT_TRUE(cache.contains(c.key));
    EXPECT_FALSE(cache.erase(a.key));
    EXPECT_TRUE(cache.release(a.key));
    EXPECT_TRUE(cache.erase(a.key));
}

TEST(Test__PrefixStateCacheLRU, TooLargeBlockIsRejected)
{
    auto backend = std::make_shared<RamPrefixStorageBackend>(32);
    PrefixStateCache cache(32, backend);
    PrefixBlockHandle handle;
    handle.key = keyFor(99);
    handle.layout = layoutBytes(64);
    handle.tier = PrefixStorageTier::Ram;
    handle.total_bytes = 64;
    EXPECT_FALSE(cache.insert(handle));
}

TEST(Test__PrefixStateCacheLRU, RecordsRequestLevelStatsAndResidentPayloadBytes)
{
    auto backend = std::make_shared<RamPrefixStorageBackend>(256);
    PrefixStateCache cache(256, backend);
    auto layout = layoutBytes(32);
    layout.includes_hybrid_state = true;
    layout.hybrid_state_bytes = 16;
    layout.includes_mtp_state = true;
    layout.mtp_kv_bytes = 8;

    auto handle = backend->allocate(keyFor(3), layout);
    ASSERT_TRUE(cache.insert(handle));

    cache.recordRequestLookup(/*requested_tokens=*/5,
                              /*matched_tokens=*/3,
                              /*matched_blocks=*/1);
    cache.recordTerminalStateHit();

    EXPECT_EQ(cache.stats().partial_hits, 1u);
    EXPECT_EQ(cache.stats().matched_tokens, 3u);
    EXPECT_EQ(cache.stats().matched_blocks, 1u);
    EXPECT_EQ(cache.stats().terminal_state_hits, 1u);
    EXPECT_EQ(cache.stats().hybrid_state_bytes, 16u);
    EXPECT_EQ(cache.stats().mtp_state_bytes, 8u);

    ASSERT_TRUE(cache.erase(handle.key));
    EXPECT_EQ(cache.stats().hybrid_state_bytes, 0u);
    EXPECT_EQ(cache.stats().mtp_state_bytes, 0u);
}

TEST(Test__PrefixStateCacheLRU, ClearReleasesResidentEntriesAndPayloadAccounting)
{
    auto backend = std::make_shared<RamPrefixStorageBackend>(256);
    PrefixStateCache cache(256, backend);
    auto layout = layoutBytes(32);
    layout.includes_hybrid_state = true;
    layout.hybrid_state_bytes = 16;
    layout.includes_mtp_state = true;
    layout.mtp_kv_bytes = 8;

    auto a = backend->allocate(keyFor(0), layout);
    auto b = backend->allocate(keyFor(1), layout);
    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(cache.insert(b));
    ASSERT_EQ(cache.usedBytes(), 112u);
    ASSERT_EQ(cache.stats().ram_bytes, 112u);
    ASSERT_EQ(cache.stats().hybrid_state_bytes, 32u);
    ASSERT_EQ(cache.stats().mtp_state_bytes, 16u);

    ASSERT_TRUE(cache.clear());

    EXPECT_EQ(cache.size(), 0u);
    EXPECT_EQ(cache.usedBytes(), 0u);
    EXPECT_EQ(cache.stats().ram_bytes, 0u);
    EXPECT_EQ(cache.stats().hybrid_state_bytes, 0u);
    EXPECT_EQ(cache.stats().mtp_state_bytes, 0u);
    EXPECT_FALSE(cache.contains(a.key));
    EXPECT_TRUE(backend->canStore(256))
        << "clear should release resident allocations back to the RAM backend";
}

TEST(Test__PrefixStateCacheLRU, ClearRefusesRetainedResidentEntries)
{
    auto backend = std::make_shared<RamPrefixStorageBackend>(128);
    PrefixStateCache cache(128, backend);
    auto handle = backend->allocate(keyFor(4), layoutBytes(32));
    ASSERT_TRUE(cache.insert(handle));
    ASSERT_TRUE(cache.retain(handle.key));

    EXPECT_FALSE(cache.clear());
    EXPECT_EQ(cache.size(), 1u);
    EXPECT_EQ(cache.usedBytes(), 32u);
    EXPECT_TRUE(cache.contains(handle.key));

    ASSERT_TRUE(cache.release(handle.key));
    EXPECT_TRUE(cache.clear());
    EXPECT_EQ(cache.size(), 0u);
}

TEST(Test__PrefixStateCacheLRU, EvictedBlockPersistsToDiskAndHydratesOnFind)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    auto ram = std::make_shared<RamPrefixStorageBackend>(128);
    auto disk = makeDiskBackend(dir, 128);
    ASSERT_TRUE(disk->ready()) << disk->initializationError();
    PrefixStateCache cache(64, ram, disk);
    auto a = ram->allocate(keyFor(0), layoutBytes(32));
    auto b = ram->allocate(keyFor(1), layoutBytes(32));
    auto c = ram->allocate(keyFor(2), layoutBytes(32));
    ASSERT_TRUE(a.valid());
    ASSERT_TRUE(b.valid());
    ASSERT_TRUE(c.valid());
    std::fill(a.kv_storage->begin(), a.kv_storage->end(), 0xa5);

    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(cache.insert(b));
    ASSERT_TRUE(insertAfterArchivePublication(cache, *disk, c));

    EXPECT_TRUE(cache.contains(a.key))
        << "disk-resident blocks remain addressable after RAM eviction";
    EXPECT_EQ(cache.usedBytes(), 64u);
    EXPECT_EQ(cache.stats().evictions, 1u);
    EXPECT_EQ(cache.stats().disk_bytes, 32u);

    auto hydrated = findAfterArchivePublication(cache, *disk, a.key);
    ASSERT_TRUE(hydrated.has_value());
    ASSERT_NE(hydrated->kv_storage, nullptr);
    EXPECT_EQ(hydrated->tier, PrefixStorageTier::Ram);
    EXPECT_EQ(*hydrated->kv_storage, *a.kv_storage);
    EXPECT_EQ(cache.stats().disk_hydrations, 1u);
    EXPECT_EQ(cache.stats().promotions, 1u);
    EXPECT_GE(cache.stats().disk_bytes, 32u);

    ASSERT_TRUE(disk->waitForPersistence());
    cleanup();
}

/**
 * A one-block RAM tier and one-block disk tier require a true swap: persisting
 * the RAM victim logically evicts the record being promoted.  The verified
 * archive ticket must retain the immutable append record without allocating a
 * second full RAM block, then hydrate it after the victim owns disk capacity.
 */
TEST(Test__PrefixStateCacheLRU,
     CapacityOneTiersSwapThroughVerifiedArchiveOffsets)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    constexpr size_t kBlockBytes = 32;
    auto ram = std::make_shared<RamPrefixStorageBackend>(kBlockBytes * 2);
    auto disk = makeDiskBackend(dir, kBlockBytes);
    ASSERT_TRUE(disk->ready()) << disk->initializationError();
    PrefixStateCache cache(kBlockBytes, ram, disk);

    auto a = ram->allocate(keyFor(0), layoutBytes(kBlockBytes));
    auto b = ram->allocate(keyFor(1), layoutBytes(kBlockBytes));
    ASSERT_TRUE(a.valid());
    ASSERT_TRUE(b.valid());
    std::fill(a.kv_storage->begin(), a.kv_storage->end(), 0xa5);
    const auto expected_a = *a.kv_storage;

    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(insertAfterArchivePublication(cache, *disk, b));
    ASSERT_TRUE(cache.isDiskResident(a.key));
    ASSERT_TRUE(cache.isRamResident(b.key));

    // Verification retains A's inode before B's asynchronous put evicts A.
    // The same lookup must complete its required victim and consume that exact
    // ticket, not return a manufactured miss or allocate full-size staging.
    const auto promoted = cache.find(a.key);
    ASSERT_TRUE(promoted.has_value());
    ASSERT_NE(promoted->kv_storage, nullptr);
    EXPECT_EQ(*promoted->kv_storage, expected_a);
    EXPECT_TRUE(cache.isRamResident(a.key));
    EXPECT_FALSE(cache.isDiskResident(a.key));
    EXPECT_FALSE(cache.isRamResident(b.key));
    EXPECT_TRUE(cache.isDiskResident(b.key));
    EXPECT_EQ(cache.stats().disk_hydrations, 1u);
    EXPECT_EQ(cache.stats().ram_to_disk_demotions, 2u);
    ASSERT_TRUE(disk->waitForPersistence());
    cleanup();
}

/** @test A selected disk restore completes its own victim, rather than reporting a miss. */
TEST(Test__PrefixStateCacheLRU, AsyncArchiveSelectedHydrationCompletesPendingSwap)
{
    const auto directory = tempDir();
    {
        auto disk = makeDiskBackend(directory, 32u);
        auto ram = RamPrefixStorageBackend::create(
            DeviceId::cpu(), 32u, prefixAuthority(32u));
        ASSERT_TRUE(disk->ready());
        ASSERT_NE(ram, nullptr);
        PrefixStateCache cache(32u, ram, disk);
        auto original = ram->allocate(keyFor(0), layoutBytes(32u));
        ASSERT_TRUE(original.valid());
        std::fill(original.kv_storage->begin(), original.kv_storage->end(), 0xa5);
        ASSERT_TRUE(cache.insert(std::move(original)));
        ASSERT_NO_THROW(cache.completeInsertPreparation(keyFor(1), 32u));
        auto victim = ram->allocate(keyFor(1), layoutBytes(32u));
        ASSERT_TRUE(victim.valid());
        ASSERT_TRUE(cache.insert(std::move(victim)));
        ASSERT_TRUE(cache.isDiskResident(keyFor(0)));
        const auto old_misses = cache.stats().misses;
        const auto restored = cache.find(keyFor(0));
        ASSERT_TRUE(restored.has_value());
        EXPECT_EQ(*restored->kv_storage, std::vector<uint8_t>(32u, 0xa5));
        EXPECT_EQ(cache.stats().misses, old_misses);
        EXPECT_EQ(cache.usedBytes(), 32u);
        EXPECT_TRUE(cache.isDiskResident(keyFor(1)));
        EXPECT_FALSE(cache.isRamResident(keyFor(1)));
        EXPECT_EQ(ram->availableAllocationBytes(), 0u)
            << "The verified inode is not a second full-size RAM allocation";
    }
    std::filesystem::remove_all(directory);
}

TEST(Test__PrefixStateCacheLRU, RepeatedPromotionDemotionAndBottomTierEvictionAreExact)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    constexpr size_t kBlockBytes = 32;
    auto ram = std::make_shared<RamPrefixStorageBackend>(kBlockBytes * 4);
    auto disk = makeDiskBackend(dir, kBlockBytes * 2);
    ASSERT_TRUE(disk->ready()) << disk->initializationError();
    PrefixStateCache cache(kBlockBytes * 2, ram, disk);

    auto a = ram->allocate(keyFor(0), layoutBytes(kBlockBytes));
    auto b = ram->allocate(keyFor(1), layoutBytes(kBlockBytes));
    auto c = ram->allocate(keyFor(2), layoutBytes(kBlockBytes));
    ASSERT_TRUE(a.valid());
    ASSERT_TRUE(b.valid());
    ASSERT_TRUE(c.valid());

    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(cache.insert(b));
    ASSERT_TRUE(cache.find(a.key).has_value()); // B is the oldest RAM block.
    ASSERT_TRUE(insertAfterArchivePublication(cache, *disk, c));

    EXPECT_TRUE(cache.isRamResident(a.key));
    EXPECT_FALSE(cache.isRamResident(b.key));
    EXPECT_TRUE(cache.isDiskResident(b.key));
    EXPECT_TRUE(cache.isRamResident(c.key));
    EXPECT_EQ(cache.stats().ram_to_disk_demotions, 1u);

    /*
     * B rises from disk to RAM. Making room demotes A, while B's durable disk
     * copy remains present and receives a persistent LRU touch.
     */
    ASSERT_TRUE(findAfterArchivePublication(cache, *disk, b.key).has_value());
    EXPECT_FALSE(cache.isRamResident(a.key));
    EXPECT_TRUE(cache.isDiskResident(a.key));
    EXPECT_TRUE(cache.isRamResident(b.key));
    EXPECT_TRUE(cache.isDiskResident(b.key));
    EXPECT_TRUE(cache.isRamResident(c.key));
    EXPECT_EQ(cache.stats().disk_hydrations, 1u);
    EXPECT_EQ(cache.stats().ram_to_disk_demotions, 2u);

    /*
     * A now rises. C must be demoted first. Disk is already full, so the newly
     * arriving C record overwrites B, the oldest bottom-tier record. B remains
     * valid in RAM and can be demoted again on a later pressure cycle.
     */
    ASSERT_TRUE(findAfterArchivePublication(cache, *disk, a.key).has_value());
    EXPECT_TRUE(cache.isRamResident(a.key));
    EXPECT_TRUE(cache.isDiskResident(a.key));
    EXPECT_TRUE(cache.isRamResident(b.key));
    EXPECT_FALSE(cache.isDiskResident(b.key));
    EXPECT_FALSE(cache.isRamResident(c.key));
    EXPECT_TRUE(cache.isDiskResident(c.key));
    EXPECT_EQ(cache.stats().disk_hydrations, 2u);
    EXPECT_EQ(cache.stats().ram_to_disk_demotions, 3u);
    EXPECT_EQ(cache.stats().disk_evictions, 1u);

    /*
     * C's second promotion proves the cycle is repeatable. B is demoted back
     * into disk and overwrites now-oldest A; no key is lost from the hierarchy.
     */
    ASSERT_TRUE(findAfterArchivePublication(cache, *disk, c.key).has_value());
    EXPECT_TRUE(cache.isRamResident(a.key));
    EXPECT_FALSE(cache.isDiskResident(a.key));
    EXPECT_FALSE(cache.isRamResident(b.key));
    EXPECT_TRUE(cache.isDiskResident(b.key));
    EXPECT_TRUE(cache.isRamResident(c.key));
    EXPECT_TRUE(cache.isDiskResident(c.key));
    EXPECT_EQ(cache.stats().disk_hydrations, 3u);
    EXPECT_EQ(cache.stats().ram_to_disk_demotions, 4u);
    EXPECT_EQ(cache.stats().disk_evictions, 2u);
    EXPECT_EQ(cache.stats().evictions, 4u);
    const auto ram_observation = ram->telemetry()->snapshot();
    const auto disk_observation = disk->telemetry()->snapshot();
    EXPECT_EQ(ram_observation.used_bytes, ram->usedBytes());
    EXPECT_EQ(disk_observation.used_bytes, kBlockBytes * 2);
    EXPECT_EQ(ram_observation.activity.operations[static_cast<size_t>(PrefixTierEvent::Demotion)], 4u);
    EXPECT_EQ(ram_observation.activity.bytes[static_cast<size_t>(PrefixTierEvent::Demotion)], 4 * kBlockBytes);
    EXPECT_EQ(ram_observation.activity.operations[static_cast<size_t>(PrefixTierEvent::Eviction)], 4u);
    EXPECT_EQ(disk_observation.activity.operations[static_cast<size_t>(PrefixTierEvent::Read)], 3u);
    EXPECT_EQ(disk_observation.activity.bytes[static_cast<size_t>(PrefixTierEvent::Read)], 3 * kBlockBytes);
    EXPECT_EQ(disk_observation.activity.operations[static_cast<size_t>(PrefixTierEvent::Eviction)], 2u);

    ASSERT_TRUE(disk->waitForPersistence());
    cleanup();
}

TEST(Test__PrefixStateCacheLRU, ClearReleasesDiskEntries)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    auto ram = std::make_shared<RamPrefixStorageBackend>(128);
    auto disk = makeDiskBackend(dir, 128);
    ASSERT_TRUE(disk->ready()) << disk->initializationError();
    PrefixStateCache cache(64, ram, disk);
    auto a = ram->allocate(keyFor(0), layoutBytes(32));
    auto b = ram->allocate(keyFor(1), layoutBytes(32));
    auto c = ram->allocate(keyFor(2), layoutBytes(32));
    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(cache.insert(b));
    ASSERT_TRUE(insertAfterArchivePublication(cache, *disk, c));
    ASSERT_TRUE(cache.contains(a.key));
    ASSERT_EQ(cache.stats().disk_bytes, 32u);

    ASSERT_TRUE(cache.clear());

    EXPECT_EQ(cache.size(), 0u);
    EXPECT_EQ(cache.usedBytes(), 0u);
    EXPECT_EQ(cache.stats().ram_bytes, 0u);
    EXPECT_EQ(cache.stats().disk_bytes, 0u);
    EXPECT_FALSE(cache.contains(a.key));
    EXPECT_FALSE(cache.contains(b.key));
    EXPECT_FALSE(cache.contains(c.key));

    cleanup();
}

/** @test Native archive failure is fatal and preserves the known committed key. */
TEST(Test__PrefixStateCacheLRU, DiskHydrationNativeFailureIsFatal)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    auto ram = std::make_shared<RamPrefixStorageBackend>(128);
    auto disk = makeDiskBackend(dir, 128);
    ASSERT_TRUE(disk->ready()) << disk->initializationError();
    PrefixStateCache cache(64, ram, disk);
    auto a = ram->allocate(keyFor(0), layoutBytes(32));
    auto b = ram->allocate(keyFor(1), layoutBytes(32));
    auto c = ram->allocate(keyFor(2), layoutBytes(32));
    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(cache.insert(b));
    ASSERT_TRUE(insertAfterArchivePublication(cache, *disk, c));

    const auto saved_archive = dir / "saved-archive";
    std::filesystem::rename(disk->archivePath(), saved_archive);
    std::filesystem::create_directory(disk->archivePath());

    EXPECT_THROW(cache.find(a.key), std::runtime_error);
    EXPECT_EQ(cache.stats().disk_read_failures, 1u);
    EXPECT_EQ(cache.stats().misses, 1u);
    EXPECT_TRUE(cache.contains(a.key));

    std::filesystem::remove(disk->archivePath());
    std::filesystem::rename(saved_archive, disk->archivePath());

    ASSERT_TRUE(disk->waitForPersistence());
    cleanup();
}
