/**
 * @file Test__DiskPrefixStorageBackend.cpp
 * @brief Device-free durability and online prefix-archive lifecycle regressions.
 *
 * Exercises the production archive worker, concurrent appends, restart-visible
 * LRU, metadata validation and retained-inode hydration. Maintenance must never
 * require inference to rewrite old payloads or abandon a selected restore.
 * Every demotion revalidates backing through the actual archive authority;
 * revalidating identical backing must not append another payload copy.
 */

#include <gtest/gtest.h>

#include "execution/prefix_cache/DiskPrefixStorageBackend.h"
#include "execution/prefix_cache/PrefixArchiveIOGeometry.h"
#include "execution/prefix_cache/RamPrefixStorageBackend.h"
#include "utils/Sha256.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <fstream>
#include <string>
#include <thread>
#include <sys/mman.h>
#include <sys/file.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

using namespace llaminar2;

namespace
{
    /** @brief Resolve the one live immutable payload owned by a single-block fixture. */
    std::filesystem::path singlePayloadPath(const std::filesystem::path &archive)
    {
        std::filesystem::path selected;
        for (const auto &entry : std::filesystem::directory_iterator(archive.string() + ".blocks"))
        {
            if (!selected.empty() || !entry.is_regular_file() || entry.path().extension() != ".kvblock")
                throw std::runtime_error("single-block fixture has unexpected payload storage");
            selected = entry.path();
        }
        if (selected.empty()) throw std::runtime_error("single-block fixture has no payload");
        return selected;
    }

    /** @brief Thread-local native read receipt for one actual payload extent. */
    struct PayloadReadObservation
    {
        dev_t device;
        ino_t inode;
        uint64_t begin;
        uint64_t end;
        uint64_t bytes = 0u;
        static thread_local PayloadReadObservation *active;
        /** @brief Observe payload bytes only; archive metadata lies outside this extent. */
        PayloadReadObservation(const std::filesystem::path &path, uint64_t payload_bytes)
        {
            struct stat state{};
            if (::stat(path.c_str(), &state) != 0 || active)
                throw std::runtime_error("invalid native read observation");
            device = state.st_dev;
            inode = state.st_ino;
            end = static_cast<uint64_t>(state.st_size);
            begin = end - payload_bytes;
            active = this;
        }
        /** @brief Stop observing before this stack-owned receipt retires. */
        ~PayloadReadObservation() { active = nullptr; }
    };
    thread_local PayloadReadObservation *PayloadReadObservation::active = nullptr;

    /** @brief Cross-thread syscall evidence for maintenance reading immutable payloads. */
    struct MaintenancePayloadReads
    {
        dev_t device;
        ino_t inode;
        uint64_t begin;
        uint64_t end;
        std::atomic<uint64_t> bytes{0};
        static std::atomic<MaintenancePayloadReads *> active;
        /** @brief Watch an exact committed payload extent on the native inode. */
        MaintenancePayloadReads(const std::filesystem::path &path, uint64_t payload_bytes)
        {
            struct stat state{};
            if (::stat(path.c_str(), &state) != 0 || active.load())
                throw std::runtime_error("invalid maintenance read observation");
            device = state.st_dev;
            inode = state.st_ino;
            end = static_cast<uint64_t>(state.st_size);
            begin = end - payload_bytes;
            active.store(this);
        }
        /** @brief Retire only after the observed maintenance frontier is joined. */
        ~MaintenancePayloadReads() { active.store(nullptr); }
    };
    std::atomic<MaintenancePayloadReads *> MaintenancePayloadReads::active{nullptr};

    /** @brief One exact native unlink failure after the new journal frontier commits. */
    struct PayloadRetirementFailure
    {
        std::string path;
        std::atomic<bool> injected{false};
        static std::atomic<PayloadRetirementFailure *> active;
        /** @brief Arm one owned file only; every other unlink remains native. */
        explicit PayloadRetirementFailure(const std::filesystem::path &target) : path(target.string())
        {
            PayloadRetirementFailure *expected = nullptr;
            if (!active.compare_exchange_strong(expected, this))
                throw std::runtime_error("overlapping retirement failure observers");
        }
        /** @brief Disarm before this exact pathname storage retires. */
        ~PayloadRetirementFailure() { active.store(nullptr); }
    };
    std::atomic<PayloadRetirementFailure *> PayloadRetirementFailure::active{nullptr};

    /** @brief Hold one actual journal-directory fsync before its durable publication. */
    class JournalPublicationBarrier
    {
    public:
        static std::atomic<JournalPublicationBarrier *> active;
        /** @brief Match only this archive's parent directory; payload fsyncs stay native. */
        explicit JournalPublicationBarrier(DiskPrefixStorageBackend &backend)
            : backend_(backend), entered_(entered_signal_.get_future()),
              resume_(resume_signal_.get_future().share())
        {
            struct stat state{};
            if (::stat(backend.archivePath().parent_path().c_str(), &state) != 0)
                throw std::runtime_error("invalid journal publication observation");
            device_ = state.st_dev;
            inode_ = state.st_ino;
            JournalPublicationBarrier *expected = nullptr;
            if (!active.compare_exchange_strong(expected, this))
                throw std::runtime_error("overlapping journal publication observers");
        }
        /** @brief Release and join the worker before observer storage retires, also on assertion failure. */
        ~JournalPublicationBarrier()
        {
            resume_signal_.set_value();
            (void)backend_.waitForCompaction();
            active.store(nullptr);
        }
        /** @return Whether the real native directory publication reached this observer. */
        bool waitForPublication()
        {
            return entered_.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        }
        /** @brief Suspend only the first fsync of the selected journal directory. */
        void observe(int fd)
        {
            struct stat state{};
            if (::fstat(fd, &state) != 0 || state.st_dev != device_ || state.st_ino != inode_ ||
                claimed_.exchange(true)) return;
            entered_signal_.set_value();
            resume_.wait();
        }

    private:
        DiskPrefixStorageBackend &backend_;
        dev_t device_{};
        ino_t inode_{};
        std::promise<void> entered_signal_;
        std::promise<void> resume_signal_;
        std::future<void> entered_;
        std::shared_future<void> resume_;
        std::atomic<bool> claimed_{false};
    };
    std::atomic<JournalPublicationBarrier *> JournalPublicationBarrier::active{nullptr};
}

/** @brief Preserve native fsync while allowing a test to hold one exact publication frontier. */
extern "C" int fsync(int fd)
{
    const int saved_errno = errno;
    if (auto *barrier = JournalPublicationBarrier::active.load()) barrier->observe(fd);
    errno = saved_errno;
    return static_cast<int>(::syscall(SYS_fsync, fd));
}

/** @brief Preserve real native unlink semantics except for one explicitly armed failure. */
extern "C" int unlink(const char *path) noexcept
{
    if (auto *failure = PayloadRetirementFailure::active.load();
        failure && failure->path == path && !failure->injected.exchange(true))
    {
        errno = EIO;
        return -1;
    }
    return static_cast<int>(::syscall(SYS_unlinkat, AT_FDCWD, path, 0));
}

/**
 * @brief Observe real foreground archive reads without changing their bytes or return value.
 * @param fd Native source descriptor.
 * @param destination Exact production read destination.
 * @param bytes Requested native extent.
 * @param offset Absolute source position.
 * @return Unmodified native syscall result; no synthetic payload path is used.
 */
extern "C" ssize_t pread(int fd, void *destination, size_t bytes, off_t offset)
{
    const auto result = static_cast<ssize_t>(::syscall(SYS_pread64, fd, destination, bytes, offset));
    const int saved_errno = errno;
    if (auto *observer = PayloadReadObservation::active; observer && result > 0 && offset >= 0)
    {
        struct stat state{};
        if (::fstat(fd, &state) == 0 && state.st_dev == observer->device && state.st_ino == observer->inode)
        {
            const auto first = std::max<uint64_t>(offset, observer->begin);
            const auto last = std::min<uint64_t>(static_cast<uint64_t>(offset) + result, observer->end);
            if (last > first) observer->bytes += last - first;
        }
    }
    if (auto *observer = MaintenancePayloadReads::active.load(); observer && result > 0 && offset >= 0)
    {
        struct stat state{};
        if (::fstat(fd, &state) == 0 && state.st_dev == observer->device && state.st_ino == observer->inode)
        {
            const auto first = std::max<uint64_t>(offset, observer->begin);
            const auto last = std::min<uint64_t>(static_cast<uint64_t>(offset) + result, observer->end);
            if (last > first) observer->bytes.fetch_add(last - first);
        }
    }
    errno = saved_errno;
    return result;
}

namespace
{
    constexpr const char *kModelArtifactIdentity =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    /** @brief Small valid attention block, with a terminal payload section. */
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

/** @brief Archive-owned gauges survive restart; counters distinguish writes, touches, selected reads and evictions. */
TEST(Test__DiskPrefixStorageBackend, TelemetryOccupancyTrafficAndRestart)
{
    const auto directory = tempDir();
    const auto layout = makeLayout();
    const auto bytes = layout.totalBytes();
    RamPrefixStorageBackend ram(bytes * 2);
    const auto a = ram.allocate(testKey(), layout);
    const auto b = ram.allocate(makePrefixCacheKey(0x12345678, 0, 1, 2, {3, 4}), layout);
    ASSERT_TRUE(a.valid());
    ASSERT_TRUE(b.valid());
    {
        auto disk = std::make_shared<DiskPrefixStorageBackend>(
            archivePath(directory), bytes, kModelArtifactIdentity);
        ASSERT_TRUE(disk->ready()) << disk->initializationError();
        const auto telemetry = disk->telemetry();
        PrefixBlockHandle stored;
        std::string error;
        ASSERT_TRUE(disk->writeBlock(a, &stored, nullptr, &error)) << error;
        ASSERT_TRUE(disk->writeBlock(a, &stored, nullptr, &error)) << error;
        auto observation = telemetry->snapshot();
        EXPECT_EQ(observation.used_bytes, bytes);
        EXPECT_EQ(observation.entries, 1u);
        EXPECT_EQ(observation.activity.operations[static_cast<size_t>(PrefixTierEvent::Write)], 1u);
        EXPECT_EQ(observation.activity.bytes[static_cast<size_t>(PrefixTierEvent::Write)], bytes);
        EXPECT_EQ(observation.activity.operations[static_cast<size_t>(PrefixTierEvent::BackingReuse)], 1u);
        EXPECT_EQ(observation.activity.bytes[static_cast<size_t>(PrefixTierEvent::BackingReuse)], 0u);
        RamPrefixStorageBackend target(bytes);
        auto ticket = disk->captureLookupSource(a.key, layout, &error);
        ASSERT_TRUE(ticket.has_value()) << error;
        ASSERT_TRUE(disk->selectHydrationSections(*ticket, PrefixPayloadReadSet::SequenceRows, &error)) << error;
        PrefixBlockHandle hydrated;
        ASSERT_TRUE(disk->hydrateSelected(*ticket, target, &hydrated, &error)) << error;
        observation = telemetry->snapshot();
        EXPECT_EQ(observation.activity.operations[static_cast<size_t>(PrefixTierEvent::Read)], 1u);
        EXPECT_EQ(observation.activity.bytes[static_cast<size_t>(PrefixTierEvent::Read)], a.kvBytes());
        EXPECT_LT(a.kvBytes(), bytes); // Endpoint logits were not transferred.
        ASSERT_TRUE(disk->writeBlock(b, &stored, nullptr, &error)) << error;
        observation = telemetry->snapshot();
        EXPECT_EQ(observation.used_bytes, bytes);
        EXPECT_EQ(observation.activity.operations[static_cast<size_t>(PrefixTierEvent::Eviction)], 1u);
        EXPECT_EQ(observation.activity.bytes[static_cast<size_t>(PrefixTierEvent::Eviction)], bytes);
        ASSERT_TRUE(disk->waitForCompaction(&error)) << error;
        const auto before_poll = telemetry->snapshot();
        PayloadReadObservation reads(singlePayloadPath(archivePath(directory)), bytes);
        for (int i = 0; i < 20; ++i) EXPECT_EQ(telemetry->snapshot().revision, before_poll.revision);
        EXPECT_EQ(reads.bytes, 0u);
    }
    {
        DiskPrefixStorageBackend reopened(archivePath(directory), bytes, kModelArtifactIdentity);
        ASSERT_TRUE(reopened.ready()) << reopened.initializationError();
        const auto observation = reopened.telemetry()->snapshot();
        EXPECT_EQ(observation.used_bytes, bytes);
        EXPECT_EQ(observation.entries, 1u);
        for (const auto count : observation.activity.operations) EXPECT_EQ(count, 0u);
    }
    std::filesystem::remove_all(directory);
}

/**
 * @brief Demotion reuses exact backing across append, replacement and compaction.
 *
 * An identical publication writes metadata only. Changed bytes and deleted
 * records require a new payload, while compaction retains the exact backing.
 * Native reads and actual file extents establish correctness and economy.
 */
TEST(Test__DiskPrefixStorageBackend, IdempotentPublicationPreservesPayloadAndBudget)
{
    const auto directory = tempDir();
    auto layout = makeLayout();
    layout.bytes_per_fa_layer_k = 16 * 1024;
    layout.bytes_per_fa_layer_v = 16 * 1024;
    RamPrefixStorageBackend ram(layout.totalBytes());
    auto source = ram.allocate(testKey(), layout);
    ASSERT_TRUE(source.valid());
    std::fill_n(static_cast<uint8_t *>(source.kv_payload), source.kvBytes(), 0x31);
    PrefixBlockHandle last;
    {
        DiskPrefixStorageBackend disk(archivePath(directory), layout.totalBytes(),
                                      kModelArtifactIdentity);
        ASSERT_TRUE(disk.ready()) << disk.initializationError();
        std::string error;
        PrefixArchiveWriteDisposition disposition;
        PrefixBlockHandle original;
        ASSERT_TRUE(disk.writeBlock(source, &original, nullptr, &error, &disposition)) << error;
        EXPECT_EQ(disposition, PrefixArchiveWriteDisposition::Stored);
        const auto before = std::filesystem::file_size(archivePath(directory));
        ASSERT_TRUE(disk.writeBlock(source, &last, nullptr, &error, &disposition)) << error;
        EXPECT_EQ(disposition, PrefixArchiveWriteDisposition::Reused);
        EXPECT_LT(std::filesystem::file_size(archivePath(directory)) - before, 1024u)
            << "Revalidation may append a touch, never another payload";

        // Recomputed bytes belong to a fresh immutable payload generation.
        ASSERT_TRUE(ram.release(source));
        source = ram.allocate(testKey(), layout);
        std::fill_n(static_cast<uint8_t *>(source.kv_payload), source.kvBytes(), 0x31);
        static_cast<uint8_t *>(source.kv_payload)[0] = 0x52;
        ASSERT_TRUE(disk.writeBlock(source, &last, nullptr, &error, &disposition)) << error;
        EXPECT_EQ(disposition, PrefixArchiveWriteDisposition::Stored);
        ASSERT_TRUE(disk.requestCompaction(&error)) << error;
        ASSERT_TRUE(disk.waitForCompaction(&error)) << error;
        const auto discovered = disk.compatibleEntries(testKey().fingerprint, layout, &error);
        ASSERT_EQ(discovered.size(), 1u) << error;
        PrefixBlockHandle hydrated;
        ASSERT_TRUE(disk.readBlock(testKey(), layout, &hydrated, &error)) << error;
        ASSERT_TRUE(hydrated.kv_storage);
        EXPECT_EQ(hydrated.kv_storage->front(), 0x52);
        EXPECT_TRUE(std::all_of(hydrated.kv_storage->begin() + 1,
                               hydrated.kv_storage->end(),
                               [](uint8_t value) { return value == 0x31; }));
        EXPECT_EQ(disk.usedBytes(), layout.totalBytes());
        const auto compacted_bytes = std::filesystem::file_size(archivePath(directory));
        ASSERT_TRUE(disk.writeBlock(source, &last, nullptr, &error, &disposition)) << error;
        EXPECT_EQ(disposition, PrefixArchiveWriteDisposition::Reused);
        EXPECT_LT(std::filesystem::file_size(archivePath(directory)) - compacted_bytes, 1024u);
        ASSERT_TRUE(disk.release(last));
        ASSERT_TRUE(disk.writeBlock(source, &last, nullptr, &error, &disposition)) << error;
        EXPECT_EQ(disposition, PrefixArchiveWriteDisposition::Stored);
    }
    std::filesystem::remove_all(directory);
}

/** @brief New payload generations and changed metadata cannot reuse an older record. */
TEST(Test__DiskPrefixStorageBackend, IdempotentPublicationAuthenticatesPayloadGeneration)
{
    const auto directory = tempDir();
    {
        auto layout = makeLayout();
        layout.includes_hybrid_state = true;
        layout.hybrid_state_bytes = 8;
        layout.hybrid_host_state_bytes = 8;
        layout.includes_mtp_state = true;
        layout.mtp_kv_bytes = 8;
        layout.includes_terminal_hidden = true;
        layout.terminal_hidden_bytes = 8;
        RamPrefixStorageBackend ram(256);
        auto source = ram.allocate(testKey(), layout);
        ASSERT_TRUE(source.valid());
        ASSERT_TRUE(ram.attachModelRuntimeState(&source,
            std::make_shared<std::vector<uint8_t>>(8, 0x18)));
        const std::array<std::pair<void *, size_t>, 6> sections{{
            {source.kv_payload, source.kvBytes()},
            {source.hybrid_payload, source.hybridBytes()},
            {source.mtp_payload, source.layout.mtpKVBytes()},
            {source.terminal_hidden, source.terminalHiddenBytes()},
            {source.terminal_logits, source.terminalLogitsBytes()},
            {source.model_runtime_state_storage->data(), source.model_runtime_state_storage->size()},
        }};
        DiskPrefixStorageBackend disk(archivePath(directory), 256, kModelArtifactIdentity);
        for (const auto &[address, bytes] : sections)
        {
            ASSERT_TRUE(address);
            ASSERT_GT(bytes, 0u);
            std::fill_n(static_cast<uint8_t *>(address), bytes, 0x18);
        }
        PrefixArchiveWriteDisposition disposition;
        std::string error;
        ASSERT_TRUE(disk.writeBlock(source, nullptr, nullptr, &error, &disposition)) << error;
        for (size_t section = 0; section < sections.size(); ++section)
        {
            SCOPED_TRACE(section);
            const auto prior_identity = source.payload_identity;
            ASSERT_TRUE(ram.release(source));
            source = ram.allocate(testKey(), layout);
            ASSERT_TRUE(ram.attachModelRuntimeState(&source,
                std::make_shared<std::vector<uint8_t>>(8, 0x39)));
            EXPECT_NE(source.payload_identity, prior_identity);
            ASSERT_TRUE(disk.writeBlock(source, nullptr, nullptr, &error, &disposition)) << error;
            EXPECT_EQ(disposition, PrefixArchiveWriteDisposition::Stored);
            ASSERT_TRUE(disk.writeBlock(source, nullptr, nullptr, &error, &disposition)) << error;
            EXPECT_EQ(disposition, PrefixArchiveWriteDisposition::Reused);
        }
        source.has_terminal_hidden = !source.has_terminal_hidden;
        ASSERT_TRUE(disk.writeBlock(source, nullptr, nullptr, &error, &disposition)) << error;
        EXPECT_EQ(disposition, PrefixArchiveWriteDisposition::Stored);
    }
    std::filesystem::remove_all(directory);
}

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

/** @test A selected, logically evicted record survives another backend's rename. */
TEST(Test__DiskPrefixStorageBackend, HydrationSurvivesBackgroundInodeReplacement)
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
        auto ticket = reader->beginHydration(key, layout, &error);
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
        ASSERT_TRUE(reader->hydrateSelected(*ticket, ram, &restored, &error)) << error;
        EXPECT_EQ(restored.key, key);
        EXPECT_TRUE(std::all_of(restored.kv_storage->begin(), restored.kv_storage->end(),
                               [](uint8_t byte) { return byte == 0xab; }));
        EXPECT_TRUE(std::all_of(restored.terminal_logits_storage->begin(),
                               restored.terminal_logits_storage->end(),
                               [](uint8_t byte) { return byte == 0xcd; }));
        EXPECT_FALSE(reader->hydrateSelected(*ticket, ram, &restored, &error));
    }
    std::filesystem::remove_all(dir);
}

/** @test Metadata reuse must work even when every RAM payload address forbids reads. */
TEST(Test__DiskPrefixStorageBackend, MetadataOnlyReuseSkipsPayloadAccess)
{
    const auto directory = tempDir();
    {
        auto layout = makeLayout();
        layout.includes_hybrid_state = layout.includes_mtp_state = layout.includes_terminal_hidden = true;
        layout.hybrid_host_state_bytes = layout.hybrid_state_bytes = 32u;
        layout.mtp_kv_bytes = 16u;
        layout.terminal_hidden_bytes = 8u;
        RamPrefixStorageBackend ram(256u);
        auto source = ram.allocate(testKey(), layout);
        ASSERT_TRUE(ram.attachModelRuntimeState(&source, std::make_shared<std::vector<uint8_t>>(8u, 0x71)));
        DiskPrefixStorageBackend disk(archivePath(directory), 256u, kModelArtifactIdentity);
        std::string error;
        PrefixArchiveWriteDisposition disposition;
        ASSERT_TRUE(disk.writeBlock(source, nullptr, nullptr, &error, &disposition)) << error;
        ASSERT_EQ(disposition, PrefixArchiveWriteDisposition::Stored);
        const auto page_bytes = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
        void *page = ::mmap(nullptr, page_bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ASSERT_NE(page, MAP_FAILED);
        auto owner = std::shared_ptr<void>(page, [page_bytes](void *address) { ::munmap(address, page_bytes); });
        auto retained = source;
        retained.kv_payload = retained.hybrid_payload = retained.mtp_payload = page;
        retained.terminal_hidden = retained.terminal_logits = page;
        retained.model_runtime_state_storage = std::make_shared<PrefixRuntimeStateStorage>(owner, 8u);
        PayloadReadObservation observation(singlePayloadPath(disk.archivePath()), source.total_bytes);
        ASSERT_TRUE(disk.writeBlock(retained, nullptr, nullptr, &error, &disposition)) << error;
        EXPECT_EQ(disposition, PrefixArchiveWriteDisposition::Reused);
        EXPECT_EQ(observation.bytes, 0u);
    }
    std::filesystem::remove_all(directory);
}

/** @test A retained version survives replacement without changing its successor's LRU. */
TEST(Test__DiskPrefixStorageBackend, MetadataSelectionRetainsVersionAndRecency)
{
    const auto directory = tempDir();
    {
        const auto layout = makeLayout();
        const auto a = testKey();
        const auto b = makePrefixCacheKey(a.fingerprint, 0, 1, 2, {3, 4});
        const auto c = makePrefixCacheKey(a.fingerprint, 0, 2, 4, {5, 6});
        RamPrefixStorageBackend ram(3u * layout.totalBytes());
        auto disk = std::make_shared<DiskPrefixStorageBackend>(
            archivePath(directory), 2u * layout.totalBytes(), kModelArtifactIdentity);
        auto source = ram.allocate(a, layout);
        std::fill_n(static_cast<uint8_t *>(source.kv_payload), source.kvBytes(), 0x31);
        const auto original_identity = source.payload_identity;
        std::string error;
        ASSERT_TRUE(disk->writeBlock(source, nullptr, nullptr, &error)) << error;
        auto ticket = disk->captureLookupSource(a, layout, &error);
        ASSERT_TRUE(ticket) << error;

        ASSERT_TRUE(ram.release(source));
        source = ram.allocate(a, layout);
        std::fill_n(static_cast<uint8_t *>(source.kv_payload), source.kvBytes(), 0x52);
        ASSERT_NE(source.payload_identity, original_identity);
        ASSERT_TRUE(disk->writeBlock(source, nullptr, nullptr, &error)) << error;
        auto second = ram.allocate(b, layout);
        ASSERT_TRUE(disk->writeBlock(second, nullptr, nullptr, &error)) << error;
        ASSERT_TRUE(disk->requestCompaction(&error)) << error;
        ASSERT_TRUE(disk->waitForCompaction(&error)) << error;

        ASSERT_TRUE(disk->selectHydrationSections(*ticket, PrefixPayloadReadSet::WholeArchive, &error)) << error;
        RamPrefixStorageBackend destination(layout.totalBytes());
        PrefixBlockHandle restored;
        ASSERT_TRUE(disk->hydrateSelected(*ticket, destination, &restored, &error)) << error;
        EXPECT_EQ(restored.payload_identity, original_identity);
        EXPECT_TRUE(std::all_of(restored.kv_storage->begin(), restored.kv_storage->end(),
            [](uint8_t byte) { return byte == 0x31; }));

        // Both selection and hydration of the old version must leave the new
        // A older than B. C therefore evicts A, proving actual victim order.
        auto third = ram.allocate(c, layout);
        std::vector<PrefixCacheKey> evicted;
        ASSERT_TRUE(disk->writeBlock(third, nullptr, &evicted, &error)) << error;
        ASSERT_EQ(evicted.size(), 1u);
        EXPECT_EQ(evicted.front(), a);
    }
    std::filesystem::remove_all(directory);
}

/** @test An incompatible archive is rejected intact rather than reinterpreted or erased. */
TEST(Test__DiskPrefixStorageBackend, MetadataFormatRejectsLegacyArchiveIntact)
{
    for (const char version : {1, 2})
    {
        SCOPED_TRACE(static_cast<unsigned>(version));
        const auto directory = tempDir();
        {
            RamPrefixStorageBackend ram(1024u);
            auto source = ram.allocate(testKey(), makeLayout());
            DiskPrefixStorageBackend disk(archivePath(directory), 1024u, kModelArtifactIdentity);
            std::string error;
            ASSERT_TRUE(disk.writeBlock(source, nullptr, nullptr, &error)) << error;
        }
        std::fstream archive(archivePath(directory), std::ios::binary | std::ios::in | std::ios::out);
        archive.seekp(8);
        const std::array<char, 4> legacy_version{version, 0, 0, 0};
        archive.write(legacy_version.data(), legacy_version.size());
        archive.seekg(0);
        const std::string before((std::istreambuf_iterator<char>(archive)), {});
        archive.close();
        {
            DiskPrefixStorageBackend reopened(archivePath(directory), 1024u, kModelArtifactIdentity);
            EXPECT_FALSE(reopened.ready());
            EXPECT_NE(reopened.initializationError().find("header"), std::string::npos);
            EXPECT_NE(reopened.initializationError().find("expected 3; use a fresh cache directory"), std::string::npos);
        }
        std::ifstream retained(archivePath(directory), std::ios::binary);
        const std::string after((std::istreambuf_iterator<char>(retained)), {});
        EXPECT_EQ(after, before);
        retained.close();
        std::filesystem::remove_all(directory);
    }
}

/** @test Deferred row reads copy exactly KV/MTP, leaving endpoint state on disk. */
TEST(Test__DiskPrefixStorageBackend, SelectedReadCopiesOnlyConsumedSections)
{
    const auto directory = tempDir();
    {
        auto layout = makeLayout();
        layout.includes_hybrid_state = true;
        layout.hybrid_host_state_bytes = layout.hybrid_state_bytes = 32;
        layout.includes_mtp_state = true;
        layout.mtp_kv_bytes = 16;
        layout.includes_terminal_hidden = true;
        layout.terminal_hidden_bytes = 8;
        RamPrefixStorageBackend source_ram(256);
        auto source = source_ram.allocate(testKey(), layout);
        ASSERT_TRUE(source.valid());
        source.has_hybrid_state = source.has_terminal_hidden = source.has_terminal_logits = true;
        std::fill_n(source.kvKData(), source.kvBytes(), 0xa5);
        std::fill_n(static_cast<uint8_t *>(source.mtp_payload), layout.mtpKVBytes(), 0x6e);
        ASSERT_TRUE(source_ram.attachModelRuntimeState(&source,
            std::make_shared<std::vector<uint8_t>>(8, 0x7f)));
        auto disk = std::make_shared<DiskPrefixStorageBackend>(
            archivePath(directory), 256, kModelArtifactIdentity);
        std::string error;
        ASSERT_TRUE(disk->writeBlock(source, nullptr, nullptr, &error)) << error;
        const auto payload_path = singlePayloadPath(disk->archivePath());
        PayloadReadObservation observation(payload_path, source.total_bytes);
        auto rows = disk->captureLookupSource(testKey(), layout, &error);
        auto whole = disk->captureLookupSource(testKey(), layout, &error);
        ASSERT_TRUE(rows && whole) << error;
        EXPECT_EQ(rows->hydratedPayloadBytes(), 0u);

        // Payload contents are opaque. Metadata selection must not scan them;
        // hydration copies the selected sections without a separate hash pass.
        const auto last_payload = std::filesystem::file_size(payload_path) - 1u;
        std::fstream archive(payload_path, std::ios::binary | std::ios::in | std::ios::out);
        archive.seekp(last_payload);
        archive.put('\0');
        archive.close();
        ASSERT_TRUE(disk->selectHydrationSections(*rows, PrefixPayloadReadSet::SequenceRows, &error)) << error;
        EXPECT_EQ(observation.bytes, 0u) << "Metadata selection must not pre-read any payload";
        const size_t row_bytes = layout.faKVBytes() + layout.mtpKVBytes();
        RamPrefixStorageBackend destination(row_bytes);
        PrefixBlockHandle restored;
        ASSERT_TRUE(disk->hydrateSelected(*rows, destination, &restored, &error)) << error;
        EXPECT_EQ(rows->hydratedPayloadBytes(), row_bytes);
        EXPECT_EQ(observation.bytes, row_bytes) << "Hydration must read the selected bytes exactly once";
        EXPECT_EQ(destination.usedBytes(), row_bytes);
        EXPECT_TRUE(std::all_of(restored.kv_storage->begin(), restored.kv_storage->end(),
            [](uint8_t byte) { return byte == 0xa5; }));
        EXPECT_TRUE(std::all_of(restored.mtp_storage->begin(), restored.mtp_storage->end(),
            [](uint8_t byte) { return byte == 0x6e; }));
        EXPECT_FALSE(restored.hybrid_storage);
        EXPECT_FALSE(restored.terminal_hidden_storage);
        EXPECT_FALSE(restored.terminal_logits_storage);
        EXPECT_FALSE(restored.model_runtime_state_storage);
        EXPECT_FALSE(restored.has_hybrid_state || restored.has_terminal_hidden ||
                     restored.has_terminal_logits || restored.has_model_runtime_state);
        EXPECT_FALSE(disk->hydrateSelected(*rows, destination, &restored, &error));
        ASSERT_TRUE(disk->selectHydrationSections(*whole, PrefixPayloadReadSet::WholeArchive, &error));
        EXPECT_FALSE(disk->selectHydrationSections(*whole, PrefixPayloadReadSet::SequenceRows, &error));
        RamPrefixStorageBackend endpoint(source.total_bytes);
        ASSERT_TRUE(disk->hydrateSelected(*whole, endpoint, &restored, &error)) << error;
        EXPECT_EQ(observation.bytes, row_bytes + source.total_bytes);
        EXPECT_EQ(restored.model_runtime_state_storage->bytes().back(), 0u);
        EXPECT_EQ(restored.payload_identity, source.payload_identity);
        EXPECT_FALSE(whole->valid());
    }
    std::filesystem::remove_all(directory);
}

/** @test An inaccessible archive is a storage failure, never an ordinary missing key. */
TEST(Test__DiskPrefixStorageBackend, SelectedReadReportsNativeOpenFailure)
{
    const auto directory = tempDir();
    {
        auto disk = std::make_shared<DiskPrefixStorageBackend>(
            archivePath(directory), 4096u, kModelArtifactIdentity);
        ASSERT_TRUE(disk->ready());
        std::string error = "stale diagnostic";
        EXPECT_FALSE(disk->captureLookupSource(testKey(), makeLayout(), &error));
        EXPECT_TRUE(error.empty()); // A genuinely absent key is an ordinary miss.
        const auto saved = directory / "retained-original.kvcache";
        std::filesystem::rename(archivePath(directory), saved);
        std::filesystem::create_directory(archivePath(directory));
        EXPECT_FALSE(disk->captureLookupSource(testKey(), makeLayout(), &error));
        EXPECT_NE(error.find("failed to open prefix archive lookup source"), std::string::npos)
            << error;
        std::filesystem::remove(archivePath(directory));
        std::filesystem::rename(saved, archivePath(directory));
    }
    std::filesystem::remove_all(directory);
}

/** @test Truncated KV or shifted-MTP extents fail before any destination allocation. */
TEST(Test__DiskPrefixStorageBackend, SelectedReadRejectsTruncatedExtent)
{
    for (const bool shifted_mtp : {false, true})
    {
        const auto directory = tempDir();
        {
            auto layout = makeLayout();
            layout.includes_terminal_logits = false;
            layout.includes_mtp_state = shifted_mtp;
            layout.mtp_kv_bytes = shifted_mtp ? 16u : 0u;
            RamPrefixStorageBackend source_ram(layout.totalBytes());
            auto source = source_ram.allocate(testKey(), layout);
            auto disk = std::make_shared<DiskPrefixStorageBackend>(
                archivePath(directory), layout.totalBytes(), kModelArtifactIdentity);
            std::string error;
            ASSERT_TRUE(disk->writeBlock(source, nullptr, nullptr, &error)) << error;
            auto ticket = disk->captureLookupSource(testKey(), layout, &error);
            ASSERT_TRUE(ticket) << error;
            const auto payload_path = singlePayloadPath(disk->archivePath());
            const auto last_payload = std::filesystem::file_size(payload_path) - 1u;
            std::filesystem::resize_file(payload_path, last_payload);
            EXPECT_FALSE(disk->selectHydrationSections(*ticket, PrefixPayloadReadSet::SequenceRows, &error));
            EXPECT_NE(error.find("file extent"), std::string::npos) << error;
            EXPECT_FALSE(ticket->valid());
            RamPrefixStorageBackend destination(layout.totalBytes());
            PrefixBlockHandle restored;
            EXPECT_FALSE(disk->hydrateSelected(*ticket, destination, &restored, &error));
            EXPECT_EQ(destination.usedBytes(), 0u);
            EXPECT_EQ(ticket->hydratedPayloadBytes(), 0u);
        }
        std::filesystem::remove_all(directory);
    }
}

/** @test A captured cohort shares one inode lease and survives deletion plus compaction. */
TEST(Test__DiskPrefixStorageBackend, SelectedReadCohortRetainsOneImmutableInode)
{
    const auto directory = tempDir();
    {
        const auto layout = makeLayout();
        auto disk = std::make_shared<DiskPrefixStorageBackend>(
            archivePath(directory), layout.totalBytes(), kModelArtifactIdentity);
        RamPrefixStorageBackend source_ram(layout.totalBytes());
        auto source = source_ram.allocate(testKey(), layout);
        std::fill(source.kv_storage->begin(), source.kv_storage->end(), 0xab);
        std::string error;
        ASSERT_TRUE(disk->writeBlock(source, nullptr, nullptr, &error)) << error;
        const auto retained_payload_path = singlePayloadPath(disk->archivePath());
        const auto archive_descriptors = [&]()
        {
            size_t count = 0u;
            for (const auto &entry : std::filesystem::directory_iterator("/proc/self/fd"))
            {
                std::error_code ignored;
                const auto target = std::filesystem::read_symlink(entry.path(), ignored).string();
                if (!ignored && (target == retained_payload_path.string() ||
                                 target == retained_payload_path.string() + " (deleted)")) ++count;
            }
            return count;
        };
        const auto initial_fds = archive_descriptors();
        std::vector<DiskPrefixStorageBackend::HydrationTicket> cohort;
        for (int index = 0; index < 128; ++index)
        {
            auto ticket = disk->captureLookupSource(testKey(), layout, &error);
            ASSERT_TRUE(ticket) << error;
            cohort.push_back(std::move(*ticket));
            EXPECT_FALSE(ticket->valid());
        }
        EXPECT_EQ(archive_descriptors(), initial_fds + 1u);
        PrefixBlockHandle output;
        RamPrefixStorageBackend destination(layout.faKVBytes());
        EXPECT_FALSE(disk->hydrateSelected(cohort.back(), destination, &output, &error));
        DiskPrefixStorageBackend peer(archivePath(directory), layout.totalBytes(), kModelArtifactIdentity);
        auto replacement = source;
        replacement.key = makePrefixCacheKey(testKey().fingerprint, 0, 1, 2, {9, 10});
        ASSERT_TRUE(peer.writeBlock(replacement, nullptr, nullptr, &error)) << error;
        ASSERT_TRUE(peer.requestCompaction(&error)) << error;
        ASSERT_TRUE(peer.waitForCompaction(&error)) << error;
        ASSERT_TRUE(disk->selectHydrationSections(cohort.back(), PrefixPayloadReadSet::SequenceRows, &error)) << error;
        ASSERT_TRUE(disk->hydrateSelected(cohort.back(), destination, &output, &error)) << error;
        EXPECT_TRUE(std::all_of(output.kv_storage->begin(), output.kv_storage->end(),
            [](uint8_t byte) { return byte == 0xab; }));
        EXPECT_EQ(cohort.back().hydratedPayloadBytes(), layout.faKVBytes());
        cohort.clear();
        EXPECT_EQ(archive_descriptors(), initial_fds);
    }
    std::filesystem::remove_all(directory);
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

/** @test Real stale metadata volume schedules automatic off-request reclamation. */
TEST(Test__DiskPrefixStorageBackend, BackgroundCompactionReclaimsAutomaticCapacityHistory)
{
    const auto dir = tempDir();
    {
        auto layout = makeLayout();
        layout.bytes_per_fa_layer_k = 128u;
        layout.bytes_per_fa_layer_v = 128u;
        layout.includes_terminal_logits = false;
        layout.terminal_logits_bytes = 0u;
        const auto bytes = layout.totalBytes();
        RamPrefixStorageBackend ram(bytes);
        auto handle = ram.allocate(testKey(), layout);
        ASSERT_TRUE(handle.valid());
        DiskPrefixStorageBackend disk(archivePath(dir), bytes, kModelArtifactIdentity);
        std::string error;
        for (uint8_t version = 0; version < 96; ++version)
        {
            if (version != 0)
            {
                // Published payloads are immutable. A new harvest owns a new
                // identity even when it replaces the same logical prefix key.
                ASSERT_TRUE(ram.release(handle));
                handle = ram.allocate(testKey(), layout);
                ASSERT_TRUE(handle.valid());
            }
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
        EXPECT_LT(std::filesystem::file_size(disk.archivePath()), 64u * 1024u);
        EXPECT_EQ(std::filesystem::file_size(singlePayloadPath(disk.archivePath())), bytes);
        PrefixBlockHandle restored;
        ASSERT_TRUE(disk.readBlock(testKey(), layout, &restored, &error)) << error;
        EXPECT_TRUE(std::all_of(restored.kv_storage->begin(), restored.kv_storage->end(),
                               [](uint8_t byte) { return byte == 95u; }));
    }
    std::filesystem::remove_all(dir);
}

/** @test Maintenance never revisits payloads already made durable by a demotion. */
TEST(Test__DiskPrefixStorageBackend, CompactionNeverReadsPayloadBytes)
{
    const auto directory = tempDir();
    {
        const auto layout = makeLayout();
        RamPrefixStorageBackend ram(layout.totalBytes());
        const auto handle = ram.allocate(testKey(), layout);
        DiskPrefixStorageBackend disk(archivePath(directory), layout.totalBytes(), kModelArtifactIdentity);
        std::string error;
        ASSERT_TRUE(disk.writeBlock(handle, nullptr, nullptr, &error)) << error;
        MaintenancePayloadReads observed(singlePayloadPath(disk.archivePath()), handle.total_bytes);
        ASSERT_TRUE(disk.requestCompaction(&error)) << error;
        ASSERT_TRUE(disk.waitForCompaction(&error)) << error;
        EXPECT_EQ(observed.bytes.load(), 0u)
            << "Maintenance must compact metadata without repeatedly copying cache payloads";
        EXPECT_GT(disk.compactionStatus().publications, 0u);
    }
    std::filesystem::remove_all(directory);
}

/** @test Retired payloads are reclaimed even when another process owns maintenance. */
TEST(Test__DiskPrefixStorageBackend, ChurnStorageIsBoundedWithoutCompaction)
{
    const auto directory = tempDir();
    {
        auto layout = makeLayout();
        layout.bytes_per_fa_layer_k = 64u * 1024u;
        layout.bytes_per_fa_layer_v = 64u * 1024u;
        const auto bytes = layout.totalBytes();
        RamPrefixStorageBackend ram(bytes);
        DiskPrefixStorageBackend disk(archivePath(directory), bytes, kModelArtifactIdentity);
        const auto lock_path = disk.archivePath().string() + ".compact.lock";
        const int lease = ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        ASSERT_GE(lease, 0);
        const auto close_lease = [](int *fd) { ::close(*fd); delete fd; };
        std::unique_ptr<int, decltype(close_lease)> lease_owner(new int(lease), close_lease);
        ASSERT_EQ(::flock(lease, LOCK_EX), 0);
        std::string error;
        for (uint8_t version = 0; version < 32; ++version)
        {
            auto source = ram.allocate(testKey(), layout);
            ASSERT_TRUE(source.valid());
            std::fill(source.kv_storage->begin(), source.kv_storage->end(), version);
            ASSERT_TRUE(disk.writeBlock(source, nullptr, nullptr, &error)) << error;
            ASSERT_TRUE(disk.requestCompaction(&error)) << error;
            ASSERT_TRUE(disk.waitForCompaction(&error)) << error;
            ASSERT_TRUE(ram.release(source));
        }
        EXPECT_EQ(disk.compactionStatus().publications, 0u);
        uint64_t physical_bytes = 0;
        for (const auto &entry : std::filesystem::recursive_directory_iterator(directory))
            if (entry.is_regular_file()) physical_bytes += entry.file_size();
        EXPECT_LE(physical_bytes, bytes + 32u * 4096u)
            << "A blocked compactor may retain metadata history, never obsolete payload copies";
        PrefixBlockHandle restored;
        ASSERT_TRUE(disk.readBlock(testKey(), layout, &restored, &error)) << error;
        EXPECT_TRUE(std::all_of(restored.kv_storage->begin(), restored.kv_storage->end(),
            [](uint8_t byte) { return byte == 31; }));
    }
    std::filesystem::remove_all(directory);
}

/** @test Crash-orphaned files retire only after the recovered journal frontier is durable. */
TEST(Test__DiskPrefixStorageBackend, RestartReclaimsUncommittedPayloadFiles)
{
    const auto directory = tempDir();
    const auto layout = makeLayout();
    RamPrefixStorageBackend ram(layout.totalBytes());
    const auto source = ram.allocate(testKey(), layout);
    std::fill(source.kv_storage->begin(), source.kv_storage->end(), 0x62);
    {
        DiskPrefixStorageBackend disk(archivePath(directory), layout.totalBytes(), kModelArtifactIdentity);
        std::string error;
        ASSERT_TRUE(disk.writeBlock(source, nullptr, nullptr, &error)) << error;
    }
    const auto live = singlePayloadPath(archivePath(directory));
    const auto committed_end = std::filesystem::file_size(archivePath(directory));
    const auto orphan = live.parent_path() / "ffffffffffffffffffffffffffffffff.kvblock";
    ASSERT_NE(live, orphan);
    { std::ofstream abandoned(orphan, std::ios::binary); abandoned << "uncommitted opaque bytes"; }
    { std::ofstream interrupted(archivePath(directory), std::ios::binary | std::ios::app); interrupted << "LLKVR001-partial-put"; }
    {
        MaintenancePayloadReads observed(live, source.total_bytes);
        DiskPrefixStorageBackend reopened(archivePath(directory), layout.totalBytes(), kModelArtifactIdentity);
        ASSERT_TRUE(reopened.ready()) << reopened.initializationError();
        EXPECT_EQ(observed.bytes.load(), 0u);
        EXPECT_EQ(std::filesystem::file_size(archivePath(directory)), committed_end);
        EXPECT_EQ(singlePayloadPath(archivePath(directory)), live);
        EXPECT_FALSE(std::filesystem::exists(orphan));
        PrefixBlockHandle restored;
        std::string error;
        ASSERT_TRUE(reopened.readBlock(testKey(), layout, &restored, &error)) << error;
        EXPECT_EQ(*restored.kv_storage, *source.kv_storage);
        EXPECT_EQ(observed.bytes.load(), source.total_bytes);
    }
    std::filesystem::remove_all(directory);
}

/** @test A missing or truncated committed payload is fatal; the journal is retained intact. */
TEST(Test__DiskPrefixStorageBackend, InvalidCommittedPayloadFailsRestart)
{
    for (const bool missing : {false, true})
    {
        const auto directory = tempDir();
        const auto layout = makeLayout();
        {
            RamPrefixStorageBackend ram(layout.totalBytes());
            const auto source = ram.allocate(testKey(), layout);
            DiskPrefixStorageBackend disk(archivePath(directory), layout.totalBytes(), kModelArtifactIdentity);
            std::string error;
            ASSERT_TRUE(disk.writeBlock(source, nullptr, nullptr, &error)) << error;
        }
        const auto payload = singlePayloadPath(archivePath(directory));
        if (missing) std::filesystem::remove(payload);
        else std::filesystem::resize_file(payload, layout.totalBytes() - 1u);
        std::ifstream before_file(archivePath(directory), std::ios::binary);
        const std::string before((std::istreambuf_iterator<char>(before_file)), {});
        before_file.close();
        {
            DiskPrefixStorageBackend reopened(archivePath(directory), layout.totalBytes(), kModelArtifactIdentity);
            EXPECT_FALSE(reopened.ready());
            EXPECT_NE(reopened.initializationError().find("invalid committed extent"), std::string::npos);
        }
        std::ifstream after_file(archivePath(directory), std::ios::binary);
        const std::string after((std::istreambuf_iterator<char>(after_file)), {});
        EXPECT_EQ(after, before);
        after_file.close();
        std::filesystem::remove_all(directory);
    }
}

/** @test A failed native retirement poisons the writer; restart retains the new committed version. */
TEST(Test__DiskPrefixStorageBackend, RetirementFailureIsFatalAndRecoverable)
{
    const auto directory = tempDir();
    const auto layout = makeLayout();
    const auto replacement_key = makePrefixCacheKey(testKey().fingerprint, 0, 1, 2, {3, 4});
    std::filesystem::path old_payload;
    RamPrefixStorageBackend ram(layout.totalBytes() * 2u);
    const auto original = ram.allocate(testKey(), layout);
    const auto replacement = ram.allocate(replacement_key, layout);
    std::fill(replacement.kv_storage->begin(), replacement.kv_storage->end(), 0x83);
    {
        DiskPrefixStorageBackend disk(archivePath(directory), layout.totalBytes(), kModelArtifactIdentity);
        std::string error;
        ASSERT_TRUE(disk.writeBlock(original, nullptr, nullptr, &error)) << error;
        old_payload = singlePayloadPath(archivePath(directory));
        PayloadRetirementFailure failure(old_payload);
        EXPECT_FALSE(disk.writeBlock(replacement, nullptr, nullptr, &error));
        EXPECT_TRUE(failure.injected.load());
        EXPECT_FALSE(disk.ready());
        EXPECT_NE(error.find("failed to retire committed prefix payload"), std::string::npos);
        EXPECT_FALSE(disk.writeBlock(original, nullptr, nullptr, &error));
        EXPECT_TRUE(std::filesystem::exists(old_payload));
        EXPECT_EQ(disk.usedBytes(), replacement.total_bytes);
    }
    {
        DiskPrefixStorageBackend reopened(archivePath(directory), layout.totalBytes(), kModelArtifactIdentity);
        ASSERT_TRUE(reopened.ready()) << reopened.initializationError();
        EXPECT_FALSE(std::filesystem::exists(old_payload));
        EXPECT_EQ(std::filesystem::file_size(singlePayloadPath(archivePath(directory))), replacement.total_bytes);
        std::string error;
        PrefixBlockHandle restored;
        ASSERT_TRUE(reopened.readBlock(replacement_key, layout, &restored, &error)) << error;
        EXPECT_EQ(*restored.kv_storage, *replacement.kv_storage);
    }
    std::filesystem::remove_all(directory);
}

/** @test Payload retirement cannot pass a compacted journal's undurable rename. */
TEST(Test__DiskPrefixStorageBackend, JournalPublicationRetainsWriterLeaseUntilDurable)
{
    const auto directory = tempDir();
    {
        RamPrefixStorageBackend ram(1024u);
        const auto source = ram.allocate(testKey(), makeLayout());
        DiskPrefixStorageBackend disk(archivePath(directory), 1024u, kModelArtifactIdentity);
        std::string error;
        ASSERT_TRUE(disk.writeBlock(source, nullptr, nullptr, &error)) << error;
        {
            JournalPublicationBarrier barrier(disk);
            ASSERT_TRUE(disk.requestCompaction(&error)) << error;
            ASSERT_TRUE(barrier.waitForPublication());
            // A separate descriptor models another process's writer authority.
            // If it can enter before directory fsync, it can unlink a payload
            // still referenced by the pre-rename journal after a power loss.
            const auto path = disk.archivePath().string() + ".lock";
            const int peer = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
            ASSERT_GE(peer, 0);
            const int result = ::flock(peer, LOCK_EX | LOCK_NB);
            const int lock_error = errno;
            EXPECT_EQ(result, -1);
            if (result != 0) EXPECT_EQ(lock_error, EWOULDBLOCK);
            if (result == 0) (void)::flock(peer, LOCK_UN);
            ::close(peer);
        }
        EXPECT_GT(disk.compactionStatus().publications, 0u);
        EXPECT_TRUE(disk.ready()) << disk.initializationError();
    }
    std::filesystem::remove_all(directory);
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

TEST(Test__DiskPrefixStorageBackend, WritesAndReadsRamBlockWithMetadata)
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

TEST(Test__DiskPrefixStorageBackend, PayloadContentsAreOpaqueToMetadataValidation)
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

    const auto payload_path = singlePayloadPath(disk.archivePath());
    const auto archive_bytes = std::filesystem::file_size(payload_path);
    ASSERT_GT(archive_bytes, 17u);
    std::fstream archive(
        payload_path,
        std::ios::binary | std::ios::in | std::ios::out);
    archive.seekg(static_cast<std::streamoff>(archive_bytes - 1));
    char byte = 0;
    archive.read(&byte, 1);
    byte ^= 0x5a;
    archive.seekp(static_cast<std::streamoff>(archive_bytes - 1));
    archive.write(&byte, 1);
    archive.close();

    PrefixBlockHandle hydrated;
    ASSERT_TRUE(disk.readBlock(testKey(), layout, &hydrated, &error)) << error;
    EXPECT_EQ(hydrated.terminal_logits_storage->back(), static_cast<uint8_t>(byte));
    EXPECT_EQ(hydrated.payload_identity, handle.payload_identity);
    cleanup();
}

TEST(Test__DiskPrefixStorageBackend, DiscoversCommittedBlocksAfterRestart)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };
    const auto layout = makeLayout();
    PrefixPayloadIdentity original_identity;

    {
        RamPrefixStorageBackend ram(1024);
        auto handle = ram.allocate(testKey(), layout);
        ASSERT_TRUE(handle.valid());
        original_identity = handle.payload_identity;
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
    EXPECT_EQ(entries.front().payload_identity, original_identity);

    PrefixBlockHandle hydrated;
    ASSERT_TRUE(reopened.readBlock(testKey(), layout, &hydrated, &error)) << error;
    EXPECT_EQ(hydrated.payload_identity, original_identity);
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
