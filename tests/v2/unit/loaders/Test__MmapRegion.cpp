/**
 * @file Test__MmapRegion.cpp
 * @brief Device-free policy and lifecycle tests for mapped model reclamation.
 *
 * These tests cover prefault behavior, address provenance, backing-filesystem
 * classification, tmpfs/ramfs reclaim suppression, and the exactly-once
 * asynchronous operation used after first prefill. They intentionally use
 * tiny local/memfd files and never load a model or initialize a GPU runtime.
 */

#include <gtest/gtest.h>

#include "loaders/MmapRegion.h"
#include "loaders/MmapReclaimLifecycle.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <string>
#include <thread>

#include <cstdlib>
#include <fcntl.h>
#include <linux/magic.h>
#include <linux/memfd.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <numa.h>

namespace llaminar2::test
{
    namespace
    {
        class ScopedEnvVar
        {
        public:
            ScopedEnvVar(const char *name, const char *value)
                : name_(name)
            {
                const char *old = std::getenv(name);
                if (old)
                {
                    had_old_ = true;
                    old_value_ = old;
                }
                if (value)
                    setenv(name, value, 1);
                else
                    unsetenv(name);
            }

            ~ScopedEnvVar()
            {
                if (had_old_)
                    setenv(name_.c_str(), old_value_.c_str(), 1);
                else
                    unsetenv(name_.c_str());
            }

            ScopedEnvVar(const ScopedEnvVar &) = delete;
            ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;

        private:
            std::string name_;
            bool had_old_ = false;
            std::string old_value_;
        };

        /**
         * @brief Creates a small temporary file that is safe to mmap in unit tests.
         *
         * The contents are irrelevant; the test is about the mmap prefault policy
         * chosen by MmapRegion::create(), not filesystem throughput.
         */
        class TemporaryMmapFile
        {
        public:
            TemporaryMmapFile()
            {
                path_ = std::filesystem::temp_directory_path() /
                        ("llaminar_mmap_region_test_" + std::to_string(::getpid()) + ".bin");
                std::ofstream out(path_, std::ios::binary | std::ios::trunc);
                std::string page(4096, '\x5a');
                out.write(page.data(), static_cast<std::streamsize>(page.size()));
            }

            ~TemporaryMmapFile()
            {
                std::error_code ignored;
                std::filesystem::remove(path_, ignored);
            }

            const std::filesystem::path &path() const { return path_; }

        private:
            std::filesystem::path path_;
        };

        /**
         * @brief Owns a small memfd whose fstatfs identity is tmpfs.
         *
         * Opening the procfs descriptor path gives MmapRegion its own read-only
         * descriptor while this helper keeps the anonymous file named and alive.
         */
        class MemoryBackedMmapFile
        {
        public:
            MemoryBackedMmapFile()
            {
                fd_ = static_cast<int>(::syscall(
                    SYS_memfd_create,
                    "llaminar_mmap_reclaim_test",
                    MFD_CLOEXEC));
                if (fd_ < 0)
                {
                    return;
                }

                constexpr off_t bytes = 3 * 4096;
                if (::ftruncate(fd_, bytes) != 0)
                {
                    ::close(fd_);
                    fd_ = -1;
                    return;
                }

                const unsigned char first = 0x5a;
                const unsigned char last = 0xa5;
                if (::pwrite(fd_, &first, 1, 0) != 1 ||
                    ::pwrite(fd_, &last, 1, bytes - 1) != 1)
                {
                    ::close(fd_);
                    fd_ = -1;
                    return;
                }
                path_ = "/proc/self/fd/" + std::to_string(fd_);
            }

            ~MemoryBackedMmapFile()
            {
                if (fd_ >= 0)
                    ::close(fd_);
            }

            MemoryBackedMmapFile(const MemoryBackedMmapFile &) = delete;
            MemoryBackedMmapFile &operator=(const MemoryBackedMmapFile &) = delete;

            /** @return Whether memfd creation and initialization succeeded. */
            [[nodiscard]] bool valid() const noexcept { return fd_ >= 0; }

            /** @return Procfs path that can be independently opened by MmapRegion. */
            [[nodiscard]] const std::string &path() const noexcept { return path_; }

        private:
            int fd_ = -1;
            std::string path_;
        };
    } // namespace

    TEST(Test__MmapRegion, ClassifiesEverySupportedMemoryFilesystemMagic)
    {
        using Policy = MmapRegion::PageReclaimPolicy;
        EXPECT_EQ(
            MmapRegion::classifyPageReclaimPolicy(TMPFS_MAGIC),
            Policy::RetainMemoryFilesystemPages);
#ifdef RAMFS_MAGIC
        EXPECT_EQ(
            MmapRegion::classifyPageReclaimPolicy(RAMFS_MAGIC),
            Policy::RetainMemoryFilesystemPages);
#endif
        EXPECT_EQ(
            MmapRegion::classifyPageReclaimPolicy(EXT4_SUPER_MAGIC),
            Policy::AdviseDontneed);
        EXPECT_EQ(
            MmapRegion::classifyPageReclaimPolicy(0x7fffffffL),
            Policy::AdviseDontneed)
            << "Unknown durable filesystems must retain the historical advice policy";
    }

    TEST(Test__MmapRegion, MemfdCarriesRetainPolicyAcrossWholeAndRangeAdvice)
    {
        MemoryBackedMmapFile file;
        ASSERT_TRUE(file.valid());

        auto region = MmapRegion::create(
            file.path(),
            /*numa_node=*/-1,
            /*skip_cache_eviction=*/false,
            MmapRegion::PrefaultPolicy::DemandPaged);
        ASSERT_NE(region, nullptr);
        EXPECT_EQ(
            region->pageReclaimPolicy(),
            MmapRegion::PageReclaimPolicy::RetainMemoryFilesystemPages);

        const auto source =
            MmapRegion::resolveFileSource(region->data() + 4093, 9);
        ASSERT_TRUE(source.has_value());
        EXPECT_EQ(
            source->page_reclaim_policy,
            MmapRegion::PageReclaimPolicy::RetainMemoryFilesystemPages);

        EXPECT_EQ(region->adviseDontneed(), 0u);
        EXPECT_EQ(
            MmapRegion::adviseDontneedRange(region->data() + 4093, 9),
            0u);
        EXPECT_EQ(region->data()[0], 0x5a);
        EXPECT_EQ(region->data()[region->size() - 1], 0xa5);
    }

    TEST(Test__MmapReclaimLifecycle, SubmissionIsNonBlockingExactlyOnceAndBarrierWaits)
    {
        std::mutex mutex;
        std::condition_variable cv;
        bool worker_entered = false;
        bool release_worker = false;
        std::atomic<int> calls{0};

        MmapReclaimLifecycle lifecycle([&]() -> std::size_t {
            ++calls;
            std::unique_lock<std::mutex> lock(mutex);
            worker_entered = true;
            cv.notify_all();
            cv.wait(lock, [&]() { return release_worker; });
            return 12345u;
        });

        EXPECT_EQ(
            lifecycle.schedule(),
            MmapReclaimLifecycle::Submission::Scheduled);

        {
            std::unique_lock<std::mutex> lock(mutex);
            const bool entered = cv.wait_for(
                lock,
                std::chrono::seconds(2),
                [&]() { return worker_entered; });
            if (!entered)
            {
                release_worker = true;
                lock.unlock();
                cv.notify_all();
                FAIL() << "Prestarted reclaim worker did not observe submission";
            }
        }

        EXPECT_EQ(lifecycle.state(), MmapReclaimLifecycle::State::Running);
        EXPECT_EQ(
            lifecycle.schedule(),
            MmapReclaimLifecycle::Submission::AlreadyRunning);

        auto allocation_barrier = std::async(std::launch::async, [&]() {
            return lifecycle.awaitBeforeHostAllocation();
        });
        EXPECT_EQ(
            allocation_barrier.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout)
            << "Allocation barrier returned before reclaim capacity was published";

        {
            std::lock_guard<std::mutex> lock(mutex);
            release_worker = true;
        }
        cv.notify_all();

        const auto completion = allocation_barrier.get();
        EXPECT_EQ(completion.state, MmapReclaimLifecycle::State::Complete);
        EXPECT_EQ(completion.advised_bytes, 12345u);
        EXPECT_EQ(calls.load(), 1);
        EXPECT_EQ(
            lifecycle.schedule(),
            MmapReclaimLifecycle::Submission::AlreadyComplete);
    }

    TEST(Test__MmapReclaimLifecycle, TeardownJoinsSubmittedWork)
    {
        std::atomic<bool> completed{false};
        {
            MmapReclaimLifecycle lifecycle([&]() -> std::size_t {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                completed.store(true);
                return 0;
            });
            EXPECT_EQ(
                lifecycle.schedule(),
                MmapReclaimLifecycle::Submission::Scheduled);
        }
        EXPECT_TRUE(completed.load())
            << "Lifecycle destruction must join borrowed owner work";
    }

    TEST(Test__MmapRegion, DemandPagedPolicyDoesNotEagerPrefault)
    {
        TemporaryMmapFile file;

        auto region = MmapRegion::create(
            file.path().string(),
            /*numa_node=*/-1,
            /*skip_cache_eviction=*/false,
            MmapRegion::PrefaultPolicy::DemandPaged);

        ASSERT_NE(region, nullptr);
        EXPECT_FALSE(region->wasEagerPrefaulted())
            << "GPU-target mmap must not use MAP_POPULATE or NUMA first-touch";
    }

    TEST(Test__MmapRegion, AutoPolicyPreservesHistoricalEagerPrefault)
    {
        TemporaryMmapFile file;

        auto region = MmapRegion::create(
            file.path().string(),
            /*numa_node=*/-1,
            /*skip_cache_eviction=*/false,
            MmapRegion::PrefaultPolicy::Auto);

        ASSERT_NE(region, nullptr);
        EXPECT_TRUE(region->wasEagerPrefaulted())
            << "CPU/non-GPU mmap keeps the historical eager-prefault behavior";
    }

    TEST(Test__MmapRegion, ResolvesMappedSubrangeToFileCoordinates)
    {
        TemporaryMmapFile file;
        auto region = MmapRegion::create(
            file.path().string(),
            /*numa_node=*/-1,
            /*skip_cache_eviction=*/false,
            MmapRegion::PrefaultPolicy::DemandPaged);
        ASSERT_NE(region, nullptr);

        const auto source =
            MmapRegion::resolveFileSource(region->data() + 37, 113);

        ASSERT_TRUE(source.has_value());
        EXPECT_EQ(source->path, file.path().string());
        EXPECT_EQ(source->offset, uint64_t{37});
        EXPECT_EQ(source->available_bytes, region->size() - 37);
        EXPECT_FALSE(MmapRegion::resolveFileSource(&source, sizeof(source)).has_value())
            << "Heap addresses must never be mistaken for model-file ranges";
    }

    TEST(Test__MmapRegion, RequestedNumaBindFailureFailsFastByDefault)
    {
        ScopedEnvVar allow_fallback("LLAMINAR_ALLOW_NUMA_BIND_FALLBACK", nullptr);
        TemporaryMmapFile file;

        int requested_node = numa_available() >= 0 ? numa_max_node() + 1 : 0;

        auto region = MmapRegion::create(
            file.path().string(),
            requested_node,
            /*skip_cache_eviction=*/false,
            MmapRegion::PrefaultPolicy::Auto);

        EXPECT_EQ(region, nullptr)
            << "Requested NUMA placement must not silently fall back when binding cannot be satisfied";
    }
} // namespace llaminar2::test
