/**
 * @file Test__MmapReclaimLifecycle.cpp
 * @brief Model-free integration proof for asynchronous split-mmap reclamation.
 *
 * Production parity may expose the same ModelContext to rank and participant
 * lifecycle owners. This test models a split GGUF with three memory-backed
 * mappings, submits its reclaim boundary concurrently from several owners, and
 * proves that one background operation runs while all others observe the typed
 * idempotent state. It also verifies that tmpfs-backed page advice is suppressed
 * and that the explicit allocation barrier publishes completion before later
 * host allocation.
 */

#include <gtest/gtest.h>

#include "loaders/MmapReclaimLifecycle.h"
#include "loaders/MmapRegion.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <linux/memfd.h>
#include <memory>
#include <mutex>
#include <string>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Owns one memfd shard and the independent MmapRegion opened on it. */
        class MemoryBackedShard final
        {
        public:
            /**
             * @brief Create and map one deterministic memory-backed model shard.
             *
             * @param shard_index Stable byte pattern and diagnostic identity.
             * @param bytes Logical shard size; must be positive.
             */
            MemoryBackedShard(int shard_index, std::size_t bytes)
            {
                fd_ = static_cast<int>(::syscall(
                    SYS_memfd_create,
                    ("llaminar_reclaim_shard_" + std::to_string(shard_index)).c_str(),
                    MFD_CLOEXEC));
                if (fd_ < 0 || bytes == 0 ||
                    ::ftruncate(fd_, static_cast<off_t>(bytes)) != 0)
                {
                    return;
                }

                first_byte_ = static_cast<std::uint8_t>(0x30 + shard_index);
                const std::uint8_t last_byte =
                    static_cast<std::uint8_t>(0xc0 + shard_index);
                if (::pwrite(fd_, &first_byte_, 1, 0) != 1 ||
                    ::pwrite(fd_, &last_byte, 1, static_cast<off_t>(bytes - 1)) != 1)
                {
                    return;
                }

                path_ = "/proc/self/fd/" + std::to_string(fd_);
                region_ = MmapRegion::create(
                    path_,
                    /*numa_node=*/-1,
                    /*skip_cache_eviction=*/false,
                    MmapRegion::PrefaultPolicy::DemandPaged);
            }

            /** @brief Close the memfd after the mapping owner has retired. */
            ~MemoryBackedShard()
            {
                region_.reset();
                if (fd_ >= 0)
                    ::close(fd_);
            }

            MemoryBackedShard(const MemoryBackedShard &) = delete;
            MemoryBackedShard &operator=(const MemoryBackedShard &) = delete;

            /** @return Whether both file initialization and mapping succeeded. */
            [[nodiscard]] bool valid() const noexcept
            {
                return fd_ >= 0 && region_ != nullptr;
            }

            /** @return Mutable access to the owned mapping for lifecycle tests. */
            [[nodiscard]] MmapRegion &region() noexcept { return *region_; }

            /** @return Path naming this shard while its memfd remains alive. */
            [[nodiscard]] const std::string &path() const noexcept { return path_; }

            /** @return Expected first byte used to prove mapping continuity. */
            [[nodiscard]] std::uint8_t firstByte() const noexcept { return first_byte_; }

        private:
            int fd_ = -1;
            std::string path_;
            std::unique_ptr<MmapRegion> region_;
            std::uint8_t first_byte_ = 0;
        };
    } // namespace

    TEST(Test__MmapReclaimLifecycleIntegration,
         SplitMemoryFilesystemMappingsReclaimExactlyOnceOffAuthorityThread)
    {
        constexpr std::size_t kShardBytes = 2u * 1024u * 1024u;
        std::vector<std::unique_ptr<MemoryBackedShard>> shards;
        for (int index = 0; index < 3; ++index)
        {
            auto shard = std::make_unique<MemoryBackedShard>(index, kShardBytes);
            ASSERT_TRUE(shard->valid());
            ASSERT_EQ(
                shard->region().pageReclaimPolicy(),
                MmapRegion::PageReclaimPolicy::RetainMemoryFilesystemPages);
            EXPECT_TRUE(MmapRegion::prepopulatePageCache(shard->path()))
                << "Memory-backed shards require no synthetic page-cache read";
            shards.push_back(std::move(shard));
        }

        std::mutex worker_mutex;
        std::condition_variable worker_cv;
        bool worker_entered = false;
        bool allow_reclaim = false;
        bool residual_host_release_complete = false;
        std::atomic<int> reclaim_calls{0};
        const std::thread::id authority_thread = std::this_thread::get_id();
        std::thread::id reclaim_thread;

        MmapReclaimLifecycle lifecycle([&]() -> std::size_t {
            ++reclaim_calls;
            {
                std::unique_lock<std::mutex> lock(worker_mutex);
                reclaim_thread = std::this_thread::get_id();
                worker_entered = true;
                worker_cv.notify_all();
                worker_cv.wait(lock, [&]() { return allow_reclaim; });
                /* Model residual buffers are retired before registration and
                 * page advice in the production WeightManager operation. */
                residual_host_release_complete = true;
            }

            std::size_t advised = 0;
            for (const auto &shard : shards)
            {
                advised += shard->region().adviseDontneed();
                advised += MmapRegion::adviseDontneedRange(
                    shard->region().data() + 31,
                    8192);
            }
            return advised;
        });

        constexpr int kLifecycleOwners = 8;
        std::vector<MmapReclaimLifecycle::Submission> submissions(
            kLifecycleOwners,
            MmapReclaimLifecycle::Submission::AlreadyFailed);
        std::vector<std::thread> submitters;
        submitters.reserve(kLifecycleOwners);
        for (int owner = 0; owner < kLifecycleOwners; ++owner)
        {
            submitters.emplace_back([&, owner]() {
                submissions[owner] = lifecycle.schedule();
            });
        }
        for (auto &submitter : submitters)
            submitter.join();

        {
            std::unique_lock<std::mutex> lock(worker_mutex);
            const bool entered = worker_cv.wait_for(
                lock,
                std::chrono::seconds(2),
                [&]() { return worker_entered; });
            if (!entered)
            {
                allow_reclaim = true;
                lock.unlock();
                worker_cv.notify_all();
                FAIL() << "Reclaim worker did not observe the concurrent submissions";
            }
        }

        const auto scheduled_count = std::count(
            submissions.begin(),
            submissions.end(),
            MmapReclaimLifecycle::Submission::Scheduled);
        const auto already_running_count = std::count(
            submissions.begin(),
            submissions.end(),
            MmapReclaimLifecycle::Submission::AlreadyRunning);
        EXPECT_EQ(scheduled_count, 1);
        EXPECT_EQ(already_running_count, kLifecycleOwners - 1);
        EXPECT_EQ(reclaim_calls.load(), 1);

        {
            std::lock_guard<std::mutex> lock(worker_mutex);
            allow_reclaim = true;
        }
        worker_cv.notify_all();

        const auto completion = lifecycle.awaitBeforeHostAllocation();
        EXPECT_EQ(completion.state, MmapReclaimLifecycle::State::Complete);
        EXPECT_EQ(completion.advised_bytes, 0u)
            << "tmpfs/ramfs mappings must not perform page-table discard work";
        EXPECT_NE(reclaim_thread, authority_thread)
            << "First-prefill reclamation must execute on the prestarted worker";
        EXPECT_TRUE(residual_host_release_complete)
            << "The allocation barrier must include residual host-buffer release";

        // This allocation represents JIT/model admission after it has crossed
        // the explicit reclaim dependency rather than waiting in inference.
        std::vector<std::uint8_t> admitted_host_storage(4u * 1024u * 1024u, 0x7b);
        ASSERT_EQ(admitted_host_storage.front(), 0x7b);
        ASSERT_EQ(admitted_host_storage.back(), 0x7b);

        for (const auto &shard : shards)
        {
            EXPECT_EQ(shard->region().data()[0], shard->firstByte())
                << "Retained memory-filesystem mapping must remain readable";
        }
    }
} // namespace llaminar2::test
