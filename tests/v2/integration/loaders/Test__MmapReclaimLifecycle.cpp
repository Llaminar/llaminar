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
 * Separate native-affinity cases exercise the actual reclaim constructor and
 * worker under enabled/disabled OpenMP binding, explicit pthread defaults and
 * one-shot construction failure, without touching model files or accelerators.
 */

#include <gtest/gtest.h>

#include "loaders/MmapReclaimLifecycle.h"
#include "loaders/MmapRegion.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <future>
#include <fcntl.h>
#include <linux/memfd.h>
#include <memory>
#include <mutex>
#include <omp.h>
#include <sched.h>
#include <string>
#include <system_error>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <pthread.h>

namespace mmap_reclaim_affinity_fault
{
    thread_local bool fail_next_thread = false;
}

/** @brief Inject one native constructor failure; every ordinary call preserves libc attributes exactly. */
extern "C" int pthread_create(pthread_t *thread, const pthread_attr_t *attributes,
                              void *(*entry)(void *), void *argument) noexcept
{
    if (mmap_reclaim_affinity_fault::fail_next_thread)
    {
        mmap_reclaim_affinity_fault::fail_next_thread = false;
        return EAGAIN;
    }
    using Create = int (*)(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
    static const auto native = reinterpret_cast<Create>(dlsym(RTLD_NEXT, "pthread_create"));
    if (!native) std::abort();
    return native(thread, attributes, entry, argument);
}

namespace llaminar2::test
{
    namespace
    {
        /** @return The calling OS thread's exact scheduler mask, or a precise observation failure. */
        cpu_set_t reclaimCallerMask()
        {
            cpu_set_t value{};
            if (sched_getaffinity(0, sizeof(value), &value) != 0)
                throw std::system_error(errno, std::generic_category(), "Cannot observe mmap reclaim caller mask");
            return value;
        }

        /** @return Independent initial-rank place union, or the unchanged native unbound mask. */
        cpu_set_t reclaimPartitionMask()
        {
            const char *policy = std::getenv("OMP_PROC_BIND");
            if (!policy || (std::string(policy) != "close" && std::string(policy) != "false") ||
                ((omp_get_proc_bind() == omp_proc_bind_false) != (std::string(policy) == "false")))
                throw std::runtime_error("Mmap affinity requires an authenticated pre-exec binding policy");
            if (omp_get_proc_bind() == omp_proc_bind_false) return reclaimCallerMask();
            if (omp_get_num_places() < 2)
                throw std::runtime_error("Mmap affinity requires two admitted OpenMP places");
            cpu_set_t result{};
            for (int place = 0; place < omp_get_num_places(); ++place)
            {
                std::vector<int> cpus(static_cast<std::size_t>(omp_get_place_num_procs(place)));
                omp_get_place_proc_ids(place, cpus.data());
                for (const int cpu : cpus) CPU_SET(cpu, &result);
            }
            return result;
        }

        /** @brief Restore the complete caller-installed pthread default attribute set after one check. */
        class ReclaimDefaultAffinity final
        {
        public:
            /** @brief Install one exact native affinity override without changing other attributes. */
            explicit ReclaimDefaultAffinity(const cpu_set_t &mask)
            {
                pthread_attr_t changed;
                if (pthread_getattr_default_np(&original_) != 0 || pthread_getattr_default_np(&changed) != 0)
                    throw std::runtime_error("Cannot retain reclaim pthread defaults");
                if (pthread_attr_setaffinity_np(&changed, sizeof(mask), &mask) != 0 ||
                    pthread_setattr_default_np(&changed) != 0 || pthread_attr_destroy(&changed) != 0)
                    std::terminate();
            }
            /** @brief Never leave altered defaults for a later lifecycle owner. */
            ~ReclaimDefaultAffinity()
            {
                if (pthread_setattr_default_np(&original_) != 0 || pthread_attr_destroy(&original_) != 0)
                    std::terminate();
            }
        private:
            pthread_attr_t original_{};
        };

        /** @brief Inspect actual deferred work/descendant masks and exact creator restoration. */
        void checkReclaimWorkerMask(const cpu_set_t &expected, int repetitions)
        {
            const auto creator = reclaimCallerMask();
            for (int iteration = 0; iteration < repetitions; ++iteration)
            {
                cpu_set_t worker{}, descendant{};
                {
                    MmapReclaimLifecycle lifecycle([&]() -> std::size_t {
                        worker = reclaimCallerMask();
                        std::thread child([&] { descendant = reclaimCallerMask(); });
                        child.join();
                        return 17u;
                    });
                    const auto constructed = reclaimCallerMask();
                    EXPECT_TRUE(CPU_EQUAL(&creator, &constructed));
                    EXPECT_EQ(lifecycle.schedule(), MmapReclaimLifecycle::Submission::Scheduled);
                    const auto completion = lifecycle.awaitBeforeHostAllocation();
                    EXPECT_EQ(completion.state, MmapReclaimLifecycle::State::Complete);
                    EXPECT_EQ(completion.advised_bytes, 17u);
                    EXPECT_TRUE(CPU_EQUAL(&worker, &expected));
                    EXPECT_TRUE(CPU_EQUAL(&descendant, &expected));
                    EXPECT_EQ(lifecycle.schedule(), MmapReclaimLifecycle::Submission::AlreadyComplete);
                }
                const auto retired = reclaimCallerMask();
                EXPECT_TRUE(CPU_EQUAL(&creator, &retired));
            }
        }
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

    TEST(Test__MmapReclaimAffinity, SubmittedWorkerAndDescendantsRetainAdmissionAndNativeDefaults)
    {
        const auto expected = reclaimPartitionMask();
        checkReclaimWorkerMask(expected, 20);
        cpu_set_t explicit_default{};
        for (int cpu = CPU_SETSIZE - 1; cpu >= 0; --cpu)
            if (CPU_ISSET(cpu, &expected)) { CPU_SET(cpu, &explicit_default); break; }
        const ReclaimDefaultAffinity defaults(explicit_default);
        checkReclaimWorkerMask(explicit_default, 1);
    }

    TEST(Test__MmapReclaimAffinity, IdleAndFailedWorkerRetirementPreserveCreator)
    {
        (void)reclaimPartitionMask();
        const auto creator = reclaimCallerMask();
        {
            MmapReclaimLifecycle idle([] { return 0u; });
            EXPECT_EQ(idle.awaitBeforeHostAllocation().state, MmapReclaimLifecycle::State::Idle);
        }
        {
            MmapReclaimLifecycle failed([]() -> std::size_t { throw std::runtime_error("authored reclaim failure"); });
            EXPECT_EQ(failed.schedule(), MmapReclaimLifecycle::Submission::Scheduled);
            EXPECT_THROW((void)failed.awaitBeforeHostAllocation(), std::runtime_error);
            EXPECT_EQ(failed.schedule(), MmapReclaimLifecycle::Submission::AlreadyFailed);
        }
        const auto retired = reclaimCallerMask();
        EXPECT_TRUE(CPU_EQUAL(&creator, &retired));
    }

    TEST(Test__MmapReclaimAffinity, RejectedConstructionAndNestedEntryRestoreCreator)
    {
        (void)reclaimPartitionMask();
        const auto creator = reclaimCallerMask();
        EXPECT_THROW((MmapReclaimLifecycle{MmapReclaimLifecycle::Work{}}), std::invalid_argument);
        mmap_reclaim_affinity_fault::fail_next_thread = true;
        EXPECT_THROW((MmapReclaimLifecycle{[] { return 0u; }}), std::system_error);
        EXPECT_FALSE(mmap_reclaim_affinity_fault::fail_next_thread);
        const auto rejected = reclaimCallerMask();
        EXPECT_TRUE(CPU_EQUAL(&creator, &rejected));
        bool nested_rejected = false, nested_preserved = false;
#pragma omp parallel num_threads(1)
        {
            try
            {
                const auto before = reclaimCallerMask();
                try { MmapReclaimLifecycle nested([] { return 0u; }); }
                catch (const std::runtime_error &error)
                {
                    nested_rejected = std::string(error.what()).find("outer host thread") != std::string::npos;
                    const auto after = reclaimCallerMask();
                    nested_preserved = CPU_EQUAL(&before, &after);
                }
            }
            catch (...) { nested_rejected = false; nested_preserved = false; }
        }
        EXPECT_TRUE(nested_rejected && nested_preserved);
        const auto restored = reclaimCallerMask();
        EXPECT_TRUE(CPU_EQUAL(&creator, &restored));
    }

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
