/**
 * @file ImmutableLookupWorkerAffinityObservation.h
 * @brief Test-only exact-inode observation of the real immutable service worker.
 *
 * Every fstat delegates the original libc function and preserves its result and
 * errno. An explicit scoped fixture arms only after real service construction;
 * source bytes, metadata validation and protocol behavior remain unchanged.
 */
#pragma once
#include <atomic>
#include <exception>
#include <array>
#include <cerrno>
#include <dlfcn.h>
#include <mutex>
#include <omp.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdexcept>
#include <vector>

namespace immutable_lookup_affinity_test {
using Fstat = int (*)(int, struct stat *);
/** @return Original libc fstat, resolved before a fixture arms observation. */
inline Fstat realFstat()
{
    static const auto function = reinterpret_cast<Fstat>(dlsym(RTLD_NEXT, "fstat"));
    if (!function) std::terminate();
    return function;
}
/** @brief One actual source-validator thread and its scheduler placement. */
struct Observation { pid_t tid{}; cpu_set_t mask{}; };
inline std::atomic<bool> armed{false};
// Interposed calls from unrelated runtime threads can outlive a fixture's
// disarm. Protect both filtering metadata and records, including those callers,
// rather than assuming that joining the service quiesces every libc caller.
inline std::mutex observationMutex;
inline unsigned count{};
inline std::array<Observation, 64> observations{};
inline dev_t device{};
inline ino_t inode{};
inline pid_t creator{};
/** @brief Read only this actual calling thread's scheduler mask. */
inline cpu_set_t mask()
{
    cpu_set_t result; CPU_ZERO(&result);
    if (sched_getaffinity(0, sizeof(result), &result)) throw std::runtime_error("Lookup affinity observation failed");
    return result;
}
/** @brief Bound cold construction and exact-inode observation to one fixture lifetime. */
class Probe final {
public:
    /** @param fd Original populated synthetic table descriptor. */
    explicit Probe(int fd)
    {
        (void)realFstat();
        struct stat source{};
        if (realFstat()(fd, &source)) throw std::runtime_error("Lookup fixture fstat failed");
        std::lock_guard guard(observationMutex);
        if (armed.load(std::memory_order_relaxed)) throw std::logic_error("Overlapping lookup affinity observers");
        original_ = mask(); expected_ = original_; before_ = original_;
        if (omp_get_level() != 0) throw std::runtime_error("Lookup affinity fixture requires outer host");
        if (omp_get_proc_bind() != omp_proc_bind_false) {
            CPU_ZERO(&expected_);
            const int size = omp_get_partition_num_places();
            if (size <= 0) throw std::runtime_error("Missing admitted lookup partition");
            std::vector<int> places(static_cast<size_t>(size)); omp_get_partition_place_nums(places.data());
            cpu_set_t first; CPU_ZERO(&first);
            for (int i = 0; i < size; ++i) {
                const int width = omp_get_place_num_procs(places[i]);
                if (width <= 0) throw std::runtime_error("Empty lookup CPU place");
                std::vector<int> cpus(static_cast<size_t>(width)); omp_get_place_proc_ids(places[i], cpus.data());
                for (const int cpu : cpus) {
                    if (cpu < 0 || cpu >= CPU_SETSIZE) throw std::runtime_error("Lookup CPU exceeds mask");
                    CPU_SET(cpu, &expected_); if (i == 0) CPU_SET(cpu, &first);
                }
            }
            // Reproduce the legal initial-thread single-place creator even when
            // this native fixture is called on an already-independent GPU worker.
            before_ = first;
            if (sched_setaffinity(0, sizeof(first), &first)) throw std::runtime_error("Cannot narrow fixture creator");
        }
        device = source.st_dev; inode = source.st_ino; creator = static_cast<pid_t>(syscall(SYS_gettid));
        count = 0;
    }
    /** @brief Require constructor restoration, restore fixture creator, then observe real service work. */
    void created()
    {
        const auto after = mask();
        if (!CPU_EQUAL(&before_, &after)) throw std::runtime_error("Service constructor changed parent affinity");
        restore();
        std::lock_guard guard(observationMutex);
        armed.store(true, std::memory_order_release);
    }
    /** @brief Check actual non-parent source validator thread and persistent ownership. */
    void verify()
    {
        const auto parent = mask();
        if (!CPU_EQUAL(&parent, &original_)) throw std::runtime_error("Service generation changed creator affinity");
        std::lock_guard guard(observationMutex);
        const auto size = count;
        if (size <= verified_ || size > observations.size())
            throw std::runtime_error("Missing fresh/overflowed original worker observation");
        for (unsigned i = 0; i < size; ++i) {
            const auto &row = observations[i];
            if (row.tid == creator || row.tid != observations[0].tid || !CPU_EQUAL(&row.mask, &expected_))
                throw std::runtime_error("Actual immutable service worker lost original admitted partition/identity");
        }
        verified_ = size;
    }
    /** @brief Disarm before source teardown and restore on every assertion/exception exit. */
    ~Probe()
    {
        {
            std::lock_guard guard(observationMutex);
            armed.store(false, std::memory_order_release);
        }
        restore();
    }
    Probe(const Probe &) = delete;
    Probe &operator=(const Probe &) = delete;
private:
    /** @brief Exact test caller restoration; never modify the service child. */
    void restore() noexcept { if (sched_setaffinity(0, sizeof(original_), &original_)) std::terminate(); }
    cpu_set_t original_{}, before_{}, expected_{};
    unsigned verified_{}; ///< Every checked generation must contribute a fresh actual service observation.
};
}
/** @brief Delegate libc unchanged; observe only exact armed fixture inode on its actual worker. */
extern "C" int fstat(int fd, struct stat *result) noexcept
{
    const int incoming = errno;
    const auto original = immutable_lookup_affinity_test::realFstat();
    errno = incoming;
    const int rc = original(fd, result);
    const int saved = errno;
    using namespace immutable_lookup_affinity_test;
    if (rc == 0 && armed.load(std::memory_order_acquire)) {
        std::lock_guard guard(observationMutex);
        if (armed.load(std::memory_order_relaxed) && result->st_dev == device && result->st_ino == inode) {
            const auto tid = static_cast<pid_t>(syscall(SYS_gettid));
            if (tid != creator) {
                if (count < observations.size()) {
                    observations[count].tid = tid;
                    if (sched_getaffinity(0, sizeof(cpu_set_t), &observations[count].mask)) std::terminate();
                }
                // Saturate at one overflow sentinel; an unexpected concurrent
                // matching caller is recorded safely and fails identity checks.
                if (count <= observations.size()) ++count;
            }
        }
    }
    errno = saved;
    return rc;
}
