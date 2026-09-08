/**
 * @file NativeVNNIProfilerControl.h
 * @brief Shared control-plane helpers for isolated NativeVNNI profiler launches.
 *
 * Canonical NativeVNNI timing and hardware-counter collection are separate
 * process invocations. Backend trainers use the request ID exposed here only
 * to append one extra production launch after setup, correctness checks, and
 * warmup have completed. That launch is never inserted into a timing sample.
 * A CPU trainer process may profile several exact requests sequentially to
 * amortize fixture and OpenMP-team setup, but every request still owns a fresh
 * reset/enable/disable/read transaction and a separate atomic output file.
 *
 * GPU backends use their native profiler start/stop APIs around the launch.
 * CPU profiling uses process-owned Linux `perf_event_open` groups attached to
 * the exact persistent OpenMP thread IDs after warmup. `LinuxPerfControl`
 * resets and enables those groups immediately before one grouped kernel,
 * disables them immediately after all workers return, and writes the scaled
 * counts in the same semicolon-separated representation consumed by the parser.
 * No external polling loop or system-wide counter interval exists, so another
 * process or a delayed `perf stat` control acknowledgement cannot contaminate
 * the candidate's evidence.
 *
 * This header belongs only to performance tooling. Production inference never
 * reads these environment variables and never opens profiler-control FIFOs.
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <linux/perf_event.h>
#include <omp.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

namespace llaminar2::test::native_vnni_dispatch
{
    /** Environment variable carrying the immutable profiler request ID. */
    inline constexpr const char *kProfilerRequestIdEnvironment =
        "LLAMINAR_NATIVE_VNNI_PROFILE_REQUEST_ID";

    /** Output path for one process-owned CPU hardware-counter transaction. */
    inline constexpr const char *kPerfStatsPathEnvironment =
        "LLAMINAR_NATIVE_VNNI_PERF_STATS_PATH";

    /** Optional TSV plan for several sequential exact CPU profile requests. */
    inline constexpr const char *kProfilerBatchPathEnvironment =
        "LLAMINAR_NATIVE_VNNI_PROFILE_BATCH_PATH";

    /** Return a trimmed environment string or an empty string when absent. */
    inline std::string profilerEnvironment(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw)
            return {};
        std::string value(raw);
        const size_t begin = value.find_first_not_of(" \t\r\n");
        if (begin == std::string::npos)
            return {};
        const size_t end = value.find_last_not_of(" \t\r\n");
        return value.substr(begin, end - begin + 1);
    }

    /** Return the selected profiler request ID, or empty in ordinary sweeps. */
    inline std::string profilerRequestId()
    {
        return profilerEnvironment(kProfilerRequestIdEnvironment);
    }

    /** Return whether this process must execute one isolated profile launch. */
    inline bool isolatedProfilerLaunchRequested()
    {
        return !profilerRequestId().empty() ||
               !profilerEnvironment(kProfilerBatchPathEnvironment).empty();
    }

    /**
     * @brief Process-scoped Linux perf region for one CPU candidate launch.
     *
     * After warmup creates the persistent OpenMP team, ``prepare()`` attaches
     * one schedulable event group to each exact worker TID. No inheritance and
     * no system-wide CPU scope is involved. ``begin()`` arms worker groups
     * before the main-thread group; ``end()`` disables the main thread before
     * parked workers. This ordering keeps control syscalls outside the measured
     * main-thread interval while profiler-only passive worker waiting keeps the
     * other groups quiescent until the requested production launch wakes them.
     */
    class LinuxPerfControl
    {
    public:
        /** Validate immutable request/output provenance without opening events. */
        explicit LinuxPerfControl(
            std::string request_id,
            std::string output_path = {})
            : request_id_(std::move(request_id))
        {
            if (request_id_.empty())
                throw std::runtime_error("LinuxPerfControl requires a request ID");
            output_path_ = std::move(output_path);
            if (output_path_.empty())
                output_path_ = profilerEnvironment(kPerfStatsPathEnvironment);
            if (output_path_.empty())
                throw std::runtime_error(
                    "isolated CPU profiling requires a perf stats output path");
        }

        /** Disable and close counters if an exception escapes the target. */
        ~LinuxPerfControl()
        {
            if (enabled_)
            {
                for (ThreadGroup &group : groups_)
                    (void)::ioctl(
                        group.leaderFd(),
                        PERF_EVENT_IOC_DISABLE,
                        PERF_IOC_FLAG_GROUP);
            }
            closeEvents();
        }

        /**
         * @brief Attach one disabled event group to each persistent OMP thread.
         *
         * ``thread_ids`` must be indexed by OpenMP thread number, making entry
         * zero the calling/main thread. The trainer gathers these IDs in a
         * dedicated untimed parallel region after warmup. Duplicate TIDs or a
         * second preparation indicate that the supposedly stable team changed
         * underneath one profiler request and are rejected.
         */
        void prepare(const std::vector<pid_t> &thread_ids)
        {
            if (!groups_.empty())
                throw std::runtime_error("Linux perf thread groups already prepared");
            if (thread_ids.empty())
                throw std::runtime_error("Linux perf requires an OpenMP thread team");
            std::vector<pid_t> unique = thread_ids;
            std::sort(unique.begin(), unique.end());
            if (std::adjacent_find(unique.begin(), unique.end()) != unique.end())
                throw std::runtime_error("Linux perf OpenMP thread IDs are not unique");
            try
            {
                for (pid_t thread_id : thread_ids)
                    groups_.push_back(openGroup(thread_id));
            }
            catch (...)
            {
                closeEvents();
                throw;
            }
        }

        /** Return whether the persistent worker event groups are open. */
        [[nodiscard]] bool prepared() const noexcept
        {
            return !groups_.empty();
        }

        /**
         * @brief Bind the next exact request to the reusable worker groups.
         *
         * Rebinding is legal only between completed transactions. Event groups
         * contain no retained count because ``begin()`` resets the complete
         * group before every launch. Changing both identities together keeps a
         * later publish from being attributed to an earlier batch member.
         */
        void rebind(std::string request_id, std::string output_path)
        {
            if (enabled_)
                throw std::runtime_error(
                    "cannot rebind an enabled Linux perf transaction");
            if (request_id.empty() || output_path.empty())
                throw std::runtime_error(
                    "Linux perf batch member requires request and output IDs");
            request_id_ = std::move(request_id);
            output_path_ = std::move(output_path);
            wall_clock_ns_ = 0;
        }

        /**
         * @brief Reset and enable all thread groups concurrently.
         *
         * The caller supplies the production OpenMP team's parallel-for
         * primitive. Running one ioctl on every worker concurrently avoids a
         * serial arm tail during which active-wait workers would accumulate
         * unrelated spin instructions. The exact wall interval starts only
         * after both control regions have joined.
         */
        template <typename ParallelFor>
        void begin(ParallelFor &&parallel_for)
        {
            if (enabled_)
                throw std::runtime_error("Linux perf region is already enabled");
            if (groups_.empty())
                throw std::runtime_error("Linux perf thread groups are not prepared");
            runConcurrentControl(
                parallel_for,
                PERF_EVENT_IOC_RESET,
                "reset");
            runConcurrentControl(
                parallel_for,
                PERF_EVENT_IOC_ENABLE,
                "enable");
            profile_begin_ = std::chrono::steady_clock::now();
            enabled_ = true;
        }

        /** Disable, read, scale, and publish the one completed target region. */
        template <typename ParallelFor>
        void end(ParallelFor &&parallel_for)
        {
            if (!enabled_)
                throw std::runtime_error("Linux perf region is not enabled");
            const auto profile_end = std::chrono::steady_clock::now();
            wall_clock_ns_ = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    profile_end - profile_begin_)
                    .count());
            runConcurrentControl(
                parallel_for,
                PERF_EVENT_IOC_DISABLE,
                "disable");
            enabled_ = false;
            publish();
        }

        LinuxPerfControl(const LinuxPerfControl &) = delete;
        LinuxPerfControl &operator=(const LinuxPerfControl &) = delete;

    private:
        /** One portable Linux perf event and its corpus-facing spelling. */
        struct EventSpec
        {
            const char *name;
            uint32_t type;
            uint64_t config;
            bool nanoseconds = false;
        };

        /** One opened event descriptor paired with its immutable definition. */
        struct Event
        {
            EventSpec spec;
            int fd = -1;
        };

        /** One simultaneously scheduled metric group for one exact OMP TID. */
        struct ThreadGroup
        {
            pid_t thread_id = -1;
            std::vector<Event> events;

            int leaderFd() const
            {
                if (events.empty())
                    throw std::runtime_error("Linux perf thread group is empty");
                return events.front().fd;
            }
        };

        /** Kernel read format for one independently multiplexed event. */
        struct EventRead
        {
            uint64_t value = 0;
            uint64_t time_enabled = 0;
            uint64_t time_running = 0;
        };

        /** Encode one portable hardware-cache event for `perf_event_open`. */
        static constexpr uint64_t cacheConfig(
            uint64_t cache,
            uint64_t result)
        {
            return cache |
                   (static_cast<uint64_t>(PERF_COUNT_HW_CACHE_OP_READ) << 8) |
                   (result << 16);
        }

        /** Return the complete reviewed CPU profiler metric inventory. */
        static const std::array<EventSpec, 7> &eventSpecs()
        {
            /*
             * Cascade Lake and the other supported CPU targets expose three
             * fixed counters plus four programmable counters per core. The
             * production host keeps the NMI watchdog enabled, consuming one
             * programmable slot, so this transaction uses at most three. Keep
             * the one-launch inventory within that physical capacity: a short
             * decode/GEMM launch cannot wait for Linux to multiplex an eleven-
             * event wish list, and an event with zero running time is not
             * evidence. L1/LLC behavior is more directly explanatory for
             * NativeVNNI than generic branch counters, so it owns the three
             * programmable slots.
             */
            static const std::array<EventSpec, 7> specs{{
                {"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
                {"ref-cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES},
                {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
                {"task-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, true},
                {"L1-dcache-loads", PERF_TYPE_HW_CACHE,
                 cacheConfig(PERF_COUNT_HW_CACHE_L1D,
                             PERF_COUNT_HW_CACHE_RESULT_ACCESS)},
                {"L1-dcache-load-misses", PERF_TYPE_HW_CACHE,
                 cacheConfig(PERF_COUNT_HW_CACHE_L1D,
                             PERF_COUNT_HW_CACHE_RESULT_MISS)},
                {"LLC-load-misses", PERF_TYPE_HW_CACHE,
                 cacheConfig(PERF_COUNT_HW_CACHE_LL,
                             PERF_COUNT_HW_CACHE_RESULT_MISS)},
            }};
            return specs;
        }

        /** Open one event attached only to one known persistent worker TID. */
        Event openEvent(
            const EventSpec &spec,
            pid_t thread_id,
            int group_fd)
        {
            perf_event_attr attribute{};
            attribute.size = sizeof(attribute);
            attribute.type = spec.type;
            attribute.config = spec.config;
            attribute.disabled = group_fd < 0 ? 1 : 0;
            attribute.inherit = 0;
            attribute.exclude_hv = 1;
            attribute.read_format =
                PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
            const int fd = static_cast<int>(::syscall(
                __NR_perf_event_open,
                &attribute,
                thread_id,
                -1,
                group_fd,
                0));
            if (fd < 0)
            {
                throw std::runtime_error(
                    "perf_event_open failed for " + std::string(spec.name) +
                    " in request " + request_id_ + ": " +
                    std::strerror(errno));
            }
            return Event{spec, fd};
        }

        /** Open the complete simultaneously schedulable group for one TID. */
        ThreadGroup openGroup(pid_t thread_id)
        {
            ThreadGroup group{.thread_id = thread_id};
            try
            {
                for (const EventSpec &spec : eventSpecs())
                {
                    const int leader = group.events.empty()
                        ? -1
                        : group.events.front().fd;
                    group.events.push_back(openEvent(spec, thread_id, leader));
                }
            }
            catch (...)
            {
                for (Event &event : group.events)
                    if (event.fd >= 0)
                        (void)::close(event.fd);
                throw;
            }
            return group;
        }

        /** Execute one group ioctl on every OMP worker and report after join. */
        template <typename ParallelFor>
        void runConcurrentControl(
            ParallelFor &parallel_for,
            unsigned long request,
            const char *operation)
        {
            std::atomic<bool> failed{false};
            parallel_for([&](size_t thread_number)
            {
                if (thread_number >= groups_.size() ||
                    ::ioctl(
                        groups_[thread_number].leaderFd(),
                        request,
                        PERF_IOC_FLAG_GROUP) != 0)
                {
                    failed.store(true, std::memory_order_relaxed);
                }
            });
            if (failed.load(std::memory_order_relaxed))
                throwSystemError(operation, "OpenMP thread groups");
        }

        /** Raise one operation-specific diagnostic while preserving `errno`. */
        [[noreturn]] void throwSystemError(
            const char *operation,
            const char *event_name) const
        {
            const int error = errno;
            throw std::runtime_error(
                "failed to " + std::string(operation) + " Linux perf event " +
                event_name + " for request " + request_id_ + ": " +
                std::strerror(error));
        }

        /** Read every event, scale multiplexed counts, and atomically write CSV. */
        void publish() const
        {
            const std::string staged_path = output_path_ + ".partial";
            std::FILE *output = std::fopen(staged_path.c_str(), "w");
            if (!output)
                throwSystemError("open output for", "all");
            bool failed = false;
            std::string failure_detail;
            std::array<long double, 7> scaled_totals{};
            std::array<uint64_t, 7> enabled_totals{};
            std::array<uint64_t, 7> running_totals{};
            for (const ThreadGroup &group : groups_)
            {
                for (size_t index = 0; index < group.events.size(); ++index)
                {
                    const Event &event = group.events[index];
                    EventRead reading{};
                    const ssize_t bytes = ::read(
                        event.fd, &reading, sizeof(reading));
                    if (bytes != static_cast<ssize_t>(sizeof(reading)) ||
                        reading.time_running == 0)
                    {
                        failed = true;
                        failure_detail =
                            std::string(event.spec.name) + " tid=" +
                            std::to_string(group.thread_id) + " bytes=" +
                            std::to_string(bytes) + " errno=" +
                            std::to_string(bytes < 0 ? errno : 0) +
                            " time_enabled=" +
                            std::to_string(reading.time_enabled) +
                            " time_running=" +
                            std::to_string(reading.time_running);
                        break;
                    }
                    scaled_totals[index] +=
                        static_cast<long double>(reading.value) *
                        static_cast<long double>(reading.time_enabled) /
                        static_cast<long double>(reading.time_running);
                    enabled_totals[index] += reading.time_enabled;
                    running_totals[index] += reading.time_running;
                }
                if (failed)
                    break;
            }
            for (size_t index = 0; !failed && index < eventSpecs().size(); ++index)
            {
                const EventSpec &spec = eventSpecs()[index];
                const double running_percent =
                    100.0 * static_cast<double>(running_totals[index]) /
                    static_cast<double>(enabled_totals[index]);
                std::fprintf(
                    output,
                    "%.0Lf;%s;%s;%llu;%.6f;;\n",
                    scaled_totals[index],
                    spec.nanoseconds ? "nsec" : "",
                    spec.name,
                    static_cast<unsigned long long>(running_totals[index]),
                    running_percent);
            }
            if (!failed)
            {
                std::fprintf(
                    output,
                    "%llu;nsec;wall-clock;%llu;100.000000;;\n",
                    static_cast<unsigned long long>(wall_clock_ns_),
                    static_cast<unsigned long long>(wall_clock_ns_));
            }
            if (std::fclose(output) != 0)
                failed = true;
            if (failed)
            {
                (void)::unlink(staged_path.c_str());
                throw std::runtime_error(
                    "failed to read complete Linux perf event inventory for " +
                    request_id_ + ": " + failure_detail);
            }
            if (::rename(staged_path.c_str(), output_path_.c_str()) != 0)
            {
                (void)::unlink(staged_path.c_str());
                throwSystemError("publish output for", "all");
            }
        }

        /** Close every successfully opened event descriptor exactly once. */
        void closeEvents() noexcept
        {
            for (ThreadGroup &group : groups_)
            {
                for (Event &event : group.events)
                {
                    if (event.fd >= 0)
                        (void)::close(event.fd);
                    event.fd = -1;
                }
            }
            groups_.clear();
        }

        std::string request_id_;
        std::string output_path_;
        std::vector<ThreadGroup> groups_;
        std::chrono::steady_clock::time_point profile_begin_{};
        uint64_t wall_clock_ns_ = 0;
        bool enabled_ = false;
    };

    /**
     * @brief Capture the exact persistent OpenMP team used by a CPU kernel.
     *
     * Performance events are attached to concrete Linux thread IDs, not to a
     * process-wide inheritance tree. The returned vector is indexed by OpenMP
     * thread number so element zero is the calling thread and every later
     * profiler control operation addresses the same stable worker.
     *
     * @return Linux TIDs ordered by OpenMP thread number.
     * @throws std::runtime_error If the runtime creates fewer workers than its
     *         advertised maximum or maps two slots to the same thread.
     */
    inline std::vector<pid_t> capturePersistentOpenMPThreadIds()
    {
        std::vector<pid_t> thread_ids(
            static_cast<size_t>(omp_get_max_threads()),
            static_cast<pid_t>(-1));
#pragma omp parallel shared(thread_ids)
        {
            const int thread_number = omp_get_thread_num();
            thread_ids[static_cast<size_t>(thread_number)] =
                static_cast<pid_t>(::syscall(SYS_gettid));
        }
        if (std::find(thread_ids.begin(), thread_ids.end(), -1) !=
            thread_ids.end())
        {
            throw std::runtime_error(
                "OpenMP profiler failed to capture every advertised worker TID");
        }
        std::vector<pid_t> unique = thread_ids;
        std::sort(unique.begin(), unique.end());
        if (std::adjacent_find(unique.begin(), unique.end()) != unique.end())
        {
            throw std::runtime_error(
                "OpenMP profiler captured a duplicate persistent worker TID");
        }
        return thread_ids;
    }

    /**
     * @brief Profile exactly one launch on a warmed persistent OpenMP team.
     *
     * Fixture construction, route publication, correctness checks, warmup, and
     * canonical timing must occur before this function. Counter reset/enable
     * and disable/read happen on the same workers immediately around `launch`,
     * yielding one uncontaminated candidate record.
     *
     * @tparam Launch Nullary callable that enters one production kernel route.
     * @param control Reusable process-owned event groups.
     * @param request_id Immutable corpus/profiler launch identity.
     * @param output_path Atomic output path for this exact request.
     * @param launch One production kernel invocation.
     */
    template <typename Launch>
    inline void profileExactOpenMPRegion(
        LinuxPerfControl &control,
        const std::string &request_id,
        const std::string &output_path,
        Launch &&launch)
    {
        control.rebind(request_id, output_path);
        if (!control.prepared())
            control.prepare(capturePersistentOpenMPThreadIds());
        const auto parallel_perf_control = [](auto &&action)
        {
#pragma omp parallel
            {
                action(static_cast<size_t>(omp_get_thread_num()));
            }
        };
        control.begin(parallel_perf_control);
        std::forward<Launch>(launch)();
        control.end(parallel_perf_control);
    }
} // namespace llaminar2::test::native_vnni_dispatch
