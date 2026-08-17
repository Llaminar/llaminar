/**
 * @file DeviceLoadPipeline.cpp
 * @brief Implements bounded asynchronous GPU weight staging and repacking.
 *
 * Source coalescing is deliberately a planning operation: immutable parent
 * identities group adjacent expert views, while their tensor views retain the
 * mapped GGUF lifetime until all staging work has completed.
 */

#include "loaders/gpu_pipeline/DeviceLoadPipeline.h"
#include "loaders/gpu_pipeline/WeightVRAMPool.h"
#include "loaders/gpu_pipeline/PinnedRingBuffer.h"
#include "loaders/MmapRegion.h"
#include "backends/IBackend.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"
#include "utils/WeightLoadingProfiler.h"
#include "utils/DebugEnv.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unistd.h>

namespace llaminar2
{
    size_t orderWeightJobsForSequentialHostAccess(std::vector<WeightJob> &jobs)
    {
        size_t backward_jumps = 0;
        uintptr_t previous_address = 0;
        bool have_previous_address = false;

        for (const auto &job : jobs)
        {
            if (!job.host_raw_data)
                continue;

            const auto address = reinterpret_cast<uintptr_t>(job.host_raw_data);
            if (have_previous_address && address < previous_address)
                ++backward_jumps;
            previous_address = address;
            have_previous_address = true;
        }

        std::stable_sort(
            jobs.begin(), jobs.end(),
            [](const WeightJob &lhs, const WeightJob &rhs)
            {
                // Invalid jobs remain at the end so processJobs() emits its
                // existing precise null-source diagnostic before touching them.
                const auto lhs_address = lhs.host_raw_data
                                             ? reinterpret_cast<uintptr_t>(lhs.host_raw_data)
                                             : std::numeric_limits<uintptr_t>::max();
                const auto rhs_address = rhs.host_raw_data
                                             ? reinterpret_cast<uintptr_t>(rhs.host_raw_data)
                                             : std::numeric_limits<uintptr_t>::max();
                return lhs_address < rhs_address;
            });

        return backward_jumps;
    }

    std::vector<CoalescedWeightJobRun> coalesceContiguousWeightJobs(
        const std::vector<WeightJob> &jobs,
        const std::vector<const void *> &source_identities)
    {
        if (jobs.size() != source_identities.size())
        {
            throw std::invalid_argument(
                "coalesceContiguousWeightJobs requires one source identity per job");
        }
        if (jobs.empty())
            return {};

        std::vector<size_t> ordered_indices(jobs.size());
        std::iota(ordered_indices.begin(), ordered_indices.end(), size_t{0});

        for (size_t index = 0; index < jobs.size(); ++index)
        {
            const auto &job = jobs[index];
            const int full_n = job.full_N > 0 ? job.full_N : job.N;
            const int full_k = job.full_K > 0 ? job.full_K : job.K;
            if (!source_identities[index])
            {
                throw std::invalid_argument(
                    "coalesceContiguousWeightJobs requires a non-null immutable source identity for '" +
                    job.name + "'");
            }
            if (!job.host_raw_data || job.raw_bytes == 0 || job.N <= 0 || job.K <= 0 ||
                job.raw_bytes % static_cast<size_t>(job.N) != 0)
            {
                throw std::invalid_argument(
                    "coalesceContiguousWeightJobs received malformed whole-matrix job '" +
                    job.name + "'");
            }
            if (job.row_offset != 0 || full_n != job.N || full_k != job.K)
            {
                throw std::invalid_argument(
                    "coalesceContiguousWeightJobs must run before row chunking for '" +
                    job.name + "'");
            }
            if (job.packed_group_rows != 0)
            {
                throw std::invalid_argument(
                    "coalesceContiguousWeightJobs requires ungrouped logical input for '" +
                    job.name + "'");
            }
            if (job.packed_payload_capacity_bytes_per_block < 0)
            {
                throw std::invalid_argument(
                    "coalesceContiguousWeightJobs received a negative payload capacity for '" +
                    job.name + "'");
            }
        }

        std::stable_sort(
            ordered_indices.begin(), ordered_indices.end(),
            [&](size_t lhs_index, size_t rhs_index)
            {
                const auto lhs_identity =
                    reinterpret_cast<uintptr_t>(source_identities[lhs_index]);
                const auto rhs_identity =
                    reinterpret_cast<uintptr_t>(source_identities[rhs_index]);
                if (lhs_identity != rhs_identity)
                    return lhs_identity < rhs_identity;

                const auto lhs_source = reinterpret_cast<uintptr_t>(
                    jobs[lhs_index].host_raw_data);
                const auto rhs_source = reinterpret_cast<uintptr_t>(
                    jobs[rhs_index].host_raw_data);
                return lhs_source < rhs_source;
            });

        std::vector<CoalescedWeightJobRun> runs;
        runs.reserve(jobs.size());
        size_t previous_index = std::numeric_limits<size_t>::max();

        for (const size_t source_index : ordered_indices)
        {
            const auto &source = jobs[source_index];
            const size_t source_bytes_per_row =
                source.raw_bytes / static_cast<size_t>(source.N);

            bool append = false;
            if (!runs.empty() && previous_index != std::numeric_limits<size_t>::max())
            {
                const auto &previous = jobs[previous_index];
                const size_t previous_bytes_per_row =
                    previous.raw_bytes / static_cast<size_t>(previous.N);
                const auto previous_address = reinterpret_cast<uintptr_t>(
                    previous.host_raw_data);
                const auto source_address = reinterpret_cast<uintptr_t>(
                    source.host_raw_data);
                const bool previous_end_representable =
                    previous_address <=
                    std::numeric_limits<uintptr_t>::max() - previous.raw_bytes;

                append =
                    source_identities[source_index] == source_identities[previous_index] &&
                    source.format == previous.format &&
                    source.N == previous.N &&
                    source.K == previous.K &&
                    source.is_asymmetric == previous.is_asymmetric &&
                    source.packed_payload_capacity_bytes_per_block ==
                        previous.packed_payload_capacity_bytes_per_block &&
                    source_bytes_per_row == previous_bytes_per_row &&
                    previous_end_representable &&
                    source_address == previous_address + previous.raw_bytes;
            }

            if (!append)
            {
                CoalescedWeightJobRun run;
                run.job = source;
                run.job.row_offset = 0;
                run.job.full_N = source.N;
                run.job.full_K = source.K;
                run.members.push_back({
                    .source_job_index = source_index,
                    .row_offset = 0,
                });
                runs.push_back(std::move(run));
                previous_index = source_index;
                continue;
            }

            auto &run = runs.back();
            if (source.N > std::numeric_limits<int>::max() - run.job.N)
            {
                throw std::overflow_error(
                    "coalesceContiguousWeightJobs row count overflow for '" +
                    source.name + "'");
            }
            if (source.raw_bytes >
                std::numeric_limits<size_t>::max() - run.job.raw_bytes)
            {
                throw std::overflow_error(
                    "coalesceContiguousWeightJobs byte count overflow for '" +
                    source.name + "'");
            }

            const int member_row_offset = run.job.N;
            run.members.push_back({
                .source_job_index = source_index,
                .row_offset = member_row_offset,
            });
            run.job.N += source.N;
            run.job.full_N = run.job.N;
            run.job.raw_bytes += source.raw_bytes;
            run.job.packed_group_rows = source.N;
            previous_index = source_index;
        }

        return runs;
    }

    namespace
    {
        struct FileReadPlan
        {
            int fd = -1;
            uint64_t file_offset = 0;
            size_t read_bytes = 0;
        };

        /**
         * @brief Resolve mmap jobs into reusable buffered file descriptors.
         *
         * Exact buffered pread() calls retain the loader's fixed pinned-ring
         * memory bound while allowing Linux readahead and the page cache to
         * service the sequential GGUF stream. This is materially faster than
         * uncached direct I/O on consumer NVMe devices, where the model's many
         * sub-megabyte tensor ranges otherwise become synchronous storage
         * operations.
         *
         * One descriptor per split GGUF file is shared by independent pread()
         * calls. pread() does not mutate descriptor position, so producer lanes
         * can fill persistent pinned slots concurrently without a file-offset
         * lock or an intermediate host allocation.
         */
        class MmapFileReader
        {
        public:
            MmapFileReader() = default;

            ~MmapFileReader()
            {
                for (const int fd : owned_fds_)
                {
                    if (fd >= 0)
                        ::close(fd);
                }
            }

            MmapFileReader(const MmapFileReader &) = delete;
            MmapFileReader &operator=(const MmapFileReader &) = delete;

            bool prepare(const std::vector<WeightJob> &jobs,
                         std::vector<FileReadPlan> &plans,
                         std::string &error)
            {
                plans.assign(jobs.size(), {});
                for (size_t job_index = 0; job_index < jobs.size(); ++job_index)
                {
                    const auto &job = jobs[job_index];
                    // MmapRegion owns the process-wide registry of live model
                    // mappings. Resolve the concrete pointer range instead of
                    // trusting a tensor-class hint: expert views and other
                    // borrowed slices can point into a GGUF mapping without
                    // carrying the root tensor's mmap metadata.
                    const auto source =
                        MmapRegion::resolveFileSource(job.host_raw_data, job.raw_bytes);
                    if (!source)
                        continue;

                    int fd = -1;
                    const auto existing = fds_by_path_.find(source->path);
                    if (existing != fds_by_path_.end())
                    {
                        fd = existing->second;
                    }
                    else
                    {
#ifdef __linux__
                        fd = ::open(
                            source->path.c_str(),
                            O_RDONLY | O_CLOEXEC);
#endif
                        if (fd < 0)
                        {
                            error = "open for buffered pread failed for '" +
                                    source->path +
                                    "' errno=" + std::to_string(errno);
                            return false;
                        }
#ifdef POSIX_FADV_SEQUENTIAL
                        /*
                         * This is a scheduling hint, not an eager read request.
                         * It neither grows the process staging ring nor requires
                         * the complete model to be resident before loading.
                         */
                        (void)::posix_fadvise(
                            fd,
                            0,
                            0,
                            POSIX_FADV_SEQUENTIAL);
#endif
                        fds_by_path_.emplace(source->path, fd);
                        owned_fds_.push_back(fd);
                    }

                    plans[job_index] = {
                        .fd = fd,
                        .file_offset = source->offset,
                        .read_bytes = job.raw_bytes,
                    };
                }
                return true;
            }

            bool read(const FileReadPlan &plan,
                      void *slot,
                      size_t slot_bytes,
                      const void *&payload,
                      size_t &file_bytes,
                      std::string &error) const
            {
                if (plan.fd < 0)
                {
                    error = "file-read plan has no file descriptor";
                    return false;
                }
                if (!slot || plan.read_bytes > slot_bytes)
                {
                    error = "buffered file read exceeds pinned staging slot";
                    return false;
                }

                size_t completed = 0;
                while (completed < plan.read_bytes)
                {
                    ssize_t result = -1;
                    do
                    {
                        result = ::pread(
                            plan.fd,
                            static_cast<uint8_t *>(slot) + completed,
                            plan.read_bytes - completed,
                            static_cast<off_t>(
                                plan.file_offset + completed));
                    } while (result < 0 && errno == EINTR);

                    if (result < 0)
                    {
                        error = "buffered pread failed with errno=" +
                                std::to_string(errno);
                        return false;
                    }
                    if (result == 0)
                    {
                        error = "short buffered pread: required=" +
                                std::to_string(plan.read_bytes) +
                                " received=" + std::to_string(completed);
                        return false;
                    }
                    completed += static_cast<size_t>(result);
                }

                file_bytes = completed;
                payload = slot;
                return true;
            }

        private:
            std::unordered_map<std::string, int> fds_by_path_;
            std::vector<int> owned_fds_;
        };

        struct StagedSource
        {
            const void *payload = nullptr;
            size_t file_bytes = 0;
            double read_ms = 0.0;
            double h2d_wait_ms = 0.0;
            std::string error;
        };

        /**
         * @brief Persistent producer threads for pinned upload slots.
         *
         * A pinned slot becomes reusable when its H2D event completes, which is
         * earlier than completion of the repack kernel consuming the paired
         * device slot. Each worker waits only for that H2D event, refills its host
         * slot, and lets the main thread protect device-slot reuse with a
         * stream-side wait on the repack event. This overlaps storage, H2D, and
         * repack without a full-device synchronization or another allocation.
         */
        class StagingLanePool
        {
        public:
            StagingLanePool(
                IBackend &backend,
                int device_id,
                PinnedRingBuffer &pinned,
                const MmapFileReader &file_reader,
                const std::vector<FileReadPlan> &file_read_plans,
                const std::vector<WeightJob> &jobs,
                const std::vector<void *> &h2d_done_events,
                int lane_count)
                : backend_(backend),
                  device_id_(device_id),
                  pinned_(pinned),
                  file_reader_(file_reader),
                  file_read_plans_(file_read_plans),
                  jobs_(jobs),
                  h2d_done_events_(h2d_done_events)
            {
                lanes_.reserve(static_cast<size_t>(lane_count));
                for (int lane = 0; lane < lane_count; ++lane)
                {
                    lanes_.push_back(std::make_unique<Lane>());
                    lanes_.back()->worker = std::thread(
                        [this, lane]
                        { workerLoop(lane); });
                }
            }

            ~StagingLanePool()
            {
                for (auto &lane : lanes_)
                {
                    {
                        std::lock_guard<std::mutex> lock(lane->mutex);
                        lane->stop = true;
                    }
                    lane->condition.notify_all();
                }
                for (auto &lane : lanes_)
                {
                    if (lane->worker.joinable())
                        lane->worker.join();
                }
            }

            StagingLanePool(const StagingLanePool &) = delete;
            StagingLanePool &operator=(const StagingLanePool &) = delete;

            bool submit(int lane_index, size_t job_index, bool wait_for_prior_h2d)
            {
                auto &lane = *lanes_.at(static_cast<size_t>(lane_index));
                {
                    std::lock_guard<std::mutex> lock(lane.mutex);
                    if (lane.pending || lane.busy || lane.ready || lane.stop)
                        return false;
                    lane.job_index = job_index;
                    lane.wait_for_prior_h2d = wait_for_prior_h2d;
                    lane.pending = true;
                }
                lane.condition.notify_all();
                return true;
            }

            bool wait(int lane_index, size_t expected_job_index, StagedSource &result)
            {
                auto &lane = *lanes_.at(static_cast<size_t>(lane_index));
                std::unique_lock<std::mutex> lock(lane.mutex);
                lane.condition.wait(
                    lock,
                    [&lane]
                    { return lane.ready || lane.stop; });
                if (!lane.ready || lane.job_index != expected_job_index)
                    return false;
                result = std::move(lane.result);
                lane.result = {};
                lane.ready = false;
                lane.condition.notify_all();
                return true;
            }

        private:
            struct Lane
            {
                std::mutex mutex;
                std::condition_variable condition;
                std::thread worker;
                size_t job_index = 0;
                bool wait_for_prior_h2d = false;
                bool pending = false;
                bool busy = false;
                bool ready = false;
                bool stop = false;
                StagedSource result;
            };

            void workerLoop(int lane_index)
            {
                auto &lane = *lanes_[static_cast<size_t>(lane_index)];
                while (true)
                {
                    size_t job_index = 0;
                    bool wait_for_prior_h2d = false;
                    {
                        std::unique_lock<std::mutex> lock(lane.mutex);
                        lane.condition.wait(
                            lock,
                            [&lane]
                            { return lane.pending || lane.stop; });
                        if (lane.stop && !lane.pending)
                            return;
                        job_index = lane.job_index;
                        wait_for_prior_h2d = lane.wait_for_prior_h2d;
                        lane.pending = false;
                        lane.busy = true;
                    }

                    StagedSource result;
                    if (wait_for_prior_h2d)
                    {
                        const auto wait_start = std::chrono::high_resolution_clock::now();
                        if (!backend_.waitForEvent(
                                h2d_done_events_[static_cast<size_t>(lane_index)],
                                device_id_))
                        {
                            result.error = "failed waiting for prior H2D completion";
                        }
                        result.h2d_wait_ms =
                            std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() - wait_start)
                                .count();
                    }

                    if (result.error.empty())
                    {
                        const auto read_start = std::chrono::high_resolution_clock::now();
                        void *destination = pinned_.getSlot(lane_index);
                        if (!destination)
                        {
                            result.error = "pinned slot is null";
                        }
                        else
                        {
                            const auto &job = jobs_[job_index];
                            const auto &plan =
                                file_read_plans_[job_index];
                            if (plan.fd >= 0)
                            {
                                file_reader_.read(
                                    plan,
                                    destination,
                                    pinned_.slotSize(),
                                    result.payload,
                                    result.file_bytes,
                                    result.error);
                            }
                            else
                            {
                                std::memcpy(destination, job.host_raw_data, job.raw_bytes);
                                result.payload = destination;
                            }
                        }
                        result.read_ms =
                            std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() - read_start)
                                .count();
                    }

                    {
                        std::lock_guard<std::mutex> lock(lane.mutex);
                        lane.result = std::move(result);
                        lane.busy = false;
                        lane.ready = true;
                    }
                    lane.condition.notify_all();
                }
            }

            IBackend &backend_;
            int device_id_;
            PinnedRingBuffer &pinned_;
            const MmapFileReader &file_reader_;
            const std::vector<FileReadPlan> &file_read_plans_;
            const std::vector<WeightJob> &jobs_;
            const std::vector<void *> &h2d_done_events_;
            std::vector<std::unique_ptr<Lane>> lanes_;
        };

        const char *repackFormatName(RepackFormat format)
        {
            switch (format)
            {
            case RepackFormat::Q4_0:
                return "Q4_0";
            case RepackFormat::Q4_1:
                return "Q4_1";
            case RepackFormat::Q5_0:
                return "Q5_0";
            case RepackFormat::Q5_1:
                return "Q5_1";
            case RepackFormat::Q8_0:
                return "Q8_0";
            case RepackFormat::Q8_1:
                return "Q8_1";
            case RepackFormat::Q8_K:
                return "Q8_K";
            case RepackFormat::Q4_K:
                return "Q4_K";
            case RepackFormat::Q5_K:
                return "Q5_K";
            case RepackFormat::Q6_K:
                return "Q6_K";
            case RepackFormat::Q3_K:
                return "Q3_K";
            case RepackFormat::Q2_K:
                return "Q2_K";
            case RepackFormat::IQ4_NL:
                return "IQ4_NL";
            case RepackFormat::IQ4_XS:
                return "IQ4_XS";
            case RepackFormat::IQ3_S:
                return "IQ3_S";
            case RepackFormat::IQ3_XXS:
                return "IQ3_XXS";
            case RepackFormat::IQ2_S:
                return "IQ2_S";
            case RepackFormat::IQ2_XS:
                return "IQ2_XS";
            case RepackFormat::IQ2_XXS:
                return "IQ2_XXS";
            case RepackFormat::IQ1_S:
                return "IQ1_S";
            case RepackFormat::IQ1_M:
                return "IQ1_M";
            case RepackFormat::RAW_FP:
                return "RAW_FP";
            default:
                return "UNKNOWN";
            }
        }
    }

    DeviceLoadPipeline::DeviceLoadPipeline(IBackend &backend,
                                           int device_id,
                                           WeightVRAMPool &pool,
                                           PinnedRingBuffer &pinned,
                                           const RepackKernels &kernels,
                                           int num_h2d_streams)
        : backend_(backend),
          device_id_(device_id),
          pool_(pool),
          pinned_(pinned),
          kernels_(kernels),
          num_streams_(num_h2d_streams)
    {
    }

    DeviceLoadPipeline::~DeviceLoadPipeline() { release(); }

    bool DeviceLoadPipeline::initialize()
    {
        if (initialized_)
            return true;

        if (!backend_.setDevice(device_id_))
        {
            LOG_ERROR("DeviceLoadPipeline: setDevice(" << device_id_ << ") failed");
            return false;
        }

        // Create H2D streams
        h2d_streams_.resize(num_streams_, nullptr);
        for (int i = 0; i < num_streams_; ++i)
        {
            void *stream = backend_.createStream(device_id_);
            if (!stream)
            {
                LOG_ERROR("DeviceLoadPipeline: createStream H2D[" << i << "] failed");
                release();
                return false;
            }
            h2d_streams_[i] = stream;
        }

        // Create repack stream
        repack_stream_ = backend_.createStream(device_id_);
        if (!repack_stream_)
        {
            LOG_ERROR("DeviceLoadPipeline: createStream repack failed");
            release();
            return false;
        }

        // Create H2D done events
        h2d_done_events_.resize(num_streams_, nullptr);
        for (int i = 0; i < num_streams_; ++i)
        {
            void *event = backend_.createEvent(device_id_);
            if (!event)
            {
                LOG_ERROR("DeviceLoadPipeline: createEvent h2d_done[" << i << "] failed");
                release();
                return false;
            }
            h2d_done_events_[i] = event;
        }

        // Create repack done events
        repack_done_events_.resize(num_streams_, nullptr);
        for (int i = 0; i < num_streams_; ++i)
        {
            void *event = backend_.createEvent(device_id_);
            if (!event)
            {
                LOG_ERROR("DeviceLoadPipeline: createEvent repack_done[" << i << "] failed");
                release();
                return false;
            }
            repack_done_events_[i] = event;
        }

        initialized_ = true;
        LOG_DEBUG("DeviceLoadPipeline: initialized with " << num_streams_
                                                          << " H2D streams on device " << device_id_);
        return true;
    }

    bool DeviceLoadPipeline::processJobs(const std::vector<WeightJob> &jobs,
                                         ProgressCallback progress_cb)
    {
        num_processed_ = 0;

        if (jobs.empty())
            return true;

        if (!initialized_)
        {
            LOG_ERROR("DeviceLoadPipeline: not initialized");
            return false;
        }

        if (!backend_.setDevice(device_id_))
        {
            LOG_ERROR("DeviceLoadPipeline: setDevice(" << device_id_ << ") failed");
            return false;
        }

        const size_t max_staging = pool_.maxStagingSlotBytes();
        const bool profiling = WeightLoadingProfiler::isEnabled();
        const auto &env = debugEnv();
        const bool trace_weights = env.weight_lifecycle_trace;
        const bool sync_after_repack_job = env.rocm.sync_after_kernel;
        size_t buffered_file_read_bytes = 0;

        MmapFileReader file_reader;
        std::vector<FileReadPlan> file_read_plans;
        std::string file_read_error;
        if (!file_reader.prepare(jobs, file_read_plans, file_read_error))
        {
            LOG_ERROR(
                "DeviceLoadPipeline: failed to prepare bounded buffered file reads: "
                << file_read_error);
            return false;
        }

        // Precompute total planned bytes for progress reporting
        size_t total_planned_bytes = 0;
        if (progress_cb)
        {
            for (const auto &j : jobs)
                total_planned_bytes += j.raw_bytes;
        }

        using Clock = std::chrono::high_resolution_clock;
        const auto pipeline_start = Clock::now();
        double cpu_staging_ms = 0.0;
        double event_wait_ms = 0.0;
        size_t total_bytes = 0;

        // Catch malformed jobs before starting producer threads so any failure
        // exits without partially populated lane state.
        for (const auto &job : jobs)
        {
            const int full_n = job.full_N > 0 ? job.full_N : job.N;
            const int full_k = job.full_K > 0 ? job.full_K : job.K;
            if (job.raw_bytes == 0 || !job.host_raw_data)
            {
                LOG_ERROR("DeviceLoadPipeline: invalid host source for '" << job.name
                                                                          << "' raw_bytes=" << job.raw_bytes);
                return false;
            }
            if (job.N <= 0 || full_n <= 0 || job.row_offset < 0 ||
                job.row_offset + job.N > full_n)
            {
                LOG_ERROR("DeviceLoadPipeline: invalid row chunk for '" << job.name
                                                                         << "' row_offset=" << job.row_offset
                                                                         << " N=" << job.N
                                                                         << " full_N=" << full_n);
                return false;
            }
            if (job.packed_group_rows < 0 ||
                (job.packed_group_rows > 0 &&
                 (job.packed_group_rows > full_n ||
                  full_n % job.packed_group_rows != 0)))
            {
                LOG_ERROR("DeviceLoadPipeline: invalid packed group geometry for '"
                          << job.name << "' packed_group_rows="
                          << job.packed_group_rows << " full_N=" << full_n);
                return false;
            }
            if (job.packed_payload_capacity_bytes_per_block < 0)
            {
                LOG_ERROR("DeviceLoadPipeline: invalid payload capacity for '"
                          << job.name << "'");
                return false;
            }
            if (job.K <= 0 || full_k <= 0 || job.raw_bytes > max_staging)
            {
                LOG_ERROR("DeviceLoadPipeline: invalid bounded row-chunk geometry for '"
                          << job.name << "'");
                return false;
            }
        }

        StagingLanePool staging_lanes(
            backend_,
            device_id_,
            pinned_,
            file_reader,
            file_read_plans,
            jobs,
            h2d_done_events_,
            num_streams_);
        std::vector<bool> lane_has_prior_repack(
            static_cast<size_t>(num_streams_), false);
        std::vector<double> lane_read_ms(
            static_cast<size_t>(num_streams_), 0.0);

        const size_t initial_jobs = std::min(
            jobs.size(), static_cast<size_t>(num_streams_));
        for (size_t job_index = 0; job_index < initial_jobs; ++job_index)
        {
            if (!staging_lanes.submit(
                    static_cast<int>(job_index), job_index, false))
            {
                LOG_ERROR("DeviceLoadPipeline: failed to seed staging lane "
                          << job_index);
                return false;
            }
        }

        for (size_t job_index = 0; job_index < jobs.size(); ++job_index)
        {
            const auto &job = jobs[job_index];
            const int stream_index =
                static_cast<int>(job_index % static_cast<size_t>(num_streams_));
            const int full_n = job.full_N > 0 ? job.full_N : job.N;
            const int full_k = job.full_K > 0 ? job.full_K : job.K;

            StagedSource staged;
            if (!staging_lanes.wait(stream_index, job_index, staged))
            {
                LOG_ERROR("DeviceLoadPipeline: staging lane " << stream_index
                                                               << " returned an unexpected job");
                return false;
            }
            if (!staged.error.empty() || !staged.payload)
            {
                LOG_ERROR("DeviceLoadPipeline: staging failed for '" << job.name
                                                                      << "': " << staged.error);
                return false;
            }

            lane_read_ms[static_cast<size_t>(stream_index)] += staged.read_ms;
            event_wait_ms += staged.h2d_wait_ms;
            buffered_file_read_bytes += staged.file_bytes;

            if (trace_weights)
            {
                LOG_INFO("[DeviceLoadPipeline] device=" << device_id_
                                                        << " job=" << (job_index + 1) << "/" << jobs.size()
                                                        << " name=" << job.name
                                                        << " format=" << repackFormatName(job.format)
                                                        << " raw_bytes=" << job.raw_bytes
                                                        << " N=" << job.N
                                                        << " K=" << job.K
                                                        << " source=" << job.host_raw_data
                                                        << " source_mode="
                                                        << (file_read_plans[job_index].fd >= 0
                                                                ? "buffered_file"
                                                                : "memory")
                                                        << " stream_slot=" << stream_index);
            }

            total_bytes += job.raw_bytes;

            if (progress_cb)
                progress_cb(total_bytes, total_planned_bytes);

            uint8_t *staging_ptr = pool_.getStagingSlot(stream_index);
            if (!staging_ptr)
            {
                LOG_ERROR("DeviceLoadPipeline: staging slot " << stream_index << " is null");
                return false;
            }

            // Pinned reuse is guarded by the worker's H2D completion wait.
            // Device staging lives longer, through repack, so its dependency is
            // inserted directly into this H2D stream without blocking the host.
            if (lane_has_prior_repack[static_cast<size_t>(stream_index)] &&
                !backend_.streamWaitEvent(
                    h2d_streams_[stream_index],
                    repack_done_events_[stream_index],
                    device_id_))
            {
                LOG_ERROR("DeviceLoadPipeline: H2D stream failed to wait for prior repack on lane "
                          << stream_index);
                return false;
            }

            if (!backend_.hostToDeviceOnStream(
                    staging_ptr,
                    staged.payload,
                    job.raw_bytes,
                    device_id_,
                    h2d_streams_[stream_index]))
            {
                LOG_ERROR("DeviceLoadPipeline: hostToDeviceOnStream for '"
                          << job.name << "' failed");
                return false;
            }

            if (!backend_.recordEvent(
                    h2d_done_events_[stream_index],
                    device_id_,
                    h2d_streams_[stream_index]))
            {
                LOG_ERROR("DeviceLoadPipeline: recordEvent h2d_done["
                          << stream_index << "] failed");
                return false;
            }

            if (!backend_.streamWaitEvent(
                    repack_stream_,
                    h2d_done_events_[stream_index],
                    device_id_))
            {
                LOG_ERROR("DeviceLoadPipeline: streamWaitEvent failed");
                return false;
            }

            auto slot = pool_.getSlot(job.name);
            if (!slot)
            {
                LOG_ERROR("DeviceLoadPipeline: no pool slot for weight '" << job.name << "'");
                return false;
            }

            if (job.format == RepackFormat::RAW_FP)
            {
                if (slot->payload_bytes % static_cast<size_t>(full_n) != 0)
                {
                    LOG_ERROR("DeviceLoadPipeline: FP payload bytes for '" << job.name
                                                                            << "' are not divisible by full_N=" << full_n);
                    return false;
                }
                const size_t payload_bytes_per_row =
                    slot->payload_bytes / static_cast<size_t>(full_n);
                auto *chunk_payload = slot->d_native_vnni_payload +
                                      static_cast<size_t>(job.row_offset) * payload_bytes_per_row;
                if (!backend_.deviceCopyAsync(
                        chunk_payload, staging_ptr, job.raw_bytes,
                        device_id_, repack_stream_))
                {
                    LOG_ERROR("DeviceLoadPipeline: D2D copy failed for FP weight '"
                              << job.name << "'");
                    return false;
                }
            }
            else
            {
                if (job.K != full_k || job.K % 32 != 0 ||
                    full_k % 32 != 0)
                {
                    LOG_ERROR("DeviceLoadPipeline: quantized row chunks must span full K in 32-wide blocks for '"
                              << job.name << "'");
                    return false;
                }
                const size_t total_blocks_per_row = static_cast<size_t>(full_k) / 32;
                const size_t total_output_blocks =
                    total_blocks_per_row * static_cast<size_t>(full_n);
                if (total_output_blocks == 0 ||
                    slot->payload_bytes % total_output_blocks != 0)
                {
                    LOG_ERROR("DeviceLoadPipeline: invalid quantized payload layout for '"
                              << job.name << "'");
                    return false;
                }
                const size_t allocated_payload_bytes_per_block =
                    slot->payload_bytes / total_output_blocks;
                const int compact_payload_bytes =
                    repackPayloadBytesPerBlock(job.format);
                if (compact_payload_bytes <= 0 ||
                    allocated_payload_bytes_per_block <
                        static_cast<size_t>(compact_payload_bytes) ||
                    allocated_payload_bytes_per_block >
                        static_cast<size_t>(std::numeric_limits<int>::max()) ||
                    (job.packed_payload_capacity_bytes_per_block > 0 &&
                     allocated_payload_bytes_per_block !=
                         static_cast<size_t>(
                             job.packed_payload_capacity_bytes_per_block)))
                {
                    LOG_ERROR("DeviceLoadPipeline: persistent payload capacity disagrees with repack geometry for '"
                              << job.name << "'");
                    return false;
                }
                const bool repack_ok = kernels_.vnniRepack(
                    job.format,
                    staging_ptr,
                    slot->d_native_vnni_payload,
                    static_cast<uint16_t *>(slot->d_native_vnni_scales),
                    static_cast<uint16_t *>(slot->d_native_vnni_mins),
                    static_cast<uint32_t *>(slot->d_native_vnni_emins),
                    job.N, job.K,
                    full_n, job.row_offset,
                    job.packed_group_rows,
                    static_cast<int>(allocated_payload_bytes_per_block),
                    repack_stream_);

                if (!repack_ok)
                {
                    LOG_ERROR("DeviceLoadPipeline: repack kernel failed for '"
                              << job.name << "'");
                    return false;
                }
            }

            if (!backend_.recordEvent(
                    repack_done_events_[stream_index],
                    device_id_,
                    repack_stream_))
            {
                LOG_ERROR("DeviceLoadPipeline: recordEvent repack_done["
                          << stream_index << "] failed");
                return false;
            }
            lane_has_prior_repack[static_cast<size_t>(stream_index)] = true;

            if (sync_after_repack_job)
            {
                if (!backend_.synchronizeStream(repack_stream_, device_id_))
                {
                    LOG_ERROR("DeviceLoadPipeline: synchronizeStream repack failed after job '"
                              << job.name << "' on device " << device_id_);
                    return false;
                }
            }

            const size_t next_job =
                job_index + static_cast<size_t>(num_streams_);
            if (next_job < jobs.size() &&
                !staging_lanes.submit(stream_index, next_job, true))
            {
                LOG_ERROR("DeviceLoadPipeline: failed to refill staging lane "
                          << stream_index << " with job " << next_job);
                return false;
            }

            ++num_processed_;
        }

        if (profiling)
        {
            cpu_staging_ms = *std::max_element(
                lane_read_ms.begin(), lane_read_ms.end());
        }

        // Wait for all repack work to complete
        const auto drain_start = Clock::now();
        if (!backend_.synchronizeStream(repack_stream_, device_id_))
        {
            LOG_ERROR("DeviceLoadPipeline: synchronizeStream repack failed");
            return false;
        }

        // Synchronize all H2D streams
        for (int i = 0; i < num_streams_; ++i)
        {
            if (!backend_.synchronizeStream(h2d_streams_[i], device_id_))
            {
                LOG_ERROR("DeviceLoadPipeline: synchronizeStream h2d[" << i << "] failed");
                return false;
            }
        }
        const double drain_ms = std::chrono::duration<double, std::milli>(
                                    Clock::now() - drain_start)
                                    .count();
        const double pipeline_ms = std::chrono::duration<double, std::milli>(
                                       Clock::now() - pipeline_start)
                                       .count();

        // Report profiling metrics
        if (profiling)
        {
            const std::string dev = "gpu_pipeline.device_" + std::to_string(device_id_);
            WeightLoadingProfiler::addDetail(dev + ".total", pipeline_ms);
            WeightLoadingProfiler::addDetail(dev + ".cpu_staging", cpu_staging_ms);
            WeightLoadingProfiler::addDetail(dev + ".event_wait", event_wait_ms);
            WeightLoadingProfiler::addDetail(dev + ".gpu_drain", drain_ms);
        }

        // Report throughput
        const double total_mb = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
        const double throughput_gbs = (pipeline_ms > 0.0)
                                          ? (total_mb / 1024.0) / (pipeline_ms / 1000.0)
                                          : 0.0;
        const double cpu_staging_gbs = (cpu_staging_ms > 0.0)
                                           ? (total_mb / 1024.0) / (cpu_staging_ms / 1000.0)
                                           : 0.0;
        if (PerfStatsCollector::isEnabled())
        {
            const std::string device = "gpu:" + std::to_string(device_id_);
            PerfStatsCollector::addCounter("weight_loading", "gpu_pipeline_bytes",
                                           static_cast<double>(total_bytes), "load", device);
            PerfStatsCollector::addCounter("weight_loading", "gpu_pipeline_throughput_gbs",
                                           throughput_gbs, "load", device);
            PerfStatsCollector::addCounter("weight_loading", "gpu_pipeline_cpu_staging_ms",
                                           cpu_staging_ms, "load", device);
            PerfStatsCollector::addCounter("weight_loading", "gpu_pipeline_cpu_staging_gbs",
                                           cpu_staging_gbs, "load", device);
            PerfStatsCollector::addCounter("weight_loading", "gpu_pipeline_event_wait_ms",
                                           event_wait_ms, "load", device);
            PerfStatsCollector::addCounter("weight_loading", "gpu_pipeline_drain_ms",
                                           drain_ms, "load", device);
            PerfStatsCollector::addCounter("weight_loading", "gpu_pipeline_job_count",
                                           static_cast<double>(jobs.size()), "load", device);
            PerfStatsCollector::addCounter(
                "weight_loading",
                "gpu_pipeline_buffered_file_read_bytes",
                static_cast<double>(buffered_file_read_bytes),
                "load",
                device);
        }
        LOG_DEBUG("DeviceLoadPipeline: device " << device_id_ << " loaded "
                                                << num_processed_ << " weights, " << std::fixed << std::setprecision(1)
                                                << total_mb << " MB in " << pipeline_ms << " ms ("
                                                << throughput_gbs << " GB/s)");
        return true;
    }

    void DeviceLoadPipeline::release()
    {
        if (!initialized_)
            return;

        backend_.setDevice(device_id_);

        // Destroy events first
        for (auto &ev : repack_done_events_)
        {
            if (ev)
            {
                backend_.destroyEvent(ev, device_id_);
                ev = nullptr;
            }
        }
        repack_done_events_.clear();

        for (auto &ev : h2d_done_events_)
        {
            if (ev)
            {
                backend_.destroyEvent(ev, device_id_);
                ev = nullptr;
            }
        }
        h2d_done_events_.clear();

        // Destroy streams
        if (repack_stream_)
        {
            backend_.destroyStream(repack_stream_, device_id_);
            repack_stream_ = nullptr;
        }

        for (auto &s : h2d_streams_)
        {
            if (s)
            {
                backend_.destroyStream(s, device_id_);
                s = nullptr;
            }
        }
        h2d_streams_.clear();

        initialized_ = false;
        num_processed_ = 0;
    }

} // namespace llaminar2
