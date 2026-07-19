#include "loaders/gpu_pipeline/DeviceLoadPipeline.h"
#include "loaders/gpu_pipeline/WeightVRAMPool.h"
#include "loaders/gpu_pipeline/PinnedRingBuffer.h"
#include "loaders/MmapRegion.h"
#include "backends/IBackend.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"
#include "utils/WeightLoadingProfiler.h"
#include "utils/DebugEnv.h"

#include <chrono>
#include <cstring>
#include <iomanip>

namespace llaminar2
{
    namespace
    {
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

        for (size_t job_idx = 0; job_idx < jobs.size(); ++job_idx)
        {
            const auto &job = jobs[job_idx];
            const int stream_idx = static_cast<int>(job_idx % static_cast<size_t>(num_streams_));
            if (trace_weights)
            {
                LOG_INFO("[DeviceLoadPipeline] device=" << device_id_
                                                        << " job=" << (job_idx + 1) << "/" << jobs.size()
                                                        << " name=" << job.name
                                                        << " format=" << repackFormatName(job.format)
                                                        << " raw_bytes=" << job.raw_bytes
                                                        << " N=" << job.N
                                                        << " K=" << job.K
                                                        << " stream_slot=" << stream_idx);
            }

            if (job.raw_bytes == 0)
            {
                LOG_ERROR("DeviceLoadPipeline: job '" << job.name << "' has raw_bytes=0");
                return false;
            }

            if (!job.host_raw_data)
            {
                LOG_ERROR("DeviceLoadPipeline: job '" << job.name
                                                      << "' has null host_raw_data for " << job.raw_bytes
                                                      << " raw bytes (host weight data was likely released before GPU repack)");
                return false;
            }

            const int full_n = job.full_N > 0 ? job.full_N : job.N;
            const int full_k = job.full_K > 0 ? job.full_K : job.K;
            if (job.N <= 0 || full_n <= 0 || job.row_offset < 0 ||
                job.row_offset + job.N > full_n)
            {
                LOG_ERROR("DeviceLoadPipeline: invalid row chunk for '" << job.name
                                                                         << "' row_offset=" << job.row_offset
                                                                         << " N=" << job.N
                                                                         << " full_N=" << full_n);
                return false;
            }
            if (job.K <= 0 || full_k <= 0 || job.output_block_offset < 0)
            {
                LOG_ERROR("DeviceLoadPipeline: invalid K chunk for '" << job.name
                                                                        << "' K=" << job.K
                                                                        << " full_K=" << full_k
                                                                        << " output_block_offset=" << job.output_block_offset);
                return false;
            }

            // Validate raw bytes fit in staging slot
            if (job.raw_bytes > max_staging)
            {
                LOG_ERROR("DeviceLoadPipeline: job '" << job.name << "' raw_bytes="
                                                      << job.raw_bytes << " exceeds max staging slot="
                                                      << max_staging);
                return false;
            }

            // 1. Wait for previous repack on this staging slot to complete
            if (job_idx >= static_cast<size_t>(num_streams_))
            {
                const auto wait_start = Clock::now();
                if (!backend_.waitForEvent(repack_done_events_[stream_idx], device_id_))
                {
                    LOG_ERROR("DeviceLoadPipeline: waitForEvent repack_done["
                              << stream_idx << "] failed");
                    return false;
                }
                if (profiling)
                {
                    event_wait_ms += std::chrono::duration<double, std::milli>(
                                         Clock::now() - wait_start)
                                         .count();
                }
            }

            // 2. CPU memcpy: mmap → pinned slot
            void *pinned_ptr = pinned_.getSlot(stream_idx);
            if (!pinned_ptr)
            {
                LOG_ERROR("DeviceLoadPipeline: pinned slot " << stream_idx << " is null");
                return false;
            }
            {
                const auto memcpy_start = Clock::now();
                if (job.host_row_stride_bytes > 0)
                {
                    if (job.host_row_copy_bytes == 0 ||
                        job.raw_bytes != static_cast<size_t>(job.N) * job.host_row_copy_bytes)
                    {
                        LOG_ERROR("DeviceLoadPipeline: invalid gathered host chunk for '"
                                  << job.name << "'");
                        return false;
                    }
                    auto *dst = static_cast<uint8_t *>(pinned_ptr);
                    const auto *src = static_cast<const uint8_t *>(job.host_raw_data);
                    for (int row = 0; row < job.N; ++row)
                    {
                        std::memcpy(dst + static_cast<size_t>(row) * job.host_row_copy_bytes,
                                    src + static_cast<size_t>(row) * job.host_row_stride_bytes,
                                    job.host_row_copy_bytes);
                    }
                }
                else
                {
                    std::memcpy(pinned_ptr, job.host_raw_data, job.raw_bytes);
                }
                if (profiling)
                {
                    cpu_staging_ms += std::chrono::duration<double, std::milli>(
                                          Clock::now() - memcpy_start)
                                          .count();
                }
            }
            total_bytes += job.raw_bytes;

            // The source bytes are no longer needed once they have reached the
            // pinned slot. For demand-paged GGUF mappings, discard the process
            // PTEs incrementally so RSS follows the bounded ring instead of the
            // total model size. The underlying file remains available for a
            // harmless re-fault if another alias still needs the same pages.
            if (job.advise_mmap_dontneed_after_staging)
            {
                MmapRegion::adviseDontneedRange(job.host_raw_data, job.raw_bytes);
            }
            else if (job.mmap_discard_data && job.mmap_discard_bytes > 0)
            {
                MmapRegion::adviseDontneedRange(job.mmap_discard_data,
                                                job.mmap_discard_bytes);
            }

            // Fire progress callback after host memcpy completes
            if (progress_cb)
                progress_cb(total_bytes, total_planned_bytes);

            // 3. H2D async: pinned → device staging slot
            uint8_t *staging_ptr = pool_.getStagingSlot(stream_idx);
            if (!staging_ptr)
            {
                LOG_ERROR("DeviceLoadPipeline: staging slot " << stream_idx << " is null");
                return false;
            }

            if (!backend_.hostToDeviceOnStream(staging_ptr, pinned_ptr, job.raw_bytes,
                                               device_id_, h2d_streams_[stream_idx]))
            {
                LOG_ERROR("DeviceLoadPipeline: hostToDeviceOnStream for '"
                          << job.name << "' failed");
                return false;
            }

            // 4. Record H2D completion
            if (!backend_.recordEvent(h2d_done_events_[stream_idx], device_id_, h2d_streams_[stream_idx]))
            {
                LOG_ERROR("DeviceLoadPipeline: recordEvent h2d_done["
                          << stream_idx << "] failed");
                return false;
            }

            // 5. Repack stream waits for this H2D
            if (!backend_.streamWaitEvent(repack_stream_, h2d_done_events_[stream_idx], device_id_))
            {
                LOG_ERROR("DeviceLoadPipeline: streamWaitEvent failed");
                return false;
            }

            // 6. Launch GPU repack kernel on repack stream (or direct D2D copy for raw FP)
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
                // Floating-point passthrough: copy staging → payload (no repack needed).
                // Use the repack stream for ordering consistency with other slots.
                if (!backend_.deviceToDevice(
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
                if (job.N != full_n || job.row_offset != 0 || job.K % 32 != 0 ||
                    full_k % 32 != 0)
                {
                    LOG_ERROR("DeviceLoadPipeline: quantized chunks must span full N and 32-wide K blocks for '"
                              << job.name << "'");
                    return false;
                }
                const size_t total_blocks_per_row = static_cast<size_t>(full_k) / 32;
                const size_t chunk_blocks_per_row = static_cast<size_t>(job.K) / 32;
                const size_t output_block_offset = static_cast<size_t>(job.output_block_offset);
                if (output_block_offset + chunk_blocks_per_row > total_blocks_per_row)
                {
                    LOG_ERROR("DeviceLoadPipeline: quantized K chunk exceeds planned output for '"
                              << job.name << "'");
                    return false;
                }
                const size_t total_output_blocks =
                    total_blocks_per_row * static_cast<size_t>(full_n);
                if (total_output_blocks == 0 ||
                    slot->payload_bytes % total_output_blocks != 0)
                {
                    LOG_ERROR("DeviceLoadPipeline: invalid quantized payload layout for '"
                              << job.name << "'");
                    return false;
                }
                const size_t payload_bytes_per_block =
                    slot->payload_bytes / total_output_blocks;
                const size_t block_offset =
                    output_block_offset * static_cast<size_t>(full_n);
                auto *chunk_payload = slot->d_native_vnni_payload +
                                      block_offset * payload_bytes_per_block;
                auto *chunk_scales = slot->d_native_vnni_scales
                                         ? static_cast<uint16_t *>(slot->d_native_vnni_scales) + block_offset
                                         : nullptr;
                auto *chunk_mins = slot->d_native_vnni_mins
                                       ? static_cast<uint16_t *>(slot->d_native_vnni_mins) + block_offset
                                       : nullptr;
                auto *chunk_emins = slot->d_native_vnni_emins
                                        ? static_cast<uint32_t *>(slot->d_native_vnni_emins) + block_offset
                                        : nullptr;
                bool repack_ok = kernels_.vnniRepack(
                    job.format,
                    staging_ptr,
                    chunk_payload,
                    chunk_scales,
                    chunk_mins,
                    chunk_emins,
                    job.N, job.K, repack_stream_);

                if (!repack_ok)
                {
                    LOG_ERROR("DeviceLoadPipeline: repack kernel failed for '"
                              << job.name << "'");
                    return false;
                }
            }

            // 7. Record repack completion for this staging slot
            if (!backend_.recordEvent(repack_done_events_[stream_idx], device_id_, repack_stream_))
            {
                LOG_ERROR("DeviceLoadPipeline: recordEvent repack_done["
                          << stream_idx << "] failed");
                return false;
            }

            if (sync_after_repack_job)
            {
                if (!backend_.synchronizeStream(repack_stream_, device_id_))
                {
                    LOG_ERROR("DeviceLoadPipeline: synchronizeStream repack failed after job '"
                              << job.name << "' on device " << device_id_);
                    return false;
                }
            }

            ++num_processed_;
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
