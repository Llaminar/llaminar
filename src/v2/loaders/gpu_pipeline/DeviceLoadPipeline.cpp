#include "loaders/gpu_pipeline/DeviceLoadPipeline.h"
#include "loaders/gpu_pipeline/WeightVRAMPool.h"
#include "loaders/gpu_pipeline/PinnedRingBuffer.h"
#include "backends/IBackend.h"
#include "utils/Logger.h"
#include "utils/WeightLoadingProfiler.h"

#include <chrono>
#include <cstring>
#include <iomanip>

namespace llaminar2
{

DeviceLoadPipeline::DeviceLoadPipeline(IBackend& backend,
                                       int device_id,
                                       WeightVRAMPool& pool,
                                       PinnedRingBuffer& pinned,
                                       const RepackKernels& kernels,
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
    if (initialized_) return true;

    if (!backend_.setDevice(device_id_))
    {
        LOG_ERROR("DeviceLoadPipeline: setDevice(" << device_id_ << ") failed");
        return false;
    }

    // Create H2D streams
    h2d_streams_.resize(num_streams_, nullptr);
    for (int i = 0; i < num_streams_; ++i)
    {
        void* stream = backend_.createStream(device_id_);
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
        void* event = backend_.createEvent(device_id_);
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
        void* event = backend_.createEvent(device_id_);
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

bool DeviceLoadPipeline::processJobs(const std::vector<WeightJob>& jobs)
{
    num_processed_ = 0;

    if (jobs.empty()) return true;

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

    using Clock = std::chrono::high_resolution_clock;
    const auto pipeline_start = Clock::now();
    double cpu_staging_ms = 0.0;
    double event_wait_ms = 0.0;
    size_t total_bytes = 0;

    for (size_t job_idx = 0; job_idx < jobs.size(); ++job_idx)
    {
        const auto& job = jobs[job_idx];
        const int stream_idx = static_cast<int>(job_idx % static_cast<size_t>(num_streams_));

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
                    Clock::now() - wait_start).count();
            }
        }

        // 2. CPU memcpy: mmap → pinned slot
        void* pinned_ptr = pinned_.getSlot(stream_idx);
        if (!pinned_ptr)
        {
            LOG_ERROR("DeviceLoadPipeline: pinned slot " << stream_idx << " is null");
            return false;
        }
        {
            const auto memcpy_start = Clock::now();
            std::memcpy(pinned_ptr, job.host_raw_data, job.raw_bytes);
            if (profiling)
            {
                cpu_staging_ms += std::chrono::duration<double, std::milli>(
                    Clock::now() - memcpy_start).count();
            }
        }
        total_bytes += job.raw_bytes;

        // 3. H2D async: pinned → device staging slot
        uint8_t* staging_ptr = pool_.getStagingSlot(stream_idx);
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

        // 6. Launch GPU repack kernel on repack stream
        auto slot = pool_.getSlot(job.name);
        if (!slot)
        {
            LOG_ERROR("DeviceLoadPipeline: no pool slot for weight '" << job.name << "'");
            return false;
        }

        bool repack_ok = kernels_.vnniRepack(
            job.format,
            staging_ptr,
            slot->d_native_vnni_payload,
            static_cast<uint16_t*>(slot->d_native_vnni_scales),
            static_cast<uint16_t*>(slot->d_native_vnni_mins),
            static_cast<uint32_t*>(slot->d_native_vnni_emins),
            job.N, job.K, repack_stream_);

        if (!repack_ok)
        {
            LOG_ERROR("DeviceLoadPipeline: repack kernel failed for '"
                      << job.name << "'");
            return false;
        }

        // 7. Record repack completion for this staging slot
        if (!backend_.recordEvent(repack_done_events_[stream_idx], device_id_, repack_stream_))
        {
            LOG_ERROR("DeviceLoadPipeline: recordEvent repack_done["
                      << stream_idx << "] failed");
            return false;
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
        Clock::now() - drain_start).count();
    const double pipeline_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - pipeline_start).count();

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
    LOG_INFO("DeviceLoadPipeline: device " << device_id_ << " loaded "
             << num_processed_ << " weights, " << std::fixed << std::setprecision(1)
             << total_mb << " MB in " << pipeline_ms << " ms ("
             << throughput_gbs << " GB/s)");
    return true;
}

void DeviceLoadPipeline::release()
{
    if (!initialized_) return;

    backend_.setDevice(device_id_);

    // Destroy events first
    for (auto& ev : repack_done_events_)
    {
        if (ev)
        {
            backend_.destroyEvent(ev, device_id_);
            ev = nullptr;
        }
    }
    repack_done_events_.clear();

    for (auto& ev : h2d_done_events_)
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

    for (auto& s : h2d_streams_)
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
