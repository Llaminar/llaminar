/**
 * @file ExpertTierWeightTransferLane.cpp
 * @brief Event-polled implementation of background ExpertOverlay CPU edges.
 *
 * The implementation intentionally owns no inference stream. Each chunk is
 * ordered on a context-owned auxiliary stream, followed by one reusable event.
 * The maintenance worker queries that event and either commits a completed D2H
 * chunk to final CPU storage or submits the next H2D/conversion chunk.
 */

#include "ExpertTierWeightTransferLane.h"

#include "../../backends/BackendManager.h"
#include "../../backends/GPUDeviceContextPool.h"
#include "../../backends/IBackend.h"
#include "../../backends/IWorkerGPUContext.h"
#include "../../utils/Logger.h"
#include "../../utils/PerfStatsCollector.h"

#ifdef HAVE_CUDA
#include "../../kernels/cuda/repack/CUDAExpertTierWeightKernels.h"
#endif
#ifdef HAVE_ROCM
#include "../../kernels/rocm/repack/ROCmExpertTierWeightKernels.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @brief Store a diagnostic only when the caller requested one. */
        void assignError(std::string *error, const std::string &message)
        {
            if (error)
                *error = message;
        }

        /** @brief Stable PerfStats direction tag for one active lane. */
        const char *directionName(bool gpu_to_cpu) noexcept
        {
            return gpu_to_cpu ? "gpu_to_cpu" : "cpu_to_gpu";
        }
    } // namespace

    ExpertTierWeightTransferLane::ExpertTierWeightTransferLane(Config config)
        : config_(std::move(config))
    {
        if (!config_.device.is_gpu())
            throw std::invalid_argument(
                "Expert tier transfer lane requires one exact GPU endpoint");
        if (config_.staging_capacity_bytes == 0)
            throw std::invalid_argument(
                "Expert tier transfer lane requires non-zero staging capacity");
        if (config_.lane_name.empty())
            throw std::invalid_argument(
                "Expert tier transfer lane requires a stable non-empty name");
        if (config_.perf_device.empty())
            config_.perf_device = config_.device.to_string();
    }

    ExpertTierWeightTransferLane::~ExpertTierWeightTransferLane()
    {
        /*
         * Freeing staging storage while DMA is in flight would be corruption;
         * synchronizing here would turn an ownership bug into an inference
         * pause. Treat the invalid lifecycle as fatal and require the scheduler
         * to retain lanes until poll() reports Ready.
         */
        if (work_may_be_in_flight_)
        {
            LOG_ERROR("[ExpertTierWeightTransferLane] Destroyed with pending work"
                      << " lane=" << config_.lane_name
                      << " device=" << config_.device.to_string());
            std::terminate();
        }
        releaseQuiescentResources();
    }

    bool ExpertTierWeightTransferLane::materialized() const noexcept
    {
        return backend_ && gpu_context_ && device_ordinal_ >= 0 &&
               transfer_stream_ && chunk_ready_event_ && device_chunk_ &&
               pinned_chunk_ &&
               (!config_.collect_timing_measurements ||
                (chunk_timing_start_event_ && chunk_timing_stop_event_));
    }

    bool ExpertTierWeightTransferLane::materialize(std::string *error) noexcept
    {
        if (materialized())
            return true;
        if (backend_ || gpu_context_ || device_ordinal_ >= 0 ||
            transfer_stream_ || chunk_ready_event_ || device_chunk_ ||
            pinned_chunk_ || chunk_timing_start_event_ ||
            chunk_timing_stop_event_)
        {
            assignError(error, "Tier transfer lane has a partial resource set");
            return false;
        }

        try
        {
            backend_ = getBackendFor(config_.device);
            if (!backend_)
                throw std::runtime_error("No backend owns the configured GPU endpoint");
            device_ordinal_ = config_.device.gpu_ordinal();
            gpu_context_ =
                &GPUDeviceContextPool::instance().getContext(config_.device);
            transfer_stream_ = gpu_context_->getOrCreateAuxiliaryStream(
                "expert_tier_weight_transfer:" + config_.lane_name);
            if (!transfer_stream_)
                throw std::runtime_error("Could not create the auxiliary transfer stream");

            /* All allocations happen before any inference ticket can use this lane. */
            chunk_ready_event_ = backend_->createEvent(device_ordinal_);
            if (config_.collect_timing_measurements)
            {
                chunk_timing_start_event_ =
                    backend_->createTimingEvent(device_ordinal_);
                chunk_timing_stop_event_ =
                    backend_->createTimingEvent(device_ordinal_);
            }
            device_chunk_ = static_cast<std::uint8_t *>(
                backend_->allocate(config_.staging_capacity_bytes, device_ordinal_));
            pinned_chunk_ = static_cast<std::uint8_t *>(
                backend_->allocatePinned(
                    config_.staging_capacity_bytes, device_ordinal_));
            if (!chunk_ready_event_ || !device_chunk_ || !pinned_chunk_ ||
                (config_.collect_timing_measurements &&
                 (!chunk_timing_start_event_ || !chunk_timing_stop_event_)))
                throw std::runtime_error("Could not allocate persistent event/staging resources");
        }
        catch (const std::exception &exception)
        {
            assignError(error, exception.what());
            releaseQuiescentResources();
            return false;
        }
        catch (...)
        {
            assignError(error, "Tier transfer lane materialization threw a non-standard exception");
            releaseQuiescentResources();
            return false;
        }
        return true;
    }

    bool ExpertTierWeightTransferLane::startGpuToCpu(
        const ExpertTierWeightDeviceLayout &layout,
        const ExpertTierGpuConstProjectionView &source,
        std::span<std::uint8_t> final_cpu_bytes,
        const ExpertTierSourceReadiness &source_readiness,
        std::string *error) noexcept
    {
        if (!materialized())
        {
            assignError(error, "GPU-to-CPU tier transfer lane is not materialized");
            return false;
        }
        if (progress_ == ExpertTierWeightTransferProgress::Pending ||
            progress_ == ExpertTierWeightTransferProgress::Failed)
        {
            assignError(error, "GPU-to-CPU tier transfer lane is already occupied");
            return false;
        }
        if (!layout.valid() ||
            layout.direction != ExpertTierWeightConversionDirection::GpuToCpu ||
            !source.validFor(layout))
        {
            assignError(error, "GPU-to-CPU tier transfer has invalid layout or source arrays");
            return false;
        }
        const std::size_t total_bytes = layout.chunkBytes(layout.unit_count);
        if (final_cpu_bytes.size() != total_bytes ||
            layout.chunkBytes(layout.maximum_units_per_chunk) >
                config_.staging_capacity_bytes)
        {
            assignError(error, "GPU-to-CPU tier transfer storage does not match its manifest");
            return false;
        }
        /*
         * Fresh preparation requires the exact producer edge. An installed RCU
         * bank was admitted only after that edge completed and is retained by
         * the migration transaction, so adding a fabricated event would be both
         * redundant and an ownership lie.
         */
        if (source_readiness.requiresProducerWait() &&
            !gpu_context_->waitEventChecked(
                source_readiness.event(), transfer_stream_))
        {
            assignError(error, "Could not enqueue source-to-transfer event dependency");
            return false;
        }

        direction_ = Direction::GpuToCpu;
        layout_ = layout;
        gpu_source_ = source;
        gpu_destination_ = {};
        cpu_destination_ = final_cpu_bytes;
        cpu_source_ = {};
        completed_units_ = 0;
        in_flight_units_ = 0;
        in_flight_bytes_ = 0;
        work_may_be_in_flight_ = false;
        fail_after_in_flight_event_ = false;
        in_flight_timing_valid_ = false;
        transfer_device_nanoseconds_ = 0;
        transfer_host_nanoseconds_ = 0;
        transfer_bytes_ = static_cast<std::uint64_t>(total_bytes);
        failure_.clear();
        progress_ = ExpertTierWeightTransferProgress::Pending;
        transfer_started_at_ = std::chrono::steady_clock::now();
        ++stats_.transfers_started;
        return enqueueNextChunk(error);
    }

    bool ExpertTierWeightTransferLane::startCpuToGpu(
        const ExpertTierWeightDeviceLayout &layout,
        std::span<const std::uint8_t> cpu_bytes,
        const ExpertTierGpuMutableProjectionView &destination,
        std::string *error) noexcept
    {
        if (!materialized())
        {
            assignError(error, "CPU-to-GPU tier transfer lane is not materialized");
            return false;
        }
        if (progress_ == ExpertTierWeightTransferProgress::Pending ||
            progress_ == ExpertTierWeightTransferProgress::Failed)
        {
            assignError(error, "CPU-to-GPU tier transfer lane is already occupied");
            return false;
        }
        if (!layout.valid() ||
            layout.direction != ExpertTierWeightConversionDirection::CpuToGpu ||
            !destination.validFor(layout))
        {
            assignError(error, "CPU-to-GPU tier transfer has invalid layout or destination arrays");
            return false;
        }
        const std::size_t total_bytes = layout.chunkBytes(layout.unit_count);
        if (cpu_bytes.size() != total_bytes ||
            layout.chunkBytes(layout.maximum_units_per_chunk) >
                config_.staging_capacity_bytes)
        {
            assignError(error, "CPU-to-GPU tier transfer storage does not match its manifest");
            return false;
        }

        direction_ = Direction::CpuToGpu;
        layout_ = layout;
        gpu_source_ = {};
        gpu_destination_ = destination;
        cpu_destination_ = {};
        cpu_source_ = cpu_bytes;
        completed_units_ = 0;
        in_flight_units_ = 0;
        in_flight_bytes_ = 0;
        work_may_be_in_flight_ = false;
        fail_after_in_flight_event_ = false;
        in_flight_timing_valid_ = false;
        transfer_device_nanoseconds_ = 0;
        transfer_host_nanoseconds_ = 0;
        transfer_bytes_ = static_cast<std::uint64_t>(total_bytes);
        failure_.clear();
        progress_ = ExpertTierWeightTransferProgress::Pending;
        transfer_started_at_ = std::chrono::steady_clock::now();
        ++stats_.transfers_started;
        return enqueueNextChunk(error);
    }

    ExpertTierWeightTransferProgress ExpertTierWeightTransferLane::fail(
        const std::string &message,
        std::string *error) noexcept
    {
        failure_ = message;
        assignError(error, failure_);
        progress_ = ExpertTierWeightTransferProgress::Failed;
        ++stats_.failed_transfers;
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "tier_transfer_failures",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"lane", config_.lane_name},
             {"device", config_.device.to_string()}});
        return progress_;
    }

    bool ExpertTierWeightTransferLane::launchGpuToCpuChunk(
        std::uint32_t first_unit,
        std::uint32_t unit_count,
        std::size_t bytes) noexcept
    {
        if (config_.device.is_cuda())
        {
#ifdef HAVE_CUDA
            return launchGpuToCpuExpertTierChunkCUDA(
                gpu_source_, layout_, first_unit, unit_count, device_chunk_,
                config_.staging_capacity_bytes, transfer_stream_);
#else
            return false;
#endif
        }
        if (config_.device.is_rocm())
        {
#ifdef HAVE_ROCM
            return launchGpuToCpuExpertTierChunkROCm(
                gpu_source_, layout_, first_unit, unit_count, device_chunk_,
                config_.staging_capacity_bytes, transfer_stream_);
#else
            return false;
#endif
        }
        (void)bytes;
        return false;
    }

    bool ExpertTierWeightTransferLane::launchCpuToGpuChunk(
        std::uint32_t first_unit,
        std::uint32_t unit_count,
        std::size_t bytes) noexcept
    {
        if (config_.device.is_cuda())
        {
#ifdef HAVE_CUDA
            return launchCpuToGpuExpertTierChunkCUDA(
                device_chunk_, bytes, layout_, first_unit, unit_count,
                gpu_destination_, transfer_stream_);
#else
            return false;
#endif
        }
        if (config_.device.is_rocm())
        {
#ifdef HAVE_ROCM
            return launchCpuToGpuExpertTierChunkROCm(
                device_chunk_, bytes, layout_, first_unit, unit_count,
                gpu_destination_, transfer_stream_);
#else
            return false;
#endif
        }
        return false;
    }

    bool ExpertTierWeightTransferLane::enqueueNextChunk(
        std::string *error) noexcept
    {
        const std::uint32_t remaining = layout_.unit_count - completed_units_;
        const std::uint32_t unit_count =
            std::min(layout_.maximum_units_per_chunk, remaining);
        const std::size_t bytes = layout_.chunkBytes(unit_count);
        if (unit_count == 0 || bytes == 0 ||
            bytes > config_.staging_capacity_bytes)
        {
            fail("Tier transfer computed an invalid next chunk", error);
            return false;
        }

        bool timing_started = true;
        if (config_.collect_timing_measurements)
        {
            timing_started = backend_->recordEvent(
                chunk_timing_start_event_,
                device_ordinal_,
                transfer_stream_);
        }

        bool submitted = timing_started;
        if (direction_ == Direction::GpuToCpu)
        {
            const bool converted = submitted && launchGpuToCpuChunk(
                completed_units_, unit_count, bytes);
            const bool copied = converted && backend_->deviceToHostOnStream(
                pinned_chunk_, device_chunk_, bytes,
                device_ordinal_, transfer_stream_);
            submitted = converted && copied;
        }
        else if (direction_ == Direction::CpuToGpu)
        {
            /*
             * Copying into pinned staging is CPU maintenance work. The source
             * expert remains immutable for the old epoch, so no inference
             * ownership edge is required.
             */
            const auto host_copy_started = std::chrono::steady_clock::now();
            std::memcpy(
                pinned_chunk_,
                cpu_source_.data() + layout_.chunkBytes(completed_units_),
                bytes);
            recordHostCopyDuration(
                std::chrono::steady_clock::now() - host_copy_started);
            const bool copied = submitted && backend_->hostToDeviceOnStream(
                device_chunk_, pinned_chunk_, bytes,
                device_ordinal_, transfer_stream_);
            const bool converted = copied && launchCpuToGpuChunk(
                completed_units_, unit_count, bytes);
            submitted = copied && converted;
        }

        bool timing_stopped = true;
        if (config_.collect_timing_measurements)
        {
            timing_stopped = backend_->recordEvent(
                chunk_timing_stop_event_,
                device_ordinal_,
                transfer_stream_);
            submitted = submitted && timing_stopped;
        }

        /*
         * Record a fence even after a partial submission failure. This keeps
         * staging resources alive until every successfully enqueued operation
         * is known complete; the maintenance poll then reports the failure.
         */
        if (!backend_->recordEvent(
                chunk_ready_event_, device_ordinal_, transfer_stream_))
        {
            failure_ = "Tier transfer could not fence partially submitted device work";
            assignError(error, failure_);
            work_may_be_in_flight_ = true;
            progress_ = ExpertTierWeightTransferProgress::Failed;
            ++stats_.failed_transfers;
            return false;
        }

        in_flight_units_ = unit_count;
        in_flight_bytes_ = bytes;
        work_may_be_in_flight_ = true;
        fail_after_in_flight_event_ = !submitted;
        in_flight_timing_valid_ = timing_started && timing_stopped;
        if (!submitted)
        {
            failure_ = in_flight_timing_valid_
                           ? "Tier transfer failed to submit a conversion/DMA chunk"
                           : "Tier transfer failed to record exact timing events";
            if (!in_flight_timing_valid_ &&
                config_.collect_timing_measurements)
            {
                ++stats_.timing_measurement_failures;
            }
        }
        recordSubmittedChunk(bytes);
        return true;
    }

    ExpertTierWeightTransferProgress ExpertTierWeightTransferLane::poll(
        std::string *error) noexcept
    {
        if (progress_ != ExpertTierWeightTransferProgress::Pending)
        {
            if (progress_ == ExpertTierWeightTransferProgress::Failed)
                assignError(error, failure_);
            return progress_;
        }

        bool ready = false;
        if (!gpu_context_->queryEventChecked(chunk_ready_event_, ready))
            return fail("Tier transfer completion event query failed", error);
        if (!ready)
        {
            ++stats_.pending_event_polls;
            return progress_;
        }

        work_may_be_in_flight_ = false;
        if (config_.collect_timing_measurements &&
            in_flight_timing_valid_ &&
            !collectCompletedChunkTiming(error))
        {
            in_flight_timing_valid_ = false;
            return fail(
                "Tier transfer could not read its exact device timing interval",
                error);
        }
        in_flight_timing_valid_ = false;
        if (fail_after_in_flight_event_)
        {
            fail_after_in_flight_event_ = false;
            return fail(
                failure_.empty()
                    ? "Tier transfer chunk failed before its completion fence"
                    : failure_,
                error);
        }

        if (direction_ == Direction::GpuToCpu)
        {
            /* D2H completion makes the pinned bytes safe for final CPU copy. */
            const auto host_copy_started = std::chrono::steady_clock::now();
            std::memcpy(
                cpu_destination_.data() + layout_.chunkBytes(completed_units_),
                pinned_chunk_,
                in_flight_bytes_);
            recordHostCopyDuration(
                std::chrono::steady_clock::now() - host_copy_started);
        }

        completed_units_ += in_flight_units_;
        in_flight_units_ = 0;
        in_flight_bytes_ = 0;
        if (completed_units_ == layout_.unit_count)
        {
            progress_ = ExpertTierWeightTransferProgress::Ready;
            ++stats_.transfers_completed;
            recordCompletion();
            return progress_;
        }

        if (!enqueueNextChunk(error))
            return progress_;
        return ExpertTierWeightTransferProgress::Pending;
    }

    void ExpertTierWeightTransferLane::recordSubmittedChunk(
        std::size_t bytes) noexcept
    {
        ++stats_.chunks_submitted;
        stats_.bytes_submitted += bytes;
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "tier_transfer_chunks_submitted",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"lane", config_.lane_name},
             {"direction", directionName(direction_ == Direction::GpuToCpu)},
             {"device", config_.device.to_string()}});
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "tier_transfer_bytes_submitted",
            static_cast<double>(bytes),
            "maintenance",
            config_.perf_device,
            {{"lane", config_.lane_name},
             {"direction", directionName(direction_ == Direction::GpuToCpu)},
             {"device", config_.device.to_string()}});
    }

    void ExpertTierWeightTransferLane::recordCompletion() noexcept
    {
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - transfer_started_at_);
        const std::uint64_t wall_nanoseconds = static_cast<std::uint64_t>(
            std::max<std::int64_t>(1, elapsed.count()));
        stats_.last_measurement = {
            .sequence = stats_.transfers_completed,
            .bytes = transfer_bytes_,
            .wall_nanoseconds = wall_nanoseconds,
            .device_nanoseconds = transfer_device_nanoseconds_,
            .host_nanoseconds = transfer_host_nanoseconds_,
        };
        const bool gpu_to_cpu = direction_ == Direction::GpuToCpu;
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "tier_transfers_completed",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"lane", config_.lane_name},
             {"direction", directionName(gpu_to_cpu)},
             {"device", config_.device.to_string()}});
        PerfStatsCollector::recordTimingNs(
            "moe_overlay_residency",
            "tier_transfer_wall_time",
            wall_nanoseconds,
            "maintenance",
            config_.perf_device,
            {{"lane", config_.lane_name},
             {"direction", directionName(gpu_to_cpu)},
             {"device", config_.device.to_string()}});
        if (config_.collect_timing_measurements)
        {
            PerfStatsCollector::recordTimingNs(
                "moe_overlay_residency",
                "tier_transfer_device_work_time",
                transfer_device_nanoseconds_,
                "maintenance",
                config_.perf_device,
                {{"lane", config_.lane_name},
                 {"direction", directionName(gpu_to_cpu)},
                 {"device", config_.device.to_string()}});
            PerfStatsCollector::recordTimingNs(
                "moe_overlay_residency",
                "tier_transfer_host_work_time",
                transfer_host_nanoseconds_,
                "maintenance",
                config_.perf_device,
                {{"lane", config_.lane_name},
                 {"direction", directionName(gpu_to_cpu)},
                 {"device", config_.device.to_string()}});
        }
    }

    bool ExpertTierWeightTransferLane::collectCompletedChunkTiming(
        std::string *error) noexcept
    {
        float elapsed_ms = 0.0F;
        if (!backend_->eventElapsedTimeMs(
                chunk_timing_start_event_,
                chunk_timing_stop_event_,
                device_ordinal_,
                &elapsed_ms) ||
            !std::isfinite(elapsed_ms) || elapsed_ms < 0.0F)
        {
            ++stats_.timing_measurement_failures;
            assignError(error, "Tier transfer timing-event interval is unavailable");
            return false;
        }
        const auto nanoseconds = static_cast<std::uint64_t>(std::max(
            1.0,
            std::ceil(static_cast<double>(elapsed_ms) * 1'000'000.0)));
        transfer_device_nanoseconds_ = saturatingExpertTierMeasurementAdd(
            transfer_device_nanoseconds_, nanoseconds);
        return true;
    }

    void ExpertTierWeightTransferLane::recordHostCopyDuration(
        std::chrono::steady_clock::duration duration) noexcept
    {
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            duration);
        const std::uint64_t nanoseconds = static_cast<std::uint64_t>(
            std::max<std::int64_t>(1, elapsed.count()));
        transfer_host_nanoseconds_ = saturatingExpertTierMeasurementAdd(
            transfer_host_nanoseconds_, nanoseconds);
    }

    void ExpertTierWeightTransferLane::releaseQuiescentResources() noexcept
    {
        if (backend_ && device_ordinal_ >= 0)
        {
            if (chunk_ready_event_)
                backend_->destroyEvent(chunk_ready_event_, device_ordinal_);
            if (chunk_timing_start_event_)
                backend_->destroyEvent(
                    chunk_timing_start_event_, device_ordinal_);
            if (chunk_timing_stop_event_)
                backend_->destroyEvent(
                    chunk_timing_stop_event_, device_ordinal_);
            if (device_chunk_)
                backend_->free(device_chunk_, device_ordinal_);
            if (pinned_chunk_)
                backend_->freePinned(pinned_chunk_, device_ordinal_);
        }
        chunk_ready_event_ = nullptr;
        chunk_timing_start_event_ = nullptr;
        chunk_timing_stop_event_ = nullptr;
        device_chunk_ = nullptr;
        pinned_chunk_ = nullptr;
        /* Auxiliary stream lifetime belongs to the worker GPU context. */
        transfer_stream_ = nullptr;
        gpu_context_ = nullptr;
        backend_ = nullptr;
        device_ordinal_ = -1;
    }
} // namespace llaminar2
