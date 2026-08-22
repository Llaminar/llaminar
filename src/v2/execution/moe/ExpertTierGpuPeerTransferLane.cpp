/**
 * @file ExpertTierGpuPeerTransferLane.cpp
 * @brief Direct peer-copy implementation for same-backend ExpertOverlay tiers.
 */

#include "ExpertTierGpuPeerTransferLane.h"
#include "GPUExpertTransferBackend.h"

#include "../../backends/BackendManager.h"
#include "../../backends/GPUDeviceContextPool.h"
#include "../../backends/IBackend.h"
#include "../../backends/IWorkerGPUContext.h"
#include "../../utils/Logger.h"
#include "../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** Store a peer-lane diagnostic only when the caller requested one. */
        void assignPeerError(
            std::string *error,
            const std::string &message) noexcept
        {
            if (error)
                *error = message;
        }

        /** Return whether two endpoints use the same accelerator runtime. */
        bool sameGpuBackend(DeviceId lhs, DeviceId rhs) noexcept
        {
            return (lhs.is_cuda() && rhs.is_cuda()) ||
                   (lhs.is_rocm() && rhs.is_rocm());
        }

        /**
         * @brief Preserve the maintenance thread's ambient runtime device.
         *
         * Context resource/event methods select the resource-owning ordinal.
         * A migration lane can therefore touch a destination device while the
         * caller is about to continue submitting inference work to another
         * participant.  Restoring that caller-owned state at every public lane
         * boundary prevents migration from redirecting subsequent hot-path
         * launches to the wrong GPU.
         */
        class AmbientGpuDeviceGuard final
        {
        public:
            /** Capture the current device for the endpoint's accelerator API. */
            explicit AmbientGpuDeviceGuard(DeviceId endpoint) noexcept
                : endpoint_(endpoint)
            {
#ifdef HAVE_CUDA
                if (endpoint_.is_cuda())
                {
                    saved_ordinal_ = detail::currentCUDADeviceOrdinal();
                    valid_ = saved_ordinal_ >= 0;
                    return;
                }
#endif
#ifdef HAVE_ROCM
                if (endpoint_.is_rocm())
                {
                    saved_ordinal_ = detail::currentROCmDeviceOrdinal();
                    valid_ = saved_ordinal_ >= 0;
                }
#endif
            }

            /** Restore both runtime state and ROCm's device-selection cache. */
            ~AmbientGpuDeviceGuard()
            {
                if (!valid_)
                    return;
#ifdef HAVE_CUDA
                if (endpoint_.is_cuda())
                {
                    if (!detail::restoreCUDADeviceOrdinal(saved_ordinal_))
                    {
                        LOG_WARN("[ExpertTierGpuPeerTransferLane] Could not restore CUDA device "
                                 << saved_ordinal_);
                    }
                    return;
                }
#endif
#ifdef HAVE_ROCM
                if (endpoint_.is_rocm())
                {
                    if (!detail::restoreROCmDeviceOrdinal(saved_ordinal_))
                    {
                        LOG_WARN("[ExpertTierGpuPeerTransferLane] Could not restore ROCm device "
                                 << saved_ordinal_);
                    }
                }
#endif
            }

            AmbientGpuDeviceGuard(const AmbientGpuDeviceGuard &) = delete;
            AmbientGpuDeviceGuard &operator=(
                const AmbientGpuDeviceGuard &) = delete;

        private:
            DeviceId endpoint_;
            int saved_ordinal_ = -1;
            bool valid_ = false;
        };
    } // namespace

    ExpertTierGpuPeerTransferLane::ExpertTierGpuPeerTransferLane(Config config)
        : config_(std::move(config))
    {
        if (!config_.source_device.is_gpu() ||
            !config_.destination_device.is_gpu() ||
            !sameGpuBackend(
                config_.source_device,
                config_.destination_device))
        {
            throw std::invalid_argument(
                "GPU peer tier lane requires two same-backend GPU endpoints");
        }
        if (config_.lane_name.empty())
            throw std::invalid_argument(
                "GPU peer tier lane requires a stable non-empty name");
        if (config_.perf_device.empty())
        {
            config_.perf_device = config_.source_device.to_string() + "->" +
                                  config_.destination_device.to_string();
        }
    }

    ExpertTierGpuPeerTransferLane::~ExpertTierGpuPeerTransferLane()
    {
        if (work_may_be_in_flight_)
        {
            LOG_ERROR("[ExpertTierGpuPeerTransferLane] Destroyed with pending work"
                      << " lane=" << config_.lane_name
                      << " src=" << config_.source_device.to_string()
                      << " dst=" << config_.destination_device.to_string());
            std::terminate();
        }
        releaseQuiescentResources();
    }

    bool ExpertTierGpuPeerTransferLane::materialized() const noexcept
    {
        return backend_ && source_context_ && destination_context_ &&
               destination_ordinal_ >= 0 && transfer_stream_ &&
               completion_event_ &&
               (!config_.collect_timing_measurements ||
                (timing_start_event_ && timing_stop_event_));
    }

    bool ExpertTierGpuPeerTransferLane::materialize(
        std::string *error) noexcept
    {
        const AmbientGpuDeviceGuard ambient_device(
            config_.destination_device);
        if (materialized())
            return true;
        if (backend_ || source_context_ || destination_context_ ||
            destination_ordinal_ >= 0 || transfer_stream_ || completion_event_ ||
            timing_start_event_ || timing_stop_event_)
        {
            assignPeerError(error, "GPU peer tier lane has a partial resource set");
            return false;
        }

        try
        {
            if (config_.source_device != config_.destination_device)
            {
                bool direct_peer_ready = false;
#ifdef HAVE_CUDA
                if (config_.source_device.is_cuda())
                {
                    direct_peer_ready =
                        detail::prepareDirectPeerAccessCUDABackend(
                            config_.source_device,
                            config_.destination_device);
                }
#endif
#ifdef HAVE_ROCM
                if (config_.source_device.is_rocm())
                {
                    direct_peer_ready =
                        detail::prepareDirectPeerAccessROCmBackend(
                            config_.source_device,
                            config_.destination_device);
                }
#endif
                if (!direct_peer_ready)
                {
                    throw std::runtime_error(
                        "GPU peer tier lane requires a driver-authorized direct edge; use the explicit host-relay lane when peer access is unavailable");
                }
            }

            IBackend *source_backend = getBackendFor(config_.source_device);
            IBackend *destination_backend =
                getBackendFor(config_.destination_device);
            if (!source_backend || source_backend != destination_backend)
            {
                throw std::runtime_error(
                    "GPU peer tier lane endpoints do not share one backend authority");
            }
            backend_ = destination_backend;
            source_context_ = &GPUDeviceContextPool::instance().getContext(
                config_.source_device);
            destination_context_ =
                &GPUDeviceContextPool::instance().getContext(
                    config_.destination_device);
            destination_ordinal_ = config_.destination_device.gpu_ordinal();
            transfer_stream_ = destination_context_->getOrCreateAuxiliaryStream(
                "expert_tier_gpu_peer:" + config_.lane_name);
            completion_event_ = backend_->createEvent(destination_ordinal_);
            if (config_.collect_timing_measurements)
            {
                timing_start_event_ =
                    backend_->createTimingEvent(destination_ordinal_);
                timing_stop_event_ =
                    backend_->createTimingEvent(destination_ordinal_);
            }
            if (!transfer_stream_ || !completion_event_ ||
                (config_.collect_timing_measurements &&
                 (!timing_start_event_ || !timing_stop_event_)))
                throw std::runtime_error(
                    "Could not materialize GPU peer stream/event resources");
        }
        catch (const std::exception &exception)
        {
            assignPeerError(error, exception.what());
            releaseQuiescentResources();
            return false;
        }
        catch (...)
        {
            assignPeerError(
                error,
                "GPU peer tier lane materialization threw a non-standard exception");
            releaseQuiescentResources();
            return false;
        }
        return true;
    }

    bool ExpertTierGpuPeerTransferLane::start(
        const GpuExpertPackedDescriptor &source,
        const GpuExpertPackedDescriptor &destination,
        const ExpertTierSourceReadiness &source_readiness,
        std::string *error) noexcept
    {
        if (!gpuExpertPackedDescriptorsCompatible(source, destination))
        {
            assignPeerError(
                error,
                "GPU peer tier transfer descriptors are not byte-compatible");
            return false;
        }
        return startRegions(
            source.ptrs,
            destination.ptrs,
            source.vnni_bytes,
            source.scales_bytes,
            source.mins_bytes,
            source.emins_bytes,
            source_readiness,
            error);
    }

    bool ExpertTierGpuPeerTransferLane::startContiguous(
        const void *source,
        void *destination,
        std::size_t bytes,
        const ExpertTierSourceReadiness &source_readiness,
        std::string *error) noexcept
    {
        if (!source || !destination || bytes == 0)
        {
            assignPeerError(
                error,
                "GPU peer contiguous transfer requires two pointers and non-zero bytes");
            return false;
        }
        GPUExpertPointers source_regions;
        source_regions.d_vnni = static_cast<std::uint8_t *>(
            const_cast<void *>(source));
        GPUExpertPointers destination_regions;
        destination_regions.d_vnni =
            static_cast<std::uint8_t *>(destination);
        return startRegions(
            source_regions,
            destination_regions,
            bytes,
            0,
            0,
            0,
            source_readiness,
            error);
    }

    bool ExpertTierGpuPeerTransferLane::startRegions(
        const GPUExpertPointers &source,
        const GPUExpertPointers &destination,
        std::size_t payload_bytes,
        std::size_t scales_bytes,
        std::size_t mins_bytes,
        std::size_t emins_bytes,
        const ExpertTierSourceReadiness &source_readiness,
        std::string *error) noexcept
    {
        const AmbientGpuDeviceGuard ambient_device(
            config_.destination_device);
        if (!materialized())
        {
            assignPeerError(error, "GPU peer tier lane is not materialized");
            return false;
        }
        if (progress_ == ExpertTierGpuPeerTransferProgress::Pending ||
            progress_ == ExpertTierGpuPeerTransferProgress::Failed)
        {
            assignPeerError(error, "GPU peer tier lane is already occupied");
            return false;
        }

        /*
         * A fresh producer edge is joined by the destination auxiliary stream.
         * A published bank needs no event: its preparation event completed
         * before installation and its epoch lifetime keeps bytes immutable.
         */
        if (source_readiness.requiresProducerWait())
        {
            if (!destination_context_->waitEventChecked(
                    source_readiness.event(), transfer_stream_))
            {
                assignPeerError(
                    error,
                    "Could not enqueue GPU peer source producer dependency");
                return false;
            }
            ++stats_.producer_event_waits;
        }
        else
        {
            ++stats_.published_bank_sources;
        }

        transfer_started_at_ = std::chrono::steady_clock::now();
        bool timing_started = true;
        if (config_.collect_timing_measurements)
        {
            timing_started = backend_->recordEvent(
                timing_start_event_,
                destination_ordinal_,
                transfer_stream_);
        }

        submitted_bytes_ = payload_bytes + scales_bytes + mins_bytes +
                           emins_bytes;
        bool copied = timing_started && GPUExpertTransfer::transferExpert(
            source,
            destination,
            config_.source_device,
            config_.destination_device,
            payload_bytes,
            scales_bytes,
            mins_bytes,
            emins_bytes,
            transfer_stream_);
        bool timing_stopped = true;
        if (config_.collect_timing_measurements)
        {
            timing_stopped = backend_->recordEvent(
                timing_stop_event_,
                destination_ordinal_,
                transfer_stream_);
            copied = copied && timing_stopped;
            if (!timing_started || !timing_stopped)
                ++stats_.timing_measurement_failures;
        }

        /* Fence even a partially accepted copy sequence before reporting it. */
        if (!backend_->recordEvent(
                completion_event_,
                destination_ordinal_,
                transfer_stream_))
        {
            failure_ =
                "GPU peer tier transfer could not fence submitted device work";
            assignPeerError(error, failure_);
            work_may_be_in_flight_ = true;
            progress_ = ExpertTierGpuPeerTransferProgress::Failed;
            ++stats_.failed_transfers;
            return false;
        }

        work_may_be_in_flight_ = true;
        fail_after_event_ = !copied;
        failure_ = copied
                       ? std::string{}
                       : (!timing_started || !timing_stopped)
                             ? "GPU peer tier transfer failed to record exact timing events"
                             : "GPU peer tier transfer failed to submit every packed region";
        progress_ = ExpertTierGpuPeerTransferProgress::Pending;
        ++stats_.transfers_started;
        stats_.bytes_submitted += submitted_bytes_;
        return true;
    }

    ExpertTierGpuPeerTransferProgress ExpertTierGpuPeerTransferLane::poll(
        std::string *error) noexcept
    {
        const AmbientGpuDeviceGuard ambient_device(
            config_.destination_device);
        if (progress_ != ExpertTierGpuPeerTransferProgress::Pending)
        {
            if (progress_ == ExpertTierGpuPeerTransferProgress::Failed)
                assignPeerError(error, failure_);
            return progress_;
        }

        bool ready = false;
        if (!destination_context_->queryEventChecked(completion_event_, ready))
        {
            return fail(
                "GPU peer tier completion event query failed", error);
        }
        if (!ready)
        {
            ++stats_.pending_event_polls;
            return progress_;
        }

        work_may_be_in_flight_ = false;
        if (fail_after_event_)
        {
            fail_after_event_ = false;
            return fail(failure_, error);
        }

        progress_ = ExpertTierGpuPeerTransferProgress::Ready;
        ++stats_.transfers_completed;
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - transfer_started_at_);
        const std::uint64_t wall_nanoseconds = static_cast<std::uint64_t>(
            std::max<std::int64_t>(1, elapsed.count()));
        std::uint64_t device_nanoseconds = 0;
        if (config_.collect_timing_measurements)
        {
            float elapsed_ms = 0.0F;
            if (!backend_->eventElapsedTimeMs(
                    timing_start_event_,
                    timing_stop_event_,
                    destination_ordinal_,
                    &elapsed_ms) ||
                !std::isfinite(elapsed_ms) || elapsed_ms < 0.0F)
            {
                ++stats_.timing_measurement_failures;
                --stats_.transfers_completed;
                return fail(
                    "GPU peer tier transfer could not read its exact device timing interval",
                    error);
            }
            device_nanoseconds = static_cast<std::uint64_t>(std::max(
                1.0,
                std::ceil(static_cast<double>(elapsed_ms) * 1'000'000.0)));
        }
        stats_.last_measurement = {
            .sequence = stats_.transfers_completed,
            .bytes = static_cast<std::uint64_t>(submitted_bytes_),
            .wall_nanoseconds = wall_nanoseconds,
            .device_nanoseconds = device_nanoseconds,
            .host_nanoseconds = 0,
        };
        const PerfStatsCollector::Tags tags{
            {"lane", config_.lane_name},
            {"src", config_.source_device.to_string()},
            {"dst", config_.destination_device.to_string()}};
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "same_backend_gpu_transfers_completed",
            1.0,
            "maintenance",
            config_.perf_device,
            tags);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "same_backend_gpu_bytes_completed",
            static_cast<double>(submitted_bytes_),
            "maintenance",
            config_.perf_device,
            tags);
        PerfStatsCollector::recordTimingNs(
            "moe_overlay_residency",
            "same_backend_gpu_transfer_wall_time",
            wall_nanoseconds,
            "maintenance",
            config_.perf_device,
            tags);
        if (config_.collect_timing_measurements)
        {
            PerfStatsCollector::recordTimingNs(
                "moe_overlay_residency",
                "same_backend_gpu_transfer_device_work_time",
                device_nanoseconds,
                "maintenance",
                config_.perf_device,
                tags);
        }
        return progress_;
    }

    ExpertTierGpuPeerTransferProgress ExpertTierGpuPeerTransferLane::fail(
        const std::string &message,
        std::string *error) noexcept
    {
        failure_ = message.empty()
                       ? "GPU peer tier transfer failed"
                       : message;
        assignPeerError(error, failure_);
        progress_ = ExpertTierGpuPeerTransferProgress::Failed;
        ++stats_.failed_transfers;
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "same_backend_gpu_transfer_failures",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"lane", config_.lane_name},
             {"src", config_.source_device.to_string()},
             {"dst", config_.destination_device.to_string()}});
        return progress_;
    }

    void ExpertTierGpuPeerTransferLane::releaseQuiescentResources() noexcept
    {
        const AmbientGpuDeviceGuard ambient_device(
            config_.destination_device);
        if (work_may_be_in_flight_)
            return;
        if (backend_ && completion_event_ && destination_ordinal_ >= 0)
            backend_->destroyEvent(completion_event_, destination_ordinal_);
        if (backend_ && timing_start_event_ && destination_ordinal_ >= 0)
            backend_->destroyEvent(timing_start_event_, destination_ordinal_);
        if (backend_ && timing_stop_event_ && destination_ordinal_ >= 0)
            backend_->destroyEvent(timing_stop_event_, destination_ordinal_);
        completion_event_ = nullptr;
        timing_start_event_ = nullptr;
        timing_stop_event_ = nullptr;
        transfer_stream_ = nullptr; // Context owns the named stream.
        source_context_ = nullptr;
        destination_context_ = nullptr;
        backend_ = nullptr;
        destination_ordinal_ = -1;
    }
} // namespace llaminar2
