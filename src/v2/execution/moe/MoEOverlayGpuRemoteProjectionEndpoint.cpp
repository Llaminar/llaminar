/**
 * @file MoEOverlayGpuRemoteProjectionEndpoint.cpp
 * @brief Non-blocking device progression for remote ExpertOverlay projections.
 *
 * The maintenance worker calls every method in this file opportunistically.
 * Device ordering is expressed solely by one named auxiliary stream and exact
 * events; no method waits for a stream/device, allocates runtime storage, or
 * makes an inference stream depend on background migration.
 * CPU repack and packed GPU blobs use the same TransferEngine background-copy
 * contract as node-local relay lanes. Exact mapped aliases belong to persistent
 * staging setup, never to a hot-path registration or assumed host-pointer cast.
 * Command-local stall warnings reuse already-performed native event queries;
 * they add no device observation and cannot decide transfer readiness.
 */

#include "MoEOverlayGpuRemoteProjectionEndpoint.h"

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IWorkerGPUContext.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#ifdef HAVE_CUDA
#include "kernels/cuda/repack/CUDAExpertTierWeightKernels.h"
#endif
#ifdef HAVE_ROCM
#include "kernels/rocm/repack/ROCmExpertTierWeightKernels.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @return Monotonic nanoseconds for observational chunk-stall timing. */
        std::uint64_t remoteProjectionNanoseconds() noexcept
        {
            return static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
        }

        /** @brief Write a caller-facing error only when storage was supplied. */
        void setGpuRemoteError(
            std::string *error,
            const std::string &message) noexcept
        {
            if (error)
                *error = message;
        }

        /** @brief Exact read view over a common separated GPU descriptor. */
        ExpertTierGpuConstProjectionView gpuConstView(
            const GpuExpertPackedDescriptor &descriptor) noexcept
        {
            return {
                .payload = descriptor.ptrs.d_vnni,
                .scales = static_cast<const std::uint16_t *>(
                    descriptor.ptrs.d_scales),
                .mins = static_cast<const std::uint16_t *>(
                    descriptor.ptrs.d_mins),
                .emins = static_cast<const std::uint32_t *>(
                    descriptor.ptrs.d_emins),
                .payload_bytes = descriptor.vnni_bytes,
                .scales_bytes = descriptor.scales_bytes,
                .mins_bytes = descriptor.mins_bytes,
                .emins_bytes = descriptor.emins_bytes,
            };
        }

        /** @brief Exact writable view over a common separated GPU descriptor. */
        ExpertTierGpuMutableProjectionView gpuMutableView(
            const GpuExpertPackedDescriptor &descriptor) noexcept
        {
            return {
                .payload = descriptor.ptrs.d_vnni,
                .scales = static_cast<std::uint16_t *>(
                    descriptor.ptrs.d_scales),
                .mins = static_cast<std::uint16_t *>(
                    descriptor.ptrs.d_mins),
                .emins = static_cast<std::uint32_t *>(
                    descriptor.ptrs.d_emins),
                .payload_bytes = descriptor.vnni_bytes,
                .scales_bytes = descriptor.scales_bytes,
                .mins_bytes = descriptor.mins_bytes,
                .emins_bytes = descriptor.emins_bytes,
            };
        }

        /** @brief Return whether optional GPU regions have matching pointers. */
        bool descriptorPointersComplete(
            const GpuExpertPackedDescriptor &descriptor) noexcept
        {
            return descriptor.valid() &&
                   (descriptor.mins_bytes == 0 ||
                    descriptor.ptrs.d_mins != nullptr) &&
                   (descriptor.emins_bytes == 0 ||
                    descriptor.ptrs.d_emins != nullptr);
        }

        /** @brief Compare the complete semantic chunk header. */
        bool sameChunkHeader(
            const MoEOverlayRemoteProjectionChunkHeader &left,
            const MoEOverlayRemoteProjectionChunkHeader &right) noexcept
        {
            return left.manifest_hash == right.manifest_hash &&
                   left.payload_hash == right.payload_hash &&
                   left.sequence == right.sequence &&
                   left.region == right.region &&
                   std::equal(
                       std::begin(left.reserved_region),
                       std::end(left.reserved_region),
                       std::begin(right.reserved_region)) &&
                   left.region_offset == right.region_offset &&
                   left.payload_bytes == right.payload_bytes &&
                   left.final_chunk == right.final_chunk &&
                   std::equal(
                       std::begin(left.reserved_final),
                       std::end(left.reserved_final),
                       std::begin(right.reserved_final));
        }

    } // namespace

    MoEOverlayGpuRemoteProjectionLane::MoEOverlayGpuRemoteProjectionLane(
        Config config)
        : config_(std::move(config))
    {
        if (!config_.device.is_gpu())
            throw std::invalid_argument(
                "Remote ExpertOverlay GPU lane requires one exact GPU device");
        if (!config_.staging.valid() ||
            config_.staging.device() != config_.device)
            throw std::invalid_argument(
                "Remote ExpertOverlay GPU lane requires one matching persistent staging slice");
        if (!config_.execution.valid() ||
            config_.execution.device() != config_.device)
            throw std::invalid_argument(
                "Remote ExpertOverlay GPU lane requires one matching persistent execution lane");
        if (config_.lane_name.empty())
            throw std::invalid_argument(
                "Remote ExpertOverlay GPU lane requires a stable name");
        if (config_.progress.epoch() && config_.progress.epoch()->device() != config_.device)
            throw std::invalid_argument("Remote projection progress authority belongs to another device");
        if (config_.perf_device.empty())
            config_.perf_device = config_.device.to_string();
    }

    MoEOverlayGpuRemoteProjectionLane::~MoEOverlayGpuRemoteProjectionLane()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (owner_ || work_may_be_in_flight_ || unfenced_work_)
            {
                LOG_ERROR(
                    "[MoEOverlayGpuRemoteProjectionLane] Destroyed before quiescent release"
                    << " lane=" << config_.lane_name
                    << " device=" << config_.device.to_string());
                std::terminate();
            }
        }
        releaseResources();
    }

    bool MoEOverlayGpuRemoteProjectionLane::materialized() const noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return backend_ && context_ && device_ordinal_ >= 0 && stream_ &&
               completion_event_ && device_chunk_ && pinned_chunk_;
    }

    bool MoEOverlayGpuRemoteProjectionLane::materialize(
        std::string *error) noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (backend_ && context_ && device_ordinal_ >= 0 && stream_ &&
            completion_event_ && device_chunk_ && pinned_chunk_)
        {
            if (error)
                error->clear();
            return true;
        }
        if (backend_ || context_ || device_ordinal_ >= 0 || stream_ ||
            completion_event_ || device_chunk_ || pinned_chunk_)
        {
            setGpuRemoteError(
                error,
                "Remote ExpertOverlay GPU lane has a partial resource set");
            return false;
        }

        try
        {
            backend_ = getBackendFor(config_.device);
            if (!backend_)
                throw std::runtime_error(
                    "No backend owns the remote ExpertOverlay GPU lane");
            device_ordinal_ = config_.device.gpu_ordinal();
            context_ = &GPUDeviceContextPool::instance().getContext(
                config_.device);
            stream_ = config_.execution.stream();
            completion_event_ = backend_->createEvent(device_ordinal_);
            /* The pool-wide TransferEngine slab owns both stable addresses. */
            device_chunk_ = static_cast<std::uint8_t *>(
                config_.staging.mutableDeviceData());
            pinned_chunk_ = static_cast<std::uint8_t *>(
                config_.staging.mutablePinnedData());
            if (const auto &epoch = config_.progress.epoch())
            {
                auto mapped = TransferEngine::instance().mappedStagingView(config_.staging);
                service_read_ = epoch->reserveSlot(MappedTransferDirection::DeviceToHost,
                    mapped, config_.lane_name + ":read");
                service_write_ = epoch->reserveSlot(MappedTransferDirection::HostToDevice,
                    std::move(mapped), config_.lane_name + ":write");
            }
            if (!stream_ || !completion_event_ || !device_chunk_ ||
                !pinned_chunk_)
            {
                throw std::runtime_error(
                    "Could not materialize remote ExpertOverlay GPU stream/event/staging resources");
            }
        }
        catch (const std::exception &exception)
        {
            setGpuRemoteError(error, exception.what());
            /*
             * No work exists during model-time setup, so the lane-owned event
             * can be reclaimed immediately.  The two chunk pointers are only
             * views into TransferEngine-owned slabs retained by config_.staging;
             * releaseResources() deliberately clears those views without
             * attempting to free an interior address.
             */
            releaseResources();
            return false;
        }
        catch (...)
        {
            setGpuRemoteError(
                error,
                "Remote ExpertOverlay GPU lane setup threw a non-standard exception");
            /* The shared slab owners remain exclusively in config_.staging. */
            releaseResources();
            return false;
        }
        if (error)
            error->clear();
        return true;
    }

    bool MoEOverlayGpuRemoteProjectionLane::tryAcquire(
        const void *owner) noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!owner || owner_ || !backend_ || !context_ || !stream_ ||
            !completion_event_ || !device_chunk_ || !pinned_chunk_)
        {
            return false;
        }
        owner_ = owner;
        progress_ = MoEOverlayGpuRemoteLaneProgress::Idle;
        operation_kind_ = OperationKind::None;
        operation_bytes_ = 0;
        source_readiness_bound_ = false;
        source_dependency_ = TransferProducerDependency::published();
        work_may_be_in_flight_ = false;
        fail_after_event_ = false;
        unfenced_work_ = false;
        failure_.clear();
        ++stats_.owners_acquired;
        return true;
    }

    bool MoEOverlayGpuRemoteProjectionLane::release(
        const void *owner,
        std::string *error) noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!owner || owner_ != owner)
        {
            setGpuRemoteError(
                error,
                "Remote ExpertOverlay GPU lane release has the wrong owner");
            return false;
        }
        if (work_may_be_in_flight_ || unfenced_work_ ||
            progress_ == MoEOverlayGpuRemoteLaneProgress::Pending)
        {
            setGpuRemoteError(
                error,
                "Remote ExpertOverlay GPU lane cannot release in-flight storage");
            return false;
        }
        owner_ = nullptr;
        progress_ = MoEOverlayGpuRemoteLaneProgress::Idle;
        operation_kind_ = OperationKind::None;
        operation_bytes_ = 0;
        source_readiness_bound_ = false;
        fail_after_event_ = false;
        failure_.clear();
        if (error)
            error->clear();
        return true;
    }

    bool MoEOverlayGpuRemoteProjectionLane::bindSourceReadiness(
        const void *owner,
        const ExpertTierSourceReadiness &readiness,
        std::string *error) noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (owner_ != owner || !owner ||
            progress_ != MoEOverlayGpuRemoteLaneProgress::Idle ||
            source_readiness_bound_)
        {
            setGpuRemoteError(
                error,
                "Remote ExpertOverlay GPU source readiness has an invalid lane lifecycle");
            return false;
        }
        try
        {
            if (readiness.requiresProducerWait())
            {
                source_dependency_ = TransferProducerDependency::afterEvent(readiness.event());
                if (!context_->waitEventChecked(readiness.event(), stream_))
                {
                    setGpuRemoteError(
                        error,
                        "Remote ExpertOverlay GPU lane could not enqueue its producer event edge");
                    return false;
                }
                ++stats_.producer_event_waits;
            }
            else
            {
                ++stats_.published_bank_sources;
            }
        }
        catch (const std::exception &exception)
        {
            setGpuRemoteError(error, exception.what());
            return false;
        }
        catch (...)
        {
            setGpuRemoteError(
                error,
                "Remote ExpertOverlay GPU source ordering threw a non-standard exception");
            return false;
        }
        source_readiness_bound_ = true;
        if (error)
            error->clear();
        return true;
    }

    bool MoEOverlayGpuRemoteProjectionLane::canSubmitLocked(
        const void *owner,
        bool source_operation,
        std::string *error) noexcept
    {
        if (!owner || owner_ != owner ||
            progress_ != MoEOverlayGpuRemoteLaneProgress::Idle ||
            work_may_be_in_flight_ || unfenced_work_ ||
            (source_operation && !source_readiness_bound_))
        {
            setGpuRemoteError(
                error,
                "Remote ExpertOverlay GPU chunk submission has an invalid lane owner, state, or source edge");
            return false;
        }
        return true;
    }

    bool MoEOverlayGpuRemoteProjectionLane::launchGpuToCpuLocked(
        const ExpertTierWeightDeviceLayout &layout,
        const ExpertTierGpuConstProjectionView &source,
        std::uint32_t first_unit,
        std::uint32_t unit_count) noexcept
    {
        /*
         * A maintenance worker can progress lanes for several devices on one
         * host thread.  Streams and pointers retain their owning device, but
         * CUDA/HIP kernel launch state is thread-local.  Select this lane's
         * exact device before dispatch so a device-1 stream is never submitted
         * while the thread still names device 0.  setDevice() is a context
         * selection operation; it does not synchronize either stream.
         */
        if (!backend_ || !backend_->setDevice(device_ordinal_))
            return false;
        if (config_.device.is_cuda())
        {
#ifdef HAVE_CUDA
            return launchGpuToCpuExpertTierChunkCUDA(
                source,
                layout,
                first_unit,
                unit_count,
                device_chunk_,
                config_.staging.sizeBytes(),
                stream_);
#else
            return false;
#endif
        }
        if (config_.device.is_rocm())
        {
#ifdef HAVE_ROCM
            return launchGpuToCpuExpertTierChunkROCm(
                source,
                layout,
                first_unit,
                unit_count,
                device_chunk_,
                config_.staging.sizeBytes(),
                stream_);
#else
            return false;
#endif
        }
        return false;
    }

    bool MoEOverlayGpuRemoteProjectionLane::launchCpuToGpuLocked(
        const ExpertTierWeightDeviceLayout &layout,
        const ExpertTierGpuMutableProjectionView &destination,
        std::uint32_t first_unit,
        std::uint32_t unit_count,
        std::size_t bytes) noexcept
    {
        /* Keep repack dispatch device-exact even if the preceding H2D copy is
         * later replaced or reordered.  Kernel correctness must not depend on
         * an unrelated backend call incidentally selecting this device. */
        if (!backend_ || !backend_->setDevice(device_ordinal_))
            return false;
        if (config_.device.is_cuda())
        {
#ifdef HAVE_CUDA
            return launchCpuToGpuExpertTierChunkCUDA(
                device_chunk_,
                bytes,
                layout,
                first_unit,
                unit_count,
                destination,
                stream_);
#else
            return false;
#endif
        }
        if (config_.device.is_rocm())
        {
#ifdef HAVE_ROCM
            return launchCpuToGpuExpertTierChunkROCm(
                device_chunk_,
                bytes,
                layout,
                first_unit,
                unit_count,
                destination,
                stream_);
#else
            return false;
#endif
        }
        return false;
    }

    bool MoEOverlayGpuRemoteProjectionLane::fenceSubmissionLocked(
        bool submitted,
        OperationKind kind,
        std::size_t bytes,
        std::string *error) noexcept
    {
        /*
         * A failed compound operation may still have accepted its first DMA or
         * kernel. Record the event unconditionally so staging ownership remains
         * explicit until the maintenance worker observes that exact stream point.
         */
        if (!backend_->recordEvent(
                completion_event_, device_ordinal_, stream_))
        {
            failure_ =
                "Remote ExpertOverlay GPU lane could not fence possibly submitted work";
            progress_ = MoEOverlayGpuRemoteLaneProgress::Failed;
            unfenced_work_ = true;
            ++stats_.failures;
            setGpuRemoteError(error, failure_);
            return false;
        }
        progress_ = MoEOverlayGpuRemoteLaneProgress::Pending;
        operation_kind_ = kind;
        operation_bytes_ = bytes;
        progress_watch_.published(remoteProjectionNanoseconds());
        work_may_be_in_flight_ = true;
        fail_after_event_ = !submitted;
        if (!submitted && failure_.empty())
            failure_ = "Remote ExpertOverlay GPU chunk submission failed before its completion fence";
        recordSubmissionLocked(kind, bytes);
        if (!submitted)
            setGpuRemoteError(error, failure_);
        else if (error)
            error->clear();
        return true;
    }

    bool MoEOverlayGpuRemoteProjectionLane::submitGpuToCpuRepack(
        const void *owner,
        const ExpertTierWeightDeviceLayout &layout,
        const ExpertTierGpuConstProjectionView &source,
        std::uint32_t first_unit,
        std::uint32_t unit_count,
        std::string *error) noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::size_t bytes = layout.chunkBytes(unit_count);
        if (!canSubmitLocked(owner, true, error) || !layout.valid() ||
            layout.direction !=
                ExpertTierWeightConversionDirection::GpuToCpu ||
            !source.validFor(layout) || unit_count == 0 ||
            first_unit >= layout.unit_count ||
            unit_count > layout.unit_count - first_unit ||
            unit_count > layout.maximum_units_per_chunk || bytes == 0 ||
            bytes > config_.staging.sizeBytes())
        {
            if (error && error->empty())
                *error = "Remote GPU-to-CPU repack chunk is outside its authenticated layout";
            return false;
        }
        const bool converted = launchGpuToCpuLocked(
            layout, source, first_unit, unit_count);
        const bool copied = converted && TransferEngine::instance().enqueueBackgroundStagingCopy(
            config_.execution, MappedTransferDirection::DeviceToHost,
            device_chunk_, config_.staging.sizeBytes(), 0u,
            config_.staging, 0u, bytes, &failure_);
        return fenceSubmissionLocked(
            converted && copied,
            OperationKind::GpuToCpuRepack,
            bytes,
            error);
    }

    bool MoEOverlayGpuRemoteProjectionLane::submitGpuBlobRead(
        const void *owner,
        const std::uint8_t *source,
        std::size_t bytes,
        std::string *error) noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!canSubmitLocked(owner, true, error) || !source || bytes == 0 ||
            bytes > config_.staging.sizeBytes())
        {
            if (error && error->empty())
                *error = "Remote GPU blob read has an invalid source range";
            return false;
        }
        if (config_.progress.epoch())
            return publishBlobLocked(OperationKind::GpuBlobRead,
                const_cast<std::uint8_t *>(source), bytes, error);
        const bool copied = TransferEngine::instance().enqueueBackgroundStagingCopy(
            config_.execution, MappedTransferDirection::DeviceToHost,
            const_cast<std::uint8_t *>(source), bytes, 0u,
            config_.staging, 0u, bytes, &failure_);
        return fenceSubmissionLocked(
            copied,
            OperationKind::GpuBlobRead,
            bytes,
            error);
    }

    bool MoEOverlayGpuRemoteProjectionLane::submitCpuToGpuRepack(
        const void *owner,
        const ExpertTierWeightDeviceLayout &layout,
        const ExpertTierGpuMutableProjectionView &destination,
        std::uint32_t first_unit,
        std::uint32_t unit_count,
        std::span<const std::uint8_t> payload,
        std::string *error) noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::size_t bytes = layout.chunkBytes(unit_count);
        if (!canSubmitLocked(owner, false, error) || !layout.valid() ||
            layout.direction !=
                ExpertTierWeightConversionDirection::CpuToGpu ||
            !destination.validFor(layout) || unit_count == 0 ||
            first_unit >= layout.unit_count ||
            unit_count > layout.unit_count - first_unit ||
            unit_count > layout.maximum_units_per_chunk || bytes == 0 ||
            bytes != payload.size() || bytes > config_.staging.sizeBytes())
        {
            if (error && error->empty())
                *error = "Remote CPU-to-GPU repack chunk is outside its authenticated layout";
            return false;
        }

        /* The MPI buffer is persistent but not necessarily GPU-DMA pinned. */
        std::memcpy(pinned_chunk_, payload.data(), bytes);
        const bool copied = TransferEngine::instance().enqueueBackgroundStagingCopy(
            config_.execution, MappedTransferDirection::HostToDevice,
            device_chunk_, config_.staging.sizeBytes(), 0u,
            config_.staging, 0u, bytes, &failure_);
        const bool converted = copied && launchCpuToGpuLocked(
            layout, destination, first_unit, unit_count, bytes);
        return fenceSubmissionLocked(
            copied && converted,
            OperationKind::CpuToGpuRepack,
            bytes,
            error);
    }

    bool MoEOverlayGpuRemoteProjectionLane::submitGpuBlobWrite(
        const void *owner,
        std::uint8_t *destination,
        std::span<const std::uint8_t> payload,
        std::string *error) noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!canSubmitLocked(owner, false, error) || !destination ||
            payload.empty() ||
            payload.size() > config_.staging.sizeBytes())
        {
            if (error && error->empty())
                *error = "Remote GPU blob write has an invalid destination range";
            return false;
        }
        std::memcpy(pinned_chunk_, payload.data(), payload.size());
        if (config_.progress.epoch())
            return publishBlobLocked(OperationKind::GpuBlobWrite,
                destination, payload.size(), error);
        const bool copied = TransferEngine::instance().enqueueBackgroundStagingCopy(
            config_.execution, MappedTransferDirection::HostToDevice,
            destination, payload.size(), 0u,
            config_.staging, 0u, payload.size(), &failure_);
        return fenceSubmissionLocked(
            copied,
            OperationKind::GpuBlobWrite,
            payload.size(),
            error);
    }

    MappedTransferProgressSlot *MoEOverlayGpuRemoteProjectionLane::serviceCommandLocked() noexcept
    {
        if (!config_.progress.epoch())
            return nullptr;
        if (operation_kind_ == OperationKind::GpuBlobRead)
            return &service_read_;
        if (operation_kind_ == OperationKind::GpuBlobWrite)
            return &service_write_;
        return nullptr;
    }

    bool MoEOverlayGpuRemoteProjectionLane::publishBlobLocked(
        OperationKind kind, void *address, std::size_t bytes, std::string *error) noexcept
    {
        try
        {
            if (kind == OperationKind::GpuBlobRead)
                service_read_.publishDeviceToMappedHost(address, bytes, 0u, bytes, source_dependency_);
            else if (kind == OperationKind::GpuBlobWrite)
                service_write_.publishMappedHostToDevice(address, bytes, 0u, bytes);
            else
                throw std::logic_error("Raw graph-service publication requires a blob operation");

            // Acceptance retains the owner and staging even if a later runtime
            // submission fails. No native event may certify these copied bytes.
            operation_kind_ = kind;
            operation_bytes_ = bytes;
            progress_ = MoEOverlayGpuRemoteLaneProgress::Pending;
            work_may_be_in_flight_ = true;
            fail_after_event_ = false;
            progress_watch_.published(remoteProjectionNanoseconds());
            recordSubmissionLocked(kind, bytes);
            if (!config_.progress.epoch()->submitOutstandingProgress())
                throw std::runtime_error("Remote projection graph service could not progress accepted work");
            if (error)
                error->clear();
            return true;
        }
        catch (const std::exception &exception)
        {
            failure_ = exception.what();
            progress_ = MoEOverlayGpuRemoteLaneProgress::Failed;
            ++stats_.failures;
            setGpuRemoteError(error, failure_);
            return false;
        }
    }

    MoEOverlayGpuRemoteLaneProgress
    MoEOverlayGpuRemoteProjectionLane::poll(
        const void *owner,
        std::string *error) noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!owner || owner_ != owner)
        {
            setGpuRemoteError(
                error,
                "Remote ExpertOverlay GPU lane poll has the wrong owner");
            return MoEOverlayGpuRemoteLaneProgress::Failed;
        }
        if (progress_ != MoEOverlayGpuRemoteLaneProgress::Pending)
        {
            if (progress_ == MoEOverlayGpuRemoteLaneProgress::Failed)
                setGpuRemoteError(error, failure_);
            else if (error)
                error->clear();
            return progress_;
        }

        bool ready = false;
        try
        {
            if (auto *command = serviceCommandLocked())
            {
                if (!config_.progress.epoch()->submitOutstandingProgress())
                    throw std::runtime_error("Remote projection graph progress failed");
                const auto result = command->poll(&failure_);
                if (result == MappedTransferProgress::Failed)
                {
                    work_may_be_in_flight_ = command->pending();
                    throw std::runtime_error(failure_);
                }
                ready = result == MappedTransferProgress::Ready;
            }
            else if (!context_->queryEventChecked(completion_event_, ready))
            {
                failure_ =
                    "Remote ExpertOverlay GPU lane completion-event query failed";
                progress_ = MoEOverlayGpuRemoteLaneProgress::Failed;
                unfenced_work_ = true;
                ++stats_.failures;
                setGpuRemoteError(error, failure_);
                return progress_;
            }
        }
        catch (const std::exception &exception)
        {
            failure_ = exception.what();
            progress_ = MoEOverlayGpuRemoteLaneProgress::Failed;
            unfenced_work_ = true;
            ++stats_.failures;
            setGpuRemoteError(error, failure_);
            return progress_;
        }
        catch (...)
        {
            failure_ =
                "Remote ExpertOverlay GPU event query threw a non-standard exception";
            progress_ = MoEOverlayGpuRemoteLaneProgress::Failed;
            unfenced_work_ = true;
            ++stats_.failures;
            setGpuRemoteError(error, failure_);
            return progress_;
        }
        if (!ready)
        {
            ++stats_.pending_event_polls;
            // Reuse the exact not-ready observation above, never query again
            // for logging. Each chunk resets the watch, so a healthy busy lane
            // cannot be confused with one stalled command.
            const auto now = remoteProjectionNanoseconds();
            progress_watch_.observedIncompleteEvent(now);
            if (const auto age = progress_watch_.pending(now))
            {
                LOG_WARN("[MoEOverlayGpuRemoteProjectionLane] chunk event remains pending"
                         << " device=" << config_.device.toString()
                         << " lane=" << config_.lane_name
                         << " operation=" << operationName(operation_kind_)
                         << " bytes=" << operation_bytes_
                         << " command_pending_ns=" << *age
                         << " native_not_ready_queries=" << progress_watch_.incompleteEventQueries()
                         << " chunks_submitted=" << stats_.chunks_submitted
                         << " chunks_completed=" << stats_.chunks_completed
                         << " stream=" << stream_);
            }
            if (error)
                error->clear();
            return progress_;
        }

        work_may_be_in_flight_ = false;
        if (fail_after_event_)
        {
            fail_after_event_ = false;
            progress_ = MoEOverlayGpuRemoteLaneProgress::Failed;
            ++stats_.failures;
            setGpuRemoteError(error, failure_);
            return progress_;
        }
        progress_ = MoEOverlayGpuRemoteLaneProgress::Ready;
        ++stats_.chunks_completed;
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "remote_gpu_chunks_completed",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"lane", config_.lane_name},
             {"device", config_.device.to_string()},
             {"operation", operationName(operation_kind_)},
             {"background", "true"},
             {"blocking", "false"}});
        if (error)
            error->clear();
        return progress_;
    }

    std::span<const std::uint8_t>
    MoEOverlayGpuRemoteProjectionLane::pinnedOutput(
        const void *owner,
        std::size_t bytes) const noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool is_read =
            operation_kind_ == OperationKind::GpuToCpuRepack ||
            operation_kind_ == OperationKind::GpuBlobRead;
        if (!owner || owner_ != owner ||
            progress_ != MoEOverlayGpuRemoteLaneProgress::Ready || !is_read ||
            bytes == 0 || bytes != operation_bytes_ ||
            bytes > config_.staging.sizeBytes())
        {
            return {};
        }
        return {pinned_chunk_, bytes};
    }

    MoEOverlayGpuRemoteProjectionLaneStats
    MoEOverlayGpuRemoteProjectionLane::stats() const noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    const char *MoEOverlayGpuRemoteProjectionLane::operationName(
        OperationKind kind) noexcept
    {
        switch (kind)
        {
        case OperationKind::GpuToCpuRepack:
            return "gpu_to_cpu_repack";
        case OperationKind::CpuToGpuRepack:
            return "cpu_to_gpu_repack";
        case OperationKind::GpuBlobRead:
            return "gpu_blob_read";
        case OperationKind::GpuBlobWrite:
            return "gpu_blob_write";
        case OperationKind::None:
            break;
        }
        return "none";
    }

    void MoEOverlayGpuRemoteProjectionLane::recordSubmissionLocked(
        OperationKind kind,
        std::size_t bytes) noexcept
    {
        ++stats_.chunks_submitted;
        stats_.bytes_submitted += bytes;
        switch (kind)
        {
        case OperationKind::GpuToCpuRepack:
            ++stats_.gpu_to_cpu_repack_chunks;
            ++stats_.device_to_host_submissions;
            break;
        case OperationKind::CpuToGpuRepack:
            ++stats_.cpu_to_gpu_repack_chunks;
            ++stats_.host_to_device_submissions;
            break;
        case OperationKind::GpuBlobRead:
            ++stats_.gpu_blob_read_chunks;
            ++stats_.device_to_host_submissions;
            break;
        case OperationKind::GpuBlobWrite:
            ++stats_.gpu_blob_write_chunks;
            ++stats_.host_to_device_submissions;
            break;
        case OperationKind::None:
            break;
        }
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "remote_gpu_chunk_bytes_submitted",
            static_cast<double>(bytes),
            "maintenance",
            config_.perf_device,
            {{"lane", config_.lane_name},
             {"device", config_.device.to_string()},
             {"operation", operationName(kind)},
             {"background", "true"},
             {"blocking", "false"}});
    }

    void MoEOverlayGpuRemoteProjectionLane::releaseResources() noexcept
    {
        if (backend_ && device_ordinal_ >= 0)
        {
            if (completion_event_)
                backend_->destroyEvent(completion_event_, device_ordinal_);
        }
        backend_ = nullptr;
        context_ = nullptr;
        device_ordinal_ = -1;
        stream_ = nullptr;
        completion_event_ = nullptr;
        device_chunk_ = nullptr;
        pinned_chunk_ = nullptr;
    }

    MoEOverlayGpuRemoteProjectionSource::
        MoEOverlayGpuRemoteProjectionSource(
            MoEOverlayRemoteProjectionManifest manifest,
            std::shared_ptr<MoEOverlayGpuRemoteProjectionLane> lane,
            GpuExpertPackedDescriptor source,
            ExpertTierSourceReadiness readiness,
            std::shared_ptr<void> lifetime)
        : manifest_(std::move(manifest)),
          lane_(std::move(lane)),
          source_(source),
          readiness_(readiness),
          lifetime_(std::move(lifetime))
    {
        std::string error;
        if (!manifest_.valid(&error) || !lane_ || !lane_->materialized() ||
            !descriptorPointersComplete(source_) || !lifetime_ ||
            manifest_.identity.source_device != lane_->device() ||
            !manifest_.identity.source_device.is_gpu())
        {
            throw std::invalid_argument(
                error.empty()
                    ? "Remote ExpertOverlay GPU source has incomplete topology, storage, or ownership"
                    : std::move(error));
        }
        if (readiness_.kind() ==
                ExpertTierSourceReadiness::Kind::
                    PublishedQuiescentResidencyBank &&
            readiness_.epoch() != manifest_.identity.expected_epoch)
        {
            throw std::invalid_argument(
                "Remote ExpertOverlay GPU source bank epoch differs from its transaction");
        }

        if (manifest_.carriesCpuBytes())
        {
            cpu_layout_ = remoteCpuProjectionDeviceLayout(manifest_);
            if (cpu_layout_->direction !=
                    ExpertTierWeightConversionDirection::GpuToCpu ||
                !gpuConstView(source_).validFor(*cpu_layout_))
            {
                throw std::invalid_argument(
                    "Remote GPU-to-CPU source descriptor differs from its repack manifest");
            }
        }
        else if (manifest_.carriesGpuBytes())
        {
            if (!manifest_.identity.destination_device.is_gpu() ||
                manifest_.N != source_.n || manifest_.K != source_.k ||
                manifest_.blocks_per_row !=
                    static_cast<std::int32_t>(source_.blocks_per_row) ||
                manifest_.gpu_codebook_id != source_.codebook_id ||
                manifest_.gpu_payload_bytes_per_block !=
                    source_.payload_bytes_per_block ||
                manifest_.gpu_is_asymmetric != source_.is_asymmetric ||
                manifest_.gpu_has_emins != source_.has_emins ||
                manifest_.region_bytes !=
                    std::array<std::uint64_t, 4>{
                        source_.vnni_bytes,
                        source_.scales_bytes,
                        source_.mins_bytes,
                        source_.emins_bytes})
            {
                throw std::invalid_argument(
                    "Remote GPU blob source descriptor differs from its manifest");
            }
            seekNextGpuRegion();
        }
        else
        {
            throw std::invalid_argument(
                "Remote ExpertOverlay GPU source uses an unknown packing");
        }
    }

    MoEOverlayGpuRemoteProjectionSource::
        MoEOverlayGpuRemoteProjectionSource(
            MoEOverlayRemoteProjectionManifest manifest,
            std::shared_ptr<MoEOverlayGpuRemoteProjectionLane> lane,
            ContiguousFloatingPointWeightDescriptor source,
            ExpertTierSourceReadiness readiness,
            std::shared_ptr<void> lifetime)
        : manifest_(std::move(manifest)),
          lane_(std::move(lane)),
          floating_source_(source),
          readiness_(readiness),
          lifetime_(std::move(lifetime))
    {
        std::string error;
        const auto format = ExpertWeightFormat::floating(source.type);
        if (!manifest_.valid(&error) || !lane_ || !lane_->materialized() ||
            !source.valid() || !format.valid() || !lifetime_ ||
            manifest_.identity.source_device != lane_->device() ||
            !manifest_.identity.source_device.is_gpu() ||
            !manifest_.carriesFloatingBytes() ||
            manifest_.format_kind != format.kind ||
            manifest_.N != source.n || manifest_.K != source.k ||
            manifest_.region_bytes !=
                std::array<std::uint64_t, 4>{
                    source.bytes, 0u, 0u, 0u})
        {
            throw std::invalid_argument(
                error.empty()
                    ? "Remote ExpertOverlay floating GPU source has incomplete or mismatched topology, storage, or precision"
                    : std::move(error));
        }
        if (readiness_.kind() ==
                ExpertTierSourceReadiness::Kind::
                    PublishedQuiescentResidencyBank &&
            readiness_.epoch() != manifest_.identity.expected_epoch)
        {
            throw std::invalid_argument(
                "Remote ExpertOverlay floating GPU source bank epoch differs from its transaction");
        }
        seekNextGpuRegion();
    }

    MoEOverlayGpuRemoteProjectionSource::
        ~MoEOverlayGpuRemoteProjectionSource()
    {
        if (lane_owned_)
        {
            LOG_ERROR(
                "[MoEOverlayGpuRemoteProjectionSource] Destroyed while retaining a GPU lane"
                << " layer=" << manifest_.identity.layer_idx
                << " expert=" << manifest_.identity.expert_id);
            std::terminate();
        }
    }

    void MoEOverlayGpuRemoteProjectionSource::seekNextGpuRegion() noexcept
    {
        while (region_ < manifest_.region_bytes.size() &&
               region_offset_ == manifest_.region_bytes[region_])
        {
            ++region_;
            region_offset_ = 0;
        }
    }

    const std::uint8_t *
    MoEOverlayGpuRemoteProjectionSource::currentGpuRegionPointer()
        const noexcept
    {
        if (floating_source_.valid())
        {
            return region_ == 0u
                ? static_cast<const std::uint8_t *>(
                      floating_source_.data) +
                      region_offset_
                : nullptr;
        }
        const std::uint8_t *base = nullptr;
        switch (region_)
        {
        case 0:
            base = source_.ptrs.d_vnni;
            break;
        case 1:
            base = static_cast<const std::uint8_t *>(
                source_.ptrs.d_scales);
            break;
        case 2:
            base = static_cast<const std::uint8_t *>(source_.ptrs.d_mins);
            break;
        case 3:
            base = static_cast<const std::uint8_t *>(source_.ptrs.d_emins);
            break;
        default:
            return nullptr;
        }
        return base ? base + region_offset_ : nullptr;
    }

    bool MoEOverlayGpuRemoteProjectionSource::submitNextChunk(
        std::string *error) noexcept
    {
        if (!lane_owned_ || chunk_submitted_ || payload_outstanding_ ||
            complete_ || aborted_ ||
            !lane_->bindSourceReadiness(this, readiness_, error))
        {
            if (error && error->empty())
                *error = "Remote ExpertOverlay GPU source cannot submit its next chunk";
            return false;
        }

        active_header_ = {
            .manifest_hash = manifest_.manifest_hash,
            .sequence = sequence_,
        };
        bool submitted = false;
        if (cpu_layout_)
        {
            const auto first_unit = static_cast<std::uint32_t>(
                emitted_bytes_ / cpu_layout_->cpu_block_stride);
            const auto remaining = cpu_layout_->unit_count - first_unit;
            const auto units = std::min(
                cpu_layout_->maximum_units_per_chunk, remaining);
            const auto bytes = cpu_layout_->chunkBytes(units);
            active_header_.region = 0;
            active_header_.region_offset = emitted_bytes_;
            active_header_.payload_bytes =
                static_cast<std::uint32_t>(bytes);
            active_header_.final_chunk = static_cast<std::uint8_t>(
                emitted_bytes_ + bytes == manifest_.total_bytes);
            submitted = lane_->submitGpuToCpuRepack(
                this,
                *cpu_layout_,
                gpuConstView(source_),
                first_unit,
                units,
                error);
        }
        else
        {
            seekNextGpuRegion();
            if (region_ >= manifest_.region_bytes.size())
            {
                setGpuRemoteError(
                    error,
                    "Remote GPU blob source exhausted before its final marker");
                return false;
            }
            const std::uint64_t remaining =
                manifest_.region_bytes[region_] - region_offset_;
            const std::uint64_t bytes = std::min<std::uint64_t>(
                remaining, manifest_.maximum_chunk_bytes);
            active_header_.region = static_cast<std::uint8_t>(region_);
            active_header_.region_offset = region_offset_;
            active_header_.payload_bytes =
                static_cast<std::uint32_t>(bytes);
            active_header_.final_chunk = static_cast<std::uint8_t>(
                emitted_bytes_ + bytes == manifest_.total_bytes);
            submitted = lane_->submitGpuBlobRead(
                this,
                currentGpuRegionPointer(),
                static_cast<std::size_t>(bytes),
                error);
        }
        chunk_submitted_ = submitted;
        return submitted;
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayGpuRemoteProjectionSource::pollNextChunk(
        MoEOverlayRemoteProjectionChunkView *chunk,
        std::string *error) noexcept
    {
        if (!chunk || aborted_ || complete_ || payload_outstanding_)
        {
            setGpuRemoteError(
                error,
                !chunk
                    ? "Remote ExpertOverlay GPU source requires a chunk destination"
                    : aborted_
                          ? "Remote ExpertOverlay GPU source is aborted"
                          : complete_
                                ? "Remote ExpertOverlay GPU source was polled after completion"
                                : "Remote ExpertOverlay GPU source payload is still owned by MPI");
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        if (!lane_owned_)
        {
            lane_owned_ = lane_->tryAcquire(this);
            if (!lane_owned_)
            {
                if (error)
                    error->clear();
                return MoEOverlayResidencyWaveProgress::Pending;
            }
        }
        if (!chunk_submitted_)
        {
            if (!submitNextChunk(error))
                return MoEOverlayResidencyWaveProgress::Failed;
            return MoEOverlayResidencyWaveProgress::Pending;
        }

        const auto progress = lane_->poll(this, error);
        if (progress == MoEOverlayGpuRemoteLaneProgress::Pending)
            return MoEOverlayResidencyWaveProgress::Pending;
        if (progress != MoEOverlayGpuRemoteLaneProgress::Ready)
            return MoEOverlayResidencyWaveProgress::Failed;

        const auto payload = lane_->pinnedOutput(
            this, active_header_.payload_bytes);
        if (payload.size() != active_header_.payload_bytes)
        {
            setGpuRemoteError(
                error,
                "Remote ExpertOverlay GPU source lost its event-ready pinned payload");
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        active_header_.payload_hash = expertTierWeightBytesHash(payload);
        *chunk = {
            .header = active_header_,
            .payload = payload,
        };
        payload_outstanding_ = true;
        if (error)
            error->clear();
        return MoEOverlayResidencyWaveProgress::Ready;
    }

    bool MoEOverlayGpuRemoteProjectionSource::acknowledgeChunkSent(
        const MoEOverlayRemoteProjectionChunkHeader &header,
        std::string *error) noexcept
    {
        if (aborted_ || !payload_outstanding_ || !chunk_submitted_ ||
            !sameChunkHeader(header, active_header_))
        {
            setGpuRemoteError(
                error,
                aborted_
                    ? "Remote ExpertOverlay GPU source is aborted"
                    : "Remote ExpertOverlay GPU source received an unexpected MPI acknowledgement");
            return false;
        }

        const bool final = active_header_.final_chunk != 0;
        emitted_bytes_ += active_header_.payload_bytes;
        if (!cpu_layout_)
        {
            region_offset_ += active_header_.payload_bytes;
            seekNextGpuRegion();
        }
        ++sequence_;
        payload_outstanding_ = false;
        chunk_submitted_ = false;
        active_header_ = {};
        if (!releaseLane(error))
            return false;
        complete_ = final;
        if (complete_ && emitted_bytes_ != manifest_.total_bytes)
        {
            setGpuRemoteError(
                error,
                "Remote ExpertOverlay GPU source final acknowledgement has an incomplete byte count");
            return false;
        }
        if (error)
            error->clear();
        return true;
    }

    bool MoEOverlayGpuRemoteProjectionSource::releaseLane(
        std::string *error) noexcept
    {
        if (!lane_owned_)
            return true;
        if (!lane_->release(this, error))
            return false;
        lane_owned_ = false;
        return true;
    }

    void MoEOverlayGpuRemoteProjectionSource::abort() noexcept
    {
        aborted_ = true;
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayGpuRemoteProjectionSource::pollAbort(
        std::string *error) noexcept
    {
        if (!aborted_)
        {
            setGpuRemoteError(
                error,
                "Remote ExpertOverlay GPU source abort cleanup was polled before abort");
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        if (!lane_owned_)
        {
            if (error)
                error->clear();
            return MoEOverlayResidencyWaveProgress::Ready;
        }
        const auto progress = lane_->poll(this, error);
        if (progress == MoEOverlayGpuRemoteLaneProgress::Pending)
            return MoEOverlayResidencyWaveProgress::Pending;
        if (!releaseLane(error))
            return MoEOverlayResidencyWaveProgress::Failed;
        payload_outstanding_ = false;
        chunk_submitted_ = false;
        if (error)
            error->clear();
        return MoEOverlayResidencyWaveProgress::Ready;
    }

    MoEOverlayGpuRemoteProjectionDestination::
        MoEOverlayGpuRemoteProjectionDestination(
            MoEOverlayRemoteProjectionIdentity expected_identity,
            std::shared_ptr<MoEOverlayGpuRemoteProjectionLane> lane,
            MoEOverlayGpuRemoteProjectionDestinationFactory factory,
            std::shared_ptr<void> lifetime)
        : expected_identity_(expected_identity),
          lane_(std::move(lane)),
          factory_(std::move(factory)),
          lifetime_(std::move(lifetime))
    {
        if (!expected_identity_.valid() ||
            !expected_identity_.destination_device.is_gpu() || !lane_ ||
            !lane_->materialized() ||
            lane_->device() != expected_identity_.destination_device ||
            !factory_ || !lifetime_)
        {
            throw std::invalid_argument(
                "Remote ExpertOverlay GPU destination requires complete topology, lane, factory, and slot ownership");
        }
    }

    MoEOverlayGpuRemoteProjectionDestination::
        ~MoEOverlayGpuRemoteProjectionDestination()
    {
        if (lane_owned_)
        {
            LOG_ERROR(
                "[MoEOverlayGpuRemoteProjectionDestination] Destroyed while retaining a GPU lane"
                << " layer=" << expected_identity_.layer_idx
                << " expert=" << expected_identity_.expert_id);
            std::terminate();
        }
    }

    bool MoEOverlayGpuRemoteProjectionDestination::beginManifest(
        const MoEOverlayRemoteProjectionManifest &manifest,
        std::string *error) noexcept
    {
        if (aborted_ || manifest_ || !manifest.valid(error) ||
            manifest.identity != expected_identity_ ||
            manifest.maximum_chunk_bytes > lane_->stagingCapacityBytes())
        {
            if (aborted_)
                setGpuRemoteError(
                    error,
                    "Remote ExpertOverlay GPU destination is aborted");
            else if (manifest_)
                setGpuRemoteError(
                    error,
                    "Remote ExpertOverlay GPU destination already accepted a manifest");
            else if (manifest.identity != expected_identity_)
                setGpuRemoteError(
                    error,
                    "Remote ExpertOverlay GPU destination received an unexpected identity");
            else if (manifest.maximum_chunk_bytes >
                     lane_->stagingCapacityBytes())
                setGpuRemoteError(
                    error,
                    "Remote ExpertOverlay GPU manifest exceeds its persistent lane capacity");
            return false;
        }

        MoEOverlayGpuRemoteProjectionDestinationBinding candidate;
        try
        {
            if (!factory_(manifest, &candidate, error) ||
                !candidate.valid() ||
                (candidate.descriptor.valid() &&
                 !descriptorPointersComplete(candidate.descriptor)))
            {
                if (error && error->empty())
                    *error = "Remote ExpertOverlay GPU destination factory returned incomplete storage";
                return false;
            }
        }
        catch (const std::exception &exception)
        {
            setGpuRemoteError(error, exception.what());
            return false;
        }
        catch (...)
        {
            setGpuRemoteError(
                error,
                "Remote ExpertOverlay GPU destination factory threw a non-standard exception");
            return false;
        }

        std::optional<ExpertTierWeightDeviceLayout> candidate_cpu_layout;
        if (manifest.carriesCpuBytes())
        {
            try
            {
                candidate_cpu_layout = remoteCpuProjectionDeviceLayout(manifest);
            }
            catch (const std::exception &exception)
            {
                setGpuRemoteError(error, exception.what());
                return false;
            }
            if (!candidate.descriptor.valid() ||
                candidate_cpu_layout->direction !=
                    ExpertTierWeightConversionDirection::CpuToGpu ||
                !gpuMutableView(candidate.descriptor)
                     .validFor(*candidate_cpu_layout))
            {
                setGpuRemoteError(
                    error,
                    "Remote CPU-to-GPU destination descriptor differs from its repack manifest");
                return false;
            }
        }
        else if (manifest.carriesGpuBytes())
        {
            const auto &descriptor = candidate.descriptor;
            if (!descriptor.valid() ||
                !manifest.identity.source_device.is_gpu() ||
                manifest.N != descriptor.n || manifest.K != descriptor.k ||
                manifest.blocks_per_row !=
                    static_cast<std::int32_t>(descriptor.blocks_per_row) ||
                manifest.gpu_codebook_id != descriptor.codebook_id ||
                manifest.gpu_payload_bytes_per_block !=
                    descriptor.payload_bytes_per_block ||
                manifest.gpu_is_asymmetric != descriptor.is_asymmetric ||
                manifest.gpu_has_emins != descriptor.has_emins ||
                manifest.region_bytes !=
                    std::array<std::uint64_t, 4>{
                        descriptor.vnni_bytes,
                        descriptor.scales_bytes,
                        descriptor.mins_bytes,
                        descriptor.emins_bytes})
            {
                setGpuRemoteError(
                    error,
                    "Remote GPU blob destination descriptor differs from its manifest");
                return false;
            }
        }
        else if (manifest.carriesFloatingBytes())
        {
            const auto &descriptor = candidate.floating_descriptor;
            const auto format = ExpertWeightFormat::floating(
                descriptor.type);
            if (!descriptor.valid() || !format.valid() ||
                manifest.format_kind != format.kind ||
                manifest.N != descriptor.n ||
                manifest.K != descriptor.k ||
                manifest.region_bytes !=
                    std::array<std::uint64_t, 4>{
                        descriptor.bytes, 0u, 0u, 0u})
            {
                setGpuRemoteError(
                    error,
                    "Remote floating GPU blob destination descriptor differs from its manifest");
                return false;
            }
        }
        else
        {
            setGpuRemoteError(
                error,
                "Remote ExpertOverlay GPU destination received unknown packing");
            return false;
        }

        try
        {
            validator_.emplace(manifest);
        }
        catch (const std::exception &exception)
        {
            setGpuRemoteError(error, exception.what());
            return false;
        }
        manifest_ = manifest;
        cpu_layout_ = candidate_cpu_layout;
        binding_ = std::move(candidate);
        if (error)
            error->clear();
        return true;
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayGpuRemoteProjectionDestination::beginChunk(
        const MoEOverlayRemoteProjectionChunkHeader &header,
        std::span<const std::uint8_t> payload,
        std::string *error) noexcept
    {
        if (aborted_ || !manifest_ || !validator_ || chunk_active_ ||
            !validator_->accept(header, payload, error))
        {
            if (aborted_)
                setGpuRemoteError(
                    error,
                    "Remote ExpertOverlay GPU destination is aborted");
            else if (!manifest_ || !validator_)
                setGpuRemoteError(
                    error,
                    "Remote ExpertOverlay GPU destination has no accepted manifest");
            else if (chunk_active_)
                setGpuRemoteError(
                    error,
                    "Remote ExpertOverlay GPU destination already owns an MPI chunk");
            return MoEOverlayResidencyWaveProgress::Failed;
        }

        bool range_valid = false;
        if (cpu_layout_)
        {
            range_valid = header.region == 0 &&
                          (header.region_offset %
                           cpu_layout_->cpu_block_stride) == 0 &&
                          (payload.size() %
                           cpu_layout_->cpu_block_stride) == 0;
        }
        else
        {
            range_valid = gpuRegionPointer(header.region) != nullptr &&
                          header.region_offset <=
                              manifest_->region_bytes[header.region] &&
                          payload.size() <=
                              manifest_->region_bytes[header.region] -
                                  header.region_offset;
        }
        if (!range_valid)
        {
            validator_->abort();
            setGpuRemoteError(
                error,
                "Remote ExpertOverlay GPU chunk does not name a complete repack unit or bounded blob range");
            return MoEOverlayResidencyWaveProgress::Failed;
        }

        active_header_ = header;
        active_payload_ = payload;
        chunk_active_ = true;
        chunk_submitted_ = false;
        return startPendingChunk(error);
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayGpuRemoteProjectionDestination::startPendingChunk(
        std::string *error) noexcept
    {
        if (!chunk_active_ || chunk_submitted_ || aborted_)
        {
            setGpuRemoteError(
                error,
                "Remote ExpertOverlay GPU destination has no submit-ready chunk");
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        if (!lane_owned_)
        {
            lane_owned_ = lane_->tryAcquire(this);
            if (!lane_owned_)
            {
                if (error)
                    error->clear();
                return MoEOverlayResidencyWaveProgress::Pending;
            }
        }

        bool submitted = false;
        if (cpu_layout_)
        {
            const auto first_unit = static_cast<std::uint32_t>(
                active_header_.region_offset /
                cpu_layout_->cpu_block_stride);
            const auto units = static_cast<std::uint32_t>(
                active_payload_.size() /
                cpu_layout_->cpu_block_stride);
            submitted = lane_->submitCpuToGpuRepack(
                this,
                *cpu_layout_,
                gpuMutableView(binding_.descriptor),
                first_unit,
                units,
                active_payload_,
                error);
        }
        else
        {
            auto *base = gpuRegionPointer(active_header_.region);
            submitted = lane_->submitGpuBlobWrite(
                this,
                base + active_header_.region_offset,
                active_payload_,
                error);
        }
        chunk_submitted_ = submitted;
        return submitted ? MoEOverlayResidencyWaveProgress::Pending
                         : MoEOverlayResidencyWaveProgress::Failed;
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayGpuRemoteProjectionDestination::pollChunk(
        std::string *error) noexcept
    {
        if (aborted_ || !chunk_active_)
        {
            setGpuRemoteError(
                error,
                aborted_
                    ? "Remote ExpertOverlay GPU destination is aborted"
                    : "Remote ExpertOverlay GPU destination has no active chunk");
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        if (!chunk_submitted_)
            return startPendingChunk(error);

        const auto progress = lane_->poll(this, error);
        if (progress == MoEOverlayGpuRemoteLaneProgress::Pending)
            return MoEOverlayResidencyWaveProgress::Pending;
        if (progress != MoEOverlayGpuRemoteLaneProgress::Ready)
            return MoEOverlayResidencyWaveProgress::Failed;

        const bool final = active_header_.final_chunk != 0;
        active_header_ = {};
        active_payload_ = {};
        chunk_active_ = false;
        chunk_submitted_ = false;
        if (!releaseLane(error))
            return MoEOverlayResidencyWaveProgress::Failed;
        if (final)
            final_chunk_committed_ = true;
        if (error)
            error->clear();
        return MoEOverlayResidencyWaveProgress::Ready;
    }

    std::uint8_t *
    MoEOverlayGpuRemoteProjectionDestination::gpuRegionPointer(
        std::uint8_t region) const noexcept
    {
        if (binding_.floating_descriptor.valid())
        {
            return region == 0u
                ? static_cast<std::uint8_t *>(const_cast<void *>(
                      binding_.floating_descriptor.data))
                : nullptr;
        }
        switch (region)
        {
        case 0:
            return binding_.descriptor.ptrs.d_vnni;
        case 1:
            return static_cast<std::uint8_t *>(
                binding_.descriptor.ptrs.d_scales);
        case 2:
            return static_cast<std::uint8_t *>(
                binding_.descriptor.ptrs.d_mins);
        case 3:
            return static_cast<std::uint8_t *>(
                binding_.descriptor.ptrs.d_emins);
        default:
            return nullptr;
        }
    }

    bool MoEOverlayGpuRemoteProjectionDestination::releaseLane(
        std::string *error) noexcept
    {
        if (!lane_owned_)
            return true;
        if (!lane_->release(this, error))
            return false;
        lane_owned_ = false;
        return true;
    }

    bool MoEOverlayGpuRemoteProjectionDestination::complete() const noexcept
    {
        return validator_ && validator_->complete() &&
               final_chunk_committed_ && !chunk_active_ && !lane_owned_;
    }

    bool MoEOverlayGpuRemoteProjectionDestination::publishFinal(
        std::string *error) noexcept
    {
        if (!complete() || !binding_.engine)
        {
            setGpuRemoteError(
                error,
                "Remote ExpertOverlay GPU destination cannot publish before final event readiness");
            return false;
        }
        if (error)
            error->clear();
        return true;
    }

    void MoEOverlayGpuRemoteProjectionDestination::abort() noexcept
    {
        aborted_ = true;
        if (validator_)
            validator_->abort();
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayGpuRemoteProjectionDestination::pollAbort(
        std::string *error) noexcept
    {
        if (!aborted_)
        {
            setGpuRemoteError(
                error,
                "Remote ExpertOverlay GPU destination abort cleanup was polled before abort");
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        if (!lane_owned_)
        {
            active_payload_ = {};
            chunk_active_ = false;
            chunk_submitted_ = false;
            if (error)
                error->clear();
            return MoEOverlayResidencyWaveProgress::Ready;
        }
        const auto progress = lane_->poll(this, error);
        if (progress == MoEOverlayGpuRemoteLaneProgress::Pending)
            return MoEOverlayResidencyWaveProgress::Pending;
        if (!releaseLane(error))
            return MoEOverlayResidencyWaveProgress::Failed;
        active_payload_ = {};
        chunk_active_ = false;
        chunk_submitted_ = false;
        if (error)
            error->clear();
        return MoEOverlayResidencyWaveProgress::Ready;
    }
} // namespace llaminar2
