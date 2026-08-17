/**
 * @file ExpertTierGpuBlobTransferLane.cpp
 * @brief Event-polled CUDA/ROCm packed expert blob relay implementation.
 *
 * Device work is split into source-runtime and destination-runtime phases. A
 * host event query is the only portable ownership handoff between CUDA and HIP;
 * cross-runtime event waits are deliberately forbidden. Two persistent slots
 * allow the source to prepare a later chunk while the destination consumes an
 * earlier one, without allocating, synchronizing, or rebinding inference data.
 */

#include "ExpertTierGpuBlobTransferLane.h"

#include "../../backends/BackendManager.h"
#include "../../backends/GPUDeviceContextPool.h"
#include "../../backends/IBackend.h"
#include "../../backends/IWorkerGPUContext.h"
#include "../../utils/Logger.h"
#include "../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @brief Store a diagnostic only when the caller requested one. */
        void assignBlobTransferError(
            std::string *error,
            const std::string &message)
        {
            if (error)
                *error = message;
        }

        /** @brief Validate optional region pointers before any DMA is submitted. */
        bool descriptorPointersCoverRegions(
            const GpuExpertPackedDescriptor &descriptor) noexcept
        {
            return descriptor.ptrs.d_vnni != nullptr &&
                   descriptor.ptrs.d_scales != nullptr &&
                   (descriptor.mins_bytes == 0 ||
                    descriptor.ptrs.d_mins != nullptr) &&
                   (descriptor.emins_bytes == 0 ||
                    descriptor.ptrs.d_emins != nullptr);
        }

        /** @brief Convert a scoped region enum to an array index. */
        constexpr std::size_t regionIndex(
            ExpertTierGpuBlobRegion region) noexcept
        {
            return static_cast<std::size_t>(region);
        }
    } // namespace

    ExpertTierGpuBlobChunkProtocol::ExpertTierGpuBlobChunkProtocol(
        std::size_t chunk_capacity_bytes)
        : chunk_capacity_bytes_(chunk_capacity_bytes)
    {
        if (chunk_capacity_bytes_ == 0)
            throw std::invalid_argument(
                "GPU blob chunk protocol requires non-zero staging capacity");
    }

    bool ExpertTierGpuBlobChunkProtocol::begin(
        const GpuExpertPackedDescriptor &source,
        const GpuExpertPackedDescriptor &destination,
        std::string *error) noexcept
    {
        begun_ = false;
        region_bytes_.fill(0);
        region_index_ = 0;
        region_offset_ = 0;
        total_bytes_ = 0;
        submitted_bytes_ = 0;
        next_sequence_ = 0;

        if (!gpuExpertPackedDescriptorsCompatible(source, destination) ||
            !descriptorPointersCoverRegions(source) ||
            !descriptorPointersCoverRegions(destination))
        {
            assignBlobTransferError(
                error,
                "GPU blob transfer requires byte-compatible complete descriptors");
            return false;
        }

        region_bytes_ = {
            source.vnni_bytes,
            source.scales_bytes,
            source.mins_bytes,
            source.emins_bytes,
        };
        total_bytes_ = std::accumulate(
            region_bytes_.begin(),
            region_bytes_.end(),
            std::size_t{0});
        if (total_bytes_ == 0)
        {
            assignBlobTransferError(error, "GPU blob transfer is empty");
            return false;
        }

        begun_ = true;
        seekNextNonEmptyRegion();
        return true;
    }

    bool ExpertTierGpuBlobChunkProtocol::beginContiguous(
        std::size_t bytes,
        std::string *error) noexcept
    {
        begun_ = false;
        region_bytes_.fill(0);
        region_index_ = 0;
        region_offset_ = 0;
        total_bytes_ = 0;
        submitted_bytes_ = 0;
        next_sequence_ = 0;
        if (bytes == 0)
        {
            assignBlobTransferError(
                error,
                "GPU blob contiguous transfer requires non-zero bytes");
            return false;
        }
        region_bytes_[regionIndex(ExpertTierGpuBlobRegion::Payload)] = bytes;
        total_bytes_ = bytes;
        begun_ = true;
        return true;
    }

    void ExpertTierGpuBlobChunkProtocol::seekNextNonEmptyRegion() noexcept
    {
        while (region_index_ < region_bytes_.size() &&
               region_offset_ == region_bytes_[region_index_])
        {
            ++region_index_;
            region_offset_ = 0;
        }
    }

    std::optional<ExpertTierGpuBlobChunk>
    ExpertTierGpuBlobChunkProtocol::takeNext() noexcept
    {
        if (exhausted())
            return std::nullopt;

        const std::size_t remaining =
            region_bytes_[region_index_] - region_offset_;
        const std::size_t bytes =
            std::min(chunk_capacity_bytes_, remaining);
        ExpertTierGpuBlobChunk chunk{
            .region = static_cast<ExpertTierGpuBlobRegion>(region_index_),
            .region_offset = region_offset_,
            .bytes = bytes,
            .sequence = next_sequence_++,
        };
        region_offset_ += bytes;
        submitted_bytes_ += bytes;
        seekNextNonEmptyRegion();
        return chunk;
    }

    bool ExpertTierGpuBlobChunkProtocol::exhausted() const noexcept
    {
        return begun_ && region_index_ == region_bytes_.size();
    }

    ExpertTierGpuBlobTransferLane::ExpertTierGpuBlobTransferLane(Config config)
        : config_(std::move(config)),
          protocol_(config_.staging_capacity_bytes)
    {
        if (!config_.source_device.is_gpu() ||
            !config_.destination_device.is_gpu())
        {
            throw std::invalid_argument(
                "Heterogeneous GPU blob lane requires two GPU endpoints");
        }
        if (config_.source_device.type == config_.destination_device.type)
        {
            throw std::invalid_argument(
                "Heterogeneous GPU blob lane requires different backend types");
        }
        if (config_.lane_name.empty())
            throw std::invalid_argument(
                "Heterogeneous GPU blob lane requires a stable non-empty name");
        if (config_.perf_device.empty())
        {
            config_.perf_device =
                config_.source_device.to_string() + "->" +
                config_.destination_device.to_string();
        }
    }

    ExpertTierGpuBlobTransferLane::~ExpertTierGpuBlobTransferLane()
    {
        if (hasInFlightWork() || unfenced_work_)
        {
            LOG_ERROR("[ExpertTierGpuBlobTransferLane] Destroyed with unresolved DMA"
                      << " lane=" << config_.lane_name
                      << " src=" << config_.source_device.to_string()
                      << " dst=" << config_.destination_device.to_string());
            std::terminate();
        }
        releaseQuiescentResources();
    }

    bool ExpertTierGpuBlobTransferLane::materialized() const noexcept
    {
        if (!source_backend_ || !destination_backend_ ||
            !source_context_ || !destination_context_ ||
            source_ordinal_ < 0 || destination_ordinal_ < 0 ||
            !source_stream_ || !destination_stream_)
        {
            return false;
        }
        return std::all_of(
            slots_.begin(),
            slots_.end(),
            [this](const Slot &slot)
            {
                const bool base = slot.source_pinned &&
                                  slot.destination_pinned &&
                                  slot.source_event &&
                                  slot.destination_event;
                const bool timing = !config_.collect_timing_measurements ||
                                    (slot.source_timing_start_event &&
                                     slot.source_timing_stop_event &&
                                     slot.destination_timing_start_event &&
                                     slot.destination_timing_stop_event);
                return base && timing;
            });
    }

    bool ExpertTierGpuBlobTransferLane::materialize(
        std::string *error) noexcept
    {
        if (materialized())
            return true;
        if (source_backend_ || destination_backend_ || source_context_ ||
            destination_context_ || source_ordinal_ >= 0 ||
            destination_ordinal_ >= 0 || source_stream_ || destination_stream_)
        {
            assignBlobTransferError(
                error,
                "Heterogeneous GPU blob lane has a partial resource set");
            return false;
        }

        try
        {
            source_backend_ = getBackendFor(config_.source_device);
            destination_backend_ = getBackendFor(config_.destination_device);
            if (!source_backend_ || !destination_backend_)
                throw std::runtime_error("A configured GPU backend is unavailable");

            source_ordinal_ = config_.source_device.gpu_ordinal();
            destination_ordinal_ = config_.destination_device.gpu_ordinal();
            source_context_ = &GPUDeviceContextPool::instance().getContext(
                config_.source_device);
            destination_context_ = &GPUDeviceContextPool::instance().getContext(
                config_.destination_device);
            source_stream_ = source_context_->getOrCreateAuxiliaryStream(
                "expert_tier_gpu_blob_source:" + config_.lane_name);
            destination_stream_ =
                destination_context_->getOrCreateAuxiliaryStream(
                    "expert_tier_gpu_blob_destination:" + config_.lane_name);
            if (!source_stream_ || !destination_stream_)
                throw std::runtime_error("Could not create both auxiliary streams");

            /*
             * Each runtime owns the host memory used by its own DMA engine.
             * A bounded CPU memcpy is the portable bridge between CUDA and HIP.
             */
            for (Slot &slot : slots_)
            {
                slot.source_event =
                    source_backend_->createEvent(source_ordinal_);
                slot.destination_event =
                    destination_backend_->createEvent(destination_ordinal_);
                if (config_.collect_timing_measurements)
                {
                    slot.source_timing_start_event =
                        source_backend_->createTimingEvent(source_ordinal_);
                    slot.source_timing_stop_event =
                        source_backend_->createTimingEvent(source_ordinal_);
                    slot.destination_timing_start_event =
                        destination_backend_->createTimingEvent(
                            destination_ordinal_);
                    slot.destination_timing_stop_event =
                        destination_backend_->createTimingEvent(
                            destination_ordinal_);
                }
                slot.source_pinned = static_cast<std::uint8_t *>(
                    source_backend_->allocatePinned(
                        config_.staging_capacity_bytes,
                        source_ordinal_));
                slot.destination_pinned = static_cast<std::uint8_t *>(
                    destination_backend_->allocatePinned(
                        config_.staging_capacity_bytes,
                        destination_ordinal_));
                if (!slot.source_event || !slot.destination_event ||
                    !slot.source_pinned || !slot.destination_pinned ||
                    (config_.collect_timing_measurements &&
                     (!slot.source_timing_start_event ||
                      !slot.source_timing_stop_event ||
                      !slot.destination_timing_start_event ||
                      !slot.destination_timing_stop_event)))
                {
                    throw std::runtime_error(
                        "Could not allocate double-buffered events/pinned storage");
                }
            }
        }
        catch (const std::exception &exception)
        {
            assignBlobTransferError(error, exception.what());
            releaseQuiescentResources();
            return false;
        }
        catch (...)
        {
            assignBlobTransferError(
                error,
                "Heterogeneous GPU blob lane materialization threw a non-standard exception");
            releaseQuiescentResources();
            return false;
        }
        return true;
    }

    bool ExpertTierGpuBlobTransferLane::start(
        const GpuExpertPackedDescriptor &source,
        const GpuExpertPackedDescriptor &destination,
        const ExpertTierSourceReadiness &source_readiness,
        std::string *error) noexcept
    {
        if (!materialized())
        {
            assignBlobTransferError(
                error,
                "Heterogeneous GPU blob lane is not materialized");
            return false;
        }
        if (progress_ == ExpertTierGpuBlobTransferProgress::Pending ||
            progress_ == ExpertTierGpuBlobTransferProgress::Failed)
        {
            assignBlobTransferError(
                error,
                "Heterogeneous GPU blob lane is already occupied");
            return false;
        }
        if (!protocol_.begin(source, destination, error))
            return false;

        source_ = source;
        destination_ = destination;
        contiguous_source_ = nullptr;
        contiguous_destination_ = nullptr;
        carries_contiguous_projection_ = false;
        return beginTransfer(source_readiness, error);
    }

    bool ExpertTierGpuBlobTransferLane::startContiguous(
        const void *source,
        void *destination,
        std::size_t bytes,
        const ExpertTierSourceReadiness &source_readiness,
        std::string *error) noexcept
    {
        if (!materialized())
        {
            assignBlobTransferError(
                error,
                "Heterogeneous GPU blob lane is not materialized");
            return false;
        }
        if (progress_ == ExpertTierGpuBlobTransferProgress::Pending ||
            progress_ == ExpertTierGpuBlobTransferProgress::Failed)
        {
            assignBlobTransferError(
                error,
                "Heterogeneous GPU blob lane is already occupied");
            return false;
        }
        if (!source || !destination ||
            !protocol_.beginContiguous(bytes, error))
        {
            if ((!source || !destination) && error)
                *error =
                    "Heterogeneous GPU contiguous transfer requires two device pointers";
            return false;
        }

        source_ = {};
        destination_ = {};
        contiguous_source_ =
            static_cast<const std::uint8_t *>(source);
        contiguous_destination_ = static_cast<std::uint8_t *>(destination);
        carries_contiguous_projection_ = true;
        return beginTransfer(source_readiness, error);
    }

    bool ExpertTierGpuBlobTransferLane::beginTransfer(
        const ExpertTierSourceReadiness &source_readiness,
        std::string *error) noexcept
    {

        /*
         * Only a freshly produced source needs the source-runtime dependency.
         * An installed RCU bank is already quiescent and immutable. The
         * destination never sees a CUDA event in HIP or a HIP event in CUDA.
         */
        if (source_readiness.requiresProducerWait() &&
            !source_context_->waitEventChecked(
                source_readiness.event(),
                source_stream_))
        {
            assignBlobTransferError(
                error,
                "Could not enqueue source producer dependency");
            return false;
        }

        for (Slot &slot : slots_)
        {
            slot.chunk = {};
            slot.phase = SlotPhase::Idle;
            slot.source_timing_valid = false;
            slot.destination_timing_valid = false;
        }
        active_slots_ = 0;
        completed_bytes_ = 0;
        last_destination_event_ = nullptr;
        failure_requested_ = false;
        failure_counted_ = false;
        unfenced_work_ = false;
        failure_.clear();
        transfer_device_nanoseconds_ = 0;
        transfer_host_nanoseconds_ = 0;
        progress_ = ExpertTierGpuBlobTransferProgress::Pending;
        transfer_started_at_ = std::chrono::steady_clock::now();
        ++stats_.transfers_started;
        return enqueueAvailableSourceChunks(error);
    }

    void ExpertTierGpuBlobTransferLane::requestFailure(
        const std::string &message,
        std::string *error) noexcept
    {
        if (!failure_requested_)
        {
            failure_requested_ = true;
            failure_ = message;
        }
        assignBlobTransferError(error, failure_);
        if (!failure_counted_)
        {
            failure_counted_ = true;
            ++stats_.failed_transfers;
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "heterogeneous_gpu_blob_transfer_failures",
                1.0,
                "maintenance",
                config_.perf_device,
                {{"lane", config_.lane_name},
                 {"src", config_.source_device.to_string()},
                 {"dst", config_.destination_device.to_string()}});
        }
    }

    const std::uint8_t *ExpertTierGpuBlobTransferLane::sourceChunkPointer(
        const ExpertTierGpuBlobChunk &chunk) const noexcept
    {
        if (carries_contiguous_projection_)
        {
            return chunk.region == ExpertTierGpuBlobRegion::Payload &&
                           contiguous_source_
                       ? contiguous_source_ + chunk.region_offset
                       : nullptr;
        }
        const std::uint8_t *base = nullptr;
        switch (chunk.region)
        {
        case ExpertTierGpuBlobRegion::Payload:
            base = source_.ptrs.d_vnni;
            break;
        case ExpertTierGpuBlobRegion::Scales:
            base = static_cast<const std::uint8_t *>(source_.ptrs.d_scales);
            break;
        case ExpertTierGpuBlobRegion::Mins:
            base = static_cast<const std::uint8_t *>(source_.ptrs.d_mins);
            break;
        case ExpertTierGpuBlobRegion::Emins:
            base = static_cast<const std::uint8_t *>(source_.ptrs.d_emins);
            break;
        }
        return base ? base + chunk.region_offset : nullptr;
    }

    std::uint8_t *ExpertTierGpuBlobTransferLane::destinationChunkPointer(
        const ExpertTierGpuBlobChunk &chunk) const noexcept
    {
        if (carries_contiguous_projection_)
        {
            return chunk.region == ExpertTierGpuBlobRegion::Payload &&
                           contiguous_destination_
                       ? contiguous_destination_ + chunk.region_offset
                       : nullptr;
        }
        std::uint8_t *base = nullptr;
        switch (chunk.region)
        {
        case ExpertTierGpuBlobRegion::Payload:
            base = destination_.ptrs.d_vnni;
            break;
        case ExpertTierGpuBlobRegion::Scales:
            base = static_cast<std::uint8_t *>(destination_.ptrs.d_scales);
            break;
        case ExpertTierGpuBlobRegion::Mins:
            base = static_cast<std::uint8_t *>(destination_.ptrs.d_mins);
            break;
        case ExpertTierGpuBlobRegion::Emins:
            base = static_cast<std::uint8_t *>(destination_.ptrs.d_emins);
            break;
        }
        return base ? base + chunk.region_offset : nullptr;
    }

    bool ExpertTierGpuBlobTransferLane::enqueueSourceChunk(
        Slot &slot,
        const ExpertTierGpuBlobChunk &chunk,
        std::string *error) noexcept
    {
        const std::uint8_t *source_pointer = sourceChunkPointer(chunk);
        if (!chunk.valid() || !source_pointer ||
            chunk.bytes > config_.staging_capacity_bytes)
        {
            requestFailure(
                "Heterogeneous GPU blob protocol produced an invalid source chunk",
                error);
            return false;
        }

        bool timing_started = true;
        if (config_.collect_timing_measurements)
        {
            timing_started = source_backend_->recordEvent(
                slot.source_timing_start_event,
                source_ordinal_,
                source_stream_);
        }
        bool copied = timing_started && source_backend_->deviceToHostOnStream(
            slot.source_pinned,
            source_pointer,
            chunk.bytes,
            source_ordinal_,
            source_stream_);
        bool timing_stopped = true;
        if (config_.collect_timing_measurements)
        {
            timing_stopped = source_backend_->recordEvent(
                slot.source_timing_stop_event,
                source_ordinal_,
                source_stream_);
            copied = copied && timing_stopped;
            if (!timing_started || !timing_stopped)
                ++stats_.timing_measurement_failures;
        }
        /*
         * Fence even after an enqueue error: a runtime may have accepted part
         * of the operation before reporting failure. The slot is not reusable
         * until its source event is observed ready.
         */
        if (!source_backend_->recordEvent(
                slot.source_event,
                source_ordinal_,
                source_stream_))
        {
            unfenced_work_ = true;
            requestFailure(
                "Could not fence heterogeneous source DMA",
                error);
            progress_ = ExpertTierGpuBlobTransferProgress::Failed;
            return false;
        }

        slot.chunk = chunk;
        slot.phase = SlotPhase::SourceDmaPending;
        slot.source_timing_valid = timing_started && timing_stopped;
        slot.destination_timing_valid = false;
        ++active_slots_;
        ++stats_.chunks_submitted;
        ++stats_.source_d2h_submissions;
        stats_.bytes_submitted += chunk.bytes;
        stats_.maximum_in_flight_chunks = std::max<std::uint64_t>(
            stats_.maximum_in_flight_chunks,
            static_cast<std::uint64_t>(active_slots_));
        if (!copied)
        {
            requestFailure(
                slot.source_timing_valid
                    ? "Could not submit heterogeneous source D2H chunk"
                    : "Could not record heterogeneous source timing events",
                error);
            return false;
        }
        return true;
    }

    bool ExpertTierGpuBlobTransferLane::enqueueAvailableSourceChunks(
        std::string *error) noexcept
    {
        if (failure_requested_)
            return false;
        for (Slot &slot : slots_)
        {
            if (slot.phase != SlotPhase::Idle)
                continue;
            const std::optional<ExpertTierGpuBlobChunk> chunk =
                protocol_.takeNext();
            if (!chunk.has_value())
                break;
            if (!enqueueSourceChunk(slot, *chunk, error))
                return false;
        }
        return active_slots_ != 0;
    }

    bool ExpertTierGpuBlobTransferLane::relayAndEnqueueDestination(
        Slot &slot,
        std::string *error) noexcept
    {
        std::uint8_t *destination_pointer =
            destinationChunkPointer(slot.chunk);
        if (!destination_pointer)
        {
            requestFailure(
                "Heterogeneous GPU blob destination pointer is null",
                error);
            return false;
        }

        /*
         * The source event made source_pinned CPU-owned. Copying into the
         * destination runtime's pinned allocation is the only cross-runtime
         * handoff; it is bounded by the lane capacity and runs on maintenance.
         */
        const auto host_copy_started = std::chrono::steady_clock::now();
        std::memcpy(
            slot.destination_pinned,
            slot.source_pinned,
            slot.chunk.bytes);
        const auto host_elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - host_copy_started);
        transfer_host_nanoseconds_ = saturatingExpertTierMeasurementAdd(
            transfer_host_nanoseconds_,
            static_cast<std::uint64_t>(
                std::max<std::int64_t>(1, host_elapsed.count())));
        ++stats_.host_relay_copies;
        stats_.host_relay_bytes += slot.chunk.bytes;

        bool timing_started = true;
        if (config_.collect_timing_measurements)
        {
            timing_started = destination_backend_->recordEvent(
                slot.destination_timing_start_event,
                destination_ordinal_,
                destination_stream_);
        }
        bool copied = timing_started && destination_backend_->hostToDeviceOnStream(
            destination_pointer,
            slot.destination_pinned,
            slot.chunk.bytes,
            destination_ordinal_,
            destination_stream_);
        bool timing_stopped = true;
        if (config_.collect_timing_measurements)
        {
            timing_stopped = destination_backend_->recordEvent(
                slot.destination_timing_stop_event,
                destination_ordinal_,
                destination_stream_);
            copied = copied && timing_stopped;
            if (!timing_started || !timing_stopped)
                ++stats_.timing_measurement_failures;
        }
        if (!destination_backend_->recordEvent(
                slot.destination_event,
                destination_ordinal_,
                destination_stream_))
        {
            unfenced_work_ = true;
            requestFailure(
                "Could not fence heterogeneous destination DMA",
                error);
            progress_ = ExpertTierGpuBlobTransferProgress::Failed;
            return false;
        }

        slot.phase = SlotPhase::DestinationDmaPending;
        slot.destination_timing_valid = timing_started && timing_stopped;
        last_destination_event_ = slot.destination_event;
        ++stats_.destination_h2d_submissions;
        if (!copied)
        {
            requestFailure(
                slot.destination_timing_valid
                    ? "Could not submit heterogeneous destination H2D chunk"
                    : "Could not record heterogeneous destination timing events",
                error);
            return false;
        }
        return true;
    }

    ExpertTierGpuBlobTransferProgress ExpertTierGpuBlobTransferLane::poll(
        std::string *error) noexcept
    {
        if (progress_ != ExpertTierGpuBlobTransferProgress::Pending)
        {
            if (progress_ == ExpertTierGpuBlobTransferProgress::Failed)
                assignBlobTransferError(error, failure_);
            return progress_;
        }

        bool observed_pending = false;
        for (Slot &slot : slots_)
        {
            if (slot.phase == SlotPhase::SourceDmaPending)
            {
                bool ready = false;
                if (!source_context_->queryEventChecked(
                        slot.source_event,
                        ready))
                {
                    unfenced_work_ = true;
                    requestFailure(
                        "Heterogeneous source event query failed",
                        error);
                    progress_ = ExpertTierGpuBlobTransferProgress::Failed;
                    return progress_;
                }
                if (!ready)
                {
                    observed_pending = true;
                    continue;
                }
                if (config_.collect_timing_measurements &&
                    slot.source_timing_valid &&
                    !collectSourceTiming(slot, error))
                {
                    requestFailure(
                        "Heterogeneous source timing interval is unavailable",
                        error);
                }
                slot.source_timing_valid = false;
                if (!failure_requested_ &&
                    !relayAndEnqueueDestination(slot, error))
                {
                    /* The destination event, when recorded, still owns slot. */
                }
                else if (failure_requested_)
                {
                    /* No new DMA after failure; source completion frees slot. */
                    slot.phase = SlotPhase::Idle;
                    slot.chunk = {};
                    --active_slots_;
                }
            }

            if (slot.phase == SlotPhase::DestinationDmaPending)
            {
                bool ready = false;
                if (!destination_context_->queryEventChecked(
                        slot.destination_event,
                        ready))
                {
                    unfenced_work_ = true;
                    requestFailure(
                        "Heterogeneous destination event query failed",
                        error);
                    progress_ = ExpertTierGpuBlobTransferProgress::Failed;
                    return progress_;
                }
                if (!ready)
                {
                    observed_pending = true;
                    continue;
                }

                if (config_.collect_timing_measurements &&
                    slot.destination_timing_valid &&
                    !collectDestinationTiming(slot, error))
                {
                    requestFailure(
                        "Heterogeneous destination timing interval is unavailable",
                        error);
                }
                slot.destination_timing_valid = false;

                completed_bytes_ += slot.chunk.bytes;
                ++stats_.chunks_completed;
                slot.phase = SlotPhase::Idle;
                slot.chunk = {};
                --active_slots_;
            }
        }

        if (failure_requested_)
        {
            if (active_slots_ == 0)
                progress_ = ExpertTierGpuBlobTransferProgress::Failed;
            else
                ++stats_.pending_event_polls;
            assignBlobTransferError(error, failure_);
            return progress_;
        }

        (void)enqueueAvailableSourceChunks(error);
        if (failure_requested_)
        {
            if (active_slots_ == 0)
                progress_ = ExpertTierGpuBlobTransferProgress::Failed;
            assignBlobTransferError(error, failure_);
            return progress_;
        }

        if (protocol_.exhausted() && active_slots_ == 0)
        {
            if (completed_bytes_ != protocol_.totalBytes())
            {
                requestFailure(
                    "Heterogeneous GPU blob completion byte count mismatched",
                    error);
                progress_ = ExpertTierGpuBlobTransferProgress::Failed;
                return progress_;
            }
            progress_ = ExpertTierGpuBlobTransferProgress::Ready;
            ++stats_.transfers_completed;
            recordCompletion();
            return progress_;
        }

        if (observed_pending || active_slots_ != 0)
            ++stats_.pending_event_polls;
        return ExpertTierGpuBlobTransferProgress::Pending;
    }

    bool ExpertTierGpuBlobTransferLane::collectSourceTiming(
        const Slot &slot,
        std::string *error) noexcept
    {
        float elapsed_ms = 0.0F;
        if (!source_backend_->eventElapsedTimeMs(
                slot.source_timing_start_event,
                slot.source_timing_stop_event,
                source_ordinal_,
                &elapsed_ms) ||
            !std::isfinite(elapsed_ms) || elapsed_ms < 0.0F)
        {
            ++stats_.timing_measurement_failures;
            assignBlobTransferError(
                error,
                "Could not read heterogeneous source timing events");
            return false;
        }
        const auto nanoseconds = static_cast<std::uint64_t>(std::max(
            1.0,
            std::ceil(static_cast<double>(elapsed_ms) * 1'000'000.0)));
        transfer_device_nanoseconds_ = saturatingExpertTierMeasurementAdd(
            transfer_device_nanoseconds_, nanoseconds);
        return true;
    }

    bool ExpertTierGpuBlobTransferLane::collectDestinationTiming(
        const Slot &slot,
        std::string *error) noexcept
    {
        float elapsed_ms = 0.0F;
        if (!destination_backend_->eventElapsedTimeMs(
                slot.destination_timing_start_event,
                slot.destination_timing_stop_event,
                destination_ordinal_,
                &elapsed_ms) ||
            !std::isfinite(elapsed_ms) || elapsed_ms < 0.0F)
        {
            ++stats_.timing_measurement_failures;
            assignBlobTransferError(
                error,
                "Could not read heterogeneous destination timing events");
            return false;
        }
        const auto nanoseconds = static_cast<std::uint64_t>(std::max(
            1.0,
            std::ceil(static_cast<double>(elapsed_ms) * 1'000'000.0)));
        transfer_device_nanoseconds_ = saturatingExpertTierMeasurementAdd(
            transfer_device_nanoseconds_, nanoseconds);
        return true;
    }

    void ExpertTierGpuBlobTransferLane::recordCompletion() noexcept
    {
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - transfer_started_at_);
        const std::uint64_t wall_nanoseconds = static_cast<std::uint64_t>(
            std::max<std::int64_t>(1, elapsed.count()));
        stats_.last_measurement = {
            .sequence = stats_.transfers_completed,
            .bytes = static_cast<std::uint64_t>(completed_bytes_),
            .wall_nanoseconds = wall_nanoseconds,
            .device_nanoseconds = transfer_device_nanoseconds_,
            .host_nanoseconds = transfer_host_nanoseconds_,
        };
        const PerfStatsCollector::Tags tags{
            {"lane", config_.lane_name},
            {"src", config_.source_device.to_string()},
            {"dst", config_.destination_device.to_string()},
            {"slots", std::to_string(slots_.size())},
            {"layout", carries_contiguous_projection_ ? "contiguous"
                                                        : "separated"},
            {"background", "true"},
            {"blocking", "false"}};
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "heterogeneous_gpu_blob_transfers_completed",
            1.0,
            "maintenance",
            config_.perf_device,
            tags);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "heterogeneous_gpu_blob_bytes_completed",
            static_cast<double>(completed_bytes_),
            "maintenance",
            config_.perf_device,
            tags);
        PerfStatsCollector::recordTimingNs(
            "moe_overlay_residency",
            "heterogeneous_gpu_blob_transfer_wall_time",
            wall_nanoseconds,
            "maintenance",
            config_.perf_device,
            tags);
        if (config_.collect_timing_measurements)
        {
            PerfStatsCollector::recordTimingNs(
                "moe_overlay_residency",
                "heterogeneous_gpu_blob_device_work_time",
                transfer_device_nanoseconds_,
                "maintenance",
                config_.perf_device,
                tags);
            PerfStatsCollector::recordTimingNs(
                "moe_overlay_residency",
                "heterogeneous_gpu_blob_host_relay_time",
                transfer_host_nanoseconds_,
                "maintenance",
                config_.perf_device,
                tags);
        }
    }

    bool ExpertTierGpuBlobTransferLane::hasInFlightWork() const noexcept
    {
        return std::any_of(
            slots_.begin(),
            slots_.end(),
            [](const Slot &slot)
            {
                return slot.phase != SlotPhase::Idle;
            });
    }

    void ExpertTierGpuBlobTransferLane::releaseQuiescentResources() noexcept
    {
        for (Slot &slot : slots_)
        {
            if (source_backend_ && source_ordinal_ >= 0)
            {
                if (slot.source_event)
                    source_backend_->destroyEvent(
                        slot.source_event,
                        source_ordinal_);
                if (slot.source_timing_start_event)
                    source_backend_->destroyEvent(
                        slot.source_timing_start_event,
                        source_ordinal_);
                if (slot.source_timing_stop_event)
                    source_backend_->destroyEvent(
                        slot.source_timing_stop_event,
                        source_ordinal_);
                if (slot.source_pinned)
                    source_backend_->freePinned(
                        slot.source_pinned,
                        source_ordinal_);
            }
            if (destination_backend_ && destination_ordinal_ >= 0)
            {
                if (slot.destination_event)
                    destination_backend_->destroyEvent(
                        slot.destination_event,
                        destination_ordinal_);
                if (slot.destination_timing_start_event)
                    destination_backend_->destroyEvent(
                        slot.destination_timing_start_event,
                        destination_ordinal_);
                if (slot.destination_timing_stop_event)
                    destination_backend_->destroyEvent(
                        slot.destination_timing_stop_event,
                        destination_ordinal_);
                if (slot.destination_pinned)
                    destination_backend_->freePinned(
                        slot.destination_pinned,
                        destination_ordinal_);
            }
            slot = {};
        }
        /* Both auxiliary streams remain owned by their worker contexts. */
        source_stream_ = nullptr;
        destination_stream_ = nullptr;
        source_context_ = nullptr;
        destination_context_ = nullptr;
        source_backend_ = nullptr;
        destination_backend_ = nullptr;
        source_ordinal_ = -1;
        destination_ordinal_ = -1;
    }
} // namespace llaminar2
