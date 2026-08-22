/**
 * @file ExpertTierGpuBlobTransferLane.cpp
 * @brief Async-DMA CUDA/ROCm packed expert blob relay implementation.
 *
 * Device work is split into source D2H and destination H2D event-polled phases.
 * Host completion is the portable ownership handoff between CUDA and HIP;
 * cross-runtime event waits are deliberately forbidden. Two persistent slots
 * let the source prepare a later chunk while the destination consumes an
 * earlier one, without allocation, synchronization, or inference-stream joins.
 */

#include "ExpertTierGpuBlobTransferLane.h"

#include "../../transfer/TransferEngine.h"
#include "../../utils/Logger.h"
#include "../../utils/PerfStatsCollector.h"

#include <algorithm>
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
                "GPU host-relay lane requires two GPU endpoints");
        }
        const bool same_backend =
            config_.source_device.type == config_.destination_device.type;
        const bool topology_matches =
            (config_.relay_kind == ExpertTierGpuBlobRelayKind::CrossBackend &&
             !same_backend) ||
            (config_.relay_kind ==
                 ExpertTierGpuBlobRelayKind::SameBackendWithoutPeerAccess &&
             same_backend &&
             config_.source_device != config_.destination_device);
        if (!topology_matches)
        {
            throw std::invalid_argument(
                "GPU blob host relay kind contradicts its endpoint topology");
        }
        if (config_.lane_name.empty())
            throw std::invalid_argument(
                "GPU host-relay lane requires a stable non-empty name");
        if (!config_.source_progress_epoch ||
            !config_.destination_progress_epoch ||
            config_.source_progress_epoch->device() != config_.source_device ||
            config_.destination_progress_epoch->device() !=
                config_.destination_device ||
            config_.source_progress_epoch->maximumBytes() <
                config_.staging_capacity_bytes ||
            config_.destination_progress_epoch->maximumBytes() <
                config_.staging_capacity_bytes)
        {
            throw std::invalid_argument(
                "GPU host-relay lane requires matching source and destination retained epochs with sufficient byte capacity");
        }
        if (config_.perf_device.empty())
        {
            config_.perf_device =
                config_.source_device.to_string() + "->" +
                config_.destination_device.to_string();
        }
    }

    ExpertTierGpuBlobTransferLane::~ExpertTierGpuBlobTransferLane()
    {
        if (hasInFlightWork())
        {
            LOG_ERROR("[ExpertTierGpuBlobTransferLane] Destroyed with unresolved retained progress"
                      << " lane=" << config_.lane_name
                      << " src=" << config_.source_device.to_string()
                      << " dst=" << config_.destination_device.to_string());
            std::terminate();
        }
        releaseQuiescentResources();
    }

    bool ExpertTierGpuBlobTransferLane::materialized() const noexcept
    {
        return std::all_of(
            slots_.begin(),
            slots_.end(),
            [](const Slot &slot)
            {
                return slot.source_mapped &&
                       slot.source_mapped->isBound() &&
                       slot.destination_mapped &&
                       slot.destination_mapped->isBound() &&
                       slot.source_progress.valid() &&
                       slot.destination_progress.valid();
            });
    }

    bool ExpertTierGpuBlobTransferLane::materialize(
        std::string *error) noexcept
    {
        if (materialized())
            return true;
        if (std::any_of(
                slots_.begin(),
                slots_.end(),
                [](const Slot &slot)
                {
                    return slot.source_mapped || slot.destination_mapped ||
                           slot.source_progress.valid() ||
                           slot.destination_progress.valid();
                }))
        {
            assignBlobTransferError(
                error,
                "GPU host-relay lane has a partial resource set");
            return false;
        }

        try
        {
            /*
             * Each staging allocation is mapped only into the GPU that touches
             * it. The setup thread first-touches both host regions; a bounded
             * CPU memcpy is the sole portable CUDA/HIP ownership bridge.
             */
            TransferEngine transfer_engine;
            const std::array<DeviceId, 1> source_devices{
                config_.source_device};
            const std::array<DeviceId, 1> destination_devices{
                config_.destination_device};
            for (std::size_t slot_index = 0u;
                 slot_index < slots_.size(); ++slot_index)
            {
                Slot &slot = slots_[slot_index];
                slot.source_mapped =
                    transfer_engine.allocateMappedHostRegion(
                        config_.staging_capacity_bytes,
                        source_devices);
                slot.destination_mapped =
                    transfer_engine.allocateMappedHostRegion(
                        config_.staging_capacity_bytes,
                        destination_devices);
                slot.source_progress =
                    config_.source_progress_epoch->reserveSlot(
                        MappedTransferDirection::DeviceToHost,
                        slot.source_mapped,
                        config_.lane_name + ":source:" +
                        std::to_string(slot_index));
                slot.destination_progress =
                    config_.destination_progress_epoch->reserveSlot(
                        MappedTransferDirection::HostToDevice,
                        slot.destination_mapped,
                        config_.lane_name + ":destination:" +
                        std::to_string(slot_index));
                if (!slot.source_mapped || !slot.source_mapped->isBound() ||
                    !slot.destination_mapped ||
                    !slot.destination_mapped->isBound() ||
                    !slot.source_progress.valid() ||
                    !slot.destination_progress.valid())
                {
                    throw std::runtime_error(
                        "Could not allocate double-buffered mapped storage and epoch slots");
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
                "GPU host-relay lane materialization threw a non-standard exception");
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
                "GPU host-relay lane is not materialized");
            return false;
        }
        if (progress_ == ExpertTierGpuBlobTransferProgress::Pending ||
            progress_ == ExpertTierGpuBlobTransferProgress::Failed)
        {
            assignBlobTransferError(
                error,
                "GPU host-relay lane is already occupied");
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
                "GPU host-relay lane is not materialized");
            return false;
        }
        if (progress_ == ExpertTierGpuBlobTransferProgress::Pending ||
            progress_ == ExpertTierGpuBlobTransferProgress::Failed)
        {
            assignBlobTransferError(
                error,
                "GPU host-relay lane is already occupied");
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

    bool ExpertTierGpuBlobTransferLane::validateBoundDeviceStorage(
        std::string *error) noexcept
    {
        std::array<const void *, 4> source_addresses{};
        std::array<const void *, 4> destination_addresses{};
        std::size_t source_count = 0u;
        std::size_t destination_count = 0u;
        if (carries_contiguous_projection_)
        {
            source_addresses[source_count++] = contiguous_source_;
            destination_addresses[destination_count++] =
                contiguous_destination_;
        }
        else
        {
            source_addresses[source_count++] = source_.ptrs.d_vnni;
            source_addresses[source_count++] = source_.ptrs.d_scales;
            destination_addresses[destination_count++] =
                destination_.ptrs.d_vnni;
            destination_addresses[destination_count++] =
                destination_.ptrs.d_scales;
            if (source_.mins_bytes != 0u)
                source_addresses[source_count++] = source_.ptrs.d_mins;
            if (source_.emins_bytes != 0u)
                source_addresses[source_count++] = source_.ptrs.d_emins;
            if (destination_.mins_bytes != 0u)
                destination_addresses[destination_count++] =
                    destination_.ptrs.d_mins;
            if (destination_.emins_bytes != 0u)
                destination_addresses[destination_count++] =
                    destination_.ptrs.d_emins;
        }

        std::string validation_error;
        if (!config_.source_progress_epoch->validateDeviceAddresses(
                std::span<const void *const>(
                    source_addresses.data(), source_count),
                config_.lane_name + ":source",
                &validation_error) ||
            !config_.destination_progress_epoch->validateDeviceAddresses(
                std::span<const void *const>(
                    destination_addresses.data(), destination_count),
                config_.lane_name + ":destination",
                &validation_error))
        {
            assignBlobTransferError(
                error,
                validation_error.empty()
                    ? "GPU host-relay endpoint pointer ownership validation failed"
                    : validation_error);
            return false;
        }
        return true;
    }

    bool ExpertTierGpuBlobTransferLane::beginTransfer(
        const ExpertTierSourceReadiness &source_readiness,
        std::string *error) noexcept
    {
        /*
         * A retained replay may already be between claim and copy while the
         * maintenance thread enters here. Publishing a producer-dependent
         * command and then appending an event wait would therefore be racy.
         * Physical ExpertOverlay migration always reads an installed immutable
         * RCU bank, so make that production invariant an explicit type gate.
         */
        if (source_readiness.requiresProducerWait())
        {
            assignBlobTransferError(
                error,
                "Retained GPU host relay requires an immutable published-residency-bank source");
            return false;
        }
        if (!validateBoundDeviceStorage(error))
            return false;

        for (Slot &slot : slots_)
        {
            slot.chunk = {};
            slot.phase = SlotPhase::Idle;
            if (slot.source_progress.pending() ||
                slot.destination_progress.pending())
            {
                assignBlobTransferError(
                    error,
                    "GPU host-relay lane retained an unexpected outstanding command");
                return false;
            }
        }
        active_slots_ = 0;
        completed_bytes_ = 0;
        failure_requested_ = false;
        failure_counted_ = false;
        failure_.clear();
        transfer_device_nanoseconds_ = 0;
        transfer_host_nanoseconds_ = 0;
        max_source_dma_residence_nanoseconds_ = 0;
        max_destination_dma_residence_nanoseconds_ = 0;
        max_maintenance_poll_gap_nanoseconds_ = 0;
        progress_ = ExpertTierGpuBlobTransferProgress::Pending;
        transfer_started_at_ = std::chrono::steady_clock::now();
        last_poll_at_ = transfer_started_at_;
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
                "gpu_host_relay_transfer_failures",
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
                "GPU host-relay protocol produced an invalid source chunk",
                error);
            return false;
        }

        if (config_.collect_timing_measurements)
            slot.source_submitted_at = std::chrono::steady_clock::now();

        bool published = false;
        std::string submission_failure;
        try
        {
            slot.source_progress.publishDeviceToMappedHost(
                source_pointer,
                chunk.bytes,
                0u,
                chunk.bytes);
            published = true;
        }
        catch (const std::exception &exception)
        {
            submission_failure = exception.what();
        }
        catch (...)
        {
            submission_failure =
                "Source retained progress publication threw a non-standard exception";
        }

        if (!published)
        {
            requestFailure(
                submission_failure.empty()
                    ? "Could not publish GPU host-relay source command"
                    : submission_failure,
                error);
            return false;
        }
        slot.chunk = chunk;
        slot.phase = SlotPhase::SourceProgressPending;
        ++active_slots_;
        ++stats_.chunks_submitted;
        ++stats_.source_d2h_submissions;
        ++stats_.source_progress_kernel_submissions;
        stats_.bytes_submitted += chunk.bytes;
        stats_.maximum_in_flight_chunks = std::max<std::uint64_t>(
            stats_.maximum_in_flight_chunks,
            static_cast<std::uint64_t>(active_slots_));
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
                "GPU host-relay destination pointer is null",
                error);
            return false;
        }

        /*
         * The source completion generation made the mapped source pages
         * CPU-owned. Copying into destination-mapped pages is the only
         * cross-runtime handoff; it is bounded and runs on maintenance.
         */
        const auto host_copy_started = std::chrono::steady_clock::now();
        std::memcpy(
            slot.destination_mapped->mutableHostData(),
            slot.source_mapped->mutableHostData(),
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

        if (config_.collect_timing_measurements)
            slot.destination_submitted_at = std::chrono::steady_clock::now();

        bool published = false;
        std::string submission_failure;
        try
        {
            slot.destination_progress.publishMappedHostToDevice(
                destination_pointer,
                slot.chunk.bytes,
                0u,
                slot.chunk.bytes);
            published = true;
        }
        catch (const std::exception &exception)
        {
            submission_failure = exception.what();
        }
        catch (...)
        {
            submission_failure =
                "Destination retained progress publication threw a non-standard exception";
        }

        if (!published)
        {
            requestFailure(
                submission_failure.empty()
                    ? "Could not publish GPU host-relay destination command"
                    : submission_failure,
                error);
            return false;
        }
        slot.phase = SlotPhase::DestinationProgressPending;
        ++stats_.destination_h2d_submissions;
        ++stats_.destination_progress_kernel_submissions;
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

        if (config_.collect_timing_measurements)
        {
            const auto now = std::chrono::steady_clock::now();
            const auto poll_gap =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    now - last_poll_at_).count();
            max_maintenance_poll_gap_nanoseconds_ = std::max(
                max_maintenance_poll_gap_nanoseconds_,
                static_cast<std::uint64_t>(
                    std::max<std::int64_t>(1, poll_gap)));
            last_poll_at_ = now;
        }

        bool observed_pending = false;
        for (Slot &slot : slots_)
        {
            if (slot.phase == SlotPhase::SourceProgressPending)
            {
                std::string progress_error;
                const MappedTransferProgress source_progress =
                    slot.source_progress.poll(&progress_error);
                if (source_progress == MappedTransferProgress::Failed)
                {
                    requestFailure(
                        progress_error.empty()
                            ? "GPU host-relay source progress command failed"
                            : progress_error,
                        error);
                    slot.phase = SlotPhase::Idle;
                    slot.chunk = {};
                    --active_slots_;
                    continue;
                }
                if (source_progress == MappedTransferProgress::Pending)
                {
                    observed_pending = true;
                    continue;
                }
                if (config_.collect_timing_measurements)
                {
                    const auto residence =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() -
                            slot.source_submitted_at).count();
                    max_source_dma_residence_nanoseconds_ = std::max(
                        max_source_dma_residence_nanoseconds_,
                        static_cast<std::uint64_t>(
                            std::max<std::int64_t>(1, residence)));
                }
                if (failure_requested_ ||
                    !relayAndEnqueueDestination(slot, error))
                {
                    /* Source completion owns no storage after this point. */
                    slot.phase = SlotPhase::Idle;
                    slot.chunk = {};
                    --active_slots_;
                }
            }

            if (slot.phase == SlotPhase::DestinationProgressPending)
            {
                std::string progress_error;
                const MappedTransferProgress destination_progress =
                    slot.destination_progress.poll(&progress_error);
                if (destination_progress == MappedTransferProgress::Failed)
                {
                    requestFailure(
                        progress_error.empty()
                            ? "GPU host-relay destination progress command failed"
                            : progress_error,
                        error);
                    slot.phase = SlotPhase::Idle;
                    slot.chunk = {};
                    --active_slots_;
                    continue;
                }
                if (destination_progress == MappedTransferProgress::Pending)
                {
                    observed_pending = true;
                    continue;
                }

                if (config_.collect_timing_measurements)
                {
                    const auto residence =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() -
                            slot.destination_submitted_at).count();
                    max_destination_dma_residence_nanoseconds_ = std::max(
                        max_destination_dma_residence_nanoseconds_,
                        static_cast<std::uint64_t>(
                            std::max<std::int64_t>(1, residence)));
                }

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
                    "GPU host-relay completion byte count mismatched",
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
        stats_.last_max_source_dma_residence_nanoseconds =
            max_source_dma_residence_nanoseconds_;
        stats_.last_max_destination_dma_residence_nanoseconds =
            max_destination_dma_residence_nanoseconds_;
        stats_.last_max_maintenance_poll_gap_nanoseconds =
            max_maintenance_poll_gap_nanoseconds_;
        const PerfStatsCollector::Tags tags{
            {"lane", config_.lane_name},
            {"src", config_.source_device.to_string()},
            {"dst", config_.destination_device.to_string()},
            {"relay_kind",
             config_.relay_kind == ExpertTierGpuBlobRelayKind::CrossBackend
                 ? "cross_backend"
                 : "same_backend_no_peer"},
            {"slots", std::to_string(slots_.size())},
            {"layout", carries_contiguous_projection_ ? "contiguous"
                                                        : "separated"},
            {"source_transport", "retained_mapped_progress_epoch"},
            {"destination_transport", "retained_mapped_progress_epoch"},
            {"submission", "ahead_of_inference"},
            {"stream_class", "latency_critical"},
            {"background", "true"},
            {"blocking", "false"}};
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "gpu_host_relay_transfers_completed",
            1.0,
            "maintenance",
            config_.perf_device,
            tags);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "gpu_host_relay_bytes_completed",
            static_cast<double>(completed_bytes_),
            "maintenance",
            config_.perf_device,
            tags);
        PerfStatsCollector::recordTimingNs(
            "moe_overlay_residency",
            "gpu_host_relay_transfer_wall_time",
            wall_nanoseconds,
            "maintenance",
            config_.perf_device,
            tags);
        if (config_.collect_timing_measurements)
        {
            PerfStatsCollector::recordTimingNs(
                "moe_overlay_residency",
                "gpu_host_relay_device_work_time",
                transfer_device_nanoseconds_,
                "maintenance",
                config_.perf_device,
                tags);
            PerfStatsCollector::recordTimingNs(
                "moe_overlay_residency",
                "gpu_host_relay_host_copy_time",
                transfer_host_nanoseconds_,
                "maintenance",
                config_.perf_device,
                tags);
            PerfStatsCollector::recordTimingNs(
                "moe_overlay_residency",
                "gpu_host_relay_source_dma_max_chunk_residence_time",
                max_source_dma_residence_nanoseconds_,
                "maintenance",
                config_.perf_device,
                tags);
            PerfStatsCollector::recordTimingNs(
                "moe_overlay_residency",
                "gpu_host_relay_destination_dma_max_chunk_residence_time",
                max_destination_dma_residence_nanoseconds_,
                "maintenance",
                config_.perf_device,
                tags);
            PerfStatsCollector::recordTimingNs(
                "moe_overlay_residency",
                "gpu_host_relay_maintenance_max_poll_gap",
                max_maintenance_poll_gap_nanoseconds_,
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
            /* Slot destruction releases leases and mapped registrations only
             * after both completion generations have been observed. */
            slot = {};
        }
    }
} // namespace llaminar2
