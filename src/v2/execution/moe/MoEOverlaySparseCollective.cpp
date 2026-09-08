/**
 * @file MoEOverlaySparseCollective.cpp
 * @brief Compact sparse payload transport for graph-native MoE overlay collectives.
 */

#include "execution/moe/MoEOverlaySparseCollective.h"
#include "execution/moe/MoEOverlayActivationPacketABI.h"

#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "interfaces/IMPIContext.h"
#include "transfer/TransferEngine.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>

namespace llaminar2
{
    bool MoEOverlayDispatchTicketHeader::isValid() const noexcept
    {
        if (magic != kMagic || abi_version != kABIVersion ||
            workspace_generation == 0 || layer_idx < 0 ||
            bucket_row_capacity <= 0 || top_k <= 0 || d_model <= 0 ||
            logical_row_count <= 0 ||
            logical_row_count > bucket_row_capacity ||
            return_logical_row_count < 0 ||
            return_logical_row_count > bucket_row_capacity)
        {
            return false;
        }

        const uint64_t expected_routes =
            static_cast<uint64_t>(bucket_row_capacity) *
            static_cast<uint64_t>(top_k);
        return expected_routes <=
                   static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) &&
               route_capacity == static_cast<int32_t>(expected_routes) &&
               source_device_kind >=
                   static_cast<int32_t>(DeviceType::CPU) &&
               source_device_kind <=
                   static_cast<int32_t>(DeviceType::ROCm) &&
               source_device_ordinal >= 0;
    }

    bool MoEOverlayDispatchTicket::isValid() const noexcept
    {
        return header && header->isValid() && routing_indices_fp32 &&
               routing_weights_fp32 && hidden_rows_fp32 && return_rows_fp32;
    }

    bool MoEOverlayDispatchTicket::returnPayloadReady() const noexcept
    {
        return isValid() &&
               header->return_logical_row_count ==
                   header->logical_row_count;
    }

    namespace
    {
        size_t checkedTicketProduct(
            size_t lhs,
            size_t rhs,
            const char *description)
        {
            if (lhs != 0 &&
                rhs > std::numeric_limits<size_t>::max() / lhs)
            {
                throw std::overflow_error(
                    std::string("MoE overlay ticket ") + description +
                    " size overflow");
            }
            return lhs * rhs;
        }

        size_t checkedTicketAdd(
            size_t lhs,
            size_t rhs,
            const char *description)
        {
            if (rhs > std::numeric_limits<size_t>::max() - lhs)
            {
                throw std::overflow_error(
                    std::string("MoE overlay ticket ") + description +
                    " size overflow");
            }
            return lhs + rhs;
        }

        size_t alignTicketOffset(size_t offset, size_t alignment)
        {
            const size_t remainder = offset % alignment;
            return remainder == 0
                       ? offset
                       : checkedTicketAdd(
                             offset,
                             alignment - remainder,
                             "alignment");
        }
    } // namespace

    MoEOverlayDispatchTicketStorage::~MoEOverlayDispatchTicketStorage()
    {
        release();
    }

    void MoEOverlayDispatchTicketStorage::release() noexcept
    {
        cpu_storage_.clear();
        allocation_ = nullptr;
        allocation_bytes_ = 0;
        payload_region_.reset();
        publication_timeline_ = nullptr;
        publication_region_.reset();
        source_device_ = DeviceId::cpu();
        layer_idx_ = -1;
        bucket_rows_ = 0;
        top_k_ = 0;
        d_model_ = 0;
        workspace_generation_ = 0;
        ticket_ = {};
    }

    void MoEOverlayDispatchTicketStorage::bindFixedCapacity(
        int layer_idx,
        int bucket_rows,
        int top_k,
        int d_model,
        DeviceId source_device,
        uint64_t workspace_generation,
        std::shared_ptr<MappedHostTransferArena> mapped_arena)
    {
        if (layer_idx < 0 || bucket_rows <= 0 || top_k <= 0 ||
            d_model <= 0 || !source_device.is_valid() ||
            workspace_generation == 0)
        {
            throw std::invalid_argument(
                "MoE overlay dispatch ticket requires valid layer, geometry, "
                "source device, and workspace generation");
        }
        if (allocation_)
        {
            throw std::logic_error(
                "MoE overlay dispatch ticket capacity is immutable after binding");
        }

        const size_t row_count = static_cast<size_t>(bucket_rows);
        const size_t route_count = checkedTicketProduct(
            row_count, static_cast<size_t>(top_k), "route capacity");
        if (route_count >
            static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        {
            throw std::invalid_argument(
                "MoE overlay dispatch ticket route capacity exceeds INT32 ABI");
        }
        const size_t hidden_count = checkedTicketProduct(
            row_count, static_cast<size_t>(d_model), "hidden capacity");

        size_t offset = sizeof(MoEOverlayDispatchTicketHeader);
        offset = alignTicketOffset(offset, alignof(float));
        const size_t routing_indices_offset = offset;
        offset = checkedTicketAdd(
            offset,
            checkedTicketProduct(route_count, sizeof(float), "route ids"),
            "route ids");
        offset = alignTicketOffset(offset, alignof(float));
        const size_t routing_weights_offset = offset;
        offset = checkedTicketAdd(
            offset,
            checkedTicketProduct(route_count, sizeof(float), "route weights"),
            "route weights");
        offset = alignTicketOffset(offset, alignof(float));
        const size_t hidden_offset = offset;
        offset = checkedTicketAdd(
            offset,
            checkedTicketProduct(hidden_count, sizeof(float), "hidden rows"),
            "hidden rows");
        offset = alignTicketOffset(offset, alignof(float));
        const size_t return_rows_offset = offset;
        offset = checkedTicketAdd(
            offset,
            checkedTicketProduct(hidden_count, sizeof(float), "return rows"),
            "return rows");
        allocation_bytes_ = offset;

        if (source_device.is_gpu())
        {
            if (!mapped_arena ||
                mapped_arena->devices().size() != 1u ||
                mapped_arena->devices().front() != source_device)
            {
                throw std::invalid_argument(
                    "GPU MoE overlay dispatch tickets require one model-owned mapped arena for the exact source device");
            }
            /*
             * Both logical regions are cache-line-isolated slices of a
             * geometrically growing model-owned arena. They retain independent
             * bounds for TransferEngine validation without multiplying native
             * host registrations by layers, buckets, or MTP depth. Device
             * kernels publish through the payload alias; no retained graph
             * records D2H DMA.
             */
            constexpr size_t kPublicationCacheLineBytes = 64u;
            payload_region_ = mapped_arena->allocate(
                allocation_bytes_, kPublicationCacheLineBytes);
            allocation_ = payload_region_->mutableHostData();
            publication_region_ = mapped_arena->allocate(
                sizeof(std::uint64_t), kPublicationCacheLineBytes);
            publication_timeline_ = static_cast<std::uint64_t *>(
                publication_region_->mutableHostData());
            std::atomic_ref<std::uint64_t>(*publication_timeline_).store(
                0u, std::memory_order_release);
        }
        else
        {
            const size_t words =
                (allocation_bytes_ + sizeof(std::max_align_t) - 1u) /
                sizeof(std::max_align_t);
            cpu_storage_.resize(words);
            allocation_ = cpu_storage_.data();
        }

        std::memset(allocation_, 0, allocation_bytes_);
        auto *const base = static_cast<std::byte *>(allocation_);
        ticket_.header =
            reinterpret_cast<MoEOverlayDispatchTicketHeader *>(base);
        ticket_.routing_indices_fp32 = reinterpret_cast<float *>(
            base + routing_indices_offset);
        ticket_.routing_weights_fp32 = reinterpret_cast<float *>(
            base + routing_weights_offset);
        ticket_.hidden_rows_fp32 = reinterpret_cast<float *>(
            base + hidden_offset);
        ticket_.return_rows_fp32 = reinterpret_cast<float *>(
            base + return_rows_offset);

        *ticket_.header = MoEOverlayDispatchTicketHeader{
            .magic = MoEOverlayDispatchTicketHeader::kMagic,
            .abi_version = MoEOverlayDispatchTicketHeader::kABIVersion,
            .workspace_generation = workspace_generation,
            .residency_epoch = 0,
            .layer_idx = layer_idx,
            .bucket_row_capacity = bucket_rows,
            .route_capacity = static_cast<int32_t>(route_count),
            .top_k = top_k,
            .d_model = d_model,
            .logical_row_count = bucket_rows,
            .return_logical_row_count = 0,
            .source_device_kind = static_cast<int32_t>(source_device.type),
            .source_device_ordinal = source_device.ordinal,
        };
        source_device_ = source_device;
        layer_idx_ = layer_idx;
        bucket_rows_ = bucket_rows;
        top_k_ = top_k;
        d_model_ = d_model;
        workspace_generation_ = workspace_generation;
    }

    bool MoEOverlayDispatchTicketStorage::hasValidBoundIdentity() const noexcept
    {
        if (!allocation_ || !ticket_.isValid() || !ticket_.header)
            return false;
        const auto &header = *ticket_.header;
        return header.layer_idx == layer_idx_ &&
               header.bucket_row_capacity == bucket_rows_ &&
               header.top_k == top_k_ &&
               header.d_model == d_model_ &&
               header.workspace_generation == workspace_generation_ &&
               header.source_device_kind ==
                   static_cast<int32_t>(source_device_.type) &&
               header.source_device_ordinal == source_device_.ordinal &&
               (!source_device_.is_gpu() ||
                hasCapturedPublicationContract());
    }

    bool MoEOverlayDispatchTicketStorage::hasCapturedPublicationContract()
        const noexcept
    {
        return source_device_.is_gpu() && publication_region_ &&
               payload_region_ && payload_region_->isBound() &&
               payload_region_->hasDevice(source_device_) &&
               payload_region_->contains(0u, allocation_bytes_) &&
               publication_region_->isBound() && publication_timeline_ &&
               publication_region_->hasDevice(source_device_) &&
               publication_region_->contains(
                   0u, sizeof(std::uint64_t)) &&
               (reinterpret_cast<std::uintptr_t>(publication_timeline_) &
                (alignof(std::uint64_t) - 1u)) == 0u;
    }

    bool MoEOverlayDispatchTicketStorage::enqueueCapturedPayload(
        const CapturedDevicePayload &payload,
        void *stream,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!source_device_.is_gpu())
            return true;
        if (!stream || !hasCapturedPublicationContract() ||
            !payload.routing_indices || !payload.routing_weights ||
            !payload.hidden_rows || payload.route_bytes == 0u ||
            payload.hidden_bytes == 0u)
        {
            if (error)
            {
                *error =
                    "captured ticket payload requires mapped storage, exact stream, and complete positive device geometry";
            }
            return false;
        }

        const auto destination_offset = [&](const void *destination,
                                            size_t bytes) -> size_t
        {
            const std::uintptr_t base =
                reinterpret_cast<std::uintptr_t>(allocation_);
            const std::uintptr_t target =
                reinterpret_cast<std::uintptr_t>(destination);
            if (target < base || target - base > allocation_bytes_ ||
                bytes > allocation_bytes_ -
                            static_cast<size_t>(target - base))
            {
                throw std::out_of_range(
                    "captured ticket payload destination exceeds its immutable mapped region");
            }
            return static_cast<size_t>(target - base);
        };

        try
        {
            auto &transfer = TransferEngine::instance();
            const auto enqueue = [&](const void *source,
                                     const void *destination,
                                     size_t bytes)
            {
                transfer.enqueuePersistentDeviceRegionToMappedHostByKernel(
                    source,
                    bytes,
                    /*source_offset=*/0u,
                    *payload_region_,
                    destination_offset(destination, bytes),
                    bytes,
                    source_device_,
                    stream);
            };
            if (payload.logical_row_count)
            {
                enqueue(
                    payload.logical_row_count,
                    &ticket_.header->logical_row_count,
                    sizeof(ticket_.header->logical_row_count));
            }
            enqueue(
                payload.routing_indices,
                ticket_.routing_indices_fp32,
                payload.route_bytes);
            enqueue(
                payload.routing_weights,
                ticket_.routing_weights_fp32,
                payload.route_bytes);
            enqueue(
                payload.hidden_rows,
                ticket_.hidden_rows_fp32,
                payload.hidden_bytes);
            return true;
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
            return false;
        }
        catch (...)
        {
            if (error)
            {
                *error =
                    "captured ticket payload publication threw a non-standard exception";
            }
            return false;
        }
    }

    bool MoEOverlayDispatchTicketStorage::armCapturedPublication(
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!source_device_.is_gpu())
            return true;
        if (!hasCapturedPublicationContract())
        {
            if (error)
            {
                *error =
                    "captured ticket has no complete mapped publication contract";
            }
            return false;
        }

        /*
         * Each layer owns a distinct ticket and segmented transactions are
         * serial for that graph identity. Resetting before launch cannot race a
         * previous replay; the preceding CPU consumer already acquired it.
         */
        std::atomic_ref<std::uint64_t>(*publication_timeline_).store(
            0u, std::memory_order_release);
        return true;
    }

    bool MoEOverlayDispatchTicketStorage::enqueueCapturedPublication(
        void *stream,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!source_device_.is_gpu())
            return true;
        if (!stream || !hasCapturedPublicationContract())
        {
            if (error)
            {
                *error =
                    "captured ticket publication requires its exact non-null stream and mapped contract";
            }
            return false;
        }

        try
        {
            constexpr std::uint64_t kPublished = 1u;
            TransferEngine::instance().enqueueMappedTimelinePublish64(
                *publication_region_,
                /*signal_offset=*/0u,
                kPublished,
                source_device_,
                stream);
            return true;
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
            return false;
        }
        catch (...)
        {
            if (error)
            {
                *error =
                    "captured ticket publication threw a non-standard exception";
            }
            return false;
        }
    }

    bool MoEOverlayDispatchTicketStorage::awaitCapturedPublication(
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!source_device_.is_gpu())
            return true;
        if (!hasCapturedPublicationContract())
        {
            if (error)
            {
                *error =
                    "captured ticket wait has no complete mapped publication contract";
            }
            return false;
        }

        constexpr std::uint64_t kPublished = 1u;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(
                                  collective_timeout_policy::
                                      kDefaultCollectiveTimeoutMs);
        std::uint64_t polls = 0u;
        for (;;)
        {
            const std::uint64_t observed =
                std::atomic_ref<std::uint64_t>(*publication_timeline_).load(
                    std::memory_order_acquire);
            if (observed == kPublished)
                return true;
            if (observed > kPublished)
            {
                if (error)
                {
                    *error =
                        "captured ticket publication timeline exceeded its fixed replay value";
                }
                return false;
            }
            if (std::chrono::steady_clock::now() >= deadline)
            {
                if (error)
                {
                    *error =
                        "captured ticket publication exceeded the canonical 30-second protocol deadline; storage=" +
                        std::to_string(
                            reinterpret_cast<std::uintptr_t>(this)) +
                        ",publication=" +
                        std::to_string(reinterpret_cast<std::uintptr_t>(
                            publication_timeline_));
                }
                return false;
            }
            ++polls;

            /* Keep sub-millisecond handoff latency without monopolizing the
             * controller core if a peer or device stalls pathologically. */
            if ((polls & 1023u) == 0u)
                std::this_thread::yield();
        }
    }

    void MoEOverlayCanonicalRouteReturnTicketStorage::bindFixedCapacity(
        int layer_idx,
        size_t route_capacity,
        int d_model,
        DeviceId continuation_device,
        uint64_t workspace_generation,
        std::shared_ptr<MappedHostTransferRegion> contribution_region,
        std::shared_ptr<MappedHostTransferArena> metadata_arena)
    {
        if (metadata_region_ || contribution_region_)
        {
            throw std::logic_error(
                "canonical route return ticket capacity is immutable after binding");
        }
        if (layer_idx < 0 || route_capacity == 0u || d_model <= 0 ||
            !continuation_device.is_gpu() || workspace_generation == 0u ||
            route_capacity >
                static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
            !contribution_region || !contribution_region->isBound() ||
            !contribution_region->hasDevice(continuation_device) ||
            !metadata_arena || metadata_arena->devices().size() != 1u ||
            metadata_arena->devices().front() != continuation_device)
        {
            throw std::invalid_argument(
                "canonical route return ticket requires complete layer, route, mapped contribution, exact GPU, and model-owned metadata arena identity");
        }

        const size_t contribution_elements = checkedTicketProduct(
            route_capacity,
            static_cast<size_t>(d_model),
            "canonical contribution capacity");
        const size_t contribution_bytes = checkedTicketProduct(
            contribution_elements,
            sizeof(float),
            "canonical contribution bytes");
        if (!contribution_region->contains(0u, contribution_bytes))
        {
            throw std::invalid_argument(
                "canonical route return contribution region is smaller than immutable ticket geometry");
        }

        size_t offset = sizeof(MoEOverlayCanonicalRouteTicketControl);
        offset = alignTicketOffset(offset, alignof(int32_t));
        original_route_slots_offset_ = offset;
        offset = checkedTicketAdd(
            offset,
            checkedTicketProduct(
                route_capacity, sizeof(int32_t), "original route slots"),
            "original route slots");
        offset = alignTicketOffset(offset, alignof(int32_t));
        compact_route_slots_offset_ = offset;
        offset = checkedTicketAdd(
            offset,
            checkedTicketProduct(
                route_capacity, sizeof(int32_t), "compact route slots"),
            "compact route slots");

        constexpr size_t kControlCacheLineBytes = 64u;
        metadata_region_ = metadata_arena->allocate(
            offset, kControlCacheLineBytes);
        contribution_region_ = std::move(contribution_region);
        control_host_ = static_cast<MoEOverlayCanonicalRouteTicketControl *>(
            metadata_region_->mutableHostData());
        original_route_slots_host_ = static_cast<int32_t *>(
            metadata_region_->mutableHostData(original_route_slots_offset_));
        compact_route_slots_host_ = static_cast<int32_t *>(
            metadata_region_->mutableHostData(compact_route_slots_offset_));
        *control_host_ = MoEOverlayCanonicalRouteTicketControl{
            .magic = MoEOverlayCanonicalRouteTicketControl::kMagic,
            .abi_version =
                MoEOverlayCanonicalRouteTicketControl::kABIVersion,
            .workspace_generation = workspace_generation,
            .published_sequence = 0u,
            .consumed_sequence = 0u,
            .live_entry_count = 0u,
            .residency_epoch = 0u,
            .layer_idx = layer_idx,
            .route_capacity = static_cast<int32_t>(route_capacity),
            .d_model = d_model,
            .publication_status = static_cast<std::int32_t>(
                MoEOverlayCanonicalRouteTicketStatus::Empty),
        };
        continuation_device_ = continuation_device;
        layer_idx_ = layer_idx;
        route_capacity_ = route_capacity;
        d_model_ = d_model;
        workspace_generation_ = workspace_generation;
        armed_sequence_ = 0u;
        producer_lifecycle_ = ProducerLifecycle::Quiescent;
    }

    MoEOverlayCanonicalRouteReturnTicketStorage::Publication::~Publication()
    {
        reset();
    }

    MoEOverlayCanonicalRouteReturnTicketStorage::Publication::Publication(
        Publication &&other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          sequence_(std::exchange(other.sequence_, 0u))
    {
    }

    MoEOverlayCanonicalRouteReturnTicketStorage::Publication &
    MoEOverlayCanonicalRouteReturnTicketStorage::Publication::operator=(
        Publication &&other) noexcept
    {
        if (this == &other)
            return *this;
        reset();
        owner_ = std::exchange(other.owner_, nullptr);
        sequence_ = std::exchange(other.sequence_, 0u);
        return *this;
    }

    bool MoEOverlayCanonicalRouteReturnTicketStorage::Publication::publish(
        size_t live_entry_count) noexcept
    {
        if (!owner_ || sequence_ == 0u ||
            !owner_->publishArmed(sequence_, live_entry_count))
        {
            return false;
        }
        owner_ = nullptr;
        sequence_ = 0u;
        return true;
    }

    void MoEOverlayCanonicalRouteReturnTicketStorage::Publication::reset()
        noexcept
    {
        if (owner_ && sequence_ != 0u)
            owner_->cancelArmed(sequence_);
        owner_ = nullptr;
        sequence_ = 0u;
    }

    MoEOverlayCanonicalRouteReturnTicketStorage::Publication
    MoEOverlayCanonicalRouteReturnTicketStorage::arm(
        uint64_t residency_epoch) noexcept
    {
        if (!hasValidBoundIdentity() || residency_epoch == 0u ||
            producer_lifecycle_ != ProducerLifecycle::Quiescent)
            return {};

        const std::uint64_t published =
            std::atomic_ref<std::uint64_t>(control_host_->published_sequence)
                .load(std::memory_order_acquire);
        const std::uint64_t consumed =
            std::atomic_ref<std::uint64_t>(control_host_->consumed_sequence)
                .load(std::memory_order_acquire);
        /* One mapped payload has one producer and one consumer. Refusing reuse
         * while a publication is live makes overwrite-before-consume impossible
         * and turns a missing GPU acknowledgement into a precise hard failure. */
        if (published != consumed ||
            published == std::numeric_limits<std::uint64_t>::max())
        {
            return {};
        }

        control_host_->live_entry_count = 0u;
        control_host_->residency_epoch = residency_epoch;
        control_host_->publication_status = static_cast<std::int32_t>(
            MoEOverlayCanonicalRouteTicketStatus::Empty);
        armed_sequence_ = published + 1u;
        producer_lifecycle_ = ProducerLifecycle::Armed;
        return Publication(this, armed_sequence_);
    }

    bool MoEOverlayCanonicalRouteReturnTicketStorage::publishArmed(
        uint64_t sequence, size_t live_entry_count) noexcept
    {
        if (!hasValidBoundIdentity() ||
            producer_lifecycle_ != ProducerLifecycle::Armed ||
            live_entry_count > route_capacity_ ||
            control_host_->residency_epoch == 0u || armed_sequence_ == 0u ||
            sequence != armed_sequence_)
        {
            return false;
        }
        const std::uint64_t published =
            std::atomic_ref<std::uint64_t>(control_host_->published_sequence)
                .load(std::memory_order_acquire);
        const std::uint64_t consumed =
            std::atomic_ref<std::uint64_t>(control_host_->consumed_sequence)
                .load(std::memory_order_acquire);
        if (published != consumed || armed_sequence_ != published + 1u)
            return false;

        control_host_->live_entry_count = live_entry_count;
        control_host_->publication_status = static_cast<std::int32_t>(
            MoEOverlayCanonicalRouteTicketStatus::Success);
        std::atomic_ref<std::uint64_t>(control_host_->published_sequence)
            .store(armed_sequence_, std::memory_order_release);
        armed_sequence_ = 0u;
        producer_lifecycle_ = ProducerLifecycle::Quiescent;
        return true;
    }

    void MoEOverlayCanonicalRouteReturnTicketStorage::cancelArmed(
        uint64_t sequence) noexcept
    {
        if (producer_lifecycle_ != ProducerLifecycle::Armed ||
            sequence == 0u || sequence != armed_sequence_)
        {
            return;
        }
        /* Nothing was release-published, so no GPU may observe these partial
         * bytes. Restore only host producer ownership; monotonic cursors remain
         * unchanged and the next arm receives the same unused sequence. */
        control_host_->live_entry_count = 0u;
        control_host_->residency_epoch = 0u;
        control_host_->publication_status = static_cast<std::int32_t>(
            MoEOverlayCanonicalRouteTicketStatus::Empty);
        armed_sequence_ = 0u;
        producer_lifecycle_ = ProducerLifecycle::Quiescent;
    }

    bool MoEOverlayCanonicalRouteReturnTicketStorage::publishAbort() noexcept
    {
        if (!hasValidBoundIdentity() ||
            producer_lifecycle_ != ProducerLifecycle::Quiescent)
        {
            return false;
        }

        const std::uint64_t published =
            std::atomic_ref<std::uint64_t>(control_host_->published_sequence)
                .load(std::memory_order_acquire);
        const std::uint64_t consumed =
            std::atomic_ref<std::uint64_t>(control_host_->consumed_sequence)
                .load(std::memory_order_acquire);
        if (published < consumed || published - consumed > 1u)
            return false;
        if (published > consumed)
        {
            /* A successfully published payload (or an earlier abort in this
             * drain pass) already gives the captured consumer a progress edge.
             * Never mutate its bytes while the GPU owns them. */
            return true;
        }
        if (published == std::numeric_limits<std::uint64_t>::max())
            return false;

        control_host_->live_entry_count = 0u;
        control_host_->residency_epoch = 0u;
        control_host_->publication_status = static_cast<std::int32_t>(
            MoEOverlayCanonicalRouteTicketStatus::Aborted);
        std::atomic_ref<std::uint64_t>(control_host_->published_sequence)
            .store(published + 1u, std::memory_order_release);
        return true;
    }

    bool MoEOverlayCanonicalRouteReturnTicketStorage::payloadReady()
        const noexcept
    {
        if (!hasValidBoundIdentity() ||
            producer_lifecycle_ != ProducerLifecycle::Quiescent)
        {
            return false;
        }
        const std::uint64_t published =
            std::atomic_ref<std::uint64_t>(control_host_->published_sequence)
                .load(std::memory_order_acquire);
        const std::uint64_t consumed =
            std::atomic_ref<std::uint64_t>(control_host_->consumed_sequence)
                .load(std::memory_order_acquire);
        return published > consumed && published - consumed == 1u &&
               control_host_->publication_status ==
                   static_cast<std::int32_t>(
                       MoEOverlayCanonicalRouteTicketStatus::Success) &&
               control_host_->residency_epoch != 0u &&
               control_host_->live_entry_count <= route_capacity_;
    }

    bool MoEOverlayCanonicalRouteReturnTicketStorage::payloadReadyFor(
        uint64_t residency_epoch) const noexcept
    {
        return residency_epoch != 0u && payloadReady() &&
               control_host_->residency_epoch == residency_epoch;
    }

    bool MoEOverlayCanonicalRouteReturnTicketStorage::
        publicationSucceededFor(uint64_t residency_epoch) const noexcept
    {
        if (residency_epoch == 0u || !hasValidBoundIdentity() ||
            producer_lifecycle_ != ProducerLifecycle::Quiescent)
        {
            return false;
        }

        const std::uint64_t published =
            std::atomic_ref<std::uint64_t>(control_host_->published_sequence)
                .load(std::memory_order_acquire);
        const std::uint64_t consumed =
            std::atomic_ref<std::uint64_t>(control_host_->consumed_sequence)
                .load(std::memory_order_acquire);
        /*
         * The GPU may advance consumed_sequence immediately after the CPU's
         * release publication. Both the one-pending state and the exact
         * acknowledged state certify producer success; an initial, malformed,
         * armed, or aborted record does not.
         */
        return published != 0u && published >= consumed &&
               published - consumed <= 1u &&
               control_host_->publication_status ==
                   static_cast<std::int32_t>(
                       MoEOverlayCanonicalRouteTicketStatus::Success) &&
               control_host_->residency_epoch == residency_epoch &&
               control_host_->live_entry_count <= route_capacity_;
    }

    bool MoEOverlayCanonicalRouteReturnTicketStorage::hasValidBoundIdentity()
        const noexcept
    {
        return metadata_region_ && metadata_region_->isBound() &&
               metadata_region_->hasDevice(continuation_device_) &&
               contribution_region_ && contribution_region_->isBound() &&
               contribution_region_->hasDevice(continuation_device_) &&
               control_host_ && control_host_->valid() &&
               control_host_->workspace_generation == workspace_generation_ &&
               control_host_->layer_idx == layer_idx_ &&
               control_host_->route_capacity ==
                   static_cast<int32_t>(route_capacity_) &&
               control_host_->d_model == d_model_ &&
               original_route_slots_host_ && compact_route_slots_host_ &&
               continuation_device_.is_gpu();
    }

    int32_t *MoEOverlayCanonicalRouteReturnTicketStorage::
        originalRouteSlotsHost() const noexcept
    {
        return original_route_slots_host_;
    }

    int32_t *MoEOverlayCanonicalRouteReturnTicketStorage::
        compactRouteSlotsHost() const noexcept
    {
        return compact_route_slots_host_;
    }

    float *MoEOverlayCanonicalRouteReturnTicketStorage::
        contributionRowsHost() const noexcept
    {
        try
        {
            return contribution_region_
                       ? static_cast<float *>(
                             contribution_region_->mutableHostData())
                       : nullptr;
        }
        catch (...)
        {
            return nullptr;
        }
    }

    MoEOverlayCanonicalRouteTicketControl *
    MoEOverlayCanonicalRouteReturnTicketStorage::controlDeviceAlias()
        const noexcept
    {
        try
        {
            return metadata_region_
                       ? static_cast<MoEOverlayCanonicalRouteTicketControl *>(
                             metadata_region_->deviceAlias(
                                 continuation_device_))
                       : nullptr;
        }
        catch (...)
        {
            return nullptr;
        }
    }

    const int32_t *MoEOverlayCanonicalRouteReturnTicketStorage::
        originalRouteSlotsDeviceAlias() const noexcept
    {
        try
        {
            return metadata_region_
                       ? static_cast<const int32_t *>(
                             metadata_region_->deviceAlias(
                                 continuation_device_,
                                 original_route_slots_offset_))
                       : nullptr;
        }
        catch (...)
        {
            return nullptr;
        }
    }

    const int32_t *MoEOverlayCanonicalRouteReturnTicketStorage::
        compactRouteSlotsDeviceAlias() const noexcept
    {
        try
        {
            return metadata_region_
                       ? static_cast<const int32_t *>(
                             metadata_region_->deviceAlias(
                                 continuation_device_,
                                 compact_route_slots_offset_))
                       : nullptr;
        }
        catch (...)
        {
            return nullptr;
        }
    }

    const float *MoEOverlayCanonicalRouteReturnTicketStorage::
        contributionRowsDeviceAlias() const noexcept
    {
        try
        {
            return contribution_region_
                       ? static_cast<const float *>(
                             contribution_region_->deviceAlias(
                                 continuation_device_))
                       : nullptr;
        }
        catch (...)
        {
            return nullptr;
        }
    }

    namespace
    {
        constexpr uint32_t kPacketMagic = 0x32454f4dU; // "MOE2"
        /* Version 5 adds authenticated original/compact route-slot identities. */
        constexpr uint32_t kPacketVersion = 5;
        constexpr uint8_t kPacketKindDispatch = 1;
        constexpr uint8_t kPacketKindReturn = 2;

        template <typename T>
        void appendPod(std::vector<uint8_t> &buffer, const T &value)
        {
            const auto *ptr = reinterpret_cast<const uint8_t *>(&value);
            buffer.insert(buffer.end(), ptr, ptr + sizeof(T));
        }

        template <typename T>
        bool readPod(const uint8_t *data,
                     size_t size,
                     size_t *offset,
                     T *out)
        {
            if (!data || !offset || !out || *offset + sizeof(T) > size)
                return false;
            std::memcpy(out, data + *offset, sizeof(T));
            *offset += sizeof(T);
            return true;
        }

        std::string keyToStableString(const MoEOverlayCollectiveKey &key)
        {
            std::ostringstream ss;
            ss << static_cast<int>(key.key_namespace) << ':'
               << static_cast<int>(key.histogram_source) << ':'
               << key.generation_id << ':'
               << key.step_id << ':'
               << key.mtp_depth << ':'
               << key.layer_idx << ':'
               << key.tier_idx << ':'
               << key.domain_id << ':'
               << key.participant_id << ':'
               << static_cast<int>(key.direction) << ':'
               << key.sequence;
            return ss.str();
        }

        uint64_t makeCollectiveSequence(MoEOverlayCollectiveNamespace key_namespace,
                                        int mtp_depth,
                                        int layer_idx,
                                        int tier_idx,
                                        int participant_id,
                                        MoEOverlayCollectiveDirection direction)
        {
            const uint64_t namespace_offset =
                key_namespace == MoEOverlayCollectiveNamespace::MTP ? (1ull << 56) : 0ull;
            const uint64_t depth_component =
                key_namespace == MoEOverlayCollectiveNamespace::MTP
                    ? static_cast<uint64_t>(std::max(mtp_depth, 0)) * (1ull << 40)
                    : 0ull;
            const uint64_t direction_offset =
                direction == MoEOverlayCollectiveDirection::Dispatch ? 0ull : 2048ull;
            return namespace_offset +
                   depth_component +
                   static_cast<uint64_t>(std::max(layer_idx, 0)) * 8192ull +
                   direction_offset +
                   static_cast<uint64_t>(std::max(tier_idx, 0)) * 128ull +
                   static_cast<uint64_t>(std::max(participant_id, 0));
        }

        bool validateSparseRows(const MoEOverlaySparseRows &rows,
                                std::string *error)
        {
            if (rows.d_model <= 0 || rows.top_k <= 0)
            {
                if (error)
                    *error = "invalid sparse payload dimensions";
                return false;
            }
            if (!rows.row_ids_host || !rows.entry_offsets_host ||
                !rows.expert_ids_host || !rows.route_weights_host ||
                !rows.original_route_slots_host ||
                !rows.compact_route_slots_host || !rows.hidden_rows_fp32)
            {
                if (error)
                    *error = "sparse payload is missing host buffers";
                return false;
            }
            if (rows.live_row_count > rows.row_capacity || rows.live_entry_count > rows.entry_capacity)
            {
                if (error)
                    *error = "sparse payload live counts exceed capacity";
                return false;
            }
            if ((rows.live_row_count != 0 || rows.live_entry_count != 0) &&
                rows.residency_epoch == 0)
            {
                if (error)
                    *error = "non-empty sparse payload is missing a residency epoch";
                return false;
            }
            return true;
        }

        bool validateReturnRows(const MoEOverlayReturnRows &rows,
                                std::string *error)
        {
            if (rows.d_model <= 0)
            {
                if (error)
                    *error = "invalid return payload dimensions";
                return false;
            }
            if (!rows.row_ids_host || !rows.output_rows_fp32)
            {
                if (error)
                    *error = "return payload is missing host buffers";
                return false;
            }
            if (rows.live_row_count > rows.row_capacity)
            {
                if (error)
                    *error = "return payload live row count exceeds capacity";
                return false;
            }
            if (rows.live_row_count != 0 && rows.residency_epoch == 0)
            {
                if (error)
                    *error = "non-empty return payload is missing a residency epoch";
                return false;
            }
            return true;
        }

        bool ensureSparseInboundCapacity(MoEOverlaySparseRows *inbound,
                                         size_t required_rows,
                                         size_t required_entries,
                                         std::string *error)
        {
            if (!inbound)
            {
                if (error)
                    *error = "null inbound sparse payload";
                return false;
            }

            if (required_rows > inbound->row_capacity || required_entries > inbound->entry_capacity)
            {
                if (error)
                    *error = "inbound sparse payload capacity exceeded";
                return false;
            }
            return true;
        }

        bool ensureReturnInboundCapacity(MoEOverlayReturnRows *inbound,
                                         size_t required_rows,
                                         std::string *error)
        {
            if (!inbound)
            {
                if (error)
                    *error = "null inbound return payload";
                return false;
            }
            if (required_rows > inbound->row_capacity)
            {
                if (error)
                    *error = "inbound return payload capacity exceeded";
                return false;
            }
            return true;
        }

        struct DispatchPayloadCopy
        {
            MoEOverlayCollectiveKey key;
            uint64_t residency_epoch = 0;
            int32_t source_participant = -1;
            int32_t target_participant = -1;
            int32_t d_model = 0;
            int32_t top_k = 0;
            std::vector<int32_t> row_ids;
            std::vector<int32_t> entry_offsets;
            std::vector<int32_t> expert_ids;
            std::vector<float> route_weights;
            std::vector<int32_t> original_route_slots;
            std::vector<int32_t> compact_route_slots;
            std::vector<float> hidden_rows;
        };

        struct ReturnPayloadCopy
        {
            MoEOverlayCollectiveKey key;
            uint64_t residency_epoch = 0;
            int32_t source_participant = -1;
            int32_t target_participant = -1;
            int32_t d_model = 0;
            std::vector<int32_t> row_ids;
            std::vector<float> output_rows;
        };

        DispatchPayloadCopy copyDispatchPayload(const MoEOverlaySparseRows &rows)
        {
            DispatchPayloadCopy copy;
            copy.key = rows.key;
            copy.residency_epoch = rows.residency_epoch;
            copy.source_participant = rows.source_participant;
            copy.target_participant = rows.target_participant;
            copy.d_model = rows.d_model;
            copy.top_k = rows.top_k;
            copy.row_ids.assign(rows.row_ids_host, rows.row_ids_host + rows.live_row_count);
            copy.entry_offsets.assign(rows.entry_offsets_host,
                                      rows.entry_offsets_host + rows.live_row_count + 1);
            copy.expert_ids.assign(rows.expert_ids_host,
                                   rows.expert_ids_host + rows.live_entry_count);
            copy.route_weights.assign(rows.route_weights_host,
                                      rows.route_weights_host + rows.live_entry_count);
            copy.original_route_slots.assign(
                rows.original_route_slots_host,
                rows.original_route_slots_host + rows.live_entry_count);
            copy.compact_route_slots.assign(
                rows.compact_route_slots_host,
                rows.compact_route_slots_host + rows.live_entry_count);
            copy.hidden_rows.resize(
                rows.live_row_count * static_cast<size_t>(rows.d_model));
            for (size_t compact_row = 0u;
                 compact_row < rows.live_row_count;
                 ++compact_row)
            {
                const float *const source =
                    rows.hiddenRowForCompactIndex(compact_row);
                if (!source)
                {
                    throw std::invalid_argument(
                        "sparse dispatch payload has an invalid hidden-row address contract");
                }
                std::copy_n(
                    source,
                    static_cast<size_t>(rows.d_model),
                    copy.hidden_rows.data() +
                        compact_row * static_cast<size_t>(rows.d_model));
            }
            return copy;
        }

        ReturnPayloadCopy copyReturnPayload(const MoEOverlayReturnRows &rows)
        {
            ReturnPayloadCopy copy;
            copy.key = rows.key;
            copy.residency_epoch = rows.residency_epoch;
            copy.source_participant = rows.source_participant;
            copy.target_participant = rows.target_participant;
            copy.d_model = rows.d_model;
            copy.row_ids.assign(rows.row_ids_host, rows.row_ids_host + rows.live_row_count);
            copy.output_rows.assign(rows.output_rows_fp32,
                                    rows.output_rows_fp32 + rows.live_row_count * static_cast<size_t>(rows.d_model));
            return copy;
        }

        bool appendDispatchInbound(const DispatchPayloadCopy &payload,
                                   MoEOverlaySparseRows *inbound,
                                   std::string *error)
        {
            const size_t rows = payload.row_ids.size();
            const size_t entries = payload.expert_ids.size();
            const size_t total_rows = inbound->live_row_count + rows;
            const size_t total_entries = inbound->live_entry_count + entries;
            if (!ensureSparseInboundCapacity(inbound, total_rows, total_entries, error))
                return false;

            if (inbound->d_model != payload.d_model || inbound->top_k != payload.top_k)
            {
                if (error)
                    *error = "inbound sparse payload dimension mismatch";
                return false;
            }

            /*
             * Several sources may contribute to one participant packet. They
             * must all have routed against the same immutable residency
             * snapshot; combining epochs would make the selected prepared
             * expert bank ambiguous.
             */
            if (payload.residency_epoch != 0)
            {
                if (inbound->residency_epoch != 0 &&
                    inbound->residency_epoch != payload.residency_epoch)
                {
                    if (error)
                        *error = "inbound sparse payload mixes residency epochs";
                    return false;
                }
                inbound->residency_epoch = payload.residency_epoch;
            }

            std::copy(payload.row_ids.begin(),
                      payload.row_ids.end(),
                      inbound->row_ids_host + inbound->live_row_count);
            std::copy(payload.expert_ids.begin(),
                      payload.expert_ids.end(),
                      inbound->expert_ids_host + inbound->live_entry_count);
            std::copy(payload.route_weights.begin(),
                      payload.route_weights.end(),
                      inbound->route_weights_host + inbound->live_entry_count);
            std::copy(payload.original_route_slots.begin(),
                      payload.original_route_slots.end(),
                      inbound->original_route_slots_host + inbound->live_entry_count);
            std::copy(payload.compact_route_slots.begin(),
                      payload.compact_route_slots.end(),
                      inbound->compact_route_slots_host + inbound->live_entry_count);

            const size_t d_model = static_cast<size_t>(inbound->d_model);
            std::copy(payload.hidden_rows.begin(),
                      payload.hidden_rows.end(),
                      inbound->hidden_rows_fp32 + inbound->live_row_count * d_model);

            if (rows == 0)
            {
                if (inbound->live_row_count == 0)
                    inbound->entry_offsets_host[0] = 0;
                return true;
            }

            if (inbound->live_row_count == 0)
                inbound->entry_offsets_host[0] = 0;

            for (size_t row = 0; row < rows; ++row)
            {
                inbound->entry_offsets_host[inbound->live_row_count + row] =
                    static_cast<int32_t>(inbound->live_entry_count + static_cast<size_t>(payload.entry_offsets[row]));
            }
            inbound->entry_offsets_host[inbound->live_row_count + rows] =
                static_cast<int32_t>(inbound->live_entry_count + static_cast<size_t>(payload.entry_offsets[rows]));

            inbound->live_row_count = total_rows;
            inbound->live_entry_count = total_entries;
            return true;
        }

        bool appendReturnInbound(const ReturnPayloadCopy &payload,
                                 MoEOverlayReturnRows *inbound,
                                 std::string *error)
        {
            const size_t rows = payload.row_ids.size();
            const size_t total_rows = inbound->live_row_count + rows;
            if (!ensureReturnInboundCapacity(inbound, total_rows, error))
                return false;

            if (inbound->d_model != payload.d_model)
            {
                if (error)
                    *error = "inbound return payload dimension mismatch";
                return false;
            }

            if (payload.residency_epoch != 0)
            {
                if (inbound->residency_epoch != 0 &&
                    inbound->residency_epoch != payload.residency_epoch)
                {
                    if (error)
                        *error = "inbound return payload mixes residency epochs";
                    return false;
                }
                inbound->residency_epoch = payload.residency_epoch;
            }

            const size_t d_model = static_cast<size_t>(inbound->d_model);
            std::copy(payload.row_ids.begin(),
                      payload.row_ids.end(),
                      inbound->row_ids_host + inbound->live_row_count);
            std::copy(payload.output_rows.begin(),
                      payload.output_rows.end(),
                      inbound->output_rows_fp32 + inbound->live_row_count * d_model);

            inbound->live_row_count = total_rows;
            return true;
        }

        std::vector<uint8_t> serializeDispatchPacket(const DispatchPayloadCopy &payload)
        {
            std::vector<uint8_t> bytes;
            bytes.reserve(128 +
                          payload.row_ids.size() * sizeof(int32_t) +
                          payload.entry_offsets.size() * sizeof(int32_t) +
                          payload.expert_ids.size() * sizeof(int32_t) +
                          payload.route_weights.size() * sizeof(float) +
                          payload.original_route_slots.size() * sizeof(int32_t) +
                          payload.compact_route_slots.size() * sizeof(int32_t) +
                          payload.hidden_rows.size() * sizeof(float));

            appendPod(bytes, kPacketMagic);
            appendPod(bytes, kPacketVersion);
            appendPod(bytes, kPacketKindDispatch);
            appendPod(bytes, static_cast<uint8_t>(payload.key.key_namespace));
            appendPod(bytes, static_cast<uint8_t>(payload.key.histogram_source));
            appendPod(bytes, payload.key.generation_id);
            appendPod(bytes, payload.key.step_id);
            appendPod(bytes, payload.key.mtp_depth);
            appendPod(bytes, payload.key.layer_idx);
            appendPod(bytes, payload.key.tier_idx);
            appendPod(bytes, payload.key.domain_id);
            appendPod(bytes, payload.key.participant_id);
            appendPod(bytes, static_cast<uint8_t>(payload.key.direction));
            appendPod(bytes, payload.key.sequence);
            appendPod(bytes, payload.residency_epoch);
            appendPod(bytes, payload.source_participant);
            appendPod(bytes, payload.target_participant);
            appendPod(bytes, payload.d_model);
            appendPod(bytes, payload.top_k);

            const uint32_t row_count = static_cast<uint32_t>(payload.row_ids.size());
            const uint32_t entry_count = static_cast<uint32_t>(payload.expert_ids.size());
            appendPod(bytes, row_count);
            appendPod(bytes, entry_count);

            if (!payload.row_ids.empty())
                bytes.insert(bytes.end(),
                             reinterpret_cast<const uint8_t *>(payload.row_ids.data()),
                             reinterpret_cast<const uint8_t *>(payload.row_ids.data() + payload.row_ids.size()));
            if (!payload.entry_offsets.empty())
                bytes.insert(bytes.end(),
                             reinterpret_cast<const uint8_t *>(payload.entry_offsets.data()),
                             reinterpret_cast<const uint8_t *>(payload.entry_offsets.data() + payload.entry_offsets.size()));
            if (!payload.expert_ids.empty())
                bytes.insert(bytes.end(),
                             reinterpret_cast<const uint8_t *>(payload.expert_ids.data()),
                             reinterpret_cast<const uint8_t *>(payload.expert_ids.data() + payload.expert_ids.size()));
            if (!payload.route_weights.empty())
                bytes.insert(bytes.end(),
                             reinterpret_cast<const uint8_t *>(payload.route_weights.data()),
                             reinterpret_cast<const uint8_t *>(payload.route_weights.data() + payload.route_weights.size()));
            if (!payload.original_route_slots.empty())
                bytes.insert(bytes.end(),
                             reinterpret_cast<const uint8_t *>(payload.original_route_slots.data()),
                             reinterpret_cast<const uint8_t *>(payload.original_route_slots.data() + payload.original_route_slots.size()));
            if (!payload.compact_route_slots.empty())
                bytes.insert(bytes.end(),
                             reinterpret_cast<const uint8_t *>(payload.compact_route_slots.data()),
                             reinterpret_cast<const uint8_t *>(payload.compact_route_slots.data() + payload.compact_route_slots.size()));
            if (!payload.hidden_rows.empty())
                bytes.insert(bytes.end(),
                             reinterpret_cast<const uint8_t *>(payload.hidden_rows.data()),
                             reinterpret_cast<const uint8_t *>(payload.hidden_rows.data() + payload.hidden_rows.size()));
            return bytes;
        }

        bool deserializeDispatchPacket(const uint8_t *data,
                                       size_t size,
                                       DispatchPayloadCopy *out,
                                       std::string *error)
        {
            if (!data || !out)
            {
                if (error)
                    *error = "invalid dispatch packet buffer";
                return false;
            }

            size_t offset = 0;
            uint32_t magic = 0;
            uint32_t version = 0;
            uint8_t kind = 0;
            if (!readPod(data, size, &offset, &magic) ||
                !readPod(data, size, &offset, &version) ||
                !readPod(data, size, &offset, &kind))
            {
                if (error)
                    *error = "dispatch packet header is truncated";
                return false;
            }
            if (magic != kPacketMagic || version != kPacketVersion || kind != kPacketKindDispatch)
            {
                if (error)
                    *error = "dispatch packet header is invalid";
                return false;
            }

            uint8_t key_namespace = 0;
            uint8_t histogram_source = 0;
            uint8_t direction = 0;
            uint32_t row_count = 0;
            uint32_t entry_count = 0;
            if (!readPod(data, size, &offset, &key_namespace) ||
                !readPod(data, size, &offset, &histogram_source) ||
                !readPod(data, size, &offset, &out->key.generation_id) ||
                !readPod(data, size, &offset, &out->key.step_id) ||
                !readPod(data, size, &offset, &out->key.mtp_depth) ||
                !readPod(data, size, &offset, &out->key.layer_idx) ||
                !readPod(data, size, &offset, &out->key.tier_idx) ||
                !readPod(data, size, &offset, &out->key.domain_id) ||
                !readPod(data, size, &offset, &out->key.participant_id) ||
                !readPod(data, size, &offset, &direction) ||
                !readPod(data, size, &offset, &out->key.sequence) ||
                !readPod(data, size, &offset, &out->residency_epoch) ||
                !readPod(data, size, &offset, &out->source_participant) ||
                !readPod(data, size, &offset, &out->target_participant) ||
                !readPod(data, size, &offset, &out->d_model) ||
                !readPod(data, size, &offset, &out->top_k) ||
                !readPod(data, size, &offset, &row_count) ||
                !readPod(data, size, &offset, &entry_count))
            {
                if (error)
                    *error = "dispatch packet metadata is truncated";
                return false;
            }

            out->key.key_namespace = static_cast<MoEOverlayCollectiveNamespace>(key_namespace);
            out->key.histogram_source =
                static_cast<ExpertHistogramSource>(histogram_source);
            out->key.direction = static_cast<MoEOverlayCollectiveDirection>(direction);
            if (out->key.direction != MoEOverlayCollectiveDirection::Dispatch)
            {
                if (error)
                    *error = "dispatch packet has wrong collective direction";
                return false;
            }
            if (!out->key.isValid())
            {
                if (error)
                    *error = "dispatch packet has invalid collective key";
                return false;
            }
            if ((row_count != 0u || entry_count != 0u) &&
                out->residency_epoch == 0)
            {
                if (error)
                    *error = "non-empty dispatch packet has no residency epoch";
                return false;
            }

            out->row_ids.resize(row_count);
            out->entry_offsets.resize(static_cast<size_t>(row_count) + 1u);
            out->expert_ids.resize(entry_count);
            out->route_weights.resize(entry_count);
            out->original_route_slots.resize(entry_count);
            out->compact_route_slots.resize(entry_count);
            out->hidden_rows.resize(static_cast<size_t>(row_count) * static_cast<size_t>(out->d_model));

            const auto copyInto = [&](void *dst, size_t bytes, const char *name) -> bool
            {
                if (offset + bytes > size)
                {
                    if (error)
                    {
                        std::ostringstream ss;
                        ss << "dispatch packet is truncated while reading " << name;
                        *error = ss.str();
                    }
                    return false;
                }
                std::memcpy(dst, data + offset, bytes);
                offset += bytes;
                return true;
            };

            if (!copyInto(out->row_ids.data(), out->row_ids.size() * sizeof(int32_t), "row ids") ||
                !copyInto(out->entry_offsets.data(), out->entry_offsets.size() * sizeof(int32_t), "entry offsets") ||
                !copyInto(out->expert_ids.data(), out->expert_ids.size() * sizeof(int32_t), "expert ids") ||
                !copyInto(out->route_weights.data(), out->route_weights.size() * sizeof(float), "route weights") ||
                !copyInto(out->original_route_slots.data(), out->original_route_slots.size() * sizeof(int32_t), "original route slots") ||
                !copyInto(out->compact_route_slots.data(), out->compact_route_slots.size() * sizeof(int32_t), "compact route slots") ||
                !copyInto(out->hidden_rows.data(), out->hidden_rows.size() * sizeof(float), "hidden rows"))
            {
                return false;
            }

            return true;
        }

        std::vector<uint8_t> serializeReturnPacket(const ReturnPayloadCopy &payload)
        {
            std::vector<uint8_t> bytes;
            bytes.reserve(96 +
                          payload.row_ids.size() * sizeof(int32_t) +
                          payload.output_rows.size() * sizeof(float));

            appendPod(bytes, kPacketMagic);
            appendPod(bytes, kPacketVersion);
            appendPod(bytes, kPacketKindReturn);
            appendPod(bytes, static_cast<uint8_t>(payload.key.key_namespace));
            appendPod(bytes, static_cast<uint8_t>(payload.key.histogram_source));
            appendPod(bytes, payload.key.generation_id);
            appendPod(bytes, payload.key.step_id);
            appendPod(bytes, payload.key.mtp_depth);
            appendPod(bytes, payload.key.layer_idx);
            appendPod(bytes, payload.key.tier_idx);
            appendPod(bytes, payload.key.domain_id);
            appendPod(bytes, payload.key.participant_id);
            appendPod(bytes, static_cast<uint8_t>(payload.key.direction));
            appendPod(bytes, payload.key.sequence);
            appendPod(bytes, payload.residency_epoch);
            appendPod(bytes, payload.source_participant);
            appendPod(bytes, payload.target_participant);
            appendPod(bytes, payload.d_model);

            const uint32_t row_count = static_cast<uint32_t>(payload.row_ids.size());
            appendPod(bytes, row_count);

            if (!payload.row_ids.empty())
                bytes.insert(bytes.end(),
                             reinterpret_cast<const uint8_t *>(payload.row_ids.data()),
                             reinterpret_cast<const uint8_t *>(payload.row_ids.data() + payload.row_ids.size()));
            if (!payload.output_rows.empty())
                bytes.insert(bytes.end(),
                             reinterpret_cast<const uint8_t *>(payload.output_rows.data()),
                             reinterpret_cast<const uint8_t *>(payload.output_rows.data() + payload.output_rows.size()));
            return bytes;
        }

        bool deserializeReturnPacket(const uint8_t *data,
                                     size_t size,
                                     ReturnPayloadCopy *out,
                                     std::string *error)
        {
            if (!data || !out)
            {
                if (error)
                    *error = "invalid return packet buffer";
                return false;
            }

            size_t offset = 0;
            uint32_t magic = 0;
            uint32_t version = 0;
            uint8_t kind = 0;
            if (!readPod(data, size, &offset, &magic) ||
                !readPod(data, size, &offset, &version) ||
                !readPod(data, size, &offset, &kind))
            {
                if (error)
                    *error = "return packet header is truncated";
                return false;
            }
            if (magic != kPacketMagic || version != kPacketVersion || kind != kPacketKindReturn)
            {
                if (error)
                    *error = "return packet header is invalid";
                return false;
            }

            uint8_t key_namespace = 0;
            uint8_t histogram_source = 0;
            uint8_t direction = 0;
            uint32_t row_count = 0;
            if (!readPod(data, size, &offset, &key_namespace) ||
                !readPod(data, size, &offset, &histogram_source) ||
                !readPod(data, size, &offset, &out->key.generation_id) ||
                !readPod(data, size, &offset, &out->key.step_id) ||
                !readPod(data, size, &offset, &out->key.mtp_depth) ||
                !readPod(data, size, &offset, &out->key.layer_idx) ||
                !readPod(data, size, &offset, &out->key.tier_idx) ||
                !readPod(data, size, &offset, &out->key.domain_id) ||
                !readPod(data, size, &offset, &out->key.participant_id) ||
                !readPod(data, size, &offset, &direction) ||
                !readPod(data, size, &offset, &out->key.sequence) ||
                !readPod(data, size, &offset, &out->residency_epoch) ||
                !readPod(data, size, &offset, &out->source_participant) ||
                !readPod(data, size, &offset, &out->target_participant) ||
                !readPod(data, size, &offset, &out->d_model) ||
                !readPod(data, size, &offset, &row_count))
            {
                if (error)
                    *error = "return packet metadata is truncated";
                return false;
            }

            out->key.key_namespace = static_cast<MoEOverlayCollectiveNamespace>(key_namespace);
            out->key.histogram_source =
                static_cast<ExpertHistogramSource>(histogram_source);
            out->key.direction = static_cast<MoEOverlayCollectiveDirection>(direction);
            if (out->key.direction != MoEOverlayCollectiveDirection::ReturnReduce)
            {
                if (error)
                    *error = "return packet has wrong collective direction";
                return false;
            }
            if (!out->key.isValid())
            {
                if (error)
                    *error = "return packet has invalid collective key";
                return false;
            }
            if (row_count != 0u && out->residency_epoch == 0)
            {
                if (error)
                    *error = "non-empty return packet has no residency epoch";
                return false;
            }

            out->row_ids.resize(row_count);
            out->output_rows.resize(static_cast<size_t>(row_count) * static_cast<size_t>(out->d_model));

            const auto copyInto = [&](void *dst, size_t bytes, const char *name) -> bool
            {
                if (offset + bytes > size)
                {
                    if (error)
                    {
                        std::ostringstream ss;
                        ss << "return packet is truncated while reading " << name;
                        *error = ss.str();
                    }
                    return false;
                }
                std::memcpy(dst, data + offset, bytes);
                offset += bytes;
                return true;
            };

            if (!copyInto(out->row_ids.data(), out->row_ids.size() * sizeof(int32_t), "row ids") ||
                !copyInto(out->output_rows.data(), out->output_rows.size() * sizeof(float), "output rows"))
            {
                return false;
            }

            return true;
        }

        std::vector<uint8_t> gatherAllPayloads(const IMPIContext &mpi_ctx,
                                               const std::vector<uint8_t> &send_payload,
                                               std::vector<int> *recv_counts,
                                               std::vector<int> *displs)
        {
            const int world_size = mpi_ctx.world_size();
            std::vector<int32_t> counts32(static_cast<size_t>(world_size), 0);
            int32_t send_count = static_cast<int32_t>(send_payload.size());
            mpi_ctx.allgather_bytes(&send_count, counts32.data(), sizeof(int32_t));

            recv_counts->assign(static_cast<size_t>(world_size), 0);
            displs->assign(static_cast<size_t>(world_size), 0);
            int total = 0;
            for (int rank = 0; rank < world_size; ++rank)
            {
                (*recv_counts)[static_cast<size_t>(rank)] = counts32[static_cast<size_t>(rank)];
                (*displs)[static_cast<size_t>(rank)] = total;
                total += counts32[static_cast<size_t>(rank)];
            }

            std::vector<uint8_t> gathered(static_cast<size_t>(std::max(0, total)), 0);
            mpi_ctx.allgatherv_bytes(send_payload.empty() ? nullptr : send_payload.data(),
                                     send_count,
                                     gathered.empty() ? nullptr : gathered.data(),
                                     recv_counts->data(),
                                     displs->data());
            return gathered;
        }

    } // namespace

    const char *toString(MoEOverlayCollectiveDirection direction)
    {
        switch (direction)
        {
        case MoEOverlayCollectiveDirection::Dispatch:
            return "Dispatch";
        case MoEOverlayCollectiveDirection::ReturnReduce:
            return "ReturnReduce";
        }
        return "Unknown";
    }

    const char *toString(MoEOverlayCollectiveNamespace key_namespace)
    {
        switch (key_namespace)
        {
        case MoEOverlayCollectiveNamespace::Main:
            return "Main";
        case MoEOverlayCollectiveNamespace::MTP:
            return "MTP";
        }
        return "Unknown";
    }

    bool MoEOverlayCollectiveKey::isValid() const
    {
        if (layer_idx < 0 || tier_idx < 0 || domain_id < 0)
            return false;

        switch (key_namespace)
        {
        case MoEOverlayCollectiveNamespace::Main:
            return mtp_depth < 0 &&
                   (histogram_source ==
                        ExpertHistogramSource::DecodeToken ||
                    histogram_source ==
                        ExpertHistogramSource::PrefillChunk);
        case MoEOverlayCollectiveNamespace::MTP:
            return mtp_depth >= 0 &&
                   histogram_source ==
                       ExpertHistogramSource::GroupedVerifier;
        }
        return false;
    }

    std::string MoEOverlayCollectiveKey::toString() const
    {
        return keyToStableString(*this);
    }

    bool operator==(const MoEOverlayCollectiveKey &lhs, const MoEOverlayCollectiveKey &rhs)
    {
        return lhs.generation_id == rhs.generation_id &&
               lhs.step_id == rhs.step_id &&
               lhs.key_namespace == rhs.key_namespace &&
               lhs.histogram_source == rhs.histogram_source &&
               lhs.mtp_depth == rhs.mtp_depth &&
               lhs.layer_idx == rhs.layer_idx &&
               lhs.tier_idx == rhs.tier_idx &&
               lhs.domain_id == rhs.domain_id &&
               lhs.participant_id == rhs.participant_id &&
               lhs.direction == rhs.direction &&
               lhs.sequence == rhs.sequence;
    }

    bool operator!=(const MoEOverlayCollectiveKey &lhs, const MoEOverlayCollectiveKey &rhs)
    {
        return !(lhs == rhs);
    }

    bool operator<(const MoEOverlayCollectiveKey &lhs, const MoEOverlayCollectiveKey &rhs)
    {
        return std::tie(lhs.generation_id,
                        lhs.step_id,
                        lhs.key_namespace,
                        lhs.histogram_source,
                        lhs.mtp_depth,
                        lhs.layer_idx,
                        lhs.tier_idx,
                        lhs.domain_id,
                        lhs.participant_id,
                        lhs.direction,
                        lhs.sequence) <
               std::tie(rhs.generation_id,
                        rhs.step_id,
                        rhs.key_namespace,
                        rhs.histogram_source,
                        rhs.mtp_depth,
                        rhs.layer_idx,
                        rhs.tier_idx,
                        rhs.domain_id,
                        rhs.participant_id,
                        rhs.direction,
                        rhs.sequence);
    }

    MoEOverlayCollectiveKey makeMoEOverlayCollectiveKey(
        uint64_t generation_id,
        uint64_t step_id,
        int layer_idx,
        int tier_idx,
        int domain_id,
        int participant_id,
        MoEOverlayCollectiveDirection direction)
    {
        MoEOverlayCollectiveKey key;
        key.generation_id = generation_id;
        key.step_id = step_id;
        key.key_namespace = MoEOverlayCollectiveNamespace::Main;
        key.histogram_source = ExpertHistogramSource::DecodeToken;
        key.mtp_depth = -1;
        key.layer_idx = layer_idx;
        key.tier_idx = tier_idx;
        key.domain_id = domain_id;
        key.participant_id = participant_id;
        key.direction = direction;
        key.sequence = makeCollectiveSequence(key.key_namespace,
                                              key.mtp_depth,
                                              key.layer_idx,
                                              key.tier_idx,
                                              key.participant_id,
                                              key.direction);
        return key;
    }

    MoEOverlayCollectiveKey makeMTPMoEOverlayCollectiveKey(
        uint64_t generation_id,
        uint64_t decode_step_id,
        int mtp_depth,
        int layer_idx,
        int tier_idx,
        int domain_id,
        int participant_id,
        MoEOverlayCollectiveDirection direction)
    {
        MoEOverlayCollectiveKey key;
        key.generation_id = generation_id;
        key.step_id = decode_step_id;
        key.key_namespace = MoEOverlayCollectiveNamespace::MTP;
        key.histogram_source = ExpertHistogramSource::GroupedVerifier;
        key.mtp_depth = mtp_depth;
        key.layer_idx = layer_idx;
        key.tier_idx = tier_idx;
        key.domain_id = domain_id;
        key.participant_id = participant_id;
        key.direction = direction;
        key.sequence = makeCollectiveSequence(key.key_namespace,
                                              key.mtp_depth,
                                              key.layer_idx,
                                              key.tier_idx,
                                              key.participant_id,
                                              key.direction);
        return key;
    }

    size_t denseMoEOverlayDispatchBytes(int seq_len, int top_k, int d_model)
    {
        if (seq_len <= 0 || top_k <= 0 || d_model <= 0)
            return 0;

        const size_t rows = static_cast<size_t>(seq_len);
        const size_t hidden_row_bytes = static_cast<size_t>(d_model) * sizeof(float);
        const size_t routing_row_bytes = static_cast<size_t>(top_k) * 2u * sizeof(float);
        return rows * (hidden_row_bytes + routing_row_bytes);
    }

    size_t denseMoEOverlayReturnBytes(int seq_len, int d_model)
    {
        if (seq_len <= 0 || d_model <= 0)
            return 0;

        return static_cast<size_t>(seq_len) * static_cast<size_t>(d_model) * sizeof(float);
    }

    size_t compactMoEOverlayDispatchBytes(const MoEOverlaySparseRows &rows)
    {
        if (rows.d_model <= 0)
            return 0;

        /* Keep host-side transport accounting byte-identical to the device
         * packet ABI. In particular, an empty contribution advances only the
         * timeline: its dormant CSR sentinel is setup-owned capacity, not a
         * published payload byte. */
        return static_cast<size_t>(moeOverlayDispatchPayloadBytes(
            static_cast<std::uint64_t>(rows.live_row_count),
            static_cast<std::uint64_t>(rows.live_entry_count),
            static_cast<std::uint32_t>(rows.d_model)));
    }

    size_t compactMoEOverlayReturnBytes(const MoEOverlayReturnRows &rows)
    {
        if (rows.d_model <= 0)
            return 0;

        return rows.live_row_count * sizeof(int32_t) +
               rows.live_row_count * static_cast<size_t>(rows.d_model) * sizeof(float);
    }

    size_t canonicalMoEOverlayTicketReturnBytes(
        size_t live_entry_count,
        int d_model) noexcept
    {
        if (live_entry_count == 0u || d_model <= 0)
            return 0u;

        constexpr size_t kSlotBytes = 2u * sizeof(std::int32_t);
        const size_t width = static_cast<size_t>(d_model);
        if (width >
            (std::numeric_limits<size_t>::max() - kSlotBytes) /
                sizeof(float))
        {
            return 0u;
        }
        const size_t bytes_per_entry =
            kSlotBytes + width * sizeof(float);
        if (live_entry_count >
            (std::numeric_limits<size_t>::max() -
             sizeof(MoEOverlayCanonicalRouteTicketControl)) /
                bytes_per_entry)
        {
            return 0u;
        }
        return sizeof(MoEOverlayCanonicalRouteTicketControl) +
               live_entry_count * bytes_per_entry;
    }

    MoEOverlaySparseTransferCounters measureMoEOverlaySparseTransferCounters(
        int seq_len,
        int top_k,
        int d_model,
        const MoEOverlaySparseRows *dispatch_rows,
        const MoEOverlayReturnRows *return_rows)
    {
        MoEOverlaySparseTransferCounters counters;
        counters.dense_dispatch_bytes = denseMoEOverlayDispatchBytes(seq_len, top_k, d_model);
        counters.dense_return_bytes = denseMoEOverlayReturnBytes(seq_len, d_model);

        if (dispatch_rows)
        {
            counters.compact_dispatch_bytes = compactMoEOverlayDispatchBytes(*dispatch_rows);
            counters.compact_row_count = dispatch_rows->live_row_count;
            counters.compact_entry_count = dispatch_rows->live_entry_count;
        }
        if (return_rows)
        {
            counters.compact_return_bytes = compactMoEOverlayReturnBytes(*return_rows);
            counters.compact_row_count = std::max(counters.compact_row_count, return_rows->live_row_count);
        }

        return counters;
    }

    MoEOverlayCollectiveWorkspace::MoEOverlayCollectiveWorkspace(
        FixedCapacityConfig config)
        : max_rows_(config.max_rows),
          max_entries_(config.max_entries),
          d_model_(config.d_model),
          top_k_(config.top_k),
          device_(config.device),
          reuse_policy_(config.reuse_policy),
          fixed_capacity_(true)
    {
        if (max_rows_ == 0 || max_entries_ == 0 || d_model_ <= 0 ||
            top_k_ <= 0 || !device_.is_valid())
        {
            throw std::invalid_argument(
                "Fixed ExpertOverlay collective workspace requires positive "
                "geometry and a valid device");
        }
    }

    void MoEOverlayCollectiveWorkspace::ensureCapacity(size_t max_rows,
                                                       size_t max_entries,
                                                       int d_model,
                                                       int top_k,
                                                       DeviceId device)
    {
        if (fixed_capacity_)
        {
            if (max_rows != max_rows_ || max_entries != max_entries_ ||
                d_model != d_model_ || top_k != top_k_ || device != device_)
            {
                throw std::logic_error(
                    "Fixed ExpertOverlay collective workspace cannot be rebound");
            }
            return;
        }

        max_rows_ = std::max(max_rows_, max_rows);
        max_entries_ = std::max(max_entries_, max_entries);
        d_model_ = std::max(d_model_, d_model);
        top_k_ = std::max(top_k_, top_k);
        device_ = device;

        for (auto &[_, buffers] : buffers_by_layer_tier_)
        {
            ensureSparseStorage(buffers.dispatch_receive);
            ensureSparseStorage(buffers.local_expert_input);
            ensureReturnStorage(buffers.local_expert_output);
            ensureReturnStorage(buffers.return_receive);
        }
    }

    void MoEOverlayCollectiveWorkspace::resetForStep(uint64_t generation_id,
                                                     uint64_t step_id)
    {
        generation_id_ = generation_id;
        step_id_ = step_id;

        for (auto &[_, buffers] : buffers_by_layer_tier_)
        {
            if (!buffers.dispatch_receive.entry_offsets_host.empty())
                buffers.dispatch_receive.entry_offsets_host[0] = 0;
            if (!buffers.local_expert_input.entry_offsets_host.empty())
                buffers.local_expert_input.entry_offsets_host[0] = 0;
        }
    }

    MoEOverlayCollectiveWorkspace::LayerTierBuffers &
    MoEOverlayCollectiveWorkspace::buffersFor(int layer_idx, int tier_idx)
    {
        /*
         * A participant graph executes dispatch, local compute, and return in
         * strict graph order. Its typed SerialGraphFamily policy therefore
         * aliases all semantic keys to one physical slot. Other callers retain
         * the full key so independently runnable nodes can never overlap.
         */
        const std::pair<int, int> storage_key =
            reuse_policy_ == StorageReusePolicy::SerialGraphFamily
                ? std::pair<int, int>{0, 0}
                : std::pair<int, int>{layer_idx, tier_idx};
        auto &buffers = buffers_by_layer_tier_[storage_key];
        ensureSparseStorage(buffers.dispatch_receive);
        ensureSparseStorage(buffers.local_expert_input);
        ensureReturnStorage(buffers.local_expert_output);
        ensureReturnStorage(buffers.return_receive);
        return buffers;
    }

    void MoEOverlayCollectiveWorkspace::ensureSparseStorage(SparseStorage &storage)
    {
        if (max_rows_ == 0 || max_entries_ == 0 || d_model_ <= 0 || top_k_ <= 0)
            return;

        if (storage.row_ids_host.size() < max_rows_)
            storage.row_ids_host.resize(max_rows_);
        if (storage.entry_offsets_host.size() < max_rows_ + 1u)
            storage.entry_offsets_host.resize(max_rows_ + 1u, 0);
        if (storage.expert_ids_host.size() < max_entries_)
            storage.expert_ids_host.resize(max_entries_);
        if (storage.route_weights_host.size() < max_entries_)
            storage.route_weights_host.resize(max_entries_);
        if (storage.original_route_slots_host.size() < max_entries_)
            storage.original_route_slots_host.resize(max_entries_);
        if (storage.compact_route_slots_host.size() < max_entries_)
            storage.compact_route_slots_host.resize(max_entries_);
        const size_t hidden_count = max_rows_ * static_cast<size_t>(d_model_);
        if (storage.hidden_rows_fp32.size() < hidden_count)
            storage.hidden_rows_fp32.resize(hidden_count);
    }

    void MoEOverlayCollectiveWorkspace::ensureReturnStorage(ReturnStorage &storage)
    {
        if (max_rows_ == 0 || d_model_ <= 0)
            return;

        if (storage.row_ids_host.size() < max_rows_)
            storage.row_ids_host.resize(max_rows_);
        const size_t output_count = max_rows_ * static_cast<size_t>(d_model_);
        if (storage.output_rows_fp32.size() < output_count)
            storage.output_rows_fp32.resize(output_count);
    }

    MoEOverlaySparseRows MoEOverlayCollectiveWorkspace::dispatchReceive(int layer_idx, int tier_idx)
    {
        auto &storage = buffersFor(layer_idx, tier_idx).dispatch_receive;
        MoEOverlaySparseRows view;
        view.d_model = d_model_;
        view.top_k = top_k_;
        view.row_capacity = storage.row_ids_host.size();
        view.entry_capacity = storage.expert_ids_host.size();
        view.hidden_row_capacity = storage.row_ids_host.size();
        view.hidden_payload_layout =
            MoEOverlayActivationHiddenPayloadLayout::CompactRows;
        view.row_ids_host = storage.row_ids_host.data();
        view.entry_offsets_host = storage.entry_offsets_host.data();
        view.expert_ids_host = storage.expert_ids_host.data();
        view.route_weights_host = storage.route_weights_host.data();
        view.original_route_slots_host =
            storage.original_route_slots_host.data();
        view.compact_route_slots_host =
            storage.compact_route_slots_host.data();
        view.hidden_rows_fp32 = storage.hidden_rows_fp32.data();
        view.live_row_count = 0;
        view.live_entry_count = 0;
        if (view.entry_offsets_host && view.row_capacity > 0)
            view.entry_offsets_host[0] = 0;
        return view;
    }

    MoEOverlaySparseRows MoEOverlayCollectiveWorkspace::localExpertInput(int layer_idx, int tier_idx)
    {
        auto &storage = buffersFor(layer_idx, tier_idx).local_expert_input;
        MoEOverlaySparseRows view;
        view.d_model = d_model_;
        view.top_k = top_k_;
        view.row_capacity = storage.row_ids_host.size();
        view.entry_capacity = storage.expert_ids_host.size();
        view.hidden_row_capacity = storage.row_ids_host.size();
        view.hidden_payload_layout =
            MoEOverlayActivationHiddenPayloadLayout::CompactRows;
        view.row_ids_host = storage.row_ids_host.data();
        view.entry_offsets_host = storage.entry_offsets_host.data();
        view.expert_ids_host = storage.expert_ids_host.data();
        view.route_weights_host = storage.route_weights_host.data();
        view.original_route_slots_host =
            storage.original_route_slots_host.data();
        view.compact_route_slots_host =
            storage.compact_route_slots_host.data();
        view.hidden_rows_fp32 = storage.hidden_rows_fp32.data();
        view.live_row_count = 0;
        view.live_entry_count = 0;
        if (view.entry_offsets_host && view.row_capacity > 0)
            view.entry_offsets_host[0] = 0;
        return view;
    }

    MoEOverlayReturnRows MoEOverlayCollectiveWorkspace::localExpertOutput(int layer_idx, int tier_idx)
    {
        auto &storage = buffersFor(layer_idx, tier_idx).local_expert_output;
        MoEOverlayReturnRows view;
        view.d_model = d_model_;
        view.row_capacity = storage.row_ids_host.size();
        view.row_ids_host = storage.row_ids_host.data();
        view.output_rows_fp32 = storage.output_rows_fp32.data();
        view.live_row_count = 0;
        return view;
    }

    MoEOverlayReturnRows MoEOverlayCollectiveWorkspace::returnReceive(int layer_idx, int tier_idx)
    {
        auto &storage = buffersFor(layer_idx, tier_idx).return_receive;
        MoEOverlayReturnRows view;
        view.d_model = d_model_;
        view.row_capacity = storage.row_ids_host.size();
        view.row_ids_host = storage.row_ids_host.data();
        view.output_rows_fp32 = storage.output_rows_fp32.data();
        view.live_row_count = 0;
        return view;
    }

    MoEOverlayRankLocalSparseCollectiveContext::
        MoEOverlayRankLocalSparseCollectiveContext(Config config)
        : dispatch_slots_(config.slot_count),
          return_slots_(config.slot_count)
    {
        if (config.slot_count == 0u)
        {
            throw std::invalid_argument(
                "rank-local sparse collective requires slot_count > 0");
        }
    }

    std::vector<MoEOverlayRankLocalSparseCollectiveContext::ReplaySlot> &
    MoEOverlayRankLocalSparseCollectiveContext::ledgerFor(
        const MoEOverlayCollectiveKey &key) noexcept
    {
        return key.direction == MoEOverlayCollectiveDirection::Dispatch
                   ? dispatch_slots_
                   : return_slots_;
    }

    MoEOverlayCollectiveResult
    MoEOverlayRankLocalSparseCollectiveContext::dispatch(
        const MoEOverlayCollectiveKey &key,
        const MoEOverlaySparseRows &outbound,
        MoEOverlaySparseRows *inbound,
        IDeviceContext *)
    {
        MoEOverlayCollectiveResult result;
        std::string validation_error;
        if (key.direction != MoEOverlayCollectiveDirection::Dispatch ||
            !key.isValid() || outbound.key != key ||
            outbound.target_participant != key.participant_id ||
            outbound.source_participant < 0 ||
            !validateSparseRows(outbound, &validation_error) ||
            !ensureSparseInboundCapacity(
                inbound,
                outbound.live_row_count,
                outbound.live_entry_count,
                &validation_error))
        {
            result.ok = false;
            result.error_code = 1;
            result.error = validation_error.empty()
                               ? "invalid rank-local sparse dispatch contract"
                               : validation_error;
            return result;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        auto &ledger = ledgerFor(key);
        auto &slot = ledger[static_cast<size_t>(key.sequence) % ledger.size()];
        if (slot.aborted && *slot.aborted == key)
        {
            result.ok = false;
            result.error_code = slot.abort_reason == 0 ? 2 : slot.abort_reason;
            result.error = "rank-local sparse dispatch key was aborted";
            return result;
        }
        if (slot.completed && *slot.completed == key)
        {
            result.ok = false;
            result.error_code = 3;
            result.error = "stale rank-local sparse dispatch key reuse rejected";
            return result;
        }

        /* Both views are model-lifetime fixed storage. memmove deliberately
         * permits the zero-copy alias used by a colocated CPU endpoint while
         * preserving the same code path for distinct diagnostic workspaces. */
        inbound->key = key;
        inbound->residency_epoch = outbound.residency_epoch;
        inbound->source_participant = outbound.source_participant;
        inbound->target_participant = outbound.target_participant;
        inbound->d_model = outbound.d_model;
        inbound->top_k = outbound.top_k;
        inbound->live_row_count = outbound.live_row_count;
        inbound->live_entry_count = outbound.live_entry_count;
        if (inbound->row_ids_host != outbound.row_ids_host)
        {
            std::memmove(
                inbound->row_ids_host,
                outbound.row_ids_host,
                outbound.live_row_count * sizeof(std::int32_t));
        }
        if (inbound->entry_offsets_host != outbound.entry_offsets_host)
        {
            std::memmove(
                inbound->entry_offsets_host,
                outbound.entry_offsets_host,
                (outbound.live_row_count + 1u) * sizeof(std::int32_t));
        }
        if (inbound->expert_ids_host != outbound.expert_ids_host)
        {
            std::memmove(
                inbound->expert_ids_host,
                outbound.expert_ids_host,
                outbound.live_entry_count * sizeof(std::int32_t));
        }
        if (inbound->route_weights_host != outbound.route_weights_host)
        {
            std::memmove(
                inbound->route_weights_host,
                outbound.route_weights_host,
                outbound.live_entry_count * sizeof(float));
        }
        if (inbound->original_route_slots_host !=
            outbound.original_route_slots_host)
        {
            std::memmove(
                inbound->original_route_slots_host,
                outbound.original_route_slots_host,
                outbound.live_entry_count * sizeof(std::int32_t));
        }
        if (inbound->compact_route_slots_host !=
            outbound.compact_route_slots_host)
        {
            std::memmove(
                inbound->compact_route_slots_host,
                outbound.compact_route_slots_host,
                outbound.live_entry_count * sizeof(std::int32_t));
        }
        if (inbound->hidden_rows_fp32 != outbound.hidden_rows_fp32)
        {
            std::memmove(
                inbound->hidden_rows_fp32,
                outbound.hidden_rows_fp32,
                outbound.live_row_count *
                    static_cast<size_t>(outbound.d_model) * sizeof(float));
        }
        if (!validateSparseRows(*inbound, &validation_error))
        {
            result.ok = false;
            result.error_code = 4;
            result.error = validation_error;
            return result;
        }

        slot.aborted.reset();
        slot.abort_reason = 0;
        slot.completed = key;
        result.collective_complete = true;
        return result;
    }

    MoEOverlayCollectiveResult
    MoEOverlayRankLocalSparseCollectiveContext::returnReduce(
        const MoEOverlayCollectiveKey &key,
        const MoEOverlayReturnRows &outbound,
        MoEOverlayReturnRows *inbound,
        IDeviceContext *)
    {
        MoEOverlayCollectiveResult result;
        std::string validation_error;
        if (key.direction !=
                MoEOverlayCollectiveDirection::ReturnReduce ||
            !key.isValid() || outbound.key != key ||
            outbound.source_participant != key.participant_id ||
            outbound.target_participant < 0 ||
            !validateReturnRows(outbound, &validation_error) ||
            !ensureReturnInboundCapacity(
                inbound,
                outbound.live_row_count,
                &validation_error))
        {
            result.ok = false;
            result.error_code = 1;
            result.error = validation_error.empty()
                               ? "invalid rank-local sparse return contract"
                               : validation_error;
            return result;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        auto &ledger = ledgerFor(key);
        auto &slot = ledger[static_cast<size_t>(key.sequence) % ledger.size()];
        if (slot.aborted && *slot.aborted == key)
        {
            result.ok = false;
            result.error_code = slot.abort_reason == 0 ? 2 : slot.abort_reason;
            result.error = "rank-local sparse return key was aborted";
            return result;
        }
        if (slot.completed && *slot.completed == key)
        {
            result.ok = false;
            result.error_code = 3;
            result.error = "stale rank-local sparse return key reuse rejected";
            return result;
        }

        inbound->key = key;
        inbound->residency_epoch = outbound.residency_epoch;
        inbound->source_participant = outbound.source_participant;
        inbound->target_participant = outbound.target_participant;
        inbound->d_model = outbound.d_model;
        inbound->live_row_count = outbound.live_row_count;
        if (inbound->row_ids_host != outbound.row_ids_host)
        {
            std::memmove(
                inbound->row_ids_host,
                outbound.row_ids_host,
                outbound.live_row_count * sizeof(std::int32_t));
        }
        if (inbound->output_rows_fp32 != outbound.output_rows_fp32)
        {
            std::memmove(
                inbound->output_rows_fp32,
                outbound.output_rows_fp32,
                outbound.live_row_count *
                    static_cast<size_t>(outbound.d_model) * sizeof(float));
        }
        if (!validateReturnRows(*inbound, &validation_error))
        {
            result.ok = false;
            result.error_code = 4;
            result.error = validation_error;
            return result;
        }

        slot.aborted.reset();
        slot.abort_reason = 0;
        slot.completed = key;
        result.collective_complete = true;
        return result;
    }

    void MoEOverlayRankLocalSparseCollectiveContext::abort(
        const MoEOverlayCollectiveKey &key,
        int reason_code)
    {
        if (!key.isValid())
            return;
        std::lock_guard<std::mutex> lock(mutex_);
        auto &ledger = ledgerFor(key);
        auto &slot = ledger[static_cast<size_t>(key.sequence) % ledger.size()];
        slot.completed.reset();
        slot.aborted = key;
        slot.abort_reason = reason_code == 0 ? 1 : reason_code;
    }

    struct MoEOverlayLocalSparseCollectiveContext::Slot
    {
        explicit Slot(int participant_count)
            : seen(static_cast<size_t>(participant_count), false),
              dispatch_payloads(static_cast<size_t>(participant_count)),
              return_payloads(static_cast<size_t>(participant_count))
        {
        }

        std::mutex mutex;
        bool active = false;
        MoEOverlayCollectiveKey key;
        int seen_count = 0;
        std::vector<bool> seen;
        std::vector<DispatchPayloadCopy> dispatch_payloads;
        std::vector<ReturnPayloadCopy> return_payloads;

        void reset()
        {
            active = false;
            key = {};
            seen_count = 0;
            std::fill(seen.begin(), seen.end(), false);
            for (auto &payload : dispatch_payloads)
                payload = {};
            for (auto &payload : return_payloads)
                payload = {};
        }
    };

    MoEOverlayLocalSparseCollectiveContext::MoEOverlayLocalSparseCollectiveContext(Config config)
        : participant_count_(config.participant_count)
    {
        if (participant_count_ <= 0)
            throw std::invalid_argument("local sparse collective requires participant_count > 0");
        if (config.slot_count == 0)
            throw std::invalid_argument("local sparse collective requires slot_count > 0");

        slots_.reserve(config.slot_count);
        for (size_t slot = 0; slot < config.slot_count; ++slot)
            slots_.push_back(std::make_unique<Slot>(participant_count_));
    }

    MoEOverlayLocalSparseCollectiveContext::~MoEOverlayLocalSparseCollectiveContext() = default;

    MoEOverlayCollectiveResult MoEOverlayLocalSparseCollectiveContext::dispatch(const MoEOverlayCollectiveKey &key,
                                                                                const MoEOverlaySparseRows &outbound,
                                                                                MoEOverlaySparseRows *inbound,
                                                                                IDeviceContext *)
    {
        return publishDispatch(key, outbound, inbound);
    }

    MoEOverlayCollectiveResult MoEOverlayLocalSparseCollectiveContext::returnReduce(const MoEOverlayCollectiveKey &key,
                                                                                    const MoEOverlayReturnRows &outbound,
                                                                                    MoEOverlayReturnRows *inbound,
                                                                                    IDeviceContext *)
    {
        return publishReturn(key, outbound, inbound);
    }

    void MoEOverlayLocalSparseCollectiveContext::abort(const MoEOverlayCollectiveKey &key,
                                                       int reason_code)
    {
        aborted_keys_[keyToStableString(key)] = reason_code == 0 ? 1 : reason_code;
    }

    MoEOverlayCollectiveResult MoEOverlayLocalSparseCollectiveContext::publishDispatch(const MoEOverlayCollectiveKey &key,
                                                                                       const MoEOverlaySparseRows &outbound,
                                                                                       MoEOverlaySparseRows *inbound)
    {
        MoEOverlayCollectiveResult result;
        if (key.direction != MoEOverlayCollectiveDirection::Dispatch)
        {
            result.ok = false;
            result.error_code = 1;
            result.error = "dispatch called with non-dispatch key";
            return result;
        }
        if (!key.isValid())
        {
            result.ok = false;
            result.error_code = 2;
            result.error = "invalid dispatch key";
            return result;
        }

        std::string validation_error;
        if (!validateSparseRows(outbound, &validation_error))
        {
            result.ok = false;
            result.error_code = 3;
            result.error = validation_error;
            return result;
        }

        const auto key_string = keyToStableString(key);
        if (const auto aborted = aborted_keys_.find(key_string); aborted != aborted_keys_.end())
        {
            result.ok = false;
            result.error_code = aborted->second;
            result.error = "dispatch key was aborted";
            return result;
        }
        if (completed_keys_.count(key_string) > 0)
        {
            result.ok = false;
            result.error_code = 4;
            result.error = "stale dispatch key reuse rejected";
            return result;
        }

        if (outbound.source_participant < 0 || outbound.source_participant >= participant_count_)
        {
            result.ok = false;
            result.error_code = 5;
            result.error = "dispatch source participant out of range";
            return result;
        }

        auto &slot = *slots_[static_cast<size_t>(key.sequence) % slots_.size()];
        std::lock_guard<std::mutex> guard(slot.mutex);
        if (!slot.active)
        {
            slot.active = true;
            slot.key = key;
            slot.seen_count = 0;
        }
        else if (slot.key != key)
        {
            result.ok = false;
            result.error_code = 6;
            result.error = "dispatch slot occupied by another key";
            return result;
        }

        const size_t participant = static_cast<size_t>(outbound.source_participant);
        if (slot.seen[participant])
        {
            result.ok = false;
            result.error_code = 7;
            result.error = "duplicate dispatch publish for participant";
            return result;
        }

        slot.seen[participant] = true;
        ++slot.seen_count;
        slot.dispatch_payloads[participant] = copyDispatchPayload(outbound);

        if (inbound)
        {
            inbound->key = key;
            inbound->residency_epoch = 0;
            inbound->source_participant = outbound.source_participant;
            inbound->target_participant = outbound.source_participant;
            inbound->d_model = outbound.d_model;
            inbound->top_k = outbound.top_k;
            inbound->live_row_count = 0;
            inbound->live_entry_count = 0;
            if (inbound->entry_offsets_host && inbound->row_capacity > 0)
                inbound->entry_offsets_host[0] = 0;
        }

        if (slot.seen_count != participant_count_)
            return result;

        result.collective_complete = true;
        const int target = outbound.source_participant;
        if (inbound)
        {
            for (const auto &payload : slot.dispatch_payloads)
            {
                if (payload.target_participant != target)
                    continue;
                std::string append_error;
                if (!appendDispatchInbound(payload, inbound, &append_error))
                {
                    result.ok = false;
                    result.error_code = 8;
                    result.error = append_error;
                    slot.reset();
                    return result;
                }
            }
        }

        completed_keys_.insert(key_string);
        slot.reset();
        return result;
    }

    MoEOverlayCollectiveResult MoEOverlayLocalSparseCollectiveContext::publishReturn(const MoEOverlayCollectiveKey &key,
                                                                                     const MoEOverlayReturnRows &outbound,
                                                                                     MoEOverlayReturnRows *inbound)
    {
        MoEOverlayCollectiveResult result;
        if (key.direction != MoEOverlayCollectiveDirection::ReturnReduce)
        {
            result.ok = false;
            result.error_code = 1;
            result.error = "returnReduce called with non-return key";
            return result;
        }
        if (!key.isValid())
        {
            result.ok = false;
            result.error_code = 2;
            result.error = "invalid return key";
            return result;
        }

        std::string validation_error;
        if (!validateReturnRows(outbound, &validation_error))
        {
            result.ok = false;
            result.error_code = 3;
            result.error = validation_error;
            return result;
        }

        const auto key_string = keyToStableString(key);
        if (const auto aborted = aborted_keys_.find(key_string); aborted != aborted_keys_.end())
        {
            result.ok = false;
            result.error_code = aborted->second;
            result.error = "return key was aborted";
            return result;
        }
        if (completed_keys_.count(key_string) > 0)
        {
            result.ok = false;
            result.error_code = 4;
            result.error = "stale return key reuse rejected";
            return result;
        }

        if (outbound.source_participant < 0 || outbound.source_participant >= participant_count_)
        {
            result.ok = false;
            result.error_code = 5;
            result.error = "return source participant out of range";
            return result;
        }

        auto &slot = *slots_[static_cast<size_t>(key.sequence) % slots_.size()];
        std::lock_guard<std::mutex> guard(slot.mutex);
        if (!slot.active)
        {
            slot.active = true;
            slot.key = key;
            slot.seen_count = 0;
        }
        else if (slot.key != key)
        {
            result.ok = false;
            result.error_code = 6;
            result.error = "return slot occupied by another key";
            return result;
        }

        const size_t participant = static_cast<size_t>(outbound.source_participant);
        if (slot.seen[participant])
        {
            result.ok = false;
            result.error_code = 7;
            result.error = "duplicate return publish for participant";
            return result;
        }

        slot.seen[participant] = true;
        ++slot.seen_count;
        slot.return_payloads[participant] = copyReturnPayload(outbound);

        if (inbound)
        {
            inbound->key = key;
            inbound->residency_epoch = 0;
            inbound->source_participant = outbound.source_participant;
            inbound->target_participant = outbound.source_participant;
            inbound->d_model = outbound.d_model;
            inbound->live_row_count = 0;
        }

        if (slot.seen_count != participant_count_)
            return result;

        result.collective_complete = true;
        const int target = outbound.source_participant;
        if (inbound)
        {
            for (const auto &payload : slot.return_payloads)
            {
                if (payload.target_participant != target)
                    continue;
                std::string append_error;
                if (!appendReturnInbound(payload, inbound, &append_error))
                {
                    result.ok = false;
                    result.error_code = 8;
                    result.error = append_error;
                    slot.reset();
                    return result;
                }
            }
        }

        completed_keys_.insert(key_string);
        slot.reset();
        return result;
    }

    MoEOverlayMPISparseCollectiveContext::MoEOverlayMPISparseCollectiveContext(Config config)
        : config_(std::move(config))
    {
        if (!config_.mpi_ctx)
        {
            throw std::invalid_argument(
                "MPI sparse collective requires a non-null MPI context");
        }

        std::sort(
            config_.local_participant_ids.begin(),
            config_.local_participant_ids.end());
        if (std::any_of(
                config_.local_participant_ids.begin(),
                config_.local_participant_ids.end(),
                [](int participant) { return participant < 0; }))
        {
            throw std::invalid_argument(
                "MPI sparse collective local participant ids must be non-negative");
        }
        if (std::adjacent_find(
                config_.local_participant_ids.begin(),
                config_.local_participant_ids.end()) !=
            config_.local_participant_ids.end())
        {
            throw std::invalid_argument(
                "MPI sparse collective local participant ids must be unique");
        }
    }

    MoEOverlayMPISparseCollectiveContext::~MoEOverlayMPISparseCollectiveContext() = default;

    bool MoEOverlayMPISparseCollectiveContext::ownsLocalParticipant(
        int participant_id) const noexcept
    {
        return std::binary_search(
            config_.local_participant_ids.begin(),
            config_.local_participant_ids.end(),
            participant_id);
    }

    MoEOverlayCollectiveResult MoEOverlayMPISparseCollectiveContext::dispatch(const MoEOverlayCollectiveKey &key,
                                                                              const MoEOverlaySparseRows &outbound,
                                                                              MoEOverlaySparseRows *inbound,
                                                                              IDeviceContext *)
    {
        return dispatchHostStaged(key, outbound, inbound);
    }

    MoEOverlayCollectiveResult MoEOverlayMPISparseCollectiveContext::returnReduce(const MoEOverlayCollectiveKey &key,
                                                                                  const MoEOverlayReturnRows &outbound,
                                                                                  MoEOverlayReturnRows *inbound,
                                                                                  IDeviceContext *)
    {
        return returnHostStaged(key, outbound, inbound);
    }

    void MoEOverlayMPISparseCollectiveContext::abort(const MoEOverlayCollectiveKey &key,
                                                     int reason_code)
    {
        aborted_keys_[keyToStableString(key)] = reason_code == 0 ? 1 : reason_code;
    }

    MoEOverlayCollectiveResult MoEOverlayMPISparseCollectiveContext::dispatchHostStaged(const MoEOverlayCollectiveKey &key,
                                                                                        const MoEOverlaySparseRows &outbound,
                                                                                        MoEOverlaySparseRows *inbound)
    {
        MoEOverlayCollectiveResult result;
        if (!config_.mpi_ctx)
        {
            result.ok = false;
            result.error_code = 1;
            result.error = "MPI sparse collective has no MPI context";
            return result;
        }
        if (key.direction != MoEOverlayCollectiveDirection::Dispatch)
        {
            result.ok = false;
            result.error_code = 2;
            result.error = "dispatch called with non-dispatch key";
            return result;
        }

        const auto key_string = keyToStableString(key);
        if (const auto aborted = aborted_keys_.find(key_string); aborted != aborted_keys_.end())
        {
            result.ok = false;
            result.error_code = aborted->second;
            result.error = "dispatch key was aborted";
            return result;
        }
        if (completed_keys_.count(key_string) > 0)
        {
            result.ok = false;
            result.error_code = 3;
            result.error = "stale dispatch key reuse rejected";
            return result;
        }

        std::string validation_error;
        if (!validateSparseRows(outbound, &validation_error))
        {
            result.ok = false;
            result.error_code = 4;
            result.error = validation_error;
            return result;
        }

        DispatchPayloadCopy outbound_copy = copyDispatchPayload(outbound);
        std::vector<uint8_t> send_packet = serializeDispatchPacket(outbound_copy);
        std::vector<int> recv_counts;
        std::vector<int> displs;
        std::vector<uint8_t> recv_payloads = gatherAllPayloads(*config_.mpi_ctx, send_packet, &recv_counts, &displs);

        if (inbound)
        {
            inbound->key = key;
            inbound->residency_epoch = 0;
            inbound->source_participant = -1;
            inbound->target_participant = -1;
            inbound->d_model = outbound.d_model;
            inbound->top_k = outbound.top_k;
            inbound->live_row_count = 0;
            inbound->live_entry_count = 0;
            if (inbound->entry_offsets_host && inbound->row_capacity > 0)
                inbound->entry_offsets_host[0] = 0;
        }

        for (int rank = 0; rank < config_.mpi_ctx->world_size(); ++rank)
        {
            const int count = recv_counts[static_cast<size_t>(rank)];
            const int disp = displs[static_cast<size_t>(rank)];
            if (count <= 0)
                continue;

            DispatchPayloadCopy packet;
            std::string parse_error;
            if (!deserializeDispatchPacket(recv_payloads.data() + disp,
                                           static_cast<size_t>(count),
                                           &packet,
                                           &parse_error))
            {
                result.ok = false;
                result.error_code = 5;
                result.error = parse_error;
                return result;
            }
            if (packet.key != key)
            {
                result.ok = false;
                result.error_code = 6;
                result.error =
                    "dispatch key mismatch across MPI ranks: expected=" +
                    key.toString() + " received=" +
                    packet.key.toString() + " sender_rank=" +
                    std::to_string(rank);
                return result;
            }

            /*
             * Address packets by participant ownership, never by MPI rank.
             * This is what permits one socket-local process to own a CUDA hot
             * endpoint, a ROCm warm endpoint, and a CPU cold endpoint while
             * still entering this collective exactly once.
             */
            if (inbound &&
                ownsLocalParticipant(packet.target_participant))
            {
                if (inbound->target_participant < 0)
                {
                    inbound->source_participant =
                        packet.source_participant;
                    inbound->target_participant =
                        packet.target_participant;
                }
                else if (inbound->target_participant !=
                         packet.target_participant)
                {
                    result.ok = false;
                    result.error_code = 7;
                    result.error =
                        "dispatch key addressed multiple local participants";
                    return result;
                }
                std::string append_error;
                if (!appendDispatchInbound(packet, inbound, &append_error))
                {
                    result.ok = false;
                    result.error_code = 8;
                    result.error = append_error;
                    return result;
                }
            }
        }

        completed_keys_.insert(key_string);
        result.collective_complete = true;
        return result;
    }

    MoEOverlayCollectiveResult MoEOverlayMPISparseCollectiveContext::returnHostStaged(const MoEOverlayCollectiveKey &key,
                                                                                      const MoEOverlayReturnRows &outbound,
                                                                                      MoEOverlayReturnRows *inbound)
    {
        MoEOverlayCollectiveResult result;
        if (!config_.mpi_ctx)
        {
            result.ok = false;
            result.error_code = 1;
            result.error = "MPI sparse collective has no MPI context";
            return result;
        }
        if (key.direction != MoEOverlayCollectiveDirection::ReturnReduce)
        {
            result.ok = false;
            result.error_code = 2;
            result.error = "returnReduce called with non-return key";
            return result;
        }

        const auto key_string = keyToStableString(key);
        if (const auto aborted = aborted_keys_.find(key_string); aborted != aborted_keys_.end())
        {
            result.ok = false;
            result.error_code = aborted->second;
            result.error = "return key was aborted";
            return result;
        }
        if (completed_keys_.count(key_string) > 0)
        {
            result.ok = false;
            result.error_code = 3;
            result.error = "stale return key reuse rejected";
            return result;
        }

        std::string validation_error;
        if (!validateReturnRows(outbound, &validation_error))
        {
            result.ok = false;
            result.error_code = 4;
            result.error = validation_error;
            return result;
        }

        ReturnPayloadCopy outbound_copy = copyReturnPayload(outbound);
        std::vector<uint8_t> send_packet = serializeReturnPacket(outbound_copy);
        std::vector<int> recv_counts;
        std::vector<int> displs;
        std::vector<uint8_t> recv_payloads = gatherAllPayloads(*config_.mpi_ctx, send_packet, &recv_counts, &displs);

        if (inbound)
        {
            inbound->key = key;
            inbound->residency_epoch = 0;
            inbound->source_participant = -1;
            inbound->target_participant = -1;
            inbound->d_model = outbound.d_model;
            inbound->live_row_count = 0;
        }

        for (int rank = 0; rank < config_.mpi_ctx->world_size(); ++rank)
        {
            const int count = recv_counts[static_cast<size_t>(rank)];
            const int disp = displs[static_cast<size_t>(rank)];
            if (count <= 0)
                continue;

            ReturnPayloadCopy packet;
            std::string parse_error;
            if (!deserializeReturnPacket(recv_payloads.data() + disp,
                                         static_cast<size_t>(count),
                                         &packet,
                                         &parse_error))
            {
                result.ok = false;
                result.error_code = 5;
                result.error = parse_error;
                return result;
            }
            if (packet.key != key)
            {
                result.ok = false;
                result.error_code = 6;
                result.error =
                    "return key mismatch across MPI ranks: expected=" +
                    key.toString() + " received=" +
                    packet.key.toString() + " sender_rank=" +
                    std::to_string(rank);
                return result;
            }

            if (inbound &&
                ownsLocalParticipant(packet.target_participant))
            {
                if (inbound->target_participant < 0)
                {
                    inbound->source_participant =
                        packet.source_participant;
                    inbound->target_participant =
                        packet.target_participant;
                }
                else if (inbound->target_participant !=
                         packet.target_participant)
                {
                    result.ok = false;
                    result.error_code = 7;
                    result.error =
                        "return key addressed multiple local participants";
                    return result;
                }
                std::string append_error;
                if (!appendReturnInbound(packet, inbound, &append_error))
                {
                    result.ok = false;
                    result.error_code = 8;
                    result.error = append_error;
                    return result;
                }
            }
        }

        completed_keys_.insert(key_string);
        result.collective_complete = true;
        return result;
    }

} // namespace llaminar2
