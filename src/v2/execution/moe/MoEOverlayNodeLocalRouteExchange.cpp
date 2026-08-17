/**
 * @file MoEOverlayNodeLocalRouteExchange.cpp
 * @brief Mapped-memory ownership for topology-generic sparse route exchange.
 *
 * Setup validates one exact continuation topology, lays out cache-line/page
 * aligned producer lanes, registers the pages with every endpoint through
 * TransferEngine, and resolves endpoint-specific aliases on demand.  Runtime
 * progress is entirely device-owned by kernels described in the companion ABI;
 * this file contains no inference-time polling, copying, or synchronization.
 */

#include "MoEOverlayNodeLocalRouteExchange.h"

#include "transfer/TransferEngine.h"
#include "utils/PerfStatsCollector.h"

#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace llaminar2
{
    namespace
    {
        constexpr std::size_t kCacheLineBytes = 64u;
        constexpr std::size_t kPageBytes = 4096u;

        /** @return `value` rounded up to `alignment`, rejecting overflow. */
        std::size_t checkedAlignUp(
            std::size_t value,
            std::size_t alignment)
        {
            if (alignment == 0u || (alignment & (alignment - 1u)) != 0u)
            {
                throw std::invalid_argument(
                    "node-local route exchange alignment must be a power of two");
            }
            const std::size_t mask = alignment - 1u;
            if (value > std::numeric_limits<std::size_t>::max() - mask)
            {
                throw std::overflow_error(
                    "node-local route exchange alignment overflows size_t");
            }
            return (value + mask) & ~mask;
        }

        /** @return Checked product of two byte-capacity dimensions. */
        std::size_t checkedMultiply(
            std::size_t left,
            std::size_t right,
            const char *description)
        {
            if (left != 0u &&
                right > std::numeric_limits<std::size_t>::max() / left)
            {
                throw std::overflow_error(
                    std::string("node-local route exchange ") + description +
                    " overflows size_t");
            }
            return left * right;
        }

        /** @return Checked sum of two byte-capacity dimensions. */
        std::size_t checkedAdd(
            std::size_t left,
            std::size_t right,
            const char *description)
        {
            if (right > std::numeric_limits<std::size_t>::max() - left)
            {
                throw std::overflow_error(
                    std::string("node-local route exchange ") + description +
                    " overflows size_t");
            }
            return left + right;
        }

        /** @return Stable key for duplicate-device validation. */
        std::string deviceKey(DeviceId device)
        {
            return device.toString();
        }

        /** @return Whether two endpoint lists describe the same sorted identity. */
        bool sameEndpoints(
            const std::vector<MoENodeLocalRouteEndpoint> &left,
            const std::vector<MoENodeLocalRouteEndpoint> &right)
        {
            return left == right;
        }
    } // namespace

    MoEOverlayNodeLocalRouteExchange::MoEOverlayNodeLocalRouteExchange(
        Config config)
        : config_(std::move(config))
    {
        if (config_.devices.size() < 2u || !config_.root_device.is_gpu())
        {
            throw std::invalid_argument(
                "node-local route exchange requires at least two GPUs and an exact GPU root");
        }

        std::unordered_set<std::string> seen;
        bool root_seen = false;
        for (const DeviceId device : config_.devices)
        {
            if (!device.is_gpu())
            {
                throw std::invalid_argument(
                    "node-local route exchange endpoints must be GPUs: " +
                    device.toString());
            }
            if (!seen.insert(deviceKey(device)).second)
            {
                throw std::invalid_argument(
                    "node-local route exchange contains duplicate endpoint " +
                    device.toString());
            }
            root_seen = root_seen || device == config_.root_device;
        }
        if (!root_seen)
        {
            throw std::invalid_argument(
                "node-local route exchange root is absent from its device set");
        }
        if (config_.identity.empty())
            config_.identity = "node_local_continuation_routes";
    }

    MoEOverlayNodeLocalRouteExchange::~MoEOverlayNodeLocalRouteExchange()
    {
        abortForShutdown();
        // The owning orchestrator drains retained graphs before destruction;
        // release graph-private root VRAM before unregistering shared pages.
        root_staging_.reset();
        // MappedHostTransferRegion unregisters CUDA/HIP pages while the mmap
        // lifetime is still retained. Reversing this order would leave backend
        // registrations referring to unmapped virtual memory.
        mapped_region_.reset();
        mapping_lifetime_.reset();
    }

    void MoEOverlayNodeLocalRouteExchange::materialize(
        std::vector<MoENodeLocalRouteEndpoint> endpoints,
        int root_participant,
        std::uint32_t max_rows,
        std::uint32_t top_k,
        std::uint32_t d_model)
    {
        if (root_participant < 0 || max_rows == 0u || top_k == 0u ||
            d_model == 0u)
        {
            throw std::invalid_argument(
                "node-local route exchange materialization requires positive topology and geometry");
        }
        const std::uint64_t route_capacity_64 =
            static_cast<std::uint64_t>(max_rows) *
            static_cast<std::uint64_t>(top_k);
        if (route_capacity_64 == 0u ||
            route_capacity_64 > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::overflow_error(
                "node-local route exchange route capacity exceeds uint32_t");
        }
        const auto route_capacity =
            static_cast<std::uint32_t>(route_capacity_64);

        std::sort(
            endpoints.begin(), endpoints.end(),
            [](const auto &left, const auto &right)
            {
                return left.participant_id < right.participant_id;
            });
        if (endpoints.size() != config_.devices.size())
        {
            throw std::invalid_argument(
                "node-local route exchange endpoint count differs from its configured device set");
        }

        std::unordered_set<int> participant_ids;
        std::unordered_set<std::string> endpoint_devices;
        bool root_identity_seen = false;
        for (const auto &endpoint : endpoints)
        {
            if (endpoint.participant_id < 0 || !endpoint.device.is_gpu() ||
                !participant_ids.insert(endpoint.participant_id).second ||
                !endpoint_devices.insert(deviceKey(endpoint.device)).second)
            {
                throw std::invalid_argument(
                    "node-local route exchange endpoints require unique non-negative participants and GPUs");
            }
            const bool configured_device = std::find(
                config_.devices.begin(), config_.devices.end(),
                endpoint.device) != config_.devices.end();
            if (!configured_device)
            {
                throw std::invalid_argument(
                    "node-local route exchange endpoint is outside its configured device set: " +
                    endpoint.device.toString());
            }
            if (endpoint.participant_id == root_participant)
            {
                root_identity_seen = endpoint.device == config_.root_device;
            }
        }
        if (!root_identity_seen)
        {
            throw std::invalid_argument(
                "node-local route exchange logical root does not name the configured root device");
        }

        std::lock_guard lock(mutex_);
        if (materialized_)
        {
            if (!sameEndpoints(endpoints_, endpoints) ||
                root_participant_ != root_participant ||
                route_capacity_ != route_capacity || d_model_ != d_model)
            {
                throw std::logic_error(
                    "node-local route exchange cannot change topology or capacity after materialization");
            }
            return;
        }

        std::vector<LaneLayout> layouts;
        layouts.reserve(endpoints.size() - 1u);
        std::size_t cursor = 0u;
        const std::size_t slot_epoch_bytes = checkedMultiply(
            static_cast<std::size_t>(route_capacity),
            sizeof(std::uint64_t),
            "slot-epoch bytes");
        const std::size_t route_elements = checkedMultiply(
            static_cast<std::size_t>(route_capacity),
            static_cast<std::size_t>(d_model),
            "payload elements");
        const std::size_t payload_bytes = checkedMultiply(
            route_elements, sizeof(float), "payload bytes");

        for (const auto &endpoint : endpoints)
        {
            if (endpoint.participant_id == root_participant)
                continue;

            cursor = checkedAlignUp(cursor, kPageBytes);
            LaneLayout layout;
            layout.producer_device = endpoint.device;
            layout.producer_participant = endpoint.participant_id;
            layout.control_offset = cursor;
            cursor = checkedAdd(
                cursor, sizeof(MoENodeLocalRouteLaneControl),
                "control layout");
            cursor = checkedAlignUp(cursor, kCacheLineBytes);
            layout.slot_epochs_offset = cursor;
            cursor = checkedAdd(
                cursor, slot_epoch_bytes, "slot-epoch layout");
            cursor = checkedAlignUp(cursor, kPageBytes);
            layout.payload_offset = cursor;
            cursor = checkedAdd(cursor, payload_bytes, "payload layout");
            layout.root_staging_offset = checkedMultiply(
                layouts.size(), payload_bytes, "root staging offset");
            layouts.push_back(layout);
        }
        const std::size_t mapping_bytes = checkedAlignUp(cursor, kPageBytes);

        void *const mapping = ::mmap(
            nullptr,
            mapping_bytes,
            PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_ANONYMOUS,
            -1,
            0);
        if (mapping == MAP_FAILED)
        {
            throw std::runtime_error(
                "node-local route exchange mmap failed for " +
                std::to_string(mapping_bytes) + " bytes");
        }
        auto lifetime = std::shared_ptr<void>(
            mapping,
            [mapping_bytes](void *address)
            {
                if (address && address != MAP_FAILED)
                    (void)::munmap(address, mapping_bytes);
            });

        // Transparent hugepage backing reduces IOMMU/page-walk pressure for
        // route payloads while retaining ordinary mmap teardown semantics.
#if defined(MADV_HUGEPAGE)
        (void)::madvise(mapping, mapping_bytes, MADV_HUGEPAGE);
#endif
        std::memset(mapping, 0, mapping_bytes);

        TransferEngine transfer_engine;
        auto region = transfer_engine.registerExternalMappedHostRegion(
            mapping,
            mapping_bytes,
            config_.devices,
            lifetime);
        if (!region || !region->isBound())
        {
            throw std::runtime_error(
                "node-local route exchange could not bind every device alias");
        }
        const std::size_t root_staging_bytes = checkedMultiply(
            payload_bytes, layouts.size(), "root staging bytes");
        auto root_staging = transfer_engine.allocateDeviceTransferBuffer(
            root_staging_bytes, config_.root_device);
        if (!root_staging || !root_staging->isBound())
        {
            throw std::runtime_error(
                "node-local route exchange could not allocate stable root DMA scratch");
        }

        for (const auto &layout : layouts)
        {
            auto *const control = static_cast<MoENodeLocalRouteLaneControl *>(
                region->mutableHostData(layout.control_offset));
            *control = MoENodeLocalRouteLaneControl{
                .producer_participant = layout.producer_participant,
                .root_participant = root_participant,
                .route_capacity = route_capacity,
                .d_model = d_model,
            };
        }

        endpoints_ = std::move(endpoints);
        lanes_ = std::move(layouts);
        root_participant_ = root_participant;
        route_capacity_ = route_capacity;
        d_model_ = d_model;
        mapping_bytes_ = mapping_bytes;
        mapping_lifetime_ = std::move(lifetime);
        mapped_region_ = std::move(region);
        root_staging_ = std::move(root_staging);
        materialized_ = true;

        PerfStatsCollector::addCounter(
            "moe_overlay",
            "node_local_sparse_route_exchange_bytes",
            static_cast<double>(mapping_bytes_),
            "graph_build",
            config_.root_device.toString(),
            {{"identity", config_.identity},
             {"participants", std::to_string(endpoints_.size())},
             {"peer_lanes", std::to_string(lanes_.size())},
             {"route_capacity", std::to_string(route_capacity_)},
             {"d_model", std::to_string(d_model_)},
             {"root_staging_bytes",
              std::to_string(root_staging_bytes)}});
    }

    bool MoEOverlayNodeLocalRouteExchange::materialized() const noexcept
    {
        std::lock_guard lock(mutex_);
        return materialized_;
    }

    int MoEOverlayNodeLocalRouteExchange::rootParticipant() const noexcept
    {
        std::lock_guard lock(mutex_);
        return root_participant_;
    }

    std::uint32_t MoEOverlayNodeLocalRouteExchange::routeCapacity() const noexcept
    {
        std::lock_guard lock(mutex_);
        return route_capacity_;
    }

    std::uint32_t MoEOverlayNodeLocalRouteExchange::dModel() const noexcept
    {
        std::lock_guard lock(mutex_);
        return d_model_;
    }

    MoENodeLocalRoutePeerDeviceBinding
    MoEOverlayNodeLocalRouteExchange::bindingLocked(
        const LaneLayout &lane,
        DeviceId alias_device) const
    {
        if (!materialized_ || !mapped_region_ ||
            !mapped_region_->hasDevice(alias_device))
        {
            throw std::logic_error(
                "node-local route exchange binding requested before complete materialization");
        }
        auto *const mapped_route_payload = static_cast<float *>(
            mapped_region_->deviceAlias(
                alias_device, lane.payload_offset));
        float *route_payload = nullptr;
        if (alias_device == config_.root_device)
        {
            if (!root_staging_ || !root_staging_->isBound())
            {
                throw std::logic_error(
                    "node-local route exchange root scratch is unavailable");
            }
            route_payload = static_cast<float *>(
                root_staging_->mutableDeviceData(
                    lane.root_staging_offset));
        }
        else
        {
            route_payload = mapped_route_payload;
        }
        return MoENodeLocalRoutePeerDeviceBinding{
            .control = static_cast<MoENodeLocalRouteLaneControl *>(
                mapped_region_->deviceAlias(
                    alias_device, lane.control_offset)),
            .slot_epochs = static_cast<std::uint64_t *>(
                mapped_region_->deviceAlias(
                    alias_device, lane.slot_epochs_offset)),
            .route_payload = route_payload,
            .mapped_route_payload = mapped_route_payload,
            .producer_participant = lane.producer_participant,
            .route_capacity = route_capacity_,
            .d_model = d_model_,
        };
    }

    MoENodeLocalRoutePeerDeviceBinding
    MoEOverlayNodeLocalRouteExchange::producerBinding(
        DeviceId producer_device) const
    {
        std::lock_guard lock(mutex_);
        const auto lane = std::find_if(
            lanes_.begin(), lanes_.end(),
            [&](const LaneLayout &candidate)
            {
                return candidate.producer_device == producer_device;
            });
        if (lane == lanes_.end())
        {
            throw std::logic_error(
                "node-local route exchange has no producer lane for " +
                producer_device.toString());
        }
        return bindingLocked(*lane, producer_device);
    }

    std::vector<MoENodeLocalRoutePeerDeviceBinding>
    MoEOverlayNodeLocalRouteExchange::rootPeerBindings(
        DeviceId root_device) const
    {
        std::lock_guard lock(mutex_);
        if (root_device != config_.root_device)
        {
            throw std::logic_error(
                "node-local route exchange root binding requested from " +
                root_device.toString() + " instead of " +
                config_.root_device.toString());
        }
        std::vector<MoENodeLocalRoutePeerDeviceBinding> result;
        result.reserve(lanes_.size());
        for (const auto &lane : lanes_)
            result.push_back(bindingLocked(lane, root_device));
        std::sort(
            result.begin(), result.end(),
            [](const auto &left, const auto &right)
            {
                return left.producer_participant < right.producer_participant;
            });
        return result;
    }

    std::vector<MoENodeLocalRouteEndpoint>
    MoEOverlayNodeLocalRouteExchange::endpoints() const
    {
        std::lock_guard lock(mutex_);
        return endpoints_;
    }

    std::string MoEOverlayNodeLocalRouteExchange::diagnostics() const
    {
        std::lock_guard lock(mutex_);
        std::ostringstream out;
        out << "identity=" << config_.identity
            << " root_device=" << config_.root_device.toString()
            << " root_participant=" << root_participant_
            << " materialized=" << (materialized_ ? "true" : "false")
            << " route_capacity=" << route_capacity_
            << " d_model=" << d_model_
            << " mapping_bytes=" << mapping_bytes_
            << " root_staging_bytes="
            << (root_staging_ ? root_staging_->sizeBytes() : 0u)
            << " endpoints=[";
        for (std::size_t index = 0u; index < endpoints_.size(); ++index)
        {
            if (index != 0u)
                out << ',';
            out << endpoints_[index].participant_id << '@'
                << endpoints_[index].device.toString();
        }
        out << ']';
        return out.str();
    }

    void MoEOverlayNodeLocalRouteExchange::abortForShutdown() noexcept
    {
        try
        {
            std::lock_guard lock(mutex_);
            if (!materialized_ || !mapped_region_)
                return;
            for (const auto &lane : lanes_)
            {
                auto *const control =
                    static_cast<MoENodeLocalRouteLaneControl *>(
                        mapped_region_->mutableHostData(
                            lane.control_offset));
                if (!control)
                    continue;

                // Publish semantic failure before the terminal epochs. A
                // waiting device that observes either order still rechecks the
                // state/code and exits without consuming payload bytes.
                std::atomic_ref<std::uint32_t>(control->code).store(
                    static_cast<std::uint32_t>(
                        MoENodeLocalRouteExchangeCode::PeerAborted),
                    std::memory_order_release);
                std::atomic_ref<std::uint32_t>(control->state).store(
                    static_cast<std::uint32_t>(
                        MoENodeLocalRouteExchangeState::Aborted),
                    std::memory_order_release);
                std::atomic_ref<std::uint64_t>(control->produced_epoch).store(
                    kMoENodeLocalRouteExchangeAbortEpoch,
                    std::memory_order_release);
                std::atomic_ref<std::uint64_t>(control->consumed_epoch).store(
                    kMoENodeLocalRouteExchangeAbortEpoch,
                    std::memory_order_release);
            }
        }
        catch (...)
        {
            // Destructors and failure unwinding cannot report a secondary
            // mutex/runtime failure; mapped-region lifetime is still retained.
        }
    }
} // namespace llaminar2
