/**
 * @file GpuExpertSlotPool.cpp
 * @brief Epoch-aware lease accounting for persistent GPU expert slots.
 *
 * Physical device addresses are allocated once by `LoadOrchestrator`. This
 * implementation changes only logical assignment under a mutex; destruction
 * of the last lease alias returns a slot without freeing device storage.
 */

#include "execution/moe/GpuExpertSlotPool.h"

#include "../../backends/IBackend.h"
#include "../../loaders/gpu_pipeline/LoadOrchestrator.h"
#include "../../utils/Logger.h"
#include "../../utils/PerfStatsCollector.h"
#include "../../utils/VramBillOfMaterials.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        struct SlotLeaseToken
        {
            int slot_index = -1;
            int expert_id = -1;
            uint64_t residency_epoch = 0;
            bool transfer_slot = false;
            std::shared_ptr<LoadOrchestrator> orchestrator;
            std::weak_ptr<GpuExpertSlotPool> pool;
        };
    } // namespace

    std::shared_ptr<GpuExpertSlotPool> GpuExpertSlotPool::create(
        IBackend *backend,
        DeviceId device,
        int device_ordinal,
        int layer_idx,
        int active_capacity,
        std::vector<ProjectionSpec> specs,
        size_t vram_safety_margin_bytes,
        int transfer_capacity)
    {
        if (active_capacity <= 0)
            throw std::invalid_argument("GpuExpertSlotPool active capacity must be positive");
        if (transfer_capacity < 0)
            throw std::invalid_argument("GpuExpertSlotPool transfer capacity cannot be negative");
        if (specs.empty())
            throw std::invalid_argument("GpuExpertSlotPool requires at least one projection spec");

        auto orchestrator = std::make_shared<LoadOrchestrator>(backend);
        orchestrator->setVramPreflightSafetyMarginBytes(vram_safety_margin_bytes);
        orchestrator->addDevice(device_ordinal);

        auto plan_slot_family = [&](int count,
                                    const auto &slot_name_for)
        {
            for (int slot = 0; slot < count; ++slot)
            {
                for (const auto &spec : specs)
                {
                    if (spec.N <= 0 || spec.K <= 0 || !spec.format.valid())
                        throw std::invalid_argument("GpuExpertSlotPool projection spec has invalid shape or format");
                    if (spec.format.isFloating())
                    {
                        const auto elements = static_cast<std::size_t>(spec.N) *
                                              static_cast<std::size_t>(spec.K);
                        const auto element_bytes =
                            spec.format.floatingElementBytes();
                        if (element_bytes == 0 ||
                            elements >
                                std::numeric_limits<std::size_t>::max() /
                                    element_bytes)
                        {
                            throw std::invalid_argument(
                                "GpuExpertSlotPool floating projection size overflows");
                        }
                        orchestrator->planRawWeight(
                            device_ordinal,
                            slot_name_for(slot, spec.label),
                            spec.N,
                            spec.K,
                            elements * element_bytes);
                    }
                    else
                    {
                        if ((spec.K % 32) != 0 ||
                            spec.payload_bytes_per_block <= 0)
                        {
                            throw std::invalid_argument(
                                "GpuExpertSlotPool NativeVNNI projection spec has invalid block geometry");
                        }
                        orchestrator->planWeight(
                            device_ordinal,
                            slot_name_for(slot, spec.label),
                            spec.N,
                            spec.K,
                            spec.payload_bytes_per_block,
                            spec.is_asymmetric,
                            spec.has_emins,
                            /*raw_gguf_bytes=*/0);
                    }
                }
            }
        };

        plan_slot_family(
            active_capacity,
            [](int slot, const std::string &label)
            {
                return activeSlotName(slot, label);
            });
        plan_slot_family(
            transfer_capacity,
            [](int slot, const std::string &label)
            {
                return transferSlotName(slot, label);
            });

        orchestrator->allocate(/*pinned_slot_size=*/0, /*num_h2d_streams=*/0);

        auto pool = std::shared_ptr<GpuExpertSlotPool>(new GpuExpertSlotPool(
            backend,
            device,
            device_ordinal,
            layer_idx,
            active_capacity,
            transfer_capacity,
            std::move(specs),
            std::move(orchestrator)));

        logVramBomLine(
            "moe_gpu_expert_slot_pool",
                "backend=" + std::string(backend ? backend->backendName() : "none") +
                    " device=" + device.to_string() +
                    " device_id=" + std::to_string(device_ordinal) +
                    " layer=" + std::to_string(layer_idx) +
                    " active_capacity=" + std::to_string(active_capacity) +
                    " transfer_capacity=" + std::to_string(transfer_capacity) +
                    " total_slots=" + std::to_string(active_capacity + transfer_capacity) +
                    " projections_per_slot=" + std::to_string(pool->specs_.size()) +
                    " planned_bytes=" + std::to_string(pool->orchestrator_->getPool(device_ordinal)->totalPlannedBytes()) +
                    " planned_mib=" + vramBomMiB(pool->orchestrator_->getPool(device_ordinal)->totalPlannedBytes()));

        return pool;
    }

    int GpuExpertSlotPool::recommendedCapacity(int num_experts, size_t arrival_batch_size)
    {
        const int ten_percent = std::max(1, (num_experts + 9) / 10);
        const int churn_headroom = ten_percent + std::max(1, (ten_percent + 1) / 2);
        const int batch = static_cast<int>(std::min<size_t>(
            arrival_batch_size,
            static_cast<size_t>(std::max(0, num_experts))));
        return std::min(std::max(0, num_experts), std::max(churn_headroom, batch));
    }

    int GpuExpertSlotPool::recommendedTransferCapacity(int num_experts, size_t arrival_batch_size)
    {
        if (num_experts <= 0)
            return 0;
        const int batch = static_cast<int>(std::min<size_t>(
            std::max<size_t>(1, arrival_batch_size),
            static_cast<size_t>(num_experts)));
        // Staging can arrive in multiple source-stage groups before a single
        // publish consumes the transfer slots, so reserve the same bounded churn
        // headroom as active slots instead of sizing to only the first subbatch.
        return std::min(
            std::max(0, num_experts),
            std::max(batch, recommendedCapacity(num_experts, arrival_batch_size)));
    }

    GpuExpertSlotPool::GpuExpertSlotPool(IBackend *backend,
                                         DeviceId device,
                                         int device_ordinal,
                                         int layer_idx,
                                         int active_capacity,
                                         int transfer_capacity,
                                         std::vector<ProjectionSpec> specs,
                                         std::shared_ptr<LoadOrchestrator> orchestrator)
        : backend_(backend),
          device_(device),
          device_ordinal_(device_ordinal),
          layer_idx_(layer_idx),
          active_capacity_(active_capacity),
          transfer_capacity_(transfer_capacity),
          specs_(std::move(specs)),
          orchestrator_(std::move(orchestrator)),
          expert_by_slot_(static_cast<size_t>(active_capacity), -1),
          epoch_by_slot_(static_cast<size_t>(active_capacity), 0),
          expert_by_transfer_slot_(static_cast<size_t>(transfer_capacity), -1),
          epoch_by_transfer_slot_(static_cast<size_t>(transfer_capacity), 0)
    {
    }

    size_t GpuExpertSlotPool::SlotIdentityHash::operator()(
        const SlotIdentity &identity) const noexcept
    {
        /* Mix both fields because successive epochs of one hot expert coexist. */
        const size_t epoch_hash =
            std::hash<uint64_t>{}(identity.residency_epoch);
        const size_t expert_hash = std::hash<int>{}(identity.expert_id);
        return epoch_hash ^
               (expert_hash + static_cast<size_t>(0x9e3779b9u) +
                (epoch_hash << 6u) + (epoch_hash >> 2u));
    }

    std::optional<GpuExpertSlotPool::AcquiredSlot>
    GpuExpertSlotPool::acquire(int expert_id)
    {
        return acquire(expert_id, /*residency_epoch=*/0);
    }

    std::optional<GpuExpertSlotPool::AcquiredSlot>
    GpuExpertSlotPool::acquire(
        int expert_id,
        uint64_t residency_epoch)
    {
        if (expert_id < 0)
            return std::nullopt;

        const SlotIdentity identity{expert_id, residency_epoch};
        int slot_index = -1;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (slot_by_identity_.count(identity) > 0)
                return std::nullopt;

            for (int i = 0; i < active_capacity_; ++i)
            {
                if (expert_by_slot_[static_cast<size_t>(i)] < 0)
                {
                    slot_index = i;
                    break;
                }
            }
            if (slot_index < 0)
                return std::nullopt;

            expert_by_slot_[static_cast<size_t>(slot_index)] = expert_id;
            epoch_by_slot_[static_cast<size_t>(slot_index)] = residency_epoch;
            slot_by_identity_[identity] = slot_index;
        }

        auto *pool =
            orchestrator_ ? orchestrator_->getPool(device_ordinal_) : nullptr;
        if (!pool)
        {
            releaseSlot(slot_index, identity);
            return std::nullopt;
        }

        AcquiredSlot acquired;
        acquired.slot_index = slot_index;
        acquired.expert_id = expert_id;
        acquired.residency_epoch = residency_epoch;
        acquired.projections.reserve(specs_.size());
        for (const auto &spec : specs_)
        {
            auto slot = pool->getSlot(activeSlotName(slot_index, spec.label));
            if (!slot)
            {
                releaseSlot(slot_index, identity);
                return std::nullopt;
            }

            ProjectionSlot projection;
            projection.spec = spec;
            projection.slot = *slot;
            projection.blocks_per_row = spec.format.isNativeVnni()
                                            ? static_cast<uint32_t>(spec.K / 32)
                                            : 0;
            acquired.projections.push_back(std::move(projection));
        }

        auto token = std::make_shared<SlotLeaseToken>(SlotLeaseToken{
            slot_index,
            expert_id,
            residency_epoch,
            false,
            orchestrator_,
            weak_from_this()});
        /*
         * Capture the stored pointer before moving `token` into the deleter.
         * Constructor-argument evaluation order is not a lifetime contract;
         * reading token.get() in the other argument can otherwise observe the
         * moved-from shared_ptr and manufacture a false/null lease handle.
         */
        void *const token_pointer = token.get();
        acquired.lifetime = std::shared_ptr<void>(
            token_pointer,
            [token = std::move(token)](void *) mutable
            {
                if (auto pool = token->pool.lock())
                {
                    const SlotIdentity token_identity{
                        token->expert_id,
                        token->residency_epoch};
                    if (token->transfer_slot)
                    {
                        pool->releaseTransferSlot(
                            token->slot_index, token_identity);
                    }
                    else
                    {
                        pool->releaseSlot(
                            token->slot_index, token_identity);
                    }
                }
            });

        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "gpu_direct_slot_pool_acquire",
            1.0,
            "rebalance",
            device_.to_string(),
            {{"layer", std::to_string(layer_idx_)},
             {"slot", std::to_string(slot_index)},
             {"epoch", std::to_string(residency_epoch)}});
        return acquired;
    }

    std::optional<GpuExpertSlotPool::TransferSlot>
    GpuExpertSlotPool::acquireTransferSlot(int expert_id)
    {
        return acquireTransferSlot(expert_id, /*residency_epoch=*/0);
    }

    std::optional<GpuExpertSlotPool::TransferSlot>
    GpuExpertSlotPool::acquireTransferSlot(
        int expert_id,
        uint64_t residency_epoch)
    {
        if (expert_id < 0)
            return std::nullopt;

        const SlotIdentity identity{expert_id, residency_epoch};
        int slot_index = -1;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (transfer_slot_by_identity_.count(identity) > 0)
                return std::nullopt;

            for (int i = 0; i < transfer_capacity_; ++i)
            {
                if (expert_by_transfer_slot_[static_cast<size_t>(i)] < 0)
                {
                    slot_index = i;
                    break;
                }
            }
            if (slot_index < 0)
                return std::nullopt;

            expert_by_transfer_slot_[static_cast<size_t>(slot_index)] =
                expert_id;
            epoch_by_transfer_slot_[static_cast<size_t>(slot_index)] =
                residency_epoch;
            transfer_slot_by_identity_[identity] = slot_index;
        }

        auto *pool =
            orchestrator_ ? orchestrator_->getPool(device_ordinal_) : nullptr;
        if (!pool)
        {
            releaseTransferSlot(slot_index, identity);
            return std::nullopt;
        }

        TransferSlot acquired;
        acquired.slot_index = slot_index;
        acquired.expert_id = expert_id;
        acquired.residency_epoch = residency_epoch;
        acquired.projections.reserve(specs_.size());
        for (const auto &spec : specs_)
        {
            auto slot = pool->getSlot(transferSlotName(slot_index, spec.label));
            if (!slot)
            {
                releaseTransferSlot(slot_index, identity);
                return std::nullopt;
            }

            ProjectionSlot projection;
            projection.spec = spec;
            projection.slot = *slot;
            projection.blocks_per_row = spec.format.isNativeVnni()
                                            ? static_cast<uint32_t>(spec.K / 32)
                                            : 0;
            acquired.projections.push_back(std::move(projection));
        }

        auto token = std::make_shared<SlotLeaseToken>(SlotLeaseToken{
            slot_index,
            expert_id,
            residency_epoch,
            true,
            orchestrator_,
            weak_from_this()});
        /* See acquire(): the stable stored pointer must precede token capture. */
        void *const token_pointer = token.get();
        acquired.lifetime = std::shared_ptr<void>(
            token_pointer,
            [token = std::move(token)](void *) mutable
            {
                if (auto pool = token->pool.lock())
                {
                    const SlotIdentity token_identity{
                        token->expert_id,
                        token->residency_epoch};
                    if (token->transfer_slot)
                    {
                        pool->releaseTransferSlot(
                            token->slot_index, token_identity);
                    }
                    else
                    {
                        pool->releaseSlot(
                            token->slot_index, token_identity);
                    }
                }
            });

        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "gpu_direct_transfer_slot_acquire",
            1.0,
            "rebalance",
            device_.to_string(),
            {{"layer", std::to_string(layer_idx_)},
             {"slot", std::to_string(slot_index)},
             {"epoch", std::to_string(residency_epoch)}});
        return acquired;
    }

    size_t GpuExpertSlotPool::activeCapacity() const
    {
        return static_cast<size_t>(std::max(0, active_capacity_));
    }

    size_t GpuExpertSlotPool::transferCapacity() const
    {
        return static_cast<size_t>(std::max(0, transfer_capacity_));
    }

    size_t GpuExpertSlotPool::capacity() const
    {
        return activeCapacity();
    }

    size_t GpuExpertSlotPool::usedSlots() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return slot_by_identity_.size();
    }

    size_t GpuExpertSlotPool::usedTransferSlots() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return transfer_slot_by_identity_.size();
    }

    size_t GpuExpertSlotPool::availableSlots() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t capacity = activeCapacity();
        const size_t used = slot_by_identity_.size();
        return used < capacity ? capacity - used : 0;
    }

    size_t GpuExpertSlotPool::availableTransferSlots() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t capacity = transferCapacity();
        const size_t used = transfer_slot_by_identity_.size();
        return used < capacity ? capacity - used : 0;
    }

    std::optional<int> GpuExpertSlotPool::slotForExpert(int expert_id) const
    {
        return slotForExpert(expert_id, /*residency_epoch=*/0);
    }

    std::optional<int> GpuExpertSlotPool::slotForExpert(
        int expert_id,
        uint64_t residency_epoch) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = slot_by_identity_.find({expert_id, residency_epoch});
        if (it == slot_by_identity_.end())
            return std::nullopt;
        return it->second;
    }

    std::optional<int> GpuExpertSlotPool::transferSlotForExpert(int expert_id) const
    {
        return transferSlotForExpert(expert_id, /*residency_epoch=*/0);
    }

    std::optional<int> GpuExpertSlotPool::transferSlotForExpert(
        int expert_id,
        uint64_t residency_epoch) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = transfer_slot_by_identity_.find(
            {expert_id, residency_epoch});
        if (it == transfer_slot_by_identity_.end())
            return std::nullopt;
        return it->second;
    }

    std::string GpuExpertSlotPool::activeSlotName(int slot_index, const std::string &label)
    {
        return "expert_slot_" + std::to_string(slot_index) + "_" + label;
    }

    std::string GpuExpertSlotPool::transferSlotName(int slot_index, const std::string &label)
    {
        return "expert_transfer_slot_" + std::to_string(slot_index) + "_" + label;
    }

    void GpuExpertSlotPool::releaseSlot(
        int slot_index,
        SlotIdentity identity) noexcept
    {
        bool released = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (slot_index < 0 || slot_index >= active_capacity_)
                return;

            auto &assigned = expert_by_slot_[static_cast<size_t>(slot_index)];
            auto &assigned_epoch =
                epoch_by_slot_[static_cast<size_t>(slot_index)];
            if (assigned != identity.expert_id ||
                assigned_epoch != identity.residency_epoch)
                return;

            assigned = -1;
            assigned_epoch = 0;
            slot_by_identity_.erase(identity);
            released = true;
        }

        if (released)
        {
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "gpu_direct_slot_pool_release",
                1.0,
                "rebalance",
                device_.to_string(),
                {{"layer", std::to_string(layer_idx_)},
                 {"slot", std::to_string(slot_index)},
                 {"epoch", std::to_string(identity.residency_epoch)}});
        }
    }

    void GpuExpertSlotPool::releaseTransferSlot(
        int slot_index,
        SlotIdentity identity) noexcept
    {
        bool released = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (slot_index < 0 || slot_index >= transfer_capacity_)
                return;

            auto &assigned =
                expert_by_transfer_slot_[static_cast<size_t>(slot_index)];
            auto &assigned_epoch =
                epoch_by_transfer_slot_[static_cast<size_t>(slot_index)];
            if (assigned != identity.expert_id ||
                assigned_epoch != identity.residency_epoch)
                return;

            assigned = -1;
            assigned_epoch = 0;
            transfer_slot_by_identity_.erase(identity);
            released = true;
        }

        if (released)
        {
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "gpu_direct_transfer_slot_release",
                1.0,
                "rebalance",
                device_.to_string(),
                {{"layer", std::to_string(layer_idx_)},
                 {"slot", std::to_string(slot_index)},
                 {"epoch", std::to_string(identity.residency_epoch)}});
        }
    }

} // namespace llaminar2
