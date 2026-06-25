#include "execution/moe/GpuExpertSlotPool.h"

#include "../../backends/IBackend.h"
#include "../../loaders/gpu_pipeline/LoadOrchestrator.h"
#include "../../utils/Logger.h"
#include "../../utils/PerfStatsCollector.h"
#include "../../utils/VramBillOfMaterials.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        struct SlotLeaseToken
        {
            int slot_index = -1;
            int expert_id = -1;
            std::shared_ptr<LoadOrchestrator> orchestrator;
            std::weak_ptr<GpuExpertSlotPool> pool;
        };
    } // namespace

    std::shared_ptr<GpuExpertSlotPool> GpuExpertSlotPool::create(
        IBackend *backend,
        DeviceId device,
        int device_ordinal,
        int layer_idx,
        int capacity,
        std::vector<ProjectionSpec> specs,
        size_t vram_safety_margin_bytes)
    {
        if (capacity <= 0)
            throw std::invalid_argument("GpuExpertSlotPool capacity must be positive");
        if (specs.empty())
            throw std::invalid_argument("GpuExpertSlotPool requires at least one projection spec");

        auto orchestrator = std::make_shared<LoadOrchestrator>(backend);
        orchestrator->setVramPreflightSafetyMarginBytes(vram_safety_margin_bytes);
        orchestrator->addDevice(device_ordinal);

        for (int slot = 0; slot < capacity; ++slot)
        {
            for (const auto &spec : specs)
            {
                if (spec.N <= 0 || spec.K <= 0 || spec.payload_bytes_per_block <= 0)
                    throw std::invalid_argument("GpuExpertSlotPool projection spec has invalid shape");
                orchestrator->planWeight(
                    device_ordinal,
                    slotName(slot, spec.label),
                    spec.N,
                    spec.K,
                    spec.payload_bytes_per_block,
                    spec.is_asymmetric,
                    spec.has_emins,
                    /*raw_gguf_bytes=*/0);
            }
        }

        orchestrator->allocate(/*pinned_slot_size=*/0, /*num_h2d_streams=*/0);

        auto pool = std::shared_ptr<GpuExpertSlotPool>(new GpuExpertSlotPool(
            backend,
            device,
            device_ordinal,
            layer_idx,
            capacity,
            std::move(specs),
            std::move(orchestrator)));

        logVramBomLine(
            "moe_gpu_expert_slot_pool",
            "backend=" + std::string(backend ? backend->backendName() : "none") +
                " device=" + device.to_string() +
                " device_id=" + std::to_string(device_ordinal) +
                " layer=" + std::to_string(layer_idx) +
                " capacity=" + std::to_string(capacity) +
                " projections_per_slot=" + std::to_string(pool->specs_.size()) +
                " planned_bytes=" + std::to_string(pool->orchestrator_->getPool(device_ordinal)->totalPlannedBytes()) +
                " planned_mib=" + vramBomMiB(pool->orchestrator_->getPool(device_ordinal)->totalPlannedBytes()));

        return pool;
    }

    int GpuExpertSlotPool::recommendedCapacity(int num_experts, size_t arrival_batch_size)
    {
        const int ten_percent = std::max(1, (num_experts + 9) / 10);
        const int batch = static_cast<int>(std::min<size_t>(
            arrival_batch_size,
            static_cast<size_t>(std::max(0, num_experts))));
        return std::max(ten_percent, batch);
    }

    GpuExpertSlotPool::GpuExpertSlotPool(IBackend *backend,
                                         DeviceId device,
                                         int device_ordinal,
                                         int layer_idx,
                                         int capacity,
                                         std::vector<ProjectionSpec> specs,
                                         std::shared_ptr<LoadOrchestrator> orchestrator)
        : backend_(backend),
          device_(device),
          device_ordinal_(device_ordinal),
          layer_idx_(layer_idx),
          capacity_(capacity),
          specs_(std::move(specs)),
          orchestrator_(std::move(orchestrator)),
          expert_by_slot_(static_cast<size_t>(capacity), -1)
    {
    }

    std::optional<GpuExpertSlotPool::AcquiredSlot> GpuExpertSlotPool::acquire(int expert_id)
    {
        int slot_index = -1;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (slot_by_expert_.count(expert_id) > 0)
                return std::nullopt;

            for (int i = 0; i < capacity_; ++i)
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
            slot_by_expert_[expert_id] = slot_index;
        }

        auto *pool = orchestrator_ ? orchestrator_->getPool(device_ordinal_) : nullptr;
        if (!pool)
        {
            releaseSlot(slot_index, expert_id);
            return std::nullopt;
        }

        AcquiredSlot acquired;
        acquired.slot_index = slot_index;
        acquired.expert_id = expert_id;
        acquired.projections.reserve(specs_.size());
        for (const auto &spec : specs_)
        {
            auto slot = pool->getSlot(slotName(slot_index, spec.label));
            if (!slot)
            {
                releaseSlot(slot_index, expert_id);
                return std::nullopt;
            }

            ProjectionSlot projection;
            projection.spec = spec;
            projection.slot = *slot;
            projection.blocks_per_row = static_cast<uint32_t>(spec.K / 32);
            acquired.projections.push_back(std::move(projection));
        }

        SlotLeaseToken *token = new SlotLeaseToken{
            slot_index,
            expert_id,
            orchestrator_,
            weak_from_this()};
        acquired.lifetime = std::shared_ptr<void>(
            token,
            [](void *ptr)
            {
                std::unique_ptr<SlotLeaseToken> owned(static_cast<SlotLeaseToken *>(ptr));
                if (auto pool = owned->pool.lock())
                    pool->releaseSlot(owned->slot_index, owned->expert_id);
            });

        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "gpu_direct_slot_pool_acquire",
            1.0,
            "rebalance",
            device_.to_string(),
            {{"layer", std::to_string(layer_idx_)},
             {"slot", std::to_string(slot_index)}});
        return acquired;
    }

    size_t GpuExpertSlotPool::capacity() const
    {
        return static_cast<size_t>(std::max(0, capacity_));
    }

    size_t GpuExpertSlotPool::usedSlots() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return slot_by_expert_.size();
    }

    std::optional<int> GpuExpertSlotPool::slotForExpert(int expert_id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = slot_by_expert_.find(expert_id);
        if (it == slot_by_expert_.end())
            return std::nullopt;
        return it->second;
    }

    std::string GpuExpertSlotPool::slotName(int slot_index, const std::string &label)
    {
        return "expert_slot_" + std::to_string(slot_index) + "_" + label;
    }

    void GpuExpertSlotPool::releaseSlot(int slot_index, int expert_id)
    {
        bool released = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (slot_index < 0 || slot_index >= capacity_)
                return;

            auto &assigned = expert_by_slot_[static_cast<size_t>(slot_index)];
            if (assigned != expert_id)
                return;

            assigned = -1;
            slot_by_expert_.erase(expert_id);
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
                 {"slot", std::to_string(slot_index)}});
        }
    }

} // namespace llaminar2
