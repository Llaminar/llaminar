#include "execution/moe/GpuExpertTransferStagingPool.h"

#include "../../backends/IBackend.h"
#include "../../loaders/gpu_pipeline/LoadOrchestrator.h"
#include "../../utils/Logger.h"
#include "../../utils/PerfStatsCollector.h"
#include "../../utils/VramBillOfMaterials.h"

#include <algorithm>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        struct StagingLeaseToken
        {
            int slot_index = -1;
            int expert_id = -1;
            std::shared_ptr<LoadOrchestrator> orchestrator;
            std::weak_ptr<GpuExpertTransferStagingPool> pool;
        };
    } // namespace

    std::shared_ptr<GpuExpertTransferStagingPool> GpuExpertTransferStagingPool::create(
        IBackend *backend,
        DeviceId device,
        int device_ordinal,
        int capacity,
        std::vector<ProjectionSpec> specs,
        size_t vram_safety_margin_bytes)
    {
        if (capacity <= 0)
            throw std::invalid_argument("GpuExpertTransferStagingPool capacity must be positive");
        if (specs.empty())
            throw std::invalid_argument("GpuExpertTransferStagingPool requires at least one projection spec");

        auto orchestrator = std::make_shared<LoadOrchestrator>(backend);
        orchestrator->setVramPreflightSafetyMarginBytes(vram_safety_margin_bytes);
        orchestrator->addDevice(device_ordinal);

        for (int slot = 0; slot < capacity; ++slot)
        {
            for (const auto &spec : specs)
            {
                if (spec.N <= 0 || spec.K <= 0 || spec.payload_bytes_per_block <= 0)
                    throw std::invalid_argument("GpuExpertTransferStagingPool projection spec has invalid shape");
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

        auto pool = std::shared_ptr<GpuExpertTransferStagingPool>(
            new GpuExpertTransferStagingPool(
                backend,
                device,
                device_ordinal,
                capacity,
                std::move(specs),
                std::move(orchestrator)));

        const auto *weight_pool = pool->orchestrator_->getPool(device_ordinal);
        const size_t planned_bytes = weight_pool ? weight_pool->totalPlannedBytes() : 0;
        logVramBomLine(
            "moe_gpu_expert_transfer_staging_pool",
            "backend=" + std::string(backend ? backend->backendName() : "none") +
                " device=" + device.to_string() +
                " device_id=" + std::to_string(device_ordinal) +
                " capacity=" + std::to_string(capacity) +
                " projections_per_slot=" + std::to_string(pool->specs_.size()) +
                " planned_bytes=" + std::to_string(planned_bytes) +
                " planned_mib=" + vramBomMiB(planned_bytes));

        return pool;
    }

    int GpuExpertTransferStagingPool::recommendedCapacity(
        int num_experts,
        size_t rolling_wave_capacity)
    {
        if (num_experts <= 0)
            return 0;
        const size_t requested = std::max<size_t>(1, rolling_wave_capacity);
        return static_cast<int>(std::min<size_t>(
            static_cast<size_t>(num_experts),
            requested));
    }

    GpuExpertTransferStagingPool::GpuExpertTransferStagingPool(
        IBackend *backend,
        DeviceId device,
        int device_ordinal,
        int capacity,
        std::vector<ProjectionSpec> specs,
        std::shared_ptr<LoadOrchestrator> orchestrator)
        : backend_(backend),
          device_(device),
          device_ordinal_(device_ordinal),
          capacity_(capacity),
          specs_(std::move(specs)),
          orchestrator_(std::move(orchestrator)),
          expert_by_slot_(static_cast<size_t>(capacity), -1)
    {
    }

    std::optional<GpuExpertTransferStagingPool::Lease>
    GpuExpertTransferStagingPool::acquire(int expert_id)
    {
        int slot_index = -1;
        {
            std::lock_guard<std::mutex> lock(mutex_);
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

        Lease lease;
        lease.slot_index = slot_index;
        lease.expert_id = expert_id;
        lease.projections.reserve(specs_.size());
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
            lease.projections.push_back(std::move(projection));
        }

        auto token = std::make_shared<StagingLeaseToken>(StagingLeaseToken{
            slot_index,
            expert_id,
            orchestrator_,
            weak_from_this()});
        lease.lifetime = std::shared_ptr<void>(
            token.get(),
            [token = std::move(token)](void *) mutable
            {
                if (auto pool = token->pool.lock())
                    pool->releaseSlot(token->slot_index, token->expert_id);
            });

        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "gpu_direct_transfer_staging_pool_acquire",
            1.0,
            "rebalance",
            device_.to_string(),
            {{"slot", std::to_string(slot_index)}});
        return lease;
    }

    size_t GpuExpertTransferStagingPool::capacity() const
    {
        return static_cast<size_t>(std::max(0, capacity_));
    }

    size_t GpuExpertTransferStagingPool::usedSlots() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<size_t>(std::count_if(
            expert_by_slot_.begin(),
            expert_by_slot_.end(),
            [](int expert_id)
            {
                return expert_id >= 0;
            }));
    }

    size_t GpuExpertTransferStagingPool::availableSlots() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t cap = capacity();
        const size_t used = static_cast<size_t>(std::count_if(
            expert_by_slot_.begin(),
            expert_by_slot_.end(),
            [](int expert_id)
            {
                return expert_id >= 0;
            }));
        return used < cap ? cap - used : 0;
    }

    std::optional<int> GpuExpertTransferStagingPool::slotForExpert(int expert_id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int slot = 0; slot < capacity_; ++slot)
        {
            if (expert_by_slot_[static_cast<size_t>(slot)] == expert_id)
                return slot;
        }
        return std::nullopt;
    }

    bool GpuExpertTransferStagingPool::compatibleWith(
        const std::vector<ProjectionSpec> &specs) const
    {
        if (specs.size() != specs_.size())
            return false;
        for (size_t i = 0; i < specs.size(); ++i)
        {
            const auto &a = specs[i];
            const auto &b = specs_[i];
            if (a.label != b.label ||
                a.N != b.N ||
                a.K != b.K ||
                a.payload_bytes_per_block != b.payload_bytes_per_block ||
                a.is_asymmetric != b.is_asymmetric ||
                a.has_emins != b.has_emins ||
                a.codebook_id != b.codebook_id)
            {
                return false;
            }
        }
        return true;
    }

    std::string GpuExpertTransferStagingPool::slotName(
        int slot_index,
        const std::string &label)
    {
        return "expert_transfer_staging_slot_" + std::to_string(slot_index) + "_" + label;
    }

    void GpuExpertTransferStagingPool::releaseSlot(int slot_index, int expert_id)
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
            auto map_it = slot_by_expert_.find(expert_id);
            if (map_it != slot_by_expert_.end() && map_it->second == slot_index)
            {
                slot_by_expert_.erase(map_it);
                for (int i = 0; i < capacity_; ++i)
                {
                    if (expert_by_slot_[static_cast<size_t>(i)] == expert_id)
                    {
                        slot_by_expert_[expert_id] = i;
                        break;
                    }
                }
            }
            released = true;
        }

        if (released)
        {
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "gpu_direct_transfer_staging_pool_release",
                1.0,
                "rebalance",
                device_.to_string(),
                {{"slot", std::to_string(slot_index)}});
        }
    }
} // namespace llaminar2
