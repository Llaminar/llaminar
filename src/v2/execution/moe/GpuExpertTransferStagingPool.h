#pragma once

/**
 * @file GpuExpertTransferStagingPool.h
 * @brief First-class GPU scratch arena for rolling MoE expert transfers.
 *
 * Active expert slots are per-layer because GEMM wrappers point at them for as
 * long as the expert is resident. Transfer staging is different: it is a
 * transient destination-device scratch arena used by the rebalance wave before
 * copying arrivals into active slots.
 */

#include "../../backends/DeviceId.h"
#include "GpuExpertSlotPool.h"

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    class IBackend;
    class LoadOrchestrator;

    class GpuExpertTransferStagingPool
        : public std::enable_shared_from_this<GpuExpertTransferStagingPool>
    {
    public:
        using ProjectionSpec = GpuExpertSlotPool::ProjectionSpec;
        using ProjectionSlot = GpuExpertSlotPool::ProjectionSlot;

        struct Lease
        {
            int slot_index = -1;
            int expert_id = -1;
            std::shared_ptr<void> lifetime;
            std::vector<ProjectionSlot> projections;
        };

        static std::shared_ptr<GpuExpertTransferStagingPool> create(
            IBackend *backend,
            DeviceId device,
            int device_ordinal,
            int capacity,
            std::vector<ProjectionSpec> specs);

        static int recommendedCapacity(int num_experts, size_t rolling_wave_capacity);

        std::optional<Lease> acquire(int expert_id);

        size_t capacity() const;
        size_t usedSlots() const;
        size_t availableSlots() const;
        std::optional<int> slotForExpert(int expert_id) const;
        bool compatibleWith(const std::vector<ProjectionSpec> &specs) const;

    private:
        GpuExpertTransferStagingPool(
            IBackend *backend,
            DeviceId device,
            int device_ordinal,
            int capacity,
            std::vector<ProjectionSpec> specs,
            std::shared_ptr<LoadOrchestrator> orchestrator);

        static std::string slotName(int slot_index, const std::string &label);
        void releaseSlot(int slot_index, int expert_id);

        IBackend *backend_ = nullptr;
        DeviceId device_;
        int device_ordinal_ = -1;
        int capacity_ = 0;
        std::vector<ProjectionSpec> specs_;
        std::shared_ptr<LoadOrchestrator> orchestrator_;

        mutable std::mutex mutex_;
        std::vector<int> expert_by_slot_;
        std::unordered_map<int, int> slot_by_expert_;
    };

} // namespace llaminar2
