#pragma once

/**
 * @file GpuExpertSlotPool.h
 * @brief Reusable GPU-resident slots for dynamically arrived MoE experts.
 *
 * Rebalanced GPU arrivals all share the same per-layer expert shapes.  This pool
 * preallocates a bounded number of physical gate/up/down slots and hands callers
 * a lease that keeps the underlying allocation alive until the final GEMM wrapper
 * for the logical expert is released.
 */

#include "../../backends/DeviceId.h"
#include "../../loaders/gpu_pipeline/WeightVRAMPool.h"

#include <cstddef>
#include <cstdint>
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

    class GpuExpertSlotPool : public std::enable_shared_from_this<GpuExpertSlotPool>
    {
    public:
        struct ProjectionSpec
        {
            std::string label;
            int N = 0;
            int K = 0;
            int payload_bytes_per_block = 0;
            bool is_asymmetric = false;
            bool has_emins = false;
            uint8_t codebook_id = 0;
        };

        struct ProjectionSlot
        {
            ProjectionSpec spec;
            WeightVRAMPool::WeightSlot slot;
            uint32_t blocks_per_row = 0;
        };

        struct AcquiredSlot
        {
            int slot_index = -1;
            int expert_id = -1;
            std::shared_ptr<void> lifetime;
            std::vector<ProjectionSlot> projections;
        };

        static std::shared_ptr<GpuExpertSlotPool> create(
            IBackend *backend,
            DeviceId device,
            int device_ordinal,
            int layer_idx,
            int capacity,
            std::vector<ProjectionSpec> specs,
            size_t vram_safety_margin_bytes);

        static int recommendedCapacity(int num_experts, size_t arrival_batch_size);

        std::optional<AcquiredSlot> acquire(int expert_id);

        size_t capacity() const;
        size_t usedSlots() const;
        std::optional<int> slotForExpert(int expert_id) const;

    private:
        GpuExpertSlotPool(IBackend *backend,
                          DeviceId device,
                          int device_ordinal,
                          int layer_idx,
                          int capacity,
                          std::vector<ProjectionSpec> specs,
                          std::shared_ptr<LoadOrchestrator> orchestrator);

        static std::string slotName(int slot_index, const std::string &label);
        void releaseSlot(int slot_index, int expert_id);

        IBackend *backend_ = nullptr;
        DeviceId device_;
        int device_ordinal_ = -1;
        int layer_idx_ = -1;
        int capacity_ = 0;
        std::vector<ProjectionSpec> specs_;
        std::shared_ptr<LoadOrchestrator> orchestrator_;

        mutable std::mutex mutex_;
        std::vector<int> expert_by_slot_;
        std::unordered_map<int, int> slot_by_expert_;
    };

} // namespace llaminar2
