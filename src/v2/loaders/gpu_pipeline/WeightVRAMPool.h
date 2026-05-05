#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

class IBackend;

class WeightVRAMPool
{
  public:
    struct WeightSlot
    {
        uint8_t* d_native_vnni_payload = nullptr;
        void* d_native_vnni_scales = nullptr;
        void* d_native_vnni_mins = nullptr;
        void* d_native_vnni_emins = nullptr;
        size_t payload_bytes = 0;
        size_t staging_bytes = 0;
    };

    WeightVRAMPool();
    ~WeightVRAMPool();

    WeightVRAMPool(const WeightVRAMPool&) = delete;
    WeightVRAMPool& operator=(const WeightVRAMPool&) = delete;
    WeightVRAMPool(WeightVRAMPool&& other) noexcept;
    WeightVRAMPool& operator=(WeightVRAMPool&& other) noexcept;

    /// Phase 1: Plan — calculate sizes, no allocation.
    /// Call once per weight that will be stored on this device.
    void planWeight(const std::string& name, int N, int K,
                    int payload_bytes_per_block, bool is_asymmetric, bool has_emins,
                    size_t raw_gguf_bytes);

    /// Phase 1 (raw): Plan a floating-point weight that needs no repack.
    /// Allocates a contiguous region for raw bytes (H2D copy only, no repack).
    void planRawWeight(const std::string& name, int N, int K, size_t raw_bytes);

    /// Phase 2: Allocate — single device allocation for all planned weights + staging.
    /// @param backend  GPU backend (CUDA or ROCm) — if null, falls back to no-op (unit tests)
    /// @param device_id  Device ordinal
    /// @param staging_slot_count  How many staging slots (ring-buffer overlap). 0 = no staging.
    bool allocate(IBackend* backend, int device_id, int staging_slot_count = 0);

    /// Phase 3: Get slot — zero-cost offset lookup by weight name.
    std::optional<WeightSlot> getSlot(const std::string& name) const;

    size_t totalPlannedBytes() const;
    size_t numPlannedWeights() const;
    bool isAllocated() const;
    int deviceId() const;

    void release();

    /// Get device pointer to staging slot at given index.
    /// Returns nullptr if not allocated or index out of range.
    uint8_t* getStagingSlot(int slot_index) const;

    /// Number of staging slots.
    int stagingSlotCount() const;

    /// Maximum bytes per staging slot.
    size_t maxStagingSlotBytes() const;

  private:
    static constexpr size_t kAlignment = 256;

    struct WeightPlan
    {
        size_t payload_offset = 0;
        size_t payload_bytes = 0;
        size_t scales_offset = 0;
        size_t scales_bytes = 0;
        size_t mins_offset = 0;
        size_t mins_bytes = 0;
        size_t emins_offset = 0;
        size_t emins_bytes = 0;
        size_t staging_bytes = 0;
        int N = 0, K = 0;
    };

    static size_t alignUp(size_t offset, size_t alignment);
    size_t allocateRegion(size_t bytes);

    void* d_base_ = nullptr;
    size_t total_bytes_ = 0;
    size_t staging_region_offset_ = 0;
    size_t staging_region_bytes_ = 0;
    int staging_slot_count_ = 0;
    size_t max_staging_slot_bytes_ = 0;
    IBackend* backend_ = nullptr;
    int device_id_ = -1;
    bool allocated_ = false;

    size_t current_offset_ = 0;

    std::unordered_map<std::string, WeightPlan> plans_;
    std::vector<std::string> weight_order_;
};

} // namespace llaminar2
