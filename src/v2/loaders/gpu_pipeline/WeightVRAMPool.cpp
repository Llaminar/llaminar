#include "loaders/gpu_pipeline/WeightVRAMPool.h"
#include "backends/IBackend.h"
#include "utils/Logger.h"

#include <algorithm>
#include <stdexcept>

namespace llaminar2
{

WeightVRAMPool::WeightVRAMPool() = default;

WeightVRAMPool::~WeightVRAMPool() { release(); }

WeightVRAMPool::WeightVRAMPool(WeightVRAMPool&& other) noexcept
    : d_base_(other.d_base_),
      total_bytes_(other.total_bytes_),
      staging_region_offset_(other.staging_region_offset_),
      staging_region_bytes_(other.staging_region_bytes_),
      staging_slot_count_(other.staging_slot_count_),
      max_staging_slot_bytes_(other.max_staging_slot_bytes_),
      backend_(other.backend_),
      device_id_(other.device_id_),
      allocated_(other.allocated_),
      current_offset_(other.current_offset_),
      plans_(std::move(other.plans_)),
      weight_order_(std::move(other.weight_order_))
{
    other.d_base_ = nullptr;
    other.allocated_ = false;
    other.total_bytes_ = 0;
    other.backend_ = nullptr;
}

WeightVRAMPool& WeightVRAMPool::operator=(WeightVRAMPool&& other) noexcept
{
    if (this != &other)
    {
        release();
        d_base_ = other.d_base_;
        total_bytes_ = other.total_bytes_;
        staging_region_offset_ = other.staging_region_offset_;
        staging_region_bytes_ = other.staging_region_bytes_;
        staging_slot_count_ = other.staging_slot_count_;
        max_staging_slot_bytes_ = other.max_staging_slot_bytes_;
        backend_ = other.backend_;
        device_id_ = other.device_id_;
        allocated_ = other.allocated_;
        current_offset_ = other.current_offset_;
        plans_ = std::move(other.plans_);
        weight_order_ = std::move(other.weight_order_);
        other.d_base_ = nullptr;
        other.allocated_ = false;
        other.total_bytes_ = 0;
        other.backend_ = nullptr;
    }
    return *this;
}

size_t WeightVRAMPool::alignUp(size_t offset, size_t alignment)
{
    return (offset + alignment - 1) & ~(alignment - 1);
}

size_t WeightVRAMPool::allocateRegion(size_t bytes)
{
    if (bytes == 0) return 0;
    current_offset_ = alignUp(current_offset_, kAlignment);
    size_t offset = current_offset_;
    current_offset_ += bytes;
    return offset;
}

void WeightVRAMPool::planWeight(const std::string& name, int N, int K,
                                int payload_bytes_per_block, bool is_asymmetric,
                                bool has_emins, size_t raw_gguf_bytes)
{
    if (allocated_)
    {
        LOG_ERROR("Cannot plan weights after allocation");
        throw std::runtime_error("WeightVRAMPool: cannot plan after allocate()");
    }
    if (plans_.count(name))
    {
        LOG_ERROR("Duplicate weight name: " << name);
        throw std::runtime_error("WeightVRAMPool: duplicate weight name '" + name + "'");
    }

    const int blocks_per_row = K / 32;

    WeightPlan plan;
    plan.N = N;
    plan.K = K;
    plan.staging_bytes = raw_gguf_bytes;

    // Payload region
    plan.payload_bytes = static_cast<size_t>(blocks_per_row) * N * payload_bytes_per_block;
    plan.payload_offset = allocateRegion(plan.payload_bytes);

    // Scales region (FP16 = uint16_t)
    plan.scales_bytes = static_cast<size_t>(blocks_per_row) * N * sizeof(uint16_t);
    plan.scales_offset = allocateRegion(plan.scales_bytes);

    // Mins region (asymmetric only)
    if (is_asymmetric)
    {
        plan.mins_bytes = plan.scales_bytes; // Same size as scales
        plan.mins_offset = allocateRegion(plan.mins_bytes);
    }

    // Emins region (Q2_K only)
    if (has_emins)
    {
        plan.emins_bytes = static_cast<size_t>(blocks_per_row) * N * sizeof(uint32_t);
        plan.emins_offset = allocateRegion(plan.emins_bytes);
    }

    plans_[name] = plan;
    weight_order_.push_back(name);

    LOG_DEBUG("Planned weight '" << name << "' N=" << N << " K=" << K
              << " payload=" << plan.payload_bytes
              << " scales=" << plan.scales_bytes
              << " mins=" << plan.mins_bytes
              << " emins=" << plan.emins_bytes);
}

void WeightVRAMPool::planRawWeight(const std::string& name, int N, int K, size_t raw_bytes)
{
    if (allocated_)
    {
        LOG_ERROR("Cannot plan weights after allocation");
        throw std::runtime_error("WeightVRAMPool: cannot plan after allocate()");
    }
    if (plans_.count(name))
    {
        LOG_ERROR("Duplicate weight name: " << name);
        throw std::runtime_error("WeightVRAMPool: duplicate weight name '" + name + "'");
    }

    WeightPlan plan;
    plan.N = N;
    plan.K = K;
    plan.staging_bytes = raw_bytes;

    // For raw FP weights, the entire allocation is in the payload region.
    // No scales/mins/emins needed.
    plan.payload_bytes = raw_bytes;
    plan.payload_offset = allocateRegion(raw_bytes);

    plans_[name] = plan;
    weight_order_.push_back(name);

    LOG_DEBUG("Planned raw weight '" << name << "' N=" << N << " K=" << K
              << " bytes=" << raw_bytes);
}

bool WeightVRAMPool::allocate(IBackend* backend, int device_id, int staging_slot_count)
{
    if (allocated_)
    {
        LOG_ERROR("WeightVRAMPool already allocated");
        return false;
    }

    backend_ = backend;
    device_id_ = device_id;

    // Compute staging region
    size_t max_staging = 0;
    staging_region_offset_ = alignUp(current_offset_, kAlignment);
    if (staging_slot_count > 0 && !plans_.empty())
    {
        for (const auto& [name, plan] : plans_)
        {
            max_staging = std::max(max_staging, plan.staging_bytes);
        }
        staging_region_bytes_ = max_staging * staging_slot_count;
    }
    staging_slot_count_ = staging_slot_count;
    max_staging_slot_bytes_ = max_staging;

    total_bytes_ = staging_region_offset_ + staging_region_bytes_;

    if (total_bytes_ == 0)
    {
        allocated_ = true;
        LOG_INFO("WeightVRAMPool: nothing to allocate (0 bytes)");
        return true;
    }

    if (backend_)
    {
        if (!backend_->setDevice(device_id_))
        {
            LOG_ERROR("WeightVRAMPool: setDevice(" << device_id_ << ") failed");
            return false;
        }

        d_base_ = backend_->allocate(total_bytes_, device_id_);
        if (!d_base_)
        {
            LOG_ERROR("WeightVRAMPool: allocate(" << total_bytes_ << " bytes) failed");
            return false;
        }
    }
    else
    {
        // No backend: CPU fallback for unit tests
        d_base_ = nullptr;
    }

    allocated_ = true;
    LOG_INFO("WeightVRAMPool: allocated " << total_bytes_ << " bytes on device "
             << device_id_ << " for " << plans_.size() << " weights"
             << " (staging: " << staging_region_bytes_ << " bytes, "
             << staging_slot_count << " slots)");
    return true;
}

std::optional<WeightVRAMPool::WeightSlot> WeightVRAMPool::getSlot(const std::string& name) const
{
    if (!allocated_) return std::nullopt;

    auto it = plans_.find(name);
    if (it == plans_.end()) return std::nullopt;

    const auto& plan = it->second;
    auto* base = static_cast<uint8_t*>(d_base_);

    WeightSlot slot;
    slot.payload_bytes = plan.payload_bytes;
    slot.staging_bytes = plan.staging_bytes;

    if (base)
    {
        slot.d_native_vnni_payload = base + plan.payload_offset;
        slot.d_native_vnni_scales = plan.scales_bytes > 0 ? base + plan.scales_offset : nullptr;
        slot.d_native_vnni_mins = plan.mins_bytes > 0 ? base + plan.mins_offset : nullptr;
        slot.d_native_vnni_emins = plan.emins_bytes > 0 ? base + plan.emins_offset : nullptr;
    }

    return slot;
}

size_t WeightVRAMPool::totalPlannedBytes() const
{
    if (allocated_)
        return total_bytes_;
    // Before allocation: current_offset_ tracks the weight regions planned so far
    return current_offset_;
}

size_t WeightVRAMPool::numPlannedWeights() const { return plans_.size(); }

bool WeightVRAMPool::isAllocated() const { return allocated_; }

int WeightVRAMPool::deviceId() const { return device_id_; }

void WeightVRAMPool::release()
{
    if (d_base_)
    {
        if (backend_)
        {
            backend_->setDevice(device_id_);
            backend_->free(d_base_, device_id_);
        }
        d_base_ = nullptr;
    }
    allocated_ = false;
    total_bytes_ = 0;
}

uint8_t* WeightVRAMPool::getStagingSlot(int slot_index) const
{
    if (!allocated_ || !d_base_ || slot_index < 0 || slot_index >= staging_slot_count_)
        return nullptr;
    if (max_staging_slot_bytes_ == 0)
        return nullptr;
    auto* base = static_cast<uint8_t*>(d_base_);
    return base + staging_region_offset_ + static_cast<size_t>(slot_index) * max_staging_slot_bytes_;
}

int WeightVRAMPool::stagingSlotCount() const { return staging_slot_count_; }

size_t WeightVRAMPool::maxStagingSlotBytes() const { return max_staging_slot_bytes_; }

} // namespace llaminar2
