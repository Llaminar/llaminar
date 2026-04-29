#include "loaders/gpu_pipeline/PinnedRingBuffer.h"
#include "backends/IBackend.h"
#include "utils/Logger.h"

#include <cstdlib>
#include <cstring>

namespace llaminar2
{

PinnedRingBuffer::PinnedRingBuffer(size_t slot_size, int num_slots)
    : slot_size_(slot_size), num_slots_(num_slots)
{
}

PinnedRingBuffer::~PinnedRingBuffer() { release(); }

PinnedRingBuffer::PinnedRingBuffer(PinnedRingBuffer&& other) noexcept
    : pinned_base_(other.pinned_base_),
      slot_size_(other.slot_size_),
      num_slots_(other.num_slots_),
      head_(other.head_),
      allocated_(other.allocated_),
      backend_(other.backend_),
      device_id_(other.device_id_)
{
    other.pinned_base_ = nullptr;
    other.allocated_ = false;
    other.backend_ = nullptr;
}

PinnedRingBuffer& PinnedRingBuffer::operator=(PinnedRingBuffer&& other) noexcept
{
    if (this != &other)
    {
        release();
        pinned_base_ = other.pinned_base_;
        slot_size_ = other.slot_size_;
        num_slots_ = other.num_slots_;
        head_ = other.head_;
        allocated_ = other.allocated_;
        backend_ = other.backend_;
        device_id_ = other.device_id_;
        other.pinned_base_ = nullptr;
        other.allocated_ = false;
        other.backend_ = nullptr;
    }
    return *this;
}

bool PinnedRingBuffer::allocate(IBackend* backend, int device_id)
{
    if (allocated_)
    {
        LOG_ERROR("PinnedRingBuffer already allocated");
        return false;
    }

    const size_t total = slot_size_ * num_slots_;
    if (total == 0)
    {
        LOG_ERROR("PinnedRingBuffer: zero-size allocation requested");
        return false;
    }

    backend_ = backend;
    device_id_ = device_id;

    if (backend_)
    {
        pinned_base_ = backend_->allocatePinned(total, device_id_);
        if (!pinned_base_)
        {
            LOG_ERROR("PinnedRingBuffer: allocatePinned(" << total << " bytes) failed");
            return false;
        }
    }
    else
    {
        // CPU fallback for unit tests
        pinned_base_ = std::malloc(total);
        if (!pinned_base_)
        {
            LOG_ERROR("malloc(" << total << " bytes) failed for PinnedRingBuffer");
            return false;
        }
        std::memset(pinned_base_, 0, total);
    }

    allocated_ = true;
    head_ = 0;
    LOG_DEBUG("PinnedRingBuffer: allocated " << total << " bytes (" << num_slots_
              << " slots x " << slot_size_ << " bytes)");
    return true;
}

void* PinnedRingBuffer::acquireSlot(int& slot_index)
{
    if (!allocated_ || !pinned_base_) return nullptr;
    slot_index = head_;
    return static_cast<uint8_t*>(pinned_base_) + static_cast<size_t>(head_) * slot_size_;
}

void PinnedRingBuffer::advance()
{
    head_ = (head_ + 1) % num_slots_;
}

void* PinnedRingBuffer::getSlot(int slot_index) const
{
    if (!allocated_ || !pinned_base_) return nullptr;
    if (slot_index < 0 || slot_index >= num_slots_) return nullptr;
    return static_cast<uint8_t*>(pinned_base_) + static_cast<size_t>(slot_index) * slot_size_;
}

size_t PinnedRingBuffer::slotSize() const { return slot_size_; }

int PinnedRingBuffer::numSlots() const { return num_slots_; }

bool PinnedRingBuffer::isAllocated() const { return allocated_; }

void PinnedRingBuffer::release()
{
    if (pinned_base_)
    {
        if (backend_)
        {
            backend_->freePinned(pinned_base_, device_id_);
        }
        else
        {
            std::free(pinned_base_);
        }
        pinned_base_ = nullptr;
    }
    allocated_ = false;
    head_ = 0;
}

} // namespace llaminar2
