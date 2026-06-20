#pragma once

#include <cstddef>

namespace llaminar2
{

class IBackend;

class PinnedRingBuffer
{
  public:
    /// @param slot_size  Bytes per slot.
    /// @param num_slots  Number of ring buffer slots (typically 3-4).
    PinnedRingBuffer(size_t slot_size, int num_slots);
    ~PinnedRingBuffer();

    PinnedRingBuffer(const PinnedRingBuffer&) = delete;
    PinnedRingBuffer& operator=(const PinnedRingBuffer&) = delete;
    PinnedRingBuffer(PinnedRingBuffer&& other) noexcept;
    PinnedRingBuffer& operator=(PinnedRingBuffer&& other) noexcept;

    /// Allocate pinned memory via IBackend (or malloc fallback if null).
    /// @param backend  GPU backend for pinned alloc (null = CPU fallback)
    /// @param device_id  Device ordinal (ignored for CPU fallback)
    bool allocate(IBackend* backend = nullptr, int device_id = 0);

    /// Acquire next slot. Returns pointer to slot memory, or nullptr if not allocated.
    /// @param slot_index  Set to the index of the acquired slot (0..num_slots-1).
    void* acquireSlot(int& slot_index);

    /// Advance to next slot (wrap around).
    void advance();

    /// Get slot by index.
    void* getSlot(int slot_index) const;

    size_t slotSize() const;
    int numSlots() const;
    bool isAllocated() const;

    void release();

  private:
    void* pinned_base_ = nullptr;
    size_t slot_size_ = 0;
    int num_slots_ = 0;
    int head_ = 0;
    bool allocated_ = false;
    IBackend* backend_ = nullptr;  // Track allocation backend for correct free
    int device_id_ = 0;
};

} // namespace llaminar2
