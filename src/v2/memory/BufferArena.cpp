/**
 * @file BufferArena.cpp
 * @brief Implementation of the central buffer management system
 */

#include "BufferArena.h"
#include "CoherenceTracker.h"
#include "tensors/TensorClasses.h"
#include "tensors/TensorFactory.h"
#include "tensors/ITensor.h"
#include "utils/Logger.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace llaminar2
{

    // =========================================================================
    // Internal helpers
    // =========================================================================

    BufferArena::ManagedBuffer &BufferArena::buf(BufferId id)
    {
        auto idx = static_cast<size_t>(id);
        assert(idx < kBufferCount && "BufferId out of range");
        auto &b = buffers_[idx];
        assert(b.registered && "Buffer not registered");
        return b;
    }

    const BufferArena::ManagedBuffer &BufferArena::buf(BufferId id) const
    {
        auto idx = static_cast<size_t>(id);
        assert(idx < kBufferCount && "BufferId out of range");
        auto &b = buffers_[idx];
        assert(b.registered && "Buffer not registered");
        return b;
    }

    ITensor *BufferArena::ManagedBuffer::tensor() const
    {
        if (owned_tensor)
            return owned_tensor.get();
        return external_tensor;
    }

    TensorBase *BufferArena::ManagedBuffer::tensorBase() const
    {
        if (owned_tensor)
            return owned_tensor.get();
        return dynamic_cast<TensorBase *>(external_tensor);
    }

    // =========================================================================
    // Registration
    // =========================================================================

    bool BufferArena::registerBuffer(BufferId id, size_t rows, size_t cols,
                                     const char *dtype, DeviceId device)
    {
        auto idx = static_cast<size_t>(id);
        if (idx >= kBufferCount)
            return false;

        auto &b = buffers_[idx];
        if (b.registered)
        {
            LOG_WARN("BufferArena: buffer " << bufferIdName(id) << " already registered");
            return false;
        }

        b.registered = true;
        b.rows = rows;
        b.cols = cols;
        b.dtype = dtype;
        b.home_device = device;
        b.coherence = {}; // UNINITIALIZED
        return true;
    }

    bool BufferArena::registerExternalBuffer(BufferId id, ITensor *tensor)
    {
        auto idx = static_cast<size_t>(id);
        if (idx >= kBufferCount)
            return false;

        auto &b = buffers_[idx];
        if (b.registered)
        {
            LOG_WARN("BufferArena: buffer " << bufferIdName(id) << " already registered");
            return false;
        }

        b.registered = true;
        b.external_tensor = tensor;
        if (tensor)
        {
            b.rows = tensor->rows();
            b.cols = tensor->cols();
            // External buffers start as HOST-authoritative (weights loaded on CPU)
            b.coherence.authority = CoherenceState::HOST;
        }
        return true;
    }

    void BufferArena::registerAlias(BufferId a, BufferId b)
    {
        auto idx_a = static_cast<size_t>(a);
        auto idx_b = static_cast<size_t>(b);
        assert(idx_a < kBufferCount && idx_b < kBufferCount);

        auto &ba = buffers_[idx_a];
        auto &bb = buffers_[idx_b];

        // Assign to same alias group
        if (ba.alias_group >= 0 && bb.alias_group >= 0)
        {
            // Both already in groups — merge by rewriting all of b's group to a's
            int old_group = bb.alias_group;
            int new_group = ba.alias_group;
            for (auto &buf : buffers_)
            {
                if (buf.alias_group == old_group)
                    buf.alias_group = new_group;
            }
        }
        else if (ba.alias_group >= 0)
        {
            bb.alias_group = ba.alias_group;
        }
        else if (bb.alias_group >= 0)
        {
            ba.alias_group = bb.alias_group;
        }
        else
        {
            // Neither in a group — create new
            int g = next_alias_group_++;
            ba.alias_group = g;
            bb.alias_group = g;
        }
    }

    // =========================================================================
    // Allocation
    // =========================================================================

    bool BufferArena::allocate()
    {
        if (allocated_)
        {
            LOG_WARN("BufferArena: already allocated");
            return false;
        }

        // For now, we only create FP32 tensors for arena-owned buffers.
        // Additional dtype support can be added incrementally.
        for (size_t i = 0; i < kBufferCount; ++i)
        {
            auto &b = buffers_[i];
            if (!b.registered)
                continue;
            if (b.external_tensor)
                continue; // External — already has storage
            if (b.owned_tensor)
                continue; // Should not happen

            // Create an FP32Tensor for the registered shape
            // TODO: support more dtypes as stages are migrated
            auto tensor = std::make_shared<FP32Tensor>(
                std::vector<size_t>{b.rows, b.cols}, b.home_device);

            b.owned_tensor = tensor;
            b.coherence.authority = CoherenceState::UNINITIALIZED;
        }

        allocated_ = true;
        return true;
    }

    // =========================================================================
    // Runtime coherence
    // =========================================================================

    bool BufferArena::prepareForRead(BufferId id, DeviceId target)
    {
        auto &b = buf(id);
        return CoherenceTracker::prepareForRead(b.tensorBase(), b.coherence, target);
    }

    bool BufferArena::prepareForWrite(BufferId id, DeviceId target)
    {
        auto &b = buf(id);
        return CoherenceTracker::prepareForWrite(b.tensorBase(), b.coherence, target);
    }

    void BufferArena::markWritten(BufferId id, DeviceId device, void *stream)
    {
        auto &b = buf(id);
        CoherenceTracker::markWrittenWithEvent(b.tensorBase(), b.coherence, device, stream);
    }

    void BufferArena::markWrittenFlagsOnly(BufferId id, DeviceId device)
    {
        auto &b = buf(id);
        CoherenceTracker::markWrittenFlagsOnly(b.tensorBase(), b.coherence, device);
    }

    // =========================================================================
    // Borrow tracking
    // =========================================================================

    void BufferArena::acquireReadBorrow(BufferId id)
    {
#ifndef NDEBUG
        validateBorrowSafe(id, BufferAccess::READ);
#endif
        buf(id).active_read_borrows++;
    }

    void BufferArena::acquireWriteBorrow(BufferId id)
    {
#ifndef NDEBUG
        validateBorrowSafe(id, BufferAccess::WRITE);
#endif
        buf(id).active_write_borrow = true;
    }

    void BufferArena::releaseReadBorrow(BufferId id)
    {
        auto &b = buf(id);
        assert(b.active_read_borrows > 0 && "Releasing read borrow that wasn't acquired");
        b.active_read_borrows--;
    }

    void BufferArena::releaseWriteBorrow(BufferId id)
    {
        auto &b = buf(id);
        assert(b.active_write_borrow && "Releasing write borrow that wasn't acquired");
        b.active_write_borrow = false;
    }

    bool BufferArena::validateNoBorrowsActive() const
    {
        for (size_t i = 0; i < kBufferCount; ++i)
        {
            const auto &b = buffers_[i];
            if (!b.registered)
                continue;
            if (b.active_read_borrows > 0 || b.active_write_borrow)
            {
                LOG_ERROR("BufferArena: buffer " << bufferIdName(static_cast<BufferId>(i))
                                                 << " has active borrows (read=" << b.active_read_borrows
                                                 << " write=" << b.active_write_borrow << ")");
                return false;
            }
        }
        return true;
    }

    void BufferArena::validateBorrowSafe(BufferId id, BufferAccess access) const
    {
        auto idx = static_cast<size_t>(id);
        const auto &b = buffers_[idx];

        if (access == BufferAccess::READ)
        {
            // Read borrows can coexist with other reads, but not with writes
            if (b.active_write_borrow)
            {
                LOG_ERROR("BufferArena: READ borrow on " << bufferIdName(id)
                                                         << " while WRITE borrow is active");
                assert(false && "active write borrow conflicts with read borrow");
            }
            // Check alias group for write borrows
            if (b.alias_group >= 0 && aliasGroupHasWriteBorrow(b.alias_group, id))
            {
                LOG_ERROR("BufferArena: READ borrow on " << bufferIdName(id)
                                                         << " while aliased buffer has WRITE borrow");
                assert(false && "aliased buffer has active write borrow");
            }
        }
        else
        {
            // Write borrows conflict with everything
            if (b.active_read_borrows > 0)
            {
                LOG_ERROR("BufferArena: WRITE borrow on " << bufferIdName(id)
                                                          << " while " << b.active_read_borrows
                                                          << " READ borrows are active");
                assert(false && "active read borrow conflicts with write borrow");
            }
            if (b.active_write_borrow)
            {
                LOG_ERROR("BufferArena: WRITE borrow on " << bufferIdName(id)
                                                          << " while another WRITE borrow is active");
                assert(false && "double write borrow");
            }
            // Check alias group for any borrows
            if (b.alias_group >= 0 && aliasGroupHasAnyBorrow(b.alias_group, id))
            {
                LOG_ERROR("BufferArena: WRITE borrow on " << bufferIdName(id)
                                                          << " while aliased buffer has active borrow");
                assert(false && "aliased buffer has active borrow conflicts with write");
            }
        }
    }

    bool BufferArena::aliasGroupHasWriteBorrow(int group, BufferId exclude) const
    {
        for (size_t i = 0; i < kBufferCount; ++i)
        {
            if (static_cast<BufferId>(i) == exclude)
                continue;
            const auto &b = buffers_[i];
            if (b.registered && b.alias_group == group && b.active_write_borrow)
                return true;
        }
        return false;
    }

    bool BufferArena::aliasGroupHasAnyBorrow(int group, BufferId exclude) const
    {
        for (size_t i = 0; i < kBufferCount; ++i)
        {
            if (static_cast<BufferId>(i) == exclude)
                continue;
            const auto &b = buffers_[i];
            if (b.registered && b.alias_group == group &&
                (b.active_read_borrows > 0 || b.active_write_borrow))
                return true;
        }
        return false;
    }

    // =========================================================================
    // Buffer access
    // =========================================================================

    ITensor *BufferArena::getTensor(BufferId id) const
    {
        if (!isRegistered(id))
            return nullptr;
        return buf(id).tensor();
    }

    void *BufferArena::getDevicePtr(BufferId id, DeviceId target) const
    {
        const auto &b = buf(id);
        auto *t = b.tensorBase();
        if (!t)
            return nullptr;

        if (target.is_gpu())
        {
            return t->gpu_data_ptr();
        }
        else
        {
            // raw_data() is the public const accessor for host memory
            return const_cast<void *>(t->raw_data());
        }
    }

    size_t BufferArena::getRows(BufferId id) const
    {
        return buf(id).rows;
    }

    size_t BufferArena::getCols(BufferId id) const
    {
        return buf(id).cols;
    }

    // =========================================================================
    // Introspection
    // =========================================================================

    CoherenceState BufferArena::getCoherenceState(BufferId id) const
    {
        return buf(id).coherence;
    }

    bool BufferArena::isRegistered(BufferId id) const
    {
        auto idx = static_cast<size_t>(id);
        if (idx >= kBufferCount)
            return false;
        return buffers_[idx].registered;
    }

    size_t BufferArena::registeredCount() const
    {
        size_t count = 0;
        for (const auto &b : buffers_)
        {
            if (b.registered)
                ++count;
        }
        return count;
    }

} // namespace llaminar2
