/**
 * @file BufferAccess.h
 * @brief Buffer access mode enum and BufferView RAII handle
 *
 * BufferView provides typed, access-controlled views into device-ready memory.
 * Stages receive BufferView handles from StageBoundBuffers — they cannot
 * call data(), ensureOnDevice(), or transitionTo() because BufferView
 * simply doesn't expose those methods.
 */

#pragma once

#include <cstddef>
#include "BufferId.h"
#include "backends/DeviceId.h"

namespace llaminar2
{

    // Forward declarations
    class ITensor;

    /**
     * @brief Access mode for buffer borrows.
     *
     * Controls which pointer accessors are available on BufferView.
     */
    enum class BufferAccess : uint8_t
    {
        READ,      ///< Read-only: read_ptr() only
        WRITE,     ///< Write-only: write_ptr() only (data undefined on entry)
        READWRITE, ///< Both read and write access
    };

    /**
     * @brief RAII typed view into a device-ready buffer.
     *
     * Returned by StageBoundBuffers for use inside stage execute().
     * The view holds a pointer that is already on the correct device.
     * No coherence, no data(), no ensureOnDevice().
     *
     * Access control is compile-time via C++20 requires clauses:
     * - READ  views expose read_ptr()  only.
     * - WRITE views expose write_ptr() only.
     * - READWRITE views expose both.
     *
     * @tparam T Element type (float, Q8_1Block, etc.)
     * @tparam Access BufferAccess mode
     */
    template <typename T, BufferAccess Access>
    class BufferView
    {
    public:
        BufferView() = default;

        BufferView(void *device_ptr, ITensor *tensor, size_t rows, size_t cols, DeviceId device)
            : device_ptr_(device_ptr), tensor_(tensor), rows_(rows), cols_(cols), device_(device) {}

        /// Read pointer (available for READ and READWRITE)
        const T *read_ptr() const
            requires(Access != BufferAccess::WRITE)
        {
            return static_cast<const T *>(device_ptr_);
        }

        /// Write pointer (available for WRITE and READWRITE)
        T *write_ptr()
            requires(Access != BufferAccess::READ)
        {
            return static_cast<T *>(device_ptr_);
        }

        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
        size_t numel() const { return rows_ * cols_; }
        DeviceId device() const { return device_; }

        /// Get underlying tensor for kernel dispatch (e.g., KernelFactory lookup)
        ITensor *tensor() const { return tensor_; }

        /// Check validity
        bool valid() const { return device_ptr_ != nullptr; }
        explicit operator bool() const { return valid(); }

    private:
        void *device_ptr_ = nullptr;
        ITensor *tensor_ = nullptr;
        size_t rows_ = 0;
        size_t cols_ = 0;
        DeviceId device_;
    };

} // namespace llaminar2
