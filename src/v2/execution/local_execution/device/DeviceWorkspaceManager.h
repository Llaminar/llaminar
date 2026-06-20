/**
 * @file DeviceWorkspaceManager.h
 * @brief Per-device workspace buffer management
 *
 * DeviceWorkspaceManager allocates workspace buffers within a memory budget
 * and provides named buffer access for kernels.
 *
 * Works with any device type: CPU, CUDA, ROCm.
 * (Formerly GpuWorkspaceManager.h)
 *
 * Design:
 * - One manager per device (DeviceId)
 * - Allocates contiguous block up front
 * - Suballocates named buffers from the block
 * - No reallocation during inference (hot path is zero-alloc)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "WorkspaceDescriptor.h"
#include "../../../backends/DeviceId.h"
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Manages pre-allocated workspace buffers for a single device
     *
     * DeviceWorkspaceManager provides centralized workspace buffer management for
     * kernels. Instead of each kernel allocating its own workspace, the
     * manager pre-allocates a single contiguous block and suballocates named
     * buffers within it.
     *
     * Works with any device type: CPU, CUDA, ROCm.
     * (Formerly GpuWorkspaceManager)
     *
     * **Usage**:
     * ```cpp
     * DeviceWorkspaceManager mgr(DeviceId::cuda(0), 256 * 1024 * 1024);  // 256MB budget
     *
     * WorkspaceRequirements reqs;
     * reqs.buffers.push_back({"gemm_workspace", 64 * 1024 * 1024});
     * reqs.buffers.push_back({"attention_scores", 32 * 1024 * 1024});
     *
     * if (mgr.allocate(reqs)) {
     *     void* gemm_ws = mgr.getBuffer("gemm_workspace");
     *     void* attn_scores = mgr.getBuffer("attention_scores");
     *     // ... use buffers ...
     * }
     *
     * mgr.release();  // Free all memory
     * ```
     *
     * **Thread Safety**: Not thread-safe. Caller must synchronize access.
     */
    class DeviceWorkspaceManager
    {
    public:
        /**
         * @brief Construct a workspace manager for a device
         * @param device Target device (CPU, CUDA, or ROCm)
         * @param budget_bytes Maximum bytes available for workspace
         */
        DeviceWorkspaceManager(DeviceId device, size_t budget_bytes);

        ~DeviceWorkspaceManager();

        // Non-copyable, non-movable (owns device memory)
        DeviceWorkspaceManager(const DeviceWorkspaceManager &) = delete;
        DeviceWorkspaceManager &operator=(const DeviceWorkspaceManager &) = delete;
        DeviceWorkspaceManager(DeviceWorkspaceManager &&) = delete;
        DeviceWorkspaceManager &operator=(DeviceWorkspaceManager &&) = delete;

        // =========================================================================
        // Allocation
        // =========================================================================

        /**
         * @brief Allocate workspace buffers from requirements
         * @param requirements Collection of buffer descriptors
         * @return true if all required buffers allocated, false on failure
         *
         * Allocates a single contiguous block from the backend and suballocates
         * named buffers at aligned offsets within the block.
         *
         * Note: Non-required buffers that don't fit are silently skipped.
         */
        bool allocate(const WorkspaceRequirements &requirements);

        /**
         * @brief Check if buffers have been allocated
         */
        bool isAllocated() const { return allocated_; }

        /**
         * @brief Release all allocated buffers
         *
         * Frees the underlying memory block and clears all buffer mappings.
         * After release(), isAllocated() returns false.
         */
        void release();

        // =========================================================================
        // Buffer Access
        // =========================================================================

        /**
         * @brief Get pointer to a named buffer
         * @param name Buffer name
         * @return Pointer to buffer (nullptr if not found)
         */
        void *getBuffer(const std::string &name) const;

        /**
         * @brief Get size of a named buffer
         * @param name Buffer name
         * @return Size in bytes (0 if not found)
         */
        size_t getBufferSize(const std::string &name) const;

        /**
         * @brief Check if a named buffer exists
         * @param name Buffer name
         */
        bool hasBuffer(const std::string &name) const;

        /**
         * @brief Get all buffer names
         */
        std::vector<std::string> bufferNames() const;

        // =========================================================================
        // Metrics
        // =========================================================================

        /**
         * @brief Get the device this manager is bound to
         */
        DeviceId device() const { return device_; }

        /**
         * @brief Monotonic host identity for this manager instance.
         *
         * Kernels that cache raw workspace sub-buffer pointers use this to
         * distinguish a genuinely unchanged manager from a new manager that
         * happened to be allocated at the same host address.
         */
        uint64_t id() const { return id_; }

        /**
         * @brief Get the total budget in bytes
         */
        size_t budget() const { return budget_bytes_; }

        /**
         * @brief Get the number of bytes used (including alignment padding)
         */
        size_t used() const { return used_bytes_; }

        /**
         * @brief Get the remaining budget in bytes
         */
        size_t remaining() const { return budget_bytes_ - used_bytes_; }

        /**
         * @brief Get the number of allocated buffers
         */
        size_t bufferCount() const { return buffers_.size(); }

    private:
        DeviceId device_;
        uint64_t id_;
        size_t budget_bytes_;
        size_t used_bytes_ = 0;
        bool allocated_ = false;

        // Main allocation block
        void *block_ = nullptr;
        size_t block_size_ = 0;

        // Named buffer offsets within block
        struct BufferInfo
        {
            size_t offset;
            size_t size;
        };
        std::unordered_map<std::string, BufferInfo> buffers_;

        /**
         * @brief Align offset to alignment boundary
         * @param offset Current offset
         * @param alignment Required alignment (must be power of 2)
         * @return Aligned offset >= input offset
         */
        static size_t alignUp(size_t offset, size_t alignment);

        /**
         * @brief Internal helper to allocate buffers
         * @param buffers Pointers to buffer descriptors
         * @param total_size Total size to allocate
         * @return true on success
         */
        bool allocateBuffers(
            const std::vector<const WorkspaceDescriptor *> &buffers,
            size_t total_size);
    };

} // namespace llaminar2
