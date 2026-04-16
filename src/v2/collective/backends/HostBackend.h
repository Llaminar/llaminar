/**
 * @file HostBackend.h
 * @brief CPU-based fallback collective backend
 *
 * The HostBackend provides CPU-mediated collective operations as a fallback
 * when GPU-specific backends (NCCL/RCCL) are unavailable. It is used for:
 * - Heterogeneous device groups (mixed CUDA/ROCm/CPU)
 * - Systems without GPU collective libraries
 * - Testing and development
 *
 * This is a simple single-threaded implementation prioritizing correctness
 * over performance. For production multi-GPU workloads, prefer NCCL/RCCL.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../ICollectiveBackend.h"
#include "../DeviceGroup.h"
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @brief CPU-based fallback collective backend
     *
     * Works as a fallback when GPU-specific backends are unavailable.
     * Supports all device types by staging through host memory.
     *
     * For single-process scenarios (which is the current implementation),
     * most operations are no-ops or simple memory copies.
     *
     * Thread Safety: Not thread-safe. Use one instance per thread.
     */
    class HostBackend : public ICollectiveBackend
    {
    public:
        HostBackend();
        ~HostBackend() override;

        // =====================================================================
        // Identity
        // =====================================================================

        CollectiveBackendType type() const override { return CollectiveBackendType::HOST; }
        std::string name() const override { return "Host"; }

        // =====================================================================
        // Capability Queries
        // =====================================================================

        /**
         * @brief Check if backend supports a device type
         *
         * HostBackend supports ALL device types via host memory staging.
         * This is what makes it the universal fallback.
         *
         * @param type Device type (CPU, CUDA, ROCm, etc.)
         * @return true for all device types
         */
        bool supportsDevice(DeviceType type) const override { return true; }

        /**
         * @brief Check if backend supports direct transfer between devices
         *
         * Direct transfer (no host staging) is only possible between CPU buffers.
         * GPU↔GPU transfers require staging through host memory.
         *
         * @param src Source device
         * @param dst Destination device
         * @return true only if both devices are CPU
         */
        bool supportsDirectTransfer(DeviceId src, DeviceId dst) const override;

        /**
         * @brief Check if backend is available
         *
         * HostBackend is always available as it has no external dependencies.
         *
         * @return Always true
         */
        bool isAvailable() const override { return true; }

        // =====================================================================
        // Lifecycle
        // =====================================================================

        /**
         * @brief Initialize backend for a device group
         *
         * Stores the group configuration for later operations.
         *
         * @param group Device group that will participate in collectives
         * @return true on success
         */
        bool initialize(const DeviceGroup &group) override;

        /**
         * @brief Check if backend is initialized
         * @return true if initialize() was called successfully
         */
        bool isInitialized() const override;

        /**
         * @brief Shutdown backend, release resources
         */
        void shutdown() override;

        // =====================================================================
        // Collective Operations
        // =====================================================================

        /**
         * @brief In-place AllReduce operation
         *
         * For single-device groups: no-op (data already "reduced")
         * For multi-device groups: placeholder (requires device memory access)
         *
         * @param buffer Device buffer (in-place, input and output)
         * @param count Number of elements
         * @param dtype Data type of elements
         * @param op Reduction operation (SUM, MAX, MIN)
         * @return true on success
         */
        bool allreduce(
            void *buffer,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op) override;

        /**
         * @brief Multi-buffer AllReduce operation
         *
         * Each buffers[i] is on group_.devices[i]. Host-staged reduction:
         * D2H each device buffer, reduce on host, H2D result back to each device.
         * This is the bridge-backend path used by HeterogeneousBackend Phase 2.
         *
         * @param buffers One device buffer per member of the group (in-place).
         * @param count Number of elements per buffer
         * @param dtype Data type of elements
         * @param op Reduction operation (SUM, MAX, MIN)
         * @return true on success
         */
        bool allreduceMulti(
            const std::vector<void *> &buffers,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op) override;

        /**
         * @brief AllGather operation
         *
         * For single-device groups: copies send_buf to recv_buf
         * For multi-device groups: placeholder
         *
         * @param send_buf Local slice to send
         * @param recv_buf Buffer for full gathered result
         * @param send_count Elements per device
         * @param dtype Data type
         * @return true on success
         */
        bool allgather(
            const void *send_buf,
            void *recv_buf,
            size_t send_count,
            CollectiveDataType dtype) override;

        /**
         * @brief Variable-count AllGather operation
         *
         * Each rank may send a different amount of data.
         * For single-device groups: copies send_buf to appropriate offset in recv_buf
         * For multi-device groups: gathers variable amounts from each device
         *
         * @param send_buf Local data to send
         * @param send_count Number of elements this rank sends
         * @param recv_buf Buffer to receive all data
         * @param recv_counts Array of counts per rank (size = world_size)
         * @param displacements Array of offsets in recv_buf per rank
         * @param dtype Data type
         * @return true on success
         */
        bool allgatherv(
            const void *send_buf,
            size_t send_count,
            void *recv_buf,
            const std::vector<int> &recv_counts,
            const std::vector<int> &displacements,
            CollectiveDataType dtype) override;

        /**
         * @brief ReduceScatter operation
         *
         * Reduce across all devices, then scatter result slices.
         * Currently a placeholder for multi-device scenarios.
         *
         * @param send_buf Full buffer to reduce
         * @param recv_buf Local slice of reduced result
         * @param recv_count Elements per device in result
         * @param dtype Data type
         * @param op Reduction operation
         * @return true on success
         */
        bool reduceScatter(
            const void *send_buf,
            void *recv_buf,
            size_t recv_count,
            CollectiveDataType dtype,
            CollectiveOp op) override;

        /**
         * @brief Broadcast from root device to all
         *
         * For single-device groups: no-op
         * For multi-device groups: placeholder
         *
         * @param buffer Buffer (root sends, others receive)
         * @param count Number of elements
         * @param dtype Data type
         * @param root_rank Rank of broadcasting device
         * @return true on success
         */
        bool broadcast(
            void *buffer,
            size_t count,
            CollectiveDataType dtype,
            int root_rank) override;

        /**
         * @brief Wait for all pending operations to complete
         *
         * No-op for CPU backend (operations are synchronous).
         *
         * @return Always true
         */
        bool synchronize() override { return true; }

        // =====================================================================
        // Device-to-Device Copy Operations
        // =====================================================================

        /**
         * @brief Copy data between any device pair via host staging
         *
         * Supports CPU\u2194CPU, CPU\u2194GPU, and GPU\u2194GPU transfers.
         * GPU\u2194GPU copies stage through a pre-allocated pinned host buffer
         * that is dual-registered with both CUDA and ROCm runtimes.
         *
         * @param dst_ptr Destination pointer (device or host memory)
         * @param dst_device Destination device
         * @param src_ptr Source pointer (device or host memory)
         * @param src_device Source device
         * @param bytes Number of bytes to copy
         * @return true on success
         */
        bool copy(
            void *dst_ptr, DeviceId dst_device,
            const void *src_ptr, DeviceId src_device,
            size_t bytes) override;

        /**
         * @brief Async copy (delegates to synchronous copy)
         *
         * All HostBackend copies are synchronous. The stream parameter
         * is accepted for interface compatibility but ignored.
         *
         * @param dst_ptr Destination pointer
         * @param dst_device Destination device
         * @param src_ptr Source pointer
         * @param src_device Source device
         * @param bytes Number of bytes to copy
         * @param stream Ignored
         * @return true on success
         */
        bool copyAsync(
            void *dst_ptr, DeviceId dst_device,
            const void *src_ptr, DeviceId src_device,
            size_t bytes, void *stream) override;

        /**
         * @brief Check if this backend supports copy between given device pair
         *
         * HostBackend supports any device pair via host staging.
         *
         * @param src_device Source device
         * @param dst_device Destination device
         * @return Always true
         */
        bool supportsCopy(DeviceId src_device, DeviceId dst_device) const override;

        // =====================================================================
        // Diagnostics
        // =====================================================================

        std::string lastError() const override { return last_error_; }

    private:
        /// Device group this backend operates on
        DeviceGroup group_;

        /// Initialization state
        bool initialized_ = false;

        /// Last error message
        std::string last_error_;

        /// Host staging buffer for reductions (dual-registered with both runtimes)
        void *staging_buffer_ = nullptr;
        size_t staging_buffer_size_ = 0;

        /// Track whether staging buffer is registered with each runtime
        /// (independent of has_cuda_/has_rocm_ which are cleared by shutdown())
        bool staging_registered_cuda_ = false;
        bool staging_registered_rocm_ = false;

        /// Track CUDA and ROCm device presence in group
        bool has_cuda_ = false;
        bool has_rocm_ = false;
        DeviceId cuda_device_;
        DeviceId rocm_device_;

        /**
         * @brief Get size in bytes for a collective data type
         * @param dtype Data type
         * @return Size in bytes per element
         */
        size_t elementSize(CollectiveDataType dtype) const;

        /**
         * @brief Copy data from GPU device to host staging buffer
         * @param host_dst Host destination pointer
         * @param device_src GPU source pointer
         * @param device Source device ID
         * @param bytes Number of bytes to copy
         * @return true on success
         */
        bool copyToHost(void *host_dst, const void *device_src, DeviceId device, size_t bytes);

        /**
         * @brief Copy data from host staging buffer to GPU device
         * @param device_dst GPU destination pointer
         * @param host_src Host source pointer
         * @param device Destination device ID
         * @param bytes Number of bytes to copy
         * @return true on success
         */
        bool copyFromHost(void *device_dst, const void *host_src, DeviceId device, size_t bytes);

        /**
         * @brief Perform reduction on host buffers
         * @param dst Destination buffer (accumulator)
         * @param src Source buffer to reduce into dst
         * @param count Number of elements
         * @param dtype Data type
         * @param op Reduction operation
         */
        void reduceOnHost(void *dst, const void *src, size_t count,
                          CollectiveDataType dtype, CollectiveOp op);

        /**
         * @brief Ensure staging buffer is large enough
         * @param required_bytes Minimum size needed
         * @return true if buffer is ready
         */
        bool ensureStagingBuffer(size_t required_bytes);
    };

} // namespace llaminar2
