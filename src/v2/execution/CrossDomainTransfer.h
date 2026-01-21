/**
 * @file CrossDomainTransfer.h
 * @brief Cross-domain activation transfer utility for heterogeneous execution
 *
 * Handles activation transfers between GPU and CPU domains when execution
 * transitions between devices. This is critical for:
 * - CPU offload scenarios (first/last layers on CPU, middle on GPU)
 * - Memory-constrained systems with partial GPU offload
 * - Hybrid compute where different layers benefit from different devices
 *
 * Optimizations:
 * - Uses pinned memory for faster PCIe transfers (DMA-enabled)
 * - NUMA-aware CPU allocation for multi-socket systems
 * - Leverages existing tensor coherence protocol when possible
 * - Async transfers when supported (optional)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../tensors/Tensors.h"
#include "../backends/DeviceId.h"
#include "../memory/NUMAAllocator.h"
#include <chrono>
#include <memory>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;

    /**
     * @brief Cross-domain activation transfer utility
     *
     * Handles efficient activation data transfer between GPU and CPU domains.
     * Optimized for the common patterns in heterogeneous transformer inference:
     * - GPU→CPU: After GPU layers, before CPU layers
     * - CPU→GPU: After CPU layers, before GPU layers
     *
     * Uses the tensor coherence protocol where possible, falling back to
     * explicit memcpy when needed.
     *
     * Thread safety: Not thread-safe. Designed for single-threaded pipeline execution.
     *
     * @code
     * CrossDomainTransfer::Config config;
     * config.use_pinned_memory = true;
     * config.numa_allocator = &numa_alloc;
     * config.target_numa_node = 0;
     *
     * CrossDomainTransfer transfer(config);
     *
     * // Transfer GPU tensor to CPU
     * transfer.gpuToCpu(gpu_tensor, cpu_tensor);
     *
     * // Or with automatic allocation
     * auto cpu_copy = transfer.transfer(gpu_tensor, DeviceType::CPU);
     * @endcode
     */
    class CrossDomainTransfer
    {
    public:
        /**
         * @brief Configuration for cross-domain transfers
         */
        struct Config
        {
            /// Use pinned (page-locked) memory for faster PCIe transfers
            /// When true, host memory is registered for DMA access
            bool use_pinned_memory = true;

            /// Enable async transfers (requires stream management)
            /// When true, transfers return immediately; call synchronize() before using data
            bool async_transfers = false;

            /// NUMA allocator for CPU destination tensors (optional)
            /// If nullptr, uses standard allocation
            NUMAAllocator *numa_allocator = nullptr;

            /// Target NUMA node for CPU allocations (-1 for local node)
            int target_numa_node = -1;
        };

        /// Returns a default Config instance
        static Config defaultConfig()
        {
            Config cfg;
            cfg.use_pinned_memory = true;
            cfg.async_transfers = false;
            cfg.numa_allocator = nullptr;
            cfg.target_numa_node = -1;
            return cfg;
        }

        /**
         * @brief Transfer statistics for performance analysis
         */
        struct TransferStats
        {
            int gpu_to_cpu_count = 0;        ///< Number of GPU→CPU transfers
            int cpu_to_gpu_count = 0;        ///< Number of CPU→GPU transfers
            size_t gpu_to_cpu_bytes = 0;     ///< Total bytes transferred GPU→CPU
            size_t cpu_to_gpu_bytes = 0;     ///< Total bytes transferred CPU→GPU
            double gpu_to_cpu_time_ms = 0.0; ///< Total time in GPU→CPU transfers (ms)
            double cpu_to_gpu_time_ms = 0.0; ///< Total time in CPU→GPU transfers (ms)

            /// Get GPU→CPU bandwidth in GB/s (0 if no transfers)
            double gpuToCpuBandwidthGBps() const
            {
                if (gpu_to_cpu_time_ms <= 0)
                    return 0.0;
                return (gpu_to_cpu_bytes / 1e9) / (gpu_to_cpu_time_ms / 1e3);
            }

            /// Get CPU→GPU bandwidth in GB/s (0 if no transfers)
            double cpuToGpuBandwidthGBps() const
            {
                if (cpu_to_gpu_time_ms <= 0)
                    return 0.0;
                return (cpu_to_gpu_bytes / 1e9) / (cpu_to_gpu_time_ms / 1e3);
            }

            /// Reset all statistics
            void reset()
            {
                gpu_to_cpu_count = 0;
                cpu_to_gpu_count = 0;
                gpu_to_cpu_bytes = 0;
                cpu_to_gpu_bytes = 0;
                gpu_to_cpu_time_ms = 0.0;
                cpu_to_gpu_time_ms = 0.0;
            }
        };

        /**
         * @brief Construct with configuration
         * @param config Transfer configuration
         */
        explicit CrossDomainTransfer(Config config = defaultConfig());

        /**
         * @brief Destructor - waits for pending async transfers
         */
        ~CrossDomainTransfer();

        // Non-copyable, movable
        CrossDomainTransfer(const CrossDomainTransfer &) = delete;
        CrossDomainTransfer &operator=(const CrossDomainTransfer &) = delete;
        CrossDomainTransfer(CrossDomainTransfer &&) noexcept = default;
        CrossDomainTransfer &operator=(CrossDomainTransfer &&) noexcept = default;

        // =========================================================================
        // Transfer Operations
        // =========================================================================

        /**
         * @brief Transfer tensor data from GPU to CPU
         *
         * Copies data from a GPU-resident tensor to a CPU-resident tensor.
         * Uses the tensor coherence protocol: calls src->data() which syncs
         * GPU→host if needed, then memcpy to destination.
         *
         * @param src Source tensor (must have valid GPU data)
         * @param dst Destination tensor (must be pre-allocated with matching size)
         * @return true on success, false on error
         *
         * @note dst must have at least src->byte_size() bytes allocated
         * @note If src is already host-valid (not device-dirty), this is a simple memcpy
         */
        bool gpuToCpu(const TensorBase *src, TensorBase *dst);

        /**
         * @brief Transfer tensor data from CPU to GPU
         *
         * Copies data from a CPU-resident tensor to a GPU-resident tensor.
         * Ensures destination is allocated on GPU, then copies via backend API.
         *
         * @param src Source tensor (CPU-resident)
         * @param dst Destination tensor (must be pre-allocated)
         * @param device_id Target GPU device for the transfer
         * @return true on success, false on error
         *
         * @note dst must have at least src->byte_size() bytes allocated
         * @note After this call, dst has valid GPU data (device_valid_ = true)
         */
        bool cpuToGpu(const TensorBase *src, TensorBase *dst, DeviceId device_id);

        /**
         * @brief Transfer with automatic destination allocation
         *
         * Creates a new tensor on the target device and transfers data to it.
         * The returned tensor has the same shape and element count as src.
         *
         * @param src Source tensor
         * @param target_device Target device type (CPU or GPU)
         * @param target_device_id Target device ID (required for GPU, ignored for CPU)
         * @return Newly allocated tensor on target device, or nullptr on failure
         *
         * @note For CPU targets, uses NUMA allocator if configured
         * @note For GPU targets, allocates on specified device_id
         */
        std::unique_ptr<TensorBase> transfer(const TensorBase *src,
                                             DeviceType target_device,
                                             DeviceId target_device_id = DeviceId::cpu());

        /**
         * @brief Synchronize all pending async transfers
         *
         * Blocks until all async transfers initiated with async_transfers=true
         * have completed. Safe to call even if no async transfers pending.
         */
        void synchronize();

        // =========================================================================
        // Statistics
        // =========================================================================

        /**
         * @brief Get transfer statistics
         * @return Current transfer statistics
         */
        TransferStats getStats() const { return stats_; }

        /**
         * @brief Reset transfer statistics
         */
        void resetStats() { stats_ = TransferStats{}; }

        /**
         * @brief Get configuration
         * @return Current configuration
         */
        const Config &config() const { return config_; }

    private:
        // =========================================================================
        // Internal Implementation
        // =========================================================================

        /**
         * @brief Low-level GPU→CPU transfer implementation
         * @param src Source GPU pointer
         * @param dst Destination host pointer
         * @param bytes Number of bytes to transfer
         * @param src_device Source GPU device
         * @return true on success
         */
        bool transferGpuToCpuImpl(const void *src, void *dst, size_t bytes, DeviceId src_device);

        /**
         * @brief Low-level CPU→GPU transfer implementation
         * @param src Source host pointer
         * @param dst Destination GPU pointer
         * @param bytes Number of bytes to transfer
         * @param dst_device Destination GPU device
         * @return true on success
         */
        bool transferCpuToGpuImpl(const void *src, void *dst, size_t bytes, DeviceId dst_device);

        /**
         * @brief Allocate a CPU tensor with NUMA awareness
         * @param shape Tensor shape
         * @return Allocated tensor or nullptr on failure
         */
        std::unique_ptr<TensorBase> allocateCpuTensor(const std::vector<size_t> &shape);

        Config config_;
        TransferStats stats_;
    };

} // namespace llaminar2
