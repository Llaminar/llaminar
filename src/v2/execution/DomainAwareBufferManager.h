/**
 * @file DomainAwareBufferManager.h
 * @brief Domain-aware buffer allocation for heterogeneous execution
 *
 * Extends buffer management with device-aware allocation based on layer placement.
 * This enables:
 * - GPU layers: Buffers allocated in GPU memory (CUDA/ROCm)
 * - CPU layers: Buffers allocated as NUMA-local host memory
 *
 * Key integration points:
 * - LayerPlacementConfig: Determines which device each layer runs on
 * - NUMAAllocator: NUMA-aware allocation for CPU buffers
 * - TensorFactory: Tensor creation with device affinity
 *
 * @author David Sanftenberg
 * @date 2026-01-21
 */

#pragma once

#include "GraphBufferManager.h"
#include "BufferRole.h" // For BufferTensorType
#include "../config/LayerPlacementConfig.h"
#include "../memory/NUMAAllocator.h"
#include "../tensors/TensorFactory.h"
#include "../backends/DeviceId.h"
#include "../utils/MPIContext.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    // Note: TensorBase is already defined as a typedef in GraphBufferManager.h

    /**
     * @brief Configuration for DomainAwareBufferManager
     *
     * Configures the domain-aware allocation behavior including
     * device placement and NUMA settings.
     */
    struct DomainAwareBufferConfig
    {
        /// Layer placement configuration (required for allocateForLayer)
        /// If nullptr, allocateForLayer will use default_device
        LayerPlacementConfig *placement_config = nullptr;

        /// NUMA allocator for CPU buffer allocation (optional)
        /// If nullptr, standard allocation is used
        NUMAAllocator *numa_allocator = nullptr;

        /// Default NUMA node for CPU allocations (-1 for local node)
        int default_numa_node = -1;

        /// Default device when placement_config is nullptr
        DeviceId default_device = DeviceId::cpu();

        /// MPI context for rank-aware allocation
        const MPIContext *mpi_ctx = nullptr;
    };

    /**
     * @brief Allocation statistics for domain-aware buffer manager
     */
    struct DomainAllocationStats
    {
        size_t gpu_bytes_allocated = 0;
        size_t cpu_bytes_allocated = 0;
        int gpu_buffer_count = 0;
        int cpu_buffer_count = 0;

        /// Per-device breakdown
        std::unordered_map<DeviceId, size_t> bytes_per_device;
        std::unordered_map<DeviceId, int> buffers_per_device;

        /// NUMA node breakdown for CPU allocations
        std::unordered_map<int, size_t> bytes_per_numa_node;

        void reset()
        {
            gpu_bytes_allocated = 0;
            cpu_bytes_allocated = 0;
            gpu_buffer_count = 0;
            cpu_buffer_count = 0;
            bytes_per_device.clear();
            buffers_per_device.clear();
            bytes_per_numa_node.clear();
        }

        size_t total_bytes() const
        {
            return gpu_bytes_allocated + cpu_bytes_allocated;
        }

        int total_buffers() const
        {
            return gpu_buffer_count + cpu_buffer_count;
        }
    };

    /**
     * @brief Domain-aware buffer manager for heterogeneous execution
     *
     * Extends GraphBufferManager with device-aware allocation based on
     * layer placement configuration. Automatically routes buffer allocation
     * to the correct device (GPU or CPU) based on which device will
     * execute the layer.
     *
     * ## Usage Example
     *
     * @code
     * // Create placement config: layers 0-3 on CPU, rest on GPU
     * auto placement = LayerPlacementConfig::cpuFirstLayers(4, 28, DeviceId::cuda(0));
     *
     * // Create domain-aware buffer manager
     * DomainAwareBufferConfig config;
     * config.placement_config = &placement;
     * config.numa_allocator = &NUMAAllocator::instance();
     * config.mpi_ctx = &mpi_ctx;
     *
     * DomainAwareBufferManager manager(config);
     *
     * // Allocate buffer for layer 0 (will be on CPU/NUMA)
     * auto* cpu_buffer = manager.allocateForLayer(0, "attention_output", {seq_len, hidden}, DataType::FP32);
     *
     * // Allocate buffer for layer 5 (will be on GPU)
     * auto* gpu_buffer = manager.allocateForLayer(5, "attention_output", {seq_len, hidden}, DataType::FP32);
     * @endcode
     *
     * ## Thread Safety
     *
     * NOT thread-safe. Should be used from a single thread or with
     * external synchronization.
     */
    class DomainAwareBufferManager
    {
    public:
        /**
         * @brief Construct domain-aware buffer manager
         * @param config Configuration specifying placement and NUMA settings
         */
        explicit DomainAwareBufferManager(DomainAwareBufferConfig config);

        /**
         * @brief Destructor - releases all owned buffers
         */
        ~DomainAwareBufferManager();

        // Non-copyable
        DomainAwareBufferManager(const DomainAwareBufferManager &) = delete;
        DomainAwareBufferManager &operator=(const DomainAwareBufferManager &) = delete;

        // Movable
        DomainAwareBufferManager(DomainAwareBufferManager &&) = default;
        DomainAwareBufferManager &operator=(DomainAwareBufferManager &&) = default;

        // =========================================================================
        // Layer-Aware Allocation
        // =========================================================================

        /**
         * @brief Allocate buffer for a specific layer
         *
         * Queries the placement configuration to determine the device for
         * this layer, then allocates the buffer on that device.
         *
         * @param layer_idx Layer index (0-based)
         * @param buffer_name Buffer name (for tracking/debugging)
         * @param shape Tensor dimensions
         * @param dtype Data type (BufferTensorType)
         * @return Pointer to allocated tensor (owned by this manager)
         * @throws std::out_of_range if layer_idx is not in placement config
         */
        TensorBase *allocateForLayer(int layer_idx, const std::string &buffer_name,
                                     const std::vector<size_t> &shape, BufferTensorType dtype);

        // =========================================================================
        // Device-Specific Allocation
        // =========================================================================

        /**
         * @brief Allocate buffer on a specific device
         *
         * Directly allocates on the specified device without consulting
         * the placement configuration.
         *
         * @param device Target device (CPU, CUDA, ROCm)
         * @param buffer_name Buffer name (for tracking/debugging)
         * @param shape Tensor dimensions
         * @param dtype Data type (BufferTensorType)
         * @return Pointer to allocated tensor (owned by this manager)
         */
        TensorBase *allocateOnDevice(DeviceId device, const std::string &buffer_name,
                                     const std::vector<size_t> &shape, BufferTensorType dtype);

        /**
         * @brief Allocate NUMA-local CPU buffer
         *
         * Allocates a CPU buffer with NUMA-local placement. Uses NUMAAllocator
         * if available, otherwise falls back to standard allocation.
         *
         * @param numa_node NUMA node (-1 for local node)
         * @param buffer_name Buffer name (for tracking/debugging)
         * @param shape Tensor dimensions
         * @param dtype Data type (BufferTensorType)
         * @return Pointer to allocated tensor (owned by this manager)
         */
        TensorBase *allocateNUMALocal(int numa_node, const std::string &buffer_name,
                                      const std::vector<size_t> &shape, BufferTensorType dtype);

        // =========================================================================
        // Query Methods
        // =========================================================================

        /**
         * @brief Get device for a layer's buffers
         *
         * Delegates to placement configuration if available.
         *
         * @param layer_idx Layer index
         * @return Device for this layer
         */
        DeviceId getDeviceForLayer(int layer_idx) const;

        /**
         * @brief Check if a layer runs on GPU
         * @param layer_idx Layer index
         * @return true if layer runs on CUDA or ROCm device
         */
        bool isGPULayer(int layer_idx) const;

        /**
         * @brief Check if a layer runs on CPU
         * @param layer_idx Layer index
         * @return true if layer runs on CPU
         */
        bool isCPULayer(int layer_idx) const;

        // =========================================================================
        // Buffer Retrieval
        // =========================================================================

        /**
         * @brief Get a previously allocated buffer
         * @param buffer_name Buffer name
         * @return Pointer to tensor (nullptr if not found)
         */
        TensorBase *getBuffer(const std::string &buffer_name);

        /**
         * @brief Check if a buffer exists
         * @param buffer_name Buffer name
         * @return true if buffer is allocated
         */
        bool hasBuffer(const std::string &buffer_name) const;

        // =========================================================================
        // Lifecycle Management
        // =========================================================================

        /**
         * @brief Release all managed buffers
         */
        void releaseAll();

        // =========================================================================
        // Statistics
        // =========================================================================

        /**
         * @brief Get allocation statistics
         */
        const DomainAllocationStats &getStats() const { return stats_; }

        /**
         * @brief Reset statistics
         */
        void resetStats() { stats_.reset(); }

        /**
         * @brief Get number of allocated buffers
         */
        size_t bufferCount() const { return owned_buffers_.size(); }

        /**
         * @brief Dump buffer inventory to log
         */
        void dumpBufferInventory() const;

    private:
        DomainAwareBufferConfig config_;
        DomainAllocationStats stats_;

        /// Buffer storage: name -> owned tensor
        std::unordered_map<std::string, std::unique_ptr<TensorBase>> owned_buffers_;

        /// TensorFactory for creating tensors (owned)
        std::unique_ptr<TensorFactory> tensor_factory_;

        // Internal helpers
        std::unique_ptr<TensorBase> createTensorOnDevice(DeviceId device,
                                                         const std::vector<size_t> &shape,
                                                         BufferTensorType dtype);
        void updateStats(DeviceId device, size_t bytes, int numa_node = -1);
        size_t computeTensorBytes(const std::vector<size_t> &shape, BufferTensorType dtype) const;
    };

} // namespace llaminar2
