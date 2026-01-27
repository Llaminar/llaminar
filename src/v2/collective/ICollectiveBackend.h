/**
 * @file ICollectiveBackend.h
 * @brief Abstract interface for collective communication backends
 *
 * INTERNAL IMPLEMENTATION - Not exposed to model graphs!
 *
 * This is an internal interface used by CollectiveContext to execute
 * collective operations. Model graphs (like Qwen2Graph) do NOT interact
 * with this interface directly - they use abstract AllreduceStage/AllGatherStage
 * and the GraphExecutor handles backend selection via CollectiveContext.
 *
 * Provides a unified interface for collective operations (AllReduce, AllGather, etc.)
 * across different backends:
 * - MPI: Inter-node communication (CPU-mediated)
 * - NCCL: Intra-node NVIDIA GPU communication
 * - RCCL: Intra-node AMD GPU communication
 * - Host: Heterogeneous fallback via host memory staging
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../backends/DeviceId.h"
#include "../backends/DeviceType.h"
#include "../config/OrchestrationConfig.h" // For CollectiveBackendType (canonical definition)
#include "DeviceGroup.h"
#include "IBufferRegistration.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    // CollectiveBackendType is now defined in OrchestrationConfig.h
    // to avoid duplicate definitions. See config/OrchestrationConfig.h.

    /**
     * @brief Collective operation types
     */
    enum class CollectiveOp
    {
        ALLREDUCE_SUM,  ///< Sum reduction across all devices
        ALLREDUCE_MAX,  ///< Max reduction across all devices
        ALLREDUCE_MIN,  ///< Min reduction across all devices
        ALLGATHER,      ///< Gather slices from all devices into full buffer
        REDUCE_SCATTER, ///< Reduce then scatter result slices
        BROADCAST       ///< Broadcast from one device to all
    };

    /**
     * @brief Data types for collective operations
     */
    enum class CollectiveDataType
    {
        FLOAT32,
        FLOAT16,
        BFLOAT16,
        INT32,
        INT8
    };

    /**
     * @brief Convert string to CollectiveBackendType
     *
     * Note: This is a convenience wrapper. Prefer parseCollectiveBackendType()
     * from OrchestrationConfig.h for new code.
     */
    inline CollectiveBackendType parseBackendType(const std::string &s)
    {
        // Delegate to the canonical parser
        auto result = parseCollectiveBackendType(s);
        if (result)
            return *result;

        // Fallback: handle PCIE_BAR variations (case-sensitive upper-case)
        if (s == "PCIE_BAR" || s == "PCIe_BAR")
            return CollectiveBackendType::PCIE_BAR;

        return CollectiveBackendType::AUTO;
    }

    /**
     * @brief Convert CollectiveBackendType to string
     *
     * Note: This is a convenience wrapper. Prefer collectiveBackendTypeToString()
     * from OrchestrationConfig.h for new code.
     */
    inline std::string toString(CollectiveBackendType type)
    {
        return collectiveBackendTypeToString(type);
    }

    /**
     * @brief Convert CollectiveOp to string
     */
    inline std::string toString(CollectiveOp op)
    {
        switch (op)
        {
        case CollectiveOp::ALLREDUCE_SUM:
            return "AllReduceSum";
        case CollectiveOp::ALLREDUCE_MAX:
            return "AllReduceMax";
        case CollectiveOp::ALLREDUCE_MIN:
            return "AllReduceMin";
        case CollectiveOp::ALLGATHER:
            return "AllGather";
        case CollectiveOp::REDUCE_SCATTER:
            return "ReduceScatter";
        case CollectiveOp::BROADCAST:
            return "Broadcast";
        }
        return "Unknown";
    }

    /**
     * @brief Abstract interface for collective communication backends
     *
     * Each backend implements device-specific collective operations.
     * Backends are stateful (hold communicator handles, streams, etc.)
     *
     * Lifecycle:
     * 1. Create backend instance
     * 2. Call initialize() with device group
     * 3. Call collective operations as needed
     * 4. Call shutdown() when done (or destructor handles it)
     *
     * Thread Safety:
     * - Single backend instance should be used from one thread
     * - Multiple backends (one per stream) can run concurrently
     *
     * Buffer Registration:
     * - Backends that need to track buffer locations inherit from IBufferRegistration
     * - Default implementations are provided that return success/empty for backends
     *   that don't require registration (like MPI, NCCL, RCCL)
     * - PCIeBARBackend overrides these to track BAR offsets for cross-vendor transfers
     */
    class ICollectiveBackend : public IBufferRegistration
    {
    public:
        virtual ~ICollectiveBackend() = default;

        // =====================================================================
        // Identity
        // =====================================================================

        /// Get backend type identifier
        virtual CollectiveBackendType type() const = 0;

        /// Get human-readable backend name
        virtual std::string name() const = 0;

        // =====================================================================
        // Capability Queries
        // =====================================================================

        /**
         * @brief Check if backend supports a device type
         * @param type Device type (CPU, CUDA, ROCm, etc.)
         * @return true if backend can operate on this device type
         */
        virtual bool supportsDevice(DeviceType type) const = 0;

        /**
         * @brief Check if backend supports direct transfer between devices
         *
         * Direct transfer means no host staging required.
         * Example: NCCL supports direct CUDA↔CUDA but not CUDA↔ROCm.
         *
         * @param src Source device
         * @param dst Destination device
         * @return true if direct transfer is possible
         */
        virtual bool supportsDirectTransfer(DeviceId src, DeviceId dst) const = 0;

        /**
         * @brief Check if backend is available (library compiled in)
         * @return true if backend can be used
         */
        virtual bool isAvailable() const = 0;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        /**
         * @brief Initialize backend for a device group
         *
         * Must be called before any collective operations.
         * Creates communicators, allocates resources, etc.
         *
         * @param group Device group that will participate in collectives
         * @return true on success
         */
        virtual bool initialize(const DeviceGroup &group) = 0;

        /**
         * @brief Check if backend is initialized
         */
        virtual bool isInitialized() const = 0;

        /**
         * @brief Shutdown backend, release resources
         */
        virtual void shutdown() = 0;

        /**
         * @brief Reserve temporary buffer capacity for collective operations
         *
         * Pre-allocates internal temp buffers to avoid allocation in the hot path.
         * The buffer will grow if needed but never shrink during operation.
         * Buffer is only freed during shutdown().
         *
         * Call this during initialization after model dimensions are known:
         * @code
         * size_t max_elements = max_seq_len * hidden_size;
         * size_t buffer_bytes = activationPrecisionBufferBytes(max_elements, precision);
         * backend->reserveTempBufferBytes(buffer_bytes * 1.1);  // 10% margin
         * @endcode
         *
         * @param bytes Minimum buffer capacity in bytes
         * @return true if reservation succeeded (or no-op for backends that don't need it)
         */
        virtual bool reserveTempBufferBytes(size_t bytes) { (void)bytes; return true; }

        // =====================================================================
        // Collective Operations
        // =====================================================================

        /**
         * @brief In-place AllReduce operation
         *
         * All devices contribute their buffer values, result is the reduction
         * (sum/max/min) placed back in each device's buffer.
         *
         * @param buffer Device buffer (in-place, input and output)
         * @param count Number of elements
         * @param dtype Data type of elements
         * @param op Reduction operation (SUM, MAX, MIN)
         * @return true on success
         */
        virtual bool allreduce(
            void *buffer,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op) = 0;

        /**
         * @brief AllGather operation
         *
         * Each device contributes send_count elements, receives all slices
         * concatenated into recv_buf (total size = send_count * group_size).
         *
         * @param send_buf Local slice to send
         * @param recv_buf Buffer for full gathered result
         * @param send_count Elements per device
         * @param dtype Data type
         * @return true on success
         */
        virtual bool allgather(
            const void *send_buf,
            void *recv_buf,
            size_t send_count,
            CollectiveDataType dtype) = 0;

        /**
         * @brief Variable-count AllGather operation
         *
         * Each rank may send a different amount of data. This is needed for
         * heterogeneous tensor parallelism where devices have different head counts.
         *
         * @param send_buf Local data to send
         * @param send_count Number of elements this rank sends
         * @param recv_buf Buffer to receive all data
         * @param recv_counts Array of counts per rank (size = world_size)
         * @param displacements Array of offsets in recv_buf per rank
         * @param dtype Data type
         * @return true on success
         */
        virtual bool allgatherv(
            const void *send_buf,
            size_t send_count,
            void *recv_buf,
            const std::vector<int> &recv_counts,
            const std::vector<int> &displacements,
            CollectiveDataType dtype) = 0;

        /**
         * @brief ReduceScatter operation
         *
         * Reduce across all devices, then scatter result slices.
         * Each device gets recv_count elements of the reduced result.
         *
         * @param send_buf Full buffer to reduce (size = recv_count * group_size)
         * @param recv_buf Local slice of reduced result
         * @param recv_count Elements per device in result
         * @param dtype Data type
         * @param op Reduction operation
         * @return true on success
         */
        virtual bool reduceScatter(
            const void *send_buf,
            void *recv_buf,
            size_t recv_count,
            CollectiveDataType dtype,
            CollectiveOp op) = 0;

        /**
         * @brief Broadcast from root device to all
         *
         * @param buffer Buffer (root sends, others receive)
         * @param count Number of elements
         * @param dtype Data type
         * @param root_rank Rank of broadcasting device (index in group)
         * @return true on success
         */
        virtual bool broadcast(
            void *buffer,
            size_t count,
            CollectiveDataType dtype,
            int root_rank) = 0;

        // =====================================================================
        // Multi-GPU Single-Process Collective Operations
        // =====================================================================
        // These methods are for single-process scenarios managing multiple GPUs.
        // They take arrays of buffers (one per GPU) and issue the collective
        // across all GPUs using ncclGroupStart/ncclGroupEnd.
        //
        // Backends that don't support multi-GPU single-process return false.
        // NCCL and RCCL backends support these when initialized with multiple GPUs.

        /**
         * @brief Check if multi-GPU single-process mode is active
         * @return true if initialized with multiple GPUs in single process
         */
        virtual bool isMultiGpuSingleProcess() const { return false; }

        /**
         * @brief Multi-GPU AllReduce (single process)
         *
         * Each buffer[i] is on GPU i. All buffers are reduced together,
         * result placed back in each buffer (in-place).
         *
         * @param buffers Array of device buffers (one per GPU)
         * @param count Elements per buffer
         * @param dtype Data type
         * @param op Reduction operation
         * @return true on success, false if not supported
         */
        virtual bool allreduceMulti(
            const std::vector<void *> &buffers,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op)
        {
            (void)buffers;
            (void)count;
            (void)dtype;
            (void)op;
            return false; // Not supported by default
        }

        /**
         * @brief Multi-GPU AllGather (single process)
         *
         * Each send_bufs[i] on GPU i contributes send_count elements.
         * Each recv_bufs[i] receives all data (size = send_count * num_gpus).
         *
         * @param send_bufs Array of send buffers (one per GPU)
         * @param recv_bufs Array of receive buffers (one per GPU)
         * @param send_count Elements per GPU to send
         * @param dtype Data type
         * @return true on success, false if not supported
         */
        virtual bool allgatherMulti(
            const std::vector<const void *> &send_bufs,
            const std::vector<void *> &recv_bufs,
            size_t send_count,
            CollectiveDataType dtype)
        {
            (void)send_bufs;
            (void)recv_bufs;
            (void)send_count;
            (void)dtype;
            return false; // Not supported by default
        }

        /**
         * @brief Multi-GPU Broadcast (single process)
         *
         * Root GPU's buffer is broadcast to all other GPUs' buffers.
         *
         * @param buffers Array of buffers (one per GPU)
         * @param count Elements to broadcast
         * @param dtype Data type
         * @param root GPU index (0 to num_gpus-1) that broadcasts
         * @return true on success, false if not supported
         */
        virtual bool broadcastMulti(
            const std::vector<void *> &buffers,
            size_t count,
            CollectiveDataType dtype,
            int root)
        {
            (void)buffers;
            (void)count;
            (void)dtype;
            (void)root;
            return false; // Not supported by default
        }

        // =====================================================================
        // Registered Buffer Operations
        // =====================================================================

        /**
         * @brief In-place AllReduce using registered buffers
         *
         * Uses pre-registered buffer locations instead of explicit pointers.
         * This is required for backends like PCIeBARBackend that need to know
         * buffer locations in advance (e.g., BAR offsets for cross-vendor P2P).
         *
         * If the backend doesn't require buffer registration, this falls back
         * to the regular allreduce() using the registered buffer pointer.
         *
         * @param collective_id Identifier matching previous registerBuffer() calls
         * @param count Number of elements
         * @param dtype Data type of elements
         * @param op Reduction operation (SUM, MAX, MIN)
         * @return true on success, false if collective_id not found or operation fails
         */
        virtual bool allreduceRegistered(
            const std::string &collective_id,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op)
        {
            // Default implementation: not supported for base backends
            // Backends that support registration (like PCIeBARBackend) override this
            (void)collective_id;
            (void)count;
            (void)dtype;
            (void)op;
            return false;
        }

        // =====================================================================
        // Synchronization
        // =====================================================================

        /**
         * @brief Wait for all pending operations to complete
         * @return true on success
         */
        virtual bool synchronize() = 0;

        // =====================================================================
        // Diagnostics
        // =====================================================================

        /**
         * @brief Get last error message (if any)
         */
        virtual std::string lastError() const { return ""; }

        // =====================================================================
        // IBufferRegistration Default Implementations
        // =====================================================================
        // Most backends (MPI, NCCL, RCCL) don't need buffer registration.
        // These defaults allow them to work without implementing the interface.
        // PCIeBARBackend overrides these to track BAR offsets.

        /**
         * @brief Register a buffer (default: always succeeds, no-op)
         */
        bool registerBuffer(const std::string & /*collective_id*/,
                            DeviceId /*device*/,
                            void * /*buffer*/,
                            size_t /*size*/) override
        {
            return true; // No-op success for backends that don't need registration
        }

        /**
         * @brief Unregister a buffer (default: no-op)
         */
        void unregisterBuffer(const std::string & /*collective_id*/,
                              DeviceId /*device*/) override
        {
            // No-op for backends that don't track buffers
        }

        /**
         * @brief Get buffer info (default: not found)
         */
        std::optional<RegisteredBuffer> getBuffer(const std::string & /*collective_id*/,
                                                  DeviceId /*device*/) const override
        {
            return std::nullopt; // No registration info available
        }

        /**
         * @brief Check if registration is required (default: false)
         */
        bool requiresBufferRegistration() const override
        {
            return false; // Most backends don't need registration
        }
    };

    /**
     * @brief Factory for creating collective backends
     */
    class CollectiveBackendFactory
    {
    public:
        /**
         * @brief Create a backend of the specified type
         * @param type Backend type
         * @return Backend instance, or nullptr if type not available
         */
        static std::unique_ptr<ICollectiveBackend> create(CollectiveBackendType type);

        /**
         * @brief Check if a backend type is available
         */
        static bool isAvailable(CollectiveBackendType type);

        /**
         * @brief Get list of available backend types
         */
        static std::vector<CollectiveBackendType> availableBackends();
    };

} // namespace llaminar2

// Hash specialization for CollectiveBackendType (for std::unordered_map)
namespace std
{
    template <>
    struct hash<llaminar2::CollectiveBackendType>
    {
        size_t operator()(const llaminar2::CollectiveBackendType &type) const noexcept
        {
            return std::hash<int>{}(static_cast<int>(type));
        }
    };
} // namespace std
