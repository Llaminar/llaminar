/**
 * @file ICollectiveContext.h
 * @brief Interface for collective context operations
 *
 * Abstracts collective communication context to enable:
 * 1. Unit testing without MPI/NCCL runtime
 * 2. Mock collective operations for isolated stage testing
 * 3. Deterministic testing of distributed algorithms
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../collective/ICollectiveBackend.h"
#include "../backends/DeviceId.h"
#include <memory>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class ITensor;
    class IBackendRouter;
    class MPIContext;
    struct TPDomain;

    /**
     * @brief Abstract interface for collective context operations
     *
     * This interface abstracts the CollectiveContext to enable:
     * - Unit testing of pipeline stages without real MPI/NCCL runtime
     * - Mock collective operations for isolated testing
     * - Injection of test doubles for failure injection and verification
     *
     * Implementations:
     * - CollectiveContext: Real backend-routing implementation
     * - MockCollectiveContext: Test implementation with configurable behavior
     */
    class ICollectiveContext
    {
    public:
        virtual ~ICollectiveContext() = default;

        // =========================================================================
        // High-Level Collective Operations
        // These are called by DeviceGraphExecutor when executing collective stages
        // =========================================================================

        /**
         * @brief Execute an AllReduce operation
         *
         * Automatically selects the best backend based on tensor location.
         *
         * @param buffer In-place buffer to reduce
         * @param count Number of elements (0 = use buffer->numel())
         * @param tensor_device Device where tensor resides
         * @param op Reduction operation (default: SUM)
         * @return true on success
         */
        virtual bool executeAllreduce(
            ITensor *buffer,
            size_t count,
            DeviceId tensor_device,
            CollectiveOp op = CollectiveOp::ALLREDUCE_SUM) = 0;

        /**
         * @brief Execute an AllGather operation
         *
         * @param local_input Local slice [seq_len, local_dim]
         * @param full_output Full output [seq_len, full_dim]
         * @param actual_seq_len Actual sequence length (0 = use tensor rows)
         * @param tensor_device Device where tensors reside
         * @return true on success
         */
        virtual bool executeAllgather(
            ITensor *local_input,
            ITensor *full_output,
            size_t actual_seq_len,
            DeviceId tensor_device) = 0;

        /**
         * @brief Execute a GPU-native strided AllGather operation
         *
         * This is an optimized path for column-parallel GEMM outputs where:
         * - Standard allgather would collect into contiguous layout
         * - But output needs strided interleaved layout for next operation
         *
         * Uses NCCL AllGather to temp buffer, then CUDA kernel to deinterleave.
         * Falls back to false if NCCL not available or not on CUDA device.
         *
         * @param local_input Local slice [seq_len, local_dim]
         * @param full_output Full output [seq_len, full_dim] (strided)
         * @param actual_seq_len Actual sequence length (0 = use tensor rows)
         * @param tensor_device Device where tensors reside (must be CUDA)
         * @return true if successfully executed via GPU-native path, false to fallback
         */
        virtual bool executeStridedAllgather(
            ITensor *local_input,
            ITensor *full_output,
            size_t actual_seq_len,
            DeviceId tensor_device) = 0;

        /**
         * @brief Execute a variable-sized AllGather operation (allgatherv)
         *
         * Unlike executeAllgather which assumes equal send counts per rank,
         * this method supports variable send counts needed for heterogeneous
         * tensor parallelism (e.g., different head counts per device).
         *
         * @param local_input Local slice to send [seq_len, local_dim]
         * @param full_output Full output [seq_len, sum(recv_counts)]
         * @param recv_counts Elements per rank (size = world_size)
         * @param displacements Offset in output per rank (size = world_size)
         * @param actual_seq_len Actual sequence length (0 = use tensor rows)
         * @param tensor_device Device where tensors reside
         * @return true on success
         */
        virtual bool executeAllgatherv(
            ITensor *local_input,
            ITensor *full_output,
            const std::vector<int> &recv_counts,
            const std::vector<int> &displacements,
            size_t actual_seq_len,
            DeviceId tensor_device) = 0;

        /**
         * @brief Execute a Broadcast operation
         *
         * @param buffer Buffer to broadcast (in-place)
         * @param count Number of elements
         * @param root_rank Rank that holds the source data
         * @param tensor_device Device where buffer resides
         * @return true on success
         */
        virtual bool executeBroadcast(
            ITensor *buffer,
            size_t count,
            int root_rank,
            DeviceId tensor_device) = 0;

        // =========================================================================
        // Query Methods
        // =========================================================================

        /**
         * @brief Does this configuration require collectives?
         * @return true if world_size > 1
         */
        virtual bool requiresCollectives() const = 0;

        /**
         * @brief Get the world size (number of ranks)
         * @return World size
         */
        virtual int worldSize() const = 0;

        /**
         * @brief Get the local rank
         * @return Rank (0-indexed)
         */
        virtual int rank() const = 0;

        /**
         * @brief Get devices on this rank
         * @return Vector of device IDs
         */
        virtual const std::vector<DeviceId> &localDevices() const = 0;

        /**
         * @brief Check if a specific backend is available
         * @param type Backend type to check
         * @return true if available
         */
        virtual bool isBackendAvailable(CollectiveBackendType type) const = 0;

        // =========================================================================
        // Advanced: Direct Access (for testing/debugging)
        // =========================================================================

        /**
         * @brief Get the backend router interface (internal use)
         * @return Pointer to the router (may be nullptr for mocks)
         */
        virtual IBackendRouter *router() = 0;

        /**
         * @brief Get the MPI context (may be nullptr)
         * @return Pointer to the MPI context
         */
        virtual MPIContext *mpiContext() = 0;

        // =========================================================================
        // Domain-Aware Collective Operations
        // These methods route operations through specific TP domains for
        // heterogeneous parallelism (e.g., GPU domain vs CPU cross-rank domain).
        // =========================================================================

        /**
         * @brief Execute an AllReduce operation in a specific TP domain
         *
         * Domain-aware routing:
         * - GPU_INTRA_RANK domains → PCIeBAR/NCCL/RCCL backends
         * - CPU_CROSS_RANK domains → UPI/MPI backends
         *
         * @param buffer In-place buffer to reduce
         * @param count Number of elements (0 = use buffer->numel())
         * @param tensor_device Device where tensor resides
         * @param op Reduction operation
         * @param domain TP domain to use (nullptr = fallback to non-domain method)
         * @return true on success
         */
        virtual bool executeAllreduceInDomain(
            ITensor *buffer,
            size_t count,
            DeviceId tensor_device,
            CollectiveOp op,
            const TPDomain *domain) = 0;

        /**
         * @brief Execute an AllGather operation in a specific TP domain
         *
         * @param local_input Local slice [seq_len, local_dim]
         * @param full_output Full output [seq_len, full_dim]
         * @param actual_seq_len Actual sequence length (0 = use tensor rows)
         * @param tensor_device Device where tensors reside
         * @param domain TP domain to use (nullptr = fallback to non-domain method)
         * @return true on success
         */
        virtual bool executeAllgatherInDomain(
            ITensor *local_input,
            ITensor *full_output,
            size_t actual_seq_len,
            DeviceId tensor_device,
            const TPDomain *domain) = 0;

        /**
         * @brief Execute a variable-sized AllGather operation in a specific TP domain
         *
         * @param local_input Local slice to send [seq_len, local_dim]
         * @param full_output Full output [seq_len, sum(recv_counts)]
         * @param recv_counts Elements per rank (size = world_size)
         * @param displacements Offset in output per rank (size = world_size)
         * @param actual_seq_len Actual sequence length (0 = use tensor rows)
         * @param tensor_device Device where tensors reside
         * @param domain TP domain to use (nullptr = fallback to non-domain method)
         * @return true on success
         */
        virtual bool executeAllgathervInDomain(
            ITensor *local_input,
            ITensor *full_output,
            const std::vector<int> &recv_counts,
            const std::vector<int> &displacements,
            size_t actual_seq_len,
            DeviceId tensor_device,
            const TPDomain *domain) = 0;
    };

} // namespace llaminar2
