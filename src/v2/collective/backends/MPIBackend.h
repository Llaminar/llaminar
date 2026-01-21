/**
 * @file MPIBackend.h
 * @brief MPI-based collective backend for inter-node communication
 *
 * The MPIBackend provides MPI-based collective operations for distributed
 * inference across multiple nodes. It wraps MPI collective primitives
 * (MPI_Allreduce, MPI_Allgather, MPI_Bcast, etc.) to implement the
 * ICollectiveBackend interface.
 *
 * This backend is used for:
 * - Inter-node tensor parallelism across MPI ranks
 * - CPU-only collective operations
 * - Global-scope collectives in hybrid setups (MPI + NCCL/RCCL)
 *
 * Note: MPI operates on host memory. For GPU buffers, data must first
 * be staged to host memory (handled by higher-level orchestration).
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../ICollectiveBackend.h"
#include "../DeviceGroup.h"
#include "../../utils/MPIContext.h"
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @brief MPI-based collective backend for inter-node communication
     *
     * Implements collective operations via MPI. Requires an active MPIContext
     * for actual MPI operations. When mpi_ctx is nullptr, operations fail
     * gracefully (useful for unit testing capability queries).
     *
     * Thread Safety: MPI operations should be called from a single thread
     * unless MPI was initialized with MPI_THREAD_MULTIPLE.
     */
    class MPIBackend : public ICollectiveBackend
    {
    public:
        /**
         * @brief Construct MPI backend
         * @param mpi_ctx MPI context (may be nullptr for testing)
         */
        explicit MPIBackend(std::shared_ptr<MPIContext> mpi_ctx);

        ~MPIBackend() override;

        // =====================================================================
        // Identity
        // =====================================================================

        CollectiveBackendType type() const override { return CollectiveBackendType::MPI; }
        std::string name() const override { return "MPI"; }

        // =====================================================================
        // Capability Queries
        // =====================================================================

        /**
         * @brief Check if backend supports a device type
         *
         * MPI operates on host memory only. CPU is supported directly;
         * GPU buffers require host staging (not handled by this backend).
         *
         * @param type Device type (CPU, CUDA, ROCm, etc.)
         * @return true only for CPU
         */
        bool supportsDevice(DeviceType type) const override;

        /**
         * @brief Check if backend supports direct transfer between devices
         *
         * MPI can only directly transfer between CPU buffers.
         * GPU↔GPU or GPU↔CPU transfers require explicit staging.
         *
         * @param src Source device
         * @param dst Destination device
         * @return true only if both devices are CPU
         */
        bool supportsDirectTransfer(DeviceId src, DeviceId dst) const override;

        /**
         * @brief Check if backend is available
         *
         * MPI backend requires a valid MPIContext with world_size > 0.
         *
         * @return true if MPI context is available and valid
         */
        bool isAvailable() const override;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        /**
         * @brief Initialize backend for a device group
         *
         * Validates the group configuration and prepares for collective ops.
         * Requires a valid MPIContext.
         *
         * @param group Device group that will participate in collectives
         * @return true on success, false if no MPI context
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
         * @brief In-place AllReduce operation via MPI_Allreduce
         *
         * Uses MPI_IN_PLACE for efficient in-place reduction.
         *
         * @param buffer Host buffer (in-place, input and output)
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
         * @brief AllGather operation via MPI_Allgather
         *
         * Each rank contributes send_count elements, receives all slices
         * concatenated into recv_buf (total size = send_count * world_size).
         *
         * @param send_buf Local slice to send
         * @param recv_buf Buffer for full gathered result
         * @param send_count Elements per rank
         * @param dtype Data type
         * @return true on success
         */
        bool allgather(
            const void *send_buf,
            void *recv_buf,
            size_t send_count,
            CollectiveDataType dtype) override;

        /**
         * @brief Variable-count AllGather operation via MPI_Allgatherv
         *
         * Each rank may send a different amount of data.
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
         * @brief ReduceScatter operation via MPI_Reduce_scatter
         *
         * Reduce across all ranks, then scatter result slices.
         * Each rank gets recv_count elements of the reduced result.
         *
         * @param send_buf Full buffer to reduce (size = recv_count * world_size)
         * @param recv_buf Local slice of reduced result
         * @param recv_count Elements per rank in result
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
         * @brief Broadcast from root rank to all via MPI_Bcast
         *
         * @param buffer Buffer (root sends, others receive)
         * @param count Number of elements
         * @param dtype Data type
         * @param root_rank Rank of broadcasting process
         * @return true on success
         */
        bool broadcast(
            void *buffer,
            size_t count,
            CollectiveDataType dtype,
            int root_rank) override;

        /**
         * @brief Synchronize via MPI_Barrier
         * @return true on success
         */
        bool synchronize() override;

        // =====================================================================
        // Diagnostics
        // =====================================================================

        std::string lastError() const override { return last_error_; }

    private:
        std::shared_ptr<MPIContext> mpi_ctx_;
        DeviceGroup group_;
        bool initialized_ = false;
        mutable std::string last_error_;

        /**
         * @brief Convert CollectiveDataType to MPI_Datatype
         */
        MPI_Datatype toMPIDatatype(CollectiveDataType dtype) const;

        /**
         * @brief Convert CollectiveOp to MPI_Op
         */
        MPI_Op toMPIOp(CollectiveOp op) const;
    };

} // namespace llaminar2
