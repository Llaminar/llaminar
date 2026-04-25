/**
 * @file UPIBackend.h
 * @brief UPI-based collective backend for cross-socket CPU tensor parallelism
 *
 * The UPICollectiveBackend provides MPI-based collective operations for
 * cross-socket CPU tensor parallelism using a domain-specific MPI communicator.
 *
 * Unlike the general-purpose MPIBackend which uses MPI_COMM_WORLD, this backend:
 * - Uses a domain-specific communicator (created via MPI_Comm_split)
 * - Is optimized for UPI/QPI/Infinity Fabric interconnect (~50 GB/s)
 * - Expects NUMA-local buffers (no cross-socket memory access)
 *
 * Use Case:
 * - Cross-socket CPU tensor parallelism in hybrid CPU+GPU setups
 * - CPUs handling their proportional share of model weights
 * - Allreduce across CPU-only TP domains
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../ICollectiveBackend.h"
#include <mpi.h>
#include <string>

namespace llaminar2
{

    // Forward declaration
    class NodeTopology;

    /**
     * @brief Collective backend for cross-socket CPU tensor parallelism using MPI over UPI
     *
     * This backend:
     * - Uses a domain-specific MPI communicator (created via MPI_Comm_split)
     * - Expects NUMA-local buffers (no cross-socket memory access)
     * - Communicates over UPI/QPI/Infinity Fabric interconnect (~50 GB/s)
     *
     * Usage:
     * - Create domain communicator with TPDomainBuilder::createCPUCrossRankDomain()
     * - Pass communicator to UPICollectiveBackend
     * - Call allreduce() for cross-socket CPU tensor parallelism
     *
     * Thread Safety:
     * - Single backend instance should be used from one thread
     * - Multiple backends (different domains) can run concurrently
     */
    class UPICollectiveBackend : public ICollectiveBackend
    {
    public:
        /**
         * @brief Create UPI backend with domain communicator
         * @param domain_comm MPI communicator for this TP domain (not MPI_COMM_WORLD)
         * @param topology System topology for optimization hints (may be nullptr)
         */
        explicit UPICollectiveBackend(MPI_Comm domain_comm,
                                      const NodeTopology *topology = nullptr);

        ~UPICollectiveBackend() override;

        // =========================================================================
        // Identity (ICollectiveBackend)
        // =========================================================================

        /**
         * @brief Get backend type
         * @return CollectiveBackendType::MPI (UPI uses MPI internally)
         */
        CollectiveBackendType type() const override { return CollectiveBackendType::MPI; }

        /**
         * @brief Get human-readable backend name
         * @return "UPI"
         */
        std::string name() const override { return "UPI"; }

        // =========================================================================
        // Capability Queries (ICollectiveBackend)
        // =========================================================================

        /**
         * @brief Check if backend supports a device type
         *
         * UPI operates on host memory only - CPU is supported directly.
         * GPU buffers would require staging (not handled by this backend).
         *
         * @param type Device type (CPU, CUDA, ROCm, etc.)
         * @return true only for CPU
         */
        bool supportsDevice(DeviceType type) const override;

        /**
         * @brief Check if backend supports direct transfer between devices
         *
         * UPI can only directly transfer between CPU buffers.
         * Both buffers should be NUMA-local for optimal performance.
         *
         * @param src Source device
         * @param dst Destination device
         * @return true only if both devices are CPU
         */
        bool supportsDirectTransfer(DeviceId src, DeviceId dst) const override;

        /**
         * @brief Check if backend is available (has valid communicator)
         * @return true if domain communicator is valid
         */
        bool isAvailable() const override;

        // =========================================================================
        // Lifecycle (ICollectiveBackend)
        // =========================================================================

        /**
         * @brief Initialize backend for a device group
         *
         * The UPI backend is pre-initialized with a domain communicator,
         * so this validates the group and marks as initialized.
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
         *
         * Note: Does NOT free the domain communicator (caller owns it)
         */
        void shutdown() override;

        // =========================================================================
        // Collective Operations (ICollectiveBackend)
        // =========================================================================

        /**
         * @brief In-place AllReduce operation via MPI_Allreduce
         *
         * Uses MPI_IN_PLACE for efficient in-place reduction over UPI.
         * Buffer should be NUMA-local for optimal performance.
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
         * concatenated into recv_buf (total size = send_count * domain_size).
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
         * Each rank may send a different amount of data. This is needed for
         * heterogeneous tensor parallelism where devices have different head counts.
         *
         * @param send_buf Local data to send
         * @param send_count Number of elements this rank sends
         * @param recv_buf Buffer to receive all data
         * @param recv_counts Array of counts per rank (size = domain_size)
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
         * @param send_buf Full buffer to reduce (size = recv_count * domain_size)
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
         * @param root_rank Rank of broadcasting process (within domain)
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

        // =========================================================================
        // Diagnostics (ICollectiveBackend)
        // =========================================================================

        /**
         * @brief Get last error message (if any)
         * @return Error message string
         */
        std::string lastError() const override { return last_error_; }

        // =========================================================================
        // UPI-Specific Methods
        // =========================================================================

        /**
         * @brief Get domain communicator rank
         * @return This process's rank within the domain (0 to domain_size-1)
         */
        int domainRank() const { return domain_rank_; }

        /**
         * @brief Get domain communicator size
         * @return Number of processes in this domain
         */
        int domainSize() const { return domain_size_; }

        /**
         * @brief Get estimated UPI bandwidth in GB/s
         *
         * Returns heuristic estimate based on CPU vendor:
         * - Intel UPI: ~50 GB/s per link (typical dual-socket)
         * - AMD Infinity Fabric: ~100 GB/s (EPYC dual-socket)
         *
         * @return Estimated bandwidth in GB/s
         */
        float estimatedBandwidthGBps() const { return bandwidth_gbps_; }

        /**
         * @brief Check if this backend is valid (has valid communicator)
         * @return true if domain_comm is not MPI_COMM_NULL
         */
        bool isValid() const { return domain_comm_ != MPI_COMM_NULL; }

        /**
         * @brief Get the domain MPI communicator
         * @return The domain communicator (may be MPI_COMM_NULL if invalid)
         */
        MPI_Comm domainComm() const { return domain_comm_; }

        // =========================================================================
        // Static Helpers (for testing and utilities)
        // =========================================================================

        /**
         * @brief Convert CollectiveDataType to MPI_Datatype
         * @param dtype Llaminar collective data type
         * @return Corresponding MPI_Datatype
         */
        static MPI_Datatype toMPIDatatype(CollectiveDataType dtype);

        /**
         * @brief Convert CollectiveOp to MPI_Op
         * @param op Llaminar collective operation
         * @return Corresponding MPI_Op
         */
        static MPI_Op toMPIOp(CollectiveOp op);

    private:
        MPI_Comm domain_comm_;           ///< Domain-specific MPI communicator
        int domain_rank_;                ///< Rank within domain
        int domain_size_;                ///< Size of domain
        float bandwidth_gbps_;           ///< Estimated interconnect bandwidth
        DeviceGroup group_;              ///< Device group for this domain
        bool initialized_ = false;       ///< Whether initialize() was called
        mutable std::string last_error_; ///< Last error message

        /**
         * @brief Half-precision allreduce via allgather + local FP32 reduce
         *
         * MPI has no native FP16/BF16 reduction support, so we allgather
         * the raw half-precision data and reduce locally in FP32.
         */
        bool allreduceHalfPrecision(void *buffer, size_t count,
                                    CollectiveDataType dtype, CollectiveOp op);

        /**
         * @brief Estimate UPI/IF bandwidth based on topology
         * @param topology System topology (may be nullptr)
         * @return Estimated bandwidth in GB/s
         */
        static float estimateBandwidth(const NodeTopology *topology);
    };

} // namespace llaminar2
