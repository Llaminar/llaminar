/**
 * @file IMPIContext.h
 * @brief Interface for MPI context operations
 *
 * Abstracts MPI communication to enable:
 * 1. Unit testing without MPI runtime
 * 2. Multi-rank simulation in single process
 * 3. Deterministic testing of distributed algorithms
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include <mpi.h>
#include <cstddef>
#include <memory>
#include <utility>
#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declarations for quantized block types
    struct Q8_1Block;
    struct Q16_1Block;
    struct Q16_1Block_64;
    struct Q16_1Block_128;

    /**
     * @brief Abstract interface for MPI context operations
     *
     * This interface abstracts MPI communication primitives to enable:
     * - Unit testing of distributed logic without real MPI runtime
     * - Multi-rank simulation in a single process for deterministic testing
     * - Mocking of collective operations for failure injection
     *
     * Implementations:
     * - MPIContext: Real MPI-backed implementation
     * - MockMPIContext: Test implementation with configurable behavior
     */
    class IMPIContext
    {
    public:
        virtual ~IMPIContext() = default;

        // =========================================================================
        // Identity
        // =========================================================================

        /**
         * @brief Get the MPI rank of this process
         * @return Rank (0-indexed)
         */
        virtual int rank() const = 0;

        /**
         * @brief Get the total number of MPI processes
         * @return World size
         */
        virtual int world_size() const = 0;

        /**
         * @brief Check if this is the root rank (rank 0)
         * @return true if rank == 0
         */
        virtual bool is_root() const = 0;

        // =========================================================================
        // Synchronization
        // =========================================================================

        /**
         * @brief Synchronization barrier - blocks until all ranks reach this point
         */
        virtual void barrier() const = 0;

        // =========================================================================
        // All-Reduce Operations (FP32)
        // =========================================================================

        /**
         * @brief All-reduce sum operation (out-of-place)
         * @param send_data Input data (local contribution)
         * @param recv_data Output data (global sum)
         * @param count Number of elements
         */
        virtual void allreduce_sum(const float *send_data, float *recv_data, size_t count) const = 0;

        /**
         * @brief All-reduce sum operation (in-place)
         * @param data Data buffer (input and output)
         * @param count Number of elements
         */
        virtual void allreduce_sum_inplace(float *data, size_t count) const = 0;

        // =========================================================================
        // All-Reduce Operations (Quantized)
        // =========================================================================

        /**
         * @brief All-reduce sum for Q8_1 blocks (in-place)
         *
         * Performs N-way reduction of Q8_1 blocks across all ranks without
         * FP32 conversion overhead. Uses allgather + vectorized local reduction.
         *
         * @param data Q8_1 block buffer (input and output)
         * @param n_blocks Number of Q8_1 blocks per rank
         */
        virtual void allreduce_q8_1_inplace(Q8_1Block *data, size_t n_blocks) const = 0;

        /**
         * @brief All-reduce sum for Q16_1 blocks (in-place)
         *
         * @param data Q16_1 block buffer (input and output)
         * @param n_blocks Number of Q16_1 blocks per rank
         */
        virtual void allreduce_q16_1_inplace(Q16_1Block *data, size_t n_blocks) const = 0;

        // =========================================================================
        // All-Reduce Operations (FP16/BF16)
        // =========================================================================

        /**
         * @brief All-reduce sum for FP16 elements (in-place)
         * @param data FP16 buffer (uint16_t, input and output)
         * @param count Number of FP16 elements per rank
         */
        virtual void allreduce_fp16_inplace(uint16_t *data, size_t count) const = 0;

        /**
         * @brief All-reduce sum for BF16 elements (in-place)
         * @param data BF16 buffer (uint16_t, input and output)
         * @param count Number of BF16 elements per rank
         */
        virtual void allreduce_bf16_inplace(uint16_t *data, size_t count) const = 0;

        // =========================================================================
        // Broadcast
        // =========================================================================

        /**
         * @brief Broadcast data from root to all ranks
         * @param data Data buffer (input on root, output on others)
         * @param count Number of elements
         * @param root Root rank (default: 0)
         */
        virtual void broadcast(float *data, size_t count, int root = 0) const = 0;

        // =========================================================================
        // All-Gather Operations
        // =========================================================================

        /**
         * @brief All-gather operation for floats
         * @param send_data Local data
         * @param recv_data Global data (concatenated from all ranks)
         * @param count Number of elements per rank
         */
        virtual void allgather(const float *send_data, float *recv_data, size_t count) const = 0;

        /**
         * @brief All-gather bytes operation (type-agnostic)
         * @param send_data Local data (raw bytes)
         * @param recv_data Global data (concatenated from all ranks)
         * @param byte_count Number of bytes per rank
         */
        virtual void allgather_bytes(const void *send_data, void *recv_data, size_t byte_count) const = 0;

        /**
         * @brief All-gather with variable counts per rank (bytes)
         * @param send_data Local data (raw bytes)
         * @param send_count Number of bytes this rank is sending
         * @param recv_data Global data buffer (must be pre-sized)
         * @param recv_counts Array of counts from each rank
         * @param displs Array of displacements in recv_data for each rank
         */
        virtual void allgatherv_bytes(const void *send_data, int send_count,
                                      void *recv_data, const int *recv_counts,
                                      const int *displs) const = 0;

        // =========================================================================
        // Work Distribution
        // =========================================================================

        /**
         * @brief Get local slice for distributed work
         * @param total_elements Total number of elements to distribute
         * @return {start_index, count} for this rank
         */
        virtual std::pair<size_t, size_t> get_local_slice(size_t total_elements) const = 0;

        /**
         * @brief Distribute rows across ranks (for matrix sharding)
         * @param total_rows Total number of rows
         * @return {start_row, num_rows} for this rank
         */
        virtual std::pair<size_t, size_t> distribute_rows(size_t total_rows) const = 0;

        // =========================================================================
        // Point-to-Point Operations (Pipeline Parallelism)
        // =========================================================================

        /**
         * @brief Synchronous send operation
         * @param data Pointer to data to send
         * @param count Number of elements
         * @param type MPI datatype
         * @param dest Destination rank
         * @param tag Message tag
         */
        virtual void send(const void *data, size_t count, MPI_Datatype type, int dest, int tag) const = 0;

        /**
         * @brief Synchronous receive operation
         * @param data Pointer to receive buffer
         * @param count Maximum number of elements to receive
         * @param type MPI datatype
         * @param src Source rank (or MPI_ANY_SOURCE)
         * @param tag Message tag (or MPI_ANY_TAG)
         * @param status Optional status object
         */
        virtual void recv(void *data, size_t count, MPI_Datatype type, int src, int tag,
                          MPI_Status *status = nullptr) const = 0;

        /**
         * @brief Non-blocking send operation
         * @param data Pointer to data to send
         * @param count Number of elements
         * @param type MPI datatype
         * @param dest Destination rank
         * @param tag Message tag
         * @return MPI_Request handle
         */
        virtual MPI_Request isend(const void *data, size_t count, MPI_Datatype type,
                                  int dest, int tag) const = 0;

        /**
         * @brief Non-blocking receive operation
         * @param data Pointer to receive buffer
         * @param count Maximum number of elements to receive
         * @param type MPI datatype
         * @param src Source rank (or MPI_ANY_SOURCE)
         * @param tag Message tag (or MPI_ANY_TAG)
         * @return MPI_Request handle
         */
        virtual MPI_Request irecv(void *data, size_t count, MPI_Datatype type,
                                  int src, int tag) const = 0;

        /**
         * @brief Wait for a single non-blocking operation to complete
         * @param request Pointer to MPI_Request handle
         * @param status Optional status object
         */
        virtual void wait(MPI_Request *request, MPI_Status *status = nullptr) const = 0;

        /**
         * @brief Wait for all non-blocking operations to complete
         * @param requests Vector of MPI_Request handles
         */
        virtual void waitAll(std::vector<MPI_Request> &requests) const = 0;

        /**
         * @brief Blocking probe for incoming message
         * @param src Source rank (or MPI_ANY_SOURCE)
         * @param tag Message tag (or MPI_ANY_TAG)
         * @param status Status object to receive message info
         */
        virtual void probe(int src, int tag, MPI_Status *status) const = 0;

        /**
         * @brief Non-blocking probe for incoming message
         * @param src Source rank (or MPI_ANY_SOURCE)
         * @param tag Message tag (or MPI_ANY_TAG)
         * @param status Status object to receive message info
         * @return true if a matching message is available
         */
        virtual bool iprobe(int src, int tag, MPI_Status *status) const = 0;

        // =========================================================================
        // Typed Point-to-Point Convenience Methods
        // =========================================================================

        /**
         * @brief Send float data
         */
        virtual void sendFloat(const float *data, size_t count, int dest, int tag) const = 0;

        /**
         * @brief Receive float data
         */
        virtual void recvFloat(float *data, size_t count, int src, int tag,
                               MPI_Status *status = nullptr) const = 0;

        /**
         * @brief Send raw bytes
         */
        virtual void sendBytes(const void *data, size_t byte_count, int dest, int tag) const = 0;

        /**
         * @brief Receive raw bytes
         */
        virtual void recvBytes(void *data, size_t byte_count, int src, int tag,
                               MPI_Status *status = nullptr) const = 0;

        /**
         * @brief Get message count from status
         * @param status MPI_Status from recv or probe
         * @param type MPI datatype
         * @return Number of elements in the message
         */
        virtual int getCount(const MPI_Status &status, MPI_Datatype type) const = 0;

        // =========================================================================
        // Factory Methods
        // =========================================================================

        /**
         * @brief Create a mock MPI context for testing
         * @param rank Simulated rank
         * @param world_size Simulated world size
         * @return Mock context that implements IMPIContext
         */
        static std::shared_ptr<IMPIContext> createMock(int rank, int world_size);
    };

} // namespace llaminar2
