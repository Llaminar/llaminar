/**
 * @file MPIContext.h
 * @brief MPI coordination abstraction for distributed inference
 * 
 * Separates MPI coordination from device execution, enabling:
 * - Distributed inference without GPU (OpenBLAS + MPI)
 * - Single-GPU inference without MPI
 * - Clean separation of concerns
 * 
 * @author David Sanftenberg
 */

#pragma once

#include <mpi.h>
#include <memory>
#include <utility>
#include <cstddef>

namespace llaminar2 {

/**
 * @brief MPI context for distributed coordination
 * 
 * Encapsulates MPI state and provides clean API for collective operations.
 * Thread-safe for MPI_THREAD_MULTIPLE environments.
 */
class MPIContext {
public:
    /**
     * @brief Construct MPI context
     * 
     * @param rank MPI rank (0-indexed)
     * @param world_size Total number of ranks
     * @param comm MPI communicator (default: MPI_COMM_WORLD)
     */
    MPIContext(int rank, int world_size, MPI_Comm comm = MPI_COMM_WORLD)
        : rank_(rank), world_size_(world_size), comm_(comm) {}

    // Accessors
    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    MPI_Comm comm() const { return comm_; }

    /**
     * @brief All-reduce sum operation
     * 
     * @param send_data Input data (local contribution)
     * @param recv_data Output data (global sum)
     * @param count Number of elements
     */
    void allreduce_sum(const float* send_data, float* recv_data, size_t count) const {
        MPI_Allreduce(send_data, recv_data, count, MPI_FLOAT, MPI_SUM, comm_);
    }

    /**
     * @brief Broadcast data from root to all ranks
     * 
     * @param data Data buffer (input on root, output on others)
     * @param count Number of elements
     * @param root Root rank (default: 0)
     */
    void broadcast(float* data, size_t count, int root = 0) const {
        MPI_Bcast(data, count, MPI_FLOAT, root, comm_);
    }

    /**
     * @brief All-gather operation
     * 
     * @param send_data Local data
     * @param recv_data Global data (concatenated from all ranks)
     * @param count Number of elements per rank
     */
    void allgather(const float* send_data, float* recv_data, size_t count) const {
        MPI_Allgather(send_data, count, MPI_FLOAT, recv_data, count, MPI_FLOAT, comm_);
    }

    /**
     * @brief Synchronization barrier
     * 
     * Blocks until all ranks reach this point.
     */
    void barrier() const {
        MPI_Barrier(comm_);
    }

    /**
     * @brief Get local slice for distributed work
     * 
     * @param total_elements Total number of elements to distribute
     * @return {start_index, count} for this rank
     */
    std::pair<size_t, size_t> get_local_slice(size_t total_elements) const {
        size_t base_count = total_elements / world_size_;
        size_t remainder = total_elements % world_size_;
        
        size_t start = rank_ * base_count + std::min(static_cast<size_t>(rank_), remainder);
        size_t count = base_count + (rank_ < remainder ? 1 : 0);
        
        return {start, count};
    }

    /**
     * @brief Distribute rows across ranks (for matrix sharding)
     * 
     * @param total_rows Total number of rows
     * @return {start_row, num_rows} for this rank
     */
    std::pair<size_t, size_t> distribute_rows(size_t total_rows) const {
        return get_local_slice(total_rows);
    }

private:
    int rank_;
    int world_size_;
    MPI_Comm comm_;
};

/**
 * @brief Factory for creating MPI contexts
 */
class MPIContextFactory {
public:
    /**
     * @brief Get global MPI context (MPI_COMM_WORLD)
     * 
     * @return Shared pointer to global context (singleton)
     */
    static std::shared_ptr<MPIContext> global() {
        static std::shared_ptr<MPIContext> instance;
        if (!instance) {
            int rank, world_size;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            instance = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);
        }
        return instance;
    }

    /**
     * @brief Create mock MPI context for testing
     * 
     * @param rank Simulated rank
     * @param world_size Simulated world size
     * @return Mock context (no actual MPI calls)
     */
    static std::shared_ptr<MPIContext> create_mock(int rank, int world_size) {
        return std::make_shared<MPIContext>(rank, world_size, MPI_COMM_SELF);
    }
};

} // namespace llaminar2
