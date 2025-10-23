#pragma once

#include <mpi.h>
#include <memory>
#include <vector>

namespace llaminar
{

/**
 * @brief MPI execution context for distributed inference
 * 
 * Encapsulates MPI communicator, rank information, and collective operations.
 * Passed to all kernels that need MPI coordination (Allreduce, broadcast, etc.)
 * 
 * Design goals:
 * - Single source of truth for MPI state
 * - Easy testing (can mock for single-rank tests)
 * - Efficient collective operations with minimal boilerplate
 */
class MPIContext
{
public:
    /**
     * @brief Create MPI context from global communicator
     * @param comm MPI communicator (default: MPI_COMM_WORLD)
     */
    explicit MPIContext(MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * @brief Create MPI context with explicit rank/size (for testing)
     * @param rank Process rank
     * @param world_size Total number of processes
     * @param comm MPI communicator
     */
    MPIContext(int rank, int world_size, MPI_Comm comm = MPI_COMM_WORLD);

    ~MPIContext() = default;

    // Getters
    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    bool is_distributed() const { return world_size_ > 1; }
    bool is_root() const { return rank_ == 0; }
    MPI_Comm comm() const { return comm_; }

    // Collective operations (convenience wrappers)
    
    /**
     * @brief Allreduce with SUM operation
     * @param send_data Source buffer
     * @param recv_data Destination buffer
     * @param count Number of elements
     */
    void allreduce_sum(const float* send_data, float* recv_data, size_t count) const;
    
    /**
     * @brief Broadcast from root rank
     * @param data Buffer to broadcast
     * @param count Number of elements
     * @param root_rank Source rank (default: 0)
     */
    void broadcast(float* data, size_t count, int root_rank = 0) const;
    
    /**
     * @brief Allgather: gather data from all ranks to all ranks
     * @param send_data Source buffer (local data)
     * @param recv_data Destination buffer (all data)
     * @param send_count Elements per rank to send
     */
    void allgather(const float* send_data, float* recv_data, size_t send_count) const;
    
    /**
     * @brief Barrier synchronization
     */
    void barrier() const;

    /**
     * @brief Calculate local data slice for this rank
     * @param total_size Total elements to distribute
     * @return Pair of (local_start_idx, local_count)
     */
    std::pair<size_t, size_t> get_local_slice(size_t total_size) const;

    /**
     * @brief Calculate row distribution for tensor sharding
     * @param total_rows Total rows in global tensor
     * @return Vector of row counts per rank
     */
    std::vector<size_t> distribute_rows(size_t total_rows) const;

private:
    int rank_;
    int world_size_;
    MPI_Comm comm_;
};

/**
 * @brief Factory for creating MPI contexts
 */
class MPIContextFactory
{
public:
    /**
     * @brief Get global MPI context (singleton)
     */
    static std::shared_ptr<MPIContext> global();

    /**
     * @brief Create sub-context for specific communicator
     */
    static std::shared_ptr<MPIContext> create(MPI_Comm comm);

    /**
     * @brief Create mock context for testing (no MPI initialization required)
     */
    static std::shared_ptr<MPIContext> create_mock(int rank = 0, int world_size = 1);
};

} // namespace llaminar
