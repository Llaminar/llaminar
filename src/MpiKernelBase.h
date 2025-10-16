#pragma once

#include "KernelBase.h"
#include "tensors/TensorFactory.h"
#include "MpiContext.h"
#include <mpi.h>
#include <vector>
#include <memory>
#include <stdexcept>

namespace llaminar
{

    /**
     * Distribution strategies for MPI kernel parallelization
     */
    enum class DistributionType
    {
        ROW_WISE,     // Distribute rows across ranks
        COL_WISE,     // Distribute columns across ranks
        BLOCK_WISE,   // 2D block distribution
        HEAD_WISE,    // Distribute attention heads
        VOCAB_WISE,   // Distribute vocabulary entries
        SEQUENCE_WISE // Distribute sequence positions
    };

    /**
     * Communication patterns for MPI operations
     */
    enum class CommunicationPattern
    {
        ALL_GATHER,     // Gather data from all ranks
        ALL_REDUCE,     // Reduce data across all ranks
        REDUCE_SCATTER, // Reduce and scatter results
        BROADCAST,      // Broadcast from root rank
        POINT_TO_POINT  // Direct rank-to-rank communication
    };

    /**
     * Base class for MPI-enabled operators providing common MPI functionality
     * and distribution utilities for parallel transformer operator execution.
     */
    class MPIOperatorBase : public OperatorBase
    {
    protected:
        MPIContext mpi_ctx_;   // MPI context (rank, size, comm)
        MPI_Comm comm_;        // MPI communicator (alias for mpi_ctx_.comm)
        int rank_;             // Current process rank (alias for mpi_ctx_.rank)
        int size_;             // Total number of processes (alias for mpi_ctx_.size)
        bool mpi_initialized_; // Whether MPI was initialized by this class

        // Communication buffers for optimization
        mutable std::vector<float> send_buffer_;
        mutable std::vector<float> recv_buffer_;

    public:
        /**
         * Constructor with optional MPI communicator
         * @param comm MPI communicator (defaults to MPI_COMM_WORLD)
         * @param init_mpi Whether to initialize MPI if not already initialized
         */
        explicit MPIOperatorBase(MPI_Comm comm = MPI_COMM_WORLD, bool init_mpi = true);

        /**
         * Constructor with MPIContext (preferred for new code)
         * @param ctx MPI context with rank, size, and communicator
         * @param init_mpi Whether to initialize MPI if not already initialized
         */
        explicit MPIOperatorBase(const MPIContext &ctx, bool init_mpi = true);

        /**
         * Destructor - handles MPI cleanup if this class initialized it
         */
        virtual ~MPIOperatorBase();

        // Disable copy constructor and assignment
        MPIOperatorBase(const MPIOperatorBase &) = delete;
        MPIOperatorBase &operator=(const MPIOperatorBase &) = delete;

        // Allow move operations
        MPIOperatorBase(MPIOperatorBase &&other) noexcept;
        MPIOperatorBase &operator=(MPIOperatorBase &&other) noexcept;

        /**
         * Get MPI context
         */
        const MPIContext &mpiContext() const { return mpi_ctx_; }

        /**
         * Get current MPI rank
         */
        int getRank() const { return rank_; }

        /**
         * Get total number of MPI processes
         */
        int getSize() const { return size_; }

        /**
         * Get MPI communicator
         */
        MPI_Comm getComm() const { return comm_; }

        /**
         * Check if this is the root process (rank 0)
         */
        bool isRoot() const { return rank_ == 0; }

    protected:
        /**
         * Distribution utility functions
         */

        /**
         * Calculate local size and offset for row-wise distribution
         * @param global_size Total number of elements to distribute
         * @param rank Target rank (defaults to current rank)
         * @return pair of (local_size, offset)
         */
        std::pair<int, int> getRowDistribution(int global_size, int rank = -1) const;

        /**
         * Calculate local size and offset for column-wise distribution
         * @param global_size Total number of elements to distribute
         * @param rank Target rank (defaults to current rank)
         * @return pair of (local_size, offset)
         */
        std::pair<int, int> getColDistribution(int global_size, int rank = -1) const;

        /**
         * Get 2D block distribution parameters
         * @param rows Total number of rows
         * @param cols Total number of columns
         * @param rank Target rank (defaults to current rank)
         * @return tuple of (local_rows, local_cols, row_offset, col_offset)
         */
        std::tuple<int, int, int, int> getBlockDistribution(int rows, int cols, int rank = -1) const;

        /**
         * Communication utility functions
         */

        /**
         * All-gather operation with automatic buffer management
         * @param send_data Local data to send
         * @param send_count Number of elements to send per rank
         * @param recv_data Output buffer for gathered data
         */
        void allGather(const float *send_data, int send_count, std::vector<float> &recv_data) const;

        /**
         * All-reduce operation (sum) with automatic buffer management
         * @param send_data Local data to reduce
         * @param recv_data Output buffer for reduced data
         * @param count Number of elements
         */
        void allReduceSum(const float *send_data, float *recv_data, int count) const;

        /**
         * All-reduce operation (max) with automatic buffer management
         * @param send_data Local data to reduce
         * @param recv_data Output buffer for reduced data
         * @param count Number of elements
         */
        void allReduceMax(const float *send_data, float *recv_data, int count) const;

        /**
         * Reduce-scatter operation with automatic buffer management
         * @param send_data Local data to reduce and scatter
         * @param recv_data Output buffer for scattered results
         * @param recv_counts Array of receive counts per rank
         */
        void reduceScatter(const float *send_data, float *recv_data, const int *recv_counts) const;

        /**
         * Broadcast operation
         * @param data Data buffer (input on root, output on others)
         * @param count Number of elements
         * @param root Root rank for broadcast
         */
        void broadcast(float *data, int count, int root = 0) const;

        /**
         * Asynchronous communication helpers
         */

        /**
         * Start non-blocking all-gather
         * @param send_data Local data to send
         * @param send_count Number of elements to send
         * @param recv_data Output buffer
         * @param request MPI request handle
         */
        void allGatherStart(const float *send_data, int send_count,
                            std::vector<float> &recv_data, MPI_Request *request) const;

        /**
         * Wait for completion of non-blocking operation
         * @param request MPI request handle
         */
        void waitCompletion(MPI_Request *request) const;

        /**
         * Error handling utilities
         */

        /**
         * Check MPI error code and throw exception if needed
         * @param error_code MPI error code
         * @param operation_name Name of the operation for error reporting
         */
        void checkMPIError(int error_code, const std::string &operation_name) const;

        /**
         * Synchronize all processes (barrier)
         */
        void synchronize() const;

        /**
         * Virtual method for kernel-specific MPI initialization
         * Called after base MPI setup is complete
         */
        virtual void initializeMPI() {}

        /**
         * Virtual method for kernel-specific MPI cleanup
         * Called before base MPI cleanup
         */
        virtual void finalizeMPI() {}

        /**
         * Create a local tensor with the specified shape
         * @param shape Vector of dimensions for the tensor
         * @return Shared pointer to the created tensor
         */
        std::shared_ptr<TensorBase> createLocalTensor(const std::vector<size_t> &shape) const;

        /**
         * Creates a simple tensor that's guaranteed to be compatible for MPI broadcast.
         * Use this for tensors that will be broadcast between ranks.
         * @param shape Vector of dimensions for the tensor
         * @return Shared pointer to the created SimpleTensor
         */
        std::shared_ptr<TensorBase> createBroadcastTensor(const std::vector<size_t> &shape) const;

        /**
         * Create a distributed COSMA tensor with optimal strategy for matrix operations.
         * @param shape Tensor shape as vector of dimensions
         * @param label COSMA label for the tensor (A, B, or C)
         * @param m Matrix multiplication M dimension
         * @param n Matrix multiplication N dimension
         * @param k Matrix multiplication K dimension
         * @return Shared pointer to the created COSMA tensor
         */
        std::shared_ptr<TensorBase> createDistributedTensor(const std::vector<size_t> &shape,
                                                            const std::string &label,
                                                            int m, int n, int k) const;
    };

} // namespace llaminar