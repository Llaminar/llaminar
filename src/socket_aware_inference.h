#pragma once

#include <vector>
#include <memory>
#include <mpi.h>
#include "topology_manager.h"
#include "tensors/tensor_base.h"

namespace llaminar
{

    /**
     * SocketAwareInferenceManager - Optimizes inference for multi-socket Xeon servers
     *
     * Architecture:
     * - One MPI process per CPU socket/NUMA node
     * - Socket-local memory allocation and computation
     * - Efficient work distribution across sockets
     * - Minimal cross-socket data movement
     */
    class SocketAwareInferenceManager
    {
    public:
        struct SocketInfo
        {
            int socket_id;              // Physical socket ID
            int numa_node;              // Corresponding NUMA node
            int mpi_rank;               // MPI rank for this socket
            std::vector<int> cpu_cores; // CPU cores in this socket
            size_t memory_mb;           // Available memory on this socket
            bool is_local;              // Is this the current process's socket?
        };

        struct WorkDistribution
        {
            int tensor_rows_per_socket;   // Rows assigned to each socket
            int remaining_rows;           // Extra rows for last socket
            std::vector<int> row_offsets; // Starting row for each socket
            std::vector<int> row_counts;  // Row count for each socket
        };

    private:
        SystemTopology topology_;
        std::vector<SocketInfo> sockets_;
        int local_socket_rank_;
        int total_sockets_;
        MPI_Comm socket_comm_; // Communicator for all socket processes
        bool numa_binding_enabled_;

    public:
        SocketAwareInferenceManager();
        ~SocketAwareInferenceManager();

        // Initialization and topology detection
        bool initialize();
        void detect_socket_topology();
        void setup_numa_binding();
        void print_socket_configuration();

        // Work distribution for inference
        WorkDistribution plan_tensor_distribution(int total_rows, int cols);
        void distribute_tensor_work(const TensorBase *input,
                                    std::vector<std::unique_ptr<TensorBase>> &socket_tensors);

        // Kernel execution coordination
        void execute_distributed_matmul(const TensorBase *A, const TensorBase *B, TensorBase *C);
        void execute_distributed_attention(const TensorBase *input, TensorBase *output);

        // Result aggregation
        void gather_tensor_results(const std::vector<std::unique_ptr<TensorBase>> &socket_results,
                                   TensorBase *final_result);

        // Getters
        int get_local_socket_rank() const { return local_socket_rank_; }
        int get_total_sockets() const { return total_sockets_; }
        const std::vector<SocketInfo> &get_sockets() const { return sockets_; }
        bool is_socket_coordinator() const { return local_socket_rank_ == 0; }

        // Memory management
        std::unique_ptr<TensorBase> allocate_socket_local_tensor(
            const std::vector<int> &shape);
        void optimize_memory_placement();

    private:
        void bind_to_socket(int socket_id);
        void setup_socket_communicator();
        int detect_local_socket();
        void validate_socket_configuration();
    };

} // namespace llaminar