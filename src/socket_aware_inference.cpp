#include "socket_aware_inference.h"
#include "logger.h"
#include "tensors/tensor_base.h"
#include <numa.h>
#include <sched.h>
#include <unistd.h>
#include <algorithm>
#include <sstream>
#include <iostream>

namespace llaminar
{

    SocketAwareInferenceManager::SocketAwareInferenceManager()
        : local_socket_rank_(-1), total_sockets_(0), socket_comm_(MPI_COMM_NULL),
          numa_binding_enabled_(false)
    {
    }

    SocketAwareInferenceManager::~SocketAwareInferenceManager()
    {
        if (socket_comm_ != MPI_COMM_NULL)
        {
            MPI_Comm_free(&socket_comm_);
        }
    }

    bool SocketAwareInferenceManager::initialize()
    {
        LOG_INFO("Initializing Socket-Aware Inference Manager");

        // Detect system topology
        TopologyManager topo_mgr;
        topology_ = topo_mgr.detectSystemTopology(false, false);

        if (!topology_.numa.numa_available)
        {
            LOG_ERROR("NUMA not available - socket optimization not possible");
            return false;
        }

        // Detect and configure socket topology
        detect_socket_topology();

        // Setup NUMA binding
        setup_numa_binding();

        // Setup socket-level MPI communication
        setup_socket_communicator();

        // Validate configuration
        validate_socket_configuration();

        print_socket_configuration();

        LOG_INFO("Socket-Aware Inference Manager initialized successfully");
        return true;
    }

    void SocketAwareInferenceManager::detect_socket_topology()
    {
        LOG_INFO("Detecting socket topology");

        // In most Xeon systems, each socket corresponds to a NUMA node
        total_sockets_ = topology_.numa.numa_nodes;

        sockets_.clear();
        sockets_.reserve(total_sockets_);

        for (int socket_id = 0; socket_id < total_sockets_; ++socket_id)
        {
            SocketInfo socket_info;
            socket_info.socket_id = socket_id;
            socket_info.numa_node = socket_id; // Assume 1:1 mapping
            socket_info.mpi_rank = socket_id;  // Will be updated based on actual MPI ranks

            // Get CPUs for this NUMA node
            auto numa_iter = topology_.numa.node_to_cpus.find(socket_id);
            if (numa_iter != topology_.numa.node_to_cpus.end())
            {
                socket_info.cpu_cores = numa_iter->second;
            }

            // Get memory for this NUMA node
            auto mem_iter = topology_.numa.node_memory_gb.find(socket_id);
            if (mem_iter != topology_.numa.node_memory_gb.end())
            {
                socket_info.memory_mb = mem_iter->second * 1024; // Convert GB to MB
            }

            // Determine if this is the local socket
            socket_info.is_local = (socket_id == detect_local_socket());
            if (socket_info.is_local)
            {
                local_socket_rank_ = socket_id;
            }

            sockets_.push_back(socket_info);

            LOG_DEBUG("Socket " << socket_id << " - NUMA node: " << socket_info.numa_node << ", CPUs: " << socket_info.cpu_cores.size() << ", Memory: " << socket_info.memory_mb << " MB");
        }
    }

    int SocketAwareInferenceManager::detect_local_socket()
    {
        // Get current CPU and determine its NUMA node
        int current_cpu = sched_getcpu();

        // Find which NUMA node this CPU belongs to
        for (const auto &[numa_node, cpus] : topology_.numa.node_to_cpus)
        {
            if (std::find(cpus.begin(), cpus.end(), current_cpu) != cpus.end())
            {
                return numa_node; // Assuming socket_id == numa_node
            }
        }

        LOG_WARN("Could not detect local socket for CPU " << current_cpu << ", defaulting to 0");
        return 0;
    }

    void SocketAwareInferenceManager::setup_numa_binding()
    {
        if (local_socket_rank_ >= 0 && local_socket_rank_ < total_sockets_)
        {
            bind_to_socket(local_socket_rank_);
            numa_binding_enabled_ = true;
            LOG_INFO("Bound MPI process to socket " << local_socket_rank_);
        }
        else
        {
            LOG_WARN("Invalid local socket rank " << local_socket_rank_ << ", skipping NUMA binding");
        }
    }

    void SocketAwareInferenceManager::bind_to_socket(int socket_id)
    {
#ifdef HAVE_NUMA
        if (socket_id < 0 || socket_id >= total_sockets_)
        {
            LOG_ERROR("Invalid socket ID " << socket_id << " for binding");
            return;
        }

        // Bind memory allocation to this NUMA node
        struct bitmask *numa_mask = numa_allocate_nodemask();
        numa_bitmask_setbit(numa_mask, socket_id);

        // Set memory policy
        numa_set_membind(numa_mask);
        numa_set_preferred(socket_id);

        // Bind process to CPUs of this socket
        const auto &socket_cpus = sockets_[socket_id].cpu_cores;
        cpu_set_t cpu_mask;
        CPU_ZERO(&cpu_mask);

        for (int cpu : socket_cpus)
        {
            CPU_SET(cpu, &cpu_mask);
        }

        if (sched_setaffinity(0, sizeof(cpu_mask), &cpu_mask) != 0)
        {
            LOG_WARN("Failed to set CPU affinity for socket " << socket_id);
        }

        numa_free_nodemask(numa_mask);

        LOG_DEBUG("Bound process to socket " << socket_id << " with " << socket_cpus.size() << " CPUs");
#else
        LOG_WARN("NUMA support not compiled, cannot bind to socket " << socket_id);
#endif
    }

    void SocketAwareInferenceManager::setup_socket_communicator()
    {
        // For socket-aware inference, we want all socket processes in one communicator
        // Each process represents one socket

        int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        if (mpi_size != total_sockets_)
        {
            LOG_WARN("MPI process count (" << mpi_size << ") doesn't match socket count (" << total_sockets_ << ")");
            LOG_WARN("For optimal performance, run with: mpirun -np " << total_sockets_ << " --bind-to socket");
        }

        // For now, use MPI_COMM_WORLD as socket communicator
        // In future, could create socket-specific subcommunicators
        MPI_Comm_dup(MPI_COMM_WORLD, &socket_comm_);

        // Update actual MPI rank mapping
        if (mpi_rank < sockets_.size())
        {
            sockets_[mpi_rank].mpi_rank = mpi_rank;
            sockets_[mpi_rank].is_local = true;
            local_socket_rank_ = mpi_rank;
        }
    }

    void SocketAwareInferenceManager::validate_socket_configuration()
    {
        int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        if (mpi_size > total_sockets_)
        {
            LOG_ERROR("Too many MPI processes (" << mpi_size << ") for available sockets (" << total_sockets_ << ")");
            throw std::runtime_error("Invalid MPI configuration for socket optimization");
        }

        if (local_socket_rank_ < 0)
        {
            LOG_ERROR("Could not determine local socket for MPI rank " << mpi_rank);
            throw std::runtime_error("Socket detection failed");
        }
    }

    void SocketAwareInferenceManager::print_socket_configuration()
    {
        if (is_socket_coordinator())
        {
            LOG_INFO("=== Socket-Aware Inference Configuration ===");
            LOG_INFO("Total sockets detected: " << total_sockets_);
            LOG_INFO("NUMA binding enabled: " << (numa_binding_enabled_ ? "true" : "false"));

            for (const auto &socket : sockets_)
            {
                LOG_INFO("Socket " << socket.socket_id << ": NUMA node " << socket.numa_node << ", " << socket.cpu_cores.size() << " CPUs, " << socket.memory_mb << " MB memory");
            }
            LOG_INFO("============================================");
        }

        // Each process logs its own binding
        LOG_INFO("MPI rank " << topology_.mpi_rank << " bound to socket " << local_socket_rank_);
    }

    SocketAwareInferenceManager::WorkDistribution
    SocketAwareInferenceManager::plan_tensor_distribution(int total_rows, int cols)
    {
        WorkDistribution dist;

        // Distribute rows across sockets as evenly as possible
        dist.tensor_rows_per_socket = total_rows / total_sockets_;
        dist.remaining_rows = total_rows % total_sockets_;

        dist.row_offsets.resize(total_sockets_);
        dist.row_counts.resize(total_sockets_);

        int current_offset = 0;
        for (int socket = 0; socket < total_sockets_; ++socket)
        {
            dist.row_offsets[socket] = current_offset;

            // Give extra rows to first few sockets if there's a remainder
            int rows_for_socket = dist.tensor_rows_per_socket;
            if (socket < dist.remaining_rows)
            {
                rows_for_socket++;
            }

            dist.row_counts[socket] = rows_for_socket;
            current_offset += rows_for_socket;
        }

        LOG_DEBUG("Tensor distribution: " << total_rows << " total rows across " << total_sockets_ << " sockets");
        LOG_DEBUG("Base rows per socket: " << dist.tensor_rows_per_socket << ", remainder: " << dist.remaining_rows);

        return dist;
    }

    std::unique_ptr<TensorBase> SocketAwareInferenceManager::allocate_socket_local_tensor(
        const std::vector<int> &shape)
    {

        // This would integrate with your tensor factory to create socket-local tensors
        // For now, return nullptr as placeholder - actual implementation would depend
        // on your tensor system
        LOG_DEBUG("Allocating socket-local tensor with shape [" << shape[0] << ", "
                                                                << (shape.size() > 1 ? shape[1] : 1) << "] on socket " << local_socket_rank_);

        // TODO: Integrate with actual tensor factory
        return nullptr;
    }

    void SocketAwareInferenceManager::execute_distributed_matmul(
        const TensorBase *A, const TensorBase *B, TensorBase *C)
    {

        LOG_DEBUG("Executing distributed MatMul on socket " << local_socket_rank_);

        // This would coordinate the actual distributed matrix multiplication
        // across sockets using your enhanced MatMul kernel

        // TODO: Implement socket-aware MatMul coordination
        // 1. Distribute tensor slices to sockets
        // 2. Execute local computation on each socket
        // 3. Gather results back
    }

} // namespace llaminar