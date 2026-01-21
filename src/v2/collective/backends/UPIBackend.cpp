/**
 * @file UPIBackend.cpp
 * @brief UPI-based collective backend implementation
 *
 * Implements cross-socket CPU tensor parallelism using MPI over UPI.
 * Uses domain-specific MPI communicators for isolated collective operations.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "UPIBackend.h"
#include "../../utils/Logger.h"
#include "../../utils/NodeTopology.h"
#include <fstream>
#include <sstream>

namespace llaminar2
{

    // =============================================================================
    // Constructor / Destructor
    // =============================================================================

    UPICollectiveBackend::UPICollectiveBackend(MPI_Comm domain_comm,
                                               const NodeTopology *topology)
        : domain_comm_(domain_comm), domain_rank_(-1), domain_size_(0), bandwidth_gbps_(estimateBandwidth(topology))
    {
        if (domain_comm_ != MPI_COMM_NULL)
        {
            MPI_Comm_rank(domain_comm_, &domain_rank_);
            MPI_Comm_size(domain_comm_, &domain_size_);

            LOG_DEBUG("UPICollectiveBackend: Created with domain_rank=" << domain_rank_
                                                                        << ", domain_size=" << domain_size_
                                                                        << ", estimated_bandwidth=" << bandwidth_gbps_ << " GB/s");
        }
        else
        {
            LOG_WARN("UPICollectiveBackend: Created with MPI_COMM_NULL communicator");
        }
    }

    UPICollectiveBackend::~UPICollectiveBackend()
    {
        shutdown();
        // NOTE: We do NOT free domain_comm_ here - the caller owns it
        // (typically TPDomainBuilder or similar creates and manages the communicator)
    }

    // =============================================================================
    // Capability Queries
    // =============================================================================

    bool UPICollectiveBackend::supportsDevice(DeviceType type) const
    {
        // UPI operates on host memory only - CPU is directly supported
        return type == DeviceType::CPU;
    }

    bool UPICollectiveBackend::supportsDirectTransfer(DeviceId src, DeviceId dst) const
    {
        // UPI can only directly transfer between CPU buffers
        // Both should ideally be NUMA-local for optimal performance
        return src.type == DeviceType::CPU && dst.type == DeviceType::CPU;
    }

    bool UPICollectiveBackend::isAvailable() const
    {
        // Available if we have a valid communicator with at least 1 rank
        return domain_comm_ != MPI_COMM_NULL && domain_size_ > 0;
    }

    // =============================================================================
    // Lifecycle
    // =============================================================================

    bool UPICollectiveBackend::initialize(const DeviceGroup &group)
    {
        if (domain_comm_ == MPI_COMM_NULL)
        {
            last_error_ = "Cannot initialize UPI backend with null communicator";
            LOG_ERROR("UPICollectiveBackend::initialize - " << last_error_);
            return false;
        }

        // Validate group scope - UPI is used for LOCAL scope (same node, cross-socket)
        if (group.scope != CollectiveScope::LOCAL)
        {
            LOG_WARN("UPICollectiveBackend: Expected LOCAL scope for cross-socket CPU TP, got "
                     << (group.scope == CollectiveScope::GLOBAL ? "GLOBAL" : "unknown"));
        }

        group_ = group;
        initialized_ = true;

        LOG_DEBUG("UPICollectiveBackend initialized for group '" << group.name
                                                                 << "' with " << domain_size_ << " ranks over UPI");

        return true;
    }

    bool UPICollectiveBackend::isInitialized() const
    {
        return initialized_;
    }

    void UPICollectiveBackend::shutdown()
    {
        if (initialized_)
        {
            LOG_DEBUG("UPICollectiveBackend shutdown");
        }
        initialized_ = false;
        // NOTE: We do NOT free domain_comm_ - caller owns it
    }

    // =============================================================================
    // Collective Operations
    // =============================================================================

    bool UPICollectiveBackend::allreduce(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        if (domain_comm_ == MPI_COMM_NULL)
        {
            last_error_ = "UPI backend has null communicator";
            return false;
        }

        if (!initialized_)
        {
            last_error_ = "UPI backend not initialized";
            return false;
        }

        MPI_Datatype mpi_dtype = toMPIDatatype(dtype);
        MPI_Op mpi_op = toMPIOp(op);

        // Use MPI_IN_PLACE for efficient in-place allreduce
        int result = MPI_Allreduce(
            MPI_IN_PLACE,
            buffer,
            static_cast<int>(count),
            mpi_dtype,
            mpi_op,
            domain_comm_);

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Allreduce failed with code " + std::to_string(result);
            LOG_ERROR("UPICollectiveBackend::allreduce - " << last_error_);
            return false;
        }

        return true;
    }

    bool UPICollectiveBackend::allgather(
        const void *send_buf,
        void *recv_buf,
        size_t send_count,
        CollectiveDataType dtype)
    {
        if (domain_comm_ == MPI_COMM_NULL)
        {
            last_error_ = "UPI backend has null communicator";
            return false;
        }

        if (!initialized_)
        {
            last_error_ = "UPI backend not initialized";
            return false;
        }

        MPI_Datatype mpi_dtype = toMPIDatatype(dtype);

        int result = MPI_Allgather(
            send_buf,
            static_cast<int>(send_count),
            mpi_dtype,
            recv_buf,
            static_cast<int>(send_count),
            mpi_dtype,
            domain_comm_);

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Allgather failed with code " + std::to_string(result);
            LOG_ERROR("UPICollectiveBackend::allgather - " << last_error_);
            return false;
        }

        return true;
    }

    bool UPICollectiveBackend::allgatherv(
        const void *send_buf,
        size_t send_count,
        void *recv_buf,
        const std::vector<int> &recv_counts,
        const std::vector<int> &displacements,
        CollectiveDataType dtype)
    {
        if (domain_comm_ == MPI_COMM_NULL)
        {
            last_error_ = "UPI backend has null communicator";
            return false;
        }

        if (!initialized_)
        {
            last_error_ = "UPI backend not initialized";
            return false;
        }

        MPI_Datatype mpi_dtype = toMPIDatatype(dtype);

        int result = MPI_Allgatherv(
            send_buf,
            static_cast<int>(send_count),
            mpi_dtype,
            recv_buf,
            recv_counts.data(),
            displacements.data(),
            mpi_dtype,
            domain_comm_);

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Allgatherv failed with code " + std::to_string(result);
            LOG_ERROR("UPICollectiveBackend::allgatherv - " << last_error_);
            return false;
        }

        return true;
    }

    bool UPICollectiveBackend::reduceScatter(
        const void *send_buf,
        void *recv_buf,
        size_t recv_count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        if (domain_comm_ == MPI_COMM_NULL)
        {
            last_error_ = "UPI backend has null communicator";
            return false;
        }

        if (!initialized_)
        {
            last_error_ = "UPI backend not initialized";
            return false;
        }

        MPI_Datatype mpi_dtype = toMPIDatatype(dtype);
        MPI_Op mpi_op = toMPIOp(op);

        // MPI_Reduce_scatter requires an array of recvcounts (one per rank)
        // For uniform distribution, all ranks get the same count
        std::vector<int> recvcounts(domain_size_, static_cast<int>(recv_count));

        int result = MPI_Reduce_scatter(
            send_buf,
            recv_buf,
            recvcounts.data(),
            mpi_dtype,
            mpi_op,
            domain_comm_);

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Reduce_scatter failed with code " + std::to_string(result);
            LOG_ERROR("UPICollectiveBackend::reduceScatter - " << last_error_);
            return false;
        }

        return true;
    }

    bool UPICollectiveBackend::broadcast(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        int root_rank)
    {
        if (domain_comm_ == MPI_COMM_NULL)
        {
            last_error_ = "UPI backend has null communicator";
            return false;
        }

        if (!initialized_)
        {
            last_error_ = "UPI backend not initialized";
            return false;
        }

        MPI_Datatype mpi_dtype = toMPIDatatype(dtype);

        int result = MPI_Bcast(
            buffer,
            static_cast<int>(count),
            mpi_dtype,
            root_rank,
            domain_comm_);

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Bcast failed with code " + std::to_string(result);
            LOG_ERROR("UPICollectiveBackend::broadcast - " << last_error_);
            return false;
        }

        return true;
    }

    bool UPICollectiveBackend::synchronize()
    {
        if (domain_comm_ == MPI_COMM_NULL)
        {
            last_error_ = "UPI backend has null communicator";
            return false;
        }

        if (!initialized_)
        {
            last_error_ = "UPI backend not initialized";
            return false;
        }

        int result = MPI_Barrier(domain_comm_);

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Barrier failed with code " + std::to_string(result);
            LOG_ERROR("UPICollectiveBackend::synchronize - " << last_error_);
            return false;
        }

        return true;
    }

    // =============================================================================
    // Static Helpers
    // =============================================================================

    MPI_Datatype UPICollectiveBackend::toMPIDatatype(CollectiveDataType dtype)
    {
        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
            return MPI_FLOAT;
        case CollectiveDataType::FLOAT16:
            // FP16 doesn't have native MPI type - use MPI_UINT16_T for raw bytes
            // Note: Reduction operations (SUM/MAX/MIN) won't work correctly for FP16
            // with this approach - would need custom MPI_Op for true FP16 reduction
            return MPI_UINT16_T;
        case CollectiveDataType::BFLOAT16:
            // BF16 doesn't have native MPI type - use MPI_UINT16_T for raw bytes
            // Same caveat as FP16 for reduction operations
            return MPI_UINT16_T;
        case CollectiveDataType::INT32:
            return MPI_INT;
        case CollectiveDataType::INT8:
            return MPI_INT8_T;
        default:
            LOG_WARN("UPICollectiveBackend::toMPIDatatype - Unknown dtype, defaulting to MPI_BYTE");
            return MPI_BYTE;
        }
    }

    MPI_Op UPICollectiveBackend::toMPIOp(CollectiveOp op)
    {
        switch (op)
        {
        case CollectiveOp::ALLREDUCE_SUM:
            return MPI_SUM;
        case CollectiveOp::ALLREDUCE_MAX:
            return MPI_MAX;
        case CollectiveOp::ALLREDUCE_MIN:
            return MPI_MIN;
        default:
            // For non-reduction ops (ALLGATHER, BROADCAST), return SUM as default
            // (caller should not use reduction op for non-reduction collectives)
            return MPI_SUM;
        }
    }

    float UPICollectiveBackend::estimateBandwidth(const NodeTopology *topology)
    {
        // If we have topology info with inter-socket links, use that
        if (topology != nullptr && topology->numSockets() > 1)
        {
            // Get link info from topology if available
            // Default to Intel UPI bandwidth for now
            // AMD Infinity Fabric is typically faster (~100 GB/s)

            // Try to detect CPU vendor from /proc/cpuinfo
            std::ifstream cpuinfo("/proc/cpuinfo");
            if (cpuinfo.is_open())
            {
                std::string line;
                while (std::getline(cpuinfo, line))
                {
                    if (line.find("vendor_id") != std::string::npos)
                    {
                        if (line.find("AuthenticAMD") != std::string::npos)
                        {
                            LOG_DEBUG("UPICollectiveBackend: Detected AMD CPU, "
                                      "estimating Infinity Fabric bandwidth ~100 GB/s");
                            return 100.0f; // AMD Infinity Fabric
                        }
                        else if (line.find("GenuineIntel") != std::string::npos)
                        {
                            LOG_DEBUG("UPICollectiveBackend: Detected Intel CPU, "
                                      "estimating UPI bandwidth ~50 GB/s");
                            return 50.0f; // Intel UPI
                        }
                        break;
                    }
                }
            }
        }

        // Default to conservative Intel UPI estimate
        return 50.0f;
    }

} // namespace llaminar2
