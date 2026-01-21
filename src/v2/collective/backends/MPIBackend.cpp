/**
 * @file MPIBackend.cpp
 * @brief MPI-based collective backend implementation
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "MPIBackend.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    MPIBackend::MPIBackend(std::shared_ptr<MPIContext> mpi_ctx)
        : mpi_ctx_(std::move(mpi_ctx))
    {
    }

    MPIBackend::~MPIBackend()
    {
        shutdown();
    }

    // =========================================================================
    // Capability Queries
    // =========================================================================

    bool MPIBackend::supportsDevice(DeviceType type) const
    {
        // MPI operates on host memory only
        return type == DeviceType::CPU;
    }

    bool MPIBackend::supportsDirectTransfer(DeviceId src, DeviceId dst) const
    {
        // MPI can only directly transfer between CPU buffers
        return src.type == DeviceType::CPU && dst.type == DeviceType::CPU;
    }

    bool MPIBackend::isAvailable() const
    {
        return mpi_ctx_ != nullptr && mpi_ctx_->world_size() > 0;
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    bool MPIBackend::initialize(const DeviceGroup &group)
    {
        if (!mpi_ctx_)
        {
            last_error_ = "Cannot initialize MPI backend without MPI context";
            LOG_ERROR("MPIBackend::initialize - " << last_error_);
            return false;
        }

        // Validate group scope - MPI is typically used for GLOBAL scope
        // but can also be used for LOCAL scope with appropriate configuration
        if (group.scope != CollectiveScope::GLOBAL &&
            group.scope != CollectiveScope::LOCAL)
        {
            last_error_ = "Invalid group scope for MPI backend";
            LOG_ERROR("MPIBackend::initialize - " << last_error_);
            return false;
        }

        group_ = group;
        initialized_ = true;

        LOG_DEBUG("MPIBackend initialized for group '" << group.name
                                                       << "' with " << mpi_ctx_->world_size() << " ranks");

        return true;
    }

    bool MPIBackend::isInitialized() const
    {
        return initialized_;
    }

    void MPIBackend::shutdown()
    {
        if (initialized_)
        {
            LOG_DEBUG("MPIBackend shutdown");
        }
        initialized_ = false;
        // Note: We don't call MPI_Finalize here - that's managed by MPIContext
    }

    // =========================================================================
    // Collective Operations
    // =========================================================================

    bool MPIBackend::allreduce(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        if (!initialized_ || !mpi_ctx_)
        {
            last_error_ = "MPI backend not initialized";
            return false;
        }

        MPI_Datatype mpi_dtype = toMPIDatatype(dtype);
        MPI_Op mpi_op = toMPIOp(op);

        int result = MPI_Allreduce(
            MPI_IN_PLACE,
            buffer,
            static_cast<int>(count),
            mpi_dtype,
            mpi_op,
            mpi_ctx_->comm());

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Allreduce failed with code " + std::to_string(result);
            LOG_ERROR("MPIBackend::allreduce - " << last_error_);
            return false;
        }

        return true;
    }

    bool MPIBackend::allgather(
        const void *send_buf,
        void *recv_buf,
        size_t send_count,
        CollectiveDataType dtype)
    {
        if (!initialized_ || !mpi_ctx_)
        {
            last_error_ = "MPI backend not initialized";
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
            mpi_ctx_->comm());

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Allgather failed with code " + std::to_string(result);
            LOG_ERROR("MPIBackend::allgather - " << last_error_);
            return false;
        }

        return true;
    }

    bool MPIBackend::allgatherv(
        const void *send_buf,
        size_t send_count,
        void *recv_buf,
        const std::vector<int> &recv_counts,
        const std::vector<int> &displacements,
        CollectiveDataType dtype)
    {
        if (!initialized_ || !mpi_ctx_)
        {
            last_error_ = "MPI backend not initialized";
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
            mpi_ctx_->comm());

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Allgatherv failed with code " + std::to_string(result);
            LOG_ERROR("MPIBackend::allgatherv - " << last_error_);
            return false;
        }

        return true;
    }

    bool MPIBackend::reduceScatter(
        const void *send_buf,
        void *recv_buf,
        size_t recv_count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        if (!initialized_ || !mpi_ctx_)
        {
            last_error_ = "MPI backend not initialized";
            return false;
        }

        MPI_Datatype mpi_dtype = toMPIDatatype(dtype);
        MPI_Op mpi_op = toMPIOp(op);

        // MPI_Reduce_scatter requires an array of recvcounts (one per rank)
        // For uniform distribution, all ranks get the same count
        std::vector<int> recvcounts(mpi_ctx_->world_size(), static_cast<int>(recv_count));

        int result = MPI_Reduce_scatter(
            send_buf,
            recv_buf,
            recvcounts.data(),
            mpi_dtype,
            mpi_op,
            mpi_ctx_->comm());

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Reduce_scatter failed with code " + std::to_string(result);
            LOG_ERROR("MPIBackend::reduceScatter - " << last_error_);
            return false;
        }

        return true;
    }

    bool MPIBackend::broadcast(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        int root_rank)
    {
        if (!initialized_ || !mpi_ctx_)
        {
            last_error_ = "MPI backend not initialized";
            return false;
        }

        MPI_Datatype mpi_dtype = toMPIDatatype(dtype);

        int result = MPI_Bcast(
            buffer,
            static_cast<int>(count),
            mpi_dtype,
            root_rank,
            mpi_ctx_->comm());

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Bcast failed with code " + std::to_string(result);
            LOG_ERROR("MPIBackend::broadcast - " << last_error_);
            return false;
        }

        return true;
    }

    bool MPIBackend::synchronize()
    {
        if (!initialized_ || !mpi_ctx_)
        {
            last_error_ = "MPI backend not initialized";
            return false;
        }

        int result = MPI_Barrier(mpi_ctx_->comm());

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Barrier failed with code " + std::to_string(result);
            LOG_ERROR("MPIBackend::synchronize - " << last_error_);
            return false;
        }

        return true;
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    MPI_Datatype MPIBackend::toMPIDatatype(CollectiveDataType dtype) const
    {
        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
            return MPI_FLOAT;
        case CollectiveDataType::FLOAT16:
            // FP16 doesn't have native MPI type - pack as 2 bytes
            return MPI_SHORT;
        case CollectiveDataType::BFLOAT16:
            // BF16 doesn't have native MPI type - pack as 2 bytes
            return MPI_SHORT;
        case CollectiveDataType::INT32:
            return MPI_INT;
        case CollectiveDataType::INT8:
            return MPI_CHAR;
        default:
            LOG_WARN("MPIBackend::toMPIDatatype - Unknown dtype, defaulting to MPI_BYTE");
            return MPI_BYTE;
        }
    }

    MPI_Op MPIBackend::toMPIOp(CollectiveOp op) const
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
            // For non-reduction ops, return SUM as default
            // (caller should not use reduction op for non-reduction collectives)
            return MPI_SUM;
        }
    }

} // namespace llaminar2
