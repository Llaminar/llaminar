/**
 * @file RCCLBackend.cpp
 * @brief RCCL-based collective backend implementation
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "RCCLBackend.h"
#include "../../utils/Logger.h"

#ifdef HAVE_RCCL
#include <hip/hip_runtime.h>
#endif

namespace llaminar2
{

    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    RCCLBackend::RCCLBackend()
    {
        LOG_DEBUG("RCCLBackend: Created");
    }

    RCCLBackend::~RCCLBackend()
    {
        if (initialized_)
        {
            shutdown();
        }
    }

    // =========================================================================
    // Availability Check
    // =========================================================================

    bool RCCLBackend::isAvailable() const
    {
#ifdef HAVE_RCCL
        int rocm_device_count = 0;
        hipError_t err = hipGetDeviceCount(&rocm_device_count);
        return (err == hipSuccess && rocm_device_count > 0);
#else
        return false;
#endif
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    bool RCCLBackend::initialize(const DeviceGroup &group)
    {
#ifdef HAVE_RCCL
        if (initialized_)
        {
            LOG_WARN("RCCLBackend::initialize: Already initialized, shutting down first");
            shutdown();
        }

        // Validate all devices are ROCm
        for (const auto &device : group.devices)
        {
            if (device.type != DeviceType::ROCm)
            {
                last_error_ = "RCCLBackend only supports ROCm devices";
                LOG_ERROR(last_error_);
                return false;
            }
        }

        num_ranks_ = static_cast<int>(group.size());
        local_rank_ = group.local_rank;

        if (num_ranks_ < 1)
        {
            last_error_ = "RCCLBackend requires at least 1 device";
            LOG_ERROR(last_error_);
            return false;
        }

        // Set the HIP device for this rank
        DeviceId local_device = group.localDevice();
        hipError_t hip_err = hipSetDevice(local_device.ordinal);
        if (hip_err != hipSuccess)
        {
            last_error_ = "Failed to set HIP device " + std::to_string(local_device.ordinal);
            LOG_ERROR(last_error_);
            return false;
        }

        // Create HIP stream
        hip_err = hipStreamCreate(&stream_);
        if (hip_err != hipSuccess)
        {
            last_error_ = "Failed to create HIP stream";
            LOG_ERROR(last_error_);
            return false;
        }

        // Create RCCL communicator
        // For single-process multi-GPU, we use ncclCommInitAll
        // For multi-process, we'd use ncclCommInitRank with unique ID
        if (num_ranks_ == 1)
        {
            // Single GPU - create a trivial communicator
            ncclResult_t rccl_err = ncclCommInitAll(&comm_, 1, nullptr);
            if (rccl_err != ncclSuccess)
            {
                last_error_ = "ncclCommInitAll failed: " + std::string(ncclGetErrorString(rccl_err));
                LOG_ERROR(last_error_);
                hipStreamDestroy(stream_);
                stream_ = nullptr;
                return false;
            }
        }
        else
        {
            // Multi-GPU single process
            // TODO: For multi-process, need to broadcast unique ID via MPI
            ncclUniqueId id;
            ncclResult_t rccl_err = ncclGetUniqueId(&id);
            if (rccl_err != ncclSuccess)
            {
                last_error_ = "ncclGetUniqueId failed";
                LOG_ERROR(last_error_);
                hipStreamDestroy(stream_);
                stream_ = nullptr;
                return false;
            }

            rccl_err = ncclCommInitRank(&comm_, num_ranks_, id, local_rank_);
            if (rccl_err != ncclSuccess)
            {
                last_error_ = "ncclCommInitRank failed: " + std::string(ncclGetErrorString(rccl_err));
                LOG_ERROR(last_error_);
                hipStreamDestroy(stream_);
                stream_ = nullptr;
                return false;
            }
        }

        initialized_ = true;
        LOG_INFO("RCCLBackend: Initialized with " << num_ranks_ << " GPU(s), local_rank=" << local_rank_);
        return true;
#else
        last_error_ = "RCCL not available (HAVE_RCCL not defined)";
        LOG_ERROR(last_error_);
        return false;
#endif
    }

    void RCCLBackend::shutdown()
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            return;
        }

        if (comm_)
        {
            ncclCommDestroy(comm_);
            comm_ = nullptr;
        }

        if (stream_)
        {
            hipStreamDestroy(stream_);
            stream_ = nullptr;
        }

        initialized_ = false;
        LOG_DEBUG("RCCLBackend: Shutdown complete");
#endif
    }

    // =========================================================================
    // Collective Operations
    // =========================================================================

    bool RCCLBackend::allreduce(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            return false;
        }

        ncclResult_t err = ncclAllReduce(
            buffer,
            buffer, // In-place: send and recv are the same
            count,
            toRcclDataType(dtype),
            toRcclRedOp(op),
            comm_,
            stream_);

        if (err != ncclSuccess)
        {
            last_error_ = "ncclAllReduce failed: " + std::string(ncclGetErrorString(err));
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLBackend::allgather(
        const void *send_buf,
        void *recv_buf,
        size_t send_count,
        CollectiveDataType dtype)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            return false;
        }

        ncclResult_t err = ncclAllGather(
            send_buf,
            recv_buf,
            send_count,
            toRcclDataType(dtype),
            comm_,
            stream_);

        if (err != ncclSuccess)
        {
            last_error_ = "ncclAllGather failed: " + std::string(ncclGetErrorString(err));
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLBackend::allgatherv(
        const void *send_buf,
        size_t send_count,
        void *recv_buf,
        const std::vector<int> &recv_counts,
        const std::vector<int> &displacements,
        CollectiveDataType dtype)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            return false;
        }

        // RCCL does not have a native allgatherv. We emulate it using ncclSend/ncclRecv.
        // For now, we use a simpler approach: regular allgather with max count, then extract.
        // This is less efficient but works correctly.

        // Find max recv count to use as uniform count
        int max_count = 0;
        for (int c : recv_counts)
        {
            max_count = std::max(max_count, c);
        }

        // If all counts are equal, use regular allgather
        bool all_equal = true;
        for (int c : recv_counts)
        {
            if (c != recv_counts[0])
            {
                all_equal = false;
                break;
            }
        }

        if (all_equal)
        {
            return allgather(send_buf, recv_buf, send_count, dtype);
        }

        // Variable counts - RCCL doesn't support this natively
        // Fall back to point-to-point sends/recvs
        ncclDataType_t rccl_dtype = toRcclDataType(dtype);

        ncclResult_t err = ncclGroupStart();
        if (err != ncclSuccess)
        {
            last_error_ = "ncclGroupStart failed";
            return false;
        }

        size_t dtype_size = 0;
        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
            dtype_size = 4;
            break;
        case CollectiveDataType::FLOAT16:
        case CollectiveDataType::BFLOAT16:
            dtype_size = 2;
            break;
        case CollectiveDataType::INT32:
            dtype_size = 4;
            break;
        case CollectiveDataType::INT8:
            dtype_size = 1;
            break;
        }

        // Each rank sends to all others and receives from all others
        for (int peer = 0; peer < num_ranks_; ++peer)
        {
            // Send my data to peer
            err = ncclSend(send_buf, send_count, rccl_dtype, peer, comm_, stream_);
            if (err != ncclSuccess)
            {
                ncclGroupEnd();
                last_error_ = "ncclSend failed in allgatherv";
                return false;
            }

            // Receive from peer at their offset
            char *recv_ptr = static_cast<char *>(recv_buf) + displacements[peer] * dtype_size;
            err = ncclRecv(recv_ptr, recv_counts[peer], rccl_dtype, peer, comm_, stream_);
            if (err != ncclSuccess)
            {
                ncclGroupEnd();
                last_error_ = "ncclRecv failed in allgatherv";
                return false;
            }
        }

        err = ncclGroupEnd();
        if (err != ncclSuccess)
        {
            last_error_ = "ncclGroupEnd failed";
            return false;
        }

        return true;
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLBackend::reduceScatter(
        const void *send_buf,
        void *recv_buf,
        size_t recv_count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            return false;
        }

        ncclResult_t err = ncclReduceScatter(
            send_buf,
            recv_buf,
            recv_count,
            toRcclDataType(dtype),
            toRcclRedOp(op),
            comm_,
            stream_);

        if (err != ncclSuccess)
        {
            last_error_ = "ncclReduceScatter failed: " + std::string(ncclGetErrorString(err));
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLBackend::broadcast(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        int root)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            return false;
        }

        ncclResult_t err = ncclBroadcast(
            buffer,
            buffer,
            count,
            toRcclDataType(dtype),
            root,
            comm_,
            stream_);

        if (err != ncclSuccess)
        {
            last_error_ = "ncclBroadcast failed: " + std::string(ncclGetErrorString(err));
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    // =========================================================================
    // Synchronization
    // =========================================================================

    bool RCCLBackend::synchronize()
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            return true;
        }

        hipError_t err = hipStreamSynchronize(stream_);
        if (err != hipSuccess)
        {
            last_error_ = "hipStreamSynchronize failed";
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        return true;
#endif
    }

    // =========================================================================
    // Type Conversion Helpers
    // =========================================================================

#ifdef HAVE_RCCL
    ncclDataType_t RCCLBackend::toRcclDataType(CollectiveDataType dtype)
    {
        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
            return ncclFloat32;
        case CollectiveDataType::FLOAT16:
            return ncclFloat16;
        case CollectiveDataType::BFLOAT16:
            return ncclBfloat16;
        case CollectiveDataType::INT32:
            return ncclInt32;
        case CollectiveDataType::INT8:
            return ncclInt8;
        default:
            LOG_WARN("RCCLBackend: Unknown dtype, defaulting to float32");
            return ncclFloat32;
        }
    }

    ncclRedOp_t RCCLBackend::toRcclRedOp(CollectiveOp op)
    {
        switch (op)
        {
        case CollectiveOp::ALLREDUCE_SUM:
            return ncclSum;
        case CollectiveOp::ALLREDUCE_MAX:
            return ncclMax;
        case CollectiveOp::ALLREDUCE_MIN:
            return ncclMin;
        default:
            LOG_WARN("RCCLBackend: Unknown op, defaulting to SUM");
            return ncclSum;
        }
    }
#endif

} // namespace llaminar2
