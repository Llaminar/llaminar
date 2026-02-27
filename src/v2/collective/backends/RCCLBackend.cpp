/**
 * @file RCCLBackend.cpp
 * @brief RCCL-based collective backend implementation
 *
 * All HIP runtime and RCCL API calls are isolated in RCCLBackendHIP.cpp to avoid
 * conflicts with CUDA headers when building with both CUDA and ROCm support.
 * RCCL is loaded dynamically via dlopen to avoid symbol conflicts with NCCL.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "RCCLBackend.h"
#include "../coordinators/RCCLCoordinator.h"
#include "../../utils/Logger.h"

#ifdef HAVE_RCCL
#include <mpi.h>
#include <atomic>
#include <thread>
#include <string>
#include <cstring>
#include <cstdlib> // for setenv

// Forward declarations for HIP and RCCL wrappers (implemented in RCCLBackendHIP.cpp)
namespace llaminar2
{
    namespace rccl_backend_detail
    {
        // HIP runtime wrappers
        bool hipSetDeviceOrdinal(int device_ordinal);
        bool hipGetDeviceCountWrapper(int *count);
        bool hipCheckP2PAvailable(const std::vector<int> &device_ordinals);
        bool hipEnablePeerAccessForDevices(const std::vector<int> &device_ordinals, std::string &error_out);
        bool hipCreateStream(void **stream_ptr);
        bool hipDestroyStream(void *stream);
        bool hipSynchronizeStream(void *stream);
        std::string hipGetLastErrorString();
        std::string hipErrorToString(int error_code);

        // RCCL pre-initialization (checks P2P, sets env vars, loads RCCL)
        bool rcclPreInitialize(const std::vector<int> &device_ordinals, bool &p2p_available_out);

        // RCCL unique ID
        size_t rcclUniqueIdSize();
        bool rcclGetUniqueIdWrapper(void *id_out);

        // RCCL communicator management
        bool rcclCommInitRankWrapper(void **comm_out, int nranks, void *unique_id, int rank, std::string &error_out);
        bool rcclCommInitAllWrapper(void **comms_out, int ndevs, const int *devlist, std::string &error_out);
        void rcclCommDestroyWrapper(void *comm);
        void rcclCommAbortWrapper(void *comm);
        void rcclPrimeAndDestroyComm(void *comm, void *stream);

        // RCCL collective operations
        bool rcclAllReduceWrapper(void *sendbuff, void *recvbuff, size_t count,
                                  int dtype_int, int op_int, void *comm, void *stream,
                                  std::string &error_out);
        bool rcclAllGatherWrapper(const void *sendbuff, void *recvbuff, size_t sendcount,
                                  int dtype_int, void *comm, void *stream,
                                  std::string &error_out);
        bool rcclBroadcastWrapper(void *buff, size_t count, int dtype_int, int root,
                                  void *comm, void *stream, std::string &error_out);
        bool rcclReduceScatterWrapper(const void *sendbuff, void *recvbuff, size_t recvcount,
                                      int dtype_int, int op_int, void *comm, void *stream,
                                      std::string &error_out);

        // RCCL group operations (for multi-GPU single process)
        bool rcclGroupStartWrapper(std::string &error_out);
        bool rcclGroupEndWrapper(std::string &error_out);

        // Reduce operations (for heterogeneous intra-domain reduce)
        bool rcclReduceInGroupWrapper(const void *sendbuff, void *recvbuff, size_t count,
                                      int dtype_int, int op_int, int root,
                                      void *comm, void *stream, std::string &error_out);

        // Reduce-scatter operations (for bandwidth-efficient heterogeneous allreduce)
        bool rcclReduceScatterInGroupWrapper(const void *sendbuff, void *recvbuff, size_t recvcount,
                                             int dtype_int, int op_int, void *comm, void *stream,
                                             std::string &error_out);

        // Point-to-point operations (for allgatherv emulation)
        bool rcclSendWrapper(const void *sendbuff, size_t count, int dtype_int, int peer,
                             void *comm, void *stream, std::string &error_out);
        bool rcclRecvWrapper(void *recvbuff, size_t count, int dtype_int, int peer,
                             void *comm, void *stream, std::string &error_out);

        // Device memory copy operations
        bool hipMemcpySameDevice(void *dst, const void *src, size_t bytes, int device_ordinal);
        bool hipMemcpyPeerDevice(void *dst, int dst_device, const void *src, int src_device, size_t bytes);
        bool hipMemcpyAsyncSameDevice(void *dst, const void *src, size_t bytes, int device_ordinal, void *stream);
        bool hipMemcpyPeerAsyncDevice(void *dst, int dst_device, const void *src, int src_device, size_t bytes, void *stream);
        bool hipCanAccessPeerDevice(int dst_device, int src_device);
        bool hipEnablePeerAccessDevice(int peer_device);
        bool hipDeviceSynchronizeWrapper();

        // Host staging memory operations (for non-P2P fallback)
        void *hipHostMallocWrapper(size_t bytes);
        bool hipHostFreeWrapper(void *ptr);
        bool hipMemcpyD2H(void *dst_host, const void *src_device, size_t bytes, int src_device_ordinal);
        bool hipMemcpyH2D(void *dst_device, const void *src_host, size_t bytes, int dst_device_ordinal);
        bool hipMemcpyD2HAsync(void *dst_host, const void *src_device, size_t bytes, int src_device_ordinal, void *stream);
        bool hipMemcpyH2DAsync(void *dst_device, const void *src_host, size_t bytes, int dst_device_ordinal, void *stream);
    } // namespace rccl_backend_detail
} // namespace llaminar2

// Helper macro for HIP wrapper error checking
#define HIP_WRAPPER_CHECK(cmd, msg)                                                                      \
    do                                                                                                   \
    {                                                                                                    \
        if (!(cmd))                                                                                      \
        {                                                                                                \
            last_error_ = std::string(msg) + " failed: " + rccl_backend_detail::hipGetLastErrorString(); \
            LOG_ERROR(last_error_);                                                                      \
            return false;                                                                                \
        }                                                                                                \
    } while (0)

#endif // HAVE_RCCL

namespace llaminar2
{

    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    RCCLBackend::RCCLBackend(std::shared_ptr<MPIContext> mpi_ctx)
        : mpi_ctx_(std::move(mpi_ctx))
    {
        LOG_DEBUG("RCCLBackend: Created" << (mpi_ctx_ ? " with MPI context (world_size=" + std::to_string(mpi_ctx_->world_size()) + ")" : ""));
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
        bool success = rccl_backend_detail::hipGetDeviceCountWrapper(&rocm_device_count);
        return (success && rocm_device_count > 0);
#else
        return false;
#endif
    }

#ifdef HAVE_RCCL

    // =========================================================================
    // Type Conversion Helpers
    // =========================================================================

    int RCCLBackend::toRcclDataTypeInt(CollectiveDataType dtype)
    {
        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
            return 0;
        case CollectiveDataType::FLOAT16:
            return 1;
        case CollectiveDataType::BFLOAT16:
            return 2;
        case CollectiveDataType::INT32:
            return 3;
        case CollectiveDataType::INT8:
            return 4;
        default:
            return 0;
        }
    }

    int RCCLBackend::toRcclRedOpInt(CollectiveOp op)
    {
        switch (op)
        {
        case CollectiveOp::ALLREDUCE_SUM:
            return 0;
        case CollectiveOp::ALLREDUCE_MAX:
            return 3;
        case CollectiveOp::ALLREDUCE_MIN:
            return 2;
        default:
            return 0;
        }
    }

#endif // HAVE_RCCL

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

        // Determine if this is multi-process (MPI) or single-process
        const bool is_multi_process = mpi_ctx_ && mpi_ctx_->world_size() > 1;

        if (is_multi_process)
        {
            // Multi-process mode: use MPI world size/rank
            num_ranks_ = mpi_ctx_->world_size();
            local_rank_ = mpi_ctx_->rank();
        }
        else
        {
            // Single-process mode: use DeviceGroup size/rank
            num_ranks_ = static_cast<int>(group.size());
            local_rank_ = group.local_rank;
        }

        if (num_ranks_ < 1)
        {
            last_error_ = "RCCLBackend requires at least 1 device";
            LOG_ERROR(last_error_);
            return false;
        }

        // Set the HIP device for this rank
        DeviceId local_device = group.localDevice();
        HIP_WRAPPER_CHECK(rccl_backend_detail::hipSetDeviceOrdinal(local_device.ordinal),
                          "hipSetDevice(" + std::to_string(local_device.ordinal) + ")");

        // Create HIP stream
        HIP_WRAPPER_CHECK(rccl_backend_detail::hipCreateStream(&stream_),
                          "hipStreamCreate");

        // Create RCCL communicator
        if (is_multi_process)
        {
            // Multi-process: coordinate via MPI
            // Rank 0 generates unique ID, broadcasts to all ranks
            std::vector<char> id_buffer(rccl_backend_detail::rcclUniqueIdSize());

            if (mpi_ctx_->rank() == 0)
            {
                if (!rccl_backend_detail::rcclGetUniqueIdWrapper(id_buffer.data()))
                {
                    last_error_ = "rcclGetUniqueId failed";
                    LOG_ERROR(last_error_);
                    rccl_backend_detail::hipDestroyStream(stream_);
                    stream_ = nullptr;
                    return false;
                }
            }

            // Broadcast the unique ID from rank 0 to all other ranks
            int mpi_err = MPI_Bcast(id_buffer.data(), static_cast<int>(id_buffer.size()),
                                    MPI_BYTE, 0, mpi_ctx_->comm());
            if (mpi_err != MPI_SUCCESS)
            {
                last_error_ = "MPI_Bcast of ncclUniqueId failed";
                LOG_ERROR(last_error_);
                rccl_backend_detail::hipDestroyStream(stream_);
                stream_ = nullptr;
                return false;
            }

            // All ranks initialize their communicator with the shared unique ID
            std::string rccl_error;
            if (!rccl_backend_detail::rcclCommInitRankWrapper(&comm_, num_ranks_,
                                                              id_buffer.data(), local_rank_, rccl_error))
            {
                last_error_ = "rcclCommInitRank failed (multi-process): " + rccl_error;
                LOG_ERROR(last_error_);
                rccl_backend_detail::hipDestroyStream(stream_);
                stream_ = nullptr;
                return false;
            }

            LOG_INFO("RCCLBackend: Initialized multi-process with " << num_ranks_
                                                                    << " MPI ranks, local_rank=" << local_rank_
                                                                    << ", device=" << local_device.ordinal);
        }
        else if (num_ranks_ == 1)
        {
            // Single GPU - create a trivial communicator
            std::string rccl_error;
            if (!rccl_backend_detail::rcclCommInitAllWrapper(&comm_, 1, nullptr, rccl_error))
            {
                last_error_ = "rcclCommInitAll failed: " + rccl_error;
                LOG_ERROR(last_error_);
                rccl_backend_detail::hipDestroyStream(stream_);
                stream_ = nullptr;
                return false;
            }
            LOG_INFO("RCCLBackend: Initialized single-GPU mode");
        }
        else
        {
            // Multi-GPU single process (no MPI context)
            // Use RCCLCoordinator which owns all RCCL comms/streams on a dedicated thread
            LOG_DEBUG("RCCLBackend: Single-process multi-GPU mode with " << num_ranks_ << " GPUs (using RCCLCoordinator)");

            is_multi_gpu_single_process_ = true;

            // The coordinator owns its own streams, so the stream_ created above
            // is not needed in multi-GPU mode. Destroy it to avoid orphaned resources.
            if (stream_)
            {
                rccl_backend_detail::hipDestroyStream(stream_);
                stream_ = nullptr;
            }

            // Store device ordinals from the group
            device_ordinals_.clear();
            for (const auto &device : group.devices)
            {
                device_ordinals_.push_back(device.ordinal);
            }

            // Create and initialize the coordinator
            coordinator_ = std::make_unique<RCCLCoordinator>();
            if (!coordinator_->initialize(device_ordinals_))
            {
                last_error_ = "RCCLCoordinator initialization failed: " + coordinator_->lastError();
                LOG_ERROR(last_error_);
                coordinator_.reset();
                return false;
            }

            LOG_INFO("RCCLBackend: Initialized multi-GPU single-process (via RCCLCoordinator) with "
                     << num_ranks_ << " GPU(s), local_rank=" << local_rank_);
        }

        initialized_ = true;
        return true;
#else
        (void)group;
        last_error_ = "RCCL not available (HAVE_RCCL not defined)";
        LOG_ERROR(last_error_);
        return false;
#endif
    }

    void RCCLBackend::setComputeStreams(const std::vector<void *> &compute_streams)
    {
        if (coordinator_)
        {
            coordinator_->setComputeStreams(compute_streams);
        }
    }

    void RCCLBackend::shutdown()
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            return;
        }

        // Clean up multi-GPU single-process resources
        if (is_multi_gpu_single_process_)
        {
            // Coordinator owns all RCCL comms/streams - just shut it down
            if (coordinator_)
            {
                coordinator_->shutdown();
                coordinator_.reset();
            }
            device_ordinals_.clear();
            is_multi_gpu_single_process_ = false;
        }

        // Destroy single-process communicator.
        // Use rcclPrimeAndDestroyComm which performs a trivial allreduce to force
        // RCCL to allocate its internal buffers, then uses ncclCommDestroy for
        // graceful cleanup. This avoids ROCm CLR crashes from both:
        // - ncclCommAbort/Destroy on unused communicators ("Memobj map" error)
        // - ncclCommAbort corrupting ROCm CLR state across repeated cycles
        if (comm_)
        {
            if (stream_)
            {
                rccl_backend_detail::hipSynchronizeStream(stream_);
            }
            rccl_backend_detail::rcclPrimeAndDestroyComm(comm_, stream_);
            comm_ = nullptr;
        }

        if (stream_)
        {
            rccl_backend_detail::hipDestroyStream(stream_);
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

        std::string rccl_error;
        if (!rccl_backend_detail::rcclAllReduceWrapper(
                buffer, buffer, count,
                toRcclDataTypeInt(dtype), toRcclRedOpInt(op),
                comm_, stream_, rccl_error))
        {
            last_error_ = "rcclAllReduce failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)buffer;
        (void)count;
        (void)dtype;
        (void)op;
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

        std::string rccl_error;
        if (!rccl_backend_detail::rcclAllGatherWrapper(
                send_buf, recv_buf, send_count,
                toRcclDataTypeInt(dtype),
                comm_, stream_, rccl_error))
        {
            last_error_ = "rcclAllGather failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)send_buf;
        (void)recv_buf;
        (void)send_count;
        (void)dtype;
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

        // If all counts are equal, use regular allgather
        bool all_equal = true;
        for (size_t i = 1; i < recv_counts.size(); ++i)
        {
            if (recv_counts[i] != recv_counts[0])
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
        std::string rccl_error;
        if (!rccl_backend_detail::rcclGroupStartWrapper(rccl_error))
        {
            last_error_ = "rcclGroupStart failed: " + rccl_error;
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
            if (!rccl_backend_detail::rcclSendWrapper(send_buf, send_count, toRcclDataTypeInt(dtype),
                                                      peer, comm_, stream_, rccl_error))
            {
                rccl_backend_detail::rcclGroupEndWrapper(rccl_error);
                last_error_ = "rcclSend failed in allgatherv: " + rccl_error;
                return false;
            }

            // Receive from peer at their offset
            char *recv_ptr = static_cast<char *>(recv_buf) + displacements[peer] * dtype_size;
            if (!rccl_backend_detail::rcclRecvWrapper(recv_ptr, recv_counts[peer], toRcclDataTypeInt(dtype),
                                                      peer, comm_, stream_, rccl_error))
            {
                rccl_backend_detail::rcclGroupEndWrapper(rccl_error);
                last_error_ = "rcclRecv failed in allgatherv: " + rccl_error;
                return false;
            }
        }

        if (!rccl_backend_detail::rcclGroupEndWrapper(rccl_error))
        {
            last_error_ = "rcclGroupEnd failed: " + rccl_error;
            return false;
        }

        return true;
#else
        (void)send_buf;
        (void)send_count;
        (void)recv_buf;
        (void)recv_counts;
        (void)displacements;
        (void)dtype;
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

        std::string rccl_error;
        if (!rccl_backend_detail::rcclReduceScatterWrapper(
                send_buf, recv_buf, recv_count,
                toRcclDataTypeInt(dtype), toRcclRedOpInt(op),
                comm_, stream_, rccl_error))
        {
            last_error_ = "rcclReduceScatter failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)send_buf;
        (void)recv_buf;
        (void)recv_count;
        (void)dtype;
        (void)op;
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

        std::string rccl_error;
        if (!rccl_backend_detail::rcclBroadcastWrapper(
                buffer, count, toRcclDataTypeInt(dtype), root,
                comm_, stream_, rccl_error))
        {
            last_error_ = "rcclBroadcast failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)buffer;
        (void)count;
        (void)dtype;
        (void)root;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    // =========================================================================
    // Point-to-Point Operations
    // =========================================================================

    bool RCCLBackend::send(void *buffer, size_t count, CollectiveDataType dtype,
                           int peer, int tag)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (peer < 0 || peer >= num_ranks_)
        {
            last_error_ = "send: Invalid peer rank " + std::to_string(peer) +
                          " (valid range: 0-" + std::to_string(num_ranks_ - 1) + ")";
            LOG_ERROR(last_error_);
            return false;
        }

        (void)tag; // RCCL doesn't support message tags

        std::string rccl_error;
        int dtype_int = toRcclDataTypeInt(dtype);

        // RCCL send/recv must be paired - use group for safety
        if (!rccl_backend_detail::rcclGroupStartWrapper(rccl_error))
        {
            last_error_ = "rcclGroupStart failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        if (!rccl_backend_detail::rcclSendWrapper(buffer, count, dtype_int, peer, comm_, stream_, rccl_error))
        {
            rccl_backend_detail::rcclGroupEndWrapper(rccl_error);
            last_error_ = "rcclSend failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        if (!rccl_backend_detail::rcclGroupEndWrapper(rccl_error))
        {
            last_error_ = "rcclGroupEnd failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)buffer;
        (void)count;
        (void)dtype;
        (void)peer;
        (void)tag;
        last_error_ = "RCCL not available";
        LOG_ERROR(last_error_);
        return false;
#endif
    }

    bool RCCLBackend::recv(void *buffer, size_t count, CollectiveDataType dtype,
                           int peer, int tag)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (peer < 0 || peer >= num_ranks_)
        {
            last_error_ = "recv: Invalid peer rank " + std::to_string(peer) +
                          " (valid range: 0-" + std::to_string(num_ranks_ - 1) + ")";
            LOG_ERROR(last_error_);
            return false;
        }

        (void)tag; // RCCL doesn't support message tags

        std::string rccl_error;
        int dtype_int = toRcclDataTypeInt(dtype);

        // RCCL send/recv must be paired - use group for safety
        if (!rccl_backend_detail::rcclGroupStartWrapper(rccl_error))
        {
            last_error_ = "rcclGroupStart failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        if (!rccl_backend_detail::rcclRecvWrapper(buffer, count, dtype_int, peer, comm_, stream_, rccl_error))
        {
            rccl_backend_detail::rcclGroupEndWrapper(rccl_error);
            last_error_ = "rcclRecv failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        if (!rccl_backend_detail::rcclGroupEndWrapper(rccl_error))
        {
            last_error_ = "rcclGroupEnd failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)buffer;
        (void)count;
        (void)dtype;
        (void)peer;
        (void)tag;
        last_error_ = "RCCL not available";
        LOG_ERROR(last_error_);
        return false;
#endif
    }

    bool RCCLBackend::sendrecv(void *sendbuf, void *recvbuf, size_t count,
                               CollectiveDataType dtype, int peer)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (sendbuf == nullptr)
        {
            last_error_ = "sendrecv: send buffer is null";
            LOG_ERROR(last_error_);
            return false;
        }

        if (recvbuf == nullptr)
        {
            last_error_ = "sendrecv: recv buffer is null";
            LOG_ERROR(last_error_);
            return false;
        }

        if (peer < 0 || peer >= num_ranks_)
        {
            last_error_ = "sendrecv: Invalid peer rank " + std::to_string(peer) +
                          " (valid range: 0-" + std::to_string(num_ranks_ - 1) + ")";
            LOG_ERROR(last_error_);
            return false;
        }

        std::string rccl_error;
        int dtype_int = toRcclDataTypeInt(dtype);

        // Use rcclGroupStart/End to issue send and recv together
        if (!rccl_backend_detail::rcclGroupStartWrapper(rccl_error))
        {
            last_error_ = "rcclGroupStart failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        // Send to peer
        if (!rccl_backend_detail::rcclSendWrapper(sendbuf, count, dtype_int, peer, comm_, stream_, rccl_error))
        {
            rccl_backend_detail::rcclGroupEndWrapper(rccl_error);
            last_error_ = "rcclSend failed in sendrecv: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        // Receive from peer
        if (!rccl_backend_detail::rcclRecvWrapper(recvbuf, count, dtype_int, peer, comm_, stream_, rccl_error))
        {
            rccl_backend_detail::rcclGroupEndWrapper(rccl_error);
            last_error_ = "rcclRecv failed in sendrecv: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        if (!rccl_backend_detail::rcclGroupEndWrapper(rccl_error))
        {
            last_error_ = "rcclGroupEnd failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)sendbuf;
        (void)recvbuf;
        (void)count;
        (void)dtype;
        (void)peer;
        last_error_ = "RCCL not available";
        LOG_ERROR(last_error_);
        return false;
#endif
    }

    // =========================================================================
    // Async Point-to-Point Operations
    // =========================================================================

    bool RCCLBackend::sendAsync(void *buffer, size_t count, CollectiveDataType dtype,
                                int peer, void *stream, int tag)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (peer < 0 || peer >= num_ranks_)
        {
            last_error_ = "sendAsync: Invalid peer rank " + std::to_string(peer) +
                          " (valid range: 0-" + std::to_string(num_ranks_ - 1) + ")";
            LOG_ERROR(last_error_);
            return false;
        }

        (void)tag; // RCCL doesn't support message tags

        std::string rccl_error;
        int dtype_int = toRcclDataTypeInt(dtype);

        // Use the caller-provided stream for async operation
        void *target_stream = stream ? stream : stream_;

        if (!rccl_backend_detail::rcclGroupStartWrapper(rccl_error))
        {
            last_error_ = "rcclGroupStart failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        if (!rccl_backend_detail::rcclSendWrapper(buffer, count, dtype_int, peer, comm_, target_stream, rccl_error))
        {
            rccl_backend_detail::rcclGroupEndWrapper(rccl_error);
            last_error_ = "rcclSend failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        if (!rccl_backend_detail::rcclGroupEndWrapper(rccl_error))
        {
            last_error_ = "rcclGroupEnd failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)buffer;
        (void)count;
        (void)dtype;
        (void)peer;
        (void)stream;
        (void)tag;
        last_error_ = "RCCL not available";
        LOG_ERROR(last_error_);
        return false;
#endif
    }

    bool RCCLBackend::recvAsync(void *buffer, size_t count, CollectiveDataType dtype,
                                int peer, void *stream, int tag)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (peer < 0 || peer >= num_ranks_)
        {
            last_error_ = "recvAsync: Invalid peer rank " + std::to_string(peer) +
                          " (valid range: 0-" + std::to_string(num_ranks_ - 1) + ")";
            LOG_ERROR(last_error_);
            return false;
        }

        (void)tag; // RCCL doesn't support message tags

        std::string rccl_error;
        int dtype_int = toRcclDataTypeInt(dtype);

        // Use the caller-provided stream for async operation
        void *target_stream = stream ? stream : stream_;

        if (!rccl_backend_detail::rcclGroupStartWrapper(rccl_error))
        {
            last_error_ = "rcclGroupStart failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        if (!rccl_backend_detail::rcclRecvWrapper(buffer, count, dtype_int, peer, comm_, target_stream, rccl_error))
        {
            rccl_backend_detail::rcclGroupEndWrapper(rccl_error);
            last_error_ = "rcclRecv failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        if (!rccl_backend_detail::rcclGroupEndWrapper(rccl_error))
        {
            last_error_ = "rcclGroupEnd failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)buffer;
        (void)count;
        (void)dtype;
        (void)peer;
        (void)stream;
        (void)tag;
        last_error_ = "RCCL not available";
        LOG_ERROR(last_error_);
        return false;
#endif
    }

    bool RCCLBackend::sendrecvAsync(void *sendbuf, void *recvbuf, size_t count,
                                    CollectiveDataType dtype, int peer, void *stream)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (peer < 0 || peer >= num_ranks_)
        {
            last_error_ = "sendrecvAsync: Invalid peer rank " + std::to_string(peer) +
                          " (valid range: 0-" + std::to_string(num_ranks_ - 1) + ")";
            LOG_ERROR(last_error_);
            return false;
        }

        std::string rccl_error;
        int dtype_int = toRcclDataTypeInt(dtype);

        // Use the caller-provided stream for async operation
        void *target_stream = stream ? stream : stream_;

        if (!rccl_backend_detail::rcclGroupStartWrapper(rccl_error))
        {
            last_error_ = "rcclGroupStart failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        // Send to peer
        if (!rccl_backend_detail::rcclSendWrapper(sendbuf, count, dtype_int, peer, comm_, target_stream, rccl_error))
        {
            rccl_backend_detail::rcclGroupEndWrapper(rccl_error);
            last_error_ = "rcclSend failed in sendrecvAsync: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        // Receive from peer
        if (!rccl_backend_detail::rcclRecvWrapper(recvbuf, count, dtype_int, peer, comm_, target_stream, rccl_error))
        {
            rccl_backend_detail::rcclGroupEndWrapper(rccl_error);
            last_error_ = "rcclRecv failed in sendrecvAsync: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        if (!rccl_backend_detail::rcclGroupEndWrapper(rccl_error))
        {
            last_error_ = "rcclGroupEnd failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)sendbuf;
        (void)recvbuf;
        (void)count;
        (void)dtype;
        (void)peer;
        (void)stream;
        last_error_ = "RCCL not available";
        LOG_ERROR(last_error_);
        return false;
#endif
    }

    // =========================================================================
    // Multi-GPU Single-Process Collective Operations
    // =========================================================================

    bool RCCLBackend::isMultiGpuSingleProcess() const
    {
        return is_multi_gpu_single_process_;
    }

    bool RCCLBackend::allreduceMulti(
        const std::vector<void *> &buffers,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (!is_multi_gpu_single_process_)
        {
            last_error_ = "allreduceMulti requires multi-GPU single-process mode";
            LOG_ERROR(last_error_);
            return false;
        }

        if (buffers.size() != static_cast<size_t>(num_ranks_))
        {
            last_error_ = "Buffer count (" + std::to_string(buffers.size()) +
                          ") doesn't match GPU count (" + std::to_string(num_ranks_) + ")";
            LOG_ERROR(last_error_);
            return false;
        }

        // Delegate to coordinator
        if (!coordinator_)
        {
            last_error_ = "RCCLCoordinator not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (!coordinator_->allreduceMulti(buffers, count, dtype, op))
        {
            last_error_ = "RCCLCoordinator allreduceMulti failed: " + coordinator_->lastError();
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)buffers;
        (void)count;
        (void)dtype;
        (void)op;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLBackend::allreduceMultiAndSynchronize(
        const std::vector<void *> &buffers,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (!is_multi_gpu_single_process_)
        {
            last_error_ = "allreduceMultiAndSynchronize requires multi-GPU single-process mode";
            LOG_ERROR(last_error_);
            return false;
        }

        if (buffers.size() != static_cast<size_t>(num_ranks_))
        {
            last_error_ = "Buffer count (" + std::to_string(buffers.size()) +
                          ") doesn't match GPU count (" + std::to_string(num_ranks_) + ")";
            LOG_ERROR(last_error_);
            return false;
        }

        if (!coordinator_)
        {
            last_error_ = "RCCLCoordinator not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (!coordinator_->allreduceMultiAndSynchronize(buffers, count, dtype, op))
        {
            last_error_ = "RCCLCoordinator allreduceMultiAndSynchronize failed: " + coordinator_->lastError();
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)buffers;
        (void)count;
        (void)dtype;
        (void)op;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLBackend::allgatherMulti(
        const std::vector<const void *> &send_bufs,
        const std::vector<void *> &recv_bufs,
        size_t send_count,
        CollectiveDataType dtype)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (!is_multi_gpu_single_process_)
        {
            last_error_ = "allgatherMulti requires multi-GPU single-process mode";
            LOG_ERROR(last_error_);
            return false;
        }

        if (send_bufs.size() != static_cast<size_t>(num_ranks_) ||
            recv_bufs.size() != static_cast<size_t>(num_ranks_))
        {
            last_error_ = "Buffer count doesn't match GPU count";
            LOG_ERROR(last_error_);
            return false;
        }

        // Delegate to coordinator
        if (!coordinator_)
        {
            last_error_ = "RCCLCoordinator not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (!coordinator_->allgatherMulti(send_bufs, recv_bufs, send_count, dtype))
        {
            last_error_ = "RCCLCoordinator allgatherMulti failed: " + coordinator_->lastError();
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)send_bufs;
        (void)recv_bufs;
        (void)send_count;
        (void)dtype;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLBackend::broadcastMulti(
        const std::vector<void *> &buffers,
        size_t count,
        CollectiveDataType dtype,
        int root)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (!is_multi_gpu_single_process_)
        {
            last_error_ = "broadcastMulti requires multi-GPU single-process mode";
            LOG_ERROR(last_error_);
            return false;
        }

        if (buffers.size() != static_cast<size_t>(num_ranks_))
        {
            last_error_ = "Buffer count (" + std::to_string(buffers.size()) +
                          ") doesn't match GPU count (" + std::to_string(num_ranks_) + ")";
            LOG_ERROR(last_error_);
            return false;
        }

        if (root < 0 || root >= num_ranks_)
        {
            last_error_ = "Invalid root rank: " + std::to_string(root);
            LOG_ERROR(last_error_);
            return false;
        }

        // Delegate to coordinator
        if (!coordinator_)
        {
            last_error_ = "RCCLCoordinator not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (!coordinator_->broadcastMulti(buffers, count, dtype, root))
        {
            last_error_ = "RCCLCoordinator broadcastMulti failed: " + coordinator_->lastError();
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)buffers;
        (void)count;
        (void)dtype;
        (void)root;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLBackend::reduceMulti(
        const std::vector<void *> &buffers,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op,
        int root)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (!is_multi_gpu_single_process_)
        {
            last_error_ = "reduceMulti requires multi-GPU single-process mode";
            LOG_ERROR(last_error_);
            return false;
        }

        // reduceMulti not yet implemented in RCCLCoordinator
        // TODO: Add reduceMulti to RCCLCoordinator when needed
        (void)buffers;
        (void)count;
        (void)dtype;
        (void)op;
        (void)root;
        last_error_ = "reduceMulti not yet implemented in coordinator mode";
        LOG_ERROR(last_error_);
        return false;
#else
        (void)buffers;
        (void)count;
        (void)dtype;
        (void)op;
        (void)root;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLBackend::reduceScatterMulti(
        const std::vector<const void *> &send_buffers,
        const std::vector<void *> &recv_buffers,
        size_t recv_count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (!is_multi_gpu_single_process_)
        {
            last_error_ = "reduceScatterMulti requires multi-GPU single-process mode";
            LOG_ERROR(last_error_);
            return false;
        }

        if (send_buffers.size() != static_cast<size_t>(num_ranks_) ||
            recv_buffers.size() != static_cast<size_t>(num_ranks_))
        {
            last_error_ = "Buffer count mismatch: send=" + std::to_string(send_buffers.size()) +
                          ", recv=" + std::to_string(recv_buffers.size()) +
                          ", expected=" + std::to_string(num_ranks_);
            LOG_ERROR(last_error_);
            return false;
        }

        // Delegate to coordinator
        if (!coordinator_)
        {
            last_error_ = "RCCLCoordinator not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (!coordinator_->reduceScatterMulti(send_buffers, recv_buffers, recv_count, dtype, op))
        {
            last_error_ = "RCCLCoordinator reduceScatterMulti failed: " + coordinator_->lastError();
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)send_buffers;
        (void)recv_buffers;
        (void)recv_count;
        (void)dtype;
        (void)op;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLBackend::sendrecvMulti(
        void *src_buffer,
        void *dst_buffer,
        size_t count,
        CollectiveDataType dtype,
        int src_gpu,
        int dst_gpu)
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            last_error_ = "RCCLBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (!is_multi_gpu_single_process_)
        {
            last_error_ = "sendrecvMulti requires multi-GPU single-process mode";
            LOG_ERROR(last_error_);
            return false;
        }

        if (src_gpu == dst_gpu)
        {
            // Self-transfer: just log and succeed - caller shouldn't do this
            LOG_DEBUG("sendrecvMulti: src_gpu == dst_gpu (" << src_gpu << "), treating as no-op");
            return true;
        }

        // sendrecvMulti not yet implemented in RCCLCoordinator
        // TODO: Add sendrecvMulti to RCCLCoordinator when needed
        (void)src_buffer;
        (void)dst_buffer;
        (void)count;
        (void)dtype;
        (void)src_gpu;
        (void)dst_gpu;
        last_error_ = "sendrecvMulti not yet implemented in coordinator mode";
        LOG_ERROR(last_error_);
        return false;
#else
        (void)src_buffer;
        (void)dst_buffer;
        (void)count;
        (void)dtype;
        (void)src_gpu;
        (void)dst_gpu;
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

        // In multi-GPU single-process mode, delegate to coordinator
        if (is_multi_gpu_single_process_)
        {
            if (!coordinator_)
            {
                last_error_ = "RCCLCoordinator not initialized";
                LOG_ERROR(last_error_);
                return false;
            }

            if (!coordinator_->synchronize())
            {
                last_error_ = "RCCLCoordinator synchronize failed: " + coordinator_->lastError();
                LOG_ERROR(last_error_);
                return false;
            }
            return true;
        }

        // Single-GPU or MPI mode: just sync the main stream
        if (!rccl_backend_detail::hipSynchronizeStream(stream_))
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
    // Data Copy Operations
    // =========================================================================

    bool RCCLBackend::copy(void *dst_ptr, DeviceId dst_device,
                           const void *src_ptr, DeviceId src_device,
                           size_t bytes)
    {
#ifdef HAVE_RCCL
        // RCCLBackend only supports ROCm↔ROCm copies
        if (!src_device.is_rocm() || !dst_device.is_rocm())
        {
            LOG_DEBUG("RCCLBackend::copy: requires ROCm devices, got "
                      << src_device.toString() << " -> " << dst_device.toString());
            return false;
        }

        if (bytes == 0)
            return true;
        if (!dst_ptr || !src_ptr)
            return false;

        int src_idx = src_device.toKernelDeviceIndex();
        int dst_idx = dst_device.toKernelDeviceIndex();

        // Require coordinator for all copies (including same-device for consistency)
        if (!coordinator_)
        {
            last_error_ = "RCCLBackend::copy: coordinator not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        // Delegate to coordinator (handles both same-device and cross-device)
        if (!coordinator_->copy(dst_ptr, dst_idx, src_ptr, src_idx, bytes))
        {
            last_error_ = "RCCLBackend::copy: " + coordinator_->lastError();
            LOG_ERROR(last_error_);
            return false;
        }
        return true;
#else
        (void)dst_ptr;
        (void)dst_device;
        (void)src_ptr;
        (void)src_device;
        (void)bytes;
        return false;
#endif
    }

    bool RCCLBackend::copyAsync(void *dst_ptr, DeviceId dst_device,
                                const void *src_ptr, DeviceId src_device,
                                size_t bytes, void *stream)
    {
#ifdef HAVE_RCCL
        (void)stream; // Coordinator manages its own streams

        // RCCLBackend only supports ROCm↔ROCm copies
        if (!src_device.is_rocm() || !dst_device.is_rocm())
        {
            LOG_DEBUG("RCCLBackend::copyAsync: requires ROCm devices, got "
                      << src_device.toString() << " -> " << dst_device.toString());
            return false;
        }

        if (bytes == 0)
            return true;
        if (!dst_ptr || !src_ptr)
            return false;

        int src_idx = src_device.toKernelDeviceIndex();
        int dst_idx = dst_device.toKernelDeviceIndex();

        // Require coordinator for all copies
        if (!coordinator_)
        {
            last_error_ = "RCCLBackend::copyAsync: coordinator not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        // Delegate to coordinator async copy (enqueues work, returns immediately)
        // Caller should use coordinator_->getCompletionEvent(dst_idx) to synchronize
        if (!coordinator_->copyAsync(dst_ptr, dst_idx, src_ptr, src_idx, bytes))
        {
            last_error_ = "RCCLBackend::copyAsync: " + coordinator_->lastError();
            LOG_ERROR(last_error_);
            return false;
        }
        return true;
#else
        (void)dst_ptr;
        (void)dst_device;
        (void)src_ptr;
        (void)src_device;
        (void)bytes;
        (void)stream;
        return false;
#endif
    }

    bool RCCLBackend::supportsCopy(DeviceId src_device, DeviceId dst_device) const
    {
#ifdef HAVE_RCCL
        if (!src_device.is_rocm() || !dst_device.is_rocm())
        {
            return false;
        }

        // Same-device copy is always supported via hipMemcpy (no coordinator needed)
        if (src_device == dst_device)
        {
            return true;
        }

        // Cross-device copies require coordinator (RCCL send/recv or xGMI)
        // Coordinator handles: cross-device via best available transport (xGMI, P2P, or host staging)
        return coordinator_ != nullptr;
#else
        (void)src_device;
        (void)dst_device;
        return false;
#endif
    }

} // namespace llaminar2
