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
#include "../../utils/Logger.h"

#ifdef HAVE_RCCL
#include <mpi.h>
#include <atomic>
#include <thread>
#include <string>
#include <cstring>

// Forward declarations for HIP and RCCL wrappers (implemented in RCCLBackendHIP.cpp)
namespace llaminar2
{
    namespace rccl_backend_detail
    {
        // HIP runtime wrappers
        bool hipSetDeviceOrdinal(int device_ordinal);
        bool hipGetDeviceCountWrapper(int *count);
        bool hipCreateStream(void **stream_ptr);
        bool hipDestroyStream(void *stream);
        bool hipSynchronizeStream(void *stream);
        std::string hipGetLastErrorString();
        std::string hipErrorToString(int error_code);

        // RCCL unique ID
        size_t rcclUniqueIdSize();
        bool rcclGetUniqueIdWrapper(void *id_out);

        // RCCL communicator management
        bool rcclCommInitRankWrapper(void **comm_out, int nranks, void *unique_id, int rank, std::string &error_out);
        bool rcclCommInitAllWrapper(void **comms_out, int ndevs, const int *devlist, std::string &error_out);
        void rcclCommDestroyWrapper(void *comm);

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

        // Point-to-point operations (for allgatherv emulation)
        bool rcclSendWrapper(const void *sendbuff, size_t count, int dtype_int, int peer,
                             void *comm, void *stream, std::string &error_out);
        bool rcclRecvWrapper(void *recvbuff, size_t count, int dtype_int, int peer,
                             void *comm, void *stream, std::string &error_out);
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
            // Use threaded ncclCommInitRank - each GPU gets its own thread
            std::vector<char> id_buffer(rccl_backend_detail::rcclUniqueIdSize());
            if (!rccl_backend_detail::rcclGetUniqueIdWrapper(id_buffer.data()))
            {
                last_error_ = "rcclGetUniqueId failed";
                LOG_ERROR(last_error_);
                rccl_backend_detail::hipDestroyStream(stream_);
                stream_ = nullptr;
                return false;
            }

            // Allocate arrays for per-GPU resources
            all_comms_.resize(num_ranks_, nullptr);
            all_streams_.resize(num_ranks_, nullptr);
            device_ordinals_.resize(num_ranks_);
            std::atomic<int> error_count{0};
            std::vector<std::string> thread_errors(num_ranks_);

            // Build device list from group
            for (int i = 0; i < num_ranks_; ++i)
            {
                device_ordinals_[i] = group.devices[i].ordinal;
            }

            // Launch threads - each thread initializes one GPU's communicator and stream
            std::vector<std::thread> threads;
            threads.reserve(num_ranks_);

            for (int rank = 0; rank < num_ranks_; ++rank)
            {
                threads.emplace_back([this, rank, &id_buffer, &error_count, &thread_errors]()
                                     {
                    // Set device for this thread
                    if (!rccl_backend_detail::hipSetDeviceOrdinal(device_ordinals_[rank])) {
                        thread_errors[rank] = "hipSetDevice failed for rank " + std::to_string(rank);
                        error_count.fetch_add(1, std::memory_order_relaxed);
                        return;
                    }

                    // Create stream for this GPU
                    if (!rccl_backend_detail::hipCreateStream(&all_streams_[rank])) {
                        thread_errors[rank] = "hipStreamCreate failed for rank " + std::to_string(rank);
                        error_count.fetch_add(1, std::memory_order_relaxed);
                        return;
                    }

                    // Initialize communicator for this rank
                    std::string rccl_error;
                    if (!rccl_backend_detail::rcclCommInitRankWrapper(&all_comms_[rank], num_ranks_,
                                                                       const_cast<char*>(id_buffer.data()), rank, rccl_error)) {
                        thread_errors[rank] = "rcclCommInitRank failed for rank " + std::to_string(rank) + ": " + rccl_error;
                        error_count.fetch_add(1, std::memory_order_relaxed);
                        rccl_backend_detail::hipDestroyStream(all_streams_[rank]);
                        all_streams_[rank] = nullptr;
                        return;
                    } });
            }

            // Wait for all threads to complete
            for (auto &t : threads)
            {
                t.join();
            }

            // Check for errors
            if (error_count.load() > 0)
            {
                std::string combined_errors;
                for (int i = 0; i < num_ranks_; ++i)
                {
                    if (!thread_errors[i].empty())
                    {
                        if (!combined_errors.empty())
                            combined_errors += "; ";
                        combined_errors += thread_errors[i];
                    }
                }
                last_error_ = "Threaded rcclCommInitRank failed: " + combined_errors;
                LOG_ERROR(last_error_);

                // Cleanup any successfully created resources
                for (int i = 0; i < num_ranks_; ++i)
                {
                    if (all_comms_[i])
                    {
                        rccl_backend_detail::rcclCommDestroyWrapper(all_comms_[i]);
                        all_comms_[i] = nullptr;
                    }
                    if (all_streams_[i])
                    {
                        rccl_backend_detail::hipDestroyStream(all_streams_[i]);
                        all_streams_[i] = nullptr;
                    }
                }
                all_comms_.clear();
                all_streams_.clear();
                device_ordinals_.clear();
                rccl_backend_detail::hipDestroyStream(stream_);
                stream_ = nullptr;
                return false;
            }

            // Store the communicator and stream for local_rank
            comm_ = all_comms_[local_rank_];
            is_multi_gpu_single_process_ = true;

            LOG_INFO("RCCLBackend: Initialized multi-GPU single-process (threaded rcclCommInitRank) with "
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

    void RCCLBackend::shutdown()
    {
#ifdef HAVE_RCCL
        if (!initialized_)
        {
            return;
        }

        // If we used multi-GPU single-process mode, destroy all resources
        if (!all_comms_.empty())
        {
            for (size_t i = 0; i < all_comms_.size(); ++i)
            {
                if (all_comms_[i])
                {
                    rccl_backend_detail::rcclCommDestroyWrapper(all_comms_[i]);
                }
                if (i < all_streams_.size() && all_streams_[i])
                {
                    rccl_backend_detail::hipDestroyStream(all_streams_[i]);
                }
            }
            all_comms_.clear();
            all_streams_.clear();
            device_ordinals_.clear();
            comm_ = nullptr;
        }
        else if (comm_)
        {
            rccl_backend_detail::rcclCommDestroyWrapper(comm_);
            comm_ = nullptr;
        }

        if (stream_)
        {
            rccl_backend_detail::hipDestroyStream(stream_);
            stream_ = nullptr;
        }

        is_multi_gpu_single_process_ = false;
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
            return false;
        }

        if (!is_multi_gpu_single_process_)
        {
            last_error_ = "allreduceMulti requires multi-GPU single-process mode";
            return false;
        }

        if (buffers.size() != static_cast<size_t>(num_ranks_))
        {
            last_error_ = "Buffer count (" + std::to_string(buffers.size()) +
                          ") doesn't match GPU count (" + std::to_string(num_ranks_) + ")";
            return false;
        }

        std::string rccl_error;
        if (!rccl_backend_detail::rcclGroupStartWrapper(rccl_error))
        {
            last_error_ = "rcclGroupStart failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        // Issue AllReduce on each GPU with its buffer, communicator, and stream
        for (int i = 0; i < num_ranks_; ++i)
        {
            // Set device context
            rccl_backend_detail::hipSetDeviceOrdinal(device_ordinals_[i]);

            if (!rccl_backend_detail::rcclAllReduceWrapper(
                    buffers[i], buffers[i], count,
                    toRcclDataTypeInt(dtype), toRcclRedOpInt(op),
                    all_comms_[i], all_streams_[i], rccl_error))
            {
                rccl_backend_detail::rcclGroupEndWrapper(rccl_error);
                last_error_ = "rcclAllReduce failed on GPU " + std::to_string(i) + ": " + rccl_error;
                LOG_ERROR(last_error_);
                return false;
            }
        }

        if (!rccl_backend_detail::rcclGroupEndWrapper(rccl_error))
        {
            last_error_ = "rcclGroupEnd failed: " + rccl_error;
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
            return false;
        }

        if (!is_multi_gpu_single_process_)
        {
            last_error_ = "allgatherMulti requires multi-GPU single-process mode";
            return false;
        }

        if (send_bufs.size() != static_cast<size_t>(num_ranks_) ||
            recv_bufs.size() != static_cast<size_t>(num_ranks_))
        {
            last_error_ = "Buffer count doesn't match GPU count";
            return false;
        }

        std::string rccl_error;
        if (!rccl_backend_detail::rcclGroupStartWrapper(rccl_error))
        {
            last_error_ = "rcclGroupStart failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        for (int i = 0; i < num_ranks_; ++i)
        {
            rccl_backend_detail::hipSetDeviceOrdinal(device_ordinals_[i]);

            if (!rccl_backend_detail::rcclAllGatherWrapper(
                    send_bufs[i], recv_bufs[i], send_count,
                    toRcclDataTypeInt(dtype),
                    all_comms_[i], all_streams_[i], rccl_error))
            {
                rccl_backend_detail::rcclGroupEndWrapper(rccl_error);
                last_error_ = "rcclAllGather failed on GPU " + std::to_string(i) + ": " + rccl_error;
                LOG_ERROR(last_error_);
                return false;
            }
        }

        if (!rccl_backend_detail::rcclGroupEndWrapper(rccl_error))
        {
            last_error_ = "rcclGroupEnd failed: " + rccl_error;
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
            return false;
        }

        if (!is_multi_gpu_single_process_)
        {
            last_error_ = "broadcastMulti requires multi-GPU single-process mode";
            return false;
        }

        if (buffers.size() != static_cast<size_t>(num_ranks_))
        {
            last_error_ = "Buffer count doesn't match GPU count";
            return false;
        }

        if (root < 0 || root >= num_ranks_)
        {
            last_error_ = "Invalid root rank: " + std::to_string(root);
            return false;
        }

        std::string rccl_error;
        if (!rccl_backend_detail::rcclGroupStartWrapper(rccl_error))
        {
            last_error_ = "rcclGroupStart failed: " + rccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        for (int i = 0; i < num_ranks_; ++i)
        {
            rccl_backend_detail::hipSetDeviceOrdinal(device_ordinals_[i]);

            if (!rccl_backend_detail::rcclBroadcastWrapper(
                    buffers[i], count, toRcclDataTypeInt(dtype), root,
                    all_comms_[i], all_streams_[i], rccl_error))
            {
                rccl_backend_detail::rcclGroupEndWrapper(rccl_error);
                last_error_ = "rcclBroadcast failed on GPU " + std::to_string(i) + ": " + rccl_error;
                LOG_ERROR(last_error_);
                return false;
            }
        }

        if (!rccl_backend_detail::rcclGroupEndWrapper(rccl_error))
        {
            last_error_ = "rcclGroupEnd failed: " + rccl_error;
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

        // In multi-GPU single-process mode, synchronize ALL streams
        if (is_multi_gpu_single_process_)
        {
            for (int i = 0; i < num_ranks_; ++i)
            {
                rccl_backend_detail::hipSetDeviceOrdinal(device_ordinals_[i]);
                if (!rccl_backend_detail::hipSynchronizeStream(all_streams_[i]))
                {
                    last_error_ = "hipStreamSynchronize failed on GPU " + std::to_string(i);
                    LOG_ERROR(last_error_);
                    return false;
                }
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

} // namespace llaminar2
