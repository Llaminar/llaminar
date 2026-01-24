/**
 * @file NCCLBackend.cpp
 * @brief NCCL-based collective backend implementation
 *
 * All CUDA runtime and NCCL API calls are isolated in NCCLBackendCUDA.cu to avoid
 * conflicts with HIP headers when building with both CUDA and ROCm support.
 * This file only uses wrapper functions from the nccl_cuda_wrappers namespace.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "NCCLBackend.h"
#include "../../utils/Logger.h"

#ifdef HAVE_NCCL
#include <mpi.h>
#include <atomic>
#include <thread>
#include <string>
#include <cstring>

// Forward declarations for CUDA and NCCL wrappers (implemented in NCCLBackendCUDA.cu)
namespace llaminar2
{
    namespace nccl_backend_detail
    {
        // CUDA runtime wrappers
        bool cudaSetDeviceOrdinal(int device_ordinal);
        bool cudaGetDeviceCountWrapper(int *count);
        bool cudaCreateStream(void **stream_ptr);
        bool cudaDestroyStream(void *stream);
        bool cudaSynchronizeStream(void *stream);
        std::string cudaGetLastErrorString();

        // NCCL unique ID
        size_t ncclUniqueIdSize();
        bool ncclGetUniqueIdWrapper(void *id_out);

        // NCCL communicator management
        bool ncclCommInitRankWrapper(void **comm_out, int nranks, void *unique_id, int rank, std::string &error_out);
        bool ncclCommInitAllWrapper(void **comms_out, int ndevs, const int *devlist, std::string &error_out);
        void ncclCommDestroyWrapper(void *comm);

        // NCCL collective operations
        bool ncclAllReduceWrapper(void *sendbuff, void *recvbuff, size_t count,
                                  int dtype_int, int op_int, void *comm, void *stream,
                                  std::string &error_out);
        bool ncclAllGatherWrapper(const void *sendbuff, void *recvbuff, size_t sendcount,
                                  int dtype_int, void *comm, void *stream,
                                  std::string &error_out);
        bool ncclBroadcastWrapper(void *buff, size_t count, int dtype_int, int root,
                                  void *comm, void *stream, std::string &error_out);
        bool ncclReduceScatterWrapper(const void *sendbuff, void *recvbuff, size_t recvcount,
                                      int dtype_int, int op_int, void *comm, void *stream,
                                      std::string &error_out);

        // NCCL group operations (for multi-GPU single process)
        bool ncclGroupStartWrapper(std::string &error_out);
        bool ncclGroupEndWrapper(std::string &error_out);
        bool ncclAllReduceInGroupWrapper(void *sendbuff, void *recvbuff, size_t count,
                                         int dtype_int, int op_int, void *comm, void *stream,
                                         std::string &error_out);
        bool ncclAllGatherInGroupWrapper(const void *sendbuff, void *recvbuff, size_t sendcount,
                                         int dtype_int, void *comm, void *stream,
                                         std::string &error_out);
        bool ncclBroadcastInGroupWrapper(void *buff, size_t count, int dtype_int, int root,
                                         void *comm, void *stream, std::string &error_out);

        // Point-to-point operations (for allgatherv emulation)
        bool ncclSendWrapper(const void *sendbuff, size_t count, int dtype_int, int peer,
                             void *comm, void *stream, std::string &error_out);
        bool ncclRecvWrapper(void *recvbuff, size_t count, int dtype_int, int peer,
                             void *comm, void *stream, std::string &error_out);

        // Strided deinterleave kernel (for column-parallel AllGather)
        bool launchDeinterleaveKernel(const void *input, void *output,
                                      int seq_len, int local_dim, int world_size,
                                      void *stream);

        // Temporary buffer allocation
        bool cudaAllocTempBuffer(void **ptr, size_t bytes);
        void cudaFreeTempBuffer(void *ptr);
    } // namespace nccl_backend_detail
} // namespace llaminar2

// Helper macro for CUDA wrapper error checking
#define CUDA_WRAPPER_CHECK(cmd, msg)                                                                      \
    do                                                                                                    \
    {                                                                                                     \
        if (!(cmd))                                                                                       \
        {                                                                                                 \
            last_error_ = std::string(msg) + " failed: " + nccl_backend_detail::cudaGetLastErrorString(); \
            LOG_ERROR(last_error_);                                                                       \
            return false;                                                                                 \
        }                                                                                                 \
    } while (0)

// Convert CollectiveDataType to int for wrapper functions
static int toNcclDataTypeInt(llaminar2::CollectiveDataType dtype)
{
    switch (dtype)
    {
    case llaminar2::CollectiveDataType::FLOAT32:
        return 0;
    case llaminar2::CollectiveDataType::FLOAT16:
        return 1;
    case llaminar2::CollectiveDataType::BFLOAT16:
        return 2;
    case llaminar2::CollectiveDataType::INT32:
        return 3;
    case llaminar2::CollectiveDataType::INT8:
        return 4; // Maps to ncclInt8
    default:
        return 0;
    }
}

// Convert CollectiveOp to int for wrapper functions
static int toNcclRedOpInt(llaminar2::CollectiveOp op)
{
    switch (op)
    {
    case llaminar2::CollectiveOp::ALLREDUCE_SUM:
        return 0; // ncclSum
    case llaminar2::CollectiveOp::REDUCE_SCATTER:
        return 0; // Default to sum for reduce-scatter
    // Note: ALLREDUCE_MAX and ALLREDUCE_MIN map to ncclMax and ncclMin
    case llaminar2::CollectiveOp::ALLREDUCE_MAX:
        return 3; // ncclMax
    case llaminar2::CollectiveOp::ALLREDUCE_MIN:
        return 2; // ncclMin
    default:
        return 0; // Default to sum
    }
}

#endif

namespace llaminar2
{

    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    NCCLBackend::NCCLBackend(std::shared_ptr<MPIContext> mpi_ctx)
        : mpi_ctx_(std::move(mpi_ctx))
    {
        LOG_DEBUG("NCCLBackend: Created" << (mpi_ctx_ ? " with MPI context (world_size=" + std::to_string(mpi_ctx_->world_size()) + ")" : " (single-process mode)"));
    }

    NCCLBackend::~NCCLBackend()
    {
        if (initialized_)
        {
            shutdown();
        }
    }

    // =========================================================================
    // Availability Check
    // =========================================================================

    bool NCCLBackend::isAvailable() const
    {
#ifdef HAVE_NCCL
        int cuda_device_count = 0;
        if (!nccl_backend_detail::cudaGetDeviceCountWrapper(&cuda_device_count))
        {
            return false;
        }
        return (cuda_device_count > 0);
#else
        return false;
#endif
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    bool NCCLBackend::initialize(const DeviceGroup &group)
    {
#ifdef HAVE_NCCL
        if (initialized_)
        {
            LOG_WARN("NCCLBackend::initialize: Already initialized, shutting down first");
            shutdown();
        }

        // Validate all devices are CUDA
        for (const auto &device : group.devices)
        {
            if (device.type != DeviceType::CUDA)
            {
                last_error_ = "NCCLBackend only supports CUDA devices";
                LOG_ERROR(last_error_);
                return false;
            }
        }

        num_ranks_ = static_cast<int>(group.size());
        local_rank_ = group.local_rank;

        if (num_ranks_ < 1)
        {
            last_error_ = "NCCLBackend requires at least 1 device";
            LOG_ERROR(last_error_);
            return false;
        }

        // Set the CUDA device for this rank BEFORE creating stream/communicator
        DeviceId local_device = group.localDevice();
        CUDA_WRAPPER_CHECK(nccl_backend_detail::cudaSetDeviceOrdinal(local_device.ordinal), "cudaSetDevice");
        LOG_DEBUG("NCCLBackend: Set CUDA device to " << local_device.ordinal);

        // Create CUDA stream on the selected device
        void *stream_ptr = nullptr;
        CUDA_WRAPPER_CHECK(nccl_backend_detail::cudaCreateStream(&stream_ptr), "cudaStreamCreate");
        stream_ = stream_ptr;

        // Determine if we're in multi-process (MPI) mode
        const bool is_multi_process = mpi_ctx_ && mpi_ctx_->world_size() > 1;

        // Allocate buffer for NCCL unique ID (opaque structure, ~128 bytes)
        const size_t unique_id_size = nccl_backend_detail::ncclUniqueIdSize();
        std::vector<char> unique_id_buffer(unique_id_size);

        if (is_multi_process)
        {
            // Multi-process mode: coordinate NCCL unique ID via MPI
            LOG_DEBUG("NCCLBackend: Multi-process mode - MPI rank " << mpi_ctx_->rank()
                                                                    << "/" << mpi_ctx_->world_size());

            // Rank 0 generates the unique ID
            if (mpi_ctx_->rank() == 0)
            {
                if (!nccl_backend_detail::ncclGetUniqueIdWrapper(unique_id_buffer.data()))
                {
                    last_error_ = "ncclGetUniqueId failed";
                    LOG_ERROR(last_error_);
                    nccl_backend_detail::cudaDestroyStream(stream_);
                    stream_ = nullptr;
                    return false;
                }
                LOG_DEBUG("NCCLBackend: Rank 0 generated NCCL unique ID");
            }

            // Broadcast unique ID from rank 0 to all other ranks
            int mpi_err = MPI_Bcast(unique_id_buffer.data(), static_cast<int>(unique_id_size), MPI_BYTE, 0, mpi_ctx_->comm());
            if (mpi_err != MPI_SUCCESS)
            {
                last_error_ = "MPI_Bcast of NCCL unique ID failed";
                LOG_ERROR(last_error_);
                nccl_backend_detail::cudaDestroyStream(stream_);
                stream_ = nullptr;
                return false;
            }
            LOG_DEBUG("NCCLBackend: NCCL unique ID broadcast complete");

            // Initialize NCCL communicator with the broadcast ID
            // Use MPI world_size and rank for the communicator
            std::string nccl_error;
            void *comm_ptr = nullptr;
            if (!nccl_backend_detail::ncclCommInitRankWrapper(&comm_ptr, mpi_ctx_->world_size(),
                                                              unique_id_buffer.data(), mpi_ctx_->rank(), nccl_error))
            {
                last_error_ = "ncclCommInitRank failed: " + nccl_error;
                LOG_ERROR(last_error_);
                nccl_backend_detail::cudaDestroyStream(stream_);
                stream_ = nullptr;
                return false;
            }
            comm_ = comm_ptr;

            // Update num_ranks_ to reflect MPI world size
            num_ranks_ = mpi_ctx_->world_size();
            local_rank_ = mpi_ctx_->rank();

            LOG_INFO("NCCLBackend: Initialized multi-process communicator with "
                     << num_ranks_ << " ranks, local_rank=" << local_rank_
                     << ", device=" << local_device.ordinal);
        }
        else if (num_ranks_ == 1)
        {
            // Single GPU - create a trivial communicator
            LOG_DEBUG("NCCLBackend: Single-GPU mode");
            std::string nccl_error;
            void *comm_ptr = nullptr;
            if (!nccl_backend_detail::ncclCommInitAllWrapper(&comm_ptr, 1, nullptr, nccl_error))
            {
                last_error_ = "ncclCommInitAll failed: " + nccl_error;
                LOG_ERROR(last_error_);
                nccl_backend_detail::cudaDestroyStream(stream_);
                stream_ = nullptr;
                return false;
            }
            comm_ = comm_ptr;
            LOG_INFO("NCCLBackend: Initialized single-GPU communicator");
        }
        else
        {
            // Multi-GPU single process (no MPI)
            // This requires per-GPU communicators and streams with threaded initialization
            LOG_DEBUG("NCCLBackend: Single-process multi-GPU mode with " << num_ranks_ << " GPUs");

            is_multi_gpu_single_process_ = true;

            // Store device ordinals from the group
            device_ordinals_.clear();
            for (const auto &device : group.devices)
            {
                device_ordinals_.push_back(device.ordinal);
            }

            // Generate unique ID for this communicator group
            std::vector<char> multi_gpu_id_buffer(nccl_backend_detail::ncclUniqueIdSize());
            if (!nccl_backend_detail::ncclGetUniqueIdWrapper(multi_gpu_id_buffer.data()))
            {
                last_error_ = "ncclGetUniqueId failed";
                LOG_ERROR(last_error_);
                nccl_backend_detail::cudaDestroyStream(stream_);
                stream_ = nullptr;
                return false;
            }

            // Resize vectors for per-GPU resources
            all_comms_.resize(num_ranks_, nullptr);
            all_streams_.resize(num_ranks_, nullptr);

            // Create per-GPU streams
            for (int i = 0; i < num_ranks_; ++i)
            {
                nccl_backend_detail::cudaSetDeviceOrdinal(device_ordinals_[i]);
                void *stream_ptr = nullptr;
                if (!nccl_backend_detail::cudaCreateStream(&stream_ptr))
                {
                    last_error_ = "cudaStreamCreate failed for GPU " + std::to_string(i);
                    LOG_ERROR(last_error_);
                    // Clean up already-created streams
                    for (int j = 0; j < i; ++j)
                    {
                        nccl_backend_detail::cudaSetDeviceOrdinal(device_ordinals_[j]);
                        nccl_backend_detail::cudaDestroyStream(all_streams_[j]);
                    }
                    all_streams_.clear();
                    return false;
                }
                all_streams_[i] = stream_ptr;
            }

            // Initialize communicators using threaded ncclCommInitRank
            // Each GPU must call ncclCommInitRank in a separate thread
            std::vector<std::thread> init_threads;
            std::atomic<int> init_errors{0};
            std::vector<std::string> thread_errors(num_ranks_);

            for (int i = 0; i < num_ranks_; ++i)
            {
                init_threads.emplace_back([this, i, &multi_gpu_id_buffer, &init_errors, &thread_errors]()
                                          {
                    nccl_backend_detail::cudaSetDeviceOrdinal(device_ordinals_[i]);
                    void* comm_ptr = nullptr;
                    if (!nccl_backend_detail::ncclCommInitRankWrapper(&comm_ptr, num_ranks_, 
                                                                      multi_gpu_id_buffer.data(), i, thread_errors[i]))
                    {
                        LOG_ERROR("ncclCommInitRank failed for GPU " << i << ": " << thread_errors[i]);
                        init_errors++;
                    }
                    else
                    {
                        all_comms_[i] = comm_ptr;
                    } });
            }

            // Wait for all threads to complete
            for (auto &t : init_threads)
            {
                t.join();
            }

            if (init_errors > 0)
            {
                last_error_ = "ncclCommInitRank failed for " + std::to_string(init_errors.load()) + " GPU(s)";
                LOG_ERROR(last_error_);
                // Clean up streams
                for (int i = 0; i < num_ranks_; ++i)
                {
                    nccl_backend_detail::cudaSetDeviceOrdinal(device_ordinals_[i]);
                    if (all_streams_[i])
                        nccl_backend_detail::cudaDestroyStream(all_streams_[i]);
                    if (all_comms_[i])
                        nccl_backend_detail::ncclCommDestroyWrapper(all_comms_[i]);
                }
                all_streams_.clear();
                all_comms_.clear();
                device_ordinals_.clear();
                return false;
            }

            LOG_INFO("NCCLBackend: Initialized multi-GPU single-process (threaded ncclCommInitRank) with "
                     << num_ranks_ << " GPU(s), local_rank=" << local_rank_);
        }

        initialized_ = true;
        return true;
#else
        last_error_ = "NCCL not available (HAVE_NCCL not defined)";
        LOG_ERROR(last_error_);
        return false;
#endif
    }

    void NCCLBackend::shutdown()
    {
#ifdef HAVE_NCCL
        if (!initialized_)
        {
            return;
        }

        // Clean up multi-GPU single-process resources
        if (is_multi_gpu_single_process_)
        {
            for (size_t i = 0; i < all_comms_.size(); ++i)
            {
                if (all_comms_[i])
                {
                    nccl_backend_detail::cudaSetDeviceOrdinal(device_ordinals_[i]);
                    nccl_backend_detail::ncclCommDestroyWrapper(all_comms_[i]);
                }
            }
            all_comms_.clear();

            for (size_t i = 0; i < all_streams_.size(); ++i)
            {
                if (all_streams_[i])
                {
                    nccl_backend_detail::cudaSetDeviceOrdinal(device_ordinals_[i]);
                    nccl_backend_detail::cudaDestroyStream(all_streams_[i]);
                }
            }
            all_streams_.clear();
            device_ordinals_.clear();
            is_multi_gpu_single_process_ = false;
        }

        if (comm_)
        {
            nccl_backend_detail::ncclCommDestroyWrapper(comm_);
            comm_ = nullptr;
        }

        if (stream_)
        {
            nccl_backend_detail::cudaDestroyStream(stream_);
            stream_ = nullptr;
        }

        // Free persistent stridedAllgather temp buffer
        if (strided_allgather_temp_buf_ != nullptr)
        {
            nccl_backend_detail::cudaFreeTempBuffer(strided_allgather_temp_buf_);
            strided_allgather_temp_buf_ = nullptr;
            strided_allgather_temp_size_ = 0;
        }

        initialized_ = false;
        LOG_DEBUG("NCCLBackend: Shutdown complete");
#endif
    }

    // =========================================================================
    // Collective Operations
    // =========================================================================

    bool NCCLBackend::allreduce(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
#ifdef HAVE_NCCL
        if (!initialized_)
        {
            last_error_ = "NCCLBackend not initialized";
            return false;
        }

        std::string nccl_error;
        if (!nccl_backend_detail::ncclAllReduceWrapper(
                buffer,
                buffer, // In-place: send and recv are the same
                count,
                toNcclDataTypeInt(dtype),
                toNcclRedOpInt(op),
                comm_,
                stream_,
                nccl_error))
        {
            last_error_ = "ncclAllReduce failed: " + nccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLBackend::allgather(
        const void *send_buf,
        void *recv_buf,
        size_t send_count,
        CollectiveDataType dtype)
    {
#ifdef HAVE_NCCL
        if (!initialized_)
        {
            last_error_ = "NCCLBackend not initialized";
            return false;
        }

        std::string nccl_error;
        if (!nccl_backend_detail::ncclAllGatherWrapper(
                send_buf,
                recv_buf,
                send_count,
                toNcclDataTypeInt(dtype),
                comm_,
                stream_,
                nccl_error))
        {
            last_error_ = "ncclAllGather failed: " + nccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLBackend::allgatherv(
        const void *send_buf,
        size_t send_count,
        void *recv_buf,
        const std::vector<int> &recv_counts,
        const std::vector<int> &displacements,
        CollectiveDataType dtype)
    {
#ifdef HAVE_NCCL
        if (!initialized_)
        {
            last_error_ = "NCCLBackend not initialized";
            return false;
        }

        // NCCL does not have a native allgatherv. We emulate it using ncclSend/ncclRecv.
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

        // Variable counts - NCCL doesn't support this natively
        // Fall back to point-to-point sends/recvs
        int dtype_int = toNcclDataTypeInt(dtype);

        std::string error_out;
        if (!nccl_backend_detail::ncclGroupStartWrapper(error_out))
        {
            last_error_ = "ncclGroupStart failed: " + error_out;
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
            if (!nccl_backend_detail::ncclSendWrapper(send_buf, send_count, dtype_int, peer, comm_, stream_, error_out))
            {
                nccl_backend_detail::ncclGroupEndWrapper(error_out);
                last_error_ = "ncclSend failed in allgatherv: " + error_out;
                return false;
            }

            // Receive from peer at their offset
            char *recv_ptr = static_cast<char *>(recv_buf) + displacements[peer] * dtype_size;
            if (!nccl_backend_detail::ncclRecvWrapper(recv_ptr, recv_counts[peer], dtype_int, peer, comm_, stream_, error_out))
            {
                nccl_backend_detail::ncclGroupEndWrapper(error_out);
                last_error_ = "ncclRecv failed in allgatherv: " + error_out;
                return false;
            }
        }

        if (!nccl_backend_detail::ncclGroupEndWrapper(error_out))
        {
            last_error_ = "ncclGroupEnd failed: " + error_out;
            return false;
        }

        return true;
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLBackend::reduceScatter(
        const void *send_buf,
        void *recv_buf,
        size_t recv_count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
#ifdef HAVE_NCCL
        if (!initialized_)
        {
            last_error_ = "NCCLBackend not initialized";
            return false;
        }

        std::string nccl_error;
        if (!nccl_backend_detail::ncclReduceScatterWrapper(
                send_buf,
                recv_buf,
                recv_count,
                toNcclDataTypeInt(dtype),
                toNcclRedOpInt(op),
                comm_,
                stream_,
                nccl_error))
        {
            last_error_ = "ncclReduceScatter failed: " + nccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLBackend::broadcast(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        int root)
    {
#ifdef HAVE_NCCL
        if (!initialized_)
        {
            last_error_ = "NCCLBackend not initialized";
            return false;
        }

        std::string nccl_error;
        if (!nccl_backend_detail::ncclBroadcastWrapper(
                buffer,
                count,
                toNcclDataTypeInt(dtype),
                root,
                comm_,
                stream_,
                nccl_error))
        {
            last_error_ = "ncclBroadcast failed: " + nccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    // =========================================================================
    // Column-Parallel (Strided) AllGather Implementation
    // =========================================================================

    bool NCCLBackend::stridedAllgather(
        const void *send_buf,
        void *recv_buf,
        size_t seq_len,
        size_t local_dim,
        CollectiveDataType dtype)
    {
#ifdef HAVE_NCCL
        if (!initialized_)
        {
            last_error_ = "NCCLBackend not initialized";
            return false;
        }

        if (dtype != CollectiveDataType::FLOAT32)
        {
            last_error_ = "stridedAllgather only supports FLOAT32";
            return false;
        }

        if (seq_len == 0 || local_dim == 0)
        {
            // Nothing to do
            return true;
        }

        // Total elements per rank = seq_len * local_dim
        // After AllGather, temp buffer has: [num_ranks * seq_len, local_dim]
        // Final output layout: [seq_len, local_dim * num_ranks]

        const size_t send_count = seq_len * local_dim;
        const size_t temp_buffer_bytes = send_count * num_ranks_ * sizeof(float);

        // Use persistent temp buffer to avoid per-call alloc/sync overhead.
        // Only reallocate if current buffer is too small.
        if (strided_allgather_temp_buf_ == nullptr || strided_allgather_temp_size_ < temp_buffer_bytes)
        {
            // Free old buffer if exists
            if (strided_allgather_temp_buf_ != nullptr)
            {
                nccl_backend_detail::cudaFreeTempBuffer(strided_allgather_temp_buf_);
                strided_allgather_temp_buf_ = nullptr;
                strided_allgather_temp_size_ = 0;
            }

            // Allocate new buffer (with some headroom to reduce reallocations)
            size_t alloc_bytes = temp_buffer_bytes + (temp_buffer_bytes / 4); // 25% extra
            if (!nccl_backend_detail::cudaAllocTempBuffer(&strided_allgather_temp_buf_, alloc_bytes))
            {
                last_error_ = "Failed to allocate temp buffer for stridedAllgather: " +
                              std::to_string(alloc_bytes) + " bytes";
                LOG_ERROR(last_error_);
                return false;
            }
            strided_allgather_temp_size_ = alloc_bytes;
            LOG_DEBUG("[NCCLBackend] Allocated persistent stridedAllgather temp buffer: " << alloc_bytes << " bytes");
        }

        // Step 1: NCCL AllGather to contiguous temp buffer (async)
        std::string nccl_error;
        bool success = nccl_backend_detail::ncclAllGatherWrapper(
            send_buf,
            strided_allgather_temp_buf_, // Use persistent buffer
            send_count,
            toNcclDataTypeInt(dtype),
            comm_,
            stream_,
            nccl_error);

        if (!success)
        {
            last_error_ = "ncclAllGather failed in stridedAllgather: " + nccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        // Step 2: Launch deinterleave kernel to convert contiguous→strided (async)
        // Kernel is queued AFTER NCCL op on same stream, so ordering is correct.
        success = nccl_backend_detail::launchDeinterleaveKernel(
            strided_allgather_temp_buf_, // Use persistent buffer
            recv_buf,
            static_cast<int>(seq_len),
            static_cast<int>(local_dim),
            num_ranks_,
            stream_);

        if (!success)
        {
            last_error_ = "Deinterleave kernel launch failed in stridedAllgather";
            LOG_ERROR(last_error_);
            return false;
        }

        // NO SYNC NEEDED: Both NCCL op and deinterleave kernel are enqueued on stream_.
        // The persistent temp buffer is safe to reuse because:
        // 1. Next stridedAllgather call will enqueue AFTER this one completes (CUDA stream ordering)
        // 2. The temp buffer is not freed, so no race with cudaFree()
        //
        // This removes the ~200ms sync overhead that was killing decode performance!

        LOG_DEBUG("[NCCLBackend] stridedAllgather completed: seq_len=" << seq_len
                                                                       << " local_dim=" << local_dim << " num_ranks=" << num_ranks_);
        return true;
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    // =========================================================================
    // Multi-GPU Single-Process Support
    // =========================================================================

    bool NCCLBackend::isMultiGpuSingleProcess() const
    {
        return is_multi_gpu_single_process_;
    }

    bool NCCLBackend::allreduceMulti(const std::vector<void *> &buffers, size_t count,
                                     CollectiveDataType dtype, CollectiveOp op)
    {
#ifdef HAVE_NCCL
        if (!initialized_)
        {
            last_error_ = "NCCLBackend not initialized";
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
                          ") does not match GPU count (" + std::to_string(num_ranks_) + ")";
            LOG_ERROR(last_error_);
            return false;
        }

        int dtype_int = toNcclDataTypeInt(dtype);
        int op_int = toNcclRedOpInt(op);
        std::string nccl_error;

        // Use group API for multi-GPU single-process
        if (!nccl_backend_detail::ncclGroupStartWrapper(nccl_error))
        {
            last_error_ = "ncclGroupStart failed: " + nccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        for (int i = 0; i < num_ranks_; ++i)
        {
            nccl_backend_detail::cudaSetDeviceOrdinal(device_ordinals_[i]);
            if (!nccl_backend_detail::ncclAllReduceInGroupWrapper(buffers[i], buffers[i], count, dtype_int, op_int,
                                                                  all_comms_[i], all_streams_[i], nccl_error))
            {
                last_error_ = "ncclAllReduce failed for GPU " + std::to_string(i) + ": " + nccl_error;
                LOG_ERROR(last_error_);
                nccl_backend_detail::ncclGroupEndWrapper(nccl_error);
                return false;
            }
        }

        if (!nccl_backend_detail::ncclGroupEndWrapper(nccl_error))
        {
            last_error_ = "ncclGroupEnd failed: " + nccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)buffers;
        (void)count;
        (void)dtype;
        (void)op;
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLBackend::allgatherMulti(const std::vector<const void *> &send_buffers,
                                     const std::vector<void *> &recv_buffers, size_t send_count,
                                     CollectiveDataType dtype)
    {
#ifdef HAVE_NCCL
        if (!initialized_)
        {
            last_error_ = "NCCLBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (!is_multi_gpu_single_process_)
        {
            last_error_ = "allgatherMulti requires multi-GPU single-process mode";
            LOG_ERROR(last_error_);
            return false;
        }

        if (send_buffers.size() != static_cast<size_t>(num_ranks_) ||
            recv_buffers.size() != static_cast<size_t>(num_ranks_))
        {
            last_error_ = "Buffer count does not match GPU count";
            LOG_ERROR(last_error_);
            return false;
        }

        int dtype_int = toNcclDataTypeInt(dtype);
        std::string nccl_error;

        // Use group API for multi-GPU single-process
        if (!nccl_backend_detail::ncclGroupStartWrapper(nccl_error))
        {
            last_error_ = "ncclGroupStart failed: " + nccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        for (int i = 0; i < num_ranks_; ++i)
        {
            nccl_backend_detail::cudaSetDeviceOrdinal(device_ordinals_[i]);
            if (!nccl_backend_detail::ncclAllGatherInGroupWrapper(send_buffers[i], recv_buffers[i], send_count, dtype_int,
                                                                  all_comms_[i], all_streams_[i], nccl_error))
            {
                last_error_ = "ncclAllGather failed for GPU " + std::to_string(i) + ": " + nccl_error;
                LOG_ERROR(last_error_);
                nccl_backend_detail::ncclGroupEndWrapper(nccl_error);
                return false;
            }
        }

        if (!nccl_backend_detail::ncclGroupEndWrapper(nccl_error))
        {
            last_error_ = "ncclGroupEnd failed: " + nccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)send_buffers;
        (void)recv_buffers;
        (void)send_count;
        (void)dtype;
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLBackend::broadcastMulti(const std::vector<void *> &buffers, size_t count,
                                     CollectiveDataType dtype, int root)
    {
#ifdef HAVE_NCCL
        if (!initialized_)
        {
            last_error_ = "NCCLBackend not initialized";
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
                          ") does not match GPU count (" + std::to_string(num_ranks_) + ")";
            LOG_ERROR(last_error_);
            return false;
        }

        if (root < 0 || root >= num_ranks_)
        {
            last_error_ = "Invalid root rank: " + std::to_string(root);
            LOG_ERROR(last_error_);
            return false;
        }

        int dtype_int = toNcclDataTypeInt(dtype);
        std::string nccl_error;

        // Use group API for multi-GPU single-process
        if (!nccl_backend_detail::ncclGroupStartWrapper(nccl_error))
        {
            last_error_ = "ncclGroupStart failed: " + nccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        for (int i = 0; i < num_ranks_; ++i)
        {
            nccl_backend_detail::cudaSetDeviceOrdinal(device_ordinals_[i]);
            if (!nccl_backend_detail::ncclBroadcastInGroupWrapper(buffers[i], count, dtype_int, root,
                                                                  all_comms_[i], all_streams_[i], nccl_error))
            {
                last_error_ = "ncclBroadcast failed for GPU " + std::to_string(i) + ": " + nccl_error;
                LOG_ERROR(last_error_);
                nccl_backend_detail::ncclGroupEndWrapper(nccl_error);
                return false;
            }
        }

        if (!nccl_backend_detail::ncclGroupEndWrapper(nccl_error))
        {
            last_error_ = "ncclGroupEnd failed: " + nccl_error;
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        (void)buffers;
        (void)count;
        (void)dtype;
        (void)root;
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    // =========================================================================
    // Synchronization
    // =========================================================================

    bool NCCLBackend::synchronize()
    {
#ifdef HAVE_NCCL
        if (!initialized_)
        {
            return true;
        }

        // In multi-GPU single-process mode, synchronize all streams
        if (is_multi_gpu_single_process_)
        {
            for (size_t i = 0; i < all_streams_.size(); ++i)
            {
                nccl_backend_detail::cudaSetDeviceOrdinal(device_ordinals_[i]);
                if (!nccl_backend_detail::cudaSynchronizeStream(all_streams_[i]))
                {
                    last_error_ = "cudaStreamSynchronize failed for GPU " + std::to_string(i);
                    LOG_ERROR(last_error_);
                    return false;
                }
            }
            return true;
        }

        if (!nccl_backend_detail::cudaSynchronizeStream(stream_))
        {
            last_error_ = "cudaStreamSynchronize failed";
            LOG_ERROR(last_error_);
            return false;
        }

        return true;
#else
        return true;
#endif
    }

    // =========================================================================
    // Type Conversion Helpers (removed - now using int-based converters above)
    // =========================================================================
    // See toNcclDataTypeInt() and toNcclRedOpInt() for the portable implementations.

} // namespace llaminar2
