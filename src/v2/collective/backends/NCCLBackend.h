/**
 * @file NCCLBackend.h
 * @brief NCCL-based collective backend for NVIDIA GPUs
 *
 * The NCCLBackend provides high-performance collective operations for
 * homogeneous NVIDIA GPU configurations. NCCL automatically detects
 * and uses the optimal interconnect (NVLink, PCIe, etc.).
 *
 * Requirements:
 * - NVIDIA GPUs with CUDA support
 * - NCCL library installed (libnccl-dev)
 * - All participating GPUs must be CUDA devices
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../ICollectiveBackend.h"
#include "../DeviceGroup.h"
#include "../coordinators/NCCLCoordinator.h"
#include "../../utils/MPIContext.h"

// Note: NCCL types are NOT exposed in this header to avoid conflicts with RCCL/HIP
// when building with both CUDA and ROCm. All NCCL-specific operations are isolated
// in NCCLBackendCUDA.cu, and void* is used for opaque handles here.
// The actual types (ncclComm_t, cudaStream_t) are defined in the implementation.

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    /**
     * @brief NCCL-based collective backend for NVIDIA GPUs
     *
     * Provides optimal GPU-GPU collective operations using NVIDIA's
     * Collective Communications Library. Supports NVLink when available.
     *
     * Thread Safety: NCCL communicators are not thread-safe.
     *                Use one NCCLBackend instance per device/stream.
     */
    class NCCLBackend : public ICollectiveBackend
    {
    public:
        /**
         * @brief Construct NCCLBackend
         * @param mpi_ctx Optional MPI context for multi-process initialization.
         *                If provided and world_size > 1, uses MPI to broadcast
         *                NCCL unique ID for distributed communicator setup.
         */
        explicit NCCLBackend(std::shared_ptr<MPIContext> mpi_ctx = nullptr);
        ~NCCLBackend() override;

        // =====================================================================
        // Identity
        // =====================================================================

        CollectiveBackendType type() const override { return CollectiveBackendType::NCCL; }
        std::string name() const override { return "NCCL"; }

        // =====================================================================
        // Capability Queries
        // =====================================================================

        bool supportsDevice(DeviceType type) const override
        {
            return type == DeviceType::CUDA;
        }

        bool supportsDirectTransfer(DeviceId src, DeviceId dst) const override
        {
            // NCCL handles all CUDA-CUDA transfers directly
            return src.type == DeviceType::CUDA && dst.type == DeviceType::CUDA;
        }

        bool isAvailable() const override;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        bool initialize(const DeviceGroup &group) override;
        bool isInitialized() const override { return initialized_; }
        void shutdown() override;

        // =====================================================================
        // Collective Operations
        // =====================================================================

        bool allreduce(
            void *buffer,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op) override;

        bool allgather(
            const void *send_buf,
            void *recv_buf,
            size_t send_count,
            CollectiveDataType dtype) override;

        bool allgatherv(
            const void *send_buf,
            size_t send_count,
            void *recv_buf,
            const std::vector<int> &recv_counts,
            const std::vector<int> &displacements,
            CollectiveDataType dtype) override;

        bool reduceScatter(
            const void *send_buf,
            void *recv_buf,
            size_t recv_count,
            CollectiveDataType dtype,
            CollectiveOp op) override;

        bool broadcast(
            void *buffer,
            size_t count,
            CollectiveDataType dtype,
            int root) override;

        // =====================================================================
        // Point-to-Point Operations
        // =====================================================================

        bool send(void *buffer, size_t count, CollectiveDataType dtype,
                  int peer, int tag = 0) override;

        bool recv(void *buffer, size_t count, CollectiveDataType dtype,
                  int peer, int tag = 0) override;

        bool sendrecv(void *sendbuf, void *recvbuf, size_t count,
                      CollectiveDataType dtype, int peer) override;

        // =====================================================================
        // Async Point-to-Point Operations
        // =====================================================================

        /**
         * @brief Async send using caller-provided stream
         *
         * Issues ncclSend on the provided stream. Completion is tracked by
         * the stream, allowing the caller to overlap with other operations.
         *
         * @param buffer Source buffer
         * @param count Number of elements
         * @param dtype Data type
         * @param peer Target rank
         * @param stream CUDA stream (cudaStream_t cast to void*)
         * @param tag Ignored for NCCL (uses rank for matching)
         * @return true if operation was issued
         */
        bool sendAsync(void *buffer, size_t count, CollectiveDataType dtype,
                       int peer, void *stream, int tag = 0) override;

        /**
         * @brief Async receive using caller-provided stream
         *
         * Issues ncclRecv on the provided stream.
         */
        bool recvAsync(void *buffer, size_t count, CollectiveDataType dtype,
                       int peer, void *stream, int tag = 0) override;

        /**
         * @brief Async bidirectional send-receive using caller-provided stream
         *
         * Issues ncclSend and ncclRecv in a group call on the provided stream.
         */
        bool sendrecvAsync(void *sendbuf, void *recvbuf, size_t count,
                           CollectiveDataType dtype, int peer, void *stream) override;

        // =====================================================================
        // Column-Parallel (Strided) AllGather for LM Head
        // =====================================================================

        /**
         * @brief Strided AllGather for column-parallel operations
         *
         * This is optimized for column-parallel LM head where:
         * - Each rank has local logits [seq_len, vocab_local]
         * - Output is [seq_len, vocab_full] with interleaved columns
         *
         * Implementation:
         * 1. NCCL AllGather to temp buffer (contiguous by rank)
         * 2. CUDA kernel to deinterleave into strided output
         *
         * This avoids host memory staging that MPI_Type_vector would require.
         *
         * @param send_buf Local slice [seq_len * local_dim] contiguous
         * @param recv_buf Output [seq_len * full_dim] with strided layout
         * @param seq_len Number of rows (tokens)
         * @param local_dim Columns per rank (vocab_local)
         * @param dtype Data type (must be FLOAT32)
         * @return true on success
         */
        bool stridedAllgather(
            const void *send_buf,
            void *recv_buf,
            size_t seq_len,
            size_t local_dim,
            CollectiveDataType dtype);

        // =====================================================================
        // Multi-GPU Single-Process Operations
        // =====================================================================

        /// Check if initialized in multi-GPU single-process mode
        bool isMultiGpuSingleProcess() const override;

        /// Multi-GPU AllReduce (each GPU has its own buffer)
        bool allreduceMulti(
            const std::vector<void *> &buffers,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op) override;

        bool allreduceMultiAndSynchronize(
            const std::vector<void *> &buffers,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op) override;

        /// Multi-GPU AllGather (each GPU sends from send_bufs, receives to recv_bufs)
        bool allgatherMulti(
            const std::vector<const void *> &send_bufs,
            const std::vector<void *> &recv_bufs,
            size_t send_count,
            CollectiveDataType dtype) override;

        /// Multi-GPU Broadcast (root's buffer is broadcast to all)
        bool broadcastMulti(
            const std::vector<void *> &buffers,
            size_t count,
            CollectiveDataType dtype,
            int root) override;

        /// Multi-GPU Reduce (all buffers reduced to root)
        bool reduceMulti(
            const std::vector<void *> &buffers,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op,
            int root) override;

        /// Multi-GPU Reduce-Scatter (each GPU gets 1/N of reduced result)
        bool reduceScatterMulti(
            const std::vector<const void *> &send_buffers,
            const std::vector<void *> &recv_buffers,
            size_t recv_count,
            CollectiveDataType dtype,
            CollectiveOp op) override;

        /// Multi-GPU P2P Send/Recv (coordinates both endpoints in single group)
        bool sendrecvMulti(
            void *src_buffer,
            void *dst_buffer,
            size_t count,
            CollectiveDataType dtype,
            int src_gpu,
            int dst_gpu) override;

        // =====================================================================
        // Data Copy Operations
        // =====================================================================

        /**
         * @brief Synchronous copy between CUDA devices
         *
         * Supports:
         * - Same device: cudaMemcpy DeviceToDevice
         * - Different devices: cudaMemcpyPeer (requires P2P access)
         *
         * FAIL-FAST: Returns false if P2P is not available between different devices.
         * Does NOT fall back to host staging.
         *
         * @return false for non-CUDA devices or if P2P unavailable
         */
        bool copy(void *dst_ptr, DeviceId dst_device,
                  const void *src_ptr, DeviceId src_device,
                  size_t bytes) override;

        /**
         * @brief Async copy between CUDA devices
         *
         * Same semantics as copy() but non-blocking.
         */
        bool copyAsync(void *dst_ptr, DeviceId dst_device,
                       const void *src_ptr, DeviceId src_device,
                       size_t bytes, void *stream = nullptr) override;

        /**
         * @brief Check if copy is supported between device pair
         *
         * Returns true only for CUDA↔CUDA pairs where P2P is available
         * (or same device).
         */
        bool supportsCopy(DeviceId src_device, DeviceId dst_device) const override;

        // =====================================================================
        // Synchronization
        // =====================================================================

        bool synchronize() override;

        // =====================================================================
        // Error Handling
        // =====================================================================

        std::string lastError() const override { return last_error_; }

        // =====================================================================
        // NCCL-specific
        // =====================================================================

        /// Get the number of GPUs in the communicator
        int numRanks() const { return num_ranks_; }

        /// Get the local rank within the communicator
        int localRank() const { return local_rank_; }

    private:
        bool initialized_ = false;
        std::string last_error_;
        int num_ranks_ = 0;
        int local_rank_ = 0;
        std::shared_ptr<MPIContext> mpi_ctx_;
        bool is_multi_gpu_single_process_ = false;

#ifdef HAVE_NCCL
        // NOTE: We use void* for comm and streams to avoid CUDA/HIP/NCCL header conflicts.
        // When building with both CUDA and ROCm, NCCL headers can conflict with RCCL/HIP.
        // The actual types (ncclComm_t, cudaStream_t) are cast in the .cpp/.cu implementation.
        void *comm_ = nullptr;
        void *stream_ = nullptr;

        // Multi-GPU single-process: per-GPU communicators and streams
        std::vector<void *> all_comms_;    // One communicator per GPU (ncclComm_t) - ONLY for non-coordinator mode
        std::vector<void *> all_streams_;  // One stream per GPU (cudaStream_t) - ONLY for non-coordinator mode
        std::vector<int> device_ordinals_; // Device ordinals for each GPU

        // Coordinator for multi-GPU single-process mode (owns all NCCL comms/streams)
        std::unique_ptr<NCCLCoordinator> coordinator_;

        // Persistent temp buffer for stridedAllgather (avoids per-call alloc/sync)
        void *strided_allgather_temp_buf_ = nullptr;
        size_t strided_allgather_temp_size_ = 0;

        // =====================================================================
        // All-GPU NCCL communicator for copy() operations
        // =====================================================================
        // Initialized once during initialize() to span ALL enumerated CUDA devices.
        // This allows efficient GPU-to-GPU copy via ncclSend/ncclRecv.
        // NCCL handles all transport details (P2P, NVLink, PCIe staging) internally.

        int copy_num_gpus_ = 0;               // Number of CUDA GPUs (all enumerated)
        std::vector<void *> copy_comms_;      // Per-GPU NCCL communicators (size = copy_num_gpus_)
        std::vector<void *> copy_streams_;    // Per-GPU CUDA streams (size = copy_num_gpus_)
        bool copy_comms_initialized_ = false; // Flag to track initialization status

        /// Initialize the all-GPU copy communicator (called from initialize())
        bool initializeCopyComms();

        /// Shutdown the copy communicator (called from shutdown())
        void shutdownCopyComms();
#endif
    };

} // namespace llaminar2
