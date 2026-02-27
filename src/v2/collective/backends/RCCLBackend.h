/**
 * @file RCCLBackend.h
 * @brief RCCL-based collective backend for AMD ROCm GPUs
 *
 * The RCCLBackend provides high-performance collective operations for
 * homogeneous AMD GPU configurations. RCCL automatically detects
 * and uses the optimal interconnect (Infinity Fabric, PCIe, etc.).
 *
 * Requirements:
 * - AMD GPUs with ROCm support
 * - RCCL library installed (librccl.so available)
 * - All participating GPUs must be ROCm devices
 *
 * IMPORTANT: Uses dynamic loader (dlopen/dlsym) for RCCL to avoid symbol
 * conflicts with NCCL, which exports identical symbol names. This allows
 * both libraries to coexist in the same process.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../ICollectiveBackend.h"
#include "../DeviceGroup.h"
#include "../../utils/MPIContext.h"

// Note: No longer include rccl.h directly - use dynamic loading via wrappers

#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declaration
    class RCCLCoordinator;

    /**
     * @brief RCCL-based collective backend for AMD ROCm GPUs
     *
     * Provides optimal GPU-GPU collective operations using AMD's
     * ROCm Collective Communications Library. Supports Infinity Fabric
     * when available.
     *
     * Thread Safety: RCCL communicators are not thread-safe.
     *                Use one RCCLBackend instance per device/stream.
     */
    class RCCLBackend : public ICollectiveBackend
    {
    public:
        /**
         * @brief Construct RCCLBackend
         * @param mpi_ctx Optional MPI context for multi-process initialization.
         *                If provided and world_size > 1, uses MPI to coordinate
         *                RCCL communicator creation across processes.
         */
        explicit RCCLBackend(std::shared_ptr<MPIContext> mpi_ctx = nullptr);
        ~RCCLBackend() override;

        // =====================================================================
        // Identity
        // =====================================================================

        CollectiveBackendType type() const override { return CollectiveBackendType::RCCL; }
        std::string name() const override { return "RCCL"; }

        // =====================================================================
        // Capability Queries
        // =====================================================================

        bool supportsDevice(DeviceType type) const override
        {
            return type == DeviceType::ROCm;
        }

        bool supportsDirectTransfer(DeviceId src, DeviceId dst) const override
        {
            // RCCL handles all ROCm-ROCm transfers directly
            return src.type == DeviceType::ROCm && dst.type == DeviceType::ROCm;
        }

        bool isAvailable() const override;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        bool initialize(const DeviceGroup &group) override;
        bool isInitialized() const override { return initialized_; }
        void shutdown() override;
        void setComputeStreams(const std::vector<void *> &compute_streams) override;

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
         * Issues rcclSend on the provided stream. Completion is tracked by
         * the stream, allowing the caller to overlap with other operations.
         *
         * @param buffer Source buffer
         * @param count Number of elements
         * @param dtype Data type
         * @param peer Target rank
         * @param stream HIP stream (hipStream_t cast to void*)
         * @param tag Ignored for RCCL (uses rank for matching)
         * @return true if operation was issued
         */
        bool sendAsync(void *buffer, size_t count, CollectiveDataType dtype,
                       int peer, void *stream, int tag = 0) override;

        /**
         * @brief Async receive using caller-provided stream
         *
         * Issues rcclRecv on the provided stream.
         */
        bool recvAsync(void *buffer, size_t count, CollectiveDataType dtype,
                       int peer, void *stream, int tag = 0) override;

        /**
         * @brief Async bidirectional send-receive using caller-provided stream
         *
         * Issues rcclSend and rcclRecv in a group call on the provided stream.
         */
        bool sendrecvAsync(void *sendbuf, void *recvbuf, size_t count,
                           CollectiveDataType dtype, int peer, void *stream) override;

        // =====================================================================
        // Data Copy Operations
        // =====================================================================

        /**
         * @brief Synchronous copy between ROCm devices
         *
         * Supports:
         * - Same device: hipMemcpy DeviceToDevice
         * - Different devices: hipMemcpyPeer (requires P2P access)
         *
         * FAIL-FAST: Returns false if P2P is not available between different devices.
         * Does NOT fall back to host staging.
         *
         * @return false for non-ROCm devices or if P2P unavailable
         */
        bool copy(void *dst_ptr, DeviceId dst_device,
                  const void *src_ptr, DeviceId src_device,
                  size_t bytes) override;

        /**
         * @brief Async copy between ROCm devices
         *
         * Same semantics as copy() but non-blocking.
         */
        bool copyAsync(void *dst_ptr, DeviceId dst_device,
                       const void *src_ptr, DeviceId src_device,
                       size_t bytes, void *stream = nullptr) override;

        /**
         * @brief Check if copy is supported between device pair
         *
         * Returns true only for ROCm↔ROCm pairs where P2P is available
         * (or same device).
         */
        bool supportsCopy(DeviceId src_device, DeviceId dst_device) const override;

        // =====================================================================
        // Multi-GPU Single-Process Collective Operations
        // =====================================================================

        bool isMultiGpuSingleProcess() const override;

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

        bool allgatherMulti(
            const std::vector<const void *> &send_bufs,
            const std::vector<void *> &recv_bufs,
            size_t send_count,
            CollectiveDataType dtype) override;

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
        // Synchronization
        // =====================================================================

        bool synchronize() override;

        // =====================================================================
        // Error Handling
        // =====================================================================

        std::string lastError() const override { return last_error_; }

        // =====================================================================
        // RCCL-specific
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
        std::shared_ptr<MPIContext> mpi_ctx_;      // Optional MPI context for multi-process
        bool is_multi_gpu_single_process_ = false; // True if multi-GPU without MPI
        bool p2p_available_ = false;               // True if P2P available between all devices

#ifdef HAVE_RCCL
        // Use void* for opaque RCCL types - dynamic loading hides the actual types
        void *comm_ = nullptr;   // rcclComm_t (opaque pointer)
        void *stream_ = nullptr; // hipStream_t (opaque pointer)

        // Multi-GPU single-process: per-GPU communicators and streams
        std::vector<void *> all_comms_;    // One communicator per GPU - ONLY for non-coordinator mode
        std::vector<void *> all_streams_;  // One stream per GPU - ONLY for non-coordinator mode
        std::vector<int> device_ordinals_; // Device ordinals for each GPU

        // Coordinator for multi-GPU single-process mode (owns all RCCL comms/streams)
        std::unique_ptr<RCCLCoordinator> coordinator_;

        // Helper to convert our types to integer values for wrapper functions
        static int toRcclDataTypeInt(CollectiveDataType dtype);
        static int toRcclRedOpInt(CollectiveOp op);
#endif
    };

} // namespace llaminar2
