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

#ifdef HAVE_NCCL
#include <nccl.h>
#endif

#include <memory>
#include <string>
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
        NCCLBackend();
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

#ifdef HAVE_NCCL
        ncclComm_t comm_ = nullptr;
        cudaStream_t stream_ = nullptr;

        // Helper to convert our types to NCCL types
        static ncclDataType_t toNcclDataType(CollectiveDataType dtype);
        static ncclRedOp_t toNcclRedOp(CollectiveOp op);
#endif
    };

} // namespace llaminar2
