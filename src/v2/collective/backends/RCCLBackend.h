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
 * - RCCL library installed (rccl-dev)
 * - All participating GPUs must be ROCm devices
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../ICollectiveBackend.h"
#include "../DeviceGroup.h"

#ifdef HAVE_RCCL
#include <rccl/rccl.h>
#endif

#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

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
        RCCLBackend();
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

#ifdef HAVE_RCCL
        ncclComm_t comm_ = nullptr; // RCCL uses same types as NCCL
        hipStream_t stream_ = nullptr;

        // Helper to convert our types to RCCL types
        static ncclDataType_t toRcclDataType(CollectiveDataType dtype);
        static ncclRedOp_t toRcclRedOp(CollectiveOp op);
#endif
    };

} // namespace llaminar2
