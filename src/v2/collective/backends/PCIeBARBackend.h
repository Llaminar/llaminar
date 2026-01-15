/**
 * @file PCIeBARBackend.h
 * @brief Direct CUDA↔ROCm collective backend via PCIe BAR mapping
 *
 * Enables direct GPU-to-GPU communication between NVIDIA (CUDA) and AMD (ROCm)
 * GPUs WITHOUT host memory staging, by mapping AMD GPU's BAR (Base Address
 * Register) into CUDA's address space via cuMemHostRegister.
 *
 * Performance characteristics (measured on RTX 3090 ↔ MI50):
 * - Bandwidth: ~2.65 GB/s (PCIe 3.0 x16)
 * - Latency: Lower than host-staged (no CPU involvement)
 * - Symmetric: Read/write speeds nearly identical
 *
 * This backend is preferred over HostBackend when:
 * - Device group contains exactly 1 CUDA and 1+ ROCm GPUs (or vice versa)
 * - PCIe BAR P2P is available (AMD GPU with large BAR, proper permissions)
 *
 * Requirements:
 * - HAVE_CUDA and HAVE_ROCM both defined
 * - AMD GPU with large BAR support (e.g., MI50 with 32GB BAR)
 * - CAP_SYS_ADMIN or appropriate permissions for /sys/bus/pci BAR access
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../ICollectiveBackend.h"
#include "../../backends/benchmarks/DirectP2P.h"
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
    // Full implementation when both CUDA and ROCm are available

    /**
     * @brief Direct CUDA↔ROCm collective backend via PCIe BAR mapping
     *
     * Algorithm for AllReduce (CUDA + ROCm, 2 devices):
     * 1. Both GPUs compute partial results in their buffers
     * 2. CUDA reads ROCm's result via PCIe BAR → local temp buffer
     * 3. CUDA performs reduction (local + remote)
     * 4. CUDA writes result back to ROCm via PCIe BAR
     * 5. Both GPUs now have identical reduced values
     *
     * For >2 devices, a tree reduction pattern is used.
     */
    class PCIeBARBackend : public ICollectiveBackend
    {
    public:
        /**
         * @brief Construct PCIe BAR backend
         * @param p2p_engine DirectP2PEngine for BAR transfers (takes ownership)
         */
        explicit PCIeBARBackend(std::unique_ptr<DirectP2PEngine> p2p_engine = nullptr);

        ~PCIeBARBackend() override;

        // =====================================================================
        // Identity
        // =====================================================================

        CollectiveBackendType type() const override { return CollectiveBackendType::PCIE_BAR; }
        std::string name() const override { return "PCIe_BAR"; }

        // =====================================================================
        // Capability Queries
        // =====================================================================

        bool supportsDevice(DeviceType type) const override
        {
            return type == DeviceType::CUDA || type == DeviceType::ROCm;
        }

        bool supportsDirectTransfer(DeviceId src, DeviceId dst) const override;

        bool isAvailable() const override;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        bool initialize(const DeviceGroup &group) override;
        void shutdown() override;
        bool isInitialized() const override { return initialized_; }

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
        // Diagnostics
        // =====================================================================

        /**
         * @brief Get P2P bandwidth (for logging/diagnostics)
         */
        double getMeasuredBandwidthGBps() const { return measured_bandwidth_gbps_; }

        /**
         * @brief Check if P2P engine is active
         */
        bool isPCIeBarActive() const;

        // =====================================================================
        // BAR Region Allocator
        // =====================================================================

        /**
         * @brief Allocate a buffer within the BAR-mapped region
         *
         * Uses a bump allocator to carve out regions from the mmap'd BAR.
         * The returned pointer IS directly accessible by ROCm kernels since
         * it maps to AMD GPU memory.
         *
         * @param size Size in bytes to allocate
         * @return Pair of (ROCm-accessible pointer, BAR offset), or nullopt if allocation fails
         */
        std::optional<std::pair<void *, size_t>> allocateInBarRegion(size_t size);

        /**
         * @brief Free a buffer previously allocated via allocateInBarRegion
         *
         * Note: This removes tracking only. The bump allocator does not
         * compact freed space (lifetime-based deallocation expected).
         *
         * @param ptr Pointer returned from allocateInBarRegion
         */
        void freeBarBuffer(void *ptr);

        /**
         * @brief Get current BAR allocation offset (for diagnostics)
         */
        size_t getBarAllocOffset() const { return bar_alloc_offset_; }

        /**
         * @brief Get total BAR mapped size
         */
        size_t getBarTotalMappedSize() const { return bar_total_mapped_size_; }

        // =====================================================================
        // IBufferRegistration Overrides
        // =====================================================================

        /**
         * @brief Register a buffer for a collective operation
         *
         * For CUDA buffers: stores the pointer directly
         * For ROCm buffers: must be allocated via allocateInBarRegion()
         *
         * @param collective_id Unique identifier for the collective
         * @param device Device where the buffer resides
         * @param buffer Device pointer
         * @param size Size in bytes
         * @return true on success
         */
        bool registerBuffer(const std::string &collective_id,
                            DeviceId device,
                            void *buffer,
                            size_t size) override;

        /**
         * @brief Unregister a buffer
         */
        void unregisterBuffer(const std::string &collective_id,
                              DeviceId device) override;

        /**
         * @brief Get registered buffer info
         */
        std::optional<RegisteredBuffer> getBuffer(const std::string &collective_id,
                                                  DeviceId device) const override;

        /**
         * @brief This backend requires buffer registration for cross-vendor P2P
         */
        bool requiresBufferRegistration() const override { return true; }

        // =====================================================================
        // Registered Buffer Operations
        // =====================================================================

        /**
         * @brief AllReduce using pre-registered buffers
         *
         * Uses registered buffer locations to perform cross-vendor allreduce
         * via PCIe BAR mapping.
         */
        bool allreduceRegistered(
            const std::string &collective_id,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op) override;

    private:
        std::unique_ptr<DirectP2PEngine> p2p_engine_;
        bool initialized_ = false;

        // Device group info (set during initialize)
        DeviceId cuda_device_;
        DeviceId rocm_device_;
        bool has_cuda_ = false;
        bool has_rocm_ = false;

        // Temp buffer on CUDA side for reduction operations
        void *cuda_temp_buffer_ = nullptr;
        size_t cuda_temp_buffer_size_ = 0;

        // Performance metrics
        double measured_bandwidth_gbps_ = 0.0;

        // =====================================================================
        // BAR Allocator State
        // =====================================================================

        /// Current allocation offset (bump allocator)
        size_t bar_alloc_offset_ = 0;

        /// Total mapped BAR size
        size_t bar_total_mapped_size_ = 0;

        /// Host-mapped pointer to BAR region (directly accessible by ROCm)
        void *bar_host_ptr_ = nullptr;

        /// Alignment for BAR allocations (256 bytes for GPU efficiency)
        static constexpr size_t BAR_ALLOC_ALIGNMENT = 256;

        /// Track individual BAR allocations
        struct BarAllocation
        {
            void *ptr;     ///< ROCm-accessible pointer (bar_host_ptr_ + offset)
            size_t offset; ///< Offset within BAR region
            size_t size;   ///< Allocation size (aligned)
        };
        std::vector<BarAllocation> bar_allocations_;

        // =====================================================================
        // Registered Buffer Storage
        // =====================================================================

        /// Registered buffers for a collective operation
        struct CollectiveBuffers
        {
            RegisteredBuffer cuda_buffer; ///< CUDA side buffer
            RegisteredBuffer rocm_buffer; ///< ROCm side buffer (with BAR offset)
            bool cuda_registered = false;
            bool rocm_registered = false;
        };
        std::unordered_map<std::string, CollectiveBuffers> registered_collectives_;

        // Internal helpers
        bool ensureTempBuffer(size_t bytes);
        void freeTempBuffer();

        /**
         * @brief Transfer data from CUDA to ROCm via BAR
         */
        bool transferCUDAtoROCm(const void *cuda_src, size_t offset, size_t bytes);

        /**
         * @brief Transfer data from ROCm to CUDA via BAR
         */
        bool transferROCmtoCUDA(size_t offset, void *cuda_dst, size_t bytes);

        /**
         * @brief Perform element-wise reduction on CUDA device
         */
        bool reduceOnCUDA(
            void *output,
            const void *input1,
            const void *input2,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op);

        size_t datatypeSize(CollectiveDataType dtype) const;
    };

#else
    // Stub implementation when CUDA and ROCm are not both available

    /**
     * @brief Stub PCIeBARBackend (requires both CUDA and ROCm)
     *
     * This stub is provided so code can compile without both GPU backends.
     * All operations return failure/unavailable.
     */
    class PCIeBARBackend : public ICollectiveBackend
    {
    public:
        explicit PCIeBARBackend(std::unique_ptr<DirectP2PEngine> = nullptr) {}
        ~PCIeBARBackend() override = default;

        CollectiveBackendType type() const override { return CollectiveBackendType::PCIE_BAR; }
        std::string name() const override { return "PCIe_BAR (stub)"; }
        bool supportsDevice(DeviceType) const override { return false; }
        bool supportsDirectTransfer(DeviceId, DeviceId) const override { return false; }
        bool isAvailable() const override { return false; }
        bool initialize(const DeviceGroup &) override { return false; }
        void shutdown() override {}
        bool isInitialized() const override { return false; }

        bool allreduce(void *, size_t, CollectiveDataType, CollectiveOp) override { return false; }
        bool allgather(const void *, void *, size_t, CollectiveDataType) override { return false; }
        bool reduceScatter(const void *, void *, size_t, CollectiveDataType, CollectiveOp) override { return false; }
        bool broadcast(void *, size_t, CollectiveDataType, int) override { return false; }
        bool synchronize() override { return false; }

        double getMeasuredBandwidthGBps() const { return 0.0; }
        bool isPCIeBarActive() const { return false; }

        std::optional<std::pair<void *, size_t>> allocateInBarRegion(size_t) { return std::nullopt; }
        void freeBarBuffer(void *) {}
        size_t getBarAllocOffset() const { return 0; }
        size_t getBarTotalMappedSize() const { return 0; }

        bool registerBuffer(const std::string &, DeviceId, void *, size_t) override { return false; }
        void unregisterBuffer(const std::string &, DeviceId) override {}
        std::optional<RegisteredBuffer> getBuffer(const std::string &, DeviceId) const override { return std::nullopt; }
        bool requiresBufferRegistration() const override { return false; }
        bool allreduceRegistered(const std::string &, size_t, CollectiveDataType, CollectiveOp) override { return false; }
    };

#endif // HAVE_CUDA && HAVE_ROCM

} // namespace llaminar2
