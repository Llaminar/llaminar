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
#include "../../backends/p2p/DirectP2P.h"
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <thread>
#include <queue>
#include <future>
#include <functional>
#include <mutex>
#include <condition_variable>

namespace llaminar2
{

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
    // Full implementation when both CUDA and ROCm are available

    /**
     * @brief Represents a CUDA↔ROCm device pair for multi-pair PCIe BAR operations
     *
     * Used for GCD-based parallel bridging where multiple CUDA↔ROCm pairs
     * can perform allreduce operations simultaneously.
     */
    struct DevicePair
    {
        DeviceId cuda_device; ///< CUDA device in this pair
        DeviceId rocm_device; ///< ROCm device in this pair
        int pair_index;       ///< Index into BAR resources (0, 1, 2...)

        bool operator==(const DevicePair &other) const
        {
            return cuda_device == other.cuda_device && rocm_device == other.rocm_device;
        }
    };

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

        /**
         * @brief Initialize backend for multiple CUDA↔ROCm device pairs
         *
         * Enables GCD-based parallel bridging where multiple device pairs
         * can perform allreduce operations simultaneously.
         *
         * @param pairs Vector of device pairs to initialize
         * @return true if all pairs initialized successfully
         */
        bool initializeMultiPair(const std::vector<DevicePair> &pairs);

        /**
         * @brief Get the configured device pairs
         * @return Vector of device pairs (empty if single-pair mode)
         */
        const std::vector<DevicePair> &getDevicePairs() const { return device_pairs_; }

        /**
         * @brief Check if multi-pair mode is active
         */
        bool isMultiPairMode() const { return !device_pairs_.empty(); }

        /**
         * @brief Reserve temp buffer capacity to avoid hot-path allocation
         *
         * Pre-allocates the CUDA temp buffer used for allreduce operations.
         * The buffer will grow if needed but never shrink during operation.
         *
         * @param bytes Minimum buffer capacity in bytes
         * @return true if reservation succeeded
         */
        bool reserveTempBufferBytes(size_t bytes) override;

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

        /**
         * @brief Send data to a peer device via PCIe BAR
         *
         * For PCIeBAR, this performs a direct memory transfer to the peer's
         * BAR-mapped memory region. Only supports CUDA→ROCm or ROCm→CUDA transfers.
         *
         * @param buffer Source buffer on this device
         * @param count Number of elements
         * @param dtype Data type
         * @param peer Target device rank (0=CUDA, 1=ROCm in 2-device mode)
         * @param tag Message tag (ignored for BAR transfers)
         * @return true on success
         */
        bool send(void *buffer, size_t count, CollectiveDataType dtype,
                  int peer, int tag = 0) override;

        /**
         * @brief Receive data from a peer device via PCIe BAR
         *
         * For PCIeBAR, this reads data from the peer's BAR-mapped memory region.
         * Only supports CUDA←ROCm or ROCm←CUDA transfers.
         *
         * @param buffer Destination buffer on this device
         * @param count Number of elements
         * @param dtype Data type
         * @param peer Source device rank (0=CUDA, 1=ROCm in 2-device mode)
         * @param tag Message tag (ignored for BAR transfers)
         * @return true on success
         */
        bool recv(void *buffer, size_t count, CollectiveDataType dtype,
                  int peer, int tag = 0) override;

        /**
         * @brief Bidirectional send-receive via PCIe BAR
         *
         * Performs simultaneous send and receive with a peer device.
         * For PCIeBAR, this means one device writes to and reads from the
         * other device's BAR-mapped memory.
         *
         * @param sendbuf Buffer to send from
         * @param recvbuf Buffer to receive into
         * @param count Number of elements (same both directions)
         * @param dtype Data type
         * @param peer Peer device rank
         * @return true on success
         */
        bool sendrecv(void *sendbuf, void *recvbuf, size_t count,
                      CollectiveDataType dtype, int peer) override;

        // =====================================================================
        // Async Point-to-Point Operations
        // =====================================================================

        /**
         * @brief Async send via PCIe BAR using caller-provided stream
         *
         * Issues an async memcpy to the peer's BAR-mapped region.
         * For CUDA→ROCm: Uses cudaMemcpyAsync to write to BAR.
         * For ROCm→CUDA: Uses hipMemcpyAsync (if supported).
         *
         * @param buffer Source buffer
         * @param count Number of elements
         * @param dtype Data type
         * @param peer Target device rank (0=CUDA, 1=ROCm typically)
         * @param stream GPU stream (cudaStream_t or hipStream_t)
         * @param tag Ignored for BAR transfers
         * @return true if operation was issued
         */
        bool sendAsync(void *buffer, size_t count, CollectiveDataType dtype,
                       int peer, void *stream, int tag = 0) override;

        /**
         * @brief Async receive via PCIe BAR using caller-provided stream
         *
         * Issues an async memcpy from the peer's BAR-mapped region.
         */
        bool recvAsync(void *buffer, size_t count, CollectiveDataType dtype,
                       int peer, void *stream, int tag = 0) override;

        /**
         * @brief Async bidirectional send-receive via PCIe BAR
         *
         * Issues both send and receive as async memcpy operations.
         */
        bool sendrecvAsync(void *sendbuf, void *recvbuf, size_t count,
                           CollectiveDataType dtype, int peer, void *stream) override;

        // =====================================================================
        // Synchronization
        // =====================================================================

        bool synchronize() override;

        // =====================================================================
        // Cross-Thread CUDA Event Synchronization
        // =====================================================================

        /**
         * @brief Wait for a CUDA event (direct call)
         *
         * This function establishes proper CUDA context and waits for the event.
         *
         * @param event CUDA event to wait for (cudaEvent_t)
         * @param device_id CUDA device ID
         * @return true if wait succeeded
         */
        bool waitForCUDAEvent(void *event, int device_id);

        /**
         * @brief Get the global PCIeBARBackend instance (if any)
         *
         * Used by TensorBase to access CUDA event wait functionality.
         */
        static PCIeBARBackend *getInstance() { return s_instance_; }

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

        /**
         * @brief Get BAR host pointer (mmap'd address accessible by both CPU and ROCm DMA)
         *
         * This pointer can be used as a staging area for cross-vendor transfers:
         * - hipMemcpy(D2D) can write to this address (uses AMD DMA engine)
         * - cudaMemcpy(D2D) can read from this address (via PCIe)
         *
         * @return Pointer to BAR region, or nullptr if not initialized
         */
        void *getBarHostPtr() const { return bar_host_ptr_; }

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

        // =====================================================================
        // Multi-Pair Collective Operations
        // =====================================================================

        /**
         * @brief AllReduce across all device pairs in parallel
         *
         * Performs bidirectional exchange for each pair simultaneously:
         * 1. For each pair: CUDA reads ROCm data via BAR
         * 2. For each pair: CUDA performs local reduction
         * 3. For each pair: CUDA writes result back to ROCm via BAR
         * 4. Synchronize all pairs
         *
         * @param cuda_buffers One buffer per pair (on respective CUDA devices)
         * @param rocm_buffers One buffer per pair (on respective ROCm devices)
         * @param count_per_pair Number of elements per buffer
         * @param dtype Data type of elements
         * @return true if all pair operations succeeded
         */
        bool allreduceMultiPair(
            std::vector<void *> &cuda_buffers,
            std::vector<void *> &rocm_buffers,
            size_t count_per_pair,
            CollectiveDataType dtype);

        // =====================================================================
        // Public CUDA Reduction (for Zero-Copy Allreduce)
        // =====================================================================

        /**
         * @brief Perform element-wise reduction on CUDA device
         *
         * Used by LocalTPContext::executePCIeBarAllreduce() for zero-copy path
         * where CUDA reads directly from ROCm's BAR region.
         *
         * @param output Output buffer (CUDA device memory)
         * @param input1 First input buffer (CUDA device memory) - typically same as output
         * @param input2 Second input buffer (can be BAR-mapped memory readable by CUDA)
         * @param count Number of elements
         * @param dtype Data type
         * @param op Collective operation (SUM supported)
         * @return true on success
         */
        bool reduceOnCUDA(
            void *output,
            const void *input1,
            const void *input2,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op);

        /**
         * @brief Get DirectP2PEngine for BAR memory management
         *
         * Provides access to the engine for BAR-backed tensor allocation.
         * Returns nullptr if engine is not initialized.
         *
         * @return Pointer to DirectP2PEngine, or nullptr if not available
         */
        DirectP2PEngine *getDirectP2PEngine() const
        {
            return p2p_engine_.get();
        }

    private:
        // Use shared_ptr to leverage the process-wide singleton
        // This prevents BAR resource cleanup/re-init issues between tests
        std::shared_ptr<DirectP2PEngine> p2p_engine_;
        bool initialized_ = false;

        // Device group info (set during initialize) - single-pair mode
        DeviceId cuda_device_;
        DeviceId rocm_device_;
        bool has_cuda_ = false;
        bool has_rocm_ = false;

        // Multi-pair mode support
        std::vector<DevicePair> device_pairs_;

        // Temp buffer on CUDA side for reduction operations
        void *cuda_temp_buffer_ = nullptr;
        size_t cuda_temp_buffer_size_ = 0;

        // Persistent CUDA stream for reduction operations (created once in initialize)
        void *cuda_reduction_stream_ = nullptr;

        // =====================================================================
        // Pipelined Transfer Infrastructure
        // =====================================================================

        /// Stream for BAR read operations (ROCm→CUDA)
        void *cuda_read_stream_ = nullptr;

        /// Stream for BAR write operations (CUDA→ROCm)
        void *cuda_write_stream_ = nullptr;

        /// Double buffer for pipelined transfers (second temp buffer)
        void *cuda_temp_buffer2_ = nullptr;

        /// Events for stream synchronization (one per ping-pong buffer)
        void *cuda_read_complete_event_[2] = {nullptr, nullptr};
        void *cuda_compute_complete_event_[2] = {nullptr, nullptr};

        /// Minimum size (bytes) to use pipelined allreduce
        static constexpr size_t PIPELINE_THRESHOLD = 32768; // 32KB

        /// Chunk size for pipelined transfers
        static constexpr size_t PIPELINE_CHUNK_SIZE = 16384; // 16KB

        // Performance metrics
        double measured_bandwidth_gbps_ = 0.0;

        // =====================================================================
        // CUDA Worker Thread for HIP-safe Event Waits
        // =====================================================================
        // When running heterogeneous CUDA+ROCm LOCAL TP, the ROCm executor thread
        // may need to wait for CUDA events. Direct cudaEventSynchronize() fails
        // with "context is destroyed" from a HIP-contaminated thread.
        // This worker thread provides a clean CUDA context for event waits.

        /**
         * @brief Start the CUDA worker thread
         */
        bool startCUDAWorker();

        /**
         * @brief Stop the CUDA worker thread
         */
        void stopCUDAWorker();

        /**
         * @brief Submit work to the CUDA worker thread
         */
        std::future<bool> submitCUDAWork(std::function<bool()> work);

        /**
         * @brief Main loop for CUDA worker thread
         */
        void cudaWorkerLoop();

        /// Worker thread for CUDA operations
        std::thread cuda_worker_thread_;

        /// Work queue for CUDA operations
        std::queue<std::packaged_task<bool()>> cuda_work_queue_;

        /// Mutex for work queue
        std::mutex cuda_work_mutex_;

        /// Condition variable for work queue
        std::condition_variable cuda_work_cv_;

        /// Flag to stop worker thread
        std::atomic<bool> cuda_worker_stop_{false};

        /// Flag indicating worker is running
        std::atomic<bool> cuda_worker_running_{false};

        /// CUDA stream for worker thread operations (created on worker thread)
        void *cuda_worker_stream_ = nullptr;

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
         * @brief Pipelined allreduce for large buffers
         *
         * Uses triple-buffering with overlapped transfers:
         * - Stream 1: Read from ROCm BAR
         * - Stream 2: Compute reduction
         * - Stream 3: Write to ROCm BAR
         *
         * @return true on success
         */
        bool allreducePipelined(
            void *buffer,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op);

        /**
         * @brief Pipelined allreduce with explicit BAR offset
         *
         * Version of allreducePipelined that takes the ROCm BAR offset explicitly,
         * for use with registered buffers that may not be at offset 0.
         */
        bool allreducePipelinedWithOffset(
            void *cuda_buffer,
            size_t rocm_bar_offset,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op);

        /**
         * @brief Async reduction kernel launch (no sync)
         */
        bool reduceOnCUDAAsync(
            void *output,
            const void *input,
            size_t count,
            CollectiveDataType dtype,
            void *stream);

        /**
         * @brief Async BAR transfer ROCm→CUDA
         */
        bool transferROCmtoCUDAAsync(size_t bar_offset, void *cuda_dst, size_t bytes, void *stream);

        /**
         * @brief Async BAR transfer CUDA→ROCm
         */
        bool transferCUDAtoROCmAsync(const void *cuda_src, size_t bar_offset, size_t bytes, void *stream);

        size_t datatypeSize(CollectiveDataType dtype) const;

        /// Global instance pointer for cross-thread event wait proxying
        static PCIeBARBackend *s_instance_;
    };

#else
    // Stub implementation when CUDA and ROCm are not both available

    /**
     * @brief Stub DevicePair (requires both CUDA and ROCm)
     */
    struct DevicePair
    {
        DeviceId cuda_device;
        DeviceId rocm_device;
        int pair_index = 0;

        bool operator==(const DevicePair &) const { return false; }
    };

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
        bool initializeMultiPair(const std::vector<DevicePair> &) { return false; }
        const std::vector<DevicePair> &getDevicePairs() const { return device_pairs_; }
        bool isMultiPairMode() const { return false; }
        void shutdown() override {}
        bool isInitialized() const override { return false; }

        bool allreduce(void *, size_t, CollectiveDataType, CollectiveOp) override { return false; }
        bool allgather(const void *, void *, size_t, CollectiveDataType) override { return false; }
        bool reduceScatter(const void *, void *, size_t, CollectiveDataType, CollectiveOp) override { return false; }
        bool broadcast(void *, size_t, CollectiveDataType, int) override { return false; }
        bool send(void *, size_t, CollectiveDataType, int, int = 0) override { return false; }
        bool recv(void *, size_t, CollectiveDataType, int, int = 0) override { return false; }
        bool sendrecv(void *, void *, size_t, CollectiveDataType, int) override { return false; }
        bool allreduceMultiPair(std::vector<void *> &, std::vector<void *> &, size_t, CollectiveDataType) { return false; }
        bool synchronize() override { return false; }

        double getMeasuredBandwidthGBps() const { return 0.0; }
        bool isPCIeBarActive() const { return false; }
        bool waitForCUDAEvent(void *, int) { return false; }
        static PCIeBARBackend *getInstance() { return nullptr; }

        std::optional<std::pair<void *, size_t>> allocateInBarRegion(size_t) { return std::nullopt; }
        void freeBarBuffer(void *) {}
        size_t getBarAllocOffset() const { return 0; }
        size_t getBarTotalMappedSize() const { return 0; }
        void *getBarHostPtr() const { return nullptr; }

        bool registerBuffer(const std::string &, DeviceId, void *, size_t) override { return false; }
        void unregisterBuffer(const std::string &, DeviceId) override {}
        std::optional<RegisteredBuffer> getBuffer(const std::string &, DeviceId) const override { return std::nullopt; }
        bool requiresBufferRegistration() const override { return false; }
        bool allreduceRegistered(const std::string &, size_t, CollectiveDataType, CollectiveOp) override { return false; }

    private:
        std::vector<DevicePair> device_pairs_;
    };

#endif // HAVE_CUDA && HAVE_ROCM

} // namespace llaminar2
