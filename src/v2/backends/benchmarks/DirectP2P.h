/**
 * @file DirectP2P.h
 * @brief Direct cross-vendor GPU P2P via PCIe BAR mapping
 *
 * Implements true zero-copy P2P between CUDA and ROCm devices using PCIe BAR mapping.
 * With resizable BAR enabled, both NVIDIA and AMD GPUs expose 32GB BARs for full VRAM access.
 *
 * ## Benchmarked Performance (RTX 3090 + MI50, PCIe 3.0)
 *
 * ### With Resizable BAR Enabled (RECOMMENDED)
 *
 * | Direction         | Method              | Speed     | Notes                        |
 * |-------------------|---------------------|-----------|------------------------------|
 * | NVIDIA → AMD      | CUDA writes to BAR  | 2.85 GB/s | PCIe posted writes           |
 * | AMD → NVIDIA      | CUDA reads from BAR | 2.86 GB/s | SYMMETRIC with rBAR!         |
 * | NVIDIA → 2x AMD   | Concurrent writes   | 2.85 GB/s | Shared PCIe bandwidth        |
 * | 2x AMD → NVIDIA   | Concurrent reads    | 2.86 GB/s | Also symmetric               |
 * | Mixed read+write  | Overlapped streams  | 5.62 GB/s | Full bidirectional PCIe!     |
 *
 * ### Without Resizable BAR (legacy 256MB BAR1)
 *
 * | Direction         | Method              | Speed     | Notes                        |
 * |-------------------|---------------------|-----------|------------------------------|
 * | NVIDIA → AMD      | CUDA writes to BAR  | 2.65 GB/s | PCIe posted writes (fast)    |
 * | AMD → NVIDIA      | CUDA reads from BAR | 0.79 GB/s | PCIe read completion (slow)  |
 * | Mixed read+write  | Overlapped streams  | 1.58 GB/s | 30% improvement from overlap |
 *
 * ## Key Insight: Resizable BAR Eliminates Asymmetry
 *
 * Without rBAR, PCIe has a fundamental 3.4x asymmetry between writes and reads:
 * - **Posted writes**: Fire-and-forget, no response needed (~2.65 GB/s)
 * - **Non-posted reads**: Requires completion packet (~0.79 GB/s)
 *
 * With rBAR enabled on NVIDIA (32GB BAR1), reads become as fast as writes!
 * This enables symmetric bidirectional transfers at full PCIe bandwidth.
 *
 * **Optimal Strategy**: Enable Resizable BAR in BIOS for both GPUs. NVIDIA must
 * initiate all transfers (HIP cannot register NVIDIA BAR as IoMemory).
 *
 * ## Implementation
 *
 *   1. Discover GPU PCIe BARs via sysfs (/sys/bus/pci/.../resource0)
 *   2. mmap() the BAR with O_SYNC for uncached access
 *   3. Register with CUDA using IOMEMORY flag:
 *      `cuMemHostRegister(bar_ptr, size, CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_IOMEMORY)`
 *   4. Get CUDA device pointer via cuMemHostGetDevicePointer()
 *   5. Use async cudaMemcpyAsync with multiple streams for overlap
 *
 * ## Requirements
 *
 * - CAP_SYS_ADMIN capability (for CUDA IOMEMORY registration)
 * - Resizable BAR enabled in BIOS (for symmetric performance + full VRAM access)
 * - AMD/NVIDIA GPUs with large BAR support (32GB for MI50, RTX 3090)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "backends/DeviceId.h"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
    // Full implementation when at least one GPU backend is available

    /**
     * @brief Information about a discovered PCIe BAR
     */
    struct PCIeBarInfo
    {
        std::string pci_address;         // e.g., "0000:b1:00.0"
        std::string sysfs_path;          // e.g., "/sys/bus/pci/devices/0000:b1:00.0/resource0"
        size_t bar_size = 0;             // Size in bytes (e.g., 32GB for MI50/RTX 3090)
        int bar_fd = -1;                 // File descriptor when opened
        void *mapped_ptr = nullptr;      // mmap'd pointer
        size_t mapped_size = 0;          // Actually mapped size
        bool is_amd = false;             // True if this is an AMD GPU
        bool is_nvidia = false;          // True if this is an NVIDIA GPU
        int device_ordinal = -1;         // Device ordinal (ROCm or CUDA)
        void *cuda_device_ptr = nullptr; // CUDA-accessible pointer (after registration)
    };

    /**
     * @brief Result of a direct P2P capability probe
     */
    struct DirectP2PCapability
    {
        // DMA-BUF capabilities (often fail in practice)
        bool dmabuf_export_cuda = false; // CUDA can export DMA-BUF handles
        bool dmabuf_export_rocm = false; // ROCm can export DMA-BUF handles
        bool dmabuf_import_cuda = false; // CUDA can import external DMA-BUF
        bool dmabuf_import_rocm = false; // ROCm can import external DMA-BUF

        // PCIe BAR capabilities (the working path!)
        bool pcie_bar_accessible = false;         // Can access GPU BAR directly (need root)
        bool pcie_bar_iomemory_supported = false; // CUDA supports IOMEMORY registration
        std::vector<PCIeBarInfo> discovered_bars; // Found AMD GPU BARs

        bool iommu_enabled = false; // IOMMU may block direct access

        std::string cuda_driver_version;
        std::string rocm_driver_version;
        std::string kernel_version;

        bool canDoDirectP2P() const
        {
            // PCIe BAR is the reliable path
            if (pcie_bar_accessible && pcie_bar_iomemory_supported && !discovered_bars.empty())
                return true;
            // DMA-BUF is experimental fallback
            return (dmabuf_export_cuda && dmabuf_import_rocm) ||
                   (dmabuf_export_rocm && dmabuf_import_cuda);
        }

        bool canDoPCIeBarP2P() const
        {
            return pcie_bar_accessible && pcie_bar_iomemory_supported && !discovered_bars.empty();
        }

        std::string describe() const;
    };

    /**
     * @brief Result of a direct P2P transfer attempt
     */
    struct DirectP2PResult
    {
        double throughput_gbps = 0.0;
        double transfer_time_ms = 0.0;
        size_t bytes_transferred = 0;
        bool success = false;
        bool used_dmabuf = false;         // True if DMA-BUF path succeeded
        bool used_pcie_bar = false;       // True if PCIe BAR mapping path succeeded
        bool fell_back_to_staged = false; // True if had to use host staging
        bool used_overlap = false;        // True if overlapped read+write for speedup

        // Direction-specific metrics for PCIe BAR
        double read_gbps = 0.0;       // AMD→NVIDIA (read from BAR) ~0.79 GB/s
        double write_gbps = 0.0;      // NVIDIA→AMD (write to BAR) ~2.65 GB/s
        double concurrent_gbps = 0.0; // Overlapped operations ~1.58 GB/s effective

        // Multi-GPU metrics
        double dual_write_gbps = 0.0; // Writing to 2 AMD GPUs simultaneously
        double dual_read_gbps = 0.0;  // Reading from 2 AMD GPUs simultaneously

        std::string error_message;
        std::string transfer_path; // Description of path used
    };

    /**
     * @brief Shared buffer handle for cross-vendor access
     */
    struct DirectP2PBuffer
    {
        void *device_ptr = nullptr; // Native device pointer
        int dmabuf_fd = -1;         // Linux DMA-BUF file descriptor
        size_t size = 0;
        DeviceId owner_device;
        bool is_exported = false;

        ~DirectP2PBuffer();

        // Non-copyable
        DirectP2PBuffer() = default;
        DirectP2PBuffer(const DirectP2PBuffer &) = delete;
        DirectP2PBuffer &operator=(const DirectP2PBuffer &) = delete;
        DirectP2PBuffer(DirectP2PBuffer &&) noexcept;
        DirectP2PBuffer &operator=(DirectP2PBuffer &&) noexcept;
    };

    /**
     * @brief EXPERIMENTAL: Direct cross-vendor P2P engine
     *
     * Attempts to establish true zero-copy P2P between CUDA and ROCm GPUs
     * using PCIe BAR mapping (preferred) or Linux DMA-BUF (experimental).
     *
     * ## Usage (PCIe BAR - Recommended)
     *
     * ```cpp
     * DirectP2PEngine engine;
     * // Initialize with all AMD GPUs for maximum parallelism
     * if (engine.initializePCIeBarMultiGPU(cuda_device, {rocm_device0, rocm_device1})) {
     *     // Tensor parallel: broadcast weights to all AMD GPUs
     *     engine.broadcastToAMD(cuda_weights, size);  // ~2.65 GB/s
     *
     *     // Gather activations from AMD GPUs with overlap
     *     engine.gatherFromAMDOverlapped(cuda_activations, size);  // ~1.58 GB/s
     * }
     * ```
     *
     * ## Direction Asymmetry (Measured)
     *
     * PCIe BAR transfers have 3.4x asymmetric performance:
     * - **NVIDIA→AMD (write to BAR)**: 2.65 GB/s - PCIe posted writes
     * - **AMD→NVIDIA (read from BAR)**: 0.79 GB/s - PCIe read latency
     * - **Concurrent read+write**: 1.58 GB/s - 30% speedup from overlap
     */
    class DirectP2PEngine
    {
    public:
        /**
         * @brief Direction of PCIe BAR transfer
         */
        enum class Direction
        {
            ToNVIDIA, // AMD → NVIDIA (read from BAR)
            ToAMD     // NVIDIA → AMD (write to BAR)
        };

        DirectP2PEngine();
        ~DirectP2PEngine();

        /**
         * @brief Probe system capabilities for direct P2P
         *
         * Checks driver versions, kernel support, and hardware capabilities.
         * Discovers PCIe BARs for AMD GPUs.
         */
        static DirectP2PCapability probeCapabilities();

        //----------------------------------------------------------------------
        // PCIe BAR Direct P2P (RECOMMENDED - Actually works!)
        //----------------------------------------------------------------------

        /**
         * @brief Initialize PCIe BAR-based direct P2P
         *
         * Maps AMD GPU's PCIe BAR and registers it with CUDA for DMA access.
         * **Requires root/sudo privileges.**
         *
         * @param cuda_device NVIDIA GPU device
         * @param rocm_device AMD GPU device
         * @param bar_offset Offset into the BAR to map (usually 0)
         * @param map_size Size to map (0 = auto-detect from BAR)
         * @return true if PCIe BAR P2P was established
         */
        bool initializePCIeBar(DeviceId cuda_device, DeviceId rocm_device,
                               size_t bar_offset = 0, size_t map_size = 0);

        /**
         * @brief Check if PCIe BAR P2P is active
         */
        bool isPCIeBarActive() const { return pcie_bar_active_; }

        /**
         * @brief Get the CUDA-accessible pointer to AMD BAR region
         *
         * This pointer can be used directly in CUDA memcpy operations
         * to transfer data to/from AMD GPU memory.
         */
        void *getCudaBarPointer() const;

        /**
         * @brief Get the size of the mapped BAR region
         */
        size_t getBarMappedSize() const;

        /**
         * @brief Get the host-mapped pointer to the BAR region
         *
         * This pointer is directly accessible by ROCm since it maps
         * to AMD GPU memory. Used for BAR region allocation.
         *
         * @return Host pointer to mmap'd BAR region, or nullptr if not initialized
         */
        void *getBarHostPtr() const;

        /**
         * @brief Get the BAR offset used during initialization
         *
         * @return The bar_offset parameter passed to initializePCIeBar()
         */
        size_t getBarOffset() const;

        /**
         * @brief Transfer data via PCIe BAR
         *
         * For ToNVIDIA: Copies from BAR[bar_offset] to cuda_ptr
         * For ToAMD: Copies from cuda_ptr to BAR[bar_offset]
         *
         * @param cuda_ptr Pointer to CUDA device memory
         * @param bar_offset Offset into BAR region
         * @param num_bytes Number of bytes to transfer
         * @param direction Transfer direction
         * @return Transfer result with timing
         */
        DirectP2PResult transferViaPCIeBar(void *cuda_ptr, size_t bar_offset,
                                           size_t num_bytes, Direction direction);

        /**
         * @brief Benchmark PCIe BAR transfer in both directions
         *
         * @param num_bytes Size of transfer to benchmark
         * @param iterations Number of iterations (default 5)
         * @return Result with read_gbps and write_gbps populated
         */
        DirectP2PResult benchmarkPCIeBar(size_t num_bytes, int iterations = 5);

        //----------------------------------------------------------------------
        // Multi-GPU and Concurrent Transfer APIs
        //----------------------------------------------------------------------

        /**
         * @brief Initialize PCIe BAR P2P with multiple AMD GPUs
         *
         * Enables optimal tensor parallel strategies:
         * - Broadcast from NVIDIA to all AMD GPUs (~2.65 GB/s per target)
         * - Gather from all AMD GPUs with overlap (~1.58 GB/s effective)
         *
         * @param cuda_device NVIDIA GPU device
         * @param rocm_devices List of AMD GPU devices
         * @param map_size Size to map per BAR (0 = default 64MB)
         * @return true if all BARs were successfully mapped
         */
        bool initializePCIeBarMultiGPU(DeviceId cuda_device,
                                       const std::vector<DeviceId> &rocm_devices,
                                       size_t map_size = 0);

        /**
         * @brief Get number of AMD GPU BARs mapped
         */
        size_t getNumMappedBars() const;

        /**
         * @brief Get CUDA-accessible pointer for a specific AMD GPU BAR
         * @param rocm_ordinal AMD GPU device ordinal
         * @return CUDA device pointer to that GPU's BAR (nullptr if not mapped)
         */
        void *getCudaBarPointerForDevice(int rocm_ordinal) const;

        /**
         * @brief Broadcast data from NVIDIA to all mapped AMD GPUs
         *
         * Uses fast posted writes (~2.65 GB/s). For tensor parallel weight loading.
         * Concurrent writes to multiple BARs share PCIe bandwidth but provide
         * simpler programming model than sequential transfers.
         *
         * @param cuda_src Source pointer in NVIDIA GPU memory
         * @param num_bytes Number of bytes to broadcast
         * @param stream CUDA stream (nullptr for default stream)
         * @return Result with transfer statistics
         */
        DirectP2PResult broadcastToAMD(const void *cuda_src, size_t num_bytes,
                                       void *stream = nullptr);

        /**
         * @brief Gather data from all AMD GPUs to NVIDIA with overlap
         *
         * Uses read operations (~0.79 GB/s per BAR) but overlaps them for
         * ~30% improvement. For tensor parallel activation gathering.
         *
         * @param cuda_dst Destination pointer in NVIDIA GPU memory
         * @param num_bytes Number of bytes to gather per GPU
         * @param stream CUDA stream (nullptr for default stream)
         * @return Result with transfer statistics
         */
        DirectP2PResult gatherFromAMDOverlapped(void *cuda_dst, size_t num_bytes,
                                                void *stream = nullptr);

        /**
         * @brief Transfer with concurrent read+write overlap
         *
         * Simultaneously performs:
         * - Read from one AMD BAR (AMD→NVIDIA)
         * - Write to another AMD BAR (NVIDIA→AMD)
         *
         * Achieves ~1.58 GB/s effective (30% faster than sequential).
         *
         * @param cuda_read_dst Destination for read (nullptr to skip)
         * @param read_bar_offset BAR offset to read from
         * @param read_bytes Bytes to read
         * @param cuda_write_src Source for write (nullptr to skip)
         * @param write_bar_offset BAR offset to write to
         * @param write_bytes Bytes to write
         * @return Result with concurrent_gbps populated
         */
        DirectP2PResult transferOverlapped(void *cuda_read_dst, size_t read_bar_offset, size_t read_bytes,
                                           const void *cuda_write_src, size_t write_bar_offset, size_t write_bytes);

        /**
         * @brief Full benchmark including multi-GPU and concurrent operations
         *
         * Tests all transfer modes:
         * - Single direction read/write
         * - Concurrent read+write overlap
         * - Dual BAR parallel operations
         */
        DirectP2PResult benchmarkAllModes(size_t num_bytes, int iterations = 5);

        //----------------------------------------------------------------------
        // DMA-BUF Direct P2P (Experimental - Often Fails)
        //----------------------------------------------------------------------

        /**
         * @brief Initialize DMA-BUF-based direct P2P (experimental)
         *
         * @param device_a First device (CUDA or ROCm)
         * @param device_b Second device (different vendor)
         * @return true if direct P2P was established
         */
        bool initialize(DeviceId device_a, DeviceId device_b);

        /**
         * @brief Check if DMA-BUF direct P2P is active
         */
        bool isDirectP2PActive() const { return direct_p2p_active_; }

        /**
         * @brief Allocate a buffer exportable for cross-vendor access
         *
         * @param device Device to allocate on
         * @param size Buffer size in bytes
         * @return Shared buffer handle (empty on failure)
         */
        std::unique_ptr<DirectP2PBuffer> allocateExportable(DeviceId device, size_t size);

        /**
         * @brief Import a buffer from another vendor's device
         *
         * @param buffer Source buffer (must be exported)
         * @param target_device Device to import to (different vendor)
         * @return Device pointer on target device (nullptr on failure)
         */
        void *importBuffer(const DirectP2PBuffer &buffer, DeviceId target_device);

        /**
         * @brief Attempt direct P2P copy between devices
         *
         * Tries PCIe BAR first, then DMA-BUF, then host staging fallback.
         *
         * @param src_device Source device
         * @param src_ptr Source pointer (on src_device)
         * @param dst_device Destination device
         * @param dst_ptr Destination pointer (on dst_device)
         * @param num_bytes Bytes to copy
         * @return Transfer result
         */
        DirectP2PResult transfer(DeviceId src_device, const void *src_ptr,
                                 DeviceId dst_device, void *dst_ptr,
                                 size_t num_bytes);

        /**
         * @brief Benchmark direct P2P between initialized devices
         */
        DirectP2PResult benchmark(size_t num_bytes, int iterations = 5);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;

        DeviceId device_a_;
        DeviceId device_b_;
        bool direct_p2p_active_ = false;
        bool pcie_bar_active_ = false;
    };

#else
    // Stub implementations when no GPU backends are available

    struct PCIeBarInfo
    {
    };

    struct DirectP2PCapability
    {
        bool canDoDirectP2P() const { return false; }
        bool canDoPCIeBarP2P() const { return false; }
        std::string describe() const { return "No GPU backends available"; }
    };

    struct DirectP2PResult
    {
        double throughput_gbps = 0.0;
        double transfer_time_ms = 0.0;
        size_t bytes_transferred = 0;
        bool success = false;
        std::string error_message = "No GPU backends available";
        std::string transfer_path;
    };

    struct DirectP2PBuffer
    {
        void *device_ptr = nullptr;
        int dmabuf_fd = -1;
        size_t size = 0;
        DeviceId owner_device;
        bool is_exported = false;
    };

    class DirectP2PEngine
    {
    public:
        enum class Direction
        {
            ToNVIDIA,
            ToAMD
        };

        DirectP2PEngine() = default;
        ~DirectP2PEngine() = default;

        static DirectP2PCapability probeCapabilities() { return {}; }
        static std::vector<PCIeBarInfo> discoverBars() { return {}; }

        bool initializePCIeBar(DeviceId, DeviceId, size_t = 0) { return false; }
        bool initializePCIeBarMultiGPU(DeviceId, const std::vector<DeviceId> &, size_t = 0) { return false; }
        bool isPCIeBarActive() const { return false; }
        std::vector<size_t> getMappedBarSizes() const { return {}; }
        size_t getMappedBarSize(int = 0) const { return 0; }
        DirectP2PResult transferPCIeBar(Direction, const void *, void *, size_t, int = 0) { return {}; }
        DirectP2PResult transferPCIeBarAsync(Direction, const void *, void *, size_t, int = 0) { return {}; }
        DirectP2PResult broadcastToAMD(const void *, size_t) { return {}; }
        DirectP2PResult gatherFromAMDSequential(void *, size_t) { return {}; }
        DirectP2PResult gatherFromAMDOverlapped(void *, size_t) { return {}; }
        DirectP2PResult transferOverlapped(void *, size_t, size_t, const void *, size_t, size_t) { return {}; }
        DirectP2PResult benchmarkAllModes(size_t, int = 5) { return {}; }
        bool initialize(DeviceId, DeviceId) { return false; }
        bool isDirectP2PActive() const { return false; }
        std::unique_ptr<DirectP2PBuffer> allocateExportable(DeviceId, size_t) { return nullptr; }
        void *importBuffer(const DirectP2PBuffer &, DeviceId) { return nullptr; }
        DirectP2PResult transfer(DeviceId, const void *, DeviceId, void *, size_t) { return {}; }
        DirectP2PResult benchmark(size_t, int = 5) { return {}; }
    };

#endif // HAVE_CUDA || HAVE_ROCM

} // namespace llaminar2
