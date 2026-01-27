/**
 * @file DirectP2P.h
 * @brief Direct cross-vendor GPU P2P via PCIe BAR mapping
 *
 * Implements true zero-copy P2P between CUDA and ROCm devices using PCIe BAR mapping.
 * With resizable BAR enabled, both NVIDIA and AMD GPUs expose 32GB BARs for full VRAM access.
 *
 * ## Benchmarked Performance (RTX 3090 + MI50, PCIe 3.0)
 *
 * | Direction         | Method              | Speed     | Notes                        |
 * |-------------------|---------------------|-----------|------------------------------|
 * | NVIDIA → AMD      | CUDA writes to BAR  | 2.65 GB/s | PCIe posted writes           |
 * | AMD → NVIDIA      | CUDA reads from BAR | 2.67 GB/s | SYMMETRIC with rBAR enabled  |
 * | Mixed read+write  | Overlapped streams  | ~5.3 GB/s | Full bidirectional PCIe      |
 *
 * ### Without Resizable BAR (legacy 256MB BAR)
 *
 * | Direction         | Method              | Speed     | Notes                        |
 * |-------------------|---------------------|-----------|------------------------------|
 * | NVIDIA → AMD      | CUDA writes to BAR  | ~2.6 GB/s | PCIe posted writes (fast)    |
 * | AMD → NVIDIA      | CUDA reads from BAR | ~0.8 GB/s | PCIe read completion (slow)  |
 *
 * **Note**: With Resizable BAR enabled, reads become symmetric with writes.
 *
 * ## Key Insight: Resizable BAR Eliminates Read/Write Asymmetry
 *
 * Without rBAR, PCIe has significant asymmetry between writes and reads:
 * - **Posted writes**: Fire-and-forget, no response needed (~2.6 GB/s)
 * - **Non-posted reads**: Requires completion packet (~0.8 GB/s)
 *
 * With Resizable BAR enabled, reads become nearly as fast as writes (~2.67 vs 2.65 GB/s).
 *
 * **Important**: Only NVIDIA can initiate transfers because HIP/ROCm cannot register
 * NVIDIA's BAR as IoMemory. All cuMemcpy operations originate from CUDA.
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

#include "../DeviceId.h"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
    // Full implementation requires both CUDA and ROCm backends

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
        // PCIe BAR capabilities
        bool pcie_bar_accessible = false;         // Can access GPU BAR directly (need root)
        bool pcie_bar_iomemory_supported = false; // CUDA supports IOMEMORY registration
        std::vector<PCIeBarInfo> discovered_bars; // Found AMD GPU BARs

        bool iommu_enabled = false; // IOMMU may block direct access

        std::string cuda_driver_version;
        std::string rocm_driver_version;
        std::string kernel_version;

        bool canDoDirectP2P() const
        {
            return pcie_bar_accessible && pcie_bar_iomemory_supported && !discovered_bars.empty();
        }

        // Alias for backward compatibility with code that used the more specific name
        bool canDoPCIeBarP2P() const { return canDoDirectP2P(); }

        std::string describe() const;
    };

    /**
     * @brief Result of a direct P2P transfer
     */
    struct DirectP2PResult
    {
        double throughput_gbps = 0.0;
        double transfer_time_ms = 0.0;
        size_t bytes_transferred = 0;
        bool success = false;
        bool used_overlap = false; // True if overlapped read+write for speedup

        // Direction-specific metrics for PCIe BAR
        double read_gbps = 0.0;       // AMD→NVIDIA (read from BAR)
        double write_gbps = 0.0;      // NVIDIA→AMD (write to BAR)
        double concurrent_gbps = 0.0; // Overlapped operations

        // Multi-GPU metrics
        double dual_write_gbps = 0.0; // Writing to 2 AMD GPUs simultaneously
        double dual_read_gbps = 0.0;  // Reading from 2 AMD GPUs simultaneously

        std::string error_message;
        std::string transfer_path; // Description of path used
    };

    /**
     * @brief Direct cross-vendor P2P engine via PCIe BAR mapping
     *
     * Establishes true zero-copy P2P between CUDA and ROCm GPUs
     * using PCIe BAR mapping. No host memory staging - data flows
     * directly over the PCIe bus between GPUs.
     *
     * ## Usage
     *
     * ```cpp
     * DirectP2PEngine engine;
     * // Initialize with all AMD GPUs for maximum parallelism
     * if (engine.initializePCIeBarMultiGPU(cuda_device, {rocm_device0, rocm_device1})) {
     *     // Tensor parallel: broadcast weights to all AMD GPUs
     *     engine.broadcastToAMD(cuda_weights, size);  // ~2.85 GB/s with rBAR
     *
     *     // Gather activations from AMD GPUs with overlap
     *     engine.gatherFromAMDOverlapped(cuda_activations, size);
     * } else {
     *     // PCIe BAR P2P not available - handle error
     * }
     * ```
     *
     * ## Transfer Performance
     *
     * Typical PCIe 3.0 x16 performance with Resizable BAR:
     * - **NVIDIA→AMD (write to BAR)**: ~2.65 GB/s - PCIe posted writes
     * - **AMD→NVIDIA (read from BAR)**: ~2.67 GB/s - Symmetric with rBAR
     *
     * Without rBAR, reads are ~3x slower than writes due to PCIe protocol.
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
         * @brief Get the shared singleton instance of DirectP2PEngine
         *
         * The DirectP2PEngine manages CUDA IOMEMORY registrations and BAR mappings
         * which cannot be reliably re-initialized after cleanup within a process.
         * Using a singleton ensures that:
         * 1. BAR resources are initialized once and shared
         * 2. Tests don't conflict by creating/destroying separate engines
         * 3. CUDA primary context state remains stable
         *
         * The singleton is lazily initialized on first call and lives until
         * process exit.
         *
         * @return Shared pointer to the singleton engine
         */
        static std::shared_ptr<DirectP2PEngine> getSharedInstance();

        /**
         * @brief Probe system capabilities for direct P2P
         *
         * Checks driver versions, kernel support, and hardware capabilities.
         * Discovers PCIe BARs for AMD GPUs.
         */
        static DirectP2PCapability probeCapabilities();

        //----------------------------------------------------------------------
        // Single-GPU PCIe BAR P2P
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
         * @param map_size Size to map (0 = default 64MB)
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
         * to AMD GPU memory.
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
         * Enables tensor parallel weight distribution and activation gathering
         * across multiple AMD GPUs. With rBAR, achieves ~2.65 GB/s per target.
         *
         * @param cuda_device NVIDIA GPU device
         * @param rocm_devices List of AMD GPU devices
         * @param map_size Size to map per BAR (0 = default 64MB)
         * @return true if at least one BAR was successfully mapped
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
         * Uses read operations from AMD BARs. Performance depends on rBAR:
         * - With rBAR: ~2.67 GB/s (symmetric with writes)
         * - Without rBAR: ~0.8 GB/s (PCIe read latency)
         *
         * For tensor parallel activation gathering.
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
         * With rBAR enabled, achieves ~5.3 GB/s aggregate bidirectional bandwidth.
         * Without rBAR, reads are slower so benefit from overlap is more modest.
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

        /**
         * @brief Get the cached measured bandwidth (GB/s)
         *
         * Returns the bandwidth measured during initialization, or 0 if
         * the engine hasn't been benchmarked yet.
         *
         * @return Measured bandwidth in GB/s
         */
        double getCachedBandwidthGBps() const { return cached_bandwidth_gbps_; }

        /**
         * @brief Set the cached measured bandwidth
         *
         * Called after benchmarking during initialization.
         *
         * @param bandwidth Measured bandwidth in GB/s
         */
        void setCachedBandwidthGBps(double bandwidth) { cached_bandwidth_gbps_ = bandwidth; }

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;

        bool pcie_bar_active_ = false;
        double cached_bandwidth_gbps_ = 0.0;
    };

#else
    // Stub implementations when both GPU backends are not available

    struct PCIeBarInfo
    {
    };

    struct DirectP2PCapability
    {
        bool canDoDirectP2P() const { return false; }
        std::string describe() const { return "DirectP2P requires both CUDA and ROCm backends"; }
    };

    struct DirectP2PResult
    {
        double throughput_gbps = 0.0;
        double transfer_time_ms = 0.0;
        size_t bytes_transferred = 0;
        bool success = false;
        std::string error_message = "DirectP2P requires both CUDA and ROCm backends";
        std::string transfer_path;
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

        bool initializePCIeBar(DeviceId, DeviceId, size_t = 0, size_t = 0) { return false; }
        bool initializePCIeBarMultiGPU(DeviceId, const std::vector<DeviceId> &, size_t = 0) { return false; }
        bool isPCIeBarActive() const { return false; }
        void *getCudaBarPointer() const { return nullptr; }
        size_t getBarMappedSize() const { return 0; }
        void *getBarHostPtr() const { return nullptr; }
        size_t getBarOffset() const { return 0; }
        size_t getNumMappedBars() const { return 0; }
        void *getCudaBarPointerForDevice(int) const { return nullptr; }
        DirectP2PResult transferViaPCIeBar(void *, size_t, size_t, Direction) { return {}; }
        DirectP2PResult benchmarkPCIeBar(size_t, int = 5) { return {}; }
        DirectP2PResult broadcastToAMD(const void *, size_t, void * = nullptr) { return {}; }
        DirectP2PResult gatherFromAMDOverlapped(void *, size_t, void * = nullptr) { return {}; }
        DirectP2PResult transferOverlapped(void *, size_t, size_t, const void *, size_t, size_t) { return {}; }
        DirectP2PResult benchmarkAllModes(size_t, int = 5) { return {}; }
    };

#endif // HAVE_CUDA && HAVE_ROCM

} // namespace llaminar2
