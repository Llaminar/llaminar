/**
 * @file CrossVendorP2P.h
 * @brief Cross-vendor (CUDA ↔ ROCm) P2P transfer infrastructure
 *
 * Implements optimized cross-vendor GPU transfers using dual-registered
 * host staging memory accessible by both CUDA and ROCm runtimes.
 *
 * ## Dual-Registration Approach
 *
 * Standard APIs don't support direct CUDA-ROCm P2P, but we discovered that
 * both runtimes can register the SAME host memory buffer:
 *
 *   1. Allocate page-aligned host memory with posix_memalign()
 *   2. Register with CUDA: cudaHostRegister() → cudaHostGetDevicePointer()
 *   3. Register with ROCm: hipHostRegister() → hipHostGetDevicePointer()
 *   4. Both GPUs now have DMA access to the same host buffer
 *
 * This enables zero-copy-ish transfers where data flows:
 *   CUDA GPU → PCIe → Host RAM → PCIe → ROCm GPU
 *
 * ## Pipelining Optimization
 *
 * With double-buffered staging and async streams, we overlap D2H and H2D:
 *
 *   Src GPU: [D2H buf0]───[D2H buf1]───[D2H buf0]───
 *   Dst GPU:      [H2D buf0]───[H2D buf1]───[H2D buf0]───
 *
 * This achieves throughput close to min(src_d2h, dst_h2d) rather than
 * harmonic_mean(src_d2h, dst_h2d) for serial transfers.
 *
 * ## Performance Notes
 *
 * - Optimal chunk size is typically 4-8 MB (enables good overlap)
 * - Cross-NUMA transfers are limited by QPI/UPI bandwidth (~2-3 GB/s)
 * - Same-NUMA transfers can achieve ~6+ GB/s with PCIe Gen3 x16
 * - Use auto_tune=true in config to find optimal chunk size for your system
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "backends/DeviceId.h"
#include <cstddef>
#include <memory>

namespace llaminar2
{

    /**
     * @brief Configuration for cross-vendor P2P transfers
     *
     * The engine uses dual-registered host memory that is accessible by both
     * CUDA and ROCm runtimes, enabling efficient GPU-to-GPU transfers through
     * pinned host staging buffers.
     *
     * Optimal chunk_size depends on system topology:
     * - Same NUMA node: Larger chunks (16-32 MB) may be better
     * - Cross-NUMA: Smaller chunks (4-8 MB) provide better pipelining overlap
     */
    struct CrossVendorP2PConfig
    {
        size_t buffer_size = 64 * 1024 * 1024; // 64 MB per buffer (2 buffers = 128 MB staging)
        size_t chunk_size = 4 * 1024 * 1024;   // 4 MB transfer chunks (optimal for pipelining)
        int num_buffers = 2;                   // Double buffering for D2H/H2D overlap
        bool enable_pipelining = true;         // Overlap D2H and H2D on different buffers
        bool auto_tune = false;                // Run calibration during initialize()
        bool allow_same_vendor = false;        // Allow same-vendor transfers (e.g., ROCm↔ROCm)
    };

    /**
     * @brief Result of a cross-vendor P2P transfer
     */
    struct CrossVendorP2PResult
    {
        double throughput_gbps = 0.0;    // Effective throughput
        double d2h_time_ms = 0.0;        // Total D2H time
        double h2d_time_ms = 0.0;        // Total H2D time
        double overlap_efficiency = 0.0; // How much D2H/H2D overlapped (0-1)
        size_t bytes_transferred = 0;
        size_t optimal_chunk_size = 0; // If auto_tune was used, the optimal chunk size
        bool success = false;
    };

#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
    // Full implementation when at least one GPU backend is available

    /**
     * @brief Cross-vendor P2P transfer engine
     *
     * Manages host staging buffers registered with both CUDA and ROCm
     * for optimized cross-vendor GPU-to-GPU transfers.
     */
    class CrossVendorP2PEngine
    {
    public:
        CrossVendorP2PEngine(const CrossVendorP2PConfig &config = {});
        ~CrossVendorP2PEngine();

        // Non-copyable, movable
        CrossVendorP2PEngine(const CrossVendorP2PEngine &) = delete;
        CrossVendorP2PEngine &operator=(const CrossVendorP2PEngine &) = delete;
        CrossVendorP2PEngine(CrossVendorP2PEngine &&) noexcept;
        CrossVendorP2PEngine &operator=(CrossVendorP2PEngine &&) noexcept;

        /**
         * @brief Initialize the engine for transfers between two devices
         *
         * Sets up staging buffers registered with both runtimes.
         * Must be called before transfer().
         *
         * @param src Source device (CUDA or ROCm)
         * @param dst Destination device (CUDA or ROCm, different vendor from src)
         * @return true if initialization succeeded
         */
        bool initialize(DeviceId src, DeviceId dst);

        /**
         * @brief Check if engine is initialized for a device pair
         */
        bool isInitialized() const { return initialized_; }

        /**
         * @brief Transfer data between GPU buffers across vendors
         *
         * Uses pipelined double-buffered transfers for optimal throughput.
         *
         * @param d_src Source device pointer
         * @param d_dst Destination device pointer
         * @param num_bytes Number of bytes to transfer
         * @return Transfer result with timing and throughput
         */
        CrossVendorP2PResult transfer(const void *d_src, void *d_dst, size_t num_bytes);

        /**
         * @brief Benchmark the transfer rate for this device pair
         *
         * Allocates test buffers and measures actual throughput.
         *
         * @param num_bytes Size of test transfer
         * @param iterations Number of iterations to average
         * @return Benchmark result
         */
        CrossVendorP2PResult benchmark(size_t num_bytes, int iterations = 5);

        /**
         * @brief Get the theoretical max throughput for initialized devices
         *
         * Based on individual D2H and H2D rates measured during init.
         */
        double theoreticalMaxGbps() const { return theoretical_max_gbps_; }

        /**
         * @brief Get the current chunk size being used
         */
        size_t chunkSize() const { return config_.chunk_size; }

        /**
         * @brief Set chunk size (for manual tuning)
         * @param chunk_size New chunk size in bytes
         */
        void setChunkSize(size_t chunk_size) { config_.chunk_size = chunk_size; }

    private:
        /**
         * @brief Auto-tune chunk size by testing different values
         *
         * Tests chunk sizes from 2MB to 32MB and selects the one with
         * highest throughput. Updates config_.chunk_size with the result.
         */
        void autoTuneChunkSize();

        struct Impl;
        std::unique_ptr<Impl> impl_;

        CrossVendorP2PConfig config_;
        DeviceId src_device_;
        DeviceId dst_device_;
        bool initialized_ = false;
        double theoretical_max_gbps_ = 0.0;
    };

    /**
     * @brief Static helper functions for cross-vendor P2P
     *
     * These can be used without creating an engine instance.
     */
    class CrossVendorP2PHelper
    {
    public:
        /**
         * @brief Check if cross-vendor P2P is possible between two devices
         *
         * Returns true if devices are different vendors (CUDA vs ROCm).
         * Direct P2P isn't possible, but optimized host-staged transfer is.
         */
        static bool canTransfer(DeviceId src, DeviceId dst);

        /**
         * @brief Check if P2P transfer is possible (with same-vendor option)
         *
         * @param src Source device
         * @param dst Destination device
         * @param allow_same_vendor If true, also allows same-vendor P2P (e.g., ROCm↔ROCm)
         * @return true if transfer is possible
         */
        static bool canTransfer(DeviceId src, DeviceId dst, bool allow_same_vendor);

        /**
         * @brief Estimate transfer time for a given size
         *
         * @param src Source device
         * @param dst Destination device
         * @param num_bytes Bytes to transfer
         * @param pipelined Whether pipelining is enabled
         * @return Estimated time in milliseconds
         */
        static double estimateTransferTimeMs(DeviceId src, DeviceId dst,
                                             size_t num_bytes, bool pipelined = true);

        /**
         * @brief Quick benchmark between two devices
         *
         * Simpler than creating an engine - useful for one-off measurements.
         */
        static CrossVendorP2PResult quickBenchmark(DeviceId src, DeviceId dst,
                                                   size_t num_bytes = 64 * 1024 * 1024);
    };

#else
    // Stub implementations when no GPU backends are available

    /**
     * @brief Stub CrossVendorP2PEngine (no GPU backends available)
     */
    class CrossVendorP2PEngine
    {
    public:
        explicit CrossVendorP2PEngine(const CrossVendorP2PConfig & = {}) {}
        ~CrossVendorP2PEngine() = default;

        CrossVendorP2PEngine(const CrossVendorP2PEngine &) = delete;
        CrossVendorP2PEngine &operator=(const CrossVendorP2PEngine &) = delete;
        CrossVendorP2PEngine(CrossVendorP2PEngine &&) noexcept = default;
        CrossVendorP2PEngine &operator=(CrossVendorP2PEngine &&) noexcept = default;

        bool initialize(DeviceId, DeviceId) { return false; }
        bool isInitialized() const { return false; }
        CrossVendorP2PResult transfer(const void *, void *, size_t) { return {}; }
        CrossVendorP2PResult benchmark(size_t, int = 5) { return {}; }
        double theoreticalMaxGbps() const { return 0.0; }
        size_t chunkSize() const { return 0; }
        void setChunkSize(size_t) {}
    };

    /**
     * @brief Stub CrossVendorP2PHelper (no GPU backends available)
     */
    class CrossVendorP2PHelper
    {
    public:
        static bool canTransfer(DeviceId, DeviceId) { return false; }
        static bool canTransfer(DeviceId, DeviceId, bool) { return false; }
        static double estimateTransferTimeMs(DeviceId, DeviceId, size_t, bool = true) { return 0.0; }
        static CrossVendorP2PResult quickBenchmark(DeviceId, DeviceId, size_t = 64 * 1024 * 1024) { return {}; }
    };

#endif // HAVE_CUDA || HAVE_ROCM

} // namespace llaminar2
