/**
 * @file CPUBackend.h
 * @brief CPU/NUMA backend implementing IBackend interface
 *
 * CPUBackend provides a rank-local view of system memory, treating each
 * MPI rank's NUMA node as a "device" for unified memory query/allocation.
 *
 * **Design**: Rank-local view where each MPI rank creates its own CPUBackend
 * with its local NUMA node. This enables unified memory management across
 * heterogeneous compute backends (CPU, CUDA, ROCm).
 *
 * ```
 * 2-socket machine with 2 MPI ranks:
 * ┌─────────────────────────────────┬─────────────────────────────────┐
 * │           Rank 0                │           Rank 1                │
 * ├─────────────────────────────────┼─────────────────────────────────┤
 * │ CPUBackend(numa_node=0)         │ CPUBackend(numa_node=1)         │
 * │ deviceCount() → 1               │ deviceCount() → 1               │
 * │ deviceMemoryTotal(0) → socket0  │ deviceMemoryTotal(0) → socket1  │
 * │ deviceMemoryFree(0) → socket0   │ deviceMemoryFree(0) → socket1   │
 * └─────────────────────────────────┴─────────────────────────────────┘
 * ```
 *
 * @author David Sanftenberg
 */

#pragma once

#include "IBackend.h"
#include <future>
#include <memory>

namespace llaminar2
{

    /**
     * @brief CPU/NUMA backend implementing IBackend interface
     *
     * Provides rank-local memory management for CPU execution, with NUMA-aware
     * allocation when libnuma is available.
     */
    class CPUBackend : public IBackend
    {
    public:
        /**
         * @brief Construct CPUBackend for this rank's NUMA node
         * @param local_numa_node NUMA node for this MPI rank (from MPITopology::placement().numa_node)
         *
         * If local_numa_node is -1 or invalid, uses system-wide memory (no NUMA binding).
         */
        explicit CPUBackend(int local_numa_node);
        ~CPUBackend() override;

        // ====================================================================
        // Device Enumeration (rank-local: always returns 1)
        // ====================================================================

        /**
         * @brief Get number of CPU "devices" (always 1 for rank-local view)
         * @return 1
         */
        int deviceCount() const override;

        /**
         * @brief Get backend name
         * @return "CPU"
         */
        std::string backendName() const override;

        /**
         * @brief Get device name including NUMA node
         * @param device_id Must be 0 (rank-local view)
         * @return "CPU:NUMA{N}" where N is the local NUMA node
         */
        std::string deviceName(int device_id) const override;

        // ====================================================================
        // Memory Query (queries this rank's NUMA node)
        // ====================================================================

        /**
         * @brief Get total memory for this rank's NUMA node
         * @param device_id Must be 0
         * @return Total memory in bytes from /sys/devices/system/node/nodeN/meminfo
         */
        size_t deviceMemoryTotal(int device_id) const override;

        /**
         * @brief Get free memory for this rank's NUMA node
         * @param device_id Must be 0
         * @return Free memory in bytes from /sys/devices/system/node/nodeN/meminfo
         */
        size_t deviceMemoryFree(int device_id) const override;

        // ====================================================================
        // Memory Operations (NUMA-aware allocation)
        // ====================================================================

        /**
         * @brief Allocate memory on this rank's NUMA node
         * @param bytes Number of bytes to allocate
         * @param device_id Must be 0
         * @return Pointer to allocated memory, or nullptr on failure
         *
         * Uses numa_alloc_onnode() if libnuma is available, otherwise falls back
         * to aligned_alloc() with parallel first-touch initialization for NUMA locality.
         */
        void *allocate(size_t bytes, int device_id) override;

        /**
         * @brief Free memory allocated by this backend
         * @param ptr Pointer to free (may be nullptr)
         * @param device_id Must be 0
         */
        void free(void *ptr, int device_id) override;

        /**
         * @brief Set memory to a byte value
         * @param ptr Pointer to fill
         * @param value Byte value to set (0-255)
         * @param bytes Number of bytes to set
         * @param device_id Must be 0
         * @return true on success
         */
        bool memset(void *ptr, int value, size_t bytes, int device_id) override;

        /**
         * @brief Allocate "mapped" memory (on CPU, just regular allocation)
         * @param bytes Number of bytes to allocate
         * @param device_id Must be 0
         * @param device_ptr Set to same value as returned host pointer (no GPU)
         * @return Pointer to allocated memory, or nullptr on failure
         */
        void *allocateMapped(size_t bytes, int device_id, void **device_ptr) override;

        /**
         * @brief Free mapped memory (on CPU, just regular free)
         * @param host_ptr Pointer to free (may be nullptr)
         * @param device_id Must be 0
         */
        void freeMapped(void *host_ptr, int device_id) override;

        // ====================================================================
        // Transfer Operations (memcpy for CPU)
        // ====================================================================

        /**
         * @brief Copy from "device" (CPU) to host (CPU) - just memcpy
         * @param dst Destination pointer
         * @param src Source pointer
         * @param bytes Number of bytes to copy
         * @param device_id Must be 0
         * @return true on success
         */
        bool deviceToHost(void *dst, const void *src, size_t bytes, int device_id) override;

        /**
         * @brief Copy from host (CPU) to "device" (CPU) - just memcpy
         * @param dst Destination pointer
         * @param src Source pointer
         * @param bytes Number of bytes to copy
         * @param device_id Must be 0
         * @return true on success
         */
        bool hostToDevice(void *dst, const void *src, size_t bytes, int device_id) override;

        /**
         * @brief Synchronize (no-op for CPU - always synchronous)
         * @param device_id Must be 0
         * @return true
         */
        bool synchronize(int device_id) override;

        /**
         * @brief Stream synchronize (no-op for CPU - always synchronous)
         * @param device_id Must be 0
         * @return true
         */
        bool streamSynchronize(int device_id) override;

        // ====================================================================
        // Async Operations (Trivial for CPU - immediate completion)
        // ====================================================================

        /**
         * @brief Async deviceToHost (returns immediately-completed future)
         */
        std::future<bool> deviceToHostAsync(void *dst, const void *src, size_t bytes, int device_id) override;

        /**
         * @brief Async hostToDevice (returns immediately-completed future)
         */
        std::future<bool> hostToDeviceAsync(void *dst, const void *src, size_t bytes, int device_id) override;

        /**
         * @brief Async synchronize (returns immediately-completed future)
         */
        std::future<bool> synchronizeAsync(int device_id) override;

        /**
         * @brief Async allocate (returns immediately-completed future)
         */
        std::future<void *> allocateAsync(size_t bytes, int device_id) override;

        /**
         * @brief Async free (returns immediately-completed future)
         */
        std::future<void> freeAsync(void *ptr, int device_id) override;

        /**
         * @brief Async memset (returns immediately-completed future)
         */
        std::future<bool> memsetAsync(void *ptr, int value, size_t bytes, int device_id) override;

        /**
         * @brief Set active device (no-op if device_id == 0)
         * @param device_id Must be 0
         * @return true if device_id == 0, false otherwise
         */
        bool setDevice(int device_id) override;

        // ====================================================================
        // Event Operations (no-op for CPU - always synchronous)
        // ====================================================================

        /**
         * @brief Create event (returns dummy pointer for CPU)
         * @param device_id Must be 0
         * @return Non-null dummy pointer (CPU is always synchronous)
         */
        void *createEvent(int device_id) override;

        /**
         * @brief Destroy event (no-op for CPU)
         * @param event Event handle (ignored)
         * @param device_id Must be 0
         */
        void destroyEvent(void *event, int device_id) override;

        /**
         * @brief Record event (no-op for CPU - always synchronous)
         * @param event Event handle
         * @param device_id Must be 0
         * @return true
         */
        bool recordEvent(void *event, int device_id) override;

        /**
         * @brief Wait for event (no-op for CPU - always synchronous)
         * @param event Event handle
         * @param device_id Must be 0
         * @return true
         */
        bool waitForEvent(void *event, int device_id) override;

        // ====================================================================
        // Capability Queries
        // ====================================================================

        /**
         * @brief Check if BF16 is supported (CPU supports via AVX-512 BF16)
         * @param device_id Must be 0
         * @return true (CPU always supports BF16 via software emulation)
         */
        bool supportsBF16(int device_id) const override;

        /**
         * @brief Check if FP16 is supported
         * @param device_id Must be 0
         * @return true (CPU supports FP16 via F16C or software)
         */
        bool supportsFP16(int device_id) const override;

        /**
         * @brief Check if INT8 is supported
         * @param device_id Must be 0
         * @return true (CPU supports INT8 via VNNI or software)
         */
        bool supportsINT8(int device_id) const override;

        // ====================================================================
        // Compute Operations
        // ====================================================================

        /**
         * @brief IQ4_NL GEMM (not implemented for CPU backend)
         * @return false (use CPU kernel directly instead)
         *
         * CPU kernels should be called directly, not through the backend interface.
         */
        bool gemmIQ4NL(
            const void *A_device,
            const void *B_device,
            void *C_device,
            int m,
            int n,
            int k,
            int device_id) override;

        // ====================================================================
        // CPUBackend-specific
        // ====================================================================

        /**
         * @brief Get the NUMA node this backend is bound to
         * @return NUMA node index (0-based), or -1 if no NUMA binding
         */
        int numaNode() const { return local_numa_node_; }

    private:
        int local_numa_node_;

        // Read NUMA memory info from /sys/devices/system/node/nodeN/meminfo
        size_t readNumaMemTotal() const;
        size_t readNumaMemFree() const;

        // Read system-wide memory info from /proc/meminfo (fallback)
        size_t readSystemMemTotal() const;
        size_t readSystemMemFree() const;

        // Validate device_id (must be 0 for rank-local view)
        bool isValidDeviceId(int device_id) const;
    };

} // namespace llaminar2
