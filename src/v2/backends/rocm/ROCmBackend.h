/**
 * @file ROCmBackend.h
 * @brief ROCm/HIP backend public API (no HIP headers exposed)
 *
 * **Purpose**: Public interface for ROCm backend. Implementation lives in .cpp file
 * to avoid exposing hip_runtime.h to other compilation units.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../IBackend.h"
#include <future>
#include <memory>
#include <cstdint>
#include <vector>

namespace llaminar2
{

    struct ROCmPointerOwnerInfo
    {
        void *base_ptr = nullptr;
        size_t size_bytes = 0;
        int device_id = -1;
        uint64_t sequence = 0;
        uint64_t thread_hash = 0;
        bool active = false;
    };

    /**
     * @class ROCmBackend
     * @brief ROCm/HIP compute backend implementation
     *
     * **Implementation**: See ROCmBackend.cpp
     * **Requirements**: AMD GPU with ROCm 5.0+
     * **Compilation**: Requires hipcc compiler, -DHAVE_ROCM=ON
     */
    class ROCmBackend : public IBackend
    {
    public:
        ROCmBackend();
        ~ROCmBackend() override;

        // Memory transfer operations (see IBackend documentation)
        bool deviceToHost(void *dst, const void *src, size_t bytes, int device_id) override;
        bool deviceToHostFast(void *dst, const void *src, size_t bytes, int device_id) override;
        bool hostToDevice(void *dst, const void *src, size_t bytes, int device_id) override;
        bool synchronize(int device_id) override;
        bool streamSynchronize(int device_id) override;
        bool setDevice(int device_id) override;

        // Host memory pinning for async DMA
        bool pinHostMemory(void *ptr, size_t bytes) override;
        bool unpinHostMemory(void *ptr) override;

        // GPU-side argmax for greedy sampling
        bool argmaxF32(const void *data_device, int n, int device_id,
                       float *out_value, int *out_index) override;

        // GPU-side top-k selection for sampling
        bool topKF32(const void *data_device, int n, int k, int device_id,
                     float *out_values, int *out_indices) override;

        // Event operations (fine-grained synchronization)
        void *createEvent(int device_id) override;
        void destroyEvent(void *event, int device_id) override;
        bool recordEvent(void *event, int device_id, void *stream = nullptr) override;
        bool waitForEvent(void *event, int device_id) override;

        // Memory allocation operations
        void *allocate(size_t bytes, int device_id) override;
        void free(void *ptr, int device_id) override;
        bool memset(void *ptr, int value, size_t bytes, int device_id) override;

        // Zero-copy mapped memory operations
        void *allocateMapped(size_t bytes, int device_id, void **device_ptr) override;
        void freeMapped(void *host_ptr, int device_id) override;

        // Device query operations
        int deviceCount() const override;
        std::string backendName() const override;
        std::string deviceName(int device_id) const override;
        size_t deviceMemoryTotal(int device_id) const override;
        size_t deviceMemoryFree(int device_id) const override;

        // Capability queries
        bool supportsBF16(int device_id) const override;
        bool supportsFP16(int device_id) const override;
        bool supportsINT8(int device_id) const override;

        // Backend identity
        DeviceType backendDeviceType() const override { return DeviceType::ROCm; }

        // Compute operations
        bool gemmIQ4NL(
            const void *A_device,
            const void *B_device,
            void *C_device,
            int m,
            int n,
            int k,
            int device_id) override;

        // ==== Async operations (submitted via AMDDeviceContext worker) ====

        std::future<bool> deviceToHostAsync(void *dst, const void *src, size_t bytes, int device_id) override;
        std::future<bool> hostToDeviceAsync(void *dst, const void *src, size_t bytes, int device_id) override;
        std::future<bool> synchronizeAsync(int device_id) override;
        std::future<void *> allocateAsync(size_t bytes, int device_id) override;
        std::future<void> freeAsync(void *ptr, int device_id) override;
        std::future<bool> memsetAsync(void *ptr, int value, size_t bytes, int device_id) override;

        // ==== Extended operations (not in IBackend) ====

        /**
         * @brief Query pointer attributes to understand address space
         * @param ptr Pointer to query
         * @param is_device_ptr Output: true if ptr is a device pointer
         * @param is_host_ptr Output: true if ptr is a host pointer
         * @param is_managed Output: true if ptr is managed memory
         * @param device_id Output: device ID if device pointer
         * @return true if query succeeded
         */
        bool queryPointerAttributes(const void *ptr, bool &is_device_ptr, bool &is_host_ptr,
                                    bool &is_managed, int &device_id) const;

        /**
         * @brief Copy device-to-device
         * @param dst Destination device pointer
         * @param src Source device pointer
         * @param bytes Number of bytes
         * @param device_id Device to use for the copy
         * @return true on success
         */
        bool deviceToDevice(void *dst, const void *src, size_t bytes, int device_id);

        /**
         * @brief Register IO memory with HIP using hipHostRegisterIoMemory flag
         *
         * This attempts to register memory-mapped I/O regions (like PCIe BAR)
         * with HIP so that kernels can access them directly.
         *
         * @param ptr Host pointer to the IO memory (e.g., BAR mmap address)
         * @param size Size of the region in bytes
         * @param device_ptr Output: Device pointer that kernels can use
         * @return true if registration succeeded, false otherwise
         */
        bool registerIoMemory(void *ptr, size_t size, void **device_ptr);

        /**
         * @brief Unregister previously registered IO memory
         * @param ptr The host pointer that was registered
         */
        void unregisterIoMemory(void *ptr);

        /**
         * @brief Get detailed pointer info including device pointer
         *
         * @param ptr Pointer to query
         * @param device_ptr Output: The device-accessible pointer (may be same or different)
         * @param host_ptr Output: The host-accessible pointer
         * @param mem_type Output: Memory type string for debugging
         * @return true if query succeeded
         */
        bool getPointerInfo(const void *ptr, void **device_ptr, void **host_ptr,
                            std::string &mem_type) const;

        /**
         * @brief Lock host memory with HSA API and get GPU-accessible pointer
         *
         * Uses hsa_amd_memory_lock() to pin host memory and get a pointer
         * that GPU agents can use. This is a lower-level API than hipHostRegister
         * and may work for memory regions that hipHostRegister rejects.
         *
         * @param host_ptr Host pointer to lock (can be mmap'd memory)
         * @param size Size of the region in bytes
         * @param agent_ptr Output: GPU-accessible pointer
         * @return true if lock succeeded, false otherwise
         */
        bool hsaMemoryLock(void *host_ptr, size_t size, void **agent_ptr);

        /**
         * @brief Unlock previously locked memory
         * @param host_ptr Host pointer that was locked
         */
        void hsaMemoryUnlock(void *host_ptr);

        /**
         * @brief Map a dmabuf/interop buffer using HSA interop API
         *
         * Uses hsa_amd_interop_map_buffer() to map a dmabuf file descriptor
         * into the GPU address space. This is the most direct path to get
         * kernel-accessible pointers for external memory.
         *
         * @param dmabuf_fd File descriptor of the dmabuf (or PCIe BAR resource fd)
         * @param size Output: Size of the mapped buffer (filled by API)
         * @param device_ptr Output: GPU-accessible pointer
         * @return true if mapping succeeded, false otherwise
         */
        bool hsaInteropMapBuffer(int dmabuf_fd, size_t *size, void **device_ptr);

        /**
         * @brief Unmap a previously mapped interop buffer
         * @param device_ptr The device pointer that was returned from hsaInteropMapBuffer
         */
        void hsaInteropUnmapBuffer(void *device_ptr);

        /**
         * @brief Import external memory via HIP external memory API
         *
         * Uses hipImportExternalMemory() to import memory from a file descriptor.
         * This is the HIP-level API for external memory interop.
         *
         * @param fd File descriptor of the external memory
         * @param size Size of the external memory region
         * @param device_ptr Output: Device-accessible pointer
         * @return true if import succeeded, false otherwise
         */
        bool importExternalMemory(int fd, size_t size, void **device_ptr);

        /**
         * @brief Get the HSA agent handle for a given device
         *
         * This is needed for low-level HSA operations.
         *
         * @param device_id Device index
         * @param agent Output: HSA agent handle (as uint64_t to avoid header deps)
         * @return true if query succeeded
         */
        bool getHsaAgent(int device_id, uint64_t *agent);

        /**
         * @brief Query recorded allocation ownership for a pointer address
         *
         * Checks whether @p ptr falls inside any active or recently-freed ROCm
         * allocation tracked by ROCmBackend::allocate()/free(). This is useful
         * for diagnosing cross-device pointer usage bugs.
         *
         * @param ptr Pointer address to inspect
         * @param info Output ownership metadata
         * @return true if matching allocation range was found
         */
        static bool queryPointerOwner(const void *ptr, ROCmPointerOwnerInfo &info);

        /**
         * @brief Dump recently tracked allocation events to logs
         * @param max_events Maximum recent events to print
         */
        static void dumpRecentPointerEvents(size_t max_events = 64);

    private:
        int device_count_;

        // Per-device argmax result buffers (lazily allocated)
        // Indices map to device_id. Stores device-side memory for kernel output.
        struct ArgmaxDeviceBuffers
        {
            void *value_ptr = nullptr; // Device pointer for 1 float
            void *index_ptr = nullptr; // Device pointer for 1 int
        };
        std::vector<ArgmaxDeviceBuffers> argmax_buffers_;

        // Per-device top-k result buffers (lazily allocated, resized to max k used)
        struct TopKDeviceBuffers
        {
            void *values_ptr = nullptr;  // Device pointer for k floats
            void *indices_ptr = nullptr; // Device pointer for k ints
            int allocated_k = 0;         // Current allocation size
        };
        std::vector<TopKDeviceBuffers> topk_buffers_;
    };

} // namespace llaminar2
