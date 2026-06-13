/**
 * @file IBackend.h
 * @brief Abstract compute backend interface for GPU operations
 *
 * **Purpose**: Type-safe abstraction over CUDA/ROCm without exposing GPU headers.
 * Prevents header conflicts between cuda_runtime.h and hip_runtime.h.
 *
 * **Phase 3 Objective**: Enable separate compilation units for CUDA (.cu) and ROCm (.cpp)
 * backends while maintaining a unified API.
 *
 * **Design Principles**:
 * - No GPU-specific types in this header (no dim3, float4, cudaError_t, hipError_t)
 * - Pure virtual interface with pointer-based memory operations
 * - Implementations live in backend-specific compilation units
 *
 * @author David Sanftenberg
 */

#pragma once

#include "DeviceType.h"
#include <cstddef>
#include <cstdint>
#include <future>
#include <string>

namespace llaminar2
{

    /**
     * @class IBackend
     * @brief Abstract interface for compute backend operations
     *
     * **Implementations**:
     * - `CUDABackend` (backends/cuda/CUDABackend.cu)
     * - `ROCmBackend` (backends/rocm/ROCmBackend.cpp)
     * - `CPUBackend` (backends/CPUBackend.cpp) - for completeness
     *
     * **Usage**:
     * ```cpp
     * IBackend* backend = nullptr;
     * #ifdef HAVE_CUDA
     *     backend = new CUDABackend();
     * #elif HAVE_ROCM
     *     backend = new ROCmBackend();
     * #endif
     *
     * if (backend) {
     *     backend->deviceToHost(device_ptr, host_ptr, bytes, device_id);
     *     backend->synchronize(device_id);
     * }
     * ```
     */
    class IBackend
    {
    public:
        virtual ~IBackend() = default;

        // ====================================================================
        // Memory Transfer Operations
        // ====================================================================

        /**
         * @brief Copy data from device to host
         *
         * @param dst Host destination pointer (must be pre-allocated)
         * @param src Device source pointer
         * @param bytes Number of bytes to copy
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaMemcpyAsync on stream, then cudaStreamSynchronize
         * - ROCm: hipMemcpyAsync on stream, then hipStreamSynchronize
         * - CPU: memcpy(dst, src, bytes)
         *
         * @param stream Opaque GPU stream handle. When nullptr, backends
         *               auto-resolve to the device context's default stream
         *               (never the null CUDA/HIP stream).
         */
        virtual bool deviceToHost(void *dst, const void *src, size_t bytes, int device_id, void *stream = nullptr) = 0;

        /**
         * @brief Fast D2H copy — skips pointer validation for hot paths
         *
         * Caller guarantees:
         * - src is a valid device pointer on device_id
         * - GPU work writing to src has already completed (stream synced)
         * - dst is a valid host pointer with sufficient space
         *
         * Default implementation delegates to deviceToHost().
         * ROCm override skips hipPointerGetAttributes() + HipDeviceSaveRestore (~30-60µs savings).
         */
        virtual bool deviceToHostFast(void *dst, const void *src, size_t bytes, int device_id, void *stream = nullptr)
        {
            return deviceToHost(dst, src, bytes, device_id, stream);
        }

        /**
         * @brief Async variant of deviceToHost() - returns immediately
         *
         * @param dst Host destination pointer (must be pre-allocated)
         * @param src Device source pointer
         * @param bytes Number of bytes to copy
         * @param device_id GPU device ID (0-based)
         * @return Future that resolves to true on success, false on error
         *
         * **Semantics**:
         * - Submits copy request to device worker thread
         * - Returns immediately without blocking
         * - Call future.get() to wait for completion and get result
         */
        virtual std::future<bool> deviceToHostAsync(void *dst, const void *src, size_t bytes, int device_id) = 0;

        /**
         * @brief Copy data from host to device
         *
         * @param dst Device destination pointer (must be pre-allocated)
         * @param src Host source pointer
         * @param bytes Number of bytes to copy
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaMemcpyAsync on stream, then cudaStreamSynchronize
         * - ROCm: hipMemcpyAsync on stream, then hipStreamSynchronize
         * - CPU: memcpy(dst, src, bytes)
         *
         * @param stream Opaque GPU stream handle. When nullptr, backends
         *               auto-resolve to the device context's default stream
         *               (never the null CUDA/HIP stream).
         */
        virtual bool hostToDevice(void *dst, const void *src, size_t bytes, int device_id, void *stream = nullptr) = 0;

        /**
         * @brief Copy data between two device pointers on the same device
         *
         * @param dst Device destination pointer
         * @param src Device source pointer
         * @param bytes Number of bytes to copy
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice)
         * - ROCm: hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice)
         * - CPU: memcpy(dst, src, bytes)
         *
         * Used for device-to-device tensor transfers where both src and dst
         * are in the same GPU's VRAM (standard device memory pointers).
         *
         * @param stream Opaque GPU stream handle. When nullptr, backends
         *               auto-resolve to the device context's default stream.
         */
        virtual bool deviceToDevice(void *dst, const void *src, size_t bytes, int device_id, void *stream = nullptr)
        {
            // Default implementation: not supported
            (void)dst;
            (void)src;
            (void)bytes;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Async variant of hostToDevice() - returns immediately
         *
         * @param dst Device destination pointer (must be pre-allocated)
         * @param src Host source pointer
         * @param bytes Number of bytes to copy
         * @param device_id GPU device ID (0-based)
         * @return Future that resolves to true on success, false on error
         *
         * **Semantics**:
         * - Submits copy request to device worker thread
         * - Returns immediately without blocking
         * - Call future.get() to wait for completion and get result
         */
        virtual std::future<bool> hostToDeviceAsync(void *dst, const void *src, size_t bytes, int device_id) = 0;

        /**
         * @brief Synchronize all operations on a device
         *
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaDeviceSynchronize()
         * - ROCm: hipDeviceSynchronize()
         * - CPU: no-op (always synchronous)
         */
        virtual bool synchronize(int device_id) = 0;

        /**
         * @brief Async variant of synchronize() - returns immediately
         *
         * @param device_id GPU device ID (0-based)
         * @return Future that resolves to true on success, false on error
         *
         * **Semantics**:
         * - Submits sync request to device worker thread
         * - Returns immediately without blocking
         * - Call future.get() to wait for completion and get result
         */
        virtual std::future<bool> synchronizeAsync(int device_id) = 0;

        /**
         * @brief Synchronize the default stream on a device
         *
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaStreamSynchronize(nullptr)
         * - ROCm: hipStreamSynchronize(nullptr)
         * - CPU: no-op (always synchronous)
         *
         * **Performance Note**: This is lighter than synchronize() because it only
         * waits for the default stream, not all streams on the device.
         */
        virtual bool streamSynchronize(int device_id) = 0;

        // ====================================================================
        // Event Operations (Fine-grained Synchronization)
        // ====================================================================

        /**
         * @brief Create an event for fine-grained synchronization
         *
         * @param device_id GPU device ID (0-based)
         * @return Opaque event handle (nullptr on failure)
         *
         * **Semantics**:
         * - CUDA: cudaEventCreate()
         * - ROCm: hipEventCreate()
         * - CPU: returns dummy non-null pointer
         *
         * **Thread Safety**: Caller must ensure device is set before calling
         * **Lifetime**: Caller owns the event and must call destroyEvent()
         */
        virtual void *createEvent(int device_id) = 0;

        /**
         * @brief Destroy an event created by createEvent()
         *
         * @param event Opaque event handle (may be nullptr)
         * @param device_id GPU device ID (0-based)
         *
         * **Semantics**:
         * - CUDA: cudaEventDestroy()
         * - ROCm: hipEventDestroy()
         * - CPU: no-op
         */
        virtual void destroyEvent(void *event, int device_id) = 0;

        /**
         * @brief Record an event on the specified stream
         *
         * Marks the current point in the given stream. All operations
         * submitted before this call on that stream will complete before
         * the event is signaled.
         *
         * @param event Opaque event handle from createEvent()
         * @param device_id GPU device ID (0-based)
         * @param stream Opaque stream handle (nullptr = default stream)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaEventRecord(event, stream)
         * - ROCm: hipEventRecord(event, stream)
         * - CPU: no-op (returns true)
         */
        virtual bool recordEvent(void *event, int device_id, void *stream = nullptr) = 0;

        /**
         * @brief Wait for an event to complete
         *
         * Blocks the host until all operations recorded before the event
         * have completed on the device.
         *
         * @param event Opaque event handle from createEvent()
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaEventSynchronize(event)
         * - ROCm: hipEventSynchronize(event)
         * - CPU: no-op (returns true)
         *
         * **Performance Note**: This is lighter than synchronize() because it only
         * waits for a specific point in the stream, not all device operations.
         */
        virtual bool waitForEvent(void *event, int device_id) = 0;

        /**
         * @brief Set active device for subsequent operations
         *
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaSetDevice(device_id)
         * - ROCm: hipSetDevice(device_id)
         * - CPU: no-op (single device)
         */
        virtual bool setDevice(int device_id) = 0;

        // ====================================================================
        // Memory Allocation Operations
        // ====================================================================

        /**
         * @brief Allocate device memory
         *
         * @param bytes Number of bytes to allocate
         * @param device_id GPU device ID (0-based)
         * @return Pointer to allocated memory (nullptr on failure)
         *
         * **Semantics**:
         * - CUDA: cudaMalloc(&ptr, bytes)
         * - ROCm: hipMalloc(&ptr, bytes)
         * - CPU: malloc(bytes)
         *
         * **Thread Safety**: Caller must ensure device is set before calling
         */
        virtual void *allocate(size_t bytes, int device_id) = 0;

        /**
         * @brief Async variant of allocate() - returns immediately
         *
         * @param bytes Number of bytes to allocate
         * @param device_id GPU device ID (0-based)
         * @return Future that resolves to the allocated pointer (nullptr on failure)
         *
         * **Semantics**:
         * - Submits allocation request to device worker thread
         * - Returns immediately without blocking
         * - Call future.get() to wait for completion and get result
         */
        virtual std::future<void *> allocateAsync(size_t bytes, int device_id) = 0;

        /**
         * @brief Free device memory
         *
         * @param ptr Device pointer to free (may be nullptr)
         * @param device_id GPU device ID (0-based)
         *
         * **Semantics**:
         * - CUDA: cudaFree(ptr)
         * - ROCm: hipFree(ptr)
         * - CPU: free(ptr)
         *
         * **Thread Safety**: Caller must ensure device is set before calling
         */
        virtual void free(void *ptr, int device_id) = 0;

        /**
         * @brief Async variant of free() - returns immediately
         *
         * @param ptr Device pointer to free (may be nullptr)
         * @param device_id GPU device ID (0-based)
         * @return Future that completes when free is done
         *
         * **Semantics**:
         * - Submits free request to device worker thread
         * - Returns immediately without blocking
         * - Call future.wait() to wait for completion
         */
        virtual std::future<void> freeAsync(void *ptr, int device_id) = 0;

        /**
         * @brief Set device memory to a byte value
         *
         * @param ptr Device pointer to fill
         * @param value Byte value to set (0-255)
         * @param bytes Number of bytes to set
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaMemset(ptr, value, bytes)
         * - ROCm: hipMemset(ptr, value, bytes)
         * - CPU: memset(ptr, value, bytes)
         *
         * **Common Use Cases**:
         * - Zero-initialize output buffers before kernel execution
         * - Set corruption patterns (0xCC) for data integrity verification
         * - Clear buffers between operations in P2P testing
         *
         * **Thread Safety**: Caller must ensure device is set before calling
         *
         * @param stream Opaque GPU stream handle. When nullptr, backends
         *               auto-resolve to the device context's default stream.
         */
        virtual bool memset(void *ptr, int value, size_t bytes, int device_id, void *stream = nullptr) = 0;

        /**
         * @brief Async variant of memset() - returns immediately
         *
         * @param ptr Device pointer to fill
         * @param value Byte value to set (0-255)
         * @param bytes Number of bytes to set
         * @param device_id GPU device ID (0-based)
         * @return Future that resolves to true on success, false on error
         *
         * **Semantics**:
         * - Submits memset request to device worker thread
         * - Returns immediately without blocking
         * - Call future.get() to wait for completion and get result
         */
        virtual std::future<bool> memsetAsync(void *ptr, int value, size_t bytes, int device_id) = 0;

        // ====================================================================
        // Zero-Copy Mapped Memory Operations (GPU writes directly to host)
        // ====================================================================

        /**
         * @brief Allocate zero-copy mapped memory (host memory accessible by GPU)
         *
         * Allocates pinned host memory that can be written to directly by GPU kernels.
         * The GPU writes via PCIe, so no explicit D2H memcpy is needed - but writes
         * are slower than to device memory (PCIe bandwidth limited).
         *
         * @param bytes Number of bytes to allocate
         * @param device_id GPU device ID that will access this memory
         * @param[out] device_ptr Receives the device-visible pointer for the GPU
         * @return Host pointer (nullptr on failure). device_ptr set to GPU-visible address.
         *
         * **Use Case**: Output tensors that need to be read by host (verification, snapshots).
         * Instead of: kernel writes to device mem -> hipMemcpy D2H
         * Do: kernel writes directly to mapped host mem -> hipDeviceSynchronize -> read host
         *
         * **Performance Trade-off**:
         * - Pro: Eliminates D2H memcpy entirely
         * - Pro: Can read partial data while kernel is running (with care)
         * - Con: GPU writes to host are PCIe-limited (~15 GB/s vs ~1 TB/s HBM)
         *
         * **Semantics**:
         * - CUDA: cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped)
         *         cudaHostGetDevicePointer(&dev_ptr, ptr, 0)
         * - ROCm: hipHostMalloc(&ptr, bytes, hipHostMallocMapped)
         *         hipHostGetDevicePointer(&dev_ptr, ptr, 0)
         * - CPU: malloc(bytes), device_ptr = nullptr (no GPU)
         *
         * **Thread Safety**: Caller must ensure device is set before calling
         * **Lifetime**: Caller owns memory, must call freeMapped()
         */
        virtual void *allocateMapped(size_t bytes, int device_id, void **device_ptr) = 0;

        /**
         * @brief Free zero-copy mapped memory allocated by allocateMapped()
         *
         * @param host_ptr Host pointer returned by allocateMapped() (may be nullptr)
         * @param device_id GPU device ID used for allocation
         *
         * **Semantics**:
         * - CUDA: cudaFreeHost(host_ptr)
         * - ROCm: hipHostFree(host_ptr)
         * - CPU: free(host_ptr)
         */
        virtual void freeMapped(void *host_ptr, int device_id) = 0;

        // ====================================================================
        // Device Query Operations
        // ====================================================================

        /**
         * @brief Get number of available devices
         *
         * @return Device count (0 if backend unavailable)
         *
         * **Semantics**:
         * - CUDA: cudaGetDeviceCount()
         * - ROCm: hipGetDeviceCount()
         * - CPU: Always returns 1
         */
        virtual int deviceCount() const = 0;

        /**
         * @brief Get backend name (for logging/debugging)
         *
         * @return Backend identifier ("CUDA", "ROCm", "CPU")
         */
        virtual std::string backendName() const = 0;

        /**
         * @brief Get device name string
         *
         * @param device_id GPU device ID (0-based)
         * @return Device name (e.g., "NVIDIA A100", "AMD MI250X")
         */
        virtual std::string deviceName(int device_id) const = 0;

        /**
         * @brief Get total device memory in bytes
         *
         * @param device_id GPU device ID (0-based)
         * @return Total memory in bytes (0 on error)
         */
        virtual size_t deviceMemoryTotal(int device_id) const = 0;

        /**
         * @brief Get free device memory in bytes
         *
         * @param device_id GPU device ID (0-based)
         * @return Free memory in bytes (0 on error)
         */
        virtual size_t deviceMemoryFree(int device_id) const = 0;

        // ====================================================================
        // Capability Queries
        // ====================================================================

        /**
         * @brief Check if backend supports BF16 compute
         *
         * @param device_id GPU device ID (0-based)
         * @return true if BF16 supported (e.g., CUDA compute capability ≥ 8.0)
         */
        virtual bool supportsBF16(int device_id) const = 0;

        /**
         * @brief Check if backend supports FP16 compute
         *
         * @param device_id GPU device ID (0-based)
         * @return true if FP16 supported (e.g., CUDA compute capability ≥ 5.3)
         */
        virtual bool supportsFP16(int device_id) const = 0;

        /**
         * @brief Check if backend supports INT8 compute
         *
         * @param device_id GPU device ID (0-based)
         * @return true if INT8 supported (e.g., CUDA compute capability ≥ 6.1)
         */
        virtual bool supportsINT8(int device_id) const = 0;

        // ====================================================================
        // Host Memory Pinning (for async DMA)
        // ====================================================================

        /**
         * @brief Pin host memory for zero-copy DMA transfers
         *
         * Pinned (page-locked) memory enables true async DMA transfers and
         * eliminates internal staging copies in hipMemcpy/cudaMemcpy.
         *
         * @param ptr Host pointer to pin
         * @param bytes Size of the region in bytes
         * @return true on success, false on error
         *
         * Call unpinHostMemory() when done. No-op on CPU backend.
         */
        virtual bool pinHostMemory(void *ptr, size_t bytes) { return true; }
        virtual bool unpinHostMemory(void *ptr) { return true; }

        /**
         * @brief GPU-side argmax over FP32 data
         *
         * Finds the index and value of the maximum element entirely on the GPU,
         * avoiding a full D2H transfer of the logits tensor.
         * Used for greedy decode sampling.
         *
         * @param data_device Device pointer to FP32 data
         * @param n Number of elements
         * @param device_id Device where data resides
         * @param out_value Receives the maximum value
         * @param out_index Receives the index of the maximum element
         * @param stream Optional device stream to enqueue work on
         * @param partial_vals Device scratch [partial_capacity] for the
         *        two-pass multi-block reduction (per-block partial max values).
         *        Production GPU backends require caller-owned workspace/arena
         *        scratch and fail loud when it is missing or undersized.
         * @param partial_idxs Device scratch [partial_capacity] for the
         *        per-block partial max indices (paired with @p partial_vals).
         * @param partial_capacity Number of entries in the partial scratch buffers.
         * @return true if executed on device, false if not supported (caller should fall back)
         */
        virtual bool argmaxF32(const void *data_device, int n, int device_id,
                               float *out_value, int *out_index, void *stream = nullptr,
                               void *partial_vals = nullptr, void *partial_idxs = nullptr,
                               int partial_capacity = 0)
        {
            (void)data_device;
            (void)n;
            (void)device_id;
            (void)out_value;
            (void)out_index;
            (void)stream;
            (void)partial_vals;
            (void)partial_idxs;
            (void)partial_capacity;
            return false; // Not supported by default
        }

        /**
         * @brief GPU-side greedy argmax for several contiguous FP32 rows.
         *
         * @param data_device Device pointer to row-major FP32 data.
         * @param rows Number of rows to sample.
         * @param cols Number of columns per row.
         * @param device_id Device where data resides.
         * @param out_values Host buffer [rows] for max values.
         * @param out_indices Host buffer [rows] for row-local argmax indices.
         * @param stream Optional device stream to enqueue work on.
         * @param partial_vals Device scratch shared across rows. Backends that
         *        implement a fused batched kernel should partition this scratch
         *        internally. The default implementation calls argmaxF32() once
         *        per row, preserving existing backend behavior.
         * @param partial_idxs Device scratch paired with @p partial_vals.
         * @param partial_capacity Number of entries in each scratch buffer.
         * @return true if every row was sampled on device.
         */
        virtual bool argmaxF32BatchedRows(const void *data_device, int rows, int cols, int device_id,
                                          float *out_values, int *out_indices, void *stream = nullptr,
                                          void *partial_vals = nullptr, void *partial_idxs = nullptr,
                                          int partial_capacity = 0)
        {
            if (!data_device || rows <= 0 || cols <= 0 || !out_values || !out_indices)
                return false;

            const auto *base = static_cast<const float *>(data_device);
            for (int row = 0; row < rows; ++row)
            {
                if (!argmaxF32(base + static_cast<size_t>(row) * static_cast<size_t>(cols),
                               cols,
                               device_id,
                               out_values + row,
                               out_indices + row,
                               stream,
                               partial_vals,
                               partial_idxs,
                               partial_capacity))
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief Enqueue GPU-side greedy argmax for several contiguous FP32 rows.
         *
         * Unlike argmaxF32BatchedRows(), this vLLM-style primitive leaves the
         * row-local argmax values and indices on device. It performs no
         * allocation, host copy, or synchronization, and is suitable for graph
         * capture plus downstream device-side speculative verification summary.
         *
         * @param out_values_device Device buffer [rows] for max values.
         * @param out_indices_device Device buffer [rows] for row-local token ids.
         */
        virtual bool enqueueArgmaxF32BatchedRowsDevice(
            const void *data_device,
            int rows,
            int cols,
            int device_id,
            void *stream,
            void *out_values_device,
            void *out_indices_device,
            void *partial_vals = nullptr,
            void *partial_idxs = nullptr,
            int partial_capacity = 0)
        {
            (void)data_device;
            (void)rows;
            (void)cols;
            (void)device_id;
            (void)stream;
            (void)out_values_device;
            (void)out_indices_device;
            (void)partial_vals;
            (void)partial_idxs;
            (void)partial_capacity;
            return false;
        }

        /**
         * @brief GPU-side top-k selection over FP32 data
         *
         * Finds the k largest elements (value and index) entirely on the GPU.
         * Results are in descending order of value (highest first).
         * Used for top-k/top-p sampling to avoid a full D2H transfer.
         *
         * @param data_device Device pointer to FP32 data
         * @param n Number of elements
         * @param k Number of top elements to select (1..256)
         * @param device_id Device where data resides
         * @param out_values Host buffer for k float values (descending order)
         * @param out_indices Host buffer for k int indices
         * @return true if executed on device, false if not supported
         */
        virtual bool topKF32(const void *data_device, int n, int k, int device_id,
                             float *out_values, int *out_indices, void *stream = nullptr)
        {
            (void)data_device;
            (void)n;
            (void)k;
            (void)device_id;
            (void)out_values;
            (void)out_indices;
            (void)stream;
            return false; // Not supported by default
        }

        /**
         * @brief GPU-side top-k/top-p/temperature sampling over FP32 logits.
         *
         * This synchronous convenience wrapper only copies the selected token
         * back to the host. It must not materialize full logits on the CPU.
         *
         * @param data_device Device pointer to FP32 logits [n]
         * @param n Vocabulary size
         * @param top_k Top-k candidate limit (1..256, clamped by backend)
         * @param top_p Nucleus probability threshold (<=0 or >=1 disables)
         * @param temperature Sampling temperature (<=0 treated as 1)
         * @param rng_seed Deterministic RNG seed
         * @param rng_offset Per-sample RNG offset/counter
         * @param device_id Device where data resides
         * @param out_token Host pointer for selected token
         * @param stream Explicit GPU stream
         * @return true if sampled on device
         */
        virtual bool sampleTopKTopPF32(const void *data_device, int n,
                                       int top_k, float top_p, float temperature,
                                       uint64_t rng_seed, uint64_t rng_offset,
                                       int device_id, int *out_token,
                                       void *stream = nullptr)
        {
            (void)data_device;
            (void)n;
            (void)top_k;
            (void)top_p;
            (void)temperature;
            (void)rng_seed;
            (void)rng_offset;
            (void)device_id;
            (void)out_token;
            (void)stream;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable top-k/top-p/temperature sampling.
         *
         * The selected token is written to out_token_device. This method performs
         * no allocation, host/device copies, or synchronization, and requires an
         * explicit non-null stream.
         */
        virtual bool enqueueSampleTopKTopPF32Device(const void *data_device, int n,
                                                    int top_k, float top_p, float temperature,
                                                    uint64_t rng_seed, uint64_t rng_offset,
                                                    int device_id, void *stream,
                                                    void *out_token_device)
        {
            (void)data_device;
            (void)n;
            (void)top_k;
            (void)top_p;
            (void)temperature;
            (void)rng_seed;
            (void)rng_offset;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable top-k/top-p distribution construction.
         *
         * Writes a compact probability table of length top_k to device buffers.
         * Entries outside the selected top-p nucleus are marked with token id -1
         * and probability 0. This method performs no allocation, host/device
         * copies, or synchronization, and requires an explicit non-null stream.
         */
        virtual bool enqueueBuildTopKTopPDistributionF32Device(const void *data_device, int n,
                                                               int top_k, float top_p, float temperature,
                                                               int device_id, void *stream,
                                                               void *out_token_ids_device,
                                                               void *out_probs_device,
                                                               void *scratch_values_device = nullptr,
                                                               void *scratch_indices_device = nullptr,
                                                               int scratch_capacity = 0)
        {
            (void)data_device;
            (void)n;
            (void)top_k;
            (void)top_p;
            (void)temperature;
            (void)device_id;
            (void)stream;
            (void)out_token_ids_device;
            (void)out_probs_device;
            (void)scratch_values_device;
            (void)scratch_indices_device;
            (void)scratch_capacity;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable top-k/top-p distribution construction for rows.
         *
         * Builds `row_count` compact probability tables from a contiguous logits
         * matrix without allocating, synchronizing, or touching the default GPU
         * stream. `row_stride` is measured in FP32 elements between input rows;
         * `out_stride` is measured in INT32/FP32 entries between compact output
         * slots. Scratch capacity is the total number of `(value,index)` entries
         * available across every row in the batched launch.
         *
         * This is the vLLM-style target/bonus verifier-row companion to the
         * scalar distribution builder above. It lets the runner queue all
         * all-position verifier target rows behind one explicit stream handoff
         * instead of launching a scalar table builder per row.
         */
        virtual bool enqueueBuildTopKTopPDistributionsF32Device(
            const void *data_device,
            int row_count,
            int n,
            int row_stride,
            int top_k,
            float top_p,
            float temperature,
            int device_id,
            void *stream,
            void *out_token_ids_device,
            int out_stride,
            void *out_probs_device,
            void *scratch_values_device = nullptr,
            void *scratch_indices_device = nullptr,
            int scratch_capacity = 0)
        {
            (void)data_device;
            (void)row_count;
            (void)n;
            (void)row_stride;
            (void)top_k;
            (void)top_p;
            (void)temperature;
            (void)device_id;
            (void)stream;
            (void)out_token_ids_device;
            (void)out_stride;
            (void)out_probs_device;
            (void)scratch_values_device;
            (void)scratch_indices_device;
            (void)scratch_capacity;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable top-k/top-p processing into full logits.
         *
         * Converts raw logits into the processed-logit representation consumed by
         * the vLLM-style stochastic verifier: temperature is applied, tokens
         * outside the top-k/top-p nucleus are set to -inf, and active tokens keep
         * logits whose softmax exactly matches the compact top-k/top-p
         * distribution. `out_logits_device` may alias `data_device` after the
         * implementation has computed top-k partials, which lets graph stages
         * process LM-head output in place when ownership permits.
         *
         * Implementations must only enqueue work on `stream`; they must not
         * allocate, synchronize, or use a default/null GPU stream.
         */
        virtual bool enqueueBuildTopKTopPProcessedLogitsF32Device(
            const void *data_device,
            int row_count,
            int n,
            int row_stride,
            int top_k,
            float top_p,
            float temperature,
            int device_id,
            void *stream,
            void *out_logits_device,
            int out_row_stride,
            void *scratch_values_device = nullptr,
            void *scratch_indices_device = nullptr,
            int scratch_capacity = 0)
        {
            (void)data_device;
            (void)row_count;
            (void)n;
            (void)row_stride;
            (void)top_k;
            (void)top_p;
            (void)temperature;
            (void)device_id;
            (void)stream;
            (void)out_logits_device;
            (void)out_row_stride;
            (void)scratch_values_device;
            (void)scratch_indices_device;
            (void)scratch_capacity;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable sampling from a compact probability table.
         *
         * The threshold is a host-provided random draw in [0, 1), allowing callers
         * to preserve their existing deterministic RNG stream while keeping logits
         * and distribution math on the device. The output token is written to a
         * scalar device buffer. Requires an explicit non-null stream.
         */
        virtual bool enqueueSampleDistributionF32Device(
            const void *token_ids_device,
            const void *probs_device,
            int top_k,
            float threshold,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_probability_device = nullptr)
        {
            (void)token_ids_device;
            (void)probs_device;
            (void)top_k;
            (void)threshold;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_probability_device;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable sampling from processed full logits.
         *
         * The logits row is already in sampling space: temperature, penalties,
         * and any token masks have been applied. This is the vLLM-style
         * full-logit companion to enqueueSampleDistributionF32Device(), used for
         * bonus-ready or residual rows without materializing compact top-k
         * tables. Implementations must use the explicit non-null stream only.
         */
        virtual bool enqueueSampleProcessedLogitsF32Device(
            const void *logits_device,
            int vocab_size,
            int row_stride,
            float threshold,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_probability_device = nullptr)
        {
            (void)logits_device;
            (void)vocab_size;
            (void)row_stride;
            (void)threshold;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_probability_device;
            return false;
        }

        /**
         * @brief Lazily sample a processed bonus row only when a batch needs it.
         *
         * Stochastic MTP only consumes the bonus ready token when the first
         * target token does not stop and every verified speculative row
         * accepts. This graph-capturable primitive checks the compact verifier
         * outputs on device, returns `-1` immediately when the bonus is not
         * semantically needed, and otherwise samples @p logits_device exactly
         * like enqueueSampleProcessedLogitsF32Device().
         *
         * Backends must use the explicit non-null @p stream only. They must not
         * allocate, synchronize, or fall back to a default/null GPU stream.
         */
        virtual bool enqueueSampleProcessedLogitsF32DeviceIfSpeculativeBatchNeedsBonus(
            const void *logits_device,
            int vocab_size,
            int row_stride,
            float threshold,
            const void *verify_tokens_device,
            const void *verify_accepted_device,
            int row_count,
            int first_token,
            const void *first_token_device,
            const int *stop_tokens_host,
            int stop_token_count,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_probability_device = nullptr)
        {
            (void)logits_device;
            (void)vocab_size;
            (void)row_stride;
            (void)threshold;
            (void)verify_tokens_device;
            (void)verify_accepted_device;
            (void)row_count;
            (void)first_token;
            (void)first_token_device;
            (void)stop_tokens_host;
            (void)stop_token_count;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_probability_device;
            return false;
        }

        /**
         * @brief Enqueue vLLM-style draft proposal from raw logits.
         *
         * Draft proposal deliberately uses only temperature-scaled raw logits:
         * target rows own top-k/top-p, penalties, and residual correction. The
         * kernel writes the full proposal probability row plus the sampled draft
         * token and q(sampled_token), giving the verifier everything it needs
         * without building a compact top-k/top-p draft table.
         *
         * Implementations must only enqueue work on `stream`; they must not
         * allocate, synchronize, or use a device-default/null stream.
         */
        virtual bool enqueueSoftmaxAndSampleTemperatureLogitsF32Device(
            const void *logits_device,
            int vocab_size,
            int row_stride,
            float temperature,
            float threshold,
            int device_id,
            void *stream,
            void *out_probabilities_device,
            int out_row_stride,
            void *out_token_device,
            void *out_probability_device = nullptr)
        {
            (void)logits_device;
            (void)vocab_size;
            (void)row_stride;
            (void)temperature;
            (void)threshold;
            (void)device_id;
            (void)stream;
            (void)out_probabilities_device;
            (void)out_row_stride;
            (void)out_token_device;
            (void)out_probability_device;
            return false;
        }

        /**
         * @brief Enqueue vLLM-style draft proposal while preserving draft logits.
         *
         * This is the production draft-side companion to
         * enqueueSoftmaxAndSampleTemperatureLogitsF32Device(). It samples from
         * the temperature-only proposal distribution, writes the sampled token
         * plus q(sampled_token), and stores the temperature-scaled proposal
         * logits in `out_logits_device` for rejection verification.
         *
         * The verifier can then compute p/q and recovered-token weights from
         * logits/logsumexp, avoiding a full-vocab draft probability matrix.
         * Implementations must only enqueue work on `stream`; they must not
         * allocate, synchronize, or use a device-default/null stream.
         */
        virtual bool enqueueScaleAndSampleTemperatureLogitsF32Device(
            const void *logits_device,
            int vocab_size,
            int row_stride,
            float temperature,
            float threshold,
            int device_id,
            void *stream,
            void *out_logits_device,
            int out_row_stride,
            void *out_token_device,
            void *out_probability_device = nullptr)
        {
            (void)logits_device;
            (void)vocab_size;
            (void)row_stride;
            (void)temperature;
            (void)threshold;
            (void)device_id;
            (void)stream;
            (void)out_logits_device;
            (void)out_row_stride;
            (void)out_token_device;
            (void)out_probability_device;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable softmax for processed full-logit rows.
         *
         * The input rows are already in sampling space: temperature, penalties,
         * and token masks have been applied. Non-finite logits are written as
         * probability zero. This is the vLLM-style materialization step that
         * feeds full-probability rejection sampling.
         *
         * Implementations must only enqueue work on `stream`; they must not
         * allocate, synchronize, or use a device-default/null stream.
         */
        virtual bool enqueueSoftmaxProcessedLogitsF32Device(
            const void *logits_device,
            int row_count,
            int vocab_size,
            int row_stride,
            int device_id,
            void *stream,
            void *out_probabilities_device,
            int out_row_stride)
        {
            (void)logits_device;
            (void)row_count;
            (void)vocab_size;
            (void)row_stride;
            (void)device_id;
            (void)stream;
            (void)out_probabilities_device;
            (void)out_row_stride;
            return false;
        }

        /**
         * @brief Fill vLLM-style inverse-exponential rejection samples on device.
         *
         * Writes `row_count` full-vocab rows. Row `r` uses logical position
         * `first_logical_position + r` so the same speculative verification
         * step produces identical recovered-token choices whether it is captured
         * or replayed. Implementations must only enqueue work on the explicit
         * non-null stream; no allocation, synchronization, or default/null
         * stream use is allowed.
         */
        virtual bool enqueueFillInverseExponentialSamplesF32Device(
            void *out_samples_device,
            int row_count,
            int vocab_size,
            int row_stride,
            uint64_t seed,
            int first_logical_position,
            int device_id,
            void *stream)
        {
            (void)out_samples_device;
            (void)row_count;
            (void)vocab_size;
            (void)row_stride;
            (void)seed;
            (void)first_logical_position;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable speculative verify from compact distributions.
         *
         * target/draft distributions must be the top-k probability tables
         * produced by enqueueBuildTopKTopPDistributionF32Device(). The kernel
         * accepts draft_token with min(1, p/q), otherwise samples from the
         * residual max(p - q, 0). It writes only small scalar outputs to device
         * buffers and requires an explicit non-null stream.
         */
        virtual bool enqueueSpeculativeVerifyDistributionsF32Device(
            const void *target_token_ids_device,
            const void *target_probs_device,
            const void *draft_token_ids_device,
            const void *draft_probs_device,
            int top_k,
            int draft_token,
            uint64_t accept_seed,
            uint64_t accept_offset,
            uint64_t residual_seed,
            uint64_t residual_offset,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr)
        {
            (void)target_token_ids_device;
            (void)target_probs_device;
            (void)draft_token_ids_device;
            (void)draft_probs_device;
            (void)top_k;
            (void)draft_token;
            (void)accept_seed;
            (void)accept_offset;
            (void)residual_seed;
            (void)residual_offset;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_accepted_device;
            (void)out_accept_probability_device;
            (void)out_accept_threshold_device;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable speculative verify using caller RNG draws.
         *
         * Equivalent to enqueueSpeculativeVerifyDistributionsF32Device(), but
         * consumes explicit accept/residual thresholds instead of deriving random
         * numbers from seed/offset pairs.
         */
        virtual bool enqueueSpeculativeVerifyDistributionsF32DeviceThresholds(
            const void *target_token_ids_device,
            const void *target_probs_device,
            const void *draft_token_ids_device,
            const void *draft_probs_device,
            int top_k,
            int draft_token,
            float accept_threshold,
            float residual_threshold,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr)
        {
            (void)target_token_ids_device;
            (void)target_probs_device;
            (void)draft_token_ids_device;
            (void)draft_probs_device;
            (void)top_k;
            (void)draft_token;
            (void)accept_threshold;
            (void)residual_threshold;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_accepted_device;
            (void)out_accept_probability_device;
            (void)out_accept_threshold_device;
            return false;
        }

        /**
         * @brief Enqueue batched speculative verification using caller RNG draws.
         *
         * This checks accept/reject decisions and computes each row's candidate
         * residual correction token for several contiguous target/draft
         * distribution slots in one launch. Callers still decide which first
         * rejected row is semantically consumed.
         */
        virtual bool enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatch(
            const void *target_token_ids_device,
            const void *target_probs_device,
            const void *draft_token_ids_device,
            const void *draft_probs_device,
            int top_k,
            int distribution_stride,
            const int *draft_tokens_host,
            const float *accept_thresholds_host,
            const float *residual_thresholds_host,
            int row_count,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr)
        {
            (void)target_token_ids_device;
            (void)target_probs_device;
            (void)draft_token_ids_device;
            (void)draft_probs_device;
            (void)top_k;
            (void)distribution_stride;
            (void)draft_tokens_host;
            (void)accept_thresholds_host;
            (void)residual_thresholds_host;
            (void)row_count;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_accepted_device;
            (void)out_accept_probability_device;
            (void)out_accept_threshold_device;
            return false;
        }

        /**
         * @brief Enqueue batched speculative verification using device draft tokens.
         *
         * This is the vLLM-style sibling of
         * enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatch(): the
         * sampled draft token sequence already lives in arena-owned device
         * memory, so the verifier kernel reads `draft_tokens_device[row]`
         * directly instead of receiving draft tokens as host scalar kernel
         * arguments. If `draft_token_probabilities_device` is provided, row `r`
         * contains q(draft_tokens_device[r]) from the draft sampler and the
         * verifier can skip the sampled-token lookup in the compact draft
         * table. Residual sampling still uses the full draft table.
         *
         * Thresholds remain scalar host values for now because they are
         * deterministic RNG outputs owned by the runner; moving them to device
         * memory is a later plumbing step.
         *
         * Implementations must only enqueue work on `stream`; they must not
         * allocate, synchronize, or use a default/null GPU stream.
         */
        virtual bool enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
            const void *target_token_ids_device,
            const void *target_probs_device,
            const void *draft_token_ids_device,
            const void *draft_probs_device,
            int top_k,
            int distribution_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            const float *residual_thresholds_host,
            int row_count,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            const void *draft_token_probabilities_device = nullptr)
        {
            (void)target_token_ids_device;
            (void)target_probs_device;
            (void)draft_token_ids_device;
            (void)draft_probs_device;
            (void)top_k;
            (void)distribution_stride;
            (void)draft_tokens_device;
            (void)accept_thresholds_host;
            (void)residual_thresholds_host;
            (void)row_count;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_accepted_device;
            (void)out_accept_probability_device;
            (void)out_accept_threshold_device;
            (void)draft_token_probabilities_device;
            return false;
        }

        /**
         * @brief Enqueue batched stochastic verification from processed full logits.
         *
         * This is the vLLM-style sibling of the compact-table verifier. The
         * target and draft rows are already processed into sampling space
         * (temperature/penalties/masks applied), and the kernel recovers only
         * the sampled draft token probabilities needed for accept/reject. On a
         * rejection it samples the residual distribution from the full logits.
         *
         * Implementations must launch only on the explicit non-null `stream`;
         * no allocation, synchronization, or device-default/null stream use is
         * allowed.
         */
        virtual bool enqueueSpeculativeVerifyProcessedLogitsF32DeviceThresholdsBatchDeviceTokens(
            const void *target_logits_device,
            const void *draft_logits_device,
            int row_count,
            int vocab_size,
            int target_row_stride,
            int draft_row_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            const float *residual_thresholds_host,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            const void *draft_token_probabilities_device = nullptr)
        {
            (void)target_logits_device;
            (void)draft_logits_device;
            (void)row_count;
            (void)vocab_size;
            (void)target_row_stride;
            (void)draft_row_stride;
            (void)draft_tokens_device;
            (void)accept_thresholds_host;
            (void)residual_thresholds_host;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_accepted_device;
            (void)out_accept_probability_device;
            (void)out_accept_threshold_device;
            (void)draft_token_probabilities_device;
            return false;
        }

        /**
         * @brief Enqueue vLLM-style verify from target logits and draft probabilities.
         *
         * The target rows are processed logits: penalties, top-k/top-p masks,
         * and temperature have already been applied. The draft rows are
         * temperature-only proposal probabilities captured when each MTP draft
         * token was sampled. In vLLM-style greedy-draft mode
         * @p no_draft_probabilities treats the draft distribution as one-hot at
         * the sampled token. The kernel computes p(draft) from the target row,
         * reads or synthesizes q(draft), and on rejection samples the recovered
         * token by reducing `max(p - q, 0) * inverse_exp(token)`.
         *
         * This keeps the production path closer to vLLM by avoiding target
         * full-probability rows and inverse-random matrices. Implementations
         * must launch only on the explicit non-null `stream`; no allocation,
         * synchronization, or device-default/null stream use is allowed.
         */
        virtual bool enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
            const void *target_logits_device,
            const void *draft_probabilities_device,
            int row_count,
            int vocab_size,
            int target_row_stride,
            int draft_row_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            uint64_t inverse_sample_seed,
            int inverse_sample_first_logical_position,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            bool no_draft_probabilities = false)
        {
            (void)target_logits_device;
            (void)draft_probabilities_device;
            (void)row_count;
            (void)vocab_size;
            (void)target_row_stride;
            (void)draft_row_stride;
            (void)draft_tokens_device;
            (void)accept_thresholds_host;
            (void)inverse_sample_seed;
            (void)inverse_sample_first_logical_position;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_accepted_device;
            (void)out_accept_probability_device;
            (void)out_accept_threshold_device;
            (void)no_draft_probabilities;
            return false;
        }

        /**
         * @brief Enqueue vLLM-style verify from target and draft logits.
         *
         * Target rows are processed logits: penalties, top-k/top-p masks, and
         * temperature have already been applied. Draft rows are temperature-only
         * proposal logits captured when each MTP draft token was sampled. The
         * kernel computes p(draft) and q(draft) from row-local logsumexp and, on
         * rejection, samples the recovered token with the same inverse-exp race
         * used by vLLM-style probability rejection.
         *
         * This is a measured alternative to the probability-row path. It avoids
         * materializing full draft probability rows while preserving exact
         * rejection semantics, but production promotion still depends on
         * backend-specific perf evidence. Implementations must launch only on
         * the explicit non-null `stream`; no allocation, synchronization, or
         * default/null stream use is allowed.
         */
        virtual bool enqueueSpeculativeVerifyProcessedTargetDraftLogitsF32DeviceThresholdsBatchDeviceTokens(
            const void *target_logits_device,
            const void *draft_logits_device,
            int row_count,
            int vocab_size,
            int target_row_stride,
            int draft_row_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            uint64_t inverse_sample_seed,
            int inverse_sample_first_logical_position,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            const void *draft_token_probabilities_device = nullptr)
        {
            (void)target_logits_device;
            (void)draft_logits_device;
            (void)row_count;
            (void)vocab_size;
            (void)target_row_stride;
            (void)draft_row_stride;
            (void)draft_tokens_device;
            (void)accept_thresholds_host;
            (void)inverse_sample_seed;
            (void)inverse_sample_first_logical_position;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_accepted_device;
            (void)out_accept_probability_device;
            (void)out_accept_threshold_device;
            (void)draft_token_probabilities_device;
            return false;
        }

        /**
         * @brief Enqueue vLLM-style stochastic verify from full probability rows.
         *
         * `target_probabilities_device` and `draft_probabilities_device` are
         * full-vocab probability rows for each verifier row. On rejection, the
         * kernel samples the recovered token by reducing
         * `max(target_prob - draft_prob, 0) * inverse_rejection_samples`.
         *
         * This mirrors vLLM's random rejection sampler and is the target
         * production contract for stochastic MTP. Implementations must launch
         * only on the explicit non-null `stream`; no allocation,
         * synchronization, or device-default/null stream use is allowed.
         */
        virtual bool enqueueSpeculativeVerifyProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
            const void *target_probabilities_device,
            const void *draft_probabilities_device,
            const void *inverse_rejection_samples_device,
            int row_count,
            int vocab_size,
            int target_row_stride,
            int draft_row_stride,
            int inverse_sample_row_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            bool no_draft_probabilities = false)
        {
            (void)target_probabilities_device;
            (void)draft_probabilities_device;
            (void)inverse_rejection_samples_device;
            (void)row_count;
            (void)vocab_size;
            (void)target_row_stride;
            (void)draft_row_stride;
            (void)inverse_sample_row_stride;
            (void)draft_tokens_device;
            (void)accept_thresholds_host;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_accepted_device;
            (void)out_accept_probability_device;
            (void)out_accept_threshold_device;
            (void)no_draft_probabilities;
            return false;
        }

        /**
         * @brief Enqueue device-side reduction of batched speculative verifier rows.
         *
         * The row verifier writes `verify_tokens_device` and
         * `verify_accepted_device`. This graph-capturable reducer converts those
         * row-local decisions into the vLLM-style output contract: committed
         * output tokens plus a small integer metadata table. Stop tokens are
         * host-side scalars copied into kernel arguments by backend wrappers;
         * the kernel never dereferences host memory. Requires an explicit
         * non-null stream.
         */
        virtual bool enqueueSummarizeSpeculativeVerifyBatch(
            const void *verify_tokens_device,
            const void *verify_accepted_device,
            int row_count,
            int first_token,
            const int *stop_tokens_host,
            int stop_token_count,
            const void *bonus_token_device,
            bool has_bonus_token,
            int device_id,
            void *stream,
            void *out_tokens_device,
            void *out_meta_device)
        {
            (void)verify_tokens_device;
            (void)verify_accepted_device;
            (void)row_count;
            (void)first_token;
            (void)stop_tokens_host;
            (void)stop_token_count;
            (void)bonus_token_device;
            (void)has_bonus_token;
            (void)device_id;
            (void)stream;
            (void)out_tokens_device;
            (void)out_meta_device;
            return false;
        }

        /**
         * @brief Enqueue batch summary using a device-resident first token.
         *
         * This is the graph-friendly companion to
         * enqueueSummarizeSpeculativeVerifyBatch(). The first main-model token
         * is sampled on GPU and remains in arena scratch until this reducer
         * consumes it. Backends must launch on the explicit `stream`, read
         * `first_token_device` inside the kernel, and call the same shared
         * SamplingMath reducer as the host-token variant.
         *
         * @param first_token_device Device pointer to one INT32 sampled token.
         * @return true when the reducer launch was queued successfully.
         */
        virtual bool enqueueSummarizeSpeculativeVerifyBatchDeviceFirstToken(
            const void *verify_tokens_device,
            const void *verify_accepted_device,
            int row_count,
            const void *first_token_device,
            const int *stop_tokens_host,
            int stop_token_count,
            const void *bonus_token_device,
            bool has_bonus_token,
            int device_id,
            void *stream,
            void *out_tokens_device,
            void *out_meta_device)
        {
            (void)verify_tokens_device;
            (void)verify_accepted_device;
            (void)row_count;
            (void)first_token_device;
            (void)stop_tokens_host;
            (void)stop_token_count;
            (void)bonus_token_device;
            (void)has_bonus_token;
            (void)device_id;
            (void)stream;
            (void)out_tokens_device;
            (void)out_meta_device;
            return false;
        }

        /**
         * @brief Enqueue device-side reduction of greedy verifier rows.
         *
         * Greedy MTP uses the same vLLM-style compact batch contract as the
         * stochastic verifier, but row acceptance is simply
         * `verify_tokens[row] == draft_tokens[row + 1]`. Backends must launch a
         * small graph-capturable kernel on the explicit non-null `stream`,
         * compare the device-resident verifier argmax rows with the
         * device-resident compact verifier input row, and write only compact
         * output tokens plus metadata for the host handoff.
         *
         * @param verify_tokens_device INT32 verifier argmax rows
         *        `[compare_row_count + 1]`; the final row is the bonus ready
         *        token used only when all speculative rows accept.
         * @param draft_tokens_device INT32 verifier input row
         *        `[first_token, draft_1, ...]`.
         * @param compare_row_count Number of speculative rows to compare.
         * @param first_token Host copy of `draft_tokens_device[0]`.
         */
        virtual bool enqueueSummarizeGreedySpeculativeVerifyBatch(
            const void *verify_tokens_device,
            const void *draft_tokens_device,
            int compare_row_count,
            int first_token,
            const int *stop_tokens_host,
            int stop_token_count,
            int device_id,
            void *stream,
            void *out_tokens_device,
            void *out_meta_device)
        {
            (void)verify_tokens_device;
            (void)draft_tokens_device;
            (void)compare_row_count;
            (void)first_token;
            (void)stop_tokens_host;
            (void)stop_token_count;
            (void)device_id;
            (void)stream;
            (void)out_tokens_device;
            (void)out_meta_device;
            return false;
        }

        /**
         * @brief GPU-side sparse logit penalty application
         *
         * Applies a sparse set of additive penalties to logits in-place on the GPU.
         * Each entry (token_id, penalty) subtracts the penalty from logits[token_id].
         * Used to apply presence, frequency, and DRY penalties without a full D2H
         * transfer of the logits tensor (~600KB for 151K vocab).
         *
         * @param logits_device Device pointer to FP32 logits [vocab_size] — modified in-place
         * @param token_ids_host Host array of token IDs to penalize
         * @param penalties_host Host array of penalty values (positive = penalize)
         * @param num_penalties Number of entries in token_ids and penalties arrays
         * @param vocab_size Total vocabulary size (for bounds checking)
         * @param device_id Device where logits reside
         * @param stream Optional stream for async execution
         * @return true if executed on device, false if not supported
         */
        virtual bool applyLogitPenaltiesF32(void *logits_device,
                                            const int *token_ids_host,
                                            const float *penalties_host,
                                            int num_penalties, int vocab_size,
                                            int device_id, void *stream = nullptr)
        {
            (void)logits_device;
            (void)token_ids_host;
            (void)penalties_host;
            (void)num_penalties;
            (void)vocab_size;
            (void)device_id;
            (void)stream;
            return false; // Not supported by default
        }

        /**
         * @brief Enqueue sparse logit penalties from device-resident inputs.
         *
         * This is the graph-capturable form of applyLogitPenaltiesF32(): token IDs
         * and penalty values already live on the target device, the caller supplies
         * an explicit non-null stream, and the backend only enqueues the penalty
         * kernel. It performs no allocation, host/device copies, or synchronization.
         *
         * @param logits_device Device pointer to FP32 logits [vocab_size], modified in-place
         * @param token_ids_device Device pointer to int token IDs [num_penalties]
         * @param penalties_device Device pointer to FP32 penalties [num_penalties]
         * @param num_penalties Number of entries in token_ids_device and penalties_device
         * @param vocab_size Total vocabulary size for bounds checking
         * @param device_id Device where all pointers reside
         * @param stream Explicit non-null GPU stream
         * @return true if the kernel launch was enqueued
         */
        virtual bool enqueueLogitPenaltiesF32Device(void *logits_device,
                                                    const void *token_ids_device,
                                                    const void *penalties_device,
                                                    int num_penalties, int vocab_size,
                                                    int device_id, void *stream)
        {
            (void)logits_device;
            (void)token_ids_device;
            (void)penalties_device;
            (void)num_penalties;
            (void)vocab_size;
            (void)device_id;
            (void)stream;
            return false;
        }

        // ====================================================================
        // Compute Operations
        // ====================================================================

        /**
         * @brief Quantized matrix multiplication: C = A * B (IQ4_NL format)
         *
         * Performs GEMM with FP32 activations and IQ4_NL quantized weights.
         *
         * @param A_device Device pointer to FP32 matrix A [m × k] row-major
         * @param B_device Device pointer to IQ4_NL quantized matrix B [n × k/32] blocks
         * @param C_device Device pointer to FP32 output matrix C [m × n] row-major
         * @param m Number of rows in A and C
         * @param n Number of columns in B and C
         * @param k Number of columns in A and rows in B (must be multiple of 32)
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Requirements**:
         * - All pointers must be valid device memory
         * - k must be a multiple of 32 (IQ4_NL block size)
         * - Matrices must be in row-major format
         *
         * **IQ4_NL Format** (each block encodes 32 floats in 18 bytes):
         * - 2 bytes: FP16 scale factor
         * - 16 bytes: Packed 4-bit indices (2 per byte)
         * - Effective: 4.5 bits/value (~7.1× compression vs FP32)
         *
         * **Semantics**:
         * - CUDA: Calls IQ4_NL_Gemm.cu kernel
         * - ROCm: Calls IQ4_NL_Gemm.hip kernel (future)
         * - CPU: Not applicable (use CPU kernel directly)
         */
        virtual bool gemmIQ4NL(
            const void *A_device,
            const void *B_device,
            void *C_device,
            int m,
            int n,
            int k,
            int device_id) = 0;

        // ====================================================================
        // Stream Management
        // ====================================================================

        /**
         * @brief Create a non-blocking stream on the specified device
         *
         * @param device_id GPU device ID (0-based)
         * @return Opaque stream handle (nullptr on failure)
         *
         * **Semantics**:
         * - CUDA: cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
         * - ROCm: hipStreamCreateWithFlags(&stream, hipStreamNonBlocking)
         * - CPU: returns dummy non-null pointer
         *
         * **Lifetime**: Caller owns the stream and must call destroyStream()
         */
        virtual void *createStream(int device_id)
        {
            (void)device_id;
            return nullptr;
        }

        /**
         * @brief Destroy a stream created by createStream()
         *
         * @param stream Opaque stream handle (may be nullptr)
         * @param device_id GPU device ID (0-based)
         */
        virtual void destroyStream(void *stream, int device_id)
        {
            (void)stream;
            (void)device_id;
        }

        /**
         * @brief Synchronize a specific stream (wait for all queued work to complete)
         *
         * @param stream Opaque stream handle (nullptr = default stream)
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaStreamSynchronize(stream)
         * - ROCm: hipStreamSynchronize(stream)
         * - CPU: no-op (always synchronous)
         */
        virtual bool synchronizeStream(void *stream, int device_id)
        {
            (void)stream;
            (void)device_id;
            return true;
        }

        /**
         * @brief Make a stream wait for an event before proceeding
         *
         * All operations enqueued on the stream after this call will wait until
         * the event is recorded and completed.
         *
         * @param stream Opaque stream handle
         * @param event Opaque event handle
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaStreamWaitEvent(stream, event, 0)
         * - ROCm: hipStreamWaitEvent(stream, event, 0)
         * - CPU: no-op (always synchronous)
         */
        virtual bool streamWaitEvent(void *stream, void *event, int device_id)
        {
            (void)stream;
            (void)event;
            (void)device_id;
            return true;
        }

        // ====================================================================
        // Async Host-to-Device Transfer (no implicit sync)
        // ====================================================================

        /**
         * @brief Submit async H2D copy on a specific stream WITHOUT synchronizing
         *
         * Unlike hostToDevice() which syncs after the memcpy, this submits the
         * copy and returns immediately. Caller is responsible for synchronization
         * (typically via recordEvent + streamWaitEvent or synchronizeStream).
         *
         * @param dst Device destination pointer (must be pre-allocated)
         * @param src Host source pointer (should be pinned for true async DMA)
         * @param bytes Number of bytes to copy
         * @param device_id GPU device ID (0-based)
         * @param stream Opaque stream handle (must not be nullptr)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream)
         * - ROCm: hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream)
         * - CPU: memcpy (synchronous fallback)
         */
        virtual bool hostToDeviceOnStream(void *dst, const void *src, size_t bytes,
                                          int device_id, void *stream)
        {
            // Default: fall back to synchronous hostToDevice
            return hostToDevice(dst, src, bytes, device_id, stream);
        }

        /**
         * @brief Submit async D2H copy on a specific stream WITHOUT synchronizing
         *
         * This is the device-to-host companion to hostToDeviceOnStream().  It is
         * intended for small compact summaries where several D2H copies should
         * be queued on one explicit producer stream and followed by exactly one
         * synchronizeStream() at the ownership handoff.  GPU implementations
         * must reject nullptr streams; CPU implementations may treat the stream
         * as an ignored synchronous marker.
         *
         * @param dst Host destination pointer (should be pinned for true async DMA)
         * @param src Device source pointer
         * @param bytes Number of bytes to copy
         * @param device_id GPU device ID (0-based)
         * @param stream Opaque stream handle (must not be nullptr for GPU)
         * @return true on successful enqueue/copy, false on error
         *
         * **Semantics**:
         * - CUDA: cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream)
         * - ROCm: hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream)
         * - CPU: memcpy (synchronous fallback)
         */
        virtual bool deviceToHostOnStream(void *dst, const void *src, size_t bytes,
                                          int device_id, void *stream)
        {
            // Default: fall back to synchronous deviceToHost.
            return deviceToHost(dst, src, bytes, device_id, stream);
        }

        // ====================================================================
        // Pinned Host Memory Allocation
        // ====================================================================

        /**
         * @brief Allocate pinned (page-locked) host memory for async DMA transfers
         *
         * Pinned memory enables true async H2D/D2H transfers without internal
         * staging copies. Required for overlapped pipeline transfers.
         *
         * @param bytes Number of bytes to allocate
         * @param device_id GPU device ID (for device affinity, 0-based)
         * @return Host pointer (nullptr on failure)
         *
         * **Semantics**:
         * - CUDA: cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault)
         * - ROCm: hipHostMalloc(&ptr, bytes, hipHostMallocDefault)
         * - CPU: malloc(bytes)
         *
         * **Lifetime**: Caller owns memory, must call freePinned()
         */
        virtual void *allocatePinned(size_t bytes, int device_id)
        {
            (void)bytes;
            (void)device_id;
            return nullptr;
        }

        /**
         * @brief Free pinned host memory allocated by allocatePinned()
         *
         * @param ptr Host pointer from allocatePinned() (may be nullptr)
         * @param device_id GPU device ID used for allocation
         */
        virtual void freePinned(void *ptr, int device_id)
        {
            (void)ptr;
            (void)device_id;
        }

        // ====================================================================
        // Stream-Aware Memory Operations
        // ====================================================================

        /**
         * @brief Async device-to-device copy on a specific stream
         *
         * Both src and dst must be device pointers on the same device (or
         * accessible from that device, e.g., BAR-mapped UVA pointers).
         *
         * @param dst Device destination pointer
         * @param src Device source pointer
         * @param bytes Number of bytes to copy
         * @param device_id GPU device ID (0-based)
         * @param stream Opaque stream handle (nullptr = default stream)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream)
         * - ROCm: hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream)
         * - CPU: memcpy(dst, src, bytes)
         */
        virtual bool deviceCopyAsync(void *dst, const void *src, size_t bytes,
                                     int device_id, void *stream = nullptr)
        {
            (void)dst;
            (void)src;
            (void)bytes;
            (void)device_id;
            (void)stream;
            return false;
        }

        // ====================================================================
        // Collective Reduction Primitives
        // ====================================================================

        /**
         * @brief In-place element-wise vector addition: output += input
         *
         * Performs output[i] += input[i] for count elements on the GPU.
         * Used by cross-vendor collective backends for allreduce operations.
         *
         * @param output Device pointer to accumulate into (read+write)
         * @param input Device pointer to add (read-only)
         * @param count Number of elements
         * @param element_size Size of each element in bytes (4=FP32, 2=FP16/BF16, 1=INT8)
         * @param device_id GPU device ID (0-based)
         * @param stream Opaque stream handle (nullptr = synchronous on default stream)
         * @return true on success, false if not supported or error
         *
         * The element_size parameter determines the data type:
         * - 4 bytes: FP32 addition
         * - 2 bytes: FP16 or BF16 addition (backend-specific)
         * - 1 byte: INT8 saturating addition
         *
         * **Semantics**:
         * - CUDA: Launches vectorAdd kernel
         * - ROCm: Launches HIP vectorAdd kernel (future)
         * - CPU: Scalar loop
         */
        virtual bool vectorAddInplace(void *output, const void *input, size_t count,
                                      int element_size, int device_id, void *stream = nullptr)
        {
            (void)output;
            (void)input;
            (void)count;
            (void)element_size;
            (void)device_id;
            (void)stream;
            return false;
        }

        // ====================================================================
        // Backend Identity
        // ====================================================================

        /**
         * @brief Return the DeviceType this backend handles
         *
         * Used for defensive validation in cross-device scenarios (e.g., verifying
         * that a GPU stream matches the backend before calling recordEvent).
         */
        virtual DeviceType backendDeviceType() const = 0;
    };

} // namespace llaminar2
