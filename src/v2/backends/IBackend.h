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
         * - CUDA: cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost)
         * - ROCm: hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost)
         * - CPU: memcpy(dst, src, bytes)
         */
        virtual bool deviceToHost(void *dst, const void *src, size_t bytes, int device_id) = 0;

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
        virtual bool deviceToHostFast(void *dst, const void *src, size_t bytes, int device_id)
        {
            return deviceToHost(dst, src, bytes, device_id);
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
         * - CUDA: cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice)
         * - ROCm: hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice)
         * - CPU: memcpy(dst, src, bytes)
         */
        virtual bool hostToDevice(void *dst, const void *src, size_t bytes, int device_id) = 0;

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
         */
        virtual bool deviceToDevice(void *dst, const void *src, size_t bytes, int device_id)
        {
            // Default implementation: not supported
            (void)dst;
            (void)src;
            (void)bytes;
            (void)device_id;
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
         */
        virtual bool memset(void *ptr, int value, size_t bytes, int device_id) = 0;

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
         * @return true if executed on device, false if not supported (caller should fall back)
         */
        virtual bool argmaxF32(const void *data_device, int n, int device_id,
                               float *out_value, int *out_index)
        {
            (void)data_device;
            (void)n;
            (void)device_id;
            (void)out_value;
            (void)out_index;
            return false; // Not supported by default
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
                             float *out_values, int *out_indices)
        {
            (void)data_device;
            (void)n;
            (void)k;
            (void)device_id;
            (void)out_values;
            (void)out_indices;
            return false; // Not supported by default
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
