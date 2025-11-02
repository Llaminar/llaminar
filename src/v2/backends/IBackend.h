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

#include <cstddef>
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
    };

} // namespace llaminar2
