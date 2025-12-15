/**
 * @file DeviceContext.h
 * @brief Abstract execution context for CPU and GPU devices
 * @author David Sanftenberg
 * @date December 2025
 *
 * DeviceContext provides a unified interface for executing compute operations
 * on different device types (CPU, CUDA, ROCm). It manages:
 * - Thread/stream execution context
 * - Device memory allocation
 * - Host-device transfers
 * - Synchronization primitives
 *
 * Usage:
 * @code
 *   // CPU device context with 28 threads
 *   auto cpu_ctx = CPUDeviceContext::create(0, 28);
 *   cpu_ctx->runParallel([](int tid, int nthreads) {
 *       // Each thread does work on its slice
 *   });
 *
 *   // GPU device context
 *   auto gpu_ctx = CUDADeviceContext::create(1, cuda_device_id);
 *   gpu_ctx->copyToDevice(gpu_ptr, host_ptr, bytes);
 *   gpu_ctx->synchronize();
 * @endcode
 */

#pragma once

#include "../backends/ComputeBackend.h"
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Abstract base class for device execution contexts
     *
     * Each IDeviceContext represents an execution context on a specific device.
     * For CPU, this manages OpenMP thread pool.
     * For GPU, this manages CUDA/ROCm stream and workspace memory.
     */
    class IDeviceContext
    {
    public:
        virtual ~IDeviceContext() = default;

        // =========================================================================
        // Device Information
        // =========================================================================

        /**
         * @brief Get the logical device index
         * @return Device index in DeviceManager's device list
         */
        virtual int deviceIndex() const = 0;

        /**
         * @brief Get the backend type (CPU/CUDA/ROCm/etc)
         * @return Backend type enum
         */
        virtual ComputeBackendType backendType() const = 0;

        /**
         * @brief Check if this is a GPU device
         * @return true for CUDA/ROCm/Vulkan/Metal devices
         */
        virtual bool isGPU() const = 0;

        /**
         * @brief Get human-readable device name
         * @return Device name string
         */
        virtual std::string deviceName() const = 0;

        // =========================================================================
        // Synchronization
        // =========================================================================

        /**
         * @brief Wait for all pending operations to complete
         *
         * CPU: No-op (operations are synchronous)
         * GPU: cudaStreamSynchronize or equivalent
         */
        virtual void synchronize() = 0;

        /**
         * @brief Thread barrier (for parallel regions)
         *
         * CPU: OpenMP barrier
         * GPU: No-op (use synchronize for stream sync)
         */
        virtual void barrier() = 0;

        // =========================================================================
        // Memory Management
        // =========================================================================

        /**
         * @brief Allocate device memory
         *
         * CPU: Aligned malloc with NUMA awareness
         * GPU: cudaMalloc or equivalent
         *
         * @param bytes Number of bytes to allocate
         * @return Pointer to allocated memory (nullptr on failure)
         */
        virtual void *allocate(size_t bytes) = 0;

        /**
         * @brief Free device memory
         * @param ptr Pointer returned by allocate()
         */
        virtual void free(void *ptr) = 0;

        /**
         * @brief Get workspace memory (temporary scratch space)
         *
         * Returns a pointer to at least 'bytes' of workspace memory.
         * This memory is reused across calls and should not be freed.
         * Content is NOT preserved between calls.
         *
         * @param bytes Required workspace size
         * @return Pointer to workspace memory
         */
        virtual void *getWorkspace(size_t bytes) = 0;

        /**
         * @brief Get available device memory
         * @return Free memory in bytes
         */
        virtual size_t availableMemory() const = 0;

        /**
         * @brief Get total device memory
         * @return Total memory in bytes
         */
        virtual size_t totalMemory() const = 0;

        // =========================================================================
        // Data Transfers
        // =========================================================================

        /**
         * @brief Copy data from host to device
         *
         * CPU: memcpy (immediate)
         * GPU: Async copy to device memory
         *
         * @param dst Destination pointer (device memory)
         * @param src Source pointer (host memory)
         * @param bytes Number of bytes to copy
         * @return true on success
         */
        virtual bool copyToDevice(void *dst, const void *src, size_t bytes) = 0;

        /**
         * @brief Copy data from device to host
         *
         * CPU: memcpy (immediate)
         * GPU: Async copy to host memory
         *
         * @param dst Destination pointer (host memory)
         * @param src Source pointer (device memory)
         * @param bytes Number of bytes to copy
         * @return true on success
         */
        virtual bool copyToHost(void *dst, const void *src, size_t bytes) = 0;

        /**
         * @brief Copy data between devices (or same device)
         *
         * @param dst Destination pointer
         * @param src Source pointer
         * @param bytes Number of bytes to copy
         * @param src_ctx Source device context (can be this for same-device)
         * @return true on success
         */
        virtual bool copyFromDevice(void *dst, const void *src, size_t bytes,
                                    IDeviceContext *src_ctx) = 0;

        // =========================================================================
        // Parallel Execution (CPU-specific, overridden in CPUDeviceContext)
        // =========================================================================

        /**
         * @brief Execute a function in parallel across threads
         *
         * CPU: Uses OpenMP parallel region
         * GPU: No-op (GPU parallelism is implicit in kernel launches)
         *
         * @param work Function taking (thread_id, num_threads)
         */
        virtual void runParallel(std::function<void(int, int)> work)
        {
            // Default: run on single "thread"
            work(0, 1);
        }

        /**
         * @brief Execute a worksharing loop
         *
         * CPU: Uses #pragma omp for
         * GPU: No-op
         *
         * @param start Loop start
         * @param end Loop end (exclusive)
         * @param work Function taking loop index
         */
        virtual void runFor(size_t start, size_t end, std::function<void(size_t)> work)
        {
            for (size_t i = start; i < end; ++i)
            {
                work(i);
            }
        }

        /**
         * @brief Get number of parallel threads/workers
         * @return Thread count (CPU) or 1 (GPU, parallelism is in kernels)
         */
        virtual int numThreads() const { return 1; }

        // =========================================================================
        // Factory
        // =========================================================================

        /**
         * @brief Create a device context for the given device index
         *
         * Automatically selects CPU or GPU context based on device type.
         *
         * @param device_idx Device index in DeviceManager
         * @param num_threads Number of threads for CPU devices (ignored for GPU)
         * @return Unique pointer to device context (nullptr on failure)
         */
        static std::unique_ptr<IDeviceContext> create(int device_idx, int num_threads = 0);
    };

    /**
     * @brief CPU device context using OpenMP for parallelism
     */
    class CPUDeviceContext : public IDeviceContext
    {
    public:
        /**
         * @brief Construct CPU context
         * @param device_idx Logical device index
         * @param num_threads OpenMP thread count (0 = use OMP_NUM_THREADS)
         */
        explicit CPUDeviceContext(int device_idx, int num_threads = 0);
        ~CPUDeviceContext() override;

        // Device info
        int deviceIndex() const override { return device_idx_; }
        ComputeBackendType backendType() const override { return ComputeBackendType::CPU_OPENBLAS; }
        bool isGPU() const override { return false; }
        std::string deviceName() const override;

        // Synchronization
        void synchronize() override; // No-op for CPU
        void barrier() override;     // OpenMP barrier

        // Memory
        void *allocate(size_t bytes) override;
        void free(void *ptr) override;
        void *getWorkspace(size_t bytes) override;
        size_t availableMemory() const override;
        size_t totalMemory() const override;

        // Transfers (memcpy for CPU)
        bool copyToDevice(void *dst, const void *src, size_t bytes) override;
        bool copyToHost(void *dst, const void *src, size_t bytes) override;
        bool copyFromDevice(void *dst, const void *src, size_t bytes,
                            IDeviceContext *src_ctx) override;

        // Parallel execution
        void runParallel(std::function<void(int, int)> work) override;
        void runFor(size_t start, size_t end, std::function<void(size_t)> work) override;
        int numThreads() const override { return num_threads_; }

        /**
         * @brief Check if currently in a parallel region
         * @return true if omp_in_parallel() is true
         */
        bool inParallelRegion() const;

    private:
        int device_idx_;
        int num_threads_;
        std::vector<char> workspace_;
        size_t workspace_size_ = 0;
    };

    /**
     * @brief Abstract base class for GPU device contexts (CUDA/ROCm)
     *
     * Provides common patterns for GPU contexts:
     * - Backend delegation (IBackend* for actual operations)
     * - Stream/workspace management
     * - Memory tracking
     *
     * Derived classes (CUDADeviceContext, ROCmDeviceContext) provide
     * backend-specific implementations.
     */
    class IGPUDeviceContext : public IDeviceContext
    {
    public:
        /**
         * @brief Construct GPU context
         * @param device_idx Logical device index in DeviceManager
         * @param gpu_device_id Physical GPU device ID
         * @param backend_type CUDA or ROCm backend type
         */
        IGPUDeviceContext(int device_idx, int gpu_device_id, ComputeBackendType backend_type);
        ~IGPUDeviceContext() override;

        // Device info
        int deviceIndex() const override { return device_idx_; }
        ComputeBackendType backendType() const override { return backend_type_; }
        bool isGPU() const override { return true; }
        int gpuDeviceId() const { return gpu_device_id_; }

        // Memory (delegates to derived class backend operations)
        void *getWorkspace(size_t bytes) override;

        // Parallel execution (no-op for GPU - parallelism is in kernels)
        void runParallel(std::function<void(int, int)> work) override;
        void runFor(size_t start, size_t end, std::function<void(size_t)> work) override;
        int numThreads() const override { return 1; }

    protected:
        int device_idx_;
        int gpu_device_id_;
        ComputeBackendType backend_type_;
        void *workspace_ = nullptr;
        size_t workspace_size_ = 0;
    };

// Forward declaration for GPU context (implemented when CUDA/ROCm available)
#ifdef HAVE_CUDA
    /**
     * @brief CUDA device context
     *
     * Uses CUDABackend for actual operations, manages CUDA stream
     * and device memory allocations.
     */
    class CUDADeviceContext : public IGPUDeviceContext
    {
    public:
        /**
         * @brief Construct CUDA context
         * @param device_idx Logical device index
         * @param cuda_device_id Physical CUDA device ID
         */
        CUDADeviceContext(int device_idx, int cuda_device_id);
        ~CUDADeviceContext() override;

        std::string deviceName() const override;

        // Synchronization
        void synchronize() override;
        void barrier() override;

        // Memory
        void *allocate(size_t bytes) override;
        void free(void *ptr) override;
        size_t availableMemory() const override;
        size_t totalMemory() const override;

        // Transfers
        bool copyToDevice(void *dst, const void *src, size_t bytes) override;
        bool copyToHost(void *dst, const void *src, size_t bytes) override;
        bool copyFromDevice(void *dst, const void *src, size_t bytes,
                            IDeviceContext *src_ctx) override;
    };
#endif

#ifdef HAVE_ROCM
    /**
     * @brief ROCm/HIP device context
     *
     * Uses ROCmBackend for actual operations, manages HIP stream
     * and device memory allocations.
     */
    class ROCmDeviceContext : public IGPUDeviceContext
    {
    public:
        /**
         * @brief Construct ROCm context
         * @param device_idx Logical device index
         * @param hip_device_id Physical HIP device ID
         */
        ROCmDeviceContext(int device_idx, int hip_device_id);
        ~ROCmDeviceContext() override;

        std::string deviceName() const override;

        // Synchronization
        void synchronize() override;
        void barrier() override;

        // Memory
        void *allocate(size_t bytes) override;
        void free(void *ptr) override;
        size_t availableMemory() const override;
        size_t totalMemory() const override;

        // Transfers
        bool copyToDevice(void *dst, const void *src, size_t bytes) override;
        bool copyToHost(void *dst, const void *src, size_t bytes) override;
        bool copyFromDevice(void *dst, const void *src, size_t bytes,
                            IDeviceContext *src_ctx) override;
    };
#endif // HAVE_ROCM

} // namespace llaminar2
