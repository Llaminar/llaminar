/**
 * @file ComputeBackend.h
 * @brief Device manager and compute context interfaces
 *
 * Supports heterogeneous multi-GPU execution on a single MPI rank.
 * Enables mixing CUDA, ROCm, Vulkan backends simultaneously.
 *
 * @author David Sanftenberg
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstddef>

namespace llaminar2
{

    /**
     * @brief Compute backend types
     */
    enum class ComputeBackendType
    {
        CPU_OPENBLAS, // CPU with OpenBLAS
        CPU_MKL,      // CPU with Intel MKL
        GPU_CUDA,     // NVIDIA GPU with CUDA
        GPU_ROCM,     // AMD GPU with ROCm
        GPU_VULKAN,   // Cross-platform GPU with Vulkan
        GPU_METAL     // Apple GPU with Metal
    };

    /**
     * @brief Device descriptor
     */
    struct ComputeDevice
    {
        ComputeBackendType type;
        int device_id;             // Backend-specific device ID (e.g., CUDA device 0, 1, ...)
        std::string name;          // Human-readable name
        size_t total_memory_bytes; // Total device memory
        size_t free_memory_bytes;  // Free device memory (approximate)
        int compute_capability;    // Backend-specific capability (e.g., CUDA compute 8.6 → 86)
        bool supports_fp16;        // Hardware FP16 support
        bool supports_bf16;        // Hardware BF16 support
        bool supports_int8;        // Hardware INT8 support
    };

    /**
     * @brief Abstract compute context interface
     *
     * Implementations: CPUComputeContext, CUDAComputeContext, ROCmComputeContext, VulkanComputeContext
     */
    class ComputeContext
    {
    public:
        virtual ~ComputeContext() = default;

        // Memory management
        virtual void *allocate(size_t bytes) = 0;
        virtual void free(void *ptr) = 0;

        // Data transfers
        virtual void copy_to_device(void *dst, const void *src, size_t bytes) = 0;
        virtual void copy_from_device(void *dst, const void *src, size_t bytes) = 0;

        // Synchronization
        virtual void synchronize() = 0;

        // Capabilities
        virtual ComputeBackendType backend_type() const = 0;
        virtual bool supports_bf16() const = 0;
        virtual bool supports_fp16() const = 0;
        virtual bool supports_int8() const = 0;
    };

    /**
     * @brief CPU compute context (OpenBLAS)
     */
    class CPUComputeContext : public ComputeContext
    {
    public:
        void *allocate(size_t bytes) override;
        void free(void *ptr) override;
        void copy_to_device(void *dst, const void *src, size_t bytes) override;
        void copy_from_device(void *dst, const void *src, size_t bytes) override;
        void synchronize() override { /* no-op for CPU */ }

        ComputeBackendType backend_type() const override { return ComputeBackendType::CPU_OPENBLAS; }
        bool supports_bf16() const override { return true; } // Software emulation
        bool supports_fp16() const override { return true; }
        bool supports_int8() const override { return true; }
    };

#ifdef HAVE_CUDA
    /**
     * @brief CUDA compute context
     */
    class CUDAComputeContext : public ComputeContext
    {
    public:
        CUDAComputeContext() = default;

        void *allocate(size_t bytes) override;
        void free(void *ptr) override;
        void copy_to_device(void *dst, const void *src, size_t bytes) override;
        void copy_from_device(void *dst, const void *src, size_t bytes) override;
        void synchronize() override;

        ComputeBackendType backend_type() const override { return ComputeBackendType::GPU_CUDA; }
        bool supports_bf16() const override { return true; } // CUDA_R_16BF
        bool supports_fp16() const override { return true; }
        bool supports_int8() const override { return true; }

        // Public members for device management
        int device_id = 0;
        cudaStream_t stream = nullptr;
        cublasHandle_t cublas_handle = nullptr;
    };
#endif

#ifdef HAVE_ROCM
    /**
     * @brief ROCm compute context
     */
    class ROCmComputeContext : public ComputeContext
    {
    public:
        ROCmComputeContext() = default;

        void *allocate(size_t bytes) override;
        void free(void *ptr) override;
        void copy_to_device(void *dst, const void *src, size_t bytes) override;
        void copy_from_device(void *dst, const void *src, size_t bytes) override;
        void synchronize() override;

        ComputeBackendType backend_type() const override { return ComputeBackendType::GPU_ROCM; }
        bool supports_bf16() const override { return true; }
        bool supports_fp16() const override { return true; }
        bool supports_int8() const override { return true; }

        // Public members for device management
        int device_id = 0;
        hipStream_t stream = nullptr;
        hipblasHandle_t hipblas_handle = nullptr;
    };
#endif

#ifdef HAVE_VULKAN
    /**
     * @brief Vulkan compute context
     */
    class VulkanComputeContext : public ComputeContext
    {
    public:
        VulkanComputeContext() = default;

        void *allocate(size_t bytes) override;
        void free(void *ptr) override;
        void copy_to_device(void *dst, const void *src, size_t bytes) override;
        void copy_from_device(void *dst, const void *src, size_t bytes) override;
        void synchronize() override;

        ComputeBackendType backend_type() const override { return ComputeBackendType::GPU_VULKAN; }
        bool supports_bf16() const override { return false; } // Depends on extension
        bool supports_fp16() const override { return true; }
        bool supports_int8() const override { return true; }

        // Public members for device management
        int device_id = 0;
        // TODO: VkDevice, VkQueue, VkCommandPool, etc.
    };
#endif

    /**
     * @brief Device manager singleton
     *
     * Enumerates all available compute devices and manages context creation.
     */
    class DeviceManager
    {
    public:
        static DeviceManager &instance()
        {
            static DeviceManager instance;
            return instance;
        }

        /**
         * @brief Initialize device manager (enumerate all devices)
         *
         * Call once at startup. Scans for:
         * - CPU (OpenBLAS/MKL)
         * - CUDA devices (cudaGetDeviceCount)
         * - ROCm devices (hipGetDeviceCount)
         * - Vulkan devices (vkEnumeratePhysicalDevices)
         */
        void initialize();

        /**
         * @brief Get all enumerated devices
         */
        const std::vector<ComputeDevice> &devices() const { return devices_; }

        /**
         * @brief Create context for specific device
         *
         * @param device_index Index into devices() vector
         * @return Cached context (created on first call, reused thereafter)
         */
        std::shared_ptr<ComputeContext> create_context(size_t device_index);

        /**
         * @brief Find device by backend type and device ID
         *
         * @param type Backend type (e.g., GPU_CUDA)
         * @param device_id Backend-specific device ID (default: 0)
         * @return Device index in devices() vector, or -1 if not found
         */
        int find_device(ComputeBackendType type, int device_id = 0) const;

        /**
         * @brief Auto-select best device
         *
         * Heuristics:
         * 1. Prefer GPU over CPU
         * 2. Among GPUs, select one with most free memory
         * 3. Round-robin among GPUs with similar memory
         *
         * @param estimated_memory_bytes Estimated memory needed (default: 0 = don't filter)
         * @return Device index, or 0 (CPU) if no suitable GPU found
         */
        size_t select_device(size_t estimated_memory_bytes = 0);

        /**
         * @brief Check if any GPU is available
         */
        bool has_gpu() const;

        /**
         * @brief Get all devices of specific type
         *
         * @param type Backend type
         * @return Vector of device indices
         */
        std::vector<size_t> get_devices_by_type(ComputeBackendType type) const;

    private:
        DeviceManager() = default;

        std::vector<ComputeDevice> devices_;
        std::vector<std::shared_ptr<ComputeContext>> contexts_; // Cached per device
        size_t last_selected_device_ = 0;                       // Round-robin state
    };

} // namespace llaminar2
