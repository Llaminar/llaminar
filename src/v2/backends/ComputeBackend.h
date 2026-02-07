/**
 * @file ComputeBackend.h
 * @brief Device manager and compute context interfaces (LEGACY)
 *
 * ⚠️ DEPRECATION NOTICE (Phase 3 - October 2025):
 * GPU contexts (CUDAComputeContext, ROCmComputeContext, VulkanComputeContext) have been
 * disabled and replaced with IBackend architecture (see backends/IBackend.h).
 *
 * CPU device enumeration (DeviceManager) is still functional and used by Main.cpp.
 * Full removal is deferred until V2 has a production-ready device manager.
 *
 * For GPU operations, use:
 * - backends/IBackend.h (abstract interface)
 * - backends/cuda/CUDABackend.h (CUDA implementation)
 * - backends/rocm/ROCmBackend.h (ROCm implementation)
 *
 * @author David Sanftenberg
 */

#pragma once

#include "DeviceId.h"
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
        CPU,        // CPU with custom kernels (AVX-512, VNNI, etc.)
        GPU_CUDA,   // NVIDIA GPU with CUDA
        GPU_ROCM,   // AMD GPU with ROCm
        GPU_VULKAN, // Cross-platform GPU with Vulkan
        GPU_METAL   // Apple GPU with Metal
    };

    /**
     * @brief Device descriptor
     */
    struct ComputeDevice
    {
        ComputeBackendType type;
        int device_id;             // Backend-specific device ID (e.g., CUDA device 0, 1, ...)
        int numa_node;             // NUMA node/socket affinity (-1 if unknown)
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

    // Forward declarations for kernel types
    class ITensorGemm;
    class ITensorRoPE;
    class ITensorSoftmax;
    class ITensorRMSNorm;
    class ITensorSwiGLU;

    /**
     * @brief CPU compute context (OpenBLAS or Intel MKL)
     */
    class CPUComputeContext : public ComputeContext
    {
    public:
        CPUComputeContext();
        ~CPUComputeContext() override;

        void *allocate(size_t bytes) override;
        void free(void *ptr) override;
        void copy_to_device(void *dst, const void *src, size_t bytes) override;
        void copy_from_device(void *dst, const void *src, size_t bytes) override;
        void synchronize() override { /* no-op for CPU */ }

        ComputeBackendType backend_type() const override
        {
            return ComputeBackendType::CPU;
        }
        bool supports_bf16() const override { return true; } // Software emulation (or MKL native)
        bool supports_fp16() const override { return true; }
        bool supports_int8() const override { return true; }

        // Kernel access (lazily created on first access)
        ITensorRoPE *get_rope_kernel();
        ITensorSoftmax *get_softmax_kernel();
        ITensorSwiGLU *get_swiglu_kernel();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

    // ============================================================================
    // GPU Compute Contexts (DEPRECATED - Phase 3)
    // ============================================================================
    // These classes are DEPRECATED as of Phase 3. Use IBackend interface instead:
    //   - src/v2/backends/IBackend.h (abstract interface)
    //   - src/v2/backends/cuda/CUDABackend.{h,cu} (CUDA implementation)
    //   - src/v2/backends/rocm/ROCmBackend.{h,cpp} (ROCm implementation)
    //
    // The code below is kept for Main.cpp device enumeration compatibility but
    // should NOT be extended. It will be removed in Phase 4.
    // ============================================================================

#if 0 // DISABLED: GPU contexts moved to IBackend (Phase 3)

#ifdef HAVE_CUDA
/**
 * @brief CUDA compute context (DEPRECATED - use CUDABackend)
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
 * @brief ROCm compute context (DEPRECATED - use ROCmBackend)
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
 * @brief Vulkan compute context (DEPRECATED - will use IBackend)
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

#endif // #if 0 - GPU contexts disabled (Phase 3)

    /**
     * @brief Device manager singleton
     *
     * Enumerates all available compute devices and manages context creation.
     *
     * NUMA-Aware Filtering (Phase 1):
     * When initialized with a specific NUMA node, only enumerates devices
     * affine to that socket. This is critical for MPI multi-socket execution
     * to avoid cross-socket performance penalties (40-60% slower).
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
         * @brief Initialize device manager with optional NUMA filtering
         *
         * Call once at startup. Scans for:
         * - CPU (OpenBLAS/MKL) - always on local_numa_node
         * - CUDA devices (cudaGetDeviceCount) - filtered by NUMA affinity
         * - ROCm devices (hipGetDeviceCount) - filtered by NUMA affinity
         * - Vulkan devices (vkEnumeratePhysicalDevices) - not filtered (unknown affinity)
         *
         * @param local_numa_node NUMA node for this process/rank (-1 = enumerate all devices)
         *
         * Usage:
         *   // MPI rank bound to socket 0
         *   dm.initialize(0);  // Only sees GPUs on socket 0
         *
         *   // Testing or single-socket system
         *   dm.initialize(-1);  // Sees all devices
         */
        void initialize(int local_numa_node = -1);

        /**
         * @brief Get all enumerated devices
         */
        const std::vector<ComputeDevice> &devices() const { return devices_; }

        /**
         * @brief Get local NUMA node (socket) this manager is filtering for
         *
         * @return NUMA node or -1 if no filtering active
         */
        int local_numa_node() const { return local_numa_node_; }

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
         * @brief Get the default CPU device index
         *
         * CPU is always enumerated first (index 0) after initialization.
         * Use this instead of hardcoding 0 or using magic values like -1.
         *
         * @return Index of the CPU device (always 0 after initialization)
         */
        int cpuDeviceIndex() const { return 0; }

        /**
         * @brief Validate a device index
         *
         * @param device_idx Device index to validate
         * @return true if valid (>= 0 and < devices().size())
         */
        bool isValidDeviceIndex(int device_idx) const
        {
            return device_idx >= 0 && static_cast<size_t>(device_idx) < devices_.size();
        }

        /**
         * @brief Auto-select primary device (DEPRECATED - transitional)
         *
         * NOTE: This method is for legacy single-device code paths and will be
         * deprecated in favor of heterogeneous multi-device orchestration.
         *
         * Current behavior: Always returns CPU (index 0) since GPU kernels
         * are not yet implemented. All devices remain enumerated and available
         * for future heterogeneous tensor-parallel execution.
         *
         * Future direction: Work distribution will use all devices based on
         * their capabilities. CPUs may get 0% prefill but significant decode
         * work due to memory bandwidth characteristics.
         *
         * @param estimated_memory_bytes Ignored (kept for API compatibility)
         * @return Device index (currently always 0 = CPU)
         */
        size_t select_device(size_t estimated_memory_bytes = 0);

        /**
         * @brief Check if any GPU is available
         */
        bool has_gpu() const;

        /**
         * @brief Get total number of enumerated devices
         */
        size_t device_count() const { return devices_.size(); }

        /**
         * @brief Get count of CUDA devices
         */
        int cuda_device_count() const;

        /**
         * @brief Get count of ROCm devices
         */
        int rocm_device_count() const;

        /**
         * @brief Get all devices of specific type
         *
         * @param type Backend type
         * @return Vector of device indices
         */
        std::vector<size_t> get_devices_by_type(ComputeBackendType type) const;

        /**
         * @brief Get the backend-specific device ID for the N-th device of a given type
         *
         * This is useful for NUMA-aware device selection. When user specifies "rocm:0",
         * this method returns the actual HIP device ordinal of the first NUMA-local ROCm
         * device, which may differ from 0 if the rank is on NUMA node 1.
         *
         * @param type Backend type (e.g., GPU_ROCM)
         * @param local_index Index into the filtered list of devices of this type (0-based)
         * @return The backend-specific device_id, or -1 if index out of range
         */
        int get_device_id_for_type(ComputeBackendType type, int local_index) const;

        /**
         * @brief Check if a device with the given DeviceId actually exists
         *
         * Validates that the hardware device is enumerated and available.
         * Use this to fail early with a clear error message instead of
         * cascading failures deep in the stack.
         *
         * @param device DeviceId to validate
         * @return true if a matching device was found in the inventory
         */
        bool deviceExists(const DeviceId &device) const;

        /**
         * @brief Get a formatted string listing all available devices
         *
         * Useful for error messages when a requested device doesn't exist.
         * Format: "CPU, CUDA:0 (NVIDIA A100), ROCm:0 (AMD MI300X)"
         *
         * @return Comma-separated list of available devices
         */
        std::string availableDevicesString() const;

    private:
        DeviceManager() = default;

        std::vector<ComputeDevice> devices_;
        std::vector<std::shared_ptr<ComputeContext>> contexts_; // Cached per device
        size_t last_selected_device_ = 0;                       // Round-robin state
        int local_numa_node_ = -1;                              // NUMA node filter (-1 = no filter)
    };

} // namespace llaminar2
