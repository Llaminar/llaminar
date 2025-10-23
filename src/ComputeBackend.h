#pragma once

#include <memory>
#include <string>
#include <vector>

namespace llaminar
{

/**
 * @brief Enumeration of supported compute backends
 */
enum class ComputeBackendType
{
    CPU_OPENBLAS,    // OpenBLAS on CPU
    CPU_MKL,         // Intel MKL on CPU  
    GPU_CUDA,        // NVIDIA CUDA
    GPU_ROCM,        // AMD ROCm/hipBLAS
    GPU_VULKAN,      // Cross-platform Vulkan compute
    GPU_METAL        // Apple Metal (future)
};

/**
 * @brief Device descriptor for heterogeneous compute
 */
struct ComputeDevice
{
    ComputeBackendType type;
    int device_id = 0;           // GPU device index (0 for CPU)
    std::string name;            // Device name (e.g., "NVIDIA RTX 4090")
    size_t memory_bytes = 0;     // Available memory
    int compute_capability = 0;  // GPU compute capability (CUDA) or equivalent
    
    bool is_cpu() const {
        return type == ComputeBackendType::CPU_OPENBLAS || 
               type == ComputeBackendType::CPU_MKL;
    }
    
    bool is_gpu() const { return !is_cpu(); }
};

/**
 * @brief Abstract compute context for device-specific operations
 * 
 * Encapsulates device selection, memory allocation, and kernel dispatch.
 * Each backend (CUDA, ROCm, Vulkan) implements this interface.
 */
class ComputeContext
{
public:
    virtual ~ComputeContext() = default;

    // Device information
    virtual ComputeBackendType backend_type() const = 0;
    virtual const ComputeDevice& device() const = 0;
    virtual std::string device_name() const = 0;

    // Memory management
    /**
     * @brief Allocate device memory
     * @param size_bytes Allocation size
     * @return Device pointer (opaque, cast to void* or specific type)
     */
    virtual void* allocate(size_t size_bytes) = 0;
    
    /**
     * @brief Free device memory
     * @param ptr Device pointer
     */
    virtual void free(void* ptr) = 0;
    
    /**
     * @brief Copy host → device
     * @param dst Device pointer
     * @param src Host pointer
     * @param size_bytes Bytes to copy
     */
    virtual void copy_to_device(void* dst, const void* src, size_t size_bytes) = 0;
    
    /**
     * @brief Copy device → host
     * @param dst Host pointer
     * @param src Device pointer
     * @param size_bytes Bytes to copy
     */
    virtual void copy_from_device(void* dst, const void* src, size_t size_bytes) = 0;

    // Synchronization
    virtual void synchronize() = 0;

    // Kernel capabilities
    virtual bool supports_bf16() const = 0;
    virtual bool supports_fp16() const = 0;
    virtual bool supports_int8() const = 0;
};

/**
 * @brief CPU compute context (OpenBLAS/MKL)
 */
class CPUComputeContext : public ComputeContext
{
public:
    explicit CPUComputeContext(ComputeBackendType cpu_backend = ComputeBackendType::CPU_OPENBLAS);
    
    ComputeBackendType backend_type() const override { return backend_; }
    const ComputeDevice& device() const override { return device_; }
    std::string device_name() const override;

    // CPU memory is host memory (no-op for copy)
    void* allocate(size_t size_bytes) override;
    void free(void* ptr) override;
    void copy_to_device(void* dst, const void* src, size_t size_bytes) override;
    void copy_from_device(void* dst, const void* src, size_t size_bytes) override;
    void synchronize() override {} // No-op for CPU

    bool supports_bf16() const override;
    bool supports_fp16() const override { return true; }
    bool supports_int8() const override { return true; }

private:
    ComputeBackendType backend_;
    ComputeDevice device_;
};

#ifdef HAVE_CUDA
/**
 * @brief CUDA compute context (NVIDIA GPUs)
 */
class CUDAComputeContext : public ComputeContext
{
public:
    explicit CUDAComputeContext(int device_id = 0);
    ~CUDAComputeContext() override;

    ComputeBackendType backend_type() const override { return ComputeBackendType::GPU_CUDA; }
    const ComputeDevice& device() const override { return device_; }
    std::string device_name() const override { return device_.name; }

    void* allocate(size_t size_bytes) override;
    void free(void* ptr) override;
    void copy_to_device(void* dst, const void* src, size_t size_bytes) override;
    void copy_from_device(void* dst, const void* src, size_t size_bytes) override;
    void synchronize() override;

    bool supports_bf16() const override;
    bool supports_fp16() const override { return true; }
    bool supports_int8() const override { return true; }

    // CUDA-specific
    void* cuda_stream() const { return stream_; }

private:
    ComputeDevice device_;
    void* stream_ = nullptr;  // cudaStream_t (opaque)
    void query_device_properties(int device_id);
};
#endif // HAVE_CUDA

#ifdef HAVE_ROCM
/**
 * @brief ROCm compute context (AMD GPUs)
 */
class ROCmComputeContext : public ComputeContext
{
public:
    explicit ROCmComputeContext(int device_id = 0);
    ~ROCmComputeContext() override;

    ComputeBackendType backend_type() const override { return ComputeBackendType::GPU_ROCM; }
    const ComputeDevice& device() const override { return device_; }
    std::string device_name() const override { return device_.name; }

    void* allocate(size_t size_bytes) override;
    void free(void* ptr) override;
    void copy_to_device(void* dst, const void* src, size_t size_bytes) override;
    void copy_from_device(void* dst, const void* src, size_t size_bytes) override;
    void synchronize() override;

    bool supports_bf16() const override { return true; } // MI200+
    bool supports_fp16() const override { return true; }
    bool supports_int8() const override { return true; }

    // ROCm-specific
    void* hip_stream() const { return stream_; }

private:
    ComputeDevice device_;
    void* stream_ = nullptr;  // hipStream_t (opaque)
    void query_device_properties(int device_id);
};
#endif // HAVE_ROCM

#ifdef HAVE_VULKAN
/**
 * @brief Vulkan compute context (cross-platform)
 */
class VulkanComputeContext : public ComputeContext
{
public:
    explicit VulkanComputeContext(int device_id = 0);
    ~VulkanComputeContext() override;

    ComputeBackendType backend_type() const override { return ComputeBackendType::GPU_VULKAN; }
    const ComputeDevice& device() const override { return device_; }
    std::string device_name() const override { return device_.name; }

    void* allocate(size_t size_bytes) override;
    void free(void* ptr) override;
    void copy_to_device(void* dst, const void* src, size_t size_bytes) override;
    void copy_from_device(void* dst, const void* src, size_t size_bytes) override;
    void synchronize() override;

    bool supports_bf16() const override;
    bool supports_fp16() const override { return true; }
    bool supports_int8() const override { return true; }

private:
    ComputeDevice device_;
    void* vk_device_ = nullptr;       // VkDevice
    void* vk_queue_ = nullptr;        // VkQueue
    void* vk_command_pool_ = nullptr; // VkCommandPool
    void query_device_properties(int device_id);
};
#endif // HAVE_VULKAN

/**
 * @brief Device manager for heterogeneous multi-device inference
 * 
 * Supports multiple GPUs of different types on same rank:
 * - RTX 3090 (CUDA) + RX 7900 XTX (ROCm)
 * - Multiple CUDA devices
 * - Mix of discrete GPUs + integrated graphics (Vulkan)
 */
class DeviceManager
{
public:
    /**
     * @brief Get singleton instance
     */
    static DeviceManager& instance();

    /**
     * @brief Initialize and enumerate all available devices
     * 
     * Detects:
     * - All CUDA devices (if HAVE_CUDA)
     * - All ROCm devices (if HAVE_ROCM)
     * - All Vulkan devices (if HAVE_VULKAN)
     * - CPU backends (OpenBLAS, MKL)
     */
    void initialize();

    /**
     * @brief Get all available devices
     */
    const std::vector<ComputeDevice>& devices() const { return devices_; }

    /**
     * @brief Create compute context for specific device
     * 
     * @param device_index Index into devices() array
     * @return Compute context for that device
     */
    std::shared_ptr<ComputeContext> create_context(size_t device_index);

    /**
     * @brief Get device by type and device_id
     * 
     * @param type Backend type (CUDA, ROCm, etc.)
     * @param device_id Device index within that backend (0, 1, 2, ...)
     * @return Device index in devices() array, or -1 if not found
     */
    int find_device(ComputeBackendType type, int device_id = 0) const;

    /**
     * @brief Auto-select best device for operation
     * 
     * Heuristics:
     * - Prefer GPU over CPU
     * - Prefer device with most free memory
     * - Round-robin across multiple GPUs
     */
    size_t select_device(size_t estimated_memory_bytes = 0);

    /**
     * @brief Get total available compute devices
     */
    size_t device_count() const { return devices_.size(); }

    /**
     * @brief Check if any GPU is available
     */
    bool has_gpu() const;

    /**
     * @brief Get all devices of specific type
     */
    std::vector<size_t> get_devices_by_type(ComputeBackendType type) const;

private:
    DeviceManager() = default;
    std::vector<ComputeDevice> devices_;
    std::vector<std::shared_ptr<ComputeContext>> contexts_;  // Cached contexts
    size_t last_selected_device_ = 0;  // For round-robin
};

/**
 * @brief Factory for creating compute contexts (legacy, prefer DeviceManager)
 */
class ComputeContextFactory
{
public:
    /**
     * @brief Auto-detect and create optimal compute context
     * 
     * Priority: CUDA > ROCm > Vulkan > MKL > OpenBLAS
     */
    static std::shared_ptr<ComputeContext> create_auto();

    /**
     * @brief Create specific backend context
     */
    static std::shared_ptr<ComputeContext> create(ComputeBackendType type, int device_id = 0);

    /**
     * @brief Enumerate all available devices (delegates to DeviceManager)
     */
    static std::vector<ComputeDevice> enumerate_devices();

    /**
     * @brief Check if specific backend is available at compile/runtime
     */
    static bool is_available(ComputeBackendType type);
};

} // namespace llaminar
