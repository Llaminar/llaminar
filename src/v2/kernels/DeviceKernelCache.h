/**
 * @file DeviceKernelCache.h
 * @brief Universal kernel cache keyed by (DeviceId, KernelType)
 *
 * This provides a single, clean abstraction for caching device-specific kernels.
 * Instead of creating a new kernel for every tensor or stage, we create ONE
 * kernel per (device, kernel_type) combination and share it.
 *
 * **Why This Matters**:
 * - GPU kernels have expensive JIT compilation on first invocation
 * - BLAS handles (cuBLAS, hipBLAS) are expensive to create
 * - Creating duplicate kernels wastes time and memory
 *
 * **Design**:
 * - KernelType enum identifies what kind of kernel (GEMM, Attention, etc.)
 * - IDeviceKernel is a type-erased base that all device kernels derive from
 * - DeviceKernelCache holds one kernel per (DeviceId, KernelType)
 * - Type-safe getKernel<T>() retrieves and downcasts to the expected type
 *
 * **Usage**:
 * ```cpp
 * // Get or create shared hipBLAS GEMM kernel for ROCm device 0
 * auto* gemm = DeviceKernelCache::getKernel<HipBLASGemmKernel>(
 *     DeviceId::rocm(0), KernelType::BLAS_GEMM);
 *
 * // All subsequent calls return the same cached instance
 * ```
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../backends/DeviceId.h"
#include <memory>
#include <mutex>
#include <unordered_map>
#include <functional>
#include <typeindex>
#include <stdexcept>

namespace llaminar2
{
    // =========================================================================
    // Kernel Type Enumeration
    // =========================================================================

    /**
     * @brief Types of device kernels that can be cached
     *
     * This enum identifies the "kind" of kernel, not its implementation.
     * A BLAS_GEMM kernel might be hipBLAS on ROCm, cuBLAS on CUDA, or
     * OpenBLAS on CPU - the cache stores one per device.
     */
    enum class KernelType
    {
        // BLAS operations (hold expensive handles)
        BLAS_GEMM,          ///< General matrix multiply (cuBLAS/hipBLAS/OpenBLAS)
        BLAS_GEMV,          ///< Matrix-vector multiply

        // Attention kernels (may have JIT compilation)
        FLASH_ATTENTION,    ///< Flash attention (prefill + decode)
        PAGED_ATTENTION,    ///< Paged attention for decode

        // Normalization
        RMS_NORM,           ///< RMS normalization kernel
        LAYER_NORM,         ///< Layer normalization kernel

        // Element-wise operations
        SWIGLU,             ///< SwiGLU activation
        ROPE,               ///< Rotary position embeddings
        RESIDUAL_ADD,       ///< Residual addition
        SOFTMAX,            ///< Softmax

        // Embedding / output
        EMBEDDING_LOOKUP,   ///< Token embedding lookup
        LM_HEAD,            ///< Language model head projection

        // Quantization
        QUANTIZE,           ///< Quantization kernel
        DEQUANTIZE,         ///< Dequantization kernel

        // Memory operations
        KV_CACHE,           ///< KV cache management kernel

        // Collective operations
        ALLREDUCE,          ///< MPI/NCCL allreduce
        ALLGATHER,          ///< MPI/NCCL allgather

        // Count for array sizing
        _COUNT
    };

    /**
     * @brief Convert KernelType to string for logging
     */
    inline const char* kernelTypeName(KernelType type)
    {
        switch (type)
        {
        case KernelType::BLAS_GEMM:         return "BLAS_GEMM";
        case KernelType::BLAS_GEMV:         return "BLAS_GEMV";
        case KernelType::FLASH_ATTENTION:   return "FLASH_ATTENTION";
        case KernelType::PAGED_ATTENTION:   return "PAGED_ATTENTION";
        case KernelType::RMS_NORM:          return "RMS_NORM";
        case KernelType::LAYER_NORM:        return "LAYER_NORM";
        case KernelType::SWIGLU:            return "SWIGLU";
        case KernelType::ROPE:              return "ROPE";
        case KernelType::RESIDUAL_ADD:      return "RESIDUAL_ADD";
        case KernelType::SOFTMAX:           return "SOFTMAX";
        case KernelType::EMBEDDING_LOOKUP:  return "EMBEDDING_LOOKUP";
        case KernelType::LM_HEAD:           return "LM_HEAD";
        case KernelType::QUANTIZE:          return "QUANTIZE";
        case KernelType::DEQUANTIZE:        return "DEQUANTIZE";
        case KernelType::KV_CACHE:          return "KV_CACHE";
        case KernelType::ALLREDUCE:         return "ALLREDUCE";
        case KernelType::ALLGATHER:         return "ALLGATHER";
        default:                            return "UNKNOWN";
        }
    }

    // =========================================================================
    // Device Kernel Base Interface
    // =========================================================================

    /**
     * @brief Type-erased base class for all cacheable device kernels
     *
     * This provides a common base so we can store different kernel types
     * in a single cache. Specific kernel interfaces (ITensorGemm, etc.)
     * do NOT need to inherit from this - the wrapper kernels that use
     * shared BLAS handles do.
     *
     * Examples of IDeviceKernel implementations:
     * - HipBLASGemmKernel (holds hipblas_handle_t)
     * - CUBLASGemmKernel (holds cublasHandle_t)
     * - ROCmFlashAttentionKernel (holds compiled HIP kernels)
     * - CUDAFlashAttentionKernel (holds compiled CUDA kernels)
     */
    class IDeviceKernel
    {
    public:
        virtual ~IDeviceKernel() = default;

        /**
         * @brief Get the kernel type
         */
        virtual KernelType type() const = 0;

        /**
         * @brief Get the device this kernel is bound to
         */
        virtual DeviceId device() const = 0;

        /**
         * @brief Check if kernel is ready for use
         *
         * Some kernels may have lazy initialization (JIT on first use).
         * This returns false until the kernel is fully initialized.
         */
        virtual bool isReady() const { return true; }
    };

    // =========================================================================
    // Device Kernel Cache Key
    // =========================================================================

    /**
     * @brief Cache key combining device and kernel type
     */
    struct DeviceKernelKey
    {
        DeviceId device;
        KernelType kernel_type;

        bool operator==(const DeviceKernelKey& other) const
        {
            return device == other.device && kernel_type == other.kernel_type;
        }
    };

    /**
     * @brief Hash function for DeviceKernelKey
     */
    struct DeviceKernelKeyHash
    {
        size_t operator()(const DeviceKernelKey& key) const
        {
            // Combine device hash with kernel type
            size_t h1 = std::hash<DeviceId>{}(key.device);
            size_t h2 = std::hash<int>{}(static_cast<int>(key.kernel_type));
            return h1 ^ (h2 << 1);
        }
    };

    // =========================================================================
    // Device Kernel Cache
    // =========================================================================

    /**
     * @brief Universal cache for device-specific kernels
     *
     * Thread-safe singleton cache that stores one kernel per (device, type).
     * Use getKernel<T>() to retrieve a typed kernel, or registerKernel() to
     * add a custom kernel factory.
     *
     * **Thread Safety**: All public methods are thread-safe (mutex protected).
     *
     * **Lifetime**: Kernels are owned by the cache and destroyed on clear()
     * or at program exit. Do not store raw pointers beyond immediate use.
     */
    class DeviceKernelCache
    {
    public:
        /**
         * @brief Kernel factory function type
         *
         * Takes a DeviceId and returns a unique_ptr to a new kernel.
         * Registered factories are called lazily on first getKernel() for
         * a (device, type) combination.
         */
        using KernelFactory = std::function<std::unique_ptr<IDeviceKernel>(const DeviceId&)>;

        // =====================================================================
        // Kernel Access
        // =====================================================================

        /**
         * @brief Get or create a kernel of the specified type
         *
         * If a kernel for (device, type) exists, returns it.
         * Otherwise, creates one using the registered factory.
         *
         * @tparam T Expected kernel type (for downcast)
         * @param device Target device
         * @param type Kernel type
         * @return Pointer to the kernel (owned by cache)
         * @throws std::runtime_error if no factory registered or type mismatch
         */
        template <typename T>
        static T* getKernel(const DeviceId& device, KernelType type)
        {
            IDeviceKernel* kernel = getOrCreate(device, type);
            T* typed = dynamic_cast<T*>(kernel);
            if (!typed)
            {
                throw std::runtime_error(
                    std::string("DeviceKernelCache: type mismatch for ") +
                    kernelTypeName(type) + " on " + device.to_string());
            }
            return typed;
        }

        /**
         * @brief Get kernel without type checking (returns base pointer)
         */
        static IDeviceKernel* getKernel(const DeviceId& device, KernelType type)
        {
            return getOrCreate(device, type);
        }

        /**
         * @brief Check if a kernel exists in the cache
         */
        static bool hasKernel(const DeviceId& device, KernelType type);

        // =====================================================================
        // Factory Registration
        // =====================================================================

        /**
         * @brief Register a factory for creating kernels of a specific type
         *
         * The factory will be called lazily when getKernel() is first called
         * for this (device_type, kernel_type) combination. Multiple factories
         * can be registered for different DeviceTypes (CPU, CUDA, ROCm).
         *
         * @param device_type Device type this factory handles
         * @param kernel_type Kernel type this factory creates
         * @param factory Factory function
         */
        static void registerFactory(
            DeviceType device_type,
            KernelType kernel_type,
            KernelFactory factory);

        // =====================================================================
        // Cache Management
        // =====================================================================

        /**
         * @brief Clear all cached kernels
         *
         * Use this for cleanup or to force re-creation of all kernels.
         */
        static void clear();

        /**
         * @brief Clear cached kernels for a specific device
         *
         * Useful when a device is being removed or reset.
         */
        static void clearDevice(const DeviceId& device);

        /**
         * @brief Get number of cached kernels
         */
        static size_t size();

        /**
         * @brief Get cache statistics
         * @return Pair of (total_kernels, memory_estimate_bytes)
         */
        static std::pair<size_t, size_t> stats();

    private:
        // Get or create kernel (internal implementation)
        static IDeviceKernel* getOrCreate(const DeviceId& device, KernelType type);

        // Factory key combines device type with kernel type
        struct FactoryKey
        {
            DeviceType device_type;
            KernelType kernel_type;

            bool operator==(const FactoryKey& other) const
            {
                return device_type == other.device_type && kernel_type == other.kernel_type;
            }
        };

        struct FactoryKeyHash
        {
            size_t operator()(const FactoryKey& key) const
            {
                return std::hash<int>{}(static_cast<int>(key.device_type)) ^
                       (std::hash<int>{}(static_cast<int>(key.kernel_type)) << 1);
            }
        };

        // Static members (singleton pattern)
        static std::mutex mutex_;
        static std::unordered_map<DeviceKernelKey, std::unique_ptr<IDeviceKernel>, DeviceKernelKeyHash> cache_;
        static std::unordered_map<FactoryKey, KernelFactory, FactoryKeyHash> factories_;
    };

} // namespace llaminar2
