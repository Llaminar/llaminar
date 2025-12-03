/**
 * @file KernelFactory.h
 * @brief Centralized kernel dispatch factory by device type
 * @author David Sanftenberg
 *
 * @details
 * KernelFactory provides a single point of dispatch for kernel creation based
 * on device type. This eliminates duplicate switch statements across tensor
 * types and provides a clean abstraction for adding new backends.
 *
 * ## Design Rationale
 *
 * Before: Each tensor's createGemm() had identical device routing code:
 * ```cpp
 * // Duplicated in IQ4_NLTensor, Q4_0Tensor, Q6_KTensor, etc.
 * switch (device.type) {
 * #ifdef HAVE_CUDA
 * case GPU_CUDA: return createCudaGemm(this);
 * #endif
 * default: return createCPUGemm(this);
 * }
 * ```
 *
 * After: Single dispatch through KernelFactory:
 * ```cpp
 * return KernelFactory::createGemm(DeviceType::CUDA, createCudaGemm, createCPUGemm);
 * ```
 *
 * ## Supported Device Types
 *
 * - **CPU**: OpenBLAS/MKL backends (AVX-512 VNNI JIT kernels)
 * - **CUDA**: NVIDIA GPUs (Tensor Core / WMMA)
 * - **ROCm**: AMD GPUs (Matrix Core)
 * - **Vulkan**: Cross-platform compute shaders
 * - **Metal**: Apple Silicon
 *
 * ## Usage Example
 *
 * ```cpp
 * // In IQ4_NLTensor::createGemm():
 * DeviceType dev = KernelFactory::getDeviceType(device_idx_);
 * return KernelFactory::createGemm<IQ4_NLTensor>(this, dev);
 * ```
 */

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <mutex>

// Forward declarations
namespace llaminar2
{
    class TensorBase;
    class ITensorGemm;
    class IQ4_NLTensor;
    class Q4_0Tensor;
    class Q4_1Tensor;
    class Q5_0Tensor;
    class Q5_1Tensor;
    class Q6_KTensor;
    class Q8_0Tensor;
    class Q8_1Tensor;
    class Q2_KTensor;
    class Q3_KTensor;
    class Q4_KTensor;
    class Q5_KTensor;
    class Q8_KTensor;
    class IQ1_MTensor;
    class IQ1_STensor;
    class IQ2_STensor;
    class IQ2_XSTensor;
    class IQ2_XXSTensor;
    class IQ3_STensor;
    class IQ3_XXSTensor;
    class IQ4_XSTensor;
    class FP32Tensor;
    class FP16Tensor;
    class BF16Tensor;
} // namespace llaminar2

namespace llaminar
{
    namespace v2
    {
        namespace kernels
        {
            /**
             * @brief Simplified device type enum for kernel dispatch
             *
             * This is a simplified version of ComputeBackendType that groups
             * CPU variants (OpenBLAS, MKL) together since they share kernel implementations.
             */
            enum class DeviceType
            {
                CPU,    ///< Any CPU backend (OpenBLAS, MKL, etc.)
                CUDA,   ///< NVIDIA CUDA
                ROCm,   ///< AMD ROCm/HIP
                Vulkan, ///< Vulkan compute shaders
                Metal   ///< Apple Metal Performance Shaders
            };

            /**
             * @brief Convert DeviceType to string for logging
             */
            inline std::string to_string(DeviceType type)
            {
                switch (type)
                {
                case DeviceType::CPU:
                    return "CPU";
                case DeviceType::CUDA:
                    return "CUDA";
                case DeviceType::ROCm:
                    return "ROCm";
                case DeviceType::Vulkan:
                    return "Vulkan";
                case DeviceType::Metal:
                    return "Metal";
                default:
                    return "Unknown";
                }
            }

            /**
             * @brief Centralized kernel factory for device-aware dispatch
             *
             * KernelFactory provides static methods to create kernels based on device type,
             * eliminating duplicate dispatch logic across tensor implementations.
             */
            class KernelFactory
            {
            public:
                /**
                 * @brief Get the DeviceType for a given device index
                 *
                 * @param device_idx Device index (-1 for CPU, 0+ for GPU devices)
                 * @return DeviceType for the device
                 *
                 * @throws std::runtime_error if device_idx is invalid
                 */
                static DeviceType getDeviceType(int device_idx);

                // ==========================================================================
                // GEMM Kernel Creation - One overload per tensor type
                // ==========================================================================

                /**
                 * @brief Create GEMM kernel for IQ4_NL tensor
                 * @param tensor Pointer to IQ4_NL tensor (quantized weights)
                 * @param dev_type Target device type
                 * @return ITensorGemm implementation for the device
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ4_NLTensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create raw GEMM kernel pointer for IQ4_NL tensor
                 * @param tensor Pointer to IQ4_NL tensor
                 * @param dev_type Target device type
                 * @return Raw pointer to ITensorGemm (caller owns)
                 */
                static llaminar2::ITensorGemm *createGemmRaw(
                    const llaminar2::IQ4_NLTensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for Q4_0 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q4_0Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for Q4_1 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q4_1Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for Q5_0 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q5_0Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for Q5_1 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q5_1Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for Q6_K tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q6_KTensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for Q8_0 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q8_0Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for Q8_1 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for Q2_K tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q2_KTensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for Q3_K tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q3_KTensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for Q4_K tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q4_KTensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for Q5_K tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q5_KTensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for Q8_K tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q8_KTensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for IQ1_M tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ1_MTensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for IQ1_S tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ1_STensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for IQ2_S tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ2_STensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for IQ2_XS tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ2_XSTensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for IQ2_XXS tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ2_XXSTensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for IQ3_S tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ3_STensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for IQ3_XXS tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ3_XXSTensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for IQ4_XS tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ4_XSTensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for FP32 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::FP32Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for FP16 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::FP16Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create GEMM kernel for BF16 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::BF16Tensor *tensor, DeviceType dev_type);

                // ==========================================================================
                // Cached GEMM Kernel API - Pack Once, Use Many
                // ==========================================================================

                /**
                 * @brief Get or create a cached GEMM kernel for a tensor
                 *
                 * This is the preferred API for getting GEMM kernels. It maintains a cache
                 * of kernels keyed by tensor pointer, ensuring that weight packing (the
                 * expensive operation) happens only once per weight tensor.
                 *
                 * @param tensor Weight tensor to create kernel for
                 * @return Raw pointer to cached kernel (lifetime managed by cache)
                 *
                 * @note Thread-safe via mutex protection
                 * @note Kernel is cached until clearCache() is called or program exit
                 * @note The tensor pointer is used as a key - if tensor is moved/destroyed,
                 *       call clearCache() or the entry becomes stale
                 */
                static llaminar2::ITensorGemm *getOrCreateGemm(const llaminar2::TensorBase *tensor);

                /**
                 * @brief Clear cached kernel for a specific tensor
                 *
                 * Call this when a tensor is about to be destroyed to prevent
                 * stale cache entries.
                 *
                 * @param tensor Tensor whose cached kernel should be removed
                 */
                static void clearCacheFor(const llaminar2::TensorBase *tensor);

                /**
                 * @brief Clear all cached kernels
                 *
                 * Useful for memory cleanup or when tensor lifetimes are uncertain.
                 */
                static void clearCache();

                /**
                 * @brief Get statistics about the kernel cache
                 * @return Pair of (cache_size, total_packed_bytes)
                 */
                static std::pair<size_t, size_t> cacheStats();

            private:
                KernelFactory() = delete; // Static-only class

                // Cache storage - owns the kernels
                static std::unordered_map<const llaminar2::TensorBase *, std::unique_ptr<llaminar2::ITensorGemm>> kernel_cache_;
                static std::mutex cache_mutex_;
            };

        } // namespace kernels
    } // namespace v2
} // namespace llaminar
