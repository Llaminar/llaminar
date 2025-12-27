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
 * ## Activation/Weight Type Compatibility
 *
 * **CRITICAL**: Not all activation types work with all weight types!
 *
 * The QuantisedGemmKernel uses INT8 VNNI instructions (vpdpbusd) which require:
 * - Activations quantized to Q8_1 format (8-bit with scale+sum)
 * - Weights in quantized format (Q8_0, Q4_0, IQ4_NL, etc.)
 *
 * Use `isActivationWeightCompatible()` to validate before creating kernels.
 *
 * Compatibility Matrix:
 * | Activation | FP32 Weight | BF16 Weight | FP16 Weight | Quantized Weight |
 * |------------|-------------|-------------|-------------|------------------|
 * | FP32       | ✓           | ✓           | ✓           | ✓               |
 * | BF16       | ✓           | ✓           | ✓           | ✓               |
 * | FP16       | ✓           | ✓           | ✓           | ✓               |
 * | Q8_1       | ✗           | ✗           | ✗           | ✓               |
 *
 * Why Q8_1 only works with quantized weights:
 * - Q8_1 activations use INT8×INT8 dot products (AVX512-VNNI vpdpbusd)
 * - FP32/FP16/BF16 weights would require different GEMM kernel (FloatingPointGemmKernel)
 * - FloatingPointGemmKernel expects FP32 activations (not Q8_1 blocks)
 * - Mixing Q8_1 activations with FP weights produces garbage results
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
    class ITensorRoPE;
    class ITensorSwiGLU;
    class ITensorSoftmax;
    class ITensorRMSNorm;
    class ITensorAttention;
    class ITensorEmbedding;
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
    class Q16_1Tensor;
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

    enum class TensorType; // Forward declare from Tensors.h

    namespace gemm_v4
    {
        struct QuantisedPackedWeights;
    } // namespace gemm_v4
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
                // Activation/Weight Type Compatibility
                // ==========================================================================

                /**
                 * @brief Check if an activation type is compatible with a weight type for GEMM
                 *
                 * **Design Rationale**:
                 * The QuantisedGemmKernel uses INT8 VNNI instructions which require:
                 * - Activations in Q8_1 format (8-bit with scale+sum per block)
                 * - Weights in quantized format (Q8_0, Q4_0, IQ4_NL, etc.)
                 *
                 * FP32/FP16/BF16 activations use FloatingPointGemmKernel with OpenBLAS/MKL,
                 * which dequantizes weights on-the-fly. This is more flexible but slower.
                 *
                 * @param activation_type Type of the activation tensor
                 * @param weight_type Type of the weight tensor
                 * @return true if compatible, false otherwise
                 *
                 * @note FP32/FP16/BF16 activations work with ANY weight type
                 * @note Q8_1 activations ONLY work with quantized weights (not FP32/FP16/BF16)
                 *
                 * Example:
                 * @code
                 * if (!KernelFactory::isActivationWeightCompatible(TensorType::Q8_1, TensorType::FP32)) {
                 *     throw std::runtime_error("Q8_1 activations require quantized weights!");
                 * }
                 * @endcode
                 */
                static bool isActivationWeightCompatible(
                    llaminar2::TensorType activation_type,
                    llaminar2::TensorType weight_type);

                /**
                 * @brief Check if a tensor type is a floating-point format
                 *
                 * @param type Tensor type to check
                 * @return true if FP32, FP16, or BF16
                 */
                static bool isFloatingPointType(llaminar2::TensorType type);

                /**
                 * @brief Check if a tensor type is a quantized format
                 *
                 * @param type Tensor type to check
                 * @return true if any quantized format (Q8_0, Q4_0, IQ4_NL, etc.)
                 */
                static bool isQuantizedType(llaminar2::TensorType type);

                /**
                 * @brief Get a human-readable error message for incompatible types
                 *
                 * @param activation_type Activation tensor type
                 * @param weight_type Weight tensor type
                 * @return Error message explaining why types are incompatible
                 */
                static std::string getCompatibilityErrorMessage(
                    llaminar2::TensorType activation_type,
                    llaminar2::TensorType weight_type);

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
                // RoPE Kernel Creation - Device-aware dispatch
                // ==========================================================================

                /**
                 * @brief Create RoPE kernel for FP32 activation tensor
                 * @param tensor Pointer to FP32 tensor
                 * @param dev_type Target device type
                 * @return ITensorRoPE implementation for the device
                 */
                static std::unique_ptr<llaminar2::ITensorRoPE> createRoPE(
                    const llaminar2::FP32Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create RoPE kernel for BF16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorRoPE> createRoPE(
                    const llaminar2::BF16Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create RoPE kernel for FP16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorRoPE> createRoPE(
                    const llaminar2::FP16Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create RoPE kernel for Q8_1 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorRoPE> createRoPE(
                    const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type);

                // ==========================================================================
                // SwiGLU Kernel Creation - Device-aware dispatch
                // ==========================================================================

                /**
                 * @brief Create SwiGLU kernel for FP32 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorSwiGLU> createSwiGLU(
                    const llaminar2::FP32Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create SwiGLU kernel for BF16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorSwiGLU> createSwiGLU(
                    const llaminar2::BF16Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create SwiGLU kernel for FP16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorSwiGLU> createSwiGLU(
                    const llaminar2::FP16Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create SwiGLU kernel for Q8_1 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorSwiGLU> createSwiGLU(
                    const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type);

                // ==========================================================================
                // Softmax Kernel Creation - Device-aware dispatch
                // ==========================================================================

                /**
                 * @brief Create Softmax kernel for FP32 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorSoftmax> createSoftmax(
                    const llaminar2::FP32Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create Softmax kernel for BF16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorSoftmax> createSoftmax(
                    const llaminar2::BF16Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create Softmax kernel for FP16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorSoftmax> createSoftmax(
                    const llaminar2::FP16Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create Softmax kernel for Q8_1 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorSoftmax> createSoftmax(
                    const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type);

                // ==========================================================================
                // RMSNorm Kernel Creation - Device-aware dispatch
                // ==========================================================================

                /**
                 * @brief Create RMSNorm kernel for FP32 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorRMSNorm> createRMSNorm(
                    const llaminar2::FP32Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create RMSNorm kernel for BF16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorRMSNorm> createRMSNorm(
                    const llaminar2::BF16Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create RMSNorm kernel for FP16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorRMSNorm> createRMSNorm(
                    const llaminar2::FP16Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create RMSNorm kernel for Q8_1 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorRMSNorm> createRMSNorm(
                    const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create RMSNorm kernel for Q16_1 activation tensor
                 *
                 * Special case for typed residual stream: Q16_1 input → FP32 output.
                 * Used when residual is stored in high-precision Q16_1 format.
                 */
                static std::unique_ptr<llaminar2::ITensorRMSNorm> createRMSNorm(
                    const llaminar2::Q16_1Tensor *tensor, DeviceType dev_type);

                // ==========================================================================
                // Attention Kernel Creation - Device-aware dispatch
                // ==========================================================================

                /**
                 * @brief Create Attention kernel for FP32 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorAttention> createAttention(
                    const llaminar2::FP32Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create Attention kernel for BF16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorAttention> createAttention(
                    const llaminar2::BF16Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create Attention kernel for FP16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorAttention> createAttention(
                    const llaminar2::FP16Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create Attention kernel for Q8_1 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorAttention> createAttention(
                    const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type);

                // ==========================================================================
                // Embedding Kernel Creation - Device-aware dispatch
                // ==========================================================================

                /**
                 * @brief Create Embedding kernel for FP32 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorEmbedding> createEmbedding(
                    const llaminar2::FP32Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create Embedding kernel for BF16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorEmbedding> createEmbedding(
                    const llaminar2::BF16Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create Embedding kernel for FP16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorEmbedding> createEmbedding(
                    const llaminar2::FP16Tensor *tensor, DeviceType dev_type);

                /**
                 * @brief Create Embedding kernel for Q8_1 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorEmbedding> createEmbedding(
                    const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type);

                // ==========================================================================
                // Generic TensorBase* Factory Methods - Auto-dispatch by native_type()
                // ==========================================================================

                /**
                 * @brief Create RMSNorm kernel for any tensor type via dynamic dispatch
                 *
                 * Dispatches to the appropriate typed createRMSNorm overload based on
                 * tensor->native_type(). Use this when you have a TensorBase* and want
                 * to avoid manual type switching.
                 *
                 * @param tensor Input tensor (FP32, BF16, FP16, or Q8_1)
                 * @param dev_type Target device type
                 * @return ITensorRMSNorm implementation appropriate for the tensor type
                 * @throws std::runtime_error if tensor type is unsupported
                 */
                static std::unique_ptr<llaminar2::ITensorRMSNorm> createRMSNorm(
                    const llaminar2::TensorBase *tensor, DeviceType dev_type);

                /**
                 * @brief Create RoPE kernel for any tensor type via dynamic dispatch
                 *
                 * Dispatches to the appropriate typed createRoPE overload based on
                 * tensor->native_type().
                 *
                 * @param tensor Input tensor (FP32, BF16, FP16, or Q8_1)
                 * @param dev_type Target device type
                 * @return ITensorRoPE implementation appropriate for the tensor type
                 * @throws std::runtime_error if tensor type is unsupported
                 */
                static std::unique_ptr<llaminar2::ITensorRoPE> createRoPE(
                    const llaminar2::TensorBase *tensor, DeviceType dev_type);

                /**
                 * @brief Create SwiGLU kernel for any tensor type via dynamic dispatch
                 *
                 * Dispatches to the appropriate typed createSwiGLU overload based on
                 * tensor->native_type().
                 *
                 * @param tensor Input tensor (FP32, BF16, FP16, or Q8_1)
                 * @param dev_type Target device type
                 * @return ITensorSwiGLU implementation appropriate for the tensor type
                 * @throws std::runtime_error if tensor type is unsupported
                 */
                static std::unique_ptr<llaminar2::ITensorSwiGLU> createSwiGLU(
                    const llaminar2::TensorBase *tensor, DeviceType dev_type);

                /**
                 * @brief Create Attention kernel for any tensor type via dynamic dispatch
                 *
                 * Dispatches to the appropriate typed createAttention overload based on
                 * tensor->native_type().
                 *
                 * @param tensor Input tensor (FP32, BF16, FP16, or Q8_1)
                 * @param dev_type Target device type
                 * @return ITensorAttention implementation appropriate for the tensor type
                 * @throws std::runtime_error if tensor type is unsupported
                 */
                static std::unique_ptr<llaminar2::ITensorAttention> createAttention(
                    const llaminar2::TensorBase *tensor, DeviceType dev_type);

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
                 * @brief Get or create a cached row-sliced GEMM kernel for tensor parallelism
                 *
                 * Creates a kernel that only packs rows [row_start, row_end) from the weight tensor.
                 * This is used for row-parallel tensor parallelism where each MPI rank computes
                 * a slice of the output dimension.
                 *
                 * For a weight matrix [N, K]:
                 * - Full kernel: packs all N rows, output C is [M, N]
                 * - Sliced kernel: packs only (row_end - row_start) rows, output C is [M, row_end - row_start]
                 *
                 * @param tensor Weight tensor (quantized)
                 * @param row_start First row to include (0-indexed)
                 * @param row_end One past the last row to include
                 * @return Cached GEMM kernel pointer (lifetime managed by cache)
                 *
                 * @note Cache key includes row range, so different slices get different kernels
                 * @note Thread-safe via mutex protection
                 * @note The resulting kernel has N = (row_end - row_start), K unchanged
                 * @note After sliced GEMM, caller is responsible for AllReduce if needed
                 */
                static llaminar2::ITensorGemm *getOrCreateGemmSliced(
                    const llaminar2::TensorBase *tensor,
                    size_t row_start,
                    size_t row_end);

                /**
                 * @brief Ensure tensor has packed weights in its cache and return pointer
                 *
                 * This is the canonical way to get packed weights for a tensor. The weights
                 * are stored in tensor->cache_ so they outlive any individual kernel.
                 *
                 * Thread-safe: can be called from multiple threads.
                 *
                 * @param tensor Quantized tensor that implements IINT8Unpackable
                 * @return Pointer to packed weights (stored in tensor's cache_)
                 * @throws std::runtime_error if packing fails
                 */
                static const llaminar2::gemm_v4::QuantisedPackedWeights *
                ensurePackedWeightsInTensorCache(const llaminar2::TensorBase *tensor);

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

                // Sliced GEMM cache - keyed by (tensor, row_start, row_end)
                struct SlicedCacheKey
                {
                    const llaminar2::TensorBase *tensor;
                    size_t row_start;
                    size_t row_end;

                    bool operator==(const SlicedCacheKey &other) const
                    {
                        return tensor == other.tensor &&
                               row_start == other.row_start &&
                               row_end == other.row_end;
                    }
                };

                struct SlicedKeyHash
                {
                    size_t operator()(const SlicedCacheKey &k) const
                    {
                        return std::hash<const void *>()(k.tensor) ^
                               (std::hash<size_t>()(k.row_start) << 1) ^
                               (std::hash<size_t>()(k.row_end) << 2);
                    }
                };

                static std::unordered_map<SlicedCacheKey, std::unique_ptr<llaminar2::ITensorGemm>, SlicedKeyHash> sliced_cache_;
            };

        } // namespace kernels
    } // namespace v2
} // namespace llaminar
