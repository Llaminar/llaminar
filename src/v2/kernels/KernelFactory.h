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

#include "../backends/DeviceType.h"            // Shared DeviceType enum
#include "../backends/DeviceId.h"              // Type-safe device identification
#include "../execution/config/RuntimeConfig.h" // ActivationPrecision
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <mutex>

// Forward declarations
namespace llaminar2
{
    // KVCache types
    class IKVCache;
    class ICPUKVCache;
    class ICUDARingKVCache;
    class MPIContext;
    enum class KVCacheLayoutMode : uint8_t;
    class ITensor;
    class TensorBase;
    class ITensorGemm;
    class ITensorRoPE;
    class ITensorSwiGLU;
    class ITensorSoftmax;
    class ITensorRMSNorm;
    class ITensorResidualAdd;
    class ITensorAttention;
    class ITensorEmbedding;
    class ITensorFusedQKVGemm;
    class ITensorFusedGateUpGemm;
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

    namespace cuda
    {
        struct CUDAPackedWeights; // Forward declare from CUDAQuantisedGemmKernel.h
    } // namespace cuda

    namespace rocm
    {
        struct ROCmPackedWeights; // Forward declare from ROCmQuantisedGemmKernel.h
    } // namespace rocm

} // namespace llaminar2

namespace llaminar
{
    namespace v2
    {
        namespace kernels
        {
            // Use the shared DeviceType from backends/DeviceType.h
            using ::llaminar2::DeviceType;

            /**
             * @brief Canonical kernel family identifier for device-scoped registry
             *
             * Phase A scaffold: no behavior changes in existing code paths.
             */
            enum class KernelKind
            {
                GEMM,
                ROPE,
                RMSNORM,
                SWIGLU,
                SOFTMAX,
                RESIDUAL_ADD,
                ATTENTION,
                EMBEDDING,
                FUSED_QKV,
                FUSED_GATE_UP,
            };

            /**
             * @brief Device-scoped kernel registry key (Phase A scaffold)
             */
            struct DeviceKernelKey
            {
                llaminar2::DeviceId device_id;
                KernelKind kind{KernelKind::GEMM};
                int variant{0}; // Precision/layout/algorithm discriminator

                bool operator==(const DeviceKernelKey &other) const
                {
                    return device_id == other.device_id &&
                           kind == other.kind &&
                           variant == other.variant;
                }
            };

            struct DeviceKernelKeyHash
            {
                size_t operator()(const DeviceKernelKey &k) const
                {
                    return std::hash<int>()(static_cast<int>(k.device_id.type)) ^
                           (std::hash<int>()(k.device_id.ordinal) << 1) ^
                           (std::hash<int>()(static_cast<int>(k.kind)) << 2) ^
                           (std::hash<int>()(k.variant) << 3);
                }
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

            // ==========================================================================
            // KVCache Configuration
            // ==========================================================================

            /**
             * @brief Configuration for KV cache creation
             *
             * Encapsulates all parameters needed to create a KV cache, supporting both
             * CPU and CUDA backends with optional tensor parallelism (sharding).
             *
             * @see KernelFactory::createKVCache()
             */
            struct KVCacheConfig
            {
                // Required parameters
                ::llaminar2::ActivationPrecision precision = ::llaminar2::ActivationPrecision::FP32;
                ::llaminar2::DeviceId device = ::llaminar2::DeviceId::cpu();
                int num_layers = 0;
                int batch_size = 1;
                int max_seq_len = 2048;
                int n_kv_heads = 0;
                int head_dim = 64;

                // Optional sharding (for tensor parallelism)
                int local_n_kv_heads = 0; ///< 0 = use full n_kv_heads (no sharding)
                int kv_head_start = 0;    ///< Starting KV head index for this rank

                /// First layer index for Pipeline Parallelism (PP).
                /// When set (non-zero), the KV cache remaps global layer indices:
                /// - Incoming layer_idx is remapped to (layer_idx - first_layer_index)
                /// - This allows PP stage 1 with layers [12,23] to use cache layers [0,11]
                /// Default is 0 (no remapping).
                int first_layer_index = 0;

                // Layout mode - forward declared, defined in CPUKVCache.h
                ::llaminar2::KVCacheLayoutMode layout_mode{}; // Default-initialized (POSITION_MAJOR = 0)

                // MPI context (required for CPU cache)
                const ::llaminar2::MPIContext *mpi_ctx = nullptr;

                /**
                 * @brief Check if this config requests sharded (tensor-parallel) cache
                 * @return true if local_n_kv_heads > 0 and < n_kv_heads
                 */
                bool is_sharded() const
                {
                    return local_n_kv_heads > 0 && local_n_kv_heads < n_kv_heads;
                }

                /**
                 * @brief Check if this config targets a CUDA device
                 * @return true if device is a CUDA GPU
                 */
                bool is_cuda() const
                {
                    return device.is_cuda();
                }

                /**
                 * @brief Check if this config targets a ROCm device
                 * @return true if device is an AMD GPU
                 */
                bool is_rocm() const
                {
                    return device.is_rocm();
                }
            };

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
                 * @brief Get the DeviceType for a given DeviceId
                 *
                 * @param device_id Type-safe device identifier
                 * @return DeviceType for the device
                 *
                 * @throws std::runtime_error if device_id is invalid
                 */
                static DeviceType getDeviceType(llaminar2::DeviceId device_id);

                /**
                 * @brief Get the DeviceType for a given device index (legacy)
                 *
                 * @param device_idx Device index (-1 for CPU, 0+ for GPU devices)
                 * @return DeviceType for the device
                 * @deprecated Use getDeviceType(DeviceId) instead
                 *
                 * @throws std::runtime_error if device_idx is invalid
                 */
                static DeviceType getDeviceType(int device_idx);

                // ==========================================================================
                // CUDA Device Ordinal Threading Support
                // ==========================================================================

                /**
                 * @brief RAII guard to set thread-local CUDA device ordinal for kernel creation
                 *
                 * In multi-GPU CUDA setups, when creating prepared GEMM handles,
                 * we need to know which specific CUDA device to target. This guard sets a thread-local
                 * variable that getCUDADeviceIdForKernel() will use.
                 *
                 * Usage:
                 * @code
                 * {
                 *     KernelFactory::CUDAOrdinalGuard guard(1); // Target CUDA device 1
                 *     auto* prepared = KernelFactory::getOrCreatePreparedGemmWeights(tensor, DeviceId::cuda(1));
                 *     auto* kernel = KernelFactory::getOrCreateGemmEngine(prepared);
                 *     // kernel will target CUDA device 1
                 * } // guard goes out of scope, thread-local cleared
                 * @endcode
                 */
                struct CUDAOrdinalGuard
                {
                    CUDAOrdinalGuard(int ordinal);
                    ~CUDAOrdinalGuard();
                    CUDAOrdinalGuard(const CUDAOrdinalGuard &) = delete;
                    CUDAOrdinalGuard &operator=(const CUDAOrdinalGuard &) = delete;
                };

                // ==========================================================================
                // ROCm Device Ordinal Threading Support
                // ==========================================================================

                /**
                 * @brief RAII guard to set thread-local ROCm device ordinal for kernel creation
                 *
                 * In multi-GPU ROCm setups, when creating prepared GEMM handles,
                 * we need to know which specific ROCm device to target. This guard sets a thread-local
                 * variable that getROCmDeviceIdForKernel() will use.
                 *
                 * Usage:
                 * @code
                 * {
                 *     KernelFactory::ROCmOrdinalGuard guard(1); // Target ROCm device 1
                 *     auto* prepared = KernelFactory::getOrCreatePreparedGemmWeights(tensor, DeviceId::rocm(1));
                 *     auto* kernel = KernelFactory::getOrCreateGemmEngine(prepared);
                 *     // kernel will target ROCm device 1
                 * } // guard goes out of scope, thread-local cleared
                 * @endcode
                 */
                struct ROCmOrdinalGuard
                {
                    ROCmOrdinalGuard(int ordinal);
                    ~ROCmOrdinalGuard();
                    ROCmOrdinalGuard(const ROCmOrdinalGuard &) = delete;
                    ROCmOrdinalGuard &operator=(const ROCmOrdinalGuard &) = delete;
                };

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
                 * @brief Create GEMM kernel for any tensor type (dynamic dispatch)
                 * @param tensor Pointer to TensorBase (any quantized or float tensor)
                 * @param dev_type Target device type
                 * @return ITensorGemm implementation for the device
                 * @throws std::runtime_error if tensor type not supported on target device
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::TensorBase *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for IQ4_NL tensor
                 * @param tensor Pointer to IQ4_NL tensor (quantized weights)
                 * @param dev_type Target device type
                 * @return ITensorGemm implementation for the device
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ4_NLTensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create raw GEMM kernel pointer for IQ4_NL tensor
                 * @param tensor Pointer to IQ4_NL tensor
                 * @param dev_type Target device type
                 * @return Raw pointer to ITensorGemm (caller owns)
                 */
                static llaminar2::ITensorGemm *createGemmRaw(
                    const llaminar2::IQ4_NLTensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for Q4_0 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q4_0Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for Q4_1 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q4_1Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for Q5_0 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q5_0Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for Q5_1 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q5_1Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for Q6_K tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q6_KTensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for Q8_0 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q8_0Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for Q8_1 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for Q2_K tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q2_KTensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for Q3_K tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q3_KTensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for Q4_K tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q4_KTensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for Q5_K tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q5_KTensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for Q8_K tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::Q8_KTensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for IQ1_M tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ1_MTensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for IQ1_S tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ1_STensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for IQ2_S tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ2_STensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for IQ2_XS tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ2_XSTensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for IQ2_XXS tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ2_XXSTensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for IQ3_S tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ3_STensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for IQ3_XXS tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ3_XXSTensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for IQ4_XS tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::IQ4_XSTensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for FP32 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::FP32Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for FP16 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::FP16Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create GEMM kernel for BF16 tensor
                 */
                static std::unique_ptr<llaminar2::ITensorGemm> createGemm(
                    const llaminar2::BF16Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                // ==========================================================================
                // Fused QKV GEMM Kernel Creation
                // ==========================================================================

                /**
                 * @brief Create fused QKV GEMM kernel
                 *
                 * Creates a kernel that performs three concurrent GEMM operations
                 * for Q, K, V projections with shared activation quantization.
                 *
                 * @param wq Q weight tensor
                 * @param wk K weight tensor
                 * @param wv V weight tensor
                 * @param dev_type Target device type
                 * @return Kernel instance (caller owns)
                 *
                 * @note Compatibility overload. Prefer explicit DeviceId overload.
                 */
                static std::unique_ptr<llaminar2::ITensorFusedQKVGemm> createFusedQKVGemm(
                    const llaminar2::TensorBase *wq,
                    const llaminar2::TensorBase *wk,
                    const llaminar2::TensorBase *wv,
                    DeviceType dev_type = DeviceType::CPU);

                /**
                 * @brief Create fused QKV GEMM kernel with explicit DeviceId
                 *
                 * Preferred create path for deterministic multi-device behavior.
                 *
                 * Internally, this resolves prepared handles for all three projection
                 * weights on `target_device` and constructs the fused adapter from
                 * those prepared identities.
                 */
                static std::unique_ptr<llaminar2::ITensorFusedQKVGemm> createFusedQKVGemm(
                    const llaminar2::TensorBase *wq,
                    const llaminar2::TensorBase *wk,
                    const llaminar2::TensorBase *wv,
                    llaminar2::DeviceId target_device);

                /**
                 * @brief Get or create cached fused QKV GEMM kernel
                 *
                 * Maintains a cache of kernels keyed by prepared-handle identity
                 * (wq, wk, wv prepared handles + device), ensuring prepared GEMM
                 * state and fused kernel identity stay aligned.
                 *
                 * @param wq Q weight tensor (used as primary cache key)
                 * @param wk K weight tensor
                 * @param wv V weight tensor
                 * @param dev_type Target device type
                 * @return Kernel pointer (factory owns lifetime)
                 *
                 * @note Compatibility overload. For multi-GPU callers, prefer
                 * explicit DeviceId overload to avoid implicit ordinal resolution.
                 */
                static llaminar2::ITensorFusedQKVGemm *getOrCreateFusedQKVGemm(
                    const llaminar2::TensorBase *wq,
                    const llaminar2::TensorBase *wk,
                    const llaminar2::TensorBase *wv,
                    DeviceType dev_type = DeviceType::CPU);

                /**
                 * @brief Get or create cached fused QKV GEMM kernel with explicit DeviceId
                 *
                 * Preferred API for multi-device execution because it avoids
                 * implicit thread-local ordinal resolution.
                 *
                 * Cache keying is based on prepared-handle identity plus device,
                 * so cache membership reflects actual backend-ready state instead
                 * of only raw tensor pointers.
                 */
                static llaminar2::ITensorFusedQKVGemm *getOrCreateFusedQKVGemm(
                    const llaminar2::TensorBase *wq,
                    const llaminar2::TensorBase *wk,
                    const llaminar2::TensorBase *wv,
                    llaminar2::DeviceId target_device);

                // ==========================================================================
                // Fused Gate/Up GEMM Kernel Creation
                // ==========================================================================

                /**
                 * @brief Create fused Gate/Up GEMM kernel for SwiGLU FFN
                 *
                 * Creates a kernel that wraps two individual GEMM kernels for
                 * gate and up projections. The adapter manages the underlying
                 * GEMM kernels through prepared-handle-aware engine resolution.
                 *
                 * @param w_gate Gate weight tensor
                 * @param w_up Up weight tensor
                 * @param dev_type Target device type
                 * @return Kernel instance (caller owns)
                 */
                static std::unique_ptr<llaminar2::ITensorFusedGateUpGemm> createFusedGateUpGemm(
                    const llaminar2::TensorBase *w_gate,
                    const llaminar2::TensorBase *w_up,
                    DeviceType dev_type = DeviceType::CPU);

                /**
                 * @brief Create fused Gate/Up GEMM kernel with explicit DeviceId
                 *
                 * Same as createFusedGateUpGemm but uses DeviceId to specify BOTH
                 * the device type AND ordinal. Use for multi-GPU scenarios.
                 *
                 * @param w_gate Gate weight tensor
                 * @param w_up Up weight tensor
                 * @param target_device DeviceId specifying type and ordinal
                 * @return Kernel instance (caller owns)
                 */
                static std::unique_ptr<llaminar2::ITensorFusedGateUpGemm> createFusedGateUpGemm(
                    const llaminar2::TensorBase *w_gate,
                    const llaminar2::TensorBase *w_up,
                    llaminar2::DeviceId target_device);

                /**
                 * @brief Get or create cached fused Gate/Up GEMM kernel
                 *
                 * Compatibility overload for call sites that only provide DeviceType.
                 * Internally delegates to the DeviceId overload.
                 *
                 * @param w_gate Gate weight tensor
                 * @param w_up Up weight tensor
                 * @param dev_type Target device type
                 * @return Kernel pointer (factory owns lifetime)
                 */
                static llaminar2::ITensorFusedGateUpGemm *getOrCreateFusedGateUpGemm(
                    const llaminar2::TensorBase *w_gate,
                    const llaminar2::TensorBase *w_up,
                    DeviceType dev_type = DeviceType::CPU);

                /**
                 * @brief Get or create cached fused Gate/Up GEMM kernel with explicit DeviceId
                 *
                 * Same as getOrCreateFusedGateUpGemm but uses DeviceId to specify BOTH
                 * the device type AND the device ordinal (e.g., ROCm:1 vs ROCm:0).
                 * Use this when weights are on CPU but need to run on a specific GPU.
                 *
                 * @param w_gate Gate weight tensor
                 * @param w_up Up weight tensor
                 * @param target_device DeviceId specifying type and ordinal
                 * @return Kernel pointer (factory owns lifetime)
                 *
                 * @note Cache key includes prepared-handle identity and DeviceId,
                 * so entries stay aligned with per-device prepared GEMM readiness.
                 */
                static llaminar2::ITensorFusedGateUpGemm *getOrCreateFusedGateUpGemm(
                    const llaminar2::TensorBase *w_gate,
                    const llaminar2::TensorBase *w_up,
                    llaminar2::DeviceId target_device);

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
                    const llaminar2::FP32Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create RoPE kernel for BF16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorRoPE> createRoPE(
                    const llaminar2::BF16Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create RoPE kernel for FP16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorRoPE> createRoPE(
                    const llaminar2::FP16Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create RoPE kernel for Q8_1 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorRoPE> createRoPE(
                    const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                // ==========================================================================
                // SwiGLU Kernel Creation - Device-aware dispatch
                // ==========================================================================

                /**
                 * @brief Create SwiGLU kernel for FP32 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorSwiGLU> createSwiGLU(
                    const llaminar2::FP32Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create SwiGLU kernel for BF16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorSwiGLU> createSwiGLU(
                    const llaminar2::BF16Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create SwiGLU kernel for FP16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorSwiGLU> createSwiGLU(
                    const llaminar2::FP16Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create SwiGLU kernel for Q8_1 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorSwiGLU> createSwiGLU(
                    const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                // ==========================================================================
                // Softmax Kernel Creation - Device-aware dispatch
                // ==========================================================================

                /**
                 * @brief Create Softmax kernel for FP32 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorSoftmax> createSoftmax(
                    const llaminar2::FP32Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create Softmax kernel for BF16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorSoftmax> createSoftmax(
                    const llaminar2::BF16Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create Softmax kernel for FP16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorSoftmax> createSoftmax(
                    const llaminar2::FP16Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create Softmax kernel for Q8_1 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorSoftmax> createSoftmax(
                    const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                // ==========================================================================
                // RMSNorm Kernel Creation - Device-aware dispatch
                // ==========================================================================

                /**
                 * @brief Create RMSNorm kernel for FP32 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorRMSNorm> createRMSNorm(
                    const llaminar2::FP32Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create RMSNorm kernel for BF16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorRMSNorm> createRMSNorm(
                    const llaminar2::BF16Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create RMSNorm kernel for FP16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorRMSNorm> createRMSNorm(
                    const llaminar2::FP16Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create RMSNorm kernel for Q8_1 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorRMSNorm> createRMSNorm(
                    const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create RMSNorm kernel for Q16_1 activation tensor
                 *
                 * Special case for typed residual stream: Q16_1 input → FP32 output.
                 * Used when residual is stored in high-precision Q16_1 format.
                 */
                static std::unique_ptr<llaminar2::ITensorRMSNorm> createRMSNorm(
                    const llaminar2::Q16_1Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                // ==========================================================================
                // Attention Kernel Creation - Device-aware dispatch
                // ==========================================================================

                /**
                 * @brief Create Attention kernel for FP32 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorAttention> createAttention(
                    const llaminar2::FP32Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create Attention kernel for BF16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorAttention> createAttention(
                    const llaminar2::BF16Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create Attention kernel for FP16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorAttention> createAttention(
                    const llaminar2::FP16Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create Attention kernel for Q8_1 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorAttention> createAttention(
                    const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                // ==========================================================================
                // Embedding Kernel Creation - Device-aware dispatch
                // ==========================================================================

                /**
                 * @brief Create Embedding kernel for FP32 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorEmbedding> createEmbedding(
                    const llaminar2::FP32Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create Embedding kernel for BF16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorEmbedding> createEmbedding(
                    const llaminar2::BF16Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create Embedding kernel for FP16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorEmbedding> createEmbedding(
                    const llaminar2::FP16Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create Embedding kernel for Q8_1 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorEmbedding> createEmbedding(
                    const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                // ==========================================================================
                // ResidualAdd Kernel Creation - Device-aware dispatch
                // ==========================================================================

                /**
                 * @brief Create ResidualAdd kernel for FP32 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorResidualAdd> createResidualAdd(
                    const llaminar2::FP32Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create ResidualAdd kernel for BF16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorResidualAdd> createResidualAdd(
                    const llaminar2::BF16Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create ResidualAdd kernel for FP16 activation tensor
                 */
                static std::unique_ptr<llaminar2::ITensorResidualAdd> createResidualAdd(
                    const llaminar2::FP16Tensor *tensor, DeviceType dev_type, int device_ordinal = -1);

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
                    const llaminar2::TensorBase *tensor, DeviceType dev_type, int device_ordinal = -1);

                // ==========================================================================
                // Device-scoped cached non-GEMM kernels
                // ==========================================================================

                /**
                 * @brief Get or create a device-scoped RoPE kernel
                 *
                 * Cache key is (target_device, tensor native_type). This ensures
                 * a single RoPE kernel instance per device+precision variant.
                 */
                static llaminar2::ITensorRoPE *getOrCreateRoPE(
                    const llaminar2::TensorBase *tensor,
                    llaminar2::DeviceId target_device);

                /**
                 * @brief Get or create a device-scoped RMSNorm kernel
                 *
                 * Cache key is (target_device, tensor native_type). This ensures
                 * a single RMSNorm kernel instance per device+precision variant.
                 */
                static llaminar2::ITensorRMSNorm *getOrCreateRMSNorm(
                    const llaminar2::TensorBase *tensor,
                    llaminar2::DeviceId target_device);

                /**
                 * @brief Get or create a device-scoped SwiGLU kernel
                 *
                 * Cache key is (target_device, tensor native_type). This ensures
                 * a single SwiGLU kernel instance per device+precision variant.
                 */
                static llaminar2::ITensorSwiGLU *getOrCreateSwiGLU(
                    const llaminar2::TensorBase *tensor,
                    llaminar2::DeviceId target_device);

                /**
                 * @brief Get or create a device-scoped Softmax kernel
                 *
                 * Cache key is (target_device, tensor native_type). This ensures
                 * a single Softmax kernel instance per device+precision variant.
                 */
                static llaminar2::ITensorSoftmax *getOrCreateSoftmax(
                    const llaminar2::TensorBase *tensor,
                    llaminar2::DeviceId target_device);

                /**
                 * @brief Get or create a device-scoped ResidualAdd kernel
                 *
                 * Cache key is (target_device, tensor native_type). This ensures
                 * a single ResidualAdd kernel instance per device+precision variant.
                 */
                static llaminar2::ITensorResidualAdd *getOrCreateResidualAdd(
                    const llaminar2::TensorBase *tensor,
                    llaminar2::DeviceId target_device);

                /**
                 * @brief Get or create a device-scoped Attention kernel
                 *
                 * Cache key is (target_device, tensor native_type). This ensures
                 * a single Attention kernel instance per device+precision variant.
                 */
                static llaminar2::ITensorAttention *getOrCreateAttention(
                    const llaminar2::ITensor *tensor,
                    llaminar2::DeviceId target_device);

                /**
                 * @brief Get or create a device-scoped Embedding kernel
                 *
                 * Cache key is (target_device, tensor native_type). This ensures
                 * a single Embedding kernel instance per device+precision variant.
                 */
                static llaminar2::ITensorEmbedding *getOrCreateEmbedding(
                    const llaminar2::TensorBase *tensor,
                    llaminar2::DeviceId target_device);

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
                    const llaminar2::TensorBase *tensor, DeviceType dev_type, int device_ordinal = -1);

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
                    const llaminar2::TensorBase *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create Softmax kernel for any tensor type via dynamic dispatch
                 *
                 * Dispatches to the appropriate typed createSoftmax overload based on
                 * tensor->native_type().
                 *
                 * @param tensor Input tensor (FP32, BF16, FP16, or Q8_1)
                 * @param dev_type Target device type
                 * @return ITensorSoftmax implementation appropriate for the tensor type
                 * @throws std::runtime_error if tensor type is unsupported
                 */
                static std::unique_ptr<llaminar2::ITensorSoftmax> createSoftmax(
                    const llaminar2::TensorBase *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create ResidualAdd kernel for any tensor type via dynamic dispatch
                 *
                 * Dispatches to the appropriate typed createResidualAdd overload based on
                 * tensor->native_type().
                 *
                 * @param tensor Input tensor (FP32, BF16, or FP16)
                 * @param dev_type Target device type
                 * @return ITensorResidualAdd implementation appropriate for the tensor type
                 * @throws std::runtime_error if tensor type is unsupported
                 */
                static std::unique_ptr<llaminar2::ITensorResidualAdd> createResidualAdd(
                    const llaminar2::TensorBase *tensor, DeviceType dev_type, int device_ordinal = -1);

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
                    const llaminar2::TensorBase *tensor, DeviceType dev_type, int device_ordinal = -1);

                /**
                 * @brief Create Attention kernel for any ITensor via device-aware dispatch
                 *
                 * For GPU tensors (is_on_gpu() == true), creates CUDAFlashAttentionKernelT.
                 * For CPU tensors, delegates to the TensorBase* overload.
                 *
                 * @param tensor Input tensor (ITensor* - works with both CPU and GPU tensors)
                 * @param dev_type Target device type
                 * @return ITensorAttention implementation appropriate for the tensor type and device
                 * @throws std::runtime_error if tensor type is unsupported
                 */
                static std::unique_ptr<llaminar2::ITensorAttention> createAttention(
                    const llaminar2::ITensor *tensor, DeviceType dev_type, int device_ordinal = -1);

                // ==========================================================================
                // KVCache Factory Methods
                // ==========================================================================

                /**
                 * @brief Create a KV cache with device-aware dispatch
                 *
                 * Dispatches to the appropriate backend based on config.device:
                 * - CPU device → createCPUKVCache()
                 * - CUDA device → createCUDAKVCache() (with HAVE_CUDA guard)
                 *
                 * @param config KVCacheConfig with all required parameters
                 * @return Unique pointer to IKVCache (unified interface for CPU/CUDA)
                 * @throws std::runtime_error if device type is not supported or config is invalid
                 */
                static std::unique_ptr<llaminar2::IKVCache> createKVCache(const KVCacheConfig &config);

                /**
                 * @brief Create a CPU KV cache
                 *
                 * Creates a ring-buffer CPU KV cache implementation.
                 *
                 * Supports both standard and sharded modes based on config:
                 * - If config.is_sharded() → sharded cache for tensor parallelism
                 * - Otherwise → standard cache
                 *
                 * @param config KVCacheConfig with all required parameters
                 *               - mpi_ctx must be non-null
                 * @return Unique pointer to ICPUKVCache implementation
                 * @throws std::runtime_error if mpi_ctx is null or parameters are invalid
                 */
                static std::unique_ptr<llaminar2::ICPUKVCache> createCPUKVCache(const KVCacheConfig &config);

#ifdef HAVE_CUDA
                /**
                 * @brief Create a CUDA ring buffer KV cache
                 *
                 * Creates a CUDARingKVCache with O(1) append and eviction.
                 *
                 * Limitations:
                 * - Only FP32, BF16, FP16 precisions supported (no Q8_1/Q16_1)
                 *
                 * Sharding Support:
                 * - Sharded CUDA KV cache IS supported for tensor parallelism
                 * - Set config.local_n_kv_heads and config.kv_head_start for sharding
                 *
                 * @param config KVCacheConfig with CUDA device
                 * @return Unique pointer to ICUDARingKVCache implementation
                 * @throws std::runtime_error if precision is not supported
                 */
                static std::unique_ptr<llaminar2::ICUDARingKVCache> createCUDAKVCache(const KVCacheConfig &config);
#endif

#ifdef HAVE_ROCM
                /**
                 * @brief Create a ROCm ring buffer KV cache
                 *
                 * Creates a ROCmRingKVCache with O(1) append and eviction.
                 *
                 * Limitations:
                 * - Only FP32, BF16, FP16 precisions supported (no Q8_1/Q16_1)
                 * - Sharded ROCm KV cache supported for tensor parallelism
                 *
                 * @param config KVCacheConfig with ROCm device
                 * @return Unique pointer to IKVCache implementation (underlying is ROCmRingKVCache)
                 * @throws std::runtime_error if precision is not supported
                 */
                static std::unique_ptr<llaminar2::IKVCache> createROCmKVCache(const KVCacheConfig &config);
#endif

                // ==========================================================================
                // Cached GEMM Kernel API - Pack Once, Use Many
                // ==========================================================================

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

#ifdef HAVE_CUDA
                /**
                 * @brief Ensure tensor has CUDA INT8 packed weights in its cache and return pointer
                 *
                 * Similar to ensurePackedWeightsInTensorCache but for CUDA backend.
                 * Converts any quantized tensor to INT8 + per-column scales for CUTLASS.
                 *
                 * The packed weights are stored in tensor->cuda_cache_ so they outlive
                 * any individual kernel and can be shared across multiple kernel instances.
                 *
                 * Thread-safe: can be called from multiple threads.
                 *
                 * @param tensor Quantized tensor
                 * @return Pointer to CUDA packed weights (stored in tensor's cuda_cache_)
                 * @throws std::runtime_error if packing fails
                 */
                static llaminar2::cuda::CUDAPackedWeights *
                ensureCUDAPackedWeightsInTensorCache(const llaminar2::TensorBase *tensor);
#endif

#ifdef HAVE_ROCM
                /**
                 * @brief Ensure ROCm INT8 packed weights are cached in tensor
                 *
                 * Similar to ensurePackedWeightsInTensorCache but for ROCm backend.
                 * Converts any quantized tensor to INT8 + per-column scales for CK.
                 *
                 * The packed weights are stored in tensor->rocm_cache_ so they outlive
                 * any individual kernel and can be shared across multiple kernel instances.
                 *
                 * Device upload happens lazily on first kernel use (ROCmPackedWeights::uploaded flag).
                 *
                 * Thread-safe: can be called from multiple threads.
                 *
                 * @param tensor Quantized tensor
                 * @return Pointer to ROCm packed weights (stored in tensor's rocm_cache_)
                 * @throws std::runtime_error if packing fails
                 */
                static llaminar2::rocm::ROCmPackedWeights *
                ensureROCmPackedWeightsInTensorCache(const llaminar2::TensorBase *tensor);
#endif

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
                 * @brief Reset input-dependent dynamic state on all cached kernels
                 *
                 * Called on session boundaries (e.g., clear_cache between inference
                 * requests). Unlike clearCache() which destroys kernel objects,
                 * this preserves all weight-dependent state and GPU handles while
                 * resetting only cached input data (token IDs, positions, etc.).
                 *
                 * In debug/integration builds, also asserts that no kernel retains
                 * stale dynamic state after the reset.
                 */
                static void resetAllDynamicState();

                /**
                 * @brief Get statistics about the kernel cache
                 * @return Pair of (cache_size, total_packed_bytes)
                 */
                static std::pair<size_t, size_t> cacheStats();

                // ==========================================================================
                // Phase A: Generic device kernel registry (scaffold, no behavior change)
                // ==========================================================================

                /**
                 * @brief Retrieve a device-scoped kernel entry from generic registry
                 */
                static std::shared_ptr<void> getDeviceKernelEntry(
                    const DeviceKernelKey &key);

                /**
                 * @brief Store/replace a device-scoped kernel entry in generic registry
                 */
                static void putDeviceKernelEntry(
                    const DeviceKernelKey &key,
                    std::shared_ptr<void> entry);

                /**
                 * @brief Clear only Phase A device kernel registry
                 */
                static void clearDeviceKernelRegistry();

                /**
                 * @brief Current number of entries in Phase A device kernel registry
                 */
                static size_t deviceKernelRegistrySize();

                // ==========================================================================
                // Phase C: GEMM engine + prepared-weights architecture
                // ==========================================================================

                /**
                 * @brief Strategy used to materialize weight-side GEMM state.
                 *
                 * The preparation kind controls how a weight tensor is transformed
                 * before execution (plain FP path vs backend-specific packed formats).
                 * This value participates in prepared-handle identity, so changing it
                 * intentionally creates a distinct prepared entry.
                 */
                enum class GemmPreparationKind
                {
                    // Let KernelFactory choose based on tensor dtype and target backend.
                    AUTO = 0,

                    // No explicit packed representation; use floating-point execution path.
                    FLOATING_POINT = 1,

                    // CPU packed quantized format (e.g., VNNI/OpenBLAS-friendly layout).
                    CPU_PACKED = 2,

                    // CUDA INT8 packed representation for GPU execution.
                    CUDA_INT8_PACKED = 3,

                    // ROCm INT8 packed representation for GPU execution.
                    ROCM_INT8_PACKED = 4,
                };

                /**
                 * @brief Prepared, backend-ready GEMM payload shared by handles.
                 *
                 * This object stores tensor-affine execution state that engines consume.
                 * Engines remain device-scoped and mostly tensor-independent; prepared
                 * payloads carry the tensor-specific readiness details.
                 */
                struct PreparedGemmWeights
                {
                    // Preparation mode used to materialize backend-ready weight state
                    // (e.g., CPU packed, CUDA INT8 packed, ROCm INT8 packed).
                    GemmPreparationKind kind{GemmPreparationKind::AUTO};

                    // Bound GEMM kernel instance for this prepared weight payload.
                    // Bound during prepared-handle creation via explicit-device
                    // create path (not via legacy tensor-keyed GEMM cache APIs),
                    // then reused by subsequent engine lookups.
                    llaminar2::ITensorGemm *kernel{nullptr};

                    // Ownership of the prepared-bound GEMM kernel instance.
                    // Kept here so kernel lifetime is tied to prepared payload
                    // rather than legacy tensor-keyed cache entries.
                    std::shared_ptr<llaminar2::ITensorGemm> owned_kernel;
                };

                /**
                 * @brief Stable identity object for prepared GEMM state.
                 *
                 * A PreparedGemmHandle represents one
                 * `(tensor, device, preparation kind, variant)` tuple and is used by
                 * direct GEMM execution and fused-kernel cache keying.
                 */
                struct PreparedGemmHandle
                {
                    // Source weight tensor this prepared entry belongs to.
                    const llaminar2::TensorBase *tensor{nullptr};

                    // Exact device target (type + ordinal) this prepared state is valid for.
                    llaminar2::DeviceId device_id{};

                    // Effective preparation mode used for this handle.
                    GemmPreparationKind kind{GemmPreparationKind::AUTO};

                    // Activation/weight variant discriminator used by engine registry keying.
                    int variant{0};

                    // Shared prepared payload (kernel binding and preparation metadata).
                    // Shared ownership allows cache entries and fused paths to reference a
                    // stable prepared identity.
                    std::shared_ptr<PreparedGemmWeights> prepared_weights;
                };

                class IGemmEngine
                {
                public:
                    virtual ~IGemmEngine() = default;

                    // Resolve the executable GEMM kernel for a prepared handle.
                    // Implementations must treat `prepared` as the authoritative source
                    // of tensor-affine execution state.
                    virtual llaminar2::ITensorGemm *resolveKernel(
                        const PreparedGemmHandle *prepared) const = 0;
                };

                /**
                 * @brief Resolve a GEMM kernel from a prepared handle
                 *
                 * Preferred API: engine resolution is based on prepared
                 * handle identity and prepared payload.
                 *
                 * Use this when prepared handles are already available (for example,
                 * when building fused cache keys). This avoids redundant preparation
                 * lookups and keeps kernel resolution aligned with prepared identity.
                 */
                static llaminar2::ITensorGemm *getOrCreateGemmEngine(
                    const PreparedGemmHandle *prepared);

                /**
                 * @brief Get or create a prepared GEMM-weights handle
                 *
                 * Resolves preparation kind, ensures packed state where applicable,
                 * and binds an executable GEMM kernel via explicit-device create path.
                 *
                 * Returned handles are stable identity objects for cache keying and
                 * may be shared by multiple callers (stages, fused adapters, preloaders)
                 * targeting the same tensor/device/preparation tuple.
                 */
                static const PreparedGemmHandle *getOrCreatePreparedGemmWeights(
                    const llaminar2::TensorBase *tensor,
                    llaminar2::DeviceId target_device,
                    GemmPreparationKind prep_kind = GemmPreparationKind::AUTO);

                /**
                 * @brief Clear prepared GEMM handle entries associated with a tensor
                 *
                 * Clears all prepared entries for the tensor across devices and prep kinds.
                 * Implementation is responsible for coordinated fused-cache eviction so
                 * no fused entry retains stale prepared-handle identity.
                 */
                static void clearPreparedGemmWeightsFor(const llaminar2::TensorBase *tensor);

                /**
                 * @brief Number of active GEMM engine registry entries
                 *
                 * Alias for the device-scoped GEMM engine registry size.
                 */
                static size_t gemmEngineRegistrySize();

                /**
                 * @brief Number of prepared GEMM registry entries
                 */
                static size_t preparedGemmRegistrySize();

                /**
                 * @brief Number of device-scoped GEMM engine entries
                 */
                static size_t deviceScopedGemmEngineRegistrySize();

            private:
                KernelFactory() = delete; // Static-only class

                struct PreparedGemmKey
                {
                    const llaminar2::TensorBase *tensor{nullptr};
                    llaminar2::DeviceId device_id;
                    int prep_kind{0};

                    bool operator==(const PreparedGemmKey &other) const
                    {
                        return tensor == other.tensor &&
                               device_id == other.device_id &&
                               prep_kind == other.prep_kind;
                    }
                };

                struct PreparedGemmKeyHash
                {
                    size_t operator()(const PreparedGemmKey &k) const
                    {
                        return std::hash<const void *>()(k.tensor) ^
                               (std::hash<int>()(static_cast<int>(k.device_id.type)) << 1) ^
                               (std::hash<int>()(k.device_id.ordinal) << 2) ^
                               (std::hash<int>()(k.prep_kind) << 3);
                    }
                };

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

                // Fused QKV GEMM cache - keyed by prepared-handle identity + device
                // This keeps fused cache identity aligned with prepared GEMM state
                // instead of raw tensor pointer tuples.
                struct FusedQKVCacheKey
                {
                    const PreparedGemmHandle *wq_handle{nullptr};
                    const PreparedGemmHandle *wk_handle{nullptr};
                    const PreparedGemmHandle *wv_handle{nullptr};
                    llaminar2::DeviceId device_id{};

                    bool operator==(const FusedQKVCacheKey &other) const
                    {
                        return wq_handle == other.wq_handle &&
                               wk_handle == other.wk_handle &&
                               wv_handle == other.wv_handle &&
                               device_id == other.device_id;
                    }
                };

                struct FusedQKVKeyHash
                {
                    size_t operator()(const FusedQKVCacheKey &k) const
                    {
                        return std::hash<const void *>()(k.wq_handle) ^
                               (std::hash<const void *>()(k.wk_handle) << 1) ^
                               (std::hash<const void *>()(k.wv_handle) << 2) ^
                               (std::hash<int>()(static_cast<int>(k.device_id.type)) << 3) ^
                               (std::hash<int>()(k.device_id.ordinal) << 8);
                    }
                };

                static std::unordered_map<FusedQKVCacheKey, std::unique_ptr<llaminar2::ITensorFusedQKVGemm>, FusedQKVKeyHash> fused_qkv_cache_;

                // Fused Gate/Up GEMM cache - keyed by prepared-handle identity + device
                struct FusedGateUpCacheKey
                {
                    const PreparedGemmHandle *w_gate_handle{nullptr};
                    const PreparedGemmHandle *w_up_handle{nullptr};
                    llaminar2::DeviceId device_id; // Full device ID including ordinal

                    bool operator==(const FusedGateUpCacheKey &other) const
                    {
                        return w_gate_handle == other.w_gate_handle &&
                               w_up_handle == other.w_up_handle &&
                               device_id == other.device_id;
                    }
                };

                struct FusedGateUpKeyHash
                {
                    size_t operator()(const FusedGateUpCacheKey &k) const
                    {
                        return std::hash<const void *>()(k.w_gate_handle) ^
                               (std::hash<const void *>()(k.w_up_handle) << 1) ^
                               (std::hash<int>()(static_cast<int>(k.device_id.type)) << 2) ^
                               (std::hash<int>()(k.device_id.ordinal) << 4);
                    }
                };

                static std::unordered_map<FusedGateUpCacheKey, std::unique_ptr<llaminar2::ITensorFusedGateUpGemm>, FusedGateUpKeyHash> fused_gate_up_cache_;

                struct RoPECacheKey
                {
                    llaminar2::DeviceId device_id;
                    int tensor_type{0};

                    bool operator==(const RoPECacheKey &other) const
                    {
                        return device_id == other.device_id && tensor_type == other.tensor_type;
                    }
                };

                struct RoPECacheKeyHash
                {
                    size_t operator()(const RoPECacheKey &k) const
                    {
                        return std::hash<int>()(static_cast<int>(k.device_id.type)) ^
                               (std::hash<int>()(k.device_id.ordinal) << 1) ^
                               (std::hash<int>()(k.tensor_type) << 2);
                    }
                };

                struct RMSNormCacheKey
                {
                    llaminar2::DeviceId device_id;
                    int tensor_type{0};

                    bool operator==(const RMSNormCacheKey &other) const
                    {
                        return device_id == other.device_id && tensor_type == other.tensor_type;
                    }
                };

                struct RMSNormCacheKeyHash
                {
                    size_t operator()(const RMSNormCacheKey &k) const
                    {
                        return std::hash<int>()(static_cast<int>(k.device_id.type)) ^
                               (std::hash<int>()(k.device_id.ordinal) << 1) ^
                               (std::hash<int>()(k.tensor_type) << 2);
                    }
                };

                struct SwiGLUCacheKey
                {
                    llaminar2::DeviceId device_id;
                    int tensor_type{0};

                    bool operator==(const SwiGLUCacheKey &other) const
                    {
                        return device_id == other.device_id && tensor_type == other.tensor_type;
                    }
                };

                struct SwiGLUCacheKeyHash
                {
                    size_t operator()(const SwiGLUCacheKey &k) const
                    {
                        return std::hash<int>()(static_cast<int>(k.device_id.type)) ^
                               (std::hash<int>()(k.device_id.ordinal) << 1) ^
                               (std::hash<int>()(k.tensor_type) << 2);
                    }
                };

                struct ResidualAddCacheKey
                {
                    llaminar2::DeviceId device_id;
                    int tensor_type{0};

                    bool operator==(const ResidualAddCacheKey &other) const
                    {
                        return device_id == other.device_id && tensor_type == other.tensor_type;
                    }
                };

                struct SoftmaxCacheKey
                {
                    llaminar2::DeviceId device_id;
                    int tensor_type{0};

                    bool operator==(const SoftmaxCacheKey &other) const
                    {
                        return device_id == other.device_id && tensor_type == other.tensor_type;
                    }
                };

                struct SoftmaxCacheKeyHash
                {
                    size_t operator()(const SoftmaxCacheKey &k) const
                    {
                        return std::hash<int>()(static_cast<int>(k.device_id.type)) ^
                               (std::hash<int>()(k.device_id.ordinal) << 1) ^
                               (std::hash<int>()(k.tensor_type) << 2);
                    }
                };

                struct ResidualAddCacheKeyHash
                {
                    size_t operator()(const ResidualAddCacheKey &k) const
                    {
                        return std::hash<int>()(static_cast<int>(k.device_id.type)) ^
                               (std::hash<int>()(k.device_id.ordinal) << 1) ^
                               (std::hash<int>()(k.tensor_type) << 2);
                    }
                };

                struct AttentionCacheKey
                {
                    llaminar2::DeviceId device_id;
                    int tensor_type{0};

                    bool operator==(const AttentionCacheKey &other) const
                    {
                        return device_id == other.device_id && tensor_type == other.tensor_type;
                    }
                };

                struct AttentionCacheKeyHash
                {
                    size_t operator()(const AttentionCacheKey &k) const
                    {
                        return std::hash<int>()(static_cast<int>(k.device_id.type)) ^
                               (std::hash<int>()(k.device_id.ordinal) << 1) ^
                               (std::hash<int>()(k.tensor_type) << 2);
                    }
                };

                struct EmbeddingCacheKey
                {
                    llaminar2::DeviceId device_id;
                    int tensor_type{0};

                    bool operator==(const EmbeddingCacheKey &other) const
                    {
                        return device_id == other.device_id && tensor_type == other.tensor_type;
                    }
                };

                struct EmbeddingCacheKeyHash
                {
                    size_t operator()(const EmbeddingCacheKey &k) const
                    {
                        return std::hash<int>()(static_cast<int>(k.device_id.type)) ^
                               (std::hash<int>()(k.device_id.ordinal) << 1) ^
                               (std::hash<int>()(k.tensor_type) << 2);
                    }
                };

                static std::unordered_map<RoPECacheKey, std::unique_ptr<llaminar2::ITensorRoPE>, RoPECacheKeyHash> rope_cache_;
                static std::unordered_map<RMSNormCacheKey, std::unique_ptr<llaminar2::ITensorRMSNorm>, RMSNormCacheKeyHash> rmsnorm_cache_;
                static std::unordered_map<SwiGLUCacheKey, std::unique_ptr<llaminar2::ITensorSwiGLU>, SwiGLUCacheKeyHash> swiglu_cache_;
                static std::unordered_map<SoftmaxCacheKey, std::unique_ptr<llaminar2::ITensorSoftmax>, SoftmaxCacheKeyHash> softmax_cache_;
                static std::unordered_map<ResidualAddCacheKey, std::unique_ptr<llaminar2::ITensorResidualAdd>, ResidualAddCacheKeyHash> residual_add_cache_;
                static std::unordered_map<AttentionCacheKey, std::unique_ptr<llaminar2::ITensorAttention>, AttentionCacheKeyHash> attention_cache_;
                static std::unordered_map<EmbeddingCacheKey, std::unique_ptr<llaminar2::ITensorEmbedding>, EmbeddingCacheKeyHash> embedding_cache_;

                // Generic device-scoped non-GEMM registry
                static std::unordered_map<DeviceKernelKey, std::shared_ptr<void>, DeviceKernelKeyHash> device_kernel_registry_;

                // Prepared-weight and device-scoped GEMM engine registries
                static std::unordered_map<PreparedGemmKey, std::shared_ptr<PreparedGemmHandle>, PreparedGemmKeyHash> prepared_gemm_registry_;
                static std::unordered_map<DeviceKernelKey, std::shared_ptr<IGemmEngine>, DeviceKernelKeyHash> device_gemm_engine_registry_;

                // NOTE: Universal device kernel caching (hipBLAS, cuBLAS handles, etc.)
                // has moved to DeviceKernelCache. See kernels/DeviceKernelCache.h
            };

        } // namespace kernels
    } // namespace v2
} // namespace llaminar
