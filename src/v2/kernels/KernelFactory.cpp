/**
 * @file KernelFactory.cpp
 * @brief Implementation of centralized kernel dispatch factory
 * @author David Sanftenberg
 */

#include "KernelFactory.h"
#include "../backends/BackendManager.h"
#include "../planning/KVCacheMemoryEstimator.h"
#include "cpu/native_vnni/CPUNativeVNNIGemmKernel.h"
#include "cpu/native_vnni/CPUPackedWeights.h"
#include "PackedWeightsSerialization.h"
#include "cpu/gemm/FloatingPointGemmKernel.h"
#include "../tensors/TensorSlice.h"
#include "cpu/ops/CPURoPEKernelT.h"
#include "cpu/ops/CPUSwiGLUKernelT.h"
#include "cpu/ops/CPUSoftmaxKernelT.h"
#include "cpu/ops/CPURMSNormKernelT.h"
#include "cpu/ops/CPUResidualAddKernelT.h"
#include "cpu/ops/CPUEmbeddingKernelT.h"
#include "cpu/attention/CPUFlashAttentionKernelT.h"
#include "cpu/gdn/CPUShortConvolution.h"
#include "cpu/gdn/CPUGatedDeltaNet.h"
#include "cpu/moe/CPUMoEKernel.h"
#include "../utils/Assertions.h"
#include <unordered_set>

// KVCache includes
#include "cpu/CPURingKVCache.h"
#include "cpu/CPUHybridRingKVCache.h"
#include "HybridKVCacheConfig.h"
#include "IHybridKVCache.h"
#ifdef HAVE_CUDA
#include "cuda/kvcache/CUDARingKVCache.h"
#include "cuda/kvcache/CUDARingKVCacheTQ.h"
#include "cuda/kvcache/CUDAHybridRingKVCache.h"
#endif

#include "../tensors/Tensors.h"
#include "../tensors/TensorKernels.h"
#include "../execution/local_execution/device/DeviceContext.h"
#include "../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../interfaces/IWorkspaceConsumer.h"
#include "../backends/ComputeBackend.h"
#include "../utils/Logger.h"
#include "../utils/DebugEnv.h"
#include "common/EmbedQ8Repack.h"
// CUDA kernel classes
#ifdef HAVE_CUDA
#include "cuda/gemm/CUDAQuantisedGemmKernel.h"        // Quantized tensors via CUTLASS INT8
#include "cuda/gemm/CUDAFloatingPointGemmKernel.h"    // FP32/FP16/BF16 via cuBLAS
#include "cuda/ops/CUDARMSNormKernelT.h"              // RMSNorm FP32/FP16/BF16
#include "cuda/ops/CUDARoPEKernelT.h"                 // RoPE FP32/FP16/BF16
#include "cuda/ops/CUDASwiGLUKernelT.h"               // SwiGLU FP32/FP16/BF16
#include "cuda/ops/CUDAResidualAddKernelT.h"          // ResidualAdd FP32/FP16/BF16
#include "cuda/ops/CUDAEmbeddingKernelT.h"            // Embedding FP32
#include "cuda/attention/CUDAFlashAttentionKernelT.h" // Flash Attention
#include "cuda/gdn/CUDAGatedDeltaNet.h"               // GDN recurrence
#include "cuda/gdn/CUDAShortConvolution.h"            // GDN short conv1d
#include "cuda/moe/CUDAMoEKernel.h"                   // MoE routing/gather/scatter/gate/swiglu

extern "C" void cudaNativeVNNIGemvTuned_clearStaticState();
#endif

// ROCm kernel classes
#ifdef HAVE_ROCM
#include "rocm/gemm/ROCmFloatingPointGemmKernel.h"     // FP32/FP16/BF16 via hipBLAS
#include "rocm/gemm/ROCmQuantisedGemmKernel.h"         // Quantized tensors via CK INT8/FP16
#include "rocm/kvcache/ROCmRingKVCacheFactory.h"       // ROCm Ring Buffer KV Cache factory
#include "rocm/kvcache/ROCmRingKVCacheTQFactory.h"     // ROCm TurboQuant KV Cache factory
#include "rocm/kvcache/ROCmHybridRingKVCacheFactory.h" // ROCm Hybrid KV Cache factory
#include "rocm/ops/ROCmEmbeddingKernelT.h"             // Embedding FP32/BF16/FP16/Q8_1
#include "rocm/ops/ROCmRMSNormKernelT.h"               // RMSNorm FP32/BF16/FP16
#include "rocm/ops/ROCmRoPEKernelT.h"                  // RoPE FP32
#include "rocm/ops/ROCmSwiGLUKernelT.h"                // SwiGLU FP32
#include "rocm/ops/ROCmResidualAddKernelT.h"           // ResidualAdd FP32
#include "rocm/attention/ROCmFlashAttentionKernelT.h"  // Flash Attention
#include "rocm/gdn/ROCmGatedDeltaNet.h"                // GDN recurrence
#include "rocm/gdn/ROCmShortConvolution.h"             // GDN short conv1d
#include "rocm/moe/ROCmMoEKernel.h"                    // MoE routing/gather/scatter/gate/swiglu
#endif

namespace llaminar
{
    namespace v2
    {
        namespace kernels
        {

            static const llaminar2::NativeVnniFormatInfo *vnniFormatInfoIfPackable(
                const llaminar2::TensorBase *tensor)
            {
                const auto *unpackable = dynamic_cast<const llaminar2::IINT8Unpackable *>(tensor);
                return unpackable ? unpackable->vnniFormatInfo() : nullptr;
            }

            static bool isVnniPackableTensor(const llaminar2::TensorBase *tensor)
            {
                return vnniFormatInfoIfPackable(tensor) != nullptr;
            }

            // ==========================================================================
            // Device Type Resolution
            // ==========================================================================

            DeviceType KernelFactory::getDeviceType(llaminar2::DeviceId device_id)
            {
                // Direct type extraction from DeviceId - no DeviceManager lookup needed
                // DeviceId already carries the type information
                return device_id.type;
            }

            DeviceType KernelFactory::getDeviceType(int device_idx)
            {
                // Legacy overload: convert to DeviceId then delegate
                // This maintains backward compatibility with existing callers
                if (device_idx < 0)
                {
                    return DeviceType::CPU;
                }

                auto &dm = llaminar2::DeviceManager::instance();
                const auto &devices = dm.devices();

                if (static_cast<size_t>(device_idx) >= devices.size())
                {
                    LOG_ERROR("[KernelFactory] Invalid device_idx: " << device_idx
                                                                     << " (only " << devices.size() << " devices)");
                    throw std::runtime_error("KernelFactory::getDeviceType: invalid device index");
                }

                const auto &device = devices[device_idx];

                switch (device.type)
                {
                case llaminar2::ComputeBackendType::CPU:
                    return DeviceType::CPU;
                case llaminar2::ComputeBackendType::GPU_CUDA:
                    return DeviceType::CUDA;
                case llaminar2::ComputeBackendType::GPU_ROCM:
                    return DeviceType::ROCm;
                case llaminar2::ComputeBackendType::GPU_VULKAN:
                    return DeviceType::Vulkan;
                case llaminar2::ComputeBackendType::GPU_METAL:
                    return DeviceType::Metal;
                default:
                    LOG_ERROR("[KernelFactory] Unknown backend type: " << static_cast<int>(device.type));
                    throw std::runtime_error("Unknown backend type");
                }
            }

            // ==========================================================================
            // Helper: Dispatch based on DeviceType
            // ==========================================================================

            namespace
            {
                // Helper to throw for unsupported GPU backends (GEMM kernels)
                [[noreturn]] void throwUnsupportedBackend(DeviceType dev_type, const char *tensor_type)
                {
                    LOG_ERROR("[KernelFactory] " << tensor_type << " GEMM not supported on "
                                                 << to_string(dev_type));
                    throw std::runtime_error(std::string(tensor_type) + " GEMM not supported on " +
                                             to_string(dev_type));
                }

                // Generic helper to throw for unsupported kernel/backend combinations
                [[noreturn]] void throwUnsupportedKernel(DeviceType dev_type, const char *kernel_type, const char *tensor_type)
                {
                    LOG_ERROR("[KernelFactory] " << kernel_type << " for " << tensor_type
                                                 << " not supported on " << to_string(dev_type)
                                                 << " - no silent fallback to CPU");
                    throw std::runtime_error(std::string(kernel_type) + " for " + tensor_type +
                                             " not supported on " + to_string(dev_type));
                }

#ifdef HAVE_CUDA
                // Thread-local storage for target CUDA device ordinal
                // Used when getOrCreateGemm is called with DeviceId to pass the ordinal
                // to getCUDADeviceIdForKernel (since createGemm only receives DeviceType)
                thread_local std::optional<int> tl_target_cuda_ordinal;

                /**
                 * @brief Get CUDA device ID for kernel creation
                 *
                 * Resolution order:
                 * 1. Thread-local target ordinal (set via CUDAOrdinalGuard)
                 * 2. Tensor's current device (if already on CUDA)
                 *
                 * @param tensor The tensor to check (usually a weight tensor)
                 * @return CUDA device ID (backend-specific ordinal)
                 * @throws std::runtime_error if no explicit device can be determined
                 */
                int getCUDADeviceIdForKernel(const llaminar2::TensorBase *tensor)
                {
                    // First check if a target CUDA ordinal was set via DeviceId overload
                    if (tl_target_cuda_ordinal.has_value())
                    {
                        LOG_DEBUG("[KernelFactory] Using thread-local CUDA ordinal: " << tl_target_cuda_ordinal.value());
                        return tl_target_cuda_ordinal.value();
                    }

                    // Check if tensor is currently on a CUDA device
                    auto current_dev = tensor->current_device();
                    if (current_dev.has_value() && current_dev->is_cuda())
                    {
                        return current_dev->cuda_ordinal();
                    }

                    // No implicit fallback - require explicit device specification
                    LOG_ERROR("[KernelFactory] Cannot determine CUDA device for kernel creation. "
                              "Tensor is not on a CUDA device and no target device was specified. "
                              "Use prepareGemmHandleLocal(tensor, DeviceId) and getOrCreateGemmEngine(prepared.get()) to specify the target device explicitly.");
                    throw std::runtime_error(
                        "KernelFactory: CUDA device must be specified explicitly - tensor is not on CUDA "
                        "and no target device provided via CUDAOrdinalGuard or DeviceId");
                }
#endif

#ifdef HAVE_ROCM
                // Thread-local storage for target ROCm device ordinal
                // Used when getOrCreateGemm is called with DeviceId to pass the ordinal
                // to getROCmDeviceIdForKernel (since createGemm only receives DeviceType)
                thread_local std::optional<int> tl_target_rocm_ordinal;

                /**
                 * @brief Get ROCm device ID for kernel creation
                 *
                 * Resolution order:
                 * 1. Thread-local target ordinal (set via ROCmOrdinalGuard)
                 * 2. Tensor's current device (if already on ROCm)
                 *
                 * @param tensor The tensor to check (usually a weight tensor)
                 * @return ROCm device ID (backend-specific ordinal)
                 * @throws std::runtime_error if no explicit device can be determined
                 */
                int getROCmDeviceIdForKernel(const llaminar2::TensorBase *tensor)
                {
                    // First check if a target ROCm ordinal was set via DeviceId overload
                    if (tl_target_rocm_ordinal.has_value())
                    {
                        LOG_DEBUG("[KernelFactory] Using thread-local ROCm ordinal: " << tl_target_rocm_ordinal.value());
                        return tl_target_rocm_ordinal.value();
                    }

                    // Check if tensor is currently on a ROCm device
                    auto current_dev = tensor->current_device();
                    if (current_dev.has_value() && current_dev->is_rocm())
                    {
                        return current_dev->rocm_ordinal();
                    }

                    // No implicit fallback - require explicit device specification
                    LOG_ERROR("[KernelFactory] Cannot determine ROCm device for kernel creation. "
                              "Tensor is not on a ROCm device and no target device was specified. "
                              "Use prepareGemmHandleLocal(tensor, DeviceId) and getOrCreateGemmEngine(prepared.get()) to specify the target device explicitly.");
                    throw std::runtime_error(
                        "KernelFactory: ROCm device must be specified explicitly - tensor is not on ROCm "
                        "and no target device provided via ROCmOrdinalGuard or DeviceId");
                }
#endif

                // =========================================================================
                // Adapter: CPURoPEKernelT<FP32> -> ITensorRoPE
                // Only FP32 is supported through this interface (other precisions use
                // their native typed kernels directly).
                // =========================================================================
                class RoPEKernelAdapter : public llaminar2::ITensorRoPE
                {
                public:
                    bool supports_device(int device_idx) const override
                    {
                        return kernel_.supports_device(device_idx);
                    }

                    // Tensor-based interface (used by RoPEOpTyped)
                    bool apply_tensor(
                        llaminar2::TensorBase *Q,
                        llaminar2::TensorBase *K,
                        const int *position_ids,
                        int seq_len,
                        int n_heads,
                        int n_kv_heads,
                        int head_dim,
                        float rope_theta,
                        const llaminar2::IMPIContext *mpi_ctx,
                        int device_idx,
                        int pos_offset = 0,
                        int rotary_dim = 0) override
                    {
                        (void)mpi_ctx;    // Not used in typed kernel
                        (void)pos_offset; // CPU kernel doesn't need this optimization

                        // Validate tensors are FP32
                        if (!Q)
                        {
                            LOG_ERROR("[RoPEKernelAdapter] Q tensor is null");
                            return false;
                        }
                        if (Q->native_type() != llaminar2::TensorType::FP32)
                        {
                            LOG_ERROR("[RoPEKernelAdapter] Q tensor must be FP32, got " << static_cast<int>(Q->native_type()));
                            return false;
                        }
                        if (K && K->native_type() != llaminar2::TensorType::FP32)
                        {
                            LOG_ERROR("[RoPEKernelAdapter] K tensor must be FP32, got " << static_cast<int>(K->native_type()));
                            return false;
                        }

                        // Get raw pointers from tensors
                        float *Q_data = Q->mutable_data();
                        float *K_data = K ? K->mutable_data() : nullptr;

                        return kernel_.apply_typed(Q_data, K_data, position_ids,
                                                   seq_len, n_heads, n_kv_heads, head_dim,
                                                   rope_theta, device_idx, rotary_dim);
                    }

                private:
                    llaminar2::CPURoPEKernelT<llaminar2::ActivationPrecision::FP32> kernel_;
                };

                // =========================================================================
                // Adapter: CPURoPEKernelT<Q8_1> -> ITensorRoPE
                // =========================================================================
                class Q8_1RoPEKernelAdapter : public llaminar2::ITensorRoPE
                {
                public:
                    bool supports_device(int device_idx) const override
                    {
                        return kernel_.supports_device(device_idx);
                    }

                    // Tensor-based interface (used by RoPEOpTyped)
                    bool apply_tensor(
                        llaminar2::TensorBase *Q,
                        llaminar2::TensorBase *K,
                        const int *position_ids,
                        int seq_len,
                        int n_heads,
                        int n_kv_heads,
                        int head_dim,
                        float rope_theta,
                        const llaminar2::IMPIContext *mpi_ctx,
                        int device_idx,
                        int pos_offset = 0,
                        int rotary_dim = 0) override
                    {
                        (void)mpi_ctx;    // Not used in typed kernel
                        (void)pos_offset; // CPU kernel doesn't need this optimization

                        // Validate tensors are Q8_1
                        if (!Q)
                        {
                            LOG_ERROR("[Q8_1RoPEKernelAdapter] Q tensor is null");
                            return false;
                        }
                        if (Q->native_type() != llaminar2::TensorType::Q8_1)
                        {
                            LOG_ERROR("[Q8_1RoPEKernelAdapter] Q tensor must be Q8_1, got " << static_cast<int>(Q->native_type()));
                            return false;
                        }
                        if (K && K->native_type() != llaminar2::TensorType::Q8_1)
                        {
                            LOG_ERROR("[Q8_1RoPEKernelAdapter] K tensor must be Q8_1, got " << static_cast<int>(K->native_type()));
                            return false;
                        }

                        // Cast to Q8_1Tensor to access Q8_1 blocks
                        auto *Q_q8 = dynamic_cast<llaminar2::Q8_1Tensor *>(Q);
                        auto *K_q8 = K ? dynamic_cast<llaminar2::Q8_1Tensor *>(K) : nullptr;

                        if (!Q_q8)
                        {
                            LOG_ERROR("[Q8_1RoPEKernelAdapter] Failed to cast Q to Q8_1Tensor");
                            return false;
                        }
                        if (K && !K_q8)
                        {
                            LOG_ERROR("[Q8_1RoPEKernelAdapter] Failed to cast K to Q8_1Tensor");
                            return false;
                        }

                        // Get Q8_1 block pointers
                        llaminar2::Q8_1Block *Q_blocks = Q_q8->mutable_typed_data();
                        llaminar2::Q8_1Block *K_blocks = K_q8 ? K_q8->mutable_typed_data() : nullptr;

                        return kernel_.apply_typed(Q_blocks, K_blocks, position_ids,
                                                   seq_len, n_heads, n_kv_heads, head_dim,
                                                   rope_theta, device_idx, rotary_dim);
                    }

                private:
                    llaminar2::CPURoPEKernelT<llaminar2::ActivationPrecision::Q8_1> kernel_;
                };

                // =========================================================================
                // Adapter: CPUSwiGLUKernelT<FP32> -> ITensorSwiGLU
                // =========================================================================
                class SwiGLUKernelAdapter : public llaminar2::ITensorSwiGLU
                {
                public:
                    bool supports_device(int device_idx) const override
                    {
                        return kernel_.supports_device(device_idx);
                    }

                    // Original raw pointer interface (legacy)
                    bool apply(
                        const float *gate, const float *up, float *output,
                        int rows, int cols,
                        bool add_residual,
                        const llaminar2::IMPIContext *mpi_ctx,
                        int device_idx)
                    {
                        (void)add_residual; // Not used in typed kernel
                        (void)mpi_ctx;      // Not used in typed kernel
                        return kernel_.apply_typed(gate, up, output, rows * cols, device_idx);
                    }

                    // Tensor-based interface (used by SwiGLUStage)
                    bool apply_tensor(
                        const llaminar2::TensorBase *gate,
                        const llaminar2::TensorBase *up,
                        llaminar2::TensorBase *output,
                        int rows, int cols,
                        bool add_residual,
                        const llaminar2::IMPIContext *mpi_ctx,
                        int device_idx) override
                    {
                        (void)add_residual; // Not used in typed kernel
                        (void)mpi_ctx;      // Not used in typed kernel

                        // Validate tensors are FP32
                        if (!gate || gate->native_type() != llaminar2::TensorType::FP32)
                        {
                            LOG_ERROR("[SwiGLUKernelAdapter] gate tensor must be FP32");
                            return false;
                        }
                        if (!up || up->native_type() != llaminar2::TensorType::FP32)
                        {
                            LOG_ERROR("[SwiGLUKernelAdapter] up tensor must be FP32");
                            return false;
                        }
                        if (!output || output->native_type() != llaminar2::TensorType::FP32)
                        {
                            LOG_ERROR("[SwiGLUKernelAdapter] output tensor must be FP32");
                            return false;
                        }

                        // Get raw pointers from tensors
                        const float *gate_data = gate->data();
                        const float *up_data = up->data();
                        float *output_data = output->mutable_data();

                        return kernel_.apply_typed(gate_data, up_data, output_data, rows * cols, device_idx);
                    }

                private:
                    llaminar2::CPUSwiGLUKernelT<llaminar2::ActivationPrecision::FP32> kernel_;
                };

                // =========================================================================
                // Adapter: CPUSwiGLUKernelT<Q8_1> -> ITensorSwiGLU
                // =========================================================================
                class Q8_1SwiGLUKernelAdapter : public llaminar2::ITensorSwiGLU
                {
                public:
                    bool supports_device(int device_idx) const override
                    {
                        return kernel_.supports_device(device_idx);
                    }

                    // Original raw pointer interface (legacy) - not supported for Q8_1
                    bool apply(
                        const float *gate, const float *up, float *output,
                        int rows, int cols,
                        bool add_residual,
                        const llaminar2::IMPIContext *mpi_ctx,
                        int device_idx)
                    {
                        (void)gate;
                        (void)up;
                        (void)output;
                        (void)rows;
                        (void)cols;
                        (void)add_residual;
                        (void)mpi_ctx;
                        (void)device_idx;
                        LOG_ERROR("[Q8_1SwiGLUKernelAdapter] Raw float* interface not supported for Q8_1");
                        return false;
                    }

                    // Tensor-based interface (used by SwiGLUOpTyped)
                    bool apply_tensor(
                        const llaminar2::TensorBase *gate,
                        const llaminar2::TensorBase *up,
                        llaminar2::TensorBase *output,
                        int rows, int cols,
                        bool add_residual,
                        const llaminar2::IMPIContext *mpi_ctx,
                        int device_idx) override
                    {
                        (void)add_residual; // Not used in typed kernel
                        (void)mpi_ctx;      // Not used in typed kernel

                        // Validate tensors are Q8_1
                        if (!gate || gate->native_type() != llaminar2::TensorType::Q8_1)
                        {
                            LOG_ERROR("[Q8_1SwiGLUKernelAdapter] gate tensor must be Q8_1, got "
                                      << (gate ? static_cast<int>(gate->native_type()) : -1));
                            return false;
                        }
                        if (!up || up->native_type() != llaminar2::TensorType::Q8_1)
                        {
                            LOG_ERROR("[Q8_1SwiGLUKernelAdapter] up tensor must be Q8_1, got "
                                      << (up ? static_cast<int>(up->native_type()) : -1));
                            return false;
                        }
                        if (!output || output->native_type() != llaminar2::TensorType::Q8_1)
                        {
                            LOG_ERROR("[Q8_1SwiGLUKernelAdapter] output tensor must be Q8_1, got "
                                      << (output ? static_cast<int>(output->native_type()) : -1));
                            return false;
                        }

                        // Cast to Q8_1Tensor to access Q8_1 blocks
                        const auto *gate_q8 = dynamic_cast<const llaminar2::Q8_1Tensor *>(gate);
                        const auto *up_q8 = dynamic_cast<const llaminar2::Q8_1Tensor *>(up);
                        auto *output_q8 = dynamic_cast<llaminar2::Q8_1Tensor *>(output);

                        if (!gate_q8 || !up_q8 || !output_q8)
                        {
                            LOG_ERROR("[Q8_1SwiGLUKernelAdapter] Failed to cast tensors to Q8_1Tensor");
                            return false;
                        }

                        // Get Q8_1 block pointers
                        const llaminar2::Q8_1Block *gate_blocks = gate_q8->typed_data();
                        const llaminar2::Q8_1Block *up_blocks = up_q8->typed_data();
                        llaminar2::Q8_1Block *output_blocks = output_q8->mutable_typed_data();

                        return kernel_.apply_typed(gate_blocks, up_blocks, output_blocks, rows * cols, device_idx);
                    }

                private:
                    llaminar2::CPUSwiGLUKernelT<llaminar2::ActivationPrecision::Q8_1> kernel_;
                };

                // NOTE: SoftmaxKernelAdapter removed - CPUSoftmaxKernelT now directly
                // implements ITensorSoftmax interface

            } // namespace

            // ==========================================================================
            // CUDAOrdinalGuard implementation (must be outside anonymous namespace)
            // ==========================================================================
#ifdef HAVE_CUDA
            KernelFactory::CUDAOrdinalGuard::CUDAOrdinalGuard(int ordinal)
            {
                tl_target_cuda_ordinal = ordinal;
                LOG_DEBUG("[KernelFactory] CUDAOrdinalGuard: set thread-local CUDA ordinal to " << ordinal);
            }

            KernelFactory::CUDAOrdinalGuard::~CUDAOrdinalGuard()
            {
                LOG_DEBUG("[KernelFactory] CUDAOrdinalGuard: cleared thread-local CUDA ordinal");
                tl_target_cuda_ordinal = std::nullopt;
            }
#else
            // Stub implementations when CUDA is not available
            KernelFactory::CUDAOrdinalGuard::CUDAOrdinalGuard(int) {}
            KernelFactory::CUDAOrdinalGuard::~CUDAOrdinalGuard() {}
#endif

            // ==========================================================================
            // ROCmOrdinalGuard implementation (must be outside anonymous namespace)
            // ==========================================================================
#ifdef HAVE_ROCM
            KernelFactory::ROCmOrdinalGuard::ROCmOrdinalGuard(int ordinal)
            {
                tl_target_rocm_ordinal = ordinal;
                LOG_DEBUG("[KernelFactory] ROCmOrdinalGuard: set thread-local ROCm ordinal to " << ordinal);
            }

            KernelFactory::ROCmOrdinalGuard::~ROCmOrdinalGuard()
            {
                LOG_DEBUG("[KernelFactory] ROCmOrdinalGuard: cleared thread-local ROCm ordinal");
                tl_target_rocm_ordinal = std::nullopt;
            }
#else
            // Stub implementations when ROCm is not available
            KernelFactory::ROCmOrdinalGuard::ROCmOrdinalGuard(int) {}
            KernelFactory::ROCmOrdinalGuard::~ROCmOrdinalGuard() {}
#endif

            // ==========================================================================
            // TensorBase Dynamic Dispatch GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::TensorBase *tensor, DeviceType dev_type, int device_ordinal)
            {
                // Two-way dispatch based on interface:
                // 1. IINT8Unpackable → CPUNativeVNNIGemmKernel (quantized weights)
                // 2. Otherwise → FloatingPointGemmKernel (FP32/FP16/BF16)

                const bool quantized = isVnniPackableTensor(tensor);

                if (quantized)
                {
                    // Quantized tensor - use INT8 GEMM kernel
                    //
                    // For TensorSlice wrapping a pre-sliced inner tensor (INPUT_PARALLEL
                    // column-sliced weights), unwrap to the inner tensor so the VNNI kernel
                    // can use the native Q8_0 GEMV bypass (avoids the packed VNNI path
                    // which has higher overhead for M=1 decode). The inner tensor has the
                    // correct sliced shape and data.
                    const llaminar2::TensorBase *kernel_tensor = tensor;
                    if (auto *slice = dynamic_cast<const llaminar2::TensorSlice *>(tensor))
                    {
                        if (slice->metadata().inner_is_presliced)
                        {
                            kernel_tensor = slice->inner();
                            LOG_DEBUG("[KernelFactory] Unwrapped TensorSlice to inner "
                                      << llaminar2::tensorTypeName(kernel_tensor->native_type())
                                      << " [" << kernel_tensor->shape()[0] << "x" << kernel_tensor->shape()[1] << "]");
                        }
                    }

                    switch (dev_type)
                    {
                    case DeviceType::CPU:
                    {
                        // CPUNativeVNNIGemmKernel handles both M=1 and M>1 paths
                        return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(kernel_tensor);
                    }
#ifdef HAVE_CUDA
                    case DeviceType::CUDA:
                    {
                        throw std::runtime_error(
                            "[KernelFactory] Legacy createGemm() GPU path removed. "
                            "Use GPU pipeline (DeviceLoadPipeline/WeightVRAMPool) for CUDA weight preparation.");
                    }
#endif
#ifdef HAVE_ROCM
                    case DeviceType::ROCm:
                    {
                        throw std::runtime_error(
                            "[KernelFactory] Legacy createGemm() GPU path removed. "
                            "Use GPU pipeline (DeviceLoadPipeline/WeightVRAMPool) for ROCm weight preparation.");
                    }
#endif
                    default:
                        throw std::runtime_error(
                            "Quantized GEMM not supported on device " + std::to_string(static_cast<int>(dev_type)));
                    }
                }
                else
                {
                    // Floating point tensor - use BLAS-based kernel
                    switch (dev_type)
                    {
                    case DeviceType::CPU:
                        return std::make_unique<llaminar2::gemm::FloatingPointGemmKernel>(tensor);
#ifdef HAVE_CUDA
                    case DeviceType::CUDA:
                    {
                        int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                        auto precision = llaminar2::cuda::CUDAFloatingPointGemmKernel::Precision::FP32;
                        if (tensor->native_type() == llaminar2::TensorType::FP16)
                            precision = llaminar2::cuda::CUDAFloatingPointGemmKernel::Precision::FP16;
                        else if (tensor->native_type() == llaminar2::TensorType::BF16)
                            precision = llaminar2::cuda::CUDAFloatingPointGemmKernel::Precision::BF16;
                        return std::make_unique<llaminar2::cuda::CUDAFloatingPointGemmKernel>(
                            tensor, cuda_device_id, precision);
                    }
#endif
#ifdef HAVE_ROCM
                    case DeviceType::ROCm:
                    {
                        int rocm_device_id = getROCmDeviceIdForKernel(tensor);
                        auto precision = llaminar2::rocm::ROCmFloatingPointGemmKernel::Precision::FP32;
                        if (tensor->native_type() == llaminar2::TensorType::FP16)
                            precision = llaminar2::rocm::ROCmFloatingPointGemmKernel::Precision::FP16;
                        else if (tensor->native_type() == llaminar2::TensorType::BF16)
                            precision = llaminar2::rocm::ROCmFloatingPointGemmKernel::Precision::BF16;
                        return std::make_unique<llaminar2::rocm::ROCmFloatingPointGemmKernel>(
                            tensor, rocm_device_id, precision);
                    }
#endif
                    default:
                        throw std::runtime_error(
                            "Floating-point GEMM not supported on device " + std::to_string(static_cast<int>(dev_type)));
                    }
                }
            }

            // ==========================================================================
            // IQ4_NL Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::IQ4_NLTensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                case DeviceType::ROCm:
#ifdef HAVE_ROCM
                    // TODO: return createRocmGemm(tensor);
#endif
                    throwUnsupportedBackend(dev_type, "IQ4_NL");

                case DeviceType::Vulkan:
                case DeviceType::Metal:
                default:
                    throwUnsupportedBackend(dev_type, "IQ4_NL");
                }
            }

            llaminar2::ITensorGemm *KernelFactory::createGemmRaw(
                const llaminar2::IQ4_NLTensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return new llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    throwUnsupportedBackend(dev_type, "IQ4_NL");
                }
            }

            // ==========================================================================
            // Q4_0 Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::Q4_0Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for ROCm weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "Q4_0");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // Q4_1 Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::Q4_1Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "Q4_1");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // Q5_0 Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::Q5_0Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "Q5_0");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // Q5_1 Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::Q5_1Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "Q5_1");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // Q6_K Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::Q6_KTensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "Q6_K");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // Q8_0 Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::Q8_0Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "Q8_0");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // Q8_1 Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "Q8_1");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // Q2_K Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::Q2_KTensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "Q2_K");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // Q3_K Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::Q3_KTensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "Q3_K");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // Q4_K Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::Q4_KTensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "Q4_K");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // Q5_K Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::Q5_KTensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "Q5_K");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // Q8_K Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::Q8_KTensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "Q8_K");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // IQ1_M Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::IQ1_MTensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "IQ1_M");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // IQ1_S Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::IQ1_STensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "IQ1_S");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // IQ2_S Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::IQ2_STensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "IQ2_S");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // IQ2_XS Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::IQ2_XSTensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "IQ2_XS");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // IQ2_XXS Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::IQ2_XXSTensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "IQ2_XXS");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // IQ3_S Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::IQ3_STensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "IQ3_S");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // IQ3_XXS Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::IQ3_XXSTensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "IQ3_XXS");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // IQ4_XS Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::IQ4_XSTensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    throw std::runtime_error(
                        "[KernelFactory] Legacy createGemm() GPU path removed. "
                        "Use GPU pipeline for CUDA weight preparation.");
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "IQ4_XS");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // FP32 Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::FP32Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::gemm::FloatingPointGemmKernel>(tensor);

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] FP32 GEMM: Using CUDAFloatingPointGemmKernel");
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAFloatingPointGemmKernel>(
                        tensor, cuda_device_id, llaminar2::cuda::CUDAFloatingPointGemmKernel::Precision::FP32);
                }
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                {
                    LOG_DEBUG("[KernelFactory] FP32 GEMM: Using ROCmFloatingPointGemmKernel");
                    int rocm_device_id = getROCmDeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::rocm::ROCmFloatingPointGemmKernel>(
                        tensor, rocm_device_id, llaminar2::rocm::ROCmFloatingPointGemmKernel::Precision::FP32);
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "FP32");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // FP16 Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::FP16Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::gemm::FloatingPointGemmKernel>(tensor);

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] FP16 GEMM: Using CUDAFloatingPointGemmKernel");
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAFloatingPointGemmKernel>(
                        tensor, cuda_device_id, llaminar2::cuda::CUDAFloatingPointGemmKernel::Precision::FP16);
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "FP16");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // BF16 Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::BF16Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::gemm::FloatingPointGemmKernel>(tensor);

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] BF16 GEMM: Using CUDAFloatingPointGemmKernel");
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAFloatingPointGemmKernel>(
                        tensor, cuda_device_id, llaminar2::cuda::CUDAFloatingPointGemmKernel::Precision::BF16);
                }
#endif

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "BF16");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // RoPE Kernel Creation - Device-aware dispatch
            // Now returns typed kernels directly (they implement ITensorRoPE)
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorRoPE> KernelFactory::createRoPE(
                const llaminar2::FP32Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor; // RoPE kernels don't need tensor state for creation
                switch (dev_type)
                {
                case DeviceType::CPU:
                    // Return typed kernel directly - it implements ITensorRoPE
                    return std::make_unique<llaminar2::CPURoPEKernelT<llaminar2::ActivationPrecision::FP32>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDARoPEKernelT<llaminar2::ActivationPrecision::FP32>>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmRoPEKernelT<llaminar2::ActivationPrecision::FP32>>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "RoPE", "FP32");
                }
            }

            std::unique_ptr<llaminar2::ITensorRoPE> KernelFactory::createRoPE(
                const llaminar2::BF16Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    // Return typed kernel directly - it implements ITensorRoPE
                    return std::make_unique<llaminar2::CPURoPEKernelT<llaminar2::ActivationPrecision::BF16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDARoPEKernelT<llaminar2::ActivationPrecision::BF16>>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmRoPEKernelT<llaminar2::ActivationPrecision::BF16>>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "RoPE", "BF16");
                }
            }

            std::unique_ptr<llaminar2::ITensorRoPE> KernelFactory::createRoPE(
                const llaminar2::FP16Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    // Return typed kernel directly - it implements ITensorRoPE
                    return std::make_unique<llaminar2::CPURoPEKernelT<llaminar2::ActivationPrecision::FP16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDARoPEKernelT<llaminar2::ActivationPrecision::FP16>>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmRoPEKernelT<llaminar2::ActivationPrecision::FP16>>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "RoPE", "FP16");
                }
            }

            std::unique_ptr<llaminar2::ITensorRoPE> KernelFactory::createRoPE(
                const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    // Return typed kernel directly - it implements ITensorRoPE
                    return std::make_unique<llaminar2::CPURoPEKernelT<llaminar2::ActivationPrecision::Q8_1>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    throwUnsupportedKernel(dev_type, "RoPE", "Q8_1");
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    throwUnsupportedKernel(dev_type, "RoPE", "Q8_1");
#endif

                default:
                    throwUnsupportedKernel(dev_type, "RoPE", "Q8_1");
                }
            }

            // ==========================================================================
            // SwiGLU Kernel Creation - Device-aware dispatch
            // NO CPU FALLBACK: If GPU requested but not implemented, throws
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorSwiGLU> KernelFactory::createSwiGLU(
                const llaminar2::FP32Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor; // SwiGLU kernels don't need tensor state for creation
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUSwiGLUKernelT<llaminar2::ActivationPrecision::FP32>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDASwiGLUKernelT<llaminar2::ActivationPrecision::FP32>>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmSwiGLUKernelT<llaminar2::ActivationPrecision::FP32>>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "SwiGLU", "FP32");
                }
            }

            std::unique_ptr<llaminar2::ITensorSwiGLU> KernelFactory::createSwiGLU(
                const llaminar2::BF16Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUSwiGLUKernelT<llaminar2::ActivationPrecision::BF16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDASwiGLUKernelT<llaminar2::ActivationPrecision::BF16>>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmSwiGLUKernelT<llaminar2::ActivationPrecision::BF16>>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "SwiGLU", "BF16");
                }
            }

            std::unique_ptr<llaminar2::ITensorSwiGLU> KernelFactory::createSwiGLU(
                const llaminar2::FP16Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUSwiGLUKernelT<llaminar2::ActivationPrecision::FP16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDASwiGLUKernelT<llaminar2::ActivationPrecision::FP16>>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmSwiGLUKernelT<llaminar2::ActivationPrecision::FP16>>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "SwiGLU", "FP16");
                }
            }

            std::unique_ptr<llaminar2::ITensorSwiGLU> KernelFactory::createSwiGLU(
                const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUSwiGLUKernelT<llaminar2::ActivationPrecision::Q8_1>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    throwUnsupportedKernel(dev_type, "SwiGLU", "Q8_1");
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    throwUnsupportedKernel(dev_type, "SwiGLU", "Q8_1");
#endif

                default:
                    throwUnsupportedKernel(dev_type, "SwiGLU", "Q8_1");
                }
            }

            // ==========================================================================
            // Softmax Kernel Creation - Device-aware dispatch
            // Now uses typed kernels directly (no adapter needed)
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorSoftmax> KernelFactory::createSoftmax(
                const llaminar2::FP32Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUSoftmaxKernelT<llaminar2::ActivationPrecision::FP32>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    throwUnsupportedKernel(dev_type, "Softmax", "FP32");
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    throwUnsupportedKernel(dev_type, "Softmax", "FP32");
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Softmax", "FP32");
                }
            }

            std::unique_ptr<llaminar2::ITensorSoftmax> KernelFactory::createSoftmax(
                const llaminar2::BF16Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUSoftmaxKernelT<llaminar2::ActivationPrecision::BF16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    throwUnsupportedKernel(dev_type, "Softmax", "BF16");
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    throwUnsupportedKernel(dev_type, "Softmax", "BF16");
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Softmax", "BF16");
                }
            }

            std::unique_ptr<llaminar2::ITensorSoftmax> KernelFactory::createSoftmax(
                const llaminar2::FP16Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUSoftmaxKernelT<llaminar2::ActivationPrecision::FP16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    throwUnsupportedKernel(dev_type, "Softmax", "FP16");
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    throwUnsupportedKernel(dev_type, "Softmax", "FP16");
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Softmax", "FP16");
                }
            }

            std::unique_ptr<llaminar2::ITensorSoftmax> KernelFactory::createSoftmax(
                const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUSoftmaxKernelT<llaminar2::ActivationPrecision::Q8_1>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    throwUnsupportedKernel(dev_type, "Softmax", "Q8_1");
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    throwUnsupportedKernel(dev_type, "Softmax", "Q8_1");
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Softmax", "Q8_1");
                }
            }

            // ==========================================================================
            // ResidualAdd Kernel Creation - Device-aware dispatch
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorResidualAdd> KernelFactory::createResidualAdd(
                const llaminar2::FP32Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUResidualAddKernelT<llaminar2::ActivationPrecision::FP32>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDAResidualAddKernelT<llaminar2::ActivationPrecision::FP32>>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmResidualAddKernelT<llaminar2::ActivationPrecision::FP32>>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "ResidualAdd", "FP32");
                }
            }

            std::unique_ptr<llaminar2::ITensorResidualAdd> KernelFactory::createResidualAdd(
                const llaminar2::BF16Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUResidualAddKernelT<llaminar2::ActivationPrecision::BF16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDAResidualAddKernelT<llaminar2::ActivationPrecision::BF16>>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmResidualAddKernelT<llaminar2::ActivationPrecision::BF16>>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "ResidualAdd", "BF16");
                }
            }

            std::unique_ptr<llaminar2::ITensorResidualAdd> KernelFactory::createResidualAdd(
                const llaminar2::FP16Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUResidualAddKernelT<llaminar2::ActivationPrecision::FP16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDAResidualAddKernelT<llaminar2::ActivationPrecision::FP16>>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmResidualAddKernelT<llaminar2::ActivationPrecision::FP16>>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "ResidualAdd", "FP16");
                }
            }

            // ==========================================================================
            // GDN (Gated Delta Net) Kernel Creation - FP32 only
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorShortConvolution> KernelFactory::createShortConvolution(
                DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUShortConvolution>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::CUDAShortConvolution>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::ROCmShortConvolution>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "ShortConvolution", "FP32");
                }
            }

            std::unique_ptr<llaminar2::ITensorGatedDeltaNet> KernelFactory::createGatedDeltaNet(
                DeviceType dev_type, int device_ordinal)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUGatedDeltaNet>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::CUDAGatedDeltaNet>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::ROCmGatedDeltaNet>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "GatedDeltaNet", "FP32");
                }
            }

            // ==========================================================================
            // RMSNorm Kernel Creation - Device-aware dispatch
            // NO CPU FALLBACK: If GPU requested but not implemented, throws
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorRMSNorm> KernelFactory::createRMSNorm(
                const llaminar2::FP32Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPURMSNormKernelT<llaminar2::ActivationPrecision::FP32>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDARMSNormKernelT<llaminar2::ActivationPrecision::FP32>>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmRMSNormKernelT<llaminar2::ActivationPrecision::FP32>>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "RMSNorm", "FP32");
                }
            }

            std::unique_ptr<llaminar2::ITensorRMSNorm> KernelFactory::createRMSNorm(
                const llaminar2::BF16Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPURMSNormKernelT<llaminar2::ActivationPrecision::BF16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDARMSNormKernelT<llaminar2::ActivationPrecision::BF16>>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmRMSNormKernelT<llaminar2::ActivationPrecision::BF16>>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "RMSNorm", "BF16");
                }
            }

            std::unique_ptr<llaminar2::ITensorRMSNorm> KernelFactory::createRMSNorm(
                const llaminar2::FP16Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPURMSNormKernelT<llaminar2::ActivationPrecision::FP16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDARMSNormKernelT<llaminar2::ActivationPrecision::FP16>>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmRMSNormKernelT<llaminar2::ActivationPrecision::FP16>>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "RMSNorm", "FP16");
                }
            }

            std::unique_ptr<llaminar2::ITensorRMSNorm> KernelFactory::createRMSNorm(
                const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPURMSNormKernelT<llaminar2::ActivationPrecision::Q8_1>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    throwUnsupportedKernel(dev_type, "RMSNorm", "Q8_1");
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    throwUnsupportedKernel(dev_type, "RMSNorm", "Q8_1");
#endif

                default:
                    throwUnsupportedKernel(dev_type, "RMSNorm", "Q8_1");
                }
            }

            std::unique_ptr<llaminar2::ITensorRMSNorm> KernelFactory::createRMSNorm(
                const llaminar2::Q16_1Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPURMSNormKernelT<llaminar2::ActivationPrecision::Q16_1>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    throwUnsupportedKernel(dev_type, "RMSNorm", "Q16_1");
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    throwUnsupportedKernel(dev_type, "RMSNorm", "Q16_1");
#endif

                default:
                    throwUnsupportedKernel(dev_type, "RMSNorm", "Q16_1");
                }
            }

            // ==========================================================================
            // Generic TensorBase* Factory Methods - Auto-dispatch by native_type()
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorSoftmax> KernelFactory::createSoftmax(
                const llaminar2::TensorBase *tensor, DeviceType dev_type, int device_ordinal)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::createSoftmax: null tensor");
                }

                switch (tensor->native_type())
                {
                case llaminar2::TensorType::FP32:
                    return createSoftmax(static_cast<const llaminar2::FP32Tensor *>(tensor), dev_type, device_ordinal);
                case llaminar2::TensorType::BF16:
                    return createSoftmax(static_cast<const llaminar2::BF16Tensor *>(tensor), dev_type, device_ordinal);
                case llaminar2::TensorType::FP16:
                    return createSoftmax(static_cast<const llaminar2::FP16Tensor *>(tensor), dev_type, device_ordinal);
                case llaminar2::TensorType::Q8_1:
                    return createSoftmax(static_cast<const llaminar2::Q8_1Tensor *>(tensor), dev_type, device_ordinal);
                default:
                    throw std::runtime_error(
                        "KernelFactory::createSoftmax: unsupported tensor type " +
                        std::string(tensor->dtype_name()));
                }
            }

            std::unique_ptr<llaminar2::ITensorRMSNorm> KernelFactory::createRMSNorm(
                const llaminar2::TensorBase *tensor, DeviceType dev_type, int device_ordinal)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::createRMSNorm: null tensor");
                }

                switch (tensor->native_type())
                {
                case llaminar2::TensorType::FP32:
                    return createRMSNorm(static_cast<const llaminar2::FP32Tensor *>(tensor), dev_type, device_ordinal);
                case llaminar2::TensorType::BF16:
                    return createRMSNorm(static_cast<const llaminar2::BF16Tensor *>(tensor), dev_type, device_ordinal);
                case llaminar2::TensorType::FP16:
                    return createRMSNorm(static_cast<const llaminar2::FP16Tensor *>(tensor), dev_type, device_ordinal);
                case llaminar2::TensorType::Q8_1:
                    return createRMSNorm(static_cast<const llaminar2::Q8_1Tensor *>(tensor), dev_type, device_ordinal);
                case llaminar2::TensorType::Q16_1:
                    return createRMSNorm(static_cast<const llaminar2::Q16_1Tensor *>(tensor), dev_type, device_ordinal);
                default:
                    throw std::runtime_error(
                        "KernelFactory::createRMSNorm: unsupported tensor type " +
                        std::string(tensor->dtype_name()));
                }
            }

            llaminar2::ITensorRoPE *KernelFactory::getOrCreateRoPE(
                const llaminar2::TensorBase *tensor,
                llaminar2::DeviceId target_device)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::getOrCreateRoPE: null tensor");
                }

                const DeviceKernelKey registry_key{
                    target_device,
                    KernelKind::ROPE,
                    static_cast<int>(tensor->native_type())};

                RoPECacheKey key{target_device, static_cast<int>(tensor->native_type())};

                std::lock_guard<std::mutex> lock(cache_mutex_);

                // Phase A registry front-door (behavior-preserving shim)
                auto reg_it = device_kernel_registry_.find(registry_key);
                if (reg_it != device_kernel_registry_.end())
                {
                    LOG_DEBUG("[KernelFactory][Registry] hit kind=ROPE dev=" << static_cast<int>(target_device.type)
                                                                             << ":" << target_device.ordinal
                                                                             << " variant=" << static_cast<int>(tensor->native_type())
                                                                             << " ptr=" << reg_it->second.get());
                    return static_cast<llaminar2::ITensorRoPE *>(reg_it->second.get());
                }

                auto it = rope_cache_.find(key);
                if (it != rope_cache_.end())
                {
                    auto *raw_ptr = it->second.get();
                    device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
                    LOG_DEBUG("[KernelFactory][Registry] backfill kind=ROPE dev=" << static_cast<int>(target_device.type)
                                                                                  << ":" << target_device.ordinal
                                                                                  << " variant=" << static_cast<int>(tensor->native_type())
                                                                                  << " ptr=" << raw_ptr);
                    return raw_ptr;
                }

                auto kernel = createRoPE(tensor, getDeviceType(target_device), target_device.ordinal);
                auto *raw_ptr = kernel.get();
                rope_cache_[key] = std::move(kernel);
                device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
                LOG_DEBUG("[KernelFactory][Registry] create kind=ROPE dev=" << static_cast<int>(target_device.type)
                                                                            << ":" << target_device.ordinal
                                                                            << " variant=" << static_cast<int>(tensor->native_type())
                                                                            << " ptr=" << raw_ptr);
                return raw_ptr;
            }

            llaminar2::ITensorRMSNorm *KernelFactory::getOrCreateRMSNorm(
                const llaminar2::TensorBase *tensor,
                llaminar2::DeviceId target_device)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::getOrCreateRMSNorm: null tensor");
                }

                const DeviceKernelKey registry_key{
                    target_device,
                    KernelKind::RMSNORM,
                    static_cast<int>(tensor->native_type())};

                RMSNormCacheKey key{target_device, static_cast<int>(tensor->native_type())};

                std::lock_guard<std::mutex> lock(cache_mutex_);

                auto reg_it = device_kernel_registry_.find(registry_key);
                if (reg_it != device_kernel_registry_.end())
                {
                    LOG_DEBUG("[KernelFactory][RMSNORM] registry hit dev=" << static_cast<int>(target_device.type)
                                                                           << ":" << target_device.ordinal
                                                                           << " tensor_type=" << static_cast<int>(tensor->native_type())
                                                                           << " kernel=" << reg_it->second.get());
                    return static_cast<llaminar2::ITensorRMSNorm *>(reg_it->second.get());
                }

                auto it = rmsnorm_cache_.find(key);
                if (it != rmsnorm_cache_.end())
                {
                    auto *raw_ptr = it->second.get();
                    device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
                    LOG_DEBUG("[KernelFactory][RMSNORM] cache hit dev=" << static_cast<int>(target_device.type)
                                                                        << ":" << target_device.ordinal
                                                                        << " tensor_type=" << static_cast<int>(tensor->native_type())
                                                                        << " kernel=" << static_cast<const void *>(raw_ptr));
                    return raw_ptr;
                }

                auto kernel = createRMSNorm(tensor, getDeviceType(target_device), target_device.ordinal);
                auto *raw_ptr = kernel.get();
                rmsnorm_cache_[key] = std::move(kernel);
                device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
                LOG_DEBUG("[KernelFactory][RMSNORM] create dev=" << static_cast<int>(target_device.type)
                                                                 << ":" << target_device.ordinal
                                                                 << " tensor_type=" << static_cast<int>(tensor->native_type())
                                                                 << " kernel=" << static_cast<const void *>(raw_ptr));
                LOG_DEBUG("[KernelFactory][Registry] create kind=RMSNORM dev=" << static_cast<int>(target_device.type)
                                                                               << ":" << target_device.ordinal
                                                                               << " variant=" << static_cast<int>(tensor->native_type())
                                                                               << " ptr=" << raw_ptr);
                return raw_ptr;
            }

            llaminar2::ITensorSwiGLU *KernelFactory::getOrCreateSwiGLU(
                const llaminar2::TensorBase *tensor,
                llaminar2::DeviceId target_device)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::getOrCreateSwiGLU: null tensor");
                }

                const DeviceKernelKey registry_key{
                    target_device,
                    KernelKind::SWIGLU,
                    static_cast<int>(tensor->native_type())};

                SwiGLUCacheKey key{target_device, static_cast<int>(tensor->native_type())};

                std::lock_guard<std::mutex> lock(cache_mutex_);

                auto reg_it = device_kernel_registry_.find(registry_key);
                if (reg_it != device_kernel_registry_.end())
                {
                    LOG_DEBUG("[KernelFactory][Registry] hit kind=SWIGLU dev=" << static_cast<int>(target_device.type)
                                                                               << ":" << target_device.ordinal
                                                                               << " variant=" << static_cast<int>(tensor->native_type())
                                                                               << " ptr=" << reg_it->second.get());
                    return static_cast<llaminar2::ITensorSwiGLU *>(reg_it->second.get());
                }

                auto it = swiglu_cache_.find(key);
                if (it != swiglu_cache_.end())
                {
                    auto *raw_ptr = it->second.get();
                    device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
                    LOG_DEBUG("[KernelFactory][Registry] backfill kind=SWIGLU dev=" << static_cast<int>(target_device.type)
                                                                                    << ":" << target_device.ordinal
                                                                                    << " variant=" << static_cast<int>(tensor->native_type())
                                                                                    << " ptr=" << raw_ptr);
                    return raw_ptr;
                }

                auto kernel = createSwiGLU(tensor, getDeviceType(target_device), target_device.ordinal);
                auto *raw_ptr = kernel.get();
                swiglu_cache_[key] = std::move(kernel);
                device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
                LOG_DEBUG("[KernelFactory][Registry] create kind=SWIGLU dev=" << static_cast<int>(target_device.type)
                                                                              << ":" << target_device.ordinal
                                                                              << " variant=" << static_cast<int>(tensor->native_type())
                                                                              << " ptr=" << raw_ptr);
                return raw_ptr;
            }

            llaminar2::ITensorSoftmax *KernelFactory::getOrCreateSoftmax(
                const llaminar2::TensorBase *tensor,
                llaminar2::DeviceId target_device)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::getOrCreateSoftmax: null tensor");
                }

                const DeviceKernelKey registry_key{
                    target_device,
                    KernelKind::SOFTMAX,
                    static_cast<int>(tensor->native_type())};

                SoftmaxCacheKey key{target_device, static_cast<int>(tensor->native_type())};

                std::lock_guard<std::mutex> lock(cache_mutex_);

                auto reg_it = device_kernel_registry_.find(registry_key);
                if (reg_it != device_kernel_registry_.end())
                {
                    LOG_DEBUG("[KernelFactory][Registry] hit kind=SOFTMAX dev=" << static_cast<int>(target_device.type)
                                                                                << ":" << target_device.ordinal
                                                                                << " variant=" << static_cast<int>(tensor->native_type())
                                                                                << " ptr=" << reg_it->second.get());
                    return static_cast<llaminar2::ITensorSoftmax *>(reg_it->second.get());
                }

                auto it = softmax_cache_.find(key);
                if (it != softmax_cache_.end())
                {
                    auto *raw_ptr = it->second.get();
                    device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
                    LOG_DEBUG("[KernelFactory][Registry] backfill kind=SOFTMAX dev=" << static_cast<int>(target_device.type)
                                                                                     << ":" << target_device.ordinal
                                                                                     << " variant=" << static_cast<int>(tensor->native_type())
                                                                                     << " ptr=" << raw_ptr);
                    return raw_ptr;
                }

                auto kernel = createSoftmax(tensor, getDeviceType(target_device), target_device.ordinal);
                auto *raw_ptr = kernel.get();
                softmax_cache_[key] = std::move(kernel);
                device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
                LOG_DEBUG("[KernelFactory][Registry] create kind=SOFTMAX dev=" << static_cast<int>(target_device.type)
                                                                               << ":" << target_device.ordinal
                                                                               << " variant=" << static_cast<int>(tensor->native_type())
                                                                               << " ptr=" << raw_ptr);
                return raw_ptr;
            }

            llaminar2::ITensorResidualAdd *KernelFactory::getOrCreateResidualAdd(
                const llaminar2::TensorBase *tensor,
                llaminar2::DeviceId target_device)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::getOrCreateResidualAdd: null tensor");
                }

                const DeviceKernelKey registry_key{
                    target_device,
                    KernelKind::RESIDUAL_ADD,
                    static_cast<int>(tensor->native_type())};

                ResidualAddCacheKey key{target_device, static_cast<int>(tensor->native_type())};

                std::lock_guard<std::mutex> lock(cache_mutex_);

                auto reg_it = device_kernel_registry_.find(registry_key);
                if (reg_it != device_kernel_registry_.end())
                {
                    LOG_DEBUG("[KernelFactory][Registry] hit kind=RESIDUAL_ADD dev=" << static_cast<int>(target_device.type)
                                                                                     << ":" << target_device.ordinal
                                                                                     << " variant=" << static_cast<int>(tensor->native_type())
                                                                                     << " ptr=" << reg_it->second.get());
                    return static_cast<llaminar2::ITensorResidualAdd *>(reg_it->second.get());
                }

                auto it = residual_add_cache_.find(key);
                if (it != residual_add_cache_.end())
                {
                    auto *raw_ptr = it->second.get();
                    device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
                    LOG_DEBUG("[KernelFactory][Registry] backfill kind=RESIDUAL_ADD dev=" << static_cast<int>(target_device.type)
                                                                                          << ":" << target_device.ordinal
                                                                                          << " variant=" << static_cast<int>(tensor->native_type())
                                                                                          << " ptr=" << raw_ptr);
                    return raw_ptr;
                }

                auto kernel = createResidualAdd(tensor, getDeviceType(target_device), target_device.ordinal);
                auto *raw_ptr = kernel.get();
                residual_add_cache_[key] = std::move(kernel);
                device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
                LOG_DEBUG("[KernelFactory][Registry] create kind=RESIDUAL_ADD dev=" << static_cast<int>(target_device.type)
                                                                                    << ":" << target_device.ordinal
                                                                                    << " variant=" << static_cast<int>(tensor->native_type())
                                                                                    << " ptr=" << raw_ptr);
                return raw_ptr;
            }

            llaminar2::ITensorAttention *KernelFactory::getOrCreateAttention(
                const llaminar2::ITensor *tensor,
                llaminar2::DeviceId target_device)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::getOrCreateAttention: null tensor");
                }

                const DeviceKernelKey registry_key{
                    target_device,
                    KernelKind::ATTENTION,
                    static_cast<int>(tensor->native_type())};

                AttentionCacheKey key{target_device, static_cast<int>(tensor->native_type())};

                std::lock_guard<std::mutex> lock(cache_mutex_);

                auto reg_it = device_kernel_registry_.find(registry_key);
                if (reg_it != device_kernel_registry_.end())
                {
                    LOG_DEBUG("[KernelFactory][Registry] hit kind=ATTENTION dev=" << static_cast<int>(target_device.type)
                                                                                  << ":" << target_device.ordinal
                                                                                  << " variant=" << static_cast<int>(tensor->native_type())
                                                                                  << " ptr=" << reg_it->second.get());
                    return static_cast<llaminar2::ITensorAttention *>(reg_it->second.get());
                }

                auto it = attention_cache_.find(key);
                if (it != attention_cache_.end())
                {
                    auto *raw_ptr = it->second.get();
                    device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
                    LOG_DEBUG("[KernelFactory][Registry] backfill kind=ATTENTION dev=" << static_cast<int>(target_device.type)
                                                                                       << ":" << target_device.ordinal
                                                                                       << " variant=" << static_cast<int>(tensor->native_type())
                                                                                       << " ptr=" << raw_ptr);
                    return raw_ptr;
                }

                auto kernel = createAttention(tensor, getDeviceType(target_device), target_device.ordinal);
                auto *raw_ptr = kernel.get();
                attention_cache_[key] = std::move(kernel);
                device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
                LOG_DEBUG("[KernelFactory][Registry] create kind=ATTENTION dev=" << static_cast<int>(target_device.type)
                                                                                 << ":" << target_device.ordinal
                                                                                 << " variant=" << static_cast<int>(tensor->native_type())
                                                                                 << " ptr=" << raw_ptr);
                return raw_ptr;
            }

            llaminar2::ITensorEmbedding *KernelFactory::getOrCreateEmbedding(
                const llaminar2::TensorBase *tensor,
                llaminar2::DeviceId target_device)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::getOrCreateEmbedding: null tensor");
                }

                const DeviceKernelKey registry_key{
                    target_device,
                    KernelKind::EMBEDDING,
                    static_cast<int>(tensor->native_type())};

                EmbeddingCacheKey key{target_device, static_cast<int>(tensor->native_type())};

                std::lock_guard<std::mutex> lock(cache_mutex_);

                auto reg_it = device_kernel_registry_.find(registry_key);
                if (reg_it != device_kernel_registry_.end())
                {
                    LOG_DEBUG("[KernelFactory][Registry] hit kind=EMBEDDING dev=" << static_cast<int>(target_device.type)
                                                                                  << ":" << target_device.ordinal
                                                                                  << " variant=" << static_cast<int>(tensor->native_type())
                                                                                  << " ptr=" << reg_it->second.get());
                    return static_cast<llaminar2::ITensorEmbedding *>(reg_it->second.get());
                }

                auto it = embedding_cache_.find(key);
                if (it != embedding_cache_.end())
                {
                    auto *raw_ptr = it->second.get();
                    device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
                    LOG_DEBUG("[KernelFactory][Registry] backfill kind=EMBEDDING dev=" << static_cast<int>(target_device.type)
                                                                                       << ":" << target_device.ordinal
                                                                                       << " variant=" << static_cast<int>(tensor->native_type())
                                                                                       << " ptr=" << raw_ptr);
                    return raw_ptr;
                }

                auto dev_type = getDeviceType(target_device);
                std::unique_ptr<llaminar2::ITensorEmbedding> kernel;

                switch (tensor->native_type())
                {
                case llaminar2::TensorType::FP32:
                    kernel = createEmbedding(static_cast<const llaminar2::FP32Tensor *>(tensor), dev_type, target_device.ordinal);
                    break;
                case llaminar2::TensorType::BF16:
                    kernel = createEmbedding(static_cast<const llaminar2::BF16Tensor *>(tensor), dev_type, target_device.ordinal);
                    break;
                case llaminar2::TensorType::FP16:
                    kernel = createEmbedding(static_cast<const llaminar2::FP16Tensor *>(tensor), dev_type, target_device.ordinal);
                    break;
                case llaminar2::TensorType::Q8_1:
                    kernel = createEmbedding(static_cast<const llaminar2::Q8_1Tensor *>(tensor), dev_type, target_device.ordinal);
                    break;
                default:
                    throw std::runtime_error(
                        "KernelFactory::getOrCreateEmbedding: unsupported tensor type " +
                        std::string(tensor->dtype_name()));
                }

                auto *raw_ptr = kernel.get();
                embedding_cache_[key] = std::move(kernel);
                device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
                LOG_DEBUG("[KernelFactory][Registry] create kind=EMBEDDING dev=" << static_cast<int>(target_device.type)
                                                                                 << ":" << target_device.ordinal
                                                                                 << " variant=" << static_cast<int>(tensor->native_type())
                                                                                 << " ptr=" << raw_ptr);
                return raw_ptr;
            }

            llaminar2::IMoEKernel *KernelFactory::getOrCreateMoEKernel(
                llaminar2::DeviceId target_device)
            {
                const DeviceKernelKey registry_key{
                    target_device,
                    KernelKind::MOE,
                    0}; // No variant (always FP32)

                MoECacheKey key{target_device};

                std::lock_guard<std::mutex> lock(cache_mutex_);

                auto reg_it = device_kernel_registry_.find(registry_key);
                if (reg_it != device_kernel_registry_.end())
                {
                    return static_cast<llaminar2::IMoEKernel *>(reg_it->second.get());
                }

                auto it = moe_cache_.find(key);
                if (it != moe_cache_.end())
                {
                    auto *raw_ptr = it->second.get();
                    device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
                    return raw_ptr;
                }

                // Create device-appropriate MoE kernel. CUDA/ROCm devices must
                // never silently fall through to CPU: that would pull routing
                // tensors back to host and break graph-capture assumptions.
                std::unique_ptr<llaminar2::IMoEKernel> kernel;
#ifdef HAVE_CUDA
                if (target_device.is_cuda())
                {
                    kernel = std::make_unique<llaminar2::CUDAMoEKernel>(target_device.cuda_ordinal());
                }
                else
#else
                if (target_device.is_cuda())
                {
                    throw std::runtime_error("KernelFactory::getOrCreateMoEKernel: CUDA device requested but HAVE_CUDA is OFF");
                }
                else
#endif
#ifdef HAVE_ROCM
                if (target_device.is_rocm())
                {
                    kernel = std::make_unique<llaminar2::ROCmMoEKernel>(target_device.rocm_ordinal());
                }
                else
#else
                if (target_device.is_rocm())
                {
                    throw std::runtime_error("KernelFactory::getOrCreateMoEKernel: ROCm device requested but HAVE_ROCM is OFF");
                }
                else
#endif
                if (target_device.is_cpu())
                {
                    kernel = std::make_unique<llaminar2::CPUMoEKernel>();
                }
                else
                {
                    throw std::runtime_error("KernelFactory::getOrCreateMoEKernel: invalid target device " + target_device.to_string());
                }

                auto *raw_ptr = kernel.get();
                moe_cache_[key] = std::move(kernel);
                device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
                LOG_DEBUG("[KernelFactory][MOE] create dev=" << static_cast<int>(target_device.type)
                                                             << ":" << target_device.ordinal
                                                             << " kernel=" << static_cast<const void *>(raw_ptr));
                return raw_ptr;
            }

            std::unique_ptr<llaminar2::ITensorRoPE> KernelFactory::createRoPE(
                const llaminar2::TensorBase *tensor, DeviceType dev_type, int device_ordinal)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::createRoPE: null tensor");
                }

                switch (tensor->native_type())
                {
                case llaminar2::TensorType::FP32:
                    return createRoPE(static_cast<const llaminar2::FP32Tensor *>(tensor), dev_type, device_ordinal);
                case llaminar2::TensorType::BF16:
                    return createRoPE(static_cast<const llaminar2::BF16Tensor *>(tensor), dev_type, device_ordinal);
                case llaminar2::TensorType::FP16:
                    return createRoPE(static_cast<const llaminar2::FP16Tensor *>(tensor), dev_type, device_ordinal);
                case llaminar2::TensorType::Q8_1:
                    return createRoPE(static_cast<const llaminar2::Q8_1Tensor *>(tensor), dev_type, device_ordinal);
                default:
                    throw std::runtime_error(
                        "KernelFactory::createRoPE: unsupported tensor type " +
                        std::string(tensor->dtype_name()));
                }
            }

            std::unique_ptr<llaminar2::ITensorSwiGLU> KernelFactory::createSwiGLU(
                const llaminar2::TensorBase *tensor, DeviceType dev_type, int device_ordinal)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::createSwiGLU: null tensor");
                }

                switch (tensor->native_type())
                {
                case llaminar2::TensorType::FP32:
                    return createSwiGLU(static_cast<const llaminar2::FP32Tensor *>(tensor), dev_type, device_ordinal);
                case llaminar2::TensorType::BF16:
                    return createSwiGLU(static_cast<const llaminar2::BF16Tensor *>(tensor), dev_type, device_ordinal);
                case llaminar2::TensorType::FP16:
                    return createSwiGLU(static_cast<const llaminar2::FP16Tensor *>(tensor), dev_type, device_ordinal);
                case llaminar2::TensorType::Q8_1:
                    return createSwiGLU(static_cast<const llaminar2::Q8_1Tensor *>(tensor), dev_type, device_ordinal);
                default:
                    throw std::runtime_error(
                        "KernelFactory::createSwiGLU: unsupported tensor type " +
                        std::string(tensor->dtype_name()));
                }
            }

            std::unique_ptr<llaminar2::ITensorResidualAdd> KernelFactory::createResidualAdd(
                const llaminar2::TensorBase *tensor, DeviceType dev_type, int device_ordinal)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::createResidualAdd: null tensor");
                }

                switch (tensor->native_type())
                {
                case llaminar2::TensorType::FP32:
                    return createResidualAdd(static_cast<const llaminar2::FP32Tensor *>(tensor), dev_type, device_ordinal);
                case llaminar2::TensorType::BF16:
                    return createResidualAdd(static_cast<const llaminar2::BF16Tensor *>(tensor), dev_type, device_ordinal);
                case llaminar2::TensorType::FP16:
                    return createResidualAdd(static_cast<const llaminar2::FP16Tensor *>(tensor), dev_type, device_ordinal);
                default:
                    throw std::runtime_error(
                        "KernelFactory::createResidualAdd: unsupported tensor type " +
                        std::string(tensor->dtype_name()));
                }
            }

            std::unique_ptr<llaminar2::ITensorAttention> KernelFactory::createAttention(
                const llaminar2::TensorBase *tensor, DeviceType dev_type, int device_ordinal)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::createAttention: null tensor");
                }

                switch (tensor->native_type())
                {
                case llaminar2::TensorType::FP32:
                    return createAttention(static_cast<const llaminar2::FP32Tensor *>(tensor), dev_type, device_ordinal);
                case llaminar2::TensorType::BF16:
                    return createAttention(static_cast<const llaminar2::BF16Tensor *>(tensor), dev_type, device_ordinal);
                case llaminar2::TensorType::FP16:
                    return createAttention(static_cast<const llaminar2::FP16Tensor *>(tensor), dev_type, device_ordinal);
                default:
                    throw std::runtime_error(
                        "KernelFactory::createAttention: unsupported tensor type " +
                        std::string(tensor->dtype_name()));
                }
            }

            std::unique_ptr<llaminar2::ITensorAttention> KernelFactory::createAttention(
                const llaminar2::ITensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::createAttention: null tensor");
                }

                // Dispatch based on REQUESTED device type, not tensor location.
                // Tensor data location is a coherence concern handled at execution time.
                // Stage device_id determines which kernel to use.
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    // For CPU execution, use TensorBase* dispatch path
                    auto *cpu_tensor = dynamic_cast<const llaminar2::TensorBase *>(tensor);
                    if (!cpu_tensor)
                    {
                        throw std::runtime_error("KernelFactory::createAttention: ITensor for CPU must be TensorBase");
                    }
                    return createAttention(cpu_tensor, dev_type, device_ordinal);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    switch (tensor->native_type())
                    {
                    case llaminar2::TensorType::FP32:
                        return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::FP32>>(device_ordinal);
                    case llaminar2::TensorType::FP16:
                        return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::FP16>>(device_ordinal);
                    case llaminar2::TensorType::BF16:
                        return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::BF16>>(device_ordinal);
                    default:
                        throw std::runtime_error(
                            "KernelFactory::createAttention: unsupported CUDA tensor type " +
                            std::string(tensor->dtype_name()));
                    }
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    switch (tensor->native_type())
                    {
                    case llaminar2::TensorType::FP32:
                        return std::make_unique<llaminar2::rocm::ROCmFlashAttentionKernelT<llaminar2::ActivationPrecision::FP32>>(device_ordinal);
                    case llaminar2::TensorType::FP16:
                        return std::make_unique<llaminar2::rocm::ROCmFlashAttentionKernelT<llaminar2::ActivationPrecision::FP16>>(device_ordinal);
                    case llaminar2::TensorType::BF16:
                        return std::make_unique<llaminar2::rocm::ROCmFlashAttentionKernelT<llaminar2::ActivationPrecision::BF16>>(device_ordinal);
                    default:
                        throw std::runtime_error(
                            "KernelFactory::createAttention: unsupported ROCm tensor type " +
                            std::string(tensor->dtype_name()));
                    }
#endif

                default:
                    throw std::runtime_error(
                        "KernelFactory::createAttention: unsupported device type for ITensor");
                }
            }

            // ==========================================================================
            // Attention Kernel Creation - Device-aware dispatch (typed overloads)
            // NO CPU FALLBACK: If GPU requested but not implemented, throws
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorAttention> KernelFactory::createAttention(
                const llaminar2::FP32Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUFlashAttentionKernelT<llaminar2::ActivationPrecision::FP32>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::FP32>>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmFlashAttentionKernelT<llaminar2::ActivationPrecision::FP32>>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Attention", "FP32");
                }
            }

            std::unique_ptr<llaminar2::ITensorAttention> KernelFactory::createAttention(
                const llaminar2::BF16Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUFlashAttentionKernelT<llaminar2::ActivationPrecision::BF16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::BF16>>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmFlashAttentionKernelT<llaminar2::ActivationPrecision::BF16>>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Attention", "BF16");
                }
            }

            std::unique_ptr<llaminar2::ITensorAttention> KernelFactory::createAttention(
                const llaminar2::FP16Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUFlashAttentionKernelT<llaminar2::ActivationPrecision::FP16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::FP16>>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmFlashAttentionKernelT<llaminar2::ActivationPrecision::FP16>>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Attention", "FP16");
                }
            }

            // ==========================================================================
            // Embedding Kernel Creation - Device-aware dispatch
            // NO CPU FALLBACK: If GPU requested but not implemented, throws
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorEmbedding> KernelFactory::createEmbedding(
                const llaminar2::FP32Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUEmbeddingKernelT<llaminar2::FP32Tensor>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::CUDAEmbeddingKernelT>(device_ordinal);
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::ROCmEmbeddingKernelT>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Embedding", "FP32");
                }
            }

            std::unique_ptr<llaminar2::ITensorEmbedding> KernelFactory::createEmbedding(
                const llaminar2::BF16Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUEmbeddingKernelT<llaminar2::BF16Tensor>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    throwUnsupportedKernel(dev_type, "Embedding", "BF16");
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::ROCmEmbeddingKernelT>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Embedding", "BF16");
                }
            }

            std::unique_ptr<llaminar2::ITensorEmbedding> KernelFactory::createEmbedding(
                const llaminar2::FP16Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUEmbeddingKernelT<llaminar2::FP16Tensor>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    throwUnsupportedKernel(dev_type, "Embedding", "FP16");
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::ROCmEmbeddingKernelT>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Embedding", "FP16");
                }
            }

            std::unique_ptr<llaminar2::ITensorEmbedding> KernelFactory::createEmbedding(
                const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type, int device_ordinal)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUEmbeddingKernelT<llaminar2::Q8_1Tensor>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    throwUnsupportedKernel(dev_type, "Embedding", "Q8_1");
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::ROCmEmbeddingKernelT>(device_ordinal);
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Embedding", "Q8_1");
                }
            }

            // ==========================================================================
            // Kernel Cache - Static Members
            // ==========================================================================

            std::mutex KernelFactory::cache_mutex_;
            std::unordered_map<KernelFactory::FusedGateUpCacheKey, std::unique_ptr<llaminar2::ITensorFusedGateUpGemm>, KernelFactory::FusedGateUpKeyHash> KernelFactory::fused_gate_up_cache_;
            std::unordered_map<KernelFactory::RoPECacheKey, std::unique_ptr<llaminar2::ITensorRoPE>, KernelFactory::RoPECacheKeyHash> KernelFactory::rope_cache_;
            std::unordered_map<KernelFactory::RMSNormCacheKey, std::unique_ptr<llaminar2::ITensorRMSNorm>, KernelFactory::RMSNormCacheKeyHash> KernelFactory::rmsnorm_cache_;
            std::unordered_map<KernelFactory::SwiGLUCacheKey, std::unique_ptr<llaminar2::ITensorSwiGLU>, KernelFactory::SwiGLUCacheKeyHash> KernelFactory::swiglu_cache_;
            std::unordered_map<KernelFactory::SoftmaxCacheKey, std::unique_ptr<llaminar2::ITensorSoftmax>, KernelFactory::SoftmaxCacheKeyHash> KernelFactory::softmax_cache_;
            std::unordered_map<KernelFactory::ResidualAddCacheKey, std::unique_ptr<llaminar2::ITensorResidualAdd>, KernelFactory::ResidualAddCacheKeyHash> KernelFactory::residual_add_cache_;
            std::unordered_map<KernelFactory::AttentionCacheKey, std::unique_ptr<llaminar2::ITensorAttention>, KernelFactory::AttentionCacheKeyHash> KernelFactory::attention_cache_;
            std::unordered_map<KernelFactory::EmbeddingCacheKey, std::unique_ptr<llaminar2::ITensorEmbedding>, KernelFactory::EmbeddingCacheKeyHash> KernelFactory::embedding_cache_;
            std::unordered_map<KernelFactory::MoECacheKey, std::unique_ptr<llaminar2::IMoEKernel>, KernelFactory::MoECacheKeyHash> KernelFactory::moe_cache_;
            std::unordered_map<DeviceKernelKey, std::shared_ptr<void>, DeviceKernelKeyHash> KernelFactory::device_kernel_registry_;
            std::unordered_map<DeviceKernelKey, std::shared_ptr<KernelFactory::IGemmEngine>, DeviceKernelKeyHash> KernelFactory::device_gemm_engine_registry_;
            // NOTE: Device-level kernel caching (hipBLAS, cuBLAS handles, etc.)
            // has moved to DeviceKernelCache. See kernels/DeviceKernelCache.h

            // ==========================================================================
            // Kernel Cache - Implementation
            // ==========================================================================

            /**
             * @brief Ensure tensor has packed weights in its cache and return pointer
             *
             * For CPU NativeVNNI: no-op since kernels pack weights internally on construction.
             * For legacy JIT path: removed.
             *
             * Thread-safe: acquires cache_mutex_ internally.
             *
             * @param tensor Quantized tensor that implements IINT8Unpackable
             * @return nullptr (CPU NativeVNNI kernels manage their own packing)
             */
            const llaminar2::gemm::QuantisedPackedWeights *
            KernelFactory::ensurePackedWeightsInTensorCache(const llaminar2::TensorBase *tensor)
            {
                // NativeVNNI kernels pack weights on construction — no external packing needed.
                return nullptr;
            }

            // NOTE: Shared device kernel caching (hipBLAS, cuBLAS handles, etc.)
            // has been moved to DeviceKernelCache. See kernels/DeviceKernelCache.h

            void KernelFactory::clearCacheFor(const llaminar2::TensorBase *tensor)
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);

                // Also clean up tensor's packed weights cache if present (CPU VNNI)
                // Note: caches store std::shared_ptr<T>, so resetting the std::any
                // drops the shared_ptr which handles cleanup automatically.
                {
                    std::lock_guard<std::mutex> tensor_lock(tensor->packed_cache_mutex_);
                    if (tensor->cache_.has_value())
                    {
                        tensor->cache_.reset();
                    }
                }

                // Clear any fused Gate/Up kernels that reference this tensor
                for (auto it = fused_gate_up_cache_.begin(); it != fused_gate_up_cache_.end();)
                {
                    const auto *gate_handle = it->first.w_gate_handle;
                    const auto *up_handle = it->first.w_up_handle;
                    const bool references_tensor =
                        (gate_handle && gate_handle->tensor == tensor) ||
                        (up_handle && up_handle->tensor == tensor);

                    if (references_tensor)
                    {
                        it = fused_gate_up_cache_.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
            }

            void KernelFactory::clearCache()
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);

                std::unordered_set<const llaminar2::TensorBase *> tracked_tensors;
                tracked_tensors.reserve(fused_gate_up_cache_.size());

                for (const auto &entry : fused_gate_up_cache_)
                {
                    if (entry.first.w_gate_handle && entry.first.w_gate_handle->tensor)
                    {
                        tracked_tensors.insert(entry.first.w_gate_handle->tensor);
                    }
                    if (entry.first.w_up_handle && entry.first.w_up_handle->tensor)
                    {
                        tracked_tensors.insert(entry.first.w_up_handle->tensor);
                    }
                }

                // Clean up all tensor packed weights caches (CPU VNNI and CUDA INT8)
                for (const auto *tensor : tracked_tensors)
                {
                    if (!tensor)
                    {
                        continue;
                    }

                    std::lock_guard<std::mutex> tensor_lock(tensor->packed_cache_mutex_);

                    // Clean up CPU packed weights
                    // Note: caches store std::shared_ptr<T>, so resetting the std::any
                    // drops the shared_ptr which handles cleanup automatically.
                    if (tensor->cache_.has_value())
                    {
                        tensor->cache_.reset();
                    }
                }

                fused_gate_up_cache_.clear();
                rope_cache_.clear();
                rmsnorm_cache_.clear();
                swiglu_cache_.clear();
                softmax_cache_.clear();
                residual_add_cache_.clear();
                attention_cache_.clear();
                embedding_cache_.clear();
                moe_cache_.clear();
                device_kernel_registry_.clear();
                device_gemm_engine_registry_.clear();
#ifdef HAVE_CUDA
                // Clear CUDA GEMV static caches (row-major weight transpose, sweep
                // state) to prevent cross-test contamination.
                cudaNativeVNNIGemvTuned_clearStaticState();

                // Release per-device shared CUDAConcurrentPrefillPool singletons
                // (CUDA streams, events, scratch GPU buffers).
                llaminar2::cuda::CUDAQuantisedGemmKernel::clearSharedPrefillPools();
#endif

#ifdef HAVE_ROCM
                // Release per-device shared ConcurrentPrefillPool singletons
                // (HIP streams, events, scratch/scatter GPU buffers).
                llaminar2::rocm::ROCmQuantisedGemmKernel::clearSharedPrefillPools();
#endif
            }

            void KernelFactory::resetAllDynamicState()
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);

                auto reset_kernel_cache = [](auto &cache)
                {
                    for (auto &[_, kernel] : cache)
                    {
                        if (!kernel)
                            continue;
                        kernel->resetDynamicState();
                        kernel->setGPUStream(nullptr);
                    }
                };

                reset_kernel_cache(fused_gate_up_cache_);
                reset_kernel_cache(rope_cache_);
                reset_kernel_cache(rmsnorm_cache_);
                reset_kernel_cache(swiglu_cache_);
                reset_kernel_cache(softmax_cache_);
                reset_kernel_cache(residual_add_cache_);
                reset_kernel_cache(attention_cache_);
                reset_kernel_cache(embedding_cache_);
                reset_kernel_cache(moe_cache_);

#if LLAMINAR_ASSERTIONS_ACTIVE
                // Post-reset assertion: no kernel should retain stale dynamic state
                for (const auto &[key, kernel] : embedding_cache_)
                {
                    if (kernel)
                    {
                        LLAMINAR_ASSERT(!kernel->hasDynamicStateActive(),
                                        "[KernelFactory] Embedding kernel still has dynamic state active after reset");
                    }
                }
#endif

                LOG_DEBUG("[KernelFactory] Reset dynamic state on "
                          << embedding_cache_.size() << " embedding, "
                          << attention_cache_.size() << " attention, "
                          << rope_cache_.size() << " RoPE, "
                          << moe_cache_.size() << " MoE kernels");
            }

            std::pair<size_t, size_t> KernelFactory::cacheStats()
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                size_t total_bytes = 0;
                // Note: packed_bytes not tracked per-kernel
                // For now, just return count (includes all caches)
                return {fused_gate_up_cache_.size() + rope_cache_.size() + rmsnorm_cache_.size() + swiglu_cache_.size() + softmax_cache_.size() + residual_add_cache_.size() + attention_cache_.size() + embedding_cache_.size() + moe_cache_.size() + device_kernel_registry_.size(), total_bytes};
            }

            std::shared_ptr<void> KernelFactory::getDeviceKernelEntry(const DeviceKernelKey &key)
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                auto it = device_kernel_registry_.find(key);
                if (it == device_kernel_registry_.end())
                {
                    return {};
                }
                return it->second;
            }

            void KernelFactory::putDeviceKernelEntry(const DeviceKernelKey &key, std::shared_ptr<void> entry)
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                device_kernel_registry_[key] = std::move(entry);
            }

            void KernelFactory::clearDeviceKernelRegistry()
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                device_kernel_registry_.clear();
            }

            size_t KernelFactory::deviceKernelRegistrySize()
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                return device_kernel_registry_.size();
            }

            namespace
            {
                static std::unique_ptr<llaminar2::ITensorGemm> createPreparedKernelForDevice(
                    const llaminar2::TensorBase *tensor,
                    llaminar2::DeviceId target_device)
                {
                    const auto dev_type = KernelFactory::getDeviceType(target_device);

#ifdef HAVE_ROCM
                    if (dev_type == DeviceType::ROCm)
                    {
                        KernelFactory::ROCmOrdinalGuard guard(target_device.rocm_ordinal());
                        return KernelFactory::createGemm(tensor, dev_type);
                    }
#endif

#ifdef HAVE_CUDA
                    if (dev_type == DeviceType::CUDA)
                    {
                        KernelFactory::CUDAOrdinalGuard guard(target_device.cuda_ordinal());
                        return KernelFactory::createGemm(tensor, dev_type);
                    }
#endif

                    return KernelFactory::createGemm(tensor, dev_type);
                }

                class PhaseCGemmDeviceEngine : public KernelFactory::IGemmEngine
                {
                public:
                    PhaseCGemmDeviceEngine(llaminar2::DeviceId device_id, int variant)
                        : device_id_(device_id), variant_(variant)
                    {
                    }

                    llaminar2::ITensorGemm *resolveKernel(
                        const llaminar::v2::kernels::KernelFactory::PreparedGemmHandle *prepared) const override
                    {
                        if (!prepared || !prepared->tensor)
                        {
                            throw std::runtime_error("PhaseCGemmDeviceEngine::resolveKernel: null input");
                        }

                        if (prepared->variant != variant_)
                        {
                            throw std::runtime_error("PhaseCGemmDeviceEngine::resolveKernel: variant mismatch");
                        }

                        if (prepared->device_id != device_id_)
                        {
                            throw std::runtime_error("PhaseCGemmDeviceEngine::resolveKernel: device mismatch");
                        }

                        if (!prepared->prepared_weights)
                        {
                            throw std::runtime_error("PhaseCGemmDeviceEngine::resolveKernel: missing prepared weights");
                        }

                        if (prepared->prepared_weights->kind != prepared->kind)
                        {
                            throw std::runtime_error("PhaseCGemmDeviceEngine::resolveKernel: prepared kind mismatch");
                        }

                        if (!prepared->prepared_weights->kernel)
                        {
                            throw std::runtime_error("PhaseCGemmDeviceEngine::resolveKernel: missing prepared kernel binding");
                        }

                        return prepared->prepared_weights->kernel;
                    }

                    llaminar2::DeviceId device_id() const { return device_id_; }
                    int variant() const { return variant_; }

                private:
                    llaminar2::DeviceId device_id_;
                    int variant_{0};
                };
            } // namespace

            llaminar2::ITensorGemm *KernelFactory::getOrCreateGemmEngine(
                const PreparedGemmHandle *prepared)
            {
                if (!prepared)
                {
                    throw std::runtime_error("KernelFactory::getOrCreateGemmEngine: null prepared handle");
                }
                if (!prepared->tensor)
                {
                    throw std::runtime_error("KernelFactory::getOrCreateGemmEngine: prepared handle has null tensor");
                }

                const DeviceKernelKey device_engine_key{
                    prepared->device_id,
                    KernelKind::GEMM,
                    prepared->variant};

                std::shared_ptr<IGemmEngine> device_engine;
                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    auto dev_it = device_gemm_engine_registry_.find(device_engine_key);
                    if (dev_it == device_gemm_engine_registry_.end())
                    {
                        device_engine = std::make_shared<PhaseCGemmDeviceEngine>(prepared->device_id, prepared->variant);
                        device_gemm_engine_registry_[device_engine_key] = device_engine;
                        LOG_DEBUG("[KernelFactory][PhaseC] device GEMM engine create dev=" << static_cast<int>(prepared->device_id.type)
                                                                                           << ":" << prepared->device_id.ordinal
                                                                                           << " variant=" << prepared->variant);
                    }
                    else
                    {
                        device_engine = dev_it->second;
                        LOG_DEBUG("[KernelFactory][PhaseC] device GEMM engine hit dev=" << static_cast<int>(prepared->device_id.type)
                                                                                        << ":" << prepared->device_id.ordinal
                                                                                        << " variant=" << prepared->variant);
                    }
                }

                auto *kernel = device_engine->resolveKernel(prepared);
                LOG_DEBUG("[KernelFactory][PhaseC] GEMM kernel via device engine dev=" << static_cast<int>(prepared->device_id.type)
                                                                                       << ":" << prepared->device_id.ordinal
                                                                                       << " prepared=" << prepared
                                                                                       << " tensor=" << prepared->tensor
                                                                                       << " variant=" << prepared->variant
                                                                                       << " ptr=" << kernel);
                return kernel;
            }

            namespace
            {
                KernelFactory::GemmPreparationKind resolveGemmPreparationKind(
                    const llaminar2::TensorBase *tensor,
                    llaminar2::DeviceId target_device,
                    KernelFactory::GemmPreparationKind prep_kind)
                {
                    const bool quantized = isVnniPackableTensor(tensor);
                    if (prep_kind != KernelFactory::GemmPreparationKind::AUTO)
                        return prep_kind;

                    if (!quantized)
                        return KernelFactory::GemmPreparationKind::FLOATING_POINT;

                    switch (KernelFactory::getDeviceType(target_device))
                    {
                    case DeviceType::CPU:
                        return KernelFactory::GemmPreparationKind::CPU_PACKED;
                    case DeviceType::CUDA:
                        return KernelFactory::GemmPreparationKind::CUDA_INT8_PACKED;
                    case DeviceType::ROCm:
                        return KernelFactory::GemmPreparationKind::ROCM_INT8_PACKED;
                    default:
                        return KernelFactory::GemmPreparationKind::FLOATING_POINT;
                    }
                }
            }

            std::shared_ptr<KernelFactory::PreparedGemmHandle> KernelFactory::prepareGemmHandleLocal(
                const llaminar2::TensorBase *tensor,
                llaminar2::DeviceId target_device,
                GemmPreparationKind prep_kind)
            {
                if (!tensor)
                    throw std::runtime_error("KernelFactory::prepareGemmHandleLocal: null tensor");

                const bool quantized = isVnniPackableTensor(tensor);
                const GemmPreparationKind resolved_kind = resolveGemmPreparationKind(tensor, target_device, prep_kind);

                if (!tensor->raw_data())
                {
                    throw std::runtime_error(
                        "KernelFactory::prepareGemmHandleLocal: tensor host data has already been released");
                }

                if (resolved_kind == GemmPreparationKind::CUDA_INT8_PACKED ||
                    resolved_kind == GemmPreparationKind::ROCM_INT8_PACKED)
                {
                    throw std::runtime_error(
                        "KernelFactory::prepareGemmHandleLocal: GPU prepared GEMM handles must be registered by the PreparedWeightStore-backed load pipeline");
                }

                if (quantized && resolved_kind == GemmPreparationKind::CPU_PACKED)
                    (void)ensurePackedWeightsInTensorCache(tensor);

                auto bound_kernel = createPreparedKernelForDevice(tensor, target_device);
                if (!bound_kernel)
                    throw std::runtime_error("KernelFactory::prepareGemmHandleLocal: failed to bind prepared kernel");

                auto handle = std::make_shared<PreparedGemmHandle>();
                handle->tensor = tensor;
                handle->device_id = target_device;
                handle->kind = resolved_kind;
                handle->variant = static_cast<int>(tensor->native_type());
                handle->prepared_weights = std::make_shared<PreparedGemmWeights>();
                handle->prepared_weights->kind = resolved_kind;
                handle->prepared_weights->owned_kernel = std::shared_ptr<llaminar2::ITensorGemm>(std::move(bound_kernel));
                handle->prepared_weights->kernel = handle->prepared_weights->owned_kernel.get();
                return handle;
            }

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemmSlicedLocal(
                const llaminar2::TensorBase *tensor,
                size_t row_start,
                size_t row_end)
            {
                if (!tensor)
                {
                    LOG_ERROR("[KernelFactory] createGemmSlicedLocal: tensor is null");
                    return nullptr;
                }

                const auto &shape = tensor->shape();
                if (shape.size() != 2)
                    throw std::runtime_error("KernelFactory: tensor must be 2D");
                const size_t tensor_n = shape[0];
                if (row_start >= row_end || row_end > tensor_n)
                    throw std::runtime_error("KernelFactory: invalid row range for sliced GEMM");

                auto current_dev = tensor->current_device();
                DeviceType dev_type = DeviceType::CPU;
                if (current_dev.has_value() && current_dev->is_gpu())
                    dev_type = current_dev->is_cuda() ? DeviceType::CUDA : DeviceType::ROCm;
                if (dev_type != DeviceType::CPU)
                    throw std::runtime_error("KernelFactory: sliced GEMM only supported on CPU");

                return std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(
                    tensor, static_cast<int>(row_start), static_cast<int>(row_end));
            }

            std::shared_ptr<llaminar2::ITensorGemm> KernelFactory::prepareExpertGemmLocal(
                const llaminar2::TensorBase *tensor,
                llaminar2::DeviceId target_device,
                GemmPreparationKind prep_kind)
            {
                if (!tensor)
                    return nullptr;

                const bool quantized = isVnniPackableTensor(tensor);

                GemmPreparationKind resolved_kind = prep_kind;
                if (resolved_kind == GemmPreparationKind::AUTO)
                {
                    if (!quantized)
                    {
                        resolved_kind = GemmPreparationKind::FLOATING_POINT;
                    }
                    else
                    {
                        switch (getDeviceType(target_device))
                        {
                        case DeviceType::CPU:
                            resolved_kind = GemmPreparationKind::CPU_PACKED;
                            break;
                        case DeviceType::CUDA:
                            resolved_kind = GemmPreparationKind::CUDA_INT8_PACKED;
                            break;
                        case DeviceType::ROCm:
                            resolved_kind = GemmPreparationKind::ROCM_INT8_PACKED;
                            break;
                        default:
                            resolved_kind = GemmPreparationKind::FLOATING_POINT;
                            break;
                        }
                    }
                }

                // GPU devices should use LoadOrchestrator pipeline, not this path
                if (resolved_kind == GemmPreparationKind::CUDA_INT8_PACKED ||
                    resolved_kind == GemmPreparationKind::ROCM_INT8_PACKED)
                {
                    LOG_ERROR("[KernelFactory::prepareExpertGemmLocal] GPU experts should use "
                              "LoadOrchestrator pipeline, not local preparation");
                    return nullptr;
                }

                if (!tensor->raw_data())
                {
                    LOG_ERROR("[KernelFactory::prepareExpertGemmLocal] Tensor has no raw data "
                              "(host data already released?)");
                    return nullptr;
                }

                // Pack weights into tensor-local cache (same as global path)
                if (quantized && resolved_kind == GemmPreparationKind::CPU_PACKED)
                    (void)ensurePackedWeightsInTensorCache(tensor);

                // Create the GEMM kernel bound to this tensor
                auto kernel = createPreparedKernelForDevice(tensor, target_device);
                if (!kernel)
                {
                    LOG_ERROR("[KernelFactory::prepareExpertGemmLocal] Failed to create kernel for expert view");
                    return nullptr;
                }

                // Return ownership to caller — NO global registry insertion,
                // NO has_prepared_device_state_ flag set
                return std::shared_ptr<llaminar2::ITensorGemm>(std::move(kernel));
            }

            std::shared_ptr<llaminar2::ITensorGemm> KernelFactory::createExpertGemmFromTransferBlob(
                const std::vector<uint8_t> &blob)
            {
                if (blob.empty())
                    return nullptr;

                auto packed_weights = llaminar2::packed_weights_serialization::deserialize(
                    blob.data(), blob.size());
                if (!packed_weights)
                {
                    LOG_ERROR("[KernelFactory::createExpertGemmFromTransferBlob] "
                              "Failed to deserialize transferred blob ("
                              << blob.size() << " bytes)");
                    return nullptr;
                }

                auto *cpu_pw = dynamic_cast<llaminar2::cpu::native_vnni::CPUPackedWeights *>(
                    packed_weights.get());
                if (!cpu_pw)
                {
                    LOG_ERROR("[KernelFactory::createExpertGemmFromTransferBlob] "
                              "Deserialized weights are not CPUPackedWeights");
                    return nullptr;
                }

                if (dynamic_cast<llaminar2::cpu::native_vnni::CPUPackedWeightsWithNativeBlocks *>(
                        packed_weights.get()))
                {
                    LOG_ERROR("[KernelFactory::createExpertGemmFromTransferBlob] "
                              "Deferred/native-block CPU VNNI blobs are no longer accepted; expected eager interleaved packed weights");
                    return nullptr;
                }

                auto kernel = std::make_shared<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(
                    cpu_pw->takePacked());
                if (!kernel->isValid())
                {
                    LOG_ERROR("[KernelFactory::createExpertGemmFromTransferBlob] "
                              "Transferred CPU VNNI blob did not contain eager interleaved weights");
                    return nullptr;
                }
                return kernel;
            }

            size_t KernelFactory::gemmEngineRegistrySize()
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                return device_gemm_engine_registry_.size();
            }

            size_t KernelFactory::deviceScopedGemmEngineRegistrySize()
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                return device_gemm_engine_registry_.size();
            }

            // ==========================================================================
            // Prepared Embedding Weights
            // ==========================================================================

            std::shared_ptr<llaminar2::PreparedEmbeddingHandle> KernelFactory::prepareEmbeddingHandleLocal(
                const llaminar2::TensorBase *tensor,
                int d_model,
                llaminar2::DeviceId target_device,
                size_t vocab_offset,
                size_t total_vocab)
            {
                if (!tensor)
                    throw std::runtime_error("prepareEmbeddingHandleLocal: null tensor");

                if (!isVnniPackableTensor(tensor))
                    return nullptr;

                const size_t shard_rows = tensor->rows();
                const size_t effective_total = (total_vocab > 0) ? total_vocab : shard_rows;
                const bool is_sharded = (shard_rows < effective_total);

                auto repacked = llaminar2::repackEmbeddingToQ8(tensor, d_model);

                auto weights = std::make_shared<llaminar2::PreparedEmbeddingWeights>();
                weights->byte_size = repacked.byte_size;
                weights->blocks_per_row = repacked.blocks_per_row;
                weights->vocab_size = repacked.vocab_size;
                weights->vocab_offset = vocab_offset;
                weights->total_vocab = effective_total;
                weights->d_model = d_model;
                weights->device_id = target_device;

                llaminar2::IBackend *backend = llaminar2::getBackendFor(target_device);
                if (!backend)
                {
                    LOG_ERROR("[PreparedEmbeddingWeights] No backend for device " << target_device.to_string());
                    return nullptr;
                }

                weights->device_data = backend->allocate(repacked.byte_size, target_device.ordinal);
                if (!weights->device_data)
                {
                    LOG_ERROR("[PreparedEmbeddingWeights] GPU allocation failed for "
                              << target_device.to_string() << " (" << (repacked.byte_size / (1024 * 1024)) << " MB)");
                    return nullptr;
                }

                const bool upload_ok = backend->hostToDevice(weights->device_data, repacked.data.data(),
                                                             repacked.byte_size, target_device.ordinal);
                if (!upload_ok)
                {
                    LOG_ERROR("[PreparedEmbeddingWeights] H2D upload failed for " << target_device.to_string());
                    backend->free(weights->device_data, target_device.ordinal);
                    weights->device_data = nullptr;
                    return nullptr;
                }

                LOG_DEBUG("[PreparedEmbeddingWeights] Prepared embedding for "
                          << target_device.to_string() << ": "
                          << llaminar2::tensorTypeName(tensor->native_type()) << " "
                          << repacked.vocab_size << "x" << d_model
                          << (is_sharded ? (" (vocab_offset=" + std::to_string(vocab_offset) +
                                            " of " + std::to_string(effective_total) + ")")
                                         : "")
                          << " → " << (repacked.byte_size / (1024 * 1024)) << " MB"
                          << " (" << repacked.blocks_per_row << " blocks/row)");

                auto handle = std::make_shared<llaminar2::PreparedEmbeddingHandle>();
                handle->tensor = tensor;
                handle->device_id = target_device;
                handle->weights = std::move(weights);
                return handle;
            }

            // ==========================================================================
            // Fused Gate/Up GEMM Adapter and Factory Methods
            // ==========================================================================

            /**
             * @brief Adapter class that wraps two ITensorGemm kernels for Gate/Up fusion
             *
             * This adapter holds raw pointers to cached GEMM kernels from KernelFactory.
             * The kernels are obtained via prepared-handle + getOrCreateGemmEngine()
             * which ensures proper caching.
             *
             * Note: The adapter does NOT own the GEMM kernels - they are owned by
             * KernelFactory's kernel cache. The adapter must not outlive the cache.
             */
            class FusedGateUpGemmAdapter : public llaminar2::ITensorFusedGateUpGemm
            {
            public:
                FusedGateUpGemmAdapter(llaminar2::ITensorGemm *gemm_gate,
                                       llaminar2::ITensorGemm *gemm_up,
                                       DeviceType dev_type)
                    : gemm_gate_(gemm_gate), gemm_up_(gemm_up), dev_type_(dev_type)
                {
                }

                // Propagate GPU stream to internal GEMM kernels so they
                // launch on the correct stream during graph capture/replay.
                void setGPUStream(void *stream) override
                {
                    llaminar2::ITensorFusedGateUpGemm::setGPUStream(stream);
                    if (gemm_gate_)
                        gemm_gate_->setGPUStream(stream);
                    if (gemm_up_)
                        gemm_up_->setGPUStream(stream);
                }

                bool execute(
                    const llaminar2::TensorBase *input,
                    llaminar2::TensorBase *output_gate,
                    llaminar2::TensorBase *output_up,
                    int m, int k, int n_gate, int n_up,
                    llaminar2::IDeviceContext *ctx,
                    int device_idx) override
                {
                    (void)ctx; // Currently unused, kernels use device_idx directly
                    (void)device_idx;

                    if (!gemm_gate_ || !gemm_up_)
                    {
                        LOG_ERROR("[FusedGateUpGemmAdapter] Null GEMM kernel(s)");
                        return false;
                    }

                    // Standard per-projection path (shares activation quantization)
                    std::vector<llaminar2::ITensorGemm::TensorProjectionDesc> projections = {
                        {gemm_gate_, output_gate, n_gate, nullptr, "gate"},
                        {gemm_up_, output_up, n_up, nullptr, "up"}};

                    return gemm_gate_->multiply_fused_tensor(input, projections, m, k, nullptr, bound_workspace_);
                }

                bool execute_with_bias(
                    const llaminar2::TensorBase *input,
                    llaminar2::TensorBase *output_gate,
                    llaminar2::TensorBase *output_up,
                    const llaminar2::TensorBase *bias_gate,
                    const llaminar2::TensorBase *bias_up,
                    int m, int k, int n_gate, int n_up,
                    llaminar2::IDeviceContext *ctx,
                    int device_idx) override
                {
                    (void)ctx;

                    // For bias support, use the tensor-aware fused path
                    if (!gemm_gate_ || !gemm_up_)
                    {
                        LOG_ERROR("[FusedGateUpGemmAdapter] Null GEMM kernel(s)");
                        return false;
                    }

                    // Use tensor-aware fused API
                    {
                        // Build tensor projection descriptors
                        std::vector<llaminar2::ITensorGemm::TensorProjectionDesc> projections = {
                            {gemm_gate_, output_gate, n_gate, bias_gate, "gate"},
                            {gemm_up_, output_up, n_up, bias_up, "up"}};

                        return gemm_gate_->multiply_fused_tensor(input, projections, m, k, nullptr, bound_workspace_);
                    }
                }

                bool supports_device(int device_idx) const override
                {
                    // Supports same device as underlying kernels
                    if (dev_type_ == DeviceType::CPU)
                    {
                        return device_idx == -1;
                    }
                    return device_idx >= 0;
                }

                // =============================================================
                // IWorkspaceConsumer Implementation (forwards to both kernels)
                // =============================================================

                llaminar2::WorkspaceRequirements getWorkspaceRequirements(int m, int n, int k) const override
                {
                    // Aggregate requirements from both kernels
                    llaminar2::WorkspaceRequirements combined;

                    auto *gate_consumer = dynamic_cast<llaminar2::IWorkspaceConsumer *>(gemm_gate_);
                    auto *up_consumer = dynamic_cast<llaminar2::IWorkspaceConsumer *>(gemm_up_);

                    if (gate_consumer)
                    {
                        auto gate_reqs = gate_consumer->getWorkspaceRequirements(m, n, k);
                        // Merge gate requirements into combined
                        for (const auto &buf : gate_reqs.buffers)
                        {
                            combined.buffers.push_back(buf);
                        }
                    }
                    if (up_consumer)
                    {
                        auto up_reqs = up_consumer->getWorkspaceRequirements(m, n, k);
                        // Merge up requirements into combined
                        for (const auto &buf : up_reqs.buffers)
                        {
                            combined.buffers.push_back(buf);
                        }
                    }

                    return combined;
                }

                void bindWorkspace(llaminar2::DeviceWorkspaceManager *workspace) override
                {
                    // Forward to both underlying kernels
                    auto *gate_consumer = dynamic_cast<llaminar2::IWorkspaceConsumer *>(gemm_gate_);
                    auto *up_consumer = dynamic_cast<llaminar2::IWorkspaceConsumer *>(gemm_up_);

                    if (gate_consumer)
                    {
                        gate_consumer->bindWorkspace(workspace);
                        LOG_DEBUG("[FusedGateUpGemmAdapter] Bound workspace to gate kernel");
                    }
                    if (up_consumer)
                    {
                        up_consumer->bindWorkspace(workspace);
                        LOG_DEBUG("[FusedGateUpGemmAdapter] Bound workspace to up kernel");
                    }

                    bound_workspace_ = workspace;
                }

                void unbindWorkspace() override
                {
                    auto *gate_consumer = dynamic_cast<llaminar2::IWorkspaceConsumer *>(gemm_gate_);
                    auto *up_consumer = dynamic_cast<llaminar2::IWorkspaceConsumer *>(gemm_up_);

                    if (gate_consumer)
                    {
                        gate_consumer->unbindWorkspace();
                    }
                    if (up_consumer)
                    {
                        up_consumer->unbindWorkspace();
                    }

                    bound_workspace_ = nullptr;
                }

                bool hasWorkspace() const override
                {
                    return bound_workspace_ != nullptr;
                }

                llaminar2::DeviceWorkspaceManager *getWorkspace() const override
                {
                    return bound_workspace_;
                }

            private:
                llaminar2::ITensorGemm *gemm_gate_;
                llaminar2::ITensorGemm *gemm_up_;
                DeviceType dev_type_;
                llaminar2::DeviceWorkspaceManager *bound_workspace_ = nullptr;
            };

            /**
             * @brief Build fused Gate/Up adapter from already-resolved prepared handles.
             *
             * Mirrors fused QKV helper semantics:
             * - prepared-handle-native input contract
             * - centralized validation and adapter creation
             * - shared by explicit create and cache-miss paths
             */
            static std::unique_ptr<llaminar2::ITensorFusedGateUpGemm> createFusedGateUpAdapterFromPrepared(
                const KernelFactory::PreparedGemmHandle *prepared_gate,
                const KernelFactory::PreparedGemmHandle *prepared_up,
                llaminar2::DeviceId target_device)
            {
                // This helper keeps fused Gate/Up construction prepared-handle-native.
                // Both create() and getOrCreate() cache-miss paths call into this logic,
                // preventing behavioral drift between those two entrypoints.
                if (!prepared_gate || !prepared_up)
                {
                    LOG_ERROR("[KernelFactory] createFusedGateUpAdapterFromPrepared: null prepared handle(s)");
                    throw std::runtime_error("FusedGateUpGemm requires non-null prepared handles");
                }

                auto *gemm_gate = KernelFactory::getOrCreateGemmEngine(prepared_gate);
                auto *gemm_up = KernelFactory::getOrCreateGemmEngine(prepared_up);

                if (!gemm_gate || !gemm_up)
                {
                    LOG_ERROR("[KernelFactory] createFusedGateUpAdapterFromPrepared: Failed to create GEMM kernels");
                    throw std::runtime_error("FusedGateUpGemm failed to create underlying GEMM kernels");
                }

                // Underlying GEMM kernels are already resolved for the exact DeviceId
                // through prepared handles; adapter keeps device type metadata for
                // runtime capability checks.
                return std::make_unique<FusedGateUpGemmAdapter>(gemm_gate, gemm_up, target_device.type);
            }

            std::unique_ptr<llaminar2::ITensorFusedGateUpGemm> KernelFactory::createFusedGateUpGemmLocal(
                const PreparedGemmHandle *prepared_gate,
                const PreparedGemmHandle *prepared_up,
                llaminar2::DeviceId target_device)
            {
                return createFusedGateUpAdapterFromPrepared(prepared_gate, prepared_up, target_device);
            }

            // ==========================================================================
            // KVCacheConfig estimation
            // ==========================================================================

            size_t KVCacheConfig::estimateBytes() const
            {
                int effective_kv_heads = (local_n_kv_heads > 0) ? local_n_kv_heads : n_kv_heads;
                if (effective_kv_heads <= 0 || num_layers <= 0 || head_dim <= 0)
                    return 0;

                std::string prec_str;
                switch (precision)
                {
                case ::llaminar2::ActivationPrecision::FP32:
                    prec_str = "fp32";
                    break;
                case ::llaminar2::ActivationPrecision::FP16:
                    prec_str = "fp16";
                    break;
                case ::llaminar2::ActivationPrecision::Q8_1:
                    prec_str = "q8_1";
                    break;
                default:
                    prec_str = "fp16";
                    break;
                }

                return ::llaminar2::KVCacheMemoryEstimator::estimate(
                    num_layers, batch_size, max_seq_len,
                    effective_kv_heads, head_dim, prec_str, device);
            }

            // ==========================================================================
            // KVCache Factory Methods
            // ==========================================================================

            std::unique_ptr<llaminar2::IKVCache> KernelFactory::createKVCache(const KVCacheConfig &config)
            {
                // If hybrid config is provided, create a hybrid cache
                if (config.is_hybrid())
                {
                    return createHybridKVCache(config);
                }

                if (config.device.is_cpu())
                {
                    return createCPUKVCache(config);
                }
                else if (config.device.is_cuda())
                {
#ifdef HAVE_CUDA
                    // TQ precision uses a separate non-template class
                    if (config.precision == llaminar2::ActivationPrecision::TQ4 ||
                        config.precision == llaminar2::ActivationPrecision::TQ8)
                    {
                        const int cuda_device = config.device.cuda_ordinal();
                        return std::make_unique<llaminar2::CUDARingKVCacheTQ>(
                            config.num_layers, config.batch_size, config.max_seq_len,
                            config.n_kv_heads, config.head_dim,
                            config.turboquant_ctx, cuda_device);
                    }
                    return createCUDAKVCache(config);
#else
                    LOG_ERROR("[KernelFactory] CUDA KVCache requested but HAVE_CUDA not defined");
                    throw std::runtime_error("KernelFactory::createKVCache: CUDA support not compiled in");
#endif
                }
                else if (config.device.is_rocm())
                {
#ifdef HAVE_ROCM
                    // TQ precision uses a separate non-template class
                    if (config.precision == llaminar2::ActivationPrecision::TQ4 ||
                        config.precision == llaminar2::ActivationPrecision::TQ8)
                    {
                        const int rocm_device = config.device.rocm_ordinal();
                        return llaminar2::createROCmRingKVCacheTQ(
                            config.num_layers, config.batch_size, config.max_seq_len,
                            config.n_kv_heads, config.head_dim,
                            config.turboquant_ctx, rocm_device);
                    }
                    return createROCmKVCache(config);
#else
                    LOG_ERROR("[KernelFactory] ROCm KVCache requested but HAVE_ROCM not defined");
                    throw std::runtime_error("KernelFactory::createKVCache: ROCm support not compiled in");
#endif
                }
                else
                {
                    LOG_ERROR("[KernelFactory] Unsupported device type for KVCache: " << config.device.to_string());
                    throw std::runtime_error("KernelFactory::createKVCache: Unsupported device type");
                }
            }

            std::unique_ptr<llaminar2::ICPUKVCache> KernelFactory::createCPUKVCache(const KVCacheConfig &config)
            {
                // Validate MPI context
                if (!config.mpi_ctx)
                {
                    LOG_ERROR("[KernelFactory] createCPUKVCache requires non-null mpi_ctx");
                    throw std::runtime_error("KernelFactory::createCPUKVCache: mpi_ctx is required");
                }

                // Validate basic parameters
                if (config.num_layers <= 0)
                {
                    LOG_ERROR("[KernelFactory] createCPUKVCache: num_layers must be positive, got " << config.num_layers);
                    throw std::runtime_error("KernelFactory::createCPUKVCache: invalid num_layers");
                }
                if (config.n_kv_heads <= 0)
                {
                    LOG_ERROR("[KernelFactory] createCPUKVCache: n_kv_heads must be positive, got " << config.n_kv_heads);
                    throw std::runtime_error("KernelFactory::createCPUKVCache: invalid n_kv_heads");
                }
                if (config.head_dim <= 0)
                {
                    LOG_ERROR("[KernelFactory] createCPUKVCache: head_dim must be positive, got " << config.head_dim);
                    throw std::runtime_error("KernelFactory::createCPUKVCache: invalid head_dim");
                }

                if (config.is_sharded())
                {
                    LOG_DEBUG("[KernelFactory] Creating sharded CPU RingKVCache: "
                              << "precision=" << llaminar2::activationPrecisionToString(config.precision)
                              << ", layers=" << config.num_layers
                              << ", n_kv_heads=" << config.n_kv_heads
                              << ", local_n_kv_heads=" << config.local_n_kv_heads
                              << ", kv_head_start=" << config.kv_head_start
                              << ", head_dim=" << config.head_dim
                              << ", max_seq_len=" << config.max_seq_len);

                    return llaminar2::createShardedCPURingKVCache(
                        config.precision,
                        *config.mpi_ctx,
                        config.num_layers,
                        config.batch_size,
                        config.max_seq_len,
                        config.n_kv_heads,
                        config.local_n_kv_heads,
                        config.kv_head_start,
                        config.head_dim,
                        config.device,
                        config.layout_mode);
                }
                else
                {
                    LOG_DEBUG("[KernelFactory] Creating standard CPU RingKVCache: "
                              << "precision=" << llaminar2::activationPrecisionToString(config.precision)
                              << ", layers=" << config.num_layers
                              << ", n_kv_heads=" << config.n_kv_heads
                              << ", head_dim=" << config.head_dim
                              << ", max_seq_len=" << config.max_seq_len);

                    return llaminar2::createCPURingKVCache(
                        config.precision,
                        *config.mpi_ctx,
                        config.num_layers,
                        config.batch_size,
                        config.max_seq_len,
                        config.n_kv_heads,
                        config.head_dim,
                        config.device,
                        config.layout_mode);
                }
            }

#ifdef HAVE_CUDA
            std::unique_ptr<llaminar2::ICUDARingKVCache> KernelFactory::createCUDAKVCache(const KVCacheConfig &config)
            {
                // Validate precision - CUDA ring cache supports FP32/BF16/FP16/Q8_1
                switch (config.precision)
                {
                case llaminar2::ActivationPrecision::FP32:
                case llaminar2::ActivationPrecision::BF16:
                case llaminar2::ActivationPrecision::FP16:
                case llaminar2::ActivationPrecision::Q8_1:
                    break; // Supported

                case llaminar2::ActivationPrecision::Q16_1:
                case llaminar2::ActivationPrecision::Hybrid:
                case llaminar2::ActivationPrecision::HybridQ16:
                    LOG_ERROR("[KernelFactory] CUDA KVCache does not support precision: "
                              << llaminar2::activationPrecisionToString(config.precision)
                              << ". Use FP32, BF16, FP16, or Q8_1.");
                    throw std::runtime_error("KernelFactory::createCUDAKVCache: Unsupported precision");

                default:
                    LOG_ERROR("[KernelFactory] Unknown precision for CUDA KVCache: "
                              << static_cast<int>(config.precision));
                    throw std::runtime_error("KernelFactory::createCUDAKVCache: Unknown precision");
                }

                // Validate basic parameters
                if (config.num_layers <= 0 || config.n_kv_heads <= 0 || config.head_dim <= 0)
                {
                    LOG_ERROR("[KernelFactory] createCUDAKVCache: invalid parameters");
                    throw std::runtime_error("KernelFactory::createCUDAKVCache: invalid parameters");
                }

                int cuda_device = config.device.cuda_ordinal();

                // Handle sharding for tensor parallelism
                if (config.is_sharded())
                {
                    LOG_DEBUG("[KernelFactory] Creating sharded CUDA Ring KVCache: "
                              << "precision=" << llaminar2::activationPrecisionToString(config.precision)
                              << ", device=CUDA:" << cuda_device
                              << ", layers=" << config.num_layers
                              << ", total_kv_heads=" << config.n_kv_heads
                              << ", local_kv_heads=" << config.local_n_kv_heads
                              << ", kv_head_start=" << config.kv_head_start
                              << ", head_dim=" << config.head_dim
                              << ", max_seq_len=" << config.max_seq_len);

                    return llaminar2::createShardedCUDARingKVCache(
                        config.precision,
                        config.num_layers,
                        config.batch_size,
                        config.max_seq_len,
                        config.n_kv_heads,
                        config.local_n_kv_heads,
                        config.kv_head_start,
                        config.head_dim,
                        cuda_device);
                }

                LOG_DEBUG("[KernelFactory] Creating CUDA Ring KVCache: "
                          << "precision=" << llaminar2::activationPrecisionToString(config.precision)
                          << ", device=CUDA:" << cuda_device
                          << ", layers=" << config.num_layers
                          << ", n_kv_heads=" << config.n_kv_heads
                          << ", head_dim=" << config.head_dim
                          << ", max_seq_len=" << config.max_seq_len);

                // Create appropriate precision variant using ActivationPrecision template parameter
                switch (config.precision)
                {
                case llaminar2::ActivationPrecision::FP32:
                    return std::make_unique<llaminar2::CUDARingKVCache<llaminar2::ActivationPrecision::FP32>>(
                        config.num_layers,
                        config.batch_size,
                        config.max_seq_len,
                        config.n_kv_heads,
                        config.head_dim,
                        cuda_device);

                case llaminar2::ActivationPrecision::BF16:
                    return std::make_unique<llaminar2::CUDARingKVCache<llaminar2::ActivationPrecision::BF16>>(
                        config.num_layers,
                        config.batch_size,
                        config.max_seq_len,
                        config.n_kv_heads,
                        config.head_dim,
                        cuda_device);

                case llaminar2::ActivationPrecision::FP16:
                    return std::make_unique<llaminar2::CUDARingKVCache<llaminar2::ActivationPrecision::FP16>>(
                        config.num_layers,
                        config.batch_size,
                        config.max_seq_len,
                        config.n_kv_heads,
                        config.head_dim,
                        cuda_device);

                case llaminar2::ActivationPrecision::Q8_1:
                    return std::make_unique<llaminar2::CUDARingKVCache<llaminar2::ActivationPrecision::Q8_1>>(
                        config.num_layers,
                        config.batch_size,
                        config.max_seq_len,
                        config.n_kv_heads,
                        config.head_dim,
                        cuda_device);

                default:
                    // Should not reach here due to earlier validation
                    throw std::runtime_error("KernelFactory::createCUDAKVCache: Unexpected precision");
                }
            }
#endif // HAVE_CUDA

#ifdef HAVE_ROCM
            std::unique_ptr<llaminar2::IKVCache> KernelFactory::createROCmKVCache(const KVCacheConfig &config)
            {
                // Validate precision - ROCm ring cache only supports FP32/BF16/FP16
                switch (config.precision)
                {
                case llaminar2::ActivationPrecision::FP32:
                case llaminar2::ActivationPrecision::BF16:
                case llaminar2::ActivationPrecision::FP16:
                case llaminar2::ActivationPrecision::Q8_1:
                    break; // Supported

                case llaminar2::ActivationPrecision::Q16_1:
                case llaminar2::ActivationPrecision::Hybrid:
                case llaminar2::ActivationPrecision::HybridQ16:
                    LOG_ERROR("[KernelFactory] ROCm KVCache does not support precision: "
                              << llaminar2::activationPrecisionToString(config.precision)
                              << ". Use FP32, BF16, FP16, or Q8_1.");
                    throw std::runtime_error("KernelFactory::createROCmKVCache: Unsupported precision");

                default:
                    LOG_ERROR("[KernelFactory] Unknown precision for ROCm KVCache: "
                              << static_cast<int>(config.precision));
                    throw std::runtime_error("KernelFactory::createROCmKVCache: Unknown precision");
                }

                // Validate basic parameters
                if (config.num_layers <= 0 || config.n_kv_heads <= 0 || config.head_dim <= 0)
                {
                    LOG_ERROR("[KernelFactory] createROCmKVCache: invalid parameters");
                    throw std::runtime_error("KernelFactory::createROCmKVCache: invalid parameters");
                }

                int rocm_device = config.device.rocm_ordinal();

                // Handle sharding - ROCm supports it via factory function
                if (config.is_sharded())
                {
                    LOG_DEBUG("[KernelFactory] Creating sharded ROCm Ring KVCache: "
                              << "precision=" << llaminar2::activationPrecisionToString(config.precision)
                              << ", device=ROCm:" << rocm_device
                              << ", layers=" << config.num_layers
                              << ", n_kv_heads=" << config.n_kv_heads
                              << ", local_n_kv_heads=" << config.local_n_kv_heads
                              << ", kv_head_start=" << config.kv_head_start
                              << ", head_dim=" << config.head_dim
                              << ", max_seq_len=" << config.max_seq_len);

                    return llaminar2::createShardedROCmRingKVCache(
                        config.precision,
                        config.num_layers,
                        config.batch_size,
                        config.max_seq_len,
                        config.n_kv_heads,
                        config.local_n_kv_heads,
                        config.kv_head_start,
                        config.head_dim,
                        rocm_device);
                }
                else
                {
                    LOG_DEBUG("[KernelFactory] Creating ROCm Ring KVCache: "
                              << "precision=" << llaminar2::activationPrecisionToString(config.precision)
                              << ", device=ROCm:" << rocm_device
                              << ", layers=" << config.num_layers
                              << ", n_kv_heads=" << config.n_kv_heads
                              << ", head_dim=" << config.head_dim
                              << ", max_seq_len=" << config.max_seq_len);

                    return llaminar2::createROCmRingKVCache(
                        config.precision,
                        config.num_layers,
                        config.batch_size,
                        config.max_seq_len,
                        config.n_kv_heads,
                        config.head_dim,
                        rocm_device);
                }
            }
#endif // HAVE_ROCM

            // ==========================================================================
            // Hybrid KV Cache Factory
            // ==========================================================================

            std::unique_ptr<llaminar2::IKVCache> KernelFactory::createHybridKVCache(const KVCacheConfig &config)
            {
                if (!config.hybrid_config)
                {
                    throw std::runtime_error("KernelFactory::createHybridKVCache: hybrid_config is null");
                }

                const auto &hc = *config.hybrid_config;
                std::unique_ptr<llaminar2::IKVCache> cache;

                if (config.device.is_cpu())
                {
                    if (!config.mpi_ctx)
                    {
                        throw std::runtime_error("KernelFactory::createHybridKVCache: mpi_ctx is required for CPU");
                    }

                    LOG_DEBUG("[KernelFactory] Creating CPU Hybrid KVCache: "
                              << "precision=" << llaminar2::activationPrecisionToString(config.precision)
                              << ", total_layers=" << config.num_layers
                              << ", kv_layers=" << hc.countKVLayers()
                              << ", n_kv_heads=" << config.n_kv_heads
                              << ", head_dim=" << config.head_dim);

                    if (config.is_sharded())
                    {
                        switch (config.precision)
                        {
                        case llaminar2::ActivationPrecision::FP32:
                            cache = std::make_unique<llaminar2::CPUHybridRingKVCache<llaminar2::ActivationPrecision::FP32>>(
                                hc, *config.mpi_ctx, config.num_layers, config.batch_size,
                                config.max_seq_len, config.n_kv_heads, config.local_n_kv_heads,
                                config.kv_head_start, config.head_dim, config.device, config.layout_mode);
                            break;
                        case llaminar2::ActivationPrecision::BF16:
                            cache = std::make_unique<llaminar2::CPUHybridRingKVCache<llaminar2::ActivationPrecision::BF16>>(
                                hc, *config.mpi_ctx, config.num_layers, config.batch_size,
                                config.max_seq_len, config.n_kv_heads, config.local_n_kv_heads,
                                config.kv_head_start, config.head_dim, config.device, config.layout_mode);
                            break;
                        case llaminar2::ActivationPrecision::FP16:
                            cache = std::make_unique<llaminar2::CPUHybridRingKVCache<llaminar2::ActivationPrecision::FP16>>(
                                hc, *config.mpi_ctx, config.num_layers, config.batch_size,
                                config.max_seq_len, config.n_kv_heads, config.local_n_kv_heads,
                                config.kv_head_start, config.head_dim, config.device, config.layout_mode);
                            break;
                        case llaminar2::ActivationPrecision::Q8_1:
                            cache = std::make_unique<llaminar2::CPUHybridRingKVCache<llaminar2::ActivationPrecision::Q8_1>>(
                                hc, *config.mpi_ctx, config.num_layers, config.batch_size,
                                config.max_seq_len, config.n_kv_heads, config.local_n_kv_heads,
                                config.kv_head_start, config.head_dim, config.device, config.layout_mode);
                            break;
                        case llaminar2::ActivationPrecision::Q16_1:
                            cache = std::make_unique<llaminar2::CPUHybridRingKVCache<llaminar2::ActivationPrecision::Q16_1>>(
                                hc, *config.mpi_ctx, config.num_layers, config.batch_size,
                                config.max_seq_len, config.n_kv_heads, config.local_n_kv_heads,
                                config.kv_head_start, config.head_dim, config.device, config.layout_mode);
                            break;
                        default:
                            throw std::runtime_error("KernelFactory::createHybridKVCache: Unsupported CPU precision");
                        }
                    }
                    else
                    {
                        switch (config.precision)
                        {
                        case llaminar2::ActivationPrecision::FP32:
                            cache = std::make_unique<llaminar2::CPUHybridRingKVCacheFP32>(
                                hc, *config.mpi_ctx, config.num_layers, config.batch_size,
                                config.max_seq_len, config.n_kv_heads, config.head_dim,
                                config.device, config.layout_mode);
                            break;
                        case llaminar2::ActivationPrecision::BF16:
                            cache = std::make_unique<llaminar2::CPUHybridRingKVCacheBF16>(
                                hc, *config.mpi_ctx, config.num_layers, config.batch_size,
                                config.max_seq_len, config.n_kv_heads, config.head_dim,
                                config.device, config.layout_mode);
                            break;
                        case llaminar2::ActivationPrecision::FP16:
                            cache = std::make_unique<llaminar2::CPUHybridRingKVCacheFP16>(
                                hc, *config.mpi_ctx, config.num_layers, config.batch_size,
                                config.max_seq_len, config.n_kv_heads, config.head_dim,
                                config.device, config.layout_mode);
                            break;
                        case llaminar2::ActivationPrecision::Q8_1:
                            cache = std::make_unique<llaminar2::CPUHybridRingKVCacheQ8_1>(
                                hc, *config.mpi_ctx, config.num_layers, config.batch_size,
                                config.max_seq_len, config.n_kv_heads, config.head_dim,
                                config.device, config.layout_mode);
                            break;
                        case llaminar2::ActivationPrecision::Q16_1:
                            cache = std::make_unique<llaminar2::CPUHybridRingKVCacheQ16_1>(
                                hc, *config.mpi_ctx, config.num_layers, config.batch_size,
                                config.max_seq_len, config.n_kv_heads, config.head_dim,
                                config.device, config.layout_mode);
                            break;
                        default:
                            throw std::runtime_error("KernelFactory::createHybridKVCache: Unsupported CPU precision");
                        }
                    }
                }
#ifdef HAVE_CUDA
                else if (config.device.is_cuda())
                {
                    const int cuda_device = config.device.cuda_ordinal();

                    LOG_DEBUG("[KernelFactory] Creating CUDA Hybrid KVCache: "
                              << "precision=" << llaminar2::activationPrecisionToString(config.precision)
                              << ", device=CUDA:" << cuda_device
                              << ", total_layers=" << config.num_layers
                              << ", kv_layers=" << hc.countKVLayers()
                              << ", n_kv_heads=" << config.n_kv_heads
                              << ", head_dim=" << config.head_dim);

                    if (config.is_sharded())
                    {
                        switch (config.precision)
                        {
                        case llaminar2::ActivationPrecision::FP32:
                            cache = std::make_unique<llaminar2::CUDAHybridRingKVCacheFP32>(
                                hc, config.num_layers, config.batch_size, config.max_seq_len,
                                config.n_kv_heads, config.local_n_kv_heads, config.kv_head_start,
                                config.head_dim, cuda_device);
                            break;
                        case llaminar2::ActivationPrecision::FP16:
                            cache = std::make_unique<llaminar2::CUDAHybridRingKVCacheFP16>(
                                hc, config.num_layers, config.batch_size, config.max_seq_len,
                                config.n_kv_heads, config.local_n_kv_heads, config.kv_head_start,
                                config.head_dim, cuda_device);
                            break;
                        case llaminar2::ActivationPrecision::BF16:
                            cache = std::make_unique<llaminar2::CUDAHybridRingKVCacheBF16>(
                                hc, config.num_layers, config.batch_size, config.max_seq_len,
                                config.n_kv_heads, config.local_n_kv_heads, config.kv_head_start,
                                config.head_dim, cuda_device);
                            break;
                        case llaminar2::ActivationPrecision::Q8_1:
                            cache = std::make_unique<llaminar2::CUDAHybridRingKVCacheQ8_1>(
                                hc, config.num_layers, config.batch_size, config.max_seq_len,
                                config.n_kv_heads, config.local_n_kv_heads, config.kv_head_start,
                                config.head_dim, cuda_device);
                            break;
                        default:
                            throw std::runtime_error("KernelFactory::createHybridKVCache: Unsupported CUDA precision");
                        }
                    }
                    else
                    {
                        switch (config.precision)
                        {
                        case llaminar2::ActivationPrecision::FP32:
                            cache = std::make_unique<llaminar2::CUDAHybridRingKVCacheFP32>(
                                hc, config.num_layers, config.batch_size, config.max_seq_len,
                                config.n_kv_heads, config.head_dim, cuda_device);
                            break;
                        case llaminar2::ActivationPrecision::FP16:
                            cache = std::make_unique<llaminar2::CUDAHybridRingKVCacheFP16>(
                                hc, config.num_layers, config.batch_size, config.max_seq_len,
                                config.n_kv_heads, config.head_dim, cuda_device);
                            break;
                        case llaminar2::ActivationPrecision::BF16:
                            cache = std::make_unique<llaminar2::CUDAHybridRingKVCacheBF16>(
                                hc, config.num_layers, config.batch_size, config.max_seq_len,
                                config.n_kv_heads, config.head_dim, cuda_device);
                            break;
                        case llaminar2::ActivationPrecision::Q8_1:
                            cache = std::make_unique<llaminar2::CUDAHybridRingKVCacheQ8_1>(
                                hc, config.num_layers, config.batch_size, config.max_seq_len,
                                config.n_kv_heads, config.head_dim, cuda_device);
                            break;
                        default:
                            throw std::runtime_error("KernelFactory::createHybridKVCache: Unsupported CUDA precision");
                        }
                    }
                }
#endif // HAVE_CUDA
#ifdef HAVE_ROCM
                else if (config.device.is_rocm())
                {
                    const int rocm_device = config.device.rocm_ordinal();

                    LOG_DEBUG("[KernelFactory] Creating ROCm Hybrid KVCache: "
                              << "precision=" << llaminar2::activationPrecisionToString(config.precision)
                              << ", device=ROCm:" << rocm_device
                              << ", total_layers=" << config.num_layers
                              << ", kv_layers=" << hc.countKVLayers()
                              << ", n_kv_heads=" << config.n_kv_heads
                              << ", head_dim=" << config.head_dim);

                    if (config.is_sharded())
                    {
                        cache = llaminar2::createShardedROCmHybridRingKVCache(
                            hc, config.precision, config.num_layers, config.batch_size,
                            config.max_seq_len, config.n_kv_heads, config.local_n_kv_heads,
                            config.kv_head_start, config.head_dim, rocm_device);
                    }
                    else
                    {
                        cache = llaminar2::createROCmHybridRingKVCache(
                            hc, config.precision, config.num_layers, config.batch_size,
                            config.max_seq_len, config.n_kv_heads, config.head_dim, rocm_device);
                    }
                }
#endif // HAVE_ROCM
                else
                {
                    throw std::runtime_error("KernelFactory::createHybridKVCache: Unsupported device type: " +
                                             config.device.to_string());
                }

                // Post-creation: initialize GDN kernel instances in each GDN layer's state.
                // This must happen after cache construction since initHybrid() only allocates
                // host-side state buffers — kernel creation requires KernelFactory access.
                auto *hybrid = dynamic_cast<llaminar2::IHybridKVCache *>(cache.get());
                if (hybrid)
                {
                    auto dev_type = getDeviceType(config.device);
                    int dev_ordinal = config.device.toKernelDeviceIndex();

                    for (int i = 0; i < config.num_layers; ++i)
                    {
                        // Use global layer index for PP-aware GDN state lookup.
                        // For PP stage 2 with first_layer_index=12, local layer 0
                        // corresponds to global layer 12.
                        int global_layer = i + config.first_layer_index;
                        auto *gdn_state = hybrid->getGDNState(global_layer);
                        if (!gdn_state)
                            continue; // FA layer

                        gdn_state->conv_kernel = createShortConvolution(dev_type, dev_ordinal);
                        gdn_state->rec_kernel = createGatedDeltaNet(dev_type, dev_ordinal);

                        // Allocate device-resident state buffers for GPU kernels
                        // (no-op for CPU via virtual dispatch)
                        gdn_state->conv_kernel->allocateGPUState(
                            static_cast<int>(gdn_state->conv_state.size()));
                        // In-place prefill scratch is supplied by ShortConv1dStage
                        // through DeviceWorkspaceManager. Keeping it out of the
                        // per-layer KV-cache state avoids one persistent
                        // max_seq_len * qkv_dim allocation for every GDN layer.
                        gdn_state->rec_kernel->allocateGPUState(
                            static_cast<int>(gdn_state->recurrence_state.size()));
                    }

                    LOG_DEBUG("[KernelFactory] Initialized GDN kernels for "
                              << hybrid->gdnLayerCount() << " GDN layers on "
                              << config.device.to_string());
                }

                return cache;
            }

            // ==========================================================================
            // Activation/Weight Type Compatibility
            // ==========================================================================

            bool KernelFactory::isFloatingPointType(llaminar2::TensorType type)
            {
                switch (type)
                {
                case llaminar2::TensorType::FP32:
                case llaminar2::TensorType::FP16:
                case llaminar2::TensorType::BF16:
                    return true;
                default:
                    return false;
                }
            }

            bool KernelFactory::isQuantizedType(llaminar2::TensorType type)
            {
                switch (type)
                {
                // 8-bit formats
                case llaminar2::TensorType::Q8_0:
                case llaminar2::TensorType::Q8_1:
                case llaminar2::TensorType::Q8_K:
                // 4-bit formats
                case llaminar2::TensorType::Q4_0:
                case llaminar2::TensorType::Q4_1:
                case llaminar2::TensorType::Q4_K:
                case llaminar2::TensorType::IQ4_NL:
                case llaminar2::TensorType::IQ4_XS:
                // 5-bit formats
                case llaminar2::TensorType::Q5_0:
                case llaminar2::TensorType::Q5_1:
                case llaminar2::TensorType::Q5_K:
                // 6-bit format
                case llaminar2::TensorType::Q6_K:
                // K-quant formats
                case llaminar2::TensorType::Q2_K:
                case llaminar2::TensorType::Q3_K:
                // I-quant formats
                case llaminar2::TensorType::IQ1_M:
                case llaminar2::TensorType::IQ1_S:
                case llaminar2::TensorType::IQ2_S:
                case llaminar2::TensorType::IQ2_XS:
                case llaminar2::TensorType::IQ2_XXS:
                case llaminar2::TensorType::IQ3_S:
                case llaminar2::TensorType::IQ3_XXS:
                    return true;
                default:
                    return false;
                }
            }

            bool KernelFactory::isActivationWeightCompatible(
                llaminar2::TensorType activation_type,
                llaminar2::TensorType weight_type)
            {
                // FP32, FP16, BF16 activations work with ANY weight type
                // (FloatingPointGemmKernel dequantizes weights on-the-fly)
                if (isFloatingPointType(activation_type))
                {
                    return true;
                }

                // Q8_1 activations work with quantized weights
                // (NativeVNNI/CUDA kernels use INT8×INT8 dot products)
                if (activation_type == llaminar2::TensorType::Q8_1)
                {
                    // Q8_1 activations can work with any quantized weight type
                    // (the kernel repacks weights to INT8 VNNI format)
                    return isQuantizedType(weight_type);
                }

                // Other activation types (INT8, INT32, etc.) are not supported
                // in GEMM operations currently
                return false;
            }

            std::string KernelFactory::getCompatibilityErrorMessage(
                llaminar2::TensorType activation_type,
                llaminar2::TensorType weight_type)
            {
                const char *act_name = llaminar2::tensorTypeName(activation_type);
                const char *wgt_name = llaminar2::tensorTypeName(weight_type);

                if (activation_type == llaminar2::TensorType::Q8_1 && isFloatingPointType(weight_type))
                {
                    return std::string("Q8_1 activation precision requires quantized weights, but got ") +
                           wgt_name + " weights. The GEMM kernel uses INT8×INT8 VNNI " +
                           "instructions which cannot process floating-point weights. Either:\n" +
                           "  1. Use FP32 activation precision (slower but compatible with any weights)\n" +
                           "  2. Use a quantized model (Q8_0, Q4_0, IQ4_NL, etc.)";
                }

                if (!isFloatingPointType(activation_type) && activation_type != llaminar2::TensorType::Q8_1)
                {
                    return std::string("Unsupported activation type: ") + act_name +
                           ". Only FP32, FP16, BF16, and Q8_1 activation precisions are supported.";
                }

                return std::string("Activation type ") + act_name +
                       " is not compatible with weight type " + wgt_name;
            }

        } // namespace kernels
    } // namespace v2
} // namespace llaminar
