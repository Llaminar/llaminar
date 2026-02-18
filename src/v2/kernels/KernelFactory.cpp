/**
 * @file KernelFactory.cpp
 * @brief Implementation of centralized kernel dispatch factory
 * @author David Sanftenberg
 */

#include "KernelFactory.h"
#include "cpu/gemm_v4/QuantisedGemmKernel.h"
#include "cpu/gemm_v4/FloatingPointGemmKernel.h"
#include "cpu/gemm_v4/FusedGEMM.h"
#include "../tensors/TensorSlice.h"
#include "cpu/ops/CPURoPEKernelT.h"
#include "cpu/ops/CPUSwiGLUKernelT.h"
#include "cpu/ops/CPUSoftmaxKernelT.h"
#include "cpu/ops/CPURMSNormKernelT.h"
#include "cpu/ops/CPUResidualAddKernelT.h"
#include "cpu/ops/CPUEmbeddingKernelT.h"
#include "cpu/attention/CPUAttentionKernelT.h"
#include "cpu/attention/CPUFlashAttentionKernelT.h"
#include <unordered_set>

// KVCache includes
#include "cpu/CPURingKVCache.h"
#ifdef HAVE_CUDA
#include "cuda/kvcache/CUDARingKVCache.h"
#endif

#include "../tensors/Tensors.h"
#include "../tensors/TensorKernels.h"
#include "../execution/local_execution/device/DeviceContext.h"
#include "../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../interfaces/IWorkspaceConsumer.h"
#include "../backends/ComputeBackend.h"
#include "../utils/Logger.h"

// CUDA kernel classes
#ifdef HAVE_CUDA
#include "cuda/CUDAQuantisedGemmKernel.h"             // Quantized tensors via CUTLASS INT8
#include "cuda/CUDAFloatingPointGemmKernel.h"         // FP32/FP16/BF16 via cuBLAS
#include "cuda/ops/CUDARMSNormKernelT.h"              // RMSNorm FP32/FP16/BF16
#include "cuda/ops/CUDARoPEKernelT.h"                 // RoPE FP32/FP16/BF16
#include "cuda/ops/CUDASwiGLUKernelT.h"               // SwiGLU FP32/FP16/BF16
#include "cuda/ops/CUDAResidualAddKernelT.h"          // ResidualAdd FP32/FP16/BF16
#include "cuda/ops/CUDAEmbeddingKernelT.h"            // Embedding FP32
#include "cuda/attention/CUDAFlashAttentionKernelT.h" // Flash Attention
#endif

// ROCm kernel classes
#ifdef HAVE_ROCM
#include "rocm/ROCmFloatingPointGemmKernel.h"         // FP32/FP16/BF16 via hipBLAS
#include "rocm/ROCmQuantisedGemmKernel.h"             // Quantized tensors via CK INT8/FP16
#include "rocm/kvcache/ROCmRingKVCacheFactory.h"      // ROCm Ring Buffer KV Cache factory
#include "rocm/ops/ROCmEmbeddingKernelT.h"            // Embedding FP32/BF16/FP16/Q8_1
#include "rocm/ops/ROCmRMSNormKernelT.h"              // RMSNorm FP32/BF16/FP16
#include "rocm/ops/ROCmRoPEKernelT.h"                 // RoPE FP32
#include "rocm/ops/ROCmSwiGLUKernelT.h"               // SwiGLU FP32
#include "rocm/ops/ROCmResidualAddKernelT.h"          // ResidualAdd FP32
#include "rocm/attention/ROCmFlashAttentionKernelT.h" // Flash Attention
#endif

namespace llaminar
{
    namespace v2
    {
        namespace kernels
        {

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
                              "Use getOrCreatePreparedGemmWeights(tensor, DeviceId) and getOrCreateGemmEngine(prepared) to specify the target device explicitly.");
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
                              "Use getOrCreatePreparedGemmWeights(tensor, DeviceId) and getOrCreateGemmEngine(prepared) to specify the target device explicitly.");
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

                    // Original raw pointer interface (legacy)
                    bool apply(
                        float *data, float *output,
                        const int *pos_ids,
                        int batch_size, int seq_len, int head_dim, int num_heads,
                        float theta_base, bool interleaved,
                        const llaminar2::MPIContext *mpi_ctx,
                        int device_idx) override
                    {
                        (void)interleaved; // Not used in typed kernel
                        (void)mpi_ctx;     // Not used in typed kernel
                        // Legacy interface treats data as Q and output as K
                        // Treat batch_size * seq_len as total seq_len
                        return kernel_.apply_typed(data, output, pos_ids,
                                                   batch_size * seq_len, num_heads, num_heads, head_dim,
                                                   theta_base, device_idx);
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
                        const llaminar2::MPIContext *mpi_ctx,
                        int device_idx,
                        int pos_offset = 0) override
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
                                                   rope_theta, device_idx);
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

                    // Original raw pointer interface (legacy) - not supported for Q8_1
                    bool apply(
                        float *data, float *output,
                        const int *pos_ids,
                        int batch_size, int seq_len, int head_dim, int num_heads,
                        float theta_base, bool interleaved,
                        const llaminar2::MPIContext *mpi_ctx,
                        int device_idx) override
                    {
                        (void)data;
                        (void)output;
                        (void)pos_ids;
                        (void)batch_size;
                        (void)seq_len;
                        (void)head_dim;
                        (void)num_heads;
                        (void)theta_base;
                        (void)interleaved;
                        (void)mpi_ctx;
                        (void)device_idx;
                        LOG_ERROR("[Q8_1RoPEKernelAdapter] Raw float* interface not supported for Q8_1");
                        return false;
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
                        const llaminar2::MPIContext *mpi_ctx,
                        int device_idx,
                        int pos_offset = 0) override
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
                                                   rope_theta, device_idx);
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
                        const llaminar2::MPIContext *mpi_ctx,
                        int device_idx) override
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
                        const llaminar2::MPIContext *mpi_ctx,
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
                        const llaminar2::MPIContext *mpi_ctx,
                        int device_idx) override
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
                        const llaminar2::MPIContext *mpi_ctx,
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
                const llaminar2::TensorBase *tensor, DeviceType dev_type)
            {
                // Two-way dispatch based on interface:
                // 1. IINT8Unpackable → QuantisedGemmKernel (quantized weights)
                // 2. Otherwise → FloatingPointGemmKernel (FP32/FP16/BF16)

                const auto *quantized = dynamic_cast<const llaminar2::IINT8Unpackable *>(tensor);

                if (quantized)
                {
                    // Quantized tensor - use INT8 GEMM kernel
                    switch (dev_type)
                    {
                    case DeviceType::CPU:
                    {
                        auto *packed = ensurePackedWeightsInTensorCache(tensor);
                        return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                    }
#ifdef HAVE_CUDA
                    case DeviceType::CUDA:
                    {
                        auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                        int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                        return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
                    }
#endif
#ifdef HAVE_ROCM
                    case DeviceType::ROCm:
                    {
                        auto *packed = ensureROCmPackedWeightsInTensorCache(tensor);
                        int rocm_device_id = getROCmDeviceIdForKernel(tensor);
                        return std::make_unique<llaminar2::rocm::ROCmQuantisedGemmKernel>(packed, rocm_device_id);
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
                        return std::make_unique<llaminar2::gemm_v4::FloatingPointGemmKernel>(tensor);
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
                const llaminar2::IQ4_NLTensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] IQ4_NL GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::IQ4_NLTensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return new llaminar2::gemm_v4::QuantisedGemmKernel(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return new llaminar2::cuda::CUDAQuantisedGemmKernel(packed, cuda_device_id);
                }
#endif

                case DeviceType::ROCm:
                case DeviceType::Vulkan:
                case DeviceType::Metal:
                default:
                    throwUnsupportedBackend(dev_type, "IQ4_NL");
                }
            }

            // ==========================================================================
            // Q4_0 Tensor GEMM
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
                const llaminar2::Q4_0Tensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] Q4_0 GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
                }
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                {
                    LOG_DEBUG("[KernelFactory] Q4_0 GEMM: Using ROCmQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureROCmPackedWeightsInTensorCache(tensor);
                    int rocm_device_id = getROCmDeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::rocm::ROCmQuantisedGemmKernel>(packed, rocm_device_id);
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
                const llaminar2::Q4_1Tensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] Q4_1 GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::Q5_0Tensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] Q5_0 GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::Q5_1Tensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] Q5_1 GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::Q6_KTensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] Q6_K GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::Q8_0Tensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] Q8_0 GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] Q8_1 GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::Q2_KTensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] Q2_K GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::Q3_KTensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] Q3_K GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::Q4_KTensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] Q4_K GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::Q5_KTensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] Q5_K GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::Q8_KTensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] Q8_K GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::IQ1_MTensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] IQ1_M GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::IQ1_STensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] IQ1_S GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::IQ2_STensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] IQ2_S GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::IQ2_XSTensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] IQ2_XS GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::IQ2_XXSTensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] IQ2_XXS GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::IQ3_STensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] IQ3_S GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::IQ3_XXSTensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] IQ3_XXS GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::IQ4_XSTensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                {
                    auto *packed = ensurePackedWeightsInTensorCache(tensor);
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                {
                    LOG_DEBUG("[KernelFactory] IQ4_XS GEMM: Using CUDAQuantisedGemmKernel (pre-packed)");
                    auto *packed = ensureCUDAPackedWeightsInTensorCache(tensor);
                    const auto &dm = llaminar2::DeviceManager::instance();
                    int cuda_device_id = getCUDADeviceIdForKernel(tensor);
                    return std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(packed, cuda_device_id);
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
                const llaminar2::FP32Tensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::gemm_v4::FloatingPointGemmKernel>(tensor);

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
                const llaminar2::FP16Tensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::gemm_v4::FloatingPointGemmKernel>(tensor);

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
                const llaminar2::BF16Tensor *tensor, DeviceType dev_type)
            {
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::gemm_v4::FloatingPointGemmKernel>(tensor);

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
                const llaminar2::FP32Tensor *tensor, DeviceType dev_type)
            {
                (void)tensor; // RoPE kernels don't need tensor state for creation
                switch (dev_type)
                {
                case DeviceType::CPU:
                    // Return typed kernel directly - it implements ITensorRoPE
                    return std::make_unique<llaminar2::CPURoPEKernelT<llaminar2::ActivationPrecision::FP32>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDARoPEKernelT<llaminar2::ActivationPrecision::FP32>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmRoPEKernelT<llaminar2::ActivationPrecision::FP32>>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "RoPE", "FP32");
                }
            }

            std::unique_ptr<llaminar2::ITensorRoPE> KernelFactory::createRoPE(
                const llaminar2::BF16Tensor *tensor, DeviceType dev_type)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    // Return typed kernel directly - it implements ITensorRoPE
                    return std::make_unique<llaminar2::CPURoPEKernelT<llaminar2::ActivationPrecision::BF16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDARoPEKernelT<llaminar2::ActivationPrecision::BF16>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmRoPEKernelT<llaminar2::ActivationPrecision::BF16>>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "RoPE", "BF16");
                }
            }

            std::unique_ptr<llaminar2::ITensorRoPE> KernelFactory::createRoPE(
                const llaminar2::FP16Tensor *tensor, DeviceType dev_type)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    // Return typed kernel directly - it implements ITensorRoPE
                    return std::make_unique<llaminar2::CPURoPEKernelT<llaminar2::ActivationPrecision::FP16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDARoPEKernelT<llaminar2::ActivationPrecision::FP16>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmRoPEKernelT<llaminar2::ActivationPrecision::FP16>>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "RoPE", "FP16");
                }
            }

            std::unique_ptr<llaminar2::ITensorRoPE> KernelFactory::createRoPE(
                const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type)
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
                const llaminar2::FP32Tensor *tensor, DeviceType dev_type)
            {
                (void)tensor; // SwiGLU kernels don't need tensor state for creation
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUSwiGLUKernelT<llaminar2::ActivationPrecision::FP32>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDASwiGLUKernelT<llaminar2::ActivationPrecision::FP32>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmSwiGLUKernelT<llaminar2::ActivationPrecision::FP32>>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "SwiGLU", "FP32");
                }
            }

            std::unique_ptr<llaminar2::ITensorSwiGLU> KernelFactory::createSwiGLU(
                const llaminar2::BF16Tensor *tensor, DeviceType dev_type)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUSwiGLUKernelT<llaminar2::ActivationPrecision::BF16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDASwiGLUKernelT<llaminar2::ActivationPrecision::BF16>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmSwiGLUKernelT<llaminar2::ActivationPrecision::BF16>>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "SwiGLU", "BF16");
                }
            }

            std::unique_ptr<llaminar2::ITensorSwiGLU> KernelFactory::createSwiGLU(
                const llaminar2::FP16Tensor *tensor, DeviceType dev_type)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUSwiGLUKernelT<llaminar2::ActivationPrecision::FP16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDASwiGLUKernelT<llaminar2::ActivationPrecision::FP16>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmSwiGLUKernelT<llaminar2::ActivationPrecision::FP16>>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "SwiGLU", "FP16");
                }
            }

            std::unique_ptr<llaminar2::ITensorSwiGLU> KernelFactory::createSwiGLU(
                const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type)
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
                const llaminar2::FP32Tensor *tensor, DeviceType dev_type)
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
                const llaminar2::BF16Tensor *tensor, DeviceType dev_type)
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
                const llaminar2::FP16Tensor *tensor, DeviceType dev_type)
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
                const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type)
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
                const llaminar2::FP32Tensor *tensor, DeviceType dev_type)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUResidualAddKernelT<llaminar2::ActivationPrecision::FP32>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDAResidualAddKernelT<llaminar2::ActivationPrecision::FP32>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmResidualAddKernelT<llaminar2::ActivationPrecision::FP32>>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "ResidualAdd", "FP32");
                }
            }

            std::unique_ptr<llaminar2::ITensorResidualAdd> KernelFactory::createResidualAdd(
                const llaminar2::BF16Tensor *tensor, DeviceType dev_type)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUResidualAddKernelT<llaminar2::ActivationPrecision::BF16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDAResidualAddKernelT<llaminar2::ActivationPrecision::BF16>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmResidualAddKernelT<llaminar2::ActivationPrecision::BF16>>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "ResidualAdd", "BF16");
                }
            }

            std::unique_ptr<llaminar2::ITensorResidualAdd> KernelFactory::createResidualAdd(
                const llaminar2::FP16Tensor *tensor, DeviceType dev_type)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUResidualAddKernelT<llaminar2::ActivationPrecision::FP16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDAResidualAddKernelT<llaminar2::ActivationPrecision::FP16>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmResidualAddKernelT<llaminar2::ActivationPrecision::FP16>>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "ResidualAdd", "FP16");
                }
            }

            // ==========================================================================
            // RMSNorm Kernel Creation - Device-aware dispatch
            // NO CPU FALLBACK: If GPU requested but not implemented, throws
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorRMSNorm> KernelFactory::createRMSNorm(
                const llaminar2::FP32Tensor *tensor, DeviceType dev_type)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPURMSNormKernelT<llaminar2::ActivationPrecision::FP32>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDARMSNormKernelT<llaminar2::ActivationPrecision::FP32>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmRMSNormKernelT<llaminar2::ActivationPrecision::FP32>>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "RMSNorm", "FP32");
                }
            }

            std::unique_ptr<llaminar2::ITensorRMSNorm> KernelFactory::createRMSNorm(
                const llaminar2::BF16Tensor *tensor, DeviceType dev_type)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPURMSNormKernelT<llaminar2::ActivationPrecision::BF16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDARMSNormKernelT<llaminar2::ActivationPrecision::BF16>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmRMSNormKernelT<llaminar2::ActivationPrecision::BF16>>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "RMSNorm", "BF16");
                }
            }

            std::unique_ptr<llaminar2::ITensorRMSNorm> KernelFactory::createRMSNorm(
                const llaminar2::FP16Tensor *tensor, DeviceType dev_type)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPURMSNormKernelT<llaminar2::ActivationPrecision::FP16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDARMSNormKernelT<llaminar2::ActivationPrecision::FP16>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmRMSNormKernelT<llaminar2::ActivationPrecision::FP16>>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "RMSNorm", "FP16");
                }
            }

            std::unique_ptr<llaminar2::ITensorRMSNorm> KernelFactory::createRMSNorm(
                const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type)
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
                const llaminar2::Q16_1Tensor *tensor, DeviceType dev_type)
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
                const llaminar2::TensorBase *tensor, DeviceType dev_type)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::createSoftmax: null tensor");
                }

                switch (tensor->native_type())
                {
                case llaminar2::TensorType::FP32:
                    return createSoftmax(static_cast<const llaminar2::FP32Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::BF16:
                    return createSoftmax(static_cast<const llaminar2::BF16Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::FP16:
                    return createSoftmax(static_cast<const llaminar2::FP16Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::Q8_1:
                    return createSoftmax(static_cast<const llaminar2::Q8_1Tensor *>(tensor), dev_type);
                default:
                    throw std::runtime_error(
                        "KernelFactory::createSoftmax: unsupported tensor type " +
                        std::string(tensor->dtype_name()));
                }
            }

            std::unique_ptr<llaminar2::ITensorRMSNorm> KernelFactory::createRMSNorm(
                const llaminar2::TensorBase *tensor, DeviceType dev_type)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::createRMSNorm: null tensor");
                }

                switch (tensor->native_type())
                {
                case llaminar2::TensorType::FP32:
                    return createRMSNorm(static_cast<const llaminar2::FP32Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::BF16:
                    return createRMSNorm(static_cast<const llaminar2::BF16Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::FP16:
                    return createRMSNorm(static_cast<const llaminar2::FP16Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::Q8_1:
                    return createRMSNorm(static_cast<const llaminar2::Q8_1Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::Q16_1:
                    return createRMSNorm(static_cast<const llaminar2::Q16_1Tensor *>(tensor), dev_type);
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

                auto kernel = createRoPE(tensor, getDeviceType(target_device));
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
                    LOG_DEBUG("[KernelFactory][Registry] hit kind=RMSNORM dev=" << static_cast<int>(target_device.type)
                                                                                  << ":" << target_device.ordinal
                                                                                  << " variant=" << static_cast<int>(tensor->native_type())
                                                                                  << " ptr=" << reg_it->second.get());
                    return static_cast<llaminar2::ITensorRMSNorm *>(reg_it->second.get());
                }

                auto it = rmsnorm_cache_.find(key);
                if (it != rmsnorm_cache_.end())
                {
                    auto *raw_ptr = it->second.get();
                    device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
                    LOG_DEBUG("[KernelFactory][Registry] backfill kind=RMSNORM dev=" << static_cast<int>(target_device.type)
                                                                                       << ":" << target_device.ordinal
                                                                                       << " variant=" << static_cast<int>(tensor->native_type())
                                                                                       << " ptr=" << raw_ptr);
                    return raw_ptr;
                }

                auto kernel = createRMSNorm(tensor, getDeviceType(target_device));
                auto *raw_ptr = kernel.get();
                rmsnorm_cache_[key] = std::move(kernel);
                device_kernel_registry_[registry_key] = std::shared_ptr<void>(raw_ptr, [](void *) {});
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

                auto kernel = createSwiGLU(tensor, getDeviceType(target_device));
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

                auto kernel = createSoftmax(tensor, getDeviceType(target_device));
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

                auto kernel = createResidualAdd(tensor, getDeviceType(target_device));
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

                auto kernel = createAttention(tensor, getDeviceType(target_device));
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
                    kernel = createEmbedding(static_cast<const llaminar2::FP32Tensor *>(tensor), dev_type);
                    break;
                case llaminar2::TensorType::BF16:
                    kernel = createEmbedding(static_cast<const llaminar2::BF16Tensor *>(tensor), dev_type);
                    break;
                case llaminar2::TensorType::FP16:
                    kernel = createEmbedding(static_cast<const llaminar2::FP16Tensor *>(tensor), dev_type);
                    break;
                case llaminar2::TensorType::Q8_1:
                    kernel = createEmbedding(static_cast<const llaminar2::Q8_1Tensor *>(tensor), dev_type);
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

            std::unique_ptr<llaminar2::ITensorRoPE> KernelFactory::createRoPE(
                const llaminar2::TensorBase *tensor, DeviceType dev_type)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::createRoPE: null tensor");
                }

                switch (tensor->native_type())
                {
                case llaminar2::TensorType::FP32:
                    return createRoPE(static_cast<const llaminar2::FP32Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::BF16:
                    return createRoPE(static_cast<const llaminar2::BF16Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::FP16:
                    return createRoPE(static_cast<const llaminar2::FP16Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::Q8_1:
                    return createRoPE(static_cast<const llaminar2::Q8_1Tensor *>(tensor), dev_type);
                default:
                    throw std::runtime_error(
                        "KernelFactory::createRoPE: unsupported tensor type " +
                        std::string(tensor->dtype_name()));
                }
            }

            std::unique_ptr<llaminar2::ITensorSwiGLU> KernelFactory::createSwiGLU(
                const llaminar2::TensorBase *tensor, DeviceType dev_type)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::createSwiGLU: null tensor");
                }

                switch (tensor->native_type())
                {
                case llaminar2::TensorType::FP32:
                    return createSwiGLU(static_cast<const llaminar2::FP32Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::BF16:
                    return createSwiGLU(static_cast<const llaminar2::BF16Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::FP16:
                    return createSwiGLU(static_cast<const llaminar2::FP16Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::Q8_1:
                    return createSwiGLU(static_cast<const llaminar2::Q8_1Tensor *>(tensor), dev_type);
                default:
                    throw std::runtime_error(
                        "KernelFactory::createSwiGLU: unsupported tensor type " +
                        std::string(tensor->dtype_name()));
                }
            }

            std::unique_ptr<llaminar2::ITensorResidualAdd> KernelFactory::createResidualAdd(
                const llaminar2::TensorBase *tensor, DeviceType dev_type)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::createResidualAdd: null tensor");
                }

                switch (tensor->native_type())
                {
                case llaminar2::TensorType::FP32:
                    return createResidualAdd(static_cast<const llaminar2::FP32Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::BF16:
                    return createResidualAdd(static_cast<const llaminar2::BF16Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::FP16:
                    return createResidualAdd(static_cast<const llaminar2::FP16Tensor *>(tensor), dev_type);
                default:
                    throw std::runtime_error(
                        "KernelFactory::createResidualAdd: unsupported tensor type " +
                        std::string(tensor->dtype_name()));
                }
            }

            std::unique_ptr<llaminar2::ITensorAttention> KernelFactory::createAttention(
                const llaminar2::TensorBase *tensor, DeviceType dev_type)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::createAttention: null tensor");
                }

                switch (tensor->native_type())
                {
                case llaminar2::TensorType::FP32:
                    return createAttention(static_cast<const llaminar2::FP32Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::BF16:
                    return createAttention(static_cast<const llaminar2::BF16Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::FP16:
                    return createAttention(static_cast<const llaminar2::FP16Tensor *>(tensor), dev_type);
                case llaminar2::TensorType::Q8_1:
                    return createAttention(static_cast<const llaminar2::Q8_1Tensor *>(tensor), dev_type);
                default:
                    throw std::runtime_error(
                        "KernelFactory::createAttention: unsupported tensor type " +
                        std::string(tensor->dtype_name()));
                }
            }

            std::unique_ptr<llaminar2::ITensorAttention> KernelFactory::createAttention(
                const llaminar2::ITensor *tensor, DeviceType dev_type)
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
                    return createAttention(cpu_tensor, dev_type);
                }

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    switch (tensor->native_type())
                    {
                    case llaminar2::TensorType::FP32:
                        return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::FP32>>();
                    case llaminar2::TensorType::FP16:
                        return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::FP16>>();
                    case llaminar2::TensorType::BF16:
                        return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::BF16>>();
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
                        return std::make_unique<llaminar2::rocm::ROCmFlashAttentionKernelT<llaminar2::ActivationPrecision::FP32>>();
                    case llaminar2::TensorType::FP16:
                        return std::make_unique<llaminar2::rocm::ROCmFlashAttentionKernelT<llaminar2::ActivationPrecision::FP16>>();
                    case llaminar2::TensorType::BF16:
                        return std::make_unique<llaminar2::rocm::ROCmFlashAttentionKernelT<llaminar2::ActivationPrecision::BF16>>();
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
                const llaminar2::FP32Tensor *tensor, DeviceType dev_type)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUFlashAttentionKernelT<llaminar2::ActivationPrecision::FP32>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::FP32>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmFlashAttentionKernelT<llaminar2::ActivationPrecision::FP32>>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Attention", "FP32");
                }
            }

            std::unique_ptr<llaminar2::ITensorAttention> KernelFactory::createAttention(
                const llaminar2::BF16Tensor *tensor, DeviceType dev_type)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUFlashAttentionKernelT<llaminar2::ActivationPrecision::BF16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::BF16>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmFlashAttentionKernelT<llaminar2::ActivationPrecision::BF16>>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Attention", "BF16");
                }
            }

            std::unique_ptr<llaminar2::ITensorAttention> KernelFactory::createAttention(
                const llaminar2::FP16Tensor *tensor, DeviceType dev_type)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUFlashAttentionKernelT<llaminar2::ActivationPrecision::FP16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::FP16>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::rocm::ROCmFlashAttentionKernelT<llaminar2::ActivationPrecision::FP16>>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Attention", "FP16");
                }
            }

            std::unique_ptr<llaminar2::ITensorAttention> KernelFactory::createAttention(
                const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUAttentionKernelT<llaminar2::ActivationPrecision::Q8_1>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    throwUnsupportedKernel(dev_type, "Attention", "Q8_1");
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    throwUnsupportedKernel(dev_type, "Attention", "Q8_1");
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Attention", "Q8_1");
                }
            }

            // ==========================================================================
            // Embedding Kernel Creation - Device-aware dispatch
            // NO CPU FALLBACK: If GPU requested but not implemented, throws
            // ==========================================================================

            std::unique_ptr<llaminar2::ITensorEmbedding> KernelFactory::createEmbedding(
                const llaminar2::FP32Tensor *tensor, DeviceType dev_type)
            {
                (void)tensor;
                switch (dev_type)
                {
                case DeviceType::CPU:
                    return std::make_unique<llaminar2::CPUEmbeddingKernelT<llaminar2::FP32Tensor>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::CUDAEmbeddingKernelT>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    return std::make_unique<llaminar2::ROCmEmbeddingKernelT>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Embedding", "FP32");
                }
            }

            std::unique_ptr<llaminar2::ITensorEmbedding> KernelFactory::createEmbedding(
                const llaminar2::BF16Tensor *tensor, DeviceType dev_type)
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
                    return std::make_unique<llaminar2::ROCmEmbeddingKernelT>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Embedding", "BF16");
                }
            }

            std::unique_ptr<llaminar2::ITensorEmbedding> KernelFactory::createEmbedding(
                const llaminar2::FP16Tensor *tensor, DeviceType dev_type)
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
                    return std::make_unique<llaminar2::ROCmEmbeddingKernelT>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Embedding", "FP16");
                }
            }

            std::unique_ptr<llaminar2::ITensorEmbedding> KernelFactory::createEmbedding(
                const llaminar2::Q8_1Tensor *tensor, DeviceType dev_type)
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
                    return std::make_unique<llaminar2::ROCmEmbeddingKernelT>();
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Embedding", "Q8_1");
                }
            }

            // ==========================================================================
            // Kernel Cache - Static Members
            // ==========================================================================

            std::mutex KernelFactory::cache_mutex_;
            std::unordered_map<KernelFactory::SlicedCacheKey, std::unique_ptr<llaminar2::ITensorGemm>, KernelFactory::SlicedKeyHash> KernelFactory::sliced_cache_;
            std::unordered_map<KernelFactory::FusedQKVCacheKey, std::unique_ptr<llaminar2::ITensorFusedQKVGemm>, KernelFactory::FusedQKVKeyHash> KernelFactory::fused_qkv_cache_;
            std::unordered_map<KernelFactory::FusedGateUpCacheKey, std::unique_ptr<llaminar2::ITensorFusedGateUpGemm>, KernelFactory::FusedGateUpKeyHash> KernelFactory::fused_gate_up_cache_;
            std::unordered_map<KernelFactory::RoPECacheKey, std::unique_ptr<llaminar2::ITensorRoPE>, KernelFactory::RoPECacheKeyHash> KernelFactory::rope_cache_;
            std::unordered_map<KernelFactory::RMSNormCacheKey, std::unique_ptr<llaminar2::ITensorRMSNorm>, KernelFactory::RMSNormCacheKeyHash> KernelFactory::rmsnorm_cache_;
            std::unordered_map<KernelFactory::SwiGLUCacheKey, std::unique_ptr<llaminar2::ITensorSwiGLU>, KernelFactory::SwiGLUCacheKeyHash> KernelFactory::swiglu_cache_;
            std::unordered_map<KernelFactory::SoftmaxCacheKey, std::unique_ptr<llaminar2::ITensorSoftmax>, KernelFactory::SoftmaxCacheKeyHash> KernelFactory::softmax_cache_;
            std::unordered_map<KernelFactory::ResidualAddCacheKey, std::unique_ptr<llaminar2::ITensorResidualAdd>, KernelFactory::ResidualAddCacheKeyHash> KernelFactory::residual_add_cache_;
            std::unordered_map<KernelFactory::AttentionCacheKey, std::unique_ptr<llaminar2::ITensorAttention>, KernelFactory::AttentionCacheKeyHash> KernelFactory::attention_cache_;
            std::unordered_map<KernelFactory::EmbeddingCacheKey, std::unique_ptr<llaminar2::ITensorEmbedding>, KernelFactory::EmbeddingCacheKeyHash> KernelFactory::embedding_cache_;
            std::unordered_map<DeviceKernelKey, std::shared_ptr<void>, DeviceKernelKeyHash> KernelFactory::device_kernel_registry_;
            std::unordered_map<KernelFactory::PreparedGemmKey, std::shared_ptr<KernelFactory::PreparedGemmHandle>, KernelFactory::PreparedGemmKeyHash> KernelFactory::prepared_gemm_registry_;
            std::unordered_map<DeviceKernelKey, std::shared_ptr<KernelFactory::IGemmEngine>, DeviceKernelKeyHash> KernelFactory::device_gemm_engine_registry_;

            // NOTE: Device-level kernel caching (hipBLAS, cuBLAS handles, etc.)
            // has moved to DeviceKernelCache. See kernels/DeviceKernelCache.h

            // ==========================================================================
            // Kernel Cache - Implementation
            // ==========================================================================

            /**
             * @brief Wrapper for packed weights stored in tensor's cache_
             *
             * This wrapper enables tensor-owned packed weights. The tensor stores
             * QuantisedPackedWeights in its std::any cache_, and lightweight kernels
             * reference the packed data via external_packed_.
             */
            struct TensorPackedWeightsCache
            {
                llaminar2::gemm_v4::QuantisedPackedWeights packed;
            };

            /**
             * @brief Ensure tensor has packed weights in its cache and return pointer
             *
             * This is the canonical way to get packed weights for a tensor. The weights
             * are stored in tensor->cache_ so they outlive any individual kernel.
             *
             * Thread-safe: acquires cache_mutex_ internally.
             *
             * @param tensor Quantized tensor that implements IINT8Unpackable
             * @return Pointer to packed weights (stored in tensor's cache_)
             * @throws std::runtime_error if packing fails
             */
            const llaminar2::gemm_v4::QuantisedPackedWeights *
            KernelFactory::ensurePackedWeightsInTensorCache(const llaminar2::TensorBase *tensor)
            {
                // Check if tensor already has packed weights cached
                TensorPackedWeightsCache *packed_cache = nullptr;
                if (tensor->cache_.has_value())
                {
                    try
                    {
                        packed_cache = std::any_cast<TensorPackedWeightsCache *>(tensor->cache_);
                        if (packed_cache)
                        {
                            return &packed_cache->packed;
                        }
                    }
                    catch (const std::bad_any_cast &)
                    {
                        // cache_ contains something else - overwrite
                    }
                }

                // Pack weights into tensor's cache_
                auto *new_cache = new TensorPackedWeightsCache();
                if (!llaminar2::gemm_v4::QuantisedGemmKernel::packWeightsInto(
                        tensor, new_cache->packed, 0, -1))
                {
                    delete new_cache;
                    LOG_ERROR("[KernelFactory] Failed to pack weights for tensor type "
                              << static_cast<int>(tensor->native_type()));
                    throw std::runtime_error("KernelFactory: failed to pack weights");
                }

                // Store in tensor's cache_ (tensor owns the packed data)
                tensor->cache_ = new_cache;

                LOG_TRACE("[KernelFactory] Packed weights for tensor "
                          << tensor << " (" << new_cache->packed.N << "x" << new_cache->packed.K << ")");

                return &new_cache->packed;
            }

#ifdef HAVE_CUDA
            /**
             * @brief Cache wrapper for CUDA packed weights
             *
             * Stored in tensor->cuda_cache_ to manage lifetime.
             */
            struct TensorCUDAPackedWeightsCache
            {
                llaminar2::cuda::CUDAPackedWeights packed;
            };

            llaminar2::cuda::CUDAPackedWeights *
            KernelFactory::ensureCUDAPackedWeightsInTensorCache(const llaminar2::TensorBase *tensor)
            {
                // Check if tensor already has CUDA packed weights cached
                TensorCUDAPackedWeightsCache *packed_cache = nullptr;
                if (tensor->cuda_cache_.has_value())
                {
                    try
                    {
                        packed_cache = std::any_cast<TensorCUDAPackedWeightsCache *>(tensor->cuda_cache_);
                        if (packed_cache)
                        {
                            return &packed_cache->packed;
                        }
                    }
                    catch (const std::bad_any_cast &)
                    {
                        // cuda_cache_ contains something else - overwrite
                    }
                }

                // Pack weights into tensor's cuda_cache_
                auto *new_cache = new TensorCUDAPackedWeightsCache();
                if (!llaminar2::cuda::packWeightsToCUDA(tensor, new_cache->packed))
                {
                    delete new_cache;
                    LOG_ERROR("[KernelFactory] Failed to pack CUDA weights for tensor type "
                              << static_cast<int>(tensor->native_type()));
                    throw std::runtime_error("KernelFactory: failed to pack CUDA weights");
                }

                // Store back-reference to source tensor for coherence marking
                // const_cast is safe here because we only use it to mark device_valid_
                // after uploading the packed weights to GPU
                new_cache->packed.source_tensor_ = const_cast<llaminar2::TensorBase *>(tensor);

                // Store in tensor's cuda_cache_ (tensor owns the packed data)
                tensor->cuda_cache_ = new_cache;

                LOG_DEBUG("[KernelFactory] Packed CUDA INT8 weights for tensor "
                          << tensor << " (" << new_cache->packed.N << "x" << new_cache->packed.K << ")");

                return &new_cache->packed;
            }
#endif

#ifdef HAVE_ROCM
            /**
             * @brief Cache wrapper for ROCm packed weights
             *
             * Stored in tensor->rocm_cache_ to manage lifetime.
             * The ROCmPackedWeights struct contains both host (int8_data, scales)
             * and device (d_int8_data, d_scales) pointers. Device memory is uploaded
             * lazily on first kernel use.
             */
            struct TensorROCmPackedWeightsCache
            {
                llaminar2::rocm::ROCmPackedWeights packed;
            };

            llaminar2::rocm::ROCmPackedWeights *
            KernelFactory::ensureROCmPackedWeightsInTensorCache(const llaminar2::TensorBase *tensor)
            {
                // Check if tensor already has ROCm packed weights cached
                TensorROCmPackedWeightsCache *packed_cache = nullptr;
                if (tensor->rocm_cache_.has_value())
                {
                    try
                    {
                        packed_cache = std::any_cast<TensorROCmPackedWeightsCache *>(tensor->rocm_cache_);
                        if (packed_cache)
                        {
                            return &packed_cache->packed;
                        }
                    }
                    catch (const std::bad_any_cast &)
                    {
                        // rocm_cache_ contains something else - overwrite
                    }
                }

                // Pack weights into tensor's rocm_cache_
                auto *new_cache = new TensorROCmPackedWeightsCache();
                if (!llaminar2::rocm::packWeightsToROCm(tensor, new_cache->packed))
                {
                    delete new_cache;
                    LOG_ERROR("[KernelFactory] Failed to pack ROCm weights for tensor type "
                              << static_cast<int>(tensor->native_type()));
                    throw std::runtime_error("KernelFactory: failed to pack ROCm weights");
                }

                // Store in tensor's rocm_cache_ (tensor owns the packed data)
                tensor->rocm_cache_ = new_cache;

                LOG_DEBUG("[KernelFactory] Packed ROCm INT8 weights for tensor "
                          << tensor << " (" << new_cache->packed.N << "x" << new_cache->packed.K << ")");

                return &new_cache->packed;
            }
#endif

            // NOTE: Shared device kernel caching (hipBLAS, cuBLAS handles, etc.)
            // has been moved to DeviceKernelCache. See kernels/DeviceKernelCache.h

            llaminar2::ITensorGemm *KernelFactory::getOrCreateGemmSliced(
                const llaminar2::TensorBase *tensor,
                size_t row_start,
                size_t row_end)
            {
                if (!tensor)
                {
                    LOG_ERROR("[KernelFactory] getOrCreateGemmSliced: tensor is null");
                    return nullptr;
                }

                // Validate row range
                const auto &shape = tensor->shape();
                if (shape.size() != 2)
                {
                    LOG_ERROR("[KernelFactory] Tensor must be 2D for GEMM, got " << shape.size() << "D");
                    throw std::runtime_error("KernelFactory: tensor must be 2D");
                }
                size_t tensor_n = shape[0]; // rows = output features
                if (row_start >= row_end || row_end > tensor_n)
                {
                    LOG_ERROR("[KernelFactory] Invalid row range [" << row_start << ", " << row_end
                                                                    << ") for tensor with N=" << tensor_n);
                    throw std::runtime_error("KernelFactory: invalid row range for sliced GEMM");
                }

                // Check sliced cache
                SlicedCacheKey key{tensor, row_start, row_end};

                std::lock_guard<std::mutex> lock(cache_mutex_);
                auto it = sliced_cache_.find(key);
                if (it != sliced_cache_.end())
                {
                    LOG_DEBUG("[KernelFactory] Sliced GEMM cache HIT: tensor=" << tensor
                                                                               << " rows=[" << row_start << "," << row_end << ")");
                    return it->second.get();
                }

                // Cache miss - create sliced kernel
                LOG_DEBUG("[KernelFactory] Sliced GEMM cache MISS: creating kernel for tensor=" << tensor
                                                                                                << " rows=[" << row_start << "," << row_end << ")");

                // Get device type from tensor's current location
                auto current_dev = tensor->current_device();
                DeviceType dev_type = DeviceType::CPU;
                if (current_dev.has_value() && current_dev->is_gpu())
                {
                    dev_type = current_dev->is_cuda() ? DeviceType::CUDA : DeviceType::ROCm;
                }

                if (dev_type != DeviceType::CPU)
                {
                    LOG_ERROR("[KernelFactory] Sliced GEMM currently only supported on CPU, got " << to_string(dev_type));
                    throw std::runtime_error("KernelFactory: sliced GEMM only supported on CPU");
                }

                // Create sliced kernel using row-range constructor
                auto kernel = std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(
                    tensor, static_cast<int>(row_start), static_cast<int>(row_end));

                auto *raw_ptr = kernel.get();
                sliced_cache_[key] = std::move(kernel);

                LOG_DEBUG("[KernelFactory] Created sliced GEMM kernel: N=" << (row_end - row_start)
                                                                           << ", K=" << shape[1]);

                return raw_ptr;
            }

            void KernelFactory::clearCacheFor(const llaminar2::TensorBase *tensor)
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);

                // Also clear any sliced kernels for this tensor
                for (auto it = sliced_cache_.begin(); it != sliced_cache_.end();)
                {
                    if (it->first.tensor == tensor)
                    {
                        it = sliced_cache_.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }

                // Also clean up tensor's packed weights cache if present (CPU VNNI)
                if (tensor->cache_.has_value())
                {
                    try
                    {
                        auto *packed_cache = std::any_cast<TensorPackedWeightsCache *>(tensor->cache_);
                        if (packed_cache)
                        {
                            delete packed_cache;
                            tensor->cache_.reset();
                        }
                    }
                    catch (const std::bad_any_cast &)
                    {
                        // cache_ contains something else - leave it alone
                    }
                }

#ifdef HAVE_CUDA
                // Also clean up tensor's CUDA packed weights cache if present
                if (tensor->cuda_cache_.has_value())
                {
                    try
                    {
                        auto *cuda_packed_cache = std::any_cast<TensorCUDAPackedWeightsCache *>(tensor->cuda_cache_);
                        if (cuda_packed_cache)
                        {
                            delete cuda_packed_cache; // ~CUDAPackedWeights frees device memory
                            tensor->cuda_cache_.reset();
                        }
                    }
                    catch (const std::bad_any_cast &)
                    {
                        // cuda_cache_ contains something else - leave it alone
                    }
                }
#endif

#ifdef HAVE_ROCM
                // Also clean up tensor's ROCm packed weights cache if present
                if (tensor->rocm_cache_.has_value())
                {
                    try
                    {
                        auto *rocm_packed_cache = std::any_cast<TensorROCmPackedWeightsCache *>(tensor->rocm_cache_);
                        if (rocm_packed_cache)
                        {
                            delete rocm_packed_cache; // ~ROCmPackedWeights frees device memory
                            tensor->rocm_cache_.reset();
                        }
                    }
                    catch (const std::bad_any_cast &)
                    {
                        // rocm_cache_ contains something else - leave it alone
                    }
                }
#endif

                // Clear any fused QKV kernels that reference this tensor
                for (auto it = fused_qkv_cache_.begin(); it != fused_qkv_cache_.end();)
                {
                    const auto *wq_handle = it->first.wq_handle;
                    const auto *wk_handle = it->first.wk_handle;
                    const auto *wv_handle = it->first.wv_handle;
                    const bool references_tensor =
                        (wq_handle && wq_handle->tensor == tensor) ||
                        (wk_handle && wk_handle->tensor == tensor) ||
                        (wv_handle && wv_handle->tensor == tensor);

                    if (references_tensor)
                    {
                        it = fused_qkv_cache_.erase(it);
                    }
                    else
                    {
                        ++it;
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

                for (auto it = prepared_gemm_registry_.begin(); it != prepared_gemm_registry_.end();)
                {
                    if (it->first.tensor == tensor)
                    {
                        it = prepared_gemm_registry_.erase(it);
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
                tracked_tensors.reserve(prepared_gemm_registry_.size() + sliced_cache_.size() + fused_qkv_cache_.size() + fused_gate_up_cache_.size());

                for (const auto &entry : prepared_gemm_registry_)
                {
                    if (entry.first.tensor)
                    {
                        tracked_tensors.insert(entry.first.tensor);
                    }
                }
                for (const auto &entry : sliced_cache_)
                {
                    if (entry.first.tensor)
                    {
                        tracked_tensors.insert(entry.first.tensor);
                    }
                }
                for (const auto &entry : fused_qkv_cache_)
                {
                    if (entry.first.wq_handle && entry.first.wq_handle->tensor)
                    {
                        tracked_tensors.insert(entry.first.wq_handle->tensor);
                    }
                    if (entry.first.wk_handle && entry.first.wk_handle->tensor)
                    {
                        tracked_tensors.insert(entry.first.wk_handle->tensor);
                    }
                    if (entry.first.wv_handle && entry.first.wv_handle->tensor)
                    {
                        tracked_tensors.insert(entry.first.wv_handle->tensor);
                    }
                }
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

                    // Clean up CPU packed weights
                    if (tensor->cache_.has_value())
                    {
                        try
                        {
                            auto *packed_cache = std::any_cast<TensorPackedWeightsCache *>(tensor->cache_);
                            if (packed_cache)
                            {
                                delete packed_cache;
                                tensor->cache_.reset();
                            }
                        }
                        catch (const std::bad_any_cast &)
                        {
                            // cache_ contains something else - leave it alone
                        }
                    }

#ifdef HAVE_CUDA
                    // Clean up CUDA packed weights
                    if (tensor->cuda_cache_.has_value())
                    {
                        try
                        {
                            auto *cuda_packed_cache = std::any_cast<TensorCUDAPackedWeightsCache *>(tensor->cuda_cache_);
                            if (cuda_packed_cache)
                            {
                                delete cuda_packed_cache; // ~CUDAPackedWeights frees device memory
                                tensor->cuda_cache_.reset();
                            }
                        }
                        catch (const std::bad_any_cast &)
                        {
                            // cuda_cache_ contains something else - leave it alone
                        }
                    }
#endif

#ifdef HAVE_ROCM
                    // Clean up ROCm packed weights
                    if (tensor->rocm_cache_.has_value())
                    {
                        try
                        {
                            auto *rocm_packed_cache = std::any_cast<TensorROCmPackedWeightsCache *>(tensor->rocm_cache_);
                            if (rocm_packed_cache)
                            {
                                delete rocm_packed_cache; // ~ROCmPackedWeights frees device memory
                                tensor->rocm_cache_.reset();
                            }
                        }
                        catch (const std::bad_any_cast &)
                        {
                            // rocm_cache_ contains something else - leave it alone
                        }
                    }
#endif
                }

                sliced_cache_.clear();
                fused_qkv_cache_.clear();
                fused_gate_up_cache_.clear();
                rope_cache_.clear();
                rmsnorm_cache_.clear();
                swiglu_cache_.clear();
                softmax_cache_.clear();
                residual_add_cache_.clear();
                attention_cache_.clear();
                embedding_cache_.clear();
                device_kernel_registry_.clear();
                prepared_gemm_registry_.clear();
                device_gemm_engine_registry_.clear();
            }

            std::pair<size_t, size_t> KernelFactory::cacheStats()
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                size_t total_bytes = 0;
                // Note: Can't easily compute packed_bytes without RTTI to QuantisedGemmKernel
                // For now, just return count (includes all caches)
                return {sliced_cache_.size() + fused_qkv_cache_.size() + fused_gate_up_cache_.size() + rope_cache_.size() + rmsnorm_cache_.size() + swiglu_cache_.size() + softmax_cache_.size() + residual_add_cache_.size() + attention_cache_.size() + embedding_cache_.size() + device_kernel_registry_.size() + prepared_gemm_registry_.size(), total_bytes};
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

            const KernelFactory::PreparedGemmHandle *KernelFactory::getOrCreatePreparedGemmWeights(
                const llaminar2::TensorBase *tensor,
                llaminar2::DeviceId target_device,
                GemmPreparationKind prep_kind)
            {
                if (!tensor)
                {
                    throw std::runtime_error("KernelFactory::getOrCreatePreparedGemmWeights: null tensor");
                }

                const bool quantized = dynamic_cast<const llaminar2::IINT8Unpackable *>(tensor) != nullptr;

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

                const PreparedGemmKey key{
                    tensor,
                    target_device,
                    static_cast<int>(resolved_kind)};

                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    auto it = prepared_gemm_registry_.find(key);
                    if (it != prepared_gemm_registry_.end())
                    {
                        LOG_DEBUG("[KernelFactory][PhaseC] prepared weights hit dev=" << static_cast<int>(target_device.type)
                                                                                        << ":" << target_device.ordinal
                                                                                        << " kind=" << static_cast<int>(resolved_kind)
                                                                                        << " tensor=" << tensor);
                        return it->second.get();
                    }
                }

                if (quantized)
                {
                    switch (resolved_kind)
                    {
                    case GemmPreparationKind::CPU_PACKED:
                        (void)ensurePackedWeightsInTensorCache(tensor);
                        break;
                    case GemmPreparationKind::CUDA_INT8_PACKED:
#ifdef HAVE_CUDA
                        (void)ensureCUDAPackedWeightsInTensorCache(tensor);
#endif
                        break;
                    case GemmPreparationKind::ROCM_INT8_PACKED:
#ifdef HAVE_ROCM
                        (void)ensureROCmPackedWeightsInTensorCache(tensor);
#endif
                        break;
                    default:
                        break;
                    }
                }

                auto handle = std::make_shared<PreparedGemmHandle>();
                handle->tensor = tensor;
                handle->device_id = target_device;
                handle->kind = resolved_kind;
                handle->variant = static_cast<int>(tensor->native_type());
                handle->prepared_weights = std::make_shared<PreparedGemmWeights>();
                handle->prepared_weights->kind = resolved_kind;

                // Bind executable GEMM kernel via explicit-device create path.
                // This avoids direct dependency on legacy tensor-keyed
                // legacy GEMM cache-style behavior in the prepared path.
                auto bound_kernel = createPreparedKernelForDevice(tensor, target_device);
                if (!bound_kernel)
                {
                    throw std::runtime_error("KernelFactory::getOrCreatePreparedGemmWeights: failed to bind prepared kernel");
                }
                handle->prepared_weights->owned_kernel = std::shared_ptr<llaminar2::ITensorGemm>(std::move(bound_kernel));
                handle->prepared_weights->kernel = handle->prepared_weights->owned_kernel.get();

                std::lock_guard<std::mutex> lock(cache_mutex_);
                auto [it, inserted] = prepared_gemm_registry_.emplace(key, std::move(handle));
                LOG_DEBUG("[KernelFactory][PhaseC] prepared weights " << (inserted ? "create" : "race-hit")
                                                                        << " dev=" << static_cast<int>(target_device.type)
                                                                        << ":" << target_device.ordinal
                                                                        << " kind=" << static_cast<int>(resolved_kind)
                                                                        << " tensor=" << tensor);
                return it->second.get();
            }

            void KernelFactory::clearPreparedGemmWeightsFor(const llaminar2::TensorBase *tensor)
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);

                // Evict fused cache entries that reference prepared handles for this tensor.
                // This prevents stale prepared-handle keys after prepared_gemm_registry_ erase.
                for (auto it = fused_qkv_cache_.begin(); it != fused_qkv_cache_.end();)
                {
                    const auto *wq_handle = it->first.wq_handle;
                    const auto *wk_handle = it->first.wk_handle;
                    const auto *wv_handle = it->first.wv_handle;
                    const bool references_tensor =
                        (wq_handle && wq_handle->tensor == tensor) ||
                        (wk_handle && wk_handle->tensor == tensor) ||
                        (wv_handle && wv_handle->tensor == tensor);

                    if (references_tensor)
                    {
                        it = fused_qkv_cache_.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }

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

                for (auto it = prepared_gemm_registry_.begin(); it != prepared_gemm_registry_.end();)
                {
                    if (it->first.tensor == tensor)
                    {
                        it = prepared_gemm_registry_.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
            }

            size_t KernelFactory::gemmEngineRegistrySize()
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                return device_gemm_engine_registry_.size();
            }

            size_t KernelFactory::preparedGemmRegistrySize()
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                return prepared_gemm_registry_.size();
            }

            size_t KernelFactory::deviceScopedGemmEngineRegistrySize()
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                return device_gemm_engine_registry_.size();
            }

            // ==========================================================================
            // Fused QKV GEMM Adapter and Factory Methods
            // ==========================================================================

            /**
             * @brief Adapter class that wraps FusedGEMM to implement ITensorFusedQKVGemm
             *
             * This adapter bridges the FusedGEMM implementation to the ITensorFusedQKVGemm
             * interface, enabling KernelFactory to provide cached fused QKV GEMM kernels.
             */
            class FusedQKVGemmAdapter : public llaminar2::ITensorFusedQKVGemm
            {
            public:
                FusedQKVGemmAdapter(const llaminar2::TensorBase *wq,
                                    const llaminar2::TensorBase *wk,
                                    const llaminar2::TensorBase *wv)
                    : impl_(std::make_unique<llaminar2::FusedGEMM>(wq, wk, wv))
                {
                }

                bool execute_fp32(
                    const float *input_fp32,
                    void *output_q,
                    void *output_k,
                    void *output_v,
                    const llaminar2::TensorBase *bias_q,
                    const llaminar2::TensorBase *bias_k,
                    const llaminar2::TensorBase *bias_v,
                    int m, int n_q, int n_k, int k,
                    int k_block_size) override
                {
                    // FusedGEMM.execute_to_q8_1 handles FP32→Q8_1 quantization internally
                    return impl_->execute_to_q8_1(
                        input_fp32,
                        output_q, output_k, output_v,
                        bias_q, bias_k, bias_v,
                        m, n_q, n_k, k,
                        nullptr, -1);
                }

                bool execute_q8_1(
                    const llaminar2::Q8_1Block *input_q8_1,
                    void *output_q,
                    void *output_k,
                    void *output_v,
                    const llaminar2::TensorBase *bias_q,
                    const llaminar2::TensorBase *bias_k,
                    const llaminar2::TensorBase *bias_v,
                    int m, int n_q, int n_k, int k,
                    int k_block_size) override
                {
                    // Use mixed-precision path: Q=Q8_1, K=Q16_1, V=Q8_1
                    return impl_->execute_q8_1_mixed_qkv(
                        input_q8_1,
                        output_q, output_k, output_v,
                        bias_q, bias_k, bias_v,
                        m, n_q, n_k, k,
                        k_block_size,
                        nullptr, -1);
                }

                bool execute_q8_1_to_q8_1(
                    const llaminar2::Q8_1Block *input_q8_1,
                    void *output_q,
                    void *output_k,
                    void *output_v,
                    const llaminar2::TensorBase *bias_q,
                    const llaminar2::TensorBase *bias_k,
                    const llaminar2::TensorBase *bias_v,
                    int m, int n_q, int n_k, int k) override
                {
                    // Uniform Q8_1 output path
                    return impl_->execute_q8_1_to_q8_1(
                        input_q8_1,
                        output_q, output_k, output_v,
                        bias_q, bias_k, bias_v,
                        m, n_q, n_k, k,
                        nullptr, -1);
                }

                bool supports_device(int device_idx) const override
                {
                    // FusedQKVGemm currently only supports CPU (device_idx == -1)
                    return device_idx == -1;
                }

            private:
                std::unique_ptr<llaminar2::FusedGEMM> impl_;
            };

            /**
             * @brief Build fused QKV adapter from already-resolved prepared handles
             *
             * This helper centralizes fused QKV construction semantics so both
             * public `createFusedQKVGemm(..., DeviceId)` and cache-miss handling in
             * `getOrCreateFusedQKVGemm(..., DeviceId)` use exactly the same path.
             *
             * Why this helper exists:
             * - Keeps a single authority for validation and adapter construction.
             * - Avoids repeating prepared-handle resolution in cache-miss paths.
             * - Maintains prepared-handle-first architecture for fused cache identity.
             */
            static std::unique_ptr<llaminar2::ITensorFusedQKVGemm> createFusedQKVAdapterFromPrepared(
                const KernelFactory::PreparedGemmHandle *prepared_q,
                const KernelFactory::PreparedGemmHandle *prepared_k,
                const KernelFactory::PreparedGemmHandle *prepared_v,
                llaminar2::DeviceId target_device)
            {
                // Prepared-handle-native contract: callers provide already-resolved
                // handle identities, and this helper performs only construction-time
                // validation plus adapter creation. This keeps cache miss behavior
                // and explicit create behavior semantically identical.
                if (!prepared_q || !prepared_k || !prepared_v)
                {
                    LOG_ERROR("[KernelFactory] createFusedQKVAdapterFromPrepared: null prepared handle(s)");
                    throw std::runtime_error("FusedQKVGemm requires non-null prepared handles");
                }

                if (!prepared_q->tensor || !prepared_k->tensor || !prepared_v->tensor)
                {
                    LOG_ERROR("[KernelFactory] createFusedQKVAdapterFromPrepared: null tensor in prepared handle(s)");
                    throw std::runtime_error("FusedQKVGemm prepared handles must reference valid tensors");
                }

                // Current fused QKV adapter implementation is CPU-only.
                // Keeping this check here ensures both create() and getOrCreate() paths
                // apply identical device support validation.
                if (target_device.type != DeviceType::CPU)
                {
                    LOG_ERROR("[KernelFactory] FusedQKVGemm currently only supports CPU device");
                    throw std::runtime_error("FusedQKVGemm only supports CPU device");
                }

                return std::make_unique<FusedQKVGemmAdapter>(
                    prepared_q->tensor,
                    prepared_k->tensor,
                    prepared_v->tensor);
            }

            std::unique_ptr<llaminar2::ITensorFusedQKVGemm> KernelFactory::createFusedQKVGemm(
                const llaminar2::TensorBase *wq,
                const llaminar2::TensorBase *wk,
                const llaminar2::TensorBase *wv,
                DeviceType dev_type)
            {
                return createFusedQKVGemm(wq, wk, wv, llaminar2::DeviceId(dev_type, 0));
            }

            std::unique_ptr<llaminar2::ITensorFusedQKVGemm> KernelFactory::createFusedQKVGemm(
                const llaminar2::TensorBase *wq,
                const llaminar2::TensorBase *wk,
                const llaminar2::TensorBase *wv,
                llaminar2::DeviceId target_device)
            {
                if (!wq || !wk || !wv)
                {
                    LOG_ERROR("[KernelFactory] createFusedQKVGemm: null weight tensor(s)");
                    throw std::runtime_error("FusedQKVGemm requires non-null weight tensors");
                }

                const auto *prepared_q = getOrCreatePreparedGemmWeights(wq, target_device);
                const auto *prepared_k = getOrCreatePreparedGemmWeights(wk, target_device);
                const auto *prepared_v = getOrCreatePreparedGemmWeights(wv, target_device);

                // Use the same helper as cache-miss path so explicit creation and
                // cached creation cannot diverge in validation or adapter wiring.
                return createFusedQKVAdapterFromPrepared(prepared_q, prepared_k, prepared_v, target_device);
            }

            llaminar2::ITensorFusedQKVGemm *KernelFactory::getOrCreateFusedQKVGemm(
                const llaminar2::TensorBase *wq,
                const llaminar2::TensorBase *wk,
                const llaminar2::TensorBase *wv,
                DeviceType dev_type)
            {
                // Get ordinal from thread-local (for future GPU support) or default to 0 for CPU
                int target_ordinal = 0;
#ifdef HAVE_ROCM
                if (dev_type == DeviceType::ROCm && tl_target_rocm_ordinal.has_value())
                {
                    target_ordinal = tl_target_rocm_ordinal.value();
                }
#endif
#ifdef HAVE_CUDA
                if (dev_type == DeviceType::CUDA && tl_target_cuda_ordinal.has_value())
                {
                    target_ordinal = tl_target_cuda_ordinal.value();
                }
#endif

                const llaminar2::DeviceId target_device_id(dev_type, target_ordinal);
                return getOrCreateFusedQKVGemm(wq, wk, wv, target_device_id);
            }

            llaminar2::ITensorFusedQKVGemm *KernelFactory::getOrCreateFusedQKVGemm(
                const llaminar2::TensorBase *wq,
                const llaminar2::TensorBase *wk,
                const llaminar2::TensorBase *wv,
                llaminar2::DeviceId target_device)
            {
                const auto *prepared_q = getOrCreatePreparedGemmWeights(wq, target_device);
                const auto *prepared_k = getOrCreatePreparedGemmWeights(wk, target_device);
                const auto *prepared_v = getOrCreatePreparedGemmWeights(wv, target_device);

                FusedQKVCacheKey key{prepared_q, prepared_k, prepared_v, target_device};

                // Fast path: check cache under lock.
                // Cache key is prepared-handle identity + device, so hits are aligned
                // with backend-ready weight state, not just raw tensor pointers.
                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    auto it = fused_qkv_cache_.find(key);
                    if (it != fused_qkv_cache_.end())
                    {
                        LOG_TRACE("[KernelFactory::getOrCreateFusedQKVGemm] CACHE HIT");
                        return it->second.get();
                    }
                }

                LOG_DEBUG("[KernelFactory::getOrCreateFusedQKVGemm] CACHE MISS - creating new kernel");

                // Create kernel WITHOUT holding cache_mutex_. Use already-resolved
                // prepared handles so cache-miss path does not repeat preparation lookup.
                auto kernel = createFusedQKVAdapterFromPrepared(prepared_q, prepared_k, prepared_v, target_device);

                // Re-acquire lock to insert into cache (double-checked locking).
                // Another thread may have populated the same key while construction ran.
                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);

                    // Double-check another thread didn't create it while we were unlocked
                    auto it = fused_qkv_cache_.find(key);
                    if (it != fused_qkv_cache_.end())
                    {
                        LOG_TRACE("[KernelFactory::getOrCreateFusedQKVGemm] Lost race, returning existing kernel");
                        return it->second.get();
                    }

                    auto *raw_ptr = kernel.get();
                    fused_qkv_cache_[key] = std::move(kernel);
                    return raw_ptr;
                }
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

                    if (!gemm_gate_ || !gemm_up_)
                    {
                        LOG_ERROR("[FusedGateUpGemmAdapter] Null GEMM kernel(s)");
                        return false;
                    }

                    // Execute both GEMMs via multiply_tensor for proper type dispatch
                    bool gate_ok = gemm_gate_->multiply_tensor(
                        input, output_gate,
                        m, n_gate, k,
                        false,      // transpose_B
                        1.0f, 0.0f, // alpha, beta
                        nullptr,    // bias
                        nullptr,    // mpi_ctx
                        device_idx);

                    if (!gate_ok)
                    {
                        LOG_ERROR("[FusedGateUpGemmAdapter] Gate GEMM failed");
                        return false;
                    }

                    bool up_ok = gemm_up_->multiply_tensor(
                        input, output_up,
                        m, n_up, k,
                        false,      // transpose_B
                        1.0f, 0.0f, // alpha, beta
                        nullptr,    // bias
                        nullptr,    // mpi_ctx
                        device_idx);

                    if (!up_ok)
                    {
                        LOG_ERROR("[FusedGateUpGemmAdapter] Up GEMM failed");
                        return false;
                    }

                    return true;
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

                    // For bias support, we need to use the optimized fused path
                    // which is available on FP32 outputs via multiply_fused
                    if (!gemm_gate_ || !gemm_up_)
                    {
                        LOG_ERROR("[FusedGateUpGemmAdapter] Null GEMM kernel(s)");
                        return false;
                    }

                    // Check if output is FP32 - if so, use multiply_fused
                    if (output_gate->native_type() == llaminar2::TensorType::FP32 &&
                        output_up->native_type() == llaminar2::TensorType::FP32)
                    {
                        // Get FP32 pointers
                        const float *input_fp32 = input->fp32_data();
                        float *output_gate_fp32 = output_gate->mutable_data();
                        float *output_up_fp32 = output_up->mutable_data();

                        if (!input_fp32 || !output_gate_fp32 || !output_up_fp32)
                        {
                            LOG_ERROR("[FusedGateUpGemmAdapter] Failed to get FP32 data");
                            return false;
                        }

                        // Build projection descriptors for fused multiply
                        // FusedProjectionDesc takes TensorBase* for bias
                        std::vector<llaminar2::ITensorGemm::FusedProjectionDesc> projections = {
                            {gemm_gate_, output_gate_fp32, n_gate, bias_gate, nullptr, false, "gate"},
                            {gemm_up_, output_up_fp32, n_up, bias_up, nullptr, false, "up"}};

                        return gemm_gate_->multiply_fused(input_fp32, projections, m, k);
                    }

                    // For non-FP32 outputs (e.g., Q8_1), bias is not supported
                    if (bias_gate || bias_up)
                    {
                        LOG_WARN("[FusedGateUpGemmAdapter] Bias not supported with non-FP32 output - ignoring");
                    }

                    // Fall back to regular execute without bias
                    return execute(input, output_gate, output_up, m, k, n_gate, n_up, ctx, device_idx);
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

            std::unique_ptr<llaminar2::ITensorFusedGateUpGemm> KernelFactory::createFusedGateUpGemm(
                const llaminar2::TensorBase *w_gate,
                const llaminar2::TensorBase *w_up,
                DeviceType dev_type)
            {
                // Delegate to DeviceId version with ordinal 0
                return createFusedGateUpGemm(w_gate, w_up, llaminar2::DeviceId(dev_type, 0));
            }

            std::unique_ptr<llaminar2::ITensorFusedGateUpGemm> KernelFactory::createFusedGateUpGemm(
                const llaminar2::TensorBase *w_gate,
                const llaminar2::TensorBase *w_up,
                llaminar2::DeviceId target_device)
            {
                if (!w_gate || !w_up)
                {
                    LOG_ERROR("[KernelFactory] createFusedGateUpGemm: null weight tensor(s)");
                    throw std::runtime_error("FusedGateUpGemm requires non-null weight tensors");
                }

                const auto *prepared_gate = getOrCreatePreparedGemmWeights(w_gate, target_device);
                const auto *prepared_up = getOrCreatePreparedGemmWeights(w_up, target_device);
                return createFusedGateUpAdapterFromPrepared(prepared_gate, prepared_up, target_device);
            }

            llaminar2::ITensorFusedGateUpGemm *KernelFactory::getOrCreateFusedGateUpGemm(
                const llaminar2::TensorBase *w_gate,
                const llaminar2::TensorBase *w_up,
                DeviceType dev_type)
            {
                // Delegate to DeviceId version with ordinal 0
                return getOrCreateFusedGateUpGemm(w_gate, w_up, llaminar2::DeviceId(dev_type, 0));
            }

            llaminar2::ITensorFusedGateUpGemm *KernelFactory::getOrCreateFusedGateUpGemm(
                const llaminar2::TensorBase *w_gate,
                const llaminar2::TensorBase *w_up,
                llaminar2::DeviceId target_device)
            {
                const auto *prepared_gate = getOrCreatePreparedGemmWeights(w_gate, target_device);
                const auto *prepared_up = getOrCreatePreparedGemmWeights(w_up, target_device);
                FusedGateUpCacheKey key{prepared_gate, prepared_up, target_device};

                // Fast path: acquire lock and check cache.
                // Key uses prepared-handle identity + device so it tracks per-device
                // prepared state rather than only raw tensor addresses.
                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    auto it = fused_gate_up_cache_.find(key);
                    if (it != fused_gate_up_cache_.end())
                    {
                        LOG_TRACE("[KernelFactory::getOrCreateFusedGateUpGemm] CACHE HIT for "
                                  << target_device.to_string());
                        return it->second.get();
                    }
                }

                LOG_DEBUG("[KernelFactory::getOrCreateFusedGateUpGemm] CACHE MISS for "
                          << target_device.to_string() << " - creating new kernel");

                // Create kernel WITHOUT holding lock. Reuse already-resolved prepared handles
                // to avoid duplicate prepared lookup on cache miss.
                auto kernel = createFusedGateUpAdapterFromPrepared(prepared_gate, prepared_up, target_device);

                // Second check: re-acquire lock and insert (double-checked locking)
                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    auto it = fused_gate_up_cache_.find(key);
                    if (it != fused_gate_up_cache_.end())
                    {
                        // Another thread inserted while we were creating - use theirs
                        LOG_TRACE("[KernelFactory::getOrCreateFusedGateUpGemm] Race detected - using existing kernel");
                        return it->second.get();
                    }
                    auto *raw_ptr = kernel.get();
                    fused_gate_up_cache_[key] = std::move(kernel);
                    return raw_ptr;
                }
            }

            // ==========================================================================
            // KVCache Factory Methods
            // ==========================================================================

            std::unique_ptr<llaminar2::IKVCache> KernelFactory::createKVCache(const KVCacheConfig &config)
            {
                if (config.device.is_cpu())
                {
                    return createCPUKVCache(config);
                }
                else if (config.device.is_cuda())
                {
#ifdef HAVE_CUDA
                    return createCUDAKVCache(config);
#else
                    LOG_ERROR("[KernelFactory] CUDA KVCache requested but HAVE_CUDA not defined");
                    throw std::runtime_error("KernelFactory::createKVCache: CUDA support not compiled in");
#endif
                }
                else if (config.device.is_rocm())
                {
#ifdef HAVE_ROCM
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

                // Q8_1 activations ONLY work with quantized weights
                // (QuantisedGemmKernel uses INT8×INT8 dot products)
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
                           wgt_name + " weights. The QuantisedGemmKernel uses INT8×INT8 VNNI " +
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
