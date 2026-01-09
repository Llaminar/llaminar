/**
 * @file KernelFactory.cpp
 * @brief Implementation of centralized kernel dispatch factory
 * @author David Sanftenberg
 */

#include "KernelFactory.h"
#include "cpu/gemm_v4/QuantisedGemmKernel.h"
#include "cpu/gemm_v4/FloatingPointGemmKernel.h"
#include "../tensors/TensorSlice.h"
#include "cpu/ops/CPURoPEKernelT.h"
#include "cpu/ops/CPUSwiGLUKernelT.h"
#include "cpu/ops/CPUSoftmaxKernelT.h"
#include "cpu/ops/CPURMSNormKernelT.h"
#include "cpu/ops/CPUResidualAddKernelT.h"
#include "cpu/ops/CPUEmbeddingKernelT.h"
#include "cpu/attention/CPUAttentionKernelT.h"

#include "../tensors/Tensors.h"
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

// ROCm kernel factory (when implemented)
#ifdef HAVE_ROCM
// #include "rocm/RocmGemmFactory.h"
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
                /**
                 * @brief Get CUDA device ID for kernel creation
                 *
                 * For weight tensors that are CPU-only (typical for GGUF models), we use
                 * the first available CUDA device (device 0). For tensors already on GPU,
                 * we use their actual device.
                 *
                 * @param tensor The tensor to check (usually a weight tensor)
                 * @return CUDA device ID (backend-specific, usually 0)
                 */
                int getCUDADeviceIdForKernel(const llaminar2::TensorBase *tensor)
                {
                    const auto &dm = llaminar2::DeviceManager::instance();
                    const auto &devices = dm.devices();

                    // Check if tensor is currently on a CUDA device
                    auto current_dev = tensor->current_device();
                    if (current_dev.has_value() && current_dev->is_cuda())
                    {
                        return current_dev->cuda_ordinal();
                    }

                    // Tensor not on CUDA GPU - find the first CUDA device
                    int cuda_dm_idx = dm.find_device(llaminar2::ComputeBackendType::GPU_CUDA, 0);
                    if (cuda_dm_idx >= 0 && static_cast<size_t>(cuda_dm_idx) < devices.size())
                    {
                        return devices[cuda_dm_idx].device_id;
                    }

                    // Fallback to device 0 if no CUDA device found via DeviceManager
                    LOG_WARN("[KernelFactory] No CUDA device found in DeviceManager, using device 0");
                    return 0;
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
                        int device_idx) override
                    {
                        (void)mpi_ctx; // Not used in typed kernel

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
                        int device_idx) override
                    {
                        (void)mpi_ctx; // Not used in typed kernel

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
                    throwUnsupportedKernel(dev_type, "RoPE", "FP32");
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
                    throwUnsupportedKernel(dev_type, "RoPE", "BF16");
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
                    throwUnsupportedKernel(dev_type, "RoPE", "FP16");
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
                    throwUnsupportedKernel(dev_type, "SwiGLU", "FP32");
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
                    throwUnsupportedKernel(dev_type, "SwiGLU", "BF16");
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
                    throwUnsupportedKernel(dev_type, "SwiGLU", "FP16");
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
                    throwUnsupportedKernel(dev_type, "ResidualAdd", "FP32");
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
                    throwUnsupportedKernel(dev_type, "ResidualAdd", "BF16");
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
                    throwUnsupportedKernel(dev_type, "ResidualAdd", "FP16");
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
                    throwUnsupportedKernel(dev_type, "RMSNorm", "FP32");
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
                    throwUnsupportedKernel(dev_type, "RMSNorm", "BF16");
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
                    throwUnsupportedKernel(dev_type, "RMSNorm", "FP16");
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

                // For GPU tensors, dispatch to CUDA Flash Attention directly
                if (tensor->is_on_gpu())
                {
#ifdef HAVE_CUDA
                    // GPU tensors use CUDAFlashAttentionKernelT based on native_type
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
                            "KernelFactory::createAttention: unsupported GPU tensor type " +
                            std::string(tensor->dtype_name()));
                    }
#else
                    throw std::runtime_error("KernelFactory::createAttention: GPU tensor but CUDA not available");
#endif
                }

                // For CPU tensors, use existing TensorBase* dispatch
                auto *cpu_tensor = dynamic_cast<const llaminar2::TensorBase *>(tensor);
                if (!cpu_tensor)
                {
                    throw std::runtime_error("KernelFactory::createAttention: non-GPU ITensor must be TensorBase");
                }
                return createAttention(cpu_tensor, dev_type);
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
                    return std::make_unique<llaminar2::CPUAttentionKernelT<llaminar2::ActivationPrecision::FP32>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::FP32>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    throwUnsupportedKernel(dev_type, "Attention", "FP32");
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
                    return std::make_unique<llaminar2::CPUAttentionKernelT<llaminar2::ActivationPrecision::BF16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::BF16>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    throwUnsupportedKernel(dev_type, "Attention", "BF16");
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
                    return std::make_unique<llaminar2::CPUAttentionKernelT<llaminar2::ActivationPrecision::FP16>>();

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::FP16>>();
#endif

#ifdef HAVE_ROCM
                case DeviceType::ROCm:
                    throwUnsupportedKernel(dev_type, "Attention", "FP16");
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
                    throwUnsupportedKernel(dev_type, "Embedding", "FP32");
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
                    throwUnsupportedKernel(dev_type, "Embedding", "BF16");
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
                    throwUnsupportedKernel(dev_type, "Embedding", "FP16");
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
                    throwUnsupportedKernel(dev_type, "Embedding", "Q8_1");
#endif

                default:
                    throwUnsupportedKernel(dev_type, "Embedding", "Q8_1");
                }
            }

            // ==========================================================================
            // Kernel Cache - Static Members
            // ==========================================================================

            std::unordered_map<const llaminar2::TensorBase *, std::unique_ptr<llaminar2::ITensorGemm>> KernelFactory::kernel_cache_;
            std::mutex KernelFactory::cache_mutex_;
            std::unordered_map<KernelFactory::SlicedCacheKey, std::unique_ptr<llaminar2::ITensorGemm>, KernelFactory::SlicedKeyHash> KernelFactory::sliced_cache_;
            std::unordered_map<KernelFactory::DeviceTargetedCacheKey, std::unique_ptr<llaminar2::ITensorGemm>, KernelFactory::DeviceTargetedKeyHash> KernelFactory::device_targeted_cache_;

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

                // Store in tensor's cuda_cache_ (tensor owns the packed data)
                tensor->cuda_cache_ = new_cache;

                LOG_DEBUG("[KernelFactory] Packed CUDA INT8 weights for tensor "
                          << tensor << " (" << new_cache->packed.N << "x" << new_cache->packed.K << ")");

                return &new_cache->packed;
            }
#endif

            llaminar2::ITensorGemm *KernelFactory::getOrCreateGemm(const llaminar2::TensorBase *tensor)
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);

                // Get tensor dimensions for validation
                const auto &shape = tensor->shape();
                if (shape.size() != 2)
                {
                    LOG_ERROR("[KernelFactory] Tensor must be 2D for GEMM, got " << shape.size() << "D");
                    throw std::runtime_error("KernelFactory: tensor must be 2D");
                }
                int tensor_n = static_cast<int>(shape[0]); // rows = output features
                int tensor_k = static_cast<int>(shape[1]); // cols = input features

                // Debug: Log cache lookup for Q weight tensors
                if (tensor_n == 896 && tensor_k == 896)
                {
                    LOG_TRACE("[KernelFactory::getOrCreateGemm] Looking up tensor=" << static_cast<const void *>(tensor)
                                                                                    << " shape=[" << tensor_n << "," << tensor_k << "]"
                                                                                    << " cache_size=" << kernel_cache_.size());
                }

                // Check kernel cache first
                auto it = kernel_cache_.find(tensor);
                if (it != kernel_cache_.end())
                {
                    // Validate cached kernel dimensions match tensor (handles memory reuse)
                    auto *quantised = dynamic_cast<llaminar2::gemm_v4::QuantisedGemmKernel *>(it->second.get());
                    if (quantised)
                    {
                        if (quantised->get_n() == tensor_n && quantised->get_k() == tensor_k)
                        {
                            if (tensor_n == 896 && tensor_k == 896)
                            {
                                LOG_TRACE("[KernelFactory::getOrCreateGemm] CACHE HIT: returning cached kernel=" << static_cast<const void *>(it->second.get()));
                            }
                            return it->second.get();
                        }
                        // Dimensions mismatch - tensor memory was reused, evict stale entry
                        LOG_DEBUG("[KernelFactory] Cache eviction due to dimension mismatch: "
                                  << "cached N=" << quantised->get_n() << ", K=" << quantised->get_k()
                                  << " vs tensor N=" << tensor_n << ", K=" << tensor_k);
                        kernel_cache_.erase(it);
                    }
                    else
                    {
                        // Non-quantised kernel (FP32/FP16/BF16) - just return it
                        // TODO: Add dimension validation for floating-point kernels too
                        return it->second.get();
                    }
                }

                // Get device type from tensor's current location
                auto current_dev = tensor->current_device();
                DeviceType dev_type = DeviceType::CPU;
                if (current_dev.has_value() && current_dev->is_gpu())
                {
                    dev_type = current_dev->is_cuda() ? DeviceType::CUDA : DeviceType::ROCm;
                }

                // Handle TensorSlice: delegate to inner tensor's kernel creation
                // TensorSlice always implements IINT8Unpackable, but the inner tensor may not
                if (auto *slice = dynamic_cast<const llaminar2::TensorSlice *>(tensor))
                {
                    // Check if inner tensor actually supports quantized GEMM
                    if (!slice->supports_int8_unpack())
                    {
                        // Inner tensor is FP32/FP16/BF16 - dispatch based on inner type
                        const auto *inner = slice->inner();
                        if (auto *t = dynamic_cast<const llaminar2::FP32Tensor *>(inner))
                        {
                            auto kernel = createGemm(t, dev_type);
                            auto *raw_ptr = kernel.get();
                            kernel_cache_[tensor] = std::move(kernel);
                            return raw_ptr;
                        }
                        else if (auto *t = dynamic_cast<const llaminar2::FP16Tensor *>(inner))
                        {
                            auto kernel = createGemm(t, dev_type);
                            auto *raw_ptr = kernel.get();
                            kernel_cache_[tensor] = std::move(kernel);
                            return raw_ptr;
                        }
                        else if (auto *t = dynamic_cast<const llaminar2::BF16Tensor *>(inner))
                        {
                            auto kernel = createGemm(t, dev_type);
                            auto *raw_ptr = kernel.get();
                            kernel_cache_[tensor] = std::move(kernel);
                            return raw_ptr;
                        }
                        else
                        {
                            LOG_ERROR("[KernelFactory] TensorSlice wraps unknown non-quantized type: "
                                      << static_cast<int>(inner->native_type()));
                            throw std::runtime_error("KernelFactory: TensorSlice wraps unknown type");
                        }
                    }
                    // Inner tensor supports INT8 unpack - fall through to quantized path below
                }

                // For CPU quantized tensors, use tensor-owned packed weights pattern
                if (dev_type == DeviceType::CPU)
                {
                    // Check if tensor implements IINT8Unpackable (quantized weight tensor)
                    const auto *unpackable = dynamic_cast<const llaminar2::IINT8Unpackable *>(tensor);
                    if (unpackable)
                    {
                        // For TensorSlice, verify inner actually supports it (already checked above but be safe)
                        if (auto *slice = dynamic_cast<const llaminar2::TensorSlice *>(tensor))
                        {
                            if (!slice->supports_int8_unpack())
                            {
                                // This should not happen if the above check passed
                                LOG_ERROR("[KernelFactory] TensorSlice passed IINT8Unpackable check but supports_int8_unpack=false");
                                throw std::runtime_error("KernelFactory: TensorSlice type mismatch");
                            }
                        }

                        // Get or create packed weights in tensor's cache_
                        TensorPackedWeightsCache *packed_cache = nullptr;

                        // Check if tensor already has packed weights cached
                        if (tensor->cache_.has_value())
                        {
                            try
                            {
                                packed_cache = std::any_cast<TensorPackedWeightsCache *>(tensor->cache_);
                            }
                            catch (const std::bad_any_cast &)
                            {
                                // cache_ contains something else (e.g., different type) - overwrite
                                packed_cache = nullptr;
                            }
                        }

                        if (!packed_cache)
                        {
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
                            packed_cache = new_cache;

                            LOG_TRACE("[KernelFactory] Packed weights for tensor "
                                      << tensor << " (" << packed_cache->packed.N << "x" << packed_cache->packed.K << ")");
                        }

                        // Create lightweight kernel referencing tensor's packed data
                        auto kernel = std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(
                            &packed_cache->packed);

                        auto *raw_ptr = kernel.get();

                        // Debug: Log cache miss for Q weight tensors
                        if (tensor_n == 896 && tensor_k == 896)
                        {
                            LOG_TRACE("[KernelFactory::getOrCreateGemm] CACHE MISS: created new kernel=" << static_cast<const void *>(raw_ptr)
                                                                                                         << " for tensor=" << static_cast<const void *>(tensor)
                                                                                                         << " pw.data=" << static_cast<const void *>(packed_cache->packed.packed_data.data()));
                        }

                        kernel_cache_[tensor] = std::move(kernel);
                        return raw_ptr;
                    }
                }

                // Fall through to legacy per-type dispatch for FP32/FP16/BF16 and GPU backends
                std::unique_ptr<llaminar2::ITensorGemm> kernel;

                // Floating-point tensors (no packing needed)
                if (auto *t = dynamic_cast<const llaminar2::FP32Tensor *>(tensor))
                {
                    kernel = createGemm(t, dev_type);
                }
                else if (auto *t = dynamic_cast<const llaminar2::FP16Tensor *>(tensor))
                {
                    kernel = createGemm(t, dev_type);
                }
                else if (auto *t = dynamic_cast<const llaminar2::BF16Tensor *>(tensor))
                {
                    kernel = createGemm(t, dev_type);
                }
#ifdef HAVE_CUDA
                // GPU backends for quantized types (CUDA uses different packing)
                else if (dev_type != DeviceType::CPU)
                {
                    // Handle TensorSlice: unwrap to get inner tensor type for dispatch
                    // This occurs when weights are sharded across MPI ranks
                    const llaminar2::TensorBase *dispatch_tensor = tensor;
                    if (auto *slice = dynamic_cast<const llaminar2::TensorSlice *>(tensor))
                    {
                        dispatch_tensor = slice->inner();
                        LOG_DEBUG("[KernelFactory] CUDA: Unwrapping TensorSlice -> inner type "
                                  << static_cast<int>(dispatch_tensor->native_type()));
                    }

                    // Type-based dispatch for GPU backends
                    if (auto *t = dynamic_cast<const llaminar2::IQ4_NLTensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q4_0Tensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q4_1Tensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q5_0Tensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q5_1Tensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q6_KTensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q8_0Tensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q8_1Tensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q2_KTensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q3_KTensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q4_KTensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q5_KTensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q8_KTensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::IQ1_MTensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::IQ1_STensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::IQ2_STensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::IQ2_XSTensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::IQ2_XXSTensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::IQ3_STensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::IQ3_XXSTensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::IQ4_XSTensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else
                    {
                        LOG_ERROR("[KernelFactory] Unknown tensor type for GPU: " << static_cast<int>(tensor->native_type()));
                        throw std::runtime_error("KernelFactory::getOrCreateGemm: unknown tensor type for GPU");
                    }
                }
#endif
                else
                {
                    LOG_ERROR("[KernelFactory] Unknown tensor type: " << static_cast<int>(tensor->native_type()));
                    throw std::runtime_error("KernelFactory::getOrCreateGemm: unknown tensor type");
                }

                // Cache and return
                auto *raw_ptr = kernel.get();
                kernel_cache_[tensor] = std::move(kernel);
                return raw_ptr;
            }

            llaminar2::ITensorGemm *KernelFactory::getOrCreateGemm(
                const llaminar2::TensorBase *tensor,
                DeviceType target_device)
            {
                // If target is CPU, just delegate to the regular version
                // Do this BEFORE acquiring lock to avoid deadlock
                if (target_device == DeviceType::CPU)
                {
                    return getOrCreateGemm(tensor);
                }

                std::lock_guard<std::mutex> lock(cache_mutex_);

                // Check device-targeted cache first
                DeviceTargetedCacheKey key{tensor, target_device};
                auto it = device_targeted_cache_.find(key);
                if (it != device_targeted_cache_.end())
                {
                    return it->second.get();
                }

                // Get tensor dimensions for validation
                const auto &shape = tensor->shape();
                if (shape.size() != 2)
                {
                    LOG_ERROR("[KernelFactory] Tensor must be 2D for GEMM, got " << shape.size() << "D");
                    throw std::runtime_error("KernelFactory: tensor must be 2D");
                }

                // Get device type from tensor's current location
                auto current_dev = tensor->current_device();
                DeviceType tensor_dev_type = DeviceType::CPU;
                if (current_dev.has_value() && current_dev->is_gpu())
                {
                    tensor_dev_type = current_dev->is_cuda() ? DeviceType::CUDA : DeviceType::ROCm;
                }
                LOG_DEBUG("[KernelFactory::getOrCreateGemm] Creating kernel with target device "
                          << static_cast<int>(target_device)
                          << " for tensor on device " << static_cast<int>(tensor_dev_type));

                std::unique_ptr<llaminar2::ITensorGemm> kernel;

                // Dispatch by tensor type, creating kernel for target_device
#ifdef HAVE_CUDA
                if (target_device == DeviceType::CUDA)
                {
                    // For CUDA target, create CUDA kernel regardless of where weights are stored
                    // The weights will be uploaded to GPU on first use
                    const auto *dispatch_tensor = tensor;

                    // Handle TensorSlice - get inner tensor for type dispatch
                    if (auto *slice = dynamic_cast<const llaminar2::TensorSlice *>(tensor))
                    {
                        if (!slice->supports_int8_unpack())
                        {
                            // NO FALLBACK: TensorSlice of non-quantized type not supported on CUDA
                            throw std::runtime_error(
                                "TensorSlice of non-quantized type not supported on CUDA - no silent fallback to CPU");
                        }
                    }

                    // Create CUDA kernel based on tensor type
                    if (auto *t = dynamic_cast<const llaminar2::IQ4_NLTensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, target_device);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q4_0Tensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, target_device);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q8_0Tensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, target_device);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q8_1Tensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, target_device);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::FP32Tensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, target_device);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::FP16Tensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, target_device);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::BF16Tensor *>(dispatch_tensor))
                    {
                        kernel = createGemm(t, target_device);
                    }
                    else
                    {
                        // NO FALLBACK: tensor type not supported on CUDA
                        throw std::runtime_error(
                            "Tensor type " + std::to_string(static_cast<int>(tensor->native_type())) +
                            " not supported on CUDA - no silent fallback to CPU");
                    }
                }
#endif
#ifdef HAVE_ROCM
                else if (target_device == DeviceType::ROCm)
                {
                    // NO FALLBACK: ROCm device-targeted kernels not yet implemented
                    throw std::runtime_error("ROCm device-targeted GEMM kernels not yet implemented - no silent fallback to CPU");
                }
#endif
                else
                {
                    throw std::runtime_error("Unsupported target device type: " + std::to_string(static_cast<int>(target_device)));
                }

                // We should always have a kernel if we get here (all error paths throw)
                if (!kernel)
                {
                    throw std::runtime_error("[KernelFactory] Internal error: Failed to create kernel for target device");
                }

                // Cache and return
                auto *raw_ptr = kernel.get();
                device_targeted_cache_[key] = std::move(kernel);
                LOG_DEBUG("[KernelFactory] Cached device-targeted kernel: tensor="
                          << static_cast<const void *>(tensor) << " device=" << static_cast<int>(target_device));
                return raw_ptr;
            }

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
                kernel_cache_.erase(tensor);

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
            }

            void KernelFactory::clearCache()
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);

                // Clean up all tensor packed weights caches (CPU VNNI and CUDA INT8)
                for (auto &pair : kernel_cache_)
                {
                    const auto *tensor = pair.first;
                    if (tensor)
                    {
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
                    }
                }

                kernel_cache_.clear();
                sliced_cache_.clear();
                device_targeted_cache_.clear();
            }

            std::pair<size_t, size_t> KernelFactory::cacheStats()
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                size_t total_bytes = 0;
                // Note: Can't easily compute packed_bytes without RTTI to QuantisedGemmKernel
                // For now, just return count (includes all caches)
                return {kernel_cache_.size() + sliced_cache_.size() + device_targeted_cache_.size(), total_bytes};
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
