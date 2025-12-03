/**
 * @file KernelFactory.cpp
 * @brief Implementation of centralized kernel dispatch factory
 * @author David Sanftenberg
 */

#include "KernelFactory.h"
#include "cpu/gemm_v4/QuantisedGemmKernel.h"
#include "cpu/gemm_v4/FloatingPointGemmKernel.h"

#include "../tensors/Tensors.h"
#include "../backends/ComputeBackend.h"
#include "../utils/Logger.h"

// CUDA kernel factory
#ifdef HAVE_CUDA
#include "cuda/CudaGemmFactory.h"
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

            DeviceType KernelFactory::getDeviceType(int device_idx)
            {
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
                case llaminar2::ComputeBackendType::CPU_OPENBLAS:
                case llaminar2::ComputeBackendType::CPU_MKL:
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
                // Helper to throw for unsupported GPU backends
                [[noreturn]] void throwUnsupportedBackend(DeviceType dev_type, const char *tensor_type)
                {
                    LOG_ERROR("[KernelFactory] " << tensor_type << " GEMM not supported on "
                                                 << to_string(dev_type));
                    throw std::runtime_error(std::string(tensor_type) + " GEMM not supported on " +
                                             to_string(dev_type));
                }
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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return llaminar::v2::kernels::cuda::createCudaGemm(tensor);
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
                    return new llaminar2::gemm_v4::QuantisedGemmKernel(tensor);

#ifdef HAVE_CUDA
                case DeviceType::CUDA:
                    return llaminar::v2::kernels::cuda::createCudaGemmRaw(tensor);
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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

                    // GPU backends not yet implemented for Q4_0
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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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
                    return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(tensor);

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

                default:
                    if (dev_type != DeviceType::CPU)
                    {
                        throwUnsupportedBackend(dev_type, "BF16");
                    }
                    throw std::runtime_error("Unreachable");
                }
            }

            // ==========================================================================
            // Kernel Cache - Static Members
            // ==========================================================================

            std::unordered_map<const llaminar2::TensorBase *, std::unique_ptr<llaminar2::ITensorGemm>> KernelFactory::kernel_cache_;
            std::mutex KernelFactory::cache_mutex_;

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

                auto dev_type = getDeviceType(tensor->device_index());

                // For CPU quantized tensors, use tensor-owned packed weights pattern
                if (dev_type == DeviceType::CPU)
                {
                    // Check if tensor implements IINT8Unpackable (quantized weight tensor)
                    const auto *unpackable = dynamic_cast<const llaminar2::IINT8Unpackable *>(tensor);
                    if (unpackable)
                    {
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

                            LOG_DEBUG("[KernelFactory] Packed weights for tensor "
                                      << tensor << " (" << packed_cache->packed.N << "x" << packed_cache->packed.K << ")");
                        }

                        // Create lightweight kernel referencing tensor's packed data
                        auto kernel = std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(
                            &packed_cache->packed);

                        auto *raw_ptr = kernel.get();
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
                    // Type-based dispatch for GPU backends
                    if (auto *t = dynamic_cast<const llaminar2::IQ4_NLTensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q4_0Tensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q4_1Tensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q5_0Tensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q5_1Tensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q6_KTensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q8_0Tensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q8_1Tensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q2_KTensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q3_KTensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q4_KTensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q5_KTensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::Q8_KTensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::IQ1_MTensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::IQ1_STensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::IQ2_STensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::IQ2_XSTensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::IQ2_XXSTensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::IQ3_STensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::IQ3_XXSTensor *>(tensor))
                    {
                        kernel = createGemm(t, dev_type);
                    }
                    else if (auto *t = dynamic_cast<const llaminar2::IQ4_XSTensor *>(tensor))
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

            void KernelFactory::clearCacheFor(const llaminar2::TensorBase *tensor)
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                kernel_cache_.erase(tensor);

                // Also clean up tensor's packed weights cache if present
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
            }

            void KernelFactory::clearCache()
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);

                // Clean up all tensor packed weights caches
                for (auto &pair : kernel_cache_)
                {
                    const auto *tensor = pair.first;
                    if (tensor && tensor->cache_.has_value())
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
                }

                kernel_cache_.clear();
            }

            std::pair<size_t, size_t> KernelFactory::cacheStats()
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                size_t total_bytes = 0;
                // Note: Can't easily compute packed_bytes without RTTI to QuantisedGemmKernel
                // For now, just return count
                return {kernel_cache_.size(), total_bytes};
            }

        } // namespace kernels
    } // namespace v2
} // namespace llaminar
