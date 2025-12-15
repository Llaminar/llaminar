/**
 * @file CudaGemmFactory.cu
 * @brief CUDA GEMM kernel factory implementation
 *
 * Provides CudaGemmKernel wrapper that implements ITensorGemm using CUDA kernels.
 *
 * **Design**: Thin wrapper around CUDA kernels (analogous to CPU GemmKernel).
 * Does NOT manage device memory - expects all pointers already on device.
 * Memory management is the responsibility of the caller (typically the pipeline).
 *
 * @author David Sanftenberg
 */

#include "CudaGemmFactory.h"
#include "CudaGemmAutoTuner.h"
#include "CudaGemmVariantsBaseline.h"
#include "CudaGemmVariantsMemoryOpt.h"
#include "IQ4_NL_BlockDecoder.h"
#include <stdexcept>
#include <unordered_map>
#include <cstdlib>
#include <cuda_runtime.h>

// Forward declare the logging macros to avoid pulling in Logger.h dependencies
#define LOG_DEBUG(msg) \
    do                 \
    {                  \
    } while (0)
#define LOG_ERROR(msg) \
    do                 \
    {                  \
    } while (0)

// Full ITensorGemm definition needed for inheritance
// (can't include TensorKernels.h due to MPI dependencies)
namespace llaminar2
{
    class ITensorGemm
    {
    public:
        virtual ~ITensorGemm() = default;
        virtual bool multiply(const float *A, float *C, int m, int n, int k,
                              const std::unordered_map<std::string, float> &extra_params) = 0;
    };

    // Minimal tensor interface needed for implementation
    class IQ4_NLTensor
    {
    public:
        virtual int device_index() const = 0;
        virtual const uint8_t *raw_blocks() const = 0;

    protected:
        ~IQ4_NLTensor() = default; // Protected destructor for interface
    };
}

namespace llaminar
{
    namespace v2
    {
        namespace kernels
        {
            namespace cuda
            {
                /**
                 * @brief CUDA-accelerated GEMM kernel wrapper
                 *
                 * Implements ITensorGemm using CUDA kernel variants with auto-tuning.
                 *
                 * **Memory Expectations**:
                 * - A (input activations): Already on device
                 * - B (quantized weights): Already on device (IQ4_NL blocks)
                 * - C (output): Already on device
                 * - Caller manages all device memory allocation/deallocation
                 *
                 * **Auto-Tuning**:
                 * - Uses CudaGemmAutoTuner to select optimal kernel variant
                 * - Kernel selection cached per (m, n, k) shape
                 * - Variants tested: 200+ tile sizes, unroll factors
                 */
                class CudaGemmKernel : public llaminar2::ITensorGemm
                {
                public:
                    explicit CudaGemmKernel(const llaminar2::IQ4_NLTensor *tensor)
                        : tensor_(tensor)
                    {
                        if (!tensor_)
                        {
                            throw std::runtime_error("CudaGemmKernel: null tensor");
                        }

                        if (tensor_->device_index() < 0)
                        {
                            throw std::runtime_error("CudaGemmKernel: tensor must be on CUDA device");
                        }
                    }

                    ~CudaGemmKernel() override = default;

                    bool multiply(const float *A, float *C, int m, int n, int k,
                                  const std::unordered_map<std::string, float> &extra_params) override
                    {
                        try
                        {
                            // Get optimal kernel configuration from auto-tuner
                            auto &autotuner = llaminar2::cuda::CudaGemmAutoTuner::instance();
                            auto config = autotuner.getOptimalConfig(m, n, k);

                            // Get quantized weight blocks (already on device)
                            const void *B_blocks_device = tensor_->raw_blocks();
                            if (!B_blocks_device)
                            {
                                LOG_ERROR("[CudaGemmKernel] Tensor raw_blocks() returned null");
                                return false;
                            }

                            // Cast to IQ4_NL blocks
                            const llaminar2::cuda::IQ4_NLBlock *B_blocks =
                                reinterpret_cast<const llaminar2::cuda::IQ4_NLBlock *>(B_blocks_device);

                            // Check if we should use optimized kernel (Phase 1)
                            // Environment variable LLAMINAR_USE_OPTIMIZED_KERNEL=1 to enable
                            // Default: use baseline kernel (optimized kernel still experimental)
                            bool use_optimized = false;
                            const char *use_opt_env = std::getenv("LLAMINAR_USE_OPTIMIZED_KERNEL");
                            if (use_opt_env && std::atoi(use_opt_env) != 0)
                            {
                                use_optimized = true;
                            }

                            cudaError_t err;
                            if (use_optimized)
                            {
                                // Phase 1 optimized kernel (coalesced, vectorized, padded)
                                err = llaminar2::cuda::launchIQ4NLGemmVariantOptimized(
                                    A, B_blocks, C, m, n, k, config, nullptr);
                            }
                            else
                            {
                                // Baseline kernel
                                err = llaminar2::cuda::launchIQ4NLGemmVariant(
                                    A, B_blocks, C, m, n, k, config, nullptr);
                            }

                            if (err != cudaSuccess)
                            {
                                LOG_ERROR("[CudaGemmKernel] Kernel launch failed: "
                                          << cudaGetErrorString(err));
                                return false;
                            }

                            // Synchronize to ensure completion
                            err = cudaDeviceSynchronize();
                            if (err != cudaSuccess)
                            {
                                LOG_ERROR("[CudaGemmKernel] Device synchronize failed: "
                                          << cudaGetErrorString(err));
                                return false;
                            }

                            return true;
                        }
                        catch (const std::exception &e)
                        {
                            LOG_ERROR("[CudaGemmKernel] multiply failed: " << e.what());
                            return false;
                        }
                    }

                private:
                    const llaminar2::IQ4_NLTensor *tensor_;
                };

                // ========== Factory Function ==========

                std::unique_ptr<llaminar2::ITensorGemm> createCudaGemm(
                    const llaminar2::IQ4_NLTensor *tensor)
                {
                    return std::make_unique<CudaGemmKernel>(tensor);
                }

            } // namespace cuda
        } // namespace kernels
    } // namespace v2
} // namespace llaminar
