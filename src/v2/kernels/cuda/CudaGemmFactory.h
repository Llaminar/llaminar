/**
 * @file CudaGemmFactory.h
 * @brief CUDA GEMM kernel factory for IQ4_NL tensors
 *
 * Provides CUDA-accelerated GEMM kernels with auto-tuning, analogous to
 * CPU's GemmAutoTuner.h but for GPU execution.
 *
 * **Design**:
 * - Analogous to cpu/GemmAutoTuner.h::createAutoTunedGemm()
 * - Returns ITensorGemm implementation backed by CUDA kernels
 * - Uses CudaGemmAutoTuner for optimal kernel variant selection
 * - Integrates with IBackend for device memory operations
 *
 * **Usage**:
 * ```cpp
 * // In IQ4_NLTensor::createGemm():
 * if (device_idx_ >= 0) {  // CUDA device
 *     return llaminar::v2::kernels::cuda::createCudaGemm(this);
 * } else {  // CPU
 *     return llaminar::v2::kernels::createAutoTunedGemm(this);
 * }
 * ```
 *
 * @author David Sanftenberg
 */

#pragma once

#include <memory>

// Forward declarations to avoid circular dependencies
namespace llaminar2
{
    class IQ4_NLTensor;
    class ITensorGemm; // Forward declare - defined in TensorKernels.h
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
                 * @brief Create auto-tuned CUDA GEMM kernel for IQ4_NL tensors
                 *
                 * Returns a CUDA-accelerated ITensorGemm implementation that:
                 * 1. Uses CudaGemmAutoTuner to select optimal kernel variant
                 * 2. Expects weight tensor (B matrix) already on device
                 * 3. Expects input (A) and output (C) already on device
                 * 4. Does NOT manage device memory - caller's responsibility
                 *
                 * **Performance**:
                 * - Auto-tunes across 200+ kernel variants (tile sizes, unroll factors)
                 * - Achieves 193-1539 GFLOPS on RTX 3090 (matrix size dependent)
                 * - Single token: ~193 GFLOPS (128×128)
                 * - Large prefill: ~1539 GFLOPS (512×512)
                 *
                 * **Requirements**:
                 * - Tensor must be on CUDA device (device_index() >= 0)
                 * - Tensor's raw_blocks() must point to device memory
                 * - CUDA kernel variants must be compiled
                 *
                 * **Memory Management**:
                 * - All pointers (A, B, C) must already be on device
                 * - Caller responsible for device allocation/deallocation
                 * - Analogous to CPU GemmKernel design
                 *
                 * @param tensor IQ4_NL tensor (quantized weights, typically B matrix)
                 * @return ITensorGemm instance backed by CUDA kernels
                 *
                 * @throws std::runtime_error if tensor not on CUDA device
                 */
                std::unique_ptr<llaminar2::ITensorGemm> createCudaGemm(
                    const llaminar2::IQ4_NLTensor *tensor);

            } // namespace cuda
        } // namespace kernels
    } // namespace v2
} // namespace llaminar
