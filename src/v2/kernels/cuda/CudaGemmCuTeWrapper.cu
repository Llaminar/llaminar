/**
 * @file CudaGemmCuTeWrapper.cu
 * @brief CUDA compilation unit for CuTe Tensor Core GEMM template
 *
 * This wrapper provides C linkage for the CuTe template,
 * allowing it to be called from C++ test code.
 *
 * @author David Sanftenberg
 * @date November 3, 2025
 */

#include "CudaGemmKernelTemplateCuTe.h"

extern "C"
{

    /**
     * @brief C-linkage wrapper for CuTe GEMM launcher
     *
     * Instantiates the template with default parameters:
     * - TILE_M=32, TILE_N=32, TILE_K=32
     * - MMA_M=1, MMA_N=1 (1×1 atom layout)
     * - THREADS_PER_BLOCK=32 (1 warp)
     */
    void launch_iq4nl_gemm_cute_default(
        const float *A,
        const IQ4_NLBlock *B_blocks,
        float *C,
        int M, int N, int K,
        cudaStream_t stream)
    {
        launch_iq4nl_gemm_cute<32, 32, 32, 1, 1, 32>(
            A, B_blocks, C, M, N, K, stream);
    }

} // extern "C"
