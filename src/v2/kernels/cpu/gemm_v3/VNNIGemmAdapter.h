/**
 * @file VNNIGemmAdapter.h
 * @brief Adapter layer between Tensor API and low-level VNNI GEMM kernel
 * @author David Sanftenberg
 *
 * This adapter handles:
 * - Tensor format conversion (IActivationTensor/Q8_0Tensor → raw int8_t*)
 * - B-matrix packing to VNNI layout
 * - Scale extraction from quantized tensors
 * - Bias handling
 */

#pragma once

#include "VNNIGemm.h"
#include "tensors/Tensors.h"
#include <vector>
#include <cstring>

namespace llaminar2
{

    /**
     * @brief Adapter function template for VNNI GEMM kernels
     *
     * Converts high-level Tensor API to low-level kernel call with proper packing.
     *
     * @tparam M_R Micro-kernel M dimension
     * @tparam N_R Micro-kernel N dimension
     * @tparam K_BLK K block size
     * @tparam UNROLL_K K-loop unroll factor
     * @tparam PREFETCH_B_L1 L1 prefetch distance
     * @param M Number of rows in A and C
     * @param N Number of columns in B and C
     * @param K Number of columns in A / rows in B
     * @param A Activation tensor (FP32 or quantized)
     * @param B Q8_0 weight tensor
     * @param C Output FP32 matrix
     * @param ldc Leading dimension of C
     */
    template <int M_R, int N_R, int K_BLK, int UNROLL_K, int PREFETCH_B_L1>
    void vnni_gemm_adapter(
        int M, int N, int K,
        const IActivationTensor &A,
        const Q8_0Tensor &B,
        float *C, int ldc)
    {
        // This is a minimal stub adapter - full implementation requires:
        // 1. Proper INT8 quantization of activations
        // 2. Extraction of Q8_0 block data
        // 3. Packing to VNNI layout
        // 4. Scale extraction from blocks
        //
        // For now, just zero the output to prevent undefined behavior
        std::memset(C, 0, M * N * sizeof(float));

        // TODO: Implement full adapter logic:
        // - Use A.to_int8_perchannel() or similar for activation quantization
        // - Use B's ITensorGemmTileDataProvider interface to get blocks
        // - Pack B matrix using pack_B_panel_vnni<K_BLK>()
        // - Extract scales from Q8_0 blocks
        // - Call gemm_int8_vnni_kernel with proper parameters
    }

} // namespace llaminar2
