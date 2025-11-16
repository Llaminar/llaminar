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
#include "ActivationPackingAdapters.h"
#include "WeightPackingAdapters.h"
#include "tensors/Tensors.h"
#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace llaminar2
{

    /**
     * @brief Adapter function template for VNNI GEMM kernels
     *
     * Converts high-level Tensor API to low-level kernel call with proper packing.
     * First-cut implementation assumes:
     * - FP32 activations (A)
     * - Q8_0 quantized weights (B)
     * - Symmetric per-row activation quantization
     * - Symmetric per-column weight quantization
     * - No zero-points (symmetric quantization only)
     * - M, N, K must be multiples of M_R, N_R, K_BLK (edge tiles not yet implemented)
     *
     * @tparam M_R Micro-kernel M dimension
     * @tparam N_R Micro-kernel N dimension
     * @tparam K_BLK K block size
     * @tparam UNROLL_K K-loop unroll factor
     * @tparam PREFETCH_B_L1 L1 prefetch distance
     * @param M Number of rows in A and C
     * @param N Number of columns in B and C
     * @param K Number of columns in A / rows in B
     * @param A Activation tensor (FP32)
     * @param B Q8_0 weight tensor
     * @param C Output FP32 matrix (row-major, M x N)
     * @param ldc Leading dimension of C (should be >= N)
     * @param bias Optional bias vector [N], nullptr if not used
     */
    template <int M_R, int N_R, int K_BLK, int UNROLL_K, int PREFETCH_B_L1>
    void vnni_gemm_adapter(
        int M, int N, int K,
        const IActivationTensor &A,
        const Q8_0Tensor &B,
        float *C, int ldc,
        const float *bias = nullptr)
    {
        static_assert(M_R % 4 == 0, "M_R must be multiple of 4");
        static_assert(N_R % 16 == 0, "N_R must be multiple of 16");
        static_assert(K_BLK % 4 == 0, "K_BLK must be multiple of 4");

        // First-cut limitation: dimensions must be multiples of tile sizes
        // TODO: Handle edge tiles properly
        if (M % M_R != 0 || N % N_R != 0 || K % K_BLK != 0)
        {
            // For now, zero output and return
            // In production, we'd handle partial tiles
            for (int i = 0; i < M; ++i)
            {
                std::memset(C + i * ldc, 0, N * sizeof(float));
            }
            return;
        }

        const int T = K / K_BLK; // Number of K blocks

        // Pack weights once for entire GEMM (B is reused across all M tiles)
        std::vector<int8_t> B_packed_storage;
        PackedB Bp;
        std::vector<float> wgt_scales;
        pack_q8_0_weights_to_vnni_format<K_BLK>(B, K, N, B_packed_storage, Bp, wgt_scales);

        // Prepare bias (if null, use zeros)
        std::vector<float> bias_vec;
        const float *bias_ptr = bias;
        if (!bias)
        {
            bias_vec.resize(N, 0.0f);
            bias_ptr = bias_vec.data();
        }

        // Convert A to FP32 row-major (temporary for first cut)
        // Use TensorBase conversion which all IActivationTensor implementations have
        std::vector<float> A_fp32(M * K);
        const TensorBase *A_base = dynamic_cast<const TensorBase *>(&A);
        if (!A_base)
        {
            // Fallback: zero the output
            for (int i = 0; i < M; ++i)
            {
                std::memset(C + i * ldc, 0, N * sizeof(float));
            }
            return;
        }
        A_base->to_fp32_span(0, M * K, A_fp32.data());

        // Allocate scratch buffers for packed activations
        const int K_chunks = K_BLK / 4;
        const int num_groups = M_R / 4;
        const int group_stride = K_chunks * 16;
        const int A_block_bytes = num_groups * group_stride;
        const int A_tile_total_bytes = A_block_bytes * T;

        std::vector<int8_t> A_tile_packed(A_tile_total_bytes);
        std::vector<float> act_scales(M_R);

        // Loop over M tiles
        for (int M0 = 0; M0 < M; M0 += M_R)
        {
            const int mr = M_R; // First-cut: assume full tiles only

            // Pack A for all K blocks for this M_R tile
            for (int t = 0; t < T; ++t)
            {
                const int k0 = t * K_BLK;
                int8_t *A_block_tile = A_tile_packed.data() + t * A_block_bytes;

                // Pack this [M_R x K_BLK] tile of A
                pack_fp32_activations_to_4x4_grouped<M_R, K_BLK>(
                    A_fp32.data(),
                    M, K,
                    M0, k0,
                    mr, K_BLK,
                    A_block_tile,
                    act_scales.data());
            }

            // Call VNNI kernel for this M tile across all N
            // Note: kernel handles N tiling internally
            gemm_int8_vnni_kernel<
                M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1,
                64,    // PREFETCH_B_L2 (default)
                false, // ACCUM_INT32 (false = FP32 accumulation)
                false, // USE_L2_PREFETCH (disabled for now)
                true   // USE_VNNI (enabled)
                >(
                A_tile_packed.data(),
                Bp,
                C + M0 * ldc, // Output pointer for this M tile
                bias_ptr,
                act_scales.data(),
                wgt_scales.data(),
                mr, N, K);
        }
    }

} // namespace llaminar2
