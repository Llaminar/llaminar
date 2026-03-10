/**
 * @file CUDANativePayloadGemmKernels.cu
 * @brief CUDA native-payload GEMM kernels for blockwise prefill (M>1).
 *
 * This is a correctness-first prefill path that preserves compact native payload
 * storage and decodes 32-element blocks directly inside the CUDA kernel.
 *
 * Initial supported formats:
 * - Q4_0   (codebook 0)
 * - IQ4_NL (codebook 4)
 * - Q4_1   (codebook 5)
 * - Q5_0   (codebook 6)
 * - Q5_1   (codebook 7)
 */

#include "kernels/cuda/CUDANativePayloadDecodeCommon.cuh"

#include <cstdint>

namespace
{
    static constexpr int NATIVE_GEMM_BLOCK_K = 32;

    template <uint8_t CODEBOOK_ID>
    __global__ void cudaNativePayloadGemmKernel(
        const int8_t * __restrict__ d_A_int8,
        const uint8_t * __restrict__ d_payload,
        const uint16_t * __restrict__ d_scales,
        const uint16_t * __restrict__ d_mins,
        const uint32_t * __restrict__ d_emins,
        float * __restrict__ d_C_fp32,
        const float * __restrict__ d_scales_A_blockwise,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float * __restrict__ d_C_existing,
        const float * __restrict__ d_bias)
    {
        const int n = blockIdx.x * blockDim.x + threadIdx.x;
        const int m = blockIdx.y * blockDim.y + threadIdx.y;
        if (m >= M || n >= N)
            return;

        const int blocks_per_row = K / NATIVE_GEMM_BLOCK_K;
        const int8_t *row_a = d_A_int8 + static_cast<size_t>(m) * K;
        const float *row_scales_a = d_scales_A_blockwise + static_cast<size_t>(m) * blocks_per_row;
        float acc = 0.0f;

        for (int blk = 0; blk < blocks_per_row; ++blk)
        {
            const size_t linear = static_cast<size_t>(blk) * N + static_cast<size_t>(n);
            const uint8_t *payload = d_payload + linear * llaminar2::cuda_native_payload::payload_bytes_for_codebook<CODEBOOK_ID>();
            const float scale_a = row_scales_a[blk];
            const int8_t *block_a = row_a + blk * NATIVE_GEMM_BLOCK_K;
            int32_t packed_groups[8];
            int32_t a_pack[8];
            llaminar2::cuda_native_payload::decode_groups<CODEBOOK_ID>(payload, packed_groups);

            #pragma unroll
            for (int g = 0; g < 8; ++g)
            {
                a_pack[g] = *reinterpret_cast<const int32_t *>(block_a + g * 4);
            }

            if constexpr (llaminar2::cuda_native_payload::CodebookTraits<CODEBOOK_ID>::is_dual_scale)
            {
                int dot_lo = 0;
                int dot_hi = 0;
                int sum_lo = 0;
                int sum_hi = 0;

                #pragma unroll
                for (int g = 0; g < 4; ++g)
                {
                    dot_lo = __dp4a(a_pack[g], packed_groups[g], dot_lo);
                    dot_hi = __dp4a(a_pack[g + 4], packed_groups[g + 4], dot_hi);
                    sum_lo += llaminar2::cuda_native_payload::sum_packed_i8(a_pack[g]);
                    sum_hi += llaminar2::cuda_native_payload::sum_packed_i8(a_pack[g + 4]);
                }

                const float scale_lo = llaminar2::cuda_native_payload::fp16_bits_to_float(d_scales[linear]);
                const float scale_hi = d_mins ? llaminar2::cuda_native_payload::fp16_bits_to_float(d_mins[linear]) : 0.0f;
                acc += scale_a * (scale_lo * static_cast<float>(dot_lo) + scale_hi * static_cast<float>(dot_hi));

                if constexpr (llaminar2::cuda_native_payload::CodebookTraits<CODEBOOK_ID>::is_dual_scale_asym)
                {
                    const uint32_t emin_packed = d_emins ? d_emins[linear] : 0u;
                    const float min_lo = llaminar2::cuda_native_payload::fp16_bits_to_float(static_cast<uint16_t>(emin_packed));
                    const float min_hi = llaminar2::cuda_native_payload::fp16_bits_to_float(static_cast<uint16_t>(emin_packed >> 16));
                    acc += scale_a * (min_lo * static_cast<float>(sum_lo) + min_hi * static_cast<float>(sum_hi));
                }

                if constexpr (llaminar2::cuda_native_payload::CodebookTraits<CODEBOOK_ID>::is_iq1_m)
                {
                    constexpr float IQ1S_DELTA = 0.125f;
                    const uint8_t qh0 = payload[4];
                    const uint8_t qh1 = payload[5];
                    const int sum_g0 = llaminar2::cuda_native_payload::sum_packed_i8(a_pack[0]) + llaminar2::cuda_native_payload::sum_packed_i8(a_pack[1]);
                    const int sum_g1 = llaminar2::cuda_native_payload::sum_packed_i8(a_pack[2]) + llaminar2::cuda_native_payload::sum_packed_i8(a_pack[3]);
                    const int sum_g2 = llaminar2::cuda_native_payload::sum_packed_i8(a_pack[4]) + llaminar2::cuda_native_payload::sum_packed_i8(a_pack[5]);
                    const int sum_g3 = llaminar2::cuda_native_payload::sum_packed_i8(a_pack[6]) + llaminar2::cuda_native_payload::sum_packed_i8(a_pack[7]);
                    const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                    const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                    const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                    const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                    acc += scale_a * ((d0 * static_cast<float>(sum_g0) + d1 * static_cast<float>(sum_g1)) * scale_lo +
                                      (d2 * static_cast<float>(sum_g2) + d3 * static_cast<float>(sum_g3)) * scale_hi);
                }
            }
            else
            {
                int dot = 0;
                int sum_a = 0;
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    dot = __dp4a(a_pack[g], packed_groups[g], dot);
                    sum_a += llaminar2::cuda_native_payload::sum_packed_i8(a_pack[g]);
                }

                const float scale_b = llaminar2::cuda_native_payload::fp16_bits_to_float(d_scales[linear]);
                acc += scale_a * scale_b * static_cast<float>(dot);

                if constexpr (llaminar2::cuda_native_payload::CodebookTraits<CODEBOOK_ID>::is_asymmetric)
                {
                    const float min_b = d_mins ? llaminar2::cuda_native_payload::fp16_bits_to_float(d_mins[linear]) : 0.0f;
                    acc += scale_a * min_b * static_cast<float>(sum_a);
                }
            }
        }

        const size_t out_idx = static_cast<size_t>(m) * N + static_cast<size_t>(n);
        float out = alpha * acc;
        if (beta != 0.0f && d_C_existing)
            out += beta * d_C_existing[out_idx];
        if (d_bias)
            out += d_bias[n];
        d_C_fp32[out_idx] = out;
    }
}

extern "C"
{
    bool cudaNativePayloadInitIQGridTables_gemm()
    {
        return llaminar2::cuda_native_payload::initIQGridTables();
    }

    bool cudaNativePayloadGemm_fp32(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        const uint16_t *d_mins,
        const uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_blockwise,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        uint8_t codebook_id,
        int cuda_device_id,
        void *stream)
    {
        if (!d_A_int8 || !d_payload || !d_scales || !d_C_fp32 || !d_scales_A_blockwise)
            return false;
        if (M <= 1 || N <= 0 || K <= 0 || (K % NATIVE_GEMM_BLOCK_K) != 0)
            return false;

        cudaError_t err = cudaSetDevice(cuda_device_id);
        if (err != cudaSuccess)
            return false;

        const dim3 block(16, 8);
        const dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        switch (codebook_id)
        {
        case 0:
            cudaNativePayloadGemmKernel<0><<<grid, block, 0, cuda_stream>>>(
                d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_blockwise,
                M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 4:
            cudaNativePayloadGemmKernel<4><<<grid, block, 0, cuda_stream>>>(
                d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_blockwise,
                M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 5:
            cudaNativePayloadGemmKernel<5><<<grid, block, 0, cuda_stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_blockwise,
                M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 6:
            cudaNativePayloadGemmKernel<6><<<grid, block, 0, cuda_stream>>>(
                d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_blockwise,
                M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 7:
            cudaNativePayloadGemmKernel<7><<<grid, block, 0, cuda_stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_blockwise,
                M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 8:
            cudaNativePayloadGemmKernel<8><<<grid, block, 0, cuda_stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_blockwise,
                M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 9:
            cudaNativePayloadGemmKernel<9><<<grid, block, 0, cuda_stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_blockwise,
                M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 10:
            cudaNativePayloadGemmKernel<10><<<grid, block, 0, cuda_stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C_fp32, d_scales_A_blockwise,
                M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 11:
            cudaNativePayloadGemmKernel<11><<<grid, block, 0, cuda_stream>>>(
                d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_blockwise,
                M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 12:
            cudaNativePayloadGemmKernel<12><<<grid, block, 0, cuda_stream>>>(
                d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_blockwise,
                M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 13:
            cudaNativePayloadGemmKernel<13><<<grid, block, 0, cuda_stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_blockwise,
                M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 14:
            cudaNativePayloadGemmKernel<14><<<grid, block, 0, cuda_stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_blockwise,
                M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 15:
            cudaNativePayloadGemmKernel<15><<<grid, block, 0, cuda_stream>>>(
                d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_blockwise,
                M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 16:
            cudaNativePayloadGemmKernel<16><<<grid, block, 0, cuda_stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_blockwise,
                M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 17:
            cudaNativePayloadGemmKernel<17><<<grid, block, 0, cuda_stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_blockwise,
                M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        default:
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }
}
