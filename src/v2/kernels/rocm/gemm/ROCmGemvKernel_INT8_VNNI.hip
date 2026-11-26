/**
 * @file ROCmGemvKernel_INT8_VNNI.hip
 * @brief ROCm INT8 VNNI direct, blockwise, grid-K-parallel, LDS K-reduction, and prefill GEMV/GEMM kernels.
 */
#include "ROCmGemvKernel_INT8_VNNI_common.hip.inc"

namespace llaminar2::rocm::int8_vnni_internal {
std::atomic<int> g_int8_vnni_tn_override{-1};
std::atomic<int> g_int8_vnni_kb_override{-1};
std::atomic<int> g_int8_vnni_wk_override{-1};
std::atomic<int> g_int8_vnni_unroll_override{-1};  // 1=standard, 2=2x act_block unroll
std::atomic<int> g_int8_vnni_act_block_override{-1};  // -1=default(32), 32, 64, 128
std::atomic<int> g_int8_vnni_wide_tn_override{-1};
std::atomic<int> g_int8_vnni_wide_cpt_override{-1};
std::atomic<int> g_int8_vnni_wide_vec_load_override{-1};
std::atomic<int> g_int8_vnni_skip_memset_override{0};  // 1 = skip hipMemsetAsync (for benchmarking)
std::atomic<int> g_int8_vnni_force_pair_override{0};  // 1 = force batched pair path (skip split-to-singles)
}

template <int TILE_N, int CPT>
__global__ void gemv_init_output_kernel_t(
    float* d_C_fp32,
    const float* d_C_existing,
    const float* __restrict__ d_bias,
    int N,
    float beta)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N)
        {
            return;
        }

        float result = 0.0f;
        if (beta != 0.0f && d_C_existing)
        {
            result += beta * d_C_existing[n];
        }
        if (d_bias)
        {
            result += d_bias[n];
        }
        d_C_fp32[n] = result;
    }
}

template <int TILE_N, int CPT>
__global__ void gemv_init_output_batched_kernel_t(
    GemvBatchArgs args,
    float beta)
{
    const int batch = gemv_find_batch_for_block(args, blockIdx.x);
    const int local_block_x = blockIdx.x - args.grid_n_offset[batch];
    const int n_base = local_block_x * TILE_N + threadIdx.x * CPT;

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= args.N[batch])
        {
            return;
        }

        float result = 0.0f;
        if (beta != 0.0f)
        {
            result += beta * args.d_C[batch][n];
        }
        if (args.d_bias[batch])
        {
            result += args.d_bias[batch][n];
        }
        args.d_C[batch][n] = result;
    }
}

template <int TILE_N, int CPT>
__global__ void gemv_init_output_pair_kernel_t(
    float* __restrict__ d_C0,
    float* __restrict__ d_C1,
    const float* __restrict__ d_bias0,
    const float* __restrict__ d_bias1,
    int N0,
    int N1,
    int grid_n0,
    float beta)
{
    const bool second = (blockIdx.x >= grid_n0);
    const int local_block_x = second ? (blockIdx.x - grid_n0) : blockIdx.x;
    const int n_base = local_block_x * TILE_N + threadIdx.x * CPT;
    float* __restrict__ d_C = second ? d_C1 : d_C0;
    const float* __restrict__ d_bias = second ? d_bias1 : d_bias0;
    const int N = second ? N1 : N0;

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N)
        {
            return;
        }

        float result = 0.0f;
        if (beta != 0.0f)
        {
            result += beta * d_C[n];
        }
        if (d_bias)
        {
            result += d_bias[n];
        }
        d_C[n] = result;
    }
}

__global__ void gemv_int8_int8_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_TILE_N + threadIdx.x;
    if (n >= N) return;

    int32_t acc = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        // Load 4 INT8 activations as packed int32 for v_dot4_i32_i8
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + n) * 4);

        // Hardware INT8 dot product: 4 multiply-adds in 1 cycle (v_dot4_i32_i8)
        acc = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, acc);
    }

    d_C_int32[n] = acc;
}

/**
 * @brief 2D INT8 VNNI prefill kernel (M>1): computes C[M,N] in one launch.
 *
 * Thread mapping:
 * - threadIdx.y / blockIdx.y choose output row `m`
 * - threadIdx.x / blockIdx.x choose output column `n`
 *
 * For each output element C[m,n], we iterate over K in groups of 4 bytes and
 * use `v_dot4_i32_i8` (`__builtin_amdgcn_sdot4`) for efficient INT8 MAC.
 *
 * Layout reminder for junior developers:
 * - A is row-major [M x K]
 * - B is VNNI-packed [K/4 x N x 4]
 * - C is row-major [M x N]
 */
template <int BLOCK_X, int BLOCK_Y, int CPT>
__global__ void gemm_int8_int8_vnni_prefill_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K)
{
    const int n_base = blockIdx.x * (BLOCK_X * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (m >= M || n_base >= N)
    {
        return;
    }

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc[c] = 0;
    }

    const int k_groups = K / 4;

    // Compute one output element by reducing across K.
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K + kg * 4);

        if constexpr (CPT == 4)
        {
            if ((n_base + 3) < N)
            {
                const uint4 b_pack4 = *reinterpret_cast<const uint4*>(
                    d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, static_cast<int32_t>(b_pack4.x), acc[0]);
                acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, static_cast<int32_t>(b_pack4.y), acc[1]);
                acc[2] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, static_cast<int32_t>(b_pack4.z), acc[2]);
                acc[3] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, static_cast<int32_t>(b_pack4.w), acc[3]);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                            d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, acc[c]);
                    }
                }
            }
        }
        else if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                const uint2 b_pack2 = *reinterpret_cast<const uint2*>(
                    d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, static_cast<int32_t>(b_pack2.x), acc[0]);
                acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, static_cast<int32_t>(b_pack2.y), acc[1]);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                            d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, acc[c]);
                    }
                }
            }
        }
        else
        {
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
            acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, acc[0]);
        }
    }

    if constexpr (CPT == 4)
    {
        if ((n_base + 3) < N)
        {
            *reinterpret_cast<uint4*>(d_C_int32 + static_cast<int64_t>(m) * N + n_base) =
                make_uint4(static_cast<uint32_t>(acc[0]), static_cast<uint32_t>(acc[1]), static_cast<uint32_t>(acc[2]), static_cast<uint32_t>(acc[3]));
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    d_C_int32[static_cast<int64_t>(m) * N + n] = acc[c];
                }
            }
        }
    }
    else if constexpr (CPT == 2)
    {
        if ((n_base + 1) < N)
        {
            *reinterpret_cast<uint2*>(d_C_int32 + static_cast<int64_t>(m) * N + n_base) =
                make_uint2(static_cast<uint32_t>(acc[0]), static_cast<uint32_t>(acc[1]));
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    d_C_int32[static_cast<int64_t>(m) * N + n] = acc[c];
                }
            }
        }
    }
    else
    {
        d_C_int32[static_cast<int64_t>(m) * N + n_base] = acc[0];
    }
}

/**
 * @brief 2D INT8 VNNI prefill kernel with grid-kpar split-K accumulation.
 *
 * grid.z partitions K into slices. Each z-slice computes a partial sum for
 * C[m,n] and atomically accumulates into INT32 output.
 */
template <int BLOCK_X, int BLOCK_Y, int CPT>
__global__ void gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int k_groups_per_slice)
{
    const int n_base = blockIdx.x * (BLOCK_X * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BLOCK_Y + threadIdx.y;
    const int split_idx = blockIdx.z;

    if (m >= M || n_base >= N)
    {
        return;
    }

    const int total_k_groups = K / 4;
    const int k_begin = split_idx * k_groups_per_slice;
    if (k_begin >= total_k_groups)
    {
        return;
    }

    const int k_end = min(total_k_groups, k_begin + k_groups_per_slice);
    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc[c] = 0;
    }

    for (int kg = k_begin; kg < k_end; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K + kg * 4);

        if constexpr (CPT == 4)
        {
            if ((n_base + 3) < N)
            {
                const uint4 b_pack4 = *reinterpret_cast<const uint4*>(
                    d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, static_cast<int32_t>(b_pack4.x), acc[0]);
                acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, static_cast<int32_t>(b_pack4.y), acc[1]);
                acc[2] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, static_cast<int32_t>(b_pack4.z), acc[2]);
                acc[3] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, static_cast<int32_t>(b_pack4.w), acc[3]);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                            d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, acc[c]);
                    }
                }
            }
        }
        else if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                const uint2 b_pack2 = *reinterpret_cast<const uint2*>(
                    d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, static_cast<int32_t>(b_pack2.x), acc[0]);
                acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, static_cast<int32_t>(b_pack2.y), acc[1]);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                            d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, acc[c]);
                    }
                }
            }
        }
        else
        {
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
            acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, acc[0]);
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N && acc[c] != 0)
        {
            atomicAdd(&d_C_int32[static_cast<int64_t>(m) * N + n], acc[c]);
        }
    }
}


/**
 * @brief 2D payload-v2 bitwidth-8 prefill kernel (M>1): computes C[M,N].
 *
 * Layout:
 * - A is row-major INT8 [M x K]
 * - Payload is [K/32 x N x 32 bytes] where each record stores 32 signed INT8
 *   values for one (block, output-column)
 * - C is row-major INT32 [M x N]
 */
__global__ void gemm_payload_v2_i8_prefill_kernel(
    const int8_t* __restrict__ d_A_int8,
    const uint8_t* __restrict__ d_payload,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int kblocks)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const bool valid_n = (n < N);
    const bool valid_m = (m < M);

    const int blocks_per_row = K / 32;
    const int kb = max(1, kblocks);
    const int b_per = (blocks_per_row + kb - 1) / kb;
    const int b0 = min(blocks_per_row, static_cast<int>(blockIdx.z) * b_per);
    const int b1 = min(blocks_per_row, b0 + b_per);

    if (b0 >= b1)
    {
        return;
    }

    int32_t acc = 0;
    extern __shared__ int32_t shared_payload[]; // [blockDim.x * 8]

    for (int b = b0; b < b1; ++b)
    {
        if (threadIdx.y == 0)
        {
            int32_t* payload_dst = shared_payload + static_cast<int>(threadIdx.x) * 8;
            if (valid_n)
            {
                const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
                const uint4 packed0 = *reinterpret_cast<const uint4*>(d_payload + linear * 32u);
                const uint4 packed1 = *reinterpret_cast<const uint4*>(d_payload + linear * 32u + 16u);
                payload_dst[0] = static_cast<int32_t>(packed0.x);
                payload_dst[1] = static_cast<int32_t>(packed0.y);
                payload_dst[2] = static_cast<int32_t>(packed0.z);
                payload_dst[3] = static_cast<int32_t>(packed0.w);
                payload_dst[4] = static_cast<int32_t>(packed1.x);
                payload_dst[5] = static_cast<int32_t>(packed1.y);
                payload_dst[6] = static_cast<int32_t>(packed1.z);
                payload_dst[7] = static_cast<int32_t>(packed1.w);
            }
            else
            {
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    payload_dst[g] = 0;
                }
            }
        }

        __syncthreads();

        if (valid_m && valid_n)
        {
            const int32_t* a4 = reinterpret_cast<const int32_t*>(
                d_A_int8 + static_cast<int64_t>(m) * K + b * 32);
            const int32_t* payload_i32 = shared_payload + static_cast<int>(threadIdx.x) * 8;

            #pragma unroll
            for (int g = 0; g < 8; ++g)
            {
                acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], payload_i32[g], acc);
            }
        }

        __syncthreads();
    }

    if (valid_m && valid_n)
    {
        const int64_t out_idx = static_cast<int64_t>(m) * N + n;
        if (kb == 1)
        {
            d_C_int32[out_idx] = acc;
        }
        else
        {
            atomicAdd(&d_C_int32[out_idx], acc);
        }
    }
}

__global__ void gemm_payload_v2_i8_prefill_kernel_direct(
    const int8_t* __restrict__ d_A_int8,
    const uint8_t* __restrict__ d_payload,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int kblocks)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N)
    {
        return;
    }

    const int blocks_per_row = K / 32;
    const int kb = max(1, kblocks);
    const int b_per = (blocks_per_row + kb - 1) / kb;
    const int b0 = min(blocks_per_row, static_cast<int>(blockIdx.z) * b_per);
    const int b1 = min(blocks_per_row, b0 + b_per);

    if (b0 >= b1)
    {
        return;
    }

    int32_t acc = 0;

    for (int b = b0; b < b1; ++b)
    {
        const int32_t* a4 = reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K + b * 32);

        const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
        const uint4 packed0 = *reinterpret_cast<const uint4*>(d_payload + linear * 32u);
        const uint4 packed1 = *reinterpret_cast<const uint4*>(d_payload + linear * 32u + 16u);

        acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[0], static_cast<int32_t>(packed0.x), acc);
        acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[1], static_cast<int32_t>(packed0.y), acc);
        acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[2], static_cast<int32_t>(packed0.z), acc);
        acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[3], static_cast<int32_t>(packed0.w), acc);
        acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[4], static_cast<int32_t>(packed1.x), acc);
        acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[5], static_cast<int32_t>(packed1.y), acc);
        acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[6], static_cast<int32_t>(packed1.z), acc);
        acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[7], static_cast<int32_t>(packed1.w), acc);
    }

    const int64_t out_idx = static_cast<int64_t>(m) * N + n;
    if (kb == 1)
    {
        d_C_int32[out_idx] = acc;
    }
    else
    {
        atomicAdd(&d_C_int32[out_idx], acc);
    }
}

template <int BLOCK_X, int BLOCK_Y, int CPT>
__global__ void gemm_payload_v2_i8_prefill_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const uint8_t* __restrict__ d_payload,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int kblocks)
{
    const int n_base = blockIdx.x * (BLOCK_X * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (m >= M || n_base >= N)
    {
        return;
    }

    const int blocks_per_row = K / 32;
    const int kb = max(1, kblocks);
    const int b_per = (blocks_per_row + kb - 1) / kb;
    const int b0 = min(blocks_per_row, static_cast<int>(blockIdx.z) * b_per);
    const int b1 = min(blocks_per_row, b0 + b_per);

    if (b0 >= b1)
    {
        return;
    }

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc[c] = 0;
    }

    for (int b = b0; b < b1; ++b)
    {
        const int32_t* a4 = reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K + b * 32);

        const size_t linear0 = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n_base);
        const int32_t* payload_i32_0 = reinterpret_cast<const int32_t*>(d_payload + linear0 * 32u);

        if constexpr (CPT == 4)
        {
            if ((n_base + 3) < N)
            {
                const int32_t* payload_i32_1 = payload_i32_0 + 8;
                const int32_t* payload_i32_2 = payload_i32_1 + 8;
                const int32_t* payload_i32_3 = payload_i32_2 + 8;

                #pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    const int32_t a = a4[g];
                    acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a, payload_i32_0[g], acc[0]);
                    acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a, payload_i32_1[g], acc[1]);
                    acc[2] = ::llaminar2::rocm_integer_contract::signedDot4(a, payload_i32_2[g], acc[2]);
                    acc[3] = ::llaminar2::rocm_integer_contract::signedDot4(a, payload_i32_3[g], acc[3]);
                }
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
                        const int32_t* payload_i32 = reinterpret_cast<const int32_t*>(d_payload + linear * 32u);
                        #pragma unroll
                        for (int g = 0; g < 8; ++g)
                        {
                            acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], payload_i32[g], acc[c]);
                        }
                    }
                }
            }
        }
        else if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                const int32_t* payload_i32_1 = payload_i32_0 + 8;
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    const int32_t a = a4[g];
                    acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a, payload_i32_0[g], acc[0]);
                    acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a, payload_i32_1[g], acc[1]);
                }
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
                        const int32_t* payload_i32 = reinterpret_cast<const int32_t*>(d_payload + linear * 32u);
                        #pragma unroll
                        for (int g = 0; g < 8; ++g)
                        {
                            acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], payload_i32[g], acc[c]);
                        }
                    }
                }
            }
        }
        else
        {
            #pragma unroll
            for (int g = 0; g < 8; ++g)
            {
                acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], payload_i32_0[g], acc[0]);
            }
        }
    }

    const int64_t out_base = static_cast<int64_t>(m) * N + n_base;
    if (kb == 1)
    {
        if constexpr (CPT == 4)
        {
            if ((n_base + 3) < N)
            {
                *reinterpret_cast<uint4*>(d_C_int32 + out_base) = make_uint4(
                    static_cast<uint32_t>(acc[0]),
                    static_cast<uint32_t>(acc[1]),
                    static_cast<uint32_t>(acc[2]),
                    static_cast<uint32_t>(acc[3]));
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        d_C_int32[out_base + c] = acc[c];
                    }
                }
            }
        }
        else if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                *reinterpret_cast<uint2*>(d_C_int32 + out_base) = make_uint2(
                    static_cast<uint32_t>(acc[0]),
                    static_cast<uint32_t>(acc[1]));
            }
            else
            {
                d_C_int32[out_base] = acc[0];
                if ((n_base + 1) < N)
                {
                    d_C_int32[out_base + 1] = acc[1];
                }
            }
        }
        else
        {
            d_C_int32[out_base] = acc[0];
        }
    }
    else
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n < N)
            {
                atomicAdd(&d_C_int32[out_base + c], acc[c]);
            }
        }
    }
}

template <int TILE_N, int CPT, bool VEC_PACKED_LOAD>
__global__ void gemv_int8_int8_wide_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    if (n_base >= N) return;

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    const bool full_cols = (n_base + (CPT - 1) < N);
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        // Load 4 INT8 activations as packed int32 for v_dot4_i32_i8
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

        const int32_t* __restrict__ b_row = reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + n_base) * 4);

        if constexpr (VEC_PACKED_LOAD && CPT == 4)
        {
            if (full_cols)
            {
                const int4 b = *reinterpret_cast<const int4*>(b_row);
                acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.x, acc[0]);
                acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.y, acc[1]);
                acc[2] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.z, acc[2]);
                acc[3] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.w, acc[3]);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    if (n_base + c < N)
                    {
                        const int32_t b = *(b_row + c);
                        acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b, acc[c]);
                    }
                }
            }
        }
        else if constexpr (VEC_PACKED_LOAD && CPT == 2)
        {
            if (full_cols)
            {
                const int2 b = *reinterpret_cast<const int2*>(b_row);
                acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.x, acc[0]);
                acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.y, acc[1]);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    if (n_base + c < N)
                    {
                        const int32_t b = *(b_row + c);
                        acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b, acc[c]);
                    }
                }
            }
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                if (n_base + c < N)
                {
                    const int32_t b = *(b_row + c);
                    acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b, acc[c]);
                }
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        if (n_base + c < N)
        {
            d_C_int32[n_base + c] = acc[c];
        }
    }
}

// Wide-N kernel: 2 columns per thread
__global__ void gemv_int8_int8_wide2_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n2 = blockIdx.x * GEMV_INT8_VNNI_WIDE2_TILE_N + threadIdx.x * 2;
    if (n2 >= N) return;

    int32_t acc0 = 0;
    int32_t acc1 = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        // Load 4 INT8 activations as packed int32 for v_dot4_i32_i8
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

        // Load 4 INT8 weights for each of 2 columns (VNNI = already packed)
        const int32_t p0 = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n2 + 0)) * 4);
        const int32_t p1 = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n2 + 1)) * 4);

        // Hardware INT8 dot product: 4 multiply-adds in 1 cycle (v_dot4_i32_i8)
        acc0 = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, p0, acc0);
        acc1 = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, p1, acc1);
    }

    d_C_int32[n2 + 0] = acc0;
    d_C_int32[n2 + 1] = acc1;
}

// K-parallel kernel: scalar threads, shared A tiles
__global__ void gemv_int8_int8_square_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_INT8_VNNI_SQUARE_TILE_N + threadIdx.x;
    if (n >= N) return;

    int32_t acc = 0;
    __shared__ int8_t sA[GEMV_INT8_VNNI_KPAR_TILE_K];

    for (int k_base = 0; k_base < K; k_base += GEMV_INT8_VNNI_KPAR_TILE_K)
    {
        const int tile_k = min(GEMV_INT8_VNNI_KPAR_TILE_K, K - k_base);
        for (int i = threadIdx.x; i < tile_k; i += blockDim.x)
        {
            sA[i] = d_A_int8[k_base + i];
        }
        __syncthreads();

        const int tile_groups = tile_k / 4;
        const int k_group_base = k_base / 4;
        for (int kg = 0; kg < tile_groups; ++kg)
        {
            // Load 4 INT8 activations from shared memory as packed int32
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(sA + kg * 4);

            const int global_kg = k_group_base + kg;
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                d_B_int8_vnni + (static_cast<size_t>(global_kg) * N + n) * 4);

            // Hardware INT8 dot product: 4 multiply-adds in 1 cycle (v_dot4_i32_i8)
            acc = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, acc);
        }
        __syncthreads();
    }

    d_C_int32[n] = acc;
}

__global__ void gemv_int8_int8_vnni_blockwise_scaled_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    float alpha,
    float beta,
    const float* d_C_existing,
    const float* __restrict__ d_bias)
{
    const int n = blockIdx.x * GEMV_TILE_N + threadIdx.x;
    if (n >= N) return;

    float fp_acc = 0.0f;
    const int32_t* __restrict__ b_row = reinterpret_cast<const int32_t*>(
        d_B_int8_vnni + static_cast<size_t>(n) * 4);
    const size_t b_stride = static_cast<size_t>(N);
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    for (int act_block = 0; act_block < act_blocks; ++act_block)
    {
        int32_t block_acc = 0;
        const int kg_base = act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;
        #pragma unroll
        for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
        {
            const int kg = kg_base + kgi;
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);
            const int32_t b_packed = *(b_row + kg * b_stride);
            block_acc = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, block_acc);
        }
        fp_acc += static_cast<float>(block_acc) * d_scales_A_blockwise[act_block];
    }

    float result = fp_acc * d_scale_B[n];
    result *= alpha;
    if (beta != 0.0f)
    {
        result += beta * d_C_existing[n];
    }
    if (d_bias)
    {
        result += d_bias[n];
    }
    d_C_fp32[n] = result;
}

template <int TILE_N, int CPT, bool VEC_PACKED_LOAD>
__global__ void gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    float alpha,
    float beta,
    const float* d_C_existing,
    const float* __restrict__ d_bias)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    if (n_base >= N) return;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    const bool full_cols = (n_base + (CPT - 1) < N);
    const int32_t* __restrict__ b_row = reinterpret_cast<const int32_t*>(
        d_B_int8_vnni + static_cast<size_t>(n_base) * 4);
    const size_t b_stride = static_cast<size_t>(N);
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    for (int act_block = 0; act_block < act_blocks; ++act_block)
    {
        int32_t acc[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) acc[c] = 0;

        const int kg_base = act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;
        #pragma unroll
        for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
        {
            const int kg = kg_base + kgi;
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);
            const int32_t* __restrict__ b_pack_row = b_row + kg * b_stride;

            if constexpr (VEC_PACKED_LOAD && CPT == 4)
            {
                if (full_cols)
                {
                    const int4 b = *reinterpret_cast<const int4*>(b_pack_row);
                    acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.x, acc[0]);
                    acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.y, acc[1]);
                    acc[2] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.z, acc[2]);
                    acc[3] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.w, acc[3]);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        if (n_base + c < N)
                        {
                            const int32_t b = *(b_pack_row + c);
                            acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b, acc[c]);
                        }
                    }
                }
            }
            else if constexpr (VEC_PACKED_LOAD && CPT == 2)
            {
                if (full_cols)
                {
                    const int2 b = *reinterpret_cast<const int2*>(b_pack_row);
                    acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.x, acc[0]);
                    acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.y, acc[1]);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        if (n_base + c < N)
                        {
                            const int32_t b = *(b_pack_row + c);
                            acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b, acc[c]);
                        }
                    }
                }
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    if (n_base + c < N)
                    {
                        const int32_t b = *(b_pack_row + c);
                        acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b, acc[c]);
                    }
                }
            }
        }

        const float scale_a = d_scales_A_blockwise[act_block];
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            fp_acc[c] += static_cast<float>(acc[c]) * scale_a;
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            float out = fp_acc[c] * d_scale_B[n];
            out *= alpha;
            if (beta != 0.0f)
            {
                out += beta * d_C_existing[n];
            }
            if (d_bias)
            {
                out += d_bias[n];
            }
            d_C_fp32[n] = out;
        }
    }
}

__global__ void gemv_int8_int8_square_vnni_blockwise_scaled_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    float alpha,
    float beta,
    const float* d_C_existing,
    const float* __restrict__ d_bias)
{
    const int n = blockIdx.x * GEMV_INT8_VNNI_SQUARE_TILE_N + threadIdx.x;
    const bool owns_column = n < N;

    float fp_acc = 0.0f;
    __shared__ int8_t sA[GEMV_INT8_VNNI_KPAR_TILE_K];

    for (int k_tile_base = 0; k_tile_base < K; k_tile_base += GEMV_INT8_VNNI_KPAR_TILE_K)
    {
        const int tile_k = min(GEMV_INT8_VNNI_KPAR_TILE_K, K - k_tile_base);
        for (int i = threadIdx.x; i < tile_k; i += blockDim.x)
        {
            sA[i] = d_A_int8[k_tile_base + i];
        }
        __syncthreads();

        const int tile_act_blocks = tile_k / GEMV_INT8_VNNI_ACT_BLOCK_K;
        const int act_block_base = k_tile_base / GEMV_INT8_VNNI_ACT_BLOCK_K;
        const int tile_k_group_base = k_tile_base / 4;

        // A partial N tile still needs every lane to load A and reach both
        // barriers. Only live column owners read weights or accumulate.
        if (owns_column)
        {
            #pragma unroll
            for (int tile_block = 0; tile_block < GEMV_INT8_VNNI_KPAR_TILE_K / GEMV_INT8_VNNI_ACT_BLOCK_K; ++tile_block)
            {
                if (tile_block >= tile_act_blocks)
                    break;

                int32_t block_acc = 0;
                const int local_k_byte = tile_block * GEMV_INT8_VNNI_ACT_BLOCK_K;
                const int k_group_base = tile_k_group_base + tile_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;
                #pragma unroll
                for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
                {
                    const int32_t a_packed = *reinterpret_cast<const int32_t*>(sA + local_k_byte + kgi * 4);
                    const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                        d_B_int8_vnni + (static_cast<size_t>(k_group_base + kgi) * N + n) * 4);
                    block_acc = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, block_acc);
                }

                fp_acc += static_cast<float>(block_acc) * d_scales_A_blockwise[act_block_base + tile_block];
            }
        }
        __syncthreads();
    }

    if (!owns_column) return;
    float result = fp_acc * d_scale_B[n];
    result *= alpha;
    if (beta != 0.0f)
    {
        result += beta * d_C_existing[n];
    }
    if (d_bias)
    {
        result += d_bias[n];
    }
    d_C_fp32[n] = result;
}

template <int TILE_N, int CPT>
__global__ void gemv_int8_int8_batched_vnni_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const float* __restrict__ d_scales_A_blockwise,
    GemvBatchArgs args,
    int K,
    float alpha,
    float beta)
{
    const int batch = gemv_find_batch_for_block(args, blockIdx.x);
    const int local_block_x = blockIdx.x - args.grid_n_offset[batch];
    const int n_base = local_block_x * TILE_N + threadIdx.x * CPT;
    if (n_base >= args.N[batch]) return;

    const int N = args.N[batch];
    const int8_t* __restrict__ d_B_int8_vnni = args.d_B[batch];
    float* __restrict__ d_C_fp32 = args.d_C[batch];
    const float* __restrict__ d_scale_B = args.d_scales_B[batch];
    const float* __restrict__ d_bias = args.d_bias[batch];
    const bool full_cols = (n_base + (CPT - 1) < N);
    const int32_t* __restrict__ b_row = reinterpret_cast<const int32_t*>(
        d_B_int8_vnni + static_cast<size_t>(n_base) * 4);
    const size_t b_stride = static_cast<size_t>(N);
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    for (int act_block = 0; act_block < act_blocks; ++act_block)
    {
        int32_t acc[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) acc[c] = 0;

        gemv_int8_accumulate_act_block(acc, d_A_int8, b_row, b_stride,
                                       act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);

        const float scale_a = d_scales_A_blockwise[act_block];
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            fp_acc[c] += static_cast<float>(acc[c]) * scale_a;
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            float result = fp_acc[c] * d_scale_B[n];
            result *= alpha;
            if (beta != 0.0f)
            {
                result += beta * d_C_fp32[n];
            }
            if (d_bias)
            {
                result += d_bias[n];
            }
            d_C_fp32[n] = result;
        }
    }
}

template <int TILE_N, int CPT, bool VEC_PACKED_LOAD>
__global__ void gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const float* __restrict__ d_scales_A_blockwise,
    GemvBatchArgs args,
    int K,
    float alpha,
    float beta)
{
    const int batch = gemv_find_batch_for_block(args, blockIdx.x);
    const int local_block_x = blockIdx.x - args.grid_n_offset[batch];
    const int n_base = local_block_x * TILE_N + threadIdx.x * CPT;
    if (n_base >= args.N[batch]) return;

    const int N = args.N[batch];
    const int8_t* __restrict__ d_B_int8_vnni = args.d_B[batch];
    float* __restrict__ d_C_fp32 = args.d_C[batch];
    const float* __restrict__ d_scale_B = args.d_scales_B[batch];
    const float* __restrict__ d_bias = args.d_bias[batch];
    const bool full_cols = (n_base + (CPT - 1) < N);

    const int32_t* __restrict__ b_row = reinterpret_cast<const int32_t*>(
        d_B_int8_vnni + static_cast<size_t>(n_base) * 4);
    const size_t b_stride = static_cast<size_t>(N);
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    for (int act_block = 0; act_block < act_blocks; ++act_block)
    {
        int32_t acc[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) acc[c] = 0;

        const int kg_base = act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;
        #pragma unroll
        for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
        {
            const int kg = kg_base + kgi;
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);
            const int32_t* __restrict__ b_pack_row = b_row + kg * b_stride;

            if constexpr (VEC_PACKED_LOAD && CPT == 4)
            {
                if (full_cols)
                {
                    const int4 b = *reinterpret_cast<const int4*>(b_pack_row);
                    acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.x, acc[0]);
                    acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.y, acc[1]);
                    acc[2] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.z, acc[2]);
                    acc[3] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.w, acc[3]);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        if (n_base + c < N)
                        {
                            const int32_t b = *(b_pack_row + c);
                            acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b, acc[c]);
                        }
                    }
                }
            }
            else if constexpr (VEC_PACKED_LOAD && CPT == 2)
            {
                if (full_cols)
                {
                    const int2 b = *reinterpret_cast<const int2*>(b_pack_row);
                    acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.x, acc[0]);
                    acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.y, acc[1]);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        if (n_base + c < N)
                        {
                            const int32_t b = *(b_pack_row + c);
                            acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b, acc[c]);
                        }
                    }
                }
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    if (n_base + c < N)
                    {
                        const int32_t b = *(b_pack_row + c);
                        acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b, acc[c]);
                    }
                }
            }
        }

        const float scale_a = d_scales_A_blockwise[act_block];
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            fp_acc[c] += static_cast<float>(acc[c]) * scale_a;
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            float result = fp_acc[c] * d_scale_B[n];
            result *= alpha;
            if (beta != 0.0f)
            {
                result += beta * d_C_fp32[n];
            }
            if (d_bias)
            {
                result += d_bias[n];
            }
            d_C_fp32[n] = result;
        }
    }
}

__global__ void gemv_int8_int8_square_batched_vnni_blockwise_scaled_kernel(
    const int8_t* __restrict__ d_A_int8,
    const float* __restrict__ d_scales_A_blockwise,
    GemvBatchArgs args,
    int K,
    float alpha,
    float beta)
{
    const int batch = gemv_find_batch_for_block(args, blockIdx.x);
    const int local_block_x = blockIdx.x - args.grid_n_offset[batch];
    const int n = local_block_x * GEMV_INT8_VNNI_SQUARE_TILE_N + threadIdx.x;
    if (n >= args.N[batch]) return;

    const int8_t* __restrict__ d_B_int8_vnni = args.d_B[batch];
    float* __restrict__ d_C_fp32 = args.d_C[batch];
    const float* __restrict__ d_scale_B = args.d_scales_B[batch];
    const float* __restrict__ d_bias = args.d_bias[batch];
    __shared__ int8_t sA[GEMV_INT8_VNNI_KPAR_TILE_K];

    float fp_acc = 0.0f;
    for (int k_tile_base = 0; k_tile_base < K; k_tile_base += GEMV_INT8_VNNI_KPAR_TILE_K)
    {
        const int tile_k = min(GEMV_INT8_VNNI_KPAR_TILE_K, K - k_tile_base);
        for (int i = threadIdx.x; i < tile_k; i += blockDim.x)
        {
            sA[i] = d_A_int8[k_tile_base + i];
        }
        __syncthreads();

        const int tile_act_blocks = tile_k / GEMV_INT8_VNNI_ACT_BLOCK_K;
        const int act_block_base = k_tile_base / GEMV_INT8_VNNI_ACT_BLOCK_K;
        const int tile_k_group_base = k_tile_base / 4;

        #pragma unroll
        for (int tile_block = 0; tile_block < GEMV_INT8_VNNI_KPAR_TILE_K / GEMV_INT8_VNNI_ACT_BLOCK_K; ++tile_block)
        {
            if (tile_block >= tile_act_blocks)
            {
                break;
            }

            int32_t block_acc = 0;
            const int local_k_byte = tile_block * GEMV_INT8_VNNI_ACT_BLOCK_K;
            const int k_group_base = tile_k_group_base + tile_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;
            #pragma unroll
            for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(sA + local_k_byte + kgi * 4);
                const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                    d_B_int8_vnni + (static_cast<size_t>(k_group_base + kgi) * args.N[batch] + n) * 4);
                block_acc = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, block_acc);
            }

            fp_acc += static_cast<float>(block_acc) * d_scales_A_blockwise[act_block_base + tile_block];
        }
        __syncthreads();
    }

    float result = fp_acc * d_scale_B[n];
    result *= alpha;
    if (beta != 0.0f)
    {
        result += beta * d_C_fp32[n];
    }
    if (d_bias)
    {
        result += d_bias[n];
    }
    d_C_fp32[n] = result;
}

// ============================================================================
// Grid-level K-parallel VNNI kernel with atomicAdd reduction
//
// Each block processes a TILE_N/CPT-thread tile × one K-segment.
// Grid: (ceil(N/TILE_N), KB) where KB splits K across blockIdx.y.
// Output MUST be zeroed before launch (atomicAdd accumulation).
//
// Key insight: Grid-level K-splitting creates 28×48=1344 independent blocks
// (vs 56 with square_vnni), providing enough wavefronts to fully hide
// HBM2 latency on MI50 (60 CUs × 40 max concurrent waves = 2400).
//
// Benchmark results (MI50, gfx906):
//   Q/Wo (3584×3584):    22.0μs / 58.5% peak BW (9.6× vs square_vnni)
//   FFN Down (3584×18944): 95.7μs / 70.9% peak BW (2.0× vs smem kpar)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_int8_int8_grid_kpar_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,   // [K/4][N][4] flat VNNI
    int32_t* __restrict__ d_C_int32,
    int N, int K, int kblocks)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;

    // Each block processes its K-segment: [k_start, k_end)
    // CRITICAL: k_per must be a multiple of 4 to ensure k_start aligns with
    // VNNI 4-byte groups. If k_start is not a multiple of 4, kg_offset would
    // misalign with the weight K-groups, producing garbage output.
    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;  // Round up to multiple of 4
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    // Accumulate partial dot product for CPT output columns
    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    if (active)
    {
        // Process VNNI k-groups within this block's K-segment
        const int kg_start = k_start / 4;
        const int kg_end = k_end / 4;

        // Precompute B pointer: b_row[kg] points to the start of this thread's
        // columns for k-group kg. b_row = B_int32_ptr + n_base, stride = N (int32 units).
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        // Partial unroll: expose 4 k-groups per iteration for better
        // memory-level parallelism and instruction scheduling
        #pragma unroll 4
        for (int kg = kg_start; kg < kg_end; ++kg)
        {
            // Load 4 INT8 activations as packed int32 for v_dot4_i32_i8
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

            if constexpr (CPT == 2)
            {
                // Vectorized 8-byte load: two adjacent VNNI columns in one
                // memory transaction. n_base is always even (threadIdx.x * 2),
                // so int2 alignment is guaranteed for N divisible by 2.
                const int2 b_pair = *reinterpret_cast<const int2*>(
                    b_row + kg * b_stride);
                acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.x, acc[0]);
                acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.y, acc[1]);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int32_t b_packed = *(b_row + kg * b_stride + c);
                    acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, acc[c]);
                }
            }
        }

        // Accumulate partials to output via atomicAdd
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            atomicAdd(&d_C_int32[n_base + c], acc[c]);
        }
    }
}

template <int TILE_N, int CPT>
__global__ void gemv_int8_int8_grid_kpar_vnni_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    int act_block_splits,
    float alpha)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int split_idx = blockIdx.y;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
    const int act_blocks_per_split = (act_blocks + act_block_splits - 1) / act_block_splits;
    const int act_block_start = split_idx * act_blocks_per_split;
    const int act_block_end = min(act_block_start + act_blocks_per_split, act_blocks);

    if (!active || act_block_start >= act_blocks)
    {
        return;
    }

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        fp_acc[c] = 0.0f;
    }

    const int32_t* __restrict__ b_row =
        reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
    const size_t b_stride = static_cast<size_t>(N);
    __shared__ int8_t sA[GEMV_INT8_VNNI_KPAR_TILE_K];

    constexpr int ACT_BLOCKS_PER_TILE = GEMV_INT8_VNNI_KPAR_TILE_K / GEMV_INT8_VNNI_ACT_BLOCK_K;
    for (int act_block_tile = act_block_start; act_block_tile < act_block_end; act_block_tile += ACT_BLOCKS_PER_TILE)
    {
        const int tile_act_blocks = min(ACT_BLOCKS_PER_TILE, act_block_end - act_block_tile);
        const int tile_bytes = tile_act_blocks * GEMV_INT8_VNNI_ACT_BLOCK_K;
        for (int i = threadIdx.x; i < tile_bytes; i += blockDim.x)
        {
            sA[i] = d_A_int8[act_block_tile * GEMV_INT8_VNNI_ACT_BLOCK_K + i];
        }
        __syncthreads();

        #pragma unroll
        for (int tile_block = 0; tile_block < ACT_BLOCKS_PER_TILE; ++tile_block)
        {
            if (tile_block >= tile_act_blocks)
            {
                break;
            }

            int32_t block_acc[CPT];
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                block_acc[c] = 0;
            }

            const int local_k_byte = tile_block * GEMV_INT8_VNNI_ACT_BLOCK_K;
            const int kg_base = (act_block_tile + tile_block) * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;
            #pragma unroll
            for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(sA + local_k_byte + kgi * 4);
                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(b_row + (kg_base + kgi) * b_stride);
                    block_acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.x, block_acc[0]);
                    block_acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.y, block_acc[1]);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + (kg_base + kgi) * b_stride + c);
                        block_acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, block_acc[c]);
                    }
                }
            }

            gemv_int8_flush_blockwise_acc(block_acc, fp_acc, d_scales_A_blockwise, act_block_tile + tile_block);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            const float val = fp_acc[c] * d_scale_B[n] * alpha;
            if (act_block_splits == 1)
            {
                d_C_fp32[n] = val;
            }
            else
            {
                atomicAdd(&d_C_fp32[n], val);
            }
        }
    }
}

template <int TILE_N, int CPT>
__global__ void gemv_int8_int8_grid_kpar_batched_vnni_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const float* __restrict__ d_scales_A_blockwise,
    GemvBatchArgs args,
    int K,
    int act_block_splits,
    float alpha)
{
    const int batch = gemv_find_batch_for_block(args, blockIdx.x);
    const int local_block_x = blockIdx.x - args.grid_n_offset[batch];
    const int n_base = local_block_x * TILE_N + threadIdx.x * CPT;
    const int N = args.N[batch];
    const bool active = (n_base < N);
    const int split_idx = blockIdx.y;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
    const int act_blocks_per_split = (act_blocks + act_block_splits - 1) / act_block_splits;
    const int act_block_start = split_idx * act_blocks_per_split;
    const int act_block_end = min(act_block_start + act_blocks_per_split, act_blocks);

    if (!active || act_block_start >= act_blocks)
    {
        return;
    }

    float* __restrict__ d_C_fp32 = args.d_C[batch];
    const float* __restrict__ d_scale_B = args.d_scales_B[batch];
    const int32_t* __restrict__ b_row =
        reinterpret_cast<const int32_t*>(args.d_B[batch]) + n_base;
    const size_t b_stride = static_cast<size_t>(N);

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        fp_acc[c] = 0.0f;
    }

    for (int act_block = act_block_start; act_block < act_block_end; ++act_block)
    {
        int32_t block_acc[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            block_acc[c] = 0;
        }

        gemv_int8_accumulate_act_block(block_acc, d_A_int8, b_row, b_stride,
                                       act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);
        gemv_int8_flush_blockwise_acc(block_acc, fp_acc, d_scales_A_blockwise, act_block);
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            atomicAdd(&d_C_fp32[n], fp_acc[c] * d_scale_B[n] * alpha);
        }
    }
}

template <int TILE_N, int CPT>
__global__ void gemv_int8_int8_grid_kpar_pair_vnni_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const float* __restrict__ d_scales_A_blockwise,
    const int8_t* __restrict__ d_B0,
    const int8_t* __restrict__ d_B1,
    float* __restrict__ d_C0,
    float* __restrict__ d_C1,
    const float* __restrict__ d_scale_B0,
    const float* __restrict__ d_scale_B1,
    int N0,
    int N1,
    int grid_n0,
    int K,
    int act_block_splits,
    float alpha)
{
    const bool second = (blockIdx.x >= grid_n0);
    const int local_block_x = second ? (blockIdx.x - grid_n0) : blockIdx.x;
    const int n_base = local_block_x * TILE_N + threadIdx.x * CPT;
    const int N = second ? N1 : N0;
    const bool active = (n_base < N);
    const int split_idx = blockIdx.y;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
    const int act_blocks_per_split = (act_blocks + act_block_splits - 1) / act_block_splits;
    const int act_block_start = split_idx * act_blocks_per_split;
    const int act_block_end = min(act_block_start + act_blocks_per_split, act_blocks);

    if (!active || act_block_start >= act_blocks)
    {
        return;
    }

    float* __restrict__ d_C_fp32 = second ? d_C1 : d_C0;
    const float* __restrict__ d_scale_B = second ? d_scale_B1 : d_scale_B0;
    const int8_t* __restrict__ d_B_int8_vnni = second ? d_B1 : d_B0;
    const int32_t* __restrict__ b_row =
        reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
    const size_t b_stride = static_cast<size_t>(N);

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        fp_acc[c] = 0.0f;
    }

    for (int act_block = act_block_start; act_block < act_block_end; ++act_block)
    {
        int32_t block_acc[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            block_acc[c] = 0;
        }

        gemv_int8_accumulate_act_block(block_acc, d_A_int8, b_row, b_stride,
                                       act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);
        gemv_int8_flush_blockwise_acc(block_acc, fp_acc, d_scales_A_blockwise, act_block);
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            atomicAdd(&d_C_fp32[n], fp_acc[c] * d_scale_B[n] * alpha);
        }
    }
}


// ============================================================================
// LDS K-Reduction GEMV kernel: eliminates atomicAdd via intra-block reduction
//
// KEY INSIGHT: Instead of using 1-wavefront blocks with grid-level atomicAdd,
// use WK-wavefront blocks where each wavefront computes a K-slice and the
// partial sums are reduced via LDS within the block.
//
// Advantages over grid_kpar:
//   1. Eliminates atomicAdd when kb_outer==1 (large-N shapes like FFN)
//   2. Eliminates hipMemsetAsync when kb_outer==1
//   3. Reduces atomicAdd count by WK× when kb_outer>1 (small-N shapes)
//   4. Same total wavefront count → same occupancy and memory bandwidth
//
// Thread layout:
//   Block: WK × (TILE_N / CPT) threads = 4 × 64 = 256
//   wave_id = threadIdx.x / (TILE_N/CPT) → K-slice index (0..WK-1)
//   lane_id = threadIdx.x % (TILE_N/CPT) → column index within tile
//   Each wave handles TILE_N output columns across K/(WK*kb_outer) k-groups
//
// LDS usage: (WK-1) × TILE_N × sizeof(int32_t)
//   WK=4, TILE_N=128: 3 × 128 × 4 = 1536 bytes (trivial)
//
// Grid: (ceil(N/TILE_N), kb_outer)
//   When kb_outer==1: plain store (no atomicAdd, no memset)
//   When kb_outer>1:  atomicAdd (but WK× fewer than grid_kpar)
//
// Benchmark motivation (MI50, gfx906):
//   FFN Gate/Up (18944×3584): grid_kpar KB=4 → 756 GB/s (76% peak)
//     - 148 N-blocks × 4 K-blocks = 592 wavefronts = 9.9 waves/CU
//     - 18944 × 4 = 75776 atomicAdds per launch
//   NEW lds_kreduce kb_outer=1: same 592 wavefronts, zero atomicAdds
//   Q/Wo (3584×3584): grid_kpar KB=16 → 456 GB/s (46% peak)
//     - 28 × 16 = 448 wavefronts, 3584 × 16 = 57344 atomicAdds
//   NEW lds_kreduce kb_outer=4: 28×4×4 = 448 wavefronts, 3584 × 4 atomicAdds
// ============================================================================
template <int TILE_N, int CPT, int WK>
__global__ __launch_bounds__(TILE_N / CPT * WK)
void gemv_int8_int8_lds_kreduce_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,   // [K/4][N][4] flat VNNI
    int32_t* __restrict__ d_C_int32,
    int N, int K, int kb_outer)
{
    static_assert(CPT == 2 || CPT == 4, "CPT must be 2 or 4");
    static_assert(WK >= 2 && WK <= 16, "WK must be 2-16");
    constexpr int THREADS_PER_WAVE = TILE_N / CPT;
    static_assert(THREADS_PER_WAVE == 64, "TILE_N/CPT must equal 64 (one wavefront)");

    const int wave_id = threadIdx.x / THREADS_PER_WAVE;
    const int lane_id = threadIdx.x % THREADS_PER_WAVE;
    const int n_base = blockIdx.x * TILE_N + lane_id * CPT;
    const int outer_k_block = blockIdx.y;

    // --- Compute K-range for this block's outer K-slice ---
    const int total_k_groups = K / 4;

    // Evenly distribute k-groups across kb_outer grid blocks
    const int kgroups_per_outer = (total_k_groups + kb_outer - 1) / kb_outer;
    const int outer_kg_start = outer_k_block * kgroups_per_outer;
    const int outer_kg_end = min(outer_kg_start + kgroups_per_outer, total_k_groups);
    if (outer_kg_start >= total_k_groups) return;

    const int outer_kgroup_count = outer_kg_end - outer_kg_start;

    // --- Inner K-split: distribute outer range across WK wavefronts ---
    // Balanced distribution: waves 0..remainder-1 get one extra k-group
    const int kg_per_wave_base = outer_kgroup_count / WK;
    const int kg_remainder = outer_kgroup_count % WK;
    const int my_kg_start = outer_kg_start
        + wave_id * kg_per_wave_base
        + min(wave_id, kg_remainder);
    const int my_kg_count = kg_per_wave_base + (wave_id < kg_remainder ? 1 : 0);
    const int my_kg_end = my_kg_start + my_kg_count;

    // Accumulators — each thread accumulates CPT output column values
    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    // Main K-reduction loop (innermost hot loop)
    if (n_base < N && my_kg_count > 0)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        // #pragma unroll 4: expose 4 k-groups for memory-level parallelism.
        // Compiler generates 4× global_load + 4× s_load + 8× v_dot4 interleaved.
        #pragma unroll 4
        for (int kg = my_kg_start; kg < my_kg_end; ++kg)
        {
            // A is uniform across all threads in this wave (M=1)
            // → compiler uses scalar load (s_load_dword), broadcast to all lanes
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

            if constexpr (CPT == 2)
            {
                // 8-byte vectorized load: two adjacent VNNI columns
                const int2 b_pair = *reinterpret_cast<const int2*>(
                    b_row + kg * b_stride);
                acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.x, acc[0]);
                acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.y, acc[1]);
            }
            else // CPT == 4
            {
                // 16-byte vectorized load: four adjacent VNNI columns
                const int4 b_pack = *reinterpret_cast<const int4*>(
                    b_row + kg * b_stride);
                acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pack.x, acc[0]);
                acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pack.y, acc[1]);
                acc[2] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pack.z, acc[2]);
                acc[3] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pack.w, acc[3]);
            }
        }
    }

    // =========================================================================
    // LDS K-reduction: sum partial results across WK wavefronts
    //
    // Waves 1..WK-1 deposit their partial sums into LDS.
    // After barrier, wave 0 reads and sums all partials + its own accumulator.
    // This replaces global atomicAdd with fast LDS reads (5-10 cycles each).
    // =========================================================================
    __shared__ int32_t s_partial[WK - 1][TILE_N];

    // Waves 1..WK-1 write partials to LDS
    if (wave_id > 0 && n_base < N)
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            s_partial[wave_id - 1][lane_id * CPT + c] = acc[c];
        }
    }

    __syncthreads();

    // Wave 0 sums all partials and writes to global output
    if (wave_id == 0 && n_base < N)
    {
        // Sum own partial + (WK-1) LDS contributions
        #pragma unroll
        for (int w = 0; w < WK - 1; ++w)
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                acc[c] += s_partial[w][lane_id * CPT + c];
            }
        }

        // Store output
        if (kb_outer == 1)
        {
            // Plain store — no atomicAdd needed, no pre-zeroing required
            if constexpr (CPT == 2)
            {
                *reinterpret_cast<int2*>(&d_C_int32[n_base]) =
                    make_int2(acc[0], acc[1]);
            }
            else // CPT == 4
            {
                *reinterpret_cast<int4*>(&d_C_int32[n_base]) =
                    make_int4(acc[0], acc[1], acc[2], acc[3]);
            }
        }
        else
        {
            // Multi-block outer K-split: atomicAdd (but WK× fewer than grid_kpar)
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                atomicAdd(&d_C_int32[n_base + c], acc[c]);
            }
        }
    }
}

template <int TILE_N, int CPT, int WK>
__global__ __launch_bounds__(TILE_N / CPT * WK)
void gemv_int8_int8_lds_kreduce_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    int kb_outer,
    float alpha)
{
    static_assert(CPT == 2 || CPT == 4, "CPT must be 2 or 4");
    static_assert(WK >= 2 && WK <= 16, "WK must be 2-16");
    constexpr int THREADS_PER_WAVE = TILE_N / CPT;
    static_assert(THREADS_PER_WAVE == 64, "TILE_N/CPT must equal 64 (one wavefront)");

    const int wave_id = threadIdx.x / THREADS_PER_WAVE;
    const int lane_id = threadIdx.x % THREADS_PER_WAVE;
    const int n_base = blockIdx.x * TILE_N + lane_id * CPT;
    const int outer_block = blockIdx.y;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    const int act_blocks_per_outer = (act_blocks + kb_outer - 1) / kb_outer;
    const int outer_act_start = outer_block * act_blocks_per_outer;
    const int outer_act_end = min(outer_act_start + act_blocks_per_outer, act_blocks);
    if (outer_act_start >= act_blocks)
    {
        return;
    }

    const int outer_act_count = outer_act_end - outer_act_start;
    const int act_per_wave_base = outer_act_count / WK;
    const int act_remainder = outer_act_count % WK;
    const int my_act_start = outer_act_start + wave_id * act_per_wave_base + min(wave_id, act_remainder);
    const int my_act_count = act_per_wave_base + (wave_id < act_remainder ? 1 : 0);
    const int my_act_end = my_act_start + my_act_count;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    if (n_base < N && my_act_count > 0)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        #pragma unroll 1
        for (int act_block = my_act_start; act_block < my_act_end; ++act_block)
        {
            int32_t block_acc[CPT];
            #pragma unroll
            for (int c = 0; c < CPT; ++c) block_acc[c] = 0;

            gemv_int8_accumulate_act_block(block_acc, d_A_int8, b_row, b_stride,
                                           act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);
            gemv_int8_flush_blockwise_acc(block_acc, fp_acc, d_scales_A_blockwise, act_block);
        }
    }

    __shared__ float s_partial[WK - 1][TILE_N];

    if (wave_id > 0 && n_base < N)
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            s_partial[wave_id - 1][lane_id * CPT + c] = fp_acc[c];
        }
    }

    __syncthreads();

    if (wave_id == 0 && n_base < N)
    {
        #pragma unroll
        for (int w = 0; w < WK - 1; ++w)
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                fp_acc[c] += s_partial[w][lane_id * CPT + c];
            }
        }

        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            const float partial = fp_acc[c] * d_scale_B[n] * alpha;
            if (kb_outer == 1)
            {
                d_C_fp32[n] = partial;
            }
            else
            {
                atomicAdd(&d_C_fp32[n], partial);
            }
        }
    }
}

// Pair variant of the tiny K/V LDS K-reduce kernel.
// Dispatches two projections (e.g. K+V) in a single kernel launch
// with combined grid_x = ceil(N0/TN) + ceil(N1/TN).
// Doubles occupancy and saves one launch overhead vs 2x single.
template <int TILE_N, int CPT, int WK>
__global__ __launch_bounds__(64 * WK)
void gemv_int8_int8_lds_kreduce_pair_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B0,
    const int8_t* __restrict__ d_B1,
    float* __restrict__ d_C0,
    float* __restrict__ d_C1,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scales_B0,
    const float* __restrict__ d_scales_B1,
    int N0, int N1,
    int grid_n_offset,
    int K, int kb_outer,
    float alpha)
{
    static_assert(CPT == 2 || CPT == 4, "CPT must be 2 or 4");
    static_assert(WK >= 2 && WK <= 16, "WK must be 2-16");
    constexpr int THREADS_PER_WAVE = TILE_N / CPT;
    static_assert(THREADS_PER_WAVE == 64, "TILE_N/CPT must equal 64 (one wavefront)");

    // Determine which projection this block serves
    const bool is_proj1 = (static_cast<int>(blockIdx.x) >= grid_n_offset);
    const int local_block_x = is_proj1 ? (static_cast<int>(blockIdx.x) - grid_n_offset) : static_cast<int>(blockIdx.x);
    const int8_t* __restrict__ d_B = is_proj1 ? d_B1 : d_B0;
    float* __restrict__ d_C = is_proj1 ? d_C1 : d_C0;
    const float* __restrict__ d_scale_B = is_proj1 ? d_scales_B1 : d_scales_B0;
    const int N = is_proj1 ? N1 : N0;

    const int wave_id = threadIdx.x / THREADS_PER_WAVE;
    const int lane_id = threadIdx.x % THREADS_PER_WAVE;
    const int n_base = local_block_x * TILE_N + lane_id * CPT;
    const int outer_block = blockIdx.y;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    const int act_blocks_per_outer = (act_blocks + kb_outer - 1) / kb_outer;
    const int outer_act_start = outer_block * act_blocks_per_outer;
    const int outer_act_end = min(outer_act_start + act_blocks_per_outer, act_blocks);
    if (outer_act_start >= act_blocks)
    {
        return;
    }

    const int outer_act_count = outer_act_end - outer_act_start;
    const int act_per_wave_base = outer_act_count / WK;
    const int act_remainder = outer_act_count % WK;
    const int my_act_start = outer_act_start + wave_id * act_per_wave_base + min(wave_id, act_remainder);
    const int my_act_count = act_per_wave_base + (wave_id < act_remainder ? 1 : 0);
    const int my_act_end = my_act_start + my_act_count;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    if (n_base < N && my_act_count > 0)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        #pragma unroll 1
        for (int act_block = my_act_start; act_block < my_act_end; ++act_block)
        {
            int32_t block_acc[CPT];
            #pragma unroll
            for (int c = 0; c < CPT; ++c) block_acc[c] = 0;

            gemv_int8_accumulate_act_block(block_acc, d_A_int8, b_row, b_stride,
                                           act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);
            gemv_int8_flush_blockwise_acc(block_acc, fp_acc, d_scales_A_blockwise, act_block);
        }
    }

    __shared__ float s_partial[WK - 1][TILE_N];

    if (wave_id > 0 && n_base < N)
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            s_partial[wave_id - 1][lane_id * CPT + c] = fp_acc[c];
        }
    }

    __syncthreads();

    if (wave_id == 0 && n_base < N)
    {
        #pragma unroll
        for (int w = 0; w < WK - 1; ++w)
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                fp_acc[c] += s_partial[w][lane_id * CPT + c];
            }
        }

        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            const float partial = fp_acc[c] * d_scale_B[n] * alpha;
            if (kb_outer == 1)
            {
                d_C[n] = partial;
            }
            else
            {
                atomicAdd(&d_C[n], partial);
            }
        }
    }
}

template <int WK>
__global__ __launch_bounds__(64 * WK)
void gemv_int8_int8_qwo_lds_kreduce_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    int kb_outer,
    float alpha)
{
    static_assert(WK >= 2 && WK <= 16, "WK must be 2-16");

    constexpr int TILE_N = 256;
    constexpr int CPT = 4;
    constexpr int THREADS_PER_WAVE = TILE_N / CPT;
    static_assert(THREADS_PER_WAVE == 64, "TILE_N/CPT must equal 64 (one wavefront)");

    const int wave_id = threadIdx.x / THREADS_PER_WAVE;
    const int lane_id = threadIdx.x % THREADS_PER_WAVE;
    const int n_base = blockIdx.x * TILE_N + lane_id * CPT;
    const bool full_cols = (n_base + 3) < N;
    const int outer_block = blockIdx.y;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    const int act_blocks_per_outer = (act_blocks + kb_outer - 1) / kb_outer;
    const int outer_act_start = outer_block * act_blocks_per_outer;
    const int outer_act_end = min(outer_act_start + act_blocks_per_outer, act_blocks);
    if (outer_act_start >= act_blocks)
    {
        return;
    }

    const int outer_act_count = outer_act_end - outer_act_start;
    const int act_per_wave_base = outer_act_count / WK;
    const int act_remainder = outer_act_count % WK;
    const int my_act_start = outer_act_start + wave_id * act_per_wave_base + min(wave_id, act_remainder);
    const int my_act_count = act_per_wave_base + (wave_id < act_remainder ? 1 : 0);
    const int my_act_end = my_act_start + my_act_count;

    float fp_acc[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c)
    {
        fp_acc[c] = 0.0f;
    }

    if (n_base < N && my_act_count > 0)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        #pragma unroll 1
        for (int act_block = my_act_start; act_block < my_act_end; ++act_block)
        {
            int32_t block_acc[4] = {0, 0, 0, 0};
            const int kg_base = act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;

            #pragma unroll
            for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + (kg_base + kgi) * 4);
                const int32_t* __restrict__ b_pack_row = b_row + (kg_base + kgi) * b_stride;

                if (full_cols)
                {
                    const int4 b = *reinterpret_cast<const int4*>(b_pack_row);
                    block_acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.x, block_acc[0]);
                    block_acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.y, block_acc[1]);
                    block_acc[2] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.z, block_acc[2]);
                    block_acc[3] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.w, block_acc[3]);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < 4; ++c)
                    {
                        if (n_base + c < N)
                        {
                            const int32_t b = *(b_pack_row + c);
                            block_acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b, block_acc[c]);
                        }
                    }
                }
            }

            gemv_int8_flush_blockwise_acc(block_acc, fp_acc, d_scales_A_blockwise, act_block);
        }
    }

    __shared__ float4 s_partial[WK - 1][64];

    if (wave_id > 0 && n_base < N)
    {
        s_partial[wave_id - 1][lane_id] = make_float4(fp_acc[0], fp_acc[1], fp_acc[2], fp_acc[3]);
    }

    __syncthreads();

    if (wave_id == 0 && n_base < N)
    {
        #pragma unroll
        for (int w = 0; w < WK - 1; ++w)
        {
            const float4 partial = s_partial[w][lane_id];
            fp_acc[0] += partial.x;
            fp_acc[1] += partial.y;
            fp_acc[2] += partial.z;
            fp_acc[3] += partial.w;
        }

        if (full_cols)
        {
            const float4 scale = *reinterpret_cast<const float4*>(d_scale_B + n_base);
            if (kb_outer == 1)
            {
                d_C_fp32[n_base + 0] = fp_acc[0] * scale.x * alpha;
                d_C_fp32[n_base + 1] = fp_acc[1] * scale.y * alpha;
                d_C_fp32[n_base + 2] = fp_acc[2] * scale.z * alpha;
                d_C_fp32[n_base + 3] = fp_acc[3] * scale.w * alpha;
            }
            else
            {
                atomicAdd(&d_C_fp32[n_base + 0], fp_acc[0] * scale.x * alpha);
                atomicAdd(&d_C_fp32[n_base + 1], fp_acc[1] * scale.y * alpha);
                atomicAdd(&d_C_fp32[n_base + 2], fp_acc[2] * scale.z * alpha);
                atomicAdd(&d_C_fp32[n_base + 3], fp_acc[3] * scale.w * alpha);
            }
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    const float partial = fp_acc[c] * d_scale_B[n] * alpha;
                    if (kb_outer == 1)
                    {
                        d_C_fp32[n] = partial;
                    }
                    else
                    {
                        atomicAdd(&d_C_fp32[n], partial);
                    }
                }
            }
        }
    }
}


// ============================================================================
// Q/Wo 2x-unrolled LDS K-reduction blockwise kernel
//
// Processes 2 activation blocks simultaneously with separate INT32
// accumulators, interleaving dot products for better ILP. The compiler
// can overlap B loads for block1 with sdot4 computation for block0,
// and pipeline flushes from both blocks.
//
// Register pressure: 2*4 = 8 INT32 acc + 4 FP32 acc = 12 VGPRs for accums.
// ============================================================================
template <int WK>
__global__ __launch_bounds__(64 * WK)
void gemv_int8_int8_qwo_unroll2_lds_kreduce_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    int kb_outer,
    float alpha)
{
    static_assert(WK >= 2 && WK <= 16, "WK must be 2-16");

    constexpr int TILE_N = 256;
    constexpr int CPT = 4;
    constexpr int THREADS_PER_WAVE = TILE_N / CPT;
    static_assert(THREADS_PER_WAVE == 64, "TILE_N/CPT must equal 64 (one wavefront)");

    const int wave_id = threadIdx.x / THREADS_PER_WAVE;
    const int lane_id = threadIdx.x % THREADS_PER_WAVE;
    const int n_base = blockIdx.x * TILE_N + lane_id * CPT;
    const bool full_cols = (n_base + 3) < N;
    const int outer_block = blockIdx.y;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    const int act_blocks_per_outer = (act_blocks + kb_outer - 1) / kb_outer;
    const int outer_act_start = outer_block * act_blocks_per_outer;
    const int outer_act_end = min(outer_act_start + act_blocks_per_outer, act_blocks);
    if (outer_act_start >= act_blocks)
    {
        return;
    }

    const int outer_act_count = outer_act_end - outer_act_start;
    const int act_per_wave_base = outer_act_count / WK;
    const int act_remainder = outer_act_count % WK;
    const int my_act_start = outer_act_start + wave_id * act_per_wave_base + min(wave_id, act_remainder);
    const int my_act_count = act_per_wave_base + (wave_id < act_remainder ? 1 : 0);
    const int my_act_end = my_act_start + my_act_count;

    float fp_acc[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c)
    {
        fp_acc[c] = 0.0f;
    }

    if (n_base < N && my_act_count > 0)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        // Process pairs of activation blocks for ILP
        int act_block = my_act_start;
        const int pair_end = my_act_start + (my_act_count & ~1);  // round down to even

        #pragma unroll 1
        for (; act_block < pair_end; act_block += 2)
        {
            int32_t acc0[4] = {0, 0, 0, 0};
            int32_t acc1[4] = {0, 0, 0, 0};
            const int kg_base0 = act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;
            const int kg_base1 = (act_block + 1) * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;

            // Interleave k-groups from both blocks for ILP
            #pragma unroll
            for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
            {
                const int32_t a0 = *reinterpret_cast<const int32_t*>(d_A_int8 + (kg_base0 + kgi) * 4);
                const int32_t a1 = *reinterpret_cast<const int32_t*>(d_A_int8 + (kg_base1 + kgi) * 4);

                if (full_cols)
                {
                    const int4 b0 = *reinterpret_cast<const int4*>(b_row + (kg_base0 + kgi) * b_stride);
                    acc0[0] = ::llaminar2::rocm_integer_contract::signedDot4(a0, b0.x, acc0[0]);
                    acc0[1] = ::llaminar2::rocm_integer_contract::signedDot4(a0, b0.y, acc0[1]);
                    acc0[2] = ::llaminar2::rocm_integer_contract::signedDot4(a0, b0.z, acc0[2]);
                    acc0[3] = ::llaminar2::rocm_integer_contract::signedDot4(a0, b0.w, acc0[3]);

                    const int4 b1 = *reinterpret_cast<const int4*>(b_row + (kg_base1 + kgi) * b_stride);
                    acc1[0] = ::llaminar2::rocm_integer_contract::signedDot4(a1, b1.x, acc1[0]);
                    acc1[1] = ::llaminar2::rocm_integer_contract::signedDot4(a1, b1.y, acc1[1]);
                    acc1[2] = ::llaminar2::rocm_integer_contract::signedDot4(a1, b1.z, acc1[2]);
                    acc1[3] = ::llaminar2::rocm_integer_contract::signedDot4(a1, b1.w, acc1[3]);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < 4; ++c)
                    {
                        if (n_base + c < N)
                        {
                            const int32_t bv0 = *(b_row + (kg_base0 + kgi) * b_stride + c);
                            acc0[c] = ::llaminar2::rocm_integer_contract::signedDot4(a0, bv0, acc0[c]);
                            const int32_t bv1 = *(b_row + (kg_base1 + kgi) * b_stride + c);
                            acc1[c] = ::llaminar2::rocm_integer_contract::signedDot4(a1, bv1, acc1[c]);
                        }
                    }
                }
            }

            // Flush both blocks
            const float scale0 = d_scales_A_blockwise[act_block];
            const float scale1 = d_scales_A_blockwise[act_block + 1];
            #pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                fp_acc[c] += static_cast<float>(acc0[c]) * scale0;
                fp_acc[c] += static_cast<float>(acc1[c]) * scale1;
            }
        }

        // Handle odd tail (at most 1 remaining block)
        if (act_block < my_act_end)
        {
            int32_t block_acc[4] = {0, 0, 0, 0};
            const int kg_base = act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;

            #pragma unroll
            for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + (kg_base + kgi) * 4);

                if (full_cols)
                {
                    const int4 b = *reinterpret_cast<const int4*>(b_row + (kg_base + kgi) * b_stride);
                    block_acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.x, block_acc[0]);
                    block_acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.y, block_acc[1]);
                    block_acc[2] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.z, block_acc[2]);
                    block_acc[3] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b.w, block_acc[3]);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < 4; ++c)
                    {
                        if (n_base + c < N)
                        {
                            const int32_t bv = *(b_row + (kg_base + kgi) * b_stride + c);
                            block_acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, bv, block_acc[c]);
                        }
                    }
                }
            }

            gemv_int8_flush_blockwise_acc(block_acc, fp_acc, d_scales_A_blockwise, act_block);
        }
    }

    // LDS K-reduction: same as standard Q/Wo kernel
    __shared__ float4 s_partial[WK - 1][64];

    if (wave_id > 0 && n_base < N)
    {
        s_partial[wave_id - 1][lane_id] = make_float4(fp_acc[0], fp_acc[1], fp_acc[2], fp_acc[3]);
    }

    __syncthreads();

    if (wave_id == 0 && n_base < N)
    {
        #pragma unroll
        for (int w = 0; w < WK - 1; ++w)
        {
            const float4 partial = s_partial[w][lane_id];
            fp_acc[0] += partial.x;
            fp_acc[1] += partial.y;
            fp_acc[2] += partial.z;
            fp_acc[3] += partial.w;
        }

        if (full_cols)
        {
            const float4 scale = *reinterpret_cast<const float4*>(d_scale_B + n_base);
            if (kb_outer == 1)
            {
                d_C_fp32[n_base + 0] = fp_acc[0] * scale.x * alpha;
                d_C_fp32[n_base + 1] = fp_acc[1] * scale.y * alpha;
                d_C_fp32[n_base + 2] = fp_acc[2] * scale.z * alpha;
                d_C_fp32[n_base + 3] = fp_acc[3] * scale.w * alpha;
            }
            else
            {
                atomicAdd(&d_C_fp32[n_base + 0], fp_acc[0] * scale.x * alpha);
                atomicAdd(&d_C_fp32[n_base + 1], fp_acc[1] * scale.y * alpha);
                atomicAdd(&d_C_fp32[n_base + 2], fp_acc[2] * scale.z * alpha);
                atomicAdd(&d_C_fp32[n_base + 3], fp_acc[3] * scale.w * alpha);
            }
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    const float partial = fp_acc[c] * d_scale_B[n] * alpha;
                    if (kb_outer == 1)
                    {
                        d_C_fp32[n] = partial;
                    }
                    else
                    {
                        atomicAdd(&d_C_fp32[n], partial);
                    }
                }
            }
        }
    }
}


// ============================================================================
// Q/Wo 2x-unrolled LDS K-reduction blockwise kernel — TN=128, CPT=2 variant
//
// Same ILP strategy as the TN=256 version but for the TN=128/CPT=2 tile.
// Uses int2 vectorized loads (8 bytes) like the standard TN=128 kernel,
// but processes 2 activation blocks simultaneously with separate accumulators.
// Beneficial when WK is small enough that each wave gets 2+ blocks.
// ============================================================================
template <int WK, int ACT_BK = 32>
__global__ __launch_bounds__(64 * WK)
void gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    int kb_outer,
    float alpha)
{
    static_assert(WK >= 2 && WK <= 16, "WK must be 2-16");
    static_assert(ACT_BK == 32 || ACT_BK == 64 || ACT_BK == 128, "ACT_BK must be 32, 64, or 128");
    constexpr int KGROUPS_PER_ACT = ACT_BK / 4;

    constexpr int TILE_N = 128;
    constexpr int CPT = 2;
    constexpr int THREADS_PER_WAVE = TILE_N / CPT;
    static_assert(THREADS_PER_WAVE == 64, "TILE_N/CPT must equal 64 (one wavefront)");

    const int wave_id = threadIdx.x / THREADS_PER_WAVE;
    const int lane_id = threadIdx.x % THREADS_PER_WAVE;
    const int n_base = blockIdx.x * TILE_N + lane_id * CPT;
    const int outer_block = blockIdx.y;
    const int act_blocks = K / ACT_BK;

    const int act_blocks_per_outer = (act_blocks + kb_outer - 1) / kb_outer;
    const int outer_act_start = outer_block * act_blocks_per_outer;
    const int outer_act_end = min(outer_act_start + act_blocks_per_outer, act_blocks);
    if (outer_act_start >= act_blocks)
    {
        return;
    }

    const int outer_act_count = outer_act_end - outer_act_start;
    const int act_per_wave_base = outer_act_count / WK;
    const int act_remainder = outer_act_count % WK;
    const int my_act_start = outer_act_start + wave_id * act_per_wave_base + min(wave_id, act_remainder);
    const int my_act_count = act_per_wave_base + (wave_id < act_remainder ? 1 : 0);
    const int my_act_end = my_act_start + my_act_count;

    float fp_acc[2];
    fp_acc[0] = 0.0f;
    fp_acc[1] = 0.0f;

    if (n_base < N && my_act_count > 0)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        // Process pairs of activation blocks for ILP
        int act_block = my_act_start;
        const int pair_end = my_act_start + (my_act_count & ~1);

        #pragma unroll 1
        for (; act_block < pair_end; act_block += 2)
        {
            int32_t acc0[2] = {0, 0};
            int32_t acc1[2] = {0, 0};
            const int kg_base0 = act_block * KGROUPS_PER_ACT;
            const int kg_base1 = (act_block + 1) * KGROUPS_PER_ACT;

            #pragma unroll
            for (int kgi = 0; kgi < KGROUPS_PER_ACT; ++kgi)
            {
                const int32_t a0 = *reinterpret_cast<const int32_t*>(d_A_int8 + (kg_base0 + kgi) * 4);
                const int2 b0 = *reinterpret_cast<const int2*>(b_row + (kg_base0 + kgi) * b_stride);
                acc0[0] = ::llaminar2::rocm_integer_contract::signedDot4(a0, b0.x, acc0[0]);
                acc0[1] = ::llaminar2::rocm_integer_contract::signedDot4(a0, b0.y, acc0[1]);

                const int32_t a1 = *reinterpret_cast<const int32_t*>(d_A_int8 + (kg_base1 + kgi) * 4);
                const int2 b1 = *reinterpret_cast<const int2*>(b_row + (kg_base1 + kgi) * b_stride);
                acc1[0] = ::llaminar2::rocm_integer_contract::signedDot4(a1, b1.x, acc1[0]);
                acc1[1] = ::llaminar2::rocm_integer_contract::signedDot4(a1, b1.y, acc1[1]);
            }

            const float scale0 = d_scales_A_blockwise[act_block];
            const float scale1 = d_scales_A_blockwise[act_block + 1];
            fp_acc[0] += static_cast<float>(acc0[0]) * scale0;
            fp_acc[1] += static_cast<float>(acc0[1]) * scale0;
            fp_acc[0] += static_cast<float>(acc1[0]) * scale1;
            fp_acc[1] += static_cast<float>(acc1[1]) * scale1;
        }

        // Handle odd tail
        if (act_block < my_act_end)
        {
            int32_t block_acc[2] = {0, 0};
            gemv_int8_accumulate_act_block<2, KGROUPS_PER_ACT>(block_acc, d_A_int8, b_row, b_stride,
                                           act_block * KGROUPS_PER_ACT);
            gemv_int8_flush_blockwise_acc(block_acc, fp_acc, d_scales_A_blockwise, act_block);
        }
    }

    // LDS K-reduction
    __shared__ float s_partial[WK - 1][TILE_N];

    if (wave_id > 0 && n_base < N)
    {
        s_partial[wave_id - 1][lane_id * CPT + 0] = fp_acc[0];
        s_partial[wave_id - 1][lane_id * CPT + 1] = fp_acc[1];
    }

    __syncthreads();

    if (wave_id == 0 && n_base < N)
    {
        #pragma unroll
        for (int w = 0; w < WK - 1; ++w)
        {
            fp_acc[0] += s_partial[w][lane_id * CPT + 0];
            fp_acc[1] += s_partial[w][lane_id * CPT + 1];
        }

        const float partial0 = fp_acc[0] * d_scale_B[n_base + 0] * alpha;
        const float partial1 = fp_acc[1] * d_scale_B[n_base + 1] * alpha;
        if (kb_outer == 1)
        {
            float2 out_val = make_float2(partial0, partial1);
            *reinterpret_cast<int2*>(&d_C_fp32[n_base]) =
                *reinterpret_cast<const int2*>(&out_val);
        }
        else
        {
            atomicAdd(&d_C_fp32[n_base + 0], partial0);
            atomicAdd(&d_C_fp32[n_base + 1], partial1);
        }
    }
}



// ============================================================================

// C API: Launch wrappers
// ============================================================================

extern "C" {

bool rocmGemv_int8_scatter_vnni_blockwise(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    float* d_C_fp32,
    const float* d_scales_A_blockwise,
    const float* d_scales_B,
    const float* d_bias,
    float* d_partial_buf,
    int* d_counter,
    int N, int K,
    float alpha, float beta,
    const float* d_C_existing,
    int device_id, void* stream);


void rocmGemv_int8_vnni_set_tuning_overrides(
    int tn,
    int kb)
{
    g_int8_vnni_tn_override.store(tn, std::memory_order_relaxed);
    g_int8_vnni_kb_override.store(kb, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_reset_tuning_overrides()
{
    rocmGemv_int8_vnni_set_tuning_overrides(-1, -1);
}

void rocmGemv_int8_vnni_set_wk_override(int wk)
{
    g_int8_vnni_wk_override.store(wk, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_set_unroll_override(int unroll)
{
    g_int8_vnni_unroll_override.store(unroll, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_set_act_block_override(int act_bk)
{
    g_int8_vnni_act_block_override.store(act_bk, std::memory_order_relaxed);
}

int rocmGemv_int8_vnni_get_act_block_k()
{
    int v = g_int8_vnni_act_block_override.load(std::memory_order_relaxed);
    return (v == 64 || v == 128) ? v : 32;
}

void rocmGemv_int8_vnni_reset_qwo_overrides()
{
    g_int8_vnni_wk_override.store(-1, std::memory_order_relaxed);
    g_int8_vnni_unroll_override.store(-1, std::memory_order_relaxed);
    g_int8_vnni_act_block_override.store(-1, std::memory_order_relaxed);
    g_int8_vnni_skip_memset_override.store(0, std::memory_order_relaxed);
    g_int8_vnni_force_pair_override.store(0, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_set_skip_memset(int skip)
{
    g_int8_vnni_skip_memset_override.store(skip, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_set_force_pair(int force)
{
    g_int8_vnni_force_pair_override.store(force, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_set_wide_tuning_overrides(
    int wide_tn,
    int wide_cpt,
    int wide_vec_load)
{
    g_int8_vnni_wide_tn_override.store(wide_tn, std::memory_order_relaxed);
    g_int8_vnni_wide_cpt_override.store(wide_cpt, std::memory_order_relaxed);
    g_int8_vnni_wide_vec_load_override.store(wide_vec_load, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_reset_wide_tuning_overrides()
{
    rocmGemv_int8_vnni_set_wide_tuning_overrides(-1, -1, -1);
}

bool rocmGemv_int8_int8_int32_vnni(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int N, int K,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_int32_vnni] K must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    const bool use_vec4 = ((N % 4) == 0);
    const int wide_tn_override = g_int8_vnni_wide_tn_override.load(std::memory_order_relaxed);
    const int wide_cpt_override = g_int8_vnni_wide_cpt_override.load(std::memory_order_relaxed);
    const int wide_vec_load_override = g_int8_vnni_wide_vec_load_override.load(std::memory_order_relaxed);

    int wide_tn = GEMV_INT8_VNNI_WIDE_TILE_N;
    if (wide_tn_override == 128 || wide_tn_override == 256 || wide_tn_override == 512)
    {
        wide_tn = wide_tn_override;
    }

    int wide_cpt = GEMV_INT8_VNNI_WIDE_DEFAULT_CPT;
    if (wide_cpt_override == 2 || wide_cpt_override == 4)
    {
        wide_cpt = wide_cpt_override;
    }

    bool wide_vec_load = true;
    if (wide_vec_load_override == 0)
    {
        wide_vec_load = false;
    }
    else if (wide_vec_load_override == 1)
    {
        wide_vec_load = true;
    }

    const bool use_wide_vec4 = (N >= 8 * K) && (N >= wide_tn);
    // Grid-level K-parallel: handles ALL non-wide_vec4 shapes with N%4==0.
    // Previous wide2 path (N>=3K, N<8K) only generated N/256 blocks → 148 waves
    // for FFN Gate/Up, achieving just 17% peak BW. Grid_kpar K-splits via
    // blockIdx.y to generate thousands of blocks, fully saturating the GPU.
    // Profiling: grid_kpar achieves 488-649 GB/s (48-63% peak) vs wide2's 176 GB/s.
    const bool use_grid_kpar = !use_wide_vec4 && use_vec4 && (N >= GEMV_INT8_VNNI_GRID_KPAR_TILE_N);
    const bool use_square = !use_wide_vec4 && !use_grid_kpar && (N >= GEMV_INT8_VNNI_SQUARE_TILE_N);
    const int int8_tn_override = g_int8_vnni_tn_override.load(std::memory_order_relaxed);
    const int int8_kb_override = g_int8_vnni_kb_override.load(std::memory_order_relaxed);

    if (use_wide_vec4)
    {
        const int grid_n = (N + wide_tn - 1) / wide_tn;
        dim3 block(wide_tn / wide_cpt, 1, 1);

        if (wide_tn == 128)
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<128, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<128, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<128, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<128, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
        }
        else if (wide_tn == 512)
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<512, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<512, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<512, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<512, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
        }
        else
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<256, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<256, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<256, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<256, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
        }
    }
    else if (use_grid_kpar)
    {
        // Grid-level K-parallel with atomicAdd: KB independent blocks per output tile
        // Each block processes K/KB fraction; output must be zeroed before launch
        //
        // KB heuristic: unified formula for all shapes.
        //
        // Key insight from exhaustive TN×KB sweep (MI50 gfx906, Qwen7B shapes):
        //   ~8 waves/CU is the sweet spot for ALL shapes. More waves just adds
        //   atomicAdd contention per N-tile without improving BW utilization.
        //   Additionally, KB values that evenly divide k_groups avoid tail-effect
        //   work imbalance (some wavefronts processing fewer k-groups than others).
        //
        // Sweep results (optimal KB per shape):
        //   Q/Wo (3584×3584):     KB=16, 7.5 w/CU, 56 kgrp/wave → 456 GB/s (46%)
        //   K/V (512×3584):       KB=56, 3.7 w/CU, 16 kgrp/wave → 129 GB/s (13%)
        //   FFN Gate/Up (18944×3584): KB=4, 9.9 w/CU, 224 kgrp/wave → 756 GB/s (76%)
        //   FFN Down (3584×18944):    KB=16, 7.5 w/CU, 296 kgrp/wave → 760 GB/s (76%)
        //
        // Strategy:
        //   1. Compute min KB for ~8 waves/CU (memory latency hiding)
        //   2. Cap so inner loop runs ≥16 k-groups (proven floor from sweep)
        //   3. Round to nearest factor of k_groups (avoid uneven work distribution)
        //      Bias toward lower factor on ties (fewer atomicAdds = less contention)
        constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
        constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
        constexpr int TN_WIDE = GEMV_INT8_VNNI_WIDE_TILE_N;  // 256
        constexpr int NUM_CUS = 60;  // MI50

        int tn = TN;
        if (int8_tn_override == TN_WIDE)
        {
            tn = TN_WIDE;
        }
        else if (int8_tn_override == TN)
        {
            tn = TN;
        }

        const int grid_n = (N + tn - 1) / tn;
        const int k_groups = K / 4;

        constexpr int TARGET_WAVES_PER_CU = 8;
        constexpr int MIN_KGROUPS_PER_WAVE = 16;  // proven floor from sweep

        int target_blocks = TARGET_WAVES_PER_CU * NUM_CUS;  // 480
        int kb_raw = max(2, (target_blocks + grid_n - 1) / grid_n);

        // Cap: ensure minimum inner loop length
        int kb_max = max(2, k_groups / MIN_KGROUPS_PER_WAVE);
        kb_raw = min(kb_raw, kb_max);

        // Round to nearest factor of k_groups (prefer lower on tie → fewer atomicAdds)
        int kb = kb_raw;
        if (k_groups % kb_raw != 0)
        {
            for (int d = 1; d < kb_raw; ++d)
            {
                if (kb_raw - d >= 2 && k_groups % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
                if (kb_raw + d <= kb_max && k_groups % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
            }
        }

        if (int8_kb_override > 0)
        {
            kb = int8_kb_override;
        }

        (void)hipMemsetAsync(d_C_int32, 0, N * sizeof(int32_t), static_cast<hipStream_t>(stream));  // NOLINT(clang-diagnostic-unused-result)

        if (tn == TN_WIDE)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_vnni_kernel_t<TN_WIDE, CPT>),
                dim3(grid_n, kb), dim3(TN_WIDE / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_B_int8_vnni, d_C_int32, N, K, kb);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_vnni_kernel_t<TN, CPT>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_B_int8_vnni, d_C_int32, N, K, kb);
        }
    }
    else if (use_square)
    {
        const int grid_n = (N + GEMV_INT8_VNNI_SQUARE_TILE_N - 1) / GEMV_INT8_VNNI_SQUARE_TILE_N;
        dim3 block(GEMV_INT8_VNNI_SQUARE_TILE_N, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_square_vnni_kernel,
            dim3(grid_n), block, 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
    }
    else
    {
        const int grid_n = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
        hipLaunchKernelGGL(
            gemv_int8_int8_vnni_kernel,
            dim3(grid_n), dim3(GEMV_TILE_N), 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_int32_vnni] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Pair dispatch for tiny K/V shapes: launches the LDS K-reduce pair kernel
// for two projections with same K, each with N <= 512 and K >= 2048.
// ---------------------------------------------------------------------------
bool rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_pair(
    const int8_t* d_A_int8,
    const float* d_scales_A_blockwise,
    const int8_t* d_B0,
    const int8_t* d_B1,
    float* d_C0,
    float* d_C1,
    const float* d_scales_B0,
    const float* d_scales_B1,
    int N0, int N1,
    int K,
    float alpha,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
    constexpr int WK = 8;

    const int grid_n0 = (N0 + TN - 1) / TN;
    const int grid_n1 = (N1 + TN - 1) / TN;
    const int total_grid_n = grid_n0 + grid_n1;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    const int int8_kb_override = g_int8_vnni_kb_override.load(std::memory_order_relaxed);
    // Phase 3: Use unified KB selector (two-regime + occupancy floor + KB cap)
    // instead of legacy tiny-KV selector. Aligns pair dispatch with Phase 1
    // individual dispatch tuning.
    const int kb_outer = select_blockwise_qwo_outer_splits(total_grid_n, act_blocks, int8_kb_override, WK);

    // Zero outputs when KB > 1 (kernel uses atomicAdd)
    if (kb_outer > 1)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_init_output_pair_kernel_t<TN, CPT>),
            dim3(total_grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
            d_C0, d_C1, nullptr, nullptr,
            N0, N1, grid_n0, 0.0f);
    }

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_int8_int8_lds_kreduce_pair_blockwise_scaled_kernel_t<TN, CPT, WK>),
        dim3(total_grid_n, kb_outer), dim3((TN / CPT) * WK), 0, static_cast<hipStream_t>(stream),
        d_A_int8,
        d_B0, d_B1,
        d_C0, d_C1,
        d_scales_A_blockwise,
        d_scales_B0, d_scales_B1,
        N0, N1, grid_n0,
        K, kb_outer, alpha);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_pair] launch failed: %s (N0=%d N1=%d K=%d kb=%d)\n",
                hipGetErrorString(err), N0, N1, K, kb_outer);
        return false;
    }

    return true;
}

bool rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    float* d_C_fp32,
    const float* d_scales_A_blockwise,
    const float* d_scale_B,
    int N, int K,
    float alpha,
    float beta,
    const float* d_C_existing,
    const float* d_bias,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled] K must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }
    // Check K is divisible by the current act_block size (32, 64, or 128)
    const int effective_act_bk = rocmGemv_int8_vnni_get_act_block_k();
    if ((K % effective_act_bk) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled] K must be multiple of act_block_k=%d (N=%d K=%d)\n",
                effective_act_bk, N, K);
        return false;
    }
    if (beta != 0.0f && d_C_existing == nullptr)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled] d_C_existing required when beta != 0 (N=%d K=%d)\n", N, K);
        return false;
    }

    const bool use_vec4 = ((N % 4) == 0);
    const int wide_tn_override = g_int8_vnni_wide_tn_override.load(std::memory_order_relaxed);
    const int wide_cpt_override = g_int8_vnni_wide_cpt_override.load(std::memory_order_relaxed);
    const int wide_vec_load_override = g_int8_vnni_wide_vec_load_override.load(std::memory_order_relaxed);

    int wide_tn = GEMV_INT8_VNNI_WIDE_TILE_N;
    if (wide_tn_override == 128 || wide_tn_override == 256 || wide_tn_override == 512)
    {
        wide_tn = wide_tn_override;
    }

    int wide_cpt = GEMV_INT8_VNNI_WIDE_DEFAULT_CPT;
    if (wide_cpt_override == 2 || wide_cpt_override == 4)
    {
        wide_cpt = wide_cpt_override;
    }

    bool wide_vec_load = true;
    if (wide_vec_load_override == 0)
    {
        wide_vec_load = false;
    }
    else if (wide_vec_load_override == 1)
    {
        wide_vec_load = true;
    }

    const bool use_wide_vec4 = (N >= 8 * K) && (N >= wide_tn);
    const int int8_tn_override = g_int8_vnni_tn_override.load(std::memory_order_relaxed);
    const int int8_kb_override = g_int8_vnni_kb_override.load(std::memory_order_relaxed);
    const int qwo_tn = (int8_tn_override == 128 || int8_tn_override == 256) ? int8_tn_override : 128;

    // Unified LDS k-reduce path: routes all vec4-aligned shapes through the
    // high-performance LDS k-reduce kernel. The kernel has N bounds checking
    // (n_base < N guard on all reads/writes), so it handles N < TN=128 correctly
    // with grid_n=1 — excess threads simply skip. Previously gated on N >= 128,
    // but that caused TP-sharded K/V projections (N=32-64) to fall through to the
    // slow fallback kernel (TILE_N=256, no KB splitting, 2-4x slower).
    // The occupancy selector adapts KB to fill CUs at any grid_n.
    const bool use_lds_kreduce_blockwise = !use_wide_vec4
        && use_vec4;  // Any vec4-aligned shape (N % 4 == 0)
    const bool use_grid_kpar = !use_wide_vec4 && !use_lds_kreduce_blockwise && use_vec4 && (N >= GEMV_INT8_VNNI_GRID_KPAR_TILE_N);
    const bool use_square = !use_wide_vec4 && !use_grid_kpar && (N >= GEMV_INT8_VNNI_SQUARE_TILE_N);

    if (use_lds_kreduce_blockwise)
    {
        // Unified Q/Wo dispatch with WK, unroll, and act_block overrides
        const int wk_ov = g_int8_vnni_wk_override.load(std::memory_order_relaxed);
        const int unroll_ov = g_int8_vnni_unroll_override.load(std::memory_order_relaxed);
        const int tn = qwo_tn;  // 128 or 256
        const int grid_n = (N + tn - 1) / tn;

        // WK=8 default (tuned). WK=16 infrastructure available via override for future
        // tuning — static_asserts and switch cases support WK up to 16, but auto-select
        // stays at 8 since atomicAdd is not the bottleneck for small-grid_n shapes
        // (latency floor dominates; total wave count is unchanged since WK×KB = const).
        const int wk = (wk_ov >= 2 && wk_ov <= 16) ? wk_ov : 8;
        const bool use_unroll2 = (unroll_ov == -1) ? true : (unroll_ov == 2);  // default: unroll=2 (tuned)
        const int act_bk = rocmGemv_int8_vnni_get_act_block_k();  // 32, 64, or 128

        const int act_blocks = K / act_bk;
        const int kb_outer = select_blockwise_qwo_outer_splits(grid_n, act_blocks, int8_kb_override, wk);

        if (kb_outer > 1 && beta == 0.0f && d_bias == nullptr)
        {
            (void)hipMemsetAsync(d_C_fp32, 0, static_cast<size_t>(N) * sizeof(float), static_cast<hipStream_t>(stream));
        }
        else if (beta != 0.0f || d_bias != nullptr)
        {
            if (tn == 256)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_init_output_kernel_t<256, 4>),
                    dim3(grid_n), dim3(64), 0, static_cast<hipStream_t>(stream),
                    d_C_fp32, d_C_existing, d_bias, N, beta);
            }
            else
            {
                constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;
                constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_init_output_kernel_t<TN, CPT>),
                    dim3(grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                    d_C_fp32, d_C_existing, d_bias, N, beta);
            }
        }

        // Macro to dispatch with runtime WK -> compile-time template arg
        #define LAUNCH_QWO_KERNEL(KERNEL_TEMPLATE, WK_VAL)                                        \
            hipLaunchKernelGGL(                                                                    \
                HIP_KERNEL_NAME(KERNEL_TEMPLATE<WK_VAL>),                                         \
                dim3(grid_n, kb_outer), dim3(64 * WK_VAL), 0, static_cast<hipStream_t>(stream),   \
                d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B,               \
                N, K, kb_outer, alpha)

        #define LAUNCH_QWO_KERNEL_ABK(KERNEL_TEMPLATE, WK_VAL, ABK_VAL)                            \
            hipLaunchKernelGGL(                                                                    \
                HIP_KERNEL_NAME(KERNEL_TEMPLATE<WK_VAL, ABK_VAL>),                                \
                dim3(grid_n, kb_outer), dim3(64 * WK_VAL), 0, static_cast<hipStream_t>(stream),   \
                d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B,               \
                N, K, kb_outer, alpha)

        #define LAUNCH_LDS_KERNEL(TN_VAL, CPT_VAL, WK_VAL)                                       \
            hipLaunchKernelGGL(                                                                    \
                HIP_KERNEL_NAME(gemv_int8_int8_lds_kreduce_blockwise_scaled_kernel_t<TN_VAL, CPT_VAL, WK_VAL>), \
                dim3(grid_n, kb_outer), dim3(64 * WK_VAL), 0, static_cast<hipStream_t>(stream),   \
                d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B,               \
                N, K, kb_outer, alpha)

        if (tn == 256 && use_unroll2)
        {
            switch (wk)
            {
            case 2: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_lds_kreduce_blockwise_scaled_kernel_t, 2); break;
            case 3: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_lds_kreduce_blockwise_scaled_kernel_t, 3); break;
            case 6: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_lds_kreduce_blockwise_scaled_kernel_t, 6); break;
            case 8: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_lds_kreduce_blockwise_scaled_kernel_t, 8); break;
            case 16: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_lds_kreduce_blockwise_scaled_kernel_t, 16); break;
            default: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_lds_kreduce_blockwise_scaled_kernel_t, 4); break;
            }
        }
        else if (tn == 256)
        {
            switch (wk)
            {
            case 2: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_lds_kreduce_blockwise_scaled_kernel_t, 2); break;
            case 3: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_lds_kreduce_blockwise_scaled_kernel_t, 3); break;
            case 6: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_lds_kreduce_blockwise_scaled_kernel_t, 6); break;
            case 8: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_lds_kreduce_blockwise_scaled_kernel_t, 8); break;
            case 16: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_lds_kreduce_blockwise_scaled_kernel_t, 16); break;
            default: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_lds_kreduce_blockwise_scaled_kernel_t, 4); break;
            }
        }
        else if (tn == 128 && use_unroll2)
        {
            // TN=128, CPT=2, unroll2 variant — dispatch with act_block size
            if (act_bk == 128)
            {
                switch (wk)
                {
                case 2: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 2, 128); break;
                case 3: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 3, 128); break;
                case 6: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 6, 128); break;
                case 8: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 8, 128); break;
                case 16: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 16, 128); break;
                default: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 4, 128); break;
                }
            }
            else if (act_bk == 64)
            {
                switch (wk)
                {
                case 2: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 2, 64); break;
                case 3: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 3, 64); break;
                case 6: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 6, 64); break;
                case 8: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 8, 64); break;
                case 16: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 16, 64); break;
                default: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 4, 64); break;
                }
            }
            else
            {
                switch (wk)
                {
                case 2: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 2); break;
                case 3: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 3); break;
                case 6: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 6); break;
                case 8: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 8); break;
                case 16: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 16); break;
                default: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 4); break;
                }
            }
        }
        else
        {
            // TN=128, CPT=2 standard
            constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;
            constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;
            switch (wk)
            {
            case 2: LAUNCH_LDS_KERNEL(TN, CPT, 2); break;
            case 3: LAUNCH_LDS_KERNEL(TN, CPT, 3); break;
            case 6: LAUNCH_LDS_KERNEL(TN, CPT, 6); break;
            case 8: LAUNCH_LDS_KERNEL(TN, CPT, 8); break;
            case 16: LAUNCH_LDS_KERNEL(TN, CPT, 16); break;
            default: LAUNCH_LDS_KERNEL(TN, CPT, 4); break;
            }
        }

        #undef LAUNCH_QWO_KERNEL
        #undef LAUNCH_QWO_KERNEL_ABK
        #undef LAUNCH_LDS_KERNEL

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled] qwo dispatch failed: %s (N=%d K=%d tn=%d wk=%d unroll=%d act_bk=%d kb=%d)\n",
                    hipGetErrorString(err), N, K, tn, wk, use_unroll2 ? 2 : 1, act_bk, kb_outer);
            return false;
        }

        return true;
    }

    if (use_grid_kpar)
    {
        constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;
        constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;
        constexpr int TN_WIDE = GEMV_INT8_VNNI_WIDE2_TILE_N;

        int tn = TN;
        if (int8_tn_override == TN_WIDE)
        {
            tn = TN_WIDE;
        }
        else if (int8_tn_override == TN)
        {
            tn = TN;
        }

        const int grid_n = (N + tn - 1) / tn;
        const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

        // For tiny-grid decode shapes, the direct/scatter family is structurally better
        // than atomic grid_kpar because it avoids repeated output updates and can reduce
        // once per tile. Returning false asks the higher-level GEMM wrapper to use
        // its explicit workspace-backed scatter path instead of allocating scratch
        // inside this HIP translation unit.
        if (grid_n <= 8)
        {
            return false;
        }

        const int kb = select_blockwise_split_count(grid_n, act_blocks, int8_kb_override);

        if (kb > 1 && beta == 0.0f && d_bias == nullptr)
        {
            (void)hipMemsetAsync(d_C_fp32, 0, static_cast<size_t>(N) * sizeof(float), static_cast<hipStream_t>(stream));
        }
        else if (beta != 0.0f || d_bias != nullptr)
        {
            if (tn == TN_WIDE)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_init_output_kernel_t<TN_WIDE, CPT>),
                    dim3(grid_n), dim3(TN_WIDE / CPT), 0, static_cast<hipStream_t>(stream),
                    d_C_fp32, d_C_existing, d_bias, N, beta);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_init_output_kernel_t<TN, CPT>),
                    dim3(grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                    d_C_fp32, d_C_existing, d_bias, N, beta);
            }
        }

        if (tn == TN_WIDE)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_vnni_blockwise_scaled_kernel_t<TN_WIDE, CPT>),
                dim3(grid_n, kb), dim3(TN_WIDE / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, kb, alpha);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_vnni_blockwise_scaled_kernel_t<TN, CPT>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, kb, alpha);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled] grid_kpar launch failed: %s (N=%d K=%d kb=%d)\n",
                    hipGetErrorString(err), N, K, kb);
            return false;
        }

        return true;
    }

    if (use_wide_vec4)
    {
        const int grid_n = (N + wide_tn - 1) / wide_tn;
        dim3 block(wide_tn / wide_cpt, 1, 1);

        if (wide_tn == 128)
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<128, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<128, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<128, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<128, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
        }
        else if (wide_tn == 512)
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<512, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<512, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<512, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<512, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
        }
        else
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<256, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<256, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<256, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<256, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
        }
    }
    else if (use_square)
    {
        const int grid_n = (N + GEMV_INT8_VNNI_SQUARE_TILE_N - 1) / GEMV_INT8_VNNI_SQUARE_TILE_N;
        dim3 block(GEMV_INT8_VNNI_SQUARE_TILE_N, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_square_vnni_blockwise_scaled_kernel,
            dim3(grid_n), block, 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B,
            N, K, alpha, beta, d_C_existing, d_bias);
    }
    else
    {
        const int grid_n = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
        hipLaunchKernelGGL(
            gemv_int8_int8_vnni_blockwise_scaled_kernel,
            dim3(grid_n), dim3(GEMV_TILE_N), 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B,
            N, K, alpha, beta, d_C_existing, d_bias);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

bool rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_batched(
    const int8_t* d_A_int8,
    const float* d_scales_A_blockwise,
    int num_projections,
    const int8_t* const* d_B_ptrs,
    float* const* d_C_ptrs,
    const float* const* d_scales_B_ptrs,
    const float* const* d_bias_ptrs,
    const int* N_per_proj,
    int K,
    float alpha,
    float beta,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (num_projections < 1 || num_projections > MAX_GEMV_BATCH)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_batched] Invalid num_projections=%d (max=%d)\n",
                num_projections, MAX_GEMV_BATCH);
        return false;
    }
    if ((K % 4) != 0 || (K % GEMV_INT8_VNNI_ACT_BLOCK_K) != 0)
    {
        return false;
    }

    const int wide_tn_override = g_int8_vnni_wide_tn_override.load(std::memory_order_relaxed);
    const int wide_cpt_override = g_int8_vnni_wide_cpt_override.load(std::memory_order_relaxed);
    const int wide_vec_load_override = g_int8_vnni_wide_vec_load_override.load(std::memory_order_relaxed);

    int wide_tn = GEMV_INT8_VNNI_WIDE_TILE_N;
    if (wide_tn_override == 128 || wide_tn_override == 256 || wide_tn_override == 512)
    {
        wide_tn = wide_tn_override;
    }

    int wide_cpt = GEMV_INT8_VNNI_WIDE_DEFAULT_CPT;
    if (wide_cpt_override == 2 || wide_cpt_override == 4)
    {
        wide_cpt = wide_cpt_override;
    }

    bool wide_vec_load = true;
    if (wide_vec_load_override == 0)
    {
        wide_vec_load = false;
    }
    else if (wide_vec_load_override == 1)
    {
        wide_vec_load = true;
    }

    enum class GemvBatchKernelFamily
    {
        Wide,
        Square,
        Generic
    };

    GemvBatchKernelFamily families[MAX_GEMV_BATCH] = {};
    bool force_generic_pair_member[MAX_GEMV_BATCH] = {};

    const bool prefer_generic_pair_path =
        (num_projections == 2) &&
        (N_per_proj[0] == N_per_proj[1]) &&
        (N_per_proj[0] <= 2 * GEMV_INT8_VNNI_SQUARE_TILE_N);

    for (int b = 0; b + 1 < num_projections; ++b)
    {
        const bool pair_same_n = (N_per_proj[b] == N_per_proj[b + 1]);
        const bool pair_small_n = (N_per_proj[b] <= 2 * GEMV_INT8_VNNI_SQUARE_TILE_N);
        const bool pair_use_wide = (N_per_proj[b] >= 8 * K) && (N_per_proj[b] >= wide_tn);
        if (pair_same_n && pair_small_n && !pair_use_wide)
        {
            force_generic_pair_member[b] = true;
            force_generic_pair_member[b + 1] = true;
            ++b;
        }
    }

    for (int b = 0; b < num_projections; ++b)
    {
        if ((N_per_proj[b] % 4) != 0)
        {
            return false;
        }

        const bool use_wide = (N_per_proj[b] >= 8 * K) && (N_per_proj[b] >= wide_tn);
    const bool use_square = (!prefer_generic_pair_path) && (!force_generic_pair_member[b]) && (!use_wide) && (N_per_proj[b] >= GEMV_INT8_VNNI_SQUARE_TILE_N);
        families[b] = use_wide ? GemvBatchKernelFamily::Wide
                               : (use_square ? GemvBatchKernelFamily::Square
                                             : GemvBatchKernelFamily::Generic);
    }

    auto launch_single_projection = [&](int proj) -> bool {
        return rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
            d_A_int8,
            d_B_ptrs[proj],
            d_C_ptrs[proj],
            d_scales_A_blockwise,
            d_scales_B_ptrs[proj],
            N_per_proj[proj],
            K,
            alpha,
            beta,
            beta != 0.0f ? d_C_ptrs[proj] : nullptr,
            d_bias_ptrs[proj],
            device_id,
            stream);
    };

    auto launch_group = [&](int start_idx, int end_idx, GemvBatchKernelFamily family) -> bool {
        if ((end_idx - start_idx) == 1)
        {
            return launch_single_projection(start_idx);
        }

        // Fast path: LDS K-reduce pair kernel for tiny K/V projections.
        // Provides ~1.75x speedup over 2× individual tiny_kv launches by
        // doubling CU occupancy and eliminating launch overhead.
        // Phase 3: Removed K>=2048 gate — pair kernel works for all K values
        // (including 0.5B models with K=896). The unified KB selector handles
        // occupancy tuning for any K.
        const bool is_tiny_kv_pair =
            (family == GemvBatchKernelFamily::Generic) &&
            ((end_idx - start_idx) == 2) &&
            (N_per_proj[start_idx] <= 512) &&
            (N_per_proj[start_idx + 1] <= 512) &&
            (beta == 0.0f) &&
            (d_bias_ptrs[start_idx] == nullptr) &&
            (d_bias_ptrs[start_idx + 1] == nullptr);

        if (is_tiny_kv_pair)
        {
            return rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_pair(
                d_A_int8, d_scales_A_blockwise,
                d_B_ptrs[start_idx], d_B_ptrs[start_idx + 1],
                d_C_ptrs[start_idx], d_C_ptrs[start_idx + 1],
                d_scales_B_ptrs[start_idx], d_scales_B_ptrs[start_idx + 1],
                N_per_proj[start_idx], N_per_proj[start_idx + 1],
                K, alpha,
                device_id, stream);
        }

        const bool force_pair = g_int8_vnni_force_pair_override.load(std::memory_order_relaxed) != 0;
        const bool use_split_single_pair_path =
            !force_pair &&
            (family == GemvBatchKernelFamily::Generic) &&
            ((end_idx - start_idx) == 2) &&
            (N_per_proj[start_idx + 1] == N_per_proj[start_idx]) &&
            (N_per_proj[start_idx] <= 2 * GEMV_INT8_VNNI_SQUARE_TILE_N) &&
            ((N_per_proj[start_idx] >= 2 * GEMV_INT8_VNNI_SQUARE_TILE_N) ||
             ((N_per_proj[start_idx] <= GEMV_INT8_VNNI_SQUARE_TILE_N) && (K >= 2048)));

        if (use_split_single_pair_path)
        {
            return launch_single_projection(start_idx) &&
                   launch_single_projection(start_idx + 1);
        }

        GemvBatchArgs args = {};
        args.num_batches = end_idx - start_idx;
        args.total_grid_n = 0;
        args.total_N = 0;

        const int tile_n = family == GemvBatchKernelFamily::Wide
            ? wide_tn
            : (family == GemvBatchKernelFamily::Square ? GEMV_INT8_VNNI_SQUARE_TILE_N
                                                       : GEMV_INT8_VNNI_GRID_KPAR_TILE_N);

        for (int src = start_idx; src < end_idx; ++src)
        {
            const int dst = src - start_idx;
            args.d_B[dst] = d_B_ptrs[src];
            args.d_C[dst] = d_C_ptrs[src];
            args.d_scales_B[dst] = d_scales_B_ptrs[src];
            args.d_bias[dst] = d_bias_ptrs[src];
            args.N[dst] = N_per_proj[src];
            args.grid_n_offset[dst] = args.total_grid_n;
            args.n_offset[dst] = args.total_N;

            args.total_grid_n += (N_per_proj[src] + tile_n - 1) / tile_n;
            args.total_N += N_per_proj[src];
        }

        if (family == GemvBatchKernelFamily::Wide)
        {
            dim3 block(wide_tn / wide_cpt, 1, 1);
            if (wide_tn == 128)
            {
                if (wide_cpt == 2)
                {
                    if (wide_vec_load)
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<128, 2, true>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                    else
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<128, 2, false>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                }
                else
                {
                    if (wide_vec_load)
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<128, 4, true>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                    else
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<128, 4, false>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                }
            }
            else if (wide_tn == 512)
            {
                if (wide_cpt == 2)
                {
                    if (wide_vec_load)
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<512, 2, true>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                    else
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<512, 2, false>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                }
                else
                {
                    if (wide_vec_load)
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<512, 4, true>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                    else
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<512, 4, false>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                }
            }
            else
            {
                if (wide_cpt == 2)
                {
                    if (wide_vec_load)
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<256, 2, true>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                    else
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<256, 2, false>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                }
                else
                {
                    if (wide_vec_load)
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<256, 4, true>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                    else
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<256, 4, false>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                }
            }
        }
        else if (family == GemvBatchKernelFamily::Square)
        {
            dim3 block(GEMV_INT8_VNNI_SQUARE_TILE_N, 1, 1);
            hipLaunchKernelGGL(
                gemv_int8_int8_square_batched_vnni_blockwise_scaled_kernel,
                dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
        }
        else
        {
            constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;
            constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;
            constexpr int TN_WIDE = GEMV_INT8_VNNI_WIDE2_TILE_N;
            constexpr int NUM_CUS = 60;
            constexpr int TARGET_WAVES_PER_CU = 8;
            constexpr int MIN_ACT_BLOCKS_PER_WAVE = 2;
            const int int8_tn_override = g_int8_vnni_tn_override.load(std::memory_order_relaxed);
            const int int8_kb_override = g_int8_vnni_kb_override.load(std::memory_order_relaxed);
            const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

            int tn = TN;
            if (int8_tn_override == TN_WIDE)
            {
                tn = TN_WIDE;
            }
            else if (int8_tn_override == TN)
            {
                tn = TN;
            }

            const int grid_n = (args.total_N + tn - 1) / tn;
            int target_blocks = TARGET_WAVES_PER_CU * NUM_CUS;
            int kb_raw = max(1, (target_blocks + args.total_grid_n - 1) / args.total_grid_n);
            int kb_max = max(1, act_blocks / MIN_ACT_BLOCKS_PER_WAVE);
            kb_raw = min(kb_raw, kb_max);

            int kb = kb_raw;
            if (act_blocks % kb_raw != 0)
            {
                for (int d = 1; d < kb_raw; ++d)
                {
                    if (kb_raw - d >= 1 && act_blocks % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
                    if (kb_raw + d <= kb_max && act_blocks % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
                }
            }

            if (int8_kb_override > 0)
            {
                kb = min(int8_kb_override, act_blocks);
            }

            // Small generic batched groups (for example decode KV pairs with
            // 2x512 outputs) do not benefit from the aggressive occupancy-
            // driven split count used by the single-projection path. The
            // direct batched kernel is consistently best around KB<=4 there,
            // while larger split counts just add atomic/update overhead.
            if (args.total_grid_n <= 8 && int8_kb_override <= 0)
            {
                kb = min(kb, 4);
            }

            const bool use_pair_fast_path = (args.num_batches == 2);

            if (use_pair_fast_path)
            {
                if (tn == TN_WIDE)
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(gemv_init_output_pair_kernel_t<TN_WIDE, CPT>),
                        dim3(args.total_grid_n), dim3(TN_WIDE / CPT), 0, static_cast<hipStream_t>(stream),
                        args.d_C[0], args.d_C[1], args.d_bias[0], args.d_bias[1],
                        args.N[0], args.N[1], args.grid_n_offset[1], beta);
                }
                else
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(gemv_init_output_pair_kernel_t<TN, CPT>),
                        dim3(args.total_grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                        args.d_C[0], args.d_C[1], args.d_bias[0], args.d_bias[1],
                        args.N[0], args.N[1], args.grid_n_offset[1], beta);
                }

                if (tn == TN_WIDE)
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_pair_vnni_blockwise_scaled_kernel_t<TN_WIDE, CPT>),
                        dim3(args.total_grid_n, kb), dim3(TN_WIDE / CPT), 0, static_cast<hipStream_t>(stream),
                        d_A_int8, d_scales_A_blockwise,
                        args.d_B[0], args.d_B[1],
                        args.d_C[0], args.d_C[1],
                        args.d_scales_B[0], args.d_scales_B[1],
                        args.N[0], args.N[1], args.grid_n_offset[1],
                        K, kb, alpha);
                }
                else
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_pair_vnni_blockwise_scaled_kernel_t<TN, CPT>),
                        dim3(args.total_grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                        d_A_int8, d_scales_A_blockwise,
                        args.d_B[0], args.d_B[1],
                        args.d_C[0], args.d_C[1],
                        args.d_scales_B[0], args.d_scales_B[1],
                        args.N[0], args.N[1], args.grid_n_offset[1],
                        K, kb, alpha);
                }
            }
            else if (tn == TN_WIDE)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_init_output_batched_kernel_t<TN_WIDE, CPT>),
                    dim3(args.total_grid_n), dim3(TN_WIDE / CPT), 0, static_cast<hipStream_t>(stream),
                    args, beta);

                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_batched_vnni_blockwise_scaled_kernel_t<TN_WIDE, CPT>),
                    dim3(args.total_grid_n, kb), dim3(TN_WIDE / CPT), 0, static_cast<hipStream_t>(stream),
                    d_A_int8, d_scales_A_blockwise, args, K, kb, alpha);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_init_output_batched_kernel_t<TN, CPT>),
                    dim3(args.total_grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                    args, beta);

                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_batched_vnni_blockwise_scaled_kernel_t<TN, CPT>),
                    dim3(args.total_grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                    d_A_int8, d_scales_A_blockwise, args, K, kb, alpha);
            }
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_batched] Kernel launch failed: %s (K=%d num_proj=%d subgroup=%d)\n",
                    hipGetErrorString(err), K, num_projections, args.num_batches);
            return false;
        }

        return true;
    };

    int group_start = 0;
    while (group_start < num_projections)
    {
        const GemvBatchKernelFamily family = families[group_start];
        int group_end = group_start + 1;
        while (group_end < num_projections && families[group_end] == family)
        {
            ++group_end;
        }

        if (!launch_group(group_start, group_end, family))
        {
            return false;
        }

        group_start = group_end;
    }

    return true;
}



// =============================================================================
//
// Repacks weights from flat VNNI layout [K/4][N][4] to row-major [N×K]
// (equivalently [K×N] column-major for CK).
//
// VNNI layout:  vnni[(kg*N + n)*4 + i]  where kg = k/4, i = k%4
// Row-major:    rowmajor[n*K + k]       where k = kg*4 + i
//
// Mapping:  rowmajor[n*K + kg*4 + i] = vnni[(kg*N + n)*4 + i]
//
// Treat each 4-byte pack as one int32 element in matrix A[kg, n] and transpose
// to B[n, kg] using a tiled shared-memory transpose to keep both reads and
// writes coalesced.
//
// Cost: ~65-200μs for 18944×3584 (Gate/Up shape). Amortized over prefill.
//
__global__ void
repack_vnni_to_rowmajor_kernel(
    const int8_t* __restrict__ d_B_vnni,     // [K/4][N][4] input
    int8_t* __restrict__ d_B_rowmajor,       // [N×K] output
    int N, int K)
{
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;

    const int k_groups = K / 4; // number of int32 packs per output row

    const int x_in = blockIdx.x * TILE_DIM + threadIdx.x; // n index in source A[kg, n]
    const int y_in = blockIdx.y * TILE_DIM + threadIdx.y; // kg index in source A[kg, n]

    const int32_t* __restrict__ src32 = reinterpret_cast<const int32_t*>(d_B_vnni);
    int32_t* __restrict__ dst32 = reinterpret_cast<int32_t*>(d_B_rowmajor);

    __shared__ int32_t tile[TILE_DIM][TILE_DIM + 1];

    // Load tile from source (A = [k_groups, N], row-major in n)
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        const int y = y_in + j;
        if (x_in < N && y < k_groups)
        {
            tile[threadIdx.y + j][threadIdx.x] = src32[static_cast<int64_t>(y) * N + x_in];
        }
    }

    __syncthreads();

    // Store transposed tile to destination (B = [N, k_groups], row-major in kg)
    const int x_out = blockIdx.y * TILE_DIM + threadIdx.x; // kg
    const int y_out = blockIdx.x * TILE_DIM + threadIdx.y; // n

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        const int y = y_out + j;
        if (x_out < k_groups && y < N)
        {
            dst32[static_cast<int64_t>(y) * k_groups + x_out] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}


bool rocmGemv_repackVNNI_to_rowmajor(
    const int8_t* d_B_vnni,
    int8_t* d_B_rowmajor,
    int N, int K,
    int device_id, void* stream)
{
    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_repackVNNI_to_rowmajor] K must be multiple of 4 (K=%d)\n", K);
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_id);

    const int k_groups = K / 4;
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;

    const int grid_x = (N + TILE_DIM - 1) / TILE_DIM;
    const int grid_y = (k_groups + TILE_DIM - 1) / TILE_DIM;

    hipLaunchKernelGGL(
        repack_vnni_to_rowmajor_kernel,
        dim3(grid_x, grid_y), dim3(TILE_DIM, BLOCK_ROWS), 0, static_cast<hipStream_t>(stream),
        d_B_vnni, d_B_rowmajor, N, K);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_repackVNNI_to_rowmajor] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

/**
 * @brief Entrypoint for INT8 VNNI prefill GEMM.
 *
 * This function dispatches to tiled INT8-x-INT8 GEMM kernels for M>1 prefill.
 * Returning false is terminal for the selected Q8/INT8 prefill path; the
 * retired CK fallback path must not be revived implicitly.
 *
 * Junior-dev note:
 * - The decode path (M=1) already has specialized GEMV kernels.
 * - Prefill path (M>1) uses tiled INT8 VNNI GEMM kernels dispatched here.
 */
enum PrefillInt8TileVariant : int
{
    PREFILL_TILE_16x16 = 0,
    PREFILL_TILE_32x8 = 1,
    PREFILL_TILE_8x32 = 2,
    PREFILL_TILE_8x8 = 3,
};

static const char *prefillTileVariantName(int variant)
{
    switch (variant)
    {
    case PREFILL_TILE_16x16:
        return "16x16";
    case PREFILL_TILE_32x8:
        return "32x8";
    case PREFILL_TILE_8x32:
        return "8x32";
    case PREFILL_TILE_8x8:
        return "8x8";
    default:
        return "unknown";
    }
}

static int select_prefill_int8_tile_variant_auto(int M, int N, int K, bool grid_kpar)
{
    (void)K;
    if (M <= 12 && N <= 512)
    {
        return PREFILL_TILE_8x8;
    }

    if (N >= 4 * M && N >= 1024)
    {
        return PREFILL_TILE_32x8;
    }

    if (M >= 4 * N && M >= 128)
    {
        return PREFILL_TILE_8x32;
    }

    if (grid_kpar && N >= 2 * M && N >= 2048)
    {
        return PREFILL_TILE_32x8;
    }

    return PREFILL_TILE_16x16;
}

static int resolve_prefill_int8_tile_variant(int forced_variant, int M, int N, int K, bool grid_kpar)
{
    if (forced_variant >= PREFILL_TILE_16x16 && forced_variant <= PREFILL_TILE_8x8)
    {
        return forced_variant;
    }
    return select_prefill_int8_tile_variant_auto(M, N, K, grid_kpar);
}

bool rocmGemm_int8_int8_int32_vnni_prefill(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int M, int N, int K,
    int tile_variant,
    int cpt,
    int device_id, void* stream)
{
    if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_id);

    const int variant = resolve_prefill_int8_tile_variant(tile_variant, M, N, K, false);
    const int cpt_v = (cpt == 2 || cpt == 4) ? cpt : 1;

    if (variant == PREFILL_TILE_32x8)
    {
        constexpr int BX = 32;
        constexpr int BY = 8;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
    }
    else if (variant == PREFILL_TILE_8x32)
    {
        constexpr int BX = 8;
        constexpr int BY = 32;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
    }
    else if (variant == PREFILL_TILE_8x8)
    {
        constexpr int BX = 8;
        constexpr int BY = 8;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
    }
    else
    {
        constexpr int BX = 16;
        constexpr int BY = 16;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
    }

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill] Kernel launch failed: %s (M=%d N=%d K=%d)\n",
                hipGetErrorString(err), M, N, K);
        return false;
    }

    return true;
}

/**
 * @brief INT8 VNNI prefill grid-kpar split-K variant.
 *
 * This variant uses grid.z to split K reduction into slices with INT32
 * atomic accumulation. It is opt-in from higher-level dispatch.
 */
bool rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int M, int N, int K,
    int split_k_slices,
    int tile_variant,
    int cpt,
    int device_id, void* stream)
{
    if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    const int total_k_groups = K / 4;
    const int slices = max(1, min(split_k_slices, total_k_groups));
    const int k_groups_per_slice = (total_k_groups + slices - 1) / slices;

    llaminar2::HipDeviceGuard::setDevice(device_id);

    const size_t c_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(int32_t);
    hipError_t memset_err = hipMemsetAsync(d_C_int32, 0, c_bytes, static_cast<hipStream_t>(stream));
    if (memset_err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar] hipMemsetAsync failed: %s\n", hipGetErrorString(memset_err));
        return false;
    }

    const int variant = resolve_prefill_int8_tile_variant(tile_variant, M, N, K, true);
    const int cpt_v = (cpt == 2 || cpt == 4) ? cpt : 1;

    if (variant == PREFILL_TILE_32x8)
    {
        constexpr int BX = 32;
        constexpr int BY = 8;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
    }
    else if (variant == PREFILL_TILE_8x32)
    {
        constexpr int BX = 8;
        constexpr int BY = 32;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
    }
    else if (variant == PREFILL_TILE_8x8)
    {
        constexpr int BX = 8;
        constexpr int BY = 8;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
    }
    else
    {
        constexpr int BX = 16;
        constexpr int BY = 16;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
    }


    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar] Kernel launch failed: %s (M=%d N=%d K=%d slices=%d)\n",
                hipGetErrorString(err), M, N, K, slices);
        return false;
    }

    return true;
}

} // extern "C"
