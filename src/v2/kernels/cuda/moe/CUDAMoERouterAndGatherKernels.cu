/**
 * @file CUDAMoERouterAndGatherKernels.cu
 * @brief CUDA MoE router gate logits (FP32/FP16/BF16/Q8), softmax/top-k, token gather/scatter, shared expert gate, and SwiGLU kernels.
 */
#include "CUDAMoEDecodeRouterBody.cuh"
#include <mma.h>

namespace
{
    // Router GEMV: one block per (expert, token) computes logit = dot(hidden_token, gate_expert).
    //
    // Optimized memory-bound dot product. The previous implementation used scalar
    // loads (one element/thread/iter) followed by a full log2(blockDim)-step shared
    // memory tree reduction, leaving it ~15x off the memory-bound limit. This version:
    //   1. Loads gate/hidden as coalesced float4 vectors (4x fewer load instructions).
    //   2. Reduces within each warp via __shfl_down_sync (no __syncthreads, no shared
    //      traffic for the intra-warp reduction).
    //   3. Combines the (few) per-warp partials with a single short shared-memory step.
    // The block-per-expert grid is preserved so decode (num_experts blocks) keeps full
    // SM coverage.
    __global__ void route_logits_kernel(
        const float *__restrict__ hidden,
        const float *__restrict__ gate_weights,
        float *__restrict__ logits,
        int seq_len, int d_model, int num_experts)
    {
        const int expert = blockIdx.x;
        const int token = blockIdx.y;
        if (expert >= num_experts || token >= seq_len)
            return;

        const float *h = hidden + static_cast<size_t>(token) * d_model;
        const float *g = gate_weights + static_cast<size_t>(expert) * d_model;

        float sum = 0.0f;
        const bool can_vectorize =
            ((reinterpret_cast<std::uintptr_t>(h) |
              reinterpret_cast<std::uintptr_t>(g)) &
             0x0fu) == 0u;
        if (can_vectorize)
        {
            // Vectorized main loop: consecutive threads read consecutive float4 chunks
            // (fully coalesced). The row width is a multiple of 4 for supported
            // models, but graph/arena suballocations can still shift the base pointer.
            // Only use float4 loads when both row pointers are actually 16B-aligned.
            const int vec4 = d_model >> 2;
            for (int v = threadIdx.x; v < vec4; v += blockDim.x)
            {
                const float4 hv = reinterpret_cast<const float4 *>(h)[v];
                const float4 gv = reinterpret_cast<const float4 *>(g)[v];
                sum += hv.x * gv.x + hv.y * gv.y + hv.z * gv.z + hv.w * gv.w;
            }
            // Scalar tail for any d_model not divisible by 4.
            for (int j = (vec4 << 2) + threadIdx.x; j < d_model; j += blockDim.x)
                sum += h[j] * g[j];
        }
        else
        {
            for (int j = threadIdx.x; j < d_model; j += blockDim.x)
                sum += h[j] * g[j];
        }

        // Intra-warp reduction via shuffle (no shared memory, no barriers).
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);

        // Combine per-warp partials. blockDim is a multiple of 32 and <= 1024, so at
        // most 32 warps; warp 0 reduces the per-warp sums in a single shuffle pass.
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int num_warps = blockDim.x >> 5;
        __shared__ float warp_sums[32];
        if (lane == 0)
            warp_sums[warp] = sum;
        __syncthreads();
        if (warp == 0)
        {
            float v = (lane < num_warps) ? warp_sums[lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, offset);
            if (lane == 0)
                logits[static_cast<size_t>(token) * num_experts + expert] = v;
        }
    }

    template<bool Partitioned = false>
    __global__ void route_logits_bf16_kernel(
        const float *__restrict__ hidden,
        const __nv_bfloat16 *__restrict__ gate_weights,
        float *__restrict__ logits,
        int seq_len, int d_model, int num_experts,
        const int *live_rows = nullptr, llaminar2::DeviceRowPartition partition = {})
    {
        if constexpr (Partitioned)
        {
            const auto rows = partition.resolve(seq_len, live_rows);
            hidden += static_cast<size_t>(rows.first) * d_model;
            seq_len = rows.count;
        }
        const int expert = blockIdx.x;
        const int token = blockIdx.y;
        if (expert >= num_experts || token >= seq_len)
            return;

        const float *h = hidden + static_cast<size_t>(token) * d_model;
        const __nv_bfloat16 *g = gate_weights + static_cast<size_t>(expert) * d_model;

        float sum = 0.0f;
        for (int j = threadIdx.x; j < d_model; j += blockDim.x)
            sum += h[j] * __bfloat162float(g[j]);

        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);

        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int num_warps = blockDim.x >> 5;
        __shared__ float warp_sums[32];
        if (lane == 0)
            warp_sums[warp] = sum;
        __syncthreads();
        if (warp == 0)
        {
            float v = (lane < num_warps) ? warp_sums[lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, offset);
            if (lane == 0)
                logits[static_cast<size_t>(token) * num_experts + expert] = v;
        }
    }

    /**
     * @brief Decode-equivalent grouped FP32 router logits for verifier rows.
     *
     * One block owns one expert and one fixed-size verifier-row tile. The gate
     * vector is read once per K slice and reused across the row-local
     * accumulators. A second grid dimension covers any runtime row count, so
     * the fixed register tile never becomes a speculative-depth admission
     * limit. Each row still follows the same vectorized K traversal and block
     * reduction as route_logits_kernel(), preserving serial decode math while
     * removing redundant gate reads across nearby MTP verifier rows.
     */
    __global__ void route_logits_fp32_grouped_verifier_kernel(
        const float *__restrict__ hidden,
        const float *__restrict__ gate_weights,
        float *__restrict__ logits,
        int seq_len, int d_model, int num_experts)
    {
        constexpr int kRowsPerBlock = 4;
        const int expert = blockIdx.x;
        const int first_row = blockIdx.y * kRowsPerBlock;
        if (expert >= num_experts)
            return;
        const int tile_rows = min(kRowsPerBlock, seq_len - first_row);
        if (tile_rows <= 0)
            return;

        const float *g = gate_weights + static_cast<size_t>(expert) * d_model;
        float sums[kRowsPerBlock] = {0.0f, 0.0f, 0.0f, 0.0f};
        const bool can_vectorize =
            ((d_model & 3) == 0) &&
            ((reinterpret_cast<std::uintptr_t>(hidden) |
              reinterpret_cast<std::uintptr_t>(g)) &
             0x0fu) == 0u;
        if (can_vectorize)
        {
            const int vec4 = d_model >> 2;
            const float4 *g4 = reinterpret_cast<const float4 *>(g);
            for (int v = threadIdx.x; v < vec4; v += blockDim.x)
            {
                const float4 gv = g4[v];
                for (int tile_row = 0; tile_row < tile_rows; ++tile_row)
                {
                    const int row = first_row + tile_row;
                    const float *h = hidden + static_cast<size_t>(row) * d_model;
                    const float4 hv = reinterpret_cast<const float4 *>(h)[v];
                    sums[tile_row] +=
                        hv.x * gv.x + hv.y * gv.y + hv.z * gv.z + hv.w * gv.w;
                }
            }
            for (int j = (vec4 << 2) + threadIdx.x; j < d_model; j += blockDim.x)
            {
                const float gv = g[j];
                for (int tile_row = 0; tile_row < tile_rows; ++tile_row)
                {
                    const int row = first_row + tile_row;
                    sums[tile_row] += hidden[static_cast<size_t>(row) * d_model + j] * gv;
                }
            }
        }
        else
        {
            for (int j = threadIdx.x; j < d_model; j += blockDim.x)
            {
                const float gv = g[j];
                for (int tile_row = 0; tile_row < tile_rows; ++tile_row)
                {
                    const int row = first_row + tile_row;
                    sums[tile_row] += hidden[static_cast<size_t>(row) * d_model + j] * gv;
                }
            }
        }

        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int num_warps = blockDim.x >> 5;
        __shared__ float warp_sums[kRowsPerBlock][32];
        for (int tile_row = 0; tile_row < tile_rows; ++tile_row)
        {
            float v = sums[tile_row];
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, offset);
            if (lane == 0)
                warp_sums[tile_row][warp] = v;
        }
        __syncthreads();

        if (warp == 0)
        {
            for (int tile_row = 0; tile_row < tile_rows; ++tile_row)
            {
                float v = (lane < num_warps) ? warp_sums[tile_row][lane] : 0.0f;
                for (int offset = 16; offset > 0; offset >>= 1)
                    v += __shfl_down_sync(0xffffffffu, v, offset);
                if (lane == 0)
                {
                    const int row = first_row + tile_row;
                    logits[static_cast<size_t>(row) * num_experts + expert] = v;
                }
            }
        }
    }

    /**
     * @brief Decode-equivalent grouped BF16 router logits for verifier rows.
     *
     * Mirrors route_logits_bf16_kernel() for each row but shares each BF16 gate
     * load across a fixed row tile in the expert-owned block. Grid-Y tiles make
     * this implementation runtime-M without increasing per-thread registers.
     */
    __global__ void route_logits_bf16_grouped_verifier_kernel(
        const float *__restrict__ hidden,
        const __nv_bfloat16 *__restrict__ gate_weights,
        float *__restrict__ logits,
        int seq_len, int d_model, int num_experts)
    {
        constexpr int kRowsPerBlock = 4;
        const int expert = blockIdx.x;
        const int first_row = blockIdx.y * kRowsPerBlock;
        if (expert >= num_experts)
            return;
        const int tile_rows = min(kRowsPerBlock, seq_len - first_row);
        if (tile_rows <= 0)
            return;

        const __nv_bfloat16 *g = gate_weights + static_cast<size_t>(expert) * d_model;
        float sums[kRowsPerBlock] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int j = threadIdx.x; j < d_model; j += blockDim.x)
        {
            const float gv = __bfloat162float(g[j]);
            for (int tile_row = 0; tile_row < tile_rows; ++tile_row)
            {
                const int row = first_row + tile_row;
                sums[tile_row] += hidden[static_cast<size_t>(row) * d_model + j] * gv;
            }
        }

        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int num_warps = blockDim.x >> 5;
        __shared__ float warp_sums[kRowsPerBlock][32];
        for (int tile_row = 0; tile_row < tile_rows; ++tile_row)
        {
            float v = sums[tile_row];
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, offset);
            if (lane == 0)
                warp_sums[tile_row][warp] = v;
        }
        __syncthreads();

        if (warp == 0)
        {
            for (int tile_row = 0; tile_row < tile_rows; ++tile_row)
            {
                float v = (lane < num_warps) ? warp_sums[tile_row][lane] : 0.0f;
                for (int offset = 16; offset > 0; offset >>= 1)
                    v += __shfl_down_sync(0xffffffffu, v, offset);
                if (lane == 0)
                {
                    const int row = first_row + tile_row;
                    logits[static_cast<size_t>(row) * num_experts + expert] = v;
                }
            }
        }
    }

    /**
     * @brief Evaluate admitted half-router rows with the original FMA and warp tree.
     * @tparam Rows The exact maximum number of rows owned by this block.
     * @param hidden First original FP32 input row owned by the block.
     * @param weights Original FP16 weights for this expert.
     * @param logits First original FP32 output for this expert.
     * @param width Number of original hidden channels.
     * @param output_stride Number of experts separating consecutive outputs.
     * @param live_rows Real rows owned by this block, in the range [1, Rows].
     * @param warp_sums CTA scratch, readable after the original warp publication join.
     * @pre Every physical lane participates in the same original dot tree.
     */
    template<int Rows>
    __device__ __forceinline__ void route_logits_fp16_rows(
        const float* hidden, const __half* weights, float* logits,
        int width, int output_stride, int live_rows, float (&warp_sums)[Rows][32])
    {
        float sums[Rows]{};
        for (int j = threadIdx.x; j < width; j += blockDim.x)
        {
            const float weight = __half2float(weights[j]);
            for (int row = 0; row < live_rows; ++row)
                sums[row] += __half2float(__float2half_rn(hidden[static_cast<size_t>(row) * width + j])) * weight;
        }

        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int num_warps = blockDim.x >> 5;
        for (int row = 0; row < live_rows; ++row)
        {
            float value = sums[row];
            for (int offset = 16; offset > 0; offset >>= 1)
                value += __shfl_down_sync(0xffffffffu, value, offset);
            if (lane == 0)
                warp_sums[row][warp] = value;
        }
        __syncthreads();

        if (warp == 0)
        {
            for (int row = 0; row < live_rows; ++row)
            {
                float value = lane < num_warps ? warp_sums[row][lane] : 0.0f;
                for (int offset = 16; offset > 0; offset >>= 1)
                    value += __shfl_down_sync(0xffffffffu, value, offset);
                if (lane == 0)
                    logits[static_cast<size_t>(row) * output_stride] = value;
            }
        }
    }

    /**
     * @brief Actual-half router, one identical per-row dot/tree for all launch modes.
     * @tparam Partitioned Resolve device-owned row intervals before any operand load.
     * @param hidden Original FP32 hidden; register loads round RNE through half.
     * @param gate_weights Model-owned physical FP16 gate rows.
     * @param logits FP32 route logits, local to the resolved owner.
     * @param seq_len Physical row capacity. @param d_model Router reduction width.
     * @param num_experts Expert columns. @param live_rows Optional device live count.
     * @param partition Immutable participant row ownership.
     */
    template<bool Partitioned = false>
    __global__ void route_logits_fp16_grouped_kernel(
        const float *__restrict__ hidden,
        const __half *__restrict__ gate_weights,
        float *__restrict__ logits,
        int seq_len, int d_model, int num_experts,
        const int *live_rows = nullptr, llaminar2::DeviceRowPartition partition = {})
    {
        if constexpr (Partitioned)
        {
            const auto rows = partition.resolve(seq_len, live_rows);
            hidden += static_cast<size_t>(rows.first) * d_model;
            seq_len = rows.count;
        }
        else if (live_rows)
        {
            const int count = *live_rows;
            if (count < 0 || count > seq_len) __trap();
            seq_len = count;
        }
        constexpr int kRowsPerBlock = 4;
        const int expert = blockIdx.x;
        const int first_row = blockIdx.y * kRowsPerBlock;
        if (expert >= num_experts)
            return;
        const int tile_rows = min(kRowsPerBlock, seq_len - first_row);
        if (tile_rows <= 0)
            return;

        __shared__ float warp_sums[kRowsPerBlock][32];
        route_logits_fp16_rows(hidden + static_cast<size_t>(first_row) * d_model,
            gate_weights + static_cast<size_t>(expert) * d_model,
            logits + static_cast<size_t>(first_row) * num_experts + expert,
            d_model, num_experts, tile_rows, warp_sums);
    }

    /**
     * @brief Reuse half operands across four rows and four independent experts.
     *
     * Each output retains the small-row kernel's stride-256 FMA sequence and
     * descending warp/block reduction tree. Tiling shares register loads only;
     * it never combines experts, changes the hidden owner, or allocates scratch.
     * @tparam Partitioned Resolve the participant's live interval on device.
     * @param hidden Original FP32 input, rounded to half RNE in registers.
     * @param gate_weights Model-owned physical half expert rows.
     * @param logits Local FP32 output; inactive rows and columns are untouched.
     * @param seq_len Global admitted row capacity. @param d_model Reduction width.
     * @param num_experts Expert count. @param live_rows Optional device live count.
     * @param partition Immutable participant ownership, resolved before loads.
     */
    template<bool Partitioned>
    __global__ __launch_bounds__(256)
    void route_logits_fp16_expert_tile_kernel(
        const float *__restrict__ hidden, const __half *__restrict__ gate_weights,
        float *__restrict__ logits, int seq_len, int d_model, int num_experts,
        const int *live_rows, llaminar2::DeviceRowPartition partition)
    {
        if constexpr (Partitioned)
        {
            const auto rows = partition.resolve(seq_len, live_rows);
            hidden += static_cast<size_t>(rows.first) * d_model;
            seq_len = rows.count;
        }
        else if (live_rows)
        {
            const int count = *live_rows;
            if (count < 0 || count > seq_len) __trap();
            seq_len = count;
        }
        constexpr int Rows = 4, Experts = 4;
        const int first_row = blockIdx.y * Rows, first_expert = blockIdx.x * Experts;
        if (first_row >= seq_len || first_expert >= num_experts) return;
        float sums[Rows][Experts] = {};
        for (int j = threadIdx.x; j < d_model; j += 256)
        {
            float weights[Experts];
            #pragma unroll
            for (int e = 0; e < Experts; ++e)
                weights[e] = first_expert + e < num_experts
                    ? __half2float(gate_weights[static_cast<size_t>(first_expert + e) * d_model + j]) : 0.0f;
            #pragma unroll
            for (int r = 0; r < Rows; ++r)
                if (first_row + r < seq_len)
                {
                    const float h = __half2float(__float2half_rn(hidden[static_cast<size_t>(first_row + r) * d_model + j]));
                    #pragma unroll
                    for (int e = 0; e < Experts; ++e) sums[r][e] += h * weights[e];
                }
        }
        __shared__ float partial[Rows][Experts][8];
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        #pragma unroll
        for (int r = 0; r < Rows; ++r)
        {
            #pragma unroll
            for (int e = 0; e < Experts; ++e)
            {
                float value = sums[r][e];
                for (int offset = 16; offset > 0; offset >>= 1)
                    value += __shfl_down_sync(0xffffffffu, value, offset);
                if (lane == 0) partial[r][e][warp] = value;
            }
        }
        __syncthreads();
        if (warp == 0)
        {
            #pragma unroll
            for (int r = 0; r < Rows; ++r)
            {
                #pragma unroll
                for (int e = 0; e < Experts; ++e)
                {
                    float value = lane < 8 ? partial[r][e][lane] : 0.0f;
                    for (int offset = 16; offset > 0; offset >>= 1)
                        value += __shfl_down_sync(0xffffffffu, value, offset);
                    if (lane == 0 && first_row + r < seq_len && first_expert + e < num_experts)
                        logits[static_cast<size_t>(first_row + r) * num_experts + first_expert + e] = value;
                }
            }
        }
    }

    /**
     * @brief Tensor-core FP16xFP16->FP32 prefill router logits kernel (WMMA 16x16x16).
     *
     * Uses 64x64x32 SMEM tiles (bank-conflict-free stride 40 halves = 80B) with
     * 8 warps arranged as 4x2 (each warp owns 16 rows x 32 experts via two 16x16
     * FP32 WMMA accumulators). Invoked only when whole capacity >= 16, experts >= 16,
     * and width is a multiple of 16 so M <= 8 decode/verifier graphs preserve the
     * exact scalar-tile serial reduction order while prefill runs on Tensor Cores.
     */
    template<bool Partitioned>
    __global__ __launch_bounds__(256)
    void route_logits_fp16_wmma_kernel(
        const float *__restrict__ hidden, const __half *__restrict__ gate_weights,
        float *__restrict__ logits, int seq_len, int d_model, int num_experts,
        const int *live_rows, llaminar2::DeviceRowPartition partition)
    {
        if constexpr (Partitioned)
        {
            const auto rows = partition.resolve(seq_len, live_rows);
            hidden += static_cast<size_t>(rows.first) * d_model;
            seq_len = rows.count;
        }
        else if (live_rows)
        {
            const int count = *live_rows;
            if (count < 0 || count > seq_len) __trap();
            seq_len = count;
        }

        constexpr int kBM = 64, kBN = 64, kBK = 32, kLdm = 40;
        const int first_row = static_cast<int>(blockIdx.y) * kBM;
        const int first_expert = static_cast<int>(blockIdx.x) * kBN;
        if (first_row >= seq_len || first_expert >= num_experts) return;

        __shared__ union
        {
            struct
            {
                __half A[kBM][kLdm];
                __half B[kBN][kLdm];
            } ab;
            float C[8][256];
        } smem;

        const int tid = static_cast<int>(threadIdx.x);
        const int warp = tid >> 5;
        const int lane = tid & 31;
        const int warp_m = warp >> 1; // 0..3 -> 16 rows each
        const int warp_n = warp & 1;  // 0..1 -> 32 experts each

        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[2];
        nvcuda::wmma::fill_fragment(acc[0], 0.0f);
        nvcuda::wmma::fill_fragment(acc[1], 0.0f);

        const bool hidden_aligned =
            (reinterpret_cast<std::uintptr_t>(hidden) & 15u) == 0u;
        const bool gate_aligned =
            (reinterpret_cast<std::uintptr_t>(gate_weights) & 15u) == 0u;

        for (int k0 = 0; k0 < d_model; k0 += kBK)
        {
            // Load A[64][32]: 512 float4 quads across 256 threads (2 quads/thread).
            #pragma unroll
            for (int iter = 0; iter < 2; ++iter)
            {
                const int q = tid + iter * 256;
                const int r_loc = q >> 3;
                const int c = (q & 7) << 2;
                const int gr = first_row + r_loc;
                const int gk = k0 + c;
                __half h0 = __float2half_rn(0.0f);
                __half h1 = h0, h2 = h0, h3 = h0;
                if (gr < seq_len && gk < d_model)
                {
                    const float *src = hidden + static_cast<size_t>(gr) * d_model + gk;
                    if (hidden_aligned)
                    {
                        const float4 v = *reinterpret_cast<const float4 *>(src);
                        h0 = __float2half_rn(v.x);
                        h1 = __float2half_rn(v.y);
                        h2 = __float2half_rn(v.z);
                        h3 = __float2half_rn(v.w);
                    }
                    else
                    {
                        h0 = __float2half_rn(src[0]);
                        h1 = __float2half_rn(src[1]);
                        h2 = __float2half_rn(src[2]);
                        h3 = __float2half_rn(src[3]);
                    }
                }
                smem.ab.A[r_loc][c + 0] = h0;
                smem.ab.A[r_loc][c + 1] = h1;
                smem.ab.A[r_loc][c + 2] = h2;
                smem.ab.A[r_loc][c + 3] = h3;
            }

            // Load B[64][32]: 256 octets of 8 halves across 256 threads (1 uint4/thread).
            {
                const int e_loc = tid >> 2;
                const int c = (tid & 3) << 3;
                const int ge = first_expert + e_loc;
                const int gk = k0 + c;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (ge < num_experts && gk < d_model)
                {
                    const __half *src = gate_weights + static_cast<size_t>(ge) * d_model + gk;
                    if (gate_aligned)
                    {
                        v = *reinterpret_cast<const uint4 *>(src);
                    }
                    else
                    {
                        __half tmp[8];
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) tmp[i] = src[i];
                        v = *reinterpret_cast<const uint4 *>(tmp);
                    }
                }
                *reinterpret_cast<uint4 *>(&smem.ab.B[e_loc][c]) = v;
            }

            __syncthreads();

            const int k_steps = (k0 + 16 < d_model) ? 2 : 1;
            #pragma unroll
            for (int ks = 0; ks < 2; ++ks)
            {
                if (ks >= k_steps) break;
                const int k_off = ks * 16;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> b_frag0;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> b_frag1;
                nvcuda::wmma::load_matrix_sync(a_frag, &smem.ab.A[warp_m * 16][k_off], kLdm);
                nvcuda::wmma::load_matrix_sync(b_frag0, &smem.ab.B[warp_n * 32][k_off], kLdm);
                nvcuda::wmma::load_matrix_sync(b_frag1, &smem.ab.B[warp_n * 32 + 16][k_off], kLdm);
                nvcuda::wmma::mma_sync(acc[0], a_frag, b_frag0, acc[0]);
                nvcuda::wmma::mma_sync(acc[1], a_frag, b_frag1, acc[1]);
            }

            __syncthreads();
        }

        // Reuse smem.C[warp] per 16x16 tile so all 8 warps write out without extra SMEM.
        #pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            nvcuda::wmma::store_matrix_sync(smem.C[warp], acc[j], 16, nvcuda::wmma::mem_row_major);
            __syncwarp();
            const int e_base = first_expert + warp_n * 32 + j * 16;
            const int r_base = first_row + warp_m * 16;
            #pragma unroll
            for (int i = lane; i < 256; i += 32)
            {
                const int r = r_base + (i >> 4);
                const int e = e_base + (i & 15);
                if (r < seq_len && e < num_experts)
                    logits[static_cast<size_t>(r) * num_experts + e] = smem.C[warp][i];
            }
            __syncwarp();
        }
    }

    /**
     * @brief Select measured small-row, shared-expert, or Tensor-Core geometry before capture.
     * @tparam Partitioned Kernel resolves device-owned participant rows.
     * @param hidden Original FP32 rows. @param gate Model-owned physical half gate.
     * @param logits Participant-local FP32 destination.
     * @param capacity Whole-input capacity. @param local_capacity Admitted owner capacity.
     * @param width Reduction width. @param experts Expert count.
     * @param live_rows Optional device-owned live count. @param partition Row owner.
     * @param stream Exact non-null stream authenticated by the public launch bridge.
     *
     * Capacity fixes graph geometry; live counts remain device-owned. M1..M8
     * decode and MTP verifiers keep the exact scalar-tile reduction tree, while
     * prefill buckets (capacity >= 16, experts >= 16, width % 16 == 0) execute
     * on Tensor Cores across both complete and row-partitioned ownership.
     */
    template<bool Partitioned>
    void launch_fp16_router_rows(const float *hidden, const __half *gate, float *logits,
        int capacity, int local_capacity, int width, int experts, const int *live_rows,
        llaminar2::DeviceRowPartition partition, cudaStream_t stream)
    {
        if (capacity >= 16 && experts >= 16 && (width & 15) == 0)
            route_logits_fp16_wmma_kernel<Partitioned><<<
                dim3((experts + 63) / 64, (local_capacity + 63) / 64), 256, 0, stream>>>(
                hidden, gate, logits, capacity, width, experts, live_rows, partition);
        else if (local_capacity >= 8 && experts >= 4)
            route_logits_fp16_expert_tile_kernel<Partitioned><<<
                dim3((experts + 3) / 4, (local_capacity + 3) / 4), 256, 0, stream>>>(
                hidden, gate, logits, capacity, width, experts, live_rows, partition);
        else
            route_logits_fp16_grouped_kernel<Partitioned><<<
                dim3(experts, (local_capacity + 3) / 4), 256, 0, stream>>>(
                hidden, gate, logits, capacity, width, experts, live_rows, partition);
    }

    // Tiled SGEMM for the PREFILL router logits: logits[S, E] = hidden[S, D] · gateᵀ[D, E].
    //
    // The block-per-(expert,token) route_logits_kernel above has zero data reuse: it
    // re-reads hidden[token] once per expert (E× redundant) and gate[expert] once per
    // token (S× redundant), saturating L2 (~85% L2 throughput, ~1% DRAM, ~0.7 TFLOP/s).
    // This classic shared-memory tiled GEMM stages BM×BK hidden and BN×BK gate tiles
    // into smem and reuses each loaded element across the BN (resp. BM) tile dimension,
    // cutting L2 traffic by ~min(BM,BN)× and converting the kernel from L2-bound to
    // compute-bound.
    //
    // Tile geometry: BM=64 tokens × BN=64 experts per block, BK=16 along d_model.
    // Block = 16×16 = 256 threads, each thread computes a TM×TN = 4×4 output micro-tile.
    // Decode (seq_len == 1) still uses the warp-reduction kernel above for SM coverage.
    template <int BM, int BN, int BK, int TM, int TN, bool Partitioned = false>
    __global__ void route_logits_tiled_kernel(
        const float *__restrict__ hidden,       // A: [seq_len, d_model] row-major
        const float *__restrict__ gate_weights, // B: [num_experts, d_model] row-major
        float *__restrict__ logits,             // C: [seq_len, num_experts] row-major
        int seq_len, int d_model, int num_experts,
        const int *live_rows = nullptr, llaminar2::DeviceRowPartition partition = {})
    {
        if constexpr (Partitioned)
        {
            // Retain the exact increasing-K FMA sequence. Only complete input
            // rows change owner; no expert/K partials enter a new reduction.
            const auto rows = partition.resolve(seq_len, live_rows);
            hidden += static_cast<size_t>(rows.first) * d_model;
            seq_len = rows.count;
            if (static_cast<int>(blockIdx.y) * BM >= seq_len) return;
        }
        // Shared tiles, stored K-major so the compute loop reads a full column of
        // BM (resp. BN) values contiguously for each k step.
        __shared__ float As[BK * BM]; // As[k * BM + m] = hidden[blockM + m, kk + k]
        __shared__ float Bs[BK * BN]; // Bs[k * BN + n] = gate  [blockN + n, kk + k]

        // Origin of this block's output tile in (token, expert) space.
        const int blockM = blockIdx.y * BM;
        const int blockN = blockIdx.x * BN;

        // 16×16 thread grid; each thread owns a TM×TN micro-tile of the output.
        const int threadRow = threadIdx.x / (BN / TN); // 0..15
        const int threadCol = threadIdx.x % (BN / TN); // 0..15

        // Per-thread accumulators for the TM×TN micro-tile (lives in registers).
        float acc[TM][TN];
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j)
                acc[i][j] = 0.0f;

        /*
         * The production Qwen matrices and arena rows are at least 16-byte
         * aligned and K is divisible by four.  Load both operands as float4
         * vectors in that common case, then transpose the four scalars into
         * the established K-major shared-memory layout.  The arithmetic loop
         * below is deliberately untouched: every output still consumes
         * k=0,1,... in exactly the same order, so vectorizing transport cannot
         * change a router-logit byte.
         *
         * Ragged or externally supplied geometries retain scalar cooperative
         * loads.  This is geometry totality, not a production fallback: both
         * branches feed the same physical kernel and identical arithmetic.
         */
        const bool vector_loads =
            (d_model % BK) == 0 &&
            (reinterpret_cast<uintptr_t>(hidden) & (alignof(float4) - 1)) == 0 &&
            (reinterpret_cast<uintptr_t>(gate_weights) &
             (alignof(float4) - 1)) == 0;

        // March across the K (d_model) dimension one BK-wide strip at a time.
        for (int kk = 0; kk < d_model; kk += BK)
        {
            if (vector_loads)
            {
                static_assert(BK % 4 == 0);
                constexpr int kVectorsPerRow = BK / 4;
                constexpr int kHiddenVectors = BM * kVectorsPerRow;
                constexpr int kGateVectors = BN * kVectorsPerRow;
                constexpr int kTotalVectors =
                    kHiddenVectors + kGateVectors;

                for (int vector_index = threadIdx.x;
                     vector_index < kTotalVectors;
                     vector_index += blockDim.x)
                {
                    const bool loads_hidden = vector_index < kHiddenVectors;
                    const int operand_index = loads_hidden
                                                  ? vector_index
                                                  : vector_index - kHiddenVectors;
                    const int row = operand_index / kVectorsPerRow;
                    const int first_k =
                        (operand_index % kVectorsPerRow) * 4;
                    const int global_row =
                        (loads_hidden ? blockM : blockN) + row;
                    const int row_limit =
                        loads_hidden ? seq_len : num_experts;

                    float4 values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    if (global_row < row_limit)
                    {
                        const float *operand =
                            loads_hidden ? hidden : gate_weights;
                        const float *source =
                            operand +
                            static_cast<size_t>(global_row) * d_model +
                            kk + first_k;
                        values = *reinterpret_cast<const float4 *>(source);
                    }

                    float *tile = loads_hidden ? As : Bs;
                    const int tile_rows = loads_hidden ? BM : BN;
                    tile[(first_k + 0) * tile_rows + row] = values.x;
                    tile[(first_k + 1) * tile_rows + row] = values.y;
                    tile[(first_k + 2) * tile_rows + row] = values.z;
                    tile[(first_k + 3) * tile_rows + row] = values.w;
                }
            }
            else
            {
                // Scalar transport covers arbitrary positive K and alignment.
                for (int idx = threadIdx.x;
                     idx < BM * BK;
                     idx += blockDim.x)
                {
                    const int m = idx / BK;
                    const int k = idx % BK;
                    const int gm = blockM + m;
                    const int gk = kk + k;
                    As[k * BM + m] = (gm < seq_len && gk < d_model)
                                         ? hidden[static_cast<size_t>(gm) *
                                                      d_model +
                                                  gk]
                                         : 0.0f;
                }
                for (int idx = threadIdx.x;
                     idx < BN * BK;
                     idx += blockDim.x)
                {
                    const int n = idx / BK;
                    const int k = idx % BK;
                    const int gn = blockN + n;
                    const int gk = kk + k;
                    Bs[k * BN + n] = (gn < num_experts && gk < d_model)
                                         ? gate_weights[static_cast<size_t>(gn) *
                                                            d_model +
                                                        gk]
                                         : 0.0f;
                }
            }
            __syncthreads();

            // Multiply the staged strip: for each k, broadcast TM hidden values and
            // TN gate values from smem into registers and accumulate the outer product.
#pragma unroll
            for (int k = 0; k < BK; ++k)
            {
                float regM[TM];
                float regN[TN];
                /*
                 * Each thread owns contiguous rows/columns within the K-major
                 * shared tiles.  Consume those fragments with one naturally
                 * aligned vector instruction when their compile-time width
                 * permits it.  Components are still fed to the same scalar
                 * FMA nest below, so this only reduces shared-memory LSU
                 * instructions; it cannot reorder a dot product.
                 */
                if constexpr (TM == 4)
                {
                    const float4 values =
                        *reinterpret_cast<const float4 *>(
                            &As[k * BM + threadRow * TM]);
                    regM[0] = values.x;
                    regM[1] = values.y;
                    regM[2] = values.z;
                    regM[3] = values.w;
                }
                else if constexpr (TM == 2)
                {
                    const float2 values =
                        *reinterpret_cast<const float2 *>(
                            &As[k * BM + threadRow * TM]);
                    regM[0] = values.x;
                    regM[1] = values.y;
                }
                else
                {
#pragma unroll
                    for (int i = 0; i < TM; ++i)
                        regM[i] = As[k * BM + threadRow * TM + i];
                }

                if constexpr (TN == 4)
                {
                    const float4 values =
                        *reinterpret_cast<const float4 *>(
                            &Bs[k * BN + threadCol * TN]);
                    regN[0] = values.x;
                    regN[1] = values.y;
                    regN[2] = values.z;
                    regN[3] = values.w;
                }
                else if constexpr (TN == 2)
                {
                    const float2 values =
                        *reinterpret_cast<const float2 *>(
                            &Bs[k * BN + threadCol * TN]);
                    regN[0] = values.x;
                    regN[1] = values.y;
                }
                else
                {
#pragma unroll
                    for (int j = 0; j < TN; ++j)
                        regN[j] = Bs[k * BN + threadCol * TN + j];
                }
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j)
                        acc[i][j] += regM[i] * regN[j];
            }
            __syncthreads();
        }

        // Write the micro-tile back to global memory, guarding the ragged token edge
        // (num_experts is a multiple of BN for all supported models, but guard anyway).
#pragma unroll
        for (int i = 0; i < TM; ++i)
        {
            const int m = blockM + threadRow * TM + i;
            if (m >= seq_len)
                continue;
#pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                const int n = blockN + threadCol * TN + j;
                if (n < num_experts)
                    logits[static_cast<size_t>(m) * num_experts + n] = acc[i][j];
            }
        }
    }

    /**
     * @brief Submit one compile-time router geometry to an explicit stream.
     *
     * Each specialization changes only output ownership.  The K loop in
     * `route_logits_tiled_kernel` remains increasing and identical, so a
     * candidate may be promoted only after the harness proves full-output byte
     * equality with the installed production geometry.
     */
    template <int BM, int BN, int BK, int TM, int TN, bool Partitioned = false>
    bool launch_route_logits_tiled_geometry(
        const float *hidden,
        const float *gate_weights,
        float *logits,
        int seq_len,
        int d_model,
        int num_experts,
        cudaStream_t stream,
        const char *launch_name,
        const int *live_rows = nullptr, llaminar2::DeviceRowPartition partition = {})
    {
        static_assert(BM > 0 && BN > 0 && BK > 0 && TM > 0 && TN > 0);
        static_assert(BM % TM == 0 && BN % TN == 0);
        constexpr int kThreads = (BM / TM) * (BN / TN);
        static_assert(kThreads == 256);

        const int grid_rows = Partitioned ? partition.capacityFor(seq_len) : seq_len;
        const dim3 grid(
            static_cast<unsigned int>((num_experts + BN - 1) / BN),
            static_cast<unsigned int>((grid_rows + BM - 1) / BM));
        route_logits_tiled_kernel<BM, BN, BK, TM, TN, Partitioned>
            <<<grid, kThreads, 0, stream>>>(
                hidden,
                gate_weights,
                logits,
                seq_len,
                d_model,
                num_experts, live_rows, partition);
        return finishLaunch(launch_name);
    }

    /**
     * @brief Shared geometry lowering for replicated and row-owned router work.
     * @tparam Partitioned Compile away ownership resolution in the established path.
     *
     * Geometry selection remains CUDAMoERouterPrefillPolicy's authority. This
     * switch only lowers that enum into the identical physical kernel family.
     */
    template<bool Partitioned = false>
    bool dispatch_route_logits_geometry(const float *hidden, const float *gate_weights,
        float *logits, int seq_len, int d_model, int num_experts, cudaStream_t stream,
        llaminar2::CUDAMoERouterPrefillGeometry geometry,
        const int *live_rows = nullptr, llaminar2::DeviceRowPartition partition = {})
    {
        using Geometry = llaminar2::CUDAMoERouterPrefillGeometry;
        switch (geometry)
        {
        case Geometry::Tile64x64:
            return launch_route_logits_tiled_geometry<64, 64, 16, 4, 4, Partitioned>(
                hidden, gate_weights, logits, seq_len, d_model, num_experts,
                stream, "cudaMoE_route_logits_Tile64x64", live_rows, partition);
        case Geometry::Tile64x32:
            return launch_route_logits_tiled_geometry<64, 32, 16, 4, 2, Partitioned>(
                hidden, gate_weights, logits, seq_len, d_model, num_experts,
                stream, "cudaMoE_route_logits_Tile64x32", live_rows, partition);
        case Geometry::Tile32x64:
            return launch_route_logits_tiled_geometry<32, 64, 16, 2, 4, Partitioned>(
                hidden, gate_weights, logits, seq_len, d_model, num_experts,
                stream, "cudaMoE_route_logits_Tile32x64", live_rows, partition);
        case Geometry::Tile32x32:
            return launch_route_logits_tiled_geometry<32, 32, 32, 2, 2, Partitioned>(
                hidden, gate_weights, logits, seq_len, d_model, num_experts,
                stream, "cudaMoE_route_logits_Tile32x32", live_rows, partition);
        case Geometry::Tile32x32K16:
            return launch_route_logits_tiled_geometry<32, 32, 16, 2, 2, Partitioned>(
                hidden, gate_weights, logits, seq_len, d_model, num_experts,
                stream, "cudaMoE_route_logits_Tile32x32K16", live_rows, partition);
        case Geometry::Tile32x24:
            return launch_route_logits_tiled_geometry<32, 24, 16, 1, 3, Partitioned>(
                hidden, gate_weights, logits, seq_len, d_model, num_experts,
                stream, "cudaMoE_route_logits_Tile32x24", live_rows, partition);
        case Geometry::Tile24x32:
            return launch_route_logits_tiled_geometry<24, 32, 16, 3, 1, Partitioned>(
                hidden, gate_weights, logits, seq_len, d_model, num_experts,
                stream, "cudaMoE_route_logits_Tile24x32", live_rows, partition);
        case Geometry::Tile64x16:
            return launch_route_logits_tiled_geometry<64, 16, 16, 4, 1, Partitioned>(
                hidden, gate_weights, logits, seq_len, d_model, num_experts,
                stream, "cudaMoE_route_logits_Tile64x16", live_rows, partition);
        case Geometry::Tile32x16:
            return launch_route_logits_tiled_geometry<32, 16, 16, 2, 1, Partitioned>(
                hidden, gate_weights, logits, seq_len, d_model, num_experts,
                stream, "cudaMoE_route_logits_Tile32x16", live_rows, partition);
        case Geometry::Tile16x64:
            return launch_route_logits_tiled_geometry<16, 64, 16, 1, 4, Partitioned>(
                hidden, gate_weights, logits, seq_len, d_model, num_experts,
                stream, "cudaMoE_route_logits_Tile16x64", live_rows, partition);
        case Geometry::Tile16x32:
            return launch_route_logits_tiled_geometry<16, 32, 16, 1, 2, Partitioned>(
                hidden, gate_weights, logits, seq_len, d_model, num_experts,
                stream, "cudaMoE_route_logits_Tile16x32", live_rows, partition);
        case Geometry::Tile16x16:
            return launch_route_logits_tiled_geometry<16, 16, 16, 1, 1, Partitioned>(
                hidden, gate_weights, logits, seq_len, d_model, num_experts,
                stream, "cudaMoE_route_logits_Tile16x16", live_rows, partition);
        case Geometry::Tile8x64:
            return launch_route_logits_tiled_geometry<8, 64, 16, 1, 2, Partitioned>(
                hidden, gate_weights, logits, seq_len, d_model, num_experts,
                stream, "cudaMoE_route_logits_Tile8x64", live_rows, partition);
        case Geometry::Tile8x32:
            return launch_route_logits_tiled_geometry<8, 32, 16, 1, 1, Partitioned>(
                hidden, gate_weights, logits, seq_len, d_model, num_experts,
                stream, "cudaMoE_route_logits_Tile8x32", live_rows, partition);
        }
        return false;
    }

    /** Compiler-owned resource facts for one router geometry. */
    struct RouterPrefillKernelResources
    {
        int registers_per_thread = 0;
        size_t local_memory_bytes_per_thread = 0;
        size_t static_shared_memory_bytes = 0;
        int max_threads_per_block = 0;
        int max_active_blocks_per_sm = 0;
    };

    /**
     * @brief Inspect one exact router specialization without launching it.
     *
     * CUDA's function attributes expose compiler-local storage before timing,
     * while the occupancy API evaluates the same 256-thread geometry submitted
     * by production.  Candidates with local storage never enter the tournament.
     */
    template <int BM, int BN, int BK, int TM, int TN>
    bool query_route_logits_tiled_geometry_resources(
        RouterPrefillKernelResources *resources)
    {
        if (!resources)
            return false;

        constexpr int kThreads = (BM / TM) * (BN / TN);
        static_assert(kThreads == 256);
        cudaFuncAttributes attributes{};
        if (cudaFuncGetAttributes(
                &attributes,
                route_logits_tiled_kernel<BM, BN, BK, TM, TN>) != cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        int active_blocks = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,
                route_logits_tiled_kernel<BM, BN, BK, TM, TN>,
                kThreads,
                0) != cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        *resources = RouterPrefillKernelResources{
            .registers_per_thread = attributes.numRegs,
            .local_memory_bytes_per_thread = attributes.localSizeBytes,
            .static_shared_memory_bytes = attributes.sharedSizeBytes,
            .max_threads_per_block = attributes.maxThreadsPerBlock,
            .max_active_blocks_per_sm = active_blocks,
        };
        return active_blocks > 0;
    }

    /**
     * @brief Publish top-k routes only for the current device-owned live prefix.
     *
     * Captured verifier graphs retain physical rows even for an empty replay.
     * Inactive rows clear both public outputs and the optional deferred ledger;
     * they must never invent a live first row from a zero logical count.
     */
    template<bool Partitioned = false>
    __global__ void softmax_topk_kernel(
        float *__restrict__ logits,
        float *__restrict__ expert_indices,
        float *__restrict__ expert_weights,
        int seq_len, int num_experts, int top_k,
        bool normalize_weights,
        const int *__restrict__ effective_seq_len_ptr,
        DeviceMoELayerRuntimeView *__restrict__ deferred_selected_route_ledger,
        llaminar2::DeviceRowPartition partition = {},
        llaminar2::MoERouterSelectedRoute *selected_packet = nullptr,
        std::uint64_t *selected_bytes = nullptr)
    {
        if constexpr (Partitioned)
        {
            seq_len = partition.resolve(seq_len, effective_seq_len_ptr).count;
            effective_seq_len_ptr = nullptr;
            // The producer publishes even an empty message. This write is
            // ordered before the separate collective by the graph edge; it is
            // not a readiness flag polled while other top-k CTAs are running.
            if (blockIdx.x == 0 && threadIdx.x == 0)
                *selected_bytes = std::uint64_t(seq_len) * top_k * sizeof(*selected_packet);
        }
        const int token = blockIdx.x;
        if (token >= seq_len)
            return;

        const int total_slots = seq_len * top_k;
        const bool ledger_is_complete =
            deferred_selected_route_ledger == nullptr ||
            (deferred_selected_route_ledger
                 ->deferred_verifier_route_expert_ids != nullptr &&
             deferred_selected_route_ledger
                 ->deferred_verifier_route_participant_ids != nullptr &&
             deferred_selected_route_ledger
                     ->deferred_verifier_route_capacity >=
                 static_cast<uint32_t>(total_slots));
        if (!ledger_is_complete)
        {
            if (token == 0 && threadIdx.x == 0)
            {
                FAIL_FAST_INVALID_GROUPED_VERIFIER_COMMIT(
                    "overlay verifier routing requires a complete per-layer "
                    "deferred selected-route ledger");
            }
            return;
        }

        int effective_seq_len = seq_len;
        if (effective_seq_len_ptr)
        {
            const int raw_effective = *effective_seq_len_ptr;
            // Zero is an ordinary empty retained replay, not a minimum batch
            // of one. This also protects every gate format using this kernel.
            effective_seq_len = seq_len == 1
                ? (runtime_decode_row_active(effective_seq_len_ptr) ? 1 : 0)
                : (raw_effective < 0 ? 0 : (raw_effective > seq_len ? seq_len : raw_effective));
        }
        if (token >= effective_seq_len)
        {
            if (threadIdx.x == 0)
            {
                for (int k = 0; k < top_k; ++k)
                {
                    const size_t out = static_cast<size_t>(token) * top_k + k;
                    expert_indices[out] = -1.0f;
                    expert_weights[out] = 0.0f;
                    if (deferred_selected_route_ledger)
                    {
                        deferred_selected_route_ledger
                            ->deferred_verifier_route_expert_ids[out] = -1;
                        deferred_selected_route_ledger
                            ->deferred_verifier_route_participant_ids[out] = -1;
                    }
                }
            }
            for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
                logits[static_cast<size_t>(token) * num_experts + expert] = 0.0f;
            return;
        }

        __shared__ float values[kMaxExperts];
        __shared__ float reductions[kThreads];
        __shared__ int red_idx[kThreads];
        __shared__ int selected[kMaxTopK];
        __shared__ float selected_weights[kMaxTopK];

        const size_t row_offset = static_cast<size_t>(token) * num_experts;
        for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
            values[expert] = logits[row_offset + expert];
        __syncthreads();

        /*
         * Softmax is strictly monotonic, so select experts from the raw router
         * logits. This prevents an all-expert probability reduction from
         * changing a near-tie decision when M changes. The pair reduction has a
         * fixed tree and a lower-expert-id tie break, matching decode routing.
         */
        moe_select_topk_probabilities_block(
            values,
            num_experts,
            top_k,
            selected,
            selected_weights,
            reductions,
            red_idx);

        /*
         * Compute the full softmax with the same fixed 256-thread reduction
         * tree used by `softmax_topk_decode_runtime_kernel`.  The previous
         * correctness-first implementation accumulated all 256 exponentials
         * on thread zero.  Although that made the addition order obvious, it
         * left the remaining 255 threads idle during the most expensive part
         * of every router row.
         *
         * The reduction below is independent of `seq_len`: one block always
         * owns one row and `num_experts` alone determines which values each
         * thread contributes.  Exact-M and padded graph launches therefore
         * execute the same arithmetic tree for every active row.  Keeping this
         * tree identical to decode is also important for grouped MTP verifier
         * rows, whose published weights must be byte-identical to serial
         * single-token routing.
         */
        const float max_value = selected_weights[0];
        float local_sum = 0.0f;
        for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
        {
            const float exponential =
                expf(logits[row_offset + expert] - max_value);
            values[expert] = exponential;
            local_sum += exponential;
        }
        reductions[threadIdx.x] = local_sum;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                reductions[threadIdx.x] += reductions[threadIdx.x + stride];
            __syncthreads();
        }

        const float denominator = reductions[0];
        for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
        {
            const float probability =
                denominator > 0.0f ? values[expert] / denominator : 0.0f;
            values[expert] = probability;
            logits[row_offset + expert] = probability;
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            float selected_sum = 0.0f;
            for (int k = 0; k < top_k; ++k)
            {
                selected_weights[k] = values[selected[k]];
                selected_sum += selected_weights[k];
            }

            for (int k = 0; k < top_k; ++k)
            {
                const size_t out = static_cast<size_t>(token) * top_k + k;
                const float probability =
                    normalize_weights && selected_sum > 0.0f
                        ? selected_weights[k] / selected_sum
                        : selected_weights[k];
                if constexpr (Partitioned)
                    selected_packet[out] = {static_cast<float>(selected[k]), probability};
                else
                {
                    expert_indices[out] = static_cast<float>(selected[k]);
                    expert_weights[out] = probability;
                }
                if (deferred_selected_route_ledger)
                {
                    /*
                     * The continuation GPU knows selection but not which
                     * heterogeneous participant will execute the ticket.
                     * Retain only global demand; participant-local evidence is
                     * deliberately absent rather than guessed.
                     */
                    deferred_selected_route_ledger
                        ->deferred_verifier_route_expert_ids[out] = selected[k];
                    deferred_selected_route_ledger
                        ->deferred_verifier_route_participant_ids[out] = -1;
                }
            }
        }
    }

    /** @copydoc softmax_topk_decode_runtime_body */
    template <bool UpdateRuntimeHistogram, bool ApplyReadyRebalance>
    __global__ void softmax_topk_decode_runtime_kernel(
        float* __restrict__ logits,
        DeviceMoELayerRuntimeView* __restrict__ runtime,
        DeviceMoELayerRuntimeView* __restrict__ runtime_layers,
        float* legacy_indices, float* legacy_weights,
        int num_experts, int top_k, bool normalize_weights, bool write_legacy_outputs,
        llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy,
        const int32_t* __restrict__ absolute_position_ids,
        const DeviceMoERebalancePlanEntryView* rebalance_plan_entries,
        uint32_t rebalance_plan_capacity,
        DeviceMoERebalanceCommandBufferHeaderView* rebalance_command_header,
        const DeviceMoEExpertDirectoryEntryView* rebalance_local_transfer_slots,
        uint32_t rebalance_local_transfer_slot_count,
        DeviceMoERebalanceConfigView rebalance_config,
        DeviceMoERebalanceApplyStatusView* rebalance_apply_status,
        DeviceMoERebalanceGraphControllerStateView* rebalance_controller_state,
        int rebalance_target_layer, uint32_t rebalance_command_buffer_count,
        const int32_t* __restrict__ active_row_count_device)
    {
        softmax_topk_decode_runtime_body<UpdateRuntimeHistogram, ApplyReadyRebalance>(
            logits, runtime, runtime_layers, legacy_indices, legacy_weights, num_experts,
            top_k, normalize_weights, write_legacy_outputs, row_execution_policy,
            absolute_position_ids, rebalance_plan_entries, rebalance_plan_capacity,
            rebalance_command_header, rebalance_local_transfer_slots, rebalance_local_transfer_slot_count,
            rebalance_config, rebalance_apply_status, rebalance_controller_state, rebalance_target_layer,
            rebalance_command_buffer_count, active_row_count_device);
    }

    /**
     * @brief Minimal decode router for independently published owner or fragment slots.
     *
     * Replica-aware scheduling carries participant load arrays and maintenance
     * arguments that a capacity-proven single-resident graph cannot consume.
     * This distinct kernel ABI removes that state while retaining exactly the
     * same 256-thread probability reduction and top-k arithmetic as the full
     * production router.  The active runtime bank remains device-owned and is
     * authenticated on every replay.
     *
     * Selected slots retain the serial normalization order, then each lane
     * authenticates its current owner or complete intermediate fragment. Integer
     * histogram increments touch unique expert IDs and preserve exact demand.
     *
     * @tparam RowPolicy Original whole-owner or intermediate-fragment row contract.
     * @tparam UpdateRuntimeHistogram Whether this graph publishes Dynamic
     *         residency evidence.

     * @tparam PublishDescriptors Publish original arena views in the authenticated route lane.
     * @tparam ExpertCapacity Cold-admitted original expert extent for the exact selection program.
     * @param logits Original FP32 probabilities in place.
     * @param runtime Original device-owned runtime and acquired bank.
     * @param legacy_indices Complete selected IDs when requested.
     * @param legacy_weights Complete selected weight bits when requested.
     * @param num_experts Exact admitted source expert extent.
     * @param top_k Exact selected packet width.
     * @param normalize_weights Retain the original serial FP32 normalization.
     * @param write_legacy_outputs Whether complete selected outputs are required.
     * @param active_row_count_device Optional live count for this physical row.
     * @param publication Original arena outputs; unused in the control specialization.
 */
template <bool UpdateRuntimeHistogram,
          llaminar2::RoutedExpertRowExecutionPolicy RowPolicy = llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned>
__global__ void softmax_topk_decode_independent_runtime_kernel(
        float *__restrict__ logits,
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts,
        int top_k,
        bool normalize_weights,
        bool write_legacy_outputs,
        const int32_t *__restrict__ active_row_count_device)
    {
    softmax_topk_decode_independent_body<UpdateRuntimeHistogram, RowPolicy, false>(
        logits, runtime, legacy_indices, legacy_weights, num_experts, top_k,
        normalize_weights, write_legacy_outputs, active_row_count_device, {});
}

/** @brief Fused entry publishes arena descriptors from the same authenticated route lanes. */
template <bool UpdateRuntimeHistogram,
          llaminar2::RoutedExpertRowExecutionPolicy RowPolicy = llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned>
__global__ void softmax_topk_decode_descriptors_kernel(
        float *__restrict__ logits,
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts,
        int top_k,
        bool normalize_weights,
        bool write_legacy_outputs,
        const int32_t *__restrict__ active_row_count_device,
        llaminar2::MoEDecodeDescriptorPublication publication)
    {
    softmax_topk_decode_independent_body<UpdateRuntimeHistogram, RowPolicy, true>(
        logits, runtime, legacy_indices, legacy_weights, num_experts, top_k,
        normalize_weights, write_legacy_outputs, active_row_count_device, publication);
}

    __global__ void decode_route_select_runtime_kernel(
        const int *__restrict__ expert_indices,
        const float *__restrict__ expert_weights,
        DeviceMoELayerRuntimeView *__restrict__ runtime,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts,
        int top_k,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
        llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy,
        const int32_t *__restrict__ absolute_position_ids)
    {
        if (threadIdx.x != 0)
            return;

        const uint32_t execution_bank =
            runtime_require_decode_execution_bank(runtime);
        if (execution_bank > 1u)
            return;
        const bool shape_ok = runtime_shape_ok(
            runtime, execution_bank, num_experts, top_k);
        if (!runtime_route_assignment_ledger_ok(runtime, top_k))
        {
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                "decode route assignment ledger is absent or undersized");
            return;
        }
        RuntimeDecodeDispatchScratch dispatch{};
        if (shape_ok)
        {
            runtime_resolve_decode_dispatch(
                runtime,
                execution_bank,
                expert_indices,
                num_experts,
                top_k,
                absolute_position_ids
                    ? absolute_position_ids[0]
                    : -1,
                row_execution_policy,
                update_runtime_histogram,
                dispatch);
        }
        for (int k = 0; k < top_k; ++k)
        {
            const int expert = expert_indices[k];
            const float weight = expert_weights[k];
            if (shape_ok)
            {
                const bool local_compute = dispatch.local_compute_flags[k];
                runtime->topk_expert_ids[k] = local_compute ? expert : -1;
                runtime->topk_weights[k] = local_compute ? weight : 0.0f;
                runtime->route_participant_ids[k] = dispatch.assigned_participants[k];
            }
            if (write_legacy_outputs)
            {
                legacy_indices[k] = static_cast<float>(expert);
                legacy_weights[k] = weight;
            }
        }

        if (update_runtime_histogram &&
            runtime_histogram_admits_rows(*runtime))
        {
            for (int k = 0; k < top_k; ++k)
            {
                const int expert = expert_indices[k];
                if (expert < 0 || expert >= kDeviceMoEMaxExperts)
                    continue;
                atomicAdd(reinterpret_cast<unsigned long long *>(
                              &runtime_selected_histogram(*runtime, 0u)[expert]),
                          static_cast<unsigned long long>(1));
                if (shape_ok && dispatch.local_compute_flags[k])
                {
                    atomicAdd(reinterpret_cast<unsigned long long *>(
                                  &runtime_local_histogram(*runtime, 0u)[expert]),
                              static_cast<unsigned long long>(1));
                }
            }
        }
    }

    __global__ void float_to_int_kernel(const float *__restrict__ input, int *__restrict__ output, int count)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count)
            output[idx] = static_cast<int>(input[idx]);
    }

    __global__ void float_to_masked_int_kernel(
        const float *__restrict__ input,
        int *__restrict__ output,
        const uint8_t *__restrict__ expert_mask,
        int count,
        int num_experts)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count)
            return;
        const int expert = static_cast<int>(input[idx]);
        output[idx] =
            (expert >= 0 && expert < num_experts && expert_mask && expert_mask[expert] != 0u)
                ? expert
                : -1;
    }

    __global__ void gather_tokens_kernel(
        const float *__restrict__ hidden,
        float *__restrict__ batch_buffer,
        const int *__restrict__ token_indices,
        int num_tokens, int d_model)
    {
        // 2D grid: blockIdx.y selects the token row, threads cover the d_model columns
        // as float4. The original flat 1D layout paid an integer div+mod (idx/d_model,
        // idx%d_model) per element and used scalar copies; this version removes both.
        // d_model is a multiple of 32 (enforced) → safe to treat the row as float4.
        const int token_slot = logical_row_from_grid_yz();
        if (token_slot >= num_tokens)
            return;
        const int n4 = d_model >> 2;
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n4)
            return;
        const int token = token_indices[token_slot];
        const float4 *src = reinterpret_cast<const float4 *>(hidden + static_cast<size_t>(token) * d_model);
        float4 *dst = reinterpret_cast<float4 *>(batch_buffer + static_cast<size_t>(token_slot) * d_model);
        dst[i] = src[i];
    }

    __global__ void gather_tokens_scalar_kernel(
        const float *__restrict__ hidden,
        float *__restrict__ batch_buffer,
        const int *__restrict__ token_indices,
        int total_elements, int d_model)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_elements)
            return;

        const int token_slot = idx / d_model;
        const int col = idx - token_slot * d_model;
        const int token = token_indices[token_slot];
        batch_buffer[idx] = hidden[static_cast<size_t>(token) * d_model + col];
    }

    /**
     * @brief Copy one logical row without staging a host token-index buffer.
     *
     * MTP verifier replay uses this primitive to keep row selection owned by
     * the captured GPU stream.  The scalar tail keeps routing rows such as
     * top_k=8 valid even though model-width rows use the vectorized float4 path.
     */
    __global__ void copy_token_row_kernel(
        const float *__restrict__ source,
        float *__restrict__ row_buffer,
        int row_index,
        int row_width)
    {
        if (row_index < 0 || row_width <= 0)
            return;
        const float *src = source + static_cast<size_t>(row_index) * row_width;
        const int n4 = row_width >> 2;
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n4)
        {
            reinterpret_cast<float4 *>(row_buffer)[i] =
                reinterpret_cast<const float4 *>(src)[i];
        }
        const int tail_start = n4 << 2;
        for (int col = tail_start + i; col < row_width; col += gridDim.x * blockDim.x)
            row_buffer[col] = src[col];
    }

    __global__ void scatter_add_kernel(
        float *__restrict__ output,
        const float *__restrict__ expert_output,
        const int *__restrict__ token_indices,
        const float *__restrict__ weights,
        int num_tokens, int d_model)
    {
        // 2D grid + float4, mirroring gather_tokens_kernel. Each token_slot maps to a
        // distinct output token (caller guarantees uniqueness — original used a plain
        // non-atomic +=, preserved here), so the read-modify-write per float4 is race
        // free. Removes the per-element div/mod and vectorizes the accumulate.
        const int token_slot = logical_row_from_grid_yz();
        if (token_slot >= num_tokens)
            return;
        const int n4 = d_model >> 2;
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n4)
            return;
        const int token = token_indices[token_slot];
        const float w = weights[token_slot];
        const float4 *ev = reinterpret_cast<const float4 *>(expert_output + static_cast<size_t>(token_slot) * d_model);
        float4 *ov = reinterpret_cast<float4 *>(output + static_cast<size_t>(token) * d_model);
        const float4 e = ev[i];
        float4 o = ov[i];
        o.x += w * e.x;
        o.y += w * e.y;
        o.z += w * e.z;
        o.w += w * e.w;
        ov[i] = o;
    }

    __global__ void scatter_add_scalar_kernel(
        float *__restrict__ output,
        const float *__restrict__ expert_output,
        const int *__restrict__ token_indices,
        const float *__restrict__ weights,
        int total_elements, int d_model)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_elements)
            return;

        const int token_slot = idx / d_model;
        const int col = idx - token_slot * d_model;
        const int token = token_indices[token_slot];
        output[static_cast<size_t>(token) * d_model + col] += weights[token_slot] * expert_output[idx];
    }

    /**
     * @brief Overwrite one destination row from a one-row scratch tensor.
     *
     * Decode-equivalent verifier replay writes each semantic row exactly once,
     * so this direct store avoids atomics and avoids async H2D staging of
     * `{row, weight}` metadata.
     */
    __global__ void write_token_row_kernel(
        float *__restrict__ destination,
        const float *__restrict__ row_buffer,
        int row_index,
        int row_width)
    {
        if (row_index < 0 || row_width <= 0)
            return;
        float *dst = destination + static_cast<size_t>(row_index) * row_width;
        const int n4 = row_width >> 2;
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n4)
        {
            reinterpret_cast<float4 *>(dst)[i] =
                reinterpret_cast<const float4 *>(row_buffer)[i];
        }
        const int tail_start = n4 << 2;
        for (int col = tail_start + i; col < row_width; col += gridDim.x * blockDim.x)
            dst[col] = row_buffer[col];
    }


    __global__ void shared_expert_gate_kernel(
        const float *__restrict__ input,
        const float *__restrict__ gate_inp,
        float *__restrict__ shared_output,
        int seq_len, int d_model,
        const int *__restrict__ device_effective_seq_len = nullptr)
    {
        const int token = blockIdx.x;
        if (token >= seq_len)
            return;

        const int effective_seq_len = clamp_effective_seq_len(seq_len, device_effective_seq_len);
        const size_t row_offset = static_cast<size_t>(token) * d_model;
        if (token >= effective_seq_len)
        {
            if ((d_model & 3) == 0)
            {
                float4 *out4 =
                    reinterpret_cast<float4 *>(shared_output + row_offset);
                const int n4 = d_model >> 2;
                for (int i = threadIdx.x; i < n4; i += blockDim.x)
                    out4[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            else
            {
                for (int i = threadIdx.x; i < d_model; i += blockDim.x)
                    shared_output[row_offset + i] = 0.0f;
            }
            return;
        }

        __shared__ float partial[kThreads];
        const float *x = input + row_offset;

        float dot = 0.0f;
        if ((d_model & 3) == 0)
        {
            const int n4 = d_model >> 2;
            const float4 *x4 = reinterpret_cast<const float4 *>(x);
            const float4 *g4 =
                reinterpret_cast<const float4 *>(gate_inp);
            for (int i = threadIdx.x; i < n4; i += blockDim.x)
            {
                const float4 a = x4[i];
                const float4 b = g4[i];
                dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
        }
        else
        {
            for (int i = threadIdx.x; i < d_model; i += blockDim.x)
                dot += x[i] * gate_inp[i];
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        if (threadIdx.x < 32)
        {
            const int lane = threadIdx.x;
            const float low = (partial[lane] + partial[lane + 128]) +
                              (partial[lane + 64] + partial[lane + 192]);
            const float high = (partial[lane + 32] + partial[lane + 160]) +
                               (partial[lane + 96] + partial[lane + 224]);
            float total = low + high;
#pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1)
                total += __shfl_down_sync(0xffffffffu, total, stride);
            if (lane == 0)
                partial[0] = total;
        }
        __syncthreads();

        const float gate = 1.0f / (1.0f + expf(-partial[0]));
        if ((d_model & 3) == 0)
        {
            const int n4 = d_model >> 2;
            float4 *out4 =
                reinterpret_cast<float4 *>(shared_output + row_offset);
            for (int i = threadIdx.x; i < n4; i += blockDim.x)
            {
                float4 v = out4[i];
                v.x = llaminar2::device_fp32_contract::multiply(gate, v.x);
                v.y = llaminar2::device_fp32_contract::multiply(gate, v.y);
                v.z = llaminar2::device_fp32_contract::multiply(gate, v.z);
                v.w = llaminar2::device_fp32_contract::multiply(gate, v.w);
                out4[i] = v;
            }
        }
        else
        {
            for (int i = threadIdx.x; i < d_model; i += blockDim.x)
                shared_output[row_offset + i] =
                    llaminar2::device_fp32_contract::multiply(gate, shared_output[row_offset + i]);
        }
    }

    /**
     * @brief Gate a full row or its disjoint output-column slice with the identical full-width dot.
     * @tparam ColumnSlice Select disjoint output columns without changing arithmetic.
     *
     * Input stride and sigmoid reduction always use the full model width. Only
     * output addressing narrows; inactive rows overwrite both outputs with zero.
     * Standalone/full/column paths retain separate RNE product and sum boundaries,
     * including subnormals despite the translation unit fast-math FTZ policy.
     */
    template<bool ColumnSlice = false>
    __global__ void shared_expert_gate_add_kernel(
        const float *__restrict__ input,
        const float *__restrict__ gate_inp,
        float *__restrict__ shared_output,
        const float *__restrict__ routed_residual,
        float *__restrict__ combined_output,
        int seq_len, int d_model,
        const int *__restrict__ device_effective_seq_len = nullptr,
        int output_columns = 0)
    {
        const int token = blockIdx.x;
        if (token >= seq_len)
            return;

        __shared__ float partial[kThreads];
        const int columns = ColumnSlice ? output_columns : d_model;
        const size_t row_offset = static_cast<size_t>(token) * columns;
        const int effective_seq_len = clamp_effective_seq_len(seq_len, device_effective_seq_len);
        if (token >= effective_seq_len)
        {
            if ((columns & 3) == 0)
            {
                float4 *shared4 =
                    reinterpret_cast<float4 *>(shared_output + row_offset);
                float4 *out4 =
                    reinterpret_cast<float4 *>(combined_output + row_offset);
                const int n4 = columns >> 2;
                for (int i = threadIdx.x; i < n4; i += blockDim.x)
                {
                    shared4[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    out4[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
            else
            {
                for (int i = threadIdx.x; i < columns; i += blockDim.x)
                {
                    shared_output[row_offset + i] = 0.0f;
                    combined_output[row_offset + i] = 0.0f;
                }
            }
            return;
        }
        const float *x = input + static_cast<size_t>(token) * d_model;

        float dot = 0.0f;
        if ((d_model & 3) == 0)
        {
            const int n4 = d_model >> 2;
            const float4 *x4 = reinterpret_cast<const float4 *>(x);
            const float4 *g4 =
                reinterpret_cast<const float4 *>(gate_inp);
            for (int i = threadIdx.x; i < n4; i += blockDim.x)
            {
                const float4 a = x4[i];
                const float4 b = g4[i];
                dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
        }
        else
        {
            for (int i = threadIdx.x; i < d_model; i += blockDim.x)
                dot += x[i] * gate_inp[i];
        }

        partial[threadIdx.x] = dot;
        __syncthreads();
        if (threadIdx.x < 32)
        {
            const int lane = threadIdx.x;
            const float low = (partial[lane] + partial[lane + 128]) +
                              (partial[lane + 64] + partial[lane + 192]);
            const float high = (partial[lane + 32] + partial[lane + 160]) +
                               (partial[lane + 96] + partial[lane + 224]);
            float total = low + high;
#pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1)
                total += __shfl_down_sync(0xffffffffu, total, stride);
            if (lane == 0)
                partial[0] = total;
        }
        __syncthreads();

        const float gate = 1.0f / (1.0f + expf(-partial[0]));
        if ((columns & 3) == 0)
        {
            const int n4 = columns >> 2;
            float4 *shared4 =
                reinterpret_cast<float4 *>(shared_output + row_offset);
            const float4 *residual4 =
                reinterpret_cast<const float4 *>(
                    routed_residual + row_offset);
            float4 *out4 =
                reinterpret_cast<float4 *>(combined_output + row_offset);
            for (int i = threadIdx.x; i < n4; i += blockDim.x)
            {
                const float4 s = shared4[i];
                const float4 r = residual4[i];
                const float4 gated = make_float4(
                    llaminar2::device_fp32_contract::multiply(gate, s.x),
                    llaminar2::device_fp32_contract::multiply(gate, s.y),
                    llaminar2::device_fp32_contract::multiply(gate, s.z),
                    llaminar2::device_fp32_contract::multiply(gate, s.w));
                shared4[i] = gated;
                // Column completion replaces a standalone shared gate followed
                // by ResidualAdd. Preserve that intervening FP32 rounding;
                // an FMA here would change the prefill arithmetic contract.
                out4[i] = make_float4(
                    llaminar2::device_fp32_contract::add(r.x, gated.x),
                    llaminar2::device_fp32_contract::add(r.y, gated.y),
                    llaminar2::device_fp32_contract::add(r.z, gated.z),
                    llaminar2::device_fp32_contract::add(r.w, gated.w));
            }
        }
        else
        {
            for (int i = threadIdx.x; i < columns; i += blockDim.x)
            {
                const float gated = llaminar2::device_fp32_contract::multiply(
                    gate, shared_output[row_offset + i]);
                shared_output[row_offset + i] = gated;
                combined_output[row_offset + i] = llaminar2::device_fp32_contract::add(
                    routed_residual[row_offset + i], gated);
            }
        }
    }

    __global__ void swiglu_kernel(float *__restrict__ gate, const float *__restrict__ up, int count)
    {
        // Process four contiguous elements per thread via float4 to cut the load/store
        // instruction count 4×. count (= m*intermediate) is not guaranteed to be a
        // multiple of 4, so vectorize the bulk and handle the ragged tail with scalars.
        const int n4 = count >> 2;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n4)
        {
            float4 g = reinterpret_cast<float4 *>(gate)[idx];
            const float4 u = reinterpret_cast<const float4 *>(up)[idx];
            g.x = silu(g.x) * u.x;
            g.y = silu(g.y) * u.y;
            g.z = silu(g.z) * u.z;
            g.w = silu(g.w) * u.w;
            reinterpret_cast<float4 *>(gate)[idx] = g;
        }
        // Tail: the last (count & 3) elements. Only the first few threads do work.
        const int tail_base = n4 << 2;
        const int tail_idx = tail_base + idx;
        if (idx < (count & 3) && tail_idx < count)
            gate[tail_idx] = silu(gate[tail_idx]) * up[tail_idx];
    }

    __global__ void weighted_add_kernel(float *__restrict__ output, const float *__restrict__ input, float weight, int count)
    {
        // float4-vectorized fused multiply-add (output += weight*input). count is
        // typically d_model (a multiple of 32), but we still handle a scalar tail so
        // the kernel is safe for any count.
        const int n4 = count >> 2;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n4)
        {
            float4 o = reinterpret_cast<float4 *>(output)[idx];
            const float4 in = reinterpret_cast<const float4 *>(input)[idx];
            o.x += weight * in.x;
            o.y += weight * in.y;
            o.z += weight * in.z;
            o.w += weight * in.w;
            reinterpret_cast<float4 *>(output)[idx] = o;
        }
        const int tail_base = n4 << 2;
        const int tail_idx = tail_base + idx;
        if (idx < (count & 3) && tail_idx < count)
            output[tail_idx] += weight * input[tail_idx];
    }


    __global__ void grouped_hidden_quantize_blockwise_kernel(
        const float *__restrict__ hidden,
        int8_t *__restrict__ A_int8,
        float *__restrict__ scales_A_blockwise,
        int K)
    {
        constexpr int kBlockSize = 32;
        const int block_idx = blockIdx.x;
        const int lane = threadIdx.x;
        const int col = block_idx * kBlockSize + lane;
        if (lane >= kBlockSize || col >= K)
            return;

        float abs_value =

            llaminar2::device_q8_activation_contract::maximumAbsoluteValue(

                hidden[col], 0.0f);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_value =
                llaminar2::device_q8_activation_contract::maximumAbsoluteValue(
                    abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

        const float scale =
            llaminar2::device_q8_activation_contract::scale(abs_value);
        if (lane == 0)
            scales_A_blockwise[block_idx] = scale;

        A_int8[col] = static_cast<int8_t>(
            llaminar2::device_q8_activation_contract::quantize(
                hidden[col], scale));
    }

    __global__ void router_gate_quantize_q8_kernel(
        const float *__restrict__ gate_weights,
        int8_t *__restrict__ gate_weights_q8,
        float *__restrict__ gate_scales,
        int K,
        int num_experts)
    {
        constexpr int kBlockSize = 32;
        const int expert = blockIdx.x;
        const int block_idx = blockIdx.y;
        const int lane = threadIdx.x;
        if (expert >= num_experts || lane >= kBlockSize)
            return;

        const int blocks_per_row = K / kBlockSize;
        const int col = block_idx * kBlockSize + lane;
        const size_t scale_idx = static_cast<size_t>(expert) * blocks_per_row + block_idx;
        const float value = (col < K)
                                ? gate_weights[static_cast<size_t>(expert) * K + col]
                                : 0.0f;

        float abs_value =

            llaminar2::device_q8_activation_contract::maximumAbsoluteValue(

                value, 0.0f);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_value =
                llaminar2::device_q8_activation_contract::maximumAbsoluteValue(
                    abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

        const float scale =
            llaminar2::device_q8_activation_contract::scale(abs_value);
        if (lane == 0)
            gate_scales[scale_idx] = scale;

        gate_weights_q8[scale_idx * kBlockSize + lane] =
            static_cast<int8_t>(
                llaminar2::device_q8_activation_contract::quantize(
                    value, scale));
    }

    /**
     * @brief Coalesced LDS-cached Q8 router logits kernel for M=1 decode.
     *
     * Loads hidden_q8 and hidden_scales once per 8-expert CTA into shared memory,
     * then assigns one warp32 per expert with 8 contiguous int32 lanes per K32
     * block (128-byte coalesced reads, zero idle lanes). Preserves the exact
     * baseline 128-thread four-warp FP32 reduction order for blocks_per_row <= 128.
     */
    /**
     * @brief Fused hidden Q8 quantization (Original or SignedHadamard32) +
     *        2-lanes-per-K32-block coalesced int4 Q8 router logits kernel (warp32).
     *
     * Replaces up to three separate single-token router launches (Hadamard Q8
     * quantize, Q8 gate logits, and original-basis Q8 re-quantize) with one
     * 256-thread (8 experts/CTA) launch while preserving the exact warp32
     * four-quarter FP32 reduction order ((w0 + w2) + (w1 + w3)).
     */
    template <bool SignedHadamard>
    __global__ __launch_bounds__(256) void router_gate_logits_single_token_q8_fused_int4_kernel(
        const float *__restrict__ hidden,
        int8_t *__restrict__ out_orig_hidden_q8,
        float *__restrict__ out_orig_hidden_scales,
        const int8_t *__restrict__ gate_weights_q8,
        const float *__restrict__ gate_scales,
        float *__restrict__ logits,
        int K,
        int num_experts)
    {
        constexpr int kMaxBlocks = 128;
        const int blocks_per_row = K >> 5;
        __shared__ int4 sh_h4[kMaxBlocks * 2];
        __shared__ float sh_hscales[kMaxBlocks];

        const int warp = threadIdx.x >> 5; // 0..7
        const int lane = threadIdx.x & 31; // 0..31
        for (int blk = warp; blk < blocks_per_row; blk += 8)
        {
            const int col = (blk << 5) + lane;
            const float orig_value = hidden[col];
            if (blockIdx.x == 0 || !SignedHadamard)
            {
                float abs_value =
                    llaminar2::device_q8_activation_contract::maximumAbsoluteValue(orig_value, 0.0f);
#pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1)
                    abs_value = llaminar2::device_q8_activation_contract::maximumAbsoluteValue(
                        abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));
                const float scale = llaminar2::device_q8_activation_contract::scale(abs_value);
                const int8_t q = static_cast<int8_t>(
                    llaminar2::device_q8_activation_contract::quantize(orig_value, scale));
                if (blockIdx.x == 0)
                {
                    if (lane == 0)
                        out_orig_hidden_scales[blk] = scale;
                    out_orig_hidden_q8[col] = q;
                }
                if constexpr (!SignedHadamard)
                {
                    if (lane == 0)
                        sh_hscales[blk] = scale;
                    reinterpret_cast<int8_t *>(sh_h4)[col] = q;
                }
            }
            if constexpr (SignedHadamard)
            {
                namespace fp = llaminar2::device_fp32_contract;
                float value = fp::floatFromBits(
                    fp::bitsFromFloat(orig_value) ^
                    (((llaminar2::router_hadamard_q8::negative_signs >> lane) & 1u) << 31u));
#pragma unroll
                for (int stride = 1; stride < 32; stride *= 2)
                {
                    const float other = llaminar2::router_hadamard_q8::exchange(value, stride);
                    value = (lane & stride) ? fp::add(other, -value) : fp::add(value, other);
                }
                value = fp::multiply(value, 0x1.6a09e6p-3f);
                float maximum = llaminar2::device_q8_activation_contract::maximumAbsoluteValue(value, 0.0f);
#pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1)
                    maximum = llaminar2::device_q8_activation_contract::maximumAbsoluteValue(
                        maximum, llaminar2::router_hadamard_q8::exchange(maximum, mask));
                const float scale = llaminar2::device_q8_activation_contract::scale(maximum);
                if (lane == 0)
                    sh_hscales[blk] = scale;
                reinterpret_cast<int8_t *>(sh_h4)[col] =
                    static_cast<int8_t>(llaminar2::device_q8_activation_contract::quantize(value, scale));
            }
        }
        __syncthreads();

        const int expert = static_cast<int>(blockIdx.x) * 8 + warp;
        if (expert >= num_experts)
            return;

        const int blk_in_16 = lane >> 1; // 0..15
        const int half = lane & 1;       // 0..1
        const size_t scale_base = static_cast<size_t>(expert) * blocks_per_row;
        const int4 *w4_row = reinterpret_cast<const int4 *>(gate_weights_q8 + scale_base * 32u);

        auto compute_16 = [&](int base_blk) -> float {
            const int blk = base_blk + blk_in_16;
            if (blk >= blocks_per_row)
                return 0.0f;
            const int idx = (blk << 1) + half;
            const int4 w = w4_row[idx];
            const int4 h = sh_h4[idx];
            int dot = __dp4a(h.x, w.x, 0);
            dot = __dp4a(h.y, w.y, dot);
            dot = __dp4a(h.z, w.z, dot);
            dot = __dp4a(h.w, w.w, dot);
            const int full = dot + __shfl_xor_sync(0xffffffffu, dot, 1);
            return static_cast<float>(full) * sh_hscales[blk] * gate_scales[scale_base + blk];
        };

        const int src_lane = (lane & 15) << 1;
        auto pack_32 = [&](int base_blk) -> float {
            if (base_blk >= blocks_per_row)
                return 0.0f;
            const float ta = compute_16(base_blk);
            const float tb = (base_blk + 16 < blocks_per_row) ? compute_16(base_blk + 16) : 0.0f;
            const float lo = __shfl_sync(0xffffffffu, ta, src_lane);
            const float hi = __shfl_sync(0xffffffffu, tb, src_lane);
            return (lane < 16) ? lo : hi;
        };

        float w0 = pack_32(0);
        float w1 = pack_32(32);
        float w2 = pack_32(64);
        float w3 = pack_32(96);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            w0 += __shfl_down_sync(0xffffffffu, w0, offset);
            w1 += __shfl_down_sync(0xffffffffu, w1, offset);
            w2 += __shfl_down_sync(0xffffffffu, w2, offset);
            w3 += __shfl_down_sync(0xffffffffu, w3, offset);
        }
        if (lane == 0)
            logits[expert] = (w0 + w2) + (w1 + w3);
    }

    __global__ __launch_bounds__(256) void router_gate_logits_single_token_q8_coalesced_lds_kernel(
        const int8_t *__restrict__ hidden_q8,
        const float *__restrict__ hidden_scales,
        const int8_t *__restrict__ gate_weights_q8,
        const float *__restrict__ gate_scales,
        float *__restrict__ logits,
        int K,
        int num_experts)
    {
        constexpr int kMaxBlocks = 128;
        const int blocks_per_row = K >> 5;
        const int words_per_row = K >> 2;
        __shared__ int sh_h4[kMaxBlocks * 8];
        __shared__ float sh_hscales[kMaxBlocks];
        __shared__ float sh_terms[8][kMaxBlocks];

        const int *h4_src = reinterpret_cast<const int *>(hidden_q8);
        for (int i = threadIdx.x; i < words_per_row; i += 256)
            sh_h4[i] = h4_src[i];
        if (threadIdx.x < blocks_per_row)
            sh_hscales[threadIdx.x] = hidden_scales[threadIdx.x];
        __syncthreads();

        const int warp = threadIdx.x >> 5;     // 0..7 (8 experts per CTA)
        const int lane = threadIdx.x & 31;     // 0..31
        const int sub = lane & 7;              // 0..7 (int32 word within K32 block)
        const int blk_in_step = lane >> 3;     // 0..3 (4 K32 blocks per warp32 step)
        const int expert = static_cast<int>(blockIdx.x) * 8 + warp;
        const bool active = (expert < num_experts);

        if (active)
        {
            const size_t scale_base = static_cast<size_t>(expert) * blocks_per_row;
            const int *w4_row = reinterpret_cast<const int *>(
                gate_weights_q8 + scale_base * 32u);
            for (int blk_base = 0; blk_base < blocks_per_row; blk_base += 4)
            {
                const int blk = blk_base + blk_in_step;
                int dot = 0;
                if (blk < blocks_per_row)
                {
                    const int w_idx = (blk << 3) + sub;
                    dot = __dp4a(sh_h4[w_idx], w4_row[w_idx], 0);
                }
                dot += __shfl_down_sync(0xffffffffu, dot, 4, 8);
                dot += __shfl_down_sync(0xffffffffu, dot, 2, 8);
                dot += __shfl_down_sync(0xffffffffu, dot, 1, 8);
                if (sub == 0 && blk < blocks_per_row)
                {
                    sh_terms[warp][blk] = static_cast<float>(dot) *
                        sh_hscales[blk] * gate_scales[scale_base + blk];
                }
            }
        }
        __syncthreads();

        if (active)
        {
            float w0 = (lane < blocks_per_row) ? sh_terms[warp][lane] : 0.0f;
            float w1 = (lane + 32 < blocks_per_row) ? sh_terms[warp][lane + 32] : 0.0f;
            float w2 = (lane + 64 < blocks_per_row) ? sh_terms[warp][lane + 64] : 0.0f;
            float w3 = (lane + 96 < blocks_per_row) ? sh_terms[warp][lane + 96] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                w0 += __shfl_down_sync(0xffffffffu, w0, offset);
                w1 += __shfl_down_sync(0xffffffffu, w1, offset);
                w2 += __shfl_down_sync(0xffffffffu, w2, offset);
                w3 += __shfl_down_sync(0xffffffffu, w3, offset);
            }
            if (lane == 0)
                logits[expert] = (w0 + w2) + (w1 + w3);
        }
    }

    __global__ void router_gate_logits_single_token_q8_kernel(
        const int8_t *__restrict__ hidden_q8,
        const float *__restrict__ hidden_scales,
        const int8_t *__restrict__ gate_weights_q8,
        const float *__restrict__ gate_scales,
        float *__restrict__ logits,
        int K,
        int num_experts)
    {
        constexpr int kBlockSize = 32;
        const int expert = blockIdx.x;
        if (expert >= num_experts)
            return;

        const int blocks_per_row = K / kBlockSize;
        float sum = 0.0f;
        for (int block_idx = threadIdx.x; block_idx < blocks_per_row; block_idx += blockDim.x)
        {
            const int *h4 = reinterpret_cast<const int *>(hidden_q8 + block_idx * kBlockSize);
            const size_t scale_idx = static_cast<size_t>(expert) * blocks_per_row + block_idx;
            const int *w4 = reinterpret_cast<const int *>(gate_weights_q8 + scale_idx * kBlockSize);

            int block_acc = 0;
#pragma unroll
            for (int group = 0; group < 8; ++group)
                block_acc = __dp4a(h4[group], w4[group], block_acc);
            sum += static_cast<float>(block_acc) * hidden_scales[block_idx] * gate_scales[scale_idx];
        }

        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);

        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int num_warps = blockDim.x >> 5;
        __shared__ float warp_sums[32];
        if (lane == 0)
            warp_sums[warp] = sum;
        __syncthreads();
        if (warp == 0)
        {
            float v = (lane < num_warps) ? warp_sums[lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, offset);
            if (lane == 0)
                logits[expert] = v;
        }
    }

    /**
     * @brief Quantize each verifier hidden row with the same per-32-column scale
     *        rule used by serial Q8 router decode.
     *
     * The grid is `[blocks_per_row, verifier_row]`.  Each CUDA block owns one
     * 32-column quantization group for one logical decode row, so the scale and
     * rounded int8 values are byte-for-byte comparable to invoking the serial
     * single-row quantizer independently for that row.  The grouped verifier
     * path relies on that property before running top-k over all rows together.
     */
    __global__ void grouped_hidden_quantize_blockwise_rows_kernel(
        const float *__restrict__ hidden,
        int8_t *__restrict__ A_int8,
        float *__restrict__ scales_A_blockwise,
        int seq_len,
        int K, const int *live_rows = nullptr)
    {
        constexpr int kBlockSize = 32;
        const int block_idx = blockIdx.x;
        const int row = blockIdx.y;
        const int lane = threadIdx.x;
        const int col = block_idx * kBlockSize + lane;
        const int live = live_rows ? *live_rows : seq_len;
        if (live < 0 || live > seq_len) { __trap(); return; }
        if (row >= live || lane >= kBlockSize || col >= K)
            return;

        const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(K);
        const int blocks_per_row = K / kBlockSize;
        const float value = hidden[row_offset + col];
        float abs_value =
            llaminar2::device_q8_activation_contract::maximumAbsoluteValue(
                value, 0.0f);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_value =
                llaminar2::device_q8_activation_contract::maximumAbsoluteValue(
                    abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

        const float scale =
            llaminar2::device_q8_activation_contract::scale(abs_value);
        if (lane == 0)
            scales_A_blockwise[static_cast<size_t>(row) * blocks_per_row + block_idx] = scale;

        A_int8[row_offset + col] = static_cast<int8_t>(
            llaminar2::device_q8_activation_contract::quantize(
                value, scale));
    }

    /**
     * @brief Compute Q8 router logits for all verifier rows in one expert block.
     *
     * One CUDA block owns one expert and accumulates one four-row verifier tile.
     * The gate Q8 payload and gate scale for each 32-column block are loaded
     * once, then reused across the row-local DP4A accumulators.  Each row still
     * uses the same per-row hidden scales and the same K traversal order as
     * serial decode, so this is the economical grouped form of the serial Q8
     * router rather than a row-replay launch hidden inside the backend. Grid-Y
     * covers additional tiles, keeping arbitrary runtime M in one launch.
     */
    __global__ void router_gate_logits_q8_grouped_verifier_kernel(
        const int8_t *__restrict__ hidden_q8,
        const float *__restrict__ hidden_scales,
        const int8_t *__restrict__ gate_weights_q8,
        const float *__restrict__ gate_scales,
        float *__restrict__ logits,
        int seq_len,
        int K,
        int num_experts)
    {
        constexpr int kBlockSize = 32;
        constexpr int kRowsPerBlock = 4;
        const int expert = blockIdx.x;
        const int first_row = blockIdx.y * kRowsPerBlock;
        if (expert >= num_experts)
            return;
        const int tile_rows = min(kRowsPerBlock, seq_len - first_row);
        if (tile_rows <= 0)
            return;

        const int blocks_per_row = K / kBlockSize;
        float sums[kRowsPerBlock] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int block_idx = threadIdx.x; block_idx < blocks_per_row; block_idx += blockDim.x)
        {
            const size_t scale_idx = static_cast<size_t>(expert) * blocks_per_row + block_idx;
            const int *w4 = reinterpret_cast<const int *>(gate_weights_q8 + scale_idx * kBlockSize);
            const float gate_scale = gate_scales[scale_idx];

            int w_values[8];
#pragma unroll
            for (int group = 0; group < 8; ++group)
                w_values[group] = w4[group];

            for (int tile_row = 0; tile_row < tile_rows; ++tile_row)
            {
                const int row = first_row + tile_row;
                const int8_t *row_hidden =
                    hidden_q8 + static_cast<size_t>(row) * static_cast<size_t>(K);
                const float *row_scales =
                    hidden_scales + static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row);
                const int *h4 = reinterpret_cast<const int *>(row_hidden + block_idx * kBlockSize);

                int block_acc = 0;
#pragma unroll
                for (int group = 0; group < 8; ++group)
                    block_acc = __dp4a(h4[group], w_values[group], block_acc);
                sums[tile_row] +=
                    static_cast<float>(block_acc) * row_scales[block_idx] * gate_scale;
            }
        }

        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int num_warps = blockDim.x >> 5;
        __shared__ float warp_sums[kRowsPerBlock][32];

        for (int tile_row = 0; tile_row < tile_rows; ++tile_row)
        {
            float v = sums[tile_row];
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, offset);
            if (lane == 0)
                warp_sums[tile_row][warp] = v;
        }
        __syncthreads();

        if (warp == 0)
        {
            for (int tile_row = 0; tile_row < tile_rows; ++tile_row)
            {
                float v = (lane < num_warps) ? warp_sums[tile_row][lane] : 0.0f;
                for (int offset = 16; offset > 0; offset >>= 1)
                    v += __shfl_down_sync(0xffffffffu, v, offset);
                if (lane == 0)
                {
                    const int row = first_row + tile_row;
                    logits[static_cast<size_t>(row) * num_experts + expert] = v;
                }
            }
        }
    }

} // namespace

extern "C"
{
bool cudaMoE_internal_quantize_hidden_blockwise(
    const float *hidden, int8_t *hidden_q8, float *hidden_scales, int K, void *stream)
{
    const int blocks_per_row = K / 32;
    grouped_hidden_quantize_blockwise_kernel<<<blocks_per_row, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        hidden, hidden_q8, hidden_scales, K);
    return finishLaunch("cudaMoE_internal_quantize_hidden_blockwise");
}

    bool cudaMoE_publish_current_batch_llep_evidence_to_generation_control(
        const void *runtime_layers,
        int layer_count,
        int *generation_control,
        int generation_control_stride,
        int request_count,
        int movement_layer_count_index,
        int non_owner_assignment_layer_count_index,
        int device_idx,
        void *stream)
    {
        if (!runtime_layers || layer_count <= 0 || !generation_control ||
            generation_control_stride <= 0 || request_count <= 0 ||
            movement_layer_count_index < 0 ||
            movement_layer_count_index >= generation_control_stride ||
            non_owner_assignment_layer_count_index < 0 ||
            non_owner_assignment_layer_count_index >=
                generation_control_stride ||
            movement_layer_count_index ==
                non_owner_assignment_layer_count_index ||
            !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int kEvidenceThreads = 256;
        publish_current_batch_llep_evidence_to_generation_control_kernel<<<
            1,
            kEvidenceThreads,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            static_cast<const DeviceMoELayerRuntimeView *>(runtime_layers),
            layer_count,
            generation_control,
            generation_control_stride,
            request_count,
            movement_layer_count_index,
            non_owner_assignment_layer_count_index);
        return finishLaunch(
            "cudaMoE_publish_current_batch_llep_evidence_to_generation_control");
    }

    bool cudaMoE_quantize_router_gate_q8(
        const float *gate_weights, int8_t *gate_weights_q8, float *gate_scales,
        int d_model, int num_experts,
        int device_idx, void *stream, llaminar2::RouterQ8Basis basis)
    {
        if (!gate_weights || !gate_weights_q8 || !gate_scales || !stream ||
            !stream || (basis != llaminar2::RouterQ8Basis::Original &&
            basis != llaminar2::RouterQ8Basis::SignedHadamard32) ||
            d_model <= 0 || num_experts <= 0 || (d_model % 32) != 0)
        {
            std::fprintf(stderr, "[cudaMoE_quantize_router_gate_q8] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        if (basis == llaminar2::RouterQ8Basis::SignedHadamard32)
            return llaminar2::router_hadamard_q8::launch(
                gate_weights, gate_weights_q8, gate_scales, num_experts, d_model, stream);
        const int blocks_per_row = d_model / 32;
        router_gate_quantize_q8_kernel<<<dim3(num_experts, blocks_per_row), dim3(32), 0,
                                          static_cast<cudaStream_t>(stream)>>>(
            gate_weights, gate_weights_q8, gate_scales, d_model, num_experts);
        return finishLaunch("cudaMoE_quantize_router_gate_q8");
    }

    bool cudaMoE_gate_logits_single_token_q8_weights(
        const float *hidden, int8_t *hidden_q8, float *hidden_scales,
        const int8_t *gate_weights_q8, const float *gate_scales, float *logits,
        int d_model, int num_experts,
        int device_idx, void *stream, llaminar2::RouterQ8Basis basis)
    {
        if (!hidden || !hidden_q8 || !hidden_scales ||
            !gate_weights_q8 || !gate_scales || !logits || !stream ||
            !stream || (basis != llaminar2::RouterQ8Basis::Original &&
            basis != llaminar2::RouterQ8Basis::SignedHadamard32) ||
            d_model <= 0 || num_experts <= 0 || (d_model % 32) != 0)
        {
            std::fprintf(stderr, "[cudaMoE_gate_logits_single_token_q8_weights] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        const int blocks_per_row = d_model / 32;
        if (blocks_per_row <= 128 &&
            ((reinterpret_cast<std::uintptr_t>(gate_weights_q8) & 15u) == 0u))
        {
            const int blocks = (num_experts + 7) / 8;
            if (basis == llaminar2::RouterQ8Basis::SignedHadamard32)
            {
                router_gate_logits_single_token_q8_fused_int4_kernel<true><<<blocks, 256, 0, cuda_stream>>>(
                    hidden, hidden_q8, hidden_scales, gate_weights_q8, gate_scales, logits, d_model, num_experts);
            }
            else
            {
                router_gate_logits_single_token_q8_fused_int4_kernel<false><<<blocks, 256, 0, cuda_stream>>>(
                    hidden, hidden_q8, hidden_scales, gate_weights_q8, gate_scales, logits, d_model, num_experts);
            }
            return finishLaunch("cudaMoE_gate_logits_single_token_q8_fused_int4");
        }
        if (basis == llaminar2::RouterQ8Basis::SignedHadamard32)
        {
            if (!llaminar2::router_hadamard_q8::launch(
                    hidden, hidden_q8, hidden_scales, 1, d_model, stream)) return false;
        }
        else
        {
            grouped_hidden_quantize_blockwise_kernel<<<blocks_per_row, 32, 0, cuda_stream>>>(
            hidden, hidden_q8, hidden_scales, d_model);
        }
        if (!finishLaunch("cudaMoE_router_hidden_quantize_q8"))
            return false;

        if (blocks_per_row <= 128)
        {
            const int blocks = (num_experts + 7) / 8;
            router_gate_logits_single_token_q8_coalesced_lds_kernel<<<blocks, 256, 0, cuda_stream>>>(
                hidden_q8, hidden_scales, gate_weights_q8, gate_scales, logits, d_model, num_experts);
        }
        else
        {
            constexpr int kRouterThreads = 128;
            router_gate_logits_single_token_q8_kernel<<<num_experts, kRouterThreads, 0, cuda_stream>>>(
                hidden_q8, hidden_scales, gate_weights_q8, gate_scales, logits, d_model, num_experts);
        }
        if (!finishLaunch("cudaMoE_gate_logits_single_token_q8_weights")) return false;
        // The last router read precedes this overwrite on the SAME stream.
        // Expert projections borrow original coordinates, never the rotation.
        if (basis == llaminar2::RouterQ8Basis::SignedHadamard32)
        {
            grouped_hidden_quantize_blockwise_kernel<<<blocks_per_row, 32, 0, cuda_stream>>>(
            hidden, hidden_q8, hidden_scales, d_model);
            if (!finishLaunch("cudaMoE_router_original_hidden_publication")) return false;
        }
        return true;
    }

    /** @brief Explicit execution policy; both implementations retain serial Q8 arithmetic. */
    enum class RouterQ8RowsExecution { SerialEquivalent, TensorCorePrefill };

    /** @copydoc cudaMoE_publish_original_hidden_q8_rows */
    bool cudaMoE_publish_original_hidden_q8_rows(
        const float *hidden, int8_t *codes, float *scales,
        int rows, int width, int device, void *stream, const int *live_rows)
    {
        if (!hidden || !codes || !scales ||
            reinterpret_cast<std::uintptr_t>(stream) <= 2u || rows <= 0 ||
            width <= 0 || width % 32 != 0 || device < 0 ||
            cudaSetDevice(device) != cudaSuccess)
            return false;
        grouped_hidden_quantize_blockwise_rows_kernel<<<dim3(width / 32, rows), 32, 0,
            static_cast<cudaStream_t>(stream)>>>(hidden, codes, scales, rows, width, live_rows);
        return finishLaunch("cudaMoE_publish_original_hidden_q8_rows");
    }

    /**
     * @brief Share quantization and expert-bank publication across exact Q8 implementations.
     * @param hidden Original FP32 rows before any router-only rotation.
     * @param hidden_q8 Admitted signed-byte scratch and original expert publication.
     * @param hidden_scales Admitted block32 scales for that bank.
     * @param gate_weights_q8 Immutable prepared router codes in the declared basis.
     * @param gate_scales Their block32 scales. @param logits Admitted FP32 output.
     * @param seq_len Captured row capacity. @param d_model Positive block32 width.
     * @param num_experts Output width. @param device_idx Owning CUDA ordinal.
     * @param stream Exact non-null producer stream. @param basis Paired gate/input basis.
     * @param execution Explicit verifier or tensor-core prefill implementation.
     * @param live_rows Optional device-owned live extent for retained prefill replay.
     * @param hardware Prepared immutable device limits, consumed only by tensor-core prefill.
     * @return Whether every ordered launch succeeded.
     */
    static bool launchRouterQ8Rows(
        const float *hidden, int8_t *hidden_q8, float *hidden_scales,
        const int8_t *gate_weights_q8, const float *gate_scales, float *logits,
        int seq_len, int d_model, int num_experts,
        int device_idx, void *stream, llaminar2::RouterQ8Basis basis,
        RouterQ8RowsExecution execution, const int *live_rows,
        const llaminar2::cuda::RouterQ8PrefillHardware &hardware)
    {
        if (!hidden || !hidden_q8 || !hidden_scales ||
            !gate_weights_q8 || !gate_scales || !logits || !stream ||
            seq_len <= 0 ||
            (execution == RouterQ8RowsExecution::TensorCorePrefill &&
                (!hardware.valid() || hardware.device != device_idx)) ||
            (basis != llaminar2::RouterQ8Basis::Original &&
            basis != llaminar2::RouterQ8Basis::SignedHadamard32) ||
            d_model <= 0 || num_experts <= 0 || (d_model % 32) != 0)
        {
            std::fprintf(stderr, "[cudaMoE_gate_logits_q8_weights_decode_equivalent_rows] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        const int blocks_per_row = d_model / 32;
        if (basis == llaminar2::RouterQ8Basis::SignedHadamard32)
        {
            if (!llaminar2::router_hadamard_q8::launch(
                    hidden, hidden_q8, hidden_scales, seq_len, d_model, stream, live_rows)) return false;
        }
        else
        {
            grouped_hidden_quantize_blockwise_rows_kernel<<<dim3(blocks_per_row, seq_len), 32, 0, cuda_stream>>>(
            hidden, hidden_q8, hidden_scales, seq_len, d_model, live_rows);
        }
        if (!finishLaunch("cudaMoE_router_hidden_quantize_q8_rows"))
            return false;

        if (execution == RouterQ8RowsExecution::TensorCorePrefill)
        {
            if (!llaminar2::cuda::routerQ8Prefill({
                    .hidden_codes = hidden_q8, .hidden_scales = hidden_scales,
                    .gate_codes = gate_weights_q8, .gate_scales = gate_scales,
                    .logits = logits, .capacity = seq_len, .width = d_model,
                    .experts = num_experts, .live_rows = live_rows,
                    .hardware = hardware}, stream)) return false;
        }
        else
        {
            constexpr int kRouterThreads = 128, kRowsPerBlock = 4;
            const dim3 grid(num_experts, (seq_len + kRowsPerBlock - 1) / kRowsPerBlock);
            router_gate_logits_q8_grouped_verifier_kernel<<<grid, kRouterThreads, 0, cuda_stream>>>(
                hidden_q8, hidden_scales, gate_weights_q8, gate_scales, logits,
                seq_len, d_model, num_experts);
            if (!finishLaunch("cudaMoE_gate_logits_q8_weights_decode_equivalent_rows")) return false;
        }
        // The last router read precedes this overwrite on the SAME stream.
        // Expert projections borrow original coordinates, never the rotation.
        if (basis == llaminar2::RouterQ8Basis::SignedHadamard32)
        {
            grouped_hidden_quantize_blockwise_rows_kernel<<<dim3(blocks_per_row, seq_len), 32, 0, cuda_stream>>>(
            hidden, hidden_q8, hidden_scales, seq_len, d_model, live_rows);
            if (!finishLaunch("cudaMoE_router_original_hidden_publication")) return false;
        }
        return true;
    }

    /** @copydoc cudaMoE_gate_logits_q8_weights_decode_equivalent_rows */
    bool cudaMoE_gate_logits_q8_weights_decode_equivalent_rows(
        const float *hidden, int8_t *hidden_q8, float *hidden_scales,
        const int8_t *gate_weights_q8, const float *gate_scales, float *logits,
        int seq_len, int d_model, int num_experts, int device_idx, void *stream,
        llaminar2::RouterQ8Basis basis)
    {
        return launchRouterQ8Rows(hidden, hidden_q8, hidden_scales, gate_weights_q8,
            gate_scales, logits, seq_len, d_model, num_experts, device_idx, stream,
            basis, RouterQ8RowsExecution::SerialEquivalent, nullptr, {});
    }

    /** @copydoc cudaMoE_gate_logits_q8_weights_prefill_rows */
    bool cudaMoE_gate_logits_q8_weights_prefill_rows(
        const float *hidden, int8_t *hidden_q8, float *hidden_scales,
        const int8_t *gate_weights_q8, const float *gate_scales, float *logits,
        int seq_len, int d_model, int num_experts, int device_idx, void *stream,
        const int *live_rows, llaminar2::RouterQ8Basis basis,
        const llaminar2::cuda::RouterQ8PrefillHardware &hardware)
    {
        return launchRouterQ8Rows(hidden, hidden_q8, hidden_scales, gate_weights_q8,
            gate_scales, logits, seq_len, d_model, num_experts, device_idx, stream,
            basis, RouterQ8RowsExecution::TensorCorePrefill, live_rows, hardware);
    }

    /**
     * @brief Row-isolated FP32 router logits for grouped verifier publication.
     *
     * Ordinary prefill may use a tiled SGEMM once M>=2.  Verifier publication
     * cannot: accepted rows must match the M=1 serial decode accumulation order.
     * Launching route_logits_kernel over all `(expert,row)` pairs preserves that
     * order while still avoiding host-side row replay.
     */
    bool cudaMoE_route_logits_decode_equivalent_rows(
        const float *hidden, const float *gate_weights, float *logits,
        int seq_len, int d_model, int num_experts,
        int device_idx, void *stream)
    {
        if (!hidden || !gate_weights || !logits || !stream ||
            seq_len <= 0 ||
            d_model <= 0 || num_experts <= 0)
        {
            std::fprintf(stderr, "[cudaMoE_route_logits_decode_equivalent_rows] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int kRowsPerBlock = 4;
        const dim3 grid(num_experts, (seq_len + kRowsPerBlock - 1) / kRowsPerBlock);
        route_logits_fp32_grouped_verifier_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            hidden, gate_weights, logits, seq_len, d_model, num_experts);
        return finishLaunch("cudaMoE_route_logits_decode_equivalent_rows");
    }

    /**
     * @brief Row-equivalent, economical BF16 router logits for verifier rows.
     */
    bool cudaMoE_route_logits_bf16_decode_equivalent_rows(
        const float *hidden, const void *gate_weights, float *logits,
        int seq_len, int d_model, int num_experts,
        int device_idx, void *stream)
    {
        if (!hidden || !gate_weights || !logits || !stream ||
            seq_len <= 0 ||
            d_model <= 0 || num_experts <= 0)
        {
            std::fprintf(stderr, "[cudaMoE_route_logits_bf16_decode_equivalent_rows] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int kRowsPerBlock = 4;
        const dim3 grid(num_experts, (seq_len + kRowsPerBlock - 1) / kRowsPerBlock);
        route_logits_bf16_grouped_verifier_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            hidden, static_cast<const __nv_bfloat16 *>(gate_weights), logits,
            seq_len, d_model, num_experts);
        return finishLaunch("cudaMoE_route_logits_bf16_decode_equivalent_rows");
    }

    bool cudaMoE_route_logits_with_geometry(
        const float *hidden,
        const float *gate_weights,
        float *logits,
        int seq_len,
        int d_model,
        int num_experts,
        int device_idx,
        void *stream,
        llaminar2::CUDAMoERouterPrefillGeometry geometry)
    {
        if (!hidden || !gate_weights || !logits || !stream ||
            seq_len <= 0 || d_model <= 0 || num_experts <= 0)
        {
            std::fprintf(
                stderr,
                "[cudaMoE_route_logits_with_geometry] invalid arguments\n");
            return false;
        }
        if (cudaSetDevice(device_idx) != cudaSuccess)
            return false;

        return dispatch_route_logits_geometry(hidden, gate_weights, logits,
            seq_len, d_model, num_experts, static_cast<cudaStream_t>(stream), geometry);
    }

    bool cudaMoE_route_logits_query_geometry_resources(
        llaminar2::CUDAMoERouterPrefillGeometry geometry,
        int *registers_per_thread,
        std::size_t *local_memory_bytes_per_thread,
        std::size_t *static_shared_memory_bytes,
        int *max_threads_per_block,
        int *max_active_blocks_per_sm)
    {
        if (!registers_per_thread || !local_memory_bytes_per_thread ||
            !static_shared_memory_bytes || !max_threads_per_block ||
            !max_active_blocks_per_sm)
        {
            return false;
        }

        RouterPrefillKernelResources resources{};
        bool queried = false;
        using Geometry = llaminar2::CUDAMoERouterPrefillGeometry;
        switch (geometry)
        {
        case Geometry::Tile64x64:
            queried = query_route_logits_tiled_geometry_resources<64, 64, 16, 4, 4>(&resources);
            break;
        case Geometry::Tile64x32:
            queried = query_route_logits_tiled_geometry_resources<64, 32, 16, 4, 2>(&resources);
            break;
        case Geometry::Tile32x64:
            queried = query_route_logits_tiled_geometry_resources<32, 64, 16, 2, 4>(&resources);
            break;
        case Geometry::Tile32x32:
            queried = query_route_logits_tiled_geometry_resources<32, 32, 32, 2, 2>(&resources);
            break;
        case Geometry::Tile32x32K16:
            queried = query_route_logits_tiled_geometry_resources<32, 32, 16, 2, 2>(&resources);
            break;
        case Geometry::Tile32x24:
            queried = query_route_logits_tiled_geometry_resources<32, 24, 16, 1, 3>(&resources);
            break;
        case Geometry::Tile24x32:
            queried = query_route_logits_tiled_geometry_resources<24, 32, 16, 3, 1>(&resources);
            break;
        case Geometry::Tile64x16:
            queried = query_route_logits_tiled_geometry_resources<64, 16, 16, 4, 1>(&resources);
            break;
        case Geometry::Tile32x16:
            queried = query_route_logits_tiled_geometry_resources<32, 16, 16, 2, 1>(&resources);
            break;
        case Geometry::Tile16x64:
            queried = query_route_logits_tiled_geometry_resources<16, 64, 16, 1, 4>(&resources);
            break;
        case Geometry::Tile16x32:
            queried = query_route_logits_tiled_geometry_resources<16, 32, 16, 1, 2>(&resources);
            break;
        case Geometry::Tile16x16:
            queried = query_route_logits_tiled_geometry_resources<16, 16, 16, 1, 1>(&resources);
            break;
        case Geometry::Tile8x64:
            queried = query_route_logits_tiled_geometry_resources<8, 64, 16, 1, 2>(&resources);
            break;
        case Geometry::Tile8x32:
            queried = query_route_logits_tiled_geometry_resources<8, 32, 16, 1, 1>(&resources);
            break;
        }
        if (!queried)
            return false;

        *registers_per_thread = resources.registers_per_thread;
        *local_memory_bytes_per_thread =
            resources.local_memory_bytes_per_thread;
        *static_shared_memory_bytes = resources.static_shared_memory_bytes;
        *max_threads_per_block = resources.max_threads_per_block;
        *max_active_blocks_per_sm = resources.max_active_blocks_per_sm;
        return true;
    }

    bool cudaMoE_route_logits(
        const float *hidden,
        const float *gate_weights,
        float *logits,
        int seq_len,
        int d_model,
        int num_experts,
        int device_idx,
        void *stream)
    {
        // M=1 decode keeps one block per expert so all SMs receive useful work.
        // Prefill resolves a typed tile exactly once while graph topology is
        // built; captured replay contains only the selected physical kernel.
        constexpr int kRouteTiledMinTokens = 2;
        if (seq_len >= kRouteTiledMinTokens)
        {
            return cudaMoE_route_logits_with_geometry(
                hidden,
                gate_weights,
                logits,
                seq_len,
                d_model,
                num_experts,
                device_idx,
                stream,
                llaminar2::selectCUDAMoERouterPrefillGeometry(
                    seq_len, d_model, num_experts));
        }

        if (!hidden || !gate_weights || !logits || !stream ||
            seq_len <= 0 || d_model <= 0 || num_experts <= 0)
        {
            std::fprintf(stderr, "[cudaMoE_route_logits] invalid arguments\n");
            return false;
        }
        if (cudaSetDevice(device_idx) != cudaSuccess)
            return false;
        dim3 grid(num_experts, seq_len);
        route_logits_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            hidden, gate_weights, logits, seq_len, d_model, num_experts);
        return finishLaunch("cudaMoE_route_logits");
    }

    /** @brief Submit actual-half row-invariant routing without changing expert hidden storage.
     * @param hidden Original FP32 [M,K]. @param gate_weights Physical FP16 [E,K].
     * @param logits FP32 [M,E]. @param seq_len Physical rows. @param d_model K.
     * @param num_experts E. @param device_idx Device ordinal. @param stream Exact stream.
     * @param live_rows Optional device-owned active row count. @return Submission success.
     */
    bool cudaMoE_route_logits_fp16(const float *hidden, const void *gate_weights, float *logits,
        int seq_len, int d_model, int num_experts, int device_idx, void *stream,
        const int *live_rows)
    {
        if (!hidden || !gate_weights || !logits || !stream || seq_len <= 0 ||
            d_model <= 0 || num_experts <= 0 || cudaSetDevice(device_idx) != cudaSuccess) return false;
        launch_fp16_router_rows<false>(hidden, static_cast<const __half *>(gate_weights), logits,
            seq_len, seq_len, d_model, num_experts, live_rows, {}, static_cast<cudaStream_t>(stream));
        return finishLaunch("cudaMoE_route_logits_fp16");
    }

    bool cudaMoE_route_logits_bf16(const float *hidden, const void *gate_weights, float *logits,
                                   int seq_len, int d_model, int num_experts,
                                   int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        dim3 grid(num_experts, seq_len);
        route_logits_bf16_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            hidden, static_cast<const __nv_bfloat16 *>(gate_weights), logits,
            seq_len, d_model, num_experts);
        return finishLaunch("cudaMoE_route_logits_bf16");
    }

    bool cudaMoE_softmax_topk(float *logits, float *expert_indices, float *expert_weights,
                              int seq_len, int num_experts, int top_k, bool normalize_weights,
                              int device_idx, void *stream,
                              const int *device_effective_seq_len,
                              void *deferred_selected_route_ledger)
    {
        if (num_experts > kMaxExperts || top_k > kMaxTopK)
            return false;
        cudaSetDevice(device_idx);
        softmax_topk_kernel<<<seq_len, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            logits, expert_indices, expert_weights, seq_len, num_experts, top_k,
            normalize_weights, device_effective_seq_len,
            static_cast<DeviceMoELayerRuntimeView *>(
                deferred_selected_route_ledger));
        return finishLaunch("cudaMoE_softmax_topk");
    }

    bool cudaMoE_softmax_topk_decode_runtime(float *logits,
                                             void *runtime_layer,
                                             float *legacy_indices, float *legacy_weights,
                                             int num_experts, int top_k, bool normalize_weights,
                                             bool write_legacy_outputs, bool update_runtime_histogram,
                                             llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy,
                                             void *runtime_layers,
                                             const void *rebalance_plan_entries,
                                             uint32_t rebalance_plan_capacity,
                                             void *rebalance_command_header,
                                             const void *rebalance_local_transfer_slots,
                                             uint32_t rebalance_local_transfer_slot_count,
                                             const void *rebalance_config,
                                             void *rebalance_apply_status,
                                             void *rebalance_controller_state,
                                             int rebalance_target_layer,
                                             uint32_t rebalance_command_buffer_count,
                                             const int32_t *absolute_position_ids,
                                             int device_idx, void *stream,
        const int32_t *active_row_count_device)
    {
        if (!runtime_layer ||
            num_experts <= 0 ||
            num_experts > kDeviceMoEMaxExperts ||
            top_k <= 0 ||
            top_k > kMaxTopK ||
            top_k > num_experts)
            return false;
        cudaSetDevice(device_idx);
        const bool apply_ready_rebalance = runtime_layers != nullptr;
        const bool complete_rebalance_binding =
            runtime_layers && rebalance_plan_entries &&
            rebalance_plan_capacity > 0u && rebalance_command_header &&
            rebalance_config && rebalance_apply_status &&
            rebalance_controller_state &&
            rebalance_command_buffer_count > 0u;
        const bool any_rebalance_binding =
            runtime_layers || rebalance_plan_entries ||
            rebalance_plan_capacity > 0u || rebalance_command_header ||
            rebalance_local_transfer_slots ||
            rebalance_local_transfer_slot_count > 0u || rebalance_config ||
            rebalance_apply_status || rebalance_controller_state ||
            rebalance_target_layer >= 0;
        if (any_rebalance_binding && !complete_rebalance_binding)
            return false;

        // The immutable fragment policy has independent selected slots and
        // needs no replica scheduler state or maintenance ABI in its kernel.
        if (row_execution_policy == llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal &&
            !apply_ready_rebalance)
        {
            if (!logits || !stream ||
                (write_legacy_outputs && (!legacy_indices || !legacy_weights))) return false;
#define LLAMINAR_LAUNCH_FRAGMENT_ROUTE(HISTOGRAM) \
            softmax_topk_decode_independent_runtime_kernel<HISTOGRAM, \
                llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal> \
                <<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>( \
                    logits, static_cast<DeviceMoELayerRuntimeView *>(runtime_layer), \
                    legacy_indices, legacy_weights, num_experts, top_k, \
                    normalize_weights, write_legacy_outputs, active_row_count_device)
            if (update_runtime_histogram)
                LLAMINAR_LAUNCH_FRAGMENT_ROUTE(true);
            else
                LLAMINAR_LAUNCH_FRAGMENT_ROUTE(false);
#undef LLAMINAR_LAUNCH_FRAGMENT_ROUTE
            return finishLaunch("cudaMoE_softmax_topk_decode_intermediate_shard_runtime");
        }

        DeviceMoERebalanceConfigView rebalance_cfg{};
        if (rebalance_config)
            rebalance_cfg = *static_cast<const DeviceMoERebalanceConfigView *>(rebalance_config);

#define LLAMINAR_LAUNCH_DECODE_ROUTE(HISTOGRAM, APPLY)                                      \
        softmax_topk_decode_runtime_kernel<HISTOGRAM, APPLY>                                \
            <<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(                         \
                logits, static_cast<DeviceMoELayerRuntimeView *>(runtime_layer),             \
                static_cast<DeviceMoELayerRuntimeView *>(runtime_layers),                    \
                legacy_indices, legacy_weights, num_experts, top_k, normalize_weights,       \
                write_legacy_outputs, row_execution_policy, absolute_position_ids,    \
                static_cast<const DeviceMoERebalancePlanEntryView *>(rebalance_plan_entries),\
                rebalance_plan_capacity,                                                     \
                static_cast<DeviceMoERebalanceCommandBufferHeaderView *>(                    \
                    rebalance_command_header),                                               \
                static_cast<const DeviceMoEExpertDirectoryEntryView *>(                      \
                    rebalance_local_transfer_slots),                                         \
                rebalance_local_transfer_slot_count, rebalance_cfg,                          \
                static_cast<DeviceMoERebalanceApplyStatusView *>(rebalance_apply_status),     \
                static_cast<DeviceMoERebalanceGraphControllerStateView *>(                   \
                    rebalance_controller_state),                                             \
                rebalance_target_layer, rebalance_command_buffer_count, active_row_count_device)

        if (update_runtime_histogram)
        {
            if (apply_ready_rebalance)
                LLAMINAR_LAUNCH_DECODE_ROUTE(true, true);
            else
                LLAMINAR_LAUNCH_DECODE_ROUTE(true, false);
        }
        else if (apply_ready_rebalance)
        {
            LLAMINAR_LAUNCH_DECODE_ROUTE(false, true);
        }
        else
        {
            LLAMINAR_LAUNCH_DECODE_ROUTE(false, false);
        }
#undef LLAMINAR_LAUNCH_DECODE_ROUTE
        return finishLaunch("cudaMoE_softmax_topk_decode_runtime");
    }

        /**
     * @brief Publish the exact decode route and its descriptors in one retained kernel.
     * @param logits Original FP32 router logits, overwritten with probabilities.
     * @param runtime Original device runtime retaining the acquired placement bank.
     * @param legacy_indices Optional complete selected ID output.
     * @param legacy_weights Optional complete selected weight output.
     * @param num_experts Exact original source extent.
     * @param top_k Exact selected packet width.
     * @param normalize_weights Preserve the backend's original serial normalization.
     * @param write_legacy_outputs Whether both complete selected outputs are bound.
     * @param update_runtime_histogram Whether this graph owns demand evidence.
     * @param row_policy Original independent whole-owner or local-fragment contract.
     * @param publication Existing arena descriptor outputs with complete identity.
     * @param device_idx Original rank-local device ordinal.
     * @param stream Exact non-default producer and consumer stream.
     * @param active_row_count_device Optional device-owned live row count.
     * @return Whether validation and native submission succeeded.
     */
__attribute__((visibility("default"))) bool cudaMoE_softmax_topk_decode_selected_descriptors(
        float *logits, void *runtime, float *legacy_indices, float *legacy_weights,
        int num_experts, int top_k, bool normalize_weights, bool write_legacy_outputs,
        bool update_runtime_histogram, llaminar2::RoutedExpertRowExecutionPolicy row_policy,
        llaminar2::MoEDecodeDescriptorPublication publication,
        int device_idx, void *stream, const int32_t *active_row_count_device)
{
    using Policy = llaminar2::RoutedExpertRowExecutionPolicy;
    using Projections = llaminar2::DeviceMoEProjectionSet;
    if (!logits || !runtime || reinterpret_cast<std::uintptr_t>(stream) <= 2u ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kMaxTopK || top_k > num_experts ||
        !publication.valid() || publication.expert_count != num_experts ||
        (write_legacy_outputs && (!legacy_indices || !legacy_weights)) ||
        (row_policy != Policy::ParticipantAssigned && row_policy != Policy::IntermediateShardLocal) ||
        (row_policy == Policy::IntermediateShardLocal) != (publication.projections == Projections::IntermediateShard))
        return false;
    if (cudaSetDevice(device_idx) != cudaSuccess) return false;
#define LLAMINAR_LAUNCH_ROUTE_DESCRIPTORS(HISTOGRAM, POLICY) \
    softmax_topk_decode_descriptors_kernel<HISTOGRAM, POLICY><<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>( \
        logits, static_cast<DeviceMoELayerRuntimeView *>(runtime), legacy_indices, legacy_weights, \
        num_experts, top_k, normalize_weights, write_legacy_outputs, active_row_count_device, publication)
    if (row_policy == Policy::IntermediateShardLocal)
    {
        if (update_runtime_histogram) LLAMINAR_LAUNCH_ROUTE_DESCRIPTORS(true, Policy::IntermediateShardLocal);
        else LLAMINAR_LAUNCH_ROUTE_DESCRIPTORS(false, Policy::IntermediateShardLocal);
    }
    else
    {
        if (update_runtime_histogram) LLAMINAR_LAUNCH_ROUTE_DESCRIPTORS(true, Policy::ParticipantAssigned);
        else LLAMINAR_LAUNCH_ROUTE_DESCRIPTORS(false, Policy::ParticipantAssigned);
    }
#undef LLAMINAR_LAUNCH_ROUTE_DESCRIPTORS
    return finishLaunch("cudaMoE_softmax_topk_decode_selected_descriptors");
}

/**
     * @brief Launch the minimal unique-owner CUDA decode router.
     *
     * The device kernel traps if the active runtime bank contains a replica;
     * callers may not recover by replaying the replica-aware graph.
     */
    bool cudaMoE_softmax_topk_decode_unique_owner_runtime(
        float *logits,
        void *runtime_layer,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts,
        int top_k,
        bool normalize_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
        int device_idx,
        void *stream,
        const int32_t *active_row_count_device)
    {
        if (!logits || !runtime_layer || !stream || num_experts <= 0 ||
            num_experts > kDeviceMoEMaxExperts || top_k <= 0 ||
            top_k > kMaxTopK || top_k > num_experts ||
            (write_legacy_outputs && (!legacy_indices || !legacy_weights)))
        {
            return false;
        }
        cudaSetDevice(device_idx);
        if (update_runtime_histogram)
        {
            softmax_topk_decode_independent_runtime_kernel<true>
                <<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
                    logits,
                    static_cast<DeviceMoELayerRuntimeView *>(runtime_layer),
                    legacy_indices,
                    legacy_weights,
                    num_experts,
                    top_k,
                    normalize_weights,
                    write_legacy_outputs, active_row_count_device);
        }
        else
        {
            softmax_topk_decode_independent_runtime_kernel<false>
                <<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
                    logits,
                    static_cast<DeviceMoELayerRuntimeView *>(runtime_layer),
                    legacy_indices,
                    legacy_weights,
                    num_experts,
                    top_k,
                    normalize_weights,
                    write_legacy_outputs, active_row_count_device);
        }
        return finishLaunch(
            "cudaMoE_softmax_topk_decode_unique_owner_runtime");
    }

    bool cudaMoE_decode_route_select_runtime(const int *expert_indices, const float *expert_weights,
                                             void *runtime_layer,
                                             float *legacy_indices, float *legacy_weights,
                                             int num_experts, int top_k, bool write_legacy_outputs,
                                             bool update_runtime_histogram, llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy, int device_idx, void *stream)
    {
        if (!runtime_layer || !expert_indices || !expert_weights || !stream ||
            num_experts <= 0 || num_experts > kDeviceMoEMaxExperts || top_k <= 0 || top_k > kMaxTopK)
            return false;
        cudaSetDevice(device_idx);
        decode_route_select_runtime_kernel<<<1, kMaxTopK, 0, static_cast<cudaStream_t>(stream)>>>(
            expert_indices, expert_weights, static_cast<DeviceMoELayerRuntimeView *>(runtime_layer),
            legacy_indices, legacy_weights, num_experts, top_k,
            write_legacy_outputs, update_runtime_histogram,
            row_execution_policy, /*absolute_position_ids=*/nullptr);
        return finishLaunch("cudaMoE_decode_route_select_runtime");
    }


    bool cudaMoE_float_to_int(const float *input, int *output, int count, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        float_to_int_kernel<<<blocksFor(count), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(input, output, count);
        return finishLaunch("cudaMoE_float_to_int");
    }

    bool cudaMoE_float_to_masked_int(
        const float *input,
        int *output,
        const uint8_t *expert_mask,
        int count,
        int num_experts,
        int device_idx,
        void *stream)
    {
        cudaSetDevice(device_idx);
        float_to_masked_int_kernel<<<blocksFor(count), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            input, output, expert_mask, count, num_experts);
        return finishLaunch("cudaMoE_float_to_masked_int");
    }

    bool cudaMoE_gather_tokens(const float *hidden, float *batch_buffer, const int *token_indices,
                               int num_tokens, int d_model, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        if (num_tokens <= 0 || d_model <= 0)
            return true;

        if ((d_model & 3) == 0)
        {
            // 2D grid: x covers d_model/4 float4 columns, y selects the token row.
            const int n4 = d_model >> 2;
            const dim3 grid = make_row_major_grid(
                static_cast<unsigned int>((n4 + kThreads - 1) / kThreads),
                num_tokens);
            gather_tokens_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
                hidden, batch_buffer, token_indices, num_tokens, d_model);
        }
        else
        {
            const int total_elements = num_tokens * d_model;
            gather_tokens_scalar_kernel<<<blocksFor(total_elements), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
                hidden, batch_buffer, token_indices, total_elements, d_model);
        }
        return finishLaunch("cudaMoE_gather_tokens");
    }

    bool cudaMoE_copy_token_row(const float *source, float *row_buffer,
                                int row_index, int row_width, int device_idx, void *stream)
    {
        if (!source || !row_buffer || row_index < 0 || row_width <= 0 || !stream)
            return false;
        cudaSetDevice(device_idx);
        const int vector_columns = (row_width + 3) >> 2;
        copy_token_row_kernel<<<blocksFor(vector_columns), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            source, row_buffer, row_index, row_width);
        return finishLaunch("cudaMoE_copy_token_row");
    }

    bool cudaMoE_scatter_add(float *output, const float *expert_output, const int *token_indices,
                             const float *weights, int num_tokens, int d_model, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        if (num_tokens <= 0 || d_model <= 0)
            return true;

        if ((d_model & 3) == 0)
        {
            // 2D grid: x covers d_model/4 float4 columns, y selects the token row.
            const int n4 = d_model >> 2;
            const dim3 grid = make_row_major_grid(
                static_cast<unsigned int>((n4 + kThreads - 1) / kThreads),
                num_tokens);
            scatter_add_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
                output, expert_output, token_indices, weights, num_tokens, d_model);
        }
        else
        {
            const int total_elements = num_tokens * d_model;
            scatter_add_scalar_kernel<<<blocksFor(total_elements), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
                output, expert_output, token_indices, weights, total_elements, d_model);
        }
        return finishLaunch("cudaMoE_scatter_add");
    }

    bool cudaMoE_write_token_row(float *destination, const float *row_buffer,
                                 int row_index, int row_width, int device_idx, void *stream)
    {
        if (!destination || !row_buffer || row_index < 0 || row_width <= 0 || !stream)
            return false;
        cudaSetDevice(device_idx);
        const int vector_columns = (row_width + 3) >> 2;
        write_token_row_kernel<<<blocksFor(vector_columns), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            destination, row_buffer, row_index, row_width);
        return finishLaunch("cudaMoE_write_token_row");
    }

    bool cudaMoE_shared_expert_gate(const float *input, const float *gate_inp, float *shared_output,
                                    int seq_len, int d_model, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        shared_expert_gate_kernel<<<seq_len, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            input, gate_inp, shared_output, seq_len, d_model);
        return finishLaunch("cudaMoE_shared_expert_gate");
    }

    bool cudaMoE_shared_expert_gate_effective_seq_len(
        const float *input, const float *gate_inp, float *shared_output,
        int seq_len, int d_model, const int *device_effective_seq_len,
        int device_idx, void *stream)
    {
        if (!input || !gate_inp || !shared_output || !device_effective_seq_len ||
            seq_len <= 0 || d_model <= 0 || !stream)
            return false;
        cudaSetDevice(device_idx);
        shared_expert_gate_kernel<<<seq_len, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            input, gate_inp, shared_output, seq_len, d_model, device_effective_seq_len);
        return finishLaunch("cudaMoE_shared_expert_gate_effective_seq_len");
    }

    bool cudaMoE_shared_expert_gate_add(
        const float *input, const float *gate_inp, float *shared_output,
        const float *routed_residual, float *combined_output,
        int seq_len, int d_model, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        shared_expert_gate_add_kernel<false><<<seq_len, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            input, gate_inp, shared_output, routed_residual, combined_output, seq_len, d_model);
        return finishLaunch("cudaMoE_shared_expert_gate_add");
    }

    bool cudaMoE_shared_expert_gate_add_effective_seq_len(
        const float *input, const float *gate_inp, float *shared_output,
        const float *routed_residual, float *combined_output,
        int seq_len, int d_model, const int *device_effective_seq_len,
        int device_idx, void *stream)
    {
        if (!input || !gate_inp || !shared_output || !routed_residual ||
            !combined_output || !device_effective_seq_len ||
            seq_len <= 0 || d_model <= 0 || !stream)
            return false;
        cudaSetDevice(device_idx);
        shared_expert_gate_add_kernel<false><<<seq_len, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            input, gate_inp, shared_output, routed_residual, combined_output,
            seq_len, d_model, device_effective_seq_len);
        return finishLaunch("cudaMoE_shared_expert_gate_add_effective_seq_len");
    }

    /** @brief Gate and combine owned columns, retaining the original full-row dot and padded-row mask. */
    bool cudaMoE_shared_expert_column_gate_add(
        const float *input, const float *gate_inp, float *shared_output,
        const float *routed_columns, float *combined_columns,
        int rows, int model_columns, int local_columns, const int *active_rows,
        int ordinal, void *stream)
    {
        if (!input || !gate_inp || !shared_output || !routed_columns || !combined_columns ||
            rows <= 0 || local_columns <= 0 || local_columns > model_columns || !stream)
            return false;
        if (cudaSetDevice(ordinal) != cudaSuccess || cudaGetLastError() != cudaSuccess) return false;
        shared_expert_gate_add_kernel<true><<<rows, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            input, gate_inp, shared_output, routed_columns, combined_columns,
            rows, model_columns, active_rows, local_columns);
        return finishLaunch("cudaMoE_shared_expert_column_gate_add");
    }

    bool cudaMoE_swiglu(float *gate, const float *up, int count, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        // One thread per float4 lane; ceil(count/4) threads cover both the vectorized
        // bulk and the (<4) scalar tail.
        swiglu_kernel<<<blocksFor((count + 3) >> 2), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(gate, up, count);
        return finishLaunch("cudaMoE_swiglu");
    }

    bool cudaMoE_weighted_add(float *output, const float *input, float weight, int count, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        weighted_add_kernel<<<blocksFor((count + 3) >> 2), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(output, input, weight, count);
        return finishLaunch("cudaMoE_weighted_add");
    }

} // extern "C"

namespace llaminar2::cuda
{
    /** @brief Reuse canonical prefill arithmetic; distribute rows, never K reductions. */
    bool routeOwnedRows(const MoERouterOwnedRowsLaunch &launch, void *stream,
                        const RouterQ8PrefillHardware &hardware)
    {
        if (!stream || !launch.valid() || launch.experts > kMaxExperts ||
            (launch.format == MoERouterPreparedFormat::BlockQ8 && !hardware.valid()) ||
            launch.top_k > kMaxTopK ||
            (launch.format != MoERouterPreparedFormat::FP32 &&
             launch.format != MoERouterPreparedFormat::BF16 &&
             launch.format != MoERouterPreparedFormat::FP16 &&
             launch.format != MoERouterPreparedFormat::BlockQ8))
            return false;
        const int rows = launch.partition.capacityFor(launch.capacity);
        const auto native_stream = static_cast<cudaStream_t>(stream);
        if (launch.format == MoERouterPreparedFormat::BlockQ8)
        {
            // Expert consumers borrow every live local input, even if this
            // participant owns no logits or top-k rows.
            const dim3 quant_grid(launch.width / 32, launch.capacity);
            if (rows > 0 && launch.q8_basis == RouterQ8Basis::SignedHadamard32)
            {
                if (!router_hadamard_q8::launch(launch.hidden, launch.hidden_q8, launch.hidden_scales,
                        launch.capacity, launch.width, stream, launch.live_rows)) return false;
            }
            else
            {
                grouped_hidden_quantize_blockwise_rows_kernel<<<quant_grid, 32, 0, native_stream>>>(
                    launch.hidden, launch.hidden_q8, launch.hidden_scales, launch.capacity,
                    launch.width, launch.live_rows);
                if (!finishLaunch("cudaMoE_owned_q8_hidden")) return false;
            }
            if (rows > 0)
            {
                if (!routerQ8Prefill({.hidden_codes = launch.hidden_q8,
                        .hidden_scales = launch.hidden_scales,
                        .gate_codes = static_cast<const int8_t *>(launch.gate),
                        .gate_scales = launch.gate_scales, .logits = launch.logits,
                        .capacity = launch.capacity, .width = launch.width,
                        .experts = launch.experts, .live_rows = launch.live_rows,
                        .partition = launch.partition, .hardware = hardware}, stream)) return false;
                if (launch.q8_basis == RouterQ8Basis::SignedHadamard32)
                {
                    grouped_hidden_quantize_blockwise_rows_kernel<<<quant_grid, 32, 0, native_stream>>>(
                        launch.hidden, launch.hidden_q8, launch.hidden_scales, launch.capacity,
                        launch.width, launch.live_rows);
                    if (!finishLaunch("cudaMoE_owned_q8_original_publication")) return false;
                }
            }
        }
        else if (rows > 0 && launch.format == MoERouterPreparedFormat::FP32)
        {
            // Keep the established prefill family even when an owner currently
            // has one row. Scalar decode is a different numerical contract.
            if (!dispatch_route_logits_geometry<true>(launch.hidden,
                    static_cast<const float *>(launch.gate), launch.logits,
                    launch.capacity, launch.width, launch.experts, native_stream,
                    selectCUDAMoERouterPrefillGeometry(
                        launch.capacity, launch.width, launch.experts),
                    launch.live_rows, launch.partition))
                return false;
        }
        else if (rows > 0 && launch.format == MoERouterPreparedFormat::FP16)
        {
            launch_fp16_router_rows<true>(launch.hidden, static_cast<const __half *>(launch.gate), launch.logits,
                launch.capacity, rows, launch.width, launch.experts, launch.live_rows, launch.partition, native_stream);
            if (!finishLaunch("cudaMoE_owned_fp16_logits")) return false;
        }
        else if (rows > 0)
        {
            route_logits_bf16_kernel<true><<<dim3(launch.experts, rows), kThreads, 0, native_stream>>>(
                launch.hidden, static_cast<const __nv_bfloat16 *>(launch.gate), launch.logits,
                launch.capacity, launch.width, launch.experts, launch.live_rows, launch.partition);
            if (!finishLaunch("cudaMoE_owned_bf16_logits")) return false;
        }
        softmax_topk_kernel<true><<<std::max(1, rows), kThreads, 0, native_stream>>>(
            launch.logits, nullptr, nullptr, launch.capacity, launch.experts,
            launch.top_k, launch.normalize, launch.live_rows, nullptr, launch.partition,
            launch.selected, launch.selected_bytes);
        return finishLaunch("cudaMoE_owned_topk");
    }

    bool publishRouterOwnedRows(const MoERouterRowPublication &publication, void *stream)
    {
        if (!stream || !publication.valid()) return false;
        const auto slots = std::max(std::size_t(publication.layout.capacity) * publication.layout.top_k,
            std::size_t(publication.layout.partition.participants()));
        if ((slots + 127) / 128 > std::size_t(INT_MAX)) return false;
        publish_router_owned_rows_kernel<<<unsigned((slots + 127) / 128), 128, 0,
            static_cast<cudaStream_t>(stream)>>>(publication, publication.layout.packetBytes());
        return finishLaunch("cudaMoE_publish_owned_rows");
    }
}

