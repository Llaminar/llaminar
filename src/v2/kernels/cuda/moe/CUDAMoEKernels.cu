/**
 * @file CUDAMoEKernels.cu
 * @brief CUDA launch bridges for MoE routing, grouping, and scatter/gather primitives.
 *
 * These kernels intentionally cover the non-GEMM MoE glue. Expert gate/up/down
 * projections continue to use the dedicated CUDA GEMM kernels. The launch
 * wrappers are C ABI functions consumed by `CUDAMoEKernel.cpp`, matching the
 * split used by the rest of the CUDA backend.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "kernels/cuda/gemm/CUDANativeVNNIDecodeCommon.cuh"
#include "utils/DebugEnv.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace
{
    constexpr int kThreads = 256;
    constexpr int kMaxExperts = 1024;
    constexpr int kDeviceMoEMaxExperts = 256;
    constexpr int kMaxTopK = 16;

    struct DeviceNativeVNNIMatrixDesc
    {
        const uint8_t *payload = nullptr;
        const void *scales = nullptr;
        const void *mins = nullptr;
        const void *emins = nullptr;
        int n = 0;
        int k = 0;
        uint32_t blocks_per_row = 0;
        uint8_t codebook_id = 0;
        uint8_t reserved[3] = {0, 0, 0};
    };

    __device__ __forceinline__ float silu(float x)
    {
        return x / (1.0f + expf(-x));
    }

    bool finishLaunch(const char *name)
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::fprintf(stderr, "%s launch failed: %s\n", name, cudaGetErrorString(err));
            return false;
        }
        return true;
    }

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

    __global__ void route_logits_bf16_kernel(
        const float *__restrict__ hidden,
        const __nv_bfloat16 *__restrict__ gate_weights,
        float *__restrict__ logits,
        int seq_len, int d_model, int num_experts)
    {
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
    template <int BM, int BN, int BK, int TM, int TN>
    __global__ void route_logits_tiled_kernel(
        const float *__restrict__ hidden,       // A: [seq_len, d_model] row-major
        const float *__restrict__ gate_weights, // B: [num_experts, d_model] row-major
        float *__restrict__ logits,             // C: [seq_len, num_experts] row-major
        int seq_len, int d_model, int num_experts)
    {
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

        // March across the K (d_model) dimension one BK-wide strip at a time.
        for (int kk = 0; kk < d_model; kk += BK)
        {
            // Cooperative load of the hidden tile into smem (K-major). Consecutive
            // threads read consecutive k → fully coalesced within each hidden row.
            for (int idx = threadIdx.x; idx < BM * BK; idx += blockDim.x)
            {
                const int m = idx / BK;
                const int k = idx % BK;
                const int gm = blockM + m;
                const int gk = kk + k;
                As[k * BM + m] = (gm < seq_len && gk < d_model)
                                     ? hidden[static_cast<size_t>(gm) * d_model + gk]
                                     : 0.0f;
            }
            // Cooperative load of the gate tile into smem (K-major), same coalescing.
            for (int idx = threadIdx.x; idx < BN * BK; idx += blockDim.x)
            {
                const int n = idx / BK;
                const int k = idx % BK;
                const int gn = blockN + n;
                const int gk = kk + k;
                Bs[k * BN + n] = (gn < num_experts && gk < d_model)
                                     ? gate_weights[static_cast<size_t>(gn) * d_model + gk]
                                     : 0.0f;
            }
            __syncthreads();

            // Multiply the staged strip: for each k, broadcast TM hidden values and
            // TN gate values from smem into registers and accumulate the outer product.
#pragma unroll
            for (int k = 0; k < BK; ++k)
            {
                float regM[TM];
                float regN[TN];
#pragma unroll
                for (int i = 0; i < TM; ++i)
                    regM[i] = As[k * BM + threadRow * TM + i];
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    regN[j] = Bs[k * BN + threadCol * TN + j];
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

    __global__ void softmax_topk_kernel(
        float *__restrict__ logits,
        int *__restrict__ expert_indices,
        float *__restrict__ expert_weights,
        int seq_len, int num_experts, int top_k,
        bool normalize_weights,
        const int *__restrict__ effective_seq_len_ptr)
    {
        const int token = blockIdx.x;
        if (token >= seq_len)
            return;

        int effective_seq_len = seq_len;
        if (effective_seq_len_ptr)
        {
            const int raw_effective = *effective_seq_len_ptr;
            effective_seq_len = raw_effective < 1 ? 1 : (raw_effective > seq_len ? seq_len : raw_effective);
        }
        if (token >= effective_seq_len)
        {
            if (threadIdx.x == 0)
            {
                for (int k = 0; k < top_k; ++k)
                {
                    const size_t out = static_cast<size_t>(token) * top_k + k;
                    expert_indices[out] = -1;
                    expert_weights[out] = 0.0f;
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
        float local_max = -INFINITY;
        for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
            local_max = fmaxf(local_max, logits[row_offset + expert]);
        reductions[threadIdx.x] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + stride]);
            __syncthreads();
        }
        const float max_value = reductions[0];

        float local_sum = 0.0f;
        for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
        {
            const float prob = expf(logits[row_offset + expert] - max_value);
            values[expert] = prob;
            local_sum += prob;
        }
        reductions[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                reductions[threadIdx.x] += reductions[threadIdx.x + stride];
            __syncthreads();
        }
        const float denom = reductions[0];
        for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
        {
            const float prob = denom > 0.0f ? values[expert] / denom : 0.0f;
            values[expert] = prob;
            logits[row_offset + expert] = prob;
        }
        __syncthreads();

        // Parallel top-k selection. The original implementation ran the entire
        // top_k * num_experts argmax scan on thread 0 with the other 255 lanes idle.
        // Here each selection round does a block-wide parallel argmax reduction over
        // the (still-unselected) experts, masks the winner, and repeats. Ties break
        // toward the lower expert index to match the original serial scan exactly.
        float topk_sum = 0.0f;
        for (int k = 0; k < top_k; ++k)
        {
            // Each thread finds the best (value,index) over its strided expert subset.
            float best_value = -1.0f;
            int best = 0;
            for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
            {
                const float value = values[expert];
                if (value > best_value)
                {
                    best_value = value;
                    best = expert;
                }
            }
            reductions[threadIdx.x] = best_value;
            red_idx[threadIdx.x] = best;
            __syncthreads();

            // Tree reduction to the global argmax for this round.
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
            {
                if (threadIdx.x < stride)
                {
                    const float other = reductions[threadIdx.x + stride];
                    const int other_idx = red_idx[threadIdx.x + stride];
                    const float cur = reductions[threadIdx.x];
                    const int cur_idx = red_idx[threadIdx.x];
                    if (other > cur || (other == cur && other_idx < cur_idx))
                    {
                        reductions[threadIdx.x] = other;
                        red_idx[threadIdx.x] = other_idx;
                    }
                }
                __syncthreads();
            }

            const int winner = red_idx[0];
            const float winner_value = reductions[0];
            topk_sum += winner_value;
            if (threadIdx.x == 0)
            {
                selected[k] = winner;
                selected_weights[k] = winner_value;
                values[winner] = -1.0f; // mask out for the next round
            }
            __syncthreads(); // ensure the mask is visible before the next round
        }

        // Thread 0 writes the final indices + (optionally normalized) weights.
        if (threadIdx.x == 0)
        {
            for (int k = 0; k < top_k; ++k)
            {
                const size_t out = static_cast<size_t>(token) * top_k + k;
                expert_indices[out] = selected[k];
                expert_weights[out] = normalize_weights && topk_sum > 0.0f
                                          ? selected_weights[k] / topk_sum
                                          : selected_weights[k];
            }
        }
    }

    __global__ void softmax_topk_decode_runtime_kernel(
        float *__restrict__ logits,
        int *__restrict__ runtime_expert_ids,
        float *__restrict__ runtime_weights,
        uint64_t *__restrict__ runtime_histogram,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts, int top_k,
        bool normalize_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram)
    {
        __shared__ float values[kMaxExperts];
        __shared__ float reductions[kThreads];
        __shared__ int selected[kMaxTopK];
        __shared__ float selected_weights[kMaxTopK];

        float local_max = -INFINITY;
        for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
            local_max = fmaxf(local_max, logits[expert]);
        reductions[threadIdx.x] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + stride]);
            __syncthreads();
        }
        const float max_value = reductions[0];

        float local_sum = 0.0f;
        for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
        {
            const float prob = expf(logits[expert] - max_value);
            values[expert] = prob;
            local_sum += prob;
        }
        reductions[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                reductions[threadIdx.x] += reductions[threadIdx.x + stride];
            __syncthreads();
        }
        const float denom = reductions[0];
        for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
        {
            const float prob = denom > 0.0f ? values[expert] / denom : 0.0f;
            values[expert] = prob;
            logits[expert] = prob;
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            float topk_sum = 0.0f;
            for (int k = 0; k < top_k; ++k)
            {
                int best = 0;
                float best_value = -1.0f;
                for (int expert = 0; expert < num_experts; ++expert)
                {
                    if (values[expert] > best_value)
                    {
                        best_value = values[expert];
                        best = expert;
                    }
                }
                selected[k] = best;
                selected_weights[k] = best_value;
                topk_sum += best_value;
                values[best] = -1.0f;
            }

            for (int k = 0; k < top_k; ++k)
            {
                const float weight = normalize_weights && topk_sum > 0.0f
                                         ? selected_weights[k] / topk_sum
                                         : selected_weights[k];
                runtime_expert_ids[k] = selected[k];
                runtime_weights[k] = weight;
                if (write_legacy_outputs)
                {
                    legacy_indices[k] = static_cast<float>(selected[k]);
                    legacy_weights[k] = weight;
                }
                if (update_runtime_histogram)
                    atomicAdd(reinterpret_cast<unsigned long long *>(&runtime_histogram[selected[k]]),
                              static_cast<unsigned long long>(1));
            }
        }
    }

    __global__ void decode_route_select_runtime_kernel(
        const int *__restrict__ expert_indices,
        const float *__restrict__ expert_weights,
        int *__restrict__ runtime_expert_ids,
        float *__restrict__ runtime_weights,
        uint64_t *__restrict__ runtime_histogram,
        float *legacy_indices,
        float *legacy_weights,
        int top_k,
        bool write_legacy_outputs,
        bool update_runtime_histogram)
    {
        const int k = threadIdx.x;
        if (k >= top_k)
            return;
        const int expert = expert_indices[k];
        const float weight = expert_weights[k];
        runtime_expert_ids[k] = expert;
        runtime_weights[k] = weight;
        if (write_legacy_outputs)
        {
            legacy_indices[k] = static_cast<float>(expert);
            legacy_weights[k] = weight;
        }
        if (update_runtime_histogram && expert >= 0 && expert < kDeviceMoEMaxExperts)
            atomicAdd(reinterpret_cast<unsigned long long *>(&runtime_histogram[expert]),
                      static_cast<unsigned long long>(1));
    }

    __global__ void int_to_float_kernel(const int *__restrict__ input, float *__restrict__ output, int count)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count)
            output[idx] = static_cast<float>(input[idx]);
    }

    __global__ void float_to_int_kernel(const float *__restrict__ input, int *__restrict__ output, int count)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count)
            output[idx] = static_cast<int>(input[idx]);
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
        const int token_slot = blockIdx.y;
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
        const int token_slot = blockIdx.y;
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

    __device__ __forceinline__ int clamp_effective_seq_len(
        int seq_len,
        const int *__restrict__ device_effective_seq_len)
    {
        if (!device_effective_seq_len)
            return seq_len;
        const int raw = *device_effective_seq_len;
        return raw < 0 ? 0 : (raw > seq_len ? seq_len : raw);
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
            float4 *out4 = reinterpret_cast<float4 *>(shared_output + row_offset);
            const int n4 = d_model >> 2;
            for (int i = threadIdx.x; i < n4; i += blockDim.x)
                out4[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int i = (n4 << 2) + threadIdx.x; i < d_model; i += blockDim.x)
                shared_output[row_offset + i] = 0.0f;
            return;
        }

        __shared__ float partial[kThreads];
        const float *x = input + row_offset;

        // d_model is always a multiple of 32 (enforced upstream), so it is also a
        // multiple of 4 → we can process the row as float4 to cut the load/store
        // instruction count 4× (the original scalar stride loop was MIO-bound at 65%
        // memory throughput). Both rows start at token*d_model*4 bytes, which is
        // 16-byte aligned for d_model multiple of 4.
        const int n4 = d_model >> 2;
        const float4 *x4 = reinterpret_cast<const float4 *>(x);
        const float4 *g4 = reinterpret_cast<const float4 *>(gate_inp);

        float dot = 0.0f;
        for (int i = threadIdx.x; i < n4; i += blockDim.x)
        {
            const float4 a = x4[i];
            const float4 b = g4[i];
            dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }

        const float gate = 1.0f / (1.0f + expf(-partial[0]));
        float4 *out4 = reinterpret_cast<float4 *>(shared_output + row_offset);
        for (int i = threadIdx.x; i < n4; i += blockDim.x)
        {
            float4 v = out4[i];
            v.x *= gate;
            v.y *= gate;
            v.z *= gate;
            v.w *= gate;
            out4[i] = v;
        }
        for (int i = (n4 << 2) + threadIdx.x; i < d_model; i += blockDim.x)
            shared_output[row_offset + i] *= gate;
    }

    __global__ void shared_expert_gate_add_kernel(
        const float *__restrict__ input,
        const float *__restrict__ gate_inp,
        float *__restrict__ shared_output,
        const float *__restrict__ routed_residual,
        float *__restrict__ combined_output,
        int seq_len, int d_model,
        const int *__restrict__ device_effective_seq_len = nullptr)
    {
        const int token = blockIdx.x;
        if (token >= seq_len)
            return;

        __shared__ float partial[kThreads];
        const size_t row_offset = static_cast<size_t>(token) * d_model;
        const int effective_seq_len = clamp_effective_seq_len(seq_len, device_effective_seq_len);
        if (token >= effective_seq_len)
        {
            float4 *shared4 = reinterpret_cast<float4 *>(shared_output + row_offset);
            float4 *out4 = reinterpret_cast<float4 *>(combined_output + row_offset);
            const int n4 = d_model >> 2;
            for (int i = threadIdx.x; i < n4; i += blockDim.x)
            {
                shared4[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                out4[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            for (int i = (n4 << 2) + threadIdx.x; i < d_model; i += blockDim.x)
            {
                shared_output[row_offset + i] = 0.0f;
                combined_output[row_offset + i] = 0.0f;
            }
            return;
        }
        const float *x = input + row_offset;

        const int n4 = d_model >> 2;
        const float4 *x4 = reinterpret_cast<const float4 *>(x);
        const float4 *g4 = reinterpret_cast<const float4 *>(gate_inp);

        float dot = 0.0f;
        for (int i = threadIdx.x; i < n4; i += blockDim.x)
        {
            const float4 a = x4[i];
            const float4 b = g4[i];
            dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        for (int i = (n4 << 2) + threadIdx.x; i < d_model; i += blockDim.x)
            dot += x[i] * gate_inp[i];

        partial[threadIdx.x] = dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }

        const float gate = 1.0f / (1.0f + expf(-partial[0]));
        float4 *shared4 = reinterpret_cast<float4 *>(shared_output + row_offset);
        const float4 *residual4 = reinterpret_cast<const float4 *>(routed_residual + row_offset);
        float4 *out4 = reinterpret_cast<float4 *>(combined_output + row_offset);
        for (int i = threadIdx.x; i < n4; i += blockDim.x)
        {
            const float4 s = shared4[i];
            const float4 r = residual4[i];
            const float4 gated = make_float4(gate * s.x, gate * s.y, gate * s.z, gate * s.w);
            shared4[i] = gated;
            out4[i] = make_float4(
                r.x + gated.x,
                r.y + gated.y,
                r.z + gated.z,
                r.w + gated.w);
        }
        for (int i = (n4 << 2) + threadIdx.x; i < d_model; i += blockDim.x)
        {
            const float gated = gate * shared_output[row_offset + i];
            shared_output[row_offset + i] = gated;
            combined_output[row_offset + i] = routed_residual[row_offset + i] + gated;
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

    __global__ void count_per_expert_kernel(
        const int *__restrict__ routing_indices,
        int *__restrict__ expert_counts,
        int total_slots, int num_experts)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_slots)
            return;
        const int expert = routing_indices[idx];
        if (expert >= 0 && expert < num_experts)
            atomicAdd(expert_counts + expert, 1);
    }

    __global__ void exclusive_scan_kernel(const int *__restrict__ expert_counts, int *__restrict__ expert_offsets, int num_experts)
    {
        // The prefix sum is inherently serial, but the original implementation ran it
        // on thread 0 reading/writing global memory — num_experts (256) dependent
        // global loads at ~400-cycle latency dominated the ~9µs runtime. Stage the
        // counts into shared memory with a coalesced strided load, run the serial scan
        // in smem (~20-cycle latency), then write the offsets back coalesced. Arithmetic
        // order is identical to the original, so results are bit-exact.
        __shared__ int s_counts[kMaxExperts];

        for (int i = threadIdx.x; i < num_experts; i += blockDim.x)
            s_counts[i] = expert_counts[i];
        __syncthreads();

        if (threadIdx.x == 0)
        {
            int running = 0;
            for (int expert = 0; expert < num_experts; ++expert)
            {
                const int c = s_counts[expert];
                s_counts[expert] = running; // in-place exclusive prefix
                running += c;
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < num_experts; i += blockDim.x)
            expert_offsets[i] = s_counts[i];
    }

    __global__ void scatter_tokens_kernel(
        const int *__restrict__ routing_indices,
        const float *__restrict__ routing_weights,
        int *__restrict__ write_heads,
        const int *__restrict__ expert_offsets,
        int *__restrict__ grouped_token_indices,
        float *__restrict__ grouped_weights,
        int total_slots, int top_k, int num_experts)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_slots)
            return;
        const int expert = routing_indices[idx];
        if (expert < 0 || expert >= num_experts)
            return;
        const int local = atomicAdd(write_heads + expert, 1);
        const int dest = expert_offsets[expert] + local;
        grouped_token_indices[dest] = idx / top_k;
        grouped_weights[dest] = routing_weights[idx];
    }

    __global__ void scatter_tokens_deterministic_kernel(
        const int *__restrict__ routing_indices,
        const float *__restrict__ routing_weights,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ expert_counts,
        int *__restrict__ grouped_token_indices,
        int *__restrict__ original_to_grouped,
        int *__restrict__ original_expert_ids,
        float *__restrict__ grouped_weights,
        int total_slots,
        int top_k,
        int num_experts)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_slots)
            return;

        const int expert = routing_indices[idx];
        if (expert < 0 || expert >= num_experts)
            return;

        int local = 0;
        for (int prev = 0; prev < idx; ++prev)
        {
            if (routing_indices[prev] == expert)
                ++local;
        }
        if (local >= expert_counts[expert])
            return;

        const int dest = expert_offsets[expert] + local;
        grouped_token_indices[dest] = idx / top_k;
        original_to_grouped[idx] = dest;
        original_expert_ids[idx] = expert;
        grouped_weights[dest] = routing_weights[idx];
    }

    __global__ void group_tokens_small_float_kernel(
        const float *__restrict__ routing_indices,
        const float *__restrict__ routing_weights,
        int *__restrict__ expert_counts,
        int *__restrict__ expert_offsets,
        int *__restrict__ grouped_token_indices,
        int *__restrict__ original_to_grouped,
        int *__restrict__ original_expert_ids,
        float *__restrict__ grouped_weights,
        int *__restrict__ active_expert_ids,
        int total_slots,
        int num_experts,
        int top_k,
        int max_active_experts)
    {
        if (threadIdx.x != 0 || blockIdx.x != 0)
            return;

        for (int expert = 0; expert < num_experts; ++expert)
        {
            expert_counts[expert] = 0;
            expert_offsets[expert] = 0;
        }
        for (int idx = 0; idx < total_slots; ++idx)
        {
            original_to_grouped[idx] = -1;
            original_expert_ids[idx] = -1;
        }
        for (int slot = 0; slot < max_active_experts; ++slot)
            active_expert_ids[slot] = -1;

        for (int idx = 0; idx < total_slots; ++idx)
        {
            const int expert = static_cast<int>(routing_indices[idx]);
            if (expert >= 0 && expert < num_experts)
                ++expert_counts[expert];
        }

        int running = 0;
        int active_count = 0;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            expert_offsets[expert] = running;
            const int count = expert_counts[expert];
            if (count > 0 && active_count < max_active_experts)
                active_expert_ids[active_count++] = expert;
            running += count;
        }

        for (int idx = 0; idx < total_slots; ++idx)
        {
            const int expert = static_cast<int>(routing_indices[idx]);
            if (expert < 0 || expert >= num_experts)
                continue;

            int local = 0;
            for (int prev = 0; prev < idx; ++prev)
            {
                if (static_cast<int>(routing_indices[prev]) == expert)
                    ++local;
            }

            const int dest = expert_offsets[expert] + local;
            grouped_token_indices[dest] = idx / top_k;
            original_to_grouped[idx] = dest;
            original_expert_ids[idx] = expert;
            grouped_weights[dest] = routing_weights[idx];
        }
    }

    __global__ void prepare_shared_expert_group_kernel(
        int *__restrict__ expert_offsets,
        int *__restrict__ expert_counts,
        int *__restrict__ grouped_token_indices,
        int *__restrict__ original_to_grouped,
        float *__restrict__ grouped_weights,
        int *__restrict__ active_expert_ids,
        int seq_len)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx == 0)
        {
            expert_offsets[0] = 0;
            expert_counts[0] = seq_len;
            active_expert_ids[0] = 0;
        }
        if (idx < seq_len)
        {
            grouped_token_indices[idx] = idx;
            original_to_grouped[idx] = idx;
            grouped_weights[idx] = 1.0f;
        }
    }

    __global__ void gather_expert_fixed_kernel(
        const float *__restrict__ hidden,
        float *__restrict__ batch_buffer,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ expert_counts,
        const int *__restrict__ grouped_token_indices,
        int expert_id,
        int max_tokens,
        int d_model)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = max_tokens * d_model;
        if (idx >= total)
            return;

        const int token_slot = idx / d_model;
        const int col = idx % d_model;
        const int count = expert_counts[expert_id];
        float value = 0.0f;
        if (token_slot < count)
        {
            const int grouped = expert_offsets[expert_id] + token_slot;
            const int token = grouped_token_indices[grouped];
            value = hidden[static_cast<size_t>(token) * d_model + col];
        }
        batch_buffer[idx] = value;
    }

    __global__ void scatter_expert_fixed_kernel(
        float *__restrict__ output,
        const float *__restrict__ expert_output,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ expert_counts,
        const int *__restrict__ grouped_token_indices,
        const float *__restrict__ grouped_weights,
        int expert_id,
        int max_tokens,
        int d_model)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = max_tokens * d_model;
        if (idx >= total)
            return;

        const int token_slot = idx / d_model;
        const int col = idx % d_model;
        const int count = expert_counts[expert_id];
        if (token_slot >= count)
            return;

        const int grouped = expert_offsets[expert_id] + token_slot;
        const int token = grouped_token_indices[grouped];
        const float weight = grouped_weights[grouped];
        atomicAdd(output + static_cast<size_t>(token) * d_model + col,
                  weight * expert_output[idx]);
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

        float abs_value = fabsf(hidden[col]);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_value = fmaxf(abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

        const float scale = (abs_value > 0.0f) ? (abs_value / 127.0f) : 1.0f;
        if (lane == 0)
            scales_A_blockwise[block_idx] = scale;

        const float q = hidden[col] / scale;
        A_int8[col] = static_cast<int8_t>(rintf(fminf(127.0f, fmaxf(-127.0f, q))));
    }

    __global__ void grouped_swiglu_quantize_blockwise_kernel(
        const float *const *__restrict__ gate_ptrs,
        const float *const *__restrict__ up_ptrs,
        int8_t *__restrict__ A_int8,
        float *__restrict__ scales_A_blockwise,
        int num_active,
        int K)
    {
        const int slot = blockIdx.x;
        if (slot >= num_active)
            return;

        constexpr int kBlockSize = 32;
        constexpr int kWarps = 8;
        const int lane = threadIdx.x & 31;
        const int warp_id = threadIdx.x >> 5;
        const int blocks_per_row = K / kBlockSize;
        const float *gate = gate_ptrs[slot];
        const float *up = up_ptrs[slot];
        int8_t *row_int8 = A_int8 + static_cast<size_t>(slot) * K;
        float *row_scales = scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;

        for (int block_idx = warp_id; block_idx < blocks_per_row; block_idx += kWarps)
        {
            const int col = block_idx * kBlockSize + lane;
            const float g = gate[col];
            const float value = (g / (1.0f + expf(-g))) * up[col];

            float abs_value = fabsf(value);
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1)
                abs_value = fmaxf(abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

            const float scale = (abs_value > 0.0f) ? (abs_value / 127.0f) : 1.0f;
            if (lane == 0)
                row_scales[block_idx] = scale;

            const float q = value / scale;
            row_int8[col] = static_cast<int8_t>(rintf(fminf(127.0f, fmaxf(-127.0f, q))));
        }
    }

    // Gather + blockwise-int8 quantize for the prefill expert pipeline.
    //
    // Each 32-lane warp owns exactly one 32-element quant block: it loads the block,
    // computes the absmax via warp shuffle, derives the per-block scale, and writes the
    // quantized int8 + scale. The original launch used block=32 (a single warp), which
    // caps occupancy at one warp per block (profiled ~21% occupancy). Packing
    // kWarpsPerQuantBlock warps into each CUDA block lets the scheduler run many warps
    // per SM with the same per-warp work, lifting occupancy ~8× with no extra arithmetic.
    static constexpr int kWarpsPerQuantBlock = 8; // 8 warps = 256-thread block
    __global__ void grouped_prefill_gather_quantize_blockwise_kernel(
        const float *__restrict__ hidden,
        int8_t *__restrict__ A_int8,
        float *__restrict__ scales_A_blockwise,
        const int *__restrict__ grouped_token_indices,
        int total_slots,
        int K)
    {
        const int slot = blockIdx.y;
        if (slot >= total_slots)
            return;

        // Map this thread to (warp, lane); each warp handles a distinct 32-col block.
        const int warp = threadIdx.x >> 5;  // 0..kWarpsPerQuantBlock-1
        const int lane = threadIdx.x & 31;  // 0..31
        const int block_idx = blockIdx.x * kWarpsPerQuantBlock + warp;
        const int col = block_idx * 32 + lane;
        const int blocks_per_row = (K + 31) / 32;
        if (block_idx >= blocks_per_row || col >= K)
            return;

        const int source_token = grouped_token_indices[slot];
        const float value = hidden[static_cast<size_t>(source_token) * K + col];

        // Warp-local absmax reduction (each warp owns its own 32-element block).
        float abs_value = fabsf(value);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_value = fmaxf(abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

        const float scale = (abs_value > 0.0f) ? (abs_value / 127.0f) : 1.0f;
        if (lane == 0)
            scales_A_blockwise[static_cast<size_t>(slot) * blocks_per_row + block_idx] = scale;

        const float q = value / scale;
        A_int8[static_cast<size_t>(slot) * K + col] =
            static_cast<int8_t>(rintf(fminf(127.0f, fmaxf(-127.0f, q))));
    }

    template <uint8_t CodebookId, int TileM>
    __device__ __forceinline__ void accumulate_prefill_dot_block(
        const int32_t (&packed_groups)[8],
        const uint16_t *__restrict__ scale_base,
        const uint16_t *__restrict__ min_base,
        const uint32_t *__restrict__ emin_base,
        const uint8_t *__restrict__ payload,
        size_t linear,
        const int32_t *__restrict__ a4_base,
        int a4_stride_i32,
        const float *__restrict__ scale_a_base,
        int scale_a_stride,
        int tokens_in_group,
        float (&acc)[TileM])
    {
#pragma unroll
        for (int m = 0; m < TileM; ++m)
        {
            if (m >= tokens_in_group)
                break;

            // Per-token activation row: a4_base/scale_a_base point at a staged shared-memory
            // tile (contiguous, stride = 8 int32 / 1 scale). The 8 int32 are read as 2x128-bit
            // vector loads to cut the LDS instruction count 4x (relieves the MIO pipe).
            // Requires a4_base + m*stride to be 16-byte aligned (s_a is __align__(16), stride=8).
            const int4 *a4v = reinterpret_cast<const int4 *>(a4_base + static_cast<size_t>(m) * a4_stride_i32);
            const int4 av0 = a4v[0];
            const int4 av1 = a4v[1];
            const int32_t a4[8] = {av0.x, av0.y, av0.z, av0.w, av1.x, av1.y, av1.z, av1.w};
            const float scale_a = scale_a_base[static_cast<size_t>(m) * scale_a_stride];

            if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_dual_scale)
            {
                int dot_lo = 0;
                int dot_hi = 0;
                int sum_lo = 0;
                int sum_hi = 0;
#pragma unroll
                for (int group = 0; group < 4; ++group)
                {
                    dot_lo = __dp4a(a4[group], packed_groups[group], dot_lo);
                    sum_lo += llaminar2::cuda_native_vnni::sum_packed_i8(a4[group]);
                }
#pragma unroll
                for (int group = 4; group < 8; ++group)
                {
                    dot_hi = __dp4a(a4[group], packed_groups[group], dot_hi);
                    sum_hi += llaminar2::cuda_native_vnni::sum_packed_i8(a4[group]);
                }

                const float scale_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(scale_base[linear]);
                const float scale_hi = min_base ? llaminar2::cuda_native_vnni::fp16_bits_to_float(min_base[linear]) : 0.0f;
                acc[m] += scale_a * (scale_lo * static_cast<float>(dot_lo) +
                                     scale_hi * static_cast<float>(dot_hi));

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_dual_scale_asym)
                {
                    const uint32_t emin = emin_base ? emin_base[linear] : 0u;
                    const float min_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin));
                    const float min_hi = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin >> 16));
                    acc[m] += scale_a * (min_lo * static_cast<float>(sum_lo) +
                                         min_hi * static_cast<float>(sum_hi));
                }

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_iq1_m)
                {
                    constexpr float kIQ1SDelta = 0.125f;
                    const uint8_t qh0 = payload[4];
                    const uint8_t qh1 = payload[5];
                    const int sg0 = llaminar2::cuda_native_vnni::sum_packed_i8(a4[0]) +
                                    llaminar2::cuda_native_vnni::sum_packed_i8(a4[1]);
                    const int sg1 = llaminar2::cuda_native_vnni::sum_packed_i8(a4[2]) +
                                    llaminar2::cuda_native_vnni::sum_packed_i8(a4[3]);
                    const int sg2 = llaminar2::cuda_native_vnni::sum_packed_i8(a4[4]) +
                                    llaminar2::cuda_native_vnni::sum_packed_i8(a4[5]);
                    const int sg3 = llaminar2::cuda_native_vnni::sum_packed_i8(a4[6]) +
                                    llaminar2::cuda_native_vnni::sum_packed_i8(a4[7]);
                    const float delta0 = (qh0 & 0x08) ? -kIQ1SDelta : kIQ1SDelta;
                    const float delta1 = (qh0 & 0x80) ? -kIQ1SDelta : kIQ1SDelta;
                    const float delta2 = (qh1 & 0x08) ? -kIQ1SDelta : kIQ1SDelta;
                    const float delta3 = (qh1 & 0x80) ? -kIQ1SDelta : kIQ1SDelta;
                    acc[m] += scale_a * ((delta0 * static_cast<float>(sg0) + delta1 * static_cast<float>(sg1)) * scale_lo +
                                         (delta2 * static_cast<float>(sg2) + delta3 * static_cast<float>(sg3)) * scale_hi);
                }
            }
            else
            {
                int dot = 0;
                int sum_a = 0;
#pragma unroll
                for (int group = 0; group < 8; ++group)
                {
                    dot = __dp4a(a4[group], packed_groups[group], dot);
                    sum_a += llaminar2::cuda_native_vnni::sum_packed_i8(a4[group]);
                }

                const float scale_b = llaminar2::cuda_native_vnni::fp16_bits_to_float(scale_base[linear]);
                acc[m] += scale_a * scale_b * static_cast<float>(dot);

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_asymmetric)
                {
                    const float min_b = min_base ? llaminar2::cuda_native_vnni::fp16_bits_to_float(min_base[linear]) : 0.0f;
                    acc[m] += scale_a * min_b * static_cast<float>(sum_a);
                }
            }
        }
    }

    template <uint8_t CodebookId, int kTileM, int kTileN>
    __global__ void grouped_native_vnni_gate_up_prefill_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeVNNIMatrixDesc *__restrict__ gate_descs,
        const DeviceNativeVNNIMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ expert_counts,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ active_expert_ids,
        int active_expert_slots,
        float *__restrict__ gate_output,
        float *__restrict__ up_output,
        int N,
        int K)
    {
        // kTileN columns per block (one per thread); kTileM tokens processed per block.
        // Larger kTileM amortizes the expensive IQ-codebook weight decode (done once per
        // (block_idx, n)) over more tokens and reduces redundant cross-group re-decode.
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int token_group = blockIdx.y;
        const int expert_id = (active_expert_slots > 0)
                                  ? active_expert_ids[blockIdx.z]
                                  : static_cast<int>(blockIdx.z);
        if (expert_id < 0)
            return;

        const int count = expert_counts[expert_id];
        const int first_token = token_group * kTileM;
        if (first_token >= count)
            return; // Whole-block uniform exit (token_group/expert from blockIdx) — barrier-safe.

        const int tokens_in_group = min(kTileM, count - first_token);
        const int first_slot = expert_offsets[expert_id] + first_token;
        const int blocks_per_row = K / 32;

        // Per-thread output column may be out of range when N is not a multiple of kTileN.
        // We must NOT early-return such threads: they still participate in the cooperative
        // shared-memory staging + __syncthreads below. Guard only the weight work / writes.
        const bool active = (n < N);

        const DeviceNativeVNNIMatrixDesc gate_desc = gate_descs[expert_id];
        const DeviceNativeVNNIMatrixDesc up_desc = up_descs[expert_id];
        if (gate_desc.codebook_id != CodebookId || up_desc.codebook_id != CodebookId)
            return;

        const uint8_t *gate_payload_base = gate_desc.payload;
        const uint16_t *gate_scale_base = static_cast<const uint16_t *>(gate_desc.scales);
        const uint16_t *gate_min_base = static_cast<const uint16_t *>(gate_desc.mins);
        const uint32_t *gate_emin_base = static_cast<const uint32_t *>(gate_desc.emins);
        const uint8_t *up_payload_base = up_desc.payload;
        const uint16_t *up_scale_base = static_cast<const uint16_t *>(up_desc.scales);
        const uint16_t *up_min_base = static_cast<const uint16_t *>(up_desc.mins);
        const uint32_t *up_emin_base = static_cast<const uint32_t *>(up_desc.emins);

        // Staged activation tile for the current K-block: kTileM tokens × 32 int8 (= 8 int32)
        // plus one blockwise scale per token. Staging once per block (cooperatively) removes
        // the ~kTileN-way redundant L1 loads that all N-threads previously issued.
        // 16-byte aligned so the per-token 8×int32 row can be read as 2× int4 (128-bit) loads.
        __shared__ __align__(16) int32_t s_a[kTileM * 8];
        __shared__ float s_scale[kTileM];

        float gate_acc[kTileM] = {};
        float up_acc[kTileM] = {};

        // Weight decode is hoisted above the activation-staging barrier: for IQ3 codebooks
        // the decode is a data-dependent payload-byte load followed by a grid-table lookup
        // (a long-scoreboard latency chain). Issuing it before the cooperative smem staging
        // + __syncthreads lets the load latency overlap the staging loads and barrier wait,
        // with no extra register double-buffer (which would cost occupancy).
        constexpr int kPayloadBytes =
            llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::payload_bytes;

#pragma unroll 1
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx)
        {
            const size_t linear = static_cast<size_t>(block_idx) * N + static_cast<size_t>(n);

            // Decode this block's weights first so the global-load + grid-lookup latency
            // overlaps the activation staging + barrier below.
            int32_t gate_groups[8];
            int32_t up_groups[8];
            const uint8_t *gate_payload = gate_payload_base + linear * kPayloadBytes;
            const uint8_t *up_payload = up_payload_base + linear * kPayloadBytes;
            if (active)
            {
                llaminar2::cuda_native_vnni::decode_groups_vec<CodebookId>(gate_payload, gate_groups);
                llaminar2::cuda_native_vnni::decode_groups_vec<CodebookId>(up_payload, up_groups);
            }

            // Cooperatively load the activation tile for this K-block into shared memory.
            const int a_count = tokens_in_group * 8;
            for (int idx = threadIdx.x; idx < a_count; idx += kTileN)
            {
                const int m = idx >> 3;
                const int g = idx & 7;
                const int slot = first_slot + m;
                s_a[idx] = reinterpret_cast<const int32_t *>(
                    A_int8 + static_cast<size_t>(slot) * K + block_idx * 32)[g];
            }
            for (int m = threadIdx.x; m < tokens_in_group; m += kTileN)
            {
                const int slot = first_slot + m;
                s_scale[m] = scales_A_blockwise[static_cast<size_t>(slot) * blocks_per_row + block_idx];
            }
            __syncthreads();

            if (active)
            {
                accumulate_prefill_dot_block<CodebookId, kTileM>(
                    gate_groups, gate_scale_base, gate_min_base, gate_emin_base, gate_payload,
                    linear, s_a, /*a4_stride_i32=*/8, s_scale, /*scale_a_stride=*/1,
                    tokens_in_group, gate_acc);
                accumulate_prefill_dot_block<CodebookId, kTileM>(
                    up_groups, up_scale_base, up_min_base, up_emin_base, up_payload,
                    linear, s_a, /*a4_stride_i32=*/8, s_scale, /*scale_a_stride=*/1,
                    tokens_in_group, up_acc);
            }

            // Barrier before the next iteration overwrites the staged tile.
            __syncthreads();
        }

        if (active)
        {
#pragma unroll
            for (int m = 0; m < kTileM; ++m)
            {
                if (m >= tokens_in_group)
                    break;
                const int slot = first_slot + m;
                gate_output[static_cast<size_t>(slot) * N + n] = gate_acc[m];
                up_output[static_cast<size_t>(slot) * N + n] = up_acc[m];
            }
        }
    }

    // Fused gate/up GEMM + SwiGLU + blockwise int8 quantization.
    //
    // This is the fused-epilogue variant of grouped_native_vnni_gate_up_prefill_kernel: after
    // computing gate_acc/up_acc for each token in the tile, it directly evaluates
    // silu(gate)*up and blockwise-quantizes the result to int8, writing the down-projection
    // input (swiglu_int8 + swiglu_scales) in place. This eliminates the FP32 gate/up global
    // round-trip and the separate grouped_prefill_swiglu_quantize_blockwise_kernel launch.
    //
    // Quantization layout: each warp (32 lanes) spans exactly one aligned 32-wide block of the
    // intermediate (N) dimension (kTileN=128 is a multiple of 32, and N % 32 == 0), so the
    // per-block absmax is a warp-shuffle reduction with no cross-warp synchronization.
    template <uint8_t CodebookId, int kTileM, int kTileN>
    __global__ void grouped_native_vnni_gate_up_swiglu_prefill_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeVNNIMatrixDesc *__restrict__ gate_descs,
        const DeviceNativeVNNIMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ expert_counts,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ active_expert_ids,
        int active_expert_slots,
        int8_t *__restrict__ swiglu_int8,
        float *__restrict__ swiglu_scales,
        int N,
        int K)
    {
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int token_group = blockIdx.y;
        const int expert_id = (active_expert_slots > 0)
                                  ? active_expert_ids[blockIdx.z]
                                  : static_cast<int>(blockIdx.z);
        if (expert_id < 0)
            return;

        const int count = expert_counts[expert_id];
        const int first_token = token_group * kTileM;
        if (first_token >= count)
            return; // Whole-block uniform exit — barrier-safe.

        const int tokens_in_group = min(kTileM, count - first_token);
        const int first_slot = expert_offsets[expert_id] + first_token;
        const int blocks_per_row = K / 32;

        // Threads with n >= N still participate in cooperative staging + __syncthreads.
        const bool active = (n < N);

        const DeviceNativeVNNIMatrixDesc gate_desc = gate_descs[expert_id];
        const DeviceNativeVNNIMatrixDesc up_desc = up_descs[expert_id];
        if (gate_desc.codebook_id != CodebookId || up_desc.codebook_id != CodebookId)
            return;

        const uint8_t *gate_payload_base = gate_desc.payload;
        const uint16_t *gate_scale_base = static_cast<const uint16_t *>(gate_desc.scales);
        const uint16_t *gate_min_base = static_cast<const uint16_t *>(gate_desc.mins);
        const uint32_t *gate_emin_base = static_cast<const uint32_t *>(gate_desc.emins);
        const uint8_t *up_payload_base = up_desc.payload;
        const uint16_t *up_scale_base = static_cast<const uint16_t *>(up_desc.scales);
        const uint16_t *up_min_base = static_cast<const uint16_t *>(up_desc.mins);
        const uint32_t *up_emin_base = static_cast<const uint32_t *>(up_desc.emins);

        __shared__ __align__(16) int32_t s_a[kTileM * 8];
        __shared__ float s_scale[kTileM];

        float gate_acc[kTileM] = {};
        float up_acc[kTileM] = {};

        constexpr int kPayloadBytes =
            llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::payload_bytes;

#pragma unroll 1
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx)
        {
            const size_t linear = static_cast<size_t>(block_idx) * N + static_cast<size_t>(n);

            // Hoisted weight decode (overlaps activation staging + barrier latency).
            int32_t gate_groups[8];
            int32_t up_groups[8];
            const uint8_t *gate_payload = gate_payload_base + linear * kPayloadBytes;
            const uint8_t *up_payload = up_payload_base + linear * kPayloadBytes;
            if (active)
            {
                llaminar2::cuda_native_vnni::decode_groups_vec<CodebookId>(gate_payload, gate_groups);
                llaminar2::cuda_native_vnni::decode_groups_vec<CodebookId>(up_payload, up_groups);
            }

            // Cooperatively stage the activation tile for this K-block into shared memory.
            const int a_count = tokens_in_group * 8;
            for (int idx = threadIdx.x; idx < a_count; idx += kTileN)
            {
                const int m = idx >> 3;
                const int g = idx & 7;
                const int slot = first_slot + m;
                s_a[idx] = reinterpret_cast<const int32_t *>(
                    A_int8 + static_cast<size_t>(slot) * K + block_idx * 32)[g];
            }
            for (int m = threadIdx.x; m < tokens_in_group; m += kTileN)
            {
                const int slot = first_slot + m;
                s_scale[m] = scales_A_blockwise[static_cast<size_t>(slot) * blocks_per_row + block_idx];
            }
            __syncthreads();

            if (active)
            {
                accumulate_prefill_dot_block<CodebookId, kTileM>(
                    gate_groups, gate_scale_base, gate_min_base, gate_emin_base, gate_payload,
                    linear, s_a, /*a4_stride_i32=*/8, s_scale, /*scale_a_stride=*/1,
                    tokens_in_group, gate_acc);
                accumulate_prefill_dot_block<CodebookId, kTileM>(
                    up_groups, up_scale_base, up_min_base, up_emin_base, up_payload,
                    linear, s_a, /*a4_stride_i32=*/8, s_scale, /*scale_a_stride=*/1,
                    tokens_in_group, up_acc);
            }

            __syncthreads();
        }

        // Fused SwiGLU + blockwise int8 quant epilogue. N == intermediate here.
        const int lane = threadIdx.x & 31;
        const int blocks_per_row_out = N / 32;
        const int quant_block = n >> 5; // aligned 32-wide block index along intermediate
#pragma unroll
        for (int m = 0; m < kTileM; ++m)
        {
            if (m >= tokens_in_group)
                break; // tokens_in_group is block-uniform → all lanes break together (shfl-safe).

            const int slot = first_slot + m;
            // Inactive lanes (n >= N) contribute 0 to the warp absmax and skip the write.
            const float value = active ? (silu(gate_acc[m]) * up_acc[m]) : 0.0f;

            float abs_value = fabsf(value);
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1)
                abs_value = fmaxf(abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

            const float scale = (abs_value > 0.0f) ? (abs_value / 127.0f) : 1.0f;
            if (active)
            {
                if (lane == 0)
                    swiglu_scales[static_cast<size_t>(slot) * blocks_per_row_out + quant_block] = scale;
                const float q = value / scale;
                swiglu_int8[static_cast<size_t>(slot) * N + n] =
                    static_cast<int8_t>(rintf(fminf(127.0f, fmaxf(-127.0f, q))));
            }
        }
    }

    __global__ void grouped_prefill_swiglu_quantize_blockwise_kernel(
        const float *__restrict__ gate,
        const float *__restrict__ up,
        int8_t *__restrict__ A_int8,
        float *__restrict__ scales_A_blockwise,
        int total_slots,
        int K)
    {
        const int block_idx = blockIdx.x;
        const int slot = blockIdx.y;
        const int lane = threadIdx.x;
        const int col = block_idx * 32 + lane;
        if (slot >= total_slots || lane >= 32 || col >= K)
            return;

        const size_t idx = static_cast<size_t>(slot) * K + col;
        const float g = gate[idx];
        const float value = silu(g) * up[idx];

        float abs_value = fabsf(value);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_value = fmaxf(abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

        const float scale = (abs_value > 0.0f) ? (abs_value / 127.0f) : 1.0f;
        const int blocks_per_row = (K + 31) / 32;
        if (lane == 0)
            scales_A_blockwise[static_cast<size_t>(slot) * blocks_per_row + block_idx] = scale;

        const float q = value / scale;
        A_int8[idx] = static_cast<int8_t>(rintf(fminf(127.0f, fmaxf(-127.0f, q))));
    }

    template <uint8_t CodebookId, int kTileM, int kTileN>
    __global__ void grouped_native_vnni_down_prefill_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeVNNIMatrixDesc *__restrict__ descs,
        const int *__restrict__ expert_counts,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ active_expert_ids,
        int active_expert_slots,
        float *__restrict__ output,
        int N,
        int K)
    {
        // kTileM tokens processed per block; larger values amortize weight decode (see gate_up).
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int token_group = blockIdx.y;
        const int expert_id = (active_expert_slots > 0)
                                  ? active_expert_ids[blockIdx.z]
                                  : static_cast<int>(blockIdx.z);
        if (expert_id < 0)
            return;

        const int count = expert_counts[expert_id];
        const int first_token = token_group * kTileM;
        if (first_token >= count)
            return; // Whole-block uniform exit — barrier-safe.

        const int tokens_in_group = min(kTileM, count - first_token);
        const int first_slot = expert_offsets[expert_id] + first_token;
        const int blocks_per_row = K / 32;

        // Per-thread output column may exceed N; such threads must still participate in the
        // cooperative staging + __syncthreads. Guard only weight decode / output writes.
        const bool active = (n < N);

        const DeviceNativeVNNIMatrixDesc desc = descs[expert_id];
        if (desc.codebook_id != CodebookId)
            return;

        const uint8_t *payload_base = desc.payload;
        const uint16_t *scale_base = static_cast<const uint16_t *>(desc.scales);
        const uint16_t *min_base = static_cast<const uint16_t *>(desc.mins);
        const uint32_t *emin_base = static_cast<const uint32_t *>(desc.emins);

        // Staged activation tile (see gate_up kernel) to remove redundant L1 loads.
        // 16-byte aligned for vectorized 2× int4 (128-bit) per-token reads.
        __shared__ __align__(16) int32_t s_a[kTileM * 8];
        __shared__ float s_scale[kTileM];

        float acc[kTileM] = {};

#pragma unroll 1
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx)
        {
            // Cooperatively stage the activation tile for this K-block into shared memory.
            const int a_count = tokens_in_group * 8;
            for (int idx = threadIdx.x; idx < a_count; idx += kTileN)
            {
                const int m = idx >> 3;
                const int g = idx & 7;
                const int slot = first_slot + m;
                s_a[idx] = reinterpret_cast<const int32_t *>(
                    A_int8 + static_cast<size_t>(slot) * K + block_idx * 32)[g];
            }
            for (int m = threadIdx.x; m < tokens_in_group; m += kTileN)
            {
                const int slot = first_slot + m;
                s_scale[m] = scales_A_blockwise[static_cast<size_t>(slot) * blocks_per_row + block_idx];
            }
            __syncthreads();

            if (active)
            {
                const size_t linear = static_cast<size_t>(block_idx) * N + static_cast<size_t>(n);
                int32_t packed_groups[8];
                const uint8_t *payload = payload_base +
                    linear * llaminar2::cuda_native_vnni::payload_bytes_for_codebook<CodebookId>();
                llaminar2::cuda_native_vnni::decode_groups_vec<CodebookId>(payload, packed_groups);
                accumulate_prefill_dot_block<CodebookId, kTileM>(
                    packed_groups, scale_base, min_base, emin_base, payload,
                    linear, s_a, /*a4_stride_i32=*/8, s_scale, /*scale_a_stride=*/1,
                    tokens_in_group, acc);
            }

            __syncthreads();
        }

        if (active)
        {
#pragma unroll
            for (int m = 0; m < kTileM; ++m)
            {
                if (m >= tokens_in_group)
                    break;
                const int slot = first_slot + m;
                output[static_cast<size_t>(slot) * N + n] = acc[m];
            }
        }
    }

    __global__ void grouped_prefill_scatter_weighted_kernel(
        float *__restrict__ output,
        const float *__restrict__ expert_output,
        const int *__restrict__ grouped_token_indices,
        const float *__restrict__ grouped_weights,
        int total_slots,
        int d_model)
    {
        constexpr int kTileN = 64;
        const int col = blockIdx.x * kTileN + threadIdx.x;
        const int slot = blockIdx.y;
        if (slot >= total_slots || col >= d_model)
            return;

        const int token = grouped_token_indices[slot];
        const float weight = grouped_weights[slot];
        const float value = expert_output[static_cast<size_t>(slot) * d_model + col];
        atomicAdd(output + static_cast<size_t>(token) * d_model + col, weight * value);
    }

    __global__ void grouped_prefill_scatter_weighted_ordered_kernel(
        float *__restrict__ output,
        const float *__restrict__ expert_output,
        const int *__restrict__ original_to_grouped,
        const float *__restrict__ grouped_weights,
        int seq_len,
        int top_k,
        int d_model)
    {
        constexpr int kTileN = 64;
        const int col = blockIdx.x * kTileN + threadIdx.x;
        const int token = blockIdx.y;
        if (token >= seq_len || col >= d_model)
            return;

        float sum = 0.0f;
#pragma unroll 1
        for (int k = 0; k < top_k; ++k)
        {
            const int original_slot = token * top_k + k;
            const int grouped_slot = original_to_grouped[original_slot];
            if (grouped_slot < 0)
                continue;
            const float weight = grouped_weights[grouped_slot];
            const float value = expert_output[static_cast<size_t>(grouped_slot) * d_model + col];
            sum += weight * value;
        }
        output[static_cast<size_t>(token) * d_model + col] = sum;
    }

    /**
     * @brief Compute a partial dot product over a K-block subrange [b_start, b_end).
     *
     * This is the split-K building block: each caller reduces only a slice of the
     * K dimension so that multiple thread blocks can cooperate on a single output
     * column. Passing [0, K/32) reproduces the full reduction.
     *
     * @param desc                 Native-VNNI weight descriptor (column n of B).
     * @param n                    Output column index into the weight matrix.
     * @param A_int8               Quantized activation row (int8, K elements).
     * @param scales_A_blockwise   Per-K-block activation scales.
     * @param N                    Weight matrix column count (stride for payload).
     * @param K                    Reduction dimension length.
     * @param b_start              First K-block (inclusive) this call reduces.
     * @param b_end                Last K-block (exclusive) this call reduces.
     * @return Partial accumulated dot product over the requested K-block range.
     */
    template <uint8_t CodebookId>
    __device__ __forceinline__ float native_vnni_dot_desc_range(
        const DeviceNativeVNNIMatrixDesc &desc,
        int n,
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        int N,
        int K,
        int b_start,
        int b_end)
    {
        const int blocks_per_row = K / 32;
        const uint8_t *payload_base = desc.payload;
        const uint16_t *scale_base = static_cast<const uint16_t *>(desc.scales);
        const uint16_t *min_base = static_cast<const uint16_t *>(desc.mins);
        const uint32_t *emin_base = static_cast<const uint32_t *>(desc.emins);
        float acc = 0.0f;

        // Clamp the requested range to the valid K-block span.
        if (b_start < 0)
            b_start = 0;
        if (b_end > blocks_per_row)
            b_end = blocks_per_row;

#pragma unroll 1
        for (int block_idx = b_start; block_idx < b_end; ++block_idx)
        {
            const int32_t *a4 = reinterpret_cast<const int32_t *>(A_int8 + block_idx * 32);
            const size_t linear = static_cast<size_t>(block_idx) * N + static_cast<size_t>(n);
            const uint8_t *payload = payload_base +
                                     linear * llaminar2::cuda_native_vnni::payload_bytes_for_codebook<CodebookId>();

            int32_t packed_groups[8];
            llaminar2::cuda_native_vnni::decode_groups_vec<CodebookId>(payload, packed_groups);

            const float scale_a = scales_A_blockwise[block_idx];
            if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_dual_scale)
            {
                int dot_lo = 0;
                int dot_hi = 0;
                int sum_lo = 0;
                int sum_hi = 0;
#pragma unroll
                for (int group = 0; group < 4; ++group)
                {
                    dot_lo = __dp4a(a4[group], packed_groups[group], dot_lo);
                    dot_hi = __dp4a(a4[group + 4], packed_groups[group + 4], dot_hi);
                    sum_lo += llaminar2::cuda_native_vnni::sum_packed_i8(a4[group]);
                    sum_hi += llaminar2::cuda_native_vnni::sum_packed_i8(a4[group + 4]);
                }

                const float scale_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(scale_base[linear]);
                const float scale_hi = min_base ? llaminar2::cuda_native_vnni::fp16_bits_to_float(min_base[linear]) : 0.0f;
                acc += scale_a * (scale_lo * static_cast<float>(dot_lo) +
                                  scale_hi * static_cast<float>(dot_hi));

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_dual_scale_asym)
                {
                    const uint32_t emin = emin_base ? emin_base[linear] : 0u;
                    const float min_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin));
                    const float min_hi = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin >> 16));
                    acc += scale_a * (min_lo * static_cast<float>(sum_lo) +
                                      min_hi * static_cast<float>(sum_hi));
                }

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_iq1_m)
                {
                    constexpr float kIQ1SDelta = 0.125f;
                    const uint8_t qh0 = payload[4];
                    const uint8_t qh1 = payload[5];
                    const int sg0 = llaminar2::cuda_native_vnni::sum_packed_i8(a4[0]) +
                                    llaminar2::cuda_native_vnni::sum_packed_i8(a4[1]);
                    const int sg1 = llaminar2::cuda_native_vnni::sum_packed_i8(a4[2]) +
                                    llaminar2::cuda_native_vnni::sum_packed_i8(a4[3]);
                    const int sg2 = llaminar2::cuda_native_vnni::sum_packed_i8(a4[4]) +
                                    llaminar2::cuda_native_vnni::sum_packed_i8(a4[5]);
                    const int sg3 = llaminar2::cuda_native_vnni::sum_packed_i8(a4[6]) +
                                    llaminar2::cuda_native_vnni::sum_packed_i8(a4[7]);
                    const float delta0 = (qh0 & 0x08) ? -kIQ1SDelta : kIQ1SDelta;
                    const float delta1 = (qh0 & 0x80) ? -kIQ1SDelta : kIQ1SDelta;
                    const float delta2 = (qh1 & 0x08) ? -kIQ1SDelta : kIQ1SDelta;
                    const float delta3 = (qh1 & 0x80) ? -kIQ1SDelta : kIQ1SDelta;
                    acc += scale_a * ((delta0 * static_cast<float>(sg0) + delta1 * static_cast<float>(sg1)) * scale_lo +
                                      (delta2 * static_cast<float>(sg2) + delta3 * static_cast<float>(sg3)) * scale_hi);
                }
            }
            else
            {
                int dot = 0;
                int sum_a = 0;
#pragma unroll
                for (int group = 0; group < 8; ++group)
                {
                    dot = __dp4a(a4[group], packed_groups[group], dot);
                    sum_a += llaminar2::cuda_native_vnni::sum_packed_i8(a4[group]);
                }

                const float scale_b = llaminar2::cuda_native_vnni::fp16_bits_to_float(scale_base[linear]);
                acc += scale_a * scale_b * static_cast<float>(dot);

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::is_asymmetric)
                {
                    const float min_b = min_base ? llaminar2::cuda_native_vnni::fp16_bits_to_float(min_base[linear]) : 0.0f;
                    acc += scale_a * min_b * static_cast<float>(sum_a);
                }
            }
        }

        return acc;
    }

    /**
     * @brief Full-K dot product wrapper (reduces all K-blocks). Preserves the
     *        original single-shot reduction semantics for non-split-K callers.
     */
    template <uint8_t CodebookId>
    __device__ __forceinline__ float native_vnni_dot_desc(
        const DeviceNativeVNNIMatrixDesc &desc,
        int n,
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        int N,
        int K)
    {
        return native_vnni_dot_desc_range<CodebookId>(
            desc, n, A_int8, scales_A_blockwise, N, K, 0, K / 32);
    }

    /**
     * @brief Split-K scatter kernel for grouped gate/up decode projection.
     *
     * Each block reduces one K-partition of one output column for one expert
     * slot, writing its partial to a [slot][k_part][N] scratch buffer. A separate
     * reduce kernel sums the partials. This raises occupancy versus the single
     * full-K kernel (which launches only num_active * ceil(N/64) blocks) by
     * multiplying the block count by k_partitions, exposing far more warps to
     * hide global-memory latency on the weight payload loads.
     *
     * Grid:  ((N + 63)/64, k_partitions, num_active)   Block: (64)
     */
    template <uint8_t CodebookId>
    __global__ void grouped_native_vnni_gate_up_kpart_decode_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeVNNIMatrixDesc *__restrict__ gate_descs,
        const DeviceNativeVNNIMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ expert_ids,
        float *__restrict__ gate_partials,
        float *__restrict__ up_partials,
        int num_active,
        int N,
        int K,
        int num_experts,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int k_part = blockIdx.y;
        const int slot = blockIdx.z;
        if (slot >= num_active || k_part >= k_partitions || n >= N)
            return;

        // Linear index into the [num_active][k_partitions][N] partials buffer.
        const size_t partial_index =
            (static_cast<size_t>(slot) * static_cast<size_t>(k_partitions) +
             static_cast<size_t>(k_part)) *
                static_cast<size_t>(N) +
            static_cast<size_t>(n);

        const int expert_id = expert_ids[slot];
        if (expert_id < 0)
        {
            gate_partials[partial_index] = 0.0f;
            up_partials[partial_index] = 0.0f;
            return;
        }
        assert(expert_id < num_experts);
        if (expert_id >= num_experts)
        {
            gate_partials[partial_index] = 0.0f;
            up_partials[partial_index] = 0.0f;
            return;
        }

        // Evenly split the K-blocks across the k_partitions; this block owns
        // [b_start, b_end).
        const int blocks_per_row = K / 32;
        const int blocks_per_part = (blocks_per_row + k_partitions - 1) / k_partitions;
        const int b_start = k_part * blocks_per_part;
        int b_end = b_start + blocks_per_part;
        if (b_end > blocks_per_row)
            b_end = blocks_per_row;

        if (b_start >= b_end)
        {
            gate_partials[partial_index] = 0.0f;
            up_partials[partial_index] = 0.0f;
            return;
        }

        const DeviceNativeVNNIMatrixDesc gate_desc = gate_descs[expert_id];
        const DeviceNativeVNNIMatrixDesc up_desc = up_descs[expert_id];
        gate_partials[partial_index] = native_vnni_dot_desc_range<CodebookId>(
            gate_desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        up_partials[partial_index] = native_vnni_dot_desc_range<CodebookId>(
            up_desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
    }

    /**
     * @brief Reduce K-partition partials into the final grouped gate/up outputs.
     *
     * Sums the k_partitions partial contributions for each (slot, n) produced by
     * grouped_native_vnni_gate_up_kpart_decode_kernel and writes the result to
     * the per-slot gate/up output buffers.
     *
     * Grid: ((N + 63)/64, num_active)   Block: (64)
     */
    __global__ void grouped_native_vnni_gate_up_kpart_reduce_kernel(
        const float *__restrict__ gate_partials,
        const float *__restrict__ up_partials,
        float *const *__restrict__ gate_outputs,
        float *const *__restrict__ up_outputs,
        int num_active,
        int N,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int slot = blockIdx.y;
        if (slot >= num_active || n >= N)
            return;

        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        const size_t slot_base =
            static_cast<size_t>(slot) * static_cast<size_t>(k_partitions) * static_cast<size_t>(N);
        for (int k_part = 0; k_part < k_partitions; ++k_part)
        {
            const size_t idx =
                slot_base + static_cast<size_t>(k_part) * static_cast<size_t>(N) + static_cast<size_t>(n);
            gate_sum += gate_partials[idx];
            up_sum += up_partials[idx];
        }

        gate_outputs[slot][n] = gate_sum;
        up_outputs[slot][n] = up_sum;
    }

    /**
     * @brief Split-K gate/up projection for tiny grouped verifier-prefill rows.
     *
     * The ordinary grouped-prefill gate/up kernel launches one full-K CTA for
     * each (expert, M-tile, N-tile).  For MTP verifier batches M is only 2..4,
     * so that leaves the GPU under-filled even though each CTA has a long
     * quantized-weight latency chain.  This variant mirrors the single-row
     * decode split-K strategy: each CTA owns one K partition and writes partial
     * gate/up sums into [grouped_slot][k_partition][N].  A second tiny reduce
     * kernel sums partitions back into the normal grouped-prefill FP32
     * gate/up scratch buffers.
     */
    template <uint8_t CodebookId, int kTileM>
    __global__ void grouped_native_vnni_gate_up_prefill_kpart_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeVNNIMatrixDesc *__restrict__ gate_descs,
        const DeviceNativeVNNIMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ expert_counts,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ active_expert_ids,
        int active_expert_slots,
        float *__restrict__ gate_partials,
        float *__restrict__ up_partials,
        int N,
        int K,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int token_group = blockIdx.y;
        const int expert_grid_index = blockIdx.z / k_partitions;
        const int k_part = blockIdx.z - expert_grid_index * k_partitions;
        if (n >= N || k_part >= k_partitions)
            return;

        const int expert_id = (active_expert_slots > 0)
                                  ? active_expert_ids[expert_grid_index]
                                  : expert_grid_index;
        if (expert_id < 0)
            return;

        const int count = expert_counts[expert_id];
        const int first_token = token_group * kTileM;
        if (first_token >= count)
            return;

        const int tokens_in_group = min(kTileM, count - first_token);
        const int first_slot = expert_offsets[expert_id] + first_token;
        const int blocks_per_row = K / 32;
        const int blocks_per_part = (blocks_per_row + k_partitions - 1) / k_partitions;
        const int b_start = k_part * blocks_per_part;
        int b_end = b_start + blocks_per_part;
        if (b_end > blocks_per_row)
            b_end = blocks_per_row;

        const DeviceNativeVNNIMatrixDesc gate_desc = gate_descs[expert_id];
        const DeviceNativeVNNIMatrixDesc up_desc = up_descs[expert_id];
        if (gate_desc.codebook_id != CodebookId || up_desc.codebook_id != CodebookId)
            return;

#pragma unroll
        for (int m = 0; m < kTileM; ++m)
        {
            if (m >= tokens_in_group)
                break;
            const int slot = first_slot + m;
            const size_t partial_index =
                (static_cast<size_t>(slot) * static_cast<size_t>(k_partitions) +
                 static_cast<size_t>(k_part)) *
                    static_cast<size_t>(N) +
                static_cast<size_t>(n);
            if (b_start >= b_end)
            {
                gate_partials[partial_index] = 0.0f;
                up_partials[partial_index] = 0.0f;
                continue;
            }
            const int8_t *slot_A = A_int8 + static_cast<size_t>(slot) * K;
            const float *slot_scales =
                scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;
            gate_partials[partial_index] = native_vnni_dot_desc_range<CodebookId>(
                gate_desc, n, slot_A, slot_scales, N, K, b_start, b_end);
            up_partials[partial_index] = native_vnni_dot_desc_range<CodebookId>(
                up_desc, n, slot_A, slot_scales, N, K, b_start, b_end);
        }
    }

    /**
     * @brief Reduce split-K gate/up partials and quantize SwiGLU in one pass.
     *
     * Each lane sums k-partitions in the same order used by the former
     * reduce-then-quantize sequence, computes the same SwiGLU value, and writes
     * the same blockwise INT8 row layout for the down-projection kernel.  The
     * benefit is architectural: verifier graphs avoid a launch and a global
     * FP32 gate/up scratch round trip.
     */
    __global__ void grouped_native_vnni_gate_up_prefill_kpart_reduce_swiglu_kernel(
        const float *__restrict__ gate_partials,
        const float *__restrict__ up_partials,
        int8_t *__restrict__ swiglu_int8,
        float *__restrict__ swiglu_scales,
        int total_slots,
        int N,
        int k_partitions)
    {
        constexpr int kTileN = 32;
        const int lane = threadIdx.x;
        const int block_idx = blockIdx.x;
        const int slot = blockIdx.y;
        const int n = block_idx * kTileN + lane;
        if (slot >= total_slots)
            return;

        const bool active = n < N;
        const size_t slot_base =
            static_cast<size_t>(slot) * static_cast<size_t>(k_partitions) * static_cast<size_t>(N);

        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        if (active)
        {
            for (int k_part = 0; k_part < k_partitions; ++k_part)
            {
                const size_t idx =
                    slot_base + static_cast<size_t>(k_part) * static_cast<size_t>(N) + static_cast<size_t>(n);
                gate_sum += gate_partials[idx];
                up_sum += up_partials[idx];
            }
        }

        const float value = active ? (silu(gate_sum) * up_sum) : 0.0f;
        float abs_value = fabsf(value);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_value = fmaxf(abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

        const float scale = (abs_value > 0.0f) ? (abs_value / 127.0f) : 1.0f;
        const int blocks_per_row = (N + kTileN - 1) / kTileN;
        if (lane == 0)
            swiglu_scales[static_cast<size_t>(slot) * static_cast<size_t>(blocks_per_row) +
                          static_cast<size_t>(block_idx)] = scale;

        if (active)
        {
            const float q = value / scale;
            swiglu_int8[static_cast<size_t>(slot) * static_cast<size_t>(N) + static_cast<size_t>(n)] =
                static_cast<int8_t>(rintf(fminf(127.0f, fmaxf(-127.0f, q))));
        }
    }

    template <uint8_t CodebookId>
    __global__ void grouped_native_vnni_gate_up_decode_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeVNNIMatrixDesc *__restrict__ gate_descs,
        const DeviceNativeVNNIMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ expert_ids,
        float *const *__restrict__ gate_outputs,
        float *const *__restrict__ up_outputs,
        int num_active,
        int N,
        int K)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int slot = blockIdx.y;
        if (slot >= num_active || n >= N)
            return;

        const int expert_id = expert_ids[slot];
        if (expert_id < 0)
            return;

        const DeviceNativeVNNIMatrixDesc gate_desc = gate_descs[expert_id];
        const DeviceNativeVNNIMatrixDesc up_desc = up_descs[expert_id];
        gate_outputs[slot][n] = native_vnni_dot_desc<CodebookId>(gate_desc, n, A_int8, scales_A_blockwise, N, K);
        up_outputs[slot][n] = native_vnni_dot_desc<CodebookId>(up_desc, n, A_int8, scales_A_blockwise, N, K);
    }

    template <uint8_t CodebookId>
    __global__ void grouped_native_vnni_down_decode_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeVNNIMatrixDesc *__restrict__ descs,
        const int *__restrict__ expert_ids,
        const float *__restrict__ route_weights,
        float *__restrict__ output,
        int num_active,
        int N,
        int K)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        if (n >= N)
            return;

        const int blocks_per_row = K / 32;
        float total = 0.0f;
#pragma unroll 1
        for (int slot = 0; slot < num_active; ++slot)
        {
            const int expert_id = expert_ids[slot];
            if (expert_id < 0)
                continue;

            const DeviceNativeVNNIMatrixDesc desc = descs[expert_id];
            const int8_t *slot_A = A_int8 + static_cast<size_t>(slot) * K;
            const float *slot_scales = scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;
            const float expert_value = native_vnni_dot_desc<CodebookId>(desc, n, slot_A, slot_scales, N, K);
            total += route_weights[slot] * expert_value;
        }
        output[n] = total;
    }

    /**
     * @brief Split-K scatter kernel for the grouped SwiGLU down projection.
     *
     * Each block reduces one K-partition of one output column, summing the
     * route-weighted partial contributions of all active experts for that
     * K-range, and writes the partial to a [k_partitions][N] scratch buffer.
     * A separate reduce kernel sums the partials. The down projection launches
     * only ceil(N/64) blocks in the serial path (N = d_model), leaving the GPU
     * heavily under-occupied; multiplying the block count by k_partitions
     * exposes enough warps to hide the weight-payload global-memory latency.
     *
     * The expert sum and the K-partition sum commute because each is a linear
     * accumulation, so summing experts within a K-range and then summing the
     * K-ranges yields the same result as the serial full-K expert sum.
     *
     * Grid: ((N + 63)/64, k_partitions)   Block: (64)
     */
    template <uint8_t CodebookId>
    __global__ void grouped_native_vnni_down_kpart_decode_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeVNNIMatrixDesc *__restrict__ descs,
        const int *__restrict__ expert_ids,
        const float *__restrict__ route_weights,
        float *__restrict__ partials,
        int num_active,
        int N,
        int K,
        int num_experts,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int k_part = blockIdx.y;
        if (k_part >= k_partitions || n >= N)
            return;

        // Linear index into the [k_partitions][N] partials buffer.
        const size_t partial_index =
            static_cast<size_t>(k_part) * static_cast<size_t>(N) + static_cast<size_t>(n);

        // Evenly split the K-blocks across the partitions; this block owns
        // [b_start, b_end).
        const int blocks_per_row = K / 32;
        const int blocks_per_part = (blocks_per_row + k_partitions - 1) / k_partitions;
        const int b_start = k_part * blocks_per_part;
        int b_end = b_start + blocks_per_part;
        if (b_end > blocks_per_row)
            b_end = blocks_per_row;
        if (b_start >= b_end)
        {
            partials[partial_index] = 0.0f;
            return;
        }

        // Accumulate the route-weighted expert contributions for this K-range.
        float total = 0.0f;
#pragma unroll 1
        for (int slot = 0; slot < num_active; ++slot)
        {
            const int expert_id = expert_ids[slot];
            if (expert_id < 0)
                continue;
            assert(expert_id < num_experts);
            if (expert_id >= num_experts)
                continue;

            const DeviceNativeVNNIMatrixDesc desc = descs[expert_id];
            const int8_t *slot_A = A_int8 + static_cast<size_t>(slot) * K;
            const float *slot_scales = scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;
            const float expert_value = native_vnni_dot_desc_range<CodebookId>(
                desc, n, slot_A, slot_scales, N, K, b_start, b_end);
            total += route_weights[slot] * expert_value;
        }
        partials[partial_index] = total;
    }

    /**
     * @brief Reduce K-partition partials into the final grouped down output.
     *
     * Sums the k_partitions partial contributions for each output column n
     * produced by grouped_native_vnni_down_kpart_decode_kernel.
     *
     * Grid: ((N + 63)/64)   Block: (64)
     */
    __global__ void grouped_native_vnni_down_kpart_reduce_kernel(
        const float *__restrict__ partials,
        float *__restrict__ output,
        int N,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        if (n >= N)
            return;

        float sum = 0.0f;
        for (int k_part = 0; k_part < k_partitions; ++k_part)
            sum += partials[static_cast<size_t>(k_part) * static_cast<size_t>(N) + static_cast<size_t>(n)];
        output[n] = sum;
    }

    int blocksFor(int count)
    {
        return (count + kThreads - 1) / kThreads;
    }

    int select_grouped_prefill_tile_m(int requested_tile_m, int max_tokens_per_expert)
    {
        switch (requested_tile_m)
        {
        case 2:
        case 4:
        case 8:
        case 16:
            return requested_tile_m;
        default:
            break;
        }

        // Tiny MTP verifier groups (M=2..4) are fastest with the compact
        // two-row template on CUDA. Wider M tiles add per-block work without
        // improving occupancy for these active expert groups.
        if (max_tokens_per_expert <= 4)
            return 2;
        if (max_tokens_per_expert <= 8)
            return 8;
        return 16;
    }
}

extern "C"
{
    bool cudaMoE_route_logits(const float *hidden, const float *gate_weights, float *logits,
                              int seq_len, int d_model, int num_experts,
                              int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);

        // PREFILL: with many tokens this is a genuine GEMM. The naive block-per-
        // (expert,token) kernel is L2-bandwidth-bound (re-reads hidden E× and gate S×).
        // Above this token threshold the tiled SGEMM (smem reuse) is a large win; below
        // it (decode, seq_len==1) the warp-reduction kernel keeps full SM coverage.
        constexpr int kRouteTiledMinTokens = 16;
        if (seq_len >= kRouteTiledMinTokens)
        {
            // Tile geometry must match the template instantiation below. BM=BN=64
            // (256-thread block) empirically beats smaller tiles here: although it
            // yields only 44 blocks (occupancy-bound on this M=679,N=256 GEMM), the
            // 256-thread block's load efficiency and ILP outperform 32×32 (1918) and
            // 64×32 (1931) configs that produce more blocks but fewer threads each.
            constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
            constexpr int kTiledThreads = (BM / TM) * (BN / TN); // 16×16 = 256
            dim3 grid((num_experts + BN - 1) / BN, (seq_len + BM - 1) / BM);
            route_logits_tiled_kernel<BM, BN, BK, TM, TN>
                <<<grid, kTiledThreads, 0, static_cast<cudaStream_t>(stream)>>>(
                    hidden, gate_weights, logits, seq_len, d_model, num_experts);
            return finishLaunch("cudaMoE_route_logits_tiled");
        }

        dim3 grid(num_experts, seq_len);
        route_logits_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            hidden, gate_weights, logits, seq_len, d_model, num_experts);
        return finishLaunch("cudaMoE_route_logits");
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

    bool cudaMoE_softmax_topk(float *logits, int *expert_indices, float *expert_weights,
                              int seq_len, int num_experts, int top_k, bool normalize_weights,
                              int device_idx, void *stream,
                              const int *device_effective_seq_len)
    {
        if (num_experts > kMaxExperts || top_k > kMaxTopK)
            return false;
        cudaSetDevice(device_idx);
        softmax_topk_kernel<<<seq_len, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            logits, expert_indices, expert_weights, seq_len, num_experts, top_k,
            normalize_weights, device_effective_seq_len);
        return finishLaunch("cudaMoE_softmax_topk");
    }

    bool cudaMoE_softmax_topk_decode_runtime(float *logits,
                                             int *runtime_expert_ids,
                                             float *runtime_weights,
                                             uint64_t *runtime_histogram,
                                             float *legacy_indices, float *legacy_weights,
                                             int num_experts, int top_k, bool normalize_weights,
                                             bool write_legacy_outputs, bool update_runtime_histogram,
                                             int device_idx, void *stream)
    {
        if (num_experts > kMaxExperts || top_k > kMaxTopK)
            return false;
        cudaSetDevice(device_idx);
        softmax_topk_decode_runtime_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            logits, runtime_expert_ids, runtime_weights, runtime_histogram,
            legacy_indices, legacy_weights, num_experts, top_k, normalize_weights,
            write_legacy_outputs, update_runtime_histogram);
        return finishLaunch("cudaMoE_softmax_topk_decode_runtime");
    }

    bool cudaMoE_decode_route_select_runtime(const int *expert_indices, const float *expert_weights,
                                             int *runtime_expert_ids,
                                             float *runtime_weights,
                                             uint64_t *runtime_histogram,
                                             float *legacy_indices, float *legacy_weights,
                                             int, int top_k, bool write_legacy_outputs,
                                             bool update_runtime_histogram, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        decode_route_select_runtime_kernel<<<1, kMaxTopK, 0, static_cast<cudaStream_t>(stream)>>>(
            expert_indices, expert_weights, runtime_expert_ids, runtime_weights, runtime_histogram,
            legacy_indices, legacy_weights, top_k, write_legacy_outputs, update_runtime_histogram);
        return finishLaunch("cudaMoE_decode_route_select_runtime");
    }

    bool cudaMoE_int_to_float(const int *input, float *output, int count, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        int_to_float_kernel<<<blocksFor(count), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(input, output, count);
        return finishLaunch("cudaMoE_int_to_float");
    }

    bool cudaMoE_float_to_int(const float *input, int *output, int count, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        float_to_int_kernel<<<blocksFor(count), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(input, output, count);
        return finishLaunch("cudaMoE_float_to_int");
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
            dim3 grid((n4 + kThreads - 1) / kThreads, num_tokens);
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
            dim3 grid((n4 + kThreads - 1) / kThreads, num_tokens);
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
        shared_expert_gate_add_kernel<<<seq_len, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
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
        shared_expert_gate_add_kernel<<<seq_len, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            input, gate_inp, shared_output, routed_residual, combined_output,
            seq_len, d_model, device_effective_seq_len);
        return finishLaunch("cudaMoE_shared_expert_gate_add_effective_seq_len");
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

    bool cudaMoE_count_per_expert(const int *routing_indices, int *expert_counts, int total_slots,
                                  int num_experts, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        count_per_expert_kernel<<<blocksFor(total_slots), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            routing_indices, expert_counts, total_slots, num_experts);
        return finishLaunch("cudaMoE_count_per_expert");
    }

    bool cudaMoE_exclusive_scan(const int *expert_counts, int *expert_offsets,
                                int num_experts, int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        exclusive_scan_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(expert_counts, expert_offsets, num_experts);
        return finishLaunch("cudaMoE_exclusive_scan");
    }

    bool cudaMoE_scatter_tokens(const int *routing_indices, const float *routing_weights,
                                int *write_heads, const int *expert_offsets,
                                int *grouped_token_indices, float *grouped_weights,
                                int total_slots, int top_k, int num_experts,
                                int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        scatter_tokens_kernel<<<blocksFor(total_slots), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            routing_indices, routing_weights, write_heads, expert_offsets,
            grouped_token_indices, grouped_weights, total_slots, top_k, num_experts);
        return finishLaunch("cudaMoE_scatter_tokens");
    }

    bool cudaMoE_scatter_tokens_deterministic(
        const int *routing_indices,
        const float *routing_weights,
        const int *expert_offsets,
        const int *expert_counts,
        int *grouped_token_indices,
        int *original_to_grouped,
        int *original_expert_ids,
        float *grouped_weights,
        int total_slots,
        int top_k,
        int num_experts,
        int device_idx,
        void *stream)
    {
        if (!stream)
            return false;
        cudaSetDevice(device_idx);
        scatter_tokens_deterministic_kernel<<<blocksFor(total_slots), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            routing_indices, routing_weights, expert_offsets, expert_counts,
            grouped_token_indices, original_to_grouped, original_expert_ids, grouped_weights,
            total_slots, top_k, num_experts);
        return finishLaunch("cudaMoE_scatter_tokens_deterministic");
    }

    bool cudaMoE_group_tokens_small_float(
        const float *routing_indices,
        const float *routing_weights,
        int *expert_counts,
        int *expert_offsets,
        int *grouped_token_indices,
        int *original_to_grouped,
        int *original_expert_ids,
        float *grouped_weights,
        int *active_expert_ids,
        int total_slots,
        int num_experts,
        int top_k,
        int max_active_experts,
        int device_idx,
        void *stream)
    {
        if (!stream)
            return false;
        cudaSetDevice(device_idx);
        group_tokens_small_float_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            routing_indices, routing_weights, expert_counts, expert_offsets,
            grouped_token_indices, original_to_grouped, original_expert_ids, grouped_weights,
            active_expert_ids, total_slots, num_experts, top_k, max_active_experts);
        return finishLaunch("cudaMoE_group_tokens_small_float");
    }

    bool cudaMoE_prepare_shared_expert_group(
        int *expert_offsets,
        int *expert_counts,
        int *grouped_token_indices,
        int *original_to_grouped,
        float *grouped_weights,
        int *active_expert_ids,
        int seq_len,
        int device_idx,
        void *stream)
    {
        cudaSetDevice(device_idx);
        prepare_shared_expert_group_kernel<<<blocksFor(seq_len), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            expert_offsets, expert_counts, grouped_token_indices, original_to_grouped,
            grouped_weights, active_expert_ids, seq_len);
        return finishLaunch("cudaMoE_prepare_shared_expert_group");
    }

    bool cudaMoE_gather_expert_fixed(const float *hidden, float *batch_buffer,
                                     const int *expert_offsets, const int *expert_counts,
                                     const int *grouped_token_indices,
                                     int expert_id, int max_tokens, int d_model,
                                     int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        const int total = max_tokens * d_model;
        gather_expert_fixed_kernel<<<blocksFor(total), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            hidden, batch_buffer, expert_offsets, expert_counts, grouped_token_indices,
            expert_id, max_tokens, d_model);
        return finishLaunch("cudaMoE_gather_expert_fixed");
    }

    bool cudaMoE_scatter_expert_fixed(float *output, const float *expert_output,
                                      const int *expert_offsets, const int *expert_counts,
                                      const int *grouped_token_indices,
                                      const float *grouped_weights,
                                      int expert_id, int max_tokens, int d_model,
                                      int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        const int total = max_tokens * d_model;
        scatter_expert_fixed_kernel<<<blocksFor(total), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            output, expert_output, expert_offsets, expert_counts, grouped_token_indices,
            grouped_weights, expert_id, max_tokens, d_model);
        return finishLaunch("cudaMoE_scatter_expert_fixed");
    }

    bool cudaMoE_grouped_gate_up_native_vnni_decode_table(
        const float *d_hidden,
        const DeviceNativeVNNIMatrixDesc *d_gate_desc_table,
        const DeviceNativeVNNIMatrixDesc *d_up_desc_table,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        int num_active,
        int N,
        int K,
        uint8_t codebook_id,
        int device_idx,
        void *stream)
    {
        if (!d_hidden || !d_gate_desc_table || !d_up_desc_table || !d_expert_ids ||
            !d_gate_outputs || !d_up_outputs || !d_hidden_int8 || !d_hidden_scales ||
            num_active <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_gate_up_native_vnni_decode_table] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        const int blocks_per_row = K / 32;
        grouped_hidden_quantize_blockwise_kernel<<<blocks_per_row, 32, 0, cuda_stream>>>(
            d_hidden, d_hidden_int8, d_hidden_scales, K);
        if (!finishLaunch("cudaMoE_grouped_gate_up_hidden_quantize"))
            return false;

        constexpr int kTileN = 64;
        dim3 grid((N + kTileN - 1) / kTileN, num_active);
        dim3 block(kTileN);

#define LAUNCH_GROUPED_GATE_UP(CB)                                                                  \
    grouped_native_vnni_gate_up_decode_kernel<CB><<<grid, block, 0, cuda_stream>>>(                 \
        d_hidden_int8, d_hidden_scales, d_gate_desc_table, d_up_desc_table, d_expert_ids,           \
        d_gate_outputs, d_up_outputs, num_active, N, K)

        switch (codebook_id)
        {
        case 0:  LAUNCH_GROUPED_GATE_UP(0); break;
        case 4:  LAUNCH_GROUPED_GATE_UP(4); break;
        case 5:  LAUNCH_GROUPED_GATE_UP(5); break;
        case 6:  LAUNCH_GROUPED_GATE_UP(6); break;
        case 7:  LAUNCH_GROUPED_GATE_UP(7); break;
        case 8:  LAUNCH_GROUPED_GATE_UP(8); break;
        case 9:  LAUNCH_GROUPED_GATE_UP(9); break;
        case 10: LAUNCH_GROUPED_GATE_UP(10); break;
        case 11: LAUNCH_GROUPED_GATE_UP(11); break;
        case 12: LAUNCH_GROUPED_GATE_UP(12); break;
        case 13: LAUNCH_GROUPED_GATE_UP(13); break;
        case 14: LAUNCH_GROUPED_GATE_UP(14); break;
        case 15: LAUNCH_GROUPED_GATE_UP(15); break;
        case 16: LAUNCH_GROUPED_GATE_UP(16); break;
        case 17: LAUNCH_GROUPED_GATE_UP(17); break;
        case 19: LAUNCH_GROUPED_GATE_UP(19); break;
        default:
            std::fprintf(stderr, "[cudaMoE_grouped_gate_up_native_vnni_decode_table] unsupported codebook_id=%u\n",
                         static_cast<unsigned>(codebook_id));
            return false;
        }

#undef LAUNCH_GROUPED_GATE_UP

        return finishLaunch("cudaMoE_grouped_gate_up_native_vnni_decode_table");
    }

    bool cudaMoE_grouped_gate_up_native_vnni_decode_table_kpart(
        const float *d_hidden,
        const DeviceNativeVNNIMatrixDesc *d_gate_desc_table,
        const DeviceNativeVNNIMatrixDesc *d_up_desc_table,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        float *d_gate_partials,
        float *d_up_partials,
        int num_active,
        int N,
        int K,
        int num_experts,
        uint8_t codebook_id,
        int k_partitions,
        int device_idx,
        void *stream)
    {
        const bool valid_k_partitions = (k_partitions == 2 || k_partitions == 4 || k_partitions == 8 ||
                                         k_partitions == 16 || k_partitions == 32);
        if (!d_hidden || !d_gate_desc_table || !d_up_desc_table || !d_expert_ids ||
            !d_gate_outputs || !d_up_outputs || !d_hidden_int8 || !d_hidden_scales ||
            !d_gate_partials || !d_up_partials ||
            num_active <= 0 || N <= 0 || K <= 0 || num_experts <= 0 ||
            (K % 32) != 0 || !valid_k_partitions)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_gate_up_native_vnni_decode_table_kpart] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        // Step 1: blockwise-quantize the shared hidden activation row once.
        const int blocks_per_row = K / 32;
        grouped_hidden_quantize_blockwise_kernel<<<blocks_per_row, 32, 0, cuda_stream>>>(
            d_hidden, d_hidden_int8, d_hidden_scales, K);
        if (!finishLaunch("cudaMoE_grouped_gate_up_kpart_hidden_quantize"))
            return false;

        // Step 2: split-K scatter — each (n-tile, k-partition, expert-slot) block
        // reduces its K-slice into the partials buffer.
        constexpr int kTileN = 64;
        dim3 partial_grid((N + kTileN - 1) / kTileN, k_partitions, num_active);
        dim3 block(kTileN);

#define LAUNCH_GROUPED_GATE_UP_KPART(CB)                                                            \
    grouped_native_vnni_gate_up_kpart_decode_kernel<CB><<<partial_grid, block, 0, cuda_stream>>>(   \
        d_hidden_int8, d_hidden_scales, d_gate_desc_table, d_up_desc_table, d_expert_ids,           \
        d_gate_partials, d_up_partials, num_active, N, K, num_experts, k_partitions)

        switch (codebook_id)
        {
        case 0:  LAUNCH_GROUPED_GATE_UP_KPART(0); break;
        case 4:  LAUNCH_GROUPED_GATE_UP_KPART(4); break;
        case 5:  LAUNCH_GROUPED_GATE_UP_KPART(5); break;
        case 6:  LAUNCH_GROUPED_GATE_UP_KPART(6); break;
        case 7:  LAUNCH_GROUPED_GATE_UP_KPART(7); break;
        case 8:  LAUNCH_GROUPED_GATE_UP_KPART(8); break;
        case 9:  LAUNCH_GROUPED_GATE_UP_KPART(9); break;
        case 10: LAUNCH_GROUPED_GATE_UP_KPART(10); break;
        case 11: LAUNCH_GROUPED_GATE_UP_KPART(11); break;
        case 12: LAUNCH_GROUPED_GATE_UP_KPART(12); break;
        case 13: LAUNCH_GROUPED_GATE_UP_KPART(13); break;
        case 14: LAUNCH_GROUPED_GATE_UP_KPART(14); break;
        case 15: LAUNCH_GROUPED_GATE_UP_KPART(15); break;
        case 16: LAUNCH_GROUPED_GATE_UP_KPART(16); break;
        case 17: LAUNCH_GROUPED_GATE_UP_KPART(17); break;
        case 19: LAUNCH_GROUPED_GATE_UP_KPART(19); break;
        default:
            std::fprintf(stderr, "[cudaMoE_grouped_gate_up_native_vnni_decode_table_kpart] unsupported codebook_id=%u\n",
                         static_cast<unsigned>(codebook_id));
            return false;
        }

#undef LAUNCH_GROUPED_GATE_UP_KPART

        if (!finishLaunch("cudaMoE_grouped_gate_up_kpart_partial"))
            return false;

        // Step 3: reduce the k_partitions partials into the final gate/up outputs.
        dim3 reduce_grid((N + kTileN - 1) / kTileN, num_active);
        grouped_native_vnni_gate_up_kpart_reduce_kernel<<<reduce_grid, block, 0, cuda_stream>>>(
            d_gate_partials, d_up_partials, d_gate_outputs, d_up_outputs,
            num_active, N, k_partitions);

        return finishLaunch("cudaMoE_grouped_gate_up_native_vnni_decode_table_kpart");
    }

    bool cudaMoE_grouped_swiglu_down_native_vnni_decode_table(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const DeviceNativeVNNIMatrixDesc *d_desc_table,
        const int *d_expert_ids,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_output,
        int num_active,
        int N,
        int K,
        uint8_t codebook_id,
        int device_idx,
        void *stream)
    {
        if (!d_gate_ptrs || !d_up_ptrs || !d_desc_table || !d_expert_ids || !d_weights ||
            !d_swiglu_int8 || !d_swiglu_scales || !d_output ||
            num_active <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_swiglu_down_native_vnni_decode_table] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        grouped_swiglu_quantize_blockwise_kernel<<<num_active, kThreads, 0, cuda_stream>>>(
            d_gate_ptrs, d_up_ptrs, d_swiglu_int8, d_swiglu_scales, num_active, K);
        if (!finishLaunch("cudaMoE_grouped_swiglu_quantize"))
            return false;

        constexpr int kTileN = 64;
        dim3 grid((N + kTileN - 1) / kTileN);
        dim3 block(kTileN);

#define LAUNCH_GROUPED_DOWN(CB)                                                                  \
    grouped_native_vnni_down_decode_kernel<CB><<<grid, block, 0, cuda_stream>>>(                 \
        d_swiglu_int8, d_swiglu_scales, d_desc_table, d_expert_ids, d_weights,                   \
        d_output, num_active, N, K)

        switch (codebook_id)
        {
        case 0:  LAUNCH_GROUPED_DOWN(0); break;
        case 4:  LAUNCH_GROUPED_DOWN(4); break;
        case 5:  LAUNCH_GROUPED_DOWN(5); break;
        case 6:  LAUNCH_GROUPED_DOWN(6); break;
        case 7:  LAUNCH_GROUPED_DOWN(7); break;
        case 8:  LAUNCH_GROUPED_DOWN(8); break;
        case 9:  LAUNCH_GROUPED_DOWN(9); break;
        case 10: LAUNCH_GROUPED_DOWN(10); break;
        case 11: LAUNCH_GROUPED_DOWN(11); break;
        case 12: LAUNCH_GROUPED_DOWN(12); break;
        case 13: LAUNCH_GROUPED_DOWN(13); break;
        case 14: LAUNCH_GROUPED_DOWN(14); break;
        case 15: LAUNCH_GROUPED_DOWN(15); break;
        case 16: LAUNCH_GROUPED_DOWN(16); break;
        case 17: LAUNCH_GROUPED_DOWN(17); break;
        case 19: LAUNCH_GROUPED_DOWN(19); break;
        default:
            std::fprintf(stderr, "[cudaMoE_grouped_swiglu_down_native_vnni_decode_table] unsupported codebook_id=%u\n",
                         static_cast<unsigned>(codebook_id));
            return false;
        }

#undef LAUNCH_GROUPED_DOWN

        return finishLaunch("cudaMoE_grouped_swiglu_down_native_vnni_decode_table");
    }

    bool cudaMoE_grouped_swiglu_down_native_vnni_decode_table_kpart(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const DeviceNativeVNNIMatrixDesc *d_desc_table,
        const int *d_expert_ids,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_down_partials,
        float *d_output,
        int num_active,
        int d_model,
        int intermediate,
        int num_experts,
        uint8_t codebook_id,
        int k_partitions,
        int device_idx,
        void *stream)
    {
        const bool valid_k_partitions =
            (k_partitions == 2 || k_partitions == 4 || k_partitions == 8 ||
             k_partitions == 16);
        if (!d_gate_ptrs || !d_up_ptrs || !d_desc_table || !d_expert_ids || !d_weights ||
            !d_swiglu_int8 || !d_swiglu_scales || !d_down_partials || !d_output ||
            num_active <= 0 || d_model <= 0 || intermediate <= 0 || num_experts <= 0 ||
            (intermediate % 32) != 0 || !valid_k_partitions)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_swiglu_down_native_vnni_decode_table_kpart] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        // Step 1: quantize the per-slot SwiGLU activations (shared with serial path).
        grouped_swiglu_quantize_blockwise_kernel<<<num_active, kThreads, 0, cuda_stream>>>(
            d_gate_ptrs, d_up_ptrs, d_swiglu_int8, d_swiglu_scales, num_active, intermediate);
        if (!finishLaunch("cudaMoE_grouped_swiglu_quantize"))
            return false;

        constexpr int kTileN = 64;
        const int N = d_model;
        const int K = intermediate;
        dim3 scatter_grid((N + kTileN - 1) / kTileN, k_partitions);
        dim3 reduce_grid((N + kTileN - 1) / kTileN);
        dim3 block(kTileN);

        // Step 2: scatter — each (n, k_part) block writes a route-weighted partial.
#define LAUNCH_GROUPED_DOWN_KPART(CB)                                                            \
    grouped_native_vnni_down_kpart_decode_kernel<CB><<<scatter_grid, block, 0, cuda_stream>>>(   \
        d_swiglu_int8, d_swiglu_scales, d_desc_table, d_expert_ids, d_weights,                   \
        d_down_partials, num_active, N, K, num_experts, k_partitions)

        switch (codebook_id)
        {
        case 0:  LAUNCH_GROUPED_DOWN_KPART(0); break;
        case 4:  LAUNCH_GROUPED_DOWN_KPART(4); break;
        case 5:  LAUNCH_GROUPED_DOWN_KPART(5); break;
        case 6:  LAUNCH_GROUPED_DOWN_KPART(6); break;
        case 7:  LAUNCH_GROUPED_DOWN_KPART(7); break;
        case 8:  LAUNCH_GROUPED_DOWN_KPART(8); break;
        case 9:  LAUNCH_GROUPED_DOWN_KPART(9); break;
        case 10: LAUNCH_GROUPED_DOWN_KPART(10); break;
        case 11: LAUNCH_GROUPED_DOWN_KPART(11); break;
        case 12: LAUNCH_GROUPED_DOWN_KPART(12); break;
        case 13: LAUNCH_GROUPED_DOWN_KPART(13); break;
        case 14: LAUNCH_GROUPED_DOWN_KPART(14); break;
        case 15: LAUNCH_GROUPED_DOWN_KPART(15); break;
        case 16: LAUNCH_GROUPED_DOWN_KPART(16); break;
        case 17: LAUNCH_GROUPED_DOWN_KPART(17); break;
        case 19: LAUNCH_GROUPED_DOWN_KPART(19); break;
        default:
            std::fprintf(stderr, "[cudaMoE_grouped_swiglu_down_native_vnni_decode_table_kpart] unsupported codebook_id=%u\n",
                         static_cast<unsigned>(codebook_id));
            return false;
        }

#undef LAUNCH_GROUPED_DOWN_KPART

        if (!finishLaunch("cudaMoE_grouped_swiglu_down_native_vnni_decode_table_kpart scatter"))
            return false;

        // Step 3: reduce — sum the k_partitions partials into the final output.
        grouped_native_vnni_down_kpart_reduce_kernel<<<reduce_grid, block, 0, cuda_stream>>>(
            d_down_partials, d_output, N, k_partitions);

        return finishLaunch("cudaMoE_grouped_swiglu_down_native_vnni_decode_table_kpart reduce");
    }

    bool cudaMoE_grouped_prefill_pipeline(
        const float *d_hidden,
        const DeviceNativeVNNIMatrixDesc *d_gate_desc_table,
        const DeviceNativeVNNIMatrixDesc *d_up_desc_table,
        const DeviceNativeVNNIMatrixDesc *d_down_desc_table,
        const int *d_group_counts,
        const int *d_group_offsets,
        const int *d_group_token_indices,
        const int *d_original_to_grouped,
        const int *d_active_expert_ids,
        const float *d_group_weights,
        int8_t *d_scratch_A_int8,
        float *d_scratch_scales,
        float *d_scratch_gate,
        float *d_scratch_up,
        float *d_gate_partials,
        float *d_up_partials,
        int8_t *d_scratch_swiglu_int8,
        float *d_scratch_swiglu_scales,
        float *d_scratch_down_out,
        float *d_output,
        int num_experts,
        int d_model,
        int intermediate,
        int max_tokens_per_expert,
        int total_slots,
        int top_k,
        int active_expert_slots,
        uint8_t gateup_codebook_id,
        uint8_t down_codebook_id,
        uint32_t gateup_codebook_mask,
        uint32_t down_codebook_mask,
        int gateup_k_partitions,
        int device_idx,
        void *stream)
    {
        if (!d_hidden || !d_gate_desc_table || !d_up_desc_table || !d_down_desc_table ||
            !d_group_counts || !d_group_offsets || !d_group_token_indices || !d_group_weights ||
            !d_scratch_A_int8 || !d_scratch_scales || !d_scratch_gate || !d_scratch_up ||
            !d_scratch_swiglu_int8 || !d_scratch_swiglu_scales || !d_scratch_down_out || !d_output ||
            num_experts <= 0 || d_model <= 0 || intermediate <= 0 ||
            max_tokens_per_expert <= 0 || total_slots <= 0 || top_k <= 0 ||
            active_expert_slots < 0 ||
            (d_model % 32) != 0 || (intermediate % 32) != 0)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_prefill_pipeline] invalid arguments\n");
            return false;
        }
        const bool use_active_expert_grid = active_expert_slots > 0;
        const bool use_gateup_kpart =
            use_active_expert_grid &&
            max_tokens_per_expert <= 4 &&
            gateup_k_partitions > 1 &&
            d_gate_partials &&
            d_up_partials;
        if (use_active_expert_grid &&
            (!d_active_expert_ids ||
             active_expert_slots > total_slots ||
             active_expert_slots > num_experts))
            return false;
        const int expert_grid = use_active_expert_grid ? active_expert_slots : num_experts;

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        {
            // One warp per 32-col quant block; pack kWarpsPerQuantBlock warps per CUDA
            // block so each block covers kWarpsPerQuantBlock*32 columns. grid.x rounds up.
            const int blocks_per_row = d_model / 32;
            dim3 grid((blocks_per_row + kWarpsPerQuantBlock - 1) / kWarpsPerQuantBlock, total_slots);
            dim3 block(kWarpsPerQuantBlock * 32);
            grouped_prefill_gather_quantize_blockwise_kernel<<<grid, block, 0, cuda_stream>>>(
                d_hidden, d_scratch_A_int8, d_scratch_scales,
                d_group_token_indices, total_slots, d_model);
            if (!finishLaunch("cudaMoE_grouped_prefill_gather_quantize"))
                return false;
        }

        {
            const int requestedTileM =
                llaminar2::debugEnv().gemm.cuda_moe_prefill_tile_m;
            const int kTileM =
                select_grouped_prefill_tile_m(requestedTileM, max_tokens_per_expert);
            const bool tiny_active_verifier =
                active_expert_slots > 0 && max_tokens_per_expert <= 4;
            const int kTileN = tiny_active_verifier ? 64 : 128;
            // When fusion is enabled the gate/up kernel computes SwiGLU + blockwise int8 quant
            // in its epilogue, writing the down-projection input directly (no FP32 gate/up
            // round-trip, no separate swiglu_quantize launch).
            const bool fuse_swiglu = llaminar2::debugEnv().gemm.cuda_moe_prefill_fuse_swiglu;
            dim3 grid((intermediate + kTileN - 1) / kTileN,
                      (max_tokens_per_expert + kTileM - 1) / kTileM,
                      expert_grid);
            dim3 block(kTileN);

#define LAUNCH_GROUPED_GATEUP_PREFILL_TM(CB, TM)                                                   \
    do {                                                                                           \
        dim3 cb_grid(grid.x, grid.y, expert_grid);                                                  \
        if (kTileN == 64)                                                                            \
            grouped_native_vnni_gate_up_prefill_kernel<CB, TM, 64><<<cb_grid, block, 0, cuda_stream>>>( \
                d_scratch_A_int8, d_scratch_scales, d_gate_desc_table, d_up_desc_table,             \
                d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,          \
                d_scratch_gate, d_scratch_up, intermediate, d_model);                               \
        else                                                                                         \
            grouped_native_vnni_gate_up_prefill_kernel<CB, TM, 128><<<cb_grid, block, 0, cuda_stream>>>( \
                d_scratch_A_int8, d_scratch_scales, d_gate_desc_table, d_up_desc_table,             \
                d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,          \
                d_scratch_gate, d_scratch_up, intermediate, d_model);                               \
    } while (0)
#define LAUNCH_GROUPED_GATEUP_KPART_PREFILL_TM(CB, TM)                                             \
    do {                                                                                           \
        dim3 cb_grid(grid.x, grid.y, expert_grid * gateup_k_partitions);                           \
        grouped_native_vnni_gate_up_prefill_kpart_kernel<CB, TM><<<cb_grid, block, 0, cuda_stream>>>( \
            d_scratch_A_int8, d_scratch_scales, d_gate_desc_table, d_up_desc_table,                 \
            d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,              \
            d_gate_partials, d_up_partials, intermediate, d_model, gateup_k_partitions);            \
    } while (0)
#define LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM(CB, TM)                                            \
    do {                                                                                           \
        dim3 cb_grid(grid.x, grid.y, expert_grid);                                                  \
        if (kTileN == 64)                                                                            \
            grouped_native_vnni_gate_up_swiglu_prefill_kernel<CB, TM, 64><<<cb_grid, block, 0, cuda_stream>>>( \
                d_scratch_A_int8, d_scratch_scales, d_gate_desc_table, d_up_desc_table,             \
                d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,          \
                d_scratch_swiglu_int8, d_scratch_swiglu_scales,                                    \
                intermediate, d_model);                                                             \
        else                                                                                         \
            grouped_native_vnni_gate_up_swiglu_prefill_kernel<CB, TM, 128><<<cb_grid, block, 0, cuda_stream>>>( \
                d_scratch_A_int8, d_scratch_scales, d_gate_desc_table, d_up_desc_table,             \
                d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,          \
                d_scratch_swiglu_int8, d_scratch_swiglu_scales,                                    \
                intermediate, d_model);                                                             \
    } while (0)
#define LAUNCH_GROUPED_GATEUP_PREFILL(CB)                                                          \
    do {                                                                                           \
        if (use_gateup_kpart) {                                                                     \
            if (kTileM == 16)      LAUNCH_GROUPED_GATEUP_KPART_PREFILL_TM(CB, 16);                 \
            else if (kTileM == 8)  LAUNCH_GROUPED_GATEUP_KPART_PREFILL_TM(CB, 8);                  \
            else if (kTileM == 4)  LAUNCH_GROUPED_GATEUP_KPART_PREFILL_TM(CB, 4);                  \
            else                   LAUNCH_GROUPED_GATEUP_KPART_PREFILL_TM(CB, 2);                  \
        } else if (fuse_swiglu) {                                                                   \
            if (kTileM == 16)      LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM(CB, 16);                 \
            else if (kTileM == 8)  LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM(CB, 8);                  \
            else if (kTileM == 4)  LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM(CB, 4);                  \
            else                   LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM(CB, 2);                  \
        } else {                                                                                    \
            if (kTileM == 16)      LAUNCH_GROUPED_GATEUP_PREFILL_TM(CB, 16);                        \
            else if (kTileM == 8)  LAUNCH_GROUPED_GATEUP_PREFILL_TM(CB, 8);                         \
            else if (kTileM == 4)  LAUNCH_GROUPED_GATEUP_PREFILL_TM(CB, 4);                         \
            else                   LAUNCH_GROUPED_GATEUP_PREFILL_TM(CB, 2);                         \
        }                                                                                           \
    } while (0)

            bool launched_gateup = false;
#define LAUNCH_GROUPED_GATEUP_IF_PRESENT(CB)                                                       \
    do {                                                                                           \
        if (gateup_codebook_mask & (uint32_t{1} << (CB))) {                                        \
            LAUNCH_GROUPED_GATEUP_PREFILL(CB);                                                      \
            if (!finishLaunch("cudaMoE_grouped_gate_up_prefill"))                                  \
                return false;                                                                       \
            launched_gateup = true;                                                                 \
        }                                                                                           \
    } while (0)

            LAUNCH_GROUPED_GATEUP_IF_PRESENT(0);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(4);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(5);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(6);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(7);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(8);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(9);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(10);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(11);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(12);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(13);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(14);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(15);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(16);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(17);
            LAUNCH_GROUPED_GATEUP_IF_PRESENT(19);
            if (!launched_gateup)
            {
                std::fprintf(stderr,
                             "[cudaMoE_grouped_prefill_pipeline] unsupported gate/up codebook_id=%u mask=0x%x\n",
                             static_cast<unsigned>(gateup_codebook_id),
                             static_cast<unsigned>(gateup_codebook_mask));
                return false;
            }

            if (use_gateup_kpart)
            {
                constexpr int kReduceTileN = 32;
                dim3 reduce_grid((intermediate + kReduceTileN - 1) / kReduceTileN, total_slots);
                dim3 reduce_block(kReduceTileN);
                grouped_native_vnni_gate_up_prefill_kpart_reduce_swiglu_kernel<<<
                    reduce_grid, reduce_block, 0, cuda_stream>>>(
                    d_gate_partials, d_up_partials,
                    d_scratch_swiglu_int8, d_scratch_swiglu_scales,
                    total_slots, intermediate, gateup_k_partitions);
                if (!finishLaunch("cudaMoE_grouped_gate_up_prefill_kpart_reduce_swiglu"))
                    return false;
            }

#undef LAUNCH_GROUPED_GATEUP_IF_PRESENT
#undef LAUNCH_GROUPED_GATEUP_PREFILL
#undef LAUNCH_GROUPED_GATEUP_KPART_PREFILL_TM
#undef LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM
#undef LAUNCH_GROUPED_GATEUP_PREFILL_TM
        }

        // Separate SwiGLU + blockwise-quant pass. Skipped when fusion is enabled — the fused
        // gate/up kernel already produced d_scratch_swiglu_int8 / d_scratch_swiglu_scales.
        if (!use_gateup_kpart && !llaminar2::debugEnv().gemm.cuda_moe_prefill_fuse_swiglu)
        {
            dim3 grid(intermediate / 32, total_slots);
            dim3 block(32);
            grouped_prefill_swiglu_quantize_blockwise_kernel<<<grid, block, 0, cuda_stream>>>(
                d_scratch_gate, d_scratch_up,
                d_scratch_swiglu_int8, d_scratch_swiglu_scales,
                total_slots, intermediate);
            if (!finishLaunch("cudaMoE_grouped_swiglu_quantize_prefill"))
                return false;
        }

        {
            const int requestedTileM =
                llaminar2::debugEnv().gemm.cuda_moe_prefill_tile_m;
            const int kTileM =
                select_grouped_prefill_tile_m(requestedTileM, max_tokens_per_expert);
            const bool tiny_active_verifier =
                active_expert_slots > 0 && max_tokens_per_expert <= 4;
            const int kTileN = tiny_active_verifier ? 64 : 128;
            dim3 grid((d_model + kTileN - 1) / kTileN,
                      (max_tokens_per_expert + kTileM - 1) / kTileM,
                      expert_grid);
            dim3 block(kTileN);

#define LAUNCH_GROUPED_DOWN_PREFILL_TM(CB, TM)                                                     \
    do {                                                                                           \
        dim3 cb_grid(grid.x, grid.y, expert_grid);                                                  \
        if (kTileN == 64)                                                                            \
            grouped_native_vnni_down_prefill_kernel<CB, TM, 64><<<cb_grid, block, 0, cuda_stream>>>( \
                d_scratch_swiglu_int8, d_scratch_swiglu_scales, d_down_desc_table,                  \
                d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,          \
                d_scratch_down_out, d_model, intermediate);                                         \
        else                                                                                         \
            grouped_native_vnni_down_prefill_kernel<CB, TM, 128><<<cb_grid, block, 0, cuda_stream>>>( \
                d_scratch_swiglu_int8, d_scratch_swiglu_scales, d_down_desc_table,                  \
                d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,          \
                d_scratch_down_out, d_model, intermediate);                                         \
    } while (0)
#define LAUNCH_GROUPED_DOWN_PREFILL(CB)                                                            \
    do {                                                                                           \
        if (kTileM == 16)      LAUNCH_GROUPED_DOWN_PREFILL_TM(CB, 16);                              \
        else if (kTileM == 8)  LAUNCH_GROUPED_DOWN_PREFILL_TM(CB, 8);                               \
        else if (kTileM == 4)  LAUNCH_GROUPED_DOWN_PREFILL_TM(CB, 4);                               \
        else                   LAUNCH_GROUPED_DOWN_PREFILL_TM(CB, 2);                               \
    } while (0)

            bool launched_down = false;
#define LAUNCH_GROUPED_DOWN_IF_PRESENT(CB)                                                         \
    do {                                                                                           \
        if (down_codebook_mask & (uint32_t{1} << (CB))) {                                          \
            LAUNCH_GROUPED_DOWN_PREFILL(CB);                                                        \
            if (!finishLaunch("cudaMoE_grouped_down_prefill"))                                    \
                return false;                                                                       \
            launched_down = true;                                                                   \
        }                                                                                           \
    } while (0)

            LAUNCH_GROUPED_DOWN_IF_PRESENT(0);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(4);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(5);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(6);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(7);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(8);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(9);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(10);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(11);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(12);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(13);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(14);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(15);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(16);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(17);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(19);
            if (!launched_down)
            {
                std::fprintf(stderr,
                             "[cudaMoE_grouped_prefill_pipeline] unsupported down codebook_id=%u mask=0x%x\n",
                             static_cast<unsigned>(down_codebook_id),
                             static_cast<unsigned>(down_codebook_mask));
                return false;
            }

#undef LAUNCH_GROUPED_DOWN_IF_PRESENT
#undef LAUNCH_GROUPED_DOWN_PREFILL
#undef LAUNCH_GROUPED_DOWN_PREFILL_TM
        }

        {
            constexpr int kTileN = 64;
            dim3 block(kTileN);
            if (d_original_to_grouped)
            {
                const int seq_len = total_slots / top_k;
                dim3 grid((d_model + kTileN - 1) / kTileN, seq_len);
                grouped_prefill_scatter_weighted_ordered_kernel<<<grid, block, 0, cuda_stream>>>(
                    d_output, d_scratch_down_out,
                    d_original_to_grouped, d_group_weights,
                    seq_len, top_k, d_model);
            }
            else
            {
                dim3 grid((d_model + kTileN - 1) / kTileN, total_slots);
                grouped_prefill_scatter_weighted_kernel<<<grid, block, 0, cuda_stream>>>(
                    d_output, d_scratch_down_out,
                    d_group_token_indices, d_group_weights,
                    total_slots, d_model);
            }
            if (!finishLaunch("cudaMoE_grouped_scatter_prefill"))
                return false;
        }

        return true;
    }
}
