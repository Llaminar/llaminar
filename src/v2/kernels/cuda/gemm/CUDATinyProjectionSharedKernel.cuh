/**
 * @file CUDATinyProjectionSharedKernel.cuh
 * @brief Shared-operand floating projection with the canonical 256-part FP32 tree.
 *
 * A CTA stages eight native weight rows and a small tile of FP32 activation
 * rows. Eight column-owning warps reuse those operands without changing any
 * multiply-add, partial accumulator, or reduction edge from serial projection.
 * The kernel owns only CTA-local storage: model weights, graph pointer arrays,
 * activations and outputs remain in their existing persistent allocations.
 * This header defines physical candidates, not their dispatch/economy policy.
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <type_traits>

namespace llaminar2::cuda::detail
{
/**
 * @brief Expand a persistent native floating weight without changing its bits.
 * @tparam Weight Native FP32, FP16 or BF16 storage type.
 * @param value One element loaded from the unchanged model weight allocation.
 * @return The exact FP32 expansion of a finite native weight.
 */
template<class Weight>
__device__ __forceinline__ float tinyProjectionWeight(Weight value)
{
    static_assert(std::is_same_v<Weight, float> || std::is_same_v<Weight, __half> ||
                  std::is_same_v<Weight, __nv_bfloat16>);
    if constexpr (std::is_same_v<Weight, float>)
        return value;
    else if constexpr (std::is_same_v<Weight, __half>)
        return __half2float(value);
    else
        // BF16 is the high half of an FP32 bit pattern. Keep this expression
        // visible to the compiler when scheduling tile loads. CUDA's BF16
        // helper uses inline PTX (a mov on Ampere); that opaque form was exact
        // but materially slower in the captured shared-operand tournament.
        return __uint_as_float(static_cast<unsigned int>(__bfloat16_as_ushort(value)) << 16);
}

/**
 * @brief Project a row/column tile while preserving every serial FP32 operation.
 *
 * Lane L owns eight independent partials, corresponding to the original block
 * threads L, L+32, ..., L+224. Each partial visits K with stride 256. Staging
 * changes which threads fetch operands, never their arithmetic ownership.
 * Register reductions reproduce strides 128/64/32; warp shuffles reproduce
 * strides 16/8/4/2/1. This is consequently byte-compatible with both the serial
 * block projection and the one-output-per-warp projection.
 *
 * All threads participate in both barriers, even for ragged M/N tiles. The
 * first publishes the freshly loaded operands; the second prevents a fast
 * warp from overwriting the tile while a neighbor is still computing. A K
 * tail skips the absent multiply-add instead of introducing padded arithmetic.
 *
 * @tparam Weight Native persistent weight element type, not activation precision.
 * @tparam RowsPerCTA Compiled physical row tile: 2, 4 or 8.
 * @param activations Device pointer array of FP32 activation matrices [M,K].
 * @param weights Device pointer array using the existing floating-projection ABI;
 *                each address owns a native Weight matrix [N,K].
 * @param outputs Device pointer array of disjoint FP32 output matrices [M,N].
 * @param m Positive activation row count.
 * @param n Positive projection column count.
 * @param k Positive reduction width, including unaligned tails.
 * @pre Launch exactly 256 threads on an explicit non-null stream, with grid
 *      (ceil(N/8),ceil(M/RowsPerCTA),batch_count). Pointer arrays and matrices
 *      must be prepared, admitted, and alive for the complete captured replay.
 */
template<class Weight, int RowsPerCTA>
__global__ __launch_bounds__(256, 2) void sharedTinyProjectionKernel(
    const float *const *activations, const float *const *weights,
    float *const *outputs, int m, int n, int k)
{
    static_assert(RowsPerCTA == 2 || RowsPerCTA == 4 || RowsPerCTA == 8);
    constexpr int warp_width = 32;
    constexpr int columns_per_cta = 8;
    constexpr int arithmetic_parts = 256;
    constexpr int parts_per_lane = arithmetic_parts / warp_width;

    __shared__ float activation_tile[RowsPerCTA][arithmetic_parts];
    __shared__ float weight_tile[columns_per_cta][arithmetic_parts];
    const int lane = static_cast<int>(threadIdx.x) % warp_width;
    const int column = static_cast<int>(threadIdx.x) / warp_width;
    const int first_row = static_cast<int>(blockIdx.y) * RowsPerCTA;
    const int first_column = static_cast<int>(blockIdx.x) * columns_per_cta;
    const float *a = activations[blockIdx.z];
    const Weight *b = reinterpret_cast<const Weight *>(weights[blockIdx.z]);
    float *c = outputs[blockIdx.z];
    float partials[RowsPerCTA][parts_per_lane] = {};

    // The public width is a positive signed int. An unsigned induction value
    // can advance beyond its final partial tile without overflowing at INT_MAX;
    // the last valid address and every arithmetic guard still use the exact K.
    for (unsigned int start = 0; start < static_cast<unsigned int>(k); start += arithmetic_parts)
    {
        const unsigned int load_k = start + threadIdx.x;
        // Each operand element is fetched once per tile. Inactive output tails
        // still stage neutral storage so every warp can obey the same barriers.
        #pragma unroll
        for (int row = 0; row < RowsPerCTA; ++row)
            activation_tile[row][threadIdx.x] = first_row + row < m && load_k < static_cast<unsigned int>(k)
                ? a[static_cast<size_t>(first_row + row) * k + load_k] : 0.0f;
        #pragma unroll
        for (int col = 0; col < columns_per_cta; ++col)
            weight_tile[col][threadIdx.x] = first_column + col < n && load_k < static_cast<unsigned int>(k)
                ? tinyProjectionWeight(b[static_cast<size_t>(first_column + col) * k + load_k])
                : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int part = 0; part < parts_per_lane; ++part)
        {
            const int tile_k = lane + part * warp_width;
            if (start + tile_k < static_cast<unsigned int>(k))
            {
                const float weight = weight_tile[column][tile_k];
                #pragma unroll
                for (int row = 0; row < RowsPerCTA; ++row)
                    partials[row][part] += activation_tile[row][tile_k] * weight;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int row = 0; row < RowsPerCTA; ++row)
    {
        // Keep the same partial pairings as the legacy block reduction. Do not
        // collapse the eight accumulators or use an order-dependent atomic sum.
        #pragma unroll
        for (int stride = parts_per_lane / 2; stride > 0; stride >>= 1)
        {
            #pragma unroll
            for (int part = 0; part < stride; ++part)
                partials[row][part] += partials[row][part + stride];
        }
        float result = partials[row][0];
        #pragma unroll
        for (int stride = warp_width / 2; stride > 0; stride >>= 1)
            result += __shfl_down_sync(0xffffffffu, result, stride);
        if (lane == 0 && first_row + row < m && first_column + column < n)
            c[static_cast<size_t>(first_row + row) * n + first_column + column] = result;
    }
}
} // namespace llaminar2::cuda::detail
