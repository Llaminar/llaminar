/**
 * @file CUDAAttentionKeyQ8Kernels.cu
 * @brief CUDA kernels for cubic-companded attention-key Q8 blocks.
 *
 * One CUDA block owns one complete attention head. Quantization performs a
 * deterministic max reduction followed by seven rational codebook-threshold
 * comparisons per coordinate. Dequantization uses only two multiplies per
 * coordinate. Both paths use persistent caller-owned buffers and an explicit
 * stream, making them safe to place inside a retained CUDA graph.
 */

#include "CUDAAttentionKeyQ8Kernels.h"

#include "tensors/BlockStructures.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace llaminar2
{
    namespace
    {
        /** @brief Reduce one warp to its maximum finite non-negative value. */
        __device__ __forceinline__ float warpMaximum(float value)
        {
            for (int offset = warpSize / 2; offset > 0; offset /= 2)
            {
                value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
            }
            return value;
        }

        /**
         * @brief Encode one non-negative magnitude using rational cubic thresholds.
         *
         * This is the device mirror of
         * `attentionKeyQ8MagnitudeCodeReference()`. Integer threshold squares and
         * float multiplication order intentionally match the scalar oracle.
         */
        __device__ __forceinline__ int8_t encodeMagnitude(float magnitude, float max_abs)
        {
            constexpr int kThresholdDenominator = 254;
            constexpr int kThresholdDenominatorSquared =
                kThresholdDenominator * kThresholdDenominator;
            const float scaled_magnitude =
                magnitude * static_cast<float>(kThresholdDenominatorSquared);

            int lower = 0;
            int upper = 127;
#pragma unroll
            for (int step = 0; step < 7; ++step)
            {
                const int midpoint = (lower + upper) / 2;
                const int odd_boundary = 2 * midpoint + 1;
                const int odd_boundary_squared = odd_boundary * odd_boundary;
                const float scaled_boundary =
                    max_abs * static_cast<float>(odd_boundary_squared);
                if (scaled_magnitude < scaled_boundary)
                {
                    upper = midpoint;
                }
                else
                {
                    lower = midpoint + 1;
                }
            }
            return static_cast<int8_t>(lower);
        }

        /**
         * @brief Encode one complete head into its blob-transferable physical block.
         *
         * @tparam D Compile-time attention-head width.
         */
        template <int D>
        __global__ __launch_bounds__(256, 2) void attentionKeyQ8QuantizeKernel(
            const float *__restrict__ input,
            AttentionKeyQ8Block<D> *__restrict__ output)
        {
            const int coordinate = threadIdx.x;
            const int block_index = blockIdx.x;
            const float value = input[static_cast<size_t>(block_index) * D + coordinate];

            // Each warp publishes one maximum. Warp zero then reduces those
            // values, avoiding atomics and giving every block one fixed writer.
            __shared__ float warp_maxima[8];
            __shared__ float head_maximum;
            const int lane = coordinate % warpSize;
            const int warp = coordinate / warpSize;
            const int warp_count = D / warpSize;
            float maximum = warpMaximum(fabsf(value));
            if (lane == 0)
            {
                warp_maxima[warp] = maximum;
            }
            __syncthreads();

            if (warp == 0)
            {
                maximum = lane < warp_count ? warp_maxima[lane] : 0.0f;
                maximum = warpMaximum(maximum);
                if (lane == 0)
                {
                    head_maximum = maximum;
                    output[block_index].quadratic_scale =
                        maximum * AttentionKeyQ8Block<D>::INVERSE_MAX_CODE_SQUARED;
                }
            }
            __syncthreads();

            if (head_maximum == 0.0f)
            {
                output[block_index].codes[coordinate] = 0;
                return;
            }

            const int8_t magnitude_code = encodeMagnitude(fabsf(value), head_maximum);
            output[block_index].codes[coordinate] =
                value < 0.0f ? static_cast<int8_t>(-magnitude_code) : magnitude_code;
        }

        /**
         * @brief Decode one complete physical block to a contiguous FP32 head.
         *
         * @tparam D Compile-time attention-head width.
         */
        template <int D>
        __global__ __launch_bounds__(256, 4) void attentionKeyQ8DequantizeKernel(
            const AttentionKeyQ8Block<D> *__restrict__ input,
            float *__restrict__ output)
        {
            const int coordinate = threadIdx.x;
            const int block_index = blockIdx.x;
            const float q = static_cast<float>(input[block_index].codes[coordinate]);
            const float scale = input[block_index].quadratic_scale;
            // Preserve the scalar oracle's explicit multiplication order.
            output[static_cast<size_t>(block_index) * D + coordinate] =
                (scale * q) * fabsf(q);
        }

        /** @brief Launch the compile-time quantizer selected by a validated width. */
        template <int D>
        bool launchQuantize(
            const float *input,
            void *output,
            int block_count,
            cudaStream_t stream)
        {
            attentionKeyQ8QuantizeKernel<D><<<block_count, D, 0, stream>>>(
                input,
                static_cast<AttentionKeyQ8Block<D> *>(output));
            return cudaGetLastError() == cudaSuccess;
        }

        /** @brief Launch the compile-time dequantizer selected by a validated width. */
        template <int D>
        bool launchDequantize(
            const void *input,
            float *output,
            int block_count,
            cudaStream_t stream)
        {
            attentionKeyQ8DequantizeKernel<D><<<block_count, D, 0, stream>>>(
                static_cast<const AttentionKeyQ8Block<D> *>(input),
                output);
            return cudaGetLastError() == cudaSuccess;
        }
    } // namespace

    bool cudaAttentionKeyQ8Quantize(
        const float *input,
        void *output,
        int block_count,
        int head_dim,
        cudaStream_t stream)
    {
        if (!input || !output || block_count <= 0 || !stream)
        {
            return false;
        }
        switch (head_dim)
        {
        case 64:
            return launchQuantize<64>(input, output, block_count, stream);
        case 128:
            return launchQuantize<128>(input, output, block_count, stream);
        case 256:
            return launchQuantize<256>(input, output, block_count, stream);
        default:
            return false;
        }
    }

    bool cudaAttentionKeyQ8Dequantize(
        const void *input,
        float *output,
        int block_count,
        int head_dim,
        cudaStream_t stream)
    {
        if (!input || !output || block_count <= 0 || !stream)
        {
            return false;
        }
        switch (head_dim)
        {
        case 64:
            return launchDequantize<64>(input, output, block_count, stream);
        case 128:
            return launchDequantize<128>(input, output, block_count, stream);
        case 256:
            return launchDequantize<256>(input, output, block_count, stream);
        default:
            return false;
        }
    }
} // namespace llaminar2
