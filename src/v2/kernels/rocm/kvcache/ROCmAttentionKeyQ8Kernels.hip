/**
 * @file ROCmAttentionKeyQ8Kernels.hip
 * @brief ROCm kernels for cubic-companded attention-key Q8 blocks.
 *
 * One HIP workgroup owns one complete attention head. Quantization uses a
 * deterministic wave reduction and rational codebook thresholds; decoding is
 * a fixed two-multiply polynomial. Caller-owned buffers and an explicit HIP
 * stream make both launches suitable for retained production graphs.
 */

#include "ROCmAttentionKeyQ8Kernels.h"

#include "tensors/BlockStructures.h"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace llaminar2
{
    namespace
    {
        constexpr int kWavefrontSize = 64;

        /** @brief Reduce one gfx906 wavefront to its maximum value. */
        __device__ __forceinline__ float waveMaximum(float value)
        {
            for (int offset = kWavefrontSize / 2; offset > 0; offset /= 2)
            {
                value = fmaxf(value, __shfl_down(value, offset, kWavefrontSize));
            }
            return value;
        }

        /** @brief Device mirror of the scalar rational cubic interval search. */
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

        /** @brief Encode one complete attention head on gfx906. */
        template <int D>
        __global__ __launch_bounds__(256, 2) void attentionKeyQ8QuantizeKernel(
            const float *__restrict__ input,
            AttentionKeyQ8Block<D> *__restrict__ output)
        {
            const int coordinate = threadIdx.x;
            const int block_index = blockIdx.x;
            const float value = input[static_cast<size_t>(block_index) * D + coordinate];

            __shared__ float wave_maxima[4];
            __shared__ float head_maximum;
            const int lane = coordinate % kWavefrontSize;
            const int wave = coordinate / kWavefrontSize;
            const int wave_count = D / kWavefrontSize;
            float maximum = waveMaximum(fabsf(value));
            if (lane == 0)
            {
                wave_maxima[wave] = maximum;
            }
            __syncthreads();

            if (wave == 0)
            {
                maximum = lane < wave_count ? wave_maxima[lane] : 0.0f;
                maximum = waveMaximum(maximum);
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

        /** @brief Decode one physical attention-key block on gfx906. */
        template <int D>
        __global__ __launch_bounds__(256, 4) void attentionKeyQ8DequantizeKernel(
            const AttentionKeyQ8Block<D> *__restrict__ input,
            float *__restrict__ output)
        {
            const int coordinate = threadIdx.x;
            const int block_index = blockIdx.x;
            const float q = static_cast<float>(input[block_index].codes[coordinate]);
            const float scale = input[block_index].quadratic_scale;
            output[static_cast<size_t>(block_index) * D + coordinate] =
                (scale * q) * fabsf(q);
        }

        /** @brief Launch the compile-time quantizer selected by a validated width. */
        template <int D>
        bool launchQuantize(
            const float *input,
            void *output,
            int block_count,
            hipStream_t stream)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(attentionKeyQ8QuantizeKernel<D>),
                dim3(block_count),
                dim3(D),
                0,
                stream,
                input,
                static_cast<AttentionKeyQ8Block<D> *>(output));
            return hipGetLastError() == hipSuccess;
        }

        /** @brief Launch the compile-time dequantizer selected by a validated width. */
        template <int D>
        bool launchDequantize(
            const void *input,
            float *output,
            int block_count,
            hipStream_t stream)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(attentionKeyQ8DequantizeKernel<D>),
                dim3(block_count),
                dim3(D),
                0,
                stream,
                static_cast<const AttentionKeyQ8Block<D> *>(input),
                output);
            return hipGetLastError() == hipSuccess;
        }
    } // namespace

    bool rocmAttentionKeyQ8Quantize(
        const float *input,
        void *output,
        int block_count,
        int head_dim,
        hipStream_t stream)
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

    bool rocmAttentionKeyQ8Dequantize(
        const void *input,
        float *output,
        int block_count,
        int head_dim,
        hipStream_t stream)
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
