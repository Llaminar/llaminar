/**
 * @file DeviceRouterHadamardQ8.h
 * @brief Allocation-free signed H32 followed by the canonical first Q8 conversion.
 *
 * One 32-lane subgroup owns an original floating block. Both weight and hidden
 * rows execute x D H / sqrt(32), preserving the unquantized dot product. Signs
 * are fixed before measurement (mt19937_64 seed31, first32 LSB0 means minus).
 * The ascending butterfly uses explicit FP32 RNE edges and no private arrays.
 * Eight subgroups share a workgroup for large matrices, including both halves
 * of an AMD wave64. Only the retained Q8 bank is written; source rows stay intact.
 */
#pragma once
#include <cstddef>
#include "DeviceFP32NumericalContract.h"
#include "DeviceQ8ActivationNumericalContract.h"
#include "RouterQ8Launch.h"

namespace llaminar2::router_hadamard_q8
{
    /** @brief Fixed signed Hadamard matrix, expressed without a constant-memory table. */
    inline constexpr unsigned negative_signs = 0x624970a4u;

    /** @brief Shuffle inside one 32-lane mathematical block on either GPU ISA.
     * @param value Current butterfly value. @param mask Ascending butterfly stride.
     * @return The paired coordinate, with no adjacent wave64 subgroup mixing. */
    __device__ __forceinline__ float exchange(float value, int mask)
    {
#if defined(__HIPCC__)
        return __shfl_xor(value, mask, 32);
#else
        return __shfl_xor_sync(0xffffffffu, value, mask, 32);
#endif
    }

    /** @brief Quantize only live rows in the paired H32 basis.
     * @param source Original immutable FP32 rows, never dequantized codebook values.
     * @param codes Retained Q8 destination. @param scales Canonical block32 scales.
     * @param rows Captured row capacity. @param width Positive multiple of 32.
     * @param live_rows Optional device-owned active row count, clamped to capacity. */
    static __global__ void quantize_signed_hadamard32(const float *__restrict__ source,
        std::int8_t *__restrict__ codes, float *__restrict__ scales,
        int rows, int width, const int *__restrict__ live_rows)
    {
        const int lane = threadIdx.x % 32;
        const int block = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
        const int row = blockIdx.y;
        const int live = live_rows ? max(0, min(rows, *live_rows)) : rows;
        if (row >= live || block >= width / 32) return;
        const std::size_t column = std::size_t(row) * width + block * 32 + lane;
        const std::size_t scale_index = std::size_t(row) * (width / 32) + block;
        namespace fp = device_fp32_contract;
        float value = fp::floatFromBits(fp::bitsFromFloat(source[column]) ^
            (((negative_signs >> lane) & 1u) << 31u));
#pragma unroll
        for (int stride = 1; stride < 32; stride *= 2)
        {
            const float other = exchange(value, stride);
            value = (lane & stride) ? fp::add(other, -value) : fp::add(value, other);
        }
        value = fp::multiply(value, 0x1.6a09e6p-3f);
        float maximum = device_q8_activation_contract::maximumAbsoluteValue(value, 0.0f);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            maximum = device_q8_activation_contract::maximumAbsoluteValue(maximum, exchange(maximum, mask));
        const float scale = device_q8_activation_contract::scale(maximum);
        if (lane == 0) scales[scale_index] = scale;
        codes[column] = static_cast<std::int8_t>(device_q8_activation_contract::quantize(value, scale));
    }

    /** @brief Enqueue the shared transform using persistent output bindings.
     * @param source Original FP32 rows. @param codes Retained Q8 bank.
     * @param scales Retained scales. @param rows Captured capacity.
     * @param width Hidden width. @param stream Exact non-null native stream.
     * @param live_rows Optional device live count. @return Native enqueue success. */
    static inline bool launch(const float *source, std::int8_t *codes, float *scales,
        int rows, int width, void *stream, const int *live_rows = nullptr)
    {
        if (!stream || !source || !codes || !scales || rows <= 0 || width <= 0 || width % 32) return false;
        const int groups = rows > 16 ? 8 : 1;
        const dim3 grid((width / 32 + groups - 1) / groups, rows);
#if defined(__HIPCC__)
        hipLaunchKernelGGL(quantize_signed_hadamard32, grid, dim3(groups * 32), 0,
            static_cast<hipStream_t>(stream), source, codes, scales, rows, width, live_rows);
        return hipGetLastError() == hipSuccess;
#else
        quantize_signed_hadamard32<<<grid, groups * 32, 0, static_cast<cudaStream_t>(stream)>>>(
            source, codes, scales, rows, width, live_rows);
        return cudaGetLastError() == cudaSuccess;
#endif
    }
}
