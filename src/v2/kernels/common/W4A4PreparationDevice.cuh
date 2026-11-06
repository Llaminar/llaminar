/**
 * @file W4A4PreparationDevice.cuh
 * @brief Device-resident original-row sealing and exact whole-token A4 packing.
 *
 * The values input is either the original FP32 row for W4A4 or the complete,
 * ordered transform output for W4A4H. Transform production is a separate event
 * edge, never a local-K approximation. A single block owns each full token's
 * ranges and unit; no host count readback, global reduction atomic, allocation
 * or mutable execution-state mirror participates in preparation.
 *
 * Backend launchers must bind admitted nonoverlapping arenas, exact physical
 * strides and source policy before capture. Source/transform policy and every
 * pointer belong to capture identity. These device bodies do not admit sources,
 * allocate storage, publish readiness or expose an execution entrypoint.
 */
#pragma once

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#else
#error "W4A4PreparationDevice.cuh requires a CUDA or HIP device translation unit"
#endif

#include "kernels/common/DeviceFP32NumericalContract.h"
#include "kernels/common/ProjectionRowWork.h"
#include "kernels/common/W4A4DeviceViews.h"
#include <cstddef>
#include <cstdint>

// Keep CUDA and HIP registration stubs distinct in full-backend binaries.
#if defined(__HIPCC__)
namespace llaminar2::w4a4::device::rocm
#else
namespace llaminar2::w4a4::device::cuda
#endif
{
    /**
     * @brief Reject corrupt policy, inputs or intermediate values on the executing device.
     * @param condition Required invariant; failure is fatal, never clamped into valid work.
     */
    __device__ __forceinline__ void requirePreparation(bool condition)
    {
        if (condition) return;
#if defined(__HIPCC__)
        __builtin_trap();
#else
        __trap();
#endif
    }

    /**
     * @brief Negate a binary32 word without introducing an FTZ arithmetic operation.
     * @param value Original operand, including a signed zero or subnormal.
     * @return Identical magnitude with only the sign bit inverted.
     */
    __device__ __forceinline__ float negateExact(float value)
    {
        return device_fp32_contract::floatFromBits(
            device_fp32_contract::bitsFromFloat(value) ^ 0x80000000u);
    }

    /**
     * @brief Divide with the pinned correctly rounded binary32 operation.
     * @param numerator Original numerator, including zero or an overflowed positive infinity.
     * @param denominator Positive finite denominator.
     * @return RNE division result, retaining subnormals and the completed rounding edge.
     */
    __device__ __forceinline__ float divideExact(float numerator, float denominator)
    {
#if defined(__HIPCC__)
        return device_fp32_contract::persistRounded(__fdiv_rn(numerator, denominator));
#else
        float result;
        asm("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(numerator), "f"(denominator));
        return device_fp32_contract::persistRounded(result);
#endif
    }

    /**
     * @brief Round to the nearest integral float, with ties to even, before saturation.
     * @param value Original scaled value, potentially an overflowed infinity.
     * @return Rounded float; infinity remains infinity until the following clamp.
     */
    __device__ __forceinline__ float roundEvenFloat(float value)
    {
#if defined(__HIPCC__)
        return device_fp32_contract::persistRounded(rintf(value));
#else
        float result;
        asm("cvt.rni.f32.f32 %0, %1;" : "=f"(result) : "f"(value));
        return result;
#endif
    }

    /**
     * @brief Round upward without changing the expression that reaches ceil.
     * @param value Nonnegative group-width ratio, potentially positive infinity.
     * @return Integral float rounded toward positive infinity.
     */
    __device__ __forceinline__ float ceilFloat(float value)
    {
#if defined(__HIPCC__)
        return device_fp32_contract::persistRounded(ceilf(value));
#else
        float result;
        asm("cvt.rpi.f32.f32 %0, %1;" : "=f"(result) : "f"(value));
        return result;
#endif
    }

    /**
     * @brief Clamp a floating result to a nonnegative integral interval without FTZ comparisons.
     * @param value Rounded input; NaN is invalid, infinities saturate like the pinned implementation.
     * @param lower Exactly representable nonnegative lower endpoint.
     * @param upper Exactly representable positive upper endpoint.
     * @return Saturated float, safe for the following bounded integer conversion.
     */
    __device__ __forceinline__ float clampIntegral(float value, float lower, float upper)
    {
        const auto bits = device_fp32_contract::bitsFromFloat(value);
        const auto magnitude = bits & 0x7fffffffu;
        requirePreparation(magnitude <= 0x7f800000u);
        if ((bits & 0x80000000u) || magnitude < device_fp32_contract::bitsFromFloat(lower)) return lower;
        if (magnitude > device_fp32_contract::bitsFromFloat(upper)) return upper;
        return value;
    }

    /**
     * @brief Inspect finite positive binary32 without a flushing floating comparison.
     * @param value Candidate unit or reciprocal.
     * @return True only for finite strictly positive values, including positive subnormals.
     */
    __device__ __forceinline__ bool positiveFinite(float value)
    {
        const auto bits = device_fp32_contract::bitsFromFloat(value);
        return bits != 0 && bits < 0x7f800000u;
    }

    /**
     * @brief Encode a finite value in unsigned order while canonicalizing only extrema zeros.
     * @param value Original input; the transformed/input array itself is never modified.
     * @return Monotonic unsigned key; both zeros equal the initial positive-zero anchor.
     */
    __device__ __forceinline__ std::uint32_t extremaKey(float value)
    {
        auto bits = device_fp32_contract::bitsFromFloat(value);
        requirePreparation((bits & 0x7fffffffu) < 0x7f800000u);
        if ((bits & 0x7fffffffu) == 0) bits = 0;
        return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
    }

    /**
     * @brief Recover the exact value selected by an extrema key.
     * @param key Key produced by extremaKey or the zero anchor.
     * @return Original selected nonzero value, or positive zero.
     */
    __device__ __forceinline__ float extremaValue(std::uint32_t key)
    {
        return device_fp32_contract::floatFromBits(key & 0x80000000u ? key ^ 0x80000000u : ~key);
    }

    /**
     * @brief Publish each original physical row's validity from its device-owned request once.
     * @param work Borrowed original request publication, already cold-admitted.
     * @param validity Exact output bank with work.physicalRows() bytes.
     *
     * Every row, including empty work, overwrites its validity byte. Subsequent
     * transform, packing and projection kernels consume this bank and never
     * consult the mutable request counts again for this invocation.
     */
    static __global__ void sealRows(ProjectionRowWork work, std::uint8_t *validity)
    {
        const int row = int(blockIdx.x * blockDim.x + threadIdx.x);
        if (row >= work.physicalRows()) return;
        const auto span = work.resolveDevice(work.requestForRow(row));
        validity[row] = std::uint8_t(span.contains(row));
    }

    /**
     * @brief Pack one whole-token row per256-thread block after its transform producer completes.
     * @param view Exact prepared-value, validity and A4 arena addresses.
     *
     * Dynamic shared storage contains columns/64 GroupPreparation objects.
     * Each group's extrema have a unique writer. The whole-token maximum uses
     * integer ordering of nonnegative width words, so neither scheduling nor
     * floating reduction reassociation can select a different unit. A packed
     * native code word and its block header each have exactly one writer.
     */
    static __global__ __launch_bounds__(256) void packRows(PreparationView view)
    {
        const int row = int(blockIdx.x), thread = int(threadIdx.x);
        if (row >= view.rows) return;
        const auto validity = view.active_rows[row];
        requirePreparation(validity <= 1);
        if (!validity) return;
        const int groups = view.columns / kGroupElements;
        extern __shared__ GroupPreparation ranges[];
        __shared__ std::uint32_t maxima[256];
        __shared__ float row_width, unit;
        const auto *values = view.values + std::size_t(row) * view.input_stride;

        std::uint32_t maximum = 0;
        for (int group = thread; group < groups; group += 256)
        {
            std::uint32_t lower = 0x80000000u, upper = 0x80000000u;
            for (int index = 0; index < kGroupElements; ++index)
            {
                const auto key = extremaKey(values[group * kGroupElements + index]);
                lower = key < lower ? key : lower;
                upper = key > upper ? key : upper;
            }
            auto &range = ranges[group];
            range.lower = device_fp32_contract::multiply(extremaValue(lower), view.clip);
            range.upper = device_fp32_contract::multiply(extremaValue(upper), view.clip);
            const auto width = device_fp32_contract::add(range.upper, negateExact(range.lower));
            const auto bits = device_fp32_contract::bitsFromFloat(width);
            requirePreparation(bits < 0x7f800000u);
            maximum = bits > maximum ? bits : maximum;
        }
        maxima[thread] = maximum;
        __syncthreads();
        for (int stride = 128; stride != 0; stride >>= 1)
        {
            if (thread < stride && maxima[thread + stride] > maxima[thread])
                maxima[thread] = maxima[thread + stride];
            __syncthreads();
        }
        if (thread == 0)
        {
            row_width = maxima[0] ? device_fp32_contract::floatFromBits(maxima[0]) : 2.0f;
            unit = divideExact(row_width, float(15 * kActivationMultiplierMax));
            requirePreparation(positiveFinite(unit));
        }
        __syncthreads();
        for (int group = thread; group < groups; group += 256)
        {
            auto &range = ranges[group];
            const auto width = device_fp32_contract::add(range.upper, negateExact(range.lower));
            const auto multiplier = clampIntegral(ceilFloat(divideExact(
                device_fp32_contract::multiply(width, float(kActivationMultiplierMax)), row_width)),
                1.0f, float(kActivationMultiplierMax));
            range.inverse = divideExact(1.0f, device_fp32_contract::multiply(multiplier, unit));
            requirePreparation(positiveFinite(range.inverse));
            const auto zero = clampIntegral(roundEvenFloat(device_fp32_contract::multiply(
                negateExact(range.lower), range.inverse)), 0.0f, 15.0f);
            range.quantizer = std::uint32_t(multiplier) | (std::uint32_t(zero) << 8);
        }
        __syncthreads();
        auto *blocks = view.blocks + std::size_t(row) * (view.columns / kBlockElements);
        for (int word = thread; word < view.columns / 8; word += 256)
        {
            const int block = word / 32, word_in_block = word % 32;
            const auto &range = ranges[word / 8];
            const auto zero = float((range.quantizer >> 8) & 15u);
            std::uint32_t packed = 0;
#pragma unroll
            for (int index = 0; index < 8; ++index)
            {
                const auto rounded = roundEvenFloat(device_fp32_contract::multiply(
                    values[word * 8 + index], range.inverse));
                const auto code = clampIntegral(device_fp32_contract::add(rounded, zero), 0.0f, 15.0f);
                packed |= std::uint32_t((int(code) - 8) & 15) << (4 * index);
            }
#pragma unroll
            for (int byte = 0; byte < 4; ++byte)
                blocks[block].codes[word_in_block * 4 + byte] = std::uint8_t(packed >> (8 * byte));
            if (word_in_block == 0)
            {
                auto &output = blocks[block];
                output.unit = unit;
                output.rotation_length = view.rotation_length;
                std::uint32_t zero_points = kActivationGroupTag, multipliers = 0;
#pragma unroll
                for (int group = 0; group < kGroupsPerBlock; ++group)
                {
                    const auto quantizer = ranges[block * kGroupsPerBlock + group].quantizer;
                    zero_points |= ((quantizer >> 8) & 15u) << (4 * group);
                    multipliers |= (quantizer & 255u) << (8 * group);
                }
                output.zero_points = zero_points;
                output.multipliers = multipliers;
            }
        }
    }
} // namespace llaminar2::w4a4::device::{cuda,rocm}
