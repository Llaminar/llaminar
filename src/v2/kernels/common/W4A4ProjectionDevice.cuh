/**
 * @file W4A4ProjectionDevice.cuh
 * @brief Shared CUDA/HIP native signed-dot4 projection over sealed whole-token A4.
 *
 * Eight lanes cooperate on each complete group64. Weight bytes and their code
 * sum are shared across a token tile; correction/multiplication/shift happen
 * before any group accumulation. Only the complete int32 result enters the
 * explicitly rounded FP32 epilogue. This is not a NativeVNNI codebook adapter.
 *
 * Cold backend binding must authenticate the original-K source policy, prepared
 * operand policy, exact extents, nonaliasing and retained arena/pool ownership.
 * The preparation producer seals physical-row validity and publishes its exact
 * event before this consumer. These borrowed addresses never own allocations,
 * read controller counts, publish readiness or replace those admission APIs.
 */
#pragma once

#if defined(__HIPCC__)
#include "kernels/rocm/common/ROCmSignedDot4.h"
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#else
#error "W4A4ProjectionDevice.cuh requires a CUDA or HIP device translation unit"
#endif

#include "kernels/common/DeviceFP32NumericalContract.h"
#include "kernels/common/W4A4DeviceViews.h"
#include <cstddef>
#include <cstdint>

// Distinct host-stub identities are required when both native backends link.
#if defined(__HIPCC__)
namespace llaminar2::w4a4::device::rocm
#else
namespace llaminar2::w4a4::device::cuda
#endif
{
    /** @brief Two signed-byte planes for eight adjacent signed source nibbles. */
    struct SignedPlanes
    {
        std::int32_t even; ///< Codes at original positions 0,2,4,6.
        std::int32_t odd;  ///< Codes at original positions 1,3,5,7.
    };

    /**
     * @brief Load four little-endian bytes without violating byte-array aliasing.
     * @param source Four readable bytes in one admitted native codes array.
     * @return Their packed word, independent of an untyped pointer cast.
     */
    __device__ __forceinline__ std::uint32_t loadFourBytes(const std::uint8_t *source)
    {
        return std::uint32_t(source[0]) | (std::uint32_t(source[1]) << 8) |
            (std::uint32_t(source[2]) << 16) | (std::uint32_t(source[3]) << 24);
    }

    /**
     * @brief Sign-extend adjacent int4 codes into independent signed-byte planes.
     * @param packed Four original bytes, never dequantized floating weights.
     * @return Even and odd planes suitable for native signed dot4.
     *
     * Every isolated sign bit times31 fills its own upper five byte bits. Those
     * fields do not overlap, so no cross-byte carry or subtraction borrow can
     * change a neighbour's nibble. Unsigned intermediate arithmetic is explicit.
     */
    __device__ __forceinline__ SignedPlanes signedPlanes(std::uint32_t packed)
    {
        const auto high = packed >> 4;
        const auto even = (packed & 0x07070707u) | ((packed & 0x08080808u) * 31u);
        const auto odd = (high & 0x07070707u) | ((high & 0x08080808u) * 31u);
        return {static_cast<std::int32_t>(even), static_cast<std::int32_t>(odd)};
    }

    /**
     * @brief Execute a native unsaturated signed-byte dot4 on either backend.
     * @param left Four signed-byte operands packed into one word.
     * @param right Corresponding four signed-byte operands.
     * @param initial Exact bounded integer accumulation before these products.
     * @return Initial plus all four signed products; no scalar GPU substitute.
     */
    __device__ __forceinline__ std::int32_t signedDot4(
        std::int32_t left, std::int32_t right, std::int32_t initial)
    {
#if defined(__HIPCC__)
        return rocm_integer_contract::signedDot4(left, right, initial);
#else
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
#error "Native W4A4 CUDA projection requires signed INT8 dot4"
#endif
        return __dp4a(left, right, initial);
#endif
    }

    /**
     * @brief Sum one complete eight-lane group with fixed native shuffle edges.
     * @param value This lane's contribution; all eight group lanes participate.
     * @return Complete sum in lane zero; other lanes are not publication owners.
     *
     * CUDA masks name only this eight-lane group, including a partial last N
     * tile. HIP width8 partitions both wave32 and wave64; it never assumes a
     * CUDA-sized wave or reads an adjacent output column's lanes.
     */
    __device__ __forceinline__ std::int32_t sumEight(std::int32_t value)
    {
#if defined(__HIPCC__)
        value += __shfl_down(value, 4, 8);
        value += __shfl_down(value, 2, 8);
        value += __shfl_down(value, 1, 8);
#else
        const unsigned mask = 0xffu << (threadIdx.x & 24u);
        value += __shfl_down_sync(mask, value, 4, 8);
        value += __shfl_down_sync(mask, value, 2, 8);
        value += __shfl_down_sync(mask, value, 1, 8);
#endif
        return value;
    }

    /**
     * @brief Fail on corrupted sealed validity instead of exposing stale A4 rows.
     * @param value Producer-published zero or one, read only on device.
     * @return Whether the original physical row has complete prepared operands.
     */
    __device__ __forceinline__ bool active(std::uint8_t value)
    {
        if (value > 1)
        {
#if defined(__HIPCC__)
            __builtin_trap();
#else
            __trap();
#endif
        }
        return value != 0;
    }

    /**
     * @brief Project a weight-sharing token tile with exact group-before-sum arithmetic.
     * @tparam Rows Token rows sharing each native weight load, one through four.
     * @param view Already-admitted immutable source and invocation addresses.
     *
     * Launch128 threads; x tiles16 output columns and y tilesRows physical
     * tokens. All eight lanes for an output either retire together or execute
     * every shuffle. The prepared validity bank, not mutable controller words,
     * chooses live rows. Empty tiles read no weight or activation payload.
     */
    template<int Rows>
    static __global__ __launch_bounds__(128) void project(ProjectionView view)
    {
        static_assert(Rows >= 1 && Rows <= 4);
        constexpr int kLanes = 8, kColumns = 128 / kLanes;
        const int lane = int(threadIdx.x) & (kLanes - 1);
        const int column = int(blockIdx.x) * kColumns + int(threadIdx.x) / kLanes;
        const int first = int(blockIdx.y) * Rows;
        if (column >= view.columns || first >= view.rows) return;

        bool live[Rows];
        std::int32_t totals[Rows]{};
        bool any = false;
#pragma unroll
        for (int row = 0; row < Rows; ++row)
        {
            live[row] = first + row < view.rows && active(view.active_rows[first + row]);
            any = any || live[row];
        }
        if (!any) return;

        const auto *weights = view.weights + std::size_t(column) * view.blocks_per_row;
        for (int block = 0; block < view.blocks_per_row; ++block)
        {
#pragma unroll
            for (unsigned group = 0; group < kGroupsPerBlock; ++group)
            {
                const int byte_offset = int(group) * (kGroupElements / 2) + lane * 4;
                const auto weight = signedPlanes(loadFourBytes(weights[block].codes + byte_offset));
                const auto weight_sum = sumEight(signedDot4(weight.even, 0x01010101,
                    signedDot4(weight.odd, 0x01010101, 0)));
#pragma unroll
                for (int row = 0; row < Rows; ++row)
                {
                    if (!live[row]) continue;
                    const auto &input = view.activations[
                        std::size_t(first + row) * view.blocks_per_row + block];
                    const auto activation = signedPlanes(loadFourBytes(input.codes + byte_offset));
                    const auto dot = sumEight(signedDot4(weight.even, activation.even,
                        signedDot4(weight.odd, activation.odd, 0)));
                    if (lane == 0)
                        totals[row] += shiftedGroupTerm(dot, weight_sum,
                            int((input.zero_points >> (4 * group)) & 15u),
                            int((input.multipliers >> (8 * group)) & 255u),
                            weightMultiplier(weights[block].descriptor, group));
                }
            }
        }
        if (lane != 0) return;
#pragma unroll
        for (int row = 0; row < Rows; ++row)
        {
            if (!live[row]) continue;
            const auto &input = view.activations[std::size_t(first + row) * view.blocks_per_row];
            // Each FP32 edge is explicit, including the pinned 16*A-unit*W-unit
            // order. The shared primitives retain subnormals under CUDA FTZ.
            float value = device_fp32_contract::multiply(float(totals[row]), 16.0f);
            value = device_fp32_contract::multiply(value, input.unit);
            value = device_fp32_contract::multiply(value, weights[0].unit);
            value = device_fp32_contract::multiply(value, view.alpha);
            auto *destination = view.output + std::size_t(first + row) * view.output_stride + column;
            // A CUDA FTZ comparison must not discard a nonzero subnormal beta.
            if ((device_fp32_contract::bitsFromFloat(view.beta) & 0x7fffffffu) != 0)
                value = device_fp32_contract::add(value,
                    device_fp32_contract::multiply(view.beta, *destination));
            if (view.bias) value = device_fp32_contract::add(value, view.bias[column]);
            *destination = value;
        }
    }
} // namespace llaminar2::w4a4::device::{cuda,rocm}
