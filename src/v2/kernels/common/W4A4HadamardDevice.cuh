/**
 * @file W4A4HadamardDevice.cuh
 * @brief Ordered native W4A4H transforms over sealed original physical rows.
 *
 * Sylvester pairs retain increasing-stride FP32 edges. The following Paley
 * producer accumulates original source rows in ascending order and applies the
 * immutable pinned normalization only once. A pure Sylvester row bypasses the
 * zero-initialized Paley sum, preserving signed zero. No input or source policy
 * is replaced with local-K transforms or a different arithmetic reduction.
 *
 * Backend binding owns the explicit shared-block or global-stage launch policy,
 * native shared-memory limits, immutable residue bits and both persistent arena
 * banks. Global stages are separate ordered nodes for large Sylvester geometry;
 * they are not an eager retry of the shared path. Every address, geometry,
 * normalization bit pattern and launch policy participates in capture identity.
 * These borrowed device views do not allocate, admit or publish their storage.
 */
#pragma once

#include "kernels/common/W4A4PreparationDevice.cuh"

#if defined(__HIPCC__)
namespace llaminar2::w4a4::device::rocm
#else
namespace llaminar2::w4a4::device::cuda
#endif
{
    /**
     * @brief Finish one complete Sylvester subrow in block-local shared storage.
     * @param view Sealed source geometry, input and arena addresses.
     *
     * Launch256 threads, x=(columns/rotation_length)*paley and y=rows, with
     * sylvester*sizeof(float) native dynamic shared bytes. Cold binding must
     * prove that exact native shared-memory geometry before recording. One
     * thread owns each butterfly pair; the block barrier publishes every stage
     * before any pair of the following stage reads it.
     */
    static __global__ __launch_bounds__(256) void sylvesterShared(HadamardView view)
    {
        const int row = int(blockIdx.y), subrow = int(blockIdx.x), thread = int(threadIdx.x);
        if (row >= view.rows || subrow >= view.columns / view.sylvester) return;
        const auto validity = view.active_rows[row];
        requirePreparation(validity <= 1);
        if (!validity) return;
        extern __shared__ float values[];
        const auto *input = view.input + std::size_t(row) * view.input_stride + subrow * view.sylvester;
        for (int index = thread; index < view.sylvester; index += 256) values[index] = input[index];
        __syncthreads();
        for (int stride = 1; stride < view.sylvester; stride *= 2)
        {
            for (int pair = thread; pair < view.sylvester / 2; pair += 256)
            {
                const int first = (pair / stride) * (2 * stride) + pair % stride;
                const auto lower = values[first], upper = values[first + stride];
                values[first] = device_fp32_contract::add(lower, upper);
                values[first + stride] = device_fp32_contract::add(lower, negateExact(upper));
            }
            __syncthreads();
        }
        auto *output = view.butterfly + std::size_t(row) * view.columns + subrow * view.sylvester;
        for (int index = thread; index < view.sylvester; index += 256) output[index] = values[index];
    }

    /**
     * @brief Execute one complete globally stored Sylvester stage for large geometry.
     * @param view Original input and already-admitted full-row scratch.
     * @param stride Positive power-of-two stage distance, strictly less than sylvester.
     *
     * Launch256 threads and ceil(columns/512) blocks per original row. Stage1
     * consumes original FP32 input and initializes the complete scratch bank;
     * later stages consume that bank in place. Pairs are disjoint even across
     * blocks. The exact stream/graph edge, not a host synchronization, orders
     * increasing stages before Paley normalization.
     */
    static __global__ __launch_bounds__(256) void sylvesterGlobalStage(HadamardView view, int stride)
    {
        const int row = int(blockIdx.y), pair = int(blockIdx.x * blockDim.x + threadIdx.x);
        if (row >= view.rows || pair >= view.columns / 2) return;
        const auto validity = view.active_rows[row];
        requirePreparation(validity <= 1);
        if (!validity) return;
        requirePreparation(stride > 0 && (stride & (stride - 1)) == 0 && stride < view.sylvester);
        const int pairs_per_subrow = view.sylvester / 2;
        const int subrow = pair / pairs_per_subrow, local_pair = pair % pairs_per_subrow;
        const int first = subrow * view.sylvester +
            (local_pair / stride) * (2 * stride) + local_pair % stride;
        const float *input = stride == 1 ? view.input + std::size_t(row) * view.input_stride :
            view.butterfly + std::size_t(row) * view.columns;
        auto *output = view.butterfly + std::size_t(row) * view.columns;
        const auto lower = input[first], upper = input[first + stride];
        output[first] = device_fp32_contract::add(lower, upper);
        output[first + stride] = device_fp32_contract::add(lower, negateExact(upper));
    }

    /**
     * @brief Publish the exact ordered Paley result and final FP32 normalization.
     * @param view Completed Sylvester bank, immutable residue bits and final transform bank.
     *
     * Launch256 threads, ceil(columns/256) x blocks and rows y blocks. Output
     * elements are independent, but each one's source-row arithmetic order is
     * fixed. No atomic or partition-dependent floating reduction is permitted.
     * Inactive rows retain their prior scratch and destination guards.
     */
    static __global__ __launch_bounds__(256) void paleyNormalize(HadamardView view)
    {
        const int row = int(blockIdx.y), index = int(blockIdx.x * blockDim.x + threadIdx.x);
        if (row >= view.rows || index >= view.columns) return;
        const auto validity = view.active_rows[row];
        requirePreparation(validity <= 1);
        if (!validity) return;
        const auto row_offset = std::size_t(row) * view.columns;
        if (view.paley == 1)
        {
            view.transformed[row_offset + index] = device_fp32_contract::multiply(
                view.butterfly[row_offset + index], view.normalization);
            return;
        }
        const int part = index / view.rotation_length;
        const int local = index % view.rotation_length;
        const int column = local / view.sylvester, element = local % view.sylvester;
        const auto *input = view.butterfly + row_offset + part * view.rotation_length;
        float value = 0.0f;
        for (int source = 0; source < view.paley; ++source)
        {
            int difference = column - source;
            if (difference < 0) difference += view.paley - 1;
            // Admission bounds Paley order by128. A fixed immutable bitset
            // carries this arithmetic policy without a table allocation,
            // upload or independent initialization/publication authority.
            const auto residue_word = difference < 64 ? view.residues_low : view.residues_high;
            const bool residue = (residue_word >> (difference & 63)) & 1u;
            const bool positive = source == column || source == 0 || (column != 0 && residue);
            const auto operand = input[source * view.sylvester + element];
            value = device_fp32_contract::add(value, positive ? operand : negateExact(operand));
        }
        view.transformed[row_offset + index] = device_fp32_contract::multiply(value, view.normalization);
    }
} // namespace llaminar2::w4a4::device::{cuda,rocm}
