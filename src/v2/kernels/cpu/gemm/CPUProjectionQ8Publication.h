/**
 * @file CPUProjectionQ8Publication.h
 * @brief Shared original-row activation publication for native CPU projections.
 *
 * The arithmetic policy owns Q8 scale/rounding: ordinary CPU activation blocks
 * retain the backend-native SIMD program; heterogeneous experts retain the
 * canonical device-aligned program. Source/frontier words and physical row
 * addresses are borrowed unchanged. Destination Q8 storage is invocation-owned,
 * admitted before execution, and never retained, allocated or rebound here.
 */
#pragma once
#include "CPUProjectionRowGrid.h"
#include "CPUProjectionStorageProof.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "kernels/cpu/primitives/GPUAlignedExpertQ8Primitives.h"
#include "tensors/SIMDHelpers.h"
#include "utils/OpenMPUtils.h"
#include "kernels/cpu/primitives/SwiGLUPrimitives.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace llaminar2::cpu::native_quantised
{
    /**
     * @brief Publish one Q8 block under the matrix's arithmetic policy.
     * @param source Source FP32 values.
     * @param destination Destination Q8_1 block.
     * @param valid_elements Number of valid source values.
     * @param policy Backend-native or placement-invariant publication.
     * @throws std::invalid_argument For an unknown policy or invalid live extent.
     */
    inline void quantizeActivationBlock(
        const float *source,
        Q8_1Block &destination,
        int valid_elements,
        CPUProjectionNumericalPolicy policy)
    {
        if (valid_elements < 0 || valid_elements > 32)
            throw std::invalid_argument("CPU activation publication requires a live block extent in [0,32]");
        if (policy != CPUProjectionNumericalPolicy::BackendNative && policy != CPUProjectionNumericalPolicy::GPUAlignedExpert)
            throw std::invalid_argument("CPU activation publication requires a known arithmetic policy");
        if (policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
        {
            gpu_aligned_expert_q8::quantizeBlock(
                source, destination, valid_elements);
            return;
        }
        simd::quantize_single_block(
            source, destination, valid_elements);
    }

    /**
     * @brief Publish two complete adjacent Q8 blocks under one policy.
     * @param source Source pointer containing 64 FP32 values.
     * @param first First destination block.
     * @param second Second destination block.
     * @param policy Backend-native or placement-invariant publication.
     * @throws std::invalid_argument For an unknown arithmetic policy.
     */
    inline void quantizeTwoActivationBlocks(
        const float *source,
        Q8_1Block &first,
        Q8_1Block &second,
        CPUProjectionNumericalPolicy policy)
    {
        if (policy != CPUProjectionNumericalPolicy::BackendNative && policy != CPUProjectionNumericalPolicy::GPUAlignedExpert)
            throw std::invalid_argument("CPU activation publication requires a known arithmetic policy");
        if (policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
        {
            gpu_aligned_expert_q8::quantizeTwoBlocks(source, first, second);
            return;
        }
#if defined(__AVX512F__)
        simd::quantize_two_blocks_avx512(source, first, second);
#else
        simd::quantize_single_block(source, first);
        simd::quantize_single_block(
            source + Q8_1Block::BLOCK_SIZE, second);
#endif
    }


    /**
     * @brief Quantize only authored original activation rows into admitted Q8 storage.
     * @param A_fp32 Original immutable matrix with physical M-by-K coordinates.
     * @param A_q8 Invocation-owned physical-M Q8 bank; unused rows stay dirty.
     * @param M Original physical row capacity, never useful row count.
     * @param K Original logical input width.
     * @param K_blocks Original ceil(K/32) blocks per physical row.
     * @param policy Existing per-block quantization arithmetic, unchanged.
     * @param work Optional explicit original-row selection borrowed before OpenMP.
     * @throws std::invalid_argument For inconsistent work/publication geometry.
     */
    inline void quantize_activations_to_q8_1(
        const float *A_fp32,
        Q8_1Block *A_q8,
        int M,
        int K,
        int K_blocks,
        CPUProjectionNumericalPolicy policy =
            CPUProjectionNumericalPolicy::BackendNative,
        const ProjectionRowWork *work = nullptr)
    {
        const CPUProjectionRowGrid row_grid(M,work);
        if (K <= 0 || std::int64_t(K_blocks) != (std::int64_t(K) + 31) / 32 ||
            (policy != CPUProjectionNumericalPolicy::BackendNative && policy != CPUProjectionNumericalPolicy::GPUAlignedExpert))
            throw std::invalid_argument("CPU activation rows require exact K-block geometry and arithmetic policy");
        if(!row_grid.beginWorkshare())return;
        if (!A_fp32 || !A_q8)
            throw std::invalid_argument("CPU activation publication requires retained source and destination storage");

        auto do_quantize = [&]()
        {
#pragma omp for schedule(static)
            for (int m = 0; m < M; ++m)
            {
                if(!row_grid.active(m))continue;
                const float *row_a = A_fp32 + std::size_t(m) * K;
                Q8_1Block *row_q8 = A_q8 + static_cast<size_t>(m) * K_blocks;
                int kb = 0;
#if defined(__AVX512F__)
                if (K % 32 == 0)
                {
                    for (; kb + 1 < K_blocks; kb += 2)
                    {
                        quantizeTwoActivationBlocks(
                            row_a + kb * 32,
                            row_q8[kb],
                            row_q8[kb + 1],
                            policy);
                    }
                }
#endif
                for (; kb < K_blocks; ++kb)
                {
                    int block_start = kb * 32;
                    int block_len = std::min(32, K - block_start);
                    quantizeActivationBlock(
                        row_a + block_start,
                        row_q8[kb],
                        block_len,
                        policy);
                }
            }
        };

        OMP_WORKSHARE_REGION(do_quantize);
    }

    /**
     * @brief Apply SwiGLU and publish its exact Q8_1 rows in one workshare.
     *
     * Routed CPU MoE execution needs the FP32 gate/up projections only long
     * enough to form the activation consumed by the expert down projection.
     * Materializing that activation in a third FP32 matrix and then opening a
     * second OpenMP region to quantize it adds avoidable memory traffic and a
     * team boundary to every MoE layer. This helper instead owns independent
     * 64-element tasks: each task computes two canonical 32-element Q8_1 blocks
     * and publishes them directly to the down-projection input.
     *
     * The arithmetic is intentionally identical to the unfused production
     * sequence. `compute_swiglu_serial()` uses the same ISA-dispatched FP32
     * vectors as `compute_swiglu()`, and the temporary is stored as FP32 before
     * the unchanged Q8_1 quantizer observes it. A 64-element task is an exact
     * multiple of NEON, AVX2 and AVX-512 vector widths, so task boundaries cannot
     * change scalar-tail selection. K must be Q8_1-block aligned; NativeQuantised
     * expert matrices already require that invariant.
     *
     * @param gate Row-major FP32 gate projection, shape `[M, K]`.
     * @param up Row-major FP32 up projection, shape `[M, K]`.
     * @param output_q8 Row-major Q8_1 destination, shape `[M, K_blocks]`.
     * @param M Number of independently routed expert rows.
     * @param K Logical FP32 elements in each row; must be divisible by 32.
     * @param K_blocks Number of Q8_1 blocks per row; must equal `K / 32`.
     * @param policy Complete ordinary or heterogeneous-expert numerical program.
     * @throws std::invalid_argument Before mutation on invalid geometry/policy or output/source overlap.
     */
    inline void swiglu_quantize_activations_to_q8_1(
        const float *gate,
        const float *up,
        Q8_1Block *output_q8,
        int M,
        int K,
        int K_blocks,
        CPUProjectionNumericalPolicy policy =
            CPUProjectionNumericalPolicy::BackendNative)
    {
        constexpr int kQ8BlockSize =
            static_cast<int>(Q8_1Block::BLOCK_SIZE);
        if (!gate || !up || !output_q8 || M <= 0 || K <= 0 ||
            K % kQ8BlockSize != 0 ||
            K_blocks != K / kQ8BlockSize)
        {
            throw std::invalid_argument(
                "CPU NativeQuantised fused SwiGLU/Q8_1 publication requires "
                "positive block-aligned geometry and non-null buffers");
        }

        if (policy != CPUProjectionNumericalPolicy::BackendNative && policy != CPUProjectionNumericalPolicy::GPUAlignedExpert)
            throw std::invalid_argument("CPU fused SwiGLU/Q8 publication requires a known arithmetic policy");
        const storage_proof::Bytes write{output_q8, std::size_t(M) * K_blocks * sizeof(Q8_1Block)};
        const auto input_bytes = std::size_t(M) * K * sizeof(float);
        storage_proof::requireDisjoint(write, {gate, input_bytes});
        storage_proof::requireDisjoint(write, {up, input_bytes});

        constexpr int kBlocksPerTask = 2;
        constexpr int kElementsPerTask =
            kBlocksPerTask * kQ8BlockSize;
        // Both positive int dimensions fit this widened task product; avoid
        // overflowing K's ceil addition or M's physical-row task count.
        const std::int64_t tasks_per_row = K / kElementsPerTask + (K % kElementsPerTask != 0);
        const std::int64_t task_count = std::int64_t(M) * tasks_per_row;

        auto do_swiglu_quantize = [&]()
        {
#pragma omp for schedule(static)
            for (std::int64_t task = 0; task < task_count; ++task)
            {
                const int row = static_cast<int>(task / tasks_per_row);
                const int row_task = static_cast<int>(task % tasks_per_row);
                const int element_begin = row_task * kElementsPerTask;
                const int element_count =
                    std::min(kElementsPerTask, K - element_begin);
                const size_t fp32_offset =
                    static_cast<size_t>(row) * static_cast<size_t>(K) +
                    static_cast<size_t>(element_begin);
                const int first_block =
                    element_begin / kQ8BlockSize;
                Q8_1Block *destination =
                    output_q8 +
                    static_cast<size_t>(row) *
                        static_cast<size_t>(K_blocks) +
                    static_cast<size_t>(first_block);

                /*
                 * The explicit FP32 temporary is part of the equivalence
                 * contract: the former path stored SwiGLU to FP32 memory before
                 * quantization, so keeping the same rounding point prevents a
                 * compiler from forwarding wider intermediate precision.
                 */
                alignas(64) float activated[kElementsPerTask] = {};
                if (policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                {
                    primitives::compute_swiglu_gpu_aligned_expert_serial(
                        gate + fp32_offset,
                        up + fp32_offset,
                        activated,
                        element_count);
                }
                else
                {
                    primitives::compute_swiglu_serial(
                        gate + fp32_offset,
                        up + fp32_offset,
                        activated,
                        element_count);
                }

#if defined(__AVX512F__)
                if (element_count == kElementsPerTask)
                {
                    quantizeTwoActivationBlocks(
                        activated,
                        destination[0],
                        destination[1],
                        policy);
                    continue;
                }
#endif
                const int block_count =
                    (element_count + kQ8BlockSize - 1) /
                    kQ8BlockSize;
                for (int block = 0; block < block_count; ++block)
                {
                    const int block_begin =
                        block * kQ8BlockSize;
                    const int block_length = std::min(
                        kQ8BlockSize,
                        element_count - block_begin);
                    quantizeActivationBlock(
                        activated + block_begin,
                        destination[block],
                        block_length,
                        policy);
                }
            }
        };

        OMP_WORKSHARE_REGION(do_swiglu_quantize);
    }

} // namespace llaminar2::cpu::native_quantised
