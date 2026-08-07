/**
 * @file CUDAMoEGroupedPrefillKernels.h
 * @brief Typed launch ABI for device-planned CUDA grouped MoE IMMA prefill.
 *
 * Long grouped prefill is represented by a compact directory authored from
 * device-resident expert counts. Projection launches consume that directory
 * without host inspection, dynamic allocation, synchronization, or replay-time
 * topology changes. Every launch uses the public serial-M1 K-partition tree so
 * tensor-core integer dot products remain byte-equivalent to decode.
 */

#pragma once

#include "../../common/DeviceNativeVNNIMatrixDesc.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace llaminar2::cuda::moe
{
    /** Number of grouped rows consumed by one CUDA IMMA work item. */
    inline constexpr int kGroupedImmaTileRows = 16;

    /** Maximum descriptor-table width encoded by one directory entry. */
    inline constexpr int kGroupedImmaMaximumExperts = 512;

    /**
     * @brief Return the conservative graph-captured directory capacity.
     *
     * Every active expert contributes at most one partially occupied row tile.
     * All remaining tiles are bounded by `ceil(total_slots / tile_rows)`. The
     * returned capacity therefore covers every possible route distribution
     * without reserving a dense expert-by-row rectangle.
     *
     * @param total_slots Number of grouped route rows in the captured bucket.
     * @param num_experts Descriptor-table width.
     * @return Positive number of packed directory words required.
     */
    [[nodiscard]] inline constexpr std::size_t groupedImmaDirectoryEntries(
        std::size_t total_slots,
        std::size_t num_experts) noexcept
    {
        total_slots = std::max<std::size_t>(1, total_slots);
        num_experts = std::max<std::size_t>(1, num_experts);
        const std::size_t active_expert_bound =
            std::min(total_slots, num_experts);
        return std::min(
            total_slots,
            active_expert_bound +
                (total_slots + kGroupedImmaTileRows - 1) /
                    kGroupedImmaTileRows);
    }
} // namespace llaminar2::cuda::moe

extern "C"
{
    /**
     * @brief Build the compact grouped-row directory on the producer stream.
     *
     * @param d_group_counts Device-resident row count for every expert.
     * @param d_directory Persistent graph-owned packed directory storage.
     * @param num_experts Number of descriptor/count entries.
     * @param total_slots Exact grouped route-row capacity.
     * @param directory_entries Captured conservative directory capacity.
     * @param device_idx CUDA ordinal owning every pointer.
     * @param stream Exact non-null producer stream.
     * @return true when the planner launch was accepted by CUDA.
     */
    bool cudaMoEGroupedImma_buildDirectory(
        const int *d_group_counts,
        uint32_t *d_directory,
        int num_experts,
        int total_slots,
        int directory_entries,
        int device_idx,
        void *stream);

    /**
     * @brief Launch all codebook-specialized IMMA projections in one table.
     *
     * The codebook mask may contain several formats. Each specialization walks
     * the same compact directory and accepts only descriptors carrying its own
     * codebook id, so mixed expert tables remain one device-owned operation.
     * Output rows retain grouped-slot order.
     *
     * @param d_A_int8 Grouped blockwise-INT8 activation rows `[slots,K]`.
     * @param d_scales_A Per-row/per-32-value activation scales.
     * @param d_desc_table Device descriptor table indexed by expert id.
     * @param d_group_counts Device-resident row count per expert.
     * @param d_group_offsets Device-resident first grouped slot per expert.
     * @param d_directory Packed `(expert, local_first_row)` work directory.
     * @param d_partition_weights Optional route weight for every grouped row.
     *        When non-null, each public-M1 K-partition is multiplied before the
     *        ascending partition fold. This is the serial down-projection tree;
     *        gate/up projections pass null.
     * @param d_output FP32 grouped projection output `[slots,N]`.
     * @param directory_entries Captured conservative directory capacity.
     * @param num_experts Descriptor-table width.
     * @param total_slots Exact grouped route-row capacity.
     * @param N Projection output width.
     * @param K Projection reduction width.
     * @param codebook_mask OR-mask of every descriptor codebook in the table.
     * @param k_partitions Public serial-M1 partition count.
     * @param device_idx CUDA ordinal owning every pointer.
     * @param stream Exact non-null producer stream.
     * @return true when every required specialization launch was accepted.
     */
    bool cudaMoEGroupedImma_project(
        const int8_t *d_A_int8,
        const float *d_scales_A,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_desc_table,
        const int *d_group_counts,
        const int *d_group_offsets,
        const uint32_t *d_directory,
        const float *d_partition_weights,
        float *d_output,
        int directory_entries,
        int num_experts,
        int total_slots,
        int N,
        int K,
        uint32_t codebook_mask,
        int k_partitions,
        int device_idx,
        void *stream);

    /**
     * @brief Fuse gate/up IMMA projection with exact SwiGLU requantization.
     *
     * Gate and up descriptor upload guarantees one execution codebook per
     * expert pair. A single CTA assigns four warps to each projection, shares
     * the activation tile, retains each projection's canonical K-partition
     * arithmetic, and emits the blockwise-INT8 down-projection input directly.
     * The operation therefore removes FP32 gate/up intermediates from the
     * production grouped-IMMA path without changing a serial-row result byte.
     *
     * @param d_A_int8 Grouped blockwise-INT8 activation rows `[slots,K]`.
     * @param d_scales_A Per-row/per-32-value activation scales.
     * @param d_gate_desc_table Device gate descriptor table by expert id.
     * @param d_up_desc_table Device up descriptor table by expert id.
     * @param d_group_counts Device-resident row count per expert.
     * @param d_group_offsets Device-resident first grouped slot per expert.
     * @param d_directory Packed `(expert, local_first_row)` work directory.
     * @param d_swiglu_int8 Blockwise-INT8 down-projection input `[slots,N]`.
     * @param d_swiglu_scales Per-row/per-32-value SwiGLU scales.
     * @param directory_entries Captured conservative directory capacity.
     * @param num_experts Descriptor-table width.
     * @param total_slots Exact grouped route-row capacity.
     * @param N Gate/up output and SwiGLU width; must be divisible by 32.
     * @param K Gate/up reduction width; must be divisible by 32.
     * @param codebook_mask OR-mask of every paired descriptor codebook.
     * @param k_partitions Public serial-M1 gate/up partition count.
     * @param device_idx CUDA ordinal owning every pointer.
     * @param stream Exact non-null producer stream.
     * @return true when every required specialization launch was accepted.
     */
    bool cudaMoEGroupedImma_projectGateUpSwiGlu(
        const int8_t *d_A_int8,
        const float *d_scales_A,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_gate_desc_table,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_up_desc_table,
        const int *d_group_counts,
        const int *d_group_offsets,
        const uint32_t *d_directory,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        int directory_entries,
        int num_experts,
        int total_slots,
        int N,
        int K,
        uint32_t codebook_mask,
        int k_partitions,
        int device_idx,
        void *stream);

    /**
     * @brief Inspect one compiled codebook specialization without launching it.
     *
     * This setup-only API supplies the all-format spill and occupancy gate.
     * It performs no allocation, transfer, launch, or synchronization.
     *
     * @param codebook_id NativeVNNI execution codebook specialization.
     * @param registers_per_thread Receives compiler-assigned registers/thread.
     * @param local_memory_bytes_per_thread Receives compiler local-memory use.
     * @param static_shared_memory_bytes Receives static shared-memory bytes.
     * @param max_threads_per_block Receives the compiled thread limit.
     * @param max_active_blocks_per_sm Receives occupancy at production width.
     * @return true when the specialization exists and every query succeeds.
     */
    bool cudaMoEGroupedImma_queryKernelResources(
        uint8_t codebook_id,
        int *registers_per_thread,
        std::size_t *local_memory_bytes_per_thread,
        std::size_t *static_shared_memory_bytes,
        int *max_threads_per_block,
        int *max_active_blocks_per_sm);

    /**
     * @brief Inspect one fused gate/up/SwiGLU specialization without launching.
     *
     * The setup-only resource query lets the all-format performance suite reject
     * spills or an uneconomical occupancy change before model-level timing.
     *
     * @param codebook_id NativeVNNI execution codebook specialization.
     * @param registers_per_thread Receives compiler-assigned registers/thread.
     * @param local_memory_bytes_per_thread Receives compiler local-memory use.
     * @param static_shared_memory_bytes Receives static shared-memory bytes.
     * @param max_threads_per_block Receives the compiled thread limit.
     * @param max_active_blocks_per_sm Receives occupancy at production width.
     * @return true when the specialization exists and every query succeeds.
     */
    bool cudaMoEGroupedImma_queryGateUpKernelResources(
        uint8_t codebook_id,
        int *registers_per_thread,
        std::size_t *local_memory_bytes_per_thread,
        std::size_t *static_shared_memory_bytes,
        int *max_threads_per_block,
        int *max_active_blocks_per_sm);
}
