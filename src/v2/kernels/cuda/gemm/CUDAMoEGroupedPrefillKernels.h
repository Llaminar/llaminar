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
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace llaminar2::cuda::moe
{
    /** Number of grouped rows consumed by one CUDA IMMA work item. */
    inline constexpr int kGroupedImmaTileRows = 16;

    /** Maximum descriptor-table width encoded by one directory entry. */
    inline constexpr int kGroupedImmaMaximumExperts = 512;

    /**
     * @brief Compiled output-column ownership for one grouped IMMA CTA.
     *
     * Each warp owns one eight-column `m16n8k32` fragment. Wider CTAs reuse
     * the same 16-row activation tile across more output columns and reduce
     * directory/CTA overhead, at the cost of fewer resident CTAs. These are
     * geometry-only choices: every output element retains the same K-block and
     * K-partition arithmetic order.
     */
    enum class GroupedImmaColumns : uint16_t
    {
        Columns32 = 32,
        Columns64 = 64,
        Columns128 = 128,
        Columns256 = 256,
    };

    /**
     * @brief Warp ownership policy for the fused gate/up projection.
     *
     * `ParallelProjections` assigns disjoint warp banks to gate and up. The
     * `PairedProjections` candidate assigns both projections for one column
     * fragment to the same warp, reuses one A fragment, and retains gate/up in
     * registers until exact SwiGLU publication. Both schedules use identical
     * per-projection reduction trees and 32-column quantization reductions.
     */
    enum class GroupedImmaGateUpSchedule : uint8_t
    {
        ParallelProjections = 0,
        PairedProjections = 1,
    };

    /** Capture-time geometry for the two grouped MoE projection phases. */
    struct GroupedImmaLaunchPolicy
    {
        GroupedImmaColumns gate_up_columns = GroupedImmaColumns::Columns32;
        GroupedImmaColumns down_columns = GroupedImmaColumns::Columns32;
        GroupedImmaGateUpSchedule gate_up_schedule =
            GroupedImmaGateUpSchedule::PairedProjections;
        bool exact_overlay = false;
    };

    /** NativeVNNI execution codebook used by IQ4_XS routed projections. */
    inline constexpr int kQwenIQ4ExecutionCodebook = 4;

    /** NativeVNNI execution codebook used by Q6_K routed projections. */
    inline constexpr int kQwenQ6KExecutionCodebook = 8;

    /** NativeVNNI execution codebook used by IQ3_S routed projections. */
    inline constexpr int kQwenIQ3SExecutionCodebook = 11;

    /** NativeVNNI execution codebook used by IQ2_S routed projections. */
    inline constexpr int kQwenIQ2SExecutionCodebook = 13;

    /**
     * @brief One measured codebook regime for the Qwen 35B-A3B geometry.
     *
     * The threshold is intentionally contiguous rather than a literal list of
     * per-bucket minima. Candidate differences below one percent occasionally
     * changed sign at an isolated bucket, while adjacent buckets and p95 kept
     * the same regime. A contiguous boundary preserves the robust physical
     * mode shift and remains total for M values between or beyond captures.
     */
    struct GroupedImmaCodebookRegime
    {
        int gate_up_codebook = -1;
        int down_codebook = -1;
        int columns32_max_rows = 0;
    };

    /**
     * Capture-time overlays measured across every production bucket M=64..4096.
     *
     * These tuples cover Qwen3.6-35B-A3B UD-IQ3_S's forty main routed layers
     * (`37 x IQ2_S/IQ4_XS`, `2 x IQ2_S/Q6_K`, and `1 x IQ3_S/Q6_K`) plus the
     * already-swept IQ2_S/IQ3_S tuple used by Qwen3.5/3.6 variants. The IQ3_S
     * gate/up regime never produced a stable reason to widen its CTA, so its
     * threshold deliberately extends across every positive `int` M.
     */
    inline constexpr std::array<GroupedImmaCodebookRegime, 4>
        kQwen35BGroupedImmaCodebookRegimes{{
            {
                .gate_up_codebook = kQwenIQ2SExecutionCodebook,
                .down_codebook = kQwenIQ4ExecutionCodebook,
                .columns32_max_rows = 768,
            },
            {
                .gate_up_codebook = kQwenIQ2SExecutionCodebook,
                .down_codebook = kQwenQ6KExecutionCodebook,
                .columns32_max_rows = 600,
            },
            {
                .gate_up_codebook = kQwenIQ3SExecutionCodebook,
                .down_codebook = kQwenQ6KExecutionCodebook,
                .columns32_max_rows = std::numeric_limits<int>::max(),
            },
            {
                .gate_up_codebook = kQwenIQ2SExecutionCodebook,
                .down_codebook = kQwenIQ3SExecutionCodebook,
                .columns32_max_rows = 768,
            },
        }};

    /**
     * @brief Select a total capture-time grouped-IMMA launch policy.
     *
     * Paired projection ownership is the generic schedule because it removes a
     * duplicate A-fragment load and gate/up shared-memory round trip while
     * preserving each projection's public serial-M1 reduction tree. Exact
     * Qwen 35B-A3B overlays come from production-graph tournaments over every
     * captured bucket. Positive unseen M values remain total by extending the
     * adjacent measured regime rather than requiring an exact bucket lookup.
     *
     * @param gateup_codebook Sole gate/up execution codebook, or a negative
     *        value when a mixed descriptor table has no single codebook.
     * @param down_codebook Sole down execution codebook, or a negative value
     *        when a mixed descriptor table has no single codebook.
     * @param seq_len Captured grouped-row bucket.
     * @param hidden_size Model hidden width.
     * @param expert_width Routed expert intermediate width.
     * @param expert_count Routed expert table width.
     * @param top_k Routed experts selected per token.
     * @return Complete compiled policy; exact overlays are identified so the
     *         caller can publish their provenance through PerfStats.
     */
    [[nodiscard]] inline constexpr GroupedImmaLaunchPolicy
    selectGroupedImmaLaunchPolicy(
        int gateup_codebook,
        int down_codebook,
        int seq_len,
        int hidden_size,
        int expert_width,
        int expert_count,
        int top_k) noexcept
    {
        GroupedImmaLaunchPolicy policy{};
        const bool qwen_35b_geometry =
            hidden_size == 2048 && expert_width == 512 &&
            expert_count == 256 && top_k == 8;
        if (!qwen_35b_geometry)
            return policy;

        for (const auto &regime : kQwen35BGroupedImmaCodebookRegimes)
        {
            if (gateup_codebook == regime.gate_up_codebook &&
                down_codebook == regime.down_codebook)
            {
                policy.gate_up_columns =
                    seq_len <= regime.columns32_max_rows
                        ? GroupedImmaColumns::Columns32
                        : GroupedImmaColumns::Columns64;
                policy.exact_overlay = true;
                break;
            }
        }
        return policy;
    }

    /** @return the integer column width represented by a typed geometry. */
    [[nodiscard]] inline constexpr int groupedImmaColumns(
        GroupedImmaColumns geometry) noexcept
    {
        return static_cast<int>(geometry);
    }

    /** @return true when a geometry is compiled for fused gate/up execution. */
    [[nodiscard]] inline constexpr bool validGroupedImmaGateUpColumns(
        GroupedImmaColumns geometry) noexcept
    {
        const int columns = groupedImmaColumns(geometry);
        return columns == 32 || columns == 64 || columns == 128;
    }

    /** @return true when a geometry is compiled for down-projection execution. */
    [[nodiscard]] inline constexpr bool validGroupedImmaDownColumns(
        GroupedImmaColumns geometry) noexcept
    {
        const int columns = groupedImmaColumns(geometry);
        return columns == 32 || columns == 64 || columns == 128 ||
               columns == 256;
    }

    /** @return true when a gate/up warp schedule names a compiled kernel. */
    [[nodiscard]] inline constexpr bool validGroupedImmaGateUpSchedule(
        GroupedImmaGateUpSchedule schedule) noexcept
    {
        return schedule == GroupedImmaGateUpSchedule::ParallelProjections ||
               schedule == GroupedImmaGateUpSchedule::PairedProjections;
    }

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
     * @param columns Capture-time output-column ownership per CTA.
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
        llaminar2::cuda::moe::GroupedImmaColumns columns,
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
     * @param columns Capture-time output-column ownership per CTA.
     * @param schedule Capture-time projection-to-warp ownership policy.
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
        llaminar2::cuda::moe::GroupedImmaColumns columns,
        llaminar2::cuda::moe::GroupedImmaGateUpSchedule schedule,
        int device_idx,
        void *stream);

    /**
     * @brief Inspect one compiled codebook specialization without launching it.
     *
     * This setup-only API supplies the all-format spill and occupancy gate.
     * It performs no allocation, transfer, launch, or synchronization.
     *
     * @param codebook_id NativeVNNI execution codebook specialization.
     * @param columns Exact compiled down-projection geometry to inspect.
     * @param registers_per_thread Receives compiler-assigned registers/thread.
     * @param local_memory_bytes_per_thread Receives compiler local-memory use.
     * @param static_shared_memory_bytes Receives static shared-memory bytes.
     * @param max_threads_per_block Receives the compiled thread limit.
     * @param max_active_blocks_per_sm Receives occupancy at production width.
     * @return true when the specialization exists and every query succeeds.
     */
    bool cudaMoEGroupedImma_queryKernelResources(
        uint8_t codebook_id,
        llaminar2::cuda::moe::GroupedImmaColumns columns,
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
     * @param columns Exact compiled gate/up geometry to inspect.
     * @param schedule Exact compiled projection-to-warp schedule to inspect.
     * @param registers_per_thread Receives compiler-assigned registers/thread.
     * @param local_memory_bytes_per_thread Receives compiler local-memory use.
     * @param static_shared_memory_bytes Receives static shared-memory bytes.
     * @param max_threads_per_block Receives the compiled thread limit.
     * @param max_active_blocks_per_sm Receives occupancy at production width.
     * @return true when the specialization exists and every query succeeds.
     */
    bool cudaMoEGroupedImma_queryGateUpKernelResources(
        uint8_t codebook_id,
        llaminar2::cuda::moe::GroupedImmaColumns columns,
        llaminar2::cuda::moe::GroupedImmaGateUpSchedule schedule,
        int *registers_per_thread,
        std::size_t *local_memory_bytes_per_thread,
        std::size_t *static_shared_memory_bytes,
        int *max_threads_per_block,
        int *max_active_blocks_per_sm);
}
