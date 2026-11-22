/**
 * @file CUDARetainedHCPrefix.cuh
 * @brief Resident CUDA arithmetic from the rounded attention write to the mixed FFN input.
 *
 * The native grid joins the original publication edges. Lanes retain their
 * integer accumulation and the original serial FP32 folds. The entry is called
 * only for an admitted live row and uses the existing quantization banks.
 */
#pragma once

#include "CUDARetainedHCPrefix.h"
#include "kernels/common/DeviceNativeQuantisedContributionContract.h"
#include "kernels/common/GPUHCDownPublication.cuh"
#include "kernels/qwen4exp/GPUBranchRMSNormBody.cuh"

#include <cooperative_groups.h>
#include <cuda_fp16.h>

namespace llaminar2::cuda::moe
{
    /** @brief Reuse the original CTA-local norm reduction storage in later FFN phases. */
    struct RetainedHCPrefixShared
    {
        float norm[256]; ///< Original fixed 256-lane square reduction.
    };

    /**
     * @brief Compute a CUDA Q8_0 packet contribution using its original FP32 scale order.
     * @param weights Base of one aligned prepared 32-byte packet.
     * @param codes Base of one aligned prepared activation packet.
     * @param weight_scale Widened prepared FP16 scale.
     * @param activation_scale Original FP32 scale.
     * @return The original rounded contribution of this K32 packet.
     */
    __device__ __forceinline__ float retainedHCQ8Contribution(
        const uint8_t* weights, const int8_t* codes, float weight_scale, float activation_scale)
    {
        const auto* w = reinterpret_cast<const int4*>(weights);
        const auto* a = reinterpret_cast<const int4*>(codes);
        const int4 wl = w[0], wh = w[1];
        const int4 al = a[0], ah = a[1];
        int lo = __dp4a(wl.x, al.x, 0);
        int hi = __dp4a(wh.x, ah.x, 0);
        lo = __dp4a(wl.y, al.y, lo);
        hi = __dp4a(wh.y, ah.y, hi);
        lo = __dp4a(wl.z, al.z, lo);
        hi = __dp4a(wh.z, ah.z, hi);
        lo = __dp4a(wl.w, al.w, lo);
        hi = __dp4a(wh.w, ah.w, hi);
        return device_native_quantised_contract::singleScaleBlock(lo + hi, weight_scale, activation_scale);
    }

    /**
     * @brief Fold one original serial Down partition for one output, without public scratch.
     * @param prefix Sealed original Down and normalized input operands.
     * @param column Original Down output column.
     * @param partition Original serial partition number.
     * @return The original rounded partial, including positive zero for an empty partition.
     */
    __device__ __forceinline__ float retainedHCDownPartial(const RetainedHCPrefix& prefix,
        int column, int partition)
    {
        const auto& down = prefix.down;
        const int first = partition * prefix.down_partitions.blocks_per_partition;
        const int last = min(static_cast<int>(down.blocks_per_row),
            first + prefix.down_partitions.blocks_per_partition);
        const auto* weight_scales = static_cast<const uint16_t*>(down.scales);
        float sum = 0.0f;
        for (int group = first; group < last; ++group)
        {
            const int linear = group * down.n + column;
            const float weight_scale = __half2float(__ushort_as_half(weight_scales[linear]));
            const float value = retainedHCQ8Contribution(down.payload + linear * 32,
                prefix.norm_codes + group * 32, weight_scale, prefix.norm_scales[group]);
            sum = __fadd_rn(sum, value);
        }
        return sum;
    }

    /**
     * @brief Fold both original Up terms in increasing K32 order for one branch/channel.
     * @param up Original prepared matrix and two Q8 quantizations produced by Down.
     * @param column Original branch-major output column.
     * @return One rounded Up value after the original add of the complete terms.
     */
    __device__ __forceinline__ float retainedHCUpColumn(const GPUQ8ResidualCorrectionArgs& up, int column)
    {
        const auto& weight = up.weights;
        const auto* weight_scales = static_cast<const uint16_t*>(weight.scales);
        float primary = 0.0f;
        float residual = 0.0f;
        for (unsigned group = 0; group < weight.blocks_per_row; ++group)
        {
            const unsigned linear = group * weight.n + column;
            const uint8_t* packet = weight.payload + linear * 32;
            const float scale = __half2float(__ushort_as_half(weight_scales[linear]));
            primary = __fadd_rn(primary, retainedHCQ8Contribution(packet, up.primary + group * 32,
                scale, up.primaryScales[group]));
            residual = __fadd_rn(residual, retainedHCQ8Contribution(packet, up.residual + group * 32,
                scale, up.residualScales[group]));
        }
        return __fadd_rn(primary, residual);
    }

    /**
     * @brief Publish the original post-attention mixed row inside an enclosing resident program.
     * @tparam PublishActivated The graph reads the original rounded Down activation.
     * @tparam PublishUp The graph reads the original branch logits.
     * @param prefix Cold-prepared original HC operands and fold.
     * @param shared Borrowed CTA storage, reusable after the final grid join.
     * @param grid Resident native grid, with one row active for every participant.
     * @pre All CTAs enter using 256 threads and at least four native blocks.
     *      Both weight sources are original prepared Q8_0. The Down output has
     *      complete K32 quantizer groups, and the Up output width is 4H.
     */
    template<bool PublishActivated, bool PublishUp>
    __device__ __forceinline__ void publishRetainedHCPrefix(const RetainedHCPrefix& prefix,
        RetainedHCPrefixShared& shared, cooperative_groups::grid_group grid)
    {
        using namespace qwen4exp::branch_norm_native;
        const unsigned hidden = prefix.norm.hidden;
        const unsigned tid = threadIdx.x;
        const unsigned lane = tid % 32;
        const unsigned global_thread = blockIdx.x * blockDim.x + tid;
        const unsigned grid_threads = gridDim.x * blockDim.x;

        if (blockIdx.x < 4)
        {
            const unsigned branch = blockIdx.x;
            const unsigned offset = branch * hidden;
            const ResidualBranchInput input{prefix.write.residual + offset, prefix.write.block,
                prefix.write.output + offset, residualGate(prefix.write.gateLogits[branch])};
            taskFromInput(input, prefix.norm.gain,
                CUDAQuantizedPublication{prefix.norm.output, prefix.norm_codes, prefix.norm_scales},
                prefix.norm.hidden, prefix.norm.epsilon, branch, shared.norm, tid, true);
        }
        grid.sync();

        const unsigned partitions = prefix.down_partitions.count;
        const unsigned columns = prefix.down.n;
        const bool ordered = prefix.down_fold == HCDownFold::OrderedPartitions;
        if (ordered)
        {
            for (unsigned task = global_thread; task < partitions * columns; task += grid_threads)
            {
                const unsigned column = task % columns;
                const unsigned partition = task / columns;
                prefix.down_partials[task] = retainedHCDownPartial(prefix, column, partition);
            }
            grid.sync();
        }

        // Every publishing warp owns a complete quantizer packet. It folds
        // the original partition bank in order, or computes the single admitted
        // direct partition without materializing a partial or joining the grid.
        for (unsigned first = (global_thread / 32) * 32; first < columns; first += grid_threads)
        {
            const unsigned column = first + lane;
            float sum = 0.0f;
            if (ordered)
            {
                for (unsigned part = 0; part < partitions; ++part)
                    sum = __fadd_rn(sum, prefix.down_partials[part * columns + column]);
            }
            else
                sum = retainedHCDownPartial(prefix, column, 0);
            prefix.raw_down[column] = sum;
            hc_down_publication::publish<hc_down_publication::OriginalScalePublication, PublishActivated>(
                sum, prefix.down_publication, column);
        }
        grid.sync();

        // A warp owns eight channels from all four branches. Lanes fold
        // their two Up terms independently, then exchange only final logits.
        // The first eight lanes retain the original branch order in the read.
        constexpr unsigned branch_width = retained_hc_read_warp_channels;
        const unsigned branch = lane / branch_width;
        // Assign one packet to each CTA before assigning its second warp. The
        // admitted grid is bounded by these always-live HC Up packets.
        const unsigned read_warp = tid / 32 * gridDim.x + blockIdx.x;
        for (unsigned first = read_warp * branch_width; first < hidden;
             first += grid_threads / 32 * branch_width)
        {
            const unsigned channel = first + lane % branch_width;
            const unsigned column = branch * hidden + channel;
            const float logit = retainedHCUpColumn(prefix.up, column);
            if constexpr (PublishUp)
                prefix.up.output[column] = logit;
            float mixed = 0.0f;
#pragma unroll
            for (unsigned current = 0; current < 4; ++current)
            {
                const float term = __shfl_sync(0xffffffffu, logit, current * branch_width + lane % branch_width);
                if (branch == 0)
                    mixed = appendReadBranch(mixed, term, prefix.norm.output[current * hidden + channel]);
            }
            if (branch == 0)
                prefix.mixed[channel] = mixed * 0.25f;
        }
        grid.sync();
    }
}
