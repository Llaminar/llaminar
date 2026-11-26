/**
 * @file CUDARetainedHCFFNSuffix.cuh
 * @brief Device-sized split-K Gate/Up and cooperative Down suffix kernels for all 18 codebooks.
 *
 * Included by CUDARetainedHCFFNSuffixShard0.cu and CUDARetainedHCFFNSuffixShard1.cu so
 * each translation unit stays compact, compiles in parallel with zero register spills,
 * and supports arbitrary local routed/shared intermediate widths (320, 640, etc.) and
 * every NativeQuantised codebook including dynamic ExpandedInt8Min / IQ4XSSourceScale promotion.
 */
#pragma once

#include "CUDAMoENativeQuantisedDecodeHelpers.cuh"

namespace llaminar2::cuda::moe::retained_suffix_detail
{
    constexpr int kRetainedRouteCounts = 2;
    constexpr uint8_t kRetainedSuffixDynamicCodebook = 0xFFu;

    /** @brief Exact device-wide clock for optional suffix profiling. */
    __device__ __forceinline__ std::uint64_t suffixRouteClock()
    {
        std::uint64_t tick;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tick));
        return tick;
    }

    /** @brief General all-18-codebook Gate/Up partition helper (inlined ONLY in the 0xFFu dynamic entry -> STACK:0). */
    static __device__ __forceinline__ void retainedSuffixSharedGateUpGeneral(
        const DeviceNativeQuantisedMatrixDesc& gate_desc,
        const DeviceNativeQuantisedMatrixDesc& up_desc,
        const int8_t* hidden_codes,
        const float* hidden_scales,
        int column,
        int first_packet,
        float& gate,
        float& up)
    {
        using namespace llaminar2::cuda::moe;
        constexpr int packets = kQFNDecodeGateUpPackets;
        constexpr auto order = NativeMoEDotOrder::VerifierK32;
        if (gate_desc.codebook_id == up_desc.codebook_id)
        {
            switch (gate_desc.codebook_id)
            {
            case 0:  native_quantised_gate_up_fixed_partition<0,  0, packets, order>(gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up); return;
            case 4:  native_quantised_gate_up_fixed_partition<4,  0, packets, order>(gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up); return;
            case 5:  native_quantised_gate_up_fixed_partition<5,  0, packets, order>(gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up); return;
            case 6:  native_quantised_gate_up_fixed_partition<6,  0, packets, order>(gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up); return;
            case 7:  native_quantised_gate_up_fixed_partition<7,  0, packets, order>(gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up); return;
            case 8:  native_quantised_gate_up_fixed_partition<8,  0, packets, order>(gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up); return;
            case 9:  native_quantised_gate_up_fixed_partition<9,  0, packets, order>(gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up); return;
            case 10: native_quantised_gate_up_fixed_partition<10, 0, packets, order>(gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up); return;
            case 11: native_quantised_gate_up_fixed_partition<11, 0, packets, order>(gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up); return;
            case 12: native_quantised_gate_up_fixed_partition<12, 0, packets, order>(gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up); return;
            case 13: native_quantised_gate_up_fixed_partition<13, 0, packets, order>(gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up); return;
            case 14: native_quantised_gate_up_fixed_partition<14, 0, packets, order>(gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up); return;
            case 15: native_quantised_gate_up_fixed_partition<15, 0, packets, order>(gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up); return;
            case 16: native_quantised_gate_up_fixed_partition<16, 0, packets, order>(gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up); return;
            case 17: native_quantised_gate_up_fixed_partition<17, 0, packets, order>(gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up); return;
            case 19: native_quantised_gate_up_fixed_partition<19, 0, packets, order>(gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up); return;
            case llaminar2::kNativeQuantisedExpandedInt8MinCodebook:
                native_quantised_gate_up_fixed_partition<llaminar2::kNativeQuantisedExpandedInt8MinCodebook, 0, packets, order>(
                    gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up);
                return;
            case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook:
                native_quantised_gate_up_fixed_partition<llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook, 0, packets, order>(
                    gate_desc, up_desc, hidden_codes, hidden_scales, column, first_packet, gate, up);
                return;
            default:
                __trap();
            }
        }
        gate = native_quantised_dot_desc_range_dynamic<order>(
            gate_desc, column, hidden_codes, hidden_scales, gate_desc.n, gate_desc.k,
            first_packet, first_packet + packets);
        up = native_quantised_dot_desc_range_dynamic<order>(
            up_desc, column, hidden_codes, hidden_scales, up_desc.n, up_desc.k,
            first_packet, first_packet + packets);
    }

    /** @brief General all-18-codebook Down partition helper (inlined ONLY in the 0xFFu dynamic entry -> STACK:0). */
    static __device__ __forceinline__ float retainedSuffixSharedDownGeneral(
        const DeviceNativeQuantisedMatrixDesc& down_desc,
        int column,
        const int8_t* down_codes,
        const float* down_scales,
        int hidden,
        int b_start,
        int b_end)
    {
        return native_quantised_dot_desc_range_dynamic<NativeMoEDotOrder::PartitionPartial>(
            down_desc, column, down_codes, down_scales, hidden, down_desc.k, b_start, b_end);
    }

#if CUDART_VERSION >= 12080
    /**
     * @brief Device-sized 8-way split-K Gate/Up + exact ordered SwiGLU Q8 publication.
     *
     * Grid is `(8, local_count * routed_packets + shared_packets)` with `64` threads
     * (2 warps) per CTA. Works for arbitrary 32-aligned `routed_width` and `shared.gate.n`
     * (e.g., 320/320 on TensorSharded or 640/320 on Apportioned) across all 18 codebooks.
     */
    template<uint8_t GateCodebook>
    __global__ __launch_bounds__(
        llaminar2::cuda::moe::retained_suffix_gate_up_threads,
        GateCodebook == kRetainedSuffixDynamicCodebook ? 4 : 8)
    void retainedQFNM1SuffixGateUp(const __grid_constant__ RetainedM1HCFFN args)
    {
        using namespace llaminar2::cuda::moe;
        constexpr bool kDynamic = (GateCodebook == kRetainedSuffixDynamicCodebook);
        constexpr auto SharedCodebook = native_formats::Q8_0.codebook_id;
        constexpr auto Promoted = llaminar2::kNativeQuantisedExpandedInt8MinCodebook;
        constexpr auto IQ4XS = llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook;
        constexpr bool can_promote_expanded =
            !kDynamic &&
            (GateCodebook == native_formats::Q4_1.codebook_id ||
             GateCodebook == native_formats::Q5_1.codebook_id);
        constexpr bool is_iq4_family =
            !kDynamic &&
            (GateCodebook == native_formats::IQ4_NL.codebook_id ||
             GateCodebook == IQ4XS);
        constexpr int partitions = llaminar2::CUDAMoEBatchInvariantPolicy::gate_up_k_partitions;
        constexpr int packets = kQFNDecodeGateUpPackets;
        constexpr auto order = NativeMoEDotOrder::VerifierK32;

        const unsigned warp = threadIdx.x / 32u;
        const unsigned lane = threadIdx.x % 32u;
        const unsigned routed_width = static_cast<unsigned>(args.routed_width);
        const unsigned routed_packets = routed_width / 32u;
        const unsigned shared_packets = static_cast<unsigned>(args.shared.gate.n / 32);
        const unsigned local_count = static_cast<unsigned>(args.route_order[0]);
        const unsigned total_routed_packets = local_count * routed_packets;
        const unsigned total_packets = total_routed_packets + shared_packets;

        const bool resident_loop = args.suffix_resident_loop != 0u;
        const unsigned total_tasks = resident_loop
            ? total_packets * retained_suffix_gate_up_k_splits
            : 1u;
        const unsigned task_begin = resident_loop ? blockIdx.x : 0u;
        const unsigned task_stride = resident_loop ? gridDim.x : 1u;

        if (args.profile_ns && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
            args.profile_ns[7] = suffixRouteClock();

        for (unsigned task = task_begin; task < total_tasks; task += task_stride)
        {
            const unsigned split = resident_loop ? (task & (retained_suffix_gate_up_k_splits - 1u)) : blockIdx.x;
            const unsigned packet_id = resident_loop ? (task / retained_suffix_gate_up_k_splits) : blockIdx.y;
            const unsigned part = split * 2u + warp;

            const bool is_shared = packet_id >= total_routed_packets;
            const unsigned expert_idx = is_shared ? local_count : (packet_id / routed_packets);
            const unsigned pkt = is_shared ? (packet_id - total_routed_packets) : (packet_id % routed_packets);
            const unsigned column = pkt * 32u + lane;

            float gate = 0.0f, up = 0.0f;
            if constexpr (kDynamic)
            {
                const auto& g_desc = is_shared ? args.shared.gate : args.selected[expert_idx].gate;
                const auto& u_desc = is_shared ? args.shared.up : args.selected[expert_idx].up;
                retainedSuffixSharedGateUpGeneral(
                    g_desc, u_desc, args.hidden_codes, args.hidden_scales,
                    static_cast<int>(column), static_cast<int>(part * packets), gate, up);
            }
            else if (is_shared)
            {
                native_quantised_gate_up_fixed_partition<SharedCodebook, 0, packets, order>(
                    args.shared.gate, args.shared.up, args.hidden_codes, args.hidden_scales,
                    static_cast<int>(column), static_cast<int>(part * packets), gate, up);
            }
            else if constexpr (is_iq4_family)
            {
                const auto& g_desc = args.selected[expert_idx].gate;
                const auto& u_desc = args.selected[expert_idx].up;
                if (g_desc.codebook_id == GateCodebook && u_desc.codebook_id == GateCodebook)
                {
                    native_quantised_gate_up_fixed_partition<GateCodebook, 0, packets, order>(
                        g_desc, u_desc, args.hidden_codes, args.hidden_scales,
                        static_cast<int>(column), static_cast<int>(part * packets), gate, up);
                }
                else
                {
                    constexpr uint8_t OtherIQ4 = (GateCodebook == IQ4XS)
                        ? native_formats::IQ4_NL.codebook_id : IQ4XS;
                    native_quantised_gate_up_fixed_partition<OtherIQ4, 0, packets, order>(
                        g_desc, u_desc, args.hidden_codes, args.hidden_scales,
                        static_cast<int>(column), static_cast<int>(part * packets), gate, up);
                }
            }
            else if constexpr (can_promote_expanded)
            {
                const unsigned canonical_count = static_cast<unsigned>(args.route_order[1]);
                if (expert_idx >= canonical_count)
                    native_quantised_gate_up_fixed_partition<Promoted, 0, packets, order>(
                        args.selected[expert_idx].gate, args.selected[expert_idx].up,
                        args.hidden_codes, args.hidden_scales,
                        static_cast<int>(column), static_cast<int>(part * packets), gate, up);
                else
                    native_quantised_gate_up_fixed_partition<GateCodebook, 0, packets, order>(
                        args.selected[expert_idx].gate, args.selected[expert_idx].up,
                        args.hidden_codes, args.hidden_scales,
                        static_cast<int>(column), static_cast<int>(part * packets), gate, up);
            }
            else
            {
                native_quantised_gate_up_fixed_partition<GateCodebook, 0, packets, order>(
                    args.selected[expert_idx].gate, args.selected[expert_idx].up,
                    args.hidden_codes, args.hidden_scales,
                    static_cast<int>(column), static_cast<int>(part * packets), gate, up);
            }

            const std::size_t base = (static_cast<std::size_t>(packet_id) * partitions + part) * 64u;
            args.suffix_partials[base + lane] = gate;
            args.suffix_partials[base + 32u + lane] = up;
            __syncthreads();

            __shared__ unsigned completed;
            if (threadIdx.x == 0)
            {
                __threadfence();
                completed = atomicAdd(args.suffix_done + packet_id, 1u);
            }
            __syncthreads();

            if (completed == retained_suffix_gate_up_k_splits - 1u && warp == 0)
            {
                float sum_gate = 0.0f, sum_up = 0.0f;
                const float* packet_partials = args.suffix_partials + static_cast<std::size_t>(packet_id) * partitions * 64u;
#pragma unroll
                for (int k = 0; k < partitions; ++k)
                {
                    sum_gate = __fadd_rn(sum_gate, __ldcg(packet_partials + static_cast<std::size_t>(k) * 64u + lane));
                    sum_up = __fadd_rn(sum_up, __ldcg(packet_partials + static_cast<std::size_t>(k) * 64u + 32u + lane));
                }
                int8_t* dst_codes = is_shared ? args.shared_workspace.down_codes
                    : args.routed_workspace.down_codes + static_cast<std::size_t>(expert_idx) * routed_width;
                float* dst_scales = is_shared ? args.shared_workspace.down_scales
                    : args.routed_workspace.down_scales + static_cast<std::size_t>(expert_idx) * routed_packets;
                publish_decode_swiglu_q8(sum_gate, sum_up, dst_codes + pkt * 32u, dst_scales + pkt, static_cast<int>(lane));
                if (args.profile_ns && lane == 0)
                    atomicMax(reinterpret_cast<unsigned long long*>(args.profile_ns + 8),
                        static_cast<unsigned long long>(suffixRouteClock()));
            }
            if (resident_loop)
                __syncthreads();
        }
    }

    /**
     * @brief Device-sized 5-warp cooperative Down + remote-slot zeroing for arbitrary widths.
     *
     * Grid is `(hidden / 32, local_count + 1)` with `160` threads (5 warps) per CTA.
     * Partitioning uses the exact canonical `down_k_partitions = 16` schedule:
     * - For `width = 320` (`blocks = 10`): `blocks_per_part = 1`, 10 active partitions (2 per warp).
     * - For `width = 640` (`blocks = 20`): `blocks_per_part = 2`, 10 active partitions (2 per warp).
     * - For any larger `width`: strides over `active_parts <= 16` into `block_vals[16][32]`.
     * Warp 0 folds `0..active_parts-1` in exact serial order and matches empty tail `+0.0f`
     * when required by the partition-partial contract.
     */
    template<uint8_t DownCodebook>
    __global__ __launch_bounds__(
        llaminar2::cuda::moe::retained_suffix_down_threads,
        DownCodebook == kRetainedSuffixDynamicCodebook ? 2 : 4)
    void retainedQFNM1SuffixDown(const __grid_constant__ RetainedM1HCFFN args)
    {
        using namespace llaminar2::cuda::moe;
        constexpr bool kDynamic = (DownCodebook == kRetainedSuffixDynamicCodebook);
        constexpr auto SharedCodebook = native_formats::Q8_0.codebook_id;
        constexpr auto Promoted = llaminar2::kNativeQuantisedExpandedInt8MinCodebook;
        constexpr auto IQ4XS = llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook;
        constexpr bool can_promote_expanded =
            !kDynamic &&
            (DownCodebook == native_formats::Q4_1.codebook_id ||
             DownCodebook == native_formats::Q5_1.codebook_id);
        constexpr bool is_iq4_family =
            !kDynamic &&
            (DownCodebook == native_formats::IQ4_NL.codebook_id ||
             DownCodebook == IQ4XS);
        constexpr int hidden = kQFNDecodeGateUpK;
        constexpr int kPartitions = llaminar2::CUDAMoEBatchInvariantPolicy::down_k_partitions;

        const unsigned warp = threadIdx.x / 32u;
        const unsigned lane = threadIdx.x % 32u;
        if (args.profile_ns && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
            args.profile_ns[9] = suffixRouteClock();
        const int column = static_cast<int>(blockIdx.x * 32u + lane);
        const int local_count = args.route_order[0];
        const bool resident_loop = args.suffix_resident_loop != 0u;
        const int expert_begin = resident_loop ? 0 : static_cast<int>(blockIdx.y);
        const int expert_end = resident_loop ? (local_count + 1) : (expert_begin + 1);

        __shared__ float block_vals[kPartitions][32];
        for (int expert_idx = expert_begin; expert_idx < expert_end; ++expert_idx)
        {
            const bool is_shared = expert_idx == local_count;
            const int width = is_shared ? args.shared.down.k : args.routed_width;
            const int blocks = width / 32;
            const int blocks_per_part = (blocks + kPartitions - 1) / kPartitions;
            const int active_parts = (blocks + blocks_per_part - 1) / blocks_per_part;

            if constexpr (kDynamic)
            {
                const auto& down_desc = is_shared ? args.shared.down : args.selected[expert_idx].down;
                const auto* codes = is_shared
                    ? args.shared_workspace.down_codes
                    : args.routed_workspace.down_codes + static_cast<std::size_t>(expert_idx) * width;
                const auto* scales = is_shared
                    ? args.shared_workspace.down_scales
                    : args.routed_workspace.down_scales + static_cast<std::size_t>(expert_idx) * blocks;
                for (int p = static_cast<int>(warp); p < active_parts; p += 5)
                {
                    const int b_start = p * blocks_per_part;
                    const int b_end = min(blocks, b_start + blocks_per_part);
                    block_vals[p][lane] = retainedSuffixSharedDownGeneral(
                        down_desc, column, codes, scales, hidden, b_start, b_end);
                }
            }
            else if (is_shared)
            {
                for (int p = static_cast<int>(warp); p < active_parts; p += 5)
                {
                    const int b_start = p * blocks_per_part;
                    const int b_end = min(blocks, b_start + blocks_per_part);
                    block_vals[p][lane] = native_quantised_dot_admitted_range<SharedCodebook,
                        NativeMoEDotOrder::PartitionPartial, 0, 1>(args.shared.down, column,
                        args.shared_workspace.down_codes, args.shared_workspace.down_scales, hidden, b_start, b_end);
                }
            }
            else
            {
                const auto* codes = args.routed_workspace.down_codes + static_cast<std::size_t>(expert_idx) * width;
                const auto* scales = args.routed_workspace.down_scales + static_cast<std::size_t>(expert_idx) * blocks;
                const auto& down_desc = args.selected[expert_idx].down;
                if constexpr (is_iq4_family)
                {
                    const bool matches_primary = down_desc.codebook_id == DownCodebook;
                    constexpr uint8_t OtherIQ4 = (DownCodebook == IQ4XS)
                        ? native_formats::IQ4_NL.codebook_id : IQ4XS;
                    for (int p = static_cast<int>(warp); p < active_parts; p += 5)
                    {
                        const int b_start = p * blocks_per_part;
                        const int b_end = min(blocks, b_start + blocks_per_part);
                        block_vals[p][lane] = matches_primary
                            ? native_quantised_dot_admitted_range<DownCodebook, NativeMoEDotOrder::PartitionPartial, 0, 1>(
                                  down_desc, column, codes, scales, hidden, b_start, b_end)
                            : native_quantised_dot_admitted_range<OtherIQ4, NativeMoEDotOrder::PartitionPartial, 0, 1>(
                                  down_desc, column, codes, scales, hidden, b_start, b_end);
                    }
                }
                else if constexpr (can_promote_expanded)
                {
                    const int canonical_count = args.route_order[1];
                    if (expert_idx >= canonical_count)
                    {
                        for (int p = static_cast<int>(warp); p < active_parts; p += 5)
                        {
                            const int b_start = p * blocks_per_part;
                            const int b_end = min(blocks, b_start + blocks_per_part);
                            block_vals[p][lane] = native_quantised_dot_admitted_range<Promoted,
                                NativeMoEDotOrder::DecodeK32, 0, 1>(down_desc, column,
                                codes, scales, hidden, b_start, b_end);
                        }
                    }
                    else
                    {
                        for (int p = static_cast<int>(warp); p < active_parts; p += 5)
                        {
                            const int b_start = p * blocks_per_part;
                            const int b_end = min(blocks, b_start + blocks_per_part);
                            block_vals[p][lane] = native_quantised_dot_admitted_range<DownCodebook,
                                NativeMoEDotOrder::DecodeK32, 0, 1>(down_desc, column,
                                codes, scales, hidden, b_start, b_end);
                        }
                    }
                }
                else
                {
                    for (int p = static_cast<int>(warp); p < active_parts; p += 5)
                    {
                        const int b_start = p * blocks_per_part;
                        const int b_end = min(blocks, b_start + blocks_per_part);
                        block_vals[p][lane] = native_quantised_dot_admitted_range<DownCodebook,
                            NativeMoEDotOrder::PartitionPartial, 0, 1>(down_desc, column,
                            codes, scales, hidden, b_start, b_end);
                    }
                }
            }
            __syncthreads();

            if (is_shared)
            {
                const int remote_count = args.router.top_k - local_count;
                for (int r = static_cast<int>(warp); r < remote_count; r += 5)
                {
                    const int slot = args.route_order[kRetainedRouteCounts + local_count + r];
                    args.routed_contributions[slot * hidden + column] = 0.0f;
                }
                if (warp == 0)
                {
                    float raw = 0.0f;
                    for (int p = 0; p < active_parts; ++p)
                        raw = moe_accumulate_rn(raw, block_vals[p][lane]);
                    // Match empty tail partitions in k_partitions=16 for exact signed-zero preservation.
                    if (active_parts < kPartitions)
                        raw = moe_accumulate_rn(raw, 0.0f);
                    const bool gate_shared = args.shared_publication == RetainedM1SharedPublication::GatedComplete;
                    const float shared_factor = gate_shared ? *args.shared_gate_factor : 1.0f;
                    args.shared_contribution[column] = gate_shared
                        ? llaminar2::device_fp32_contract::multiply(shared_factor, raw) : raw;
                }
            }
            else if (warp == 0)
            {
                float contribution = 0.0f;
                for (int p = 0; p < active_parts; ++p)
                    contribution = moe_accumulate_rn(contribution, block_vals[p][lane]);
                const bool direct_k32 =
                    args.router.local_policy == llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal &&
                    blocks == 10;
                if (!direct_k32 && active_parts < kPartitions)
                    contribution = moe_accumulate_rn(contribution, 0.0f);
                const int slot = args.route_order[kRetainedRouteCounts + expert_idx];
                const bool weight_routes = args.route_publication == RetainedM1RoutePublication::Preweighted;
                const auto* runtime = reinterpret_cast<const DeviceMoELayerRuntimeView*>(args.router.runtime);
                args.routed_contributions[slot * hidden + column] = weight_routes
                    ? moe_weight_route_rn(runtime->topk_weights[slot], contribution) : contribution;
            }
            if (resident_loop && expert_idx + 1 < expert_end)
                __syncthreads();
        }
        if (args.profile_ns && warp == 0 && lane == 0)
            atomicMax(reinterpret_cast<unsigned long long*>(args.profile_ns + 10),
                static_cast<unsigned long long>(suffixRouteClock()));
    }
#endif
} // namespace llaminar2::cuda::moe::retained_suffix_detail
